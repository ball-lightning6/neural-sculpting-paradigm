"""小型真实训练诊断：随机记忆样本的BCE异质性；不运行SMC。"""

from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from config import WORK_DIR, EVIDENCE, CUDA_VISIBLE_DEVICES

ROOT=WORK_DIR / "loss"
CONFIG={'input_bits':8,'data_seed':202609112301,'datasets':4,'counts':[32,80,128],
        'architectures':[[32],[32,32],[32,32,32]],'models':16,'nodes':[.01,.001,.0001],
        'optimizer':'Adam','lr':.003,'weight_decay':0.,'max_steps':20000,
        'seconds_per_case':90.,'total_seconds':600.,'dtype':'float32','tf32':False}


def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def array_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def make_data(dataset):
    rng=np.random.default_rng(CONFIG['data_seed']+1_000_003*dataset)
    order=rng.permutation(256).astype(np.int64)
    labels=rng.integers(0,2,256,dtype=np.uint8)
    x=((np.arange(256,dtype=np.uint32)[:,None]>>np.arange(7,-1,-1))&1).astype(np.float32)
    return x,labels,order


def layout(hidden):
    dims=(8,*hidden,1);rows=[];start=0
    for fan_in,fan_out in zip(dims[:-1],dims[1:]):
        end=start+fan_in*fan_out;stop=end+fan_out
        rows.append((start,end,stop,fan_in,fan_out));start=stop
    return rows,start


def forward(theta,x,layers):
    value=x[None].expand(len(theta),-1,-1)
    for i,(start,end,stop,fan_in,fan_out) in enumerate(layers):
        weight=theta[:,start:end].reshape(len(theta),fan_out,fan_in)
        value=torch.bmm(value,weight.transpose(1,2))+theta[:,None,end:stop]
        if i<len(layers)-1:value=torch.tanh(value)
    return value[:,:,0]


def numpy_forward(theta,x,layers):
    theta=theta.astype(np.float64);x=x.astype(np.float64)
    value=np.broadcast_to(x,(len(theta),*x.shape))
    for i,(start,end,stop,fan_in,fan_out) in enumerate(layers):
        weight=theta[:,start:end].reshape(len(theta),fan_out,fan_in)
        value=value@weight.transpose(0,2,1)+theta[:,None,end:stop]
        if i<len(layers)-1:value=np.tanh(value)
    return value[:,:,0]


def prepare_data():
    prior={'datasets':json.loads((EVIDENCE/'datasets.json').read_text(encoding='utf-8'))}
    hashes={}
    for dataset in range(CONFIG['datasets']):
        x,y,order=make_data(dataset)
        actual=array_hash(x,y,order)
        if actual!=prior['datasets'][str(dataset)]['raw_dataset_sha256']:raise RuntimeError('未复现原随机表')
        hashes[str(dataset)]={'full':actual,'prefix':{str(n):array_hash(x[order[:n]],y[order[:n]],order[:n]) for n in CONFIG['counts']}}
    return hashes


def train_case(dataset,n,hidden,kind,deadline):
    key=f"{kind}_d{dataset}_n{n}_h{'_'.join(map(str,hidden))}"
    folder=ROOT/'results'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():return json.loads((folder/'RESULT.json').read_text(encoding='utf-8'))
    full_x,original_y,order=make_data(dataset)
    full_y=original_y if kind=='random' else full_x[:,0].astype(np.uint8)
    selected=order[:n];x0=full_x[selected];y0=full_y[selected]
    x=torch.from_numpy(x0).to('cuda:0');y=torch.tensor(y0,dtype=torch.float32,device='cuda:0')
    layers,size=layout(hidden);models=CONFIG['models'];nodes=CONFIG['nodes']
    # 同数据表和架构在不同n/正对照间使用相同初始化，避免改变初态。
    seed=int.from_bytes(hashlib.sha256(f"loss_heterogeneity_v1:{dataset}:{hidden}".encode()).digest()[:8],'big')%(2**62)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    initial=torch.randn(models,size,generator=generator)
    for start,end,stop,fan_in,fan_out in layers:initial[:,start:stop]/=math.sqrt(fan_in)
    theta=torch.nn.Parameter(initial.to('cuda:0'))
    optimizer=torch.optim.Adam([theta],lr=CONFIG['lr'],weight_decay=0.)
    initial_logits=numpy_forward(initial.numpy(),x0,layers)
    initial_losses=np.logaddexp(0.,initial_logits*(1-2*y0.astype(float)))
    weights=np.full((len(nodes),models,size),np.nan,dtype=np.float32)
    losses=np.full((len(nodes),models,n),np.nan,dtype=np.float64)
    hit_steps=np.full((len(nodes),models),-1,dtype=np.int64)
    hit_mean_float32=np.full((len(nodes),models),np.nan,dtype=np.float64)
    max_independent_error=0.;begin=time.monotonic();step=0;status='running'
    while True:
        logits=forward(theta,x,layers)
        per_sample=torch.nn.functional.softplus(logits*(1-2*y[None]))
        mean=per_sample.mean(1)
        values=mean.detach().cpu().numpy()
        if not np.isfinite(values).all():raise RuntimeError('训练产生非有限loss')
        for i,epsilon in enumerate(nodes):
            hit=np.flatnonzero((hit_steps[i]<0)&(values<=epsilon))
            if len(hit):
                raw=theta[torch.from_numpy(hit).to('cuda:0')].detach().cpu().numpy().copy()
                recomputed=np.logaddexp(0.,numpy_forward(raw,x0,layers)*(1-2*y0.astype(float)))
                cached=per_sample[torch.from_numpy(hit).to('cuda:0')].detach().cpu().numpy()
                max_independent_error=max(max_independent_error,float(np.max(abs(recomputed-cached))))
                weights[i,hit]=raw;losses[i,hit]=recomputed;hit_steps[i,hit]=step;hit_mean_float32[i,hit]=values[hit]
        if np.all(hit_steps[-1]>=0):status='completed';break
        if step>=CONFIG['max_steps'] or time.monotonic()-begin>=CONFIG['seconds_per_case']:status='budget_censored';break
        if time.monotonic()>=deadline or (ROOT/'STOP').exists():status='stopped';break
        optimizer.zero_grad(set_to_none=True)
        mean.sum().backward()
        optimizer.step();step+=1
    elapsed=time.monotonic()-begin
    path=folder/'observations.npz'
    np.savez_compressed(path,inputs=x0,labels=y0,input_ids=selected,layers=np.array(layers,np.int64),
                        initial_theta_raw=initial.numpy(),initial_logits=initial_logits,initial_BCE=initial_losses,
                        node_theta_raw=weights,node_BCE=losses,node_steps=hit_steps,node_mean_float32=hit_mean_float32,
                        node_epsilon=np.array(nodes),final_theta_raw=theta.detach().cpu().numpy())
    result={'key':key,'kind':kind,'dataset':dataset,'n':n,'hidden':hidden,'models':models,'seed':seed,'parameters':size,
            'training_prefix_sha256':array_hash(x0,y0,selected),'status':status,'steps':step,'seconds':elapsed,
            'reached_by_node':{str(eps):int((hit_steps[i]>=0).sum()) for i,eps in enumerate(nodes)},
            'max_numpy_float64_vs_torch_float32_per_sample_BCE_error':max_independent_error,
            'observations_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'loss_recording':'每个初始化第一次到达节点时单独保存，float64复算逐样本BCE；不以验证集早停。'}
    atomic_json(folder/'RESULT.json',result)
    print(stamp(),key,status,step,f'{elapsed:.2f}s',result['reached_by_node'],flush=True)
    return result


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=CUDA_VISIBLE_DEVICES:raise RuntimeError('GPU映射与顶部配置不同')
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用，不静默回退CPU')
    if (ROOT/'RUNNING.lock').exists():raise RuntimeError('已有运行标记，拒绝重复启动')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    hashes=prepare_data()
    environment={'python':__import__('sys').version,'torch':str(torch.__version__),'cuda':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(0),'physical_gpu':CUDA_VISIBLE_DEVICES,'dtype':'float32','tf32':False,
                 'cpu_threads':2,'deterministic':True,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']}
    atomic_json(ROOT/'PLAN.json',{'config':CONFIG,'dataset_hashes':hashes,'environment':environment,
                                 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'scope':'只有真实训练，无SMC；旧队列不恢复。固定样本loss异质性及初始化间稳定性诊断。'})
    lock=ROOT/'RUNNING.lock'
    with lock.open('x') as f:json.dump({'pid':os.getpid()},f)
    start=time.monotonic();deadline=start+CONFIG['total_seconds'];rows=[];status='completed'
    jobs=[(d,n,h,'random') for d in range(4) for n in CONFIG['counts'] for h in CONFIG['architectures']]
    jobs += [(0,n,h,'projection0') for n in CONFIG['counts'] for h in CONFIG['architectures']]
    try:
        for job in jobs:
            if time.monotonic()>=deadline or (ROOT/'STOP').exists():status='stopped';break
            row=train_case(*job,deadline);rows.append(row)
            atomic_json(ROOT/'STATUS.json',{'status':'running','updated':stamp(),'finished_groups':len(rows),'total_groups':len(jobs),
                                         'seconds':time.monotonic()-start,'last':row['key']})
            if row['status']=='stopped':status='stopped';break
        atomic_json(ROOT/'RESULTS.json',{'status':status,'config':CONFIG,'environment':environment,'rows':rows,
                                        'seconds':time.monotonic()-start,'expected_groups':len(jobs)})
    except BaseException as exc:
        status='needs_attention'
        atomic_json(ROOT/'RESULTS.json',{'status':status,'config':CONFIG,'environment':environment,'rows':rows,
                                        'seconds':time.monotonic()-start,'expected_groups':len(jobs),'error':repr(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)
        atomic_json(ROOT/'STATUS.json',{'status':status,'updated':stamp(),'finished_groups':len(rows),'total_groups':len(jobs),
                                     'seconds':time.monotonic()-start,'active':False})


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
