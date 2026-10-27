"""真实Adam训练的快速agreement筛查；固定数据、独立初始化、未见输入评估。"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import frozen_memory as m
from memory_geometry import raw_forward
from config import WORK_DIR, DEVICE, RUN_HOURS, CUDA_VISIBLE_DEVICES
from data import random_table

ROOT=WORK_DIR / "agreement"
HIDDEN=((32,),(32,32),(32,32,32))
COUNTS=(16,32,48,64,80,96,112,128)
NODES=(.01,.002,.001)
MODELS=32


def agreement_stats(predictions,labels):
    """不同初始化的无放回成对一致率；不把模型对当独立重复。"""
    pred=np.asarray(predictions,dtype=bool)
    k=pred.sum(0).astype(float);count=len(pred)
    p=k/count
    agreement=float(np.mean((k*(k-1)+(count-k)*(count-k-1))/(count*(count-1)))) if count>1 else None
    majority=np.where(p>.5,1.,np.where(p<.5,0.,.5))
    correct=np.where(majority==.5,.5,majority==labels)
    bias=float(p.mean())
    return {"models":count,"unseen_points":len(labels),"pair_agreement":agreement,
            "mean_model_accuracy":float(np.mean(pred==labels[None])),
            "majority_accuracy_ties_half":float(np.mean(correct)),
            "mean_predicted_one_fraction":bias,"label_one_fraction":float(np.mean(labels)),
            "constant_bias_agreement_reference":bias*bias+(1-bias)**2,
            "mean_2p_minus1_squared":float(np.mean((2*p-1)**2)),
            "caution":"成对统计的独立单位是初始化，不是所有模型对；类别偏置也能贡献agreement。"}


def jobs():
    result=[]
    for dataset in range(4):
        for n in COUNTS:
            for hidden in HIDDEN:
                result.append((dataset,n,hidden,"random"))
    # 固定一个真实简单规则作正对照，不用它挑选学习率或随机表。
    for n in COUNTS:
        for hidden in HIDDEN:result.append((0,n,hidden,"projection0"))
    return result


def train_case(dataset,n,hidden,kind,deadline):
    name=f"{kind}_d{dataset}_n{n:03d}_h{'_'.join(map(str,hidden))}"
    folder=ROOT/"agreement"/name
    folder.mkdir(parents=True,exist_ok=True)
    result_file=folder/"RESULT.json"
    if result_file.exists():return json.loads(result_file.read_text(encoding="utf-8"))
    x0,original_y,order=random_table(dataset)
    y0=original_y if kind=="random" else (x0[:,0]>0).astype(np.uint8)
    indices=order[:n];unseen=order[n:]
    x=torch.from_numpy(x0[indices]).to('cuda:0');y=torch.tensor(y0[indices],dtype=torch.float32,device='cuda:0')
    all_x=torch.from_numpy(x0).to('cuda:0')
    _,layers,size=m.parameter_blocks(hidden,8)
    seed=int.from_bytes(hashlib.sha256(("training_agreement_v1:"+name).encode()).digest()[:8],'big')%(2**62)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    initial=torch.randn(MODELS,size,generator=gen)
    for start,_,end,fan_in,_ in layers:initial[:,start:end]/=math.sqrt(fan_in)
    theta=torch.nn.Parameter(initial.to('cuda:0'))
    optimizer=torch.optim.Adam([theta],lr=.003,weight_decay=0.)
    identity={"key":name,"dataset":dataset,"n":n,"hidden":list(hidden),"kind":kind,"models":MODELS,"seed":seed,
              "lr":.003,"optimizer":"Adam","weight_decay":0.,"raw_parameter_optimization":True,"full_batch":True,
              "parameters":size,"max_steps":20000,"max_seconds":90.,"nodes":list(NODES),"dtype":"float32",
              "dataset_hash":m.array_digest(x0,original_y,order),"training_prefix_hash":m.array_digest(x0[indices],y0[indices],indices),
              "torch":str(torch.__version__),"gpu":torch.cuda.get_device_name(0),"physical_gpu":CUDA_VISIBLE_DEVICES}
    prediction=np.full((len(NODES),MODELS,256),-1,dtype=np.int8)
    loss_at_hit=np.full((len(NODES),MODELS),np.nan)
    step_at_hit=np.full((len(NODES),MODELS),-1,dtype=np.int64)
    step=0;seconds=0.
    checkpoint=folder/"checkpoint.pt"
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise RuntimeError("agreement续跑身份变化")
        with torch.no_grad():theta.copy_(saved['theta'].to('cuda:0'))
        optimizer.load_state_dict(saved['optimizer'])
        prediction=saved['predictions'].numpy();loss_at_hit=saved['losses'].numpy();step_at_hit=saved['steps'].numpy()
        step=saved['step'];seconds=saved['seconds']
    begin=time.monotonic();base=seconds;last_save=begin
    status='partial'
    while True:
        logits=raw_forward(theta,x,layers)
        losses=torch.nn.functional.softplus((1-2*y[None])*logits).mean(1)
        values=losses.detach().cpu().numpy()
        if not np.isfinite(values).all():raise RuntimeError("agreement非有限loss")
        new=[np.flatnonzero((step_at_hit[i]<0)&(values<=eps)) for i,eps in enumerate(NODES)]
        if any(len(indices)>0 for indices in new):
            with torch.no_grad():bits=(raw_forward(theta,all_x,layers)>=0).cpu().numpy()
            for i,hit in enumerate(new):
                prediction[i,hit]=bits[hit];loss_at_hit[i,hit]=values[hit];step_at_hit[i,hit]=step
        seconds=base+time.monotonic()-begin
        if np.all(step_at_hit[-1]>=0):status='completed'
        elif step>=20000 or seconds>=90.:status='budget_censored'
        paused=time.monotonic()>=deadline or (ROOT/'STOP').exists()
        if status!='partial' or paused or time.monotonic()-last_save>=20:
            payload={"identity":identity,"theta":theta.detach().cpu(),"optimizer":optimizer.state_dict(),
                     "predictions":torch.from_numpy(prediction),"losses":torch.from_numpy(loss_at_hit),"steps":torch.from_numpy(step_at_hit),
                     "step":step,"seconds":seconds}
            m.atomic_write(checkpoint,lambda stream:torch.save(payload,stream),binary=True)
            m.atomic_npz(folder/"predictions.npz",predictions=prediction,loss_at_hit=loss_at_hit,step_at_hit=step_at_hit,
                         labels=y0,order=order,unseen_indices=unseen)
            last_save=time.monotonic()
        if status!='partial' or paused:break
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step();step+=1
    nodes=[]
    for i,eps in enumerate(NODES):
        hit=step_at_hit[i]>=0
        row={"epsilon":eps,"reached":int(hit.sum()),"expected":MODELS,"all_reached":bool(hit.all()),
             "actual_loss_min":float(loss_at_hit[i,hit].min()) if hit.any() else None,
             "actual_loss_max":float(loss_at_hit[i,hit].max()) if hit.any() else None}
        if hit.any():row.update(agreement_stats(prediction[i,hit][:,unseen],y0[unseen]))
        nodes.append(row)
    result={"identity":identity,"status":status,"steps":step,"seconds":seconds,"nodes":nodes,
            "scope":"先达到训练BCE阈值时保存输出；未到达者是删失，非零agreement。未见标签只做事后评分。"}
    m.atomic_json(folder/("RESULT.json" if status!='partial' else "PARTIAL.json"),result)
    print(name,status,step,round(seconds,2),[(r['epsilon'],r['reached'],r.get('pair_agreement')) for r in nodes],flush=True)
    return result


def write_report(rows,status,seconds):
    m.atomic_json(ROOT/"AGREEMENT.json",{"status":status,"elapsed_seconds":seconds,"rows":rows})
    lines=["# 真实训练agreement快速筛查","",f"状态：{status}；本次累计{seconds/60:.1f}分钟。",
           "32个独立初始化；Adam lr=.003，无权重衰减；原四份随机表、嵌套前缀。仅在未训练的输入上测agreement。",
           "同一结构/数据固定训练BCE节点.01,.002,.001，记录第一次穿越的实际loss；未到达的模型单独计数。",
           "单组90秒/20000步上限；不根据agreement或测试标签修改超参。另加固定projection0正对照。",
           "高agreement是值得追查的筛查信号，不是规律证明；零假设应看同样协议随机标签的分布，不应一律设成0.5。",
           "", "以下为BCE<=.001；括号显示达到阈值的初始化数。缺失不补零。", "",
           "| 类型 | 表 | n | 隐层 | 到达 | agreement | 平均未见准确率 | 多数投票准确率 |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        info=row['identity'];node=row['nodes'][-1]
        show=lambda key:'未达到' if key not in node or node[key] is None else f"{node[key]:.4f}"
        lines.append(f"| {info['kind']} | {info['dataset']} | {info['n']} | {info['hidden']} | {node['reached']}/32 | {show('pair_agreement')} | {show('mean_model_accuracy')} | {show('majority_accuracy_ties_half')} |")
    lines += ["", "这些是训练算法推前分布的指标，不是SMC体积、K或收缩系数的估计。四份表只能作有限筛查，不能校准极小p值。"]
    m.atomic_text(ROOT/"AGREEMENT.md","\n".join(lines)+"\n")


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=CUDA_VISIBLE_DEVICES:raise RuntimeError('GPU映射与顶部配置不同')
    budget=RUN_HOURS*3600.
    cfg=m.resolve_config({"DEVICE":"cuda:0","CPU_THREADS":2})
    begin=time.monotonic();deadline=begin+budget;rows=[];status='completed'
    with m.execution_environment(cfg) as (_,environment,_):
        m.atomic_json(ROOT/'AGREEMENT_ENVIRONMENT.json',environment)
        for args in jobs():
            if time.monotonic()>=deadline or (ROOT/'STOP').exists():status='budget_stopped';break
            result=train_case(*args,deadline);rows.append(result)
            write_report(rows,'running',time.monotonic()-begin)
            if result['status']=='partial':status='budget_stopped';break
    write_report(rows,status,time.monotonic()-begin)


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
