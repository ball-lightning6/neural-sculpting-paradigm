"""同一 W8 / 数据 / 初始化的 STP 对照，无梯度，无测试集调参。"""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from review_utils import ROOT, Review, dump
from replay_wide_sgd import load_core, OLD_DATA, REFERENCE_SHA


MODE = os.environ.get("WIDE_STP_MODE", "benchmark")
DEVICE = "cuda:1"
CASE = "partition0_seed0_n32"
SCALES = (.003, .001, .0003)
MODELS = int(os.environ.get("WIDE_STP_MODELS", "128"))
MAX_STEPS = 50000
MAX_SECONDS = float(os.environ.get("WIDE_STP_SECONDS", "300"))
OUT = ROOT / ("local_wide_stp_" + str(MODELS))


def load_data(review, core, device):
    prefix = "cv/original_static/cnn_w8__" + CASE + "/"
    expected = review.json(prefix + "dataset_manifest.json")["arrays"]
    with zipfile.ZipFile(OLD_DATA) as archive:
        raw = archive.read("smc_scale_suite_20261023/results_static/cnn_w4__" + CASE + "/dataset.npz")
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            cpu = {k:torch.from_numpy(saved[k].copy()) for k in saved.files}
    assert {k:core.digest_array(v) for k,v in cpu.items()} == expected
    return {k:cpu[k].to(device) for k in ("x","y","probe_x")}, cpu["probe_y"].numpy()


@torch.no_grad()
def stp_step(theta, current_loss, current_hard, sigma, generator, data, model, cfg, core, active):
    ids = torch.nonzero(active).flatten()
    if not len(ids):
        return theta, current_loss, current_hard, 0
    old = theta[ids]
    delta = torch.randn(old.shape,device=theta.device,generator=generator)*sigma
    if cfg["fixed_zero_bias"]:
        delta[:,model["biases"]]=0
    plus = old+delta
    minus = old-delta
    lp,hp,_ = core.energies(plus,data["x"],data["y"],model,cfg)
    lm,hm,_ = core.energies(minus,data["x"],data["y"],model,cfg)
    lp=torch.where(torch.isfinite(lp),lp,float("inf"))
    lm=torch.where(torch.isfinite(lm),lm,float("inf"))
    cp = torch.isfinite(lp) & (lp<current_loss[ids]) & (lp<=lm)
    cm = torch.isfinite(lm) & (lm<current_loss[ids]) & (lm<lp)
    theta[ids] = torch.where(cp[:,None],plus,torch.where(cm[:,None],minus,old))
    current_loss[ids] = torch.where(cp,lp,torch.where(cm,lm,current_loss[ids]))
    current_hard[ids] = torch.where(cp,hp,torch.where(cm,hm,current_hard[ids]))
    return theta,current_loss,current_hard,int((cp|cm).sum())


@torch.no_grad()
def record_nodes(theta,initial,loss,hard,seen,nodes,step,core,cfg,data,model,device):
    for j,eps in enumerate(cfg["nodes"]):
        ids=np.flatnonzero((hard & (loss<=eps)).cpu().numpy() & ~seen[:,j])
        if not len(ids):
            continue
        ii=torch.as_tensor(ids,device=device)
        pred=core.predictions(theta[ii],data["probe_x"],model,cfg)
        for idx,p in zip(ids,pred):
            nodes[str(eps)].append(dict(init_id=int(idx),step=step,train_loss=float(loss[idx]),
                outside_initial_box_fraction=float((theta[idx].abs()>cfg["prior_scale"]+1e-7).float().mean()),
                displacement_norm=float(torch.linalg.vector_norm(theta[idx]-initial[idx])),prediction=torch.from_numpy(p)))
        seen[ids,j]=True


@torch.no_grad()
def benchmark(core,cfg,data,model,device):
    rows=[]
    for n in (8,32,128):
        initial=torch.Generator().manual_seed(cfg["data_seed"]+7000)
        theta=core.sample_prior(128,model,cfg,initial,"cpu")[:n].to(device)
        loss,hard,_=core.energies(theta,data["x"],data["y"],model,cfg)
        reference_cfg={**cfg,"model_chunk":128}
        reference_loss,reference_hard,_=core.energies(theta,data["x"],data["y"],model,reference_cfg)
        torch.testing.assert_close(loss,reference_loss,atol=1e-4,rtol=2e-5)
        assert bool((hard==reference_hard).all())
        chunk_error=float((loss-reference_loss).abs().max())
        active=torch.ones(n,device=device,dtype=torch.bool)
        generator=torch.Generator(device=device).manual_seed(202610250901+n)
        for _ in range(3):
            theta,loss,hard,_=stp_step(theta,loss,hard,.001,generator,data,model,cfg,core,active)
        torch.cuda.synchronize()
        tick=time.perf_counter()
        for _ in range(20):
            old=loss.clone()
            theta,loss,hard,_=stp_step(theta,loss,hard,.001,generator,data,model,cfg,core,active)
            assert bool((loss<=old).all())
        torch.cuda.synchronize()
        rows.append(dict(models=n,seconds_per_step=(time.perf_counter()-tick)/20,
                         objective_nonincreasing=True,biases_still_zero=bool((theta[:,model["biases"]]==0).all()),
                         evaluation_chunk=cfg["model_chunk"],chunk_loss_max_error=chunk_error))
    return rows


@torch.no_grad()
def run_group(index,sigma0,core,cfg,data,model,device,signature):
    folder=OUT/f"scale_{sigma0:g}"
    folder.mkdir(exist_ok=True)
    if (folder/"summary.json").exists():
        old=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        if old["status"]=="completed":
            return old
    generator=torch.Generator(device=device).manual_seed(202610250911+index*100003)
    ig=torch.Generator().manual_seed(cfg["data_seed"]+7000)
    initial=core.sample_prior(128,model,cfg,ig,"cpu")[:MODELS].to(device)
    checkpoint=folder/"checkpoint.pt"
    if checkpoint.exists():
        old=torch.load(checkpoint,weights_only=True,map_location="cpu")
        assert old["signature"]==signature
        theta,loss,hard=[old[k].to(device) for k in ("theta","loss","hard")]
        seen=old["seen"].numpy()
        nodes=old["nodes"]
        step_start=old["step"]
        prior_seconds=old["seconds"]
        generator.set_state(old["rng"])
    else:
        theta=initial.clone()
        loss,hard,_=core.energies(theta,data["x"],data["y"],model,cfg)
        seen=np.zeros((MODELS,len(cfg["nodes"])),dtype=bool)
        nodes={str(v):[] for v in cfg["nodes"]}
        step_start=0
        prior_seconds=0.
    tick=last_log=time.perf_counter()
    steps=step_start
    record_nodes(theta,initial,loss,hard,seen,nodes,steps,core,cfg,data,model,device)
    print(f"START STP sigma={sigma0}, {MODELS}初始化，从step={step_start}继续；本次最多{MAX_SECONDS}s",flush=True)
    for steps in range(step_start+1,MAX_STEPS+1):
        active=torch.from_numpy(~seen[:,-1]).to(device)
        if not bool(active.any()) or time.perf_counter()-tick>=MAX_SECONDS:
            steps-=1
            break
        before=loss.clone()
        sigma=sigma0/math.sqrt(1+(steps-1)/2000)
        theta,loss,hard,moved=stp_step(theta,loss,hard,sigma,generator,data,model,cfg,core,active)
        if not bool((loss<=before).all()):
            raise RuntimeError("STP 未满足只接受更低训练loss")
        record_nodes(theta,initial,loss,hard,seen,nodes,steps,core,cfg,data,model,device)
        if seen.all():
            break
        if time.perf_counter()-last_log>25:
            print(f"STP sigma={sigma0}, step={steps}, 到达={seen.sum(0).tolist()}/{MODELS}, 中位loss={float(loss.median()):.5g}",flush=True)
            last_log=time.perf_counter()
    elapsed=prior_seconds+time.perf_counter()-tick
    for eps,rows in nodes.items():
        if rows:
            rows=sorted(rows,key=lambda x:x["init_id"])
            arrays={k:np.array([r[k] for r in rows]) for k in rows[0] if k!="prediction"}
            arrays["prediction"]=np.stack([r["prediction"].numpy() for r in rows])
            np.savez_compressed(folder/("stp_"+core.node_label(float(eps))+".npz"),**arrays)
    payload=dict(signature=signature,theta=theta.cpu(),loss=loss.cpu(),hard=hard.cpu(),
                 seen=torch.from_numpy(seen),nodes=nodes,step=steps,seconds=elapsed,rng=generator.get_state())
    temp=checkpoint.with_suffix(".tmp")
    torch.save(payload,temp)
    temp.replace(checkpoint)
    result=dict(sigma0=sigma0,status="completed" if seen.all() else "budget_limited",models=MODELS,
                nodes_reached=seen.sum(0).tolist(),steps=steps,seconds=elapsed,
                lr_or_scale_selection="三档预先固定，全部保留；未按测试表现或SMC吻合选择。")
    dump(folder/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def analyze(review,cfg,labels,rows):
    prefix="cv/original_static/cnn_w8__"+CASE+"/"
    output=[]
    for group in rows:
        folder=OUT/f"scale_{group['sigma0']:g}"
        for eps in cfg["nodes"]:
            path=folder/("stp_"+review_label(eps)+".npz")
            if not path.exists():
                continue
            with np.load(path,allow_pickle=False) as saved:
                stp={k:saved[k].copy() for k in saved.files}
            old=review.arrays(prefix+"sgd_"+review_label(eps)+".review.npz")
            original={int(i):p for i,p in zip(old["init_id"],old["prediction"])}
            sgd=np.stack([original[int(i)] for i in stp["init_id"]])
            pools=[review.arrays(prefix+f"replica_{r}/"+review_label(eps)+".review.npz") for r in range(4)]
            q=np.concatenate([p["prediction"] for p in pools]).mean(0)
            individual=(stp["prediction"]==labels).mean(1)
            paired=individual-(sgd==labels).mean(1)
            output.append(dict(scale=group["sigma0"],epsilon=eps,models_reached=len(individual),models_planned=MODELS,
                stp_accuracy=float(individual.mean()),matched_sgd_accuracy=float((sgd==labels).mean()),
                smc_accuracy=float(np.where(labels,q,1-q).mean()),
                paired_accuracy_gap_pp=100*float(paired.mean()),
                paired_accuracy_gap_standard_error_pp=100*float(paired.std(ddof=1)/np.sqrt(len(paired))) if len(paired)>1 else None,
                point_mae_smc=float(np.abs(stp["prediction"].mean(0)-q).mean()),
                actual_stp_loss_mean=float(stp["train_loss"].mean()),
                actual_stp_loss_quantiles=np.quantile(stp["train_loss"],[0,.1,.5,.9,1]).tolist(),
                initial_box_outside_fraction_mean=float(stp["outside_initial_box_fraction"].mean()),
                note="未到达时仅是已到达子集，不能代表全部初始化；STP不是指定先验条件分布的采样器。"))
    dump(OUT/"ANALYSIS.json",output)
    print(json.dumps([r for r in output if r["epsilon"]==.003],ensure_ascii=False,indent=2))


def review_label(eps):
    return "bce_"+format(eps,".6g").replace(".","p")


def main():
    if MODE not in ("benchmark","run") or MODELS not in (8,32,64,128):
        raise ValueError("非法 MODE/MODELS")
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA")
    device=torch.device(DEVICE)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    core,review=load_core(),Review("A")
    cfg=review.json("cv/original_static/protocol.json")["config"]
    cfg["model_chunk"]=32
    data,labels=load_data(review,core,device)
    model=core.build_model({"name":"cnn_w8","kind":"cnn","scale":8})
    if MODE=="benchmark":
        result=dict(gpu=torch.cuda.get_device_name(device),benchmarks=benchmark(core,cfg,data,model,device),
                    no_test_labels_used_in_benchmark=True,core_sha256=REFERENCE_SHA)
        dump(ROOT/"STP_BENCHMARK_chunk32.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return
    OUT.mkdir(exist_ok=True)
    protocol=dict(core_sha256=REFERENCE_SHA,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        condition="cnn_w8__"+CASE,models=MODELS,scales=SCALES,decay="sigma0/sqrt(1+(step-1)/2000)",
        max_steps=MAX_STEPS,nodes=cfg["nodes"],device=str(device),gpu=torch.cuda.get_device_name(device),
        torch=str(torch.__version__),prior_scale=cfg["prior_scale"],fixed_zero_bias=cfg["fixed_zero_bias"],
        evaluation_chunk=cfg["model_chunk"],input_chunk=cfg["input_chunk"],
        training_rule="比较当前、加Gaussian扰动、减同一扰动三者的完整训练BCE；不反向传播，不裁剪到初始化盒。",
        early_stop="各初始化达到最深节点后冻结；只记录每个节点的首次hard-fit通过。")
    protocol=json.loads(json.dumps(protocol))
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (OUT/"protocol.json").exists():
        assert json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))==protocol
    else:
        dump(OUT/"protocol.json",protocol)
    lock=OUT/"RUNNING.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    try:
        results=[run_group(i,s,core,cfg,data,model,device,signature) for i,s in enumerate(SCALES)]
        dump(OUT/"summary.json",results)
        analyze(review,cfg,labels,results)
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    main()
