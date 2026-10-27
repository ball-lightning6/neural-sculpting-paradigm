"""固定16初始化的CPU Adam试跑，包含跨阈值诊断；不改已有队列。"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=REPO/"research/wide_cnn_local_diagnostics_20261025"
BACKEND=REPO/"research/smc_focus_20261024/followup"
sys.path.insert(0,str(BACKEND))
from legacy_adapter import old
import common as c

CONFIG={"cases":["partition0_n32_s0","partition2_n32_s0"],"rates":[.0001,.001],
        "init_ids":list(range(16)),"nodes":[.03,.01,.003,.001],"diagnostic_nodes":[.003,.001],
        "max_steps":4000,"max_seconds_per_model":60.,"total_seconds":1500.,
        "cpu_threads":2,"device":"cpu","dtype":"float32","weight_decay":0.,
        "betas":[.9,.999],"adam_eps":1e-8,"interpolation_iterations":24,
        "interpolation_relative_loss_tolerance":.002,
        "scope":"固定的16个初始化小试；插值诊断不是优化器实际状态，不是SMC样本。"}


def node_name(eps):
    return old.node_label(eps)


def atomic_json(path,value):
    c.atomic_json(path,value)


@torch.no_grad()
def observe(theta,loss,z,initial,x,model,cfg,probe,step):
    margins=(z[0,:,1].double()-z[0,:,0].double())*(2*x["y"]-1)
    prediction=old.predictions(theta,probe,model,cfg)[0]
    return {"step":int(step),"train_loss":float(loss),"training_all_correct":bool((z[0].argmax(-1)==x["y"]).all()),
            "margin_mean":float(margins.mean()),"margin_variance":float(margins.var(correction=0)),
            "minimum_margin":float(margins.min()),
            "outside_box_fraction":float((theta.abs()>.2000001).float().mean()),
            "displacement_norm":float(torch.linalg.vector_norm(theta-initial)),
            "prediction":prediction.astype(np.uint8)}


@torch.no_grad()
def energy(theta,x,model):
    z=old.forward(theta,x["x"],model)
    gap=z[0,:,1].double()-z[0,:,0].double()
    losses=F.softplus(gap*(1-2*x["y"]))
    return float(losses.mean()),z


@torch.no_grad()
def interpolate(before,after,eps,x,model):
    left,right=0.,1.
    left_loss,_=energy(before,x,model)
    right_loss,right_z=energy(after,x,model)
    if not left_loss>eps>=right_loss:
        raise RuntimeError("插值没有有效端点包围")
    best=after.detach().clone()
    for _ in range(CONFIG["interpolation_iterations"]):
        if eps-right_loss<=eps*CONFIG["interpolation_relative_loss_tolerance"]:
            break
        middle=(left+right)/2
        candidate=before+(after-before)*middle
        value,z=energy(candidate,x,model)
        if value<=eps:
            right,right_loss,right_z,best=middle,value,z,candidate
        else:
            left,left_loss=middle,value
    return best,right_loss,right_z,{"alpha":right,"bracket_alpha":[left,right],
        "relative_loss_error":(eps-right_loss)/eps,
        "scope":"连续更新线段内的一个包围根，不保证非单调线段上的最早根；不改变训练轨迹。"}


def serialize(folder,records,diagnostics,result):
    meta=[]
    for kind,eps,row in records:
        name=f"{kind}_{node_name(eps)}"
        c.atomic_npz(folder/(name+".npz"),prediction=row["prediction"],train_loss=np.array(row["train_loss"]))
        meta.append({"kind":kind,"epsilon":eps,"file":name+".npz",**{k:v for k,v in row.items() if k!="prediction"}})
    atomic_json(folder/"RESULT.json",{**result,"records":meta,"crossings":diagnostics})


def run_model(case,lr,init_id,initial,data,model,cfg,deadline):
    folder=ROOT/"results"/case/f"adam_{lr:g}"/f"init_{init_id:03d}"
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"RESULT.json").exists():
        return json.loads((folder/"RESULT.json").read_text(encoding="utf-8"))
    theta=initial.clone().requires_grad_(True)
    optimizer=torch.optim.Adam([theta],lr=lr,betas=tuple(CONFIG["betas"]),eps=CONFIG["adam_eps"],weight_decay=0.,foreach=False)
    generator=torch.Generator().manual_seed(cfg["data_seed"]+7001)
    x={"x":torch.from_numpy(data["x"]),"y":torch.from_numpy(data["y"])}
    probe=torch.from_numpy(data["probe_x"])
    old_loss,z=energy(theta,x,model)
    seen,records,diagnostics=set(),[],[]
    started=time.monotonic()
    step,epoch,offset=0,0,0
    order=None
    while True:
        for eps in CONFIG["nodes"]:
            if eps not in seen and old_loss<=eps:
                if not bool((z[0].argmax(-1)==x["y"]).all()):
                    continue
                records.append(("actual",eps,observe(theta,old_loss,z,initial,x,model,cfg,probe,step)))
                seen.add(eps)
        if len(seen)==len(CONFIG["nodes"]):
            status="completed"
            break
        if step>=CONFIG["max_steps"] or time.monotonic()-started>=CONFIG["max_seconds_per_model"] or time.monotonic()>=deadline:
            status="budget_censored"
            break
        if order is None or offset==len(x["y"]):
            order=torch.randperm(len(x["y"]),generator=generator)
            offset=0
            epoch+=1
        ids=order[offset:offset+cfg["sgd_batch_size"]]
        offset+=len(ids)
        before=theta.detach().clone()
        before_loss,before_z=old_loss,z
        optimizer.zero_grad(set_to_none=True)
        logits=old.forward(theta,x["x"][ids],model)
        F.cross_entropy(logits.reshape(-1,2),x["y"][ids]).backward()
        theta.grad[:,model["biases"]]=0
        optimizer.step()
        step+=1
        old_loss,z=energy(theta,x,model)
        if not math.isfinite(old_loss):
            raise RuntimeError("Adam出现非有限loss，停止并保留已有结果。")
        crossed=[eps for eps in CONFIG["diagnostic_nodes"] if eps not in seen and before_loss>eps>=old_loss]
        for eps in crossed:
            before_row=observe(before,before_loss,before_z,initial,x,model,cfg,probe,step-1)
            point,value,point_z,detail=interpolate(before,theta.detach(),eps,x,model)
            interpolated=observe(point,value,point_z,initial,x,model,cfg,probe,step)
            records.extend([("before",eps,before_row),("interpolated",eps,interpolated)])
            diagnostics.append({"epsilon":eps,"step":step,"loss_before":before_loss,"loss_after":old_loss,
                                "after_over_threshold":old_loss/eps,"crossed_node_count":sum(old_loss<=v<before_loss for v in CONFIG["nodes"]),
                                **detail})
    result={"status":status,"case":case,"lr":lr,"init_id":init_id,"step":step,"epoch":epoch,
            "seconds":time.monotonic()-started,"final_loss":old_loss,"initial_sha256":c.array_hash(initial.numpy()),
            "device":"cpu","reached":sorted(seen,reverse=True)}
    serialize(folder,records,diagnostics,result)
    print(f"{case} Adam{lr:g} init={init_id} {status} steps={step} loss={old_loss:.6g} {result['seconds']:.1f}s",flush=True)
    return result


def main():
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    plan=json.loads((SOURCE/"PLAN.json").read_text(encoding="utf-8"))
    for name,digest in plan["source_hashes"].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("原科学源码改变")
    model=old.build_model(plan["network"])
    cfg={**plan["reference_config"],"model_chunk":1}
    generator=torch.Generator().manual_seed(cfg["data_seed"]+7000)
    initial=old.sample_prior(128,model,cfg,generator,"cpu")
    protocol={"config":CONFIG,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_sources":plan["source_hashes"],"initial_bank_sha256":c.array_hash(initial.numpy()),
              "parameters":model["parameters"],"torch":str(torch.__version__),"batch_size":cfg["sgd_batch_size"],
              "pairing":"初值及每模型batch顺序与原本地SGD方案一致；CPU/GPU数值轨迹不声称位级一致。"}
    c.guard_json(ROOT/"PROTOCOL.json",protocol)
    deadline=time.monotonic()+CONFIG["total_seconds"]
    results=[]
    datasets={}
    for case in CONFIG["cases"]:
        with np.load(SOURCE/"data"/case/"dataset.npz",allow_pickle=False) as a:
            data={k:a[k].copy() for k in ("x","y","probe_x")}
        expected=plan["datasets"][case]["arrays"] if "arrays" in plan["datasets"][case] else plan["datasets"][case]["array_hashes"]
        for k,v in data.items():
            if c.array_hash(v)!=expected[k]:
                raise RuntimeError("数据hash不符："+k)
        datasets[case]=data
    for init_id in CONFIG["init_ids"]:
        for case in CONFIG["cases"]:
            for lr in CONFIG["rates"]:
                if time.monotonic()>=deadline:
                    atomic_json(ROOT/"STATUS.json",{"status":"budget_stopped","results":results})
                    return
                result=run_model(case,lr,init_id,initial[init_id:init_id+1].clone(),datasets[case],model,cfg,deadline)
                results.append(result)
                atomic_json(ROOT/"STATUS.json",{"status":"running","results":results,"active_case":case,"active_lr":lr})
    atomic_json(ROOT/"STATUS.json",{"status":"completed","results":results})


if __name__=="__main__":
    main()
