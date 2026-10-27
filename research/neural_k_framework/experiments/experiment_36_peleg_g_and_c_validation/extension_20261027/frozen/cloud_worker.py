"""云端B：深loss优化器对照和增强SMC；数据不变，优化器采用独立种子。"""

import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from common_local import ROOT,c,e,old,LegacyModel,load_job,stop_requested,dump,now


def collect(theta,initial,loss,hard,seen,records,step,epoch,data,model,cfg):
    with torch.no_grad():
        for j,eps in enumerate(cfg["nodes"]):
            ids=np.flatnonzero((hard & torch.isfinite(loss) & (loss<=eps)).cpu().numpy() & ~seen[:,j])
            if not len(ids):
                continue
            ii=torch.as_tensor(ids,device=theta.device)
            pred=old.predictions(theta[ii].detach(),data["probe_x"],model,cfg)
            z=old.logits_chunked(theta[ii].detach(),data["x"],model,cfg).double()
            margins=(z[...,1]-z[...,0])*(2*data["y"]-1)
            for k,(i,p) in enumerate(zip(ids,pred)):
                records[str(eps)].append(dict(init_id=int(i),step=step,epoch=epoch,train_loss=float(loss[i]),
                    margin_mean=float(margins[k].mean()),margin_variance=float(margins[k].var(correction=0)),
                    outside_box_fraction=float((theta[i].abs()>.2000001).float().mean()),
                    displacement_norm=float(torch.linalg.vector_norm(theta[i]-initial[i])),
                    prediction=torch.from_numpy(p)))
            seen[ids,j]=True


def save_optimizer(folder,state,theta,g,elapsed):
    packed={**state,"theta":theta.detach().cpu(),"batch_rng":g.get_state(),"seconds":elapsed,
            "seen":torch.from_numpy(state["seen"])}
    c.atomic_torch(folder/"checkpoint.pt",packed)
    for eps,rows in state["records"].items():
        if rows:
            rows=sorted(rows,key=lambda r:r["init_id"])
            arrays={k:np.array([r[k] for r in rows]) for k in rows[0] if k!="prediction"}
            arrays["prediction"]=np.stack([r["prediction"].numpy() for r in rows])
            c.atomic_npz(folder/(old.node_label(float(eps))+".npz"),**arrays)


def optimize(plan,job,arrays,folder,device,deadline):
    cfg={**plan["reference_config"],"nodes":plan["deep_nodes"] if job.get("deep") else plan["primary_nodes"],
         "model_chunk":32}
    model=old.build_model(plan["network"])
    seed_offset=plan["optimizer_seed_offset"]
    ig=torch.Generator().manual_seed(cfg["data_seed"]+7000+seed_offset)
    models=4 if plan["smoke"] else 128
    initial=old.sample_prior(128,model,cfg,ig,"cpu")[:models].to(device)
    if job["method"]=="sgd":
        g=torch.Generator().manual_seed(cfg["data_seed"]+7001+seed_offset)
    else:
        g=torch.Generator(device=device).manual_seed(c.seed("cloud_B_deep_20261025",job["case"],job["value"],seed_offset))
    data={k:torch.from_numpy(arrays[k]).to(device) for k in ("x","y","probe_x")}
    identity=c.digest(dict(job=job,sources=plan["source_hashes"],data=plan["datasets"][job["case"]],
                           nodes=cfg["nodes"],models=models,optimizer_seed_offset=seed_offset))
    if (folder/"checkpoint.pt").exists():
        state=torch.load(folder/"checkpoint.pt",weights_only=True,map_location="cpu")
        if state.get("identity")!=identity:
            raise RuntimeError("优化器检查点身份不一致")
        theta=state.pop("theta").to(device)
        g.set_state(state.pop("batch_rng"))
        state["seen"]=state["seen"].numpy()
    else:
        theta=initial.clone()
        state=dict(identity=identity,step=0,epoch=0,seconds=0.,seen=np.zeros((models,len(cfg["nodes"])),bool),
                   records={str(v):[] for v in cfg["nodes"]})
    base_seconds=state["seconds"]
    tick=last_save=last_log=time.monotonic()
    with torch.no_grad():
        loss,hard,_=old.energies(theta,data["x"],data["y"],model,cfg)
        collect(theta,initial,loss,hard,state["seen"],state["records"],state["step"],state["epoch"],data,model,cfg)
    if job["method"]=="sgd":
        theta.requires_grad_(True)
        opt=torch.optim.SGD([theta],lr=job["value"])
    while not state["seen"].all() and not stop_requested() and time.monotonic()<deadline:
        if plan["smoke"] and state["step"]>=2:
            break
        if job["method"]=="sgd":
            if state["epoch"]>=4000:
                break
            order=torch.randperm(len(data["y"]),generator=g)
            for ids in order.split(cfg["sgd_batch_size"]):
                ids=ids.to(device)
                opt.zero_grad(set_to_none=True)
                for first in range(0,models,cfg["sgd_model_chunk"]):
                    z=old.forward(theta[first:first+cfg["sgd_model_chunk"]],data["x"][ids],model)
                    values=F.cross_entropy(z.reshape(-1,2),data["y"][ids].expand(len(z),-1).reshape(-1),reduction="none")
                    values.reshape(len(z),-1).mean(1).sum().backward()
                theta.grad[:,model["biases"]]=0
                opt.step()
                state["step"]+=1
                with torch.no_grad():
                    loss,hard,_=old.energies(theta,data["x"],data["y"],model,cfg)
                    collect(theta,initial,loss,hard,state["seen"],state["records"],state["step"],
                            state["epoch"]+1,data,model,cfg)
            state["epoch"]+=1
        else:
            if state["step"]>=50000:
                break
            with torch.no_grad():
                active=torch.as_tensor(np.flatnonzero(~state["seen"][:,-1]),device=device)
                sigma=job["value"]/math.sqrt(1+state["step"]/2000)
                delta=torch.randn((len(active),model["parameters"]),device=device,generator=g)*sigma
                delta[:,model["biases"]]=0
                current=theta[active]
                plus,minus=current+delta,current-delta
                lp,hp,_=old.energies(plus,data["x"],data["y"],model,cfg)
                lm,hm,_=old.energies(minus,data["x"],data["y"],model,cfg)
                lp=torch.where(torch.isfinite(lp),lp,float("inf"))
                lm=torch.where(torch.isfinite(lm),lm,float("inf"))
                cp=(lp<loss[active]) & (lp<=lm)
                cm=(lm<loss[active]) & (lm<lp)
                theta[active]=torch.where(cp[:,None],plus,torch.where(cm[:,None],minus,current))
                loss[active]=torch.where(cp,lp,torch.where(cm,lm,loss[active]))
                hard[active]=torch.where(cp,hp,torch.where(cm,hm,hard[active]))
                state["step"]+=1
                collect(theta,initial,loss,hard,state["seen"],state["records"],state["step"],0,data,model,cfg)
        if time.monotonic()-last_log>30:
            print(f"{job['id']} step={state['step']} 到达={state['seen'].sum(0).tolist()}",flush=True)
            last_log=time.monotonic()
        if time.monotonic()-last_save>120:
            save_optimizer(folder,state,theta,g,base_seconds+time.monotonic()-tick)
            last_save=time.monotonic()
    seconds=base_seconds+time.monotonic()-tick
    save_optimizer(folder,state,theta,g,seconds)
    exhausted=state["epoch"]>=4000 if job["method"]=="sgd" else state["step"]>=50000
    status="completed" if state["seen"].all() else ("unresolved" if exhausted else "partial")
    result=dict(status=status,job=job,seconds=seconds,reached=state["seen"].sum(0).tolist(),
                planned=models,epoch=state["epoch"],step=state["step"],rows=[])
    for eps in cfg["nodes"]:
        path=folder/(old.node_label(eps)+".npz")
        if path.exists():
            records=c.read_npz(path)
            result["rows"].append(dict(epsilon=eps,models=len(records["init_id"]),
                accuracy=float((records["prediction"]==arrays["probe_y"]).mean()),
                loss_mean=float(records["train_loss"].mean())))
    dump(folder/"STATUS.json",result)
    if status=="completed":
        (folder/"checkpoint.pt").unlink(missing_ok=True)
    return result


def sample(plan,job,arrays,folder,device,deadline,environment):
    enhanced=job["enhanced"]
    cfg={**e.DEFAULTS,"particles":2048 if enhanced else 1024,"replicas":4,
         "sweeps":32 if enhanced else 8,"refresh_sweeps":96 if enhanced else 24,
         "nodes":plan["deep_nodes"] if enhanced else plan["primary_nodes"],
         "model_chunk":32,"input_chunk":16,"mutation_chunk":32,"slice_seconds":600.,
         "keep_final_checkpoint":False}
    if plan["smoke"]:
        cfg.update(particles=16,replicas=1,sweeps=1,refresh_sweeps=2,refresh_blocks=1,
                   max_levels_this_call=2,slice_seconds=5.)
    model=LegacyModel(plan["network"])
    data={k:torch.from_numpy(arrays[k]).to(device) for k in ("x","y","probe_x")}
    provenance=dict(protocol=plan["version"],sources=plan["source_hashes"],job=job,
                    dataset=plan["datasets"][job["case"]])
    c.guard_json(folder/"protocol.json",dict(sampler=e.scientific_config(cfg),provenance=provenance))
    if not enhanced and not plan["smoke"]:
        gc_cfg={**plan["reference_config"],"model_chunk":32,"output_dir":str(folder)}
        old.run_gc(data,model.old,gc_cfg,folder,deadline)
    for r in range(cfg["replicas"]):
        pool=folder/f"replica_{r}"
        while not (pool/"final.json").exists() and not stop_requested() and time.monotonic()<deadline:
            dump(folder/"STATUS.json",dict(status="running",job=job,pool=r,updated=now()))
            result=e.run_smc(model,data,cfg,pool,f"local20261025:{job['id']}:{r}",environment,deadline,provenance)
            if result["status"]!="partial" or plan["smoke"]:
                break
    finals=[c.read_json(p) for p in folder.glob("replica_*/final.json")]
    result=dict(status="completed" if len(finals)==cfg["replicas"] and all(f["status"]=="completed" for f in finals) else
                ("unresolved" if len(finals)==cfg["replicas"] else "partial"),
                job=job,terminal_pools=len(finals),completed_pools=sum(f["status"]=="completed" for f in finals))
    dump(folder/"STATUS.json",result)
    return result


def main():
    plan,job,arrays=load_job(os.environ["LOCAL_DIAG_JOB"])
    if not torch.cuda.is_available():
        raise RuntimeError("需要指定的CUDA设备")
    device=os.environ.get("LOCAL_DIAG_DEVICE","cuda:0")
    if job["method"]=="smc" and not plan["smoke"] and torch.cuda.get_device_properties(device).total_memory<26*2**30:
        raise RuntimeError("正式2048粒子配置需要32GB级显存；不自动降低粒子数。")
    environment=c.environment(device,False)
    environment["physical_gpu_requested"]=plan["physical_gpu"]
    folder=ROOT/"results"/job["id"]
    folder.mkdir(parents=True,exist_ok=True)
    dump(folder/f"environment_{time.time_ns()}.json",environment)
    c.install_signals()
    original_expired=c.expired
    c.expired=lambda deadline: original_expired(deadline) or (ROOT/"STOP").exists()
    seconds=float(os.environ["LOCAL_DIAG_SECONDS"])
    deadline=time.monotonic()+seconds
    print(f"{now()} START {job['id']} | {environment.get('gpu')} | 预算{seconds/3600:.2f}h",flush=True)
    if job["method"]=="smc":
        result=sample(plan,job,arrays,folder,device,deadline,environment)
    else:
        result=optimize(plan,job,arrays,folder,device,deadline)
    print(f"{now()} END {job['id']} {result['status']}",flush=True)


if __name__=="__main__":
    main()

