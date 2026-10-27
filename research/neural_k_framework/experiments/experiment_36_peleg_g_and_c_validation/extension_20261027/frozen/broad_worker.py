"""单组STP：复用原三点搜索核，保存随机流和各节点预测。"""

import hashlib
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
import shared as s

sys.path.insert(0,str(s.REVIEW))
from replay_wide_sgd import load_core
from wide_stp_control import stp_step


def save(folder,state,theta,loss,hard,generator):
    payload={**state,"theta":theta.detach().cpu(),"loss":loss.cpu(),"hard":hard.cpu(),
             "seen":torch.from_numpy(state["seen"]),"rng":generator.get_state()}
    temporary=folder/"checkpoint.tmp"
    torch.save(payload,temporary)
    s.replace(temporary,folder/"checkpoint.pt")
    for name,rows in state["records"].items():
        if not rows:
            continue
        rows=sorted(rows,key=lambda r:r["init_id"])
        arrays={key:np.array([r[key] for r in rows]) for key in rows[0] if key!="prediction"}
        arrays["prediction"]=np.stack([r["prediction"].numpy() for r in rows])
        path=folder/(name+".npz")
        temporary=folder/(name+".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**arrays)
        s.replace(temporary,path)


@torch.no_grad()
def record(theta,initial,loss,hard,state,data,model,cfg,core):
    for j,epsilon in enumerate([None]+cfg["nodes"]):
        valid=hard & torch.isfinite(loss)
        if epsilon is not None:
            valid &= loss<=epsilon
        indices=np.flatnonzero(valid.cpu().numpy() & ~state["seen"][:,j])
        if not len(indices):
            continue
        ids=torch.as_tensor(indices,device=theta.device)
        pred=core.predictions(theta[ids],data["probe_x"],model,cfg)
        z=core.logits_chunked(theta[ids],data["x"],model,cfg).double()
        margin=(z[...,1]-z[...,0])*(2*data["y"]-1)
        name="hard_first" if epsilon is None else core.node_label(epsilon)
        for k,(i,p) in enumerate(zip(indices,pred)):
            state["records"][name].append({"init_id":int(i),"step":state["step"],"train_loss":float(loss[i]),
                "margin_mean":float(margin[k].mean()),"margin_variance":float(margin[k].var(correction=0)),
                "outside_initial_box_fraction":float((theta[i].abs()>cfg["prior_scale"]+1e-7).float().mean()),
                "displacement_norm":float(torch.linalg.vector_norm(theta[i]-initial[i])),"prediction":torch.from_numpy(p.copy())})
        state["seen"][indices,j]=True


def run_group(folder,job,plan,arrays,device,seconds,stop,environment,step_limit=None):
    core=load_core()
    folder.mkdir(parents=True,exist_ok=True)
    existing=folder/"STATUS.json"
    cfg={**plan["reference_config"],"nodes":plan["nodes"],"model_chunk":plan["model_chunk"],"input_chunk":plan["input_chunk"]}
    model=core.build_model(job["network"])
    data={k:torch.from_numpy(arrays[k]).to(device) for k in ("x","y","probe_x")}
    ig=torch.Generator().manual_seed(plan["initial_seed"])
    initial=core.sample_prior(plan["models"],model,cfg,ig,"cpu").to(device)
    seed=int.from_bytes(hashlib.sha256(f"{plan['perturbation_seed_base']}:{job['id']}".encode()).digest()[:8],"big")%(2**62)
    generator=torch.Generator(device=device).manual_seed(seed)
    identity=s.object_hash({"job":job,"plan_science":{k:plan[k] for k in ("source_hashes","nodes","models","model_chunk","input_chunk","initial_seed","perturbation_seed_base","decay","dtype","tf32","deterministic")},
                            "data":plan["datasets"][job["case"]]["arrays"],"environment":environment})
    if existing.exists() and s.read(existing)["status"] in ("completed","budget_limited","step_limited"):
        result=s.read(existing)
        if result.get("identity")!=identity:
            raise RuntimeError("已完成结果的协议或环境身份不符")
        return result
    checkpoint=folder/"checkpoint.pt"
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=True,map_location="cpu")
        if state["identity"]!=identity:
            raise RuntimeError("STP断点身份/环境不匹配")
        theta,loss,hard=[state.pop(k).to(device) for k in ("theta","loss","hard")]
        generator.set_state(state.pop("rng"))
        state["seen"]=state["seen"].numpy()
    else:
        theta=initial.clone()
        loss,hard,_=core.energies(theta,data["x"],data["y"],model,cfg)
        state={"identity":identity,"step":0,"seconds":0.,"seen":np.zeros((plan["models"],len(cfg["nodes"])+1),dtype=bool),
               "records":{name:[] for name in ["hard_first"]+[core.node_label(eps) for eps in cfg["nodes"]]},
               "environment":environment,"seed":seed}
    initial_step=state["step"]
    base=state["seconds"]
    start=last_save=last_log=time.monotonic()
    deadline=start+min(seconds,max(0.,job["max_seconds"]-base))
    record(theta,initial,loss,hard,state,data,model,cfg,core)
    while not state["seen"].all() and state["step"]<job["max_steps"] and time.monotonic()<deadline and not stop[0]:
        if (s.ROOT/"STOP").exists():
            stop[0]=True
            break
        if step_limit is not None and state["step"]-initial_step>=step_limit:
            break
        active=torch.from_numpy(~state["seen"][:,-1]).to(device)
        before=loss.clone()
        sigma=job["sigma0"]/math.sqrt(1+state["step"]/2000)
        theta,loss,hard,_=stp_step(theta,loss,hard,sigma,generator,data,model,cfg,core,active)
        state["step"]+=1
        if not bool(torch.isfinite(loss).all()) or not bool((loss<=before).all()):
            raise RuntimeError("STP单调下降或有限性核验失败")
        if torch.count_nonzero(theta[:,model["biases"]]).item()!=0:
            raise RuntimeError("固定bias发生变化")
        record(theta,initial,loss,hard,state,data,model,cfg,core)
        state["seconds"]=base+time.monotonic()-start
        if time.monotonic()-last_log>=30:
            print(f"{job['id']} step={state['step']} 到达={state['seen'].sum(0).tolist()} 累计{state['seconds']:.1f}s",flush=True)
            last_log=time.monotonic()
        if time.monotonic()-last_save>=plan["checkpoint_seconds"]:
            save(folder,state,theta,loss,hard,generator)
            s.dump(existing,{"status":"running","step":state["step"],"seconds":state["seconds"],"reached":state["seen"].sum(0).tolist()})
            last_save=time.monotonic()
    state["seconds"]=base+time.monotonic()-start
    status="completed" if state["seen"].all() else ("step_limited" if state["step"]>=job["max_steps"] else ("budget_limited" if state["seconds"]>=job["max_seconds"] else "partial"))
    save(folder,state,theta,loss,hard,generator)
    result={"status":status,"job":job,"identity":identity,"step":state["step"],"seconds":state["seconds"],
            "reached":state["seen"].sum(0).tolist(),"planned":plan["models"],"environment":environment,
            "note":"hard_first是优化器首次硬拟合，不是G&C；未到达不补零；粒子与初始化不是同一类独立样本。"}
    s.dump(existing,result)
    if status=="completed":
        checkpoint.unlink(missing_ok=True)
    return result


def main():
    plan=s.read(s.ROOT/"PLAN.json")
    s.verify(plan)
    job=next(j for j in plan["jobs"] if j["id"]==os.environ["BROAD_STP_JOB"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动CPU长跑")
    device=torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(plan["cpu_threads"])
    torch.use_deterministic_algorithms(plan["deterministic"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    with np.load(s.ROOT/plan["datasets"][job["case"]]["file"],allow_pickle=False) as a:
        arrays={k:a[k].copy() for k in ("x","y","probe_x")}
    core=load_core()
    for k,v in arrays.items():
        if core.digest_array(torch.from_numpy(v))!=plan["datasets"][job["case"]]["arrays"][k]:
            raise RuntimeError("训练数据数组身份不符")
    environment={"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),
                 "physical_gpu":plan["physical_gpu"],"dtype":"float32 parameters, float64 BCE","tf32":False,
                 "deterministic":plan["deterministic"],"python":sys.version,"platform":sys.platform}
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    print(f"{s.now()} START {job['id']} GPU={environment['gpu']} 128初始化",flush=True)
    result=run_group(s.ROOT/"results"/job["id"],job,plan,arrays,device,float(os.environ["BROAD_STP_SECONDS"]),stop,environment)
    print(f"{s.now()} END {job['id']} {result['status']} 到达={result['reached']}",flush=True)


if __name__=="__main__":
    main()
