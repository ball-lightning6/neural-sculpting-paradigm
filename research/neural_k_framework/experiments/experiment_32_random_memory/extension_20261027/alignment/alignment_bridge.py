"""在固定BCE父事件内收紧margin极差；只估计真实嵌套事件的条件质量。"""

import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
import frozen_memory as m

DEFAULTS={"targets":[1.,.3,.1],"survival_quantile":.5,"quantile_device":"cpu","adapt_sweeps":2,"sweeps":64,"refresh_sweeps":96,
          "target_acceptance":.3,"adapt_rate":.35,"min_scale":1e-4,"max_scale":.95,
          "raw_tolerance":1e-9,"range_tolerance":1e-7,"max_levels":4000,"max_stalls":64,
          "max_seconds":600.,"log_seconds":30.}


@torch.no_grad()
def neural_scores(theta,x,y,layers,micro_batch=2048):
    losses,ranges=[],[]
    signs=(2*y-1)[None]
    for start in range(0,len(theta),micro_batch):
        logits=m.forward_logits(theta[start:start+micro_batch],x,layers)
        margins=signs*logits
        losses.append(torch.logaddexp(torch.zeros_like(logits),-margins).mean(-1))
        wide=margins.double()
        ranges.append(wide.max(-1).values-wide.min(-1).values)
    return torch.cat(losses),torch.cat(ranges)


@torch.no_grad()
def mutate(state,blocks,scorer,generator,config,threshold,sweeps,adapt=False):
    rates=np.zeros(len(blocks),float)
    for _ in range(sweeps):
        for i,block in enumerate(blocks):
            proposal=m.pcn_proposal(state["theta"],block,state["scales"][i],generator)
            loss,spread=scorer(proposal)
            accept=torch.isfinite(loss)&torch.isfinite(spread)&(loss<=state["parent_epsilon"]+config["raw_tolerance"])&(spread<=threshold+config["range_tolerance"])
            state["theta"][accept]=proposal[accept]
            state["bce"][accept]=loss[accept]
            state["spread"][accept]=spread[accept]
            rate=float(accept.double().mean())
            rates[i]+=rate
            if adapt:
                scale=state["scales"][i]*math.exp(config["adapt_rate"]*(rate-config["target_acceptance"]))
                state["scales"][i]=min(config["max_scale"],max(config["min_scale"],scale))
    return (rates/max(1,sweeps)).tolist()


def save(folder,state,generators):
    payload={k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in state.items()}
    payload["rng"]={k:g.get_state().cpu().clone() for k,g in generators.items()}
    m.atomic_write(folder/"checkpoint.pt",lambda stream:torch.save(payload,stream),binary=True)
    m.atomic_json(folder/"diagnostics.json",{k:v for k,v in state.items() if k not in ("theta","bce","spread","lineages")})


@torch.no_grad()
def run_bridge(folder,parent,blocks,scorer,device,identity,seed,deadline,stop,audit=None,config=None,step_limit=None,progress=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg={**DEFAULTS,**(config or {})}
    identity=m.json_hash({"identity":identity,"bridge_config":cfg,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    status_file=folder/"STATUS.json"
    if status_file.exists():
        import json
        old=json.loads(status_file.read_text(encoding="utf-8"))
        if old["identity"]!=identity:
            raise RuntimeError("桥接结果身份不符")
        if old["status"] in ("completed","budget_limited","stalled","level_limited"):
            return old
    generators={name:torch.Generator(device=device).manual_seed((seed+offset)%(2**63-1)) for name,offset in (("resample",1009),("mutation",2017))}
    checkpoint=folder/"checkpoint.pt"
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if state["identity"]!=identity:
            raise RuntimeError("桥接断点身份不符")
        rng=state.pop("rng")
        for name,g in generators.items():g.set_state(rng[name])
        for key in ("theta","bce","spread","lineages"):state[key]=state[key].to(device)
    else:
        theta=parent["particles"].to(device).clone()
        loss,spread=scorer(theta)
        if not bool(torch.isfinite(loss).all()) or not bool(torch.isfinite(spread).all()) or not bool((loss<=parent["epsilon"]+cfg["raw_tolerance"]).all()):
            raise RuntimeError("父样本不属于指定BCE事件，拒绝用更深或错误父分布冒充")
        torch.testing.assert_close(loss,parent["losses"].to(device),atol=2e-7,rtol=0)
        state={"identity":identity,"theta":theta,"bce":loss,"spread":spread,"lineages":parent["lineages"].to(device).clone(),
               "scales":list(parent["proposal_scales"]),"parent_epsilon":parent["epsilon"],"parent_K_bits":parent["K_bits"],
               "conditional_K_bits":0.,"upper":float("inf"),"level":0,"target_index":0,"stalls":0,"seconds":0.,"nodes":{},"history":[]}
    base=state["seconds"];started=last_log=time.monotonic();first_level=state["level"]
    own_deadline=min(deadline,started+max(0.,cfg["max_seconds"]-base))
    status="partial"
    while state["target_index"]<len(cfg["targets"]):
        if stop[0] or time.monotonic()>=own_deadline:
            break
        if step_limit is not None and state["level"]-first_level>=step_limit:
            break
        if state["level"]>=cfg["max_levels"]:
            status="level_limited";break
        if state["stalls"]>=cfg["max_stalls"]:
            status="stalled";break
        target=cfg["targets"][state["target_index"]]
        adaptive=float(torch.quantile(state["spread"].detach().cpu(),cfg["survival_quantile"]))
        threshold=min(state["upper"],max(target,adaptive))
        survivors=torch.nonzero(state["spread"]<=threshold+cfg["range_tolerance"]).flatten()
        if not len(survivors):raise RuntimeError("桥接无幸存粒子，不伪造质量")
        count=len(state["theta"])
        chosen=survivors[torch.randint(len(survivors),(count,),generator=generators["resample"],device=device)]
        for key in ("theta","bce","spread","lineages"):state[key]=state[key][chosen].clone()
        survival=len(survivors)/count
        state["conditional_K_bits"]-=math.log2(survival)
        mutate(state,blocks,scorer,generators["mutation"],cfg,threshold,cfg["adapt_sweeps"],True)
        rates=mutate(state,blocks,scorer,generators["mutation"],cfg,threshold,cfg["sweeps"])
        state["stalls"]=state["stalls"]+1 if threshold>=state["upper"] else 0
        state["upper"]=threshold;state["level"]+=1
        hit=threshold<=target+1e-12
        if hit:
            rates=mutate(state,blocks,scorer,generators["mutation"],cfg,threshold,cfg["refresh_sweeps"])
            node={"delta":target,"level":state["level"],"parent_epsilon":state["parent_epsilon"],
                  "parent_K_bits":state["parent_K_bits"],"conditional_K_bits":state["conditional_K_bits"],
                  "total_K_bits":state["parent_K_bits"]+state["conditional_K_bits"],
                  "raw_BCE_mean":float(state["bce"].mean()),"raw_BCE_max":float(state["bce"].max()),
                  "margin_range_mean":float(state["spread"].mean()),"margin_range_max":float(state["spread"].max()),
                  "initial_ancestors_remaining":int(state["lineages"].unique().numel()),"acceptance":rates,
                  "warning":"继承父池偏差；桥接不是新的独立先验重复；祖先数和接受率不是混合证明。"}
            if audit is not None:node["audit"]=audit(folder,target,state)
            state["nodes"][format(target,".12g")]=node
            state["target_index"]+=1
        if not bool((state["bce"]<=state["parent_epsilon"]+cfg["raw_tolerance"]).all()) or not bool((state["spread"]<=threshold+cfg["range_tolerance"]).all()):
            raise RuntimeError("桥接约束自检失败")
        state["history"].append({"level":state["level"],"threshold":threshold,"survivors":len(survivors),"particles":count,
                                 "survival_fraction":survival,"conditional_K_bits":state["conditional_K_bits"],"acceptance":rates})
        state["seconds"]=base+time.monotonic()-started
        save(folder,state,generators)
        if hit or time.monotonic()-last_log>=cfg["log_seconds"]:
            if progress is not None:progress(state)
            print(f"BRIDGE level={state['level']} range<={threshold:.6g} 增量K={state['conditional_K_bits']:.3f} 原BCE<={state['parent_epsilon']}",flush=True)
            last_log=time.monotonic()
    state["seconds"]=base+time.monotonic()-started
    if state["target_index"]==len(cfg["targets"]):status="completed"
    elif status=="partial" and state["seconds"]>=cfg["max_seconds"]:status="budget_limited"
    save(folder,state,generators)
    result={"identity":identity,"status":status,"level":state["level"],"seconds":state["seconds"],"nodes":state["nodes"],
            "parent_epsilon":state["parent_epsilon"],"parent_K_bits":state["parent_K_bits"],"conditional_K_bits":state["conditional_K_bits"],
            "note":"未到达delta保留缺失；不是零体积，也不自动视为不可行。"}
    m.atomic_json(status_file,result)
    return result
