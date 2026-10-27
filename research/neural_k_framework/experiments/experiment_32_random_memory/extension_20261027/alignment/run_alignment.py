"""三层记忆等幅小试：局部BCE父池与固定极差条件桥接分别记账。"""

from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
import frozen_memory as m
from memory_geometry import geometry
from alignment_bridge import run_bridge,neural_scores,DEFAULTS
from config import WORK_DIR, EVIDENCE, DEVICE, RUN_HOURS
from data import random_table

ROOT=WORK_DIR / "alignment"
CASES=[{"name":f"{kind}_n{n}","kind":kind,"n":n} for n in (32,80,128) for kind in ("random","projection")]
EPSILONS=(.002,.001)
PARENT_SECONDS_PER_CASE=1200.
ACTIVE_CASE=None
CURRENT_FOLDER=None
CURRENT_SIGNATURE=None
CURRENT_ENV=None
ORIGINAL_NODE=m.node_diagnostics
ORIGINAL_SLICE=m.run_slice
ORIGINAL_REPORT=m.write_reports


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_data(kind):
    x,y,order=random_table(0)
    if kind=="projection":y=x[:,0].astype(np.uint8)
    elif kind!="random":raise ValueError("未知标签类型")
    return x,y,order


def parent_data(dataset_id,config):
    if dataset_id!=0 or config["INPUT_BITS"]!=8 or ACTIVE_CASE is None:
        raise ValueError("父事件数据身份错误")
    return load_data(ACTIVE_CASE["kind"])


def snapshot_name(eps):
    return "parent_snapshot_"+m.node_key(eps).replace(".","p")+".pt"


def audit_arrays(path,theta,x,y,layers,metadata):
    selected=theta[:8].detach().cpu().clone()
    inputs=x.detach().cpu().double();labels=y.detach().cpu().double()
    with torch.no_grad():
        logits=m.forward_logits(selected.double(),inputs,layers)
        margins=logits*(2*labels-1)
        losses=torch.nn.functional.softplus(-margins).mean(1)
        spread=margins.max(1).values-margins.min(1).values
    m.atomic_npz(path,theta_standardized=selected.numpy(),inputs=inputs.numpy(),labels=labels.numpy(),
                 logits_float64=logits.numpy(),BCE_float64=losses.numpy(),margin_range_float64=spread.numpy(),
                 layers=np.array(layers,dtype=np.int64))
    return {**metadata,"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"particles":len(selected),
            "BCE_float64_max":float(losses.max()),"margin_range_float64_max":float(spread.max()),
            "scope":"前8粒子可行性审计，不能当作独立样本或混合证明。"}


def parent_node(state,epsilon,x,y,layers,config,rates,change):
    node=ORIGINAL_NODE(state,epsilon,x,y,layers,config,rates,change)
    node["implementation_geometry"]=geometry(state.particles,x,y,layers)
    if epsilon in EPSILONS:
        snapshot={"version":1,"particles":state.particles.detach().cpu().clone(),"losses":state.losses.detach().cpu().clone(),
                  "lineages":state.lineages.detach().cpu().clone(),"proposal_scales":list(state.proposal_scales),
                  "epsilon":epsilon,"K_bits":-state.log_volume_fraction/math.log(2),"level":state.level,
                  "case":ACTIVE_CASE,"signature":CURRENT_SIGNATURE,"environment_identity":CURRENT_ENV}
        path=CURRENT_FOLDER/snapshot_name(epsilon)
        m.atomic_write(path,lambda stream:torch.save(snapshot,stream),binary=True)
        meta={"snapshot":path.name,"snapshot_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "epsilon":epsilon,"K_bits":snapshot["K_bits"],"level":state.level,"case":ACTIVE_CASE,"signature":CURRENT_SIGNATURE}
        meta["audit"]=audit_arrays(CURRENT_FOLDER/("parent_audit_"+m.node_key(epsilon).replace(".","p")+".npz"),state.particles,x,y,layers,{})
        m.atomic_json(CURRENT_FOLDER/("parent_"+m.node_key(epsilon).replace(".","p")+".json"),meta)
        node["parent_snapshot"]=meta
    return node


def parent_slice(job,config,manifest,device,environment_identity,deadline,stop,progress):
    global CURRENT_FOLDER,CURRENT_SIGNATURE,CURRENT_ENV
    CURRENT_FOLDER=config["OUTPUT_DIR"]/job["key"]
    CURRENT_SIGNATURE=m.job_signature(job,manifest)
    CURRENT_ENV=environment_identity
    def progress_checked():
        if (ROOT/"STOP").exists():stop.requested,stop.reason=True,"STOP_file"
        progress()
    return ORIGINAL_SLICE(job,config,manifest,device,environment_identity,deadline,stop,progress_checked)


def parent_report(manifest,config,reason="running",error=None):
    result=ORIGINAL_REPORT(manifest,config,reason,error)
    result["case"]=ACTIVE_CASE
    m.atomic_json(config["OUTPUT_DIR"]/"STATUS.json",result)
    text=["# 对齐小试的原BCE父池","",f"条件：{ACTIVE_CASE['name']}；完成{result['completed_pools']}/2池。",
          "0.002和0.001节点分别保存完整父粒子，不使用更深节点冒充较高阈值父分布。"]
    m.atomic_text(config["OUTPUT_DIR"]/"PROGRESS.md","\n".join(text)+"\n")
    return result


def parent_config(case,device,seconds):
    m.PROTOCOL="aligned_memory_parent_v1:"+case["kind"]
    shift=30_000_000+(0 if case["kind"]=="random" else 1_000_003)
    return {"OUTPUT_DIR":ROOT/"results"/case["name"]/"parent","DEVICE":device,"INPUT_BITS":8,"ARCHITECTURES":((32,32,32),),
            "COUNTS":(case["n"],),"DATASETS":1,"POOLS":2,"PARTICLES":2048,"ADAPT_SWEEPS":2,"MUTATION_SWEEPS":48,
            "PRIMARY_WINDOW":EPSILONS,"TOTAL_SECONDS":seconds,"SECONDS_PER_REPLICA":600.,"KEEP_COMPLETED_CHECKPOINTS":True,
            "PRIOR_SEED":m.CONFIG["PRIOR_SEED"]+shift,"RESAMPLE_SEED":m.CONFIG["RESAMPLE_SEED"]+shift,
            "MUTATION_SEED":m.CONFIG["MUTATION_SEED"]+shift}


def parent_folder(case,pool):
    return ROOT/"results"/case["name"]/"parent"/f"h32_32_32/data_000/n{case['n']:03d}/pool_{pool}"


def summarize():
    rows=[];parents_complete=bridges_complete=bridges_terminal=0
    high,low=EPSILONS
    terminal=("completed","budget_limited","stalled","level_limited")
    for case in CASES:
        parent_manifest=ROOT/"results"/case["name"]/"parent/manifest.json"
        if parent_manifest.exists():
            data=json.loads(parent_manifest.read_text(encoding="utf-8"))
            parents_complete+=sum(j["status"]=="completed" for j in data["jobs"])
        pool_rows=[]
        for pool in range(2):
            values={}
            for eps in EPSILONS:
                key=m.node_key(eps).replace(".","p")
                meta=parent_folder(case,pool)/("parent_"+key+".json")
                parent=json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
                status_path=ROOT/"results"/case["name"]/"bridges"/f"pool_{pool}"/key/"STATUS.json"
                status=json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
                if status:
                    bridges_complete+=int(status["status"]=="completed")
                    bridges_terminal+=int(status["status"] in terminal)
                values[eps]={"parent":parent,"bridge":status}
            raw=None
            if all(values[e]["parent"] for e in EPSILONS):
                raw=(values[low]["parent"]["K_bits"]-values[high]["parent"]["K_bits"])/math.log2(high/low)
            aligned={}
            for delta in DEFAULTS["targets"]:
                node=[values[e]["bridge"]["nodes"].get(m.node_key(delta)) if values[e]["bridge"] else None for e in EPSILONS]
                full=all(v is not None for v in node)
                c=(node[1]["total_K_bits"]-node[0]["total_K_bits"])/math.log2(high/low) if full else None
                aligned[str(delta)]={"c":c,"additional_c":c-raw if c is not None and raw is not None else None,
                                      "conditional_K_by_epsilon":[v["conditional_K_bits"] if v else None for v in node]}
            pool_rows.append({"pool":pool,"raw_c":raw,"aligned":aligned})
        raw_values=[r["raw_c"] for r in pool_rows]
        aligned_summary={}
        for delta in DEFAULTS["targets"]:
            values=[r["aligned"][str(delta)]["c"] for r in pool_rows]
            aligned_summary[str(delta)]={"by_pool":values,"median":float(np.median(values)) if all(v is not None for v in values) else None}
        rows.append({**case,"pools":pool_rows,"raw_c_median":float(np.median(raw_values)) if all(v is not None for v in raw_values) else None,
                     "aligned_c":aligned_summary})
    result={"rows":rows,"parent_pools_complete":parents_complete,"parent_pools_total":2*len(CASES),"bridges_complete":bridges_complete,
            "bridges_terminal":bridges_terminal,"bridges_total":2*len(CASES)*len(EPSILONS),"primary_delta":DEFAULTS["targets"][-1],
            "scope":"只有一份随机表与三个锚点、每条件两父池；桥接继承父误差，不证明普遍正比。"}
    m.atomic_json(ROOT/"ALIGNMENT_SUMMARY.json",result)
    lines=["# 三层记忆幅度对齐小试","",f"原BCE父池完成{parents_complete}/{result['parent_pools_total']}；条件桥接完成{bridges_complete}/{result['bridges_total']}，已有终态{bridges_terminal}/{result['bridges_total']}。",
           "margin极差固定为0.1是主判据，1.0和0.3为预置过程节点。不混合BCE与对齐分数。","",
           "| 条件 | 原BCE c | 对齐delta=1.0 | 对齐delta=0.3 | 对齐delta=0.1 |","|---|---:|---:|---:|---:|"]
    for row in rows:
        v=[row["aligned_c"][str(d)]["median"] for d in DEFAULTS["targets"]]
        lines.append(f"| {row['name']} | {row['raw_c_median']} | {v[0]} | {v[1]} | {v[2]} |")
    lines += ["","空值表示尚未完成或未到达，不是零体积。额外条件成本始终非负，但两个epsilon下额外成本之差可以为负。",
              "复制bit对照用于识别通用几何成本；不以恢复直线为成功标准。"]
    m.atomic_text(ROOT/"PROGRESS.md","\n".join(lines)+"\n")
    return result


def main():
    global ACTIVE_CASE
    settings={"device":DEVICE,"run_hours":RUN_HOURS}
    expected=json.loads((EVIDENCE/"datasets.json").read_text(encoding="utf-8"))["0"]
    x0,y0,o0=load_data("random")
    if m.array_digest(x0,y0,o0)!=expected["raw_dataset_sha256"]:raise RuntimeError("原随机表hash不符")
    for n in (32,80,128):
        if m.array_digest(x0[o0[:n]],y0[o0[:n]],o0[:n])!=expected["prefix_sha256"][str(n)]:raise RuntimeError("原数据前缀hash不符")
    m.make_data=parent_data;m.node_diagnostics=parent_node;m.run_slice=parent_slice;m.write_reports=parent_report
    stop=[False]
    handlers={sig:signal.signal(sig,lambda *_:stop.__setitem__(0,True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    deadline=time.monotonic()+settings["run_hours"]*3600
    status={"status":"running","active_case":None,"active_bridge":None}
    lock=ROOT/"RUNNING.lock"
    with lock.open("x",encoding="utf-8") as f:json.dump({"pid":__import__('os').getpid()},f)
    try:
        for case in CASES:
            if stop[0] or (ROOT/"STOP").exists() or deadline-time.monotonic()<10:break
            ACTIVE_CASE=case
            status.update(active_case=case["name"],active_bridge=None,updated=now())
            m.atomic_json(ROOT/"STATUS.json",status)
            cfg=parent_config(case,settings["device"],max(0.,deadline-time.monotonic()))
            parent_path=cfg["OUTPUT_DIR"]
            while deadline-time.monotonic()>10 and not stop[0]:
                manifest=json.loads((parent_path/"manifest.json").read_text(encoding="utf-8")) if (parent_path/"manifest.json").exists() else None
                if manifest and all(j["status"] in ("completed","failed") for j in manifest["jobs"]):break
                spent=sum(call.get("elapsed_seconds",0.) for call in manifest["calls"]) if manifest else 0.
                if spent>=PARENT_SECONDS_PER_CASE:break
                cfg["TOTAL_SECONDS"]=min(deadline-time.monotonic(),PARENT_SECONDS_PER_CASE-spent)
                print(f"PARENT {case['name']} 两池，局部基线，不重做原八档扫描。",flush=True)
                result=m.run(cfg)
                reason=str(result.get("reason",""))
                if "signal" in reason.lower() or reason in ("keyboard_interrupt","STOP_file"):
                    stop[0]=True;break
                summarize()
            if stop[0] or (ROOT/"STOP").exists():break
            x,y,order=load_data(case["kind"])
            selected=order[:case["n"]]
            environment_cfg=m.resolve_config({**cfg,"TOTAL_SECONDS":0.})
            with m.execution_environment(environment_cfg) as (device,environment,env_identity):
                inputs=torch.tensor(x[selected],device=device,dtype=torch.float32)
                labels=torch.tensor(y[selected],device=device,dtype=torch.float32)
                blocks,layers,_=m.parameter_blocks((32,32,32),8)
                for pool in range(2):
                    for eps in EPSILONS:
                        if stop[0] or (ROOT/"STOP").exists() or deadline-time.monotonic()<10:break
                        source=parent_folder(case,pool)/snapshot_name(eps)
                        if not source.exists():continue
                        parent=torch.load(source,map_location="cpu",weights_only=True)
                        if parent["epsilon"]!=eps or parent["case"]!=case:raise RuntimeError("父事件不符")
                        if parent["environment_identity"]!=env_identity:raise RuntimeError("父采样与桥接环境不一致，不静默迁移")
                        key=m.node_key(eps).replace(".","p")
                        folder=ROOT/"results"/case["name"]/"bridges"/f"pool_{pool}"/key
                        status.update(active_bridge=f"pool{pool}/BCE{eps}",updated=now());m.atomic_json(ROOT/"STATUS.json",status)
                        def scorer(theta):return neural_scores(theta,inputs,labels,layers)
                        def audit(path,delta,state):
                            return audit_arrays(path/("audit_delta_"+str(delta).replace(".","p")+".npz"),state["theta"],inputs,labels,layers,{"delta":delta,"parent_epsilon":eps})
                        def refresh_progress(bridge_state):
                            if (ROOT/"STOP").exists():stop[0]=True
                            status.update(bridge_level=bridge_state["level"],bridge_range_upper=bridge_state["upper"],updated=now())
                            m.atomic_json(ROOT/"STATUS.json",status)
                        identity={"case":case,"pool":pool,"epsilon":eps,"parent_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                                  "environment":environment,"definition":"mean_BCE<=epsilon AND range(signed_margin)<=delta"}
                        seed=int(m.json_hash([case,pool,eps,"alignment_bridge_20261025"])[:15],16)
                        print(f"BRIDGE {case['name']} pool={pool} 固定BCE<={eps}，主极差目标0.1",flush=True)
                        run_bridge(folder,parent,blocks,scorer,device,identity,seed,deadline,stop,audit,progress=refresh_progress)
                        summarize()
            if stop[0]:break
        summary=summarize()
        if stop[0] or (ROOT/"STOP").exists():status["status"]="stopped"
        elif summary["parent_pools_complete"]==summary["parent_pools_total"] and summary["bridges_complete"]==summary["bridges_total"]:status["status"]="completed"
        elif time.monotonic()>=deadline-10:status["status"]="budget_stopped"
        else:status["status"]="completed_with_limits"
    except BaseException as exc:
        status.update(status="needs_attention",error=repr(exc));raise
    finally:
        summary=summarize()
        status.update(active_case=None,active_bridge=None,updated=now(),parent_pools_complete=summary["parent_pools_complete"],bridges_complete=summary["bridges_complete"])
        m.atomic_json(ROOT/"STATUS.json",status)
        lock.unlink(missing_ok=True)
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
