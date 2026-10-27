"""同一8→32→32→32→1网络上的五规则full-target SMC校准。"""

import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import frozen_memory as m
from memory_geometry import geometry

from config import WORK_DIR, DEVICE, RUN_HOURS

ROOT=WORK_DIR / "calibration"
RULES=("parity1","majority3","mux3","parity2","parity4")
ACTIVE_RULE=None
AUDIT_FOLDER=None
ORIGINAL_NODE=m.node_diagnostics
ORIGINAL_SLICE=m.run_slice
ORIGINAL_REPORT=m.write_reports


def rule_data(rule,input_bits=8):
    if input_bits!=8:
        raise ValueError("校准保持8-bit输入，不静默更改架构")
    values=np.arange(256,dtype=np.uint32)
    x=((values[:,None]>>np.arange(7,-1,-1))&1).astype(np.float32)
    if rule=="parity1":
        y=x[:,0]
    elif rule=="majority3":
        y=x[:,:3].sum(1)>=2
    elif rule=="mux3":
        y=np.where(x[:,0]==1,x[:,1],x[:,2])
    elif rule=="parity2":
        y=x[:,:2].sum(1).astype(np.int64)%2
    elif rule=="parity4":
        y=x[:,:4].sum(1).astype(np.int64)%2
    else:
        raise ValueError("未知规则")
    return x,y.astype(np.uint8),np.arange(256,dtype=np.int64)


def calibration_data(dataset_id,config):
    if dataset_id!=0 or ACTIVE_RULE not in RULES:
        raise ValueError("每个规则独立目录、一个完整标签表，不把不同规则混成随机重复")
    return rule_data(ACTIVE_RULE,config["INPUT_BITS"])


def calibration_slice(job,config,manifest,device,environment_identity,deadline,stop,progress):
    global AUDIT_FOLDER
    AUDIT_FOLDER=config["OUTPUT_DIR"]/job["key"]/"node_audits"
    def progress_with_stop():
        if (ROOT/"STOP").exists():
            stop.requested,stop.reason=True,"STOP_file"
        progress()
    return ORIGINAL_SLICE(job,config,manifest,device,environment_identity,deadline,stop,progress_with_stop)


def calibration_node(state,epsilon,x,y,layers,config,rates,change):
    node=ORIGINAL_NODE(state,epsilon,x,y,layers,config,rates,change)
    node["implementation_geometry"]=geometry(state.particles,x,y,layers)
    count=min(8,len(state.particles))
    theta=state.particles[:count].detach().cpu().clone()
    inputs=x.detach().cpu().double()
    labels=y.detach().cpu().double()
    with torch.no_grad():
        logits=m.forward_logits(theta.double(),inputs,layers)
        losses=torch.nn.functional.softplus(logits*(1-2*labels)).mean(1)
    cached=state.losses[:count].detach().cpu().numpy().copy()
    path=AUDIT_FOLDER/("bce_"+m.node_key(epsilon).replace(".","p")+".npz")
    m.atomic_npz(path,theta_standardized=theta.numpy(),inputs=inputs.numpy(),labels=labels.numpy(),
                 logits_float64=logits.numpy(),BCE_float64=losses.numpy(),BCE_cached=cached,
                 predicted_bits=(logits.numpy()>=0).astype(np.uint8),layers=np.array(layers,dtype=np.int64),
                 support_threshold=np.array(epsilon,dtype=np.float64))
    node["audit"]={"file":path.relative_to(config["OUTPUT_DIR"]).as_posix(),
                   "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"particles":count,
                   "max_float64_cached_loss_difference":float(np.max(np.abs(losses.numpy()-cached))),
                   "float64_feasible_with_2e_minus7_tolerance":bool((losses<=epsilon+2e-7).all()),
                   "scope":"前8粒子的只读审计，不是独立ESS或全局混合证明。"}
    return node


def calibration_report(manifest,config,reason="running",error=None):
    result=ORIGINAL_REPORT(manifest,config,reason,error)
    result.update(target_rule=ACTIVE_RULE,measurement="full_target_256_points",reference_measure="Gaussian_neural_coordinates")
    summary=json.loads((config["OUTPUT_DIR"]/"summary.json").read_text(encoding="utf-8"))
    summary.update(target_rule=ACTIVE_RULE,measurement="full_target_256_points",dataset_semantics="一个固定规则的完整真值表；不是五规则的平均")
    m.atomic_json(config["OUTPUT_DIR"]/"summary.json",summary)
    lines=[f"# 三层网络规则校准 {ACTIVE_RULE}","",f"状态：{result['status']}；原因：{reason}。",
           f"完成池：{result['completed_pools']}/4；失败池：{result['failed_pools']}。",
           "完整256点上的目标BCE事件，不额外交hard-function cell。", "",
           "| 池 | 状态 | 层数 | 最近BCE阈值 |", "|---:|---|---:|---:|"]
    for j in manifest["jobs"]:
        lines.append(f"| {j['pool']} | {j['status']} | {j['level']} | {j.get('reached_bce')} |")
    m.atomic_text(config["OUTPUT_DIR"]/"PROGRESS.md","\n".join(lines)+"\n")
    m.atomic_json(config["OUTPUT_DIR"]/"STATUS.json",result)
    if config["OUTPUT_DIR"].resolve().parent==(ROOT/"results").resolve():
        total=summarize()
        path=ROOT/"STATUS.json"
        state=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        state.update(status="running",active_rule=ACTIVE_RULE,completed_pools=total["completed_pools"],failed_pools=total["failed_pools"],total_pools=20)
        m.atomic_json(path,state)
    return result


def configure(rule,output,device,seconds):
    index=RULES.index(rule)
    m.PROTOCOL="tanh3_full_rule_calibration_v1:"+rule
    return {"OUTPUT_DIR":output,"DEVICE":device,"INPUT_BITS":8,"ARCHITECTURES":((32,32,32),),
            "COUNTS":(256,),"DATASETS":1,"POOLS":4,"PARTICLES":4096,"ADAPT_SWEEPS":2,"MUTATION_SWEEPS":48,
            "THRESHOLDS":(*m.CONFIG["THRESHOLDS"],.0005,.0002,.0001),"PRIMARY_WINDOW":(.002,.001),
            "PRIOR_SEED":m.CONFIG["PRIOR_SEED"]+20_000_000+index*1_000_003,
            "RESAMPLE_SEED":m.CONFIG["RESAMPLE_SEED"]+20_000_000+index*1_000_003,
            "MUTATION_SEED":m.CONFIG["MUTATION_SEED"]+20_000_000+index*1_000_003,
            "TOTAL_SECONDS":seconds,"SECONDS_PER_REPLICA":900.,"KEEP_COMPLETED_CHECKPOINTS":True}


def pool_mass_K(values):
    a=np.array(values,dtype=np.float64)
    return float(a.min()-np.log2(np.exp2(-(a-a.min())).mean()))


def summarize(root=ROOT):
    rows=[]
    windows=((.01,.005),(.005,.002),(.002,.001),(.001,.0005),(.0005,.0002),(.0002,.0001))
    for rule in RULES:
        path=root/"results"/rule/"manifest.json"
        if not path.exists():
            rows.append({"rule":rule,"status":"pending","completed_pools":0,"failed_pools":0,"nodes":[],"windows":[]})
            continue
        manifest=json.loads(path.read_text(encoding="utf-8"))
        jobs=manifest["jobs"]
        completed=sum(j["status"]=="completed" for j in jobs)
        failed=sum(j["status"]=="failed" for j in jobs)
        nodes=[]
        for epsilon in manifest["science"]["config"]["THRESHOLDS"]:
            values=[j["nodes"].get(m.node_key(epsilon),{}).get("K_bits") for j in jobs]
            full=all(v is not None for v in values)
            nodes.append({"epsilon":epsilon,"K_by_pool":values,"all_pools_reached":full,
                          "K_pooled_mass":pool_mass_K(values) if full else None,
                          "K_median":float(np.median(values)) if full else None})
        rates=[]
        for hi,lo in windows:
            values=[m.pool_window(j["nodes"],hi,lo) for j in jobs]
            full=all(v is not None for v in values)
            rates.append({"epsilon_high":hi,"epsilon_low":lo,"c_by_pool":values,
                          "c_median":float(np.median(values)) if full else None})
        rows.append({"rule":rule,"status":"completed" if completed==4 else ("completed_with_failures" if completed+failed==4 else "partial"),
                     "completed_pools":completed,"failed_pools":failed,"nodes":nodes,"windows":rates})
    out={"rows":rows,"total_pools":20,"completed_pools":sum(r["completed_pools"] for r in rows),
         "failed_pools":sum(r["failed_pools"] for r in rows),"hard_exact_boundary":math.log(2)/256,
         "scope":"五条固定规则分别估计full-target质量；不把不同规则当随机重复，不与部分训练集体积直接等同。"}
    m.atomic_json(root/"CALIBRATION_SUMMARY.json",out)
    lines=["# 三层网络熟悉规则校准","",f"完成{out['completed_pools']}/20池；失败{out['failed_pools']}池。",
           "相同8→32→32→32→1 tanh及高斯参数化，使用完整256输入。","",
           "| 规则 | 状态 | 完成池 | K(0.001)，先均质量后取-log2 | c(0.002至0.001)，池中位数 |",
           "|---|---|---:|---:|---:|"]
    for row in rows:
        node=next((v for v in row["nodes"] if v["epsilon"]==.001),{})
        window=next((v for v in row["windows"] if v["epsilon_high"]==.002 and v["epsilon_low"]==.001),{})
        lines.append(f"| {row['rule']} | {row['status']} | {row['completed_pools']}/4 | {node.get('K_pooled_mass')} | {window.get('c_median')} |")
    lines += ["","未达到节点保留空值，不补零、不外推。大体排序是待检验对象，不因局部换序删除规则。",
              "BCE小于ln(2)/256约0.002708才保证完整标签符号一致；更高阈值的full-target事件可以重叠。"]
    m.atomic_text(root/"PROGRESS.md","\n".join(lines)+"\n")
    return out


def verify_bundle():
    from review import verify_sources
    return verify_sources()


def run(settings):
    global ACTIVE_RULE
    manifest=verify_bundle()
    m.make_data=calibration_data
    m.node_diagnostics=calibration_node
    m.run_slice=calibration_slice
    m.write_reports=calibration_report
    m.PROVENANCE={**m.PROVENANCE,"calibration_bundle":manifest,"definition":"完整256点目标BCE事件，无额外hard cell交集"}
    deadline=time.monotonic()+settings["run_hours"]*3600
    lock=ROOT/"RUNNING.lock"
    with lock.open("x",encoding="utf-8") as stream:
        json.dump({"pid":__import__('os').getpid()},stream)
    state={"status":"running","source_queue":settings["source_queue"],"active_rule":None}
    outer_stop=[False]
    handlers={sig:signal.signal(sig,lambda *_:outer_stop.__setitem__(0,True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        while time.monotonic()<deadline:
            summary=summarize()
            pending=[r["rule"] for r in summary["rows"] if r["status"] not in ("completed","completed_with_failures")]
            if not pending:
                state["status"]="completed_with_unresolved" if summary["failed_pools"] else "completed"
                break
            if outer_stop[0] or (ROOT/"STOP").exists():
                state["status"]="stopped"
                break
            for rule in pending:
                if time.monotonic()>=deadline or outer_stop[0] or (ROOT/"STOP").exists():
                    break
                ACTIVE_RULE=rule
                state.update(active_rule=rule,status="running")
                m.atomic_json(ROOT/"STATUS.json",state)
                config=configure(rule,ROOT/"results"/rule,settings["device"],max(0.,deadline-time.monotonic()))
                print(f"规则={rule}，4个独立池，完整256输入，剩余{(deadline-time.monotonic())/3600:.2f}小时",flush=True)
                result=m.run(config)
                summarize()
                reason=str(result.get("reason",""))
                if "signal" in reason.lower() or reason in ("keyboard_interrupt","SIGINT","SIGTERM","STOP_file"):
                    state["status"]="stopped"
                    return
        if state["status"]=="running":
            state["status"]="stopped" if outer_stop[0] or (ROOT/"STOP").exists() else "budget_stopped"
    except BaseException as exc:
        state.update(status="needs_attention",error=repr(exc))
        raise
    finally:
        summary=summarize()
        state.update(active_rule=None,completed_pools=summary["completed_pools"],failed_pools=summary["failed_pools"],total_pools=20)
        m.atomic_json(ROOT/"STATUS.json",state)
        lock.unlink(missing_ok=True)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择 calibration。")
