"""先验证冻结的外部证书，再读取已有Neural K作回顾性对照。"""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exact_circuits import LANGUAGES, atom_tables, apply_gate, check_lrat, digest, replay_witness


HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
OLD=HERE/"data"


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def joint_upper(first,second):
    """仅合并真值表完全相同的已有信号；返回可独立重放的共享实现上界。"""
    if first["language"]!=second["language"] or first["n"]!=second["n"]:
        raise ValueError("语言或维数不同")
    language,n=first["language"],first["n"]
    values,names=atom_tables(n,LANGUAGES[language]["free_input_negation"])
    mask=(1<<(1<<n))-1
    gates,outputs=[],[]
    for witness in (first,second):
        mapping={i:i for i in range(len(values)-len(gates))}
        atoms_count=len(names)
        for index,gate in enumerate(witness["gates"]):
            a,b=mapping[gate["a"]],mapping[gate["b"]]
            value=apply_gate(gate["op"],values[a],values[b],mask)
            if value in values:
                mapped=values.index(value)
            else:
                mapped=len(values)
                values.append(value)
                gates.append({"op":gate["op"],"a":a,"b":b})
            mapping[atoms_count+index]=mapped
        outputs.extend({"signal":mapping[o["signal"]],"invert":o["invert"]} for o in witness["outputs"])
    result={"n":n,"language":language,"atoms":names,"gates":gates,"outputs":outputs,
            "targets":first["targets"]+second["targets"]}
    replay_witness(result)
    return result


def main():
    frozen=json.loads((OUT/"PANEL_FROZEN.json").read_text(encoding="utf-8"))
    summary=json.loads((OUT/"SUMMARY.json").read_text(encoding="utf-8"))
    if digest(OUT/"PANEL_FROZEN.json")!=summary["panel_sha256"]:
        raise RuntimeError("面板hash变化")
    for name,key in (("run_synthesis.py","source_sha256"),("exact_circuits.py","core_sha256")):
        if digest(HERE/name)!=frozen[key]:
            raise RuntimeError("合成源码变化")
    records=summary["records"]
    expected=len(frozen["panel"])*len(frozen["config"]["languages"])
    if len(records)!=expected:
        raise RuntimeError(f"矩阵尚未结束：{len(records)}/{expected}")
    proof_count=0
    proof_steps=0
    witness_count=0
    index={}
    for record in records:
        language,target=record["language"],record["target"]["name"]
        index[language,target]=record
        for attempt in record["attempted"]:
            folder=OUT/language/target/f"gates_{attempt['budget']:02d}"
            if digest(folder/"problem.cnf")!=attempt["cnf_sha256"] or digest(folder/"result.json")!=attempt["result_sha256"]:
                raise RuntimeError("证书文件发生变化")
            if attempt["status"]=="unsat_verified":
                check=check_lrat(folder/"problem.cnf",folder/"proof.lrat")
                proof_count+=1
                proof_steps+=check["additions"]
        if record["status"]=="exact_verified":
            replay_witness(record["witness"])
            witness_count+=1
            if len(record["attempted"])!=record["cost"]+1 or any(a["status"]!="unsat_verified" for a in record["attempted"][:-1]):
                raise RuntimeError("未完整证明更小预算不可行")
    certificate_freeze={"panel_sha256":digest(OUT/"PANEL_FROZEN.json"),"summary_sha256":digest(OUT/"SUMMARY.json"),
                        "proofs_verified":proof_count,"rup_steps_verified":proof_steps,"witnesses_verified":witness_count}
    (OUT/"CERTIFICATES_FROZEN.json").write_text(json.dumps(certificate_freeze,ensure_ascii=False,indent=2),encoding="utf-8")
    # 到这里才读取旧神经测量。旧3-bit已知，不声称这一对照为新盲测。
    rows=read_csv(OLD/"rank_trajectories.csv")
    curves=defaultdict(dict)
    for r in rows:
        if r["stage"]=="neural_k":
            curves[r["architecture"],float(r["threshold"])][int(r["function_id"])]=float(r["complexity_bits"])
    architectures=sorted({a for a,t in curves})
    correlations=[]
    for language in LANGUAGES:
        chosen=[r for r in records if r["language"]==language and r["target"]["n"]==3 and r["status"]=="exact_verified"]
        for a in architectures:
            for name,calculate in (
                ("K_0p001",lambda f:curves[a,.001][f]),
                ("slope_0p01_to_0p001",lambda f:(curves[a,.001][f]-curves[a,.01][f])/math.log2(10)),
                ("slope_0p0025_to_0p001",lambda f:(curves[a,.001][f]-curves[a,.0025][f])/math.log2(2.5)),
            ):
                x=[r["cost"] for r in chosen]
                y=[calculate(r["target"]["targets"][0]) for r in chosen]
                rho=float(spearmanr(x,y).statistic) if len(set(x))>1 and len(set(y))>1 else None
                correlations.append({"language":language,"architecture":a,"measurement":name,"n_orbits":len(x),"spearman":rho})
    sharing=[]
    for language in LANGUAGES:
        left=index[language,"rule4_shared_left"]
        right=index[language,"rule4_shared_right"]
        other=index[language,"rule4_independent_right"]
        for name,a,b in (("joint4_shared",left,right),("joint4_independent",left,other)):
            r=index[language,name]
            upper=joint_upper(a["witness"],b["witness"])
            lower=r.get("cost",r.get("lower_bound"))
            bound=r.get("cost",len(upper["gates"]))
            if lower>bound:
                raise RuntimeError("上下界冲突")
            sharing.append({"language":language,"task":name,"lower_bound":lower,"upper_bound":bound,
                            "separate_sum":a["cost"]+b["cost"],
                            "upper_witness":r["witness"] if r["status"]=="exact_verified" else upper,"status":r["status"]})
    # 4-bit证书对应的规则不依赖第4位；常数免费时限制该位给出同一3-bit门数。
    atoms,_=atom_tables(3)
    a,b,c=atoms[2:]
    examples={"and2":a&b,"and2_or3":(a&b)|c}
    user_pair={name:{"function_id":fid,"K_0p001":{arch:curves[arch,.001][fid] for arch in architectures},
                    "logic_cost":{lang:index[lang,"rule4_"+name]["cost"] for lang in LANGUAGES}}
               for name,fid in examples.items()}
    report={"certificates":certificate_freeze,"record_count":len(records),"exact_count":witness_count,
            "unknown_count":len(records)-witness_count,"correlations":correlations,"sharing":sharing,"user_pair":user_pair,
            "old_neural_source_sha256":digest(OLD/"rank_trajectories.csv"),
            "scope":"独立逻辑求解+既有神经数据回顾性校准；未新增神经采样，未传播SMC误差。"}
    (OUT/"ANALYSIS.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# 外部复杂度证书与Neural K对照", "", report["scope"], "",
           f"完成{len(records)}项固定目标/语言组合，其中{witness_count}项获得最小门数的SAT见证及全部更低预算UNSAT证明，{len(records)-witness_count}项保留界。",
           f"独立检查器重放{proof_count}份LRAT，共{proof_steps}条RUP新增步骤。", "",
           "## 3-bit最小共享DAG", "", "三个语言均允许免费输出反相；NOT是否对输入收费不同。", "",
           "|函数|AON输入反相计费|含XOR|AON输入literal反相免费|", "|---|---:|---:|---:|"]
    for target in frozen["panel"]:
        if target["n"]!=3:
            continue
        cells=[index[lang,target["name"]] for lang in LANGUAGES]
        lines.append("|"+"|".join([f"0x{target['targets'][0]:02X}"]+[str(r["cost"]) if r["status"]=="exact_verified" else f">={r['lower_bound']} 未决" for r in cells])+"|")
    lines += ["", "## 与已有Neural K的关联", "", "每种架构只有10个平衡轨道，相关性是描述性结果，不作新的显著性确认。", "",
              "|门库|K(0.001) Spearman范围|窗口0.01到0.001|窗口0.0025到0.001|", "|---|---:|---:|---:|"]
    for lang in LANGUAGES:
        cells=[]
        for m in ("K_0p001","slope_0p01_to_0p001","slope_0p0025_to_0p001"):
            vals=[r["spearman"] for r in correlations if r["language"]==lang and r["measurement"]==m and r["spearman"] is not None]
            cells.append(f"{min(vals):.3f}至{max(vals):.3f}" if vals else "未决")
        lines.append("|"+"|".join([lang]+cells)+"|")
    lines += ["", "## 用户提出的严格逻辑对", "", "AND2为1门，AND2再OR第三位为2门，三种门库都通过了最小性验证。", "",
              "已有3-bit同网络下的K如下（新4-bit逻辑证书不等于新增4-bit神经测量）：", "",
              "|架构|AND2 K|AND2 OR第三位 K|", "|---|---:|---:|"]
    for arch in architectures:
        lines.append(f"|{arch}|{user_pair['and2']['K_0p001'][arch]:.4f}|{user_pair['and2_or3']['K_0p001'][arch]:.4f}|")
    lines += ["", "该对输出平衡度不同，不能单独排除标签比例或几何因素。", "", "## 多输出共享", "",
              "shared为((a XOR b) AND c, (a XOR b) AND d)，对照为((a XOR b) AND c, (a XOR d) AND b)。后者只是不同前缀，不是已证明算法独立。", "",
              "|门库|任务|联合成本界|分开实现最小成本之和|", "|---|---|---:|---:|"]
    for r in sharing:
        lines.append(f"|{r['language']}|{r['task']}|[{r['lower_bound']},{r['upper_bound']}]|{r['separate_sum']}|")
    lines += ["", "共享程序的逻辑收益有证书，但尚未据此宣称Neural K也取得同等收益；新多输出SMC仍是待确认的独立命题。",
              "", "## 边界与可复核性", "",
              "- 所有门库预先固定并全部报告；含XOR的分歧不删除。",
              "- 较低预算UNSAT不是求解器一句话：其正提示LRAT已由独立Python检查器重放。该检查器本身经过单元测试，但不是形式化验证过的证明内核；CNF编码的语义正确性另由小世界枚举测试支持。",
              "- 任何超时均保留unknown和已获界，没有把超时记作无穷复杂。",
              "- 原证据与原SMC不变；本轮未验证旧SMC的全局混合，不把10个轨道当作数百独立规则。",
              "- 原始CNF、LRAT、SAT见证、各预算运行日志及hash均在results的语言/目标子目录。", ""]
    (HERE/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"records":len(records),"exact":witness_count,"proofs":proof_count,"rup_steps":proof_steps,
                      "correlations":correlations,"user_pair":user_pair},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
