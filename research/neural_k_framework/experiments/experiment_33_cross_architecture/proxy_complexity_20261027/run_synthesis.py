"""固定目标面板后逐项求最小电路；只读旧外部标尺，不读取Neural K。"""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from exact_circuits import LANGUAGES, atom_tables, digest, encode, replay_witness, solve, check_lrat


HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
CONFIG = {"max_gates":10,"seconds_per_instance":30,"total_seconds":1200,
          "languages":list(LANGUAGES),"output_negation_free":True,"seed":0}
PANEL_CSV = HERE/"data/function_complexity_panel.csv"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def build_panel():
    with PANEL_CSV.open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    targets = [{"name":"balanced3_"+r["function_hex"],"n":3,"targets":[int(r["function_id"])],
                "scope":"retrospective_all_balanced_orbits",
                "formula_aon_upper":int(r["formula_gate_count_aon_output_symmetric"]),
                "formula_aonx_upper":int(r["formula_gate_count_aonx_output_symmetric"])}
               for r in rows if r["balanced_4_of_8"] == "True" and r["is_orbit_representative"] == "True"]
    assert len(targets) == 10
    atoms,_ = atom_tables(4)
    a,b,c,d = atoms[2:]
    mask = (1 << 16)-1
    logical = {
        "and2":a&b, "and2_or3":(a&b)|c, "majority3":(a&b)|(a&c)|(b&c),
        "mux3":(a&b)|((mask^a)&c),
        "two_and_or":(a&b)|(c&d),
        "shared_left":(a^b)&c, "shared_right":(a^b)&d,
        "independent_right":(a^d)&b,
    }
    targets.extend({"name":"rule4_"+name,"n":4,"targets":[table],"scope":"fixed_external_4bit_panel"}
                   for name,table in logical.items())
    targets.extend([
        {"name":"joint4_shared","n":4,"targets":[logical["shared_left"],logical["shared_right"]],"scope":"shared_dag_certificate"},
        {"name":"joint4_independent","n":4,"targets":[logical["shared_left"],logical["independent_right"]],"scope":"shared_dag_certificate"},
    ])
    return targets


def progress(records, total, current):
    exact = sum(r["status"] == "exact_verified" for r in records)
    unknown = sum(r["status"] != "exact_verified" for r in records)
    lines = ["# 外部复杂度补强进度", "", "日期：2026-10-24。CPU精确逻辑合成，未启动新神经训练或采样。", "",
             "- [完成] 固定三个门库和完整目标面板。",
             "- [完成] 六项编码/证明检查测试通过。",
             f"- [进行中] 电路最小化：{len(records)}/{total}项，exact={exact}，unknown={unknown}。",
             "- [待做] 与旧Neural K对照，保留反序与不确定性。",
             "- [待做] 编码/测度控制核验和结果汇总。", "", f"当前：{current}", "",
             "|语言|目标|状态|最小门数或下界|", "|---|---|---|---:|"]
    for r in records:
        lines.append(f"|{r['language']}|{r['target']['name']}|{r['status']}|{r.get('cost',r.get('lower_bound'))}|")
    (HERE/"PROGRESS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run_target(target, language, deadline):
    folder = OUT/language/target["name"]
    folder.mkdir(parents=True,exist_ok=True)
    result_path = folder/"minimum.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved["status"] == "exact_verified":
            return saved
    attempted = []
    for budget in range(CONFIG["max_gates"]+1):
        bound = folder/f"gates_{budget:02d}"
        if time.perf_counter() >= deadline:
            result = {"status":"unknown_total_budget","language":language,"target":target,
                      "lower_bound":budget,"attempted":attempted}
            write_json(result_path,result)
            return result
        cnf,metadata = encode(target["n"],target["targets"],budget,language)
        bound_result = bound/"result.json"
        if bound_result.exists():
            trial = json.loads(bound_result.read_text(encoding="utf-8"))
            if trial["status"] == "sat_verified":
                replay_witness(trial["witness"])
            elif trial["status"] == "unsat_verified":
                check_lrat(bound/"problem.cnf",bound/"proof.lrat")
            else:
                trial = solve(cnf,metadata,bound,min(CONFIG["seconds_per_instance"],max(1,deadline-time.perf_counter())))
        else:
            trial = solve(cnf,metadata,bound,min(CONFIG["seconds_per_instance"],max(1,deadline-time.perf_counter())))
        attempted.append({"budget":budget,"status":trial["status"],"seconds":trial["seconds"],
                          "cnf_sha256":digest(bound/"problem.cnf"),"result_sha256":digest(bound_result)})
        print(f"{language} {target['name']} gates={budget} {trial['status']} {trial['seconds']:.3f}s",flush=True)
        if trial["status"] == "sat_verified":
            upper = target.get("formula_aonx_upper" if language == "aonx_paid" else "formula_aon_upper")
            if upper is not None and budget > upper:
                raise RuntimeError("DAG最小门数超过已知公式树上界")
            result = {"status":"exact_verified","language":language,"target":target,"cost":budget,
                      "witness":trial["witness"],"attempted":attempted,
                      "certificate":"所有更低预算的LRAT已独立核验；SAT见证已逐点核验"}
            write_json(result_path,result)
            return result
        if trial["status"] != "unsat_verified":
            result = {"status":trial["status"],"language":language,"target":target,"lower_bound":budget,"attempted":attempted}
            write_json(result_path,result)
            return result
    result = {"status":"unknown_gate_cap","language":language,"target":target,
              "lower_bound":CONFIG["max_gates"]+1,"attempted":attempted}
    write_json(result_path,result)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    solver = shutil.which("cadical")
    if not solver:
        raise RuntimeError("缺少CaDiCaL")
    panel = build_panel()
    manifest = {"config":CONFIG,"panel":panel,"input_external_panel_sha256":digest(PANEL_CSV),
        "source_sha256":digest(__file__),"core_sha256":digest(HERE/"exact_circuits.py"),
        "solver_path":solver,"solver_sha256":digest(solver),
        "solver_version":subprocess.check_output([solver,"--version"],text=True).strip(),
        "neural_k_read_during_selection":False}
    path = OUT/"PANEL_FROZEN.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("外部面板或程序变化，不能覆盖旧证书")
    if not path.exists():
        write_json(path,manifest)
    records=[]
    deadline = time.perf_counter()+CONFIG["total_seconds"]
    for language in CONFIG["languages"]:
        for target in panel:
            progress(records,len(panel)*len(CONFIG["languages"]),language+" / "+target["name"])
            result = run_target(target,language,deadline)
            records.append(result)
            write_json(OUT/"SUMMARY.json",{"records":records,"panel_sha256":digest(path)})
    progress(records,len(panel)*len(CONFIG["languages"]),"电路阶段结束，准备对照分析")
    print(json.dumps({"jobs":len(records),"exact":sum(r["status"]=="exact_verified" for r in records)},ensure_ascii=False))


if __name__ == "__main__":
    main()
