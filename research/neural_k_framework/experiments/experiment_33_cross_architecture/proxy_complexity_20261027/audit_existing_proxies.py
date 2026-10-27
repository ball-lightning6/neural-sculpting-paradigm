"""只读旧3-bit数据：核对外部代理覆盖、恒等关系和有限窗口收缩。"""

import csv
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data"
FILES = ["function_complexity_panel.csv", "complexity_proxy_correlations.csv", "rank_trajectories.csv"]
KEY_PROXIES = [
    "total_influence", "boolean_operations_output_symmetric", "formula_gate_count_aon_output_symmetric",
    "formula_gate_count_aonx_output_symmetric", "decision_tree_internal_tests",
    "mingard_generalization_complexity_c1_c2", "critical_sample_ratio",
    "mingard_lz_symmetry_min", "ctm_b2_d12_symmetry_mean", "anf_term_count_output_symmetric",
    "walsh_spectral_entropy",
]


def read_csv(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def essential(outputs):
    return sum(any(outputs[i] != outputs[i ^ (1 << bit)] for i in range(8)) for bit in range(3))


def main():
    panel = read_csv(FILES[0])
    old_correlations = read_csv(FILES[1])
    trajectories = read_csv(FILES[2])
    assert len(panel) == 256 and {int(row["function_id"]) for row in panel} == set(range(256))
    panel.sort(key=lambda r: int(r["function_id"]))
    proxies = sorted({row["proxy"] for row in old_correlations})
    balance, representatives, essentials = [], [], []
    max_influence_error = max_walsh_error = 0.
    for row in panel:
        outputs = list(map(int, row["truth_table_x000_to_x111"]))
        assert sum(v << i for i, v in enumerate(outputs)) == int(row["function_id"])
        inf = sum(outputs[i] != outputs[i ^ (1 << bit)] for i in range(8) for bit in range(3)) / 8
        max_influence_error = max(max_influence_error, abs(inf-float(row["total_influence"])))
        max_walsh_error = max(max_walsh_error, abs(inf-float(row["walsh_mean_degree"])))
        balance.append(sum(outputs) == 4)
        representatives.append(row["is_orbit_representative"].lower() == "true")
        essentials.append(essential(outputs))
    assert max_influence_error < 1e-12 and max_walsh_error < 1e-12
    scopes = {
        "all": list(range(256)),
        "balanced": [i for i in range(256) if balance[i]],
        "orbit_representatives": [i for i in range(256) if representatives[i]],
        "balanced_orbit_representatives": [i for i in range(256) if balance[i] and representatives[i]],
    }
    curves = defaultdict(dict)
    for row in trajectories:
        if row["stage"] == "neural_k":
            curves[(row["architecture"], float(row["threshold"]))][int(row["function_id"])] = float(row["complexity_bits"])
    architectures = sorted({name for name, _ in curves})
    targets = {"K_at_0p001": lambda a, f: curves[a,.001][f],
               "slope_0p01_to_0p001": lambda a,f: (curves[a,.001][f]-curves[a,.01][f])/math.log2(10),
               "slope_0p0025_to_0p001": lambda a,f: (curves[a,.001][f]-curves[a,.0025][f])/math.log2(2.5)}
    fresh = []
    old_max_error = 0.
    old_index = {(r["architecture"], r["proxy"], r["scope"]): r for r in old_correlations
                 if r["measurement"] == "neural_k" and r["threshold"] == "0.001"}
    for architecture in architectures:
        for scope, ids in scopes.items():
            for proxy in proxies:
                for measurement, value in targets.items():
                    rho = correlation([float(panel[i][proxy]) for i in ids], [value(architecture,i) for i in ids])
                    fresh.append({"architecture":architecture,"scope":scope,"proxy":proxy,
                                  "measurement":measurement,"n_functions":len(ids),"spearman":rho})
                    if measurement == "K_at_0p001":
                        old = old_index[architecture,proxy,scope]["spearman"]
                        if old and rho is not None:
                            old_max_error = max(old_max_error, abs(float(old)-rho))
                        else:
                            assert not old and rho is None
    assert old_max_error < 1e-12
    matched = []
    for scope in ("balanced", "balanced_orbit_representatives"):
        groups = defaultdict(list)
        for i in scopes[scope]:
            # 固定输出平衡度、实际相关变量数、超立方体边界翻转数。
            key = (int(panel[i]["positive_count"]), essentials[i], round(8*float(panel[i]["total_influence"])))
            groups[key].append(i)
        for proxy in KEY_PROXIES:
            pairs = []
            for ids in groups.values():
                for i,j in combinations(ids, 2):
                    a,b = float(panel[i][proxy]),float(panel[j][proxy])
                    if a != b:
                        pairs.append((i,j) if a < b else (j,i))
            for architecture in architectures:
                for measurement,value in targets.items():
                    differences = [value(architecture,j)-value(architecture,i) for i,j in pairs]
                    pos = sum(d > 1e-9 for d in differences)
                    neg = sum(d < -1e-9 for d in differences)
                    ties = len(differences)-pos-neg
                    matched.append({"scope":scope,"proxy":proxy,"architecture":architecture,
                        "measurement":measurement,"pair_count":len(pairs),"positive":pos,"negative":neg,"ties":ties,
                        "pair_ids":[[f"0x{i:02X}",f"0x{j:02X}"] for i,j in pairs] if scope.endswith("representatives") else [],
                        "note":"配对共享函数，不能作为独立样本计算显著性；未传播旧SMC副本误差。"})
    result = {
        "source_files": {name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in FILES},
        "proxy_count_in_old_correlations":len(proxies),"proxies":proxies,"scope_counts":{k:len(v) for k,v in scopes.items()},
        "architectures":architectures,"checks":{"influence_recomputed_max_error":max_influence_error,
        "walsh_mean_degree_equals_influence_max_error":max_walsh_error,"old_spearman_max_error":old_max_error},
        "correlations":fresh,"matched_pairs":matched,
        "scope":"CPU读取已完成聚合数据的探索性重分析；不新增SMC，不证明旧采样全局正确，不把代理个数当独立证据数。",
    }
    left, right = 0x17, 0x2B
    assert all(((left >> (i ^ 1)) & 1) == ((right >> i) & 1) for i in range(8))
    result["input_flip_example"] = {
        "left":"0x17", "right":"0x2B", "transformation":"第三输入位取反；旧编码为zero_one",
        "aon_costs":[float(panel[i]["formula_gate_count_aon_output_symmetric"]) for i in (left,right)],
        "influence":[float(panel[i]["total_influence"]) for i in (left,right)],
        "K_at_0p001":{a:[curves[a,.001][i] for i in (left,right)] for a in architectures},
        "note":"只核对真值表与旧聚合数值；编码/先验解释未做新的干预确认。",
    }
    (HERE/"LOCAL_EVIDENCE_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    def rho_range(scope, measurement, proxy):
        values = [r["spearman"] for r in fresh if r["scope"]==scope and r["measurement"]==measurement and r["proxy"]==proxy and r["spearman"] is not None]
        return "常量/无定义" if not values else f"{min(values):.3f} 至 {max(values):.3f}"
    lines = ["# 现有外部复杂度证据核对", "", result["scope"], "",
             f"旧代码相关性表已经覆盖{len(proxies)}种代理列；256个函数中70个平衡，全部函数有40个既定对称轨道，平衡函数有10个轨道。",
             "", "这里的轨道只使用旧协议规定的输入变量置换和输出取反，不等于允许输入位取反的NPN类。",
             "", "## 覆盖与重复", "",
             "- Mingard/Valle-Perez五类代理已经实现，包括LZ、SOP操作数、C1+C2、输出熵与CSR。",
             "- 还有精确公式树（两种门库）、精确决策树、ANF、CTM和Walsh/Fourier统计；公式树允许重复子表达式但不是共享子计算的最小DAG电路。",
             "- 直接重算256个函数的total influence，与旧表相同；Walsh平均次数和total influence数值上精确相等，不是两份独立证据。",
             "- 平衡函数的输出熵恒为1，不能在该组内给出复杂度排序。",
             "- 重新计算最深节点的全部Spearman，与旧表一致；不因此认证SMC估计无偏。",
             "", "## 平衡函数的外部关联", "",
             "四个架构的Spearman范围。后两列是在本轮指定窗口后作的探索性重分析，不是预注册新预测。",
             "", "|代理|K(0.001)|窗口斜率0.01到0.001|窗口斜率0.0025到0.001|",
             "|---|---:|---:|---:|"]
    for proxy in KEY_PROXIES:
        lines.append("|"+"|".join([proxy]+[rho_range("balanced",measurement,proxy) for measurement in targets])+"|")
    lines += ["", "## 对称轨道代表", "",
              "这里每个架构只有10个平衡轨道，不能把70个对称函数或4个高度相关架构解释成数百个独立规则确认。",
              "", "|代理|K(0.001)|窗口斜率0.01到0.001|窗口斜率0.0025到0.001|",
              "|---|---:|---:|---:|"]
    for proxy in KEY_PROXIES:
        lines.append("|"+"|".join([proxy]+[rho_range("balanced_orbit_representatives",measurement,proxy) for measurement in targets])+"|")
    lines += ["", "## 匹配几何后的辨别能力", "",
              "只在输出比例、相关变量数、total influence相同的函数间比较外部成本的严格大小；下表是10个平衡轨道代表可提供的对照。数量较少，不能据此宣称控制了所有混杂。",
              "", "|代理|可比对数|四架构K方向：正/负/平|", "|---|---:|---|"]
    for proxy in KEY_PROXIES:
        rows = [r for r in matched if r["scope"]=="balanced_orbit_representatives" and r["proxy"]==proxy and r["measurement"]=="K_at_0p001"]
        lines.append(f"|{proxy}|{rows[0]['pair_count']}|"+"; ".join(f"{r['architecture']}:{r['positive']}/{r['negative']}/{r['ties']}" for r in rows)+"|")
    example = result["input_flip_example"]["K_at_0p001"]["tanh_w16_d1"]
    lines += ["", "一个具体例子：`0x17`为NOT majority3，`0x2B`是其第三输入位取反。两者均平衡、依赖3个变量、total influence=1.5；输出取反取最小后的AON公式成本为4和5，但tanh16的K(0.001)分别为"
              f"{example[0]:.6f}和{example[1]:.6f} bit。",
              "", "这不是一个无需解释的普适复杂度倒序：旧输入编码是0/1，输入位取反一般需要权重变号和bias平移，独立参数先验不自动保持；而公式成本对输入NOT收费。它提示必须区分编码成本、逻辑门库和神经参考测度。不能事后删除这对，也不能将其直接推广成全部描述复杂度关系失败。",
              "", "## 解释", "",
              "当前不是没有外部校准，而是外部校准主要集中于很小的Boolean世界；某些语言有明显正关联，另一些有负关联。不能只保留相关最高的指标，也不能把这些代理一概改名为Kolmogorov复杂度。",
              "", "下一步应优先采用固定语言的更大函数面板与可证偏序、匹配几何的逻辑成本对照、结构化/随机例外和共享计算干预；不必重新实施已存在的19列指标。",
              "", "机器可读结果与源文件hash见[LOCAL_EVIDENCE_AUDIT.json](LOCAL_EVIDENCE_AUDIT.json)。", ""]
    (HERE/"LOCAL_EVIDENCE_AUDIT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("proxy_count_in_old_correlations","scope_counts","checks")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
