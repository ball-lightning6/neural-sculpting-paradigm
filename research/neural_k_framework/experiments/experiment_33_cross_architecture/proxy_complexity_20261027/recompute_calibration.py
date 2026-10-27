"""从小型公开聚合表复算排序、门库关联和池间倒序，不调用求解器或 SMC。

dag_costs.csv 是历史结果摘录，不是随包附带的最小性证明。
完整证书须由 run_synthesis.py 生成，再由 analyze_certificates.py 重放检查。
"""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audit_existing_proxies import correlation, essential, KEY_PROXIES
from exact_circuits import LANGUAGES


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def calculate():
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8-sig"))
    for row in provenance["data_files"]:
        if hashlib.sha256((DATA / row["file"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"数据hash不符：{row['file']}")
    panel = {int(r["function_id"]): r for r in read("function_complexity_panel.csv")}
    chosen = sorted(f for f, r in panel.items() if r["balanced_4_of_8"] == "True" and r["is_orbit_representative"] == "True")
    curves = {(r["architecture"], float(r["threshold"]), int(r["function_id"])): float(r["complexity_bits"]) for r in read("rank_trajectories.csv")}
    architectures = sorted({a for a, _, _ in curves})
    recorded = read("dag_costs.csv")
    costs = {(r["language"], int(r["function_id"])): int(r["lower_bound"]) for r in recorded if r["n"] == "3" and r["status"] == "exact_verified"}
    measurements = {
        "K_0p001": lambda a, f: curves[a, .001, f],
        "slope_0p01_to_0p001": lambda a, f: (curves[a, .001, f] - curves[a, .01, f]) / math.log2(10),
        "slope_0p0025_to_0p001": lambda a, f: (curves[a, .001, f] - curves[a, .0025, f]) / math.log2(2.5),
    }
    correlations = []
    for language in LANGUAGES:
        for arch in architectures:
            for name, value in measurements.items():
                correlations.append({"language": language, "architecture": arch, "measurement": name,
                                     "spearman": correlation([costs[language, f] for f in chosen], [value(arch, f) for f in chosen])})
    pair_directions = []
    for a, b in itertools.combinations(chosen, 2):
        signs = [int(np.sign(curves[arch, .001, b] - curves[arch, .001, a])) for arch in architectures]
        pair_directions.append({"a": a, "b": b, "same_direction": len(set(signs)) == 1})
    affine = [f for f in chosen if int(panel[f]["anf_degree"]) == 1 and int(panel[f]["decision_tree_internal_tests"]) > 1]
    sensitivity = []
    for language in LANGUAGES:
        ids = [f for f in chosen if f not in affine]
        for arch in architectures:
            sensitivity.append({"language": language, "architecture": arch,
                                "spearman": correlation([costs[language, f] for f in ids], [curves[arch, .001, f] for f in ids])})
    groups = {}
    for f in chosen:
        r = panel[f]
        key = (int(r["positive_count"]), essential(list(map(int, r["truth_table_x000_to_x111"]))), float(r["total_influence"]))
        groups.setdefault(key, []).append(f)
    matched = []
    for ids in groups.values():
        for a, b in itertools.combinations(ids, 2):
            ca, cb = [int(panel[f]["formula_gate_count_aon_output_symmetric"]) for f in (a, b)]
            if ca != cb:
                matched.append((a, b) if ca < cb else (b, a))
    pools = {}
    for row in read("replica_endpoints.csv"):
        key = (row["architecture"], int(row["function_id"]))
        pools.setdefault(key, {})[int(row["replica"])] = float(row["neural_k_bits"])
    reversals = []
    for arch in architectures:
        for low, high in matched:
            a, b = [np.array(list(pools[arch, f].values())) for f in (low, high)]
            if len(a) != 8 or len(b) != 8:
                raise ValueError("每目标必须保留8个池")
            delta = b[:, None] - a[None, :]
            reversals.append({"architecture": arch, "lower_cost": f"0x{low:02X}", "higher_cost": f"0x{high:02X}",
                              "median_difference": float(np.median(b) - np.median(a)),
                              "cross_pool_min": float(delta.min()), "cross_pool_max": float(delta.max())})
    proxies = []
    for proxy in KEY_PROXIES:
        for arch in architectures:
            proxies.append({"proxy": proxy, "architecture": arch,
                            "spearman": correlation([float(panel[f][proxy]) for f in chosen], [curves[arch, .001, f] for f in chosen])})
    return {"recorded_certificate_count": len(recorded),
            "recorded_exact_count": sum(r["status"] == "exact_verified" for r in recorded),
            "unresolved": [r for r in recorded if r["status"] != "exact_verified"],
            "balanced_orbits": chosen, "shared_directions": sum(r["same_direction"] for r in pair_directions),
            "pair_count": len(pair_directions), "affine_parity_ids": affine,
            "correlations": correlations, "without_affine_parity": sensitivity, "proxy_correlations": proxies,
            "reversals": reversals, "negative_medians": sum(r["median_difference"] < 0 for r in reversals),
            "all_cross_pool_negative": sum(r["cross_pool_max"] < 0 for r in reversals),
            "scope": "历史聚合数据复算；未重新证明59项最小性，未重跑旧神经体积测量。"}


def main():
    report = calculate()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
