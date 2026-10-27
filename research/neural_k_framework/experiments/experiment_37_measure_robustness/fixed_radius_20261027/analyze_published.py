"""仅复核随包轻量表、重画网页图；不调用 SMC、训练或外部结果包。"""

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import statistics

from runtime import HERE, read_json, verify_bundle

TARGETS = ("projection", "majority3", "parity2")
MEASURES = ("uniform_ball", "wide_gaussian", "spherical_t8", "uniform_cube")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pooled_K(values):
    if not values or any(value is None for value in values):
        return None
    low = min(values)
    return low - math.log2(statistics.mean(2 ** (low - value) for value in values))


def rank_correlation(left, right):
    # 本冻结面板没有并列值；不把未知并列规则静默处理成唯一排序。
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("出现并列值，须显式使用平均秩处理。")
    rank = lambda values: [sorted(values).index(value) for value in values]
    return statistics.correlation(rank(left), rank(right))


def check_data():
    verified = verify_bundle()
    review = read_json(HERE / "data/measure_review.json")
    curves = read_csv(HERE / "data/measure_curves.csv")
    curves = [{key: value if key == "target" else float(value)
               for key, value in row.items()} for row in curves]
    terminals = read_csv(HERE / "data/measure_terminal_status.csv")
    for row in terminals:
        for key in ("radius", "last_support"):
            row[key] = float(row[key])
        for key in ("replica", "nodes_reached", "level"):
            row[key] = int(row[key])
    if len(terminals) != 24 or len(curves) != 48:
        raise ValueError("固定半径表不完整。")
    for actual, expected in zip(curves, review["aggregates"], strict=True):
        if actual != expected:
            raise ValueError("曲线 CSV 与冻结复核摘要不一致。")
    for actual, expected in zip(terminals, review["terminal_details"], strict=True):
        if {**actual, "reason": actual["reason"] or None} != expected:
            raise ValueError("终态 CSV 与冻结复核摘要不一致。")
    completed = sum(row["phase"] == "completed" for row in terminals)
    stalled = sum(row["reason"] == "stalled_levels_budget" for row in terminals)
    if (completed, stalled) != (16, 8):
        raise ValueError("终态计数不符。")
    by_key = {(r["target"], r["radius"], r["epsilon"]): r for r in curves}
    contrasts = []
    for radius in (20.0, 40.0):
        for target in TARGETS:
            rows = sorted((r for r in curves if r["target"] == target and r["radius"] == radius),
                          key=lambda r: -r["epsilon"])
            if any(a["K_pooled"] > b["K_pooled"] for a, b in zip(rows, rows[1:])):
                raise ValueError("所测曲线的质量单调性异常。")
            high = by_key.get((target, radius, 0.03))
            low = by_key.get((target, radius, 0.001))
            delta = low["K_pooled"] - high["K_pooled"] if low and high else None
            contrasts.append({"radius": radius, "target": target,
                              "K_003": high["K_pooled"] if high else None,
                              "K_0001": low["K_pooled"] if low else None,
                              "delta_K": delta,
                              "c_log2_precision": delta / math.log2(30) if delta is not None else None})
    for expected in review["contrasts"]:
        actual = next(r for r in contrasts if (r["radius"], r["target"]) ==
                      (expected["radius"], expected["target"]))
        if not math.isclose(actual["delta_K"], expected["delta_K"], abs_tol=1e-12):
            raise ValueError("固定半径增量不符。")
        if not math.isclose(actual["c_log2_precision"], expected["finite_window_power_exponent"], abs_tol=1e-12):
            raise ValueError("固定半径精度坐标不符。")
    rules = read_csv(HERE / "historical_20260913/rules.csv")
    grouped = {}
    for row in rules:
        grouped.setdefault(row["measure"], {})[int(row["function_id"])] = row
    old_summary = read_json(HERE / "historical_20260913/summary.json")
    comparisons = []
    for source in old_summary["comparisons"]:
        a, b, metric = source["reference"], source["compared"], source["metric"]
        ids = sorted(grouped[a])
        if len(ids) != 20 or ids != sorted(grouped[b]):
            raise ValueError("旧自然测度面板身份不一致。")
        left = [float(grouped[a][fid][metric]) for fid in ids]
        right = [float(grouped[b][fid][metric]) for fid in ids]
        rho = rank_correlation(left, right)
        same = sum((left[i] - left[j]) * (right[i] - right[j]) > 0
                   for i, j in itertools.combinations(range(20), 2))
        if not math.isclose(rho, source["spearman"], abs_tol=1e-12) or same != 190-source["inversions"]:
            raise ValueError("旧面板排序复算不一致。")
        comparisons.append({"reference": a, "compared": b, "metric": metric,
                            "spearman": rho, "same_order_pairs": same})
    for measure in MEASURES:
        group = grouped[measure]
        reference = float(group[0]["slope"])
        if sum(float(r["slope"]) > reference for fid, r in group.items() if fid != 0) != 19:
            raise ValueError("旧面板相对于常数的比较不一致。")
    return {"verified_files": verified, "completed_pools": completed, "stalled_pools": stalled,
            "curve_rows": len(curves), "contrasts": contrasts, "historical_comparisons": comparisons,
            "historical_audit_summary_only": review["audits"],
            "sampling_started": False}, curves, grouped


def render_figures(curves, grouped):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = HERE.parents[2] / "assets/figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"projection": "#2675b5", "majority3": "#238461", "parity2": "#ba493e"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), constrained_layout=True)
    for ax, radius in zip(axes, (20.0, 40.0)):
        for target in TARGETS:
            rows = sorted((r for r in curves if r["target"] == target and r["radius"] == radius
                           and r["epsilon"] <= 0.3), key=lambda r: -r["epsilon"])
            x = [math.log2(0.3 / r["epsilon"]) for r in rows]
            ax.plot(x, [r["K_pooled"] for r in rows], "o-", color=colors[target], label=target)
            ax.fill_between(x, [r["K_min"] for r in rows], [r["K_max"] for r in rows],
                            color=colors[target], alpha=0.17)
            if radius == 20 and target != "projection":
                ax.annotate("last reached node", (x[-1], rows[-1]["K_pooled"]),
                            xytext=(7, -14), textcoords="offset points", fontsize=8, color=colors[target])
        ax.set_title(f"Fixed radius R = {radius:g}")
        ax.set_xlabel(r"Precision coordinate $\log_2(0.3/\epsilon)$")
        ax.set_ylabel("Angular K (bits)")
        ax.set_xticks([math.log2(0.3 / e) for e in (0.3, 0.03, 0.003, 0.001)],
                      ["0", "3.32", "6.64", "8.23"])
        ax.set_xlim(-0.25, 8.8)
        ax.set_ylim(0, 165)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right" if radius == 20 else "upper left", fontsize=9)
    fig.suptitle("Fixed-radius angular contraction | October 2026", fontsize=13)
    path = figures / "e37_ext_fixed_radius.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    labels = ["Uniform ball", "Wide Gaussian", "Spherical t8", "Uniform cube"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    for ax, metric in zip(axes, ("K", "slope")):
        matrix = []
        for left in MEASURES:
            ids = sorted(grouped[left])
            matrix.append([rank_correlation([float(grouped[left][fid][metric]) for fid in ids],
                                            [float(grouped[right][fid][metric]) for fid in ids])
                           for right in MEASURES])
        ax.imshow(matrix, vmin=0.85, vmax=1.0, cmap="YlGnBu")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.5f}", ha="center", va="center",
                        fontsize=10, color="white" if value >= 0.95 else "black")
        ax.set_xticks(range(4), labels, rotation=22, ha="right", fontsize=9)
        ax.set_yticks(range(4), labels, fontsize=9)
        ax.set_title("K ranks at BCE = 0.001" if metric == "K" else "Margin-coordinate coefficient ranks")
    fig.suptitle("Historical 20-rule panel | September 13, 2026", fontsize=13)
    old_path = figures / "e37_ext_natural_rank.png"
    fig.savefig(old_path, dpi=170)
    plt.close(fig)
    return [path.name, old_path.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    result, curves, grouped = check_data()
    if args.figures:
        result["figures"] = render_figures(curves, grouped)
    if args.write_summary:
        output = HERE / "analysis"
        output.mkdir(exist_ok=True)
        (output / "recomputed.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
