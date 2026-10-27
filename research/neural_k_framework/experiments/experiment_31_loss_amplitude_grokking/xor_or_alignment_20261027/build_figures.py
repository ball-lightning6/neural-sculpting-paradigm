"""仅从随包聚合数据重画双语共用图；不导入或运行采样器。"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[2] / "assets/figures"
COLORS = {"mean_bce": "#2374ab", "max_bce": "#c45543", "aligned": "#208163"}
LABELS = {"mean_bce": "Mean BCE", "max_bce": "Pointwise BCE cap", "aligned": "Mean BCE + 0.1 Var(m)"}


def read(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    primary = read("primary_differences.csv")
    confirm = read("confirm_differences.csv")
    bridge = json.loads((ROOT / "data/bridge_verification.json").read_text(encoding="utf-8"))["differences"]
    for kind, color in COLORS.items():
        selected = [r for r in primary if r["kind"] == kind]
        axes[0].plot([float(r["threshold"]) for r in selected], [float(r["delta_K_median"]) for r in selected], color=color, label=LABELS[kind])
        if kind != "mean_bce":
            selected = [r for r in confirm if r["kind"] == kind]
            axes[0].plot([float(r["threshold"]) for r in selected], [float(r["delta_K_median"]) for r in selected], "--", color=color)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set(xlabel="Score threshold (different events)", ylabel="Median K(g) - median K(f) (bits)", title="Solid: initial; dashed: increased budget")
    axes[0].legend(fontsize=8)
    axes[0].annotate("0.08: observed grid node\nnot a predicted threshold", xy=(.08, 0), xytext=(.25, 28), arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=8)
    for index, kind in enumerate(COLORS):
        for offset, records, field, label in ((-.2, primary, "delta_K_median", "Initial"), (0, confirm, "delta_K_median", "Increased budget"), (.2, bridge, "delta_K", "Parent-event bridge")):
            row = next((r for r in records if r["kind"] == kind and float(r["threshold"]) == .001), None)
            if row is None:
                continue
            value = float(row[field])
            axes[1].bar(index + offset, value, width=.18, color=COLORS[kind], alpha={-.2: .45, 0: 1., .2: .7}[offset])
            axes[1].annotate(f"{value:+.2f}", (index + offset, value), xytext=(0, 5 if value > 0 else -12), textcoords="offset points", ha="center", fontsize=8, rotation=90 if value > 0 else 0)
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set_xticks(range(3), ["Mean BCE", "Pointwise cap", "Aligned"])
    axes[1].set(ylabel="Median K(g) - median K(f) (bits)", title="Endpoint: score threshold = 0.001", ylim=(-60, 73))
    axes[1].text(.38, .04, "Within each group, left to right:\ninitial / increased budget / bridge\nMean BCE: initial only", transform=axes[1].transAxes, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(FIGURES / "e31_ext_xor_or_order.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    chosen = []
    for kind in COLORS:
        data = read("primary_aggregates.csv" if kind == "mean_bce" else "confirm_aggregates.csv")
        chosen.append(next(r for r in data if r["rule"] == "g" and r["kind"] == kind and float(r["threshold"]) == .001))
    x = np.arange(3)
    axes[0].bar(x - .18, [float(r["c0_bce_mean"]) for r in chosen], .36, color="#2374ab", label="c = 0: XOR branch")
    axes[0].bar(x + .18, [float(r["c1_bce_mean"]) for r in chosen], .36, color="#c45543", label="c = 1: constant branch")
    axes[0].axhline(.001, color="#555555", linestyle=":", label="Score threshold = 0.001")
    axes[0].set(ylabel="Branch mean BCE", title="g = (a XOR b) OR c")
    axes[0].legend(fontsize=8)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].bar(x, [float(r["margin_variance_mean"]) for r in chosen], color=list(COLORS.values()), width=.55)
    axes[1].set(yscale="log", ylabel="Mean within-network Var(signed margin)", title="Precision imbalance across the 16 inputs")
    for ax in axes:
        ax.set_xticks(x, ["Mean BCE\n(initial)", "Pointwise cap\n(increased budget)", "Aligned\n(increased budget)"])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(FIGURES / "e31_ext_branch_bce.png", dpi=180)
    plt.close(fig)
    print("e31_ext_xor_or_order.png; e31_ext_branch_bce.png")


if __name__ == "__main__":
    main()
