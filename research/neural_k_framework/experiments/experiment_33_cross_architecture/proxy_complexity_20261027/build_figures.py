"""从公开聚合数据生成英文标注图；不调用求解器、训练器或采样器。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recompute_calibration import calculate


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[2] / "assets/figures"


def main():
    report = calculate()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    names = ["aon_paid", "aonx_paid", "aon_free_inputs"]
    for y, name, color in zip(range(3), names, ("#2374ab", "#c45543", "#208163")):
        values = [r["spearman"] for r in report["correlations"] if r["language"] == name and r["measurement"] == "K_0p001"]
        axes[0].plot([min(values), max(values)], [y, y], color=color, linewidth=6, solid_capstyle="round")
        axes[0].scatter(values, [y] * 4, color=color, s=35)
        axes[0].text((min(values) + max(values)) / 2, y + .15, f"{min(values):.3f} to {max(values):.3f}", ha="center", va="center", fontsize=9)
    axes[0].set_yticks(range(3), ["AND / OR / NOT", "+ unit-cost XOR", "Free input literals"])
    axes[0].set(xlim=(-.5, 1.0), xlabel="Spearman rho with K(0.001)", title="10 balanced orbits; four architectures")
    axes[0].set_xticks([-.5, -.25, 0, .25, .5, .75, 1])
    axes[0].axvline(0, color="#555555", linewidth=.8)
    axes[0].set_ylim(2.3, -.3)
    rows = [r for r in report["reversals"] if r["lower_cost"] == "0x17" and r["higher_cost"] == "0x2B"]
    for y, r in enumerate(rows):
        axes[1].plot([r["cross_pool_min"], r["cross_pool_max"]], [y, y], color="#2374ab", linewidth=3)
        axes[1].plot(r["median_difference"], y, "o", color="#c45543")
    axes[1].set_yticks(range(len(rows)), [r["architecture"] for r in rows])
    axes[1].set(xlim=(-110, 5), xlabel="K(0x2B) - K(0x17) (bits)", title="AON formula cost: 5 versus 4")
    axes[1].axvline(0, color="#555555", linewidth=.8)
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Recorded calibration: gate-language dependence and persistent local reversals\nRanges describe architectures or observed pools, not global-bias confidence intervals", fontsize=11)
    fig.savefig(FIGURES / "e33_ext_complexity_calibration.png", dpi=180)
    plt.close(fig)
    print("e33_ext_complexity_calibration.png")


if __name__ == "__main__":
    main()
