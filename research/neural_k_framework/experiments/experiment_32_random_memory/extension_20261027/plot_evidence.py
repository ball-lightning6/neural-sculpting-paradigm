"""从轻量证据表生成中英文共用静态图，不运行实验。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import FIGURE_DIR
from review import read, fits

COLORS = ("#007C91", "#B2474B", "#6C6E1D")
ARCHES = ("[32]", "[32,32]", "[32,32,32]")


def finish(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / ("e32_ext_" + name + ".png"), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curve = read("main_curve.json")["curve"]
    n = np.array([r["n"] for r in curve])
    fit = fits(curve)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, key in enumerate(("one", "two", "three")):
        y = np.array([r[key] for r in curve])
        ax = axes[0, i]
        ax.plot(n, y, "o-", color=COLORS[i], label="Observed")
        ax.plot(n, fit[key]["a"]*n+fit[key]["b"], "--", color="#555555", label="With intercept")
        ax.plot(n, fit[key]["q"]*n, ":", color="#9A9A9A", label="Through origin")
        ax.set(title=ARCHES[i], ylabel="Window coefficient c", xlabel="Training samples n")
        ax.legend(fontsize=8)
        axes[1, i].plot(n, y/n, "o-", color=COLORS[i])
        axes[1, i].set(xlabel="Training samples n", ylabel="c / n")
    fig.suptitle("Same four random tables; 384 independent SMC pools")
    finish(fig, "depth")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    enhanced = read("enhanced.json")["aggregates"]
    x = [r["n"] for r in enhanced]
    for field, label, color in (
        ("old_c_matched_two_datasets", "Base budget: .002 to .001", "#777777"),
        ("new_c_matched_two_datasets", "Enhanced: .002 to .001", COLORS[0]),
    ):
        axes[0].plot(x, [r[field] for r in enhanced], "o-", label=label, color=color)
    axes[0].plot(x, [r["window_medians_mean"][-1] for r in enhanced], "s--", label="Enhanced: .0002 to .0001", color=COLORS[1])
    axes[0].set(xlabel="Training samples n", ylabel="Window coefficient c", title="Matched tables 0 and 1")
    axes[0].legend(fontsize=9)
    rows = read("calibration_alignment.json")["calibration"]
    for i, row in enumerate(rows):
        axes[1].scatter(np.full(4, i)+np.linspace(-.08, .08, 4), row["c_by_pool"], color=COLORS[0], s=28)
        axes[1].plot([i-.18,i+.18], [row["c_median"]]*2, color=COLORS[1], linewidth=2)
    axes[1].set(xticks=range(5), xticklabels=[r["rule"] for r in rows], ylabel="Window coefficient c", title="Five full-target rules; four pools each")
    axes[1].tick_params(axis="x", labelrotation=15)
    finish(fig, "controls")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    rows = read("calibration_alignment.json")["alignment"]
    for ax, kind in zip(axes, ("random", "projection")):
        selected = sorted((r for r in rows if r["kind"] == kind), key=lambda r:r["n"])
        for pool in range(2):
            ax.plot([r["n"] for r in selected], [r["pools"][pool]["raw_c"] for r in selected],
                    "o--", color="#777777", alpha=.8, label="Parent BCE" if pool == 0 else None)
            for delta, color in zip(("1.0", "0.3", "0.1"), COLORS):
                ax.plot([r["n"] for r in selected], [r["pools"][pool]["aligned"][delta]["c"] for r in selected],
                        "o-" if pool == 0 else "s:", color=color, label="Range <= "+delta if pool == 0 else None)
        ax.set(title=kind, xlabel="Training samples n", ylabel="Window coefficient c")
        ax.legend(fontsize=9)
    fig.suptitle("12 parent pools; 24 conditional bridges (two pools shown separately)")
    finish(fig, "alignment")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    agreement = read("agreement.json")["means"]
    loss = read("loss.json")["groups"]
    for hidden, color, label in zip(([32],[32,32],[32,32,32]), COLORS, ARCHES):
        group = sorted((r for r in agreement if r["kind"]=="random" and r["hidden"]==hidden), key=lambda r:r["n"])
        for ax, key in ((axes[0,0], "agreement"), (axes[0,1], "model_accuracy")):
            ax.plot([r["n"] for r in group], [r[key] for r in group], "o-", color=color, label=label)
        group = sorted((r for r in loss if r["kind"]=="random" and r["hidden"]==hidden and r["epsilon"]==.001), key=lambda r:r["n"])
        axes[1,0].plot([r["n"] for r in group], [r["max_to_mean_median"] for r in group], "o-", color=color, label=label)
    axes[0,0].set(title="Random labels: pointwise agreement", ylabel="Agreement", ylim=(.45,1.01))
    axes[0,1].set(title="Unseen random-label accuracy", ylabel="Accuracy", ylim=(.45,.55))
    axes[0,1].axhline(.5, color="#777777", linestyle="--")
    axes[1,0].set(title="Per-sample BCE at mean BCE <= .001", ylabel="Maximum / mean BCE")
    shortcuts = read("shortcuts.json")
    group = [r for r in shortcuts if r["n"] <= 128]
    axes[1,1].plot([r["n"] for r in group], [r["linear"] for r in group], "o-", color=COLORS[0])
    axes[1,1].set(title="128 reference tables: linear separability", ylabel="Fraction", ylim=(-.03,1.03))
    for ax in axes.ravel():
        ax.set_xlabel("Training samples n")
        if ax is not axes[1,1]:
            ax.legend(fontsize=9)
    finish(fig, "diagnostics")


if __name__ == "__main__":
    main()

