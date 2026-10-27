"""仅复算公开的历史汇总、导出 CSV 和英文图轴；不加载训练或 SMC。"""

import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
FIGURES = ROOT.parents[2] / "assets/figures"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8-sig"))


def export(name, rows):
    columns = sorted({key for row in rows for key in row})
    with (EVIDENCE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = read("cv58.json")
    rows = data["original_static"]["rows"] + data["extension"]["rows"]
    assert len(rows) == 58 and len({r["condition"] for r in rows}) == 58
    assert all(r["sgd_models"] == 128 and r["pool_counts"] == [1024] * 4 for r in rows)
    groups = []
    for stage, network, n in [
        ("original_static", "cnn_w4", 32), ("original_static", "mlp_256_128", 32),
        ("original_static", "cnn_w8", 32), ("extension", "cnn_w4", 8),
        ("extension", "cnn_w4", 64), ("extension", "mlp_256_128", 8),
        ("extension", "mlp_256_128", 64),
    ]:
        selected = [r for r in rows if r["stage"] == stage and r["network"] == network
                    and re.fullmatch(r".*__partition\d_seed\d_n" + str(n), r["condition"])]
        groups.append(dict(stage=stage, network=network, n=n, conditions=len(selected),
            hard_mae_pp=float(np.mean([abs(r["hard_accuracy"] - r["sgd_accuracy"]) for r in selected]) * 100),
            low_loss_mae_pp=float(np.mean([abs(r["smc_accuracy"] - r["sgd_accuracy"]) for r in selected]) * 100),
            point_mae_pp=float(np.mean([r["point_mae"] for r in selected]) * 100)))
    export("cv58_table.csv", [{k: r[k] for k in (
        "stage", "condition", "network", "sample_count", "sgd_models", "sgd_accuracy",
        "smc_accuracy", "hard_accuracy", "point_mae", "sgd_loss_mean", "smc_loss_mean")}
        for r in rows])
    export("cv_group_mae.csv", groups)
    gc = read("direct_gc58.json")
    export("gc58_table.csv", [dict(stage=r["stage"], condition=r["condition"], **r["record"]) for r in gc])
    calibration = read("gc_calibration13.json")
    assert len(calibration) == 13
    assert sum(r["record"]["accepted"] == 512 for r in gc) == 13
    assert sum(r["record"]["accepted"] == 0 for r in gc) == 45
    stp = read("stp48.json")
    stp003 = [r for r in stp["rows"] if r["epsilon"] == .003]
    assert len(stp["rows"]) == 240 and len(stp003) == 48 and not stp["missing"]
    assert all(r["stp_models"] == r["planned"] == 128 for r in stp["rows"])
    export("stp48_table.csv", [{k: r[k] for k in (
        "condition", "network", "control", "sigma0", "epsilon", "stp_models", "planned",
        "stp_accuracy", "sgd_accuracy", "smc_accuracy", "hard_fit_accuracy", "stp_smc_point_mae_pp",
        "stp_sgd_point_mae_pp")} for r in stp["rows"]])
    rechecks = [r for r in read("recheck24.json") if r["epsilon"] == .003 and r["view"] == "cumulative"]
    export("recheck_table.csv", [dict(condition=r["condition"], pools=r["static_pools_reached"],
        smc_accuracy=r["static"]["accuracy"], sgd_accuracy=r["sgd"]["accuracy"],
        gap_pp=100*r["accuracy_absolute_error"], K_range_bits=r["K_range_bits"]) for r in rechecks])
    summary = dict(
        cv_groups=groups,
        direct_gc=dict(sufficient=13, zero_hits=45, target=512,
            n8_calibration_mae_pp=float(np.mean([r["absolute_gap_pp"] for r in calibration if r["condition"].endswith("_n8")]))),
        stp_clean=[r for r in stp["summary"] if r["control"] == "clean"],
        source_note="由公开历史数值复算；不是重新训练、采样或独立前向审计。",
    )
    (EVIDENCE / "derived_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = ["CNN W4 / n32", "MLP / n32", "CNN W8 / n32", "CNN W4 / n8", "CNN W4 / n64", "MLP / n8", "MLP / n64"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(len(groups))
    ax.bar(x - .19, [r["hard_mae_pp"] for r in groups], .38, label="Hard-fit SMC", color="#667078")
    ax.bar(x + .19, [r["low_loss_mae_pp"] for r in groups], .38, label="Low-loss SMC (BCE <= 0.003)", color="#167b73")
    ax.set_xticks(x, labels, rotation=23, ha="right")
    ax.set_ylabel("MAE of mean test accuracy vs SGD (pp)")
    ax.set_title("Fixed-condition predictions: clean subsets of the 58-condition matrix")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / "e36_ext_cv_mae.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout="constrained")
    for ax, network, title in zip(axes, ("cnn_w4", "mlp_256_128"), ("CNN W4", "MLP 256/128")):
        selected = [r for r in summary["stp_clean"] if r["network"] == network]
        ax.plot(range(3), [r["stp_sgd_accuracy_mae_pp"] for r in selected], "o-", color="#167b73", label="STP vs SGD")
        ax.plot(range(3), [r["stp_smc_accuracy_mae_pp"] for r in selected], "s--", color="#ab3d58", label="STP vs low-loss SMC")
        ax.set_xticks(range(3), [str(r["sigma0"]) for r in selected])
        ax.set_xlabel("Initial STP step scale")
        ax.set_title(title + "\n6 clean conditions; 128/128 reached", fontsize=10)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Mean-accuracy MAE (pp), BCE <= 0.003")
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "e36_ext_stp_steps.png", dpi=180)
    plt.close(fig)
    wide = read("wide_deep.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    methods = [("smc", None, "Enhanced SMC", "#20282c"),
               ("sgd", .001, "SGD lr=0.001", "#167b73"),
               ("sgd", .0001, "SGD lr=0.0001", "#858100"),
               ("stp", .003, "STP scale=0.003", "#a94762"),
               ("stp", .001, "STP scale=0.001", "#406dab"),
               ("stp", .0003, "STP scale=0.0003", "#84558d")]
    for ax, case in zip(axes, ("partition0_n32_s0", "partition2_n32_s0")):
        for method, value, label, color in methods:
            selected = sorted([r for r in wide["rows"] if r["case"] == case and r["method"] == method
                               and r["value"] == value and r["epsilon"] <= .003 and r.get("accuracy") is not None],
                              key=lambda r: -r["epsilon"])
            complete = [r for r in selected if r["complete"]]
            ax.plot([r["epsilon"] for r in complete], [r["accuracy"] * 100 for r in complete],
                    "o-", markersize=3, color=color, label=label)
            for r in selected:
                if not r["complete"]:
                    ax.scatter(r["epsilon"], r["accuracy"] * 100, color=color, marker="x", s=40)
                    ax.annotate(f"{r['models']}/{r['planned']}", (r["epsilon"], r["accuracy"] * 100),
                                xytext=(0, 22), textcoords="offset points", fontsize=8,
                                arrowprops={"arrowstyle": "-", "color": color})
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(case.replace("_", " "))
        ax.set_xlabel("BCE upper threshold (smaller to the right)")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Mean single-model test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="outside upper center")
    fig.supxlabel("Both datasets: SGD lr=0.0001 reaches 0/128 at 1e-5; crosses denote reached subsets.", fontsize=9)
    fig.savefig(FIGURES / "e36_ext_wide_boundary.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cv_groups": groups, "gc": summary["direct_gc"],
                      "stp_clean": summary["stp_clean"], "figures": 3}, ensure_ascii=False))


if __name__ == "__main__":
    main()
