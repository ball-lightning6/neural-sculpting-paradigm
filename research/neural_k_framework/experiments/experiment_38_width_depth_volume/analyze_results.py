"""E38只读结果汇总。先运行同目录driver，再运行本脚本；不启动训练或采样。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results_4bit_smc_width_depth"
OUTPUT_DIR = HERE / "analysis"
WIDTH_CELLS = ("w64_d2", "w128_d2", "w256_d2", "w512_d2")
CELLS = (*WIDTH_CELLS, "w256_d1")
REFERENCE = "fn_0x0F0F"
EXAMPLE = "fn_0xAE8C"
THRESHOLD = 0.30


def summarize() -> dict:
    hashes = {}

    def read(path: Path):
        raw = path.read_bytes()
        hashes[path.relative_to(RESULT_ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("utf-8-sig"))

    cells = {}
    threshold_rows = 0
    for name in CELLS:
        directory = RESULT_ROOT / name
        config = read(directory / "config.json")
        overall = read(directory / "summary.json")
        if overall["status"] != "completed":
            raise ValueError(f"{name}尚未完成，不能作为完整结果汇总。")
        targets = {}
        for target_name in config["TARGET_NAMES"]:
            record = read(directory / "targets" / target_name / "summary.json")
            rows = record["threshold_rows"]
            if record["status"] != "completed" or record["completed_thresholds"] != len(rows):
                raise ValueError(f"{name}/{target_name}尚未完成。")
            if [r["threshold"] for r in rows] != config["TARGET_THRESHOLDS"]:
                raise ValueError(f"{name}/{target_name}阈值网格不完整。")
            for row in rows:
                threshold_rows += 1
                if not math.isfinite(row["estimated_log_volume_fraction_median"]):
                    raise ValueError("发现非有限log-volume。")
                if row["loss_max"] > row["threshold"] + 1e-6:
                    raise ValueError("保存的粒子loss超过约束容差。")
            selected = [r for r in rows if math.isclose(r["threshold"], THRESHOLD)]
            if len(selected) != 1:
                raise ValueError(f"{name}/{target_name}缺少唯一的比较阈值。")
            row = selected[0]
            targets[target_name] = {
                "cost_log10": -row["estimated_log_volume_fraction_median"] / math.log(10),
                "target_hard_frequency": row["target_probability"],
                "minimum_lineages": row["unique_lineages_min"],
                "pool_log10_mass_range": row["replica_log_volume_range"] / math.log(10),
            }
        baseline = targets[REFERENCE]["cost_log10"]
        for target in targets.values():
            target["gap_log10"] = target["cost_log10"] - baseline
        cells[name] = {
            "width": config["WIDTH"], "hidden_layers": config["HIDDEN_LAYERS"],
            "parameter_count": overall["parameter_count"],
            "replicas": config["REPLICAS"],
            "particles_per_replica": config["PARTICLES_PER_REPLICA"],
            "targets": targets,
        }
    names = tuple(cells[WIDTH_CELLS[0]]["targets"])
    if any(tuple(c["targets"]) != names for c in cells.values()):
        raise ValueError("不同配置的目标集合或顺序不一致。")
    selected_names = [n for n in names if n not in (REFERENCE, "fn_0xCCCC")]
    gaps = {
        name: [cells[cell]["targets"][name]["gap_log10"] for cell in WIDTH_CELLS]
        for name in selected_names
    }
    increasing = [name for name, values in gaps.items()
                  if all(b > a for a, b in zip(values, values[1:]))]
    return {
        "threshold": THRESHOLD,
        "reference": REFERENCE,
        "example": EXAMPLE,
        "completed_targets": sum(len(c["targets"]) for c in cells.values()),
        "checked_threshold_rows": threshold_rows,
        "monotone_count": len(increasing),
        "compared_target_count": len(selected_names),
        "monotone_targets": increasing,
        "gaps_by_target": gaps,
        "median_gap_by_width": [
            median(cells[cell]["targets"][n]["gap_log10"] for n in selected_names)
            for cell in WIDTH_CELLS
        ],
        "cells": cells,
        "source_sha256": hashes,
        "interpretation": "当前协议下完整目标loss事件的相对质量，不是测度无关复杂度或hard cell质量；池间跨度不是独立置信区间。",
    }


def make_plot(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    widths = [report["cells"][name]["width"] for name in WIDTH_CELLS]
    highlighted = {"fn_0xAE8C": ("#b33b4b", "0xAE8C"),
                   "fn_0x055F": ("#147d72", "0x055F")}
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    first = True
    for name, values in report["gaps_by_target"].items():
        if name in highlighted:
            continue
        ax.plot(widths, values, color="#92979c", alpha=0.6, lw=1.2,
                label="Other selected targets (15)" if first else None)
        first = False
    for name, (color, label) in highlighted.items():
        ax.plot(widths, report["gaps_by_target"][name], "o-", color=color, lw=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Hidden width (two hidden layers)")
    ax.set_ylabel("log10 mass ratio: reference / target")
    ax.set_title("Full-target mean BCE <= 0.30; reference: 0x0F0F")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "width_relative_mass.png")
    plt.close(fig)


def main() -> None:
    report = summarize()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(report)
    for name in WIDTH_CELLS:
        targets = report["cells"][name]["targets"]
        print(f"{name}: reference={targets[REFERENCE]['cost_log10']:.4f}, "
              f"example={targets[EXAMPLE]['cost_log10']:.4f}, "
              f"gap={targets[EXAMPLE]['gap_log10']:.4f}")
    print(f"完成目标={report['completed_targets']}，阈值记录={report['checked_threshold_rows']}，"
          f"单调目标={report['monotone_count']}/{report['compared_target_count']}")


if __name__ == "__main__":
    main()
