#!/usr/bin/env python3
"""离线分析 E15 中 Grokking 前后的未见输入 Agreement。

主指标是 validation_seed_pairwise_agreement，不能使用包含训练输入的
full_seed_pairwise_agreement；后者会被已拟合训练点机械抬高。

阶段定义：
- hard fit：同一训练比例下全部 seed 的训练准确率首次不低于 0.999；
- grokking：平均 validation accuracy 首次不低于 0.90；
- accuracy baseline：若各 seed 以概率 a 给出同一个正确标签，其余错误标签
  在另外 C-1 类上独立均匀分布，则 pairwise agreement 为
  a^2 + (1-a)^2/(C-1)。

该脚本只重分析既有轨迹，不重新训练模型。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_FIT_THRESHOLD = 0.999
GROKKING_ACCURACY_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-zip",
        type=Path,
        required=True,
        help="E15 results_mod97_matched_loss_function_distribution_package.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="分析输出目录",
    )
    return parser.parse_args()


def read_trajectory(result_zip: Path) -> list[dict[str, str]]:
    if not result_zip.is_file():
        raise FileNotFoundError(result_zip)
    with zipfile.ZipFile(result_zip) as archive:
        with archive.open("trajectory.csv", "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    groups: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["train_fraction"]), int(row["step"]))].append(row)

    result: list[dict[str, float]] = []
    for (fraction, step), group in sorted(groups.items()):
        train = np.asarray([float(row["train_accuracy"]) for row in group])
        validation = np.asarray(
            [float(row["validation_accuracy"]) for row in group]
        )
        unseen_agreement = float(group[0]["validation_seed_pairwise_agreement"])
        full_agreement = float(group[0]["full_seed_pairwise_agreement"])
        mean_validation = float(validation.mean())
        class_count = 97
        accuracy_baseline = mean_validation**2 + (1.0 - mean_validation) ** 2 / (
            class_count - 1
        )
        result.append(
            {
                "train_fraction": fraction,
                "step": float(step),
                "mean_train_accuracy": float(train.mean()),
                "min_train_accuracy": float(train.min()),
                "mean_validation_accuracy": mean_validation,
                "unseen_pairwise_agreement": unseen_agreement,
                "full_domain_pairwise_agreement": full_agreement,
                "accuracy_baseline_agreement": accuracy_baseline,
                "excess_unseen_agreement": unseen_agreement - accuracy_baseline,
                "modal_function_accuracy": float(
                    group[0]["full_modal_function_accuracy"]
                ),
                "target_function_probability": float(
                    group[0]["full_target_function_probability"]
                ),
            }
        )
    return result


def first_where(rows: list[dict[str, float]], predicate):
    return next((row for row in rows if predicate(row)), None)


def previous_row(rows: list[dict[str, float]], step: float):
    previous = [row for row in rows if row["step"] < step]
    return previous[-1] if previous else None


def build_milestones(aggregated: list[dict[str, float]]) -> list[dict[str, object]]:
    by_fraction: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in aggregated:
        by_fraction[row["train_fraction"]].append(row)

    milestones: list[dict[str, object]] = []
    for fraction, rows in sorted(by_fraction.items()):
        rows.sort(key=lambda row: row["step"])
        hard_fit = first_where(
            rows, lambda row: row["min_train_accuracy"] >= TRAIN_FIT_THRESHOLD
        )
        grokking = first_where(
            rows,
            lambda row: row["mean_validation_accuracy"]
            >= GROKKING_ACCURACY_THRESHOLD,
        )
        selected = [("hard_fit", hard_fit)]
        if grokking is not None:
            selected.extend(
                [
                    ("pre_grokking", previous_row(rows, grokking["step"])),
                    ("grokking", grokking),
                ]
            )
        selected.append(("final", rows[-1]))
        for name, row in selected:
            if row is None:
                continue
            milestones.append(
                {
                    "train_fraction": fraction,
                    "milestone": name,
                    **row,
                }
            )
    return milestones


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(aggregated: list[dict[str, float]], output_path: Path) -> None:
    by_fraction: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in aggregated:
        by_fraction[row["train_fraction"]].append(row)

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    colors = {0.6: "#5B8FF9", 0.7: "#61DDAA", 0.8: "#F6BD16", 0.9: "#E8684A"}
    for fraction, rows in sorted(by_fraction.items()):
        rows.sort(key=lambda row: row["step"])
        steps = np.asarray([row["step"] for row in rows])
        val = np.asarray([row["mean_validation_accuracy"] for row in rows])
        agreement = np.asarray([row["unseen_pairwise_agreement"] for row in rows])
        baseline = np.asarray([row["accuracy_baseline_agreement"] for row in rows])
        excess = agreement - baseline
        color = colors.get(fraction)
        label = f"train={fraction:.0%}"

        axes[0, 0].plot(steps, val, color=color, label=label)
        axes[0, 1].plot(steps, agreement, color=color, label=label)
        axes[1, 0].plot(val, agreement, color=color, label=label)
        axes[1, 1].plot(steps, excess, color=color, label=label)

    grid = np.linspace(0.0, 1.0, 400)
    baseline_grid = grid**2 + (1.0 - grid) ** 2 / 96.0
    axes[1, 0].plot(
        grid,
        baseline_grid,
        linestyle="--",
        color="#444444",
        label="accuracy-only baseline",
    )
    chance = 1.0 / 97.0
    axes[0, 1].axhline(chance, linestyle=":", color="#777777", label="1/97")
    axes[1, 1].axhline(0.0, linestyle=":", color="#777777")

    axes[0, 0].set_title("Unseen accuracy over training")
    axes[0, 1].set_title("Unseen pairwise Agreement")
    axes[1, 0].set_title("Agreement versus unseen accuracy")
    axes[1, 1].set_title("Agreement beyond accuracy baseline")
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 1].set_ylabel("pairwise Agreement")
    axes[1, 0].set_xlabel("unseen accuracy")
    axes[1, 0].set_ylabel("pairwise Agreement")
    axes[1, 1].set_ylabel("excess Agreement")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        axis.set_xscale("symlog", linthresh=100.0)
        axis.set_xlabel("optimization step")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_trajectory(args.result_zip)
    aggregated = aggregate(rows)
    milestones = build_milestones(aggregated)
    write_csv(args.output_dir / "agreement_trajectory_aggregate.csv", aggregated)
    write_csv(args.output_dir / "agreement_grokking_milestones.csv", milestones)
    (args.output_dir / "agreement_grokking_milestones.json").write_text(
        json.dumps(milestones, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(aggregated, args.output_dir / "agreement_around_grokking.png")
    print(json.dumps(milestones, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
