"""用 E23 的多随机训练集 agreement 曲线细化样本相变位置。

该脚本是对既有 E23 结果的后验诊断，不是原预注册主判决。它利用每个 n、
每条规则的 64 份随机训练集和各自 24-seed committee：

1. 跳过小 n 错误常数造成的高 agreement；
2. 从 agreement 全局最低点后的“重新凝聚支”开始；
3. 对 mean unseen agreement 与 target accuracy 分别做 isotonic 拟合；
4. 线性插值估计 agreement 阈值的亚网格 crossing；
5. 通过 dataset bootstrap 给出区间，并在相同 n 上做规则间配对比较。

默认 agreement 阈值为 0.95/0.99/0.995，目标准确率守门阈值为 0.90。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TARGETS = (
    "parity2",
    "mux3",
    "parity3",
    "parity4",
    "random_balanced",
)

EXPECTED_PAIRS = (
    ("parity2", "mux3"),
    ("parity3", "random_balanced"),
    ("parity4", "random_balanced"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/root/results_8bit_volume_to_data_transition"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026082704)
    parser.add_argument(
        "--agreement-thresholds",
        type=float,
        nargs="+",
        default=(0.95, 0.99, 0.995),
    )
    parser.add_argument("--accuracy-threshold", type=float, default=0.90)
    args, unknown = parser.parse_known_args()
    if unknown:
        is_jupyter_kernel_arg = (
            len(unknown) == 2
            and unknown[0] == "-f"
            and unknown[1].endswith(".json")
        )
        if not is_jupyter_kernel_arg:
            parser.error("unrecognized arguments: " + " ".join(unknown))
    return args


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> dict[str, dict[int, list[dict[str, float]]]]:
    grouped: dict[str, dict[int, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in TARGETS
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row["target_name"]
            if name not in grouped:
                continue
            n = int(row["n"])
            if n >= 2 ** 8 or not row.get("unseen_bit_agreement"):
                continue
            grouped[name][n].append({
                "dataset_index": float(row["dataset_index"]),
                "agreement": float(row["unseen_bit_agreement"]),
                "accuracy": float(row["unseen_target_bit_accuracy_mean"]),
                "target_mass": float(row["target_function_mass"]),
                "collision": float(row["function_collision"]),
            })
    return grouped


def isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """等权 PAVA；输出长度与输入一致。"""
    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(values.astype(np.float64)):
        blocks.append({
            "start": index,
            "stop": index + 1,
            "weight": 1.0,
            "sum": float(value),
        })
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["sum"]) / float(left["weight"])
            right_mean = float(right["sum"]) / float(right["weight"])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [{
                "start": int(left["start"]),
                "stop": int(right["stop"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "sum": float(left["sum"]) + float(right["sum"]),
            }]
    result = np.empty(len(values), dtype=np.float64)
    for block in blocks:
        result[int(block["start"]):int(block["stop"])] = (
            float(block["sum"]) / float(block["weight"])
        )
    return result


def interpolate_crossing(
    counts: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float:
    indices = np.flatnonzero(values >= threshold)
    if not len(indices):
        return math.inf
    index = int(indices[0])
    if index == 0:
        return float(counts[0])
    x0, x1 = float(counts[index - 1]), float(counts[index])
    y0, y1 = float(values[index - 1]), float(values[index])
    if y1 <= y0 + 1e-15:
        return x1
    weight = min(max((threshold - y0) / (y1 - y0), 0.0), 1.0)
    return x0 + weight * (x1 - x0)


def summarize_curve(
    means: dict[int, tuple[float, float]],
    agreement_threshold: float,
    accuracy_threshold: float,
) -> dict[str, float | int | bool | None]:
    counts = np.asarray(sorted(means), dtype=np.float64)
    agreement = np.asarray([means[int(n)][0] for n in counts])
    accuracy = np.asarray([means[int(n)][1] for n in counts])
    minimum_index = int(np.argmin(agreement))
    branch_counts = counts[minimum_index:]
    agreement_iso = isotonic_increasing(agreement[minimum_index:])
    accuracy_iso = isotonic_increasing(accuracy[minimum_index:])
    agreement_cross = interpolate_crossing(
        branch_counts, agreement_iso, agreement_threshold
    )
    accuracy_cross = interpolate_crossing(
        branch_counts, accuracy_iso, accuracy_threshold
    )
    combined = max(agreement_cross, accuracy_cross)
    return {
        "minimum_agreement_n": int(counts[minimum_index]),
        "minimum_agreement": float(agreement[minimum_index]),
        "agreement_cross": (
            float(agreement_cross) if math.isfinite(agreement_cross) else None
        ),
        "accuracy_cross": (
            float(accuracy_cross) if math.isfinite(accuracy_cross) else None
        ),
        "combined_cross": float(combined) if math.isfinite(combined) else None,
        "right_censored": not math.isfinite(combined),
        "last_n": int(counts[-1]),
        "last_agreement": float(agreement[-1]),
        "last_accuracy": float(accuracy[-1]),
    }


def group_means(
    groups: dict[int, list[dict[str, float]]],
    rng: np.random.Generator | None = None,
) -> dict[int, tuple[float, float]]:
    result = {}
    for n, rows in groups.items():
        indices = np.arange(len(rows))
        if rng is not None:
            indices = rng.choice(indices, size=len(indices), replace=True)
        agreement = np.asarray([rows[i]["agreement"] for i in indices])
        accuracy = np.asarray([rows[i]["accuracy"] for i in indices])
        result[n] = (float(agreement.mean()), float(accuracy.mean()))
    return result


def quantile_or_none(values: list[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    return float(np.quantile(finite, q)) if len(finite) else None


def bootstrap_crossings(
    groups: dict[int, list[dict[str, float]]],
    agreement_threshold: float,
    accuracy_threshold: float,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = []
    for _ in range(replicates):
        summary = summarize_curve(
            group_means(groups, rng),
            agreement_threshold,
            accuracy_threshold,
        )
        value = summary["combined_cross"]
        values.append(float(value) if value is not None else math.inf)
    return {
        "bootstrap_median": quantile_or_none(values, 0.5),
        "bootstrap_q025": quantile_or_none(values, 0.025),
        "bootstrap_q975": quantile_or_none(values, 0.975),
        "bootstrap_right_censored_fraction": float(np.mean(
            ~np.isfinite(np.asarray(values))
        )),
    }


def paired_same_n_rows(
    data: dict[str, dict[int, list[dict[str, float]]]],
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    output = []
    for easy, hard in EXPECTED_PAIRS:
        common = sorted(set(data[easy]) & set(data[hard]))
        for n in common:
            easy_map = {
                int(row["dataset_index"]): row for row in data[easy][n]
            }
            hard_map = {
                int(row["dataset_index"]): row for row in data[hard][n]
            }
            indices = np.asarray(sorted(set(easy_map) & set(hard_map)))
            differences = np.asarray([
                easy_map[i]["agreement"] - hard_map[i]["agreement"]
                for i in indices
            ])
            bootstrap = []
            for _ in range(replicates):
                selected = rng.choice(
                    len(differences), size=len(differences), replace=True
                )
                bootstrap.append(float(differences[selected].mean()))
            easy_accuracy = np.mean([
                easy_map[i]["accuracy"] for i in indices
            ])
            hard_accuracy = np.mean([
                hard_map[i]["accuracy"] for i in indices
            ])
            output.append({
                "easy": easy,
                "hard": hard,
                "n": n,
                "dataset_count": len(indices),
                "agreement_difference_easy_minus_hard": float(
                    differences.mean()
                ),
                "agreement_difference_q025": float(np.quantile(
                    bootstrap, 0.025
                )),
                "agreement_difference_q975": float(np.quantile(
                    bootstrap, 0.975
                )),
                "easy_accuracy": float(easy_accuracy),
                "hard_accuracy": float(hard_accuracy),
                "both_target_aligned_90pct": bool(
                    easy_accuracy >= 0.90 and hard_accuracy >= 0.90
                ),
            })
    return output


def save_plot(
    output: Path,
    data: dict[str, dict[int, list[dict[str, float]]]],
    crossing_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in TARGETS:
        means = group_means(data[name])
        n = np.asarray(sorted(means))
        agreement = [means[int(value)][0] for value in n]
        accuracy = [means[int(value)][1] for value in n]
        axes[0].plot(n, agreement, marker="o", ms=3, label=name)
        axes[1].plot(n, accuracy, marker="o", ms=3, label=name)
    axes[0].axhline(0.99, color="black", ls="--", lw=1)
    axes[0].set_title("unseen agreement: split then reconcentrate")
    axes[0].set_xlabel("training sample count")
    axes[0].set_ylabel("mean agreement")
    axes[1].set_title("unseen target accuracy guard")
    axes[1].set_xlabel("training sample count")
    axes[1].set_ylabel("mean target accuracy")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.set_ylim(0.45, 1.01)
    figure.tight_layout()
    figure.savefig(output / "agreement_subgrid_transitions.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    source = args.result_dir / "all_dataset_final.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    output = args.output_dir or (
        args.result_dir / "agreement_subgrid_analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    data = load_rows(source)
    rng = np.random.default_rng(args.seed)

    crossing_rows = []
    for threshold in args.agreement_thresholds:
        for name in TARGETS:
            point = summarize_curve(
                group_means(data[name]),
                threshold,
                args.accuracy_threshold,
            )
            boot = bootstrap_crossings(
                data[name], threshold, args.accuracy_threshold,
                args.bootstrap, rng,
            )
            crossing_rows.append({
                "target_name": name,
                "agreement_threshold": threshold,
                "accuracy_threshold": args.accuracy_threshold,
                **point,
                **boot,
            })

    paired_rows = paired_same_n_rows(data, args.bootstrap, rng)
    write_csv(output / "agreement_subgrid_crossings.csv", crossing_rows)
    write_csv(output / "paired_same_n_agreement.csv", paired_rows)
    write_json(output / "summary.json", {
        "analysis_status": "posthoc_subgrid_diagnostic",
        "source": str(source),
        "dataset_replicates_per_n": 64,
        "model_seeds_per_dataset": 24,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "method": (
            "post-minimum isotonic agreement crossing with target-accuracy "
            "guard; linear interpolation between sample-count grid points"
        ),
        "crossings": crossing_rows,
    })
    save_plot(output, data, crossing_rows)

    print("=== Agreement sub-grid transition diagnostic ===")
    for row in crossing_rows:
        if math.isclose(float(row["agreement_threshold"]), 0.99):
            estimate = (
                row["combined_cross"]
                if not row["right_censored"] else f">{row['last_n']}"
            )
            print(
                f"{row['target_name']:<16} agreement99={estimate} "
                f"bootstrap=[{row['bootstrap_q025']}, {row['bootstrap_q975']}]"
            )
    print(f"output={output}")


if __name__ == "__main__":
    main()
