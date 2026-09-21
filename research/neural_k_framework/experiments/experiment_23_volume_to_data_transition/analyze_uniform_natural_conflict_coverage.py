"""离线分析 uniform MUX3 数据中的自然 shortcut 反证覆盖。

该脚本不训练模型、不运行 SMC。它从原实验固定 seed 重建每份 uniform
训练集，按 dataset path 做 block bootstrap、block permutation 和 grouped
cross-validation。使用 parse_known_args，直接粘贴到 Jupyter 也不会被 -f 参数打断。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


INPUT_BITS = 8
STATE_COUNT = 2 ** INPUT_BITS
TRAIN_COUNTS = tuple(range(32, 113, 8))
DATASETS_PER_N = 128
DATASET_SEED = 2026082901
DEFAULT_RESULT_DIR = Path(
    "/root/results_parity2_mux3_coverage_shortcut_intervention"
)
DEFAULT_BOOTSTRAP = 5_000
DEFAULT_PERMUTATIONS = 5_000
DEFAULT_SEED = 2026083001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 uniform MUX3 自然冲突覆盖与恢复率。"
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"忽略宿主附加参数：{unknown}", flush=True)
    return args


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rendered: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    value = json.dumps(
                        json_ready(value), ensure_ascii=False, separators=(",", ":")
                    )
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                rendered[key] = value
            writer.writerow(rendered)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth_table_inputs() -> np.ndarray:
    values = np.arange(STATE_COUNT, dtype=np.uint32)
    shifts = np.arange(INPUT_BITS - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def target_outputs(inputs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "parity2": np.bitwise_xor.reduce(inputs[:, :2], axis=1).astype(np.uint8),
        "mux3": np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]).astype(
            np.uint8
        ),
    }


def uniform_order(dataset_index: int) -> np.ndarray:
    rng = np.random.default_rng(DATASET_SEED + dataset_index)
    return rng.permutation(STATE_COUNT).astype(np.int64)


def dataset_signature(indices: np.ndarray) -> str:
    payload = (
        b"uniform_random"
        + np.asarray(sorted(map(int, indices)), dtype=np.uint16).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def cell_counts(cell_ids: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(cell_ids.astype(np.int64), minlength=count).astype(np.int64)


def reconstruct_features(n: int, dataset_index: int) -> dict[str, Any]:
    inputs = truth_table_inputs()
    outputs = target_outputs(inputs)
    indices = np.sort(uniform_order(dataset_index)[:n])
    selected = inputs[indices]
    mux_labels = outputs["mux3"][indices]
    parity_labels = outputs["parity2"][indices]

    mux_cells = 4 * selected[:, 0] + 2 * selected[:, 1] + selected[:, 2]
    mux_counts = cell_counts(mux_cells, 8)
    parity_cells = 2 * selected[:, 0] + selected[:, 1]
    parity_counts = cell_counts(parity_cells, 4)

    conflict = selected[:, 1] != selected[:, 2]
    refute_copy_x1 = mux_labels != selected[:, 1]
    refute_copy_x2 = mux_labels != selected[:, 2]
    conflict_ids = (4 * selected[:, 0] + 2 * selected[:, 1] + selected[:, 2])[
        conflict
    ]
    conflict_cell_order = np.asarray([1, 2, 5, 6], dtype=np.int64)
    conflict_counts = np.asarray(
        [int(np.sum(conflict_ids == cell)) for cell in conflict_cell_order],
        dtype=np.int64,
    )
    d1 = int(refute_copy_x1.sum())
    d2 = int(refute_copy_x2.sum())
    return {
        "n": n,
        "dataset_index": dataset_index,
        "dataset_signature_reconstructed": dataset_signature(indices),
        "copy_x1_refutation_count": d1,
        "copy_x2_refutation_count": d2,
        "copy_refutation_min": min(d1, d2),
        "copy_refutation_total": d1 + d2,
        "copy_refutation_absdiff": abs(d1 - d2),
        "copy_refutation_min_fraction": min(d1, d2) / n,
        "conflict_count": int(conflict.sum()),
        "conflict_fraction_reconstructed": float(conflict.mean()),
        "distinct_conflict_cells": int(np.count_nonzero(conflict_counts)),
        "minimum_conflict_cell_count": int(conflict_counts.min()),
        "maximum_conflict_cell_count": int(conflict_counts.max()),
        "conflict_cell_count_std": float(conflict_counts.std()),
        "mux_distinct_cells": int(np.count_nonzero(mux_counts)),
        "mux_minimum_cell_count": int(mux_counts.min()),
        "mux_maximum_cell_count": int(mux_counts.max()),
        "mux_cell_count_std": float(mux_counts.std()),
        "parity_distinct_cells": int(np.count_nonzero(parity_counts)),
        "parity_minimum_cell_count": int(parity_counts.min()),
        "parity_maximum_cell_count": int(parity_counts.max()),
        "parity_cell_count_std": float(parity_counts.std()),
        "mux_label_fraction": float(mux_labels.mean()),
        "mux_label_imbalance": float(abs(mux_labels.mean() - 0.5)),
        "parity_label_fraction": float(parity_labels.mean()),
        "parity_label_imbalance": float(abs(parity_labels.mean() - 0.5)),
        "copy_x1_accuracy_reconstructed": float(
            np.mean(selected[:, 1] == mux_labels)
        ),
        "copy_x2_accuracy_reconstructed": float(
            np.mean(selected[:, 2] == mux_labels)
        ),
        "conflict_cell_counts": conflict_counts.tolist(),
        "mux_cell_counts": mux_counts.tolist(),
        "parity_cell_counts": parity_counts.tolist(),
    }


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_analysis_rows(source_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in source_rows
        if row.get("optimizer_name") == "adamw"
        and row.get("sampling_protocol") == "uniform_random"
        and int(row.get("step", "-1")) == 40_000
        and row.get("target_name") in {"parity2", "mux3"}
    ]
    expected = len(TRAIN_COUNTS) * DATASETS_PER_N * 2
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} uniform rows, found {len(selected)}")
    feature_cache = {
        (n, dataset_index): reconstruct_features(n, dataset_index)
        for n in TRAIN_COUNTS
        for dataset_index in range(DATASETS_PER_N)
    }
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    outcome_float_fields = (
        "target_function_mass",
        "modal_probability",
        "function_collision",
        "unseen_bit_agreement",
        "unseen_target_bit_accuracy_mean",
        "train_fit_rate",
    )
    for row in selected:
        target = str(row["target_name"])
        n = int(row["n"])
        dataset_index = int(row["dataset_index"])
        key = (target, n, dataset_index)
        if key in seen:
            raise RuntimeError(f"duplicate row {key}")
        seen.add(key)
        features = dict(feature_cache[(n, dataset_index)])
        if features["dataset_signature_reconstructed"] != row["dataset_signature"]:
            raise RuntimeError(
                f"signature mismatch target={target} n={n} dataset={dataset_index}"
            )
        if not math.isclose(
            float(features["conflict_fraction_reconstructed"]),
            float(row["conflict_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"conflict fraction mismatch {key}")
        if target == "mux3":
            label_imbalance = features["mux_label_imbalance"]
            relevant_std = features["mux_cell_count_std"]
            relevant_min = features["mux_minimum_cell_count"]
        else:
            label_imbalance = features["parity_label_imbalance"]
            relevant_std = features["parity_cell_count_std"]
            relevant_min = features["parity_minimum_cell_count"]
        output.append({
            **features,
            "target_name": target,
            "label_imbalance": label_imbalance,
            "relevant_cell_count_std": relevant_std,
            "relevant_minimum_cell_count": relevant_min,
            "dataset_recovered": parse_bool(row["dataset_recovered"]),
            "modal_is_target": parse_bool(row["modal_is_target"]),
            **{field: parse_float(row[field]) for field in outcome_float_fields},
        })
    return sorted(
        output,
        key=lambda row: (str(row["target_name"]), int(row["n"]), int(row["dataset_index"])),
    )


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rank_average(x), rank_average(y))


def matrix_for(
    rows: Sequence[dict[str, Any]],
    target: str,
    field: str,
) -> np.ndarray:
    lookup = {
        (int(row["n"]), int(row["dataset_index"])): float(row[field])
        for row in rows
        if row["target_name"] == target
    }
    return np.asarray([
        [lookup[(n, dataset_index)] for dataset_index in range(DATASETS_PER_N)]
        for n in TRAIN_COUNTS
    ], dtype=np.float64)


def feature_matrix(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> np.ndarray:
    lookup = {
        (int(row["n"]), int(row["dataset_index"])): float(row[field])
        for row in rows
        if row["target_name"] == "mux3"
    }
    return np.asarray([
        [lookup[(n, dataset_index)] for dataset_index in range(DATASETS_PER_N)]
        for n in TRAIN_COUNTS
    ], dtype=np.float64)


def equal_n_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    values: list[float] = []
    for local_x, local_y in zip(x, y):
        value = spearman(local_x, local_y)
        if math.isfinite(value):
            values.append(value)
    return (
        float(np.mean(values)) if values else float("nan"),
        values,
    )


def percentile_interval(values: Sequence[float]) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if not len(finite):
        return [float("nan"), float("nan")]
    return list(map(float, np.quantile(finite, [0.025, 0.975])))


def block_inference(
    x: np.ndarray,
    mux_y: np.ndarray,
    parity_y: np.ndarray,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed_mux, per_n_mux = equal_n_spearman(x, mux_y)
    observed_parity, per_n_parity = equal_n_spearman(x, parity_y)
    observed_difference = observed_mux - observed_parity
    rng = np.random.default_rng(seed)
    boot_mux: list[float] = []
    boot_parity: list[float] = []
    boot_difference: list[float] = []
    for _ in range(bootstrap):
        sample = rng.integers(0, DATASETS_PER_N, size=DATASETS_PER_N)
        local_mux, _ = equal_n_spearman(x[:, sample], mux_y[:, sample])
        local_parity, _ = equal_n_spearman(x[:, sample], parity_y[:, sample])
        boot_mux.append(local_mux)
        boot_parity.append(local_parity)
        boot_difference.append(local_mux - local_parity)
    null_mux: list[float] = []
    for _ in range(permutations):
        permutation = rng.permutation(DATASETS_PER_N)
        value, _ = equal_n_spearman(x, mux_y[:, permutation])
        null_mux.append(value)
    finite_null = np.asarray([value for value in null_mux if math.isfinite(value)])
    p_value = float(
        (1 + np.sum(finite_null >= observed_mux)) / (1 + len(finite_null))
    )
    return {
        "mux_equal_n_spearman": observed_mux,
        "mux_per_n_spearman": per_n_mux,
        "mux_bootstrap_95": percentile_interval(boot_mux),
        "mux_block_permutation_p_one_sided": p_value,
        "parity_equal_n_spearman": observed_parity,
        "parity_per_n_spearman": per_n_parity,
        "parity_bootstrap_95": percentile_interval(boot_parity),
        "mux_minus_parity_spearman": observed_difference,
        "difference_bootstrap_95": percentile_interval(boot_difference),
    }


def n_design(n_values: np.ndarray) -> np.ndarray:
    levels = list(TRAIN_COUNTS)
    return np.stack(
        [(n_values == level).astype(np.float64) for level in levels], axis=1
    )


def standardized_features(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train_values - mean) / scale, (test_values - mean) / scale


def build_design(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_values = np.asarray([int(row["n"]) for row in rows], dtype=np.int64)
    train_n = n_design(n_values[train_mask])
    test_n = n_design(n_values[test_mask])
    if not feature_names:
        return train_n, test_n
    values = np.asarray([
        [float(row[name]) for name in feature_names] for row in rows
    ], dtype=np.float64)
    train_features, test_features = standardized_features(
        values[train_mask], values[test_mask]
    )
    return (
        np.concatenate([train_n, train_features], axis=1),
        np.concatenate([test_n, test_features], axis=1),
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float = 1e-3) -> np.ndarray:
    regularizer = np.eye(x.shape[1], dtype=np.float64) * penalty
    regularizer[: len(TRAIN_COUNTS), : len(TRAIN_COUNTS)] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float = 1e-2,
    max_iterations: int = 100,
) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=np.float64)
    regularizer = np.eye(x.shape[1], dtype=np.float64) * penalty
    regularizer[: len(TRAIN_COUNTS), : len(TRAIN_COUNTS)] = 0.0
    for _ in range(max_iterations):
        probability = sigmoid(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        gradient = x.T @ (y - probability) - regularizer @ beta
        hessian = (x.T * weight) @ x + regularizer
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta += update
        if np.max(np.abs(update)) < 1e-8:
            break
    return beta


def grouped_cross_validation(
    rows: Sequence[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    groups = rng.permutation(DATASETS_PER_N)
    folds = np.array_split(groups, 5)
    output: list[dict[str, Any]] = []
    for target in ("mux3", "parity2"):
        generic_features = (
            "label_imbalance",
            "relevant_cell_count_std",
            "relevant_minimum_cell_count",
        )
        # 对 parity2 额外控制同一输入集的 MUX 八格覆盖；对 MUX3 而言
        # relevant_cell_* 本身就是八格统计，不能重复加入同一列。
        if target == "parity2":
            generic_features += (
                "mux_cell_count_std",
                "mux_minimum_cell_count",
            )
        feature_sets = {
            "n_only": (),
            "generic": generic_features,
            "shortcut": generic_features + (
                "copy_refutation_min",
                "copy_refutation_absdiff",
                "distinct_conflict_cells",
                "minimum_conflict_cell_count",
            ),
        }
        local_rows = [row for row in rows if row["target_name"] == target]
        dataset_indices = np.asarray(
            [int(row["dataset_index"]) for row in local_rows], dtype=np.int64
        )
        mass = np.asarray(
            [float(row["target_function_mass"]) for row in local_rows],
            dtype=np.float64,
        )
        recovered = np.asarray(
            [float(bool(row["dataset_recovered"])) for row in local_rows],
            dtype=np.float64,
        )
        for model_name, feature_names in feature_sets.items():
            mass_predictions = np.empty(len(local_rows), dtype=np.float64)
            recovery_predictions = np.empty(len(local_rows), dtype=np.float64)
            for fold in folds:
                test_mask = np.isin(dataset_indices, fold)
                train_mask = ~test_mask
                train_x, test_x = build_design(
                    local_rows, feature_names, train_mask, test_mask
                )
                ridge = fit_ridge(train_x, mass[train_mask])
                mass_predictions[test_mask] = np.clip(test_x @ ridge, 0.0, 1.0)
                logistic = fit_logistic_ridge(train_x, recovered[train_mask])
                recovery_predictions[test_mask] = sigmoid(test_x @ logistic)
            mse = float(np.mean((mass - mass_predictions) ** 2))
            variance = float(np.mean((mass - mass.mean()) ** 2))
            r2 = 1.0 - mse / variance if variance > 0 else float("nan")
            clipped = np.clip(recovery_predictions, 1e-8, 1.0 - 1e-8)
            brier = float(np.mean((recovered - clipped) ** 2))
            log_loss = float(-np.mean(
                recovered * np.log(clipped)
                + (1.0 - recovered) * np.log(1.0 - clipped)
            ))
            output.append({
                "target_name": target,
                "model_name": model_name,
                "feature_names": list(feature_names),
                "mass_cv_mse": mse,
                "mass_cv_r2": r2,
                "recovery_cv_brier": brier,
                "recovery_cv_log_loss": log_loss,
            })
    return output


def stratified_curves(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in ("mux3", "parity2"):
        for n in TRAIN_COUNTS:
            local = [
                row for row in rows
                if row["target_name"] == target and int(row["n"]) == n
            ]
            ordered = sorted(
                local,
                key=lambda row: (
                    float(row["copy_refutation_min"]),
                    int(row["dataset_index"]),
                ),
            )
            groups = {
                "low_half": ordered[: len(ordered) // 2],
                "high_half": ordered[len(ordered) // 2 :],
            }
            for name, group in groups.items():
                output.append({
                    "target_name": target,
                    "n": n,
                    "dmin_group": name,
                    "dataset_count": len(group),
                    "copy_refutation_min_mean": float(np.mean([
                        float(row["copy_refutation_min"]) for row in group
                    ])),
                    "target_function_mass_mean": float(np.mean([
                        float(row["target_function_mass"]) for row in group
                    ])),
                    "dataset_recovery_rate": float(np.mean([
                        bool(row["dataset_recovered"]) for row in group
                    ])),
                    "modal_target_rate": float(np.mean([
                        bool(row["modal_is_target"]) for row in group
                    ])),
                    "unseen_target_accuracy_mean": float(np.mean([
                        float(row["unseen_target_bit_accuracy_mean"])
                        for row in group
                    ])),
                })
    return output


def lookup_cv(
    rows: Sequence[dict[str, Any]], target: str, model: str
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["target_name"] == target and row["model_name"] == model
    )


def save_plot(
    output_dir: Path,
    analysis_rows: Sequence[dict[str, Any]],
    curves: Sequence[dict[str, Any]],
    inference: dict[str, Any],
    cv_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"绘图跳过：{exc}", flush=True)
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"low_half": "#64748b", "high_half": "#dc2626"}
    for group in ("low_half", "high_half"):
        local = [
            row for row in curves
            if row["target_name"] == "mux3" and row["dmin_group"] == group
        ]
        axes[0, 0].plot(
            [row["n"] for row in local],
            [row["target_function_mass_mean"] for row in local],
            marker="o", label=group, color=colors[group],
        )
        axes[0, 1].plot(
            [row["n"] for row in local],
            [row["dataset_recovery_rate"] for row in local],
            marker="o", label=group, color=colors[group],
        )
    axes[0, 0].set_title("MUX3 target mass by natural d_min")
    axes[0, 1].set_title("MUX3 recovery by natural d_min")
    for axis in axes[0]:
        axis.set_xlabel("n")
        axis.set_ylim(-0.02, 1.02)
        axis.legend()

    x_values = []
    y_values = []
    for n in TRAIN_COUNTS:
        local = [
            row for row in analysis_rows
            if row["target_name"] == "mux3" and int(row["n"]) == n
        ]
        x = np.asarray([float(row["copy_refutation_min"]) for row in local])
        y = np.asarray([float(row["target_function_mass"]) for row in local])
        x_values.extend(x - x.mean())
        y_values.extend(y - y.mean())
    axes[1, 0].scatter(x_values, y_values, s=10, alpha=0.3, color="#2563eb")
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].axvline(0, color="black", lw=0.8)
    axes[1, 0].set_xlabel("d_min minus same-n mean")
    axes[1, 0].set_ylabel("target mass minus same-n mean")
    axes[1, 0].set_title(
        "Within-n association "
        f"rho={inference['mux_equal_n_spearman']:.3f}"
    )

    models = ["n_only", "generic", "shortcut"]
    x = np.arange(len(models))
    width = 0.35
    for offset, target in enumerate(("mux3", "parity2")):
        values = [
            float(lookup_cv(cv_rows, target, model)["mass_cv_r2"])
            for model in models
        ]
        axes[1, 1].bar(
            x + (offset - 0.5) * width,
            values,
            width=width,
            label=target,
            color="#dc2626" if target == "mux3" else "#2563eb",
        )
    axes[1, 1].set_xticks(x, models)
    axes[1, 1].set_ylabel("grouped-CV R² for target mass")
    axes[1, 1].set_title("Incremental predictive value")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "uniform_natural_conflict_analysis.png", dpi=180)
    plt.close(fig)


def package_output(output_dir: Path) -> Path:
    archive = output_dir.parent / f"{output_dir.name}_package.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir.parent))
    return archive


def main() -> None:
    args = parse_args()
    source = args.result_dir / "all_dataset_final.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir = args.output_dir or (
        args.result_dir / "uniform_natural_conflict_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== Uniform natural conflict coverage analysis ===", flush=True)
    print(f"source={source}", flush=True)
    print(f"output={output_dir}", flush=True)

    source_rows = read_csv(source)
    analysis_rows = build_analysis_rows(source_rows)
    write_csv(output_dir / "dataset_level_features_and_outcomes.csv", analysis_rows)

    dmin = feature_matrix(analysis_rows, "copy_refutation_min")
    mux_mass = matrix_for(analysis_rows, "mux3", "target_function_mass")
    parity_mass = matrix_for(analysis_rows, "parity2", "target_function_mass")
    inference = block_inference(
        dmin,
        mux_mass,
        parity_mass,
        args.bootstrap,
        args.permutations,
        args.seed,
    )
    cv_rows = grouped_cross_validation(analysis_rows, args.seed + 1)
    curves = stratified_curves(analysis_rows)
    write_json(output_dir / "block_inference.json", inference)
    write_csv(output_dir / "grouped_cross_validation.csv", cv_rows)
    write_csv(output_dir / "stratified_curves.csv", curves)

    mux_generic = lookup_cv(cv_rows, "mux3", "generic")
    mux_shortcut = lookup_cv(cv_rows, "mux3", "shortcut")
    ci_mux = inference["mux_bootstrap_95"]
    ci_difference = inference["difference_bootstrap_95"]
    p_value = float(inference["mux_block_permutation_p_one_sided"])
    cv_improved = (
        float(mux_shortcut["mass_cv_mse"]) < float(mux_generic["mass_cv_mse"])
        and float(mux_shortcut["recovery_cv_brier"])
        < float(mux_generic["recovery_cv_brier"])
    )
    if (
        float(ci_mux[0]) > 0
        and p_value < 0.01
        and float(ci_difference[0]) > 0
        and cv_improved
    ):
        verdict = "strong_support_for_natural_shortcut_discrimination"
    elif float(ci_mux[0]) > 0 and p_value < 0.05:
        verdict = "partial_support_natural_coverage_is_informative"
    else:
        verdict = "no_clear_support_in_uniform_natural_variation"
    summary = {
        "verdict": verdict,
        "source_row_count": len(source_rows),
        "uniform_analysis_row_count": len(analysis_rows),
        "independent_dataset_paths": DATASETS_PER_N,
        "nested_n_levels": list(TRAIN_COUNTS),
        "primary_feature": "copy_refutation_min",
        "primary_outcome": "mux3_target_function_mass",
        "block_inference": inference,
        "mux_generic_cv": mux_generic,
        "mux_shortcut_cv": mux_shortcut,
        "shortcut_model_improves_mux_mass_and_recovery": cv_improved,
        "interpretation_boundary": (
            "This is a uniform-sampling natural-variation analysis. "
            "The existing conflict-enriched intervention carries the causal claim."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    save_plot(output_dir, analysis_rows, curves, inference, cv_rows)
    archive = package_output(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
