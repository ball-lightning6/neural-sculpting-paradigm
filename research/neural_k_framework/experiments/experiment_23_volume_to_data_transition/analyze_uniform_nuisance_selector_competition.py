"""检验8-bit MUX3 的冗余位假 selector 是否解释额外恢复差异。

只读取原 `all_dataset_final.csv`，不训练、不运行 SMC。所有重采样都按
dataset_index 的 nested path 分块。parse_known_args 兼容 Jupyter 的 `-f`。
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
DEFAULT_BOOTSTRAP = 3_000
DEFAULT_PERMUTATIONS = 3_000
DEFAULT_SEED = 2026083002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 uniform MUX3 的 nuisance-selector 竞争。"
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
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rendered = {}
            for key in fields:
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


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def truth_table_inputs() -> np.ndarray:
    values = np.arange(STATE_COUNT, dtype=np.uint32)
    shifts = np.arange(INPUT_BITS - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def uniform_order(dataset_index: int) -> np.ndarray:
    rng = np.random.default_rng(DATASET_SEED + dataset_index)
    return rng.permutation(STATE_COUNT).astype(np.int64)


def dataset_signature(indices: np.ndarray) -> str:
    payload = (
        b"uniform_random"
        + np.asarray(sorted(map(int, indices)), dtype=np.uint16).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def mux_outputs(inputs: np.ndarray) -> np.ndarray:
    return np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]).astype(
        np.uint8
    )


def parity_outputs(inputs: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(inputs[:, 0], inputs[:, 1]).astype(np.uint8)


def selector_error_metrics(selected: np.ndarray) -> dict[str, Any]:
    conflict = selected[:, 1] != selected[:, 2]
    true_selector = selected[conflict, 0]
    nuisance = selected[conflict, 3:]
    conflict_count = len(true_selector)
    if conflict_count == 0:
        raise RuntimeError("uniform dataset unexpectedly has no MUX conflict sample")

    copy_errors = np.asarray([
        int(np.sum(true_selector != 0)),
        int(np.sum(true_selector != 1)),
    ], dtype=np.int64)
    copy_best_errors = int(copy_errors.min())

    literal_names: list[str] = []
    literal_errors: list[int] = []
    for offset in range(nuisance.shape[1]):
        variable = offset + 3
        values = nuisance[:, offset]
        literal_names.extend([f"x{variable}", f"NOT_x{variable}"])
        literal_errors.extend([
            int(np.sum(values != true_selector)),
            int(np.sum((1 - values) != true_selector)),
        ])
    literal_errors_array = np.asarray(literal_errors, dtype=np.int64)
    literal_best_index = int(np.argmin(literal_errors_array))
    literal_best_errors = int(literal_errors_array[literal_best_index])
    literal_or_copy_best = min(copy_best_errors, literal_best_errors)

    affine_names: list[str] = []
    affine_errors: list[int] = []
    nuisance_count = nuisance.shape[1]
    for mask in range(1, 1 << nuisance_count):
        variables = [bit for bit in range(nuisance_count) if mask & (1 << bit)]
        base = np.bitwise_xor.reduce(nuisance[:, variables], axis=1)
        for bias in (0, 1):
            selector = base ^ bias
            affine_names.append(
                (" XOR ".join(f"x{bit+3}" for bit in variables))
                + (" XOR 1" if bias else "")
            )
            affine_errors.append(int(np.sum(selector != true_selector)))
    affine_errors_array = np.asarray(affine_errors, dtype=np.int64)
    affine_best_index = int(np.argmin(affine_errors_array))
    affine_best_errors = int(affine_errors_array[affine_best_index])
    affine_or_copy_best = min(copy_best_errors, affine_best_errors)

    return {
        "conflict_count": conflict_count,
        "copy_best_selector_errors": copy_best_errors,
        "literal_best_selector_errors": literal_best_errors,
        "literal_best_selector_name": literal_names[literal_best_index],
        "literal_best_conflict_accuracy": 1.0 - literal_best_errors / conflict_count,
        "literal_best_full_mux_accuracy": 1.0 - literal_best_errors / len(selected),
        "literal_or_copy_best_errors": literal_or_copy_best,
        "literal_advantage_over_copy": copy_best_errors - literal_or_copy_best,
        "literal_outperform_copy_count": int(
            np.sum(literal_errors_array < copy_best_errors)
        ),
        "literal_tie_or_beat_copy_count": int(
            np.sum(literal_errors_array <= copy_best_errors)
        ),
        "affine_best_selector_errors": affine_best_errors,
        "affine_best_selector_name": affine_names[affine_best_index],
        "affine_best_conflict_accuracy": 1.0 - affine_best_errors / conflict_count,
        "affine_best_full_mux_accuracy": 1.0 - affine_best_errors / len(selected),
        "affine_or_copy_best_errors": affine_or_copy_best,
        "affine_advantage_over_copy": copy_best_errors - affine_or_copy_best,
        "affine_outperform_copy_count": int(
            np.sum(affine_errors_array < copy_best_errors)
        ),
        "literal_selector_count": len(literal_errors),
        "affine_selector_count": len(affine_errors),
    }


def general_features(n: int, dataset_index: int) -> dict[str, Any]:
    inputs = truth_table_inputs()
    indices = np.sort(uniform_order(dataset_index)[:n])
    selected = inputs[indices]
    mux_y = mux_outputs(selected)
    parity_y = parity_outputs(selected)
    mux_cells = 4 * selected[:, 0] + 2 * selected[:, 1] + selected[:, 2]
    mux_counts = np.bincount(mux_cells, minlength=8).astype(np.int64)
    parity_cells = 2 * selected[:, 0] + selected[:, 1]
    parity_counts = np.bincount(parity_cells, minlength=4).astype(np.int64)
    conflict = selected[:, 1] != selected[:, 2]
    conflict_ids = mux_cells[conflict]
    conflict_counts = np.asarray([
        int(np.sum(conflict_ids == cell)) for cell in (1, 2, 5, 6)
    ], dtype=np.int64)
    d1 = int(np.sum(mux_y != selected[:, 1]))
    d2 = int(np.sum(mux_y != selected[:, 2]))
    return {
        "n": n,
        "dataset_index": dataset_index,
        "dataset_signature_reconstructed": dataset_signature(indices),
        "label_imbalance_mux": float(abs(mux_y.mean() - 0.5)),
        "label_imbalance_parity": float(abs(parity_y.mean() - 0.5)),
        "mux_cell_count_std": float(mux_counts.std()),
        "mux_minimum_cell_count": int(mux_counts.min()),
        "parity_cell_count_std": float(parity_counts.std()),
        "parity_minimum_cell_count": int(parity_counts.min()),
        "copy_refutation_min": min(d1, d2),
        "copy_refutation_absdiff": abs(d1 - d2),
        "distinct_conflict_cells": int(np.count_nonzero(conflict_counts)),
        "minimum_conflict_cell_count": int(conflict_counts.min()),
        **selector_error_metrics(selected),
    }


def build_rows(source_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [
        row for row in source_rows
        if row.get("optimizer_name") == "adamw"
        and row.get("sampling_protocol") == "uniform_random"
        and int(row.get("step", "-1")) == 40_000
        and row.get("target_name") in {"mux3", "parity2"}
    ]
    expected = len(TRAIN_COUNTS) * DATASETS_PER_N * 2
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(selected)}")
    cache = {
        (n, dataset_index): general_features(n, dataset_index)
        for n in TRAIN_COUNTS
        for dataset_index in range(DATASETS_PER_N)
    }
    output: list[dict[str, Any]] = []
    for row in selected:
        target = str(row["target_name"])
        n = int(row["n"])
        dataset_index = int(row["dataset_index"])
        features = dict(cache[(n, dataset_index)])
        if features["dataset_signature_reconstructed"] != row["dataset_signature"]:
            raise RuntimeError(
                f"signature mismatch target={target} n={n} dataset={dataset_index}"
            )
        output.append({
            **features,
            "target_name": target,
            "label_imbalance": (
                features["label_imbalance_mux"]
                if target == "mux3"
                else features["label_imbalance_parity"]
            ),
            "relevant_cell_count_std": (
                features["mux_cell_count_std"]
                if target == "mux3"
                else features["parity_cell_count_std"]
            ),
            "relevant_minimum_cell_count": (
                features["mux_minimum_cell_count"]
                if target == "mux3"
                else features["parity_minimum_cell_count"]
            ),
            "target_function_mass": parse_float(row["target_function_mass"]),
            "dataset_recovered": parse_bool(row["dataset_recovered"]),
            "modal_is_target": parse_bool(row["modal_is_target"]),
            "unseen_target_bit_accuracy_mean": parse_float(
                row["unseen_target_bit_accuracy_mean"]
            ),
        })
    return sorted(
        output,
        key=lambda row: (
            str(row["target_name"]), int(row["n"]), int(row["dataset_index"])
        ),
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
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def control_features_for_target(target: str) -> tuple[str, ...]:
    features = (
        "label_imbalance",
        "relevant_cell_count_std",
        "relevant_minimum_cell_count",
    )
    # 对 MUX3，relevant_cell_* 已经是 MUX 八格统计；只在 parity2
    # 阴性对照中额外控制同一输入集的 MUX 八格覆盖。
    if target == "parity2":
        features += ("mux_cell_count_std", "mux_minimum_cell_count")
    return features + (
        "copy_refutation_min",
        "copy_refutation_absdiff",
        "distinct_conflict_cells",
        "minimum_conflict_cell_count",
    )


def design_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    n_values = np.asarray([int(row["n"]) for row in rows])
    n_columns = np.stack([
        (n_values == n).astype(np.float64) for n in TRAIN_COUNTS
    ], axis=1)
    if not feature_names:
        return n_columns
    features = np.asarray([
        [float(row[name]) for name in feature_names] for row in rows
    ], dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return np.concatenate([n_columns, (features - mean) / scale], axis=1)


def residualized_rank_correlation(
    rows: Sequence[dict[str, Any]],
    feature: str,
    outcome: str,
) -> float:
    x = rank_average(np.asarray([float(row[feature]) for row in rows]))
    y = rank_average(np.asarray([float(row[outcome]) for row in rows]))
    target = str(rows[0]["target_name"])
    design = design_matrix(rows, control_features_for_target(target))
    ridge = np.eye(design.shape[1]) * 1e-4
    ridge[: len(TRAIN_COUNTS), : len(TRAIN_COUNTS)] = 0.0
    solve = lambda target: np.linalg.solve(
        design.T @ design + ridge, design.T @ target
    )
    residual_x = x - design @ solve(x)
    residual_y = y - design @ solve(y)
    return pearson(residual_x, residual_y)


def resample_rows(
    rows: Sequence[dict[str, Any]], sample: np.ndarray
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["n"]), int(row["dataset_index"])): row for row in rows
    }
    output = []
    for pseudo_index, original_index in enumerate(sample):
        for n in TRAIN_COUNTS:
            row = dict(lookup[(n, int(original_index))])
            row["dataset_index"] = pseudo_index
            output.append(row)
    return output


def block_inference(
    all_rows: Sequence[dict[str, Any]],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    target_rows = {
        target: [row for row in all_rows if row["target_name"] == target]
        for target in ("mux3", "parity2")
    }
    observed = {
        target: residualized_rank_correlation(
            rows, "literal_advantage_over_copy", "target_function_mass"
        )
        for target, rows in target_rows.items()
    }
    rng = np.random.default_rng(seed)
    boot = {"mux3": [], "parity2": [], "difference": []}
    for _ in range(bootstrap):
        sample = rng.integers(0, DATASETS_PER_N, size=DATASETS_PER_N)
        local_values = {}
        for target, rows in target_rows.items():
            local_values[target] = residualized_rank_correlation(
                resample_rows(rows, sample),
                "literal_advantage_over_copy",
                "target_function_mass",
            )
            boot[target].append(local_values[target])
        boot["difference"].append(local_values["mux3"] - local_values["parity2"])

    mux_lookup = {
        (int(row["n"]), int(row["dataset_index"])): row
        for row in target_rows["mux3"]
    }
    null_values = []
    for _ in range(permutations):
        permutation = rng.permutation(DATASETS_PER_N)
        permuted = []
        for dataset_index, source_index in enumerate(permutation):
            for n in TRAIN_COUNTS:
                row = dict(mux_lookup[(n, dataset_index)])
                row["target_function_mass"] = mux_lookup[
                    (n, int(source_index))
                ]["target_function_mass"]
                permuted.append(row)
        null_values.append(residualized_rank_correlation(
            permuted, "literal_advantage_over_copy", "target_function_mass"
        ))

    def interval(values: Sequence[float]) -> list[float]:
        finite = np.asarray([value for value in values if math.isfinite(value)])
        return list(map(float, np.quantile(finite, [0.025, 0.975])))

    finite_null = np.asarray([value for value in null_values if math.isfinite(value)])
    p_value = float(
        (1 + np.sum(finite_null <= observed["mux3"])) / (1 + len(finite_null))
    )
    return {
        "mux_partial_spearman": observed["mux3"],
        "mux_bootstrap_95": interval(boot["mux3"]),
        "mux_block_permutation_p_one_sided": p_value,
        "parity_partial_spearman": observed["parity2"],
        "parity_bootstrap_95": interval(boot["parity2"]),
        "mux_minus_parity": observed["mux3"] - observed["parity2"],
        "difference_bootstrap_95": interval(boot["difference"]),
    }


def standardized_features(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def fold_design(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_values = np.asarray([int(row["n"]) for row in rows])
    n_all = np.stack([
        (n_values == n).astype(np.float64) for n in TRAIN_COUNTS
    ], axis=1)
    if not feature_names:
        return n_all[train_mask], n_all[test_mask]
    values = np.asarray([
        [float(row[name]) for name in feature_names] for row in rows
    ], dtype=np.float64)
    train_values, test_values = standardized_features(
        values[train_mask], values[test_mask]
    )
    return (
        np.concatenate([n_all[train_mask], train_values], axis=1),
        np.concatenate([n_all[test_mask], test_values], axis=1),
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float = 1e-3) -> np.ndarray:
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[: len(TRAIN_COUNTS), : len(TRAIN_COUNTS)] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    regularizer = np.eye(x.shape[1]) * 1e-2
    regularizer[: len(TRAIN_COUNTS), : len(TRAIN_COUNTS)] = 0.0
    for _ in range(100):
        probability = sigmoid(x @ beta)
        weight = np.maximum(probability * (1 - probability), 1e-6)
        gradient = x.T @ (y - probability) - regularizer @ beta
        hessian = (x.T * weight) @ x + regularizer
        update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta += update
        if np.max(np.abs(update)) < 1e-8:
            break
    return beta


def grouped_cv(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(DATASETS_PER_N), 5)
    output = []
    for target in ("mux3", "parity2"):
        baseline = control_features_for_target(target)
        feature_sets = {
            "copy_baseline": baseline,
            "literal_nuisance": baseline + (
                "literal_best_selector_errors",
                "literal_advantage_over_copy",
                "literal_outperform_copy_count",
            ),
            "affine_nuisance": baseline + (
                "literal_best_selector_errors",
                "literal_advantage_over_copy",
                "literal_outperform_copy_count",
                "affine_best_selector_errors",
                "affine_advantage_over_copy",
                "affine_outperform_copy_count",
            ),
        }
        local = [row for row in rows if row["target_name"] == target]
        ids = np.asarray([int(row["dataset_index"]) for row in local])
        mass = np.asarray([float(row["target_function_mass"]) for row in local])
        recovered = np.asarray([float(bool(row["dataset_recovered"])) for row in local])
        for name, features in feature_sets.items():
            mass_prediction = np.empty(len(local))
            recovery_prediction = np.empty(len(local))
            for fold in folds:
                test_mask = np.isin(ids, fold)
                train_mask = ~test_mask
                train_x, test_x = fold_design(local, features, train_mask, test_mask)
                mass_prediction[test_mask] = np.clip(
                    test_x @ fit_ridge(train_x, mass[train_mask]), 0.0, 1.0
                )
                recovery_prediction[test_mask] = sigmoid(
                    test_x @ fit_logistic(train_x, recovered[train_mask])
                )
            mse = float(np.mean((mass - mass_prediction) ** 2))
            variance = float(np.mean((mass - mass.mean()) ** 2))
            probability = np.clip(recovery_prediction, 1e-8, 1 - 1e-8)
            output.append({
                "target_name": target,
                "model_name": name,
                "feature_names": list(features),
                "mass_cv_mse": mse,
                "mass_cv_r2": 1.0 - mse / variance,
                "recovery_cv_brier": float(np.mean((recovered - probability) ** 2)),
                "recovery_cv_log_loss": float(-np.mean(
                    recovered * np.log(probability)
                    + (1 - recovered) * np.log(1 - probability)
                )),
            })
    return output


def lookup_cv(rows: Sequence[dict[str, Any]], target: str, model: str) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["target_name"] == target and row["model_name"] == model
    )


def save_plot(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    inference: dict[str, Any],
    cv_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"绘图跳过：{exc}", flush=True)
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for target, color in (("mux3", "#dc2626"), ("parity2", "#2563eb")):
        local = [row for row in rows if row["target_name"] == target]
        x = np.asarray([float(row["literal_advantage_over_copy"]) for row in local])
        y = np.asarray([float(row["target_function_mass"]) for row in local])
        axes[0].scatter(x, y, s=8, alpha=0.18, color=color, label=target)
    axes[0].set_xlabel("literal advantage over best copy")
    axes[0].set_ylabel("target function mass")
    axes[0].set_title("Raw nuisance-selector association")
    axes[0].legend()

    labels = ["mux3", "parity2", "difference"]
    values = [
        inference["mux_partial_spearman"],
        inference["parity_partial_spearman"],
        inference["mux_minus_parity"],
    ]
    intervals = [
        inference["mux_bootstrap_95"],
        inference["parity_bootstrap_95"],
        inference["difference_bootstrap_95"],
    ]
    errors = np.asarray([
        [value - interval[0] for value, interval in zip(values, intervals)],
        [interval[1] - value for value, interval in zip(values, intervals)],
    ])
    axes[1].bar(labels, values, color=["#dc2626", "#2563eb", "#7c3aed"])
    axes[1].errorbar(labels, values, yerr=errors, fmt="none", color="black", capsize=4)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("partial Spearman")
    axes[1].set_title("After n/copy/coverage controls")

    models = ["copy_baseline", "literal_nuisance", "affine_nuisance"]
    x_positions = np.arange(len(models))
    width = 0.35
    for offset, (target, color) in enumerate((
        ("mux3", "#dc2626"), ("parity2", "#2563eb")
    )):
        values = [lookup_cv(cv_rows, target, model)["mass_cv_r2"] for model in models]
        axes[2].bar(
            x_positions + (offset - 0.5) * width,
            values,
            width=width,
            color=color,
            label=target,
        )
    axes[2].set_xticks(x_positions, ["copy", "+literal", "+affine"])
    axes[2].set_ylabel("grouped-CV R²")
    axes[2].set_title("Incremental target-mass prediction")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "uniform_nuisance_selector_analysis.png", dpi=180)
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
        args.result_dir / "uniform_nuisance_selector_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== Uniform nuisance-selector competition analysis ===", flush=True)
    print(f"source={source}", flush=True)
    print(f"output={output_dir}", flush=True)

    rows = build_rows(read_csv(source))
    write_csv(output_dir / "dataset_level_nuisance_features.csv", rows)
    inference = block_inference(
        rows, args.bootstrap, args.permutations, args.seed
    )
    cv_rows = grouped_cv(rows, args.seed + 1)
    write_json(output_dir / "partial_correlation_inference.json", inference)
    write_csv(output_dir / "grouped_cross_validation.csv", cv_rows)

    mux_baseline = lookup_cv(cv_rows, "mux3", "copy_baseline")
    mux_literal = lookup_cv(cv_rows, "mux3", "literal_nuisance")
    parity_baseline = lookup_cv(cv_rows, "parity2", "copy_baseline")
    parity_literal = lookup_cv(cv_rows, "parity2", "literal_nuisance")
    mux_improves = (
        mux_literal["mass_cv_mse"] < mux_baseline["mass_cv_mse"]
        and mux_literal["recovery_cv_brier"] < mux_baseline["recovery_cv_brier"]
    )
    parity_improves = (
        parity_literal["mass_cv_mse"] < parity_baseline["mass_cv_mse"]
        and parity_literal["recovery_cv_brier"] < parity_baseline["recovery_cv_brier"]
    )
    mux_ci = inference["mux_bootstrap_95"]
    difference_ci = inference["difference_bootstrap_95"]
    p_value = inference["mux_block_permutation_p_one_sided"]
    if (
        mux_ci[1] < 0
        and p_value < 0.01
        and difference_ci[1] < 0
        and mux_improves
        and not parity_improves
    ):
        verdict = "strong_support_for_nuisance_selector_competition"
    elif mux_ci[1] < 0 or mux_improves:
        verdict = "partial_support_for_nuisance_selector_competition"
    else:
        verdict = "no_independent_nuisance_selector_signal"
    summary = {
        "verdict": verdict,
        "row_count": len(rows),
        "independent_dataset_paths": DATASETS_PER_N,
        "primary_feature": "literal_advantage_over_copy",
        "primary_outcome": "target_function_mass",
        "inference": inference,
        "mux_copy_baseline_cv": mux_baseline,
        "mux_literal_cv": mux_literal,
        "parity_copy_baseline_cv": parity_baseline,
        "parity_literal_cv": parity_literal,
        "mux_literal_improves_mass_and_recovery": mux_improves,
        "parity_literal_improves_mass_and_recovery": parity_improves,
        "scope": (
            "This tests nuisance-selector traces at fixed 8-bit input. "
            "It does not by itself establish scaling with total input dimension."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    save_plot(output_dir, rows, inference, cv_rows)
    archive = package_output(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
