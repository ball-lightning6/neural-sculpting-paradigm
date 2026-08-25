"""8-bit中尺度系统的多seed/时间采样混合实验。

同时训练四个条件均衡的AND数据量和一个随机标签对照，保存全部256点预测。
训练结束后比较多seed系综与单轨迹时间系综的agreement、逐点边际、协方差、
Hamming距离和完整256-bit函数指纹。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2
    AND_TRAIN_PER_PRIMARY = (4, 6, 8, 12)
    INCLUDE_RANDOM_N32 = True
    NUISANCE_ORDER_SEED = 20261020
    RANDOM_LABEL_SEED = 20261021

    SEED_COUNT = 2_048
    INITIALIZATION_SEED = 20261022
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 20_000
    EARLY_EVAL_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
    EVAL_INTERVAL_STEPS = 100
    SAVE_INTERVAL_STEPS = 5_000

    TIME_WINDOW_POINTS = 64
    TIME_ANCHOR_COUNT = 32
    PAIR_SAMPLE_COUNT = 20_000
    IID_BOOTSTRAP_REPEATS = 200
    ANALYSIS_SEED = 20261023

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_8bit_seed_time_mixing")
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


@dataclass(frozen=True)
class Condition:
    name: str
    train_indices: tuple[int, ...]
    full_targets: np.ndarray
    kind: str


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.AND_TRAIN_PER_PRIMARY = (4, 8)
    Config.SEED_COUNT = 8
    Config.MAX_STEPS = 2
    Config.EARLY_EVAL_STEPS = (0, 1, 2)
    Config.EVAL_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_STEPS = 1
    Config.TIME_WINDOW_POINTS = 3
    Config.TIME_ANCHOR_COUNT = 4
    Config.PAIR_SAMPLE_COUNT = 20
    Config.IID_BOOTSTRAP_REPEATS = 10
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_seed_time_mixing"
    )
    Config.OVERWRITE_RESULT_DIR = True


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
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
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists():
        if Config.OVERWRITE_RESULT_DIR:
            shutil.rmtree(output)
        else:
            output = output.parent / (
                output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(256, dtype=np.uint16)
    shifts = np.arange(7, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float32)


def balanced_nuisance_order() -> tuple[int, ...]:
    rng = np.random.default_rng(Config.NUISANCE_ORDER_SEED)
    pair_representatives = np.arange(32, dtype=np.int64)
    rng.shuffle(pair_representatives)
    order: list[int] = []
    for value in pair_representatives:
        order.extend((int(value), int(63 - value)))
    return tuple(order)


def build_conditions() -> tuple[list[Condition], tuple[int, ...]]:
    inputs = truth_table_inputs()
    and_targets = (inputs[:, 0] * inputs[:, 1]).astype(np.uint8)
    nuisance_order = balanced_nuisance_order()
    conditions: list[Condition] = []
    for per_primary in Config.AND_TRAIN_PER_PRIMARY:
        nuisance = nuisance_order[:per_primary]
        train = tuple(sorted(
            (primary << 6) | suffix
            for primary in range(4)
            for suffix in nuisance
        ))
        conditions.append(Condition(
            name=f"and_n{len(train)}",
            train_indices=train,
            full_targets=and_targets.copy(),
            kind="structured_and",
        ))

    if Config.INCLUDE_RANDOM_N32:
        per_primary = 8
        nuisance = nuisance_order[:per_primary]
        train = tuple(sorted(
            (primary << 6) | suffix
            for primary in range(4)
            for suffix in nuisance
        ))
        rng = np.random.default_rng(Config.RANDOM_LABEL_SEED)
        targets = np.empty(256, dtype=np.uint8)
        train_labels = np.array([0] * 16 + [1] * 16, dtype=np.uint8)
        test_labels = np.array([0] * 112 + [1] * 112, dtype=np.uint8)
        rng.shuffle(train_labels)
        rng.shuffle(test_labels)
        test = np.setdiff1d(np.arange(256), np.asarray(train), assume_unique=True)
        targets[np.asarray(train)] = train_labels
        targets[test] = test_labels
        conditions.append(Condition(
            name="random_balanced_n32",
            train_indices=train,
            full_targets=targets,
            kind="random_labels",
        ))
    return conditions, nuisance_order


class BatchedPairedMLP(nn.Module):
    def __init__(self, condition_count: int) -> None:
        super().__init__()
        dimensions = [Config.INPUT_BITS, Config.WIDTH, Config.WIDTH, 1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.INITIALIZATION_SEED)
        base_weights: list[torch.Tensor] = []
        base_biases: list[torch.Tensor] = []
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            bound = 1.0 / math.sqrt(fan_in)
            base_weights.append(torch.empty(
                Config.SEED_COUNT, fan_out, fan_in
            ).uniform_(-bound, bound, generator=generator))
            base_biases.append(torch.empty(
                Config.SEED_COUNT, fan_out
            ).uniform_(-bound, bound, generator=generator))
        self.weights = nn.ParameterList([
            nn.Parameter(weight.repeat(condition_count, 1, 1))
            for weight in base_weights
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(bias.repeat(condition_count, 1))
            for bias in base_biases
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def build_tensors(
    conditions: Sequence[Condition],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = truth_table_inputs()
    max_count = max(len(condition.train_indices) for condition in conditions)
    train_x = np.zeros((len(conditions), max_count, 8), dtype=np.float32)
    train_y = np.zeros((len(conditions), max_count), dtype=np.float32)
    valid = np.zeros((len(conditions), max_count), dtype=np.float32)
    for index, condition in enumerate(conditions):
        ids = np.asarray(condition.train_indices, dtype=np.int64)
        train_x[index, :len(ids)] = inputs[ids]
        train_y[index, :len(ids)] = condition.full_targets[ids]
        valid[index, :len(ids)] = 1.0
    train_x = np.repeat(train_x, Config.SEED_COUNT, axis=0)
    train_y = np.repeat(train_y, Config.SEED_COUNT, axis=0)
    valid = np.repeat(valid, Config.SEED_COUNT, axis=0)
    full_x = np.repeat(
        inputs[None], len(conditions) * Config.SEED_COUNT, axis=0
    )
    return (
        torch.from_numpy(train_x).to(device),
        torch.from_numpy(train_y).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(full_x).to(device),
    )


def empirical_agreement(predictions: np.ndarray) -> float:
    probability = predictions.mean(axis=0, dtype=np.float64)
    return float(np.mean(probability ** 2 + (1.0 - probability) ** 2))


def binary_marginal_entropy(predictions: np.ndarray) -> float:
    probability = predictions.mean(axis=0, dtype=np.float64)
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(np.mean(
        -probability * np.log2(probability)
        -(1.0 - probability) * np.log2(1.0 - probability)
    ))


def fingerprint_hex(row: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest()
    return digest[:16]


def top_fingerprint_rows(
    condition: str,
    step: int,
    packed: np.ndarray,
    limit: int = 10,
) -> list[dict[str, Any]]:
    unique, counts = np.unique(packed, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [
        {
            "condition": condition,
            "step": step,
            "rank": rank,
            "fingerprint": fingerprint_hex(unique[index]),
            "count": int(counts[index]),
            "probability": float(counts[index] / len(packed)),
        }
        for rank, index in enumerate(order, start=1)
    ]


@torch.inference_mode()
def evaluate(
    model: BatchedPairedMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid: torch.Tensor,
    full_x: torch.Tensor,
    conditions: Sequence[Condition],
    step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    train_logits = model(train_x)
    per_item = F.binary_cross_entropy_with_logits(
        train_logits, train_y, reduction="none"
    )
    losses = (per_item * valid).sum(dim=1) / valid.sum(dim=1)
    train_exact = (((train_logits >= 0) == (train_y >= 0.5)) | (valid == 0)).all(dim=1)
    predictions = (model(full_x) >= 0).cpu().numpy().astype(np.uint8)
    losses_np = losses.cpu().numpy()
    exact_np = train_exact.cpu().numpy()

    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    raw: dict[str, np.ndarray] = {}
    all_states = np.arange(256)
    for condition_index, condition in enumerate(conditions):
        start = condition_index * Config.SEED_COUNT
        stop = start + Config.SEED_COUNT
        local_predictions = predictions[start:stop]
        train_indices = np.asarray(condition.train_indices, dtype=np.int64)
        test_indices = np.setdiff1d(all_states, train_indices, assume_unique=True)
        test_predictions = local_predictions[:, test_indices]
        target = condition.full_targets
        test_accuracy = (
            test_predictions == target[test_indices][None]
        ).mean(axis=1)
        full_exact = (local_predictions == target[None]).all(axis=1)
        packed = np.packbits(local_predictions, axis=1, bitorder="little")
        unique_count = int(np.unique(packed, axis=0).shape[0])
        rows.append({
            "condition": condition.name,
            "kind": condition.kind,
            "step": step,
            "model_count": Config.SEED_COUNT,
            "train_count": len(train_indices),
            "test_count": len(test_indices),
            "train_loss_min": float(losses_np[start:stop].min()),
            "train_loss_median": float(np.median(losses_np[start:stop])),
            "train_loss_mean": float(losses_np[start:stop].mean()),
            "train_hard_exact_fraction": float(exact_np[start:stop].mean()),
            "test_accuracy_mean": float(test_accuracy.mean()),
            "test_accuracy_median": float(np.median(test_accuracy)),
            "full_function_exact_fraction": float(full_exact.mean()),
            "seed_agreement_test": empirical_agreement(test_predictions),
            "test_marginal_entropy_bits": binary_marginal_entropy(test_predictions),
            "unique_full_fingerprint_count": unique_count,
        })
        top_rows.extend(top_fingerprint_rows(
            condition.name, step, packed
        ))
        raw[f"pred_{condition.name}_step{step}"] = packed
        raw[f"loss_{condition.name}_step{step}"] = losses_np[start:stop]
    return rows, top_rows, raw


def unpack_predictions(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        packed, axis=1, count=256, bitorder="little"
    ).astype(np.float64)


def covariance(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = predictions.mean(axis=0)
    second = predictions.T @ predictions / max(len(predictions), 1)
    return probability, second - np.outer(probability, probability)


def hamming_pair_summary(
    predictions: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    count = len(predictions)
    if count < 2:
        return {"mean": float("nan"), "std": float("nan"), "q10": float("nan"), "q50": float("nan"), "q90": float("nan")}
    first = rng.integers(0, count, size=Config.PAIR_SAMPLE_COUNT)
    second = rng.integers(0, count, size=Config.PAIR_SAMPLE_COUNT)
    distance = np.mean(predictions[first] != predictions[second], axis=1)
    return {
        "mean": float(distance.mean()),
        "std": float(distance.std()),
        "q10": float(np.quantile(distance, 0.10)),
        "q50": float(np.quantile(distance, 0.50)),
        "q90": float(np.quantile(distance, 0.90)),
    }


def analyze_seed_time_mixing(
    conditions: Sequence[Condition],
    raw: dict[str, np.ndarray],
    evaluated_steps: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regular = sorted(set(int(step) for step in evaluated_steps))
    window = regular[-min(Config.TIME_WINDOW_POINTS, len(regular)):]
    rng = np.random.default_rng(Config.ANALYSIS_SEED)
    anchor_count = min(Config.TIME_ANCHOR_COUNT, Config.SEED_COUNT)
    anchors = np.sort(rng.choice(
        Config.SEED_COUNT, size=anchor_count, replace=False
    ))
    summary_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    all_states = np.arange(256)

    for condition in conditions:
        train = np.asarray(condition.train_indices, dtype=np.int64)
        test = np.setdiff1d(all_states, train, assume_unique=True)
        snapshots = [
            unpack_predictions(raw[f"pred_{condition.name}_step{step}"])[:, test]
            for step in window
        ]
        reference = np.concatenate(snapshots, axis=0)
        reference_p, reference_cov = covariance(reference)
        reference_cov_norm = max(float(np.linalg.norm(reference_cov)), 1e-12)
        final = snapshots[-1]
        final_p, final_cov = covariance(final)
        first_half = np.concatenate(snapshots[:max(1, len(snapshots) // 2)], axis=0)
        second_half = np.concatenate(snapshots[len(snapshots) // 2:], axis=0)
        first_p = first_half.mean(axis=0)
        second_p = second_half.mean(axis=0)

        pooled_time = np.concatenate([
            np.stack([snapshot[anchor] for snapshot in snapshots], axis=0)
            for anchor in anchors
        ], axis=0)
        pooled_p, pooled_cov = covariance(pooled_time)
        iid_agreement: list[float] = []
        iid_marginal_mae: list[float] = []
        iid_covariance_error: list[float] = []
        iid_unique: list[int] = []
        for _ in range(Config.IID_BOOTSTRAP_REPEATS):
            iid_sample = reference[rng.integers(
                0, len(reference), size=len(window)
            )]
            iid_p, iid_cov = covariance(iid_sample)
            iid_agreement.append(empirical_agreement(iid_sample))
            iid_marginal_mae.append(float(np.mean(np.abs(
                iid_p - reference_p
            ))))
            iid_covariance_error.append(float(
                np.linalg.norm(iid_cov - reference_cov)
                / reference_cov_norm
            ))
            iid_unique.append(int(np.unique(
                np.packbits(
                    iid_sample.astype(np.uint8), axis=1, bitorder="little"
                ),
                axis=0,
            ).shape[0]))
        anchor_agreements: list[float] = []
        anchor_mae: list[float] = []
        anchor_cov_error: list[float] = []
        anchor_unique: list[int] = []
        for anchor in anchors:
            temporal = np.stack([
                snapshot[anchor] for snapshot in snapshots
            ], axis=0)
            temporal_p, temporal_cov = covariance(temporal)
            agreement = empirical_agreement(temporal)
            marginal_mae = float(np.mean(np.abs(temporal_p - reference_p)))
            cov_error = float(
                np.linalg.norm(temporal_cov - reference_cov)
                / reference_cov_norm
            )
            unique = int(np.unique(
                np.packbits(
                    temporal.astype(np.uint8), axis=1, bitorder="little"
                ),
                axis=0,
            ).shape[0])
            anchor_agreements.append(agreement)
            anchor_mae.append(marginal_mae)
            anchor_cov_error.append(cov_error)
            anchor_unique.append(unique)
            anchor_rows.append({
                "condition": condition.name,
                "anchor_seed": int(anchor),
                "time_sample_count": len(window),
                "time_agreement": agreement,
                "marginal_mae_vs_seed_time_reference": marginal_mae,
                "covariance_relative_error": cov_error,
                "unique_time_fingerprint_count": unique,
            })

        final_hamming = hamming_pair_summary(final, rng)
        pooled_hamming = hamming_pair_summary(pooled_time, rng)
        summary_rows.append({
            "condition": condition.name,
            "kind": condition.kind,
            "window_first_step": window[0],
            "window_last_step": window[-1],
            "window_point_count": len(window),
            "reference_seed_time_agreement": empirical_agreement(reference),
            "final_seed_agreement": empirical_agreement(final),
            "pooled_anchor_time_agreement": empirical_agreement(pooled_time),
            "pooled_time_marginal_mae": float(np.mean(np.abs(
                pooled_p - reference_p
            ))),
            "pooled_time_covariance_relative_error": float(
                np.linalg.norm(pooled_cov - reference_cov)
                / reference_cov_norm
            ),
            "final_seed_marginal_mae": float(np.mean(np.abs(
                final_p - reference_p
            ))),
            "final_seed_covariance_relative_error": float(
                np.linalg.norm(final_cov - reference_cov)
                / reference_cov_norm
            ),
            "window_marginal_drift_first_vs_second": float(np.mean(np.abs(
                first_p - second_p
            ))),
            "anchor_time_agreement_min": float(np.min(anchor_agreements)),
            "anchor_time_agreement_median": float(np.median(anchor_agreements)),
            "anchor_time_agreement_max": float(np.max(anchor_agreements)),
            "anchor_marginal_mae_median": float(np.median(anchor_mae)),
            "anchor_marginal_mae_max": float(np.max(anchor_mae)),
            "anchor_covariance_relative_error_median": float(
                np.median(anchor_cov_error)
            ),
            "anchor_covariance_relative_error_max": float(
                np.max(anchor_cov_error)
            ),
            "anchor_unique_time_fingerprints_median": float(
                np.median(anchor_unique)
            ),
            "iid64_agreement_median": float(np.median(iid_agreement)),
            "iid64_agreement_p05": float(np.quantile(iid_agreement, 0.05)),
            "iid64_agreement_p95": float(np.quantile(iid_agreement, 0.95)),
            "iid64_marginal_mae_median": float(
                np.median(iid_marginal_mae)
            ),
            "iid64_covariance_relative_error_median": float(
                np.median(iid_covariance_error)
            ),
            "iid64_unique_fingerprints_median": float(np.median(iid_unique)),
            "final_unique_fingerprints": int(np.unique(
                np.packbits(
                    final.astype(np.uint8), axis=1, bitorder="little"
                ),
                axis=0,
            ).shape[0]),
            "pooled_time_unique_fingerprints": int(np.unique(
                np.packbits(
                    pooled_time.astype(np.uint8), axis=1, bitorder="little"
                ),
                axis=0,
            ).shape[0]),
            **{f"final_hamming_{key}": value for key, value in final_hamming.items()},
            **{f"pooled_time_hamming_{key}": value for key, value in pooled_hamming.items()},
        })
    return summary_rows, anchor_rows


def save_progress(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    top_rows: Sequence[dict[str, Any]],
    raw: dict[str, np.ndarray],
) -> None:
    write_csv(output_dir / "training_curves.csv", rows)
    write_csv(output_dir / "top_fingerprints.csv", top_rows)
    np.savez_compressed(output_dir / "prediction_snapshots.npz", **raw)


def create_archive(output_dir: Path) -> Path:
    archive_path = output_dir.parent / f"{output_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    return archive_path


def print_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"step={int(row['step']):>6,} | {row['condition']:<20} "
            f"BCE={row['train_loss_median']:.3e} "
            f"exact={row['train_hard_exact_fraction']:.1%} "
            f"test={row['test_accuracy_mean']:.2%} "
            f"agreement={row['seed_agreement_test']:.4f} "
            f"full-exact={row['full_function_exact_fraction']:.2%} "
            f"unique={int(row['unique_full_fingerprint_count']):,}",
            flush=True,
        )


def main() -> None:
    apply_smoke_overrides()
    conditions, nuisance_order = build_conditions()
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("high")

    write_json(output_dir / "config.json", {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    })
    write_json(output_dir / "conditions.json", {
        "nuisance_order": nuisance_order,
        "conditions": [
            {
                "name": condition.name,
                "kind": condition.kind,
                "train_indices": condition.train_indices,
                "train_count": len(condition.train_indices),
            }
            for condition in conditions
        ],
    })
    train_x, train_y, valid, full_x = build_tensors(conditions, device)
    model = BatchedPairedMLP(len(conditions)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    raw: dict[str, np.ndarray] = {}
    evaluated_steps: list[int] = []
    last_eval_step = -1
    interrupted = False
    started = time.perf_counter()

    print("=== 8-bit seed/time mixing ===")
    print(
        f"device={device} | conditions={len(conditions)} | "
        f"seeds/condition={Config.SEED_COUNT:,} | "
        f"models={len(conditions) * Config.SEED_COUNT:,} | "
        f"max_steps={Config.MAX_STEPS:,}"
    )
    print(f"result={output_dir.resolve()}")

    def should_evaluate(step: int) -> bool:
        return (
            step in set(Config.EARLY_EVAL_STEPS)
            or step % Config.EVAL_INTERVAL_STEPS == 0
        )

    def evaluate_current(step: int) -> None:
        nonlocal last_eval_step
        if last_eval_step == step:
            return
        current_rows, current_top, current_raw = evaluate(
            model, train_x, train_y, valid, full_x, conditions, step
        )
        rows.extend(current_rows)
        top_rows.extend(current_top)
        raw.update(current_raw)
        evaluated_steps.append(step)
        last_eval_step = step
        print_rows(current_rows)

    step = 0
    try:
        while step <= Config.MAX_STEPS:
            if should_evaluate(step):
                evaluate_current(step)
            if step == Config.MAX_STEPS:
                break
            logits = model(train_x)
            per_item = F.binary_cross_entropy_with_logits(
                logits, train_y, reduction="none"
            )
            per_model = (per_item * valid).sum(dim=1) / valid.sum(dim=1)
            if not torch.isfinite(per_model).all():
                raise RuntimeError(f"step={step}出现非有限loss。")
            optimizer.zero_grad(set_to_none=True)
            per_model.sum().backward()
            optimizer.step()
            step += 1
            if step % Config.SAVE_INTERVAL_STEPS == 0:
                if should_evaluate(step):
                    evaluate_current(step)
                save_progress(output_dir, rows, top_rows, raw)
    except KeyboardInterrupt:
        interrupted = True
        print("收到Ctrl+C，正在分析并保存……", flush=True)
    finally:
        evaluate_current(step)
        save_progress(output_dir, rows, top_rows, raw)
        mixing_rows, anchor_rows = analyze_seed_time_mixing(
            conditions, raw, evaluated_steps
        )
        write_csv(output_dir / "mixing_summary.csv", mixing_rows)
        write_csv(output_dir / "anchor_time_mixing.csv", anchor_rows)
        write_json(output_dir / "summary.json", {
            "status": "interrupted" if interrupted else "completed",
            "last_step": step,
            "elapsed_seconds": time.perf_counter() - started,
            "conditions": [condition.name for condition in conditions],
            "mixing_summary": mixing_rows,
            "interpretation": {
                "small_anchor_errors": (
                    "单轨迹时间统计接近多seed动态系综，支持函数可观测量层面的自平均。"
                ),
                "large_anchor_errors": (
                    "单轨迹未遍历多seed动态系综，初始化/流管记忆仍显著。"
                ),
                "scope": (
                    "本实验比较动态时间系综和动态seed系综，不等同于静态loss截面。"
                ),
            },
        })
        archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
        print("=== 已保存 ===")
        if archive is not None:
            print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
