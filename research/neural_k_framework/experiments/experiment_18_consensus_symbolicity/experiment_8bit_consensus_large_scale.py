"""固定 tanh-MLP 的 8-bit 共识符号性大规模两阶段验证。

阶段一在大量随机平衡 n=12 部分真值表上用 64 个独立初始化筛选高完整函数
共识候选；阶段二只对候选使用全新 4096 个初始化重新训练。候选筛选只使用
拟合率和完整函数共识，不使用任何复杂度指标，避免按结果可读性选择样本。

脚本按数据集分片保存，可在进程中断后直接重跑并跳过完整分片。
"""

from __future__ import annotations

import csv
import functools
import hashlib
import itertools
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
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
    TRAIN_COUNT = 12
    RANDOM_DATASET_COUNT = 8_192
    DATASET_SEED = 2026082301

    SCREEN_SEED_COUNT = 64
    SCREEN_INITIALIZATION_SEED = 2026082302
    SCREEN_MAX_STEPS = 10_000
    SCREEN_EVAL_STEPS = (200, 500, 1_000, 2_500, 5_000, 10_000)
    SCREEN_DATASETS_PER_SHARD = 128
    SCREEN_CANDIDATE_MIN_FIT_RATE = 0.95
    SCREEN_CANDIDATE_MIN_MODAL = 0.75
    SCREEN_CANDIDATE_MIN_COLLISION = 0.55
    MAX_CONFIRM_CANDIDATES = 256

    CONFIRM_SEED_COUNT = 4_096
    CONFIRM_INITIALIZATION_SEED = 2026082303
    CONFIRM_MAX_STEPS = 20_000
    CONFIRM_EVAL_STEPS = (
        100, 200, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 20_000
    )
    CONFIRM_DATASETS_PER_SHARD = 2

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    STRICT_MIN_FIT_RATE = 0.95
    STRICT_MODAL_PROBABILITY = 0.95
    STRICT_FUNCTION_COLLISION = 0.90

    BDD_RANDOM_ORDERS = 16
    COMPLEXITY_SEED = 2026082304

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_8bit_consensus_large_scale")
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_index: int
    name: str
    train_indices: tuple[int, ...]
    train_labels: tuple[int, ...]
    signature: str


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.RANDOM_DATASET_COUNT = 16
    Config.SCREEN_SEED_COUNT = 4
    Config.SCREEN_MAX_STEPS = 5
    Config.SCREEN_EVAL_STEPS = (0, 1, 2, 5)
    Config.SCREEN_DATASETS_PER_SHARD = 8
    Config.SCREEN_CANDIDATE_MIN_FIT_RATE = 0.0
    Config.SCREEN_CANDIDATE_MIN_MODAL = 0.0
    Config.SCREEN_CANDIDATE_MIN_COLLISION = 0.0
    Config.MAX_CONFIRM_CANDIDATES = 2
    Config.CONFIRM_SEED_COUNT = 8
    Config.CONFIRM_MAX_STEPS = 5
    Config.CONFIRM_EVAL_STEPS = (0, 1, 2, 5)
    Config.CONFIRM_DATASETS_PER_SHARD = 1
    Config.BDD_RANDOM_ORDERS = 2
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/consensus_symbolicity/_smoke_8bit_consensus_large_scale"
    )
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True
    Config.PACKAGE_RESULTS = False


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def config_payload() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    saved_path = output / "config.json"
    current = config_payload()
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved != current:
            raise RuntimeError(
                "结果目录已有不同配置。请修改 RESULT_DIR，或明确启用覆盖。"
            )
        if not Config.RESUME:
            raise RuntimeError("结果目录已存在且 RESUME=False。")
    else:
        write_json(saved_path, current)
    (output / "screen_shards").mkdir(exist_ok=True)
    (output / "confirm_shards").mkdir(exist_ok=True)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.uint16)
    shifts = np.arange(Config.INPUT_BITS - 1, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float32)


def pack_truth(bits: np.ndarray) -> np.ndarray:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8), axis=-1, bitorder="little"
    )


def unpack_truth(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, bitorder="little"
    )[..., : 2 ** Config.INPUT_BITS]


def fingerprint_hex(bits_or_packed: np.ndarray, packed: bool = False) -> str:
    values = (
        np.asarray(bits_or_packed, dtype=np.uint8)
        if packed
        else pack_truth(np.asarray(bits_or_packed, dtype=np.uint8))
    )
    return values.tobytes().hex().upper()


def dataset_signature(indices: Sequence[int], labels: Sequence[int]) -> str:
    payload = json.dumps(
        list(zip(map(int, indices), map(int, labels))),
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def gf2_rank(rows: np.ndarray) -> int:
    values = np.asarray(rows, dtype=np.uint8).copy()
    rank = 0
    for column in range(values.shape[1]):
        candidates = np.flatnonzero(values[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        values[[rank, pivot]] = values[[pivot, rank]]
        for row in range(values.shape[0]):
            if row != rank and values[row, column]:
                values[row] ^= values[rank]
        rank += 1
        if rank == values.shape[0]:
            break
    return int(rank)


def build_datasets() -> tuple[list[DatasetSpec], list[dict[str, Any]]]:
    rng = np.random.default_rng(Config.DATASET_SEED)
    inputs = truth_table_inputs().astype(np.uint8)
    specs: list[DatasetSpec] = []
    seen: set[str] = set()
    while len(specs) < Config.RANDOM_DATASET_COUNT:
        indices = np.sort(
            rng.choice(2 ** Config.INPUT_BITS, Config.TRAIN_COUNT, replace=False)
        )
        labels = np.asarray(
            [0] * (Config.TRAIN_COUNT // 2)
            + [1] * (Config.TRAIN_COUNT - Config.TRAIN_COUNT // 2),
            dtype=np.uint8,
        )
        rng.shuffle(labels)
        signature = dataset_signature(indices, labels)
        if signature in seen:
            continue
        seen.add(signature)
        dataset_index = len(specs)
        specs.append(DatasetSpec(
            dataset_index=dataset_index,
            name=f"random_n{Config.TRAIN_COUNT}_{dataset_index:05d}_{signature[:8]}",
            train_indices=tuple(map(int, indices)),
            train_labels=tuple(map(int, labels)),
            signature=signature,
        ))

    rows: list[dict[str, Any]] = []
    for spec in specs:
        indices = np.asarray(spec.train_indices, dtype=np.int64)
        selected = inputs[indices]
        rows.append({
            **asdict(spec),
            "train_count": len(spec.train_indices),
            "positive_count": int(sum(spec.train_labels)),
            "input_gf2_rank": gf2_rank(selected),
            "input_bit_mean": selected.mean(axis=0).tolist(),
            "input_hamming_weight_mean": float(selected.sum(axis=1).mean()),
        })
    return specs, rows


class BatchedMLPEnsemble(nn.Module):
    def __init__(
        self,
        global_dataset_indices: Sequence[int],
        seed_count: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        dimensions = (
            [Config.INPUT_BITS]
            + [Config.WIDTH] * Config.HIDDEN_LAYERS
            + [1]
        )
        layer_count = len(dimensions) - 1
        weight_blocks: list[list[torch.Tensor]] = [
            [] for _ in range(layer_count)
        ]
        bias_blocks: list[list[torch.Tensor]] = [
            [] for _ in range(layer_count)
        ]
        for global_dataset_index in global_dataset_indices:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(initialization_seed)
                + 1_000_003 * int(global_dataset_index)
            )
            for layer_index, (fan_in, fan_out) in enumerate(
                zip(dimensions[:-1], dimensions[1:])
            ):
                bound = 1.0 / math.sqrt(fan_in)
                weight_blocks[layer_index].append(torch.empty(
                    seed_count, fan_out, fan_in
                ).uniform_(-bound, bound, generator=generator))
                bias_blocks[layer_index].append(torch.empty(
                    seed_count, fan_out
                ).uniform_(-bound, bound, generator=generator))
        self.weights = nn.ParameterList([
            nn.Parameter(torch.stack(blocks, dim=0).reshape(
                len(global_dataset_indices) * seed_count,
                dimensions[layer_index + 1],
                dimensions[layer_index],
            ))
            for layer_index, blocks in enumerate(weight_blocks)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.stack(blocks, dim=0).reshape(
                len(global_dataset_indices) * seed_count,
                dimensions[layer_index + 1],
            ))
            for layer_index, blocks in enumerate(bias_blocks)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if layer_index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def parameter_count_per_model() -> int:
    dimensions = (
        [Config.INPUT_BITS]
        + [Config.WIDTH] * Config.HIDDEN_LAYERS
        + [1]
    )
    return int(sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:])
    ))


def build_training_tensors(
    specs: Sequence[DatasetSpec],
    seed_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = truth_table_inputs()
    train_x = np.zeros(
        (len(specs), Config.TRAIN_COUNT, Config.INPUT_BITS), dtype=np.float32
    )
    train_y = np.zeros((len(specs), Config.TRAIN_COUNT), dtype=np.float32)
    for local_index, spec in enumerate(specs):
        train_x[local_index] = inputs[np.asarray(spec.train_indices)]
        train_y[local_index] = np.asarray(spec.train_labels, dtype=np.float32)
    return (
        torch.as_tensor(np.repeat(train_x, seed_count, axis=0), device=device),
        torch.as_tensor(np.repeat(train_y, seed_count, axis=0), device=device),
        torch.as_tensor(inputs, device=device),
    )


def distinct_collision(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total < 2:
        return float("nan")
    return float(
        np.sum(counts.astype(np.float64) * (counts - 1))
        / (total * (total - 1))
    )


def plugin_entropy(counts: np.ndarray) -> float:
    probability = counts.astype(np.float64) / counts.sum()
    positive = probability[probability > 0]
    return float(-(positive * np.log2(positive)).sum())


def bit_agreement_distinct(predictions: np.ndarray, indices: np.ndarray) -> float:
    local = predictions[:, indices]
    count = len(local)
    if count < 2 or not len(indices):
        return float("nan")
    ones = local.sum(axis=0).astype(np.float64)
    same = ones * (ones - 1) + (count - ones) * (count - ones - 1)
    return float(np.mean(same / (count * (count - 1))))


def evaluate_models(
    phase: str,
    step: int,
    model: BatchedMLPEnsemble,
    specs: Sequence[DatasetSpec],
    global_indices: Sequence[int],
    seed_count: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    full_inputs: torch.Tensor,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        losses = F.binary_cross_entropy_with_logits(
            train_logits, train_y, reduction="none"
        ).mean(axis=1)
        train_exact = torch.all(
            (train_logits >= 0) == (train_y >= 0.5), axis=1
        )
        full_batch = full_inputs[None].expand(len(train_x), -1, -1)
        predictions = (model(full_batch) >= 0).to(torch.uint8).cpu().numpy()
        losses_cpu = losses.cpu().numpy()
        exact_cpu = train_exact.cpu().numpy().astype(bool)

    packed_all = pack_truth(predictions).reshape(len(specs), seed_count, 32)
    modal_all = np.empty((len(specs), 32), dtype=np.uint8)
    all_states = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for local_index, (spec, global_index) in enumerate(
        zip(specs, global_indices)
    ):
        start = local_index * seed_count
        stop = start + seed_count
        local_predictions = predictions[start:stop]
        local_exact = exact_cpu[start:stop]
        cohort = local_predictions[local_exact]
        cohort_source = "train_hard_exact_models"
        if not len(cohort):
            cohort = local_predictions
            cohort_source = "all_models_no_fitted_cohort"
        packed = pack_truth(cohort)
        unique, counts = np.unique(packed, axis=0, return_counts=True)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]
        modal = unique[0]
        modal_all[local_index] = modal
        modal_bits = unpack_truth(modal[None])[0]
        test_indices = np.setdiff1d(
            all_states,
            np.asarray(spec.train_indices, dtype=np.int64),
            assume_unique=True,
        )
        hamming = np.mean(cohort != modal_bits[None], axis=1)
        rows.append({
            "phase": phase,
            "step": step,
            "dataset_index": int(global_index),
            "dataset_name": spec.name,
            "signature": spec.signature,
            "train_count": len(spec.train_indices),
            "seed_count": seed_count,
            "train_fit_rate": float(local_exact.mean()),
            "train_loss_mean": float(losses_cpu[start:stop].mean()),
            "train_loss_median": float(np.median(losses_cpu[start:stop])),
            "cohort_source": cohort_source,
            "cohort_model_count": len(cohort),
            "unique_function_count": len(unique),
            "modal_count": int(counts[0]),
            "modal_probability": float(counts[0] / len(cohort)),
            "function_collision": distinct_collision(counts),
            "function_entropy_plugin_bits": plugin_entropy(counts),
            "modal_fingerprint": fingerprint_hex(modal, packed=True),
            "mean_hamming_to_modal_full": float(hamming.mean()),
            "max_hamming_to_modal_full": float(hamming.max()),
            "unseen_bit_agreement": bit_agreement_distinct(
                cohort, test_indices
            ),
        })
    return rows, modal_all, packed_all


def train_shard(
    phase: str,
    shard_index: int,
    specs: Sequence[DatasetSpec],
    global_indices: Sequence[int],
    seed_count: int,
    initialization_seed: int,
    max_steps: int,
    eval_steps: Sequence[int],
    output_dir: Path,
    device: torch.device,
) -> None:
    phase_dir = output_dir / f"{phase}_shards"
    stem = f"{phase}_shard_{shard_index:04d}"
    final_path = phase_dir / f"{stem}_final.csv"
    trajectory_path = phase_dir / f"{stem}_trajectory.csv"
    predictions_path = phase_dir / f"{stem}_predictions.npz"
    if final_path.exists() and trajectory_path.exists() and predictions_path.exists():
        print(f"[{phase}] shard={shard_index:04d} 已完成，跳过。", flush=True)
        return

    model = BatchedMLPEnsemble(
        global_indices, seed_count, initialization_seed
    ).to(device)
    train_x, train_y, full_inputs = build_training_tensors(
        specs, seed_count, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    evaluation_steps = sorted(set(map(int, eval_steps)) | {int(max_steps)})
    evaluation_set = set(evaluation_steps)
    trajectory_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    final_modal = np.empty((len(specs), 32), dtype=np.uint8)
    final_predictions = np.empty((len(specs), seed_count, 32), dtype=np.uint8)
    start = time.perf_counter()

    for step in range(max_steps + 1):
        if step in evaluation_set:
            rows, modal, predictions = evaluate_models(
                phase,
                step,
                model,
                specs,
                global_indices,
                seed_count,
                train_x,
                train_y,
                full_inputs,
            )
            trajectory_rows.extend(rows)
            final_rows = rows
            final_modal = modal
            final_predictions = predictions
            fitted = sum(
                float(row["train_fit_rate"]) >= Config.STRICT_MIN_FIT_RATE
                for row in rows
            )
            strict = sum(
                float(row["train_fit_rate"]) >= Config.STRICT_MIN_FIT_RATE
                and float(row["modal_probability"])
                >= Config.STRICT_MODAL_PROBABILITY
                and float(row["function_collision"])
                >= Config.STRICT_FUNCTION_COLLISION
                for row in rows
            )
            print(
                f"[{phase}] shard={shard_index:04d} step={step:>6,} "
                f"fitted={fitted}/{len(rows)} strict={strict} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
        if step == max_steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        per_model = F.binary_cross_entropy_with_logits(
            logits, train_y, reduction="none"
        ).mean(axis=1)
        per_model.sum().backward()
        optimizer.step()

    write_csv(trajectory_path, trajectory_rows)
    write_csv(final_path, final_rows)
    np.savez_compressed(
        predictions_path,
        dataset_indices=np.asarray(global_indices, dtype=np.int64),
        predictions_packed=final_predictions,
        modal_packed=final_modal,
        seed_count=np.asarray(seed_count, dtype=np.int64),
    )
    del optimizer, model, train_x, train_y, full_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()


def shard_ranges(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def run_phase(
    phase: str,
    specs: Sequence[DatasetSpec],
    global_indices: Sequence[int],
    seed_count: int,
    initialization_seed: int,
    max_steps: int,
    eval_steps: Sequence[int],
    datasets_per_shard: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    ranges = shard_ranges(len(specs), datasets_per_shard)
    for shard_index, (start, stop) in enumerate(ranges):
        print(
            f"=== {phase} shard {shard_index+1}/{len(ranges)} | "
            f"datasets={start}:{stop} ===",
            flush=True,
        )
        train_shard(
            phase=phase,
            shard_index=shard_index,
            specs=specs[start:stop],
            global_indices=global_indices[start:stop],
            seed_count=seed_count,
            initialization_seed=initialization_seed,
            max_steps=max_steps,
            eval_steps=eval_steps,
            output_dir=output_dir,
            device=device,
        )


def aggregate_phase(
    phase: str,
    output_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    phase_dir = output_dir / f"{phase}_shards"
    final_rows: list[dict[str, str]] = []
    trajectory_rows: list[dict[str, str]] = []
    for path in sorted(phase_dir.glob(f"{phase}_shard_*_final.csv")):
        final_rows.extend(read_csv(path))
    for path in sorted(phase_dir.glob(f"{phase}_shard_*_trajectory.csv")):
        trajectory_rows.extend(read_csv(path))
    final_rows.sort(key=lambda row: int(row["dataset_index"]))
    trajectory_rows.sort(
        key=lambda row: (int(row["dataset_index"]), int(row["step"]))
    )
    write_csv(output_dir / f"{phase}_final_summary.csv", final_rows)
    write_csv(output_dir / f"{phase}_trajectory.csv", trajectory_rows)
    return final_rows, trajectory_rows


def aggregate_predictions(
    phase: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    modal: list[np.ndarray] = []
    phase_dir = output_dir / f"{phase}_shards"
    for path in sorted(phase_dir.glob(f"{phase}_shard_*_predictions.npz")):
        payload = np.load(path)
        indices.append(payload["dataset_indices"])
        predictions.append(payload["predictions_packed"])
        modal.append(payload["modal_packed"])
    if not indices:
        empty_indices = np.empty(0, dtype=np.int64)
        empty_predictions = np.empty((0, 0, 32), dtype=np.uint8)
        empty_modal = np.empty((0, 32), dtype=np.uint8)
        np.savez_compressed(
            output_dir / f"{phase}_final_predictions_packed.npz",
            dataset_indices=empty_indices,
            predictions_packed=empty_predictions,
            modal_packed=empty_modal,
        )
        return empty_indices, empty_predictions, empty_modal
    all_indices = np.concatenate(indices, axis=0)
    all_predictions = np.concatenate(predictions, axis=0)
    all_modal = np.concatenate(modal, axis=0)
    order = np.argsort(all_indices)
    all_indices = all_indices[order]
    all_predictions = all_predictions[order]
    all_modal = all_modal[order]
    np.savez_compressed(
        output_dir / f"{phase}_final_predictions_packed.npz",
        dataset_indices=all_indices,
        predictions_packed=all_predictions,
        modal_packed=all_modal,
    )
    return all_indices, all_predictions, all_modal


def select_screen_candidates(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if float(row["train_fit_rate"]) >= Config.SCREEN_CANDIDATE_MIN_FIT_RATE
        and float(row["modal_probability"]) >= Config.SCREEN_CANDIDATE_MIN_MODAL
        and float(row["function_collision"])
        >= Config.SCREEN_CANDIDATE_MIN_COLLISION
    ]
    eligible.sort(
        key=lambda row: (
            float(row["function_collision"]),
            float(row["modal_probability"]),
        ),
        reverse=True,
    )
    if Config.SMOKE_TEST and not eligible:
        eligible = sorted(
            rows,
            key=lambda row: float(row["modal_probability"]),
            reverse=True,
        )[: Config.MAX_CONFIRM_CANDIDATES]
    selected: list[dict[str, Any]] = []
    for rank, row in enumerate(eligible[: Config.MAX_CONFIRM_CANDIDATES], start=1):
        selected.append({
            **row,
            "selection_rank": rank,
            "selection_used_complexity": False,
        })
    return selected


def screen_eligible_count(rows: Sequence[dict[str, str]]) -> int:
    return sum(
        float(row["train_fit_rate"]) >= Config.SCREEN_CANDIDATE_MIN_FIT_RATE
        and float(row["modal_probability"])
        >= Config.SCREEN_CANDIDATE_MIN_MODAL
        and float(row["function_collision"])
        >= Config.SCREEN_CANDIDATE_MIN_COLLISION
        for row in rows
    )


def anf_metrics(bits: np.ndarray) -> dict[str, Any]:
    coefficients = np.asarray(bits, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(2 ** Config.INPUT_BITS):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.asarray(
        [int(value).bit_count() for value in terms], dtype=np.int64
    )
    formula = ""
    if len(terms) <= 16:
        rendered: list[str] = []
        for mask in terms:
            if mask == 0:
                rendered.append("1")
                continue
            rendered.append("*".join(
                f"x{Config.INPUT_BITS-1-bit}"
                for bit in range(Config.INPUT_BITS)
                if int(mask) & (1 << bit)
            ))
        formula = " XOR ".join(rendered) if rendered else "0"
    return {
        "anf_degree": int(degrees.max()) if len(terms) else 0,
        "anf_term_count": int(len(terms)),
        "anf_literal_count": int(degrees.sum()),
        "anf_formula_if_short": formula,
    }


def essential_variables(bits: np.ndarray) -> list[int]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    result: list[int] = []
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        if np.any(bits[base] != bits[base | (1 << bit)]):
            result.append(Config.INPUT_BITS - 1 - bit)
    return sorted(result)


def boundary_metrics(bits: np.ndarray) -> dict[str, float]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    influences: list[float] = []
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        influences.append(float(np.mean(bits[base] != bits[base | (1 << bit)])))
    return {
        "total_influence": float(sum(influences)),
        "max_variable_influence": float(max(influences)),
    }


def subcube_formula(bits: np.ndarray, positive: bool) -> str:
    target = np.flatnonzero(bits == (1 if positive else 0))
    if not len(target):
        return ""
    fixed: list[tuple[int, int]] = []
    for bit in range(Config.INPUT_BITS):
        values = (target >> bit) & 1
        if np.all(values == values[0]):
            fixed.append((bit, int(values[0])))
    expected = np.asarray([
        value for value in range(2 ** Config.INPUT_BITS)
        if all(((value >> bit) & 1) == required for bit, required in fixed)
    ], dtype=np.int64)
    if not np.array_equal(expected, target):
        return ""
    if positive:
        literals = []
        for bit, required in fixed:
            name = f"x{Config.INPUT_BITS-1-bit}"
            literals.append(name if required else f"NOT {name}")
        return " AND ".join(literals)
    negated = []
    for bit, required in fixed:
        name = f"x{Config.INPUT_BITS-1-bit}"
        negated.append(f"NOT {name}" if required else name)
    return " OR ".join(negated)


def named_symbolic_rule(
    bits: np.ndarray,
    anf: dict[str, Any],
) -> tuple[str, str]:
    ones = int(bits.sum())
    if ones == 0:
        return "constant", "0"
    if ones == len(bits):
        return "constant", "1"
    conjunction = subcube_formula(bits, positive=True)
    if conjunction:
        return "literal_conjunction", conjunction
    disjunction = subcube_formula(bits, positive=False)
    if disjunction:
        return "literal_disjunction", disjunction
    if int(anf["anf_degree"]) <= 1:
        return "affine_gf2", str(anf["anf_formula_if_short"])
    inputs = truth_table_inputs().astype(np.uint8)
    weights = inputs.sum(axis=1)
    pattern: list[int] = []
    for weight in range(Config.INPUT_BITS + 1):
        local = bits[weights == weight]
        if len(np.unique(local)) != 1:
            break
        pattern.append(int(local[0]))
    if len(pattern) == Config.INPUT_BITS + 1:
        active = [index for index, value in enumerate(pattern) if value]
        return "symmetric_hamming_weight", f"popcount(x) IN {active}"
    return "", ""


def optimal_decision_tree(bits: np.ndarray) -> tuple[int, int]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def solve(fixed_mask: int, fixed_value: int) -> tuple[int, int]:
        selected = values[(values & fixed_mask) == fixed_value]
        outputs = bits[selected]
        if np.all(outputs == outputs[0]):
            return 1, 0
        best = (10 ** 9, 10 ** 9)
        for bit in range(Config.INPUT_BITS):
            bit_mask = 1 << bit
            if fixed_mask & bit_mask:
                continue
            left = solve(fixed_mask | bit_mask, fixed_value)
            right = solve(fixed_mask | bit_mask, fixed_value | bit_mask)
            candidate = (left[0] + right[0], 1 + max(left[1], right[1]))
            if candidate < best:
                best = candidate
        return best

    return solve(0, 0)


def robdd_node_count(bits: np.ndarray, order: Sequence[int]) -> int:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    unique_nodes: dict[tuple[int, int, int], int] = {}
    memo: dict[tuple[int, int, int], int] = {}

    def build(depth: int, fixed_mask: int, fixed_value: int) -> int:
        key = (depth, fixed_mask, fixed_value)
        if key in memo:
            return memo[key]
        selected = values[(values & fixed_mask) == fixed_value]
        outputs = bits[selected]
        if np.all(outputs == 0):
            return 0
        if np.all(outputs == 1):
            return 1
        bit = int(order[depth])
        bit_mask = 1 << bit
        low = build(depth + 1, fixed_mask | bit_mask, fixed_value)
        high = build(depth + 1, fixed_mask | bit_mask, fixed_value | bit_mask)
        if low == high:
            memo[key] = low
            return low
        node_key = (bit, low, high)
        node = unique_nodes.get(node_key)
        if node is None:
            node = len(unique_nodes) + 2
            unique_nodes[node_key] = node
        memo[key] = node
        return node

    build(0, 0, 0)
    return len(unique_nodes)


def bdd_orders() -> list[tuple[int, ...]]:
    natural = tuple(range(Config.INPUT_BITS - 1, -1, -1))
    reverse = tuple(reversed(natural))
    orders = [natural, reverse]
    seen = set(orders)
    rng = np.random.default_rng(Config.COMPLEXITY_SEED)
    while len(orders) < 2 + Config.BDD_RANDOM_ORDERS:
        candidate = tuple(map(int, rng.permutation(Config.INPUT_BITS)))
        if candidate not in seen:
            seen.add(candidate)
            orders.append(candidate)
    return orders


def complexity_metrics(
    bits: np.ndarray,
    orders: Sequence[Sequence[int]],
) -> dict[str, Any]:
    bits = np.asarray(bits, dtype=np.uint8)
    anf = anf_metrics(bits)
    essential = essential_variables(bits)
    leaves, depth = optimal_decision_tree(bits)
    bdd_counts = [robdd_node_count(bits, order) for order in orders]
    family, formula = named_symbolic_rule(bits, anf)
    if family:
        tier = 1
    elif (
        len(essential) <= 3
        or leaves <= 8
        or min(bdd_counts) <= 10
        or (
            int(anf["anf_term_count"]) <= 8
            and int(anf["anf_literal_count"]) <= 24
        )
    ):
        tier = 2
    elif (
        leaves <= 32
        or min(bdd_counts) <= 32
        or int(anf["anf_term_count"]) <= 32
    ):
        tier = 3
    else:
        tier = 4
    return {
        "truth_ones": int(bits.sum()),
        "essential_variable_count": len(essential),
        "essential_variables": essential,
        **anf,
        **boundary_metrics(bits),
        "optimal_decision_tree_leaves": leaves,
        "optimal_decision_tree_depth": depth,
        "robdd_nodes_min_tested": min(bdd_counts),
        "named_symbolic_family": family,
        "named_symbolic_formula": formula,
        "symbolic_screen_tier": tier,
        "symbolic_screen_readable": tier <= 2,
    }


def minimum_compatible_junta(spec: DatasetSpec) -> tuple[int, int, list[list[int]]]:
    inputs = truth_table_inputs().astype(np.uint8)
    indices = np.asarray(spec.train_indices, dtype=np.int64)
    labels = np.asarray(spec.train_labels, dtype=np.uint8)
    for count in range(Config.INPUT_BITS + 1):
        compatible: list[list[int]] = []
        for variables in itertools.combinations(range(Config.INPUT_BITS), count):
            codes = np.zeros(len(indices), dtype=np.int64)
            for offset, variable in enumerate(variables):
                codes |= (
                    inputs[indices, variable].astype(np.int64)
                    << (count - 1 - offset)
                )
            observed: dict[int, int] = {}
            valid = True
            for code, label in zip(codes, labels):
                code = int(code)
                label = int(label)
                if code in observed and observed[code] != label:
                    valid = False
                    break
                observed[code] = label
            if valid:
                compatible.append(list(map(int, variables)))
        if compatible:
            return count, len(compatible), compatible
    raise AssertionError("完整8变量函数必然能够拟合有限训练集。")


def analyze_confirmed(
    specs_by_index: dict[int, DatasetSpec],
    rows: Sequence[dict[str, str]],
    indices: np.ndarray,
    modal_packed: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modal_by_index = {
        int(dataset_index): packed
        for dataset_index, packed in zip(indices, modal_packed)
    }
    orders = bdd_orders()
    output: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for row in rows:
        dataset_index = int(row["dataset_index"])
        spec = specs_by_index[dataset_index]
        bits = unpack_truth(modal_by_index[dataset_index][None])[0]
        compatible = bool(np.array_equal(
            bits[np.asarray(spec.train_indices, dtype=np.int64)],
            np.asarray(spec.train_labels, dtype=np.uint8),
        ))
        minimum_junta, subset_count, subsets = minimum_compatible_junta(spec)
        metrics = complexity_metrics(bits, orders)
        strict = bool(
            float(row["train_fit_rate"]) >= Config.STRICT_MIN_FIT_RATE
            and float(row["modal_probability"])
            >= Config.STRICT_MODAL_PROBABILITY
            and float(row["function_collision"])
            >= Config.STRICT_FUNCTION_COLLISION
        )
        result = {
            **row,
            "modal_train_compatible": compatible,
            "strict_high_consensus": strict,
            "minimum_compatible_junta_variables": minimum_junta,
            "minimum_compatible_junta_subset_count": subset_count,
            "minimum_compatible_junta_subsets": subsets,
            "essential_variable_excess_over_minimum_junta": (
                int(metrics["essential_variable_count"]) - minimum_junta
            ),
            **metrics,
        }
        output.append(result)
        if strict and int(metrics["symbolic_screen_tier"]) == 4:
            counterexamples.append({
                **result,
                "candidate_status": (
                    "高共识Tier-4筛查候选；需要盲法公式审计和有界电路合成"
                ),
            })
    return output, counterexamples


def save_plot(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    strict = np.asarray([bool(row["strict_high_consensus"]) for row in rows])
    collision = np.asarray([float(row["function_collision"]) for row in rows])
    bdd = np.asarray([int(row["robdd_nodes_min_tested"]) for row in rows])
    leaves = np.asarray([int(row["optimal_decision_tree_leaves"]) for row in rows])
    axes[0].scatter(collision[~strict], bdd[~strict], s=24, alpha=0.6)
    axes[0].scatter(collision[strict], bdd[strict], s=48, marker="x", color="red")
    axes[1].scatter(collision[~strict], leaves[~strict], s=24, alpha=0.6)
    axes[1].scatter(collision[strict], leaves[strict], s=48, marker="x", color="red")
    axes[0].set_xlabel("complete-function collision")
    axes[0].set_ylabel("minimum tested ROBDD nodes")
    axes[1].set_xlabel("complete-function collision")
    axes[1].set_ylabel("optimal decision-tree leaves")
    for axis in axes:
        axis.axvline(Config.STRICT_FUNCTION_COLLISION, color="black", ls="--")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "confirmed_consensus_vs_complexity.png", dpi=180)
    plt.close(figure)


def package_results(output_dir: Path) -> Path:
    archive = output_dir.parent / f"{output_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir.parent))
    return archive


def main() -> None:
    apply_smoke_overrides()
    output_dir = prepare_result_dir()
    specs, dataset_rows = build_datasets()
    datasets_path = output_dir / "datasets.csv"
    if not datasets_path.exists():
        write_csv(datasets_path, dataset_rows)

    device = torch.device(Config.DEVICE)
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    start_time = time.perf_counter()
    print("=== 8-bit Consensus Symbolicity Large-Scale Validation ===", flush=True)
    print(
        f"device={device} | datasets={len(specs):,} | n={Config.TRAIN_COUNT} | "
        f"MLP=8->{Config.WIDTH}x{Config.HIDDEN_LAYERS}->1 tanh | "
        f"params/model={parameter_count_per_model():,}",
        flush=True,
    )
    print(
        f"screen={Config.SCREEN_SEED_COUNT} seeds x {Config.SCREEN_MAX_STEPS:,} steps | "
        f"confirm={Config.CONFIRM_SEED_COUNT:,} fresh seeds x "
        f"{Config.CONFIRM_MAX_STEPS:,} steps",
        flush=True,
    )

    try:
        all_indices = [spec.dataset_index for spec in specs]
        run_phase(
            "screen",
            specs,
            all_indices,
            Config.SCREEN_SEED_COUNT,
            Config.SCREEN_INITIALIZATION_SEED,
            Config.SCREEN_MAX_STEPS,
            Config.SCREEN_EVAL_STEPS,
            Config.SCREEN_DATASETS_PER_SHARD,
            output_dir,
            device,
        )
        screen_rows, _ = aggregate_phase("screen", output_dir)
        aggregate_predictions("screen", output_dir)
        candidates = select_screen_candidates(screen_rows)
        eligible_count = screen_eligible_count(screen_rows)
        write_csv(output_dir / "screen_candidates.csv", candidates)
        print(
            f"筛选完成：{len(screen_rows):,} datasets，"
            f"宽松候选={eligible_count:,}，进入独立确认={len(candidates):,}",
            flush=True,
        )

        candidate_indices = [int(row["dataset_index"]) for row in candidates]
        candidate_specs = [specs[index] for index in candidate_indices]
        if candidate_specs:
            run_phase(
                "confirm",
                candidate_specs,
                candidate_indices,
                Config.CONFIRM_SEED_COUNT,
                Config.CONFIRM_INITIALIZATION_SEED,
                Config.CONFIRM_MAX_STEPS,
                Config.CONFIRM_EVAL_STEPS,
                Config.CONFIRM_DATASETS_PER_SHARD,
                output_dir,
                device,
            )
            confirm_rows, _ = aggregate_phase("confirm", output_dir)
            confirm_indices, _, confirm_modal = aggregate_predictions(
                "confirm", output_dir
            )
        else:
            confirm_rows = []
            write_csv(output_dir / "confirm_final_summary.csv", [])
            write_csv(output_dir / "confirm_trajectory.csv", [])
            confirm_indices, _, confirm_modal = aggregate_predictions(
                "confirm", output_dir
            )
    except KeyboardInterrupt:
        write_json(output_dir / "progress.json", {
            "status": "interrupted",
            "elapsed_seconds": time.perf_counter() - start_time,
            "resume_instruction": "直接重跑同一脚本，将跳过完整分片。",
        })
        print("收到Ctrl+C。完整分片均已保存，直接重跑即可继续。", flush=True)
        return

    specs_by_index = {spec.dataset_index: spec for spec in specs}
    analyzed, counterexamples = analyze_confirmed(
        specs_by_index,
        confirm_rows,
        confirm_indices,
        confirm_modal,
    )
    write_csv(output_dir / "confirmed_complexity.csv", analyzed)
    high = [row for row in analyzed if row["strict_high_consensus"]]
    write_csv(output_dir / "confirmed_high_consensus.csv", high)
    write_csv(output_dir / "counterexample_candidates.csv", counterexamples)
    save_plot(output_dir, analyzed)

    high_count = len(high)
    counterexample_count = len(counterexamples)
    upper_95 = (
        1.0 - 0.05 ** (1.0 / high_count)
        if high_count and counterexample_count == 0
        else None
    )
    summary = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - start_time,
        "screen_dataset_count": len(specs),
        "screen_eligible_candidate_count": eligible_count,
        "screen_candidate_count": len(candidates),
        "screen_candidates_truncated": eligible_count > len(candidates),
        "confirmed_dataset_count": len(analyzed),
        "strict_high_consensus_count": high_count,
        "strict_high_consensus_tier_counts": {
            str(tier): sum(
                int(row["symbolic_screen_tier"]) == tier for row in high
            )
            for tier in range(1, 5)
        },
        "counterexample_candidate_count": counterexample_count,
        "zero_counterexample_binomial_95pct_upper_bound": upper_95,
        "verdict": (
            "screening_counterexample_found"
            if counterexample_count
            else (
                "no_strict_high_consensus_after_fresh_seed_confirmation"
                if not high_count
                else "no_tier4_counterexample_among_confirmed_high_consensus_modes"
            )
        ),
        "important_scope": (
            "该统计结论只适用于固定tanh16x2训练协议；跨协议证据来自单独的"
            "width/intervention实验。Tier-4也只是需要进一步合成审计的候选。"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "runtime.json", {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "elapsed_seconds": summary["elapsed_seconds"],
    })
    archive = package_results(output_dir) if Config.PACKAGE_RESULTS else None

    print("=== 最终判决 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录：{output_dir}", flush=True)
    if archive:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
