"""
静态 loss-conditioned prior 与优化器函数分布的直接比较。

固定：

- 4-bit -> 1-bit；
- 4 -> 16 x 2 -> 1 tanh MLP；
- balanced AND n=10 训练集；
- Xavier uniform 初始化测度；
- raw BCE。

先采样大量未训练网络，记录初始化 prior 中 loss 与完整16点 hard function 的
联合分布。再用 AdamW、full-batch SGD、momentum SGD 从同一初始化分布训练，
在首次跨过各 raw BCE 阈值时记录 hard function。对每个优化器 cohort，按其
实际 crossing loss 从 prior 的细 loss bin 中抽取匹配 cohort，比较完整65,536
函数分布的 JSD、TV、target odds、熵和 top functions。

本实验只裁决 prior 与 optimizer 具有重叠支持的中等 loss 区域，不把结果
外推到初始化 prior 几乎采不到的极深 low-loss 尾部。
"""

from __future__ import annotations

import csv
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
    INPUT_BITS = 4
    WIDTH = 16
    HIDDEN_LAYERS = 2
    ACTIVATION = "tanh"

    TRAIN_INDICES = (1, 2, 3, 5, 7, 8, 11, 12, 14, 15)
    LOSS_THRESHOLDS = (0.68, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)

    PRIOR_MODEL_COUNT = 33_554_432
    PRIOR_MICRO_BATCH = 65_536
    PRIOR_SEED = 20260910
    LOSS_BIN_WIDTH = 0.002
    LOSS_BIN_MAX = 2.0
    RESERVOIR_PER_BIN = 4_096
    MATCH_SAMPLES_PER_OPT_MODEL = 8
    MIN_MATCH_BIN_COUNT = 200
    MAX_MATCH_BIN_DISTANCE = 1

    OPTIMIZER_MODEL_COUNT = 4_096
    OPTIMIZER_MAX_STEPS = 5_000
    OPTIMIZER_SEED = 20260911
    OPTIMIZERS = (
        ("adamw", 1e-3, 0.0),
        ("full_batch_sgd", 5e-2, 0.0),
        ("momentum_sgd", 2e-2, 0.9),
    )
    LOG_INTERVAL = 100

    MIN_STATIC_COHORT = 200
    TOP_FUNCTIONS = 20
    MATCH_RANDOM_SEED = 20260912

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_static_loss_vs_optimizer_distribution")
    PACKAGE_RESULTS = True
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass
class PriorSummary:
    baseline_counts: np.ndarray
    cumulative_counts: np.ndarray
    cumulative_sample_counts: np.ndarray
    bin_counts: np.ndarray
    reservoir_ids: np.ndarray
    reservoir_keys: np.ndarray
    loss_min: float
    loss_max: float


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.PRIOR_MODEL_COUNT = 16_384
    Config.PRIOR_MICRO_BATCH = 1_024
    Config.RESERVOIR_PER_BIN = 64
    Config.MATCH_SAMPLES_PER_OPT_MODEL = 2
    Config.MIN_MATCH_BIN_COUNT = 2
    Config.MAX_MATCH_BIN_DISTANCE = 1
    Config.OPTIMIZER_MODEL_COUNT = 64
    Config.OPTIMIZER_MAX_STEPS = 100
    Config.LOSS_THRESHOLDS = (0.70, 0.68, 0.65, 0.60)
    Config.LOG_INTERVAL = 20
    Config.MIN_STATIC_COHORT = 10
    Config.TOP_FUNCTIONS = 5
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_static_loss_vs_optimizer_distribution"
    )
    Config.OVERWRITE_RESULT_DIR = True


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
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


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


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


def validate_config() -> None:
    if Config.INPUT_BITS != 4:
        raise ValueError("当前实验固定为4-bit输入。")
    if Config.WIDTH < 1 or Config.HIDDEN_LAYERS != 2:
        raise ValueError("当前实现要求两个隐藏层且宽度为正。")
    thresholds = tuple(float(value) for value in Config.LOSS_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("LOSS_THRESHOLDS 必须严格从高到低且不重复。")
    if Config.PRIOR_MODEL_COUNT % Config.PRIOR_MICRO_BATCH:
        raise ValueError("PRIOR_MODEL_COUNT 必须整除 PRIOR_MICRO_BATCH。")
    if Config.LOSS_BIN_WIDTH <= 0 or Config.LOSS_BIN_MAX <= 0:
        raise ValueError("loss bin 参数必须为正。")


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def and_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs[:, 0] & inputs[:, 1]).astype(np.uint8)


def outputs_to_function_id(outputs: np.ndarray) -> int:
    bits = np.asarray(outputs, dtype=np.uint64).reshape(-1)
    powers = np.left_shift(np.uint64(1), np.arange(16, dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def function_bits(function_id: int) -> str:
    return "".join(str((function_id >> index) & 1) for index in range(16))


def function_ids_from_logits(logits: torch.Tensor) -> np.ndarray:
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=logits.device),
        torch.arange(16, dtype=torch.int64, device=logits.device),
    )
    values = ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)
    return values.cpu().numpy().astype(np.uint16)


def fan_in_uniform(
    shape: tuple[int, ...],
    fan_in: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    bound = 1.0 / math.sqrt(fan_in)
    return torch.empty(shape, device=device).uniform_(
        -bound, bound, generator=generator
    )


def sample_prior_logits(
    count: int,
    inputs: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    device = inputs.device
    hidden = inputs[None].expand(count, -1, -1)
    dimensions = [Config.INPUT_BITS, Config.WIDTH, Config.WIDTH, 1]
    for layer_index, (fan_in, fan_out) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        weight = fan_in_uniform(
            (count, fan_out, fan_in), fan_in, generator, device
        )
        bias = fan_in_uniform(
            (count, fan_out), fan_in, generator, device
        )
        hidden = torch.bmm(hidden, weight.transpose(1, 2))
        hidden = hidden + bias[:, None, :]
        if layer_index < len(dimensions) - 2:
            hidden = torch.tanh(hidden)
    return hidden.squeeze(-1)


def raw_bce_losses(
    logits: torch.Tensor,
    train_indices: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    local_logits = logits[:, train_indices]
    local_targets = targets[train_indices]
    return F.binary_cross_entropy_with_logits(
        local_logits,
        local_targets[None].expand_as(local_logits),
        reduction="none",
    ).mean(dim=1)


def update_bin_reservoir(
    reservoir_ids: np.ndarray,
    reservoir_keys: np.ndarray,
    bin_counts: np.ndarray,
    bin_indices: np.ndarray,
    function_ids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for bin_index in np.unique(bin_indices):
        mask = bin_indices == bin_index
        local_ids = function_ids[mask]
        local_keys = rng.random(len(local_ids), dtype=np.float32)
        bin_counts[bin_index] += len(local_ids)
        existing_mask = np.isfinite(reservoir_keys[bin_index])
        existing_ids = reservoir_ids[bin_index, existing_mask]
        existing_keys = reservoir_keys[bin_index, existing_mask]
        combined_ids = np.concatenate([existing_ids, local_ids])
        combined_keys = np.concatenate([existing_keys, local_keys])
        keep = min(Config.RESERVOIR_PER_BIN, len(combined_ids))
        if keep < len(combined_ids):
            selected = np.argpartition(combined_keys, keep - 1)[:keep]
            combined_ids = combined_ids[selected]
            combined_keys = combined_keys[selected]
        reservoir_ids[bin_index].fill(0)
        reservoir_keys[bin_index].fill(np.inf)
        reservoir_ids[bin_index, :keep] = combined_ids
        reservoir_keys[bin_index, :keep] = combined_keys


def sample_prior_joint(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> PriorSummary:
    threshold_count = len(Config.LOSS_THRESHOLDS)
    baseline_counts = np.zeros(65_536, dtype=np.int64)
    cumulative_counts = np.zeros(
        (threshold_count, 65_536), dtype=np.int64
    )
    cumulative_sample_counts = np.zeros(threshold_count, dtype=np.int64)
    bin_total = int(math.ceil(Config.LOSS_BIN_MAX / Config.LOSS_BIN_WIDTH)) + 1
    bin_counts = np.zeros(bin_total, dtype=np.int64)
    reservoir_ids = np.zeros(
        (bin_total, Config.RESERVOIR_PER_BIN), dtype=np.uint16
    )
    reservoir_keys = np.full(
        (bin_total, Config.RESERVOIR_PER_BIN), np.inf, dtype=np.float32
    )
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(Config.PRIOR_SEED)
    reservoir_rng = np.random.default_rng(Config.PRIOR_SEED + 1)
    train_indices = torch.tensor(
        Config.TRAIN_INDICES, dtype=torch.int64, device=inputs.device
    )
    minimum = float("inf")
    maximum = float("-inf")
    started = time.perf_counter()

    for start in range(0, Config.PRIOR_MODEL_COUNT, Config.PRIOR_MICRO_BATCH):
        logits = sample_prior_logits(
            Config.PRIOR_MICRO_BATCH, inputs, generator
        )
        losses = raw_bce_losses(logits, train_indices, targets)
        ids = function_ids_from_logits(logits)
        losses_np = losses.cpu().numpy().astype(np.float32)
        minimum = min(minimum, float(losses_np.min()))
        maximum = max(maximum, float(losses_np.max()))
        baseline_counts += np.bincount(ids, minlength=65_536)
        for threshold_index, threshold in enumerate(Config.LOSS_THRESHOLDS):
            mask = losses_np <= threshold
            if np.any(mask):
                cumulative_counts[threshold_index] += np.bincount(
                    ids[mask], minlength=65_536
                )
                cumulative_sample_counts[threshold_index] += int(mask.sum())
        bins = np.floor(losses_np / Config.LOSS_BIN_WIDTH).astype(np.int64)
        bins = np.clip(bins, 0, bin_total - 1)
        update_bin_reservoir(
            reservoir_ids,
            reservoir_keys,
            bin_counts,
            bins,
            ids,
            reservoir_rng,
        )
        completed = start + Config.PRIOR_MICRO_BATCH
        if completed == Config.PRIOR_MODEL_COUNT or completed % (
            Config.PRIOR_MICRO_BATCH * 32
        ) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"prior {completed:,}/{Config.PRIOR_MODEL_COUNT:,} | "
                f"{completed / max(elapsed, 1e-9):,.0f} models/s | "
                f"loss min={minimum:.4f}",
                flush=True,
            )
        del logits, losses

    return PriorSummary(
        baseline_counts=baseline_counts,
        cumulative_counts=cumulative_counts,
        cumulative_sample_counts=cumulative_sample_counts,
        bin_counts=bin_counts,
        reservoir_ids=reservoir_ids,
        reservoir_keys=reservoir_keys,
        loss_min=minimum,
        loss_max=maximum,
    )


class BatchedTanhMLP(nn.Module):
    def __init__(self, model_count: int, device: torch.device) -> None:
        super().__init__()
        self.model_count = model_count
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        generator = torch.Generator(device=device)
        generator.manual_seed(Config.OPTIMIZER_SEED)
        dimensions = [Config.INPUT_BITS, Config.WIDTH, Config.WIDTH, 1]
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            self.weights.append(nn.Parameter(fan_in_uniform(
                (model_count, fan_out, fan_in),
                fan_in,
                generator,
                device,
            )))
            self.biases.append(nn.Parameter(fan_in_uniform(
                (model_count, fan_out),
                fan_in,
                generator,
                device,
            )))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs[None].expand(self.model_count, -1, -1)
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if layer_index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def make_optimizer(
    name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.0
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=0.0,
    )


def train_optimizer_distribution(
    name: str,
    learning_rate: float,
    momentum: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, np.ndarray]:
    model = BatchedTanhMLP(Config.OPTIMIZER_MODEL_COUNT, inputs.device)
    optimizer = make_optimizer(name, model, learning_rate, momentum)
    train_indices = torch.tensor(
        Config.TRAIN_INDICES, dtype=torch.int64, device=inputs.device
    )
    thresholds = torch.tensor(
        Config.LOSS_THRESHOLDS, dtype=torch.float32, device=inputs.device
    )
    shape = (Config.OPTIMIZER_MODEL_COUNT, len(Config.LOSS_THRESHOLDS))
    crossed = torch.zeros(shape, dtype=torch.bool, device=inputs.device)
    crossing_step = torch.full(
        shape, -1, dtype=torch.int32, device=inputs.device
    )
    crossing_loss = torch.full(
        shape, float("nan"), dtype=torch.float32, device=inputs.device
    )
    crossing_function_id = torch.full(
        shape, -1, dtype=torch.int64, device=inputs.device
    )
    crossing_train_exact = torch.zeros(
        shape, dtype=torch.bool, device=inputs.device
    )
    started = time.perf_counter()

    for step in range(Config.OPTIMIZER_MAX_STEPS + 1):
        logits = model(inputs)
        losses = raw_bce_losses(logits, train_indices, targets)
        with torch.no_grad():
            predictions = logits >= 0
            train_exact = (
                predictions[:, train_indices]
                == targets[train_indices].bool()[None]
            ).all(dim=1)
            ids_np = function_ids_from_logits(logits)
            ids = torch.from_numpy(ids_np.astype(np.int64)).to(inputs.device)
            new = (losses[:, None] <= thresholds[None]) & ~crossed
            if bool(new.any().item()):
                crossed[new] = True
                crossing_step[new] = step
                crossing_loss[new] = losses[:, None].expand_as(new)[new]
                crossing_function_id[new] = ids[:, None].expand_as(new)[new]
                crossing_train_exact[new] = (
                    train_exact[:, None].expand_as(new)[new]
                )
        if step <= 20 or step % Config.LOG_INTERVAL == 0:
            reached = crossed.float().mean(dim=0)
            print(
                f"{name} step={step:>5,} | loss median="
                f"{losses.median().item():.4f} | reached="
                + ",".join(
                    f"{threshold:g}:{fraction.item():.0%}"
                    for threshold, fraction in zip(
                        Config.LOSS_THRESHOLDS, reached
                    )
                ),
                flush=True,
            )
        if bool(crossed[:, -1].all().item()) or step == Config.OPTIMIZER_MAX_STEPS:
            break
        per_model = losses
        optimizer.zero_grad(set_to_none=True)
        per_model.sum().backward()
        optimizer.step()

    print(
        f"{name} finished in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return {
        "crossing_step": crossing_step.cpu().numpy(),
        "crossing_loss": crossing_loss.cpu().numpy(),
        "crossing_function_id": crossing_function_id.cpu().numpy(),
        "crossing_train_exact": crossing_train_exact.cpu().numpy(),
    }


def probability_from_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / total


def distribution_entropy(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-np.sum(probability[positive] * np.log2(probability[positive])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    m = 0.5 * (p + q)
    value = 0.0
    for distribution in (p, q):
        mask = distribution > 0
        value += 0.5 * float(np.sum(
            distribution[mask]
            * np.log2(distribution[mask] / m[mask])
        ))
    return value


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(0.5 * np.abs(p - q).sum())


def nearest_supported_bin(
    requested: int,
    bin_counts: np.ndarray,
    reservoir_keys: np.ndarray,
) -> int | None:
    bin_count = len(reservoir_keys)
    maximum_distance = min(Config.MAX_MATCH_BIN_DISTANCE, bin_count - 1)
    for distance in range(maximum_distance + 1):
        for candidate in (requested - distance, requested + distance):
            if candidate < 0 or candidate >= bin_count:
                continue
            if (
                bin_counts[candidate] >= Config.MIN_MATCH_BIN_COUNT
                and np.isfinite(reservoir_keys[candidate]).any()
            ):
                return candidate
    return None


def draw_loss_matched_static_ids(
    losses: np.ndarray,
    prior: PriorSummary,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int | None, float | None]:
    output: list[np.ndarray] = []
    mismatch: list[float] = []
    source_counts: list[int] = []
    matched_mask = np.zeros(len(losses), dtype=bool)
    for loss_index, loss in enumerate(losses):
        requested = int(math.floor(float(loss) / Config.LOSS_BIN_WIDTH))
        requested = min(max(requested, 0), len(prior.bin_counts) - 1)
        selected_bin = nearest_supported_bin(
            requested, prior.bin_counts, prior.reservoir_keys
        )
        if selected_bin is None:
            continue
        matched_mask[loss_index] = True
        valid = np.flatnonzero(np.isfinite(prior.reservoir_keys[selected_bin]))
        chosen = rng.choice(
            valid,
            size=Config.MATCH_SAMPLES_PER_OPT_MODEL,
            replace=len(valid) < Config.MATCH_SAMPLES_PER_OPT_MODEL,
        )
        output.append(prior.reservoir_ids[selected_bin, chosen])
        bin_center = (selected_bin + 0.5) * Config.LOSS_BIN_WIDTH
        mismatch.append(abs(float(loss) - bin_center))
        source_counts.append(int(prior.bin_counts[selected_bin]))
    if not output:
        return (
            np.empty(0, dtype=np.uint16),
            matched_mask,
            float("nan"),
            None,
            None,
        )
    return (
        np.concatenate(output),
        matched_mask,
        float(np.mean(mismatch)),
        int(np.min(source_counts)),
        float(np.median(source_counts)),
    )


def distribution_row(
    label: str,
    counts: np.ndarray,
    target_id: int,
) -> dict[str, Any]:
    probability = probability_from_counts(counts)
    top_id = int(np.argmax(counts)) if counts.sum() else -1
    return {
        "distribution": label,
        "sample_count": int(counts.sum()),
        "function_support": int(np.count_nonzero(counts)),
        "function_entropy_bits": distribution_entropy(probability),
        "target_function_id": target_id,
        "target_probability": float(probability[target_id]),
        "top_function_id": top_id,
        "top_function_hex": f"0x{top_id:04X}" if top_id >= 0 else None,
        "top_function_probability": (
            float(probability[top_id]) if top_id >= 0 else None
        ),
    }


def append_top_functions(
    rows: list[dict[str, Any]],
    label: str,
    counts: np.ndarray,
    target_id: int,
) -> None:
    total = counts.sum()
    if total == 0:
        return
    top_ids = np.argsort(counts)[::-1][: Config.TOP_FUNCTIONS]
    for rank, function_id in enumerate(top_ids, start=1):
        count = int(counts[function_id])
        if count == 0:
            break
        rows.append({
            "distribution": label,
            "rank": rank,
            "function_id": int(function_id),
            "function_hex": f"0x{int(function_id):04X}",
            "truth_table_x0_to_x15": function_bits(int(function_id)),
            "count": count,
            "probability": count / total,
            "is_target": int(function_id) == target_id,
        })


def analyze_distributions(
    prior: PriorSummary,
    optimizer_results: dict[str, dict[str, np.ndarray]],
    target_id: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    distribution_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    optimizer_pair_rows: list[dict[str, Any]] = []
    distribution_counts: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(Config.MATCH_RANDOM_SEED)

    baseline_label = "prior_all"
    distribution_rows.append(distribution_row(
        baseline_label, prior.baseline_counts, target_id
    ))
    distribution_counts[baseline_label] = prior.baseline_counts
    append_top_functions(
        top_rows, baseline_label, prior.baseline_counts, target_id
    )

    optimizer_probabilities: dict[tuple[str, float], np.ndarray] = {}
    for threshold_index, threshold in enumerate(Config.LOSS_THRESHOLDS):
        static_counts = prior.cumulative_counts[threshold_index]
        static_label = f"prior_cumulative_le_{threshold:g}"
        distribution_rows.append(distribution_row(
            static_label, static_counts, target_id
        ))
        distribution_counts[static_label] = static_counts
        append_top_functions(top_rows, static_label, static_counts, target_id)

        for optimizer_name, result in optimizer_results.items():
            valid = result["crossing_step"][:, threshold_index] >= 0
            ids = result["crossing_function_id"][valid, threshold_index]
            losses = result["crossing_loss"][valid, threshold_index]
            ids = ids.astype(np.int64)
            optimizer_counts = np.bincount(ids, minlength=65_536)
            optimizer_label = f"{optimizer_name}_cross_{threshold:g}"
            distribution_rows.append(distribution_row(
                optimizer_label, optimizer_counts, target_id
            ))
            distribution_counts[optimizer_label] = optimizer_counts
            append_top_functions(
                top_rows, optimizer_label, optimizer_counts, target_id
            )
            optimizer_probability = probability_from_counts(optimizer_counts)
            optimizer_probabilities[(optimizer_name, threshold)] = (
                optimizer_probability
            )

            (
                matched_ids,
                matched_optimizer_mask,
                mismatch,
                minimum_source_bin_count,
                median_source_bin_count,
            ) = draw_loss_matched_static_ids(losses, prior, rng)
            matched_optimizer_counts = np.bincount(
                ids[matched_optimizer_mask], minlength=65_536
            )
            matched_counts = np.bincount(
                matched_ids.astype(np.int64), minlength=65_536
            )
            matched_label = (
                f"prior_loss_matched_to_{optimizer_name}_{threshold:g}"
            )
            distribution_rows.append(distribution_row(
                matched_label, matched_counts, target_id
            ))
            distribution_counts[matched_label] = matched_counts
            append_top_functions(
                top_rows, matched_label, matched_counts, target_id
            )

            matched_probability = probability_from_counts(matched_counts)
            matched_optimizer_probability = probability_from_counts(
                matched_optimizer_counts
            )
            cumulative_probability = probability_from_counts(static_counts)
            static_match_supported = bool(len(matched_ids))
            comparison_rows.append({
                "optimizer": optimizer_name,
                "threshold": threshold,
                "optimizer_sample_count": int(valid.sum()),
                "optimizer_crossing_loss_mean": (
                    float(np.mean(losses)) if len(losses) else None
                ),
                "optimizer_crossing_loss_min": (
                    float(np.min(losses)) if len(losses) else None
                ),
                "optimizer_crossing_loss_max": (
                    float(np.max(losses)) if len(losses) else None
                ),
                "matched_static_sample_count": int(len(matched_ids)),
                "matched_optimizer_sample_count": int(
                    matched_optimizer_mask.sum()
                ),
                "matched_optimizer_fraction": float(
                    matched_optimizer_mask.mean()
                ) if len(matched_optimizer_mask) else 0.0,
                "static_match_supported": static_match_supported,
                "minimum_source_bin_count": minimum_source_bin_count,
                "median_source_bin_count": median_source_bin_count,
                "mean_loss_bin_mismatch": mismatch,
                "js_optimizer_vs_loss_matched_prior": js_divergence(
                    matched_optimizer_probability, matched_probability
                ),
                "tv_optimizer_vs_loss_matched_prior": total_variation(
                    matched_optimizer_probability, matched_probability
                ),
                "js_optimizer_vs_cumulative_prior": js_divergence(
                    optimizer_probability, cumulative_probability
                ),
                "tv_optimizer_vs_cumulative_prior": total_variation(
                    optimizer_probability, cumulative_probability
                ),
                "optimizer_target_probability": float(
                    optimizer_probability[target_id]
                ),
                "matched_optimizer_target_probability": (
                    float(matched_optimizer_probability[target_id])
                    if static_match_supported else float("nan")
                ),
                "loss_matched_prior_target_probability": (
                    float(matched_probability[target_id])
                    if static_match_supported else float("nan")
                ),
                "cumulative_prior_target_probability": float(
                    cumulative_probability[target_id]
                ),
                "optimizer_entropy_bits": distribution_entropy(
                    optimizer_probability
                ),
                "loss_matched_prior_entropy_bits": distribution_entropy(
                    matched_probability
                ) if static_match_supported else float("nan"),
            })

    optimizer_names = list(optimizer_results)
    for threshold in Config.LOSS_THRESHOLDS:
        for first_index, first in enumerate(optimizer_names):
            for second in optimizer_names[first_index + 1 :]:
                p = optimizer_probabilities[(first, threshold)]
                q = optimizer_probabilities[(second, threshold)]
                optimizer_pair_rows.append({
                    "threshold": threshold,
                    "optimizer_a": first,
                    "optimizer_b": second,
                    "js_divergence": js_divergence(p, q),
                    "total_variation": total_variation(p, q),
                })
    return (
        distribution_rows,
        top_rows,
        comparison_rows,
        optimizer_pair_rows,
        distribution_counts,
    )


def plot_comparisons(
    output_dir: Path,
    comparison_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for optimizer in sorted({row["optimizer"] for row in comparison_rows}):
        rows = [row for row in comparison_rows if row["optimizer"] == optimizer]
        axes[0].plot(
            [row["threshold"] for row in rows],
            [row["js_optimizer_vs_loss_matched_prior"] for row in rows],
            marker="o",
            label=optimizer,
        )
        axes[1].plot(
            [row["threshold"] for row in rows],
            [row["optimizer_target_probability"] for row in rows],
            marker="o",
            label=f"{optimizer} target",
        )
        axes[1].plot(
            [row["threshold"] for row in rows],
            [row["loss_matched_prior_target_probability"] for row in rows],
            linestyle="--",
            alpha=0.7,
            label=f"matched prior for {optimizer}",
        )
    for axis in axes:
        axis.invert_xaxis()
        axis.set_xlabel("raw BCE threshold (deeper to the right)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Jensen-Shannon divergence (bits)")
    axes[0].set_title("Optimizer vs loss-matched static prior")
    axes[1].set_ylabel("AND target probability")
    axes[1].set_title("Target mass")
    figure.savefig(output_dir / "static_vs_optimizer_distribution.png", dpi=180)
    plt.close(figure)


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    output_dir = prepare_result_dir()
    write_json(output_dir / "config.json", config_dict())
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE 要求 CUDA，但 PyTorch 看不到 GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    inputs_np = truth_table_inputs()
    targets_np = and_targets(inputs_np)
    target_id = outputs_to_function_id(targets_np)
    inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    targets = torch.from_numpy(targets_np.astype(np.float32)).to(device)

    print("=== Static loss geometry vs optimizer-induced function distribution ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(
        f"task=balanced AND n=10 | target=0x{target_id:04X} | "
        f"network=4->{Config.WIDTH}x2->1 tanh"
    )
    print(
        f"prior={Config.PRIOR_MODEL_COUNT:,} | optimizer models="
        f"{Config.OPTIMIZER_MODEL_COUNT:,} | result={output_dir.resolve()}"
    )

    print("\n--- sample static initialization prior ---", flush=True)
    prior = sample_prior_joint(inputs, targets)
    np.savez_compressed(
        output_dir / "prior_summary_counts.npz",
        baseline_counts=prior.baseline_counts,
        cumulative_counts=prior.cumulative_counts,
        cumulative_sample_counts=prior.cumulative_sample_counts,
        thresholds=np.asarray(Config.LOSS_THRESHOLDS),
        bin_counts=prior.bin_counts,
    )
    prior_threshold_rows = [
        {
            "threshold": threshold,
            "sample_count": int(prior.cumulative_sample_counts[index]),
            "sample_fraction": (
                prior.cumulative_sample_counts[index]
                / Config.PRIOR_MODEL_COUNT
            ),
            "target_count": int(
                prior.cumulative_counts[index, target_id]
            ),
            "target_probability": (
                prior.cumulative_counts[index, target_id]
                / prior.cumulative_sample_counts[index]
                if prior.cumulative_sample_counts[index] else None
            ),
        }
        for index, threshold in enumerate(Config.LOSS_THRESHOLDS)
    ]
    write_csv(output_dir / "prior_threshold_summary.csv", prior_threshold_rows)

    optimizer_results: dict[str, dict[str, np.ndarray]] = {}
    optimizer_crossing_rows: list[dict[str, Any]] = []
    for name, learning_rate, momentum in Config.OPTIMIZERS:
        print(f"\n--- optimizer: {name} ---", flush=True)
        result = train_optimizer_distribution(
            name,
            float(learning_rate),
            float(momentum),
            inputs,
            targets,
        )
        optimizer_results[name] = result
        np.savez_compressed(output_dir / f"crossing_{name}.npz", **result)
        for threshold_index, threshold in enumerate(Config.LOSS_THRESHOLDS):
            reached = result["crossing_step"][:, threshold_index] >= 0
            optimizer_crossing_rows.append({
                "optimizer": name,
                "threshold": threshold,
                "reached_count": int(reached.sum()),
                "reached_fraction": float(reached.mean()),
                "median_crossing_step": (
                    float(np.median(
                        result["crossing_step"][reached, threshold_index]
                    ))
                    if reached.any() else None
                ),
                "mean_crossing_loss": (
                    float(np.mean(
                        result["crossing_loss"][reached, threshold_index]
                    ))
                    if reached.any() else None
                ),
                "train_exact_at_crossing_fraction": (
                    float(np.mean(
                        result["crossing_train_exact"][
                            reached, threshold_index
                        ]
                    ))
                    if reached.any() else None
                ),
            })
    write_csv(output_dir / "optimizer_crossing_summary.csv", optimizer_crossing_rows)

    (
        distribution_rows,
        top_rows,
        comparison_rows,
        optimizer_pair_rows,
        distribution_counts,
    ) = analyze_distributions(prior, optimizer_results, target_id)
    write_csv(output_dir / "distribution_summary.csv", distribution_rows)
    write_csv(output_dir / "top_functions.csv", top_rows)
    write_csv(output_dir / "static_optimizer_comparison.csv", comparison_rows)
    write_csv(output_dir / "optimizer_pair_comparison.csv", optimizer_pair_rows)
    np.savez_compressed(
        output_dir / "distribution_counts.npz", **distribution_counts
    )
    plot_comparisons(output_dir, comparison_rows)

    summary = {
        "status": "completed",
        "protocol": "static_loss_vs_optimizer_distribution_v1",
        "target_function_id": target_id,
        "target_function_hex": f"0x{target_id:04X}",
        "prior_loss_min": prior.loss_min,
        "prior_loss_max": prior.loss_max,
        "prior_model_count": Config.PRIOR_MODEL_COUNT,
        "optimizer_model_count": Config.OPTIMIZER_MODEL_COUNT,
        "comparison_rows": comparison_rows,
        "interpretation": {
            "small_js_all_optimizers": (
                "静态 loss-conditioned 几何是一阶函数选择规律；优化器主要承担输运。"
            ),
            "optimizer_specific_but_same_direction": (
                "静态几何决定粗方向，优化器提供不可忽略的二阶重加权。"
            ),
            "large_optimizer_static_gap": (
                "SGD/Adam具有静态 prior 条件化无法吸收的不可约函数选择偏置。"
            ),
            "scope": (
                "本实验只覆盖初始化 prior 有足够样本支持的中等 loss；极深 low-loss 需要SMC/Gibbs扩展。"
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    archive_path: Path | None = None
    if Config.PACKAGE_RESULTS:
        archive_path = create_archive(output_dir)
    print("\n=== 实验完成 ===", flush=True)
    print(f"prior loss range: [{prior.loss_min:.6g}, {prior.loss_max:.6g}]")
    print(f"汇总：{output_dir / 'summary.json'}")
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}")


if __name__ == "__main__":
    main()
