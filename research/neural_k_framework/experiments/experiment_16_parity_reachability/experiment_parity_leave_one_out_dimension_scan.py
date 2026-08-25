"""
Parity leave-one-out 位数扫描。

对 n-bit -> 1-bit parity，训练完整真值表减去一个输入状态，并扫描
n = 4, 6, 8, 10, 12, 14, 16。实验把三个现象严格分开：

1. 完整真值表对照也无法拟合：优化/可达性障碍；
2. 留一训练集可以 hard-exact，但留出点预测错误：函数补全偏置排斥 parity；
3. 留一训练集 hard-exact 且留出点正确：冗余约束使完整 parity 胜出。

每个位数预注册选择 8 个留出状态，奇偶标签平衡，并覆盖不同 Hamming
weight。每个留出条件和完整真值表对照共享配对初始化。

AutoDL 用法：修改 Config 后，将整个文件复制到 notebook 单元运行。
"""

from __future__ import annotations

import csv
import json
import math
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
    BIT_LENGTHS = (4, 6, 8, 10, 12, 14, 16)
    HOLDOUTS_PER_BITS = 8
    SEEDS_PER_CONDITION = 8

    WIDTH = 64
    HIDDEN_LAYERS = 3
    ACTIVATION = "gelu"
    USE_LAYER_NORM = True
    LAYERNORM_EPS = 1e-5

    OPTIMIZER = "adamw"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 30_000
    TRAIN_BATCH_SIZE = 512
    EVAL_CHUNK_SIZE = 1_024
    EVAL_INTERVAL = 1_000
    LOG_INTERVAL = 1_000

    RAW_BCE_THRESHOLDS = (
        0.3,
        0.1,
        0.03,
        0.01,
        0.003,
        0.001,
        3e-4,
        1e-4,
        3e-5,
        1e-5,
        3e-6,
        1e-6,
        1e-7,
        1e-8,
    )

    HOLDOUT_SEED = 20260821
    INITIALIZATION_SEED = 20260822
    BATCH_SEED = 20260823
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_parity_leave_one_out_dimension_scan")
    PACKAGE_RESULTS = True
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass(frozen=True)
class Condition:
    index: int
    name: str
    heldout_index: int | None
    is_full_control: bool
    seeds: int


@dataclass
class Evaluation:
    train_loss: torch.Tensor
    train_exact: torch.Tensor
    heldout_correct: torch.Tensor
    heldout_margin: torch.Tensor
    full_parity_exact: torch.Tensor


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.BIT_LENGTHS = (4, 6)
    Config.HOLDOUTS_PER_BITS = 4
    Config.SEEDS_PER_CONDITION = 2
    Config.WIDTH = 16
    Config.MAX_STEPS = 30
    Config.TRAIN_BATCH_SIZE = 64
    Config.EVAL_CHUNK_SIZE = 64
    Config.EVAL_INTERVAL = 10
    Config.LOG_INTERVAL = 10
    Config.RAW_BCE_THRESHOLDS = (0.7, 0.5, 0.3)
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_parity_leave_one_out_dimension_scan"
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
    bit_lengths = tuple(int(value) for value in Config.BIT_LENGTHS)
    if not bit_lengths or any(value < 2 for value in bit_lengths):
        raise ValueError("BIT_LENGTHS 必须包含至少一个不小于2的整数。")
    if tuple(sorted(set(bit_lengths))) != bit_lengths:
        raise ValueError("BIT_LENGTHS 必须严格递增且不重复。")
    if Config.HOLDOUTS_PER_BITS < 2 or Config.HOLDOUTS_PER_BITS % 2:
        raise ValueError("HOLDOUTS_PER_BITS 必须是至少为2的偶数。")
    if Config.SEEDS_PER_CONDITION < 1:
        raise ValueError("SEEDS_PER_CONDITION 必须为正。")
    if Config.WIDTH < 1 or Config.HIDDEN_LAYERS < 1:
        raise ValueError("网络宽度和隐藏层数必须为正。")
    if Config.ACTIVATION not in {"gelu", "tanh", "relu"}:
        raise ValueError("ACTIVATION 只支持 gelu/tanh/relu。")
    if Config.OPTIMIZER not in {"adamw", "sgd"}:
        raise ValueError("OPTIMIZER 只支持 adamw/sgd。")
    thresholds = tuple(float(value) for value in Config.RAW_BCE_THRESHOLDS)
    if any(value <= 0 for value in thresholds):
        raise ValueError("RAW_BCE_THRESHOLDS 必须为正。")
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("RAW_BCE_THRESHOLDS 必须严格从高到低且不重复。")


def truth_table_inputs(bits: int) -> np.ndarray:
    values = np.arange(1 << bits, dtype=np.uint64)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def parity_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs.sum(axis=1) % 2).astype(np.uint8)


def choose_evenly_spread(
    candidates: Sequence[int],
    inputs: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    # 先按 Hamming weight 分层，再用很小的随机 tie-break 避免只选字典序端点。
    ranked = sorted(
        (int(index) for index in candidates),
        key=lambda index: (
            int(inputs[index].sum()),
            float(rng.random()),
        ),
    )
    if count >= len(ranked):
        return ranked
    positions = np.linspace(0, len(ranked) - 1, count)
    selected = []
    used: set[int] = set()
    for position in positions:
        cursor = int(round(float(position)))
        while cursor in used and cursor + 1 < len(ranked):
            cursor += 1
        while cursor in used and cursor - 1 >= 0:
            cursor -= 1
        used.add(cursor)
        selected.append(ranked[cursor])
    return selected


def select_holdouts(
    bits: int,
    inputs: np.ndarray,
    labels: np.ndarray,
) -> list[int]:
    per_label = Config.HOLDOUTS_PER_BITS // 2
    zeros = np.flatnonzero(labels == 0).tolist()
    ones = np.flatnonzero(labels == 1).tolist()
    if per_label > len(zeros) or per_label > len(ones):
        raise ValueError(f"bits={bits} 无法选择足够多的平衡留出点。")
    rng = np.random.default_rng(Config.HOLDOUT_SEED + bits * 1009)
    selected = (
        choose_evenly_spread(zeros, inputs, per_label, rng)
        + choose_evenly_spread(ones, inputs, per_label, rng)
    )
    return sorted(selected)


def build_conditions(holdouts: Sequence[int]) -> list[Condition]:
    conditions = [
        Condition(
            index=index,
            name=f"leaveout_{heldout:0{max(2, len(str(max(holdouts))))}d}",
            heldout_index=int(heldout),
            is_full_control=False,
            seeds=Config.SEEDS_PER_CONDITION,
        )
        for index, heldout in enumerate(holdouts)
    ]
    conditions.append(Condition(
        index=len(conditions),
        name="full_truth_table_control",
        heldout_index=None,
        is_full_control=True,
        seeds=Config.SEEDS_PER_CONDITION,
    ))
    return conditions


def build_model_layout(
    conditions: Sequence[Condition],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition_indices: list[int] = []
    seed_indices: list[int] = []
    heldout_indices: list[int] = []
    for condition in conditions:
        condition_indices.extend([condition.index] * condition.seeds)
        seed_indices.extend(range(condition.seeds))
        heldout = -1 if condition.heldout_index is None else condition.heldout_index
        heldout_indices.extend([heldout] * condition.seeds)
    return (
        np.asarray(condition_indices, dtype=np.int64),
        np.asarray(seed_indices, dtype=np.int64),
        np.asarray(heldout_indices, dtype=np.int64),
    )


class BatchedIndependentMLP(nn.Module):
    def __init__(
        self,
        input_bits: int,
        model_seed_indices: np.ndarray,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.model_count = len(model_seed_indices)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()

        seed_indices = torch.from_numpy(model_seed_indices).to(device)
        base_seed_count = int(model_seed_indices.max()) + 1
        generator = torch.Generator(device=device)
        generator.manual_seed(Config.INITIALIZATION_SEED + input_bits * 7919)
        dimensions = (
            [input_bits]
            + [Config.WIDTH] * Config.HIDDEN_LAYERS
            + [1]
        )
        for layer_index, (input_size, output_size) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            bound = 1.0 / math.sqrt(input_size)
            base_weight = torch.empty(
                base_seed_count,
                output_size,
                input_size,
                device=device,
            ).uniform_(-bound, bound, generator=generator)
            base_bias = torch.empty(
                base_seed_count,
                output_size,
                device=device,
            ).uniform_(-bound, bound, generator=generator)
            self.weights.append(nn.Parameter(base_weight[seed_indices].clone()))
            self.biases.append(nn.Parameter(base_bias[seed_indices].clone()))
            if layer_index < Config.HIDDEN_LAYERS:
                self.norm_weights.append(nn.Parameter(torch.ones(
                    self.model_count, output_size, device=device
                )))
                self.norm_biases.append(nn.Parameter(torch.zeros(
                    self.model_count, output_size, device=device
                )))

    def activate(self, value: torch.Tensor) -> torch.Tensor:
        if Config.ACTIVATION == "gelu":
            return F.gelu(value)
        if Config.ACTIVATION == "tanh":
            return torch.tanh(value)
        return F.relu(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            hidden = inputs[None].expand(self.model_count, -1, -1)
        elif inputs.ndim == 3 and inputs.shape[0] == self.model_count:
            hidden = inputs
        else:
            raise ValueError(
                "inputs 必须是 [states,bits] 或 [models,states,bits]。"
            )
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if layer_index < Config.HIDDEN_LAYERS:
                hidden = self.activate(hidden)
                if Config.USE_LAYER_NORM:
                    mean = hidden.mean(dim=-1, keepdim=True)
                    variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
                    hidden = (hidden - mean) * torch.rsqrt(
                        variance + Config.LAYERNORM_EPS
                    )
                    hidden = (
                        hidden * self.norm_weights[layer_index][:, None, :]
                        + self.norm_biases[layer_index][:, None, :]
                    )
        return hidden.squeeze(-1)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if Config.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )


def training_batch_loss(
    model: BatchedIndependentMLP,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    batch_indices: torch.Tensor,
    model_heldouts: torch.Tensor,
) -> torch.Tensor:
    inputs = all_inputs[batch_indices]
    targets = all_targets[batch_indices]
    logits = model(inputs)
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets[None].expand_as(logits),
        reduction="none",
    )
    valid_holdout = model_heldouts >= 0
    mask = ~(
        valid_holdout[:, None]
        & (batch_indices[None, :] == model_heldouts[:, None])
    )
    denominator = mask.sum(dim=1).clamp_min(1)
    return (elementwise * mask).sum(dim=1) / denominator


@torch.no_grad()
def evaluate_models(
    model: BatchedIndependentMLP,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    model_heldouts: torch.Tensor,
) -> Evaluation:
    model_count = model.model_count
    device = all_inputs.device
    loss_sum = torch.zeros(model_count, device=device)
    train_count = torch.zeros(model_count, device=device)
    train_errors = torch.zeros(model_count, dtype=torch.int64, device=device)
    state_count = len(all_inputs)

    for start in range(0, state_count, Config.EVAL_CHUNK_SIZE):
        end = min(start + Config.EVAL_CHUNK_SIZE, state_count)
        inputs = all_inputs[start:end]
        targets = all_targets[start:end]
        logits = model(inputs)
        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            targets[None].expand_as(logits),
            reduction="none",
        )
        state_indices = torch.arange(start, end, device=device)
        valid_holdout = model_heldouts >= 0
        mask = ~(
            valid_holdout[:, None]
            & (state_indices[None, :] == model_heldouts[:, None])
        )
        loss_sum += (elementwise * mask).sum(dim=1)
        train_count += mask.sum(dim=1)
        predictions = logits >= 0
        errors = predictions != targets.bool()[None]
        train_errors += (errors & mask).sum(dim=1)

    train_loss = loss_sum / train_count.clamp_min(1)
    train_exact = train_errors == 0

    valid_holdout = model_heldouts >= 0
    safe_heldouts = model_heldouts.clamp_min(0)
    heldout_inputs = all_inputs[safe_heldouts][:, None, :]
    heldout_logits = model(heldout_inputs).squeeze(1)
    heldout_targets = all_targets[safe_heldouts]
    heldout_predictions = heldout_logits >= 0
    heldout_correct = heldout_predictions == heldout_targets.bool()
    heldout_correct = torch.where(
        valid_holdout,
        heldout_correct,
        torch.ones_like(heldout_correct),
    )
    signed_targets = heldout_targets * 2.0 - 1.0
    heldout_margin = heldout_logits * signed_targets
    heldout_margin = torch.where(
        valid_holdout,
        heldout_margin,
        torch.full_like(heldout_margin, float("nan")),
    )
    full_parity_exact = train_exact & heldout_correct
    return Evaluation(
        train_loss=train_loss,
        train_exact=train_exact,
        heldout_correct=heldout_correct,
        heldout_margin=heldout_margin,
        full_parity_exact=full_parity_exact,
    )


def safe_fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def aggregate_evaluation_rows(
    bits: int,
    step: int,
    evaluation: Evaluation,
    model_heldouts: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, mask in (
        ("leave_one_out", model_heldouts >= 0),
        ("full_control", model_heldouts < 0),
    ):
        local_loss = evaluation.train_loss[mask]
        local_exact = evaluation.train_exact[mask]
        local_full = evaluation.full_parity_exact[mask]
        exact_count = int(local_exact.sum().item())
        full_count = int(local_full.sum().item())
        row: dict[str, Any] = {
            "bits": bits,
            "step": step,
            "group": group_name,
            "model_count": int(mask.sum().item()),
            "train_loss_mean": float(local_loss.mean().item()),
            "train_loss_median": float(local_loss.median().item()),
            "train_loss_min": float(local_loss.min().item()),
            "train_loss_max": float(local_loss.max().item()),
            "train_exact_fraction": float(local_exact.float().mean().item()),
            "full_parity_fraction": float(local_full.float().mean().item()),
            "full_parity_given_train_exact": safe_fraction(
                full_count, exact_count
            ),
        }
        if group_name == "leave_one_out":
            local_heldout = evaluation.heldout_correct[mask]
            exact_and_heldout = int((local_exact & local_heldout).sum().item())
            local_margin = evaluation.heldout_margin[mask]
            row.update({
                "heldout_correct_fraction": float(
                    local_heldout.float().mean().item()
                ),
                "heldout_correct_given_train_exact": safe_fraction(
                    exact_and_heldout, exact_count
                ),
                "heldout_margin_mean": float(local_margin.mean().item()),
                "heldout_margin_median": float(local_margin.median().item()),
            })
        else:
            row.update({
                "heldout_correct_fraction": None,
                "heldout_correct_given_train_exact": None,
                "heldout_margin_mean": None,
                "heldout_margin_median": None,
            })
        rows.append(row)
    return rows


@torch.no_grad()
def update_crossings(
    step: int,
    evaluation: Evaluation,
    thresholds: torch.Tensor,
    crossed: torch.Tensor,
    crossing_step: torch.Tensor,
    crossing_heldout_correct: torch.Tensor,
) -> int:
    # 只有训练集已经 hard-exact 时，才把“最后一个点如何补全”作为候选函数
    # 之间的有效比较。
    new = (
        (evaluation.train_loss[:, None] <= thresholds[None])
        & evaluation.train_exact[:, None]
        & ~crossed
    )
    count = int(new.sum().item())
    if count == 0:
        return 0
    crossed[new] = True
    crossing_step[new] = int(step)
    crossing_heldout_correct[new] = (
        evaluation.heldout_correct[:, None].expand_as(new)[new]
    )
    return count


def summarize_crossings(
    bits: int,
    thresholds: Sequence[float],
    model_condition_indices: np.ndarray,
    conditions: Sequence[Condition],
    crossing_step: np.ndarray,
    crossing_heldout_correct: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    is_control = np.asarray([
        conditions[index].is_full_control for index in model_condition_indices
    ])
    aggregate_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for group_name, group_mask in (
        ("leave_one_out", ~is_control),
        ("full_control", is_control),
    ):
        for threshold_index, threshold in enumerate(thresholds):
            reached = group_mask & (crossing_step[:, threshold_index] >= 0)
            reached_count = int(reached.sum())
            correct_count = int(np.sum(
                reached & crossing_heldout_correct[:, threshold_index]
            ))
            aggregate_rows.append({
                "bits": bits,
                "group": group_name,
                "raw_bce_threshold": threshold,
                "eligible_model_count": int(group_mask.sum()),
                "reached_count": reached_count,
                "reached_fraction": safe_fraction(
                    reached_count, int(group_mask.sum())
                ),
                "median_crossing_step": (
                    float(np.median(crossing_step[reached, threshold_index]))
                    if reached_count else None
                ),
                "heldout_correct_count": (
                    correct_count if group_name == "leave_one_out" else None
                ),
                "heldout_correct_fraction": (
                    safe_fraction(correct_count, reached_count)
                    if group_name == "leave_one_out" else None
                ),
            })

    for condition in conditions:
        mask = model_condition_indices == condition.index
        for threshold_index, threshold in enumerate(thresholds):
            reached = mask & (crossing_step[:, threshold_index] >= 0)
            reached_count = int(reached.sum())
            correct_count = int(np.sum(
                reached & crossing_heldout_correct[:, threshold_index]
            ))
            condition_rows.append({
                "bits": bits,
                "condition": condition.name,
                "heldout_index": condition.heldout_index,
                "is_full_control": condition.is_full_control,
                "raw_bce_threshold": threshold,
                "eligible_model_count": int(mask.sum()),
                "reached_count": reached_count,
                "heldout_correct_count": (
                    correct_count if not condition.is_full_control else None
                ),
                "heldout_correct_fraction": (
                    safe_fraction(correct_count, reached_count)
                    if not condition.is_full_control else None
                ),
            })
    return aggregate_rows, condition_rows


def final_condition_rows(
    bits: int,
    inputs: np.ndarray,
    labels: np.ndarray,
    conditions: Sequence[Condition],
    model_condition_indices: np.ndarray,
    evaluation: Evaluation,
) -> list[dict[str, Any]]:
    losses = evaluation.train_loss.cpu().numpy()
    train_exact = evaluation.train_exact.cpu().numpy()
    heldout_correct = evaluation.heldout_correct.cpu().numpy()
    full_exact = evaluation.full_parity_exact.cpu().numpy()
    margins = evaluation.heldout_margin.cpu().numpy()
    rows: list[dict[str, Any]] = []
    sensitivity_gap = bits / (2 ** (bits - 1))
    for condition in conditions:
        mask = model_condition_indices == condition.index
        exact_count = int(train_exact[mask].sum())
        full_count = int(full_exact[mask].sum())
        row: dict[str, Any] = {
            "bits": bits,
            "condition": condition.name,
            "heldout_index": condition.heldout_index,
            "is_full_control": condition.is_full_control,
            "model_count": int(mask.sum()),
            "train_loss_mean": float(losses[mask].mean()),
            "train_loss_median": float(np.median(losses[mask])),
            "train_exact_fraction": float(train_exact[mask].mean()),
            "full_parity_fraction": float(full_exact[mask].mean()),
            "full_parity_given_train_exact": safe_fraction(
                full_count, exact_count
            ),
            "parity_average_sensitivity": float(bits),
            "single_flip_average_sensitivity": float(
                bits - sensitivity_gap
            ),
            "sensitivity_gap": float(sensitivity_gap),
        }
        if condition.heldout_index is not None:
            heldout = int(condition.heldout_index)
            exact_and_correct = int(
                (train_exact[mask] & heldout_correct[mask]).sum()
            )
            row.update({
                "heldout_bits": "".join(
                    str(int(value)) for value in inputs[heldout]
                ),
                "heldout_hamming_weight": int(inputs[heldout].sum()),
                "heldout_target": int(labels[heldout]),
                "heldout_correct_fraction": float(
                    heldout_correct[mask].mean()
                ),
                "heldout_correct_given_train_exact": safe_fraction(
                    exact_and_correct, exact_count
                ),
                "heldout_margin_mean": float(margins[mask].mean()),
                "heldout_margin_median": float(np.median(margins[mask])),
            })
        else:
            row.update({
                "heldout_bits": None,
                "heldout_hamming_weight": None,
                "heldout_target": None,
                "heldout_correct_fraction": None,
                "heldout_correct_given_train_exact": None,
                "heldout_margin_mean": None,
                "heldout_margin_median": None,
            })
        rows.append(row)
    return rows


def evaluation_steps() -> set[int]:
    return {
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        *range(Config.EVAL_INTERVAL, Config.MAX_STEPS + 1, Config.EVAL_INTERVAL),
        Config.MAX_STEPS,
    }


def train_one_bit_length(
    bits: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    bit_dir = output_dir / f"bits_{bits:02d}"
    bit_dir.mkdir(parents=True, exist_ok=True)
    inputs_np = truth_table_inputs(bits)
    labels_np = parity_targets(inputs_np)
    holdouts = select_holdouts(bits, inputs_np, labels_np)
    conditions = build_conditions(holdouts)
    (
        model_condition_indices,
        model_seed_indices,
        model_heldouts_np,
    ) = build_model_layout(conditions)

    all_inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    all_targets = torch.from_numpy(labels_np.astype(np.float32)).to(device)
    model_heldouts = torch.from_numpy(model_heldouts_np).to(device)
    model = BatchedIndependentMLP(bits, model_seed_indices, device)
    optimizer = make_optimizer(model)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(Config.BATCH_SEED + bits * 104729)

    thresholds = torch.tensor(
        Config.RAW_BCE_THRESHOLDS,
        dtype=torch.float32,
        device=device,
    )
    crossing_shape = (model.model_count, len(Config.RAW_BCE_THRESHOLDS))
    crossed = torch.zeros(crossing_shape, dtype=torch.bool, device=device)
    crossing_step = torch.full(
        crossing_shape, -1, dtype=torch.int32, device=device
    )
    crossing_heldout_correct = torch.zeros(
        crossing_shape, dtype=torch.bool, device=device
    )

    write_json(bit_dir / "conditions.json", [asdict(item) for item in conditions])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameters_per_model = parameter_count // model.model_count
    state_count = len(inputs_np)
    train_batch_size = min(Config.TRAIN_BATCH_SIZE, state_count)
    print(
        f"\n=== bits={bits} | states={state_count:,} | "
        f"models={model.model_count} | params/model={parameters_per_model:,} | "
        f"batch={train_batch_size} ===",
        flush=True,
    )
    print(
        "holdouts="
        + str([
            {
                "state": index,
                "weight": int(inputs_np[index].sum()),
                "target": int(labels_np[index]),
            }
            for index in holdouts
        ]),
        flush=True,
    )

    trajectory_rows: list[dict[str, Any]] = []
    eval_steps = evaluation_steps()
    started = time.perf_counter()
    interrupted = False
    final_step = 0

    try:
        for step in range(Config.MAX_STEPS + 1):
            if step in eval_steps:
                evaluation = evaluate_models(
                    model, all_inputs, all_targets, model_heldouts
                )
                update_crossings(
                    step,
                    evaluation,
                    thresholds,
                    crossed,
                    crossing_step,
                    crossing_heldout_correct,
                )
                current = aggregate_evaluation_rows(
                    bits, step, evaluation, model_heldouts
                )
                trajectory_rows.extend(current)
                leaveout = next(
                    row for row in current if row["group"] == "leave_one_out"
                )
                control = next(
                    row for row in current if row["group"] == "full_control"
                )
                if step <= 500 or step % Config.LOG_INTERVAL == 0:
                    parity_given_exact = leaveout[
                        "heldout_correct_given_train_exact"
                    ]
                    print(
                        f"bits={bits:>2} step={step:>7,} | "
                        f"leaveout loss={leaveout['train_loss_median']:.2e} "
                        f"train-exact={leaveout['train_exact_fraction']:.1%} "
                        f"last|exact="
                        + (
                            "NA"
                            if parity_given_exact is None
                            else f"{parity_given_exact:.1%}"
                        )
                        + " | "
                        f"full-control exact={control['train_exact_fraction']:.1%} "
                        f"loss={control['train_loss_median']:.2e} | "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                write_csv(bit_dir / "trajectory.csv", trajectory_rows)

            final_step = step
            if step == Config.MAX_STEPS:
                break

            if train_batch_size == state_count:
                batch_indices = torch.arange(state_count, device=device)
            else:
                batch_indices = torch.randint(
                    0,
                    state_count,
                    (train_batch_size,),
                    generator=batch_generator,
                    device="cpu",
                ).to(device)
            losses = training_batch_loss(
                model,
                all_inputs,
                all_targets,
                batch_indices,
                model_heldouts,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            optimizer.step()
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\n收到中断，正在保存 bits={bits} 的当前结果……",
            flush=True,
        )

    evaluation = evaluate_models(model, all_inputs, all_targets, model_heldouts)
    final_current = aggregate_evaluation_rows(
        bits, final_step, evaluation, model_heldouts
    )
    if not any(row["step"] == final_step for row in trajectory_rows):
        trajectory_rows.extend(final_current)

    crossing_step_np = crossing_step.cpu().numpy()
    crossing_correct_np = crossing_heldout_correct.cpu().numpy()
    aggregate_crossing_rows, condition_crossing_rows = summarize_crossings(
        bits=bits,
        thresholds=Config.RAW_BCE_THRESHOLDS,
        model_condition_indices=model_condition_indices,
        conditions=conditions,
        crossing_step=crossing_step_np,
        crossing_heldout_correct=crossing_correct_np,
    )
    condition_final_rows = final_condition_rows(
        bits=bits,
        inputs=inputs_np,
        labels=labels_np,
        conditions=conditions,
        model_condition_indices=model_condition_indices,
        evaluation=evaluation,
    )
    write_csv(bit_dir / "trajectory.csv", trajectory_rows)
    write_csv(bit_dir / "loss_crossing_summary.csv", aggregate_crossing_rows)
    write_csv(
        bit_dir / "condition_loss_crossing_summary.csv",
        condition_crossing_rows,
    )
    write_csv(bit_dir / "final_condition_summary.csv", condition_final_rows)
    np.savez_compressed(
        bit_dir / "crossing_state.npz",
        model_condition_indices=model_condition_indices,
        model_seed_indices=model_seed_indices,
        model_heldouts=model_heldouts_np,
        thresholds=np.asarray(Config.RAW_BCE_THRESHOLDS, dtype=np.float64),
        crossing_step=crossing_step_np,
        crossing_heldout_correct=crossing_correct_np,
    )

    leaveout_final = next(
        row for row in final_current if row["group"] == "leave_one_out"
    )
    control_final = next(
        row for row in final_current if row["group"] == "full_control"
    )
    if control_final["train_exact_fraction"] < 0.75:
        verdict = "optimization_obstruction_dominant"
    elif leaveout_final["train_exact_fraction"] < 0.75:
        verdict = "leaveout_reachability_mixed"
    elif (
        leaveout_final["heldout_correct_given_train_exact"] is not None
        and leaveout_final["heldout_correct_given_train_exact"] >= 0.75
    ):
        verdict = "parity_completion_preferred"
    elif (
        leaveout_final["heldout_correct_given_train_exact"] is not None
        and leaveout_final["heldout_correct_given_train_exact"] <= 0.25
    ):
        verdict = "single_flip_completion_preferred"
    else:
        verdict = "mixed_completion_distribution"

    summary = {
        "status": "interrupted" if interrupted else "completed",
        "bits": bits,
        "state_count": state_count,
        "train_count_leave_one_out": state_count - 1,
        "final_step": final_step,
        "elapsed_seconds": time.perf_counter() - started,
        "holdout_indices": holdouts,
        "model_count": model.model_count,
        "parameters_per_model": parameters_per_model,
        "parity_average_sensitivity": float(bits),
        "single_flip_average_sensitivity": float(
            bits - bits / (2 ** (bits - 1))
        ),
        "sensitivity_gap": float(bits / (2 ** (bits - 1))),
        "leaveout_final": leaveout_final,
        "full_control_final": control_final,
        "verdict": verdict,
    }
    write_json(bit_dir / "summary.json", summary)
    print(
        f"bits={bits} 完成 | verdict={verdict} | "
        f"last|exact={leaveout_final['heldout_correct_given_train_exact']} | "
        f"control-exact={control_final['train_exact_fraction']:.1%}",
        flush=True,
    )
    return summary, interrupted


def plot_overall(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", flush=True)
        return

    ordered = sorted(summaries, key=lambda row: int(row["bits"]))
    bits = [int(row["bits"]) for row in ordered]
    leaveout_train_exact = [
        float(row["leaveout_final"]["train_exact_fraction"])
        for row in ordered
    ]
    control_train_exact = [
        float(row["full_control_final"]["train_exact_fraction"])
        for row in ordered
    ]
    heldout = [
        (
            float(row["leaveout_final"]["heldout_correct_given_train_exact"])
            if row["leaveout_final"]["heldout_correct_given_train_exact"]
            is not None
            else np.nan
        )
        for row in ordered
    ]
    sensitivity_gap = [float(row["sensitivity_gap"]) for row in ordered]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(bits, leaveout_train_exact, marker="o", label="leave-one-out")
    axes[0].plot(bits, control_train_exact, marker="o", label="full control")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("input bits")
    axes[0].set_ylabel("fraction train hard-exact")
    axes[0].set_title("Reachability")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(bits, heldout, marker="o", color="tab:red")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("input bits")
    axes[1].set_ylabel("P(last point correct | train hard-exact)")
    axes[1].set_title("Parity completion preference")
    axes[1].grid(alpha=0.25)
    twin = axes[1].twinx()
    twin.plot(
        bits,
        sensitivity_gap,
        marker="x",
        color="tab:blue",
        alpha=0.55,
        label="sensitivity gap",
    )
    twin.set_yscale("log")
    twin.set_ylabel("sensitivity advantage of one-point flip")

    figure.savefig(output_dir / "parity_leave_one_out_dimension_scan.png", dpi=180)
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

    print("=== Parity leave-one-out dimension scan ===", flush=True)
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"bits={list(Config.BIT_LENGTHS)} | holdouts/bits="
        f"{Config.HOLDOUTS_PER_BITS} | seeds/condition="
        f"{Config.SEEDS_PER_CONDITION}",
        flush=True,
    )
    print(
        f"MLP: width={Config.WIDTH}, layers={Config.HIDDEN_LAYERS}, "
        f"activation={Config.ACTIVATION}, layernorm={Config.USE_LAYER_NORM} | "
        f"max_steps={Config.MAX_STEPS:,}",
        flush=True,
    )
    print(f"结果目录：{output_dir.resolve()}", flush=True)

    summaries: list[dict[str, Any]] = []
    interrupted = False
    overall_started = time.perf_counter()
    for bits in Config.BIT_LENGTHS:
        summary, bit_interrupted = train_one_bit_length(
            int(bits), output_dir, device
        )
        summaries.append(summary)
        write_csv(output_dir / "dimension_summary.csv", [
            {
                "bits": row["bits"],
                "state_count": row["state_count"],
                "final_step": row["final_step"],
                "status": row["status"],
                "verdict": row["verdict"],
                "sensitivity_gap": row["sensitivity_gap"],
                "leaveout_train_loss_median": row["leaveout_final"][
                    "train_loss_median"
                ],
                "leaveout_train_exact_fraction": row["leaveout_final"][
                    "train_exact_fraction"
                ],
                "heldout_correct_given_train_exact": row["leaveout_final"][
                    "heldout_correct_given_train_exact"
                ],
                "full_control_train_loss_median": row["full_control_final"][
                    "train_loss_median"
                ],
                "full_control_train_exact_fraction": row["full_control_final"][
                    "train_exact_fraction"
                ],
            }
            for row in summaries
        ])
        if bit_interrupted:
            interrupted = True
            break

    plot_overall(output_dir, summaries)
    overall = {
        "status": "interrupted" if interrupted else "completed",
        "elapsed_seconds": time.perf_counter() - overall_started,
        "bit_summaries": summaries,
        "interpretation": {
            "control_fails": (
                "完整真值表对照也无法 hard-exact，当前位数首先受到优化可达性限制。"
            ),
            "control_succeeds_leaveout_wrong": (
                "网络可学会被强制指定的 parity，但在少一个标签时偏向单点翻转补全。"
            ),
            "control_succeeds_leaveout_correct": (
                "冗余约束已经使完整 parity 成为 SGD 诱导分布中的优势补全。"
            ),
        },
    }
    write_json(output_dir / "summary.json", overall)

    archive_path: Path | None = None
    if Config.PACKAGE_RESULTS:
        archive_path = create_archive(output_dir)
    print("\n=== 扫描结束 ===", flush=True)
    print(json.dumps(json_ready(overall), ensure_ascii=False, indent=2), flush=True)
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}", flush=True)


if __name__ == "__main__":
    main()
