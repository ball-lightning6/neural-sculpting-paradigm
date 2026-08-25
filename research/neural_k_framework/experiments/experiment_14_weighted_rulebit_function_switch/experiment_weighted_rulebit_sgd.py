"""加权 Rule110/Rule30 复合任务的 loss 层级函数换序实验。

输入为 ``(rule_bit, left, center, right)``，输出为一个 ECA 更新 bit。
训练集包含完整 16 行真值表，但 Rule110 与 Rule30 分支在训练目标中的权重
高度不平衡。脚本另算一个 1:1 分支平衡目标，它只是反事实诊断参考，
不是验证分布，也不参与“哪个函数正确”的裁决。

实验同时测量：

1. 完整 16-bit hard function 分布；
2. Rule110-only、Rule30-only、真实复合规则及交换复合规则的概率；
3. 两个分支、三个冲突 neighborhood 和五个共享 neighborhood 的 raw BCE；
4. 加权训练目标与反事实 1:1 参考目标的全参数梯度夹角；
5. loss 继续下降时，优势函数的绝对程序复杂度是否可以上升。

该脚本只研究真实 SGD 动力学，不把训练轨迹解释为静态 prior 条件采样。
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    RULE0 = 110
    RULE1 = 30
    # ratio 表示 Rule110:Rule30。1 是平衡控制。
    MAJORITY_RATIOS = (1, 10, 100, 1_000, 10_000)

    WIDTH = 64
    HIDDEN_LAYERS = 3
    SEED_COUNT = 512
    INITIALIZATION_SEED = 20260821

    OPTIMIZER = "sgd"
    LEARNING_RATE = 5e-2
    MOMENTUM = 0.0
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 200_000

    # 每个 seed 第一次跨过这些 raw-BCE 水平时，立即保存完整 hard function。
    # 这是“不同 loss 区域偏好不同函数”的主判决口径。
    MATCHED_LOSS_LEVELS = (
        0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05,
        0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001,
        0.0005, 0.0003, 0.0002, 0.0001, 0.00003, 0.00001,
    )

    EARLY_EVAL_STEPS = (
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        1_000, 2_000, 5_000, 10_000, 20_000,
    )
    EVAL_INTERVAL_STEPS = 1_000
    SAVE_INTERVAL_STEPS = 20_000
    SAVE_INTERVAL_SECONDS = 90.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_weighted_rulebit_sgd")
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.MAJORITY_RATIOS = (1, 10)
    Config.WIDTH = 16
    Config.HIDDEN_LAYERS = 2
    Config.SEED_COUNT = 8
    Config.MAX_STEPS = 5
    Config.MATCHED_LOSS_LEVELS = (0.75, 0.70, 0.65, 0.60)
    Config.EARLY_EVAL_STEPS = (0, 1, 2, 3, 4, 5)
    Config.EVAL_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_SECONDS = 0.01
    Config.RESULT_DIR = Path(
        "research/loss_level_function_switch/_smoke_weighted_rulebit_sgd"
    )
    Config.RESUME = False
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rule_output(rule: int, neighborhood: int) -> int:
    return (int(rule) >> int(neighborhood)) & 1


def build_truth_table(device: torch.device) -> dict[str, torch.Tensor]:
    neighborhoods = torch.arange(8, dtype=torch.long)
    local_bits = torch.stack(
        (
            (neighborhoods >> 2) & 1,
            (neighborhoods >> 1) & 1,
            neighborhoods & 1,
        ),
        dim=1,
    ).to(torch.float32)
    rule0_bits = torch.zeros((8, 1), dtype=torch.float32)
    rule1_bits = torch.ones((8, 1), dtype=torch.float32)
    x0 = torch.cat((rule0_bits, local_bits), dim=1)
    x1 = torch.cat((rule1_bits, local_bits), dim=1)
    y0 = torch.tensor(
        [rule_output(Config.RULE0, value) for value in range(8)],
        dtype=torch.float32,
    )
    y1 = torch.tensor(
        [rule_output(Config.RULE1, value) for value in range(8)],
        dtype=torch.float32,
    )
    conflict = y0 != y1
    shared = ~conflict
    return {
        "x0": x0.to(device),
        "x1": x1.to(device),
        "y0": y0.to(device),
        "y1": y1.to(device),
        "conflict": conflict.to(device),
        "shared": shared.to(device),
        "neighborhoods": neighborhoods.to(device),
    }


def candidate_function_ids() -> dict[str, int]:
    low_rule0 = int(Config.RULE0)
    low_rule1 = int(Config.RULE1)
    return {
        "rule110_both": low_rule0 | (low_rule0 << 8),
        "rule30_both": low_rule1 | (low_rule1 << 8),
        "true_composite": low_rule0 | (low_rule1 << 8),
        "swapped_composite": low_rule1 | (low_rule0 << 8),
    }


def binary_entropy(probability: float) -> float:
    p = float(probability)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


class BatchedIndependentMLP(nn.Module):
    def __init__(self, seed_count: int, condition_count: int) -> None:
        super().__init__()
        self.seed_count = int(seed_count)
        self.condition_count = int(condition_count)
        dimensions = [4] + [Config.WIDTH] * Config.HIDDEN_LAYERS + [1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.INITIALIZATION_SEED)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()

        for layer_index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            bound = 1.0 / math.sqrt(input_dim)
            base_weight = torch.empty(
                seed_count, output_dim, input_dim, dtype=torch.float32
            ).uniform_(-bound, bound, generator=generator)
            base_bias = torch.empty(
                seed_count, output_dim, dtype=torch.float32
            ).uniform_(-bound, bound, generator=generator)
            weight = base_weight.repeat(condition_count, 1, 1)
            bias = base_bias.repeat(condition_count, 1)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))
            if layer_index < len(dimensions) - 2:
                model_count = seed_count * condition_count
                self.norm_weights.append(nn.Parameter(torch.ones(model_count, output_dim)))
                self.norm_biases.append(nn.Parameter(torch.zeros(model_count, output_dim)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        norm_index = 0
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if layer_index < len(self.weights) - 1:
                hidden = F.gelu(hidden)
                mean = hidden.mean(dim=-1, keepdim=True)
                variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
                hidden = (hidden - mean) * torch.rsqrt(variance + 1e-5)
                hidden = (
                    hidden * self.norm_weights[norm_index][:, None, :]
                    + self.norm_biases[norm_index][:, None, :]
                )
                norm_index += 1
        return hidden.squeeze(-1)

    def parameter_blocks(self) -> list[str]:
        # nn.Module 按子 ParameterList 的注册顺序遍历参数。
        linear_names = [f"linear_{index}" for index in range(len(self.weights))]
        norm_names = [f"norm_{index}" for index in range(len(self.norm_weights))]
        return linear_names + linear_names + norm_names + norm_names


def repeat_inputs(table: dict[str, torch.Tensor], model_count: int) -> dict[str, torch.Tensor]:
    return {
        key: value[None].expand(model_count, *value.shape)
        for key, value in table.items()
        if key in {"x0", "x1"}
    }


def condition_epsilons(device: torch.device) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    conditions: list[dict[str, Any]] = []
    values: list[float] = []
    conflict_fraction = int((Config.RULE0 ^ Config.RULE1).bit_count()) / 8.0
    for ratio in Config.MAJORITY_RATIOS:
        epsilon = 1.0 / (float(ratio) + 1.0)
        conditions.append({
            "ratio": int(ratio),
            "epsilon": epsilon,
            "conflict_fraction": conflict_fraction,
            # 若 soft logits 也严格忽略 rule bit，则冲突行的最优预测是标签混合。
            "strict_rulebit_invariant_soft_loss_floor": (
                conflict_fraction * binary_entropy(epsilon)
            ),
            # 若只固定 hard ID=Rule110-both，minority 错误 logit 可从错误一侧
            # 趋近 0，每个错误输出的 BCE 下确界为 ln(2)。
            "hard_rule110_both_loss_infimum": (
                conflict_fraction * epsilon * math.log(2.0)
            ),
        })
        values.extend([epsilon] * Config.SEED_COUNT)
    return torch.tensor(values, dtype=torch.float32, device=device), conditions


def loss_parts(
    model: BatchedIndependentMLP,
    inputs: dict[str, torch.Tensor],
    table: dict[str, torch.Tensor],
    epsilons: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits0 = model(inputs["x0"])
    logits1 = model(inputs["x1"])
    targets0 = table["y0"][None].expand_as(logits0)
    targets1 = table["y1"][None].expand_as(logits1)
    per0 = F.binary_cross_entropy_with_logits(logits0, targets0, reduction="none")
    per1 = F.binary_cross_entropy_with_logits(logits1, targets1, reduction="none")
    loss0 = per0.mean(dim=1)
    loss1 = per1.mean(dim=1)
    conflict0 = per0[:, table["conflict"]].mean(dim=1)
    conflict1 = per1[:, table["conflict"]].mean(dim=1)
    shared0 = per0[:, table["shared"]].mean(dim=1)
    shared1 = per1[:, table["shared"]].mean(dim=1)
    train = (1.0 - epsilons) * loss0 + epsilons * loss1
    balanced = 0.5 * (loss0 + loss1)
    conflict_train = (1.0 - epsilons) * conflict0 + epsilons * conflict1
    conflict_balanced = 0.5 * (conflict0 + conflict1)
    result = {
        "logits0": logits0,
        "logits1": logits1,
        "loss0": loss0,
        "loss1": loss1,
        "train": train,
        "balanced": balanced,
        "conflict0": conflict0,
        "conflict1": conflict1,
        "shared0": shared0,
        "shared1": shared1,
        "conflict_train": conflict_train,
        "conflict_balanced": conflict_balanced,
    }
    for neighborhood in torch.nonzero(
        table["conflict"], as_tuple=False
    ).flatten().tolist():
        name = f"{int(neighborhood):03b}"
        result[f"conflict_{name}_rule110"] = per0[:, int(neighborhood)]
        result[f"conflict_{name}_rule30"] = per1[:, int(neighborhood)]
        result[f"conflict_{name}_train"] = (
            (1.0 - epsilons) * per0[:, int(neighborhood)]
            + epsilons * per1[:, int(neighborhood)]
        )
        result[f"conflict_{name}_balanced"] = 0.5 * (
            per0[:, int(neighborhood)] + per1[:, int(neighborhood)]
        )
    return result


def objective_gradients(
    objective: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.autograd.grad(
        objective.sum(), parameters, retain_graph=retain_graph, create_graph=False
    ))


def gradient_geometry(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    blocks: Sequence[str],
) -> dict[str, torch.Tensor]:
    model_count = first[0].shape[0]
    dot = torch.zeros(model_count, device=first[0].device)
    first_sq = torch.zeros_like(dot)
    second_sq = torch.zeros_like(dot)
    block_values: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for first_value, second_value, block in zip(first, second, blocks):
        axes = tuple(range(1, first_value.ndim))
        local_dot = (first_value * second_value).sum(dim=axes)
        local_first = first_value.square().sum(dim=axes)
        local_second = second_value.square().sum(dim=axes)
        dot += local_dot
        first_sq += local_first
        second_sq += local_second
        if block not in block_values:
            block_values[block] = (
                torch.zeros_like(dot), torch.zeros_like(dot), torch.zeros_like(dot)
            )
        bd, bf, bs = block_values[block]
        block_values[block] = (bd + local_dot, bf + local_first, bs + local_second)
    denominator = torch.sqrt(first_sq * second_sq).clamp_min(1e-30)
    result = {
        "dot": dot,
        "first_norm": torch.sqrt(first_sq),
        "second_norm": torch.sqrt(second_sq),
        "cosine": (dot / denominator).clamp(-1.0, 1.0),
    }
    for block, (bd, bf, bs) in block_values.items():
        result[f"cosine_{block}"] = (
            bd / torch.sqrt(bf * bs).clamp_min(1e-30)
        ).clamp(-1.0, 1.0)
    return result


def function_ids(logits0: torch.Tensor, logits1: torch.Tensor) -> torch.Tensor:
    predictions = torch.cat((logits0 >= 0, logits1 >= 0), dim=1).to(torch.int64)
    powers = (1 << torch.arange(16, device=predictions.device, dtype=torch.int64))[None]
    return (predictions * powers).sum(dim=1)


def initialize_crossing_state(
    raw_payload: dict[str, np.ndarray], model_count: int
) -> None:
    levels = np.asarray(Config.MATCHED_LOSS_LEVELS, dtype=np.float64)
    if "matched_loss_levels" in raw_payload:
        existing = np.asarray(raw_payload["matched_loss_levels"], dtype=np.float64)
        if not np.array_equal(existing, levels):
            raise RuntimeError("checkpoint 的 MATCHED_LOSS_LEVELS 与当前配置不一致。")
        expected = (model_count, len(levels))
        if tuple(raw_payload["crossing_steps"].shape) != expected:
            raise RuntimeError("checkpoint 的 crossing shape 与当前配置不一致。")
        return
    raw_payload["matched_loss_levels"] = levels
    raw_payload["crossing_steps"] = np.full(
        (model_count, len(levels)), -1, dtype=np.int64
    )
    raw_payload["crossing_function_ids"] = np.full(
        (model_count, len(levels)), 65_535, dtype=np.uint16
    )
    raw_payload["crossing_actual_losses"] = np.full(
        (model_count, len(levels)), np.nan, dtype=np.float32
    )


@torch.no_grad()
def record_matched_loss_crossings(
    parts: dict[str, torch.Tensor],
    step: int,
    raw_payload: dict[str, np.ndarray],
) -> int:
    losses = parts["train"].detach().cpu().numpy().astype(np.float64)
    ids = function_ids(parts["logits0"], parts["logits1"])
    ids_np = ids.detach().cpu().numpy().astype(np.uint16)
    levels = np.asarray(raw_payload["matched_loss_levels"], dtype=np.float64)
    crossing_steps = raw_payload["crossing_steps"]
    crossing_ids = raw_payload["crossing_function_ids"]
    crossing_losses = raw_payload["crossing_actual_losses"]
    new_count = 0
    for level_index, level in enumerate(levels):
        newly_crossed = (crossing_steps[:, level_index] < 0) & (losses <= level)
        if not np.any(newly_crossed):
            continue
        crossing_steps[newly_crossed, level_index] = int(step)
        crossing_ids[newly_crossed, level_index] = ids_np[newly_crossed]
        crossing_losses[newly_crossed, level_index] = losses[newly_crossed]
        new_count += int(np.sum(newly_crossed))
    return new_count


def build_matched_loss_rows(
    raw_payload: dict[str, np.ndarray],
    conditions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    levels = np.asarray(raw_payload["matched_loss_levels"], dtype=np.float64)
    crossing_steps = np.asarray(raw_payload["crossing_steps"], dtype=np.int64)
    crossing_ids = np.asarray(raw_payload["crossing_function_ids"], dtype=np.uint16)
    crossing_losses = np.asarray(
        raw_payload["crossing_actual_losses"], dtype=np.float64
    )
    candidates = candidate_function_ids()
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        start = condition_index * Config.SEED_COUNT
        end = start + Config.SEED_COUNT
        for level_index, level in enumerate(levels):
            local_steps = crossing_steps[start:end, level_index]
            valid = local_steps >= 0
            selected_count = int(np.sum(valid))
            local_ids = crossing_ids[start:end, level_index][valid]
            counts = np.bincount(local_ids.astype(np.int64), minlength=65_536)
            if selected_count:
                probabilities = counts[counts > 0] / selected_count
                entropy = float(-np.sum(probabilities * np.log2(probabilities)))
                median_step = float(np.median(local_steps[valid]))
                median_loss = float(np.median(
                    crossing_losses[start:end, level_index][valid]
                ))
            else:
                entropy = float("nan")
                median_step = float("nan")
                median_loss = float("nan")
            count_simple = int(counts[candidates["rule110_both"]])
            count_composite = int(counts[candidates["true_composite"]])
            odds = (count_composite + 0.5) / (count_simple + 0.5)
            row: dict[str, Any] = {
                "ratio": int(condition["ratio"]),
                "epsilon": float(condition["epsilon"]),
                "loss_level": float(level),
                "selected_count": selected_count,
                "crossed_fraction": selected_count / Config.SEED_COUNT,
                "median_crossing_step": median_step,
                "median_actual_loss": median_loss,
                "function_entropy_bits": entropy,
                "effective_function_count": (
                    float(2.0 ** entropy) if math.isfinite(entropy) else float("nan")
                ),
                "composite_over_rule110_odds_jeffreys": float(odds),
                "log_composite_over_rule110_odds": float(math.log(odds)),
            }
            for name, function_id in candidates.items():
                row[f"count_{name}"] = int(counts[function_id])
                row[f"p_{name}"] = (
                    int(counts[function_id]) / selected_count
                    if selected_count else float("nan")
                )
            rows.append(row)
    return rows


def branch_exact(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((logits >= 0) == (targets[None] >= 0.5)).all(dim=1).to(torch.float32)


def evaluate(
    model: BatchedIndependentMLP,
    inputs: dict[str, torch.Tensor],
    table: dict[str, torch.Tensor],
    epsilons: torch.Tensor,
    conditions: Sequence[dict[str, Any]],
    step: int,
    elapsed_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    parts = loss_parts(model, inputs, table, epsilons)
    parameters = tuple(model.parameters())
    train_grad = objective_gradients(parts["train"], parameters, True)
    balanced_grad = objective_gradients(parts["balanced"], parameters, True)
    rule0_grad = objective_gradients(parts["loss0"], parameters, True)
    rule1_grad = objective_gradients(parts["loss1"], parameters, True)
    conflict_train_grad = objective_gradients(parts["conflict_train"], parameters, True)
    conflict_balanced_grad = objective_gradients(
        parts["conflict_balanced"], parameters, False
    )
    blocks = model.parameter_blocks()
    train_balanced = gradient_geometry(train_grad, balanced_grad, blocks)
    train_rule0 = gradient_geometry(train_grad, rule0_grad, blocks)
    train_rule1 = gradient_geometry(train_grad, rule1_grad, blocks)
    conflict_geometry = gradient_geometry(
        conflict_train_grad, conflict_balanced_grad, blocks
    )

    ids = function_ids(parts["logits0"], parts["logits1"])
    candidates = candidate_function_ids()
    rule0_exact = branch_exact(parts["logits0"], table["y0"])
    rule1_exact = branch_exact(parts["logits1"], table["y1"])
    true_exact = rule0_exact * rule1_exact

    tensor_metrics = {
        **{key: value for key, value in parts.items() if not key.startswith("logits")},
        "rule0_exact": rule0_exact,
        "rule1_exact": rule1_exact,
        "true_exact": true_exact,
        "train_balanced_cosine": train_balanced["cosine"],
        "train_balanced_angle_deg": torch.rad2deg(
            torch.acos(train_balanced["cosine"].clamp(-1.0, 1.0))
        ),
        "train_rule0_cosine": train_rule0["cosine"],
        "train_rule1_cosine": train_rule1["cosine"],
        "conflict_train_balanced_cosine": conflict_geometry["cosine"],
        "train_gradient_norm": train_balanced["first_norm"],
        "balanced_gradient_norm": train_balanced["second_norm"],
    }
    for block in dict.fromkeys(blocks):
        tensor_metrics[f"train_balanced_cosine_{block}"] = train_balanced[
            f"cosine_{block}"
        ]

    numpy_metrics = {
        key: value.detach().cpu().numpy() for key, value in tensor_metrics.items()
    }
    ids_np = ids.detach().cpu().numpy().astype(np.uint16)
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    raw: dict[str, np.ndarray] = {}

    for condition_index, condition in enumerate(conditions):
        start = condition_index * Config.SEED_COUNT
        end = start + Config.SEED_COUNT
        local_ids = ids_np[start:end]
        counts = np.bincount(local_ids.astype(np.int64), minlength=65_536)
        probabilities = counts[counts > 0] / Config.SEED_COUNT
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        row: dict[str, Any] = {
            "step": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "ratio": int(condition["ratio"]),
            "epsilon": float(condition["epsilon"]),
            "strict_rulebit_invariant_soft_loss_floor": float(
                condition["strict_rulebit_invariant_soft_loss_floor"]
            ),
            "hard_rule110_both_loss_infimum": float(
                condition["hard_rule110_both_loss_infimum"]
            ),
            "function_entropy_bits": entropy,
            "effective_function_count": float(2.0 ** entropy),
            "unique_function_count": int(np.sum(counts > 0)),
        }
        for key, values in numpy_metrics.items():
            local = values[start:end]
            row[f"{key}_mean"] = float(np.mean(local))
            row[f"{key}_median"] = float(np.median(local))
            row[f"{key}_std"] = float(np.std(local))
        for name, function_id in candidates.items():
            row[f"p_{name}"] = int(counts[function_id]) / Config.SEED_COUNT
            row[f"count_{name}"] = int(counts[function_id])
        summary_rows.append(row)

        top_ids = np.argsort(counts)[::-1][:20]
        for rank, function_id in enumerate(top_ids, start=1):
            count = int(counts[function_id])
            if count == 0:
                break
            top_rows.append({
                "step": int(step),
                "ratio": int(condition["ratio"]),
                "rank": rank,
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "count": count,
                "probability": count / Config.SEED_COUNT,
                "candidate_name": next((
                    name for name, value in candidates.items()
                    if value == int(function_id)
                ), ""),
            })

        tag = f"ratio{int(condition['ratio'])}"
        raw[f"ids_{tag}_step{step}"] = local_ids
        for key, values in numpy_metrics.items():
            raw[f"{key}_{tag}_step{step}"] = values[start:end].astype(np.float32)

    model.train()
    return summary_rows, top_rows, raw


def print_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"step={int(row['step']):8,d} | ratio={int(row['ratio']):>5}:1 "
            f"| train={row['train_median']:.3e} "
            f"cf1:1={row['balanced_median']:.3e} "
            f"conflict={row['conflict_train_median']:.3e} "
            f"| 110both={row['p_rule110_both']:.2%} "
            f"composite={row['p_true_composite']:.2%} "
            f"| branch exact={row['rule0_exact_mean']:.1%}/"
            f"{row['rule1_exact_mean']:.1%} "
            f"| grad cos={row['train_balanced_cosine_mean']:+.3f} "
            f"angle={row['train_balanced_angle_deg_mean']:.1f}deg",
            flush=True,
        )


def prepare_output_dir() -> Path:
    output = Config.RESULT_DIR
    if output.exists():
        checkpoint = output / "latest_checkpoint.pt"
        if Config.RESUME and checkpoint.exists() and not Config.OVERWRITE_RESULT_DIR:
            return output
        if Config.OVERWRITE_RESULT_DIR:
            shutil.rmtree(output)
        else:
            output = output.parent / (
                output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_checkpoint(
    output_dir: Path,
    model: BatchedIndependentMLP,
    optimizer: torch.optim.Optimizer,
    step: int,
    elapsed_seconds: float,
    summary_rows: Sequence[dict[str, Any]],
    top_rows: Sequence[dict[str, Any]],
    raw_payload: dict[str, np.ndarray],
    conditions: Sequence[dict[str, Any]],
) -> None:
    temporary = output_dir / "latest_checkpoint.pt.tmp"
    target = output_dir / "latest_checkpoint.pt"
    torch.save({
        "step": int(step),
        "elapsed_seconds": float(elapsed_seconds),
        "ratios": tuple(Config.MAJORITY_RATIOS),
        "seed_count": int(Config.SEED_COUNT),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "summary_rows": list(summary_rows),
        "top_rows": list(top_rows),
        "raw_payload": raw_payload,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }, temporary)
    os.replace(temporary, target)
    write_csv(output_dir / "trajectory_aggregated.csv", summary_rows)
    write_csv(output_dir / "top_functions.csv", top_rows)
    write_csv(
        output_dir / "matched_loss_function_distribution.csv",
        build_matched_loss_rows(raw_payload, conditions),
    )
    np.savez_compressed(output_dir / "checkpoint_states.npz", **raw_payload)
    print(f"checkpoint saved | step={step:,}", flush=True)


def load_checkpoint(
    output_dir: Path,
    model: BatchedIndependentMLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    path = output_dir / "latest_checkpoint.pt"
    if not (Config.RESUME and path.exists()):
        return 0, 0.0, [], [], {}
    payload = torch.load(path, map_location=device, weights_only=False)
    if tuple(payload["ratios"]) != tuple(Config.MAJORITY_RATIOS):
        raise RuntimeError("checkpoint 的 MAJORITY_RATIOS 与当前配置不一致。")
    if int(payload["seed_count"]) != int(Config.SEED_COUNT):
        raise RuntimeError("checkpoint 的 SEED_COUNT 与当前配置不一致。")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.random.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
        # map_location=device 会把 RNG 状态也搬到 CUDA；PyTorch 的 RNG
        # 恢复接口只接受位于 CPU 的 ByteTensor。
        cuda_rng_states = [
            torch.as_tensor(state, dtype=torch.uint8, device="cpu")
            for state in payload["cuda_rng_state_all"]
        ]
        for device_index, state in enumerate(
            cuda_rng_states[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(state, device=device_index)
    print(f"resumed checkpoint | step={int(payload['step']):,}", flush=True)
    return (
        int(payload["step"]),
        float(payload["elapsed_seconds"]),
        list(payload["summary_rows"]),
        list(payload["top_rows"]),
        dict(payload["raw_payload"]),
    )


def make_plot(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(Config.MAJORITY_RATIOS)))
    for color, ratio in zip(colors, Config.MAJORITY_RATIOS):
        group = sorted(
            (row for row in rows if int(row["ratio"]) == int(ratio)),
            key=lambda row: int(row["step"]),
        )
        steps = np.asarray([row["step"] for row in group], dtype=np.float64)
        label = f"{ratio}:1"
        axes[0, 0].plot(
            steps, [row["p_rule110_both"] for row in group],
            color=color, linestyle="--", label=f"110-both {label}"
        )
        axes[0, 0].plot(
            steps, [row["p_true_composite"] for row in group],
            color=color, linestyle="-", label=f"composite {label}"
        )
        axes[0, 1].plot(
            steps, [row["train_median"] for row in group],
            color=color, linestyle="-", label=label
        )
        axes[0, 1].axhline(
            float(group[0]["hard_rule110_both_loss_infimum"]),
            color=color, linestyle=":", alpha=0.6,
        )
        axes[1, 0].plot(
            steps, [row["train_balanced_cosine_mean"] for row in group],
            color=color, label=label
        )
        axes[1, 1].plot(
            steps, [row["conflict0_median"] for row in group],
            color=color, linestyle="--", label=f"110 {label}"
        )
        axes[1, 1].plot(
            steps, [row["conflict1_median"] for row in group],
            color=color, linestyle="-", label=f"30 {label}"
        )
    axes[0, 0].set_title("Hard-function probability transport")
    axes[0, 0].set_ylabel("fraction of seeds")
    axes[0, 1].set_title("Weighted training raw BCE")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_title("Train vs counterfactual 1:1 gradient cosine")
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 1].set_title("Conflict-neighborhood branch BCE")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xlabel("SGD step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "weighted_rulebit_dynamics.png", dpi=180)
    plt.close(figure)


def make_matched_loss_plot(
    output_dir: Path, rows: Sequence[dict[str, Any]]
) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(Config.MAJORITY_RATIOS)))
    for color, ratio in zip(colors, Config.MAJORITY_RATIOS):
        group = [
            row for row in rows
            if int(row["ratio"]) == int(ratio)
            and int(row["selected_count"]) > 0
        ]
        group.sort(key=lambda row: float(row["loss_level"]), reverse=True)
        if not group:
            continue
        levels = np.asarray([row["loss_level"] for row in group], dtype=np.float64)
        label = f"{ratio}:1"
        axes[0, 0].plot(
            levels, [row["p_rule110_both"] for row in group],
            color=color, linestyle="--", marker="o", markersize=3,
            label=f"110-both {label}",
        )
        axes[0, 0].plot(
            levels, [row["p_true_composite"] for row in group],
            color=color, linestyle="-", marker="o", markersize=3,
            label=f"composite {label}",
        )
        axes[0, 1].plot(
            levels, [row["log_composite_over_rule110_odds"] for row in group],
            color=color, marker="o", markersize=3, label=label,
        )
        axes[1, 0].plot(
            levels, [row["median_crossing_step"] for row in group],
            color=color, marker="o", markersize=3, label=label,
        )
        axes[1, 1].plot(
            levels, [row["crossed_fraction"] for row in group],
            color=color, marker="o", markersize=3, label=label,
        )
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("matched weighted raw BCE (high -> low)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[0, 0].set_title("Function probability at first loss crossing")
    axes[0, 0].set_ylabel("fraction among crossed seeds")
    axes[0, 1].set_title("Composite / Rule110-only log odds")
    axes[0, 1].axhline(0.0, color="black", linewidth=1, linestyle=":")
    axes[1, 0].set_title("Median first-crossing step")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("Fraction of seeds reaching loss level")
    axes[1, 1].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "matched_loss_function_switch.png", dpi=180)
    plt.close(figure)


def create_archive(output_dir: Path) -> Path:
    archive = Path(str(output_dir) + "_package.zip")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in output_dir.rglob("*"):
            if path.is_file():
                handle.write(path, arcname=str(path.relative_to(output_dir)))
    return archive


def main() -> None:
    apply_smoke_overrides()
    device = torch.device(Config.DEVICE)
    torch.manual_seed(Config.INITIALIZATION_SEED)
    np.random.seed(Config.INITIALIZATION_SEED)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    output_dir = prepare_output_dir()
    table = build_truth_table(device)
    epsilons, conditions = condition_epsilons(device)
    model_count = Config.SEED_COUNT * len(conditions)
    inputs = repeat_inputs(table, model_count)
    model = BatchedIndependentMLP(Config.SEED_COUNT, len(conditions)).to(device)
    if Config.OPTIMIZER.lower() != "sgd":
        raise ValueError("当前判别实验固定使用 plain SGD。")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    write_json(output_dir / "config.json", {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    })
    write_json(output_dir / "conditions.json", conditions)
    write_json(output_dir / "truth_table.json", {
        "rule0": Config.RULE0,
        "rule1": Config.RULE1,
        "rule0_bits_000_to_111": [
            rule_output(Config.RULE0, value) for value in range(8)
        ],
        "rule1_bits_000_to_111": [
            rule_output(Config.RULE1, value) for value in range(8)
        ],
        "conflict_neighborhoods": [
            f"{value:03b}" for value in range(8)
            if rule_output(Config.RULE0, value) != rule_output(Config.RULE1, value)
        ],
        "candidate_function_ids": {
            key: f"0x{value:04X}" for key, value in candidate_function_ids().items()
        },
    })

    print("=== Weighted Rule110/Rule30 function switch ===", flush=True)
    print(
        f"device={device} | ratios={list(Config.MAJORITY_RATIOS)} | "
        f"seeds/ratio={Config.SEED_COUNT:,} | models={model_count:,} | "
        f"optimizer=SGD lr={Config.LEARNING_RATE:g} | max_steps={Config.MAX_STEPS:,}",
        flush=True,
    )
    print(
        "conflict neighborhoods="
        + str([
            f"{value:03b}" for value in range(8)
            if rule_output(Config.RULE0, value) != rule_output(Config.RULE1, value)
        ]),
        flush=True,
    )

    step, elapsed_before, summary_rows, top_rows, raw_payload = load_checkpoint(
        output_dir, model, optimizer, device
    )
    initialize_crossing_state(raw_payload, model_count)
    evaluated_steps = {int(row["step"]) for row in summary_rows}
    wall_start = time.perf_counter()
    last_save_step = step
    last_save_wall = time.perf_counter()
    interrupted = False

    def elapsed() -> float:
        return elapsed_before + (time.perf_counter() - wall_start)

    def run_evaluation(current_step: int) -> None:
        nonlocal summary_rows, top_rows, raw_payload
        with torch.no_grad():
            crossing_parts = loss_parts(model, inputs, table, epsilons)
            record_matched_loss_crossings(
                crossing_parts, current_step, raw_payload
            )
        if current_step in evaluated_steps:
            return
        rows, tops, raw = evaluate(
            model, inputs, table, epsilons, conditions, current_step, elapsed()
        )
        summary_rows.extend(rows)
        top_rows.extend(tops)
        raw_payload.update(raw)
        evaluated_steps.add(current_step)
        print_rows(rows)

    try:
        run_evaluation(step)
        while step < Config.MAX_STEPS:
            optimizer.zero_grad(set_to_none=True)
            parts = loss_parts(model, inputs, table, epsilons)
            record_matched_loss_crossings(parts, step, raw_payload)
            parts["train"].sum().backward()
            optimizer.step()
            step += 1

            if (
                step in set(Config.EARLY_EVAL_STEPS)
                or step % Config.EVAL_INTERVAL_STEPS == 0
                or step == Config.MAX_STEPS
            ):
                run_evaluation(step)

            now = time.perf_counter()
            if (
                step - last_save_step >= Config.SAVE_INTERVAL_STEPS
                or now - last_save_wall >= Config.SAVE_INTERVAL_SECONDS
            ):
                save_checkpoint(
                    output_dir, model, optimizer, step, elapsed(),
                    summary_rows, top_rows, raw_payload, conditions,
                )
                last_save_step = step
                last_save_wall = now
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C，正在评估并安全保存……", flush=True)
        run_evaluation(step)
    finally:
        with torch.no_grad():
            final_parts = loss_parts(model, inputs, table, epsilons)
            record_matched_loss_crossings(final_parts, step, raw_payload)
        save_checkpoint(
            output_dir, model, optimizer, step, elapsed(),
            summary_rows, top_rows, raw_payload, conditions,
        )
        make_plot(output_dir, summary_rows)
        make_matched_loss_plot(
            output_dir, build_matched_loss_rows(raw_payload, conditions)
        )
        write_json(output_dir / "summary.json", {
            "last_step": int(step),
            "elapsed_seconds": elapsed(),
            "interrupted": interrupted,
            "model_count": model_count,
            "candidate_function_ids": {
                key: f"0x{value:04X}" for key, value in candidate_function_ids().items()
            },
            "question": (
                "对于固定 hard function，其 SGD 诱导的相对概率是否会随 "
                "matched weighted loss 先降后升或发生换序？"
            ),
            "primary_evaluation_distribution": (
                "与训练目标相同的 Rule110:Rule30 加权分布"
            ),
            "balanced_reference_role": (
                "人为设置的反事实 1:1 梯度诊断；不是验证分布或真理分布"
            ),
        })
        archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
        print("=== 完成 ===", flush=True)
        if archive:
            print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
