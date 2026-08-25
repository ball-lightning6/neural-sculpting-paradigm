"""测量数据量如何重塑 Rule 30 训练中的 train/validation 梯度对齐。

核心问题：
1. 数据充分时，为什么 train/validation raw BCE 会从训练早期同步下降？
2. 增加数据是否会在相同 raw train BCE 水平上，提高真实规则方向的优势？
3. grokking 区间是否对应训练梯度内部相干性和 train/validation 梯度对齐的延迟形成？

脚本完全独立生成一层、循环边界的 Rule 30 数据，不依赖仓库或 AutoDL 上的
外部数据文件。不同训练样本数使用同一随机排列的前缀，所有条件共享同一个、
与最大训练集不相交的验证集。多个模型 seed 采用批量独立 MLP 并行训练。

主口径始终是未经归一化的 raw BCEWithLogitsLoss。梯度诊断使用完整训练集和
完整验证集，不使用当前 minibatch 代替。
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    # =========================
    # 任务与数据
    # =========================
    RULE = 30
    CA_STEPS = 1
    BIT_LENGTH = 30
    BOUNDARY_MODE = "circular"

    # 这些训练集严格嵌套。旧实验的一层 Rule 30 相变点大致在 1,000 附近，
    # 因而同时覆盖明显不足、相变附近和明显充分三个区域。
    TRAIN_COUNTS = (256, 512, 768, 1024, 1280, 1536, 2048, 4096)
    VALIDATION_COUNT = 4096
    DATA_SEED = 20260819

    # =========================
    # 模型
    # =========================
    MODEL_SEEDS = (0, 1, 2, 3, 4)
    HIDDEN_SIZE = 1024
    # 这里直接表示隐藏层总数，不采用旧脚本“首层 + HIDDEN_LAYERS”的歧义口径。
    HIDDEN_LAYER_COUNT = 3
    ACTIVATION = "gelu"
    LAYERNORM_EPS = 1e-5

    # =========================
    # 训练
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 512
    MAX_STEPS = 20_000
    TRAIN_ORDER_SEED = 314159

    # 小训练集使用精确 full batch；大训练集使用同一批次顺序的 minibatch。
    FULL_BATCH_WHEN_COUNT_LEQ_BATCH = True

    # 固定时间轴诊断。另有 raw BCE 阈值触发诊断，用于相同 loss 水平比较。
    FIXED_EVAL_STEPS = (
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000, 4000, 8000, 12_000, 16_000, 20_000,
    )
    RAW_BCE_CROSSING_LEVELS = (
        0.68, 0.60, 0.50, 0.40, 0.30, 0.20,
        0.10, 0.05, 0.02, 0.01, 0.003,
    )
    LOG_INTERVAL_STEPS = 250

    # 梯度诊断默认使用完整数据。若以后扩大到很大数据集，可设置上限。
    MAX_TRAIN_GRADIENT_SAMPLES = None
    MAX_VALIDATION_GRADIENT_SAMPLES = None

    # =========================
    # 输出
    # =========================
    RESULT_DIR = Path("/root/results_rule30_train_val_gradient_alignment")
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


@dataclass(frozen=True)
class RuntimeConfig:
    rule: int
    ca_steps: int
    bit_length: int
    boundary_mode: str
    train_counts: tuple[int, ...]
    validation_count: int
    data_seed: int
    model_seeds: tuple[int, ...]
    hidden_size: int
    hidden_layer_count: int
    activation: str
    layernorm_eps: float
    device: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_steps: int
    train_order_seed: int
    full_batch_when_count_leq_batch: bool
    fixed_eval_steps: tuple[int, ...]
    raw_bce_crossing_levels: tuple[float, ...]
    log_interval_steps: int
    max_train_gradient_samples: int | None
    max_validation_gradient_samples: int | None
    result_dir: Path
    package_results: bool
    smoke_test: bool


@dataclass(frozen=True)
class ParameterBundle:
    hidden_weights: tuple[torch.Tensor, ...]
    hidden_biases: tuple[torch.Tensor, ...]
    norm_weights: tuple[torch.Tensor, ...]
    norm_biases: tuple[torch.Tensor, ...]
    output_weight: torch.Tensor
    output_bias: torch.Tensor

    def flat(self) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []
        for layer_index in range(len(self.hidden_weights)):
            tensors.extend((
                self.hidden_weights[layer_index],
                self.hidden_biases[layer_index],
                self.norm_weights[layer_index],
                self.norm_biases[layer_index],
            ))
        tensors.extend((self.output_weight, self.output_bias))
        return tuple(tensors)


@dataclass
class ObjectiveSnapshot:
    gradients: tuple[torch.Tensor, ...]
    loss: torch.Tensor
    bit_accuracy: torch.Tensor
    exact_accuracy: torch.Tensor
    signed_margin: torch.Tensor
    predictions: torch.Tensor


def build_runtime_config() -> RuntimeConfig:
    if Config.SMOKE_TEST:
        return RuntimeConfig(
            rule=Config.RULE,
            ca_steps=Config.CA_STEPS,
            bit_length=12,
            boundary_mode=Config.BOUNDARY_MODE,
            train_counts=(16, 64),
            validation_count=128,
            data_seed=Config.DATA_SEED,
            model_seeds=(0, 1),
            hidden_size=64,
            hidden_layer_count=2,
            activation=Config.ACTIVATION,
            layernorm_eps=Config.LAYERNORM_EPS,
            device=Config.DEVICE,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            batch_size=64,
            max_steps=10,
            train_order_seed=Config.TRAIN_ORDER_SEED,
            full_batch_when_count_leq_batch=True,
            fixed_eval_steps=(0, 1, 2, 5, 10),
            raw_bce_crossing_levels=(0.68, 0.60, 0.50, 0.30),
            log_interval_steps=1,
            max_train_gradient_samples=None,
            max_validation_gradient_samples=None,
            result_dir=Path(
                "research/ca_phase_transition/"
                "_smoke_rule30_train_val_gradient_alignment"
            ),
            package_results=True,
            smoke_test=True,
        )

    return RuntimeConfig(
        rule=Config.RULE,
        ca_steps=Config.CA_STEPS,
        bit_length=Config.BIT_LENGTH,
        boundary_mode=Config.BOUNDARY_MODE,
        train_counts=tuple(int(value) for value in Config.TRAIN_COUNTS),
        validation_count=int(Config.VALIDATION_COUNT),
        data_seed=int(Config.DATA_SEED),
        model_seeds=tuple(int(value) for value in Config.MODEL_SEEDS),
        hidden_size=int(Config.HIDDEN_SIZE),
        hidden_layer_count=int(Config.HIDDEN_LAYER_COUNT),
        activation=str(Config.ACTIVATION),
        layernorm_eps=float(Config.LAYERNORM_EPS),
        device=str(Config.DEVICE),
        learning_rate=float(Config.LEARNING_RATE),
        weight_decay=float(Config.WEIGHT_DECAY),
        batch_size=int(Config.BATCH_SIZE),
        max_steps=int(Config.MAX_STEPS),
        train_order_seed=int(Config.TRAIN_ORDER_SEED),
        full_batch_when_count_leq_batch=bool(
            Config.FULL_BATCH_WHEN_COUNT_LEQ_BATCH
        ),
        fixed_eval_steps=tuple(int(value) for value in Config.FIXED_EVAL_STEPS),
        raw_bce_crossing_levels=tuple(
            float(value) for value in Config.RAW_BCE_CROSSING_LEVELS
        ),
        log_interval_steps=int(Config.LOG_INTERVAL_STEPS),
        max_train_gradient_samples=Config.MAX_TRAIN_GRADIENT_SAMPLES,
        max_validation_gradient_samples=Config.MAX_VALIDATION_GRADIENT_SAMPLES,
        result_dir=Path(Config.RESULT_DIR),
        package_results=bool(Config.PACKAGE_RESULTS),
        smoke_test=False,
    )


def validate_config(cfg: RuntimeConfig) -> None:
    if cfg.boundary_mode != "circular":
        raise ValueError("当前实验只实现 circular 边界，以匹配既有 Rule 30 数据。")
    if cfg.hidden_layer_count < 1:
        raise ValueError("HIDDEN_LAYER_COUNT 必须至少为 1。")
    if cfg.activation not in {"gelu", "relu", "tanh"}:
        raise ValueError(f"未知激活函数：{cfg.activation}")
    if not cfg.train_counts:
        raise ValueError("TRAIN_COUNTS 不能为空。")
    if tuple(sorted(set(cfg.train_counts))) != cfg.train_counts:
        raise ValueError("TRAIN_COUNTS 必须严格递增且不能重复。")
    if min(cfg.train_counts) < 2:
        raise ValueError("每个训练集至少需要两个样本，才能计算两半梯度对齐。")
    if cfg.validation_count < 2:
        raise ValueError("验证集至少需要两个样本。")
    if not cfg.model_seeds:
        raise ValueError("MODEL_SEEDS 不能为空。")
    if cfg.max_steps < 1:
        raise ValueError("MAX_STEPS 必须为正数。")
    if max(cfg.fixed_eval_steps) > cfg.max_steps:
        raise ValueError("FIXED_EVAL_STEPS 不能超过 MAX_STEPS。")
    if 0 not in cfg.fixed_eval_steps:
        raise ValueError("FIXED_EVAL_STEPS 必须包含 step=0。")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def integer_ids_to_bits(ids: np.ndarray, bit_length: int) -> np.ndarray:
    shifts = np.arange(bit_length - 1, -1, -1, dtype=np.uint64)
    return ((ids[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def evolve_elementary_ca(
    inputs: np.ndarray,
    rule: int,
    steps: int,
) -> np.ndarray:
    state = inputs.astype(np.uint8, copy=True)
    for _ in range(steps):
        left = np.roll(state, shift=1, axis=1)
        right = np.roll(state, shift=-1, axis=1)
        pattern = (left << 2) | (state << 1) | right
        state = ((rule >> pattern) & 1).astype(np.uint8)
    return state.astype(np.float32)


def make_nested_dataset(cfg: RuntimeConfig):
    total_needed = max(cfg.train_counts) + cfg.validation_count
    input_space = 1 << cfg.bit_length
    if total_needed > input_space:
        raise ValueError(
            f"需要 {total_needed} 个互异输入，但输入空间只有 {input_space}。"
        )

    rng = random.Random(cfg.data_seed)
    ids = np.asarray(rng.sample(range(input_space), total_needed), dtype=np.uint64)
    inputs = integer_ids_to_bits(ids, cfg.bit_length)
    outputs = evolve_elementary_ca(inputs, cfg.rule, cfg.ca_steps)
    max_train = max(cfg.train_counts)
    return {
        "train_ids": ids[:max_train],
        "validation_ids": ids[max_train:],
        "train_x": inputs[:max_train],
        "train_y": outputs[:max_train],
        "validation_x": inputs[max_train:],
        "validation_y": outputs[max_train:],
    }


class BatchedIndependentMLP(nn.Module):
    """把多个彼此独立的 MLP 放进参数张量的第 0 维并行训练。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layer_count: int,
        model_seeds: Sequence[int],
        activation: str,
        layernorm_eps: float,
    ) -> None:
        super().__init__()
        self.model_count = len(model_seeds)
        self.hidden_layer_count = hidden_layer_count
        self.activation_name = activation
        self.layernorm_eps = layernorm_eps

        dimensions = [input_size]
        dimensions.extend([hidden_size] * hidden_layer_count)
        dimensions.append(output_size)

        per_model_hidden_weights: list[list[torch.Tensor]] = []
        per_model_hidden_biases: list[list[torch.Tensor]] = []
        per_model_norm_weights: list[list[torch.Tensor]] = []
        per_model_norm_biases: list[list[torch.Tensor]] = []
        per_model_output_weights: list[torch.Tensor] = []
        per_model_output_biases: list[torch.Tensor] = []

        for seed in model_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                hidden_weights = []
                hidden_biases = []
                norm_weights = []
                norm_biases = []
                for layer_index in range(hidden_layer_count):
                    linear = nn.Linear(
                        dimensions[layer_index], dimensions[layer_index + 1]
                    )
                    norm = nn.LayerNorm(
                        dimensions[layer_index + 1], eps=layernorm_eps
                    )
                    hidden_weights.append(linear.weight.detach().clone())
                    hidden_biases.append(linear.bias.detach().clone())
                    norm_weights.append(norm.weight.detach().clone())
                    norm_biases.append(norm.bias.detach().clone())

                output = nn.Linear(dimensions[-2], dimensions[-1])
                per_model_hidden_weights.append(hidden_weights)
                per_model_hidden_biases.append(hidden_biases)
                per_model_norm_weights.append(norm_weights)
                per_model_norm_biases.append(norm_biases)
                per_model_output_weights.append(output.weight.detach().clone())
                per_model_output_biases.append(output.bias.detach().clone())

        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model_weights[layer_index]
                for model_weights in per_model_hidden_weights
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.hidden_biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model_biases[layer_index]
                for model_biases in per_model_hidden_biases
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.norm_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model_weights[layer_index]
                for model_weights in per_model_norm_weights
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.norm_biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model_biases[layer_index]
                for model_biases in per_model_norm_biases
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.output_weight = nn.Parameter(torch.stack(per_model_output_weights))
        self.output_bias = nn.Parameter(torch.stack(per_model_output_biases))

    def full_bundle(self) -> ParameterBundle:
        return ParameterBundle(
            hidden_weights=tuple(self.hidden_weights),
            hidden_biases=tuple(self.hidden_biases),
            norm_weights=tuple(self.norm_weights),
            norm_biases=tuple(self.norm_biases),
            output_weight=self.output_weight,
            output_bias=self.output_bias,
        )

    def selected_bundle(self, model_indices: torch.Tensor) -> ParameterBundle:
        return ParameterBundle(
            hidden_weights=tuple(
                tensor.index_select(0, model_indices)
                for tensor in self.hidden_weights
            ),
            hidden_biases=tuple(
                tensor.index_select(0, model_indices)
                for tensor in self.hidden_biases
            ),
            norm_weights=tuple(
                tensor.index_select(0, model_indices)
                for tensor in self.norm_weights
            ),
            norm_biases=tuple(
                tensor.index_select(0, model_indices)
                for tensor in self.norm_biases
            ),
            output_weight=self.output_weight.index_select(0, model_indices),
            output_bias=self.output_bias.index_select(0, model_indices),
        )

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(values)
        if self.activation_name == "relu":
            return F.relu(values)
        if self.activation_name == "tanh":
            return torch.tanh(values)
        raise ValueError(f"未知激活函数：{self.activation_name}")

    def forward_with_bundle(
        self,
        inputs: torch.Tensor,
        bundle: ParameterBundle,
    ) -> torch.Tensor:
        hidden = inputs
        for layer_index in range(self.hidden_layer_count):
            hidden = torch.bmm(
                hidden, bundle.hidden_weights[layer_index].transpose(1, 2)
            )
            hidden = hidden + bundle.hidden_biases[layer_index][:, None, :]
            hidden = self.activate(hidden)

            mean = hidden.mean(dim=-1, keepdim=True)
            variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
            hidden = (hidden - mean) * torch.rsqrt(
                variance + self.layernorm_eps
            )
            hidden = (
                hidden * bundle.norm_weights[layer_index][:, None, :]
                + bundle.norm_biases[layer_index][:, None, :]
            )

        return (
            torch.bmm(hidden, bundle.output_weight.transpose(1, 2))
            + bundle.output_bias[:, None, :]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_bundle(inputs, self.full_bundle())

    def parameter_blocks(self) -> tuple[str, ...]:
        blocks: list[str] = []
        for layer_index in range(self.hidden_layer_count):
            blocks.extend([f"hidden_{layer_index + 1}"] * 4)
        blocks.extend(["output_head"] * 2)
        return tuple(blocks)


def make_model_layout(cfg: RuntimeConfig):
    model_train_counts: list[int] = []
    model_seeds: list[int] = []
    model_indices_by_count: dict[int, list[int]] = {}
    for train_count in cfg.train_counts:
        indices = []
        for model_seed in cfg.model_seeds:
            indices.append(len(model_train_counts))
            model_train_counts.append(train_count)
            model_seeds.append(model_seed)
        model_indices_by_count[train_count] = indices
    return model_train_counts, model_seeds, model_indices_by_count


def select_diagnostic_rows(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cap: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cap is None or len(inputs) <= cap:
        return inputs, targets
    # 数据排列本身已固定；均匀抽取整段，避免只看前缀。
    indices = torch.linspace(
        0, len(inputs) - 1, steps=cap, device=inputs.device
    ).round().long()
    return inputs.index_select(0, indices), targets.index_select(0, indices)


def objective_snapshot(
    model: BatchedIndependentMLP,
    model_indices: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> ObjectiveSnapshot:
    bundle = model.selected_bundle(model_indices)
    model_count = len(model_indices)
    batched_inputs = inputs.unsqueeze(0).expand(model_count, -1, -1)
    batched_targets = targets.unsqueeze(0).expand(model_count, -1, -1)
    logits = model.forward_with_bundle(batched_inputs, bundle)
    element_loss = F.binary_cross_entropy_with_logits(
        logits, batched_targets, reduction="none"
    )
    loss = element_loss.mean(dim=(1, 2))
    gradients = torch.autograd.grad(
        loss.sum(), bundle.flat(), retain_graph=False, create_graph=False
    )

    with torch.no_grad():
        predictions = logits >= 0
        target_bits = batched_targets >= 0.5
        bit_accuracy = (predictions == target_bits).float().mean(dim=(1, 2))
        exact_accuracy = (
            (predictions == target_bits).all(dim=2).float().mean(dim=1)
        )
        signed_margin = (
            logits * (batched_targets * 2.0 - 1.0)
        ).mean(dim=(1, 2))

    return ObjectiveSnapshot(
        gradients=tuple(gradient.detach() for gradient in gradients),
        loss=loss.detach(),
        bit_accuracy=bit_accuracy.detach(),
        exact_accuracy=exact_accuracy.detach(),
        signed_margin=signed_margin.detach(),
        predictions=predictions.detach(),
    )


def weighted_gradients(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    first_weight: float,
) -> tuple[torch.Tensor, ...]:
    second_weight = 1.0 - first_weight
    return tuple(
        first_weight * first_gradient + second_weight * second_gradient
        for first_gradient, second_gradient in zip(first, second)
    )


def gradient_geometry(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    blocks: Sequence[str],
) -> dict[str, torch.Tensor]:
    model_count = first[0].shape[0]
    block_names = ("total", *dict.fromkeys(blocks))
    result: dict[str, torch.Tensor] = {}

    for block_name in block_names:
        dot = torch.zeros(model_count, dtype=torch.float64, device=first[0].device)
        first_sq = torch.zeros_like(dot)
        second_sq = torch.zeros_like(dot)
        for first_gradient, second_gradient, parameter_block in zip(
            first, second, blocks
        ):
            if block_name != "total" and parameter_block != block_name:
                continue
            first_flat = first_gradient.reshape(model_count, -1)
            second_flat = second_gradient.reshape(model_count, -1)
            dot += (first_flat * second_flat).sum(dim=1, dtype=torch.float64)
            first_sq += first_flat.square().sum(dim=1, dtype=torch.float64)
            second_sq += second_flat.square().sum(dim=1, dtype=torch.float64)

        first_norm = torch.sqrt(first_sq)
        second_norm = torch.sqrt(second_sq)
        denominator = first_norm * second_norm
        cosine = torch.where(
            denominator > 0,
            dot / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.full_like(dot, float("nan")),
        )
        result[f"{block_name}_dot"] = dot
        result[f"{block_name}_first_norm"] = first_norm
        result[f"{block_name}_second_norm"] = second_norm
        result[f"{block_name}_cosine"] = cosine

    return result


def pairwise_agreement(predictions: torch.Tensor) -> tuple[float, float, float]:
    model_count = predictions.shape[0]
    if model_count < 2:
        return 1.0, 1.0, 1.0

    bit_values = []
    exact_values = []
    for first in range(model_count):
        for second in range(first + 1, model_count):
            equal = predictions[first] == predictions[second]
            bit_values.append(float(equal.float().mean().item()))
            exact_values.append(float(equal.all(dim=1).float().mean().item()))
    unanimous = float(
        (predictions == predictions[:1]).all(dim=0).float().mean().item()
    )
    return float(np.mean(bit_values)), float(np.mean(exact_values)), unanimous


def weighted_metric(
    first: torch.Tensor,
    second: torch.Tensor,
    first_count: int,
    second_count: int,
) -> torch.Tensor:
    total = first_count + second_count
    return (first * first_count + second * second_count) / total


def diagnose_condition(
    model: BatchedIndependentMLP,
    cfg: RuntimeConfig,
    train_count: int,
    model_indices_list: Sequence[int],
    model_seeds: Sequence[int],
    train_x_pool: torch.Tensor,
    train_y_pool: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    step: int,
    trigger: str,
    crossed_levels: Sequence[float],
    elapsed_seconds: float,
) -> list[dict[str, object]]:
    model.eval()
    device = train_x_pool.device
    model_indices = torch.as_tensor(
        model_indices_list, dtype=torch.long, device=device
    )

    train_x = train_x_pool[:train_count]
    train_y = train_y_pool[:train_count]
    train_x, train_y = select_diagnostic_rows(
        train_x, train_y, cfg.max_train_gradient_samples
    )
    validation_x_used, validation_y_used = select_diagnostic_rows(
        validation_x, validation_y, cfg.max_validation_gradient_samples
    )

    midpoint = len(train_x) // 2
    train_first = objective_snapshot(
        model, model_indices, train_x[:midpoint], train_y[:midpoint]
    )
    train_second = objective_snapshot(
        model, model_indices, train_x[midpoint:], train_y[midpoint:]
    )
    validation_midpoint = len(validation_x_used) // 2
    validation_first = objective_snapshot(
        model,
        model_indices,
        validation_x_used[:validation_midpoint],
        validation_y_used[:validation_midpoint],
    )
    validation_second = objective_snapshot(
        model,
        model_indices,
        validation_x_used[validation_midpoint:],
        validation_y_used[validation_midpoint:],
    )

    first_weight = midpoint / len(train_x)
    train_gradients = weighted_gradients(
        train_first.gradients, train_second.gradients, first_weight
    )
    validation_first_weight = validation_midpoint / len(validation_x_used)
    validation_gradients = weighted_gradients(
        validation_first.gradients,
        validation_second.gradients,
        validation_first_weight,
    )
    blocks = model.parameter_blocks()
    train_validation_geometry = gradient_geometry(
        train_gradients, validation_gradients, blocks
    )
    train_half_geometry = gradient_geometry(
        train_first.gradients, train_second.gradients, blocks
    )
    validation_half_geometry = gradient_geometry(
        validation_first.gradients, validation_second.gradients, blocks
    )

    train_loss = weighted_metric(
        train_first.loss, train_second.loss, midpoint, len(train_x) - midpoint
    )
    train_bit_accuracy = weighted_metric(
        train_first.bit_accuracy,
        train_second.bit_accuracy,
        midpoint,
        len(train_x) - midpoint,
    )
    train_exact_accuracy = weighted_metric(
        train_first.exact_accuracy,
        train_second.exact_accuracy,
        midpoint,
        len(train_x) - midpoint,
    )
    train_signed_margin = weighted_metric(
        train_first.signed_margin,
        train_second.signed_margin,
        midpoint,
        len(train_x) - midpoint,
    )
    validation_loss = weighted_metric(
        validation_first.loss,
        validation_second.loss,
        validation_midpoint,
        len(validation_x_used) - validation_midpoint,
    )
    validation_bit_accuracy = weighted_metric(
        validation_first.bit_accuracy,
        validation_second.bit_accuracy,
        validation_midpoint,
        len(validation_x_used) - validation_midpoint,
    )
    validation_exact_accuracy = weighted_metric(
        validation_first.exact_accuracy,
        validation_second.exact_accuracy,
        validation_midpoint,
        len(validation_x_used) - validation_midpoint,
    )
    validation_signed_margin = weighted_metric(
        validation_first.signed_margin,
        validation_second.signed_margin,
        validation_midpoint,
        len(validation_x_used) - validation_midpoint,
    )
    validation_predictions = torch.cat(
        (validation_first.predictions, validation_second.predictions), dim=1
    )
    bit_agreement, exact_agreement, unanimous_bit_agreement = pairwise_agreement(
        validation_predictions
    )

    rows: list[dict[str, object]] = []
    for local_index, model_seed in enumerate(model_seeds):
        total_dot = train_validation_geometry["total_dot"][local_index]
        row: dict[str, object] = {
            "step": step,
            "trigger": trigger,
            "crossed_raw_bce_levels": json.dumps(list(crossed_levels)),
            "elapsed_seconds": elapsed_seconds,
            "train_count": train_count,
            "model_seed": int(model_seed),
            "train_gradient_sample_count": len(train_x),
            "validation_gradient_sample_count": len(validation_x_used),
            "train_raw_bce": float(train_loss[local_index].item()),
            "validation_raw_bce": float(validation_loss[local_index].item()),
            "validation_minus_train_raw_bce": float(
                validation_loss[local_index].item()
                - train_loss[local_index].item()
            ),
            "train_bit_accuracy": float(train_bit_accuracy[local_index].item()),
            "validation_bit_accuracy": float(
                validation_bit_accuracy[local_index].item()
            ),
            "train_exact_accuracy": float(
                train_exact_accuracy[local_index].item()
            ),
            "validation_exact_accuracy": float(
                validation_exact_accuracy[local_index].item()
            ),
            "train_signed_margin": float(
                train_signed_margin[local_index].item()
            ),
            "validation_signed_margin": float(
                validation_signed_margin[local_index].item()
            ),
            "train_validation_gradient_dot": float(total_dot.item()),
            "train_gradient_norm": float(
                train_validation_geometry["total_first_norm"][local_index].item()
            ),
            "validation_gradient_norm": float(
                train_validation_geometry["total_second_norm"][local_index].item()
            ),
            "train_validation_gradient_cosine": float(
                train_validation_geometry["total_cosine"][local_index].item()
            ),
            "train_half_gradient_cosine": float(
                train_half_geometry["total_cosine"][local_index].item()
            ),
            "validation_half_gradient_cosine": float(
                validation_half_geometry["total_cosine"][local_index].item()
            ),
            # 仅表示同一点沿原始 full-train 梯度走一个 SGD 学习率时的
            # 一阶预测；真实优化器为 AdamW，因此不能当作实际 loss 变化。
            "raw_sgd_predicted_validation_delta": float(
                -cfg.learning_rate * total_dot.item()
            ),
            "validation_seed_pairwise_bit_agreement": bit_agreement,
            "validation_seed_pairwise_exact_agreement": exact_agreement,
            "validation_seed_unanimous_bit_agreement": unanimous_bit_agreement,
        }

        for block_name in dict.fromkeys(blocks):
            row[f"train_validation_gradient_cosine_{block_name}"] = float(
                train_validation_geometry[f"{block_name}_cosine"][
                    local_index
                ].item()
            )
            row[f"train_half_gradient_cosine_{block_name}"] = float(
                train_half_geometry[f"{block_name}_cosine"][
                    local_index
                ].item()
            )
        rows.append(row)

    cosine_values = [
        float(row["train_validation_gradient_cosine"]) for row in rows
    ]
    half_values = [float(row["train_half_gradient_cosine"]) for row in rows]
    print(
        f"  [诊断] step={step:>6,} | n={train_count:>5,} | "
        f"train/val BCE={np.mean([float(row['train_raw_bce']) for row in rows]):.5g}/"
        f"{np.mean([float(row['validation_raw_bce']) for row in rows]):.5g} | "
        f"grad cos={np.mean(cosine_values):+.4f} | "
        f"train-half cos={np.mean(half_values):+.4f} | "
        f"val-ceiling={np.mean([float(row['validation_half_gradient_cosine']) for row in rows]):+.4f} | "
        f"val bit={np.mean([float(row['validation_bit_accuracy']) for row in rows]):.3%} | "
        f"trigger={trigger}"
    )
    model.train()
    return rows


def make_training_batch(
    cfg: RuntimeConfig,
    train_counts_tensor: torch.Tensor,
    count_count: int,
    seed_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = train_counts_tensor.device
    count_batches: list[torch.Tensor] = []
    count_masks: list[torch.Tensor] = []
    positions = torch.arange(cfg.batch_size, device=device)

    for count_index in range(count_count):
        train_count = int(train_counts_tensor[count_index].item())
        if cfg.full_batch_when_count_leq_batch and train_count <= cfg.batch_size:
            indices = positions.remainder(train_count)
            mask = positions < train_count
        else:
            indices = torch.randint(
                0,
                train_count,
                size=(cfg.batch_size,),
                generator=generator,
                device=device,
            )
            mask = torch.ones(cfg.batch_size, dtype=torch.bool, device=device)
        count_batches.append(indices)
        count_masks.append(mask)

    indices_by_count = torch.stack(count_batches)
    masks_by_count = torch.stack(count_masks)
    return (
        indices_by_count.repeat_interleave(seed_count, dim=0),
        masks_by_count.repeat_interleave(seed_count, dim=0),
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    numeric_metrics = (
        "train_raw_bce",
        "validation_raw_bce",
        "validation_minus_train_raw_bce",
        "train_bit_accuracy",
        "validation_bit_accuracy",
        "train_exact_accuracy",
        "validation_exact_accuracy",
        "train_validation_gradient_cosine",
        "train_half_gradient_cosine",
        "validation_half_gradient_cosine",
        "train_gradient_norm",
        "validation_gradient_norm",
        "validation_seed_pairwise_bit_agreement",
    )
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["train_count"]), int(row["step"]), str(row["trigger"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for (train_count, step, trigger), group in sorted(grouped.items()):
        aggregate: dict[str, object] = {
            "train_count": train_count,
            "step": step,
            "trigger": trigger,
            "seed_count": len(group),
        }
        for metric in numeric_metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            aggregate[f"{metric}_mean"] = float(np.nanmean(values))
            aggregate[f"{metric}_std"] = float(np.nanstd(values))
            aggregate[f"{metric}_min"] = float(np.nanmin(values))
            aggregate[f"{metric}_max"] = float(np.nanmax(values))
        output.append(aggregate)
    return output


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 2:
        return float("nan")
    first_ranks = average_ranks(first)
    second_ranks = average_ranks(second)
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def make_fixed_loss_comparison(
    rows: Sequence[dict[str, object]],
    loss_levels: Sequence[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["train_count"]), int(row["model_seed"])), []
        ).append(row)

    selected_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for target_loss in loss_levels:
        level_rows: list[dict[str, object]] = []
        for (train_count, model_seed), trajectory in grouped.items():
            best = min(
                trajectory,
                key=lambda row: abs(
                    math.log(max(float(row["train_raw_bce"]), 1e-12))
                    - math.log(target_loss)
                ),
            )
            actual_loss = float(best["train_raw_bce"])
            ratio = max(actual_loss / target_loss, target_loss / actual_loss)
            # 诊断点与目标相差超过 35%，就不伪装成同 loss 比较。
            if ratio > 1.35:
                continue
            selected = dict(best)
            selected["target_train_raw_bce"] = target_loss
            selected["actual_to_target_loss_ratio"] = actual_loss / target_loss
            selected_rows.append(selected)
            level_rows.append(selected)

        if not level_rows:
            continue
        count_means: dict[int, list[float]] = {}
        gap_means: dict[int, list[float]] = {}
        for row in level_rows:
            count = int(row["train_count"])
            count_means.setdefault(count, []).append(
                float(row["train_validation_gradient_cosine"])
            )
            gap_means.setdefault(count, []).append(
                float(row["validation_minus_train_raw_bce"])
            )
        counts = sorted(count_means)
        cosine_means = [float(np.mean(count_means[count])) for count in counts]
        generalization_gaps = [float(np.mean(gap_means[count])) for count in counts]
        summaries.append({
            "target_train_raw_bce": target_loss,
            "available_train_counts": counts,
            "condition_count": len(counts),
            "row_count": len(level_rows),
            "spearman_log_n_vs_gradient_cosine": spearman_correlation(
                [math.log(count) for count in counts], cosine_means
            ),
            "spearman_log_n_vs_generalization_gap": spearman_correlation(
                [math.log(count) for count in counts], generalization_gaps
            ),
            "gradient_cosine_by_train_count": dict(zip(counts, cosine_means)),
            "generalization_gap_by_train_count": dict(
                zip(counts, generalization_gaps)
            ),
        })
    return selected_rows, summaries


def fixed_step_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if "fixed_step" in str(row["trigger"])]


def plot_dashboard(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
) -> None:
    selected = fixed_step_rows(rows)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(cfg.train_counts)))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for color, train_count in zip(colors, cfg.train_counts):
        condition_rows = [
            row for row in selected if int(row["train_count"]) == train_count
        ]
        steps = sorted({int(row["step"]) for row in condition_rows})

        def means(metric: str) -> np.ndarray:
            return np.asarray([
                np.mean([
                    float(row[metric])
                    for row in condition_rows
                    if int(row["step"]) == step
                ])
                for step in steps
            ])

        label = f"n={train_count}"
        axes[0, 0].plot(
            steps, means("train_validation_gradient_cosine"),
            color=color, label=label,
        )
        axes[0, 0].plot(
            steps,
            means("validation_half_gradient_cosine"),
            color=color,
            linestyle=":",
            alpha=0.55,
        )
        axes[0, 1].plot(
            steps, means("train_half_gradient_cosine"), color=color, label=label
        )
        axes[0, 2].plot(
            means("train_raw_bce"),
            means("train_validation_gradient_cosine"),
            color=color,
            marker="o",
            markersize=2,
            label=label,
        )
        axes[1, 0].plot(
            steps, means("train_raw_bce"), color=color, linestyle="-", label=label
        )
        axes[1, 0].plot(
            steps,
            means("validation_raw_bce"),
            color=color,
            linestyle="--",
        )
        axes[1, 1].plot(
            steps,
            means("validation_minus_train_raw_bce"),
            color=color,
            label=label,
        )
        axes[1, 2].plot(
            steps, means("validation_bit_accuracy"), color=color, label=label
        )

    axes[0, 0].set_title("Train / validation gradient cosine")
    axes[0, 1].set_title("Gradient cosine between train halves")
    axes[0, 2].set_title("Gradient alignment at equal raw train BCE")
    axes[1, 0].set_title("Raw BCE: solid=train, dashed=validation")
    axes[1, 1].set_title("Validation BCE - train BCE")
    axes[1, 2].set_title("Validation bit accuracy")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 1], axes[1, 2]):
        axis.set_xscale("symlog", linthresh=10)
        axis.grid(alpha=0.25)
    axes[0, 2].set_xscale("log")
    axes[0, 2].invert_xaxis()
    axes[0, 2].grid(alpha=0.25)
    axes[1, 0].set_xscale("symlog", linthresh=10)
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(alpha=0.25)

    axes[0, 0].set_ylabel("cosine")
    axes[0, 1].set_ylabel("cosine")
    axes[0, 2].set_xlabel("raw train BCE")
    axes[0, 2].set_ylabel("cosine")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("raw BCE")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("generalization gap")
    axes[1, 2].set_xlabel("step")
    axes[1, 2].set_ylabel("accuracy")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 2].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_alignment_dashboard.png", dpi=180)
    plt.close(fig)


def plot_fixed_loss_comparison(
    output_dir: Path,
    selected_rows: Sequence[dict[str, object]],
) -> None:
    if not selected_rows:
        return
    levels = sorted({
        float(row["target_train_raw_bce"]) for row in selected_rows
    }, reverse=True)
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(levels)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for color, level in zip(colors, levels):
        level_rows = [
            row
            for row in selected_rows
            if float(row["target_train_raw_bce"]) == level
        ]
        counts = sorted({int(row["train_count"]) for row in level_rows})
        cosine = [
            np.mean([
                float(row["train_validation_gradient_cosine"])
                for row in level_rows
                if int(row["train_count"]) == count
            ])
            for count in counts
        ]
        gap = [
            np.mean([
                float(row["validation_minus_train_raw_bce"])
                for row in level_rows
                if int(row["train_count"]) == count
            ])
            for count in counts
        ]
        axes[0].plot(counts, cosine, marker="o", color=color, label=f"BCE={level:g}")
        axes[1].plot(counts, gap, marker="o", color=color, label=f"BCE={level:g}")

    axes[0].set_title("Equal-loss train/validation gradient alignment")
    axes[1].set_title("Equal-loss generalization gap")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("training sample count")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("gradient cosine")
    axes[1].set_ylabel("validation BCE - train BCE")
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_raw_bce_comparison.png", dpi=180)
    plt.close(fig)


def make_final_summary(
    rows: Sequence[dict[str, object]],
    fixed_loss_summaries: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
    elapsed_seconds: float,
) -> dict[str, object]:
    fixed = fixed_step_rows(rows)
    initial = [row for row in fixed if int(row["step"]) == 0]
    final = [row for row in fixed if int(row["step"]) == cfg.max_steps]

    def summarize_by_count(source: Sequence[dict[str, object]]):
        output = {}
        for train_count in cfg.train_counts:
            group = [
                row for row in source if int(row["train_count"]) == train_count
            ]
            if not group:
                continue
            output[str(train_count)] = {
                "train_raw_bce": float(np.mean([
                    float(row["train_raw_bce"]) for row in group
                ])),
                "validation_raw_bce": float(np.mean([
                    float(row["validation_raw_bce"]) for row in group
                ])),
                "validation_bit_accuracy": float(np.mean([
                    float(row["validation_bit_accuracy"]) for row in group
                ])),
                "validation_exact_accuracy": float(np.mean([
                    float(row["validation_exact_accuracy"]) for row in group
                ])),
                "train_validation_gradient_cosine": float(np.mean([
                    float(row["train_validation_gradient_cosine"])
                    for row in group
                ])),
                "train_half_gradient_cosine": float(np.mean([
                    float(row["train_half_gradient_cosine"]) for row in group
                ])),
                "validation_half_gradient_cosine": float(np.mean([
                    float(row["validation_half_gradient_cosine"])
                    for row in group
                ])),
            }
        return output

    initial_count_means = summarize_by_count(initial)
    initial_counts = [
        count for count in cfg.train_counts if str(count) in initial_count_means
    ]
    initial_cosines = [
        initial_count_means[str(count)]["train_validation_gradient_cosine"]
        for count in initial_counts
    ]

    return {
        "experiment": "Rule 30 train/validation gradient alignment",
        "primary_loss": "raw BCEWithLogitsLoss",
        "elapsed_seconds": elapsed_seconds,
        "model_count": len(cfg.train_counts) * len(cfg.model_seeds),
        "initial_by_train_count": initial_count_means,
        "final_by_train_count": summarize_by_count(final),
        "initial_spearman_log_n_vs_gradient_cosine": spearman_correlation(
            [math.log(count) for count in initial_counts], initial_cosines
        ),
        "fixed_raw_bce_comparisons": list(fixed_loss_summaries),
        "interpretation_keys": {
            "train_validation_gradient_cosine": (
                "越接近 1，完整训练集 raw BCE 梯度与独立验证集 raw BCE "
                "梯度越一致；它直接预测原始梯度下降是否同时降低验证 loss。"
            ),
            "train_half_gradient_cosine": (
                "训练集随机排列前后两半的梯度一致性；用于测量训练样本内部的"
                "规则方向是否相干。"
            ),
            "validation_half_gradient_cosine": (
                "独立验证集两半的梯度一致性，是有限验证样本下可达到的参考"
                "上限；train/validation 余弦应结合它解释。"
            ),
            "fixed_raw_bce_comparisons": (
                "在近似相同 raw train BCE 水平比较不同 n，直接检验数据是否"
                "重塑高 loss 和中等 loss 区域，而不只改变最终极小值。"
            ),
        },
    }


def package_results(output_dir: Path) -> Path:
    archive_base = output_dir.parent / f"{output_dir.name}_package"
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
    return archive_path


def main() -> None:
    cfg = build_runtime_config()
    validate_config(cfg)
    set_global_seed(cfg.data_seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = cfg.result_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {
        **asdict(cfg),
        "result_dir": str(cfg.result_dir),
    })

    device = torch.device(cfg.device)
    dataset = make_nested_dataset(cfg)
    np.savez_compressed(
        output_dir / "dataset_split_ids.npz",
        train_ids=dataset["train_ids"],
        validation_ids=dataset["validation_ids"],
    )
    train_x_pool = torch.from_numpy(dataset["train_x"]).to(device)
    train_y_pool = torch.from_numpy(dataset["train_y"]).to(device)
    validation_x = torch.from_numpy(dataset["validation_x"]).to(device)
    validation_y = torch.from_numpy(dataset["validation_y"]).to(device)

    model_train_counts, expanded_model_seeds, indices_by_count = make_model_layout(cfg)
    model = BatchedIndependentMLP(
        input_size=cfg.bit_length,
        output_size=cfg.bit_length,
        hidden_size=cfg.hidden_size,
        hidden_layer_count=cfg.hidden_layer_count,
        model_seeds=expanded_model_seeds,
        activation=cfg.activation,
        layernorm_eps=cfg.layernorm_eps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    parameter_count_per_model = sum(
        parameter.numel() for parameter in model.parameters()
    ) // len(model_train_counts)
    print("=== Rule 30 train/validation 梯度对齐实验 ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(
        f"任务：Rule {cfg.rule} | steps={cfg.ca_steps} | "
        f"{cfg.bit_length} bit -> {cfg.bit_length} bit | {cfg.boundary_mode}"
    )
    print(
        f"MLP：{cfg.bit_length} -> {cfg.hidden_size} x "
        f"{cfg.hidden_layer_count} -> {cfg.bit_length} | "
        f"activation={cfg.activation} | dropout=0 | weight_decay={cfg.weight_decay}"
    )
    print(
        f"训练样本数：{list(cfg.train_counts)} | seeds={list(cfg.model_seeds)} | "
        f"并行模型={len(model_train_counts)}"
    )
    print(
        f"单模型参数={parameter_count_per_model:,} | batch={cfg.batch_size} | "
        f"max_steps={cfg.max_steps:,} | validation={cfg.validation_count:,}"
    )
    print(f"结果目录：{output_dir}")

    trajectory_path = output_dir / "trajectory.jsonl"
    all_rows: list[dict[str, object]] = []
    evaluated_pairs: set[tuple[int, int]] = set()
    pending_levels = {
        train_count: set(cfg.raw_bce_crossing_levels)
        for train_count in cfg.train_counts
    }
    started = time.time()

    def run_diagnostics(
        step: int,
        requests: dict[int, dict[str, object]],
    ) -> None:
        for train_count in cfg.train_counts:
            if train_count not in requests:
                continue
            if (step, train_count) in evaluated_pairs:
                continue
            request = requests[train_count]
            rows = diagnose_condition(
                model=model,
                cfg=cfg,
                train_count=train_count,
                model_indices_list=indices_by_count[train_count],
                model_seeds=cfg.model_seeds,
                train_x_pool=train_x_pool,
                train_y_pool=train_y_pool,
                validation_x=validation_x,
                validation_y=validation_y,
                step=step,
                trigger="+".join(request["triggers"]),
                crossed_levels=request["crossed_levels"],
                elapsed_seconds=time.time() - started,
            )
            all_rows.extend(rows)
            append_jsonl(trajectory_path, rows)
            evaluated_pairs.add((step, train_count))

    run_diagnostics(
        0,
        {
            train_count: {"triggers": ["fixed_step"], "crossed_levels": []}
            for train_count in cfg.train_counts
        },
    )

    count_count = len(cfg.train_counts)
    seed_count = len(cfg.model_seeds)
    train_counts_tensor = torch.as_tensor(
        cfg.train_counts, dtype=torch.long, device=device
    )
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(cfg.train_order_seed)
    fixed_steps = set(cfg.fixed_eval_steps)
    model.train()

    for step in range(1, cfg.max_steps + 1):
        batch_indices, batch_mask = make_training_batch(
            cfg,
            train_counts_tensor,
            count_count=count_count,
            seed_count=seed_count,
            generator=batch_generator,
        )
        batch_x = train_x_pool[batch_indices]
        batch_y = train_y_pool[batch_indices]
        logits = model(batch_x)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            logits, batch_y, reduction="none"
        ).mean(dim=2)
        mask_float = batch_mask.float()
        loss_per_model = (
            (per_sample_loss * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0)
        )

        optimizer.zero_grad(set_to_none=True)
        loss_per_model.sum().backward()
        optimizer.step()

        requests: dict[int, dict[str, object]] = {}
        if step in fixed_steps:
            for train_count in cfg.train_counts:
                requests[train_count] = {
                    "triggers": ["fixed_step"],
                    "crossed_levels": [],
                }

        mean_batch_loss_by_count = loss_per_model.detach().view(
            count_count, seed_count
        ).mean(dim=1)
        for count_index, train_count in enumerate(cfg.train_counts):
            current_loss = float(mean_batch_loss_by_count[count_index].item())
            crossed = sorted(
                [
                    level
                    for level in pending_levels[train_count]
                    if current_loss <= level
                ],
                reverse=True,
            )
            if not crossed:
                continue
            request = requests.setdefault(
                train_count, {"triggers": [], "crossed_levels": []}
            )
            request["triggers"].append("raw_bce_crossing")
            request["crossed_levels"].extend(crossed)
            pending_levels[train_count].difference_update(crossed)

        if requests:
            run_diagnostics(step, requests)

        if step % cfg.log_interval_steps == 0 or step == cfg.max_steps:
            elapsed = time.time() - started
            compact_losses = " | ".join(
                f"n={train_count}:{mean_batch_loss_by_count[index].item():.3g}"
                for index, train_count in enumerate(cfg.train_counts)
            )
            print(
                f"step={step:>6,}/{cfg.max_steps:,} | "
                f"{step / max(elapsed, 1e-9):.1f} step/s | {compact_losses}"
            )

    elapsed_seconds = time.time() - started
    write_csv(output_dir / "trajectory.csv", all_rows)
    aggregated = aggregate_rows(all_rows)
    write_csv(output_dir / "trajectory_aggregated.csv", aggregated)
    write_json(output_dir / "trajectory_aggregated.json", aggregated)

    selected_fixed_loss, fixed_loss_summaries = make_fixed_loss_comparison(
        all_rows, cfg.raw_bce_crossing_levels
    )
    write_csv(output_dir / "fixed_raw_bce_comparison.csv", selected_fixed_loss)
    write_json(
        output_dir / "fixed_raw_bce_summary.json", fixed_loss_summaries
    )
    plot_dashboard(output_dir, all_rows, cfg)
    plot_fixed_loss_comparison(output_dir, selected_fixed_loss)

    summary = make_final_summary(
        all_rows, fixed_loss_summaries, cfg, elapsed_seconds
    )
    write_json(output_dir / "summary.json", summary)

    print("\n=== 实验完成 ===")
    print(f"耗时：{elapsed_seconds / 60.0:.2f} 分钟")
    print(f"轨迹：{output_dir / 'trajectory.csv'}")
    print(f"同 raw BCE 比较：{output_dir / 'fixed_raw_bce_comparison.csv'}")
    print(f"汇总：{output_dir / 'summary.json'}")
    if cfg.package_results:
        print(f"下载压缩包：{package_results(output_dir)}")


if __name__ == "__main__":
    main()
