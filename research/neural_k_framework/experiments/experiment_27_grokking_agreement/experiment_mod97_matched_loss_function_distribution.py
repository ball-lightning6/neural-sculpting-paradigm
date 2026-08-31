"""模97除法 grokking 的 matched-loss 完整函数分布实验。

数据严格采用原论文定义：

    x / y (mod 97),  0 <= x < 97,  0 < y < 97

输入是两个 97 维 one-hot 的拼接，模型看不到整数大小或模运算结构。第一层
Linear 等价于分别查询两个符号 embedding 后相加。输出是 97 类分类。

脚本在每个 seed 第一次跨过预注册 train raw-CE 水平时，保存完整9312输入上的
hard function。它同时测量目标函数质量、非目标插值函数质量、函数熵、seed
agreement 和 modal function，用于区分：

1. 高 loss 区域是否集中到某个可重复 shortcut；
2. grokking 是否对应完整 hard-function 概率运输；
3. 增加数据是否把目标函数占优区域推向更高 loss。

这是 SGD 诱导分布实验，不把结果解释成静态 prior 条件体积。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    # =========================
    # 原版模除法任务
    # =========================
    PRIME = 97
    # 从低数据过拟合区扫到高数据同步泛化区。
    TRAIN_FRACTIONS = (0.60, 0.70, 0.80, 0.90)
    DATA_SEED = 20260820

    # =========================
    # MLP：参数量约 67 万，接近原论文约 40 万参数的 Transformer
    # =========================
    # 32 seed 是5090上约数小时的首轮判别；结果含糊时再扩到64。
    MODEL_SEEDS = tuple(range(32))
    HIDDEN_SIZE = 512
    HIDDEN_LAYER_COUNT = 3
    ACTIVATION = "gelu"
    LAYERNORM_EPS = 1e-5

    # =========================
    # 训练：复用原论文主图的 Adam、无 weight decay、lr=1e-3
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-3
    WARMUP_STEPS = 10
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 512
    MAX_STEPS = 200_000
    TRAIN_ORDER_SEED = 314159

    # 避免 softmax true_probability - 1 在 float32 中提前消减为零。
    NUMERICALLY_STABLE_CROSS_ENTROPY = True

    EVAL_BATCH_SIZE = 2048
    LATE_EVAL_INTERVAL = 2_000
    LOG_INTERVAL_STEPS = 1_000
    CHECKPOINT_INTERVAL_STEPS = 20_000

    MATCHED_LOSS_LEVELS = (
        4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1,
        0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001,
        5e-4, 3e-4, 2e-4, 1e-4, 3e-5, 1e-5,
        3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8,
        3e-9, 1e-9, 3e-10, 1e-10, 3e-11, 1e-11, 3e-12, 1e-12,
    )

    ANALYSIS_START_STEP = 10
    MIN_RISE_RATIO = 1.25
    MIN_SECOND_DESCENT_RATIO = 1.50

    RESULT_DIR = Path("/root/results_mod97_matched_loss_function_distribution")
    RESUME = True
    OVERWRITE_EXISTING = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


@dataclass(frozen=True)
class RuntimeConfig:
    prime: int
    train_fractions: tuple[float, ...]
    train_counts: tuple[int, ...]
    data_seed: int
    model_seeds: tuple[int, ...]
    hidden_size: int
    hidden_layer_count: int
    activation: str
    layernorm_eps: float
    device: str
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    batch_size: int
    max_steps: int
    train_order_seed: int
    numerically_stable_cross_entropy: bool
    eval_batch_size: int
    late_eval_interval: int
    log_interval_steps: int
    checkpoint_interval_steps: int
    matched_loss_levels: tuple[float, ...]
    analysis_start_step: int
    min_rise_ratio: float
    min_second_descent_ratio: float
    result_dir: Path
    resume: bool
    overwrite_existing: bool
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


def fraction_to_count(total: int, fraction: float) -> int:
    return int(round(total * fraction))


def build_runtime_config() -> RuntimeConfig:
    if Config.SMOKE_TEST:
        prime = 11
        fractions = (0.35, 0.50, 0.70)
        total = prime * (prime - 1)
        return RuntimeConfig(
            prime=prime,
            train_fractions=fractions,
            train_counts=tuple(fraction_to_count(total, value) for value in fractions),
            data_seed=Config.DATA_SEED,
            model_seeds=(0, 1),
            hidden_size=64,
            hidden_layer_count=2,
            activation=Config.ACTIVATION,
            layernorm_eps=Config.LAYERNORM_EPS,
            device=Config.DEVICE,
            learning_rate=Config.LEARNING_RATE,
            warmup_steps=3,
            weight_decay=0.0,
            batch_size=64,
            max_steps=20,
            train_order_seed=Config.TRAIN_ORDER_SEED,
            numerically_stable_cross_entropy=True,
            eval_batch_size=128,
            late_eval_interval=5,
            log_interval_steps=5,
            checkpoint_interval_steps=10,
            matched_loss_levels=(3.0, 2.0, 1.0, 0.5, 0.2),
            analysis_start_step=3,
            min_rise_ratio=Config.MIN_RISE_RATIO,
            min_second_descent_ratio=Config.MIN_SECOND_DESCENT_RATIO,
            result_dir=Path(
                "research/ca_phase_transition/"
                "_smoke_mod97_matched_loss_function_distribution"
            ),
            resume=False,
            overwrite_existing=True,
            package_results=True,
            smoke_test=True,
        )

    prime = int(Config.PRIME)
    fractions = tuple(float(value) for value in Config.TRAIN_FRACTIONS)
    total = prime * (prime - 1)
    return RuntimeConfig(
        prime=prime,
        train_fractions=fractions,
        train_counts=tuple(fraction_to_count(total, value) for value in fractions),
        data_seed=int(Config.DATA_SEED),
        model_seeds=tuple(int(value) for value in Config.MODEL_SEEDS),
        hidden_size=int(Config.HIDDEN_SIZE),
        hidden_layer_count=int(Config.HIDDEN_LAYER_COUNT),
        activation=str(Config.ACTIVATION),
        layernorm_eps=float(Config.LAYERNORM_EPS),
        device=str(Config.DEVICE),
        learning_rate=float(Config.LEARNING_RATE),
        warmup_steps=int(Config.WARMUP_STEPS),
        weight_decay=float(Config.WEIGHT_DECAY),
        batch_size=int(Config.BATCH_SIZE),
        max_steps=int(Config.MAX_STEPS),
        train_order_seed=int(Config.TRAIN_ORDER_SEED),
        numerically_stable_cross_entropy=bool(
            Config.NUMERICALLY_STABLE_CROSS_ENTROPY
        ),
        eval_batch_size=int(Config.EVAL_BATCH_SIZE),
        late_eval_interval=int(Config.LATE_EVAL_INTERVAL),
        log_interval_steps=int(Config.LOG_INTERVAL_STEPS),
        checkpoint_interval_steps=int(Config.CHECKPOINT_INTERVAL_STEPS),
        matched_loss_levels=tuple(float(value) for value in Config.MATCHED_LOSS_LEVELS),
        analysis_start_step=int(Config.ANALYSIS_START_STEP),
        min_rise_ratio=float(Config.MIN_RISE_RATIO),
        min_second_descent_ratio=float(Config.MIN_SECOND_DESCENT_RATIO),
        result_dir=Path(Config.RESULT_DIR),
        resume=bool(Config.RESUME),
        overwrite_existing=bool(Config.OVERWRITE_EXISTING),
        package_results=bool(Config.PACKAGE_RESULTS),
        smoke_test=False,
    )


def validate_config(cfg: RuntimeConfig) -> None:
    if cfg.prime < 3:
        raise ValueError("PRIME 必须至少为 3。")
    if tuple(sorted(set(cfg.train_fractions))) != cfg.train_fractions:
        raise ValueError("TRAIN_FRACTIONS 必须严格递增且不能重复。")
    if min(cfg.train_fractions) <= 0 or max(cfg.train_fractions) >= 1:
        raise ValueError("训练比例必须严格位于 (0, 1)。")
    if len(set(cfg.train_counts)) != len(cfg.train_counts):
        raise ValueError("不同训练比例被舍入成了相同样本数。")
    if not cfg.model_seeds:
        raise ValueError("MODEL_SEEDS 不能为空。")
    if cfg.activation not in {"gelu", "relu", "tanh"}:
        raise ValueError(f"未知激活函数：{cfg.activation}")
    if not cfg.matched_loss_levels:
        raise ValueError("MATCHED_LOSS_LEVELS 不能为空。")
    if any(value <= 0 for value in cfg.matched_loss_levels):
        raise ValueError("MATCHED_LOSS_LEVELS 必须全部大于0。")
    if tuple(sorted(set(cfg.matched_loss_levels), reverse=True)) != (
        cfg.matched_loss_levels
    ):
        raise ValueError("MATCHED_LOSS_LEVELS 必须严格递减且不能重复。")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_modular_division_dataset(cfg: RuntimeConfig):
    pairs = np.asarray(
        [(x, y) for x in range(cfg.prime) for y in range(1, cfg.prime)],
        dtype=np.int64,
    )
    inverse = np.zeros(cfg.prime, dtype=np.int64)
    for value in range(1, cfg.prime):
        inverse[value] = pow(value, -1, cfg.prime)
    targets = (pairs[:, 0] * inverse[pairs[:, 1]]) % cfg.prime

    inputs = np.zeros((len(pairs), cfg.prime * 2), dtype=np.float32)
    row_indices = np.arange(len(pairs))
    inputs[row_indices, pairs[:, 0]] = 1.0
    inputs[row_indices, cfg.prime + pairs[:, 1]] = 1.0

    permutation = np.random.default_rng(cfg.data_seed).permutation(len(pairs))
    return {
        "pairs": pairs[permutation],
        "inputs": inputs[permutation],
        "targets": targets[permutation],
        "permutation": permutation,
    }


class BatchedIndependentMLP(nn.Module):
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
        self.hidden_layer_count = hidden_layer_count
        self.activation_name = activation
        self.layernorm_eps = layernorm_eps
        dimensions = [input_size]
        dimensions.extend([hidden_size] * hidden_layer_count)
        dimensions.append(output_size)

        all_hidden_weights: list[list[torch.Tensor]] = []
        all_hidden_biases: list[list[torch.Tensor]] = []
        all_norm_weights: list[list[torch.Tensor]] = []
        all_norm_biases: list[list[torch.Tensor]] = []
        all_output_weights: list[torch.Tensor] = []
        all_output_biases: list[torch.Tensor] = []

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
                all_hidden_weights.append(hidden_weights)
                all_hidden_biases.append(hidden_biases)
                all_norm_weights.append(norm_weights)
                all_norm_biases.append(norm_biases)
                all_output_weights.append(output.weight.detach().clone())
                all_output_biases.append(output.bias.detach().clone())

        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_hidden_weights
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.hidden_biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_hidden_biases
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.norm_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_norm_weights
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.norm_biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_norm_biases
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.output_weight = nn.Parameter(torch.stack(all_output_weights))
        self.output_bias = nn.Parameter(torch.stack(all_output_biases))

    def full_bundle(self) -> ParameterBundle:
        return ParameterBundle(
            hidden_weights=tuple(self.hidden_weights),
            hidden_biases=tuple(self.hidden_biases),
            norm_weights=tuple(self.norm_weights),
            norm_biases=tuple(self.norm_biases),
            output_weight=self.output_weight,
            output_bias=self.output_bias,
        )

    def selected_bundle(self, indices: torch.Tensor) -> ParameterBundle:
        return ParameterBundle(
            hidden_weights=tuple(
                value.index_select(0, indices) for value in self.hidden_weights
            ),
            hidden_biases=tuple(
                value.index_select(0, indices) for value in self.hidden_biases
            ),
            norm_weights=tuple(
                value.index_select(0, indices) for value in self.norm_weights
            ),
            norm_biases=tuple(
                value.index_select(0, indices) for value in self.norm_biases
            ),
            output_weight=self.output_weight.index_select(0, indices),
            output_bias=self.output_bias.index_select(0, indices),
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
        self, inputs: torch.Tensor, bundle: ParameterBundle
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


def stable_multiclass_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """逐样本多类 CE，避免 true_probability - 1 的消减。"""
    target_logits = logits.gather(-1, targets.unsqueeze(-1))
    relative = logits - target_logits
    relative = relative.scatter(
        -1,
        targets.unsqueeze(-1),
        torch.full_like(target_logits, -torch.inf),
    )
    log_other_exp_sum = torch.logsumexp(relative, dim=-1)
    return F.softplus(log_other_exp_sum)


def elementwise_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    stable: bool,
) -> torch.Tensor:
    if stable:
        return stable_multiclass_cross_entropy(logits, targets)
    return F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none"
    )


def make_model_layout(cfg: RuntimeConfig):
    expanded_seeds: list[int] = []
    indices_by_count: dict[int, list[int]] = {}
    for count in cfg.train_counts:
        indices = []
        for seed in cfg.model_seeds:
            indices.append(len(expanded_seeds))
            expanded_seeds.append(seed)
        indices_by_count[count] = indices
    return expanded_seeds, indices_by_count


def make_training_batch(
    cfg: RuntimeConfig,
    train_counts: torch.Tensor,
    seed_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batches = [
        torch.randint(
            0,
            int(count.item()),
            (cfg.batch_size,),
            generator=generator,
            device=train_counts.device,
        )
        for count in train_counts
    ]
    return torch.stack(batches).repeat_interleave(seed_count, dim=0)


def make_eval_steps(cfg: RuntimeConfig) -> set[int]:
    steps = {
        0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300,
        500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10_000,
        12_500, 15_000, 17_500, 20_000,
    }
    steps.update(range(1_000, min(cfg.max_steps, 20_000) + 1, 500))
    if cfg.max_steps > 20_000:
        steps.update(range(
            20_000 + cfg.late_eval_interval,
            cfg.max_steps + 1,
            cfg.late_eval_interval,
        ))
    steps.add(cfg.max_steps)
    return {step for step in steps if 0 <= step <= cfg.max_steps}


@torch.no_grad()
def evaluate_group(
    model: BatchedIndependentMLP,
    model_indices: Sequence[int],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    stable_cross_entropy: bool,
    keep_predictions: bool,
) -> dict[str, object]:
    device = inputs.device
    indices = torch.as_tensor(model_indices, dtype=torch.long, device=device)
    bundle = model.selected_bundle(indices)
    model_count = len(indices)
    loss_sum = torch.zeros(model_count, dtype=torch.float64, device=device)
    correct = torch.zeros_like(loss_sum)
    margin_sum = torch.zeros_like(loss_sum)
    correct_loss_sum = torch.zeros_like(loss_sum)
    incorrect_loss_sum = torch.zeros_like(loss_sum)
    correct_margin_sum = torch.zeros_like(loss_sum)
    incorrect_margin_sum = torch.zeros_like(loss_sum)
    incorrect_confidence_sum = torch.zeros_like(loss_sum)
    correct_count = torch.zeros_like(loss_sum)
    incorrect_count = torch.zeros_like(loss_sum)
    true_probability_sum = torch.zeros_like(loss_sum)
    predicted_confidence_sum = torch.zeros_like(loss_sum)
    prediction_chunks: list[torch.Tensor] = []

    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        x = inputs[start:end].unsqueeze(0).expand(model_count, -1, -1)
        y = targets[start:end].unsqueeze(0).expand(model_count, -1)
        logits = model.forward_with_bundle(x, bundle)
        losses = elementwise_cross_entropy(logits, y, stable_cross_entropy)
        predictions = logits.argmax(dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        true_probabilities = probabilities.gather(
            -1, y.unsqueeze(-1)
        ).squeeze(-1)
        predicted_confidences = probabilities.max(dim=-1).values
        correct_mask = predictions == y
        incorrect_mask = ~correct_mask
        true_logits = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        masked = logits.scatter(
            -1,
            y.unsqueeze(-1),
            torch.full_like(y.unsqueeze(-1), -torch.inf, dtype=logits.dtype),
        )
        margins = true_logits - masked.max(dim=-1).values
        loss_sum += losses.sum(dim=1, dtype=torch.float64)
        correct += correct_mask.sum(dim=1, dtype=torch.float64)
        margin_sum += margins.sum(dim=1, dtype=torch.float64)
        correct_loss_sum += (losses * correct_mask).sum(
            dim=1, dtype=torch.float64
        )
        incorrect_loss_sum += (losses * incorrect_mask).sum(
            dim=1, dtype=torch.float64
        )
        correct_margin_sum += (margins * correct_mask).sum(
            dim=1, dtype=torch.float64
        )
        incorrect_margin_sum += (margins * incorrect_mask).sum(
            dim=1, dtype=torch.float64
        )
        incorrect_confidence_sum += (
            predicted_confidences * incorrect_mask
        ).sum(dim=1, dtype=torch.float64)
        correct_count += correct_mask.sum(dim=1, dtype=torch.float64)
        incorrect_count += incorrect_mask.sum(dim=1, dtype=torch.float64)
        true_probability_sum += true_probabilities.sum(
            dim=1, dtype=torch.float64
        )
        predicted_confidence_sum += predicted_confidences.sum(
            dim=1, dtype=torch.float64
        )
        if keep_predictions:
            prediction_chunks.append(predictions.cpu())

    nan_values = torch.full_like(loss_sum, float("nan"))
    result: dict[str, object] = {
        "raw_cross_entropy": (loss_sum / len(inputs)).cpu().numpy(),
        "accuracy": (correct / len(inputs)).cpu().numpy(),
        "classification_margin": (margin_sum / len(inputs)).cpu().numpy(),
        "correct_example_cross_entropy": torch.where(
            correct_count > 0,
            correct_loss_sum / correct_count.clamp_min(1.0),
            nan_values,
        ).cpu().numpy(),
        "incorrect_example_cross_entropy": torch.where(
            incorrect_count > 0,
            incorrect_loss_sum / incorrect_count.clamp_min(1.0),
            nan_values,
        ).cpu().numpy(),
        "correct_example_margin": torch.where(
            correct_count > 0,
            correct_margin_sum / correct_count.clamp_min(1.0),
            nan_values,
        ).cpu().numpy(),
        "incorrect_example_margin": torch.where(
            incorrect_count > 0,
            incorrect_margin_sum / incorrect_count.clamp_min(1.0),
            nan_values,
        ).cpu().numpy(),
        "mean_true_class_probability": (
            true_probability_sum / len(inputs)
        ).cpu().numpy(),
        "mean_predicted_confidence": (
            predicted_confidence_sum / len(inputs)
        ).cpu().numpy(),
        "incorrect_predicted_confidence": torch.where(
            incorrect_count > 0,
            incorrect_confidence_sum / incorrect_count.clamp_min(1.0),
            nan_values,
        ).cpu().numpy(),
    }
    if keep_predictions:
        result["predictions"] = torch.cat(prediction_chunks, dim=1)
    return result


def pairwise_agreement(predictions: torch.Tensor) -> float:
    if predictions.shape[0] < 2:
        return 1.0
    values = predictions.numpy()
    class_count = int(values.max()) + 1
    return efficient_pairwise_agreement(values, class_count)


def initialize_function_crossings(
    cfg: RuntimeConfig,
    model_count: int,
    domain_size: int,
) -> dict[str, np.ndarray]:
    level_count = len(cfg.matched_loss_levels)
    return {
        "matched_loss_levels": np.asarray(
            cfg.matched_loss_levels, dtype=np.float64
        ),
        "crossing_steps": np.full(
            (model_count, level_count), -1, dtype=np.int64
        ),
        "crossing_train_loss": np.full(
            (model_count, level_count), np.nan, dtype=np.float32
        ),
        "crossing_train_accuracy": np.full(
            (model_count, level_count), np.nan, dtype=np.float32
        ),
        "crossing_validation_accuracy": np.full(
            (model_count, level_count), np.nan, dtype=np.float32
        ),
        "crossing_full_accuracy": np.full(
            (model_count, level_count), np.nan, dtype=np.float32
        ),
        # 97类可安全存入 uint8；255保留为“尚未 crossing”。
        "crossing_predictions": np.full(
            (model_count, level_count, domain_size), 255, dtype=np.uint8
        ),
    }


def validate_function_crossings(
    state: dict[str, np.ndarray],
    cfg: RuntimeConfig,
    model_count: int,
    domain_size: int,
) -> None:
    expected_levels = np.asarray(cfg.matched_loss_levels, dtype=np.float64)
    actual_levels = np.asarray(state["matched_loss_levels"], dtype=np.float64)
    if not np.array_equal(expected_levels, actual_levels):
        raise ValueError("checkpoint 的 MATCHED_LOSS_LEVELS 与当前配置不一致。")
    expected = (model_count, len(expected_levels))
    if tuple(state["crossing_steps"].shape) != expected:
        raise ValueError("checkpoint 的 crossing step shape 不一致。")
    if tuple(state["crossing_predictions"].shape) != (*expected, domain_size):
        raise ValueError("checkpoint 的 crossing prediction shape 不一致。")


@torch.no_grad()
def record_function_crossings(
    model: BatchedIndependentMLP,
    cfg: RuntimeConfig,
    indices_by_count: dict[int, list[int]],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    current_rows: Sequence[dict[str, object]],
    step: int,
    state: dict[str, np.ndarray],
) -> int:
    """在每个 seed 首次到达指定 train CE 时保存完整 hard function。"""
    row_lookup = {
        (int(row["train_count"]), int(row["model_seed"])): row
        for row in current_rows
    }
    levels = state["matched_loss_levels"]
    seed_count = len(cfg.model_seeds)
    new_crossings = 0

    for condition_index, count in enumerate(cfg.train_counts):
        start = condition_index * seed_count
        end = start + seed_count
        losses = np.asarray([
            float(row_lookup[(count, seed)]["train_raw_cross_entropy"])
            for seed in cfg.model_seeds
        ])
        pending = state["crossing_steps"][start:end] < 0
        crossed = losses[:, None] <= levels[None, :]
        newly_crossed = pending & crossed
        if not np.any(newly_crossed):
            continue

        full = evaluate_group(
            model,
            indices_by_count[count],
            inputs,
            targets,
            cfg.eval_batch_size,
            cfg.numerically_stable_cross_entropy,
            keep_predictions=True,
        )
        predictions = full["predictions"].numpy().astype(np.uint8, copy=False)
        full_accuracy = np.asarray(full["accuracy"], dtype=np.float32)

        for local_index, seed in enumerate(cfg.model_seeds):
            global_index = start + local_index
            row = row_lookup[(count, seed)]
            for level_index in np.flatnonzero(newly_crossed[local_index]):
                state["crossing_steps"][global_index, level_index] = int(step)
                state["crossing_train_loss"][global_index, level_index] = float(
                    row["train_raw_cross_entropy"]
                )
                state["crossing_train_accuracy"][global_index, level_index] = float(
                    row["train_accuracy"]
                )
                state["crossing_validation_accuracy"][
                    global_index, level_index
                ] = float(row["validation_accuracy"])
                state["crossing_full_accuracy"][global_index, level_index] = float(
                    full_accuracy[local_index]
                )
                state["crossing_predictions"][global_index, level_index] = (
                    predictions[local_index]
                )
                new_crossings += 1
    return new_crossings


def prediction_fingerprint(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prediction, dtype=np.uint8).tobytes()).hexdigest()[:16]


def efficient_pairwise_agreement(predictions: np.ndarray, class_count: int) -> float:
    model_count, domain_size = predictions.shape
    if model_count < 2:
        return 1.0
    agreeing_pairs = 0
    for class_index in range(class_count):
        counts = np.sum(predictions == class_index, axis=0, dtype=np.int64)
        agreeing_pairs += int(np.sum(counts * (counts - 1) // 2))
    denominator = math.comb(model_count, 2) * domain_size
    return agreeing_pairs / denominator


def modal_function_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    votes = np.stack([
        np.sum(predictions == class_index, axis=0, dtype=np.int64)
        for class_index in range(class_count)
    ])
    modal = votes.argmax(axis=0).astype(np.uint8)
    return {
        "modal_function_accuracy": float(np.mean(modal == targets)),
        "modal_function_is_target": float(np.array_equal(modal, targets)),
        "mean_modal_vote_fraction": float(np.mean(votes.max(axis=0) / len(predictions))),
    }


def build_function_distribution_rows(
    cfg: RuntimeConfig,
    state: dict[str, np.ndarray],
    targets: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_count = len(cfg.model_seeds)
    levels = state["matched_loss_levels"]

    for condition_index, (fraction, count) in enumerate(
        zip(cfg.train_fractions, cfg.train_counts)
    ):
        start = condition_index * seed_count
        end = start + seed_count
        for level_index, level in enumerate(levels):
            steps = state["crossing_steps"][start:end, level_index]
            valid = steps >= 0
            selected_count = int(np.sum(valid))
            row: dict[str, object] = {
                "train_fraction": float(fraction),
                "train_count": int(count),
                "loss_level": float(level),
                "selected_count": selected_count,
                "crossed_fraction": selected_count / seed_count,
            }
            if selected_count == 0:
                rows.append(row)
                continue

            selected_predictions = state["crossing_predictions"][
                start:end, level_index
            ][valid]
            unique, counts = np.unique(
                selected_predictions, axis=0, return_counts=True
            )
            probabilities = counts / selected_count
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            target_matches = np.all(selected_predictions == targets[None], axis=1)
            unique_target = np.all(unique == targets[None], axis=1)
            top_index = int(np.argmax(counts))
            non_target_indices = np.flatnonzero(~unique_target)
            if len(non_target_indices):
                top_non_target_index = int(
                    non_target_indices[np.argmax(counts[non_target_indices])]
                )
                top_non_target_count = int(counts[top_non_target_index])
                top_non_target_fingerprint = prediction_fingerprint(
                    unique[top_non_target_index]
                )
                top_non_target_accuracy = float(np.mean(
                    unique[top_non_target_index] == targets
                ))
            else:
                top_non_target_count = 0
                top_non_target_fingerprint = ""
                top_non_target_accuracy = float("nan")

            train_accuracy = state["crossing_train_accuracy"][
                start:end, level_index
            ][valid]
            full_accuracy = state["crossing_full_accuracy"][
                start:end, level_index
            ][valid]
            target_count = int(np.sum(target_matches))
            target_odds = (target_count + 0.5) / (top_non_target_count + 0.5)
            row.update({
                "median_crossing_step": float(np.median(steps[valid])),
                "median_actual_train_loss": float(np.median(
                    state["crossing_train_loss"][start:end, level_index][valid]
                )),
                "mean_train_accuracy": float(np.mean(train_accuracy)),
                "mean_validation_accuracy": float(np.mean(
                    state["crossing_validation_accuracy"][
                        start:end, level_index
                    ][valid]
                )),
                "mean_full_accuracy": float(np.mean(full_accuracy)),
                "p_target_function": target_count / selected_count,
                "p_non_target_interpolator": float(np.mean(
                    (train_accuracy >= 1.0 - 1e-12) & (~target_matches)
                )),
                "unique_function_count": int(len(unique)),
                "function_entropy_bits": entropy,
                "effective_function_count": float(2.0 ** entropy),
                "top_function_probability": int(counts[top_index]) / selected_count,
                "top_function_fingerprint": prediction_fingerprint(unique[top_index]),
                "top_function_is_target": float(unique_target[top_index]),
                "top_function_accuracy": float(np.mean(unique[top_index] == targets)),
                "top_non_target_probability": top_non_target_count / selected_count,
                "top_non_target_fingerprint": top_non_target_fingerprint,
                "top_non_target_accuracy": top_non_target_accuracy,
                "target_over_top_non_target_odds_jeffreys": float(target_odds),
                "log_target_over_top_non_target_odds": float(math.log(target_odds)),
                "seed_pairwise_agreement": efficient_pairwise_agreement(
                    selected_predictions, cfg.prime
                ),
                **modal_function_metrics(
                    selected_predictions, targets, cfg.prime
                ),
            })
            rows.append(row)
    return rows


def plot_function_distribution(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(cfg.train_counts)))
    for color, fraction, count in zip(
        colors, cfg.train_fractions, cfg.train_counts
    ):
        group = [
            row for row in rows
            if int(row["train_count"]) == count
            and int(row.get("selected_count", 0)) > 0
        ]
        group.sort(key=lambda row: float(row["loss_level"]), reverse=True)
        levels = np.asarray([row["loss_level"] for row in group])
        label = f"{fraction:.0%} (n={count})"
        axes[0, 0].plot(
            levels, [row["p_target_function"] for row in group],
            color=color, marker="o", markersize=3, label=f"target {label}",
        )
        axes[0, 0].plot(
            levels, [row["top_non_target_probability"] for row in group],
            color=color, linestyle="--", marker="o", markersize=3,
            label=f"top non-target {label}",
        )
        axes[0, 1].plot(
            levels, [row["mean_full_accuracy"] for row in group],
            color=color, marker="o", markersize=3, label=f"mean {label}",
        )
        axes[0, 1].plot(
            levels, [row["modal_function_accuracy"] for row in group],
            color=color, linestyle="--", marker="o", markersize=3,
            label=f"modal {label}",
        )
        axes[1, 0].plot(
            levels, [row["function_entropy_bits"] for row in group],
            color=color, marker="o", markersize=3, label=label,
        )
        axes[1, 1].plot(
            levels, [row["seed_pairwise_agreement"] for row in group],
            color=color, marker="o", markersize=3, label=label,
        )
    titles = (
        "Exact target vs dominant non-target function",
        "Full-domain target accuracy",
        "Full hard-function entropy",
        "Seed pairwise function agreement",
    )
    for axis, title in zip(axes.reshape(-1), titles):
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("matched train raw CE (high -> low)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "matched_loss_function_distribution.png", dpi=180)
    plt.close(figure)


def persist_function_distribution(
    output_dir: Path,
    cfg: RuntimeConfig,
    state: dict[str, np.ndarray],
    targets: np.ndarray,
) -> list[dict[str, object]]:
    rows = build_function_distribution_rows(cfg, state, targets)
    write_csv(output_dir / "matched_loss_function_distribution.csv", rows)
    np.savez_compressed(output_dir / "matched_loss_functions.npz", **state)
    plot_function_distribution(output_dir, rows, cfg)
    return rows


def evaluate_all_conditions(
    model: BatchedIndependentMLP,
    cfg: RuntimeConfig,
    indices_by_count: dict[int, list[int]],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    step: int,
    elapsed_seconds: float,
) -> list[dict[str, object]]:
    model.eval()
    max_train = max(cfg.train_counts)
    common_validation_x = inputs[max_train:]
    common_validation_y = targets[max_train:]
    rows: list[dict[str, object]] = []
    print(f"\n[完整评估] step={step:,}")

    for fraction, count in zip(cfg.train_fractions, cfg.train_counts):
        indices = indices_by_count[count]
        train = evaluate_group(
            model,
            indices,
            inputs[:count],
            targets[:count],
            cfg.eval_batch_size,
            cfg.numerically_stable_cross_entropy,
            keep_predictions=True,
        )
        validation = evaluate_group(
            model,
            indices,
            inputs[count:],
            targets[count:],
            cfg.eval_batch_size,
            cfg.numerically_stable_cross_entropy,
            keep_predictions=True,
        )
        common_validation = evaluate_group(
            model,
            indices,
            common_validation_x,
            common_validation_y,
            cfg.eval_batch_size,
            cfg.numerically_stable_cross_entropy,
            keep_predictions=False,
        )
        agreement = pairwise_agreement(validation["predictions"])
        full_predictions = torch.cat(
            (train["predictions"], validation["predictions"]), dim=1
        )
        full_np = full_predictions.numpy().astype(np.uint8, copy=False)
        target_np = targets.detach().cpu().numpy().astype(np.uint8, copy=False)
        full_target_exact = np.all(full_np == target_np[None], axis=1)
        full_accuracy = np.mean(full_np == target_np[None], axis=1)
        unique, function_counts = np.unique(
            full_np, axis=0, return_counts=True
        )
        function_probabilities = function_counts / len(full_np)
        function_entropy = float(-np.sum(
            function_probabilities * np.log2(function_probabilities)
        ))
        unique_target = np.all(unique == target_np[None], axis=1)
        top_index = int(np.argmax(function_counts))
        non_target_indices = np.flatnonzero(~unique_target)
        top_non_target_probability = (
            int(np.max(function_counts[non_target_indices])) / len(full_np)
            if len(non_target_indices) else 0.0
        )
        modal_metrics = modal_function_metrics(full_np, target_np, cfg.prime)
        full_population_metrics = {
            "full_target_function_probability": float(np.mean(full_target_exact)),
            "full_unique_function_count": int(len(unique)),
            "full_function_entropy_bits": function_entropy,
            "full_effective_function_count": float(2.0 ** function_entropy),
            "full_top_function_probability": int(function_counts[top_index]) / len(full_np),
            "full_top_function_is_target": float(unique_target[top_index]),
            "full_top_non_target_probability": top_non_target_probability,
            "full_seed_pairwise_agreement": efficient_pairwise_agreement(
                full_np, cfg.prime
            ),
            **{
                f"full_{key}": value for key, value in modal_metrics.items()
            },
        }
        for local_index, seed in enumerate(cfg.model_seeds):
            rows.append({
                "step": step,
                "elapsed_seconds": elapsed_seconds,
                "train_fraction": fraction,
                "train_count": count,
                "validation_count": len(inputs) - count,
                "model_seed": seed,
                "loss_implementation": (
                    "stable_relative_logsumexp"
                    if cfg.numerically_stable_cross_entropy
                    else "torch_cross_entropy"
                ),
                "train_raw_cross_entropy": float(
                    train["raw_cross_entropy"][local_index]
                ),
                "validation_raw_cross_entropy": float(
                    validation["raw_cross_entropy"][local_index]
                ),
                "common_validation_raw_cross_entropy": float(
                    common_validation["raw_cross_entropy"][local_index]
                ),
                "train_accuracy": float(train["accuracy"][local_index]),
                "validation_accuracy": float(
                    validation["accuracy"][local_index]
                ),
                "common_validation_accuracy": float(
                    common_validation["accuracy"][local_index]
                ),
                "train_margin": float(
                    train["classification_margin"][local_index]
                ),
                "validation_margin": float(
                    validation["classification_margin"][local_index]
                ),
                "validation_correct_example_cross_entropy": float(
                    validation["correct_example_cross_entropy"][local_index]
                ),
                "validation_incorrect_example_cross_entropy": float(
                    validation["incorrect_example_cross_entropy"][local_index]
                ),
                "validation_correct_example_margin": float(
                    validation["correct_example_margin"][local_index]
                ),
                "validation_incorrect_example_margin": float(
                    validation["incorrect_example_margin"][local_index]
                ),
                "validation_mean_true_class_probability": float(
                    validation["mean_true_class_probability"][local_index]
                ),
                "validation_mean_predicted_confidence": float(
                    validation["mean_predicted_confidence"][local_index]
                ),
                "validation_incorrect_predicted_confidence": float(
                    validation["incorrect_predicted_confidence"][local_index]
                ),
                "validation_seed_pairwise_agreement": agreement,
                "full_accuracy": float(full_accuracy[local_index]),
                "full_target_function_exact": float(
                    full_target_exact[local_index]
                ),
                **full_population_metrics,
            })
        print(
            f"  fraction={fraction:.0%} n={count:>4} | train/val CE="
            f"{np.mean(train['raw_cross_entropy']):.6g}/"
            f"{np.mean(validation['raw_cross_entropy']):.6g} | "
            f"train/val acc={np.mean(train['accuracy']):.3%}/"
            f"{np.mean(validation['accuracy']):.3%} | "
            f"wrong CE={np.nanmean(validation['incorrect_example_cross_entropy']):.3g} | "
            f"wrong conf={np.nanmean(validation['incorrect_predicted_confidence']):.3%} | "
            f"agreement={agreement:.3%} | target-fn="
            f"{full_population_metrics['full_target_function_probability']:.3%} | "
            f"function-H={function_entropy:.2f} bit"
        )
    model.train()
    return rows


def analyze_curve(
    rows: Sequence[dict[str, object]], cfg: RuntimeConfig
) -> dict[str, object]:
    ordered = sorted(
        (row for row in rows if int(row["step"]) >= cfg.analysis_start_step),
        key=lambda row: int(row["step"]),
    )
    if len(ordered) < 3:
        return {
            "classification": "insufficient_history",
            "rise_ratio": 1.0,
            "second_descent_ratio": 1.0,
        }
    losses = np.asarray([
        max(float(row["validation_raw_cross_entropy"]), 1e-30)
        for row in ordered
    ])
    steps = np.asarray([int(row["step"]) for row in ordered])
    accuracies = np.asarray([float(row["validation_accuracy"]) for row in ordered])

    running_min_index = 0
    strongest_rise = (0, 0, 1.0)
    best_double = None
    best_score = -math.inf
    for peak_index in range(1, len(losses)):
        if losses[peak_index - 1] < losses[running_min_index]:
            running_min_index = peak_index - 1
        valley_index = running_min_index
        rise_ratio = losses[peak_index] / losses[valley_index]
        if rise_ratio > strongest_rise[2]:
            strongest_rise = (valley_index, peak_index, rise_ratio)
        if peak_index == len(losses) - 1:
            continue
        future_offset = int(np.argmin(losses[peak_index + 1:]))
        second_index = peak_index + 1 + future_offset
        descent_ratio = losses[peak_index] / losses[second_index]
        score = math.log(max(rise_ratio, 1.0)) + math.log(
            max(descent_ratio, 1.0)
        )
        if (
            rise_ratio >= cfg.min_rise_ratio
            and descent_ratio >= cfg.min_second_descent_ratio
            and score > best_score
        ):
            best_score = score
            best_double = (
                valley_index, peak_index, second_index,
                rise_ratio, descent_ratio,
            )

    if best_double is not None:
        valley_index, peak_index, second_index, rise_ratio, descent_ratio = (
            best_double
        )
        classification = "double_descent_detected"
    else:
        valley_index, peak_index, rise_ratio = strongest_rise
        second_index = peak_index + int(np.argmin(losses[peak_index:]))
        descent_ratio = losses[peak_index] / losses[second_index]
        if rise_ratio >= cfg.min_rise_ratio:
            classification = "rise_without_second_descent"
        elif accuracies[-1] >= 0.99:
            classification = "smooth_generalization"
        else:
            classification = "partial_or_no_generalization"

    return {
        "classification": classification,
        "first_valley_step": int(steps[valley_index]),
        "first_valley_loss": float(losses[valley_index]),
        "peak_step": int(steps[peak_index]),
        "peak_loss": float(losses[peak_index]),
        "second_descent_step": int(steps[second_index]),
        "second_descent_loss": float(losses[second_index]),
        "rise_ratio": float(rise_ratio),
        "second_descent_ratio": float(descent_ratio),
        "final_step": int(steps[-1]),
        "final_validation_loss": float(losses[-1]),
        "final_validation_accuracy": float(accuracies[-1]),
    }


def analyze_all_curves(
    rows: Sequence[dict[str, object]], cfg: RuntimeConfig
) -> list[dict[str, object]]:
    output = []
    for fraction, count in zip(cfg.train_fractions, cfg.train_counts):
        for seed in cfg.model_seeds:
            curve = [
                row for row in rows
                if int(row["train_count"]) == count
                and int(row["model_seed"]) == seed
            ]
            summary = analyze_curve(curve, cfg)
            summary.update({
                "train_fraction": fraction,
                "train_count": count,
                "model_seed": seed,
            })
            output.append(summary)
    return output


def append_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
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
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(
    rows: Sequence[dict[str, object]], count: int, seed: int | None = None
) -> list[dict[str, object]]:
    selected = [row for row in rows if int(row["train_count"]) == count]
    if seed is not None:
        selected = [row for row in selected if int(row["model_seed"]) == seed]
    return sorted(selected, key=lambda row: int(row["step"]))


def mean_curve(
    rows: Sequence[dict[str, object]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    steps = sorted({int(row["step"]) for row in rows})
    values = [
        np.nanmean([
            float(row[metric]) for row in rows if int(row["step"]) == step
        ])
        for step in steps
    ]
    return np.asarray(steps), np.asarray(values)


def plot_loss_curves(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    shapes: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
    log_y: bool,
) -> None:
    columns = 4
    row_count = math.ceil(len(cfg.train_counts) / columns)
    fig, axes = plt.subplots(row_count, columns, figsize=(19, 5 * row_count))
    axes = np.asarray(axes).reshape(-1)
    chance_loss = math.log(cfg.prime)
    for axis, fraction, count in zip(
        axes, cfg.train_fractions, cfg.train_counts
    ):
        group = condition_rows(rows, count)
        for seed in cfg.model_seeds:
            seed_rows = condition_rows(rows, count, seed)
            axis.plot(
                [int(row["step"]) for row in seed_rows],
                [float(row["validation_raw_cross_entropy"]) for row in seed_rows],
                color="#2b8c6b", alpha=0.3, linewidth=1.0,
            )
        train_steps, train_loss = mean_curve(group, "train_raw_cross_entropy")
        val_steps, val_loss = mean_curve(group, "validation_raw_cross_entropy")
        axis.plot(
            train_steps, train_loss, "--", color="#c23b3b",
            linewidth=2, label="train mean",
        )
        axis.plot(
            val_steps, val_loss, color="#13795b",
            linewidth=2.2, label="validation mean",
        )
        axis.axhline(
            chance_loss, color="#777777", linestyle=":", linewidth=1,
            label="chance CE",
        )
        classifications = [
            str(row["classification"])
            for row in shapes if int(row["train_count"]) == count
        ]
        doubles = classifications.count("double_descent_detected")
        rises = classifications.count("rise_without_second_descent")
        axis.set_title(
            f"train={fraction:.0%} ({count}) | double={doubles}, rise={rises}"
        )
        axis.set_xscale("symlog", linthresh=10)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("optimization step")
        axis.set_ylabel("raw cross entropy")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    for axis in axes[len(cfg.train_counts):]:
        axis.axis("off")
    fig.tight_layout()
    filename = "loss_curves_log_y.png" if log_y else "loss_curves_linear_y.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_accuracy_curves(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
) -> None:
    columns = 4
    row_count = math.ceil(len(cfg.train_counts) / columns)
    fig, axes = plt.subplots(row_count, columns, figsize=(19, 5 * row_count))
    axes = np.asarray(axes).reshape(-1)
    for axis, fraction, count in zip(
        axes, cfg.train_fractions, cfg.train_counts
    ):
        group = condition_rows(rows, count)
        train_steps, train_accuracy = mean_curve(group, "train_accuracy")
        val_steps, val_accuracy = mean_curve(group, "validation_accuracy")
        axis.plot(
            train_steps, train_accuracy, "--", color="#c23b3b",
            linewidth=2, label="train",
        )
        axis.plot(
            val_steps, val_accuracy, color="#13795b",
            linewidth=2.2, label="validation",
        )
        axis.axhline(
            1.0 / cfg.prime, color="#777777", linestyle=":", linewidth=1
        )
        axis.set_title(f"train={fraction:.0%} ({count})")
        axis.set_xscale("symlog", linthresh=10)
        axis.set_ylim(-0.01, 1.01)
        axis.set_xlabel("optimization step")
        axis.set_ylabel("accuracy")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    for axis in axes[len(cfg.train_counts):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curves.png", dpi=180)
    plt.close(fig)


def plot_middle_phase_diagnostics(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.04, 0.96, len(cfg.train_counts)))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = (
        (
            "validation_incorrect_example_cross_entropy",
            "Cross entropy on currently wrong validation examples",
            "wrong-example CE",
        ),
        (
            "validation_incorrect_predicted_confidence",
            "Confidence of wrong validation predictions",
            "wrong predicted confidence",
        ),
        (
            "validation_incorrect_example_margin",
            "True-class margin on wrong validation examples",
            "true logit - best wrong logit",
        ),
        (
            "validation_mean_true_class_probability",
            "Mean probability assigned to the true class",
            "true-class probability",
        ),
        (
            "validation_seed_pairwise_agreement",
            "Pairwise seed agreement on validation functions",
            "agreement",
        ),
        (
            "validation_accuracy",
            "Validation accuracy",
            "accuracy",
        ),
    )
    for color, fraction, count in zip(
        colors, cfg.train_fractions, cfg.train_counts
    ):
        group = condition_rows(rows, count)
        label = f"{fraction:.0%}"
        for axis, (metric, title, ylabel) in zip(axes.reshape(-1), metrics):
            steps, values = mean_curve(group, metric)
            axis.plot(steps, values, color=color, linewidth=1.8, label=label)
            axis.set_title(title)
            axis.set_ylabel(ylabel)

    for axis in axes.reshape(-1):
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, ncol=2)
    axes[0, 1].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 2].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "middle_phase_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_time_function_distribution(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(cfg.train_counts)))
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = (
        ("full_target_function_probability", "Exact target function probability"),
        ("full_top_non_target_probability", "Dominant non-target function probability"),
        ("full_function_entropy_bits", "Full hard-function entropy"),
        ("full_modal_function_accuracy", "Ensemble modal-function accuracy"),
    )
    for color, fraction, count in zip(
        colors, cfg.train_fractions, cfg.train_counts
    ):
        group = condition_rows(rows, count)
        label = f"{fraction:.0%} (n={count})"
        for axis, (metric, _) in zip(axes.reshape(-1), metrics):
            steps, values = mean_curve(group, metric)
            axis.plot(steps, values, color=color, label=label)
    for axis, (_, title) in zip(axes.reshape(-1), metrics):
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("optimization step")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "time_function_distribution.png", dpi=180)
    plt.close(figure)


def build_summary(
    rows: Sequence[dict[str, object]],
    shapes: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
    current_step: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    by_fraction = {}
    for fraction, count in zip(cfg.train_fractions, cfg.train_counts):
        shape_group = [row for row in shapes if int(row["train_count"]) == count]
        final_group = [
            row for row in rows
            if int(row["train_count"]) == count
            and int(row["step"]) == current_step
        ]
        classifications = {}
        for row in shape_group:
            key = str(row["classification"])
            classifications[key] = classifications.get(key, 0) + 1
        by_fraction[f"{fraction:.2f}"] = {
            "train_count": count,
            "classification_counts": classifications,
            "final_validation_loss_mean": (
                float(np.mean([
                    float(row["validation_raw_cross_entropy"])
                    for row in final_group
                ])) if final_group else None
            ),
            "final_validation_accuracy_mean": (
                float(np.mean([
                    float(row["validation_accuracy"]) for row in final_group
                ])) if final_group else None
            ),
        }
    return {
        "experiment": "mod 97 division MLP grokking replication",
        "task": "x / y (mod 97), x=0..96, y=1..96",
        "total_equations": cfg.prime * (cfg.prime - 1),
        "chance_cross_entropy": math.log(cfg.prime),
        "chance_accuracy": 1.0 / cfg.prime,
        "current_step": current_step,
        "target_max_steps": cfg.max_steps,
        "elapsed_seconds": elapsed_seconds,
        "model_count": len(cfg.train_counts) * len(cfg.model_seeds),
        "by_fraction": by_fraction,
    }


def persist_analysis(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    cfg: RuntimeConfig,
    current_step: int,
    elapsed_seconds: float,
) -> list[dict[str, object]]:
    shapes = analyze_all_curves(rows, cfg)
    write_csv(output_dir / "trajectory.csv", rows)
    write_csv(output_dir / "curve_shape_by_seed.csv", shapes)
    write_json(output_dir / "curve_shape_by_seed.json", shapes)
    write_json(
        output_dir / "summary.json",
        build_summary(rows, shapes, cfg, current_step, elapsed_seconds),
    )
    plot_loss_curves(output_dir, rows, shapes, cfg, log_y=False)
    plot_loss_curves(output_dir, rows, shapes, cfg, log_y=True)
    plot_accuracy_curves(output_dir, rows, cfg)
    plot_middle_phase_diagnostics(output_dir, rows, cfg)
    plot_time_function_distribution(output_dir, rows, cfg)
    return shapes


def config_signature(cfg: RuntimeConfig) -> dict[str, object]:
    return {
        "prime": cfg.prime,
        "train_fractions": cfg.train_fractions,
        "train_counts": cfg.train_counts,
        "data_seed": cfg.data_seed,
        "model_seeds": cfg.model_seeds,
        "hidden_size": cfg.hidden_size,
        "hidden_layer_count": cfg.hidden_layer_count,
        "activation": cfg.activation,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "train_order_seed": cfg.train_order_seed,
        "stable_cross_entropy": cfg.numerically_stable_cross_entropy,
        "matched_loss_levels": cfg.matched_loss_levels,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    rows: Sequence[dict[str, object]],
    function_crossings: dict[str, np.ndarray],
    step: int,
    cfg: RuntimeConfig,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator_state": generator.get_state(),
        "all_rows": list(rows),
        "function_crossings": function_crossings,
        "config_signature": config_signature(cfg),
    }, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    cfg: RuntimeConfig,
    device: torch.device,
) -> tuple[int, list[dict[str, object]], dict[str, np.ndarray]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload["config_signature"] != config_signature(cfg):
        raise ValueError("checkpoint 配置与当前 Config 不一致。")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["generator_state"].cpu())
    return (
        int(payload["step"]),
        list(payload["all_rows"]),
        dict(payload["function_crossings"]),
    )


def package_results(output_dir: Path) -> Path:
    archive = output_dir.parent / f"{output_dir.name}_package.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_dir.iterdir()):
            if path.name in {"latest_checkpoint.pt", "latest_checkpoint.tmp"}:
                continue
            if path.is_file():
                handle.write(path, arcname=path.name)
    return archive


def print_shape_summary(
    shapes: Sequence[dict[str, object]], cfg: RuntimeConfig
) -> None:
    print("[当前曲线形状]")
    for fraction, count in zip(cfg.train_fractions, cfg.train_counts):
        group = [row for row in shapes if int(row["train_count"]) == count]
        labels = [str(row["classification"]) for row in group]
        print(
            f"  {fraction:.0%} (n={count}): "
            f"double={labels.count('double_descent_detected')}/{len(group)} | "
            f"rise-only={labels.count('rise_without_second_descent')}/{len(group)} | "
            f"mean rise={np.mean([float(row['rise_ratio']) for row in group]):.2f}x | "
            f"mean second descent="
            f"{np.mean([float(row['second_descent_ratio']) for row in group]):.2f}x"
        )


def main() -> None:
    cfg = build_runtime_config()
    validate_config(cfg)
    set_global_seed(cfg.data_seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = cfg.result_dir
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    if output_dir.exists() and cfg.overwrite_existing and not cfg.resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_modular_division_dataset(cfg)
    device = torch.device(cfg.device)
    inputs = torch.from_numpy(dataset["inputs"]).to(device)
    targets = torch.from_numpy(dataset["targets"]).long().to(device)
    np.savez_compressed(
        output_dir / "dataset_split.npz",
        ordered_pairs=dataset["pairs"],
        ordered_targets=dataset["targets"],
        permutation=dataset["permutation"],
    )
    write_json(output_dir / "config.json", {
        **asdict(cfg), "result_dir": str(cfg.result_dir)
    })

    expanded_seeds, indices_by_count = make_model_layout(cfg)
    model = BatchedIndependentMLP(
        input_size=cfg.prime * 2,
        output_size=cfg.prime,
        hidden_size=cfg.hidden_size,
        hidden_layer_count=cfg.hidden_layer_count,
        model_seeds=expanded_seeds,
        activation=cfg.activation,
        layernorm_eps=cfg.layernorm_eps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.train_order_seed)

    start_step = 0
    all_rows: list[dict[str, object]] = []
    function_crossings = initialize_function_crossings(
        cfg, len(expanded_seeds), len(inputs)
    )
    if cfg.resume and checkpoint_path.exists():
        start_step, all_rows, function_crossings = load_checkpoint(
            checkpoint_path, model, optimizer, generator, cfg, device
        )
        validate_function_crossings(
            function_crossings, cfg, len(expanded_seeds), len(inputs)
        )
        print(f"从 checkpoint 续训：step={start_step:,}")
    elif (
        (output_dir / "trajectory.jsonl").exists()
        and not cfg.overwrite_existing
        and not cfg.smoke_test
    ):
        raise FileExistsError(
            f"结果目录已存在但没有 checkpoint：{output_dir}\n"
            "确认不需要旧结果后，将 OVERWRITE_EXISTING=True。"
        )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    ) // len(expanded_seeds)
    print("=== Mod 97 division MLP grokking replication ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(
        f"数据：x/y mod {cfg.prime} | total={len(inputs):,} | "
        f"chance CE={math.log(cfg.prime):.6f} | chance acc={1/cfg.prime:.4%}"
    )
    print(
        f"MLP：{cfg.prime * 2} -> {cfg.hidden_size} x "
        f"{cfg.hidden_layer_count} -> {cfg.prime} | 参数={parameter_count:,}"
    )
    print(
        f"训练比例：{list(cfg.train_fractions)} | counts={list(cfg.train_counts)} | "
        f"seeds={list(cfg.model_seeds)} | 并行模型={len(expanded_seeds)}"
    )
    print(
        f"Adam lr={cfg.learning_rate} | warmup={cfg.warmup_steps} | "
        f"weight_decay={cfg.weight_decay} | stable_CE="
        f"{cfg.numerically_stable_cross_entropy} | max_steps={cfg.max_steps:,}"
    )
    print(
        f"matched train CE：{list(cfg.matched_loss_levels)} | "
        "每个 crossing 保存完整 hard function"
    )
    print(f"结果目录：{output_dir}")

    eval_steps = make_eval_steps(cfg)
    evaluated_steps = {int(row["step"]) for row in all_rows}
    trajectory_path = output_dir / "trajectory.jsonl"
    started = time.time()
    if start_step == 0 and 0 not in evaluated_steps:
        rows = evaluate_all_conditions(
            model, cfg, indices_by_count, inputs, targets, 0, 0.0
        )
        new_crossings = record_function_crossings(
            model, cfg, indices_by_count, inputs, targets,
            rows, 0, function_crossings,
        )
        all_rows.extend(rows)
        append_jsonl(trajectory_path, rows)
        evaluated_steps.add(0)
        print(f"matched-loss 新 crossing：{new_crossings}")

    train_counts_tensor = torch.as_tensor(
        cfg.train_counts, dtype=torch.long, device=device
    )
    seed_count = len(cfg.model_seeds)
    model.train()
    latest_losses = None

    current_step = start_step
    interrupted = False
    try:
        for step in range(start_step + 1, cfg.max_steps + 1):
            current_step = step
            warmup_scale = min(step / max(cfg.warmup_steps, 1), 1.0)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = cfg.learning_rate * warmup_scale

            batch_indices = make_training_batch(
                cfg, train_counts_tensor, seed_count, generator
            )
            batch_x = inputs[batch_indices]
            batch_y = targets[batch_indices]
            logits = model(batch_x)
            loss_per_sample = elementwise_cross_entropy(
                logits, batch_y, cfg.numerically_stable_cross_entropy
            )
            loss_per_model = loss_per_sample.mean(dim=1)
            optimizer.zero_grad(set_to_none=True)
            loss_per_model.sum().backward()
            optimizer.step()
            latest_losses = loss_per_model.detach().view(
                len(cfg.train_counts), seed_count
            ).mean(dim=1)

            if step in eval_steps and step not in evaluated_steps:
                rows = evaluate_all_conditions(
                    model, cfg, indices_by_count, inputs, targets,
                    step, time.time() - started,
                )
                new_crossings = record_function_crossings(
                    model, cfg, indices_by_count, inputs, targets,
                    rows, step, function_crossings,
                )
                all_rows.extend(rows)
                append_jsonl(trajectory_path, rows)
                evaluated_steps.add(step)
                print(f"matched-loss 新 crossing：{new_crossings}")

            if step % cfg.log_interval_steps == 0:
                elapsed = time.time() - started
                compact = " | ".join(
                    f"{fraction:.0%}:{latest_losses[index].item():.2e}"
                    for index, fraction in enumerate(cfg.train_fractions)
                )
                print(
                    f"step={step:>7,}/{cfg.max_steps:,} | "
                    f"{(step - start_step) / max(elapsed, 1e-9):.1f} step/s | "
                    f"{compact}"
                )

            if (
                step % cfg.checkpoint_interval_steps == 0
                or step == cfg.max_steps
            ):
                elapsed = time.time() - started
                shapes = persist_analysis(
                    output_dir, all_rows, cfg, step, elapsed
                )
                persist_function_distribution(
                    output_dir, cfg, function_crossings, dataset["targets"]
                )
                save_checkpoint(
                    checkpoint_path, model, optimizer, generator,
                    all_rows, function_crossings, step, cfg,
                )
                print_shape_summary(shapes, cfg)
                print(f"checkpoint：{checkpoint_path}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C，正在做完整评估并安全保存……")
        if current_step not in evaluated_steps:
            rows = evaluate_all_conditions(
                model, cfg, indices_by_count, inputs, targets,
                current_step, time.time() - started,
            )
            record_function_crossings(
                model, cfg, indices_by_count, inputs, targets,
                rows, current_step, function_crossings,
            )
            all_rows.extend(rows)
            append_jsonl(trajectory_path, rows)
            evaluated_steps.add(current_step)
    finally:
        elapsed = time.time() - started
        shapes = persist_analysis(
            output_dir, all_rows, cfg, current_step, elapsed
        )
        function_rows = persist_function_distribution(
            output_dir, cfg, function_crossings, dataset["targets"]
        )
        save_checkpoint(
            checkpoint_path, model, optimizer, generator,
            all_rows, function_crossings, current_step, cfg,
        )
        write_json(output_dir / "function_distribution_summary.json", {
            "last_step": int(current_step),
            "interrupted": bool(interrupted),
            "model_count": len(expanded_seeds),
            "question": (
                "目标完整 hard function 与非目标函数族如何随 matched train CE "
                "及训练样本量发生概率运输？"
            ),
            "distribution_type": "SGD-induced first matched-loss crossing",
            "completed_rows": sum(
                int(row.get("selected_count", 0)) > 0 for row in function_rows
            ),
        })
        print("\n=== 实验已保存 ===")
        print_shape_summary(shapes, cfg)
        print(f"当前 step：{current_step:,}")
        print(f"本次运行耗时：{elapsed / 60:.2f} 分钟")
        print(f"汇总：{output_dir / 'summary.json'}")
        if cfg.package_results:
            print(f"下载压缩包：{package_results(output_dir)}")


if __name__ == "__main__":
    main()
