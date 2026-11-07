"""
隐藏宽度 x 数据量 x 低-loss 的容量压力实验。

核心问题：在同一 ECA 规则、同一嵌套训练集和同一 raw BCE
层级下，更小的过参数化 MLP 是否更容易把完整函数分布压向真实规则；
以及有限预算内的实用泛化阈值是否随宽度增大而上移。

证据分两层：

1. matched-loss 横截面（主证据）：比较各宽度首次穿越相同低 raw BCE
   时的 probe bit/exact accuracy 与多 seed 函数 agreement。
2. practical plateau（辅助证据）：所有模型最多训练固定预算，再用末段
   loss、monitor accuracy 与完整预测稳定性做事后平台判定。

设计边界：

- train exact/raw BCE 只是进入插值区的诊断，不是提前停止条件；
- `0.01/0.003` 是插值边界诊断，核心结论只用 `0.001/0.0003/0.0001`；
- 保留早期 Rule 30 相变实验的 LayerNorm，但不使用 dropout 或 weight decay；
- 参数范数跨宽度不可直接比较，只作轨迹旁证；
- practical plateau 是有限预算定义，不声称无限时间真稳态。

AutoDL 用法：修改 Config 后，将本文件整体复制到 notebook 单元运行。
所有配置都在 Config 内，不读取环境变量。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    # =========================
    # 任务与数据
    # =========================
    # Pilot 先用 Rule 30。正式确认可改成 (30, 110)。
    RULES = (30,)
    CA_STEPS = 1
    BIT_LENGTH = 30
    BOUNDARY_MODE = "circular"

    # 所有训练集严格嵌套，所有宽度使用同一批输入。
    # width=1024 的历史无 dropout/无 weight-decay 相变点约在 900
    # 附近，因此在 640--1152 加密，两端只留边界点。
    TRAIN_COUNTS = (
        256, 384, 512, 640, 704, 768, 832, 896,
        960, 1024, 1152, 1280, 1536, 2048,
    )
    MONITOR_COUNT = 4096
    PROBE_COUNT = 4096
    DATA_SEED = 20260820

    # =========================
    # 容量扫描
    # =========================
    WIDTHS = (128, 256, 512, 1024, 2048)
    HIDDEN_LAYER_COUNT = 3
    ACTIVATION = "gelu"
    NORMALIZATION = "layernorm"  # "none" 或 "layernorm"
    LAYERNORM_EPS = 1e-5
    MODEL_SEEDS = (0, 1, 2)

    # =========================
    # 训练
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OPTIMIZER = "adamw"  # "adamw" 或 "sgd_momentum"
    LEARNING_RATE = 1e-3
    SGD_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 512
    MAX_STEPS = 20_000
    TRAIN_ORDER_SEED = 314159

    # 高 loss 不用于当前核心命题。0.01/0.003 只是插值边界诊断。
    RAW_BCE_LEVELS = (0.01, 0.003, 0.001, 0.0003, 0.0001)
    PRIMARY_RAW_BCE_LEVELS = (0.001, 0.0003, 0.0001)

    FIXED_EVAL_STEPS = (
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000, 5000, 10_000, 15_000, 20_000,
    )
    DENSE_EVAL_UNTIL_STEP = 1000
    DENSE_EVAL_INTERVAL_STEPS = 25
    EVAL_INTERVAL_STEPS = 100
    LOG_INTERVAL_STEPS = 500
    EVAL_CHUNK_SIZE = 512

    # =========================
    # practical plateau（事后判定）
    # =========================
    PLATEAU_WINDOW_STEPS = 5000
    PLATEAU_MIN_EVALS = 10
    PLATEAU_MONITOR_BIT_RANGE = 0.001
    PLATEAU_MONITOR_EXACT_RANGE = 0.02
    PLATEAU_PREDICTION_STABILITY_THRESHOLD = 0.999
    PLATEAU_REQUIRED_STABLE_FRACTION = 0.80
    PLATEAU_LOSS_RELATIVE_DROP_MAX = 0.20
    PLATEAU_LOSS_FLOOR = 1e-7
    ENDPOINT_MEDIAN_EVALS = 5

    # 相变点操作定义。
    GENERALIZATION_EXACT_THRESHOLD = 0.99
    GENERALIZATION_REQUIRED_SEED_FRACTION = 0.80
    TRAIN_EXACT_TOLERANCE = 1e-12

    # =========================
    # 输出
    # =========================
    RESULT_DIR = Path("/root/results_capacity_pressure_width_scan")
    PACKAGE_RESULTS = True
    SAVE_FINAL_WEIGHTS = False
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass(frozen=True)
class RuntimeConfig:
    rules: tuple[int, ...]
    ca_steps: int
    bit_length: int
    boundary_mode: str
    train_counts: tuple[int, ...]
    monitor_count: int
    probe_count: int
    data_seed: int
    widths: tuple[int, ...]
    hidden_layer_count: int
    activation: str
    normalization: str
    layernorm_eps: float
    model_seeds: tuple[int, ...]
    device: str
    optimizer: str
    learning_rate: float
    sgd_momentum: float
    weight_decay: float
    batch_size: int
    max_steps: int
    train_order_seed: int
    raw_bce_levels: tuple[float, ...]
    primary_raw_bce_levels: tuple[float, ...]
    fixed_eval_steps: tuple[int, ...]
    dense_eval_until_step: int
    dense_eval_interval_steps: int
    eval_interval_steps: int
    log_interval_steps: int
    eval_chunk_size: int
    plateau_window_steps: int
    plateau_min_evals: int
    plateau_monitor_bit_range: float
    plateau_monitor_exact_range: float
    plateau_prediction_stability_threshold: float
    plateau_required_stable_fraction: float
    plateau_loss_relative_drop_max: float
    plateau_loss_floor: float
    endpoint_median_evals: int
    generalization_exact_threshold: float
    generalization_required_seed_fraction: float
    train_exact_tolerance: float
    result_dir: Path
    package_results: bool
    save_final_weights: bool
    overwrite_result_dir: bool
    smoke_test: bool


@dataclass(frozen=True)
class Condition:
    index: int
    rule: int
    train_count: int
    name: str


def build_runtime_config() -> RuntimeConfig:
    if Config.SMOKE_TEST:
        return RuntimeConfig(
            rules=(30,),
            ca_steps=1,
            bit_length=10,
            boundary_mode="circular",
            train_counts=(16, 32),
            monitor_count=64,
            probe_count=128,
            data_seed=Config.DATA_SEED,
            widths=(16, 32),
            hidden_layer_count=2,
            activation="gelu",
            normalization="layernorm",
            layernorm_eps=Config.LAYERNORM_EPS,
            model_seeds=(0, 1),
            device=Config.DEVICE,
            optimizer="adamw",
            learning_rate=2e-3,
            sgd_momentum=Config.SGD_MOMENTUM,
            weight_decay=0.0,
            batch_size=32,
            max_steps=1000,
            train_order_seed=Config.TRAIN_ORDER_SEED,
            raw_bce_levels=(0.1, 0.03),
            primary_raw_bce_levels=(0.03,),
            fixed_eval_steps=(
                0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 750, 1000
            ),
            dense_eval_until_step=200,
            dense_eval_interval_steps=10,
            eval_interval_steps=25,
            log_interval_steps=100,
            eval_chunk_size=64,
            plateau_window_steps=200,
            plateau_min_evals=4,
            plateau_monitor_bit_range=0.05,
            plateau_monitor_exact_range=0.10,
            plateau_prediction_stability_threshold=0.95,
            plateau_required_stable_fraction=0.50,
            plateau_loss_relative_drop_max=0.50,
            plateau_loss_floor=1e-4,
            endpoint_median_evals=3,
            generalization_exact_threshold=0.90,
            generalization_required_seed_fraction=0.50,
            train_exact_tolerance=Config.TRAIN_EXACT_TOLERANCE,
            result_dir=Path(
                "research/ca_phase_transition/"
                "_smoke_capacity_pressure_width_scan"
            ),
            package_results=True,
            save_final_weights=False,
            overwrite_result_dir=True,
            smoke_test=True,
        )

    return RuntimeConfig(
        rules=tuple(int(value) for value in Config.RULES),
        ca_steps=int(Config.CA_STEPS),
        bit_length=int(Config.BIT_LENGTH),
        boundary_mode=str(Config.BOUNDARY_MODE),
        train_counts=tuple(int(value) for value in Config.TRAIN_COUNTS),
        monitor_count=int(Config.MONITOR_COUNT),
        probe_count=int(Config.PROBE_COUNT),
        data_seed=int(Config.DATA_SEED),
        widths=tuple(int(value) for value in Config.WIDTHS),
        hidden_layer_count=int(Config.HIDDEN_LAYER_COUNT),
        activation=str(Config.ACTIVATION),
        normalization=str(Config.NORMALIZATION),
        layernorm_eps=float(Config.LAYERNORM_EPS),
        model_seeds=tuple(int(value) for value in Config.MODEL_SEEDS),
        device=str(Config.DEVICE),
        optimizer=str(Config.OPTIMIZER),
        learning_rate=float(Config.LEARNING_RATE),
        sgd_momentum=float(Config.SGD_MOMENTUM),
        weight_decay=float(Config.WEIGHT_DECAY),
        batch_size=int(Config.BATCH_SIZE),
        max_steps=int(Config.MAX_STEPS),
        train_order_seed=int(Config.TRAIN_ORDER_SEED),
        raw_bce_levels=tuple(float(value) for value in Config.RAW_BCE_LEVELS),
        primary_raw_bce_levels=tuple(
            float(value) for value in Config.PRIMARY_RAW_BCE_LEVELS
        ),
        fixed_eval_steps=tuple(int(value) for value in Config.FIXED_EVAL_STEPS),
        dense_eval_until_step=int(Config.DENSE_EVAL_UNTIL_STEP),
        dense_eval_interval_steps=int(Config.DENSE_EVAL_INTERVAL_STEPS),
        eval_interval_steps=int(Config.EVAL_INTERVAL_STEPS),
        log_interval_steps=int(Config.LOG_INTERVAL_STEPS),
        eval_chunk_size=int(Config.EVAL_CHUNK_SIZE),
        plateau_window_steps=int(Config.PLATEAU_WINDOW_STEPS),
        plateau_min_evals=int(Config.PLATEAU_MIN_EVALS),
        plateau_monitor_bit_range=float(Config.PLATEAU_MONITOR_BIT_RANGE),
        plateau_monitor_exact_range=float(Config.PLATEAU_MONITOR_EXACT_RANGE),
        plateau_prediction_stability_threshold=float(
            Config.PLATEAU_PREDICTION_STABILITY_THRESHOLD
        ),
        plateau_required_stable_fraction=float(
            Config.PLATEAU_REQUIRED_STABLE_FRACTION
        ),
        plateau_loss_relative_drop_max=float(
            Config.PLATEAU_LOSS_RELATIVE_DROP_MAX
        ),
        plateau_loss_floor=float(Config.PLATEAU_LOSS_FLOOR),
        endpoint_median_evals=int(Config.ENDPOINT_MEDIAN_EVALS),
        generalization_exact_threshold=float(
            Config.GENERALIZATION_EXACT_THRESHOLD
        ),
        generalization_required_seed_fraction=float(
            Config.GENERALIZATION_REQUIRED_SEED_FRACTION
        ),
        train_exact_tolerance=float(Config.TRAIN_EXACT_TOLERANCE),
        result_dir=Path(Config.RESULT_DIR),
        package_results=bool(Config.PACKAGE_RESULTS),
        save_final_weights=bool(Config.SAVE_FINAL_WEIGHTS),
        overwrite_result_dir=bool(Config.OVERWRITE_RESULT_DIR),
        smoke_test=False,
    )


def validate_config(cfg: RuntimeConfig) -> None:
    if cfg.boundary_mode != "circular":
        raise ValueError("当前实验只实现 circular 边界。")
    if not cfg.rules or not cfg.widths or not cfg.train_counts or not cfg.model_seeds:
        raise ValueError("规则、宽度、数据量和 seeds 都不能为空。")
    if tuple(sorted(set(cfg.widths))) != cfg.widths:
        raise ValueError("WIDTHS 必须严格递增且不重复。")
    if tuple(sorted(set(cfg.train_counts))) != cfg.train_counts:
        raise ValueError("TRAIN_COUNTS 必须严格递增且不重复。")
    if min(cfg.widths) < 1 or min(cfg.train_counts) < 2:
        raise ValueError("宽度或训练样本数非法。")
    if cfg.hidden_layer_count < 1:
        raise ValueError("HIDDEN_LAYER_COUNT 必须至少为 1。")
    if cfg.activation not in {"gelu", "relu", "tanh"}:
        raise ValueError("ACTIVATION 只能是 gelu/relu/tanh。")
    if cfg.normalization not in {"none", "layernorm"}:
        raise ValueError("NORMALIZATION 只能是 none 或 layernorm。")
    if cfg.optimizer not in {"adamw", "sgd_momentum"}:
        raise ValueError("OPTIMIZER 只能是 adamw 或 sgd_momentum。")
    if cfg.weight_decay != 0.0:
        raise ValueError("本实验要求 WEIGHT_DECAY=0。")
    if cfg.max_steps < 1 or cfg.batch_size < 1:
        raise ValueError("MAX_STEPS/BATCH_SIZE 必须为正数。")
    if sorted(cfg.raw_bce_levels, reverse=True) != list(cfg.raw_bce_levels):
        raise ValueError("RAW_BCE_LEVELS 必须从高到低排列。")
    if not set(cfg.primary_raw_bce_levels).issubset(cfg.raw_bce_levels):
        raise ValueError("PRIMARY_RAW_BCE_LEVELS 必须是 RAW_BCE_LEVELS 的子集。")
    if cfg.bit_length >= 63:
        raise ValueError("当前输入 ID 采样要求 BIT_LENGTH < 63。")
    required = max(cfg.train_counts) + cfg.monitor_count + cfg.probe_count
    if required >= 2 ** cfg.bit_length:
        raise ValueError("训练/monitor/probe 超过可用输入空间。")
    for rule in cfg.rules:
        if not 0 <= rule <= 255:
            raise ValueError(f"ECA 规则越界：{rule}")
    for value, name in (
        (cfg.plateau_required_stable_fraction, "PLATEAU_REQUIRED_STABLE_FRACTION"),
        (cfg.generalization_exact_threshold, "GENERALIZATION_EXACT_THRESHOLD"),
        (
            cfg.generalization_required_seed_fraction,
            "GENERALIZATION_REQUIRED_SEED_FRACTION",
        ),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} 必须在 (0, 1] 内。")


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_ready(row), ensure_ascii=False) + "\n"
            )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
                for key, value in row.items()
            })


def prepare_output_dir(cfg: RuntimeConfig) -> Path:
    output_dir = cfg.result_dir
    if output_dir.exists():
        if cfg.overwrite_result_dir:
            shutil.rmtree(output_dir)
        else:
            suffix = time.strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.parent / f"{output_dir.name}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sample_unique_ids(
    count: int,
    bit_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    upper = 2 ** bit_length
    selected: set[int] = set()
    while len(selected) < count:
        need = count - len(selected)
        candidates = rng.integers(
            0,
            upper,
            size=max(need * 2, 1024),
            dtype=np.int64,
        )
        selected.update(int(value) for value in candidates)
    values = np.fromiter(selected, dtype=np.int64, count=len(selected))
    rng.shuffle(values)
    return values[:count]


def integer_ids_to_bits(ids: np.ndarray, bit_length: int) -> np.ndarray:
    shifts = np.arange(bit_length - 1, -1, -1, dtype=np.int64)
    return ((ids[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def evolve_elementary_ca(
    states: np.ndarray,
    rule: int,
    steps: int,
) -> np.ndarray:
    current = states.astype(np.int64, copy=True)
    for _ in range(steps):
        left = np.roll(current, 1, axis=1)
        center = current
        right = np.roll(current, -1, axis=1)
        pattern = (left << 2) | (center << 1) | right
        current = ((rule >> pattern) & 1).astype(np.int64)
    return current.astype(np.float32)


def build_conditions(cfg: RuntimeConfig) -> list[Condition]:
    conditions: list[Condition] = []
    for rule in cfg.rules:
        for train_count in cfg.train_counts:
            conditions.append(Condition(
                index=len(conditions),
                rule=rule,
                train_count=train_count,
                name=f"rule{rule}_n{train_count}",
            ))
    return conditions


def make_dataset(
    cfg: RuntimeConfig,
    conditions: Sequence[Condition],
) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.data_seed)
    total = max(cfg.train_counts) + cfg.monitor_count + cfg.probe_count
    ids = sample_unique_ids(total, cfg.bit_length, rng)
    max_train = max(cfg.train_counts)
    monitor_end = max_train + cfg.monitor_count
    train_ids = ids[:max_train]
    monitor_ids = ids[max_train:monitor_end]
    probe_ids = ids[monitor_end:]
    train_x = integer_ids_to_bits(train_ids, cfg.bit_length)
    monitor_x = integer_ids_to_bits(monitor_ids, cfg.bit_length)
    probe_x = integer_ids_to_bits(probe_ids, cfg.bit_length)

    train_targets = {
        rule: evolve_elementary_ca(train_x, rule, cfg.ca_steps)
        for rule in cfg.rules
    }
    monitor_targets = {
        rule: evolve_elementary_ca(monitor_x, rule, cfg.ca_steps)
        for rule in cfg.rules
    }
    probe_targets = {
        rule: evolve_elementary_ca(probe_x, rule, cfg.ca_steps)
        for rule in cfg.rules
    }
    return {
        "train_ids": train_ids,
        "monitor_ids": monitor_ids,
        "probe_ids": probe_ids,
        "train_x": train_x,
        "monitor_x": monitor_x,
        "probe_x": probe_x,
        "condition_train_targets": np.stack([
            train_targets[condition.rule] for condition in conditions
        ]).astype(np.float32),
        "condition_monitor_targets": np.stack([
            monitor_targets[condition.rule] for condition in conditions
        ]).astype(np.float32),
        "condition_probe_targets": np.stack([
            probe_targets[condition.rule] for condition in conditions
        ]).astype(np.float32),
        "probe_targets_by_rule": probe_targets,
    }


class BatchedIndependentMLP(nn.Module):
    """参数完全独立、但在 model 维使用 bmm 并行的 MLP。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layer_count: int,
        model_seeds: Sequence[int],
        activation: str,
        normalization: str,
        layernorm_eps: float,
    ) -> None:
        super().__init__()
        self.model_count = len(model_seeds)
        self.hidden_layer_count = hidden_layer_count
        self.activation_name = activation
        self.normalization = normalization
        self.layernorm_eps = layernorm_eps
        dimensions = [input_size]
        dimensions.extend([hidden_size] * hidden_layer_count)
        dimensions.append(output_size)

        weights_by_model: list[list[torch.Tensor]] = []
        biases_by_model: list[list[torch.Tensor]] = []
        for seed in model_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                model_weights: list[torch.Tensor] = []
                model_biases: list[torch.Tensor] = []
                for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
                    layer = nn.Linear(input_dim, output_dim)
                    model_weights.append(layer.weight.detach().clone())
                    model_biases.append(layer.bias.detach().clone())
                weights_by_model.append(model_weights)
                biases_by_model.append(model_biases)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model_weights[layer_index]
                for model_weights in weights_by_model
            ]))
            for layer_index in range(len(dimensions) - 1)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model_biases[layer_index]
                for model_biases in biases_by_model
            ]))
            for layer_index in range(len(dimensions) - 1)
        ])
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()
        if normalization == "layernorm":
            for _ in range(hidden_layer_count):
                self.norm_weights.append(nn.Parameter(torch.ones(
                    self.model_count, hidden_size
                )))
                self.norm_biases.append(nn.Parameter(torch.zeros(
                    self.model_count, hidden_size
                )))

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(values)
        if self.activation_name == "relu":
            return F.relu(values)
        if self.activation_name == "tanh":
            return torch.tanh(values)
        raise ValueError(f"未知激活函数：{self.activation_name}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if layer_index < len(self.weights) - 1:
                hidden = self.activate(hidden)
                if self.normalization == "layernorm":
                    mean = hidden.mean(dim=-1, keepdim=True)
                    variance = hidden.var(
                        dim=-1, unbiased=False, keepdim=True
                    )
                    hidden = (hidden - mean) * torch.rsqrt(
                        variance + self.layernorm_eps
                    )
                    hidden = (
                        hidden * self.norm_weights[layer_index][:, None, :]
                        + self.norm_biases[layer_index][:, None, :]
                    )
        return hidden

    def named_batched_parameters(self) -> list[tuple[str, torch.Tensor]]:
        rows: list[tuple[str, torch.Tensor]] = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            label = "output" if index == len(self.weights) - 1 else f"hidden_{index + 1}"
            rows.append((f"{label}_weight", weight))
            rows.append((f"{label}_bias", bias))
            if index < self.hidden_layer_count and self.normalization == "layernorm":
                rows.append((
                    f"{label}_norm_weight", self.norm_weights[index]
                ))
                rows.append((
                    f"{label}_norm_bias", self.norm_biases[index]
                ))
        return rows


def make_model_layout(
    conditions: Sequence[Condition],
    seeds: Sequence[int],
) -> tuple[list[int], list[int], list[int], list[str]]:
    condition_indices: list[int] = []
    expanded_seeds: list[int] = []
    train_counts: list[int] = []
    names: list[str] = []
    for condition in conditions:
        for seed in seeds:
            condition_indices.append(condition.index)
            expanded_seeds.append(int(seed))
            train_counts.append(condition.train_count)
            names.append(f"{condition.name}_seed{seed}")
    return condition_indices, expanded_seeds, train_counts, names


def assert_paired_initialization(
    model: BatchedIndependentMLP,
    conditions: Sequence[Condition],
    seeds: Sequence[int],
) -> None:
    seed_count = len(seeds)
    for seed_index, seed in enumerate(seeds):
        for condition_index in range(1, len(conditions)):
            candidate_index = condition_index * seed_count + seed_index
            for name, parameter in model.named_batched_parameters():
                if not torch.equal(parameter[seed_index], parameter[candidate_index]):
                    raise RuntimeError(
                        "配对初始化失败："
                        f"seed={seed}, condition={condition_index}, block={name}"
                    )


def snapshot_initial_parameters(
    model: BatchedIndependentMLP,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_batched_parameters()
    }


@torch.no_grad()
def parameter_metrics(
    model: BatchedIndependentMLP,
    initial: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    count = model.model_count
    total_sq = torch.zeros(count, device=model.weights[0].device)
    weight_sq = torch.zeros_like(total_sq)
    delta_sq = torch.zeros_like(total_sq)
    delta_weight_sq = torch.zeros_like(total_sq)
    log_product = torch.zeros_like(total_sq)
    for name, parameter in model.named_batched_parameters():
        flat = parameter.reshape(count, -1)
        block_sq = flat.square().sum(dim=1)
        total_sq += block_sq
        delta = parameter - initial[name]
        delta_block_sq = delta.reshape(count, -1).square().sum(dim=1)
        delta_sq += delta_block_sq
        if name.endswith("_weight") and parameter.ndim == 3:
            weight_sq += block_sq
            delta_weight_sq += delta_block_sq
            log_product += torch.log(block_sq.sqrt().clamp_min(1e-30))
    return {
        "parameter_norm": total_sq.sqrt().cpu().numpy(),
        "weight_norm": weight_sq.sqrt().cpu().numpy(),
        "parameter_delta_norm": delta_sq.sqrt().cpu().numpy(),
        "weight_delta_norm": delta_weight_sq.sqrt().cpu().numpy(),
        "log_frobenius_weight_product": log_product.cpu().numpy(),
    }


@torch.no_grad()
def evaluate_models(
    model: BatchedIndependentMLP,
    inputs: torch.Tensor,
    targets_by_model: torch.Tensor,
    chunk_size: int,
    sample_counts: torch.Tensor | None = None,
    return_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    model_count = model.model_count
    sample_count = len(inputs)
    output_bits = targets_by_model.shape[-1]
    device = inputs.device
    loss_sum = torch.zeros(model_count, device=device, dtype=torch.float64)
    bit_correct = torch.zeros_like(loss_sum)
    exact_correct = torch.zeros_like(loss_sum)
    valid_sample_total = torch.zeros_like(loss_sum)
    prediction_chunks: list[np.ndarray] = []

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        shared_x = inputs[start:end]
        batch_x = shared_x.unsqueeze(0).expand(model_count, -1, -1)
        logits = model(batch_x)
        targets = targets_by_model[:, start:end]
        predictions = logits >= 0
        label_bool = targets >= 0.5
        per_sample_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).mean(dim=-1)

        if sample_counts is None:
            mask = torch.ones(
                model_count, end - start, device=device, dtype=torch.bool
            )
        else:
            positions = torch.arange(start, end, device=device)[None, :]
            mask = positions < sample_counts[:, None]

        mask_float = mask.to(torch.float64)
        loss_sum += (per_sample_loss.to(torch.float64) * mask_float).sum(dim=1)
        bit_correct += (
            (predictions == label_bool) & mask[:, :, None]
        ).sum(dim=(1, 2)).to(torch.float64)
        exact_correct += (
            torch.all(predictions == label_bool, dim=-1) & mask
        ).sum(dim=1).to(torch.float64)
        valid_sample_total += mask_float.sum(dim=1)

        if return_predictions:
            prediction_chunks.append(
                predictions.detach().cpu().numpy().astype(np.uint8)
            )

    denominator = valid_sample_total.clamp_min(1.0)
    result: dict[str, Any] = {
        "raw_bce": (loss_sum / denominator).cpu().numpy(),
        "bit_accuracy": (
            bit_correct / (denominator * output_bits)
        ).cpu().numpy(),
        "exact_accuracy": (exact_correct / denominator).cpu().numpy(),
    }
    if return_predictions:
        result["predictions"] = np.concatenate(prediction_chunks, axis=1)
    return result


def prediction_stability(
    before: np.ndarray | None,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_count = after.shape[0]
    if before is None:
        nan = np.full(model_count, np.nan, dtype=np.float64)
        return nan.copy(), nan.copy()
    equal = before == after
    return (
        equal.mean(axis=(1, 2), dtype=np.float64),
        equal.all(axis=2).mean(axis=1, dtype=np.float64),
    )


def build_current_rows(
    width: int,
    step: int,
    conditions: Sequence[Condition],
    condition_indices: Sequence[int],
    seeds: Sequence[int],
    train_counts: Sequence[int],
    fit_steps: Sequence[int | None],
    train_metrics: dict[str, Any],
    monitor_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    norm_metrics: dict[str, np.ndarray],
    monitor_bit_stability: np.ndarray,
    monitor_exact_stability: np.ndarray,
    probe_bit_stability: np.ndarray,
    probe_exact_stability: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_index, condition_index in enumerate(condition_indices):
        condition = conditions[condition_index]
        row: dict[str, Any] = {
            "width": width,
            "step": step,
            "model_index_within_width": model_index,
            "condition_index": condition.index,
            "condition": condition.name,
            "rule": condition.rule,
            "train_count": train_counts[model_index],
            "model_seed": seeds[model_index],
            "train_fit_step": fit_steps[model_index],
            "train_raw_bce": float(train_metrics["raw_bce"][model_index]),
            "train_bit_accuracy": float(
                train_metrics["bit_accuracy"][model_index]
            ),
            "train_exact_accuracy": float(
                train_metrics["exact_accuracy"][model_index]
            ),
            "monitor_raw_bce": float(monitor_metrics["raw_bce"][model_index]),
            "monitor_bit_accuracy": float(
                monitor_metrics["bit_accuracy"][model_index]
            ),
            "monitor_exact_accuracy": float(
                monitor_metrics["exact_accuracy"][model_index]
            ),
            "probe_raw_bce": float(probe_metrics["raw_bce"][model_index]),
            "probe_bit_accuracy": float(
                probe_metrics["bit_accuracy"][model_index]
            ),
            "probe_exact_accuracy": float(
                probe_metrics["exact_accuracy"][model_index]
            ),
            "monitor_prediction_bit_stability": float(
                monitor_bit_stability[model_index]
            ),
            "monitor_prediction_exact_stability": float(
                monitor_exact_stability[model_index]
            ),
            "probe_prediction_bit_stability": float(
                probe_bit_stability[model_index]
            ),
            "probe_prediction_exact_stability": float(
                probe_exact_stability[model_index]
            ),
        }
        for key, values in norm_metrics.items():
            row[key] = float(values[model_index])
        rows.append(row)
    return rows


def interpolation_fraction(before_loss: float, after_loss: float, target: float) -> float:
    before_log = math.log(max(before_loss, 1e-300))
    after_log = math.log(max(after_loss, 1e-300))
    target_log = math.log(max(target, 1e-300))
    denominator = before_log - after_log
    if abs(denominator) < 1e-15:
        return 1.0
    return float(np.clip((before_log - target_log) / denominator, 0.0, 1.0))


def make_crossing_row(
    target_loss: float,
    before: dict[str, Any],
    after: dict[str, Any],
    batch_size: int,
    prediction_source: str,
    prediction_observed_loss: float,
    prediction_index: int,
    train_exact_tolerance: float,
) -> dict[str, Any]:
    fraction = interpolation_fraction(
        float(before["train_raw_bce"]),
        float(after["train_raw_bce"]),
        target_loss,
    )
    row: dict[str, Any] = {
        key: after[key]
        for key in (
            "width",
            "model_index_within_width",
            "condition_index",
            "condition",
            "rule",
            "train_count",
            "model_seed",
            "train_fit_step",
        )
    }
    row.update({
        "target_raw_bce": target_loss,
        "crossing_step_before": int(before["step"]),
        "crossing_step_after": int(after["step"]),
        "observed_raw_bce_before": float(before["train_raw_bce"]),
        "observed_raw_bce_after": float(after["train_raw_bce"]),
        "crossing_interpolation_fraction": fraction,
        "prediction_checkpoint_source": prediction_source,
        "prediction_observed_raw_bce": prediction_observed_loss,
        "crossing_prediction_index": prediction_index,
        "train_exact_before": float(before["train_exact_accuracy"]),
        "train_exact_after": float(after["train_exact_accuracy"]),
        "valid_after_train_fit": bool(
            float(after["train_exact_accuracy"])
            >= 1.0 - train_exact_tolerance
        ),
        "valid_strict_post_interpolation": bool(
            float(before["train_exact_accuracy"])
            >= 1.0 - train_exact_tolerance
            and float(after["train_exact_accuracy"])
            >= 1.0 - train_exact_tolerance
        ),
    })
    identity_keys = {
        "width", "step", "model_index_within_width", "condition_index",
        "condition", "rule", "train_count", "model_seed", "train_fit_step",
    }
    for key, after_value in after.items():
        if key in identity_keys or not isinstance(after_value, (int, float)):
            continue
        before_value = before.get(key)
        if not isinstance(before_value, (int, float)):
            continue
        before_float = float(before_value)
        after_float = float(after_value)
        if not (math.isfinite(before_float) and math.isfinite(after_float)):
            continue
        row[f"matched_{key}"] = before_float + fraction * (
            after_float - before_float
        )
    row["matched_train_raw_bce"] = target_loss
    row["matched_step_to_loss"] = float(before["step"]) + fraction * (
        float(after["step"]) - float(before["step"])
    )
    row["matched_examples_seen_to_loss"] = (
        row["matched_step_to_loss"] * batch_size
    )
    return row


def make_optimizer(
    cfg: RuntimeConfig,
    model: nn.Module,
) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=0.0,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.sgd_momentum,
        weight_decay=0.0,
    )


def should_evaluate(step: int, cfg: RuntimeConfig) -> bool:
    if step in set(cfg.fixed_eval_steps) or step == cfg.max_steps:
        return True
    if (
        step <= cfg.dense_eval_until_step
        and step % cfg.dense_eval_interval_steps == 0
    ):
        return True
    return step % cfg.eval_interval_steps == 0


def run_width(
    width: int,
    cfg: RuntimeConfig,
    conditions: Sequence[Condition],
    dataset: dict[str, Any],
    crossing_prediction_offset: int,
) -> dict[str, Any]:
    device = torch.device(cfg.device)
    (
        condition_indices,
        expanded_seeds,
        model_train_counts,
        model_names,
    ) = make_model_layout(conditions, cfg.model_seeds)
    model_count = len(condition_indices)
    condition_index_tensor = torch.tensor(
        condition_indices, dtype=torch.long, device=device
    )
    train_count_tensor = torch.tensor(
        model_train_counts, dtype=torch.long, device=device
    )

    train_x = torch.from_numpy(dataset["train_x"]).to(device)
    monitor_x = torch.from_numpy(dataset["monitor_x"]).to(device)
    probe_x = torch.from_numpy(dataset["probe_x"]).to(device)
    condition_train_targets = torch.from_numpy(
        dataset["condition_train_targets"]
    ).to(device)
    condition_monitor_targets = torch.from_numpy(
        dataset["condition_monitor_targets"]
    ).to(device)
    condition_probe_targets = torch.from_numpy(
        dataset["condition_probe_targets"]
    ).to(device)
    model_train_targets = condition_train_targets.index_select(
        0, condition_index_tensor
    )
    model_monitor_targets = condition_monitor_targets.index_select(
        0, condition_index_tensor
    )
    model_probe_targets = condition_probe_targets.index_select(
        0, condition_index_tensor
    )

    model = BatchedIndependentMLP(
        input_size=cfg.bit_length,
        output_size=cfg.bit_length,
        hidden_size=width,
        hidden_layer_count=cfg.hidden_layer_count,
        model_seeds=expanded_seeds,
        activation=cfg.activation,
        normalization=cfg.normalization,
        layernorm_eps=cfg.layernorm_eps,
    ).to(device)
    assert_paired_initialization(model, conditions, cfg.model_seeds)
    initial = snapshot_initial_parameters(model)
    optimizer = make_optimizer(cfg, model)
    per_model_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    ) // model_count

    print(
        f"\n=== width={width:,} | models={model_count} | "
        f"parameters/model={per_model_parameters:,} ===",
        flush=True,
    )
    pending_levels = [set(cfg.raw_bce_levels) for _ in range(model_count)]
    fit_steps: list[int | None] = [None] * model_count
    previous_rows: list[dict[str, Any]] | None = None
    previous_monitor_predictions: np.ndarray | None = None
    previous_probe_predictions: np.ndarray | None = None
    trajectory_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    crossing_predictions: list[np.ndarray] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.train_order_seed)
    started = time.time()

    def diagnostics(step: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
        nonlocal previous_rows
        nonlocal previous_monitor_predictions
        nonlocal previous_probe_predictions

        train_metrics = evaluate_models(
            model,
            train_x,
            model_train_targets,
            cfg.eval_chunk_size,
            sample_counts=train_count_tensor,
            return_predictions=False,
        )
        monitor_metrics = evaluate_models(
            model,
            monitor_x,
            model_monitor_targets,
            cfg.eval_chunk_size,
            return_predictions=True,
        )
        probe_metrics = evaluate_models(
            model,
            probe_x,
            model_probe_targets,
            cfg.eval_chunk_size,
            return_predictions=True,
        )
        monitor_predictions = monitor_metrics.pop("predictions")
        probe_predictions = probe_metrics.pop("predictions")
        monitor_bit_stability, monitor_exact_stability = prediction_stability(
            previous_monitor_predictions, monitor_predictions
        )
        probe_bit_stability, probe_exact_stability = prediction_stability(
            previous_probe_predictions, probe_predictions
        )

        for model_index in range(model_count):
            if (
                fit_steps[model_index] is None
                and float(train_metrics["exact_accuracy"][model_index])
                >= 1.0 - cfg.train_exact_tolerance
            ):
                fit_steps[model_index] = step

        norm_metrics = parameter_metrics(model, initial)
        current_rows = build_current_rows(
            width,
            step,
            conditions,
            condition_indices,
            expanded_seeds,
            model_train_counts,
            fit_steps,
            train_metrics,
            monitor_metrics,
            probe_metrics,
            norm_metrics,
            monitor_bit_stability,
            monitor_exact_stability,
            probe_bit_stability,
            probe_exact_stability,
        )
        trajectory_rows.extend(current_rows)

        if previous_rows is not None and previous_probe_predictions is not None:
            for model_index, current in enumerate(current_rows):
                before = previous_rows[model_index]
                before_loss = float(before["train_raw_bce"])
                after_loss = float(current["train_raw_bce"])
                crossed = [
                    level
                    for level in pending_levels[model_index]
                    if before_loss > level >= after_loss
                ]
                for level in sorted(crossed, reverse=True):
                    before_distance = abs(
                        math.log(max(before_loss, 1e-300)) - math.log(level)
                    )
                    after_distance = abs(
                        math.log(max(after_loss, 1e-300)) - math.log(level)
                    )
                    if before_distance <= after_distance:
                        prediction = previous_probe_predictions[model_index].copy()
                        source = "before"
                        observed_loss = before_loss
                    else:
                        prediction = probe_predictions[model_index].copy()
                        source = "after"
                        observed_loss = after_loss
                    prediction_index = (
                        crossing_prediction_offset + len(crossing_predictions)
                    )
                    crossing_predictions.append(prediction)
                    crossing_rows.append(make_crossing_row(
                        level,
                        before,
                        current,
                        cfg.batch_size,
                        source,
                        observed_loss,
                        prediction_index,
                        cfg.train_exact_tolerance,
                    ))
                    pending_levels[model_index].remove(level)

        previous_rows = [dict(row) for row in current_rows]
        previous_monitor_predictions = monitor_predictions.copy()
        previous_probe_predictions = probe_predictions.copy()
        return current_rows, monitor_predictions, probe_predictions

    current_rows, _, final_probe_predictions = diagnostics(0)
    for step in range(1, cfg.max_steps + 1):
        indices_by_count = {
            train_count: torch.randint(
                0,
                train_count,
                size=(cfg.batch_size,),
                generator=generator,
                device=device,
            )
            for train_count in cfg.train_counts
        }
        batch_indices = torch.stack([
            indices_by_count[int(train_count)]
            for train_count in model_train_counts
        ])
        batch_x = train_x[batch_indices]
        batch_y = torch.gather(
            model_train_targets,
            1,
            batch_indices[:, :, None].expand(-1, -1, cfg.bit_length),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        per_model_loss = F.binary_cross_entropy_with_logits(
            logits, batch_y, reduction="none"
        ).mean(dim=(1, 2))
        per_model_loss.sum().backward()
        optimizer.step()

        if not should_evaluate(step, cfg):
            continue
        current_rows, _, final_probe_predictions = diagnostics(step)
        if step % cfg.log_interval_steps == 0 or step in cfg.fixed_eval_steps:
            elapsed = time.time() - started
            print(
                f"width={width:>4,} | step={step:>6,} | "
                f"elapsed={elapsed:>7.1f}s | "
                f"train_loss={np.mean([row['train_raw_bce'] for row in current_rows]):.6g} | "
                f"probe_exact={np.mean([row['probe_exact_accuracy'] for row in current_rows]):.3%} | "
                f"crossings={len(crossing_rows):,}",
                flush=True,
            )

    final_rows = [dict(row) for row in current_rows]
    final_prediction_records: list[dict[str, Any]] = []
    for model_index, row in enumerate(final_rows):
        final_prediction_records.append({
            "width": width,
            "rule": int(row["rule"]),
            "train_count": int(row["train_count"]),
            "model_seed": int(row["model_seed"]),
            "prediction": final_probe_predictions[model_index].copy(),
        })

    state_dict = None
    if cfg.save_final_weights:
        state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

    elapsed = time.time() - started
    result = {
        "width": width,
        "per_model_parameter_count": per_model_parameters,
        "elapsed_seconds": elapsed,
        "trajectory_rows": trajectory_rows,
        "crossing_rows": crossing_rows,
        "crossing_predictions": crossing_predictions,
        "final_prediction_records": final_prediction_records,
        "unreached": [
            {
                "width": width,
                "model_name": model_names[model_index],
                "rule": conditions[condition_indices[model_index]].rule,
                "train_count": model_train_counts[model_index],
                "model_seed": expanded_seeds[model_index],
                "unreached_raw_bce": level,
            }
            for model_index, levels in enumerate(pending_levels)
            for level in sorted(levels, reverse=True)
        ],
        "state_dict": state_dict,
    }

    del optimizer
    del model
    del initial
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def numeric_values(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    result: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        output = dict(zip(group_keys, group))
        output["row_count"] = len(group_rows)
        numeric_keys = sorted({
            key
            for row in group_rows
            for key, value in row.items()
            if key not in group_keys and isinstance(value, (int, float))
        })
        for key in numeric_keys:
            values = numeric_values(group_rows, key)
            if len(values) == 0:
                continue
            output[f"{key}_mean"] = float(np.mean(values))
            output[f"{key}_median"] = float(np.median(values))
            output[f"{key}_std"] = float(np.std(values))
            output[f"{key}_min"] = float(np.min(values))
            output[f"{key}_max"] = float(np.max(values))
        result.append(output)
    return result


def mean_pairwise_agreement(predictions: Sequence[np.ndarray]) -> tuple[float, float]:
    bit_values: list[float] = []
    exact_values: list[float] = []
    for left_index in range(len(predictions)):
        for right_index in range(left_index + 1, len(predictions)):
            equal = predictions[left_index] == predictions[right_index]
            bit_values.append(float(equal.mean()))
            exact_values.append(float(equal.all(axis=1).mean()))
    if not bit_values:
        return float("nan"), float("nan")
    return float(np.mean(bit_values)), float(np.mean(exact_values))


def prediction_group_metrics(
    predictions: Sequence[np.ndarray],
    targets: np.ndarray,
) -> dict[str, Any]:
    stack = np.stack(predictions).astype(np.uint8)
    pair_bit, pair_exact = mean_pairwise_agreement(predictions)
    majority = (stack.mean(axis=0) >= 0.5).astype(np.uint8)
    target_bool = targets.astype(np.uint8)
    unique_functions = len({
        hashlib.sha256(np.packbits(prediction).tobytes()).hexdigest()
        for prediction in predictions
    })
    return {
        "mean_pairwise_bit_agreement": pair_bit,
        "mean_pairwise_exact_agreement": pair_exact,
        "unanimous_bit_fraction": float(
            np.all(stack == stack[:1], axis=0).mean()
        ),
        "majority_probe_bit_accuracy": float((majority == target_bool).mean()),
        "majority_probe_exact_accuracy": float(
            (majority == target_bool).all(axis=1).mean()
        ),
        "unique_probe_function_count": unique_functions,
    }


def build_crossing_agreement(
    crossing_rows: Sequence[dict[str, Any]],
    crossing_predictions: Sequence[np.ndarray],
    probe_targets_by_rule: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in crossing_rows:
        grouped[(
            int(row["width"]),
            int(row["rule"]),
            int(row["train_count"]),
            float(row["target_raw_bce"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    for (width, rule, train_count, level), rows in sorted(grouped.items()):
        predictions = [
            crossing_predictions[int(row["crossing_prediction_index"])]
            for row in rows
        ]
        metrics = prediction_group_metrics(
            predictions, probe_targets_by_rule[rule]
        )
        output.append({
            "width": width,
            "rule": rule,
            "train_count": train_count,
            "target_raw_bce": level,
            "seed_count": len(rows),
            "strict_valid_seed_count": int(sum(
                bool(row["valid_strict_post_interpolation"]) for row in rows
            )),
            "after_fit_seed_count": int(sum(
                bool(row["valid_after_train_fit"]) for row in rows
            )),
            **metrics,
        })
    return output


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) < 2:
        return float("nan")
    rank_x = rankdata(values_x)
    rank_y = rankdata(values_y)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def build_capacity_pressure_trends(
    crossing_rows: Sequence[dict[str, Any]],
    cfg: RuntimeConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in crossing_rows:
        if not bool(row["valid_strict_post_interpolation"]):
            continue
        grouped[(
            int(row["rule"]),
            int(row["train_count"]),
            float(row["target_raw_bce"]),
        )].append(row)
    metrics = (
        "matched_probe_bit_accuracy",
        "matched_probe_exact_accuracy",
        "matched_monitor_bit_accuracy",
        "matched_monitor_exact_accuracy",
        "matched_step_to_loss",
    )
    output: list[dict[str, Any]] = []
    for (rule, train_count, level), rows in sorted(grouped.items()):
        by_width: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_width[int(row["width"])].append(row)
        widths = sorted(by_width)
        if len(widths) < 2:
            continue
        for metric in metrics:
            medians = [
                float(np.median(numeric_values(by_width[width], metric)))
                for width in widths
            ]
            decreasing_pairs = sum(
                medians[index] >= medians[index + 1]
                for index in range(len(medians) - 1)
            )
            output.append({
                "rule": rule,
                "train_count": train_count,
                "target_raw_bce": level,
                "is_primary_loss": level in cfg.primary_raw_bce_levels,
                "metric": metric,
                "width_count": len(widths),
                "widths": widths,
                "median_values": medians,
                "smallest_width": widths[0],
                "largest_width": widths[-1],
                "smallest_width_value": medians[0],
                "largest_width_value": medians[-1],
                "smallest_minus_largest": medians[0] - medians[-1],
                "spearman_width_vs_value": spearman(widths, medians),
                "fraction_adjacent_nonincreasing": (
                    decreasing_pairs / max(len(widths) - 1, 1)
                ),
            })
    return output


def build_matched_transition_thresholds(
    crossing_rows: Sequence[dict[str, Any]],
    cfg: RuntimeConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in crossing_rows:
        grouped[(
            int(row["width"]),
            int(row["rule"]),
            int(row["train_count"]),
            float(row["target_raw_bce"]),
        )].append(row)
    curve_rows: list[dict[str, Any]] = []
    for (width, rule, train_count, level), rows in sorted(grouped.items()):
        strict_rows = [
            row for row in rows
            if bool(row["valid_strict_post_interpolation"])
        ]
        exact_values = numeric_values(
            strict_rows, "matched_probe_exact_accuracy"
        )
        bit_values = numeric_values(
            strict_rows, "matched_probe_bit_accuracy"
        )
        success_count = int(np.sum(
            exact_values >= cfg.generalization_exact_threshold
        ))
        curve_rows.append({
            "width": width,
            "rule": rule,
            "train_count": train_count,
            "target_raw_bce": level,
            "is_primary_loss": level in cfg.primary_raw_bce_levels,
            "seed_count": len(rows),
            "strict_valid_seed_count": len(strict_rows),
            "strict_valid_seed_fraction": len(strict_rows) / len(rows),
            "probe_exact_median": (
                float(np.median(exact_values)) if len(exact_values) else None
            ),
            "probe_bit_median": (
                float(np.median(bit_values)) if len(bit_values) else None
            ),
            "success_count": success_count,
            "success_fraction_all_seeds": success_count / len(rows),
            "generalization_exact_threshold": (
                cfg.generalization_exact_threshold
            ),
        })

    summary_rows: list[dict[str, Any]] = []
    for width in cfg.widths:
        for rule in cfg.rules:
            for level in cfg.raw_bce_levels:
                rows = sorted([
                    row for row in curve_rows
                    if int(row["width"]) == width
                    and int(row["rule"]) == rule
                    and math.isclose(float(row["target_raw_bce"]), level)
                ], key=lambda row: int(row["train_count"]))

                def first_count(predicate: Any) -> int | None:
                    hit = next((row for row in rows if predicate(row)), None)
                    return None if hit is None else int(hit["train_count"])

                n50 = first_count(lambda row: (
                    row["probe_exact_median"] is not None
                    and float(row["probe_exact_median"]) >= 0.50
                ))
                n99_median = first_count(lambda row: (
                    row["probe_exact_median"] is not None
                    and float(row["probe_exact_median"])
                    >= cfg.generalization_exact_threshold
                ))
                n_stable = first_count(lambda row: (
                    float(row["strict_valid_seed_fraction"])
                    >= cfg.generalization_required_seed_fraction
                    and float(row["success_fraction_all_seeds"])
                    >= cfg.generalization_required_seed_fraction
                ))
                summary_rows.append({
                    "width": width,
                    "rule": rule,
                    "target_raw_bce": level,
                    "is_primary_loss": level in cfg.primary_raw_bce_levels,
                    "n50_median_exact": n50,
                    "n99_median_exact": n99_median,
                    "n_stable": n_stable,
                    "required_seed_fraction": (
                        cfg.generalization_required_seed_fraction
                    ),
                })
    return curve_rows, summary_rows


def build_practical_plateau_models(
    trajectory_rows: Sequence[dict[str, Any]],
    cfg: RuntimeConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(
            int(row["width"]),
            int(row["rule"]),
            int(row["train_count"]),
            int(row["model_seed"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    endpoint_metrics = (
        "train_raw_bce",
        "train_bit_accuracy",
        "train_exact_accuracy",
        "monitor_raw_bce",
        "monitor_bit_accuracy",
        "monitor_exact_accuracy",
        "probe_raw_bce",
        "probe_bit_accuracy",
        "probe_exact_accuracy",
        "parameter_norm",
        "weight_norm",
        "parameter_delta_norm",
        "weight_delta_norm",
        "log_frobenius_weight_product",
    )
    for (width, rule, train_count, seed), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["step"]))
        final_step = int(rows[-1]["step"])
        window = [
            row for row in rows
            if int(row["step"]) >= final_step - cfg.plateau_window_steps
        ]
        endpoint = rows[-cfg.endpoint_median_evals:]
        loss_start = float(window[0]["train_raw_bce"])
        loss_end = float(window[-1]["train_raw_bce"])
        relative_drop = (
            (loss_start - loss_end) / max(abs(loss_start), 1e-30)
        )
        bit_values = numeric_values(window, "monitor_bit_accuracy")
        exact_values = numeric_values(window, "monitor_exact_accuracy")
        stability_values = numeric_values(
            window, "monitor_prediction_bit_stability"
        )
        stable_fraction = (
            float(np.mean(
                stability_values
                >= cfg.plateau_prediction_stability_threshold
            ))
            if len(stability_values) else 0.0
        )
        enough_evals = len(window) >= cfg.plateau_min_evals
        train_fit = float(window[-1]["train_exact_accuracy"]) >= (
            1.0 - cfg.train_exact_tolerance
        )
        loss_stable = (
            loss_end <= cfg.plateau_loss_floor
            or relative_drop <= cfg.plateau_loss_relative_drop_max
        )
        bit_range = float(np.max(bit_values) - np.min(bit_values))
        exact_range = float(np.max(exact_values) - np.min(exact_values))
        accuracy_stable = (
            bit_range <= cfg.plateau_monitor_bit_range
            and exact_range <= cfg.plateau_monitor_exact_range
        )
        prediction_stable = (
            stable_fraction >= cfg.plateau_required_stable_fraction
        )
        practical_plateau = bool(
            enough_evals
            and train_fit
            and loss_stable
            and accuracy_stable
            and prediction_stable
        )
        row: dict[str, Any] = {
            "width": width,
            "rule": rule,
            "train_count": train_count,
            "model_seed": seed,
            "final_step": final_step,
            "train_fit_step": rows[-1].get("train_fit_step"),
            "window_eval_count": len(window),
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_relative_drop": relative_drop,
            "monitor_bit_range": bit_range,
            "monitor_exact_range": exact_range,
            "prediction_stable_fraction": stable_fraction,
            "enough_plateau_evals": enough_evals,
            "train_fitted": train_fit,
            "loss_stable": loss_stable,
            "accuracy_stable": accuracy_stable,
            "prediction_stable": prediction_stable,
            "practical_plateau": practical_plateau,
        }
        for metric in endpoint_metrics:
            values = numeric_values(endpoint, metric)
            row[f"endpoint_{metric}"] = (
                float(np.median(values)) if len(values) else None
            )
        output.append(row)
    return output


def build_final_agreement(
    records: Sequence[dict[str, Any]],
    probe_targets_by_rule: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(
            int(record["width"]),
            int(record["rule"]),
            int(record["train_count"]),
        )].append(record)
    output: list[dict[str, Any]] = []
    for (width, rule, train_count), rows in sorted(grouped.items()):
        predictions = [row["prediction"] for row in rows]
        output.append({
            "width": width,
            "rule": rule,
            "train_count": train_count,
            "seed_count": len(rows),
            **prediction_group_metrics(
                predictions, probe_targets_by_rule[rule]
            ),
        })
    return output


def build_plateau_by_count(
    plateau_models: Sequence[dict[str, Any]],
    final_agreement: Sequence[dict[str, Any]],
    cfg: RuntimeConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agreement_map = {
        (int(row["width"]), int(row["rule"]), int(row["train_count"])): row
        for row in final_agreement
    }
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in plateau_models:
        grouped[(
            int(row["width"]),
            int(row["rule"]),
            int(row["train_count"]),
        )].append(row)
    curve_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        width, rule, train_count = key
        exact = numeric_values(rows, "endpoint_probe_exact_accuracy")
        bit = numeric_values(rows, "endpoint_probe_bit_accuracy")
        plateau_flags = np.asarray([
            bool(row["practical_plateau"]) for row in rows
        ])
        success = exact >= cfg.generalization_exact_threshold
        stable_success = plateau_flags & success
        agreement = agreement_map.get(key, {})
        curve_rows.append({
            "width": width,
            "rule": rule,
            "train_count": train_count,
            "seed_count": len(rows),
            "practical_plateau_fraction": float(np.mean(plateau_flags)),
            "endpoint_probe_exact_median": float(np.median(exact)),
            "endpoint_probe_exact_min": float(np.min(exact)),
            "endpoint_probe_exact_max": float(np.max(exact)),
            "endpoint_probe_bit_median": float(np.median(bit)),
            "fixed_budget_success_fraction": float(np.mean(success)),
            "stable_success_fraction": float(np.mean(stable_success)),
            **{
                key_name: value
                for key_name, value in agreement.items()
                if key_name not in {"width", "rule", "train_count", "seed_count"}
            },
        })

    threshold_rows: list[dict[str, Any]] = []
    for width in cfg.widths:
        for rule in cfg.rules:
            rows = sorted([
                row for row in curve_rows
                if int(row["width"]) == width and int(row["rule"]) == rule
            ], key=lambda row: int(row["train_count"]))

            def first_count(predicate: Any) -> int | None:
                hit = next((row for row in rows if predicate(row)), None)
                return None if hit is None else int(hit["train_count"])

            def first_sustained_count(predicate: Any) -> int | None:
                for index, row in enumerate(rows):
                    if predicate(row) and all(
                        predicate(later) for later in rows[index:]
                    ):
                        return int(row["train_count"])
                return None

            median_50 = lambda row: (
                float(row["endpoint_probe_exact_median"]) >= 0.50
            )
            median_99 = lambda row: (
                float(row["endpoint_probe_exact_median"])
                >= cfg.generalization_exact_threshold
            )
            stable_success = lambda row: (
                float(row["practical_plateau_fraction"])
                >= cfg.generalization_required_seed_fraction
                and float(row["stable_success_fraction"])
                >= cfg.generalization_required_seed_fraction
            )

            threshold_rows.append({
                "width": width,
                "rule": rule,
                "n50_fixed_budget": first_count(median_50),
                "n99_fixed_budget": first_count(median_99),
                "n_stable_practical_plateau": first_count(stable_success),
                "n50_fixed_budget_sustained": first_sustained_count(median_50),
                "n99_fixed_budget_sustained": first_sustained_count(median_99),
                "n_stable_practical_plateau_sustained": (
                    first_sustained_count(stable_success)
                ),
                "fixed_budget_success_nonmonotonic": bool(
                    first_count(median_99) != first_sustained_count(median_99)
                ),
                "practical_plateau_success_nonmonotonic": bool(
                    first_count(stable_success)
                    != first_sustained_count(stable_success)
                ),
                "required_seed_fraction": (
                    cfg.generalization_required_seed_fraction
                ),
            })
    return curve_rows, threshold_rows


def save_packed_predictions(
    path: Path,
    crossing_predictions: Sequence[np.ndarray],
    final_prediction_records: Sequence[dict[str, Any]],
) -> None:
    crossing_stack = (
        np.stack(crossing_predictions).astype(np.uint8)
        if crossing_predictions
        else np.empty((0, 0, 0), dtype=np.uint8)
    )
    final_stack = (
        np.stack([row["prediction"] for row in final_prediction_records]).astype(np.uint8)
        if final_prediction_records
        else np.empty((0, 0, 0), dtype=np.uint8)
    )
    final_metadata = np.asarray([
        [
            int(row["width"]),
            int(row["rule"]),
            int(row["train_count"]),
            int(row["model_seed"]),
        ]
        for row in final_prediction_records
    ], dtype=np.int64)
    np.savez_compressed(
        path,
        crossing_predictions_packed=np.packbits(crossing_stack, axis=-1),
        crossing_output_bits=(
            crossing_stack.shape[-1] if crossing_stack.ndim == 3 else 0
        ),
        final_predictions_packed=np.packbits(final_stack, axis=-1),
        final_output_bits=(final_stack.shape[-1] if final_stack.ndim == 3 else 0),
        final_metadata_width_rule_n_seed=final_metadata,
    )


def plot_matched_loss_curves(
    crossing_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    cfg: RuntimeConfig,
) -> None:
    strict_rows = [
        row for row in crossing_rows
        if bool(row["valid_strict_post_interpolation"])
        and float(row["target_raw_bce"]) in cfg.primary_raw_bce_levels
    ]
    if not strict_rows:
        return
    for rule in cfg.rules:
        levels = list(cfg.primary_raw_bce_levels)
        figure, axes = plt.subplots(
            1,
            len(levels),
            figsize=(5.2 * len(levels), 4.2),
            squeeze=False,
        )
        for axis, level in zip(axes[0], levels):
            for width in cfg.widths:
                points = []
                for train_count in cfg.train_counts:
                    rows = [
                        row for row in strict_rows
                        if int(row["rule"]) == rule
                        and int(row["width"]) == width
                        and int(row["train_count"]) == train_count
                        and math.isclose(
                            float(row["target_raw_bce"]), level
                        )
                    ]
                    values = numeric_values(
                        rows, "matched_probe_exact_accuracy"
                    )
                    if len(values):
                        points.append((train_count, float(np.median(values))))
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=f"width={width}",
                    )
            axis.set_xscale("log", base=2)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("train count")
            axis.set_ylabel("probe exact accuracy")
            axis.set_title(f"matched BCE={level:g}")
            axis.grid(True, alpha=0.3)
        axes[0, -1].legend(fontsize=8)
        figure.suptitle(f"Rule {rule}: capacity pressure at matched loss")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"rule{rule}_matched_loss_width_curves.png",
            dpi=180,
        )
        plt.close(figure)


def plot_plateau_curves(
    curve_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    cfg: RuntimeConfig,
) -> None:
    if not curve_rows:
        return
    for rule in cfg.rules:
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        for width in cfg.widths:
            rows = sorted([
                row for row in curve_rows
                if int(row["rule"]) == rule and int(row["width"]) == width
            ], key=lambda row: int(row["train_count"]))
            if not rows:
                continue
            axis.plot(
                [int(row["train_count"]) for row in rows],
                [float(row["endpoint_probe_exact_median"]) for row in rows],
                marker="o",
                label=f"width={width}",
            )
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("train count")
        axis.set_ylabel("endpoint probe exact accuracy")
        axis.set_title(f"Rule {rule}: fixed-budget/practical-plateau curves")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"rule{rule}_practical_plateau_width_curves.png",
            dpi=180,
        )
        plt.close(figure)


def plot_thresholds(
    matched_thresholds: Sequence[dict[str, Any]],
    plateau_thresholds: Sequence[dict[str, Any]],
    output_dir: Path,
    cfg: RuntimeConfig,
) -> None:
    for rule in cfg.rules:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        plotted = False
        for level in cfg.primary_raw_bce_levels:
            rows = sorted([
                row for row in matched_thresholds
                if int(row["rule"]) == rule
                and math.isclose(float(row["target_raw_bce"]), level)
                and row.get("n_stable") is not None
            ], key=lambda row: int(row["width"]))
            if rows:
                axis.plot(
                    [int(row["width"]) for row in rows],
                    [int(row["n_stable"]) for row in rows],
                    marker="o",
                    label=f"matched BCE={level:g}",
                )
                plotted = True
        plateau_rows = sorted([
            row for row in plateau_thresholds
            if int(row["rule"]) == rule
            and row.get("n_stable_practical_plateau_sustained") is not None
        ], key=lambda row: int(row["width"]))
        if plateau_rows:
            axis.plot(
                [int(row["width"]) for row in plateau_rows],
                [
                    int(row["n_stable_practical_plateau_sustained"])
                    for row in plateau_rows
                ],
                marker="s",
                linewidth=2.5,
                label="practical plateau (sustained)",
            )
            plotted = True
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("hidden width")
        axis.set_ylabel("estimated transition train count")
        axis.set_title(f"Rule {rule}: transition count versus width")
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"rule{rule}_transition_count_vs_width.png",
            dpi=180,
        )
        plt.close(figure)


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

    output_dir = prepare_output_dir(cfg)
    write_json(output_dir / "config.json", {
        **asdict(cfg),
        "result_dir": str(output_dir),
    })
    conditions = build_conditions(cfg)
    write_json(output_dir / "conditions.json", [
        asdict(condition) for condition in conditions
    ])
    dataset = make_dataset(cfg, conditions)
    np.savez_compressed(
        output_dir / "dataset_split_ids.npz",
        train_ids=dataset["train_ids"],
        monitor_ids=dataset["monitor_ids"],
        probe_ids=dataset["probe_ids"],
    )

    print("=== Capacity-pressure width scan ===", flush=True)
    print(f"设备：{cfg.device}", flush=True)
    if torch.device(cfg.device).type == "cuda":
        print(
            f"GPU：{torch.cuda.get_device_name(torch.device(cfg.device))}",
            flush=True,
        )
    print(
        f"规则={list(cfg.rules)} | widths={list(cfg.widths)} | "
        f"n={list(cfg.train_counts)} | seeds={list(cfg.model_seeds)}",
        flush=True,
    )
    print(
        f"MLP：{cfg.bit_length} -> width x {cfg.hidden_layer_count} -> "
        f"{cfg.bit_length} | activation={cfg.activation} | "
        f"normalization={cfg.normalization} | dropout=0 | weight_decay=0",
        flush=True,
    )
    print(
        f"max_steps={cfg.max_steps:,} | batch={cfg.batch_size} | "
        f"matched BCE={list(cfg.raw_bce_levels)} | "
        f"primary={list(cfg.primary_raw_bce_levels)}",
        flush=True,
    )
    print(f"结果目录：{output_dir}", flush=True)

    all_trajectory_rows: list[dict[str, Any]] = []
    all_crossing_rows: list[dict[str, Any]] = []
    all_crossing_predictions: list[np.ndarray] = []
    all_final_prediction_records: list[dict[str, Any]] = []
    all_unreached: list[dict[str, Any]] = []
    width_runtime_rows: list[dict[str, Any]] = []
    started = time.time()

    for width in cfg.widths:
        result = run_width(
            width,
            cfg,
            conditions,
            dataset,
            crossing_prediction_offset=len(all_crossing_predictions),
        )
        all_trajectory_rows.extend(result["trajectory_rows"])
        all_crossing_rows.extend(result["crossing_rows"])
        all_crossing_predictions.extend(result["crossing_predictions"])
        all_final_prediction_records.extend(result["final_prediction_records"])
        all_unreached.extend(result["unreached"])
        width_runtime_rows.append({
            "width": width,
            "elapsed_seconds": result["elapsed_seconds"],
            "per_model_parameter_count": result["per_model_parameter_count"],
            "crossing_count": len(result["crossing_rows"]),
            "unreached_count": len(result["unreached"]),
        })
        if cfg.save_final_weights and result["state_dict"] is not None:
            torch.save(
                result["state_dict"],
                output_dir / f"final_width_{width}.pt",
            )

        # 每个宽度结束就落盘，防止后续宽度失败丢掉已有数据。
        write_jsonl(output_dir / "trajectory_models.jsonl", all_trajectory_rows)
        write_csv(output_dir / "trajectory_models.csv", all_trajectory_rows)
        write_jsonl(output_dir / "matched_loss_crossings.jsonl", all_crossing_rows)
        write_csv(output_dir / "matched_loss_crossings.csv", all_crossing_rows)
        write_csv(output_dir / "unreached_thresholds.csv", all_unreached)
        write_csv(output_dir / "width_runtime.csv", width_runtime_rows)
        save_packed_predictions(
            output_dir / "probe_predictions_packed.npz",
            all_crossing_predictions,
            all_final_prediction_records,
        )

    crossing_summary = aggregate_rows(
        all_crossing_rows,
        ("width", "rule", "train_count", "target_raw_bce"),
    )
    write_csv(output_dir / "matched_loss_summary.csv", crossing_summary)
    write_json(output_dir / "matched_loss_summary.json", crossing_summary)

    crossing_agreement = build_crossing_agreement(
        all_crossing_rows,
        all_crossing_predictions,
        dataset["probe_targets_by_rule"],
    )
    write_csv(output_dir / "matched_loss_agreement.csv", crossing_agreement)
    write_json(output_dir / "matched_loss_agreement.json", crossing_agreement)

    capacity_trends = build_capacity_pressure_trends(
        all_crossing_rows, cfg
    )
    write_csv(output_dir / "capacity_pressure_trends.csv", capacity_trends)
    write_json(output_dir / "capacity_pressure_trends.json", capacity_trends)

    matched_curve, matched_thresholds = build_matched_transition_thresholds(
        all_crossing_rows, cfg
    )
    write_csv(output_dir / "matched_loss_transition_curve.csv", matched_curve)
    write_json(output_dir / "matched_loss_transition_curve.json", matched_curve)
    write_csv(
        output_dir / "matched_loss_transition_thresholds.csv",
        matched_thresholds,
    )
    write_json(
        output_dir / "matched_loss_transition_thresholds.json",
        matched_thresholds,
    )

    plateau_models = build_practical_plateau_models(
        all_trajectory_rows, cfg
    )
    write_csv(output_dir / "practical_plateau_models.csv", plateau_models)
    write_json(output_dir / "practical_plateau_models.json", plateau_models)

    final_agreement = build_final_agreement(
        all_final_prediction_records,
        dataset["probe_targets_by_rule"],
    )
    write_csv(output_dir / "final_function_agreement.csv", final_agreement)
    write_json(output_dir / "final_function_agreement.json", final_agreement)

    plateau_curve, plateau_thresholds = build_plateau_by_count(
        plateau_models, final_agreement, cfg
    )
    write_csv(output_dir / "practical_plateau_by_count.csv", plateau_curve)
    write_json(output_dir / "practical_plateau_by_count.json", plateau_curve)
    write_csv(
        output_dir / "practical_plateau_thresholds.csv",
        plateau_thresholds,
    )
    write_json(
        output_dir / "practical_plateau_thresholds.json",
        plateau_thresholds,
    )

    plot_matched_loss_curves(all_crossing_rows, output_dir, cfg)
    plot_plateau_curves(plateau_curve, output_dir, cfg)
    plot_thresholds(
        matched_thresholds, plateau_thresholds, output_dir, cfg
    )

    elapsed = time.time() - started
    primary_trends = [
        row for row in capacity_trends
        if bool(row["is_primary_loss"])
        and row["metric"] in {
            "matched_probe_bit_accuracy",
            "matched_probe_exact_accuracy",
        }
    ]
    summary = {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "rules": list(cfg.rules),
        "widths": list(cfg.widths),
        "train_counts": list(cfg.train_counts),
        "seed_count": len(cfg.model_seeds),
        "independent_model_runs": (
            len(cfg.rules)
            * len(cfg.widths)
            * len(cfg.train_counts)
            * len(cfg.model_seeds)
        ),
        "matched_crossing_count": len(all_crossing_rows),
        "expected_crossing_count": (
            len(cfg.rules)
            * len(cfg.widths)
            * len(cfg.train_counts)
            * len(cfg.model_seeds)
            * len(cfg.raw_bce_levels)
        ),
        "unreached_threshold_count": len(all_unreached),
        "practical_plateau_model_fraction": float(np.mean([
            bool(row["practical_plateau"]) for row in plateau_models
        ])),
        "matched_loss_thresholds": matched_thresholds,
        "practical_plateau_thresholds": plateau_thresholds,
        "primary_capacity_trends": primary_trends,
        "interpretation_boundary": (
            "Matched-loss differences test width-dependent function selection "
            "at equal optimization depth. Practical plateau is a finite-budget "
            "endpoint, not an infinite-time stationary distribution."
        ),
    }
    write_json(output_dir / "summary.json", summary)

    archive = package_results(output_dir) if cfg.package_results else None
    print("\n=== 实验完成 ===", flush=True)
    print(f"总耗时：{elapsed:.1f}s", flush=True)
    print(
        f"crossings={len(all_crossing_rows):,}/"
        f"{summary['expected_crossing_count']:,} | "
        f"unreached={len(all_unreached):,}",
        flush=True,
    )
    print(f"汇总：{output_dir / 'summary.json'}", flush=True)
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
