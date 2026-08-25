#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4-bit Boolean 函数空间上的训练 loss 条件先验实验。

核心问题
--------
神经网络只在有限训练样本上降低连续 BCE loss。即便 hard predictions 已经
满足这些训练约束，进一步进入更低训练 loss 的初始化权重区域时，完整未见
输入上的函数概率是否系统性地向低复杂度函数迁移？

相对于旧 3-bit 实验的扩展
-------------------------
1. 输入从 3 bit 扩展到 4 bit，完整函数空间从 256 增至 65,536；
2. 不再依赖三组手选约束。对 k=2/4/6/8 分别预注册 projection、parity、
   majority 和三组 balanced-random 条件，共 24 组；
3. 支持 MLP、Residual MLP、1D CNN、GRU、Tiny Transformer 和
   MLP-Mixer 的向量化初始化先验；
4. 完整保存每个低-loss 切片的 65,536 维函数计数，而不是只保存复杂度均值；
5. raw BCE 是主命题口径；另报 RMS-normalized BCE、固定 logit-scale 子群，
   仅用于分析整体 logit scale 是否参与该效应；
6. 复杂度不依赖单一 DSL，报告 ANF、Walsh、最小 DNF/CNF、最小决策树、
   truth-table transition 等互补代理，并提供探索性的多代理综合排名。

重要边界
--------
- loss 只在 ConditionSpec.input_indices 指定的有限训练样本上计算；
- function id 与复杂度使用全部 16 个输入状态的 hard outputs 计算；
- 这是固定初始化分布下的静态 loss-conditioned 几何，不等价于 SGD 分布；
- Kolmogorov complexity 不可计算，本文所有复杂度量均为明确标注的代理；
- raw BCE 同时选择 margin 和 logit scale，而这正是实际训练目标的一部分。
  normalized/fixed-scale 是机制诊断，不参与 raw BCE 主命题的真伪判决。
- 非 MLP 架构是针对 4-token 布尔任务的 tiny 参考机，不声称
  逐细节复刻工业库中的完整模块。

AutoDL / Jupyter
----------------
1. 默认 Config.PROFILE="architecture_pilot"，先进行轻量跨架构压力测试；
2. 方向稳定后再增加对应架构的采样量，不默认直接跑旧 MLP full；
3. 所有路径和参数都在 Config 内，不依赖环境变量；
4. 先验 logits 按 shard 保存，可中断续跑；最终 zip 自动排除大体积 shard。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import random
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# 配置
# =============================================================================


def script_directory() -> Path:
    source = globals().get("__file__")
    if source:
        return Path(source).resolve().parent
    return Path.cwd()


@dataclass
class Config:
    # 默认是轻量跨架构 Pilot；旧 MLP full 配置仍保留在下方。
    PROFILE: str = "architecture_pilot"  # architecture_pilot / pilot / full / smoke
    RESULT_DIR: Path = Path(
        "/root/results_loss_conditioned_prior_architecture_pilot"
    )
    RESUME_PRIOR_SHARDS: bool = True
    CREATE_ARCHIVE: bool = True
    INCLUDE_PRIOR_SHARDS_IN_ARCHIVE: bool = False
    GENERATE_PLOTS: bool = True

    INPUT_BITS: int = 4
    CONSTRAINT_SIZES: tuple[int, ...] = (2, 4, 6, 8)
    RANDOM_BALANCED_REPLICATES: int = 3
    CONDITION_SEED: int = 20260823

    # 只检验每个参考机内部的 low-loss 方向，不强行比较
    # 不同架构的绝对函数概率。
    ARCHITECTURES: tuple[str, ...] = (
        "tanh16x2",
        "gelu_ln64x3",
        "resmlp64x4",
        "cnn1d16x3",
        "gru32x1",
        "transformer32x2_h4",
        "mixer32x2",
    )
    PRIOR_SEED_BASE: int = 83_000_000
    GLOBAL_SEED: int = 20260823

    PILOT_MODEL_COUNTS: dict[str, int] = field(
        default_factory=lambda: {
            "tanh16x2": 1_048_576,
            "gelu_ln1024x3": 262_144,
        }
    )
    ARCHITECTURE_PILOT_MODEL_COUNTS: dict[str, int] = field(
        default_factory=lambda: {
            "tanh16x2": 1_048_576,
            "gelu_ln64x3": 524_288,
            "resmlp64x4": 524_288,
            "cnn1d16x3": 524_288,
            "gru32x1": 262_144,
            "transformer32x2_h4": 131_072,
            "mixer32x2": 262_144,
        }
    )
    FULL_MODEL_COUNTS: dict[str, int] = field(
        default_factory=lambda: {
            # tanh 小网络吞吐极高，提高样本量以稳定 k=8 的 0.1% 尾部。
            "tanh16x2": 67_108_864,
            # 旧 3-bit 1024x3 pilot 在 5090 上约 4.4s / 262,144 models；
            # 4-bit 前向状态翻倍。下面共 384 个块，预计约 50-65 分钟。
            "gelu_ln1024x3": 100_663_296,
        }
    )
    # micro batch 控制显存；storage shard 控制磁盘文件数量和续跑粒度。
    MICRO_BATCH_SIZES: dict[str, int] = field(
        default_factory=lambda: {
            "tanh16x2": 65_536,
            "gelu_ln1024x3": 256,
            "gelu_ln64x3": 4_096,
            "resmlp64x4": 2_048,
            "cnn1d16x3": 4_096,
            "gru32x1": 2_048,
            "transformer32x2_h4": 256,
            "mixer32x2": 1_024,
        }
    )
    STORAGE_SHARD_SIZES: dict[str, int] = field(
        default_factory=lambda: {
            "tanh16x2": 1_048_576,
            "gelu_ln1024x3": 262_144,
            "gelu_ln64x3": 262_144,
            "resmlp64x4": 262_144,
            "cnn1d16x3": 262_144,
            "gru32x1": 262_144,
            "transformer32x2_h4": 131_072,
            "mixer32x2": 262_144,
        }
    )

    QUANTILE_FRACTIONS: tuple[float, ...] = (
        1.0,
        0.5,
        0.2,
        0.1,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.001,
    )
    FIXED_SCALE_QUANTILES: tuple[float, float] = (0.40, 0.60)
    MIN_RELIABLE_SELECTED_PILOT: int = 100
    MIN_RELIABLE_SELECTED_FULL: int = 200
    TOP_FUNCTIONS_PER_SLICE: int = 40
    SAVE_FULL_RAW_LOSS_HISTOGRAMS: bool = True

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32: bool = False
    LOG_INTERVAL_SECONDS: float = 15.0

    _SMOKE_RESULT_DIR: Path = field(
        default=Path(
            "research/function_information_conservation/"
            "_smoke_loss_conditioned_prior_architecture_pilot"
        ),
        repr=False,
    )


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    hidden_size: int
    hidden_layers: int
    activation: str
    use_layer_norm: bool
    heads: int
    ff_multiplier: int
    kernel_size: int
    model_count: int
    micro_batch_size: int
    storage_shard_size: int


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    constraint_size: int
    family: str
    input_indices: tuple[int, ...]
    targets: tuple[int, ...]
    full_target_function_id: int | None
    description: str


@dataclass
class EffectiveRun:
    profile: str
    result_dir: Path
    min_reliable_selected: int
    specs: list[ArchitectureSpec]
    condition_sizes: tuple[int, ...]
    random_replicates: int
    quantile_fractions: tuple[float, ...]


def architecture_catalog() -> dict[str, dict[str, Any]]:
    return {
        "tanh16x2": {
            "family": "mlp", "hidden_size": 16, "hidden_layers": 2,
            "activation": "tanh", "use_layer_norm": False,
        },
        "gelu_ln1024x3": {
            "family": "mlp", "hidden_size": 1024, "hidden_layers": 3,
            "activation": "gelu", "use_layer_norm": True,
        },
        "gelu_ln64x3": {
            "family": "mlp", "hidden_size": 64, "hidden_layers": 3,
            "activation": "gelu", "use_layer_norm": True,
        },
        "resmlp64x4": {
            "family": "resmlp", "hidden_size": 64, "hidden_layers": 4,
            "activation": "gelu", "use_layer_norm": True,
        },
        "cnn1d16x3": {
            "family": "cnn1d", "hidden_size": 16, "hidden_layers": 3,
            "activation": "gelu", "use_layer_norm": True,
            "kernel_size": 3,
        },
        "gru32x1": {
            "family": "gru", "hidden_size": 32, "hidden_layers": 1,
            "activation": "tanh", "use_layer_norm": False,
        },
        "transformer32x2_h4": {
            "family": "transformer", "hidden_size": 32, "hidden_layers": 2,
            "activation": "gelu", "use_layer_norm": True,
            "heads": 4, "ff_multiplier": 2,
        },
        "mixer32x2": {
            "family": "mixer", "hidden_size": 32, "hidden_layers": 2,
            "activation": "gelu", "use_layer_norm": True,
            "ff_multiplier": 2,
        },
    }


def resolve_run(cfg: Config) -> EffectiveRun:
    profile = cfg.PROFILE.strip().lower()
    catalog = architecture_catalog()
    selected_architectures = cfg.ARCHITECTURES
    unknown = sorted(set(selected_architectures) - set(catalog))
    if unknown:
        raise ValueError(f"未知架构：{unknown}；可选值={sorted(catalog)}")

    if profile == "architecture_pilot":
        counts = cfg.ARCHITECTURE_PILOT_MODEL_COUNTS
        result_dir = cfg.RESULT_DIR
        min_reliable = 50
        condition_sizes = cfg.CONSTRAINT_SIZES
        random_replicates = min(2, cfg.RANDOM_BALANCED_REPLICATES)
        fractions = cfg.QUANTILE_FRACTIONS
    elif profile == "pilot":
        counts = cfg.PILOT_MODEL_COUNTS
        result_dir = cfg.RESULT_DIR
        min_reliable = cfg.MIN_RELIABLE_SELECTED_PILOT
        condition_sizes = cfg.CONSTRAINT_SIZES
        random_replicates = cfg.RANDOM_BALANCED_REPLICATES
        fractions = cfg.QUANTILE_FRACTIONS
    elif profile == "full":
        counts = cfg.FULL_MODEL_COUNTS
        result_dir = cfg.RESULT_DIR
        min_reliable = cfg.MIN_RELIABLE_SELECTED_FULL
        condition_sizes = cfg.CONSTRAINT_SIZES
        random_replicates = cfg.RANDOM_BALANCED_REPLICATES
        fractions = cfg.QUANTILE_FRACTIONS
    elif profile == "smoke":
        counts = {name: 512 for name in cfg.ARCHITECTURES}
        result_dir = cfg._SMOKE_RESULT_DIR
        min_reliable = 8
        condition_sizes = (2, 4)
        random_replicates = 1
        fractions = (1.0, 0.5, 0.2)
    else:
        raise ValueError(
            "PROFILE 只能是 architecture_pilot、pilot、full 或 smoke。"
        )

    specs: list[ArchitectureSpec] = []
    for name in selected_architectures:
        architecture = catalog[name]
        model_count = int(counts[name])
        micro_batch_size = min(
            int(cfg.MICRO_BATCH_SIZES[name]), model_count
        )
        storage_shard_size = min(
            int(cfg.STORAGE_SHARD_SIZES[name]), model_count
        )
        if profile == "smoke":
            micro_batch_size = min(32, model_count)
            storage_shard_size = min(128, model_count)
        specs.append(
            ArchitectureSpec(
                name=name,
                family=str(architecture["family"]),
                hidden_size=int(architecture["hidden_size"]),
                hidden_layers=int(architecture["hidden_layers"]),
                activation=str(architecture["activation"]),
                use_layer_norm=bool(architecture["use_layer_norm"]),
                heads=int(architecture.get("heads", 1)),
                ff_multiplier=int(architecture.get("ff_multiplier", 2)),
                kernel_size=int(architecture.get("kernel_size", 3)),
                model_count=model_count,
                micro_batch_size=micro_batch_size,
                storage_shard_size=storage_shard_size,
            )
        )
    return EffectiveRun(
        profile=profile,
        result_dir=Path(result_dir),
        min_reliable_selected=min_reliable,
        specs=specs,
        condition_sizes=tuple(condition_sizes),
        random_replicates=int(random_replicates),
        quantile_fractions=tuple(fractions),
    )


def validate_config(cfg: Config, run: EffectiveRun) -> None:
    if cfg.INPUT_BITS != 4:
        raise ValueError("本实验固定为 4-bit -> 1-bit，以完整枚举 65,536 个函数。")
    if not run.specs:
        raise ValueError("至少需要一个架构。")
    if any(size <= 0 or size > 16 for size in run.condition_sizes):
        raise ValueError("约束数量必须位于 1..16。")
    if any(size % 2 != 0 for size in run.condition_sizes):
        raise ValueError("为控制输出 Hamming weight，默认约束数量必须为偶数。")
    low, high = cfg.FIXED_SCALE_QUANTILES
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("FIXED_SCALE_QUANTILES 非法。")
    if any(not 0.0 < value <= 1.0 for value in run.quantile_fractions):
        raise ValueError("所有 retained fraction 必须位于 (0, 1]。")


# =============================================================================
# 通用 I/O
# =============================================================================


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def stable_hash(payload: Any, length: int = 16) -> str:
    text = json.dumps(
        json_ready(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 输入空间和预注册条件
# =============================================================================


def truth_table_input_bits(input_bits: int) -> np.ndarray:
    values = np.arange(1 << input_bits, dtype=np.uint16)
    shifts = np.arange(input_bits - 1, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def outputs_to_function_id(outputs: np.ndarray) -> int:
    bits = np.asarray(outputs, dtype=np.uint64).reshape(-1)
    powers = np.left_shift(np.uint64(1), np.arange(len(bits), dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def choose_balanced_indices(
    labels: np.ndarray,
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    zeros = np.flatnonzero(labels == 0)
    ones = np.flatnonzero(labels == 1)
    half = count // 2
    if len(zeros) < half or len(ones) < half:
        raise ValueError("目标函数无法提供指定数量的平衡约束。")
    selected = np.concatenate(
        [
            generator.choice(zeros, size=half, replace=False),
            generator.choice(ones, size=half, replace=False),
        ]
    )
    generator.shuffle(selected)
    return np.sort(selected.astype(np.int64))


def build_conditions(
    cfg: Config,
    run: EffectiveRun,
) -> list[ConditionSpec]:
    inputs = truth_table_input_bits(cfg.INPUT_BITS)
    structured_targets = {
        "projection_x0": inputs[:, 0].astype(np.uint8),
        "parity4": (inputs.sum(axis=1) % 2).astype(np.uint8),
        "majority_ge2": (inputs.sum(axis=1) >= 2).astype(np.uint8),
    }
    conditions: list[ConditionSpec] = []
    master = np.random.default_rng(cfg.CONDITION_SEED)

    for count in run.condition_sizes:
        for family, full_outputs in structured_targets.items():
            seed = int(master.integers(0, 2**63 - 1))
            generator = np.random.default_rng(seed)
            indices = choose_balanced_indices(full_outputs, count, generator)
            targets = full_outputs[indices]
            conditions.append(
                ConditionSpec(
                    name=f"k{count:02d}_{family}",
                    constraint_size=count,
                    family=family,
                    input_indices=tuple(int(value) for value in indices),
                    targets=tuple(int(value) for value in targets),
                    full_target_function_id=outputs_to_function_id(full_outputs),
                    description=(
                        f"从完整 {family} truth table 中预注册抽取 {count} 个平衡约束"
                    ),
                )
            )

        for replicate in range(run.random_replicates):
            seed = int(master.integers(0, 2**63 - 1))
            generator = np.random.default_rng(seed)
            indices = np.sort(
                generator.choice(16, size=count, replace=False).astype(np.int64)
            )
            targets = np.asarray(
                [0] * (count // 2) + [1] * (count // 2), dtype=np.uint8
            )
            generator.shuffle(targets)
            conditions.append(
                ConditionSpec(
                    name=f"k{count:02d}_random_balanced_r{replicate}",
                    constraint_size=count,
                    family="random_balanced",
                    input_indices=tuple(int(value) for value in indices),
                    targets=tuple(int(value) for value in targets),
                    full_target_function_id=None,
                    description=(
                        f"随机输入位置与独立平衡标签，replicate={replicate}"
                    ),
                )
            )

    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise RuntimeError("条件名称发生重复。")
    return conditions


# =============================================================================
# 高吞吐初始化先验采样
# =============================================================================


def truth_inputs_torch(input_bits: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(truth_table_input_bits(input_bits)).to(
        device=device, dtype=torch.float32
    )


def activate_tensor(values: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "tanh":
        return torch.tanh(values)
    if activation == "gelu":
        return F.gelu(values)
    if activation == "relu":
        return F.relu(values)
    raise ValueError(f"未知激活函数：{activation}")


def layer_norm_last(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=-1, keepdim=True)
    variance = (values - mean).square().mean(dim=-1, keepdim=True)
    return (values - mean) * torch.rsqrt(variance + 1e-5)


def sample_uniform(
    shape: tuple[int, ...],
    bound: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        *shape, device=device, dtype=torch.float32
    ).uniform_(-bound, bound, generator=generator)


def sampled_linear(
    values: torch.Tensor,
    output_features: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """values 的第一维是独立 prior model，最后一维是 feature。"""
    model_count = values.shape[0]
    input_features = values.shape[-1]
    bound = 1.0 / math.sqrt(input_features)
    weight = sample_uniform(
        (model_count, output_features, input_features),
        bound,
        generator,
        values.device,
    )
    bias = sample_uniform(
        (model_count, output_features),
        bound,
        generator,
        values.device,
    )
    original_shape = values.shape[:-1]
    flat = values.reshape(model_count, -1, input_features)
    output = torch.bmm(flat, weight.transpose(1, 2)) + bias[:, None, :]
    return output.reshape(*original_shape, output_features)


def sampled_output_head(
    hidden: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return sampled_linear(hidden, 1, generator).squeeze(-1)


def sample_mlp_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    hidden = inputs[None, :, :].expand(count, -1, -1)
    for _ in range(spec.hidden_layers):
        hidden = sampled_linear(hidden, spec.hidden_size, generator)
        hidden = activate_tensor(hidden, spec.activation)
        if spec.use_layer_norm:
            hidden = layer_norm_last(hidden)
    return sampled_output_head(hidden, generator)


def sample_resmlp_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    hidden = inputs[None, :, :].expand(count, -1, -1)
    hidden = sampled_linear(hidden, spec.hidden_size, generator)
    hidden = activate_tensor(hidden, spec.activation)
    if spec.use_layer_norm:
        hidden = layer_norm_last(hidden)
    for _ in range(max(spec.hidden_layers - 1, 0)):
        residual_input = layer_norm_last(hidden) if spec.use_layer_norm else hidden
        delta = sampled_linear(residual_input, spec.hidden_size, generator)
        delta = activate_tensor(delta, spec.activation)
        hidden = (hidden + delta) / math.sqrt(2.0)
    if spec.use_layer_norm:
        hidden = layer_norm_last(hidden)
    return sampled_output_head(hidden, generator)


def sample_cnn1d_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    # [model, truth-state, token-position, channel]
    hidden = inputs[None, :, :, None].expand(count, -1, -1, 1)
    input_channels = 1
    kernel_size = spec.kernel_size
    center = kernel_size // 2
    for _ in range(spec.hidden_layers):
        bound = 1.0 / math.sqrt(input_channels * kernel_size)
        weight = sample_uniform(
            (count, spec.hidden_size, input_channels, kernel_size),
            bound,
            generator,
            inputs.device,
        )
        bias = sample_uniform(
            (count, spec.hidden_size), bound, generator, inputs.device
        )
        output = torch.zeros(
            count,
            inputs.shape[0],
            inputs.shape[1],
            spec.hidden_size,
            device=inputs.device,
            dtype=torch.float32,
        )
        for kernel_index in range(kernel_size):
            shifted = torch.roll(
                hidden,
                shifts=kernel_index - center,
                dims=2,
            )
            output += torch.einsum(
                "msli,moi->mslo",
                shifted,
                weight[:, :, :, kernel_index],
            )
        hidden = output + bias[:, None, None, :]
        hidden = activate_tensor(hidden, spec.activation)
        if spec.use_layer_norm:
            hidden = layer_norm_last(hidden)
        input_channels = spec.hidden_size
    hidden = hidden.reshape(count, inputs.shape[0], -1)
    return sampled_output_head(hidden, generator)


def sample_gru_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    hidden_size = spec.hidden_size
    bound = 1.0 / math.sqrt(hidden_size)
    weight_ih = sample_uniform(
        (count, 3 * hidden_size, 1), bound, generator, inputs.device
    )
    weight_hh = sample_uniform(
        (count, 3 * hidden_size, hidden_size),
        bound,
        generator,
        inputs.device,
    )
    bias_ih = sample_uniform(
        (count, 3 * hidden_size), bound, generator, inputs.device
    )
    bias_hh = sample_uniform(
        (count, 3 * hidden_size), bound, generator, inputs.device
    )
    state = torch.zeros(
        count,
        inputs.shape[0],
        hidden_size,
        device=inputs.device,
        dtype=torch.float32,
    )
    for token_index in range(inputs.shape[1]):
        token = inputs[None, :, token_index : token_index + 1].expand(
            count, -1, -1
        )
        gates_input = (
            torch.bmm(token, weight_ih.transpose(1, 2))
            + bias_ih[:, None, :]
        )
        gates_hidden = (
            torch.bmm(state, weight_hh.transpose(1, 2))
            + bias_hh[:, None, :]
        )
        input_reset, input_update, input_new = gates_input.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = gates_hidden.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        state = candidate + update * (state - candidate)
    return sampled_output_head(state, generator)


def sample_transformer_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    width = spec.hidden_size
    heads = spec.heads
    if width % heads != 0:
        raise ValueError(f"Transformer width={width} 不能被 heads={heads} 整除。")
    scale = 1.0 / math.sqrt(width)
    token_embedding = torch.empty(
        count, 2, width, device=inputs.device
    ).normal_(0.0, scale, generator=generator)
    position_embedding = torch.empty(
        count, inputs.shape[1], width, device=inputs.device
    ).normal_(0.0, scale, generator=generator)
    bit_values = inputs[None, :, :, None]
    hidden = (
        token_embedding[:, 0][:, None, None, :] * (1.0 - bit_values)
        + token_embedding[:, 1][:, None, None, :] * bit_values
        + position_embedding[:, None, :, :]
    )
    head_dim = width // heads
    for _ in range(spec.hidden_layers):
        normalized = layer_norm_last(hidden)
        query = sampled_linear(normalized, width, generator)
        key = sampled_linear(normalized, width, generator)
        value = sampled_linear(normalized, width, generator)
        query = query.reshape(
            count, inputs.shape[0], inputs.shape[1], heads, head_dim
        ).permute(0, 1, 3, 2, 4)
        key = key.reshape(
            count, inputs.shape[0], inputs.shape[1], heads, head_dim
        ).permute(0, 1, 3, 2, 4)
        value = value.reshape(
            count, inputs.shape[0], inputs.shape[1], heads, head_dim
        ).permute(0, 1, 3, 2, 4)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim),
            dim=-1,
        )
        context = torch.matmul(attention, value).permute(0, 1, 3, 2, 4)
        context = context.reshape(
            count, inputs.shape[0], inputs.shape[1], width
        )
        hidden = hidden + sampled_linear(context, width, generator)
        normalized = layer_norm_last(hidden)
        feed_forward = sampled_linear(
            normalized, width * spec.ff_multiplier, generator
        )
        feed_forward = activate_tensor(feed_forward, spec.activation)
        feed_forward = sampled_linear(feed_forward, width, generator)
        hidden = hidden + feed_forward
    hidden = layer_norm_last(hidden).reshape(count, inputs.shape[0], -1)
    return sampled_output_head(hidden, generator)


def sample_mixer_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    token_count = inputs.shape[1]
    width = spec.hidden_size
    hidden = inputs[None, :, :, None].expand(count, -1, -1, -1)
    hidden = sampled_linear(hidden, width, generator)
    position_scale = 1.0 / math.sqrt(width)
    position = torch.empty(
        count, token_count, width, device=inputs.device
    ).normal_(0.0, position_scale, generator=generator)
    hidden = hidden + position[:, None, :, :]
    token_hidden = token_count * spec.ff_multiplier
    for _ in range(spec.hidden_layers):
        normalized = layer_norm_last(hidden)
        first_bound = 1.0 / math.sqrt(token_count)
        token_weight_1 = sample_uniform(
            (count, token_hidden, token_count),
            first_bound,
            generator,
            inputs.device,
        )
        token_bias_1 = sample_uniform(
            (count, token_hidden), first_bound, generator, inputs.device
        )
        token_mixed = torch.einsum(
            "mstc,mht->mshc", normalized, token_weight_1
        ) + token_bias_1[:, None, :, None]
        token_mixed = activate_tensor(token_mixed, spec.activation)
        second_bound = 1.0 / math.sqrt(token_hidden)
        token_weight_2 = sample_uniform(
            (count, token_count, token_hidden),
            second_bound,
            generator,
            inputs.device,
        )
        token_bias_2 = sample_uniform(
            (count, token_count), second_bound, generator, inputs.device
        )
        token_mixed = torch.einsum(
            "mshc,mth->mstc", token_mixed, token_weight_2
        ) + token_bias_2[:, None, :, None]
        hidden = hidden + token_mixed

        normalized = layer_norm_last(hidden)
        channel_mixed = sampled_linear(
            normalized, width * spec.ff_multiplier, generator
        )
        channel_mixed = activate_tensor(channel_mixed, spec.activation)
        channel_mixed = sampled_linear(channel_mixed, width, generator)
        hidden = hidden + channel_mixed
    hidden = layer_norm_last(hidden).reshape(count, inputs.shape[0], -1)
    return sampled_output_head(hidden, generator)


@torch.inference_mode()
def sample_vectorized_logits(
    spec: ArchitectureSpec,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    if spec.family == "mlp":
        return sample_mlp_logits(spec, count, generator, inputs)
    if spec.family == "resmlp":
        return sample_resmlp_logits(spec, count, generator, inputs)
    if spec.family == "cnn1d":
        return sample_cnn1d_logits(spec, count, generator, inputs)
    if spec.family == "gru":
        return sample_gru_logits(spec, count, generator, inputs)
    if spec.family == "transformer":
        return sample_transformer_logits(spec, count, generator, inputs)
    if spec.family == "mixer":
        return sample_mixer_logits(spec, count, generator, inputs)
    raise ValueError(f"未实现的架构 family：{spec.family}")


def function_ids_from_logits(logits: torch.Tensor) -> np.ndarray:
    domain = logits.shape[1]
    powers = torch.bitwise_left_shift(
        torch.ones(domain, dtype=torch.int64, device=logits.device),
        torch.arange(domain, dtype=torch.int64, device=logits.device),
    )
    identifiers = ((logits > 0).to(torch.int64) * powers[None, :]).sum(dim=1)
    return identifiers.detach().cpu().numpy().astype(np.uint16)


def shard_signature(cfg: Config, spec: ArchitectureSpec) -> str:
    return stable_hash(
        {
            "protocol": "loss_conditioned_prior_scaling_4bit_v1",
            "input_bits": cfg.INPUT_BITS,
            "architecture": asdict(spec),
            "prior_seed_base": cfg.PRIOR_SEED_BASE,
        }
    )


def sample_prior_shards(
    cfg: Config,
    spec: ArchitectureSpec,
    architecture_dir: Path,
    device: torch.device,
) -> list[Path]:
    shard_dir = architecture_dir / "prior_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    signature = shard_signature(cfg, spec)
    inputs = truth_inputs_torch(cfg.INPUT_BITS, device)
    shard_paths: list[Path] = []
    total_shards = math.ceil(spec.model_count / spec.storage_shard_size)
    start_time = time.perf_counter()
    last_log = start_time

    metadata_path = shard_dir / "metadata.json"
    expected_metadata = {
        "signature": signature,
        "architecture": asdict(spec),
        "input_bits": cfg.INPUT_BITS,
        "dtype": "float16_logits_uint16_function_ids",
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(
                f"{shard_dir} 中存在不兼容 shard；请换 RESULT_DIR 或清空该目录。"
            )
    else:
        write_json(metadata_path, expected_metadata)

    for shard_index in range(total_shards):
        start = shard_index * spec.storage_shard_size
        count = min(spec.storage_shard_size, spec.model_count - start)
        path = shard_dir / f"shard_{shard_index:05d}_{start:09d}_{count:07d}.npz"
        shard_paths.append(path)
        if cfg.RESUME_PRIOR_SHARDS and path.exists():
            continue

        generator = torch.Generator(device=device)
        architecture_offset = int(
            hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:8], 16
        )
        generator.manual_seed(
            cfg.PRIOR_SEED_BASE + architecture_offset + shard_index
        )
        logits_np = np.empty((count, 1 << cfg.INPUT_BITS), dtype=np.float16)
        function_ids = np.empty(count, dtype=np.uint16)
        for micro_start in range(0, count, spec.micro_batch_size):
            micro_count = min(spec.micro_batch_size, count - micro_start)
            logits = sample_vectorized_logits(
                spec, micro_count, generator, inputs
            )
            logits_np[micro_start : micro_start + micro_count] = (
                logits.detach().cpu().numpy().astype(np.float16)
            )
            function_ids[micro_start : micro_start + micro_count] = (
                function_ids_from_logits(logits)
            )
            del logits
        temp = path.with_suffix(".tmp.npz")
        np.savez(
            temp,
            logits=logits_np,
            function_ids=function_ids,
            start=np.asarray([start], dtype=np.int64),
            count=np.asarray([count], dtype=np.int64),
        )
        temp.replace(path)
        del logits_np, function_ids

        now = time.perf_counter()
        if now - last_log >= cfg.LOG_INTERVAL_SECONDS or shard_index + 1 == total_shards:
            completed = start + count
            speed = completed / max(now - start_time, 1e-9)
            print(
                f"  [{spec.name}] prior={completed:,}/{spec.model_count:,} | "
                f"{speed:,.0f} models/s | shard={shard_index + 1}/{total_shards}",
                flush=True,
            )
            last_log = now
    return shard_paths


# =============================================================================
# 65,536 个函数的复杂度面板
# =============================================================================


def all_function_bits(input_bits: int) -> np.ndarray:
    domain = 1 << input_bits
    total = 1 << domain
    identifiers = np.arange(total, dtype=np.uint32)
    shifts = np.arange(domain, dtype=np.uint32)
    return ((identifiers[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def popcount_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([int(value).bit_count() for value in values], dtype=np.uint8)


def anf_metrics(bits: np.ndarray, input_bits: int) -> dict[str, np.ndarray]:
    coefficients = bits.copy()
    domain = 1 << input_bits
    for variable in range(input_bits):
        bit = 1 << variable
        for mask in range(domain):
            if mask & bit:
                coefficients[:, mask] ^= coefficients[:, mask ^ bit]
    degrees = popcount_array(np.arange(domain, dtype=np.uint32))
    terms = coefficients.sum(axis=1).astype(np.uint8)
    degree = np.zeros(len(bits), dtype=np.uint8)
    weighted = np.zeros(len(bits), dtype=np.float32)
    for monomial in range(domain):
        active = coefficients[:, monomial].astype(bool)
        degree[active] = np.maximum(degree[active], degrees[monomial])
        weighted += coefficients[:, monomial] * float(2 ** int(degrees[monomial]))
    return {
        "anf_terms": terms,
        "anf_degree": degree,
        "anf_weighted_terms": weighted,
    }


def walsh_metrics(bits: np.ndarray, input_bits: int) -> dict[str, np.ndarray]:
    values = (1 - 2 * bits.astype(np.int16)).copy()
    domain = 1 << input_bits
    step = 1
    while step < domain:
        for start in range(0, domain, 2 * step):
            left = values[:, start : start + step].copy()
            right = values[:, start + step : start + 2 * step].copy()
            values[:, start : start + step] = left + right
            values[:, start + step : start + 2 * step] = left - right
        step *= 2
    energy = values.astype(np.float64) ** 2
    energy /= float(domain * domain)
    safe = np.where(energy > 0, energy, 1.0)
    entropy = -np.sum(np.where(energy > 0, energy * np.log2(safe), 0.0), axis=1)
    degrees = popcount_array(np.arange(domain, dtype=np.uint32))
    return {
        "walsh_entropy": entropy.astype(np.float32),
        "walsh_mass_degree_le_1": energy[:, degrees <= 1].sum(axis=1).astype(
            np.float32
        ),
        "walsh_mass_degree_le_2": energy[:, degrees <= 2].sum(axis=1).astype(
            np.float32
        ),
        "walsh_max_character_degree": np.max(
            np.where(energy > 0, degrees[None, :], 0), axis=1
        ).astype(np.uint8),
    }


def cube_coverages(input_bits: int) -> list[tuple[int, int]]:
    inputs = truth_table_input_bits(input_bits)
    best_by_coverage: dict[int, int] = {}
    for specification in itertools.product((-1, 0, 1), repeat=input_bits):
        selected = np.ones(len(inputs), dtype=bool)
        literals = 0
        for column, value in enumerate(specification):
            if value == -1:
                continue
            literals += 1
            selected &= inputs[:, column] == value
        coverage = 0
        for index in np.flatnonzero(selected):
            coverage |= 1 << int(index)
        if coverage == 0:
            continue
        previous = best_by_coverage.get(coverage)
        if previous is None or literals < previous:
            best_by_coverage[coverage] = literals
    return sorted(best_by_coverage.items())


def all_minimum_dnf_costs(input_bits: int) -> tuple[np.ndarray, np.ndarray]:
    function_total = 1 << (1 << input_bits)
    cubes = cube_coverages(input_bits)
    terms = np.full(function_total, 255, dtype=np.uint8)
    literals = np.full(function_total, 255, dtype=np.uint8)
    terms[0] = 0
    literals[0] = 0
    for target in range(1, function_total):
        best_terms = 255
        best_literals = 255
        for coverage, cube_literals in cubes:
            if coverage & ~target:
                continue
            remainder = target & ~coverage
            candidate_terms = int(terms[remainder]) + 1
            candidate_literals = int(literals[remainder]) + cube_literals
            if candidate_terms < best_terms or (
                candidate_terms == best_terms
                and candidate_literals < best_literals
            ):
                best_terms = candidate_terms
                best_literals = candidate_literals
        terms[target] = best_terms
        literals[target] = best_literals
    return terms, literals


def restrict_function_id(
    function_id: int,
    input_bits: int,
    variable: int,
    value: int,
) -> int:
    result = 0
    output_index = 0
    for input_index in range(1 << input_bits):
        if ((input_index >> variable) & 1) != value:
            continue
        bit = (function_id >> input_index) & 1
        result |= bit << output_index
        output_index += 1
    return result


def decision_tree_leaf_costs(max_input_bits: int) -> np.ndarray:
    costs_by_bits: dict[int, np.ndarray] = {
        0: np.asarray([1, 1], dtype=np.uint8)
    }
    for input_bits in range(1, max_input_bits + 1):
        domain = 1 << input_bits
        function_total = 1 << domain
        costs = np.full(function_total, 255, dtype=np.uint8)
        costs[0] = 1
        costs[-1] = 1
        previous = costs_by_bits[input_bits - 1]
        for function_id in range(1, function_total - 1):
            best = 255
            for variable in range(input_bits):
                zero_id = restrict_function_id(
                    function_id, input_bits, variable, 0
                )
                one_id = restrict_function_id(
                    function_id, input_bits, variable, 1
                )
                candidate = int(previous[zero_id]) + int(previous[one_id])
                best = min(best, candidate)
            costs[function_id] = best
        costs_by_bits[input_bits] = costs
    return costs_by_bits[max_input_bits]


def lz_phrase_count(bits: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bits), dtype=np.uint8)
    for row_index, row in enumerate(bits):
        text = "".join("1" if value else "0" for value in row)
        dictionary: set[str] = set()
        position = 0
        phrases = 0
        while position < len(text):
            end = position + 1
            while end <= len(text) and text[position:end] in dictionary:
                end += 1
            dictionary.add(text[position:min(end, len(text))])
            phrases += 1
            position = min(end, len(text))
        counts[row_index] = phrases
    return counts


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def build_complexity_panel(input_bits: int) -> dict[str, np.ndarray]:
    bits = all_function_bits(input_bits)
    function_total, domain = bits.shape
    identifiers = np.arange(function_total, dtype=np.uint32)
    output_weight = bits.sum(axis=1).astype(np.uint8)
    transitions = np.count_nonzero(bits[:, 1:] != bits[:, :-1], axis=1).astype(
        np.uint8
    )
    cyclic_transitions = (
        transitions + (bits[:, 0] != bits[:, -1]).astype(np.uint8)
    )

    panel: dict[str, np.ndarray] = {
        "function_id": identifiers,
        "output_hamming_weight": output_weight,
        "lex_transition_count": transitions,
        "cyclic_transition_count": cyclic_transitions,
        "lz_phrase_count": lz_phrase_count(bits),
    }
    panel.update(anf_metrics(bits, input_bits))
    panel.update(walsh_metrics(bits, input_bits))

    dnf_terms, dnf_literals = all_minimum_dnf_costs(input_bits)
    complement = (function_total - 1) ^ identifiers
    cnf_terms = dnf_terms[complement]
    cnf_literals = dnf_literals[complement]
    choose_dnf = (dnf_terms < cnf_terms) | (
        (dnf_terms == cnf_terms) & (dnf_literals <= cnf_literals)
    )
    panel["min_normal_form_terms"] = np.where(
        choose_dnf, dnf_terms, cnf_terms
    ).astype(np.uint8)
    panel["min_normal_form_literals"] = np.where(
        choose_dnf, dnf_literals, cnf_literals
    ).astype(np.uint8)
    panel["decision_tree_leaves"] = decision_tree_leaf_costs(input_bits)

    proxy_names = (
        "anf_terms",
        "anf_weighted_terms",
        "walsh_entropy",
        "min_normal_form_literals",
        "decision_tree_leaves",
        "cyclic_transition_count",
        "lz_phrase_count",
    )
    ranks = np.stack(
        [average_tie_ranks(panel[name]) for name in proxy_names], axis=1
    )
    panel["multi_proxy_complexity_rank"] = ranks.mean(axis=1).astype(np.float32)
    panel["truth_table_bits"] = bits
    return panel


COMPLEXITY_EXPECTATION_FIELDS = (
    "anf_terms",
    "anf_degree",
    "anf_weighted_terms",
    "walsh_entropy",
    "walsh_mass_degree_le_1",
    "walsh_mass_degree_le_2",
    "min_normal_form_terms",
    "min_normal_form_literals",
    "decision_tree_leaves",
    "lex_transition_count",
    "cyclic_transition_count",
    "lz_phrase_count",
    "multi_proxy_complexity_rank",
)


def save_complexity_panel(
    result_dir: Path,
    panel: dict[str, np.ndarray],
) -> None:
    npz_payload = {
        key: value
        for key, value in panel.items()
        if key != "truth_table_bits"
    }
    np.savez_compressed(result_dir / "function_complexity_panel.npz", **npz_payload)
    rows: list[dict[str, Any]] = []
    bits = panel["truth_table_bits"]
    for function_id in range(len(bits)):
        row: dict[str, Any] = {
            "function_id": function_id,
            "function_hex": f"0x{function_id:04X}",
            "truth_table_bits_x0_to_x15": "".join(
                str(int(value)) for value in bits[function_id]
            ),
        }
        for field_name in COMPLEXITY_EXPECTATION_FIELDS:
            row[field_name] = json_ready(panel[field_name][function_id])
        row["output_hamming_weight"] = int(
            panel["output_hamming_weight"][function_id]
        )
        rows.append(row)
    write_csv(result_dir / "function_complexity_panel.csv", rows)


# =============================================================================
# Loss-conditioned 函数分布
# =============================================================================


def load_condition_cohort(
    shard_paths: list[Path],
    condition: ConditionSpec,
) -> dict[str, np.ndarray]:
    ids_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    normalized_parts: list[np.ndarray] = []
    rms_parts: list[np.ndarray] = []
    indices = np.asarray(condition.input_indices, dtype=np.int64)
    targets = np.asarray(condition.targets, dtype=np.float32)
    signed = targets * 2.0 - 1.0

    for path in shard_paths:
        with np.load(path) as payload:
            logits = payload["logits"].astype(np.float32)
            function_ids = payload["function_ids"].astype(np.uint16)
        selected = logits[:, indices]
        margins = selected * signed[None, :]
        hard = np.all(margins > 0.0, axis=1)
        if not np.any(hard):
            continue
        raw_loss = np.logaddexp(0.0, -margins).mean(axis=1)
        rms = np.sqrt(np.mean(np.square(logits, dtype=np.float64), axis=1)).astype(
            np.float32
        )
        normalized_margins = margins / np.maximum(rms[:, None], 1e-12)
        normalized_loss = np.logaddexp(0.0, -normalized_margins).mean(axis=1)
        ids_parts.append(function_ids[hard])
        raw_parts.append(raw_loss[hard].astype(np.float32))
        normalized_parts.append(normalized_loss[hard].astype(np.float32))
        rms_parts.append(rms[hard])

    if not ids_parts:
        return {
            "function_ids": np.empty(0, dtype=np.uint16),
            "raw_loss": np.empty(0, dtype=np.float32),
            "normalized_loss": np.empty(0, dtype=np.float32),
            "logit_rms": np.empty(0, dtype=np.float32),
        }
    return {
        "function_ids": np.concatenate(ids_parts),
        "raw_loss": np.concatenate(raw_parts),
        "normalized_loss": np.concatenate(normalized_parts),
        "logit_rms": np.concatenate(rms_parts),
    }


def entropy_bits(probability: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    positive = probability > 0
    return float(-np.sum(probability[positive] * np.log2(probability[positive])))


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    middle = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        valid = left > 0
        return float(np.sum(left[valid] * np.log2(left[valid] / right[valid])))

    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = math.sqrt(float(np.sum(first**2) * np.sum(second**2)))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(first * second) / denominator)


def distribution_summary(
    *,
    counts: np.ndarray,
    baseline_counts: np.ndarray,
    panel: dict[str, np.ndarray],
    condition: ConditionSpec,
) -> dict[str, Any]:
    counts = counts.astype(np.float64)
    baseline_counts = baseline_counts.astype(np.float64)
    total = float(counts.sum())
    baseline_total = float(baseline_counts.sum())
    probability = counts / max(total, 1.0)
    baseline_probability = baseline_counts / max(baseline_total, 1.0)
    row: dict[str, Any] = {
        "selected_count": int(total),
        "observed_function_count": int(np.count_nonzero(counts)),
        "function_entropy_bits": entropy_bits(probability),
        "top_function_id": int(np.argmax(counts)),
        "top_function_probability": float(probability.max()),
        "total_variation_from_family_baseline": total_variation(
            probability, baseline_probability
        ),
        "js_divergence_from_family_baseline_bits": js_divergence(
            probability, baseline_probability
        ),
    }
    for field_name in COMPLEXITY_EXPECTATION_FIELDS:
        row[f"expected_{field_name}"] = float(
            np.sum(probability * panel[field_name].astype(np.float64))
        )
    simple_mask = panel["multi_proxy_complexity_rank"] <= 0.10
    row["simplest_decile_probability"] = float(probability[simple_mask].sum())
    if condition.full_target_function_id is None:
        row["full_target_probability"] = float("nan")
    else:
        row["full_target_probability"] = float(
            probability[condition.full_target_function_id]
        )

    valid = baseline_counts >= 5
    if np.count_nonzero(valid) >= 10:
        function_total = len(counts)
        selected_smoothed = (counts + 0.5) / (total + 0.5 * function_total)
        baseline_smoothed = (baseline_counts + 0.5) / (
            baseline_total + 0.5 * function_total
        )
        log_enrichment = np.log2(selected_smoothed / baseline_smoothed)
        row["complexity_vs_log_enrichment_rank_correlation"] = pearson_correlation(
            average_tie_ranks(panel["multi_proxy_complexity_rank"][valid]),
            average_tie_ranks(log_enrichment[valid]),
        )
    else:
        row["complexity_vs_log_enrichment_rank_correlation"] = float("nan")
    return row


def append_top_function_rows(
    *,
    output_rows: list[dict[str, Any]],
    architecture: str,
    condition: ConditionSpec,
    family: str,
    retained_fraction: float,
    counts: np.ndarray,
    baseline_counts: np.ndarray,
    panel: dict[str, np.ndarray],
    top_count: int,
) -> None:
    total = float(counts.sum())
    baseline_total = float(baseline_counts.sum())
    top_ids = np.argsort(counts)[-top_count:][::-1]
    bits = panel["truth_table_bits"]
    for rank, function_id in enumerate(top_ids, start=1):
        count = int(counts[function_id])
        if count <= 0:
            continue
        probability = count / max(total, 1.0)
        baseline_probability = baseline_counts[function_id] / max(
            baseline_total, 1.0
        )
        output_rows.append(
            {
                "architecture": architecture,
                "condition": condition.name,
                "constraint_size": condition.constraint_size,
                "condition_family": condition.family,
                "loss_family": family,
                "retained_fraction": retained_fraction,
                "rank": rank,
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "truth_table_bits_x0_to_x15": "".join(
                    str(int(value)) for value in bits[function_id]
                ),
                "count": count,
                "probability": probability,
                "baseline_probability": baseline_probability,
                "enrichment": probability / max(baseline_probability, 1e-300),
                "multi_proxy_complexity_rank": float(
                    panel["multi_proxy_complexity_rank"][function_id]
                ),
                "anf_terms": int(panel["anf_terms"][function_id]),
                "anf_degree": int(panel["anf_degree"][function_id]),
                "min_normal_form_literals": int(
                    panel["min_normal_form_literals"][function_id]
                ),
                "decision_tree_leaves": int(
                    panel["decision_tree_leaves"][function_id]
                ),
                "walsh_entropy": float(panel["walsh_entropy"][function_id]),
            }
        )


def analyze_condition(
    *,
    cfg: Config,
    run: EffectiveRun,
    spec: ArchitectureSpec,
    condition: ConditionSpec,
    shard_paths: list[Path],
    panel: dict[str, np.ndarray],
    architecture_dir: Path,
    summary_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cohort = load_condition_cohort(shard_paths, condition)
    function_ids = cohort["function_ids"]
    if len(function_ids) == 0:
        condition_dir = architecture_dir / "conditions" / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        # 零命中本身是先验偏置的可报告结果。二项分布零成功时，95% 上界
        # 使用 rule-of-three 的精确形式 1 - 0.05**(1/N)。
        probability_upper_95 = 1.0 - math.exp(
            math.log(0.05) / float(spec.model_count)
        )
        condition_metadata = {
            **asdict(condition),
            "architecture": spec.name,
            "status": "no_hard_exact_prior_samples",
            "hard_exact_count": 0,
            "hard_exact_fraction": 0.0,
            "hard_exact_probability_upper_95": probability_upper_95,
            "sampled_prior_models": spec.model_count,
            "analysis_skipped": True,
            "skip_reason": (
                "初始化先验中没有权重同时满足全部 hard constraints，"
                "因此无法定义该条件下的低-loss 子水平集。"
            ),
        }
        write_json(condition_dir / "condition_metadata.json", condition_metadata)
        print(
            f"    零命中：{condition.name} | sampled={spec.model_count:,} | "
            f"hard-probability 95% upper bound={probability_upper_95:.3e} | "
            "跳过 loss-slice 分析",
            flush=True,
        )
        return condition_metadata

    raw_loss = cohort["raw_loss"]
    normalized_loss = cohort["normalized_loss"]
    logit_rms = cohort["logit_rms"]
    scale_low, scale_high = np.quantile(
        logit_rms, cfg.FIXED_SCALE_QUANTILES
    )
    fixed_scale_mask = (logit_rms >= scale_low) & (logit_rms <= scale_high)
    source_families = (
        ("raw_loss_hard", np.ones(len(function_ids), dtype=bool), raw_loss),
        (
            "normalized_loss_hard",
            np.ones(len(function_ids), dtype=bool),
            normalized_loss,
        ),
        ("raw_loss_fixed_scale", fixed_scale_mask, raw_loss),
    )
    condition_metadata = {
        **asdict(condition),
        "architecture": spec.name,
        "status": "ok",
        "hard_exact_count": len(function_ids),
        "hard_exact_fraction": len(function_ids) / spec.model_count,
        "raw_loss_min": float(raw_loss.min()),
        "raw_loss_q01": float(np.quantile(raw_loss, 0.01)),
        "raw_loss_q50": float(np.quantile(raw_loss, 0.50)),
        "raw_loss_max": float(raw_loss.max()),
        "fixed_scale_low": float(scale_low),
        "fixed_scale_high": float(scale_high),
        "fixed_scale_count": int(fixed_scale_mask.sum()),
    }

    condition_dir = architecture_dir / "conditions" / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    raw_hist_counts: list[np.ndarray] = []
    raw_hist_fractions: list[float] = []
    raw_hist_thresholds: list[float] = []

    for family, source_mask, score in source_families:
        source_indices = np.flatnonzero(source_mask)
        if len(source_indices) == 0:
            continue
        source_scores = score[source_indices]
        order = np.argsort(source_scores, kind="quicksort")
        sorted_scores = source_scores[order]
        sorted_global_indices = source_indices[order]
        sorted_function_ids = function_ids[sorted_global_indices].astype(np.int64)
        del order, source_scores, source_indices

        requested_prefixes: list[tuple[float, int]] = []
        for fraction in run.quantile_fractions:
            prefix = min(
                len(sorted_function_ids),
                max(1, int(math.ceil(fraction * len(sorted_function_ids)))),
            )
            if fraction < 1.0 and prefix < run.min_reliable_selected:
                continue
            requested_prefixes.append((float(fraction), prefix))

        # 从最低 loss 向外扩展，一次增量累计所有嵌套函数直方图。
        counts_by_prefix: dict[int, np.ndarray] = {}
        running_counts = np.zeros(65_536, dtype=np.int64)
        previous = 0
        for prefix in sorted({item[1] for item in requested_prefixes}):
            running_counts += np.bincount(
                sorted_function_ids[previous:prefix], minlength=65_536
            ).astype(np.int64)
            counts_by_prefix[prefix] = running_counts.copy()
            previous = prefix

        baseline_counts = np.bincount(
            sorted_function_ids, minlength=65_536
        ).astype(np.int64)
        baseline_summary = distribution_summary(
            counts=baseline_counts,
            baseline_counts=baseline_counts,
            panel=panel,
            condition=condition,
        )

        for fraction, selected_count in requested_prefixes:
            threshold = float(sorted_scores[selected_count - 1])
            selected_global = sorted_global_indices[:selected_count]
            counts = counts_by_prefix[selected_count]
            row = {
                "architecture": spec.name,
                "condition": condition.name,
                "constraint_size": condition.constraint_size,
                "condition_family": condition.family,
                "loss_family": family,
                "retained_fraction": float(fraction),
                "realized_retained_fraction": (
                    selected_count / len(sorted_function_ids)
                ),
                "loss_threshold": threshold,
                "source_count": len(sorted_function_ids),
                "mean_raw_loss": float(raw_loss[selected_global].mean()),
                "mean_normalized_loss": float(
                    normalized_loss[selected_global].mean()
                ),
                "mean_logit_rms": float(logit_rms[selected_global].mean()),
                **distribution_summary(
                    counts=counts,
                    baseline_counts=baseline_counts,
                    panel=panel,
                    condition=condition,
                ),
            }
            for key, value in baseline_summary.items():
                if key.startswith("expected_") or key in {
                    "simplest_decile_probability",
                    "function_entropy_bits",
                    "top_function_probability",
                }:
                    row[f"baseline_{key}"] = value
                    row[f"delta_{key}"] = float(row[key]) - float(value)
            summary_rows.append(row)
            append_top_function_rows(
                output_rows=top_rows,
                architecture=spec.name,
                condition=condition,
                family=family,
                retained_fraction=float(fraction),
                counts=counts,
                baseline_counts=baseline_counts,
                panel=panel,
                top_count=cfg.TOP_FUNCTIONS_PER_SLICE,
            )

            if family == "raw_loss_hard" and cfg.SAVE_FULL_RAW_LOSS_HISTOGRAMS:
                raw_hist_counts.append(counts.astype(np.int32))
                raw_hist_fractions.append(float(fraction))
                raw_hist_thresholds.append(threshold)

    if raw_hist_counts:
        np.savez_compressed(
            condition_dir / "raw_loss_full_function_counts.npz",
            counts=np.stack(raw_hist_counts, axis=0),
            retained_fractions=np.asarray(raw_hist_fractions, dtype=np.float64),
            thresholds=np.asarray(raw_hist_thresholds, dtype=np.float64),
        )
    write_json(condition_dir / "condition_metadata.json", condition_metadata)
    return condition_metadata


# =============================================================================
# 汇总和绘图
# =============================================================================


def aggregate_effects(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(row["architecture"]),
                str(row["loss_family"]),
                int(row["constraint_size"]),
            )
            for row in summary_rows
        }
    )
    tracked = (
        "delta_expected_multi_proxy_complexity_rank",
        "delta_simplest_decile_probability",
        "delta_expected_anf_terms",
        "delta_expected_min_normal_form_literals",
        "delta_expected_decision_tree_leaves",
        "delta_expected_walsh_entropy",
        "delta_function_entropy_bits",
    )
    for architecture, family, constraint_size in keys:
        subset = [
            row
            for row in summary_rows
            if row["architecture"] == architecture
            and row["loss_family"] == family
            and int(row["constraint_size"]) == constraint_size
        ]
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for row in subset:
            by_condition.setdefault(str(row["condition"]), []).append(row)
        terminal_rows: list[dict[str, Any]] = []
        for rows in by_condition.values():
            annealed_rows = [
                row
                for row in rows
                if float(row["retained_fraction"]) < 1.0
            ]
            if not annealed_rows:
                # 只有 baseline、没有可靠低-loss子切片时，不存在可判定方向。
                continue
            annealed_rows.sort(
                key=lambda item: float(item["retained_fraction"])
            )
            terminal_rows.append(annealed_rows[0])
        if not terminal_rows:
            continue
        aggregate: dict[str, Any] = {
            "architecture": architecture,
            "loss_family": family,
            "constraint_size": constraint_size,
            "condition_count": len(terminal_rows),
            "mean_terminal_retained_fraction": float(
                np.mean(
                    [float(row["retained_fraction"]) for row in terminal_rows]
                )
            ),
        }
        for field_name in tracked:
            values = np.asarray(
                [float(row[field_name]) for row in terminal_rows], dtype=np.float64
            )
            aggregate[f"mean_{field_name}"] = float(np.mean(values))
            aggregate[f"median_{field_name}"] = float(np.median(values))
            if "simplest_decile_probability" in field_name:
                expected_positive = values > 0
            else:
                expected_positive = values < 0
            aggregate[f"expected_direction_count_{field_name}"] = int(
                np.count_nonzero(expected_positive)
            )
            aggregate[f"expected_direction_fraction_{field_name}"] = float(
                np.mean(expected_positive)
            )
        output.append(aggregate)
    return output


def plot_summary(
    result_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", flush=True)
        return

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for architecture in sorted({row["architecture"] for row in summary_rows}):
        subset = [
            row
            for row in summary_rows
            if row["architecture"] == architecture
            and row["loss_family"] == "raw_loss_hard"
        ]
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        field_specs = (
            (
                "expected_multi_proxy_complexity_rank",
                "Expected multi-proxy complexity rank",
            ),
            ("simplest_decile_probability", "Probability mass in simplest decile"),
            ("expected_anf_terms", "Expected ANF terms"),
            ("function_entropy_bits", "Function-distribution entropy (bits)"),
        )
        for axis, (field_name, title) in zip(axes.flat, field_specs):
            for constraint_size in sorted(
                {int(row["constraint_size"]) for row in subset}
            ):
                rows_k = [
                    row
                    for row in subset
                    if int(row["constraint_size"]) == constraint_size
                ]
                fractions = sorted(
                    {float(row["retained_fraction"]) for row in rows_k},
                    reverse=True,
                )
                x_values: list[float] = []
                y_values: list[float] = []
                y_min: list[float] = []
                y_max: list[float] = []
                for fraction in fractions:
                    values = [
                        float(row[field_name])
                        for row in rows_k
                        if float(row["retained_fraction"]) == fraction
                    ]
                    if not values:
                        continue
                    x_values.append(fraction)
                    y_values.append(float(np.mean(values)))
                    y_min.append(float(np.min(values)))
                    y_max.append(float(np.max(values)))
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2,
                    label=f"k={constraint_size}",
                )
                axis.fill_between(x_values, y_min, y_max, alpha=0.12)
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.set_xlabel("retained lowest-loss fraction")
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle(
            f"4-bit loss-conditioned prior: {architecture}", fontsize=14
        )
        figure.savefig(
            plot_dir / f"loss_conditioned_summary_{architecture}.png", dpi=170
        )
        plt.close(figure)


def create_archive(result_dir: Path, include_shards: bool) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if not path.is_file():
                continue
            if not include_shards and "prior_shards" in path.parts:
                continue
            archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    cfg = Config()
    run = resolve_run(cfg)
    validate_config(cfg, run)
    result_dir = run.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE 要求 CUDA，但当前 PyTorch 看不到 GPU。")

    set_global_seed(cfg.GLOBAL_SEED)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.ALLOW_TF32)

    conditions = build_conditions(cfg, run)
    print("=== 4-bit Loss-conditioned Function Prior Scaling ===", flush=True)
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(f"profile={run.profile} | 结果目录={result_dir}", flush=True)
    print(
        f"函数空间=65,536 | 输入状态=16 | conditions={len(conditions)} | "
        f"k={list(run.condition_sizes)}",
        flush=True,
    )
    for spec in run.specs:
        print(
            f"  {spec.name}: models={spec.model_count:,}, "
            f"family={spec.family}, "
            f"width={spec.hidden_size}, layers={spec.hidden_layers}, "
            f"micro_batch={spec.micro_batch_size:,}, "
            f"storage_shard={spec.storage_shard_size:,}",
            flush=True,
        )

    config_payload = asdict(cfg)
    config_payload.pop("_SMOKE_RESULT_DIR", None)
    write_json(result_dir / "config.json", config_payload)
    write_json(result_dir / "conditions.json", [asdict(item) for item in conditions])
    write_json(
        result_dir / "runtime.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    )

    print("\n构建 65,536 函数复杂度面板……", flush=True)
    complexity_start = time.perf_counter()
    panel = build_complexity_panel(cfg.INPUT_BITS)
    save_complexity_panel(result_dir, panel)
    print(
        f"复杂度面板完成：{time.perf_counter() - complexity_start:.1f}s",
        flush=True,
    )

    all_summary_rows: list[dict[str, Any]] = []
    all_top_rows: list[dict[str, Any]] = []
    condition_metadata_rows: list[dict[str, Any]] = []
    architecture_summaries: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for spec in run.specs:
        architecture_dir = result_dir / spec.name
        architecture_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {spec.name}: 采样初始化先验 ===", flush=True)
        sample_start = time.perf_counter()
        shard_paths = sample_prior_shards(
            cfg, spec, architecture_dir, device
        )
        sample_seconds = time.perf_counter() - sample_start
        print(
            f"  [{spec.name}] 同一批先验已就绪：{spec.model_count:,} models；"
            f"后续 {len(conditions)} 个 conditions 全部复用这些 logits，"
            "不再采样权重。",
            flush=True,
        )

        architecture_summary_rows: list[dict[str, Any]] = []
        architecture_top_rows: list[dict[str, Any]] = []
        architecture_condition_rows: list[dict[str, Any]] = []
        analysis_start = time.perf_counter()
        for index, condition in enumerate(conditions, start=1):
            print(
                f"  [{spec.name}] 复用先验分析 condition "
                f"{index}/{len(conditions)}: "
                f"{condition.name}",
                flush=True,
            )
            metadata = analyze_condition(
                cfg=cfg,
                run=run,
                spec=spec,
                condition=condition,
                shard_paths=shard_paths,
                panel=panel,
                architecture_dir=architecture_dir,
                summary_rows=architecture_summary_rows,
                top_rows=architecture_top_rows,
            )
            architecture_condition_rows.append(metadata)
            write_csv(
                architecture_dir / "loss_slice_summary.csv",
                architecture_summary_rows,
            )
            write_csv(
                architecture_dir / "top_function_probabilities.csv",
                architecture_top_rows,
            )

        analysis_seconds = time.perf_counter() - analysis_start
        write_csv(
            architecture_dir / "condition_summary.csv",
            architecture_condition_rows,
        )
        all_summary_rows.extend(architecture_summary_rows)
        all_top_rows.extend(architecture_top_rows)
        condition_metadata_rows.extend(architecture_condition_rows)
        architecture_summaries.append(
            {
                "architecture": asdict(spec),
                "prior_sampling_seconds": sample_seconds,
                "analysis_seconds": analysis_seconds,
                "condition_count": len(conditions),
                "analyzable_condition_count": sum(
                    row.get("status") == "ok"
                    for row in architecture_condition_rows
                ),
                "zero_hard_condition_count": sum(
                    row.get("status") == "no_hard_exact_prior_samples"
                    for row in architecture_condition_rows
                ),
                "loss_slice_count": len(architecture_summary_rows),
            }
        )

    aggregate_rows = aggregate_effects(all_summary_rows)
    write_csv(result_dir / "all_loss_slice_summary.csv", all_summary_rows)
    write_csv(result_dir / "all_top_function_probabilities.csv", all_top_rows)
    write_csv(result_dir / "all_condition_summary.csv", condition_metadata_rows)
    write_csv(result_dir / "aggregate_effects.csv", aggregate_rows)
    if cfg.GENERATE_PLOTS:
        plot_summary(result_dir, all_summary_rows)

    raw_aggregate = [
        row for row in aggregate_rows if row["loss_family"] == "raw_loss_hard"
    ]
    normalized_aggregate = [
        row
        for row in aggregate_rows
        if row["loss_family"] == "normalized_loss_hard"
    ]
    fixed_aggregate = [
        row
        for row in aggregate_rows
        if row["loss_family"] == "raw_loss_fixed_scale"
    ]

    def mean_direction(rows: list[dict[str, Any]], field: str) -> float:
        if not rows:
            return float("nan")
        return float(
            np.mean([float(row[f"expected_direction_fraction_{field}"]) for row in rows])
        )

    # 这些是全函数空间上的探索性复杂度代理诊断，不是核心
    # simple-vs-complex 命题的裁决。核心裁决必须使用条件内 direct pairs。
    proxy_diagnostics = {
        "raw_composite_expected_direction_fraction": mean_direction(
            raw_aggregate, "delta_expected_multi_proxy_complexity_rank"
        ),
        "normalized_composite_expected_direction_fraction": mean_direction(
            normalized_aggregate,
            "delta_expected_multi_proxy_complexity_rank",
        ),
        "fixed_scale_composite_expected_direction_fraction": mean_direction(
            fixed_aggregate, "delta_expected_multi_proxy_complexity_rank"
        ),
        "raw_simple_mass_expected_direction_fraction": mean_direction(
            raw_aggregate, "delta_simplest_decile_probability"
        ),
        "normalized_simple_mass_expected_direction_fraction": mean_direction(
            normalized_aggregate, "delta_simplest_decile_probability"
        ),
        "fixed_scale_simple_mass_expected_direction_fraction": mean_direction(
            fixed_aggregate, "delta_simplest_decile_probability"
        ),
    }
    summary = {
        "status": "ok",
        "protocol": "loss_conditioned_prior_scaling_4bit_v1",
        "profile": run.profile,
        "function_space_size": 65_536,
        "condition_count": len(conditions),
        "architectures": architecture_summaries,
        "zero_hard_conditions": [
            {
                "architecture": row["architecture"],
                "condition": row["name"],
                "sampled_prior_models": row["sampled_prior_models"],
                "hard_exact_probability_upper_95": row[
                    "hard_exact_probability_upper_95"
                ],
            }
            for row in condition_metadata_rows
            if row.get("status") == "no_hard_exact_prior_samples"
        ],
        "exploratory_proxy_direction_fractions": proxy_diagnostics,
        "core_simple_vs_complex_verdict": (
            "not_adjudicated: run analyze_loss_conditioned_rule_pair_competition.py "
            "on the same prior shards"
        ),
        "elapsed_seconds": time.perf_counter() - overall_start,
        "interpretation": (
            "raw_loss_hard 是与真实 BCE 训练目标一致的主判决口径。normalized "
            "与 fixed-scale 仅诊断整体 logit scale 在效应中的作用，不能用于否定"
            "raw BCE 命题。本实验测量静态先验几何，不把该关联等同于 SGD 动力学。"
        ),
    }
    write_json(result_dir / "summary.json", summary)

    archive_path: Path | None = None
    if cfg.CREATE_ARCHIVE:
        archive_path = create_archive(
            result_dir,
            include_shards=cfg.INCLUDE_PRIOR_SHARDS_IN_ARCHIVE,
        )

    print("\n=== 实验完成 ===", flush=True)
    print("探索性 proxy 方向诊断（非核心裁决）：", flush=True)
    print(
        json.dumps(proxy_diagnostics, ensure_ascii=False, indent=2),
        flush=True,
    )
    print(f"汇总：{result_dir / 'summary.json'}", flush=True)
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}", flush=True)


if __name__ == "__main__":
    main()
