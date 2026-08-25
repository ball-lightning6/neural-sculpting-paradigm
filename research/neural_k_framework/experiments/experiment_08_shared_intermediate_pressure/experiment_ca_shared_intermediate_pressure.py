#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CA 共享中间表示与计算压力实验。

目标
----
检验在四种原子分支任务完全配平时，允许共享隐藏计算的 MLP 是否会因为
两个输出可以复用同一个昂贵 CA 中间状态，而以更小容量或更少训练达到
更低 loss。

四格交叉设计
------------
输入由两个独立均匀随机状态组成：x = [u, v]，每个状态 30 bit。

    F_k(u) = Rule30^k(u)
    G_k(v) = Rule110^k(v)
    A(z)   = z[:15] AND z[15:]
    B(z)   = z[:15] OR  z[15:]

四个联合目标为：

    shared_f:   [A(F_k(u)), B(F_k(u))]
    shared_g:   [A(G_k(v)), B(G_k(v))]
    separate_fg:[A(F_k(u)), B(G_k(v))]
    separate_gf:[A(G_k(v)), B(F_k(u))]

Shared 与 Separate 两组都恰好包含一次 A∘F、B∘F、A∘G、B∘G，因此
Rule30/Rule110、AND/OR 和输出位置的单任务难度主效应被交叉配平。u、v
相互独立，所以 Separate 的两个昂贵中间状态也相互独立。

架构对照
--------
1. joint：一个 60 -> W x L -> 30 的 MLP，两个输出头可共享所有隐藏表示。
2. split：两个隐藏层完全独立、总参数量尽量匹配的 MLP，各输出 15 bit。

主要判别量：

    gap_arch = mean_loss(separate) - mean_loss(shared)
    interaction = gap_joint - gap_split

若共享中间计算真的提高有限资源下的拟合效率，应观察到 joint 的 gap > 0，
split 的 gap 约为 0，且 interaction 随 CA 前缀深度 k 增长。

实现说明
--------
- 每个 (k, width, architecture) 组把 4 conditions x N seeds 一次并行训练；
- 同一 seed 的四个条件使用逐元素完全相同的初始化；
- joint/split、不同 width 与四个条件使用同一可复现在线数据流；
- 默认配置面向 RTX 5090；所有设置都在 Config 中，不依赖环境变量；
- 支持当前组断点续跑、已完成组跳过、CSV/JSON/PNG 汇总和 zip 打包。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONDITION_NAMES = (
    "shared_f",
    "shared_g",
    "separate_fg",
    "separate_gf",
)
SHARED_CONDITIONS = ("shared_f", "shared_g")
SEPARATE_CONDITIONS = ("separate_fg", "separate_gf")
ARCHITECTURES = ("joint", "split")


@dataclass
class Config:
    # =========================
    # 输出与运行模式
    # =========================
    RESULT_DIR: Path = Path("/root/results_ca_shared_intermediate_pressure")
    EXPERIMENT_NAME: str = "ca_shared_intermediate_crossed_pilot_v1"
    RESUME_COMPLETED_GROUPS: bool = True
    SAVE_ACTIVE_GROUP_CHECKPOINT: bool = True
    CHECKPOINT_INTERVAL_STEPS: int = 1000
    CREATE_ANALYSIS_ARCHIVE: bool = True
    GENERATE_PLOTS: bool = True

    # 本地验证时设为 True。正式上传到 AutoDL 前保持 False。
    SMOKE_TEST: bool = False

    # =========================
    # CA 任务
    # =========================
    STATE_BITS: int = 30
    PREFIX_RULE_F: int = 30
    PREFIX_RULE_G: int = 110
    PREFIX_DEPTHS: tuple[int, ...] = (0, 1, 2, 3)

    # =========================
    # 模型
    # =========================
    JOINT_WIDTHS: tuple[int, ...] = (256, 512, 1024)
    HIDDEN_LAYERS: int = 3
    MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)
    ACTIVATION: str = "gelu"
    USE_LAYER_NORM: bool = True
    LAYER_NORM_EPS: float = 1e-5

    # =========================
    # 在线训练
    # =========================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    ONLINE_BATCH_SIZE: int = 512
    MAX_STEPS: int = 5000
    EVAL_INTERVAL_STEPS: int = 250
    PROBE_SIZE: int = 32768
    PROBE_BATCH_SIZE: int = 4096
    DATA_SEED: int = 20260818
    PROBE_SEED: int = 20260819

    # 默认用 FP32/TF32，避免极低 loss 比较混入 BF16 前向量化误差。
    # 只想快速定位容量区间时可以手动改为 True。
    USE_BF16_AUTOCAST: bool = False
    ALLOW_TF32: bool = True

    # 记录首次达到各 probe BCE 的累计样本量。
    LOSS_THRESHOLDS: tuple[float, ...] = (
        1e-2,
        3e-3,
        1e-3,
        3e-4,
        1e-4,
    )

    # 是否保存最终模型。默认关闭，避免 1024 width 的批量权重占用大量空间。
    SAVE_FINAL_MODEL_STATE: bool = False

    # 运行时填充，不参与用户配置。
    _SMOKE_RESULT_DIR: Path = field(
        default=Path(
            "research/computational_pressure/"
            "_smoke_results_ca_shared_intermediate"
        ),
        repr=False,
    )


def apply_smoke_overrides(cfg: Config) -> Config:
    if not cfg.SMOKE_TEST:
        return cfg
    cfg.RESULT_DIR = cfg._SMOKE_RESULT_DIR
    cfg.PREFIX_DEPTHS = (0, 1)
    cfg.JOINT_WIDTHS = (12,)
    cfg.HIDDEN_LAYERS = 2
    cfg.MODEL_SEEDS = (0, 1)
    cfg.ONLINE_BATCH_SIZE = 16
    cfg.MAX_STEPS = 3
    cfg.EVAL_INTERVAL_STEPS = 1
    cfg.PROBE_SIZE = 48
    cfg.PROBE_BATCH_SIZE = 24
    cfg.SAVE_ACTIVE_GROUP_CHECKPOINT = False
    cfg.CREATE_ANALYSIS_ARCHIVE = True
    cfg.GENERATE_PLOTS = True
    cfg.RESUME_COMPLETED_GROUPS = False
    cfg.USE_BF16_AUTOCAST = False
    return cfg


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False))
            handle.write("\n")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def config_payload(cfg: Config) -> dict[str, Any]:
    payload = asdict(cfg)
    payload.pop("_SMOKE_RESULT_DIR", None)
    return json_ready(payload)


def config_signature(cfg: Config) -> str:
    payload = config_payload(cfg).copy()
    # 输出位置、绘图和续跑策略不改变实验语义。
    for key in (
        "RESULT_DIR",
        "RESUME_COMPLETED_GROUPS",
        "SAVE_ACTIVE_GROUP_CHECKPOINT",
        "CHECKPOINT_INTERVAL_STEPS",
        "CREATE_ANALYSIS_ARCHIVE",
        "GENERATE_PLOTS",
        "SAVE_FINAL_MODEL_STATE",
        "SMOKE_TEST",
    ):
        payload.pop(key, None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def validate_config(cfg: Config) -> None:
    if cfg.STATE_BITS <= 0 or cfg.STATE_BITS % 2 != 0:
        raise ValueError("STATE_BITS 必须是正偶数。")
    for rule in (cfg.PREFIX_RULE_F, cfg.PREFIX_RULE_G):
        if not 0 <= int(rule) <= 255:
            raise ValueError(f"ECA rule 必须位于 0..255，实际为 {rule}。")
    if not cfg.PREFIX_DEPTHS or any(depth < 0 for depth in cfg.PREFIX_DEPTHS):
        raise ValueError("PREFIX_DEPTHS 必须包含非负整数。")
    if not cfg.JOINT_WIDTHS or any(width < 2 for width in cfg.JOINT_WIDTHS):
        raise ValueError("JOINT_WIDTHS 必须包含至少为 2 的整数。")
    if cfg.HIDDEN_LAYERS < 1:
        raise ValueError("HIDDEN_LAYERS 必须至少为 1。")
    if not cfg.MODEL_SEEDS:
        raise ValueError("MODEL_SEEDS 不能为空。")
    if cfg.ONLINE_BATCH_SIZE < 1 or cfg.PROBE_SIZE < 1:
        raise ValueError("batch 与 probe 规模必须为正数。")
    if cfg.MAX_STEPS < 1 or cfg.EVAL_INTERVAL_STEPS < 1:
        raise ValueError("训练步数和评估间隔必须为正数。")
    if cfg.CHECKPOINT_INTERVAL_STEPS < 1:
        raise ValueError("CHECKPOINT_INTERVAL_STEPS 必须为正数。")
    if cfg.ACTIVATION not in {"gelu", "relu", "tanh"}:
        raise ValueError("ACTIVATION 只支持 gelu、relu 或 tanh。")
    if cfg.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE 指定 CUDA，但当前 PyTorch 看不到 GPU。")


def set_runtime_flags(cfg: Config) -> None:
    random.seed(int(cfg.DATA_SEED))
    np.random.seed(int(cfg.DATA_SEED))
    torch.manual_seed(int(cfg.DATA_SEED))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.DATA_SEED))
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.ALLOW_TF32)


def eca_once(states: torch.Tensor, rule: int) -> torch.Tensor:
    """对形状 (..., width) 的 uint8/long tensor 执行周期边界 ECA。"""
    left = torch.roll(states, shifts=1, dims=-1)
    center = states
    right = torch.roll(states, shifts=-1, dims=-1)
    neighborhood = (
        left.to(torch.long) * 4
        + center.to(torch.long) * 2
        + right.to(torch.long)
    )
    table = torch.tensor(
        [(int(rule) >> index) & 1 for index in range(8)],
        dtype=torch.uint8,
        device=states.device,
    )
    return table[neighborhood]


def evolve_eca(states: torch.Tensor, rule: int, depth: int) -> torch.Tensor:
    result = states.to(torch.uint8)
    for _ in range(int(depth)):
        result = eca_once(result, int(rule))
    return result


def half_pairwise_and(states: torch.Tensor) -> torch.Tensor:
    half = states.shape[-1] // 2
    return states[..., :half] & states[..., half:]


def half_pairwise_or(states: torch.Tensor) -> torch.Tensor:
    half = states.shape[-1] // 2
    return states[..., :half] | states[..., half:]


def make_condition_targets(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: int,
    cfg: Config,
) -> torch.Tensor:
    """返回 [4, batch, STATE_BITS]，顺序与 CONDITION_NAMES 一致。"""
    state_f = evolve_eca(u, cfg.PREFIX_RULE_F, depth)
    state_g = evolve_eca(v, cfg.PREFIX_RULE_G, depth)

    a_f = half_pairwise_and(state_f)
    b_f = half_pairwise_or(state_f)
    a_g = half_pairwise_and(state_g)
    b_g = half_pairwise_or(state_g)

    return torch.stack(
        (
            torch.cat((a_f, b_f), dim=-1),
            torch.cat((a_g, b_g), dim=-1),
            torch.cat((a_f, b_g), dim=-1),
            torch.cat((a_g, b_f), dim=-1),
        ),
        dim=0,
    ).to(torch.float32)


def assert_crossed_target_balance(condition_targets: torch.Tensor) -> None:
    """验证四格设计中的四种原子分支在两组间逐元素配平。"""
    if condition_targets.shape[0] != len(CONDITION_NAMES):
        raise RuntimeError("condition target 数量与 CONDITION_NAMES 不一致。")
    half = condition_targets.shape[-1] // 2
    shared_f, shared_g, separate_fg, separate_gf = condition_targets
    checks = {
        "A(F)": torch.equal(
            shared_f[..., :half],
            separate_fg[..., :half],
        ),
        "A(G)": torch.equal(
            shared_g[..., :half],
            separate_gf[..., :half],
        ),
        "B(F)": torch.equal(
            shared_f[..., half:],
            separate_gf[..., half:],
        ),
        "B(G)": torch.equal(
            shared_g[..., half:],
            separate_fg[..., half:],
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"四格 target 配平失败：{failed}")


def deterministic_bit_batch(
    batch_size: int,
    state_bits: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    u = torch.randint(
        0,
        2,
        (int(batch_size), int(state_bits)),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )
    v = torch.randint(
        0,
        2,
        (int(batch_size), int(state_bits)),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )
    return u, v


def make_probe(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    cpu = torch.device("cpu")
    u, v = deterministic_bit_batch(
        cfg.PROBE_SIZE,
        cfg.STATE_BITS,
        cfg.PROBE_SEED,
        cpu,
    )
    inputs = torch.cat((u, v), dim=-1).to(torch.float32)
    return inputs, torch.stack((u, v), dim=0)


def activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(x)
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"未知激活函数：{name}")


def mixed_seed(seed: int, salt: int) -> int:
    # 限制在 torch.Generator 接受的有符号 63 bit 范围内。
    value = (
        int(seed) * 6364136223846793005
        + int(salt) * 1442695040888963407
        + 0x9E3779B97F4A7C15
    )
    return value & ((1 << 63) - 1)


def initialized_linear_batch(
    model_seeds: tuple[int, ...],
    out_features: int,
    in_features: int,
    salt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bound = 1.0 / math.sqrt(float(in_features))
    weight_parts: list[torch.Tensor] = []
    bias_parts: list[torch.Tensor] = []
    for seed in model_seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed(seed, salt))
        weight = torch.empty(out_features, in_features, dtype=torch.float32)
        bias = torch.empty(out_features, dtype=torch.float32)
        weight.uniform_(-bound, bound, generator=generator)
        bias.uniform_(-bound, bound, generator=generator)
        weight_parts.append(weight)
        bias_parts.append(bias)
    return torch.stack(weight_parts, dim=0), torch.stack(bias_parts, dim=0)


class BatchedMLP(nn.Module):
    """把多个相同形状、互不共享参数的 MLP 存在 batch 维中。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int,
        model_seeds: tuple[int, ...],
        cfg: Config,
        *,
        salt_base: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = int(hidden_layers)
        self.model_count = len(model_seeds)
        self.activation_name = cfg.ACTIVATION
        self.use_layer_norm = bool(cfg.USE_LAYER_NORM)
        self.layer_norm_eps = float(cfg.LAYER_NORM_EPS)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()

        layer_input = self.input_dim
        for layer_index in range(self.hidden_layers):
            weight, bias = initialized_linear_batch(
                model_seeds,
                self.hidden_dim,
                layer_input,
                salt=salt_base + layer_index * 17,
            )
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))
            if self.use_layer_norm:
                self.norm_weights.append(nn.Parameter(torch.ones(
                    self.model_count,
                    self.hidden_dim,
                    dtype=torch.float32,
                )))
                self.norm_biases.append(nn.Parameter(torch.zeros(
                    self.model_count,
                    self.hidden_dim,
                    dtype=torch.float32,
                )))
            layer_input = self.hidden_dim

        readout_weight, readout_bias = initialized_linear_batch(
            model_seeds,
            self.output_dim,
            self.hidden_dim,
            salt=salt_base + 1009,
        )
        self.readout_weight = nn.Parameter(readout_weight)
        self.readout_bias = nn.Parameter(readout_bias)

    def _layer_norm(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(
            variance + self.layer_norm_eps
        )
        return (
            normalized * self.norm_weights[layer_index][:, None, :]
            + self.norm_biases[layer_index][:, None, :]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.model_count, -1, -1)
        if x.ndim != 3 or x.shape[0] != self.model_count:
            raise ValueError(
                "BatchedMLP 输入必须是 [batch, dim] 或 "
                "[model, batch, dim]。"
            )

        hidden = x
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            hidden = activation(hidden, self.activation_name)
            if self.use_layer_norm:
                hidden = self._layer_norm(hidden, layer_index)

        logits = torch.bmm(
            hidden,
            self.readout_weight.transpose(1, 2),
        )
        return logits + self.readout_bias[:, None, :]


class BatchedSplitMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        branch_width: int,
        output_dim_per_branch: int,
        hidden_layers: int,
        model_seeds: tuple[int, ...],
        cfg: Config,
    ):
        super().__init__()
        self.branch_a = BatchedMLP(
            input_dim,
            branch_width,
            output_dim_per_branch,
            hidden_layers,
            model_seeds,
            cfg,
            salt_base=20000,
        )
        self.branch_b = BatchedMLP(
            input_dim,
            branch_width,
            output_dim_per_branch,
            hidden_layers,
            model_seeds,
            cfg,
            salt_base=40000,
        )

    @property
    def model_count(self) -> int:
        return self.branch_a.model_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch_a(x), self.branch_b(x)), dim=-1)


def analytic_mlp_parameter_count(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_layers: int,
    use_layer_norm: bool,
) -> int:
    count = (int(input_dim) + 1) * int(hidden_dim)
    count += (int(hidden_layers) - 1) * (
        (int(hidden_dim) + 1) * int(hidden_dim)
    )
    if use_layer_norm:
        count += int(hidden_layers) * 2 * int(hidden_dim)
    count += (int(hidden_dim) + 1) * int(output_dim)
    return int(count)


def matched_split_width(joint_width: int, cfg: Config) -> tuple[int, int, int]:
    input_dim = 2 * int(cfg.STATE_BITS)
    output_dim = int(cfg.STATE_BITS)
    branch_output = output_dim // 2
    joint_count = analytic_mlp_parameter_count(
        input_dim,
        joint_width,
        output_dim,
        cfg.HIDDEN_LAYERS,
        cfg.USE_LAYER_NORM,
    )

    best_width = 1
    best_count = 0
    best_error = float("inf")
    search_max = max(4, int(joint_width) * 2)
    for candidate in range(1, search_max + 1):
        split_count = 2 * analytic_mlp_parameter_count(
            input_dim,
            candidate,
            branch_output,
            cfg.HIDDEN_LAYERS,
            cfg.USE_LAYER_NORM,
        )
        error = abs(split_count - joint_count)
        if error < best_error:
            best_width = candidate
            best_count = split_count
            best_error = error
    return best_width, joint_count, best_count


def model_layout(
    cfg: Config,
) -> tuple[tuple[str, int], tuple[int, ...], tuple[str, ...]]:
    model_seeds = tuple(
        int(seed)
        for condition in CONDITION_NAMES
        for seed in cfg.MODEL_SEEDS
    )
    model_conditions = tuple(
        condition
        for condition in CONDITION_NAMES
        for _ in cfg.MODEL_SEEDS
    )
    model_labels = tuple(
        f"{condition}/seed{seed}"
        for condition in CONDITION_NAMES
        for seed in cfg.MODEL_SEEDS
    )
    return tuple(zip(model_conditions, model_seeds)), model_seeds, model_labels


def build_model(
    architecture: str,
    joint_width: int,
    cfg: Config,
    device: torch.device,
) -> tuple[nn.Module, dict[str, int]]:
    _, repeated_seeds, _ = model_layout(cfg)
    input_dim = 2 * int(cfg.STATE_BITS)
    output_dim = int(cfg.STATE_BITS)

    if architecture == "joint":
        model = BatchedMLP(
            input_dim,
            joint_width,
            output_dim,
            cfg.HIDDEN_LAYERS,
            repeated_seeds,
            cfg,
            salt_base=1000,
        )
        branch_width = 0
        analytic_count = analytic_mlp_parameter_count(
            input_dim,
            joint_width,
            output_dim,
            cfg.HIDDEN_LAYERS,
            cfg.USE_LAYER_NORM,
        )
        matched_joint_count = analytic_count
        matched_split_count = 0
    elif architecture == "split":
        branch_width, matched_joint_count, matched_split_count = (
            matched_split_width(joint_width, cfg)
        )
        model = BatchedSplitMLP(
            input_dim,
            branch_width,
            output_dim // 2,
            cfg.HIDDEN_LAYERS,
            repeated_seeds,
            cfg,
        )
        analytic_count = matched_split_count
    else:
        raise ValueError(f"未知 architecture：{architecture}")

    model = model.to(device)
    model_count = len(repeated_seeds)
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_count % model_count != 0:
        raise RuntimeError("批量模型总参数量不能被模型数整除。")
    actual_per_model = actual_count // model_count
    if actual_per_model != analytic_count:
        raise RuntimeError(
            f"参数计数不一致：analytic={analytic_count}, "
            f"actual={actual_per_model}。"
        )

    metadata = {
        "joint_width": int(joint_width),
        "split_branch_width": int(branch_width),
        "parameter_count_per_model": int(actual_per_model),
        "matched_joint_parameter_count": int(matched_joint_count),
        "matched_split_parameter_count": int(matched_split_count),
    }
    assert_condition_paired_initialization(model, cfg)
    return model, metadata


def assert_condition_paired_initialization(
    model: nn.Module,
    cfg: Config,
) -> None:
    """同一 seed 在四个条件中必须逐元素使用相同初始参数。"""
    seed_count = len(cfg.MODEL_SEEDS)
    expected_models = len(CONDITION_NAMES) * seed_count
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim == 0 or parameter.shape[0] != expected_models:
            continue
        reshaped = parameter.detach().reshape(
            len(CONDITION_NAMES),
            seed_count,
            *parameter.shape[1:],
        )
        reference = reshaped[0]
        for condition_index in range(1, len(CONDITION_NAMES)):
            if not torch.equal(reference, reshaped[condition_index]):
                raise RuntimeError(
                    f"同 seed 条件配对初始化失败：{name}, "
                    f"condition_index={condition_index}。"
                )
        checked += 1
    if checked == 0:
        raise RuntimeError("没有找到带 model batch 维的参数，无法审计初始化。")


def targets_for_models(
    condition_targets: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    return condition_targets.repeat_interleave(
        len(cfg.MODEL_SEEDS),
        dim=0,
    )


def autocast_context(cfg: Config, device: torch.device):
    enabled = bool(cfg.USE_BF16_AUTOCAST and device.type == "cuda")
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    )


@torch.no_grad()
def evaluate_group(
    model: nn.Module,
    probe_inputs: torch.Tensor,
    probe_states: torch.Tensor,
    depth: int,
    cfg: Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    layout, _, model_labels = model_layout(cfg)
    model_count = len(layout)
    loss_sum = torch.zeros(model_count, dtype=torch.float64)
    bit_correct = torch.zeros(model_count, dtype=torch.long)
    exact_correct = torch.zeros(model_count, dtype=torch.long)
    element_count = 0
    sample_count = 0

    for start in range(0, len(probe_inputs), int(cfg.PROBE_BATCH_SIZE)):
        stop = min(len(probe_inputs), start + int(cfg.PROBE_BATCH_SIZE))
        x_batch = probe_inputs[start:stop].to(device, non_blocking=True)
        u_batch = probe_states[0, start:stop].to(device, non_blocking=True)
        v_batch = probe_states[1, start:stop].to(device, non_blocking=True)
        condition_targets = make_condition_targets(
            u_batch,
            v_batch,
            depth,
            cfg,
        )
        assert_crossed_target_balance(condition_targets)
        labels = targets_for_models(condition_targets, cfg)

        with autocast_context(cfg, device):
            logits = model(x_batch)
        logits_float = logits.float()
        losses = F.binary_cross_entropy_with_logits(
            logits_float,
            labels,
            reduction="none",
        )
        predictions = logits_float >= 0.0
        expected = labels >= 0.5

        loss_sum += losses.sum(dim=(1, 2)).cpu().to(torch.float64)
        bit_correct += (predictions == expected).sum(dim=(1, 2)).cpu()
        exact_correct += torch.all(
            predictions == expected,
            dim=2,
        ).sum(dim=1).cpu()
        element_count += int(expected.shape[1] * expected.shape[2])
        sample_count += int(expected.shape[1])

    rows: list[dict[str, Any]] = []
    for model_index, ((condition, seed), label) in enumerate(
        zip(layout, model_labels)
    ):
        rows.append({
            "model_index": int(model_index),
            "model_label": label,
            "condition": condition,
            "seed": int(seed),
            "loss": float(loss_sum[model_index].item() / element_count),
            "bit_accuracy": float(
                bit_correct[model_index].item() / element_count
            ),
            "exact_accuracy": float(
                exact_correct[model_index].item() / sample_count
            ),
            "sample_count": int(sample_count),
        })
    return rows


def mean_of(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_model_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, float]] = {}
    for condition in CONDITION_NAMES:
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "loss_mean": mean_of(selected, "loss"),
            "loss_std": float(np.std([row["loss"] for row in selected])),
            "bit_accuracy_mean": mean_of(selected, "bit_accuracy"),
            "exact_accuracy_mean": mean_of(selected, "exact_accuracy"),
        }

    shared_rows = [
        row for row in rows if row["condition"] in SHARED_CONDITIONS
    ]
    separate_rows = [
        row for row in rows if row["condition"] in SEPARATE_CONDITIONS
    ]
    shared_loss = mean_of(shared_rows, "loss")
    separate_loss = mean_of(separate_rows, "loss")
    tiny = 1e-30
    return {
        "conditions": by_condition,
        "shared": {
            "loss_mean": shared_loss,
            "bit_accuracy_mean": mean_of(shared_rows, "bit_accuracy"),
            "exact_accuracy_mean": mean_of(shared_rows, "exact_accuracy"),
        },
        "separate": {
            "loss_mean": separate_loss,
            "bit_accuracy_mean": mean_of(separate_rows, "bit_accuracy"),
            "exact_accuracy_mean": mean_of(separate_rows, "exact_accuracy"),
        },
        "separate_minus_shared_loss": float(separate_loss - shared_loss),
        "log_loss_ratio_separate_over_shared": float(
            math.log(max(separate_loss, tiny) / max(shared_loss, tiny))
        ),
        "shared_minus_separate_bit_accuracy": float(
            mean_of(shared_rows, "bit_accuracy")
            - mean_of(separate_rows, "bit_accuracy")
        ),
        "shared_minus_separate_exact_accuracy": float(
            mean_of(shared_rows, "exact_accuracy")
            - mean_of(separate_rows, "exact_accuracy")
        ),
    }


def best_rows_from_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """每个 condition/seed 取预算内 probe loss 最低的 checkpoint。"""
    best: dict[str, dict[str, Any]] = {}
    for row in history:
        label = str(row["model_label"])
        if label not in best or float(row["loss"]) < float(best[label]["loss"]):
            best[label] = dict(row)
    return [best[label] for label in sorted(best)]


def aggregate_threshold_hits(
    threshold_hits: dict[str, dict[str, int | None]],
    cfg: Config,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in cfg.LOSS_THRESHOLDS:
        key = f"{threshold:.12g}"
        threshold_row: dict[str, Any] = {}
        for group_name, condition_names in (
            ("shared", SHARED_CONDITIONS),
            ("separate", SEPARATE_CONDITIONS),
        ):
            all_values = [
                values[key]
                for label, values in threshold_hits.items()
                if label.split("/", 1)[0] in condition_names
            ]
            hits = [int(value) for value in all_values if value is not None]
            threshold_row[group_name] = {
                "hit_count": len(hits),
                "total_count": len(all_values),
                "median_examples": (
                    float(np.median(hits)) if hits else None
                ),
                "min_examples": min(hits) if hits else None,
                "max_examples": max(hits) if hits else None,
            }
        result[key] = threshold_row
    return result


def audit_late_collapse(
    final_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    *,
    ratio_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    best_by_label = {str(row["model_label"]): row for row in best_rows}
    collapses: list[dict[str, Any]] = []
    for final in final_rows:
        label = str(final["model_label"])
        best = best_by_label[label]
        ratio = float(final["loss"]) / max(float(best["loss"]), 1e-30)
        if ratio <= ratio_threshold:
            continue
        collapses.append({
            "model_label": label,
            "condition": final["condition"],
            "seed": int(final["seed"]),
            "best_step": int(best["step"]),
            "best_loss": float(best["loss"]),
            "final_loss": float(final["loss"]),
            "final_over_best_loss_ratio": float(ratio),
            "best_exact_accuracy": float(best["exact_accuracy"]),
            "final_exact_accuracy": float(final["exact_accuracy"]),
        })
    return collapses


def group_directory(
    cfg: Config,
    depth: int,
    joint_width: int,
    architecture: str,
) -> Path:
    return (
        cfg.RESULT_DIR
        / f"depth_{int(depth):02d}"
        / f"joint_width_{int(joint_width):04d}"
        / architecture
    )


def checkpoint_payload(
    signature: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    threshold_hits: dict[str, dict[str, int | None]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "signature": signature,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "threshold_hits": threshold_hits,
        "elapsed_seconds": float(elapsed_seconds),
    }


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def print_group_metrics(
    depth: int,
    joint_width: int,
    architecture: str,
    step: int,
    max_steps: int,
    rows: list[dict[str, Any]],
    steps_per_second: float,
) -> None:
    aggregate = aggregate_model_rows(rows)
    shared = aggregate["shared"]
    separate = aggregate["separate"]
    print(
        f"[k={depth} | W={joint_width} | {architecture}] "
        f"step={step:6d}/{max_steps} | "
        f"loss shared={shared['loss_mean']:.6e} "
        f"separate={separate['loss_mean']:.6e} | "
        f"gap={aggregate['separate_minus_shared_loss']:+.3e} | "
        f"exact shared={shared['exact_accuracy_mean']:.6f} "
        f"separate={separate['exact_accuracy_mean']:.6f} | "
        f"{steps_per_second:.2f} step/s",
        flush=True,
    )


def train_group(
    cfg: Config,
    signature: str,
    depth: int,
    joint_width: int,
    architecture: str,
    probe_inputs: torch.Tensor,
    probe_states: torch.Tensor,
) -> dict[str, Any]:
    run_dir = group_directory(cfg, depth, joint_width, architecture)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.jsonl"
    checkpoint_path = run_dir / "active_checkpoint.pt"

    if cfg.RESUME_COMPLETED_GROUPS and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "complete"
        ):
            print(
                f"跳过已完成组：k={depth}, W={joint_width}, "
                f"architecture={architecture}",
                flush=True,
            )
            return existing

    device = torch.device(cfg.DEVICE)
    model, model_metadata = build_model(
        architecture,
        joint_width,
        cfg,
        device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.LEARNING_RATE),
        weight_decay=float(cfg.WEIGHT_DECAY),
    )
    _, _, model_labels = model_layout(cfg)
    threshold_hits: dict[str, dict[str, int | None]] = {
        label: {f"{threshold:.12g}": None for threshold in cfg.LOSS_THRESHOLDS}
        for label in model_labels
    }
    history: list[dict[str, Any]] = []
    start_step = 0
    elapsed_before_resume = 0.0

    if cfg.RESUME_COMPLETED_GROUPS and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("signature") == signature:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_to_device(optimizer, device)
            start_step = int(checkpoint["step"])
            threshold_hits = checkpoint.get(
                "threshold_hits",
                threshold_hits,
            )
            elapsed_before_resume = float(
                checkpoint.get("elapsed_seconds", 0.0)
            )
            history = read_jsonl(history_path)
            print(
                f"恢复组：k={depth}, W={joint_width}, {architecture}, "
                f"从 step={start_step} 继续。",
                flush=True,
            )
        else:
            checkpoint_path.unlink(missing_ok=True)
            history = []

    if start_step == 0:
        history = []
        write_jsonl(history_path, history)

    model.train()
    wall_start = time.perf_counter()
    last_eval_wall = wall_start
    last_eval_step = start_step
    last_batch_losses = torch.full(
        (len(model_labels),),
        float("nan"),
    )

    for step in range(start_step + 1, int(cfg.MAX_STEPS) + 1):
        batch_seed = (
            int(cfg.DATA_SEED)
            + int(depth) * 10_000_000
            + int(step)
        )
        u, v = deterministic_bit_batch(
            cfg.ONLINE_BATCH_SIZE,
            cfg.STATE_BITS,
            batch_seed,
            device,
        )
        inputs = torch.cat((u, v), dim=-1).to(torch.float32)
        condition_targets = make_condition_targets(u, v, depth, cfg)
        assert_crossed_target_balance(condition_targets)
        labels = targets_for_models(condition_targets, cfg)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(cfg, device):
            logits = model(inputs)
            per_model_losses = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            ).mean(dim=(1, 2))
            # 参数在 model batch 维互不相连；求和可保持每个模型与单独训练
            # 完全相同的梯度尺度。
            objective = per_model_losses.sum()
        objective.backward()
        optimizer.step()
        last_batch_losses = per_model_losses.detach().float().cpu()

        should_evaluate = (
            step == 1
            or step % int(cfg.EVAL_INTERVAL_STEPS) == 0
            or step == int(cfg.MAX_STEPS)
        )
        if should_evaluate:
            current_wall = time.perf_counter()
            interval_steps = max(1, step - last_eval_step)
            steps_per_second = interval_steps / max(
                current_wall - last_eval_wall,
                1e-9,
            )
            model_rows = evaluate_group(
                model,
                probe_inputs,
                probe_states,
                depth,
                cfg,
                device,
            )
            elapsed = elapsed_before_resume + current_wall - wall_start
            examples_seen = int(step) * int(cfg.ONLINE_BATCH_SIZE)
            for model_index, row in enumerate(model_rows):
                row.update({
                    "depth": int(depth),
                    "joint_width": int(joint_width),
                    "architecture": architecture,
                    "step": int(step),
                    "examples_seen": int(examples_seen),
                    "online_batch_loss": float(
                        last_batch_losses[model_index].item()
                    ),
                    "elapsed_seconds": float(elapsed),
                })
                hits = threshold_hits[row["model_label"]]
                for threshold in cfg.LOSS_THRESHOLDS:
                    key = f"{threshold:.12g}"
                    if hits[key] is None and row["loss"] <= threshold:
                        hits[key] = int(examples_seen)
                history.append(row)

            write_jsonl(history_path, history)
            print_group_metrics(
                depth,
                joint_width,
                architecture,
                step,
                cfg.MAX_STEPS,
                model_rows,
                steps_per_second,
            )
            last_eval_wall = current_wall
            last_eval_step = step
            model.train()

        should_checkpoint = bool(
            cfg.SAVE_ACTIVE_GROUP_CHECKPOINT
            and (
                step % int(cfg.CHECKPOINT_INTERVAL_STEPS) == 0
                or step == int(cfg.MAX_STEPS)
            )
        )
        if should_checkpoint:
            elapsed = (
                elapsed_before_resume
                + time.perf_counter()
                - wall_start
            )
            torch.save(
                checkpoint_payload(
                    signature,
                    step,
                    model,
                    optimizer,
                    threshold_hits,
                    elapsed,
                ),
                checkpoint_path,
            )

    final_rows = evaluate_group(
        model,
        probe_inputs,
        probe_states,
        depth,
        cfg,
        device,
    )
    total_elapsed = (
        elapsed_before_resume + time.perf_counter() - wall_start
    )
    aggregate = aggregate_model_rows(final_rows)
    best_rows = best_rows_from_history(history)
    best_aggregate = aggregate_model_rows(best_rows)
    threshold_aggregate = aggregate_threshold_hits(threshold_hits, cfg)
    late_collapses = audit_late_collapse(final_rows, best_rows)

    summary = {
        "status": "complete",
        "signature": signature,
        "experiment_name": cfg.EXPERIMENT_NAME,
        "depth": int(depth),
        "joint_width": int(joint_width),
        "architecture": architecture,
        "model_metadata": model_metadata,
        "max_steps": int(cfg.MAX_STEPS),
        "online_batch_size": int(cfg.ONLINE_BATCH_SIZE),
        "examples_seen_per_model": int(
            cfg.MAX_STEPS * cfg.ONLINE_BATCH_SIZE
        ),
        "elapsed_seconds": float(total_elapsed),
        "final_models": final_rows,
        "aggregate": aggregate,
        "best_models": best_rows,
        "best_aggregate": best_aggregate,
        "threshold_first_hit_examples": threshold_hits,
        "threshold_aggregate": threshold_aggregate,
        "late_collapses": late_collapses,
    }
    write_json(summary_path, summary)

    if cfg.SAVE_FINAL_MODEL_STATE:
        torch.save(
            {
                "signature": signature,
                "model": model.state_dict(),
                "metadata": model_metadata,
            },
            run_dir / "final_model.pt",
        )

    checkpoint_path.unlink(missing_ok=True)
    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def flatten_group_summary(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    best_aggregate = summary.get("best_aggregate", aggregate)
    return {
        "depth": int(summary["depth"]),
        "joint_width": int(summary["joint_width"]),
        "architecture": summary["architecture"],
        "split_branch_width": int(
            summary["model_metadata"]["split_branch_width"]
        ),
        "parameter_count_per_model": int(
            summary["model_metadata"]["parameter_count_per_model"]
        ),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "shared_loss": float(aggregate["shared"]["loss_mean"]),
        "separate_loss": float(aggregate["separate"]["loss_mean"]),
        "separate_minus_shared_loss": float(
            aggregate["separate_minus_shared_loss"]
        ),
        "log_loss_ratio_separate_over_shared": float(
            aggregate["log_loss_ratio_separate_over_shared"]
        ),
        "shared_bit_accuracy": float(
            aggregate["shared"]["bit_accuracy_mean"]
        ),
        "separate_bit_accuracy": float(
            aggregate["separate"]["bit_accuracy_mean"]
        ),
        "shared_exact_accuracy": float(
            aggregate["shared"]["exact_accuracy_mean"]
        ),
        "separate_exact_accuracy": float(
            aggregate["separate"]["exact_accuracy_mean"]
        ),
        "best_shared_loss": float(
            best_aggregate["shared"]["loss_mean"]
        ),
        "best_separate_loss": float(
            best_aggregate["separate"]["loss_mean"]
        ),
        "best_log_loss_ratio_separate_over_shared": float(
            best_aggregate["log_loss_ratio_separate_over_shared"]
        ),
        "best_shared_exact_accuracy": float(
            best_aggregate["shared"]["exact_accuracy_mean"]
        ),
        "best_separate_exact_accuracy": float(
            best_aggregate["separate"]["exact_accuracy_mean"]
        ),
        "late_collapse_count": len(summary.get("late_collapses", [])),
    }


def build_interaction_rows(
    group_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (
            int(summary["depth"]),
            int(summary["joint_width"]),
            str(summary["architecture"]),
        ): summary
        for summary in group_summaries
    }
    keys = sorted({
        (int(summary["depth"]), int(summary["joint_width"]))
        for summary in group_summaries
    })
    rows: list[dict[str, Any]] = []
    for depth, width in keys:
        joint = indexed.get((depth, width, "joint"))
        split = indexed.get((depth, width, "split"))
        if joint is None or split is None:
            continue
        joint_agg = joint["aggregate"]
        split_agg = split["aggregate"]
        joint_best = joint.get("best_aggregate", joint_agg)
        split_best = split.get("best_aggregate", split_agg)
        rows.append({
            "depth": depth,
            "joint_width": width,
            "split_branch_width": int(
                split["model_metadata"]["split_branch_width"]
            ),
            "joint_loss_gap": float(
                joint_agg["separate_minus_shared_loss"]
            ),
            "split_loss_gap": float(
                split_agg["separate_minus_shared_loss"]
            ),
            "loss_gap_interaction": float(
                joint_agg["separate_minus_shared_loss"]
                - split_agg["separate_minus_shared_loss"]
            ),
            "joint_log_loss_ratio": float(
                joint_agg["log_loss_ratio_separate_over_shared"]
            ),
            "split_log_loss_ratio": float(
                split_agg["log_loss_ratio_separate_over_shared"]
            ),
            "log_loss_interaction": float(
                joint_agg["log_loss_ratio_separate_over_shared"]
                - split_agg["log_loss_ratio_separate_over_shared"]
            ),
            "joint_exact_gap": float(
                joint_agg["shared_minus_separate_exact_accuracy"]
            ),
            "split_exact_gap": float(
                split_agg["shared_minus_separate_exact_accuracy"]
            ),
            "exact_gap_interaction": float(
                joint_agg["shared_minus_separate_exact_accuracy"]
                - split_agg["shared_minus_separate_exact_accuracy"]
            ),
            "joint_best_log_loss_ratio": float(
                joint_best["log_loss_ratio_separate_over_shared"]
            ),
            "split_best_log_loss_ratio": float(
                split_best["log_loss_ratio_separate_over_shared"]
            ),
            "best_log_loss_interaction": float(
                joint_best["log_loss_ratio_separate_over_shared"]
                - split_best["log_loss_ratio_separate_over_shared"]
            ),
            "joint_late_collapse_count": len(
                joint.get("late_collapses", [])
            ),
            "split_late_collapse_count": len(
                split.get("late_collapses", [])
            ),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_plots(
    cfg: Config,
    flat_rows: list[dict[str, Any]],
    interaction_rows: list[dict[str, Any]],
) -> list[Path]:
    if not cfg.GENERATE_PLOTS:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", flush=True)
        return []

    plot_dir = cfg.RESULT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    # 图 1：Joint/Split 各自预算内达到的最佳 shared/separate loss。
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, architecture in zip(axes, ARCHITECTURES):
        for width in sorted({row["joint_width"] for row in flat_rows}):
            selected = sorted(
                (
                    row for row in flat_rows
                    if row["architecture"] == architecture
                    and row["joint_width"] == width
                ),
                key=lambda row: row["depth"],
            )
            if not selected:
                continue
            depths = [row["depth"] for row in selected]
            axis.plot(
                depths,
                [row["best_shared_loss"] for row in selected],
                marker="o",
                label=f"W={width} shared",
            )
            axis.plot(
                depths,
                [row["best_separate_loss"] for row in selected],
                marker="x",
                linestyle="--",
                label=f"W={width} separate",
            )
        axis.set_title(architecture)
        axis.set_xlabel("CA prefix depth k")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("best probe BCE within budget")
    figure.suptitle("Shared vs Separate: best checkpoint")
    figure.tight_layout()
    path = plot_dir / "shared_separate_probe_loss.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    output_paths.append(path)

    # 图 2：最关键的 Joint-Split interaction。
    figure, axis = plt.subplots(figsize=(8, 5))
    for width in sorted({row["joint_width"] for row in interaction_rows}):
        selected = sorted(
            (row for row in interaction_rows if row["joint_width"] == width),
            key=lambda row: row["depth"],
        )
        axis.plot(
            [row["depth"] for row in selected],
            [row["best_log_loss_interaction"] for row in selected],
            marker="o",
            label=f"W={width}",
        )
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axis.set_xlabel("CA prefix depth k")
    axis.set_ylabel("best-checkpoint log-loss interaction")
    axis.set_title("Reuse interaction from attainable loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = plot_dir / "reuse_log_loss_interaction.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    output_paths.append(path)

    # 图 3：interaction 热力图。
    depths = sorted({row["depth"] for row in interaction_rows})
    widths = sorted({row["joint_width"] for row in interaction_rows})
    if depths and widths:
        matrix = np.full((len(widths), len(depths)), np.nan, dtype=float)
        lookup = {
            (row["joint_width"], row["depth"]): row[
                "best_log_loss_interaction"
            ]
            for row in interaction_rows
        }
        for width_index, width in enumerate(widths):
            for depth_index, depth in enumerate(depths):
                matrix[width_index, depth_index] = lookup.get(
                    (width, depth),
                    np.nan,
                )
        figure, axis = plt.subplots(figsize=(8, 4.8))
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        axis.set_xticks(range(len(depths)), labels=depths)
        axis.set_yticks(range(len(widths)), labels=widths)
        axis.set_xlabel("CA prefix depth k")
        axis.set_ylabel("Joint width")
        axis.set_title("Best-checkpoint interaction heatmap")
        figure.colorbar(image, ax=axis, label="best interaction")
        figure.tight_layout()
        path = plot_dir / "reuse_interaction_heatmap.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        output_paths.append(path)

    return output_paths


def create_analysis_archive(cfg: Config) -> Path | None:
    if not cfg.CREATE_ANALYSIS_ARCHIVE:
        return None
    archive = cfg.RESULT_DIR.parent / f"{cfg.RESULT_DIR.name}.zip"
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as handle:
        for path in sorted(cfg.RESULT_DIR.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pt" or path.name.endswith(".tmp"):
                continue
            handle.write(path, path.relative_to(cfg.RESULT_DIR.parent))
    return archive


def run_experiment(cfg: Config) -> dict[str, Any]:
    cfg = apply_smoke_overrides(cfg)
    validate_config(cfg)
    set_runtime_flags(cfg)
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    signature = config_signature(cfg)
    write_json(cfg.RESULT_DIR / "config.json", {
        "signature": signature,
        "config": config_payload(cfg),
        "conditions": list(CONDITION_NAMES),
        "shared_conditions": list(SHARED_CONDITIONS),
        "separate_conditions": list(SEPARATE_CONDITIONS),
    })

    device = torch.device(cfg.DEVICE)
    probe_inputs, probe_states = make_probe(cfg)

    print("=== CA 共享中间表示计算压力实验 ===", flush=True)
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(f"结果目录：{cfg.RESULT_DIR}", flush=True)
    print(
        f"配置：depths={cfg.PREFIX_DEPTHS} | widths={cfg.JOINT_WIDTHS} | "
        f"seeds={cfg.MODEL_SEEDS} | hidden_layers={cfg.HIDDEN_LAYERS}",
        flush=True,
    )
    print(
        f"在线训练：batch={cfg.ONLINE_BATCH_SIZE} | steps={cfg.MAX_STEPS} | "
        f"probe={cfg.PROBE_SIZE}",
        flush=True,
    )

    group_summaries: list[dict[str, Any]] = []
    total_groups = (
        len(cfg.PREFIX_DEPTHS)
        * len(cfg.JOINT_WIDTHS)
        * len(ARCHITECTURES)
    )
    group_index = 0
    experiment_start = time.perf_counter()
    for depth in cfg.PREFIX_DEPTHS:
        for joint_width in cfg.JOINT_WIDTHS:
            split_width, joint_count, split_count = matched_split_width(
                joint_width,
                cfg,
            )
            mismatch = (split_count - joint_count) / max(joint_count, 1)
            print(
                f"\n[k={depth}, W={joint_width}] 参数匹配：joint={joint_count:,} | "
                f"split branch width={split_width} | split={split_count:,} | "
                f"差异={mismatch:+.3%}",
                flush=True,
            )
            for architecture in ARCHITECTURES:
                group_index += 1
                print(
                    f"--- group {group_index}/{total_groups}: "
                    f"k={depth}, W={joint_width}, {architecture} ---",
                    flush=True,
                )
                summary = train_group(
                    cfg,
                    signature,
                    int(depth),
                    int(joint_width),
                    architecture,
                    probe_inputs,
                    probe_states,
                )
                group_summaries.append(summary)

    flat_rows = [flatten_group_summary(summary) for summary in group_summaries]
    interaction_rows = build_interaction_rows(group_summaries)
    write_csv(cfg.RESULT_DIR / "group_summary.csv", flat_rows)
    write_json(cfg.RESULT_DIR / "group_summary.json", flat_rows)
    write_csv(cfg.RESULT_DIR / "interaction_summary.csv", interaction_rows)
    write_json(cfg.RESULT_DIR / "interaction_summary.json", interaction_rows)
    plot_paths = generate_plots(cfg, flat_rows, interaction_rows)

    design_warnings: list[str] = []
    for row in interaction_rows:
        if abs(float(row["split_log_loss_ratio"])) > 1e-5:
            design_warnings.append(
                "Split 阴性对照没有保持零 gap："
                f"k={row['depth']}, W={row['joint_width']}, "
                f"split_log_loss_ratio={row['split_log_loss_ratio']:.3e}。"
            )
        if abs(float(row["split_best_log_loss_ratio"])) > 1e-5:
            design_warnings.append(
                "Split 最佳-checkpoint 阴性对照没有保持零 gap："
                f"k={row['depth']}, W={row['joint_width']}, "
                f"split_best_log_loss_ratio="
                f"{row['split_best_log_loss_ratio']:.3e}。"
            )
    for warning in design_warnings:
        print(f"警告：{warning}", flush=True)

    total_elapsed = time.perf_counter() - experiment_start
    final_summary = {
        "status": "complete",
        "signature": signature,
        "experiment_name": cfg.EXPERIMENT_NAME,
        "total_elapsed_seconds": float(total_elapsed),
        "group_count": len(group_summaries),
        "interaction_rows": interaction_rows,
        "plots": [str(path) for path in plot_paths],
        "design_warnings": design_warnings,
        "late_collapse_count": int(sum(
            len(summary.get("late_collapses", []))
            for summary in group_summaries
        )),
        "interpretation": {
            "primary_positive_direction": (
                "log_loss_interaction > 0：Joint 中 Separate 相对 Shared 的 "
                "loss 劣势大于 Split 阴性对照。"
            ),
            "dose_response": (
                "若 reuse hypothesis 成立，interaction 应随 prefix depth "
                "增加，并在较窄 width 更明显。"
            ),
        },
    }
    write_json(cfg.RESULT_DIR / "summary.json", final_summary)
    archive = create_analysis_archive(cfg)

    print("\n=== 实验完成 ===", flush=True)
    print(f"总耗时：{total_elapsed:.1f}s", flush=True)
    print(f"汇总：{cfg.RESULT_DIR / 'summary.json'}", flush=True)
    print(
        f"interaction：{cfg.RESULT_DIR / 'interaction_summary.csv'}",
        flush=True,
    )
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)
    return final_summary


def main() -> None:
    run_experiment(Config())


if __name__ == "__main__":
    main()
