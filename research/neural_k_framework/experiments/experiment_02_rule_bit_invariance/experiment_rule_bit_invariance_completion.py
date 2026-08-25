"""
Rule-bit 反事实补全实验。

研究问题
========

给定一个已经学会 ECA 规则 f(x) 的 MLP，如果训练阶段的控制位 z 始终固定，
那么模型在未见过的另一条 z 分支上通常不会严格保持 f(x)。本实验逐步加入极少量
成对反事实样本

    (z_base, x) -> f(x)
    (1-z_base, x) -> f(x)

检验模型是否会把“z 与任务无关”补全为全空间不变性，以及这种补全是否显著受益于
已经学会的主体规则表示。

协议
====

1. Stage 1：只用固定 z 的样本训练主体规则，直到独立 probe 上完全泛化。
2. Stage 2 warm：从同一个 Stage-1 checkpoint 分叉，对每个 k 加入嵌套的 k 个
   反事实配对样本并继续训练。
3. Stage 2 cold：从相同初始化重新训练最终数据集，总优化步数与 warm 路径相同。
4. 同时运行 0->1 与 1->0 两个方向；前者的 rule-bit 权重列在 Stage 1 不接收
   梯度，后者则等价于一组额外 bias，不能混为一谈。
5. k=0 是“只训练更久”的时间对照。

脚本完全自包含，所有常用设置都在 Config 中；不使用 argparse、环境变量或外部模块。
支持：

    python experiment_rule_bit_invariance_completion.py
    %run experiment_rule_bit_invariance_completion.py
    把整个文件粘贴到 AutoDL Jupyter cell 直接运行
"""

from __future__ import annotations

import csv
import gc
import json
import math
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 配置
# =============================================================================


def script_directory() -> Path:
    """同时支持 .py、%run 与直接粘贴到 notebook cell。"""
    source = globals().get("__file__")
    if source and not str(source).startswith("<"):
        return Path(source).resolve().parent
    return Path.cwd()


class Config:
    BASE_DIR = script_directory()
    RESULT_ROOT = BASE_DIR / "results_rule_bit_invariance_completion"

    # 主实验默认同时确认一个混沌规则和一个结构不同的复杂规则。
    RULES = (30, 110)
    CA_STEPS = 1
    STATE_BITS = 30
    TRAIN_COUNT = 2_048
    PROBE_COUNT = 65_536
    CURVE_PROBE_COUNT = 8_192
    DATA_SEED = 20260816

    # 对齐早期 tiny Boolean / rule-bit 实验：31 -> 1024 x 3 -> 30。
    HIDDEN_SIZE = 1_024
    HIDDEN_LAYERS_AFTER_FIRST = 2
    DROPOUT = 0.0
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 512
    EVAL_BATCH_SIZE = 2_048

    MODEL_SEEDS = tuple(range(8))
    BASE_RULE_BITS = (0, 1)

    # binary 是实际 0/1 控制位；centered 用 -1/+1 检查方向不对称是否主要来自编码。
    RULE_BIT_ENCODINGS = ("binary", "centered")

    # 所有 k 使用同一个随机排列的前缀，形成真正的嵌套证据集。
    COUNTERFACTUAL_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

    STAGE1_STEPS = 8_000
    STAGE1_EVAL_STEPS = (0, 500, 2_000, 8_000)
    STAGE2_STEPS = 5_000
    STAGE2_EVAL_STEPS = (0, 25, 100, 500, 2_000, 5_000)

    # cold 从头训练相同最终数据集；总步数与 warm 的 Stage1+Stage2 相同。
    RUN_COLD_CONTROL = True
    COLD_COUNTERFACTUAL_COUNTS = (0, 1, 8, 64, 512)
    COLD_STEPS = STAGE1_STEPS + STAGE2_STEPS
    COLD_EVAL_STEPS = (0, 500, 2_000, 8_000, 10_000, 13_000)

    MODEL_CHUNK_SIZE = 16
    LOG_INTERVAL = 250
    REQUIRE_STAGE1_PERFECT = True
    ALLOW_TF32 = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CREATE_PLOTS = True
    CREATE_ZIP = True
    SMOKE_TEST = False


PROTOCOL_VERSION = "rule_bit_invariance_completion_v1"


@dataclass(frozen=True)
class EffectiveConfig:
    protocol_version: str
    result_root: str
    rules: tuple[int, ...]
    ca_steps: int
    state_bits: int
    train_count: int
    probe_count: int
    curve_probe_count: int
    data_seed: int
    hidden_size: int
    hidden_layers_after_first: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    eval_batch_size: int
    model_seeds: tuple[int, ...]
    base_rule_bits: tuple[int, ...]
    rule_bit_encodings: tuple[str, ...]
    counterfactual_counts: tuple[int, ...]
    stage1_steps: int
    stage1_eval_steps: tuple[int, ...]
    stage2_steps: int
    stage2_eval_steps: tuple[int, ...]
    run_cold_control: bool
    cold_counterfactual_counts: tuple[int, ...]
    cold_steps: int
    cold_eval_steps: tuple[int, ...]
    model_chunk_size: int
    log_interval: int
    require_stage1_perfect: bool
    allow_tf32: bool
    device: str
    create_plots: bool
    create_zip: bool
    smoke_test: bool


@dataclass(frozen=True)
class BaseTask:
    task_id: int
    rule: int
    base_rule_bit: int
    encoding: str
    model_seed: int


@dataclass(frozen=True)
class AdaptTask:
    task_id: int
    source_task_id: int
    mode: str
    counterfactual_count: int
    rule: int
    base_rule_bit: int
    encoding: str
    model_seed: int


def get_effective_config() -> EffectiveConfig:
    if Config.SMOKE_TEST:
        return EffectiveConfig(
            protocol_version=PROTOCOL_VERSION,
            result_root=str(Config.RESULT_ROOT / "smoke"),
            rules=(30,),
            ca_steps=1,
            state_bits=12,
            train_count=128,
            probe_count=512,
            curve_probe_count=128,
            data_seed=int(Config.DATA_SEED),
            hidden_size=64,
            hidden_layers_after_first=1,
            dropout=0.0,
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=64,
            eval_batch_size=128,
            model_seeds=(0, 1),
            base_rule_bits=(0, 1),
            rule_bit_encodings=("binary",),
            counterfactual_counts=(0, 1, 4),
            stage1_steps=50,
            stage1_eval_steps=(0, 25, 50),
            stage2_steps=30,
            stage2_eval_steps=(0, 10, 30),
            run_cold_control=True,
            cold_counterfactual_counts=(0, 1, 4),
            cold_steps=80,
            cold_eval_steps=(0, 50, 80),
            model_chunk_size=6,
            log_interval=10,
            require_stage1_perfect=False,
            allow_tf32=False,
            device=str(Config.DEVICE),
            create_plots=False,
            create_zip=False,
            smoke_test=True,
        )

    return EffectiveConfig(
        protocol_version=PROTOCOL_VERSION,
        result_root=str(Config.RESULT_ROOT),
        rules=tuple(int(value) for value in Config.RULES),
        ca_steps=int(Config.CA_STEPS),
        state_bits=int(Config.STATE_BITS),
        train_count=int(Config.TRAIN_COUNT),
        probe_count=int(Config.PROBE_COUNT),
        curve_probe_count=int(Config.CURVE_PROBE_COUNT),
        data_seed=int(Config.DATA_SEED),
        hidden_size=int(Config.HIDDEN_SIZE),
        hidden_layers_after_first=int(Config.HIDDEN_LAYERS_AFTER_FIRST),
        dropout=float(Config.DROPOUT),
        learning_rate=float(Config.LEARNING_RATE),
        weight_decay=float(Config.WEIGHT_DECAY),
        batch_size=int(Config.BATCH_SIZE),
        eval_batch_size=int(Config.EVAL_BATCH_SIZE),
        model_seeds=tuple(int(value) for value in Config.MODEL_SEEDS),
        base_rule_bits=tuple(int(value) for value in Config.BASE_RULE_BITS),
        rule_bit_encodings=tuple(str(value) for value in Config.RULE_BIT_ENCODINGS),
        counterfactual_counts=tuple(int(value) for value in Config.COUNTERFACTUAL_COUNTS),
        stage1_steps=int(Config.STAGE1_STEPS),
        stage1_eval_steps=tuple(int(value) for value in Config.STAGE1_EVAL_STEPS),
        stage2_steps=int(Config.STAGE2_STEPS),
        stage2_eval_steps=tuple(int(value) for value in Config.STAGE2_EVAL_STEPS),
        run_cold_control=bool(Config.RUN_COLD_CONTROL),
        cold_counterfactual_counts=tuple(
            int(value) for value in Config.COLD_COUNTERFACTUAL_COUNTS
        ),
        cold_steps=int(Config.COLD_STEPS),
        cold_eval_steps=tuple(int(value) for value in Config.COLD_EVAL_STEPS),
        model_chunk_size=int(Config.MODEL_CHUNK_SIZE),
        log_interval=int(Config.LOG_INTERVAL),
        require_stage1_perfect=bool(Config.REQUIRE_STAGE1_PERFECT),
        allow_tf32=bool(Config.ALLOW_TF32),
        device=str(Config.DEVICE),
        create_plots=bool(Config.CREATE_PLOTS),
        create_zip=bool(Config.CREATE_ZIP),
        smoke_test=False,
    )


def validate_config(cfg: EffectiveConfig) -> None:
    if not cfg.rules or any(rule not in range(256) for rule in cfg.rules):
        raise ValueError("RULES 必须是非空的 0..255 整数序列")
    if cfg.ca_steps <= 0 or cfg.state_bits < 3:
        raise ValueError("CA_STEPS 必须为正数，STATE_BITS 至少为 3")
    if cfg.train_count <= 0 or cfg.probe_count <= 0:
        raise ValueError("训练集和 probe 大小必须为正数")
    if cfg.train_count + cfg.probe_count > (1 << cfg.state_bits):
        raise ValueError("训练集与 probe 超过可用输入空间")
    if not 0 < cfg.curve_probe_count <= cfg.probe_count:
        raise ValueError("CURVE_PROBE_COUNT 必须位于 1..PROBE_COUNT")
    if set(cfg.base_rule_bits) != {0, 1}:
        raise ValueError("BASE_RULE_BITS 必须同时包含 0 和 1")
    if not set(cfg.rule_bit_encodings).issubset({"binary", "centered"}):
        raise ValueError("RULE_BIT_ENCODINGS 只支持 binary/centered")
    if not cfg.model_seeds or len(set(cfg.model_seeds)) != len(cfg.model_seeds):
        raise ValueError("MODEL_SEEDS 必须非空且不重复")
    for counts, name in (
        (cfg.counterfactual_counts, "COUNTERFACTUAL_COUNTS"),
        (cfg.cold_counterfactual_counts, "COLD_COUNTERFACTUAL_COUNTS"),
    ):
        if tuple(sorted(set(counts))) != counts:
            raise ValueError(f"{name} 必须严格递增且不重复")
        if any(value < 0 or value > cfg.train_count for value in counts):
            raise ValueError(f"{name} 必须位于 0..TRAIN_COUNT")
    for steps, maximum, name in (
        (cfg.stage1_eval_steps, cfg.stage1_steps, "STAGE1_EVAL_STEPS"),
        (cfg.stage2_eval_steps, cfg.stage2_steps, "STAGE2_EVAL_STEPS"),
        (cfg.cold_eval_steps, cfg.cold_steps, "COLD_EVAL_STEPS"),
    ):
        if not steps or steps[0] != 0 or steps[-1] != maximum:
            raise ValueError(f"{name} 必须从 0 开始并以对应最大步数结束")
        if tuple(sorted(set(steps))) != steps:
            raise ValueError(f"{name} 必须严格递增且不重复")
    if cfg.dropout != 0.0:
        raise ValueError("并行独立模型实验固定要求 DROPOUT=0.0")
    if cfg.hidden_size <= 0 or cfg.hidden_layers_after_first < 0:
        raise ValueError("网络宽度和层数非法")
    if cfg.batch_size <= 0 or cfg.eval_batch_size <= 0:
        raise ValueError("batch size 必须为正数")
    if cfg.model_chunk_size <= 0:
        raise ValueError("MODEL_CHUNK_SIZE 必须为正数")


# =============================================================================
# 通用工具
# =============================================================================


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_torch(cfg: EffectiveConfig) -> torch.device:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 未检测到 CUDA")
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    if device.type == "cuda":
        print(f"设备：cuda")
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    else:
        print("设备：cpu")
    return device


def sample_unique_values(
    count: int,
    width: int,
    seed: int,
    forbidden: set[int] | None = None,
) -> np.ndarray:
    if width > 63:
        raise ValueError("唯一整数采样只支持 width <= 63")
    forbidden = forbidden or set()
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    ordered: list[int] = []
    upper = 1 << width
    while len(selected) < count:
        need = count - len(selected)
        candidates = rng.integers(
            0, upper, size=max(need * 2, 256), dtype=np.uint64
        )
        for raw in candidates.tolist():
            value = int(raw)
            if value not in forbidden and value not in selected:
                selected.add(value)
                ordered.append(value)
                if len(selected) == count:
                    break
    return np.asarray(ordered, dtype=np.uint64)


def integers_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def apply_eca(inputs: np.ndarray, rules: Sequence[int], steps: int) -> np.ndarray:
    """返回 [rule, sample, position]，采用周期边界。"""
    states = np.broadcast_to(
        np.asarray(inputs, dtype=np.uint8)[None, :, :],
        (len(rules), inputs.shape[0], inputs.shape[1]),
    ).copy()
    rule_array = np.asarray(rules, dtype=np.uint16)
    for _ in range(steps):
        left = np.roll(states, 1, axis=2)
        right = np.roll(states, -1, axis=2)
        neighborhood = left * 4 + states * 2 + right
        states = ((rule_array[:, None, None] >> neighborhood) & 1).astype(np.uint8)
    return states


def encode_rule_bits(bits: torch.Tensor, encoding: str) -> torch.Tensor:
    values = bits.to(torch.float32)
    if encoding == "binary":
        return values
    if encoding == "centered":
        return values * 2.0 - 1.0
    raise ValueError(f"未知 rule-bit 编码：{encoding}")


# =============================================================================
# 参数完全独立的批量 MLP
# =============================================================================


def linear_template(
    in_features: int,
    out_features: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.empty(out_features, in_features, dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    bound = 1.0 / math.sqrt(in_features)
    bias = torch.empty(out_features, dtype=torch.float32)
    bias.uniform_(-bound, bound, generator=generator)
    return weight, bias


def make_seed_template(
    seed: int,
    input_size: int,
    output_size: int,
    hidden_size: int,
    hidden_layers_after_first: int,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    ln_weights: list[torch.Tensor] = []
    ln_biases: list[torch.Tensor] = []
    block_inputs = [input_size] + [hidden_size] * hidden_layers_after_first
    for in_features in block_inputs:
        weight, bias = linear_template(in_features, hidden_size, generator)
        weights.append(weight)
        biases.append(bias)
        ln_weights.append(torch.ones(hidden_size, dtype=torch.float32))
        ln_biases.append(torch.zeros(hidden_size, dtype=torch.float32))
    output_weight, output_bias = linear_template(hidden_size, output_size, generator)
    return {
        "weights": weights,
        "biases": biases,
        "ln_weights": ln_weights,
        "ln_biases": ln_biases,
        "output_weight": output_weight,
        "output_bias": output_bias,
    }


class BatchedIndependentMLP(nn.Module):
    """把多个参数完全独立的 MLP 堆叠为一个 GPU 批量。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layers_after_first: int,
        model_seeds: Sequence[int],
    ) -> None:
        super().__init__()
        self.ensemble_size = len(model_seeds)
        self.hidden_size = int(hidden_size)
        templates = {
            seed: make_seed_template(
                seed,
                input_size,
                output_size,
                hidden_size,
                hidden_layers_after_first,
            )
            for seed in sorted(set(int(value) for value in model_seeds))
        }
        block_count = 1 + hidden_layers_after_first
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.ln_weights = nn.ParameterList()
        self.ln_biases = nn.ParameterList()
        for block in range(block_count):
            self.weights.append(nn.Parameter(torch.stack([
                templates[int(seed)]["weights"][block]  # type: ignore[index]
                for seed in model_seeds
            ])))
            self.biases.append(nn.Parameter(torch.stack([
                templates[int(seed)]["biases"][block]  # type: ignore[index]
                for seed in model_seeds
            ])))
            self.ln_weights.append(nn.Parameter(torch.stack([
                templates[int(seed)]["ln_weights"][block]  # type: ignore[index]
                for seed in model_seeds
            ])))
            self.ln_biases.append(nn.Parameter(torch.stack([
                templates[int(seed)]["ln_biases"][block]  # type: ignore[index]
                for seed in model_seeds
            ])))
        self.output_weight = nn.Parameter(torch.stack([
            templates[int(seed)]["output_weight"]  # type: ignore[arg-type]
            for seed in model_seeds
        ]))
        self.output_bias = nn.Parameter(torch.stack([
            templates[int(seed)]["output_bias"]  # type: ignore[arg-type]
            for seed in model_seeds
        ]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            hidden = inputs.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        elif inputs.ndim == 3:
            hidden = inputs
        else:
            raise ValueError("输入必须为 [B,D] 或 [E,B,D]")
        for weight, bias, ln_weight, ln_bias in zip(
            self.weights, self.biases, self.ln_weights, self.ln_biases
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None, :]
            hidden = F.gelu(hidden)
            hidden = F.layer_norm(hidden, (self.hidden_size,))
            hidden = hidden * ln_weight[:, None, :] + ln_bias[:, None, :]
        return (
            torch.bmm(hidden, self.output_weight.transpose(1, 2))
            + self.output_bias[:, None, :]
        )


def independent_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return losses.mean(dim=(1, 2)).sum()


def snapshot_members(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def load_selected_members(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
    source_indices: Sequence[int],
) -> None:
    device = next(model.parameters()).device
    index = torch.as_tensor(source_indices, dtype=torch.long)
    selected = {
        name: value.index_select(0, index).to(device)
        for name, value in snapshot.items()
    }
    model.load_state_dict(selected)


# =============================================================================
# 数据与评估
# =============================================================================


def build_base_tasks(cfg: EffectiveConfig) -> list[BaseTask]:
    tasks: list[BaseTask] = []
    for encoding in cfg.rule_bit_encodings:
        for rule in cfg.rules:
            for base_bit in cfg.base_rule_bits:
                for seed in cfg.model_seeds:
                    tasks.append(BaseTask(len(tasks), rule, base_bit, encoding, seed))
    return tasks


def build_adapt_tasks(
    base_tasks: Sequence[BaseTask],
    counts: Sequence[int],
    mode: str,
) -> list[AdaptTask]:
    tasks: list[AdaptTask] = []
    for base in base_tasks:
        for count in counts:
            tasks.append(AdaptTask(
                task_id=len(tasks),
                source_task_id=base.task_id,
                mode=mode,
                counterfactual_count=int(count),
                rule=base.rule,
                base_rule_bit=base.base_rule_bit,
                encoding=base.encoding,
                model_seed=base.model_seed,
            ))
    return tasks


def task_rule_indices(
    tasks: Sequence[BaseTask | AdaptTask], rules: Sequence[int], device: torch.device
) -> torch.Tensor:
    lookup = {rule: index for index, rule in enumerate(rules)}
    return torch.tensor(
        [lookup[task.rule] for task in tasks], dtype=torch.long, device=device
    )


def task_feature_values(
    tasks: Sequence[BaseTask | AdaptTask],
    flipped: bool,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for task in tasks:
        bit = 1 - task.base_rule_bit if flipped else task.base_rule_bit
        scalar = float(bit) if task.encoding == "binary" else float(bit * 2 - 1)
        values.append(scalar)
    return torch.tensor(values, dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate_branch_pair(
    model: BatchedIndependentMLP,
    tasks: Sequence[BaseTask | AdaptTask],
    states: torch.Tensor,
    targets_by_rule: torch.Tensor,
    rules: Sequence[int],
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    ensemble_size = len(tasks)
    sample_count, output_size = states.shape
    device = states.device
    rule_indices = task_rule_indices(tasks, rules, device)
    base_features = task_feature_values(tasks, False, device)
    flip_features = task_feature_values(tasks, True, device)

    base_loss = torch.zeros(ensemble_size, dtype=torch.float64, device=device)
    flip_loss = torch.zeros_like(base_loss)
    base_bit_errors = torch.zeros(ensemble_size, dtype=torch.int64, device=device)
    flip_bit_errors = torch.zeros_like(base_bit_errors)
    base_sample_errors = torch.zeros_like(base_bit_errors)
    flip_sample_errors = torch.zeros_like(base_bit_errors)
    branch_bit_disagreements = torch.zeros_like(base_bit_errors)
    branch_sample_disagreements = torch.zeros_like(base_bit_errors)
    abs_logit_delta = torch.zeros_like(base_loss)
    abs_probability_delta = torch.zeros_like(base_loss)

    for start in range(0, sample_count, batch_size):
        end = min(start + batch_size, sample_count)
        batch_states = states[start:end]
        expanded_states = batch_states.unsqueeze(0).expand(ensemble_size, -1, -1)
        base_inputs = torch.cat(
            [base_features[:, None, None].expand(-1, end - start, 1), expanded_states],
            dim=2,
        )
        flip_inputs = torch.cat(
            [flip_features[:, None, None].expand(-1, end - start, 1), expanded_states],
            dim=2,
        )
        # 先选择每个独立模型对应的规则，再切样本维；不能把两个高级索引
        # 同时写进方括号，否则 PyTorch 会尝试逐元素配对模型维和样本维。
        targets = targets_by_rule.index_select(0, rule_indices)[:, start:end]
        base_logits = model(base_inputs)
        flip_logits = model(flip_inputs)
        base_losses = F.binary_cross_entropy_with_logits(
            base_logits, targets, reduction="none"
        )
        flip_losses = F.binary_cross_entropy_with_logits(
            flip_logits, targets, reduction="none"
        )
        base_predictions = base_logits >= 0
        flip_predictions = flip_logits >= 0
        target_bool = targets >= 0.5
        base_wrong = base_predictions != target_bool
        flip_wrong = flip_predictions != target_bool
        disagreement = base_predictions != flip_predictions

        base_loss += base_losses.sum(dim=(1, 2), dtype=torch.float64)
        flip_loss += flip_losses.sum(dim=(1, 2), dtype=torch.float64)
        base_bit_errors += base_wrong.sum(dim=(1, 2))
        flip_bit_errors += flip_wrong.sum(dim=(1, 2))
        base_sample_errors += base_wrong.any(dim=2).sum(dim=1)
        flip_sample_errors += flip_wrong.any(dim=2).sum(dim=1)
        branch_bit_disagreements += disagreement.sum(dim=(1, 2))
        branch_sample_disagreements += disagreement.any(dim=2).sum(dim=1)
        abs_logit_delta += (
            base_logits - flip_logits
        ).abs().sum(dim=(1, 2), dtype=torch.float64)
        abs_probability_delta += (
            torch.sigmoid(base_logits) - torch.sigmoid(flip_logits)
        ).abs().sum(dim=(1, 2), dtype=torch.float64)

    bit_total = sample_count * output_size
    rule_column_norm = model.weights[0][:, :, 0].norm(dim=1).detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        base_bits = int(base_bit_errors[index].item())
        flip_bits = int(flip_bit_errors[index].item())
        disagree_bits = int(branch_bit_disagreements[index].item())
        rows.append({
            "task_id": task.task_id,
            "rule": task.rule,
            "base_rule_bit": task.base_rule_bit,
            "flipped_rule_bit": 1 - task.base_rule_bit,
            "encoding": task.encoding,
            "model_seed": task.model_seed,
            "probe_samples": sample_count,
            "base_mean_bit_bce": float((base_loss[index] / bit_total).item()),
            "flip_mean_bit_bce": float((flip_loss[index] / bit_total).item()),
            "base_bit_errors": base_bits,
            "flip_bit_errors": flip_bits,
            "base_bit_accuracy": 1.0 - base_bits / bit_total,
            "flip_bit_accuracy": 1.0 - flip_bits / bit_total,
            "base_exact_errors": int(base_sample_errors[index].item()),
            "flip_exact_errors": int(flip_sample_errors[index].item()),
            "base_exact_accuracy": 1.0 - base_sample_errors[index].item() / sample_count,
            "flip_exact_accuracy": 1.0 - flip_sample_errors[index].item() / sample_count,
            "branch_bit_disagreements": disagree_bits,
            "branch_sample_disagreements": int(
                branch_sample_disagreements[index].item()
            ),
            "branch_bit_agreement": 1.0 - disagree_bits / bit_total,
            "branch_exact_agreement": 1.0
            - branch_sample_disagreements[index].item() / sample_count,
            "mean_abs_logit_delta": float((abs_logit_delta[index] / bit_total).item()),
            "mean_abs_probability_delta": float(
                (abs_probability_delta[index] / bit_total).item()
            ),
            "rule_column_l2": float(rule_column_norm[index]),
            "strict_base_success": base_bits == 0,
            "strict_flip_success": flip_bits == 0,
            "strict_branch_invariance": disagree_bits == 0,
        })
    return rows


def print_eval_summary(label: str, rows: Sequence[dict[str, Any]]) -> None:
    base_accuracy = np.median([row["base_bit_accuracy"] for row in rows])
    flip_accuracy = np.median([row["flip_bit_accuracy"] for row in rows])
    agreement = np.median([row["branch_bit_agreement"] for row in rows])
    strict = sum(
        bool(row["strict_flip_success"] and row["strict_branch_invariance"])
        for row in rows
    )
    print(
        f"[评估] {label} | base_bit={base_accuracy:.9f} | "
        f"flip_bit={flip_accuracy:.9f} | branch_agree={agreement:.9f} | "
        f"strict={strict}/{len(rows)}"
    )


# =============================================================================
# 训练
# =============================================================================


def make_optimizer(model: nn.Module, cfg: EffectiveConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


def train_stage1(
    cfg: EffectiveConfig,
    device: torch.device,
    tasks: Sequence[BaseTask],
    train_states: torch.Tensor,
    train_targets_by_rule: torch.Tensor,
    probe_states: torch.Tensor,
    probe_targets_by_rule: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    model = BatchedIndependentMLP(
        cfg.state_bits + 1,
        cfg.state_bits,
        cfg.hidden_size,
        cfg.hidden_layers_after_first,
        [task.model_seed for task in tasks],
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    rule_indices = task_rule_indices(tasks, cfg.rules, device)
    base_features = task_feature_values(tasks, False, device)
    generator = torch.Generator(device=device).manual_seed(cfg.data_seed + 101)
    eval_steps = set(cfg.stage1_eval_steps)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    interval_started = started

    for step in range(cfg.stage1_steps + 1):
        if step in eval_steps:
            probe_count = (
                cfg.probe_count if step == cfg.stage1_steps else cfg.curve_probe_count
            )
            checkpoint_rows = evaluate_branch_pair(
                model,
                tasks,
                probe_states[:probe_count],
                probe_targets_by_rule[:, :probe_count],
                cfg.rules,
                cfg.eval_batch_size,
            )
            for row in checkpoint_rows:
                row.update({
                    "stage": "stage1",
                    "mode": "base_only",
                    "counterfactual_count": 0,
                    "relative_step": step,
                    "total_optimization_step": step,
                })
            rows.extend(checkpoint_rows)
            print_eval_summary(f"Stage1 step={step:,}", checkpoint_rows)
            interval_started = time.perf_counter()
        if step == cfg.stage1_steps:
            break

        model.train()
        indices = torch.randint(
            0,
            cfg.train_count,
            (cfg.batch_size,),
            generator=generator,
            device=device,
        )
        batch_states = train_states[indices]
        expanded_states = batch_states.unsqueeze(0).expand(len(tasks), -1, -1)
        inputs = torch.cat(
            [
                base_features[:, None, None].expand(-1, cfg.batch_size, 1),
                expanded_states,
            ],
            dim=2,
        )
        targets = train_targets_by_rule.index_select(0, rule_indices)[:, indices]
        optimizer.zero_grad(set_to_none=True)
        loss = independent_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

        completed = step + 1
        if completed % cfg.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - interval_started
            speed = cfg.log_interval / max(elapsed, 1e-9)
            print(
                f"Stage1 step={completed:,}/{cfg.stage1_steps:,} | "
                f"mean_loss={loss.item() / len(tasks):.6e} | "
                f"steps/s={speed:.2f} | model-steps/s={speed * len(tasks):.1f}"
            )
            interval_started = time.perf_counter()

    final_rows = [row for row in rows if row["relative_step"] == cfg.stage1_steps]
    failed = [row for row in final_rows if not row["strict_base_success"]]
    if failed:
        message = (
            f"Stage1 有 {len(failed)}/{len(final_rows)} 个模型未在完整 probe 上零错误。"
        )
        if cfg.require_stage1_perfect:
            raise RuntimeError(message + " 请增加 TRAIN_COUNT 或 STAGE1_STEPS。")
        print("警告：" + message)

    snapshot = snapshot_members(model)
    print(f"Stage1 完成，耗时 {time.perf_counter() - started:.1f}s")
    del model, optimizer
    cleanup_cuda()
    return snapshot, rows


def build_adaptation_batch(
    tasks: Sequence[AdaptTask],
    cfg: EffectiveConfig,
    train_states: torch.Tensor,
    train_targets_by_rule: torch.Tensor,
    counterfactual_order: torch.Tensor,
    source_generators: dict[int, torch.Generator],
    rules: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = train_states.device
    ensemble_size = len(tasks)
    # 同一个 Stage-1 来源模型的所有 k 分支共用随机数。这样 k 之间、以及
    # warm/cold 在共同训练步范围内的差异不会被不同 minibatch 随机流混淆。
    uniform_by_source: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for source_task_id in sorted({task.source_task_id for task in tasks}):
        uniform_by_source[source_task_id] = tuple(torch.rand(
            (cfg.batch_size,),
            generator=source_generators[source_task_id],
            device=device,
        ) for _ in range(3))  # type: ignore[assignment]
    type_uniform = torch.stack([
        uniform_by_source[task.source_task_id][0] for task in tasks
    ])
    base_uniform = torch.stack([
        uniform_by_source[task.source_task_id][1] for task in tasks
    ])
    counter_uniform = torch.stack([
        uniform_by_source[task.source_task_id][2] for task in tasks
    ])

    counter_counts = torch.tensor(
        [task.counterfactual_count for task in tasks],
        dtype=torch.long,
        device=device,
    )
    counter_probability = counter_counts.to(torch.float32) / (
        cfg.train_count + counter_counts
    )
    is_counterfactual = type_uniform < counter_probability[:, None]
    base_state_indices = torch.floor(base_uniform * cfg.train_count).to(torch.long)
    safe_counter_counts = counter_counts.clamp_min(1)
    counter_offsets = torch.floor(
        counter_uniform * safe_counter_counts[:, None]
    ).to(torch.long)
    counter_state_indices = counterfactual_order[counter_offsets]
    state_indices = torch.where(
        is_counterfactual, counter_state_indices, base_state_indices
    )
    states = train_states[state_indices]

    base_bits = torch.tensor(
        [task.base_rule_bit for task in tasks], dtype=torch.long, device=device
    )[:, None]
    bits = torch.where(is_counterfactual, 1 - base_bits, base_bits)
    encoded = torch.empty_like(bits, dtype=torch.float32)
    for index, task in enumerate(tasks):
        encoded[index] = encode_rule_bits(bits[index], task.encoding)
    inputs = torch.cat([encoded[:, :, None], states], dim=2)

    rule_indices = task_rule_indices(tasks, rules, device)
    targets = train_targets_by_rule[rule_indices[:, None], state_indices]
    return inputs, targets


def train_adaptation_mode(
    cfg: EffectiveConfig,
    device: torch.device,
    mode: str,
    tasks: Sequence[AdaptTask],
    stage1_snapshot: dict[str, torch.Tensor],
    train_states: torch.Tensor,
    train_targets_by_rule: torch.Tensor,
    probe_states: torch.Tensor,
    probe_targets_by_rule: torch.Tensor,
    counterfactual_order: torch.Tensor,
    result_root: Path,
    accumulated_rows: list[dict[str, Any]],
) -> None:
    if mode == "warm":
        max_steps = cfg.stage2_steps
        eval_steps = set(cfg.stage2_eval_steps)
    elif mode == "cold":
        max_steps = cfg.cold_steps
        eval_steps = set(cfg.cold_eval_steps)
    else:
        raise ValueError(f"未知训练模式：{mode}")

    task_chunks = list(chunks(list(tasks), cfg.model_chunk_size))
    for chunk_index, task_chunk_raw in enumerate(task_chunks, start=1):
        task_chunk = list(task_chunk_raw)
        print(
            f"\n--- {mode} chunk {chunk_index}/{len(task_chunks)} | "
            f"models={len(task_chunk)} ---"
        )
        model = BatchedIndependentMLP(
            cfg.state_bits + 1,
            cfg.state_bits,
            cfg.hidden_size,
            cfg.hidden_layers_after_first,
            [task.model_seed for task in task_chunk],
        ).to(device)
        if mode == "warm":
            load_selected_members(
                model,
                stage1_snapshot,
                [task.source_task_id for task in task_chunk],
            )
        optimizer = make_optimizer(model, cfg)
        # 每个 Stage-1 来源模型有一条可复现的共同随机流。warm/cold 故意
        # 使用相同 seed；即便同一来源因 chunk 边界被拆开，随机流仍然一致。
        source_generators = {
            source_task_id: torch.Generator(device=device).manual_seed(
                cfg.data_seed + 30_000 + source_task_id * 1_000_003
            )
            for source_task_id in {task.source_task_id for task in task_chunk}
        }
        interval_started = time.perf_counter()

        for step in range(max_steps + 1):
            if step in eval_steps:
                final = step == max_steps
                probe_count = cfg.probe_count if final else cfg.curve_probe_count
                checkpoint_rows = evaluate_branch_pair(
                    model,
                    task_chunk,
                    probe_states[:probe_count],
                    probe_targets_by_rule[:, :probe_count],
                    cfg.rules,
                    cfg.eval_batch_size,
                )
                for row, task in zip(checkpoint_rows, task_chunk):
                    row.update({
                        "stage": "stage2",
                        "mode": mode,
                        "source_task_id": task.source_task_id,
                        "counterfactual_count": task.counterfactual_count,
                        "counterfactual_fraction": task.counterfactual_count
                        / (cfg.train_count + task.counterfactual_count),
                        "relative_step": step,
                        "total_optimization_step": (
                            cfg.stage1_steps + step if mode == "warm" else step
                        ),
                    })
                accumulated_rows.extend(checkpoint_rows)
                print_eval_summary(
                    f"{mode} step={step:,}"
                    + (" final" if final else ""),
                    checkpoint_rows,
                )
                write_csv(result_root / "adaptation_metrics.csv", accumulated_rows)
                interval_started = time.perf_counter()
            if step == max_steps:
                break

            model.train()
            inputs, targets = build_adaptation_batch(
                task_chunk,
                cfg,
                train_states,
                train_targets_by_rule,
                counterfactual_order,
                source_generators,
                cfg.rules,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = independent_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()

            completed = step + 1
            if completed % cfg.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - interval_started
                speed = cfg.log_interval / max(elapsed, 1e-9)
                print(
                    f"{mode} step={completed:,}/{max_steps:,} | "
                    f"mean_loss={loss.item() / len(task_chunk):.6e} | "
                    f"steps/s={speed:.2f} | model-steps/s={speed * len(task_chunk):.1f}"
                )
                interval_started = time.perf_counter()

        del model, optimizer
        cleanup_cuda()


# =============================================================================
# 汇总与绘图
# =============================================================================


def summarize_final_rows(
    rows: Sequence[dict[str, Any]], cfg: EffectiveConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_rows = []
    for row in rows:
        maximum = cfg.stage2_steps if row["mode"] == "warm" else cfg.cold_steps
        if int(row["relative_step"]) == maximum:
            final_rows.append(row)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        key = (
            row["mode"],
            row["encoding"],
            int(row["rule"]),
            int(row["base_rule_bit"]),
            int(row["counterfactual_count"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        mode, encoding, rule, base_bit, count = key
        strict_success = [
            bool(row["strict_flip_success"] and row["strict_branch_invariance"])
            for row in group
        ]
        summary_rows.append({
            "mode": mode,
            "encoding": encoding,
            "rule": rule,
            "base_rule_bit": base_bit,
            "counterfactual_count": count,
            "counterfactual_fraction": count / (cfg.train_count + count),
            "seed_count": len(group),
            "strict_success_count": int(sum(strict_success)),
            "strict_success_rate": float(np.mean(strict_success)),
            "median_flip_bit_accuracy": float(np.median([
                row["flip_bit_accuracy"] for row in group
            ])),
            "median_flip_exact_accuracy": float(np.median([
                row["flip_exact_accuracy"] for row in group
            ])),
            "median_branch_bit_agreement": float(np.median([
                row["branch_bit_agreement"] for row in group
            ])),
            "median_branch_exact_agreement": float(np.median([
                row["branch_exact_agreement"] for row in group
            ])),
            "median_abs_logit_delta": float(np.median([
                row["mean_abs_logit_delta"] for row in group
            ])),
            "median_rule_column_l2": float(np.median([
                row["rule_column_l2"] for row in group
            ])),
        })

    minimum_k: list[dict[str, Any]] = []
    by_condition: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_condition[
            (row["mode"], row["encoding"], row["rule"], row["base_rule_bit"])
        ].append(row)
    for key, group in sorted(by_condition.items()):
        perfect = [row for row in group if row["strict_success_rate"] == 1.0]
        minimum_k.append({
            "mode": key[0],
            "encoding": key[1],
            "rule": key[2],
            "base_rule_bit": key[3],
            "minimum_k_all_seeds_strict": (
                min(row["counterfactual_count"] for row in perfect)
                if perfect
                else None
            ),
        })

    summary = {
        "protocol_version": cfg.protocol_version,
        "interpretation": {
            "supports_invariance_completion_if": (
                "warm 在很小 k 上使 flipped probe 与 base probe 全空间一致，"
                "且 k=0 不出现同样改善"
            ),
            "supports_representation_reuse_if": (
                "warm 达到相同严格成功率所需 k 或步数明显小于 cold"
            ),
            "does_not_prove": (
                "即使成功，也只证明给定架构与优化协议下存在强不变性补全偏置，"
                "不证明 SGD 总能恢复语义上真实的无关变量"
            ),
        },
        "minimum_k": minimum_k,
        "final_group_count": len(summary_rows),
        "final_model_count": len(final_rows),
    }
    return summary_rows, summary


def save_plots(summary_rows: Sequence[dict[str, Any]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return

    warm = [row for row in summary_rows if row["mode"] == "warm"]
    if not warm:
        return
    conditions = sorted({
        (row["encoding"], row["rule"], row["base_rule_bit"])
        for row in warm
    })
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for encoding, rule, base_bit in conditions:
        group = sorted(
            [
                row for row in warm
                if row["encoding"] == encoding
                and row["rule"] == rule
                and row["base_rule_bit"] == base_bit
            ],
            key=lambda row: row["counterfactual_count"],
        )
        x = [row["counterfactual_count"] + 1 for row in group]
        label = f"{encoding} R{rule} {base_bit}->{1-base_bit}"
        axes[0].plot(
            x,
            [row["median_flip_exact_accuracy"] for row in group],
            marker="o",
            label=label,
        )
        axes[1].plot(
            x,
            [1.0 - row["median_branch_bit_agreement"] for row in group],
            marker="o",
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
        axis.set_xlabel("counterfactual count k (+1 for log axis)")
    axes[0].set_ylabel("median flipped exact accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("median branch bit disagreement")
    axes[1].set_yscale("symlog", linthresh=1e-9)
    axes[0].legend(fontsize=8)
    figure.suptitle("Rule-bit invariance completion dose response")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def package_results(result_root: Path) -> Path:
    archive = result_root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(result_root.parent))
    return archive


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    cfg = get_effective_config()
    validate_config(cfg)
    result_root = Path(cfg.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    save_json(result_root / "config.json", asdict(cfg))
    device = configure_torch(cfg)

    print("\n=== Rule-bit 反事实补全实验 ===")
    print(
        f"规则={list(cfg.rules)} | CA steps={cfg.ca_steps} | state={cfg.state_bits} bits"
    )
    print(
        f"网络={cfg.state_bits + 1} -> {cfg.hidden_size} x "
        f"{1 + cfg.hidden_layers_after_first} -> {cfg.state_bits} | "
        f"seeds={len(cfg.model_seeds)} | dropout={cfg.dropout}"
    )
    print(
        f"训练样本={cfg.train_count:,} | probe={cfg.probe_count:,} | "
        f"k={list(cfg.counterfactual_counts)}"
    )
    print(f"结果目录：{result_root}")

    train_values = sample_unique_values(
        cfg.train_count, cfg.state_bits, cfg.data_seed
    )
    forbidden = set(int(value) for value in train_values.tolist())
    probe_values = sample_unique_values(
        cfg.probe_count, cfg.state_bits, cfg.data_seed + 1, forbidden
    )
    train_bits_np = integers_to_bits(train_values, cfg.state_bits)
    probe_bits_np = integers_to_bits(probe_values, cfg.state_bits)
    train_targets_np = apply_eca(train_bits_np, cfg.rules, cfg.ca_steps)
    probe_targets_np = apply_eca(probe_bits_np, cfg.rules, cfg.ca_steps)

    rng = np.random.default_rng(cfg.data_seed + 2)
    counter_order_np = rng.permutation(cfg.train_count).astype(np.int64)
    np.savez_compressed(
        result_root / "dataset_manifest.npz",
        train_values=train_values,
        probe_values=probe_values,
        counterfactual_order=counter_order_np,
    )

    train_states = torch.from_numpy(train_bits_np).to(device=device, dtype=torch.float32)
    probe_states = torch.from_numpy(probe_bits_np).to(device=device, dtype=torch.float32)
    train_targets = torch.from_numpy(train_targets_np).to(
        device=device, dtype=torch.float32
    )
    probe_targets = torch.from_numpy(probe_targets_np).to(
        device=device, dtype=torch.float32
    )
    counter_order = torch.from_numpy(counter_order_np).to(device=device)

    base_tasks = build_base_tasks(cfg)
    write_csv(result_root / "base_task_manifest.csv", [asdict(task) for task in base_tasks])
    print(f"Stage1 独立模型数：{len(base_tasks)}")
    stage1_snapshot, stage1_rows = train_stage1(
        cfg,
        device,
        base_tasks,
        train_states,
        train_targets,
        probe_states,
        probe_targets,
    )
    write_csv(result_root / "stage1_metrics.csv", stage1_rows)

    adaptation_rows: list[dict[str, Any]] = []
    warm_tasks = build_adapt_tasks(base_tasks, cfg.counterfactual_counts, "warm")
    write_csv(result_root / "warm_task_manifest.csv", [asdict(task) for task in warm_tasks])
    train_adaptation_mode(
        cfg,
        device,
        "warm",
        warm_tasks,
        stage1_snapshot,
        train_states,
        train_targets,
        probe_states,
        probe_targets,
        counter_order,
        result_root,
        adaptation_rows,
    )

    if cfg.run_cold_control:
        cold_tasks = build_adapt_tasks(
            base_tasks, cfg.cold_counterfactual_counts, "cold"
        )
        write_csv(
            result_root / "cold_task_manifest.csv", [asdict(task) for task in cold_tasks]
        )
        train_adaptation_mode(
            cfg,
            device,
            "cold",
            cold_tasks,
            stage1_snapshot,
            train_states,
            train_targets,
            probe_states,
            probe_targets,
            counter_order,
            result_root,
            adaptation_rows,
        )

    write_csv(result_root / "adaptation_metrics.csv", adaptation_rows)
    dose_rows, summary = summarize_final_rows(adaptation_rows, cfg)
    write_csv(result_root / "dose_response_summary.csv", dose_rows)
    save_json(result_root / "summary.json", summary)
    if cfg.create_plots:
        save_plots(dose_rows, result_root / "dose_response.png")

    print("\n=== 实验完成 ===")
    for row in summary["minimum_k"]:
        print(
            f"{row['mode']:>4} | {row['encoding']:<8} | rule={row['rule']:3d} | "
            f"{row['base_rule_bit']}->{1-row['base_rule_bit']} | "
            f"all-seed strict minimum k={row['minimum_k_all_seeds_strict']}"
        )
    print(f"逐模型曲线：{result_root / 'adaptation_metrics.csv'}")
    print(f"剂量汇总：{result_root / 'dose_response_summary.csv'}")
    print(f"总摘要：{result_root / 'summary.json'}")
    if cfg.create_zip:
        archive = package_results(result_root)
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
