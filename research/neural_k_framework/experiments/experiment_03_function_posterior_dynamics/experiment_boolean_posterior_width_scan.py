"""
布尔函数后验动力学的 Oxford-tanh 小网络宽度扫描。

目标不是重新做一遍主实验，而是回答一个直接影响后续实验成本的问题：
从此前用于 Oxford 静态先验实验 smoke test 的 3 -> 16 x 2 -> 1 tanh
小网络出发，在保持深度、初始化、激活、优化器、训练约束和事件对齐协议
不变时，宽度增加会怎样改变函数后验与迁移方向？

当前扫描 3-bit -> 1-bit、2 个 tanh 隐藏层的 width=16/32/64/128。
初始化沿用 Oxford Boolean-prior 代码同族的 sigma_w=1.0、sigma_b=0.2
高斯尺度，不使用归一化层。16 是此前 smoke test 的超小网络宽度，其余宽度
用于判断差异是否随容量平滑变化。整个 hard function space 只有 256 个函数，
因此比较的是完整函数分布，而不是少量 probe 指标。

使用方式（AutoDL/Jupyter）：
1. 把本文件整段复制到一个 notebook 单元格，或直接运行本文件。
2. 按需修改 Config.WIDTHS、Config.PROFILE 和 Config.REFERENCE_RESULT_DIR。
3. 主实验结果存在时，脚本还会生成相对 1024 x 3 GELU+LayerNorm 网络的
   行为距离与抽样噪声校准；这是跨架构行为参照，不是纯宽度消融。主实验
   结果不存在时，以最大扫描宽度作为内部参照。

判读重点：
- initialization prior 是否仍相近；
- prior-consistent cohort 从 age=0 到 age=1 的迁移方向是否相同；
- 继续训练后的主导函数、分布和逐 seed 迁移率是否相近；
- 差异是否超过有限 seed 数本身造成的 bootstrap 波动。

本脚本不测量 Kolmogorov complexity，也不把网络宽度差异预先解释成复杂度
差异。它只提供后续大规模实验能否安全使用小网络的经验校准。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 配置
# =============================================================================


def script_directory() -> Path:
    source = globals().get("__file__")
    if source:
        return Path(source).resolve().parent
    return Path.cwd()


class Config:
    # full：正式尺寸扫描；quick：先看方向；smoke：只检查代码链路。
    PROFILE = "full"

    RESULT_DIR = script_directory() / "results_boolean_posterior_tanh_width_scan"
    REFERENCE_RESULT_DIR = Path("/root/results_boolean_function_posterior_dynamics")
    RESUME_EXISTING = True
    CREATE_ZIP = True

    INPUT_BITS = 3
    WIDTHS = (16, 32, 64, 128)
    # 精确复用 experiment_static_prior_vs_sgd_posterior.py 的 16 维 smoke
    # 网络族：3 -> width x 2 -> 1，tanh，无归一化。
    HIDDEN_LAYERS = 2
    SIGMA_W = 1.0
    SIGMA_B = 0.2
    LEARNING_RATE = 3e-3
    WEIGHT_DECAY = 0.0
    FIT_CHECK_INTERVAL = 1
    MAX_PREFIT_STEPS = 1_000

    PRIOR_SEED_BASE = 1_000_000
    ORDINARY_SEED_BASE = 2_000_000
    GLOBAL_SEED = 20260817

    # 与 1024 x 3 主实验使用完全相同的三组约束。
    FIXED_STATES = (
        ("single_x0_y0", (0,), (0,), "固定单样本约束"),
        ("auto_k2_x3-4_y01", (3, 4), (0, 1), "主实验选中的二样本约束"),
        (
            "auto_k4_x2-3-4-5_y0011",
            (2, 3, 4, 5),
            (0, 0, 1, 1),
            "主实验选中的四样本约束",
        ),
    )

    # full 与主实验保持相同 seed 数，便于直接比较。
    FULL_PRIOR_MODELS = 65_536
    FULL_ORDINARY_MODELS = 128
    FULL_CONSISTENT_MODELS = 512
    FULL_POST_FIT_AGES = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000)

    QUICK_PRIOR_MODELS = 16_384
    QUICK_ORDINARY_MODELS = 64
    QUICK_CONSISTENT_MODELS = 128
    QUICK_POST_FIT_AGES = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000)

    # 小网络可以一次并行很多独立模型。下面是保守上限，实际 chunk 还会按
    # width 自动缩放，以避免某个尺寸突然占满显存。
    MAX_PRIOR_CHUNK = 8_192
    MAX_TRAIN_CHUNK = 1_024

    # 只在这些关键年龄做跨宽度主表；原始文件仍保存全部年龄。
    COMPARISON_AGES = (0, 1, 10, 100, 1_000, 5_000)
    BOOTSTRAP_DRAWS = 500
    LOG_INTERVAL = 500

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False


@dataclass(frozen=True)
class EffectiveConfig:
    profile: str
    result_dir: Path
    input_bits: int
    hidden_size: int
    hidden_layers: int
    sigma_w: float
    sigma_b: float
    learning_rate: float
    weight_decay: float
    fit_check_interval: int
    max_prefit_steps: int
    prior_seed_base: int
    ordinary_seed_base: int
    prior_models: int
    prior_chunk: int
    ordinary_models: int
    consistent_models: int
    train_chunk: int
    post_fit_ages: tuple[int, ...]
    device: str
    allow_tf32: bool
    smoke_test: bool


@dataclass(frozen=True)
class StateSpec:
    name: str
    input_indices: tuple[int, ...]
    targets: tuple[int, ...]
    selection_kind: str
    description: str


def architecture_chunks(width: int) -> tuple[int, int]:
    # 主要参数量按 width^2 增长。以 width=64 时 prior=2048、train=512
    # 为基准缩放，同时设置上下界，兼顾吞吐和显存。
    scale = max((64.0 / float(width)) ** 2, 1.0)
    prior = min(Config.MAX_PRIOR_CHUNK, max(512, int(2_048 * scale)))
    train = min(Config.MAX_TRAIN_CHUNK, max(128, int(512 * scale)))
    return prior, train


def resolve_config(width: int) -> EffectiveConfig:
    profile = str(Config.PROFILE).lower().strip()
    prior_chunk, train_chunk = architecture_chunks(width)
    if profile == "full":
        values = (
            Config.FULL_PRIOR_MODELS,
            Config.FULL_ORDINARY_MODELS,
            Config.FULL_CONSISTENT_MODELS,
            Config.FULL_POST_FIT_AGES,
        )
        max_prefit = Config.MAX_PREFIT_STEPS
        smoke = False
    elif profile == "quick":
        values = (
            Config.QUICK_PRIOR_MODELS,
            Config.QUICK_ORDINARY_MODELS,
            Config.QUICK_CONSISTENT_MODELS,
            Config.QUICK_POST_FIT_AGES,
        )
        max_prefit = Config.MAX_PREFIT_STEPS
        smoke = False
    elif profile == "smoke":
        values = (512, 8, 8, (0, 1, 2, 5))
        prior_chunk = 64
        train_chunk = 4
        max_prefit = 100
        smoke = True
    else:
        raise ValueError(f"未知 PROFILE={Config.PROFILE!r}，只能是 full/quick/smoke。")

    prior_models, ordinary_models, consistent_models, ages = values
    if Config.INPUT_BITS != 3:
        raise ValueError("本扫描固定为可完整穷举的 3-bit -> 1-bit 任务。")
    return EffectiveConfig(
        profile=profile,
        result_dir=Path(Config.RESULT_DIR) / f"width_{width:04d}_depth_{Config.HIDDEN_LAYERS}",
        input_bits=Config.INPUT_BITS,
        hidden_size=int(width),
        hidden_layers=Config.HIDDEN_LAYERS,
        sigma_w=Config.SIGMA_W,
        sigma_b=Config.SIGMA_B,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        fit_check_interval=Config.FIT_CHECK_INTERVAL,
        max_prefit_steps=max_prefit,
        prior_seed_base=Config.PRIOR_SEED_BASE,
        ordinary_seed_base=Config.ORDINARY_SEED_BASE,
        prior_models=prior_models,
        prior_chunk=min(prior_chunk, prior_models),
        ordinary_models=ordinary_models,
        consistent_models=consistent_models,
        train_chunk=min(train_chunk, max(ordinary_models, consistent_models)),
        post_fit_ages=tuple(int(value) for value in ages),
        device=Config.DEVICE,
        allow_tf32=Config.ALLOW_TF32,
        smoke_test=smoke,
    )


def fixed_states() -> list[StateSpec]:
    return [
        StateSpec(
            name=name,
            input_indices=tuple(int(value) for value in indices),
            targets=tuple(int(value) for value in targets),
            selection_kind="fixed_from_1024_reference",
            description=description,
        )
        for name, indices, targets, description in Config.FIXED_STATES
    ]


# =============================================================================
# 通用工具
# =============================================================================


def stable_json_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def chunked(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def truth_table_inputs(input_bits: int) -> torch.Tensor:
    rows = 1 << input_bits
    values = torch.arange(rows, dtype=torch.long)
    shifts = torch.arange(input_bits, dtype=torch.long)
    return ((values[:, None] >> shifts[None, :]) & 1).to(torch.float32)


def function_count(input_bits: int) -> int:
    return 1 << (1 << input_bits)


def function_ids_from_logits(logits: torch.Tensor) -> torch.Tensor:
    bits = (logits >= 0).to(torch.long)
    powers = (1 << torch.arange(logits.shape[-1], device=logits.device)).to(torch.long)
    return (bits * powers).sum(dim=-1)


def compatible_function_mask(state: StateSpec, total_functions: int = 256) -> np.ndarray:
    function_ids = np.arange(total_functions, dtype=np.uint16)
    mask = np.ones(total_functions, dtype=bool)
    for input_index, target in zip(state.input_indices, state.targets):
        mask &= ((function_ids >> int(input_index)) & 1) == int(target)
    return mask


def normalized_counts(ids: np.ndarray, total_functions: int = 256) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(ids.astype(np.int64), minlength=total_functions).astype(np.int64)
    probability = counts.astype(np.float64)
    probability /= max(float(probability.sum()), 1.0)
    return counts, probability


def entropy_bits(probability: np.ndarray) -> float:
    positive = probability[probability > 0]
    return float(-(positive * np.log2(positive)).sum())


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def jensen_shannon_bits(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float((left[mask] * np.log2(left[mask] / right[mask])).sum())

    return 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)


def model_parameter_count(width: int, layers: int, input_bits: int = 3) -> int:
    linear = input_bits * width + width
    linear += max(layers - 1, 0) * (width * width + width)
    output = width + 1
    return linear + output


# =============================================================================
# 批量独立 Oxford-tanh MLP
# =============================================================================


class EnsembleGaussianLinear(nn.Module):
    def __init__(
        self,
        seeds: tuple[int, ...],
        in_features: int,
        out_features: int,
        salt: int,
        sigma_w: float,
        sigma_b: float,
    ):
        super().__init__()
        count = len(seeds)
        weight = torch.empty(count, out_features, in_features)
        bias = torch.empty(count, out_features)
        weight_scale = sigma_w / math.sqrt(in_features)
        bias_scale = sigma_b * sigma_w / in_features
        for index, seed in enumerate(seeds):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) * 1_000_003 + salt)
            weight[index].normal_(0.0, weight_scale, generator=generator)
            bias[index].normal_(0.0, bias_scale, generator=generator)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ebi,eoi->ebo", inputs, self.weight) + self.bias[:, None, :]


class BatchedSeedTanhMLP(nn.Module):
    def __init__(self, seeds: tuple[int, ...], cfg: EffectiveConfig):
        super().__init__()
        layers: list[nn.Module] = []
        width_in = cfg.input_bits
        for layer_index in range(cfg.hidden_layers):
            layers.append(
                EnsembleGaussianLinear(
                    seeds,
                    width_in,
                    cfg.hidden_size,
                    salt=100 + layer_index * 17,
                    sigma_w=cfg.sigma_w,
                    sigma_b=cfg.sigma_b,
                )
            )
            width_in = cfg.hidden_size
        self.hidden = nn.ModuleList(layers)
        self.output = EnsembleGaussianLinear(
            seeds,
            width_in,
            1,
            salt=10_007,
            sigma_w=cfg.sigma_w,
            sigma_b=cfg.sigma_b,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
        return self.output(hidden)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    probe_inputs: torch.Tensor,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
) -> dict[str, np.ndarray]:
    model.eval()
    logits = model(probe_inputs).squeeze(-1)
    train_logits = model(train_inputs).squeeze(-1)
    signed_targets = train_targets * 2.0 - 1.0
    margins = train_logits * signed_targets
    loss = F.binary_cross_entropy_with_logits(
        train_logits, train_targets, reduction="none"
    ).mean(dim=1)
    return {
        "probe_logits": logits.cpu().numpy().astype(np.float32),
        "function_ids": function_ids_from_logits(logits).cpu().numpy().astype(np.uint16),
        "train_loss": loss.cpu().numpy().astype(np.float32),
        "train_min_margin": margins.min(dim=1).values.cpu().numpy().astype(np.float32),
        "train_exact": (margins > 0).all(dim=1).cpu().numpy(),
    }


# =============================================================================
# 初始化函数先验
# =============================================================================


def config_payload(cfg: EffectiveConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["result_dir"] = str(cfg.result_dir)
    payload["protocol_version"] = "boolean_posterior_tanh_width_scan_v2"
    return payload


def sample_initialization_prior(
    cfg: EffectiveConfig,
    signature: str,
) -> dict[str, np.ndarray]:
    prior_dir = cfg.result_dir / "initialization_prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = prior_dir / "samples.npz"
    metadata_path = prior_dir / "metadata.json"
    if Config.RESUME_EXISTING and aggregate_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            print("  复用初始化函数先验。")
            return load_npz(aggregate_path)

    print(f"  采样初始化函数先验：models={cfg.prior_models:,}，chunk={cfg.prior_chunk:,}")
    started = time.perf_counter()
    device = torch.device(cfg.device)
    probe_cpu = truth_table_inputs(cfg.input_bits)
    seeds = np.arange(
        cfg.prior_seed_base,
        cfg.prior_seed_base + cfg.prior_models,
        dtype=np.int64,
    )
    all_ids: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    processed = 0
    chunks = list(chunked(seeds, cfg.prior_chunk))
    for chunk_index, seed_chunk in enumerate(chunks, start=1):
        seed_tuple = tuple(int(value) for value in seed_chunk)
        model = BatchedSeedTanhMLP(seed_tuple, cfg).to(device).eval()
        probe = probe_cpu.to(device)[None, :, :].expand(len(seed_tuple), -1, -1)
        with torch.no_grad():
            logits = model(probe).squeeze(-1)
            ids = function_ids_from_logits(logits)
        all_logits.append(logits.cpu().numpy().astype(np.float32))
        all_ids.append(ids.cpu().numpy().astype(np.uint16))
        processed += len(seed_tuple)
        del model, probe, logits, ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % 4 == 0:
            rate = processed / max(time.perf_counter() - started, 1e-9)
            print(f"    [prior] {processed:,}/{len(seeds):,} | {rate:,.1f} models/s")

    function_ids = np.concatenate(all_ids)
    probe_logits = np.concatenate(all_logits)
    counts = np.bincount(function_ids.astype(np.int64), minlength=256).astype(np.int64)
    arrays = {
        "seeds": seeds,
        "function_ids": function_ids,
        "probe_logits": probe_logits,
        "function_counts": counts,
    }
    np.savez_compressed(aggregate_path, **arrays)
    save_json(
        metadata_path,
        {
            "config_signature": signature,
            "samples": cfg.prior_models,
            "observed_functions": int(np.count_nonzero(counts)),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return arrays


# =============================================================================
# 事件对齐训练
# =============================================================================


def train_seed_chunk(
    cfg: EffectiveConfig,
    state: StateSpec,
    cohort: str,
    seeds_array: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seeds = tuple(int(value) for value in seeds_array)
    count = len(seeds)
    device = torch.device(cfg.device)
    probe_cpu = truth_table_inputs(cfg.input_bits)
    target_values = torch.tensor(state.targets, dtype=torch.float32)

    model = BatchedSeedTanhMLP(seeds, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    probe = probe_cpu.to(device)[None, :, :].expand(count, -1, -1)
    train_x = probe_cpu[list(state.input_indices)].to(device)[None, :, :].expand(count, -1, -1)
    train_y = target_values.to(device)[None, :].expand(count, -1)

    ages = tuple(sorted(set(int(value) for value in cfg.post_fit_ages)))
    output_ages = np.asarray((-1, *ages), dtype=np.int64)
    age_positions = {age: index + 1 for index, age in enumerate(ages)}
    snapshots = len(output_ages)
    recorded = np.zeros((snapshots, count), dtype=bool)
    absolute_steps = np.full((snapshots, count), -1, dtype=np.int64)
    logits_out = np.full((snapshots, count, 8), np.nan, dtype=np.float32)
    ids_out = np.zeros((snapshots, count), dtype=np.uint16)
    loss_out = np.full((snapshots, count), np.nan, dtype=np.float32)
    margin_out = np.full((snapshots, count), np.nan, dtype=np.float32)
    exact_out = np.zeros((snapshots, count), dtype=bool)
    first_fit = np.full(count, -1, dtype=np.int64)
    started = time.perf_counter()

    def assign(step: int, snapshot: dict[str, np.ndarray], masks: dict[int, np.ndarray]) -> None:
        for age, mask in masks.items():
            position = 0 if age == -1 else age_positions[age]
            logits_out[position, mask] = snapshot["probe_logits"][mask]
            ids_out[position, mask] = snapshot["function_ids"][mask]
            loss_out[position, mask] = snapshot["train_loss"][mask]
            margin_out[position, mask] = snapshot["train_min_margin"][mask]
            exact_out[position, mask] = snapshot["train_exact"][mask]
            absolute_steps[position, mask] = step
            recorded[position, mask] = True

    initial = evaluate_model(model, probe, train_x, train_y)
    assign(0, initial, {-1: np.ones(count, dtype=bool)})
    initially_fitted = initial["train_exact"]
    first_fit[initially_fitted] = 0
    if 0 in age_positions and np.any(initially_fitted):
        assign(0, initial, {0: initially_fitted})
    if cohort == "prior_consistent" and not np.all(initially_fitted):
        raise RuntimeError("prior-consistent seed 无法按相同 seed 重建，初始化链路不一致。")

    final_step = cfg.max_prefit_steps + max(ages)
    latest_loss = float(initial["train_loss"].mean())
    for step in range(1, final_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_x).squeeze(-1)
        loss_by_model = F.binary_cross_entropy_with_logits(
            train_logits, train_y, reduction="none"
        ).mean(dim=1)
        # 求和后，每个参数切片的梯度与逐模型独立训练严格一致。
        loss_by_model.sum().backward()
        optimizer.step()
        latest_loss = float(loss_by_model.mean().detach().item())

        checked: dict[str, np.ndarray] | None = None
        if np.any(first_fit < 0) and (step == 1 or step % cfg.fit_check_interval == 0):
            checked = evaluate_model(model, probe, train_x, train_y)
            newly_fitted = (first_fit < 0) & checked["train_exact"]
            first_fit[newly_fitted] = step

        assignments: dict[int, np.ndarray] = {}
        fitted = first_fit >= 0
        if np.any(fitted):
            current_ages = step - first_fit
            for age in ages:
                position = age_positions[age]
                due = fitted & (current_ages == age) & (~recorded[position])
                if np.any(due):
                    assignments[age] = due
        if assignments:
            if checked is None:
                checked = evaluate_model(model, probe, train_x, train_y)
            assign(step, checked, assignments)

        if bool(recorded[1:].all()):
            break
        if step >= cfg.max_prefit_steps and np.any(first_fit < 0):
            if not np.any((first_fit >= 0) & (~recorded[-1])):
                break
        interval = 50 if cfg.smoke_test else Config.LOG_INTERVAL
        if step % interval == 0:
            print(
                f"      step={step:,}/{final_step:,} | fitted={np.count_nonzero(first_fit>=0)}/{count} | "
                f"loss={latest_loss:.3e} | {time.perf_counter()-started:.1f}s"
            )

    arrays = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "post_fit_ages": output_ages,
        "recorded": recorded,
        "absolute_steps": absolute_steps,
        "probe_logits": logits_out,
        "function_ids": ids_out,
        "train_loss": loss_out,
        "train_min_margin": margin_out,
        "train_exact": exact_out,
        "first_fit_steps": first_fit,
    }
    metadata = {
        "state": asdict(state),
        "cohort": cohort,
        "seeds": list(seeds),
        "completed_models": int(recorded[1:].all(axis=0).sum()),
        "censored_models": int(np.count_nonzero(first_fit < 0)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del model, optimizer, probe, train_x, train_y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, arrays


def cohort_seeds(
    cfg: EffectiveConfig,
    prior: dict[str, np.ndarray],
    state: StateSpec,
    cohort: str,
) -> np.ndarray:
    if cohort == "ordinary":
        return np.arange(
            cfg.ordinary_seed_base,
            cfg.ordinary_seed_base + cfg.ordinary_models,
            dtype=np.int64,
        )
    compatible_ids = compatible_function_mask(state)[prior["function_ids"].astype(np.int64)]
    seeds = prior["seeds"][compatible_ids]
    if len(seeds) < cfg.consistent_models:
        raise RuntimeError(
            f"width={cfg.hidden_size} 的 {state.name} 只有 {len(seeds)} 个 "
            f"prior-consistent seed，少于目标 {cfg.consistent_models}。"
        )
    return seeds[: cfg.consistent_models].astype(np.int64)


def run_state_cohort(
    cfg: EffectiveConfig,
    signature: str,
    prior: dict[str, np.ndarray],
    state: StateSpec,
    cohort: str,
) -> dict[str, np.ndarray]:
    state_dir = cfg.result_dir / "training" / cohort / state.name
    chunks_dir = state_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = state_dir / "samples.npz"
    metadata_path = state_dir / "metadata.json"
    if Config.RESUME_EXISTING and aggregate_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            print(f"    [skip] {cohort}/{state.name}")
            return load_npz(aggregate_path)

    seeds = cohort_seeds(cfg, prior, state, cohort)
    chunks = list(chunked(seeds, cfg.train_chunk))
    results: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for chunk_index, seed_chunk in enumerate(chunks, start=1):
        tag = f"seed_{int(seed_chunk[0])}_{int(seed_chunk[-1])}"
        npz_path = chunks_dir / f"{tag}.npz"
        json_path = chunks_dir / f"{tag}.json"
        print(
            f"    [{cohort}] {state.name} chunk={chunk_index}/{len(chunks)} | "
            f"models={len(seed_chunk)}"
        )
        if Config.RESUME_EXISTING and npz_path.exists() and json_path.exists():
            chunk_meta = json.loads(json_path.read_text(encoding="utf-8"))
            if chunk_meta.get("config_signature") == signature:
                results.append((chunk_meta, load_npz(npz_path)))
                print("      [skip chunk]")
                continue
        chunk_meta, chunk_arrays = train_seed_chunk(cfg, state, cohort, seed_chunk)
        chunk_meta["config_signature"] = signature
        np.savez_compressed(npz_path, **chunk_arrays)
        save_json(json_path, chunk_meta)
        results.append((chunk_meta, chunk_arrays))

    reference_ages = results[0][1]["post_fit_ages"]
    aggregate = {
        "seeds": np.concatenate([arrays["seeds"] for _, arrays in results]),
        "post_fit_ages": reference_ages,
        "recorded": np.concatenate([arrays["recorded"] for _, arrays in results], axis=1),
        "absolute_steps": np.concatenate(
            [arrays["absolute_steps"] for _, arrays in results], axis=1
        ),
        "probe_logits": np.concatenate(
            [arrays["probe_logits"] for _, arrays in results], axis=1
        ),
        "function_ids": np.concatenate(
            [arrays["function_ids"] for _, arrays in results], axis=1
        ),
        "train_loss": np.concatenate([arrays["train_loss"] for _, arrays in results], axis=1),
        "train_min_margin": np.concatenate(
            [arrays["train_min_margin"] for _, arrays in results], axis=1
        ),
        "train_exact": np.concatenate(
            [arrays["train_exact"] for _, arrays in results], axis=1
        ),
        "first_fit_steps": np.concatenate(
            [arrays["first_fit_steps"] for _, arrays in results]
        ),
    }
    np.savez_compressed(aggregate_path, **aggregate)
    save_json(
        metadata_path,
        {
            "config_signature": signature,
            "state": asdict(state),
            "cohort": cohort,
            "model_count": len(seeds),
            "completed_models": int(aggregate["recorded"][1:].all(axis=0).sum()),
            "censored_models": int(np.count_nonzero(aggregate["first_fit_steps"] < 0)),
        },
    )
    return aggregate


# =============================================================================
# 单个宽度分析
# =============================================================================


def analyze_width(
    cfg: EffectiveConfig,
    prior: dict[str, np.ndarray],
    states: list[StateSpec],
    training: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict[str, Any]:
    prior_counts = prior["function_counts"].astype(np.int64)
    prior_probability = prior_counts.astype(np.float64) / prior_counts.sum()
    metrics_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for state in states:
        compatibility = compatible_function_mask(state)
        hard_counts = prior_counts * compatibility
        hard_probability = hard_counts.astype(np.float64)
        hard_probability /= hard_probability.sum()

        for cohort in ("ordinary", "prior_consistent"):
            arrays = training[(cohort, state.name)]
            ages = arrays["post_fit_ages"].astype(np.int64)
            first_fit_position = int(np.flatnonzero(ages == 0)[0])
            first_fit_ids = arrays["function_ids"][first_fit_position]
            first_fit_logits = arrays["probe_logits"][first_fit_position]
            for position, age in enumerate(ages):
                recorded = arrays["recorded"][position]
                if not np.any(recorded):
                    continue
                ids = arrays["function_ids"][position, recorded]
                logits = arrays["probe_logits"][position, recorded]
                counts, probability = normalized_counts(ids)
                valid_first = recorded & arrays["recorded"][first_fit_position]
                hard_change = float("nan")
                logit_rmse = float("nan")
                if int(age) >= 0 and np.any(valid_first):
                    hard_change = float(
                        np.mean(arrays["function_ids"][position, valid_first] != first_fit_ids[valid_first])
                    )
                    difference = (
                        arrays["probe_logits"][position, valid_first]
                        - first_fit_logits[valid_first]
                    )
                    logit_rmse = float(np.sqrt(np.mean(np.square(difference))))

                top_ids = np.argsort(probability)[::-1][:5]
                metrics_rows.append(
                    {
                        "width": cfg.hidden_size,
                        "state": state.name,
                        "cohort": cohort,
                        "post_fit_age": int(age),
                        "model_count": int(recorded.sum()),
                        "train_exact_rate": float(arrays["train_exact"][position, recorded].mean()),
                        "mean_train_loss": float(arrays["train_loss"][position, recorded].mean()),
                        "distribution_entropy_bits": entropy_bits(probability),
                        "tv_from_width_specific_hard_prior": total_variation(probability, hard_probability),
                        "js_from_width_specific_hard_prior": jensen_shannon_bits(
                            probability, hard_probability
                        ),
                        "hard_change_rate_from_first_fit": hard_change,
                        "mean_logit_rmse_from_first_fit": logit_rmse,
                        "top1_function_id": int(top_ids[0]),
                        "top1_probability": float(probability[top_ids[0]]),
                        "top5_function_ids": " ".join(str(int(value)) for value in top_ids),
                    }
                )
                for function_id in range(256):
                    distribution_rows.append(
                        {
                            "width": cfg.hidden_size,
                            "state": state.name,
                            "cohort": cohort,
                            "post_fit_age": int(age),
                            "function_id": function_id,
                            "q_count": int(counts[function_id]),
                            "q_probability": float(probability[function_id]),
                            "hard_prior_count": int(hard_counts[function_id]),
                            "hard_prior_probability": float(hard_probability[function_id]),
                            "initialization_prior_count": int(prior_counts[function_id]),
                            "initialization_prior_probability": float(prior_probability[function_id]),
                            "compatible": bool(compatibility[function_id]),
                        }
                    )

    write_csv(cfg.result_dir / "posterior_metrics_by_age.csv", metrics_rows)
    write_csv(cfg.result_dir / "function_distributions.csv", distribution_rows)
    prior_rows = [
        {
            "width": cfg.hidden_size,
            "function_id": function_id,
            "truth_table": format(function_id, "08b")[::-1],
            "prior_count": int(prior_counts[function_id]),
            "prior_probability": float(prior_probability[function_id]),
        }
        for function_id in range(256)
    ]
    write_csv(cfg.result_dir / "initialization_function_prior.csv", prior_rows)

    final_age = max(cfg.post_fit_ages)
    headline = [
        row for row in metrics_rows if int(row["post_fit_age"]) == final_age
    ]
    summary = {
        "architecture": {
            "input_bits": cfg.input_bits,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "activation": "tanh",
            "normalization": "none",
            "sigma_w": cfg.sigma_w,
            "sigma_b": cfg.sigma_b,
            "parameters_per_model": model_parameter_count(
                cfg.hidden_size, cfg.hidden_layers, cfg.input_bits
            ),
        },
        "initialization_prior": {
            "models": cfg.prior_models,
            "observed_functions": int(np.count_nonzero(prior_counts)),
            "entropy_bits": entropy_bits(prior_probability),
            "top1_function_id": int(np.argmax(prior_probability)),
            "top1_probability": float(prior_probability.max()),
        },
        "headline_final_metrics": headline,
    }
    save_json(cfg.result_dir / "summary.json", summary)
    return summary


def run_width(width: int, states: list[StateSpec]) -> dict[str, Any]:
    cfg = resolve_config(width)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    payload = config_payload(cfg)
    signature = stable_json_hash(payload)
    save_json(cfg.result_dir / "config.json", {"config_signature": signature, **payload})
    save_json(cfg.result_dir / "fixed_training_states.json", [asdict(state) for state in states])

    print(f"\n{'='*78}")
    print(
        f"width={width} | params={model_parameter_count(width, cfg.hidden_layers):,} | "
        f"prior_chunk={cfg.prior_chunk:,} | train_chunk={cfg.train_chunk:,}"
    )
    prior = sample_initialization_prior(cfg, signature)
    run_signature = stable_json_hash(
        {
            "config_signature": signature,
            "states": [asdict(state) for state in states],
            "gradient_reduction": "sum_over_independent_models",
        }
    )
    training: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for state in states:
        for cohort in ("ordinary", "prior_consistent"):
            training[(cohort, state.name)] = run_state_cohort(
                cfg, run_signature, prior, state, cohort
            )
    return analyze_width(cfg, prior, states, training)


# =============================================================================
# 跨宽度比较
# =============================================================================


def distribution_table(path: Path) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    rows = read_csv(path)
    grouped: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    buffers: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        key = (row["state"], row["cohort"], int(row["post_fit_age"]))
        if key not in buffers:
            buffers[key] = (np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.float64))
        counts, probability = buffers[key]
        function_id = int(row["function_id"])
        counts[function_id] = int(row["q_count"])
        probability[function_id] = float(row["q_probability"])
    grouped.update(buffers)
    return grouped


def prior_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(256, dtype=np.int64)
    probability = np.zeros(256, dtype=np.float64)
    for row in read_csv(path):
        function_id = int(row["function_id"])
        counts[function_id] = int(row["prior_count"])
        probability[function_id] = float(row["prior_probability"])
    return counts, probability


def bootstrap_distance_null(
    first: np.ndarray,
    second: np.ndarray,
    first_n: int,
    second_n: int,
    random: np.random.Generator,
) -> dict[str, float]:
    pooled = (first * first_n + second * second_n) / max(first_n + second_n, 1)
    pooled /= pooled.sum()
    tv_values = np.empty(Config.BOOTSTRAP_DRAWS, dtype=np.float64)
    js_values = np.empty(Config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(Config.BOOTSTRAP_DRAWS):
        left = random.multinomial(first_n, pooled).astype(np.float64) / first_n
        right = random.multinomial(second_n, pooled).astype(np.float64) / second_n
        tv_values[index] = total_variation(left, right)
        js_values[index] = jensen_shannon_bits(left, right)
    return {
        "tv_null_median": float(np.median(tv_values)),
        "tv_null_p95": float(np.quantile(tv_values, 0.95)),
        "js_null_median": float(np.median(js_values)),
        "js_null_p95": float(np.quantile(js_values, 0.95)),
    }


def compare_pair(
    width: int,
    reference_label: str,
    key: tuple[str, str, int],
    current: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
    random: np.random.Generator,
) -> dict[str, Any]:
    current_counts, current_probability = current
    reference_counts, reference_probability = reference
    current_n = int(current_counts.sum())
    reference_n = int(reference_counts.sum())
    tv = total_variation(current_probability, reference_probability)
    js = jensen_shannon_bits(current_probability, reference_probability)
    null = bootstrap_distance_null(
        current_probability,
        reference_probability,
        current_n,
        reference_n,
        random,
    )
    current_top = np.argsort(current_probability)[::-1][:5]
    reference_top = np.argsort(reference_probability)[::-1][:5]
    return {
        "width": width,
        "reference": reference_label,
        "state": key[0],
        "cohort": key[1],
        "post_fit_age": key[2],
        "current_models": current_n,
        "reference_models": reference_n,
        "tv": tv,
        "tv_null_median": null["tv_null_median"],
        "tv_null_p95": null["tv_null_p95"],
        "tv_exceeds_sampling_p95": tv > null["tv_null_p95"],
        "js_bits": js,
        "js_null_median": null["js_null_median"],
        "js_null_p95": null["js_null_p95"],
        "js_exceeds_sampling_p95": js > null["js_null_p95"],
        "top1_match": int(current_top[0]) == int(reference_top[0]),
        "current_top1_function_id": int(current_top[0]),
        "current_top1_probability": float(current_probability[current_top[0]]),
        "reference_top1_function_id": int(reference_top[0]),
        "reference_top1_probability": float(reference_probability[reference_top[0]]),
        "top5_overlap": len(set(map(int, current_top)) & set(map(int, reference_top))),
    }


def cross_width_analysis(widths: list[int]) -> dict[str, Any]:
    root = Path(Config.RESULT_DIR)
    random = np.random.default_rng(Config.GLOBAL_SEED + 91)
    tables: dict[int, dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]] = {}
    priors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for width in widths:
        directory = root / f"width_{width:04d}_depth_{Config.HIDDEN_LAYERS}"
        tables[width] = distribution_table(directory / "function_distributions.csv")
        priors[width] = prior_table(directory / "initialization_function_prior.csv")

    reference_directory = Path(Config.REFERENCE_RESULT_DIR)
    external_exists = (
        (reference_directory / "function_distributions.csv").exists()
        and (reference_directory / "initialization_function_prior.csv").exists()
    )
    if external_exists:
        reference_label = "gelu_layernorm_width_1024_depth_3_external"
        reference_table = distribution_table(reference_directory / "function_distributions.csv")
        reference_prior = prior_table(reference_directory / "initialization_function_prior.csv")
        print(
            "读取 1024 x 3 GELU+LayerNorm 外部行为参照（不是纯宽度消融）："
            f"{reference_directory}"
        )
    else:
        anchor = max(widths)
        reference_label = f"width_{anchor}_depth_{Config.HIDDEN_LAYERS}_scan_anchor"
        reference_table = tables[anchor]
        reference_prior = priors[anchor]
        print("未找到 1024 x 3 参考结果，暂以最大扫描宽度作为内部参照。")

    requested_ages = set(int(value) for value in Config.COMPARISON_AGES)
    rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    for width in widths:
        shared_keys = sorted(set(tables[width]) & set(reference_table))
        for key in shared_keys:
            if key[2] not in requested_ages:
                continue
            rows.append(
                compare_pair(
                    width,
                    reference_label,
                    key,
                    tables[width][key],
                    reference_table[key],
                    random,
                )
            )
        prior_rows.append(
            {
                **compare_pair(
                    width,
                    reference_label,
                    ("initialization_prior", "prior", -1),
                    priors[width],
                    reference_prior,
                    random,
                )
            }
        )

    write_csv(root / "comparison_to_reference.csv", rows)
    write_csv(root / "initialization_prior_comparison.csv", prior_rows)

    summaries: list[dict[str, Any]] = []
    for width in widths:
        width_rows = [row for row in rows if int(row["width"]) == width]
        final_age = max((int(row["post_fit_age"]) for row in width_rows), default=-1)
        final_rows = [row for row in width_rows if int(row["post_fit_age"]) == final_age]
        key_rows = [
            row
            for row in width_rows
            if int(row["post_fit_age"]) in {1, final_age}
            and row["cohort"] == "prior_consistent"
        ]
        summaries.append(
            {
                "width": width,
                "parameters_per_model": model_parameter_count(width, Config.HIDDEN_LAYERS),
                "reference": reference_label,
                "final_compared_age": final_age,
                "mean_final_tv": float(np.mean([row["tv"] for row in final_rows]))
                if final_rows
                else float("nan"),
                "mean_final_js_bits": float(np.mean([row["js_bits"] for row in final_rows]))
                if final_rows
                else float("nan"),
                "key_comparisons": len(key_rows),
                "key_top1_matches": int(sum(bool(row["top1_match"]) for row in key_rows)),
                "key_top5_overlap_mean": float(np.mean([row["top5_overlap"] for row in key_rows]))
                if key_rows
                else float("nan"),
                "key_js_within_sampling_p95": int(
                    sum(not bool(row["js_exceeds_sampling_p95"]) for row in key_rows)
                ),
            }
        )
    write_csv(root / "width_summary.csv", summaries)
    summary = {
        "reference": reference_label,
        "external_1024_reference_found": external_exists,
        "widths": widths,
        "width_summaries": summaries,
        "interpretation": {
            "primary": "优先看 prior-consistent cohort 在 age=1 与最终年龄的 top1/top5 和 JS。",
            "sampling": "距离超过 bootstrap p95 才有证据认为差异超出有限 seed 抽样波动。",
            "decision": "选择仍保留主导迁移模式且差异可接受的最小宽度，再用于后续大扫描。",
            "external_reference_caveat": (
                "1024 参照使用 GELU+LayerNorm、3 个隐藏层和不同初始化；"
                "它只回答行为是否接近，不能把差异单独归因于宽度。"
            ),
        },
    }
    save_json(root / "summary.json", summary)
    return summary


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    widths = sorted({int(width) for width in Config.WIDTHS})
    if not widths or min(widths) < 2:
        raise ValueError("Config.WIDTHS 至少要包含一个不小于 2 的宽度。")
    root = Path(Config.RESULT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(Config.GLOBAL_SEED)
    np.random.seed(Config.GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32

    print("=== Boolean posterior Oxford-tanh network-width scan ===")
    print(f"设备：{Config.DEVICE}")
    if Config.DEVICE == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(
        f"配置：profile={Config.PROFILE} | widths={widths} | depth={Config.HIDDEN_LAYERS} | "
        f"activation=tanh | sigma_w={Config.SIGMA_W} | sigma_b={Config.SIGMA_B} | "
        f"reference={Config.REFERENCE_RESULT_DIR}"
    )
    print(f"结果目录：{root}")

    states = fixed_states()
    started = time.perf_counter()
    width_summaries = []
    for width in widths:
        width_summaries.append(run_width(width, states))

    comparison = cross_width_analysis(widths)
    elapsed = time.perf_counter() - started
    save_json(
        root / "manifest.json",
        {
            "elapsed_seconds": elapsed,
            "profile": Config.PROFILE,
            "widths": widths,
            "width_results": width_summaries,
            "comparison": comparison,
            "main_outputs": {
                "summary": "summary.json",
                "width_summary": "width_summary.csv",
                "comparison": "comparison_to_reference.csv",
                "prior_comparison": "initialization_prior_comparison.csv",
            },
        },
    )

    archive: str | None = None
    if Config.CREATE_ZIP:
        archive = shutil.make_archive(
            str(root),
            "zip",
            root_dir=root.parent,
            base_dir=root.name,
        )
    print("\n=== 宽度扫描完成 ===")
    print(f"总耗时：{elapsed/60:.1f} min")
    print(f"主表：{root / 'width_summary.csv'}")
    if archive:
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
