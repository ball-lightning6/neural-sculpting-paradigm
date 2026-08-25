"""
复现 Mingard et al. (Nature Communications, 2025) 的 n=7 Boolean
实验，并把观测窗口从“首次达到零训练分类错误”延长到拟合之后。

论文的静态函数先验图景在零分类错误条件下给出：

    P_B(f | S) ∝ P_0(f) * 1[f 与训练集 S 一致]

论文把 SGD posterior 定义在首次达到零训练错误的时刻。本实验保留这一时刻，
随后继续使用完全相同的优化器和训练数据，直接检验函数分布是否继续系统漂移。

主要控制：
1. n=7，完整输入空间 128 点；
2. 10 个宽度 40 的 tanh 隐藏层；
3. 权重初始化 std = sigma_w / sqrt(fan_in)，sigma_w in {1, 8}；
4. 可训练 bias 从 0 开始；
5. BCE/cross-entropy、Adam、batch=16；主协议采用论文的 advSGD 难例采样；
6. 保存初始化、首次拟合以及拟合后多个年龄的完整 128-bit 函数；
7. 可选采样初始化先验，并按随机训练集被函数完全满足的精确概率进行加权，
   构造论文静态图景下的 averaged posterior。

脚本完全自包含，兼容：
    python experiment_mingard_2025_postfit_drift.py
    %run experiment_mingard_2025_postfit_drift.py
    整个文件粘贴到 AutoDL Jupyter cell 直接运行

不使用 argparse 或环境变量；所有常用设置都在 Config 中。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import zipfile
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
    """同时支持 .py、%run 和直接粘贴到 Jupyter cell。"""
    source = globals().get("__file__")
    if source and not str(source).startswith("<"):
        return Path(source).resolve().parent
    return Path.cwd()


class Config:
    BASE_DIR = script_directory()
    RESULT_ROOT = BASE_DIR / "results_mingard_2025_postfit_drift"

    # smoke: 本地流程检查；pilot: 先看方向；paper_core: 论文核心规模；
    # full: 复现论文 3 个数据量的完整网格。
    PROFILE = "pilot"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    # 原论文 Boolean FCN。
    INPUT_BITS = 7
    HIDDEN_SIZE = 40
    HIDDEN_LAYERS = 10
    BIAS_INIT_STD = 0.0
    LEARNING_RATE = 1e-3
    ADAM_EPS = 1e-7  # 对齐 Keras Adam 默认 epsilon。
    BATCH_SIZE = 16
    ADV_HISTORY = 3
    ADV_INITIAL_LOSS = 0.5

    # 论文主实验使用 advSGD。可加 "adam_uniform" 做优化器稳健性对照。
    OPTIMIZERS = ("advsgd",)

    # resampled：每个 seed 独立抽训练集，对齐主文 Figure 1/2；
    # fixed：同一条件的所有 seed 共用训练集，对齐 Supplementary Figure S9，
    #        适合更干净地检查固定 S 下的函数 posterior。
    TRAIN_SET_MODES = ("resampled",)
    FIXED_SET_REPLICATES = 1

    # 如果手动覆盖 profile，可修改以下设置后把 PROFILE 改为 "custom"。
    CUSTOM_TARGETS = ("low_31.5", "mid_66.5", "high_101.5")
    CUSTOM_SIGMA_WS = (1.0, 8.0)
    CUSTOM_TRAIN_SIZES = (64,)
    CUSTOM_MODEL_COUNT = 2_048
    CUSTOM_TRAIN_CHUNK_SIZE = 1_024
    CUSTOM_POST_FIT_AGES = (0, 100, 1_000)
    CUSTOM_MAX_TOTAL_STEPS = 12_000
    CUSTOM_PRIOR_SAMPLES = 262_144
    CUSTOM_PRIOR_CHUNK_SIZE = 4_096
    CUSTOM_PERMUTATION_REPEATS = 200
    CUSTOM_BOOTSTRAP_REPEATS = 1_000

    # 先验不是证明 post-fit 漂移所必需，但可直接比较论文的静态预测。
    RUN_INITIALIZATION_PRIOR = True
    PRIOR_SEED = 20260817
    TRAIN_SEED = 20261817
    ANALYSIS_SEED = 20262817

    LOG_EVERY_STEPS = 250
    RESUME_EXISTING = True
    CREATE_PLOTS = True
    CREATE_ZIP = True


PROTOCOL_VERSION = "mingard_2025_boolean_postfit_drift_v1"
INPUT_COUNT = 128
PACKED_BYTES = 16


def paper_sparse_target(draw_count: int) -> str:
    """复现公开代码中的 np.random.seed(10) + choice（默认允许重复抽中）。"""
    rng = np.random.RandomState(10)
    bits = np.zeros(INPUT_COUNT, dtype=np.uint8)
    bits[rng.choice(INPUT_COUNT, size=draw_count, replace=True)] = 1
    return "".join(map(str, bits.tolist()))


# Figure 1d-f / Figure S9 的三个原始 target；公开代码分别以 2、10、21 次
# 稀疏位置抽样生成，对应论文 LZ complexity 31.5、66.5、101.5。
TARGET_BITSTRINGS = {
    "low_31.5": paper_sparse_target(2),
    "mid_66.5": paper_sparse_target(10),
    "high_101.5": paper_sparse_target(21),
}

TARGET_EXPECTED_LZ_X2 = {
    "low_31.5": 63,
    "mid_66.5": 133,
    "high_101.5": 203,
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    bits: str
    expected_lz_x2: int


@dataclass(frozen=True)
class ModelSpec:
    input_bits: int
    hidden_size: int
    hidden_layers: int
    bias_init_std: float
    learning_rate: float
    adam_eps: float
    batch_size: int
    adv_history: int
    adv_initial_loss: float


@dataclass(frozen=True)
class ConditionSpec:
    target_name: str
    sigma_w: float
    train_size: int
    optimizer: str
    train_set_mode: str
    set_replicate: int

    @property
    def name(self) -> str:
        sigma = str(self.sigma_w).replace(".", "p")
        return (
            f"target-{self.target_name}_sw-{sigma}_m-{self.train_size}_"
            f"opt-{self.optimizer}_set-{self.train_set_mode}-r{self.set_replicate}"
        )


@dataclass(frozen=True)
class EffectiveConfig:
    protocol_version: str
    profile: str
    result_root: str
    device: str
    allow_tf32: bool
    model: ModelSpec
    targets: tuple[TargetSpec, ...]
    sigma_ws: tuple[float, ...]
    train_sizes: tuple[int, ...]
    optimizers: tuple[str, ...]
    train_set_modes: tuple[str, ...]
    fixed_set_replicates: int
    model_count: int
    train_chunk_size: int
    post_fit_ages: tuple[int, ...]
    max_total_steps: int
    run_initialization_prior: bool
    prior_samples: int
    prior_chunk_size: int
    prior_seed: int
    train_seed: int
    analysis_seed: int
    permutation_repeats: int
    bootstrap_repeats: int
    log_every_steps: int
    smoke_test: bool


def profile_values(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "targets": ("low_31.5",),
            "sigma_ws": (1.0,),
            "train_sizes": (32,),
            "model_count": 64,
            "train_chunk_size": 64,
            "post_fit_ages": (0, 5, 20),
            "max_total_steps": 300,
            "prior_samples": 2_048,
            "prior_chunk_size": 256,
            "permutation_repeats": 20,
            "bootstrap_repeats": 100,
            "smoke_test": True,
        }
    if profile == "pilot":
        return {
            "targets": ("low_31.5", "mid_66.5", "high_101.5"),
            "sigma_ws": (1.0, 8.0),
            "train_sizes": (64,),
            "model_count": 2_048,
            "train_chunk_size": 1_024,
            "post_fit_ages": (0, 100, 1_000),
            "max_total_steps": 12_000,
            "prior_samples": 262_144,
            "prior_chunk_size": 4_096,
            "permutation_repeats": 200,
            "bootstrap_repeats": 1_000,
            "smoke_test": False,
        }
    if profile == "paper_core":
        return {
            "targets": ("low_31.5", "mid_66.5", "high_101.5"),
            "sigma_ws": (1.0, 8.0),
            "train_sizes": (64,),
            "model_count": 8_192,
            "train_chunk_size": 4_096,
            "post_fit_ages": (0, 100, 1_000, 5_000),
            "max_total_steps": 25_000,
            "prior_samples": 1_048_576,
            "prior_chunk_size": 8_192,
            "permutation_repeats": 500,
            "bootstrap_repeats": 2_000,
            "smoke_test": False,
        }
    if profile == "full":
        return {
            "targets": ("low_31.5", "mid_66.5", "high_101.5"),
            "sigma_ws": (1.0, 8.0),
            "train_sizes": (32, 64, 85),
            "model_count": 8_192,
            "train_chunk_size": 4_096,
            "post_fit_ages": (0, 100, 1_000, 5_000),
            "max_total_steps": 25_000,
            "prior_samples": 1_048_576,
            "prior_chunk_size": 8_192,
            "permutation_repeats": 500,
            "bootstrap_repeats": 2_000,
            "smoke_test": False,
        }
    if profile == "custom":
        return {
            "targets": tuple(Config.CUSTOM_TARGETS),
            "sigma_ws": tuple(Config.CUSTOM_SIGMA_WS),
            "train_sizes": tuple(Config.CUSTOM_TRAIN_SIZES),
            "model_count": int(Config.CUSTOM_MODEL_COUNT),
            "train_chunk_size": int(Config.CUSTOM_TRAIN_CHUNK_SIZE),
            "post_fit_ages": tuple(Config.CUSTOM_POST_FIT_AGES),
            "max_total_steps": int(Config.CUSTOM_MAX_TOTAL_STEPS),
            "prior_samples": int(Config.CUSTOM_PRIOR_SAMPLES),
            "prior_chunk_size": int(Config.CUSTOM_PRIOR_CHUNK_SIZE),
            "permutation_repeats": int(Config.CUSTOM_PERMUTATION_REPEATS),
            "bootstrap_repeats": int(Config.CUSTOM_BOOTSTRAP_REPEATS),
            "smoke_test": False,
        }
    raise ValueError(f"未知 PROFILE：{profile}")


def get_effective_config() -> EffectiveConfig:
    values = profile_values(str(Config.PROFILE))
    targets = tuple(
        TargetSpec(name, TARGET_BITSTRINGS[name], TARGET_EXPECTED_LZ_X2[name])
        for name in values["targets"]
    )
    return EffectiveConfig(
        protocol_version=PROTOCOL_VERSION,
        profile=str(Config.PROFILE),
        result_root=str(Config.RESULT_ROOT / str(Config.PROFILE)),
        device=str(Config.DEVICE),
        allow_tf32=bool(Config.ALLOW_TF32),
        model=ModelSpec(
            input_bits=int(Config.INPUT_BITS),
            hidden_size=int(Config.HIDDEN_SIZE),
            hidden_layers=int(Config.HIDDEN_LAYERS),
            bias_init_std=float(Config.BIAS_INIT_STD),
            learning_rate=float(Config.LEARNING_RATE),
            adam_eps=float(Config.ADAM_EPS),
            batch_size=int(Config.BATCH_SIZE),
            adv_history=int(Config.ADV_HISTORY),
            adv_initial_loss=float(Config.ADV_INITIAL_LOSS),
        ),
        targets=targets,
        sigma_ws=tuple(map(float, values["sigma_ws"])),
        train_sizes=tuple(map(int, values["train_sizes"])),
        optimizers=tuple(map(str, Config.OPTIMIZERS)),
        train_set_modes=tuple(map(str, Config.TRAIN_SET_MODES)),
        fixed_set_replicates=int(Config.FIXED_SET_REPLICATES),
        model_count=int(values["model_count"]),
        train_chunk_size=int(values["train_chunk_size"]),
        post_fit_ages=tuple(map(int, values["post_fit_ages"])),
        max_total_steps=int(values["max_total_steps"]),
        run_initialization_prior=bool(Config.RUN_INITIALIZATION_PRIOR),
        prior_samples=int(values["prior_samples"]),
        prior_chunk_size=int(values["prior_chunk_size"]),
        prior_seed=int(Config.PRIOR_SEED),
        train_seed=int(Config.TRAIN_SEED),
        analysis_seed=int(Config.ANALYSIS_SEED),
        permutation_repeats=int(values["permutation_repeats"]),
        bootstrap_repeats=int(values["bootstrap_repeats"]),
        log_every_steps=int(Config.LOG_EVERY_STEPS),
        smoke_test=bool(values["smoke_test"]),
    )


def validate_config(cfg: EffectiveConfig) -> None:
    if cfg.model.input_bits != 7 or 2**cfg.model.input_bits != INPUT_COUNT:
        raise ValueError("本实验固定使用 n=7 Boolean 输入")
    if cfg.model.hidden_size <= 0 or cfg.model.hidden_layers <= 0:
        raise ValueError("网络宽度与层数必须为正数")
    if cfg.model.batch_size <= 0:
        raise ValueError("BATCH_SIZE 必须为正数")
    if cfg.model_count <= 0 or cfg.train_chunk_size <= 0:
        raise ValueError("模型数与 chunk size 必须为正数")
    if not cfg.post_fit_ages or cfg.post_fit_ages[0] != 0:
        raise ValueError("POST_FIT_AGES 必须从 0 开始")
    if tuple(sorted(set(cfg.post_fit_ages))) != cfg.post_fit_ages:
        raise ValueError("POST_FIT_AGES 必须严格递增且不重复")
    if cfg.max_total_steps < max(cfg.post_fit_ages):
        raise ValueError("MAX_TOTAL_STEPS 必须不小于最大 post-fit age")
    for size in cfg.train_sizes:
        if size <= 0 or size >= INPUT_COUNT:
            raise ValueError(f"非法训练集大小：{size}")
        if cfg.model.batch_size > size:
            raise ValueError("BATCH_SIZE 不得大于训练集大小")
    unknown_optimizers = set(cfg.optimizers) - {"advsgd", "adam_uniform", "adam_fullbatch"}
    if unknown_optimizers:
        raise ValueError(f"未知优化器：{sorted(unknown_optimizers)}")
    unknown_modes = set(cfg.train_set_modes) - {"resampled", "fixed"}
    if unknown_modes:
        raise ValueError(f"未知训练集模式：{sorted(unknown_modes)}")
    for target in cfg.targets:
        if len(target.bits) != INPUT_COUNT or set(target.bits) - {"0", "1"}:
            raise ValueError(f"非法 target bitstring：{target.name}")
        actual = lz_complexity_x2(np.fromiter(map(int, target.bits), dtype=np.uint8))
        if actual != target.expected_lz_x2:
            raise ValueError(
                f"target {target.name} 的 LZ 不匹配：{actual / 2} != "
                f"{target.expected_lz_x2 / 2}"
            )


# =============================================================================
# 通用工具
# =============================================================================


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def derive_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def chunk_ranges(total: int, chunk_size: int) -> Iterable[tuple[int, int, int]]:
    chunk_index = 0
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield chunk_index, start, end
        chunk_index += 1


def configure_torch(cfg: EffectiveConfig) -> torch.device:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE=cuda，但当前环境没有可用 CUDA")
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    return device


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def all_boolean_inputs(device: torch.device | None = None) -> torch.Tensor:
    values = torch.arange(INPUT_COUNT, dtype=torch.int64, device=device)
    shifts = torch.arange(6, -1, -1, dtype=torch.int64, device=device)
    return ((values[:, None] >> shifts[None, :]) & 1).to(torch.float32)


def target_array(target: TargetSpec) -> np.ndarray:
    return np.fromiter((int(bit) for bit in target.bits), dtype=np.uint8)


def endpoint_names(cfg: EffectiveConfig) -> tuple[str, ...]:
    return (
        "initialization",
        *(f"post_fit_age_{age:06d}" for age in cfg.post_fit_ages),
        "last_step",
    )


def build_conditions(cfg: EffectiveConfig) -> tuple[ConditionSpec, ...]:
    conditions: list[ConditionSpec] = []
    for target in cfg.targets:
        for sigma_w in cfg.sigma_ws:
            for train_size in cfg.train_sizes:
                for optimizer in cfg.optimizers:
                    for mode in cfg.train_set_modes:
                        replicates = cfg.fixed_set_replicates if mode == "fixed" else 1
                        for replicate in range(replicates):
                            conditions.append(
                                ConditionSpec(
                                    target.name,
                                    sigma_w,
                                    train_size,
                                    optimizer,
                                    mode,
                                    replicate,
                                )
                            )
    return tuple(conditions)


# =============================================================================
# 128-bit 函数表示与论文 LZ complexity
# =============================================================================


def lz_word_count(bits: np.ndarray) -> int:
    """逐字复现论文公开代码采用的 LZ76 边界约定。"""
    sequence = np.asarray(bits, dtype=np.uint8).reshape(-1)
    width = int(sequence.size)
    if width == 0:
        return 0
    if width == 1:
        return 1

    n = width - 1
    i = 0
    complexity = 1
    length = 1
    match = 1
    match_max = 1
    while True:
        if sequence[i + match - 1] == sequence[length + match - 1]:
            match += 1
            if length + match >= n - 1:
                complexity += 1
                break
        else:
            match_max = max(match_max, match)
            i += 1
            if i == length:
                complexity += 1
                length += match_max
                if length + 1 > n:
                    break
                i = 0
                match = 1
                match_max = 1
            else:
                match = 1
    return complexity


def lz_complexity_x2(bits: np.ndarray) -> int:
    """返回 2*K，避免 0.5 分辨率的浮点键。"""
    sequence = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if sequence.size != INPUT_COUNT:
        raise ValueError("LZ complexity 只接受 128-bit 完整函数")
    if bool(np.all(sequence == sequence[0])):
        return 14  # 2*log2(128)
    forward = lz_word_count(sequence)
    reverse = lz_word_count(sequence[::-1])
    return 7 * (forward + reverse)


def lz_word_count_batch(bits: np.ndarray) -> np.ndarray:
    """把 LZ76 状态机沿函数样本轴向量化，避免逐行 Python 循环。"""
    rows = np.asarray(bits, dtype=np.uint8)
    if rows.ndim != 2:
        raise ValueError("bits 必须是二维数组")
    batch, width = rows.shape
    if width == 0:
        return np.zeros(batch, dtype=np.int16)
    if width == 1:
        return np.ones(batch, dtype=np.int16)

    # 这里保留公开实现的 n = len(sequence) - 1 以及两个非标准边界条件，
    # 以便 LZ 数值与论文图表逐点一致。
    n = width - 1
    source = np.zeros(batch, dtype=np.int16)
    complexity = np.ones(batch, dtype=np.int16)
    cursor = np.ones(batch, dtype=np.int16)
    match = np.ones(batch, dtype=np.int16)
    match_max = np.ones(batch, dtype=np.int16)
    active = np.ones(batch, dtype=np.bool_)
    iterations = 0

    while np.any(active):
        iterations += 1
        if iterations > width * width * 2:
            raise RuntimeError("LZ76 batch 状态机未收敛")
        active_indices = np.flatnonzero(active)
        left_positions = source[active_indices] + match[active_indices] - 1
        right_positions = cursor[active_indices] + match[active_indices] - 1
        equal_local = (
            rows[active_indices, left_positions]
            == rows[active_indices, right_positions]
        )
        equal_indices = active_indices[equal_local]
        mismatch_indices = active_indices[~equal_local]

        if equal_indices.size:
            match[equal_indices] += 1
            finished = equal_indices[
                cursor[equal_indices] + match[equal_indices] >= n - 1
            ]
            complexity[finished] += 1
            active[finished] = False

        if mismatch_indices.size:
            match_max[mismatch_indices] = np.maximum(
                match_max[mismatch_indices], match[mismatch_indices]
            )
            source[mismatch_indices] += 1
            boundary = mismatch_indices[
                source[mismatch_indices] == cursor[mismatch_indices]
            ]
            non_boundary = mismatch_indices[
                source[mismatch_indices] != cursor[mismatch_indices]
            ]
            match[non_boundary] = 1

            if boundary.size:
                complexity[boundary] += 1
                cursor[boundary] += match_max[boundary]
                finished = boundary[cursor[boundary] + 1 > n]
                active[finished] = False
                continuing = boundary[cursor[boundary] + 1 <= n]
                source[continuing] = 0
                match[continuing] = 1
                match_max[continuing] = 1

    return complexity


def lz_complexity_x2_batch(bits: np.ndarray) -> np.ndarray:
    rows = np.asarray(bits, dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] != INPUT_COUNT:
        raise ValueError("LZ complexity batch 只接受 [N, 128] 数组")
    if rows.shape[0] == 0:
        return np.empty(0, dtype=np.int16)
    constant = np.all(rows == rows[:, :1], axis=1)
    forward = lz_word_count_batch(rows)
    reverse = lz_word_count_batch(rows[:, ::-1])
    result = (7 * (forward + reverse)).astype(np.int16)
    result[constant] = 14
    return result


def pack_function_bits(bits: np.ndarray) -> np.ndarray:
    rows = np.asarray(bits, dtype=np.uint8)
    return np.packbits(rows, axis=-1, bitorder="little")


def unpack_function_bits(packed: np.ndarray) -> np.ndarray:
    rows = np.asarray(packed, dtype=np.uint8)
    return np.unpackbits(rows, axis=-1, count=INPUT_COUNT, bitorder="little")


def row_keys(rows: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(rows)
    dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(dtype).reshape(-1)


# =============================================================================
# 模型：首维是相互独立、无参数共享的 FCN
# =============================================================================


class BatchedTanhFCN(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        spec: ModelSpec,
        sigma_w: float,
        initialization_seed: int,
        device: torch.device,
        trainable: bool,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(initialization_seed))

        widths = [spec.input_bits] + [spec.hidden_size] * spec.hidden_layers + [1]
        weights: list[nn.Parameter] = []
        biases: list[nn.Parameter] = []
        for in_width, out_width in zip(widths[:-1], widths[1:]):
            weight = torch.randn(
                ensemble_size,
                out_width,
                in_width,
                generator=generator,
                device=device,
                dtype=torch.float32,
            ) * (float(sigma_w) / math.sqrt(in_width))
            if spec.bias_init_std == 0.0:
                bias = torch.zeros(
                    ensemble_size,
                    out_width,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                bias = torch.randn(
                    ensemble_size,
                    out_width,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                ) * float(spec.bias_init_std)
            weights.append(nn.Parameter(weight, requires_grad=trainable))
            biases.append(nn.Parameter(bias, requires_grad=trainable))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None, :]
            if index != last:
                hidden = torch.tanh(hidden)
        return hidden

    def forward_subset(
        self,
        inputs: torch.Tensor,
        model_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            selected_weight = weight.index_select(0, model_indices)
            selected_bias = bias.index_select(0, model_indices)
            hidden = torch.bmm(hidden, selected_weight.transpose(1, 2))
            hidden = hidden + selected_bias[:, None, :]
            if index != last:
                hidden = torch.tanh(hidden)
        return hidden


@torch.inference_mode()
def evaluate_subset(
    model: BatchedTanhFCN,
    probe_inputs: torch.Tensor,
    model_indices_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if model_indices_np.size == 0:
        return (
            np.empty((0, PACKED_BYTES), dtype=np.uint8),
            np.empty((0, INPUT_COUNT), dtype=np.float16),
        )
    device = probe_inputs.device
    indices = torch.as_tensor(model_indices_np, dtype=torch.long, device=device)
    inputs = probe_inputs[None].expand(indices.numel(), -1, -1)
    logits = model.forward_subset(inputs, indices).squeeze(-1)
    bits = (logits > 0).to(torch.uint8).cpu().numpy()
    packed = pack_function_bits(bits)
    return packed, logits.cpu().to(torch.float16).numpy()


@torch.inference_mode()
def evaluate_all(
    model: BatchedTanhFCN,
    probe_inputs: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    count = model.weights[0].shape[0]
    logits = model(probe_inputs[None].expand(count, -1, -1)).squeeze(-1)
    bits = (logits > 0).to(torch.uint8).cpu().numpy()
    return pack_function_bits(bits), logits.cpu().to(torch.float16).numpy()


# =============================================================================
# 初始化先验及静态 averaged posterior
# =============================================================================


def log_fit_probability_without_replacement(correct_count: int, train_size: int) -> float:
    if correct_count < train_size:
        return -math.inf
    return (
        math.lgamma(correct_count + 1)
        - math.lgamma(train_size + 1)
        - math.lgamma(correct_count - train_size + 1)
        - math.lgamma(INPUT_COUNT + 1)
        + math.lgamma(train_size + 1)
        + math.lgamma(INPUT_COUNT - train_size + 1)
    )


def likelihood_lookup(train_size: int) -> np.ndarray:
    values = np.zeros(INPUT_COUNT + 1, dtype=np.float64)
    for correct in range(INPUT_COUNT + 1):
        log_value = log_fit_probability_without_replacement(correct, train_size)
        values[correct] = 0.0 if not math.isfinite(log_value) else math.exp(log_value)
    return values


def prior_accumulator_keys(cfg: EffectiveConfig) -> tuple[tuple[str, int], ...]:
    return tuple((target.name, size) for target in cfg.targets for size in cfg.train_sizes)


def empty_prior_accumulators(cfg: EffectiveConfig) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "prior_lz_counts": np.zeros(513, dtype=np.int64),
        "prior_hamming_counts": np.zeros(INPUT_COUNT + 1, dtype=np.int64),
    }
    for target_name, train_size in prior_accumulator_keys(cfg):
        prefix = f"{target_name}__m{train_size}"
        arrays[f"{prefix}__weight_sum"] = np.zeros(1, dtype=np.float64)
        arrays[f"{prefix}__weight_sq_sum"] = np.zeros(1, dtype=np.float64)
        arrays[f"{prefix}__lz_weight"] = np.zeros(513, dtype=np.float64)
        arrays[f"{prefix}__error_weight"] = np.zeros(INPUT_COUNT + 1, dtype=np.float64)
        arrays[f"{prefix}__bit_weight"] = np.zeros(INPUT_COUNT, dtype=np.float64)
    return arrays


def sample_initialization_prior(
    cfg: EffectiveConfig,
    sigma_w: float,
    result_dir: Path,
    signature: str,
) -> dict[str, np.ndarray]:
    sigma_tag = str(sigma_w).replace(".", "p")
    output = result_dir / "prior" / f"prior_sw_{sigma_tag}.npz"
    metadata_path = output.with_suffix(".json")
    if Config.RESUME_EXISTING and output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            arrays = load_npz(output)
            print(f"[prior sw={sigma_w:g}] 复用已有 {metadata['samples']:,} 次采样")
            return arrays

    device = torch.device(cfg.device)
    probe = all_boolean_inputs(device)
    targets = {target.name: target_array(target) for target in cfg.targets}
    lookups = {size: likelihood_lookup(size) for size in cfg.train_sizes}
    arrays = empty_prior_accumulators(cfg)
    started = time.perf_counter()
    ranges = list(chunk_ranges(cfg.prior_samples, cfg.prior_chunk_size))

    for chunk_index, start, end in ranges:
        count = end - start
        model = BatchedTanhFCN(
            count,
            cfg.model,
            sigma_w,
            derive_seed(cfg.prior_seed, sigma_w, chunk_index),
            device,
            trainable=False,
        )
        packed, _ = evaluate_all(model, probe)
        bits = unpack_function_bits(packed)
        lz_x2 = lz_complexity_x2_batch(bits)
        hamming = bits.sum(axis=1).astype(np.int16)
        arrays["prior_lz_counts"] += np.bincount(lz_x2, minlength=513)
        arrays["prior_hamming_counts"] += np.bincount(
            hamming, minlength=INPUT_COUNT + 1
        )

        for target_name, train_size in prior_accumulator_keys(cfg):
            target = targets[target_name]
            error_count = np.count_nonzero(bits != target[None, :], axis=1)
            correct_count = INPUT_COUNT - error_count
            weights = lookups[train_size][correct_count]
            prefix = f"{target_name}__m{train_size}"
            arrays[f"{prefix}__weight_sum"][0] += float(weights.sum())
            arrays[f"{prefix}__weight_sq_sum"][0] += float(np.square(weights).sum())
            arrays[f"{prefix}__lz_weight"] += np.bincount(
                lz_x2, weights=weights, minlength=513
            )
            arrays[f"{prefix}__error_weight"] += np.bincount(
                error_count, weights=weights, minlength=INPUT_COUNT + 1
            )
            arrays[f"{prefix}__bit_weight"] += (weights[:, None] * bits).sum(axis=0)

        del model, packed, bits
        clear_device_cache(device)
        if (chunk_index + 1) % max(1, len(ranges) // 20) == 0 or end == cfg.prior_samples:
            elapsed = time.perf_counter() - started
            print(
                f"[prior sw={sigma_w:g}] {end:,}/{cfg.prior_samples:,} | "
                f"{end / max(elapsed, 1e-9):,.0f} init/s"
            )

    save_npz_atomic(output, arrays)
    save_json(
        metadata_path,
        {
            "config_signature": signature,
            "sigma_w": sigma_w,
            "samples": cfg.prior_samples,
            "elapsed_seconds": time.perf_counter() - started,
            "note": (
                "静态 posterior 使用随机训练集无放回抽样的精确拟合概率加权；"
                "ESS 过小时不得据此判断与 SGD 的距离。"
            ),
        },
    )
    return arrays


# =============================================================================
# advSGD / Adam 训练与异步 post-fit endpoint
# =============================================================================


def make_training_indices(
    count: int,
    train_size: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "fixed":
        selected = np.sort(rng.choice(INPUT_COUNT, size=train_size, replace=False))
        return np.broadcast_to(selected[None, :], (count, train_size)).copy().astype(np.uint8)
    scores = rng.random((count, INPUT_COUNT), dtype=np.float32)
    selected = np.argpartition(scores, train_size - 1, axis=1)[:, :train_size]
    selected.sort(axis=1)
    return selected.astype(np.uint8)


@torch.inference_mode()
def training_status(
    model: BatchedTanhFCN,
    train_inputs: torch.Tensor,
    signed_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(train_inputs).squeeze(-1)
    margins = logits * signed_targets
    exact = (margins > 0).all(dim=1)
    minimum_margin = margins.min(dim=1).values
    losses = F.softplus(-margins).mean(dim=1)
    return exact, minimum_margin, losses


def select_minibatch_positions(
    optimizer_name: str,
    difficulty: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if optimizer_name == "adam_fullbatch":
        return torch.arange(difficulty.shape[1], device=difficulty.device)[None].expand(
            difficulty.shape[0], -1
        )
    if optimizer_name == "advsgd":
        probabilities = torch.softmax(difficulty, dim=1)
    else:
        probabilities = torch.ones_like(difficulty)
    return torch.multinomial(
        probabilities,
        num_samples=batch_size,
        replacement=False,
        generator=generator,
    )


def train_condition_chunk(
    cfg: EffectiveConfig,
    condition: ConditionSpec,
    target: TargetSpec,
    count: int,
    chunk_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(cfg.device)
    init_seed = derive_seed(cfg.train_seed, condition.name, chunk_index, "init")
    data_seed = (
        derive_seed(cfg.train_seed, condition.name, "fixed_data")
        if condition.train_set_mode == "fixed"
        else derive_seed(cfg.train_seed, condition.name, chunk_index, "data")
    )
    batch_seed = derive_seed(cfg.train_seed, condition.name, chunk_index, "batch")

    model = BatchedTanhFCN(
        count,
        cfg.model,
        condition.sigma_w,
        init_seed,
        device,
        trainable=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.model.learning_rate,
        eps=cfg.model.adam_eps,
    )
    batch_generator = torch.Generator(device=device.type)
    batch_generator.manual_seed(batch_seed)

    probe = all_boolean_inputs(device)
    target_np = target_array(target)
    target_tensor = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    training_indices_np = make_training_indices(
        count,
        condition.train_size,
        condition.train_set_mode,
        data_seed,
    )
    training_indices = torch.as_tensor(
        training_indices_np.astype(np.int64), dtype=torch.long, device=device
    )
    rows = torch.arange(count, device=device)[:, None]
    all_inputs = probe[None].expand(count, -1, -1)
    train_inputs = all_inputs[rows, training_indices]
    train_targets = target_tensor[training_indices]
    signed_targets = train_targets.mul(2.0).sub(1.0)

    names = endpoint_names(cfg)
    endpoint_count = len(names)
    packed_functions = np.full(
        (endpoint_count, count, PACKED_BYTES), 255, dtype=np.uint8
    )
    probe_logits = np.full(
        (endpoint_count, count, INPUT_COUNT), np.nan, dtype=np.float16
    )
    endpoint_steps = np.full((endpoint_count, count), -1, dtype=np.int32)
    endpoint_train_loss = np.full((endpoint_count, count), np.nan, dtype=np.float32)
    endpoint_min_margin = np.full((endpoint_count, count), np.nan, dtype=np.float32)
    endpoint_train_exact = np.zeros((endpoint_count, count), dtype=np.bool_)
    first_fit_steps = np.full(count, -1, dtype=np.int32)
    recorded = np.zeros((endpoint_count, count), dtype=np.bool_)

    initialization_index = names.index("initialization")
    last_index = names.index("last_step")
    age_indices = {
        age: names.index(f"post_fit_age_{age:06d}") for age in cfg.post_fit_ages
    }

    history = torch.full(
        (count, condition.train_size, cfg.model.adv_history),
        cfg.model.adv_initial_loss,
        dtype=torch.float32,
        device=device,
    )
    history_cursor = torch.zeros(
        (count, condition.train_size), dtype=torch.long, device=device
    )
    difficulty = history.sum(dim=2)
    batch_positions = select_minibatch_positions(
        condition.optimizer,
        difficulty,
        cfg.model.batch_size,
        batch_generator,
    )

    started = time.perf_counter()

    def record(
        endpoint_index: int,
        due_mask: np.ndarray,
        step: int,
        exact: torch.Tensor,
        minimum_margin: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        due_indices = np.flatnonzero(due_mask)
        if due_indices.size == 0:
            return
        due_tensor = torch.as_tensor(due_indices, dtype=torch.long, device=device)
        packed, logits = evaluate_subset(model, probe, due_indices)
        packed_functions[endpoint_index, due_indices] = packed
        probe_logits[endpoint_index, due_indices] = logits
        endpoint_steps[endpoint_index, due_indices] = step
        endpoint_train_loss[endpoint_index, due_indices] = (
            losses.index_select(0, due_tensor).cpu().numpy().astype(np.float32)
        )
        endpoint_min_margin[endpoint_index, due_indices] = (
            minimum_margin.index_select(0, due_tensor).cpu().numpy().astype(np.float32)
        )
        endpoint_train_exact[endpoint_index, due_indices] = (
            exact.index_select(0, due_tensor).cpu().numpy()
        )
        recorded[endpoint_index, due_indices] = True

    exact, minimum_margin, losses = training_status(model, train_inputs, signed_targets)
    all_mask = np.ones(count, dtype=np.bool_)
    record(initialization_index, all_mask, 0, exact, minimum_margin, losses)
    exact_np = exact.cpu().numpy()
    first_fit_steps[exact_np] = 0
    record(age_indices[0], exact_np, 0, exact, minimum_margin, losses)

    final_step = 0
    for step in range(1, cfg.max_total_steps + 1):
        final_step = step
        selected_inputs = train_inputs[rows, batch_positions]
        selected_targets = signed_targets[rows, batch_positions]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        selected_logits = model(selected_inputs).squeeze(-1)
        selected_margins = selected_logits * selected_targets
        loss_by_model = F.softplus(-selected_margins).mean(dim=1)
        loss_by_model.sum().backward()
        optimizer.step()

        # 与公开 advSGD 代码一致：用更新后模型在刚训练样本上的 loss
        # 刷新每个样本最近三次误差，再据此选择下一批难例。
        with torch.inference_mode():
            post_logits = model(selected_inputs).squeeze(-1)
            post_losses = F.softplus(-(post_logits * selected_targets))
            selected_slots = history_cursor[rows, batch_positions]
            history[rows, batch_positions, selected_slots] = post_losses
            history_cursor[rows, batch_positions] = (
                selected_slots + 1
            ) % cfg.model.adv_history

        exact, minimum_margin, losses = training_status(
            model, train_inputs, signed_targets
        )
        exact_np = exact.cpu().numpy()
        newly_fit = (first_fit_steps < 0) & exact_np
        first_fit_steps[newly_fit] = step

        for age, endpoint_index in age_indices.items():
            due = (
                (~recorded[endpoint_index])
                & (first_fit_steps >= 0)
                & (step >= first_fit_steps + age)
            )
            if np.any(due):
                record(endpoint_index, due, step, exact, minimum_margin, losses)

        if bool(recorded[list(age_indices.values())].all()):
            break

        difficulty = history.sum(dim=2)
        batch_positions = select_minibatch_positions(
            condition.optimizer,
            difficulty,
            cfg.model.batch_size,
            batch_generator,
        )

        if step % cfg.log_every_steps == 0:
            fitted = int((first_fit_steps >= 0).sum())
            final_age_done = int(recorded[age_indices[max(cfg.post_fit_ages)]].sum())
            elapsed = time.perf_counter() - started
            print(
                f"      chunk={chunk_index + 1} step={step:,} | "
                f"first_fit={fitted:,}/{count:,} | "
                f"age{max(cfg.post_fit_ages):,}={final_age_done:,}/{count:,} | "
                f"loss={float(losses.mean().item()):.3e} | "
                f"{step / max(elapsed, 1e-9):.1f} step/s"
            )

    exact, minimum_margin, losses = training_status(model, train_inputs, signed_targets)
    record(last_index, all_mask, final_step, exact, minimum_margin, losses)

    arrays = {
        "packed_functions": packed_functions,
        "probe_logits": probe_logits,
        "endpoint_steps": endpoint_steps,
        "endpoint_train_loss": endpoint_train_loss,
        "endpoint_min_margin": endpoint_min_margin,
        "endpoint_train_exact": endpoint_train_exact,
        "recorded": recorded,
        "first_fit_steps": first_fit_steps,
        "training_indices": training_indices_np,
    }
    metadata = {
        "condition": asdict(condition),
        "target_lz": target.expected_lz_x2 / 2,
        "endpoint_names": list(names),
        "count": count,
        "chunk_index": chunk_index,
        "final_step": final_step,
        "fitted_count": int((first_fit_steps >= 0).sum()),
        "all_ages_recorded_count": int(
            recorded[list(age_indices.values())].all(axis=0).sum()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return metadata, arrays


def aggregate_condition_chunks(
    chunk_arrays: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    endpoint_keys = (
        "packed_functions",
        "probe_logits",
        "endpoint_steps",
        "endpoint_train_loss",
        "endpoint_min_margin",
        "endpoint_train_exact",
        "recorded",
    )
    model_keys = ("first_fit_steps", "training_indices")
    aggregate: dict[str, np.ndarray] = {}
    for key in endpoint_keys:
        aggregate[key] = np.concatenate([arrays[key] for arrays in chunk_arrays], axis=1)
    for key in model_keys:
        aggregate[key] = np.concatenate([arrays[key] for arrays in chunk_arrays], axis=0)
    return aggregate


def run_condition(
    cfg: EffectiveConfig,
    condition: ConditionSpec,
    target: TargetSpec,
    result_dir: Path,
    signature: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    condition_dir = result_dir / "conditions" / condition.name
    aggregate_path = condition_dir / "aggregate.npz"
    metadata_path = condition_dir / "aggregate.json"
    if Config.RESUME_EXISTING and aggregate_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            print(f"[condition] 复用 {condition.name}")
            return metadata, load_npz(aggregate_path)

    print(
        f"\n[condition] {condition.name} | models={cfg.model_count:,} | "
        f"ages={list(cfg.post_fit_ages)}"
    )
    chunk_arrays: list[dict[str, np.ndarray]] = []
    chunk_metadata: list[dict[str, Any]] = []
    started = time.perf_counter()

    for chunk_index, start, end in chunk_ranges(cfg.model_count, cfg.train_chunk_size):
        chunk_path = condition_dir / "chunks" / f"chunk_{chunk_index:04d}.npz"
        chunk_meta_path = chunk_path.with_suffix(".json")
        arrays: dict[str, np.ndarray]
        metadata: dict[str, Any]
        if Config.RESUME_EXISTING and chunk_path.exists() and chunk_meta_path.exists():
            metadata = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
            if metadata.get("config_signature") == signature:
                print(
                    f"  [chunk {chunk_index + 1}] 复用 {end - start:,} 个模型"
                )
                arrays = load_npz(chunk_path)
                chunk_arrays.append(arrays)
                chunk_metadata.append(metadata)
                continue

        metadata, arrays = train_condition_chunk(
            cfg,
            condition,
            target,
            end - start,
            chunk_index,
        )
        metadata["config_signature"] = signature
        metadata["model_start"] = start
        metadata["model_end"] = end
        save_npz_atomic(chunk_path, arrays)
        save_json(chunk_meta_path, metadata)
        chunk_arrays.append(arrays)
        chunk_metadata.append(metadata)
        clear_device_cache(torch.device(cfg.device))

    aggregate = aggregate_condition_chunks(chunk_arrays)
    names = endpoint_names(cfg)
    fitted = aggregate["first_fit_steps"] >= 0
    final_age_index = names.index(f"post_fit_age_{max(cfg.post_fit_ages):06d}")
    metadata = {
        "config_signature": signature,
        "condition": asdict(condition),
        "target": asdict(target),
        "endpoint_names": list(names),
        "model_count": cfg.model_count,
        "fitted_count": int(fitted.sum()),
        "fitted_fraction": float(fitted.mean()),
        "all_postfit_recorded_count": int(
            aggregate["recorded"][final_age_index].sum()
        ),
        "first_fit_step_median": (
            float(np.median(aggregate["first_fit_steps"][fitted]))
            if np.any(fitted)
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "chunks": chunk_metadata,
    }
    save_npz_atomic(aggregate_path, aggregate)
    save_json(metadata_path, metadata)
    return metadata, aggregate


def run_training_grid(
    cfg: EffectiveConfig,
    result_dir: Path,
    signature: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    target_lookup = {target.name: target for target in cfg.targets}
    metadata_by_condition: dict[str, dict[str, Any]] = {}
    arrays_by_condition: dict[str, dict[str, np.ndarray]] = {}
    conditions = build_conditions(cfg)
    for condition_index, condition in enumerate(conditions):
        print(f"\n=== 条件 {condition_index + 1}/{len(conditions)} ===")
        metadata, arrays = run_condition(
            cfg,
            condition,
            target_lookup[condition.target_name],
            result_dir,
            signature,
        )
        metadata_by_condition[condition.name] = metadata
        arrays_by_condition[condition.name] = arrays
    return metadata_by_condition, arrays_by_condition


# =============================================================================
# 分布指标与置换检验
# =============================================================================


def normalize_histogram(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    return array / total if total > 0 else np.zeros_like(array)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    a = normalize_histogram(left)
    b = normalize_histogram(right)
    width = max(a.size, b.size)
    a = np.pad(a, (0, width - a.size))
    b = np.pad(b, (0, width - b.size))
    return float(0.5 * np.abs(a - b).sum())


def js_divergence_bits(left: np.ndarray, right: np.ndarray) -> float:
    a = normalize_histogram(left)
    b = normalize_histogram(right)
    width = max(a.size, b.size)
    a = np.pad(a, (0, width - a.size))
    b = np.pad(b, (0, width - b.size))
    midpoint = 0.5 * (a + b)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl(a, midpoint) + 0.5 * kl(b, midpoint)


def integer_histogram(values: np.ndarray, minimum_length: int = 0) -> np.ndarray:
    ints = np.asarray(values, dtype=np.int64)
    if ints.size == 0:
        return np.zeros(minimum_length, dtype=np.int64)
    return np.bincount(ints, minlength=max(minimum_length, int(ints.max()) + 1))


def categorical_tv(left: np.ndarray, right: np.ndarray) -> float:
    return total_variation(integer_histogram(left), integer_histogram(right))


def function_distribution(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = row_keys(rows)
    unique, counts = np.unique(keys, return_counts=True)
    return unique, counts.astype(np.int64)


def function_tv(left: np.ndarray, right: np.ndarray) -> float:
    left_keys = row_keys(left)
    right_keys = row_keys(right)
    combined = np.concatenate([left_keys, right_keys])
    _, inverse = np.unique(combined, return_inverse=True)
    left_counts = np.bincount(inverse[: left_keys.size], minlength=int(inverse.max()) + 1)
    right_counts = np.bincount(inverse[left_keys.size :], minlength=int(inverse.max()) + 1)
    return total_variation(left_counts, right_counts)


def empirical_function_entropy(rows: np.ndarray) -> float:
    _, counts = function_distribution(rows)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def empirical_pairwise_agreement(bits: np.ndarray) -> float:
    count = bits.shape[0]
    if count < 2:
        return float("nan")
    ones = bits.sum(axis=0, dtype=np.int64)
    zeros = count - ones
    agreeing_pairs = ones * (ones - 1) + zeros * (zeros - 1)
    return float(np.mean(agreeing_pairs / (count * (count - 1))))


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan")
    if data.size == 1 or repeats <= 0:
        return float(data.mean()), float(data.mean())
    estimates: list[np.ndarray] = []
    remaining = repeats
    while remaining > 0:
        batch = min(100, remaining)
        indices = rng.integers(0, data.size, size=(batch, data.size))
        estimates.append(data[indices].mean(axis=1))
        remaining -= batch
    samples = np.concatenate(estimates)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_categorical_permutation(
    left: np.ndarray,
    right: np.ndarray,
    observed: float,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if repeats <= 0 or left.size == 0:
        return {"p": float("nan"), "null_q95": float("nan"), "null_q99": float("nan")}
    null = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        swap = rng.random(left.size) < 0.5
        perm_left = np.where(swap, right, left)
        perm_right = np.where(swap, left, right)
        null[index] = categorical_tv(perm_left, perm_right)
    return {
        "p": float((1 + np.count_nonzero(null >= observed)) / (repeats + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
    }


def paired_function_permutation(
    left: np.ndarray,
    right: np.ndarray,
    observed: float,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if repeats <= 0 or left.shape[0] == 0:
        return {"p": float("nan"), "null_q95": float("nan"), "null_q99": float("nan")}
    null = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        swap = rng.random(left.shape[0]) < 0.5
        perm_left = left.copy()
        perm_right = right.copy()
        perm_left[swap] = right[swap]
        perm_right[swap] = left[swap]
        null[index] = function_tv(perm_left, perm_right)
    return {
        "p": float((1 + np.count_nonzero(null >= observed)) / (repeats + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
    }


def paired_marginal_permutation(
    left_bits: np.ndarray,
    right_bits: np.ndarray,
    observed: float,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if repeats <= 0 or left_bits.shape[0] == 0:
        return {"p": float("nan"), "null_q95": float("nan"), "null_q99": float("nan")}
    null = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        swap = rng.random(left_bits.shape[0]) < 0.5
        left = left_bits.copy()
        right = right_bits.copy()
        left[swap] = right_bits[swap]
        right[swap] = left_bits[swap]
        null[index] = float(np.abs(left.mean(axis=0) - right.mean(axis=0)).mean())
    return {
        "p": float((1 + np.count_nonzero(null >= observed)) / (repeats + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
    }


def endpoint_feature_arrays(
    arrays: dict[str, np.ndarray],
    endpoint_index: int,
    target: TargetSpec,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    valid = arrays["endpoint_steps"][endpoint_index] >= 0
    if valid_mask is not None:
        valid &= valid_mask
    model_indices = np.flatnonzero(valid)
    packed = arrays["packed_functions"][endpoint_index, model_indices]
    bits = unpack_function_bits(packed)
    target_bits = target_array(target)
    mismatch = bits != target_bits[None, :]
    training_indices = arrays["training_indices"][model_indices].astype(np.int64)
    train_errors = np.take_along_axis(mismatch, training_indices, axis=1).sum(axis=1)
    full_errors = mismatch.sum(axis=1)
    test_errors = full_errors - train_errors
    lz_x2 = lz_complexity_x2_batch(bits)
    return {
        "model_indices": model_indices,
        "packed": packed,
        "bits": bits,
        "lz_x2": lz_x2,
        "full_errors": full_errors.astype(np.int16),
        "train_errors": train_errors.astype(np.int16),
        "test_errors": test_errors.astype(np.int16),
        "hamming_weight": bits.sum(axis=1).astype(np.int16),
    }


def summarize_endpoint(
    condition: ConditionSpec,
    endpoint_name: str,
    endpoint_index: int,
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    model_indices = features["model_indices"]
    count = int(model_indices.size)
    packed = features["packed"]
    bits = features["bits"]
    _, function_counts = function_distribution(packed)
    test_denominator = INPUT_COUNT - condition.train_size
    return {
        "condition": condition.name,
        **asdict(condition),
        "endpoint": endpoint_name,
        "endpoint_age": (
            int(endpoint_name.rsplit("_", 1)[1])
            if endpoint_name.startswith("post_fit_age_")
            else None
        ),
        "count": count,
        "mean_step": (
            float(arrays["endpoint_steps"][endpoint_index, model_indices].mean())
            if count
            else None
        ),
        "mean_train_loss": (
            float(arrays["endpoint_train_loss"][endpoint_index, model_indices].mean())
            if count
            else None
        ),
        "train_exact_fraction": (
            float(arrays["endpoint_train_exact"][endpoint_index, model_indices].mean())
            if count
            else None
        ),
        "mean_lz": float(features["lz_x2"].mean() / 2) if count else None,
        "median_lz": float(np.median(features["lz_x2"]) / 2) if count else None,
        "mean_full_error": float(features["full_errors"].mean()) if count else None,
        "mean_test_error_rate": (
            float(features["test_errors"].mean() / test_denominator) if count else None
        ),
        "mean_hamming_weight": (
            float(features["hamming_weight"].mean()) if count else None
        ),
        "unique_functions": int(function_counts.size),
        "top_function_fraction": (
            float(function_counts.max() / function_counts.sum()) if count else None
        ),
        "function_entropy_bits": empirical_function_entropy(packed) if count else None,
        "mean_pairwise_agreement": empirical_pairwise_agreement(bits) if count else None,
    }


def compare_endpoints(
    cfg: EffectiveConfig,
    condition: ConditionSpec,
    target: TargetSpec,
    arrays: dict[str, np.ndarray],
    left_name: str,
    right_name: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    names = endpoint_names(cfg)
    left_index = names.index(left_name)
    right_index = names.index(right_name)
    common = (arrays["endpoint_steps"][left_index] >= 0) & (
        arrays["endpoint_steps"][right_index] >= 0
    )
    left = endpoint_feature_arrays(arrays, left_index, target, common)
    right = endpoint_feature_arrays(arrays, right_index, target, common)
    count = left["bits"].shape[0]
    if count == 0:
        return {
            "condition": condition.name,
            "left_endpoint": left_name,
            "right_endpoint": right_name,
            "count": 0,
        }

    changed = np.any(left["packed"] != right["packed"], axis=1)
    hamming_distance = np.count_nonzero(left["bits"] != right["bits"], axis=1)
    lz_delta = (right["lz_x2"].astype(np.float64) - left["lz_x2"]) / 2.0
    error_delta = right["full_errors"].astype(np.float64) - left["full_errors"]
    lz_tv = categorical_tv(left["lz_x2"], right["lz_x2"])
    error_tv = categorical_tv(left["full_errors"], right["full_errors"])
    exact_tv = function_tv(left["packed"], right["packed"])
    marginal_shift = np.abs(left["bits"].mean(axis=0) - right["bits"].mean(axis=0))

    lz_test = paired_categorical_permutation(
        left["lz_x2"],
        right["lz_x2"],
        lz_tv,
        cfg.permutation_repeats,
        rng,
    )
    error_test = paired_categorical_permutation(
        left["full_errors"],
        right["full_errors"],
        error_tv,
        cfg.permutation_repeats,
        rng,
    )
    function_test = paired_function_permutation(
        left["packed"],
        right["packed"],
        exact_tv,
        min(cfg.permutation_repeats, 100),
        rng,
    )
    marginal_test = paired_marginal_permutation(
        left["bits"],
        right["bits"],
        float(marginal_shift.mean()),
        cfg.permutation_repeats,
        rng,
    )
    lz_ci = bootstrap_mean_ci(lz_delta, cfg.bootstrap_repeats, rng)
    error_ci = bootstrap_mean_ci(error_delta, cfg.bootstrap_repeats, rng)

    return {
        "condition": condition.name,
        **asdict(condition),
        "left_endpoint": left_name,
        "right_endpoint": right_name,
        "count": count,
        "function_changed_fraction": float(changed.mean()),
        "mean_function_hamming_distance": float(hamming_distance.mean()),
        "median_function_hamming_distance": float(np.median(hamming_distance)),
        "mean_lz_delta": float(lz_delta.mean()),
        "mean_lz_delta_ci_low": lz_ci[0],
        "mean_lz_delta_ci_high": lz_ci[1],
        "mean_full_error_delta": float(error_delta.mean()),
        "mean_full_error_delta_ci_low": error_ci[0],
        "mean_full_error_delta_ci_high": error_ci[1],
        "lz_distribution_tv": lz_tv,
        "lz_distribution_js_bits": js_divergence_bits(
            integer_histogram(left["lz_x2"]), integer_histogram(right["lz_x2"])
        ),
        "lz_tv_permutation_p": lz_test["p"],
        "lz_tv_null_q99": lz_test["null_q99"],
        "error_distribution_tv": error_tv,
        "error_distribution_js_bits": js_divergence_bits(
            integer_histogram(left["full_errors"]),
            integer_histogram(right["full_errors"]),
        ),
        "error_tv_permutation_p": error_test["p"],
        "error_tv_null_q99": error_test["null_q99"],
        "exact_function_tv": exact_tv,
        "exact_function_tv_permutation_p": function_test["p"],
        "exact_function_tv_null_q99": function_test["null_q99"],
        "mean_marginal_prediction_shift": float(marginal_shift.mean()),
        "max_marginal_prediction_shift": float(marginal_shift.max()),
        "marginal_shift_permutation_p": marginal_test["p"],
        "marginal_shift_null_q99": marginal_test["null_q99"],
    }


def static_prediction_from_prior(
    prior_arrays: dict[str, np.ndarray],
    target_name: str,
    train_size: int,
) -> dict[str, Any]:
    prefix = f"{target_name}__m{train_size}"
    weight_sum = float(prior_arrays[f"{prefix}__weight_sum"][0])
    weight_sq_sum = float(prior_arrays[f"{prefix}__weight_sq_sum"][0])
    ess = weight_sum * weight_sum / weight_sq_sum if weight_sq_sum > 0 else 0.0
    return {
        "weight_sum": weight_sum,
        "ess": ess,
        "lz_distribution": normalize_histogram(prior_arrays[f"{prefix}__lz_weight"]),
        "error_distribution": normalize_histogram(
            prior_arrays[f"{prefix}__error_weight"]
        ),
        "bit_marginals": (
            prior_arrays[f"{prefix}__bit_weight"] / weight_sum
            if weight_sum > 0
            else np.full(INPUT_COUNT, np.nan)
        ),
    }


def compare_endpoint_to_static(
    cfg: EffectiveConfig,
    condition: ConditionSpec,
    target: TargetSpec,
    endpoint_name: str,
    arrays: dict[str, np.ndarray],
    prior_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    endpoint_index = endpoint_names(cfg).index(endpoint_name)
    features = endpoint_feature_arrays(arrays, endpoint_index, target)
    prediction = static_prediction_from_prior(
        prior_arrays, target.name, condition.train_size
    )
    count = int(features["bits"].shape[0])
    base_row = {
        "condition": condition.name,
        **asdict(condition),
        "endpoint": endpoint_name,
        "count": count,
        "prior_importance_ess": prediction["ess"],
        "prior_estimate_reliable": bool(prediction["ess"] >= 100),
    }
    if count == 0:
        return {
            **base_row,
            "lz_tv_vs_static": None,
            "lz_js_vs_static_bits": None,
            "error_tv_vs_static": None,
            "error_js_vs_static_bits": None,
            "mean_marginal_shift_vs_static": None,
            "max_marginal_shift_vs_static": None,
        }

    empirical_lz = integer_histogram(features["lz_x2"], minimum_length=513)
    empirical_error = integer_histogram(
        features["full_errors"], minimum_length=INPUT_COUNT + 1
    )
    empirical_bits = features["bits"].mean(axis=0)
    marginal_shift = np.abs(empirical_bits - prediction["bit_marginals"])
    return {
        **base_row,
        "lz_tv_vs_static": total_variation(
            empirical_lz, prediction["lz_distribution"]
        ),
        "lz_js_vs_static_bits": js_divergence_bits(
            empirical_lz, prediction["lz_distribution"]
        ),
        "error_tv_vs_static": total_variation(
            empirical_error, prediction["error_distribution"]
        ),
        "error_js_vs_static_bits": js_divergence_bits(
            empirical_error, prediction["error_distribution"]
        ),
        "mean_marginal_shift_vs_static": float(np.nanmean(marginal_shift)),
        "max_marginal_shift_vs_static": float(np.nanmax(marginal_shift)),
    }


# =============================================================================
# 汇总、作图、打包
# =============================================================================


def analyze_results(
    cfg: EffectiveConfig,
    result_dir: Path,
    metadata_by_condition: dict[str, dict[str, Any]],
    arrays_by_condition: dict[str, dict[str, np.ndarray]],
    priors: dict[float, dict[str, np.ndarray]],
) -> dict[str, Any]:
    target_lookup = {target.name: target for target in cfg.targets}
    condition_lookup = {condition.name: condition for condition in build_conditions(cfg)}
    rng = np.random.default_rng(cfg.analysis_seed)
    endpoint_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    names = endpoint_names(cfg)
    fit_name = "post_fit_age_000000"

    for condition_name, arrays in arrays_by_condition.items():
        condition = condition_lookup[condition_name]
        target = target_lookup[condition.target_name]
        for endpoint_index, endpoint_name in enumerate(names):
            features = endpoint_feature_arrays(arrays, endpoint_index, target)
            endpoint_rows.append(
                summarize_endpoint(
                    condition,
                    endpoint_name,
                    endpoint_index,
                    features,
                    arrays,
                )
            )

        for age in cfg.post_fit_ages[1:]:
            right_name = f"post_fit_age_{age:06d}"
            print(f"[analysis] {condition.name}: fit -> age {age:,}")
            drift_rows.append(
                compare_endpoints(
                    cfg,
                    condition,
                    target,
                    arrays,
                    fit_name,
                    right_name,
                    rng,
                )
            )

        if condition.train_set_mode == "resampled" and condition.sigma_w in priors:
            for endpoint_name in [
                fit_name,
                *(f"post_fit_age_{age:06d}" for age in cfg.post_fit_ages[1:]),
            ]:
                static_rows.append(
                    compare_endpoint_to_static(
                        cfg,
                        condition,
                        target,
                        endpoint_name,
                        arrays,
                        priors[condition.sigma_w],
                    )
                )

    analysis_dir = result_dir / "analysis"
    write_csv(analysis_dir / "endpoint_summary.csv", endpoint_rows)
    write_csv(analysis_dir / "postfit_drift.csv", drift_rows)
    write_csv(analysis_dir / "static_posterior_comparison.csv", static_rows)

    strongest_drift = sorted(
        (row for row in drift_rows if row.get("count", 0) > 0),
        key=lambda row: float(row.get("mean_marginal_prediction_shift", 0.0)),
        reverse=True,
    )
    summary = {
        "protocol_version": cfg.protocol_version,
        "profile": cfg.profile,
        "condition_count": len(arrays_by_condition),
        "endpoint_rows": len(endpoint_rows),
        "drift_rows": len(drift_rows),
        "static_rows": len(static_rows),
        "condition_metadata": metadata_by_condition,
        "strongest_marginal_drifts": strongest_drift[:10],
        "decision_rule": (
            "若 post-fit 的 LZ、泛化误差或逐输入边际分布漂移稳定超过配对置换零分布，"
            "则首次零错误并非静态函数 posterior 的稳定终点。exact-function TV 在巨大稀疏函数"
            "空间中必须与其置换零分布一起解释。静态 prior importance ESS<100 的条件不得"
            "用来评价 SGD 与 Bayesian posterior 的绝对距离。"
        ),
    }
    save_json(analysis_dir / "summary.json", summary)
    return summary


def create_plots(result_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] 未安装 matplotlib，跳过作图")
        return

    analysis_dir = result_dir / "analysis"
    endpoint_path = analysis_dir / "endpoint_summary.csv"
    drift_path = analysis_dir / "postfit_drift.csv"
    if not endpoint_path.exists() or not drift_path.exists():
        return

    with endpoint_path.open("r", encoding="utf-8-sig", newline="") as handle:
        endpoint_rows = list(csv.DictReader(handle))
    with drift_path.open("r", encoding="utf-8-sig", newline="") as handle:
        drift_rows = list(csv.DictReader(handle))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in endpoint_rows:
        if (
            not row["endpoint"].startswith("post_fit_age_")
            or int(row.get("count", "0")) == 0
            or not row.get("mean_lz")
            or not row.get("mean_test_error_rate")
        ):
            continue
        grouped.setdefault(row["condition"], []).append(row)
    for condition, rows in grouped.items():
        rows.sort(key=lambda row: int(row["endpoint_age"]))
        ages = [int(row["endpoint_age"]) for row in rows]
        x = [age + 1 for age in ages]
        axes[0].plot(x, [float(row["mean_lz"]) for row in rows], marker="o", label=condition)
        axes[1].plot(
            x,
            [float(row["mean_test_error_rate"]) for row in rows],
            marker="o",
            label=condition,
        )
    for axis, ylabel in zip(axes[:2], ["Mean LZ complexity", "Mean test error"]):
        axis.set_xscale("log")
        axis.set_xlabel("Steps after first fit (+1)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    drift_grouped: dict[str, list[dict[str, str]]] = {}
    for row in drift_rows:
        if int(row.get("count", "0")) == 0 or not row.get(
            "function_changed_fraction"
        ):
            continue
        drift_grouped.setdefault(row["condition"], []).append(row)
    for condition, rows in drift_grouped.items():
        rows.sort(key=lambda row: int(row["right_endpoint"].rsplit("_", 1)[1]))
        ages = [int(row["right_endpoint"].rsplit("_", 1)[1]) for row in rows]
        axes[2].plot(
            [age + 1 for age in ages],
            [float(row["function_changed_fraction"]) for row in rows],
            marker="o",
            label=condition,
        )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Steps after first fit (+1)")
    axes[2].set_ylabel("Function changed fraction")
    axes[2].grid(alpha=0.25)
    if drift_grouped:
        axes[2].legend(fontsize=6, bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.suptitle("Mingard 2025 Boolean FCN: post-fit function drift")
    figure.tight_layout()
    figure.savefig(analysis_dir / "postfit_drift_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_result_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(result_dir)
            # aggregate.npz 已包含全部原始 endpoint；跳过重复的可恢复 chunk。
            if "chunks" in relative.parts:
                continue
            archive.write(path, relative.as_posix())
    temporary.replace(archive_path)
    return archive_path


def model_parameter_count(spec: ModelSpec) -> int:
    widths = [spec.input_bits] + [spec.hidden_size] * spec.hidden_layers + [1]
    return int(
        sum(
            in_width * out_width + out_width
            for in_width, out_width in zip(widths[:-1], widths[1:])
        )
    )


def print_key_results(cfg: EffectiveConfig, result_dir: Path, summary: dict[str, Any]) -> None:
    drift_path = result_dir / "analysis" / "postfit_drift.csv"
    if not drift_path.exists():
        return
    with drift_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print("\n=== Post-fit 漂移关键结果 ===")
    for row in rows:
        if int(row.get("count", "0")) == 0:
            continue
        age = int(row["right_endpoint"].rsplit("_", 1)[1])
        print(
            f"{row['condition']} | age={age:,} | "
            f"函数改变={float(row['function_changed_fraction']):.3%} | "
            f"平均Hamming={float(row['mean_function_hamming_distance']):.3f} | "
            f"LZ-TV={float(row['lz_distribution_tv']):.4f} "
            f"(p={float(row['lz_tv_permutation_p']):.4g}) | "
            f"error-TV={float(row['error_distribution_tv']):.4f} "
            f"(p={float(row['error_tv_permutation_p']):.4g}) | "
            f"边际漂移={float(row['mean_marginal_prediction_shift']):.4f}"
        )
    print(
        "\n判读：单条轨迹改变不等于分布改变；优先看 LZ/error/边际分布相对配对置换"
        "零分布的显著性。若多个条件在首次零错误后仍有系统漂移，静态硬条件 posterior "
        "就不能作为训练稳定终点的完整机制。"
    )
    print(f"汇总：{result_dir / 'analysis' / 'summary.json'}")


def main() -> None:
    cfg = get_effective_config()
    validate_config(cfg)
    device = configure_torch(cfg)
    result_dir = Path(cfg.result_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(cfg)
    signature = stable_json_hash(config_payload)
    save_json(
        result_dir / "config.json",
        {
            "config_signature": signature,
            "effective_config": config_payload,
            "paper": {
                "title": "Deep neural networks have an inbuilt Occam's razor",
                "doi": "10.1038/s41467-024-54813-x",
                "matched_protocol": (
                    "n=7, 10x40 tanh FCN, zero-initialized trainable bias, "
                    "sigma_w/sqrt(fan_in), CE, Adam+advSGD batch16"
                ),
            },
        },
    )

    conditions = build_conditions(cfg)
    print("=== Mingard 2025 Boolean posterior：拟合后漂移实验 ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(
        f"profile={cfg.profile} | FCN=7 -> {cfg.model.hidden_size} x "
        f"{cfg.model.hidden_layers} -> 1 | params/model={model_parameter_count(cfg.model):,}"
    )
    print(
        f"conditions={len(conditions)} | models/condition={cfg.model_count:,} | "
        f"total trajectories={len(conditions) * cfg.model_count:,}"
    )
    print(
        f"targets={[target.name for target in cfg.targets]} | sigma_w={cfg.sigma_ws} | "
        f"m={cfg.train_sizes} | ages={cfg.post_fit_ages}"
    )
    print(f"结果目录：{result_dir}")

    priors: dict[float, dict[str, np.ndarray]] = {}
    if cfg.run_initialization_prior:
        print("\n=== 初始化函数先验与静态 averaged posterior ===")
        for sigma_w in cfg.sigma_ws:
            priors[sigma_w] = sample_initialization_prior(
                cfg, sigma_w, result_dir, signature
            )

    print("\n=== SGD 训练轨迹 ===")
    started = time.perf_counter()
    metadata_by_condition, arrays_by_condition = run_training_grid(
        cfg, result_dir, signature
    )
    training_elapsed = time.perf_counter() - started

    print("\n=== 统计分析 ===")
    summary = analyze_results(
        cfg,
        result_dir,
        metadata_by_condition,
        arrays_by_condition,
        priors,
    )
    if Config.CREATE_PLOTS:
        create_plots(result_dir)
    archive = create_result_archive(result_dir) if Config.CREATE_ZIP else None
    print_key_results(cfg, result_dir, summary)
    print("\n=== 实验完成 ===")
    print(f"训练与加载耗时：{training_elapsed:.1f}s")
    if archive is not None:
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
