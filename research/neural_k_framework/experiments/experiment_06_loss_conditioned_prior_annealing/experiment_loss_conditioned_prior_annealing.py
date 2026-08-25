"""
训练 loss 条件下的函数先验退火实验。

核心问题：在网络、初始化分布和训练集完全固定时，先只保留已经满足全部
hard labels 的随机权重，再逐步要求更低的连续 BCE loss，完整 Boolean
function 的概率质量是否会系统性地向少数低有效复杂度函数迁移？

脚本同时实现两种互补口径：

1. 微正则口径：对 hard-exact 权重按 loss 分位数构造嵌套子水平集；
2. 正则口径：使用 exp(-beta * loss) 对先验样本连续重加权，把 beta 当作
   逆温度，得到一条函数空间退火曲线。

重要边界：

- 本实验检验的是固定权重先验测度下的 loss-conditioned 几何，不预设 SGD
  等价于 Gibbs 采样，也不把任何单一代理量称为 Kolmogorov complexity。
- raw BCE 会同时选择 hard function、margin 结构和整体 logit scale，因此还
  报告 RMS-normalized BCE 与固定 logit-scale 子群作为控制。
- 参数空间裸欧氏体积受重参数化影响；这里的“质量”始终相对于脚本明确给定
  的初始化分布，而不是坐标无关的绝对体积。

AutoDL / Jupyter 使用方式：

1. 先把 Config.PROFILE 保持为 "pilot" 运行；
2. 方向成立后改为 "full" 扩大低-loss 尾部样本；
3. 整段复制到 notebook 单元格即可，所有路径都在 Config 中设置；
4. 中断后重跑会复用已完成的 shard。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass
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


class Config:
    # pilot 通常已经足以复核方向；full 用于稳定估计 0.1% 以下尾部。
    PROFILE = "pilot"  # "full" / "pilot" / "smoke"

    RESULT_DIR = script_directory() / "results_loss_conditioned_prior_annealing"
    RESUME_EXISTING = True
    CREATE_ZIP = True

    INPUT_BITS = 3
    ARCHITECTURE = "gelu_ln"
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 3

    GLOBAL_SEED = 20260818
    PRIOR_SEED_BASE = 71_000_000

    PILOT_MODELS = 262_144
    PILOT_MICRO_BATCH = 256
    PILOT_SHARD_SIZE = 16_384

    # 5090 实测 pilot 的 262,144 个 1024x3 网络只需约 4.4 秒前向采样。
    # full 因而提高到 2^22，使四样本条件的最低 0.1% 尾部仍有约 360 个样本。
    FULL_MODELS = 4_194_304
    FULL_MICRO_BATCH = 256
    FULL_SHARD_SIZE = 16_384

    # 每个 fraction 表示保留 hard-exact 子群中 loss 最低的这部分样本。
    QUANTILE_FRACTIONS = (
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

    # beta 是 inverse temperature。较大 beta 会导致 importance weight 退化，
    # 因此必须结合 ESS，而不能只看最末端曲线。
    BETAS = (
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        24.0,
        32.0,
        48.0,
        64.0,
        96.0,
        128.0,
        192.0,
        256.0,
        384.0,
        512.0,
        768.0,
        1_024.0,
        1_536.0,
        2_048.0,
        3_072.0,
        4_096.0,
    )

    FIXED_SCALE_QUANTILES = (0.40, 0.60)
    MIN_RELIABLE_SELECTED = 200
    MIN_RELIABLE_ESS = 200.0
    LOG_INTERVAL_SECONDS = 20.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    input_indices: tuple[int, ...]
    targets: tuple[int, ...]
    description: str
    reference_attractors: tuple[int, ...]


@dataclass(frozen=True)
class EffectiveConfig:
    profile: str
    result_dir: Path
    input_bits: int
    architecture: str
    hidden_size: int
    hidden_layers: int
    model_count: int
    micro_batch: int
    shard_size: int
    global_seed: int
    prior_seed_base: int
    device: str
    allow_tf32: bool
    smoke_test: bool


def experiment_conditions() -> list[ConditionSpec]:
    # 与 boolean_function_posterior_dynamics_v2 使用完全相同的三个约束。
    return [
        ConditionSpec(
            name="single_x0_y0",
            input_indices=(0,),
            targets=(0,),
            description="单个 000 -> 0 约束",
            reference_attractors=(0,),
        ),
        ConditionSpec(
            name="auto_k2_x3-4_y01",
            input_indices=(3, 4),
            targets=(0, 1),
            description="两个对角训练点：110 -> 0, 001 -> 1",
            reference_attractors=(113,),
        ),
        ConditionSpec(
            name="auto_k4_x2-3-4-5_y0011",
            input_indices=(2, 3, 4, 5),
            targets=(0, 0, 1, 1),
            description="四个平衡训练点，x0 的两种取值均出现",
            reference_attractors=(48, 243),
        ),
    ]


def resolve_config() -> EffectiveConfig:
    profile = str(Config.PROFILE).strip().lower()
    if profile == "full":
        model_count = Config.FULL_MODELS
        micro_batch = Config.FULL_MICRO_BATCH
        shard_size = Config.FULL_SHARD_SIZE
        hidden_size = Config.HIDDEN_SIZE
        hidden_layers = Config.HIDDEN_LAYERS
        smoke = False
    elif profile == "pilot":
        model_count = Config.PILOT_MODELS
        micro_batch = Config.PILOT_MICRO_BATCH
        shard_size = Config.PILOT_SHARD_SIZE
        hidden_size = Config.HIDDEN_SIZE
        hidden_layers = Config.HIDDEN_LAYERS
        smoke = False
    elif profile == "smoke":
        model_count = 4_096
        micro_batch = 128
        shard_size = 1_024
        hidden_size = 64
        hidden_layers = 2
        smoke = True
    else:
        raise ValueError(f"未知 PROFILE={Config.PROFILE!r}，只能是 full/pilot/smoke。")

    if Config.INPUT_BITS != 3:
        raise ValueError("当前版本固定为 3-bit -> 1-bit，以便完整枚举 256 个函数。")
    if Config.ARCHITECTURE not in {"gelu_ln", "tanh"}:
        raise ValueError("ARCHITECTURE 只能是 gelu_ln 或 tanh。")
    if shard_size % micro_batch != 0:
        raise ValueError("SHARD_SIZE 必须能被 MICRO_BATCH 整除。")
    if model_count % shard_size != 0:
        raise ValueError("MODEL_COUNT 必须能被 SHARD_SIZE 整除，便于可靠续跑。")

    return EffectiveConfig(
        profile=profile,
        result_dir=Path(Config.RESULT_DIR),
        input_bits=Config.INPUT_BITS,
        architecture=Config.ARCHITECTURE,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        model_count=model_count,
        micro_batch=micro_batch,
        shard_size=shard_size,
        global_seed=Config.GLOBAL_SEED,
        prior_seed_base=Config.PRIOR_SEED_BASE,
        device=Config.DEVICE,
        allow_tf32=Config.ALLOW_TF32,
        smoke_test=smoke,
    )


# =============================================================================
# 通用工具
# =============================================================================


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
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key, "")) for key in fields})


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def truth_table_inputs(input_bits: int, device: torch.device | None = None) -> torch.Tensor:
    rows = 1 << input_bits
    values = torch.arange(rows, dtype=torch.int64, device=device)
    shifts = torch.arange(input_bits, dtype=torch.int64, device=device)
    return ((values[:, None] >> shifts) & 1).to(torch.float32)


def function_ids_from_logits(logits: torch.Tensor) -> torch.Tensor:
    powers = 2 ** torch.arange(logits.shape[1], device=logits.device, dtype=torch.int64)
    return ((logits >= 0).to(torch.int64) * powers[None, :]).sum(dim=1)


def entropy_bits(probability: np.ndarray) -> float:
    values = np.asarray(probability, dtype=np.float64)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum()) if values.size else 0.0


def effective_support(probability: np.ndarray) -> float:
    values = np.asarray(probability, dtype=np.float64)
    denominator = float(np.square(values).sum())
    return 1.0 / denominator if denominator > 0 else 0.0


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return math.nan, math.nan
    normalized = weights / total
    mean = float(np.sum(normalized * values))
    variance = float(np.sum(normalized * np.square(values - mean)))
    return mean, math.sqrt(max(variance, 0.0))


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(first) - np.asarray(second)).sum())


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(first.sum()), 1e-300)
    second /= max(float(second.sum()), 1e-300)
    middle = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        valid = left > 0
        return float(np.sum(left[valid] * np.log2(left[valid] / right[valid])))

    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


# =============================================================================
# 初始化权重先验的高吞吐采样
# =============================================================================


@torch.inference_mode()
def sample_vectorized_mlp_logits(
    cfg: EffectiveConfig,
    count: int,
    generator: torch.Generator,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """只保留 logits，逐层生成并释放随机权重，避免保存整个 ensemble。"""

    hidden = inputs[None, :, :].expand(count, -1, -1)
    in_features = cfg.input_bits
    for _ in range(cfg.hidden_layers):
        bound = 1.0 / math.sqrt(in_features)
        weight = torch.empty(
            count,
            cfg.hidden_size,
            in_features,
            device=inputs.device,
            dtype=torch.float32,
        ).uniform_(-bound, bound, generator=generator)
        bias = torch.empty(
            count,
            cfg.hidden_size,
            device=inputs.device,
            dtype=torch.float32,
        ).uniform_(-bound, bound, generator=generator)
        hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None, :]
        if cfg.architecture == "gelu_ln":
            hidden = F.gelu(hidden)
            mean = hidden.mean(dim=-1, keepdim=True)
            variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
            hidden = (hidden - mean) * torch.rsqrt(variance + 1e-5)
        else:
            hidden = torch.tanh(hidden)
        del weight, bias
        in_features = cfg.hidden_size

    bound = 1.0 / math.sqrt(in_features)
    output_weight = torch.empty(
        count,
        1,
        in_features,
        device=inputs.device,
        dtype=torch.float32,
    ).uniform_(-bound, bound, generator=generator)
    output_bias = torch.empty(
        count,
        1,
        device=inputs.device,
        dtype=torch.float32,
    ).uniform_(-bound, bound, generator=generator)
    logits = torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
    logits = logits + output_bias
    return logits


def config_payload(cfg: EffectiveConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["result_dir"] = str(cfg.result_dir)
    payload["protocol_version"] = "loss_conditioned_prior_annealing_v1"
    payload["quantile_fractions"] = list(Config.QUANTILE_FRACTIONS)
    payload["betas"] = list(Config.BETAS)
    payload["fixed_scale_quantiles"] = list(Config.FIXED_SCALE_QUANTILES)
    payload["conditions"] = [asdict(item) for item in experiment_conditions()]
    return payload


def shard_metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def shard_is_reusable(path: Path, signature: str, expected_count: int) -> bool:
    metadata_path = shard_metadata_path(path)
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("config_signature") == signature
        and int(metadata.get("count", -1)) == expected_count
    )


def sample_prior(cfg: EffectiveConfig, signature: str) -> dict[str, np.ndarray]:
    output_path = cfg.result_dir / "prior_samples.npz"
    output_meta = cfg.result_dir / "prior_samples.json"
    if Config.RESUME_EXISTING and output_path.exists() and output_meta.exists():
        metadata = json.loads(output_meta.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            print("复用已聚合的初始化先验样本。")
            with np.load(output_path, allow_pickle=False) as loaded:
                return {key: loaded[key] for key in loaded.files}

    shard_dir = cfg.result_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    inputs = truth_table_inputs(cfg.input_bits, device=device)
    shard_count = cfg.model_count // cfg.shard_size

    print(
        f"采样初始化先验：models={cfg.model_count:,} | architecture="
        f"{cfg.input_bits}->{cfg.hidden_size}x{cfg.hidden_layers}->1 "
        f"{cfg.architecture} | micro_batch={cfg.micro_batch:,} | shards={shard_count}"
    )
    started = time.perf_counter()
    last_log = started
    for shard_index in range(shard_count):
        shard_start = shard_index * cfg.shard_size
        shard_end = shard_start + cfg.shard_size
        shard_path = shard_dir / f"prior_{shard_start:09d}_{shard_end:09d}.npz"
        if Config.RESUME_EXISTING and shard_is_reusable(
            shard_path, signature, cfg.shard_size
        ):
            continue

        pieces: list[np.ndarray] = []
        for local_start in range(0, cfg.shard_size, cfg.micro_batch):
            global_start = shard_start + local_start
            generator = torch.Generator(device=device)
            generator.manual_seed(
                int(cfg.prior_seed_base + global_start * 1_000_003 + cfg.global_seed)
            )
            logits = sample_vectorized_mlp_logits(
                cfg,
                cfg.micro_batch,
                generator,
                inputs,
            )
            pieces.append(logits.cpu().numpy().astype(np.float32))
            del logits

        shard_logits = np.concatenate(pieces, axis=0)
        np.savez_compressed(
            shard_path,
            sample_indices=np.arange(shard_start, shard_end, dtype=np.int64),
            logits=shard_logits,
        )
        save_json(
            shard_metadata_path(shard_path),
            {
                "config_signature": signature,
                "count": cfg.shard_size,
                "shard_index": shard_index,
                "sample_start": shard_start,
                "sample_end": shard_end,
            },
        )

        now = time.perf_counter()
        if now - last_log >= Config.LOG_INTERVAL_SECONDS or shard_index + 1 == shard_count:
            completed = shard_end
            rate = completed / max(now - started, 1e-9)
            eta = (cfg.model_count - completed) / max(rate, 1e-9)
            print(
                f"  [prior] {completed:,}/{cfg.model_count:,} | "
                f"{rate:,.1f} models/s | ETA={eta/60:.1f} min"
            )
            last_log = now

    print("聚合 prior shards……")
    all_indices: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    for shard_index in range(shard_count):
        shard_start = shard_index * cfg.shard_size
        shard_end = shard_start + cfg.shard_size
        shard_path = shard_dir / f"prior_{shard_start:09d}_{shard_end:09d}.npz"
        with np.load(shard_path, allow_pickle=False) as loaded:
            all_indices.append(loaded["sample_indices"])
            all_logits.append(loaded["logits"])

    sample_indices = np.concatenate(all_indices)
    logits = np.concatenate(all_logits)
    ids = np.zeros(len(logits), dtype=np.uint16)
    for bit in range(logits.shape[1]):
        ids |= ((logits[:, bit] >= 0).astype(np.uint16) << bit)
    logit_rms = np.sqrt(np.mean(np.square(logits.astype(np.float64)), axis=1)).astype(
        np.float32
    )
    arrays = {
        "sample_indices": sample_indices,
        "function_ids": ids,
        "logits": logits,
        "logit_rms": logit_rms,
    }
    np.savez_compressed(output_path, **arrays)
    save_json(
        output_meta,
        {
            "config_signature": signature,
            "samples": cfg.model_count,
            "observed_functions": int(np.unique(ids).size),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(
        f"先验采样完成：observed_functions={np.unique(ids).size}/256 | "
        f"耗时={(time.perf_counter()-started)/60:.1f} min"
    )
    return arrays


# =============================================================================
# 独立 Boolean 函数复杂度面板
# =============================================================================


def fwht(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float64).copy()
    width = transformed.shape[1]
    stride = 1
    while stride < width:
        view = transformed.reshape(transformed.shape[0], -1, stride * 2)
        left = view[:, :, :stride].copy()
        right = view[:, :, stride:].copy()
        view[:, :, :stride] = left + right
        view[:, :, stride:] = left - right
        stride *= 2
    return transformed / width


def anf_coefficients(bits: np.ndarray, input_bits: int) -> np.ndarray:
    coefficients = np.asarray(bits, dtype=np.uint8).copy()
    indices = np.arange(1 << input_bits, dtype=np.int64)
    for bit in range(input_bits):
        high = indices[(indices & (1 << bit)) != 0]
        coefficients[:, high] ^= coefficients[:, high ^ (1 << bit)]
    return coefficients


def all_boolean_cubes(input_bits: int) -> list[tuple[int, int]]:
    """返回 (coverage_mask, literal_count)，状态 -1 表示 don't care。"""

    cubes: list[tuple[int, int]] = []
    for specification in itertools.product((-1, 0, 1), repeat=input_bits):
        coverage = 0
        for row in range(1 << input_bits):
            valid = True
            for bit, requirement in enumerate(specification):
                if requirement >= 0 and ((row >> bit) & 1) != requirement:
                    valid = False
                    break
            if valid:
                coverage |= 1 << row
        cubes.append((coverage, sum(value >= 0 for value in specification)))
    return cubes


def minimum_dnf_cost(target_mask: int, input_bits: int) -> tuple[int, int]:
    if target_mask == 0:
        return 0, 0
    cubes = [
        (coverage, literals)
        for coverage, literals in all_boolean_cubes(input_bits)
        if coverage != 0 and (coverage & ~target_mask) == 0
    ]
    width = 1 << (1 << input_bits)
    infinity = (10_000, 10_000)
    best = [infinity for _ in range(width)]
    best[0] = (0, 0)
    for _ in range(1 << input_bits):
        changed = False
        previous = list(best)
        for covered, current in enumerate(previous):
            if current == infinity:
                continue
            for cube_mask, literals in cubes:
                updated = covered | cube_mask
                candidate = (current[0] + 1, current[1] + literals)
                if candidate < best[updated]:
                    best[updated] = candidate
                    changed = True
        if not changed:
            break
    return best[target_mask]


def build_function_complexity_panel(input_bits: int) -> list[dict[str, Any]]:
    function_total = 1 << (1 << input_bits)
    domain = 1 << input_bits
    ids = np.arange(function_total, dtype=np.uint16)
    bits = ((ids[:, None] >> np.arange(domain)) & 1).astype(np.uint8)

    signs = bits.astype(np.float64) * 2.0 - 1.0
    walsh = fwht(signs)
    energy = np.square(walsh)
    energy /= np.maximum(energy.sum(axis=1, keepdims=True), 1e-300)
    degrees = np.array([int(value).bit_count() for value in range(domain)])
    spectral_entropy = -np.sum(
        energy * np.log2(np.where(energy > 0, energy, 1.0)), axis=1
    )
    spectral_mean_degree = np.sum(energy * degrees[None, :], axis=1)
    low_order_mass = np.sum(energy[:, degrees <= 1], axis=1)

    influence = np.zeros(function_total, dtype=np.float64)
    indices = np.arange(domain)
    for bit in range(input_bits):
        influence += np.mean(bits != bits[:, indices ^ (1 << bit)], axis=1)

    anf = anf_coefficients(bits, input_bits)
    anf_terms = anf.sum(axis=1)
    anf_degree = np.zeros(function_total, dtype=np.int16)
    for index in range(function_total):
        active = degrees[anf[index].astype(bool)]
        anf_degree[index] = int(active.max()) if active.size else 0

    all_mask = (1 << domain) - 1
    rows: list[dict[str, Any]] = []
    for function_id in range(function_total):
        dnf_terms, dnf_literals = minimum_dnf_cost(function_id, input_bits)
        cnf_terms, cnf_literals = minimum_dnf_cost(all_mask ^ function_id, input_bits)
        truth = bits[function_id]
        rows.append(
            {
                "function_id": function_id,
                "truth_table": "".join(str(int(value)) for value in truth),
                "one_rate": float(truth.mean()),
                "total_influence": float(influence[function_id]),
                "spectral_entropy": float(spectral_entropy[function_id]),
                "spectral_mean_degree": float(spectral_mean_degree[function_id]),
                "walsh_mass_degree_le_1": float(low_order_mass[function_id]),
                "dominant_walsh_energy": float(energy[function_id].max()),
                "anf_terms": int(anf_terms[function_id]),
                "anf_degree": int(anf_degree[function_id]),
                "dnf_terms": int(dnf_terms),
                "dnf_literals": int(dnf_literals),
                "cnf_clauses": int(cnf_terms),
                "cnf_literals": int(cnf_literals),
                "min_normal_form_terms": int(min(dnf_terms, cnf_terms)),
                "min_normal_form_literals": int(min(dnf_literals, cnf_literals)),
                "lex_transition_count": int(np.count_nonzero(truth[1:] != truth[:-1])),
            }
        )
    return rows


# =============================================================================
# 退火与 loss 子水平集分析
# =============================================================================


COMPLEXITY_METRICS = (
    "one_rate",
    "total_influence",
    "spectral_entropy",
    "spectral_mean_degree",
    "walsh_mass_degree_le_1",
    "dominant_walsh_energy",
    "anf_terms",
    "anf_degree",
    "min_normal_form_terms",
    "min_normal_form_literals",
    "lex_transition_count",
)


def condition_observables(
    logits: np.ndarray,
    condition: ConditionSpec,
) -> dict[str, np.ndarray]:
    indices = np.asarray(condition.input_indices, dtype=np.int64)
    targets = np.asarray(condition.targets, dtype=np.float64)
    selected = logits[:, indices].astype(np.float64)
    signed = targets * 2.0 - 1.0
    margins = selected * signed[None, :]
    raw_loss = np.logaddexp(0.0, -margins).mean(axis=1)

    rms = np.sqrt(np.mean(np.square(logits.astype(np.float64)), axis=1))
    normalized = selected / np.maximum(rms[:, None], 1e-12)
    normalized_margins = normalized * signed[None, :]
    normalized_loss = np.logaddexp(0.0, -normalized_margins).mean(axis=1)
    return {
        "hard_exact": np.all(margins > 0, axis=1),
        "raw_loss": raw_loss,
        "normalized_loss": normalized_loss,
        "min_margin": margins.min(axis=1),
        "normalized_min_margin": normalized_margins.min(axis=1),
        "logit_rms": rms,
    }


def function_distribution(
    function_ids: np.ndarray,
    sample_weights: np.ndarray,
    total_functions: int,
) -> np.ndarray:
    counts = np.bincount(
        function_ids.astype(np.int64),
        weights=np.asarray(sample_weights, dtype=np.float64),
        minlength=total_functions,
    ).astype(np.float64)
    total = float(counts.sum())
    return counts / total if total > 0 else counts


def mean_pairwise_agreement(
    probability: np.ndarray,
    truth_bits: np.ndarray,
) -> float:
    one_probability = probability @ truth_bits.astype(np.float64)
    agreement = np.square(one_probability) + np.square(1.0 - one_probability)
    return float(agreement.mean())


def append_distribution_record(
    *,
    condition: ConditionSpec,
    family: str,
    level_name: str,
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    truth_bits: np.ndarray,
    complexity_arrays: dict[str, np.ndarray],
    summary_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    selected_count: int,
    source_count: int,
    ess: float,
    threshold: float | None = None,
    quantile_fraction: float | None = None,
    beta: float | None = None,
    mean_raw_loss: float | None = None,
    mean_normalized_loss: float | None = None,
    mean_logit_rms: float | None = None,
) -> None:
    top_id = int(np.argmax(probability))
    row: dict[str, Any] = {
        "condition": condition.name,
        "description": condition.description,
        "family": family,
        "level": level_name,
        "selected_count": selected_count,
        "source_count": source_count,
        "selected_fraction": selected_count / max(source_count, 1),
        "ess": ess,
        "reliable_selected": selected_count >= Config.MIN_RELIABLE_SELECTED,
        "reliable_ess": ess >= Config.MIN_RELIABLE_ESS,
        "threshold": threshold,
        "quantile_fraction": quantile_fraction,
        "beta": beta,
        "unique_functions": int(np.count_nonzero(probability)),
        "function_entropy_bits": entropy_bits(probability),
        "effective_function_support": effective_support(probability),
        "top_function_id": top_id,
        "top_function_mass": float(probability[top_id]),
        "mean_pairwise_agreement": mean_pairwise_agreement(probability, truth_bits),
        "tv_from_hard_prior": total_variation(probability, baseline_probability),
        "js_from_hard_prior_bits": js_divergence(probability, baseline_probability),
        "reference_attractor_mass": float(
            probability[list(condition.reference_attractors)].sum()
        ),
        "mean_raw_loss": mean_raw_loss,
        "mean_normalized_loss": mean_normalized_loss,
        "mean_logit_rms": mean_logit_rms,
    }
    for metric in COMPLEXITY_METRICS:
        mean, std = weighted_mean_std(complexity_arrays[metric], probability)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    summary_rows.append(row)

    for function_id, mass in enumerate(probability):
        base = float(baseline_probability[function_id])
        if mass <= 0 and base <= 0:
            continue
        distribution_rows.append(
            {
                "condition": condition.name,
                "family": family,
                "level": level_name,
                "quantile_fraction": quantile_fraction,
                "beta": beta,
                "function_id": function_id,
                "probability": float(mass),
                "hard_prior_probability": base,
                "amplification": float(mass / base) if base > 0 else math.inf,
                "log2_amplification": float(math.log2(mass / base))
                if mass > 0 and base > 0
                else math.nan,
            }
        )


def analyze_condition(
    *,
    condition: ConditionSpec,
    logits: np.ndarray,
    function_ids: np.ndarray,
    truth_bits: np.ndarray,
    complexity_arrays: dict[str, np.ndarray],
    summary_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> None:
    total_functions = truth_bits.shape[0]
    values = condition_observables(logits, condition)
    hard = values["hard_exact"]
    hard_indices = np.flatnonzero(hard)
    if hard_indices.size == 0:
        raise RuntimeError(f"条件 {condition.name} 在先验样本中没有 hard-exact 权重。")

    hard_weights = hard.astype(np.float64)
    hard_probability = function_distribution(function_ids, hard_weights, total_functions)
    raw_scale_low, raw_scale_high = np.quantile(
        values["logit_rms"][hard], Config.FIXED_SCALE_QUANTILES
    )
    fixed_scale = hard & (values["logit_rms"] >= raw_scale_low) & (
        values["logit_rms"] <= raw_scale_high
    )

    condition_rows.append(
        {
            "condition": condition.name,
            "description": condition.description,
            "total_prior_samples": len(logits),
            "hard_exact_count": int(hard.sum()),
            "hard_exact_fraction": float(hard.mean()),
            "observed_hard_functions": int(np.count_nonzero(hard_probability)),
            "hard_function_entropy_bits": entropy_bits(hard_probability),
            "hard_top_function_id": int(np.argmax(hard_probability)),
            "hard_top_function_mass": float(hard_probability.max()),
            "raw_loss_min": float(values["raw_loss"][hard].min()),
            "raw_loss_q01": float(np.quantile(values["raw_loss"][hard], 0.01)),
            "raw_loss_q10": float(np.quantile(values["raw_loss"][hard], 0.10)),
            "raw_loss_q50": float(np.quantile(values["raw_loss"][hard], 0.50)),
            "raw_loss_max": float(values["raw_loss"][hard].max()),
            "fixed_scale_low": float(raw_scale_low),
            "fixed_scale_high": float(raw_scale_high),
            "fixed_scale_count": int(fixed_scale.sum()),
        }
    )

    print(
        f"\n[{condition.name}] hard-exact={hard.sum():,}/{len(hard):,} | "
        f"functions={np.count_nonzero(hard_probability)} | "
        f"loss q01/q50={np.quantile(values['raw_loss'][hard], 0.01):.4g}/"
        f"{np.quantile(values['raw_loss'][hard], 0.50):.4g}"
    )

    quantile_families = [
        ("raw_loss_hard", hard, values["raw_loss"]),
        ("normalized_loss_hard", hard, values["normalized_loss"]),
        ("raw_loss_fixed_scale", fixed_scale, values["raw_loss"]),
    ]
    for family, source_mask, score in quantile_families:
        source_indices = np.flatnonzero(source_mask)
        source_values = score[source_indices]
        for fraction in Config.QUANTILE_FRACTIONS:
            if family == "raw_loss_fixed_scale" and fraction < 0.005:
                continue
            threshold = float(np.quantile(source_values, fraction))
            selected = source_mask & (score <= threshold)
            selected_count = int(selected.sum())
            weights = selected.astype(np.float64)
            probability = function_distribution(function_ids, weights, total_functions)
            append_distribution_record(
                condition=condition,
                family=family,
                level_name=f"q{fraction:g}",
                probability=probability,
                baseline_probability=hard_probability,
                truth_bits=truth_bits,
                complexity_arrays=complexity_arrays,
                summary_rows=summary_rows,
                distribution_rows=distribution_rows,
                selected_count=selected_count,
                source_count=len(source_indices),
                ess=float(selected_count),
                threshold=threshold,
                quantile_fraction=float(fraction),
                mean_raw_loss=float(values["raw_loss"][selected].mean()),
                mean_normalized_loss=float(values["normalized_loss"][selected].mean()),
                mean_logit_rms=float(values["logit_rms"][selected].mean()),
            )

    beta_families = [
        ("raw_beta_hard", values["raw_loss"]),
        ("normalized_beta_hard", values["normalized_loss"]),
    ]
    for family, score in beta_families:
        hard_score = score[hard]
        minimum = float(hard_score.min())
        for beta_value in Config.BETAS:
            beta = float(beta_value)
            log_weights = -beta * (score - minimum)
            log_weights[~hard] = -np.inf
            finite = np.isfinite(log_weights)
            maximum = float(np.max(log_weights[finite]))
            weights = np.zeros_like(score, dtype=np.float64)
            weights[finite] = np.exp(log_weights[finite] - maximum)
            weight_sum = float(weights.sum())
            ess = weight_sum**2 / max(float(np.square(weights).sum()), 1e-300)
            probability = function_distribution(function_ids, weights, total_functions)
            normalized_weights = weights / max(weight_sum, 1e-300)
            append_distribution_record(
                condition=condition,
                family=family,
                level_name=f"beta_{beta:g}",
                probability=probability,
                baseline_probability=hard_probability,
                truth_bits=truth_bits,
                complexity_arrays=complexity_arrays,
                summary_rows=summary_rows,
                distribution_rows=distribution_rows,
                selected_count=int(hard.sum()),
                source_count=int(hard.sum()),
                ess=ess,
                beta=beta,
                mean_raw_loss=float(np.sum(normalized_weights * values["raw_loss"])),
                mean_normalized_loss=float(
                    np.sum(normalized_weights * values["normalized_loss"])
                ),
                mean_logit_rms=float(np.sum(normalized_weights * values["logit_rms"])),
            )

    raw_low = [
        row
        for row in summary_rows
        if row["condition"] == condition.name
        and row["family"] == "raw_loss_hard"
        and row["quantile_fraction"] in {1.0, 0.01}
    ]
    raw_low.sort(key=lambda item: item["quantile_fraction"], reverse=True)
    if len(raw_low) == 2:
        print(
            f"  hard prior -> raw loss lowest 1%：H="
            f"{raw_low[0]['function_entropy_bits']:.3f} -> "
            f"{raw_low[1]['function_entropy_bits']:.3f} | attractor mass="
            f"{raw_low[0]['reference_attractor_mass']:.3f} -> "
            f"{raw_low[1]['reference_attractor_mass']:.3f}"
        )


def create_plots(
    cfg: EffectiveConfig,
    summary_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - 只在缺少 matplotlib 时触发
        print(f"跳过作图：{error}")
        return []

    plot_dir = cfg.result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    conditions = experiment_conditions()
    paths: list[str] = []

    figure, axes = plt.subplots(len(conditions), 3, figsize=(15, 4 * len(conditions)))
    for row_index, condition in enumerate(conditions):
        for family, label in [
            ("raw_loss_hard", "raw BCE"),
            ("normalized_loss_hard", "RMS-normalized BCE"),
            ("raw_loss_fixed_scale", "fixed-scale raw BCE"),
        ]:
            rows = [
                row
                for row in summary_rows
                if row["condition"] == condition.name and row["family"] == family
            ]
            rows.sort(key=lambda item: item["quantile_fraction"], reverse=True)
            if not rows:
                continue
            x = [row["quantile_fraction"] for row in rows]
            axes[row_index, 0].plot(
                x, [row["function_entropy_bits"] for row in rows], marker="o", label=label
            )
            axes[row_index, 1].plot(
                x, [row["reference_attractor_mass"] for row in rows], marker="o", label=label
            )
            axes[row_index, 2].plot(
                x, [row["mean_pairwise_agreement"] for row in rows], marker="o", label=label
            )
            unreliable = [
                index
                for index, row in enumerate(rows)
                if not bool(row["reliable_selected"])
            ]
            if unreliable:
                for column, metric in enumerate(
                    (
                        "function_entropy_bits",
                        "reference_attractor_mass",
                        "mean_pairwise_agreement",
                    )
                ):
                    axes[row_index, column].scatter(
                        [x[index] for index in unreliable],
                        [rows[index][metric] for index in unreliable],
                        marker="x",
                        color="red",
                        s=55,
                        zorder=5,
                    )
        for axis in axes[row_index]:
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.grid(alpha=0.25)
            axis.set_xlabel("retained lowest-loss fraction")
        axes[row_index, 0].set_ylabel("function entropy (bits)")
        axes[row_index, 1].set_ylabel("reference attractor mass")
        axes[row_index, 2].set_ylabel("pairwise agreement")
        axes[row_index, 0].set_title(condition.name)
        axes[row_index, 0].legend(fontsize=8)
    figure.tight_layout()
    path = plot_dir / "microcanonical_loss_slices.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(len(conditions), 3, figsize=(15, 4 * len(conditions)))
    for row_index, condition in enumerate(conditions):
        for family, label in [
            ("raw_beta_hard", "raw BCE"),
            ("normalized_beta_hard", "RMS-normalized BCE"),
        ]:
            rows = [
                row
                for row in summary_rows
                if row["condition"] == condition.name and row["family"] == family
            ]
            rows.sort(key=lambda item: item["beta"])
            x = [max(row["beta"], 1e-2) for row in rows]
            axes[row_index, 0].plot(
                x, [row["function_entropy_bits"] for row in rows], marker="o", label=label
            )
            axes[row_index, 1].plot(
                x, [row["reference_attractor_mass"] for row in rows], marker="o", label=label
            )
            axes[row_index, 2].plot(
                x, [row["ess"] for row in rows], marker="o", label=label
            )
            unreliable = [
                index for index, row in enumerate(rows) if not bool(row["reliable_ess"])
            ]
            if unreliable:
                for column, metric in enumerate(
                    ("function_entropy_bits", "reference_attractor_mass", "ess")
                ):
                    axes[row_index, column].scatter(
                        [x[index] for index in unreliable],
                        [rows[index][metric] for index in unreliable],
                        marker="x",
                        color="red",
                        s=55,
                        zorder=5,
                    )
        for axis in axes[row_index]:
            axis.set_xscale("log")
            axis.grid(alpha=0.25)
            axis.set_xlabel("inverse temperature beta")
        axes[row_index, 0].set_ylabel("function entropy (bits)")
        axes[row_index, 1].set_ylabel("reference attractor mass")
        axes[row_index, 2].set_ylabel("importance ESS")
        axes[row_index, 2].set_yscale("log")
        axes[row_index, 0].set_title(condition.name)
        axes[row_index, 0].legend(fontsize=8)
    figure.tight_layout()
    path = plot_dir / "canonical_beta_annealing.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    # 每个条件画 raw-beta 下概率质量最高的函数轨迹。
    for condition in conditions:
        rows = [
            row
            for row in distribution_rows
            if row["condition"] == condition.name and row["family"] == "raw_beta_hard"
        ]
        if not rows:
            continue
        by_id: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_id.setdefault(int(row["function_id"]), []).append(row)
        ranked = sorted(
            by_id,
            key=lambda function_id: max(item["probability"] for item in by_id[function_id]),
            reverse=True,
        )[:8]
        figure, axis = plt.subplots(figsize=(9, 5))
        for function_id in ranked:
            trajectory = sorted(by_id[function_id], key=lambda item: item["beta"])
            axis.plot(
                [max(float(item["beta"]), 1e-2) for item in trajectory],
                [float(item["probability"]) for item in trajectory],
                marker="o",
                label=f"ID {function_id}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("inverse temperature beta")
        axis.set_ylabel("function probability")
        axis.set_title(f"{condition.name}: raw-loss annealing paths")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        path = plot_dir / f"top_function_paths_{condition.name}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))

    return paths


def create_release_zip(cfg: EffectiveConfig) -> Path:
    archive = cfg.result_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(cfg.result_dir.rglob("*")):
            if not path.is_file() or "shards" in path.parts:
                continue
            handle.write(path, arcname=path.relative_to(cfg.result_dir))
    return archive


def analyze(
    cfg: EffectiveConfig,
    prior: dict[str, np.ndarray],
) -> dict[str, Any]:
    analysis_dir = cfg.result_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    complexity_rows = build_function_complexity_panel(cfg.input_bits)
    write_csv(analysis_dir / "function_complexity_panel.csv", complexity_rows)
    complexity_arrays = {
        metric: np.asarray([row[metric] for row in complexity_rows], dtype=np.float64)
        for metric in COMPLEXITY_METRICS
    }
    truth_bits = np.asarray(
        [
            [int(value) for value in row["truth_table"]]
            for row in complexity_rows
        ],
        dtype=np.uint8,
    )

    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for condition in experiment_conditions():
        analyze_condition(
            condition=condition,
            logits=prior["logits"],
            function_ids=prior["function_ids"],
            truth_bits=truth_bits,
            complexity_arrays=complexity_arrays,
            summary_rows=summary_rows,
            distribution_rows=distribution_rows,
            condition_rows=condition_rows,
        )

    write_csv(analysis_dir / "condition_summary.csv", condition_rows)
    write_csv(analysis_dir / "annealing_summary.csv", summary_rows)
    write_csv(analysis_dir / "function_distributions.csv", distribution_rows)
    plot_paths = create_plots(cfg, summary_rows, distribution_rows)

    headline: list[dict[str, Any]] = []
    for condition in experiment_conditions():
        baseline = next(
            row
            for row in summary_rows
            if row["condition"] == condition.name
            and row["family"] == "raw_loss_hard"
            and row["quantile_fraction"] == 1.0
        )
        low_one = next(
            row
            for row in summary_rows
            if row["condition"] == condition.name
            and row["family"] == "raw_loss_hard"
            and row["quantile_fraction"] == 0.01
        )
        headline.append(
            {
                "condition": condition.name,
                "hard_exact_count": baseline["selected_count"],
                "hard_entropy": baseline["function_entropy_bits"],
                "low_1pct_entropy": low_one["function_entropy_bits"],
                "hard_attractor_mass": baseline["reference_attractor_mass"],
                "low_1pct_attractor_mass": low_one["reference_attractor_mass"],
                "low_1pct_reliable": low_one["reliable_selected"],
            }
        )

    summary = {
        "protocol_version": "loss_conditioned_prior_annealing_v1",
        "question": (
            "在相同 hard constraints 已满足时，进入更低连续 loss 的初始化权重子水平集，"
            "是否会系统性放大少数函数与低有效复杂度区域？"
        ),
        "interpretation_boundary": (
            "本实验测量初始化先验下的 loss-conditioned 几何；它不自动证明 SGD 是 Gibbs "
            "采样，也不把 Boolean 代理等同于绝对 Kolmogorov complexity。"
        ),
        "headline": headline,
        "files": {
            "condition_summary": str(analysis_dir / "condition_summary.csv"),
            "annealing_summary": str(analysis_dir / "annealing_summary.csv"),
            "function_distributions": str(analysis_dir / "function_distributions.csv"),
            "function_complexity_panel": str(
                analysis_dir / "function_complexity_panel.csv"
            ),
            "plots": plot_paths,
        },
    }
    save_json(cfg.result_dir / "summary.json", summary)
    return summary


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    cfg = resolve_config()
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    payload = config_payload(cfg)
    signature = stable_hash(payload)
    save_json(cfg.result_dir / "config.json", payload)

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)

    print("=== Loss-conditioned Function Prior Annealing ===")
    print(f"设备：{cfg.device}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"结果目录：{cfg.result_dir}")
    print(
        f"配置：profile={cfg.profile} | models={cfg.model_count:,} | "
        f"network={cfg.input_bits}->{cfg.hidden_size}x{cfg.hidden_layers}->1 "
        f"{cfg.architecture} | TF32={cfg.allow_tf32}"
    )
    print(f"配置签名：{signature}")

    prior = sample_prior(cfg, signature)
    summary = analyze(cfg, prior)
    archive: Path | None = None
    if Config.CREATE_ZIP:
        archive = create_release_zip(cfg)

    print("\n=== 先验退火实验完成 ===")
    for row in summary["headline"]:
        print(
            f"{row['condition']} | hard_n={row['hard_exact_count']:,} | "
            f"H: {row['hard_entropy']:.3f}->{row['low_1pct_entropy']:.3f} | "
            f"attractor: {row['hard_attractor_mass']:.3f}->"
            f"{row['low_1pct_attractor_mass']:.3f} | "
            f"reliable={row['low_1pct_reliable']}"
        )
    print(f"汇总：{cfg.result_dir / 'summary.json'}")
    if archive is not None:
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
