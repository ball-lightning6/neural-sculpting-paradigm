"""
布尔函数后验与训练动力学实验。

核心问题：给定同一组训练约束 S，训练后的函数分布 Q_t(f|S) 到底只是
初始化函数先验 P_0(f) 的静态条件化，还是会在持续训练中进一步重加权、
收缩或迁移？

当前版本刻意先把主实验限制在 3-bit -> 1-bit：整个 hard function space 只有
256 个函数，可以完整记录函数分布。网络仍使用过参数化的 1024 x 3 MLP，
而不是为了穷举方便改成小网络。

使用方式（AutoDL/Jupyter）：
1. 修改 Config.PROFILE："full" / "quick" / "smoke"。
2. 整段复制到一个 notebook 单元格运行，或直接运行本文件。
3. 中断后重新运行会复用相同配置的 prior 与训练 chunk。

输出包含：
- 初始化 hard-function 先验；
- 自动选择的训练约束与全部候选评分；
- ordinary 与 prior-consistent 两类 cohort 的事件对齐训练轨迹；
- hard-conditioned prior、soft-likelihood baseline 与经验后验的比较；
- 初始化先验概率与训练后重加权的直接比较；
- 后验距离、逐 seed 配对转移、完整函数分布、logits、作图与 zip 压缩包。

重要边界：本脚本完全不测量或代理 Kolmogorov complexity，也不据此给函数
排序。复杂度测量可以在未来作为独立实验，对这里保存的原始函数分布离线分析；
对静态筛选图景的核心检验本来就不依赖复杂度定义。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
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
    # full：最终实验；quick：先看方向；smoke：只检查代码链路。
    PROFILE = "full"

    RESULT_DIR = script_directory() / "results_boolean_function_posterior_dynamics"
    RESUME_EXISTING = True
    CREATE_ZIP = True

    INPUT_BITS = 3
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    FIT_CHECK_INTERVAL = 1
    MAX_PREFIT_STEPS = 1_000

    PRIOR_SEED_BASE = 1_000_000
    ORDINARY_SEED_BASE = 2_000_000
    GLOBAL_SEED = 20260817

    # full 档参数。
    FULL_PRIOR_MODELS = 65_536
    FULL_PRIOR_CHUNK = 32
    FULL_ORDINARY_MODELS = 128
    FULL_CONSISTENT_MODELS = 512
    FULL_TRAIN_CHUNK = 32
    FULL_POST_FIT_AGES = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000)

    # quick 档通常在 RTX 5090 上约 20--30 分钟。
    QUICK_PRIOR_MODELS = 32_768
    QUICK_PRIOR_CHUNK = 32
    QUICK_ORDINARY_MODELS = 128
    QUICK_CONSISTENT_MODELS = 256
    QUICK_TRAIN_CHUNK = 32
    QUICK_POST_FIT_AGES = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000)

    # 自动选择一个单样本、一个二样本和一个四样本约束。
    STATE_SAMPLE_SIZES = (1, 2, 4)
    FIXED_SINGLE_INPUT = 0
    FIXED_SINGLE_TARGET = 0

    SOFT_LIKELIHOOD_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    NULL_MONTE_CARLO_DRAWS = 200
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


def resolve_config() -> EffectiveConfig:
    profile = str(Config.PROFILE).lower().strip()
    if profile == "full":
        values = (
            Config.FULL_PRIOR_MODELS,
            Config.FULL_PRIOR_CHUNK,
            Config.FULL_ORDINARY_MODELS,
            Config.FULL_CONSISTENT_MODELS,
            Config.FULL_TRAIN_CHUNK,
            Config.FULL_POST_FIT_AGES,
        )
        hidden_size = Config.HIDDEN_SIZE
        hidden_layers = Config.HIDDEN_LAYERS
        max_prefit = Config.MAX_PREFIT_STEPS
        smoke = False
    elif profile == "quick":
        values = (
            Config.QUICK_PRIOR_MODELS,
            Config.QUICK_PRIOR_CHUNK,
            Config.QUICK_ORDINARY_MODELS,
            Config.QUICK_CONSISTENT_MODELS,
            Config.QUICK_TRAIN_CHUNK,
            Config.QUICK_POST_FIT_AGES,
        )
        hidden_size = Config.HIDDEN_SIZE
        hidden_layers = Config.HIDDEN_LAYERS
        max_prefit = Config.MAX_PREFIT_STEPS
        smoke = False
    elif profile == "smoke":
        values = (256, 8, 8, 8, 4, (0, 1, 2, 5))
        hidden_size = 64
        hidden_layers = 2
        max_prefit = 100
        smoke = True
    else:
        raise ValueError(f"未知 PROFILE={Config.PROFILE!r}，只能是 full/quick/smoke。")

    prior_models, prior_chunk, ordinary, consistent, train_chunk, ages = values
    if Config.INPUT_BITS != 3:
        raise ValueError(
            "当前脚本先做可完全穷举的 3-bit 主实验。4-bit 确认实验应在主结果成立后单独扩展。"
        )
    return EffectiveConfig(
        profile=profile,
        result_dir=Path(Config.RESULT_DIR),
        input_bits=Config.INPUT_BITS,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        fit_check_interval=Config.FIT_CHECK_INTERVAL,
        max_prefit_steps=max_prefit,
        prior_seed_base=Config.PRIOR_SEED_BASE,
        ordinary_seed_base=Config.ORDINARY_SEED_BASE,
        prior_models=prior_models,
        prior_chunk=prior_chunk,
        ordinary_models=ordinary,
        consistent_models=consistent,
        train_chunk=train_chunk,
        post_fit_ages=tuple(int(v) for v in ages),
        device=Config.DEVICE,
        allow_tf32=Config.ALLOW_TF32,
        smoke_test=smoke,
    )


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
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
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


def chunked(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def function_count(input_bits: int) -> int:
    return 1 << (1 << input_bits)


def truth_table_inputs(input_bits: int) -> torch.Tensor:
    rows = 1 << input_bits
    values = torch.arange(rows, dtype=torch.int64)
    bits = ((values[:, None] >> torch.arange(input_bits)) & 1).to(torch.float32)
    return bits


def function_ids_from_logits(logits: torch.Tensor) -> torch.Tensor:
    rows = logits.shape[-1]
    powers = (2 ** torch.arange(rows, device=logits.device, dtype=torch.int64))[None, :]
    return ((logits >= 0).to(torch.int64) * powers).sum(dim=1)


def state_compatible_mask(state: StateSpec, total_functions: int) -> np.ndarray:
    ids = np.arange(total_functions, dtype=np.uint32)
    mask = np.ones(total_functions, dtype=bool)
    for index, target in zip(state.input_indices, state.targets):
        mask &= ((ids >> index) & 1) == target
    return mask


def entropy_bits(probability: np.ndarray) -> float:
    p = np.asarray(probability, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-300)
    q = q / max(float(q.sum()), 1e-300)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        valid = a > 0
        return float((a[valid] * np.log2(a[valid] / b[valid])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# =============================================================================
# 独立模型的 GPU 批量实现
# =============================================================================


class EnsembleLinear(nn.Module):
    def __init__(self, seeds: tuple[int, ...], in_features: int, out_features: int, salt: int):
        super().__init__()
        count = len(seeds)
        weight = torch.empty(count, out_features, in_features, dtype=torch.float32)
        bias = torch.empty(count, out_features, dtype=torch.float32)
        bound = 1.0 / math.sqrt(in_features)
        for index, seed in enumerate(seeds):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) * 1_000_003 + salt)
            weight[index].uniform_(-bound, bound, generator=generator)
            bias[index].uniform_(-bound, bound, generator=generator)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ebi,eoi->ebo", inputs, self.weight) + self.bias[:, None, :]


class EnsembleLayerNorm(nn.Module):
    def __init__(self, ensemble_size: int, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, width))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = (inputs - mean).square().mean(dim=-1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[:, None, :] + self.bias[:, None, :]


class BatchedSeedMLP(nn.Module):
    def __init__(self, seeds: tuple[int, ...], cfg: EffectiveConfig):
        super().__init__()
        count = len(seeds)
        layers: list[nn.Module] = []
        width_in = cfg.input_bits
        for layer_index in range(cfg.hidden_layers):
            layers.append(
                EnsembleLinear(seeds, width_in, cfg.hidden_size, salt=100 + layer_index * 17)
            )
            layers.append(nn.GELU())
            layers.append(EnsembleLayerNorm(count, cfg.hidden_size))
            width_in = cfg.hidden_size
        self.hidden = nn.ModuleList(layers)
        self.output = EnsembleLinear(seeds, width_in, 1, salt=10_007)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.hidden:
            hidden = layer(hidden)
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
    payload["protocol_version"] = "boolean_posterior_dynamics_v2"
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
            print("复用初始化函数先验。")
            return load_npz(aggregate_path)

    print(f"采样初始化函数先验：models={cfg.prior_models:,}，chunk={cfg.prior_chunk:,}")
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
        seed_tuple = tuple(int(v) for v in seed_chunk)
        model = BatchedSeedMLP(seed_tuple, cfg).to(device).eval()
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
        if chunk_index == 1 or chunk_index % max(1, len(chunks) // 10) == 0 or processed == len(seeds):
            rate = processed / max(time.perf_counter() - started, 1e-9)
            eta = (len(seeds) - processed) / max(rate, 1e-9)
            print(
                f"  [prior] {processed:,}/{len(seeds):,} | {rate:,.1f} models/s | ETA={eta/60:.1f} min"
            )

    function_ids = np.concatenate(all_ids)
    probe_logits = np.concatenate(all_logits)
    counts = np.bincount(
        function_ids.astype(np.int64), minlength=function_count(cfg.input_bits)
    ).astype(np.int64)
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
    print(
        f"初始化先验完成：observed_functions={np.count_nonzero(counts)}/256 | "
        f"耗时={time.perf_counter() - started:.1f}s"
    )
    return arrays


# =============================================================================
# 自动选择训练约束
# =============================================================================


def state_score(
    state: StateSpec,
    prior_counts: np.ndarray,
) -> dict[str, Any]:
    mask = state_compatible_mask(state, len(prior_counts))
    conditional_counts = prior_counts[mask].astype(np.float64)
    support = int(conditional_counts.sum())
    if support == 0:
        return {"support": 0, "score": -float("inf")}
    probabilities = conditional_counts / conditional_counts.sum()
    effective_support = 1.0 / float(np.square(probabilities).sum())
    largest_mass = float(probabilities.max())
    # 选择只依赖 conditioned prior 的覆盖度，不使用任何人工复杂度定义。
    score = entropy_bits(probabilities) + 0.25 * math.log2(max(effective_support, 1.0)) - largest_mass
    return {
        "support": support,
        "support_probability": support / float(prior_counts.sum()),
        "candidate_functions": int(mask.sum()),
        "observed_candidate_functions": int(np.count_nonzero(conditional_counts)),
        "conditional_entropy_bits": entropy_bits(probabilities),
        "effective_support": effective_support,
        "largest_function_mass": largest_mass,
        "score": score,
    }


def select_training_states(
    cfg: EffectiveConfig,
    prior: dict[str, np.ndarray],
) -> tuple[list[StateSpec], list[dict[str, Any]]]:
    prior_counts = prior["function_counts"]
    candidates: list[tuple[StateSpec, dict[str, Any]]] = []
    rows = 1 << cfg.input_bits

    fixed = StateSpec(
        name=f"single_x{Config.FIXED_SINGLE_INPUT}_y{Config.FIXED_SINGLE_TARGET}",
        input_indices=(int(Config.FIXED_SINGLE_INPUT),),
        targets=(int(Config.FIXED_SINGLE_TARGET),),
        selection_kind="fixed_single",
        description="固定的单样本约束，用于复核单样本后验坍缩。",
    )
    candidates.append((fixed, state_score(fixed, prior_counts)))

    all_candidate_rows: list[dict[str, Any]] = []
    selected = [fixed]
    for sample_size in (2, 4):
        pool: list[tuple[StateSpec, dict[str, Any]]] = []
        for indices in itertools.combinations(range(rows), sample_size):
            for targets in itertools.product((0, 1), repeat=sample_size):
                # 多样本条件要求正负标签都出现，避免再次退化为常量任务。
                if len(set(targets)) < 2:
                    continue
                if sample_size == 4 and sum(targets) != 2:
                    continue
                state = StateSpec(
                    name=(
                        f"auto_k{sample_size}_x{'-'.join(map(str, indices))}"
                        f"_y{''.join(map(str, targets))}"
                    ),
                    input_indices=tuple(indices),
                    targets=tuple(targets),
                    selection_kind=f"auto_high_entropy_k{sample_size}",
                    description=f"按 conditioned-prior 高熵与有效支持自动选择的 {sample_size} 样本约束。",
                )
                metrics = state_score(state, prior_counts)
                metrics["eligible"] = metrics["support"] >= cfg.consistent_models
                pool.append((state, metrics))
        eligible = [item for item in pool if item[1]["eligible"]]
        if not eligible:
            raise RuntimeError(f"k={sample_size} 没有足够 prior-consistent seeds 的候选条件。")
        winner = max(eligible, key=lambda item: (item[1]["score"], item[0].name))
        selected.append(winner[0])
        candidates.extend(pool)

    for state, metrics in candidates:
        all_candidate_rows.append(
            {
                "selected": state in selected,
                "name": state.name,
                "sample_size": len(state.input_indices),
                "input_indices": " ".join(map(str, state.input_indices)),
                "targets": "".join(map(str, state.targets)),
                **metrics,
            }
        )
    all_candidate_rows.sort(key=lambda row: (row["sample_size"], -float(row.get("score", -1e9))))
    write_csv(cfg.result_dir / "state_selection_candidates.csv", all_candidate_rows)
    save_json(
        cfg.result_dir / "selected_training_states.json",
        [{**asdict(state), **state_score(state, prior_counts)} for state in selected],
    )
    print("自动选择训练约束：")
    for state in selected:
        metrics = state_score(state, prior_counts)
        print(
            f"  {state.name} | S={list(zip(state.input_indices, state.targets))} | "
            f"prior_support={metrics['support']:,} | entropy={metrics['conditional_entropy_bits']:.3f} bit | "
            f"max_mass={metrics['largest_function_mass']:.4f}"
        )
    return selected, all_candidate_rows


# =============================================================================
# 事件对齐训练
# =============================================================================


def train_seed_chunk(
    cfg: EffectiveConfig,
    state: StateSpec,
    cohort: str,
    seeds_array: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seeds = tuple(int(v) for v in seeds_array)
    count = len(seeds)
    device = torch.device(cfg.device)
    probe_cpu = truth_table_inputs(cfg.input_bits)
    train_indices = list(state.input_indices)
    target_values = torch.tensor(state.targets, dtype=torch.float32)

    model = BatchedSeedMLP(seeds, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    probe = probe_cpu.to(device)[None, :, :].expand(count, -1, -1)
    train_x = probe_cpu[train_indices].to(device)[None, :, :].expand(count, -1, -1)
    train_y = target_values.to(device)[None, :].expand(count, -1)

    ages = tuple(sorted(set(int(v) for v in cfg.post_fit_ages)))
    output_ages = np.asarray((-1, *ages), dtype=np.int64)
    age_positions = {age: index + 1 for index, age in enumerate(ages)}
    snapshots = len(output_ages)
    recorded = np.zeros((snapshots, count), dtype=bool)
    absolute_steps = np.full((snapshots, count), -1, dtype=np.int64)
    logits_out = np.full((snapshots, count, 1 << cfg.input_bits), np.nan, dtype=np.float32)
    ids_out = np.zeros((snapshots, count), dtype=np.uint16)
    loss_out = np.full((snapshots, count), np.nan, dtype=np.float32)
    margin_out = np.full((snapshots, count), np.nan, dtype=np.float32)
    exact_out = np.zeros((snapshots, count), dtype=bool)
    first_fit = np.full(count, -1, dtype=np.int64)
    started = time.perf_counter()

    def assign(step: int, snapshot: dict[str, np.ndarray], age_masks: dict[int, np.ndarray]) -> None:
        for age, mask in age_masks.items():
            position = 0 if age == -1 else age_positions[age]
            logits_out[position, mask] = snapshot["probe_logits"][mask]
            ids_out[position, mask] = snapshot["function_ids"][mask]
            loss_out[position, mask] = snapshot["train_loss"][mask]
            margin_out[position, mask] = snapshot["train_min_margin"][mask]
            exact_out[position, mask] = snapshot["train_exact"][mask]
            absolute_steps[position, mask] = step
            recorded[position, mask] = True

    initial = evaluate_model(model, probe, train_x, train_y)
    all_mask = np.ones(count, dtype=bool)
    assign(0, initial, {-1: all_mask})
    initially_fitted = initial["train_exact"]
    first_fit[initially_fitted] = 0
    if 0 in age_positions and np.any(initially_fitted):
        assign(0, initial, {0: initially_fitted})

    if cohort == "prior_consistent" and not np.all(initially_fitted):
        raise RuntimeError(
            f"prior-consistent seed 重建后有 {(~initially_fitted).sum()} 个不再满足约束，"
            "说明初始化复现链路不一致。"
        )

    maximum_age = max(ages)
    final_step = cfg.max_prefit_steps + maximum_age
    latest_loss = float(initial["train_loss"].mean())
    for step in range(1, final_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_x).squeeze(-1)
        loss_by_model = F.binary_cross_entropy_with_logits(
            train_logits, train_y, reduction="none"
        ).mean(dim=1)
        # 每个参数切片是一张独立网络；求和才与逐个模型训练的梯度严格等价。
        loss = loss_by_model.sum()
        loss.backward()
        optimizer.step()
        latest_loss = float(loss_by_model.mean().detach().item())

        checked: dict[str, np.ndarray] | None = None
        if np.any(first_fit < 0) and (
            step == 1 or step % cfg.fit_check_interval == 0
        ):
            checked = evaluate_model(model, probe, train_x, train_y)
            newly = (first_fit < 0) & checked["train_exact"]
            first_fit[newly] = step

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
            # 未拟合者记为删失；已拟合者仍继续完成其年龄快照。
            if not np.any((first_fit >= 0) & (~recorded[-1])):
                break
        if step % (50 if cfg.smoke_test else Config.LOG_INTERVAL) == 0:
            print(
                f"      step={step:,}/{final_step:,} | fitted={np.count_nonzero(first_fit>=0)}/{count} | "
                f"mean_loss={latest_loss:.3e} | elapsed={time.perf_counter()-started:.1f}s"
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
    mask = state_compatible_mask(state, function_count(cfg.input_bits))
    compatible = mask[prior["function_ids"].astype(np.int64)]
    seeds = prior["seeds"][compatible]
    if len(seeds) < cfg.consistent_models:
        raise RuntimeError(
            f"{state.name} 只有 {len(seeds)} 个 prior-consistent seeds，"
            f"少于目标 {cfg.consistent_models}。"
        )
    return seeds[: cfg.consistent_models].astype(np.int64)


def run_state_cohort(
    cfg: EffectiveConfig,
    signature: str,
    prior: dict[str, np.ndarray],
    state: StateSpec,
    cohort: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state_dir = cfg.result_dir / "training" / cohort / state.name
    chunks_dir = state_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = state_dir / "samples.npz"
    metadata_path = state_dir / "metadata.json"
    if Config.RESUME_EXISTING and aggregate_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            print(f"  [skip] {cohort}/{state.name}")
            return metadata, load_npz(aggregate_path)

    seeds = cohort_seeds(cfg, prior, state, cohort)
    chunks = list(chunked(seeds, cfg.train_chunk))
    results: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    state_started = time.perf_counter()
    for chunk_index, seed_chunk in enumerate(chunks, start=1):
        tag = f"seed_{int(seed_chunk[0])}_{int(seed_chunk[-1])}"
        npz_path = chunks_dir / f"{tag}.npz"
        json_path = chunks_dir / f"{tag}.json"
        print(
            f"    [{cohort}] {state.name} chunk={chunk_index}/{len(chunks)} | "
            f"seeds={int(seed_chunk[0])}..{int(seed_chunk[-1])}"
        )
        if Config.RESUME_EXISTING and npz_path.exists() and json_path.exists():
            chunk_meta = json.loads(json_path.read_text(encoding="utf-8"))
            if chunk_meta.get("config_signature") == signature:
                print("      [skip chunk]")
                results.append((chunk_meta, load_npz(npz_path)))
                continue
        chunk_meta, chunk_arrays = train_seed_chunk(cfg, state, cohort, seed_chunk)
        chunk_meta["config_signature"] = signature
        np.savez_compressed(npz_path, **chunk_arrays)
        save_json(json_path, chunk_meta)
        results.append((chunk_meta, chunk_arrays))
        done = chunk_index * len(seed_chunk)
        elapsed = time.perf_counter() - state_started
        rate = chunk_index / max(elapsed, 1e-9)
        eta = (len(chunks) - chunk_index) / max(rate, 1e-9)
        print(f"      chunk 完成 | state ETA={eta/60:.1f} min")

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
        "train_loss": np.concatenate(
            [arrays["train_loss"] for _, arrays in results], axis=1
        ),
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
    metadata = {
        "config_signature": signature,
        "state": asdict(state),
        "cohort": cohort,
        "model_count": len(seeds),
        "chunk_count": len(chunks),
        "completed_models": int(aggregate["recorded"][1:].all(axis=0).sum()),
        "censored_models": int(np.count_nonzero(aggregate["first_fit_steps"] < 0)),
        "elapsed_seconds": time.perf_counter() - state_started,
    }
    save_json(metadata_path, metadata)
    return metadata, aggregate


# =============================================================================
# 后验分析
# =============================================================================


def normalized_counts(ids: np.ndarray, total_functions: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(ids.astype(np.int64), minlength=total_functions).astype(np.int64)
    probability = counts.astype(np.float64)
    probability /= max(float(probability.sum()), 1.0)
    return counts, probability


def mean_row_cosine(first: np.ndarray, second: np.ndarray) -> float:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid = denominator > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def prior_state_losses(prior_logits: np.ndarray, state: StateSpec) -> np.ndarray:
    logits = prior_logits[:, list(state.input_indices)].astype(np.float64)
    targets = np.asarray(state.targets, dtype=np.float64)[None, :]
    return (np.logaddexp(0.0, logits) - targets * logits).mean(axis=1)


def soft_baselines(
    prior: dict[str, np.ndarray],
    state: StateSpec,
    total_functions: int,
) -> dict[float, np.ndarray]:
    losses = prior_state_losses(prior["probe_logits"], state)
    ids = prior["function_ids"].astype(np.int64)
    result: dict[float, np.ndarray] = {}
    for value in Config.SOFT_LIKELIHOOD_LAMBDAS:
        lam = float(value)
        log_weights = -lam * losses
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        distribution = np.bincount(ids, weights=weights, minlength=total_functions)
        distribution = distribution.astype(np.float64)
        distribution /= distribution.sum()
        result[lam] = distribution
    return result


def analyze_experiment(
    cfg: EffectiveConfig,
    prior: dict[str, np.ndarray],
    states: list[StateSpec],
    training: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict[str, Any]:
    print("分析 hard prior、soft baseline 与训练后验动力学……")
    total_functions = function_count(cfg.input_bits)
    prior_counts = prior["function_counts"].astype(np.int64)
    prior_probability = prior_counts / prior_counts.sum()

    metrics_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_count_rows: list[dict[str, Any]] = []
    random = np.random.default_rng(Config.GLOBAL_SEED)

    for state in states:
        candidate_mask = state_compatible_mask(state, total_functions)
        hard_counts = prior_counts * candidate_mask
        hard_probability = hard_counts.astype(np.float64)
        hard_probability /= hard_probability.sum()
        soft = soft_baselines(prior, state, total_functions)

        for cohort in ("ordinary", "prior_consistent"):
            arrays = training[(cohort, state.name)]
            ages = arrays["post_fit_ages"]
            first_fit_positions = np.flatnonzero(ages == 0)
            if len(first_fit_positions) != 1:
                raise RuntimeError("post_fit_ages 必须且只能包含一个 age=0 快照。")
            first_fit_position = int(first_fit_positions[0])
            previous_post_fit_position: int | None = None
            for position, age_value in enumerate(ages):
                age = int(age_value)
                valid = arrays["recorded"][position].copy()
                ids = arrays["function_ids"][position, valid]
                if ids.size == 0:
                    continue
                q_counts, q_probability = normalized_counts(ids, total_functions)
                if age == -1 and cohort == "ordinary":
                    baseline_counts = prior_counts
                    baseline_probability = prior_probability
                    baseline_name = "raw_prior"
                else:
                    baseline_counts = hard_counts
                    baseline_probability = hard_probability
                    baseline_name = "hard_conditioned_prior"
                best_lambda, best_soft_js = min(
                    (
                        (lam, js_divergence(q_probability, probability))
                        for lam, probability in soft.items()
                    ),
                    key=lambda item: item[1],
                )

                tv = total_variation(q_probability, baseline_probability)
                null_tvs = []
                for _ in range(Config.NULL_MONTE_CARLO_DRAWS if not cfg.smoke_test else 10):
                    draw = random.multinomial(len(ids), baseline_probability)
                    null_tvs.append(total_variation(draw / max(draw.sum(), 1), baseline_probability))

                drift = {
                    "paired_models_from_first_fit": 0,
                    "tv_from_first_fit": float("nan"),
                    "js_from_first_fit": float("nan"),
                    "hard_change_rate_from_first_fit": float("nan"),
                    "mean_logit_rmse_from_first_fit": float("nan"),
                    "mean_logit_cosine_from_first_fit": float("nan"),
                    "paired_models_from_previous_age": 0,
                    "previous_post_fit_age": float("nan"),
                    "tv_from_previous_age": float("nan"),
                    "js_from_previous_age": float("nan"),
                    "hard_change_rate_from_previous_age": float("nan"),
                    "mean_logit_rmse_from_previous_age": float("nan"),
                    "mean_logit_cosine_from_previous_age": float("nan"),
                }
                if age >= 0:
                    first_valid = valid & arrays["recorded"][first_fit_position]
                    first_ids = arrays["function_ids"][first_fit_position, first_valid]
                    current_ids = arrays["function_ids"][position, first_valid]
                    _, first_probability = normalized_counts(first_ids, total_functions)
                    _, current_probability = normalized_counts(current_ids, total_functions)
                    first_logits = arrays["probe_logits"][first_fit_position, first_valid]
                    current_logits = arrays["probe_logits"][position, first_valid]
                    drift.update(
                        {
                            "paired_models_from_first_fit": int(first_valid.sum()),
                            "tv_from_first_fit": total_variation(
                                current_probability, first_probability
                            ),
                            "js_from_first_fit": js_divergence(
                                current_probability, first_probability
                            ),
                            "hard_change_rate_from_first_fit": float(
                                (current_ids != first_ids).mean()
                            ),
                            "mean_logit_rmse_from_first_fit": float(
                                np.sqrt(np.mean(np.square(current_logits - first_logits), axis=1)).mean()
                            ),
                            "mean_logit_cosine_from_first_fit": mean_row_cosine(
                                current_logits, first_logits
                            ),
                        }
                    )

                    previous_position = (
                        first_fit_position
                        if previous_post_fit_position is None
                        else previous_post_fit_position
                    )
                    previous_valid = valid & arrays["recorded"][previous_position]
                    previous_ids = arrays["function_ids"][previous_position, previous_valid]
                    current_previous_ids = arrays["function_ids"][position, previous_valid]
                    _, previous_probability = normalized_counts(previous_ids, total_functions)
                    _, current_previous_probability = normalized_counts(
                        current_previous_ids, total_functions
                    )
                    previous_logits = arrays["probe_logits"][previous_position, previous_valid]
                    current_previous_logits = arrays["probe_logits"][position, previous_valid]
                    drift.update(
                        {
                            "paired_models_from_previous_age": int(previous_valid.sum()),
                            "previous_post_fit_age": int(ages[previous_position]),
                            "tv_from_previous_age": total_variation(
                                current_previous_probability, previous_probability
                            ),
                            "js_from_previous_age": js_divergence(
                                current_previous_probability, previous_probability
                            ),
                            "hard_change_rate_from_previous_age": float(
                                (current_previous_ids != previous_ids).mean()
                            ),
                            "mean_logit_rmse_from_previous_age": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(current_previous_logits - previous_logits),
                                        axis=1,
                                    )
                                ).mean()
                            ),
                            "mean_logit_cosine_from_previous_age": mean_row_cosine(
                                current_previous_logits, previous_logits
                            ),
                        }
                    )

                row: dict[str, Any] = {
                    "state": state.name,
                    "cohort": cohort,
                    "post_fit_age": age,
                    "model_count": int(ids.size),
                    "train_exact_rate": float(arrays["train_exact"][position, valid].mean()),
                    "unique_functions": int(np.count_nonzero(q_counts)),
                    "function_entropy_bits": entropy_bits(q_probability),
                    "baseline": baseline_name,
                    "tv_to_baseline": tv,
                    "js_to_baseline": js_divergence(q_probability, baseline_probability),
                    "tv_null_mean": float(np.mean(null_tvs)),
                    "tv_null_p95": float(np.quantile(null_tvs, 0.95)),
                    "tv_exceeds_null_p95": tv > float(np.quantile(null_tvs, 0.95)),
                    "best_soft_lambda": best_lambda,
                    "js_to_best_soft_baseline": best_soft_js,
                    "largest_function_mass": float(q_probability.max()),
                    "effective_function_support": 1.0
                    / float(np.square(q_probability).sum()),
                    "mean_absolute_step": float(
                        arrays["absolute_steps"][position, valid].mean()
                    ),
                    "mean_train_loss": float(arrays["train_loss"][position, valid].mean()),
                    "mean_train_min_margin": float(
                        arrays["train_min_margin"][position, valid].mean()
                    ),
                    **drift,
                }
                metrics_rows.append(row)

                relevant = (q_counts > 0) | (baseline_counts > 0)
                for function_id in np.flatnonzero(relevant):
                    distribution_rows.append(
                        {
                            "state": state.name,
                            "cohort": cohort,
                            "post_fit_age": age,
                            "function_id": int(function_id),
                            "q_count": int(q_counts[function_id]),
                            "q_probability": float(q_probability[function_id]),
                            "baseline_count": int(baseline_counts[function_id]),
                            "baseline_probability": float(baseline_probability[function_id]),
                            "initialization_prior_count": int(prior_counts[function_id]),
                            "initialization_prior_probability": float(
                                prior_probability[function_id]
                            ),
                            "compatible": bool(candidate_mask[function_id]),
                        }
                    )

                if age >= 0:
                    for reference_name, reference_position in (
                        ("first_fit", first_fit_position),
                        (
                            "previous_age",
                            first_fit_position
                            if previous_post_fit_position is None
                            else previous_post_fit_position,
                        ),
                    ):
                        paired = valid & arrays["recorded"][reference_position]
                        sources = arrays["function_ids"][reference_position, paired].astype(
                            np.int64
                        )
                        targets = arrays["function_ids"][position, paired].astype(np.int64)
                        encoded = sources * total_functions + targets
                        pair_counts = np.bincount(
                            encoded, minlength=total_functions * total_functions
                        )
                        for encoded_pair in np.flatnonzero(pair_counts):
                            transition_count_rows.append(
                                {
                                    "state": state.name,
                                    "cohort": cohort,
                                    "post_fit_age": age,
                                    "reference": reference_name,
                                    "reference_post_fit_age": int(ages[reference_position]),
                                    "source_function_id": int(
                                        encoded_pair // total_functions
                                    ),
                                    "target_function_id": int(
                                        encoded_pair % total_functions
                                    ),
                                    "count": int(pair_counts[encoded_pair]),
                                }
                            )

                    previous_post_fit_position = position

            # 同时保留相对随机初始化与 first-fit 的聚合迁移率。
            initial_ids = arrays["function_ids"][0]
            for position in range(1, len(ages)):
                age = int(ages[position])
                valid = arrays["recorded"][position].copy()
                final_ids = arrays["function_ids"][position]
                first_valid = valid & arrays["recorded"][first_fit_position]
                transition_rows.append(
                    {
                        "state": state.name,
                        "cohort": cohort,
                        "post_fit_age": age,
                        "model_count": int(valid.sum()),
                        "function_change_rate_from_initialization": float(
                            (final_ids[valid] != initial_ids[valid]).mean()
                        ),
                        "function_change_rate_from_first_fit": float(
                            (
                                final_ids[first_valid]
                                != arrays["function_ids"][first_fit_position, first_valid]
                            ).mean()
                        )
                        if np.any(first_valid)
                        else float("nan"),
                    }
                )

    write_csv(cfg.result_dir / "posterior_metrics_by_age.csv", metrics_rows)
    write_csv(cfg.result_dir / "function_distributions.csv", distribution_rows)
    write_csv(cfg.result_dir / "paired_function_transitions.csv", transition_rows)
    write_csv(cfg.result_dir / "function_transition_counts.csv", transition_count_rows)

    final_age = max(cfg.post_fit_ages)
    headline = [row for row in metrics_rows if row["post_fit_age"] == final_age]
    drift_overview = []
    for state in states:
        for cohort in ("ordinary", "prior_consistent"):
            trajectory = [
                row
                for row in metrics_rows
                if row["state"] == state.name
                and row["cohort"] == cohort
                and int(row["post_fit_age"]) >= 0
            ]
            trajectory.sort(key=lambda row: int(row["post_fit_age"]))
            if not trajectory:
                continue
            after_first_fit = [row for row in trajectory if int(row["post_fit_age"]) > 0]
            final_row = trajectory[-1]
            overview = {
                "state": state.name,
                "cohort": cohort,
                "observed_post_fit_ages": [int(row["post_fit_age"]) for row in trajectory],
                "intervals_after_first_fit": len(after_first_fit),
                "intervals_with_hard_function_migration": sum(
                    float(row["hard_change_rate_from_previous_age"]) > 0.0
                    for row in after_first_fit
                ),
                "intervals_with_distribution_change": sum(
                    float(row["tv_from_previous_age"]) > 0.0 for row in after_first_fit
                ),
                "intervals_with_logit_movement": sum(
                    float(row["mean_logit_rmse_from_previous_age"]) > 1e-12
                    for row in after_first_fit
                ),
                "final_tv_from_first_fit": float(final_row["tv_from_first_fit"]),
                "final_js_from_first_fit": float(final_row["js_from_first_fit"]),
                "final_hard_change_rate_from_first_fit": float(
                    final_row["hard_change_rate_from_first_fit"]
                ),
                "final_mean_logit_rmse_from_first_fit": float(
                    final_row["mean_logit_rmse_from_first_fit"]
                ),
                "final_train_exact_rate": float(final_row["train_exact_rate"]),
            }
            overview["hard_distribution_drift_observed"] = (
                overview["intervals_with_distribution_change"] > 0
            )
            overview["paired_hard_migration_observed"] = (
                overview["intervals_with_hard_function_migration"] > 0
            )
            overview["continuous_logit_dynamics_observed"] = (
                overview["intervals_with_logit_movement"] > 0
            )
            drift_overview.append(overview)
            print(
                f"  [后拟合漂移] {state.name}/{cohort} | age={int(final_row['post_fit_age']):,} | "
                f"TV(first-fit)={overview['final_tv_from_first_fit']:.6f} | "
                f"hard_change={overview['final_hard_change_rate_from_first_fit']:.6f} | "
                f"logit_RMSE={overview['final_mean_logit_rmse_from_first_fit']:.6f} | "
                f"distribution_intervals={overview['intervals_with_distribution_change']}/"
                f"{overview['intervals_after_first_fit']} | "
                f"hard_migration_intervals={overview['intervals_with_hard_function_migration']}/"
                f"{overview['intervals_after_first_fit']}"
            )
    write_csv(cfg.result_dir / "post_fit_drift_overview.csv", drift_overview)
    summary = {
        "protocol_version": "boolean_posterior_dynamics_v2",
        "profile": cfg.profile,
        "states": [asdict(state) for state in states],
        "final_post_fit_age": final_age,
        "headline_final_metrics": headline,
        "post_fit_drift_overview": drift_overview,
        "interpretation_keys": {
            "static_screening": (
                "TV/JS 不超过有限样本 null，配对函数基本不迁移，且训练时间不继续改变分布。"
            ),
            "dynamic_reweighting": (
                "TV/JS 超出 null、配对函数持续迁移或分布随 post-fit age 系统变化。"
            ),
            "measurement_boundary": (
                "本实验不定义、不代理也不排序 Kolmogorov complexity；复杂度分析应在"
                "获得独立测量方法后，对保存的函数分布与逐 seed 轨迹离线完成。"
            ),
        },
    }
    save_json(cfg.result_dir / "summary.json", summary)
    return summary


def create_plots(cfg: EffectiveConfig) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"跳过作图：无法导入 matplotlib（{error}）")
        return

    metrics_path = cfg.result_dir / "posterior_metrics_by_age.csv"
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    plots_dir = cfg.result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    states = sorted({row["state"] for row in rows})
    fig, axes = plt.subplots(len(states), 4, figsize=(20, 4 * len(states)), squeeze=False)
    for state_index, state in enumerate(states):
        for cohort, color in (("ordinary", "#2f6f9f"), ("prior_consistent", "#c74b50")):
            selected = [
                row
                for row in rows
                if row["state"] == state
                and row["cohort"] == cohort
                and int(row["post_fit_age"]) >= 0
            ]
            selected.sort(key=lambda row: int(row["post_fit_age"]))
            ages = np.asarray([int(row["post_fit_age"]) for row in selected])
            if not len(ages):
                continue
            axes[state_index, 0].plot(
                ages,
                [float(row["tv_to_baseline"]) for row in selected],
                marker="o",
                label=cohort,
                color=color,
            )
            axes[state_index, 1].plot(
                ages,
                [float(row["tv_from_first_fit"]) for row in selected],
                marker="o",
                label=cohort,
                color=color,
            )
            axes[state_index, 2].plot(
                ages,
                [float(row["hard_change_rate_from_first_fit"]) for row in selected],
                marker="o",
                label=cohort,
                color=color,
            )
            axes[state_index, 3].plot(
                ages,
                [float(row["mean_logit_rmse_from_first_fit"]) for row in selected],
                marker="o",
                label=cohort,
                color=color,
            )
        for column, title in enumerate(
            (
                "TV to static baseline",
                "TV from first fit",
                "Hard change from first fit",
                "Logit RMSE from first fit",
            )
        ):
            axes[state_index, column].set_xscale("symlog", linthresh=1)
            axes[state_index, column].set_xlabel("post-fit age")
            axes[state_index, column].set_title(f"{state}\n{title}")
            axes[state_index, column].grid(alpha=0.25)
            axes[state_index, column].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "posterior_dynamics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(states), 3, figsize=(15, 4 * len(states)), squeeze=False)
    interval_metrics = (
        ("tv_from_previous_age", "TV from previous age"),
        ("hard_change_rate_from_previous_age", "Hard migration from previous age"),
        ("mean_logit_rmse_from_previous_age", "Logit RMSE from previous age"),
    )
    for state_index, state in enumerate(states):
        for cohort, color in (("ordinary", "#2f6f9f"), ("prior_consistent", "#c74b50")):
            selected = [
                row
                for row in rows
                if row["state"] == state
                and row["cohort"] == cohort
                and int(row["post_fit_age"]) >= 0
            ]
            selected.sort(key=lambda row: int(row["post_fit_age"]))
            ages = np.asarray([int(row["post_fit_age"]) for row in selected])
            if not len(ages):
                continue
            for column, (metric, _) in enumerate(interval_metrics):
                axes[state_index, column].plot(
                    ages,
                    [float(row[metric]) for row in selected],
                    marker="o",
                    label=cohort,
                    color=color,
                )
        for column, (_, title) in enumerate(interval_metrics):
            axes[state_index, column].set_xscale("symlog", linthresh=1)
            axes[state_index, column].set_xlabel("post-fit age")
            axes[state_index, column].set_title(f"{state}\n{title}")
            axes[state_index, column].grid(alpha=0.25)
            axes[state_index, column].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "post_fit_interval_drift.png", dpi=180)
    plt.close(fig)


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    cfg = resolve_config()
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(Config.GLOBAL_SEED)
    np.random.seed(Config.GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32

    payload = config_payload(cfg)
    base_signature = stable_json_hash(payload)
    save_json(cfg.result_dir / "config.json", {"config_signature": base_signature, **payload})

    print("=== Boolean function posterior dynamics ===")
    print(f"设备：{cfg.device}")
    if cfg.device == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(
        f"配置：profile={cfg.profile} | {cfg.input_bits} -> {cfg.hidden_size} x {cfg.hidden_layers} -> 1 | "
        f"prior={cfg.prior_models:,} | ordinary/state={cfg.ordinary_models:,} | "
        f"prior-consistent/state={cfg.consistent_models:,} | max_age={max(cfg.post_fit_ages):,}"
    )
    print(f"结果目录：{cfg.result_dir}")

    started = time.perf_counter()
    prior = sample_initialization_prior(cfg, base_signature)

    # 单独导出初始化函数先验，供静态基线复核和后续离线分析。
    prior_counts = prior["function_counts"].astype(np.float64)
    prior_rows = []
    for function_id, count in enumerate(prior_counts.astype(np.int64)):
        prior_rows.append(
            {
                "function_id": function_id,
                "truth_table": format(function_id, "08b")[::-1],
                "prior_count": int(count),
                "prior_probability": count / prior_counts.sum(),
            }
        )
    write_csv(cfg.result_dir / "initialization_function_prior.csv", prior_rows)

    states, _ = select_training_states(cfg, prior)
    run_signature = stable_json_hash(
        {
            "base_signature": base_signature,
            "states": [asdict(state) for state in states],
            "gradient_reduction": "sum_over_independent_models",
        }
    )
    training: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for state in states:
        print(f"\n--- state: {state.name} ---")
        for cohort in ("ordinary", "prior_consistent"):
            _, arrays = run_state_cohort(cfg, run_signature, prior, state, cohort)
            training[(cohort, state.name)] = arrays

    summary = analyze_experiment(cfg, prior, states, training)
    create_plots(cfg)
    elapsed = time.perf_counter() - started
    manifest = {
        "config_signature": base_signature,
        "run_signature": run_signature,
        "elapsed_seconds": elapsed,
        "result_dir": str(cfg.result_dir),
        "main_outputs": {
            "summary": "summary.json",
            "metrics": "posterior_metrics_by_age.csv",
            "drift_overview": "post_fit_drift_overview.csv",
            "function_distributions": "function_distributions.csv",
            "transitions": "paired_function_transitions.csv",
            "transition_counts": "function_transition_counts.csv",
            "state_selection": "selected_training_states.json",
            "initialization_prior": "initialization_function_prior.csv",
            "plot": "plots/posterior_dynamics.png",
            "interval_plot": "plots/post_fit_interval_drift.png",
        },
    }
    save_json(cfg.result_dir / "manifest.json", manifest)

    archive: str | None = None
    if Config.CREATE_ZIP:
        archive = shutil.make_archive(
            str(cfg.result_dir),
            "zip",
            root_dir=cfg.result_dir.parent,
            base_dir=cfg.result_dir.name,
        )
    print("\n=== 实验完成 ===")
    print(f"总耗时：{elapsed/60:.1f} min")
    print(f"汇总：{cfg.result_dir / 'summary.json'}")
    if archive:
        print(f"下载压缩包：{archive}")
    print(
        "最终状态数："
        f"{len(summary['headline_final_metrics'])}（3 states x 2 cohorts）"
    )


if __name__ == "__main__":
    main()
