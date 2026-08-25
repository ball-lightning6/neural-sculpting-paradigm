"""
3-bit Boolean 全函数空间上的静态先验后验 vs. SGD 终点分布实验。

这个实验专门检验一个比“神经网络偏好简单函数”更强的命题：

    P_SGD(f | S) 是否近似等于 P_0(f | f 与训练集 S 一致)

输入只有 3 bit，因此完整函数空间只有 2^8 = 256 个函数。我们可以逐函数比较，
不需要用 Lempel-Ziv complexity 直方图代替函数分布。

实验包含两部分：
1. 静态检验：大量随机初始化估计 P_0(f)，再与同一网络经 SGD 训练后的逐函数
   终点分布比较；同时记录首次拟合和拟合后继续训练的多个时间点。
2. 路径检验：最终训练集完全相同，但分别直接训练、正序逐样本加入、逆序逐样本
   加入，检查长期终点分布是否仍依赖训练历史。

脚本完全自包含，兼容：
    python experiment_static_prior_vs_sgd_posterior.py
    %run experiment_static_prior_vs_sgd_posterior.py
    整个文件粘贴到 AutoDL Jupyter cell 直接运行

所有常用设置都在 Config 中；不使用 argparse、环境变量或外部模块。
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
    RESULT_ROOT = BASE_DIR / "results_static_prior_vs_sgd_posterior"

    # 与论文 Boolean prior 实验同族：3 -> 64 x 10 -> 1，tanh，无归一化层。
    # prior 与训练必须使用同一套初始化；这是本实验最重要的控制之一。
    HIDDEN_SIZE = 64
    HIDDEN_LAYERS = 10
    SIGMA_W = 1.0
    SIGMA_B = 0.2

    # 约 100 万 prior 样本；对 k=4 的条件后验仍约有 6.5 万有效样本。
    PRIOR_SAMPLES = 1_048_576
    PRIOR_CHUNK_SIZE = 16_384
    PRIOR_SEED = 20260815

    # 每个静态训练条件的独立网络数。全部网络采用相同训练集，仅初始化不同。
    TRAIN_MODEL_COUNT = 8_192
    TRAIN_CHUNK_SIZE = 1_024
    TRAIN_INIT_SEED = 20261815
    LEARNING_RATE = 1e-3
    MAX_TRAIN_STEPS = 5_000

    # 首次达到训练集 hard accuracy=100% 后，再在这些年龄记录完整函数。
    POST_FIT_AGES = (0, 100, 1_000)
    # 另按最小 signed logit margin 记录，减少“刚越过 0”带来的偶然性。
    MARGIN_LEVELS = (2.0, 4.0, 8.0)

    # rule 30 的真值表低位顺序是 [0,1,1,1,1,0,0,0]。
    # 同时加入标签互补的单样本，检查 0/1 对称性。
    STATIC_STATES = (
        ("single_x0_y0", (0,), (0,)),
        ("single_x0_y1", (0,), (1,)),
        ("rule30_k2", (0, 1), (0, 1)),
        ("rule30_k3", (0, 1, 2), (0, 1, 1)),
        ("rule30_k4", (0, 1, 2, 3), (0, 1, 1, 1)),
    )

    # 路径实验最终都训练 rule30_k3。每个 stage 使用固定 full-batch 步数，
    # 到达最终训练集后再继续训练并记录，观察路径差异是否消失。
    RUN_PATH_TEST = True
    PATH_MODEL_COUNT = 4_096
    PATH_CHUNK_SIZE = 1_024
    PATH_STAGE_STEPS = 500
    PATH_FINAL_EXTRA_STEPS = (0, 500, 2_000)

    BOOTSTRAP_REPEATS = 1_000
    ANALYSIS_SEED = 20262815

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESUME_EXISTING = True
    CREATE_ZIP = True

    # notebook 中改成 True，可在几十秒内检查完整流程。
    SMOKE_TEST = False


PROTOCOL_VERSION = "static_prior_vs_sgd_function_posterior_v1"
INPUT_COUNT = 8
FUNCTION_COUNT = 256


@dataclass(frozen=True)
class ModelSpec:
    hidden_size: int
    hidden_layers: int
    sigma_w: float
    sigma_b: float
    learning_rate: float


@dataclass(frozen=True)
class StateSpec:
    name: str
    indices: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class EffectiveConfig:
    protocol_version: str
    result_root: str
    model: ModelSpec
    prior_samples: int
    prior_chunk_size: int
    prior_seed: int
    train_model_count: int
    train_chunk_size: int
    train_init_seed: int
    max_train_steps: int
    post_fit_ages: tuple[int, ...]
    margin_levels: tuple[float, ...]
    static_states: tuple[StateSpec, ...]
    run_path_test: bool
    path_model_count: int
    path_chunk_size: int
    path_stage_steps: int
    path_final_extra_steps: tuple[int, ...]
    bootstrap_repeats: int
    analysis_seed: int
    device: str
    allow_tf32: bool
    smoke_test: bool


def get_effective_config() -> EffectiveConfig:
    states = tuple(
        StateSpec(str(name), tuple(map(int, indices)), tuple(map(int, targets)))
        for name, indices, targets in Config.STATIC_STATES
    )
    if Config.SMOKE_TEST:
        return EffectiveConfig(
            protocol_version=PROTOCOL_VERSION,
            result_root=str(Config.RESULT_ROOT / "smoke"),
            model=ModelSpec(16, 2, 1.0, 0.2, 3e-3),
            prior_samples=2_048,
            prior_chunk_size=512,
            prior_seed=int(Config.PRIOR_SEED),
            train_model_count=128,
            train_chunk_size=32,
            train_init_seed=int(Config.TRAIN_INIT_SEED),
            max_train_steps=300,
            post_fit_ages=(0, 20),
            margin_levels=(1.0, 2.0),
            static_states=states[:3],
            run_path_test=True,
            path_model_count=64,
            path_chunk_size=32,
            path_stage_steps=30,
            path_final_extra_steps=(0, 50),
            bootstrap_repeats=100,
            analysis_seed=int(Config.ANALYSIS_SEED),
            device=str(Config.DEVICE),
            allow_tf32=bool(Config.ALLOW_TF32),
            smoke_test=True,
        )
    return EffectiveConfig(
        protocol_version=PROTOCOL_VERSION,
        result_root=str(Config.RESULT_ROOT),
        model=ModelSpec(
            int(Config.HIDDEN_SIZE),
            int(Config.HIDDEN_LAYERS),
            float(Config.SIGMA_W),
            float(Config.SIGMA_B),
            float(Config.LEARNING_RATE),
        ),
        prior_samples=int(Config.PRIOR_SAMPLES),
        prior_chunk_size=int(Config.PRIOR_CHUNK_SIZE),
        prior_seed=int(Config.PRIOR_SEED),
        train_model_count=int(Config.TRAIN_MODEL_COUNT),
        train_chunk_size=int(Config.TRAIN_CHUNK_SIZE),
        train_init_seed=int(Config.TRAIN_INIT_SEED),
        max_train_steps=int(Config.MAX_TRAIN_STEPS),
        post_fit_ages=tuple(map(int, Config.POST_FIT_AGES)),
        margin_levels=tuple(map(float, Config.MARGIN_LEVELS)),
        static_states=states,
        run_path_test=bool(Config.RUN_PATH_TEST),
        path_model_count=int(Config.PATH_MODEL_COUNT),
        path_chunk_size=int(Config.PATH_CHUNK_SIZE),
        path_stage_steps=int(Config.PATH_STAGE_STEPS),
        path_final_extra_steps=tuple(map(int, Config.PATH_FINAL_EXTRA_STEPS)),
        bootstrap_repeats=int(Config.BOOTSTRAP_REPEATS),
        analysis_seed=int(Config.ANALYSIS_SEED),
        device=str(Config.DEVICE),
        allow_tf32=bool(Config.ALLOW_TF32),
        smoke_test=False,
    )


def validate_config(cfg: EffectiveConfig) -> None:
    if cfg.prior_samples <= 0 or cfg.train_model_count <= 0:
        raise ValueError("采样数必须为正数")
    if cfg.prior_chunk_size <= 0 or cfg.train_chunk_size <= 0:
        raise ValueError("chunk size 必须为正数")
    if cfg.model.hidden_size <= 0 or cfg.model.hidden_layers <= 0:
        raise ValueError("网络宽度和隐藏层数必须为正数")
    if tuple(sorted(set(cfg.post_fit_ages))) != cfg.post_fit_ages:
        raise ValueError("POST_FIT_AGES 必须严格递增且不重复")
    if not cfg.post_fit_ages or cfg.post_fit_ages[0] != 0:
        raise ValueError("POST_FIT_AGES 必须从 0 开始")
    if tuple(sorted(set(cfg.margin_levels))) != cfg.margin_levels:
        raise ValueError("MARGIN_LEVELS 必须严格递增且不重复")
    for state in cfg.static_states:
        if not state.indices or len(state.indices) != len(state.targets):
            raise ValueError(f"非法训练状态：{state}")
        if len(set(state.indices)) != len(state.indices):
            raise ValueError(f"训练状态包含重复输入：{state.name}")
        if any(index < 0 or index >= INPUT_COUNT for index in state.indices):
            raise ValueError(f"输入索引超界：{state.name}")
        if any(target not in (0, 1) for target in state.targets):
            raise ValueError(f"标签必须是 0/1：{state.name}")
    if cfg.run_path_test:
        if cfg.path_model_count <= 0 or cfg.path_chunk_size <= 0 or cfg.path_stage_steps <= 0:
            raise ValueError("路径实验的模型数、chunk 和 stage 步数必须为正数")
        if (
            not cfg.path_final_extra_steps
            or cfg.path_final_extra_steps[0] != 0
            or tuple(sorted(set(cfg.path_final_extra_steps))) != cfg.path_final_extra_steps
        ):
            raise ValueError("PATH_FINAL_EXTRA_STEPS 必须从 0 开始、严格递增且不重复")


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def chunk_ranges(total: int, chunk_size: int) -> Iterable[tuple[int, int, int]]:
    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        yield chunk_index, start, min(total, start + chunk_size)


# =============================================================================
# 3-bit Boolean 函数表示
# =============================================================================


def all_inputs(device: torch.device | None = None) -> torch.Tensor:
    values = [[(index >> 2) & 1, (index >> 1) & 1, index & 1] for index in range(8)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def function_ids_from_logits(logits: torch.Tensor) -> torch.Tensor:
    powers = 2 ** torch.arange(INPUT_COUNT, device=logits.device, dtype=torch.int64)
    return ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)


def truth_bits(function_id: int) -> np.ndarray:
    return ((int(function_id) >> np.arange(INPUT_COUNT)) & 1).astype(np.int8)


def state_mask(state: StateSpec) -> np.ndarray:
    ids = np.arange(FUNCTION_COUNT, dtype=np.uint16)
    mask = np.ones(FUNCTION_COUNT, dtype=bool)
    for index, target in zip(state.indices, state.targets):
        mask &= (((ids >> int(index)) & 1) == int(target))
    return mask


def anf_features(function_id: int) -> tuple[int, int]:
    """返回 3 变量 Boolean 函数的 ANF 最高次数和非零项数。"""
    coefficients = truth_bits(function_id).copy()
    for bit in range(3):
        for mask in range(INPUT_COUNT):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    nonzero = np.flatnonzero(coefficients)
    if nonzero.size == 0:
        return 0, 0
    degree = max(int(int(mask).bit_count()) for mask in nonzero)
    return degree, int(nonzero.size)


def function_feature_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for function_id in range(FUNCTION_COUNT):
        bits = truth_bits(function_id)
        degree, terms = anf_features(function_id)
        rows.append(
            {
                "function_id": function_id,
                "truth_table": "".join(str(int(value)) for value in bits),
                "ones": int(bits.sum()),
                "linear_transitions": int(np.not_equal(bits[1:], bits[:-1]).sum()),
                "cyclic_transitions": int(np.not_equal(bits, np.roll(bits, 1)).sum()),
                "anf_degree": degree,
                "anf_terms": terms,
            }
        )
    return rows


# =============================================================================
# 与论文同族、但 prior 与训练严格匹配的 batched tanh FCN
# =============================================================================


class BatchedTanhFCN(nn.Module):
    """首维是相互独立的网络；不同模型之间没有任何参数共享。"""

    def __init__(
        self,
        ensemble_size: int,
        spec: ModelSpec,
        initialization_seed: int,
        device: torch.device,
        trainable: bool,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(initialization_seed))

        widths = [3] + [spec.hidden_size] * spec.hidden_layers + [1]
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
            ) * (spec.sigma_w / math.sqrt(in_width))
            # 对齐官方 prior 代码的量级：bias std 随 fan-in 缩小。
            bias = torch.randn(
                ensemble_size,
                out_width,
                generator=generator,
                device=device,
                dtype=torch.float32,
            ) * (spec.sigma_b * spec.sigma_w / in_width)
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


@torch.inference_mode()
def evaluate_functions(model: nn.Module, probe_inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits = model(probe_inputs).squeeze(-1)
    ids = function_ids_from_logits(logits)
    return (
        ids.cpu().numpy().astype(np.uint16),
        logits.cpu().to(torch.float32).numpy(),
    )


# =============================================================================
# 初始化函数先验
# =============================================================================


def sample_initialization_prior(
    cfg: EffectiveConfig,
    result_dir: Path,
    signature: str,
) -> dict[str, np.ndarray]:
    output = result_dir / "initialization_prior.npz"
    metadata_path = result_dir / "initialization_prior.json"
    if Config.RESUME_EXISTING and output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            with np.load(output) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files}
            print(f"[prior] 复用已有 {int(arrays['counts'].sum()):,} 次初始化采样")
            return arrays

    device = torch.device(cfg.device)
    probe = all_inputs(device)
    counts = np.zeros(FUNCTION_COUNT, dtype=np.int64)
    counts_a = np.zeros_like(counts)
    counts_b = np.zeros_like(counts)
    started = time.perf_counter()
    ranges = list(chunk_ranges(cfg.prior_samples, cfg.prior_chunk_size))
    for chunk_index, start, end in ranges:
        count = end - start
        model = BatchedTanhFCN(
            count,
            cfg.model,
            cfg.prior_seed + chunk_index,
            device,
            trainable=False,
        )
        probe_batch = probe[None].expand(count, -1, -1)
        function_ids, _ = evaluate_functions(model, probe_batch)
        chunk_counts = np.bincount(function_ids.astype(np.int64), minlength=FUNCTION_COUNT)
        counts += chunk_counts
        if chunk_index % 2 == 0:
            counts_a += chunk_counts
        else:
            counts_b += chunk_counts
        del model, probe_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (chunk_index + 1) % max(1, len(ranges) // 20) == 0 or end == cfg.prior_samples:
            elapsed = time.perf_counter() - started
            print(
                f"[prior] {end:,}/{cfg.prior_samples:,} | "
                f"{end / max(elapsed, 1e-9):,.0f} init/s | "
                f"已见函数={int(np.count_nonzero(counts))}/256"
            )

    arrays = {"counts": counts, "counts_a": counts_a, "counts_b": counts_b}
    np.savez_compressed(output, **arrays)
    save_json(
        metadata_path,
        {
            "config_signature": signature,
            "samples": int(counts.sum()),
            "samples_a": int(counts_a.sum()),
            "samples_b": int(counts_b.sum()),
            "observed_functions": int(np.count_nonzero(counts)),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    return arrays


# =============================================================================
# 冷启动 SGD：首次拟合、拟合后年龄和固定 margin
# =============================================================================


def endpoint_names(cfg: EffectiveConfig) -> tuple[str, ...]:
    names = [f"post_fit_age_{age:06d}" for age in cfg.post_fit_ages]
    names.extend(f"margin_{level:g}" for level in cfg.margin_levels)
    names.append("last_step")
    return tuple(names)


@torch.inference_mode()
def train_status(
    model: nn.Module,
    train_inputs: torch.Tensor,
    signed_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(train_inputs).squeeze(-1)
    margins = logits * signed_targets
    exact = (margins >= 0).all(dim=1)
    minimum_margin = margins.min(dim=1).values
    losses = F.softplus(-margins).mean(dim=1)
    return exact, minimum_margin, losses


def train_direct_chunk(
    cfg: EffectiveConfig,
    state: StateSpec,
    count: int,
    chunk_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(cfg.device)
    model = BatchedTanhFCN(
        count,
        cfg.model,
        cfg.train_init_seed + chunk_index,
        device,
        trainable=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    probe = all_inputs(device)
    probe_batch = probe[None].expand(count, -1, -1)
    train_base = probe[list(state.indices)]
    train_inputs = train_base[None].expand(count, -1, -1)
    signed_base = torch.tensor(
        [1.0 if target else -1.0 for target in state.targets],
        dtype=torch.float32,
        device=device,
    )
    signed_targets = signed_base[None].expand(count, -1)

    names = endpoint_names(cfg)
    endpoint_count = len(names)
    function_ids = np.full((endpoint_count, count), 65535, dtype=np.uint16)
    probe_logits = np.full((endpoint_count, count, INPUT_COUNT), np.nan, dtype=np.float32)
    endpoint_steps = np.full((endpoint_count, count), -1, dtype=np.int64)
    endpoint_train_loss = np.full((endpoint_count, count), np.nan, dtype=np.float32)
    endpoint_min_margin = np.full((endpoint_count, count), np.nan, dtype=np.float32)
    first_fit_steps = np.full(count, -1, dtype=np.int64)
    recorded = np.zeros((endpoint_count, count), dtype=bool)

    age_endpoint_indices = {
        age: names.index(f"post_fit_age_{age:06d}") for age in cfg.post_fit_ages
    }
    margin_endpoint_indices = {
        level: names.index(f"margin_{level:g}") for level in cfg.margin_levels
    }
    last_index = names.index("last_step")
    started = time.perf_counter()

    def record_due(
        due_by_endpoint: dict[int, np.ndarray],
        step: int,
        losses: torch.Tensor,
        minimum_margin: torch.Tensor,
    ) -> None:
        if not due_by_endpoint:
            return
        ids, logits = evaluate_functions(model, probe_batch)
        loss_np = losses.cpu().to(torch.float32).numpy()
        margin_np = minimum_margin.cpu().to(torch.float32).numpy()
        for endpoint_index, due in due_by_endpoint.items():
            if not np.any(due):
                continue
            function_ids[endpoint_index, due] = ids[due]
            probe_logits[endpoint_index, due] = logits[due]
            endpoint_steps[endpoint_index, due] = step
            endpoint_train_loss[endpoint_index, due] = loss_np[due]
            endpoint_min_margin[endpoint_index, due] = margin_np[due]
            recorded[endpoint_index, due] = True

    # step 0 也可能已经与训练集一致，这正是静态条件后验最直接的部分。
    exact, minimum_margin, losses = train_status(model, train_inputs, signed_targets)
    exact_np = exact.cpu().numpy()
    first_fit_steps[exact_np] = 0
    initial_due: dict[int, np.ndarray] = {}
    for age, endpoint_index in age_endpoint_indices.items():
        if age == 0:
            initial_due[endpoint_index] = exact_np.copy()
    margin_np = minimum_margin.cpu().numpy()
    for level, endpoint_index in margin_endpoint_indices.items():
        initial_due[endpoint_index] = margin_np >= level
    record_due(initial_due, 0, losses, minimum_margin)

    final_step = 0
    for step in range(1, cfg.max_train_steps + 1):
        final_step = step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs).squeeze(-1)
        margins = logits * signed_targets
        loss_by_model = F.softplus(-margins).mean(dim=1)
        loss_by_model.sum().backward()
        optimizer.step()

        exact, minimum_margin, losses = train_status(model, train_inputs, signed_targets)
        exact_np = exact.cpu().numpy()
        newly_fit = (first_fit_steps < 0) & exact_np
        first_fit_steps[newly_fit] = step
        due_by_endpoint: dict[int, np.ndarray] = {}
        for age, endpoint_index in age_endpoint_indices.items():
            due = (
                (~recorded[endpoint_index])
                & (first_fit_steps >= 0)
                & (step >= first_fit_steps + age)
            )
            if np.any(due):
                due_by_endpoint[endpoint_index] = due
        margin_np = minimum_margin.cpu().numpy()
        for level, endpoint_index in margin_endpoint_indices.items():
            due = (~recorded[endpoint_index]) & (margin_np >= level)
            if np.any(due):
                due_by_endpoint[endpoint_index] = due
        record_due(due_by_endpoint, step, losses, minimum_margin)

        required = recorded[:-1]
        if bool(required.all()):
            break
        if step % (100 if cfg.smoke_test else 500) == 0:
            fitted = int((first_fit_steps >= 0).sum())
            print(
                f"      {state.name} chunk={chunk_index + 1} step={step:,} | "
                f"fitted={fitted}/{count} | "
                f"loss={float(losses.mean().item()):.3e} | "
                f"margin_min={float(minimum_margin.min().item()):.3f}"
            )

    # 无论是否命中所有预设 endpoint，都保存最终状态。
    exact, minimum_margin, losses = train_status(model, train_inputs, signed_targets)
    final_due = {last_index: np.ones(count, dtype=bool)}
    record_due(final_due, final_step, losses, minimum_margin)
    arrays = {
        "function_ids": function_ids,
        "probe_logits": probe_logits,
        "endpoint_steps": endpoint_steps,
        "endpoint_train_loss": endpoint_train_loss,
        "endpoint_min_margin": endpoint_min_margin,
        "recorded": recorded,
        "first_fit_steps": first_fit_steps,
    }
    metadata = {
        "state": asdict(state),
        "endpoint_names": list(names),
        "count": count,
        "chunk_index": chunk_index,
        "final_step": final_step,
        "all_required_recorded": bool(recorded[:-1].all()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return metadata, arrays


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """避免中断后留下一个看似存在、实际损坏的 npz。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_direct_experiments(
    cfg: EffectiveConfig,
    result_dir: Path,
    signature: str,
) -> dict[str, dict[str, np.ndarray]]:
    """运行或恢复所有静态训练状态，并返回逐状态聚合数组。"""
    aggregates: dict[str, dict[str, np.ndarray]] = {}
    device = torch.device(cfg.device)
    for state_index, state in enumerate(cfg.static_states, start=1):
        state_dir = result_dir / "direct" / state.name
        chunks_dir = state_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk_arrays: list[dict[str, np.ndarray]] = []
        ranges = list(chunk_ranges(cfg.train_model_count, cfg.train_chunk_size))
        state_started = time.perf_counter()
        print(
            f"\n[direct {state_index}/{len(cfg.static_states)}] {state.name} | "
            f"S={list(zip(state.indices, state.targets))} | models={cfg.train_model_count:,}"
        )
        for chunk_index, start, end in ranges:
            npz_path = chunks_dir / f"chunk_{chunk_index:04d}.npz"
            json_path = chunks_dir / f"chunk_{chunk_index:04d}.json"
            arrays: dict[str, np.ndarray] | None = None
            if Config.RESUME_EXISTING and npz_path.exists() and json_path.exists():
                try:
                    metadata = json.loads(json_path.read_text(encoding="utf-8"))
                    if (
                        metadata.get("config_signature") == signature
                        and metadata.get("state", {}).get("name") == state.name
                        and int(metadata.get("start", -1)) == start
                        and int(metadata.get("end", -1)) == end
                    ):
                        arrays = load_npz(npz_path)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    arrays = None
            if arrays is None:
                metadata, arrays = train_direct_chunk(
                    cfg=cfg,
                    state=state,
                    count=end - start,
                    chunk_index=chunk_index,
                )
                metadata.update(
                    {
                        "config_signature": signature,
                        "start": start,
                        "end": end,
                    }
                )
                save_npz_atomic(npz_path, arrays)
                save_json(json_path, metadata)
            else:
                print(
                    f"    复用 chunk {chunk_index + 1}/{len(ranges)} | "
                    f"models={start:,}..{end - 1:,}"
                )
            chunk_arrays.append(arrays)
            clear_device_cache(device)

        endpoint_axis_keys = (
            "function_ids",
            "probe_logits",
            "endpoint_steps",
            "endpoint_train_loss",
            "endpoint_min_margin",
            "recorded",
        )
        aggregate: dict[str, np.ndarray] = {}
        for key in endpoint_axis_keys:
            aggregate[key] = np.concatenate([arrays[key] for arrays in chunk_arrays], axis=1)
        aggregate["first_fit_steps"] = np.concatenate(
            [arrays["first_fit_steps"] for arrays in chunk_arrays], axis=0
        )
        aggregates[state.name] = aggregate
        save_npz_atomic(state_dir / "aggregate.npz", aggregate)

        names = endpoint_names(cfg)
        endpoint_summary = []
        for endpoint_index, endpoint_name in enumerate(names):
            valid = aggregate["recorded"][endpoint_index]
            endpoint_summary.append(
                {
                    "endpoint": endpoint_name,
                    "recorded": int(valid.sum()),
                    "recorded_fraction": float(valid.mean()),
                    "median_step": (
                        float(np.median(aggregate["endpoint_steps"][endpoint_index, valid]))
                        if np.any(valid)
                        else None
                    ),
                }
            )
        fitted = aggregate["first_fit_steps"] >= 0
        save_json(
            state_dir / "aggregate.json",
            {
                "config_signature": signature,
                "state": asdict(state),
                "model_count": cfg.train_model_count,
                "fitted_count": int(fitted.sum()),
                "fitted_fraction": float(fitted.mean()),
                "first_fit_step_median": (
                    float(np.median(aggregate["first_fit_steps"][fitted]))
                    if np.any(fitted)
                    else None
                ),
                "endpoints": endpoint_summary,
                "elapsed_seconds": float(time.perf_counter() - state_started),
            },
        )
        print(
            f"    聚合完成：fitted={int(fitted.sum())}/{cfg.train_model_count} | "
            f"耗时={time.perf_counter() - state_started:.1f}s"
        )
    return aggregates


# =============================================================================
# 分布统计：逐函数、预测边缘和粗复杂度投影
# =============================================================================


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def ids_to_counts(function_ids: np.ndarray) -> tuple[np.ndarray, int]:
    ids = np.asarray(function_ids).reshape(-1)
    valid = ids < FUNCTION_COUNT
    counts = np.bincount(ids[valid].astype(np.int64), minlength=FUNCTION_COUNT)
    return counts.astype(np.int64), int(valid.sum())


def conditioned_distribution(counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    conditioned = np.asarray(counts, dtype=np.float64) * np.asarray(mask, dtype=np.float64)
    return normalize_counts(conditioned)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(left) - np.asarray(right)).sum())


def js_divergence_bits(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    midpoint = 0.5 * (left + right)

    def kl_bits(source: np.ndarray, target: np.ndarray) -> float:
        active = source > 0
        if not np.any(active):
            return 0.0
        return float(np.sum(source[active] * np.log2(source[active] / target[active])))

    return 0.5 * kl_bits(left, midpoint) + 0.5 * kl_bits(right, midpoint)


def split_sampling_tv(function_ids: np.ndarray) -> float | None:
    ids = np.asarray(function_ids).reshape(-1)
    ids = ids[ids < FUNCTION_COUNT]
    if ids.size < 4:
        return None
    left, _ = ids_to_counts(ids[::2])
    right, _ = ids_to_counts(ids[1::2])
    return total_variation(normalize_counts(left), normalize_counts(right))


def bootstrap_tv_null(
    reference: np.ndarray,
    sample_count: int,
    observed_tv: float,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    if sample_count <= 0 or float(reference.sum()) <= 0 or repeats <= 0:
        return {
            "repeats": 0,
            "tv_null_mean": None,
            "tv_null_p95": None,
            "tv_null_p99": None,
            "p_value_ge_observed": None,
        }
    null_tvs = np.empty(repeats, dtype=np.float64)
    # 逐次采样可避免 repeats x 256 的临时对象在极大重复数下失控。
    for repeat in range(repeats):
        simulated = rng.multinomial(sample_count, reference)
        null_tvs[repeat] = total_variation(normalize_counts(simulated), reference)
    return {
        "repeats": repeats,
        "tv_null_mean": float(null_tvs.mean()),
        "tv_null_p95": float(np.quantile(null_tvs, 0.95)),
        "tv_null_p99": float(np.quantile(null_tvs, 0.99)),
        "p_value_ge_observed": float((1 + np.sum(null_tvs >= observed_tv)) / (repeats + 1)),
    }


def feature_marginal(distribution: np.ndarray, feature_values: np.ndarray) -> np.ndarray:
    maximum = int(feature_values.max(initial=0))
    result = np.zeros(maximum + 1, dtype=np.float64)
    np.add.at(result, feature_values.astype(np.int64), distribution)
    return result


def analyze_direct_experiments(
    cfg: EffectiveConfig,
    result_dir: Path,
    prior: dict[str, np.ndarray],
    aggregates: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    analysis_dir = result_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    features = function_feature_rows()
    write_csv(analysis_dir / "function_features.csv", features)
    feature_arrays = {
        name: np.asarray([int(row[name]) for row in features], dtype=np.int64)
        for name in ("ones", "linear_transitions", "cyclic_transitions", "anf_degree", "anf_terms")
    }
    rng = np.random.default_rng(cfg.analysis_seed)
    metric_rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    predictive_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    summary_states: dict[str, Any] = {}

    for state in cfg.static_states:
        aggregate = aggregates[state.name]
        mask = state_mask(state)
        static = conditioned_distribution(prior["counts"], mask)
        static_a = conditioned_distribution(prior["counts_a"], mask)
        static_b = conditioned_distribution(prior["counts_b"], mask)
        prior_half_tv = total_variation(static_a, static_b)
        consistent_prior_samples = int((prior["counts"] * mask).sum())
        state_summary: dict[str, Any] = {
            "training_indices": list(state.indices),
            "training_targets": list(state.targets),
            "consistent_function_count": int(mask.sum()),
            "consistent_prior_samples": consistent_prior_samples,
            "prior_half_tv": prior_half_tv,
            "endpoints": {},
        }

        for endpoint_index, endpoint_name in enumerate(endpoint_names(cfg)):
            valid = aggregate["recorded"][endpoint_index]
            endpoint_ids = aggregate["function_ids"][endpoint_index, valid]
            counts, observed_count = ids_to_counts(endpoint_ids)
            empirical = normalize_counts(counts)
            if observed_count == 0:
                continue
            tv = total_variation(empirical, static)
            js = js_divergence_bits(empirical, static)
            empirical_split_tv = split_sampling_tv(endpoint_ids)
            inconsistent_count = int(counts[~mask].sum())
            unseen_prior_support = (prior["counts"] == 0) & (counts > 0)
            zero_prior_support_functions = int(unseen_prior_support.sum())
            zero_prior_support_mass = float(empirical[unseen_prior_support].sum())
            bootstrap = bootstrap_tv_null(
                reference=static,
                sample_count=observed_count,
                observed_tv=tv,
                repeats=cfg.bootstrap_repeats,
                rng=rng,
            )
            metric = {
                "state": state.name,
                "endpoint": endpoint_name,
                "observed_models": observed_count,
                "tv_function": tv,
                "js_bits_function": js,
                "prior_half_tv": prior_half_tv,
                "sgd_half_tv": empirical_split_tv,
                "inconsistent_count": inconsistent_count,
                "inconsistent_fraction": inconsistent_count / observed_count,
                "zero_sampled_prior_support_functions": zero_prior_support_functions,
                "zero_sampled_prior_support_mass": zero_prior_support_mass,
                **bootstrap,
            }
            metric_rows.append(metric)
            state_summary["endpoints"][endpoint_name] = metric

            for input_index in range(INPUT_COUNT):
                output_one = ((np.arange(FUNCTION_COUNT) >> input_index) & 1).astype(bool)
                predictive_rows.append(
                    {
                        "state": state.name,
                        "endpoint": endpoint_name,
                        "input_index": input_index,
                        "input_bits": "".join(map(str, all_inputs()[input_index].to(torch.int64).tolist())),
                        "in_training_set": input_index in state.indices,
                        "static_p_y1": float(static[output_one].sum()),
                        "sgd_p_y1": float(empirical[output_one].sum()),
                        "absolute_difference": float(
                            abs(static[output_one].sum() - empirical[output_one].sum())
                        ),
                    }
                )

            for feature_name, values in feature_arrays.items():
                static_marginal = feature_marginal(static, values)
                empirical_marginal = feature_marginal(empirical, values)
                coarse_rows.append(
                    {
                        "state": state.name,
                        "endpoint": endpoint_name,
                        "feature": feature_name,
                        "tv": total_variation(static_marginal, empirical_marginal),
                        "js_bits": js_divergence_bits(static_marginal, empirical_marginal),
                    }
                )

            for function_id in range(FUNCTION_COUNT):
                row = features[function_id]
                function_rows.append(
                    {
                        "state": state.name,
                        "endpoint": endpoint_name,
                        **row,
                        "consistent_with_state": bool(mask[function_id]),
                        "prior_count": int(prior["counts"][function_id]),
                        "static_posterior_probability": float(static[function_id]),
                        "sgd_count": int(counts[function_id]),
                        "sgd_probability": float(empirical[function_id]),
                    }
                )
        summary_states[state.name] = state_summary

    write_csv(analysis_dir / "distribution_metrics.csv", metric_rows)
    write_csv(analysis_dir / "coarse_projection_metrics.csv", coarse_rows)
    write_csv(analysis_dir / "predictive_marginals.csv", predictive_rows)
    write_csv(analysis_dir / "function_distributions.csv", function_rows)
    summary = {
        "states": summary_states,
        "interpretation_guardrails": [
            "逐函数 TV/JS 才是对静态后验命题的直接检验；粗复杂度投影相似不能推出函数分布相似。",
            "prior 中未采到某函数只表示有限 Monte Carlo 支持为零，不代表真实初始化概率严格为零。",
            "prior_half_tv 和 sgd_half_tv 分别给出 prior Monte Carlo 与 SGD 样本量导致的噪声标尺。",
            "last_step 可能包含尚未完全拟合训练集的模型，必须结合 inconsistent_fraction 解读。",
        ],
    }
    save_json(analysis_dir / "direct_summary.json", summary)
    return {
        "summary": summary,
        "metric_rows": metric_rows,
        "coarse_rows": coarse_rows,
        "predictive_rows": predictive_rows,
        "function_rows": function_rows,
    }


# =============================================================================
# 相同最终训练集，不同训练路径
# =============================================================================


def path_protocols() -> dict[str, tuple[StateSpec, ...]]:
    final_state = StateSpec("rule30_k3", (0, 1, 2), (0, 1, 1))
    return {
        "direct": (final_state, final_state, final_state),
        "forward": (
            StateSpec("forward_k1", (0,), (0,)),
            StateSpec("forward_k2", (0, 1), (0, 1)),
            final_state,
        ),
        "reverse": (
            StateSpec("reverse_k1", (2,), (1,)),
            StateSpec("reverse_k2", (2, 1), (1, 1)),
            StateSpec("reverse_k3", (2, 1, 0), (1, 1, 0)),
        ),
    }


def path_endpoint_names(cfg: EffectiveConfig) -> tuple[str, ...]:
    names = ["after_stage_1", "after_stage_2"]
    names.extend(f"final_plus_{steps:06d}" for steps in cfg.path_final_extra_steps)
    return tuple(names)


def state_training_tensors(
    state: StateSpec,
    probe: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = probe[list(state.indices)][None].expand(count, -1, -1)
    signed = torch.tensor(
        [1.0 if target else -1.0 for target in state.targets],
        dtype=torch.float32,
        device=probe.device,
    )[None].expand(count, -1)
    return inputs, signed


def optimize_fixed_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: torch.Tensor,
    signed_targets: torch.Tensor,
    steps: int,
) -> None:
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs).squeeze(-1)
        loss_by_model = F.softplus(-(logits * signed_targets)).mean(dim=1)
        loss_by_model.sum().backward()
        optimizer.step()


def train_path_chunk(
    cfg: EffectiveConfig,
    protocol_name: str,
    stages: tuple[StateSpec, ...],
    count: int,
    chunk_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(cfg.device)
    model = BatchedTanhFCN(
        count,
        cfg.model,
        cfg.train_init_seed + 100_000 + chunk_index,
        device,
        trainable=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    probe = all_inputs(device)
    probe_batch = probe[None].expand(count, -1, -1)
    final_state = StateSpec("rule30_k3", (0, 1, 2), (0, 1, 1))
    final_inputs, final_signed = state_training_tensors(final_state, probe, count)
    names = path_endpoint_names(cfg)
    endpoint_count = len(names)
    function_ids = np.full((endpoint_count, count), 65535, dtype=np.uint16)
    probe_logits = np.full((endpoint_count, count, INPUT_COUNT), np.nan, dtype=np.float32)
    final_train_loss = np.full((endpoint_count, count), np.nan, dtype=np.float32)
    final_min_margin = np.full((endpoint_count, count), np.nan, dtype=np.float32)
    endpoint_total_steps = np.full(endpoint_count, -1, dtype=np.int64)
    started = time.perf_counter()
    total_steps = 0

    def record(endpoint_index: int) -> None:
        ids, logits = evaluate_functions(model, probe_batch)
        _, margin, losses = train_status(model, final_inputs, final_signed)
        function_ids[endpoint_index] = ids
        probe_logits[endpoint_index] = logits
        final_train_loss[endpoint_index] = losses.cpu().numpy()
        final_min_margin[endpoint_index] = margin.cpu().numpy()
        endpoint_total_steps[endpoint_index] = total_steps

    for stage_index, stage in enumerate(stages):
        train_inputs, signed_targets = state_training_tensors(stage, probe, count)
        optimize_fixed_steps(
            model,
            optimizer,
            train_inputs,
            signed_targets,
            cfg.path_stage_steps,
        )
        total_steps += cfg.path_stage_steps
        if stage_index < 2:
            record(stage_index)
        print(
            f"      {protocol_name} chunk={chunk_index + 1} | "
            f"stage={stage_index + 1}/3 | total_steps={total_steps:,}"
        )

    extra_steps = tuple(sorted(set(cfg.path_final_extra_steps)))
    final_inputs_train, final_signed_train = state_training_tensors(final_state, probe, count)
    completed_extra = 0
    for extra_index, target_extra in enumerate(extra_steps, start=2):
        delta = target_extra - completed_extra
        if delta > 0:
            optimize_fixed_steps(
                model,
                optimizer,
                final_inputs_train,
                final_signed_train,
                delta,
            )
            total_steps += delta
            completed_extra = target_extra
        record(extra_index)

    arrays = {
        "function_ids": function_ids,
        "probe_logits": probe_logits,
        "final_train_loss": final_train_loss,
        "final_min_margin": final_min_margin,
        "endpoint_total_steps": endpoint_total_steps,
    }
    metadata = {
        "protocol": protocol_name,
        "stages": [asdict(stage) for stage in stages],
        "endpoint_names": list(names),
        "count": count,
        "chunk_index": chunk_index,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return metadata, arrays


def run_path_experiments(
    cfg: EffectiveConfig,
    result_dir: Path,
    signature: str,
) -> dict[str, dict[str, np.ndarray]]:
    if not cfg.run_path_test:
        return {}
    protocols = path_protocols()
    aggregates: dict[str, dict[str, np.ndarray]] = {}
    device = torch.device(cfg.device)
    for protocol_index, (protocol_name, stages) in enumerate(protocols.items(), start=1):
        protocol_dir = result_dir / "paths" / protocol_name
        chunks_dir = protocol_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        ranges = list(chunk_ranges(cfg.path_model_count, cfg.path_chunk_size))
        chunk_arrays: list[dict[str, np.ndarray]] = []
        started = time.perf_counter()
        print(
            f"\n[path {protocol_index}/{len(protocols)}] {protocol_name} | "
            f"models={cfg.path_model_count:,}"
        )
        for chunk_index, start, end in ranges:
            npz_path = chunks_dir / f"chunk_{chunk_index:04d}.npz"
            json_path = chunks_dir / f"chunk_{chunk_index:04d}.json"
            arrays: dict[str, np.ndarray] | None = None
            if Config.RESUME_EXISTING and npz_path.exists() and json_path.exists():
                try:
                    metadata = json.loads(json_path.read_text(encoding="utf-8"))
                    if (
                        metadata.get("config_signature") == signature
                        and metadata.get("protocol") == protocol_name
                        and int(metadata.get("start", -1)) == start
                        and int(metadata.get("end", -1)) == end
                    ):
                        arrays = load_npz(npz_path)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    arrays = None
            if arrays is None:
                metadata, arrays = train_path_chunk(
                    cfg=cfg,
                    protocol_name=protocol_name,
                    stages=stages,
                    count=end - start,
                    chunk_index=chunk_index,
                )
                metadata.update(
                    {
                        "config_signature": signature,
                        "start": start,
                        "end": end,
                    }
                )
                save_npz_atomic(npz_path, arrays)
                save_json(json_path, metadata)
            else:
                print(
                    f"    复用 chunk {chunk_index + 1}/{len(ranges)} | "
                    f"models={start:,}..{end - 1:,}"
                )
            chunk_arrays.append(arrays)
            clear_device_cache(device)

        aggregate = {
            "function_ids": np.concatenate(
                [arrays["function_ids"] for arrays in chunk_arrays], axis=1
            ),
            "probe_logits": np.concatenate(
                [arrays["probe_logits"] for arrays in chunk_arrays], axis=1
            ),
            "final_train_loss": np.concatenate(
                [arrays["final_train_loss"] for arrays in chunk_arrays], axis=1
            ),
            "final_min_margin": np.concatenate(
                [arrays["final_min_margin"] for arrays in chunk_arrays], axis=1
            ),
            "endpoint_total_steps": chunk_arrays[0]["endpoint_total_steps"],
        }
        aggregates[protocol_name] = aggregate
        save_npz_atomic(protocol_dir / "aggregate.npz", aggregate)
        save_json(
            protocol_dir / "aggregate.json",
            {
                "config_signature": signature,
                "protocol": protocol_name,
                "stages": [asdict(stage) for stage in stages],
                "model_count": cfg.path_model_count,
                "endpoint_names": list(path_endpoint_names(cfg)),
                "endpoint_total_steps": aggregate["endpoint_total_steps"].tolist(),
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )
    return aggregates


def analyze_path_experiments(
    cfg: EffectiveConfig,
    result_dir: Path,
    prior: dict[str, np.ndarray],
    aggregates: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    if not aggregates:
        return {}
    analysis_dir = result_dir / "analysis"
    final_state = StateSpec("rule30_k3", (0, 1, 2), (0, 1, 1))
    mask = state_mask(final_state)
    static = conditioned_distribution(prior["counts"], mask)
    endpoint_names_value = path_endpoint_names(cfg)
    distributions: dict[tuple[str, str], np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    features = function_feature_rows()

    for protocol_name, aggregate in aggregates.items():
        for endpoint_index, endpoint_name in enumerate(endpoint_names_value):
            ids = aggregate["function_ids"][endpoint_index]
            counts, sample_count = ids_to_counts(ids)
            empirical = normalize_counts(counts)
            distributions[(protocol_name, endpoint_name)] = empirical
            inconsistent = int(counts[~mask].sum())
            metric_rows.append(
                {
                    "protocol": protocol_name,
                    "endpoint": endpoint_name,
                    "total_steps": int(aggregate["endpoint_total_steps"][endpoint_index]),
                    "models": sample_count,
                    "final_state_exact_fraction": 1.0 - inconsistent / max(sample_count, 1),
                    "tv_to_static_posterior": total_variation(empirical, static),
                    "js_bits_to_static_posterior": js_divergence_bits(empirical, static),
                    "split_sampling_tv": split_sampling_tv(ids),
                    "median_final_loss": float(
                        np.median(aggregate["final_train_loss"][endpoint_index])
                    ),
                    "median_final_min_margin": float(
                        np.median(aggregate["final_min_margin"][endpoint_index])
                    ),
                }
            )
            for function_id in range(FUNCTION_COUNT):
                function_rows.append(
                    {
                        "protocol": protocol_name,
                        "endpoint": endpoint_name,
                        **features[function_id],
                        "consistent_with_final_state": bool(mask[function_id]),
                        "count": int(counts[function_id]),
                        "probability": float(empirical[function_id]),
                        "static_posterior_probability": float(static[function_id]),
                    }
                )

    protocol_names = tuple(aggregates)
    for endpoint_name in endpoint_names_value:
        for left_index, left_name in enumerate(protocol_names):
            for right_name in protocol_names[left_index + 1 :]:
                left = distributions[(left_name, endpoint_name)]
                right = distributions[(right_name, endpoint_name)]
                pairwise_rows.append(
                    {
                        "endpoint": endpoint_name,
                        "left_protocol": left_name,
                        "right_protocol": right_name,
                        "tv": total_variation(left, right),
                        "js_bits": js_divergence_bits(left, right),
                    }
                )

    write_csv(analysis_dir / "path_metrics.csv", metric_rows)
    write_csv(analysis_dir / "path_pairwise_metrics.csv", pairwise_rows)
    write_csv(analysis_dir / "path_function_distributions.csv", function_rows)
    summary = {
        "final_state": asdict(final_state),
        "metrics": metric_rows,
        "pairwise": pairwise_rows,
        "interpretation_guardrails": [
            "forward 与 reverse 的 stage 大小和总步数完全匹配，因此二者差异是最干净的顺序效应。",
            "direct 从第一步就看到最终训练集，其最终训练集暴露量更大，不能把 direct-vs-incremental 全部归因于顺序。",
            "若 final_plus 长期阶段后路径 TV 仍显著高于各自 split_sampling_tv，才构成持久路径依赖的证据。",
        ],
    }
    save_json(analysis_dir / "path_summary.json", summary)
    return summary


# =============================================================================
# 可视化、归档与入口
# =============================================================================


def create_plots(
    cfg: EffectiveConfig,
    result_dir: Path,
    prior: dict[str, np.ndarray],
    direct_aggregates: dict[str, dict[str, np.ndarray]],
    direct_analysis: dict[str, Any],
    path_analysis: dict[str, Any],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] 未安装 matplotlib，跳过图片生成。")
        return []

    plots_dir = result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    metric_rows = direct_analysis["metric_rows"]
    endpoint_order = list(endpoint_names(cfg))
    fig, axis = plt.subplots(figsize=(12, 6))
    for state in cfg.static_states:
        by_endpoint = {
            row["endpoint"]: row
            for row in metric_rows
            if row["state"] == state.name
        }
        values = [
            by_endpoint[name]["tv_function"] if name in by_endpoint else np.nan
            for name in endpoint_order
        ]
        axis.plot(range(len(endpoint_order)), values, marker="o", label=state.name)
    axis.set_xticks(range(len(endpoint_order)), endpoint_order, rotation=28, ha="right")
    axis.set_ylabel("Function-level total variation")
    axis.set_title("SGD endpoints vs. static conditioned prior")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = plots_dir / "direct_function_tv.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    selected_state = next(
        (state for state in cfg.static_states if state.name == "single_x0_y0"),
        cfg.static_states[0],
    )
    selected_endpoint = (
        "post_fit_age_000000"
        if "post_fit_age_000000" in endpoint_order
        else endpoint_order[0]
    )
    endpoint_index = endpoint_order.index(selected_endpoint)
    valid = direct_aggregates[selected_state.name]["recorded"][endpoint_index]
    ids = direct_aggregates[selected_state.name]["function_ids"][endpoint_index, valid]
    empirical_counts, _ = ids_to_counts(ids)
    empirical = normalize_counts(empirical_counts)
    static = conditioned_distribution(prior["counts"], state_mask(selected_state))
    top = np.unique(
        np.concatenate(
            [np.argsort(static)[-12:], np.argsort(empirical)[-12:]]
        )
    )
    top = top[np.argsort(np.maximum(static[top], empirical[top]))[::-1]][:20]
    positions = np.arange(top.size)
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions - 0.2, static[top], width=0.4, label="Static P0(f|S)")
    axis.bar(positions + 0.2, empirical[top], width=0.4, label="SGD endpoints")
    axis.set_xticks(positions, [f"{int(function_id):03d}" for function_id in top])
    axis.set_xlabel("Boolean function ID (truth-table bit encoding)")
    axis.set_ylabel("Probability")
    axis.set_title(f"{selected_state.name} / {selected_endpoint}: top functions")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plots_dir / "single_sample_top_functions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    if path_analysis:
        pairwise = path_analysis["pairwise"]
        fig, axis = plt.subplots(figsize=(11, 6))
        pair_names = sorted(
            {f"{row['left_protocol']} vs {row['right_protocol']}" for row in pairwise}
        )
        path_endpoints = list(path_endpoint_names(cfg))
        for pair_name in pair_names:
            values = []
            for endpoint in path_endpoints:
                row = next(
                    item
                    for item in pairwise
                    if item["endpoint"] == endpoint
                    and f"{item['left_protocol']} vs {item['right_protocol']}" == pair_name
                )
                values.append(row["tv"])
            axis.plot(range(len(path_endpoints)), values, marker="o", label=pair_name)
        axis.set_xticks(range(len(path_endpoints)), path_endpoints, rotation=25, ha="right")
        axis.set_ylabel("Function-level TV between paths")
        axis.set_title("Path dependence after reaching the same final dataset")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = plots_dir / "path_pairwise_tv.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

    return written


def create_result_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(result_dir)
            # chunk 是可恢复中间件；aggregate 已包含同样原始数组，压缩包无需重复。
            if "chunks" in relative.parts:
                continue
            archive.write(path, arcname=(Path(result_dir.name) / relative).as_posix())
    temporary.replace(archive_path)
    return archive_path


def model_parameter_count(spec: ModelSpec) -> int:
    widths = [3] + [spec.hidden_size] * spec.hidden_layers + [1]
    return int(
        sum(in_width * out_width + out_width for in_width, out_width in zip(widths[:-1], widths[1:]))
    )


def print_key_results(
    cfg: EffectiveConfig,
    direct_analysis: dict[str, Any],
    path_analysis: dict[str, Any],
) -> None:
    print("\n=== 核心判别量 ===")
    preferred_endpoint = (
        f"post_fit_age_{max(cfg.post_fit_ages):06d}"
        if cfg.post_fit_ages
        else "last_step"
    )
    for state in cfg.static_states:
        candidates = [
            row
            for row in direct_analysis["metric_rows"]
            if row["state"] == state.name and row["endpoint"] == preferred_endpoint
        ]
        if not candidates:
            candidates = [
                row
                for row in direct_analysis["metric_rows"]
                if row["state"] == state.name and row["endpoint"] == "last_step"
            ]
        if not candidates:
            continue
        row = candidates[0]
        print(
            f"{state.name:16s} {row['endpoint']:24s} | "
            f"TV={row['tv_function']:.6f} | JS={row['js_bits_function']:.6f} bit | "
            f"prior噪声={row['prior_half_tv']:.6f} | "
            f"SGD噪声={float(row['sgd_half_tv'] or 0.0):.6f}"
        )
    if path_analysis:
        final_endpoint = f"final_plus_{max(cfg.path_final_extra_steps):06d}"
        print(f"\n路径实验最终点：{final_endpoint}")
        for row in path_analysis["pairwise"]:
            if row["endpoint"] == final_endpoint:
                print(
                    f"  {row['left_protocol']} vs {row['right_protocol']} | "
                    f"TV={row['tv']:.6f} | JS={row['js_bits']:.6f} bit"
                )
    print(
        "\n判断原则：观察到的逐函数 TV 若稳定高于 prior/SGD 两种抽样噪声，"
        "且路径差异在共同终态上长期不消失，就不能把 SGD 简化为静态先验上的条件抽样。"
    )


def main() -> None:
    cfg = get_effective_config()
    validate_config(cfg)
    result_dir = Path(cfg.result_root).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE='cuda'，但当前 PyTorch 无法使用 CUDA")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    torch.set_float32_matmul_precision("highest")

    config_payload = asdict(cfg)
    signature = stable_json_hash(config_payload)
    save_json(
        result_dir / "config.json",
        {
            **config_payload,
            "config_signature": signature,
            "resume_existing": bool(Config.RESUME_EXISTING),
        },
    )
    print("=== 3-bit Boolean：静态先验后验 vs SGD 终点分布 ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(
        f"模型：3 -> {cfg.model.hidden_size} x {cfg.model.hidden_layers} -> 1 | "
        f"tanh | params={model_parameter_count(cfg.model):,}"
    )
    print(
        f"prior={cfg.prior_samples:,} | direct/state={cfg.train_model_count:,} | "
        f"path/protocol={cfg.path_model_count:,} | signature={signature}"
    )
    print(f"结果目录：{result_dir}")

    total_started = time.perf_counter()
    prior = sample_initialization_prior(cfg, result_dir, signature)
    direct_aggregates = run_direct_experiments(cfg, result_dir, signature)
    direct_analysis = analyze_direct_experiments(
        cfg, result_dir, prior, direct_aggregates
    )
    path_aggregates = run_path_experiments(cfg, result_dir, signature)
    path_analysis = analyze_path_experiments(
        cfg, result_dir, prior, path_aggregates
    )
    plot_paths = create_plots(
        cfg,
        result_dir,
        prior,
        direct_aggregates,
        direct_analysis,
        path_analysis,
    )
    elapsed = time.perf_counter() - total_started
    save_json(
        result_dir / "summary.json",
        {
            "protocol_version": cfg.protocol_version,
            "config_signature": signature,
            "elapsed_seconds": elapsed,
            "direct_summary": direct_analysis["summary"],
            "path_summary": path_analysis,
            "plots": plot_paths,
        },
    )
    print_key_results(cfg, direct_analysis, path_analysis)
    print(f"\n总耗时：{elapsed:.1f}s")
    print(f"汇总：{result_dir / 'summary.json'}")
    if Config.CREATE_ZIP:
        archive = create_result_archive(result_dir)
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
