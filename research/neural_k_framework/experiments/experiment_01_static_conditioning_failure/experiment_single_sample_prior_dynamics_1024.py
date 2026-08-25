"""
3-bit Boolean 单样本训练：静态初始化先验与首次拟合后函数分布对照。

核心问题：

    P_train(f | S) 是否只是 P_init(f | f 与单样本 S 一致)？

这里完整输入空间只有 8 个点，硬阈值函数空间只有 256 个函数，因此无需用
复杂度直方图代替逐函数比较。实验使用早期 tiny Boolean 实验的原始网络：

    3 -> 1024 -> 1024 -> 1024 -> 1
    GELU + LayerNorm，Adam，无 dropout

实验同时保存三类证据：
1. 大量未训练网络在 8 个输入上的初始化函数先验；
2. 相同初始化 seed 在单样本训练前后的逐函数迁移；
3. 首次拟合后继续训练多个年龄时的完整 logits，而不只保存 0/1 阈值。

脚本完全自包含，兼容：
    python experiment_single_sample_prior_dynamics_1024.py
    %run experiment_single_sample_prior_dynamics_1024.py
    整个文件粘贴到 AutoDL Jupyter cell 直接运行

所有常用设置都在 Config 中；不使用 argparse、环境变量或同目录模块。
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
    RESULT_ROOT = BASE_DIR / "results_single_sample_prior_dynamics_1024"

    # 默认选择两个几何极端输入，并同时测试标签 0/1。
    # 若要穷举全部 16 个单样本条件，把 RUN_ALL_16_CONDITIONS 改为 True。
    SINGLE_SAMPLE_STATES = (
        (0, 0),  # 000 -> 0
        (0, 1),  # 000 -> 1
        (7, 0),  # 111 -> 0
        (7, 1),  # 111 -> 1
    )
    RUN_ALL_16_CONDITIONS = False

    # 对齐早期 tiny Boolean 实验：总共 3 个 1024 维隐藏层。
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS_AFTER_FIRST = 2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0

    # 初始化先验只做 forward，因此可以比训练 cohort 宽得多。
    # 1024 x 3 网络的初始化本身已经不小；4,096 个先验样本足够先看清
    # 256 个 hard function 的质量分布。需要更精细的尾部频率时可再调大。
    PRIOR_MODEL_COUNT = 4_096
    PRIOR_CHUNK_SIZE = 16
    PRIOR_SEED_BASE = 1_000_000

    # 每个单样本条件都复用同一组初始化 seed，便于配对比较条件差异。
    # 这些 seed 是 prior 的前 TRAIN_SEED_COUNT 个 seed，因此训练前函数也被
    # 精确包含在初始化先验样本中。
    TRAIN_SEED_COUNT = 128
    TRAIN_CHUNK_SIZE = 16
    TRAIN_SEED_BASE = PRIOR_SEED_BASE

    # 年龄从每个模型“首次 hard 拟合单样本”的 step 单独开始计算。
    # age=0 同时检查完整 8 点 hard function；age=1,2 只用于确认立即坍缩后
    # 没有短暂反转。这里不再做与当前问题无关的两万步长尾训练。
    POST_FIT_AGES = (0, 1, 2)
    FIT_CHECK_INTERVAL = 1
    MAX_PREFIT_STEPS = 100
    LOG_INTERVAL = 20

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESUME_EXISTING = True
    CREATE_ZIP = True

    # 在 notebook 中先改成 True，可快速检查完整流程和输出文件。
    SMOKE_TEST = False


PROTOCOL_VERSION = "single_sample_prior_dynamics_1024_v1"
INPUT_COUNT = 8
FUNCTION_COUNT = 256


@dataclass(frozen=True)
class StateSpec:
    name: str
    input_index: int
    target: int


@dataclass(frozen=True)
class ModelSpec:
    hidden_size: int
    hidden_layers_after_first: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class EffectiveConfig:
    protocol_version: str
    result_root: str
    states: tuple[StateSpec, ...]
    model: ModelSpec
    prior_model_count: int
    prior_chunk_size: int
    prior_seed_base: int
    train_seed_count: int
    train_chunk_size: int
    train_seed_base: int
    post_fit_ages: tuple[int, ...]
    fit_check_interval: int
    max_prefit_steps: int
    log_interval: int
    device: str
    allow_tf32: bool
    smoke_test: bool


def state_name(input_index: int, target: int) -> str:
    return f"x{input_index}_{input_index:03b}_to_{target}"


def get_effective_config() -> EffectiveConfig:
    raw_states = (
        tuple((input_index, target) for input_index in range(8) for target in (0, 1))
        if Config.RUN_ALL_16_CONDITIONS
        else tuple(Config.SINGLE_SAMPLE_STATES)
    )
    states = tuple(
        StateSpec(state_name(int(input_index), int(target)), int(input_index), int(target))
        for input_index, target in raw_states
    )
    if Config.SMOKE_TEST:
        return EffectiveConfig(
            protocol_version=PROTOCOL_VERSION,
            result_root=str(Config.RESULT_ROOT / "smoke"),
            states=states[:2],
            model=ModelSpec(64, 1, 1e-3, 0.0),
            prior_model_count=128,
            prior_chunk_size=32,
            prior_seed_base=int(Config.PRIOR_SEED_BASE),
            train_seed_count=8,
            train_chunk_size=4,
            train_seed_base=int(Config.TRAIN_SEED_BASE),
            post_fit_ages=(0, 1, 2),
            fit_check_interval=1,
            max_prefit_steps=100,
            log_interval=20,
            device=str(Config.DEVICE),
            allow_tf32=bool(Config.ALLOW_TF32),
            smoke_test=True,
        )
    return EffectiveConfig(
        protocol_version=PROTOCOL_VERSION,
        result_root=str(Config.RESULT_ROOT),
        states=states,
        model=ModelSpec(
            int(Config.HIDDEN_SIZE),
            int(Config.HIDDEN_LAYERS_AFTER_FIRST),
            float(Config.LEARNING_RATE),
            float(Config.WEIGHT_DECAY),
        ),
        prior_model_count=int(Config.PRIOR_MODEL_COUNT),
        prior_chunk_size=int(Config.PRIOR_CHUNK_SIZE),
        prior_seed_base=int(Config.PRIOR_SEED_BASE),
        train_seed_count=int(Config.TRAIN_SEED_COUNT),
        train_chunk_size=int(Config.TRAIN_CHUNK_SIZE),
        train_seed_base=int(Config.TRAIN_SEED_BASE),
        post_fit_ages=tuple(int(value) for value in Config.POST_FIT_AGES),
        fit_check_interval=int(Config.FIT_CHECK_INTERVAL),
        max_prefit_steps=int(Config.MAX_PREFIT_STEPS),
        log_interval=int(Config.LOG_INTERVAL),
        device=str(Config.DEVICE),
        allow_tf32=bool(Config.ALLOW_TF32),
        smoke_test=False,
    )


def validate_config(cfg: EffectiveConfig) -> None:
    if not cfg.states:
        raise ValueError("至少需要一个单样本条件")
    state_keys = [(state.input_index, state.target) for state in cfg.states]
    if len(set(state_keys)) != len(state_keys):
        raise ValueError("SINGLE_SAMPLE_STATES 中存在重复条件")
    for state in cfg.states:
        if state.input_index not in range(8) or state.target not in (0, 1):
            raise ValueError(f"非法单样本条件：{state}")
    if cfg.model.hidden_size <= 0 or cfg.model.hidden_layers_after_first < 0:
        raise ValueError("网络宽度和隐藏层数必须合法")
    if cfg.prior_model_count <= 0 or cfg.train_seed_count <= 0:
        raise ValueError("采样数必须为正数")
    if cfg.prior_chunk_size <= 0 or cfg.train_chunk_size <= 0:
        raise ValueError("chunk size 必须为正数")
    if cfg.train_seed_base < cfg.prior_seed_base:
        raise ValueError("训练 seed 必须位于初始化先验 seed 区间内")
    train_end = cfg.train_seed_base + cfg.train_seed_count
    prior_end = cfg.prior_seed_base + cfg.prior_model_count
    if train_end > prior_end:
        raise ValueError("训练 seed 区间必须完整包含在初始化先验 seed 区间内")
    if not cfg.post_fit_ages or cfg.post_fit_ages[0] != 0:
        raise ValueError("POST_FIT_AGES 必须从 0 开始")
    if tuple(sorted(set(cfg.post_fit_ages))) != cfg.post_fit_ages:
        raise ValueError("POST_FIT_AGES 必须严格递增且不重复")


# =============================================================================
# 通用工具
# =============================================================================


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


def chunked(values: tuple[int, ...], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def all_inputs(device: torch.device | None = None) -> torch.Tensor:
    values = [[(index >> 2) & 1, (index >> 1) & 1, index & 1] for index in range(8)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def function_ids_from_logits(logits: torch.Tensor) -> torch.Tensor:
    powers = 2 ** torch.arange(INPUT_COUNT, device=logits.device, dtype=torch.int64)
    return ((logits >= 0).to(torch.int64) * powers[None, :]).sum(dim=1)


def truth_table_text(function_id: int) -> str:
    return "".join(str((int(function_id) >> index) & 1) for index in range(8))


def state_compatible_mask(state: StateSpec) -> np.ndarray:
    ids = np.arange(FUNCTION_COUNT, dtype=np.int64)
    return (((ids >> state.input_index) & 1) == state.target)


# =============================================================================
# 与早期实验一致的独立 seed MLP
# =============================================================================


class EnsembleLinear(nn.Module):
    """首维打包多个模型，但每个模型拥有完全独立的 Linear 参数。"""

    def __init__(
        self,
        seeds: tuple[int, ...],
        in_features: int,
        out_features: int,
        generators: list[torch.Generator],
    ) -> None:
        super().__init__()
        count = len(seeds)
        self.weight = nn.Parameter(torch.empty(count, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(count, out_features))
        bound = 1.0 / math.sqrt(in_features)
        with torch.no_grad():
            for index, generator in enumerate(generators):
                nn.init.kaiming_uniform_(
                    self.weight[index], a=math.sqrt(5), generator=generator
                )
                self.bias[index].uniform_(-bound, bound, generator=generator)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.bmm(inputs, self.weight.transpose(1, 2)) + self.bias[:, None, :]


class EnsembleLayerNorm(nn.Module):
    def __init__(self, ensemble_size: int, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, width))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, width))
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[:, None, :] + self.bias[:, None, :]


class BatchedSeedMLP(nn.Module):
    def __init__(self, seeds: tuple[int, ...], spec: ModelSpec) -> None:
        super().__init__()
        generators: list[torch.Generator] = []
        for seed in seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            generators.append(generator)

        blocks: list[nn.Module] = []
        width = 3
        total_hidden_layers = 1 + spec.hidden_layers_after_first
        for _ in range(total_hidden_layers):
            blocks.extend(
                [
                    EnsembleLinear(seeds, width, spec.hidden_size, generators),
                    nn.GELU(),
                    EnsembleLayerNorm(len(seeds), spec.hidden_size),
                ]
            )
            width = spec.hidden_size
        blocks.append(EnsembleLinear(seeds, width, 1, generators))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


@torch.inference_mode()
def evaluate_model(model: nn.Module, probe_inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(probe_inputs).squeeze(-1)
    function_ids = function_ids_from_logits(logits)
    return (
        function_ids.cpu().numpy().astype(np.uint16),
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
            arrays = load_npz(output)
            print(f"[prior] 复用已有 {arrays['seeds'].size:,} 个初始化样本")
            return arrays

    device = torch.device(cfg.device)
    probe = all_inputs(device)
    seeds = np.arange(
        cfg.prior_seed_base,
        cfg.prior_seed_base + cfg.prior_model_count,
        dtype=np.int64,
    )
    function_ids = np.empty(cfg.prior_model_count, dtype=np.uint16)
    logits = np.empty((cfg.prior_model_count, INPUT_COUNT), dtype=np.float32)
    seed_tuple = tuple(int(value) for value in seeds.tolist())
    chunks = list(chunked(seed_tuple, cfg.prior_chunk_size))
    started = time.perf_counter()
    offset = 0

    for chunk_index, seed_chunk in enumerate(chunks, start=1):
        model = BatchedSeedMLP(seed_chunk, cfg.model).to(device)
        probe_batch = probe[None, :, :].expand(len(seed_chunk), -1, -1)
        chunk_ids, chunk_logits = evaluate_model(model, probe_batch)
        end = offset + len(seed_chunk)
        function_ids[offset:end] = chunk_ids
        logits[offset:end] = chunk_logits
        offset = end
        del model, probe_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if chunk_index % max(1, len(chunks) // 20) == 0 or chunk_index == len(chunks):
            elapsed = time.perf_counter() - started
            print(
                f"[prior] {offset:,}/{cfg.prior_model_count:,} | "
                f"{offset / max(elapsed, 1e-9):,.1f} init/s | "
                f"已见函数={np.unique(function_ids[:offset]).size}/256"
            )

    counts = np.bincount(function_ids.astype(np.int64), minlength=FUNCTION_COUNT).astype(np.int64)
    arrays = {
        "seeds": seeds,
        "function_ids": function_ids,
        "probe_logits": logits,
        "counts": counts,
    }
    np.savez_compressed(output, **arrays)
    save_json(
        metadata_path,
        {
            "config_signature": signature,
            "samples": int(cfg.prior_model_count),
            "observed_functions": int(np.count_nonzero(counts)),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    return arrays


# =============================================================================
# 单样本训练
# =============================================================================


def train_seed_chunk(
    cfg: EffectiveConfig,
    state: StateSpec,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(cfg.device)
    probe = all_inputs(device)
    count = len(seeds)
    model = BatchedSeedMLP(seeds, cfg.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    probe_inputs = probe[None, :, :].expand(count, -1, -1)
    train_inputs = probe[state.input_index][None, None, :].expand(count, 1, -1)
    train_targets = torch.full(
        (count, 1),
        float(state.target),
        dtype=torch.float32,
        device=device,
    )
    signed_targets = train_targets.mul(2.0).sub(1.0)

    ages = cfg.post_fit_ages
    output_ages = np.asarray((-1, *ages), dtype=np.int64)
    age_to_position = {age: position + 1 for position, age in enumerate(ages)}
    snapshot_count = output_ages.size
    snapshot_logits = np.full((snapshot_count, count, INPUT_COUNT), np.nan, dtype=np.float32)
    snapshot_ids = np.zeros((snapshot_count, count), dtype=np.uint16)
    snapshot_train_loss = np.full((snapshot_count, count), np.nan, dtype=np.float32)
    snapshot_train_margin = np.full((snapshot_count, count), np.nan, dtype=np.float32)
    absolute_steps = np.full((snapshot_count, count), -1, dtype=np.int64)
    recorded = np.zeros((snapshot_count, count), dtype=bool)
    first_fit_steps = np.full(count, -1, dtype=np.int64)
    started = time.perf_counter()

    @torch.inference_mode()
    def evaluate() -> dict[str, np.ndarray]:
        function_ids, logits = evaluate_model(model, probe_inputs)
        selected = torch.from_numpy(logits[:, state.input_index]).to(device)
        margins = selected * signed_targets[:, 0]
        losses = F.softplus(-margins)
        return {
            "function_ids": function_ids,
            "probe_logits": logits,
            "train_margin": margins.cpu().numpy().astype(np.float32),
            "train_loss": losses.cpu().numpy().astype(np.float32),
            "train_exact": (margins >= 0).cpu().numpy(),
        }

    def assign(step: int, snapshot: dict[str, np.ndarray], position: int, mask: np.ndarray) -> None:
        snapshot_ids[position, mask] = snapshot["function_ids"][mask]
        snapshot_logits[position, mask] = snapshot["probe_logits"][mask]
        snapshot_train_loss[position, mask] = snapshot["train_loss"][mask]
        snapshot_train_margin[position, mask] = snapshot["train_margin"][mask]
        absolute_steps[position, mask] = int(step)
        recorded[position, mask] = True

    initial = evaluate()
    assign(0, initial, 0, np.ones(count, dtype=bool))
    max_age = max(ages)
    final_step = cfg.max_prefit_steps + max_age
    latest_mean_loss = float("nan")

    for step in range(1, final_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs).squeeze(-1)
        loss_by_model = F.binary_cross_entropy_with_logits(
            logits,
            train_targets,
            reduction="none",
        ).mean(dim=1)
        # 用 sum 保证每个独立模型的梯度尺度与单独训练完全一致。
        loss_by_model.sum().backward()
        optimizer.step()
        latest_mean_loss = float(loss_by_model.detach().mean().item())

        checked: dict[str, np.ndarray] | None = None
        if np.any(first_fit_steps < 0) and (
            step == 1 or step % cfg.fit_check_interval == 0
        ):
            checked = evaluate()
            newly_fitted = (first_fit_steps < 0) & checked["train_exact"]
            first_fit_steps[newly_fitted] = int(step)

        assignments: list[tuple[int, np.ndarray]] = []
        fitted = first_fit_steps >= 0
        if np.any(fitted):
            current_ages = step - first_fit_steps
            for age in ages:
                position = age_to_position[age]
                due = fitted & (current_ages == age) & (~recorded[position])
                if np.any(due):
                    assignments.append((position, due))
        if assignments:
            if checked is None:
                checked = evaluate()
            for position, mask in assignments:
                assign(step, checked, position, mask)

        if bool(recorded[1:].all()):
            break
        if step >= cfg.max_prefit_steps and np.any(first_fit_steps < 0):
            break
        if step % cfg.log_interval == 0:
            fitted_count = int(np.count_nonzero(first_fit_steps >= 0))
            print(
                f"      step={step:,}/{final_step:,} | fitted={fitted_count}/{count} | "
                f"mean_loss={latest_mean_loss:.9e} | elapsed={time.perf_counter() - started:.1f}s"
            )

    if not bool(recorded.all()):
        missing = int(np.count_nonzero(~recorded))
        raise RuntimeError(
            f"{state.name} seeds={seeds[0]}..{seeds[-1]} 未完成，缺少 {missing} 个快照；"
            f"first_fit={first_fit_steps.tolist()}"
        )

    arrays = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "snapshot_ages": output_ages,
        "function_ids": snapshot_ids,
        "probe_logits": snapshot_logits,
        "train_loss": snapshot_train_loss,
        "train_margin": snapshot_train_margin,
        "absolute_steps": absolute_steps,
        "first_fit_steps": first_fit_steps,
    }
    metadata = {
        "state": asdict(state),
        "seeds": list(seeds),
        "completed": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "slowest_first_fit_step": int(first_fit_steps.max()),
    }
    return metadata, arrays


def run_training_state(
    cfg: EffectiveConfig,
    state: StateSpec,
    result_dir: Path,
    signature: str,
) -> dict[str, np.ndarray]:
    state_dir = result_dir / "training" / state.name
    chunks_dir = state_dir / "chunks"
    aggregate_path = state_dir / "samples.npz"
    metadata_path = state_dir / "metadata.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if Config.RESUME_EXISTING and aggregate_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == signature:
            print(f"  [skip] {state.name}：复用已有结果")
            return load_npz(aggregate_path)

    seed_values = tuple(
        range(cfg.train_seed_base, cfg.train_seed_base + cfg.train_seed_count)
    )
    chunks = list(chunked(seed_values, cfg.train_chunk_size))
    results: list[dict[str, np.ndarray]] = []
    chunk_metadata: list[dict[str, Any]] = []
    state_started = time.perf_counter()
    for chunk_index, seeds in enumerate(chunks, start=1):
        tag = f"seed_{seeds[0]}_{seeds[-1]}"
        chunk_path = chunks_dir / f"{tag}.npz"
        chunk_meta_path = chunks_dir / f"{tag}.json"
        print(
            f"    chunk {chunk_index}/{len(chunks)} | seeds={seeds[0]}..{seeds[-1]}"
        )
        if Config.RESUME_EXISTING and chunk_path.exists() and chunk_meta_path.exists():
            meta = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
            if meta.get("config_signature") == signature:
                print("      [skip chunk] 复用已有结果")
                results.append(load_npz(chunk_path))
                chunk_metadata.append(meta)
                continue
        meta, arrays = train_seed_chunk(cfg, state, seeds)
        meta["config_signature"] = signature
        np.savez_compressed(chunk_path, **arrays)
        save_json(chunk_meta_path, meta)
        results.append(arrays)
        chunk_metadata.append(meta)

    reference_ages = results[0]["snapshot_ages"]
    aggregate = {
        "seeds": np.concatenate([item["seeds"] for item in results]),
        "snapshot_ages": reference_ages,
        "function_ids": np.concatenate([item["function_ids"] for item in results], axis=1),
        "probe_logits": np.concatenate([item["probe_logits"] for item in results], axis=1),
        "train_loss": np.concatenate([item["train_loss"] for item in results], axis=1),
        "train_margin": np.concatenate([item["train_margin"] for item in results], axis=1),
        "absolute_steps": np.concatenate([item["absolute_steps"] for item in results], axis=1),
        "first_fit_steps": np.concatenate([item["first_fit_steps"] for item in results]),
    }
    np.savez_compressed(aggregate_path, **aggregate)
    save_json(
        metadata_path,
        {
            "config_signature": signature,
            "state": asdict(state),
            "model_count": int(aggregate["seeds"].size),
            "snapshot_ages": aggregate["snapshot_ages"].tolist(),
            "elapsed_seconds": float(time.perf_counter() - state_started),
            "slowest_first_fit_step": int(aggregate["first_fit_steps"].max()),
            "chunks": chunk_metadata,
        },
    )
    return aggregate


# =============================================================================
# 分析
# =============================================================================


def normalized_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size <= 1:
        return 0.0
    return float(-(positive * np.log2(positive)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float((left[mask] * np.log2(left[mask] / right[mask])).sum())

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def analyze_results(
    cfg: EffectiveConfig,
    result_dir: Path,
    prior: dict[str, np.ndarray],
    state_results: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    prior_counts = prior["counts"].astype(np.float64)
    prior_probabilities = prior_counts / prior_counts.sum()
    prior_rows = []
    for function_id in range(FUNCTION_COUNT):
        prior_rows.append(
            {
                "function_id": function_id,
                "truth_table_x0_to_x7": truth_table_text(function_id),
                "count": int(prior_counts[function_id]),
                "probability": float(prior_probabilities[function_id]),
            }
        )
    write_csv(result_dir / "initialization_prior_functions.csv", prior_rows)

    summary_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    final_summary: dict[str, Any] = {}

    for state in cfg.states:
        arrays = state_results[state.name]
        ages = arrays["snapshot_ages"]
        compatible = state_compatible_mask(state)
        conditioned_counts = prior_counts * compatible
        conditioned_total = conditioned_counts.sum()
        if conditioned_total <= 0:
            raise RuntimeError(f"初始化先验中没有与 {state.name} 相容的样本")
        static_posterior = conditioned_counts / conditioned_total
        constant_id = 255 if state.target == 1 else 0
        initial_ids = arrays["function_ids"][0].astype(np.int64)

        for age_position, age in enumerate(ages):
            ids = arrays["function_ids"][age_position].astype(np.int64)
            counts = np.bincount(ids, minlength=FUNCTION_COUNT).astype(np.float64)
            probabilities = counts / counts.sum()
            modal_id = int(np.argmax(counts))
            logits = arrays["probe_logits"][age_position]
            one_probabilities = (logits >= 0).mean(axis=0)
            row = {
                "state": state.name,
                "input_index": state.input_index,
                "input_bits": f"{state.input_index:03b}",
                "target": state.target,
                "snapshot_age": int(age),
                "models": int(ids.size),
                "unique_functions": int(np.count_nonzero(counts)),
                "modal_function_id": modal_id,
                "modal_truth_table": truth_table_text(modal_id),
                "modal_probability": float(probabilities[modal_id]),
                "constant_target_probability": float(probabilities[constant_id]),
                "static_constant_target_probability": float(static_posterior[constant_id]),
                "function_entropy_bits": normalized_entropy(probabilities),
                "static_function_entropy_bits": normalized_entropy(static_posterior),
                "tv_from_static_posterior": float(
                    0.5 * np.abs(probabilities - static_posterior).sum()
                ),
                "js_from_static_posterior_bits": js_divergence(
                    probabilities, static_posterior
                ),
                "mean_train_loss": float(arrays["train_loss"][age_position].mean()),
                "mean_train_margin": float(arrays["train_margin"][age_position].mean()),
                "mean_unseen_abs_logit": float(
                    np.abs(np.delete(logits, state.input_index, axis=1)).mean()
                ),
            }
            summary_rows.append(row)

            for input_index in range(INPUT_COUNT):
                input_rows.append(
                    {
                        "state": state.name,
                        "snapshot_age": int(age),
                        "probe_input_index": input_index,
                        "probe_input_bits": f"{input_index:03b}",
                        "is_training_input": int(input_index == state.input_index),
                        "hard_one_probability": float(one_probabilities[input_index]),
                        "mean_logit": float(logits[:, input_index].mean()),
                        "std_logit": float(logits[:, input_index].std()),
                        "mean_abs_logit": float(np.abs(logits[:, input_index]).mean()),
                    }
                )

            if age >= 0:
                pair_ids = initial_ids * FUNCTION_COUNT + ids
                pair_counts = np.bincount(
                    pair_ids, minlength=FUNCTION_COUNT * FUNCTION_COUNT
                ).reshape(FUNCTION_COUNT, FUNCTION_COUNT)
                for initial_id, final_id in np.argwhere(pair_counts > 0):
                    transition_rows.append(
                        {
                            "state": state.name,
                            "snapshot_age": int(age),
                            "initial_function_id": int(initial_id),
                            "initial_truth_table": truth_table_text(int(initial_id)),
                            "trained_function_id": int(final_id),
                            "trained_truth_table": truth_table_text(int(final_id)),
                            "count": int(pair_counts[initial_id, final_id]),
                        }
                    )

        final_row = summary_rows[-1]
        final_summary[state.name] = dict(final_row)
        print(
            f"[结果] {state.name} | age={final_row['snapshot_age']:,} | "
            f"constant={final_row['constant_target_probability']:.6%} "
            f"(static={final_row['static_constant_target_probability']:.6%}) | "
            f"modal={final_row['modal_truth_table']}:{final_row['modal_probability']:.6%} | "
            f"TV={final_row['tv_from_static_posterior']:.6f}"
        )

    write_csv(result_dir / "training_summary_by_age.csv", summary_rows)
    write_csv(result_dir / "per_input_logit_summary.csv", input_rows)
    write_csv(result_dir / "paired_function_transitions.csv", transition_rows)
    summary = {
        "protocol_version": cfg.protocol_version,
        "model": asdict(cfg.model),
        "prior_samples": int(prior["seeds"].size),
        "prior_observed_functions": int(np.count_nonzero(prior_counts)),
        "conditions": len(cfg.states),
        "models_per_condition": cfg.train_seed_count,
        "final_post_fit_age": int(max(cfg.post_fit_ages)),
        "final_results": final_summary,
        "interpretation_guardrail": (
            "TV/JS 只检验硬阈值函数层面的静态条件先验；完整 logits 已另行保存，"
            "不能把硬函数相同误写成连续函数完全相同。"
        ),
    }
    save_json(result_dir / "summary.json", summary)
    return summary


def create_archive(result_dir: Path) -> Path:
    archive = result_dir.parent / f"{result_dir.name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and "chunks" not in path.parts:
                handle.write(path, path.relative_to(result_dir.parent))
    return archive


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    cfg = get_effective_config()
    validate_config(cfg)
    result_dir = Path(cfg.result_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(cfg)
    signature = stable_json_hash(config_payload)
    save_json(
        result_dir / "config.json",
        {"config_signature": signature, **config_payload},
    )

    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    device = torch.device(cfg.device)
    print("=== 3-bit Boolean 单样本：静态先验 vs 首次拟合动力学 ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    hidden_layers = 1 + cfg.model.hidden_layers_after_first
    print(
        f"网络：3 -> {cfg.model.hidden_size} x {hidden_layers} -> 1 | "
        "GELU + LayerNorm | Adam"
    )
    print(
        f"先验：{cfg.prior_model_count:,} models | "
        f"训练：{len(cfg.states)} conditions x {cfg.train_seed_count:,} seeds | "
        f"最长 post-fit age={max(cfg.post_fit_ages):,}"
    )
    print(f"结果目录：{result_dir}")

    overall_started = time.perf_counter()
    prior = sample_initialization_prior(cfg, result_dir, signature)
    state_results: dict[str, dict[str, np.ndarray]] = {}
    for state_index, state in enumerate(cfg.states, start=1):
        print(
            f"\n--- condition {state_index}/{len(cfg.states)} | "
            f"{state.input_index:03b} -> {state.target} ---"
        )
        state_results[state.name] = run_training_state(
            cfg, state, result_dir, signature
        )

    summary = analyze_results(cfg, result_dir, prior, state_results)
    elapsed = time.perf_counter() - overall_started
    print("\n=== 实验完成 ===")
    print(f"总耗时：{elapsed:.1f}s")
    print(f"汇总：{result_dir / 'summary.json'}")
    print(f"逐年龄统计：{result_dir / 'training_summary_by_age.csv'}")
    print(f"原始 logits：{result_dir / 'training'}/*/samples.npz")
    if Config.CREATE_ZIP:
        archive = create_archive(result_dir)
        print(f"下载压缩包：{archive}")
    print(
        "最终常量率："
        + ", ".join(
            f"{name}={values['constant_target_probability']:.3%}"
            for name, values in summary["final_results"].items()
        )
    )


if __name__ == "__main__":
    main()
