"""用完整真值表比较不同规则的静态 low-loss 参数质量。

对每个目标函数 f，测量初始化参数立方体均匀测度下的累计质量：

    V_f(epsilon) = P_theta[L_full(f)(theta) <= epsilon]

脚本使用不含 loss 梯度的 constrained SMC / subset simulation。每个目标拥有
独立 SMC 副本、checkpoint 和体积估计；最后自动比较 parity1--parity4、
majority3 与固定随机平衡函数的 log-volume 曲线，并计算进入 hard-exact
区域后的条件保留率 V_f(epsilon) / V_f(0.04)。

这测量的是静态初始化测度，不是 SGD 到达概率。脚本另外用直接 prior 采样校准
高 loss 重叠区，但不会把 SMC 体积直接解释成优化器终点分布。
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 4
    WIDTH = 16
    HIDDEN_LAYERS = 2
    TARGET_NAMES = (
        "parity1",
        "parity2",
        "parity3",
        "parity4",
        "majority3",
        "random_balanced",
    )
    RANDOM_TARGET_SEED = 2026082401

    REPLICAS = 8
    PARTICLES_PER_REPLICA = 4_096
    SURVIVAL_QUANTILE = 0.5
    TARGET_THRESHOLDS = (
        0.68,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.07,
        0.05,
        0.04,
        0.03,
        0.02,
    )
    HARD_EXACT_REFERENCE_THRESHOLD = 0.04
    MAX_LEVELS_PER_TARGET = 2_000
    MIN_LEVEL_DECREMENT = 1e-7

    ADAPT_SWEEPS = 8
    MUTATION_SWEEPS = 24
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.050, 0.030, 0.050, 0.015)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.30
    REFRESH_PROBABILITY = 0.02
    LOSS_TOLERANCE = 1e-7

    PRIOR_CALIBRATION_SAMPLES = 4_194_304
    PRIOR_CALIBRATION_BATCH = 8_192
    PRIOR_CALIBRATION_THRESHOLDS = (0.68, 0.65, 0.60)

    EVAL_MICRO_BATCH = 8_192
    CHECKPOINT_EVERY_LEVELS = 10
    LOG_EVERY_LEVELS = 10
    TOP_FUNCTIONS = 8

    PRIOR_SEED = 2026082410
    RESAMPLE_SEED = 2026082420
    MUTATION_SEED = 2026082430
    CALIBRATION_SEED = 2026082440
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_full_truth_rule_volume_smc")
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class TargetSpec:
    target_index: int
    name: str
    formula: str
    family: str
    outputs: tuple[int, ...]
    function_id: int
    function_hex: str
    positive_count: int
    essential_variables: tuple[int, ...]
    anf_degree: int
    anf_term_count: int
    anf_literal_count: int


@dataclass
class SMCState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume_fraction: torch.Tensor
    proposal_scales: list[float]
    current_threshold: float
    threshold_index: int
    level: int
    level_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    replica_rows: list[dict[str, Any]]
    top_rows: list[dict[str, Any]]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.TARGET_NAMES = ("parity1", "parity2")
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 128
    Config.TARGET_THRESHOLDS = (0.72, 0.70, 0.68)
    Config.HARD_EXACT_REFERENCE_THRESHOLD = 0.68
    Config.MAX_LEVELS_PER_TARGET = 12
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 2
    Config.PRIOR_CALIBRATION_SAMPLES = 4_096
    Config.PRIOR_CALIBRATION_BATCH = 256
    Config.PRIOR_CALIBRATION_THRESHOLDS = (0.72, 0.70)
    Config.EVAL_MICRO_BATCH = 256
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.TOP_FUNCTIONS = 4
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_full_truth_rule_volume_smc"
    )
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True
    Config.PACKAGE_RESULTS = False


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


def validate_config() -> None:
    if Config.INPUT_BITS != 4 or Config.HIDDEN_LAYERS != 2:
        raise ValueError("当前实现固定为4-bit输入、两个隐藏层。")
    thresholds = tuple(float(value) for value in Config.TARGET_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("TARGET_THRESHOLDS必须严格从高到低且不重复。")
    if not 0 < Config.SURVIVAL_QUANTILE < 1:
        raise ValueError("SURVIVAL_QUANTILE必须在(0,1)内。")
    if Config.REPLICAS < 2:
        raise ValueError("至少需要两个独立副本。")
    if len(Config.INITIAL_PROPOSAL_SCALES) != 4:
        raise ValueError("三个参数块和全参数块需要四个proposal scale。")
    if Config.HARD_EXACT_REFERENCE_THRESHOLD not in Config.TARGET_THRESHOLDS:
        raise ValueError("hard-exact参考阈值必须位于TARGET_THRESHOLDS。")
    hard_bound = math.log(2.0) / (2 ** Config.INPUT_BITS)
    if (
        not Config.SMOKE_TEST
        and Config.HARD_EXACT_REFERENCE_THRESHOLD >= hard_bound
    ):
        raise ValueError(
            "hard-exact参考阈值必须严格低于ln(2)/16。"
        )


def immutable_config_keys() -> tuple[str, ...]:
    return (
        "INPUT_BITS",
        "WIDTH",
        "HIDDEN_LAYERS",
        "TARGET_NAMES",
        "RANDOM_TARGET_SEED",
        "REPLICAS",
        "PARTICLES_PER_REPLICA",
        "SURVIVAL_QUANTILE",
        "ADAPT_SWEEPS",
        "MUTATION_SWEEPS",
        "INITIAL_PROPOSAL_SCALES",
        "REFRESH_PROBABILITY",
        "PRIOR_SEED",
        "RESAMPLE_SEED",
        "MUTATION_SEED",
    )


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    if output.exists() and not Config.RESUME:
        output = output.parent / (
            output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
        )
    output.mkdir(parents=True, exist_ok=True)
    saved_path = output / "config.json"
    current = config_dict()
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        for key in immutable_config_keys():
            if saved.get(key) != current.get(key):
                raise RuntimeError(f"已有结果的配置与当前不一致：{key}")
        saved_thresholds = list(saved.get("TARGET_THRESHOLDS", []))
        current_thresholds = list(current["TARGET_THRESHOLDS"])
        if current_thresholds[:len(saved_thresholds)] != saved_thresholds:
            raise RuntimeError("新阈值序列必须以已有TARGET_THRESHOLDS为前缀。")
    write_json(saved_path, current)
    (output / "targets").mkdir(exist_ok=True)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def outputs_to_function_id(outputs: np.ndarray) -> int:
    bits = np.asarray(outputs, dtype=np.uint64).reshape(-1)
    powers = np.left_shift(np.uint64(1), np.arange(16, dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def function_bits(function_id: int) -> str:
    return "".join(str((function_id >> index) & 1) for index in range(16))


def anf_metrics(outputs: np.ndarray) -> tuple[int, int, int]:
    coefficients = np.asarray(outputs, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(16):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.asarray(
        [int(term).bit_count() for term in terms], dtype=np.int64
    )
    return (
        int(degrees.max()) if len(terms) else 0,
        int(len(terms)),
        int(degrees.sum()),
    )


def essential_variables(outputs: np.ndarray) -> tuple[int, ...]:
    values = np.arange(16, dtype=np.int64)
    result: list[int] = []
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        if np.any(outputs[base] != outputs[base | (1 << bit)]):
            result.append(Config.INPUT_BITS - 1 - bit)
    return tuple(sorted(result))


def build_targets() -> list[TargetSpec]:
    inputs = truth_table_inputs()
    raw: dict[str, tuple[np.ndarray, str, str]] = {}
    for count in range(1, 5):
        outputs = np.bitwise_xor.reduce(inputs[:, :count], axis=1).astype(np.uint8)
        raw[f"parity{count}"] = (
            outputs,
            " XOR ".join(f"x{index}" for index in range(count)),
            "nested_parity",
        )
    majority = (inputs[:, :3].sum(axis=1) >= 2).astype(np.uint8)
    raw["majority3"] = (
        majority,
        "(x0+x1+x2)>=2",
        "linearly_separable_control",
    )
    rng = np.random.default_rng(Config.RANDOM_TARGET_SEED)
    random_attempt = 0
    while True:
        random_outputs = np.zeros(16, dtype=np.uint8)
        random_outputs[rng.choice(16, size=8, replace=False)] = 1
        degree, terms, literals = anf_metrics(random_outputs)
        variables = essential_variables(random_outputs)
        if (
            len(variables) == 4
            and terms >= 12
            and literals >= 22
        ):
            break
        random_attempt += 1
    raw["random_balanced"] = (
        random_outputs,
        "first balanced truth table with all 4 variables, ANF terms>=12 and "
        f"literals>=22; seed={Config.RANDOM_TARGET_SEED}, attempt={random_attempt}",
        "random_balanced_control",
    )

    targets: list[TargetSpec] = []
    for target_index, name in enumerate(Config.TARGET_NAMES):
        if name not in raw:
            raise ValueError(f"未知目标函数：{name}")
        outputs, formula, family = raw[name]
        if int(outputs.sum()) != 8:
            raise AssertionError(f"目标{name}不是严格8/8平衡。")
        degree, terms, literals = anf_metrics(outputs)
        function_id = outputs_to_function_id(outputs)
        targets.append(TargetSpec(
            target_index=target_index,
            name=name,
            formula=formula,
            family=family,
            outputs=tuple(map(int, outputs)),
            function_id=function_id,
            function_hex=f"0x{function_id:04X}",
            positive_count=int(outputs.sum()),
            essential_variables=essential_variables(outputs),
            anf_degree=degree,
            anf_term_count=terms,
            anf_literal_count=literals,
        ))
    return targets


def parameter_blocks(width: int) -> tuple[list[ParameterBlock], int]:
    cursor = 0
    blocks: list[ParameterBlock] = []
    first_size = width * Config.INPUT_BITS + width
    blocks.append(ParameterBlock("first_layer", cursor, cursor + first_size))
    cursor += first_size
    middle_size = width * width + width
    blocks.append(ParameterBlock("middle_layer", cursor, cursor + middle_size))
    cursor += middle_size
    output_size = width + 1
    blocks.append(ParameterBlock("output_layer", cursor, cursor + output_size))
    cursor += output_size
    blocks.append(ParameterBlock("all_parameters", 0, cursor))
    return blocks, cursor


def forward_logits(normalized: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    count = normalized.shape[0]
    width = Config.WIDTH
    cursor = 0
    first_weight_size = width * Config.INPUT_BITS
    first_weight = normalized[:, cursor:cursor + first_weight_size].reshape(
        count, width, Config.INPUT_BITS
    ) * (1.0 / math.sqrt(Config.INPUT_BITS))
    cursor += first_weight_size
    first_bias = normalized[:, cursor:cursor + width] * (
        1.0 / math.sqrt(Config.INPUT_BITS)
    )
    cursor += width
    middle_weight_size = width * width
    middle_weight = normalized[:, cursor:cursor + middle_weight_size].reshape(
        count, width, width
    ) * (1.0 / math.sqrt(width))
    cursor += middle_weight_size
    middle_bias = normalized[:, cursor:cursor + width] * (
        1.0 / math.sqrt(width)
    )
    cursor += width
    output_weight = normalized[:, cursor:cursor + width].reshape(
        count, 1, width
    ) * (1.0 / math.sqrt(width))
    cursor += width
    output_bias = normalized[:, cursor:cursor + 1] * (
        1.0 / math.sqrt(width)
    )
    hidden = inputs[None].expand(count, -1, -1)
    hidden = torch.tanh(
        torch.bmm(hidden, first_weight.transpose(1, 2))
        + first_bias[:, None]
    )
    hidden = torch.tanh(
        torch.bmm(hidden, middle_weight.transpose(1, 2))
        + middle_bias[:, None]
    )
    return (
        torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
        + output_bias
    )


@torch.no_grad()
def evaluate_losses(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces: list[torch.Tensor] = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(flat[start:start + Config.EVAL_MICRO_BATCH], inputs)
        local_targets = targets[None].expand_as(logits)
        pieces.append(F.binary_cross_entropy_with_logits(
            logits, local_targets, reduction="none"
        ).mean(dim=1))
    return torch.cat(pieces).reshape(particles.shape[:-1])


@torch.no_grad()
def evaluate_function_ids(
    particles: torch.Tensor,
    inputs: torch.Tensor,
) -> np.ndarray:
    flat = particles.reshape(-1, particles.shape[-1])
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=particles.device),
        torch.arange(16, dtype=torch.int64, device=particles.device),
    )
    pieces: list[np.ndarray] = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(flat[start:start + Config.EVAL_MICRO_BATCH], inputs)
        ids = ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)
        pieces.append(ids.cpu().numpy().astype(np.uint16))
    return np.concatenate(pieces).reshape(particles.shape[:-1])


def reflect_unit_interval(values: torch.Tensor) -> torch.Tensor:
    folded = torch.remainder(values + 1.0, 4.0)
    return torch.where(folded <= 2.0, folded - 1.0, 3.0 - folded)


def distribution_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    probability = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probability * np.log2(probability)))


def js_divergence_from_counts(first: np.ndarray, second: np.ndarray) -> float:
    if first.sum() == 0 or second.sum() == 0:
        return float("nan")
    p = first.astype(np.float64) / first.sum()
    q = second.astype(np.float64) / second.sum()
    middle = 0.5 * (p + q)
    value = 0.0
    for distribution in (p, q):
        mask = distribution > 0
        value += 0.5 * float(np.sum(
            distribution[mask]
            * np.log2(distribution[mask] / middle[mask])
        ))
    return value


def safe_exp(log_value: float) -> float:
    return float(math.exp(log_value)) if log_value > -745.0 else 0.0


def make_generators(
    device: torch.device,
    target_index: int,
) -> dict[str, torch.Generator]:
    offset = 1_000_003 * int(target_index)
    generators = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    generators["prior"].manual_seed(Config.PRIOR_SEED + offset)
    generators["resample"].manual_seed(Config.RESAMPLE_SEED + offset)
    generators["mutation"].manual_seed(Config.MUTATION_SEED + offset)
    return generators


def initialize_state(
    device: torch.device,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    prior_generator: torch.Generator,
    parameter_count: int,
) -> SMCState:
    particles = torch.empty(
        Config.REPLICAS,
        Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    ).uniform_(-1.0, 1.0, generator=prior_generator)
    losses = evaluate_losses(particles, inputs, targets)
    total = Config.REPLICAS * Config.PARTICLES_PER_REPLICA
    lineages = torch.arange(
        total, device=device, dtype=torch.int64
    ).reshape(Config.REPLICAS, Config.PARTICLES_PER_REPLICA)
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume_fraction=torch.zeros(
            Config.REPLICAS, device=device, dtype=torch.float64
        ),
        proposal_scales=list(Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=float("inf"),
        threshold_index=0,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
        top_rows=[],
    )


def choose_next_threshold(state: SMCState) -> tuple[float, bool]:
    target = float(Config.TARGET_THRESHOLDS[state.threshold_index])
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=1
    )
    adaptive = float(quantiles.max().item())
    next_threshold = max(target, adaptive)
    if math.isfinite(state.current_threshold):
        next_threshold = min(next_threshold, state.current_threshold)
    return next_threshold, next_threshold <= target + 1e-12


@torch.no_grad()
def resample_within_replicas(
    state: SMCState,
    threshold: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    survival = np.zeros(Config.REPLICAS, dtype=np.float64)
    new_particles = torch.empty_like(state.particles)
    new_lineages = torch.empty_like(state.lineages)
    for replica in range(Config.REPLICAS):
        survivors = torch.nonzero(
            state.losses[replica] <= threshold + Config.LOSS_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if not len(survivors):
            raise RuntimeError(
                f"副本{replica}在阈值{threshold:.6g}没有幸存粒子。"
            )
        survival[replica] = len(survivors) / Config.PARTICLES_PER_REPLICA
        choices = torch.randint(
            len(survivors),
            (Config.PARTICLES_PER_REPLICA,),
            generator=generator,
            device=state.particles.device,
        )
        selected = survivors[choices]
        new_particles[replica] = state.particles[replica, selected]
        new_lineages[replica] = state.lineages[replica, selected]
    state.particles = new_particles
    state.lineages = new_lineages
    state.losses = evaluate_losses(state.particles, inputs, targets)
    state.log_volume_fraction += torch.log(torch.from_numpy(
        survival
    ).to(state.log_volume_fraction))
    return survival


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    scale: float,
    threshold: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> float:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    noise = torch.randn(
        current.shape,
        device=current.device,
        generator=generator,
        dtype=current.dtype,
    )
    local = reflect_unit_interval(current + scale * noise)
    if Config.REFRESH_PROBABILITY > 0:
        refresh = torch.rand(
            current.shape[:-1] + (1,),
            device=current.device,
            generator=generator,
        ) < Config.REFRESH_PROBABILITY
        fresh = torch.empty_like(current).uniform_(
            -1.0, 1.0, generator=generator
        )
        local = torch.where(refresh, fresh, local)
    proposal[..., block.start:block.stop] = local
    proposal_losses = evaluate_losses(proposal, inputs, targets)
    accept = proposal_losses <= threshold + Config.LOSS_TOLERANCE
    flat_accept = accept.reshape(-1)
    flat_particles = state.particles.reshape(-1, state.particles.shape[-1])
    flat_proposal = proposal.reshape(-1, proposal.shape[-1])
    flat_particles[flat_accept] = flat_proposal[flat_accept]
    flat_losses = state.losses.reshape(-1)
    flat_losses[flat_accept] = proposal_losses.reshape(-1)[flat_accept]
    return float(accept.float().mean().item())


def rejuvenate(
    state: SMCState,
    blocks: Sequence[ParameterBlock],
    threshold: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, float]:
    scales = list(state.proposal_scales)
    for _ in range(Config.ADAPT_SWEEPS):
        for block_index, block in enumerate(blocks):
            acceptance = mutate_block(
                state,
                block,
                scales[block_index],
                threshold,
                inputs,
                targets,
                generator,
            )
            scales[block_index] *= math.exp(
                Config.ADAPT_RATE * (acceptance - Config.TARGET_ACCEPTANCE)
            )
            scales[block_index] = min(
                max(scales[block_index], Config.MIN_PROPOSAL_SCALE),
                Config.MAX_PROPOSAL_SCALE,
            )
    state.proposal_scales = scales
    acceptance_sum = np.zeros(len(blocks), dtype=np.float64)
    for _ in range(Config.MUTATION_SWEEPS):
        for block_index, block in enumerate(blocks):
            acceptance_sum[block_index] += mutate_block(
                state,
                block,
                scales[block_index],
                threshold,
                inputs,
                targets,
                generator,
            )
    acceptance_mean = acceptance_sum / max(Config.MUTATION_SWEEPS, 1)
    return {
        f"acceptance_{block.name}": float(acceptance_mean[index])
        for index, block in enumerate(blocks)
    }


def append_top_functions(
    rows: list[dict[str, Any]],
    target: TargetSpec,
    threshold: float,
    counts: np.ndarray,
) -> None:
    total = counts.sum()
    if not total:
        return
    for rank, function_id in enumerate(
        np.argsort(counts)[::-1][:Config.TOP_FUNCTIONS], start=1
    ):
        count = int(counts[function_id])
        if not count:
            break
        rows.append({
            "target_name": target.name,
            "threshold": threshold,
            "rank": rank,
            "function_id": int(function_id),
            "function_hex": f"0x{int(function_id):04X}",
            "truth_table_x0_to_x15": function_bits(int(function_id)),
            "count": count,
            "probability": count / total,
            "is_target": int(function_id) == target.function_id,
        })


def record_threshold(
    output_dir: Path,
    state: SMCState,
    threshold: float,
    inputs: torch.Tensor,
    target: TargetSpec,
) -> None:
    ids = evaluate_function_ids(state.particles, inputs)
    losses = state.losses.detach().cpu().numpy()
    counts = np.bincount(ids.reshape(-1), minlength=65_536)
    target_probability = float(counts[target.function_id] / counts.sum())
    replica_js: list[float] = []
    replica_target: list[float] = []
    replica_logs = state.log_volume_fraction.detach().cpu().numpy()
    for replica in range(Config.REPLICAS):
        replica_counts = np.bincount(ids[replica], minlength=65_536)
        local_target = float(
            replica_counts[target.function_id] / replica_counts.sum()
        )
        local_js = js_divergence_from_counts(replica_counts, counts)
        replica_target.append(local_target)
        replica_js.append(local_js)
        state.replica_rows.append({
            "target_name": target.name,
            "threshold": threshold,
            "replica": replica,
            "estimated_log_volume_fraction": float(replica_logs[replica]),
            "estimated_log10_volume_fraction": float(
                replica_logs[replica] / math.log(10.0)
            ),
            "estimated_volume_fraction": safe_exp(float(replica_logs[replica])),
            "target_probability": local_target,
            "function_entropy_bits": distribution_entropy(replica_counts),
            "function_support": int(np.count_nonzero(replica_counts)),
            "js_to_aggregate": local_js,
            "unique_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        })

    log_median = float(np.median(replica_logs))
    hard_bound = math.log(2.0) / 16.0
    hard_exact_guaranteed = threshold < hard_bound
    row = {
        "target_name": target.name,
        "target_function_hex": target.function_hex,
        "threshold": threshold,
        "level": state.level,
        "particle_count": int(counts.sum()),
        "estimated_log_volume_fraction_median": log_median,
        "estimated_log_volume_fraction_min": float(np.min(replica_logs)),
        "estimated_log_volume_fraction_max": float(np.max(replica_logs)),
        "estimated_log10_volume_fraction_median": float(
            log_median / math.log(10.0)
        ),
        "estimated_volume_fraction_median": safe_exp(log_median),
        "loss_min": float(losses.min()),
        "loss_median": float(np.median(losses)),
        "loss_max": float(losses.max()),
        "target_probability": target_probability,
        "hard_exact_guaranteed_by_threshold": hard_exact_guaranteed,
        "hard_exact_violation_count": int(
            counts.sum() - counts[target.function_id]
        ) if hard_exact_guaranteed else None,
        "function_entropy_bits": distribution_entropy(counts),
        "function_support": int(np.count_nonzero(counts)),
        "top_function_hex": f"0x{int(np.argmax(counts)):04X}",
        "top_function_probability": float(counts.max() / counts.sum()),
        "replica_target_probability_min": float(np.min(replica_target)),
        "replica_target_probability_max": float(np.max(replica_target)),
        "replica_js_to_aggregate_median": float(np.median(replica_js)),
        "replica_js_to_aggregate_max": float(np.max(replica_js)),
        "replica_log_volume_range": float(
            np.max(replica_logs) - np.min(replica_logs)
        ),
        "unique_lineages_min": int(min(
            torch.unique(state.lineages[replica]).numel()
            for replica in range(Config.REPLICAS)
        )),
        "unique_lineages_max": int(max(
            torch.unique(state.lineages[replica]).numel()
            for replica in range(Config.REPLICAS)
        )),
    }
    if hard_exact_guaranteed and row["hard_exact_violation_count"] != 0:
        raise AssertionError(
            f"{target.name}在epsilon={threshold}低于hard-error下界后仍出现"
            f"{row['hard_exact_violation_count']}个非目标hard function。"
        )
    state.threshold_rows.append(row)
    append_top_functions(state.top_rows, target, threshold, counts)
    snapshot_name = f"snapshot_threshold_{threshold:.3f}".replace(".", "p")
    np.savez_compressed(
        output_dir / f"{snapshot_name}.npz",
        target_name=np.asarray(target.name),
        threshold=np.asarray(threshold),
        function_ids=ids,
        losses=losses,
        lineages=state.lineages.detach().cpu().numpy(),
        log_volume_fraction=replica_logs,
    )
    print(
        f"[{target.name}] TARGET eps={threshold:.3f} | "
        f"log10V~{row['estimated_log10_volume_fraction_median']:.2f} | "
        f"target={target_probability:.3%} | "
        f"top={row['top_function_hex']}:{row['top_function_probability']:.2%} | "
        f"replica logV range={row['replica_log_volume_range']:.2f}",
        flush=True,
    )


def state_payload(
    state: SMCState,
    target: TargetSpec,
    generators: dict[str, torch.Generator],
) -> dict[str, Any]:
    return {
        "config": config_dict(),
        "target": asdict(target),
        "particles": state.particles.detach().cpu(),
        "losses": state.losses.detach().cpu(),
        "lineages": state.lineages.detach().cpu(),
        "log_volume_fraction": state.log_volume_fraction.detach().cpu(),
        "proposal_scales": state.proposal_scales,
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "threshold_rows": state.threshold_rows,
        "replica_rows": state.replica_rows,
        "top_rows": state.top_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }


def save_checkpoint(
    output_dir: Path,
    state: SMCState,
    target: TargetSpec,
    generators: dict[str, torch.Generator],
) -> None:
    temporary = output_dir / "checkpoint.tmp.pt"
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(state_payload(state, target, generators), temporary)
    temporary.replace(checkpoint)


def load_checkpoint(
    output_dir: Path,
    target: TargetSpec,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> SMCState:
    payload = torch.load(
        output_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    saved_target = payload["target"]
    if (
        saved_target["name"] != target.name
        or int(saved_target["function_id"]) != target.function_id
    ):
        raise RuntimeError(f"{target.name}的checkpoint目标函数不一致。")
    saved = payload["config"]
    for key in immutable_config_keys():
        if saved.get(key) != config_dict().get(key):
            raise RuntimeError(f"checkpoint配置不一致：{key}")
    saved_thresholds = list(saved["TARGET_THRESHOLDS"])
    current_thresholds = list(json_ready(Config.TARGET_THRESHOLDS))
    if current_thresholds[:len(saved_thresholds)] != saved_thresholds:
        raise RuntimeError("checkpoint阈值序列不是当前序列前缀。")
    for name, generator in generators.items():
        generator.set_state(
            payload["generator_states"][name].to(dtype=torch.uint8, device="cpu")
        )
    return SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume_fraction=payload["log_volume_fraction"].to(
            device=device, dtype=torch.float64
        ),
        proposal_scales=[float(value) for value in payload["proposal_scales"]],
        current_threshold=float(payload["current_threshold"]),
        threshold_index=int(payload["threshold_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        threshold_rows=list(payload["threshold_rows"]),
        replica_rows=list(payload["replica_rows"]),
        top_rows=list(payload["top_rows"]),
    )


def write_target_artifacts(
    output_dir: Path,
    state: SMCState,
    target: TargetSpec,
    status: str,
) -> None:
    write_csv(output_dir / "levels.csv", state.level_rows)
    write_csv(output_dir / "volume_curve.csv", state.threshold_rows)
    write_csv(output_dir / "replica_volume_curve.csv", state.replica_rows)
    write_csv(output_dir / "top_functions.csv", state.top_rows)
    write_json(output_dir / "summary.json", {
        "status": status,
        "protocol": "full_truth_rule_volume_constrained_smc_v1",
        "target": asdict(target),
        "completed_thresholds": state.threshold_index,
        "requested_thresholds": len(Config.TARGET_THRESHOLDS),
        "current_threshold": state.current_threshold,
        "level": state.level,
        "hard_error_lower_bound": math.log(2.0) / 16.0,
        "threshold_rows": state.threshold_rows,
        "interpretation": {
            "measure": (
                "初始化参数立方体均匀测度中，完整真值表raw BCE不超过阈值的"
                "累计参数质量"
            ),
            "not_sgd": "SMC mutation不使用loss梯度。",
            "deep_retention": (
                "epsilon<=0.04时hard function被数学上固定为目标；继续收紧只"
                "测量该目标函数原像内部的深margin保留率。"
            ),
        },
    })


@torch.no_grad()
def run_prior_calibration(
    output_dir: Path,
    targets: Sequence[TargetSpec],
    device: torch.device,
    inputs: torch.Tensor,
    parameter_count: int,
) -> list[dict[str, Any]]:
    path = output_dir / "prior_calibration.csv"
    if path.exists() and Config.RESUME:
        rows = read_csv(path)
        if rows and int(rows[0]["requested_sample_count"]) == int(
            Config.PRIOR_CALIBRATION_SAMPLES
        ):
            print("直接prior校准已存在，跳过。", flush=True)
            return [dict(row) for row in rows]

    target_tensor = torch.as_tensor(
        np.asarray([target.outputs for target in targets], dtype=np.float32),
        device=device,
    )
    thresholds = np.asarray(Config.PRIOR_CALIBRATION_THRESHOLDS, dtype=np.float64)
    counts = np.zeros((len(targets), len(thresholds)), dtype=np.int64)
    hard_counts = np.zeros(len(targets), dtype=np.int64)
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.CALIBRATION_SEED)
    completed = 0
    started = time.perf_counter()
    while completed < Config.PRIOR_CALIBRATION_SAMPLES:
        batch = min(
            Config.PRIOR_CALIBRATION_BATCH,
            Config.PRIOR_CALIBRATION_SAMPLES - completed,
        )
        particles = torch.empty(
            batch, parameter_count, device=device
        ).uniform_(-1.0, 1.0, generator=generator)
        logits = forward_logits(particles, inputs)
        local_losses = F.binary_cross_entropy_with_logits(
            logits[:, None, :].expand(-1, len(targets), -1),
            target_tensor[None, :, :].expand(batch, -1, -1),
            reduction="none",
        ).mean(dim=2)
        losses_np = local_losses.cpu().numpy()
        for threshold_index, threshold in enumerate(thresholds):
            counts[:, threshold_index] += np.sum(
                losses_np <= threshold, axis=0
            )
        hard = (logits >= 0).to(torch.uint8)
        hard_counts += torch.all(
            hard[:, None, :] == target_tensor[None].to(torch.uint8), dim=2
        ).sum(dim=0).cpu().numpy()
        completed += batch
        if completed % max(Config.PRIOR_CALIBRATION_SAMPLES // 8, 1) == 0:
            print(
                f"prior calibration {completed:,}/"
                f"{Config.PRIOR_CALIBRATION_SAMPLES:,} | "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        hard_probability = hard_counts[target_index] / completed
        for threshold_index, threshold in enumerate(thresholds):
            count = int(counts[target_index, threshold_index])
            probability = count / completed
            rows.append({
                "target_name": target.name,
                "target_function_hex": target.function_hex,
                "threshold": float(threshold),
                "requested_sample_count": Config.PRIOR_CALIBRATION_SAMPLES,
                "actual_sample_count": completed,
                "count_below_threshold": count,
                "estimated_volume_fraction": probability,
                "estimated_log_volume_fraction": (
                    math.log(probability) if probability > 0 else None
                ),
                "binomial_standard_error": math.sqrt(
                    probability * (1.0 - probability) / completed
                ),
                "zero_count_95pct_upper_bound": (
                    1.0 - 0.05 ** (1.0 / completed) if count == 0 else None
                ),
                "hard_function_count": int(hard_counts[target_index]),
                "hard_function_prior_probability": hard_probability,
                "hard_function_log_prior_probability": (
                    math.log(hard_probability) if hard_probability > 0 else None
                ),
            })
    write_csv(path, rows)
    return rows


def target_completed(output_dir: Path) -> bool:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(
        summary.get("status") == "completed"
        and int(summary.get("completed_thresholds", -1))
        == len(Config.TARGET_THRESHOLDS)
    )


def run_target(
    root_dir: Path,
    target: TargetSpec,
    device: torch.device,
    inputs: torch.Tensor,
    blocks: Sequence[ParameterBlock],
    parameter_count: int,
) -> str:
    output_dir = root_dir / "targets" / target.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if Config.RESUME and target_completed(output_dir):
        print(f"[{target.name}] 已完成，跳过。", flush=True)
        return "completed"

    target_tensor = torch.as_tensor(
        np.asarray(target.outputs, dtype=np.float32), device=device
    )
    generators = make_generators(device, target.target_index)
    checkpoint = output_dir / "checkpoint.pt"
    if Config.RESUME and checkpoint.exists():
        state = load_checkpoint(output_dir, target, device, generators)
        print(
            f"[{target.name}] 恢复checkpoint：level={state.level} | "
            f"eps={state.current_threshold:.6g} | "
            f"threshold_index={state.threshold_index}",
            flush=True,
        )
    else:
        state = initialize_state(
            device,
            inputs,
            target_tensor,
            generators["prior"],
            parameter_count,
        )
        save_checkpoint(output_dir, state, target, generators)

    started = time.perf_counter()
    status = "running"
    try:
        while (
            state.threshold_index < len(Config.TARGET_THRESHOLDS)
            and state.level < Config.MAX_LEVELS_PER_TARGET
        ):
            previous = state.current_threshold
            next_threshold, reaches_target = choose_next_threshold(state)
            if (
                math.isfinite(previous)
                and next_threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reaches_target
            ):
                raise RuntimeError(
                    f"[{target.name}] SMC阈值停止下降；需要检查混合。"
                )
            survival = resample_within_replicas(
                state,
                next_threshold,
                inputs,
                target_tensor,
                generators["resample"],
            )
            mutation = rejuvenate(
                state,
                blocks,
                next_threshold,
                inputs,
                target_tensor,
                generators["mutation"],
            )
            state.level += 1
            state.current_threshold = next_threshold
            replica_logs = state.log_volume_fraction.detach().cpu().numpy()
            level_row: dict[str, Any] = {
                "target_name": target.name,
                "level": state.level,
                "threshold": next_threshold,
                "reaches_requested_target": reaches_target,
                "next_requested_target": float(
                    Config.TARGET_THRESHOLDS[state.threshold_index]
                ),
                "survival_fraction_min": float(np.min(survival)),
                "survival_fraction_median": float(np.median(survival)),
                "survival_fraction_max": float(np.max(survival)),
                "loss_min": float(state.losses.min().item()),
                "loss_median": float(state.losses.median().item()),
                "loss_max": float(state.losses.max().item()),
                "estimated_log_volume_fraction_median": float(
                    np.median(replica_logs)
                ),
                "estimated_log10_volume_fraction_median": float(
                    np.median(replica_logs) / math.log(10.0)
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "proposal_scales": list(state.proposal_scales),
            }
            level_row.update(mutation)
            state.level_rows.append(level_row)
            if (
                reaches_target
                or state.level % Config.LOG_EVERY_LEVELS == 0
            ):
                acceptance = ",".join(
                    f"{block.name}:{mutation[f'acceptance_{block.name}']:.1%}"
                    for block in blocks
                )
                print(
                    f"[{target.name}] level={state.level:>4} | "
                    f"eps={next_threshold:.6f} | "
                    f"survive={np.min(survival):.1%}/"
                    f"{np.median(survival):.1%}/{np.max(survival):.1%} | "
                    f"log10V~{level_row['estimated_log10_volume_fraction_median']:.2f} | "
                    f"accept[{acceptance}]",
                    flush=True,
                )

            if reaches_target:
                requested = float(
                    Config.TARGET_THRESHOLDS[state.threshold_index]
                )
                record_threshold(
                    output_dir, state, requested, inputs, target
                )
                state.threshold_index += 1
                write_target_artifacts(
                    output_dir, state, target, "running"
                )
                save_checkpoint(output_dir, state, target, generators)
            elif state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(output_dir, state, target, generators)

        status = (
            "completed"
            if state.threshold_index == len(Config.TARGET_THRESHOLDS)
            else "stopped_max_levels"
        )
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\n[{target.name}] 收到中断，保存checkpoint。", flush=True)
    finally:
        save_checkpoint(output_dir, state, target, generators)
        write_target_artifacts(output_dir, state, target, status)
    del state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return status


def aggregate_results(
    output_dir: Path,
    targets: Sequence[TargetSpec],
) -> dict[str, Any]:
    target_by_name = {target.name: target for target in targets}
    all_rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for target in targets:
        target_dir = output_dir / "targets" / target.name
        summary_path = target_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            statuses[target.name] = str(summary.get("status", "unknown"))
        for row in read_csv(target_dir / "volume_curve.csv"):
            all_rows.append({
                **row,
                "threshold": float(row["threshold"]),
                "estimated_log_volume_fraction_median": float(
                    row["estimated_log_volume_fraction_median"]
                ),
            })

    reference = float(Config.HARD_EXACT_REFERENCE_THRESHOLD)
    reference_log: dict[str, float] = {}
    for row in all_rows:
        if math.isclose(float(row["threshold"]), reference, abs_tol=1e-12):
            reference_log[str(row["target_name"])] = float(
                row["estimated_log_volume_fraction_median"]
            )

    enriched: list[dict[str, Any]] = []
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in all_rows:
        name = str(row["target_name"])
        target = target_by_name[name]
        threshold = float(row["threshold"])
        log_volume = float(row["estimated_log_volume_fraction_median"])
        enriched_row = {
            **row,
            "target_family": target.family,
            "target_formula": target.formula,
            "essential_variable_count": len(target.essential_variables),
            "anf_degree": target.anf_degree,
            "anf_term_count": target.anf_term_count,
            "deep_log_retention_from_reference": (
                log_volume - reference_log[name]
                if name in reference_log and threshold <= reference + 1e-12
                else None
            ),
            "deep_retention_from_reference": (
                safe_exp(log_volume - reference_log[name])
                if name in reference_log and threshold <= reference + 1e-12
                else None
            ),
        }
        enriched.append(enriched_row)
        grouped.setdefault(threshold, []).append(enriched_row)

    for threshold, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: float(row["estimated_log_volume_fraction_median"]),
            reverse=True,
        )
        for rank, row in enumerate(ordered, start=1):
            row["volume_rank_at_threshold"] = rank

    write_csv(output_dir / "combined_volume_curves.csv", enriched)

    calibration_rows = read_csv(output_dir / "prior_calibration.csv")
    calibration_map = {
        (str(row["target_name"]), float(row["threshold"])): row
        for row in calibration_rows
    }
    calibration_comparison: list[dict[str, Any]] = []
    for row in enriched:
        key = (str(row["target_name"]), float(row["threshold"]))
        calibration = calibration_map.get(key)
        if calibration is None:
            continue
        direct_probability = float(calibration["estimated_volume_fraction"])
        if direct_probability <= 0:
            calibration_comparison.append({
                "target_name": key[0],
                "threshold": key[1],
                "direct_prior_count": int(calibration["count_below_threshold"]),
                "direct_prior_probability": direct_probability,
                "smc_log_volume_median": float(
                    row["estimated_log_volume_fraction_median"]
                ),
                "comparable": False,
                "reason": "直接prior在该阈值零计数",
            })
            continue
        direct_log = math.log(direct_probability)
        smc_log = float(row["estimated_log_volume_fraction_median"])
        calibration_comparison.append({
            "target_name": key[0],
            "threshold": key[1],
            "direct_prior_count": int(calibration["count_below_threshold"]),
            "direct_prior_probability": direct_probability,
            "direct_prior_log_volume": direct_log,
            "smc_log_volume_median": smc_log,
            "smc_to_direct_log_ratio": smc_log - direct_log,
            "smc_to_direct_ratio": safe_exp(smc_log - direct_log),
            "comparable": True,
            "reason": "",
        })
    write_csv(
        output_dir / "prior_smc_calibration_comparison.csv",
        calibration_comparison,
    )

    pairwise_rows: list[dict[str, Any]] = []
    for threshold, rows in sorted(grouped.items(), reverse=True):
        by_name = {str(row["target_name"]): row for row in rows}
        for left_index, left in enumerate(targets):
            if left.name not in by_name:
                continue
            for right in targets[left_index + 1:]:
                if right.name not in by_name:
                    continue
                left_log = float(by_name[left.name][
                    "estimated_log_volume_fraction_median"
                ])
                right_log = float(by_name[right.name][
                    "estimated_log_volume_fraction_median"
                ])
                pairwise_rows.append({
                    "threshold": threshold,
                    "left_target": left.name,
                    "right_target": right.name,
                    "log_volume_ratio_left_over_right": left_log - right_log,
                    "log10_volume_ratio_left_over_right": (
                        left_log - right_log
                    ) / math.log(10.0),
                })
    write_csv(output_dir / "pairwise_log_volume_ratios.csv", pairwise_rows)

    slope_rows: list[dict[str, Any]] = []
    for target in targets:
        deep = sorted([
            row for row in enriched
            if row["target_name"] == target.name
            and float(row["threshold"]) <= reference + 1e-12
        ], key=lambda row: float(row["threshold"]), reverse=True)
        if len(deep) >= 2:
            x = -np.log(np.asarray([
                float(row["threshold"]) for row in deep
            ]))
            y = np.asarray([
                float(row["estimated_log_volume_fraction_median"])
                for row in deep
            ])
            slope, intercept = np.polyfit(x, y, 1)
            slope_rows.append({
                "target_name": target.name,
                "deep_point_count": len(deep),
                "slope_log_volume_vs_negative_log_epsilon": float(slope),
                "intercept": float(intercept),
                "interpretation": (
                    "斜率越负，hard-exact区域中的体积随loss收紧越快。"
                ),
            })
    write_csv(output_dir / "deep_tail_slopes.csv", slope_rows)

    parity_order_rows: list[dict[str, Any]] = []
    parity_names = [f"parity{index}" for index in range(1, 5)]
    for threshold, rows in sorted(grouped.items(), reverse=True):
        by_name = {str(row["target_name"]): row for row in rows}
        if not all(name in by_name for name in parity_names):
            continue
        values = [float(by_name[name][
            "estimated_log_volume_fraction_median"
        ]) for name in parity_names]
        parity_order_rows.append({
            "threshold": threshold,
            "parity1_log_volume": values[0],
            "parity2_log_volume": values[1],
            "parity3_log_volume": values[2],
            "parity4_log_volume": values[3],
            "strict_expected_order": bool(
                values[0] > values[1] > values[2] > values[3]
            ),
        })
    write_csv(output_dir / "parity_order_by_threshold.csv", parity_order_rows)
    comparable_calibration = [
        row for row in calibration_comparison if row.get("comparable")
    ]
    return {
        "target_statuses": statuses,
        "completed_target_count": sum(
            status == "completed" for status in statuses.values()
        ),
        "all_target_count": len(targets),
        "common_threshold_count": len(parity_order_rows),
        "strict_parity_order_threshold_count": sum(
            bool(row["strict_expected_order"]) for row in parity_order_rows
        ),
        "deep_reference_threshold": reference,
        "calibration_comparison_count": len(comparable_calibration),
        "calibration_max_abs_log_ratio": (
            max(abs(float(row["smc_to_direct_log_ratio"]))
                for row in comparable_calibration)
            if comparable_calibration else None
        ),
    }


def save_plots(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    rows = read_csv(output_dir / "combined_volume_curves.csv")
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = sorted(set(row["target_name"] for row in rows))
    for name in names:
        local = sorted(
            [row for row in rows if row["target_name"] == name],
            key=lambda row: float(row["threshold"]),
            reverse=True,
        )
        epsilon = np.asarray([float(row["threshold"]) for row in local])
        log10_volume = np.asarray([
            float(row["estimated_log_volume_fraction_median"])
            / math.log(10.0) for row in local
        ])
        axes[0].plot(epsilon, log10_volume, marker="o", ms=3, label=name)
        deep = [
            row for row in local
            if row.get("deep_log_retention_from_reference") not in ("", None)
        ]
        if deep:
            axes[1].plot(
                [float(row["threshold"]) for row in deep],
                [float(row["deep_log_retention_from_reference"])
                 / math.log(10.0) for row in deep],
                marker="o",
                ms=4,
                label=name,
            )
    hard_bound = math.log(2.0) / 16.0
    axes[0].axvline(hard_bound, color="black", ls="--", label="hard-exact bound")
    axes[0].set_xlabel("raw BCE threshold epsilon")
    axes[0].set_ylabel("median log10 V_f(epsilon)")
    axes[1].set_xlabel("raw BCE threshold epsilon")
    axes[1].set_ylabel("log10 retention from epsilon=0.04")
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "rule_volume_curves.png", dpi=180)
    plt.close(figure)


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.name not in {
                "checkpoint.pt",
                "checkpoint.tmp.pt",
            }:
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    targets = build_targets()
    write_csv(output_dir / "target_definitions.csv", [
        asdict(target) for target in targets
    ])
    inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    blocks, parameter_count = parameter_blocks(Config.WIDTH)
    hard_bound = math.log(2.0) / 16.0
    print("=== Full-truth Rule Volume constrained SMC ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=4->{Config.WIDTH}x2->1 tanh | params={parameter_count} | "
        f"full truth table n=16 | hard-exact bound={hard_bound:.8f}",
        flush=True,
    )
    print(
        f"targets={list(Config.TARGET_NAMES)} | replicas={Config.REPLICAS} | "
        f"particles/replica={Config.PARTICLES_PER_REPLICA:,}",
        flush=True,
    )
    print(f"结果目录：{output_dir}", flush=True)

    started = time.perf_counter()
    run_prior_calibration(
        output_dir, targets, device, inputs, parameter_count
    )
    statuses: dict[str, str] = {}
    interrupted = False
    for target in targets:
        print(
            f"\n=== target {target.target_index+1}/{len(targets)}: "
            f"{target.name} {target.function_hex} | {target.formula} ===",
            flush=True,
        )
        status = run_target(
            output_dir,
            target,
            device,
            inputs,
            blocks,
            parameter_count,
        )
        statuses[target.name] = status
        if status == "interrupted":
            interrupted = True
            break

    aggregate = aggregate_results(output_dir, targets)
    save_plots(output_dir)
    summary = {
        "status": "interrupted" if interrupted else (
            "completed" if aggregate["completed_target_count"] == len(targets)
            else "partial"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "network": f"4->{Config.WIDTH}x2->1 tanh",
        "parameter_count": parameter_count,
        "hard_error_lower_bound": hard_bound,
        "target_definitions": [asdict(target) for target in targets],
        "run_statuses": statuses,
        **aggregate,
        "primary_hypothesis": (
            "若简单规则的深尾收缩更慢，则parity1相对parity2/3/4的log-volume"
            "优势应随epsilon收紧而扩大，且从0.04归一化后的保留率应更高。"
        ),
        "interpretation_boundary": (
            "静态低-loss质量可以解释目标区域厚度，但不能单独推出SGD首次"
            "到达时间；后者还受梯度相干性、势垒和路径影响。"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
    print("\n=== 当前汇总 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
