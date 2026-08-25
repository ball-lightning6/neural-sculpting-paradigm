"""4-bit完整函数的深loss体积、收缩速度与加速度实验。

对每个目标函数 f，测量标准Gaussian初始化测度下的累计质量：

    V_f(epsilon) = P_theta[L_full(f)(theta) <= epsilon]

脚本使用不含loss梯度的constrained SMC / subset simulation。目标面板包括
结构规则、全部16个parity4单例外、16个平衡双例外，以及16个预注册随机平衡
映射。loss阈值在s=-log(epsilon)上等距，自动分析绝对体积排名、局部收缩率、
加速度、成对交叉和各家族相对parity4的难度比例。

这测量的是静态初始化测度，不是 SGD 到达概率。脚本另外用直接 prior 采样校准
高 loss 重叠区，但不会把 SMC 体积直接解释成优化器终点分布。
"""

from __future__ import annotations

import csv
import hashlib
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
    HIDDEN_LAYERS = 1
    STRUCTURED_TARGET_NAMES = (
        "parity1",
        "parity2",
        "parity3",
        "parity4",
        "majority3",
        "mux3",
    )
    SINGLE_EXCEPTION_COUNT = 16
    DOUBLE_EXCEPTION_COUNT = 16
    RANDOM_TARGET_COUNT = 16
    DOUBLE_EXCEPTION_SEED = 2026082504
    RANDOM_TARGET_SEED = 2026082505
    TARGET_PANEL_VERSION = "4bit_deep_volume_flow_gaussian_pcn_v2"
    PRIOR_KIND = "standard_gaussian_pcn"

    REPLICAS = 8
    PARTICLES_PER_REPLICA = 8_192
    SURVIVAL_QUANTILE = 0.5
    DEPTH_STEP = 0.15
    DEPTH_LEVELS = 25
    TARGET_THRESHOLDS = tuple(
        float(0.70 * math.exp(-0.15 * index))
        for index in range(25)
    )
    HARD_EXACT_REFERENCE_THRESHOLD = TARGET_THRESHOLDS[19]
    MAX_LEVELS_PER_TARGET = 5_000
    MIN_LEVEL_DECREMENT = 1e-7

    ADAPT_SWEEPS = 4
    MUTATION_SWEEPS = 12
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.10, 0.10, 0.04)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.95
    REFRESH_PROBABILITY = 0.0
    LOSS_TOLERANCE = 1e-7

    PRIOR_CALIBRATION_SAMPLES = 1_048_576
    PRIOR_CALIBRATION_BATCH = 16_384
    PRIOR_CALIBRATION_THRESHOLDS = TARGET_THRESHOLDS[:4]

    EVAL_MICRO_BATCH = 32_768
    CHECKPOINT_EVERY_LEVELS = 10
    LOG_EVERY_LEVELS = 10
    TOP_FUNCTIONS = 8
    SAVE_PARTICLE_SNAPSHOTS = False

    PRIOR_SEED = 2026082510
    RESAMPLE_SEED = 2026082520
    MUTATION_SEED = 2026082530
    CALIBRATION_SEED = 2026082540
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_4bit_deep_volume_flow_gaussian")
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
    cohort: str
    exception_count: int
    source_seed: int | None
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
    Config.STRUCTURED_TARGET_NAMES = ("parity1", "parity4")
    Config.SINGLE_EXCEPTION_COUNT = 1
    Config.DOUBLE_EXCEPTION_COUNT = 1
    Config.RANDOM_TARGET_COUNT = 1
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
        "_smoke_4bit_deep_volume_flow"
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
    if Config.INPUT_BITS != 4 or Config.HIDDEN_LAYERS != 1:
        raise ValueError("当前实现固定为4-bit输入、一个隐藏层。")
    thresholds = tuple(float(value) for value in Config.TARGET_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("TARGET_THRESHOLDS必须严格从高到低且不重复。")
    if not 0 < Config.SURVIVAL_QUANTILE < 1:
        raise ValueError("SURVIVAL_QUANTILE必须在(0,1)内。")
    if Config.REPLICAS < 2:
        raise ValueError("至少需要两个独立副本。")
    if len(Config.INITIAL_PROPOSAL_SCALES) != 3:
        raise ValueError("首层、输出层和全参数块需要三个proposal scale。")
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
        "STRUCTURED_TARGET_NAMES",
        "SINGLE_EXCEPTION_COUNT",
        "DOUBLE_EXCEPTION_COUNT",
        "RANDOM_TARGET_COUNT",
        "DOUBLE_EXCEPTION_SEED",
        "RANDOM_TARGET_SEED",
        "TARGET_PANEL_VERSION",
        "PRIOR_KIND",
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


def canonical_function_id(outputs: np.ndarray) -> int:
    """按输入置换和输出取反去重；两种变换都保持当前先验测度。"""
    import itertools

    inputs = truth_table_inputs()
    candidates: list[int] = []
    for permutation in itertools.permutations(range(Config.INPUT_BITS)):
        permuted = inputs[:, permutation]
        indices = np.sum(
            permuted
            * (1 << np.arange(Config.INPUT_BITS - 1, -1, -1))[None],
            axis=1,
        ).astype(np.int64)
        transformed = np.asarray(outputs, dtype=np.uint8)[indices]
        candidates.append(outputs_to_function_id(transformed))
        candidates.append(outputs_to_function_id(1 - transformed))
    return min(candidates)


def build_targets() -> list[TargetSpec]:
    inputs = truth_table_inputs()
    entries: list[dict[str, Any]] = []
    structured: dict[str, tuple[np.ndarray, str, str]] = {}
    for count in range(1, 5):
        outputs = np.bitwise_xor.reduce(inputs[:, :count], axis=1).astype(np.uint8)
        structured[f"parity{count}"] = (
            outputs,
            " XOR ".join(f"x{index}" for index in range(count)),
            "nested_parity",
        )
    majority = (inputs[:, :3].sum(axis=1) >= 2).astype(np.uint8)
    structured["majority3"] = (
        majority,
        "(x0+x1+x2)>=2",
        "linearly_separable_control",
    )
    mux = np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]).astype(
        np.uint8
    )
    structured["mux3"] = (
        mux,
        "IF x0 THEN x1 ELSE x2",
        "multiplexer_control",
    )

    for name in Config.STRUCTURED_TARGET_NAMES:
        outputs, formula, family = structured[name]
        entries.append({
            "name": name,
            "outputs": outputs,
            "formula": formula,
            "family": family,
            "cohort": "structured",
            "exception_count": 0,
            "source_seed": None,
        })

    parity4 = structured["parity4"][0]
    single_states = (
        list(range(16)) if Config.SINGLE_EXCEPTION_COUNT >= 16
        else [15][:Config.SINGLE_EXCEPTION_COUNT]
    )
    for state in single_states:
        outputs = parity4.copy()
        outputs[state] ^= 1
        entries.append({
            "name": f"parity4_flip1_{state:04b}",
            "outputs": outputs,
            "formula": f"parity4 XOR [x=={state:04b}]",
            "family": "parity4_single_exception",
            "cohort": "single_exception",
            "exception_count": 1,
            "source_seed": None,
        })

    zeros = np.flatnonzero(parity4 == 0)
    ones = np.flatnonzero(parity4 == 1)
    all_pairs = [(int(zero), int(one)) for zero in zeros for one in ones]
    double_rng = np.random.default_rng(Config.DOUBLE_EXCEPTION_SEED)
    selected_pairs = [
        all_pairs[index] for index in double_rng.choice(
            len(all_pairs),
            size=min(Config.DOUBLE_EXCEPTION_COUNT, len(all_pairs)),
            replace=False,
        )
    ]
    for pair_index, (zero_state, one_state) in enumerate(selected_pairs):
        outputs = parity4.copy()
        outputs[zero_state] ^= 1
        outputs[one_state] ^= 1
        entries.append({
            "name": (
                f"parity4_flip2_{pair_index:02d}_"
                f"{zero_state:04b}_{one_state:04b}"
            ),
            "outputs": outputs,
            "formula": (
                f"parity4 XOR [x=={zero_state:04b}] "
                f"XOR [x=={one_state:04b}]"
            ),
            "family": "parity4_balanced_double_exception",
            "cohort": "double_exception",
            "exception_count": 2,
            "source_seed": Config.DOUBLE_EXCEPTION_SEED,
        })

    excluded_orbits = {
        canonical_function_id(np.asarray(entry["outputs"], dtype=np.uint8))
        for entry in entries
    }
    random_rng = np.random.default_rng(Config.RANDOM_TARGET_SEED)
    random_attempt = 0
    random_added = 0
    while random_added < Config.RANDOM_TARGET_COUNT:
        outputs = np.zeros(16, dtype=np.uint8)
        outputs[random_rng.choice(16, size=8, replace=False)] = 1
        canonical = canonical_function_id(outputs)
        if len(essential_variables(outputs)) < 4 or canonical in excluded_orbits:
            random_attempt += 1
            continue
        excluded_orbits.add(canonical)
        entries.append({
            "name": f"random_balanced_{random_added:02d}",
            "outputs": outputs,
            "formula": (
                f"uniform balanced random mapping; seed={Config.RANDOM_TARGET_SEED}; "
                f"accepted={random_added}; attempt={random_attempt}"
            ),
            "family": "random_balanced_ensemble",
            "cohort": "random_balanced",
            "exception_count": -1,
            "source_seed": Config.RANDOM_TARGET_SEED,
        })
        random_added += 1
        random_attempt += 1

    targets: list[TargetSpec] = []
    seen_ids: set[int] = set()
    for target_index, entry in enumerate(entries):
        name = str(entry["name"])
        outputs = np.asarray(entry["outputs"], dtype=np.uint8)
        formula = str(entry["formula"])
        family = str(entry["family"])
        degree, terms, literals = anf_metrics(outputs)
        function_id = outputs_to_function_id(outputs)
        if function_id in seen_ids:
            raise AssertionError(f"目标面板出现重复函数：{name}")
        seen_ids.add(function_id)
        targets.append(TargetSpec(
            target_index=target_index,
            name=name,
            formula=formula,
            family=family,
            cohort=str(entry["cohort"]),
            exception_count=int(entry["exception_count"]),
            source_seed=entry["source_seed"],
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
    ).normal_(mean=0.0, std=1.0, generator=prior_generator)
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
    # pCN保持标准Gaussian先验测度不变；在硬约束条件分布中只需检查可行性。
    rho = min(max(float(scale), 0.0), 0.999999)
    local = math.sqrt(1.0 - rho * rho) * current + rho * noise
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
    if Config.SAVE_PARTICLE_SNAPSHOTS:
        snapshot_name = f"snapshot_threshold_{threshold:.6f}".replace(".", "p")
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
        "protocol": "4bit_deep_volume_flow_constrained_smc_v1",
        "target": asdict(target),
        "completed_thresholds": state.threshold_index,
        "requested_thresholds": len(Config.TARGET_THRESHOLDS),
        "current_threshold": state.current_threshold,
        "level": state.level,
        "hard_error_lower_bound": math.log(2.0) / 16.0,
        "threshold_rows": state.threshold_rows,
        "interpretation": {
            "measure": (
                "标准Gaussian初始化测度中，完整真值表raw BCE不超过阈值的"
                "累计参数质量"
            ),
            "not_sgd": "SMC mutation不使用loss梯度。",
            "deep_retention": (
                f"epsilon<={Config.HARD_EXACT_REFERENCE_THRESHOLD:.6g}时"
                "hard function被数学上固定为目标；继续收紧只"
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
        ).normal_(mean=0.0, std=1.0, generator=generator)
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
                status = "stalled_threshold"
                print(
                    f"[{target.name}] 阈值在{previous:.8f}停止下降；"
                    "保存当前链并继续下一目标。",
                    flush=True,
                )
                break
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

        if status == "running":
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
            "target_cohort": target.cohort,
            "exception_count": target.exception_count,
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


def analyze_scale_flow(
    output_dir: Path,
    targets: Sequence[TargetSpec],
) -> dict[str, Any]:
    """在固定depth坐标上计算位置、速度、加速度和排名交叉。"""
    target_by_name = {target.name: target for target in targets}
    rows = read_csv(output_dir / "combined_volume_curves.csv")
    typed: list[dict[str, Any]] = []
    for row in rows:
        typed.append({
            **row,
            "threshold": float(row["threshold"]),
            "log_volume": float(row["estimated_log_volume_fraction_median"]),
            "log_volume_min": float(row["estimated_log_volume_fraction_min"]),
            "log_volume_max": float(row["estimated_log_volume_fraction_max"]),
            "unique_lineages_min": int(float(row["unique_lineages_min"])),
            "unique_lineages_max": int(float(row["unique_lineages_max"])),
        })
    by_target: dict[str, dict[float, dict[str, Any]]] = {}
    by_threshold: dict[float, list[dict[str, Any]]] = {}
    for row in typed:
        by_target.setdefault(str(row["target_name"]), {})[
            float(row["threshold"])
        ] = row
        by_threshold.setdefault(float(row["threshold"]), []).append(row)

    absolute_rows: list[dict[str, Any]] = []
    for threshold, local in sorted(by_threshold.items(), reverse=True):
        ordered = sorted(local, key=lambda row: float(row["log_volume"]), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            target = target_by_name[str(row["target_name"])]
            absolute_rows.append({
                "threshold": threshold,
                "depth": -math.log(threshold),
                "rank_largest_volume_first": rank,
                "target_name": target.name,
                "cohort": target.cohort,
                "family": target.family,
                "exception_count": target.exception_count,
                "log_volume": row["log_volume"],
                "complexity_negative_log_volume": -float(row["log_volume"]),
                "log_volume_min": row["log_volume_min"],
                "log_volume_max": row["log_volume_max"],
                "unique_lineages_min": row["unique_lineages_min"],
                "unique_lineages_max": row["unique_lineages_max"],
            })
    write_csv(output_dir / "absolute_volume_rankings.csv", absolute_rows)

    rate_rows: list[dict[str, Any]] = []
    acceleration_rows: list[dict[str, Any]] = []
    rate_by_target: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        local = by_target.get(target.name, {})
        thresholds = sorted(local, reverse=True)
        replica_csv = read_csv(
            output_dir / "targets" / target.name / "replica_volume_curve.csv"
        )
        replica_values: dict[int, dict[float, float]] = {}
        for row in replica_csv:
            replica_values.setdefault(int(row["replica"]), {})[
                float(row["threshold"])
            ] = float(row["estimated_log_volume_fraction"])
        target_rates: list[dict[str, Any]] = []
        for high, low in zip(thresholds[:-1], thresholds[1:]):
            depth_high = -math.log(high)
            depth_low = -math.log(low)
            delta_depth = depth_low - depth_high
            complexity_high = -float(local[high]["log_volume"])
            complexity_low = -float(local[low]["log_volume"])
            replica_rates = []
            for values in replica_values.values():
                if high in values and low in values:
                    replica_rates.append(
                        ((-values[low]) - (-values[high])) / delta_depth
                    )
            row = {
                "target_name": target.name,
                "cohort": target.cohort,
                "family": target.family,
                "exception_count": target.exception_count,
                "epsilon_high": high,
                "epsilon_low": low,
                "depth_mid": 0.5 * (depth_high + depth_low),
                "epsilon_geometric_mid": math.sqrt(high * low),
                "delta_depth": delta_depth,
                "contraction_rate": (
                    (complexity_low - complexity_high) / delta_depth
                ),
                "contraction_rate_replica_median": (
                    float(np.median(replica_rates)) if replica_rates else None
                ),
                "contraction_rate_replica_min": (
                    float(np.min(replica_rates)) if replica_rates else None
                ),
                "contraction_rate_replica_max": (
                    float(np.max(replica_rates)) if replica_rates else None
                ),
                "replica_count": len(replica_rates),
                "low_threshold_unique_lineages_min": int(
                    local[low]["unique_lineages_min"]
                ),
                "low_threshold_unique_lineages_max": int(
                    local[low]["unique_lineages_max"]
                ),
                "hard_exact_window": bool(
                    high < math.log(2.0) / (2 ** Config.INPUT_BITS)
                ),
            }
            target_rates.append(row)
            rate_rows.append(row)
        rate_by_target[target.name] = target_rates
        for previous, current in zip(target_rates[:-1], target_rates[1:]):
            delta_mid = float(current["depth_mid"]) - float(previous["depth_mid"])
            acceleration_rows.append({
                "target_name": target.name,
                "cohort": target.cohort,
                "family": target.family,
                "exception_count": target.exception_count,
                "depth_mid": 0.5 * (
                    float(previous["depth_mid"]) + float(current["depth_mid"])
                ),
                "epsilon_geometric_mid": math.exp(-0.5 * (
                    float(previous["depth_mid"]) + float(current["depth_mid"])
                )),
                "acceleration": (
                    float(current["contraction_rate"])
                    - float(previous["contraction_rate"])
                ) / delta_mid,
                "rate_before": previous["contraction_rate"],
                "rate_after": current["contraction_rate"],
            })

    rate_groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rate_rows:
        rate_groups.setdefault(
            (float(row["epsilon_high"]), float(row["epsilon_low"])), []
        ).append(row)
    for window, local in rate_groups.items():
        ordered = sorted(local, key=lambda row: float(row["contraction_rate"]))
        for rank, row in enumerate(ordered, start=1):
            row["rate_rank_slowest_first"] = rank
            row["window_target_count"] = len(local)
    write_csv(output_dir / "local_contraction_rates.csv", rate_rows)
    write_csv(output_dir / "local_contraction_accelerations.csv", acceleration_rows)

    cohort_absolute: list[dict[str, Any]] = []
    parity4_by_threshold = by_target.get("parity4", {})
    for threshold, local in sorted(by_threshold.items(), reverse=True):
        parity4_row = parity4_by_threshold.get(threshold)
        parity4_complexity = (
            -float(parity4_row["log_volume"]) if parity4_row else None
        )
        cohorts = sorted({target_by_name[str(row["target_name"])].cohort for row in local})
        for cohort in cohorts:
            cohort_local = [
                row for row in local
                if target_by_name[str(row["target_name"])].cohort == cohort
            ]
            values = np.asarray([-float(row["log_volume"]) for row in cohort_local])
            cohort_absolute.append({
                "threshold": threshold,
                "depth": -math.log(threshold),
                "cohort": cohort,
                "target_count": len(values),
                "complexity_median": float(np.median(values)),
                "complexity_q10": float(np.quantile(values, 0.10)),
                "complexity_q90": float(np.quantile(values, 0.90)),
                "complexity_min": float(np.min(values)),
                "complexity_max": float(np.max(values)),
                "fraction_harder_than_parity4": (
                    float(np.mean(values > parity4_complexity))
                    if parity4_complexity is not None else None
                ),
            })
    write_csv(output_dir / "cohort_absolute_volume_summary.csv", cohort_absolute)

    cohort_rates: list[dict[str, Any]] = []
    for (high, low), local in sorted(rate_groups.items(), reverse=True):
        parity_rows = [row for row in local if row["target_name"] == "parity4"]
        parity_rate = float(parity_rows[0]["contraction_rate"]) if parity_rows else None
        cohorts = sorted({str(row["cohort"]) for row in local})
        for cohort in cohorts:
            cohort_local = [row for row in local if row["cohort"] == cohort]
            values = np.asarray([
                float(row["contraction_rate"]) for row in cohort_local
            ])
            cohort_rates.append({
                "epsilon_high": high,
                "epsilon_low": low,
                "depth_mid": float(cohort_local[0]["depth_mid"]),
                "cohort": cohort,
                "target_count": len(values),
                "rate_median": float(np.median(values)),
                "rate_q10": float(np.quantile(values, 0.10)),
                "rate_q90": float(np.quantile(values, 0.90)),
                "rate_min": float(np.min(values)),
                "rate_max": float(np.max(values)),
                "fraction_faster_than_parity4": (
                    float(np.mean(values > parity_rate))
                    if parity_rate is not None else None
                ),
            })
    write_csv(output_dir / "cohort_contraction_rate_summary.csv", cohort_rates)

    crossings: list[dict[str, Any]] = []
    if "parity4" in by_target:
        base = by_target["parity4"]
        for target in targets:
            if target.name == "parity4":
                continue
            local = by_target.get(target.name, {})
            common = sorted(set(base) & set(local), reverse=True)
            relative = [
                (
                    threshold,
                    (-float(local[threshold]["log_volume"]))
                    - (-float(base[threshold]["log_volume"])),
                ) for threshold in common
            ]
            for (high, delta_high), (low, delta_low) in zip(
                relative[:-1], relative[1:]
            ):
                if delta_high == 0 or delta_low == 0 or delta_high * delta_low < 0:
                    depth_high = -math.log(high)
                    depth_low = -math.log(low)
                    fraction = (
                        -delta_high / (delta_low - delta_high)
                        if delta_low != delta_high else 0.5
                    )
                    crossing_depth = depth_high + fraction * (
                        depth_low - depth_high
                    )
                    crossings.append({
                        "comparison_target": target.name,
                        "cohort": target.cohort,
                        "epsilon_high": high,
                        "epsilon_low": low,
                        "relative_complexity_high": delta_high,
                        "relative_complexity_low": delta_low,
                        "estimated_crossing_epsilon": math.exp(-crossing_depth),
                        "direction": (
                            "target_becomes_harder_than_parity4"
                            if delta_low > delta_high
                            else "target_becomes_easier_than_parity4"
                        ),
                    })
    write_csv(output_dir / "parity4_absolute_volume_crossings.csv", crossings)

    structured_names = [
        target.name for target in targets if target.cohort == "structured"
    ]
    common_thresholds = sorted(
        set.intersection(*(
            set(by_target.get(name, {})) for name in structured_names
        )), reverse=True
    ) if structured_names else []
    structured_absolute_orders = []
    for threshold in common_thresholds:
        structured_absolute_orders.append([
            name for name in sorted(
                structured_names,
                key=lambda item: float(by_target[item][threshold]["log_volume"]),
                reverse=True,
            )
        ])
    structured_rate_orders = []
    common_windows = sorted(set.intersection(*(
        {
            (float(row["epsilon_high"]), float(row["epsilon_low"]))
            for row in rate_by_target.get(name, [])
        } for name in structured_names
    )), reverse=True) if structured_names else []
    for window in common_windows:
        structured_rate_orders.append([
            name for name in sorted(
                structured_names,
                key=lambda item: next(
                    float(row["contraction_rate"])
                    for row in rate_by_target[item]
                    if (
                        float(row["epsilon_high"]),
                        float(row["epsilon_low"]),
                    ) == window
                ),
            )
        ])
    stability = {
        "deepest_common_thresholds": common_thresholds[-5:],
        "deepest_structured_absolute_orders": structured_absolute_orders[-5:],
        "structured_absolute_order_stable_last_five": bool(
            len(structured_absolute_orders) >= 5
            and all(
                order == structured_absolute_orders[-1]
                for order in structured_absolute_orders[-5:]
            )
        ),
        "deepest_common_windows": common_windows[-5:],
        "deepest_structured_rate_orders": structured_rate_orders[-5:],
        "structured_rate_order_stable_last_five": bool(
            len(structured_rate_orders) >= 5
            and all(
                order == structured_rate_orders[-1]
                for order in structured_rate_orders[-5:]
            )
        ),
        "parity4_crossing_count": len(crossings),
    }
    write_json(output_dir / "stability_diagnostics.json", stability)
    return stability


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
    axes[1].set_ylabel(
        f"log10 retention from epsilon={Config.HARD_EXACT_REFERENCE_THRESHOLD:.4g}"
    )
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "rule_volume_curves.png", dpi=180)
    plt.close(figure)


def save_flow_plots(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "flow_plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    absolute = read_csv(output_dir / "absolute_volume_rankings.csv")
    rates = read_csv(output_dir / "local_contraction_rates.csv")
    cohort_absolute = read_csv(output_dir / "cohort_absolute_volume_summary.csv")
    cohort_rates = read_csv(output_dir / "cohort_contraction_rate_summary.csv")
    if not absolute or not rates:
        return

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    structured = [
        "parity1", "majority3", "mux3", "parity2", "parity3", "parity4"
    ]
    colors = {
        "single_exception": "tab:orange",
        "double_exception": "tab:purple",
        "random_balanced": "black",
    }
    labels = {
        "single_exception": "parity4 + 1 exception",
        "double_exception": "parity4 + 2 balanced exceptions",
        "random_balanced": "random balanced",
    }
    for name in structured:
        local = sorted(
            [row for row in absolute if row["target_name"] == name],
            key=lambda row: float(row["depth"]),
        )
        if local:
            axes[0, 0].plot(
                [float(row["depth"]) for row in local],
                [float(row["complexity_negative_log_volume"]) for row in local],
                marker="o", ms=3, label=name,
            )
        local_rate = sorted(
            [row for row in rates if row["target_name"] == name],
            key=lambda row: float(row["depth_mid"]),
        )
        if local_rate:
            axes[0, 1].plot(
                [float(row["depth_mid"]) for row in local_rate],
                [float(row["contraction_rate"]) for row in local_rate],
                marker="o", ms=3, label=name,
            )

    for cohort in ("single_exception", "double_exception", "random_balanced"):
        local = sorted(
            [
                row for row in cohort_absolute
                if row["cohort"] == cohort
                and row.get("fraction_harder_than_parity4") not in ("", None)
            ],
            key=lambda row: float(row["depth"]),
        )
        if local:
            x = np.asarray([float(row["depth"]) for row in local])
            median = np.asarray([float(row["complexity_median"]) for row in local])
            low = np.asarray([float(row["complexity_q10"]) for row in local])
            high = np.asarray([float(row["complexity_q90"]) for row in local])
            axes[0, 0].plot(
                x, median, lw=2.5, color=colors[cohort], label=labels[cohort]
            )
            axes[0, 0].fill_between(x, low, high, color=colors[cohort], alpha=0.13)
        local_rate = sorted(
            [
                row for row in cohort_rates
                if row["cohort"] == cohort
                and row.get("fraction_faster_than_parity4") not in ("", None)
            ],
            key=lambda row: float(row["depth_mid"]),
        )
        if local_rate:
            x = np.asarray([float(row["depth_mid"]) for row in local_rate])
            median = np.asarray([float(row["rate_median"]) for row in local_rate])
            low = np.asarray([float(row["rate_q10"]) for row in local_rate])
            high = np.asarray([float(row["rate_q90"]) for row in local_rate])
            axes[0, 1].plot(x, median, lw=2.5, color=colors[cohort])
            axes[0, 1].fill_between(x, low, high, color=colors[cohort], alpha=0.13)

    for cohort in ("single_exception", "double_exception", "random_balanced"):
        local = sorted(
            [
                row for row in cohort_absolute
                if row["cohort"] == cohort
                and row.get("fraction_harder_than_parity4") not in ("", None)
            ],
            key=lambda row: float(row["depth"]),
        )
        if local:
            axes[1, 0].plot(
                [float(row["depth"]) for row in local],
                [float(row["fraction_harder_than_parity4"]) for row in local],
                marker="o", ms=4, color=colors[cohort], label=labels[cohort],
            )
        local_rate = sorted(
            [
                row for row in cohort_rates
                if row["cohort"] == cohort
                and row.get("fraction_faster_than_parity4") not in ("", None)
            ],
            key=lambda row: float(row["depth_mid"]),
        )
        if local_rate:
            axes[1, 1].plot(
                [float(row["depth_mid"]) for row in local_rate],
                [float(row["fraction_faster_than_parity4"]) for row in local_rate],
                marker="o", ms=4, color=colors[cohort], label=labels[cohort],
            )

    hard_depth = -math.log(math.log(2.0) / 16.0)
    for axis in axes.flat:
        axis.axvline(hard_depth, color="gray", ls="--", alpha=0.8)
        axis.grid(alpha=0.25)
        axis.set_xlabel("loss depth s = -log(epsilon)")
    axes[0, 0].set_ylabel("complexity -log V")
    axes[0, 0].set_title("absolute low-loss volume profile")
    axes[0, 1].set_ylabel("d(-log V) / ds")
    axes[0, 1].set_title("local contraction rate")
    axes[1, 0].set_ylabel("fraction with smaller V than parity4")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_title("absolute-complexity crossings")
    axes[1, 1].set_ylabel("fraction contracting faster than parity4")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_title("rate crossings")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[1, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "deep_volume_flow.png", dpi=180)
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
    panel_payload = {
        "protocol": Config.TARGET_PANEL_VERSION,
        "created_before_smc": True,
        "network": f"4->{Config.WIDTH}->1 tanh",
        "prior": "independent standard Gaussian coordinates with fan-in scaling",
        "depth_coordinate": "s=-log(epsilon)",
        "thresholds": list(Config.TARGET_THRESHOLDS),
        "targets": [asdict(target) for target in targets],
    }
    canonical_panel = json.dumps(
        json_ready(panel_payload), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    panel_payload["panel_sha256"] = hashlib.sha256(canonical_panel).hexdigest()
    panel_path = output_dir / "preregistered_target_panel.json"
    if panel_path.exists():
        saved_panel = json.loads(panel_path.read_text(encoding="utf-8"))
        if saved_panel != json_ready(panel_payload):
            raise RuntimeError("已有结果的预注册目标面板不一致。")
    else:
        write_json(panel_path, panel_payload)
    write_csv(output_dir / "target_definitions.csv", [
        asdict(target) for target in targets
    ])
    inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    blocks, parameter_count = parameter_blocks(Config.WIDTH)
    hard_bound = math.log(2.0) / 16.0
    print("=== 4-bit Deep Volume Flow constrained SMC ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=4->{Config.WIDTH}->1 tanh | params={parameter_count} | "
        f"full truth table n=16 | hard-exact bound={hard_bound:.8f}",
        flush=True,
    )
    print(
        f"targets={len(targets)} | cohorts="
        f"structured={sum(t.cohort == 'structured' for t in targets)},"
        f"single={sum(t.cohort == 'single_exception' for t in targets)},"
        f"double={sum(t.cohort == 'double_exception' for t in targets)},"
        f"random={sum(t.cohort == 'random_balanced' for t in targets)} | "
        f"replicas={Config.REPLICAS} | "
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
    flow = analyze_scale_flow(output_dir, targets)
    save_flow_plots(output_dir)
    summary = {
        "status": "interrupted" if interrupted else (
            "completed" if aggregate["completed_target_count"] == len(targets)
            else "partial"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "network": f"4->{Config.WIDTH}->1 tanh",
        "parameter_count": parameter_count,
        "hard_error_lower_bound": hard_bound,
        "target_definitions": [asdict(target) for target in targets],
        "run_statuses": statuses,
        **aggregate,
        "flow_analysis": flow,
        "primary_hypothesis": (
            "不同函数的有效复杂度允许随loss发生有限次交叉；进入hard-exact"
            "深尾后，结构规则骨架及random/exception家族的相对顺序是否稳定。"
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
