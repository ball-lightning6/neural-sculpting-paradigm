"""E23 深尾确认：五目标共享父系综的 8-bit 锁步 SMC。

原 E23 在 raw BCE 0.690 -> 0.670 的浅层割线上冻结 volume score，
随后独立测量随机训练集恢复相变。Parity 家族严格命中，但三对浅层排序
与 n50/n90 不一致：

    parity2  < MUX3            （操作性复杂度，从易到难）
    parity3  < random_balanced
    parity4  < random_balanced

本脚本不重新估计相变，也不读取结果后修改预言。原 E23 使用
Uniform[-1,1] 归一化参数，但该有界立方体存在约 0.0142 的理论 BCE
下界，无法进入真正深尾。默认协议因此使用与 Uniform[-1,1] 同方差
（sigma=1/sqrt(3)）的无界 Gaussian 参考测度，并从共同父事件

    min_f L_full(f) <= epsilon_parent

采样，再将同一组 parent replica/lineage 分成五个目标分支并锁步推进。
共同父体积在两两比值中抵消，直接测量三条预注册比值：

    log V(parity2) - log V(mux3)
    log V(parity3) - log V(random_balanced)
    log V(parity4) - log V(random_balanced)

正值表示体积排序与 n50/n90 的操作性排序一致。脚本支持 checkpoint、
Ctrl-C 保存、扩展 TARGET_THRESHOLDS 后续跑，以及达到严格停止条件后自动停止。
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2

    TARGET_NAMES = (
        "parity2",
        "parity3",
        "parity4",
        "mux3",
        "random_balanced",
    )
    RANDOM_TARGET_SEED = 2026082401
    REFERENCE_MEASURE = "gaussian_matched_variance"
    GAUSSIAN_SIGMA = 1.0 / math.sqrt(3.0)
    EXPECTED_FUNCTION_HEX = {
        "parity2": "0x0000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000",
        "parity3": "0xFFFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFFFFFFFFFF00000000",
        "parity4": "0x0000FFFFFFFF0000FFFF00000000FFFFFFFF00000000FFFF0000FFFFFFFF0000",
        "mux3": "0xFFFFFFFFFFFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFF00000000",
        "random_balanced": "0x0FB59096D3DC7A99089D0306BFEA2A160D4D715AD394F1FAD6A050F607BB0E97",
    }

    # (操作上更易, 操作上更难, easy n50/n90, hard n50/n90)
    # random 的 (241, 241) 表示截至 n=240 仍未恢复，是右删失编码。
    EXPECTED_PAIRS = (
        ("parity2", "mux3", (64, 80), (80, 112)),
        ("parity3", "random_balanced", (96, 112), (241, 241)),
        ("parity4", "random_balanced", (160, 160), (241, 241)),
    )
    # (previous grid point, first crossing grid point); None 表示右删失。
    TRANSITION_BRACKETS = {
        "parity2": {"n50": (48, 64), "n90": (64, 80)},
        "parity3": {"n50": (80, 96), "n90": (96, 112)},
        "parity4": {"n50": (128, 160), "n90": (128, 160)},
        "mux3": {"n50": (64, 80), "n90": (96, 112)},
        "random_balanced": {"n50": (240, None), "n90": (240, None)},
    }
    # 既有 E23 数据的后验分辨率诊断，不作为原预注册主判决。
    AGREEMENT99_DIAGNOSTIC = {
        "parity2": (59.98, 58.77, 61.07),
        "mux3": (69.47, 65.02, 72.40),
        "parity3": (88.94, 85.72, 91.05),
        "parity4": (151.93, 149.65, 153.81),
        "random_balanced": (None, 240.0, None),
    }

    PARENT_THRESHOLD = 0.700
    DEPTH_STEP = 0.12
    DEPTH_LEVELS = 70
    TARGET_THRESHOLDS = tuple(
        float(0.700 * math.exp(-0.12 * index))
        for index in range(70)
    )

    REPLICAS = 8
    PARTICLES_PER_REPLICA = 8_192
    SURVIVAL_QUANTILE = 0.5
    MAX_LEVELS_PARENT = 1_000
    MAX_LEVELS_BRANCH = 30_000
    MIN_LEVEL_DECREMENT = 1e-8

    ADAPT_SWEEPS = 3
    MUTATION_SWEEPS = 8
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.050, 0.030, 0.050, 0.015)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.30
    REFRESH_PROBABILITY = 0.02
    LOSS_TOLERANCE = 1e-7

    EVAL_MICRO_BATCH = 8_192
    CHECKPOINT_EVERY_LEVELS = 10
    LOG_EVERY_LEVELS = 10

    STOP_CONSECUTIVE_WINDOWS = 5
    STOP_REQUIRE_ALL_REPLICAS_CROSSED = True
    STOP_REQUIRE_POSITIVE_MEDIAN_RATE = True

    PRIOR_SEED = 2026082701
    PARENT_RESAMPLE_SEED = 2026082702
    PARENT_MUTATION_SEED = 2026082703
    BRANCH_RESAMPLE_SEED = 2026082710
    BRANCH_MUTATION_SEED = 2026082720

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path(
        "/root/results_8bit_mismatch_gaussian_joint_deep_bridge"
    )
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
    index: int
    name: str
    formula: str
    outputs: tuple[int, ...]
    function_hex: str


@dataclass
class SMCState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume: torch.Tensor
    proposal_scales: list[float]
    current_threshold: float
    threshold_index: int
    level: int
    level_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    replica_rows: list[dict[str, Any]]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.PARENT_THRESHOLD = 0.72
    Config.TARGET_THRESHOLDS = (0.72, 0.71, 0.70)
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 128
    Config.MAX_LEVELS_PARENT = 20
    Config.MAX_LEVELS_BRANCH = 60
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 2
    Config.EVAL_MICRO_BATCH = 128
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.STOP_CONSECUTIVE_WINDOWS = 2
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/neural_k_framework/experiments/"
        "experiment_23_volume_to_data_transition/"
        "_smoke_8bit_mismatch_joint_deep_bridge"
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
                    if isinstance(value, (dict, list, tuple)) else value
                )
                for key, value in row.items()
            })


def config_payload() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    }


def validate_config() -> None:
    if Config.INPUT_BITS != 8 or Config.WIDTH != 16:
        raise ValueError("本判决固定使用 E23 的 8-bit、width=16 协议。")
    if Config.HIDDEN_LAYERS != 2:
        raise ValueError("本判决固定使用两个隐藏层。")
    if Config.TARGET_THRESHOLDS[0] != Config.PARENT_THRESHOLD:
        raise ValueError("TARGET_THRESHOLDS 首项必须等于 PARENT_THRESHOLD。")
    if tuple(sorted(set(Config.TARGET_THRESHOLDS), reverse=True)) != tuple(
        Config.TARGET_THRESHOLDS
    ):
        raise ValueError("TARGET_THRESHOLDS 必须严格递减且不重复。")
    if len(Config.INITIAL_PROPOSAL_SCALES) != 4:
        raise ValueError("三个参数层和全参数块需要四个 proposal scale。")
    if not 0 < Config.SURVIVAL_QUANTILE < 1:
        raise ValueError("SURVIVAL_QUANTILE 必须位于 (0,1)。")
    if Config.REFERENCE_MEASURE not in {
        "gaussian_matched_variance", "uniform_cube"
    }:
        raise ValueError("未知 REFERENCE_MEASURE。")
    if Config.REFERENCE_MEASURE == "uniform_cube":
        max_logit = (
            Config.WIDTH / math.sqrt(Config.WIDTH)
            + 1.0 / math.sqrt(Config.WIDTH)
        )
        theoretical_floor = math.log1p(math.exp(-max_logit))
        if min(Config.TARGET_THRESHOLDS) < theoretical_floor:
            raise ValueError(
                "uniform_cube 的目标阈值低于 bounded output 的理论 BCE "
                f"下界 {theoretical_floor:.6g}；请改用 Gaussian 深尾协议，"
                "或截断阈值序列。"
            )


def truth_table_inputs() -> np.ndarray:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.uint16)
    shifts = np.arange(
        Config.INPUT_BITS - 1, -1, -1, dtype=np.uint16
    )
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def outputs_to_hex(outputs: np.ndarray) -> str:
    function_id = 0
    for index, bit in enumerate(np.asarray(outputs, dtype=np.uint8)):
        function_id |= int(bit) << index
    width = len(outputs) // 4
    return f"0x{function_id:0{width}X}"


def anf_metrics(outputs: np.ndarray) -> tuple[int, int, int]:
    coefficients = np.asarray(outputs, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(2 ** Config.INPUT_BITS):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.asarray([int(term).bit_count() for term in terms])
    return (
        int(degrees.max()) if len(degrees) else 0,
        int(len(terms)),
        int(degrees.sum()),
    )


def essential_variable_count(outputs: np.ndarray) -> int:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    count = 0
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        count += int(np.any(outputs[base] != outputs[base | (1 << bit)]))
    return count


def build_targets() -> list[TargetSpec]:
    inputs = truth_table_inputs()
    raw: dict[str, tuple[np.ndarray, str]] = {}
    for count in (2, 3, 4):
        outputs = np.bitwise_xor.reduce(
            inputs[:, :count], axis=1
        ).astype(np.uint8)
        raw[f"parity{count}"] = (
            outputs, " XOR ".join(f"x{i}" for i in range(count))
        )
    raw["mux3"] = (
        np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]).astype(
            np.uint8
        ),
        "IF x0 THEN x1 ELSE x2",
    )

    rng = np.random.default_rng(Config.RANDOM_TARGET_SEED)
    attempt = 0
    while True:
        outputs = np.zeros(2 ** Config.INPUT_BITS, dtype=np.uint8)
        outputs[rng.choice(len(outputs), len(outputs) // 2, replace=False)] = 1
        _, terms, literals = anf_metrics(outputs)
        if (
            essential_variable_count(outputs) == Config.INPUT_BITS
            and terms >= 100 and literals >= 350
        ):
            break
        attempt += 1
    raw["random_balanced"] = (
        outputs,
        f"balanced random seed={Config.RANDOM_TARGET_SEED}, attempt={attempt}",
    )

    targets = []
    for index, name in enumerate(Config.TARGET_NAMES):
        outputs, formula = raw[name]
        if int(outputs.sum()) != len(outputs) // 2:
            raise AssertionError(f"{name} 不是平衡目标。")
        function_hex = outputs_to_hex(outputs)
        if function_hex != Config.EXPECTED_FUNCTION_HEX[name]:
            raise AssertionError(
                f"{name} 真值表与原 E23 冻结目标不一致。"
            )
        targets.append(TargetSpec(
            index=index,
            name=name,
            formula=formula,
            outputs=tuple(map(int, outputs)),
            function_hex=function_hex,
        ))
    return targets


def parameter_blocks() -> tuple[list[ParameterBlock], int]:
    cursor = 0
    blocks = []
    first = Config.WIDTH * Config.INPUT_BITS + Config.WIDTH
    blocks.append(ParameterBlock("first_layer", cursor, cursor + first))
    cursor += first
    middle = Config.WIDTH * Config.WIDTH + Config.WIDTH
    blocks.append(ParameterBlock("middle_layer", cursor, cursor + middle))
    cursor += middle
    output = Config.WIDTH + 1
    blocks.append(ParameterBlock("output_layer", cursor, cursor + output))
    cursor += output
    blocks.append(ParameterBlock("all_parameters", 0, cursor))
    return blocks, cursor


def forward_logits(coordinates: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    count = coordinates.shape[0]
    width = Config.WIDTH
    cursor = 0

    size = width * Config.INPUT_BITS
    first_weight = coordinates[:, cursor:cursor + size].reshape(
        count, width, Config.INPUT_BITS
    ) / math.sqrt(Config.INPUT_BITS)
    cursor += size
    first_bias = coordinates[:, cursor:cursor + width] / math.sqrt(
        Config.INPUT_BITS
    )
    cursor += width

    size = width * width
    middle_weight = coordinates[:, cursor:cursor + size].reshape(
        count, width, width
    ) / math.sqrt(width)
    cursor += size
    middle_bias = coordinates[:, cursor:cursor + width] / math.sqrt(width)
    cursor += width

    output_weight = coordinates[:, cursor:cursor + width].reshape(
        count, 1, width
    ) / math.sqrt(width)
    cursor += width
    output_bias = coordinates[:, cursor:cursor + 1] / math.sqrt(width)

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
def evaluate_target_loss(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        pieces.append(F.binary_cross_entropy_with_logits(
            logits, target[None].expand_as(logits), reduction="none"
        ).mean(dim=1))
    return torch.cat(pieces).reshape(particles.shape[:-1])


@torch.no_grad()
def evaluate_union_loss(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    target_matrix: torch.Tensor,
) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        losses = F.binary_cross_entropy_with_logits(
            logits[:, None, :].expand(-1, len(target_matrix), -1),
            target_matrix[None].expand(len(logits), -1, -1),
            reduction="none",
        ).mean(dim=2)
        pieces.append(losses.min(dim=1).values)
    return torch.cat(pieces).reshape(particles.shape[:-1])


@torch.no_grad()
def evaluate_hard_exact(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces = []
    expected = target.to(torch.bool)
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        pieces.append((logits >= 0).eq(expected[None]).all(dim=1))
    return torch.cat(pieces).reshape(particles.shape[:-1])


def make_target_evaluator(
    inputs: torch.Tensor, target: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda particles: evaluate_target_loss(particles, inputs, target)


def reflect_unit_interval(values: torch.Tensor) -> torch.Tensor:
    folded = torch.remainder(values + 1.0, 4.0)
    return torch.where(folded <= 2.0, folded - 1.0, 3.0 - folded)


def make_generators(
    device: torch.device, mode_index: int
) -> dict[str, torch.Generator]:
    offset = 1_000_003 * mode_index
    result = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    result["prior"].manual_seed(Config.PRIOR_SEED + offset)
    resample_base = (
        Config.PARENT_RESAMPLE_SEED
        if mode_index == 0 else Config.BRANCH_RESAMPLE_SEED
    )
    mutation_base = (
        Config.PARENT_MUTATION_SEED
        if mode_index == 0 else Config.BRANCH_MUTATION_SEED
    )
    result["resample"].manual_seed(resample_base + offset)
    result["mutation"].manual_seed(mutation_base + offset)
    return result


def initialize_prior_state(
    device: torch.device,
    parameter_count: int,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> SMCState:
    particles = torch.empty(
        Config.REPLICAS,
        Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    )
    if Config.REFERENCE_MEASURE == "uniform_cube":
        particles.uniform_(-1.0, 1.0, generator=generator)
    else:
        particles.normal_(
            mean=0.0, std=Config.GAUSSIAN_SIGMA, generator=generator
        )
    losses = evaluator(particles)
    lineages = torch.arange(
        Config.REPLICAS * Config.PARTICLES_PER_REPLICA,
        device=device,
        dtype=torch.int64,
    ).reshape(Config.REPLICAS, Config.PARTICLES_PER_REPLICA)
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume=torch.zeros(
            Config.REPLICAS, dtype=torch.float64, device=device
        ),
        proposal_scales=list(Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=float("inf"),
        threshold_index=0,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
    )


def choose_threshold(
    state: SMCState, requested: float
) -> tuple[float, bool]:
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=1
    )
    threshold = max(float(requested), float(quantiles.max().item()))
    if math.isfinite(state.current_threshold):
        threshold = min(threshold, state.current_threshold)
    return threshold, threshold <= requested + 1e-12


@torch.no_grad()
def resample_state(
    state: SMCState,
    threshold: float,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> np.ndarray:
    survival = np.zeros(Config.REPLICAS, dtype=np.float64)
    particles = torch.empty_like(state.particles)
    lineages = torch.empty_like(state.lineages)
    for replica in range(Config.REPLICAS):
        valid = torch.nonzero(
            state.losses[replica] <= threshold + Config.LOSS_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if not len(valid):
            raise RuntimeError(
                f"replica={replica} 在 threshold={threshold:.8g} 无幸存粒子"
            )
        survival[replica] = len(valid) / Config.PARTICLES_PER_REPLICA
        selected = valid[torch.randint(
            len(valid),
            (Config.PARTICLES_PER_REPLICA,),
            generator=generator,
            device=state.particles.device,
        )]
        particles[replica] = state.particles[replica, selected]
        lineages[replica] = state.lineages[replica, selected]
    state.particles = particles
    state.lineages = lineages
    state.losses = evaluator(particles)
    state.log_volume += torch.log(torch.as_tensor(
        survival, device=state.log_volume.device, dtype=torch.float64
    ))
    return survival


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    scale: float,
    threshold: float,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> float:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    if Config.REFERENCE_MEASURE == "uniform_cube":
        noise = torch.randn(
            current.shape,
            device=current.device,
            dtype=current.dtype,
            generator=generator,
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
    else:
        rho = min(max(float(scale), 0.0), 0.999999)
        noise = torch.empty_like(current).normal_(
            mean=0.0,
            std=Config.GAUSSIAN_SIGMA,
            generator=generator,
        )
        local = math.sqrt(1.0 - rho * rho) * current + rho * noise
    proposal[..., block.start:block.stop] = local
    proposal_loss = evaluator(proposal)
    accept = proposal_loss <= threshold + Config.LOSS_TOLERANCE
    flat_accept = accept.reshape(-1)
    state.particles.reshape(-1, state.particles.shape[-1])[flat_accept] = (
        proposal.reshape(-1, proposal.shape[-1])[flat_accept]
    )
    state.losses.reshape(-1)[flat_accept] = proposal_loss.reshape(-1)[
        flat_accept
    ]
    return float(accept.float().mean().item())


def rejuvenate(
    state: SMCState,
    blocks: Sequence[ParameterBlock],
    threshold: float,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> dict[str, float]:
    scales = list(state.proposal_scales)
    for _ in range(Config.ADAPT_SWEEPS):
        for index, block in enumerate(blocks):
            acceptance = mutate_block(
                state, block, scales[index], threshold, evaluator, generator
            )
            scales[index] *= math.exp(
                Config.ADAPT_RATE
                * (acceptance - Config.TARGET_ACCEPTANCE)
            )
            scales[index] = min(max(
                scales[index], Config.MIN_PROPOSAL_SCALE
            ), Config.MAX_PROPOSAL_SCALE)
    state.proposal_scales = scales

    totals = np.zeros(len(blocks), dtype=np.float64)
    for _ in range(Config.MUTATION_SWEEPS):
        for index, block in enumerate(blocks):
            totals[index] += mutate_block(
                state, block, scales[index], threshold, evaluator, generator
            )
    return {
        f"acceptance_{block.name}": float(
            totals[index] / Config.MUTATION_SWEEPS
        )
        for index, block in enumerate(blocks)
    }


def state_payload(
    state: SMCState,
    mode: str,
    generators: dict[str, torch.Generator],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "config": config_payload(),
        "particles": state.particles.detach().cpu(),
        "losses": state.losses.detach().cpu(),
        "lineages": state.lineages.detach().cpu(),
        "log_volume": state.log_volume.detach().cpu(),
        "proposal_scales": state.proposal_scales,
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "threshold_rows": state.threshold_rows,
        "replica_rows": state.replica_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }


def save_checkpoint(
    directory: Path,
    state: SMCState,
    mode: str,
    generators: dict[str, torch.Generator],
) -> None:
    temporary = directory / "checkpoint.tmp.pt"
    torch.save(state_payload(state, mode, generators), temporary)
    destination = directory / "checkpoint.pt"
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def load_checkpoint(
    directory: Path,
    mode: str,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> SMCState:
    payload = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    if payload["mode"] != mode:
        raise RuntimeError(f"checkpoint mode 不匹配：{mode}")
    saved = payload["config"]
    current = config_payload()
    for key in (
        "INPUT_BITS", "WIDTH", "HIDDEN_LAYERS", "TARGET_NAMES",
        "RANDOM_TARGET_SEED", "REPLICAS", "PARTICLES_PER_REPLICA",
        "PARENT_THRESHOLD", "SURVIVAL_QUANTILE",
        "REFERENCE_MEASURE", "GAUSSIAN_SIGMA",
    ):
        if saved.get(key) != current.get(key):
            raise RuntimeError(f"checkpoint 配置不匹配：{key}")
    saved_thresholds = list(saved["TARGET_THRESHOLDS"])
    current_thresholds = list(current["TARGET_THRESHOLDS"])
    if current_thresholds[:len(saved_thresholds)] != saved_thresholds:
        raise RuntimeError("新阈值必须以 checkpoint 阈值序列为前缀。")
    for name, generator in generators.items():
        if name in payload["generator_states"]:
            generator.set_state(
                payload["generator_states"][name].to(torch.uint8)
            )
    return SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume=payload["log_volume"].to(
            device=device, dtype=torch.float64
        ),
        proposal_scales=list(map(float, payload["proposal_scales"])),
        current_threshold=float(payload["current_threshold"]),
        threshold_index=int(payload["threshold_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        threshold_rows=list(payload["threshold_rows"]),
        replica_rows=list(payload["replica_rows"]),
    )


def prepare_output(targets: Sequence[TargetSpec]) -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("parent", *Config.TARGET_NAMES):
        (output / name).mkdir(exist_ok=True)

    protocol = {
        "protocol": "e23_8bit_mismatch_joint_gaussian_bridge_v2",
        "created_before_sampling": True,
        "config": config_payload(),
        "targets": [json_ready(target.__dict__) for target in targets],
        "expected_pairs": Config.EXPECTED_PAIRS,
        "transition_brackets": Config.TRANSITION_BRACKETS,
        "agreement99_posthoc_diagnostic": Config.AGREEMENT99_DIAGNOSTIC,
        "ratio_sign": "log(V_easy/V_hard)>0 supports n50/n90 ordering",
        "resolution_boundary": (
            "n50/n90 are grid-bracketed or right-censored; small differences "
            "inside overlapping intervals are not preregistered contradictions"
        ),
        "parent_event": "min_target_full_BCE<=PARENT_THRESHOLD",
        "reference_measure": (
            "independent Gaussian coordinates with sigma=1/sqrt(3), "
            "matched to Uniform[-1,1] variance"
            if Config.REFERENCE_MEASURE == "gaussian_matched_variance"
            else "independent uniform[-1,1] coordinates"
        ),
    }
    canonical = json.dumps(
        json_ready(protocol), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    protocol["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = output / "preregistered_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved_config = saved["config"]
        current_config = protocol["config"]
        for key in (
            "INPUT_BITS", "WIDTH", "HIDDEN_LAYERS", "TARGET_NAMES",
            "RANDOM_TARGET_SEED", "EXPECTED_PAIRS", "PARENT_THRESHOLD",
            "REFERENCE_MEASURE", "GAUSSIAN_SIGMA",
            "REPLICAS", "PARTICLES_PER_REPLICA", "SURVIVAL_QUANTILE",
        ):
            if saved_config.get(key) != current_config.get(key):
                raise RuntimeError(f"已有协议配置不匹配：{key}")
        old_thresholds = list(saved_config["TARGET_THRESHOLDS"])
        if list(current_config["TARGET_THRESHOLDS"])[:len(old_thresholds)] != (
            old_thresholds
        ):
            raise RuntimeError("新 TARGET_THRESHOLDS 不是旧协议的扩展。")
    write_json(path, protocol)
    return output


def record_threshold(
    directory: Path,
    state: SMCState,
    mode: str,
    threshold: float,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if any(math.isclose(
        float(row["threshold"]), threshold, abs_tol=1e-12
    ) for row in state.threshold_rows):
        return
    exact = evaluate_hard_exact(state.particles, inputs, target)
    logs = state.log_volume.detach().cpu().numpy()
    losses = state.losses.detach().cpu().numpy()
    exact_np = exact.detach().cpu().numpy()
    for replica in range(Config.REPLICAS):
        state.replica_rows.append({
            "mode": mode,
            "threshold": threshold,
            "replica": replica,
            "estimated_log_volume": float(logs[replica]),
            "estimated_log10_volume": float(logs[replica] / math.log(10)),
            "hard_exact_fraction": float(exact_np[replica].mean()),
            "unique_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        })
    state.threshold_rows.append({
        "mode": mode,
        "threshold": threshold,
        "depth": -math.log(threshold),
        "level": state.level,
        "estimated_log_volume_median": float(np.median(logs)),
        "estimated_log_volume_min": float(np.min(logs)),
        "estimated_log_volume_max": float(np.max(logs)),
        "hard_exact_fraction": float(exact_np.mean()),
        "hard_exact_guaranteed": bool(
            threshold < math.log(2.0) / (2 ** Config.INPUT_BITS)
        ),
        "loss_min": float(losses.min()),
        "loss_median": float(np.median(losses)),
        "loss_max": float(losses.max()),
    })
    write_state_artifacts(directory, state, "running")


def write_state_artifacts(
    directory: Path, state: SMCState, status: str
) -> None:
    write_csv(directory / "levels.csv", state.level_rows)
    write_csv(directory / "volume_curve.csv", state.threshold_rows)
    write_csv(directory / "replica_volume_curve.csv", state.replica_rows)
    write_json(directory / "summary.json", {
        "status": status,
        "level": state.level,
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "completed_thresholds": len(state.threshold_rows),
        "requested_thresholds": len(Config.TARGET_THRESHOLDS),
    })


def run_parent(
    output: Path,
    device: torch.device,
    parameter_count: int,
    blocks: Sequence[ParameterBlock],
    evaluator: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[SMCState, dict[str, torch.Generator]]:
    directory = output / "parent"
    generators = make_generators(device, 0)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() and Config.RESUME:
        state = load_checkpoint(directory, "parent", device, generators)
        print(
            f"[parent] resume level={state.level} "
            f"eps={state.current_threshold:.8g}", flush=True
        )
    else:
        state = initialize_prior_state(
            device, parameter_count, evaluator, generators["prior"]
        )

    if state.threshold_rows:
        return state, generators

    status = "running"
    try:
        while state.level < Config.MAX_LEVELS_PARENT:
            previous = state.current_threshold
            threshold, reached = choose_threshold(
                state, Config.PARENT_THRESHOLD
            )
            if (
                math.isfinite(previous)
                and threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reached
            ):
                raise RuntimeError("parent threshold 停止下降。")
            survival = resample_state(
                state, threshold, evaluator, generators["resample"]
            )
            mutation = rejuvenate(
                state, blocks, threshold, evaluator, generators["mutation"]
            )
            state.level += 1
            state.current_threshold = threshold
            row = {
                "mode": "parent",
                "level": state.level,
                "threshold": threshold,
                "survival_min": float(survival.min()),
                "survival_median": float(np.median(survival)),
                "survival_max": float(survival.max()),
                "log_volume_median": float(np.median(
                    state.log_volume.detach().cpu().numpy()
                )),
                **mutation,
            }
            state.level_rows.append(row)
            if reached or state.level % Config.LOG_EVERY_LEVELS == 0:
                print(
                    f"[parent] level={state.level} eps={threshold:.7g} "
                    f"log10V~{row['log_volume_median']/math.log(10):.2f}",
                    flush=True,
                )
            if reached:
                logs = state.log_volume.detach().cpu().numpy()
                state.threshold_rows.append({
                    "mode": "parent",
                    "threshold": Config.PARENT_THRESHOLD,
                    "estimated_log_volume_median": float(np.median(logs)),
                    "estimated_log_volume_min": float(logs.min()),
                    "estimated_log_volume_max": float(logs.max()),
                })
                status = "completed"
                break
            if state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(directory, state, "parent", generators)
    finally:
        save_checkpoint(directory, state, "parent", generators)
        write_state_artifacts(directory, state, status)
    if status != "completed":
        raise RuntimeError(f"共同父系综未完成：{status}")
    return state, generators


def initialize_branch(
    parent: SMCState,
    target: TargetSpec,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> tuple[SMCState, list[dict[str, Any]]]:
    losses = evaluator(parent.particles)
    particles = torch.empty_like(parent.particles)
    lineages = torch.empty_like(parent.lineages)
    logs = parent.log_volume.clone()
    rows = []
    for replica in range(Config.REPLICAS):
        valid = torch.nonzero(
            losses[replica]
            <= Config.PARENT_THRESHOLD + Config.LOSS_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if not len(valid):
            raise RuntimeError(
                f"{target.name} 在 parent replica={replica} 无成员"
            )
        probability = len(valid) / Config.PARTICLES_PER_REPLICA
        selected = valid[torch.randint(
            len(valid),
            (Config.PARTICLES_PER_REPLICA,),
            generator=generator,
            device=parent.particles.device,
        )]
        particles[replica] = parent.particles[replica, selected]
        lineages[replica] = parent.lineages[replica, selected]
        logs[replica] += math.log(probability)
        rows.append({
            "mode": target.name,
            "replica": replica,
            "parent_threshold": Config.PARENT_THRESHOLD,
            "member_count": len(valid),
            "conditional_probability": probability,
            "parent_log_volume": float(parent.log_volume[replica]),
            "branch_log_volume": float(logs[replica]),
        })
    return SMCState(
        particles=particles,
        losses=evaluator(particles),
        lineages=lineages,
        log_volume=logs,
        proposal_scales=list(Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=Config.PARENT_THRESHOLD,
        threshold_index=1,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
    ), rows


def load_or_initialize_branch(
    output: Path,
    parent: SMCState,
    target: TargetSpec,
    inputs: torch.Tensor,
    target_tensor: torch.Tensor,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[SMCState, dict[str, torch.Generator]]:
    directory = output / target.name
    generators = make_generators(parent.particles.device, target.index + 1)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() and Config.RESUME:
        state = load_checkpoint(
            directory, target.name, parent.particles.device, generators
        )
        print(
            f"[{target.name}] resume level={state.level} "
            f"eps={state.current_threshold:.8g} index={state.threshold_index}",
            flush=True,
        )
    else:
        state, rows = initialize_branch(
            parent, target, evaluator, generators["resample"]
        )
        write_csv(directory / "parent_membership.csv", rows)
        record_threshold(
            directory, state, target.name,
            Config.PARENT_THRESHOLD, inputs, target_tensor,
        )
        save_checkpoint(directory, state, target.name, generators)
    return state, generators


def advance_branch(
    output: Path,
    state: SMCState,
    target: TargetSpec,
    inputs: torch.Tensor,
    target_tensor: torch.Tensor,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    blocks: Sequence[ParameterBlock],
    generators: dict[str, torch.Generator],
    requested_index: int,
) -> None:
    directory = output / target.name
    while (
        state.threshold_index <= requested_index
        and state.level < Config.MAX_LEVELS_BRANCH
    ):
        requested = float(Config.TARGET_THRESHOLDS[state.threshold_index])
        previous = state.current_threshold
        threshold, reached = choose_threshold(state, requested)
        if (
            math.isfinite(previous)
            and threshold >= previous - Config.MIN_LEVEL_DECREMENT
            and not reached
        ):
            raise RuntimeError(
                f"{target.name} threshold 停止下降：{previous:.9g}"
            )
        survival = resample_state(
            state, threshold, evaluator, generators["resample"]
        )
        mutation = rejuvenate(
            state, blocks, threshold, evaluator, generators["mutation"]
        )
        state.level += 1
        state.current_threshold = threshold
        logs = state.log_volume.detach().cpu().numpy()
        row = {
            "mode": target.name,
            "level": state.level,
            "threshold": threshold,
            "requested": requested,
            "survival_min": float(survival.min()),
            "survival_median": float(np.median(survival)),
            "survival_max": float(survival.max()),
            "log_volume_median": float(np.median(logs)),
            "loss_min": float(state.losses.min()),
            "loss_median": float(state.losses.median()),
            **mutation,
        }
        state.level_rows.append(row)
        if reached or state.level % Config.LOG_EVERY_LEVELS == 0:
            print(
                f"[{target.name}] level={state.level:>5} "
                f"eps={threshold:.7g} "
                f"log10V~{row['log_volume_median']/math.log(10):.2f}",
                flush=True,
            )
        if reached:
            record_threshold(
                directory, state, target.name,
                requested, inputs, target_tensor,
            )
            state.threshold_index += 1
            save_checkpoint(directory, state, target.name, generators)
        elif state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
            save_checkpoint(directory, state, target.name, generators)
    if state.threshold_index <= requested_index:
        raise RuntimeError(f"{target.name} 达到 MAX_LEVELS_BRANCH。")


def pairwise_diagnostics(
    output: Path,
    states: dict[str, SMCState],
) -> dict[str, Any]:
    volume_maps: dict[str, dict[tuple[float, int], float]] = {}
    for name, state in states.items():
        volume_maps[name] = {
            (round(float(row["threshold"]), 14), int(row["replica"])):
            float(row["estimated_log_volume"])
            for row in state.replica_rows
        }

    ratio_rows = []
    rate_rows = []
    pair_summaries = []
    for easy, hard, easy_transition, hard_transition in Config.EXPECTED_PAIRS:
        keys = sorted(
            set(volume_maps[easy]) & set(volume_maps[hard]),
            key=lambda item: (-item[0], item[1]),
        )
        by_threshold: dict[float, list[float]] = {}
        for threshold, replica in keys:
            ratio = (
                volume_maps[easy][(threshold, replica)]
                - volume_maps[hard][(threshold, replica)]
            )
            by_threshold.setdefault(threshold, []).append(ratio)
        ordered = sorted(by_threshold, reverse=True)
        local_pair_rows = []
        for threshold in ordered:
            values = np.asarray(by_threshold[threshold], dtype=np.float64)
            row = {
                "easy": easy,
                "hard": hard,
                "easy_transition": easy_transition,
                "hard_transition": hard_transition,
                "easy_transition_brackets": Config.TRANSITION_BRACKETS[easy],
                "hard_transition_brackets": Config.TRANSITION_BRACKETS[hard],
                "threshold": threshold,
                "depth": -math.log(threshold),
                "log_volume_ratio_easy_over_hard_median": float(
                    np.median(values)
                ),
                "log_volume_ratio_min": float(values.min()),
                "log_volume_ratio_max": float(values.max()),
                "fraction_replicas_ratio_positive": float(
                    np.mean(values > 0)
                ),
            }
            ratio_rows.append(row)
            local_pair_rows.append((threshold, values, row))
        for index in range(1, len(local_pair_rows)):
            previous_threshold, previous_values, _ = local_pair_rows[index - 1]
            threshold, values, _ = local_pair_rows[index]
            delta_depth = -math.log(threshold) + math.log(previous_threshold)
            rates = (values - previous_values) / delta_depth
            rate_rows.append({
                "easy": easy,
                "hard": hard,
                "threshold_high": previous_threshold,
                "threshold_low": threshold,
                "depth_mid": -0.5 * math.log(
                    previous_threshold * threshold
                ),
                "ratio_growth_rate_median": float(np.median(rates)),
                "ratio_growth_rate_min": float(rates.min()),
                "ratio_growth_rate_max": float(rates.max()),
                "fraction_replicas_rate_positive": float(
                    np.mean(rates > 0)
                ),
            })

        hard_bound = math.log(2.0) / (2 ** Config.INPUT_BITS)
        deep_rows = [
            row for _, _, row in local_pair_rows
            if float(row["threshold"]) < hard_bound
        ]
        pair_rates = [
            row for row in rate_rows
            if row["easy"] == easy and row["hard"] == hard
            and float(row["threshold_low"]) < hard_bound
        ]
        windows = Config.STOP_CONSECUTIVE_WINDOWS
        ratio_ok = (
            len(deep_rows) >= windows
            and all(
                row["fraction_replicas_ratio_positive"] >= 1.0
                for row in deep_rows[-windows:]
            )
        )
        rate_ok = (
            len(pair_rates) >= windows
            and all(
                row["ratio_growth_rate_median"] > 0
                for row in pair_rates[-windows:]
            )
        )
        first_median_cross = next((
            float(threshold)
            for threshold, _, row in local_pair_rows
            if row["log_volume_ratio_easy_over_hard_median"] > 0
        ), None)
        first_all_cross = next((
            float(threshold)
            for threshold, _, row in local_pair_rows
            if row["fraction_replicas_ratio_positive"] >= 1.0
        ), None)
        pair_summaries.append({
            "easy": easy,
            "hard": hard,
            "easy_transition": easy_transition,
            "hard_transition": hard_transition,
            "easy_transition_brackets": Config.TRANSITION_BRACKETS[easy],
            "hard_transition_brackets": Config.TRANSITION_BRACKETS[hard],
            "first_median_crossing_threshold": first_median_cross,
            "first_all_replica_crossing_threshold": first_all_cross,
            "deep_ratio_windows_pass": ratio_ok,
            "deep_rate_windows_pass": rate_ok,
            "pair_robust": ratio_ok and (
                rate_ok if Config.STOP_REQUIRE_POSITIVE_MEDIAN_RATE else True
            ),
            "last_ratio": local_pair_rows[-1][2] if local_pair_rows else None,
            "last_rate": pair_rates[-1] if pair_rates else None,
        })

    write_csv(output / "pairwise_volume_ratios.csv", ratio_rows)
    write_csv(output / "pairwise_ratio_growth_rates.csv", rate_rows)
    summary = {
        "hard_exact_sufficient_threshold": (
            math.log(2.0) / (2 ** Config.INPUT_BITS)
        ),
        "pair_summaries": pair_summaries,
        "all_pairs_robust": bool(
            pair_summaries and all(
                row["pair_robust"] for row in pair_summaries
            )
        ),
    }
    write_json(output / "stopping_diagnostics.json", summary)
    return summary


def save_plots(output: Path, states: dict[str, SMCState]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ratio_path = output / "pairwise_volume_ratios.csv"
    if not ratio_path.exists() or not ratio_path.stat().st_size:
        return
    with ratio_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for index, (easy, hard, _, _) in enumerate(Config.EXPECTED_PAIRS):
        local = sorted(
            [row for row in rows if row["easy"] == easy and row["hard"] == hard],
            key=lambda row: float(row["depth"]),
        )
        depth = np.asarray([float(row["depth"]) for row in local])
        median = np.asarray([
            float(row["log_volume_ratio_easy_over_hard_median"])
            for row in local
        ])
        low = np.asarray([float(row["log_volume_ratio_min"]) for row in local])
        high = np.asarray([float(row["log_volume_ratio_max"]) for row in local])
        axes[index].plot(depth, median, marker="o", ms=3)
        axes[index].fill_between(depth, low, high, alpha=0.2)
        axes[index].axhline(0.0, color="black", ls="--", lw=1)
        axes[index].set_title(f"log V({easy}) / V({hard})")
        axes[index].set_xlabel("depth = -log(epsilon)")
        axes[index].set_ylabel("expected positive")

    for name, state in states.items():
        local = sorted(state.threshold_rows, key=lambda row: row["depth"])
        axes[3].plot(
            [row["depth"] for row in local],
            [row["estimated_log_volume_median"] / math.log(10) for row in local],
            label=name,
        )
    axes[3].set_title("absolute median log10 volume")
    axes[3].set_xlabel("depth = -log(epsilon)")
    axes[3].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "deep_mismatch_bridge.png", dpi=180)
    plt.close(figure)


def create_archive(output: Path) -> Path:
    archive = output.parent / f"{output.name}_package.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            if path.name in {"checkpoint.pt", "checkpoint.tmp.pt"}:
                continue
            handle.write(path, arcname=f"{output.name}/{path.relative_to(output)}")
    return archive


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    targets = build_targets()
    output = prepare_output(targets)

    device = torch.device(Config.DEVICE)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32

    inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    target_tensors = {
        target.name: torch.as_tensor(
            target.outputs, dtype=torch.float32, device=device
        )
        for target in targets
    }
    target_matrix = torch.stack([
        target_tensors[name] for name in Config.TARGET_NAMES
    ])
    blocks, parameter_count = parameter_blocks()

    evaluators = {
        name: make_target_evaluator(inputs, target_tensors[name])
        for name in Config.TARGET_NAMES
    }
    union_evaluator = lambda particles: evaluate_union_loss(
        particles, inputs, target_matrix
    )

    print("=== E23 8-bit mismatch joint deep bridge ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=8->{Config.WIDTH}x2->1 tanh | params={parameter_count} | "
        f"targets={list(Config.TARGET_NAMES)} | replicas={Config.REPLICAS} | "
        f"particles/replica={Config.PARTICLES_PER_REPLICA:,}",
        flush=True,
    )
    print(
        f"reference_measure={Config.REFERENCE_MEASURE} | "
        f"sigma={Config.GAUSSIAN_SIGMA:.9g}", flush=True
    )
    print(f"expected pairs={Config.EXPECTED_PAIRS}", flush=True)
    print(f"result_dir={output}", flush=True)

    parent, parent_generators = run_parent(
        output, device, parameter_count, blocks, union_evaluator
    )
    states: dict[str, SMCState] = {}
    generators: dict[str, dict[str, torch.Generator]] = {}
    target_map = {target.name: target for target in targets}
    for target in targets:
        state, local_generators = load_or_initialize_branch(
            output, parent, target, inputs,
            target_tensors[target.name], evaluators[target.name],
        )
        states[target.name] = state
        generators[target.name] = local_generators

    status = "running"
    try:
        for index in range(1, len(Config.TARGET_THRESHOLDS)):
            for name in Config.TARGET_NAMES:
                target = target_map[name]
                advance_branch(
                    output, states[name], target, inputs,
                    target_tensors[name], evaluators[name], blocks,
                    generators[name], index,
                )
            diagnostic = pairwise_diagnostics(output, states)
            parts = []
            for row in diagnostic["pair_summaries"]:
                ratio = row["last_ratio"] or {}
                parts.append(
                    f"{row['easy']}/{row['hard']}="
                    f"{ratio.get('log_volume_ratio_easy_over_hard_median')}"
                )
            print(
                f"[joint] eps={Config.TARGET_THRESHOLDS[index]:.7g} | "
                + " | ".join(parts)
                + f" | stop={diagnostic['all_pairs_robust']}",
                flush=True,
            )
            if diagnostic["all_pairs_robust"]:
                status = "stopped_all_pairs_robust"
                break
        if status == "running":
            status = "completed_threshold_schedule"
    except KeyboardInterrupt:
        status = "interrupted"
        print("Ctrl-C: 正在保存所有分支 checkpoint。", flush=True)
    finally:
        for name in Config.TARGET_NAMES:
            save_checkpoint(
                output / name, states[name], name, generators[name]
            )
            write_state_artifacts(output / name, states[name], status)
        save_checkpoint(output / "parent", parent, "parent", parent_generators)
        pairwise_diagnostics(output, states)
        save_plots(output, states)
        write_json(output / "summary.json", {
            "status": status,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "parameter_count": parameter_count,
            "target_count": len(targets),
            "completed_threshold_index_min": min(
                state.threshold_index for state in states.values()
            ),
            "last_common_threshold": min(
                state.current_threshold for state in states.values()
            ),
        })
        if Config.PACKAGE_RESULTS:
            archive = create_archive(output)
            print(f"archive={archive}", flush=True)
    print(f"status={status}", flush=True)


if __name__ == "__main__":
    main()
