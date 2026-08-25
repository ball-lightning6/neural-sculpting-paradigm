"""常数规则 leave-one-out 的三路静态 SMC 测量闭环。

本实验只检查静态参数测度是否数值自洽，不涉及 SGD 或泛化机制。固定同一个
4->16x2->1 tanh 网络和初始化参数立方体均匀测度，构造三个 SMC 任务：

1. subset15_zero：除输入0110外，15个训练标签全部为0；
2. full_constant0：16点完整常数零函数0x0000；
3. full_single_exception：只在0110输出1的函数0x0040。

对完整平均 BCE 阈值 epsilon，15样本配对阈值为 delta=16*epsilon/15。
因为逐样本 BCE 非负，事件 B_y={L16_y<=epsilon} 必然包含于
A={L15<=delta}，因此必须满足：

    Vol(B_y) = Vol(A) * P(B_y | A)

以及：

    Vol(B_0) / Vol(B_1) = P(B_0|A) / P(B_1|A).

左侧由两个独立 full-rule SMC 测量；右侧由 subset15 SMC 粒子交叉计算两个
完整 loss 得到。若在共同有效阈值上明显不闭合，则说明至少一个 SMC 估计、
参数测度、loss 归一化或实现存在问题。普通 heldout hard-sign odds 也会记录，
但它不应被误认为 high-margin 完整规则体积比。
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 4
    WIDTH = 16
    HOLDOUT_INDEX = 6  # 0110

    FULL_THRESHOLDS = (
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
    )
    REPLICAS = 8
    PARTICLES_PER_REPLICA = 4_096
    SURVIVAL_QUANTILE = 0.5
    MAX_LEVELS_PER_TASK = 1_200

    ADAPT_SWEEPS = 8
    MUTATION_SWEEPS = 24
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.050, 0.030, 0.050, 0.015)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.30
    REFRESH_PROBABILITY = 0.02
    LOSS_TOLERANCE = 1e-7
    MIN_LEVEL_DECREMENT = 1e-7

    PRIOR_CALIBRATION_SAMPLES = 4_194_304
    PRIOR_CALIBRATION_BATCH = 16_384
    EVAL_MICRO_BATCH = 8_192
    TOP_FUNCTIONS = 12
    CHECKPOINT_EVERY_LEVELS = 1

    PRIOR_SEED = 20261001
    RESAMPLE_SEED = 20261002
    MUTATION_SEED = 20261003
    CALIBRATION_SEED = 20261004
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_constant_leave_one_out_smc_consistency")
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    targets: tuple[int, ...]
    train_indices: tuple[int, ...]
    thresholds: tuple[float, ...]
    target_function_id: int | None


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int


@dataclass
class SMCState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume_fraction: torch.Tensor
    proposal_scales: list[float]
    current_threshold: float
    target_index: int
    level: int
    level_rows: list[dict[str, Any]]
    target_rows: list[dict[str, Any]]
    replica_rows: list[dict[str, Any]]
    top_rows: list[dict[str, Any]]


@dataclass
class TaskResult:
    spec: TaskSpec
    state: SMCState
    status: str
    output_dir: Path


GLOBAL_INPUTS: torch.Tensor
GLOBAL_TARGETS: torch.Tensor
GLOBAL_TRAIN_INDICES: torch.Tensor
GLOBAL_THRESHOLDS: tuple[float, ...]


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


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
        + first_bias[:, None, :]
    )
    hidden = torch.tanh(
        torch.bmm(hidden, middle_weight.transpose(1, 2))
        + middle_bias[:, None, :]
    )
    return (
        torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
        + output_bias
    )


@torch.no_grad()
def evaluate_primary_losses(particles: torch.Tensor) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces: list[torch.Tensor] = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], GLOBAL_INPUTS
        )
        local_logits = logits[:, GLOBAL_TRAIN_INDICES]
        local_targets = GLOBAL_TARGETS[GLOBAL_TRAIN_INDICES][None].expand_as(
            local_logits
        )
        pieces.append(F.binary_cross_entropy_with_logits(
            local_logits,
            local_targets,
            reduction="none",
        ).mean(dim=1))
    return torch.cat(pieces).reshape(particles.shape[:-1])


def reflect_unit_interval(values: torch.Tensor) -> torch.Tensor:
    folded = torch.remainder(values + 1.0, 4.0)
    return torch.where(folded <= 2.0, folded - 1.0, 3.0 - folded)


def initialize_state(
    device: torch.device,
    prior_generator: torch.Generator,
    parameter_count: int,
) -> SMCState:
    particles = torch.empty(
        Config.REPLICAS,
        Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    ).uniform_(-1.0, 1.0, generator=prior_generator)
    losses = evaluate_primary_losses(particles)
    total = Config.REPLICAS * Config.PARTICLES_PER_REPLICA
    lineages = torch.arange(total, device=device, dtype=torch.int64).reshape(
        Config.REPLICAS, Config.PARTICLES_PER_REPLICA
    )
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume_fraction=torch.zeros(
            Config.REPLICAS, device=device, dtype=torch.float64
        ),
        proposal_scales=list(Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=float("inf"),
        target_index=0,
        level=0,
        level_rows=[],
        target_rows=[],
        replica_rows=[],
        top_rows=[],
    )


def choose_next_threshold(state: SMCState) -> tuple[float, bool]:
    target = float(GLOBAL_THRESHOLDS[state.target_index])
    quantiles = torch.quantile(
        state.losses,
        Config.SURVIVAL_QUANTILE,
        dim=1,
    )
    adaptive = float(quantiles.max().item())
    next_threshold = max(target, adaptive)
    if math.isfinite(state.current_threshold):
        next_threshold = min(next_threshold, state.current_threshold)
    reached_target = next_threshold <= target + 1e-12
    return next_threshold, reached_target


@torch.no_grad()
def resample_within_replicas(
    state: SMCState,
    threshold: float,
    generator: torch.Generator,
) -> np.ndarray:
    survival_fractions = np.zeros(Config.REPLICAS, dtype=np.float64)
    new_particles = torch.empty_like(state.particles)
    new_lineages = torch.empty_like(state.lineages)
    for replica in range(Config.REPLICAS):
        survivors = torch.nonzero(
            state.losses[replica] <= threshold + Config.LOSS_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if len(survivors) == 0:
            raise RuntimeError(
                f"副本{replica}在阈值{threshold:.6g}没有幸存粒子。"
            )
        survival_fractions[replica] = (
            len(survivors) / Config.PARTICLES_PER_REPLICA
        )
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
    state.losses = evaluate_primary_losses(state.particles)
    state.log_volume_fraction += torch.log(torch.from_numpy(
        survival_fractions
    ).to(state.log_volume_fraction))
    return survival_fractions


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    scale: float,
    threshold: float,
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
    proposal_losses = evaluate_primary_losses(proposal)
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
    generator: torch.Generator,
) -> dict[str, float]:
    scales = list(state.proposal_scales)
    for _ in range(Config.ADAPT_SWEEPS):
        for block_index, block in enumerate(blocks):
            acceptance = mutate_block(
                state, block, scales[block_index], threshold, generator
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
                state, block, scales[block_index], threshold, generator
            )
    acceptance_mean = acceptance_sum / max(Config.MUTATION_SWEEPS, 1)
    return {
        f"acceptance_{block.name}": float(acceptance_mean[index])
        for index, block in enumerate(blocks)
    }


def checkpoint_payload(
    state: SMCState,
    generators: dict[str, torch.Generator],
    spec: TaskSpec,
) -> dict[str, Any]:
    return {
        "task_definition": task_definition(spec),
        "particles": state.particles.detach().cpu(),
        "losses": state.losses.detach().cpu(),
        "lineages": state.lineages.detach().cpu(),
        "log_volume_fraction": state.log_volume_fraction.detach().cpu(),
        "proposal_scales": state.proposal_scales,
        "current_threshold": state.current_threshold,
        "target_index": state.target_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "target_rows": state.target_rows,
        "replica_rows": state.replica_rows,
        "top_rows": state.top_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }


def save_checkpoint(
    task_dir: Path,
    state: SMCState,
    generators: dict[str, torch.Generator],
    spec: TaskSpec,
) -> None:
    temporary = task_dir / "checkpoint.tmp.pt"
    checkpoint = task_dir / "checkpoint.pt"
    torch.save(checkpoint_payload(state, generators, spec), temporary)
    temporary.replace(checkpoint)


def load_checkpoint(
    task_dir: Path,
    device: torch.device,
    generators: dict[str, torch.Generator],
    spec: TaskSpec,
) -> SMCState:
    payload = torch.load(
        task_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    if payload["task_definition"] != json_ready(task_definition(spec)):
        raise RuntimeError(f"checkpoint task定义不一致：{spec.name}")
    for name, generator in generators.items():
        generator.set_state(
            payload["generator_states"][name].to(
                dtype=torch.uint8, device="cpu"
            )
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
        target_index=int(payload["target_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        target_rows=list(payload["target_rows"]),
        replica_rows=list(payload["replica_rows"]),
        top_rows=list(payload["top_rows"]),
    )


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.FULL_THRESHOLDS = (0.72, 0.68)
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 128
    Config.MAX_LEVELS_PER_TASK = 12
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 2
    Config.PRIOR_CALIBRATION_SAMPLES = 2_048
    Config.PRIOR_CALIBRATION_BATCH = 256
    Config.EVAL_MICRO_BATCH = 256
    Config.TOP_FUNCTIONS = 5
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_constant_leave_one_out_smc_consistency"
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


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


def validate_config() -> None:
    thresholds = tuple(float(value) for value in Config.FULL_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("FULL_THRESHOLDS必须严格递减且不重复。")
    if not 0 <= Config.HOLDOUT_INDEX < 16:
        raise ValueError("HOLDOUT_INDEX必须在[0,15]。")
    if Config.REPLICAS < 2:
        raise ValueError("至少需要两个SMC副本。")
    if Config.INPUT_BITS != 4 or Config.WIDTH != 16:
        raise ValueError("当前一致性测试固定4-bit、width=16。")


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    if output.exists() and not Config.RESUME:
        raise FileExistsError(f"结果目录已存在且RESUME=False：{output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_task_specs() -> tuple[TaskSpec, ...]:
    constant = np.zeros(16, dtype=np.uint8)
    exception = constant.copy()
    exception[Config.HOLDOUT_INDEX] = 1
    all_indices = tuple(range(16))
    subset_indices = tuple(
        index for index in range(16) if index != Config.HOLDOUT_INDEX
    )
    full_thresholds = tuple(float(value) for value in Config.FULL_THRESHOLDS)
    subset_thresholds = tuple(
        float(value * 16.0 / 15.0) for value in full_thresholds
    )
    return (
        TaskSpec(
            name="subset15_zero",
            description="15 zero labels; hold out 0110",
            targets=tuple(int(value) for value in constant),
            train_indices=subset_indices,
            thresholds=subset_thresholds,
            target_function_id=None,
        ),
        TaskSpec(
            name="full_constant0",
            description="16-point constant-zero truth table",
            targets=tuple(int(value) for value in constant),
            train_indices=all_indices,
            thresholds=full_thresholds,
            target_function_id=0x0000,
        ),
        TaskSpec(
            name="full_single_exception",
            description="constant zero plus 0110 -> 1",
            targets=tuple(int(value) for value in exception),
            train_indices=all_indices,
            thresholds=full_thresholds,
            target_function_id=(1 << Config.HOLDOUT_INDEX),
        ),
    )


def make_generators(device: torch.device) -> dict[str, torch.Generator]:
    generators = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    generators["prior"].manual_seed(Config.PRIOR_SEED)
    generators["resample"].manual_seed(Config.RESAMPLE_SEED)
    generators["mutation"].manual_seed(Config.MUTATION_SEED)
    return generators


def task_definition(spec: TaskSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "targets": list(spec.targets),
        "train_indices": list(spec.train_indices),
        "thresholds": list(spec.thresholds),
        "target_function_id": spec.target_function_id,
        "target_function_hex": (
            f"0x{spec.target_function_id:04X}"
            if spec.target_function_id is not None else None
        ),
    }


def validate_task_resume(task_dir: Path, spec: TaskSpec) -> None:
    definition_path = task_dir / "task_definition.json"
    if not definition_path.exists():
        return
    saved = json.loads(definition_path.read_text(encoding="utf-8"))
    if saved != json_ready(task_definition(spec)):
        raise RuntimeError(f"task定义与已有结果不一致：{spec.name}")


@torch.no_grad()
def evaluate_all_observables(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    constant_targets: torch.Tensor,
    exception_targets: torch.Tensor,
    subset_indices: torch.Tensor,
    all_indices: torch.Tensor,
) -> dict[str, np.ndarray]:
    flat = particles.reshape(-1, particles.shape[-1])
    output: dict[str, list[np.ndarray]] = {
        "loss15": [],
        "loss16_constant0": [],
        "loss16_exception": [],
        "heldout_logit": [],
        "function_id": [],
    }
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=particles.device),
        torch.arange(16, dtype=torch.int64, device=particles.device),
    )
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        loss_constant = F.binary_cross_entropy_with_logits(
            logits,
            constant_targets[None].expand_as(logits),
            reduction="none",
        )
        loss_exception = F.binary_cross_entropy_with_logits(
            logits,
            exception_targets[None].expand_as(logits),
            reduction="none",
        )
        function_ids = (
            (logits >= 0).to(torch.int64) * powers[None]
        ).sum(dim=1)
        output["loss15"].append(
            loss_constant[:, subset_indices].mean(dim=1).cpu().numpy()
        )
        output["loss16_constant0"].append(
            loss_constant[:, all_indices].mean(dim=1).cpu().numpy()
        )
        output["loss16_exception"].append(
            loss_exception[:, all_indices].mean(dim=1).cpu().numpy()
        )
        output["heldout_logit"].append(
            logits[:, Config.HOLDOUT_INDEX].cpu().numpy()
        )
        output["function_id"].append(
            function_ids.cpu().numpy().astype(np.uint16)
        )
    shape = particles.shape[:-1]
    return {
        key: np.concatenate(values).reshape(shape)
        for key, values in output.items()
    }


def safe_log_probability(probability: float) -> float | None:
    if probability <= 0 or not math.isfinite(probability):
        return None
    return float(math.log(probability))


def summarize_cross_observables(
    spec: TaskSpec,
    state: SMCState,
    threshold: float,
    observables: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if spec.name == "subset15_zero":
        epsilon = float(threshold * 15.0 / 16.0)
        delta = float(threshold)
    else:
        epsilon = float(threshold)
        delta = float(threshold * 16.0 / 15.0)

    ids = observables["function_id"].astype(np.int64)
    heldout_positive = observables["heldout_logit"] >= 0
    event_constant = observables["loss16_constant0"] <= (
        epsilon + Config.LOSS_TOLERANCE
    )
    event_exception = observables["loss16_exception"] <= (
        epsilon + Config.LOSS_TOLERANCE
    )
    event_subset = observables["loss15"] <= (
        delta + Config.LOSS_TOLERANCE
    )

    replica_rows: list[dict[str, Any]] = []
    for replica in range(Config.REPLICAS):
        q0 = float(event_constant[replica].mean())
        q1 = float(event_exception[replica].mean())
        p1 = float(heldout_positive[replica].mean())
        log_volume = float(state.log_volume_fraction[replica].item())
        replica_rows.append({
            "task": spec.name,
            "threshold": threshold,
            "paired_epsilon": epsilon,
            "paired_delta": delta,
            "replica": replica,
            "estimated_log_volume": log_volume,
            "estimated_volume": float(math.exp(log_volume)),
            "q_full_constant_given_subset": q0,
            "q_full_exception_given_subset": q1,
            "log_q_full_constant_given_subset": safe_log_probability(q0),
            "log_q_full_exception_given_subset": safe_log_probability(q1),
            "cross_log_volume_constant": (
                log_volume + math.log(q0) if q0 > 0 else None
            ),
            "cross_log_volume_exception": (
                log_volume + math.log(q1) if q1 > 0 else None
            ),
            "heldout_hard_probability_zero": 1.0 - p1,
            "heldout_hard_probability_one": p1,
            "full_events_overlap_probability": float(
                (event_constant[replica] & event_exception[replica]).mean()
            ),
            "paired_subset_inclusion_violation_fraction": float(
                (~event_subset[replica]).mean()
            ),
            "constant_function_probability": float(
                (ids[replica] == 0x0000).mean()
            ),
            "single_exception_function_probability": float(
                (ids[replica] == (1 << Config.HOLDOUT_INDEX)).mean()
            ),
            "unique_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        })

    def finite_values(key: str) -> list[float]:
        return [
            float(row[key]) for row in replica_rows
            if row[key] is not None and math.isfinite(float(row[key]))
        ]

    aggregate: dict[str, Any] = {
        "task": spec.name,
        "threshold": threshold,
        "paired_epsilon": epsilon,
        "paired_delta": delta,
        "particle_count": int(ids.size),
        "loss_min": float(state.losses.min().item()),
        "loss_median": float(state.losses.median().item()),
        "loss_max": float(state.losses.max().item()),
        "estimated_log_volume_median": float(
            state.log_volume_fraction.median().item()
        ),
        "estimated_log_volume_min": float(
            state.log_volume_fraction.min().item()
        ),
        "estimated_log_volume_max": float(
            state.log_volume_fraction.max().item()
        ),
        "estimated_volume_median": float(torch.exp(
            state.log_volume_fraction
        ).median().item()),
        "heldout_hard_probability_zero": float((~heldout_positive).mean()),
        "heldout_hard_probability_one": float(heldout_positive.mean()),
        "constant_function_probability": float((ids == 0x0000).mean()),
        "single_exception_function_probability": float(
            (ids == (1 << Config.HOLDOUT_INDEX)).mean()
        ),
        "function_support": int(len(np.unique(ids))),
        "q_full_constant_given_subset": float(event_constant.mean()),
        "q_full_exception_given_subset": float(event_exception.mean()),
        "full_events_overlap_probability": float(
            (event_constant & event_exception).mean()
        ),
        "paired_subset_inclusion_violation_fraction": float(
            (~event_subset).mean()
        ),
    }
    for key in (
        "q_full_constant_given_subset",
        "q_full_exception_given_subset",
        "cross_log_volume_constant",
        "cross_log_volume_exception",
    ):
        values = finite_values(key)
        aggregate[f"{key}_replica_median"] = (
            float(np.median(values)) if values else None
        )
        aggregate[f"{key}_replica_min"] = (
            float(np.min(values)) if values else None
        )
        aggregate[f"{key}_replica_max"] = (
            float(np.max(values)) if values else None
        )
    return aggregate, replica_rows


def append_top_functions(
    rows: list[dict[str, Any]],
    task: str,
    threshold: float,
    ids: np.ndarray,
) -> None:
    counts = np.bincount(ids.reshape(-1).astype(np.int64), minlength=65_536)
    total = counts.sum()
    for rank, function_id in enumerate(
        np.argsort(counts)[::-1][:Config.TOP_FUNCTIONS], start=1
    ):
        count = int(counts[function_id])
        if count == 0:
            break
        rows.append({
            "task": task,
            "threshold": threshold,
            "rank": rank,
            "function_id": int(function_id),
            "function_hex": f"0x{int(function_id):04X}",
            "probability": float(count / total),
            "count": count,
        })


def record_task_threshold(
    task_dir: Path,
    spec: TaskSpec,
    state: SMCState,
    threshold: float,
    inputs: torch.Tensor,
    constant_targets: torch.Tensor,
    exception_targets: torch.Tensor,
    subset_indices: torch.Tensor,
    all_indices: torch.Tensor,
) -> None:
    observables = evaluate_all_observables(
        state.particles,
        inputs,
        constant_targets,
        exception_targets,
        subset_indices,
        all_indices,
    )
    aggregate, replica_rows = summarize_cross_observables(
        spec, state, threshold, observables
    )
    aggregate["level"] = state.level
    state.target_rows.append(aggregate)
    state.replica_rows.extend(replica_rows)
    append_top_functions(
        state.top_rows,
        spec.name,
        threshold,
        observables["function_id"],
    )

    snapshot_name = f"snapshot_threshold_{threshold:.6f}".replace(".", "p")
    np.savez_compressed(
        task_dir / f"{snapshot_name}.npz",
        task=np.asarray(spec.name),
        threshold=np.asarray(threshold),
        paired_epsilon=np.asarray(aggregate["paired_epsilon"]),
        paired_delta=np.asarray(aggregate["paired_delta"]),
        function_ids=observables["function_id"],
        primary_losses=state.losses.detach().cpu().numpy(),
        loss15=observables["loss15"],
        loss16_constant0=observables["loss16_constant0"],
        loss16_exception=observables["loss16_exception"],
        heldout_logits=observables["heldout_logit"],
        lineages=state.lineages.detach().cpu().numpy(),
        log_volume_fraction=state.log_volume_fraction.detach().cpu().numpy(),
    )
    print(
        f"[{spec.name}] TARGET={threshold:.6f} | "
        f"logV~{aggregate['estimated_log_volume_median']:.3f} | "
        f"hard P(0/1)={aggregate['heldout_hard_probability_zero']:.2%}/"
        f"{aggregate['heldout_hard_probability_one']:.2%} | "
        f"q0/q1={aggregate['q_full_constant_given_subset']:.3e}/"
        f"{aggregate['q_full_exception_given_subset']:.3e} | "
        f"support={aggregate['function_support']}",
        flush=True,
    )


def write_task_artifacts(
    task_dir: Path,
    spec: TaskSpec,
    state: SMCState,
    status: str,
) -> None:
    write_csv(task_dir / "levels.csv", state.level_rows)
    write_csv(task_dir / "measurements.csv", state.target_rows)
    write_csv(task_dir / "replica_measurements.csv", state.replica_rows)
    write_csv(task_dir / "top_functions.csv", state.top_rows)
    write_json(task_dir / "task_definition.json", task_definition(spec))
    write_json(task_dir / "summary.json", {
        "status": status,
        "task": task_definition(spec),
        "completed_thresholds": state.target_index,
        "requested_thresholds": len(spec.thresholds),
        "current_threshold": state.current_threshold,
        "level": state.level,
        "measurements": state.target_rows,
    })


def run_task(
    root: Path,
    spec: TaskSpec,
    device: torch.device,
    inputs: torch.Tensor,
    constant_targets: torch.Tensor,
    exception_targets: torch.Tensor,
    subset_indices: torch.Tensor,
    all_indices: torch.Tensor,
) -> TaskResult:
    global GLOBAL_INPUTS, GLOBAL_TARGETS, GLOBAL_TRAIN_INDICES, GLOBAL_THRESHOLDS

    task_dir = root / "tasks" / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)
    validate_task_resume(task_dir, spec)
    write_json(task_dir / "task_definition.json", task_definition(spec))
    GLOBAL_INPUTS = inputs
    GLOBAL_TARGETS = (
        exception_targets
        if spec.name == "full_single_exception"
        else constant_targets
    )
    GLOBAL_TRAIN_INDICES = torch.tensor(
        spec.train_indices, dtype=torch.int64, device=device
    )
    GLOBAL_THRESHOLDS = spec.thresholds
    blocks, parameter_count = parameter_blocks(Config.WIDTH)
    generators = make_generators(device)

    checkpoint = task_dir / "checkpoint.pt"
    if Config.RESUME and checkpoint.exists():
        state = load_checkpoint(task_dir, device, generators, spec)
        print(
            f"[{spec.name}] 恢复checkpoint：level={state.level} | "
            f"threshold={state.current_threshold:.6g} | "
            f"target_index={state.target_index}",
            flush=True,
        )
    else:
        state = initialize_state(
            device, generators["prior"], parameter_count
        )
        save_checkpoint(task_dir, state, generators, spec)

    started = time.perf_counter()
    status = "running"
    try:
        while (
            state.target_index < len(spec.thresholds)
            and state.level < Config.MAX_LEVELS_PER_TASK
        ):
            previous = state.current_threshold
            next_threshold, reaches_target = choose_next_threshold(state)
            if (
                math.isfinite(previous)
                and next_threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reaches_target
            ):
                raise RuntimeError(
                    f"[{spec.name}] SMC阈值停止下降，需要检查混合。"
                )
            survival = resample_within_replicas(
                state, next_threshold, generators["resample"]
            )
            mutation = rejuvenate(
                state, blocks, next_threshold, generators["mutation"]
            )
            state.level += 1
            state.current_threshold = next_threshold
            elapsed = time.perf_counter() - started
            row: dict[str, Any] = {
                "task": spec.name,
                "level": state.level,
                "threshold": next_threshold,
                "reaches_requested_target": reaches_target,
                "next_requested_target": float(spec.thresholds[state.target_index]),
                "survival_fraction_min": float(np.min(survival)),
                "survival_fraction_median": float(np.median(survival)),
                "survival_fraction_max": float(np.max(survival)),
                "loss_min": float(state.losses.min().item()),
                "loss_median": float(state.losses.median().item()),
                "loss_max": float(state.losses.max().item()),
                "estimated_log_volume_median": float(
                    state.log_volume_fraction.median().item()
                ),
                "elapsed_seconds": elapsed,
                "proposal_scales": list(state.proposal_scales),
            }
            row.update(mutation)
            state.level_rows.append(row)
            print(
                f"[{spec.name}] level={state.level:>4} | "
                f"eps={next_threshold:.6f} | "
                f"survive={np.median(survival):.1%} | "
                f"loss med={state.losses.median().item():.6f} | "
                f"logV~{row['estimated_log_volume_median']:.2f} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            if reaches_target:
                threshold = float(spec.thresholds[state.target_index])
                record_task_threshold(
                    task_dir,
                    spec,
                    state,
                    threshold,
                    inputs,
                    constant_targets,
                    exception_targets,
                    subset_indices,
                    all_indices,
                )
                state.target_index += 1
                write_task_artifacts(task_dir, spec, state, "running")

            if state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(task_dir, state, generators, spec)

        status = (
            "completed"
            if state.target_index == len(spec.thresholds)
            else "stopped_max_levels"
        )
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\n[{spec.name}] 收到中断，保存当前状态。", flush=True)
    finally:
        save_checkpoint(task_dir, state, generators, spec)
        write_task_artifacts(task_dir, spec, state, status)
    return TaskResult(spec, state, status, task_dir)


@torch.no_grad()
def run_prior_calibration(
    root: Path,
    device: torch.device,
    inputs: torch.Tensor,
    constant_targets: torch.Tensor,
    exception_targets: torch.Tensor,
    subset_indices: torch.Tensor,
) -> list[dict[str, Any]]:
    path = root / "prior_calibration.csv"
    if path.exists() and Config.RESUME:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if rows and int(rows[0]["sample_count"]) == Config.PRIOR_CALIBRATION_SAMPLES:
            print("复用已有prior calibration。", flush=True)
            return [dict(row) for row in rows]

    _, parameter_count = parameter_blocks(Config.WIDTH)
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.CALIBRATION_SEED)
    full_thresholds = np.asarray(Config.FULL_THRESHOLDS, dtype=np.float64)
    subset_thresholds = full_thresholds * 16.0 / 15.0
    counts_a = np.zeros(len(full_thresholds), dtype=np.int64)
    counts_b0 = np.zeros(len(full_thresholds), dtype=np.int64)
    counts_b1 = np.zeros(len(full_thresholds), dtype=np.int64)
    counts_hard0 = np.zeros(len(full_thresholds), dtype=np.int64)
    counts_hard1 = np.zeros(len(full_thresholds), dtype=np.int64)
    completed = 0
    while completed < Config.PRIOR_CALIBRATION_SAMPLES:
        count = min(
            Config.PRIOR_CALIBRATION_BATCH,
            Config.PRIOR_CALIBRATION_SAMPLES - completed,
        )
        particles = torch.empty(
            count, parameter_count, device=device
        ).uniform_(-1.0, 1.0, generator=generator)
        logits = forward_logits(particles, inputs)
        loss0 = F.binary_cross_entropy_with_logits(
            logits,
            constant_targets[None].expand_as(logits),
            reduction="none",
        )
        loss1 = F.binary_cross_entropy_with_logits(
            logits,
            exception_targets[None].expand_as(logits),
            reduction="none",
        )
        l15 = loss0[:, subset_indices].mean(dim=1).cpu().numpy()
        l0 = loss0.mean(dim=1).cpu().numpy()
        l1 = loss1.mean(dim=1).cpu().numpy()
        heldout_positive = (
            logits[:, Config.HOLDOUT_INDEX] >= 0
        ).cpu().numpy()
        for index, (epsilon, delta) in enumerate(
            zip(full_thresholds, subset_thresholds)
        ):
            event_a = l15 <= delta + Config.LOSS_TOLERANCE
            counts_a[index] += int(event_a.sum())
            counts_b0[index] += int((l0 <= epsilon + Config.LOSS_TOLERANCE).sum())
            counts_b1[index] += int((l1 <= epsilon + Config.LOSS_TOLERANCE).sum())
            counts_hard0[index] += int((event_a & (~heldout_positive)).sum())
            counts_hard1[index] += int((event_a & heldout_positive).sum())
        completed += count
        if completed % max(Config.PRIOR_CALIBRATION_SAMPLES // 8, 1) == 0:
            print(
                f"prior calibration {completed:,}/"
                f"{Config.PRIOR_CALIBRATION_SAMPLES:,}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    total = Config.PRIOR_CALIBRATION_SAMPLES
    for index, (epsilon, delta) in enumerate(
        zip(full_thresholds, subset_thresholds)
    ):
        a = counts_a[index]
        b0 = counts_b0[index]
        b1 = counts_b1[index]
        rows.append({
            "sample_count": total,
            "epsilon_full": float(epsilon),
            "delta_subset": float(delta),
            "count_subset": int(a),
            "count_full_constant": int(b0),
            "count_full_exception": int(b1),
            "volume_subset": float(a / total),
            "volume_full_constant": float(b0 / total),
            "volume_full_exception": float(b1 / total),
            "q_constant_given_subset": float(b0 / a) if a else None,
            "q_exception_given_subset": float(b1 / a) if a else None,
            "hard_zero_given_subset": (
                float(counts_hard0[index] / a) if a else None
            ),
            "hard_one_given_subset": (
                float(counts_hard1[index] / a) if a else None
            ),
            "identity_residual_constant": (
                float(b0 / total - (a / total) * (b0 / a))
                if a else None
            ),
            "identity_residual_exception": (
                float(b1 / total - (a / total) * (b1 / a))
                if a else None
            ),
        })
    write_csv(path, rows)
    return rows


def rows_by_epsilon(result: TaskResult) -> dict[float, dict[str, Any]]:
    output = {}
    for row in result.state.target_rows:
        epsilon = float(row["paired_epsilon"])
        output[round(epsilon, 12)] = row
    return output


def build_closure_rows(results: Sequence[TaskResult]) -> list[dict[str, Any]]:
    by_name = {result.spec.name: result for result in results}
    required = {
        "subset15_zero",
        "full_constant0",
        "full_single_exception",
    }
    if not required.issubset(by_name):
        return []
    subset = rows_by_epsilon(by_name["subset15_zero"])
    full0 = rows_by_epsilon(by_name["full_constant0"])
    full1 = rows_by_epsilon(by_name["full_single_exception"])
    rows: list[dict[str, Any]] = []
    for key in sorted(set(subset) & set(full0) & set(full1), reverse=True):
        a = subset[key]
        b0 = full0[key]
        b1 = full1[key]
        direct0 = float(b0["estimated_log_volume_median"])
        direct1 = float(b1["estimated_log_volume_median"])
        cross0 = a.get("cross_log_volume_constant_replica_median")
        cross1 = a.get("cross_log_volume_exception_replica_median")
        q0 = a.get("q_full_constant_given_subset_replica_median")
        q1 = a.get("q_full_exception_given_subset_replica_median")
        if q0 is None or float(q0) <= 0:
            cross0 = None
        if q1 is None or float(q1) <= 0:
            cross1 = None
        direct_log_ratio = direct0 - direct1
        cross_log_ratio = (
            float(math.log(float(q0)) - math.log(float(q1)))
            if q0 is not None and q1 is not None
            and float(q0) > 0 and float(q1) > 0
            else None
        )
        rows.append({
            "epsilon_full": float(key),
            "delta_subset": float(a["paired_delta"]),
            "direct_log_volume_constant": direct0,
            "direct_log_volume_exception": direct1,
            "subset_log_volume": float(a["estimated_log_volume_median"]),
            "q_constant_given_subset": q0,
            "q_exception_given_subset": q1,
            "cross_log_volume_constant": cross0,
            "cross_log_volume_exception": cross1,
            "closure_log_residual_constant": (
                direct0 - float(cross0) if cross0 is not None else None
            ),
            "closure_log_residual_exception": (
                direct1 - float(cross1) if cross1 is not None else None
            ),
            "direct_log_volume_ratio_constant_over_exception": direct_log_ratio,
            "cross_log_event_ratio_constant_over_exception": cross_log_ratio,
            "ratio_log_residual": (
                direct_log_ratio - float(cross_log_ratio)
                if cross_log_ratio is not None else None
            ),
            "direct_log10_volume_ratio": direct_log_ratio / math.log(10.0),
            "cross_log10_event_ratio": (
                float(cross_log_ratio) / math.log(10.0)
                if cross_log_ratio is not None else None
            ),
            "heldout_hard_probability_zero": a[
                "heldout_hard_probability_zero"
            ],
            "heldout_hard_probability_one": a[
                "heldout_hard_probability_one"
            ],
            "full_events_overlap_probability": a[
                "full_events_overlap_probability"
            ],
            "subset_function_support": a["function_support"],
        })
    return rows


def save_plot(root: Path, closure_rows: Sequence[dict[str, Any]]) -> None:
    if not closure_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(closure_rows, key=lambda row: row["epsilon_full"], reverse=True)
    epsilon = np.asarray([row["epsilon_full"] for row in ordered])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].plot(
        epsilon,
        [row["direct_log10_volume_ratio"] for row in ordered],
        marker="o",
        label="independent full SMC",
    )
    cross_ratio = np.asarray([
        np.nan if row["cross_log10_event_ratio"] is None
        else row["cross_log10_event_ratio"]
        for row in ordered
    ])
    axes[0, 0].plot(
        epsilon, cross_ratio, marker="s", label="subset cross-events"
    )
    axes[0, 0].set_title("log10 volume ratio: constant / exception")
    axes[0, 0].legend()

    axes[0, 1].plot(
        epsilon,
        [row["heldout_hard_probability_zero"] for row in ordered],
        marker="o",
        label="hard P(y=0)",
    )
    axes[0, 1].plot(
        epsilon,
        [row["heldout_hard_probability_one"] for row in ordered],
        marker="o",
        label="hard P(y=1)",
    )
    axes[0, 1].set_title("subset15 heldout hard prediction")
    axes[0, 1].legend()

    axes[1, 0].plot(
        epsilon,
        [
            np.nan if row["closure_log_residual_constant"] is None
            else row["closure_log_residual_constant"]
            for row in ordered
        ],
        marker="o",
        label="constant",
    )
    axes[1, 0].plot(
        epsilon,
        [
            np.nan if row["closure_log_residual_exception"] is None
            else row["closure_log_residual_exception"]
            for row in ordered
        ],
        marker="o",
        label="exception",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_title("closure residual in natural-log volume")
    axes[1, 0].legend()

    axes[1, 1].plot(
        epsilon,
        [row["subset_function_support"] for row in ordered],
        marker="o",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("subset15 hard-function support")

    for axis in axes.flat:
        axis.set_xlabel("full-rule BCE threshold epsilon")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    figure.savefig(root / "smc_consistency_closure.png", dpi=180)
    plt.close(figure)


def create_archive(root: Path) -> Path:
    archive = root.parent / f"{root.name}_package.zip"
    if archive.exists():
        archive.unlink()
    excluded = {"checkpoint.pt", "checkpoint.tmp.pt"}
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name not in excluded:
                handle.write(path, path.relative_to(root.parent))
    return archive


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    root = prepare_result_dir()
    write_json(root / "config.json", config_dict())
    specs = build_task_specs()
    write_json(root / "task_definitions.json", [
        task_definition(spec) for spec in specs
    ])

    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    inputs_np = truth_table_inputs()
    constant_np = np.zeros(16, dtype=np.float32)
    exception_np = constant_np.copy()
    exception_np[Config.HOLDOUT_INDEX] = 1.0
    inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    constant_targets = torch.from_numpy(constant_np).to(device)
    exception_targets = torch.from_numpy(exception_np).to(device)
    subset_indices = torch.tensor(
        specs[0].train_indices, dtype=torch.int64, device=device
    )
    all_indices = torch.arange(16, dtype=torch.int64, device=device)

    _, parameter_count = parameter_blocks(Config.WIDTH)
    print("=== Constant leave-one-out SMC consistency ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=4->{Config.WIDTH}x2->1 tanh | params={parameter_count} | "
        f"measure=uniform initialization cube",
        flush=True,
    )
    print(
        f"holdout={Config.HOLDOUT_INDEX:04b} | full functions="
        f"0x0000 vs 0x{1 << Config.HOLDOUT_INDEX:04X}",
        flush=True,
    )
    print(
        f"replicas={Config.REPLICAS} | particles/replica="
        f"{Config.PARTICLES_PER_REPLICA:,} | tasks=3",
        flush=True,
    )
    print(f"结果目录：{root}", flush=True)

    run_prior_calibration(
        root,
        device,
        inputs,
        constant_targets,
        exception_targets,
        subset_indices,
    )

    results: list[TaskResult] = []
    interrupted = False
    for spec in specs:
        result = run_task(
            root,
            spec,
            device,
            inputs,
            constant_targets,
            exception_targets,
            subset_indices,
            all_indices,
        )
        results.append(result)
        if result.status == "interrupted":
            interrupted = True
            break

    closure_rows = build_closure_rows(results)
    write_csv(root / "closure_consistency.csv", closure_rows)
    save_plot(root, closure_rows)
    summary = {
        "protocol": "constant_leave_one_out_smc_consistency_v1",
        "statuses": {
            result.spec.name: result.status for result in results
        },
        "closure_threshold_count": len(closure_rows),
        "finite_ratio_closure_count": sum(
            row["ratio_log_residual"] is not None for row in closure_rows
        ),
        "max_abs_ratio_log_residual": max(
            (
                abs(float(row["ratio_log_residual"]))
                for row in closure_rows
                if row["ratio_log_residual"] is not None
            ),
            default=None,
        ),
        "identity": (
            "V16_y(epsilon) = V15(16*epsilon/15) * "
            "P(V16_y event | V15 event)"
        ),
        "interpretation": (
            "比较的是同一静态参数测度的三个可测集合；不涉及SGD。"
        ),
    }
    write_json(root / "summary.json", summary)
    archive = create_archive(root) if Config.PACKAGE_RESULTS else None
    print(f"summary={summary}", flush=True)
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)
    if interrupted:
        print("保持RESUME=True重新运行即可继续未完成task。", flush=True)


if __name__ == "__main__":
    main()
