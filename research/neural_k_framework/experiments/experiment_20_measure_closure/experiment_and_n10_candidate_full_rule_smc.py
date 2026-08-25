"""AND n=10 最终四函数竞争与完整规则体积的静态 SMC 闭环。

固定旧实验完全相同的4->16x2->1 tanh网络、初始化参数立方体均匀测度和
10样本AND训练集。旧的深尾subset SMC在L10<=0.065时只剩四个函数：

    D440, F040, D040, F440.

本脚本独立运行一个subset10 SMC和四个完整真值表SMC。对完整平均BCE阈值
epsilon，subset配对阈值为delta=16*epsilon/10。对每个候选g必须满足：

    Vol(L16_g<=epsilon)
      = Vol(L10<=delta)
        * P(L16_g<=epsilon | L10<=delta).

脚本同时比较：

1. 四个独立full-rule体积的归一化竞争分布；
2. subset10粒子中的四个hard-function概率；
3. subset10粒子中的四个完整high-margin交叉事件；
4. 每个候选的绝对体积闭环残差。

最深配对点固定为delta=0.065、epsilon=0.040625，且epsilon低于
ln(2)/16，因此full事件中的粒子必然hard-exact实现相应完整规则。
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
    TRAIN_INDICES = (1, 2, 3, 5, 7, 8, 11, 12, 14, 15)
    CANDIDATE_FUNCTION_IDS = (0xD440, 0xF040, 0xD040, 0xF440)
    REFERENCE_N10_DELTA = 0.065
    REFERENCE_N10_PROBABILITIES = (
        0.5570068359375,
        0.245513916015625,
        0.139434814453125,
        0.05804443359375,
    )

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
        0.040625,
    )
    REPLICAS = 8
    PARTICLES_PER_REPLICA = 4_096
    SURVIVAL_QUANTILE = 0.5
    MAX_LEVELS_PER_TASK = 1_600

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
    TOP_FUNCTIONS = 20
    CHECKPOINT_EVERY_LEVELS = 1

    PRIOR_SEED = 20261011
    RESAMPLE_SEED = 20261012
    MUTATION_SEED = 20261013
    CALIBRATION_SEED = 20261014
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_and_n10_candidate_full_rule_smc")
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
        "_smoke_and_n10_candidate_full_rule_smc"
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
    if len(Config.TRAIN_INDICES) != 10 or len(set(Config.TRAIN_INDICES)) != 10:
        raise ValueError("TRAIN_INDICES必须包含10个不重复状态。")
    if len(Config.CANDIDATE_FUNCTION_IDS) < 2:
        raise ValueError("至少需要两个候选完整函数。")
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


def function_id_to_outputs(function_id: int) -> np.ndarray:
    return np.asarray([
        (int(function_id) >> index) & 1 for index in range(16)
    ], dtype=np.uint8)


def and_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs[:, 0] & inputs[:, 1]).astype(np.uint8)


def build_task_specs() -> tuple[TaskSpec, ...]:
    inputs = truth_table_inputs()
    subset_targets = and_targets(inputs)
    all_indices = tuple(range(16))
    subset_indices = tuple(int(index) for index in Config.TRAIN_INDICES)
    full_thresholds = tuple(float(value) for value in Config.FULL_THRESHOLDS)
    subset_thresholds = tuple(
        float(value * 16.0 / len(subset_indices)) for value in full_thresholds
    )
    specs = [
        TaskSpec(
            name="subset10_and",
            description="balanced n=10 AND subset from the original experiment",
            targets=tuple(int(value) for value in subset_targets),
            train_indices=subset_indices,
            thresholds=subset_thresholds,
            target_function_id=None,
        )
    ]
    for function_id in Config.CANDIDATE_FUNCTION_IDS:
        outputs = function_id_to_outputs(int(function_id))
        if not np.array_equal(
            outputs[list(subset_indices)], subset_targets[list(subset_indices)]
        ):
            raise ValueError(
                f"候选0x{int(function_id):04X}与n=10训练集不一致。"
            )
        specs.append(TaskSpec(
            name=f"full_{int(function_id):04x}",
            description=f"16-point candidate function 0x{int(function_id):04X}",
            targets=tuple(int(value) for value in outputs),
            train_indices=all_indices,
            thresholds=full_thresholds,
            target_function_id=int(function_id),
        ))
    return tuple(specs)


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
    subset_targets: torch.Tensor,
    candidate_targets: torch.Tensor,
    subset_indices: torch.Tensor,
) -> dict[str, np.ndarray]:
    flat = particles.reshape(-1, particles.shape[-1])
    loss10_pieces: list[np.ndarray] = []
    full_loss_pieces: list[np.ndarray] = []
    function_id_pieces: list[np.ndarray] = []
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=particles.device),
        torch.arange(16, dtype=torch.int64, device=particles.device),
    )
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        subset_loss_per_sample = F.binary_cross_entropy_with_logits(
            logits,
            subset_targets[None].expand_as(logits),
            reduction="none",
        )
        local_full_losses = []
        for targets in candidate_targets:
            local_full_losses.append(F.binary_cross_entropy_with_logits(
                logits,
                targets[None].expand_as(logits),
                reduction="none",
            ).mean(dim=1))
        function_ids = (
            (logits >= 0).to(torch.int64) * powers[None]
        ).sum(dim=1)
        loss10_pieces.append(
            subset_loss_per_sample[:, subset_indices].mean(dim=1).cpu().numpy()
        )
        full_loss_pieces.append(
            torch.stack(local_full_losses, dim=1).cpu().numpy()
        )
        function_id_pieces.append(
            function_ids.cpu().numpy().astype(np.uint16)
        )
    shape = particles.shape[:-1]
    return {
        "loss10": np.concatenate(loss10_pieces).reshape(shape),
        "full_losses": np.concatenate(full_loss_pieces).reshape(
            shape + (len(Config.CANDIDATE_FUNCTION_IDS),)
        ),
        "function_id": np.concatenate(function_id_pieces).reshape(shape),
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
    train_count = len(Config.TRAIN_INDICES)
    if spec.name == "subset10_and":
        epsilon = float(threshold * train_count / 16.0)
        delta = float(threshold)
    else:
        epsilon = float(threshold)
        delta = float(threshold * 16.0 / train_count)

    ids = observables["function_id"].astype(np.int64)
    full_events = observables["full_losses"] <= (
        epsilon + Config.LOSS_TOLERANCE
    )
    event_subset = observables["loss10"] <= (
        delta + Config.LOSS_TOLERANCE
    )

    replica_rows: list[dict[str, Any]] = []
    for replica in range(Config.REPLICAS):
        log_volume = float(state.log_volume_fraction[replica].item())
        replica_row: dict[str, Any] = {
            "task": spec.name,
            "threshold": threshold,
            "paired_epsilon": epsilon,
            "paired_delta": delta,
            "replica": replica,
            "estimated_log_volume": log_volume,
            "estimated_volume": float(math.exp(log_volume)),
            "multi_full_event_probability": float(
                (full_events[replica].sum(axis=-1) > 1).mean()
            ),
            "paired_subset_inclusion_violation_fraction": float(
                (~event_subset[replica]).mean()
            ),
            "unique_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        }
        for candidate_index, function_id in enumerate(
            Config.CANDIDATE_FUNCTION_IDS
        ):
            label = f"{int(function_id):04X}"
            q_value = float(full_events[replica, :, candidate_index].mean())
            replica_row[f"q_full_{label}_given_subset"] = q_value
            replica_row[f"log_q_full_{label}_given_subset"] = (
                safe_log_probability(q_value)
            )
            replica_row[f"cross_log_volume_{label}"] = (
                log_volume + math.log(q_value) if q_value > 0 else None
            )
            replica_row[f"hard_probability_{label}"] = float(
                (ids[replica] == int(function_id)).mean()
            )
        replica_rows.append(replica_row)

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
        "function_support": int(len(np.unique(ids))),
        "candidate_hard_mass": float(np.isin(
            ids, np.asarray(Config.CANDIDATE_FUNCTION_IDS)
        ).mean()),
        "multi_full_event_probability": float(
            (full_events.sum(axis=-1) > 1).mean()
        ),
        "paired_subset_inclusion_violation_fraction": float(
            (~event_subset).mean()
        ),
    }
    for candidate_index, function_id in enumerate(Config.CANDIDATE_FUNCTION_IDS):
        label = f"{int(function_id):04X}"
        aggregate[f"q_full_{label}_given_subset"] = float(
            full_events[..., candidate_index].mean()
        )
        aggregate[f"hard_probability_{label}"] = float(
            (ids == int(function_id)).mean()
        )
        for key in (
            f"q_full_{label}_given_subset",
            f"cross_log_volume_{label}",
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
    subset_targets: torch.Tensor,
    candidate_targets: torch.Tensor,
    subset_indices: torch.Tensor,
) -> None:
    observables = evaluate_all_observables(
        state.particles,
        inputs,
        subset_targets,
        candidate_targets,
        subset_indices,
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
        loss10=observables["loss10"],
        candidate_function_ids=np.asarray(
            Config.CANDIDATE_FUNCTION_IDS, dtype=np.uint16
        ),
        full_losses=observables["full_losses"],
        lineages=state.lineages.detach().cpu().numpy(),
        log_volume_fraction=state.log_volume_fraction.detach().cpu().numpy(),
    )
    hard_text = ",".join(
        f"{int(function_id):04X}:"
        f"{aggregate[f'hard_probability_{int(function_id):04X}']:.1%}"
        for function_id in Config.CANDIDATE_FUNCTION_IDS
    )
    q_text = ",".join(
        f"{int(function_id):04X}:"
        f"{aggregate[f'q_full_{int(function_id):04X}_given_subset']:.1e}"
        for function_id in Config.CANDIDATE_FUNCTION_IDS
    )
    print(
        f"[{spec.name}] TARGET={threshold:.6f} | "
        f"logV~{aggregate['estimated_log_volume_median']:.3f} | "
        f"hard[{hard_text}] | q[{q_text}] | "
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
    subset_targets: torch.Tensor,
    candidate_targets: torch.Tensor,
    subset_indices: torch.Tensor,
) -> TaskResult:
    global GLOBAL_INPUTS, GLOBAL_TARGETS, GLOBAL_TRAIN_INDICES, GLOBAL_THRESHOLDS

    task_dir = root / "tasks" / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)
    validate_task_resume(task_dir, spec)
    write_json(task_dir / "task_definition.json", task_definition(spec))
    GLOBAL_INPUTS = inputs
    GLOBAL_TARGETS = torch.tensor(
        spec.targets, dtype=torch.float32, device=device
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
                    subset_targets,
                    candidate_targets,
                    subset_indices,
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
    subset_targets: torch.Tensor,
    candidate_targets: torch.Tensor,
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
    subset_thresholds = full_thresholds * 16.0 / len(Config.TRAIN_INDICES)
    counts_a = np.zeros(len(full_thresholds), dtype=np.int64)
    counts_b = np.zeros(
        (len(full_thresholds), len(Config.CANDIDATE_FUNCTION_IDS)),
        dtype=np.int64,
    )
    counts_hard = np.zeros_like(counts_b)
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
        subset_loss = F.binary_cross_entropy_with_logits(
            logits,
            subset_targets[None].expand_as(logits),
            reduction="none",
        )
        l10 = subset_loss[:, subset_indices].mean(dim=1).cpu().numpy()
        full_losses = []
        for targets in candidate_targets:
            full_losses.append(F.binary_cross_entropy_with_logits(
                logits,
                targets[None].expand_as(logits),
                reduction="none",
            ).mean(dim=1))
        full_losses_np = torch.stack(full_losses, dim=1).cpu().numpy()
        powers = torch.bitwise_left_shift(
            torch.ones(16, dtype=torch.int64, device=device),
            torch.arange(16, dtype=torch.int64, device=device),
        )
        function_ids = (
            ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)
            .cpu().numpy()
        )
        for index, (epsilon, delta) in enumerate(
            zip(full_thresholds, subset_thresholds)
        ):
            event_a = l10 <= delta + Config.LOSS_TOLERANCE
            counts_a[index] += int(event_a.sum())
            for candidate_index, function_id in enumerate(
                Config.CANDIDATE_FUNCTION_IDS
            ):
                counts_b[index, candidate_index] += int((
                    full_losses_np[:, candidate_index]
                    <= epsilon + Config.LOSS_TOLERANCE
                ).sum())
                counts_hard[index, candidate_index] += int((
                    event_a & (function_ids == int(function_id))
                ).sum())
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
        row: dict[str, Any] = {
            "sample_count": total,
            "epsilon_full": float(epsilon),
            "delta_subset": float(delta),
            "count_subset": int(a),
            "volume_subset": float(a / total),
        }
        for candidate_index, function_id in enumerate(
            Config.CANDIDATE_FUNCTION_IDS
        ):
            label = f"{int(function_id):04X}"
            b_count = int(counts_b[index, candidate_index])
            row[f"count_full_{label}"] = b_count
            row[f"volume_full_{label}"] = float(b_count / total)
            row[f"q_full_{label}_given_subset"] = (
                float(b_count / a) if a else None
            )
            row[f"hard_probability_{label}_given_subset"] = (
                float(counts_hard[index, candidate_index] / a)
                if a else None
            )
            row[f"identity_residual_{label}"] = (
                float(b_count / total - (a / total) * (b_count / a))
                if a else None
            )
        rows.append(row)
    write_csv(path, rows)
    return rows


def rows_by_epsilon(result: TaskResult) -> dict[float, dict[str, Any]]:
    output = {}
    for row in result.state.target_rows:
        epsilon = float(row["paired_epsilon"])
        output[round(epsilon, 12)] = row
    return output


def normalized_from_logs(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    weights = np.exp(array - maximum)
    return weights / weights.sum()


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    middle = 0.5 * (p + q)
    value = 0.0
    for distribution in (p, q):
        mask = distribution > 0
        value += 0.5 * float(np.sum(
            distribution[mask]
            * np.log2(distribution[mask] / middle[mask])
        ))
    return value


def build_closure_rows(
    results: Sequence[TaskResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_name = {result.spec.name: result for result in results}
    required = {"subset10_and"} | {
        f"full_{int(function_id):04x}"
        for function_id in Config.CANDIDATE_FUNCTION_IDS
    }
    if not required.issubset(by_name):
        return [], []
    subset = rows_by_epsilon(by_name["subset10_and"])
    full_maps = {
        int(function_id): rows_by_epsilon(
            by_name[f"full_{int(function_id):04x}"]
        )
        for function_id in Config.CANDIDATE_FUNCTION_IDS
    }
    common = set(subset)
    for mapping in full_maps.values():
        common &= set(mapping)

    candidate_rows: list[dict[str, Any]] = []
    competition_rows: list[dict[str, Any]] = []
    for key in sorted(common, reverse=True):
        a = subset[key]
        direct_logs = np.asarray([
            float(full_maps[int(function_id)][key][
                "estimated_log_volume_median"
            ])
            for function_id in Config.CANDIDATE_FUNCTION_IDS
        ])
        full_probabilities = normalized_from_logs(direct_logs)
        hard_raw = np.asarray([
            float(a[f"hard_probability_{int(function_id):04X}"])
            for function_id in Config.CANDIDATE_FUNCTION_IDS
        ])
        hard_mass = float(hard_raw.sum())
        hard_probabilities = (
            hard_raw / hard_mass if hard_mass > 0
            else np.full_like(hard_raw, np.nan)
        )
        q_values = []
        cross_logs: list[float | None] = []
        for function_id in Config.CANDIDATE_FUNCTION_IDS:
            label = f"{int(function_id):04X}"
            q_value = a.get(f"q_full_{label}_given_subset_replica_median")
            cross_value = a.get(f"cross_log_volume_{label}_replica_median")
            if q_value is None or float(q_value) <= 0:
                q_values.append(0.0)
                cross_logs.append(None)
            else:
                q_values.append(float(q_value))
                cross_logs.append(
                    float(cross_value) if cross_value is not None else None
                )
        q_array = np.asarray(q_values, dtype=np.float64)
        q_mass = float(q_array.sum())
        q_probabilities = (
            q_array / q_mass if q_mass > 0
            else np.full_like(q_array, np.nan)
        )

        for candidate_index, function_id in enumerate(
            Config.CANDIDATE_FUNCTION_IDS
        ):
            direct_log = float(direct_logs[candidate_index])
            cross_log = cross_logs[candidate_index]
            candidate_rows.append({
                "epsilon_full": float(key),
                "delta_subset": float(a["paired_delta"]),
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "direct_log_volume": direct_log,
                "direct_log10_volume": direct_log / math.log(10.0),
                "direct_normalized_full_volume_probability": float(
                    full_probabilities[candidate_index]
                ),
                "subset_hard_probability_raw": float(hard_raw[candidate_index]),
                "subset_hard_probability_within_candidates": float(
                    hard_probabilities[candidate_index]
                ) if hard_mass > 0 else None,
                "subset_cross_event_probability": float(q_array[candidate_index]),
                "subset_cross_event_probability_within_candidates": float(
                    q_probabilities[candidate_index]
                ) if q_mass > 0 else None,
                "cross_log_volume": cross_log,
                "closure_log_residual": (
                    direct_log - float(cross_log)
                    if cross_log is not None else None
                ),
                "subset_candidate_hard_mass": hard_mass,
                "subset_cross_event_mass": q_mass,
                "subset_function_support": int(a["function_support"]),
            })

        finite_q = bool(np.all(q_array > 0))
        competition_rows.append({
            "epsilon_full": float(key),
            "delta_subset": float(a["paired_delta"]),
            "subset_function_support": int(a["function_support"]),
            "subset_candidate_hard_mass": hard_mass,
            "subset_cross_event_mass": q_mass,
            "full_vs_subset_hard_jsd": (
                js_divergence(full_probabilities, hard_probabilities)
                if hard_mass > 0 else None
            ),
            "full_vs_subset_hard_tv": (
                0.5 * float(np.abs(
                    full_probabilities - hard_probabilities
                ).sum()) if hard_mass > 0 else None
            ),
            "full_vs_cross_event_jsd": (
                js_divergence(full_probabilities, q_probabilities)
                if finite_q else None
            ),
            "full_vs_cross_event_tv": (
                0.5 * float(np.abs(
                    full_probabilities - q_probabilities
                ).sum()) if finite_q else None
            ),
            "current_hard_vs_old_n10_jsd": (
                js_divergence(
                    hard_probabilities,
                    np.asarray(Config.REFERENCE_N10_PROBABILITIES),
                ) if hard_mass > 0 else None
            ),
            "current_hard_vs_old_n10_tv": (
                0.5 * float(np.abs(
                    hard_probabilities
                    - np.asarray(Config.REFERENCE_N10_PROBABILITIES)
                ).sum()) if hard_mass > 0 else None
            ),
            "max_abs_closure_log_residual": max(
                (
                    abs(float(direct_logs[index] - cross_logs[index]))
                    for index in range(len(cross_logs))
                    if cross_logs[index] is not None
                ),
                default=None,
            ),
        })
    return candidate_rows, competition_rows


def save_plot(
    root: Path,
    candidate_rows: Sequence[dict[str, Any]],
    competition_rows: Sequence[dict[str, Any]],
) -> None:
    if not candidate_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = ("#00798C", "#D1495B", "#6A4C93", "#E0A458")
    for candidate_index, function_id in enumerate(Config.CANDIDATE_FUNCTION_IDS):
        local = sorted([
            row for row in candidate_rows
            if int(row["function_id"]) == int(function_id)
        ], key=lambda row: row["epsilon_full"], reverse=True)
        epsilon = [row["epsilon_full"] for row in local]
        label = f"0x{int(function_id):04X}"
        color = colors[candidate_index % len(colors)]
        axes[0, 0].plot(
            epsilon,
            [row["direct_normalized_full_volume_probability"] for row in local],
            marker="o", label=label, color=color,
        )
        axes[0, 1].plot(
            epsilon,
            [row["subset_hard_probability_raw"] for row in local],
            marker="o", label=label, color=color,
        )
        axes[1, 0].plot(
            epsilon,
            [
                np.nan if row["closure_log_residual"] is None
                else row["closure_log_residual"] for row in local
            ],
            marker="o", label=label, color=color,
        )
    axes[0, 0].set_title("normalized full-rule low-loss volume")
    axes[0, 0].legend()
    axes[0, 1].set_title("subset10 hard-function probabilities")
    axes[0, 1].legend()
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_title("closure residual in natural-log volume")
    axes[1, 0].legend()
    ordered_competition = sorted(
        competition_rows, key=lambda row: row["epsilon_full"], reverse=True
    )
    epsilon_competition = [row["epsilon_full"] for row in ordered_competition]
    axes[1, 1].plot(
        epsilon_competition,
        [row["full_vs_subset_hard_jsd"] for row in ordered_competition],
        marker="o", label="full volume vs hard JSD",
    )
    axes[1, 1].plot(
        epsilon_competition,
        [
            np.nan if row["full_vs_cross_event_jsd"] is None
            else row["full_vs_cross_event_jsd"]
            for row in ordered_competition
        ],
        marker="s", label="full volume vs cross-event JSD",
    )
    axes[1, 1].set_title("candidate distribution comparison")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("full-rule BCE threshold epsilon")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    figure.savefig(root / "and_n10_candidate_competition.png", dpi=180)
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
    subset_np = and_targets(inputs_np).astype(np.float32)
    candidate_np = np.stack([
        function_id_to_outputs(int(function_id)).astype(np.float32)
        for function_id in Config.CANDIDATE_FUNCTION_IDS
    ])
    inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    subset_targets = torch.from_numpy(subset_np).to(device)
    candidate_targets = torch.from_numpy(candidate_np).to(device)
    subset_indices = torch.tensor(
        specs[0].train_indices, dtype=torch.int64, device=device
    )

    _, parameter_count = parameter_blocks(Config.WIDTH)
    print("=== AND n=10 candidate full-rule SMC ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=4->{Config.WIDTH}x2->1 tanh | params={parameter_count} | "
        f"measure=uniform initialization cube",
        flush=True,
    )
    print(
        f"train_indices={list(Config.TRAIN_INDICES)} | candidates="
        f"{[f'0x{int(value):04X}' for value in Config.CANDIDATE_FUNCTION_IDS]}",
        flush=True,
    )
    print(
        f"replicas={Config.REPLICAS} | particles/replica="
        f"{Config.PARTICLES_PER_REPLICA:,} | tasks={len(specs)}",
        flush=True,
    )
    print(f"结果目录：{root}", flush=True)

    run_prior_calibration(
        root,
        device,
        inputs,
        subset_targets,
        candidate_targets,
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
            subset_targets,
            candidate_targets,
            subset_indices,
        )
        results.append(result)
        if result.status == "interrupted":
            interrupted = True
            break

    closure_rows, competition_rows = build_closure_rows(results)
    write_csv(root / "closure_by_candidate.csv", closure_rows)
    write_csv(root / "competition_summary.csv", competition_rows)
    save_plot(root, closure_rows, competition_rows)
    finite_residuals = [
        abs(float(row["closure_log_residual"]))
        for row in closure_rows
        if row["closure_log_residual"] is not None
    ]
    summary = {
        "protocol": "and_n10_candidate_full_rule_smc_v1",
        "statuses": {
            result.spec.name: result.status for result in results
        },
        "closure_candidate_row_count": len(closure_rows),
        "competition_threshold_count": len(competition_rows),
        "finite_closure_count": len(finite_residuals),
        "max_abs_closure_log_residual": (
            max(finite_residuals) if finite_residuals else None
        ),
        "deepest_pair": {
            "delta_subset": 0.065,
            "epsilon_full": 0.040625,
        },
        "identity": (
            "V16_g(epsilon) = V10(16*epsilon/10) * "
            "P(V16_g event | V10 event)"
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
