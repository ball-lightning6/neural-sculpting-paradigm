"""Parity2 / MUX3 的 fixed-D 静态条件函数分布 SMC。

该实验不训练网络。它从与 E23 Gaussian 深尾实验相同的参数参考测度出发，
分别对 uniform_random、cell_balanced、conflict_enriched 三种 n=32 训练集
条件化，并比较 parity2 与 MUX3 的完整 hard-function 分布。

目标是拆开两个不同对象：

* full-target volume：完整256标签都给定时，目标函数的低-loss参数质量；
* fixed-D posterior：只给32个标签时，目标延拓在所有兼容函数中的条件质量。

如果刚才 AdamW 中 conf 对 MUX3 的巨大提升在本实验中复现，说明主要来源
是静态竞争分母；若静态分布不复现而 AdamW 复现，才说明 optimizer 运输
具有主要作用。脚本支持 checkpoint、Ctrl-C 保存和结果打包。
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
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2
    TRAIN_COUNT = 32
    TARGET_NAMES = ("parity2", "mux3")
    SAMPLING_PROTOCOLS = (
        "uniform_random",
        "cell_balanced",
        "conflict_enriched",
    )
    DATASETS_PER_PROTOCOL = 8

    REFERENCE_MEASURE = "gaussian_matched_variance"
    GAUSSIAN_SIGMA = 1.0 / math.sqrt(3.0)
    TARGET_THRESHOLDS = (
        0.68, 0.60, 0.50, 0.40, 0.30,
        0.20, 0.10, 0.05, 0.03, 0.02,
    )
    REPLICAS = 4
    PARTICLES_PER_REPLICA = 2_048
    SURVIVAL_QUANTILE = 0.5
    MAX_LEVELS = 5_000
    MIN_LEVEL_DECREMENT = 1e-8

    ADAPT_SWEEPS = 2
    MUTATION_SWEEPS = 6
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.050, 0.030, 0.050, 0.015)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.30
    LOSS_TOLERANCE = 1e-7
    EVAL_MICRO_BATCH = 2_048
    FULL_EVAL_MICRO_BATCH = 512

    DATASET_SEED = 2026083001
    PRIOR_SEED = 2026083002
    RESAMPLE_SEED = 2026083003
    MUTATION_SEED = 2026083004

    CHECKPOINT_EVERY_LEVELS = 10
    LOG_EVERY_LEVELS = 10
    TOP_FUNCTIONS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_parity2_mux3_fixed_d_static_smc")
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


@dataclass(frozen=True)
class ConditionSpec:
    index: int
    target_index: int
    target_name: str
    sampling_protocol: str
    dataset_index: int
    indices: tuple[int, ...]
    signature: str
    mux_cell_counts: tuple[int, ...]
    target_cell_counts: tuple[int, ...]
    conflict_fraction: float
    copy_x1_accuracy: float
    copy_x2_accuracy: float
    parity_alt_accuracy: float


@dataclass
class SMCState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume: torch.Tensor
    proposal_scales: torch.Tensor
    current_threshold: float
    target_index: int
    level: int
    level_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    replica_rows: list[dict[str, Any]]
    top_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.DATASETS_PER_PROTOCOL = 2
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 64
    Config.TARGET_THRESHOLDS = (0.72, 0.70)
    Config.MAX_LEVELS = 20
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 1
    Config.EVAL_MICRO_BATCH = 128
    Config.FULL_EVAL_MICRO_BATCH = 32
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.TOP_FUNCTIONS = 3
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/neural_k_framework/experiments/"
        "experiment_23_volume_to_data_transition/"
        "_smoke_parity2_mux3_fixed_d_static_smc"
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


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_ready(payload), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def truth_table_inputs() -> np.ndarray:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.uint16)
    shifts = np.arange(Config.INPUT_BITS - 1, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def outputs_to_hex(outputs: np.ndarray) -> str:
    function_id = 0
    for index, bit in enumerate(np.asarray(outputs, dtype=np.uint8)):
        function_id |= int(bit) << index
    return f"0x{function_id:0{len(outputs)//4}X}"


def build_targets() -> list[TargetSpec]:
    inputs = truth_table_inputs()
    raw = {
        "parity2": (
            np.bitwise_xor.reduce(inputs[:, :2], axis=1).astype(np.uint8),
            "x0 XOR x1",
        ),
        "mux3": (
            np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]).astype(
                np.uint8
            ),
            "IF x0 THEN x1 ELSE x2",
        ),
    }
    result = []
    for index, name in enumerate(Config.TARGET_NAMES):
        outputs, formula = raw[name]
        result.append(TargetSpec(
            index=index,
            name=name,
            formula=formula,
            outputs=tuple(map(int, outputs)),
            function_hex=outputs_to_hex(outputs),
        ))
    return result


def mux_cell_ids(inputs: np.ndarray) -> np.ndarray:
    return (
        4 * inputs[:, 0] + 2 * inputs[:, 1] + inputs[:, 2]
    ).astype(np.int64)


def sampling_order(
    protocol: str,
    dataset_index: int,
    inputs: np.ndarray,
) -> np.ndarray:
    protocol_index = Config.SAMPLING_PROTOCOLS.index(protocol)
    rng = np.random.default_rng(
        Config.DATASET_SEED + 1_000_003 * protocol_index + dataset_index
    )
    if protocol == "uniform_random":
        return rng.permutation(len(inputs)).astype(np.int64)

    cells = mux_cell_ids(inputs)
    queues = {
        cell: list(map(int, rng.permutation(np.flatnonzero(cells == cell))))
        for cell in range(8)
    }
    if protocol == "cell_balanced":
        positions = {cell: 0 for cell in range(8)}
        order = []
        while len(order) < len(inputs):
            for cell in rng.permutation(8):
                cell = int(cell)
                position = positions[cell]
                if position < len(queues[cell]):
                    order.append(queues[cell][position])
                    positions[cell] += 1
        return np.asarray(order, dtype=np.int64)

    if protocol != "conflict_enriched":
        raise ValueError(protocol)
    positions = {cell: 0 for cell in range(8)}
    order = []
    while len(order) < Config.TRAIN_COUNT:
        doubled_first = set(map(int, rng.choice(4, size=2, replace=False)))
        for doubled in (doubled_first, set(range(4)) - doubled_first):
            half_cycle = []
            for parity_cell in range(4):
                x0, x1 = divmod(parity_cell, 2)
                ordinary = 4 * x0 + 2 * x1 + x1
                conflict = 4 * x0 + 2 * x1 + (1 - x1)
                half_cycle.append(conflict)
                half_cycle.append(
                    conflict if parity_cell in doubled else ordinary
                )
            for cell in rng.permutation(half_cycle):
                cell = int(cell)
                order.append(queues[cell][positions[cell]])
                positions[cell] += 1
            if len(order) >= Config.TRAIN_COUNT:
                break
    leftovers = [
        value
        for cell in range(8)
        for value in queues[cell][positions[cell]:]
    ]
    order.extend(map(int, rng.permutation(leftovers)))
    return np.asarray(order, dtype=np.int64)


def make_condition(
    index: int,
    target: TargetSpec,
    protocol: str,
    dataset_index: int,
    indices: tuple[int, ...],
) -> ConditionSpec:
    inputs = truth_table_inputs()
    selected = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(target.outputs, dtype=np.uint8)[selected]
    mux_cells = mux_cell_ids(inputs[selected])
    if target.name == "parity2":
        target_cells = 2 * inputs[selected, 0] + inputs[selected, 1]
        target_cell_count = 4
    else:
        target_cells = mux_cells
        target_cell_count = 8
    parity_alt = np.logical_not(
        np.logical_xor(inputs[selected, 0], inputs[selected, 2])
    ).astype(np.uint8)
    signature = hashlib.sha256(
        protocol.encode("ascii")
        + np.asarray(indices, dtype=np.uint16).tobytes()
    ).hexdigest()[:16]
    return ConditionSpec(
        index=index,
        target_index=target.index,
        target_name=target.name,
        sampling_protocol=protocol,
        dataset_index=dataset_index,
        indices=indices,
        signature=signature,
        mux_cell_counts=tuple(map(
            int, np.bincount(mux_cells, minlength=8)
        )),
        target_cell_counts=tuple(map(
            int, np.bincount(target_cells, minlength=target_cell_count)
        )),
        conflict_fraction=float(np.mean(
            inputs[selected, 1] != inputs[selected, 2]
        )),
        copy_x1_accuracy=float(np.mean(inputs[selected, 1] == labels)),
        copy_x2_accuracy=float(np.mean(inputs[selected, 2] == labels)),
        parity_alt_accuracy=float(np.mean(parity_alt == labels)),
    )


def build_conditions(targets: Sequence[TargetSpec]) -> list[ConditionSpec]:
    inputs = truth_table_inputs()
    result = []
    for protocol in Config.SAMPLING_PROTOCOLS:
        for dataset_index in range(Config.DATASETS_PER_PROTOCOL):
            order = sampling_order(protocol, dataset_index, inputs)
            indices = tuple(sorted(map(int, order[:Config.TRAIN_COUNT])))
            for target in targets:
                result.append(make_condition(
                    len(result), target, protocol, dataset_index, indices
                ))
    return result


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


def forward_logits(
    coordinates: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
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

    hidden = torch.tanh(
        torch.bmm(inputs, first_weight.transpose(1, 2))
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
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
) -> torch.Tensor:
    condition_count, replicas, particle_count, parameter_count = (
        particles.shape
    )
    flat = particles.reshape(-1, parameter_count)
    per_condition = replicas * particle_count
    pieces = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        stop = min(start + Config.EVAL_MICRO_BATCH, len(flat))
        condition_ids = torch.div(
            torch.arange(start, stop, device=flat.device),
            per_condition,
            rounding_mode="floor",
        )
        local_inputs = train_inputs[condition_ids]
        local_targets = train_targets[condition_ids]
        logits = forward_logits(flat[start:stop], local_inputs)
        pieces.append(F.binary_cross_entropy_with_logits(
            logits, local_targets, reduction="none"
        ).mean(dim=1))
    return torch.cat(pieces).reshape(
        condition_count, replicas, particle_count
    )


@torch.no_grad()
def evaluate_full_hard(
    particles: torch.Tensor,
    full_inputs: torch.Tensor,
) -> torch.Tensor:
    shape = particles.shape[:-1]
    flat = particles.reshape(-1, particles.shape[-1])
    pieces = []
    for start in range(0, len(flat), Config.FULL_EVAL_MICRO_BATCH):
        local = flat[start:start + Config.FULL_EVAL_MICRO_BATCH]
        inputs = full_inputs[None].expand(len(local), -1, -1)
        pieces.append(forward_logits(local, inputs) >= 0)
    return torch.cat(pieces).reshape(*shape, len(full_inputs))


def prepare_tensors(
    conditions: Sequence[ConditionSpec],
    targets: Sequence[TargetSpec],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    inputs = truth_table_inputs().astype(np.float32)
    target_matrix = np.asarray([
        target.outputs for target in targets
    ], dtype=np.uint8)
    train_x = np.empty(
        (len(conditions), Config.TRAIN_COUNT, Config.INPUT_BITS),
        dtype=np.float32,
    )
    train_y = np.empty(
        (len(conditions), Config.TRAIN_COUNT), dtype=np.float32
    )
    full_targets = np.empty(
        (len(conditions), 2 ** Config.INPUT_BITS), dtype=np.uint8
    )
    for condition in conditions:
        indices = np.asarray(condition.indices, dtype=np.int64)
        outputs = target_matrix[condition.target_index]
        train_x[condition.index] = inputs[indices]
        train_y[condition.index] = outputs[indices]
        full_targets[condition.index] = outputs
    return (
        torch.as_tensor(train_x, device=device),
        torch.as_tensor(train_y, device=device),
        torch.as_tensor(inputs, device=device),
        full_targets,
    )


def make_generators(device: torch.device) -> dict[str, torch.Generator]:
    result = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    result["prior"].manual_seed(Config.PRIOR_SEED)
    result["resample"].manual_seed(Config.RESAMPLE_SEED)
    result["mutation"].manual_seed(Config.MUTATION_SEED)
    return result


def initialize_state(
    condition_count: int,
    parameter_count: int,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> SMCState:
    # 所有条件共享同一批初始粒子，降低协议和目标差值的 Monte Carlo 方差。
    base = torch.empty(
        Config.REPLICAS,
        Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    ).normal_(
        mean=0.0, std=Config.GAUSSIAN_SIGMA, generator=generator
    )
    particles = base[None].expand(condition_count, -1, -1, -1).clone()
    losses = evaluate_losses(particles, train_inputs, train_targets)
    base_lineages = torch.arange(
        Config.REPLICAS * Config.PARTICLES_PER_REPLICA,
        dtype=torch.int64,
        device=device,
    ).reshape(Config.REPLICAS, Config.PARTICLES_PER_REPLICA)
    blocks, _ = parameter_blocks()
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=base_lineages[None].expand(
            condition_count, -1, -1
        ).clone(),
        log_volume=torch.zeros(
            condition_count, Config.REPLICAS,
            dtype=torch.float64, device=device,
        ),
        proposal_scales=torch.as_tensor(
            Config.INITIAL_PROPOSAL_SCALES,
            dtype=torch.float32, device=device,
        )[None].expand(condition_count, -1).clone(),
        current_threshold=float("inf"),
        target_index=0,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
        top_rows=[],
        prediction_rows=[],
    )


def choose_threshold(state: SMCState, requested: float) -> tuple[float, bool]:
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=2
    )
    threshold = max(float(requested), float(quantiles.max().item()))
    if math.isfinite(state.current_threshold):
        threshold = min(threshold, state.current_threshold)
    return threshold, threshold <= requested + 1e-12


@torch.no_grad()
def resample_state(
    state: SMCState,
    threshold: float,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    conditions, replicas = state.losses.shape[:2]
    survival = np.zeros((conditions, replicas), dtype=np.float64)
    # 保留所有幸存粒子，只用幸存者填补被淘汰的位置。相比每层对全部
    # 粒子做 multinomial bootstrap，这在部分条件存活率接近1时不会
    # 无端丢失 lineage，同时仍保持条件样本的正确期望权重。
    new_particles = state.particles.clone()
    new_lineages = state.lineages.clone()
    for condition in range(conditions):
        for replica in range(replicas):
            valid = torch.nonzero(
                state.losses[condition, replica]
                <= threshold + Config.LOSS_TOLERANCE,
                as_tuple=False,
            ).flatten()
            if not len(valid):
                raise RuntimeError(
                    f"condition={condition} replica={replica} "
                    f"在threshold={threshold:.8g}没有幸存粒子"
                )
            survival[condition, replica] = (
                len(valid) / Config.PARTICLES_PER_REPLICA
            )
            invalid = torch.nonzero(
                state.losses[condition, replica]
                > threshold + Config.LOSS_TOLERANCE,
                as_tuple=False,
            ).flatten()
            if len(invalid):
                selected = valid[torch.randint(
                    len(valid),
                    (len(invalid),),
                    generator=generator,
                    device=state.particles.device,
                )]
                new_particles[condition, replica, invalid] = state.particles[
                    condition, replica, selected
                ]
                new_lineages[condition, replica, invalid] = state.lineages[
                    condition, replica, selected
                ]
    state.particles = new_particles
    state.lineages = new_lineages
    state.losses = evaluate_losses(
        state.particles, train_inputs, train_targets
    )
    state.log_volume += torch.log(torch.as_tensor(
        survival, dtype=torch.float64, device=state.log_volume.device
    ))
    return survival


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    block_index: int,
    threshold: float,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    rho = state.proposal_scales[:, block_index].reshape(-1, 1, 1, 1)
    noise = torch.empty_like(current).normal_(
        mean=0.0, std=Config.GAUSSIAN_SIGMA, generator=generator
    )
    proposal[..., block.start:block.stop] = (
        torch.sqrt(1.0 - rho * rho) * current + rho * noise
    )
    proposal_loss = evaluate_losses(proposal, train_inputs, train_targets)
    accept = proposal_loss <= threshold + Config.LOSS_TOLERANCE
    state.particles = torch.where(
        accept[..., None], proposal, state.particles
    )
    state.losses = torch.where(accept, proposal_loss, state.losses)
    return accept.float().mean(dim=(1, 2)).cpu().numpy()


def rejuvenate(
    state: SMCState,
    blocks: Sequence[ParameterBlock],
    threshold: float,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    condition_count = state.particles.shape[0]
    for _ in range(Config.ADAPT_SWEEPS):
        for block_index, block in enumerate(blocks):
            acceptance = mutate_block(
                state, block, block_index, threshold,
                train_inputs, train_targets, generator,
            )
            rates = torch.as_tensor(
                acceptance, device=state.proposal_scales.device
            )
            state.proposal_scales[:, block_index] *= torch.exp(
                Config.ADAPT_RATE * (rates - Config.TARGET_ACCEPTANCE)
            )
            state.proposal_scales[:, block_index].clamp_(
                Config.MIN_PROPOSAL_SCALE, Config.MAX_PROPOSAL_SCALE
            )
    totals = np.zeros((condition_count, len(blocks)), dtype=np.float64)
    for _ in range(Config.MUTATION_SWEEPS):
        for block_index, block in enumerate(blocks):
            totals[:, block_index] += mutate_block(
                state, block, block_index, threshold,
                train_inputs, train_targets, generator,
            )
    return totals / Config.MUTATION_SWEEPS


def finite_pair_agreement(predictions: np.ndarray) -> float:
    particle_count = len(predictions)
    if particle_count < 2 or predictions.shape[1] == 0:
        return float("nan")
    ones = predictions.sum(axis=0).astype(np.float64)
    same = (
        ones * (ones - 1)
        + (particle_count - ones) * (particle_count - ones - 1)
    )
    return float(np.mean(same / (particle_count * (particle_count - 1))))


def packed_function_hex(packed: np.ndarray) -> str:
    bits = np.unpackbits(packed, bitorder="little")[: 2 ** Config.INPUT_BITS]
    return outputs_to_hex(bits)


@torch.no_grad()
def summarize_threshold(
    state: SMCState,
    threshold: float,
    conditions: Sequence[ConditionSpec],
    full_inputs: torch.Tensor,
    full_targets: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    hard = evaluate_full_hard(state.particles, full_inputs).cpu().numpy()
    input_bits = truth_table_inputs()
    copy_x1 = input_bits[:, 1].astype(np.uint8)
    copy_x2 = input_bits[:, 2].astype(np.uint8)
    parity_alt = np.logical_not(
        np.logical_xor(input_bits[:, 0], input_bits[:, 2])
    ).astype(np.uint8)
    all_indices = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)

    condition_rows = []
    replica_rows = []
    top_rows = []
    prediction_rows = []
    for condition in conditions:
        local = hard[condition.index]
        target = full_targets[condition.index]
        train_indices = np.asarray(condition.indices, dtype=np.int64)
        unseen = np.setdiff1d(all_indices, train_indices, assume_unique=True)
        flattened = local.reshape(-1, local.shape[-1]).astype(np.uint8)
        target_exact = np.all(flattened == target[None], axis=1)
        train_exact = np.all(
            flattened[:, train_indices] == target[train_indices][None], axis=1
        )
        packed = np.packbits(flattened, axis=1, bitorder="little")
        unique, counts = np.unique(packed, axis=0, return_counts=True)
        order = np.argsort(-counts)
        unique, counts = unique[order], counts[order]
        collision = float(np.sum(
            counts.astype(np.float64) * (counts - 1)
        ) / (len(flattened) * (len(flattened) - 1)))
        log_volume = state.log_volume[condition.index].detach().cpu().numpy()
        lineage_counts = []
        replica_agreements = []
        replica_accuracies = []
        replica_target_mass = []
        for replica in range(Config.REPLICAS):
            replica_predictions = local[replica].astype(np.uint8)
            replica_target = np.all(
                replica_predictions == target[None], axis=1
            )
            lineage_count = int(torch.unique(
                state.lineages[condition.index, replica]
            ).numel())
            lineage_counts.append(lineage_count)
            agreement = finite_pair_agreement(
                replica_predictions[:, unseen]
            )
            accuracy = float(np.mean(
                replica_predictions[:, unseen] == target[unseen][None]
            ))
            replica_agreements.append(agreement)
            replica_accuracies.append(accuracy)
            replica_target_mass.append(float(replica_target.mean()))
            replica_rows.append({
                "threshold": threshold,
                "condition_index": condition.index,
                "target_name": condition.target_name,
                "sampling_protocol": condition.sampling_protocol,
                "dataset_index": condition.dataset_index,
                "replica": replica,
                "log_volume": float(log_volume[replica]),
                "lineage_count": lineage_count,
                "train_hard_exact_mass": float(np.mean(np.all(
                    replica_predictions[:, train_indices]
                    == target[train_indices][None], axis=1
                ))),
                "target_function_mass": float(replica_target.mean()),
                "unseen_target_accuracy": accuracy,
                "unseen_agreement": agreement,
            })

        condition_rows.append({
            "threshold": threshold,
            "condition_index": condition.index,
            "target_name": condition.target_name,
            "sampling_protocol": condition.sampling_protocol,
            "dataset_index": condition.dataset_index,
            "dataset_signature": condition.signature,
            "indices": condition.indices,
            "mux_cell_counts": condition.mux_cell_counts,
            "target_cell_counts": condition.target_cell_counts,
            "conflict_fraction": condition.conflict_fraction,
            "training_copy_x1_accuracy": condition.copy_x1_accuracy,
            "training_copy_x2_accuracy": condition.copy_x2_accuracy,
            "training_parity_alt_accuracy": condition.parity_alt_accuracy,
            "loss_mean": float(
                state.losses[condition.index].mean().item()
            ),
            "loss_median": float(
                state.losses[condition.index].median().item()
            ),
            "log_volume_median": float(np.median(log_volume)),
            "log_volume_min": float(log_volume.min()),
            "log_volume_max": float(log_volume.max()),
            "lineage_count_median": float(np.median(lineage_counts)),
            "train_hard_exact_mass": float(train_exact.mean()),
            "target_function_mass": float(target_exact.mean()),
            "unseen_target_accuracy": float(np.mean(
                flattened[:, unseen] == target[unseen][None]
            )),
            "unseen_agreement_replica_mean": float(np.mean(
                replica_agreements
            )),
            "target_mass_replica_mean": float(np.mean(
                replica_target_mass
            )),
            "full_target_accuracy": float(np.mean(
                flattened == target[None]
            )),
            "copy_x1_full_similarity": float(np.mean(
                flattened == copy_x1[None]
            )),
            "copy_x2_full_similarity": float(np.mean(
                flattened == copy_x2[None]
            )),
            "parity_alt_full_similarity": float(np.mean(
                flattened == parity_alt[None]
            )),
            "function_collision": collision,
            "unique_function_count": int(len(unique)),
            "modal_probability": float(counts[0] / counts.sum()),
            "modal_function_hex": packed_function_hex(unique[0]),
            "modal_is_target": bool(np.array_equal(
                np.unpackbits(unique[0], bitorder="little")[:len(target)],
                target,
            )),
        })

        for rank in range(min(Config.TOP_FUNCTIONS, len(unique))):
            bits = np.unpackbits(
                unique[rank], bitorder="little"
            )[: 2 ** Config.INPUT_BITS].astype(np.uint8)
            top_rows.append({
                "threshold": threshold,
                "condition_index": condition.index,
                "target_name": condition.target_name,
                "sampling_protocol": condition.sampling_protocol,
                "dataset_index": condition.dataset_index,
                "rank": rank + 1,
                "function_hex": outputs_to_hex(bits),
                "probability": float(counts[rank] / counts.sum()),
                "target_accuracy": float(np.mean(bits == target)),
                "copy_x1_similarity": float(np.mean(bits == copy_x1)),
                "copy_x2_similarity": float(np.mean(bits == copy_x2)),
                "parity_alt_similarity": float(np.mean(bits == parity_alt)),
            })

        posterior_one = flattened.mean(axis=0)
        posterior_target = np.where(target == 1, posterior_one, 1-posterior_one)
        train_set = set(map(int, condition.indices))
        relevant_cells = (
            2 * input_bits[:, 0] + input_bits[:, 1]
            if condition.target_name == "parity2"
            else mux_cell_ids(input_bits)
        )
        for input_index in range(len(input_bits)):
            prediction_rows.append({
                "threshold": threshold,
                "condition_index": condition.index,
                "target_name": condition.target_name,
                "sampling_protocol": condition.sampling_protocol,
                "dataset_index": condition.dataset_index,
                "input_index": input_index,
                "input_bits": "".join(map(str, input_bits[input_index])),
                "is_training_sample": input_index in train_set,
                "relevant_cell": int(relevant_cells[input_index]),
                "target": int(target[input_index]),
                "posterior_probability_one": float(
                    posterior_one[input_index]
                ),
                "posterior_probability_target": float(
                    posterior_target[input_index]
                ),
            })
    return condition_rows, replica_rows, top_rows, prediction_rows


def aggregate_rows(
    rows: Sequence[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    result = []
    local_threshold = [
        row for row in rows if abs(float(row["threshold"]) - threshold) < 1e-12
    ]
    for protocol in Config.SAMPLING_PROTOCOLS:
        for target_name in Config.TARGET_NAMES:
            local = [
                row for row in local_threshold
                if row["sampling_protocol"] == protocol
                and row["target_name"] == target_name
            ]
            result.append({
                "threshold": threshold,
                "sampling_protocol": protocol,
                "target_name": target_name,
                "dataset_count": len(local),
                "train_hard_exact_mass_mean": float(np.mean([
                    row["train_hard_exact_mass"] for row in local
                ])),
                "target_function_mass_mean": float(np.mean([
                    row["target_function_mass"] for row in local
                ])),
                "unseen_target_accuracy_mean": float(np.mean([
                    row["unseen_target_accuracy"] for row in local
                ])),
                "unseen_agreement_mean": float(np.mean([
                    row["unseen_agreement_replica_mean"] for row in local
                ])),
                "copy_x1_similarity_mean": float(np.mean([
                    row["copy_x1_full_similarity"] for row in local
                ])),
                "copy_x2_similarity_mean": float(np.mean([
                    row["copy_x2_full_similarity"] for row in local
                ])),
                "parity_alt_similarity_mean": float(np.mean([
                    row["parity_alt_full_similarity"] for row in local
                ])),
                "log_volume_median_over_datasets": float(np.median([
                    row["log_volume_median"] for row in local
                ])),
            })
    return result


def prepare_output(
    targets: Sequence[TargetSpec],
    conditions: Sequence[ConditionSpec],
) -> tuple[Path, dict[str, Any]]:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "e23_parity2_mux3_fixed_d_static_smc_v1",
        "created_before_sampling": True,
        "config": config_payload(),
        "targets": [json_ready(target.__dict__) for target in targets],
        "conditions": [json_ready(condition.__dict__) for condition in conditions],
        "primary_question": (
            "Does static fixed-D conditioning reproduce the selective "
            "MUX3 advantage of conflict-enriched sampling seen under AdamW?"
        ),
        "interpretation": {
            "static_reproduces": (
                "fixed-D competition denominator is a first-order cause"
            ),
            "static_does_not_reproduce": (
                "optimizer transport is required for the observed separation"
            ),
        },
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    path = output / "preregistered_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != json_ready(protocol):
            raise RuntimeError("结果目录已有不同预注册协议。")
        if not Config.RESUME:
            raise RuntimeError("结果目录已存在且RESUME=False。")
    else:
        write_json(path, protocol)
    return output, protocol


def state_payload(
    state: SMCState,
    protocol_hash: str,
    generators: dict[str, torch.Generator],
) -> dict[str, Any]:
    return {
        "protocol_hash": protocol_hash,
        "particles": state.particles.detach().cpu(),
        "losses": state.losses.detach().cpu(),
        "lineages": state.lineages.detach().cpu(),
        "log_volume": state.log_volume.detach().cpu(),
        "proposal_scales": state.proposal_scales.detach().cpu(),
        "current_threshold": state.current_threshold,
        "target_index": state.target_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "threshold_rows": state.threshold_rows,
        "replica_rows": state.replica_rows,
        "top_rows": state.top_rows,
        "prediction_rows": state.prediction_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }


def save_checkpoint(
    output: Path,
    state: SMCState,
    protocol_hash: str,
    generators: dict[str, torch.Generator],
) -> None:
    temporary = output / "checkpoint.tmp.pt"
    torch.save(
        state_payload(state, protocol_hash, generators), temporary
    )
    temporary.replace(output / "checkpoint.pt")


def load_or_initialize(
    output: Path,
    protocol_hash: str,
    condition_count: int,
    parameter_count: int,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> SMCState:
    checkpoint = output / "checkpoint.pt"
    if not checkpoint.exists() or not Config.RESUME:
        return initialize_state(
            condition_count, parameter_count,
            train_inputs, train_targets, device, generators["prior"]
        )
    payload = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    if payload["protocol_hash"] != protocol_hash:
        raise RuntimeError("checkpoint协议哈希不一致。")
    for name, generator in generators.items():
        generator.set_state(payload["generator_states"][name].to("cpu"))
    print(
        f"resumed checkpoint | level={int(payload['level']):,} | "
        f"target_index={int(payload['target_index'])}",
        flush=True,
    )
    return SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume=payload["log_volume"].to(device),
        proposal_scales=payload["proposal_scales"].to(device),
        current_threshold=float(payload["current_threshold"]),
        target_index=int(payload["target_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        threshold_rows=list(payload["threshold_rows"]),
        replica_rows=list(payload["replica_rows"]),
        top_rows=list(payload["top_rows"]),
        prediction_rows=list(payload["prediction_rows"]),
    )


def save_tables(output: Path, state: SMCState) -> None:
    write_csv(output / "smc_levels.csv", state.level_rows)
    write_csv(output / "condition_threshold_summary.csv", state.threshold_rows)
    write_csv(output / "replica_threshold_summary.csv", state.replica_rows)
    write_csv(output / "top_functions.csv", state.top_rows)
    write_csv(output / "posterior_predictions.csv", state.prediction_rows)
    aggregate = []
    for threshold in Config.TARGET_THRESHOLDS[:state.target_index]:
        aggregate.extend(aggregate_rows(state.threshold_rows, threshold))
    write_csv(output / "protocol_target_summary.csv", aggregate)


def save_plot(output: Path, state: SMCState) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    aggregate = []
    for threshold in Config.TARGET_THRESHOLDS[:state.target_index]:
        aggregate.extend(aggregate_rows(state.threshold_rows, threshold))
    if not aggregate:
        return
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = (
        ("unseen_target_accuracy_mean", "unseen target accuracy"),
        ("target_function_mass_mean", "exact target function mass"),
        ("unseen_agreement_mean", "unseen agreement"),
    )
    for protocol in Config.SAMPLING_PROTOCOLS:
        for target_name in Config.TARGET_NAMES:
            local = [
                row for row in aggregate
                if row["sampling_protocol"] == protocol
                and row["target_name"] == target_name
            ]
            x = [-math.log10(float(row["threshold"])) for row in local]
            label = f"{protocol[:4]}/{target_name}"
            for axis, (metric, title) in zip(axes, metrics):
                axis.plot(
                    x, [row[metric] for row in local],
                    marker="o", ms=3, label=label,
                )
                axis.set_title(title)
                axis.set_xlabel("-log10(raw BCE threshold)")
                axis.set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "fixed_d_static_smc_curves.png", dpi=180)
    plt.close(figure)


def create_archive(output: Path) -> Path:
    archive = output.parent / f"{output.name}_package.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.name.startswith("checkpoint"):
                continue
            handle.write(path, arcname=f"{output.name}/{path.relative_to(output)}")
    return archive


def main() -> None:
    apply_smoke_overrides()
    targets = build_targets()
    conditions = build_conditions(targets)
    output, protocol = prepare_output(targets, conditions)
    device = torch.device(Config.DEVICE)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    train_inputs, train_targets, full_inputs, full_targets = prepare_tensors(
        conditions, targets, device
    )
    blocks, parameter_count = parameter_blocks()
    generators = make_generators(device)
    state = load_or_initialize(
        output, protocol["protocol_sha256"], len(conditions),
        parameter_count, train_inputs, train_targets, device, generators,
    )

    print("=== Parity2 / MUX3 fixed-D static SMC ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=8->{Config.WIDTH}x2->1 tanh | params={parameter_count} | "
        f"n={Config.TRAIN_COUNT} | conditions={len(conditions)} | "
        f"datasets/protocol={Config.DATASETS_PER_PROTOCOL}",
        flush=True,
    )
    print(
        f"SMC={Config.REPLICAS}x{Config.PARTICLES_PER_REPLICA:,}/condition | "
        f"reference=N(0,{Config.GAUSSIAN_SIGMA:.6g}^2)",
        flush=True,
    )
    print(f"result_dir={output}", flush=True)

    started = time.perf_counter()
    try:
        while state.target_index < len(Config.TARGET_THRESHOLDS):
            if state.level >= Config.MAX_LEVELS:
                raise RuntimeError("达到MAX_LEVELS仍未完成目标阈值。")
            requested = Config.TARGET_THRESHOLDS[state.target_index]
            threshold, reached = choose_threshold(state, requested)
            if (
                math.isfinite(state.current_threshold)
                and threshold >= state.current_threshold - Config.MIN_LEVEL_DECREMENT
                and not reached
            ):
                raise RuntimeError("SMC阈值停止下降，需要检查混合。")
            survival = resample_state(
                state, threshold, train_inputs, train_targets,
                generators["resample"],
            )
            acceptance = rejuvenate(
                state, blocks, threshold, train_inputs, train_targets,
                generators["mutation"],
            )
            state.current_threshold = threshold
            state.level += 1
            state.level_rows.append({
                "level": state.level,
                "requested_threshold": requested,
                "threshold": threshold,
                "reached_requested": reached,
                "survival_mean": float(survival.mean()),
                "survival_min": float(survival.min()),
                "survival_max": float(survival.max()),
                "acceptance_mean": float(acceptance.mean()),
                "acceptance_min": float(acceptance.min()),
                "acceptance_max": float(acceptance.max()),
                "elapsed_seconds": time.perf_counter() - started,
            })
            if state.level % Config.LOG_EVERY_LEVELS == 0 or reached:
                print(
                    f"level={state.level:>5,} | eps={threshold:.7g} | "
                    f"survive={survival.mean():.1%} "
                    f"[{survival.min():.1%},{survival.max():.1%}] | "
                    f"accept={acceptance.mean():.1%} | "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
            if reached:
                condition_rows, replica_rows, top_rows, prediction_rows = (
                    summarize_threshold(
                        state, requested, conditions,
                        full_inputs, full_targets,
                    )
                )
                state.threshold_rows.extend(condition_rows)
                state.replica_rows.extend(replica_rows)
                state.top_rows.extend(top_rows)
                state.prediction_rows.extend(prediction_rows)
                state.target_index += 1
                aggregate = aggregate_rows(condition_rows, requested)
                compact = []
                for row in aggregate:
                    compact.append(
                        f"{row['sampling_protocol'][:4]}/"
                        f"{row['target_name']}:fit="
                        f"{row['train_hard_exact_mass_mean']:.3f},"
                        f"U={row['unseen_target_accuracy_mean']:.3f},"
                        f"T={row['target_function_mass_mean']:.3f},"
                        f"A={row['unseen_agreement_mean']:.3f}"
                    )
                print(
                    f"[TARGET eps={requested:g}] " + " | ".join(compact),
                    flush=True,
                )
                save_tables(output, state)
            if (
                state.level % Config.CHECKPOINT_EVERY_LEVELS == 0
                or reached
            ):
                save_checkpoint(
                    output, state, protocol["protocol_sha256"], generators
                )
    except KeyboardInterrupt:
        save_checkpoint(
            output, state, protocol["protocol_sha256"], generators
        )
        save_tables(output, state)
        write_json(output / "summary.json", {
            "status": "interrupted",
            "completed_thresholds": state.target_index,
            "level": state.level,
        })
        print("已保存checkpoint；重新运行即可继续。", flush=True)
        return

    save_tables(output, state)
    save_plot(output, state)
    write_json(output / "summary.json", {
        "status": "completed",
        "completed_thresholds": state.target_index,
        "level": state.level,
        "elapsed_seconds": time.perf_counter() - started,
    })
    (output / "checkpoint.pt").unlink(missing_ok=True)
    if Config.PACKAGE_RESULTS:
        print(f"archive={create_archive(output)}", flush=True)


if __name__ == "__main__":
    main()
