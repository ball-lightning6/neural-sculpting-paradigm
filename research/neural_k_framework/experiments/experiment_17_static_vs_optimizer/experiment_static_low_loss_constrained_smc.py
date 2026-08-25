"""
用 constrained SMC / subset simulation 采样极稀有 low-loss 静态函数分布。

目标测度是初始化参数立方体上的均匀测度，条件为 raw BCE <= epsilon：

    pi_epsilon(theta) proportional to 1[theta in initialization support]
                                      1[L_D(theta) <= epsilon]

这不是 SGD，也不使用 loss 梯度。SMC 逐层收紧 epsilon，保留低 loss 粒子、
在每个独立副本内重采样，再用保持条件均匀分布不变的对称反射随机游走恢复
多样性。多个独立副本用于诊断模式丢失和 MCMC 混合不足。
"""

from __future__ import annotations

import csv
import json
import math
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
    TRAIN_INDICES = (1, 2, 3, 5, 7, 8, 11, 12, 14, 15)

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
        0.065,
        0.05,
    )
    MAX_LEVELS = 1_000
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

    EVAL_MICRO_BATCH = 8_192
    SHELL_WIDTH = 0.005
    MIN_SHELL_SAMPLES = 200
    TOP_FUNCTIONS = 20

    PRIOR_SEED = 20260920
    RESAMPLE_SEED = 20260921
    MUTATION_SEED = 20260922
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_static_low_loss_constrained_smc")
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    CHECKPOINT_EVERY_LEVELS = 1
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


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


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 128
    Config.TARGET_THRESHOLDS = (0.70, 0.68, 0.65)
    Config.MAX_LEVELS = 12
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 2
    Config.EVAL_MICRO_BATCH = 256
    Config.MIN_SHELL_SAMPLES = 4
    Config.TOP_FUNCTIONS = 5
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_static_low_loss_constrained_smc"
    )
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
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
    if Config.INPUT_BITS != 4 or Config.HIDDEN_LAYERS != 2:
        raise ValueError("当前实现固定为4-bit输入、两个隐藏层。")
    thresholds = tuple(float(value) for value in Config.TARGET_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("TARGET_THRESHOLDS 必须严格从高到低且不重复。")
    if not 0 < Config.SURVIVAL_QUANTILE < 1:
        raise ValueError("SURVIVAL_QUANTILE 必须在(0,1)内。")
    if Config.REPLICAS < 2:
        raise ValueError("至少需要两个独立副本诊断混合。")
    if len(Config.INITIAL_PROPOSAL_SCALES) != 4:
        raise ValueError("需要为三个参数块和全参数块提供四个 proposal scale。")


def prepare_result_dir() -> tuple[Path, bool]:
    output = Path(Config.RESULT_DIR)
    checkpoint = output / "checkpoint.pt"
    if output.exists() and Config.RESUME and checkpoint.exists():
        return output, True
    if output.exists():
        if Config.OVERWRITE_RESULT_DIR:
            shutil.rmtree(output)
        else:
            output = output.parent / (
                output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
            )
    output.mkdir(parents=True, exist_ok=True)
    return output, False


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def and_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs[:, 0] & inputs[:, 1]).astype(np.uint8)


def outputs_to_function_id(outputs: np.ndarray) -> int:
    bits = np.asarray(outputs, dtype=np.uint64).reshape(-1)
    powers = np.left_shift(np.uint64(1), np.arange(16, dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def function_bits(function_id: int) -> str:
    return "".join(str((function_id >> index) & 1) for index in range(16))


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
def evaluate_losses(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_indices: torch.Tensor,
) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces: list[torch.Tensor] = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(flat[start:start + Config.EVAL_MICRO_BATCH], inputs)
        local_logits = logits[:, train_indices]
        local_targets = targets[train_indices][None].expand_as(local_logits)
        pieces.append(F.binary_cross_entropy_with_logits(
            local_logits, local_targets, reduction="none"
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


def initialize_state(
    device: torch.device,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_indices: torch.Tensor,
    prior_generator: torch.Generator,
    parameter_count: int,
) -> SMCState:
    particles = torch.empty(
        Config.REPLICAS,
        Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    ).uniform_(-1.0, 1.0, generator=prior_generator)
    losses = evaluate_losses(particles, inputs, targets, train_indices)
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
    target = float(Config.TARGET_THRESHOLDS[state.target_index])
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
    state.losses = evaluate_losses(
        state.particles, GLOBAL_INPUTS, GLOBAL_TARGETS, GLOBAL_TRAIN_INDICES
    )
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
    proposal_losses = evaluate_losses(
        proposal, GLOBAL_INPUTS, GLOBAL_TARGETS, GLOBAL_TRAIN_INDICES
    )
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
                Config.ADAPT_RATE
                * (acceptance - Config.TARGET_ACCEPTANCE)
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


def append_top_functions(
    rows: list[dict[str, Any]],
    label: str,
    counts: np.ndarray,
    target_id: int,
) -> None:
    total = counts.sum()
    if total == 0:
        return
    for rank, function_id in enumerate(
        np.argsort(counts)[::-1][:Config.TOP_FUNCTIONS], start=1
    ):
        count = int(counts[function_id])
        if count == 0:
            break
        rows.append({
            "distribution": label,
            "rank": rank,
            "function_id": int(function_id),
            "function_hex": f"0x{int(function_id):04X}",
            "truth_table_x0_to_x15": function_bits(int(function_id)),
            "count": count,
            "probability": count / total,
            "is_target": int(function_id) == target_id,
        })


def record_target(
    output_dir: Path,
    state: SMCState,
    threshold: float,
    inputs: torch.Tensor,
    target_id: int,
) -> None:
    ids = evaluate_function_ids(state.particles, inputs)
    losses = state.losses.detach().cpu().numpy()
    aggregate_counts = np.bincount(ids.reshape(-1), minlength=65_536)
    shell_mask = losses >= threshold - Config.SHELL_WIDTH
    shell_ids = ids[shell_mask]
    shell_counts = np.bincount(shell_ids, minlength=65_536)
    aggregate_probability = aggregate_counts / aggregate_counts.sum()

    replica_js: list[float] = []
    replica_target: list[float] = []
    for replica in range(Config.REPLICAS):
        replica_counts = np.bincount(ids[replica], minlength=65_536)
        replica_js.append(js_divergence_from_counts(
            replica_counts, aggregate_counts
        ))
        replica_target.append(
            float(replica_counts[target_id] / replica_counts.sum())
        )
        state.replica_rows.append({
            "threshold": threshold,
            "replica": replica,
            "estimated_log_volume_fraction": float(
                state.log_volume_fraction[replica].item()
            ),
            "estimated_volume_fraction": float(math.exp(
                state.log_volume_fraction[replica].item()
            )),
            "target_probability": replica_target[-1],
            "function_entropy_bits": distribution_entropy(replica_counts),
            "function_support": int(np.count_nonzero(replica_counts)),
            "js_to_aggregate": replica_js[-1],
            "unique_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        })

    volume = torch.exp(state.log_volume_fraction).detach().cpu().numpy()
    row = {
        "threshold": threshold,
        "level": state.level,
        "particle_count": int(aggregate_counts.sum()),
        "estimated_volume_fraction_median": float(np.median(volume)),
        "estimated_volume_fraction_min": float(np.min(volume)),
        "estimated_volume_fraction_max": float(np.max(volume)),
        "loss_min": float(losses.min()),
        "loss_median": float(np.median(losses)),
        "loss_max": float(losses.max()),
        "target_probability": float(aggregate_probability[target_id]),
        "function_entropy_bits": distribution_entropy(aggregate_counts),
        "function_support": int(np.count_nonzero(aggregate_counts)),
        "top_function_id": int(np.argmax(aggregate_counts)),
        "top_function_hex": f"0x{int(np.argmax(aggregate_counts)):04X}",
        "top_function_probability": float(aggregate_counts.max() / aggregate_counts.sum()),
        "shell_width": Config.SHELL_WIDTH,
        "shell_sample_count": int(len(shell_ids)),
        "shell_supported": bool(len(shell_ids) >= Config.MIN_SHELL_SAMPLES),
        "shell_target_probability": (
            float(shell_counts[target_id] / shell_counts.sum())
            if shell_counts.sum() else float("nan")
        ),
        "shell_function_entropy_bits": distribution_entropy(shell_counts),
        "replica_target_probability_min": float(np.min(replica_target)),
        "replica_target_probability_max": float(np.max(replica_target)),
        "replica_js_to_aggregate_median": float(np.median(replica_js)),
        "replica_js_to_aggregate_max": float(np.max(replica_js)),
    }
    state.target_rows.append(row)
    label = f"smc_cumulative_le_{threshold:g}"
    append_top_functions(state.top_rows, label, aggregate_counts, target_id)
    if shell_counts.sum():
        append_top_functions(
            state.top_rows,
            f"smc_shell_{threshold:g}",
            shell_counts,
            target_id,
        )

    snapshot_name = f"snapshot_threshold_{threshold:.3f}".replace(".", "p")
    np.savez_compressed(
        output_dir / f"{snapshot_name}.npz",
        threshold=np.asarray(threshold),
        function_ids=ids,
        losses=losses,
        lineages=state.lineages.detach().cpu().numpy(),
        aggregate_counts=aggregate_counts,
        shell_counts=shell_counts,
        log_volume_fraction=state.log_volume_fraction.detach().cpu().numpy(),
    )

    print(
        f"TARGET {threshold:.3f} | volume median={np.median(volume):.3e} "
        f"[{np.min(volume):.3e},{np.max(volume):.3e}] | "
        f"AND={row['target_probability']:.3%} | "
        f"top={row['top_function_hex']}:{row['top_function_probability']:.2%} | "
        f"H={row['function_entropy_bits']:.3f} | "
        f"replica JSD max={row['replica_js_to_aggregate_max']:.4f}",
        flush=True,
    )


def state_payload(
    state: SMCState,
    generators: dict[str, torch.Generator],
) -> dict[str, Any]:
    return {
        "config": config_dict(),
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
    output_dir: Path,
    state: SMCState,
    generators: dict[str, torch.Generator],
) -> None:
    checkpoint = output_dir / "checkpoint.pt"
    temporary = output_dir / "checkpoint.tmp.pt"
    torch.save(state_payload(state, generators), temporary)
    temporary.replace(checkpoint)


def load_checkpoint(
    output_dir: Path,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> SMCState:
    payload = torch.load(
        output_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    saved = payload["config"]
    for name in (
        "WIDTH",
        "REPLICAS",
        "PARTICLES_PER_REPLICA",
        "TRAIN_INDICES",
    ):
        if json_ready(getattr(Config, name)) != saved[name]:
            raise RuntimeError(f"checkpoint配置不一致：{name}")
    saved_targets = list(saved["TARGET_THRESHOLDS"])
    current_targets = list(json_ready(Config.TARGET_THRESHOLDS))
    if current_targets[:len(saved_targets)] != saved_targets:
        raise RuntimeError(
            "checkpoint的TARGET_THRESHOLDS不是当前目标序列的前缀。"
        )
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
        target_index=int(payload["target_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        target_rows=list(payload["target_rows"]),
        replica_rows=list(payload["replica_rows"]),
        top_rows=list(payload["top_rows"]),
    )


def write_artifacts(
    output_dir: Path,
    state: SMCState,
    status: str,
    target_id: int,
) -> None:
    write_csv(output_dir / "levels.csv", state.level_rows)
    write_csv(output_dir / "targets.csv", state.target_rows)
    write_csv(output_dir / "replica_targets.csv", state.replica_rows)
    write_csv(output_dir / "top_functions.csv", state.top_rows)
    write_json(output_dir / "summary.json", {
        "status": status,
        "protocol": "static_low_loss_constrained_smc_v1",
        "target_function_id": target_id,
        "target_function_hex": f"0x{target_id:04X}",
        "completed_targets": state.target_index,
        "requested_targets": len(Config.TARGET_THRESHOLDS),
        "current_threshold": state.current_threshold,
        "level": state.level,
        "target_rows": state.target_rows,
        "interpretation": {
            "target_measure": (
                "初始化参数立方体均匀测度在 raw BCE<=threshold 下的条件分布"
            ),
            "validity": (
                "先在0.68/0.65/0.60与暴力prior结果核对；副本间JSD高时，"
                "更深阈值只能视为未混合的探索结果。"
            ),
            "not_sgd": "mutation不使用loss梯度，且经对称MH约束核保持目标测度。",
        },
    })


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


GLOBAL_INPUTS: torch.Tensor
GLOBAL_TARGETS: torch.Tensor
GLOBAL_TRAIN_INDICES: torch.Tensor


def main() -> None:
    global GLOBAL_INPUTS, GLOBAL_TARGETS, GLOBAL_TRAIN_INDICES

    apply_smoke_overrides()
    validate_config()
    output_dir, should_resume = prepare_result_dir()
    write_json(output_dir / "config.json", config_dict())
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    inputs_np = truth_table_inputs()
    targets_np = and_targets(inputs_np)
    target_id = outputs_to_function_id(targets_np)
    GLOBAL_INPUTS = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    GLOBAL_TARGETS = torch.from_numpy(targets_np.astype(np.float32)).to(device)
    GLOBAL_TRAIN_INDICES = torch.tensor(
        Config.TRAIN_INDICES, dtype=torch.int64, device=device
    )
    blocks, parameter_count = parameter_blocks(Config.WIDTH)
    generators = make_generators(device)

    print("=== Static low-loss constrained SMC ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(
        f"task=balanced AND n=10 | target=0x{target_id:04X} | "
        f"network=4->{Config.WIDTH}x2->1 tanh | params={parameter_count}"
    )
    print(
        f"replicas={Config.REPLICAS} | particles/replica="
        f"{Config.PARTICLES_PER_REPLICA:,} | total="
        f"{Config.REPLICAS * Config.PARTICLES_PER_REPLICA:,}"
    )
    print(f"结果目录：{output_dir.resolve()}")

    if should_resume:
        state = load_checkpoint(output_dir, device, generators)
        print(
            f"恢复checkpoint：level={state.level} | "
            f"threshold={state.current_threshold:.6g} | "
            f"target_index={state.target_index}",
            flush=True,
        )
    else:
        state = initialize_state(
            device,
            GLOBAL_INPUTS,
            GLOBAL_TARGETS,
            GLOBAL_TRAIN_INDICES,
            generators["prior"],
            parameter_count,
        )
        save_checkpoint(output_dir, state, generators)

    started = time.perf_counter()
    status = "running"
    try:
        while (
            state.target_index < len(Config.TARGET_THRESHOLDS)
            and state.level < Config.MAX_LEVELS
        ):
            previous_threshold = state.current_threshold
            next_threshold, reaches_target = choose_next_threshold(state)
            if (
                math.isfinite(previous_threshold)
                and next_threshold >= previous_threshold - Config.MIN_LEVEL_DECREMENT
                and not reaches_target
            ):
                raise RuntimeError(
                    "SMC阈值停止下降；需要增加mutation或检查模式塌缩。"
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
                "level": state.level,
                "threshold": next_threshold,
                "reaches_requested_target": reaches_target,
                "next_requested_target": float(
                    Config.TARGET_THRESHOLDS[state.target_index]
                ),
                "survival_fraction_min": float(np.min(survival)),
                "survival_fraction_median": float(np.median(survival)),
                "survival_fraction_max": float(np.max(survival)),
                "loss_min": float(state.losses.min().item()),
                "loss_median": float(state.losses.median().item()),
                "loss_max": float(state.losses.max().item()),
                "estimated_volume_fraction_median": float(torch.exp(
                    state.log_volume_fraction
                ).median().item()),
                "elapsed_seconds": elapsed,
                "proposal_scales": list(state.proposal_scales),
            }
            row.update(mutation)
            state.level_rows.append(row)
            acceptance_text = ",".join(
                f"{block.name}:{mutation[f'acceptance_{block.name}']:.1%}"
                for block in blocks
            )
            print(
                f"level={state.level:>3} | eps={next_threshold:.6f} | "
                f"survive={np.min(survival):.1%}/{np.median(survival):.1%}/"
                f"{np.max(survival):.1%} | loss med={state.losses.median().item():.6f} "
                f"| volume~{row['estimated_volume_fraction_median']:.3e} | "
                f"accept[{acceptance_text}] | elapsed={elapsed:.1f}s",
                flush=True,
            )

            if reaches_target:
                threshold = float(Config.TARGET_THRESHOLDS[state.target_index])
                record_target(
                    output_dir, state, threshold, GLOBAL_INPUTS, target_id
                )
                state.target_index += 1
                write_artifacts(output_dir, state, "running", target_id)

            if state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(output_dir, state, generators)

        status = (
            "completed"
            if state.target_index == len(Config.TARGET_THRESHOLDS)
            else "stopped_max_levels"
        )
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n收到中断，正在保存checkpoint和当前结果……", flush=True)
    finally:
        save_checkpoint(output_dir, state, generators)
        write_artifacts(output_dir, state, status, target_id)
        archive_path: Path | None = None
        if Config.PACKAGE_RESULTS:
            archive_path = create_archive(output_dir)
        print(f"\n状态：{status}")
        print(f"汇总：{output_dir / 'summary.json'}")
        if archive_path is not None:
            print(f"下载压缩包：{archive_path}")


if __name__ == "__main__":
    main()
