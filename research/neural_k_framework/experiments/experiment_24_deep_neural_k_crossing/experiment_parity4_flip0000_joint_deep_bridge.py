"""Parity4与单例外函数的共同父系综深尾体积比实验。

目标：

    parity4          = 0x6996
    parity4_flip0000 = 0x6997

两个独立SMC的绝对归一化副本范围差异很大。本脚本先采样共同父事件：

    U = {min(L_parity4, L_flip0000) <= epsilon_parent}

再从同一批parent particles、同一组replica和lineage分出两个条件分支。共同父
log-volume在分支比值中严格抵消，从而直接测量V_parity4/V_flip0000。分支使用
保持标准Gaussian先验不变的pCN proposal，不使用loss梯度。

每个请求阈值还保存归一化signed margin、logit Walsh谱、output-weighted hidden
谱、hidden Gram target alignment和解析empirical NTK alignment的抽样摘要，用来
区分hard function固定后的单纯margin放大与内部表示重组。
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
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 4
    WIDTH = 16
    PARENT_THRESHOLD = 0.70
    DEPTH_STEP = 0.15
    DEPTH_LEVELS = 45
    TARGET_THRESHOLDS = tuple(
        float(0.70 * math.exp(-0.15 * index)) for index in range(45)
    )

    REPLICAS = 8
    PARTICLES_PER_REPLICA = 32_768
    SURVIVAL_QUANTILE = 0.5
    MAX_LEVELS_PARENT = 500
    MAX_LEVELS_BRANCH = 10_000
    MIN_LEVEL_DECREMENT = 1e-8

    ADAPT_SWEEPS = 4
    MUTATION_SWEEPS = 10
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.10, 0.10, 0.04)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.95
    LOSS_TOLERANCE = 1e-8

    EVAL_MICRO_BATCH = 32_768
    REPRESENTATION_SAMPLE_PER_REPLICA = 256
    CHECKPOINT_EVERY_LEVELS = 10
    LOG_EVERY_LEVELS = 10

    PRIOR_SEED = 2026082601
    PARENT_RESAMPLE_SEED = 2026082602
    PARENT_MUTATION_SEED = 2026082603
    PARITY_RESAMPLE_SEED = 2026082611
    PARITY_MUTATION_SEED = 2026082612
    FLIP_RESAMPLE_SEED = 2026082621
    FLIP_MUTATION_SEED = 2026082622

    STOP_CONSECUTIVE_WINDOWS = 5
    STOP_REQUIRE_ALL_REPLICAS_CROSSED = True
    STOP_REQUIRE_POSITIVE_RATE_DIFF = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_parity4_flip0000_joint_deep_bridge")
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


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
    threshold_index: int
    level: int
    level_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    replica_rows: list[dict[str, Any]]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.PARENT_THRESHOLD = 0.72
    Config.TARGET_THRESHOLDS = (0.72, 0.70, 0.68)
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 128
    Config.MAX_LEVELS_PARENT = 8
    Config.MAX_LEVELS_BRANCH = 40
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 2
    Config.EVAL_MICRO_BATCH = 256
    Config.REPRESENTATION_SAMPLE_PER_REPLICA = 8
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.STOP_CONSECUTIVE_WINDOWS = 2
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_parity4_flip0000_joint_deep_bridge"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def config_payload() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    }


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("parent", "parity4", "flip0000", "representations"):
        (output / name).mkdir(exist_ok=True)
    protocol = {
        "protocol": "parity4_flip0000_joint_parent_gaussian_pcn_v1",
        "created_before_sampling": True,
        "config": config_payload(),
        "targets": {
            "parity4": "0x6996",
            "flip0000": "0x6997",
        },
        "parent_event": "min(L_parity4,L_flip0000)<=PARENT_THRESHOLD",
        "paired_log_ratio": "log(V_parity4/V_flip0000)",
        "stopping_rule": {
            "windows": Config.STOP_CONSECUTIVE_WINDOWS,
            "all_replicas_ratio_positive": Config.STOP_REQUIRE_ALL_REPLICAS_CROSSED,
            "all_recent_rate_differences_positive": (
                Config.STOP_REQUIRE_POSITIVE_RATE_DIFF
            ),
        },
    }
    canonical = json.dumps(
        json_ready(protocol), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    protocol["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = output / "preregistered_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != json_ready(protocol):
            raise RuntimeError("已有结果的预注册协议与当前配置不一致。")
        if not Config.RESUME:
            raise RuntimeError("结果目录已存在且RESUME=False。")
    else:
        write_json(path, protocol)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float32)


def target_outputs() -> tuple[np.ndarray, np.ndarray]:
    inputs = truth_table_inputs().astype(np.uint8)
    parity = (inputs.sum(axis=1) % 2).astype(np.float32)
    flipped = parity.copy()
    flipped[0] = 1.0 - flipped[0]
    return parity, flipped


def parameter_blocks() -> tuple[list[ParameterBlock], int]:
    first = Config.WIDTH * Config.INPUT_BITS + Config.WIDTH
    output = Config.WIDTH + 1
    blocks = [
        ParameterBlock("first_layer", 0, first),
        ParameterBlock("output_layer", first, first + output),
        ParameterBlock("all_parameters", 0, first + output),
    ]
    return blocks, first + output


def unpack_parameters(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = coordinates.shape[0]
    cursor = 0
    first_size = Config.WIDTH * Config.INPUT_BITS
    first_weight = coordinates[:, cursor:cursor + first_size].reshape(
        count, Config.WIDTH, Config.INPUT_BITS
    ) / math.sqrt(Config.INPUT_BITS)
    cursor += first_size
    first_bias = coordinates[:, cursor:cursor + Config.WIDTH] / math.sqrt(
        Config.INPUT_BITS
    )
    cursor += Config.WIDTH
    output_weight = coordinates[:, cursor:cursor + Config.WIDTH] / math.sqrt(
        Config.WIDTH
    )
    cursor += Config.WIDTH
    output_bias = coordinates[:, cursor:cursor + 1] / math.sqrt(Config.WIDTH)
    return first_weight, first_bias, output_weight, output_bias


def forward_hidden_logits(
    coordinates: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_weight, first_bias, output_weight, output_bias = unpack_parameters(
        coordinates
    )
    hidden = torch.tanh(
        torch.bmm(
            inputs[None].expand(len(coordinates), -1, -1),
            first_weight.transpose(1, 2),
        ) + first_bias[:, None]
    )
    logits = torch.bmm(
        hidden, output_weight[:, :, None]
    ).squeeze(-1) + output_bias
    return hidden, logits


@torch.no_grad()
def evaluate_both_losses(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    parity_target: torch.Tensor,
    flip_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = particles.reshape(-1, particles.shape[-1])
    parity_pieces = []
    flip_pieces = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        _, logits = forward_hidden_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        parity_pieces.append(F.binary_cross_entropy_with_logits(
            logits, parity_target[None].expand_as(logits), reduction="none"
        ).mean(dim=1))
        flip_pieces.append(F.binary_cross_entropy_with_logits(
            logits, flip_target[None].expand_as(logits), reduction="none"
        ).mean(dim=1))
    shape = particles.shape[:-1]
    return torch.cat(parity_pieces).reshape(shape), torch.cat(flip_pieces).reshape(shape)


def make_evaluator(
    mode: str,
    inputs: torch.Tensor,
    parity_target: torch.Tensor,
    flip_target: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def evaluate(particles: torch.Tensor) -> torch.Tensor:
        parity_loss, flip_loss = evaluate_both_losses(
            particles, inputs, parity_target, flip_target
        )
        if mode == "union":
            return torch.minimum(parity_loss, flip_loss)
        if mode == "parity4":
            return parity_loss
        if mode == "flip0000":
            return flip_loss
        raise ValueError(f"未知模式：{mode}")
    return evaluate


def initialize_parent(
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
    ).normal_(generator=generator)
    losses = evaluator(particles)
    lineages = torch.arange(
        Config.REPLICAS * Config.PARTICLES_PER_REPLICA,
        device=device, dtype=torch.int64,
    ).reshape(Config.REPLICAS, Config.PARTICLES_PER_REPLICA)
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume_fraction=torch.zeros(
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


def choose_next_threshold(
    state: SMCState, requested: float
) -> tuple[float, bool]:
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=1
    )
    adaptive = float(quantiles.max().item())
    threshold = max(float(requested), adaptive)
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
                f"replica={replica}在threshold={threshold:.8g}没有幸存粒子"
            )
        survival[replica] = len(valid) / Config.PARTICLES_PER_REPLICA
        choices = torch.randint(
            len(valid), (Config.PARTICLES_PER_REPLICA,),
            generator=generator, device=state.particles.device,
        )
        selected = valid[choices]
        particles[replica] = state.particles[replica, selected]
        lineages[replica] = state.lineages[replica, selected]
    state.particles = particles
    state.lineages = lineages
    state.losses = evaluator(state.particles)
    state.log_volume_fraction += torch.log(torch.as_tensor(
        survival, device=state.log_volume_fraction.device, dtype=torch.float64
    ))
    return survival


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    rho: float,
    threshold: float,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> float:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    rho = min(max(float(rho), 0.0), 0.999999)
    noise = torch.randn(
        current.shape, device=current.device,
        dtype=current.dtype, generator=generator,
    )
    proposal[..., block.start:block.stop] = (
        math.sqrt(1.0 - rho * rho) * current + rho * noise
    )
    proposal_loss = evaluator(proposal)
    accept = proposal_loss <= threshold + Config.LOSS_TOLERANCE
    flat_accept = accept.reshape(-1)
    flat_particles = state.particles.reshape(-1, state.particles.shape[-1])
    flat_proposal = proposal.reshape(-1, proposal.shape[-1])
    flat_particles[flat_accept] = flat_proposal[flat_accept]
    flat_losses = state.losses.reshape(-1)
    flat_losses[flat_accept] = proposal_loss.reshape(-1)[flat_accept]
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
                Config.ADAPT_RATE * (acceptance - Config.TARGET_ACCEPTANCE)
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
        f"acceptance_{block.name}": float(totals[index] / Config.MUTATION_SWEEPS)
        for index, block in enumerate(blocks)
    }


def state_payload(
    state: SMCState,
    mode: str,
    generator_states: dict[str, torch.Generator],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "config": config_payload(),
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
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generator_states.items()
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
    temporary.replace(directory / "checkpoint.pt")


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
        raise RuntimeError(f"checkpoint mode不匹配：{mode}")
    saved = payload["config"]
    for key in (
        "INPUT_BITS", "WIDTH", "REPLICAS", "PARTICLES_PER_REPLICA",
        "PARENT_THRESHOLD", "PRIOR_SEED",
    ):
        if saved.get(key) != config_payload().get(key):
            raise RuntimeError(f"checkpoint配置不匹配：{key}")
    saved_thresholds = list(saved["TARGET_THRESHOLDS"])
    current_thresholds = list(config_payload()["TARGET_THRESHOLDS"])
    if current_thresholds[:len(saved_thresholds)] != saved_thresholds:
        raise RuntimeError("当前threshold序列不是checkpoint序列的扩展。")
    for name, generator in generators.items():
        if name in payload["generator_states"]:
            generator.set_state(
                payload["generator_states"][name].to(dtype=torch.uint8)
            )
    return SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume_fraction=payload["log_volume_fraction"].to(
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


def walsh_matrix(device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    matrix = np.empty((16, 16), dtype=np.float32)
    degrees = np.empty(16, dtype=np.int64)
    for subset in range(16):
        degrees[subset] = int(subset).bit_count()
        for state in range(16):
            matrix[subset, state] = (
                -1.0 if (subset & state).bit_count() % 2 else 1.0
            )
    return torch.as_tensor(matrix, device=device), degrees


def kernel_alignment(kernel: torch.Tensor, target_sign: torch.Tensor) -> torch.Tensor:
    target_outer = target_sign[:, None] * target_sign[None, :]
    numerator = torch.sum(kernel * target_outer[None], dim=(1, 2))
    denominator = torch.linalg.vector_norm(kernel, dim=(1, 2)) * float(
        len(target_sign)
    )
    return numerator / denominator.clamp_min(1e-12)


@torch.no_grad()
def measure_representation(
    output_dir: Path,
    mode: str,
    threshold: float,
    state: SMCState,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> list[dict[str, Any]]:
    sample_count = min(
        Config.REPRESENTATION_SAMPLE_PER_REPLICA,
        Config.PARTICLES_PER_REPLICA,
    )
    sample_indices = torch.linspace(
        0, Config.PARTICLES_PER_REPLICA - 1,
        steps=sample_count, device=state.particles.device,
    ).round().to(torch.int64)
    sampled = state.particles[:, sample_indices]
    flat = sampled.reshape(-1, sampled.shape[-1])
    hidden, logits = forward_hidden_logits(flat, inputs)
    _, _, output_weight, _ = unpack_parameters(flat)

    target_sign = 2.0 * target - 1.0
    margins = logits * target_sign[None]
    margin_norm = torch.linalg.vector_norm(margins, dim=1).clamp_min(1e-12)
    margin_direction = margins / margin_norm[:, None]

    walsh, degrees = walsh_matrix(state.particles.device)
    logit_walsh = torch.matmul(logits, walsh.T) / 16.0
    logit_energy = logit_walsh.square()
    logit_order_energy = torch.stack([
        logit_energy[:, torch.as_tensor(
            degrees == order, device=logits.device
        )].sum(dim=1)
        for order in range(5)
    ], dim=1)
    logit_order_share = logit_order_energy / logit_order_energy.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    hidden_walsh = torch.einsum("qs,bsh->bqh", walsh, hidden) / 16.0
    weighted_hidden = hidden_walsh * output_weight[:, None, :]
    weighted_energy = weighted_hidden.square()
    hidden_order_energy = torch.stack([
        weighted_energy[:, torch.as_tensor(
            degrees == order, device=logits.device
        )].sum(dim=(1, 2))
        for order in range(5)
    ], dim=1)
    hidden_order_share = hidden_order_energy / hidden_order_energy.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    target_norm = torch.linalg.vector_norm(target_sign)
    logit_target_alignment = torch.sum(
        logits * target_sign[None], dim=1
    ) / (torch.linalg.vector_norm(logits, dim=1).clamp_min(1e-12) * target_norm)

    hidden_kernel = torch.bmm(hidden, hidden.transpose(1, 2)) / Config.WIDTH
    hidden_kernel_alignment = kernel_alignment(hidden_kernel, target_sign)

    output_kernel = (
        torch.bmm(hidden, hidden.transpose(1, 2)) + 1.0
    ) / Config.WIDTH
    derivative_feature = (
        (1.0 - hidden.square()) * output_weight[:, None, :]
        / math.sqrt(Config.INPUT_BITS)
    )
    first_kernel = torch.bmm(
        derivative_feature, derivative_feature.transpose(1, 2)
    )
    input_augmented = torch.matmul(inputs, inputs.T) + 1.0
    ntk = output_kernel + first_kernel * input_augmented[None]
    ntk_alignment = kernel_alignment(ntk, target_sign)

    shape = (Config.REPLICAS, sample_count)
    arrays = {
        "margin_mean": margins.mean(dim=1).reshape(shape),
        "margin_min": margins.min(dim=1).values.reshape(shape),
        "margin_std": margins.std(dim=1).reshape(shape),
        "logit_target_alignment": logit_target_alignment.reshape(shape),
        "hidden_kernel_alignment": hidden_kernel_alignment.reshape(shape),
        "ntk_alignment": ntk_alignment.reshape(shape),
        "logit_order4_share": logit_order_share[:, 4].reshape(shape),
        "hidden_order4_share": hidden_order_share[:, 4].reshape(shape),
    }
    direction = margin_direction.reshape(
        Config.REPLICAS, sample_count, 16
    )
    summary_rows: list[dict[str, Any]] = []
    for replica in range(Config.REPLICAS):
        row: dict[str, Any] = {
            "mode": mode,
            "threshold": threshold,
            "depth": -math.log(threshold),
            "replica": replica,
            "sample_count": sample_count,
        }
        local_direction = direction[replica]
        resultant = torch.linalg.vector_norm(
            local_direction.mean(dim=0)
        ).square().item()
        row["margin_direction_resultant_squared"] = float(resultant)
        for name, values in arrays.items():
            local = values[replica].detach().cpu().numpy()
            row[f"{name}_mean"] = float(np.mean(local))
            row[f"{name}_median"] = float(np.median(local))
            row[f"{name}_q10"] = float(np.quantile(local, 0.10))
            row[f"{name}_q90"] = float(np.quantile(local, 0.90))
        summary_rows.append(row)

    representation_dir = output_dir / "representations" / mode
    representation_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eps_{threshold:.9f}".replace(".", "p")
    write_csv(representation_dir / f"{stem}_summary.csv", summary_rows)
    np.savez_compressed(
        representation_dir / f"{stem}_samples.npz",
        threshold=np.asarray(threshold),
        lineage_ids=state.lineages[:, sample_indices].detach().cpu().numpy(),
        normalized_margins=direction.detach().cpu().numpy().astype(np.float16),
        normalized_logit_walsh=(
            logit_walsh / torch.linalg.vector_norm(
                logit_walsh, dim=1, keepdim=True
            ).clamp_min(1e-12)
        ).reshape(Config.REPLICAS, sample_count, 16).cpu().numpy().astype(np.float16),
        logit_order_share=logit_order_share.reshape(
            Config.REPLICAS, sample_count, 5
        ).cpu().numpy().astype(np.float16),
        hidden_order_share=hidden_order_share.reshape(
            Config.REPLICAS, sample_count, 5
        ).cpu().numpy().astype(np.float16),
        ntk_alignment=ntk_alignment.reshape(shape).cpu().numpy().astype(np.float32),
        hidden_kernel_alignment=hidden_kernel_alignment.reshape(
            shape
        ).cpu().numpy().astype(np.float32),
    )
    return summary_rows


def record_threshold(
    output_dir: Path,
    directory: Path,
    mode: str,
    threshold: float,
    state: SMCState,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> None:
    replica_logs = state.log_volume_fraction.detach().cpu().numpy()
    losses = state.losses.detach().cpu().numpy()
    target_bits = target.to(torch.uint8)
    flat = state.particles.reshape(-1, state.particles.shape[-1])
    hard_matches = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        _, logits = forward_hidden_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        hard_matches.append(torch.all(
            (logits >= 0).to(torch.uint8) == target_bits[None], dim=1
        ).cpu().numpy())
    hard_matches_np = np.concatenate(hard_matches).reshape(
        Config.REPLICAS, Config.PARTICLES_PER_REPLICA
    )
    row = {
        "mode": mode,
        "threshold": threshold,
        "depth": -math.log(threshold),
        "level": state.level,
        "estimated_log_volume_median": float(np.median(replica_logs)),
        "estimated_log_volume_min": float(np.min(replica_logs)),
        "estimated_log_volume_max": float(np.max(replica_logs)),
        "estimated_log10_volume_median": float(
            np.median(replica_logs) / math.log(10.0)
        ),
        "replica_log_volume_range": float(np.ptp(replica_logs)),
        "loss_min": float(np.min(losses)),
        "loss_median": float(np.median(losses)),
        "loss_max": float(np.max(losses)),
        "target_hard_probability": float(np.mean(hard_matches_np)),
        "unique_parent_lineages_min": int(min(
            np.unique(state.lineages[replica].detach().cpu().numpy()).size
            for replica in range(Config.REPLICAS)
        )),
        "unique_parent_lineages_max": int(max(
            np.unique(state.lineages[replica].detach().cpu().numpy()).size
            for replica in range(Config.REPLICAS)
        )),
    }
    state.threshold_rows.append(row)
    for replica in range(Config.REPLICAS):
        state.replica_rows.append({
            "mode": mode,
            "threshold": threshold,
            "depth": -math.log(threshold),
            "replica": replica,
            "estimated_log_volume": float(replica_logs[replica]),
            "target_hard_probability": float(hard_matches_np[replica].mean()),
            "unique_parent_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        })
    measure_representation(
        output_dir, mode, threshold, state, inputs, target
    )
    write_csv(directory / "volume_curve.csv", state.threshold_rows)
    write_csv(directory / "replica_volume_curve.csv", state.replica_rows)
    print(
        f"[{mode}] TARGET eps={threshold:.6g} | "
        f"log10V~{row['estimated_log10_volume_median']:.2f} | "
        f"hard={row['target_hard_probability']:.3%} | "
        f"replica range={row['replica_log_volume_range']:.2f}",
        flush=True,
    )


def make_generators(
    device: torch.device,
    mode: str,
) -> dict[str, torch.Generator]:
    if mode == "parent":
        seeds = {
            "prior": Config.PRIOR_SEED,
            "resample": Config.PARENT_RESAMPLE_SEED,
            "mutation": Config.PARENT_MUTATION_SEED,
        }
    elif mode == "parity4":
        seeds = {
            "resample": Config.PARITY_RESAMPLE_SEED,
            "mutation": Config.PARITY_MUTATION_SEED,
        }
    elif mode == "flip0000":
        seeds = {
            "resample": Config.FLIP_RESAMPLE_SEED,
            "mutation": Config.FLIP_MUTATION_SEED,
        }
    else:
        raise ValueError(mode)
    result = {name: torch.Generator(device=device) for name in seeds}
    for name, seed in seeds.items():
        result[name].manual_seed(seed)
    return result


def write_state_artifacts(directory: Path, state: SMCState, status: str) -> None:
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
    output_dir: Path,
    device: torch.device,
    parameter_count: int,
    blocks: Sequence[ParameterBlock],
    evaluator: Callable[[torch.Tensor], torch.Tensor],
) -> SMCState:
    directory = output_dir / "parent"
    generators = make_generators(device, "parent")
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() and Config.RESUME:
        state = load_checkpoint(
            directory, "parent", device, generators
        )
        print(
            f"[parent] resume level={state.level} "
            f"eps={state.current_threshold:.8g}", flush=True
        )
    else:
        state = initialize_parent(
            device, parameter_count, evaluator, generators["prior"]
        )

    if (
        state.threshold_rows
        and math.isclose(
            float(state.threshold_rows[-1]["threshold"]),
            Config.PARENT_THRESHOLD,
            abs_tol=1e-12,
        )
    ):
        return state

    status = "running"
    try:
        while state.level < Config.MAX_LEVELS_PARENT:
            previous = state.current_threshold
            threshold, reached = choose_next_threshold(
                state, Config.PARENT_THRESHOLD
            )
            if (
                math.isfinite(previous)
                and threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reached
            ):
                status = "stalled_threshold"
                break
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
                "survival_min": float(np.min(survival)),
                "survival_median": float(np.median(survival)),
                "survival_max": float(np.max(survival)),
                "log_volume_median": float(np.median(
                    state.log_volume_fraction.detach().cpu().numpy()
                )),
                **mutation,
            }
            state.level_rows.append(row)
            if reached or state.level % Config.LOG_EVERY_LEVELS == 0:
                print(
                    f"[parent] level={state.level} eps={threshold:.6g} "
                    f"logV~{row['log_volume_median']:.2f}", flush=True
                )
            if reached:
                replica_logs = state.log_volume_fraction.detach().cpu().numpy()
                state.threshold_rows.append({
                    "mode": "parent",
                    "threshold": Config.PARENT_THRESHOLD,
                    "level": state.level,
                    "estimated_log_volume_median": float(np.median(replica_logs)),
                    "estimated_log_volume_min": float(np.min(replica_logs)),
                    "estimated_log_volume_max": float(np.max(replica_logs)),
                })
                status = "completed"
                break
            if state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(directory, state, "parent", generators)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        save_checkpoint(directory, state, "parent", generators)
        write_state_artifacts(directory, state, status)
    if status != "completed":
        raise RuntimeError(f"共同父SMC未完成：{status}")
    return state


@torch.no_grad()
def initialize_branch_from_parent(
    parent: SMCState,
    mode: str,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> tuple[SMCState, list[dict[str, Any]]]:
    branch_losses = evaluator(parent.particles)
    particles = torch.empty_like(parent.particles)
    lineages = torch.empty_like(parent.lineages)
    log_volume = parent.log_volume_fraction.clone()
    membership_rows = []
    for replica in range(Config.REPLICAS):
        valid = torch.nonzero(
            branch_losses[replica]
            <= Config.PARENT_THRESHOLD + Config.LOSS_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if not len(valid):
            raise RuntimeError(f"{mode}在parent replica={replica}没有成员")
        probability = len(valid) / Config.PARTICLES_PER_REPLICA
        choices = torch.randint(
            len(valid), (Config.PARTICLES_PER_REPLICA,),
            generator=generator, device=parent.particles.device,
        )
        selected = valid[choices]
        particles[replica] = parent.particles[replica, selected]
        lineages[replica] = parent.lineages[replica, selected]
        log_volume[replica] += math.log(probability)
        membership_rows.append({
            "mode": mode,
            "replica": replica,
            "parent_threshold": Config.PARENT_THRESHOLD,
            "member_count": len(valid),
            "conditional_membership_probability": probability,
            "parent_log_volume": float(parent.log_volume_fraction[replica]),
            "branch_log_volume": float(log_volume[replica]),
            "unique_parent_lineages": int(torch.unique(
                parent.lineages[replica, valid]
            ).numel()),
        })
    state = SMCState(
        particles=particles,
        losses=evaluator(particles),
        lineages=lineages,
        log_volume_fraction=log_volume,
        proposal_scales=list(Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=Config.PARENT_THRESHOLD,
        threshold_index=1,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
    )
    return state, membership_rows


def run_branch(
    output_dir: Path,
    parent: SMCState,
    mode: str,
    target: torch.Tensor,
    inputs: torch.Tensor,
    blocks: Sequence[ParameterBlock],
    evaluator: Callable[[torch.Tensor], torch.Tensor],
) -> str:
    directory = output_dir / mode
    generators = make_generators(state_device(parent), mode)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() and Config.RESUME:
        state = load_checkpoint(
            directory, mode, state_device(parent), generators
        )
        print(
            f"[{mode}] resume level={state.level} "
            f"eps={state.current_threshold:.8g} index={state.threshold_index}",
            flush=True,
        )
    else:
        state, membership_rows = initialize_branch_from_parent(
            parent, mode, evaluator, generators["resample"]
        )
        write_csv(directory / "parent_membership.csv", membership_rows)
        record_threshold(
            output_dir, directory, mode, Config.PARENT_THRESHOLD,
            state, inputs, target,
        )
        save_checkpoint(directory, state, mode, generators)

    if state.threshold_index >= len(Config.TARGET_THRESHOLDS):
        write_state_artifacts(directory, state, "completed")
        return "completed"

    status = "running"
    try:
        while (
            state.threshold_index < len(Config.TARGET_THRESHOLDS)
            and state.level < Config.MAX_LEVELS_BRANCH
        ):
            requested = float(Config.TARGET_THRESHOLDS[state.threshold_index])
            previous = state.current_threshold
            threshold, reached = choose_next_threshold(state, requested)
            if (
                math.isfinite(previous)
                and threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reached
            ):
                status = "stalled_threshold"
                print(
                    f"[{mode}] threshold stalled at {previous:.9g}", flush=True
                )
                break
            survival = resample_state(
                state, threshold, evaluator, generators["resample"]
            )
            mutation = rejuvenate(
                state, blocks, threshold, evaluator, generators["mutation"]
            )
            state.level += 1
            state.current_threshold = threshold
            row = {
                "mode": mode,
                "level": state.level,
                "threshold": threshold,
                "next_requested": requested,
                "survival_min": float(np.min(survival)),
                "survival_median": float(np.median(survival)),
                "survival_max": float(np.max(survival)),
                "log_volume_median": float(np.median(
                    state.log_volume_fraction.detach().cpu().numpy()
                )),
                "loss_min": float(state.losses.min().item()),
                "loss_median": float(state.losses.median().item()),
                **mutation,
            }
            state.level_rows.append(row)
            if reached or state.level % Config.LOG_EVERY_LEVELS == 0:
                print(
                    f"[{mode}] level={state.level:>5} eps={threshold:.7g} "
                    f"log10V~{row['log_volume_median']/math.log(10):.2f}",
                    flush=True,
                )
            if reached:
                record_threshold(
                    output_dir, directory, mode, requested,
                    state, inputs, target,
                )
                state.threshold_index += 1
                write_state_artifacts(directory, state, "running")
                save_checkpoint(directory, state, mode, generators)
            elif state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(directory, state, mode, generators)
        if status == "running":
            status = (
                "completed"
                if state.threshold_index == len(Config.TARGET_THRESHOLDS)
                else "stopped_max_levels"
            )
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"[{mode}] interrupted; checkpoint saved", flush=True)
    finally:
        save_checkpoint(directory, state, mode, generators)
        write_state_artifacts(directory, state, status)
    del state
    if state_device(parent).type == "cuda":
        torch.cuda.empty_cache()
    return status


def state_device(state: SMCState) -> torch.device:
    return state.particles.device


def load_or_initialize_branch(
    output_dir: Path,
    parent: SMCState,
    mode: str,
    target: torch.Tensor,
    inputs: torch.Tensor,
    evaluator: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[SMCState, dict[str, torch.Generator]]:
    directory = output_dir / mode
    generators = make_generators(state_device(parent), mode)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() and Config.RESUME:
        state = load_checkpoint(
            directory, mode, state_device(parent), generators
        )
        print(
            f"[{mode}] lockstep resume level={state.level} "
            f"eps={state.current_threshold:.8g} index={state.threshold_index}",
            flush=True,
        )
    else:
        state, membership_rows = initialize_branch_from_parent(
            parent, mode, evaluator, generators["resample"]
        )
        write_csv(directory / "parent_membership.csv", membership_rows)
        record_threshold(
            output_dir, directory, mode, Config.PARENT_THRESHOLD,
            state, inputs, target,
        )
        save_checkpoint(directory, state, mode, generators)
    return state, generators


def advance_branch_to_index(
    output_dir: Path,
    mode: str,
    target: torch.Tensor,
    inputs: torch.Tensor,
    blocks: Sequence[ParameterBlock],
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    state: SMCState,
    generators: dict[str, torch.Generator],
    requested_index: int,
) -> str:
    """让一个分支只推进到指定共同阈值，然后把控制权交还配对调度器。"""
    directory = output_dir / mode
    if state.threshold_index > requested_index:
        return "ready"
    try:
        while (
            state.threshold_index <= requested_index
            and state.level < Config.MAX_LEVELS_BRANCH
        ):
            requested = float(Config.TARGET_THRESHOLDS[state.threshold_index])
            previous = state.current_threshold
            threshold, reached = choose_next_threshold(state, requested)
            if (
                math.isfinite(previous)
                and threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reached
            ):
                write_state_artifacts(directory, state, "stalled_threshold")
                save_checkpoint(directory, state, mode, generators)
                return "stalled_threshold"
            survival = resample_state(
                state, threshold, evaluator, generators["resample"]
            )
            mutation = rejuvenate(
                state, blocks, threshold, evaluator, generators["mutation"]
            )
            state.level += 1
            state.current_threshold = threshold
            row = {
                "mode": mode,
                "level": state.level,
                "threshold": threshold,
                "next_requested": requested,
                "survival_min": float(np.min(survival)),
                "survival_median": float(np.median(survival)),
                "survival_max": float(np.max(survival)),
                "log_volume_median": float(np.median(
                    state.log_volume_fraction.detach().cpu().numpy()
                )),
                "loss_min": float(state.losses.min().item()),
                "loss_median": float(state.losses.median().item()),
                **mutation,
            }
            state.level_rows.append(row)
            if reached or state.level % Config.LOG_EVERY_LEVELS == 0:
                print(
                    f"[{mode}] level={state.level:>5} eps={threshold:.7g} "
                    f"log10V~{row['log_volume_median']/math.log(10):.2f}",
                    flush=True,
                )
            if reached:
                record_threshold(
                    output_dir, directory, mode, requested,
                    state, inputs, target,
                )
                state.threshold_index += 1
                write_state_artifacts(directory, state, "lockstep_ready")
                save_checkpoint(directory, state, mode, generators)
            elif state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(directory, state, mode, generators)
    except KeyboardInterrupt:
        save_checkpoint(directory, state, mode, generators)
        write_state_artifacts(directory, state, "interrupted")
        return "interrupted"
    if state.threshold_index > requested_index:
        return "ready"
    write_state_artifacts(directory, state, "stopped_max_levels")
    save_checkpoint(directory, state, mode, generators)
    return "stopped_max_levels"


def run_paired_lockstep(
    output_dir: Path,
    parent: SMCState,
    parity_target: torch.Tensor,
    flip_target: torch.Tensor,
    inputs: torch.Tensor,
    blocks: Sequence[ParameterBlock],
    parity_evaluator: Callable[[torch.Tensor], torch.Tensor],
    flip_evaluator: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[str, str, dict[str, Any]]:
    parity_state, parity_generators = load_or_initialize_branch(
        output_dir, parent, "parity4", parity_target, inputs, parity_evaluator
    )
    flip_state, flip_generators = load_or_initialize_branch(
        output_dir, parent, "flip0000", flip_target, inputs, flip_evaluator
    )

    # 从第一个分支尚未共同完成的位置开始；已领先的旧checkpoint会原地等待。
    for index in range(1, len(Config.TARGET_THRESHOLDS)):
        parity_status = advance_branch_to_index(
            output_dir, "parity4", parity_target, inputs, blocks,
            parity_evaluator, parity_state, parity_generators, index,
        )
        if parity_status != "ready":
            return parity_status, "waiting", aggregate_results(output_dir)
        flip_status = advance_branch_to_index(
            output_dir, "flip0000", flip_target, inputs, blocks,
            flip_evaluator, flip_state, flip_generators, index,
        )
        if flip_status != "ready":
            return "ready", flip_status, aggregate_results(output_dir)

        diagnostics = aggregate_results(output_dir)
        ratio = diagnostics.get("last_ratio_row") or {}
        rate = diagnostics.get("last_rate_row") or {}
        print(
            f"[paired] eps={Config.TARGET_THRESHOLDS[index]:.7g} | "
            f"log(Vp/Vf)={ratio.get('log_volume_ratio_parity_over_flip_median')} | "
            f"dK={rate.get('rate_difference_flip_minus_parity_median')} | "
            f"stop={diagnostics.get('robust_stop_rule_satisfied')}",
            flush=True,
        )
        hard_exact = (
            float(Config.TARGET_THRESHOLDS[index])
            < math.log(2.0) / (2 ** Config.INPUT_BITS)
        )
        if hard_exact and diagnostics.get("robust_stop_rule_satisfied"):
            write_state_artifacts(
                output_dir / "parity4", parity_state,
                "stopped_robust_crossing",
            )
            write_state_artifacts(
                output_dir / "flip0000", flip_state,
                "stopped_robust_crossing",
            )
            return (
                "stopped_robust_crossing",
                "stopped_robust_crossing",
                diagnostics,
            )

    write_state_artifacts(output_dir / "parity4", parity_state, "completed")
    write_state_artifacts(output_dir / "flip0000", flip_state, "completed")
    return "completed", "completed", aggregate_results(output_dir)


def aggregate_results(output_dir: Path) -> dict[str, Any]:
    parity_rows = read_csv(output_dir / "parity4" / "replica_volume_curve.csv")
    flip_rows = read_csv(output_dir / "flip0000" / "replica_volume_curve.csv")
    parity_map = {
        (float(row["threshold"]), int(row["replica"])): float(
            row["estimated_log_volume"]
        ) for row in parity_rows
    }
    flip_map = {
        (float(row["threshold"]), int(row["replica"])): float(
            row["estimated_log_volume"]
        ) for row in flip_rows
    }
    thresholds = sorted({key[0] for key in parity_map} & {
        key[0] for key in flip_map
    }, reverse=True)
    ratio_rows = []
    ratio_by_threshold: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        ratios = np.asarray([
            parity_map[(threshold, replica)] - flip_map[(threshold, replica)]
            for replica in range(Config.REPLICAS)
            if (threshold, replica) in parity_map
            and (threshold, replica) in flip_map
        ])
        ratio_by_threshold[threshold] = ratios
        ratio_rows.append({
            "threshold": threshold,
            "depth": -math.log(threshold),
            "replica_count": len(ratios),
            "log_volume_ratio_parity_over_flip_median": float(np.median(ratios)),
            "log_volume_ratio_parity_over_flip_min": float(np.min(ratios)),
            "log_volume_ratio_parity_over_flip_max": float(np.max(ratios)),
            "log10_volume_ratio_parity_over_flip_median": float(
                np.median(ratios) / math.log(10.0)
            ),
            "fraction_replicas_flip_harder": float(np.mean(ratios > 0)),
            "median_order": (
                "flip_harder" if np.median(ratios) > 0 else "parity_harder"
            ),
        })
    write_csv(output_dir / "paired_volume_ratio.csv", ratio_rows)

    rate_rows = []
    for high, low in zip(thresholds[:-1], thresholds[1:]):
        delta_depth = math.log(high / low)
        parity_rates = np.asarray([
            ((-parity_map[(low, replica)]) - (-parity_map[(high, replica)]))
            / delta_depth
            for replica in range(Config.REPLICAS)
        ])
        flip_rates = np.asarray([
            ((-flip_map[(low, replica)]) - (-flip_map[(high, replica)]))
            / delta_depth
            for replica in range(Config.REPLICAS)
        ])
        differences = flip_rates - parity_rates
        rate_rows.append({
            "epsilon_high": high,
            "epsilon_low": low,
            "depth_mid": -0.5 * math.log(high * low),
            "parity_rate_median": float(np.median(parity_rates)),
            "parity_rate_min": float(np.min(parity_rates)),
            "parity_rate_max": float(np.max(parity_rates)),
            "flip_rate_median": float(np.median(flip_rates)),
            "flip_rate_min": float(np.min(flip_rates)),
            "flip_rate_max": float(np.max(flip_rates)),
            "rate_difference_flip_minus_parity_median": float(
                np.median(differences)
            ),
            "rate_difference_min": float(np.min(differences)),
            "rate_difference_max": float(np.max(differences)),
            "fraction_replicas_rate_difference_positive": float(
                np.mean(differences > 0)
            ),
        })
    write_csv(output_dir / "paired_local_rate_difference.csv", rate_rows)

    representation_rows = []
    for mode in ("parity4", "flip0000"):
        for path in sorted((output_dir / "representations" / mode).glob(
            "eps_*_summary.csv"
        )):
            representation_rows.extend(read_csv(path))
    write_csv(output_dir / "representation_replica_summary.csv", representation_rows)

    metric_names = (
        "margin_direction_resultant_squared",
        "margin_mean_median",
        "margin_min_median",
        "logit_target_alignment_median",
        "hidden_kernel_alignment_median",
        "ntk_alignment_median",
        "logit_order4_share_median",
        "hidden_order4_share_median",
    )
    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in representation_rows:
        grouped.setdefault(
            (row["mode"], float(row["threshold"])), []
        ).append(row)
    representation_aggregate = []
    for (mode, threshold), rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "mode": mode,
            "threshold": threshold,
            "depth": -math.log(threshold),
            "replica_count": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_across_replica_median"] = float(np.median(values))
            record[f"{metric}_across_replica_min"] = float(np.min(values))
            record[f"{metric}_across_replica_max"] = float(np.max(values))
        representation_aggregate.append(record)
    write_csv(
        output_dir / "representation_threshold_summary.csv",
        representation_aggregate,
    )

    required = Config.STOP_CONSECUTIVE_WINDOWS
    robust_cross_index = None
    for index in range(len(ratio_rows)):
        tail = ratio_rows[index:index + required]
        if len(tail) < required:
            continue
        ratio_ok = all(
            float(row["log_volume_ratio_parity_over_flip_min"]) > 0
            for row in tail
        ) if Config.STOP_REQUIRE_ALL_REPLICAS_CROSSED else all(
            float(row["log_volume_ratio_parity_over_flip_median"]) > 0
            for row in tail
        )
        matching_rates = [
            row for row in rate_rows
            if float(row["epsilon_low"]) in {
                float(item["threshold"]) for item in tail
            }
        ]
        rate_ok = (
            len(matching_rates) >= required - 1
            and all(float(row["rate_difference_min"]) > 0 for row in matching_rates)
        ) if Config.STOP_REQUIRE_POSITIVE_RATE_DIFF else True
        if ratio_ok and rate_ok:
            robust_cross_index = index
            break

    crossing_interpolation = None
    for previous, current in zip(ratio_rows[:-1], ratio_rows[1:]):
        high_value = float(previous[
            "log_volume_ratio_parity_over_flip_median"
        ])
        low_value = float(current[
            "log_volume_ratio_parity_over_flip_median"
        ])
        if high_value <= 0 < low_value:
            depth_high = float(previous["depth"])
            depth_low = float(current["depth"])
            fraction = -high_value / (low_value - high_value)
            crossing_interpolation = math.exp(-(
                depth_high + fraction * (depth_low - depth_high)
            ))
            break

    stopping = {
        "common_threshold_count": len(thresholds),
        "deepest_common_threshold": thresholds[-1] if thresholds else None,
        "median_crossing_observed": crossing_interpolation is not None,
        "estimated_median_crossing_epsilon": crossing_interpolation,
        "robust_stop_rule_satisfied": robust_cross_index is not None,
        "robust_stop_start_threshold": (
            float(ratio_rows[robust_cross_index]["threshold"])
            if robust_cross_index is not None else None
        ),
        "last_ratio_row": ratio_rows[-1] if ratio_rows else None,
        "last_rate_row": rate_rows[-1] if rate_rows else None,
        "interpretation": (
            "log(V_parity/V_flip)>0表示flip0000绝对体积更小、更难；"
            "rate_difference>0表示flip仍在继续追离parity。"
        ),
    }
    write_json(output_dir / "stopping_diagnostics.json", stopping)
    return stopping


def save_plots(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    ratios = read_csv(output_dir / "paired_volume_ratio.csv")
    rates = read_csv(output_dir / "paired_local_rate_difference.csv")
    representations = read_csv(output_dir / "representation_threshold_summary.csv")
    if not ratios or not rates:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    depth = np.asarray([float(row["depth"]) for row in ratios])
    median = np.asarray([
        float(row["log_volume_ratio_parity_over_flip_median"]) for row in ratios
    ])
    low = np.asarray([
        float(row["log_volume_ratio_parity_over_flip_min"]) for row in ratios
    ])
    high = np.asarray([
        float(row["log_volume_ratio_parity_over_flip_max"]) for row in ratios
    ])
    axes[0, 0].plot(depth, median, marker="o")
    axes[0, 0].fill_between(depth, low, high, alpha=0.2)
    axes[0, 0].axhline(0, color="black", ls="--")
    axes[0, 0].set_ylabel("log(V_parity4 / V_flip0000)")
    axes[0, 0].set_title("paired absolute-volume ratio")

    rate_depth = np.asarray([float(row["depth_mid"]) for row in rates])
    rate_median = np.asarray([
        float(row["rate_difference_flip_minus_parity_median"]) for row in rates
    ])
    rate_low = np.asarray([float(row["rate_difference_min"]) for row in rates])
    rate_high = np.asarray([float(row["rate_difference_max"]) for row in rates])
    axes[0, 1].plot(rate_depth, rate_median, marker="o")
    axes[0, 1].fill_between(rate_depth, rate_low, rate_high, alpha=0.2)
    axes[0, 1].axhline(0, color="black", ls="--")
    axes[0, 1].set_ylabel("kappa_flip - kappa_parity")
    axes[0, 1].set_title("paired contraction-rate difference")

    for mode, color in (("parity4", "tab:blue"), ("flip0000", "tab:orange")):
        local = sorted(
            [row for row in representations if row["mode"] == mode],
            key=lambda row: float(row["depth"]),
        )
        if not local:
            continue
        x = [float(row["depth"]) for row in local]
        axes[1, 0].plot(
            x,
            [float(row["hidden_order4_share_median_across_replica_median"])
             for row in local],
            marker="o", color=color, label=mode,
        )
        axes[1, 1].plot(
            x,
            [float(row["ntk_alignment_median_across_replica_median"])
             for row in local],
            marker="o", color=color, label=mode,
        )
    axes[1, 0].set_ylabel("output-weighted hidden order-4 share")
    axes[1, 0].set_title("hidden parity-feature spectrum")
    axes[1, 1].set_ylabel("empirical NTK target alignment")
    axes[1, 1].set_title("NTK / feature-geometry alignment")
    for axis in axes.flat:
        axis.set_xlabel("loss depth s=-log(epsilon)")
        axis.grid(alpha=0.25)
    axes[1, 0].legend()
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "joint_deep_bridge.png", dpi=180)
    plt.close(figure)


def create_archive(output_dir: Path) -> Path:
    archive_path = output_dir.parent / f"{output_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name not in {
                "checkpoint.pt", "checkpoint.tmp.pt",
            }:
                archive.write(path, path.relative_to(output_dir.parent))
    return archive_path


def main() -> None:
    apply_smoke_overrides()
    if not math.isclose(
        Config.TARGET_THRESHOLDS[0], Config.PARENT_THRESHOLD, abs_tol=1e-12
    ):
        raise ValueError("TARGET_THRESHOLDS首项必须等于PARENT_THRESHOLD")
    if tuple(sorted(set(Config.TARGET_THRESHOLDS), reverse=True)) != tuple(
        Config.TARGET_THRESHOLDS
    ):
        raise ValueError("TARGET_THRESHOLDS必须严格递减")
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    torch.set_float32_matmul_precision("highest")

    inputs_np = truth_table_inputs()
    parity_np, flip_np = target_outputs()
    inputs = torch.as_tensor(inputs_np, device=device)
    parity_target = torch.as_tensor(parity_np, device=device)
    flip_target = torch.as_tensor(flip_np, device=device)
    blocks, parameter_count = parameter_blocks()
    parent_eval = make_evaluator(
        "union", inputs, parity_target, flip_target
    )
    parity_eval = make_evaluator(
        "parity4", inputs, parity_target, flip_target
    )
    flip_eval = make_evaluator(
        "flip0000", inputs, parity_target, flip_target
    )

    print("=== Parity4 / Flip0000 Joint Deep Bridge ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=4->{Config.WIDTH}->1 tanh | params={parameter_count} | "
        f"replicas={Config.REPLICAS} | particles/replica="
        f"{Config.PARTICLES_PER_REPLICA:,}", flush=True
    )
    print(
        f"parent eps={Config.PARENT_THRESHOLD} | deepest eps="
        f"{Config.TARGET_THRESHOLDS[-1]:.8g} | thresholds="
        f"{len(Config.TARGET_THRESHOLDS)}", flush=True
    )
    print(f"结果目录：{output_dir}", flush=True)

    started = time.perf_counter()
    parent = run_parent(
        output_dir, device, parameter_count, blocks, parent_eval
    )
    parity_status = run_branch(
        output_dir, parent, "parity4", parity_target,
        inputs, blocks, parity_eval,
    )
    if parity_status == "interrupted":
        print("parity4分支中断；重新运行可续跑。", flush=True)
        return
    flip_status = run_branch(
        output_dir, parent, "flip0000", flip_target,
        inputs, blocks, flip_eval,
    )
    stopping = aggregate_results(output_dir)
    save_plots(output_dir)
    summary = {
        "status": (
            "completed"
            if parity_status == flip_status == "completed" else "partial"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "network": f"4->{Config.WIDTH}->1 tanh",
        "parameter_count": parameter_count,
        "parity_status": parity_status,
        "flip_status": flip_status,
        "stopping_diagnostics": stopping,
    }
    write_json(output_dir / "summary.json", summary)
    archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
    print("\n=== Joint bridge summary ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
