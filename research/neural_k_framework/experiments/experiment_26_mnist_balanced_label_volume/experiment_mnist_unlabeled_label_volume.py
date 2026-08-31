"""MNIST 0/1：用静态低-loss体积从无标签图像中选择平衡划分。

每个panel含5张0和5张1，只固定一张0图像的标签为0，并要求候选划分保持
5:5平衡。于是共有C(9,4)=126种候选标签。真实标签不参与体积计算。

阶段0：在独立calibration panel上训练全部候选，校准hard-fit loss范围。
阶段1：对自然划分和冻结随机候选运行Gaussian-pCN constrained SMC，确定
       可可靠测量的共同loss阈值。
阶段2：在全新evaluation panels上，对全部候选并行运行SMC并揭晓自然划分排名。

脚本自包含、支持checkpoint续跑；可整段复制到AutoDL notebook。
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import struct
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    DATA_DIR = Path("/root/mnist_dataset")
    RESULT_DIR = Path(os.environ.get(
        "NSP_MNIST_LABEL_VOLUME_RESULT_DIR",
        "/root/results_mnist_unlabeled_label_volume",
    ))
    DOWNLOAD_MNIST_IF_MISSING = False

    DIGITS = (0, 1)
    IMAGE_SIZE = 7
    PANEL_PER_CLASS = 5
    EVALUATION_PANEL_COUNT = 2
    PANEL_SEED = 2026083001
    CANDIDATE_SEED = 2026083002

    WIDTH = 32
    FIRST_BIAS_SCALE = 0.10
    OUTPUT_BIAS_SCALE = 0.10

    SGD_SEED_COUNT = 16
    SGD_INITIALIZATION_SEED = 2026083003
    SGD_LEARNING_RATE = 1e-3
    SGD_MAX_STEPS = 5_000
    SGD_EVAL_STEPS = (
        0, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 3_000, 5_000,
    )

    POTENTIAL_THRESHOLDS = (
        0.80, 0.60, 0.40, 0.30, 0.20,
        0.15, 0.10, 0.07, 0.05, 0.03,
    )
    CALIBRATION_CANDIDATE_COUNT = 8
    CALIBRATION_PARTICLES_PER_REPLICA = 1_024
    EVALUATION_PARTICLES_PER_REPLICA = 384
    SMC_REPLICAS = 4
    SURVIVAL_QUANTILE = 0.50
    MAX_SMC_LEVELS = 2_000
    MIN_LEVEL_DECREMENT = 1e-9
    LOSS_TOLERANCE = 1e-9

    ADAPT_SWEEPS = 1
    MUTATION_SWEEPS = 4
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PCN_RHOS = (0.050, 0.120, 0.020)
    MIN_PCN_RHO = 2e-4
    MAX_PCN_RHO = 0.60

    CALIBRATION_MAX_REPLICA_LOGV_RANGE = 5.0
    CALIBRATION_MIN_ACCEPTANCE = 0.005
    CALIBRATION_MIN_LINEAGES = 4

    PARTICLE_EVAL_MICRO_BATCH = 1_024
    CHECKPOINT_EVERY_LEVELS = 25
    LOG_EVERY_LEVELS = 10
    PRIOR_SEED = 2026083004
    RESAMPLE_SEED = 2026083005
    MUTATION_SEED = 2026083006

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESUME = True
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int


@dataclass
class Panel:
    panel_id: int
    role: str
    original_indices: np.ndarray
    inputs: torch.Tensor
    hidden_labels: np.ndarray
    candidate_labels: np.ndarray
    natural_candidate_id: int
    anchor_original_index: int
    panel_hash: str


@dataclass
class MultiSMCState:
    candidate_ids: list[int]
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume: torch.Tensor
    rhos: torch.Tensor
    current_threshold: float
    threshold_index: int
    level: int
    level_rows: list[dict[str, Any]]
    volume_rows: list[dict[str, Any]]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.DATA_DIR = Path(
        "research/overfitting_related_research/_smoke_mnist_dataset"
    )
    Config.RESULT_DIR = Path(".tmp_mnist_unlabeled_label_volume_smoke")
    Config.IMAGE_SIZE = 7
    Config.PANEL_PER_CLASS = 3
    Config.EVALUATION_PANEL_COUNT = 1
    Config.WIDTH = 8
    Config.SGD_SEED_COUNT = 2
    Config.SGD_MAX_STEPS = 3
    Config.SGD_EVAL_STEPS = (0, 1, 2, 3)
    Config.POTENTIAL_THRESHOLDS = (0.90, 0.80)
    Config.CALIBRATION_CANDIDATE_COUNT = 3
    Config.CALIBRATION_PARTICLES_PER_REPLICA = 32
    Config.EVALUATION_PARTICLES_PER_REPLICA = 32
    Config.SMC_REPLICAS = 2
    Config.MAX_SMC_LEVELS = 6
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 1
    Config.PARTICLE_EVAL_MICRO_BATCH = 64
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.CALIBRATION_MAX_REPLICA_LOGV_RANGE = 20.0
    Config.CALIBRATION_MIN_ACCEPTANCE = 0.0
    Config.CALIBRATION_MIN_LINEAGES = 1
    Config.DEVICE = "cpu"
    Config.RESUME = True
    Config.PACKAGE_RESULTS = True


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
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
    os.replace(temporary, path)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        json_ready(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def config_payload() -> dict[str, Any]:
    return {
        key: json_ready(value)
        for key, value in vars(Config).items()
        if key.isupper() and key not in {
            "RESULT_DIR", "DEVICE", "RESUME", "PACKAGE_RESULTS", "SMOKE_TEST"
        }
    } | {"DEVICE": Config.DEVICE}


def prepare_result_dir() -> None:
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = Config.RESULT_DIR / "config.json"
    payload = config_payload()
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("结果目录包含不同配置，请更换结果目录。")
    else:
        write_json(path, payload)


def read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        magic = handle.read(4)
        if len(magic) != 4 or magic[:2] != b"\x00\x00" or magic[2] != 0x08:
            raise RuntimeError(f"非法IDX文件：{path}")
        shape = tuple(
            struct.unpack(">I", handle.read(4))[0]
            for _ in range(magic[3])
        )
        payload = handle.read()
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(f"IDX长度错误：{path}")
    return values.reshape(shape).copy()


def raw_mnist_paths(root: Path, train: bool) -> tuple[Path, Path] | None:
    image = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    label = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
    for candidate_root in (root, root / "MNIST" / "raw"):
        for suffix in ("", ".gz"):
            image_path = candidate_root / f"{image}{suffix}"
            label_path = candidate_root / f"{label}{suffix}"
            if image_path.exists() and label_path.exists():
                return image_path, label_path
    return None


def load_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    paths = raw_mnist_paths(Config.DATA_DIR, train=True)
    if paths is not None:
        print(f"使用MNIST IDX：{paths[0].parent}", flush=True)
        return (
            torch.from_numpy(read_idx(paths[0])),
            torch.from_numpy(read_idx(paths[1])).long(),
        )
    try:
        from torchvision.datasets import MNIST
        dataset = MNIST(
            str(Config.DATA_DIR), train=True,
            download=Config.DOWNLOAD_MNIST_IF_MISSING,
        )
        return dataset.data.clone(), dataset.targets.clone()
    except Exception as exc:
        raise RuntimeError(f"无法加载MNIST：{exc!r}") from exc


def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    values = images.float().unsqueeze(1).div_(255.0)
    values = F.adaptive_avg_pool2d(
        values, (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    )
    return values.mul_(2.0).sub_(1.0).flatten(1).contiguous()


def balanced_candidates(
    hidden_labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    count = len(hidden_labels)
    zeros = count // 2
    rows = []
    for extra_zeros in itertools.combinations(range(1, count), zeros - 1):
        labels = np.ones(count, dtype=np.float32)
        labels[0] = 0.0
        labels[list(extra_zeros)] = 0.0
        rows.append(labels)
    candidates = np.stack(rows)
    matches = np.flatnonzero(np.all(
        candidates.astype(np.uint8) == hidden_labels[None], axis=1
    ))
    if len(matches) != 1:
        raise RuntimeError("自然划分未唯一出现在候选集合。")
    return candidates, int(matches[0])


def build_panels(images: torch.Tensor, labels: torch.Tensor) -> list[Panel]:
    all_x = preprocess_images(images)
    digit_orders = {}
    for offset, digit in enumerate(Config.DIGITS):
        indices = torch.nonzero(labels == digit, as_tuple=False).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.PANEL_SEED + offset)
        digit_orders[digit] = indices[
            torch.randperm(len(indices), generator=generator)
        ]
    panel_count = 1 + Config.EVALUATION_PANEL_COUNT
    panels = []
    for panel_id in range(panel_count):
        start = panel_id * Config.PANEL_PER_CLASS
        stop = start + Config.PANEL_PER_CLASS
        zero_indices = digit_orders[Config.DIGITS[0]][start:stop]
        one_indices = digit_orders[Config.DIGITS[1]][start:stop]
        anchor = int(zero_indices[0])
        remaining = torch.cat((zero_indices[1:], one_indices))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.PANEL_SEED + 100_003 * (panel_id + 1))
        remaining = remaining[
            torch.randperm(len(remaining), generator=generator)
        ]
        ordered = torch.cat((torch.tensor([anchor]), remaining)).long()
        hidden = (labels[ordered] == Config.DIGITS[1]).to(torch.uint8).numpy()
        candidates, natural_id = balanced_candidates(hidden)
        metadata = {
            "panel_id": panel_id,
            "role": "calibration" if panel_id == 0 else "evaluation",
            "original_indices": ordered.tolist(),
            "anchor_original_index": anchor,
            "candidate_count": len(candidates),
        }
        panels.append(Panel(
            panel_id=panel_id,
            role=metadata["role"],
            original_indices=ordered.numpy(),
            inputs=all_x[ordered],
            hidden_labels=hidden,
            candidate_labels=candidates,
            natural_candidate_id=natural_id,
            anchor_original_index=anchor,
            panel_hash=canonical_hash(metadata),
        ))
    blinded = [{
        "panel_id": panel.panel_id,
        "role": panel.role,
        "original_indices": panel.original_indices.tolist(),
        "anchor_position": 0,
        "anchor_original_index": panel.anchor_original_index,
        "candidate_count": len(panel.candidate_labels),
        "panel_hash": panel.panel_hash,
    } for panel in panels]
    write_json(Config.RESULT_DIR / "panels_blinded.json", blinded)
    return panels


class BatchedTinyMLP(nn.Module):
    def __init__(self, candidate_count: int, seed_count: int, input_dim: int):
        super().__init__()
        generator = torch.Generator(device="cpu")
        width = Config.WIDTH
        base = []
        for seed in range(seed_count):
            generator.manual_seed(Config.SGD_INITIALIZATION_SEED + seed)
            base.append((
                torch.randn(width, input_dim, generator=generator),
                torch.randn(width, generator=generator),
                torch.randn(1, width, generator=generator),
                torch.randn(1, generator=generator),
            ))
        self.first_weight = nn.Parameter(torch.stack([
            base[seed][0] for _ in range(candidate_count)
            for seed in range(seed_count)
        ]))
        self.first_bias = nn.Parameter(torch.stack([
            base[seed][1] for _ in range(candidate_count)
            for seed in range(seed_count)
        ]))
        self.output_weight = nn.Parameter(torch.stack([
            base[seed][2] for _ in range(candidate_count)
            for seed in range(seed_count)
        ]))
        self.output_bias = nn.Parameter(torch.stack([
            base[seed][3] for _ in range(candidate_count)
            for seed in range(seed_count)
        ]))
        self.input_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(
            torch.bmm(
                inputs,
                (self.first_weight / math.sqrt(self.input_dim)).transpose(1, 2),
            ) + Config.FIRST_BIAS_SCALE * self.first_bias[:, None]
        )
        return (
            torch.bmm(
                hidden,
                (self.output_weight / math.sqrt(Config.WIDTH)).transpose(1, 2),
            ).squeeze(-1)
            + Config.OUTPUT_BIAS_SCALE * self.output_bias
        )


def run_sgd_calibration(panel: Panel, device: torch.device) -> list[dict[str, Any]]:
    output = Config.RESULT_DIR / "sgd_calibration.csv"
    complete = Config.RESULT_DIR / "sgd_calibration_complete.json"
    if complete.exists() and output.exists():
        with output.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    candidate_count = len(panel.candidate_labels)
    seed_count = Config.SGD_SEED_COUNT
    model = BatchedTinyMLP(
        candidate_count, seed_count, panel.inputs.shape[1]
    ).to(device)
    train_x = panel.inputs.to(device)[None].expand(
        candidate_count * seed_count, -1, -1
    )
    train_y = torch.as_tensor(
        np.repeat(panel.candidate_labels, seed_count, axis=0), device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.SGD_LEARNING_RATE, weight_decay=0.0
    )
    rows: list[dict[str, Any]] = []
    checkpoints = set(Config.SGD_EVAL_STEPS)
    for step in range(Config.SGD_MAX_STEPS + 1):
        if step in checkpoints:
            with torch.no_grad():
                logits = model(train_x)
                losses = F.binary_cross_entropy_with_logits(
                    logits, train_y, reduction="none"
                ).mean(dim=1).reshape(candidate_count, seed_count)
                exact = torch.all(
                    (logits >= 0) == (train_y >= 0.5), dim=1
                ).reshape(candidate_count, seed_count)
            for candidate_id in range(candidate_count):
                local = losses[candidate_id].cpu().numpy()
                rows.append({
                    "step": step,
                    "candidate_id": candidate_id,
                    "loss_median": float(np.median(local)),
                    "loss_q10": float(np.quantile(local, 0.1)),
                    "loss_q90": float(np.quantile(local, 0.9)),
                    "hard_fit_rate": float(
                        exact[candidate_id].float().mean().item()
                    ),
                })
            write_csv(output, rows)
            local_rows = [row for row in rows if row["step"] == step]
            print(
                f"[SGD calibration] step={step:5d} "
                f"fit_min={min(r['hard_fit_rate'] for r in local_rows):.3f} "
                f"loss_median={np.median([r['loss_median'] for r in local_rows]):.3e}",
                flush=True,
            )
        if step == Config.SGD_MAX_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        per_model = F.binary_cross_entropy_with_logits(
            logits, train_y, reduction="none"
        ).mean(dim=1)
        per_model.sum().backward()
        optimizer.step()
    write_json(complete, {"complete": True, "steps": Config.SGD_MAX_STEPS})
    del optimizer, model, train_x, train_y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def parameter_blocks(input_dim: int) -> tuple[list[ParameterBlock], int]:
    cursor = 0
    first = Config.WIDTH * input_dim + Config.WIDTH
    blocks = [ParameterBlock("first_layer", cursor, cursor + first)]
    cursor += first
    output = Config.WIDTH + 1
    blocks.append(ParameterBlock("output_layer", cursor, cursor + output))
    cursor += output
    blocks.append(ParameterBlock("all_parameters", 0, cursor))
    return blocks, cursor


def forward_logits(
    particles: torch.Tensor, inputs: torch.Tensor,
) -> torch.Tensor:
    count = len(particles)
    input_dim = inputs.shape[1]
    cursor = 0
    size = Config.WIDTH * input_dim
    first_weight = particles[:, cursor:cursor + size].reshape(
        count, Config.WIDTH, input_dim
    ) / math.sqrt(input_dim)
    cursor += size
    first_bias = (
        particles[:, cursor:cursor + Config.WIDTH]
        * Config.FIRST_BIAS_SCALE
    )
    cursor += Config.WIDTH
    output_weight = particles[:, cursor:cursor + Config.WIDTH].reshape(
        count, 1, Config.WIDTH
    ) / math.sqrt(Config.WIDTH)
    cursor += Config.WIDTH
    output_bias = particles[:, cursor:cursor + 1] * Config.OUTPUT_BIAS_SCALE
    hidden = torch.tanh(
        torch.bmm(
            inputs[None].expand(count, -1, -1),
            first_weight.transpose(1, 2),
        ) + first_bias[:, None]
    )
    return (
        torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
        + output_bias
    )


@torch.no_grad()
def evaluate_losses(
    particles: torch.Tensor, inputs: torch.Tensor,
    candidate_targets: torch.Tensor,
) -> torch.Tensor:
    shape = particles.shape[:-1]
    candidate_count, replicas, particle_count = shape
    flat = particles.reshape(-1, particles.shape[-1])
    targets = candidate_targets[:, None, None].expand(
        candidate_count, replicas, particle_count, -1
    ).reshape(-1, candidate_targets.shape[1])
    pieces = []
    for start in range(0, len(flat), Config.PARTICLE_EVAL_MICRO_BATCH):
        stop = min(start + Config.PARTICLE_EVAL_MICRO_BATCH, len(flat))
        logits = forward_logits(flat[start:stop], inputs)
        pieces.append(F.binary_cross_entropy_with_logits(
            logits, targets[start:stop], reduction="none"
        ).mean(dim=1))
    return torch.cat(pieces).reshape(shape)


def make_generators(device: torch.device, offset: int) -> dict[str, torch.Generator]:
    result = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    result["prior"].manual_seed(Config.PRIOR_SEED + offset)
    result["resample"].manual_seed(Config.RESAMPLE_SEED + offset)
    result["mutation"].manual_seed(Config.MUTATION_SEED + offset)
    return result


def initialize_smc(
    panel: Panel, candidate_ids: Sequence[int], particles_per_replica: int,
    device: torch.device, generators: dict[str, torch.Generator],
) -> MultiSMCState:
    blocks, parameter_count = parameter_blocks(panel.inputs.shape[1])
    del blocks
    base = torch.randn(
        Config.SMC_REPLICAS, particles_per_replica, parameter_count,
        device=device, generator=generators["prior"],
    )
    particles = base[None].repeat(len(candidate_ids), 1, 1, 1)
    targets = torch.as_tensor(
        panel.candidate_labels[list(candidate_ids)], device=device
    )
    losses = evaluate_losses(particles, panel.inputs.to(device), targets)
    lineages = torch.arange(
        Config.SMC_REPLICAS * particles_per_replica,
        device=device, dtype=torch.int64,
    ).reshape(1, Config.SMC_REPLICAS, particles_per_replica).repeat(
        len(candidate_ids), 1, 1
    )
    return MultiSMCState(
        candidate_ids=list(map(int, candidate_ids)),
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume=torch.zeros(
            len(candidate_ids), Config.SMC_REPLICAS,
            dtype=torch.float64, device=device,
        ),
        rhos=torch.tensor(
            Config.INITIAL_PCN_RHOS, device=device
        )[None].repeat(len(candidate_ids), 1),
        current_threshold=float("inf"),
        threshold_index=0,
        level=0,
        level_rows=[],
        volume_rows=[],
    )


def choose_next_threshold(
    state: MultiSMCState, thresholds: Sequence[float]
) -> tuple[float, bool]:
    target = float(thresholds[state.threshold_index])
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=2
    )
    adaptive = float(quantiles.max().item())
    threshold = max(target, adaptive)
    if math.isfinite(state.current_threshold):
        threshold = min(threshold, state.current_threshold)
    return threshold, threshold <= target + 1e-12


@torch.no_grad()
def resample(
    state: MultiSMCState, threshold: float,
    generator: torch.Generator,
) -> np.ndarray:
    candidate_count, replicas, particle_count = state.losses.shape
    survival = np.zeros((candidate_count, replicas), dtype=np.float64)
    new_particles = torch.empty_like(state.particles)
    new_lineages = torch.empty_like(state.lineages)
    for candidate in range(candidate_count):
        for replica in range(replicas):
            survivors = torch.nonzero(
                state.losses[candidate, replica]
                <= threshold + Config.LOSS_TOLERANCE,
                as_tuple=False,
            ).flatten()
            if not len(survivors):
                raise RuntimeError(
                    f"candidate={state.candidate_ids[candidate]} replica={replica} "
                    f"在eps={threshold:.8g}无幸存粒子。"
                )
            survival[candidate, replica] = len(survivors) / particle_count
            choices = torch.randint(
                len(survivors), (particle_count,),
                device=state.particles.device, generator=generator,
            )
            selected = survivors[choices]
            new_particles[candidate, replica] = state.particles[
                candidate, replica, selected
            ]
            new_lineages[candidate, replica] = state.lineages[
                candidate, replica, selected
            ]
    state.particles = new_particles
    state.lineages = new_lineages
    state.log_volume += torch.log(torch.as_tensor(
        survival, device=state.log_volume.device
    ))
    return survival


@torch.no_grad()
def mutate_block(
    state: MultiSMCState, block: ParameterBlock, block_index: int,
    threshold: float, panel: Panel, targets: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    noise = torch.randn(
        current.shape, device=current.device,
        dtype=current.dtype, generator=generator,
    )
    rho = state.rhos[:, block_index].view(-1, 1, 1, 1)
    proposal[..., block.start:block.stop] = (
        torch.sqrt(1.0 - rho * rho) * current + rho * noise
    )
    proposal_losses = evaluate_losses(
        proposal, panel.inputs.to(proposal.device), targets
    )
    accept = proposal_losses <= threshold + Config.LOSS_TOLERANCE
    flat_accept = accept.reshape(-1)
    flat_state = state.particles.reshape(-1, state.particles.shape[-1])
    flat_proposal = proposal.reshape(-1, proposal.shape[-1])
    flat_state[flat_accept] = flat_proposal[flat_accept]
    flat_losses = state.losses.reshape(-1)
    flat_losses[flat_accept] = proposal_losses.reshape(-1)[flat_accept]
    return accept.float().mean(dim=(1, 2)).cpu().numpy()


def rejuvenate(
    state: MultiSMCState, blocks: Sequence[ParameterBlock],
    threshold: float, panel: Panel, targets: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, np.ndarray]:
    for _ in range(Config.ADAPT_SWEEPS):
        for index, block in enumerate(blocks):
            rates = mutate_block(
                state, block, index, threshold, panel, targets, generator
            )
            update = torch.as_tensor(
                np.exp(Config.ADAPT_RATE * (rates - Config.TARGET_ACCEPTANCE)),
                device=state.rhos.device,
            )
            state.rhos[:, index] = torch.clamp(
                state.rhos[:, index] * update,
                Config.MIN_PCN_RHO, Config.MAX_PCN_RHO,
            )
    accum = {block.name: [] for block in blocks}
    for _ in range(Config.MUTATION_SWEEPS):
        for index, block in enumerate(blocks):
            accum[block.name].append(mutate_block(
                state, block, index, threshold, panel, targets, generator
            ))
    return {
        name: np.mean(np.stack(values), axis=0)
        for name, values in accum.items()
    }


def checkpoint_payload(
    state: MultiSMCState, generators: dict[str, torch.Generator],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_hash": canonical_hash(protocol),
        "candidate_ids": state.candidate_ids,
        "particles": state.particles.cpu(),
        "losses": state.losses.cpu(),
        "lineages": state.lineages.cpu(),
        "log_volume": state.log_volume.cpu(),
        "rhos": state.rhos.cpu(),
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "volume_rows": state.volume_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }


def save_checkpoint(
    path: Path, state: MultiSMCState,
    generators: dict[str, torch.Generator], protocol: dict[str, Any],
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload(state, generators, protocol), temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, device: torch.device,
    generators: dict[str, torch.Generator], protocol: dict[str, Any],
) -> MultiSMCState | None:
    if not Config.RESUME or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["protocol_hash"] != canonical_hash(protocol):
        raise RuntimeError(f"checkpoint协议不一致：{path}")
    for name, generator in generators.items():
        generator.set_state(payload["generator_states"][name].to(torch.uint8))
    state = MultiSMCState(
        candidate_ids=list(map(int, payload["candidate_ids"])),
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume=payload["log_volume"].to(device),
        rhos=payload["rhos"].to(device),
        current_threshold=float(payload["current_threshold"]),
        threshold_index=int(payload["threshold_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        volume_rows=list(payload["volume_rows"]),
    )
    print(
        f"恢复SMC：{path.parent.name} level={state.level} "
        f"eps={state.current_threshold:.7g}", flush=True,
    )
    return state


def target_volume_rows(
    state: MultiSMCState, panel: Panel, threshold: float,
    acceptance: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for local_index, candidate_id in enumerate(state.candidate_ids):
        logv = state.log_volume[local_index].detach().cpu().numpy()
        lineages = [
            len(torch.unique(state.lineages[local_index, replica]))
            for replica in range(Config.SMC_REPLICAS)
        ]
        rows.append({
            "panel_id": panel.panel_id,
            "candidate_id": candidate_id,
            "threshold": threshold,
            "level": state.level,
            "log_volume_median": float(np.median(logv)),
            "log_volume_min": float(logv.min()),
            "log_volume_max": float(logv.max()),
            "replica_log_volume_range": float(logv.max() - logv.min()),
            "loss_median": float(torch.median(
                state.losses[local_index]
            ).item()),
            "unique_lineages_median": float(np.median(lineages)),
            "acceptance_min": float(min(
                values[local_index] for values in acceptance.values()
            )),
            "acceptance": {
                name: float(values[local_index])
                for name, values in acceptance.items()
            },
        })
    return rows


def run_multi_smc(
    panel: Panel, candidate_ids: Sequence[int], thresholds: Sequence[float],
    particles_per_replica: int, phase_dir: Path, device: torch.device,
    offset: int, allow_partial: bool,
) -> tuple[list[dict[str, Any]], bool]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "panel_hash": panel.panel_hash,
        "candidate_ids": list(map(int, candidate_ids)),
        "thresholds": list(map(float, thresholds)),
        "particles_per_replica": particles_per_replica,
        "replicas": Config.SMC_REPLICAS,
        "width": Config.WIDTH,
        "image_size": Config.IMAGE_SIZE,
    }
    write_json(phase_dir / "protocol.json", protocol)
    complete_path = phase_dir / "complete.json"
    volume_path = phase_dir / "volumes_unscored.csv"
    if complete_path.exists() and volume_path.exists():
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        with volume_path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle)), bool(completion.get("complete"))
    generators = make_generators(device, offset)
    checkpoint = phase_dir / "checkpoint.pt"
    state = load_checkpoint(checkpoint, device, generators, protocol)
    if state is None:
        state = initialize_smc(
            panel, candidate_ids, particles_per_replica,
            device, generators,
        )
    targets = torch.as_tensor(
        panel.candidate_labels[state.candidate_ids], device=device
    )
    blocks, _ = parameter_blocks(panel.inputs.shape[1])
    completed = False
    try:
        while state.threshold_index < len(thresholds):
            if state.level >= Config.MAX_SMC_LEVELS:
                raise RuntimeError("超过MAX_SMC_LEVELS。")
            previous = state.current_threshold
            threshold, reaches_target = choose_next_threshold(state, thresholds)
            if (
                math.isfinite(previous)
                and threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reaches_target
            ):
                raise RuntimeError(f"tau停止下降：{threshold:.9g}")
            survival = resample(state, threshold, generators["resample"])
            state.losses = evaluate_losses(
                state.particles, panel.inputs.to(device), targets
            )
            acceptance = rejuvenate(
                state, blocks, threshold, panel, targets,
                generators["mutation"],
            )
            state.current_threshold = threshold
            state.level += 1
            state.level_rows.append({
                "level": state.level,
                "threshold": threshold,
                "target_threshold": float(thresholds[state.threshold_index]),
                "survival_min": float(survival.min()),
                "survival_median": float(np.median(survival)),
                "survival_max": float(survival.max()),
                "global_replica_logv_range": float(
                    (state.log_volume.max(dim=1).values
                     - state.log_volume.min(dim=1).values).max().item()
                ),
                "acceptance_min": float(min(v.min() for v in acceptance.values())),
            })
            if reaches_target:
                target = float(thresholds[state.threshold_index])
                state.volume_rows.extend(target_volume_rows(
                    state, panel, target, acceptance
                ))
                state.threshold_index += 1
                print(
                    f"[{phase_dir.name}] TARGET eps={target:.5g} "
                    f"logV range={state.level_rows[-1]['global_replica_logv_range']:.2f}",
                    flush=True,
                )
            if state.level % Config.LOG_EVERY_LEVELS == 0 or reaches_target:
                print(
                    f"[{phase_dir.name}] level={state.level:4d} "
                    f"eps={threshold:.6g} survive={100*np.median(survival):.1f}% "
                    f"accept_min={state.level_rows[-1]['acceptance_min']:.2%}",
                    flush=True,
                )
            if state.level % Config.CHECKPOINT_EVERY_LEVELS == 0 or reaches_target:
                save_checkpoint(checkpoint, state, generators, protocol)
                write_csv(phase_dir / "smc_levels.csv", state.level_rows)
                write_csv(volume_path, state.volume_rows)
        completed = True
    except KeyboardInterrupt:
        save_checkpoint(checkpoint, state, generators, protocol)
        write_csv(phase_dir / "smc_levels.csv", state.level_rows)
        write_csv(volume_path, state.volume_rows)
        raise
    except RuntimeError as exc:
        if not allow_partial:
            raise
        print(f"[{phase_dir.name}] calibration停止：{exc}", flush=True)
    write_csv(phase_dir / "smc_levels.csv", state.level_rows)
    write_csv(volume_path, state.volume_rows)
    write_json(complete_path, {
        "complete": completed,
        "reached_threshold_count": state.threshold_index,
        "levels": state.level,
    })
    if checkpoint.exists() and completed:
        checkpoint.unlink()
    del state, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return list(csv.DictReader(volume_path.open(
        newline="", encoding="utf-8-sig"
    ))), completed


def choose_calibration_candidates(panel: Panel) -> list[int]:
    rng = np.random.default_rng(Config.CANDIDATE_SEED)
    others = np.array([
        i for i in range(len(panel.candidate_labels))
        if i != panel.natural_candidate_id
    ])
    count = min(Config.CALIBRATION_CANDIDATE_COUNT - 1, len(others))
    selected = rng.choice(others, count, replace=False).astype(int).tolist()
    return [panel.natural_candidate_id, *selected]


def freeze_thresholds(
    calibration_rows: Sequence[dict[str, Any]],
    calibration_candidate_count: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    diagnostics = []
    reliable = []
    for threshold in Config.POTENTIAL_THRESHOLDS:
        local = [
            row for row in calibration_rows
            if abs(float(row["threshold"]) - threshold) < 1e-10
        ]
        ok = (
            len(local) == calibration_candidate_count
            and max(float(r["replica_log_volume_range"]) for r in local)
            <= Config.CALIBRATION_MAX_REPLICA_LOGV_RANGE
            and min(float(r["acceptance_min"]) for r in local)
            >= Config.CALIBRATION_MIN_ACCEPTANCE
            and min(float(r["unique_lineages_median"]) for r in local)
            >= Config.CALIBRATION_MIN_LINEAGES
        ) if local else False
        diagnostics.append({
            "threshold": threshold,
            "candidate_count": len(local),
            "max_replica_log_volume_range": (
                max(float(r["replica_log_volume_range"]) for r in local)
                if local else None
            ),
            "min_acceptance": (
                min(float(r["acceptance_min"]) for r in local)
                if local else None
            ),
            "min_unique_lineages": (
                min(float(r["unique_lineages_median"]) for r in local)
                if local else None
            ),
            "reliable": ok,
        })
        if ok:
            reliable.append(float(threshold))
        elif reliable:
            break
    if not reliable:
        reached = sorted({float(r["threshold"]) for r in calibration_rows}, reverse=True)
        if not reached:
            raise RuntimeError("SMC calibration未到达任何目标阈值。")
        reliable = [reached[0]]
        diagnostics.append({
            "fallback": True,
            "reason": "没有阈值满足全部可靠性门槛，保留最浅已达阈值",
        })
    return reliable, diagnostics


def adjusted_rand(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    table = np.zeros((2, 2), dtype=np.int64)
    np.add.at(table, (labels_a.astype(int), labels_b.astype(int)), 1)
    choose2 = lambda x: x * (x - 1) / 2
    sum_cells = float(sum(choose2(int(x)) for x in table.flat))
    sum_rows = float(sum(choose2(int(x)) for x in table.sum(axis=1)))
    sum_cols = float(sum(choose2(int(x)) for x in table.sum(axis=0)))
    total = choose2(len(labels_a))
    expected = sum_rows * sum_cols / total if total else 0.0
    maximum = 0.5 * (sum_rows + sum_cols)
    return (sum_cells - expected) / (maximum - expected) if maximum != expected else 1.0


def score_panel(
    panel: Panel, unscored_rows: Sequence[dict[str, Any]], panel_dir: Path,
) -> list[dict[str, Any]]:
    unscored_path = panel_dir / "volumes_unscored.csv"
    digest = hashlib.sha256(unscored_path.read_bytes()).hexdigest()
    scored = []
    summaries = []
    for threshold in sorted({float(r["threshold"]) for r in unscored_rows}, reverse=True):
        local = [r for r in unscored_rows if abs(float(r["threshold"]) - threshold) < 1e-10]
        local.sort(key=lambda row: int(row["candidate_id"]))
        logv = np.array([float(row["log_volume_median"]) for row in local])
        weights = np.exp(logv - logv.max())
        weights /= weights.sum()
        candidate_ids = np.array([int(row["candidate_id"]) for row in local])
        natural_position = int(np.flatnonzero(
            candidate_ids == panel.natural_candidate_id
        )[0])
        rank = 1 + int(np.sum(logv > logv[natural_position] + 1e-12))
        best_position = int(np.argmax(logv))
        coassign = np.zeros((len(panel.hidden_labels), len(panel.hidden_labels)))
        expected_accuracy = 0.0
        for position, candidate_id in enumerate(candidate_ids):
            labels = panel.candidate_labels[candidate_id].astype(np.uint8)
            accuracy = float(np.mean(labels == panel.hidden_labels))
            ari = adjusted_rand(labels, panel.hidden_labels)
            expected_accuracy += weights[position] * accuracy
            coassign += weights[position] * (labels[:, None] == labels[None, :])
            row = dict(local[position])
            row.update({
                "candidate_labels": labels.astype(int).tolist(),
                "hidden_accuracy": accuracy,
                "adjusted_rand": ari,
                "is_natural": int(candidate_id == panel.natural_candidate_id),
                "normalized_volume_mass": float(weights[position]),
                "rank": 1 + int(np.sum(logv > logv[position] + 1e-12)),
            })
            scored.append(row)
        truth_coassign = (
            panel.hidden_labels[:, None] == panel.hidden_labels[None, :]
        ).astype(float)
        summaries.append({
            "panel_id": panel.panel_id,
            "threshold": threshold,
            "natural_candidate_id": panel.natural_candidate_id,
            "natural_rank": rank,
            "natural_log_volume": float(logv[natural_position]),
            "natural_volume_mass": float(weights[natural_position]),
            "gap_natural_minus_best": float(
                logv[natural_position] - logv[best_position]
            ),
            "best_candidate_id": int(candidate_ids[best_position]),
            "best_hidden_accuracy": float(np.mean(
                panel.candidate_labels[candidate_ids[best_position]]
                == panel.hidden_labels
            )),
            "expected_hidden_accuracy": expected_accuracy,
            "coassignment_mae": float(np.mean(np.abs(
                coassign - truth_coassign
            ))),
            "coassignment_matrix": coassign.tolist(),
            "unscored_sha256": digest,
        })
    write_csv(panel_dir / "volumes_scored.csv", scored)
    write_json(panel_dir / "panel_summary.json", summaries)
    write_json(panel_dir / "hidden_labels_revealed.json", {
        "hidden_labels": panel.hidden_labels.astype(int).tolist(),
        "natural_candidate_id": panel.natural_candidate_id,
        "unscored_sha256": digest,
    })
    return summaries


def final_summary(panel_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = []
    for threshold in sorted({float(r["threshold"]) for r in panel_summaries}, reverse=True):
        local = [r for r in panel_summaries if float(r["threshold"]) == threshold]
        result.append({
            "threshold": threshold,
            "panel_count": len(local),
            "natural_top1_count": sum(int(r["natural_rank"] == 1) for r in local),
            "natural_rank_median": float(np.median([
                r["natural_rank"] for r in local
            ])),
            "natural_rank_max": int(max(r["natural_rank"] for r in local)),
            "natural_volume_mass_mean": float(np.mean([
                r["natural_volume_mass"] for r in local
            ])),
            "best_hidden_accuracy_mean": float(np.mean([
                r["best_hidden_accuracy"] for r in local
            ])),
            "expected_hidden_accuracy_mean": float(np.mean([
                r["expected_hidden_accuracy"] for r in local
            ])),
            "coassignment_mae_mean": float(np.mean([
                r["coassignment_mae"] for r in local
            ])),
        })
    return {"by_threshold": result}


def package_results() -> Path | None:
    if not Config.PACKAGE_RESULTS:
        return None
    archive = Config.RESULT_DIR.parent / f"{Config.RESULT_DIR.name}_package.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(Config.RESULT_DIR.rglob("*")):
            if path.is_file() and path.name != "checkpoint.pt":
                handle.write(path, path.relative_to(Config.RESULT_DIR.parent))
        source = globals().get("__file__")
        if source:
            source_path = Path(source).resolve()
            if source_path.exists():
                handle.write(source_path, Path(Config.RESULT_DIR.name) / source_path.name)
    return archive


def main() -> None:
    apply_smoke_overrides()
    random.seed(Config.PANEL_SEED)
    np.random.seed(Config.PANEL_SEED)
    torch.manual_seed(Config.PANEL_SEED)
    torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    prepare_result_dir()
    images, labels = load_mnist()
    panels = build_panels(images, labels)
    device = torch.device(Config.DEVICE)
    print("=== MNIST unlabeled label-volume clustering ===", flush=True)
    print(
        f"device={device} panels=1+{Config.EVALUATION_PANEL_COUNT} "
        f"images/panel={2*Config.PANEL_PER_CLASS} "
        f"candidates={len(panels[0].candidate_labels)} "
        f"MLP={Config.IMAGE_SIZE**2}->{Config.WIDTH}->1 tanh",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    try:
        sgd_rows = run_sgd_calibration(panels[0], device)
        write_json(Config.RESULT_DIR / "sgd_calibration_summary.json", {
            "natural_candidate_id": panels[0].natural_candidate_id,
            "rows": sgd_rows,
        })
        calibration_ids = choose_calibration_candidates(panels[0])
        write_json(Config.RESULT_DIR / "smc_calibration_candidates.json", {
            "candidate_ids": calibration_ids,
            "natural_candidate_id": panels[0].natural_candidate_id,
        })
        calibration_rows, _ = run_multi_smc(
            panels[0], calibration_ids, Config.POTENTIAL_THRESHOLDS,
            Config.CALIBRATION_PARTICLES_PER_REPLICA,
            Config.RESULT_DIR / "smc_calibration", device,
            offset=100_003, allow_partial=True,
        )
        frozen_thresholds, diagnostics = freeze_thresholds(
            calibration_rows, len(calibration_ids)
        )
        frozen = {
            "thresholds": frozen_thresholds,
            "diagnostics": diagnostics,
            "calibration_panel_hash": panels[0].panel_hash,
            "calibration_candidate_ids": calibration_ids,
        }
        write_json(Config.RESULT_DIR / "frozen_protocol.json", frozen)
        print(f"冻结正式thresholds={frozen_thresholds}", flush=True)

        all_summaries = []
        for panel in panels[1:]:
            panel_dir = Config.RESULT_DIR / f"evaluation_panel_{panel.panel_id}"
            candidate_ids = list(range(len(panel.candidate_labels)))
            rows, complete = run_multi_smc(
                panel, candidate_ids, frozen_thresholds,
                Config.EVALUATION_PARTICLES_PER_REPLICA,
                panel_dir, device,
                offset=1_000_003 + panel.panel_id * 100_003,
                allow_partial=False,
            )
            if not complete:
                raise RuntimeError(f"panel={panel.panel_id}未完成。")
            summaries = score_panel(panel, rows, panel_dir)
            all_summaries.extend(summaries)
            deepest = min(summaries, key=lambda row: row["threshold"])
            print(
                f"[panel={panel.panel_id}] deepest={deepest['threshold']:.5g} "
                f"natural rank={deepest['natural_rank']}/"
                f"{len(panel.candidate_labels)} "
                f"best acc={deepest['best_hidden_accuracy']:.3f}",
                flush=True,
            )
        summary = final_summary(all_summaries)
        write_json(Config.RESULT_DIR / "summary.json", summary)
        write_csv(Config.RESULT_DIR / "panel_summaries.csv", all_summaries)
        archive = package_results()
        print(f"完成。结果包：{archive}", flush=True)
    except KeyboardInterrupt:
        archive = package_results()
        print(f"\n收到Ctrl+C；已保存checkpoint。部分结果包：{archive}", flush=True)


if __name__ == "__main__":
    main()
