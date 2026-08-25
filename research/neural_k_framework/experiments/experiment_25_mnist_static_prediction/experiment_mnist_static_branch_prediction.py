"""MNIST逐样本静态体积分支预测。

本脚本读取 ``experiment_mnist_loss_calibration.py`` 的冻结数据划分，固定两组
``n=4, dataset=0`` 训练集：

- 0 vs 1：四个样本已足以得到很强泛化；
- 3 vs 8：四个样本明显欠约束并出现严重validation-loss U形。

对每个训练集D，使用不含loss梯度的Gaussian-pCN constrained SMC采样：

    A_D(epsilon) = {theta: L_D(theta) <= epsilon}.

对每张未见图像x，hard候选标签把同一个parent事件严格分成两个cell：

    V_y = mu{theta in A_D(epsilon): hard(f_theta(x)) = y}.

因此 ``V_1/(V_0+V_1)`` 可直接由parent粒子的hard输出比例估计。脚本还计算
soft分支：

    P_soft(y|x,D,epsilon)
      = E[ p_theta(y|x) | theta in A_D(epsilon) ],

其 ``-log P_soft`` 是在hard-conditioned parent内加入一个Bernoulli likelihood
因子的自由能增量。预测选择剩余质量更大、自由能代价更小的标签。

测试标签不会参与SMC或阈值选择；每个截面的未评分预测先写盘并做SHA256，
随后才附加真实标签计算准确率。真实优化器只作为matched-train-loss外部对照。
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import struct
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    DATA_DIR = Path("/root/mnist_dataset")
    CALIBRATION_DIR = Path("/root/results_mnist_loss_calibration")
    RESULT_DIR = Path("/root/results_mnist_static_branch_prediction")

    # 两个任务使用相同n、相同网络和同编号训练集，只改变图像任务难度。
    CONDITIONS = (
        ("0_vs_1", 4, 0),
        ("3_vs_8", 4, 0),
    )
    # 在看到静态预测前，依据Stage 0的SGD轨迹冻结。
    TRAIN_LOSS_THRESHOLDS = (
        0.60, 0.30, 0.10, 0.03, 0.01,
        0.003, 0.00060, 0.00015, 0.00004,
    )

    REPLICAS = 6
    PARTICLES_PER_REPLICA = 4_096
    SURVIVAL_QUANTILE = 0.50
    MAX_LEVELS_PER_CONDITION = 4_000
    MIN_LEVEL_DECREMENT = 1e-9
    LOSS_TOLERANCE = 1e-9

    ADAPT_SWEEPS = 2
    MUTATION_SWEEPS = 8
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PCN_RHOS = (0.050, 0.120, 0.020)
    MIN_PCN_RHO = 2e-4
    MAX_PCN_RHO = 0.60

    PARTICLE_EVAL_MICRO_BATCH = 1_024
    PREDICTION_PARTICLE_MICRO_BATCH = 512
    PREDICTION_SAMPLE_MICRO_BATCH = 192
    CHECKPOINT_EVERY_LEVELS = 25
    LOG_EVERY_LEVELS = 10

    PRIOR_SEED = 2026082604
    RESAMPLE_SEED = 2026082605
    MUTATION_SEED = 2026082606

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
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
class ConditionSpec:
    condition_index: int
    pair_name: str
    negative_digit: int
    positive_digit: int
    n: int
    dataset_index: int
    train_original_indices: tuple[int, ...]
    validation_original_indices: tuple[int, ...]
    test_original_indices: tuple[int, ...]
    protocol_hash: str


@dataclass
class ConditionData:
    spec: ConditionSpec
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


@dataclass
class SMCState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume: torch.Tensor
    rhos: list[float]
    current_threshold: float
    threshold_index: int
    level: int
    level_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.DATA_DIR = Path(
        "research/overfitting_related_research/_smoke_mnist_dataset"
    )
    Config.CALIBRATION_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_mnist_loss_calibration"
    )
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_mnist_static_branch_prediction"
    )
    Config.CONDITIONS = (("0_vs_1", 4, 0),)
    Config.TRAIN_LOSS_THRESHOLDS = (0.90, 0.80, 0.75)
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 128
    Config.MAX_LEVELS_PER_CONDITION = 12
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 1
    Config.PARTICLE_EVAL_MICRO_BATCH = 128
    Config.PREDICTION_PARTICLE_MICRO_BATCH = 64
    Config.PREDICTION_SAMPLE_MICRO_BATCH = 16
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.DEVICE = "cpu"
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True
    Config.PACKAGE_RESULTS = False


def validate_config() -> None:
    thresholds = tuple(float(x) for x in Config.TRAIN_LOSS_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("TRAIN_LOSS_THRESHOLDS必须严格递减且不重复。")
    if not 0.0 < Config.SURVIVAL_QUANTILE < 1.0:
        raise ValueError("SURVIVAL_QUANTILE必须在(0,1)内。")
    if Config.REPLICAS < 2 or Config.PARTICLES_PER_REPLICA < 2:
        raise ValueError("SMC至少需要两个副本且每副本至少两个粒子。")
    if len(Config.INITIAL_PCN_RHOS) != 3:
        raise ValueError("当前三个proposal block需要三个pCN rho。")
    if not Config.CALIBRATION_DIR.exists():
        raise FileNotFoundError(
            f"找不到Stage 0结果目录：{Config.CALIBRATION_DIR}"
        )


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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
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
    temp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_ready(payload), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        magic = handle.read(4)
        if len(magic) != 4 or magic[:2] != b"\x00\x00" or magic[2] != 0x08:
            raise RuntimeError(f"非法或非uint8 IDX文件：{path}")
        shape = tuple(
            struct.unpack(">I", handle.read(4))[0]
            for _ in range(magic[3])
        )
        payload = handle.read()
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(f"IDX长度不符：{path}")
    return values.reshape(shape).copy()


def raw_mnist_paths(root: Path, train: bool) -> tuple[Path, Path] | None:
    image = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    label = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
    for folder in (root, root / "MNIST" / "raw"):
        for suffix in ("", ".gz"):
            image_path = folder / f"{image}{suffix}"
            label_path = folder / f"{label}{suffix}"
            if image_path.exists() and label_path.exists():
                return image_path, label_path
    return None


def load_mnist_arrays() -> tuple[torch.Tensor, ...]:
    train_paths = raw_mnist_paths(Config.DATA_DIR, True)
    test_paths = raw_mnist_paths(Config.DATA_DIR, False)
    if train_paths is None or test_paths is None:
        raise FileNotFoundError(
            f"{Config.DATA_DIR}下找不到完整MNIST IDX/gz文件。"
        )
    print(f"使用现有MNIST：{train_paths[0].parent}", flush=True)
    return (
        torch.from_numpy(read_idx(train_paths[0])),
        torch.from_numpy(read_idx(train_paths[1])).long(),
        torch.from_numpy(read_idx(test_paths[0])),
        torch.from_numpy(read_idx(test_paths[1])).long(),
    )


def preprocess(images: torch.Tensor, image_size: int) -> torch.Tensor:
    values = images.float().unsqueeze(1).div_(255.0)
    values = F.adaptive_avg_pool2d(values, (image_size, image_size))
    return values.mul_(2.0).sub_(1.0).flatten(1).contiguous()


def calibration_config() -> dict[str, Any]:
    path = Config.CALIBRATION_DIR / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_conditions() -> tuple[list[ConditionData], dict[str, Any]]:
    cfg = calibration_config()
    manifest = json.loads(
        (Config.CALIBRATION_DIR / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    image_size = int(cfg["IMAGE_SIZE"])
    width = int(cfg["WIDTH"])
    if width <= 0 or image_size <= 0:
        raise RuntimeError("Stage 0网络配置非法。")
    train_images, _, test_images, _ = load_mnist_arrays()
    train_x_all = preprocess(train_images, image_size)
    test_x_all = preprocess(test_images, image_size)
    pair_map = {item["name"]: item for item in manifest["pairs"]}
    conditions: list[ConditionData] = []

    for condition_index, (pair_name, n, dataset_index) in enumerate(
        Config.CONDITIONS
    ):
        if pair_name not in pair_map:
            raise KeyError(f"manifest中没有任务{pair_name}。")
        pair = pair_map[pair_name]
        pair_spec = pair["pair"]
        plan = pair["train_plans"][int(dataset_index)]
        half = int(n) // 2
        negative_indices = tuple(
            int(x) for x in plan["negative_original_indices"][:half]
        )
        positive_indices = tuple(
            int(x) for x in plan["positive_original_indices"][:half]
        )
        train_indices = negative_indices + positive_indices
        validation_indices = tuple(
            int(x) for x in pair["validation_original_indices"]
        )
        test_indices = tuple(int(x) for x in pair["test_original_indices"])
        protocol = {
            "pair_name": pair_name,
            "negative_digit": int(pair_spec["negative_digit"]),
            "positive_digit": int(pair_spec["positive_digit"]),
            "n": int(n),
            "dataset_index": int(dataset_index),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
            "thresholds": list(Config.TRAIN_LOSS_THRESHOLDS),
            "network": {
                "image_size": image_size,
                "width": width,
                "first_bias_scale": float(cfg["FIRST_BIAS_SCALE"]),
                "output_bias_scale": float(cfg["OUTPUT_BIAS_SCALE"]),
            },
            "prior": "independent standard Gaussian normalized coordinates",
        }
        spec = ConditionSpec(
            condition_index=condition_index,
            pair_name=str(pair_name),
            negative_digit=int(pair_spec["negative_digit"]),
            positive_digit=int(pair_spec["positive_digit"]),
            n=int(n),
            dataset_index=int(dataset_index),
            train_original_indices=train_indices,
            validation_original_indices=validation_indices,
            test_original_indices=test_indices,
            protocol_hash=canonical_hash(protocol),
        )
        train_y = torch.cat((torch.zeros(half), torch.ones(half))).float()
        validation_y = torch.cat((
            torch.zeros(len(validation_indices) // 2),
            torch.ones(len(validation_indices) // 2),
        )).float()
        test_y = torch.cat((
            torch.zeros(len(test_indices) // 2),
            torch.ones(len(test_indices) // 2),
        )).float()
        conditions.append(ConditionData(
            spec=spec,
            train_x=train_x_all[list(train_indices)],
            train_y=train_y,
            validation_x=train_x_all[list(validation_indices)],
            validation_y=validation_y,
            test_x=test_x_all[list(test_indices)],
            test_y=test_y,
        ))
    return conditions, cfg


def parameter_blocks(
    input_dim: int, width: int
) -> tuple[list[ParameterBlock], int]:
    cursor = 0
    first = width * input_dim + width
    blocks = [ParameterBlock("first_layer", cursor, cursor + first)]
    cursor += first
    output = width + 1
    blocks.append(ParameterBlock("output_layer", cursor, cursor + output))
    cursor += output
    blocks.append(ParameterBlock("all_parameters", 0, cursor))
    return blocks, cursor


def forward_logits(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    input_dim: int,
    width: int,
    first_bias_scale: float,
    output_bias_scale: float,
) -> torch.Tensor:
    count = particles.shape[0]
    cursor = 0
    size = width * input_dim
    first_weight = particles[:, cursor:cursor + size].reshape(
        count, width, input_dim
    ) * (1.0 / math.sqrt(input_dim))
    cursor += size
    first_bias = particles[:, cursor:cursor + width] * first_bias_scale
    cursor += width
    output_weight = particles[:, cursor:cursor + width].reshape(
        count, 1, width
    ) * (1.0 / math.sqrt(width))
    cursor += width
    output_bias = particles[:, cursor:cursor + 1] * output_bias_scale
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
    particles: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    network: dict[str, Any],
) -> torch.Tensor:
    shape = particles.shape[:-1]
    flat = particles.reshape(-1, particles.shape[-1])
    pieces: list[torch.Tensor] = []
    for start in range(0, len(flat), Config.PARTICLE_EVAL_MICRO_BATCH):
        local = flat[start:start + Config.PARTICLE_EVAL_MICRO_BATCH]
        logits = forward_logits(
            local, inputs,
            int(network["input_dim"]), int(network["width"]),
            float(network["first_bias_scale"]),
            float(network["output_bias_scale"]),
        )
        pieces.append(F.binary_cross_entropy_with_logits(
            logits, targets[None].expand_as(logits), reduction="none"
        ).mean(dim=1))
    return torch.cat(pieces).reshape(shape)


def make_generators(
    device: torch.device, condition_index: int
) -> dict[str, torch.Generator]:
    offset = 1_000_003 * int(condition_index)
    result = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    result["prior"].manual_seed(Config.PRIOR_SEED + offset)
    result["resample"].manual_seed(Config.RESAMPLE_SEED + offset)
    result["mutation"].manual_seed(Config.MUTATION_SEED + offset)
    return result


def initialize_state(
    condition: ConditionData,
    device: torch.device,
    parameter_count: int,
    network: dict[str, Any],
    generator: torch.Generator,
) -> SMCState:
    particles = torch.randn(
        Config.REPLICAS, Config.PARTICLES_PER_REPLICA, parameter_count,
        device=device, generator=generator,
    )
    losses = evaluate_losses(
        particles, condition.train_x.to(device), condition.train_y.to(device),
        network,
    )
    lineages = torch.arange(
        Config.REPLICAS * Config.PARTICLES_PER_REPLICA,
        device=device, dtype=torch.int64,
    ).reshape(Config.REPLICAS, Config.PARTICLES_PER_REPLICA)
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume=torch.zeros(
            Config.REPLICAS, dtype=torch.float64, device=device
        ),
        rhos=list(Config.INITIAL_PCN_RHOS),
        current_threshold=float("inf"),
        threshold_index=0,
        level=0,
        level_rows=[],
        prediction_rows=[],
        summary_rows=[],
    )


def choose_next_threshold(state: SMCState) -> tuple[float, bool]:
    target = float(Config.TRAIN_LOSS_THRESHOLDS[state.threshold_index])
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=1
    )
    adaptive = float(quantiles.max().item())
    threshold = max(target, adaptive)
    if math.isfinite(state.current_threshold):
        threshold = min(threshold, state.current_threshold)
    return threshold, threshold <= target + 1e-12


@torch.no_grad()
def resample(
    state: SMCState,
    threshold: float,
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
                f"副本{replica}在阈值{threshold:.8g}没有幸存粒子。"
            )
        survival[replica] = len(survivors) / Config.PARTICLES_PER_REPLICA
        choices = torch.randint(
            len(survivors), (Config.PARTICLES_PER_REPLICA,),
            device=state.particles.device, generator=generator,
        )
        selected = survivors[choices]
        new_particles[replica] = state.particles[replica, selected]
        new_lineages[replica] = state.lineages[replica, selected]
    state.particles = new_particles
    state.lineages = new_lineages
    state.losses = state.losses.new_empty(state.losses.shape)
    state.log_volume += torch.log(torch.from_numpy(survival).to(
        state.log_volume.device
    ))
    return survival


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    rho: float,
    threshold: float,
    condition: ConditionData,
    network: dict[str, Any],
    generator: torch.Generator,
) -> float:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    noise = torch.randn(
        current.shape, device=current.device,
        dtype=current.dtype, generator=generator,
    )
    rho = min(max(float(rho), 0.0), 0.999999)
    proposal[..., block.start:block.stop] = (
        math.sqrt(1.0 - rho * rho) * current + rho * noise
    )
    proposal_losses = evaluate_losses(
        proposal, condition.train_x.to(proposal.device),
        condition.train_y.to(proposal.device), network,
    )
    accept = proposal_losses <= threshold + Config.LOSS_TOLERANCE
    flat_accept = accept.reshape(-1)
    flat_state = state.particles.reshape(-1, state.particles.shape[-1])
    flat_proposal = proposal.reshape(-1, proposal.shape[-1])
    flat_state[flat_accept] = flat_proposal[flat_accept]
    flat_losses = state.losses.reshape(-1)
    flat_losses[flat_accept] = proposal_losses.reshape(-1)[flat_accept]
    return float(accept.float().mean().item())


def rejuvenate(
    state: SMCState,
    blocks: Sequence[ParameterBlock],
    threshold: float,
    condition: ConditionData,
    network: dict[str, Any],
    generator: torch.Generator,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for _ in range(Config.ADAPT_SWEEPS):
        for index, block in enumerate(blocks):
            rate = mutate_block(
                state, block, state.rhos[index], threshold,
                condition, network, generator,
            )
            state.rhos[index] = float(np.clip(
                state.rhos[index] * math.exp(
                    Config.ADAPT_RATE * (rate - Config.TARGET_ACCEPTANCE)
                ),
                Config.MIN_PCN_RHO, Config.MAX_PCN_RHO,
            ))
    accum = {block.name: [] for block in blocks}
    for _ in range(Config.MUTATION_SWEEPS):
        for index, block in enumerate(blocks):
            rate = mutate_block(
                state, block, state.rhos[index], threshold,
                condition, network, generator,
            )
            accum[block.name].append(rate)
    for block in blocks:
        rates[block.name] = float(np.mean(accum[block.name]))
    return rates


def restore_losses(
    state: SMCState, condition: ConditionData, network: dict[str, Any]
) -> None:
    state.losses = evaluate_losses(
        state.particles,
        condition.train_x.to(state.particles.device),
        condition.train_y.to(state.particles.device),
        network,
    )


@torch.no_grad()
def branch_probabilities(
    state: SMCState,
    inputs: torch.Tensor,
    network: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    device = state.particles.device
    inputs = inputs.to(device)
    sample_count = len(inputs)
    hard = torch.zeros(
        Config.REPLICAS, sample_count, dtype=torch.float64, device=device
    )
    soft = torch.zeros_like(hard)
    for replica in range(Config.REPLICAS):
        particles = state.particles[replica]
        for particle_start in range(
            0, len(particles), Config.PREDICTION_PARTICLE_MICRO_BATCH
        ):
            local_particles = particles[
                particle_start:
                particle_start + Config.PREDICTION_PARTICLE_MICRO_BATCH
            ]
            for sample_start in range(
                0, sample_count, Config.PREDICTION_SAMPLE_MICRO_BATCH
            ):
                sample_stop = min(
                    sample_count,
                    sample_start + Config.PREDICTION_SAMPLE_MICRO_BATCH,
                )
                logits = forward_logits(
                    local_particles, inputs[sample_start:sample_stop],
                    int(network["input_dim"]), int(network["width"]),
                    float(network["first_bias_scale"]),
                    float(network["output_bias_scale"]),
                )
                hard[replica, sample_start:sample_stop] += (
                    logits >= 0
                ).sum(dim=0, dtype=torch.float64)
                soft[replica, sample_start:sample_stop] += torch.sigmoid(
                    logits
                ).sum(dim=0, dtype=torch.float64)
    hard /= Config.PARTICLES_PER_REPLICA
    soft /= Config.PARTICLES_PER_REPLICA
    return hard.cpu().numpy(), soft.cpu().numpy()


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def score_split(
    condition: ConditionData,
    split_name: str,
    labels: np.ndarray,
    threshold: float,
    hard_by_replica: np.ndarray,
    soft_by_replica: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hard_p1 = hard_by_replica.mean(axis=0)
    soft_p1 = soft_by_replica.mean(axis=0)
    hard_prediction = hard_p1 >= 0.5
    soft_prediction = soft_p1 >= 0.5
    labels_bool = labels >= 0.5

    unscored_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for index in range(len(labels)):
        unscored = {
            "condition": condition.spec.pair_name,
            "n": condition.spec.n,
            "dataset_index": condition.spec.dataset_index,
            "split": split_name,
            "sample_index": index,
            "train_loss_threshold": threshold,
            "hard_branch_mass_y1": float(hard_p1[index]),
            "soft_branch_mass_y1": float(soft_p1[index]),
            "hard_prediction": int(hard_prediction[index]),
            "soft_prediction": int(soft_prediction[index]),
            "hard_replica_q10": float(np.quantile(
                hard_by_replica[:, index], 0.1
            )),
            "hard_replica_q90": float(np.quantile(
                hard_by_replica[:, index], 0.9
            )),
            "soft_replica_q10": float(np.quantile(
                soft_by_replica[:, index], 0.1
            )),
            "soft_replica_q90": float(np.quantile(
                soft_by_replica[:, index], 0.9
            )),
        }
        unscored_rows.append(unscored)
        scored_rows.append({
            **unscored,
            "true_binary_label": int(labels_bool[index]),
            "true_digit": (
                condition.spec.positive_digit
                if labels_bool[index] else condition.spec.negative_digit
            ),
            "hard_correct": int(hard_prediction[index] == labels_bool[index]),
            "soft_correct": int(soft_prediction[index] == labels_bool[index]),
            "hard_true_branch_mass": float(
                hard_p1[index] if labels_bool[index] else 1.0 - hard_p1[index]
            ),
            "soft_true_branch_mass": float(
                soft_p1[index] if labels_bool[index] else 1.0 - soft_p1[index]
            ),
        })

    unscored_path = output_dir / (
        f"predictions_{split_name}_eps_{threshold:.8g}_unscored.csv"
    )
    write_csv(unscored_path, unscored_rows)
    digest = hashlib.sha256(unscored_path.read_bytes()).hexdigest()
    scored_path = output_dir / (
        f"predictions_{split_name}_eps_{threshold:.8g}_scored.csv"
    )
    write_csv(scored_path, scored_rows)

    true_soft_probability = np.where(labels_bool, soft_p1, 1.0 - soft_p1)
    true_hard_probability = np.where(labels_bool, hard_p1, 1.0 - hard_p1)
    summary = {
        "pair_name": condition.spec.pair_name,
        "n": condition.spec.n,
        "dataset_index": condition.spec.dataset_index,
        "split": split_name,
        "train_loss_threshold": threshold,
        "sample_count": len(labels),
        "hard_accuracy": float(np.mean(hard_prediction == labels_bool)),
        "soft_accuracy": float(np.mean(soft_prediction == labels_bool)),
        "soft_nll": float(-np.mean(np.log(np.clip(
            true_soft_probability, 1e-12, 1.0
        )))),
        "hard_true_branch_mass_mean": float(np.mean(true_hard_probability)),
        "soft_true_branch_mass_mean": float(np.mean(true_soft_probability)),
        "hard_branch_entropy_mean_bits": float(np.mean(binary_entropy(hard_p1))),
        "soft_branch_entropy_mean_bits": float(np.mean(binary_entropy(soft_p1))),
        "hard_point_collision": float(np.mean(
            hard_p1 ** 2 + (1.0 - hard_p1) ** 2
        )),
        "hard_replica_range_mean": float(np.mean(
            hard_by_replica.max(axis=0) - hard_by_replica.min(axis=0)
        )),
        "soft_replica_range_mean": float(np.mean(
            soft_by_replica.max(axis=0) - soft_by_replica.min(axis=0)
        )),
        "unscored_prediction_sha256": digest,
    }
    return summary, scored_rows


def closest_sgd_row(
    condition: ConditionData,
    threshold: float,
    trajectory: Sequence[dict[str, str]],
) -> dict[str, Any]:
    local = [
        row for row in trajectory
        if row["pair_name"] == condition.spec.pair_name
        and int(row["n"]) == condition.spec.n
        and int(row["dataset_index"]) == condition.spec.dataset_index
    ]
    by_step: dict[int, list[dict[str, str]]] = {}
    for row in local:
        by_step.setdefault(int(row["step"]), []).append(row)
    candidates: list[dict[str, Any]] = []
    for step, rows in by_step.items():
        candidate = {
            "step": step,
            "train_loss": float(np.median([
                float(row["train_loss"]) for row in rows
            ])),
            "validation_loss": float(np.median([
                float(row["validation_loss"]) for row in rows
            ])),
            "validation_accuracy": float(np.median([
                float(row["validation_accuracy"]) for row in rows
            ])),
            "test_loss": float(np.median([
                float(row["test_loss"]) for row in rows
            ])),
            "test_accuracy": float(np.median([
                float(row["test_accuracy"]) for row in rows
            ])),
        }
        positive_train_loss = max(candidate["train_loss"], 1e-300)
        candidate["log_distance"] = abs(
            math.log(positive_train_loss) - math.log(threshold)
        )
        candidates.append(candidate)
    best = min(candidates, key=lambda row: row["log_distance"])
    return {
        "sgd_matched_step": best["step"],
        "sgd_train_loss": best["train_loss"],
        "sgd_validation_loss": best["validation_loss"],
        "sgd_validation_accuracy": best["validation_accuracy"],
        "sgd_test_loss": best["test_loss"],
        "sgd_test_accuracy": best["test_accuracy"],
        "sgd_log_loss_distance": best["log_distance"],
    }


def save_checkpoint(
    path: Path,
    state: SMCState,
    generators: dict[str, torch.Generator],
) -> None:
    payload = {
        "particles": state.particles.cpu(),
        "losses": state.losses.cpu(),
        "lineages": state.lineages.cpu(),
        "log_volume": state.log_volume.cpu(),
        "rhos": state.rhos,
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "prediction_rows": state.prediction_rows,
        "summary_rows": state.summary_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> SMCState | None:
    if not Config.RESUME or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, generator in generators.items():
        generator.set_state(
            payload["generator_states"][name].cpu().to(torch.uint8)
        )
    state = SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume=payload["log_volume"].to(device),
        rhos=[float(x) for x in payload["rhos"]],
        current_threshold=float(payload["current_threshold"]),
        threshold_index=int(payload["threshold_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        prediction_rows=list(payload["prediction_rows"]),
        summary_rows=list(payload["summary_rows"]),
    )
    print(
        f"恢复checkpoint：level={state.level} "
        f"threshold={state.current_threshold:.8g}", flush=True,
    )
    return state


def snapshot_prediction(
    state: SMCState,
    condition: ConditionData,
    threshold: float,
    network: dict[str, Any],
    condition_dir: Path,
    sgd_trajectory: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, inputs, labels in (
        ("validation", condition.validation_x, condition.validation_y),
        ("test", condition.test_x, condition.test_y),
    ):
        hard, soft = branch_probabilities(state, inputs, network)
        summary, _ = score_split(
            condition, split, labels.numpy(), threshold,
            hard, soft, condition_dir,
        )
        summary.update(closest_sgd_row(
            condition, threshold, sgd_trajectory
        ))
        summary.update({
            "log_volume_median": float(torch.median(state.log_volume).item()),
            "log_volume_min": float(state.log_volume.min().item()),
            "log_volume_max": float(state.log_volume.max().item()),
            "unique_lineages_median": float(np.median([
                len(torch.unique(state.lineages[r]))
                for r in range(Config.REPLICAS)
            ])),
        })
        rows.append(summary)
    return rows


def run_condition(
    condition: ConditionData,
    calibration_cfg: dict[str, Any],
    result_dir: Path,
    device: torch.device,
    start_time: float,
    sgd_trajectory: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    input_dim = int(calibration_cfg["IMAGE_SIZE"]) ** 2
    width = int(calibration_cfg["WIDTH"])
    network = {
        "input_dim": input_dim,
        "width": width,
        "first_bias_scale": float(calibration_cfg["FIRST_BIAS_SCALE"]),
        "output_bias_scale": float(calibration_cfg["OUTPUT_BIAS_SCALE"]),
    }
    blocks, parameter_count = parameter_blocks(input_dim, width)
    condition_dir = result_dir / condition.spec.pair_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    write_json(condition_dir / "condition.json", asdict(condition.spec))
    checkpoint_path = condition_dir / "latest_checkpoint.pt"
    generators = make_generators(device, condition.spec.condition_index)
    state = load_checkpoint(checkpoint_path, device, generators)
    if state is None:
        state = initialize_state(
            condition, device, parameter_count, network,
            generators["prior"],
        )
    else:
        restore_losses(state, condition, network)

    while state.threshold_index < len(Config.TRAIN_LOSS_THRESHOLDS):
        if state.level >= Config.MAX_LEVELS_PER_CONDITION:
            raise RuntimeError(
                f"{condition.spec.pair_name}超过最大SMC level。"
            )
        previous = state.current_threshold
        threshold, reaches_target = choose_next_threshold(state)
        if (
            math.isfinite(previous)
            and threshold >= previous - Config.MIN_LEVEL_DECREMENT
            and not reaches_target
        ):
            raise RuntimeError(
                f"tau停止下降：{threshold:.9g}，需检查混合。"
            )
        survival = resample(state, threshold, generators["resample"])
        restore_losses(state, condition, network)
        acceptance = rejuvenate(
            state, blocks, threshold, condition, network,
            generators["mutation"],
        )
        state.current_threshold = threshold
        state.level += 1
        row = {
            "pair_name": condition.spec.pair_name,
            "n": condition.spec.n,
            "dataset_index": condition.spec.dataset_index,
            "level": state.level,
            "threshold": threshold,
            "target_threshold": float(
                Config.TRAIN_LOSS_THRESHOLDS[state.threshold_index]
            ),
            "survival_min": float(np.min(survival)),
            "survival_median": float(np.median(survival)),
            "survival_max": float(np.max(survival)),
            "log_volume_median": float(torch.median(state.log_volume).item()),
            "replica_log_volume_range": float(
                (state.log_volume.max() - state.log_volume.min()).item()
            ),
            "acceptance": acceptance,
            "rhos": dict(zip([b.name for b in blocks], state.rhos)),
            "elapsed_seconds": time.time() - start_time,
        }
        state.level_rows.append(row)

        if reaches_target:
            target = float(Config.TRAIN_LOSS_THRESHOLDS[state.threshold_index])
            prediction_rows = snapshot_prediction(
                state, condition, target, network,
                condition_dir, sgd_trajectory,
            )
            state.summary_rows.extend(prediction_rows)
            print(
                f"[{condition.spec.pair_name}] TARGET eps={target:.6g} | "
                f"static test hard/soft="
                f"{100*prediction_rows[1]['hard_accuracy']:.2f}%/"
                f"{100*prediction_rows[1]['soft_accuracy']:.2f}% | "
                f"SGD~{100*prediction_rows[1]['sgd_test_accuracy']:.2f}% | "
                f"logV~{prediction_rows[1]['log_volume_median']:.2f}",
                flush=True,
            )
            state.threshold_index += 1

        if (
            state.level % Config.LOG_EVERY_LEVELS == 0
            or reaches_target
        ):
            print(
                f"[{condition.spec.pair_name}] level={state.level:5d} | "
                f"eps={threshold:.7g} | "
                f"survive={100*np.median(survival):.1f}% | "
                f"accept={','.join(f'{v:.1%}' for v in acceptance.values())} | "
                f"elapsed={time.time()-start_time:.1f}s",
                flush=True,
            )
        if (
            state.level % Config.CHECKPOINT_EVERY_LEVELS == 0
            or reaches_target
        ):
            save_checkpoint(checkpoint_path, state, generators)
            write_csv(condition_dir / "smc_levels.csv", state.level_rows)
            write_csv(
                condition_dir / "prediction_summary.csv", state.summary_rows
            )

    write_json(condition_dir / "complete.json", {
        "complete": True,
        "levels": state.level,
        "thresholds": list(Config.TRAIN_LOSS_THRESHOLDS),
        "protocol_hash": condition.spec.protocol_hash,
    })
    return state.summary_rows


def generate_plot(result_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pair_name in sorted({row["pair_name"] for row in test_rows}):
        local = sorted(
            [row for row in test_rows if row["pair_name"] == pair_name],
            key=lambda row: -float(row["train_loss_threshold"]),
        )
        x = [float(row["train_loss_threshold"]) for row in local]
        axes[0].plot(x, [row["hard_accuracy"] for row in local],
                     marker="o", label=f"{pair_name} static hard")
        axes[0].plot(x, [row["soft_accuracy"] for row in local],
                     marker="s", linestyle="--", label=f"{pair_name} static soft")
        axes[0].plot(x, [row["sgd_test_accuracy"] for row in local],
                     marker="x", linestyle=":", label=f"{pair_name} SGD matched")
        axes[1].plot(x, [row["hard_point_collision"] for row in local],
                     marker="o", label=f"{pair_name} hard collision")
        axes[1].plot(x, [1.0-row["soft_branch_entropy_mean_bits"] for row in local],
                     marker="s", linestyle="--", label=f"{pair_name} 1-Hsoft")
    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_xlabel("training raw BCE threshold (deeper to the right)")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Static branch prediction vs matched SGD")
    axes[1].set_ylabel("concentration")
    axes[1].set_title("Prediction concentration")
    figure.tight_layout()
    figure.savefig(result_dir / "mnist_static_branch_prediction.png", dpi=180)
    plt.close(figure)


def package_results(result_dir: Path) -> Path:
    archive = result_dir.parent / f"{result_dir.name}_package.zip"
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.name != "latest_checkpoint.pt":
                handle.write(path, path.relative_to(result_dir.parent))
    temp.replace(archive)
    return archive


def prepare_result_dir() -> Path:
    if Config.RESULT_DIR.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(Config.RESULT_DIR)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return Config.RESULT_DIR


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前不可用。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    result_dir = prepare_result_dir()
    conditions, calibration_cfg = load_conditions()
    protocol = {
        "created_before_static_prediction": True,
        "conditions": [asdict(item.spec) for item in conditions],
        "thresholds": list(Config.TRAIN_LOSS_THRESHOLDS),
        "smc": {
            "replicas": Config.REPLICAS,
            "particles_per_replica": Config.PARTICLES_PER_REPLICA,
            "proposal": "prior-preserving Gaussian pCN without loss gradients",
        },
    }
    write_json(result_dir / "frozen_protocol.json", {
        **protocol, "sha256": canonical_hash(protocol),
    })
    write_json(result_dir / "config.json", {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    })
    sgd_trajectory = read_csv(
        Config.CALIBRATION_DIR / "trajectory_models.csv"
    )
    start_time = time.time()
    print("=== MNIST static branch prediction ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"conditions={Config.CONDITIONS} | thresholds={Config.TRAIN_LOSS_THRESHOLDS}",
        flush=True,
    )
    print(
        f"SMC={Config.REPLICAS}x{Config.PARTICLES_PER_REPLICA:,} | "
        f"result_dir={result_dir}", flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    interrupted = False
    try:
        for condition in conditions:
            complete_path = result_dir / condition.spec.pair_name / "complete.json"
            if Config.RESUME and complete_path.exists():
                rows = read_csv(
                    complete_path.parent / "prediction_summary.csv"
                )
                all_rows.extend(dict(row) for row in rows)
                print(f"{condition.spec.pair_name}已完成，跳过。", flush=True)
                continue
            all_rows.extend(run_condition(
                condition, calibration_cfg, result_dir,
                device, start_time, sgd_trajectory,
            ))
    except KeyboardInterrupt:
        interrupted = True
        print("收到中断；当前condition最近checkpoint已保留。", flush=True)

    # 从各condition目录重新读取，确保resume和本轮结果使用同一口径。
    all_rows = []
    for condition in conditions:
        path = result_dir / condition.spec.pair_name / "prediction_summary.csv"
        if path.exists():
            all_rows.extend(dict(row) for row in read_csv(path))
    write_csv(result_dir / "prediction_summary_all.csv", all_rows)
    if all_rows:
        # 绘图前恢复数值类型。
        numeric_rows: list[dict[str, Any]] = []
        for row in all_rows:
            converted: dict[str, Any] = dict(row)
            for key, value in row.items():
                if key in {"pair_name", "split", "unscored_prediction_sha256"}:
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    pass
            numeric_rows.append(converted)
        generate_plot(result_dir, numeric_rows)
    write_json(result_dir / "runtime.json", {
        "status": "interrupted" if interrupted else "completed",
        "elapsed_seconds": time.time() - start_time,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    })
    if Config.PACKAGE_RESULTS:
        archive = package_results(result_dir)
        print(f"下载压缩包：{archive}", flush=True)
    if interrupted:
        print("保持RESUME=True重新运行即可继续。", flush=True)
    else:
        print("静态分支预测完成。", flush=True)


if __name__ == "__main__":
    main()
