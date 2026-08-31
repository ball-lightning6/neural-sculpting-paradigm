"""MNIST十分类小CNN的全网络HMC演示。

目标：使用少于一万参数、少量训练样本和完整网络HMC，在官方10,000张MNIST
测试集上达到至少95% accuracy。

网络共4,266参数：Conv(1,4)->Conv(4,8)->Linear(392,10)。卷积核、bias与
分类头全部是Hamiltonian状态；没有冻结层、Adam初始化或只采样head。16条chain
从iid Gaussian神经参数prior开始，经固定beta路径退火。默认50k协议以
beta=50,000结束，因此对应Gaussian参数先验下的标准整训练集likelihood；脚本
也保留已经完成的8k协议，便于严格复现。

HMC梯度对全部训练样本精确求和。数据分块仅降低显存，不是stochastic HMC。
validation/test标签只用于预先固定阶段结束后的报告，不控制beta、trajectory数、
step-size target或提前停止。
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import struct
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    # 可改为"8k"复现首轮结果；默认"50k"用于检验增加数据能否缩小泛化差距。
    EXPERIMENT_PROFILE = "50k"
    PROTOCOL_VERSION = ""
    TRAIN_COUNT = 0
    VALIDATION_COUNT = 0
    CHAIN_COUNT = 16
    DATA_CHUNK_SIZE = 1_000
    PARAMETER_CHUNK_SIZE_EVAL = 8

    # 由configure_experiment_profile冻结。最终stage前600条用于适应和burn-in。
    BETA_STAGES: tuple[tuple[float, int], ...] = ()
    FINAL_STAGE_BURNIN = 600
    FINAL_SAMPLE_INTERVAL = 20
    HMC_LEAPFROG_STEPS = 8
    HMC_INITIAL_STEP_SIZE = 2e-3
    HMC_TARGET_ACCEPTANCE = 0.70
    HMC_ADAPT_RATE = 0.08
    HMC_MIN_STEP_SIZE = 1e-5
    HMC_MAX_STEP_SIZE = 0.05

    DATA_SEED = 2026083121
    PARAMETER_SEED = 2026083131
    CHECKPOINT_INTERVAL = 50
    LOG_INTERVAL = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    LOCAL_MNIST_ROOT = Path("/root/mnist_dataset")
    RESULT_DIR = Path("/root/autodl-tmp/results_mnist10_hmc_unconfigured")
    RESUME = True
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


def configure_experiment_profile() -> None:
    """冻结数据量、beta路径和输出目录，避免新旧协议相互覆盖。"""
    if Config.EXPERIMENT_PROFILE == "8k":
        Config.PROTOCOL_VERSION = "mnist10_small_cnn_full_hmc_demo_v1"
        Config.TRAIN_COUNT = 8_000
        Config.VALIDATION_COUNT = 2_000
        Config.DATA_CHUNK_SIZE = 500
        Config.BETA_STAGES = (
            (10.0, 100),
            (30.0, 100),
            (100.0, 150),
            (300.0, 200),
            (1_000.0, 250),
            (3_000.0, 300),
            (8_000.0, 1_200),
        )
        Config.RESULT_DIR = Path(
            "/root/autodl-tmp/results_mnist10_small_cnn_full_hmc_demo"
        )
        return
    if Config.EXPERIMENT_PROFILE == "50k":
        Config.PROTOCOL_VERSION = "mnist10_small_cnn_full_hmc_demo_50k_v1"
        Config.TRAIN_COUNT = 50_000
        Config.VALIDATION_COUNT = 4_000
        Config.DATA_CHUNK_SIZE = 1_000
        Config.BETA_STAGES = (
            (10.0, 50),
            (30.0, 50),
            (100.0, 100),
            (300.0, 150),
            (1_000.0, 200),
            (3_000.0, 250),
            (8_000.0, 300),
            (20_000.0, 300),
            (50_000.0, 1_200),
        )
        Config.RESULT_DIR = Path(
            "/root/autodl-tmp/results_mnist10_small_cnn_full_hmc_demo_50k"
        )
        return
    raise ValueError(
        f"未知EXPERIMENT_PROFILE：{Config.EXPERIMENT_PROFILE!r}；"
        "可选值为'8k'或'50k'。"
    )


def configure_smoke() -> None:
    Config.EXPERIMENT_PROFILE = "smoke"
    Config.PROTOCOL_VERSION = "mnist10_small_cnn_full_hmc_demo_smoke"
    Config.TRAIN_COUNT = 100
    Config.VALIDATION_COUNT = 100
    Config.CHAIN_COUNT = 2
    Config.DATA_CHUNK_SIZE = 50
    Config.PARAMETER_CHUNK_SIZE_EVAL = 2
    Config.BETA_STAGES = ((10.0, 2), (100.0, 4))
    Config.FINAL_STAGE_BURNIN = 2
    Config.FINAL_SAMPLE_INTERVAL = 1
    Config.HMC_LEAPFROG_STEPS = 1
    Config.CHECKPOINT_INTERVAL = 1
    Config.LOG_INTERVAL = 1
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_mnist10_small_cnn_full_hmc_demo"
    )
    Config.RESUME = False
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count() -> int:
    return 36+4+288+8+3920+10


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


def find_idx(root: Path, stem: str) -> Path | None:
    for candidate in (root, root/"MNIST"/"raw"):
        for suffix in ("", ".gz"):
            path = candidate/f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def load_mnist(device: torch.device) -> dict[str, torch.Tensor | str]:
    if Config.SMOKE_TEST:
        generator = torch.Generator().manual_seed(Config.DATA_SEED)
        train_total = Config.TRAIN_COUNT+Config.VALIDATION_COUNT
        return {
            "train_images": torch.randn(
                train_total, 1, 28, 28, generator=generator
            ).to(device),
            "train_labels": (torch.arange(train_total) % 10).to(device),
            "test_images": torch.randn(
                100, 1, 28, 28, generator=generator
            ).to(device),
            "test_labels": (torch.arange(100) % 10).to(device),
            "source": "synthetic_smoke",
        }
    paths = {
        "train_images": find_idx(
            Config.LOCAL_MNIST_ROOT, "train-images-idx3-ubyte"
        ),
        "train_labels": find_idx(
            Config.LOCAL_MNIST_ROOT, "train-labels-idx1-ubyte"
        ),
        "test_images": find_idx(
            Config.LOCAL_MNIST_ROOT, "t10k-images-idx3-ubyte"
        ),
        "test_labels": find_idx(
            Config.LOCAL_MNIST_ROOT, "t10k-labels-idx1-ubyte"
        ),
    }
    if any(value is None for value in paths.values()):
        raise FileNotFoundError(
            f"{Config.LOCAL_MNIST_ROOT}下未找到完整MNIST IDX。"
        )
    train_images = torch.from_numpy(read_idx(paths["train_images"])).float()
    train_labels = torch.from_numpy(read_idx(paths["train_labels"])).long()
    test_images = torch.from_numpy(read_idx(paths["test_images"])).float()
    test_labels = torch.from_numpy(read_idx(paths["test_labels"])).long()
    train_images = (train_images.unsqueeze(1)/255.0-0.1307)/0.3081
    test_images = (test_images.unsqueeze(1)/255.0-0.1307)/0.3081
    return {
        "train_images": train_images.to(device),
        "train_labels": train_labels.to(device),
        "test_images": test_images.to(device),
        "test_labels": test_labels.to(device),
        "source": str(Path(paths["train_images"]).parent),
    }


def stratified_split(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if Config.TRAIN_COUNT % 10 or Config.VALIDATION_COUNT % 10:
        raise ValueError("TRAIN_COUNT和VALIDATION_COUNT必须都能被10整除。")
    labels_np = labels.cpu().numpy()
    generator = np.random.default_rng(Config.DATA_SEED)
    train_per_class = Config.TRAIN_COUNT//10
    validation_per_class = Config.VALIDATION_COUNT//10
    train_rows = []
    validation_rows = []
    for label in range(10):
        candidates = np.where(labels_np == label)[0]
        required = train_per_class+validation_per_class
        if len(candidates) < required:
            raise ValueError(
                f"数字{label}只有{len(candidates)}个样本，协议要求{required}个。"
            )
        chosen = generator.choice(
            candidates,
            required,
            replace=False,
        )
        train_rows.append(chosen[:train_per_class])
        validation_rows.append(chosen[train_per_class:])
    train = np.stack(train_rows, axis=1).reshape(-1)
    validation = np.stack(validation_rows, axis=1).reshape(-1)
    return torch.from_numpy(train).long(), torch.from_numpy(validation).long()


def initialize_parameters(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.PARAMETER_SEED)
    return torch.randn(
        Config.CHAIN_COUNT,
        parameter_count(),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def unpack(parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
    chains = len(parameters)
    cursor = 0
    w1 = parameters[:, cursor:cursor+36].reshape(chains, 4, 1, 3, 3)/3.0
    cursor += 36
    b1 = parameters[:, cursor:cursor+4]/3.0
    cursor += 4
    w2 = parameters[:, cursor:cursor+288].reshape(chains, 8, 4, 3, 3)/6.0
    cursor += 288
    b2 = parameters[:, cursor:cursor+8]/6.0
    cursor += 8
    wf = parameters[:, cursor:cursor+3920].reshape(chains, 10, 392)/math.sqrt(392)
    cursor += 3920
    bf = parameters[:, cursor:cursor+10]/math.sqrt(392)
    cursor += 10
    if cursor != parameter_count():
        raise AssertionError("参数游标错误。")
    return w1, b1, w2, b2, wf, bf


def forward_batched(parameters: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    chains = len(parameters)
    batch = len(images)
    w1, b1, w2, b2, wf, bf = unpack(parameters)
    grouped = images.expand(-1, chains, -1, -1)
    hidden = F.conv2d(
        grouped,
        w1.reshape(chains*4, 1, 3, 3),
        b1.reshape(-1),
        padding=1,
        groups=chains,
    )
    hidden = F.max_pool2d(F.relu(hidden), 2)
    hidden = F.conv2d(
        hidden,
        w2.reshape(chains*8, 4, 3, 3),
        b2.reshape(-1),
        padding=1,
        groups=chains,
    )
    hidden = F.max_pool2d(F.relu(hidden), 2)
    features = hidden.reshape(batch, chains, 392).permute(1, 0, 2)
    return torch.bmm(features, wf.transpose(1, 2))+bf[:, None]


def loss_and_gradient(
    parameters: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = parameters.detach().requires_grad_(True)
    chains = len(local)
    total = len(images)
    losses = torch.zeros(chains, dtype=torch.float64, device=local.device)
    gradient = local.detach().clone()
    for start in range(0, total, Config.DATA_CHUNK_SIZE):
        stop = min(start+Config.DATA_CHUNK_SIZE, total)
        logits = forward_batched(local, images[start:stop])
        repeated = labels[start:stop][None].expand(chains, -1).reshape(-1)
        local_sum = F.cross_entropy(
            logits.reshape(-1, 10),
            repeated,
            reduction="none",
        ).reshape(chains, stop-start).sum(dim=1)
        losses += local_sum.detach().to(torch.float64)
        gradient += torch.autograd.grad(
            (float(beta)/total*local_sum).sum(), local
        )[0].detach()
    return (losses/total).to(torch.float32), gradient


def losses_only(
    parameters: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    chains = len(parameters)
    total = len(images)
    losses = torch.zeros(chains, dtype=torch.float64, device=parameters.device)
    with torch.no_grad():
        for start in range(0, total, Config.DATA_CHUNK_SIZE):
            stop = min(start+Config.DATA_CHUNK_SIZE, total)
            logits = forward_batched(parameters, images[start:stop])
            repeated = labels[start:stop][None].expand(chains, -1).reshape(-1)
            local = F.cross_entropy(
                logits.reshape(-1, 10), repeated, reduction="none"
            ).reshape(chains, stop-start).sum(dim=1)
            losses += local.to(torch.float64)
    return (losses/total).to(torch.float32)


def hamiltonian(
    parameters: torch.Tensor,
    momentum: torch.Tensor,
    losses: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return (
        0.5*parameters.double().square().sum(dim=1)
        + 0.5*momentum.double().square().sum(dim=1)
        + float(beta)*losses.double()
    )


def hmc_update(
    parameters: torch.Tensor,
    current_losses: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    step_size: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    momentum = torch.randn(
        parameters.shape,
        dtype=parameters.dtype,
        device=parameters.device,
        generator=generator,
    )
    initial_momentum = momentum.clone()
    proposal = parameters.clone()
    _, gradient = loss_and_gradient(proposal, images, labels, beta)
    momentum -= 0.5*step_size*gradient
    proposal_losses = current_losses
    for leapfrog_index in range(Config.HMC_LEAPFROG_STEPS):
        proposal += step_size*momentum
        proposal_losses, gradient = loss_and_gradient(
            proposal, images, labels, beta
        )
        if leapfrog_index+1 < Config.HMC_LEAPFROG_STEPS:
            momentum -= step_size*gradient
    momentum -= 0.5*step_size*gradient
    current_h = hamiltonian(
        parameters, initial_momentum, current_losses, beta
    )
    proposal_h = hamiltonian(
        proposal, momentum, proposal_losses, beta
    )
    probability = torch.exp(torch.clamp(current_h-proposal_h, max=0.0))
    accepted = torch.rand(
        len(parameters),
        dtype=torch.float64,
        device=parameters.device,
        generator=generator,
    ) < probability
    return (
        torch.where(accepted[:, None], proposal, parameters),
        torch.where(accepted, proposal_losses, current_losses),
        float(accepted.float().mean().item()),
        float(probability.mean().item()),
    )


def adapt_step_size(step_size: float, acceptance: float) -> float:
    if acceptance < 0.05:
        updated = step_size*0.5
    elif acceptance < 0.2:
        updated = step_size*0.7
    elif acceptance > 0.95:
        updated = step_size*1.2
    else:
        updated = step_size*math.exp(
            Config.HMC_ADAPT_RATE*(acceptance-Config.HMC_TARGET_ACCEPTANCE)
        )
    return float(np.clip(
        updated, Config.HMC_MIN_STEP_SIZE, Config.HMC_MAX_STEP_SIZE
    ))


@torch.no_grad()
def evaluate_distribution(
    parameters: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    sample_count = len(parameters)
    individual_correct = torch.zeros(sample_count, dtype=torch.int64)
    individual_nll = torch.zeros(sample_count, dtype=torch.float64)
    predictive_correct = 0
    predictive_nll = 0.0
    for start in range(0, len(images), Config.DATA_CHUNK_SIZE):
        stop = min(start+Config.DATA_CHUNK_SIZE, len(images))
        probabilities_sum = torch.zeros(
            stop-start, 10, dtype=torch.float64, device=images.device
        )
        for pstart in range(0, sample_count, Config.PARAMETER_CHUNK_SIZE_EVAL):
            pstop = min(
                pstart+Config.PARAMETER_CHUNK_SIZE_EVAL, sample_count
            )
            logits = forward_batched(parameters[pstart:pstop], images[start:stop])
            probabilities = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            local_labels = labels[start:stop]
            individual_correct[pstart:pstop] += (
                predictions == local_labels[None]
            ).sum(dim=1).cpu()
            individual_nll[pstart:pstop] += F.cross_entropy(
                logits.reshape(-1, 10),
                local_labels[None].expand(pstop-pstart, -1).reshape(-1),
                reduction="none",
            ).reshape(pstop-pstart, -1).sum(dim=1).double().cpu()
            probabilities_sum += probabilities.double().sum(dim=0)
        predictive = probabilities_sum/sample_count
        predictive_correct += int(
            (predictive.argmax(dim=1) == labels[start:stop]).sum().item()
        )
        predictive_nll += float(
            -torch.log(torch.clamp(
                predictive[
                    torch.arange(stop-start, device=images.device),
                    labels[start:stop],
                ],
                min=1e-300,
            )).sum().item()
        )
    accuracies = individual_correct.numpy()/len(images)
    nll = individual_nll.numpy()/len(images)
    return {
        "sample_count": sample_count,
        "example_count": len(images),
        "posterior_predictive_accuracy": predictive_correct/len(images),
        "posterior_predictive_nll": predictive_nll/len(images),
        "individual_accuracy_mean": float(np.mean(accuracies)),
        "individual_accuracy_min": float(np.min(accuracies)),
        "individual_accuracy_max": float(np.max(accuracies)),
        "individual_nll_mean": float(np.mean(nll)),
    }


def protocol_payload(
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
) -> dict[str, Any]:
    return {
        "experiment_profile": Config.EXPERIMENT_PROFILE,
        "protocol_version": Config.PROTOCOL_VERSION,
        "network": "Conv1x4 -> Conv4x8 -> Linear392x10",
        "parameter_count": parameter_count(),
        "all_parameters_sampled": True,
        "train_count": Config.TRAIN_COUNT,
        "validation_count": Config.VALIDATION_COUNT,
        "test_count": 10_000 if not Config.SMOKE_TEST else 100,
        "chain_count": Config.CHAIN_COUNT,
        "beta_stages": Config.BETA_STAGES,
        "final_beta": float(Config.BETA_STAGES[-1][0]),
        "standard_likelihood_beta": Config.TRAIN_COUNT,
        "final_beta_equals_train_count": (
            float(Config.BETA_STAGES[-1][0]) == Config.TRAIN_COUNT
        ),
        "final_stage_burnin": Config.FINAL_STAGE_BURNIN,
        "final_sample_interval": Config.FINAL_SAMPLE_INTERVAL,
        "hmc_leapfrog_steps": Config.HMC_LEAPFROG_STEPS,
        "data_chunk_size": Config.DATA_CHUNK_SIZE,
        "train_indices_sha256": hashlib.sha256(
            train_indices.numpy().tobytes()
        ).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(
            validation_indices.numpy().tobytes()
        ).hexdigest(),
        "selection_boundary": (
            "Fixed schedule. Validation/test metrics never alter HMC."
        ),
    }


def protocol_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        json_ready(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(
    path: Path,
    stage_index: int,
    trajectory_in_stage: int,
    parameters: torch.Tensor,
    losses: torch.Tensor,
    step_size: float,
    step_frozen: bool,
    samples: list[torch.Tensor],
    log_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    generator: torch.Generator,
    elapsed_seconds: float,
    protocol_sha256: str,
) -> None:
    payload = {
        "protocol_sha256": protocol_sha256,
        "stage_index": stage_index,
        "trajectory_in_stage": trajectory_in_stage,
        "parameters": parameters.cpu(),
        "losses": losses.cpu(),
        "step_size": step_size,
        "step_frozen": step_frozen,
        "samples": samples,
        "log_rows": log_rows,
        "stage_rows": stage_rows,
        "generator_state": generator.get_state().cpu(),
        "elapsed_seconds": elapsed_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    protocol_sha256: str,
) -> tuple[dict[str, Any], torch.Generator]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.pop("protocol_sha256") != protocol_sha256:
        raise RuntimeError("checkpoint协议不一致。")
    generator_state = payload.pop("generator_state")
    generator = torch.Generator(device=device)
    generator.set_state(generator_state)
    payload["parameters"] = payload["parameters"].to(device)
    payload["losses"] = payload["losses"].to(device)
    return payload, generator


def create_archive(result_dir: Path) -> Path:
    path = result_dir.parent/f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for item in sorted(result_dir.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(result_dir.parent))
    return path


def main() -> None:
    configure_experiment_profile()
    if Config.SMOKE_TEST:
        configure_smoke()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch不可见。")
    torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_mnist(device)
    all_train_images = data["train_images"]
    all_train_labels = data["train_labels"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]
    train_indices, validation_indices = stratified_split(all_train_labels)
    train_indices_device = train_indices.to(device)
    validation_indices_device = validation_indices.to(device)
    train_images = all_train_images[train_indices_device]
    train_labels = all_train_labels[train_indices_device]
    validation_images = all_train_images[validation_indices_device]
    validation_labels = all_train_labels[validation_indices_device]

    protocol = protocol_payload(train_indices, validation_indices)
    protocol_sha256 = protocol_hash(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    protocol["mnist_source"] = data["source"]
    write_json(Config.RESULT_DIR/"protocol.json", protocol)
    np.savez_compressed(
        Config.RESULT_DIR/"data_split.npz",
        train_indices=train_indices.numpy(),
        validation_indices=validation_indices.numpy(),
    )

    checkpoint_path = Config.RESULT_DIR/"checkpoint.pt"
    if Config.RESUME and checkpoint_path.exists():
        state, generator = load_checkpoint(
            checkpoint_path, device, protocol_sha256
        )
        stage_index = int(state["stage_index"])
        trajectory_in_stage = int(state["trajectory_in_stage"])
        parameters = state["parameters"]
        losses = state["losses"]
        step_size = float(state["step_size"])
        step_frozen = bool(state["step_frozen"])
        samples = state["samples"]
        log_rows = state["log_rows"]
        stage_rows = state["stage_rows"]
        prior_elapsed = float(state["elapsed_seconds"])
        print(
            f"从checkpoint继续：stage={stage_index} "
            f"trajectory={trajectory_in_stage} samples={len(samples)}",
            flush=True,
        )
    else:
        parameters = initialize_parameters(device)
        first_beta = float(Config.BETA_STAGES[0][0])
        losses = losses_only(
            parameters, train_images, train_labels
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(Config.PARAMETER_SEED+1)
        stage_index = 0
        trajectory_in_stage = 0
        step_size = Config.HMC_INITIAL_STEP_SIZE
        step_frozen = False
        samples: list[torch.Tensor] = []
        log_rows: list[dict[str, Any]] = []
        stage_rows: list[dict[str, Any]] = []
        prior_elapsed = 0.0

    print(
        "=== MNIST10 small-CNN whole-network HMC demo ===\n"
        f"profile={Config.EXPERIMENT_PROFILE} | "
        f"device={device} | gpu="
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}\n"
        f"train/val/test={len(train_images)}/{len(validation_images)}/"
        f"{len(test_images)} | chains={Config.CHAIN_COUNT} | "
        f"params={parameter_count():,} | stages={Config.BETA_STAGES}",
        flush=True,
    )

    started = time.perf_counter()
    interrupted = False
    try:
        while stage_index < len(Config.BETA_STAGES):
            beta, stage_trajectories = Config.BETA_STAGES[stage_index]
            beta = float(beta)
            stage_trajectories = int(stage_trajectories)
            if trajectory_in_stage == 0 and stage_index > 0:
                previous_beta = float(Config.BETA_STAGES[stage_index-1][0])
                step_size /= (beta/previous_beta)**0.08
            final_stage = stage_index+1 == len(Config.BETA_STAGES)
            while trajectory_in_stage < stage_trajectories:
                if (
                    final_stage
                    and trajectory_in_stage >= Config.FINAL_STAGE_BURNIN
                ):
                    step_frozen = True
                (
                    parameters,
                    losses,
                    actual_acceptance,
                    mean_probability,
                ) = hmc_update(
                    parameters,
                    losses,
                    train_images,
                    train_labels,
                    beta,
                    step_size,
                    generator,
                )
                trajectory_in_stage += 1
                if not step_frozen:
                    step_size = adapt_step_size(
                        step_size, mean_probability
                    )
                if (
                    final_stage
                    and trajectory_in_stage > Config.FINAL_STAGE_BURNIN
                    and (
                        trajectory_in_stage-Config.FINAL_STAGE_BURNIN
                    ) % Config.FINAL_SAMPLE_INTERVAL == 0
                ):
                    samples.append(parameters.detach().cpu().clone())
                if trajectory_in_stage % Config.LOG_INTERVAL == 0:
                    elapsed = prior_elapsed+time.perf_counter()-started
                    total_done = sum(
                        int(item[1]) for item in Config.BETA_STAGES[:stage_index]
                    )+trajectory_in_stage
                    total_planned = sum(
                        int(item[1]) for item in Config.BETA_STAGES
                    )
                    eta = (
                        elapsed
                        / max(total_done, 1)
                        * (total_planned-total_done)
                    )
                    row = {
                        "stage_index": stage_index,
                        "beta": beta,
                        "trajectory": trajectory_in_stage,
                        "mean_train_loss": float(losses.mean().item()),
                        "median_train_loss": float(losses.median().item()),
                        "actual_acceptance": actual_acceptance,
                        "mean_acceptance_probability": mean_probability,
                        "step_size": step_size,
                        "step_frozen": step_frozen,
                        "posterior_snapshot_count": len(samples),
                        "mean_parameter_norm": float(
                            parameters.norm(dim=1).mean().item()
                        ),
                        "elapsed_seconds": elapsed,
                    }
                    log_rows.append(row)
                    print(
                        f"beta={beta:>7.0f} traj={trajectory_in_stage:>4}/"
                        f"{stage_trajectories} | L={row['mean_train_loss']:.4f} | "
                        f"acc={actual_acceptance:.3f} | eps={step_size:.2e} | "
                        f"samples={len(samples)} | ETA={format_duration(eta)}",
                        flush=True,
                    )
                    write_csv(Config.RESULT_DIR/"hmc_log.csv", log_rows)
                if trajectory_in_stage % Config.CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(
                        checkpoint_path,
                        stage_index,
                        trajectory_in_stage,
                        parameters,
                        losses,
                        step_size,
                        step_frozen,
                        samples,
                        log_rows,
                        stage_rows,
                        generator,
                        prior_elapsed+time.perf_counter()-started,
                        protocol_sha256,
                    )

            train_metrics = evaluate_distribution(
                parameters, train_images, train_labels
            )
            validation_metrics = evaluate_distribution(
                parameters, validation_images, validation_labels
            )
            test_metrics = evaluate_distribution(
                parameters, test_images, test_labels
            )
            stage_row = {
                "stage_index": stage_index,
                "beta": beta,
                "trajectory_count": stage_trajectories,
                "mean_train_loss_state": float(losses.mean().item()),
                "step_size": step_size,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            stage_rows.append(stage_row)
            write_csv(Config.RESULT_DIR/"stage_summary.csv", stage_rows)
            print(
                f"STAGE beta={beta:.0f} | train="
                f"{train_metrics['posterior_predictive_accuracy']:.3%} | "
                f"val={validation_metrics['posterior_predictive_accuracy']:.3%} | "
                f"test={test_metrics['posterior_predictive_accuracy']:.3%}",
                flush=True,
            )
            stage_index += 1
            trajectory_in_stage = 0
            step_frozen = False
            save_checkpoint(
                checkpoint_path,
                stage_index,
                trajectory_in_stage,
                parameters,
                losses,
                step_size,
                step_frozen,
                samples,
                log_rows,
                stage_rows,
                generator,
                prior_elapsed+time.perf_counter()-started,
                protocol_sha256,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("收到中断，保留最近checkpoint。", flush=True)

    synchronize(device)
    elapsed = prior_elapsed+time.perf_counter()-started
    if samples:
        posterior_samples = torch.cat(samples, dim=0).to(device)
    else:
        posterior_samples = parameters
    final_train = evaluate_distribution(
        posterior_samples, train_images, train_labels
    )
    final_validation = evaluate_distribution(
        posterior_samples, validation_images, validation_labels
    )
    final_test = evaluate_distribution(
        posterior_samples, test_images, test_labels
    )
    torch.save(
        {
            "protocol_sha256": protocol_sha256,
            "posterior_samples": posterior_samples.cpu(),
            "final_chain_state": parameters.cpu(),
            "final_metrics": {
                "train": final_train,
                "validation": final_validation,
                "test": final_test,
            },
        },
        Config.RESULT_DIR/"posterior_samples.pt",
    )
    summary = {
        "status": "interrupted" if interrupted else "completed",
        "elapsed_seconds": elapsed,
        "stage_count_completed": len(stage_rows),
        "posterior_snapshot_count": len(samples),
        "posterior_parameter_sample_count": len(posterior_samples),
        "final_train": final_train,
        "final_validation": final_validation,
        "final_test": final_test,
        "success_test_accuracy_at_least_95pct": (
            final_test["posterior_predictive_accuracy"] >= 0.95
        ),
        "interpretation_boundary": [
            "All 4,266 network parameters are sampled by HMC.",
            "HMC starts from Gaussian prior; no Adam initialization is used.",
            "Fixed beta schedule was calibrated without adapting to test labels.",
            "This is a finite-time annealed HMC demo, not proof of exact mixing."
        ],
    }
    write_json(Config.RESULT_DIR/"summary.json", summary)
    print(
        f"FINAL test={final_test['posterior_predictive_accuracy']:.3%} | "
        f"NLL={final_test['posterior_predictive_nll']:.4f} | "
        f"samples={len(posterior_samples)} | "
        f"elapsed={format_duration(elapsed)}",
        flush=True,
    )
    if Config.PACKAGE_RESULTS:
        print(f"下载压缩包：{create_archive(Config.RESULT_DIR)}", flush=True)


if __name__ == "__main__":
    main()
