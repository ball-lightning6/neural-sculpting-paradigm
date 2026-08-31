"""MNIST十分类小CNN的同协议Adam真实训练基线。

目的不是调出MNIST最高分，而是在与全网络HMC实验相同的架构、数据切分和初始
函数分布下，测量真实optimizer分布。脚本运行两个预先固定的目标：

1. plain Adam：只最小化mini-batch mean cross-entropy；
2. MAP Adam：最小化mean CE + Gaussian参考测度对应的prior_energy / n。

网络以常规实际权重坐标训练，但初始化由HMC的raw theta~N(0,I)经同一层级缩放
得到。因此初始函数分布与HMC完全一致，optimizer则是标准实际权重坐标中的Adam。
所有seed共享预先固定的mini-batch顺序，除此之外仅初始参数不同。
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
    PROTOCOL_VERSION = "mnist10_small_cnn_adam_same_protocol_v1"
    PROFILES = (
        {
            "name": "50k",
            "train_count": 50_000,
            "validation_count": 4_000,
            "split_path": Path(
                "/root/autodl-tmp/"
                "results_mnist10_small_cnn_full_hmc_demo_50k/data_split.npz"
            ),
        },
        {
            "name": "8k",
            "train_count": 8_000,
            "validation_count": 2_000,
            "split_path": Path(
                "/root/autodl-tmp/"
                "results_mnist10_small_cnn_full_hmc_demo/data_split.npz"
            ),
        },
    )
    OBJECTIVES = ("plain", "map_same_prior")
    SEED_COUNT = 32
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EVALUATION_EPOCHS = (1, 5, 10, 20, 30, 40, 50)
    DATA_CHUNK_SIZE = 500
    PARAMETER_CHUNK_SIZE = 16

    DATA_SEED = 2026083121
    PARAMETER_SEED = 2026083131
    BATCH_ORDER_SEED = 2026083141
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    LOCAL_MNIST_ROOT = Path("/root/mnist_dataset")
    RESULT_DIR = Path(
        "/root/autodl-tmp/results_mnist10_small_cnn_adam_same_protocol"
    )
    RESUME = True
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


def configure_smoke() -> None:
    Config.PROTOCOL_VERSION = "mnist10_small_cnn_adam_same_protocol_smoke"
    Config.PROFILES = ({
        "name": "smoke",
        "train_count": 100,
        "validation_count": 100,
        "split_path": None,
    },)
    Config.OBJECTIVES = ("plain", "map_same_prior")
    Config.SEED_COUNT = 2
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 50
    Config.EVALUATION_EPOCHS = (1, 2)
    Config.DATA_CHUNK_SIZE = 50
    Config.PARAMETER_CHUNK_SIZE = 2
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_mnist10_small_cnn_adam_same_protocol"
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
        total = 200
        return {
            "train_images": torch.randn(
                total, 1, 28, 28, generator=generator
            ).to(device),
            "train_labels": (torch.arange(total) % 10).to(device),
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
    if any(path is None for path in paths.values()):
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


def load_split(profile: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if Config.SMOKE_TEST:
        return torch.arange(100), torch.arange(100, 200)
    path = profile["split_path"]
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"找不到HMC冻结切分：{path}")
    with np.load(path) as split:
        train = split["train_indices"].copy()
        validation = split["validation_indices"].copy()
    if len(train) != profile["train_count"]:
        raise RuntimeError("train split数量与profile不一致。")
    if len(validation) != profile["validation_count"]:
        raise RuntimeError("validation split数量与profile不一致。")
    return torch.from_numpy(train).long(), torch.from_numpy(validation).long()


def prior_scales(device: torch.device) -> torch.Tensor:
    values = torch.empty(parameter_count(), dtype=torch.float32, device=device)
    cursor = 0
    for count, scale in (
        (36, 3.0),
        (4, 3.0),
        (288, 6.0),
        (8, 6.0),
        (3920, math.sqrt(392)),
        (10, math.sqrt(392)),
    ):
        values[cursor:cursor+count] = scale
        cursor += count
    if cursor != parameter_count():
        raise AssertionError("prior scale游标错误。")
    return values


def initialize_actual_parameters(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.PARAMETER_SEED)
    raw = torch.randn(
        Config.SEED_COUNT,
        parameter_count(),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return raw/prior_scales(device)[None]


def unpack_actual(parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
    seeds = len(parameters)
    cursor = 0
    w1 = parameters[:, cursor:cursor+36].reshape(seeds, 4, 1, 3, 3)
    cursor += 36
    b1 = parameters[:, cursor:cursor+4]
    cursor += 4
    w2 = parameters[:, cursor:cursor+288].reshape(seeds, 8, 4, 3, 3)
    cursor += 288
    b2 = parameters[:, cursor:cursor+8]
    cursor += 8
    wf = parameters[:, cursor:cursor+3920].reshape(seeds, 10, 392)
    cursor += 3920
    bf = parameters[:, cursor:cursor+10]
    cursor += 10
    if cursor != parameter_count():
        raise AssertionError("参数游标错误。")
    return w1, b1, w2, b2, wf, bf


def forward_batched(parameters: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    seeds = len(parameters)
    batch = len(images)
    w1, b1, w2, b2, wf, bf = unpack_actual(parameters)
    grouped = images.expand(-1, seeds, -1, -1)
    hidden = F.conv2d(
        grouped,
        w1.reshape(seeds*4, 1, 3, 3),
        b1.reshape(-1),
        padding=1,
        groups=seeds,
    )
    hidden = F.max_pool2d(F.relu(hidden), 2)
    hidden = F.conv2d(
        hidden,
        w2.reshape(seeds*8, 4, 3, 3),
        b2.reshape(-1),
        padding=1,
        groups=seeds,
    )
    hidden = F.max_pool2d(F.relu(hidden), 2)
    features = hidden.reshape(batch, seeds, 392).permute(1, 0, 2)
    return torch.bmm(features, wf.transpose(1, 2))+bf[:, None]


@torch.inference_mode()
def evaluate_distribution(
    parameters: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    sample_count = len(parameters)
    individual_correct = torch.zeros(sample_count, dtype=torch.int64)
    individual_nll = torch.zeros(sample_count, dtype=torch.float64)
    predictive_correct = 0
    predictive_nll = 0.0
    hard_rows = [] if return_predictions else None
    for start in range(0, len(images), Config.DATA_CHUNK_SIZE):
        stop = min(start+Config.DATA_CHUNK_SIZE, len(images))
        probabilities_sum = torch.zeros(
            stop-start, 10, dtype=torch.float64, device=images.device
        )
        hard_chunks = [] if return_predictions else None
        for pstart in range(0, sample_count, Config.PARAMETER_CHUNK_SIZE):
            pstop = min(
                pstart+Config.PARAMETER_CHUNK_SIZE, sample_count
            )
            logits = forward_batched(
                parameters[pstart:pstop], images[start:stop]
            )
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
            if hard_chunks is not None:
                hard_chunks.append(predictions.byte().cpu())
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
        if hard_rows is not None:
            hard_rows.append(torch.cat(hard_chunks, dim=0))
    accuracies = individual_correct.numpy()/len(images)
    nll = individual_nll.numpy()/len(images)
    hard = torch.cat(hard_rows, dim=1) if hard_rows is not None else None
    return {
        "sample_count": sample_count,
        "example_count": len(images),
        "ensemble_predictive_accuracy": predictive_correct/len(images),
        "ensemble_predictive_nll": predictive_nll/len(images),
        "individual_accuracy_mean": float(np.mean(accuracies)),
        "individual_accuracy_min": float(np.min(accuracies)),
        "individual_accuracy_max": float(np.max(accuracies)),
        "individual_nll_mean": float(np.mean(nll)),
    }, hard


def function_audit(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    values = predictions.numpy()
    targets = labels.cpu().numpy()
    seed_count, example_count = values.shape
    counts = np.stack(
        [(values == label).sum(axis=0) for label in range(10)], axis=0
    )
    modal = counts.argmax(axis=0)
    agreement = counts.max(axis=0)/seed_count
    correct = modal == targets
    pair_agreements = sum(
        int(((local*(local-1))//2).sum()) for local in counts
    )
    total_pairs = math.comb(seed_count, 2)*example_count
    calibration = []
    for threshold in (0.90, 0.95, 0.99, 1.00):
        selected = agreement >= threshold
        calibration.append({
            "threshold": threshold,
            "coverage": float(selected.mean()),
            "modal_accuracy": (
                float(correct[selected].mean()) if selected.any() else None
            ),
        })
    hashes = {
        hashlib.sha256(row.tobytes()).hexdigest() for row in values
    }
    return {
        "unique_complete_test_function_count": len(hashes),
        "seed_count": seed_count,
        "hard_modal_accuracy": float(correct.mean()),
        "agreement_mean": float(agreement.mean()),
        "mean_pairwise_function_disagreement": 1-pair_agreements/total_pairs,
        "agreement_calibration": calibration,
    }


def objective_per_seed(
    parameters: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective_name: str,
    train_count: int,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seeds = len(parameters)
    repeated = labels[None].expand(seeds, -1).reshape(-1)
    ce = F.cross_entropy(
        logits.reshape(-1, 10), repeated, reduction="none"
    ).reshape(seeds, -1).mean(dim=1)
    prior = 0.5*(parameters*scales[None]).square().sum(dim=1)/train_count
    if objective_name == "plain":
        objective = ce
    elif objective_name == "map_same_prior":
        objective = ce+prior
    else:
        raise ValueError(f"未知objective：{objective_name}")
    return objective, ce, prior


def protocol_payload(data_source: str) -> dict[str, Any]:
    return {
        "protocol_version": Config.PROTOCOL_VERSION,
        "network": "Conv1x4 -> Conv4x8 -> Linear392x10",
        "parameter_count": parameter_count(),
        "parameter_coordinate": "standard actual weights",
        "initial_function_distribution": (
            "raw iid N(0,1), transformed by the exact HMC layer scales"
        ),
        "profiles": Config.PROFILES,
        "objectives": Config.OBJECTIVES,
        "seed_count": Config.SEED_COUNT,
        "epochs": Config.EPOCHS,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
        "optimizer": "torch.optim.Adam",
        "weight_decay": 0.0,
        "data_augmentation": False,
        "evaluation_epochs": Config.EVALUATION_EPOCHS,
        "data_seed": Config.DATA_SEED,
        "parameter_seed": Config.PARAMETER_SEED,
        "batch_order_seed": Config.BATCH_ORDER_SEED,
        "test_labels_control_training": False,
        "mnist_source": data_source,
    }


def protocol_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        json_ready(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_condition(
    profile: dict[str, Any],
    objective_name: str,
    all_train_images: torch.Tensor,
    all_train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    protocol_sha256: str,
    condition_ordinal: int,
    condition_count: int,
    overall_started: float,
) -> dict[str, Any]:
    device = all_train_images.device
    name = f"{profile['name']}__{objective_name}"
    condition_dir = Config.RESULT_DIR/name
    condition_dir.mkdir(parents=True, exist_ok=True)
    result_path = condition_dir/"result.pt"
    if Config.RESUME and result_path.exists():
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        if result["protocol_sha256"] != protocol_sha256:
            raise RuntimeError(f"{name}已有结果的协议不一致。")
        print(f"跳过已完成条件：{name}", flush=True)
        return result["summary"]

    train_indices_device = train_indices.to(device)
    validation_indices_device = validation_indices.to(device)
    train_images = all_train_images[train_indices_device]
    train_labels = all_train_labels[train_indices_device]
    validation_images = all_train_images[validation_indices_device]
    validation_labels = all_train_labels[validation_indices_device]
    train_count = len(train_images)
    scales = prior_scales(device)

    checkpoint_path = condition_dir/"checkpoint.pt"
    log_rows: list[dict[str, Any]] = []
    if Config.RESUME and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint["protocol_sha256"] != protocol_sha256:
            raise RuntimeError(f"{name} checkpoint协议不一致。")
        parameters = torch.nn.Parameter(checkpoint["parameters"].to(device))
        optimizer = torch.optim.Adam(
            [parameters], lr=Config.LEARNING_RATE
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch_start = int(checkpoint["epoch"])+1
        log_rows = checkpoint["log_rows"]
        prior_elapsed = float(checkpoint["elapsed_seconds"])
        print(f"继续{name}：epoch={epoch_start}", flush=True)
    else:
        parameters = torch.nn.Parameter(initialize_actual_parameters(device))
        optimizer = torch.optim.Adam(
            [parameters], lr=Config.LEARNING_RATE
        )
        epoch_start = 1
        prior_elapsed = 0.0

    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(
        Config.BATCH_ORDER_SEED+int(profile["train_count"])
    )
    # 恢复时重放此前epoch的randperm，保证后续batch顺序完全相同。
    for _ in range(1, epoch_start):
        torch.randperm(train_count, generator=batch_generator)

    started = time.perf_counter()
    for epoch in range(epoch_start, Config.EPOCHS+1):
        permutation = torch.randperm(
            train_count, generator=batch_generator
        ).to(device)
        ce_sum = torch.zeros(Config.SEED_COUNT, device=device)
        prior_sum = torch.zeros(Config.SEED_COUNT, device=device)
        batch_count = 0
        for start in range(0, train_count, Config.BATCH_SIZE):
            stop = min(start+Config.BATCH_SIZE, train_count)
            rows = permutation[start:stop]
            logits = forward_batched(parameters, train_images[rows])
            objective, ce, prior = objective_per_seed(
                parameters,
                logits,
                train_labels[rows],
                objective_name,
                train_count,
                scales,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.sum().backward()
            optimizer.step()
            ce_sum += ce.detach()
            prior_sum += prior.detach()
            batch_count += 1

        elapsed = prior_elapsed+time.perf_counter()-started
        epochs_done_total = (
            condition_ordinal*Config.EPOCHS+epoch
        )
        epochs_total = condition_count*Config.EPOCHS
        eta = elapsed/max(epoch, 1)*(Config.EPOCHS-epoch)
        row: dict[str, Any] = {
            "profile": profile["name"],
            "objective": objective_name,
            "epoch": epoch,
            "mean_batch_ce": float((ce_sum/batch_count).mean().item()),
            "mean_prior_term": float((prior_sum/batch_count).mean().item()),
            "elapsed_seconds_condition": elapsed,
        }
        if epoch in Config.EVALUATION_EPOCHS:
            validation_metrics, _ = evaluate_distribution(
                parameters, validation_images, validation_labels
            )
            test_metrics, _ = evaluate_distribution(
                parameters, test_images, test_labels
            )
            row.update({
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            })
            row.update({
                f"test_{key}": value for key, value in test_metrics.items()
            })
        log_rows.append(row)
        write_csv(condition_dir/"training_log.csv", log_rows)
        torch.save({
            "protocol_sha256": protocol_sha256,
            "epoch": epoch,
            "parameters": parameters.detach().cpu(),
            "optimizer_state": optimizer.state_dict(),
            "log_rows": log_rows,
            "elapsed_seconds": elapsed,
        }, checkpoint_path)
        test_text = (
            f" | test={row['test_ensemble_predictive_accuracy']:.3%}"
            if "test_ensemble_predictive_accuracy" in row else ""
        )
        print(
            f"[{condition_ordinal+1}/{condition_count}] {name} "
            f"epoch={epoch:>2}/{Config.EPOCHS} | "
            f"CE={row['mean_batch_ce']:.4f}{test_text} | "
            f"condition_ETA={format_duration(eta)} | "
            f"global_epochs={epochs_done_total}/{epochs_total}",
            flush=True,
        )

    train_metrics, _ = evaluate_distribution(
        parameters, train_images, train_labels
    )
    validation_metrics, _ = evaluate_distribution(
        parameters, validation_images, validation_labels
    )
    test_metrics, test_predictions = evaluate_distribution(
        parameters, test_images, test_labels, return_predictions=True
    )
    audit = function_audit(test_predictions, test_labels)
    summary = {
        "profile": profile["name"],
        "objective": objective_name,
        "train_count": train_count,
        "validation_count": len(validation_images),
        "seed_count": Config.SEED_COUNT,
        "epochs": Config.EPOCHS,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "function_audit": audit,
        "elapsed_seconds": prior_elapsed+time.perf_counter()-started,
    }
    torch.save({
        "protocol_sha256": protocol_sha256,
        "summary": summary,
        "final_actual_parameters": parameters.detach().cpu(),
        "test_hard_predictions": test_predictions,
        "test_labels": test_labels.cpu(),
    }, result_path)
    write_json(condition_dir/"summary.json", summary)
    return summary


def create_archive(result_dir: Path) -> Path:
    path = result_dir.parent/f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for item in sorted(result_dir.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(result_dir.parent))
    return path


def main() -> None:
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

    protocol = protocol_payload(str(data["source"]))
    protocol_sha256 = protocol_hash(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    write_json(Config.RESULT_DIR/"protocol.json", protocol)
    conditions = [
        (profile, objective)
        for profile in Config.PROFILES
        for objective in Config.OBJECTIVES
    ]
    print(
        "=== MNIST10 same-protocol Adam baseline ===\n"
        f"device={device} | gpu="
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}\n"
        f"seeds={Config.SEED_COUNT} | epochs={Config.EPOCHS} | "
        f"batch={Config.BATCH_SIZE} | conditions={conditions}",
        flush=True,
    )

    overall_started = time.perf_counter()
    summaries = []
    for ordinal, (profile, objective) in enumerate(conditions):
        train_indices, validation_indices = load_split(profile)
        summaries.append(run_condition(
            profile,
            objective,
            all_train_images,
            all_train_labels,
            test_images,
            test_labels,
            train_indices,
            validation_indices,
            protocol_sha256,
            ordinal,
            len(conditions),
            overall_started,
        ))
    final = {
        "status": "completed",
        "protocol_sha256": protocol_sha256,
        "elapsed_seconds": time.perf_counter()-overall_started,
        "conditions": summaries,
        "interpretation_boundary": [
            "Same architecture, data splits, and initial function distribution as HMC.",
            "Adam is optimized in conventional actual-weight coordinates.",
            "All seeds share a fixed minibatch order; only initialization differs.",
            "No data augmentation, early stopping, or test-driven tuning is used.",
        ],
    }
    write_json(Config.RESULT_DIR/"summary.json", final)
    print(
        f"完成，用时{format_duration(final['elapsed_seconds'])}", flush=True
    )
    if Config.PACKAGE_RESULTS:
        print(f"下载压缩包：{create_archive(Config.RESULT_DIR)}", flush=True)


if __name__ == "__main__":
    main()
