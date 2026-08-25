#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MNIST 隐藏标签噪声的长期训练实验。

研究问题
--------
将固定的 80% 对称错误标签混入 MNIST 训练集，且不给模型任何噪声标记：

1. 模型是否先学到可复用的数字规则？
2. 继续训练到 noisy training labels 近乎完全拟合后，干净测试准确率如何变化？
3. 文献中 validation-selected 的最佳点，与真正的插值点、长期终点有多大差异？

实验判决
--------
脚本同时运行干净标签对照和 80% 噪声条件，并使用相同初始化、相同 batch
顺序。每个架构记录三个互不混淆的时间点：

- best_validation：干净验证集准确率最高的 checkpoint；
- first_interpolation：noisy train accuracy 首次达到设定阈值的 checkpoint；
- final：达到插值后继续训练一段时间，或达到最大 epoch 的最终 checkpoint。

只有 first_interpolation 或 final 同时满足近乎完全拟合 noisy labels、且干净
test accuracy 仍很高，才能支持“强版本”的良性/神奇过拟合。最佳验证点不算。

实现说明
--------
- hinton_small：尽量贴近 Guan et al. (2018) MNIST 演示的小 CNN；
- overparam_cnn：容量显著更大，用于确保“记不住噪声”不是逃避判决的理由；
- 多个 seed 和 clean/noise 条件用 grouped convolution 并行训练，彼此无参数共享；
- 同一 seed 的 clean/noise 模型逐元素使用相同初始化；
- 固定一次性污染标签，所有 seed 共用同一数据划分和噪声现场；
- 所有用户设置集中在 Config，不依赖环境变量；
- 输出 CSV、JSON、PNG，并自动生成便于下载的 zip。

参考设置
--------
Melody Y. Guan, Varun Gulshan, Andrew M. Dai, Geoffrey E. Hinton,
"Who Said What: Modeling Individual Labelers Improves Classification", 2018.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    conv1_channels: int
    conv2_channels: int
    hidden_units: int
    max_epochs: int


@dataclass
class Config:
    # =========================
    # 路径与运行模式
    # =========================
    DATA_DIR: Path = Path("/root/mnist_dataset")
    RESULT_DIR: Path = Path("/root/results_mnist_hidden_noise_long_training")
    DOWNLOAD_MNIST_IF_MISSING: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SMOKE_TEST: bool = False

    # 可单独删去某个名字，只运行一个架构。
    # 默认先跑有能力完成噪声插值的大模型，尽快给出核心判决。
    ARCHITECTURES: tuple[str, ...] = ("overparam_cnn", "hinton_small")
    MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)
    CONDITIONS: tuple[str, ...] = ("clean", "noise80")
    HINTON_SMALL_MAX_EPOCHS: int = 300
    OVERPARAM_MAX_EPOCHS: int = 300

    # =========================
    # 数据与固定噪声
    # =========================
    TRAIN_SIZE: int = 50_000
    VALIDATION_SIZE: int = 10_000
    TEST_LIMIT: int | None = None
    SPLIT_SEED: int = 20260820
    NOISE_SEED: int = 20260821
    DATA_ORDER_SEED: int = 20260822
    CORRUPTION_RATE: float = 0.80
    EXACT_CORRUPTION_COUNT: bool = True
    NUM_CLASSES: int = 10
    NORMALIZE_MEAN: float = 0.1307
    NORMALIZE_STD: float = 0.3081
    CACHE_DATA_ON_DEVICE: bool = True

    # =========================
    # 训练
    # =========================
    BATCH_SIZE: int = 200
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
    ADAM_EPS: float = 1e-8
    GRAD_CLIP_NORM: float | None = None

    # noisy train accuracy 达到此阈值，记为首次近似插值。
    INTERPOLATION_ACCURACY: float = 0.9999
    # 所有 noise seed 插值后仍继续训练，观察规则是否继续受损。
    POST_INTERPOLATION_EPOCHS: int = 40

    # =========================
    # 评估、输出与性能
    # =========================
    EVAL_INTERVAL_EPOCHS: int = 5
    DENSE_EVAL_UNTIL_EPOCH: int = 20
    EVAL_BATCH_SIZE: int = 2_000
    LOG_EVERY_EVAL: bool = True
    ALLOW_TF32: bool = True
    USE_TORCH_COMPILE: bool = False

    GENERATE_PLOTS: bool = True
    CREATE_ARCHIVE: bool = True
    SAVE_FINAL_WEIGHTS: bool = False
    INCLUDE_WEIGHTS_IN_ARCHIVE: bool = False

    _SMOKE_RESULT_DIR: Path = field(
        default=Path(
            "research/overfitting_related_research/"
            "_smoke_mnist_hidden_noise_long_training"
        ),
        repr=False,
    )
    _SMOKE_DATA_DIR: Path = field(
        default=Path(
            "research/overfitting_related_research/"
            "_smoke_mnist_dataset"
        ),
        repr=False,
    )


def architecture_specs(cfg: Config) -> list[ArchitectureSpec]:
    catalog = {
        # 两个 valid 5x5 conv、各接 2x2 max pool、FC 32、10-way softmax。
        "hinton_small": ArchitectureSpec(
            name="hinton_small",
            conv1_channels=16,
            conv2_channels=25,
            hidden_units=32,
            max_epochs=cfg.HINTON_SMALL_MAX_EPOCHS,
        ),
        # 明确增加容量，使完整记忆固定噪声在工程上可达。
        "overparam_cnn": ArchitectureSpec(
            name="overparam_cnn",
            conv1_channels=64,
            conv2_channels=128,
            hidden_units=512,
            max_epochs=cfg.OVERPARAM_MAX_EPOCHS,
        ),
    }
    unknown = sorted(set(cfg.ARCHITECTURES) - set(catalog))
    if unknown:
        raise ValueError(f"未知架构：{unknown}；可选值={sorted(catalog)}")
    if cfg.SMOKE_TEST:
        return [
            ArchitectureSpec(
                name="hinton_small",
                conv1_channels=4,
                conv2_channels=6,
                hidden_units=8,
                max_epochs=2,
            )
        ]
    return [catalog[name] for name in cfg.ARCHITECTURES]


def apply_smoke_overrides(cfg: Config) -> Config:
    if not cfg.SMOKE_TEST:
        return cfg
    cfg.RESULT_DIR = cfg._SMOKE_RESULT_DIR
    cfg.DATA_DIR = cfg._SMOKE_DATA_DIR
    cfg.ARCHITECTURES = ("hinton_small",)
    cfg.MODEL_SEEDS = (0,)
    cfg.TRAIN_SIZE = 512
    cfg.VALIDATION_SIZE = 256
    cfg.TEST_LIMIT = 256
    cfg.BATCH_SIZE = 128
    cfg.EVAL_BATCH_SIZE = 256
    cfg.EVAL_INTERVAL_EPOCHS = 1
    cfg.DENSE_EVAL_UNTIL_EPOCH = 2
    cfg.POST_INTERPOLATION_EPOCHS = 1
    cfg.GENERATE_PLOTS = True
    cfg.CREATE_ARCHIVE = True
    cfg.SAVE_FINAL_WEIGHTS = False
    cfg.USE_TORCH_COMPILE = False
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.TRAIN_SIZE <= 0 or cfg.VALIDATION_SIZE <= 0:
        raise ValueError("TRAIN_SIZE 和 VALIDATION_SIZE 必须为正数。")
    if cfg.TRAIN_SIZE + cfg.VALIDATION_SIZE > 60_000:
        raise ValueError("MNIST train split 总计只有 60,000 张图片。")
    if not 0.0 <= cfg.CORRUPTION_RATE < 1.0:
        raise ValueError("CORRUPTION_RATE 必须位于 [0, 1)。")
    if cfg.NUM_CLASSES != 10:
        raise ValueError("本实验固定使用 MNIST 十分类。")
    if not cfg.MODEL_SEEDS:
        raise ValueError("至少需要一个 MODEL_SEEDS。")
    if tuple(cfg.CONDITIONS) != ("clean", "noise80"):
        raise ValueError("CONDITIONS 当前必须保持 ('clean', 'noise80')。")
    if cfg.BATCH_SIZE <= 0 or cfg.EVAL_BATCH_SIZE <= 0:
        raise ValueError("batch size 必须为正数。")
    if not 0.0 < cfg.INTERPOLATION_ACCURACY <= 1.0:
        raise ValueError("INTERPOLATION_ACCURACY 必须位于 (0, 1]。")
    if cfg.POST_INTERPOLATION_EPOCHS < 0:
        raise ValueError("POST_INTERPOLATION_EPOCHS 不能为负数。")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetBundle:
    train_images: torch.Tensor
    train_true_labels: torch.Tensor
    train_noisy_labels: torch.Tensor
    corruption_mask: torch.Tensor
    validation_images: torch.Tensor
    validation_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    metadata: dict[str, Any]


def normalize_images(images: torch.Tensor, cfg: Config) -> torch.Tensor:
    values = images.to(dtype=torch.float32).unsqueeze(1).div_(255.0)
    return values.sub_(cfg.NORMALIZE_MEAN).div_(cfg.NORMALIZE_STD)


def load_dataset(cfg: Config, device: torch.device) -> DatasetBundle:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            "缺少 torchvision。AutoDL 常规 PyTorch 镜像已包含；否则请安装 torchvision。"
        ) from exc

    train_set = MNIST(
        root=str(cfg.DATA_DIR),
        train=True,
        download=cfg.DOWNLOAD_MNIST_IF_MISSING,
    )
    test_set = MNIST(
        root=str(cfg.DATA_DIR),
        train=False,
        download=cfg.DOWNLOAD_MNIST_IF_MISSING,
    )

    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(cfg.SPLIT_SEED)
    permutation = torch.randperm(len(train_set.data), generator=split_generator)
    train_indices = permutation[: cfg.TRAIN_SIZE]
    validation_indices = permutation[
        cfg.TRAIN_SIZE : cfg.TRAIN_SIZE + cfg.VALIDATION_SIZE
    ]

    train_images = normalize_images(train_set.data[train_indices], cfg)
    train_true = train_set.targets[train_indices].to(dtype=torch.long).clone()
    validation_images = normalize_images(train_set.data[validation_indices], cfg)
    validation_labels = train_set.targets[validation_indices].to(dtype=torch.long)

    test_count = len(test_set.data)
    if cfg.TEST_LIMIT is not None:
        test_count = min(test_count, int(cfg.TEST_LIMIT))
    test_images = normalize_images(test_set.data[:test_count], cfg)
    test_labels = test_set.targets[:test_count].to(dtype=torch.long)

    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(cfg.NOISE_SEED)
    corruption_mask = torch.zeros(cfg.TRAIN_SIZE, dtype=torch.bool)
    if cfg.EXACT_CORRUPTION_COUNT:
        corrupt_count = int(round(cfg.TRAIN_SIZE * cfg.CORRUPTION_RATE))
        corrupt_order = torch.randperm(cfg.TRAIN_SIZE, generator=noise_generator)
        corruption_mask[corrupt_order[:corrupt_count]] = True
    else:
        corruption_mask = (
            torch.rand(cfg.TRAIN_SIZE, generator=noise_generator)
            < cfg.CORRUPTION_RATE
        )
        corrupt_count = int(corruption_mask.sum().item())

    train_noisy = train_true.clone()
    # 加 1..9 的随机偏移，确保污染标签一定不同于真实标签。
    wrong_offsets = torch.randint(
        1,
        cfg.NUM_CLASSES,
        size=(corrupt_count,),
        generator=noise_generator,
    )
    train_noisy[corruption_mask] = (
        train_true[corruption_mask] + wrong_offsets
    ) % cfg.NUM_CLASSES

    transition = torch.zeros(
        cfg.NUM_CLASSES,
        cfg.NUM_CLASSES,
        dtype=torch.int64,
    )
    flat_index = train_true * cfg.NUM_CLASSES + train_noisy
    transition.view(-1).scatter_add_(
        0,
        flat_index,
        torch.ones_like(flat_index, dtype=torch.int64),
    )

    metadata = {
        "train_size": cfg.TRAIN_SIZE,
        "validation_size": cfg.VALIDATION_SIZE,
        "test_size": test_count,
        "corruption_count": corrupt_count,
        "realized_corruption_rate": corrupt_count / cfg.TRAIN_SIZE,
        "clean_count": cfg.TRAIN_SIZE - corrupt_count,
        "split_seed": cfg.SPLIT_SEED,
        "noise_seed": cfg.NOISE_SEED,
        "transition_matrix_true_rows_noisy_columns": transition.tolist(),
        "train_true_class_counts": torch.bincount(
            train_true, minlength=cfg.NUM_CLASSES
        ).tolist(),
        "train_noisy_class_counts": torch.bincount(
            train_noisy, minlength=cfg.NUM_CLASSES
        ).tolist(),
    }

    if cfg.CACHE_DATA_ON_DEVICE:
        train_images = train_images.to(device)
        train_true = train_true.to(device)
        train_noisy = train_noisy.to(device)
        corruption_mask = corruption_mask.to(device)
        validation_images = validation_images.to(device)
        validation_labels = validation_labels.to(device)
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

    return DatasetBundle(
        train_images=train_images,
        train_true_labels=train_true,
        train_noisy_labels=train_noisy,
        corruption_mask=corruption_mask,
        validation_images=validation_images,
        validation_labels=validation_labels,
        test_images=test_images,
        test_labels=test_labels,
        metadata=metadata,
    )


def uniform_parameter_slice(
    tensor: torch.Tensor,
    model_index: int,
    bound: float,
    generator: torch.Generator,
) -> None:
    tensor[model_index].uniform_(-bound, bound, generator=generator)


class ParallelIndependentCNN(nn.Module):
    """使用 grouped convolution 同时训练若干个完全独立的 CNN。"""

    def __init__(
        self,
        spec: ArchitectureSpec,
        slot_seeds: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.model_count = len(slot_seeds)
        self.num_classes = num_classes

        # 28 -> valid conv5: 24 -> pool: 12 -> valid conv5: 8 -> pool: 4
        self.flat_features = spec.conv2_channels * 4 * 4

        self.conv1_weight = nn.Parameter(
            torch.empty(self.model_count, spec.conv1_channels, 1, 5, 5)
        )
        self.conv1_bias = nn.Parameter(
            torch.empty(self.model_count, spec.conv1_channels)
        )
        self.conv2_weight = nn.Parameter(
            torch.empty(
                self.model_count,
                spec.conv2_channels,
                spec.conv1_channels,
                5,
                5,
            )
        )
        self.conv2_bias = nn.Parameter(
            torch.empty(self.model_count, spec.conv2_channels)
        )
        self.fc1_weight = nn.Parameter(
            torch.empty(
                self.model_count,
                spec.hidden_units,
                self.flat_features,
            )
        )
        self.fc1_bias = nn.Parameter(
            torch.empty(self.model_count, spec.hidden_units)
        )
        self.fc2_weight = nn.Parameter(
            torch.empty(
                self.model_count,
                num_classes,
                spec.hidden_units,
            )
        )
        self.fc2_bias = nn.Parameter(
            torch.empty(self.model_count, num_classes)
        )
        self.reset_parameters(slot_seeds)

    def reset_parameters(self, slot_seeds: list[int]) -> None:
        if len(slot_seeds) != self.model_count:
            raise ValueError("slot_seeds 数量与并行模型数量不一致。")
        with torch.no_grad():
            for model_index, seed in enumerate(slot_seeds):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed))
                uniform_parameter_slice(
                    self.conv1_weight,
                    model_index,
                    1.0 / math.sqrt(25),
                    generator,
                )
                uniform_parameter_slice(
                    self.conv1_bias,
                    model_index,
                    1.0 / math.sqrt(25),
                    generator,
                )
                conv2_fan_in = self.spec.conv1_channels * 25
                uniform_parameter_slice(
                    self.conv2_weight,
                    model_index,
                    1.0 / math.sqrt(conv2_fan_in),
                    generator,
                )
                uniform_parameter_slice(
                    self.conv2_bias,
                    model_index,
                    1.0 / math.sqrt(conv2_fan_in),
                    generator,
                )
                uniform_parameter_slice(
                    self.fc1_weight,
                    model_index,
                    1.0 / math.sqrt(self.flat_features),
                    generator,
                )
                uniform_parameter_slice(
                    self.fc1_bias,
                    model_index,
                    1.0 / math.sqrt(self.flat_features),
                    generator,
                )
                uniform_parameter_slice(
                    self.fc2_weight,
                    model_index,
                    1.0 / math.sqrt(self.spec.hidden_units),
                    generator,
                )
                uniform_parameter_slice(
                    self.fc2_bias,
                    model_index,
                    1.0 / math.sqrt(self.spec.hidden_units),
                    generator,
                )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        # 每个独立模型收到完全相同的输入图片。
        grouped = images.expand(-1, self.model_count, -1, -1)
        grouped = F.conv2d(
            grouped,
            self.conv1_weight.reshape(
                self.model_count * self.spec.conv1_channels, 1, 5, 5
            ),
            self.conv1_bias.reshape(-1),
            groups=self.model_count,
        )
        grouped = F.relu(grouped)
        grouped = F.max_pool2d(grouped, kernel_size=2, stride=2)

        grouped = F.conv2d(
            grouped,
            self.conv2_weight.reshape(
                self.model_count * self.spec.conv2_channels,
                self.spec.conv1_channels,
                5,
                5,
            ),
            self.conv2_bias.reshape(-1),
            groups=self.model_count,
        )
        grouped = F.relu(grouped)
        grouped = F.max_pool2d(grouped, kernel_size=2, stride=2)
        grouped = grouped.reshape(
            batch_size,
            self.model_count,
            self.flat_features,
        )

        hidden = torch.einsum(
            "bmf,mhf->bmh",
            grouped,
            self.fc1_weight,
        )
        hidden = F.relu(hidden + self.fc1_bias.unsqueeze(0))
        logits = torch.einsum(
            "bmh,mch->bmc",
            hidden,
            self.fc2_weight,
        )
        logits = logits + self.fc2_bias.unsqueeze(0)
        return logits.permute(1, 0, 2).contiguous()  # [models, batch, classes]


def build_slots(cfg: Config) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for condition in cfg.CONDITIONS:
        for seed in cfg.MODEL_SEEDS:
            slots.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    # clean/noise 配对必须有完全相同的初始化。
                    "init_seed": 71_000_000 + int(seed),
                }
            )
    return slots


def target_matrix_for_batch(
    slots: list[dict[str, Any]],
    true_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
) -> torch.Tensor:
    targets = [
        true_labels if slot["condition"] == "clean" else noisy_labels
        for slot in slots
    ]
    return torch.stack(targets, dim=0)


def per_model_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model_count, batch_size, num_classes = logits.shape
    losses = F.cross_entropy(
        logits.reshape(model_count * batch_size, num_classes),
        targets.reshape(model_count * batch_size),
        reduction="none",
    )
    return losses.reshape(model_count, batch_size)


def maybe_to_device(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=True)


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = maybe_to_device(images[start : start + batch_size], device)
        logits = model(batch)
        logits_parts.append(logits.detach().cpu())
        prediction_parts.append(logits.argmax(dim=-1).detach().cpu())
    return (
        torch.cat(logits_parts, dim=1),
        torch.cat(prediction_parts, dim=1),
    )


def metric_vectors(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, list[float | int]]:
    labels_cpu = labels.detach().cpu().to(dtype=torch.long)
    model_count, sample_count, num_classes = logits.shape
    repeated = labels_cpu.unsqueeze(0).expand(model_count, -1)
    losses = F.cross_entropy(
        logits.reshape(model_count * sample_count, num_classes),
        repeated.reshape(-1),
        reduction="none",
    ).reshape(model_count, sample_count)
    correct = predictions.eq(repeated)
    return {
        "loss": losses.mean(dim=1).tolist(),
        "accuracy": correct.float().mean(dim=1).tolist(),
        "correct": correct.sum(dim=1).tolist(),
        "errors": (~correct).sum(dim=1).tolist(),
    }


def subset_metric_vectors(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, list[float | int]]:
    mask_cpu = mask.detach().cpu().to(dtype=torch.bool)
    if int(mask_cpu.sum().item()) == 0:
        count = logits.shape[0]
        return {
            "loss": [float("nan")] * count,
            "accuracy": [float("nan")] * count,
            "correct": [0] * count,
            "errors": [0] * count,
        }
    return metric_vectors(
        logits[:, mask_cpu, :],
        predictions[:, mask_cpu],
        labels.detach().cpu()[mask_cpu],
    )


@torch.inference_mode()
def evaluate_all(
    model: nn.Module,
    dataset: DatasetBundle,
    slots: list[dict[str, Any]],
    device: torch.device,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    train_logits, train_predictions = collect_predictions(
        model,
        dataset.train_images,
        device,
        cfg.EVAL_BATCH_SIZE,
    )
    validation_logits, validation_predictions = collect_predictions(
        model,
        dataset.validation_images,
        device,
        cfg.EVAL_BATCH_SIZE,
    )
    test_logits, test_predictions = collect_predictions(
        model,
        dataset.test_images,
        device,
        cfg.EVAL_BATCH_SIZE,
    )

    train_true = metric_vectors(
        train_logits,
        train_predictions,
        dataset.train_true_labels,
    )
    train_noisy = metric_vectors(
        train_logits,
        train_predictions,
        dataset.train_noisy_labels,
    )
    corrupted_true = subset_metric_vectors(
        train_logits,
        train_predictions,
        dataset.train_true_labels,
        dataset.corruption_mask,
    )
    corrupted_noisy = subset_metric_vectors(
        train_logits,
        train_predictions,
        dataset.train_noisy_labels,
        dataset.corruption_mask,
    )
    uncorrupted_true = subset_metric_vectors(
        train_logits,
        train_predictions,
        dataset.train_true_labels,
        ~dataset.corruption_mask,
    )
    validation_true = metric_vectors(
        validation_logits,
        validation_predictions,
        dataset.validation_labels,
    )
    test_true = metric_vectors(
        test_logits,
        test_predictions,
        dataset.test_labels,
    )

    clean_slot_by_seed = {
        int(slot["seed"]): index
        for index, slot in enumerate(slots)
        if slot["condition"] == "clean"
    }
    paired_test_agreement: dict[int, float] = {}
    paired_validation_agreement: dict[int, float] = {}
    for index, slot in enumerate(slots):
        if slot["condition"] != "noise80":
            continue
        clean_index = clean_slot_by_seed[int(slot["seed"])]
        paired_test_agreement[index] = float(
            test_predictions[index]
            .eq(test_predictions[clean_index])
            .float()
            .mean()
            .item()
        )
        paired_validation_agreement[index] = float(
            validation_predictions[index]
            .eq(validation_predictions[clean_index])
            .float()
            .mean()
            .item()
        )

    rows: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        condition = str(slot["condition"])
        observed = train_true if condition == "clean" else train_noisy
        row = {
            "slot_index": index,
            "condition": condition,
            "seed": int(slot["seed"]),
            "train_observed_loss": observed["loss"][index],
            "train_observed_accuracy": observed["accuracy"][index],
            "train_observed_errors": observed["errors"][index],
            "train_true_loss": train_true["loss"][index],
            "train_true_accuracy": train_true["accuracy"][index],
            "train_true_errors": train_true["errors"][index],
            "train_noisy_loss": train_noisy["loss"][index],
            "train_noisy_accuracy": train_noisy["accuracy"][index],
            "train_noisy_errors": train_noisy["errors"][index],
            "corrupted_subset_true_accuracy": corrupted_true["accuracy"][index],
            "corrupted_subset_noisy_accuracy": corrupted_noisy["accuracy"][index],
            "uncorrupted_subset_true_accuracy": uncorrupted_true["accuracy"][index],
            "validation_true_loss": validation_true["loss"][index],
            "validation_true_accuracy": validation_true["accuracy"][index],
            "validation_true_errors": validation_true["errors"][index],
            "test_true_loss": test_true["loss"][index],
            "test_true_accuracy": test_true["accuracy"][index],
            "test_true_errors": test_true["errors"][index],
            "paired_clean_test_agreement": paired_test_agreement.get(
                index, float("nan")
            ),
            "paired_clean_validation_agreement": paired_validation_agreement.get(
                index, float("nan")
            ),
        }
        rows.append(row)

    prediction_payload = {
        "validation": validation_predictions,
        "test": test_predictions,
    }
    return rows, prediction_payload


def should_evaluate(
    epoch: int,
    max_epochs: int,
    cfg: Config,
    force_dense: bool,
) -> bool:
    if epoch == 0 or epoch == max_epochs:
        return True
    if epoch <= cfg.DENSE_EVAL_UNTIL_EPOCH:
        return True
    if force_dense:
        return True
    return epoch % cfg.EVAL_INTERVAL_EPOCHS == 0


def condition_summary(
    rows: list[dict[str, Any]],
    condition: str,
) -> dict[str, tuple[float, float, float]]:
    subset = [row for row in rows if row["condition"] == condition]
    fields = (
        "train_observed_accuracy",
        "train_observed_loss",
        "train_true_accuracy",
        "corrupted_subset_noisy_accuracy",
        "corrupted_subset_true_accuracy",
        "validation_true_accuracy",
        "test_true_accuracy",
        "paired_clean_test_agreement",
    )
    summary: dict[str, tuple[float, float, float]] = {}
    for field_name in fields:
        values = np.asarray(
            [float(row[field_name]) for row in subset], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[field_name] = (float("nan"), float("nan"), float("nan"))
        else:
            summary[field_name] = (
                float(finite.mean()),
                float(finite.min()),
                float(finite.max()),
            )
    return summary


def format_triplet(values: tuple[float, float, float], percent: bool) -> str:
    mean, minimum, maximum = values
    if not math.isfinite(mean):
        return "n/a"
    if percent:
        return f"{100.0 * mean:7.3f}% [{100.0 * minimum:.3f}, {100.0 * maximum:.3f}]"
    return f"{mean:.6g} [{minimum:.6g}, {maximum:.6g}]"


def print_evaluation_log(
    architecture: str,
    epoch: int,
    rows: list[dict[str, Any]],
    elapsed: float,
) -> None:
    clean = condition_summary(rows, "clean")
    noise = condition_summary(rows, "noise80")
    print(
        f"\n[{architecture}] epoch={epoch:,} | elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(
        "  clean   | observed_train="
        f"{format_triplet(clean['train_observed_accuracy'], True)} | "
        f"test={format_triplet(clean['test_true_accuracy'], True)}",
        flush=True,
    )
    print(
        "  noise80 | noisy_train="
        f"{format_triplet(noise['train_observed_accuracy'], True)} | "
        f"noisy_loss={format_triplet(noise['train_observed_loss'], False)} | "
        f"true_train={format_triplet(noise['train_true_accuracy'], True)}",
        flush=True,
    )
    print(
        "          | corrupted(noisy/true)="
        f"{format_triplet(noise['corrupted_subset_noisy_accuracy'], True)} / "
        f"{format_triplet(noise['corrupted_subset_true_accuracy'], True)} | "
        f"val={format_triplet(noise['validation_true_accuracy'], True)} | "
        f"test={format_triplet(noise['test_true_accuracy'], True)} | "
        f"paired_clean_agreement="
        f"{format_triplet(noise['paired_clean_test_agreement'], True)}",
        flush=True,
    )


def parameter_count_per_model(model: ParallelIndependentCNN) -> int:
    total = sum(parameter.numel() for parameter in model.parameters())
    return total // model.model_count


def clone_cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_architecture(
    cfg: Config,
    spec: ArchitectureSpec,
    dataset: DatasetBundle,
    device: torch.device,
    result_dir: Path,
) -> dict[str, Any]:
    slots = build_slots(cfg)
    slot_seeds = [int(slot["init_seed"]) for slot in slots]
    model = ParallelIndependentCNN(spec, slot_seeds, cfg.NUM_CLASSES).to(device)

    # 同一 seed 的 clean/noise 初始输出必须逐元素相同。
    with torch.inference_mode():
        probe = maybe_to_device(dataset.validation_images[:8], device)
        initial_logits = model(probe).detach().cpu()
    clean_index_by_seed = {
        int(slot["seed"]): index
        for index, slot in enumerate(slots)
        if slot["condition"] == "clean"
    }
    for index, slot in enumerate(slots):
        if slot["condition"] != "noise80":
            continue
        paired = clean_index_by_seed[int(slot["seed"])]
        max_diff = float(
            (initial_logits[index] - initial_logits[paired]).abs().max().item()
        )
        # grouped convolution 的不同 group 位置可能产生约 1e-8 的 FP32
        # 舍入差；权重切片仍逐元素相同，因此这里只排除真实串扰。
        if max_diff > 1e-6:
            raise RuntimeError(
                "clean/noise 配对初始化不一致："
                f"seed={slot['seed']}, max_diff={max_diff}"
            )

    if cfg.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=cfg.ADAM_BETAS,
        eps=cfg.ADAM_EPS,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    history: list[dict[str, Any]] = []
    milestone_rows: list[dict[str, Any]] = []
    first_interpolation_epoch: dict[int, int] = {}
    best_validation: dict[int, dict[str, Any]] = {}
    first_interpolation_metrics: dict[int, dict[str, Any]] = {}
    best_validation_state: dict[str, torch.Tensor] | None = None
    best_validation_mean = -float("inf")
    all_noise_interpolated_epoch: int | None = None
    force_dense_eval = False
    start_time = time.perf_counter()

    architecture_dir = result_dir / spec.name
    architecture_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== 架构 {spec.name} ===\n"
        f"单模型参数={parameter_count_per_model(model):,} | "
        f"并行模型={len(slots)} | max_epochs={spec.max_epochs}",
        flush=True,
    )

    last_epoch = 0
    for epoch in range(0, spec.max_epochs + 1):
        if epoch > 0:
            model.train()
            order_generator = torch.Generator(device="cpu")
            order_generator.manual_seed(cfg.DATA_ORDER_SEED + epoch)
            order = torch.randperm(
                cfg.TRAIN_SIZE,
                generator=order_generator,
            )
            for start in range(0, cfg.TRAIN_SIZE, cfg.BATCH_SIZE):
                batch_indices_cpu = order[start : start + cfg.BATCH_SIZE]
                if dataset.train_images.device.type == "cpu":
                    batch_images = dataset.train_images[batch_indices_cpu].to(
                        device, non_blocking=True
                    )
                    batch_true = dataset.train_true_labels[
                        batch_indices_cpu
                    ].to(device, non_blocking=True)
                    batch_noisy = dataset.train_noisy_labels[
                        batch_indices_cpu
                    ].to(device, non_blocking=True)
                else:
                    batch_indices = batch_indices_cpu.to(device)
                    batch_images = dataset.train_images[batch_indices]
                    batch_true = dataset.train_true_labels[batch_indices]
                    batch_noisy = dataset.train_noisy_labels[batch_indices]

                targets = target_matrix_for_batch(
                    slots,
                    batch_true,
                    batch_noisy,
                )
                logits = model(batch_images)
                losses = per_model_cross_entropy(logits, targets).mean(dim=1)
                # 各模型参数完全独立，求和可保持与逐模型训练相同的梯度尺度。
                total_loss = losses.sum()
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if cfg.GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.GRAD_CLIP_NORM
                    )
                optimizer.step()

        last_epoch = epoch
        if not should_evaluate(
            epoch,
            spec.max_epochs,
            cfg,
            force_dense_eval,
        ):
            continue

        eval_rows, _ = evaluate_all(model, dataset, slots, device, cfg)
        elapsed = time.perf_counter() - start_time
        for row in eval_rows:
            row["architecture"] = spec.name
            row["epoch"] = epoch
            row["elapsed_seconds"] = elapsed
            row["is_interpolated"] = bool(
                row["condition"] == "noise80"
                and float(row["train_observed_accuracy"])
                >= cfg.INTERPOLATION_ACCURACY
            )
            history.append(row)

            slot_index = int(row["slot_index"])
            previous_best = best_validation.get(slot_index)
            if (
                previous_best is None
                or float(row["validation_true_accuracy"])
                > float(previous_best["validation_true_accuracy"])
            ):
                best_validation[slot_index] = dict(row)

            if (
                row["condition"] == "noise80"
                and bool(row["is_interpolated"])
                and slot_index not in first_interpolation_epoch
            ):
                first_interpolation_epoch[slot_index] = epoch
                first_interpolation_metrics[slot_index] = dict(row)
                milestone = dict(row)
                milestone["milestone"] = "first_interpolation"
                milestone_rows.append(milestone)
                print(
                    "  >>> 首次近似插值："
                    f"{spec.name} seed={row['seed']} epoch={epoch} | "
                    f"noisy_errors={row['train_observed_errors']} | "
                    f"test={100.0 * float(row['test_true_accuracy']):.3f}%",
                    flush=True,
                )

        noise_rows = [
            row for row in eval_rows if row["condition"] == "noise80"
        ]
        minimum_noisy_train = min(
            float(row["train_observed_accuracy"]) for row in noise_rows
        )
        if minimum_noisy_train >= 0.98:
            force_dense_eval = True

        noise_val_mean = float(
            np.mean(
                [float(row["validation_true_accuracy"]) for row in noise_rows]
            )
        )
        if noise_val_mean > best_validation_mean:
            best_validation_mean = noise_val_mean
            if cfg.SAVE_FINAL_WEIGHTS:
                best_validation_state = clone_cpu_state_dict(model)

        if cfg.LOG_EVERY_EVAL:
            print_evaluation_log(spec.name, epoch, eval_rows, elapsed)

        # 即使用户中途停止，已经完成的轨迹也不会只留在内存里。
        write_csv(architecture_dir / "trajectory.csv", history)
        write_json(
            architecture_dir / "progress.json",
            {
                "architecture": spec.name,
                "last_evaluated_epoch": epoch,
                "first_interpolation_epoch_by_slot": first_interpolation_epoch,
                "elapsed_seconds": elapsed,
            },
        )

        noise_slot_indices = {
            int(row["slot_index"])
            for row in eval_rows
            if row["condition"] == "noise80"
        }
        if (
            all_noise_interpolated_epoch is None
            and noise_slot_indices.issubset(first_interpolation_epoch)
        ):
            all_noise_interpolated_epoch = max(
                first_interpolation_epoch[index]
                for index in noise_slot_indices
            )
            print(
                f"  >>> 所有 noise seed 已插值；继续训练 "
                f"{cfg.POST_INTERPOLATION_EPOCHS} epoch。",
                flush=True,
            )

        if (
            all_noise_interpolated_epoch is not None
            and epoch
            >= all_noise_interpolated_epoch + cfg.POST_INTERPOLATION_EPOCHS
        ):
            break

    # 确保最终 epoch 一定被完整评估。
    if not history or int(history[-1]["epoch"]) != last_epoch:
        eval_rows, _ = evaluate_all(model, dataset, slots, device, cfg)
        elapsed = time.perf_counter() - start_time
        for row in eval_rows:
            row["architecture"] = spec.name
            row["epoch"] = last_epoch
            row["elapsed_seconds"] = elapsed
            row["is_interpolated"] = bool(
                row["condition"] == "noise80"
                and float(row["train_observed_accuracy"])
                >= cfg.INTERPOLATION_ACCURACY
            )
            history.append(row)

    final_rows = [
        row for row in history if int(row["epoch"]) == int(last_epoch)
    ]
    for row in best_validation.values():
        milestone = dict(row)
        milestone["milestone"] = "best_validation"
        milestone_rows.append(milestone)
    for row in final_rows:
        milestone = dict(row)
        milestone["milestone"] = "final"
        milestone_rows.append(milestone)

    write_csv(architecture_dir / "trajectory.csv", history)
    write_csv(architecture_dir / "milestones.csv", milestone_rows)

    if cfg.SAVE_FINAL_WEIGHTS:
        torch.save(
            clone_cpu_state_dict(model),
            architecture_dir / "final_parallel_models.pt",
        )
        if best_validation_state is not None:
            torch.save(
                best_validation_state,
                architecture_dir / "best_noise_validation_parallel_models.pt",
            )

    noise_final_rows = [
        row for row in final_rows if row["condition"] == "noise80"
    ]
    noise_best_rows = [
        row
        for index, row in best_validation.items()
        if slots[index]["condition"] == "noise80"
    ]
    interpolated_rows = list(first_interpolation_metrics.values())

    strong_at_first_interpolation = bool(
        len(interpolated_rows) == len(cfg.MODEL_SEEDS)
        and min(
            [float(row["test_true_accuracy"]) for row in interpolated_rows]
        )
        >= 0.90
    )
    strong_at_final = bool(
        len(noise_final_rows) == len(cfg.MODEL_SEEDS)
        and min(
            float(row["train_observed_accuracy"])
            for row in noise_final_rows
        )
        >= cfg.INTERPOLATION_ACCURACY
        and min(float(row["test_true_accuracy"]) for row in noise_final_rows)
        >= 0.90
    )
    architecture_summary = {
        "architecture": asdict(spec),
        "parameter_count_per_model": parameter_count_per_model(model),
        "parallel_model_count": len(slots),
        "last_epoch": last_epoch,
        "elapsed_seconds": time.perf_counter() - start_time,
        "all_noise_interpolated_epoch": all_noise_interpolated_epoch,
        "interpolated_seed_count": len(interpolated_rows),
        "requested_seed_count": len(cfg.MODEL_SEEDS),
        "strong_version_supported_at_first_interpolation": (
            strong_at_first_interpolation
        ),
        "strong_version_supported_at_final": strong_at_final,
        "noise80_best_validation": noise_best_rows,
        "noise80_first_interpolation": interpolated_rows,
        "noise80_final": noise_final_rows,
        "final_all_conditions": final_rows,
    }
    write_json(architecture_dir / "summary.json", architecture_summary)
    return {
        "history": history,
        "milestones": milestone_rows,
        "summary": architecture_summary,
    }


def grouped_curve(
    rows: list[dict[str, Any]],
    condition: str,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = [row for row in rows if row["condition"] == condition]
    epochs = sorted({int(row["epoch"]) for row in subset})
    means: list[float] = []
    minima: list[float] = []
    maxima: list[float] = []
    for epoch in epochs:
        values = np.asarray(
            [
                float(row[field_name])
                for row in subset
                if int(row["epoch"]) == epoch
            ],
            dtype=np.float64,
        )
        means.append(float(np.mean(values)))
        minima.append(float(np.min(values)))
        maxima.append(float(np.max(values)))
    return (
        np.asarray(epochs),
        np.asarray(means),
        np.asarray(minima),
        np.asarray(maxima),
    )


def plot_architecture(
    result_dir: Path,
    architecture: str,
    history: list[dict[str, Any]],
    threshold: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", flush=True)
        return

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"clean": "#157f68", "noise80": "#c6493d"}

    for condition in ("clean", "noise80"):
        epochs, mean, minimum, maximum = grouped_curve(
            history,
            condition,
            "test_true_accuracy",
        )
        axes[0, 0].plot(
            epochs,
            100.0 * mean,
            label=condition,
            color=colors[condition],
            linewidth=2,
        )
        axes[0, 0].fill_between(
            epochs,
            100.0 * minimum,
            100.0 * maximum,
            color=colors[condition],
            alpha=0.16,
        )

        epochs, mean, minimum, maximum = grouped_curve(
            history,
            condition,
            "train_observed_accuracy",
        )
        axes[0, 1].plot(
            epochs,
            100.0 * mean,
            label=condition,
            color=colors[condition],
            linewidth=2,
        )
        axes[0, 1].fill_between(
            epochs,
            100.0 * minimum,
            100.0 * maximum,
            color=colors[condition],
            alpha=0.16,
        )

        epochs, mean, minimum, maximum = grouped_curve(
            history,
            condition,
            "train_observed_loss",
        )
        axes[1, 0].plot(
            epochs,
            mean,
            label=condition,
            color=colors[condition],
            linewidth=2,
        )
        axes[1, 0].fill_between(
            epochs,
            np.maximum(minimum, 1e-8),
            np.maximum(maximum, 1e-8),
            color=colors[condition],
            alpha=0.16,
        )

    for field_name, label, color in (
        ("train_true_accuracy", "noise80: train vs truth", "#2667a8"),
        ("train_noisy_accuracy", "noise80: train vs noisy labels", "#c6493d"),
        ("test_true_accuracy", "noise80: clean test", "#6d4c9a"),
        ("paired_clean_test_agreement", "noise80 vs clean pair agreement", "#6b7280"),
    ):
        epochs, mean, minimum, maximum = grouped_curve(
            history,
            "noise80",
            field_name,
        )
        axes[1, 1].plot(
            epochs,
            100.0 * mean,
            label=label,
            color=color,
            linewidth=2,
        )
        axes[1, 1].fill_between(
            epochs,
            100.0 * minimum,
            100.0 * maximum,
            color=color,
            alpha=0.10,
        )

    axes[0, 0].set_title("Clean test accuracy")
    axes[0, 0].set_ylabel("accuracy (%)")
    axes[0, 1].set_title("Observed-label training accuracy")
    axes[0, 1].axhline(
        100.0 * threshold,
        color="#111827",
        linestyle="--",
        linewidth=1,
        label="interpolation threshold",
    )
    axes[0, 1].set_ylabel("accuracy (%)")
    axes[1, 0].set_title("Observed-label training cross entropy")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("cross entropy")
    axes[1, 1].set_title("Noise condition: rule retention vs memorization")
    axes[1, 1].set_ylabel("accuracy / agreement (%)")

    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(
        f"MNIST fixed 80% hidden label noise: {architecture}",
        fontsize=14,
    )
    output_path = result_dir / architecture / "training_dynamics.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def create_archive(result_dir: Path, include_weights: bool) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if not path.is_file():
                continue
            if not include_weights and path.suffix in {".pt", ".pth"}:
                continue
            archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def main() -> None:
    cfg = apply_smoke_overrides(Config())
    validate_config(cfg)
    result_dir = cfg.RESULT_DIR.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE 要求 CUDA，但当前 PyTorch 看不到 GPU。")

    set_global_seed(20260820)
    if cfg.ALLOW_TF32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("=== MNIST 隐藏 80% 标签噪声长期训练实验 ===", flush=True)
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(f"结果目录：{result_dir}", flush=True)
    print(
        f"训练/验证={cfg.TRAIN_SIZE:,}/{cfg.VALIDATION_SIZE:,} | "
        f"固定噪声={100.0 * cfg.CORRUPTION_RATE:.1f}% | "
        f"seeds={list(cfg.MODEL_SEEDS)} | batch={cfg.BATCH_SIZE}",
        flush=True,
    )
    print(
        "判决标准：只有 noisy_train_accuracy 达到 "
        f"{100.0 * cfg.INTERPOLATION_ACCURACY:.3f}% 后的 test accuracy "
        "才属于强版本；best validation 单独报告。",
        flush=True,
    )

    dataset = load_dataset(cfg, device)
    print(
        "数据完成："
        f"corrupted={dataset.metadata['corruption_count']:,}/"
        f"{dataset.metadata['train_size']:,} | "
        f"test={dataset.metadata['test_size']:,}",
        flush=True,
    )

    config_payload = asdict(cfg)
    config_payload.pop("_SMOKE_RESULT_DIR", None)
    config_payload.pop("_SMOKE_DATA_DIR", None)
    runtime_payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(result_dir / "config.json", config_payload)
    write_json(result_dir / "dataset_metadata.json", dataset.metadata)
    write_json(result_dir / "runtime.json", runtime_payload)

    all_history: list[dict[str, Any]] = []
    architecture_summaries: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for spec in architecture_specs(cfg):
        result = train_architecture(
            cfg,
            spec,
            dataset,
            device,
            result_dir,
        )
        all_history.extend(result["history"])
        architecture_summaries.append(result["summary"])
        if cfg.GENERATE_PLOTS:
            plot_architecture(
                result_dir,
                spec.name,
                result["history"],
                cfg.INTERPOLATION_ACCURACY,
            )

    write_csv(result_dir / "all_trajectories.csv", all_history)
    overall_summary = {
        "status": "ok",
        "elapsed_seconds": time.perf_counter() - overall_start,
        "dataset": dataset.metadata,
        "architectures": architecture_summaries,
        "interpretation_rule": (
            "best_validation 不构成强版本证据；只有 first_interpolation/final "
            "在 noisy labels 近乎完全拟合后仍保持高 clean test accuracy 才构成。"
        ),
    }
    write_json(result_dir / "summary.json", overall_summary)

    archive_path: Path | None = None
    if cfg.CREATE_ARCHIVE:
        archive_path = create_archive(
            result_dir,
            include_weights=cfg.INCLUDE_WEIGHTS_IN_ARCHIVE,
        )

    print("\n=== 实验完成 ===", flush=True)
    for summary in architecture_summaries:
        architecture = summary["architecture"]["name"]
        print(
            f"{architecture}: interpolated="
            f"{summary['interpolated_seed_count']}/"
            f"{summary['requested_seed_count']} | "
            f"strong_at_first_interpolation="
            f"{summary['strong_version_supported_at_first_interpolation']} | "
            f"strong_at_final={summary['strong_version_supported_at_final']}",
            flush=True,
        )
    print(f"汇总：{result_dir / 'summary.json'}", flush=True)
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}", flush=True)


if __name__ == "__main__":
    main()
