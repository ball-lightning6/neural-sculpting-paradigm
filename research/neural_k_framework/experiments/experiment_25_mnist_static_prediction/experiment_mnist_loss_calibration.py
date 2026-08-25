"""MNIST逐样本体积预测的Stage 0：训练样本量与过拟合loss校准。

目标不是直接用SGD预测测试标签，而是先回答静态体积分支实验必须预先冻结的
两个问题：

1. 对当前二分类任务和小网络，多少训练样本已经足以形成可用泛化？
2. 继续降低训练集raw BCE时，validation loss从哪个训练loss附近开始恶化？

脚本使用同一Gaussian神经参考语言，扫描两个MNIST二分类任务、多个平衡训练
样本量、多份随机训练集和多个配对初始化。所有模型采用全批量AdamW、无
weight decay并长训，记录train/validation/test loss、accuracy和多seed函数
agreement。输出的 ``calibration_recommendations.json`` 只给下一阶段候选
``n`` 和 ``epsilon``，必须在查看结果后再冻结；本脚本不运行静态SMC。

下一阶段将在固定D和候选测试图像x下比较：

    V(D + (x, 0); epsilon)  vs  V(D + (x, 1); epsilon)

并预测剩余参数质量更大、自由能增量更小的标签。
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import random
import shutil
import struct
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    DATA_DIR = Path("/root/mnist_dataset")
    RESULT_DIR = Path("/root/results_mnist_loss_calibration")
    DOWNLOAD_MNIST_IF_MISSING = True

    # 0/1提供容易任务，3/8提供较难且形状相近的任务。
    DIGIT_PAIRS = ((0, 1), (3, 8))
    IMAGE_SIZE = 7
    WIDTH = 32
    TRAIN_COUNTS = (4, 8, 16, 32, 64, 128, 256, 512)
    DATASETS_PER_N = 4
    MODEL_SEEDS = tuple(range(8))
    VALIDATION_PER_CLASS = 384
    TEST_PER_CLASS = 384

    SPLIT_SEED = 2026082601
    DATASET_SEED = 2026082602
    INITIALIZATION_SEED = 2026082603

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 15_000
    EVAL_STEPS = (
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 12_500, 15_000,
    )
    LOG_EVERY_EVAL = True

    # 网络实际权重由标准Gaussian坐标乘以下列固定尺度得到。Stage 1 SMC必须
    # 使用完全相同的参数化和先验。
    FIRST_BIAS_SCALE = 0.10
    OUTPUT_BIAS_SCALE = 0.10

    TRAIN_INTERPOLATION_ACCURACY = 0.999
    OVERFIT_RELATIVE_RISE = 0.10
    OVERFIT_ABSOLUTE_RISE = 0.01
    OVERFIT_PERSISTENCE_EVALS = 2
    SUFFICIENT_BEST_VAL_ACCURACY = 0.95

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    GENERATE_PLOTS = True
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class PairSpec:
    pair_index: int
    negative_digit: int
    positive_digit: int

    @property
    def name(self) -> str:
        return f"{self.negative_digit}_vs_{self.positive_digit}"


@dataclass(frozen=True)
class Condition:
    model_index: int
    pair_index: int
    pair_name: str
    n: int
    dataset_index: int
    model_seed: int


@dataclass
class PairData:
    spec: PairSpec
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    train_plans: list[tuple[torch.Tensor, torch.Tensor]]
    metadata: dict[str, Any]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.DATA_DIR = Path(
        "research/overfitting_related_research/_smoke_mnist_dataset"
    )
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_mnist_loss_calibration"
    )
    Config.DOWNLOAD_MNIST_IF_MISSING = False
    Config.DIGIT_PAIRS = ((0, 1),)
    Config.IMAGE_SIZE = 7
    Config.WIDTH = 8
    Config.TRAIN_COUNTS = (4, 8)
    Config.DATASETS_PER_N = 1
    Config.MODEL_SEEDS = (0, 1)
    Config.VALIDATION_PER_CLASS = 16
    Config.TEST_PER_CLASS = 16
    Config.MAX_STEPS = 5
    Config.EVAL_STEPS = (0, 1, 2, 5)
    Config.DEVICE = "cpu"
    Config.GENERATE_PLOTS = True
    Config.PACKAGE_RESULTS = False
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True


def validate_config() -> None:
    if not Config.DIGIT_PAIRS:
        raise ValueError("DIGIT_PAIRS不能为空。")
    if any(a == b for a, b in Config.DIGIT_PAIRS):
        raise ValueError("每个二分类任务必须使用两个不同数字。")
    if any(n <= 0 or n % 2 for n in Config.TRAIN_COUNTS):
        raise ValueError("TRAIN_COUNTS必须全部为正偶数。")
    if tuple(sorted(set(Config.TRAIN_COUNTS))) != Config.TRAIN_COUNTS:
        raise ValueError("TRAIN_COUNTS必须严格递增且不重复。")
    if Config.EVAL_STEPS[-1] != Config.MAX_STEPS:
        raise ValueError("EVAL_STEPS最后一项必须等于MAX_STEPS。")
    if not set(Config.MODEL_SEEDS):
        raise ValueError("至少需要一个MODEL_SEEDS。")
    if Config.DATASETS_PER_N < 1:
        raise ValueError("DATASETS_PER_N必须为正数。")
    if not 0.0 < Config.SUFFICIENT_BEST_VAL_ACCURACY <= 1.0:
        raise ValueError("SUFFICIENT_BEST_VAL_ACCURACY必须位于(0,1]。")


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


def config_payload() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        magic = handle.read(4)
        if len(magic) != 4 or magic[0:2] != b"\x00\x00":
            raise RuntimeError(f"非法IDX文件：{path}")
        dtype_code = magic[2]
        dimensions = magic[3]
        if dtype_code != 0x08:
            raise RuntimeError(f"当前只支持uint8 IDX：{path}")
        shape = tuple(
            struct.unpack(">I", handle.read(4))[0]
            for _ in range(dimensions)
        )
        payload = handle.read()
    values = np.frombuffer(payload, dtype=np.uint8)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(
            f"IDX长度不符：{path}，期望{expected}，实际{values.size}。"
        )
    return values.reshape(shape).copy()


def raw_mnist_paths(root: Path, train: bool) -> tuple[Path, Path] | None:
    image_name = (
        "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    )
    label_name = (
        "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
    )
    # 用户直接放在root下的文件优先；这样即使torchvision留下了半截下载文件，
    # 也不会遮蔽已经完整存在的官方.gz。
    roots = (root, root / "MNIST" / "raw")
    for candidate_root in roots:
        for suffix in ("", ".gz"):
            image_path = candidate_root / f"{image_name}{suffix}"
            label_path = candidate_root / f"{label_name}{suffix}"
            if image_path.exists() and label_path.exists():
                return image_path, label_path
    return None


def load_mnist_arrays() -> tuple[torch.Tensor, ...]:
    # 先检查现有IDX文件。torchvision只认root/MNIST/raw的标准布局；用户常把
    # 四个官方.gz直接放在root下，若先调用download=True会无谓地重新下载。
    train_paths = raw_mnist_paths(Config.DATA_DIR, train=True)
    test_paths = raw_mnist_paths(Config.DATA_DIR, train=False)
    if train_paths is not None and test_paths is not None:
        train_images = torch.from_numpy(read_idx(train_paths[0]))
        train_labels = torch.from_numpy(read_idx(train_paths[1])).long()
        test_images = torch.from_numpy(read_idx(test_paths[0]))
        test_labels = torch.from_numpy(read_idx(test_paths[1])).long()
        print(
            "使用现有MNIST IDX文件："
            f"{train_paths[0].parent} / {test_paths[0].parent}",
            flush=True,
        )
        return train_images, train_labels, test_images, test_labels

    torchvision_error: Exception | None = None
    try:
        from torchvision.datasets import MNIST

        train_set = MNIST(
            root=str(Config.DATA_DIR), train=True,
            download=Config.DOWNLOAD_MNIST_IF_MISSING,
        )
        test_set = MNIST(
            root=str(Config.DATA_DIR), train=False,
            download=Config.DOWNLOAD_MNIST_IF_MISSING,
        )
        return (
            train_set.data.clone(), train_set.targets.clone(),
            test_set.data.clone(), test_set.targets.clone(),
        )
    except Exception as exc:  # torchvision二进制不匹配时允许IDX回退。
        torchvision_error = exc

    train_paths = raw_mnist_paths(Config.DATA_DIR, train=True)
    test_paths = raw_mnist_paths(Config.DATA_DIR, train=False)
    if train_paths is None or test_paths is None:
        raise RuntimeError(
            "无法通过torchvision或本地IDX文件加载MNIST。"
            f" torchvision错误={torchvision_error!r}"
        )
    train_images = torch.from_numpy(read_idx(train_paths[0]))
    train_labels = torch.from_numpy(read_idx(train_paths[1])).long()
    test_images = torch.from_numpy(read_idx(test_paths[0]))
    test_labels = torch.from_numpy(read_idx(test_paths[1])).long()
    return train_images, train_labels, test_images, test_labels


def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    values = images.to(dtype=torch.float32).unsqueeze(1).div_(255.0)
    if tuple(values.shape[-2:]) != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
        values = F.adaptive_avg_pool2d(
            values, (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        )
    # 固定映射到[-1,1]，不利用测试标签或测试统计量。
    return values.mul_(2.0).sub_(1.0).flatten(1).contiguous()


def shuffled_indices(
    labels: torch.Tensor,
    digit: int,
    seed: int,
) -> torch.Tensor:
    indices = torch.nonzero(labels == digit, as_tuple=False).flatten()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return indices[torch.randperm(len(indices), generator=generator)]


def build_pair_data(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[list[PairData], dict[str, Any]]:
    train_x_all = preprocess_images(train_images)
    test_x_all = preprocess_images(test_images)
    max_per_class = max(Config.TRAIN_COUNTS) // 2
    pairs: list[PairData] = []

    for pair_index, (negative, positive) in enumerate(Config.DIGIT_PAIRS):
        spec = PairSpec(pair_index, int(negative), int(positive))
        train_orders: dict[int, torch.Tensor] = {}
        test_orders: dict[int, torch.Tensor] = {}
        for offset, digit in enumerate((negative, positive)):
            train_orders[digit] = shuffled_indices(
                train_labels, digit,
                Config.SPLIT_SEED + pair_index * 10_000 + offset,
            )
            test_orders[digit] = shuffled_indices(
                test_labels, digit,
                Config.SPLIT_SEED + 1_000_000 + pair_index * 10_000 + offset,
            )

        validation_indices: list[torch.Tensor] = []
        validation_targets: list[torch.Tensor] = []
        test_indices: list[torch.Tensor] = []
        test_targets: list[torch.Tensor] = []
        candidate_indices: dict[int, torch.Tensor] = {}
        for binary_label, digit in enumerate((negative, positive)):
            order = train_orders[digit]
            validation = order[:Config.VALIDATION_PER_CLASS]
            candidate = order[Config.VALIDATION_PER_CLASS:]
            if len(candidate) < max_per_class:
                raise RuntimeError(
                    f"数字{digit}扣除validation后不足{max_per_class}张。"
                )
            validation_indices.append(validation)
            validation_targets.append(torch.full(
                (len(validation),), binary_label, dtype=torch.float32
            ))
            candidate_indices[digit] = candidate

            selected_test = test_orders[digit][:Config.TEST_PER_CLASS]
            test_indices.append(selected_test)
            test_targets.append(torch.full(
                (len(selected_test),), binary_label, dtype=torch.float32
            ))

        validation_index = torch.cat(validation_indices)
        validation_y = torch.cat(validation_targets)
        test_index = torch.cat(test_indices)
        pair_test_y = torch.cat(test_targets)

        train_plans: list[tuple[torch.Tensor, torch.Tensor]] = []
        plan_metadata: list[dict[str, Any]] = []
        for dataset_index in range(Config.DATASETS_PER_N):
            selected_by_class: list[torch.Tensor] = []
            for offset, digit in enumerate((negative, positive)):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    Config.DATASET_SEED
                    + pair_index * 1_000_003
                    + dataset_index * 10_007
                    + offset
                )
                pool = candidate_indices[digit]
                local = torch.randperm(len(pool), generator=generator)
                selected_by_class.append(pool[local[:max_per_class]])
            train_plans.append((selected_by_class[0], selected_by_class[1]))
            plan_metadata.append({
                "dataset_index": dataset_index,
                "negative_original_indices": selected_by_class[0].tolist(),
                "positive_original_indices": selected_by_class[1].tolist(),
            })

        pairs.append(PairData(
            spec=spec,
            validation_x=train_x_all[validation_index],
            validation_y=validation_y,
            test_x=test_x_all[test_index],
            test_y=pair_test_y,
            train_plans=train_plans,
            metadata={
                "pair": asdict(spec),
                "name": spec.name,
                "validation_original_indices": validation_index.tolist(),
                "test_original_indices": test_index.tolist(),
                "train_plans": plan_metadata,
            },
        ))

    metadata = {
        "preprocessing": {
            "source_shape": [28, 28],
            "target_shape": [Config.IMAGE_SIZE, Config.IMAGE_SIZE],
            "method": "adaptive average pooling then x*2-1",
        },
        "pairs": [pair.metadata for pair in pairs],
    }
    return pairs, metadata


class BatchedTinyMLP(nn.Module):
    def __init__(self, conditions: Sequence[Condition], input_dim: int):
        super().__init__()
        model_count = len(conditions)
        width = Config.WIDTH
        seed_cache: dict[int, tuple[torch.Tensor, ...]] = {}
        for condition in conditions:
            if condition.model_seed in seed_cache:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                Config.INITIALIZATION_SEED + int(condition.model_seed)
            )
            seed_cache[condition.model_seed] = (
                torch.randn(width, input_dim, generator=generator),
                torch.randn(width, generator=generator),
                torch.randn(1, width, generator=generator),
                torch.randn(1, generator=generator),
            )

        first_weight = torch.empty(model_count, width, input_dim)
        first_bias = torch.empty(model_count, width)
        output_weight = torch.empty(model_count, 1, width)
        output_bias = torch.empty(model_count, 1)
        for model_index, condition in enumerate(conditions):
            tensors = seed_cache[condition.model_seed]
            first_weight[model_index] = tensors[0]
            first_bias[model_index] = tensors[1]
            output_weight[model_index] = tensors[2]
            output_bias[model_index] = tensors[3]

        self.first_weight = nn.Parameter(first_weight)
        self.first_bias = nn.Parameter(first_bias)
        self.output_weight = nn.Parameter(output_weight)
        self.output_bias = nn.Parameter(output_bias)
        self.input_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first_weight = self.first_weight * (1.0 / math.sqrt(self.input_dim))
        hidden = torch.tanh(
            torch.bmm(inputs, first_weight.transpose(1, 2))
            + Config.FIRST_BIAS_SCALE * self.first_bias[:, None]
        )
        output_weight = self.output_weight * (
            1.0 / math.sqrt(Config.WIDTH)
        )
        return (
            torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
            + Config.OUTPUT_BIAS_SCALE * self.output_bias
        )


def individual_parameter_count(input_dim: int) -> int:
    return (
        Config.WIDTH * input_dim + Config.WIDTH
        + Config.WIDTH + 1
    )


def condition_plan(
    pairs: Sequence[PairData], n: int
) -> tuple[list[Condition], torch.Tensor, torch.Tensor]:
    conditions: list[Condition] = []
    train_x: list[torch.Tensor] = []
    train_y: list[torch.Tensor] = []
    model_index = 0
    half = n // 2
    for pair in pairs:
        for dataset_index, (negative_indices, positive_indices) in enumerate(
            pair.train_plans
        ):
            # 原始索引已经在metadata保存；这里从预处理后的完整train数组取值
            # 会需要PairData额外持有，故在build后动态挂载的缓存中读取。
            local_indices = torch.cat((
                negative_indices[:half], positive_indices[:half]
            ))
            local_targets = torch.cat((
                torch.zeros(half), torch.ones(half)
            )).float()
            local_inputs = TRAIN_INPUT_CACHE[local_indices]
            for model_seed in Config.MODEL_SEEDS:
                conditions.append(Condition(
                    model_index=model_index,
                    pair_index=pair.spec.pair_index,
                    pair_name=pair.spec.name,
                    n=n,
                    dataset_index=dataset_index,
                    model_seed=int(model_seed),
                ))
                train_x.append(local_inputs)
                train_y.append(local_targets)
                model_index += 1
    return conditions, torch.stack(train_x), torch.stack(train_y)


# build_pair_data完成后填充。使用模块级只读缓存避免在每个PairData复制60k图像。
TRAIN_INPUT_CACHE: torch.Tensor


@torch.no_grad()
def evaluate_split(
    model: BatchedTinyMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    micro_batch: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_count, sample_count = targets.shape
    losses = torch.zeros(model_count, device=inputs.device)
    correct = torch.zeros(model_count, device=inputs.device)
    prediction_parts: list[torch.Tensor] = []
    for start in range(0, sample_count, micro_batch):
        stop = min(sample_count, start + micro_batch)
        logits = model(inputs[:, start:stop])
        local_targets = targets[:, start:stop]
        losses += F.binary_cross_entropy_with_logits(
            logits, local_targets, reduction="none"
        ).sum(dim=1)
        predictions = logits >= 0
        correct += (predictions == (local_targets >= 0.5)).sum(dim=1)
        prediction_parts.append(predictions.cpu())
    return (
        (losses / sample_count).cpu().numpy(),
        (correct / sample_count).cpu().numpy(),
        torch.cat(prediction_parts, dim=1).numpy(),
    )


def build_eval_batch(
    pairs: Sequence[PairData],
    conditions: Sequence[Condition],
    split: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for condition in conditions:
        pair = pairs[condition.pair_index]
        if split == "validation":
            inputs.append(pair.validation_x)
            targets.append(pair.validation_y)
        elif split == "test":
            inputs.append(pair.test_x)
            targets.append(pair.test_y)
        else:
            raise ValueError(f"未知split：{split}")
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def pairwise_point_agreement(predictions: np.ndarray) -> float:
    seed_count, sample_count = predictions.shape
    if seed_count < 2 or sample_count == 0:
        return float("nan")
    ones = predictions.sum(axis=0, dtype=np.int64)
    zeros = seed_count - ones
    same = ones * (ones - 1) + zeros * (zeros - 1)
    return float(np.mean(same / (seed_count * (seed_count - 1))))


def exact_function_collision(predictions: np.ndarray) -> float:
    seed_count = len(predictions)
    if seed_count < 2:
        return float("nan")
    packed = np.packbits(predictions.astype(np.uint8), axis=1, bitorder="little")
    _, counts = np.unique(packed, axis=0, return_counts=True)
    return float(
        np.sum(counts * (counts - 1)) / (seed_count * (seed_count - 1))
    )


def modal_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    modal = predictions.mean(axis=0) >= 0.5
    return float(np.mean(modal == (targets >= 0.5)))


def quantile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, q)) if len(array) else float("nan")


def evaluate_group(
    step: int,
    model: BatchedTinyMLP,
    conditions: Sequence[Condition],
    pairs: Sequence[PairData],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_loss, train_acc, _ = evaluate_split(model, train_x, train_y)
    val_loss, val_acc, val_predictions = evaluate_split(
        model, validation_x, validation_y
    )
    test_loss, test_acc, test_predictions = evaluate_split(
        model, test_x, test_y
    )
    model_rows: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        model_rows.append({
            **asdict(condition),
            "step": step,
            "train_loss": float(train_loss[index]),
            "train_accuracy": float(train_acc[index]),
            "validation_loss": float(val_loss[index]),
            "validation_accuracy": float(val_acc[index]),
            "test_loss": float(test_loss[index]),
            "test_accuracy": float(test_acc[index]),
        })

    agreement_rows: list[dict[str, Any]] = []
    key_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, condition in enumerate(conditions):
        key_to_indices.setdefault(
            (condition.pair_index, condition.dataset_index), []
        ).append(index)
    for (pair_index, dataset_index), indices in key_to_indices.items():
        pair = pairs[pair_index]
        local_val = val_predictions[indices]
        local_test = test_predictions[indices]
        agreement_rows.append({
            "pair_index": pair_index,
            "pair_name": pair.spec.name,
            "n": conditions[indices[0]].n,
            "dataset_index": dataset_index,
            "step": step,
            "validation_point_agreement": pairwise_point_agreement(local_val),
            "validation_function_collision": exact_function_collision(local_val),
            "validation_modal_accuracy": modal_accuracy(
                local_val, pair.validation_y.numpy()
            ),
            "validation_exact_target_mass": float(np.mean(
                np.all(
                    local_val == (pair.validation_y.numpy()[None] >= 0.5),
                    axis=1,
                )
            )),
            "test_point_agreement": pairwise_point_agreement(local_test),
            "test_function_collision": exact_function_collision(local_test),
            "test_modal_accuracy": modal_accuracy(
                local_test, pair.test_y.numpy()
            ),
            "test_exact_target_mass": float(np.mean(np.all(
                local_test == (pair.test_y.numpy()[None] >= 0.5), axis=1
            ))),
        })
    return model_rows, agreement_rows


def run_n_group(
    n: int,
    pairs: Sequence[PairData],
    device: torch.device,
    start_time: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions, train_x_cpu, train_y_cpu = condition_plan(pairs, n)
    train_x = train_x_cpu.to(device)
    train_y = train_y_cpu.to(device)
    validation_x, validation_y = build_eval_batch(
        pairs, conditions, "validation", device
    )
    test_x, test_y = build_eval_batch(pairs, conditions, "test", device)
    model = BatchedTinyMLP(
        conditions, Config.IMAGE_SIZE * Config.IMAGE_SIZE
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    model_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    eval_set = set(Config.EVAL_STEPS)

    for step in range(Config.MAX_STEPS + 1):
        if step in eval_set:
            model.eval()
            new_models, new_agreement = evaluate_group(
                step, model, conditions, pairs,
                train_x, train_y,
                validation_x, validation_y,
                test_x, test_y,
            )
            model_rows.extend(new_models)
            agreement_rows.extend(new_agreement)
            if Config.LOG_EVERY_EVAL:
                train_med = np.median([
                    row["train_loss"] for row in new_models
                ])
                val_med = np.median([
                    row["validation_loss"] for row in new_models
                ])
                test_acc_med = np.median([
                    row["test_accuracy"] for row in new_models
                ])
                print(
                    f"n={n:4d} step={step:6d} | "
                    f"train/val BCE={train_med:.5g}/{val_med:.5g} | "
                    f"test acc={100*test_acc_med:.2f}% | "
                    f"elapsed={time.time()-start_time:.1f}s",
                    flush=True,
                )
        if step == Config.MAX_STEPS:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        per_model_loss = F.binary_cross_entropy_with_logits(
            logits, train_y, reduction="none"
        ).mean(dim=1)
        # 参数张量的首维对应独立模型；sum使每个切片获得与单模型相同的梯度。
        per_model_loss.sum().backward()
        optimizer.step()

    del model, optimizer, train_x, train_y
    del validation_x, validation_y, test_x, test_y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model_rows, agreement_rows


def detect_milestones(
    model_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in model_rows:
        key = (
            str(row["pair_name"]), int(row["n"]),
            int(row["dataset_index"]), int(row["model_seed"]),
        )
        grouped.setdefault(key, []).append(row)

    milestones: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["step"]))
        best_index = min(
            range(len(rows)), key=lambda index: float(rows[index]["validation_loss"])
        )
        best = rows[best_index]
        first_fit = next(
            (
                row for row in rows
                if float(row["train_accuracy"])
                >= Config.TRAIN_INTERPOLATION_ACCURACY
            ),
            None,
        )
        rise = max(
            Config.OVERFIT_ABSOLUTE_RISE,
            Config.OVERFIT_RELATIVE_RISE * float(best["validation_loss"]),
        )
        onset = None
        persistence = Config.OVERFIT_PERSISTENCE_EVALS
        for index in range(best_index + 1, len(rows)):
            window = rows[index:index + persistence]
            if len(window) < persistence:
                break
            if all(
                float(item["validation_loss"])
                >= float(best["validation_loss"]) + rise
                for item in window
            ) and float(rows[index]["train_loss"]) < float(best["train_loss"]):
                onset = rows[index]
                break

        milestone = {
            "pair_name": key[0],
            "n": key[1],
            "dataset_index": key[2],
            "model_seed": key[3],
            "best_validation_step": int(best["step"]),
            "best_validation_loss": float(best["validation_loss"]),
            "best_validation_accuracy": float(best["validation_accuracy"]),
            "test_accuracy_at_best_validation": float(best["test_accuracy"]),
            "train_loss_at_best_validation": float(best["train_loss"]),
            "first_interpolation_step": (
                int(first_fit["step"]) if first_fit is not None else None
            ),
            "train_loss_at_first_interpolation": (
                float(first_fit["train_loss"]) if first_fit is not None else None
            ),
            "overfit_detected": onset is not None,
            "overfit_onset_step": int(onset["step"]) if onset else None,
            "train_loss_at_overfit_onset": (
                float(onset["train_loss"]) if onset else None
            ),
            "validation_loss_at_overfit_onset": (
                float(onset["validation_loss"]) if onset else None
            ),
            "final_train_loss": float(rows[-1]["train_loss"]),
            "final_validation_loss": float(rows[-1]["validation_loss"]),
            "final_validation_accuracy": float(rows[-1]["validation_accuracy"]),
            "final_test_accuracy": float(rows[-1]["test_accuracy"]),
        }
        milestones.append(milestone)

    summary_rows: list[dict[str, Any]] = []
    pair_n_keys = sorted({
        (str(row["pair_name"]), int(row["n"])) for row in milestones
    })
    for pair_name, n in pair_n_keys:
        local = [
            row for row in milestones
            if row["pair_name"] == pair_name and row["n"] == n
        ]
        onset_losses = [
            float(row["train_loss_at_overfit_onset"])
            for row in local if row["train_loss_at_overfit_onset"] is not None
        ]
        best_acc = [float(row["best_validation_accuracy"]) for row in local]
        test_best = [
            float(row["test_accuracy_at_best_validation"]) for row in local
        ]
        train_at_best = [
            float(row["train_loss_at_best_validation"]) for row in local
        ]
        summary_rows.append({
            "pair_name": pair_name,
            "n": n,
            "run_count": len(local),
            "best_validation_accuracy_median": quantile(best_acc, 0.5),
            "best_validation_accuracy_q10": quantile(best_acc, 0.1),
            "best_validation_accuracy_q90": quantile(best_acc, 0.9),
            "test_accuracy_at_best_median": quantile(test_best, 0.5),
            "train_loss_at_best_median": quantile(train_at_best, 0.5),
            "train_loss_at_best_q10": quantile(train_at_best, 0.1),
            "train_loss_at_best_q90": quantile(train_at_best, 0.9),
            "overfit_fraction": float(np.mean([
                bool(row["overfit_detected"]) for row in local
            ])),
            "overfit_onset_train_loss_median": quantile(onset_losses, 0.5),
            "overfit_onset_train_loss_q10": quantile(onset_losses, 0.1),
            "overfit_onset_train_loss_q90": quantile(onset_losses, 0.9),
            "sufficient_by_accuracy": (
                quantile(best_acc, 0.5)
                >= Config.SUFFICIENT_BEST_VAL_ACCURACY
            ),
        })
    return milestones, summary_rows


def build_recommendations(
    summary_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {
        "status": "candidate_only_requires_manual_freeze",
        "criterion": {
            "median_best_validation_accuracy": (
                Config.SUFFICIENT_BEST_VAL_ACCURACY
            ),
            "overfit_definition": {
                "relative_validation_loss_rise": Config.OVERFIT_RELATIVE_RISE,
                "absolute_validation_loss_rise": Config.OVERFIT_ABSOLUTE_RISE,
                "persistence_evaluations": Config.OVERFIT_PERSISTENCE_EVALS,
            },
        },
        "pairs": {},
        "next_stage_note": (
            "查看完整曲线后，为每个pair人工冻结一个n和若干raw-BCE截面；"
            "不要只凭本文件自动开始SMC。"
        ),
    }
    for pair_name in sorted({str(row["pair_name"]) for row in summary_rows}):
        local = sorted(
            [row for row in summary_rows if row["pair_name"] == pair_name],
            key=lambda row: int(row["n"]),
        )
        sufficient = [row for row in local if row["sufficient_by_accuracy"]]
        chosen = sufficient[0] if sufficient else local[-1]
        recommendations["pairs"][pair_name] = {
            "minimal_sufficient_n_candidate": int(chosen["n"]),
            "train_loss_at_best_validation_candidate": (
                chosen["train_loss_at_best_median"]
            ),
            "overfit_onset_train_loss_candidate": (
                chosen["overfit_onset_train_loss_median"]
            ),
            "sufficient_threshold_reached": bool(
                chosen["sufficient_by_accuracy"]
            ),
            "all_n_rows": local,
        }
    return recommendations


def generate_plots(
    result_dir: Path,
    model_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
) -> None:
    if not Config.GENERATE_PLOTS:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("缺少matplotlib，跳过绘图。", flush=True)
        return

    pair_names = sorted({str(row["pair_name"]) for row in model_rows})
    for pair_name in pair_names:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        local = [row for row in model_rows if row["pair_name"] == pair_name]
        for n in sorted({int(row["n"]) for row in local}):
            by_step: dict[int, list[dict[str, Any]]] = {}
            for row in local:
                if int(row["n"]) == n:
                    by_step.setdefault(int(row["step"]), []).append(row)
            steps = sorted(by_step)
            train = [
                np.median([float(x["train_loss"]) for x in by_step[step]])
                for step in steps
            ]
            validation = [
                np.median([
                    float(x["validation_loss"]) for x in by_step[step]
                ])
                for step in steps
            ]
            axes[0].plot(steps, train, label=f"train n={n}")
            axes[0].plot(steps, validation, linestyle="--", label=f"val n={n}")
        axes[0].set_xscale("symlog", linthresh=10)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("raw BCE")
        axes[0].set_title(f"{pair_name}: train/validation loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=7, ncol=2)

        summary = sorted(
            [row for row in summary_rows if row["pair_name"] == pair_name],
            key=lambda row: int(row["n"]),
        )
        ns = [int(row["n"]) for row in summary]
        axes[1].plot(ns, [
            float(row["best_validation_accuracy_median"]) for row in summary
        ], marker="o", label="best validation")
        axes[1].plot(ns, [
            float(row["test_accuracy_at_best_median"]) for row in summary
        ], marker="s", label="test at val-best")
        axes[1].axhline(
            Config.SUFFICIENT_BEST_VAL_ACCURACY,
            color="black", linestyle=":", label="sufficient threshold",
        )
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylim(0.45, 1.01)
        axes[1].set_xlabel("balanced training samples n")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title(f"{pair_name}: sample sufficiency")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(result_dir / f"{pair_name}_calibration.png", dpi=170)
        plt.close(figure)


def package_results(result_dir: Path) -> Path:
    archive = result_dir.parent / f"{result_dir.name}_package.zip"
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.name not in {"latest_checkpoint.pt"}:
                handle.write(path, path.relative_to(result_dir.parent))
    temp.replace(archive)
    return archive


def prepare_result_dir() -> Path:
    result_dir = Config.RESULT_DIR
    if result_dir.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def main() -> None:
    global TRAIN_INPUT_CACHE
    apply_smoke_overrides()
    validate_config()
    set_global_seed(Config.INITIALIZATION_SEED)
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前不可用。")
    device = torch.device(Config.DEVICE)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32

    result_dir = prepare_result_dir()
    start_time = time.time()
    print("=== MNIST loss calibration for free-energy prediction ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"pairs={Config.DIGIT_PAIRS} | n={Config.TRAIN_COUNTS} | "
        f"datasets/n={Config.DATASETS_PER_N} | seeds={Config.MODEL_SEEDS}",
        flush=True,
    )
    print(
        f"MLP={Config.IMAGE_SIZE**2}->{Config.WIDTH}->1 tanh | "
        f"params/model={individual_parameter_count(Config.IMAGE_SIZE**2):,} | "
        f"max_steps={Config.MAX_STEPS:,}",
        flush=True,
    )
    print(f"result_dir={result_dir}", flush=True)

    # 为condition_plan建立一份只读预处理缓存，避免在每个PairData复制60k图像。
    train_images, train_labels, test_images, test_labels = load_mnist_arrays()
    TRAIN_INPUT_CACHE = preprocess_images(train_images)
    pairs, dataset_metadata = build_pair_data(
        train_images, train_labels, test_images, test_labels
    )
    write_json(result_dir / "config.json", config_payload())
    write_json(result_dir / "dataset_manifest.json", dataset_metadata)

    model_path = result_dir / "trajectory_models.csv"
    agreement_path = result_dir / "trajectory_agreement.csv"
    model_rows: list[dict[str, Any]] = [
        dict(row) for row in read_csv(model_path)
    ]
    agreement_rows: list[dict[str, Any]] = [
        dict(row) for row in read_csv(agreement_path)
    ]
    completed = set()
    progress_path = result_dir / "progress.json"
    if Config.RESUME and progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = {int(value) for value in payload.get("completed_n", [])}

    interrupted = False
    try:
        for n in Config.TRAIN_COUNTS:
            if n in completed:
                print(f"n={n} 已完成，跳过。", flush=True)
                continue
            # 若上次在该n中断，删除该n的不完整行后完整重跑，避免混入半条轨迹。
            model_rows = [row for row in model_rows if int(row["n"]) != n]
            agreement_rows = [
                row for row in agreement_rows if int(row["n"]) != n
            ]
            new_models, new_agreement = run_n_group(
                n, pairs, device, start_time
            )
            model_rows.extend(new_models)
            agreement_rows.extend(new_agreement)
            completed.add(n)
            write_csv(model_path, model_rows)
            write_csv(agreement_path, agreement_rows)
            write_json(progress_path, {
                "completed_n": sorted(completed),
                "elapsed_seconds": time.time() - start_time,
            })
    except KeyboardInterrupt:
        interrupted = True
        print("收到中断，保存已完成n。", flush=True)
        write_csv(model_path, model_rows)
        write_csv(agreement_path, agreement_rows)
        write_json(progress_path, {
            "completed_n": sorted(completed),
            "elapsed_seconds": time.time() - start_time,
            "interrupted": True,
        })

    milestones, summary_rows = detect_milestones(model_rows)
    recommendations = build_recommendations(summary_rows)
    write_csv(result_dir / "milestones_per_model.csv", milestones)
    write_csv(result_dir / "calibration_summary.csv", summary_rows)
    write_json(
        result_dir / "calibration_recommendations.json", recommendations
    )
    generate_plots(result_dir, model_rows, summary_rows)
    runtime = {
        "status": "interrupted" if interrupted else "completed",
        "elapsed_seconds": time.time() - start_time,
        "completed_n": sorted(completed),
        "expected_n": list(Config.TRAIN_COUNTS),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    write_json(result_dir / "runtime.json", runtime)

    if Config.PACKAGE_RESULTS:
        archive = package_results(result_dir)
        print(f"下载压缩包：{archive}", flush=True)
    if interrupted:
        print("保持RESUME=True重新运行即可继续剩余n。", flush=True)
    else:
        print("Stage 0完成；请先审阅曲线，再冻结Stage 1的n与epsilon。", flush=True)


if __name__ == "__main__":
    main()
