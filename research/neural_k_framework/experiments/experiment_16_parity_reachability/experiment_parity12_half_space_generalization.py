"""
Parity 的随机半空间训练实验。

目的：区分“统计泛化很好”与“精确恢复完整 parity”。每个数据划分从完整
输入空间中按标签平衡抽取一半训练状态，其余一半作为测试集。脚本同时报告
平均测试准确率、测试集一个不错的比例以及完整函数精确率。

同一 seed 在8个随机划分和完整真值表对照中共享相同初始化。半空间主实验
完成后，脚本还运行配对的对称缺口矩阵，比较单点、互补点对、匹配非对称
点对、相邻异标签点对和四点对称轨道。

AutoDL 用法：修改 Config 后，将整个文件复制到 notebook 单元运行。
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    INPUT_BITS = 14
    TRAIN_FRACTION = 0.5
    SPLIT_COUNT = 8
    SEEDS_PER_CONDITION = 8
    RUN_SYMMETRY_HOLE_MATRIX = False
    SYMMETRY_CASE_COUNT = 4
    SYMMETRY_SEED = 20260827
    RUN_ERROR_REVEAL_INTERVENTION = True
    ERROR_REVEAL_POST_STEPS = 5_000
    ERROR_REPLAY_FRACTION = 0.25

    WIDTH = 64
    HIDDEN_LAYERS = 3
    ACTIVATION = "gelu"
    USE_LAYER_NORM = True
    LAYERNORM_EPS = 1e-5

    OPTIMIZER = "adamw"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 20_000
    TRAIN_BATCH_SIZE = 512
    EVAL_CHUNK_SIZE = 1_024
    EVAL_INTERVAL = 500
    LOG_INTERVAL = 500

    RAW_BCE_THRESHOLDS = (
        0.3,
        0.1,
        0.03,
        0.01,
        0.003,
        0.001,
        3e-4,
        1e-4,
        3e-5,
        1e-5,
        3e-6,
        1e-6,
        1e-7,
        1e-8,
    )

    SPLIT_SEED = 20260824
    INITIALIZATION_SEED = 20260825
    BATCH_SEED = 20260826
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path("/root/results_parity14_error_reveal_intervention")
    PACKAGE_RESULTS = True
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass(frozen=True)
class Condition:
    index: int
    name: str
    is_full_control: bool
    seeds: int


@dataclass(frozen=True)
class SymmetryCondition:
    index: int
    name: str
    family: str
    heldout_indices: tuple[int, ...]
    is_full_control: bool
    seeds: int


@dataclass
class Evaluation:
    train_loss: torch.Tensor
    test_loss: torch.Tensor
    train_accuracy: torch.Tensor
    test_accuracy: torch.Tensor
    train_error_count: torch.Tensor
    test_error_count: torch.Tensor
    train_exact: torch.Tensor
    test_exact: torch.Tensor
    full_exact: torch.Tensor


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.INPUT_BITS = 8
    Config.SPLIT_COUNT = 2
    Config.SEEDS_PER_CONDITION = 2
    Config.SYMMETRY_CASE_COUNT = 2
    Config.ERROR_REVEAL_POST_STEPS = 20
    Config.WIDTH = 16
    Config.MAX_STEPS = 30
    Config.TRAIN_BATCH_SIZE = 64
    Config.EVAL_CHUNK_SIZE = 64
    Config.EVAL_INTERVAL = 10
    Config.LOG_INTERVAL = 10
    Config.RAW_BCE_THRESHOLDS = (0.7, 0.5, 0.3)
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_parity12_half_space_generalization"
    )
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


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists():
        if Config.OVERWRITE_RESULT_DIR:
            shutil.rmtree(output)
        else:
            output = output.parent / (
                output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_config() -> None:
    if Config.INPUT_BITS < 2:
        raise ValueError("INPUT_BITS 必须至少为2。")
    if Config.TRAIN_FRACTION != 0.5:
        raise ValueError("当前脚本预注册为标签平衡的50%训练集。")
    if Config.SPLIT_COUNT < 1 or Config.SEEDS_PER_CONDITION < 1:
        raise ValueError("SPLIT_COUNT/SEEDS_PER_CONDITION 必须为正。")
    if Config.SYMMETRY_CASE_COUNT < 1:
        raise ValueError("SYMMETRY_CASE_COUNT 必须为正。")
    if Config.ERROR_REVEAL_POST_STEPS < 1:
        raise ValueError("ERROR_REVEAL_POST_STEPS 必须为正。")
    if not 0.0 < Config.ERROR_REPLAY_FRACTION < 1.0:
        raise ValueError("ERROR_REPLAY_FRACTION 必须在 (0,1) 内。")
    if Config.WIDTH < 1 or Config.HIDDEN_LAYERS < 1:
        raise ValueError("网络宽度和隐藏层数必须为正。")
    if Config.ACTIVATION not in {"gelu", "tanh", "relu"}:
        raise ValueError("ACTIVATION 只支持 gelu/tanh/relu。")
    if Config.OPTIMIZER not in {"adamw", "sgd"}:
        raise ValueError("OPTIMIZER 只支持 adamw/sgd。")
    thresholds = tuple(float(value) for value in Config.RAW_BCE_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("RAW_BCE_THRESHOLDS 必须严格从高到低且不重复。")


def truth_table_inputs(bits: int) -> np.ndarray:
    values = np.arange(1 << bits, dtype=np.uint64)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def parity_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs.sum(axis=1) % 2).astype(np.uint8)


def build_conditions() -> list[Condition]:
    conditions = [
        Condition(
            index=index,
            name=f"half_split_{index:02d}",
            is_full_control=False,
            seeds=Config.SEEDS_PER_CONDITION,
        )
        for index in range(Config.SPLIT_COUNT)
    ]
    conditions.append(Condition(
        index=len(conditions),
        name="full_truth_table_control",
        is_full_control=True,
        seeds=Config.SEEDS_PER_CONDITION,
    ))
    return conditions


def build_balanced_split_masks(
    labels: np.ndarray,
    conditions: Sequence[Condition],
) -> np.ndarray:
    state_count = len(labels)
    masks = np.zeros((len(conditions), state_count), dtype=bool)
    zeros = np.flatnonzero(labels == 0)
    ones = np.flatnonzero(labels == 1)
    per_class = len(zeros) // 2
    for condition in conditions:
        if condition.is_full_control:
            masks[condition.index] = True
            continue
        rng = np.random.default_rng(
            Config.SPLIT_SEED + condition.index * 104729
        )
        zero_train = rng.choice(zeros, size=per_class, replace=False)
        one_train = rng.choice(ones, size=per_class, replace=False)
        masks[condition.index, zero_train] = True
        masks[condition.index, one_train] = True
    return masks


def build_model_layout(
    conditions: Sequence[Condition],
) -> tuple[np.ndarray, np.ndarray]:
    condition_indices: list[int] = []
    seed_indices: list[int] = []
    for condition in conditions:
        condition_indices.extend([condition.index] * condition.seeds)
        seed_indices.extend(range(condition.seeds))
    return (
        np.asarray(condition_indices, dtype=np.int64),
        np.asarray(seed_indices, dtype=np.int64),
    )


class BatchedIndependentMLP(nn.Module):
    def __init__(
        self,
        input_bits: int,
        model_seed_indices: np.ndarray,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model_count = len(model_seed_indices)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()

        seed_indices = torch.from_numpy(model_seed_indices).to(device)
        base_seed_count = int(model_seed_indices.max()) + 1
        generator = torch.Generator(device=device)
        generator.manual_seed(Config.INITIALIZATION_SEED)
        dimensions = (
            [input_bits]
            + [Config.WIDTH] * Config.HIDDEN_LAYERS
            + [1]
        )
        for layer_index, (input_size, output_size) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            bound = 1.0 / math.sqrt(input_size)
            base_weight = torch.empty(
                base_seed_count,
                output_size,
                input_size,
                device=device,
            ).uniform_(-bound, bound, generator=generator)
            base_bias = torch.empty(
                base_seed_count,
                output_size,
                device=device,
            ).uniform_(-bound, bound, generator=generator)
            self.weights.append(nn.Parameter(base_weight[seed_indices].clone()))
            self.biases.append(nn.Parameter(base_bias[seed_indices].clone()))
            if layer_index < Config.HIDDEN_LAYERS:
                self.norm_weights.append(nn.Parameter(torch.ones(
                    self.model_count, output_size, device=device
                )))
                self.norm_biases.append(nn.Parameter(torch.zeros(
                    self.model_count, output_size, device=device
                )))

    def activate(self, value: torch.Tensor) -> torch.Tensor:
        if Config.ACTIVATION == "gelu":
            return F.gelu(value)
        if Config.ACTIVATION == "tanh":
            return torch.tanh(value)
        return F.relu(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            hidden = inputs[None].expand(self.model_count, -1, -1)
        elif inputs.ndim == 3 and inputs.shape[0] == self.model_count:
            hidden = inputs
        else:
            raise ValueError(
                "inputs 必须是 [states,bits] 或 [models,states,bits]。"
            )
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if layer_index < Config.HIDDEN_LAYERS:
                hidden = self.activate(hidden)
                if Config.USE_LAYER_NORM:
                    mean = hidden.mean(dim=-1, keepdim=True)
                    variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
                    hidden = (hidden - mean) * torch.rsqrt(
                        variance + Config.LAYERNORM_EPS
                    )
                    hidden = (
                        hidden * self.norm_weights[layer_index][:, None, :]
                        + self.norm_biases[layer_index][:, None, :]
                    )
        return hidden.squeeze(-1)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if Config.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )


def sample_modelwise_batch_indices(
    condition_train_indices: Sequence[torch.Tensor],
    model_condition_indices: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    condition_batches = []
    for indices in condition_train_indices:
        positions = torch.randint(
            0,
            len(indices),
            (Config.TRAIN_BATCH_SIZE,),
            generator=generator,
            device="cpu",
        )
        condition_batches.append(indices[positions])
    table = torch.stack(condition_batches).to(device)
    return table[model_condition_indices]


def training_batch_loss(
    model: BatchedIndependentMLP,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    model_batch_indices: torch.Tensor,
) -> torch.Tensor:
    inputs = all_inputs[model_batch_indices]
    targets = all_targets[model_batch_indices]
    logits = model(inputs)
    return F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=1)


@torch.no_grad()
def evaluate_models(
    model: BatchedIndependentMLP,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    model_train_masks: torch.Tensor,
) -> Evaluation:
    model_count = model.model_count
    device = all_inputs.device
    train_loss_sum = torch.zeros(model_count, device=device)
    test_loss_sum = torch.zeros(model_count, device=device)
    train_counts = model_train_masks.sum(dim=1)
    test_counts = (~model_train_masks).sum(dim=1)
    train_errors = torch.zeros(model_count, dtype=torch.int64, device=device)
    test_errors = torch.zeros(model_count, dtype=torch.int64, device=device)

    for start in range(0, len(all_inputs), Config.EVAL_CHUNK_SIZE):
        end = min(start + Config.EVAL_CHUNK_SIZE, len(all_inputs))
        logits = model(all_inputs[start:end])
        targets = all_targets[start:end]
        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            targets[None].expand_as(logits),
            reduction="none",
        )
        train_mask = model_train_masks[:, start:end]
        test_mask = ~train_mask
        train_loss_sum += (elementwise * train_mask).sum(dim=1)
        test_loss_sum += (elementwise * test_mask).sum(dim=1)
        errors = (logits >= 0) != targets.bool()[None]
        train_errors += (errors & train_mask).sum(dim=1)
        test_errors += (errors & test_mask).sum(dim=1)

    train_loss = train_loss_sum / train_counts.clamp_min(1)
    test_loss = test_loss_sum / test_counts.clamp_min(1)
    train_accuracy = 1.0 - train_errors.float() / train_counts.clamp_min(1)
    test_accuracy = 1.0 - test_errors.float() / test_counts.clamp_min(1)
    train_exact = train_errors == 0
    has_test = test_counts > 0
    test_exact = torch.where(
        has_test,
        test_errors == 0,
        torch.ones_like(train_exact),
    )
    full_exact = train_exact & test_exact
    test_loss = torch.where(
        has_test, test_loss, torch.full_like(test_loss, float("nan"))
    )
    test_accuracy = torch.where(
        has_test,
        test_accuracy,
        torch.full_like(test_accuracy, float("nan")),
    )
    return Evaluation(
        train_loss=train_loss,
        test_loss=test_loss,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        train_error_count=train_errors,
        test_error_count=test_errors,
        train_exact=train_exact,
        test_exact=test_exact,
        full_exact=full_exact,
    )


def safe_fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def aggregate_evaluation_rows(
    step: int,
    evaluation: Evaluation,
    model_is_control: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, mask in (
        ("half_train", ~model_is_control),
        ("full_control", model_is_control),
    ):
        train_exact = evaluation.train_exact[mask]
        full_exact = evaluation.full_exact[mask]
        train_exact_count = int(train_exact.sum().item())
        row: dict[str, Any] = {
            "step": step,
            "group": group_name,
            "model_count": int(mask.sum().item()),
            "train_loss_mean": float(evaluation.train_loss[mask].mean().item()),
            "train_loss_median": float(
                evaluation.train_loss[mask].median().item()
            ),
            "train_loss_min": float(evaluation.train_loss[mask].min().item()),
            "train_loss_max": float(evaluation.train_loss[mask].max().item()),
            "train_accuracy_mean": float(
                evaluation.train_accuracy[mask].mean().item()
            ),
            "train_exact_count": train_exact_count,
            "train_exact_fraction": float(train_exact.float().mean().item()),
            "full_parity_exact_count": int(full_exact.sum().item()),
            "full_parity_exact_fraction": float(full_exact.float().mean().item()),
        }
        if group_name == "half_train":
            test_accuracy = evaluation.test_accuracy[mask]
            test_exact = evaluation.test_exact[mask]
            test_errors = evaluation.test_error_count[mask]
            exact_test_accuracy = test_accuracy[train_exact]
            exact_test_exact = test_exact[train_exact]
            row.update({
                "test_loss_mean": float(evaluation.test_loss[mask].mean().item()),
                "test_loss_median": float(
                    evaluation.test_loss[mask].median().item()
                ),
                "test_accuracy_mean": float(test_accuracy.mean().item()),
                "test_accuracy_median": float(test_accuracy.median().item()),
                "test_accuracy_min": float(test_accuracy.min().item()),
                "test_accuracy_max": float(test_accuracy.max().item()),
                "test_error_count_mean": float(test_errors.float().mean().item()),
                "test_error_count_median": float(
                    test_errors.float().median().item()
                ),
                "test_exact_count": int(test_exact.sum().item()),
                "test_exact_fraction": float(test_exact.float().mean().item()),
                "test_accuracy_given_train_exact_mean": (
                    float(exact_test_accuracy.mean().item())
                    if train_exact_count else None
                ),
                "test_exact_given_train_exact": (
                    float(exact_test_exact.float().mean().item())
                    if train_exact_count else None
                ),
            })
        else:
            row.update({
                "test_loss_mean": None,
                "test_loss_median": None,
                "test_accuracy_mean": None,
                "test_accuracy_median": None,
                "test_accuracy_min": None,
                "test_accuracy_max": None,
                "test_error_count_mean": None,
                "test_error_count_median": None,
                "test_exact_count": None,
                "test_exact_fraction": None,
                "test_accuracy_given_train_exact_mean": None,
                "test_exact_given_train_exact": None,
            })
        rows.append(row)
    return rows


@torch.no_grad()
def update_crossings(
    step: int,
    evaluation: Evaluation,
    thresholds: torch.Tensor,
    crossed: torch.Tensor,
    crossing_step: torch.Tensor,
    crossing_test_accuracy: torch.Tensor,
    crossing_test_exact: torch.Tensor,
) -> int:
    new = (
        (evaluation.train_loss[:, None] <= thresholds[None])
        & evaluation.train_exact[:, None]
        & ~crossed
    )
    count = int(new.sum().item())
    if count == 0:
        return 0
    crossed[new] = True
    crossing_step[new] = int(step)
    crossing_test_accuracy[new] = (
        evaluation.test_accuracy[:, None].expand_as(new)[new]
    )
    crossing_test_exact[new] = (
        evaluation.test_exact[:, None].expand_as(new)[new]
    )
    return count


def summarize_crossings(
    thresholds: Sequence[float],
    model_is_control: np.ndarray,
    crossing_step: np.ndarray,
    crossing_test_accuracy: np.ndarray,
    crossing_test_exact: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    half_mask = ~model_is_control
    for threshold_index, threshold in enumerate(thresholds):
        reached = half_mask & (crossing_step[:, threshold_index] >= 0)
        count = int(reached.sum())
        accuracies = crossing_test_accuracy[reached, threshold_index]
        exact = crossing_test_exact[reached, threshold_index]
        rows.append({
            "raw_bce_threshold": threshold,
            "eligible_model_count": int(half_mask.sum()),
            "reached_count": count,
            "reached_fraction": safe_fraction(count, int(half_mask.sum())),
            "median_crossing_step": (
                float(np.median(crossing_step[reached, threshold_index]))
                if count else None
            ),
            "test_accuracy_mean": (
                float(np.mean(accuracies)) if count else None
            ),
            "test_accuracy_median": (
                float(np.median(accuracies)) if count else None
            ),
            "test_accuracy_min": (
                float(np.min(accuracies)) if count else None
            ),
            "test_accuracy_max": (
                float(np.max(accuracies)) if count else None
            ),
            "test_exact_count": int(np.sum(exact)) if count else 0,
            "test_exact_fraction": (
                float(np.mean(exact)) if count else None
            ),
        })
    return rows


def final_model_rows(
    conditions: Sequence[Condition],
    model_condition_indices: np.ndarray,
    model_seed_indices: np.ndarray,
    evaluation: Evaluation,
) -> list[dict[str, Any]]:
    arrays = {
        "train_loss": evaluation.train_loss.cpu().numpy(),
        "test_loss": evaluation.test_loss.cpu().numpy(),
        "train_accuracy": evaluation.train_accuracy.cpu().numpy(),
        "test_accuracy": evaluation.test_accuracy.cpu().numpy(),
        "train_error_count": evaluation.train_error_count.cpu().numpy(),
        "test_error_count": evaluation.test_error_count.cpu().numpy(),
        "train_exact": evaluation.train_exact.cpu().numpy(),
        "test_exact": evaluation.test_exact.cpu().numpy(),
        "full_exact": evaluation.full_exact.cpu().numpy(),
    }
    rows: list[dict[str, Any]] = []
    for model_index, condition_index in enumerate(model_condition_indices):
        condition = conditions[int(condition_index)]
        rows.append({
            "model_index": model_index,
            "condition": condition.name,
            "is_full_control": condition.is_full_control,
            "seed": int(model_seed_indices[model_index]),
            **{key: value[model_index] for key, value in arrays.items()},
        })
    return rows


def model_evaluation_rows(
    step: int,
    conditions: Sequence[Condition],
    model_condition_indices: np.ndarray,
    model_seed_indices: np.ndarray,
    evaluation: Evaluation,
) -> list[dict[str, Any]]:
    rows = final_model_rows(
        conditions,
        model_condition_indices,
        model_seed_indices,
        evaluation,
    )
    for row in rows:
        row["step"] = step
    return rows


def sample_state_of_weight(
    bits: int,
    weight: int,
    rng: np.random.Generator,
    forbidden: set[int] | None = None,
    predicate: Any | None = None,
) -> int:
    forbidden = forbidden or set()
    for _ in range(10_000):
        positions = rng.choice(bits, size=weight, replace=False)
        state = sum(1 << int(position) for position in positions)
        if state in forbidden:
            continue
        if predicate is not None and not predicate(state):
            continue
        return int(state)
    raise RuntimeError(
        f"无法采样 bits={bits}, weight={weight} 的匹配对照状态。"
    )


def build_symmetry_conditions(
    bits: int,
) -> tuple[list[SymmetryCondition], list[dict[str, Any]]]:
    if bits % 2:
        raise ValueError("补对称矩阵当前要求偶数 INPUT_BITS。")
    rng = np.random.default_rng(Config.SYMMETRY_SEED)
    full_mask = (1 << bits) - 1
    raw_weights = np.linspace(
        1, bits // 2, Config.SYMMETRY_CASE_COUNT
    )
    weights: list[int] = []
    for value in raw_weights:
        weight = int(round(float(value)))
        while weight in weights and weight < bits // 2:
            weight += 1
        while weight in weights and weight > 1:
            weight -= 1
        if weight in weights:
            raise ValueError("SYMMETRY_CASE_COUNT 对当前位数过大。")
        weights.append(weight)

    conditions: list[SymmetryCondition] = []
    case_rows: list[dict[str, Any]] = []

    def append_condition(
        family: str,
        case_index: int,
        heldout: Sequence[int],
    ) -> None:
        values = tuple(sorted(set(int(value) for value in heldout)))
        conditions.append(SymmetryCondition(
            index=len(conditions),
            name=f"{family}_case_{case_index:02d}",
            family=family,
            heldout_indices=values,
            is_full_control=False,
            seeds=Config.SEEDS_PER_CONDITION,
        ))

    for case_index, weight in enumerate(weights):
        anchor = sample_state_of_weight(bits, weight, rng)
        complement = anchor ^ full_mask
        zero_positions = [
            bit for bit in range(bits) if ((anchor >> bit) & 1) == 0
        ]
        edge_bit = int(rng.choice(zero_positions))
        adjacent = anchor ^ (1 << edge_bit)
        complement_adjacent = adjacent ^ full_mask

        matched_complement = sample_state_of_weight(
            bits,
            bits - weight,
            rng,
            forbidden={anchor, complement},
            predicate=lambda state, a=anchor: (state ^ a) != full_mask,
        )

        append_condition("single_hole", case_index, (anchor,))
        append_condition(
            "complement_pair", case_index, (anchor, complement)
        )
        append_condition(
            "matched_asymmetric_pair",
            case_index,
            (anchor, matched_complement),
        )
        append_condition("adjacent_pair", case_index, (anchor, adjacent))
        append_condition(
            "four_point_orbit",
            case_index,
            (anchor, adjacent, complement, complement_adjacent),
        )
        case_rows.append({
            "case_index": case_index,
            "anchor": anchor,
            "anchor_bits": format(anchor, f"0{bits}b"),
            "anchor_weight": anchor.bit_count(),
            "anchor_target": anchor.bit_count() % 2,
            "complement": complement,
            "complement_bits": format(complement, f"0{bits}b"),
            "matched_complement": matched_complement,
            "matched_complement_bits": format(
                matched_complement, f"0{bits}b"
            ),
            "edge_bit": edge_bit,
            "adjacent": adjacent,
            "adjacent_bits": format(adjacent, f"0{bits}b"),
            "orbit": (
                anchor,
                adjacent,
                complement,
                complement_adjacent,
            ),
        })

    conditions.append(SymmetryCondition(
        index=len(conditions),
        name="full_truth_table_control",
        family="full_control",
        heldout_indices=(),
        is_full_control=True,
        seeds=Config.SEEDS_PER_CONDITION,
    ))
    return conditions, case_rows


def build_symmetry_masks(
    state_count: int,
    conditions: Sequence[SymmetryCondition],
) -> np.ndarray:
    masks = np.ones((len(conditions), state_count), dtype=bool)
    for condition in conditions:
        if condition.heldout_indices:
            masks[condition.index, list(condition.heldout_indices)] = False
    return masks


def aggregate_symmetry_rows(
    step: int,
    evaluation: Evaluation,
    model_families: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_order = (
        "single_hole",
        "complement_pair",
        "matched_asymmetric_pair",
        "adjacent_pair",
        "four_point_orbit",
        "full_control",
    )
    for family in family_order:
        mask_np = model_families == family
        mask = torch.from_numpy(mask_np).to(evaluation.train_loss.device)
        train_exact = evaluation.train_exact[mask]
        test_exact = evaluation.test_exact[mask]
        full_exact = evaluation.full_exact[mask]
        train_exact_count = int(train_exact.sum().item())
        test_accuracy = evaluation.test_accuracy[mask]
        test_errors = evaluation.test_error_count[mask]
        test_count = 0 if family == "full_control" else {
            "single_hole": 1,
            "complement_pair": 2,
            "matched_asymmetric_pair": 2,
            "adjacent_pair": 2,
            "four_point_orbit": 4,
        }[family]
        exact_test = test_exact[train_exact]
        exact_accuracy = test_accuracy[train_exact]
        relation_fraction: float | None = None
        if family in {"complement_pair", "adjacent_pair"} and train_exact_count:
            local_errors = test_errors[train_exact]
            relation_fraction = float(
                ((local_errors == 0) | (local_errors == 2))
                .float()
                .mean()
                .item()
            )
        row = {
            "step": step,
            "family": family,
            "model_count": int(mask.sum().item()),
            "heldout_count": test_count,
            "train_loss_mean": float(evaluation.train_loss[mask].mean().item()),
            "train_loss_median": float(
                evaluation.train_loss[mask].median().item()
            ),
            "train_exact_count": train_exact_count,
            "train_exact_fraction": float(train_exact.float().mean().item()),
            "heldout_bit_accuracy_mean": (
                float(test_accuracy.mean().item()) if test_count else None
            ),
            "heldout_exact_count": (
                int(test_exact.sum().item()) if test_count else None
            ),
            "heldout_exact_fraction": (
                float(test_exact.float().mean().item()) if test_count else None
            ),
            "heldout_bit_accuracy_given_train_exact": (
                float(exact_accuracy.mean().item())
                if test_count and train_exact_count else None
            ),
            "heldout_exact_given_train_exact": (
                float(exact_test.float().mean().item())
                if test_count and train_exact_count else None
            ),
            "pair_relation_given_train_exact": relation_fraction,
            "full_parity_exact_count": int(full_exact.sum().item()),
            "full_parity_exact_fraction": float(
                full_exact.float().mean().item()
            ),
        }
        rows.append(row)
    return rows


def symmetry_final_condition_rows(
    conditions: Sequence[SymmetryCondition],
    inputs: np.ndarray,
    labels: np.ndarray,
    model_condition_indices: np.ndarray,
    model_seed_indices: np.ndarray,
    evaluation: Evaluation,
) -> list[dict[str, Any]]:
    train_loss = evaluation.train_loss.cpu().numpy()
    train_exact = evaluation.train_exact.cpu().numpy()
    test_accuracy = evaluation.test_accuracy.cpu().numpy()
    test_errors = evaluation.test_error_count.cpu().numpy()
    test_exact = evaluation.test_exact.cpu().numpy()
    full_exact = evaluation.full_exact.cpu().numpy()
    rows: list[dict[str, Any]] = []
    for model_index, condition_index in enumerate(model_condition_indices):
        condition = conditions[int(condition_index)]
        heldout = condition.heldout_indices
        rows.append({
            "model_index": model_index,
            "condition": condition.name,
            "family": condition.family,
            "seed": int(model_seed_indices[model_index]),
            "heldout_indices": heldout,
            "heldout_bits": tuple(
                "".join(str(int(value)) for value in inputs[index])
                for index in heldout
            ),
            "heldout_targets": tuple(int(labels[index]) for index in heldout),
            "train_loss": train_loss[model_index],
            "train_exact": train_exact[model_index],
            "heldout_bit_accuracy": (
                test_accuracy[model_index] if heldout else None
            ),
            "heldout_error_count": (
                int(test_errors[model_index]) if heldout else None
            ),
            "heldout_exact": (
                bool(test_exact[model_index]) if heldout else None
            ),
            "full_parity_exact": bool(full_exact[model_index]),
        })
    return rows


def plot_symmetry_results(
    output_dir: Path,
    final_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [row for row in final_rows if row["family"] != "full_control"]
    families = [
        "single_hole",
        "complement_pair",
        "matched_asymmetric_pair",
        "adjacent_pair",
        "four_point_orbit",
    ]
    train_exact = []
    heldout_exact = []
    bit_accuracy = []
    for family in families:
        local = [row for row in rows if row["family"] == family]
        exact_local = [row for row in local if row["train_exact"]]
        train_exact.append(np.mean([row["train_exact"] for row in local]))
        heldout_exact.append(
            np.mean([row["heldout_exact"] for row in exact_local])
            if exact_local else np.nan
        )
        bit_accuracy.append(
            np.mean([row["heldout_bit_accuracy"] for row in exact_local])
            if exact_local else np.nan
        )
    x = np.arange(len(families))
    width = 0.26
    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.bar(x - width, train_exact, width, label="train exact")
    axis.bar(x, bit_accuracy, width, label="heldout bit accuracy | train exact")
    axis.bar(x + width, heldout_exact, width, label="heldout all exact | train exact")
    axis.set_xticks(x, families, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("fraction")
    axis.set_title("Parity-12 symmetry-preserving hole matrix")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(
        output_dir / f"parity{Config.INPUT_BITS}_symmetry_hole_matrix.png",
        dpi=180,
    )
    plt.close(figure)


def run_symmetry_hole_matrix(
    output_dir: Path,
    inputs_np: np.ndarray,
    labels_np: np.ndarray,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    symmetry_dir = output_dir / "symmetry_hole_matrix"
    symmetry_dir.mkdir(parents=True, exist_ok=True)
    conditions, case_rows = build_symmetry_conditions(Config.INPUT_BITS)
    masks_np = build_symmetry_masks(len(inputs_np), conditions)
    model_condition_indices, model_seed_indices = build_model_layout(conditions)
    model_masks_np = masks_np[model_condition_indices]
    model_families = np.asarray([
        conditions[index].family for index in model_condition_indices
    ])
    model_condition_tensor = torch.from_numpy(model_condition_indices).to(device)
    model_masks = torch.from_numpy(model_masks_np).to(device)
    condition_train_indices = [
        torch.from_numpy(np.flatnonzero(mask).astype(np.int64))
        for mask in masks_np
    ]

    write_json(symmetry_dir / "conditions.json", [
        {
            "index": condition.index,
            "name": condition.name,
            "family": condition.family,
            "heldout_indices": condition.heldout_indices,
            "heldout_bits": tuple(
                format(index, f"0{Config.INPUT_BITS}b")
                for index in condition.heldout_indices
            ),
            "is_full_control": condition.is_full_control,
            "seeds": condition.seeds,
        }
        for condition in conditions
    ])
    write_csv(symmetry_dir / "symmetry_cases.csv", case_rows)

    model = BatchedIndependentMLP(
        Config.INPUT_BITS, model_seed_indices, device
    )
    optimizer = make_optimizer(model)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(Config.BATCH_SEED + 1_000_003)
    print(
        f"\n=== {Config.INPUT_BITS}-bit parity 对称缺口矩阵 ===",
        flush=True,
    )
    print(
        f"cases={Config.SYMMETRY_CASE_COUNT} | conditions={len(conditions)} | "
        f"models={model.model_count}",
        flush=True,
    )

    eval_steps = {
        0, 1, 2, 5, 10, 20, 50, 100, 200,
        *range(Config.EVAL_INTERVAL, Config.MAX_STEPS + 1, Config.EVAL_INTERVAL),
        Config.MAX_STEPS,
    }
    trajectory_rows: list[dict[str, Any]] = []
    interrupted = False
    final_step = 0
    started = time.perf_counter()
    try:
        for step in range(Config.MAX_STEPS + 1):
            if step in eval_steps:
                evaluation = evaluate_models(
                    model, all_inputs, all_targets, model_masks
                )
                current = aggregate_symmetry_rows(
                    step, evaluation, model_families
                )
                trajectory_rows.extend(current)
                if step <= 200 or step % Config.LOG_INTERVAL == 0:
                    compact = " | ".join(
                        f"{row['family']}:train="
                        f"{row['train_exact_fraction']:.0%},held="
                        + (
                            "NA"
                            if row["heldout_exact_given_train_exact"] is None
                            else f"{row['heldout_exact_given_train_exact']:.0%}"
                        )
                        for row in current
                    )
                    print(
                        f"symmetry step={step:>7,} | {compact} | "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                write_csv(
                    symmetry_dir / "symmetry_trajectory.csv",
                    trajectory_rows,
                )
            final_step = step
            if step == Config.MAX_STEPS:
                break
            batch_indices = sample_modelwise_batch_indices(
                condition_train_indices,
                model_condition_tensor,
                batch_generator,
                device,
            )
            losses = training_batch_loss(
                model, all_inputs, all_targets, batch_indices
            )
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            optimizer.step()
    except KeyboardInterrupt:
        interrupted = True
        print("\n对称缺口矩阵收到中断，正在保存……", flush=True)

    evaluation = evaluate_models(model, all_inputs, all_targets, model_masks)
    final_aggregate = aggregate_symmetry_rows(
        final_step, evaluation, model_families
    )
    if not any(row["step"] == final_step for row in trajectory_rows):
        trajectory_rows.extend(final_aggregate)
    final_rows = symmetry_final_condition_rows(
        conditions,
        inputs_np,
        labels_np,
        model_condition_indices,
        model_seed_indices,
        evaluation,
    )
    write_csv(symmetry_dir / "symmetry_trajectory.csv", trajectory_rows)
    write_csv(
        symmetry_dir / "symmetry_final_family_summary.csv",
        final_aggregate,
    )
    write_csv(
        symmetry_dir / "symmetry_final_model_summary.csv",
        final_rows,
    )
    plot_symmetry_results(symmetry_dir, final_rows)
    summary = {
        "status": "interrupted" if interrupted else "completed",
        "final_step": final_step,
        "elapsed_seconds": time.perf_counter() - started,
        "model_count": model.model_count,
        "case_count": Config.SYMMETRY_CASE_COUNT,
        "final_family_summary": final_aggregate,
        "interpretation": {
            "complement_beats_matched": (
                "在相同 Hamming profile 和标签下，补对称闭合本身提高精确补全。"
            ),
            "relation_high_exact_low": (
                "模型保持成对对称关系，但可能整体选错相位。"
            ),
            "orbit_best": (
                "同时恢复补对称与局部 parity 反对称进一步排除了缺陷函数。"
            ),
        },
    }
    write_json(symmetry_dir / "summary.json", summary)
    return summary, interrupted


def clone_models_from_parents(
    parent: BatchedIndependentMLP,
    parent_indices: np.ndarray,
    device: torch.device,
) -> BatchedIndependentMLP:
    child = BatchedIndependentMLP(
        Config.INPUT_BITS,
        np.zeros(len(parent_indices), dtype=np.int64),
        device,
    )
    index_tensor = torch.from_numpy(parent_indices).to(device)
    with torch.no_grad():
        for destination, source in zip(child.weights, parent.weights):
            destination.copy_(source[index_tensor])
        for destination, source in zip(child.biases, parent.biases):
            destination.copy_(source[index_tensor])
        for destination, source in zip(
            child.norm_weights, parent.norm_weights
        ):
            destination.copy_(source[index_tensor])
        for destination, source in zip(
            child.norm_biases, parent.norm_biases
        ):
            destination.copy_(source[index_tensor])
    return child


def clone_optimizer_from_parents(
    parent_optimizer: torch.optim.Optimizer,
    child_model: BatchedIndependentMLP,
    parent_model: BatchedIndependentMLP,
    parent_indices: np.ndarray,
    device: torch.device,
) -> torch.optim.Optimizer:
    child_optimizer = make_optimizer(child_model)
    index_tensor = torch.from_numpy(parent_indices).to(device)
    for parent_parameter, child_parameter in zip(
        parent_model.parameters(), child_model.parameters()
    ):
        parent_state = parent_optimizer.state.get(parent_parameter, {})
        child_state: dict[str, Any] = {}
        for key, value in parent_state.items():
            if not isinstance(value, torch.Tensor):
                child_state[key] = value
            elif value.ndim > 0 and value.shape[0] == parent_model.model_count:
                child_state[key] = value[index_tensor].detach().clone()
            else:
                child_state[key] = value.detach().clone()
        if child_state:
            child_optimizer.state[child_parameter] = child_state
    return child_optimizer


@torch.no_grad()
def full_error_matrix(
    model: BatchedIndependentMLP,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(all_inputs), Config.EVAL_CHUNK_SIZE):
        end = min(start + Config.EVAL_CHUNK_SIZE, len(all_inputs))
        logits = model(all_inputs[start:end])
        errors = (logits >= 0) != all_targets[start:end].bool()[None]
        parts.append(errors.cpu().numpy())
    return np.concatenate(parts, axis=1)


def sample_from_model_masks(
    model_masks: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    model_count, state_count = model_masks.shape
    indices = torch.randint(
        0,
        state_count,
        (model_count, batch_size),
        generator=generator,
        device=model_masks.device,
    )
    for _ in range(64):
        valid = model_masks.gather(1, indices)
        if bool(valid.all().item()):
            return indices
        replacements = torch.randint(
            0,
            state_count,
            indices.shape,
            generator=generator,
            device=model_masks.device,
        )
        indices = torch.where(valid, indices, replacements)
    raise RuntimeError("从模型专属训练 mask 采样时拒绝采样未收敛。")


def evaluate_error_reveal_models(
    model: BatchedIndependentMLP,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    augmented_train_masks: torch.Tensor,
    original_test_masks_np: np.ndarray,
    initial_error_masks_np: np.ndarray,
    revealed_masks_np: np.ndarray,
) -> dict[str, np.ndarray]:
    train_evaluation = evaluate_models(
        model,
        all_inputs,
        all_targets,
        augmented_train_masks,
    )
    errors = full_error_matrix(model, all_inputs, all_targets)
    initial_correct_masks = original_test_masks_np & ~initial_error_masks_np
    unrevealed_masks = original_test_masks_np & ~revealed_masks_np
    original_test_count = original_test_masks_np.sum(axis=1)
    metrics = {
        "train_loss": train_evaluation.train_loss.cpu().numpy(),
        "augmented_train_exact": train_evaluation.train_exact.cpu().numpy(),
        "original_test_error_count": (
            errors & original_test_masks_np
        ).sum(axis=1),
        "original_test_accuracy": 1.0 - (
            (errors & original_test_masks_np).sum(axis=1)
            / np.maximum(original_test_count, 1)
        ),
        "remaining_initial_errors": (
            errors & initial_error_masks_np
        ).sum(axis=1),
        "new_error_count": (errors & initial_correct_masks).sum(axis=1),
        "remaining_revealed_errors": (
            errors & revealed_masks_np
        ).sum(axis=1),
        "unrevealed_error_count": (errors & unrevealed_masks).sum(axis=1),
        "full_error_count": errors.sum(axis=1),
        "full_exact": errors.sum(axis=1) == 0,
    }
    return metrics


def aggregate_error_reveal_rows(
    step: int,
    branch_names: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch_order = (
        "continue_only",
        "random_correct_reveal",
        "error_reveal",
        "error_replay",
        "full_reveal",
    )
    for branch in branch_order:
        mask = branch_names == branch
        rows.append({
            "step": step,
            "branch": branch,
            "model_count": int(mask.sum()),
            "train_loss_mean": float(metrics["train_loss"][mask].mean()),
            "train_loss_median": float(np.median(metrics["train_loss"][mask])),
            "augmented_train_exact_fraction": float(
                metrics["augmented_train_exact"][mask].mean()
            ),
            "original_test_accuracy_mean": float(
                metrics["original_test_accuracy"][mask].mean()
            ),
            "original_test_error_mean": float(
                metrics["original_test_error_count"][mask].mean()
            ),
            "original_test_error_median": float(
                np.median(metrics["original_test_error_count"][mask])
            ),
            "remaining_initial_errors_mean": float(
                metrics["remaining_initial_errors"][mask].mean()
            ),
            "new_error_count_mean": float(
                metrics["new_error_count"][mask].mean()
            ),
            "remaining_revealed_errors_mean": float(
                metrics["remaining_revealed_errors"][mask].mean()
            ),
            "unrevealed_error_count_mean": float(
                metrics["unrevealed_error_count"][mask].mean()
            ),
            "full_exact_count": int(metrics["full_exact"][mask].sum()),
            "full_exact_fraction": float(metrics["full_exact"][mask].mean()),
        })
    return rows


def error_reveal_model_rows(
    step: int,
    branch_names: np.ndarray,
    parent_local_indices: np.ndarray,
    parent_conditions: np.ndarray,
    parent_seeds: np.ndarray,
    initial_error_counts: np.ndarray,
    revealed_counts: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, branch in enumerate(branch_names):
        rows.append({
            "step": step,
            "branch": str(branch),
            "parent_model_index": int(parent_local_indices[index]),
            "condition": str(parent_conditions[index]),
            "seed": int(parent_seeds[index]),
            "initial_error_count": int(initial_error_counts[index]),
            "revealed_count": int(revealed_counts[index]),
            **{
                key: value[index]
                for key, value in metrics.items()
            },
        })
    return rows


def plot_error_reveal_results(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    branches = (
        "continue_only",
        "random_correct_reveal",
        "error_reveal",
        "error_replay",
        "full_reveal",
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for branch in branches:
        local = sorted(
            [row for row in rows if row["branch"] == branch],
            key=lambda item: int(item["step"]),
        )
        axes[0].plot(
            [row["step"] for row in local],
            [row["remaining_initial_errors_mean"] for row in local],
            label=branch,
        )
        axes[1].plot(
            [row["step"] for row in local],
            [row["new_error_count_mean"] for row in local],
            label=branch,
        )
        axes[2].plot(
            [row["step"] for row in local],
            [row["full_exact_fraction"] for row in local],
            label=branch,
        )
    axes[0].set_title("Remaining initially wrong points")
    axes[1].set_title("New errors on initially correct test points")
    axes[2].set_title("Exact full parity recovery")
    for axis in axes:
        axis.set_xlabel("post-intervention step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean errors/model")
    axes[1].set_ylabel("mean errors/model")
    axes[2].set_ylabel("fraction")
    axes[2].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8)
    figure.savefig(output_dir / "error_reveal_intervention.png", dpi=180)
    plt.close(figure)


def run_error_reveal_intervention(
    output_dir: Path,
    parent_model: BatchedIndependentMLP,
    parent_optimizer: torch.optim.Optimizer,
    all_inputs: torch.Tensor,
    all_targets: torch.Tensor,
    model_train_masks_np: np.ndarray,
    model_condition_indices: np.ndarray,
    model_seed_indices: np.ndarray,
    conditions: Sequence[Condition],
    model_is_control_np: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    intervention_dir = output_dir / "error_reveal_intervention"
    intervention_dir.mkdir(parents=True, exist_ok=True)
    half_parent_global = np.flatnonzero(~model_is_control_np)
    parent_count = len(half_parent_global)
    parent_errors_all = full_error_matrix(
        parent_model, all_inputs, all_targets
    )[half_parent_global]
    parent_train_masks = model_train_masks_np[half_parent_global]
    original_test_masks = ~parent_train_masks
    initial_error_masks = parent_errors_all & original_test_masks
    initial_error_counts_parent = initial_error_masks.sum(axis=1)

    branches = (
        "continue_only",
        "random_correct_reveal",
        "error_reveal",
        "error_replay",
        "full_reveal",
    )
    parent_global_indices = np.tile(half_parent_global, len(branches))
    parent_local_indices = np.tile(np.arange(parent_count), len(branches))
    branch_names = np.repeat(np.asarray(branches, dtype=object), parent_count)
    original_test_repeated = np.tile(original_test_masks, (len(branches), 1))
    initial_error_repeated = np.tile(initial_error_masks, (len(branches), 1))
    augmented_masks = np.tile(parent_train_masks, (len(branches), 1))
    revealed_masks = np.zeros_like(augmented_masks)

    rng = np.random.default_rng(Config.BATCH_SEED + 9_999_991)
    for branch_index, branch in enumerate(branches):
        start = branch_index * parent_count
        end = start + parent_count
        for local_index in range(parent_count):
            row = start + local_index
            errors = np.flatnonzero(initial_error_masks[local_index])
            if branch in {"error_reveal", "error_replay"}:
                augmented_masks[row, errors] = True
                revealed_masks[row, errors] = True
            elif branch == "random_correct_reveal":
                candidates = np.flatnonzero(
                    original_test_masks[local_index]
                    & ~initial_error_masks[local_index]
                )
                count = min(len(errors), len(candidates))
                selected = (
                    rng.choice(candidates, size=count, replace=False)
                    if count else np.empty(0, dtype=np.int64)
                )
                augmented_masks[row, selected] = True
                revealed_masks[row, selected] = True
            elif branch == "full_reveal":
                augmented_masks[row] = True
                revealed_masks[row] = original_test_masks[local_index]

    child = clone_models_from_parents(
        parent_model, parent_global_indices, device
    )
    optimizer = clone_optimizer_from_parents(
        parent_optimizer,
        child,
        parent_model,
        parent_global_indices,
        device,
    )
    augmented_masks_tensor = torch.from_numpy(augmented_masks).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.BATCH_SEED + 2_000_003)

    max_initial_errors = max(1, int(initial_error_counts_parent.max()))
    priority_table = np.zeros(
        (len(branch_names), max_initial_errors), dtype=np.int64
    )
    priority_lengths = np.zeros(len(branch_names), dtype=np.int64)
    replay_branch_start = branches.index("error_replay") * parent_count
    for local_index in range(parent_count):
        errors = np.flatnonzero(initial_error_masks[local_index])
        row = replay_branch_start + local_index
        priority_table[row, : len(errors)] = errors
        priority_lengths[row] = len(errors)
    priority_table_tensor = torch.from_numpy(priority_table).to(device)
    priority_lengths_tensor = torch.from_numpy(priority_lengths).to(device)
    replay_rows = torch.nonzero(
        priority_lengths_tensor > 0, as_tuple=False
    ).flatten()
    replay_count = max(
        1, int(round(Config.TRAIN_BATCH_SIZE * Config.ERROR_REPLAY_FRACTION))
    )

    parent_conditions_local = np.asarray([
        conditions[model_condition_indices[index]].name
        for index in half_parent_global
    ])
    parent_seeds_local = model_seed_indices[half_parent_global]
    parent_conditions_repeated = np.tile(
        parent_conditions_local, len(branches)
    )
    parent_seeds_repeated = np.tile(parent_seeds_local, len(branches))
    initial_error_counts = initial_error_repeated.sum(axis=1)
    revealed_counts = revealed_masks.sum(axis=1)

    print("\n=== 当前错误点揭示干预 ===", flush=True)
    print(
        f"parents={parent_count} | branches={list(branches)} | "
        f"child models={child.model_count} | initial errors median="
        f"{np.median(initial_error_counts_parent):.1f}",
        flush=True,
    )

    eval_steps = {
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000,
        Config.ERROR_REVEAL_POST_STEPS,
    }
    aggregate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    interrupted = False
    final_step = 0
    started = time.perf_counter()
    try:
        for step in range(Config.ERROR_REVEAL_POST_STEPS + 1):
            if step in eval_steps:
                metrics = evaluate_error_reveal_models(
                    child,
                    all_inputs,
                    all_targets,
                    augmented_masks_tensor,
                    original_test_repeated,
                    initial_error_repeated,
                    revealed_masks,
                )
                current = aggregate_error_reveal_rows(
                    step, branch_names, metrics
                )
                aggregate_rows.extend(current)
                model_rows.extend(error_reveal_model_rows(
                    step,
                    branch_names,
                    parent_local_indices,
                    parent_conditions_repeated,
                    parent_seeds_repeated,
                    initial_error_counts,
                    revealed_counts,
                    metrics,
                ))
                compact = " | ".join(
                    f"{row['branch']}:old={row['remaining_initial_errors_mean']:.2f},"
                    f"new={row['new_error_count_mean']:.2f},"
                    f"exact={row['full_exact_fraction']:.1%}"
                    for row in current
                )
                print(
                    f"reveal step={step:>5,} | {compact} | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                write_csv(
                    intervention_dir / "intervention_trajectory.csv",
                    aggregate_rows,
                )
                write_csv(
                    intervention_dir / "intervention_model_trajectory.csv",
                    model_rows,
                )
            final_step = step
            if step == Config.ERROR_REVEAL_POST_STEPS:
                break
            batch_indices = sample_from_model_masks(
                augmented_masks_tensor,
                Config.TRAIN_BATCH_SIZE,
                generator,
            )
            if len(replay_rows):
                lengths = priority_lengths_tensor[replay_rows]
                random_values = torch.rand(
                    len(replay_rows),
                    replay_count,
                    generator=generator,
                    device=device,
                )
                positions = torch.floor(
                    random_values * lengths[:, None]
                ).long()
                replay_indices = priority_table_tensor[replay_rows].gather(
                    1, positions
                )
                batch_indices[replay_rows, :replay_count] = replay_indices
            losses = training_batch_loss(
                child, all_inputs, all_targets, batch_indices
            )
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            optimizer.step()
    except KeyboardInterrupt:
        interrupted = True
        print("\n错误点揭示实验收到中断，正在保存……", flush=True)

    if not any(row["step"] == final_step for row in aggregate_rows):
        metrics = evaluate_error_reveal_models(
            child,
            all_inputs,
            all_targets,
            augmented_masks_tensor,
            original_test_repeated,
            initial_error_repeated,
            revealed_masks,
        )
        aggregate_rows.extend(aggregate_error_reveal_rows(
            final_step, branch_names, metrics
        ))
        model_rows.extend(error_reveal_model_rows(
            final_step,
            branch_names,
            parent_local_indices,
            parent_conditions_repeated,
            parent_seeds_repeated,
            initial_error_counts,
            revealed_counts,
            metrics,
        ))

    final_rows = [
        row for row in aggregate_rows if row["step"] == final_step
    ]
    write_csv(
        intervention_dir / "intervention_trajectory.csv", aggregate_rows
    )
    write_csv(
        intervention_dir / "intervention_model_trajectory.csv", model_rows
    )
    write_csv(intervention_dir / "final_branch_summary.csv", final_rows)
    plot_error_reveal_results(intervention_dir, aggregate_rows)
    summary = {
        "status": "interrupted" if interrupted else "completed",
        "pretrain_step": Config.MAX_STEPS,
        "post_step": final_step,
        "parent_model_count": parent_count,
        "initial_error_count_mean": float(initial_error_counts_parent.mean()),
        "initial_error_count_median": float(
            np.median(initial_error_counts_parent)
        ),
        "branches": branches,
        "error_replay_fraction": Config.ERROR_REPLAY_FRACTION,
        "final_branch_summary": final_rows,
        "interpretation": {
            "targeted_beats_random": (
                "最后残差主要来自训练集缺少对应判别约束。"
            ),
            "old_errors_fall_new_errors_rise": (
                "新增样本修复局部缺陷，但 endpoint 动力学把错误迁移到其他未见点。"
            ),
            "replay_or_full_recovers_exact": (
                "精确 parity 在当前权重附近可达；此前失败不是容量问题。"
            ),
        },
    }
    write_json(intervention_dir / "summary.json", summary)
    return summary, interrupted


def plot_results(
    output_dir: Path,
    trajectory_rows: Sequence[dict[str, Any]],
    final_rows: Sequence[dict[str, Any]],
    model_trajectory_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", flush=True)
        return

    half = [row for row in trajectory_rows if row["group"] == "half_train"]
    control = [
        row for row in trajectory_rows if row["group"] == "full_control"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].plot(
        [row["step"] for row in half],
        [row["train_exact_fraction"] for row in half],
        label="half train exact",
    )
    axes[0].plot(
        [row["step"] for row in control],
        [row["train_exact_fraction"] for row in control],
        label="full control exact",
    )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("Training reachability")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        [row["step"] for row in half],
        [row["test_accuracy_mean"] for row in half],
        label="mean test accuracy",
    )
    axes[1].plot(
        [row["step"] for row in half],
        [row["test_exact_fraction"] for row in half],
        label="test exact fraction",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Generalization")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    final_half = [row for row in final_rows if not row["is_full_control"]]
    axes[2].hist(
        [float(row["test_accuracy"]) for row in final_half],
        bins=20,
        range=(0.0, 1.0),
    )
    axes[2].set_xlabel("final test accuracy")
    axes[2].set_ylabel("models")
    axes[2].set_title("Final function distribution")
    axes[2].grid(alpha=0.25)

    figure.savefig(
        output_dir / f"parity{Config.INPUT_BITS}_half_space_generalization.png",
        dpi=180,
    )
    plt.close(figure)

    half_paths = [
        row for row in model_trajectory_rows if not row["is_full_control"]
    ]
    if half_paths:
        figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        for (_, _), rows in _group_rows(half_paths, ("condition", "seed")):
            rows = sorted(rows, key=lambda item: int(item["step"]))
            axis.plot(
                [max(float(row["train_loss"]), 1e-12) for row in rows],
                [float(row["test_accuracy"]) for row in rows],
                color="tab:blue",
                alpha=0.10,
                linewidth=1,
            )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_ylim(0.0, 1.01)
        axis.set_xlabel("train raw BCE (deeper to the right)")
        axis.set_ylabel("test accuracy")
        axis.set_title("Per-model loss to generalization trajectories")
        axis.grid(alpha=0.25)
        figure.savefig(
            output_dir / "parity_loss_generalization_paths.png", dpi=180
        )
        plt.close(figure)


def _group_rows(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    return list(groups.items())


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    output_dir = prepare_result_dir()
    write_json(output_dir / "config.json", config_dict())
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE 要求 CUDA，但 PyTorch 看不到 GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    inputs_np = truth_table_inputs(Config.INPUT_BITS)
    labels_np = parity_targets(inputs_np)
    conditions = build_conditions()
    condition_masks_np = build_balanced_split_masks(labels_np, conditions)
    model_condition_indices, model_seed_indices = build_model_layout(conditions)
    model_masks_np = condition_masks_np[model_condition_indices]
    model_is_control_np = np.asarray([
        conditions[index].is_full_control
        for index in model_condition_indices
    ])

    np.savez_compressed(
        output_dir / "split_masks.npz",
        condition_masks=condition_masks_np,
        model_condition_indices=model_condition_indices,
        model_seed_indices=model_seed_indices,
    )
    write_json(output_dir / "conditions.json", [
        {
            "index": condition.index,
            "name": condition.name,
            "is_full_control": condition.is_full_control,
            "seeds": condition.seeds,
            "train_count": int(condition_masks_np[condition.index].sum()),
            "test_count": int((~condition_masks_np[condition.index]).sum()),
            "train_zero_count": int(np.sum(
                condition_masks_np[condition.index] & (labels_np == 0)
            )),
            "train_one_count": int(np.sum(
                condition_masks_np[condition.index] & (labels_np == 1)
            )),
        }
        for condition in conditions
    ])

    all_inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    all_targets = torch.from_numpy(labels_np.astype(np.float32)).to(device)
    model_train_masks = torch.from_numpy(model_masks_np).to(device)
    model_condition_tensor = torch.from_numpy(model_condition_indices).to(device)
    model_is_control = torch.from_numpy(model_is_control_np).to(device)
    condition_train_indices = [
        torch.from_numpy(np.flatnonzero(mask).astype(np.int64))
        for mask in condition_masks_np
    ]

    model = BatchedIndependentMLP(
        Config.INPUT_BITS, model_seed_indices, device
    )
    optimizer = make_optimizer(model)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(Config.BATCH_SEED)
    thresholds = torch.tensor(
        Config.RAW_BCE_THRESHOLDS,
        dtype=torch.float32,
        device=device,
    )
    crossing_shape = (model.model_count, len(Config.RAW_BCE_THRESHOLDS))
    crossed = torch.zeros(crossing_shape, dtype=torch.bool, device=device)
    crossing_step = torch.full(
        crossing_shape, -1, dtype=torch.int32, device=device
    )
    crossing_test_accuracy = torch.full(
        crossing_shape, float("nan"), dtype=torch.float32, device=device
    )
    crossing_test_exact = torch.zeros(
        crossing_shape, dtype=torch.bool, device=device
    )

    parameters_per_model = (
        sum(parameter.numel() for parameter in model.parameters())
        // model.model_count
    )
    print(
        f"=== {Config.INPUT_BITS}-bit parity 50% train / 50% test ===",
        flush=True,
    )
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"states={len(inputs_np):,} | train/test="
        f"{int(condition_masks_np[0].sum()):,}/"
        f"{int((~condition_masks_np[0]).sum()):,} | "
        f"splits={Config.SPLIT_COUNT} | seeds/split="
        f"{Config.SEEDS_PER_CONDITION} | models={model.model_count}",
        flush=True,
    )
    print(
        f"MLP: {Config.INPUT_BITS} -> {Config.WIDTH} x "
        f"{Config.HIDDEN_LAYERS} -> 1 | "
        f"params/model={parameters_per_model:,} | max_steps="
        f"{Config.MAX_STEPS:,}",
        flush=True,
    )
    print(f"结果目录：{output_dir.resolve()}", flush=True)

    eval_steps = {
        0, 1, 2, 5, 10, 20, 50, 100, 200,
        *range(Config.EVAL_INTERVAL, Config.MAX_STEPS + 1, Config.EVAL_INTERVAL),
        Config.MAX_STEPS,
    }
    trajectory_rows: list[dict[str, Any]] = []
    model_trajectory_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    interrupted = False
    final_step = 0

    try:
        for step in range(Config.MAX_STEPS + 1):
            if step in eval_steps:
                evaluation = evaluate_models(
                    model, all_inputs, all_targets, model_train_masks
                )
                update_crossings(
                    step,
                    evaluation,
                    thresholds,
                    crossed,
                    crossing_step,
                    crossing_test_accuracy,
                    crossing_test_exact,
                )
                current = aggregate_evaluation_rows(
                    step, evaluation, model_is_control
                )
                trajectory_rows.extend(current)
                model_trajectory_rows.extend(model_evaluation_rows(
                    step,
                    conditions,
                    model_condition_indices,
                    model_seed_indices,
                    evaluation,
                ))
                half = next(row for row in current if row["group"] == "half_train")
                control = next(
                    row for row in current if row["group"] == "full_control"
                )
                if step <= 200 or step % Config.LOG_INTERVAL == 0:
                    print(
                        f"step={step:>7,} | "
                        f"half loss={half['train_loss_median']:.2e} "
                        f"train-exact={half['train_exact_fraction']:.1%} "
                        f"test={half['test_accuracy_mean']:.3%} "
                        f"test-exact={half['test_exact_fraction']:.1%} "
                        f"full-fn={half['full_parity_exact_fraction']:.1%} | "
                        f"control-exact={control['train_exact_fraction']:.1%} | "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                write_csv(output_dir / "trajectory.csv", trajectory_rows)
                write_csv(
                    output_dir / "model_trajectory.csv",
                    model_trajectory_rows,
                )

            final_step = step
            if step == Config.MAX_STEPS:
                break
            model_batch_indices = sample_modelwise_batch_indices(
                condition_train_indices,
                model_condition_tensor,
                batch_generator,
                device,
            )
            losses = training_batch_loss(
                model,
                all_inputs,
                all_targets,
                model_batch_indices,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            optimizer.step()
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在保存当前结果……", flush=True)

    evaluation = evaluate_models(
        model, all_inputs, all_targets, model_train_masks
    )
    final_current = aggregate_evaluation_rows(
        final_step, evaluation, model_is_control
    )
    if not any(row["step"] == final_step for row in trajectory_rows):
        trajectory_rows.extend(final_current)
        model_trajectory_rows.extend(model_evaluation_rows(
            final_step,
            conditions,
            model_condition_indices,
            model_seed_indices,
            evaluation,
        ))

    crossing_rows = summarize_crossings(
        thresholds=Config.RAW_BCE_THRESHOLDS,
        model_is_control=model_is_control_np,
        crossing_step=crossing_step.cpu().numpy(),
        crossing_test_accuracy=crossing_test_accuracy.cpu().numpy(),
        crossing_test_exact=crossing_test_exact.cpu().numpy(),
    )
    model_rows = final_model_rows(
        conditions,
        model_condition_indices,
        model_seed_indices,
        evaluation,
    )
    write_csv(output_dir / "trajectory.csv", trajectory_rows)
    write_csv(output_dir / "model_trajectory.csv", model_trajectory_rows)
    write_csv(output_dir / "loss_crossing_summary.csv", crossing_rows)
    write_csv(output_dir / "final_model_summary.csv", model_rows)
    write_csv(output_dir / "final_group_summary.csv", final_current)
    np.savez_compressed(
        output_dir / "crossing_state.npz",
        thresholds=np.asarray(Config.RAW_BCE_THRESHOLDS, dtype=np.float64),
        crossing_step=crossing_step.cpu().numpy(),
        crossing_test_accuracy=crossing_test_accuracy.cpu().numpy(),
        crossing_test_exact=crossing_test_exact.cpu().numpy(),
    )
    plot_results(
        output_dir,
        trajectory_rows,
        model_rows,
        model_trajectory_rows,
    )

    half_final = next(
        row for row in final_current if row["group"] == "half_train"
    )
    control_final = next(
        row for row in final_current if row["group"] == "full_control"
    )
    error_reveal_summary: dict[str, Any] | None = None
    error_reveal_interrupted = False
    if Config.RUN_ERROR_REVEAL_INTERVENTION and not interrupted:
        error_reveal_summary, error_reveal_interrupted = (
            run_error_reveal_intervention(
                output_dir,
                model,
                optimizer,
                all_inputs,
                all_targets,
                model_masks_np,
                model_condition_indices,
                model_seed_indices,
                conditions,
                model_is_control_np,
                device,
            )
        )
    symmetry_summary: dict[str, Any] | None = None
    symmetry_interrupted = False
    if (
        Config.RUN_SYMMETRY_HOLE_MATRIX
        and not interrupted
        and not error_reveal_interrupted
    ):
        symmetry_summary, symmetry_interrupted = run_symmetry_hole_matrix(
            output_dir,
            inputs_np,
            labels_np,
            all_inputs,
            all_targets,
            device,
        )
    summary = {
        "status": (
            "interrupted"
            if interrupted or error_reveal_interrupted or symmetry_interrupted
            else "completed"
        ),
        "final_step": final_step,
        "elapsed_seconds": time.perf_counter() - started,
        "state_count": len(inputs_np),
        "train_count": int(len(inputs_np) * Config.TRAIN_FRACTION),
        "test_count": int(len(inputs_np) * (1.0 - Config.TRAIN_FRACTION)),
        "model_count": model.model_count,
        "parameters_per_model": parameters_per_model,
        "half_train_final": half_final,
        "full_control_final": control_final,
        "error_reveal_intervention": error_reveal_summary,
        "symmetry_hole_matrix": symmetry_summary,
        "interpretation": {
            "high_test_low_exact": (
                "统计泛化很好，但多数模型仍未恢复完整 parity hard function。"
            ),
            "high_test_high_exact": (
                "随机半空间已经足以让多数模型恢复精确 parity。"
            ),
            "low_test": (
                "减少约束后，模型主要进入记忆/shortcut函数，未形成 parity 表示。"
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)

    archive_path: Path | None = None
    if Config.PACKAGE_RESULTS:
        archive_path = create_archive(output_dir)
    print("\n=== 实验结束 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}", flush=True)


if __name__ == "__main__":
    main()
