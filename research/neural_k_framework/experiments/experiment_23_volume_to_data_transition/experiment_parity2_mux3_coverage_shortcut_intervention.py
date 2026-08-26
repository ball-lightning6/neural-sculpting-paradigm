"""Parity2 / MUX3 的语义覆盖与捷径竞争干预实验。

问题：完整目标 Gaussian 体积在深低 loss 区强烈偏向 MUX3，但均匀随机
部分训练集上的操作性相变反而更早恢复 parity2。一个可检验解释是：

* parity2 只有4个相关真值表格点，同样 n 下每格平均得到 n/4 次证据；
* MUX3 有8个相关格点，每格平均仅 n/8 次；
* copy-x1 和 copy-x2 都能在 MUX3 上达到75%总体准确率，稀疏随机数据
  因而允许大体积的捷径延拓与真正 MUX3 竞争。

本实验固定 E23 的 8->16x2->1 tanh 网络、AdamW、配对初始化和完整函数
恢复判据，只干预训练输入的抽样协议：

* uniform_random：256个原始输入均匀无放回抽样；
* cell_balanced：八个(x0,x1,x2)格点轮转均衡抽样；该设计也自动均衡
  parity2 的四个(x0,x1)格点；
* conflict_enriched：MUX selector 冲突格点 x1!=x2 的抽样权重为其他
  格点的3倍，同时保持 x0、x1 和 parity2 四格的边缘平衡。

两个目标在三个协议中均被训练，所以 conflict_enriched 对 parity2 构成
负对照。脚本还按 n/相关格点数报告等效重复次数，以检验按语义覆盖
归一化后 parity2/MUX3 的相变差距是否缩小。支持按 n checkpoint 与
Ctrl-C 续跑。
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
import torch.nn as nn
import torch.nn.functional as F


class Config:
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2
    TARGET_NAMES = ("parity2", "mux3")
    EXPECTED_FUNCTION_HEX = {
        "parity2": "0x0000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000",
        "mux3": "0xFFFFFFFFFFFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFF00000000",
    }

    SAMPLING_PROTOCOLS = (
        "uniform_random",
        "cell_balanced",
        "conflict_enriched",
    )
    TRAIN_COUNTS = tuple(range(32, 113, 8))
    DATASETS_PER_N = 128
    MODEL_SEED_COUNT = 32
    DATASET_SEED = 2026082901
    INITIALIZATION_SEED = 2026082902

    OPTIMIZERS = (("adamw", 1e-3, 0.0),)
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 40_000
    EVAL_STEPS = (
        0, 100, 200, 500, 1_000, 2_000, 5_000,
        10_000, 20_000, 30_000, 40_000,
    )
    CHECKPOINT_EVERY_STEPS = 5_000

    MIN_TRAIN_FIT_RATE = 0.90
    TARGET_FUNCTION_MASS_THRESHOLD = 0.90
    FUNCTION_COLLISION_THRESHOLD = 0.80
    FULL_DATA_QUALIFICATION_TARGET_MASS = 0.90
    TRANSITION_LEVELS = (0.50, 0.90)
    AGREEMENT_THRESHOLDS = (0.95, 0.99, 0.995)
    AGREEMENT_ACCURACY_GUARD = 0.90
    BOOTSTRAP_REPLICATES = 2_000
    BOOTSTRAP_SEED = 2026082903

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path(
        "/root/results_parity2_mux3_coverage_shortcut_intervention"
    )
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


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
    n: int
    indices: tuple[int, ...]
    signature: str
    relevant_cell_count: int
    distinct_relevant_cells: int
    minimum_relevant_cell_count: int
    maximum_relevant_cell_count: int
    conflict_fraction: float
    copy_x1_accuracy: float
    copy_x2_accuracy: float


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.TRAIN_COUNTS = (8, 16)
    Config.DATASETS_PER_N = 3
    Config.MODEL_SEED_COUNT = 4
    Config.MAX_STEPS = 5
    Config.EVAL_STEPS = (0, 1, 2, 5)
    Config.CHECKPOINT_EVERY_STEPS = 2
    Config.MIN_TRAIN_FIT_RATE = 0.0
    Config.TARGET_FUNCTION_MASS_THRESHOLD = 0.0
    Config.FUNCTION_COLLISION_THRESHOLD = 0.0
    Config.BOOTSTRAP_REPLICATES = 20
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/neural_k_framework/experiments/"
        "experiment_23_volume_to_data_transition/"
        "_smoke_parity2_mux3_coverage_shortcut_intervention"
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
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
    shifts = np.arange(
        Config.INPUT_BITS - 1, -1, -1, dtype=np.uint16
    )
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
    targets = []
    for index, name in enumerate(Config.TARGET_NAMES):
        outputs, formula = raw[name]
        function_hex = outputs_to_hex(outputs)
        if function_hex != Config.EXPECTED_FUNCTION_HEX[name]:
            raise AssertionError(f"{name} 真值表与原 E23 不一致。")
        targets.append(TargetSpec(
            index=index,
            name=name,
            formula=formula,
            outputs=tuple(map(int, outputs)),
            function_hex=function_hex,
        ))
    return targets


def mux_cell_ids(inputs: np.ndarray) -> np.ndarray:
    return (
        4 * inputs[:, 0] + 2 * inputs[:, 1] + inputs[:, 2]
    ).astype(np.int64)


def target_cell_ids(target_name: str, inputs: np.ndarray) -> np.ndarray:
    if target_name == "parity2":
        return (2 * inputs[:, 0] + inputs[:, 1]).astype(np.int64)
    if target_name == "mux3":
        return mux_cell_ids(inputs)
    raise ValueError(target_name)


def interleaved_order(
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
    positions = {cell: 0 for cell in range(8)}
    order: list[int] = []

    if protocol == "cell_balanced":
        while len(order) < len(inputs):
            schedule = list(map(int, rng.permutation(8)))
            for cell in schedule:
                position = positions[cell]
                if position < len(queues[cell]):
                    order.append(queues[cell][position])
                    positions[cell] += 1
        return np.asarray(order, dtype=np.int64)

    if protocol != "conflict_enriched":
        raise ValueError(protocol)

    # x1!=x2 的四格各出现3次，其余四格各出现1次。到最大训练
    # 样本数为止保持3:1设计；剩余状态只用于补成完整排列。
    maximum_n = max(Config.TRAIN_COUNTS)
    while len(order) < maximum_n:
        doubled_first = set(map(int, rng.choice(4, size=2, replace=False)))
        for doubled in (doubled_first, set(range(4)) - doubled_first):
            half_cycle: list[int] = []
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
                position = positions[cell]
                if position >= len(queues[cell]):
                    continue
                order.append(queues[cell][position])
                positions[cell] += 1
                if len(order) >= maximum_n:
                    break
            if len(order) >= maximum_n:
                break
    leftovers = [
        value
        for cell in range(8)
        for value in queues[cell][positions[cell]:]
    ]
    order.extend(map(int, rng.permutation(leftovers)))
    if len(order) != len(inputs) or len(set(order)) != len(inputs):
        raise AssertionError("抽样顺序不是256状态的完整排列。")
    return np.asarray(order, dtype=np.int64)


def build_sampling_orders() -> dict[str, np.ndarray]:
    inputs = truth_table_inputs()
    return {
        protocol: np.stack([
            interleaved_order(protocol, dataset_index, inputs)
            for dataset_index in range(Config.DATASETS_PER_N)
        ])
        for protocol in Config.SAMPLING_PROTOCOLS
    }


def make_condition(
    index: int,
    target: TargetSpec,
    protocol: str,
    dataset_index: int,
    indices: tuple[int, ...],
) -> ConditionSpec:
    inputs = truth_table_inputs()
    selected = np.asarray(indices, dtype=np.int64)
    cells = target_cell_ids(target.name, inputs[selected])
    relevant_cell_count = 4 if target.name == "parity2" else 8
    counts = np.bincount(cells, minlength=relevant_cell_count)
    labels = np.asarray(target.outputs, dtype=np.uint8)[selected]
    conflict = inputs[selected, 1] != inputs[selected, 2]
    signature_payload = (
        protocol.encode("ascii")
        + np.asarray(indices, dtype=np.uint16).tobytes()
    )
    return ConditionSpec(
        index=index,
        target_index=target.index,
        target_name=target.name,
        sampling_protocol=protocol,
        dataset_index=dataset_index,
        n=len(indices),
        indices=indices,
        signature=hashlib.sha256(signature_payload).hexdigest()[:16],
        relevant_cell_count=relevant_cell_count,
        distinct_relevant_cells=int(np.count_nonzero(counts)),
        minimum_relevant_cell_count=int(counts.min()),
        maximum_relevant_cell_count=int(counts.max()),
        conflict_fraction=float(conflict.mean()),
        copy_x1_accuracy=float(np.mean(inputs[selected, 1] == labels)),
        copy_x2_accuracy=float(np.mean(inputs[selected, 2] == labels)),
    )


def build_conditions(
    n: int,
    targets: Sequence[TargetSpec],
    sampling_orders: dict[str, np.ndarray],
) -> list[ConditionSpec]:
    result = []
    for protocol in Config.SAMPLING_PROTOCOLS:
        orders = sampling_orders[protocol]
        for target in targets:
            for dataset_index in range(Config.DATASETS_PER_N):
                # 训练次序不影响 full-batch 更新；排序后便于稳定哈希与补集。
                indices = tuple(sorted(map(int, orders[dataset_index, :n])))
                result.append(make_condition(
                    len(result), target, protocol, dataset_index, indices
                ))
    return result


def build_full_conditions(
    targets: Sequence[TargetSpec],
) -> list[ConditionSpec]:
    indices = tuple(range(2 ** Config.INPUT_BITS))
    return [
        make_condition(
            target.index, target, "full_truth_table", 0, indices
        )
        for target in targets
    ]


def prepare_output(targets: Sequence[TargetSpec]) -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for optimizer_name, _, _ in Config.OPTIMIZERS:
        (output / "by_optimizer" / optimizer_name / "by_n").mkdir(
            parents=True, exist_ok=True
        )
        (output / "by_optimizer" / optimizer_name / "qualification").mkdir(
            parents=True, exist_ok=True
        )
    protocol = {
        "protocol": "e23_parity2_mux3_coverage_shortcut_intervention_v1",
        "created_before_training": True,
        "config": config_payload(),
        "targets": [json_ready(target.__dict__) for target in targets],
        "primary_predictions": {
            "coverage": (
                "cell balancing reduces variance but does not by itself "
                "remove the twofold relevant-cell repetition gap"
            ),
            "shortcut": (
                "conflict enrichment moves mux3 target recovery and "
                "target-aligned reconcentration to smaller n more than it "
                "moves parity2"
            ),
            "semantic_repetition": (
                "plotting transitions against n/relevant_cell_count "
                "shrinks the parity2/mux3 discrepancy"
            ),
        },
        "original_e23": {
            "parity2": {"n50": 64, "n90": 80, "agreement99": 59.98},
            "mux3": {"n50": 80, "n90": 112, "agreement99": 69.47},
        },
        "sampling": {
            "pairing": "paired targets, protocols, dataset ids, and seeds",
            "nesting": "one protocol-specific order; nested prefixes over n",
            "uniform_random": "uniform without replacement over 256 states",
            "cell_balanced": "round-robin over eight x0,x1,x2 cells",
            "conflict_enriched": (
                "3:1 weight for x1!=x2 versus x1==x2 cells through max n"
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
            raise RuntimeError("结果目录存在且 RESUME=False。")
    else:
        write_json(path, protocol)
    return output


class PairedBatchedMLP(nn.Module):
    def __init__(self, condition_count: int) -> None:
        super().__init__()
        dimensions = (
            [Config.INPUT_BITS]
            + [Config.WIDTH] * Config.HIDDEN_LAYERS
            + [1]
        )
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer_index, (fan_in, fan_out) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            base_weights = []
            base_biases = []
            for seed_index in range(Config.MODEL_SEED_COUNT):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    Config.INITIALIZATION_SEED
                    + 100_003 * seed_index
                    + 10_007 * layer_index
                )
                bound = 1.0 / math.sqrt(fan_in)
                base_weights.append(torch.empty(
                    fan_out, fan_in
                ).uniform_(-bound, bound, generator=generator))
                base_biases.append(torch.empty(fan_out).uniform_(
                    -bound, bound, generator=generator
                ))
            weights = torch.stack(base_weights)
            biases = torch.stack(base_biases)
            weights = weights[None].expand(
                condition_count, -1, -1, -1
            ).reshape(
                condition_count * Config.MODEL_SEED_COUNT,
                fan_out,
                fan_in,
            )
            biases = biases[None].expand(
                condition_count, -1, -1
            ).reshape(
                condition_count * Config.MODEL_SEED_COUNT,
                fan_out,
            )
            self.weights.append(nn.Parameter(weights.clone()))
            self.biases.append(nn.Parameter(biases.clone()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def make_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=Config.WEIGHT_DECAY,
        )
    if optimizer_name in {"full_batch_sgd", "momentum_sgd"}:
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=Config.WEIGHT_DECAY,
        )
    raise ValueError(f"未知 optimizer：{optimizer_name}")


def parameter_count() -> int:
    dims = [Config.INPUT_BITS] + [Config.WIDTH] * 2 + [1]
    return sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(dims[:-1], dims[1:])
    )


def build_tensors(
    conditions: Sequence[ConditionSpec],
    targets: Sequence[TargetSpec],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    inputs = truth_table_inputs().astype(np.float32)
    target_matrix = np.asarray([
        target.outputs for target in targets
    ], dtype=np.uint8)
    n = conditions[0].n
    train_x = np.empty(
        (len(conditions), n, Config.INPUT_BITS), dtype=np.float32
    )
    train_y = np.empty((len(conditions), n), dtype=np.float32)
    full_targets = np.empty(
        (len(conditions), 2 ** Config.INPUT_BITS), dtype=np.uint8
    )
    for local, condition in enumerate(conditions):
        indices = np.asarray(condition.indices, dtype=np.int64)
        outputs = target_matrix[condition.target_index]
        train_x[local] = inputs[indices]
        train_y[local] = outputs[indices]
        full_targets[local] = outputs
    return (
        torch.as_tensor(np.repeat(
            train_x, Config.MODEL_SEED_COUNT, axis=0
        ), device=device),
        torch.as_tensor(np.repeat(
            train_y, Config.MODEL_SEED_COUNT, axis=0
        ), device=device),
        torch.as_tensor(inputs, device=device),
        full_targets,
    )


def collision_probability(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total < 2:
        return float("nan")
    return float(
        np.sum(counts.astype(np.float64) * (counts - 1))
        / (total * (total - 1))
    )


def bit_agreement(predictions: np.ndarray, indices: np.ndarray) -> float:
    local = predictions[:, indices]
    count = len(local)
    if count < 2 or not len(indices):
        return float("nan")
    ones = local.sum(axis=0).astype(np.float64)
    same = (
        ones * (ones - 1)
        + (count - ones) * (count - ones - 1)
    )
    return float(np.mean(same / (count * (count - 1))))


@torch.no_grad()
def evaluate(
    step: int,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    model: PairedBatchedMLP,
    conditions: Sequence[ConditionSpec],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    full_inputs: torch.Tensor,
    full_targets: np.ndarray,
) -> list[dict[str, Any]]:
    model.eval()
    train_logits = model(train_x)
    train_losses = F.binary_cross_entropy_with_logits(
        train_logits, train_y, reduction="none"
    ).mean(dim=1)
    fitted = torch.all(
        (train_logits >= 0) == (train_y >= 0.5), dim=1
    )
    full_batch = full_inputs[None].expand(len(train_x), -1, -1)
    predictions = (model(full_batch) >= 0).to(torch.uint8)

    condition_count = len(conditions)
    seeds = Config.MODEL_SEED_COUNT
    losses_np = train_losses.cpu().numpy().reshape(condition_count, seeds)
    fitted_np = fitted.cpu().numpy().reshape(condition_count, seeds)
    predictions_np = predictions.cpu().numpy().reshape(
        condition_count, seeds, -1
    )
    all_indices = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    truth_inputs = truth_table_inputs()
    copy_x1_function = truth_inputs[:, 1].astype(np.uint8)
    copy_x2_function = truth_inputs[:, 2].astype(np.uint8)
    rows = []
    for local, condition in enumerate(conditions):
        local_fitted = fitted_np[local]
        cohort = predictions_np[local][local_fitted]
        source = "train_hard_exact_models"
        if not len(cohort):
            cohort = predictions_np[local]
            source = "all_models_no_fitted_cohort"
        packed = np.packbits(cohort, axis=1, bitorder="little")
        unique, counts = np.unique(packed, axis=0, return_counts=True)
        order = np.argsort(-counts)
        unique, counts = unique[order], counts[order]
        target_packed = np.packbits(
            full_targets[local], bitorder="little"
        )
        target_matches = np.all(unique == target_packed[None], axis=1)
        target_count = int(counts[target_matches].sum())
        target_mass = target_count / int(counts.sum())
        modal_is_target = bool(np.array_equal(unique[0], target_packed))
        modal_prediction = np.unpackbits(
            unique[0], bitorder="little"
        )[: 2 ** Config.INPUT_BITS]
        collision = collision_probability(counts)
        train_indices = np.asarray(condition.indices, dtype=np.int64)
        unseen = np.setdiff1d(all_indices, train_indices, assume_unique=True)
        unseen_agreement = bit_agreement(cohort, unseen)
        unseen_accuracy = (
            float(np.mean(
                cohort[:, unseen] == full_targets[local, unseen][None]
            )) if len(unseen) else float("nan")
        )
        full_accuracy = float(np.mean(
            cohort == full_targets[local][None]
        ))
        fit_rate = float(local_fitted.mean())
        recovered = bool(
            fit_rate >= Config.MIN_TRAIN_FIT_RATE
            and target_mass >= Config.TARGET_FUNCTION_MASS_THRESHOLD
            and modal_is_target
            and collision >= Config.FUNCTION_COLLISION_THRESHOLD
        )
        rows.append({
            "step": step,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "condition_index": condition.index,
            "target_index": condition.target_index,
            "target_name": condition.target_name,
            "sampling_protocol": condition.sampling_protocol,
            "dataset_index": condition.dataset_index,
            "dataset_signature": condition.signature,
            "n": condition.n,
            "relevant_cell_count": condition.relevant_cell_count,
            "semantic_repetitions": (
                condition.n / condition.relevant_cell_count
            ),
            "distinct_relevant_cells": condition.distinct_relevant_cells,
            "minimum_relevant_cell_count": (
                condition.minimum_relevant_cell_count
            ),
            "maximum_relevant_cell_count": (
                condition.maximum_relevant_cell_count
            ),
            "conflict_fraction": condition.conflict_fraction,
            "copy_x1_accuracy": condition.copy_x1_accuracy,
            "copy_x2_accuracy": condition.copy_x2_accuracy,
            "cohort_copy_x1_full_agreement_mean": float(np.mean(
                cohort == copy_x1_function[None]
            )),
            "cohort_copy_x2_full_agreement_mean": float(np.mean(
                cohort == copy_x2_function[None]
            )),
            "modal_copy_x1_full_agreement": float(np.mean(
                modal_prediction == copy_x1_function
            )),
            "modal_copy_x2_full_agreement": float(np.mean(
                modal_prediction == copy_x2_function
            )),
            "seed_count": seeds,
            "cohort_count": len(cohort),
            "cohort_source": source,
            "train_fit_rate": fit_rate,
            "train_loss_mean": float(losses_np[local].mean()),
            "train_loss_median": float(np.median(losses_np[local])),
            "target_function_mass": target_mass,
            "modal_is_target": modal_is_target,
            "modal_function_packed_hex": bytes(unique[0]).hex().upper(),
            "modal_probability": float(counts[0] / counts.sum()),
            "function_collision": collision,
            "unique_function_count": len(unique),
            "unseen_bit_agreement": unseen_agreement,
            "unseen_target_bit_accuracy_mean": unseen_accuracy,
            "full_target_bit_accuracy_mean": full_accuracy,
            "dataset_recovered": recovered,
        })
    return rows


def checkpoint_path(directory: Path) -> Path:
    return directory / "checkpoint.pt"


def save_checkpoint(
    directory: Path,
    step: int,
    model: PairedBatchedMLP,
    optimizer: torch.optim.Optimizer,
    trajectory: Sequence[dict[str, Any]],
) -> None:
    temporary = directory / "checkpoint.tmp.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trajectory": list(trajectory),
        "config": config_payload(),
    }, temporary)
    destination = checkpoint_path(directory)
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def run_n(
    output: Path,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    n: int,
    conditions: Sequence[ConditionSpec],
    targets: Sequence[TargetSpec],
    device: torch.device,
    qualification: bool = False,
) -> str:
    directory = (
        output / "by_optimizer" / optimizer_name / "qualification"
        if qualification else
        output / "by_optimizer" / optimizer_name / "by_n" / f"n_{n:03d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "final.csv"
    if final_path.exists():
        print(f"optimizer={optimizer_name} n={n} 已完成，跳过。", flush=True)
        return "completed"

    model = PairedBatchedMLP(len(conditions)).to(device)
    train_x, train_y, full_inputs, full_targets = build_tensors(
        conditions, targets, device
    )
    optimizer = make_optimizer(
        model, optimizer_name, learning_rate, momentum
    )
    trajectory = []
    start_step = 0
    checkpoint = checkpoint_path(directory)
    if checkpoint.exists() and Config.RESUME:
        payload = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        trajectory = list(payload["trajectory"])
        start_step = int(payload["step"])
        print(
            f"optimizer={optimizer_name} n={n} resume step={start_step:,}",
            flush=True,
        )

    evaluated = {int(row["step"]) for row in trajectory}
    eval_steps = set(Config.EVAL_STEPS) | {Config.MAX_STEPS}
    current_step = start_step
    last_rows = []
    started = time.perf_counter()
    try:
        for step in range(start_step, Config.MAX_STEPS + 1):
            current_step = step
            if step in eval_steps and step not in evaluated:
                last_rows = evaluate(
                    step, optimizer_name, learning_rate, momentum,
                    model, conditions, train_x, train_y,
                    full_inputs, full_targets,
                )
                trajectory.extend(last_rows)
                write_csv(directory / "trajectory.csv", trajectory)
                compact = []
                protocols = (
                    ("full_truth_table",) if qualification
                    else Config.SAMPLING_PROTOCOLS
                )
                for protocol in protocols:
                    for target in targets:
                        local = [
                            row for row in last_rows
                            if row["target_name"] == target.name
                            and row["sampling_protocol"] == protocol
                        ]
                        if not local:
                            continue
                        compact.append(
                            f"{protocol[:4]}/{target.name}:F="
                            f"{np.mean([row['train_fit_rate'] for row in local]):.3f},"
                            f"R="
                            f"{np.mean([row['dataset_recovered'] for row in local]):.3f},"
                            f"A={finite_mean([row['unseen_bit_agreement'] for row in local]):.3f},"
                            f"U={finite_mean([row['unseen_target_bit_accuracy_mean'] for row in local]):.3f},"
                            f"T={finite_mean([row['target_function_mass'] for row in local]):.3f}"
                        )
                print(
                    f"optimizer={optimizer_name:<14} n={n:>3} "
                    f"step={step:>6,} | "
                    + " | ".join(compact)
                    + f" | elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
                evaluated.add(step)
            if step == Config.MAX_STEPS:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            per_model = F.binary_cross_entropy_with_logits(
                logits, train_y, reduction="none"
            ).mean(dim=1)
            per_model.sum().backward()
            optimizer.step()
            if (step + 1) % Config.CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(
                    directory, step + 1, model, optimizer, trajectory
                )
    except KeyboardInterrupt:
        save_checkpoint(
            directory, current_step, model, optimizer, trajectory
        )
        print(
            f"optimizer={optimizer_name} n={n} 已保存中断 checkpoint。",
            flush=True,
        )
        return "interrupted"

    if not last_rows or int(last_rows[0]["step"]) != Config.MAX_STEPS:
        last_rows = evaluate(
            Config.MAX_STEPS, optimizer_name, learning_rate, momentum,
            model, conditions, train_x, train_y,
            full_inputs, full_targets,
        )
    write_csv(final_path, last_rows)
    checkpoint.unlink(missing_ok=True)
    del model, optimizer, train_x, train_y, full_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return "completed"


def load_final_rows(output: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted((output / "by_optimizer").glob("*/by_n/n_*/final.csv")):
        for row in read_csv(path):
            converted: dict[str, Any] = dict(row)
            for key in (
                "step", "condition_index", "target_index",
                "dataset_index", "n", "seed_count", "cohort_count",
                "unique_function_count", "relevant_cell_count",
                "distinct_relevant_cells", "minimum_relevant_cell_count",
                "maximum_relevant_cell_count",
            ):
                converted[key] = int(float(converted[key]))
            for key in (
                "train_fit_rate", "train_loss_mean", "train_loss_median",
                "learning_rate", "momentum",
                "target_function_mass", "modal_probability",
                "function_collision", "unseen_bit_agreement",
                "unseen_target_bit_accuracy_mean",
                "full_target_bit_accuracy_mean",
                "semantic_repetitions", "conflict_fraction",
                "copy_x1_accuracy", "copy_x2_accuracy",
                "cohort_copy_x1_full_agreement_mean",
                "cohort_copy_x2_full_agreement_mean",
                "modal_copy_x1_full_agreement",
                "modal_copy_x2_full_agreement",
            ):
                converted[key] = float(converted[key])
            for key in ("modal_is_target", "dataset_recovered"):
                converted[key] = str(converted[key]).lower() == "true"
            result.append(converted)
    return result


def qualification_result(
    output: Path, optimizer_name: str
) -> dict[str, Any]:
    path = (
        output / "by_optimizer" / optimizer_name
        / "qualification" / "final.csv"
    )
    raw = read_csv(path)
    target_rows = []
    for row in raw:
        target_rows.append({
            "target_name": row["target_name"],
            "train_fit_rate": float(row["train_fit_rate"]),
            "target_function_mass": float(row["target_function_mass"]),
            "modal_is_target": str(row["modal_is_target"]).lower() == "true",
            "function_collision": float(row["function_collision"]),
        })
    passed = bool(Config.SMOKE_TEST or (
        len(target_rows) == len(Config.TARGET_NAMES)
        and all(
            row["train_fit_rate"] >= Config.MIN_TRAIN_FIT_RATE
            and row["target_function_mass"]
            >= Config.FULL_DATA_QUALIFICATION_TARGET_MASS
            and row["modal_is_target"]
            for row in target_rows
        )
    ))
    return {
        "optimizer_name": optimizer_name,
        "passed": passed,
        "targets": target_rows,
    }


def pava(values: np.ndarray) -> np.ndarray:
    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(values):
        blocks.append({
            "start": index, "stop": index + 1,
            "weight": 1.0, "sum": float(value),
        })
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if (
                float(left["sum"]) / float(left["weight"])
                <= float(right["sum"]) / float(right["weight"])
            ):
                break
            blocks[-2:] = [{
                "start": int(left["start"]),
                "stop": int(right["stop"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "sum": float(left["sum"]) + float(right["sum"]),
            }]
    result = np.empty(len(values), dtype=np.float64)
    for block in blocks:
        result[int(block["start"]):int(block["stop"])] = (
            float(block["sum"]) / float(block["weight"])
        )
    return result


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def interpolate_missing(values: np.ndarray, fallback: float = 0.5) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(result)
    if not finite.any():
        result.fill(fallback)
        return result
    indices = np.arange(len(result))
    result[~finite] = np.interp(indices[~finite], indices[finite], result[finite])
    return result


def first_grid_crossing(
    counts: np.ndarray, values: np.ndarray, threshold: float
) -> int | None:
    indices = np.flatnonzero(values >= threshold)
    return int(counts[indices[0]]) if len(indices) else None


def interpolate_crossing(
    counts: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    if not len(indices):
        return None
    index = int(indices[0])
    if index == 0:
        return float(counts[0])
    x0, x1 = counts[index - 1], counts[index]
    y0, y1 = values[index - 1], values[index]
    if y1 <= y0 + 1e-15:
        return float(x1)
    return float(x0 + (x1 - x0) * (threshold - y0) / (y1 - y0))


def curve_from_rows(
    rows: Sequence[dict[str, Any]],
    optimizer_name: str,
    target_name: str,
    sampling_protocol: str,
    selected_datasets: np.ndarray | None = None,
) -> dict[str, Any]:
    local = [
        row for row in rows
        if row["optimizer_name"] == optimizer_name
        and row["target_name"] == target_name
        and row["sampling_protocol"] == sampling_protocol
    ]
    counts = np.asarray(sorted({int(row["n"]) for row in local}))
    recovery, agreement, accuracy, target_mass = [], [], [], []
    for n in counts:
        by_n = [row for row in local if int(row["n"]) == int(n)]
        mapping = {int(row["dataset_index"]): row for row in by_n}
        indices = (
            np.asarray(sorted(mapping))
            if selected_datasets is None else selected_datasets
        )
        chosen = [mapping[int(index)] for index in indices]
        recovery.append(np.mean([row["dataset_recovered"] for row in chosen]))
        agreement.append(finite_mean([
            row["unseen_bit_agreement"] for row in chosen
        ]))
        accuracy.append(finite_mean([
            row["unseen_target_bit_accuracy_mean"] for row in chosen
        ]))
        target_mass.append(finite_mean([
            row["target_function_mass"] for row in chosen
        ]))
    agreement = interpolate_missing(np.asarray(agreement))
    accuracy = interpolate_missing(np.asarray(accuracy))
    recovery_iso = pava(np.asarray(recovery))
    minimum = int(np.argmin(agreement))
    agreement_iso = pava(np.asarray(agreement)[minimum:])
    accuracy_iso = pava(np.asarray(accuracy)[minimum:])
    branch_counts = counts[minimum:]
    transitions = {
        f"n{int(level*100)}": first_grid_crossing(
            counts, recovery_iso, level
        )
        for level in Config.TRANSITION_LEVELS
    }
    agreement_crossings = {}
    accuracy_cross = interpolate_crossing(
        branch_counts, accuracy_iso, Config.AGREEMENT_ACCURACY_GUARD
    )
    for threshold in Config.AGREEMENT_THRESHOLDS:
        agreement_cross = interpolate_crossing(
            branch_counts, agreement_iso, threshold
        )
        if agreement_cross is None or accuracy_cross is None:
            combined = None
        else:
            combined = max(agreement_cross, accuracy_cross)
        agreement_crossings[f"agreement_{threshold:g}"] = combined
    return {
        "counts": counts,
        "recovery": np.asarray(recovery),
        "recovery_iso": recovery_iso,
        "agreement": np.asarray(agreement),
        "accuracy": np.asarray(accuracy),
        "target_mass": np.asarray(target_mass),
        "minimum_agreement_n": int(counts[minimum]),
        "transitions": transitions,
        "agreement_crossings": agreement_crossings,
    }


def bootstrap_summary(
    rows: Sequence[dict[str, Any]],
    optimizer_name: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combinations = [
        (protocol, name)
        for protocol in Config.SAMPLING_PROTOCOLS
        for name in Config.TARGET_NAMES
    ]
    point = {
        key: curve_from_rows(rows, optimizer_name, key[1], key[0])
        for key in combinations
    }
    metric_keys = ("n50", "n90", "agreement_0.99")
    samples = {
        key: {metric: [] for metric in metric_keys}
        for key in combinations
    }
    dataset_indices = np.arange(Config.DATASETS_PER_N)
    for _ in range(Config.BOOTSTRAP_REPLICATES):
        selected = rng.choice(
            dataset_indices, size=len(dataset_indices), replace=True
        )
        curves = {
            key: curve_from_rows(
                rows, optimizer_name, key[1], key[0], selected
            )
            for key in combinations
        }
        for combination, curve in curves.items():
            for metric in ("n50", "n90"):
                value = curve["transitions"][metric]
                samples[combination][metric].append(
                    float(value) if value is not None else math.inf
                )
            value = curve["agreement_crossings"]["agreement_0.99"]
            samples[combination]["agreement_0.99"].append(
                float(value) if value is not None else math.inf
            )

    summary_rows = []
    for protocol, name in combinations:
        key = (protocol, name)
        relevant_cells = 4 if name == "parity2" else 8
        row: dict[str, Any] = {
            "optimizer_name": optimizer_name,
            "sampling_protocol": protocol,
            "target_name": name,
            "relevant_cell_count": relevant_cells,
            "minimum_agreement_n": point[key]["minimum_agreement_n"],
            **point[key]["transitions"],
            **point[key]["agreement_crossings"],
        }
        for metric, values in samples[key].items():
            finite = np.asarray([v for v in values if math.isfinite(v)])
            row[f"{metric}_bootstrap_median"] = (
                float(np.median(finite)) if len(finite) else None
            )
            row[f"{metric}_bootstrap_q025"] = (
                float(np.quantile(finite, 0.025)) if len(finite) else None
            )
            row[f"{metric}_bootstrap_q975"] = (
                float(np.quantile(finite, 0.975)) if len(finite) else None
            )
            row[f"{metric}_right_censored_fraction"] = float(
                1.0 - len(finite) / len(values)
            )
            point_value = (
                point[key]["transitions"].get(metric)
                if metric in {"n50", "n90"}
                else point[key]["agreement_crossings"].get(metric)
            )
            row[f"{metric}_semantic_repetitions"] = (
                float(point_value) / relevant_cells
                if point_value is not None else None
            )
        summary_rows.append(row)

    contrast_rows = []

    def add_contrast(
        contrast_type: str,
        left: tuple[str, str],
        right: tuple[str, str],
    ) -> None:
        for metric in metric_keys:
            left_values = np.asarray(samples[left][metric], dtype=np.float64)
            right_values = np.asarray(samples[right][metric], dtype=np.float64)
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            differences = left_values[valid] - right_values[valid]
            contrast_rows.append({
                "optimizer_name": optimizer_name,
                "contrast_type": contrast_type,
                "left_protocol": left[0],
                "left_target": left[1],
                "right_protocol": right[0],
                "right_target": right[1],
                "metric": metric,
                "paired_bootstrap_count": int(valid.sum()),
                "difference_left_minus_right_median": (
                    float(np.median(differences)) if len(differences) else None
                ),
                "difference_q025": (
                    float(np.quantile(differences, 0.025))
                    if len(differences) else None
                ),
                "difference_q975": (
                    float(np.quantile(differences, 0.975))
                    if len(differences) else None
                ),
            })

    for name in Config.TARGET_NAMES:
        add_contrast(
            "conflict_enriched_minus_uniform",
            ("conflict_enriched", name),
            ("uniform_random", name),
        )
        add_contrast(
            "cell_balanced_minus_uniform",
            ("cell_balanced", name),
            ("uniform_random", name),
        )
    for protocol in Config.SAMPLING_PROTOCOLS:
        add_contrast(
            "mux3_minus_parity2",
            (protocol, "mux3"),
            (protocol, "parity2"),
        )
    return summary_rows, contrast_rows


def same_n_rows(
    rows: Sequence[dict[str, Any]],
    optimizer_name: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    output = []
    for protocol in Config.SAMPLING_PROTOCOLS:
        for name in Config.TARGET_NAMES:
            for n in Config.TRAIN_COUNTS:
                local = [
                    row for row in rows
                    if row["optimizer_name"] == optimizer_name
                    and row["sampling_protocol"] == protocol
                    and row["target_name"] == name
                    and int(row["n"]) == n
                ]
                output.append({
                    "optimizer_name": optimizer_name,
                    "sampling_protocol": protocol,
                    "target_name": name,
                    "n": n,
                    "semantic_repetitions": (
                        n / (4 if name == "parity2" else 8)
                    ),
                    "dataset_count": len(local),
                    "recovery_rate": finite_mean([
                        float(row["dataset_recovered"]) for row in local
                    ]),
                    "target_mass_mean": finite_mean([
                        row["target_function_mass"] for row in local
                    ]),
                    "unseen_agreement_mean": finite_mean([
                        row["unseen_bit_agreement"] for row in local
                    ]),
                    "unseen_target_accuracy_mean": finite_mean([
                        row["unseen_target_bit_accuracy_mean"] for row in local
                    ]),
                    "minimum_cell_count_mean": finite_mean([
                        row["minimum_relevant_cell_count"] for row in local
                    ]),
                    "maximum_cell_count_mean": finite_mean([
                        row["maximum_relevant_cell_count"] for row in local
                    ]),
                    "conflict_fraction_mean": finite_mean([
                        row["conflict_fraction"] for row in local
                    ]),
                    "copy_x1_accuracy_mean": finite_mean([
                        row["copy_x1_accuracy"] for row in local
                    ]),
                    "copy_x2_accuracy_mean": finite_mean([
                        row["copy_x2_accuracy"] for row in local
                    ]),
                    "cohort_copy_x1_full_agreement_mean": finite_mean([
                        row["cohort_copy_x1_full_agreement_mean"]
                        for row in local
                    ]),
                    "cohort_copy_x2_full_agreement_mean": finite_mean([
                        row["cohort_copy_x2_full_agreement_mean"]
                        for row in local
                    ]),
                    "modal_copy_x1_full_agreement_mean": finite_mean([
                        row["modal_copy_x1_full_agreement"] for row in local
                    ]),
                    "modal_copy_x2_full_agreement_mean": finite_mean([
                        row["modal_copy_x2_full_agreement"] for row in local
                    ]),
                })
    return output


def save_plot(output: Path, rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    optimizer_name = Config.OPTIMIZERS[0][0]
    figure, axes = plt.subplots(
        len(Config.SAMPLING_PROTOCOLS), 4,
        figsize=(18, 4.2 * len(Config.SAMPLING_PROTOCOLS)),
        squeeze=False,
    )
    for row_index, protocol in enumerate(Config.SAMPLING_PROTOCOLS):
        for name in Config.TARGET_NAMES:
            curve = curve_from_rows(rows, optimizer_name, name, protocol)
            axes[row_index, 0].plot(
                curve["counts"], curve["recovery"], marker="o", ms=3,
                label=name,
            )
            axes[row_index, 1].plot(
                curve["counts"], curve["agreement"], marker="o", ms=3,
                label=name,
            )
            axes[row_index, 2].plot(
                curve["counts"], curve["accuracy"], marker="o", ms=3,
                label=name,
            )
            axes[row_index, 3].plot(
                curve["counts"], curve["target_mass"], marker="o", ms=3,
                label=name,
            )
        axes[row_index, 0].set_title(f"{protocol}: recovery")
        axes[row_index, 1].set_title(f"{protocol}: agreement")
        axes[row_index, 2].set_title(f"{protocol}: target accuracy")
        axes[row_index, 3].set_title(f"{protocol}: target mass")
        for axis in axes[row_index]:
            axis.set_xlabel("training sample count")
            axis.set_ylim(-0.02, 1.02)
            axis.legend()
    figure.tight_layout()
    figure.savefig(output / "coverage_shortcut_intervention_curves.png", dpi=180)
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
    output = prepare_output(targets)
    sampling_orders = build_sampling_orders()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32

    print("=== Parity2 / MUX3 coverage-shortcut intervention ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=8->{Config.WIDTH}x2->1 tanh | params={parameter_count()} | "
        f"n={Config.TRAIN_COUNTS[0]}..{Config.TRAIN_COUNTS[-1]} step=8 | "
        f"datasets/n={Config.DATASETS_PER_N} | seeds/dataset="
        f"{Config.MODEL_SEED_COUNT}",
        flush=True,
    )
    print(f"sampling_protocols={Config.SAMPLING_PROTOCOLS}", flush=True)
    print(f"optimizers={Config.OPTIMIZERS}", flush=True)
    print(f"result_dir={output}", flush=True)

    status = "running"
    qualification_rows = []
    qualified_optimizers = []
    full_conditions = build_full_conditions(targets)
    for optimizer_name, learning_rate, momentum in Config.OPTIMIZERS:
        print(
            f"\n=== optimizer={optimizer_name} lr={learning_rate:g} "
            f"momentum={momentum:g} ===",
            flush=True,
        )
        status = run_n(
            output,
            optimizer_name,
            learning_rate,
            momentum,
            2 ** Config.INPUT_BITS,
            full_conditions,
            targets,
            device,
            qualification=True,
        )
        if status == "interrupted":
            break
        qualification = qualification_result(output, optimizer_name)
        qualification_rows.append(qualification)
        write_json(output / "optimizer_qualification.json", qualification_rows)
        print(
            f"optimizer={optimizer_name} full-data qualification="
            f"{qualification['passed']}",
            flush=True,
        )
        if not qualification["passed"]:
            print(
                f"optimizer={optimizer_name} 未通过完整数据资格检查，"
                "标记为 optimizer-censored 并跳过相变扫描。",
                flush=True,
            )
            continue
        qualified_optimizers.append(optimizer_name)
        for n in Config.TRAIN_COUNTS:
            conditions = build_conditions(n, targets, sampling_orders)
            status = run_n(
                output,
                optimizer_name,
                learning_rate,
                momentum,
                n,
                conditions,
                targets,
                device,
            )
            if status == "interrupted":
                break
        if status == "interrupted":
            break

    if status != "interrupted":
        rows = load_final_rows(output)
        write_csv(output / "all_dataset_final.csv", rows)
        rng = np.random.default_rng(Config.BOOTSTRAP_SEED)
        transition_rows = []
        transition_contrasts = []
        same_n = []
        verdicts = {}
        for optimizer_name in qualified_optimizers:
            local_transitions, local_contrasts = bootstrap_summary(
                rows, optimizer_name, rng
            )
            transition_rows.extend(local_transitions)
            transition_contrasts.extend(local_contrasts)
            same_n.extend(same_n_rows(rows, optimizer_name, rng))

            lookup = {
                (row["sampling_protocol"], row["target_name"]): row
                for row in local_transitions
            }
            uniform_p2 = lookup[("uniform_random", "parity2")]
            uniform_mux = lookup[("uniform_random", "mux3")]
            conflict_p2 = lookup[("conflict_enriched", "parity2")]
            conflict_mux = lookup[("conflict_enriched", "mux3")]

            def shift(
                treated: dict[str, Any],
                baseline: dict[str, Any],
                metric: str,
            ) -> float | None:
                if treated.get(metric) is None or baseline.get(metric) is None:
                    return None
                return float(treated[metric]) - float(baseline[metric])

            mux_shift = shift(conflict_mux, uniform_mux, "n50")
            parity_shift = shift(conflict_p2, uniform_p2, "n50")
            verdicts[optimizer_name] = {
                "uniform_random": {
                    "parity2": uniform_p2,
                    "mux3": uniform_mux,
                },
                "conflict_enriched": {
                    "parity2": conflict_p2,
                    "mux3": conflict_mux,
                },
                "mux_n50_shift_conflict_minus_uniform": mux_shift,
                "parity_n50_shift_conflict_minus_uniform": parity_shift,
                "shortcut_intervention_selective_support": (
                    mux_shift is not None and parity_shift is not None
                    and mux_shift < parity_shift
                ),
                "uniform_n50_semantic_repetitions": {
                    "parity2": uniform_p2.get("n50_semantic_repetitions"),
                    "mux3": uniform_mux.get("n50_semantic_repetitions"),
                },
            }
        write_csv(output / "transition_estimates.csv", transition_rows)
        write_csv(
            output / "transition_bootstrap_contrasts.csv",
            transition_contrasts,
        )
        write_csv(output / "same_n_summary.csv", same_n)
        save_plot(output, rows)
        write_json(output / "summary.json", {
            "status": "completed",
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "qualification": qualification_rows,
            "verdicts": verdicts,
        })
        print(json.dumps(verdicts, ensure_ascii=False, indent=2), flush=True)
        if Config.PACKAGE_RESULTS:
            print(f"archive={create_archive(output)}", flush=True)
    else:
        write_json(output / "summary.json", {"status": "interrupted"})


if __name__ == "__main__":
    main()
