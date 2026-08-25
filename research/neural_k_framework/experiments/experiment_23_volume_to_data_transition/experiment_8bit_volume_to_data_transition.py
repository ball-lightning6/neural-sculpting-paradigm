"""8-bit规则体积到数据相变前瞻检验：阶段B，随机训练集扫描。

必须先运行 ``experiment_8bit_rule_volume_preregister.py``。本脚本会验证并冻结
阶段A生成的 ``frozen_volume_prediction.json``，随后才训练任何模型。

每个训练样本数n使用同一批均匀随机输入子集；每个规则在这些子集上使用完全
配对的初始化。每个数据集先在多个seed内部计算完整256状态上的函数分布，再在
相同n的随机数据集之间求平均。主判据不是裸agreement，而是：

1. 训练集被绝大多数seed hard-fit；
2. 完整目标函数在已拟合seed中的质量>=0.90；
3. modal hard function就是目标；
4. 完整函数collision>=0.80。

由随机数据集恢复率曲线定义n50/n90，并检验阶段A冻结的volume contraction
排序能否前瞻预测它们。n=256只用于验证容量和优化可达性，不参与泛化相变。
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
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2
    EXPECTED_TARGET_NAMES = (
        "parity1",
        "parity2",
        "parity3",
        "parity4",
        "majority3",
        "majority5",
        "mux3",
        "random_balanced",
    )
    RANDOM_TARGET_SEED = 2026082401

    VOLUME_RESULT_DIR = Path("/root/results_8bit_rule_volume_preregister")
    RESULT_DIR = Path("/root/results_8bit_volume_to_data_transition")

    # n=256是可达性对照，不计入n50/n90。其余每个n独立均匀抽取训练输入子集。
    TRAIN_COUNTS = (
        256,
        1, 2, 4, 6, 8, 12, 16, 24, 32, 48,
        64, 80, 96, 112, 128, 160, 192, 224, 240,
    )
    DATASETS_PER_N = 64
    DATASET_SEED = 2026082501

    MODEL_SEED_COUNT = 24
    INITIALIZATION_SEED = 2026082502
    # 8*64=512个条件在5090上一次并行，避免大量小BMM导致GPU利用率过低。
    CONDITIONS_PER_SHARD = 512

    LEARNING_RATE = 1e-3
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
    TRANSITION_DATASET_RATES = (0.50, 0.90)
    BOOTSTRAP_REPLICATES = 2_000
    BOOTSTRAP_SEED = 2026082503

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class TargetSpec:
    target_index: int
    name: str
    formula: str
    family: str
    outputs: tuple[int, ...]
    function_hex: str


@dataclass(frozen=True)
class SubsetSpec:
    n: int
    dataset_index: int
    indices: tuple[int, ...]
    signature: str


@dataclass(frozen=True)
class ConditionSpec:
    condition_index: int
    target_index: int
    target_name: str
    dataset_index: int
    n: int
    indices: tuple[int, ...]
    signature: str


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.EXPECTED_TARGET_NAMES = ("parity1", "parity2")
    Config.TRAIN_COUNTS = (4, 16, 256)
    Config.DATASETS_PER_N = 3
    Config.MODEL_SEED_COUNT = 4
    Config.CONDITIONS_PER_SHARD = 4
    Config.MAX_STEPS = 5
    Config.EVAL_STEPS = (0, 1, 2, 5)
    Config.CHECKPOINT_EVERY_STEPS = 2
    Config.MIN_TRAIN_FIT_RATE = 0.0
    Config.TARGET_FUNCTION_MASS_THRESHOLD = 0.0
    Config.FUNCTION_COLLISION_THRESHOLD = 0.0
    Config.BOOTSTRAP_REPLICATES = 20
    Config.DEVICE = "cpu"
    Config.VOLUME_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_rule_volume_preregister"
    )
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_volume_to_data_transition"
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


def canonical_hash(payload: dict[str, Any]) -> str:
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
    result = 0
    for index, bit in enumerate(np.asarray(outputs, dtype=np.uint8)):
        result |= int(bit) << index
    width = len(outputs) // 4
    return f"0x{result:0{width}X}"


def anf_metrics(outputs: np.ndarray) -> tuple[int, int, int]:
    coefficients = np.asarray(outputs, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(2 ** Config.INPUT_BITS):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.asarray([int(term).bit_count() for term in terms])
    return (
        int(degrees.max()) if len(degrees) else 0,
        int(len(terms)),
        int(degrees.sum()),
    )


def essential_variable_count(outputs: np.ndarray) -> int:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    count = 0
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        count += int(np.any(outputs[base] != outputs[base | (1 << bit)]))
    return count


def build_targets() -> list[TargetSpec]:
    inputs = truth_table_inputs()
    raw: dict[str, tuple[np.ndarray, str, str]] = {}
    for count in range(1, 5):
        outputs = np.bitwise_xor.reduce(inputs[:, :count], axis=1)
        raw[f"parity{count}"] = (
            outputs.astype(np.uint8),
            " XOR ".join(f"x{index}" for index in range(count)),
            "nested_parity",
        )
    raw["majority3"] = (
        (inputs[:, :3].sum(axis=1) >= 2).astype(np.uint8),
        "(x0+x1+x2)>=2",
        "nested_majority",
    )
    raw["majority5"] = (
        (inputs[:, :5].sum(axis=1) >= 3).astype(np.uint8),
        "(x0+x1+x2+x3+x4)>=3",
        "nested_majority",
    )
    raw["mux3"] = (
        np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]).astype(
            np.uint8
        ),
        "IF x0 THEN x1 ELSE x2",
        "multiplexer_control",
    )
    rng = np.random.default_rng(Config.RANDOM_TARGET_SEED)
    attempt = 0
    while True:
        outputs = np.zeros(2 ** Config.INPUT_BITS, dtype=np.uint8)
        outputs[rng.choice(len(outputs), len(outputs) // 2, replace=False)] = 1
        _, terms, literals = anf_metrics(outputs)
        if (
            essential_variable_count(outputs) == Config.INPUT_BITS
            and terms >= 100 and literals >= 350
        ):
            break
        attempt += 1
    raw["random_balanced"] = (
        outputs,
        f"fixed balanced random truth table seed={Config.RANDOM_TARGET_SEED}, "
        f"attempt={attempt}",
        "random_balanced_control",
    )

    result = []
    for index, name in enumerate(Config.EXPECTED_TARGET_NAMES):
        outputs, formula, family = raw[name]
        result.append(TargetSpec(
            target_index=index,
            name=name,
            formula=formula,
            family=family,
            outputs=tuple(map(int, outputs)),
            function_hex=outputs_to_hex(outputs),
        ))
    return result


def verify_volume_prediction(
    targets: Sequence[TargetSpec],
) -> dict[str, Any]:
    root = Path(Config.VOLUME_RESULT_DIR)
    prediction_path = root / "frozen_volume_prediction.json"
    definitions_path = root / "target_definitions.csv"
    if not prediction_path.exists() or not definitions_path.exists():
        raise FileNotFoundError(
            "缺少阶段A冻结结果。请先完整运行"
            "experiment_8bit_rule_volume_preregister.py。"
        )
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    expected_hash = str(prediction.get("prediction_sha256", ""))
    unhashed = dict(prediction)
    unhashed.pop("prediction_sha256", None)
    actual_hash = canonical_hash(unhashed)
    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError("阶段A冻结预测的SHA256校验失败。")
    if prediction.get("protocol") != "8bit_full_rule_volume_to_data_transition_v1":
        raise RuntimeError("阶段A协议版本不匹配。")
    if int(prediction.get("input_bits", -1)) != Config.INPUT_BITS:
        raise RuntimeError("阶段A输入维度不匹配。")

    definitions = {row["name"]: row for row in read_csv(definitions_path)}
    for target in targets:
        row = definitions.get(target.name)
        if row is None or row.get("function_hex") != target.function_hex:
            raise RuntimeError(f"阶段A目标定义不匹配：{target.name}")
    scored_names = [
        str(row["target_name"]) for row in prediction["target_scores"]
    ]
    if set(scored_names) != {target.name for target in targets}:
        raise RuntimeError("阶段A并未给当前全部目标生成volume score。")
    return prediction


def build_subsets() -> dict[int, list[SubsetSpec]]:
    state_count = 2 ** Config.INPUT_BITS
    result: dict[int, list[SubsetSpec]] = {}
    for n in Config.TRAIN_COUNTS:
        if not 1 <= n <= state_count:
            raise ValueError(f"非法训练样本数：{n}")
        if n == state_count:
            arrays = [np.arange(state_count, dtype=np.int64)]
        else:
            rng = np.random.default_rng(Config.DATASET_SEED + 1_000_003 * n)
            arrays = []
            seen: set[tuple[int, ...]] = set()
            while len(arrays) < Config.DATASETS_PER_N:
                indices = tuple(sorted(map(int, rng.choice(
                    state_count, n, replace=False
                ))))
                if indices in seen:
                    continue
                seen.add(indices)
                arrays.append(np.asarray(indices, dtype=np.int64))
        specs = []
        for dataset_index, indices in enumerate(arrays):
            payload = np.asarray(indices, dtype=np.uint16).tobytes()
            signature = hashlib.sha256(payload).hexdigest()[:16]
            specs.append(SubsetSpec(
                n=n,
                dataset_index=dataset_index,
                indices=tuple(map(int, indices)),
                signature=signature,
            ))
        result[n] = specs
    return result


def build_conditions(
    targets: Sequence[TargetSpec], subsets: Sequence[SubsetSpec]
) -> list[ConditionSpec]:
    conditions = []
    for target in targets:
        for subset in subsets:
            conditions.append(ConditionSpec(
                condition_index=len(conditions),
                target_index=target.target_index,
                target_name=target.name,
                dataset_index=subset.dataset_index,
                n=subset.n,
                indices=subset.indices,
                signature=subset.signature,
            ))
    return conditions


def prepare_result_dir(
    prediction: dict[str, Any],
    targets: Sequence[TargetSpec],
    subsets: dict[int, list[SubsetSpec]],
) -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)

    protocol = {
        "protocol": "8bit_volume_to_data_transition_training_v1",
        "volume_prediction_sha256": prediction["prediction_sha256"],
        "volume_predicted_order_easy_to_hard": (
            prediction["predicted_order_easy_to_hard"]
        ),
        "primary_predicted_order_easy_to_hard": (
            prediction["primary_predicted_order_easy_to_hard"]
        ),
        "config": config_payload(),
        "targets": [asdict(target) for target in targets],
        "subsets": {
            str(n): [asdict(spec) for spec in local]
            for n, local in subsets.items()
        },
        "decision_rule": {
            "dataset_recovered_if": (
                "train_fit_rate>=MIN_TRAIN_FIT_RATE AND "
                "target_function_mass>=TARGET_FUNCTION_MASS_THRESHOLD AND "
                "modal_is_target AND "
                "function_collision>=FUNCTION_COLLISION_THRESHOLD"
            ),
            "n50_n90": (
                "最小随机训练样本数n，使跨随机训练集的isotonic恢复率分别"
                "达到0.50/0.90；n=256只作资格检查。"
            ),
        },
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    path = output / "preregistered_transition_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != json_ready(protocol):
            raise RuntimeError("结果目录已有不同的预注册训练协议。")
        if not Config.RESUME:
            raise RuntimeError("结果目录已存在且RESUME=False。")
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
                    Config.INITIALIZATION_SEED + 100_003 * seed_index
                    + 10_007 * layer_index
                )
                bound = 1.0 / math.sqrt(fan_in)
                base_weights.append(torch.empty(fan_out, fan_in).uniform_(
                    -bound, bound, generator=generator
                ))
                base_biases.append(torch.empty(fan_out).uniform_(
                    -bound, bound, generator=generator
                ))
            weights = torch.stack(base_weights)
            biases = torch.stack(base_biases)
            weights = weights[None].expand(
                condition_count, -1, -1, -1
            ).reshape(condition_count * Config.MODEL_SEED_COUNT, fan_out, fan_in)
            biases = biases[None].expand(
                condition_count, -1, -1
            ).reshape(condition_count * Config.MODEL_SEED_COUNT, fan_out)
            self.weights.append(nn.Parameter(weights.clone()))
            self.biases.append(nn.Parameter(biases.clone()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if layer_index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def parameter_count_per_model() -> int:
    dims = [Config.INPUT_BITS] + [Config.WIDTH] * Config.HIDDEN_LAYERS + [1]
    return int(sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(dims[:-1], dims[1:])
    ))


def build_tensors(
    conditions: Sequence[ConditionSpec],
    targets: Sequence[TargetSpec],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    inputs = truth_table_inputs().astype(np.float32)
    target_matrix = np.asarray([target.outputs for target in targets], dtype=np.uint8)
    n = conditions[0].n
    train_x = np.empty((len(conditions), n, Config.INPUT_BITS), dtype=np.float32)
    train_y = np.empty((len(conditions), n), dtype=np.float32)
    condition_targets = np.empty(
        (len(conditions), 2 ** Config.INPUT_BITS), dtype=np.uint8
    )
    for local_index, condition in enumerate(conditions):
        indices = np.asarray(condition.indices, dtype=np.int64)
        outputs = target_matrix[condition.target_index]
        train_x[local_index] = inputs[indices]
        train_y[local_index] = outputs[indices]
        condition_targets[local_index] = outputs
    return (
        torch.as_tensor(
            np.repeat(train_x, Config.MODEL_SEED_COUNT, axis=0), device=device
        ),
        torch.as_tensor(
            np.repeat(train_y, Config.MODEL_SEED_COUNT, axis=0), device=device
        ),
        torch.as_tensor(inputs, device=device),
        condition_targets,
    )


def collision_probability(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total < 2:
        return float("nan")
    return float(
        np.sum(counts.astype(np.float64) * (counts - 1))
        / (total * (total - 1))
    )


def plugin_entropy(counts: np.ndarray) -> float:
    probability = counts.astype(np.float64) / counts.sum()
    positive = probability[probability > 0]
    return float(-(positive * np.log2(positive)).sum())


def bit_agreement(predictions: np.ndarray, indices: np.ndarray) -> float:
    local = predictions[:, indices]
    count = len(local)
    if count < 2 or not len(indices):
        return float("nan")
    ones = local.sum(axis=0).astype(np.float64)
    same = ones * (ones - 1) + (count - ones) * (count - ones - 1)
    return float(np.mean(same / (count * (count - 1))))


@torch.no_grad()
def evaluate(
    step: int,
    model: PairedBatchedMLP,
    conditions: Sequence[ConditionSpec],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    full_inputs: torch.Tensor,
    condition_targets: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    train_logits = model(train_x)
    losses = F.binary_cross_entropy_with_logits(
        train_logits, train_y, reduction="none"
    ).mean(dim=1)
    fitted = torch.all(
        (train_logits >= 0) == (train_y >= 0.5), dim=1
    )
    full_batch = full_inputs[None].expand(len(train_x), -1, -1)
    predictions = (model(full_batch) >= 0).to(torch.uint8)

    condition_count = len(conditions)
    seed_count = Config.MODEL_SEED_COUNT
    losses_np = losses.cpu().numpy().reshape(condition_count, seed_count)
    fitted_np = fitted.cpu().numpy().reshape(condition_count, seed_count)
    predictions_np = predictions.cpu().numpy().reshape(
        condition_count, seed_count, -1
    )
    target_exact = np.all(
        predictions_np == condition_targets[:, None, :], axis=2
    )
    packed = np.packbits(predictions_np, axis=2, bitorder="little")

    all_indices = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    rows = []
    for local_index, condition in enumerate(conditions):
        local_fitted = fitted_np[local_index]
        cohort = predictions_np[local_index][local_fitted]
        cohort_exact = target_exact[local_index][local_fitted]
        cohort_source = "train_hard_exact_models"
        if not len(cohort):
            cohort = predictions_np[local_index]
            cohort_exact = target_exact[local_index]
            cohort_source = "all_models_no_fitted_cohort"
        cohort_packed = np.packbits(cohort, axis=1, bitorder="little")
        unique, counts = np.unique(cohort_packed, axis=0, return_counts=True)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]
        target_packed = np.packbits(
            condition_targets[local_index], bitorder="little"
        )
        modal_is_target = bool(np.array_equal(unique[0], target_packed))
        train_indices = np.asarray(condition.indices, dtype=np.int64)
        unseen = np.setdiff1d(all_indices, train_indices, assume_unique=True)
        target_mass = float(cohort_exact.mean())
        fit_rate = float(local_fitted.mean())
        collision = collision_probability(counts)
        recovered = bool(
            fit_rate >= Config.MIN_TRAIN_FIT_RATE
            and target_mass >= Config.TARGET_FUNCTION_MASS_THRESHOLD
            and modal_is_target
            and collision >= Config.FUNCTION_COLLISION_THRESHOLD
            and condition.n < 2 ** Config.INPUT_BITS
        )
        target_bit = np.mean(
            predictions_np[local_index] == condition_targets[local_index][None],
            axis=1,
        )
        unseen_target_bit = np.mean(
            predictions_np[local_index][:, unseen]
            == condition_targets[local_index][None, unseen], axis=1
        ) if len(unseen) else np.ones(seed_count)
        rows.append({
            "step": step,
            "condition_index": condition.condition_index,
            "target_index": condition.target_index,
            "target_name": condition.target_name,
            "dataset_index": condition.dataset_index,
            "dataset_signature": condition.signature,
            "n": condition.n,
            "seed_count": seed_count,
            "train_fit_rate": fit_rate,
            "train_loss_mean": float(losses_np[local_index].mean()),
            "train_loss_median": float(np.median(losses_np[local_index])),
            "cohort_source": cohort_source,
            "cohort_count": len(cohort),
            "target_function_mass": target_mass,
            "modal_probability": float(counts[0] / counts.sum()),
            "modal_is_target": modal_is_target,
            "function_collision": collision,
            "function_entropy_bits": plugin_entropy(counts),
            "unique_function_count": len(unique),
            "full_bit_agreement": bit_agreement(cohort, all_indices),
            "unseen_bit_agreement": bit_agreement(cohort, unseen),
            "target_bit_accuracy_mean": float(target_bit.mean()),
            "unseen_target_bit_accuracy_mean": float(unseen_target_bit.mean()),
            "dataset_recovered": recovered,
            "qualification_only": condition.n == 2 ** Config.INPUT_BITS,
        })
    raw = {
        "function_packed": packed,
        "train_fitted": fitted_np,
        "target_exact": target_exact,
        "train_loss": losses_np,
    }
    return rows, raw


def checkpoint_path(shard_dir: Path) -> Path:
    return shard_dir / "checkpoint.pt"


def save_checkpoint(
    shard_dir: Path,
    step: int,
    model: PairedBatchedMLP,
    optimizer: torch.optim.Optimizer,
    trajectory_rows: Sequence[dict[str, Any]],
) -> None:
    temporary = shard_dir / "checkpoint.tmp.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trajectory_rows": list(trajectory_rows),
        "config": config_payload(),
    }, temporary)
    temporary.replace(checkpoint_path(shard_dir))


def run_shard(
    output_dir: Path,
    n: int,
    shard_index: int,
    conditions: Sequence[ConditionSpec],
    targets: Sequence[TargetSpec],
    device: torch.device,
) -> str:
    shard_dir = output_dir / "shards" / f"n_{n:03d}" / f"shard_{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    final_path = shard_dir / "final.csv"
    trajectory_path = shard_dir / "trajectory.csv"
    raw_path = shard_dir / "final_raw.npz"
    if final_path.exists() and trajectory_path.exists() and raw_path.exists():
        print(f"n={n:>3} shard={shard_index:03d} 已完成，跳过。", flush=True)
        return "completed"

    model = PairedBatchedMLP(len(conditions)).to(device)
    train_x, train_y, full_inputs, condition_targets = build_tensors(
        conditions, targets, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    trajectory_rows: list[dict[str, Any]] = []
    start_step = 0
    checkpoint = checkpoint_path(shard_dir)
    if checkpoint.exists() and Config.RESUME:
        payload = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        trajectory_rows = list(payload["trajectory_rows"])
        start_step = int(payload["step"])
        print(
            f"n={n:>3} shard={shard_index:03d} resume step={start_step:,}",
            flush=True,
        )

    eval_steps = set(map(int, Config.EVAL_STEPS)) | {Config.MAX_STEPS}
    last_rows: list[dict[str, Any]] = []
    last_raw: dict[str, np.ndarray] | None = None
    started = time.perf_counter()
    current_step = start_step
    try:
        for step in range(start_step, Config.MAX_STEPS + 1):
            current_step = step
            if step in eval_steps and not any(
                int(row["step"]) == step for row in trajectory_rows
            ):
                last_rows, last_raw = evaluate(
                    step, model, conditions, train_x, train_y,
                    full_inputs, condition_targets,
                )
                trajectory_rows.extend(last_rows)
                write_csv(trajectory_path, trajectory_rows)
                by_target: dict[str, list[dict[str, Any]]] = {}
                for row in last_rows:
                    by_target.setdefault(str(row["target_name"]), []).append(row)
                def finite_mean(local_rows: Sequence[dict[str, Any]], key: str) -> float:
                    values = np.asarray([float(row[key]) for row in local_rows])
                    finite = values[np.isfinite(values)]
                    return float(finite.mean()) if len(finite) else float("nan")
                compact = " | ".join(
                    f"{name}:A={finite_mean(rows, 'unseen_bit_agreement'):.3f},"
                    f"T={finite_mean(rows, 'target_function_mass'):.3f},"
                    f"R={np.mean([bool(r['dataset_recovered']) for r in rows]):.2f}"
                    for name, rows in by_target.items()
                )
                print(
                    f"n={n:>3} shard={shard_index:03d} step={step:>6,} | "
                    f"{compact} | elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
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
                    shard_dir, step + 1, model, optimizer, trajectory_rows
                )
    except KeyboardInterrupt:
        save_checkpoint(
            shard_dir, current_step, model, optimizer, trajectory_rows
        )
        print(
            f"\nn={n} shard={shard_index:03d} 已保存中断checkpoint。",
            flush=True,
        )
        return "interrupted"

    if not last_rows or int(last_rows[0]["step"]) != Config.MAX_STEPS:
        last_rows, last_raw = evaluate(
            Config.MAX_STEPS, model, conditions, train_x, train_y,
            full_inputs, condition_targets,
        )
    write_csv(final_path, last_rows)
    write_csv(trajectory_path, trajectory_rows)
    assert last_raw is not None
    np.savez_compressed(
        raw_path,
        condition_indices=np.asarray(
            [condition.condition_index for condition in conditions], dtype=np.int64
        ),
        **last_raw,
    )
    checkpoint.unlink(missing_ok=True)
    del model, optimizer, train_x, train_y, full_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return "completed"


def shard_ranges(total: int, size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + size, total))
        for start in range(0, total, size)
    ]


def load_all_rows(output_dir: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "shards").glob(f"n_*/shard_*/{filename}")):
        for row in read_csv(path):
            converted: dict[str, Any] = dict(row)
            for key in (
                "step", "condition_index", "target_index", "dataset_index",
                "n", "seed_count", "cohort_count", "unique_function_count",
            ):
                converted[key] = int(float(converted[key]))
            for key in (
                "train_fit_rate", "train_loss_mean", "train_loss_median",
                "target_function_mass", "modal_probability",
                "function_collision", "function_entropy_bits",
                "full_bit_agreement", "unseen_bit_agreement",
                "target_bit_accuracy_mean", "unseen_target_bit_accuracy_mean",
            ):
                converted[key] = float(converted[key])
            for key in (
                "modal_is_target", "dataset_recovered", "qualification_only",
            ):
                converted[key] = str(converted[key]).lower() == "true"
            rows.append(converted)
    return rows


def pava(values: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(value), float(weight), float(index), float(index)])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[1] + right[1]
            mean = (left[0] * left[1] + right[0] * right[1]) / total_weight
            blocks.append([mean, total_weight, left[2], right[3]])
    result = np.empty(len(values), dtype=np.float64)
    for mean, _, start, stop in blocks:
        result[int(start):int(stop) + 1] = mean
    return result


def first_threshold_n(
    n_values: Sequence[int], rates: Sequence[float], threshold: float
) -> int | None:
    for n, rate in zip(n_values, rates):
        if rate >= threshold:
            return int(n)
    return None


def average_ranks(values: Sequence[float]) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_np, kind="mergesort")
    ranks = np.empty(len(values_np), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values_np[order[stop]] == values_np[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) < 2:
        return float("nan")
    x = average_ranks(values_x)
    y = average_ranks(values_y)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def kendall_tau_b(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    concordant = discordant = tie_x = tie_y = 0
    for left in range(len(values_x)):
        for right in range(left + 1, len(values_x)):
            dx = np.sign(values_x[left] - values_x[right])
            dy = np.sign(values_y[left] - values_y[right])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tie_x)
        * (concordant + discordant + tie_y)
    )
    return (
        (concordant - discordant) / denominator
        if denominator else float("nan")
    )


def bootstrap_transition(
    success_by_n: dict[int, np.ndarray], threshold: float, seed: int
) -> tuple[float | None, float | None, float | None]:
    n_values = sorted(success_by_n)
    rng = np.random.default_rng(seed)
    samples = []
    censor = max(n_values) + 1
    for _ in range(Config.BOOTSTRAP_REPLICATES):
        rates = []
        weights = []
        for n in n_values:
            values = success_by_n[n]
            draw = rng.choice(values, size=len(values), replace=True)
            rates.append(float(draw.mean()))
            weights.append(float(len(values)))
        fitted = pava(rates, weights)
        point = first_threshold_n(n_values, fitted, threshold)
        samples.append(censor if point is None else point)
    quantiles = np.quantile(samples, [0.025, 0.5, 0.975])
    return tuple(
        None if value >= censor else float(value) for value in quantiles
    )


def aggregate(
    output_dir: Path,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    trajectory = load_all_rows(output_dir, "trajectory.csv")
    final_rows = load_all_rows(output_dir, "final.csv")
    write_csv(output_dir / "all_dataset_trajectory.csv", trajectory)
    write_csv(output_dir / "all_dataset_final.csv", final_rows)

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in trajectory:
        grouped.setdefault(
            (str(row["target_name"]), int(row["n"]), int(row["step"])), []
        ).append(row)
    curve_rows = []
    for (name, n, step), rows in sorted(grouped.items()):
        metrics = {}
        for key in (
            "train_fit_rate", "target_function_mass", "function_collision",
            "unseen_bit_agreement", "target_bit_accuracy_mean",
            "unseen_target_bit_accuracy_mean",
        ):
            values = np.asarray([float(row[key]) for row in rows])
            finite = values[np.isfinite(values)]
            metrics[f"{key}_mean"] = (
                float(finite.mean()) if len(finite) else None
            )
            metrics[f"{key}_median"] = (
                float(np.median(finite)) if len(finite) else None
            )
            metrics[f"{key}_q10"] = (
                float(np.quantile(finite, 0.10)) if len(finite) else None
            )
            metrics[f"{key}_q90"] = (
                float(np.quantile(finite, 0.90)) if len(finite) else None
            )
        curve_rows.append({
            "target_name": name,
            "n": n,
            "step": step,
            "dataset_count": len(rows),
            "dataset_recovery_rate": float(np.mean([
                bool(row["dataset_recovered"]) for row in rows
            ])),
            **metrics,
        })
    write_csv(output_dir / "mean_agreement_recovery_curves.csv", curve_rows)

    final_nonfull = [
        row for row in final_rows
        if int(row["n"]) < 2 ** Config.INPUT_BITS
    ]
    score_map = {
        str(row["target_name"]): float(
            row["volume_contraction_score_median"]
        ) for row in prediction["target_scores"]
    }
    target_names = [str(name) for name in Config.EXPECTED_TARGET_NAMES]
    transition_rows = []
    for target_index, name in enumerate(target_names):
        local = [row for row in final_nonfull if row["target_name"] == name]
        by_n: dict[int, list[dict[str, Any]]] = {}
        for row in local:
            by_n.setdefault(int(row["n"]), []).append(row)
        n_values = sorted(by_n)
        raw_rates = [float(np.mean([
            bool(row["dataset_recovered"]) for row in by_n[n]
        ])) for n in n_values]
        weights = [len(by_n[n]) for n in n_values]
        isotonic = pava(raw_rates, weights) if n_values else np.asarray([])
        success_by_n = {
            n: np.asarray([
                bool(row["dataset_recovered"]) for row in by_n[n]
            ], dtype=np.float64) for n in n_values
        }
        result: dict[str, Any] = {
            "target_name": name,
            "volume_contraction_score": score_map[name],
            "n_values": n_values,
            "raw_recovery_rates": raw_rates,
            "isotonic_recovery_rates": isotonic.tolist(),
        }
        for threshold in Config.TRANSITION_DATASET_RATES:
            label = f"n{int(round(100 * threshold))}"
            result[label] = first_threshold_n(n_values, isotonic, threshold)
            if success_by_n:
                low, median, high = bootstrap_transition(
                    success_by_n, threshold,
                    Config.BOOTSTRAP_SEED + target_index * 101
                    + int(round(threshold * 1000)),
                )
            else:
                low = median = high = None
            result[f"{label}_bootstrap_q025"] = low
            result[f"{label}_bootstrap_median"] = median
            result[f"{label}_bootstrap_q975"] = high
        transition_rows.append(result)
    write_csv(output_dir / "transition_points.csv", transition_rows)

    qualification = []
    for name in target_names:
        rows = [
            row for row in final_rows
            if row["target_name"] == name
            and int(row["n"]) == 2 ** Config.INPUT_BITS
        ]
        qualification.append({
            "target_name": name,
            "full_data_condition_count": len(rows),
            "train_fit_rate_mean": float(np.mean([
                float(row["train_fit_rate"]) for row in rows
            ])) if rows else None,
            "target_function_mass_mean": float(np.mean([
                float(row["target_function_mass"]) for row in rows
            ])) if rows else None,
            "full_data_qualified": bool(rows) and all(
                float(row["train_fit_rate"]) >= Config.MIN_TRAIN_FIT_RATE
                for row in rows
            ),
        })
    write_csv(output_dir / "full_data_qualification.csv", qualification)

    censor = max(n for n in Config.TRAIN_COUNTS if n < 2 ** Config.INPUT_BITS) + 1
    scores = [score_map[name] for name in target_names]
    n50 = [
        censor if row.get("n50") in (None, "") else float(row["n50"])
        for row in transition_rows
    ]
    n90 = [
        censor if row.get("n90") in (None, "") else float(row["n90"])
        for row in transition_rows
    ]
    primary = list(prediction["primary_predicted_order_easy_to_hard"])
    transition_by_name = {row["target_name"]: row for row in transition_rows}
    primary_n50 = [
        censor if transition_by_name[name].get("n50") is None
        else int(transition_by_name[name]["n50"])
        for name in primary
    ]
    primary_n90 = [
        censor if transition_by_name[name].get("n90") is None
        else int(transition_by_name[name]["n90"])
        for name in primary
    ]
    qualification_map = {
        row["target_name"]: bool(row["full_data_qualified"])
        for row in qualification
    }
    primary_qualified = all(qualification_map.get(name, False) for name in primary)
    def order_verdict(values: Sequence[int]) -> str:
        if not primary_qualified:
            return "invalid_full_data_not_qualified"
        if not bool(np.all(np.diff(values) >= 0)):
            return "contradicted_order_reversal"
        if len(set(values)) < 2 or all(value == censor for value in values):
            return "uninformative_tie_or_censoring"
        return "supported_nondecreasing"
    return {
        "completed_dataset_rows": len(final_rows),
        "target_count": len(target_names),
        "volume_prediction_sha256": prediction["prediction_sha256"],
        "primary_predicted_order_easy_to_hard": primary,
        "primary_measured_n50_in_predicted_order": primary_n50,
        "primary_measured_n90_in_predicted_order": primary_n90,
        "primary_all_full_data_qualified": primary_qualified,
        "primary_n50_nondecreasing": bool(np.all(np.diff(primary_n50) >= 0)),
        "primary_n90_nondecreasing": bool(np.all(np.diff(primary_n90) >= 0)),
        "primary_n50_verdict": order_verdict(primary_n50),
        "primary_n90_verdict": order_verdict(primary_n90),
        "all_rule_spearman_volume_vs_n50": spearman(scores, n50),
        "all_rule_spearman_volume_vs_n90": spearman(scores, n90),
        "all_rule_kendall_tau_b_volume_vs_n50": kendall_tau_b(scores, n50),
        "all_rule_kendall_tau_b_volume_vs_n90": kendall_tau_b(scores, n90),
        "right_censor_code_for_correlations": censor,
        "interpretation_boundary": (
            "排序成功支持静态full-rule volume是数据相变的一阶预测量；失败则"
            "说明竞争函数结构、训练动力学或该loss区间的volume score不可忽略。"
        ),
    }


def save_plots(output_dir: Path, prediction: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    curves = read_csv(output_dir / "mean_agreement_recovery_curves.csv")
    transitions = read_csv(output_dir / "transition_points.csv")
    if not curves or not transitions:
        return
    final_step = max(int(row["step"]) for row in curves)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = list(Config.EXPECTED_TARGET_NAMES)
    for name in names:
        local = sorted([
            row for row in curves
            if row["target_name"] == name
            and int(row["step"]) == final_step
            and int(row["n"]) < 2 ** Config.INPUT_BITS
        ], key=lambda row: int(row["n"]))
        n = [int(row["n"]) for row in local]
        axes[0, 0].plot(
            n, [float(row["dataset_recovery_rate"]) for row in local],
            marker="o", ms=3, label=name,
        )
        axes[0, 1].plot(
            n, [float(row["unseen_bit_agreement_mean"]) for row in local],
            marker="o", ms=3, label=name,
        )
    axes[0, 0].set_title("random-dataset target recovery")
    axes[0, 1].set_title("mean unseen agreement")
    for axis in axes[0]:
        axis.set_xlabel("training examples n")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)

    score_rows = prediction["target_scores"]
    score_map = {
        str(row["target_name"]): float(row["volume_contraction_score_median"])
        for row in score_rows
    }
    transition_map = {row["target_name"]: row for row in transitions}
    censor = max(n for n in Config.TRAIN_COUNTS if n < 256) + 1
    x = [score_map[name] for name in names]
    y50 = [
        censor if transition_map[name]["n50"] in ("", None)
        else float(transition_map[name]["n50"]) for name in names
    ]
    y90 = [
        censor if transition_map[name]["n90"] in ("", None)
        else float(transition_map[name]["n90"]) for name in names
    ]
    axes[1, 0].scatter(x, y50)
    axes[1, 1].scatter(x, y90)
    for axis, y, title in (
        (axes[1, 0], y50, "volume score vs n50"),
        (axes[1, 1], y90, "volume score vs n90"),
    ):
        for name, xv, yv in zip(names, x, y):
            axis.annotate(name, (xv, yv), fontsize=8)
        axis.set_xscale("symlog")
        axis.set_xlabel("frozen volume contraction score")
        axis.set_ylabel("transition n (censored at max+1)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "volume_to_data_transition.png", dpi=180)
    plt.close(figure)


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.name not in {
                "checkpoint.pt", "checkpoint.tmp.pt",
            }:
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def main() -> None:
    apply_smoke_overrides()
    targets = build_targets()
    prediction = verify_volume_prediction(targets)
    subsets = build_subsets()
    output_dir = prepare_result_dir(prediction, targets, subsets)
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    torch.set_float32_matmul_precision("highest")

    print("=== 8-bit Volume -> Data Transition Pre-registered Test ===", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=8->{Config.WIDTH}x2->1 tanh | "
        f"params/model={parameter_count_per_model():,} | "
        f"targets={list(Config.EXPECTED_TARGET_NAMES)}",
        flush=True,
    )
    print(
        f"datasets/n={Config.DATASETS_PER_N} | seeds/dataset="
        f"{Config.MODEL_SEED_COUNT} | max_steps={Config.MAX_STEPS:,}",
        flush=True,
    )
    print(
        "冻结volume顺序："
        + " < ".join(prediction["predicted_order_easy_to_hard"]),
        flush=True,
    )
    print(f"结果目录：{output_dir}", flush=True)

    started = time.perf_counter()
    interrupted = False
    for n in Config.TRAIN_COUNTS:
        conditions = build_conditions(targets, subsets[n])
        ranges = shard_ranges(len(conditions), Config.CONDITIONS_PER_SHARD)
        print(
            f"\n=== n={n} | datasets={len(subsets[n])} | "
            f"conditions={len(conditions)} | shards={len(ranges)} ===",
            flush=True,
        )
        for shard_index, (start, stop) in enumerate(ranges):
            status = run_shard(
                output_dir, n, shard_index, conditions[start:stop], targets, device
            )
            if status == "interrupted":
                interrupted = True
                break
        if interrupted:
            break

    aggregate_summary = aggregate(output_dir, prediction)
    save_plots(output_dir, prediction)
    summary = {
        "status": "interrupted" if interrupted else "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "network": f"8->{Config.WIDTH}x2->1 tanh",
        "parameter_count_per_model": parameter_count_per_model(),
        **aggregate_summary,
    }
    write_json(output_dir / "summary.json", summary)
    archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
    print("\n=== 判决汇总 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
