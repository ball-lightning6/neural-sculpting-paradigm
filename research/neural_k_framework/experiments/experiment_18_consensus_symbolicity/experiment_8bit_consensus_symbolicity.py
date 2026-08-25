"""8-bit teacher-free 共识符号性 Pilot。

对随机稀疏部分真值表进行多 seed、完整 256 点函数审计，筛选高完整函数共识
训练集，并用运行前冻结的符号复杂度指标寻找“高共识但高复杂度”反例候选。
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import shutil
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
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2

    RANDOM_TRAIN_COUNTS = (8, 12, 16, 24)
    RANDOM_DATASETS_PER_N = 128
    SYMBOLIC_TRAIN_COUNTS = (12, 24)
    INCLUDE_SYMBOLIC_CONTROLS = True
    DATASET_SEED = 2026082201

    SEED_COUNT = 64
    INITIALIZATION_SEED = 2026082202
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 20_000
    EARLY_EVAL_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
    EVAL_INTERVAL_STEPS = 500
    SAVE_INTERVAL_STEPS = 2_000

    MIN_FIT_RATE = 0.95
    HIGH_MODAL_PROBABILITY = 0.95
    HIGH_FUNCTION_COLLISION = 0.90

    BDD_RANDOM_ORDERS = 16
    RANDOM_BASELINE_COUNT = 256
    COMPLEXITY_SEED = 2026082203

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_8bit_consensus_symbolicity")
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_kind: str
    train_indices: tuple[int, ...]
    train_labels: tuple[int, ...]
    teacher_name: str = ""
    teacher_formula: str = ""
    teacher_fingerprint: str = ""


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.RANDOM_TRAIN_COUNTS = (8, 12)
    Config.RANDOM_DATASETS_PER_N = 2
    Config.SYMBOLIC_TRAIN_COUNTS = (8,)
    Config.SEED_COUNT = 4
    Config.MAX_STEPS = 2
    Config.EARLY_EVAL_STEPS = (0, 1, 2)
    Config.EVAL_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_STEPS = 1
    Config.BDD_RANDOM_ORDERS = 2
    Config.RANDOM_BASELINE_COUNT = 4
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/consensus_symbolicity/_smoke_8bit_consensus_symbolicity"
    )
    Config.OVERWRITE_RESULT_DIR = True


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
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


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


def truth_table_inputs() -> np.ndarray:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.uint16)
    shifts = np.arange(
        Config.INPUT_BITS - 1, -1, -1, dtype=np.uint16
    )
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float32)


def pack_truth(bits: np.ndarray) -> np.ndarray:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8), axis=-1, bitorder="little"
    )


def unpack_truth(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, bitorder="little"
    )[..., : 2 ** Config.INPUT_BITS]


def fingerprint_hex(bits_or_packed: np.ndarray, packed: bool = False) -> str:
    values = (
        np.asarray(bits_or_packed, dtype=np.uint8)
        if packed
        else pack_truth(np.asarray(bits_or_packed, dtype=np.uint8))
    )
    return values.tobytes().hex().upper()


def dataset_signature(indices: Sequence[int], labels: Sequence[int]) -> str:
    payload = json.dumps(
        list(zip(map(int, indices), map(int, labels))),
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def gf2_rank(rows: np.ndarray) -> int:
    values = np.asarray(rows, dtype=np.uint8).copy()
    rank = 0
    for column in range(values.shape[1]):
        candidates = np.flatnonzero(values[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        values[[rank, pivot]] = values[[pivot, rank]]
        for row in range(values.shape[0]):
            if row != rank and values[row, column]:
                values[row] ^= values[rank]
        rank += 1
        if rank == values.shape[0]:
            break
    return int(rank)


def pairwise_hamming_summary(rows: np.ndarray) -> tuple[float, float]:
    if len(rows) < 2:
        return 0.0, 0.0
    distances: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            distances.append(float(np.mean(rows[left] != rows[right])))
    return float(np.mean(distances)), float(np.std(distances))


def teacher_library(inputs: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    x = inputs.astype(np.uint8)
    weight = x.sum(axis=1)
    rules: dict[str, tuple[np.ndarray, str]] = {
        "copy_x0": (x[:, 0], "x0"),
        "not_x0": (1 - x[:, 0], "NOT x0"),
        "and_x0_x1": (x[:, 0] & x[:, 1], "x0 AND x1"),
        "or_x0_x1": (x[:, 0] | x[:, 1], "x0 OR x1"),
        "xor_x0_x1": (x[:, 0] ^ x[:, 1], "x0 XOR x1"),
        "parity3": (x[:, 0] ^ x[:, 1] ^ x[:, 2], "x0 XOR x1 XOR x2"),
        "parity8": (np.bitwise_xor.reduce(x, axis=1), "XOR(x0..x7)"),
        "majority3": (
            ((x[:, 0] + x[:, 1] + x[:, 2]) >= 2).astype(np.uint8),
            "x0+x1+x2 >= 2",
        ),
        "threshold8_ge4": ((weight >= 4).astype(np.uint8), "popcount(x) >= 4"),
        "exact_weight4": ((weight == 4).astype(np.uint8), "popcount(x) == 4"),
        "mux_x0_x1_x2": (
            np.where(x[:, 0] == 1, x[:, 1], x[:, 2]).astype(np.uint8),
            "IF x0 THEN x1 ELSE x2",
        ),
        "two_pair_or": (
            ((x[:, 0] & x[:, 1]) | (x[:, 2] & x[:, 3])).astype(np.uint8),
            "(x0 AND x1) OR (x2 AND x3)",
        ),
        "rule30_center": (
            (x[:, 3] ^ (x[:, 4] | x[:, 5])).astype(np.uint8),
            "x3 XOR (x4 OR x5)",
        ),
        "staircase_gf2": (
            ((x[:, 0] & x[:, 1]) ^ (x[:, 1] & x[:, 2])).astype(np.uint8),
            "(x0 AND x1) XOR (x1 AND x2)",
        ),
    }
    return rules


def balanced_subset_for_teacher(
    targets: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    half = count // 2
    positive_count = min(half, len(positives))
    negative_count = min(count - positive_count, len(negatives))
    if positive_count + negative_count < count:
        remaining = count - positive_count - negative_count
        if len(positives) - positive_count >= remaining:
            positive_count += remaining
        else:
            negative_count += remaining
    selected = np.concatenate((
        rng.choice(positives, size=positive_count, replace=False),
        rng.choice(negatives, size=negative_count, replace=False),
    ))
    rng.shuffle(selected)
    return selected.astype(np.int64), targets[selected].astype(np.uint8)


def build_datasets() -> tuple[list[DatasetSpec], list[dict[str, Any]]]:
    inputs = truth_table_inputs().astype(np.uint8)
    rng = np.random.default_rng(Config.DATASET_SEED)
    specs: list[DatasetSpec] = []
    seen: set[str] = set()

    for count in Config.RANDOM_TRAIN_COUNTS:
        if count % 2:
            raise ValueError("随机平衡训练集的样本数必须为偶数。")
        created = 0
        while created < Config.RANDOM_DATASETS_PER_N:
            indices = rng.choice(256, size=count, replace=False).astype(np.int64)
            labels = np.array(
                [0] * (count // 2) + [1] * (count // 2), dtype=np.uint8
            )
            rng.shuffle(labels)
            order = np.argsort(indices)
            indices = indices[order]
            labels = labels[order]
            signature = dataset_signature(indices, labels)
            if signature in seen:
                continue
            seen.add(signature)
            specs.append(DatasetSpec(
                name=f"random_n{count}_{created:03d}_{signature[:8]}",
                source_kind="random_balanced_partial_table",
                train_indices=tuple(map(int, indices)),
                train_labels=tuple(map(int, labels)),
            ))
            created += 1

    if Config.INCLUDE_SYMBOLIC_CONTROLS:
        for rule_index, (name, (targets, formula)) in enumerate(
            teacher_library(inputs).items()
        ):
            for count in Config.SYMBOLIC_TRAIN_COUNTS:
                local_rng = np.random.default_rng(
                    Config.DATASET_SEED + 10_000 + rule_index * 100 + count
                )
                indices, labels = balanced_subset_for_teacher(
                    targets, count, local_rng
                )
                order = np.argsort(indices)
                indices = indices[order]
                labels = labels[order]
                signature = dataset_signature(indices, labels)
                specs.append(DatasetSpec(
                    name=f"symbolic_{name}_n{count}_{signature[:8]}",
                    source_kind="hidden_symbolic_teacher_control",
                    train_indices=tuple(map(int, indices)),
                    train_labels=tuple(map(int, labels)),
                    teacher_name=name,
                    teacher_formula=formula,
                    teacher_fingerprint=fingerprint_hex(targets),
                ))

    rows: list[dict[str, Any]] = []
    for dataset_index, spec in enumerate(specs):
        indices = np.asarray(spec.train_indices, dtype=np.int64)
        labels = np.asarray(spec.train_labels, dtype=np.uint8)
        selected = inputs[indices]
        distance_mean, distance_std = pairwise_hamming_summary(selected)
        rows.append({
            "dataset_index": dataset_index,
            **asdict(spec),
            "train_count": len(indices),
            "positive_count": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "input_gf2_rank": gf2_rank(selected),
            "input_bit_mean": selected.mean(axis=0).tolist(),
            "input_hamming_weight_mean": float(selected.sum(axis=1).mean()),
            "pairwise_hamming_fraction_mean": distance_mean,
            "pairwise_hamming_fraction_std": distance_std,
        })
    return specs, rows


class BatchedPairedMLP(nn.Module):
    def __init__(self, dataset_count: int) -> None:
        super().__init__()
        dimensions = [Config.INPUT_BITS] + [Config.WIDTH] * Config.HIDDEN_LAYERS + [1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.INITIALIZATION_SEED)
        weights: list[nn.Parameter] = []
        biases: list[nn.Parameter] = []
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            bound = 1.0 / math.sqrt(fan_in)
            base_weight = torch.empty(
                Config.SEED_COUNT, fan_out, fan_in
            ).uniform_(-bound, bound, generator=generator)
            base_bias = torch.empty(
                Config.SEED_COUNT, fan_out
            ).uniform_(-bound, bound, generator=generator)
            weights.append(nn.Parameter(
                base_weight[None].expand(dataset_count, -1, -1, -1)
                .reshape(dataset_count * Config.SEED_COUNT, fan_out, fan_in)
                .clone()
            ))
            biases.append(nn.Parameter(
                base_bias[None].expand(dataset_count, -1, -1)
                .reshape(dataset_count * Config.SEED_COUNT, fan_out)
                .clone()
            ))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if layer_index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def build_training_tensors(
    specs: Sequence[DatasetSpec],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = truth_table_inputs()
    maximum = max(len(spec.train_indices) for spec in specs)
    train_x = np.zeros((len(specs), maximum, Config.INPUT_BITS), dtype=np.float32)
    train_y = np.zeros((len(specs), maximum), dtype=np.float32)
    valid = np.zeros((len(specs), maximum), dtype=np.float32)
    for dataset_index, spec in enumerate(specs):
        indices = np.asarray(spec.train_indices, dtype=np.int64)
        labels = np.asarray(spec.train_labels, dtype=np.float32)
        train_x[dataset_index, :len(indices)] = inputs[indices]
        train_y[dataset_index, :len(indices)] = labels
        valid[dataset_index, :len(indices)] = 1.0
    train_x = np.repeat(train_x, Config.SEED_COUNT, axis=0)
    train_y = np.repeat(train_y, Config.SEED_COUNT, axis=0)
    valid = np.repeat(valid, Config.SEED_COUNT, axis=0)
    return (
        torch.as_tensor(train_x, device=device),
        torch.as_tensor(train_y, device=device),
        torch.as_tensor(valid, device=device),
        torch.as_tensor(inputs, device=device),
    )


def distinct_collision(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total < 2:
        return float("nan")
    numerator = np.sum(counts.astype(np.float64) * (counts - 1))
    return float(numerator / (total * (total - 1)))


def plugin_entropy(counts: np.ndarray) -> float:
    probabilities = counts.astype(np.float64) / counts.sum()
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log2(positive)).sum())


def bit_agreement_distinct(predictions: np.ndarray) -> float:
    model_count = len(predictions)
    if model_count < 2 or predictions.shape[1] == 0:
        return float("nan")
    ones = predictions.sum(axis=0).astype(np.float64)
    same = ones * (ones - 1) + (model_count - ones) * (model_count - ones - 1)
    return float(np.mean(same / (model_count * (model_count - 1))))


def function_distribution_metrics(
    predictions: np.ndarray,
    fitted: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    cohort = predictions[fitted]
    if not len(cohort):
        cohort = predictions
        source = "all_models_no_fitted_cohort"
    else:
        source = "train_hard_exact_models"
    packed = pack_truth(cohort)
    unique, counts = np.unique(packed, axis=0, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    modal_packed = unique[0]
    modal_bits = unpack_truth(modal_packed[None])[0]
    hamming_to_modal = np.mean(cohort != modal_bits[None], axis=1)
    metrics = {
        "cohort_source": source,
        "cohort_model_count": int(len(cohort)),
        "unique_function_count": int(len(unique)),
        "modal_count": int(counts[0]),
        "modal_probability": float(counts[0] / len(cohort)),
        "function_collision": distinct_collision(counts),
        "function_entropy_plugin_bits": plugin_entropy(counts),
        "effective_function_count": float(2 ** plugin_entropy(counts)),
        "modal_fingerprint": fingerprint_hex(modal_packed, packed=True),
        "mean_hamming_to_modal_full": float(hamming_to_modal.mean()),
        "max_hamming_to_modal_full": float(hamming_to_modal.max()),
        "unseen_bit_agreement": bit_agreement_distinct(cohort[:, test_indices]),
    }
    return metrics, modal_packed


def evaluate(
    step: int,
    model: BatchedPairedMLP,
    specs: Sequence[DatasetSpec],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid: torch.Tensor,
    full_inputs: torch.Tensor,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        full_batch = full_inputs[None].expand(len(train_x), -1, -1)
        full_logits = model(full_batch)
        losses = (
            F.binary_cross_entropy_with_logits(
                train_logits, train_y, reduction="none"
            ) * valid
        ).sum(axis=1) / valid.sum(axis=1)
        train_predictions = train_logits >= 0
        train_exact = (
            ((train_predictions == (train_y >= 0.5)).float() * valid).sum(axis=1)
            == valid.sum(axis=1)
        )
        predictions = (full_logits >= 0).to(torch.uint8).cpu().numpy()
        logits_cpu = full_logits.to(torch.float16).cpu().numpy()
        losses_cpu = losses.cpu().numpy()
        exact_cpu = train_exact.cpu().numpy().astype(bool)

    all_states = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    modal_packed: list[np.ndarray] = []
    for dataset_index, spec in enumerate(specs):
        start = dataset_index * Config.SEED_COUNT
        stop = start + Config.SEED_COUNT
        local_predictions = predictions[start:stop]
        local_exact = exact_cpu[start:stop]
        train_indices = np.asarray(spec.train_indices, dtype=np.int64)
        test_indices = np.setdiff1d(
            all_states, train_indices, assume_unique=True
        )
        distribution, modal = function_distribution_metrics(
            local_predictions, local_exact, test_indices
        )
        modal_packed.append(modal)
        row: dict[str, Any] = {
            "step": step,
            "dataset_index": dataset_index,
            "dataset_name": spec.name,
            "source_kind": spec.source_kind,
            "train_count": len(spec.train_indices),
            "train_fit_rate": float(local_exact.mean()),
            "train_loss_mean": float(losses_cpu[start:stop].mean()),
            "train_loss_median": float(np.median(losses_cpu[start:stop])),
            **distribution,
        }
        if spec.teacher_fingerprint:
            teacher = np.frombuffer(
                bytes.fromhex(spec.teacher_fingerprint), dtype=np.uint8
            )
            teacher_bits = unpack_truth(teacher[None])[0]
            modal_bits = unpack_truth(modal[None])[0]
            row.update({
                "teacher_name": spec.teacher_name,
                "modal_teacher_accuracy": float(
                    np.mean(modal_bits == teacher_bits)
                ),
                "modal_is_teacher": bool(np.array_equal(modal_bits, teacher_bits)),
                "seed_teacher_exact_rate": float(
                    np.mean(np.all(local_predictions == teacher_bits[None], axis=1))
                ),
            })
        else:
            row.update({
                "teacher_name": "",
                "modal_teacher_accuracy": None,
                "modal_is_teacher": None,
                "seed_teacher_exact_rate": None,
            })
        rows.append(row)
    return (
        rows,
        np.asarray(modal_packed, dtype=np.uint8),
        pack_truth(predictions),
        logits_cpu,
    )


def anf_metrics(bits: np.ndarray) -> dict[str, Any]:
    coefficients = np.asarray(bits, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(2 ** Config.INPUT_BITS):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.array([int(value).bit_count() for value in terms], dtype=np.int64)
    term_count = int(len(terms))
    degree = int(degrees.max()) if term_count else 0
    literal_count = int(degrees.sum())
    formula = ""
    if term_count <= 12:
        rendered: list[str] = []
        for mask in terms:
            if mask == 0:
                rendered.append("1")
                continue
            names = [
                f"x{Config.INPUT_BITS - 1 - bit}"
                for bit in range(Config.INPUT_BITS)
                if int(mask) & (1 << bit)
            ]
            rendered.append("*".join(names))
        formula = " XOR ".join(rendered) if rendered else "0"
    return {
        "anf_degree": degree,
        "anf_term_count": term_count,
        "anf_literal_count": literal_count,
        "anf_formula_if_short": formula,
    }


def essential_variables(bits: np.ndarray) -> list[int]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    essential: list[int] = []
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        if np.any(bits[base] != bits[base | (1 << bit)]):
            essential.append(Config.INPUT_BITS - 1 - bit)
    return sorted(essential)


def boundary_metrics(bits: np.ndarray) -> dict[str, float]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    influences: list[float] = []
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        influences.append(float(np.mean(bits[base] != bits[base | (1 << bit)])))
    return {
        "total_influence": float(sum(influences)),
        "mean_edge_boundary_fraction": float(np.mean(influences)),
        "max_variable_influence": float(max(influences)),
    }


def subcube_formula(bits: np.ndarray, positive: bool) -> str:
    target = np.flatnonzero(bits == (1 if positive else 0))
    if not len(target):
        return ""
    fixed: list[tuple[int, int]] = []
    for bit in range(Config.INPUT_BITS):
        values = (target >> bit) & 1
        if np.all(values == values[0]):
            fixed.append((bit, int(values[0])))
    expected = []
    for value in range(2 ** Config.INPUT_BITS):
        if all(((value >> bit) & 1) == required for bit, required in fixed):
            expected.append(value)
    if not np.array_equal(np.asarray(expected, dtype=np.int64), target):
        return ""
    literals = []
    for bit, required in fixed:
        name = f"x{Config.INPUT_BITS - 1 - bit}"
        literals.append(name if required else f"NOT {name}")
    if positive:
        return " AND ".join(literals) if literals else "1"
    negated = []
    for bit, required in fixed:
        name = f"x{Config.INPUT_BITS - 1 - bit}"
        negated.append(f"NOT {name}" if required else name)
    return " OR ".join(negated) if negated else "0"


def named_symbolic_rule(bits: np.ndarray, anf: dict[str, Any]) -> tuple[str, str]:
    ones = int(bits.sum())
    domain = len(bits)
    if ones == 0:
        return "constant", "0"
    if ones == domain:
        return "constant", "1"
    conjunction = subcube_formula(bits, positive=True)
    if conjunction:
        return "literal_conjunction", conjunction
    disjunction = subcube_formula(bits, positive=False)
    if disjunction:
        return "literal_disjunction", disjunction
    if anf["anf_degree"] <= 1:
        return "affine_gf2", str(anf["anf_formula_if_short"])
    inputs = truth_table_inputs().astype(np.uint8)
    weights = inputs.sum(axis=1)
    pattern: list[int] = []
    symmetric = True
    for weight in range(Config.INPUT_BITS + 1):
        local = bits[weights == weight]
        if len(np.unique(local)) != 1:
            symmetric = False
            break
        pattern.append(int(local[0]))
    if symmetric:
        active = [index for index, value in enumerate(pattern) if value]
        return "symmetric_hamming_weight", f"popcount(x) IN {active}"
    return "", ""


def optimal_decision_tree(bits: np.ndarray) -> tuple[int, int]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def solve(fixed_mask: int, fixed_value: int) -> tuple[int, int]:
        selected = values[(values & fixed_mask) == fixed_value]
        outputs = bits[selected]
        if np.all(outputs == outputs[0]):
            return 1, 0
        best = (10 ** 9, 10 ** 9)
        for bit in range(Config.INPUT_BITS):
            bit_mask = 1 << bit
            if fixed_mask & bit_mask:
                continue
            left = solve(fixed_mask | bit_mask, fixed_value)
            right = solve(fixed_mask | bit_mask, fixed_value | bit_mask)
            candidate = (left[0] + right[0], 1 + max(left[1], right[1]))
            if candidate < best:
                best = candidate
        return best

    return solve(0, 0)


def robdd_node_count(bits: np.ndarray, order: Sequence[int]) -> int:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    unique_nodes: dict[tuple[int, int, int], int] = {}
    memo: dict[tuple[int, int, int], int] = {}

    def build(depth: int, fixed_mask: int, fixed_value: int) -> int:
        key = (depth, fixed_mask, fixed_value)
        if key in memo:
            return memo[key]
        selected = values[(values & fixed_mask) == fixed_value]
        outputs = bits[selected]
        if np.all(outputs == 0):
            return 0
        if np.all(outputs == 1):
            return 1
        bit = int(order[depth])
        bit_mask = 1 << bit
        low = build(depth + 1, fixed_mask | bit_mask, fixed_value)
        high = build(depth + 1, fixed_mask | bit_mask, fixed_value | bit_mask)
        if low == high:
            memo[key] = low
            return low
        node_key = (bit, low, high)
        node = unique_nodes.get(node_key)
        if node is None:
            node = len(unique_nodes) + 2
            unique_nodes[node_key] = node
        memo[key] = node
        return node

    build(0, 0, 0)
    return len(unique_nodes)


def bdd_orders() -> list[tuple[int, ...]]:
    natural = tuple(range(Config.INPUT_BITS - 1, -1, -1))
    reverse = tuple(reversed(natural))
    orders = [natural, reverse]
    rng = np.random.default_rng(Config.COMPLEXITY_SEED)
    seen = set(orders)
    while len(orders) < 2 + Config.BDD_RANDOM_ORDERS:
        candidate = tuple(map(int, rng.permutation(Config.INPUT_BITS)))
        if candidate not in seen:
            seen.add(candidate)
            orders.append(candidate)
    return orders


def complexity_metrics(bits: np.ndarray, orders: Sequence[Sequence[int]]) -> dict[str, Any]:
    bits = np.asarray(bits, dtype=np.uint8)
    anf = anf_metrics(bits)
    essential = essential_variables(bits)
    tree_leaves, tree_depth = optimal_decision_tree(bits)
    bdd_counts = [robdd_node_count(bits, order) for order in orders]
    named_family, named_formula = named_symbolic_rule(bits, anf)
    if named_family:
        tier = 1
    elif (
        len(essential) <= 3
        or tree_leaves <= 8
        or min(bdd_counts) <= 10
        or (
            anf["anf_term_count"] <= 8
            and anf["anf_literal_count"] <= 24
        )
    ):
        tier = 2
    elif (
        tree_leaves <= 32
        or min(bdd_counts) <= 32
        or anf["anf_term_count"] <= 32
    ):
        tier = 3
    else:
        tier = 4
    return {
        "truth_ones": int(bits.sum()),
        "truth_positive_rate": float(bits.mean()),
        "essential_variable_count": len(essential),
        "essential_variables": essential,
        **anf,
        **boundary_metrics(bits),
        "optimal_decision_tree_leaves": tree_leaves,
        "optimal_decision_tree_depth": tree_depth,
        "robdd_nodes_natural": bdd_counts[0],
        "robdd_nodes_reverse": bdd_counts[1],
        "robdd_nodes_min_tested": min(bdd_counts),
        "named_symbolic_family": named_family,
        "named_symbolic_formula": named_formula,
        "symbolic_screen_tier": tier,
        "symbolic_screen_readable": tier <= 2,
    }


def complexity_analysis(
    specs: Sequence[DatasetSpec],
    final_rows: Sequence[dict[str, Any]],
    modal_packed: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    orders = bdd_orders()
    cache: dict[str, dict[str, Any]] = {}
    modal_rows: list[dict[str, Any]] = []
    high_indices: list[int] = []
    for dataset_index, (spec, row, packed) in enumerate(
        zip(specs, final_rows, modal_packed)
    ):
        fingerprint = fingerprint_hex(packed, packed=True)
        metrics = cache.get(fingerprint)
        if metrics is None:
            bits = unpack_truth(packed[None])[0]
            metrics = complexity_metrics(bits, orders)
            cache[fingerprint] = metrics
        high_consensus = bool(
            float(row["train_fit_rate"]) >= Config.MIN_FIT_RATE
            and float(row["modal_probability"]) >= Config.HIGH_MODAL_PROBABILITY
            and float(row["function_collision"]) >= Config.HIGH_FUNCTION_COLLISION
        )
        if high_consensus:
            high_indices.append(dataset_index)
        modal_rows.append({
            "dataset_index": dataset_index,
            "dataset_name": spec.name,
            "source_kind": spec.source_kind,
            "train_count": len(spec.train_indices),
            "train_fit_rate": row["train_fit_rate"],
            "modal_probability": row["modal_probability"],
            "function_collision": row["function_collision"],
            "function_entropy_plugin_bits": row["function_entropy_plugin_bits"],
            "modal_fingerprint": fingerprint,
            "high_consensus": high_consensus,
            **metrics,
        })

    rng = np.random.default_rng(Config.COMPLEXITY_SEED + 1)
    baseline_rows: list[dict[str, Any]] = []
    reference_counts = [
        int(modal_rows[index]["truth_ones"])
        for index in high_indices
    ] or [128]
    for baseline_index in range(Config.RANDOM_BASELINE_COUNT):
        ones = reference_counts[baseline_index % len(reference_counts)]
        bits = np.zeros(256, dtype=np.uint8)
        if ones:
            selected = rng.choice(256, size=ones, replace=False)
            bits[selected] = 1
        baseline_rows.append({
            "baseline_index": baseline_index,
            "matched_truth_ones": ones,
            "fingerprint": fingerprint_hex(bits),
            **complexity_metrics(bits, orders),
        })

    percentile_metrics = (
        "anf_term_count",
        "anf_literal_count",
        "optimal_decision_tree_leaves",
        "robdd_nodes_min_tested",
        "total_influence",
    )
    for row in modal_rows:
        if not row["high_consensus"]:
            continue
        matched = [
            baseline for baseline in baseline_rows
            if int(baseline["matched_truth_ones"]) == int(row["truth_ones"])
        ]
        if not matched:
            matched = baseline_rows
        row["random_baseline_comparison_count"] = len(matched)
        for metric in percentile_metrics:
            candidate = float(row[metric])
            row[f"random_fraction_{metric}_le_candidate"] = float(np.mean([
                float(baseline[metric]) <= candidate for baseline in matched
            ]))

    candidates: list[dict[str, Any]] = []
    for row in modal_rows:
        if row["high_consensus"] and int(row["symbolic_screen_tier"]) == 4:
            candidates.append({
                **row,
                "candidate_status": (
                    "高共识Tier-4筛查候选；必须进行有界SAT/SMT电路合成认证"
                ),
            })
    return modal_rows, baseline_rows, candidates


def aggregate_consensus_rows(
    modal_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in modal_rows:
        key = (str(row["source_kind"]), int(row["train_count"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (source_kind, train_count), rows in sorted(groups.items()):
        high = [row for row in rows if row["high_consensus"]]
        output.append({
            "source_kind": source_kind,
            "train_count": train_count,
            "dataset_count": len(rows),
            "fit_qualified_count": sum(
                float(row["train_fit_rate"]) >= Config.MIN_FIT_RATE for row in rows
            ),
            "high_consensus_count": len(high),
            "high_consensus_rate": float(len(high) / len(rows)),
            "modal_probability_median": float(np.median([
                float(row["modal_probability"]) for row in rows
            ])),
            "modal_probability_max": float(max(
                float(row["modal_probability"]) for row in rows
            )),
            "function_collision_median": float(np.nanmedian([
                float(row["function_collision"]) for row in rows
            ])),
            "function_collision_max": float(np.nanmax([
                float(row["function_collision"]) for row in rows
            ])),
            **{
                f"high_consensus_tier_{tier}_count": sum(
                    int(row["symbolic_screen_tier"]) == tier for row in high
                )
                for tier in range(1, 5)
            },
        })
    return output


def save_plots(
    output_dir: Path,
    modal_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 绘图不是数值判决前提
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    random_rows = [
        row for row in modal_rows
        if row["source_kind"] == "random_balanced_partial_table"
    ]
    symbolic_rows = [
        row for row in modal_rows
        if row["source_kind"] == "hidden_symbolic_teacher_control"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for rows, label, color in (
        (random_rows, "random partial tables", "tab:blue"),
        (symbolic_rows, "hidden symbolic controls", "tab:orange"),
    ):
        if not rows:
            continue
        axes[0].scatter(
            [row["function_collision"] for row in rows],
            [row["optimal_decision_tree_leaves"] for row in rows],
            s=22,
            alpha=0.7,
            label=label,
            color=color,
        )
        axes[1].scatter(
            [row["modal_probability"] for row in rows],
            [row["anf_term_count"] for row in rows],
            s=22,
            alpha=0.7,
            label=label,
            color=color,
        )
    axes[0].axvline(Config.HIGH_FUNCTION_COLLISION, color="black", ls="--")
    axes[1].axvline(Config.HIGH_MODAL_PROBABILITY, color="black", ls="--")
    axes[0].set_xlabel("complete-function collision agreement")
    axes[0].set_ylabel("optimal decision-tree leaves")
    axes[1].set_xlabel("modal complete-function probability")
    axes[1].set_ylabel("ANF term count")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "consensus_vs_symbolic_complexity.png", dpi=180)
    plt.close(figure)


def package_results(output_dir: Path) -> Path:
    archive = output_dir.parent / f"{output_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir.parent))
    return archive


def main() -> None:
    apply_smoke_overrides()
    output_dir = prepare_result_dir()
    config = {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    }
    write_json(output_dir / "config.json", config)
    specs, dataset_rows = build_datasets()
    write_csv(output_dir / "datasets.csv", dataset_rows)

    device = torch.device(Config.DEVICE)
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    model = BatchedPairedMLP(len(specs)).to(device)
    train_x, train_y, valid, full_inputs = build_training_tensors(specs, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    model_count = len(specs) * Config.SEED_COUNT
    print("=== 8-bit Consensus Symbolicity Pilot ===", flush=True)
    print(
        f"device={device} | datasets={len(specs):,} | seeds/dataset={Config.SEED_COUNT:,} "
        f"| models={model_count:,} | train n={list(Config.RANDOM_TRAIN_COUNTS)}",
        flush=True,
    )
    print(
        f"MLP=8->{Config.WIDTH}x{Config.HIDDEN_LAYERS}->1 tanh | "
        f"max_steps={Config.MAX_STEPS:,}",
        flush=True,
    )

    trajectory_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    final_modal = np.empty((len(specs), 32), dtype=np.uint8)
    final_predictions = np.empty((model_count, 32), dtype=np.uint8)
    final_logits = np.empty((model_count, 256), dtype=np.float16)
    start_time = time.perf_counter()
    step = 0
    interrupted = False
    eval_steps = set(Config.EARLY_EVAL_STEPS)

    def run_evaluation(current_step: int) -> None:
        nonlocal final_rows, final_modal, final_predictions, final_logits
        rows, modal, predictions, logits = evaluate(
            current_step,
            model,
            specs,
            train_x,
            train_y,
            valid,
            full_inputs,
        )
        trajectory_rows.extend(rows)
        final_rows = rows
        final_modal = modal
        final_predictions = predictions
        final_logits = logits
        write_csv(output_dir / "trajectory.csv", trajectory_rows)
        write_csv(output_dir / "latest_dataset_summary.csv", rows)
        np.savez_compressed(
            output_dir / "latest_predictions_packed.npz",
            predictions_packed=predictions,
            modal_packed=modal,
            logits_float16=logits,
        )
        qualified = [
            row for row in rows
            if row["train_fit_rate"] >= Config.MIN_FIT_RATE
        ]
        high = [
            row for row in qualified
            if row["modal_probability"] >= Config.HIGH_MODAL_PROBABILITY
            and row["function_collision"] >= Config.HIGH_FUNCTION_COLLISION
        ]
        random_high = sum(
            row["source_kind"] == "random_balanced_partial_table" for row in high
        )
        print(
            f"step={current_step:>6,} | elapsed={time.perf_counter()-start_time:8.1f}s "
            f"| fitted datasets={len(qualified):>3}/{len(rows)} "
            f"| high consensus={len(high):>3} (random={random_high})",
            flush=True,
        )

    try:
        while step <= Config.MAX_STEPS:
            should_eval = (
                step in eval_steps
                or step % Config.EVAL_INTERVAL_STEPS == 0
                or step == Config.MAX_STEPS
            )
            if should_eval:
                run_evaluation(step)
            if step == Config.MAX_STEPS:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            per_model = (
                F.binary_cross_entropy_with_logits(
                    logits, train_y, reduction="none"
                ) * valid
            ).sum(axis=1) / valid.sum(axis=1)
            per_model.sum().backward()
            optimizer.step()
            step += 1
            if step % Config.SAVE_INTERVAL_STEPS == 0:
                write_json(output_dir / "progress.json", {
                    "step": step,
                    "elapsed_seconds": time.perf_counter() - start_time,
                })
    except KeyboardInterrupt:
        interrupted = True
        print("收到Ctrl+C，执行最终评估并安全保存。", flush=True)
        run_evaluation(step)

    modal_rows, baseline_rows, candidates = complexity_analysis(
        specs, final_rows, final_modal
    )
    write_csv(output_dir / "final_dataset_summary.csv", final_rows)
    write_csv(output_dir / "modal_complexity.csv", modal_rows)
    write_csv(output_dir / "random_complexity_baseline.csv", baseline_rows)
    write_csv(output_dir / "counterexample_candidates.csv", candidates)
    write_csv(
        output_dir / "consensus_by_condition.csv",
        aggregate_consensus_rows(modal_rows),
    )
    write_csv(
        output_dir / "top_consensus_datasets.csv",
        sorted(
            modal_rows,
            key=lambda row: (
                float(row["function_collision"]),
                float(row["modal_probability"]),
            ),
            reverse=True,
        )[:100],
    )
    np.savez_compressed(
        output_dir / "final_predictions_packed.npz",
        predictions_packed=final_predictions,
        logits_float16=final_logits,
        dataset_names=np.asarray([spec.name for spec in specs]),
        seed_count=np.asarray(Config.SEED_COUNT),
    )
    np.savez_compressed(
        output_dir / "modal_functions_packed.npz",
        modal_packed=final_modal,
        dataset_names=np.asarray([spec.name for spec in specs]),
    )
    save_plots(output_dir, modal_rows)

    high = [row for row in modal_rows if row["high_consensus"]]
    high_random = [
        row for row in high
        if row["source_kind"] == "random_balanced_partial_table"
    ]
    summary = {
        "status": "interrupted" if interrupted else "complete",
        "final_step": step,
        "elapsed_seconds": time.perf_counter() - start_time,
        "dataset_count": len(specs),
        "model_count": model_count,
        "high_consensus_count": len(high),
        "high_consensus_random_partial_table_count": len(high_random),
        "high_consensus_tier_counts": {
            str(tier): sum(int(row["symbolic_screen_tier"]) == tier for row in high)
            for tier in range(1, 5)
        },
        "counterexample_candidate_count": len(candidates),
        "strong_conjecture_pilot_verdict": (
            "screening_counterexample_found"
            if candidates
            else (
                "no_high_consensus_random_dataset"
                if not high_random
                else "no_screening_counterexample_among_observed_high_consensus_modes"
            )
        ),
        "warning": (
            "Tier-4只是反例候选；正式不可符号化结论需要有界SAT/SMT合成认证。"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "runtime.json", {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "elapsed_seconds": summary["elapsed_seconds"],
    })
    archive = package_results(output_dir) if Config.PACKAGE_RESULTS else None

    print("=== 最终判决 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录：{output_dir}", flush=True)
    if archive:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
