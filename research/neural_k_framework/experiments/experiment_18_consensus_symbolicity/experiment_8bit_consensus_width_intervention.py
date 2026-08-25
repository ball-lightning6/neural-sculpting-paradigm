"""随机高共识训练集的跨容量与标签干预确认。

固定首轮 Pilot 中唯一的 teacher-free 高共识 n=12 训练集，使用新的初始化批次
顺序扫描更宽/更深 MLP。每个架构内部对四个训练集分支使用完全配对初始化：

1. 原始 base n=12；
2. 在最大分歧未见点加入主模态一致标签；
3. 在同一点加入相反标签；
4. 在全体 seed 已一致的低信息点加入一致标签。

脚本保存全部 256-bit 函数指纹，检验符号吸引子是否跨容量存在，以及高信息
一致/冲突约束是否分别强化或重定向完整函数分布。
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
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    width: int
    hidden_layers: int
    activation: str
    seed_count: int


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    train_indices: tuple[int, ...]
    train_labels: tuple[int, ...]
    intervention_kind: str


class Config:
    INPUT_BITS = 8
    ARCHITECTURES = (
        ArchitectureSpec("tanh_w16_d2", 16, 2, "tanh", 4_096),
        ArchitectureSpec("tanh_w64_d2", 64, 2, "tanh", 2_048),
        ArchitectureSpec("tanh_w256_d2", 256, 2, "tanh", 512),
        ArchitectureSpec("tanh_w256_d3", 256, 3, "tanh", 512),
        ArchitectureSpec("gelu_w256_d3", 256, 3, "gelu", 512),
        ArchitectureSpec("tanh_w1024_d3", 1024, 3, "tanh", 64),
    )

    BASE_INDICES = (15, 34, 50, 69, 103, 106, 155, 170, 181, 187, 227, 253)
    BASE_LABELS = (1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0)
    HIGH_INFORMATION_INDEX = 1       # 00000001，原 64 seeds 为 61:3
    HIGH_INFORMATION_MODAL_LABEL = 1
    LOW_INFORMATION_INDEX = 8        # 00001000，原 64 seeds 为 0:64
    LOW_INFORMATION_MODAL_LABEL = 0

    INITIALIZATION_SEED = 2026082211
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 10_000
    EARLY_EVAL_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
    EVAL_INTERVAL_STEPS = 250
    SAVE_INTERVAL_STEPS = 2_000
    MIN_FIT_RATE = 0.95

    BDD_RANDOM_ORDERS = 16
    COMPLEXITY_SEED = 2026082212

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_8bit_consensus_width_intervention")
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.ARCHITECTURES = (
        ArchitectureSpec("tanh_w8_d2_smoke", 8, 2, "tanh", 4),
        ArchitectureSpec("gelu_w16_d2_smoke", 16, 2, "gelu", 4),
    )
    Config.MAX_STEPS = 2
    Config.EARLY_EVAL_STEPS = (0, 1, 2)
    Config.EVAL_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_STEPS = 1
    Config.BDD_RANDOM_ORDERS = 2
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/consensus_symbolicity/"
        "_smoke_8bit_consensus_width_intervention"
    )
    Config.OVERWRITE_RESULT_DIR = True


def json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
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
    values = np.arange(256, dtype=np.uint16)
    shifts = np.arange(7, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float32)


def canonical_symbolic_function(inputs: np.ndarray) -> np.ndarray:
    values = inputs.astype(np.uint8)
    count = values[:, 1] + values[:, 3] + values[:, 4]
    return ((count == 0) | ((count == 1) & (values[:, 6] == 1))).astype(np.uint8)


def pack_truth(bits: np.ndarray) -> np.ndarray:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8), axis=-1, bitorder="little"
    )


def unpack_truth(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, bitorder="little"
    )[..., :256]


def fingerprint_hex(bits_or_packed: np.ndarray, packed: bool = False) -> str:
    value = (
        np.asarray(bits_or_packed, dtype=np.uint8)
        if packed
        else pack_truth(np.asarray(bits_or_packed, dtype=np.uint8))
    )
    return value.tobytes().hex().upper()


def build_conditions() -> tuple[ConditionSpec, ...]:
    base_indices = tuple(map(int, Config.BASE_INDICES))
    base_labels = tuple(map(int, Config.BASE_LABELS))
    if len(set(base_indices)) != len(base_indices):
        raise ValueError("BASE_INDICES 中存在重复输入。")
    if Config.HIGH_INFORMATION_INDEX in base_indices:
        raise ValueError("高信息干预点已经位于base训练集。")
    if Config.LOW_INFORMATION_INDEX in base_indices:
        raise ValueError("低信息干预点已经位于base训练集。")

    def extended(name: str, index: int, label: int, kind: str) -> ConditionSpec:
        pairs = list(zip(base_indices, base_labels)) + [(int(index), int(label))]
        pairs.sort()
        return ConditionSpec(
            name=name,
            train_indices=tuple(index for index, _ in pairs),
            train_labels=tuple(label for _, label in pairs),
            intervention_kind=kind,
        )

    conditions = (
        ConditionSpec(
            "base_n12",
            base_indices,
            base_labels,
            "no_intervention",
        ),
        extended(
            "high_info_consistent_n13",
            Config.HIGH_INFORMATION_INDEX,
            Config.HIGH_INFORMATION_MODAL_LABEL,
            "max_disagreement_modal_consistent",
        ),
        extended(
            "high_info_conflict_n13",
            Config.HIGH_INFORMATION_INDEX,
            1 - Config.HIGH_INFORMATION_MODAL_LABEL,
            "max_disagreement_modal_conflict",
        ),
        extended(
            "low_info_consistent_n13",
            Config.LOW_INFORMATION_INDEX,
            Config.LOW_INFORMATION_MODAL_LABEL,
            "unanimous_modal_consistent",
        ),
    )

    inputs = truth_table_inputs()
    canonical = canonical_symbolic_function(inputs)
    for index, label in zip(base_indices, base_labels):
        if int(canonical[index]) != label:
            raise AssertionError("硬编码base训练集与canonical函数不一致。")
    if int(canonical[Config.HIGH_INFORMATION_INDEX]) != Config.HIGH_INFORMATION_MODAL_LABEL:
        raise AssertionError("高信息一致标签与canonical函数不一致。")
    if int(canonical[Config.LOW_INFORMATION_INDEX]) != Config.LOW_INFORMATION_MODAL_LABEL:
        raise AssertionError("低信息一致标签与canonical函数不一致。")
    return conditions


class BatchedPairedMLP(nn.Module):
    def __init__(
        self,
        architecture: ArchitectureSpec,
        condition_count: int,
    ) -> None:
        super().__init__()
        self.activation = architecture.activation
        dimensions = (
            [Config.INPUT_BITS]
            + [architecture.width] * architecture.hidden_layers
            + [1]
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            Config.INITIALIZATION_SEED
            + sum(ord(character) for character in architecture.name)
        )
        weights: list[nn.Parameter] = []
        biases: list[nn.Parameter] = []
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            bound = 1.0 / math.sqrt(fan_in)
            base_weight = torch.empty(
                architecture.seed_count, fan_out, fan_in
            ).uniform_(-bound, bound, generator=generator)
            base_bias = torch.empty(
                architecture.seed_count, fan_out
            ).uniform_(-bound, bound, generator=generator)
            weights.append(nn.Parameter(
                base_weight[None].expand(condition_count, -1, -1, -1)
                .reshape(condition_count * architecture.seed_count, fan_out, fan_in)
                .clone()
            ))
            biases.append(nn.Parameter(
                base_bias[None].expand(condition_count, -1, -1)
                .reshape(condition_count * architecture.seed_count, fan_out)
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
                if self.activation == "tanh":
                    hidden = torch.tanh(hidden)
                elif self.activation == "gelu":
                    hidden = F.gelu(hidden)
                elif self.activation == "relu":
                    hidden = F.relu(hidden)
                else:
                    raise ValueError(f"未知activation: {self.activation}")
        return hidden.squeeze(-1)


def parameter_count_per_model(architecture: ArchitectureSpec) -> int:
    dimensions = (
        [Config.INPUT_BITS]
        + [architecture.width] * architecture.hidden_layers
        + [1]
    )
    return int(sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:])
    ))


def build_training_tensors(
    conditions: Sequence[ConditionSpec],
    architecture: ArchitectureSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = truth_table_inputs()
    maximum = max(len(condition.train_indices) for condition in conditions)
    train_x = np.zeros((len(conditions), maximum, 8), dtype=np.float32)
    train_y = np.zeros((len(conditions), maximum), dtype=np.float32)
    valid = np.zeros((len(conditions), maximum), dtype=np.float32)
    for condition_index, condition in enumerate(conditions):
        indices = np.asarray(condition.train_indices, dtype=np.int64)
        labels = np.asarray(condition.train_labels, dtype=np.float32)
        train_x[condition_index, :len(indices)] = inputs[indices]
        train_y[condition_index, :len(indices)] = labels
        valid[condition_index, :len(indices)] = 1
    train_x = np.repeat(train_x, architecture.seed_count, axis=0)
    train_y = np.repeat(train_y, architecture.seed_count, axis=0)
    valid = np.repeat(valid, architecture.seed_count, axis=0)
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
    return float(
        np.sum(counts.astype(np.float64) * (counts - 1))
        / (total * (total - 1))
    )


def plugin_entropy(counts: np.ndarray) -> float:
    probability = counts.astype(np.float64) / counts.sum()
    positive = probability[probability > 0]
    return float(-(positive * np.log2(positive)).sum())


def anf_metrics(bits: np.ndarray) -> dict[str, Any]:
    coefficients = np.asarray(bits, dtype=np.uint8).copy()
    for bit in range(8):
        step = 1 << bit
        for mask in range(256):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.array([int(term).bit_count() for term in terms], dtype=np.int64)
    formula = ""
    if len(terms) <= 16:
        rendered: list[str] = []
        for mask in terms:
            if mask == 0:
                rendered.append("1")
                continue
            rendered.append("*".join(
                f"x{7-bit}" for bit in range(8) if int(mask) & (1 << bit)
            ))
        formula = " XOR ".join(rendered) if rendered else "0"
    return {
        "anf_degree": int(degrees.max()) if len(terms) else 0,
        "anf_term_count": int(len(terms)),
        "anf_literal_count": int(degrees.sum()),
        "anf_formula_if_short": formula,
    }


def essential_variables(bits: np.ndarray) -> list[int]:
    values = np.arange(256, dtype=np.int64)
    output: list[int] = []
    for bit in range(8):
        base = values[(values & (1 << bit)) == 0]
        if np.any(bits[base] != bits[base | (1 << bit)]):
            output.append(7 - bit)
    return sorted(output)


def optimal_decision_tree(bits: np.ndarray) -> tuple[int, int]:
    values = np.arange(256, dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def solve(fixed_mask: int, fixed_value: int) -> tuple[int, int]:
        selected = values[(values & fixed_mask) == fixed_value]
        outputs = bits[selected]
        if np.all(outputs == outputs[0]):
            return 1, 0
        best = (10 ** 9, 10 ** 9)
        for bit in range(8):
            bit_mask = 1 << bit
            if fixed_mask & bit_mask:
                continue
            low = solve(fixed_mask | bit_mask, fixed_value)
            high = solve(fixed_mask | bit_mask, fixed_value | bit_mask)
            candidate = (low[0] + high[0], 1 + max(low[1], high[1]))
            best = min(best, candidate)
        return best

    return solve(0, 0)


def robdd_node_count(bits: np.ndarray, order: Sequence[int]) -> int:
    values = np.arange(256, dtype=np.int64)
    unique: dict[tuple[int, int, int], int] = {}
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
        node = unique.get(node_key)
        if node is None:
            node = len(unique) + 2
            unique[node_key] = node
        memo[key] = node
        return node

    build(0, 0, 0)
    return len(unique)


def bdd_orders() -> list[tuple[int, ...]]:
    orders = [tuple(range(7, -1, -1)), tuple(range(8))]
    seen = set(orders)
    rng = np.random.default_rng(Config.COMPLEXITY_SEED)
    while len(orders) < 2 + Config.BDD_RANDOM_ORDERS:
        candidate = tuple(map(int, rng.permutation(8)))
        if candidate not in seen:
            seen.add(candidate)
            orders.append(candidate)
    return orders


def lightweight_complexity(bits: np.ndarray, orders: Sequence[Sequence[int]]) -> dict[str, Any]:
    leaves, depth = optimal_decision_tree(bits)
    bdd = [robdd_node_count(bits, order) for order in orders]
    return {
        "truth_ones": int(bits.sum()),
        "essential_variables": essential_variables(bits),
        "essential_variable_count": len(essential_variables(bits)),
        **anf_metrics(bits),
        "optimal_decision_tree_leaves": leaves,
        "optimal_decision_tree_depth": depth,
        "robdd_nodes_min_tested": min(bdd),
    }


def evaluate_architecture(
    step: int,
    architecture: ArchitectureSpec,
    conditions: Sequence[ConditionSpec],
    model: BatchedPairedMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid: torch.Tensor,
    full_inputs: torch.Tensor,
    canonical: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        full_logits = model(full_inputs[None].expand(len(train_x), -1, -1))
        losses = (
            F.binary_cross_entropy_with_logits(
                train_logits, train_y, reduction="none"
            ) * valid
        ).sum(axis=1) / valid.sum(axis=1)
        train_predictions = train_logits >= 0
        train_exact = (
            ((train_predictions == (train_y >= 0.5)).float() * valid).sum(axis=1)
            == valid.sum(axis=1)
        ).cpu().numpy().astype(bool)
        predictions = (full_logits >= 0).to(torch.uint8).cpu().numpy()
        losses_cpu = losses.cpu().numpy()

    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    modal_packed: list[np.ndarray] = []
    canonical_packed = pack_truth(canonical)
    orders = bdd_orders()
    for condition_index, condition in enumerate(conditions):
        start = condition_index * architecture.seed_count
        stop = start + architecture.seed_count
        local_predictions = predictions[start:stop]
        local_fit = train_exact[start:stop]
        cohort = local_predictions[local_fit]
        if not len(cohort):
            cohort = local_predictions
            cohort_source = "all_models_no_fitted_cohort"
        else:
            cohort_source = "train_hard_exact_models"
        packed = pack_truth(cohort)
        unique, counts = np.unique(packed, axis=0, return_counts=True)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]
        modal = unique[0]
        modal_bits = unpack_truth(modal[None])[0]
        modal_packed.append(modal)
        canonical_count = int(np.sum(np.all(packed == canonical_packed[None], axis=1)))
        condition_row = {
            "step": step,
            "architecture": architecture.name,
            "width": architecture.width,
            "hidden_layers": architecture.hidden_layers,
            "activation": architecture.activation,
            "seed_count": architecture.seed_count,
            "parameters_per_model": parameter_count_per_model(architecture),
            "condition": condition.name,
            "intervention_kind": condition.intervention_kind,
            "train_count": len(condition.train_indices),
            "train_fit_rate": float(local_fit.mean()),
            "train_loss_median": float(np.median(losses_cpu[start:stop])),
            "cohort_source": cohort_source,
            "cohort_count": len(cohort),
            "unique_function_count": len(unique),
            "modal_count": int(counts[0]),
            "modal_probability": float(counts[0] / len(cohort)),
            "function_collision": distinct_collision(counts),
            "function_entropy_plugin_bits": plugin_entropy(counts),
            "modal_fingerprint": fingerprint_hex(modal, packed=True),
            "canonical_count": canonical_count,
            "canonical_probability": float(canonical_count / len(cohort)),
            "modal_is_canonical": bool(np.array_equal(modal, canonical_packed)),
            "modal_canonical_accuracy": float(np.mean(modal_bits == canonical)),
            "mean_seed_canonical_accuracy": float(np.mean(cohort == canonical[None])),
            "high_info_query_one_rate": float(cohort[:, Config.HIGH_INFORMATION_INDEX].mean()),
            "low_info_query_one_rate": float(cohort[:, Config.LOW_INFORMATION_INDEX].mean()),
        }
        if step == Config.MAX_STEPS:
            condition_row.update(lightweight_complexity(modal_bits, orders))
        summary_rows.append(condition_row)
        for rank in range(min(8, len(unique))):
            bits = unpack_truth(unique[rank][None])[0]
            top_rows.append({
                "step": step,
                "architecture": architecture.name,
                "condition": condition.name,
                "rank": rank + 1,
                "count": int(counts[rank]),
                "probability": float(counts[rank] / len(cohort)),
                "fingerprint": fingerprint_hex(unique[rank], packed=True),
                "canonical_accuracy": float(np.mean(bits == canonical)),
                "is_canonical": bool(np.array_equal(unique[rank], canonical_packed)),
            })
    return summary_rows, top_rows, pack_truth(predictions)


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
    conditions = build_conditions()
    canonical = canonical_symbolic_function(truth_table_inputs())
    write_json(output_dir / "config.json", {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    })
    write_json(output_dir / "conditions.json", [asdict(item) for item in conditions])
    write_json(output_dir / "canonical_function.json", {
        "fingerprint": fingerprint_hex(canonical),
        "formula": (
            "f=1 iff s=0 or (s=1 and x6=1), where s=x1+x3+x4"
        ),
        "high_information_point": {
            "index": Config.HIGH_INFORMATION_INDEX,
            "bits": format(Config.HIGH_INFORMATION_INDEX, "08b"),
            "modal_label": Config.HIGH_INFORMATION_MODAL_LABEL,
        },
        "low_information_point": {
            "index": Config.LOW_INFORMATION_INDEX,
            "bits": format(Config.LOW_INFORMATION_INDEX, "08b"),
            "modal_label": Config.LOW_INFORMATION_MODAL_LABEL,
        },
    })

    device = torch.device(Config.DEVICE)
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    print("=== 8-bit Consensus Width/Intervention Confirmation ===", flush=True)
    print(f"device={device} | conditions={[item.name for item in conditions]}", flush=True)

    all_trajectory: list[dict[str, Any]] = []
    all_top: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    interrupted = False
    overall_start = time.perf_counter()

    for architecture_index, architecture in enumerate(Config.ARCHITECTURES):
        architecture_start = time.perf_counter()
        print(
            f"--- {architecture.name} | params/model={parameter_count_per_model(architecture):,} "
            f"| seeds={architecture.seed_count:,} | "
            f"models={architecture.seed_count * len(conditions):,} ---",
            flush=True,
        )
        model = BatchedPairedMLP(architecture, len(conditions)).to(device)
        train_x, train_y, valid, full_inputs = build_training_tensors(
            conditions, architecture, device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        step = 0
        eval_steps = set(Config.EARLY_EVAL_STEPS)
        final_predictions = np.empty(
            (len(conditions) * architecture.seed_count, 32), dtype=np.uint8
        )

        try:
            while step <= Config.MAX_STEPS:
                should_eval = (
                    step in eval_steps
                    or step % Config.EVAL_INTERVAL_STEPS == 0
                    or step == Config.MAX_STEPS
                )
                if should_eval:
                    rows, top, predictions = evaluate_architecture(
                        step,
                        architecture,
                        conditions,
                        model,
                        train_x,
                        train_y,
                        valid,
                        full_inputs,
                        canonical,
                    )
                    all_trajectory.extend(rows)
                    all_top.extend(top)
                    final_predictions = predictions
                    base = next(row for row in rows if row["condition"] == "base_n12")
                    support = next(
                        row for row in rows
                        if row["condition"] == "high_info_consistent_n13"
                    )
                    conflict = next(
                        row for row in rows
                        if row["condition"] == "high_info_conflict_n13"
                    )
                    print(
                        f"step={step:>6,} | fit(base/support/conflict)="
                        f"{base['train_fit_rate']:.3f}/{support['train_fit_rate']:.3f}/"
                        f"{conflict['train_fit_rate']:.3f} | canonical="
                        f"{base['canonical_probability']:.3f}/"
                        f"{support['canonical_probability']:.3f}/"
                        f"{conflict['canonical_probability']:.3f} | base collision="
                        f"{base['function_collision']:.3f}",
                        flush=True,
                    )
                    write_csv(output_dir / "trajectory.csv", all_trajectory)
                    write_csv(output_dir / "top_functions.csv", all_top)
                    np.savez_compressed(
                        output_dir / f"latest_{architecture.name}_predictions.npz",
                        predictions_packed=predictions,
                    )
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
                        "architecture_index": architecture_index,
                        "architecture": architecture.name,
                        "step": step,
                        "elapsed_seconds": time.perf_counter() - overall_start,
                    })
        except KeyboardInterrupt:
            interrupted = True
            print("收到Ctrl+C，保存当前架构后停止后续扫描。", flush=True)
            rows, top, predictions = evaluate_architecture(
                step,
                architecture,
                conditions,
                model,
                train_x,
                train_y,
                valid,
                full_inputs,
                canonical,
            )
            all_trajectory.extend(rows)
            all_top.extend(top)
            final_predictions = predictions

        np.savez_compressed(
            output_dir / f"final_{architecture.name}_predictions.npz",
            predictions_packed=final_predictions,
            condition_names=np.asarray([item.name for item in conditions]),
            seed_count=np.asarray(architecture.seed_count),
        )
        runtime_rows.append({
            "architecture": architecture.name,
            "final_step": step,
            "elapsed_seconds": time.perf_counter() - architecture_start,
            "parameters_per_model": parameter_count_per_model(architecture),
            "seed_count": architecture.seed_count,
        })
        del optimizer, model, train_x, train_y, valid, full_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if interrupted:
            break

    write_csv(output_dir / "trajectory.csv", all_trajectory)
    write_csv(output_dir / "top_functions.csv", all_top)
    write_csv(output_dir / "runtime_by_architecture.csv", runtime_rows)
    final_rows = [
        row for row in all_trajectory
        if int(row["step"]) == Config.MAX_STEPS
    ]
    write_csv(output_dir / "final_summary.csv", final_rows)
    summary = {
        "status": "interrupted" if interrupted else "complete",
        "elapsed_seconds": time.perf_counter() - overall_start,
        "architectures_completed": [row["architecture"] for row in runtime_rows],
        "canonical_formula": (
            "f=1 iff s=0 or (s=1 and x6=1), s=x1+x3+x4"
        ),
        "interpretation": {
            "base": "检验原符号吸引子是否跨容量/激活函数存在",
            "high_info_consistent": "预测提高canonical质量和函数碰撞",
            "high_info_conflict": "canonical数学上不再兼容，观察新吸引子",
            "low_info_consistent": "控制单纯增加一个已无分歧样本的效果",
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "runtime.json", {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "elapsed_seconds": summary["elapsed_seconds"],
    })
    archive = package_results(output_dir) if Config.PACKAGE_RESULTS else None
    print("=== 完成 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录：{output_dir}", flush=True)
    if archive:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
