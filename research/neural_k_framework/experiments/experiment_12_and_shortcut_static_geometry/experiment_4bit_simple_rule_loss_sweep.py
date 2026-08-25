"""
4-bit -> 1-bit 简单规则的全 loss 区间函数概率实验。

固定一个 GELU+LayerNorm MLP，一次采样 prior 后，对 copy / NOT / AND /
OR / XOR 五条预注册简单规则，在 n=10/12/14 的嵌套训练集上计算：

- 从完整 prior 到最低 0.01% raw-BCE tail 的目标 hard-function 概率；
- 相邻 percentile loss bins 中的目标概率；
- 目标函数在全部 65,536 个 hard function 中的概率排名；
- 先限定训练集 hard-exact 后的同类曲线。

主问题不是跨架构，也不预先要求严格单调，只如实观察：对于事先认定的
简单函数，概率是否随 raw BCE 降低相对增加，以及是否存在真实反向区间。

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
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 4
    WIDTH = 64
    HIDDEN_LAYERS = 3
    ACTIVATION = "gelu"
    USE_LAYER_NORM = True

    RULE_NAMES = (
        "copy_x1",
        "not_x1",
        "and_x1_x2",
        "or_x1_x2",
        "xor_x1_x2",
    )
    TRAIN_COUNTS = (10, 12, 14)
    DATA_SEED = 20260828

    MODEL_COUNT = 33_554_432
    MICRO_BATCH_SIZE = 4_096
    STORAGE_SHARD_SIZE = 262_144
    PRIOR_SEED = 93_000_001

    # 累计保留 raw BCE 最低的 fraction。0.0001 = 最低 0.01%。
    TAIL_FRACTIONS = (
        1.0, 0.9, 0.75, 0.5, 0.25,
        0.1, 0.05, 0.02, 0.01, 0.005,
        0.002, 0.001, 0.0005, 0.0002, 0.0001,
    )
    MIN_RELIABLE_COUNT = 50
    TOP_FUNCTIONS_PER_SLICE = 20

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_4bit_simple_rule_loss_sweep")
    PACKAGE_RESULTS = True
    INCLUDE_PRIOR_SHARDS_IN_ARCHIVE = False
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass(frozen=True)
class Condition:
    name: str
    rule_name: str
    train_count: int
    target_function_id: int
    train_indices: tuple[int, ...]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.RULE_NAMES = ("copy_x1", "and_x1_x2", "xor_x1_x2")
    Config.TRAIN_COUNTS = (10, 12, 14)
    Config.MODEL_COUNT = 16_384
    Config.MICRO_BATCH_SIZE = 512
    Config.STORAGE_SHARD_SIZE = 4_096
    Config.TAIL_FRACTIONS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
    Config.MIN_RELIABLE_COUNT = 4
    Config.TOP_FUNCTIONS_PER_SLICE = 5
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_4bit_simple_rule_loss_sweep"
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


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


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


def validate_config() -> None:
    if Config.INPUT_BITS != 4:
        raise ValueError("当前实验固定为 4-bit 输入。")
    if tuple(sorted(set(Config.TRAIN_COUNTS))) != tuple(Config.TRAIN_COUNTS):
        raise ValueError("TRAIN_COUNTS 必须严格递增且不重复。")
    if min(Config.TRAIN_COUNTS) < 1 or max(Config.TRAIN_COUNTS) >= 16:
        raise ValueError("训练样本数必须在 [1, 15]。")
    if Config.MODEL_COUNT < 1 or Config.MICRO_BATCH_SIZE < 1:
        raise ValueError("MODEL_COUNT/MICRO_BATCH_SIZE 必须为正。")
    if Config.STORAGE_SHARD_SIZE % Config.MICRO_BATCH_SIZE:
        raise ValueError("STORAGE_SHARD_SIZE 必须能被 MICRO_BATCH_SIZE 整除。")
    if Config.MODEL_COUNT % Config.STORAGE_SHARD_SIZE:
        raise ValueError("MODEL_COUNT 必须能被 STORAGE_SHARD_SIZE 整除。")
    valid_rules = set(build_rule_targets(truth_table_inputs()))
    unknown = sorted(set(Config.RULE_NAMES) - valid_rules)
    if unknown:
        raise ValueError(f"未知规则：{unknown}")


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def build_rule_targets(inputs: np.ndarray) -> dict[str, np.ndarray]:
    x1 = inputs[:, 0]
    x2 = inputs[:, 1]
    return {
        "copy_x1": x1.astype(np.uint8),
        "not_x1": (1 - x1).astype(np.uint8),
        "and_x1_x2": (x1 & x2).astype(np.uint8),
        "or_x1_x2": (x1 | x2).astype(np.uint8),
        "xor_x1_x2": (x1 ^ x2).astype(np.uint8),
    }


def outputs_to_function_id(outputs: np.ndarray) -> int:
    bits = np.asarray(outputs, dtype=np.uint64).reshape(-1)
    powers = np.left_shift(np.uint64(1), np.arange(len(bits), dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def function_bits(function_id: int) -> str:
    return "".join(str((function_id >> index) & 1) for index in range(16))


def choose_nested_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, tuple[int, ...]]:
    zeros = np.flatnonzero(labels == 0)
    ones = np.flatnonzero(labels == 1)
    rng.shuffle(zeros)
    rng.shuffle(ones)
    minority_is_one = len(ones) <= len(zeros)
    minority = ones if minority_is_one else zeros
    majority = zeros if minority_is_one else ones
    result: dict[int, tuple[int, ...]] = {}
    for count in Config.TRAIN_COUNTS:
        if len(zeros) == len(ones):
            zero_count = count // 2
            one_count = count - zero_count
        else:
            minority_count = min(len(minority), count)
            majority_count = count - minority_count
            if majority_count > len(majority):
                minority_count += majority_count - len(majority)
                majority_count = len(majority)
            if minority_is_one:
                one_count, zero_count = minority_count, majority_count
            else:
                zero_count, one_count = minority_count, majority_count
        selected = np.concatenate([zeros[:zero_count], ones[:one_count]])
        result[count] = tuple(sorted(int(value) for value in selected))
    for first, second in zip(Config.TRAIN_COUNTS[:-1], Config.TRAIN_COUNTS[1:]):
        if not set(result[first]).issubset(result[second]):
            raise RuntimeError("嵌套训练集构造失败。")
    return result


def build_conditions() -> tuple[list[Condition], dict[str, np.ndarray]]:
    inputs = truth_table_inputs()
    targets = build_rule_targets(inputs)
    conditions: list[Condition] = []
    for rule_index, rule_name in enumerate(Config.RULE_NAMES):
        labels = targets[rule_name]
        rng = np.random.default_rng(Config.DATA_SEED + rule_index * 1009)
        nested = choose_nested_indices(labels, rng)
        target_id = outputs_to_function_id(labels)
        for count in Config.TRAIN_COUNTS:
            conditions.append(Condition(
                name=f"{rule_name}_n{count}",
                rule_name=rule_name,
                train_count=count,
                target_function_id=target_id,
                train_indices=nested[count],
            ))
    return conditions, targets


def sample_uniform(
    shape: tuple[int, ...],
    bound: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(
        -bound, bound, generator=generator
    )


def sample_mlp_logits(
    count: int,
    inputs: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    device = inputs.device
    dimensions = [Config.INPUT_BITS] + [Config.WIDTH] * Config.HIDDEN_LAYERS + [1]
    hidden = inputs[None].expand(count, -1, -1)
    for layer_index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        bound = 1.0 / math.sqrt(input_dim)
        weight = sample_uniform(
            (count, output_dim, input_dim), bound, generator, device
        )
        bias = sample_uniform((count, output_dim), bound, generator, device)
        hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None, :]
        if layer_index < len(dimensions) - 2:
            hidden = F.gelu(hidden)
            if Config.USE_LAYER_NORM:
                hidden = F.layer_norm(hidden, (Config.WIDTH,))
    return hidden.squeeze(-1)


def function_ids_from_logits(logits: torch.Tensor) -> np.ndarray:
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=logits.device),
        torch.arange(16, dtype=torch.int64, device=logits.device),
    )
    identifiers = ((logits > 0).to(torch.int64) * powers[None]).sum(dim=1)
    return identifiers.cpu().numpy().astype(np.uint16)


def sample_prior(
    output_dir: Path,
    device: torch.device,
) -> tuple[list[Path], np.ndarray]:
    shard_dir = output_dir / "prior_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    inputs = torch.from_numpy(truth_table_inputs().astype(np.float32)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.PRIOR_SEED)
    baseline_counts = np.zeros(65_536, dtype=np.int64)
    shard_paths: list[Path] = []
    started = time.perf_counter()
    shard_index = 0
    for shard_start in range(0, Config.MODEL_COUNT, Config.STORAGE_SHARD_SIZE):
        logits_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        shard_end = shard_start + Config.STORAGE_SHARD_SIZE
        for micro_start in range(
            shard_start, shard_end, Config.MICRO_BATCH_SIZE
        ):
            logits = sample_mlp_logits(
                Config.MICRO_BATCH_SIZE, inputs, generator
            )
            ids = function_ids_from_logits(logits)
            logits_parts.append(logits.cpu().numpy().astype(np.float16))
            id_parts.append(ids)
            baseline_counts += np.bincount(ids, minlength=65_536)
            del logits, ids
        shard_logits = np.concatenate(logits_parts)
        shard_ids = np.concatenate(id_parts)
        path = shard_dir / (
            f"shard_{shard_index:04d}_{shard_start:09d}_{shard_end:09d}.npz"
        )
        # prior logits 近似不可压缩；使用无压缩 npz 避免 CPU 压缩成为瓶颈。
        np.savez(path, logits=shard_logits, function_ids=shard_ids)
        shard_paths.append(path)
        shard_index += 1
        elapsed = time.perf_counter() - started
        print(
            f"prior {shard_end:,}/{Config.MODEL_COUNT:,} | "
            f"{shard_end / max(elapsed, 1e-9):.1f} models/s | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    np.savez_compressed(
        output_dir / "baseline_function_counts.npz",
        counts=baseline_counts,
    )
    return shard_paths, baseline_counts


def build_loss_cache(
    output_dir: Path,
    shard_paths: Sequence[Path],
    conditions: Sequence[Condition],
    targets: dict[str, np.ndarray],
) -> dict[str, Path]:
    cache_dir = output_dir / "loss_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        condition.name: cache_dir / f"{condition.name}.float32.mmap"
        for condition in conditions
    }
    maps = {
        name: np.memmap(path, mode="w+", dtype=np.float32, shape=(Config.MODEL_COUNT,))
        for name, path in paths.items()
    }
    cursor = 0
    for shard_number, path in enumerate(shard_paths, start=1):
        with np.load(path) as payload:
            logits = payload["logits"].astype(np.float32)
        softplus = np.logaddexp(0.0, logits)
        end = cursor + len(logits)
        for condition in conditions:
            indices = np.asarray(condition.train_indices, dtype=np.int64)
            labels = targets[condition.rule_name][indices].astype(np.float32)
            loss = (
                softplus[:, indices].sum(axis=1)
                - (logits[:, indices] * labels[None]).sum(axis=1)
            ) / condition.train_count
            maps[condition.name][cursor:end] = loss.astype(np.float32)
        cursor = end
        print(
            f"loss cache shard {shard_number}/{len(shard_paths)}",
            flush=True,
        )
    for values in maps.values():
        values.flush()
    del maps
    return paths


def load_all_function_ids(shard_paths: Sequence[Path]) -> np.ndarray:
    identifiers = np.empty(Config.MODEL_COUNT, dtype=np.uint16)
    cursor = 0
    for path in shard_paths:
        with np.load(path) as payload:
            values = payload["function_ids"]
        identifiers[cursor : cursor + len(values)] = values
        cursor += len(values)
    return identifiers


def hard_exact_mask(
    function_ids: np.ndarray,
    condition: Condition,
    target: np.ndarray,
) -> np.ndarray:
    mask = np.ones(len(function_ids), dtype=bool)
    for index in condition.train_indices:
        prediction = (function_ids >> np.uint16(index)) & np.uint16(1)
        mask &= prediction == np.uint16(target[index])
    return mask


def rank_interval(counts: np.ndarray, target_id: int) -> tuple[int, int]:
    target_count = int(counts[target_id])
    better = int(np.sum(counts > target_count))
    equal = int(np.sum(counts == target_count))
    return better + 1, better + equal


def analyze_cohort(
    condition: Condition,
    cohort_name: str,
    losses: np.ndarray,
    function_ids: np.ndarray,
    source_indices: np.ndarray | None,
    baseline_counts: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    if source_indices is None:
        source_loss = np.asarray(losses, dtype=np.float32)
        source_ids = function_ids.astype(np.int64)
    else:
        source_loss = np.asarray(losses[source_indices], dtype=np.float32)
        source_ids = function_ids[source_indices].astype(np.int64)
    order = np.argsort(source_loss, kind="stable")
    sorted_loss = source_loss[order]
    sorted_ids = source_ids[order]
    source_count = len(sorted_ids)
    target_id = condition.target_function_id
    prior_probability = baseline_counts[target_id] / Config.MODEL_COUNT
    source_counts = np.bincount(source_ids, minlength=65_536)
    source_probability = source_counts[target_id] / max(source_count, 1)

    fractions = sorted(set(Config.TAIL_FRACTIONS))
    cumulative_counts = np.zeros(65_536, dtype=np.int64)
    count_snapshots: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    previous = 0
    for fraction in fractions:
        selected_count = min(
            source_count,
            max(1, int(math.ceil(source_count * fraction))),
        )
        if selected_count < previous:
            raise RuntimeError("TAIL_FRACTIONS 内部排序错误。")
        cumulative_counts += np.bincount(
            sorted_ids[previous:selected_count], minlength=65_536
        )
        previous = selected_count
        if selected_count < Config.MIN_RELIABLE_COUNT:
            continue
        target_count = int(cumulative_counts[target_id])
        target_probability = target_count / selected_count
        rank_low, rank_high = rank_interval(cumulative_counts, target_id)
        row = {
            "condition": condition.name,
            "rule": condition.rule_name,
            "train_count": condition.train_count,
            "cohort": cohort_name,
            "retained_fraction": fraction,
            "source_count": source_count,
            "selected_count": selected_count,
            "loss_min": float(sorted_loss[0]),
            "loss_max": float(sorted_loss[selected_count - 1]),
            "loss_mean": float(np.mean(sorted_loss[:selected_count])),
            "target_function_id": target_id,
            "target_function_hex": f"0x{target_id:04X}",
            "target_count": target_count,
            "target_probability": target_probability,
            "prior_probability": prior_probability,
            "source_probability": source_probability,
            "enrichment_vs_prior": (
                target_probability / prior_probability
                if prior_probability > 0 else None
            ),
            "enrichment_vs_source": (
                target_probability / source_probability
                if source_probability > 0 else None
            ),
            "target_rank_low": rank_low,
            "target_rank_high": rank_high,
        }
        rows.append(row)
        top_ids = np.argsort(cumulative_counts)[::-1][
            : Config.TOP_FUNCTIONS_PER_SLICE
        ]
        for rank, function_id in enumerate(top_ids, start=1):
            count = int(cumulative_counts[function_id])
            if count == 0:
                break
            top_rows.append({
                "condition": condition.name,
                "cohort": cohort_name,
                "retained_fraction": fraction,
                "rank": rank,
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "truth_table_x0_to_x15": function_bits(int(function_id)),
                "count": count,
                "probability": count / selected_count,
                "is_target": int(function_id) == target_id,
            })
        count_snapshots.append(cumulative_counts.copy().astype(np.int32))

    count_matrix = np.stack(count_snapshots) if count_snapshots else np.empty(
        (0, 65_536), dtype=np.int32
    )
    return rows, top_rows, count_matrix


def analyze_loss_bins(
    condition: Condition,
    losses: np.ndarray,
    function_ids: np.ndarray,
    baseline_probability: float,
) -> list[dict[str, Any]]:
    order = np.argsort(losses, kind="stable")
    sorted_loss = np.asarray(losses[order], dtype=np.float32)
    sorted_ids = function_ids[order]
    fractions = sorted(set(Config.TAIL_FRACTIONS))
    boundaries = sorted(set([0.0] + fractions))
    rows: list[dict[str, Any]] = []
    for low_fraction, high_fraction in zip(boundaries[:-1], boundaries[1:]):
        start = int(math.floor(len(order) * low_fraction))
        end = int(math.ceil(len(order) * high_fraction))
        end = min(end, len(order))
        if end - start < Config.MIN_RELIABLE_COUNT:
            continue
        ids = sorted_ids[start:end]
        target_count = int(np.sum(ids == condition.target_function_id))
        probability = target_count / len(ids)
        rows.append({
            "condition": condition.name,
            "rule": condition.rule_name,
            "train_count": condition.train_count,
            "loss_quantile_low": low_fraction,
            "loss_quantile_high": high_fraction,
            "bin_count": len(ids),
            "loss_min": float(sorted_loss[start]),
            "loss_max": float(sorted_loss[end - 1]),
            "target_count": target_count,
            "target_probability": probability,
            "prior_probability": baseline_probability,
            "enrichment_vs_prior": (
                probability / baseline_probability
                if baseline_probability > 0 else None
            ),
        })
    return rows


def analyze_all(
    output_dir: Path,
    shard_paths: Sequence[Path],
    loss_paths: dict[str, Path],
    conditions: Sequence[Condition],
    targets: dict[str, np.ndarray],
    baseline_counts: np.ndarray,
) -> dict[str, Any]:
    function_ids = load_all_function_ids(shard_paths)
    all_rows: list[dict[str, Any]] = []
    all_top_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for condition in conditions:
        print(f"analyze {condition.name}", flush=True)
        losses = np.memmap(
            loss_paths[condition.name],
            mode="r",
            dtype=np.float32,
            shape=(Config.MODEL_COUNT,),
        )
        condition_dir = output_dir / "conditions" / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        rows, top_rows, counts = analyze_cohort(
            condition,
            "all_prior",
            losses,
            function_ids,
            None,
            baseline_counts,
            output_dir,
        )
        all_rows.extend(rows)
        all_top_rows.extend(top_rows)
        np.savez_compressed(
            condition_dir / "all_prior_function_counts.npz",
            counts=counts,
            retained_fractions=np.asarray([
                row["retained_fraction"] for row in rows
            ]),
        )

        exact = hard_exact_mask(
            function_ids, condition, targets[condition.rule_name]
        )
        exact_indices = np.flatnonzero(exact)
        exact_rows, exact_top, exact_counts = analyze_cohort(
            condition,
            "train_hard_exact",
            losses,
            function_ids,
            exact_indices,
            baseline_counts,
            output_dir,
        )
        all_rows.extend(exact_rows)
        all_top_rows.extend(exact_top)
        np.savez_compressed(
            condition_dir / "hard_exact_function_counts.npz",
            counts=exact_counts,
            retained_fractions=np.asarray([
                row["retained_fraction"] for row in exact_rows
            ]),
        )
        prior_probability = (
            baseline_counts[condition.target_function_id] / Config.MODEL_COUNT
        )
        bin_rows = analyze_loss_bins(
            condition, losses, function_ids, prior_probability
        )
        all_bin_rows.extend(bin_rows)
        condition_summaries.append({
            "condition": condition.name,
            "rule": condition.rule_name,
            "train_count": condition.train_count,
            "target_function_id": condition.target_function_id,
            "target_function_hex": f"0x{condition.target_function_id:04X}",
            "truth_table_x0_to_x15": function_bits(
                condition.target_function_id
            ),
            "train_indices": condition.train_indices,
            "train_labels": [
                int(targets[condition.rule_name][index])
                for index in condition.train_indices
            ],
            "prior_target_count": int(
                baseline_counts[condition.target_function_id]
            ),
            "prior_target_probability": prior_probability,
            "hard_exact_count": int(len(exact_indices)),
            "compatible_function_count": 2 ** (16 - condition.train_count),
        })
        del losses, exact, exact_indices

    write_csv(output_dir / "target_probability_curves.csv", all_rows)
    write_csv(output_dir / "top_functions_by_loss.csv", all_top_rows)
    write_csv(output_dir / "target_probability_loss_bins.csv", all_bin_rows)
    write_csv(output_dir / "condition_summary.csv", condition_summaries)
    make_plots(output_dir, all_rows, all_bin_rows)

    verdict_rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows = [
            row for row in all_rows
            if row["condition"] == condition.name
            and row["cohort"] == "all_prior"
        ]
        ordered = sorted(
            rows, key=lambda row: row["retained_fraction"], reverse=True
        )
        prior_target_count = int(
            baseline_counts[condition.target_function_id]
        )
        testable = prior_target_count >= Config.MIN_RELIABLE_COUNT
        probabilities = np.asarray([
            row["target_probability"] for row in ordered
        ])
        monotone = (
            bool(np.all(np.diff(probabilities) >= -1e-15))
            if testable else None
        )
        deepest = ordered[-1]
        verdict_rows.append({
            "condition": condition.name,
            "rule": condition.rule_name,
            "train_count": condition.train_count,
            "testable": testable,
            "prior_target_count": prior_target_count,
            "loss_depth_monotone": monotone,
            "deepest_fraction": deepest["retained_fraction"],
            "deepest_target_count": deepest["target_count"],
            "deepest_target_probability": deepest["target_probability"],
            "deepest_enrichment_vs_prior": deepest["enrichment_vs_prior"],
            "deepest_target_rank_low": deepest["target_rank_low"],
            "deepest_target_rank_high": deepest["target_rank_high"],
        })
    write_csv(output_dir / "verdict.csv", verdict_rows)
    return {
        "conditions": condition_summaries,
        "verdicts": verdict_rows,
        "all_prior_monotone_count": sum(
            row["loss_depth_monotone"] is True for row in verdict_rows
        ),
        "all_prior_testable_count": sum(
            row["testable"] for row in verdict_rows
        ),
        "all_prior_condition_count": len(verdict_rows),
    }


def make_plots(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    bin_rows: Sequence[dict[str, Any]],
) -> None:
    for rule_name in Config.RULE_NAMES:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for count in Config.TRAIN_COUNTS:
            group = sorted(
                [
                    row for row in rows
                    if row["rule"] == rule_name
                    and row["train_count"] == count
                    and row["cohort"] == "all_prior"
                ],
                key=lambda row: row["retained_fraction"],
                reverse=True,
            )
            axes[0].plot(
                [row["retained_fraction"] for row in group],
                [row["target_probability"] for row in group],
                marker="o",
                label=f"n={count}",
            )
            axes[1].plot(
                [row["retained_fraction"] for row in group],
                [row["enrichment_vs_prior"] for row in group],
                marker="o",
                label=f"n={count}",
            )
            bins = sorted(
                [
                    row for row in bin_rows
                    if row["rule"] == rule_name
                    and row["train_count"] == count
                ],
                key=lambda row: row["loss_quantile_low"],
            )
            axes[2].plot(
                [
                    math.sqrt(
                        max(row["loss_quantile_low"], 1e-8)
                        * row["loss_quantile_high"]
                    )
                    for row in bins
                ],
                [row["enrichment_vs_prior"] for row in bins],
                marker="o",
                label=f"n={count}",
            )
        for axis in axes[:2]:
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.set_xlabel("Retained lowest-loss fraction")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("Target hard-function probability")
        axes[0].set_title("Cumulative low-loss tails")
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Probability / prior probability")
        axes[1].set_title("Cumulative enrichment")
        axes[2].set_xscale("log")
        axes[2].invert_xaxis()
        axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[2].set_xlabel("Disjoint loss-percentile bin")
        axes[2].set_ylabel("Probability / prior probability")
        axes[2].set_title("Non-overlapping loss bins")
        axes[2].grid(alpha=0.25)
        axes[2].legend()
        fig.suptitle(rule_name)
        fig.tight_layout()
        fig.savefig(output_dir / f"{rule_name}_loss_sweep.png", dpi=180)
        plt.close(fig)


def create_slim_archive(output_dir: Path) -> Path:
    archive = Path(str(output_dir) + "_package.zip")
    excluded = {"prior_shards", "loss_cache"}
    with zipfile.ZipFile(
        archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in output_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(output_dir)
            if any(part in excluded for part in relative.parts):
                continue
            handle.write(path, arcname=str(relative))
    return archive


def main() -> None:
    apply_smoke_overrides()
    validate_config()
    torch.manual_seed(Config.DATA_SEED)
    np.random.seed(Config.DATA_SEED)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    device = torch.device(Config.DEVICE)
    output_dir = prepare_result_dir()
    write_json(output_dir / "config.json", config_dict())
    conditions, targets = build_conditions()
    write_json(output_dir / "conditions.json", [
        {
            **condition.__dict__,
            "target_truth_table": function_bits(condition.target_function_id),
            "train_labels": [
                int(targets[condition.rule_name][index])
                for index in condition.train_indices
            ],
        }
        for condition in conditions
    ])

    print("=== 4-bit simple-rule full-loss sweep ===", flush=True)
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"MLP：4 -> {Config.WIDTH} x {Config.HIDDEN_LAYERS} -> 1 | "
        f"models={Config.MODEL_COUNT:,}",
        flush=True,
    )
    print(f"结果目录：{output_dir}", flush=True)

    started = time.perf_counter()
    shard_paths, baseline_counts = sample_prior(output_dir, device)
    loss_paths = build_loss_cache(
        output_dir, shard_paths, conditions, targets
    )
    summary = analyze_all(
        output_dir,
        shard_paths,
        loss_paths,
        conditions,
        targets,
        baseline_counts,
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["model_count"] = Config.MODEL_COUNT
    summary["architecture"] = (
        f"gelu_ln_mlp_{Config.WIDTH}x{Config.HIDDEN_LAYERS}"
    )
    summary["primary_claim"] = (
        "预注册简单 hard function 的概率是否随 raw BCE 降低相对增加。"
    )
    write_json(output_dir / "summary.json", summary)

    archive: Path | None = None
    if Config.PACKAGE_RESULTS:
        archive = create_slim_archive(output_dir)
    print("\n=== 完成 ===", flush=True)
    print(
        f"单调条件：{summary['all_prior_monotone_count']}/"
        f"{summary['all_prior_testable_count']} 可裁决；"
        f"总条件={summary['all_prior_condition_count']}",
        flush=True,
    )
    if archive:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
