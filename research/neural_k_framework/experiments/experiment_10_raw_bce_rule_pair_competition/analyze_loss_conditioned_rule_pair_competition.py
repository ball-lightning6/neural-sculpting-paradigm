#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
有限训练 loss 下简单/复杂规则的直接竞争实验。

本脚本不重新采样网络。它复用
experiment_loss_conditioned_prior_scaling_4bit.py 已经写入磁盘的完整 4-bit
prior logits shards，并构造满足下列条件的函数对：

1. simple 与 complex 在训练样本上逐点完全一致；
2. 两者完整 truth table 的 Hamming weight 相同；
3. complex 由 simple 加入 2 或 4 个离散例外得到；
4. complex 在 ANF terms、最小 normal-form literals、最小 decision-tree
   leaves 和 multi-proxy rank 上都严格更复杂；
5. 训练样本只从二者 agreement set 中抽取，使用嵌套 k=10/12，二例外
   版本额外使用 k=14。

主判别量不是平均复杂度，而是 simple/complex 函数概率比相对于 hard-exact
baseline 的变化：

    odds_enrichment(q) =
      [P_q(simple) / P_q(complex)] /
      [P_hard(simple) / P_hard(complex)]

其中 q 是保留的最低训练 loss 分位。若 lower loss 在数据已经充分时确实偏向
更简单的兼容规则，应观察到 odds_enrichment 随 q 减小系统性大于 1。

raw BCE 是与真实训练目标一致的唯一主判决口径。RMS-normalized BCE 和
fixed-logit-scale 仅作为机制诊断，用于分析整体 logit scale 是否参与效应；
它们不属于主命题，也不能用来否定 raw BCE 命题。
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class Config:
    SOURCE_RESULT_DIR: Path = Path(
        "/root/results_loss_conditioned_prior_scaling_4bit"
    )
    RESULT_DIR: Path = Path(
        "/root/results_loss_conditioned_rule_pair_competition_reaudited"
    )
    ARCHITECTURES: tuple[str, ...] = (
        "tanh16x2",
        "gelu_ln1024x3",
    )
    CONDITION_SEED: int = 20260824

    QUANTILE_FRACTIONS: tuple[float, ...] = (
        1.0,
        0.5,
        0.2,
        0.1,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.001,
    )
    FIXED_SCALE_QUANTILES: tuple[float, float] = (0.40, 0.60)
    MIN_SELECTED_MODELS: int = 50
    MIN_BASELINE_COUNT_PER_RULE: int = 20
    MIN_EXPECTED_TAIL_COUNT_PER_RULE: float = 20.0

    CREATE_ARCHIVE: bool = True
    GENERATE_PLOTS: bool = True
    _LOCAL_SOURCE_DIR: Path = field(
        default=Path(
            "research/function_information_conservation/"
            "_smoke_loss_conditioned_prior_scaling_4bit"
        ),
        repr=False,
    )
    _LOCAL_RESULT_DIR: Path = field(
        default=Path(
            "research/function_information_conservation/"
            "_smoke_loss_conditioned_rule_pair_competition_reaudited"
        ),
        repr=False,
    )


@dataclass(frozen=True)
class RulePair:
    name: str
    simple_name: str
    variant: str
    simple_function_id: int
    complex_function_id: int
    differing_indices: tuple[int, ...]
    agreement_indices: tuple[int, ...]


@dataclass(frozen=True)
class PairCondition:
    name: str
    pair_name: str
    simple_name: str
    variant: str
    constraint_size: int
    input_indices: tuple[int, ...]
    targets: tuple[int, ...]
    simple_function_id: int
    complex_function_id: int


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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
                seen.add(key)
                fieldnames.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def function_bits(function_id: int) -> np.ndarray:
    shifts = np.arange(16, dtype=np.uint32)
    return ((np.uint32(function_id) >> shifts) & 1).astype(np.uint8)


def outputs_to_id(outputs: np.ndarray) -> int:
    return int(
        np.sum(
            outputs.astype(np.uint64)
            * np.left_shift(
                np.uint64(1), np.arange(len(outputs), dtype=np.uint64)
            )
        )
    )


def canonical_simple_rules() -> dict[str, int]:
    values = np.arange(16, dtype=np.uint8)
    bits = ((values[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
    return {
        "x0": outputs_to_id(bits[:, 0]),
        "and_x0_x1": outputs_to_id(bits[:, 0] & bits[:, 1]),
        "or_x0_x1": outputs_to_id(bits[:, 0] | bits[:, 1]),
    }


def build_rule_pairs() -> list[RulePair]:
    simple = canonical_simple_rules()
    # moderate：交换一个 0/1 输出，共两个例外，保持完整 Hamming weight。
    # high：交换两个 0/1 输出，共四个例外，同样保持 Hamming weight。
    definitions = (
        ("x0", "moderate", 64_800),
        ("x0", "high", 46_913),
        ("and_x0_x1", "moderate", 57_856),
        ("and_x0_x1", "high", 24_848),
        ("or_x0_x1", "moderate", 63_480),
        ("or_x0_x1", "high", 64_438),
    )
    pairs: list[RulePair] = []
    for simple_name, variant, complex_id in definitions:
        simple_id = simple[simple_name]
        simple_bits = function_bits(simple_id)
        complex_bits = function_bits(complex_id)
        differing = tuple(
            int(value) for value in np.flatnonzero(simple_bits != complex_bits)
        )
        agreement = tuple(
            int(value) for value in np.flatnonzero(simple_bits == complex_bits)
        )
        expected_differences = 2 if variant == "moderate" else 4
        if len(differing) != expected_differences:
            raise RuntimeError(
                f"{simple_name}/{variant} 差异位数错误：{len(differing)}"
            )
        if int(simple_bits.sum()) != int(complex_bits.sum()):
            raise RuntimeError(f"{simple_name}/{variant} Hamming weight 不匹配。")
        pairs.append(
            RulePair(
                name=f"{simple_name}_{variant}",
                simple_name=simple_name,
                variant=variant,
                simple_function_id=simple_id,
                complex_function_id=complex_id,
                differing_indices=differing,
                agreement_indices=agreement,
            )
        )
    return pairs


def validate_complexity_order(
    pair: RulePair,
    panel: dict[str, np.ndarray],
) -> dict[str, Any]:
    fields = (
        "multi_proxy_complexity_rank",
        "anf_terms",
        "min_normal_form_literals",
        "decision_tree_leaves",
    )
    report: dict[str, Any] = {}
    for field_name in fields:
        simple_value = float(panel[field_name][pair.simple_function_id])
        complex_value = float(panel[field_name][pair.complex_function_id])
        if not complex_value > simple_value:
            raise RuntimeError(
                f"{pair.name} 在 {field_name} 上不是严格复杂："
                f"simple={simple_value}, complex={complex_value}"
            )
        report[f"simple_{field_name}"] = simple_value
        report[f"complex_{field_name}"] = complex_value
    report["simple_output_hamming_weight"] = int(
        panel["output_hamming_weight"][pair.simple_function_id]
    )
    report["complex_output_hamming_weight"] = int(
        panel["output_hamming_weight"][pair.complex_function_id]
    )
    simple_threshold = is_linear_threshold_function(pair.simple_function_id)
    complex_threshold = is_linear_threshold_function(pair.complex_function_id)
    if not simple_threshold or complex_threshold:
        raise RuntimeError(
            f"{pair.name} 的线性阈值可分性不满足预注册要求："
            f"simple={simple_threshold}, complex={complex_threshold}"
        )
    report["simple_is_linear_threshold"] = simple_threshold
    report["complex_is_linear_threshold"] = complex_threshold
    return report


def is_linear_threshold_function(function_id: int) -> bool:
    """用带单位 margin 的线性规划精确检查 4-bit truth table 可分性。"""

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise RuntimeError(
            "pairwise 实验需要 scipy.optimize.linprog 验证复杂规则不是单层阈值函数。"
        ) from exc
    values = np.arange(16, dtype=np.uint8)
    inputs = ((values[:, None] >> np.arange(3, -1, -1)) & 1).astype(
        np.float64
    )
    augmented = np.concatenate(
        [inputs, np.ones((len(inputs), 1), dtype=np.float64)], axis=1
    )
    labels = np.asarray(
        [1.0 if (function_id >> index) & 1 else -1.0 for index in range(16)],
        dtype=np.float64,
    )
    result = linprog(
        c=np.zeros(5, dtype=np.float64),
        A_ub=-(labels[:, None] * augmented),
        b_ub=-np.ones(16, dtype=np.float64),
        bounds=[(None, None)] * 5,
        method="highs",
    )
    return bool(result.success)


def nested_indices(
    pair: RulePair,
    sizes: tuple[int, ...],
    seed: int,
) -> dict[int, tuple[int, ...]]:
    agreement = np.asarray(pair.agreement_indices, dtype=np.int64)
    simple_bits = function_bits(pair.simple_function_id)
    generator = np.random.default_rng(seed)
    for _ in range(10_000):
        order = agreement.copy()
        generator.shuffle(order)
        valid = True
        result: dict[int, tuple[int, ...]] = {}
        for size in sizes:
            selected = np.sort(order[:size])
            labels = simple_bits[selected]
            if len(np.unique(labels)) < 2:
                valid = False
                break
            result[size] = tuple(int(value) for value in selected)
        if valid:
            return result
    raise RuntimeError(f"无法为 {pair.name} 构造嵌套且含两类标签的训练集。")


def build_conditions(pairs: list[RulePair], seed: int) -> list[PairCondition]:
    conditions: list[PairCondition] = []
    for pair_index, pair in enumerate(pairs):
        sizes = (10, 12, 14) if pair.variant == "moderate" else (10, 12)
        nested = nested_indices(pair, sizes, seed + 10_000 * pair_index)
        simple_bits = function_bits(pair.simple_function_id)
        complex_bits = function_bits(pair.complex_function_id)
        for size in sizes:
            indices = nested[size]
            if not np.array_equal(
                simple_bits[list(indices)], complex_bits[list(indices)]
            ):
                raise RuntimeError(f"{pair.name}/k={size} 训练标签不一致。")
            conditions.append(
                PairCondition(
                    name=f"{pair.name}_k{size:02d}",
                    pair_name=pair.name,
                    simple_name=pair.simple_name,
                    variant=pair.variant,
                    constraint_size=size,
                    input_indices=indices,
                    targets=tuple(
                        int(value) for value in simple_bits[list(indices)]
                    ),
                    simple_function_id=pair.simple_function_id,
                    complex_function_id=pair.complex_function_id,
                )
            )
    return conditions


def load_complexity_panel(source_dir: Path) -> dict[str, np.ndarray]:
    path = source_dir / "function_complexity_panel.npz"
    if not path.exists():
        raise FileNotFoundError(f"找不到复杂度面板：{path}")
    with np.load(path) as payload:
        return {key: payload[key].copy() for key in payload.files}


def shard_paths_for_architecture(source_dir: Path, architecture: str) -> list[Path]:
    shard_dir = source_dir / architecture / "prior_shards"
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"找不到 {architecture} prior shards：{shard_dir}\n"
            "最终下载 zip 不包含这些大文件；本分析必须在原 AutoDL 结果目录运行。"
        )
    return paths


def model_count_for_architecture(source_dir: Path, architecture: str) -> int:
    metadata_path = source_dir / architecture / "prior_shards" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata["architecture"]["model_count"])


def load_condition_cohort(
    shard_paths: list[Path], condition: PairCondition
) -> dict[str, np.ndarray]:
    ids_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    normalized_parts: list[np.ndarray] = []
    rms_parts: list[np.ndarray] = []
    indices = np.asarray(condition.input_indices, dtype=np.int64)
    targets = np.asarray(condition.targets, dtype=np.float32)
    signed = targets * 2.0 - 1.0

    for path in shard_paths:
        with np.load(path) as payload:
            logits = payload["logits"].astype(np.float32)
            function_ids = payload["function_ids"].astype(np.uint16)
        margins = logits[:, indices] * signed[None, :]
        hard = np.all(margins > 0.0, axis=1)
        if not np.any(hard):
            continue
        raw_loss = np.logaddexp(0.0, -margins).mean(axis=1)
        rms = np.sqrt(np.mean(np.square(logits, dtype=np.float64), axis=1)).astype(
            np.float32
        )
        normalized_margins = margins / np.maximum(rms[:, None], 1e-12)
        normalized_loss = np.logaddexp(0.0, -normalized_margins).mean(axis=1)
        ids_parts.append(function_ids[hard])
        raw_parts.append(raw_loss[hard].astype(np.float32))
        normalized_parts.append(normalized_loss[hard].astype(np.float32))
        rms_parts.append(rms[hard])

    if not ids_parts:
        return {
            "function_ids": np.empty(0, dtype=np.uint16),
            "raw_loss": np.empty(0, dtype=np.float32),
            "normalized_loss": np.empty(0, dtype=np.float32),
            "logit_rms": np.empty(0, dtype=np.float32),
        }
    return {
        "function_ids": np.concatenate(ids_parts),
        "raw_loss": np.concatenate(raw_parts),
        "normalized_loss": np.concatenate(normalized_parts),
        "logit_rms": np.concatenate(rms_parts),
    }


def log2_odds(simple_count: int, complex_count: int) -> float:
    return float(math.log2((simple_count + 0.5) / (complex_count + 0.5)))


def analyze_condition(
    cfg: Config,
    architecture: str,
    condition: PairCondition,
    shard_paths: list[Path],
    model_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = load_condition_cohort(shard_paths, condition)
    function_ids = cohort["function_ids"]
    if len(function_ids) == 0:
        return [], {
            **asdict(condition),
            "architecture": architecture,
            "status": "no_hard_exact_samples",
            "sampled_prior_models": model_count,
            "hard_exact_count": 0,
        }

    raw_loss = cohort["raw_loss"]
    normalized_loss = cohort["normalized_loss"]
    logit_rms = cohort["logit_rms"]
    scale_low, scale_high = np.quantile(logit_rms, cfg.FIXED_SCALE_QUANTILES)
    fixed_mask = (logit_rms >= scale_low) & (logit_rms <= scale_high)
    source_families = (
        ("raw_loss_hard", np.ones(len(function_ids), dtype=bool), raw_loss),
        (
            "normalized_loss_hard",
            np.ones(len(function_ids), dtype=bool),
            normalized_loss,
        ),
        ("raw_loss_fixed_scale", fixed_mask, raw_loss),
    )
    rows: list[dict[str, Any]] = []

    for loss_family, source_mask, score in source_families:
        source_indices = np.flatnonzero(source_mask)
        if len(source_indices) == 0:
            continue
        order = np.argsort(score[source_indices], kind="quicksort")
        sorted_global = source_indices[order]
        sorted_scores = score[sorted_global]
        sorted_ids = function_ids[sorted_global]
        baseline_simple = int(
            np.count_nonzero(
                sorted_ids == condition.simple_function_id
            )
        )
        baseline_complex = int(
            np.count_nonzero(
                sorted_ids == condition.complex_function_id
            )
        )
        baseline_log_odds = log2_odds(baseline_simple, baseline_complex)

        for fraction in cfg.QUANTILE_FRACTIONS:
            selected_count = min(
                len(sorted_ids),
                max(1, int(math.ceil(fraction * len(sorted_ids)))),
            )
            if fraction < 1.0 and selected_count < cfg.MIN_SELECTED_MODELS:
                continue
            selected_ids = sorted_ids[:selected_count]
            simple_count = int(
                np.count_nonzero(
                    selected_ids == condition.simple_function_id
                )
            )
            complex_count = int(
                np.count_nonzero(
                    selected_ids == condition.complex_function_id
                )
            )
            current_log_odds = log2_odds(simple_count, complex_count)
            pair_count = simple_count + complex_count
            expected_simple_null = baseline_simple * float(fraction)
            expected_complex_null = baseline_complex * float(fraction)
            fisher_odds_ratio, fisher_p_value = fisher_selection_test(
                simple_count=simple_count,
                complex_count=complex_count,
                baseline_simple=baseline_simple,
                baseline_complex=baseline_complex,
            )
            rows.append(
                {
                    "architecture": architecture,
                    "condition": condition.name,
                    "pair_name": condition.pair_name,
                    "simple_name": condition.simple_name,
                    "variant": condition.variant,
                    "constraint_size": condition.constraint_size,
                    "loss_family": loss_family,
                    "retained_fraction": float(fraction),
                    "selected_model_count": selected_count,
                    "loss_threshold": float(sorted_scores[selected_count - 1]),
                    "hard_source_count": len(sorted_ids),
                    "simple_function_id": condition.simple_function_id,
                    "complex_function_id": condition.complex_function_id,
                    "simple_count": simple_count,
                    "complex_count": complex_count,
                    "pair_count": pair_count,
                    "simple_probability": simple_count / selected_count,
                    "complex_probability": complex_count / selected_count,
                    "simple_share_within_pair": (
                        (simple_count + 0.5) / (pair_count + 1.0)
                    ),
                    "log2_simple_to_complex_odds": current_log_odds,
                    "baseline_simple_count": baseline_simple,
                    "baseline_complex_count": baseline_complex,
                    "expected_simple_count_under_unchanged_odds": (
                        expected_simple_null
                    ),
                    "expected_complex_count_under_unchanged_odds": (
                        expected_complex_null
                    ),
                    "baseline_log2_simple_to_complex_odds": baseline_log_odds,
                    "log2_odds_enrichment_vs_baseline": (
                        current_log_odds - baseline_log_odds
                    ),
                    "odds_enrichment_vs_baseline": float(
                        2.0 ** (current_log_odds - baseline_log_odds)
                    ),
                    "baseline_rules_sufficiently_observed": bool(
                        baseline_simple >= cfg.MIN_BASELINE_COUNT_PER_RULE
                        and baseline_complex >= cfg.MIN_BASELINE_COUNT_PER_RULE
                    ),
                    "null_expected_counts_reliable": bool(
                        expected_simple_null
                        >= cfg.MIN_EXPECTED_TAIL_COUNT_PER_RULE
                        and expected_complex_null
                        >= cfg.MIN_EXPECTED_TAIL_COUNT_PER_RULE
                    ),
                    "fisher_selection_odds_ratio": fisher_odds_ratio,
                    "fisher_selection_two_sided_p": fisher_p_value,
                }
            )

    metadata = {
        **asdict(condition),
        "architecture": architecture,
        "status": "ok",
        "sampled_prior_models": model_count,
        "hard_exact_count": len(function_ids),
        "hard_exact_fraction": len(function_ids) / model_count,
        "fixed_scale_count": int(fixed_mask.sum()),
    }
    return rows, metadata


def fisher_selection_test(
    *,
    simple_count: int,
    complex_count: int,
    baseline_simple: int,
    baseline_complex: int,
) -> tuple[float, float]:
    try:
        from scipy.stats import fisher_exact
    except ImportError as exc:
        raise RuntimeError("pairwise 实验需要 scipy.stats.fisher_exact。") from exc
    table = np.asarray(
        [
            [simple_count, complex_count],
            [baseline_simple - simple_count, baseline_complex - complex_count],
        ],
        dtype=np.int64,
    )
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return float(odds_ratio), float(p_value)


def terminal_reliable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["architecture"]),
            str(row["loss_family"]),
            str(row["condition"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for group_rows in groups.values():
        reliable = [
            row
            for row in group_rows
            if float(row["retained_fraction"]) < 1.0
            and bool(row["baseline_rules_sufficiently_observed"])
            and bool(row["null_expected_counts_reliable"])
        ]
        if not reliable:
            continue
        reliable.sort(key=lambda item: float(item["retained_fraction"]))
        output.append(reliable[0])
    return output


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal = terminal_reliable_rows(rows)
    keys = sorted(
        {
            (str(row["architecture"]), str(row["loss_family"]))
            for row in terminal
        }
    )
    output: list[dict[str, Any]] = []
    for architecture, loss_family in keys:
        subset = [
            row
            for row in terminal
            if row["architecture"] == architecture
            and row["loss_family"] == loss_family
        ]
        enrichment = np.asarray(
            [float(row["log2_odds_enrichment_vs_baseline"]) for row in subset],
            dtype=np.float64,
        )
        p_values = np.asarray(
            [float(row["fisher_selection_two_sided_p"]) for row in subset],
            dtype=np.float64,
        )
        significant = p_values < 0.05
        output.append(
            {
                "architecture": architecture,
                "loss_family": loss_family,
                "reliable_pair_condition_count": len(subset),
                "simple_odds_increased_count": int(np.count_nonzero(enrichment > 0)),
                "simple_odds_increased_fraction": float(np.mean(enrichment > 0)),
                "significant_test_count_p_lt_0_05": int(
                    np.count_nonzero(significant)
                ),
                "significant_simple_favored_count": int(
                    np.count_nonzero(significant & (enrichment > 0))
                ),
                "significant_complex_favored_count": int(
                    np.count_nonzero(significant & (enrichment < 0))
                ),
                "mean_log2_odds_enrichment": float(np.mean(enrichment)),
                "median_log2_odds_enrichment": float(np.median(enrichment)),
                "minimum_log2_odds_enrichment": float(np.min(enrichment)),
                "maximum_log2_odds_enrichment": float(np.max(enrichment)),
            }
        )
    return output


def plot_results(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", flush=True)
        return
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    architectures = sorted({str(row["architecture"]) for row in rows})
    loss_families = sorted({str(row["loss_family"]) for row in rows})
    for architecture in architectures:
        for loss_family in loss_families:
            subset = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["loss_family"] == loss_family
            ]
            if not subset:
                continue
            figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
            for axis, simple_name in zip(
                axes, ("x0", "and_x0_x1", "or_x0_x1")
            ):
                simple_rows = [
                    row for row in subset if row["simple_name"] == simple_name
                ]
                for condition in sorted(
                    {str(row["condition"]) for row in simple_rows}
                ):
                    curve = [
                        row for row in simple_rows if row["condition"] == condition
                    ]
                    curve.sort(
                        key=lambda item: float(item["retained_fraction"]),
                        reverse=True,
                    )
                    axis.plot(
                        [float(row["retained_fraction"]) for row in curve],
                        [
                            float(row["log2_odds_enrichment_vs_baseline"])
                            for row in curve
                        ],
                        marker="o",
                        linewidth=1.6,
                        label=condition.replace(f"{simple_name}_", ""),
                    )
                axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
                axis.set_xscale("log")
                axis.invert_xaxis()
                axis.set_title(simple_name)
                axis.set_xlabel("retained lowest-loss fraction")
                axis.set_ylabel("log2 simple/complex odds enrichment")
                axis.grid(alpha=0.25)
                handles, labels = axis.get_legend_handles_labels()
                if labels:
                    axis.legend(handles, labels, fontsize=7)
            figure.suptitle(f"{architecture} / {loss_family}", fontsize=14)
            figure.savefig(
                plot_dir / f"pair_odds_{architecture}_{loss_family}.png", dpi=170
            )
            plt.close(figure)


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}.zip"
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
    cfg = Config()
    source_dir = cfg.SOURCE_RESULT_DIR
    result_dir = cfg.RESULT_DIR
    # 本地 smoke 只有在显式把 SOURCE_RESULT_DIR 改为本地目录时启用。
    if not source_dir.exists() and cfg._LOCAL_SOURCE_DIR.exists():
        source_dir = cfg._LOCAL_SOURCE_DIR
        result_dir = cfg._LOCAL_RESULT_DIR
    source_dir = source_dir.resolve()
    result_dir = result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    panel = load_complexity_panel(source_dir)
    pairs = build_rule_pairs()
    pair_complexity = {
        pair.name: validate_complexity_order(pair, panel) for pair in pairs
    }
    conditions = build_conditions(pairs, cfg.CONDITION_SEED)

    write_json(result_dir / "config.json", asdict(cfg))
    write_json(result_dir / "rule_pairs.json", [asdict(pair) for pair in pairs])
    write_json(result_dir / "pair_complexity.json", pair_complexity)
    write_json(result_dir / "conditions.json", [asdict(item) for item in conditions])
    write_json(
        result_dir / "runtime.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "source_result_dir": str(source_dir),
        },
    )

    print("=== Loss-conditioned Simple vs Complex Rule Competition ===", flush=True)
    print(f"源 prior：{source_dir}", flush=True)
    print(f"结果目录：{result_dir}", flush=True)
    print(f"pairs={len(pairs)} | conditions={len(conditions)}", flush=True)

    all_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for architecture in cfg.ARCHITECTURES:
        shard_paths = shard_paths_for_architecture(source_dir, architecture)
        model_count = model_count_for_architecture(source_dir, architecture)
        print(
            f"\n[{architecture}] 复用 {model_count:,} prior models，"
            f"shards={len(shard_paths)}",
            flush=True,
        )
        for index, condition in enumerate(conditions, start=1):
            print(
                f"  condition {index}/{len(conditions)}: {condition.name}",
                flush=True,
            )
            rows, metadata = analyze_condition(
                cfg,
                architecture,
                condition,
                shard_paths,
                model_count,
            )
            all_rows.extend(rows)
            metadata_rows.append(metadata)
            write_csv(result_dir / "pair_odds_trajectories.csv", all_rows)

    aggregate = aggregate_results(all_rows)
    terminal = terminal_reliable_rows(all_rows)
    write_csv(result_dir / "pair_odds_trajectories.csv", all_rows)
    write_csv(result_dir / "terminal_reliable_pair_tests.csv", terminal)
    write_csv(result_dir / "aggregate_pair_verdict.csv", aggregate)
    write_csv(result_dir / "condition_summary.csv", metadata_rows)
    if cfg.GENERATE_PLOTS:
        plot_results(result_dir, all_rows)

    summary = {
        "status": "ok",
        "source_result_dir": str(source_dir),
        "pair_count": len(pairs),
        "condition_count": len(conditions),
        "primary_loss_family": "raw_loss_hard",
        "secondary_diagnostic_loss_families": [
            "normalized_loss_hard",
            "raw_loss_fixed_scale",
        ],
        "aggregate_verdict": aggregate,
        "reliable_terminal_test_count": len(terminal),
        "elapsed_seconds": time.perf_counter() - start_time,
        "interpretation": (
            "主命题只由 raw_loss_hard 判决：正 log2 odds enrichment 表示，"
            "在同一训练约束和同一初始化先验中，进入更低真实 BCE loss "
            "子水平集后，严格更简单规则相对复杂规则的概率增加。normalized "
            "与 fixed-scale 只用于机制诊断，不参与主命题真伪判决。"
        ),
    }
    write_json(result_dir / "summary.json", summary)
    archive_path: Path | None = None
    if cfg.CREATE_ARCHIVE:
        archive_path = create_archive(result_dir)

    print("\n=== 分析完成 ===", flush=True)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"汇总：{result_dir / 'summary.json'}", flush=True)
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}", flush=True)


if __name__ == "__main__":
    main()
