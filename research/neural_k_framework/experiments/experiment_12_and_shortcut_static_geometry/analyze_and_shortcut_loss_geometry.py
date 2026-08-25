"""
AND 平衡 n=10 数据集的定向静态几何调查。

复用 experiment_4bit_simple_rule_loss_sweep.py 已生成的 prior shards，不重新
采样网络。所有条件固定：

- 目标函数 x1 AND x2；
- 与原始缺口训练集相比只把 1101 替换为 1000；
- 四种 (x1, x2) 模式按 3/2/2/3 覆盖；
- 总样本数 10，标签比例 3:7。

检查 AND、D000、F040、F050 等候选函数随静态 raw-BCE 深度的概率变化，
并与真实 SGD 的稳定函数分布比较。
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


class Config:
    SOURCE_DIR = Path("/root/results_4bit_simple_rule_loss_sweep")
    RESULT_DIR = Path("/root/results_and_shortcut_static_geometry_publication")
    ACTIVE_CONDITIONS = (
        "original_gap",
        "close_x1_branch",
        "close_x2_branch",
        "close_both_branches",
        "mirror_gap_x4_1",
        "balanced_minimal_n10",
    )
    DATA_SEED = 20260829
    RANDOM_SUBSET_COUNT = 6
    MIN_RELIABLE_COUNT = 50
    TAIL_FRACTIONS = (
        1.0, 0.9, 0.75, 0.5, 0.25,
        0.1, 0.05, 0.02, 0.01, 0.005,
        0.002, 0.001, 0.0005,
    )
    TRACKED_FUNCTIONS = {
        "AND_F000": 0xF000,
        "D000": 0xD000,
        "F040": 0xF040,
        "F050": 0xF050,
        "F010": 0xF010,
        "D040": 0xD040,
        "D050": 0xD050,
        "F550": 0xF550,
        "F500": 0xF500,
        "F700": 0xF700,
        "F110": 0xF110,
    }
    TOP_FUNCTIONS = 20
    PACKAGE_RESULTS = True
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass(frozen=True)
class DatasetCondition:
    name: str
    negative_indices: tuple[int, ...]
    positive_indices: tuple[int, ...] = (12, 13, 14, 15)

    @property
    def train_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.negative_indices + self.positive_indices))


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.SOURCE_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_4bit_simple_rule_loss_sweep"
    )
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_and_shortcut_loss_geometry"
    )
    Config.RANDOM_SUBSET_COUNT = 2
    Config.MIN_RELIABLE_COUNT = 4
    Config.TAIL_FRACTIONS = (1.0, 0.5, 0.2, 0.1, 0.05)
    Config.TOP_FUNCTIONS = 5
    Config.ACTIVE_CONDITIONS = ("balanced_minimal_n10",)
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
    output = Config.RESULT_DIR
    if output.exists():
        if Config.OVERWRITE_RESULT_DIR:
            shutil.rmtree(output)
        else:
            output = output.parent / (
                output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def function_bits(function_id: int) -> str:
    return "".join(str((function_id >> index) & 1) for index in range(16))


def input_bits(index: int) -> tuple[int, int, int, int]:
    return tuple((index >> shift) & 1 for shift in (3, 2, 1, 0))


def and_label(index: int) -> int:
    x1, x2, _, _ = input_bits(index)
    return x1 & x2


def build_conditions() -> list[DatasetCondition]:
    # 原实验的具体缺口。
    conditions = [
        # 与 original_gap 仅交换一个样本：删1101，加入1000。
        DatasetCondition(
            "balanced_minimal_n10",
            (1, 2, 3, 5, 7, 8, 11),
            (12, 14, 15),
        ),
        DatasetCondition("original_gap", (1, 2, 3, 5, 7, 11)),
        # 加入 1000/1010，直接封闭 F500 的 x1=1,x2=0,x4=0 分支。
        DatasetCondition("close_x1_branch", (1, 2, 3, 8, 10, 11)),
        # 加入 0100/0110，封闭 F550 的 x1=0,x2=1,x4=0 分支。
        DatasetCondition("close_x2_branch", (1, 2, 3, 4, 6, 11)),
        # 同时覆盖四个 x4=0 的 off-diagonal 状态。
        DatasetCondition("close_both_branches", (1, 4, 6, 8, 10, 11)),
        # 镜像缺口：覆盖 x4=0 off-diagonal，遗漏 x4=1 off-diagonal。
        DatasetCondition("mirror_gap_x4_1", (0, 2, 4, 6, 8, 10)),
    ]
    rng = np.random.default_rng(Config.DATA_SEED)
    negative_pool = np.arange(12, dtype=np.int64)
    used = {condition.negative_indices for condition in conditions}
    while sum(condition.name.startswith("random_") for condition in conditions) < Config.RANDOM_SUBSET_COUNT:
        chosen = tuple(sorted(int(value) for value in rng.choice(
            negative_pool, size=6, replace=False
        )))
        if chosen in used:
            continue
        used.add(chosen)
        conditions.append(DatasetCondition(
            f"random_{sum(c.name.startswith('random_') for c in conditions)}",
            chosen,
        ))
    for condition in conditions:
        if len(condition.train_indices) != 10:
            raise RuntimeError(f"{condition.name} 不是10个训练样本。")
        if len(set(condition.train_indices)) != len(condition.train_indices):
            raise RuntimeError(f"{condition.name} 含重复训练样本。")
        if any(and_label(index) != 0 for index in condition.negative_indices):
            raise RuntimeError(f"{condition.name} 混入正例。")
        if any(and_label(index) != 1 for index in condition.positive_indices):
            raise RuntimeError(f"{condition.name} 的正例索引错误。")
    by_name = {condition.name: condition for condition in conditions}
    unknown = sorted(set(Config.ACTIVE_CONDITIONS) - set(by_name))
    if unknown:
        raise ValueError(f"未知 ACTIVE_CONDITIONS：{unknown}")
    return [by_name[name] for name in Config.ACTIVE_CONDITIONS]


def load_source_config() -> dict[str, Any]:
    path = Config.SOURCE_DIR / "config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}。需要保留原实验 prior_shards。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_shards() -> list[Path]:
    paths = sorted((Config.SOURCE_DIR / "prior_shards").glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"{Config.SOURCE_DIR / 'prior_shards'} 中没有 prior shards。"
        )
    return paths


def build_cache(
    output_dir: Path,
    shard_paths: Sequence[Path],
    conditions: Sequence[DatasetCondition],
    model_count: int,
) -> tuple[dict[str, Path], Path]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    loss_paths = {
        condition.name: cache_dir / f"{condition.name}.float32.mmap"
        for condition in conditions
    }
    loss_maps = {
        name: np.memmap(path, mode="w+", dtype=np.float32, shape=(model_count,))
        for name, path in loss_paths.items()
    }
    id_path = cache_dir / "function_ids.uint16.mmap"
    id_map = np.memmap(id_path, mode="w+", dtype=np.uint16, shape=(model_count,))

    tracked_logits: dict[str, list[np.ndarray]] = {
        name: [] for name in Config.TRACKED_FUNCTIONS
    }
    cursor = 0
    for shard_number, path in enumerate(shard_paths, start=1):
        with np.load(path) as payload:
            logits = payload["logits"].astype(np.float32)
            function_ids = payload["function_ids"].astype(np.uint16)
        end = cursor + len(logits)
        id_map[cursor:end] = function_ids
        softplus = np.logaddexp(0.0, logits)
        for condition in conditions:
            indices = np.asarray(condition.train_indices, dtype=np.int64)
            labels = np.asarray([and_label(index) for index in indices], dtype=np.float32)
            loss_maps[condition.name][cursor:end] = (
                softplus[:, indices].sum(axis=1)
                - (logits[:, indices] * labels[None]).sum(axis=1)
            ) / len(indices)
        for name, function_id in Config.TRACKED_FUNCTIONS.items():
            mask = function_ids == function_id
            if np.any(mask):
                tracked_logits[name].append(logits[mask].astype(np.float16))
        cursor = end
        print(f"cache shard {shard_number}/{len(shard_paths)}", flush=True)
    if cursor != model_count:
        raise RuntimeError(f"shard 模型数 {cursor} != config {model_count}")
    for values in loss_maps.values():
        values.flush()
    id_map.flush()
    np.savez_compressed(
        output_dir / "tracked_function_logits.npz",
        **{
            name: (
                np.concatenate(parts)
                if parts else np.empty((0, 16), dtype=np.float16)
            )
            for name, parts in tracked_logits.items()
        },
    )
    del loss_maps, id_map
    return loss_paths, id_path


def hard_exact_mask(
    function_ids: np.ndarray,
    condition: DatasetCondition,
) -> np.ndarray:
    mask = np.ones(len(function_ids), dtype=bool)
    for index in condition.train_indices:
        prediction = (function_ids >> np.uint16(index)) & np.uint16(1)
        mask &= prediction == np.uint16(and_label(index))
    return mask


def analyze_condition(
    condition: DatasetCondition,
    losses: np.ndarray,
    function_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exact = hard_exact_mask(function_ids, condition)
    indices = np.flatnonzero(exact)
    source_loss = np.asarray(losses[indices], dtype=np.float32)
    source_ids = function_ids[indices].astype(np.int64)
    order = np.argsort(source_loss, kind="stable")
    sorted_loss = source_loss[order]
    sorted_ids = source_ids[order]
    source_count = len(sorted_ids)
    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    cumulative = np.zeros(65_536, dtype=np.int64)
    exception_counts = np.fromiter(
        ((function_id ^ 0xF000).bit_count() for function_id in range(65_536)),
        dtype=np.uint8,
        count=65_536,
    )
    previous = 0
    for fraction in sorted(set(Config.TAIL_FRACTIONS)):
        selected_count = min(
            source_count, max(1, int(math.ceil(source_count * fraction)))
        )
        cumulative += np.bincount(
            sorted_ids[previous:selected_count], minlength=65_536
        )
        previous = selected_count
        if selected_count < Config.MIN_RELIABLE_COUNT:
            continue
        row: dict[str, Any] = {
            "condition": condition.name,
            "retained_fraction": fraction,
            "source_count": source_count,
            "selected_count": selected_count,
            "loss_min": float(sorted_loss[0]),
            "loss_max": float(sorted_loss[selected_count - 1]),
        }
        nonzero_counts = cumulative[cumulative > 0]
        probabilities = nonzero_counts / selected_count
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        row["function_entropy_bits"] = entropy
        row["effective_function_count"] = float(2.0 ** entropy)
        row["mean_exception_count_vs_and"] = float(
            np.dot(cumulative, exception_counts.astype(np.int64))
            / selected_count
        )
        for name, function_id in Config.TRACKED_FUNCTIONS.items():
            probability = int(cumulative[function_id]) / selected_count
            row[f"p_{name}"] = probability
            row[f"count_{name}"] = int(cumulative[function_id])
        row["odds_AND_over_F550"] = (
            (cumulative[0xF000] + 0.5) / (cumulative[0xF550] + 0.5)
        )
        row["odds_AND_over_F500"] = (
            (cumulative[0xF000] + 0.5) / (cumulative[0xF500] + 0.5)
        )
        row["odds_AND_over_D000"] = (
            (cumulative[0xF000] + 0.5) / (cumulative[0xD000] + 0.5)
        )
        row["odds_AND_over_F040"] = (
            (cumulative[0xF000] + 0.5) / (cumulative[0xF040] + 0.5)
        )
        row["odds_AND_over_F050"] = (
            (cumulative[0xF000] + 0.5) / (cumulative[0xF050] + 0.5)
        )
        rows.append(row)
        top_ids = np.argsort(cumulative)[::-1][: Config.TOP_FUNCTIONS]
        for rank, function_id in enumerate(top_ids, start=1):
            count = int(cumulative[function_id])
            if count == 0:
                break
            top_rows.append({
                "condition": condition.name,
                "retained_fraction": fraction,
                "rank": rank,
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "truth_table_x0_to_x15": function_bits(int(function_id)),
                "count": count,
                "probability": count / selected_count,
                "tracked_name": next((
                    name for name, value in Config.TRACKED_FUNCTIONS.items()
                    if value == int(function_id)
                ), None),
            })
    baseline_row = next(
        (row for row in rows if row["retained_fraction"] == 1.0), None
    )
    bottom10_row = next(
        (row for row in rows if row["retained_fraction"] == 0.1), None
    )

    def enrichment(field: str) -> float | None:
        if baseline_row is None or bottom10_row is None:
            return None
        baseline = float(baseline_row[field])
        if baseline <= 0:
            return None
        return float(bottom10_row[field]) / baseline

    summary = {
        "condition": condition.name,
        "negative_indices": condition.negative_indices,
        "positive_indices": condition.positive_indices,
        "train_indices": condition.train_indices,
        "missing_negative_indices": tuple(
            sorted(set(range(12)) - set(condition.negative_indices))
        ),
        "missing_positive_indices": tuple(
            sorted(set(range(12, 16)) - set(condition.positive_indices))
        ),
        "hard_exact_count": source_count,
        "and_enrichment_bottom10": enrichment("p_AND_F000"),
        "d000_enrichment_bottom10": enrichment("p_D000"),
        "f040_enrichment_bottom10": enrichment("p_F040"),
        "f050_enrichment_bottom10": enrichment("p_F050"),
        "f550_enrichment_bottom10": enrichment("p_F550"),
        "f500_enrichment_bottom10": enrichment("p_F500"),
    }
    del exact, indices
    return rows, top_rows, summary


def analyze_margins(
    output_dir: Path,
    conditions: Sequence[DatasetCondition],
) -> list[dict[str, Any]]:
    payload = np.load(output_dir / "tracked_function_logits.npz")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        indices = np.asarray(condition.train_indices, dtype=np.int64)
        labels = np.asarray([and_label(index) for index in indices], dtype=np.float32)
        signs = labels * 2.0 - 1.0
        for function_name in Config.TRACKED_FUNCTIONS:
            logits = payload[function_name].astype(np.float32)
            if len(logits) == 0:
                continue
            margins = logits[:, indices] * signs[None]
            for local_index, input_index in enumerate(indices):
                values = margins[:, local_index]
                rows.append({
                    "condition": condition.name,
                    "function": function_name,
                    "input_index": int(input_index),
                    "input_bits": "".join(map(str, input_bits(int(input_index)))),
                    "label": int(labels[local_index]),
                    "model_count": len(values),
                    "margin_mean": float(np.mean(values)),
                    "margin_median": float(np.median(values)),
                    "margin_q10": float(np.quantile(values, 0.1)),
                    "margin_q90": float(np.quantile(values, 0.9)),
                })
            rows.append({
                "condition": condition.name,
                "function": function_name,
                "input_index": -1,
                "input_bits": "ALL",
                "label": -1,
                "model_count": len(logits),
                "margin_mean": float(np.mean(margins)),
                "margin_median": float(np.median(margins)),
                "margin_q10": float(np.quantile(margins, 0.1)),
                "margin_q90": float(np.quantile(margins, 0.9)),
            })
    write_csv(output_dir / "margin_by_sample.csv", rows)
    return rows


def create_archive(output_dir: Path) -> Path:
    archive = Path(str(output_dir) + "_package.zip")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in output_dir.rglob("*"):
            if not path.is_file() or "cache" in path.relative_to(output_dir).parts:
                continue
            handle.write(path, arcname=str(path.relative_to(output_dir)))
    return archive


def main() -> None:
    apply_smoke_overrides()
    source_config = load_source_config()
    model_count = int(source_config["MODEL_COUNT"])
    shard_paths = list_shards()
    output_dir = prepare_result_dir()
    conditions = build_conditions()
    write_json(output_dir / "config.json", {
        "source_dir": Config.SOURCE_DIR,
        "source_model_count": model_count,
        "tail_fractions": Config.TAIL_FRACTIONS,
        "conditions": [condition.__dict__ for condition in conditions],
        "tracked_functions": Config.TRACKED_FUNCTIONS,
    })
    print("=== AND shortcut loss geometry ===", flush=True)
    print(f"source models={model_count:,} | conditions={len(conditions)}", flush=True)
    loss_paths, id_path = build_cache(
        output_dir, shard_paths, conditions, model_count
    )
    function_ids = np.memmap(
        id_path, mode="r", dtype=np.uint16, shape=(model_count,)
    )
    all_rows: list[dict[str, Any]] = []
    all_top: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for condition in conditions:
        print(f"analyze {condition.name}", flush=True)
        losses = np.memmap(
            loss_paths[condition.name],
            mode="r",
            dtype=np.float32,
            shape=(model_count,),
        )
        rows, top_rows, summary = analyze_condition(
            condition, losses, function_ids
        )
        all_rows.extend(rows)
        all_top.extend(top_rows)
        summaries.append(summary)
        del losses
    write_csv(output_dir / "function_probability_curves.csv", all_rows)
    write_csv(output_dir / "top_functions.csv", all_top)
    write_csv(output_dir / "condition_summary.csv", summaries)
    analyze_margins(output_dir, conditions)
    write_json(output_dir / "summary.json", {
        "conditions": summaries,
        "primary_condition": summaries[0],
        "interpretation_boundary": (
            "本分析复用同一批未训练网络，针对平衡 n=10 数据集按 raw BCE "
            "重新排序；它描述静态 prior 的 low-loss 几何，不等同于 SGD 分布。"
        ),
    })
    archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
    print("=== 完成 ===", flush=True)
    if archive:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
