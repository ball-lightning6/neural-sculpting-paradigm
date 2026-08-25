"""Anti-prefix -> Pro-collapse 共识复杂度前沿实验。

从小型随机部分真值表出发，每轮先在当前 committee 中寻找预测分歧最大的
未见输入，再分别加入标签 0/1，从头训练两个严格配对初始化的 sibling 分支。

共享的anti-consensus主干在多个训练样本量位置冻结快照，每个快照随后切换为
pro-consensus补全，直到函数分布连续两轮达到窄后验门槛。实验测量anti前缀长度、
重新凝聚所需额外样本和终点程序复杂度之间的关系。

分支选择只读取训练拟合和函数分布指标，不读取任何符号复杂度。符号审计只在
全部路径完成后，以全新初始化重新训练最终数据集时执行。

脚本可整段复制到 AutoDL notebook 单元运行。设置环境变量NSP_SMOKE_TEST=1
可运行本地CPU冒烟测试。
"""

from __future__ import annotations

import csv
import functools
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
    WIDTH = 32
    HIDDEN_LAYERS = 2
    START_COUNT = 16
    INITIAL_TRAIN_COUNT = 8
    ROUNDS = 16
    POLICIES = ("anti_consensus", "pro_consensus", "random_label")

    # 分支选择本身会产生 winner's curse；本地配对 pilot 表明 64 seeds
    # 才能让 anti/pro 差异较稳定地迁移到全新审计 seed。
    DISCOVERY_SEED_COUNT = 64
    DISCOVERY_MAX_STEPS = 5_000
    DISCOVERY_LEARNING_RATE = 1e-3
    DISCOVERY_WEIGHT_DECAY = 0.0
    MIN_VALID_FIT_RATE = 0.90

    AUDIT_SEED_COUNT = 512
    AUDIT_MAX_STEPS = 10_000
    AUDIT_LEARNING_RATE = 1e-3
    AUDIT_WEIGHT_DECAY = 0.0

    DATASET_SEED = 2026082301
    INITIALIZATION_SEED = 2026082302
    QUERY_SEED = 2026082303
    RANDOM_POLICY_SEED = 2026082304
    AUDIT_INITIALIZATION_SEED = 2026082305

    FULL_EVAL_CHUNK_SIZE = 256
    ALLOW_TF32 = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"

    FRONTIER_CUTOFF_COUNTS = (8, 16, 24, 32, 48, 64)
    PRO_MAX_ADDITIONAL_SAMPLES = 32
    STABLE_MIN_PAIRWISE_AGREEMENT = 0.995
    STABLE_MIN_BALL_MASS_0P5PCT = 0.90
    STABLE_CONSECUTIVE_ROUNDS = 2
    FRONTIER_BDD_RANDOM_ORDERS = 16
    RESULT_DIR = Path("/root/results_consensus_complexity_frontier")


@dataclass(frozen=True)
class DatasetSpec:
    spec_id: int
    path_id: int
    start_id: int
    policy: str
    branch_label: int
    initialization_key: int
    train_indices: tuple[int, ...]
    train_labels: tuple[int, ...]


@dataclass
class PathState:
    path_id: int
    start_id: int
    policy: str
    train_indices: list[int]
    train_labels: list[int]
    selected_labels: list[int]
    queried_indices: list[int]


@dataclass
class FrontierForkState:
    fork_id: int
    start_id: int
    cutoff_train_count: int
    train_indices: list[int]
    train_labels: list[int]
    pro_queried_indices: list[int]
    pro_selected_labels: list[int]
    stable_streak: int = 0
    status: str = "active"


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.INPUT_BITS = 5
    Config.WIDTH = 8
    Config.HIDDEN_LAYERS = 2
    Config.START_COUNT = 2
    Config.INITIAL_TRAIN_COUNT = 4
    Config.ROUNDS = 2
    Config.DISCOVERY_SEED_COUNT = 4
    Config.DISCOVERY_MAX_STEPS = 5
    Config.MIN_VALID_FIT_RATE = 0.0
    Config.AUDIT_SEED_COUNT = 8
    Config.AUDIT_MAX_STEPS = 5
    Config.FULL_EVAL_CHUNK_SIZE = 32
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/consensus_symbolicity/"
        "_smoke_adversarial_disagreement_completion_pilot"
    )
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True
    Config.PACKAGE_RESULTS = False


def apply_frontier_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.INPUT_BITS = 5
    Config.WIDTH = 8
    Config.HIDDEN_LAYERS = 2
    Config.START_COUNT = 2
    Config.INITIAL_TRAIN_COUNT = 4
    Config.FRONTIER_CUTOFF_COUNTS = (4, 5, 6)
    Config.PRO_MAX_ADDITIONAL_SAMPLES = 2
    Config.DISCOVERY_SEED_COUNT = 4
    Config.DISCOVERY_MAX_STEPS = 5
    Config.MIN_VALID_FIT_RATE = 0.0
    Config.AUDIT_SEED_COUNT = 8
    Config.AUDIT_MAX_STEPS = 5
    Config.STABLE_MIN_PAIRWISE_AGREEMENT = 0.0
    Config.STABLE_MIN_BALL_MASS_0P5PCT = 0.0
    Config.STABLE_CONSECUTIVE_ROUNDS = 1
    Config.FRONTIER_BDD_RANDOM_ORDERS = 2
    Config.FULL_EVAL_CHUNK_SIZE = 32
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/consensus_symbolicity/"
        "_smoke_consensus_complexity_frontier"
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
                    if isinstance(value, (dict, list, tuple))
                    else value
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
        for name in dir(Config)
        if name.isupper()
    }


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "config.json"
    current = config_payload()
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous != current:
            raise RuntimeError(
                "结果目录已有不同配置。请修改 RESULT_DIR 或明确启用覆盖。"
            )
        if not Config.RESUME:
            raise RuntimeError("结果目录已存在且 RESUME=False。")
    else:
        write_json(saved, current)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.uint32)
    shifts = np.arange(Config.INPUT_BITS - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float32)


def parameter_count_per_model() -> int:
    dimensions = (
        [Config.INPUT_BITS]
        + [Config.WIDTH] * Config.HIDDEN_LAYERS
        + [1]
    )
    return int(sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:])
    ))


def dataset_signature(indices: Sequence[int], labels: Sequence[int]) -> str:
    payload = json.dumps(
        list(zip(map(int, indices), map(int, labels))),
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def stable_dataset_key(indices: Sequence[int], labels: Sequence[int]) -> int:
    """把数据集内容映射为稳定整数，确保镜像策略共享随机性。"""
    return int(dataset_signature(indices, labels)[:12], 16) % 2_000_000_000


def initial_datasets() -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(Config.DATASET_SEED)
    state_count = 2 ** Config.INPUT_BITS
    result: list[tuple[list[int], list[int]]] = []
    seen: set[str] = set()
    while len(result) < Config.START_COUNT:
        indices = np.sort(rng.choice(
            state_count, Config.INITIAL_TRAIN_COUNT, replace=False
        )).astype(np.int64)
        negative = Config.INITIAL_TRAIN_COUNT // 2
        labels = np.asarray(
            [0] * negative + [1] * (Config.INITIAL_TRAIN_COUNT - negative),
            dtype=np.uint8,
        )
        rng.shuffle(labels)
        signature = dataset_signature(indices, labels)
        if signature in seen:
            continue
        seen.add(signature)
        result.append((indices.tolist(), labels.astype(int).tolist()))
    return result


class BatchedMLPEnsemble(nn.Module):
    """把候选数据集和 seed 两个轴展平为独立模型轴。"""

    def __init__(
        self,
        initialization_keys: Sequence[int],
        seed_count: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        dimensions = (
            [Config.INPUT_BITS]
            + [Config.WIDTH] * Config.HIDDEN_LAYERS
            + [1]
        )
        layer_count = len(dimensions) - 1
        cache: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        weight_blocks: list[list[torch.Tensor]] = [
            [] for _ in range(layer_count)
        ]
        bias_blocks: list[list[torch.Tensor]] = [
            [] for _ in range(layer_count)
        ]

        for key in initialization_keys:
            key = int(key)
            if key not in cache:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(initialization_seed) + 1_000_003 * key
                )
                local_weights: list[torch.Tensor] = []
                local_biases: list[torch.Tensor] = []
                for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
                    bound = 1.0 / math.sqrt(fan_in)
                    local_weights.append(torch.empty(
                        seed_count, fan_out, fan_in
                    ).uniform_(-bound, bound, generator=generator))
                    local_biases.append(torch.empty(
                        seed_count, fan_out
                    ).uniform_(-bound, bound, generator=generator))
                cache[key] = (local_weights, local_biases)
            local_weights, local_biases = cache[key]
            for layer_index in range(layer_count):
                # clone 避免 sibling 参数共享存储；初值仍严格相同。
                weight_blocks[layer_index].append(
                    local_weights[layer_index].clone()
                )
                bias_blocks[layer_index].append(
                    local_biases[layer_index].clone()
                )

        self.weights = nn.ParameterList([
            nn.Parameter(torch.stack(blocks, dim=0).reshape(
                len(initialization_keys) * seed_count,
                dimensions[layer_index + 1],
                dimensions[layer_index],
            ))
            for layer_index, blocks in enumerate(weight_blocks)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.stack(blocks, dim=0).reshape(
                len(initialization_keys) * seed_count,
                dimensions[layer_index + 1],
            ))
            for layer_index, blocks in enumerate(bias_blocks)
        ])

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
    seed_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not specs:
        raise ValueError("训练候选不能为空。")
    train_counts = {len(spec.train_indices) for spec in specs}
    if len(train_counts) != 1:
        raise ValueError("同一批候选必须具有相同训练样本数。")
    all_inputs = truth_table_inputs()
    count = next(iter(train_counts))
    train_x = np.empty(
        (len(specs), count, Config.INPUT_BITS), dtype=np.float32
    )
    train_y = np.empty((len(specs), count), dtype=np.float32)
    for local_index, spec in enumerate(specs):
        train_x[local_index] = all_inputs[np.asarray(spec.train_indices)]
        train_y[local_index] = np.asarray(spec.train_labels, dtype=np.float32)
    return (
        torch.as_tensor(np.repeat(train_x, seed_count, axis=0), device=device),
        torch.as_tensor(np.repeat(train_y, seed_count, axis=0), device=device),
        torch.as_tensor(all_inputs, device=device),
    )


def distinct_collision(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total < 2:
        return float("nan")
    numerator = np.sum(
        counts.astype(np.float64) * (counts.astype(np.float64) - 1)
    )
    return float(numerator / (total * (total - 1)))


def plugin_entropy(counts: np.ndarray) -> float:
    if not int(counts.sum()):
        return float("nan")
    probability = counts.astype(np.float64) / counts.sum()
    positive = probability[probability > 0]
    return float(-(positive * np.log2(positive)).sum())


def pairwise_bit_agreement(predictions: np.ndarray) -> float:
    model_count = len(predictions)
    if model_count < 2 or predictions.shape[1] == 0:
        return float("nan")
    ones = predictions.sum(axis=0).astype(np.float64)
    same = (
        ones * (ones - 1)
        + (model_count - ones) * (model_count - ones - 1)
    )
    return float(np.mean(same / (model_count * (model_count - 1))))


def vote_entropy(predictions: np.ndarray) -> float:
    if not len(predictions) or predictions.shape[1] == 0:
        return float("nan")
    probability = predictions.mean(axis=0).astype(np.float64)
    entropy = np.zeros_like(probability)
    valid = (probability > 0) & (probability < 1)
    p = probability[valid]
    entropy[valid] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return float(entropy.mean())


def pack_truth(predictions: np.ndarray) -> np.ndarray:
    return np.packbits(
        np.asarray(predictions, dtype=np.uint8), axis=-1, bitorder="little"
    )


def fingerprint_hex(bits: np.ndarray) -> str:
    return pack_truth(np.asarray(bits, dtype=np.uint8)).tobytes().hex().upper()


def summarize_prediction_cohort(
    predictions: np.ndarray,
    fit_mask: np.ndarray,
    train_indices: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    state_count = predictions.shape[1]
    cohort = predictions[np.asarray(fit_mask, dtype=bool)]
    cohort_source = "train_hard_exact_models"
    if not len(cohort):
        cohort = predictions
        cohort_source = "all_models_no_fitted_cohort"

    packed = pack_truth(cohort)
    unique, counts = np.unique(packed, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    unique = unique[order]
    counts = counts[order]
    observed_mode = np.unpackbits(
        unique[0][None], axis=-1, bitorder="little"
    )[0, :state_count]

    ones = cohort.sum(axis=0)
    consensus = (2 * ones > len(cohort)).astype(np.uint8)
    ties = 2 * ones == len(cohort)
    consensus[ties] = observed_mode[ties]

    all_states = np.arange(state_count, dtype=np.int64)
    unseen = np.setdiff1d(
        all_states,
        np.asarray(train_indices, dtype=np.int64),
        assume_unique=False,
    )
    unseen_predictions = cohort[:, unseen]
    hamming = np.mean(
        unseen_predictions != consensus[unseen][None], axis=1
    )
    exact_consensus_mass = float(np.mean(hamming == 0))
    row = {
        "cohort_source": cohort_source,
        "cohort_model_count": int(len(cohort)),
        "unique_function_count": int(len(unique)),
        "exact_modal_probability": float(counts[0] / len(cohort)),
        "exact_function_collision": distinct_collision(counts),
        "function_entropy_plugin_bits": plugin_entropy(counts),
        "unseen_pairwise_agreement": pairwise_bit_agreement(
            unseen_predictions
        ),
        "unseen_vote_entropy_bits": vote_entropy(unseen_predictions),
        "unseen_mean_hamming_to_consensus": float(hamming.mean()),
        "unseen_max_hamming_to_consensus": float(hamming.max()),
        "unseen_exact_consensus_mass": exact_consensus_mass,
        "unseen_ball_mass_0p5pct": float(np.mean(hamming <= 0.005)),
        "unseen_ball_mass_1pct": float(np.mean(hamming <= 0.01)),
        "unseen_unanimous_fraction": float(np.mean(
            np.all(unseen_predictions == unseen_predictions[:1], axis=0)
        )),
        "consensus_fingerprint": fingerprint_hex(consensus),
        "observed_mode_fingerprint": unique[0].tobytes().hex().upper(),
    }
    return row, consensus, cohort


def train_and_evaluate(
    phase: str,
    specs: Sequence[DatasetSpec],
    seed_count: int,
    initialization_seed: int,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    model = BatchedMLPEnsemble(
        [spec.initialization_key for spec in specs],
        seed_count,
        initialization_seed,
    ).to(device)
    train_x, train_y, full_inputs = build_training_tensors(
        specs, seed_count, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    start = time.perf_counter()
    for _ in range(max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        per_model = F.binary_cross_entropy_with_logits(
            logits, train_y, reduction="none"
        ).mean(axis=1)
        per_model.sum().backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        train_losses = F.binary_cross_entropy_with_logits(
            train_logits, train_y, reduction="none"
        ).mean(axis=1)
        train_exact = torch.all(
            (train_logits >= 0) == (train_y >= 0.5), axis=1
        )
        prediction_chunks: list[np.ndarray] = []
        model_count = len(specs) * seed_count
        for state_start in range(
            0, len(full_inputs), Config.FULL_EVAL_CHUNK_SIZE
        ):
            state_stop = min(
                state_start + Config.FULL_EVAL_CHUNK_SIZE,
                len(full_inputs),
            )
            chunk = full_inputs[state_start:state_stop]
            expanded = chunk[None].expand(model_count, -1, -1)
            prediction_chunks.append(
                (model(expanded) >= 0).to(torch.uint8).cpu().numpy()
            )
        predictions = np.concatenate(prediction_chunks, axis=1)
        losses_cpu = train_losses.cpu().numpy()
        exact_cpu = train_exact.cpu().numpy().astype(bool)

    predictions = predictions.reshape(
        len(specs), seed_count, 2 ** Config.INPUT_BITS
    )
    exact_cpu = exact_cpu.reshape(len(specs), seed_count)
    losses_cpu = losses_cpu.reshape(len(specs), seed_count)
    rows: list[dict[str, Any]] = []
    consensuses = np.empty(
        (len(specs), 2 ** Config.INPUT_BITS), dtype=np.uint8
    )
    for index, spec in enumerate(specs):
        metrics, consensus, _ = summarize_prediction_cohort(
            predictions[index], exact_cpu[index], spec.train_indices
        )
        consensuses[index] = consensus
        rows.append({
            "phase": phase,
            "spec_id": spec.spec_id,
            "path_id": spec.path_id,
            "start_id": spec.start_id,
            "policy": spec.policy,
            "branch_label": spec.branch_label,
            "train_count": len(spec.train_indices),
            "train_fit_rate": float(exact_cpu[index].mean()),
            "train_loss_mean": float(losses_cpu[index].mean()),
            "train_loss_median": float(np.median(losses_cpu[index])),
            **metrics,
        })
    print(
        f"[{phase}] candidates={len(specs)} models={len(specs)*seed_count:,} "
        f"steps={max_steps:,} elapsed={time.perf_counter()-start:.1f}s",
        flush=True,
    )
    del optimizer, model, train_x, train_y, full_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, predictions, exact_cpu


def choose_query_index(
    predictions: np.ndarray,
    fit_mask: np.ndarray,
    train_indices: Sequence[int],
    dataset_key: int,
    round_index: int,
) -> tuple[int, float, float]:
    cohort = predictions[np.asarray(fit_mask, dtype=bool)]
    if len(cohort) < 2:
        cohort = predictions
    state_count = predictions.shape[1]
    unseen = np.setdiff1d(
        np.arange(state_count, dtype=np.int64),
        np.asarray(train_indices, dtype=np.int64),
        assume_unique=False,
    )
    local = cohort[:, unseen]
    ones = local.sum(axis=0).astype(np.float64)
    probability_one = ones / len(local)
    if len(local) >= 2:
        same = (
            ones * (ones - 1)
            + (len(local) - ones) * (len(local) - ones - 1)
        ) / (len(local) * (len(local) - 1))
    else:
        same = np.ones(len(unseen), dtype=np.float64)
    minimum = float(np.min(same))
    candidates = np.flatnonzero(np.isclose(same, minimum, atol=1e-12))
    rng = np.random.default_rng(
        Config.QUERY_SEED + 1_000_003 * dataset_key + 10_007 * round_index
    )
    selected_local = int(rng.choice(candidates))
    return (
        int(unseen[selected_local]),
        float(same[selected_local]),
        float(probability_one[selected_local]),
    )


def metric_for_selection(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["unseen_pairwise_agreement"]),
        float(row["unseen_ball_mass_0p5pct"]),
        float(row["exact_modal_probability"]),
    )


def select_branch(
    policy: str,
    row_zero: dict[str, Any],
    row_one: dict[str, Any],
    path_id: int,
    round_index: int,
) -> tuple[int, str]:
    rows = [row_zero, row_one]
    valid = [
        index for index, row in enumerate(rows)
        if float(row["train_fit_rate"]) >= Config.MIN_VALID_FIT_RATE
    ]
    if not valid:
        best_fit = max(float(row["train_fit_rate"]) for row in rows)
        valid = [
            index for index, row in enumerate(rows)
            if float(row["train_fit_rate"]) == best_fit
        ]
        if len(valid) > 1:
            valid = [min(
                valid,
                key=lambda index: float(rows[index]["train_loss_mean"]),
            )]
        validity = "fallback_best_fit"
    else:
        validity = "fit_valid"

    if policy == "anti_consensus":
        selected = min(valid, key=lambda index: metric_for_selection(rows[index]))
    elif policy == "pro_consensus":
        selected = max(valid, key=lambda index: metric_for_selection(rows[index]))
    elif policy == "random_label":
        rng = np.random.default_rng(
            Config.RANDOM_POLICY_SEED
            + 1_000_003 * path_id
            + 10_007 * round_index
        )
        selected = int(rng.choice(valid))
    else:
        raise ValueError(f"未知策略：{policy}")
    return int(selected), validity


def initialize_paths(
    starts: Sequence[tuple[list[int], list[int]]],
) -> list[PathState]:
    paths: list[PathState] = []
    path_id = 0
    for start_id, (indices, labels) in enumerate(starts):
        for policy in Config.POLICIES:
            paths.append(PathState(
                path_id=path_id,
                start_id=start_id,
                policy=policy,
                train_indices=list(indices),
                train_labels=list(labels),
                selected_labels=[],
                queried_indices=[],
            ))
            path_id += 1
    return paths


def initial_specs(starts: Sequence[tuple[list[int], list[int]]]) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            spec_id=start_id,
            path_id=start_id,
            start_id=start_id,
            policy="shared_initial",
            branch_label=-1,
            initialization_key=start_id,
            train_indices=tuple(indices),
            train_labels=tuple(labels),
        )
        for start_id, (indices, labels) in enumerate(starts)
    ]


def path_specs_for_audit(paths: Sequence[PathState]) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            spec_id=path.path_id,
            path_id=path.path_id,
            start_id=path.start_id,
            policy=path.policy,
            branch_label=(path.selected_labels[-1] if path.selected_labels else -1),
            initialization_key=path.path_id,
            train_indices=tuple(path.train_indices),
            train_labels=tuple(path.train_labels),
        )
        for path in paths
    ]


def checkpoint_paths(
    output_dir: Path,
    round_index: int,
    paths: Sequence[PathState],
    predictions: np.ndarray,
    fit_masks: np.ndarray,
) -> None:
    write_json(output_dir / "checkpoint.json", {
        "completed_round": int(round_index),
        "paths": [asdict(path) for path in paths],
    })
    np.savez_compressed(
        output_dir / "checkpoint_committee.npz",
        predictions=np.asarray(predictions, dtype=np.uint8),
        fit_masks=np.asarray(fit_masks, dtype=bool),
    )


def load_checkpoint(
    output_dir: Path,
) -> tuple[int, list[PathState], np.ndarray, np.ndarray] | None:
    path = output_dir / "checkpoint.json"
    if not Config.RESUME or not path.exists():
        return None
    committee_path = output_dir / "checkpoint_committee.npz"
    if not committee_path.exists():
        raise RuntimeError("checkpoint.json 存在，但 committee 快照缺失。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    committee = np.load(committee_path)
    return (
        int(payload["completed_round"]),
        [PathState(**row) for row in payload["paths"]],
        committee["predictions"],
        committee["fit_masks"],
    )


def anf_metrics(bits: np.ndarray) -> dict[str, Any]:
    coefficients = np.asarray(bits, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(2 ** Config.INPUT_BITS):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.asarray(
        [int(term).bit_count() for term in terms], dtype=np.int64
    )
    return {
        "anf_degree": int(degrees.max()) if len(terms) else 0,
        "anf_term_count": int(len(terms)),
        "anf_literal_count": int(degrees.sum()),
    }


def essential_variables(bits: np.ndarray) -> list[int]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    variables: list[int] = []
    for variable in range(Config.INPUT_BITS):
        bit = Config.INPUT_BITS - 1 - variable
        mask = 1 << bit
        base = values[(values & mask) == 0]
        if np.any(bits[base] != bits[base | mask]):
            variables.append(variable)
    return variables


def signed_cardinality_threshold(bits: np.ndarray) -> tuple[str, str]:
    """识别带输入取反和无关变量的等权基数阈值。"""
    bits = np.asarray(bits, dtype=np.uint8)
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    signs: list[tuple[int, int]] = []
    for variable in range(Config.INPUT_BITS):
        integer_bit = Config.INPUT_BITS - 1 - variable
        mask = 1 << integer_bit
        base = values[(values & mask) == 0]
        low = bits[base].astype(np.int8)
        high = bits[base | mask].astype(np.int8)
        difference = high - low
        if np.any(difference > 0) and np.any(difference < 0):
            return "", ""
        if np.any(difference > 0):
            signs.append((variable, 1))
        elif np.any(difference < 0):
            signs.append((variable, -1))

    inputs = truth_table_inputs().astype(np.uint8)
    score = np.zeros(len(inputs), dtype=np.int16)
    rendered: list[str] = []
    for variable, sign in signs:
        if sign > 0:
            score += inputs[:, variable]
            rendered.append(f"x{variable}")
        else:
            score += 1 - inputs[:, variable]
            rendered.append(f"NOT x{variable}")
    pattern: list[int] = []
    for value in range(len(signs) + 1):
        local = bits[score == value]
        if not len(local) or len(np.unique(local)) != 1:
            return "", ""
        pattern.append(int(local[0]))
    if any(pattern[index] > pattern[index + 1] for index in range(len(pattern) - 1)):
        return "", ""
    if 1 not in pattern:
        return "constant", "0"
    threshold = pattern.index(1)
    if not rendered:
        return "constant", "1"
    return (
        "signed_cardinality_threshold",
        " + ".join(rendered) + f" >= {threshold}",
    )


def symbolic_metrics(bits: np.ndarray) -> dict[str, Any]:
    bits = np.asarray(bits, dtype=np.uint8)
    anf = anf_metrics(bits)
    variables = essential_variables(bits)
    family, formula = signed_cardinality_threshold(bits)
    if not family and int(anf["anf_degree"]) <= 1:
        family = "affine_gf2"
        formula = "affine parity"
    return {
        "truth_ones": int(bits.sum()),
        "essential_variable_count": len(variables),
        "essential_variables": variables,
        "named_symbolic_family": family,
        "named_symbolic_formula": formula,
        **anf,
    }


def is_narrow_posterior(row: dict[str, Any]) -> bool:
    return bool(
        float(row["train_fit_rate"]) >= Config.MIN_VALID_FIT_RATE
        and float(row["unseen_pairwise_agreement"])
        >= Config.STABLE_MIN_PAIRWISE_AGREEMENT
        and float(row["unseen_ball_mass_0p5pct"])
        >= Config.STABLE_MIN_BALL_MASS_0P5PCT
    )


def general_linear_threshold_metrics(bits: np.ndarray) -> dict[str, Any]:
    """用L1线性规划检查完整真值表是否为一般线性阈值函数。"""
    try:
        from scipy.optimize import linprog
    except Exception as exc:
        return {
            "linear_threshold_audit_available": False,
            "linear_threshold_audit_error": str(exc),
            "is_general_linear_threshold": False,
            "linear_threshold_weights": [],
            "linear_threshold_bias": float("nan"),
            "linear_threshold_weight_support": 0,
            "linear_threshold_l1": float("nan"),
            "linear_threshold_formula": "",
        }

    inputs = truth_table_inputs().astype(np.float64)
    labels = 2 * np.asarray(bits, dtype=np.float64) - 1
    design = np.concatenate(
        [inputs, np.ones((len(inputs), 1), dtype=np.float64)], axis=1
    )
    parameter_count = Config.INPUT_BITS + 1
    # y*(w.x+b)>=1，同时最小化|w|和|b|之和，便于获得稀疏小尺度表达。
    classification = -labels[:, None] * design
    constraint_count = len(classification) + 2 * parameter_count
    a_ub = np.zeros((constraint_count, 2 * parameter_count), dtype=np.float64)
    b_ub = np.concatenate([
        -np.ones(len(classification), dtype=np.float64),
        np.zeros(2 * parameter_count, dtype=np.float64),
    ])
    a_ub[:len(classification), :parameter_count] = classification
    for index in range(parameter_count):
        offset = len(classification) + 2 * index
        a_ub[offset, index] = 1
        a_ub[offset, parameter_count + index] = -1
        a_ub[offset + 1, index] = -1
        a_ub[offset + 1, parameter_count + index] = -1
    objective = np.concatenate([
        np.zeros(parameter_count, dtype=np.float64),
        np.ones(parameter_count, dtype=np.float64),
    ])
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * parameter_count
        + [(0, None)] * parameter_count,
        method="highs",
    )
    if not result.success:
        return {
            "linear_threshold_audit_available": True,
            "linear_threshold_audit_error": "",
            "is_general_linear_threshold": False,
            "linear_threshold_weights": [],
            "linear_threshold_bias": float("nan"),
            "linear_threshold_weight_support": 0,
            "linear_threshold_l1": float("nan"),
            "linear_threshold_formula": "",
        }

    parameters = result.x[:parameter_count]
    parameters[np.abs(parameters) < 1e-8] = 0
    weights = parameters[:-1]
    bias = float(parameters[-1])
    terms = [
        f"({float(weight):.6g})*x{index}"
        for index, weight in enumerate(weights)
        if abs(float(weight)) > 1e-8
    ]
    formula = " + ".join(terms) if terms else "0"
    formula += f" + ({bias:.6g}) >= 0"
    return {
        "linear_threshold_audit_available": True,
        "linear_threshold_audit_error": "",
        "is_general_linear_threshold": True,
        "linear_threshold_weights": weights.tolist(),
        "linear_threshold_bias": bias,
        "linear_threshold_weight_support": int(np.sum(np.abs(weights) > 1e-8)),
        "linear_threshold_l1": float(np.abs(parameters).sum()),
        "linear_threshold_formula": formula,
    }


def boundary_metrics(bits: np.ndarray) -> dict[str, float]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)
    influences: list[float] = []
    for integer_bit in range(Config.INPUT_BITS):
        mask = 1 << integer_bit
        base = values[(values & mask) == 0]
        influences.append(float(np.mean(bits[base] != bits[base | mask])))
    return {
        "total_influence": float(sum(influences)),
        "max_variable_influence": float(max(influences)),
    }


def optimal_decision_tree(bits: np.ndarray) -> tuple[int, int]:
    values = np.arange(2 ** Config.INPUT_BITS, dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def solve(fixed_mask: int, fixed_value: int) -> tuple[int, int]:
        selected = values[(values & fixed_mask) == fixed_value]
        outputs = bits[selected]
        if np.all(outputs == outputs[0]):
            return 1, 0
        best = (10 ** 9, 10 ** 9)
        for integer_bit in range(Config.INPUT_BITS):
            mask = 1 << integer_bit
            if fixed_mask & mask:
                continue
            low = solve(fixed_mask | mask, fixed_value)
            high = solve(fixed_mask | mask, fixed_value | mask)
            candidate = (low[0] + high[0], 1 + max(low[1], high[1]))
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
        integer_bit = int(order[depth])
        mask = 1 << integer_bit
        low = build(depth + 1, fixed_mask | mask, fixed_value)
        high = build(depth + 1, fixed_mask | mask, fixed_value | mask)
        if low == high:
            memo[key] = low
            return low
        node_key = (integer_bit, low, high)
        node = unique_nodes.get(node_key)
        if node is None:
            node = len(unique_nodes) + 2
            unique_nodes[node_key] = node
        memo[key] = node
        return node

    build(0, 0, 0)
    return len(unique_nodes)


def frontier_bdd_orders() -> list[tuple[int, ...]]:
    natural = tuple(range(Config.INPUT_BITS - 1, -1, -1))
    reverse = tuple(reversed(natural))
    orders = [natural, reverse]
    seen = set(orders)
    rng = np.random.default_rng(Config.QUERY_SEED + 991)
    target = 2 + Config.FRONTIER_BDD_RANDOM_ORDERS
    maximum_unique = math.factorial(Config.INPUT_BITS)
    target = min(target, maximum_unique)
    while len(orders) < target:
        candidate = tuple(map(int, rng.permutation(Config.INPUT_BITS)))
        if candidate not in seen:
            seen.add(candidate)
            orders.append(candidate)
    return orders


def frontier_symbolic_metrics(
    bits: np.ndarray,
    bdd_orders: Sequence[Sequence[int]],
) -> dict[str, Any]:
    base = symbolic_metrics(bits)
    threshold = general_linear_threshold_metrics(bits)
    leaves, depth = optimal_decision_tree(np.asarray(bits, dtype=np.uint8))
    bdd_counts = [robdd_node_count(bits, order) for order in bdd_orders]
    named_family = str(base["named_symbolic_family"])
    named_formula = str(base["named_symbolic_formula"])
    if not named_family and bool(threshold["is_general_linear_threshold"]):
        named_family = "general_linear_threshold"
        named_formula = str(threshold["linear_threshold_formula"])
    readable = bool(
        named_family
        or int(base["essential_variable_count"]) <= 4
        or int(base["anf_term_count"]) <= 16
        or leaves <= 16
        or min(bdd_counts) <= 16
    )
    return {
        **base,
        **threshold,
        **boundary_metrics(np.asarray(bits, dtype=np.uint8)),
        "optimal_decision_tree_leaves": int(leaves),
        "optimal_decision_tree_depth": int(depth),
        "robdd_nodes_min_tested": int(min(bdd_counts)),
        "frontier_named_symbolic_family": named_family,
        "frontier_named_symbolic_formula": named_formula,
        "frontier_symbolically_readable": readable,
    }


def save_plot(output_dir: Path, trajectory_rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    if not trajectory_rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    policies = list(Config.POLICIES)
    colors = {
        "anti_consensus": "#d62728",
        "pro_consensus": "#2ca02c",
        "random_label": "#1f77b4",
    }
    for policy in policies:
        local = [row for row in trajectory_rows if row["policy"] == policy]
        rounds = sorted({int(row["round"]) for row in local})
        agreement = [
            np.mean([
                float(row["unseen_pairwise_agreement"])
                for row in local if int(row["round"]) == round_index
            ])
            for round_index in rounds
        ]
        entropy = [
            np.mean([
                float(row["unseen_vote_entropy_bits"])
                for row in local if int(row["round"]) == round_index
            ])
            for round_index in rounds
        ]
        axes[0].plot(
            rounds, agreement, marker="o", label=policy, color=colors[policy]
        )
        axes[1].plot(
            rounds, entropy, marker="o", label=policy, color=colors[policy]
        )
    axes[0].set_title("Unseen pairwise agreement")
    axes[1].set_title("Unseen vote entropy")
    for axis in axes:
        axis.set_xlabel("completion round")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "policy_trajectories.png", dpi=180)
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


def paired_policy_comparisons(
    audit_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (int(row["start_id"]), str(row["policy"])): row
        for row in audit_rows
    }
    start_ids = sorted({int(row["start_id"]) for row in audit_rows})
    output: dict[str, Any] = {}
    for left, right in (
        ("anti_consensus", "pro_consensus"),
        ("anti_consensus", "random_label"),
    ):
        differences = np.asarray([
            float(by_key[(start_id, left)]["unseen_pairwise_agreement"])
            - float(by_key[(start_id, right)]["unseen_pairwise_agreement"])
            for start_id in start_ids
        ], dtype=np.float64)
        non_ties = differences[differences != 0]
        lower_count = int(np.sum(non_ties < 0))
        trial_count = len(non_ties)
        if trial_count:
            denominator = float(2 ** trial_count)
            lower_tail = sum(
                math.comb(trial_count, count)
                for count in range(0, lower_count + 1)
            ) / denominator
            upper_tail = sum(
                math.comb(trial_count, count)
                for count in range(lower_count, trial_count + 1)
            ) / denominator
            one_sided_lower_p = upper_tail
            two_sided_p = min(1.0, 2.0 * min(lower_tail, upper_tail))
        else:
            one_sided_lower_p = 1.0
            two_sided_p = 1.0
        output[f"{left}_minus_{right}"] = {
            "paired_start_count": len(start_ids),
            "mean_agreement_difference": float(differences.mean()),
            "median_agreement_difference": float(np.median(differences)),
            "left_lower_count_excluding_ties": lower_count,
            "left_higher_count_excluding_ties": int(np.sum(non_ties > 0)),
            "tie_count": int(np.sum(differences == 0)),
            "one_sided_exact_sign_test_p_left_lower": one_sided_lower_p,
            "two_sided_exact_sign_test_p": two_sided_p,
            "per_start_differences": differences.tolist(),
        }
    return output


def clone_initial_metrics_for_paths(
    paths: Sequence[PathState],
    initial_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_start = {int(row["start_id"]): row for row in initial_rows}
    output: list[dict[str, Any]] = []
    for path in paths:
        base = by_start[path.start_id]
        output.append({
            **base,
            "round": 0,
            "path_id": path.path_id,
            "start_id": path.start_id,
            "policy": path.policy,
            "selected_label": -1,
            "queried_index": -1,
            "parent_query_point_agreement": float("nan"),
            "parent_query_probability_one": float("nan"),
            "selection_validity": "shared_initial",
        })
    return output


def save_frontier_checkpoint(
    output_dir: Path,
    stage: str,
    anti_round: int,
    pro_round: int,
    anti_paths: Sequence[PathState],
    forks: Sequence[FrontierForkState],
    anti_predictions: np.ndarray,
    anti_fit_masks: np.ndarray,
    fork_predictions: np.ndarray,
    fork_fit_masks: np.ndarray,
) -> None:
    write_json(output_dir / "frontier_checkpoint.json", {
        "stage": stage,
        "anti_round": int(anti_round),
        "pro_round": int(pro_round),
        "anti_paths": [asdict(path) for path in anti_paths],
        "forks": [asdict(fork) for fork in forks],
    })
    np.savez_compressed(
        output_dir / "frontier_checkpoint_committees.npz",
        anti_predictions=np.asarray(anti_predictions, dtype=np.uint8),
        anti_fit_masks=np.asarray(anti_fit_masks, dtype=bool),
        fork_predictions=np.asarray(fork_predictions, dtype=np.uint8),
        fork_fit_masks=np.asarray(fork_fit_masks, dtype=bool),
    )


def load_frontier_checkpoint(
    output_dir: Path,
) -> tuple[
    str,
    int,
    int,
    list[PathState],
    list[FrontierForkState],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None:
    metadata_path = output_dir / "frontier_checkpoint.json"
    committee_path = output_dir / "frontier_checkpoint_committees.npz"
    if not Config.RESUME or not metadata_path.exists():
        return None
    if not committee_path.exists():
        raise RuntimeError("frontier checkpoint元数据存在，但committee快照缺失。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    committees = np.load(committee_path)
    return (
        str(metadata["stage"]),
        int(metadata["anti_round"]),
        int(metadata["pro_round"]),
        [PathState(**row) for row in metadata["anti_paths"]],
        [FrontierForkState(**row) for row in metadata["forks"]],
        committees["anti_predictions"],
        committees["anti_fit_masks"],
        committees["fork_predictions"],
        committees["fork_fit_masks"],
    )


def append_cutoff_forks(
    cutoff: int,
    anti_paths: Sequence[PathState],
    anti_predictions: np.ndarray,
    anti_fit_masks: np.ndarray,
    forks: list[FrontierForkState],
    fork_prediction_list: list[np.ndarray],
    fork_fit_list: list[np.ndarray],
) -> None:
    existing = {
        (fork.start_id, fork.cutoff_train_count) for fork in forks
    }
    for path_index, path in enumerate(anti_paths):
        key = (path.start_id, cutoff)
        if key in existing:
            continue
        fork_id = len(forks)
        forks.append(FrontierForkState(
            fork_id=fork_id,
            start_id=path.start_id,
            cutoff_train_count=cutoff,
            train_indices=list(path.train_indices),
            train_labels=list(path.train_labels),
            pro_queried_indices=[],
            pro_selected_labels=[],
        ))
        fork_prediction_list.append(anti_predictions[path_index].copy())
        fork_fit_list.append(anti_fit_masks[path_index].copy())


def empty_fork_committee() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty(
            (0, Config.DISCOVERY_SEED_COUNT, 2 ** Config.INPUT_BITS),
            dtype=np.uint8,
        ),
        np.empty((0, Config.DISCOVERY_SEED_COUNT), dtype=bool),
    )


def frontier_initial_specs(
    starts: Sequence[tuple[list[int], list[int]]],
) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            spec_id=start_id,
            path_id=start_id,
            start_id=start_id,
            policy="anti_trunk",
            branch_label=-1,
            initialization_key=start_id,
            train_indices=tuple(indices),
            train_labels=tuple(labels),
        )
        for start_id, (indices, labels) in enumerate(starts)
    ]


def initialize_anti_paths(
    starts: Sequence[tuple[list[int], list[int]]],
) -> list[PathState]:
    return [
        PathState(
            path_id=start_id,
            start_id=start_id,
            policy="anti_trunk",
            train_indices=list(indices),
            train_labels=list(labels),
            selected_labels=[],
            queried_indices=[],
        )
        for start_id, (indices, labels) in enumerate(starts)
    ]


def committee_row(
    phase: str,
    predictions: np.ndarray,
    fit_mask: np.ndarray,
    train_indices: Sequence[int],
    **metadata: Any,
) -> dict[str, Any]:
    metrics, _, _ = summarize_prediction_cohort(
        predictions, fit_mask, train_indices
    )
    return {
        "phase": phase,
        "train_count": len(train_indices),
        "train_fit_rate": float(np.mean(fit_mask)),
        "train_loss_mean": float("nan"),
        "train_loss_median": float("nan"),
        **metadata,
        **metrics,
    }


def run_anti_frontier_stage(
    output_dir: Path,
    device: torch.device,
    starts: Sequence[tuple[list[int], list[int]]],
    checkpoint: tuple[Any, ...] | None,
) -> tuple[
    int,
    list[PathState],
    list[FrontierForkState],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    anti_trajectory_path = output_dir / "anti_trunk_trajectory.csv"
    anti_candidates_path = output_dir / "anti_trunk_branch_candidates.csv"
    anti_trajectory: list[dict[str, Any]] = list(read_csv(anti_trajectory_path))
    anti_candidates: list[dict[str, Any]] = list(read_csv(anti_candidates_path))
    cutoffs = tuple(sorted(map(int, Config.FRONTIER_CUTOFF_COUNTS)))
    if cutoffs[0] != Config.INITIAL_TRAIN_COUNT:
        raise ValueError("FRONTIER_CUTOFF_COUNTS首项必须等于INITIAL_TRAIN_COUNT。")
    if cutoffs[-1] >= 2 ** Config.INPUT_BITS:
        raise ValueError("最大cutoff必须小于完整输入空间。")

    if checkpoint is None:
        anti_paths = initialize_anti_paths(starts)
        rows, anti_predictions, anti_fit_masks = train_and_evaluate(
            "anti_initial",
            frontier_initial_specs(starts),
            Config.DISCOVERY_SEED_COUNT,
            Config.INITIALIZATION_SEED,
            Config.DISCOVERY_MAX_STEPS,
            Config.DISCOVERY_LEARNING_RATE,
            Config.DISCOVERY_WEIGHT_DECAY,
            device,
        )
        anti_trajectory = [
            {
                **row,
                "anti_round": 0,
                "queried_index": -1,
                "selected_label": -1,
                "parent_query_point_agreement": float("nan"),
                "parent_query_probability_one": float("nan"),
            }
            for row in rows
        ]
        forks: list[FrontierForkState] = []
        fork_prediction_list: list[np.ndarray] = []
        fork_fit_list: list[np.ndarray] = []
        append_cutoff_forks(
            Config.INITIAL_TRAIN_COUNT,
            anti_paths,
            anti_predictions,
            anti_fit_masks,
            forks,
            fork_prediction_list,
            fork_fit_list,
        )
        fork_predictions = np.stack(fork_prediction_list, axis=0)
        fork_fit_masks = np.stack(fork_fit_list, axis=0)
        anti_round = 0
    else:
        (
            stage,
            anti_round,
            _,
            anti_paths,
            forks,
            anti_predictions,
            anti_fit_masks,
            fork_predictions,
            fork_fit_masks,
        ) = checkpoint
        if stage != "anti":
            return (
                anti_round,
                anti_paths,
                forks,
                anti_predictions,
                anti_fit_masks,
                fork_predictions,
                fork_fit_masks,
            )
        fork_prediction_list = [row.copy() for row in fork_predictions]
        fork_fit_list = [row.copy() for row in fork_fit_masks]

    target_round = cutoffs[-1] - Config.INITIAL_TRAIN_COUNT
    for round_index in range(anti_round + 1, target_round + 1):
        train_count = Config.INITIAL_TRAIN_COUNT + round_index
        print(
            f"\n=== anti trunk round {round_index}/{target_round} | "
            f"train_n={train_count} ===",
            flush=True,
        )
        child_specs: list[DatasetSpec] = []
        query_metadata: dict[int, tuple[int, float, float]] = {}
        for path_index, path in enumerate(anti_paths):
            query_index, point_agreement, probability_one = choose_query_index(
                anti_predictions[path_index],
                anti_fit_masks[path_index],
                path.train_indices,
                stable_dataset_key(path.train_indices, path.train_labels),
                round_index,
            )
            query_metadata[path.path_id] = (
                query_index, point_agreement, probability_one
            )
            initialization_key = stable_dataset_key(
                path.train_indices, path.train_labels
            )
            for label in (0, 1):
                child_specs.append(DatasetSpec(
                    spec_id=len(child_specs),
                    path_id=path.path_id,
                    start_id=path.start_id,
                    policy="anti_trunk",
                    branch_label=label,
                    initialization_key=initialization_key,
                    train_indices=tuple(path.train_indices + [query_index]),
                    train_labels=tuple(path.train_labels + [label]),
                ))

        child_rows, child_predictions, child_fit = train_and_evaluate(
            f"anti_round_{round_index}",
            child_specs,
            Config.DISCOVERY_SEED_COUNT,
            Config.INITIALIZATION_SEED + 100_000_019 * round_index,
            Config.DISCOVERY_MAX_STEPS,
            Config.DISCOVERY_LEARNING_RATE,
            Config.DISCOVERY_WEIGHT_DECAY,
            device,
        )
        by_path: dict[int, dict[int, int]] = {}
        for child_index, spec in enumerate(child_specs):
            by_path.setdefault(spec.path_id, {})[spec.branch_label] = child_index
        next_predictions = np.empty_like(anti_predictions)
        next_fit = np.empty_like(anti_fit_masks)
        selected_rows: list[dict[str, Any]] = []
        for path_index, path in enumerate(anti_paths):
            zero_index = by_path[path.path_id][0]
            one_index = by_path[path.path_id][1]
            selected_label, validity = select_branch(
                "anti_consensus",
                child_rows[zero_index],
                child_rows[one_index],
                path.path_id,
                round_index,
            )
            selected_index = by_path[path.path_id][selected_label]
            query_index, point_agreement, probability_one = query_metadata[
                path.path_id
            ]
            for label, child_index in ((0, zero_index), (1, one_index)):
                anti_candidates.append({
                    **child_rows[child_index],
                    "anti_round": round_index,
                    "queried_index": query_index,
                    "parent_query_point_agreement": point_agreement,
                    "parent_query_probability_one": probability_one,
                    "selected": label == selected_label,
                    "selection_validity": validity,
                })
            path.train_indices.append(query_index)
            path.train_labels.append(selected_label)
            path.queried_indices.append(query_index)
            path.selected_labels.append(selected_label)
            next_predictions[path_index] = child_predictions[selected_index]
            next_fit[path_index] = child_fit[selected_index]
            selected_row = {
                **child_rows[selected_index],
                "anti_round": round_index,
                "queried_index": query_index,
                "selected_label": selected_label,
                "parent_query_point_agreement": point_agreement,
                "parent_query_probability_one": probability_one,
                "selection_validity": validity,
            }
            anti_trajectory.append(selected_row)
            selected_rows.append(selected_row)
        anti_predictions = next_predictions
        anti_fit_masks = next_fit
        if train_count in cutoffs:
            append_cutoff_forks(
                train_count,
                anti_paths,
                anti_predictions,
                anti_fit_masks,
                forks,
                fork_prediction_list,
                fork_fit_list,
            )
            fork_predictions = np.stack(fork_prediction_list, axis=0)
            fork_fit_masks = np.stack(fork_fit_list, axis=0)
            print(
                f"snapshot cutoff n={train_count} | total forks={len(forks)}",
                flush=True,
            )
        write_csv(anti_trajectory_path, anti_trajectory)
        write_csv(anti_candidates_path, anti_candidates)
        save_frontier_checkpoint(
            output_dir,
            "anti",
            round_index,
            0,
            anti_paths,
            forks,
            anti_predictions,
            anti_fit_masks,
            fork_predictions,
            fork_fit_masks,
        )
        print(
            f"anti mean agreement={np.mean([float(r['unseen_pairwise_agreement']) for r in selected_rows]):.5f} | "
            f"fit={np.mean([float(r['train_fit_rate']) for r in selected_rows]):.3f}",
            flush=True,
        )

    save_frontier_checkpoint(
        output_dir,
        "pro",
        target_round,
        0,
        anti_paths,
        forks,
        anti_predictions,
        anti_fit_masks,
        fork_predictions,
        fork_fit_masks,
    )
    return (
        target_round,
        anti_paths,
        forks,
        anti_predictions,
        anti_fit_masks,
        fork_predictions,
        fork_fit_masks,
    )


def run_pro_collapse_stage(
    output_dir: Path,
    device: torch.device,
    anti_round: int,
    anti_paths: list[PathState],
    forks: list[FrontierForkState],
    anti_predictions: np.ndarray,
    anti_fit_masks: np.ndarray,
    fork_predictions: np.ndarray,
    fork_fit_masks: np.ndarray,
    start_pro_round: int,
) -> tuple[list[FrontierForkState], np.ndarray, np.ndarray, int]:
    trajectory_path = output_dir / "pro_collapse_trajectory.csv"
    candidates_path = output_dir / "pro_collapse_branch_candidates.csv"
    trajectory: list[dict[str, Any]] = list(read_csv(trajectory_path))
    candidates: list[dict[str, Any]] = list(read_csv(candidates_path))

    if not trajectory:
        initial_rows: list[dict[str, Any]] = []
        for fork in forks:
            row = committee_row(
                "pro_initial_snapshot",
                fork_predictions[fork.fork_id],
                fork_fit_masks[fork.fork_id],
                fork.train_indices,
                fork_id=fork.fork_id,
                path_id=fork.fork_id,
                start_id=fork.start_id,
                policy=f"anti_n{fork.cutoff_train_count}_then_pro",
                cutoff_train_count=fork.cutoff_train_count,
                pro_round=0,
                pro_additional_samples=0,
                queried_index=-1,
                selected_label=-1,
            )
            fork.stable_streak = 1 if is_narrow_posterior(row) else 0
            initial_rows.append(row)
        trajectory.extend(initial_rows)
        write_csv(trajectory_path, trajectory)

    global_round = start_pro_round
    while True:
        active = [
            fork for fork in forks
            if fork.status == "active"
            and len(fork.pro_selected_labels)
            < Config.PRO_MAX_ADDITIONAL_SAMPLES
        ]
        if not active:
            break
        global_round += 1
        print(
            f"\n=== pro collapse global round {global_round} | "
            f"active forks={len(active)}/{len(forks)} ===",
            flush=True,
        )
        groups: dict[int, list[FrontierForkState]] = {}
        for fork in active:
            groups.setdefault(len(fork.train_indices), []).append(fork)

        for train_count in sorted(groups):
            local_forks = groups[train_count]
            child_specs: list[DatasetSpec] = []
            query_metadata: dict[int, tuple[int, float, float]] = {}
            for fork in local_forks:
                query_index, point_agreement, probability_one = choose_query_index(
                    fork_predictions[fork.fork_id],
                    fork_fit_masks[fork.fork_id],
                    fork.train_indices,
                    stable_dataset_key(fork.train_indices, fork.train_labels),
                    len(fork.pro_selected_labels) + 1,
                )
                query_metadata[fork.fork_id] = (
                    query_index, point_agreement, probability_one
                )
                initialization_key = stable_dataset_key(
                    fork.train_indices, fork.train_labels
                )
                for label in (0, 1):
                    child_specs.append(DatasetSpec(
                        spec_id=len(child_specs),
                        path_id=fork.fork_id,
                        start_id=fork.start_id,
                        policy=f"anti_n{fork.cutoff_train_count}_then_pro",
                        branch_label=label,
                        initialization_key=initialization_key,
                        train_indices=tuple(fork.train_indices + [query_index]),
                        train_labels=tuple(fork.train_labels + [label]),
                    ))

            child_rows, child_predictions, child_fit = train_and_evaluate(
                f"pro_global_{global_round}_n{train_count+1}",
                child_specs,
                Config.DISCOVERY_SEED_COUNT,
                Config.INITIALIZATION_SEED
                + 700_000_033 * global_round
                + 1_000_003 * train_count,
                Config.DISCOVERY_MAX_STEPS,
                Config.DISCOVERY_LEARNING_RATE,
                Config.DISCOVERY_WEIGHT_DECAY,
                device,
            )
            by_fork: dict[int, dict[int, int]] = {}
            for child_index, spec in enumerate(child_specs):
                by_fork.setdefault(spec.path_id, {})[spec.branch_label] = child_index

            for fork in local_forks:
                zero_index = by_fork[fork.fork_id][0]
                one_index = by_fork[fork.fork_id][1]
                selected_label, validity = select_branch(
                    "pro_consensus",
                    child_rows[zero_index],
                    child_rows[one_index],
                    fork.fork_id,
                    len(fork.pro_selected_labels) + 1,
                )
                selected_index = by_fork[fork.fork_id][selected_label]
                query_index, point_agreement, probability_one = query_metadata[
                    fork.fork_id
                ]
                for label, child_index in ((0, zero_index), (1, one_index)):
                    candidates.append({
                        **child_rows[child_index],
                        "fork_id": fork.fork_id,
                        "cutoff_train_count": fork.cutoff_train_count,
                        "pro_round": global_round,
                        "pro_additional_samples": len(fork.pro_selected_labels) + 1,
                        "queried_index": query_index,
                        "parent_query_point_agreement": point_agreement,
                        "parent_query_probability_one": probability_one,
                        "selected": label == selected_label,
                        "selection_validity": validity,
                    })
                fork.train_indices.append(query_index)
                fork.train_labels.append(selected_label)
                fork.pro_queried_indices.append(query_index)
                fork.pro_selected_labels.append(selected_label)
                fork_predictions[fork.fork_id] = child_predictions[selected_index]
                fork_fit_masks[fork.fork_id] = child_fit[selected_index]
                selected_row = {
                    **child_rows[selected_index],
                    "fork_id": fork.fork_id,
                    "cutoff_train_count": fork.cutoff_train_count,
                    "pro_round": global_round,
                    "pro_additional_samples": len(fork.pro_selected_labels),
                    "queried_index": query_index,
                    "selected_label": selected_label,
                    "parent_query_point_agreement": point_agreement,
                    "parent_query_probability_one": probability_one,
                    "selection_validity": validity,
                }
                if is_narrow_posterior(selected_row):
                    fork.stable_streak += 1
                else:
                    fork.stable_streak = 0
                if fork.stable_streak >= Config.STABLE_CONSECUTIVE_ROUNDS:
                    fork.status = "discovery_stable"
                elif (
                    len(fork.pro_selected_labels)
                    >= Config.PRO_MAX_ADDITIONAL_SAMPLES
                ):
                    fork.status = "budget_exhausted"
                selected_row["stable_streak"] = fork.stable_streak
                selected_row["fork_status"] = fork.status
                trajectory.append(selected_row)

        write_csv(trajectory_path, trajectory)
        write_csv(candidates_path, candidates)
        save_frontier_checkpoint(
            output_dir,
            "pro",
            anti_round,
            global_round,
            anti_paths,
            forks,
            anti_predictions,
            anti_fit_masks,
            fork_predictions,
            fork_fit_masks,
        )
        stable_count = sum(fork.status == "discovery_stable" for fork in forks)
        exhausted_count = sum(fork.status == "budget_exhausted" for fork in forks)
        print(
            f"pro status | stable={stable_count}/{len(forks)} | "
            f"budget_exhausted={exhausted_count}",
            flush=True,
        )
    return forks, fork_predictions, fork_fit_masks, global_round


def audit_frontier_endpoints(
    output_dir: Path,
    device: torch.device,
    forks: Sequence[FrontierForkState],
) -> list[dict[str, Any]]:
    groups: dict[int, list[FrontierForkState]] = {}
    for fork in forks:
        groups.setdefault(len(fork.train_indices), []).append(fork)
    output: list[dict[str, Any]] = []
    consensus_by_fork: dict[int, np.ndarray] = {}
    bdd_orders = frontier_bdd_orders()
    for train_count in sorted(groups):
        local = groups[train_count]
        specs = [
            DatasetSpec(
                spec_id=fork.fork_id,
                path_id=fork.fork_id,
                start_id=fork.start_id,
                policy=f"anti_n{fork.cutoff_train_count}_then_pro",
                branch_label=(
                    fork.pro_selected_labels[-1]
                    if fork.pro_selected_labels else -1
                ),
                initialization_key=fork.fork_id,
                train_indices=tuple(fork.train_indices),
                train_labels=tuple(fork.train_labels),
            )
            for fork in local
        ]
        rows, predictions, fit_masks = train_and_evaluate(
            f"fresh_audit_n{train_count}",
            specs,
            Config.AUDIT_SEED_COUNT,
            Config.AUDIT_INITIALIZATION_SEED,
            Config.AUDIT_MAX_STEPS,
            Config.AUDIT_LEARNING_RATE,
            Config.AUDIT_WEIGHT_DECAY,
            device,
        )
        fork_by_id = {fork.fork_id: fork for fork in local}
        for index, row in enumerate(rows):
            fork_id = int(row["path_id"])
            fork = fork_by_id[fork_id]
            _, consensus, _ = summarize_prediction_cohort(
                predictions[index], fit_masks[index], fork.train_indices
            )
            consensus_by_fork[fork_id] = consensus
            narrow = is_narrow_posterior(row)
            complexity = frontier_symbolic_metrics(consensus, bdd_orders)
            result = {
                **row,
                "fork_id": fork_id,
                "cutoff_train_count": fork.cutoff_train_count,
                "pro_additional_samples": len(fork.pro_selected_labels),
                "final_train_count": len(fork.train_indices),
                "discovery_status": fork.status,
                "fresh_narrow_posterior": narrow,
                "train_indices": fork.train_indices,
                "train_labels": fork.train_labels,
                "pro_queried_indices": fork.pro_queried_indices,
                "pro_selected_labels": fork.pro_selected_labels,
                **complexity,
            }
            audit_available = bool(
                complexity["linear_threshold_audit_available"]
            )
            result["counterexample_candidate"] = bool(
                narrow
                and audit_available
                and not complexity["frontier_symbolically_readable"]
            )
            result["unresolved_due_missing_linear_threshold_audit"] = bool(
                narrow
                and not audit_available
                and not complexity["frontier_symbolically_readable"]
            )
            output.append(result)

    ordered_ids = sorted(consensus_by_fork)
    np.savez_compressed(
        output_dir / "fresh_audit_consensus_functions.npz",
        fork_ids=np.asarray(ordered_ids, dtype=np.int64),
        consensus_bits=np.stack(
            [consensus_by_fork[fork_id] for fork_id in ordered_ids], axis=0
        ),
    )
    write_csv(output_dir / "fresh_audit_summary.csv", output)
    write_csv(
        output_dir / "counterexample_candidates.csv",
        [row for row in output if row["counterexample_candidate"]],
    )
    return output


def summarize_frontier(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cutoff_rows: list[dict[str, Any]] = []
    for cutoff in sorted({int(row["cutoff_train_count"]) for row in rows}):
        local = [row for row in rows if int(row["cutoff_train_count"]) == cutoff]
        narrow = [row for row in local if bool(row["fresh_narrow_posterior"])]
        readable = [
            row for row in narrow if bool(row["frontier_symbolically_readable"])
        ]
        additions = np.asarray([
            int(row["pro_additional_samples"]) for row in local
        ], dtype=np.float64)
        cutoff_rows.append({
            "cutoff_train_count": cutoff,
            "fork_count": len(local),
            "fresh_narrow_count": len(narrow),
            "fresh_narrow_fraction": len(narrow) / len(local),
            "readable_among_narrow_count": len(readable),
            "counterexample_candidate_count": sum(
                bool(row["counterexample_candidate"]) for row in local
            ),
            "pro_additional_mean": float(additions.mean()),
            "pro_additional_median": float(np.median(additions)),
            "final_train_count_mean": float(np.mean([
                int(row["final_train_count"]) for row in local
            ])),
            "unique_consensus_functions": len({
                row["consensus_fingerprint"] for row in narrow
            }),
            "linear_threshold_among_narrow": sum(
                bool(row["is_general_linear_threshold"]) for row in narrow
            ),
            "essential_variables_mean_among_narrow": (
                float(np.mean([
                    int(row["essential_variable_count"]) for row in narrow
                ])) if narrow else float("nan")
            ),
            "anf_terms_median_among_narrow": (
                float(np.median([
                    int(row["anf_term_count"]) for row in narrow
                ])) if narrow else float("nan")
            ),
            "bdd_nodes_median_among_narrow": (
                float(np.median([
                    int(row["robdd_nodes_min_tested"]) for row in narrow
                ])) if narrow else float("nan")
            ),
        })
    summary = {
        "status": "complete",
        "fork_count": len(rows),
        "fresh_narrow_count": sum(
            bool(row["fresh_narrow_posterior"]) for row in rows
        ),
        "fresh_readable_among_narrow": sum(
            bool(row["fresh_narrow_posterior"])
            and bool(row["frontier_symbolically_readable"])
            for row in rows
        ),
        "counterexample_candidate_count": sum(
            bool(row["counterexample_candidate"]) for row in rows
        ),
        "cutoff_summary": cutoff_rows,
    }
    return cutoff_rows, summary


def save_frontier_plot(
    output_dir: Path,
    cutoff_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "frontier_plot_error.txt").write_text(
            str(exc), encoding="utf-8"
        )
        return
    cutoffs = np.asarray([
        int(row["cutoff_train_count"]) for row in cutoff_rows
    ])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(cutoffs, [
        float(row["pro_additional_median"]) for row in cutoff_rows
    ], marker="o")
    axes[0].set_ylabel("median pro additions")
    axes[1].plot(cutoffs, [
        float(row["fresh_narrow_fraction"]) for row in cutoff_rows
    ], marker="o")
    axes[1].set_ylabel("fresh narrow fraction")
    axes[2].plot(cutoffs, [
        float(row["bdd_nodes_median_among_narrow"]) for row in cutoff_rows
    ], marker="o", label="ROBDD nodes")
    axes[2].plot(cutoffs, [
        float(row["essential_variables_mean_among_narrow"])
        for row in cutoff_rows
    ], marker="s", label="essential variables")
    axes[2].set_ylabel("endpoint complexity proxy")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("anti-prefix cutoff n")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "consensus_complexity_frontier.png", dpi=180)
    plt.close(figure)


def frontier_main() -> None:
    apply_frontier_smoke_overrides()
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    print("=== Consensus Complexity Frontier ===", flush=True)
    print(
        f"device={device} | task={Config.INPUT_BITS}->1 | "
        f"states={2**Config.INPUT_BITS:,} | starts={Config.START_COUNT} | "
        f"cutoffs={list(Config.FRONTIER_CUTOFF_COUNTS)}",
        flush=True,
    )
    print(
        f"MLP={Config.INPUT_BITS}->{Config.WIDTH}x{Config.HIDDEN_LAYERS}->1 tanh | "
        f"params/model={parameter_count_per_model():,} | "
        f"discovery seeds={Config.DISCOVERY_SEED_COUNT} | "
        f"fresh audit seeds={Config.AUDIT_SEED_COUNT}",
        flush=True,
    )
    start_time = time.perf_counter()
    starts = initial_datasets()
    if not (output_dir / "initial_datasets.csv").exists():
        write_csv(output_dir / "initial_datasets.csv", [
            {
                "start_id": index,
                "train_indices": indices,
                "train_labels": labels,
                "signature": dataset_signature(indices, labels),
            }
            for index, (indices, labels) in enumerate(starts)
        ])
    checkpoint = load_frontier_checkpoint(output_dir)

    try:
        (
            anti_round,
            anti_paths,
            forks,
            anti_predictions,
            anti_fit_masks,
            fork_predictions,
            fork_fit_masks,
        ) = run_anti_frontier_stage(output_dir, device, starts, checkpoint)
        start_pro_round = 0
        loaded = load_frontier_checkpoint(output_dir)
        if loaded is not None and loaded[0] == "pro":
            start_pro_round = int(loaded[2])
            forks = loaded[4]
            anti_predictions = loaded[5]
            anti_fit_masks = loaded[6]
            fork_predictions = loaded[7]
            fork_fit_masks = loaded[8]
        forks, fork_predictions, fork_fit_masks, _ = run_pro_collapse_stage(
            output_dir,
            device,
            anti_round,
            anti_paths,
            forks,
            anti_predictions,
            anti_fit_masks,
            fork_predictions,
            fork_fit_masks,
            start_pro_round,
        )
        print("\n=== fresh-seed endpoint audit ===", flush=True)
        audit_rows = audit_frontier_endpoints(output_dir, device, forks)
        cutoff_rows, summary = summarize_frontier(audit_rows)
        summary["elapsed_seconds"] = time.perf_counter() - start_time
        write_csv(output_dir / "frontier_by_cutoff.csv", cutoff_rows)
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "frontier_forks.csv", [
            asdict(fork) for fork in forks
        ])
        save_frontier_plot(output_dir, cutoff_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except KeyboardInterrupt:
        write_json(output_dir / "interrupted.json", {
            "status": "interrupted",
            "elapsed_seconds": time.perf_counter() - start_time,
            "message": "已保存到最近完整round，保持RESUME=True重跑即可继续。",
        })
        print("收到中断，已保存最近完整round。", flush=True)
    finally:
        if Config.PACKAGE_RESULTS:
            archive = package_results(output_dir)
            print(f"下载压缩包：{archive}", flush=True)


def main() -> None:
    apply_smoke_overrides()
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 PyTorch 看不到 GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    print("=== Adversarial Disagreement Completion Pilot ===", flush=True)
    print(
        f"device={device} | starts={Config.START_COUNT} | "
        f"policies={list(Config.POLICIES)} | rounds={Config.ROUNDS}",
        flush=True,
    )
    print(
        f"task={Config.INPUT_BITS}->1 | states={2**Config.INPUT_BITS:,} | "
        f"MLP={Config.INPUT_BITS}->{Config.WIDTH}x{Config.HIDDEN_LAYERS}->1 "
        f"tanh | params/model={parameter_count_per_model():,}",
        flush=True,
    )
    print(
        f"discovery={Config.DISCOVERY_SEED_COUNT} seeds x "
        f"{Config.DISCOVERY_MAX_STEPS:,} steps | "
        f"audit={Config.AUDIT_SEED_COUNT} fresh seeds x "
        f"{Config.AUDIT_MAX_STEPS:,} steps",
        flush=True,
    )

    start_time = time.perf_counter()
    starts = initial_datasets()
    starts_rows = [
        {
            "start_id": index,
            "train_indices": indices,
            "train_labels": labels,
            "signature": dataset_signature(indices, labels),
        }
        for index, (indices, labels) in enumerate(starts)
    ]
    if not (output_dir / "initial_datasets.csv").exists():
        write_csv(output_dir / "initial_datasets.csv", starts_rows)

    trajectory_path = output_dir / "selected_trajectory.csv"
    candidates_path = output_dir / "branch_candidates.csv"
    trajectory_rows: list[dict[str, Any]] = list(read_csv(trajectory_path))
    candidate_rows: list[dict[str, Any]] = list(read_csv(candidates_path))
    checkpoint = load_checkpoint(output_dir)

    # 为了恢复后仍严格得到当前 committee，重新训练当前路径一次即可；结果只由
    # 当前数据集和固定初始化键决定，不继承历史权重。
    if checkpoint is None:
        paths = initialize_paths(starts)
        initial = initial_specs(starts)
        initial_rows, initial_predictions, initial_fit = train_and_evaluate(
            "initial",
            initial,
            Config.DISCOVERY_SEED_COUNT,
            Config.INITIALIZATION_SEED,
            Config.DISCOVERY_MAX_STEPS,
            Config.DISCOVERY_LEARNING_RATE,
            Config.DISCOVERY_WEIGHT_DECAY,
            device,
        )
        trajectory_rows = clone_initial_metrics_for_paths(paths, initial_rows)
        write_csv(trajectory_path, trajectory_rows)
        current_predictions = np.stack([
            initial_predictions[path.start_id] for path in paths
        ], axis=0)
        current_fit = np.stack([
            initial_fit[path.start_id] for path in paths
        ], axis=0)
        completed_round = 0
        checkpoint_paths(
            output_dir,
            completed_round,
            paths,
            current_predictions,
            current_fit,
        )
    else:
        completed_round, paths, current_predictions, current_fit = checkpoint
        print(f"resumed after round={completed_round}", flush=True)

    try:
        for round_index in range(completed_round + 1, Config.ROUNDS + 1):
            print(
                f"\n=== completion round {round_index}/{Config.ROUNDS} | "
                f"train_n={Config.INITIAL_TRAIN_COUNT+round_index} ===",
                flush=True,
            )
            child_specs: list[DatasetSpec] = []
            query_metadata: dict[int, tuple[int, float, float]] = {}
            for path_index, path in enumerate(paths):
                query_index, point_agreement, probability_one = choose_query_index(
                    current_predictions[path_index],
                    current_fit[path_index],
                    path.train_indices,
                    stable_dataset_key(path.train_indices, path.train_labels),
                    round_index,
                )
                query_metadata[path.path_id] = (
                    query_index, point_agreement, probability_one
                )
                initialization_key = stable_dataset_key(
                    path.train_indices, path.train_labels
                )
                for label in (0, 1):
                    child_specs.append(DatasetSpec(
                        spec_id=len(child_specs),
                        path_id=path.path_id,
                        start_id=path.start_id,
                        policy=path.policy,
                        branch_label=label,
                        initialization_key=initialization_key,
                        train_indices=tuple(path.train_indices + [query_index]),
                        train_labels=tuple(path.train_labels + [label]),
                    ))

            child_rows, child_predictions, child_fit = train_and_evaluate(
                f"round_{round_index}",
                child_specs,
                Config.DISCOVERY_SEED_COUNT,
                Config.INITIALIZATION_SEED + 100_000_019 * round_index,
                Config.DISCOVERY_MAX_STEPS,
                Config.DISCOVERY_LEARNING_RATE,
                Config.DISCOVERY_WEIGHT_DECAY,
                device,
            )
            by_path: dict[int, dict[int, int]] = {}
            for child_index, spec in enumerate(child_specs):
                by_path.setdefault(spec.path_id, {})[spec.branch_label] = child_index

            next_predictions = np.empty_like(current_predictions)
            next_fit = np.empty_like(current_fit)
            round_selected: list[dict[str, Any]] = []
            for path_index, path in enumerate(paths):
                zero_index = by_path[path.path_id][0]
                one_index = by_path[path.path_id][1]
                zero_row = child_rows[zero_index]
                one_row = child_rows[one_index]
                selected_label, validity = select_branch(
                    path.policy,
                    zero_row,
                    one_row,
                    path.path_id,
                    round_index,
                )
                selected_index = by_path[path.path_id][selected_label]
                query_index, point_agreement, probability_one = query_metadata[
                    path.path_id
                ]

                for label, child_index in ((0, zero_index), (1, one_index)):
                    candidate_rows.append({
                        **child_rows[child_index],
                        "round": round_index,
                        "queried_index": query_index,
                        "parent_query_point_agreement": point_agreement,
                        "parent_query_probability_one": probability_one,
                        "selected": label == selected_label,
                        "selection_validity": validity,
                    })

                path.train_indices.append(query_index)
                path.train_labels.append(selected_label)
                path.queried_indices.append(query_index)
                path.selected_labels.append(selected_label)
                next_predictions[path_index] = child_predictions[selected_index]
                next_fit[path_index] = child_fit[selected_index]
                selected_row = {
                    **child_rows[selected_index],
                    "round": round_index,
                    "queried_index": query_index,
                    "selected_label": selected_label,
                    "parent_query_point_agreement": point_agreement,
                    "parent_query_probability_one": probability_one,
                    "selection_validity": validity,
                }
                trajectory_rows.append(selected_row)
                round_selected.append(selected_row)

            current_predictions = next_predictions
            current_fit = next_fit
            write_csv(candidates_path, candidate_rows)
            write_csv(trajectory_path, trajectory_rows)
            checkpoint_paths(
                output_dir,
                round_index,
                paths,
                current_predictions,
                current_fit,
            )

            for policy in Config.POLICIES:
                local = [row for row in round_selected if row["policy"] == policy]
                print(
                    f"  {policy:>16} | "
                    f"agreement={np.mean([float(r['unseen_pairwise_agreement']) for r in local]):.4f} | "
                    f"entropy={np.mean([float(r['unseen_vote_entropy_bits']) for r in local]):.4f} | "
                    f"fit={np.mean([float(r['train_fit_rate']) for r in local]):.3f}",
                    flush=True,
                )

        print("\n=== fresh-seed final audit ===", flush=True)
        audit_specs = path_specs_for_audit(paths)
        audit_rows, audit_predictions, audit_fit = train_and_evaluate(
            "fresh_audit",
            audit_specs,
            Config.AUDIT_SEED_COUNT,
            Config.AUDIT_INITIALIZATION_SEED,
            Config.AUDIT_MAX_STEPS,
            Config.AUDIT_LEARNING_RATE,
            Config.AUDIT_WEIGHT_DECAY,
            device,
        )
        audit_output: list[dict[str, Any]] = []
        representative_bits = np.empty(
            (len(paths), 2 ** Config.INPUT_BITS), dtype=np.uint8
        )
        for index, (path, row) in enumerate(zip(paths, audit_rows)):
            _, consensus, _ = summarize_prediction_cohort(
                audit_predictions[index], audit_fit[index], path.train_indices
            )
            representative_bits[index] = consensus
            audit_output.append({
                **row,
                "train_indices": path.train_indices,
                "train_labels": path.train_labels,
                "queried_indices": path.queried_indices,
                "selected_labels": path.selected_labels,
                **symbolic_metrics(consensus),
            })
        write_csv(output_dir / "fresh_audit_summary.csv", audit_output)
        np.savez_compressed(
            output_dir / "fresh_audit_consensus_functions.npz",
            path_ids=np.asarray([path.path_id for path in paths], dtype=np.int64),
            consensus_bits=representative_bits,
        )
        save_plot(output_dir, trajectory_rows)

        summary = {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - start_time,
            "path_count": len(paths),
            "final_train_count": Config.INITIAL_TRAIN_COUNT + Config.ROUNDS,
            "policy_summary": {
                policy: {
                    "mean_unseen_pairwise_agreement": float(np.mean([
                        float(row["unseen_pairwise_agreement"])
                        for row in audit_output if row["policy"] == policy
                    ])),
                    "mean_unseen_vote_entropy_bits": float(np.mean([
                        float(row["unseen_vote_entropy_bits"])
                        for row in audit_output if row["policy"] == policy
                    ])),
                    "unique_consensus_functions": len({
                        row["consensus_fingerprint"]
                        for row in audit_output if row["policy"] == policy
                    }),
                    "named_symbolic_count": sum(
                        bool(row["named_symbolic_family"])
                        for row in audit_output if row["policy"] == policy
                    ),
                }
                for policy in Config.POLICIES
            },
            "paired_policy_comparisons": paired_policy_comparisons(
                audit_output
            ),
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except KeyboardInterrupt:
        write_json(output_dir / "interrupted.json", {
            "status": "interrupted",
            "elapsed_seconds": time.perf_counter() - start_time,
            "message": "已保存到最近完成的 completion round，可直接重跑恢复。",
        })
        print("收到中断，已保留最近完整 round。", flush=True)
    finally:
        if Config.PACKAGE_RESULTS:
            archive = package_results(output_dir)
            print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    frontier_main()
