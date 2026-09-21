"""8-bit条件规则决斗：先检验4-bit Neural K差距最大的规则对。

输入由前后两个4-bit块组成。对规则f和g，训练集穷举且只保留：

    f(front) == g(back)

的全部组合；冲突测试集则只保留二者输出不同的组合。六条候选规则均严格8/8
平衡，因此每场决斗自动得到128个训练样本、128个冲突样本、64/64训练标签，
并且每个front/back状态在训练和冲突集中都恰好出现8次。

默认运行六条规则全部15个无序对，每对同时交换前后位置，共30个条件。模型为
与原4-bit测量尽量匹配的 ``8 -> 16 -> 1 tanh``。256个Gaussian配对初始化
使用full-batch Adam、无weight decay，并在相同训练BCE首次到达时保存冲突集
偏好、纯规则质量和完整函数指纹。

这一步测真实优化器决斗，不使用NNGP，也不把训练步数相同误当成loss状态相同。
"""

from __future__ import annotations

import argparse
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
    # AutoDL/Jupyter直接运行时默认执行完整15场循环赛，无需命令行参数。
    PROTOCOL_VERSION = "8bit_conditional_rule_duel_full_tournament_v1"
    DUEL_PANEL = "full_tournament"
    INPUT_BITS_PER_RULE = 4
    TOTAL_INPUT_BITS = 8
    WIDTH = 16
    SEEDS = 256

    LEARNING_RATE = 1e-2
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 30_000
    EVAL_EVERY = 100
    CHECKPOINT_EVERY = 1_000
    MATCHED_LOSS_THRESHOLDS = (0.1, 0.03, 0.01, 0.003, 0.001)
    PLATEAU_MIN_STEP = 10_000
    PLATEAU_WINDOW_STEPS = 10_000
    PLATEAU_RELATIVE_IMPROVEMENT = 0.01

    INITIALIZATION_SEED = 2026090501
    BOOTSTRAP_SEED = 2026090502
    BOOTSTRAP_REPLICATES = 2_000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_conditional_rule_duel_full_tournament"
    )
    CONDITION_START = 0
    CONDITION_STOP: int | None = None
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


FULL_TOURNAMENT_RESULT_DIR = Path(
    "/root/autodl-tmp/results_8bit_conditional_rule_duel_full_tournament"
)


# 来自4->16->1 tanh、Gaussian参考测度、epsilon=0.01913的深尾结果。
# 数值为自然对数单位的 -log V；只用预注册排序和差距，不在本实验中重新拟合。
FOUR_BIT_NEURAL_K_NATS = {
    "parity1": 82.54,
    "majority3": 152.57,
    "mux3": 271.93,
    "parity2": 320.74,
    "parity3": 401.16,
    "parity4": 661.85,
}


HIGH_GAP_DUELS = (
    ("parity1", "parity4"),
    ("majority3", "parity4"),
    ("mux3", "parity4"),
    ("parity1", "parity3"),
    ("majority3", "parity2"),
)


RULE_ORDER_BY_NEURAL_K = tuple(
    sorted(FOUR_BIT_NEURAL_K_NATS, key=FOUR_BIT_NEURAL_K_NATS.get)
)


FULL_TOURNAMENT_DUELS = tuple(
    (simple, hard)
    for simple_index, simple in enumerate(RULE_ORDER_BY_NEURAL_K)
    for hard in RULE_ORDER_BY_NEURAL_K[simple_index + 1:]
)


FINISHED_CONDITION_STATUSES = {
    "completed_all_thresholds",
    "stopped_plateau",
    "stopped_max_steps",
}


@dataclass(frozen=True)
class RuleSpec:
    name: str
    formula: str
    outputs: tuple[int, ...]
    neural_k_nats: float


@dataclass(frozen=True)
class DuelSpec:
    duel_index: int
    simple_rule: str
    hard_rule: str
    neural_k_gap_nats: float


@dataclass(frozen=True)
class ConditionSpec:
    condition_index: int
    duel_index: int
    name: str
    simple_rule: str
    hard_rule: str
    front_rule: str
    back_rule: str
    simple_side: str
    neural_k_gap_nats: float


@dataclass
class DuelDataset:
    all_inputs: torch.Tensor
    train_inputs: torch.Tensor
    conflict_inputs: torch.Tensor
    train_labels: torch.Tensor
    front_labels_all: torch.Tensor
    back_labels_all: torch.Tensor
    front_labels_conflict: torch.Tensor
    back_labels_conflict: torch.Tensor
    agreement_mask: np.ndarray
    conflict_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--condition-start", type=int, default=None)
    parser.add_argument("--condition-stop", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--full-tournament",
        action="store_true",
        help="运行六条规则的全部15场两两决斗，而不是五场高差距pilot。",
    )
    args, unknown = parser.parse_known_args()
    ignored: list[str] = []
    remaining = list(unknown)
    while len(remaining) >= 2 and remaining[0] == "-f":
        ignored.extend(remaining[:2])
        remaining = remaining[2:]
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if ignored:
        print(
            "忽略Jupyter内核附加的命令行参数：" + " ".join(ignored),
            flush=True,
        )
    return args


def apply_cli(args: argparse.Namespace) -> None:
    if args.full_tournament:
        Config.DUEL_PANEL = "full_tournament"
        Config.PROTOCOL_VERSION = (
            "8bit_conditional_rule_duel_full_tournament_v1"
        )
        if args.result_dir is None:
            Config.RESULT_DIR = FULL_TOURNAMENT_RESULT_DIR
    if args.result_dir is not None:
        Config.RESULT_DIR = args.result_dir
    if args.device is not None:
        Config.DEVICE = args.device
    if args.condition_start is not None:
        Config.CONDITION_START = args.condition_start
    if args.condition_stop is not None:
        Config.CONDITION_STOP = args.condition_stop
    if args.overwrite:
        Config.OVERWRITE_RESULT_DIR = True
        Config.RESUME = False
    if args.no_package:
        Config.PACKAGE_RESULTS = False
    if args.smoke:
        Config.SMOKE_TEST = True


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.WIDTH = 8
    Config.SEEDS = 8
    Config.LEARNING_RATE = 2e-2
    Config.MAX_STEPS = 400
    Config.EVAL_EVERY = 10
    Config.CHECKPOINT_EVERY = 100
    Config.MATCHED_LOSS_THRESHOLDS = (0.3, 0.1)
    Config.PLATEAU_MIN_STEP = 10_000
    Config.BOOTSTRAP_REPLICATES = 100
    Config.CONDITION_START = 0
    Config.CONDITION_STOP = 2
    Config.DEVICE = "cpu"
    smoke_name = (
        "_smoke_8bit_conditional_rule_duel_full_tournament"
        if Config.DUEL_PANEL == "full_tournament"
        else "_smoke_8bit_conditional_rule_duel_high_gap"
    )
    Config.RESULT_DIR = Path(
        "research/function_information_conservation"
    ) / smoke_name
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    }


def immutable_config_keys() -> tuple[str, ...]:
    return (
        "PROTOCOL_VERSION",
        "INPUT_BITS_PER_RULE",
        "TOTAL_INPUT_BITS",
        "WIDTH",
        "SEEDS",
        "LEARNING_RATE",
        "ADAM_BETAS",
        "ADAM_EPS",
        "WEIGHT_DECAY",
        "MAX_STEPS",
        "EVAL_EVERY",
        "MATCHED_LOSS_THRESHOLDS",
        "INITIALIZATION_SEED",
    )


def validate_config() -> None:
    if Config.TOTAL_INPUT_BITS != 2 * Config.INPUT_BITS_PER_RULE:
        raise ValueError("总输入bit必须等于两个规则块之和。")
    thresholds = tuple(float(value) for value in Config.MATCHED_LOSS_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("MATCHED_LOSS_THRESHOLDS必须严格递减且不重复。")
    if Config.SEEDS < 2:
        raise ValueError("至少需要两个seed。")
    if Config.EVAL_EVERY <= 0 or Config.MAX_STEPS < Config.EVAL_EVERY:
        raise ValueError("非法训练步数配置。")
    if Config.WEIGHT_DECAY != 0.0:
        raise ValueError("本轮预注册固定不使用weight decay。")


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "conditions").mkdir(exist_ok=True)
    path = output / "config.json"
    current = config_dict()
    if path.exists() and Config.RESUME:
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key in immutable_config_keys():
            if saved.get(key) != current.get(key):
                raise RuntimeError(f"已有结果配置与当前不一致：{key}")
    else:
        write_json(path, current)
    return output


def four_bit_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def build_rules() -> dict[str, RuleSpec]:
    inputs = four_bit_inputs()
    outputs: dict[str, tuple[np.ndarray, str]] = {
        "parity1": (inputs[:, 0], "x0"),
        "parity2": (
            np.bitwise_xor.reduce(inputs[:, :2], axis=1),
            "x0 XOR x1",
        ),
        "parity3": (
            np.bitwise_xor.reduce(inputs[:, :3], axis=1),
            "x0 XOR x1 XOR x2",
        ),
        "parity4": (
            np.bitwise_xor.reduce(inputs[:, :4], axis=1),
            "x0 XOR x1 XOR x2 XOR x3",
        ),
        "majority3": (
            (inputs[:, :3].sum(axis=1) >= 2).astype(np.uint8),
            "MAJ(x0,x1,x2)",
        ),
        "mux3": (
            np.where(inputs[:, 0] == 1, inputs[:, 1], inputs[:, 2]),
            "IF x0 THEN x1 ELSE x2",
        ),
    }
    result: dict[str, RuleSpec] = {}
    for name, (values, formula) in outputs.items():
        values = np.asarray(values, dtype=np.uint8)
        if int(values.sum()) != 8:
            raise AssertionError(f"规则{name}不是8/8平衡。")
        result[name] = RuleSpec(
            name=name,
            formula=formula,
            outputs=tuple(map(int, values)),
            neural_k_nats=float(FOUR_BIT_NEURAL_K_NATS[name]),
        )
    return result


def build_duels() -> list[DuelSpec]:
    if Config.DUEL_PANEL == "high_gap":
        duel_pairs = HIGH_GAP_DUELS
    elif Config.DUEL_PANEL == "full_tournament":
        duel_pairs = FULL_TOURNAMENT_DUELS
    else:
        raise ValueError(f"未知决斗面板：{Config.DUEL_PANEL}")
    result = []
    for index, (simple, hard) in enumerate(duel_pairs):
        simple_k = FOUR_BIT_NEURAL_K_NATS[simple]
        hard_k = FOUR_BIT_NEURAL_K_NATS[hard]
        if not simple_k < hard_k:
            raise AssertionError(f"决斗顺序错误：{simple}, {hard}")
        result.append(DuelSpec(
            duel_index=index,
            simple_rule=simple,
            hard_rule=hard,
            neural_k_gap_nats=float(hard_k - simple_k),
        ))
    return result


def build_conditions() -> list[ConditionSpec]:
    result: list[ConditionSpec] = []
    for duel in build_duels():
        for simple_side in ("front", "back"):
            if simple_side == "front":
                front, back = duel.simple_rule, duel.hard_rule
            else:
                front, back = duel.hard_rule, duel.simple_rule
            result.append(ConditionSpec(
                condition_index=len(result),
                duel_index=duel.duel_index,
                name=(
                    f"duel_{duel.duel_index:02d}_{duel.simple_rule}_vs_"
                    f"{duel.hard_rule}__simple_{simple_side}"
                ),
                simple_rule=duel.simple_rule,
                hard_rule=duel.hard_rule,
                front_rule=front,
                back_rule=back,
                simple_side=simple_side,
                neural_k_gap_nats=duel.neural_k_gap_nats,
            ))
    return result


def build_dataset(
    condition: ConditionSpec,
    rules: dict[str, RuleSpec],
    device: torch.device,
) -> DuelDataset:
    block = four_bit_inputs()
    front_index = np.repeat(np.arange(16), 16)
    back_index = np.tile(np.arange(16), 16)
    all_inputs = np.concatenate([
        block[front_index], block[back_index]
    ], axis=1).astype(np.float32)
    front_outputs = np.asarray(
        rules[condition.front_rule].outputs, dtype=np.uint8
    )[front_index]
    back_outputs = np.asarray(
        rules[condition.back_rule].outputs, dtype=np.uint8
    )[back_index]
    agreement = front_outputs == back_outputs
    conflict = ~agreement
    if int(agreement.sum()) != 128 or int(conflict.sum()) != 128:
        raise AssertionError("平衡规则决斗没有形成128/128划分。")
    train_labels = front_outputs[agreement]
    if int(train_labels.sum()) != 64:
        raise AssertionError("训练标签不是64/64平衡。")
    if not (
        np.all(np.bincount(front_index[agreement], minlength=16) == 8)
        and np.all(np.bincount(back_index[agreement], minlength=16) == 8)
        and np.all(np.bincount(front_index[conflict], minlength=16) == 8)
        and np.all(np.bincount(back_index[conflict], minlength=16) == 8)
    ):
        raise AssertionError("front/back状态边际覆盖不是严格均匀。")
    return DuelDataset(
        all_inputs=torch.as_tensor(all_inputs, device=device),
        train_inputs=torch.as_tensor(all_inputs[agreement], device=device),
        conflict_inputs=torch.as_tensor(all_inputs[conflict], device=device),
        train_labels=torch.as_tensor(
            train_labels.astype(np.float32), device=device
        ),
        front_labels_all=torch.as_tensor(
            front_outputs.astype(np.uint8), device=device
        ),
        back_labels_all=torch.as_tensor(
            back_outputs.astype(np.uint8), device=device
        ),
        front_labels_conflict=torch.as_tensor(
            front_outputs[conflict].astype(np.uint8), device=device
        ),
        back_labels_conflict=torch.as_tensor(
            back_outputs[conflict].astype(np.uint8), device=device
        ),
        agreement_mask=agreement,
        conflict_mask=conflict,
    )


class BatchedTanhMLP(nn.Module):
    def __init__(self, seeds: int, width: int, device: torch.device):
        super().__init__()
        generator = torch.Generator(device=device)
        generator.manual_seed(Config.INITIALIZATION_SEED)
        self.first_weight = nn.Parameter(torch.randn(
            seeds, width, Config.TOTAL_INPUT_BITS,
            generator=generator, device=device,
        ))
        self.first_bias = nn.Parameter(torch.randn(
            seeds, width, generator=generator, device=device,
        ))
        self.output_weight = nn.Parameter(torch.randn(
            seeds, 1, width, generator=generator, device=device,
        ))
        self.output_bias = nn.Parameter(torch.randn(
            seeds, 1, generator=generator, device=device,
        ))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seeds = self.first_weight.shape[0]
        expanded = inputs[None].expand(seeds, -1, -1)
        hidden = torch.tanh(
            torch.bmm(
                expanded,
                self.first_weight.transpose(1, 2),
            ) / math.sqrt(Config.TOTAL_INPUT_BITS)
            + self.first_bias[:, None] / math.sqrt(Config.TOTAL_INPUT_BITS)
        )
        return (
            torch.bmm(hidden, self.output_weight.transpose(1, 2)).squeeze(-1)
            / math.sqrt(Config.WIDTH)
            + self.output_bias / math.sqrt(Config.WIDTH)
        )


def parameter_count() -> int:
    return (
        Config.WIDTH * Config.TOTAL_INPUT_BITS
        + Config.WIDTH
        + Config.WIDTH
        + 1
    )


@torch.no_grad()
def evaluate_models(
    model: BatchedTanhMLP,
    dataset: DuelDataset,
    condition: ConditionSpec,
) -> dict[str, Any]:
    train_logits = model(dataset.train_inputs)
    train_targets = dataset.train_labels[None].expand_as(train_logits)
    train_losses = F.binary_cross_entropy_with_logits(
        train_logits, train_targets, reduction="none"
    ).mean(dim=1)
    train_predictions = train_logits >= 0
    train_accuracy = (
        train_predictions == train_targets.bool()
    ).float().mean(dim=1)

    all_logits = model(dataset.all_inputs)
    all_predictions = all_logits >= 0
    conflict_predictions = all_predictions[:, torch.as_tensor(
        dataset.conflict_mask, device=all_logits.device
    )]
    front_conflict = dataset.front_labels_conflict.bool()[None]
    back_conflict = dataset.back_labels_conflict.bool()[None]
    front_match = (
        conflict_predictions == front_conflict
    ).float().mean(dim=1)
    back_match = (
        conflict_predictions == back_conflict
    ).float().mean(dim=1)
    if not bool(torch.allclose(
        front_match + back_match,
        torch.ones_like(front_match),
        atol=1e-6,
    )):
        raise AssertionError("冲突集预测没有唯一支持一条规则。")

    probabilities = torch.sigmoid(
        all_logits[:, torch.as_tensor(
            dataset.conflict_mask, device=all_logits.device
        )]
    )
    front_probability = torch.where(
        front_conflict, probabilities, 1.0 - probabilities
    ).mean(dim=1)
    back_probability = 1.0 - front_probability

    pure_front = torch.all(
        all_predictions == dataset.front_labels_all.bool()[None], dim=1
    )
    pure_back = torch.all(
        all_predictions == dataset.back_labels_all.bool()[None], dim=1
    )
    if condition.simple_side == "front":
        simple_match = front_match
        simple_probability = front_probability
        pure_simple = pure_front
        pure_hard = pure_back
    else:
        simple_match = back_match
        simple_probability = back_probability
        pure_simple = pure_back
        pure_hard = pure_front

    packed = np.packbits(
        all_predictions.to(torch.uint8).cpu().numpy(),
        axis=1,
        bitorder="little",
    )
    _, counts = np.unique(packed, axis=0, return_counts=True)
    denominator = Config.SEEDS * (Config.SEEDS - 1)
    collision = (
        float(np.sum(counts * (counts - 1)) / denominator)
        if denominator else 1.0
    )
    return {
        "train_loss": train_losses.cpu().numpy(),
        "train_accuracy": train_accuracy.cpu().numpy(),
        "front_match": front_match.cpu().numpy(),
        "back_match": back_match.cpu().numpy(),
        "simple_match": simple_match.cpu().numpy(),
        "simple_probability": simple_probability.cpu().numpy(),
        "simple_preference_hard": (2.0 * simple_match - 1.0).cpu().numpy(),
        "simple_preference_soft": (
            2.0 * simple_probability - 1.0
        ).cpu().numpy(),
        "pure_simple": pure_simple.cpu().numpy(),
        "pure_hard": pure_hard.cpu().numpy(),
        "full_function_packed": packed,
        "distinct_functions": int(len(counts)),
        "function_collision": collision,
    }


def function_hex(packed: np.ndarray) -> str:
    return "0x" + bytes(np.asarray(packed, dtype=np.uint8)[::-1]).hex().upper()


def condition_summary_path(output_dir: Path, condition: ConditionSpec) -> Path:
    return output_dir / "conditions" / condition.name / "summary.json"


def condition_is_finished(output_dir: Path, condition: ConditionSpec) -> bool:
    path = condition_summary_path(output_dir, condition)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("status")) in FINISHED_CONDITION_STATUSES


def save_checkpoint(
    path: Path,
    model: BatchedTanhMLP,
    optimizer: torch.optim.Optimizer,
    step: int,
    reached: torch.Tensor,
    crossing_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    median_history: Sequence[float],
    last_new_crossing_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "protocol": Config.PROTOCOL_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "reached": reached.cpu(),
        "crossing_rows": list(crossing_rows),
        "curve_rows": list(curve_rows),
        "median_history": list(median_history),
        "last_new_crossing_step": last_new_crossing_step,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: BatchedTanhMLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]], list[float], int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("protocol") != Config.PROTOCOL_VERSION:
        raise RuntimeError(f"checkpoint协议不匹配：{path}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["step"]),
        payload["reached"].to(device),
        list(payload["crossing_rows"]),
        list(payload["curve_rows"]),
        [float(value) for value in payload["median_history"]],
        int(payload["last_new_crossing_step"]),
    )


def record_crossings(
    condition: ConditionSpec,
    step: int,
    metrics: dict[str, Any],
    reached: torch.Tensor,
    rows: list[dict[str, Any]],
) -> int:
    new_count = 0
    losses = metrics["train_loss"]
    for threshold_index, threshold in enumerate(
        Config.MATCHED_LOSS_THRESHOLDS
    ):
        eligible = np.flatnonzero(
            (~reached[threshold_index].cpu().numpy())
            & (losses <= threshold)
        )
        if not len(eligible):
            continue
        reached[threshold_index, torch.as_tensor(
            eligible, device=reached.device
        )] = True
        new_count += len(eligible)
        for seed in eligible:
            rows.append({
                "condition_index": condition.condition_index,
                "condition_name": condition.name,
                "duel_index": condition.duel_index,
                "simple_rule": condition.simple_rule,
                "hard_rule": condition.hard_rule,
                "simple_side": condition.simple_side,
                "neural_k_gap_nats": condition.neural_k_gap_nats,
                "seed": int(seed),
                "matched_loss_threshold": float(threshold),
                "crossing_step": step,
                "train_loss": float(losses[seed]),
                "train_accuracy": float(metrics["train_accuracy"][seed]),
                "front_match_fraction": float(metrics["front_match"][seed]),
                "back_match_fraction": float(metrics["back_match"][seed]),
                "simple_match_fraction": float(metrics["simple_match"][seed]),
                "simple_probability": float(metrics["simple_probability"][seed]),
                "simple_preference_hard": float(
                    metrics["simple_preference_hard"][seed]
                ),
                "simple_preference_soft": float(
                    metrics["simple_preference_soft"][seed]
                ),
                "pure_simple_rule": bool(metrics["pure_simple"][seed]),
                "pure_hard_rule": bool(metrics["pure_hard"][seed]),
                "hybrid_function": bool(
                    not metrics["pure_simple"][seed]
                    and not metrics["pure_hard"][seed]
                ),
                "full_function_hex": function_hex(
                    metrics["full_function_packed"][seed]
                ),
            })
    return new_count


def curve_row(
    condition: ConditionSpec,
    step: int,
    metrics: dict[str, Any],
    reached: torch.Tensor,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "condition_index": condition.condition_index,
        "condition_name": condition.name,
        "duel_index": condition.duel_index,
        "simple_rule": condition.simple_rule,
        "hard_rule": condition.hard_rule,
        "simple_side": condition.simple_side,
        "neural_k_gap_nats": condition.neural_k_gap_nats,
        "step": step,
        "train_loss_mean": float(np.mean(metrics["train_loss"])),
        "train_loss_median": float(np.median(metrics["train_loss"])),
        "train_loss_max": float(np.max(metrics["train_loss"])),
        "train_accuracy_mean": float(np.mean(metrics["train_accuracy"])),
        "train_exact_fit_fraction": float(np.mean(
            metrics["train_accuracy"] == 1.0
        )),
        "simple_match_mean": float(np.mean(metrics["simple_match"])),
        "simple_match_median": float(np.median(metrics["simple_match"])),
        "simple_preference_hard_mean": float(np.mean(
            metrics["simple_preference_hard"]
        )),
        "simple_preference_soft_mean": float(np.mean(
            metrics["simple_preference_soft"]
        )),
        "seed_simple_majority_fraction": float(np.mean(
            metrics["simple_preference_hard"] > 0
        )),
        "seed_hard_majority_fraction": float(np.mean(
            metrics["simple_preference_hard"] < 0
        )),
        "pure_simple_fraction": float(np.mean(metrics["pure_simple"])),
        "pure_hard_fraction": float(np.mean(metrics["pure_hard"])),
        "hybrid_fraction": float(np.mean(
            ~(metrics["pure_simple"] | metrics["pure_hard"])
        )),
        "distinct_complete_functions": metrics["distinct_functions"],
        "complete_function_collision": metrics["function_collision"],
        "reached_counts_json": [
            int(value) for value in reached.sum(dim=1).cpu().tolist()
        ],
        "elapsed_seconds": elapsed,
    }


def run_condition(
    output_dir: Path,
    condition: ConditionSpec,
    rules: dict[str, RuleSpec],
    device: torch.device,
) -> dict[str, Any]:
    condition_dir = output_dir / "conditions" / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    if Config.RESUME and condition_is_finished(output_dir, condition):
        print(f"[{condition.name}] 已完成，跳过。", flush=True)
        return json.loads(
            condition_summary_path(output_dir, condition).read_text(
                encoding="utf-8"
            )
        )
    dataset = build_dataset(condition, rules, device)
    model = BatchedTanhMLP(Config.SEEDS, Config.WIDTH, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        betas=Config.ADAM_BETAS,
        eps=Config.ADAM_EPS,
        weight_decay=Config.WEIGHT_DECAY,
    )
    reached = torch.zeros(
        len(Config.MATCHED_LOSS_THRESHOLDS), Config.SEEDS,
        dtype=torch.bool, device=device,
    )
    crossing_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    median_history: list[float] = []
    last_new_crossing_step = 0
    step = 0
    checkpoint = condition_dir / "checkpoint.pt"
    if Config.RESUME and checkpoint.exists():
        (
            step, reached, crossing_rows, curve_rows,
            median_history, last_new_crossing_step,
        ) = load_checkpoint(checkpoint, model, optimizer, device)
        print(f"[{condition.name}] 恢复step={step:,}", flush=True)

    started = time.perf_counter()
    stop_reason = "running"
    try:
        while step <= Config.MAX_STEPS:
            if step % Config.EVAL_EVERY == 0:
                metrics = evaluate_models(model, dataset, condition)
                new_count = record_crossings(
                    condition, step, metrics, reached, crossing_rows
                )
                if new_count:
                    last_new_crossing_step = step
                median_history.append(float(np.median(metrics["train_loss"])))
                elapsed = time.perf_counter() - started
                row = curve_row(
                    condition, step, metrics, reached, elapsed
                )
                curve_rows.append(row)
                reached_text = "/".join(
                    str(int(value)) for value in reached.sum(dim=1).tolist()
                )
                eta = (
                    elapsed * (Config.MAX_STEPS - step) / max(step, 1)
                    if step else None
                )
                print(
                    f"[{condition.name}] step={step:>6,} | "
                    f"Lmed={row['train_loss_median']:.3e} | "
                    f"fit={row['train_exact_fit_fraction']:.1%} | "
                    f"simple={row['simple_match_mean']:.3f} | "
                    f"pure(S/H)={row['pure_simple_fraction']:.3f}/"
                    f"{row['pure_hard_fraction']:.3f} | "
                    f"reached={reached_text} | "
                    f"ETA<={format_duration(eta)}",
                    flush=True,
                )
                write_csv(condition_dir / "training_curve.csv", curve_rows)
                write_csv(
                    condition_dir / "matched_loss_seed_records.csv",
                    crossing_rows,
                )
                if bool(torch.all(reached[-1]).item()):
                    stop_reason = "completed_all_thresholds"
                    break
                window_checks = max(
                    2, Config.PLATEAU_WINDOW_STEPS // Config.EVAL_EVERY
                )
                if (
                    step >= Config.PLATEAU_MIN_STEP
                    and step - last_new_crossing_step
                    >= Config.PLATEAU_WINDOW_STEPS
                    and len(median_history) > window_checks
                ):
                    old = median_history[-window_checks - 1]
                    new = median_history[-1]
                    relative = (old - new) / max(abs(old), 1e-12)
                    if relative < Config.PLATEAU_RELATIVE_IMPROVEMENT:
                        stop_reason = "stopped_plateau"
                        break
            if step == Config.MAX_STEPS:
                stop_reason = "stopped_max_steps"
                break

            optimizer.zero_grad(set_to_none=True)
            logits = model(dataset.train_inputs)
            targets = dataset.train_labels[None].expand_as(logits)
            per_seed = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).mean(dim=1)
            per_seed.sum().backward()
            optimizer.step()
            step += 1

            if step % Config.CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    checkpoint, model, optimizer, step, reached,
                    crossing_rows, curve_rows, median_history,
                    last_new_crossing_step,
                )
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print(f"\n[{condition.name}] 收到中断，保存checkpoint。", flush=True)
    finally:
        save_checkpoint(
            checkpoint, model, optimizer, step, reached,
            crossing_rows, curve_rows, median_history,
            last_new_crossing_step,
        )

    metrics = evaluate_models(model, dataset, condition)
    final_rows = []
    for seed in range(Config.SEEDS):
        final_rows.append({
            "condition_name": condition.name,
            "seed": seed,
            "step": step,
            "train_loss": float(metrics["train_loss"][seed]),
            "train_accuracy": float(metrics["train_accuracy"][seed]),
            "simple_match_fraction": float(metrics["simple_match"][seed]),
            "simple_probability": float(metrics["simple_probability"][seed]),
            "simple_preference_hard": float(
                metrics["simple_preference_hard"][seed]
            ),
            "simple_preference_soft": float(
                metrics["simple_preference_soft"][seed]
            ),
            "pure_simple_rule": bool(metrics["pure_simple"][seed]),
            "pure_hard_rule": bool(metrics["pure_hard"][seed]),
            "full_function_hex": function_hex(
                metrics["full_function_packed"][seed]
            ),
        })
    write_csv(condition_dir / "final_seed_metrics.csv", final_rows)
    write_csv(condition_dir / "training_curve.csv", curve_rows)
    write_csv(
        condition_dir / "matched_loss_seed_records.csv", crossing_rows
    )
    reached_counts = [
        int(value) for value in reached.sum(dim=1).cpu().tolist()
    ]
    summary = {
        "status": stop_reason,
        "condition": asdict(condition),
        "network": f"8->{Config.WIDTH}->1 tanh",
        "parameter_count": parameter_count(),
        "optimizer": "full-batch Adam without weight decay",
        "step": step,
        "matched_loss_thresholds": list(Config.MATCHED_LOSS_THRESHOLDS),
        "reached_counts": reached_counts,
        "reached_fractions": [
            value / Config.SEEDS for value in reached_counts
        ],
        "final": curve_row(
            condition, step, metrics, reached,
            time.perf_counter() - started,
        ),
        "interpretation_boundary": (
            "Conflict labels are deliberately undefined externally. A value "
            "above 0.5 means the optimizer follows the lower-4bit-Neural-K "
            "extension more often; hybrid functions remain a valid outcome."
        ),
    }
    write_json(condition_dir / "summary.json", summary)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    indices = rng.integers(
        0, len(values), size=(Config.BOOTSTRAP_REPLICATES, len(values))
    )
    means = values[indices].mean(axis=1)
    low, median, high = np.quantile(means, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """二项比例的95% Wilson区间；避免全成功时退化成[1, 1]。"""
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return float(center - half_width), float(center + half_width)


def aggregate_results(
    output_dir: Path,
    conditions: Sequence[ConditionSpec],
) -> dict[str, Any]:
    condition_rows: list[dict[str, Any]] = []
    all_crossings: list[dict[str, Any]] = []
    condition_statuses: dict[str, str] = {}
    for condition in conditions:
        condition_dir = output_dir / "conditions" / condition.name
        summary_path = condition_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            condition_statuses[condition.name] = str(summary["status"])
        for row in read_csv(condition_dir / "matched_loss_seed_records.csv"):
            typed = {
                **row,
                "duel_index": int(row["duel_index"]),
                "seed": int(row["seed"]),
                "matched_loss_threshold": float(row["matched_loss_threshold"]),
                "simple_match_fraction": float(row["simple_match_fraction"]),
                "simple_probability": float(row["simple_probability"]),
                "simple_preference_hard": float(row["simple_preference_hard"]),
                "simple_preference_soft": float(row["simple_preference_soft"]),
                "pure_simple_rule": str(row["pure_simple_rule"]).lower() == "true",
                "pure_hard_rule": str(row["pure_hard_rule"]).lower() == "true",
            }
            all_crossings.append(typed)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in all_crossings:
        grouped.setdefault((
            str(row["condition_name"]),
            float(row["matched_loss_threshold"]),
        ), []).append(row)
    rng = np.random.default_rng(Config.BOOTSTRAP_SEED)
    for condition in conditions:
        for threshold in Config.MATCHED_LOSS_THRESHOLDS:
            rows = grouped.get((condition.name, float(threshold)), [])
            values = np.asarray([
                row["simple_match_fraction"] for row in rows
            ], dtype=np.float64)
            low, median, high = bootstrap_interval(values, rng)
            signatures = [row["full_function_hex"] for row in rows]
            counts = np.asarray(
                list({value: signatures.count(value) for value in set(signatures)}.values()),
                dtype=np.int64,
            )
            denominator = len(rows) * max(len(rows) - 1, 1)
            collision = (
                float(np.sum(counts * (counts - 1)) / denominator)
                if len(rows) > 1 else 1.0 if rows else float("nan")
            )
            condition_rows.append({
                "condition_index": condition.condition_index,
                "condition_name": condition.name,
                "duel_index": condition.duel_index,
                "simple_rule": condition.simple_rule,
                "hard_rule": condition.hard_rule,
                "simple_side": condition.simple_side,
                "neural_k_gap_nats": condition.neural_k_gap_nats,
                "matched_loss_threshold": float(threshold),
                "reached_count": len(rows),
                "reached_fraction": len(rows) / Config.SEEDS,
                "simple_match_mean": float(np.mean(values)) if len(values) else None,
                "simple_match_bootstrap_low": low,
                "simple_match_bootstrap_median": median,
                "simple_match_bootstrap_high": high,
                "seed_simple_majority_fraction": (
                    float(np.mean([
                        row["simple_preference_hard"] > 0 for row in rows
                    ])) if rows else None
                ),
                "seed_hard_majority_fraction": (
                    float(np.mean([
                        row["simple_preference_hard"] < 0 for row in rows
                    ])) if rows else None
                ),
                "pure_simple_fraction": (
                    float(np.mean([row["pure_simple_rule"] for row in rows]))
                    if rows else None
                ),
                "pure_hard_fraction": (
                    float(np.mean([row["pure_hard_rule"] for row in rows]))
                    if rows else None
                ),
                "hybrid_fraction": (
                    float(np.mean([
                        not row["pure_simple_rule"] and not row["pure_hard_rule"]
                        for row in rows
                    ])) if rows else None
                ),
                "distinct_complete_functions": len(set(signatures)),
                "complete_function_collision": collision,
            })
    write_csv(output_dir / "condition_threshold_summary.csv", condition_rows)
    write_csv(output_dir / "all_matched_loss_seed_records.csv", all_crossings)

    pair_rows: list[dict[str, Any]] = []
    duels = build_duels()
    condition_by_duel_side = {
        (condition.duel_index, condition.simple_side): condition
        for condition in conditions
    }
    for duel in duels:
        front_condition = condition_by_duel_side[(duel.duel_index, "front")]
        back_condition = condition_by_duel_side[(duel.duel_index, "back")]
        for threshold in Config.MATCHED_LOSS_THRESHOLDS:
            front_rows = {
                int(row["seed"]): row for row in grouped.get(
                    (front_condition.name, float(threshold)), []
                )
            }
            back_rows = {
                int(row["seed"]): row for row in grouped.get(
                    (back_condition.name, float(threshold)), []
                )
            }
            common = sorted(set(front_rows).intersection(back_rows))
            paired = np.asarray([
                0.5 * (
                    float(front_rows[seed]["simple_match_fraction"])
                    + float(back_rows[seed]["simple_match_fraction"])
                )
                for seed in common
            ], dtype=np.float64)
            low, median, high = bootstrap_interval(paired, rng)
            simple_wins = int(np.sum(paired > 0.5))
            ties = int(np.sum(np.isclose(paired, 0.5, atol=1e-12)))
            hard_wins = int(len(paired) - simple_wins - ties)
            win_low, win_high = wilson_interval(simple_wins, len(paired))
            orientation_gap = (
                float(np.mean([
                    float(front_rows[seed]["simple_match_fraction"])
                    - float(back_rows[seed]["simple_match_fraction"])
                    for seed in common
                ])) if common else None
            )
            if len(common) < max(16, int(0.8 * Config.SEEDS)):
                verdict = "insufficient_reached_seeds"
            elif low > 0.5:
                verdict = "lower_neural_k_rule_preferred"
            elif high < 0.5:
                verdict = "higher_neural_k_rule_preferred"
            else:
                verdict = "mixed_or_inconclusive"
            pair_rows.append({
                "duel_index": duel.duel_index,
                "simple_rule": duel.simple_rule,
                "hard_rule": duel.hard_rule,
                "neural_k_gap_nats": duel.neural_k_gap_nats,
                "matched_loss_threshold": float(threshold),
                "paired_seed_count": len(common),
                "simple_match_paired_mean": (
                    float(np.mean(paired)) if len(paired) else None
                ),
                "simple_match_bootstrap_low": low,
                "simple_match_bootstrap_median": median,
                "simple_match_bootstrap_high": high,
                "paired_simple_win_count": simple_wins,
                "paired_tie_count": ties,
                "paired_hard_win_count": hard_wins,
                "paired_simple_win_fraction": (
                    simple_wins / len(paired) if len(paired) else None
                ),
                "paired_simple_win_wilson_low": win_low,
                "paired_simple_win_wilson_high": win_high,
                "orientation_gap_front_minus_back": orientation_gap,
                "verdict": verdict,
            })
    write_csv(output_dir / "paired_duel_verdicts.csv", pair_rows)
    save_plots(output_dir, pair_rows, condition_rows)
    write_report(output_dir, pair_rows)
    final_threshold = float(Config.MATCHED_LOSS_THRESHOLDS[-1])
    final_rows = [
        row for row in pair_rows
        if math.isclose(float(row["matched_loss_threshold"]), final_threshold)
    ]
    directed_rows: list[dict[str, Any]] = []
    for row in final_rows:
        value = row.get("simple_match_paired_mean")
        if value in (None, ""):
            continue
        directed_rows.extend([
            {
                "row_rule": row["simple_rule"],
                "opponent_rule": row["hard_rule"],
                "row_rule_neural_k_nats": FOUR_BIT_NEURAL_K_NATS[
                    str(row["simple_rule"])
                ],
                "opponent_neural_k_nats": FOUR_BIT_NEURAL_K_NATS[
                    str(row["hard_rule"])
                ],
                "row_rule_win_probability": float(value),
                "matched_loss_threshold": final_threshold,
            },
            {
                "row_rule": row["hard_rule"],
                "opponent_rule": row["simple_rule"],
                "row_rule_neural_k_nats": FOUR_BIT_NEURAL_K_NATS[
                    str(row["hard_rule"])
                ],
                "opponent_neural_k_nats": FOUR_BIT_NEURAL_K_NATS[
                    str(row["simple_rule"])
                ],
                "row_rule_win_probability": 1.0 - float(value),
                "matched_loss_threshold": final_threshold,
            },
        ])
    write_csv(output_dir / "directed_rule_preference_matrix.csv", directed_rows)
    return {
        "condition_statuses": condition_statuses,
        "finished_condition_count": sum(
            status in FINISHED_CONDITION_STATUSES
            for status in condition_statuses.values()
        ),
        "all_condition_count": len(conditions),
        "final_threshold": final_threshold,
        "final_duel_verdicts": final_rows,
        "lower_K_preference_count": sum(
            row["verdict"] == "lower_neural_k_rule_preferred"
            for row in final_rows
        ),
    }


def save_plots(
    output_dir: Path,
    pair_rows: Sequence[dict[str, Any]],
    condition_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    figure_size = (18, 12) if len(build_duels()) > 5 else (14, 10)
    figure, axes = plt.subplots(2, 2, figsize=figure_size)
    for duel in build_duels():
        rows = sorted([
            row for row in pair_rows if int(row["duel_index"]) == duel.duel_index
            and row["simple_match_paired_mean"] not in (None, "")
        ], key=lambda row: float(row["matched_loss_threshold"]), reverse=True)
        if not rows:
            continue
        x = [float(row["matched_loss_threshold"]) for row in rows]
        y = [float(row["simple_match_paired_mean"]) for row in rows]
        low = [float(row["simple_match_bootstrap_low"]) for row in rows]
        high = [float(row["simple_match_bootstrap_high"]) for row in rows]
        label = f"{duel.simple_rule} vs {duel.hard_rule}"
        axes[0, 0].plot(x, y, marker="o", label=label)
        axes[0, 0].fill_between(x, low, high, alpha=0.12)
    axes[0, 0].axhline(0.5, color="black", linewidth=1)
    axes[0, 0].invert_xaxis()
    axes[0, 0].set_xlabel("matched train BCE")
    axes[0, 0].set_ylabel("conflict match to lower-K rule")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(
        fontsize=7 if len(build_duels()) > 5 else 8,
        ncol=2 if len(build_duels()) > 5 else 1,
    )
    axes[0, 0].set_title("Paired rule preference across loss")

    final_threshold = float(Config.MATCHED_LOSS_THRESHOLDS[-1])
    final = [
        row for row in pair_rows
        if math.isclose(float(row["matched_loss_threshold"]), final_threshold)
    ]
    labels = [f"{row['simple_rule']}\nvs {row['hard_rule']}" for row in final]
    values = [
        float(row["simple_match_paired_mean"])
        if row["simple_match_paired_mean"] not in (None, "") else np.nan
        for row in final
    ]
    axes[0, 1].bar(range(len(final)), values, color="#2563eb")
    axes[0, 1].axhline(0.5, color="black", linewidth=1)
    axes[0, 1].set_xticks(range(len(final)), labels, rotation=25, ha="right")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title(f"Final matched loss {final_threshold:g}")

    orientations = [
        row for row in condition_rows
        if math.isclose(
            float(row["matched_loss_threshold"]), final_threshold
        ) and row["simple_match_mean"] not in (None, "")
    ]
    for side, color in (("front", "#c2410c"), ("back", "#0f766e")):
        local = [row for row in orientations if row["simple_side"] == side]
        axes[1, 0].scatter(
            [float(row["neural_k_gap_nats"]) for row in local],
            [float(row["simple_match_mean"]) for row in local],
            label=f"simple on {side}", color=color, s=55,
        )
    axes[1, 0].axhline(0.5, color="black", linewidth=1)
    axes[1, 0].set_xlabel("4-bit Neural K gap (nats)")
    axes[1, 0].set_ylabel("lower-K rule match")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].set_title("Position counterbalance")

    pure_simple = []
    pure_hard = []
    hybrid = []
    local_labels = []
    for duel in build_duels():
        rows = [
            row for row in orientations
            if int(row["duel_index"]) == duel.duel_index
        ]
        if not rows:
            continue
        pure_simple.append(float(np.mean([
            float(row["pure_simple_fraction"]) for row in rows
        ])))
        pure_hard.append(float(np.mean([
            float(row["pure_hard_fraction"]) for row in rows
        ])))
        hybrid.append(float(np.mean([
            float(row["hybrid_fraction"]) for row in rows
        ])))
        local_labels.append(f"{duel.simple_rule}/{duel.hard_rule}")
    x = np.arange(len(local_labels))
    axes[1, 1].bar(x, pure_simple, label="pure lower-K", color="#16a34a")
    axes[1, 1].bar(
        x, hybrid, bottom=pure_simple, label="hybrid", color="#a1a1aa"
    )
    axes[1, 1].bar(
        x, pure_hard,
        bottom=np.asarray(pure_simple) + np.asarray(hybrid),
        label="pure higher-K", color="#dc2626",
    )
    axes[1, 1].set_xticks(x, local_labels, rotation=25, ha="right")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Pure rules versus hybrid functions")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "conditional_rule_duel.png", dpi=180)
    plt.close(figure)

    if Config.DUEL_PANEL != "full_tournament":
        return
    order = list(RULE_ORDER_BY_NEURAL_K)
    matrix = np.full((len(order), len(order)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 0.5)
    for row in final:
        value = row.get("simple_match_paired_mean")
        if value in (None, ""):
            continue
        simple_index = order.index(str(row["simple_rule"]))
        hard_index = order.index(str(row["hard_rule"]))
        matrix[simple_index, hard_index] = float(value)
        matrix[hard_index, simple_index] = 1.0 - float(value)
    matrix_figure, matrix_axis = plt.subplots(figsize=(9, 8))
    image = matrix_axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    matrix_axis.set_xticks(range(len(order)), order, rotation=35, ha="right")
    matrix_axis.set_yticks(range(len(order)), order)
    matrix_axis.set_xlabel("opponent rule")
    matrix_axis.set_ylabel("row rule")
    matrix_axis.set_title(
        f"Probability that row rule wins at BCE={final_threshold:g}"
    )
    for row_index in range(len(order)):
        for column_index in range(len(order)):
            value = matrix[row_index, column_index]
            if math.isfinite(value):
                matrix_axis.text(
                    column_index, row_index, f"{value:.3f}",
                    ha="center", va="center",
                    color="white" if value < 0.2 or value > 0.8 else "black",
                    fontsize=9,
                )
    matrix_figure.colorbar(image, ax=matrix_axis, label="row-rule preference")
    matrix_figure.tight_layout()
    matrix_figure.savefig(
        output_dir / "full_tournament_preference_matrix.png", dpi=180
    )
    plt.close(matrix_figure)


def write_report(
    output_dir: Path,
    pair_rows: Sequence[dict[str, Any]],
) -> None:
    final_threshold = float(Config.MATCHED_LOSS_THRESHOLDS[-1])
    final = [
        row for row in pair_rows
        if math.isclose(float(row["matched_loss_threshold"]), final_threshold)
    ]
    lines = [
        "# 8-bit 条件规则决斗结果",
        "",
        "训练集只包含两条4-bit规则输出一致的128个组合；测试集只包含输出冲突的",
        "另外128个组合。所有结果均已对调前后位置并按seed配对。",
        "",
        f"## 最深匹配训练loss：{final_threshold:g}",
        "",
        "| 低Neural K规则 | 高Neural K规则 | 4-bit差距 | 配对seed | "
        "低K匹配率 | 低K胜出seed | Wilson 95%区间 | 判决 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in final:
        mean = row.get("simple_match_paired_mean")
        win_fraction = row.get("paired_simple_win_fraction")
        win_low = row.get("paired_simple_win_wilson_low")
        win_high = row.get("paired_simple_win_wilson_high")
        lines.append(
            f"| {row['simple_rule']} | {row['hard_rule']} | "
            f"{float(row['neural_k_gap_nats']):.2f} | "
            f"{row['paired_seed_count']} | "
            f"{float(mean):.3f} | {float(win_fraction):.3f} | "
            f"[{float(win_low):.3f}, {float(win_high):.3f}] | "
            f"{row['verdict']} |"
            if mean not in (None, "") else
            f"| {row['simple_rule']} | {row['hard_rule']} | "
            f"{float(row['neural_k_gap_nats']):.2f} | 0 | n/a | n/a | n/a | "
            "insufficient |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "冲突集没有外部真值标签；这里测量的是训练集无法区分两条规则时，优化器",
        "更常沿用哪一个完整扩展。低K匹配率高于0.5支持4-bit绝对复杂度在直接",
        "竞争中的预测作用；混合函数占优、位置效应或反向选择都是有效反结果。",
    ])
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.name not in {
                "checkpoint.pt", "checkpoint.tmp"
            }:
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def main() -> None:
    args = parse_args()
    apply_cli(args)
    apply_smoke_overrides()
    validate_config()
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    rules = build_rules()
    duels = build_duels()
    conditions = build_conditions()
    plan = {
        "protocol": Config.PROTOCOL_VERSION,
        "created_before_training": True,
        "network": f"8->{Config.WIDTH}->1 tanh",
        "parameter_count": parameter_count(),
        "optimizer": "full-batch Adam without weight decay",
        "rules": [asdict(rule) for rule in rules.values()],
        "duels": [asdict(duel) for duel in duels],
        "conditions": [asdict(condition) for condition in conditions],
        "training_set": (
            "all 8-bit inputs where front-rule output equals back-rule output"
        ),
        "conflict_set": (
            "all remaining inputs where the two rule outputs disagree"
        ),
        "matched_loss_thresholds": list(Config.MATCHED_LOSS_THRESHOLDS),
        "primary_decision": (
            "For each selected duel, the paired bootstrap interval of conflict "
            "match to the lower-4bit-Neural-K rule is above 0.5."
        ),
    }
    canonical = json.dumps(
        json_ready(plan), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    plan["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    plan_path = output_dir / "preregistered_duel_plan.json"
    if plan_path.exists() and Config.RESUME:
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != json_ready(plan):
            raise RuntimeError("已有预注册决斗面板与当前脚本不一致。")
    else:
        write_json(plan_path, plan)
    write_csv(output_dir / "duel_conditions.csv", [
        asdict(condition) for condition in conditions
    ])

    start = max(0, int(Config.CONDITION_START))
    stop = len(conditions) if Config.CONDITION_STOP is None else min(
        len(conditions), int(Config.CONDITION_STOP)
    )
    if not 0 <= start <= stop <= len(conditions):
        raise ValueError(f"非法条件切片：[{start}, {stop})")
    selected = conditions[start:stop]
    print(
        f"=== 8-bit conditional rule duel: {Config.DUEL_PANEL} ===",
        flush=True,
    )
    print(
        f"device={device} | network=8->{Config.WIDTH}->1 tanh | "
        f"params={parameter_count()} | seeds={Config.SEEDS}",
        flush=True,
    )
    print(
        f"duels={len(duels)} | orientations={len(conditions)} | "
        "train/conflict=128/128 | every 4-bit state appears 8 times",
        flush=True,
    )
    print(f"结果目录：{output_dir}", flush=True)

    started = time.perf_counter()
    summaries = []
    interrupted = False
    for local_index, condition in enumerate(selected, start=1):
        print(
            f"\n### condition {condition.condition_index + 1}/"
            f"{len(conditions)} | slice {local_index}/{len(selected)} | "
            f"{condition.name} ###",
            flush=True,
        )
        summary = run_condition(output_dir, condition, rules, device)
        summaries.append(summary)
        if summary["status"] == "interrupted":
            interrupted = True
            break

    aggregate = aggregate_results(output_dir, conditions)
    summary = {
        "status": "interrupted" if interrupted else (
            "completed" if aggregate["finished_condition_count"] == len(conditions)
            else "partial"
        ),
        "protocol": Config.PROTOCOL_VERSION,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "network": f"8->{Config.WIDTH}->1 tanh",
        "parameter_count": parameter_count(),
        "condition_slice": [start, stop],
        "plan_sha256": plan["plan_sha256"],
        "invocation_summaries": summaries,
        **aggregate,
    }
    write_json(output_dir / "summary.json", summary)
    archive = (
        create_archive(output_dir)
        if Config.PACKAGE_RESULTS and not interrupted else None
    )
    print("\n=== 当前汇总 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
