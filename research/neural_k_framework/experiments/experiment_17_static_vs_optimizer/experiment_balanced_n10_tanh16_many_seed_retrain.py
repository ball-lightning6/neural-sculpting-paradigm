"""均衡 n=10、tanh16x2 的大规模多 seed 真实训练复核。

严格匹配 constrained SMC 的网络、初始化测度、训练集和 raw BCE，只把
静态条件采样换成 full-batch AdamW 训练。用32,768个独立初始化长期训练，
直接检验真实优化最终选择 AND，还是 SMC 深尾中的 D440/F040/D040/F440。
"""

from __future__ import annotations

import csv
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
    WIDTH = 16
    HIDDEN_LAYERS = 2
    # 与原始 n=10 相比只做一次样本替换：1101 -> 1000。
    ACTIVE_CONDITIONS = ("balanced_minimal_n10",)
    SEED_COUNT = 32_768
    INITIALIZATION_SEED = 20261001

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 50_000
    EARLY_EVAL_STEPS = (
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    )
    EVAL_INTERVAL_STEPS = 500
    SAVE_INTERVAL_STEPS = 5_000
    SAVE_INTERVAL_SECONDS = 120.0
    RESUME = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path(
        "/root/results_balanced_n10_tanh16_many_seed_retrain"
    )
    PACKAGE_RESULTS = True
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass(frozen=True)
class TrainCondition:
    name: str
    train_indices: tuple[int, ...]


def build_conditions() -> list[TrainCondition]:
    all_conditions = [
        TrainCondition(
            "original_gap_n10",
            tuple(sorted((1, 2, 3, 5, 7, 11, 12, 13, 14, 15))),
        ),
        TrainCondition(
            "balanced_minimal_n10",
            # 保留原训练集中的9个样本；补入1000并移除1101。
            # (x1, x2)=00/01/10/11 的覆盖数分别为3/2/2/3。
            tuple(sorted((1, 2, 3, 5, 7, 8, 11, 12, 14, 15))),
        ),
        TrainCondition(
            "close_x1_branch_n10",
            tuple(sorted((1, 2, 3, 8, 10, 11, 12, 13, 14, 15))),
        ),
        TrainCondition(
            "close_both_branches_n10",
            tuple(sorted((1, 4, 6, 8, 10, 11, 12, 13, 14, 15))),
        ),
        TrainCondition(
            "original_gap_n12",
            tuple(sorted((1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15))),
        ),
    ]
    by_name = {condition.name: condition for condition in all_conditions}
    unknown = sorted(set(Config.ACTIVE_CONDITIONS) - set(by_name))
    if unknown:
        raise ValueError(f"未知 ACTIVE_CONDITIONS：{unknown}")
    return [by_name[name] for name in Config.ACTIVE_CONDITIONS]


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.ACTIVE_CONDITIONS = ("balanced_minimal_n10",)
    Config.SEED_COUNT = 8
    Config.MAX_STEPS = 2
    Config.EARLY_EVAL_STEPS = (0, 1, 2)
    Config.EVAL_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_SECONDS = 0.01
    Config.RESUME = False
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_balanced_n10_tanh16_many_seed_retrain"
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


def prepare_result_dir() -> Path:
    output = Config.RESULT_DIR
    if output.exists():
        if Config.RESUME and (output / "latest_checkpoint.pt").exists():
            return output
        if Config.OVERWRITE_RESULT_DIR:
            shutil.rmtree(output)
        else:
            output = output.parent / (
                output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def and_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs[:, 0] * inputs[:, 1]).astype(np.float32)


class BatchedIndependentMLP(nn.Module):
    def __init__(self, seed_count: int, condition_count: int) -> None:
        super().__init__()
        self.seed_count = seed_count
        self.condition_count = condition_count
        model_count = seed_count * condition_count
        dimensions = [4] + [Config.WIDTH] * Config.HIDDEN_LAYERS + [1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.INITIALIZATION_SEED)
        base_weights: list[torch.Tensor] = []
        base_biases: list[torch.Tensor] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            bound = 1.0 / math.sqrt(input_dim)
            base_weights.append(torch.empty(
                seed_count, output_dim, input_dim
            ).uniform_(-bound, bound, generator=generator))
            base_biases.append(torch.empty(
                seed_count, output_dim
            ).uniform_(-bound, bound, generator=generator))
        self.weights = nn.ParameterList([
            nn.Parameter(weight.repeat(condition_count, 1, 1))
            for weight in base_weights
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(bias.repeat(condition_count, 1))
            for bias in base_biases
        ])
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)


def build_training_tensors(
    conditions: Sequence[TrainCondition],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs_np = truth_table_inputs()
    targets_np = and_targets(inputs_np)
    max_count = max(len(condition.train_indices) for condition in conditions)
    train_x = np.zeros((len(conditions), max_count, 4), dtype=np.float32)
    train_y = np.zeros((len(conditions), max_count), dtype=np.float32)
    valid = np.zeros((len(conditions), max_count), dtype=np.float32)
    for index, condition in enumerate(conditions):
        ids = np.asarray(condition.train_indices, dtype=np.int64)
        train_x[index, : len(ids)] = inputs_np[ids]
        train_y[index, : len(ids)] = targets_np[ids]
        valid[index, : len(ids)] = 1.0
    train_x = np.repeat(train_x, Config.SEED_COUNT, axis=0)
    train_y = np.repeat(train_y, Config.SEED_COUNT, axis=0)
    valid = np.repeat(valid, Config.SEED_COUNT, axis=0)
    full_x = np.repeat(
        inputs_np[None], len(conditions) * Config.SEED_COUNT, axis=0
    )
    return (
        torch.from_numpy(train_x).to(device),
        torch.from_numpy(train_y).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(full_x).to(device),
    )


def function_ids(logits: torch.Tensor) -> np.ndarray:
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=logits.device),
        torch.arange(16, dtype=torch.int64, device=logits.device),
    )
    return (((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1))\
        .cpu().numpy().astype(np.uint16)


@torch.inference_mode()
def evaluate(
    model: BatchedIndependentMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid: torch.Tensor,
    full_x: torch.Tensor,
    conditions: Sequence[TrainCondition],
    step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    train_logits = model(train_x)
    per_bit_loss = F.binary_cross_entropy_with_logits(
        train_logits, train_y, reduction="none"
    )
    losses = (per_bit_loss * valid).sum(dim=1) / valid.sum(dim=1)
    train_exact = (((train_logits >= 0) == (train_y >= 0.5)) | (valid == 0))\
        .all(dim=1)
    ids = function_ids(model(full_x))
    losses_np = losses.cpu().numpy()
    exact_np = train_exact.cpu().numpy()
    tracked = {
        "AND_F000": 0xF000,
        "D440": 0xD440,
        "F040": 0xF040,
        "D040": 0xD040,
        "F440": 0xF440,
        "F050": 0xF050,
        "D000": 0xD000,
        "F500": 0xF500,
    }
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    raw_payload: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(conditions):
        start = condition_index * Config.SEED_COUNT
        end = start + Config.SEED_COUNT
        condition_ids = ids[start:end].astype(np.int64)
        condition_loss = losses_np[start:end]
        condition_exact = exact_np[start:end]
        counts = np.bincount(condition_ids, minlength=65_536)
        differences = np.bitwise_xor(
            condition_ids.astype(np.uint16), np.uint16(0xF000)
        )
        exception_count = np.unpackbits(
            differences.view(np.uint8).reshape(-1, 2), axis=1
        ).sum(axis=1)
        probabilities = counts[counts > 0] / Config.SEED_COUNT
        function_entropy = float(
            -np.sum(probabilities * np.log2(probabilities))
        )
        row: dict[str, Any] = {
            "step": step,
            "condition": condition.name,
            "model_count": Config.SEED_COUNT,
            "train_loss_mean": float(np.mean(condition_loss)),
            "train_loss_median": float(np.median(condition_loss)),
            "train_loss_min": float(np.min(condition_loss)),
            "train_hard_exact_fraction": float(np.mean(condition_exact)),
            "unique_function_count": int(np.sum(counts > 0)),
            "function_entropy_bits": function_entropy,
            "effective_function_count": float(2.0 ** function_entropy),
            "mean_exception_count_vs_and": float(np.mean(exception_count)),
            "median_exception_count_vs_and": float(np.median(exception_count)),
        }
        for exception_number in range(7):
            row[f"p_exception_count_{exception_number}"] = float(
                np.mean(exception_count == exception_number)
            )
        for name, function_id in tracked.items():
            row[f"p_{name}"] = int(counts[function_id]) / Config.SEED_COUNT
            row[f"count_{name}"] = int(counts[function_id])
        top_ids = np.argsort(counts)[::-1][:20]
        for rank, function_id in enumerate(top_ids[:3], start=1):
            row[f"top{rank}_function_hex"] = f"0x{int(function_id):04X}"
            row[f"top{rank}_probability"] = (
                int(counts[function_id]) / Config.SEED_COUNT
            )
        summary_rows.append(row)
        for rank, function_id in enumerate(top_ids, start=1):
            count = int(counts[function_id])
            if count == 0:
                break
            top_rows.append({
                "step": step,
                "condition": condition.name,
                "rank": rank,
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "count": count,
                "probability": count / Config.SEED_COUNT,
            })
        raw_payload[f"loss_{condition.name}_step{step}"] = condition_loss
        raw_payload[f"ids_{condition.name}_step{step}"] = condition_ids.astype(
            np.uint16
        )
    return summary_rows, top_rows, raw_payload


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, rows)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def save_progress_files(
    output_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    top_rows: Sequence[dict[str, Any]],
    raw_payload: dict[str, np.ndarray],
) -> None:
    atomic_write_csv(output_dir / "tracked_function_curves.csv", summary_rows)
    atomic_write_csv(output_dir / "top_functions.csv", top_rows)
    atomic_save_npz(
        output_dir / "checkpoint_model_states.npz", raw_payload
    )


def save_checkpoint(
    output_dir: Path,
    model: BatchedIndependentMLP,
    optimizer: torch.optim.Optimizer,
    step: int,
    elapsed_seconds: float,
    summary_rows: Sequence[dict[str, Any]],
    top_rows: Sequence[dict[str, Any]],
    raw_payload: dict[str, np.ndarray],
) -> None:
    save_progress_files(output_dir, summary_rows, top_rows, raw_payload)
    payload: dict[str, Any] = {
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "summary_rows": list(summary_rows),
        "top_rows": list(top_rows),
        "raw_payload": raw_payload,
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "active_conditions": tuple(Config.ACTIVE_CONDITIONS),
        "seed_count": Config.SEED_COUNT,
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    path = output_dir / "latest_checkpoint.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(
        f"checkpoint saved | step={step:,} | {path}", flush=True
    )


def load_checkpoint(
    output_dir: Path,
    model: BatchedIndependentMLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    path = output_dir / "latest_checkpoint.pt"
    if not Config.RESUME or not path.exists():
        return 0, 0.0, [], [], {}
    payload = torch.load(path, map_location=device, weights_only=False)
    if tuple(payload["active_conditions"]) != tuple(Config.ACTIVE_CONDITIONS):
        raise RuntimeError("checkpoint 的 ACTIVE_CONDITIONS 与当前配置不一致。")
    if int(payload["seed_count"]) != Config.SEED_COUNT:
        raise RuntimeError("checkpoint 的 SEED_COUNT 与当前配置不一致。")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.random.set_rng_state(payload["torch_rng_state"].cpu())
    np.random.set_state(payload["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in payload["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    print(f"resumed checkpoint | step={int(payload['step']):,}", flush=True)
    return (
        int(payload["step"]),
        float(payload["elapsed_seconds"]),
        list(payload["summary_rows"]),
        list(payload["top_rows"]),
        dict(payload["raw_payload"]),
    )


def print_eval_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        top_summary = " ".join(
            f"{row[f'top{rank}_function_hex']}="
            f"{row[f'top{rank}_probability']:.2%}"
            for rank in range(1, 4)
        )
        print(
            f"step={int(row['step']):8,d} | {row['condition']} "
            f"| BCE={row['train_loss_median']:.3e} "
            f"| exact={row['train_hard_exact_fraction']:.2%} "
            f"| AND={row['p_AND_F000']:.2%} "
            f"D440={row['p_D440']:.2%} F040={row['p_F040']:.2%} "
            f"D040={row['p_D040']:.2%} F440={row['p_F440']:.2%} "
            f"F050={row['p_F050']:.2%} D000={row['p_D000']:.2%} "
            f"F500={row['p_F500']:.2%} "
            f"| exceptions={row['mean_exception_count_vs_and']:.3f} "
            f"| H={row['function_entropy_bits']:.3f} "
            f"| top={top_summary}",
            flush=True,
        )


def main() -> None:
    apply_smoke_overrides()
    conditions = build_conditions()
    device = torch.device(Config.DEVICE)
    torch.manual_seed(Config.INITIALIZATION_SEED)
    np.random.seed(Config.INITIALIZATION_SEED)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    output_dir = prepare_result_dir()
    write_json(output_dir / "config.json", {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    })
    write_json(output_dir / "conditions.json", [condition.__dict__ for condition in conditions])
    train_x, train_y, valid, full_x = build_training_tensors(conditions, device)
    model = BatchedIndependentMLP(
        Config.SEED_COUNT, len(conditions)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    print("=== Balanced n=10 tanh16x2 many-seed retrain ===", flush=True)
    max_steps_label = (
        "infinite" if Config.MAX_STEPS is None else f"{Config.MAX_STEPS:,}"
    )
    print(
        f"device={device} | conditions={len(conditions)} | "
        f"seeds/condition={Config.SEED_COUNT:,} | "
        f"models={len(conditions) * Config.SEED_COUNT:,} | "
        f"max_steps={max_steps_label}",
        flush=True,
    )
    (
        step,
        elapsed_before,
        summary_rows,
        top_rows,
        raw_payload,
    ) = load_checkpoint(output_dir, model, optimizer, device)
    last_eval_step = max(
        (int(row["step"]) for row in summary_rows), default=-1
    )
    started = time.perf_counter()
    last_save_wall = started
    last_save_step = step
    interrupted = False

    def elapsed_total() -> float:
        return elapsed_before + (time.perf_counter() - started)

    def should_evaluate(current_step: int) -> bool:
        return (
            current_step in set(Config.EARLY_EVAL_STEPS)
            or current_step % Config.EVAL_INTERVAL_STEPS == 0
        )

    def evaluate_current() -> None:
        nonlocal last_eval_step
        if last_eval_step == step:
            return
        rows, tops, raw = evaluate(
            model, train_x, train_y, valid, full_x, conditions, step
        )
        summary_rows.extend(rows)
        top_rows.extend(tops)
        raw_payload.update(raw)
        last_eval_step = step
        print_eval_rows(rows)

    try:
        while Config.MAX_STEPS is None or step <= Config.MAX_STEPS:
            if should_evaluate(step):
                evaluate_current()
            if Config.MAX_STEPS is not None and step == Config.MAX_STEPS:
                break

            logits = model(train_x)
            per_bit = F.binary_cross_entropy_with_logits(
                logits, train_y, reduction="none"
            )
            per_model = (per_bit * valid).sum(dim=1) / valid.sum(dim=1)
            if not torch.isfinite(per_model).all():
                raise RuntimeError(
                    f"step={step} 出现非有限 loss，停止并保存。"
                )
            loss = per_model.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            now = time.perf_counter()
            save_due = (
                step - last_save_step >= Config.SAVE_INTERVAL_STEPS
                or now - last_save_wall >= Config.SAVE_INTERVAL_SECONDS
            )
            if save_due:
                if should_evaluate(step):
                    evaluate_current()
                save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    step,
                    elapsed_total(),
                    summary_rows,
                    top_rows,
                    raw_payload,
                )
                last_save_step = step
                last_save_wall = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\n收到 Ctrl+C，正在评估并保存 step={step:,}...",
            flush=True,
        )
    finally:
        evaluate_current()
        save_checkpoint(
            output_dir,
            model,
            optimizer,
            step,
            elapsed_total(),
            summary_rows,
            top_rows,
            raw_payload,
        )
        write_json(output_dir / "summary.json", {
            "elapsed_seconds": elapsed_total(),
            "last_step": step,
            "interrupted": interrupted,
            "conditions": [condition.name for condition in conditions],
            "seed_count": Config.SEED_COUNT,
            "tracked_functions": [
                "AND_F000", "D440", "F040", "D040",
                "F440", "F050", "D000", "F500",
            ],
            "question": (
                "同SMC配置的真实AdamW长期训练最终选择AND还是静态深尾函数。"
            ),
        })
        archive = None
        if Config.PACKAGE_RESULTS:
            archive_path = output_dir.parent / f"{output_dir.name}_package.zip"
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as handle:
                for path in sorted(output_dir.rglob("*")):
                    if path.is_file() and path.name != "latest_checkpoint.pt":
                        handle.write(path, path.relative_to(output_dir.parent))
            archive = str(archive_path)
        print("=== 已安全保存 ===", flush=True)
        if archive:
            print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
