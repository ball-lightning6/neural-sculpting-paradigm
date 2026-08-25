"""从SMC极深low-loss粒子出发继续真实训练。

自动读取 constrained SMC checkpoint，在当前最低loss条件粒子中找到概率最高
的hard function，抽取多个真实参数点，复制成严格配对的两组：

1. unconstrained AdamW：正常训练；
2. projected AdamW：每步更新后投影回初始化参数立方体。

直接判决SMC深尾函数是否沿梯度迁移到AND，以及离开初始化支撑是否必要。
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    SMC_CHECKPOINT = Path(
        "/root/results_static_low_loss_constrained_smc/checkpoint.pt"
    )
    TRAIN_INDICES = (1, 2, 3, 5, 7, 8, 11, 12, 14, 15)
    WIDTH = 16
    SOURCE_COUNT = 4_096
    SOURCE_SELECTION_SEED = 20261010

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 20_000
    EARLY_EVAL_STEPS = (
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    )
    EVAL_INTERVAL_STEPS = 100
    SAVE_INTERVAL_STEPS = 2_000
    SAVE_INTERVAL_SECONDS = 120.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_train_from_smc_deep_tail")
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


TRACKED_FUNCTIONS = {
    "AND_F000": 0xF000,
    "D440": 0xD440,
    "F040": 0xF040,
    "D040": 0xD040,
    "F440": 0xF440,
    "D000": 0xD000,
    "F050": 0xF050,
    "F500": 0xF500,
}


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.SMC_CHECKPOINT = Path(
        "research/function_information_conservation/"
        "_smoke_static_low_loss_constrained_smc/checkpoint.pt"
    )
    Config.SOURCE_COUNT = 8
    Config.MAX_STEPS = 2
    Config.EARLY_EVAL_STEPS = (0, 1, 2)
    Config.EVAL_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_STEPS = 1
    Config.SAVE_INTERVAL_SECONDS = 0.01
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_train_from_smc_deep_tail"
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
    values = np.arange(16, dtype=np.uint8)
    shifts = np.arange(3, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def and_targets(inputs: np.ndarray) -> np.ndarray:
    return (inputs[:, 0] * inputs[:, 1]).astype(np.float32)


def function_ids_from_logits(logits: torch.Tensor) -> np.ndarray:
    powers = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int64, device=logits.device),
        torch.arange(16, dtype=torch.int64, device=logits.device),
    )
    return (((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)) \
        .cpu().numpy().astype(np.uint16)


def normalized_forward(
    normalized: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    count = normalized.shape[0]
    width = Config.WIDTH
    cursor = 0
    size = width * 4
    w1 = normalized[:, cursor:cursor + size].reshape(count, width, 4) * 0.5
    cursor += size
    b1 = normalized[:, cursor:cursor + width] * 0.5
    cursor += width
    size = width * width
    w2 = normalized[:, cursor:cursor + size].reshape(count, width, width) * 0.25
    cursor += size
    b2 = normalized[:, cursor:cursor + width] * 0.25
    cursor += width
    w3 = normalized[:, cursor:cursor + width].reshape(count, 1, width) * 0.25
    cursor += width
    b3 = normalized[:, cursor:cursor + 1] * 0.25
    hidden = torch.tanh(
        torch.bmm(inputs[None].expand(count, -1, -1), w1.transpose(1, 2))
        + b1[:, None]
    )
    hidden = torch.tanh(
        torch.bmm(hidden, w2.transpose(1, 2)) + b2[:, None]
    )
    return torch.bmm(hidden, w3.transpose(1, 2)).squeeze(-1) + b3


def select_source_particles(
    checkpoint: dict[str, Any],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, Any]]:
    saved_config = checkpoint["config"]
    if int(saved_config["WIDTH"]) != Config.WIDTH:
        raise RuntimeError("SMC checkpoint的WIDTH与实验不一致。")
    if tuple(saved_config["TRAIN_INDICES"]) != tuple(Config.TRAIN_INDICES):
        raise RuntimeError("SMC checkpoint的TRAIN_INDICES与实验不一致。")
    particles = checkpoint["particles"].reshape(-1, checkpoint["particles"].shape[-1])
    particles_device = particles.to(inputs.device)
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(particles_device), 8_192):
            logits = normalized_forward(particles_device[start:start + 8_192], inputs)
            pieces.append(function_ids_from_logits(logits))
    ids = np.concatenate(pieces)
    counts = np.bincount(ids.astype(np.int64), minlength=65_536)
    source_function_id = int(np.argmax(counts))
    candidates = np.flatnonzero(ids == source_function_id)
    if len(candidates) < Config.SOURCE_COUNT:
        raise RuntimeError(
            f"最高概率函数0x{source_function_id:04X}只有{len(candidates)}个粒子，"
            f"少于SOURCE_COUNT={Config.SOURCE_COUNT}。"
        )
    rng = np.random.default_rng(Config.SOURCE_SELECTION_SEED)
    selected = np.sort(rng.choice(
        candidates, size=Config.SOURCE_COUNT, replace=False
    ))
    flat_losses = checkpoint["losses"].reshape(-1).cpu().numpy()
    flat_lineages = checkpoint["lineages"].reshape(-1).cpu().numpy()
    source = particles[selected].to(dtype=torch.float32)
    summary = {
        "checkpoint_level": int(checkpoint["level"]),
        "checkpoint_threshold": float(checkpoint["current_threshold"]),
        "checkpoint_particle_count": int(len(ids)),
        "checkpoint_top_function_id": source_function_id,
        "checkpoint_top_function_hex": f"0x{source_function_id:04X}",
        "checkpoint_top_function_probability": float(
            counts[source_function_id] / counts.sum()
        ),
        "selected_count": int(len(selected)),
        "selected_loss_min": float(flat_losses[selected].min()),
        "selected_loss_median": float(np.median(flat_losses[selected])),
        "selected_loss_max": float(flat_losses[selected].max()),
        "selected_unique_lineages": int(np.unique(flat_lineages[selected]).size),
        "checkpoint_top_functions": [
            {
                "function_hex": f"0x{int(function_id):04X}",
                "count": int(counts[function_id]),
                "probability": float(counts[function_id] / counts.sum()),
            }
            for function_id in np.argsort(counts)[::-1][:10]
            if counts[function_id] > 0
        ],
    }
    return source, selected, flat_lineages[selected], summary


class PairedBatchedMLP(nn.Module):
    def __init__(self, normalized_source: torch.Tensor, device: torch.device) -> None:
        super().__init__()
        source = normalized_source.to(device)
        paired = torch.cat([source, source], dim=0)
        count = paired.shape[0]
        width = Config.WIDTH
        cursor = 0
        size = width * 4
        w1 = paired[:, cursor:cursor + size].reshape(count, width, 4) * 0.5
        cursor += size
        b1 = paired[:, cursor:cursor + width] * 0.5
        cursor += width
        size = width * width
        w2 = paired[:, cursor:cursor + size].reshape(count, width, width) * 0.25
        cursor += size
        b2 = paired[:, cursor:cursor + width] * 0.25
        cursor += width
        w3 = paired[:, cursor:cursor + width].reshape(count, 1, width) * 0.25
        cursor += width
        b3 = paired[:, cursor:cursor + 1] * 0.25
        self.weights = nn.ParameterList([
            nn.Parameter(w1.clone()),
            nn.Parameter(w2.clone()),
            nn.Parameter(w3.clone()),
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(b1.clone()),
            nn.Parameter(b2.clone()),
            nn.Parameter(b3.clone()),
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs[None].expand(len(self.weights[0]), -1, -1)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if index < len(self.weights) - 1:
                hidden = torch.tanh(hidden)
        return hidden.squeeze(-1)

    @torch.no_grad()
    def project_second_arm(self, source_count: int) -> None:
        bounds = (0.5, 0.25, 0.25)
        for weight, bias, bound in zip(self.weights, self.biases, bounds):
            weight[source_count:].clamp_(-bound, bound)
            bias[source_count:].clamp_(-bound, bound)

    @torch.no_grad()
    def normalized_vectors(self) -> torch.Tensor:
        bounds = (0.5, 0.25, 0.25)
        pieces: list[torch.Tensor] = []
        for weight, bias, bound in zip(self.weights, self.biases, bounds):
            pieces.append(weight.flatten(1) / bound)
            pieces.append(bias.flatten(1) / bound)
        return torch.cat(pieces, dim=1)


def entropy_from_counts(counts: np.ndarray) -> float:
    probability = counts[counts > 0] / counts.sum()
    return float(-np.sum(probability * np.log2(probability)))


@torch.inference_mode()
def evaluate(
    model: PairedBatchedMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_indices: torch.Tensor,
    source_vectors: torch.Tensor,
    source_function_id: int,
    step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    logits = model(inputs)
    local = logits[:, train_indices]
    local_targets = targets[train_indices][None].expand_as(local)
    losses = F.binary_cross_entropy_with_logits(
        local, local_targets, reduction="none"
    ).mean(dim=1)
    exact = ((local >= 0) == (local_targets >= 0.5)).all(dim=1)
    ids = function_ids_from_logits(logits)
    normalized = model.normalized_vectors()
    paired_source = torch.cat([
        source_vectors.to(normalized.device),
        source_vectors.to(normalized.device),
    ], dim=0)
    displacement = torch.linalg.vector_norm(normalized - paired_source, dim=1)
    outside = normalized.abs() > 1.0 + 1e-6

    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    count = Config.SOURCE_COUNT
    for arm_index, arm in enumerate(("unconstrained", "projected")):
        start = arm_index * count
        stop = start + count
        arm_ids = ids[start:stop].astype(np.int64)
        arm_counts = np.bincount(arm_ids, minlength=65_536)
        arm_loss = losses[start:stop].cpu().numpy()
        arm_exact = exact[start:stop].cpu().numpy()
        arm_outside = outside[start:stop]
        row: dict[str, Any] = {
            "step": step,
            "arm": arm,
            "model_count": count,
            "train_loss_min": float(np.min(arm_loss)),
            "train_loss_median": float(np.median(arm_loss)),
            "train_loss_mean": float(np.mean(arm_loss)),
            "train_hard_exact_fraction": float(np.mean(arm_exact)),
            "source_function_probability": float(
                arm_counts[source_function_id] / count
            ),
            "function_entropy_bits": entropy_from_counts(arm_counts),
            "unique_function_count": int(np.count_nonzero(arm_counts)),
            "mean_normalized_l2_displacement": float(
                displacement[start:stop].mean().item()
            ),
            "coordinate_outside_fraction": float(
                arm_outside.float().mean().item()
            ),
            "model_any_coordinate_outside_fraction": float(
                arm_outside.any(dim=1).float().mean().item()
            ),
            "max_normalized_abs": float(
                normalized[start:stop].abs().max().item()
            ),
        }
        for name, function_id in TRACKED_FUNCTIONS.items():
            row[f"p_{name}"] = float(arm_counts[function_id] / count)
            row[f"count_{name}"] = int(arm_counts[function_id])
        top_ids = np.argsort(arm_counts)[::-1][:20]
        for rank, function_id in enumerate(top_ids[:3], start=1):
            row[f"top{rank}_function_hex"] = f"0x{int(function_id):04X}"
            row[f"top{rank}_probability"] = float(
                arm_counts[function_id] / count
            )
        rows.append(row)
        for rank, function_id in enumerate(top_ids, start=1):
            function_count = int(arm_counts[function_id])
            if function_count == 0:
                break
            top_rows.append({
                "step": step,
                "arm": arm,
                "rank": rank,
                "function_id": int(function_id),
                "function_hex": f"0x{int(function_id):04X}",
                "count": function_count,
                "probability": function_count / count,
            })
    return rows, top_rows, ids


def print_rows(rows: Sequence[dict[str, Any]], source_hex: str) -> None:
    for row in rows:
        print(
            f"step={int(row['step']):>6,} | {row['arm']:<13} "
            f"BCE={row['train_loss_median']:.3e} "
            f"exact={row['train_hard_exact_fraction']:.2%} "
            f"source({source_hex})={row['source_function_probability']:.2%} "
            f"AND={row['p_AND_F000']:.2%} F040={row['p_F040']:.2%} "
            f"D440={row['p_D440']:.2%} D040={row['p_D040']:.2%} "
            f"F440={row['p_F440']:.2%} "
            f"outside-models={row['model_any_coordinate_outside_fraction']:.2%} "
            f"drift={row['mean_normalized_l2_displacement']:.3f}",
            flush=True,
        )


def save_progress(
    output_dir: Path,
    model: PairedBatchedMLP,
    optimizer: torch.optim.Optimizer,
    step: int,
    rows: Sequence[dict[str, Any]],
    top_rows: Sequence[dict[str, Any]],
    first_and_step: np.ndarray,
) -> None:
    write_csv(output_dir / "trajectory.csv", rows)
    write_csv(output_dir / "top_functions.csv", top_rows)
    np.savez_compressed(
        output_dir / "first_and_steps.npz",
        first_and_step=first_and_step,
    )
    temporary = output_dir / "latest_checkpoint.tmp.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rows": list(rows),
        "top_rows": list(top_rows),
        "first_and_step": first_and_step,
    }, temporary)
    os.replace(temporary, output_dir / "latest_checkpoint.pt")


def create_archive(output_dir: Path) -> Path:
    archive_path = output_dir.parent / f"{output_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name != "latest_checkpoint.pt":
                archive.write(path, path.relative_to(output_dir.parent))
    return archive_path


def main() -> None:
    apply_smoke_overrides()
    output_dir = prepare_result_dir()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    inputs_np = truth_table_inputs()
    targets_np = and_targets(inputs_np)
    inputs = torch.from_numpy(inputs_np).to(device)
    targets = torch.from_numpy(targets_np).to(device)
    train_indices = torch.tensor(
        Config.TRAIN_INDICES, dtype=torch.int64, device=device
    )

    if not Config.SMC_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"找不到SMC checkpoint：{Config.SMC_CHECKPOINT}"
        )
    checkpoint = torch.load(
        Config.SMC_CHECKPOINT, map_location="cpu", weights_only=False
    )
    source, selected, selected_lineages, source_summary = select_source_particles(
        checkpoint, inputs
    )
    source_function_id = int(source_summary["checkpoint_top_function_id"])
    source_hex = f"0x{source_function_id:04X}"
    write_json(output_dir / "config.json", {
        name: json_ready(getattr(Config, name))
        for name in dir(Config) if name.isupper()
    })
    write_json(output_dir / "source_summary.json", source_summary)
    np.savez_compressed(
        output_dir / "selected_source_particles.npz",
        normalized_particles=source.cpu().numpy(),
        flat_checkpoint_indices=selected,
        lineages=selected_lineages,
    )

    model = PairedBatchedMLP(source, device).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    first_and_step = np.full(
        (2, Config.SOURCE_COUNT), -1, dtype=np.int32
    )
    last_eval_ids: np.ndarray | None = None
    last_eval_step = -1
    interrupted = False
    started = time.perf_counter()
    last_save = started

    print("=== Train from SMC deep-tail particles ===")
    print(f"device={device} | source={source_hex} | pairs={Config.SOURCE_COUNT:,}")
    print(
        f"checkpoint level={source_summary['checkpoint_level']} "
        f"threshold={source_summary['checkpoint_threshold']:.6g} | "
        f"source probability={source_summary['checkpoint_top_function_probability']:.2%}"
    )
    print(f"result={output_dir.resolve()}")

    eval_steps = set(Config.EARLY_EVAL_STEPS)

    def should_evaluate(step: int) -> bool:
        return step in eval_steps or step % Config.EVAL_INTERVAL_STEPS == 0

    def evaluate_current(step: int) -> None:
        nonlocal last_eval_ids, last_eval_step
        if last_eval_step == step:
            return
        current_rows, current_top, ids = evaluate(
            model,
            inputs,
            targets,
            train_indices,
            source,
            source_function_id,
            step,
        )
        rows.extend(current_rows)
        top_rows.extend(current_top)
        reshaped = ids.reshape(2, Config.SOURCE_COUNT)
        newly_and = (reshaped == 0xF000) & (first_and_step < 0)
        first_and_step[newly_and] = step
        last_eval_ids = reshaped
        last_eval_step = step
        print_rows(current_rows, source_hex)

    step = 0
    try:
        while step <= Config.MAX_STEPS:
            if should_evaluate(step):
                evaluate_current(step)
            if step == Config.MAX_STEPS:
                break
            logits = model(inputs)
            local = logits[:, train_indices]
            local_targets = targets[train_indices][None].expand_as(local)
            per_model = F.binary_cross_entropy_with_logits(
                local, local_targets, reduction="none"
            ).mean(dim=1)
            if not torch.isfinite(per_model).all():
                raise RuntimeError(f"step={step}出现非有限loss。")
            optimizer.zero_grad(set_to_none=True)
            per_model.sum().backward()
            optimizer.step()
            model.project_second_arm(Config.SOURCE_COUNT)
            step += 1

            now = time.perf_counter()
            if (
                step % Config.SAVE_INTERVAL_STEPS == 0
                or now - last_save >= Config.SAVE_INTERVAL_SECONDS
            ):
                if should_evaluate(step):
                    evaluate_current(step)
                save_progress(
                    output_dir,
                    model,
                    optimizer,
                    step,
                    rows,
                    top_rows,
                    first_and_step,
                )
                last_save = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
        print("收到Ctrl+C，正在保存……", flush=True)
    finally:
        if not rows or int(rows[-1]["step"]) != step:
            evaluate_current(step)
        save_progress(
            output_dir,
            model,
            optimizer,
            step,
            rows,
            top_rows,
            first_and_step,
        )
        final_rows = [row for row in rows if int(row["step"]) == step]
        write_json(output_dir / "summary.json", {
            "status": "interrupted" if interrupted else "completed",
            "last_step": step,
            "elapsed_seconds": time.perf_counter() - started,
            "source": source_summary,
            "final_rows": final_rows,
            "first_and_reached_fraction": {
                "unconstrained": float(np.mean(first_and_step[0] >= 0)),
                "projected": float(np.mean(first_and_step[1] >= 0)),
            },
        })
        archive = create_archive(output_dir) if Config.PACKAGE_RESULTS else None
        print("=== 已保存 ===")
        if archive is not None:
            print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
