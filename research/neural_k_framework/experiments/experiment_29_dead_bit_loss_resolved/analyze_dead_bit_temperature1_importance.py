"""直接重要性积分标准温度1 dead-bit Bayesian posterior。

训练集只有z=0的8个状态。对Gaussian prior样本使用精确权重

    exp(-8 * mean_BCE)

即可得到与标准温度1 HMC完全相同的目标分布。该脚本报告重要性ESS、posterior
平均训练loss、各dead-bit shift下的严格样本质量、不变质量和BMA准确率。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    PRIOR_SAMPLES = 1_048_576
    CHUNK_SIZE = 8_192
    RANDOM_SEED = 2026090699
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULT_DIR = Path(
        "/root/autodl-tmp/results_dead_bit_temperature1_importance"
    )
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


def load_core():
    path = Path(__file__).with_name(
        "experiment_dead_bit_static_sgd_nngp.py"
    )
    spec = importlib.util.spec_from_file_location("dead_bit_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载核心脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    core = load_core()
    if Config.SMOKE_TEST:
        Config.PRIOR_SAMPLES = 256
        Config.CHUNK_SIZE = 64
        Config.DEVICE = "cpu"
        Config.RESULT_DIR = Path(
            "research/function_information_conservation/"
            "_smoke_dead_bit_temperature1_importance"
        )
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch不可见。")
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.RANDOM_SEED)
    all_inputs = core.shifted_inputs(device)
    flat_inputs = all_inputs.reshape(-1, core.Config.INPUT_BITS)
    beta = float(1 << core.Config.ACTIVE_BITS)
    rows = []

    for function_name in core.Config.FUNCTION_NAMES:
        targets = core.function_targets(function_name, device)
        target_bits = targets.bool()
        weight_sum = 0.0
        weight_square_sum = 0.0
        weighted_loss_sum = 0.0
        sigmoid_sum = torch.zeros(
            len(core.Config.SHIFT_VALUES),
            len(targets),
            dtype=torch.float64,
            device=device,
        )
        strict_sum = torch.zeros(
            len(core.Config.SHIFT_VALUES),
            dtype=torch.float64,
            device=device,
        )
        invariance_sum = torch.zeros_like(strict_sum)
        completed = 0
        while completed < Config.PRIOR_SAMPLES:
            count = min(
                Config.CHUNK_SIZE,
                Config.PRIOR_SAMPLES-completed,
            )
            parameters = torch.randn(
                count,
                core.parameter_count(),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            logits = core.forward_normalized(
                parameters, flat_inputs
            ).reshape(
                count,
                len(core.Config.SHIFT_VALUES),
                len(targets),
            )
            losses = F.binary_cross_entropy_with_logits(
                logits[:, 0],
                targets[None].expand(count, -1),
                reduction="none",
            ).mean(dim=1).to(torch.float64)
            weights = torch.exp(-beta*losses)
            hard = logits >= 0
            exact = torch.all(
                hard == target_bits[None, None], dim=2
            )
            invariant = torch.all(
                hard == hard[:, 0, None], dim=2
            )
            weight_sum += float(weights.sum().item())
            weight_square_sum += float(weights.square().sum().item())
            weighted_loss_sum += float((weights*losses).sum().item())
            sigmoid_sum += (
                weights[:, None, None]
                * torch.sigmoid(logits).to(torch.float64)
            ).sum(dim=0)
            strict_sum += (
                weights[:, None]*exact.to(torch.float64)
            ).sum(dim=0)
            invariance_sum += (
                weights[:, None]*invariant.to(torch.float64)
            ).sum(dim=0)
            completed += count
            if completed % max(Config.CHUNK_SIZE*16, 1) == 0:
                print(
                    f"{function_name}: {completed}/{Config.PRIOR_SAMPLES}",
                    flush=True,
                )

        predictive = sigmoid_sum/weight_sum
        predictive_hard = predictive >= 0.5
        ess = weight_sum*weight_sum/weight_square_sum
        row: dict[str, Any] = {
            "function": function_name,
            "prior_samples": Config.PRIOR_SAMPLES,
            "beta": beta,
            "ess": ess,
            "ess_fraction": ess/Config.PRIOR_SAMPLES,
            "posterior_mean_train_loss": (
                weighted_loss_sum/weight_sum
            ),
        }
        for index, shift in enumerate(core.Config.SHIFT_VALUES):
            key = f"z{float(shift):g}"
            row[f"{key}_strict_mass"] = float(
                strict_sum[index].item()/weight_sum
            )
            row[f"{key}_invariance_mass"] = float(
                invariance_sum[index].item()/weight_sum
            )
            row[f"{key}_bma_accuracy"] = float(
                (
                    predictive_hard[index] == target_bits
                ).float().mean().item()
            )
        rows.append(row)
        print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)

    summary = {
        "status": "completed",
        "method": (
            "Gaussian-prior importance integration of the exact "
            "temperature-1 Bernoulli posterior"
        ),
        "prior_samples": Config.PRIOR_SAMPLES,
        "beta": beta,
        "minimum_ess_fraction": min(
            row["ess_fraction"] for row in rows
        ),
        "results": rows,
    }
    write_csv(Config.RESULT_DIR/"temperature1_importance.csv", rows)
    write_json(Config.RESULT_DIR/"summary.json", summary)
    print(f"结果目录：{Config.RESULT_DIR}", flush=True)


if __name__ == "__main__":
    main()
