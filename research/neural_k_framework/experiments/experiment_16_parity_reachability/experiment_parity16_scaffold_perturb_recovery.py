"""
16-bit parity 的 scaffold -> 撤去解耦 -> 微扰恢复判决实验。

同一组初始化同时产生：

1. endpoint-only 单输出 MLP，直接训练最终 parity；
2. prefix-XOR scaffold MLP，同时输出全部累计 parity。

scaffold 训练后抽取最后一个输出 head，复制到与 direct 完全相同的单输出
架构，并把 Adam 的 step/exp_avg/exp_avg_sq 一起迁移。撤去全部中间监督后，
先测试 endpoint-only 稳定性，再施加不同相对 L2 半径的全参数扰动，只用
endpoint BCE 训练并测量是否返回精确 parity。

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
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    INPUT_BITS = 16
    WIDTH = 256
    HIDDEN_LAYERS = 3
    ACTIVATION = "gelu"
    USE_LAYER_NORM = True
    LAYERNORM_EPS = 1e-5

    SEED_COUNT = 8
    INITIALIZATION_SEED = 20260831
    BATCH_SEED = 20260901
    PERTURBATION_SEED = 20260902

    OPTIMIZER = "adamw"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    TRAIN_BATCH_SIZE = 512
    EVAL_CHUNK_SIZE = 1_024

    DIRECT_MAX_STEPS = 30_000
    SCAFFOLD_MAX_STEPS = 30_000
    ENDPOINT_SETTLE_STEPS = 20_000
    RECOVERY_MAX_STEPS = 5_000
    EVAL_INTERVAL = 500
    LOG_INTERVAL = 500

    ANCHOR_REQUIRED_LOSS = 1e-4
    MAX_ANCHORS = 8
    PERTURBATIONS_PER_ANCHOR = 4
    RELATIVE_L2_RADII = (0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path("/root/results_parity16_scaffold_perturb_recovery_w256")
    PACKAGE_RESULTS = True
    OVERWRITE_RESULT_DIR = False
    SMOKE_TEST = False


@dataclass
class BinaryEvaluation:
    loss: torch.Tensor
    error_count: torch.Tensor
    exact: torch.Tensor


@dataclass
class ScaffoldEvaluation:
    per_prefix_loss: torch.Tensor
    per_prefix_exact: torch.Tensor
    all_prefix_exact: torch.Tensor


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.INPUT_BITS = 4
    Config.WIDTH = 16
    Config.SEED_COUNT = 2
    Config.TRAIN_BATCH_SIZE = 16
    Config.EVAL_CHUNK_SIZE = 16
    Config.DIRECT_MAX_STEPS = 100
    Config.SCAFFOLD_MAX_STEPS = 500
    Config.ENDPOINT_SETTLE_STEPS = 50
    Config.RECOVERY_MAX_STEPS = 100
    Config.EVAL_INTERVAL = 50
    Config.LOG_INTERVAL = 50
    Config.ANCHOR_REQUIRED_LOSS = 0.05
    Config.MAX_ANCHORS = 2
    Config.PERTURBATIONS_PER_ANCHOR = 2
    Config.RELATIVE_L2_RADII = (0.0, 0.05, 0.2)
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_parity16_scaffold_perturb_recovery"
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


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


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
    if Config.INPUT_BITS < 2:
        raise ValueError("INPUT_BITS 必须至少为2。")
    if Config.WIDTH < 1 or Config.HIDDEN_LAYERS < 1:
        raise ValueError("WIDTH/HIDDEN_LAYERS 必须为正。")
    if Config.SEED_COUNT < 1:
        raise ValueError("SEED_COUNT 必须为正。")
    if Config.OPTIMIZER not in {"adamw", "sgd"}:
        raise ValueError("OPTIMIZER 只支持 adamw/sgd。")
    radii = tuple(float(value) for value in Config.RELATIVE_L2_RADII)
    if tuple(sorted(set(radii))) != radii or radii[0] != 0.0:
        raise ValueError("RELATIVE_L2_RADII 必须从0严格递增。")


def truth_table(bits: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(1 << bits, dtype=np.uint64)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    inputs = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    prefix = np.bitwise_xor.accumulate(inputs, axis=1).astype(np.uint8)
    return inputs, prefix


class BatchedMLP(nn.Module):
    def __init__(
        self,
        input_bits: int,
        output_size: int,
        model_seed_indices: np.ndarray,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.output_size = output_size
        self.model_count = len(model_seed_indices)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()

        indices = torch.from_numpy(model_seed_indices).to(device)
        base_count = int(model_seed_indices.max()) + 1
        generator = torch.Generator(device=device)
        generator.manual_seed(Config.INITIALIZATION_SEED)
        dimensions = (
            [input_bits]
            + [Config.WIDTH] * Config.HIDDEN_LAYERS
            + [output_size]
        )
        for layer_index, (input_size, layer_output) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            bound = 1.0 / math.sqrt(input_size)
            base_weight = torch.empty(
                base_count,
                layer_output,
                input_size,
                device=device,
            ).uniform_(-bound, bound, generator=generator)
            base_bias = torch.empty(
                base_count, layer_output, device=device
            ).uniform_(-bound, bound, generator=generator)
            self.weights.append(nn.Parameter(base_weight[indices].clone()))
            self.biases.append(nn.Parameter(base_bias[indices].clone()))
            if layer_index < Config.HIDDEN_LAYERS:
                self.norm_weights.append(nn.Parameter(torch.ones(
                    self.model_count, layer_output, device=device
                )))
                self.norm_biases.append(nn.Parameter(torch.zeros(
                    self.model_count, layer_output, device=device
                )))

    def activate(self, value: torch.Tensor) -> torch.Tensor:
        if Config.ACTIVATION == "gelu":
            return F.gelu(value)
        if Config.ACTIVATION == "tanh":
            return torch.tanh(value)
        return F.relu(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            hidden = inputs[None].expand(self.model_count, -1, -1)
        elif inputs.ndim == 3 and inputs.shape[0] == self.model_count:
            hidden = inputs
        else:
            raise ValueError(
                "inputs 必须是 [states,bits] 或 [models,states,bits]。"
            )
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            hidden = torch.bmm(hidden, weight.transpose(1, 2))
            hidden = hidden + bias[:, None, :]
            if layer_index < Config.HIDDEN_LAYERS:
                hidden = self.activate(hidden)
                if Config.USE_LAYER_NORM:
                    mean = hidden.mean(dim=-1, keepdim=True)
                    variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
                    hidden = (hidden - mean) * torch.rsqrt(
                        variance + Config.LAYERNORM_EPS
                    )
                    hidden = (
                        hidden * self.norm_weights[layer_index][:, None, :]
                        + self.norm_biases[layer_index][:, None, :]
                    )
        return hidden


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if Config.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )


def copy_endpoint_head(
    destination: BatchedMLP,
    scaffold: BatchedMLP,
    head_index: int,
) -> None:
    if destination.output_size != 1:
        raise ValueError("destination 必须是单输出网络。")
    with torch.no_grad():
        for index in range(Config.HIDDEN_LAYERS):
            destination.weights[index].copy_(scaffold.weights[index])
            destination.biases[index].copy_(scaffold.biases[index])
            destination.norm_weights[index].copy_(scaffold.norm_weights[index])
            destination.norm_biases[index].copy_(scaffold.norm_biases[index])
        destination.weights[-1][:, 0].copy_(
            scaffold.weights[-1][:, head_index]
        )
        destination.biases[-1][:, 0].copy_(
            scaffold.biases[-1][:, head_index]
        )


def copy_optimizer_tensor(
    value: torch.Tensor,
    head_index: int | None = None,
) -> torch.Tensor:
    if head_index is None or value.ndim == 0:
        return value.detach().clone()
    return value[:, head_index : head_index + 1].detach().clone()


def transfer_scaffold_optimizer(
    scaffold_optimizer: torch.optim.Optimizer,
    endpoint_optimizer: torch.optim.Optimizer,
    scaffold: BatchedMLP,
    endpoint: BatchedMLP,
    head_index: int,
) -> None:
    mappings: list[tuple[torch.Tensor, torch.Tensor, int | None]] = []
    for index in range(Config.HIDDEN_LAYERS):
        mappings.extend([
            (scaffold.weights[index], endpoint.weights[index], None),
            (scaffold.biases[index], endpoint.biases[index], None),
            (
                scaffold.norm_weights[index],
                endpoint.norm_weights[index],
                None,
            ),
            (
                scaffold.norm_biases[index],
                endpoint.norm_biases[index],
                None,
            ),
        ])
    mappings.extend([
        (scaffold.weights[-1], endpoint.weights[-1], head_index),
        (scaffold.biases[-1], endpoint.biases[-1], head_index),
    ])
    for source, destination, selected_head in mappings:
        source_state = scaffold_optimizer.state.get(source, {})
        destination_state: dict[str, Any] = {}
        for key, value in source_state.items():
            if not isinstance(value, torch.Tensor):
                destination_state[key] = value
            else:
                destination_state[key] = copy_optimizer_tensor(
                    value, selected_head
                )
        if destination_state:
            endpoint_optimizer.state[destination] = destination_state


@torch.no_grad()
def evaluate_binary(
    model: BatchedMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> BinaryEvaluation:
    loss_sum = torch.zeros(model.model_count, device=inputs.device)
    errors = torch.zeros(
        model.model_count, dtype=torch.int64, device=inputs.device
    )
    for start in range(0, len(inputs), Config.EVAL_CHUNK_SIZE):
        end = min(start + Config.EVAL_CHUNK_SIZE, len(inputs))
        logits = model(inputs[start:end]).squeeze(-1)
        local_targets = targets[start:end]
        loss_sum += F.binary_cross_entropy_with_logits(
            logits,
            local_targets[None].expand_as(logits),
            reduction="none",
        ).sum(dim=1)
        errors += (
            (logits >= 0) != local_targets.bool()[None]
        ).sum(dim=1)
    return BinaryEvaluation(
        loss=loss_sum / len(inputs),
        error_count=errors,
        exact=errors == 0,
    )


@torch.no_grad()
def evaluate_scaffold(
    model: BatchedMLP,
    inputs: torch.Tensor,
    prefix_targets: torch.Tensor,
) -> ScaffoldEvaluation:
    loss_sum = torch.zeros(
        model.model_count, Config.INPUT_BITS, device=inputs.device
    )
    errors = torch.zeros(
        model.model_count,
        Config.INPUT_BITS,
        dtype=torch.int64,
        device=inputs.device,
    )
    for start in range(0, len(inputs), Config.EVAL_CHUNK_SIZE):
        end = min(start + Config.EVAL_CHUNK_SIZE, len(inputs))
        logits = model(inputs[start:end])
        targets = prefix_targets[start:end]
        loss_sum += F.binary_cross_entropy_with_logits(
            logits,
            targets[None].expand_as(logits),
            reduction="none",
        ).sum(dim=1)
        errors += (
            (logits >= 0) != targets.bool()[None]
        ).sum(dim=1)
    exact = errors == 0
    return ScaffoldEvaluation(
        per_prefix_loss=loss_sum / len(inputs),
        per_prefix_exact=exact,
        all_prefix_exact=exact.all(dim=1),
    )


def common_eval_steps(max_steps: int) -> set[int]:
    return {
        0, 1, 2, 5, 10, 20, 50, 100, 200,
        *range(Config.EVAL_INTERVAL, max_steps + 1, Config.EVAL_INTERVAL),
        max_steps,
    }


def train_direct(
    model: BatchedMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    optimizer = make_optimizer(model)
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(Config.BATCH_SEED)
    rows: list[dict[str, Any]] = []
    eval_steps = common_eval_steps(Config.DIRECT_MAX_STEPS)
    for step in range(Config.DIRECT_MAX_STEPS + 1):
        if step in eval_steps:
            evaluation = evaluate_binary(model, inputs, targets)
            rows.append({
                "step": step,
                "loss_mean": float(evaluation.loss.mean().item()),
                "loss_median": float(evaluation.loss.median().item()),
                "loss_min": float(evaluation.loss.min().item()),
                "loss_max": float(evaluation.loss.max().item()),
                "hard_exact_count": int(evaluation.exact.sum().item()),
                "hard_exact_fraction": float(
                    evaluation.exact.float().mean().item()
                ),
            })
            if step <= 200 or step % Config.LOG_INTERVAL == 0:
                print(
                    f"direct step={step:>7,} | "
                    f"loss={rows[-1]['loss_median']:.3e} | "
                    f"exact={rows[-1]['hard_exact_fraction']:.1%}",
                    flush=True,
                )
        if step == Config.DIRECT_MAX_STEPS:
            break
        batch_indices = torch.randint(
            0,
            len(inputs),
            (Config.TRAIN_BATCH_SIZE,),
            generator=generator,
            device=inputs.device,
        )
        logits = model(inputs[batch_indices]).squeeze(-1)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets[batch_indices][None].expand_as(logits),
            reduction="none",
        ).mean(dim=1)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
    return rows, optimizer


def train_scaffold(
    model: BatchedMLP,
    inputs: torch.Tensor,
    prefix_targets: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    optimizer = make_optimizer(model)
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(Config.BATCH_SEED)
    rows: list[dict[str, Any]] = []
    eval_steps = common_eval_steps(Config.SCAFFOLD_MAX_STEPS)
    for step in range(Config.SCAFFOLD_MAX_STEPS + 1):
        if step in eval_steps:
            evaluation = evaluate_scaffold(model, inputs, prefix_targets)
            rows.append({
                "step": step,
                "loss_mean": float(
                    evaluation.per_prefix_loss.mean().item()
                ),
                "loss_median": float(
                    evaluation.per_prefix_loss.mean(dim=1).median().item()
                ),
                "all_prefix_exact_count": int(
                    evaluation.all_prefix_exact.sum().item()
                ),
                "all_prefix_exact_fraction": float(
                    evaluation.all_prefix_exact.float().mean().item()
                ),
                "final_prefix_exact_count": int(
                    evaluation.per_prefix_exact[:, -1].sum().item()
                ),
                "final_prefix_exact_fraction": float(
                    evaluation.per_prefix_exact[:, -1].float().mean().item()
                ),
                "final_prefix_loss_mean": float(
                    evaluation.per_prefix_loss[:, -1].mean().item()
                ),
                "final_prefix_loss_median": float(
                    evaluation.per_prefix_loss[:, -1].median().item()
                ),
            })
            if step <= 200 or step % Config.LOG_INTERVAL == 0:
                print(
                    f"scaffold step={step:>7,} | "
                    f"loss={rows[-1]['loss_median']:.3e} | "
                    f"all-prefix={rows[-1]['all_prefix_exact_fraction']:.1%} | "
                    f"final={rows[-1]['final_prefix_exact_fraction']:.1%}",
                    flush=True,
                )
        if step == Config.SCAFFOLD_MAX_STEPS:
            break
        batch_indices = torch.randint(
            0,
            len(inputs),
            (Config.TRAIN_BATCH_SIZE,),
            generator=generator,
            device=inputs.device,
        )
        logits = model(inputs[batch_indices])
        losses = F.binary_cross_entropy_with_logits(
            logits,
            prefix_targets[batch_indices][None].expand_as(logits),
            reduction="none",
        ).mean(dim=(1, 2))
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
    return rows, optimizer


def settle_endpoint(
    model: BatchedMLP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(Config.BATCH_SEED + 1)
    rows: list[dict[str, Any]] = []
    eval_steps = common_eval_steps(Config.ENDPOINT_SETTLE_STEPS)
    for step in range(Config.ENDPOINT_SETTLE_STEPS + 1):
        if step in eval_steps:
            evaluation = evaluate_binary(model, inputs, targets)
            rows.append({
                "step": step,
                "loss_mean": float(evaluation.loss.mean().item()),
                "loss_median": float(evaluation.loss.median().item()),
                "loss_min": float(evaluation.loss.min().item()),
                "loss_max": float(evaluation.loss.max().item()),
                "hard_exact_count": int(evaluation.exact.sum().item()),
                "hard_exact_fraction": float(
                    evaluation.exact.float().mean().item()
                ),
            })
            if step <= 200 or step % Config.LOG_INTERVAL == 0:
                print(
                    f"endpoint-settle step={step:>6,} | "
                    f"loss={rows[-1]['loss_median']:.3e} | "
                    f"exact={rows[-1]['hard_exact_fraction']:.1%}",
                    flush=True,
                )
        if step == Config.ENDPOINT_SETTLE_STEPS:
            break
        batch_indices = torch.randint(
            0,
            len(inputs),
            (Config.TRAIN_BATCH_SIZE,),
            generator=generator,
            device=inputs.device,
        )
        logits = model(inputs[batch_indices]).squeeze(-1)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets[batch_indices][None].expand_as(logits),
            reduction="none",
        ).mean(dim=1)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
    return rows


def clone_endpoint_models(
    parent: BatchedMLP,
    parent_indices: np.ndarray,
    device: torch.device,
) -> BatchedMLP:
    child = BatchedMLP(
        Config.INPUT_BITS,
        1,
        np.zeros(len(parent_indices), dtype=np.int64),
        device,
    )
    indices = torch.from_numpy(parent_indices).to(device)
    with torch.no_grad():
        for destination, source in zip(child.weights, parent.weights):
            destination.copy_(source[indices])
        for destination, source in zip(child.biases, parent.biases):
            destination.copy_(source[indices])
        for destination, source in zip(
            child.norm_weights, parent.norm_weights
        ):
            destination.copy_(source[indices])
        for destination, source in zip(
            child.norm_biases, parent.norm_biases
        ):
            destination.copy_(source[indices])
    return child


def clone_endpoint_optimizer(
    parent_optimizer: torch.optim.Optimizer,
    parent_model: BatchedMLP,
    child_model: BatchedMLP,
    parent_indices: np.ndarray,
    device: torch.device,
) -> torch.optim.Optimizer:
    child_optimizer = make_optimizer(child_model)
    indices = torch.from_numpy(parent_indices).to(device)
    for source, destination in zip(
        parent_model.parameters(), child_model.parameters()
    ):
        source_state = parent_optimizer.state.get(source, {})
        destination_state: dict[str, Any] = {}
        for key, value in source_state.items():
            if not isinstance(value, torch.Tensor):
                destination_state[key] = value
            elif value.ndim > 0 and value.shape[0] == parent_model.model_count:
                destination_state[key] = value[indices].detach().clone()
            else:
                destination_state[key] = value.detach().clone()
        if destination_state:
            child_optimizer.state[destination] = destination_state
    return child_optimizer


def perturb_model_parameters(
    model: BatchedMLP,
    radius: float,
    generator: torch.Generator,
) -> None:
    if radius == 0.0:
        return
    parameters = list(model.parameters())
    model_count = model.model_count
    parameter_norm_sq = torch.zeros(
        model_count, device=parameters[0].device
    )
    noises: list[torch.Tensor] = []
    noise_norm_sq = torch.zeros_like(parameter_norm_sq)
    for parameter in parameters:
        axes = tuple(range(1, parameter.ndim))
        parameter_norm_sq += parameter.detach().square().sum(dim=axes)
        noise = torch.randn(
            parameter.shape,
            generator=generator,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        noises.append(noise)
        noise_norm_sq += noise.square().sum(dim=axes)
    scales = (
        float(radius)
        * torch.sqrt(parameter_norm_sq.clamp_min(1e-30))
        / torch.sqrt(noise_norm_sq.clamp_min(1e-30))
    )
    with torch.no_grad():
        for parameter, noise in zip(parameters, noises):
            shape = (model_count,) + (1,) * (parameter.ndim - 1)
            parameter.add_(noise * scales.view(shape))


def run_recovery_radius(
    radius: float,
    parent_model: BatchedMLP,
    parent_optimizer: torch.optim.Optimizer,
    anchor_indices: np.ndarray,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    perturbations = (
        1 if radius == 0.0 else Config.PERTURBATIONS_PER_ANCHOR
    )
    parent_indices = np.repeat(anchor_indices, perturbations)
    model = clone_endpoint_models(parent_model, parent_indices, device)
    optimizer = clone_endpoint_optimizer(
        parent_optimizer, parent_model, model, parent_indices, device
    )
    perturb_generator = torch.Generator(device=device)
    radius_index = list(Config.RELATIVE_L2_RADII).index(radius)
    perturb_generator.manual_seed(
        Config.PERTURBATION_SEED + radius_index * 1_000_003
    )
    perturb_model_parameters(model, radius, perturb_generator)

    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(
        Config.BATCH_SEED + 10_000_019 + radius_index
    )
    eval_steps = {
        0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000,
        Config.RECOVERY_MAX_STEPS,
    }
    aggregate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    initial_exact: np.ndarray | None = None
    first_exact_step = np.full(model.model_count, -1, dtype=np.int64)

    for step in range(Config.RECOVERY_MAX_STEPS + 1):
        if step in eval_steps:
            evaluation = evaluate_binary(model, inputs, targets)
            exact_np = evaluation.exact.cpu().numpy()
            if initial_exact is None:
                initial_exact = exact_np.copy()
            newly_exact = (first_exact_step < 0) & exact_np
            first_exact_step[newly_exact] = step
            initially_broken = ~initial_exact
            broken_count = int(initially_broken.sum())
            recovered_broken = int(
                (initially_broken & (first_exact_step >= 0)).sum()
            )
            aggregate_rows.append({
                "radius": radius,
                "step": step,
                "model_count": model.model_count,
                "loss_mean": float(evaluation.loss.mean().item()),
                "loss_median": float(evaluation.loss.median().item()),
                "loss_min": float(evaluation.loss.min().item()),
                "loss_max": float(evaluation.loss.max().item()),
                "hard_exact_count": int(evaluation.exact.sum().item()),
                "hard_exact_fraction": float(
                    evaluation.exact.float().mean().item()
                ),
                "initially_broken_count": broken_count,
                "ever_recovered_broken_count": recovered_broken,
                "ever_recovered_broken_fraction": (
                    recovered_broken / broken_count if broken_count else None
                ),
            })
            for local_index in range(model.model_count):
                model_rows.append({
                    "radius": radius,
                    "step": step,
                    "model_index": local_index,
                    "anchor_seed": int(parent_indices[local_index]),
                    "perturbation_index": int(
                        local_index % perturbations
                    ),
                    "initial_exact": bool(initial_exact[local_index]),
                    "loss": float(evaluation.loss[local_index].item()),
                    "error_count": int(
                        evaluation.error_count[local_index].item()
                    ),
                    "hard_exact": bool(exact_np[local_index]),
                    "first_exact_step": int(first_exact_step[local_index]),
                })
            if step <= 200 or step in {
                500, 1_000, 2_000, Config.RECOVERY_MAX_STEPS
            }:
                row = aggregate_rows[-1]
                print(
                    f"radius={radius:g} step={step:>5,} | "
                    f"loss={row['loss_median']:.3e} | "
                    f"exact={row['hard_exact_fraction']:.1%} | "
                    f"broken-recovered="
                    + (
                        "NA"
                        if row["ever_recovered_broken_fraction"] is None
                        else f"{row['ever_recovered_broken_fraction']:.1%}"
                    ),
                    flush=True,
                )
        if step == Config.RECOVERY_MAX_STEPS:
            break
        batch_indices = torch.randint(
            0,
            len(inputs),
            (Config.TRAIN_BATCH_SIZE,),
            generator=batch_generator,
            device=device,
        )
        logits = model(inputs[batch_indices]).squeeze(-1)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets[batch_indices][None].expand_as(logits),
            reduction="none",
        ).mean(dim=1)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()

    assert initial_exact is not None
    initially_broken = ~initial_exact
    recovered_steps = first_exact_step[initially_broken]
    recovered_steps = recovered_steps[recovered_steps >= 0]
    final_row = aggregate_rows[-1]
    summary = {
        "radius": radius,
        "model_count": model.model_count,
        "anchor_count": len(anchor_indices),
        "perturbations_per_anchor": perturbations,
        "initial_exact_count": int(initial_exact.sum()),
        "initial_exact_fraction": float(initial_exact.mean()),
        "initially_broken_count": int(initially_broken.sum()),
        "ever_recovered_broken_count": int(len(recovered_steps)),
        "ever_recovered_broken_fraction": (
            len(recovered_steps) / int(initially_broken.sum())
            if initially_broken.any() else None
        ),
        "median_first_recovery_step": (
            float(np.median(recovered_steps)) if len(recovered_steps) else None
        ),
        "final_exact_count": int(final_row["hard_exact_count"]),
        "final_exact_fraction": float(final_row["hard_exact_fraction"]),
        "final_loss_median": float(final_row["loss_median"]),
    }
    return aggregate_rows, model_rows, summary


def plot_results(
    output_dir: Path,
    direct_rows: Sequence[dict[str, Any]],
    scaffold_rows: Sequence[dict[str, Any]],
    settle_rows: Sequence[dict[str, Any]],
    recovery_rows: Sequence[dict[str, Any]],
    radius_summaries: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].plot(
        [row["step"] for row in direct_rows],
        [row["hard_exact_fraction"] for row in direct_rows],
        label="direct",
    )
    axes[0, 0].plot(
        [row["step"] for row in settle_rows],
        [row["hard_exact_fraction"] for row in settle_rows],
        label="scaffold -> endpoint",
    )
    axes[0, 0].set_title("Random reachability vs scaffold endpoint stability")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("hard-exact fraction")
    axes[0, 0].legend()

    axes[0, 1].plot(
        [row["step"] for row in scaffold_rows],
        [row["all_prefix_exact_fraction"] for row in scaffold_rows],
        label="all prefixes exact",
    )
    axes[0, 1].plot(
        [row["step"] for row in scaffold_rows],
        [row["final_prefix_exact_fraction"] for row in scaffold_rows],
        label="final prefix exact",
    )
    axes[0, 1].set_title("Scaffold training")
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].legend()

    for radius in sorted({float(row["radius"]) for row in recovery_rows}):
        local = [
            row for row in recovery_rows if float(row["radius"]) == radius
        ]
        axes[1, 0].plot(
            [row["step"] for row in local],
            [row["hard_exact_fraction"] for row in local],
            label=f"r={radius:g}",
        )
    axes[1, 0].set_title("Endpoint-only recovery after perturbation")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel("recovery step")
    axes[1, 0].set_ylabel("hard-exact fraction")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        [row["radius"] for row in radius_summaries],
        [
            np.nan
            if row["ever_recovered_broken_fraction"] is None
            else row["ever_recovered_broken_fraction"]
            for row in radius_summaries
        ],
        marker="o",
        label="ever recovered among initially broken",
    )
    axes[1, 1].plot(
        [row["radius"] for row in radius_summaries],
        [row["final_exact_fraction"] for row in radius_summaries],
        marker="o",
        label="final exact",
    )
    axes[1, 1].set_xscale("symlog", linthresh=1e-3)
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_xlabel("relative full-parameter L2 radius")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].set_title("Local basin recovery radius")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.savefig(
        output_dir / "parity_scaffold_perturb_recovery.png", dpi=180
    )
    plt.close(figure)


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
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
    apply_smoke_overrides()
    validate_config()
    output_dir = prepare_result_dir()
    write_json(output_dir / "config.json", config_dict())
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config.DEVICE 要求 CUDA，但 PyTorch 看不到 GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    inputs_np, prefix_np = truth_table(Config.INPUT_BITS)
    inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(device)
    prefix_targets = torch.from_numpy(prefix_np.astype(np.float32)).to(device)
    final_targets = prefix_targets[:, -1]
    seed_indices = np.arange(Config.SEED_COUNT, dtype=np.int64)

    scaffold = BatchedMLP(
        Config.INPUT_BITS,
        Config.INPUT_BITS,
        seed_indices,
        device,
    )
    direct = BatchedMLP(
        Config.INPUT_BITS, 1, seed_indices, device
    )
    copy_endpoint_head(direct, scaffold, Config.INPUT_BITS - 1)

    print("=== Parity scaffold -> endpoint -> perturb recovery ===", flush=True)
    print(f"设备：{device}", flush=True)
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"bits={Config.INPUT_BITS} | states={len(inputs_np):,} | "
        f"seeds={Config.SEED_COUNT} | width={Config.WIDTH} | "
        f"layers={Config.HIDDEN_LAYERS}",
        flush=True,
    )
    print(f"结果目录：{output_dir.resolve()}", flush=True)

    print("\n--- direct endpoint-only baseline ---", flush=True)
    direct_rows, _ = train_direct(direct, inputs, final_targets)
    write_csv(output_dir / "direct_trajectory.csv", direct_rows)

    print("\n--- prefix-XOR scaffold ---", flush=True)
    scaffold_rows, scaffold_optimizer = train_scaffold(
        scaffold, inputs, prefix_targets
    )
    write_csv(output_dir / "scaffold_trajectory.csv", scaffold_rows)
    scaffold_evaluation = evaluate_scaffold(
        scaffold, inputs, prefix_targets
    )
    write_csv(output_dir / "scaffold_prefix_summary.csv", [
        {
            "prefix_order": order + 1,
            "loss_mean": float(
                scaffold_evaluation.per_prefix_loss[:, order].mean().item()
            ),
            "loss_median": float(
                scaffold_evaluation.per_prefix_loss[:, order].median().item()
            ),
            "hard_exact_count": int(
                scaffold_evaluation.per_prefix_exact[:, order].sum().item()
            ),
            "hard_exact_fraction": float(
                scaffold_evaluation.per_prefix_exact[:, order]
                .float()
                .mean()
                .item()
            ),
        }
        for order in range(Config.INPUT_BITS)
    ])

    endpoint = BatchedMLP(
        Config.INPUT_BITS, 1, seed_indices, device
    )
    copy_endpoint_head(endpoint, scaffold, Config.INPUT_BITS - 1)
    endpoint_optimizer = make_optimizer(endpoint)
    transfer_scaffold_optimizer(
        scaffold_optimizer,
        endpoint_optimizer,
        scaffold,
        endpoint,
        Config.INPUT_BITS - 1,
    )

    print("\n--- remove scaffold: endpoint-only settle ---", flush=True)
    settle_rows = settle_endpoint(
        endpoint, endpoint_optimizer, inputs, final_targets
    )
    write_csv(output_dir / "endpoint_settle_trajectory.csv", settle_rows)
    endpoint_evaluation = evaluate_binary(endpoint, inputs, final_targets)
    anchor_mask = (
        endpoint_evaluation.exact
        & (endpoint_evaluation.loss <= Config.ANCHOR_REQUIRED_LOSS)
    )
    anchor_indices = torch.nonzero(
        anchor_mask, as_tuple=False
    ).flatten().cpu().numpy()
    anchor_indices = anchor_indices[: Config.MAX_ANCHORS]
    anchor_rows = [
        {
            "seed": int(index),
            "loss": float(endpoint_evaluation.loss[index].item()),
            "error_count": int(
                endpoint_evaluation.error_count[index].item()
            ),
            "eligible": bool(anchor_mask[index].item()),
        }
        for index in range(Config.SEED_COUNT)
    ]
    write_csv(output_dir / "anchor_summary.csv", anchor_rows)

    recovery_rows: list[dict[str, Any]] = []
    recovery_model_rows: list[dict[str, Any]] = []
    radius_summaries: list[dict[str, Any]] = []
    if len(anchor_indices):
        print(
            f"\n--- perturb and recover: anchors={anchor_indices.tolist()} ---",
            flush=True,
        )
        for radius in Config.RELATIVE_L2_RADII:
            aggregate, per_model, radius_summary = run_recovery_radius(
                float(radius),
                endpoint,
                endpoint_optimizer,
                anchor_indices,
                inputs,
                final_targets,
                device,
            )
            recovery_rows.extend(aggregate)
            recovery_model_rows.extend(per_model)
            radius_summaries.append(radius_summary)
            write_csv(
                output_dir / "recovery_trajectory.csv", recovery_rows
            )
            write_csv(
                output_dir / "recovery_model_trajectory.csv",
                recovery_model_rows,
            )
            write_csv(
                output_dir / "radius_summary.csv", radius_summaries
            )
    else:
        print(
            "没有 scaffold endpoint anchor 满足 exact/loss 要求，跳过扰动恢复。",
            flush=True,
        )

    plot_results(
        output_dir,
        direct_rows,
        scaffold_rows,
        settle_rows,
        recovery_rows,
        radius_summaries,
    )
    direct_final = direct_rows[-1]
    scaffold_final = scaffold_rows[-1]
    settle_final = settle_rows[-1]
    summary = {
        "status": "completed",
        "direct_final": direct_final,
        "scaffold_final": scaffold_final,
        "endpoint_settle_final": settle_final,
        "anchor_count": len(anchor_indices),
        "anchor_indices": anchor_indices.tolist(),
        "radius_summaries": radius_summaries,
        "interpretation": {
            "direct_low_scaffold_high": (
                "同一架构能够表示 parity；随机初始化失败主要是全局可达性。"
            ),
            "settle_stable": (
                "撤去中间监督后，精确 parity 是 endpoint-only loss 下的稳定解。"
            ),
            "broken_recovers": (
                "被扰动到 hard function 已改变后仍能返回，说明 endpoint 解具有有限局部吸引域。"
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    archive_path: Path | None = None
    if Config.PACKAGE_RESULTS:
        archive_path = create_archive(output_dir)
    print("\n=== 实验完成 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    if archive_path is not None:
        print(f"下载压缩包：{archive_path}", flush=True)


if __name__ == "__main__":
    main()
