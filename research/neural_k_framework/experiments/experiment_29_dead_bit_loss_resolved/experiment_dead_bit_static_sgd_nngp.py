"""Dead-bit反事实上的静态SMC、真实优化器与NNGP三方对照。

输入由一个dead bit z和三个有效bit组成。训练集完整覆盖三个有效bit的8个状态，
但z恒为0；测试保持有效bit不变，仅把z改为正值。外部生成器规定目标函数与z
无关，但这项不变性不进入训练loss。

主判据不是哪种方法必须完美不变，而是：在相同Gaussian参数坐标、相同架构和
matched train loss下，静态低-loss参数质量能否预测无weight-decay Adam多seed
在反事实分支上的函数分布。显式L2仅作为MAP机制对照；MC-NNGP-like kernel
作为只保留prior输出二阶结构的基线。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    PROTOCOL_VERSION = "dead_bit_static_sgd_nngp_v1"
    ACTIVE_BITS = 3
    INPUT_BITS = 4
    WIDTH = 16
    SHIFT_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
    FUNCTION_NAMES = (
        "projection_x0",
        "xor_x1_x2",
        "majority3",
        "parity3",
        "mux_x0_x1_x2",
    )
    PARAMETER_COORDINATES = (
        "iid_standard_Gaussian_with_fan_in_forward_scaling"
    )

    SMC_REPLICAS = 4
    SMC_PARTICLES = 2_048
    SMC_SURVIVAL_QUANTILE = 0.5
    SMC_TARGET_THRESHOLDS = (
        0.50, 0.20, 0.10, 0.05, 0.02,
        0.01, 0.005, 0.002, 0.001,
    )
    SMC_MUTATION_SWEEPS = 8
    SMC_FINAL_SWEEPS = 12
    SMC_INITIAL_PROPOSAL_SCALE = 0.10
    SMC_MIN_PROPOSAL_SCALE = 2e-4
    SMC_MAX_PROPOSAL_SCALE = 0.8
    SMC_TARGET_ACCEPTANCE = 0.30
    SMC_ADAPT_RATE = 0.30
    SMC_MAX_LEVELS = 2_000

    OPTIMIZER_SEEDS = 512
    OPTIMIZER_STEPS = 10_000
    OPTIMIZER_LR = 3e-3
    OPTIMIZER_EVAL_INTERVAL = 250
    OPTIMIZER_PROTOCOLS = ("adam_no_decay", "adam_l2")
    L2_COEFFICIENT = 1e-3
    OPTIMIZER_SEED = 2026090601
    PRIMARY_MEAN_ABSOLUTE_GAP_MAX = 0.10
    PRIMARY_WORST_ABSOLUTE_GAP_MAX = 0.20

    NNGP_NETWORKS = 131_072
    NNGP_CHUNK = 8_192
    NNGP_RIDGES = (
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
    )
    NNGP_SEED = 2026090602
    SMC_SEED = 2026090603

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    RESULT_DIR = Path(
        "/root/autodl-tmp/results_dead_bit_static_sgd_nngp"
    )
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


def configure_smoke() -> None:
    Config.PROTOCOL_VERSION += "_smoke"
    Config.FUNCTION_NAMES = Config.FUNCTION_NAMES[:2]
    Config.SHIFT_VALUES = (0.0, 1.0)
    Config.SMC_REPLICAS = 2
    Config.SMC_PARTICLES = 64
    Config.SMC_TARGET_THRESHOLDS = (0.9, 0.8, 0.7)
    Config.SMC_MUTATION_SWEEPS = 1
    Config.SMC_FINAL_SWEEPS = 1
    Config.SMC_MAX_LEVELS = 20
    Config.OPTIMIZER_SEEDS = 8
    Config.OPTIMIZER_STEPS = 5
    Config.OPTIMIZER_EVAL_INTERVAL = 1
    Config.NNGP_NETWORKS = 128
    Config.NNGP_CHUNK = 32
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_dead_bit_static_sgd_nngp"
    )
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parameter_count() -> int:
    return (
        Config.INPUT_BITS*Config.WIDTH+Config.WIDTH
        + Config.WIDTH+1
    )


def active_inputs() -> torch.Tensor:
    values = torch.arange(1 << Config.ACTIVE_BITS, dtype=torch.int64)
    shifts = torch.arange(
        Config.ACTIVE_BITS-1, -1, -1, dtype=torch.int64
    )
    return ((values[:, None] >> shifts[None]) & 1).to(torch.float32)


def shifted_inputs(device: torch.device) -> torch.Tensor:
    active = active_inputs().to(device)
    result = []
    for value in Config.SHIFT_VALUES:
        dead = torch.full(
            (len(active), 1),
            float(value),
            dtype=torch.float32,
            device=device,
        )
        result.append(torch.cat([dead, active], dim=1))
    return torch.stack(result)


def function_targets(name: str, device: torch.device) -> torch.Tensor:
    bits = active_inputs().to(device)
    x0, x1, x2 = bits[:, 0], bits[:, 1], bits[:, 2]
    if name == "projection_x0":
        target = x0
    elif name == "xor_x1_x2":
        target = torch.remainder(x1+x2, 2.0)
    elif name == "majority3":
        target = (x0+x1+x2 >= 2.0).to(torch.float32)
    elif name == "parity3":
        target = torch.remainder(x0+x1+x2, 2.0)
    elif name == "mux_x0_x1_x2":
        target = torch.where(x0.bool(), x1, x2)
    else:
        raise ValueError(f"未知函数：{name}")
    return target


def forward_normalized(
    parameters: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    count = len(parameters)
    cursor = 0
    size = Config.INPUT_BITS*Config.WIDTH
    first_weight = parameters[:, cursor:cursor+size].reshape(
        count, Config.WIDTH, Config.INPUT_BITS
    )/math.sqrt(Config.INPUT_BITS)
    cursor += size
    first_bias = parameters[:, cursor:cursor+Config.WIDTH]/math.sqrt(
        Config.INPUT_BITS
    )
    cursor += Config.WIDTH
    hidden = torch.tanh(
        torch.bmm(
            inputs[None].expand(count, -1, -1),
            first_weight.transpose(1, 2),
        )
        + first_bias[:, None]
    )
    output_weight = parameters[
        :, cursor:cursor+Config.WIDTH
    ]/math.sqrt(Config.WIDTH)
    cursor += Config.WIDTH
    output_bias = parameters[:, cursor:cursor+1]/math.sqrt(Config.WIDTH)
    cursor += 1
    if cursor != parameter_count():
        raise AssertionError("参数游标错误。")
    return torch.bmm(
        hidden,
        output_weight[:, :, None],
    ).squeeze(2)+output_bias


def train_loss(
    parameters: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    inputs = shifted_inputs(parameters.device)[0]
    logits = forward_normalized(parameters, inputs)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets[None].expand_as(logits),
        reduction="none",
    ).mean(dim=1)


@torch.inference_mode()
def population_observables(
    parameters: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    inputs = shifted_inputs(parameters.device)
    shifts, states, _ = inputs.shape
    logits = forward_normalized(
        parameters,
        inputs.reshape(shifts*states, Config.INPUT_BITS),
    ).reshape(len(parameters), shifts, states)
    hard = logits >= 0
    target_bits = targets.bool()[None, None]
    correct = hard == target_bits
    exact = torch.all(correct, dim=2)
    base = hard[:, 0]
    invariant = torch.all(hard == base[:, None], dim=2)
    sigmoid_sum = torch.sigmoid(logits).sum(dim=0)
    predictive = sigmoid_sum/len(parameters)
    predictive_hard = predictive >= 0.5
    predictive_correct = predictive_hard == targets.bool()[None]
    dead_column = parameters[
        :, :Config.INPUT_BITS*Config.WIDTH
    ].reshape(len(parameters), Config.WIDTH, Config.INPUT_BITS)[:, :, 0]
    losses = train_loss(parameters, targets)
    row: dict[str, Any] = {
        "particle_count": len(parameters),
        "mean_train_loss": float(losses.mean().item()),
        "median_train_loss": float(losses.median().item()),
        "train_exact_mass": float(exact[:, 0].float().mean().item()),
        "dead_weight_mean": float(dead_column.mean().item()),
        "dead_weight_variance": float(
            dead_column.to(torch.float64).var(unbiased=True).item()
        ),
        "dead_weight_mean_squared_norm": float(
            dead_column.square().sum(dim=1).mean().item()
        ),
    }
    for index, shift in enumerate(Config.SHIFT_VALUES):
        key = f"z{float(shift):g}"
        row[f"{key}_mean_particle_accuracy"] = float(
            correct[:, index].float().mean().item()
        )
        row[f"{key}_strict_correct_mass"] = float(
            exact[:, index].float().mean().item()
        )
        row[f"{key}_particle_invariance_mass"] = float(
            invariant[:, index].float().mean().item()
        )
        row[f"{key}_predictive_accuracy"] = float(
            predictive_correct[index].float().mean().item()
        )
        row[f"{key}_predictive_exact"] = bool(
            torch.all(predictive_correct[index]).item()
        )
    artifacts = {
        "sigmoid_sum": sigmoid_sum.cpu().numpy(),
        "hard_correct_counts": correct.sum(dim=0).cpu().numpy(),
        "strict_correct_counts": exact.sum(dim=0).cpu().numpy(),
        "invariance_counts": invariant.sum(dim=0).cpu().numpy(),
    }
    return row, artifacts


def pcn_mutate(
    particles: torch.Tensor,
    losses: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    proposal_scale: float,
    sweeps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    rates = []
    scale = float(proposal_scale)
    for _ in range(sweeps):
        noise = torch.randn(
            particles.shape,
            dtype=particles.dtype,
            device=particles.device,
            generator=generator,
        )
        proposal = (
            math.sqrt(max(1.0-scale*scale, 0.0))*particles
            + scale*noise
        )
        proposal_losses = train_loss(proposal, targets)
        accepted = proposal_losses <= float(threshold)+1e-7
        particles = torch.where(
            accepted[:, None], proposal, particles
        )
        losses = torch.where(accepted, proposal_losses, losses)
        rate = float(accepted.float().mean().item())
        rates.append(rate)
        scale *= math.exp(
            Config.SMC_ADAPT_RATE*(rate-Config.SMC_TARGET_ACCEPTANCE)
        )
        scale = float(np.clip(
            scale,
            Config.SMC_MIN_PROPOSAL_SCALE,
            Config.SMC_MAX_PROPOSAL_SCALE,
        ))
    return particles, losses, scale, float(np.mean(rates))


def run_smc_replica(
    function_name: str,
    replica: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    targets = function_targets(function_name, device)
    generator = torch.Generator(device=device)
    function_index = Config.FUNCTION_NAMES.index(function_name)
    generator.manual_seed(
        Config.SMC_SEED+function_index*10_000+replica
    )
    particles = torch.randn(
        Config.SMC_PARTICLES,
        parameter_count(),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    losses = train_loss(particles, targets)
    proposal_scale = Config.SMC_INITIAL_PROPOSAL_SCALE
    current_threshold = float("inf")
    rows: list[dict[str, Any]] = []
    level = 0
    log_volume = 0.0
    for target_threshold in Config.SMC_TARGET_THRESHOLDS:
        while current_threshold > float(target_threshold)+1e-12:
            if level >= Config.SMC_MAX_LEVELS:
                raise RuntimeError(
                    f"{function_name} replica={replica}超过SMC层数上限。"
                )
            quantile = float(torch.quantile(
                losses, Config.SMC_SURVIVAL_QUANTILE
            ).item())
            next_threshold = max(float(target_threshold), quantile)
            if math.isfinite(current_threshold):
                next_threshold = min(next_threshold, current_threshold-1e-8)
            survivors = losses <= next_threshold+1e-7
            survivor_indices = torch.nonzero(
                survivors, as_tuple=False
            ).flatten()
            if len(survivor_indices) == 0:
                raise RuntimeError("SMC没有survivor。")
            survival = len(survivor_indices)/len(particles)
            parent_local = torch.randint(
                0,
                len(survivor_indices),
                (len(particles),),
                device=device,
                generator=generator,
            )
            parents = survivor_indices[parent_local]
            particles = particles[parents].clone()
            losses = losses[parents].clone()
            log_volume += math.log(survival)
            particles, losses, proposal_scale, acceptance = pcn_mutate(
                particles,
                losses,
                targets,
                next_threshold,
                proposal_scale,
                Config.SMC_MUTATION_SWEEPS,
                generator,
            )
            current_threshold = next_threshold
            level += 1
            if level == 1 or level % 10 == 0 or math.isclose(
                current_threshold, float(target_threshold),
                rel_tol=1e-8, abs_tol=1e-10,
            ):
                print(
                    f"SMC {function_name} r={replica} level={level:>3} | "
                    f"eps={current_threshold:.4g} | "
                    f"L={losses.mean().item():.4g} | "
                    f"survive={survival:.3f} | pCN={acceptance:.3f}",
                    flush=True,
                )
        particles, losses, proposal_scale, acceptance = pcn_mutate(
            particles,
            losses,
            targets,
            float(target_threshold),
            proposal_scale,
            Config.SMC_FINAL_SWEEPS,
            generator,
        )
        observation, artifacts = population_observables(
            particles, targets
        )
        row = {
            "function": function_name,
            "replica": replica,
            "epsilon": float(target_threshold),
            "level": level,
            "log_volume_fraction_estimate": log_volume,
            "proposal_scale": proposal_scale,
            "final_mutation_acceptance": acceptance,
            **observation,
            "_sigmoid_sum": artifacts["sigmoid_sum"],
            "_hard_correct_counts": artifacts["hard_correct_counts"],
            "_strict_correct_counts": artifacts[
                "strict_correct_counts"
            ],
            "_invariance_counts": artifacts["invariance_counts"],
        }
        rows.append(row)
    return rows


def aggregate_smc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = []
    for function_name in Config.FUNCTION_NAMES:
        local_function = [
            row for row in rows if row["function"] == function_name
        ]
        for epsilon in Config.SMC_TARGET_THRESHOLDS:
            local = [
                row for row in local_function
                if math.isclose(row["epsilon"], float(epsilon))
            ]
            if len(local) != Config.SMC_REPLICAS:
                raise RuntimeError("SMC replica数量不完整。")
            total = Config.SMC_REPLICAS*Config.SMC_PARTICLES
            sigmoid_sum = sum(row["_sigmoid_sum"] for row in local)
            hard_counts = sum(
                row["_hard_correct_counts"] for row in local
            )
            strict_counts = sum(
                row["_strict_correct_counts"] for row in local
            )
            invariance_counts = sum(
                row["_invariance_counts"] for row in local
            )
            targets = function_targets(
                function_name, torch.device("cpu")
            ).bool().numpy()
            predictive = sigmoid_sum/total
            predictive_hard = predictive >= 0.5
            row_out: dict[str, Any] = {
                "function": function_name,
                "epsilon": float(epsilon),
                "particle_count": total,
                "mean_train_loss": float(np.mean([
                    row["mean_train_loss"] for row in local
                ])),
                "mean_train_loss_replica_range": float(
                    max(row["mean_train_loss"] for row in local)
                    - min(row["mean_train_loss"] for row in local)
                ),
                "train_exact_mass": float(
                    sum(
                        row["train_exact_mass"] for row in local
                    )/len(local)
                ),
                "dead_weight_mean": float(np.mean([
                    row["dead_weight_mean"] for row in local
                ])),
                "dead_weight_variance": float(np.mean([
                    row["dead_weight_variance"] for row in local
                ])),
                "dead_weight_mean_squared_norm": float(np.mean([
                    row["dead_weight_mean_squared_norm"] for row in local
                ])),
                "final_mutation_acceptance_mean": float(np.mean([
                    row["final_mutation_acceptance"] for row in local
                ])),
            }
            for index, shift in enumerate(Config.SHIFT_VALUES):
                key = f"z{float(shift):g}"
                row_out[f"{key}_mean_particle_accuracy"] = float(
                    hard_counts[index].sum()/(total*len(targets))
                )
                row_out[f"{key}_strict_correct_mass"] = float(
                    strict_counts[index]/total
                )
                row_out[f"{key}_particle_invariance_mass"] = float(
                    invariance_counts[index]/total
                )
                correct = predictive_hard[index] == targets
                row_out[f"{key}_predictive_accuracy"] = float(
                    np.mean(correct)
                )
                row_out[f"{key}_predictive_exact"] = bool(
                    np.all(correct)
                )
                replica_predictive = []
                for replica_row in local:
                    local_predictive = (
                        replica_row["_sigmoid_sum"][index]
                        / Config.SMC_PARTICLES
                    ) >= 0.5
                    replica_predictive.append(local_predictive)
                pairwise = []
                for left in range(len(replica_predictive)):
                    for right in range(left+1, len(replica_predictive)):
                        pairwise.append(float(np.mean(
                            replica_predictive[left]
                            != replica_predictive[right]
                        )))
                row_out[f"{key}_replica_predictive_disagreement"] = (
                    float(np.mean(pairwise)) if pairwise else 0.0
                )
            aggregate.append(row_out)
    return aggregate


def initialize_optimizer_parameters(
    function_index: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.OPTIMIZER_SEED+function_index)
    return torch.randn(
        Config.OPTIMIZER_SEEDS,
        parameter_count(),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def run_optimizer(
    function_name: str,
    protocol: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    function_index = Config.FUNCTION_NAMES.index(function_name)
    targets = function_targets(function_name, device)
    initial = initialize_optimizer_parameters(function_index, device)
    parameters = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([parameters], lr=Config.OPTIMIZER_LR)
    initial_dead = initial[
        :, :Config.INPUT_BITS*Config.WIDTH
    ].reshape(
        Config.OPTIMIZER_SEEDS,
        Config.WIDTH,
        Config.INPUT_BITS,
    )[:, :, 0].clone()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(Config.OPTIMIZER_STEPS+1):
        if (
            step == 0
            or step % Config.OPTIMIZER_EVAL_INTERVAL == 0
            or step == Config.OPTIMIZER_STEPS
        ):
            observation, _ = population_observables(
                parameters, targets
            )
            dead_now = parameters[
                :, :Config.INPUT_BITS*Config.WIDTH
            ].reshape(
                Config.OPTIMIZER_SEEDS,
                Config.WIDTH,
                Config.INPUT_BITS,
            )[:, :, 0]
            row = {
                "function": function_name,
                "protocol": protocol,
                "step": step,
                "elapsed_seconds": time.perf_counter()-started,
                "dead_column_max_absolute_change": float(
                    torch.max(torch.abs(dead_now-initial_dead)).item()
                ),
                **observation,
            }
            rows.append(row)
            if step % max(Config.OPTIMIZER_EVAL_INTERVAL*4, 1) == 0:
                print(
                    f"{protocol} {function_name} step={step:>5} | "
                    f"L={observation['mean_train_loss']:.4g} | "
                    f"train={observation['train_exact_mass']:.3f} | "
                    f"z1={observation.get('z1_strict_correct_mass', float('nan')):.3f}",
                    flush=True,
                )
        if step == Config.OPTIMIZER_STEPS:
            break
        loss_by_seed = train_loss(parameters, targets)
        objective = loss_by_seed.sum()
        if protocol == "adam_l2":
            objective = objective+(
                0.5*Config.L2_COEFFICIENT*parameters.square().sum()
            )
        elif protocol != "adam_no_decay":
            raise ValueError(f"未知optimizer协议：{protocol}")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    summary = {
        "function": function_name,
        "protocol": protocol,
        "seed_count": Config.OPTIMIZER_SEEDS,
        "final_step": Config.OPTIMIZER_STEPS,
        "elapsed_seconds": time.perf_counter()-started,
        "final": rows[-1],
    }
    if protocol == "adam_no_decay":
        dead_change = float(
            rows[-1]["dead_column_max_absolute_change"]
        )
        if dead_change > 1e-7:
            raise RuntimeError(
                "无weight-decay协议的dead列发生变化，实验实现有误："
                f"{dead_change}"
            )
    return summary, rows


@torch.inference_mode()
def run_nngp(device: torch.device) -> tuple[list[dict[str, Any]], np.ndarray]:
    inputs = shifted_inputs(device).reshape(
        -1, Config.INPUT_BITS
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.NNGP_SEED)
    gram = torch.zeros(
        len(inputs), len(inputs), dtype=torch.float64, device=device
    )
    completed = 0
    while completed < Config.NNGP_NETWORKS:
        count = min(Config.NNGP_CHUNK, Config.NNGP_NETWORKS-completed)
        parameters = torch.randn(
            count,
            parameter_count(),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        logits = forward_normalized(parameters, inputs).to(torch.float64)
        gram += logits.transpose(0, 1)@logits
        completed += count
        if completed % max(Config.NNGP_CHUNK*4, 1) == 0:
            print(
                f"NNGP prior networks {completed}/{Config.NNGP_NETWORKS}",
                flush=True,
            )
    kernel = (gram/Config.NNGP_NETWORKS).cpu().numpy()
    train_indices = np.arange(8, dtype=np.int64)
    rows = []
    for function_name in Config.FUNCTION_NAMES:
        targets = function_targets(
            function_name, torch.device("cpu")
        ).numpy()
        signs = targets*2.0-1.0
        for ridge in Config.NNGP_RIDGES:
            system = (
                kernel[np.ix_(train_indices, train_indices)]
                + float(ridge)*np.eye(len(train_indices))
            )
            coefficients = np.linalg.solve(system, signs)
            means = (
                kernel[:, train_indices]@coefficients
            ).reshape(len(Config.SHIFT_VALUES), 8)
            hard = means >= 0
            row: dict[str, Any] = {
                "function": function_name,
                "ridge": float(ridge),
            }
            for index, shift in enumerate(Config.SHIFT_VALUES):
                key = f"z{float(shift):g}"
                row[f"{key}_accuracy"] = float(np.mean(
                    hard[index] == targets.astype(bool)
                ))
                row[f"{key}_exact"] = bool(np.all(
                    hard[index] == targets.astype(bool)
                ))
                row[f"{key}_invariant"] = bool(np.all(
                    hard[index] == hard[0]
                ))
            rows.append(row)
    return rows, kernel


def matched_loss_rows(
    smc_rows: list[dict[str, Any]],
    optimizer_summaries: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for optimizer in optimizer_summaries:
        function_name = optimizer["function"]
        candidates = [
            row for row in smc_rows if row["function"] == function_name
        ]
        trajectory = [
            row for row in optimizer_rows
            if row["function"] == function_name
            and row["protocol"] == optimizer["protocol"]
        ]
        if optimizer["protocol"] == "adam_no_decay":
            matched = min(candidates, key=lambda row: row["epsilon"])
            optimizer_observation = min(
                trajectory,
                key=lambda row: abs(
                    math.log(max(row["mean_train_loss"], 1e-300))
                    - math.log(max(
                        matched["mean_train_loss"], 1e-300
                    ))
                ),
            )
        else:
            optimizer_observation = optimizer["final"]
            target_loss = float(
                optimizer_observation["mean_train_loss"]
            )
            matched = min(
                candidates,
                key=lambda row: abs(
                    math.log(max(row["mean_train_loss"], 1e-300))
                    - math.log(max(target_loss, 1e-300))
                ),
            )
        target_loss = float(optimizer_observation["mean_train_loss"])
        row_out = {
            "function": function_name,
            "optimizer_protocol": optimizer["protocol"],
            "optimizer_step": optimizer_observation["step"],
            "optimizer_train_loss": target_loss,
            "matched_smc_epsilon": matched["epsilon"],
            "matched_smc_mean_train_loss": matched["mean_train_loss"],
            "matched_log_loss_distance": abs(
                math.log(max(target_loss, 1e-300))
                - math.log(max(matched["mean_train_loss"], 1e-300))
            ),
        }
        for shift in Config.SHIFT_VALUES:
            key = f"z{float(shift):g}"
            row_out[f"{key}_optimizer_strict_mass"] = (
                optimizer_observation[f"{key}_strict_correct_mass"]
            )
            row_out[f"{key}_smc_strict_mass"] = matched[
                f"{key}_strict_correct_mass"
            ]
            row_out[f"{key}_strict_mass_gap"] = (
                row_out[f"{key}_optimizer_strict_mass"]
                - row_out[f"{key}_smc_strict_mass"]
            )
            row_out[f"{key}_optimizer_predictive_accuracy"] = (
                optimizer_observation[f"{key}_predictive_accuracy"]
            )
            row_out[f"{key}_smc_predictive_accuracy"] = matched[
                f"{key}_predictive_accuracy"
            ]
        rows.append(row_out)
    return rows


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent/f"{result_dir.name}_package.zip"
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
    if Config.SMOKE_TEST:
        configure_smoke()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch不可见。")
    torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    protocol = {
        "protocol_version": Config.PROTOCOL_VERSION,
        "task": (
            "z is zero on all 3-bit truth-table training states; "
            "the target is invariant when z is shifted at test time"
        ),
        "function_names": Config.FUNCTION_NAMES,
        "shift_values": Config.SHIFT_VALUES,
        "network": f"4->{Config.WIDTH}->1 tanh",
        "parameter_count": parameter_count(),
        "parameter_coordinates": Config.PARAMETER_COORDINATES,
        "smc": {
            "replicas": Config.SMC_REPLICAS,
            "particles": Config.SMC_PARTICLES,
            "thresholds": Config.SMC_TARGET_THRESHOLDS,
            "mutation": "prior-preserving constrained pCN",
        },
        "optimizer": {
            "seed_count": Config.OPTIMIZER_SEEDS,
            "steps": Config.OPTIMIZER_STEPS,
            "learning_rate": Config.OPTIMIZER_LR,
            "protocols": Config.OPTIMIZER_PROTOCOLS,
            "l2_coefficient": Config.L2_COEFFICIENT,
        },
        "nngp": {
            "prior_networks": Config.NNGP_NETWORKS,
            "ridges": Config.NNGP_RIDGES,
            "kind": "Monte Carlo finite-architecture prior-output covariance",
        },
        "primary_comparison": (
            "adam_no_decay versus static SMC at matched mean train loss; "
            "adam_l2 is a mechanism control, not a required static match"
        ),
        "primary_preregistered_pass": {
            "mean_absolute_z1_strict_mass_gap_max": (
                Config.PRIMARY_MEAN_ABSOLUTE_GAP_MAX
            ),
            "worst_absolute_z1_strict_mass_gap_max": (
                Config.PRIMARY_WORST_ABSOLUTE_GAP_MAX
            ),
        },
        "full_label_boundary": (
            "Only z=0 labels enter SMC losses, optimizer gradients and NNGP fit. "
            "z>0 labels are used only after predictions are frozen."
        ),
    }
    protocol_sha256 = hashlib.sha256(json.dumps(
        json_ready(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    protocol["protocol_sha256"] = protocol_sha256
    write_json(Config.RESULT_DIR/"protocol.json", protocol)
    np.savez_compressed(
        Config.RESULT_DIR/"dataset.npz",
        inputs=shifted_inputs(torch.device("cpu")).numpy(),
        targets=np.stack([
            function_targets(name, torch.device("cpu")).numpy()
            for name in Config.FUNCTION_NAMES
        ]),
        function_names=np.asarray(Config.FUNCTION_NAMES),
        shift_values=np.asarray(Config.SHIFT_VALUES),
    )

    print(
        "=== Dead-bit static / optimizer / NNGP comparison ===\n"
        f"device={device} | gpu="
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} | "
        f"network=4->{Config.WIDTH}->1 | params={parameter_count()}\n"
        f"functions={Config.FUNCTION_NAMES} | shifts={Config.SHIFT_VALUES}",
        flush=True,
    )
    started = time.perf_counter()

    smc_replica_rows: list[dict[str, Any]] = []
    for function_name in Config.FUNCTION_NAMES:
        for replica in range(Config.SMC_REPLICAS):
            smc_replica_rows.extend(
                run_smc_replica(function_name, replica, device)
            )
    smc_rows = aggregate_smc(smc_replica_rows)
    serializable_replica_rows = [{
        key: value for key, value in row.items() if not key.startswith("_")
    } for row in smc_replica_rows]
    write_csv(Config.RESULT_DIR/"smc_replicas.csv", serializable_replica_rows)
    write_csv(Config.RESULT_DIR/"smc_aggregate.csv", smc_rows)

    optimizer_summaries = []
    optimizer_rows = []
    for function_name in Config.FUNCTION_NAMES:
        for optimizer_protocol in Config.OPTIMIZER_PROTOCOLS:
            summary, rows = run_optimizer(
                function_name, optimizer_protocol, device
            )
            optimizer_summaries.append(summary)
            optimizer_rows.extend(rows)
    write_csv(Config.RESULT_DIR/"optimizer_trajectories.csv", optimizer_rows)
    write_json(
        Config.RESULT_DIR/"optimizer_summaries.json",
        optimizer_summaries,
    )

    nngp_rows, kernel = run_nngp(device)
    write_csv(Config.RESULT_DIR/"nngp_predictions.csv", nngp_rows)
    np.savez_compressed(
        Config.RESULT_DIR/"nngp_kernel.npz",
        kernel=kernel,
    )

    matched = matched_loss_rows(
        smc_rows, optimizer_summaries, optimizer_rows
    )
    write_csv(Config.RESULT_DIR/"matched_loss_comparison.csv", matched)
    no_decay_rows = [
        row for row in matched
        if row["optimizer_protocol"] == "adam_no_decay"
    ]
    z1_gaps = [
        abs(float(row.get("z1_strict_mass_gap", 0.0)))
        for row in no_decay_rows
        if "z1_strict_mass_gap" in row
    ]
    gap_mean = float(np.mean(z1_gaps)) if z1_gaps else None
    gap_max = float(np.max(z1_gaps)) if z1_gaps else None
    primary_pass = bool(
        gap_mean is not None
        and gap_max is not None
        and gap_mean <= Config.PRIMARY_MEAN_ABSOLUTE_GAP_MAX
        and gap_max <= Config.PRIMARY_WORST_ABSOLUTE_GAP_MAX
    )
    summary = {
        "status": (
            "completed"
            if primary_pass else "completed_with_primary_mismatch"
        ),
        "elapsed_seconds": time.perf_counter()-started,
        "protocol_sha256": protocol_sha256,
        "smc_conditions": len(smc_rows),
        "optimizer_conditions": len(optimizer_summaries),
        "nngp_conditions": len(nngp_rows),
        "primary_no_decay_z1_absolute_gap_mean": gap_mean,
        "primary_no_decay_z1_absolute_gap_max": gap_max,
        "primary_static_prediction_pass": primary_pass,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        "interpretation_boundary": [
            "A dead input direction is not identified by the training data.",
            "Static SMC is not required to match the L2/MAP control.",
            "A mismatch with no-decay Adam measures optimizer reweighting beyond "
            "matched-loss static Gaussian mass; it does not by itself invalidate "
            "the full framework.",
        ],
    }
    write_json(Config.RESULT_DIR/"summary.json", summary)
    print(
        f"FINAL elapsed={format_duration(summary['elapsed_seconds'])} | "
        f"no-decay z1 gap mean="
        f"{summary['primary_no_decay_z1_absolute_gap_mean']}",
        flush=True,
    )
    if Config.PACKAGE_RESULTS:
        print(f"下载压缩包：{create_archive(Config.RESULT_DIR)}", flush=True)


if __name__ == "__main__":
    main()
