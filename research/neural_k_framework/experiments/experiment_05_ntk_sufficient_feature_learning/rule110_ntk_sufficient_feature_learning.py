"""
检验：当初始化 NTK 已足以学习一层 Rule 110 时，端到端网络是否仍会特征学习。

实验分为三层：
1. 从仓库旧实验的 ReLU arc-cosine 公式递推三层无限宽 NTK，
   对完整 30-bit 输出做 KRR；
2. 对同一个有限宽网络计算真实经验 NTK，对选定输出位做初始化 KRR；
3. 端到端训练网络，在函数达到 100% 后继续跟踪经验 NTK、隐藏 CK 和 gate。

脚本不依赖外部数据文件，复制到 AutoDL notebook 的单个代码单元即可运行。
通常只需要修改 Config。默认完整配置面向 32 GB 显存的 RTX 5090。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    # 本地检查时可改成 True；完整实验保持 False。
    SMOKE_TEST = False

    RESULT_DIR = Path("/root/results_rule110_ntk_sufficient_feature_learning_v1")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_SEED = 123
    LENGTH = 30
    RULE = 110
    TRAIN_COUNT = 8_000
    TEST_COUNT = 20_000
    PROBE_COUNT = 128

    WIDTH = 1_024
    HIDDEN_LAYERS = 3
    MODEL_SEEDS = (0, 1, 2)
    SELECTED_OUTPUT_BITS = (0, 15, 29)

    # BCE 检查常规分类训练；MSE 排除无界 margin 带来的伪 kernel 漂移。
    LOSS_MODES = ("bce", "mse")
    OPTIMIZER = "adam"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 1_024
    TRAIN_STEPS = 30_000
    SNAPSHOT_STEPS = (0, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000)
    EVAL_BATCH_SIZE = 4_096

    KERNEL_RIDGE_FACTOR = 1e-5
    KERNEL_TEST_CHUNK = 2_000
    RUN_ANALYTIC_NTK = True
    RUN_EMPIRICAL_NTK = True
    RUN_EMPIRICAL_FORMULA_SELF_CHECK = True

    SAVE_MODELS = True
    SKIP_COMPLETED_RUNS = True
    CREATE_ZIP = True

    # 仅用于自动摘要，不作为统计显著性阈值。
    STRUCTURAL_CKA_CHANGE_THRESHOLD = 1e-3
    GATE_FLIP_THRESHOLD = 1e-3


@dataclass
class ResolvedConfig:
    smoke_test: bool
    result_dir: Path
    device: str
    data_seed: int
    length: int
    rule: int
    train_count: int
    test_count: int
    probe_count: int
    width: int
    hidden_layers: int
    model_seeds: tuple[int, ...]
    selected_output_bits: tuple[int, ...]
    loss_modes: tuple[str, ...]
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    train_steps: int
    snapshot_steps: tuple[int, ...]
    eval_batch_size: int
    kernel_ridge_factor: float
    kernel_test_chunk: int
    run_analytic_ntk: bool
    run_empirical_ntk: bool
    run_empirical_formula_self_check: bool
    save_models: bool
    skip_completed_runs: bool
    create_zip: bool
    structural_cka_change_threshold: float
    gate_flip_threshold: float


def resolve_config() -> ResolvedConfig:
    if Config.SMOKE_TEST:
        base = Path.cwd() / "results_rule110_ntk_feature_learning_smoke"
        return ResolvedConfig(
            smoke_test=True,
            result_dir=base,
            device=Config.DEVICE,
            data_seed=123,
            length=12,
            rule=110,
            train_count=96,
            test_count=192,
            probe_count=24,
            width=32,
            hidden_layers=3,
            model_seeds=(0,),
            selected_output_bits=(0, 6, 11),
            loss_modes=("bce", "mse"),
            optimizer="adam",
            learning_rate=2e-3,
            weight_decay=0.0,
            batch_size=48,
            train_steps=30,
            snapshot_steps=(0, 1, 3, 10, 30),
            eval_batch_size=192,
            kernel_ridge_factor=1e-4,
            kernel_test_chunk=96,
            run_analytic_ntk=True,
            run_empirical_ntk=True,
            run_empirical_formula_self_check=True,
            save_models=False,
            skip_completed_runs=False,
            create_zip=False,
            structural_cka_change_threshold=1e-3,
            gate_flip_threshold=1e-3,
        )

    return ResolvedConfig(
        smoke_test=False,
        result_dir=Path(Config.RESULT_DIR),
        device=Config.DEVICE,
        data_seed=int(Config.DATA_SEED),
        length=int(Config.LENGTH),
        rule=int(Config.RULE),
        train_count=int(Config.TRAIN_COUNT),
        test_count=int(Config.TEST_COUNT),
        probe_count=int(Config.PROBE_COUNT),
        width=int(Config.WIDTH),
        hidden_layers=int(Config.HIDDEN_LAYERS),
        model_seeds=tuple(int(x) for x in Config.MODEL_SEEDS),
        selected_output_bits=tuple(int(x) for x in Config.SELECTED_OUTPUT_BITS),
        loss_modes=tuple(str(x).lower() for x in Config.LOSS_MODES),
        optimizer=str(Config.OPTIMIZER).lower(),
        learning_rate=float(Config.LEARNING_RATE),
        weight_decay=float(Config.WEIGHT_DECAY),
        batch_size=int(Config.BATCH_SIZE),
        train_steps=int(Config.TRAIN_STEPS),
        snapshot_steps=tuple(int(x) for x in Config.SNAPSHOT_STEPS),
        eval_batch_size=int(Config.EVAL_BATCH_SIZE),
        kernel_ridge_factor=float(Config.KERNEL_RIDGE_FACTOR),
        kernel_test_chunk=int(Config.KERNEL_TEST_CHUNK),
        run_analytic_ntk=bool(Config.RUN_ANALYTIC_NTK),
        run_empirical_ntk=bool(Config.RUN_EMPIRICAL_NTK),
        run_empirical_formula_self_check=bool(
            Config.RUN_EMPIRICAL_FORMULA_SELF_CHECK
        ),
        save_models=bool(Config.SAVE_MODELS),
        skip_completed_runs=bool(Config.SKIP_COMPLETED_RUNS),
        create_zip=bool(Config.CREATE_ZIP),
        structural_cka_change_threshold=float(
            Config.STRUCTURAL_CKA_CHANGE_THRESHOLD
        ),
        gate_flip_threshold=float(Config.GATE_FLIP_THRESHOLD),
    )


class DeepReluMLP(nn.Module):
    """无 bias、He/NTK 参数化的等宽深层 ReLU MLP。"""

    def __init__(
        self,
        input_dim: int,
        width: int,
        hidden_layers: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.hidden_layers = int(hidden_layers)
        self.output_dim = int(output_dim)
        if self.hidden_layers < 1:
            raise ValueError("HIDDEN_LAYERS 至少为 1。")
        weights = [nn.Parameter(torch.randn(width, input_dim))]
        weights.extend(
            nn.Parameter(torch.randn(width, width))
            for _ in range(self.hidden_layers - 1)
        )
        self.hidden_weights = nn.ParameterList(weights)
        self.output_weight = nn.Parameter(torch.randn(output_dim, width))

    def features(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        activations = [x]
        gates = []
        current = x
        for layer_index, weight in enumerate(self.hidden_weights):
            fan_in = self.input_dim if layer_index == 0 else self.width
            pre = current @ weight.t() / math.sqrt(fan_in)
            gates.append(pre > 0)
            current = math.sqrt(2.0) * F.relu(pre)
            activations.append(current)
        return activations, gates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations, _ = self.features(x)
        return activations[-1] @ self.output_weight.t() / math.sqrt(self.width)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(x) for x in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_fingerprint(cfg: ResolvedConfig) -> str:
    payload = json.dumps(json_ready(asdict(cfg)), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_config(cfg: ResolvedConfig) -> None:
    if cfg.rule != 110:
        raise ValueError("当前实验只实现并审计了 Rule 110。")
    if cfg.length < 3:
        raise ValueError("LENGTH 至少为 3。")
    if cfg.train_count <= 0 or cfg.test_count <= 0:
        raise ValueError("训练集和测试集必须非空。")
    if cfg.probe_count > cfg.test_count:
        raise ValueError("PROBE_COUNT 不能超过 TEST_COUNT。")
    if cfg.train_count + cfg.test_count >= 2 ** cfg.length:
        raise ValueError("唯一输入数量超过或等于完整输入空间。")
    if any(bit < 0 or bit >= cfg.length for bit in cfg.selected_output_bits):
        raise ValueError("SELECTED_OUTPUT_BITS 存在越界索引。")
    if any(mode not in {"bce", "mse"} for mode in cfg.loss_modes):
        raise ValueError("LOSS_MODES 只能包含 bce 和 mse。")
    if cfg.optimizer not in {"adam", "sgd"}:
        raise ValueError("OPTIMIZER 只能为 adam 或 sgd。")
    if 0 not in cfg.snapshot_steps or cfg.train_steps not in cfg.snapshot_steps:
        raise ValueError("SNAPSHOT_STEPS 必须包含 0 和 TRAIN_STEPS。")


def generate_unique_states(
    count: int,
    length: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    integers = rng.choice(2 ** length, size=count, replace=False)
    shifts = np.arange(length - 1, -1, -1, dtype=np.int64)
    bits = ((integers[:, None] >> shifts[None, :]) & 1).astype(np.float32)
    return torch.from_numpy(bits)


def evolve_rule110_once(states: torch.Tensor) -> torch.Tensor:
    left = torch.roll(states, shifts=1, dims=1).long()
    center = states.long()
    right = torch.roll(states, shifts=-1, dims=1).long()
    index = (left << 2) | (center << 1) | right
    # 索引顺序为 000..111；Rule 110 的二进制为 01101110。
    table = torch.tensor(
        [0, 1, 1, 1, 0, 1, 1, 0],
        dtype=torch.float32,
        device=states.device,
    )
    return table[index]


def make_dataset(cfg: ResolvedConfig) -> dict[str, torch.Tensor]:
    states = generate_unique_states(
        cfg.train_count + cfg.test_count,
        cfg.length,
        cfg.data_seed,
    )
    targets = evolve_rule110_once(states)
    return {
        "train_x": states[: cfg.train_count],
        "train_y": targets[: cfg.train_count],
        "test_x": states[cfg.train_count :],
        "test_y": targets[cfg.train_count :],
    }


def analytic_relu_ntk(
    x1: torch.Tensor,
    x2: torch.Tensor,
    hidden_layers: int,
) -> torch.Tensor:
    """与 DeepReluMLP 参数化匹配的无限宽深层 ReLU NTK。"""
    input_dim = x1.shape[1]
    sigma = (x1 @ x2.t()) / input_dim
    diagonal1 = (x1.square().sum(dim=1, keepdim=True)) / input_dim
    diagonal2 = (x2.square().sum(dim=1, keepdim=True).t()) / input_dim
    ntk = sigma
    for _ in range(hidden_layers):
        normalizer = torch.sqrt(diagonal1 @ diagonal2).clamp_min(1e-12)
        cosine = (sigma / normalizer).clamp(-1.0, 1.0)
        theta = torch.acos(cosine)
        # sqrt(2) * ReLU 保持每层对角方差不变。
        sigma_next = normalizer * (
            torch.sin(theta) + (math.pi - theta) * cosine
        ) / math.pi
        derivative_kernel = (math.pi - theta) / math.pi
        ntk = sigma_next + ntk * derivative_kernel
        sigma = sigma_next
    return ntk


@torch.no_grad()
def network_components(
    model: DeepReluMLP,
    x: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    activations, boolean_gates = model.features(x)
    gates = [gate.to(activations[0].dtype) for gate in boolean_gates]
    return activations, gates


@torch.no_grad()
def output_sensitivities(
    model: DeepReluMLP,
    gates: list[torch.Tensor],
    output_bit: int,
) -> list[torch.Tensor]:
    """返回每个隐藏层 pre-activation 对指定输出的导数。"""
    sensitivities: list[torch.Tensor | None] = [None] * model.hidden_layers
    current = (
        model.output_weight[output_bit][None, :]
        / math.sqrt(model.width)
        * math.sqrt(2.0)
        * gates[-1]
    )
    sensitivities[-1] = current
    for layer_index in range(model.hidden_layers - 2, -1, -1):
        next_weight = model.hidden_weights[layer_index + 1]
        current = (
            current @ next_weight
            / math.sqrt(model.width)
            * math.sqrt(2.0)
            * gates[layer_index]
        )
        sensitivities[layer_index] = current
    return [value for value in sensitivities if value is not None]


@torch.no_grad()
def activation_parameter_grams(
    model: DeepReluMLP,
    activations1: list[torch.Tensor],
    activations2: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """预计算各隐藏权重层与输出权重层公用的 activation Gram。"""
    hidden_weight_grams = []
    for layer_index in range(model.hidden_layers):
        fan_in = model.input_dim if layer_index == 0 else model.width
        hidden_weight_grams.append(
            activations1[layer_index] @ activations2[layer_index].t() / fan_in
        )
    output_weight_gram = (
        activations1[-1] @ activations2[-1].t() / model.width
    )
    return hidden_weight_grams, output_weight_gram


@torch.no_grad()
def empirical_ntk_from_grams(
    hidden_weight_grams: list[torch.Tensor],
    output_weight_gram: torch.Tensor,
    sensitivities1: list[torch.Tensor],
    sensitivities2: list[torch.Tensor],
) -> torch.Tensor:
    kernel = output_weight_gram.clone()
    for activation_gram, sensitivity1, sensitivity2 in zip(
        hidden_weight_grams,
        sensitivities1,
        sensitivities2,
    ):
        kernel.add_(activation_gram * (sensitivity1 @ sensitivity2.t()))
    return kernel


@torch.no_grad()
def empirical_ntk_diagonal(
    model: DeepReluMLP,
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_bit: int,
    components1: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    components2: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
) -> torch.Tensor:
    """精确计算单个输出坐标的经验 NTK，不做宽度极限近似。"""
    activations1, gates1 = components1 or network_components(model, x1)
    activations2, gates2 = components2 or network_components(model, x2)
    sensitivities1 = output_sensitivities(model, gates1, output_bit)
    sensitivities2 = output_sensitivities(model, gates2, output_bit)
    hidden_weight_grams, output_weight_gram = activation_parameter_grams(
        model,
        activations1,
        activations2,
    )
    return empirical_ntk_from_grams(
        hidden_weight_grams,
        output_weight_gram,
        sensitivities1,
        sensitivities2,
    )


@torch.no_grad()
def empirical_ntk_block(
    model: DeepReluMLP,
    x: torch.Tensor,
    output_bits: tuple[int, ...],
) -> dict[str, torch.Tensor | float]:
    """计算选定输出位之间包含交叉块的完整经验 NTK。"""
    activations, gates = network_components(model, x)
    ck = activations[-1] @ activations[-1].t() / model.width
    activation_grams = []
    for layer_index in range(model.hidden_layers):
        fan_in = model.input_dim if layer_index == 0 else model.width
        activation_grams.append(
            activations[layer_index] @ activations[layer_index].t() / fan_in
        )
    sensitivities = {
        output_bit: output_sensitivities(model, gates, output_bit)
        for output_bit in output_bits
    }
    rows = []
    diagonal_rows = []
    for row_index, output_row in enumerate(output_bits):
        row_blocks = []
        diagonal_blocks = []
        for col_index, output_col in enumerate(output_bits):
            block = torch.zeros_like(ck)
            for layer_index in range(model.hidden_layers):
                sensitivity_gram = (
                    sensitivities[output_row][layer_index]
                    @ sensitivities[output_col][layer_index].t()
                )
                block = block + activation_grams[layer_index] * sensitivity_gram
            if row_index == col_index:
                block = block + ck
                diagonal_blocks.append(block)
            else:
                diagonal_blocks.append(torch.zeros_like(block))
            row_blocks.append(block)
        rows.append(torch.cat(row_blocks, dim=1))
        diagonal_rows.append(torch.cat(diagonal_blocks, dim=1))
    block_ntk = torch.cat(rows, dim=0)
    diagonal_ntk = torch.cat(diagonal_rows, dim=0)
    off_diagonal = block_ntk - diagonal_ntk
    off_diagonal_ratio = float(
        torch.linalg.vector_norm(off_diagonal)
        / torch.linalg.vector_norm(block_ntk).clamp_min(1e-20)
    )
    return {
        "block_ntk": block_ntk,
        "diagonal_ntk": diagonal_ntk,
        "ck": ck,
        "gate": torch.cat(gates, dim=1),
        "off_diagonal_ratio": off_diagonal_ratio,
    }


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    row_mean = kernel.mean(dim=1, keepdim=True)
    col_mean = kernel.mean(dim=0, keepdim=True)
    return kernel - row_mean - col_mean + kernel.mean()


def centered_kernel_alignment(
    kernel1: torch.Tensor,
    kernel2: torch.Tensor,
) -> float:
    centered1 = center_kernel(kernel1)
    centered2 = center_kernel(kernel2)
    denominator = (
        torch.linalg.vector_norm(centered1)
        * torch.linalg.vector_norm(centered2)
    ).clamp_min(1e-20)
    return float((centered1 * centered2).sum() / denominator)


def relative_frobenius_change(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    return float(
        torch.linalg.vector_norm(current - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-20)
    )


def effective_rank(kernel: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(center_kernel(kernel).double())
    eigenvalues = eigenvalues.clamp_min(0)
    total = eigenvalues.sum()
    if float(total) <= 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return float(torch.exp(entropy))


def empirical_formula_self_check(device: torch.device) -> dict[str, float]:
    """用逐参数 autograd Jacobian 审计闭式经验 NTK。"""
    set_seed(77)
    model = DeepReluMLP(5, 11, 3, 3).to(device=device, dtype=torch.float64)
    x = torch.randn(4, 5, device=device, dtype=torch.float64)
    selected = (0, 2)
    formula = empirical_ntk_block(model, x, selected)["block_ntk"]

    jacobian_rows = []
    for output_bit in selected:
        for sample_index in range(len(x)):
            model.zero_grad(set_to_none=True)
            value = model(x[sample_index : sample_index + 1])[0, output_bit]
            gradients = torch.autograd.grad(value, tuple(model.parameters()))
            jacobian_rows.append(torch.cat([grad.reshape(-1) for grad in gradients]))
    jacobian = torch.stack(jacobian_rows)
    reference = jacobian @ jacobian.t()
    max_abs_diff = float((formula - reference).abs().max())
    max_rel_diff = max_abs_diff / max(float(reference.abs().max()), 1e-20)
    if max_rel_diff > 1e-10:
        raise RuntimeError(
            "经验 NTK 闭式公式未通过 Jacobian 审计："
            f"max_abs={max_abs_diff:.3e}, max_rel={max_rel_diff:.3e}"
        )
    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }


def solve_kernel_system(
    kernel: torch.Tensor,
    targets: torch.Tensor,
    ridge_factor: float,
) -> tuple[torch.Tensor, float]:
    n = kernel.shape[0]
    ridge = float(ridge_factor * torch.trace(kernel) / n)
    identity = torch.eye(n, device=kernel.device, dtype=kernel.dtype)
    # 小 ridge 在 float32 上偶尔会遇到病态矩阵；逐级增加并明确记录。
    last_error: RuntimeError | None = None
    for multiplier in (1.0, 10.0, 100.0, 1_000.0):
        used_ridge = ridge * multiplier
        try:
            solution = torch.linalg.solve(
                kernel + used_ridge * identity,
                targets,
            )
            return solution, used_ridge
        except RuntimeError as error:
            last_error = error
    raise RuntimeError("kernel ridge 线性系统求解失败。") from last_error


@torch.no_grad()
def score_predictions(
    prediction: torch.Tensor,
    target01: torch.Tensor,
    threshold: float = 0.0,
) -> dict[str, Any]:
    hard = prediction > threshold
    truth = target01.bool()
    wrong = hard != truth
    return {
        "bit_accuracy": float((~wrong).float().mean()),
        "exact_accuracy": float((~wrong).all(dim=1).float().mean()),
        "bit_errors": int(wrong.sum()),
        "exact_errors": int(wrong.any(dim=1).sum()),
    }


@torch.no_grad()
def run_analytic_ntk_baseline(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    cfg: ResolvedConfig,
) -> dict[str, Any]:
    print("\n[解析无限宽 NTK] 计算训练核矩阵并求解完整输出 KRR……")
    started = time.perf_counter()
    kernel_train = analytic_relu_ntk(train_x, train_x, cfg.hidden_layers)
    target_pm = train_y * 2.0 - 1.0
    alpha, ridge = solve_kernel_system(
        kernel_train,
        target_pm,
        cfg.kernel_ridge_factor,
    )
    del kernel_train

    bit_errors = 0
    exact_errors = 0
    total_bits = 0
    total_samples = 0
    for start in range(0, len(test_x), cfg.kernel_test_chunk):
        end = min(start + cfg.kernel_test_chunk, len(test_x))
        prediction = (
            analytic_relu_ntk(
                test_x[start:end], train_x, cfg.hidden_layers
            )
            @ alpha
        )
        hard = prediction > 0.0
        truth = test_y[start:end].bool()
        wrong = hard != truth
        bit_errors += int(wrong.sum())
        exact_errors += int(wrong.any(dim=1).sum())
        total_bits += wrong.numel()
        total_samples += len(wrong)
    result = {
        "kind": "infinite_width_deep_relu_ntk",
        "hidden_layers": cfg.hidden_layers,
        "train_count": len(train_x),
        "test_count": len(test_x),
        "ridge": ridge,
        "bit_errors": bit_errors,
        "exact_errors": exact_errors,
        "bit_accuracy": 1.0 - bit_errors / total_bits,
        "exact_accuracy": 1.0 - exact_errors / total_samples,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        "[解析无限宽 NTK] "
        f"bit={result['bit_accuracy']:.9%} | "
        f"exact={result['exact_accuracy']:.9%} | "
        f"bit_errors={bit_errors} | exact_errors={exact_errors} | "
        f"{result['elapsed_seconds']:.1f}s"
    )
    return result


@torch.no_grad()
def run_empirical_ntk_baseline(
    model: DeepReluMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    cfg: ResolvedConfig,
) -> dict[str, Any]:
    print("\n[初始化经验 NTK] 对选定输出位做真实 finite-width KRR……")
    started = time.perf_counter()
    train_components = network_components(model, train_x)
    train_activation_grams, train_output_gram = activation_parameter_grams(
        model,
        train_components[0],
        train_components[0],
    )
    selected_predictions = torch.empty(
        len(test_x),
        len(cfg.selected_output_bits),
        dtype=torch.bool,
        device=test_x.device,
    )
    per_bit = {}
    train_sensitivities = {}
    alphas = {}
    test_bit_errors = {bit: 0 for bit in cfg.selected_output_bits}

    for selected_index, output_bit in enumerate(cfg.selected_output_bits):
        bit_started = time.perf_counter()
        sensitivities = output_sensitivities(
            model,
            train_components[1],
            output_bit,
        )
        train_sensitivities[output_bit] = sensitivities
        kernel_train = empirical_ntk_from_grams(
            train_activation_grams,
            train_output_gram,
            sensitivities,
            sensitivities,
        )
        target_pm = train_y[:, output_bit] * 2.0 - 1.0
        alpha, ridge = solve_kernel_system(
            kernel_train,
            target_pm,
            cfg.kernel_ridge_factor,
        )
        train_prediction = kernel_train @ alpha
        train_metrics = score_predictions(
            train_prediction[:, None],
            train_y[:, output_bit : output_bit + 1],
        )
        alphas[output_bit] = alpha
        del kernel_train
        per_bit[str(output_bit)] = {
            "ridge": ridge,
            "train_bit_accuracy": train_metrics["bit_accuracy"],
            "train_bit_errors": train_metrics["bit_errors"],
            "fit_seconds": time.perf_counter() - bit_started,
        }
        print(
            f"  output[{output_bit:2d}] | "
            f"train_errors={train_metrics['bit_errors']} | "
            f"KRR fit={per_bit[str(output_bit)]['fit_seconds']:.1f}s"
        )

    for start in range(0, len(test_x), cfg.kernel_test_chunk):
        end = min(start + cfg.kernel_test_chunk, len(test_x))
        test_components = network_components(model, test_x[start:end])
        cross_activation_grams, cross_output_gram = activation_parameter_grams(
            model,
            test_components[0],
            train_components[0],
        )
        for selected_index, output_bit in enumerate(cfg.selected_output_bits):
            test_sensitivities = output_sensitivities(
                model,
                test_components[1],
                output_bit,
            )
            kernel_test = empirical_ntk_from_grams(
                cross_activation_grams,
                cross_output_gram,
                test_sensitivities,
                train_sensitivities[output_bit],
            )
            prediction = kernel_test @ alphas[output_bit]
            hard = prediction > 0.0
            selected_predictions[start:end, selected_index] = hard
            test_bit_errors[output_bit] += int(
                (hard != test_y[start:end, output_bit].bool()).sum()
            )

    for output_bit in cfg.selected_output_bits:
        bit_errors = test_bit_errors[output_bit]
        per_bit[str(output_bit)].update(
            {
                "test_bit_accuracy": 1.0 - bit_errors / len(test_x),
                "test_bit_errors": bit_errors,
            }
        )
        print(
            f"  output[{output_bit:2d}] | test_errors={bit_errors}"
        )

    selected_truth = test_y[:, cfg.selected_output_bits].bool()
    selected_wrong = selected_predictions != selected_truth
    result = {
        "kind": "finite_width_empirical_ntk_diagonal_blocks",
        "width": model.width,
        "hidden_layers": model.hidden_layers,
        "selected_output_bits": cfg.selected_output_bits,
        "per_bit": per_bit,
        "selected_bit_errors": int(selected_wrong.sum()),
        "selected_exact_errors": int(selected_wrong.any(dim=1).sum()),
        "selected_bit_accuracy": float((~selected_wrong).float().mean()),
        "selected_exact_accuracy": float(
            (~selected_wrong).all(dim=1).float().mean()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        "[初始化经验 NTK] "
        f"selected_bit={result['selected_bit_accuracy']:.9%} | "
        f"selected_exact={result['selected_exact_accuracy']:.9%} | "
        f"bit_errors={result['selected_bit_errors']} | "
        f"exact_errors={result['selected_exact_errors']}"
    )
    return result


@torch.no_grad()
def evaluate_model(
    model: DeepReluMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mode: str,
    batch_size: int,
) -> dict[str, Any]:
    total_loss = 0.0
    total_elements = 0
    bit_errors = 0
    exact_errors = 0
    minimum_margin = float("inf")
    margin_sum = 0.0
    logit_abs_sum = 0.0
    model.eval()
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        logits = model(x[start:end])
        target = y[start:end]
        target_pm = target * 2.0 - 1.0
        if loss_mode == "bce":
            batch_loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="sum",
            )
        else:
            batch_loss = F.mse_loss(logits, target_pm, reduction="sum")
        hard = logits > 0.0
        wrong = hard != target.bool()
        margins = target_pm * logits
        total_loss += float(batch_loss)
        total_elements += target.numel()
        bit_errors += int(wrong.sum())
        exact_errors += int(wrong.any(dim=1).sum())
        minimum_margin = min(minimum_margin, float(margins.min()))
        margin_sum += float(margins.sum())
        logit_abs_sum += float(logits.abs().sum())
    return {
        "loss": total_loss / total_elements,
        "bit_accuracy": 1.0 - bit_errors / total_elements,
        "exact_accuracy": 1.0 - exact_errors / len(x),
        "bit_errors": bit_errors,
        "exact_errors": exact_errors,
        "minimum_margin": minimum_margin,
        "mean_margin": margin_sum / total_elements,
        "mean_abs_logit": logit_abs_sum / total_elements,
    }


def parameter_snapshot(model: DeepReluMLP) -> dict[str, Any]:
    return {
        "hidden_weights": [
            weight.detach().clone() for weight in model.hidden_weights
        ],
        "output_weight": model.output_weight.detach().clone(),
    }


@torch.no_grad()
def parameter_movement(
    model: DeepReluMLP,
    initial: dict[str, Any],
) -> dict[str, float]:
    hidden_delta_sq = torch.zeros((), device=model.output_weight.device)
    hidden_initial_sq = torch.zeros((), device=model.output_weight.device)
    for weight, initial_weight in zip(
        model.hidden_weights, initial["hidden_weights"]
    ):
        hidden_delta_sq += (weight - initial_weight).square().sum()
        hidden_initial_sq += initial_weight.square().sum()
    hidden_relative = float(
        torch.sqrt(hidden_delta_sq)
        / torch.sqrt(hidden_initial_sq).clamp_min(1e-20)
    )
    output_relative = float(
        torch.linalg.vector_norm(model.output_weight - initial["output_weight"])
        / torch.linalg.vector_norm(initial["output_weight"]).clamp_min(1e-20)
    )
    return {
        "hidden_weight_relative_movement": hidden_relative,
        "output_weight_relative_movement": output_relative,
    }


@torch.no_grad()
def kernel_snapshot(
    model: DeepReluMLP,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    output_bits: tuple[int, ...],
) -> dict[str, Any]:
    bundle = empirical_ntk_block(model, probe_x, output_bits)
    selected_target = probe_y[:, output_bits] * 2.0 - 1.0
    sample_target_kernel = (
        selected_target @ selected_target.t() / len(output_bits)
    )
    vector_target = selected_target.t().reshape(-1, 1)
    block_target_kernel = vector_target @ vector_target.t()
    bundle["ck_target_alignment"] = centered_kernel_alignment(
        bundle["ck"],
        sample_target_kernel,
    )
    bundle["block_ntk_target_alignment"] = centered_kernel_alignment(
        bundle["block_ntk"],
        block_target_kernel,
    )
    bundle["ck_effective_rank"] = effective_rank(bundle["ck"])
    bundle["block_ntk_effective_rank"] = effective_rank(bundle["block_ntk"])
    return bundle


@torch.no_grad()
def compare_kernel_snapshot(
    current: dict[str, Any],
    initial: dict[str, Any],
    solved: dict[str, Any] | None,
) -> dict[str, float | None]:
    gate_flip_init = float((current["gate"] != initial["gate"]).float().mean())
    result: dict[str, float | None] = {
        "block_ntk_cka_to_init": centered_kernel_alignment(
            current["block_ntk"], initial["block_ntk"]
        ),
        "diagonal_ntk_cka_to_init": centered_kernel_alignment(
            current["diagonal_ntk"], initial["diagonal_ntk"]
        ),
        "ck_cka_to_init": centered_kernel_alignment(current["ck"], initial["ck"]),
        "block_ntk_relative_change": relative_frobenius_change(
            current["block_ntk"], initial["block_ntk"]
        ),
        "ck_relative_change": relative_frobenius_change(
            current["ck"], initial["ck"]
        ),
        "block_ntk_norm_ratio": float(
            torch.linalg.vector_norm(current["block_ntk"])
            / torch.linalg.vector_norm(initial["block_ntk"]).clamp_min(1e-20)
        ),
        "ck_norm_ratio": float(
            torch.linalg.vector_norm(current["ck"])
            / torch.linalg.vector_norm(initial["ck"]).clamp_min(1e-20)
        ),
        "gate_flip_fraction_from_init": gate_flip_init,
        "block_ntk_target_alignment": float(
            current["block_ntk_target_alignment"]
        ),
        "ck_target_alignment": float(current["ck_target_alignment"]),
        "block_ntk_effective_rank": float(current["block_ntk_effective_rank"]),
        "ck_effective_rank": float(current["ck_effective_rank"]),
        "off_diagonal_ratio": float(current["off_diagonal_ratio"]),
        "block_ntk_cka_to_first_perfect": None,
        "ck_cka_to_first_perfect": None,
        "gate_flip_fraction_from_first_perfect": None,
    }
    if solved is not None:
        result.update(
            {
                "block_ntk_cka_to_first_perfect": centered_kernel_alignment(
                    current["block_ntk"], solved["block_ntk"]
                ),
                "ck_cka_to_first_perfect": centered_kernel_alignment(
                    current["ck"], solved["ck"]
                ),
                "gate_flip_fraction_from_first_perfect": float(
                    (current["gate"] != solved["gate"]).float().mean()
                ),
            }
        )
    return result


def make_optimizer(
    model: DeepReluMLP,
    cfg: ResolvedConfig,
) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=0.0,
        weight_decay=cfg.weight_decay,
    )


def train_one_run(
    initial_state: dict[str, torch.Tensor],
    seed: int,
    loss_mode: str,
    data: dict[str, torch.Tensor],
    cfg: ResolvedConfig,
) -> dict[str, Any]:
    run_name = f"{loss_mode}_seed{seed}"
    run_dir = cfg.result_dir / "runs" / run_name
    summary_path = run_dir / "summary.json"
    if cfg.skip_completed_runs and summary_path.exists():
        print(f"\n[{run_name}] 已完成，跳过。")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model = DeepReluMLP(
        cfg.length,
        cfg.width,
        cfg.hidden_layers,
        cfg.length,
    ).to(device)
    model.load_state_dict(initial_state)
    optimizer = make_optimizer(model, cfg)
    initial_parameters = parameter_snapshot(model)

    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]
    probe_x = test_x[: cfg.probe_count]
    probe_y = test_y[: cfg.probe_count]

    initial_kernel = kernel_snapshot(
        model,
        probe_x,
        probe_y,
        cfg.selected_output_bits,
    )
    first_perfect_kernel = None
    first_perfect_step = None
    history: list[dict[str, Any]] = []
    snapshot_steps = set(cfg.snapshot_steps)
    generator = torch.Generator(device=device)
    generator.manual_seed(50_000 + seed)
    permutation = torch.randperm(len(train_x), generator=generator, device=device)
    cursor = 0
    started = time.perf_counter()

    def record(step: int) -> None:
        nonlocal first_perfect_kernel, first_perfect_step
        train_metrics = evaluate_model(
            model, train_x, train_y, loss_mode, cfg.eval_batch_size
        )
        test_metrics = evaluate_model(
            model, test_x, test_y, loss_mode, cfg.eval_batch_size
        )
        current_kernel = kernel_snapshot(
            model,
            probe_x,
            probe_y,
            cfg.selected_output_bits,
        )
        just_became_perfect = (
            first_perfect_kernel is None
            and train_metrics["exact_errors"] == 0
            and test_metrics["exact_errors"] == 0
        )
        if just_became_perfect:
            first_perfect_kernel = {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in current_kernel.items()
            }
            first_perfect_step = step
        kernel_metrics = compare_kernel_snapshot(
            current_kernel,
            initial_kernel,
            first_perfect_kernel,
        )
        row = {
            "run_name": run_name,
            "seed": seed,
            "loss_mode": loss_mode,
            "step": step,
            "equivalent_epochs": step * cfg.batch_size / len(train_x),
            "elapsed_seconds": time.perf_counter() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **kernel_metrics,
            **parameter_movement(model, initial_parameters),
            "is_first_perfect_snapshot": bool(just_became_perfect),
        }
        history.append(row)
        post_text = ""
        if row["block_ntk_cka_to_first_perfect"] is not None:
            post_text = (
                " | CKA(perfect->now)="
                f"{row['block_ntk_cka_to_first_perfect']:.6f}"
            )
        print(
            f"[{run_name}] step={step:6d} | "
            f"test bit={row['test_bit_accuracy']:.7%} "
            f"exact={row['test_exact_accuracy']:.7%} | "
            f"NTK CKA(init)={row['block_ntk_cka_to_init']:.6f} | "
            f"CK CKA(init)={row['ck_cka_to_init']:.6f} | "
            f"gate_flip={row['gate_flip_fraction_from_init']:.4%}"
            f"{post_text}"
        )

    record(0)
    model.train()
    for step in range(1, cfg.train_steps + 1):
        if cursor + cfg.batch_size > len(train_x):
            permutation = torch.randperm(
                len(train_x), generator=generator, device=device
            )
            cursor = 0
        batch_index = permutation[cursor : cursor + cfg.batch_size]
        cursor += cfg.batch_size
        batch_x = train_x[batch_index]
        batch_y = train_y[batch_index]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        if loss_mode == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        else:
            loss = F.mse_loss(logits, batch_y * 2.0 - 1.0)
        loss.backward()
        optimizer.step()
        if step in snapshot_steps:
            record(step)
            model.train()

    final = history[-1]
    post_solution_detected = False
    if first_perfect_step is not None:
        cka_change = 1.0 - float(final["ck_cka_to_first_perfect"])
        gate_change = float(final["gate_flip_fraction_from_first_perfect"])
        post_solution_detected = (
            cka_change >= cfg.structural_cka_change_threshold
            and gate_change >= cfg.gate_flip_threshold
        )
    structural_feature_learning = (
        1.0 - float(final["ck_cka_to_init"])
        >= cfg.structural_cka_change_threshold
        and float(final["gate_flip_fraction_from_init"])
        >= cfg.gate_flip_threshold
    )
    summary = {
        "run_name": run_name,
        "seed": seed,
        "loss_mode": loss_mode,
        "first_perfect_snapshot_step": first_perfect_step,
        "final_step": cfg.train_steps,
        "final": final,
        "structural_feature_learning_detected": structural_feature_learning,
        "post_solution_feature_learning_detected": post_solution_detected,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_csv(run_dir / "trajectory.csv", history)
    write_json(summary_path, summary)
    plot_run(history, run_dir / "trajectory.png", run_name)
    if cfg.save_models:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "loss_mode": loss_mode,
                "config": json_ready(asdict(cfg)),
            },
            run_dir / "final_model.pt",
        )
    print(
        f"[{run_name}] 完成 | first_perfect={first_perfect_step} | "
        f"feature_learning={structural_feature_learning} | "
        f"post_solution={post_solution_detected} | "
        f"耗时={summary['elapsed_seconds']:.1f}s"
    )
    return summary


def plot_run(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    steps = np.array([row["step"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(steps, [row["train_bit_accuracy"] for row in rows], label="train bit")
    axes[0, 0].plot(steps, [row["test_bit_accuracy"] for row in rows], label="test bit")
    axes[0, 0].plot(steps, [row["test_exact_accuracy"] for row in rows], label="test exact")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, [row["train_loss"] for row in rows], label="train")
    axes[0, 1].plot(steps, [row["test_loss"] for row in rows], label="test")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [1.0 - row["block_ntk_cka_to_init"] for row in rows],
        label="1 - block NTK CKA",
    )
    axes[1, 0].plot(
        steps,
        [1.0 - row["ck_cka_to_init"] for row in rows],
        label="1 - hidden CK CKA",
    )
    axes[1, 0].plot(
        steps,
        [row["gate_flip_fraction_from_init"] for row in rows],
        label="gate flip",
    )
    axes[1, 0].set_ylabel("structural change")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["block_ntk_target_alignment"] for row in rows],
        label="block NTK-target",
    )
    axes[1, 1].plot(
        steps,
        [row["ck_target_alignment"] for row in rows],
        label="hidden CK-target",
    )
    axes[1, 1].plot(
        steps,
        [row["off_diagonal_ratio"] for row in rows],
        label="cross-output ratio",
    )
    axes[1, 1].set_ylabel("alignment / ratio")
    axes[1, 1].legend()

    for axis in axes.ravel():
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_aggregate(summaries: list[dict[str, Any]], result_dir: Path) -> None:
    rows = []
    for summary in summaries:
        trajectory_path = (
            result_dir / "runs" / summary["run_name"] / "trajectory.csv"
        )
        with trajectory_path.open("r", encoding="utf-8-sig") as handle:
            rows.extend(list(csv.DictReader(handle)))
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    loss_modes = sorted({row["run_name"].split("_seed")[0] for row in rows})
    for loss_mode in loss_modes:
        mode_rows = [row for row in rows if row["run_name"].startswith(loss_mode)]
        steps = sorted({int(row["step"]) for row in mode_rows})
        for axis, key, label in (
            (axes[0], "ck_cka_to_init", "hidden CK"),
            (axes[1], "block_ntk_cka_to_init", "block NTK"),
        ):
            means = []
            lows = []
            highs = []
            for step in steps:
                values = [
                    1.0 - float(row[key])
                    for row in mode_rows
                    if int(row["step"]) == step
                ]
                means.append(float(np.mean(values)))
                lows.append(float(np.min(values)))
                highs.append(float(np.max(values)))
            axis.plot(steps, means, marker="o", label=f"{loss_mode}: {label}")
            axis.fill_between(steps, lows, highs, alpha=0.15)
    for axis in axes:
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("1 - CKA to initialization")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(result_dir / "aggregate_feature_learning.png", dpi=180)
    plt.close(figure)


def main() -> None:
    cfg = resolve_config()
    validate_config(cfg)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        cfg.result_dir / "config.json",
        {**asdict(cfg), "fingerprint": config_fingerprint(cfg)},
    )

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "无"
    print("=== NTK 已足够时，Rule 110 是否仍发生特征学习 ===")
    print(f"设备：{device} | GPU：{gpu_name}")
    print(
        f"数据：Rule 110 一层 | {cfg.length} -> {cfg.length} bit | "
        f"train={cfg.train_count:,} | test={cfg.test_count:,}"
    )
    print(
        f"网络：{cfg.length} -> {cfg.width:,} x {cfg.hidden_layers} ReLU "
        f"-> {cfg.length} | "
        f"seeds={cfg.model_seeds} | losses={cfg.loss_modes}"
    )
    print(f"结果：{cfg.result_dir}")

    if cfg.run_empirical_formula_self_check:
        audit = empirical_formula_self_check(device)
        write_json(cfg.result_dir / "empirical_ntk_formula_audit.json", audit)
        print(
            "经验 NTK 公式审计通过："
            f"max_abs={audit['max_abs_diff']:.3e} | "
            f"max_rel={audit['max_rel_diff']:.3e}"
        )

    cpu_data = make_dataset(cfg)
    data = {key: value.to(device) for key, value in cpu_data.items()}

    analytic_summary = None
    analytic_path = cfg.result_dir / "analytic_ntk_baseline.json"
    if cfg.run_analytic_ntk:
        if cfg.skip_completed_runs and analytic_path.exists():
            analytic_summary = json.loads(analytic_path.read_text(encoding="utf-8"))
            print("\n[解析无限宽 NTK] 已有结果，跳过重新计算。")
        else:
            analytic_summary = run_analytic_ntk_baseline(
                data["train_x"],
                data["train_y"],
                data["test_x"],
                data["test_y"],
                cfg,
            )
            write_json(analytic_path, analytic_summary)

    empirical_summaries: dict[str, Any] = {}
    run_summaries = []
    for seed in cfg.model_seeds:
        set_seed(seed)
        initial_model = DeepReluMLP(
            cfg.length,
            cfg.width,
            cfg.hidden_layers,
            cfg.length,
        ).to(device)
        initial_state = {
            key: value.detach().clone()
            for key, value in initial_model.state_dict().items()
        }
        empirical_path = cfg.result_dir / f"empirical_ntk_seed{seed}.json"
        if cfg.run_empirical_ntk:
            if cfg.skip_completed_runs and empirical_path.exists():
                empirical_summary = json.loads(
                    empirical_path.read_text(encoding="utf-8")
                )
                print(f"\n[初始化经验 NTK seed={seed}] 已有结果，跳过。")
            else:
                empirical_summary = run_empirical_ntk_baseline(
                    initial_model,
                    data["train_x"],
                    data["train_y"],
                    data["test_x"],
                    data["test_y"],
                    cfg,
                )
                empirical_summary["seed"] = seed
                write_json(empirical_path, empirical_summary)
            empirical_summaries[str(seed)] = empirical_summary

        for loss_mode in cfg.loss_modes:
            run_summaries.append(
                train_one_run(
                    initial_state,
                    seed,
                    loss_mode,
                    data,
                    cfg,
                )
            )
        del initial_model, initial_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_aggregate(run_summaries, cfg.result_dir)
    solved_runs = [
        row for row in run_summaries
        if row["first_perfect_snapshot_step"] is not None
    ]
    final_summary = {
        "question": (
            "当初始化固定 NTK 已足以学习一层 Rule 110 时，"
            "端到端训练是否仍会改变特征和经验 NTK？"
        ),
        "config_fingerprint": config_fingerprint(cfg),
        "analytic_ntk": analytic_summary,
        "analytic_ntk_solved_rule": (
            analytic_summary is not None
            and analytic_summary["exact_errors"] == 0
        ),
        "empirical_ntk": empirical_summaries,
        "runs": run_summaries,
        "all_runs_structural_feature_learning": all(
            row["structural_feature_learning_detected"] for row in run_summaries
        ),
        "solved_run_count": len(solved_runs),
        "all_solved_runs_continue_after_solution": (
            all(row["post_solution_feature_learning_detected"] for row in solved_runs)
            if solved_runs
            else None
        ),
    }
    write_json(cfg.result_dir / "summary.json", final_summary)

    print("\n=== 实验完成 ===")
    if analytic_summary is not None:
        print(
            "解析 NTK："
            f"bit_errors={analytic_summary['bit_errors']} | "
            f"exact_errors={analytic_summary['exact_errors']}"
        )
    print(
        "端到端运行："
        f"{len(run_summaries)} | "
        "结构性特征学习="
        f"{sum(x['structural_feature_learning_detected'] for x in run_summaries)}"
        f"/{len(run_summaries)}"
    )
    print(f"汇总：{cfg.result_dir / 'summary.json'}")
    if cfg.create_zip:
        archive = shutil.make_archive(
            str(cfg.result_dir),
            "zip",
            root_dir=cfg.result_dir,
        )
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
