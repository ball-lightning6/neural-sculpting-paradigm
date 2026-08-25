"""
3-bit Boolean 静态自由能、逐样本惊讶度与路径不变量实验。

研究问题
========

对 3-bit -> 1-bit 的 256 条完整规则，把 8 个真值表样本逐个加入。每个样本
定义一个单点 loss：

    ell_i(theta) = BCEWithLogits(f_theta(x_i), y_i)

样本集合 S 的总能量（注意不是 mean loss）为：

    E_S(theta) = sum_{i in S} ell_i(theta)

固定初始化参数测度 mu 和逆温度 beta，定义配分函数与无量纲自由能：

    Z_beta(S)   = E_{theta~mu}[exp(-beta E_S(theta))]
    Phi_beta(S) = -log Z_beta(S)

加入样本 j 的信息增量：

    I_beta(j | S) = Phi_beta(S U {j}) - Phi_beta(S)

于是对任意样本排列 pi 都严格满足：

    sum_t I_beta(pi_t | S_{t-1})
      = Phi_beta(D_full) - Phi_beta(empty)

这就是此前“累计惊讶度路径守恒”真正对应的静态势。beta=1 时
exp(-BCE) 是 Bernoulli likelihood，因此增量就是 Bayesian predictive
surprisal；其他 beta 是 tempered/generalized surprise。

脚本同时计算 hard-function 条件化极限：

    Z_hard(S) = P_mu[f_theta 与 S 的全部 hard labels 相容]

此时增量精确等于：

    -log P_mu[f(x_j)=y_j | f 与 S 相容]

脚本一次 prior 采样后离线重建全部 3^8=6,561 个部分标注数据集、全部 256 条
完整规则和全部 8!=40,320 条样本顺序，并输出：

1. hard 与 Gibbs 自由能的路径守恒误差；
2. 256 条规则在不同 beta 下的静态难度排名；
3. 每个样本数量阶段的平均信息增量；
4. 每个输入样本的 Shapley 信息贡献；
5. 完整规则 direct low-loss volume 曲线；
6. prior function mass、有效样本量 ESS 和深尾可靠性；
7. Rule 150/parity 与常量、copy、majority、Rule30/110 的对照。

重要边界
========

- 这是静态初始化测度上的精确恒等式，不是假设 SGD 是无偏采样器；
- 旧 optimizer-ensemble surprise 不守恒，说明它含有非保守的路径重加权；
- 当前脚本先测静态势。真实 SGD 的“环流/残差场”应在同一参考机上作为下一
  个独立实验测量，不能混入静态守恒判决；
- direct prior 在深 beta/低 loss 处可能 ESS 退化，脚本会明确标为 unsupported。

运行方式
========

    python experiment_static_free_energy_information_invariant.py
    %run experiment_static_free_energy_information_invariant.py
    或将整个文件粘贴到 AutoDL Jupyter cell。

默认针对 RTX 5090；支持 Ctrl+C 保存 checkpoint 和部分分析结果。
本地 smoke：设置环境变量 NSP_SMOKE_TEST=1。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# 配置
# =============================================================================


def script_directory() -> Path:
    source = globals().get("__file__")
    if source and not str(source).startswith("<"):
        return Path(source).resolve().parent
    return Path.cwd()


class Config:
    INPUT_BITS = 3
    INPUT_COUNT = 2**INPUT_BITS
    FUNCTION_COUNT = 2**INPUT_COUNT

    # 与 full-truth volume SMC 同族：小 tanh MLP，参数立方体均匀测度。
    WIDTH = 16
    HIDDEN_LAYERS = 2

    PRIOR_SAMPLES = 2_097_152
    PRIOR_BATCH = 4_096
    PRIOR_SEED = 2026082501

    # beta=1 具有标准 predictive-evidence 含义；更高 beta 强调 low-loss 深尾。
    BETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    MICROCANONICAL_MEAN_BCE_THRESHOLDS = (
        0.72,
        0.70,
        0.68,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
    )

    # 全函数 Dirichlet 总伪计数为 1；避免有限 prior 样本给尾部函数无穷成本。
    HARD_FUNCTION_ALPHA = 1.0 / FUNCTION_COUNT
    MIN_SUPPORTED_ESS = 64.0
    MIN_SUPPORTED_MICRO_COUNT = 32

    SELECTED_RULES = (0, 255, 204, 170, 240, 232, 90, 150, 30, 110)

    CHECKPOINT_EVERY_BATCHES = 16
    LOG_EVERY_BATCHES = 8
    SAVE_LOGIT_SAMPLE = 262_144
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = (
        Path("/root/results_static_free_energy_information_invariant")
        if Path("/root").exists()
        else script_directory() / "results_static_free_energy_information_invariant"
    )
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


PROTOCOL_VERSION = "static_free_energy_information_invariant_v1"

RULE_NAMES = {
    0: "constant_0",
    255: "constant_1",
    204: "copy_center",
    170: "copy_right",
    240: "copy_left",
    232: "majority3",
    90: "xor_left_right",
    150: "parity3_rule150",
    30: "rule30",
    110: "rule110",
}


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.WIDTH = 8
    Config.PRIOR_SAMPLES = 8_192
    Config.PRIOR_BATCH = 512
    Config.BETAS = (0.5, 1.0, 2.0)
    Config.MICROCANONICAL_MEAN_BCE_THRESHOLDS = (0.72, 0.70, 0.68)
    Config.CHECKPOINT_EVERY_BATCHES = 2
    Config.LOG_EVERY_BATCHES = 1
    Config.SAVE_LOGIT_SAMPLE = 2_048
    Config.MIN_SUPPORTED_ESS = 8.0
    Config.MIN_SUPPORTED_MICRO_COUNT = 4
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = (
        script_directory() / "_smoke_static_free_energy_information_invariant"
    )
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True
    Config.PACKAGE_RESULTS = False


@dataclass(frozen=True)
class ParameterLayout:
    parameter_count: int
    first_weight_stop: int
    first_bias_stop: int
    middle_weight_stop: int
    middle_bias_stop: int
    output_weight_stop: int
    output_bias_stop: int


def config_payload() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_bits": Config.INPUT_BITS,
        "width": Config.WIDTH,
        "hidden_layers": Config.HIDDEN_LAYERS,
        "prior_samples": Config.PRIOR_SAMPLES,
        "prior_batch": Config.PRIOR_BATCH,
        "prior_seed": Config.PRIOR_SEED,
        "betas": list(Config.BETAS),
        "microcanonical_mean_bce_thresholds": list(
            Config.MICROCANONICAL_MEAN_BCE_THRESHOLDS
        ),
        "hard_function_alpha": Config.HARD_FUNCTION_ALPHA,
        "min_supported_ess": Config.MIN_SUPPORTED_ESS,
        "min_supported_micro_count": Config.MIN_SUPPORTED_MICRO_COUNT,
        "selected_rules": list(Config.SELECTED_RULES),
        "device": Config.DEVICE,
        "allow_tf32": Config.ALLOW_TF32,
        "smoke_test": Config.SMOKE_TEST,
    }


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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


def prepare_result_dir() -> Path:
    root = Path(Config.RESULT_DIR)
    if root.exists() and Config.OVERWRITE_RESULT_DIR and not Config.RESUME:
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


# =============================================================================
# 3-bit 输入、256 函数与复杂度代理
# =============================================================================


def all_inputs(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [(index >> 2) & 1, (index >> 1) & 1, index & 1]
            for index in range(Config.INPUT_COUNT)
        ],
        dtype=torch.float32,
        device=device,
    )


def all_function_outputs() -> np.ndarray:
    rules = np.arange(Config.FUNCTION_COUNT, dtype=np.uint16)
    shifts = np.arange(Config.INPUT_COUNT, dtype=np.uint16)
    return ((rules[:, None] >> shifts[None]) & 1).astype(np.uint8)


def all_state_digits() -> np.ndarray:
    keys = np.arange(3**Config.INPUT_COUNT, dtype=np.int32)
    powers = (3 ** np.arange(Config.INPUT_COUNT, dtype=np.int64))[None]
    return ((keys[:, None] // powers) % 3).astype(np.int8)


def rule_state_keys(outputs: np.ndarray) -> np.ndarray:
    """返回某条完整规则的 2^8 个子集对应的 ternary state key。"""
    keys = np.zeros(2**Config.INPUT_COUNT, dtype=np.int32)
    ternary_powers = 3 ** np.arange(Config.INPUT_COUNT, dtype=np.int64)
    for subset in range(2**Config.INPUT_COUNT):
        value = 0
        for index in range(Config.INPUT_COUNT):
            if (subset >> index) & 1:
                value += int(1 + outputs[index]) * int(ternary_powers[index])
        keys[subset] = value
    return keys


def anf_features(outputs: np.ndarray) -> tuple[int, int, int]:
    coefficients = outputs.astype(np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        for mask in range(Config.INPUT_COUNT):
            if (mask >> bit) & 1:
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    terms = np.flatnonzero(coefficients)
    if len(terms) == 0:
        return 0, 0, 0
    degrees = [int(int(mask).bit_count()) for mask in terms]
    return max(degrees), len(terms), int(sum(degrees))


def essential_variable_count(outputs: np.ndarray) -> int:
    count = 0
    for bit in range(Config.INPUT_BITS):
        if any(
            outputs[index] != outputs[index ^ (1 << bit)]
            for index in range(Config.INPUT_COUNT)
        ):
            count += 1
    return count


def total_influence(outputs: np.ndarray) -> float:
    changes = 0
    for bit in range(Config.INPUT_BITS):
        changes += sum(
            int(outputs[index] != outputs[index ^ (1 << bit)])
            for index in range(Config.INPUT_COUNT)
        )
    return float(changes / (2 * Config.INPUT_COUNT))


def walsh_features(outputs: np.ndarray) -> tuple[int, int, float]:
    signs = 1.0 - 2.0 * outputs.astype(np.float64)
    coefficients = np.zeros(Config.INPUT_COUNT, dtype=np.float64)
    for subset in range(Config.INPUT_COUNT):
        total = 0.0
        for x in range(Config.INPUT_COUNT):
            parity = (int(subset) & int(x)).bit_count() & 1
            total += signs[x] * (-1.0 if parity else 1.0)
        coefficients[subset] = total / Config.INPUT_COUNT
    energy = coefficients**2
    nonzero = np.flatnonzero(energy > 1e-12)
    degrees = [int(int(mask).bit_count()) for mask in nonzero]
    probabilities = energy[energy > 1e-15]
    probabilities = probabilities / probabilities.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return min(degrees), max(degrees), entropy


def function_feature_rows(outputs: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in range(Config.FUNCTION_COUNT):
        values = outputs[rule]
        degree, terms, literals = anf_features(values)
        walsh_min, walsh_max, walsh_entropy = walsh_features(values)
        rows.append({
            "rule": rule,
            "rule_hex": f"0x{rule:02X}",
            "rule_name": RULE_NAMES.get(rule, ""),
            "positive_count": int(values.sum()),
            "essential_variables": essential_variable_count(values),
            "anf_degree": degree,
            "anf_terms": terms,
            "anf_literals": literals,
            "total_influence": total_influence(values),
            "walsh_min_order": walsh_min,
            "walsh_max_order": walsh_max,
            "walsh_entropy_bits": walsh_entropy,
        })
    return rows


# =============================================================================
# 小 tanh MLP 的参数立方体测度
# =============================================================================


def parameter_layout(width: int) -> ParameterLayout:
    cursor = 0
    cursor += width * Config.INPUT_BITS
    first_weight_stop = cursor
    cursor += width
    first_bias_stop = cursor
    cursor += width * width
    middle_weight_stop = cursor
    cursor += width
    middle_bias_stop = cursor
    cursor += width
    output_weight_stop = cursor
    cursor += 1
    return ParameterLayout(
        parameter_count=cursor,
        first_weight_stop=first_weight_stop,
        first_bias_stop=first_bias_stop,
        middle_weight_stop=middle_weight_stop,
        middle_bias_stop=middle_bias_stop,
        output_weight_stop=output_weight_stop,
        output_bias_stop=cursor,
    )


def forward_logits(
    normalized: torch.Tensor,
    inputs: torch.Tensor,
    layout: ParameterLayout,
) -> torch.Tensor:
    count = normalized.shape[0]
    width = Config.WIDTH
    cursor = 0

    first_weight = normalized[:, cursor:layout.first_weight_stop].reshape(
        count, width, Config.INPUT_BITS
    ) * (1.0 / math.sqrt(Config.INPUT_BITS))
    cursor = layout.first_weight_stop
    first_bias = normalized[:, cursor:layout.first_bias_stop] * (
        1.0 / math.sqrt(Config.INPUT_BITS)
    )
    cursor = layout.first_bias_stop

    middle_weight = normalized[:, cursor:layout.middle_weight_stop].reshape(
        count, width, width
    ) * (1.0 / math.sqrt(width))
    cursor = layout.middle_weight_stop
    middle_bias = normalized[:, cursor:layout.middle_bias_stop] * (
        1.0 / math.sqrt(width)
    )
    cursor = layout.middle_bias_stop

    output_weight = normalized[:, cursor:layout.output_weight_stop].reshape(
        count, 1, width
    ) * (1.0 / math.sqrt(width))
    cursor = layout.output_weight_stop
    output_bias = normalized[:, cursor:layout.output_bias_stop] * (
        1.0 / math.sqrt(width)
    )

    hidden = inputs[None].expand(count, -1, -1)
    hidden = torch.tanh(
        torch.bmm(hidden, first_weight.transpose(1, 2))
        + first_bias[:, None]
    )
    hidden = torch.tanh(
        torch.bmm(hidden, middle_weight.transpose(1, 2))
        + middle_bias[:, None]
    )
    return (
        torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
        + output_bias
    )


def function_ids_from_logits(logits: torch.Tensor) -> torch.Tensor:
    powers = 2 ** torch.arange(
        Config.INPUT_COUNT, dtype=torch.int64, device=logits.device
    )
    return ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)


def all_partial_energies(loss_zero: torch.Tensor, loss_one: torch.Tensor) -> torch.Tensor:
    """按 base-3 state key 顺序构造每个模型的全部 6,561 个总损失。"""
    energies = torch.zeros(
        (loss_zero.shape[0], 1), dtype=loss_zero.dtype, device=loss_zero.device
    )
    for index in range(Config.INPUT_COUNT):
        energies = torch.cat(
            (
                energies,
                energies + loss_zero[:, index : index + 1],
                energies + loss_one[:, index : index + 1],
            ),
            dim=1,
        )
    return energies


def full_rule_energies(
    loss_zero: torch.Tensor,
    loss_one: torch.Tensor,
    outputs_device: torch.Tensor,
) -> torch.Tensor:
    base = loss_zero.sum(dim=1, keepdim=True)
    delta = loss_one - loss_zero
    return base + delta @ outputs_device.transpose(0, 1)


# =============================================================================
# Prior 采样、checkpoint 与静态 partition sums
# =============================================================================


def initial_sampling_state(beta_count: int) -> dict[str, Any]:
    state_count = 3**Config.INPUT_COUNT
    return {
        "processed": 0,
        "function_counts": np.zeros(Config.FUNCTION_COUNT, dtype=np.int64),
        "micro_counts": np.zeros(
            (
                len(Config.MICROCANONICAL_MEAN_BCE_THRESHOLDS),
                Config.FUNCTION_COUNT,
            ),
            dtype=np.int64,
        ),
        "log_sum_w": np.full((beta_count, state_count), -np.inf, dtype=np.float64),
        "log_sum_w2": np.full((beta_count, state_count), -np.inf, dtype=np.float64),
        "saved_logits": [],
    }


def checkpoint_path(root: Path) -> Path:
    return root / "checkpoint.pt"


def save_checkpoint(
    root: Path,
    state: dict[str, Any],
    generator: torch.Generator,
    signature: str,
) -> None:
    payload = {
        "signature": signature,
        "processed": int(state["processed"]),
        "function_counts": state["function_counts"],
        "micro_counts": state["micro_counts"],
        "log_sum_w": state["log_sum_w"],
        "log_sum_w2": state["log_sum_w2"],
        "saved_logits": state["saved_logits"],
        "generator_state": generator.get_state().cpu(),
    }
    torch.save(payload, checkpoint_path(root))


def load_checkpoint(
    root: Path,
    generator: torch.Generator,
    signature: str,
) -> dict[str, Any] | None:
    path = checkpoint_path(root)
    if not Config.RESUME or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        raise RuntimeError(
            "checkpoint 配置签名不同；请更换 RESULT_DIR 或设置 "
            "OVERWRITE_RESULT_DIR=True、RESUME=False。"
        )
    generator.set_state(payload["generator_state"].to(torch.uint8))
    return {
        "processed": int(payload["processed"]),
        "function_counts": np.asarray(payload["function_counts"], dtype=np.int64),
        "micro_counts": np.asarray(payload["micro_counts"], dtype=np.int64),
        "log_sum_w": np.asarray(payload["log_sum_w"], dtype=np.float64),
        "log_sum_w2": np.asarray(payload["log_sum_w2"], dtype=np.float64),
        "saved_logits": list(payload.get("saved_logits", [])),
    }


@torch.no_grad()
def sample_prior(
    root: Path,
    inputs: torch.Tensor,
    outputs: np.ndarray,
    signature: str,
) -> tuple[dict[str, Any], bool, float]:
    device = inputs.device
    layout = parameter_layout(Config.WIDTH)
    generator = torch.Generator(device=device)
    generator.manual_seed(Config.PRIOR_SEED)
    state = load_checkpoint(root, generator, signature)
    if state is None:
        state = initial_sampling_state(len(Config.BETAS))

    outputs_device = torch.tensor(outputs, dtype=torch.float32, device=device)
    thresholds = torch.tensor(
        Config.MICROCANONICAL_MEAN_BCE_THRESHOLDS,
        dtype=torch.float32,
        device=device,
    )
    started = time.perf_counter()
    interrupted = False
    start_batch = int(state["processed"] // Config.PRIOR_BATCH)
    total_batches = math.ceil(Config.PRIOR_SAMPLES / Config.PRIOR_BATCH)

    try:
        while state["processed"] < Config.PRIOR_SAMPLES:
            batch_index = int(state["processed"] // Config.PRIOR_BATCH)
            count = min(
                Config.PRIOR_BATCH,
                Config.PRIOR_SAMPLES - int(state["processed"]),
            )
            parameters = torch.empty(
                (count, layout.parameter_count),
                dtype=torch.float32,
                device=device,
            ).uniform_(-1.0, 1.0, generator=generator)
            logits = forward_logits(parameters, inputs, layout)

            function_ids = function_ids_from_logits(logits)
            counts = torch.bincount(
                function_ids, minlength=Config.FUNCTION_COUNT
            ).cpu().numpy()
            state["function_counts"] += counts.astype(np.int64)

            remaining_logits = Config.SAVE_LOGIT_SAMPLE - sum(
                len(piece) for piece in state["saved_logits"]
            )
            if remaining_logits > 0:
                take = min(remaining_logits, count)
                state["saved_logits"].append(
                    logits[:take].cpu().to(torch.float16).numpy()
                )

            loss_zero = F.softplus(logits)
            loss_one = F.softplus(-logits)

            full_energies = full_rule_energies(
                loss_zero, loss_one, outputs_device
            )
            full_mean = full_energies / Config.INPUT_COUNT
            for threshold_index, threshold in enumerate(thresholds):
                state["micro_counts"][threshold_index] += (
                    full_mean <= threshold
                ).sum(dim=0).cpu().numpy().astype(np.int64)

            partial_energies = all_partial_energies(loss_zero, loss_one)
            for beta_index, beta in enumerate(Config.BETAS):
                log_weights = -float(beta) * partial_energies
                chunk_lse = torch.logsumexp(log_weights, dim=0).cpu().numpy()
                chunk_lse2 = torch.logsumexp(2.0 * log_weights, dim=0).cpu().numpy()
                state["log_sum_w"][beta_index] = np.logaddexp(
                    state["log_sum_w"][beta_index], chunk_lse
                )
                state["log_sum_w2"][beta_index] = np.logaddexp(
                    state["log_sum_w2"][beta_index], chunk_lse2
                )

            state["processed"] += count
            completed_batch = batch_index + 1
            if (
                completed_batch % Config.CHECKPOINT_EVERY_BATCHES == 0
                or state["processed"] == Config.PRIOR_SAMPLES
            ):
                save_checkpoint(root, state, generator, signature)

            if (
                completed_batch % Config.LOG_EVERY_BATCHES == 0
                or completed_batch == start_batch + 1
                or state["processed"] == Config.PRIOR_SAMPLES
            ):
                elapsed = time.perf_counter() - started
                rate = (state["processed"] - start_batch * Config.PRIOR_BATCH) / max(
                    elapsed, 1e-9
                )
                print(
                    f"prior batch={completed_batch:,}/{total_batches:,} | "
                    f"samples={state['processed']:,}/{Config.PRIOR_SAMPLES:,} | "
                    f"{rate:,.0f} model/s | elapsed={elapsed:.1f}s",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C，正在保存 checkpoint 和当前部分结果。", flush=True)
        save_checkpoint(root, state, generator, signature)

    elapsed = time.perf_counter() - started
    return state, interrupted, elapsed


# =============================================================================
# 静态 hard/Gibbs 势、路径守恒与规则难度
# =============================================================================


def hard_state_masses(
    outputs: np.ndarray,
    state_digits: np.ndarray,
    function_probabilities: np.ndarray,
) -> np.ndarray:
    masses = np.empty(len(state_digits), dtype=np.float64)
    for start in range(0, len(state_digits), 512):
        digits = state_digits[start : start + 512]
        observed = digits[:, None, :] != 0
        labels = digits[:, None, :] - 1
        consistent = np.all(
            (~observed) | (labels == outputs[None, :, :]), axis=2
        )
        masses[start : start + len(digits)] = (
            consistent.astype(np.float64) @ function_probabilities
        )
    return masses


def subset_paths() -> np.ndarray:
    permutations = np.asarray(
        list(itertools.permutations(range(Config.INPUT_COUNT))), dtype=np.int16
    )
    paths = np.zeros((len(permutations), Config.INPUT_COUNT + 1), dtype=np.uint16)
    for step in range(Config.INPUT_COUNT):
        paths[:, step + 1] = paths[:, step] | (
            np.uint16(1) << permutations[:, step].astype(np.uint16)
        )
    return paths


def subset_sizes() -> np.ndarray:
    return np.asarray(
        [int(mask).bit_count() for mask in range(2**Config.INPUT_COUNT)],
        dtype=np.int8,
    )


def stage_and_shapley_rows(
    rule: int,
    measure: str,
    phi_state_bits: np.ndarray,
    state_keys: np.ndarray,
    sizes: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phi_subset = phi_state_bits[state_keys]
    stage_rows: list[dict[str, Any]] = []
    shapley = np.zeros(Config.INPUT_COUNT, dtype=np.float64)
    factorial = math.factorial
    normalizer = factorial(Config.INPUT_COUNT)

    for size in range(Config.INPUT_COUNT):
        increments: list[float] = []
        for subset in range(2**Config.INPUT_COUNT):
            if sizes[subset] != size:
                continue
            for sample_index in range(Config.INPUT_COUNT):
                if (subset >> sample_index) & 1:
                    continue
                next_subset = subset | (1 << sample_index)
                delta = float(phi_subset[next_subset] - phi_subset[subset])
                increments.append(delta)
                weight = (
                    factorial(size)
                    * factorial(Config.INPUT_COUNT - size - 1)
                    / normalizer
                )
                shapley[sample_index] += weight * delta
        values = np.asarray(increments, dtype=np.float64)
        stage_rows.append({
            "rule": rule,
            "rule_name": RULE_NAMES.get(rule, ""),
            "measure": measure,
            "observed_count_before": size,
            "edge_count": len(values),
            "mean_increment_bits": float(values.mean()),
            "median_increment_bits": float(np.median(values)),
            "min_increment_bits": float(values.min()),
            "max_increment_bits": float(values.max()),
            "std_increment_bits": float(values.std()),
        })

    shapley_rows = [
        {
            "rule": rule,
            "rule_name": RULE_NAMES.get(rule, ""),
            "measure": measure,
            "input_index": index,
            "input_bits": format(index, f"0{Config.INPUT_BITS}b"),
            "target": int((rule >> index) & 1),
            "shapley_information_bits": float(shapley[index]),
        }
        for index in range(Config.INPUT_COUNT)
    ]
    return stage_rows, shapley_rows


def invariant_row(
    rule: int,
    measure: str,
    phi_state_bits: np.ndarray,
    state_keys: np.ndarray,
    paths: np.ndarray,
) -> dict[str, Any]:
    phi_subset = phi_state_bits[state_keys]
    path_phi = phi_subset[paths]
    increments = np.diff(path_phi, axis=1)
    totals = increments.sum(axis=1)
    endpoint = float(phi_subset[-1] - phi_subset[0])
    errors = totals - endpoint
    return {
        "rule": rule,
        "rule_hex": f"0x{rule:02X}",
        "rule_name": RULE_NAMES.get(rule, ""),
        "measure": measure,
        "order_count": len(paths),
        "endpoint_cost_bits": endpoint,
        "mean_path_total_bits": float(totals.mean()),
        "std_path_total_bits": float(totals.std()),
        "min_path_total_bits": float(totals.min()),
        "max_path_total_bits": float(totals.max()),
        "max_abs_invariance_error_bits": float(np.max(np.abs(errors))),
    }


def predictive_normalization_error(
    log_mass: np.ndarray,
    state_digits: np.ndarray,
) -> float:
    """检查每个未观察输入的两个标签子状态是否构成完整分割。"""
    powers = 3 ** np.arange(Config.INPUT_COUNT, dtype=np.int64)
    maximum = 0.0
    for state_key, digits in enumerate(state_digits):
        for input_index in range(Config.INPUT_COUNT):
            if digits[input_index] != 0:
                continue
            zero_key = state_key + int(powers[input_index])
            one_key = state_key + 2 * int(powers[input_index])
            q_zero = math.exp(float(log_mass[zero_key] - log_mass[state_key]))
            q_one = math.exp(float(log_mass[one_key] - log_mass[state_key]))
            maximum = max(maximum, abs(q_zero + q_one - 1.0))
    return maximum


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(
    root: Path,
    state: dict[str, Any],
    interrupted: bool,
    sampling_seconds: float,
) -> dict[str, Any]:
    processed = int(state["processed"])
    if processed <= 0:
        raise RuntimeError("尚无 prior 样本可分析")

    outputs = all_function_outputs()
    state_digits = all_state_digits()
    features = function_feature_rows(outputs)
    write_csv(root / "function_features.csv", features)

    raw_counts = state["function_counts"].astype(np.float64)
    alpha = float(Config.HARD_FUNCTION_ALPHA)
    smoothed_probs = (raw_counts + alpha) / (
        processed + alpha * Config.FUNCTION_COUNT
    )
    raw_probs = raw_counts / processed

    hard_masses = hard_state_masses(outputs, state_digits, smoothed_probs)
    hard_phi_bits = -np.log2(np.maximum(hard_masses, np.finfo(np.float64).tiny))

    function_rows = []
    for feature in features:
        rule = int(feature["rule"])
        function_rows.append({
            **feature,
            "prior_count": int(raw_counts[rule]),
            "prior_probability_raw": float(raw_probs[rule]),
            "prior_probability_smoothed": float(smoothed_probs[rule]),
            "hard_prior_surprisal_bits": float(-math.log2(smoothed_probs[rule])),
        })
    write_csv(root / "prior_function_counts.csv", function_rows)

    log_n = math.log(processed)
    log_z = state["log_sum_w"] - log_n
    log_z2 = state["log_sum_w2"] - log_n
    phi_bits = -log_z / math.log(2.0)
    ess = np.exp(
        np.minimum(
            2.0 * state["log_sum_w"] - state["log_sum_w2"],
            math.log(processed),
        )
    )
    ess_fraction = ess / processed

    # 采样误差会让 empty state 偏离 0 极小量；统一扣除基线。
    phi_bits = phi_bits - phi_bits[:, :1]

    state_rows: list[dict[str, Any]] = []
    for state_key, digits in enumerate(state_digits):
        observed_count = int(np.count_nonzero(digits))
        state_rows.append({
            "measure": "hard_conditioned_prior",
            "state_key": state_key,
            "state_digits": "".join(map(str, digits.tolist())),
            "observed_count": observed_count,
            "mass": float(hard_masses[state_key]),
            "phi_bits": float(hard_phi_bits[state_key]),
            "ess": None,
            "ess_fraction": None,
            "supported": bool(hard_masses[state_key] > 0),
        })
        for beta_index, beta in enumerate(Config.BETAS):
            state_rows.append({
                "measure": f"gibbs_beta_{beta:g}",
                "state_key": state_key,
                "state_digits": "".join(map(str, digits.tolist())),
                "observed_count": observed_count,
                "mass": float(math.exp(log_z[beta_index, state_key])),
                "phi_bits": float(phi_bits[beta_index, state_key]),
                "ess": float(ess[beta_index, state_key]),
                "ess_fraction": float(ess_fraction[beta_index, state_key]),
                "supported": bool(
                    ess[beta_index, state_key] >= Config.MIN_SUPPORTED_ESS
                ),
            })
    write_csv(root / "state_free_energy.csv", state_rows)

    paths = subset_paths()
    sizes = subset_sizes()
    invariant_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    shapley_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    costs_by_measure: dict[str, np.ndarray] = {}

    measures: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        ("hard_conditioned_prior", hard_phi_bits, None)
    ]
    for beta_index, beta in enumerate(Config.BETAS):
        measures.append(
            (f"gibbs_beta_{beta:g}", phi_bits[beta_index], ess[beta_index])
        )

    for measure, measure_phi, measure_ess in measures:
        costs = np.zeros(Config.FUNCTION_COUNT, dtype=np.float64)
        for rule in range(Config.FUNCTION_COUNT):
            keys = rule_state_keys(outputs[rule])
            row = invariant_row(rule, measure, measure_phi, keys, paths)
            invariant_rows.append(row)
            costs[rule] = row["endpoint_cost_bits"]

            local_stage, local_shapley = stage_and_shapley_rows(
                rule, measure, measure_phi, keys, sizes
            )
            stage_rows.extend(local_stage)
            shapley_rows.extend(local_shapley)

            stage_sum = float(sum(
                item["mean_increment_bits"] for item in local_stage
            ))
            shapley_sum = float(sum(
                item["shapley_information_bits"] for item in local_shapley
            ))
            row["stage_mean_sum_bits"] = stage_sum
            row["stage_decomposition_error_bits"] = float(
                stage_sum - row["endpoint_cost_bits"]
            )
            row["shapley_sum_bits"] = shapley_sum
            row["shapley_efficiency_error_bits"] = float(
                shapley_sum - row["endpoint_cost_bits"]
            )

            full_key = int(keys[-1])
            difficulty_rows.append({
                **features[rule],
                "measure": measure,
                "difficulty_bits": float(costs[rule]),
                "full_state_key": full_key,
                "full_state_ess": (
                    None if measure_ess is None else float(measure_ess[full_key])
                ),
                "supported": (
                    True
                    if measure_ess is None
                    else bool(measure_ess[full_key] >= Config.MIN_SUPPORTED_ESS)
                ),
            })
        costs_by_measure[measure] = costs

    # 1=最难，另给 1=最易，避免排序方向歧义。
    for row in difficulty_rows:
        values = costs_by_measure[row["measure"]]
        row["hardest_rank"] = int(
            1 + np.count_nonzero(values > row["difficulty_bits"] + 1e-12)
        )
        row["easiest_rank"] = int(
            1 + np.count_nonzero(values < row["difficulty_bits"] - 1e-12)
        )

    write_csv(root / "invariant_verification.csv", invariant_rows)
    write_csv(root / "stage_information.csv", stage_rows)
    write_csv(root / "sample_shapley_information.csv", shapley_rows)
    write_csv(root / "rule_difficulty.csv", difficulty_rows)

    micro_rows: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(
        Config.MICROCANONICAL_MEAN_BCE_THRESHOLDS
    ):
        counts = state["micro_counts"][threshold_index]
        for rule in range(Config.FUNCTION_COUNT):
            count = int(counts[rule])
            raw_probability = count / processed
            jeffreys_probability = (count + 0.5) / (processed + 1.0)
            micro_rows.append({
                **features[rule],
                "mean_bce_threshold": float(threshold),
                "count": count,
                "probability_raw": float(raw_probability),
                "probability_jeffreys": float(jeffreys_probability),
                "volume_cost_bits_jeffreys": float(
                    -math.log2(jeffreys_probability)
                ),
                "supported": bool(count >= Config.MIN_SUPPORTED_MICRO_COUNT),
            })
    write_csv(root / "microcanonical_full_rule_volume.csv", micro_rows)

    proxy_names = (
        "positive_count",
        "essential_variables",
        "anf_degree",
        "anf_terms",
        "anf_literals",
        "total_influence",
        "walsh_min_order",
        "walsh_max_order",
        "walsh_entropy_bits",
    )
    proxy_rows: list[dict[str, Any]] = []
    for measure, costs in costs_by_measure.items():
        for proxy in proxy_names:
            values = np.asarray([row[proxy] for row in features], dtype=np.float64)
            proxy_rows.append({
                "measure": measure,
                "proxy": proxy,
                "spearman_rho": spearman(costs, values),
            })
    write_csv(root / "difficulty_proxy_correlations.csv", proxy_rows)

    logits_path = None
    if state["saved_logits"]:
        logits_sample = np.concatenate(state["saved_logits"], axis=0)
        logits_path = root / "prior_logits_sample_float16.npy"
        np.save(logits_path, logits_sample)

    max_invariance_error = max(
        float(row["max_abs_invariance_error_bits"]) for row in invariant_rows
    )
    max_stage_error = max(
        abs(float(row["stage_decomposition_error_bits"]))
        for row in invariant_rows
    )
    max_shapley_error = max(
        abs(float(row["shapley_efficiency_error_bits"]))
        for row in invariant_rows
    )
    hard_normalization_error = predictive_normalization_error(
        np.log(np.maximum(hard_masses, np.finfo(np.float64).tiny)),
        state_digits,
    )
    beta1_index = list(Config.BETAS).index(1.0)
    beta1_normalization_error = predictive_normalization_error(
        log_z[beta1_index], state_digits
    )
    beta1_name = "gibbs_beta_1"
    beta1_rule150 = next(
        row
        for row in difficulty_rows
        if row["measure"] == beta1_name and row["rule"] == 150
    )
    hard_rule150 = next(
        row
        for row in difficulty_rows
        if row["measure"] == "hard_conditioned_prior" and row["rule"] == 150
    )
    all_static_pass = bool(
        max_invariance_error <= 1e-9
        and max_stage_error <= 1e-9
        and max_shapley_error <= 1e-9
        and hard_normalization_error <= 1e-9
        and beta1_normalization_error <= 1e-5
    )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "interrupted": interrupted,
        "processed_prior_samples": processed,
        "target_prior_samples": Config.PRIOR_SAMPLES,
        "sampling_seconds_current_run": sampling_seconds,
        "state_count": len(state_digits),
        "function_count": Config.FUNCTION_COUNT,
        "order_count_per_rule": math.factorial(Config.INPUT_COUNT),
        "measure_count": len(measures),
        "max_abs_invariance_error_bits": max_invariance_error,
        "max_abs_stage_decomposition_error_bits": max_stage_error,
        "max_abs_shapley_efficiency_error_bits": max_shapley_error,
        "hard_predictive_normalization_error": hard_normalization_error,
        "beta1_predictive_normalization_error": beta1_normalization_error,
        "static_path_invariant_pass": all_static_pass,
        "rule150": {
            "hard_difficulty_bits": hard_rule150["difficulty_bits"],
            "hard_hardest_rank": hard_rule150["hardest_rank"],
            "beta1_difficulty_bits": beta1_rule150["difficulty_bits"],
            "beta1_hardest_rank": beta1_rule150["hardest_rank"],
            "beta1_full_state_ess": beta1_rule150["full_state_ess"],
            "beta1_supported": beta1_rule150["supported"],
        },
        "interpretation": {
            "confirmed_if_pass": (
                "同一静态参数测度下，逐样本广义惊讶度是自由能势差；"
                "任意样本顺序累计值严格等于完整规则端点成本。"
            ),
            "not_claimed": (
                "该恒等式不说明真实 SGD 是 Gibbs/Bayesian 采样器；"
                "optimizer-ensemble 路径不守恒应作为独立动力学残差测量。"
            ),
        },
        "saved_logit_sample": str(logits_path) if logits_path else None,
    }
    write_json(root / "summary.json", summary)
    return summary


# =============================================================================
# 图表与打包
# =============================================================================


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_plots(root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"跳过绘图：{exc}", flush=True)
        return

    difficulty = read_csv_rows(root / "rule_difficulty.csv")
    stage = read_csv_rows(root / "stage_information.csv")
    selected = set(Config.SELECTED_RULES)

    beta_measures = [f"gibbs_beta_{beta:g}" for beta in Config.BETAS]
    fig, axis = plt.subplots(figsize=(10, 6))
    for rule in Config.SELECTED_RULES:
        rows = [
            row
            for row in difficulty
            if int(row["rule"]) == rule and row["measure"] in beta_measures
        ]
        rows.sort(key=lambda row: beta_measures.index(row["measure"]))
        axis.plot(
            Config.BETAS,
            [float(row["difficulty_bits"]) for row in rows],
            marker="o",
            label=f"{rule} {RULE_NAMES.get(rule, '')}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("inverse temperature beta")
    axis.set_ylabel("canonical rule cost -log2 Z_full (bits)")
    axis.set_title("Static free-energy rule difficulty")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "selected_rule_free_energy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for rule in Config.SELECTED_RULES:
        rows = [
            row
            for row in stage
            if int(row["rule"]) == rule and row["measure"] == "gibbs_beta_1"
        ]
        rows.sort(key=lambda row: int(row["observed_count_before"]))
        axis.plot(
            [int(row["observed_count_before"]) for row in rows],
            [float(row["mean_increment_bits"]) for row in rows],
            marker="o",
            label=f"{rule} {RULE_NAMES.get(rule, '')}",
        )
    axis.set_xlabel("number of samples already observed")
    axis.set_ylabel("mean next-sample predictive surprise (bits)")
    axis.set_title("Where each rule pays its information cost (beta=1)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "selected_rule_stage_information.png", dpi=180)
    plt.close(fig)

    # 全规则 beta=1 vs hard prior 成本。
    hard = {
        int(row["rule"]): float(row["difficulty_bits"])
        for row in difficulty
        if row["measure"] == "hard_conditioned_prior"
    }
    beta1 = {
        int(row["rule"]): float(row["difficulty_bits"])
        for row in difficulty
        if row["measure"] == "gibbs_beta_1"
    }
    fig, axis = plt.subplots(figsize=(7, 7))
    x = np.asarray([hard[rule] for rule in range(Config.FUNCTION_COUNT)])
    y = np.asarray([beta1[rule] for rule in range(Config.FUNCTION_COUNT)])
    axis.scatter(x, y, s=12, alpha=0.55)
    for rule in selected:
        axis.annotate(str(rule), (hard[rule], beta1[rule]), fontsize=8)
    axis.set_xlabel("hard prior surprisal (bits)")
    axis.set_ylabel("beta=1 evidence cost (bits)")
    axis.set_title(f"All 256 rules, Spearman={spearman(x, y):.3f}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "hard_vs_beta1_rule_cost.png", dpi=180)
    plt.close(fig)


def package_results(root: Path) -> Path | None:
    if not Config.PACKAGE_RESULTS:
        return None
    archive = root.parent / f"{root.name}_package.zip"
    excluded = {"checkpoint.pt"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in excluded:
                continue
            handle.write(path, arcname=f"{root.name}/{path.relative_to(root)}")
    return archive


def print_selected_summary(root: Path) -> None:
    rows = read_csv_rows(root / "rule_difficulty.csv")
    measures = ["hard_conditioned_prior"] + [
        f"gibbs_beta_{beta:g}" for beta in Config.BETAS
    ]
    print("\n=== 代表规则静态难度（bits；hardest_rank=1 最难） ===")
    for rule in Config.SELECTED_RULES:
        pieces = []
        for measure in measures:
            row = next(
                item
                for item in rows
                if int(item["rule"]) == rule and item["measure"] == measure
            )
            label = "hard" if measure == "hard_conditioned_prior" else measure.replace(
                "gibbs_beta_", "b"
            )
            pieces.append(
                f"{label}={float(row['difficulty_bits']):.3f}"
                f"(rank {int(row['hardest_rank'])})"
            )
        print(f"rule={rule:3d} {RULE_NAMES.get(rule, ''):20s} | " + " | ".join(pieces))


def main() -> None:
    apply_smoke_overrides()
    payload = config_payload()
    signature = stable_hash(payload)
    root = prepare_result_dir()
    write_json(root / "config.json", {**payload, "config_signature": signature})

    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)

    device = torch.device(Config.DEVICE)
    inputs = all_inputs(device)
    outputs = all_function_outputs()
    layout = parameter_layout(Config.WIDTH)

    print("=== Static Free-Energy Information Invariant ===", flush=True)
    print(f"device={device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network=3->{Config.WIDTH}x2->1 tanh | params={layout.parameter_count:,}",
        flush=True,
    )
    print(
        f"prior samples={Config.PRIOR_SAMPLES:,} | batch={Config.PRIOR_BATCH:,} | "
        f"states=3^8={3**Config.INPUT_COUNT:,} | rules=256 | orders/rule=8!={math.factorial(8):,}",
        flush=True,
    )
    print(f"betas={list(Config.BETAS)}", flush=True)
    print(f"result_dir={root}", flush=True)

    state, interrupted, sampling_seconds = sample_prior(
        root, inputs, outputs, signature
    )
    summary = analyze(root, state, interrupted, sampling_seconds)
    make_plots(root)
    print_selected_summary(root)
    archive = package_results(root)

    print("\n=== 自动判决 ===", flush=True)
    print(
        f"processed={summary['processed_prior_samples']:,} | "
        f"max invariant error={summary['max_abs_invariance_error_bits']:.3e} bits | "
        f"PASS={summary['static_path_invariant_pass']}",
        flush=True,
    )
    print(
        f"stage error={summary['max_abs_stage_decomposition_error_bits']:.3e} | "
        f"Shapley error={summary['max_abs_shapley_efficiency_error_bits']:.3e} | "
        f"hard q0+q1 error={summary['hard_predictive_normalization_error']:.3e} | "
        f"beta1 q0+q1 error={summary['beta1_predictive_normalization_error']:.3e}",
        flush=True,
    )
    print(
        "Rule150 | "
        f"hard={summary['rule150']['hard_difficulty_bits']:.3f} bits "
        f"rank={summary['rule150']['hard_hardest_rank']} | "
        f"beta1={summary['rule150']['beta1_difficulty_bits']:.3f} bits "
        f"rank={summary['rule150']['beta1_hardest_rank']} | "
        f"ESS={summary['rule150']['beta1_full_state_ess']:.1f} | "
        f"supported={summary['rule150']['beta1_supported']}",
        flush=True,
    )
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)
    if interrupted:
        print("本次为 Ctrl+C 后的部分结果；保持 RESUME=True 重跑可继续。", flush=True)


if __name__ == "__main__":
    main()
