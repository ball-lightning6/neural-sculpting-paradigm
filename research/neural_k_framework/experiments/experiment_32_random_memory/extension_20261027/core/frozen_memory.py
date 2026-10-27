"""随机记忆架构扩展；仅依赖标准库、numpy 和 torch，可单独复制到 AutoDL。

发布版只通过 extension_20261027/run.py 显式选择任务，直接执行本文件不采样。
顶部 CONFIG 保留历史默认配置；公开主矩阵范围由 smc/protocols.py 明确覆盖。

科学核心移植自 E32 experiment_8bit_random_memory_scaling.py，动态层布局
参照 benchmark_runtime.py；运行时不导入任何仓库模块。标准正态坐标 theta
经权重和偏置共同的 1/sqrt(fan_in) 缩放；分块 pCN 保持高斯先验，约束为
训练集平均 signed-logaddexp BCE。四个独立池不共享阈值、步长或随机流。

只在完整 level 边界原子提交 checkpoint（含全部使用中的 RNG）。预算和
第一次 Ctrl+C 在边界生效，可能多用一个 level 的时间；强制第二次中断只
回退最后完整 level，不把半层当结果。每个池仅保留最新 CPU 检查点，不保留
历代 GPU 权重。时间切片不会消耗失败重试额度；真正异常有累计次数上限。

主量 c=[K(.001)-K(.002)]/log2(.002/.001)，先同池相减，再取四池中位数，
最后对独立数据表汇总；缺任一端点一律为 null，不补 0。只研究同架构内
c 与 n 的关系，不比较跨架构绝对 K。接受率、谱系和节点前后预测变化均
不是 MCMC ESS，也不能证明混合/收敛；结果仅是有限 loss 的体积估计。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import io
import json
import math
import os
import platform
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


OUTPUT_DIR = Path(__file__).resolve().parent / "results_random_memory"
SMOKE = False
CONFIG = {
    "OUTPUT_DIR": OUTPUT_DIR,
    "DEVICE": "cuda:0",
    "DTYPE": "float32",
    "TF32": False,
    "DETERMINISTIC": True,
    "CPU_THREADS": 4,
    "INPUT_BITS": 8,
    "ARCHITECTURES": ((32,), (64, 32), (32, 32, 32)),
    "COUNTS": tuple(range(16, 257, 16)),
    "DATASETS": 12,
    "POOLS": 4,
    "PARTICLES": 2048,
    "THRESHOLDS": (.70, .69, .65, .60, .50, .40, .30, .20, .10,
                   .05, .025, .01, .005, .002, .001),
    "PRIMARY_WINDOW": (.002, .001),
    "DATA_SEED": 202609112301,
    "PRIOR_SEED": 202609112311,
    "RESAMPLE_SEED": 202609112312,
    "MUTATION_SEED": 202609112313,
    "SURVIVAL_QUANTILE": .5,
    "ADAPT_SWEEPS": 2,
    "MUTATION_SWEEPS": 12,
    "HIDDEN_FIRST_SCALE": .08,
    "HIDDEN_OTHER_SCALE": .06,
    "OUTPUT_SCALE": .08,
    "ALL_SCALE": .025,
    "TARGET_ACCEPTANCE": .30,
    "ADAPT_RATE": .35,
    "MIN_PROPOSAL_SCALE": 1e-4,
    "MAX_PROPOSAL_SCALE": .95,
    "LOSS_TOLERANCE": 1e-9,
    "EVAL_MICRO_BATCH": 2048,
    "DIAGNOSTIC_PARTICLES": 64,
    "TOTAL_SECONDS": 4 * 3600.,
    "SECONDS_PER_REPLICA": 30 * 60.,
    "MAX_LEVELS_PER_POOL": 20000,
    "MAX_STALLED_LEVELS": 64,
    "MAX_FAILURE_ATTEMPTS": 3,
    "MAX_NO_PROGRESS_VISITS": 3,
    "LOG_SECONDS": 30.,
    # 以下两项只用于短测/父进程切片；None 不额外限制。
    "LEVELS_PER_SLICE": None,
    "MAX_REPLICAS_PER_CALL": None,
    "KEEP_COMPLETED_CHECKPOINTS": False,
}
SMOKE_CONFIG = {
    "DEVICE": "cpu", "INPUT_BITS": 2, "ARCHITECTURES": ((3,),),
    "COUNTS": (2, 4), "DATASETS": 2, "POOLS": 2, "PARTICLES": 24,
    "THRESHOLDS": (1.5, 1.0), "PRIMARY_WINDOW": (1.5, 1.0),
    "ADAPT_SWEEPS": 1, "MUTATION_SWEEPS": 2, "EVAL_MICRO_BATCH": 16,
    "DIAGNOSTIC_PARTICLES": 8, "TOTAL_SECONDS": 60.,
    "SECONDS_PER_REPLICA": 15., "CPU_THREADS": 1,
}
if SMOKE:
    CONFIG.update(SMOKE_CONFIG)

PROTOCOL = "random_memory_independent_pcn_smc_v1"
CHECKPOINT_VERSION = 1
PROVENANCE = {
    "e32_source": "research/neural_k_framework/experiments/experiment_32_random_memory/experiment_8bit_random_memory_scaling.py",
    "e32_sha256": "f7959583f8b8cdd118a204976864823263e0a94ef8177eaafb2d4292cc0b4ebc",
    "layout_reference": "research/random_memory_architecture_extension_20261023/benchmark_runtime.py",
    "layout_reference_sha256": "987d2acd7c7e66f3b69667939ed0b9a94e1aa2a043803270a1c12af07ce2657e",
}
OPERATIONAL_KEYS = {
    "OUTPUT_DIR", "TOTAL_SECONDS", "SECONDS_PER_REPLICA", "LOG_SECONDS",
    "MAX_LEVELS_PER_POOL", "MAX_STALLED_LEVELS", "MAX_FAILURE_ATTEMPTS",
    "MAX_NO_PROGRESS_VISITS", "LEVELS_PER_SLICE", "MAX_REPLICAS_PER_CALL",
    "KEEP_COMPLETED_CHECKPOINTS",
}
RELIABILITY = "FINITE_LOSS_ESTIMATE_NOT_CONVERGENCE_CERTIFIED"


class CheckpointMismatch(ValueError):
    """拒绝混合不同科学配置、数据或执行环境。"""


class SamplingFailure(RuntimeError):
    """可有界重试的采样异常。"""


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    stop: int


@dataclass
class PoolState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume_fraction: float
    proposal_scales: list
    current_threshold: float
    threshold_index: int = 0
    level: int = 0
    stalled_levels: int = 0
    elapsed_seconds: float = 0.
    level_rows: list = None
    nodes: dict = None

    def __post_init__(self):
        if self.level_rows is None:
            self.level_rows = []
        if self.nodes is None:
            self.nodes = {}


@dataclass
class StopRequest:
    requested: bool = False
    reason: str = ""


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, np.generic):
        return ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_hash(value):
    text = json.dumps(ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_digest(*arrays):
    # 与 E32 的原始数组哈希逐字节一致，包含形状和 dtype。
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _replace(temporary, destination):
    for attempt in range(8):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    options = {} if binary else {"encoding": "utf-8", "newline": ""}
    with temporary.open("wb" if binary else "w", **options) as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path)


def atomic_json(path, value):
    atomic_write(path, lambda stream: json.dump(ready(value), stream, ensure_ascii=False,
                                              indent=2, allow_nan=False))


def atomic_text(path, value):
    atomic_write(path, lambda stream: stream.write(value))


def atomic_npz(path, **arrays):
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays), binary=True)


def resolve_config(config=None):
    overrides = {} if config is None else dict(config)
    unknown = set(overrides) - set(CONFIG)
    if unknown:
        raise ValueError(f"未知配置项: {sorted(unknown)}")
    result = copy.deepcopy(CONFIG)
    result.update(overrides)
    result["OUTPUT_DIR"] = Path(result["OUTPUT_DIR"]).resolve()
    result["ARCHITECTURES"] = tuple(tuple(h) for h in result["ARCHITECTURES"])
    result["COUNTS"] = tuple(result["COUNTS"])
    result["THRESHOLDS"] = tuple(float(v) for v in result["THRESHOLDS"])
    result["PRIMARY_WINDOW"] = tuple(float(v) for v in result["PRIMARY_WINDOW"])
    for name in ("INPUT_BITS", "DATASETS", "POOLS", "PARTICLES", "CPU_THREADS",
                 "MUTATION_SWEEPS", "EVAL_MICRO_BATCH", "DIAGNOSTIC_PARTICLES",
                 "MAX_LEVELS_PER_POOL", "MAX_STALLED_LEVELS", "MAX_FAILURE_ATTEMPTS",
                 "MAX_NO_PROGRESS_VISITS"):
        if not isinstance(result[name], int) or result[name] < 1:
            raise ValueError(f"{name} 必须为正整数")
    if result["INPUT_BITS"] > 16:
        raise ValueError("INPUT_BITS 最大 16，避免误生成巨大真值表")
    if not result["ARCHITECTURES"] or len(set(result["ARCHITECTURES"])) != len(result["ARCHITECTURES"]):
        raise ValueError("架构必须非空且不重复")
    if any(not h or any(not isinstance(w, int) or w < 1 for w in h)
           for h in result["ARCHITECTURES"]):
        raise ValueError("每个隐藏层宽度必须为正整数")
    counts = result["COUNTS"]
    if not counts or tuple(sorted(set(counts))) != counts or any(
            not isinstance(n, int) or not 1 <= n <= 2 ** result["INPUT_BITS"] for n in counts):
        raise ValueError("COUNTS 必须为不重复递增的有效前缀长度")
    nodes = result["THRESHOLDS"]
    if not nodes or any(not math.isfinite(v) or v <= 0 for v in nodes) or any(
            a <= b for a, b in zip(nodes[:-1], nodes[1:])):
        raise ValueError("THRESHOLDS 必须为严格递减的正数")
    high, low = result["PRIMARY_WINDOW"]
    if high <= low or high not in nodes or low not in nodes:
        raise ValueError("PRIMARY_WINDOW 的高低端点必须存在于 THRESHOLDS")
    if result["DTYPE"] not in ("float32", "float64"):
        raise ValueError("DTYPE 仅支持 float32/float64")
    if result["ADAPT_SWEEPS"] < 0 or not isinstance(result["ADAPT_SWEEPS"], int):
        raise ValueError("ADAPT_SWEEPS 必须为非负整数")
    if not 0 < result["SURVIVAL_QUANTILE"] < 1:
        raise ValueError("SURVIVAL_QUANTILE 必须在 (0,1)")
    if not 0 < result["MIN_PROPOSAL_SCALE"] <= result["MAX_PROPOSAL_SCALE"] < 1:
        raise ValueError("pCN 步长上下限必须在 (0,1)")
    for name in ("HIDDEN_FIRST_SCALE", "HIDDEN_OTHER_SCALE", "OUTPUT_SCALE", "ALL_SCALE"):
        if not result["MIN_PROPOSAL_SCALE"] <= result[name] <= result["MAX_PROPOSAL_SCALE"]:
            raise ValueError(f"{name} 超出 pCN 步长范围")
    if not 0 < result["TARGET_ACCEPTANCE"] < 1 or result["ADAPT_RATE"] < 0:
        raise ValueError("接受率目标或自适应速率无效")
    for name in ("TOTAL_SECONDS", "SECONDS_PER_REPLICA", "LOG_SECONDS", "LOSS_TOLERANCE"):
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"{name} 必须为有限非负数")
    if result["SECONDS_PER_REPLICA"] <= 0:
        raise ValueError("SECONDS_PER_REPLICA 必须大于 0")
    for name in ("LEVELS_PER_SLICE", "MAX_REPLICAS_PER_CALL"):
        if result[name] is not None and (not isinstance(result[name], int) or result[name] < 1):
            raise ValueError(f"{name} 必须为 None 或正整数")
    return result


def make_data(dataset_id, config):
    count = 2 ** config["INPUT_BITS"]
    rng = np.random.default_rng(config["DATA_SEED"] + 1_000_003 * dataset_id)
    order = rng.permutation(count).astype(np.int64)
    labels = rng.integers(0, 2, count, dtype=np.uint8)
    values = np.arange(count, dtype=np.uint32)
    inputs = ((values[:, None] >> np.arange(config["INPUT_BITS"] - 1, -1, -1)) & 1).astype(np.float32)
    return inputs, labels, order


def parameter_blocks(hidden, input_bits=8):
    dimensions = (input_bits, *hidden, 1)
    blocks, layers, cursor = [], [], 0
    for index, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        start = cursor
        weight_end = start + fan_in * fan_out
        cursor = weight_end + fan_out
        layers.append((start, weight_end, cursor, fan_in, fan_out))
        blocks.append(Block(f"layer_{index}", start, cursor))
    blocks.append(Block("all", 0, cursor))
    return blocks, layers, cursor


def forward_logits(theta, inputs, layers):
    batch = theta.shape[0]
    values = inputs[None].expand(batch, -1, -1)
    for index, (start, weight_end, stop, fan_in, fan_out) in enumerate(layers):
        weight = theta[:, start:weight_end].reshape(batch, fan_out, fan_in) / math.sqrt(fan_in)
        bias = theta[:, weight_end:stop] / math.sqrt(fan_in)
        values = torch.bmm(values, weight.transpose(1, 2)) + bias[:, None]
        if index < len(layers) - 1:
            values = torch.tanh(values)
    return values.squeeze(-1)


@torch.no_grad()
def evaluate_losses(particles, inputs, targets, layers, micro_batch):
    values = []
    signs = (2 * targets - 1)[None]
    for start in range(0, len(particles), micro_batch):
        logits = forward_logits(particles[start:start + micro_batch], inputs, layers)
        values.append(torch.logaddexp(torch.zeros_like(logits), -signs * logits).mean(-1))
    losses = torch.cat(values)
    if not bool(torch.isfinite(losses).all()):
        raise SamplingFailure("粒子 loss 出现非有限值")
    return losses


def make_generators(device, job, config):
    identity = {key: job[key] for key in ("hidden", "n", "dataset_id", "pool")}
    offset = int(json_hash(identity)[:15], 16)
    return {name: torch.Generator(device=device).manual_seed((config[seed] + offset) % (2 ** 63 - 1))
            for name, seed in (("prior", "PRIOR_SEED"), ("resample", "RESAMPLE_SEED"),
                               ("mutation", "MUTATION_SEED"))}


def initialize_state(device, x, y, layers, config, generators):
    dimensions = layers[-1][2]
    particles = torch.randn(config["PARTICLES"], dimensions, dtype=getattr(torch, config["DTYPE"]),
                            device=device, generator=generators["prior"])
    losses = evaluate_losses(particles, x, y, layers, config["EVAL_MICRO_BATCH"])
    scales = ([config["HIDDEN_FIRST_SCALE"]] +
              [config["HIDDEN_OTHER_SCALE"]] * (len(layers) - 2) +
              [config["OUTPUT_SCALE"], config["ALL_SCALE"]])
    return PoolState(particles, losses, torch.arange(len(particles), device=device),
                     0., scales, float("inf"))


def choose_next_threshold(state, config):
    target = config["THRESHOLDS"][state.threshold_index]
    adaptive = float(torch.quantile(state.losses, config["SURVIVAL_QUANTILE"]))
    epsilon = min(max(target, adaptive), state.current_threshold)
    return epsilon, epsilon <= target + 1e-12


@torch.no_grad()
def resample(state, epsilon, generator, tolerance):
    survivors = torch.nonzero(state.losses <= epsilon + tolerance).flatten()
    if len(survivors) == 0:
        raise SamplingFailure("当前独立池没有幸存粒子；不伪造体积")
    choices = torch.randint(len(survivors), (len(state.particles),),
                            device=state.particles.device, generator=generator)
    selected = survivors[choices]
    state.particles = state.particles[selected].clone()
    state.losses = state.losses[selected].clone()
    state.lineages = state.lineages[selected].clone()
    survival = len(survivors) / len(selected)
    state.log_volume_fraction += math.log(survival)
    return survival, len(survivors)


def pcn_proposal(particles, block, scale, generator):
    if not 0 <= scale < 1:
        raise ValueError("pCN scale 必须在 [0,1)")
    proposal = particles.clone()
    current = particles[:, block.start:block.stop]
    noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
    proposal[:, block.start:block.stop] = math.sqrt(1 - scale * scale) * current + scale * noise
    return proposal


@torch.no_grad()
def mutate_block(state, block, scale, epsilon, x, y, layers, config, generator):
    proposal = pcn_proposal(state.particles, block, scale, generator)
    losses = evaluate_losses(proposal, x, y, layers, config["EVAL_MICRO_BATCH"])
    accepted = losses <= epsilon + config["LOSS_TOLERANCE"]
    state.particles[accepted] = proposal[accepted]
    state.losses[accepted] = losses[accepted]
    return float(accepted.double().mean())


def rejuvenate(state, blocks, epsilon, x, y, layers, config, generator):
    scales = list(state.proposal_scales)
    adapt_rates = np.zeros(len(blocks), dtype=np.float64)
    for _ in range(config["ADAPT_SWEEPS"]):
        for index, block in enumerate(blocks):
            rate = mutate_block(state, block, scales[index], epsilon, x, y, layers, config, generator)
            adapt_rates[index] += rate
            scales[index] *= math.exp(config["ADAPT_RATE"] * (rate - config["TARGET_ACCEPTANCE"]))
            scales[index] = min(max(scales[index], config["MIN_PROPOSAL_SCALE"]),
                                config["MAX_PROPOSAL_SCALE"])
    state.proposal_scales = scales
    rates = np.zeros(len(blocks), dtype=np.float64)
    for _ in range(config["MUTATION_SWEEPS"]):
        for index, block in enumerate(blocks):
            rates[index] += mutate_block(state, block, scales[index], epsilon, x, y, layers, config, generator)
    return (rates / config["MUTATION_SWEEPS"]).tolist(), (
        (adapt_rates / config["ADAPT_SWEEPS"]).tolist() if config["ADAPT_SWEEPS"] else None)


def prediction_change(before, after):
    return {
        "particle_count": len(before),
        "mean_abs_probability_change": float((before.sigmoid() - after.sigmoid()).abs().mean()),
        "rms_logit_change": float((before - after).square().mean().sqrt()),
        "classification_flip_fraction": float(((before >= 0) != (after >= 0)).double().mean()),
        "meaning": "同一节点重采样后、pCN 更新前后配对预测变化；不是 ESS",
    }


@torch.no_grad()
def node_diagnostics(state, epsilon, x, y, layers, config, rates, change):
    correct, all_correct = 0, 0
    for start in range(0, len(state.particles), config["EVAL_MICRO_BATCH"]):
        logits = forward_logits(state.particles[start:start + config["EVAL_MICRO_BATCH"]], x, layers)
        ok = (logits >= 0) == y.bool()[None]
        correct += int(ok.sum())
        all_correct += int(ok.all(-1).sum())
    count = len(state.particles)
    if epsilon + config["LOSS_TOLERANCE"] < math.log(2) / len(x) and all_correct != count:
        raise SamplingFailure("BCE 已低于 ln(2)/n，却存在训练分类错误")
    _, lineage_counts = torch.unique(state.lineages, return_counts=True)
    frequencies = lineage_counts.double() / count
    warnings = []
    if min(rates) < .03:
        warnings.append("LOW_BLOCK_ACCEPTANCE")
    if len(lineage_counts) < max(2, .01 * count):
        warnings.append("LINEAGE_COLLAPSE_NOT_ESS")
    return {
        "epsilon": epsilon, "level": state.level, "K_bits": -state.log_volume_fraction / math.log(2),
        "mean_bce": float(state.losses.mean()), "max_bce": float(state.losses.max()),
        "training_bit_accuracy": correct / (count * len(x)),
        "training_all_correct_fraction": all_correct / count,
        "initial_lineages_remaining": len(lineage_counts),
        "lineage_inverse_simpson_not_mcmc_ess": float(1 / frequencies.square().sum()),
        "acceptance_by_block": rates, "proposal_scales": list(state.proposal_scales),
        "prediction_change_not_ess": change, "warnings": warnings,
    }


@torch.no_grad()
def one_level(state, blocks, layers, x, y, config, generators):
    epsilon, hit = choose_next_threshold(state, config)
    previous = state.current_threshold
    survival, survivor_count = resample(state, epsilon, generators["resample"], config["LOSS_TOLERANCE"])
    before = None
    if hit:
        sample = state.particles[:config["DIAGNOSTIC_PARTICLES"]]
        before = forward_logits(sample, x, layers)
    rates, adapt_rates = rejuvenate(state, blocks, epsilon, x, y, layers, config, generators["mutation"])
    if float(state.losses.max()) > epsilon + config["LOSS_TOLERANCE"] + 2e-7 * max(epsilon, 1e-3):
        raise SamplingFailure("更新后的粒子违反 BCE 约束")
    state.current_threshold = epsilon
    state.threshold_index += int(hit)
    state.level += 1
    state.stalled_levels = state.stalled_levels + 1 if epsilon >= previous else 0
    _, lineage_counts = torch.unique(state.lineages, return_counts=True)
    row = {
        "level": state.level, "epsilon": epsilon, "hit_node": hit,
        "survival_fraction": survival, "survivor_count": survivor_count,
        "K_bits": -state.log_volume_fraction / math.log(2),
        "initial_lineages_remaining": len(lineage_counts),
        "acceptance_by_block": rates, "adapt_acceptance_by_block": adapt_rates,
        "proposal_scales": list(state.proposal_scales),
    }
    state.level_rows.append(row)
    if hit:
        after = forward_logits(state.particles[:len(before)], x, layers)
        node = node_diagnostics(state, epsilon, x, y, layers, config, rates, prediction_change(before, after))
        state.nodes[node_key(epsilon)] = node
    return row


def node_key(value):
    return format(float(value), ".12g")


def science_signature(config):
    core = (Block, PoolState, array_digest, make_data, parameter_blocks, forward_logits,
            evaluate_losses, make_generators, initialize_state, choose_next_threshold,
            resample, pcn_proposal, mutate_block, rejuvenate, prediction_change,
            node_diagnostics, one_level, node_key)
    source = "\n".join(inspect.getsource(value) for value in core)
    signature = {
        "protocol": PROTOCOL, "checkpoint_version": CHECKPOINT_VERSION,
        "scientific_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "config": {key: value for key, value in config.items() if key not in OPERATIONAL_KEYS},
        "coordinate_prior": "iid N(0,1); weights AND biases scaled by 1/sqrt(fan_in)",
        "loss": "mean signed-logaddexp BCE on selected training prefix only",
        "log_volume_dtype": "IEEE float64", "amp": False,
        "independence": "各池独立随机流、阈值、自适应步长和重采样；无跨池状态共享",
    }
    return ready(signature), json_hash(signature)


def pack_state(state, generators, signature, environment_identity):
    values = {}
    for field in fields(PoolState):
        value = getattr(state, field.name)
        values[field.name] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return {
        "version": CHECKPOINT_VERSION, "signature": signature,
        "environment_identity": environment_identity, "state": values,
        "rng": {name: generator.get_state().cpu().clone() for name, generator in generators.items()},
        "boundary": "complete_level_only", "source_sha256": source_sha256(),
    }


def save_checkpoint(path, state, generators, signature, environment_identity):
    payload = pack_state(state, generators, signature, environment_identity)
    atomic_write(path, lambda stream: torch.save(payload, stream), binary=True)


def load_checkpoint(path, generators, device, signature, environment_identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION or payload.get("signature") != signature:
        raise CheckpointMismatch(f"{path}: 源科学配置/数据 hash 不匹配，拒绝续跑")
    if payload.get("environment_identity") != environment_identity:
        raise CheckpointMismatch(f"{path}: torch/设备/dtype/TF32 执行环境变化，拒绝混合续跑")
    if payload.get("boundary") != "complete_level_only" or set(payload["rng"]) != set(generators):
        raise CheckpointMismatch(f"{path}: 非完整 level 或 RNG 不完整")
    for name, generator in generators.items():
        generator.set_state(payload["rng"][name])
    values = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in payload["state"].items()}
    return PoolState(**values)


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@contextmanager
def execution_environment(config):
    device = torch.device(config["DEVICE"])
    if device.type not in ("cuda", "cpu"):
        raise ValueError("仅支持显式 cuda 或 CPU 短测")
    if device.type == "cuda" and (not torch.cuda.is_available() or
                                  (device.index or 0) >= torch.cuda.device_count()):
        raise RuntimeError(f"配置设备 {device} 不可用；默认实验要求 CUDA，不会静默退回 CPU。请检查 GPU 版 torch 和驱动。")
    old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_float32_matmul_precision(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    old_device = torch.cuda.current_device() if device.type == "cuda" else None
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.set_num_threads(config["CPU_THREADS"])
        torch.set_float32_matmul_precision("high" if config["TF32"] else "highest")
        torch.backends.cuda.matmul.allow_tf32 = config["TF32"]
        torch.backends.cudnn.allow_tf32 = config["TF32"]
        torch.use_deterministic_algorithms(config["DETERMINISTIC"])
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": str(np.__version__), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device), "dtype": config["DTYPE"],
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic": torch.are_deterministic_algorithms_enabled(), "amp": False,
            "cpu_threads": torch.get_num_threads(), "torch_build_config": torch.__config__.show(),
            "environment_variables": {name: os.environ.get(name) for name in
                                      ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE")},
        }
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            environment.update(gpu=properties.name, gpu_total_memory_bytes=properties.total_memory,
                               compute_capability=list(torch.cuda.get_device_capability(device)))
        else:
            environment["gpu"] = None
        identity = json_hash(environment)
        yield device, environment, identity
    finally:
        torch.set_num_threads(old[0])
        torch.use_deterministic_algorithms(old[1], warn_only=old[2])
        torch.set_float32_matmul_precision(old[3])
        torch.backends.cuda.matmul.allow_tf32 = old[4]
        torch.backends.cudnn.allow_tf32 = old[5]
        if old_device is not None:
            torch.cuda.set_device(old_device)


@contextmanager
def stop_signals(stop):
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            if stop.requested:
                raise KeyboardInterrupt
            stop.requested, stop.reason = True, "keyboard_interrupt"
            print("已请求停止：完成当前 level 并提交原子检查点后退出。", flush=True)

        signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def make_jobs(config):
    jobs = []
    for n in config["COUNTS"]:
        for dataset_id in range(config["DATASETS"]):
            for hidden in config["ARCHITECTURES"]:
                architecture = "h" + "_".join(str(width) for width in hidden)
                for pool in range(config["POOLS"]):
                    key = f"{architecture}/data_{dataset_id:03d}/n{n:03d}/pool_{pool}"
                    jobs.append({
                        "key": key, "architecture": architecture, "hidden": list(hidden),
                        "n": n, "dataset_id": dataset_id, "pool": pool,
                        "status": "pending", "level": 0, "nodes": {}, "reached_bce": None,
                        "failure_attempts": 0, "no_progress_visits": 0, "visits": 0,
                        "elapsed_seconds": 0., "last_error": None,
                    })
    return jobs


def prepare_manifest(config, signature, science_hash):
    out = config["OUTPUT_DIR"]
    path = out / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("science_hash") != science_hash or manifest.get("science") != signature:
            raise CheckpointMismatch("结果目录的源科学配置 hash 不匹配；请使用新 OUTPUT_DIR，预算可单独修改。")
    else:
        manifest = {"science_hash": science_hash, "science": signature, "provenance": PROVENANCE,
                    "queue_cursor": 0, "jobs": make_jobs(config), "calls": [], "datasets": {}}
    for dataset_id in range(config["DATASETS"]):
        inputs, labels, order = make_data(dataset_id, config)
        raw_hash = array_digest(inputs, labels, order)
        key = str(dataset_id)
        if key in manifest["datasets"] and manifest["datasets"][key]["raw_dataset_sha256"] != raw_hash:
            raise CheckpointMismatch(f"原始数据表 {dataset_id} 的 hash 不匹配")
        data_path = out / "datasets" / f"data_{dataset_id:03d}.npz"
        if data_path.exists():
            with np.load(data_path, allow_pickle=False) as stored:
                actual = array_digest(stored["inputs"], stored["labels"], stored["order"])
            if actual != raw_hash:
                raise CheckpointMismatch(f"{data_path}: 已落盘原始数据被修改，拒绝续跑")
        else:
            atomic_npz(data_path, inputs=inputs, labels=labels, order=order)
        manifest["datasets"][key] = {
            "raw_dataset_sha256": raw_hash, "path": str(data_path),
            "prefix_sha256": {str(n): array_digest(inputs[order[:n]], labels[order[:n]], order[:n])
                              for n in config["COUNTS"]},
        }
    return manifest


def job_signature(job, manifest):
    data = manifest["datasets"][str(job["dataset_id"])]
    return {"science_hash": manifest["science_hash"], "key": job["key"],
            "raw_dataset_sha256": data["raw_dataset_sha256"],
            "training_prefix_sha256": data["prefix_sha256"][str(job["n"])],
            "hidden": job["hidden"], "dataset_id": job["dataset_id"], "n": job["n"], "pool": job["pool"]}


def update_job(job, state, config):
    job.update(level=state.level, nodes=copy.deepcopy(state.nodes),
               reached_bce=ready(state.current_threshold), elapsed_seconds=state.elapsed_seconds,
               status="completed" if state.threshold_index == len(config["THRESHOLDS"]) else "partial")


def export_pool(path, state, job, signature):
    atomic_json(path / "diagnostics.json", {
        "job": job, "signature": signature, "levels": state.level_rows, "nodes": state.nodes,
        "checkpoint_boundary": "完整 level；半层回退最后原子检查点",
        "reliability": RELIABILITY,
    })


def run_slice(job, config, manifest, device, environment_identity, deadline, stop, progress):
    started = time.perf_counter()
    path = config["OUTPUT_DIR"] / job["key"]
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "checkpoint.pt"
    signature = job_signature(job, manifest)
    inputs, labels, order = make_data(job["dataset_id"], config)
    selected = order[:job["n"]]
    dtype = getattr(torch, config["DTYPE"])
    x = torch.tensor(inputs[selected], device=device, dtype=dtype)
    y = torch.tensor(labels[selected], device=device, dtype=dtype)
    blocks, layers, _ = parameter_blocks(job["hidden"], config["INPUT_BITS"])
    generators = make_generators(device, job, config)
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, generators, device, signature, environment_identity)
    else:
        state = initialize_state(device, x, y, layers, config, generators)
        save_checkpoint(checkpoint, state, generators, signature, environment_identity)
    base_seconds, first_level = state.elapsed_seconds, state.level
    update_job(job, state, config)
    logged = started
    error = None
    try:
        while state.threshold_index < len(config["THRESHOLDS"]):
            now = time.perf_counter()
            if stop.requested or (config["OUTPUT_DIR"] / "STOP").exists():
                stop.requested = True
                stop.reason = stop.reason or "STOP_file"
                break
            if now >= deadline or now - started >= config["SECONDS_PER_REPLICA"]:
                break
            if config["LEVELS_PER_SLICE"] is not None and state.level - first_level >= config["LEVELS_PER_SLICE"]:
                break
            if state.level >= config["MAX_LEVELS_PER_POOL"]:
                job.update(status="failed", last_error="达到 MAX_LEVELS_PER_POOL；保留已达到节点")
                break
            if state.stalled_levels >= config["MAX_STALLED_LEVELS"]:
                raise SamplingFailure("阈值长期不下降，达到 MAX_STALLED_LEVELS")
            row = one_level(state, blocks, layers, x, y, config, generators)
            state.elapsed_seconds = base_seconds + time.perf_counter() - started
            # 每层先提交含 RNG 的检查点，汇总文件可由该检查点恢复。
            save_checkpoint(checkpoint, state, generators, signature, environment_identity)
            update_job(job, state, config)
            now = time.perf_counter()
            if row["hit_node"] or now - logged >= config["LOG_SECONDS"]:
                export_pool(path, state, job, signature)
                progress()
                print(f"{job['key']} level={state.level} BCE<={state.current_threshold:.6g} "
                      f"K={-state.log_volume_fraction / math.log(2):.3f} bits", flush=True)
                logged = now
    except KeyboardInterrupt:
        stop.requested, stop.reason = True, "keyboard_interrupt"
        state = load_checkpoint(checkpoint, generators, device, signature, environment_identity)
        update_job(job, state, config)
    except CheckpointMismatch:
        raise
    except Exception:
        error = traceback.format_exc()
        state = load_checkpoint(checkpoint, generators, device, signature, environment_identity)
        update_job(job, state, config)
    if job["status"] != "failed":
        update_job(job, state, config)
    export_pool(path, state, job, signature)
    return {"error": error, "progressed": state.level > first_level,
            "budget_exhausted": time.perf_counter() >= deadline or
                                time.perf_counter() - started >= config["SECONDS_PER_REPLICA"]}


def pool_window(nodes, high, low):
    a, b = nodes.get(node_key(high)), nodes.get(node_key(low))
    if a is None or b is None:
        return None
    return (b["K_bits"] - a["K_bits"]) / math.log2(high / low)


def summarize(manifest, config):
    rows = []
    high, low = config["PRIMARY_WINDOW"]
    index = {(job["architecture"], job["dataset_id"], job["n"], job["pool"]): job
             for job in manifest["jobs"]}
    for hidden in config["ARCHITECTURES"]:
        architecture = "h" + "_".join(str(width) for width in hidden)
        for n in config["COUNTS"]:
            for dataset_id in range(config["DATASETS"]):
                jobs = [index[(architecture, dataset_id, n, pool)] for pool in range(config["POOLS"])]
                coefficients = [pool_window(job["nodes"], high, low) for job in jobs]
                complete = all(value is not None for value in coefficients)
                center = float(np.median(coefficients)) if complete else None
                row = {
                    "architecture": architecture, "hidden": list(hidden), "n": n, "dataset_id": dataset_id,
                    "raw_dataset_sha256": manifest["datasets"][str(dataset_id)]["raw_dataset_sha256"],
                    "training_prefix_sha256": manifest["datasets"][str(dataset_id)]["prefix_sha256"][str(n)],
                    "status": "failed" if any(job["status"] == "failed" for job in jobs) else
                              ("completed" if all(job["status"] == "completed" for job in jobs) else "partial"),
                    "pool_statuses": [job["status"] for job in jobs],
                    "completed_pools": sum(job["status"] == "completed" for job in jobs),
                    "epsilon_high": high, "epsilon_low": low, "c_by_pool": coefficients,
                    "c_dataset_median": center, "c_per_n": center / n if complete else None,
                    "c_pool_sd": float(np.std(coefficients, ddof=1)) if complete and len(jobs) > 1 else None,
                    "c_complete_pools": sum(value is not None for value in coefficients),
                    "levels_by_pool": [job["level"] for job in jobs],
                }
                for epsilon in config["THRESHOLDS"]:
                    row[f"K_{node_key(epsilon)}_bits_by_pool"] = [
                        job["nodes"].get(node_key(epsilon), {}).get("K_bits") for job in jobs]
                rows.append(row)
    groups, fits = [], []
    for hidden in config["ARCHITECTURES"]:
        architecture = "h" + "_".join(str(width) for width in hidden)
        for n in config["COUNTS"]:
            available = [row["c_dataset_median"] for row in rows
                         if row["architecture"] == architecture and row["n"] == n and row["c_dataset_median"] is not None]
            groups.append({
                "architecture": architecture, "n": n, "complete_datasets": len(available),
                "expected_datasets": config["DATASETS"], "all_datasets_complete": len(available) == config["DATASETS"],
                "c_mean_of_dataset_medians": float(np.mean(available)) if available else None,
                "c_sd_across_datasets": float(np.std(available, ddof=1)) if len(available) > 1 else None,
            })
        points = [group for group in groups if group["architecture"] == architecture and group["all_datasets_complete"]]
        fit = {"architecture": architecture, "complete_n_values": [point["n"] for point in points],
               "slope_c_vs_n": None, "intercept": None,
               "scope": "仅同架构；仅使用全部独立数据表主窗口齐全的 n；缺失不插值"}
        if len(points) >= 2:
            design = np.array([[point["n"], 1.] for point in points], dtype=np.float64)
            response = np.array([point["c_mean_of_dataset_medians"] for point in points])
            slope, intercept = np.linalg.lstsq(design, response, rcond=None)[0]
            fit.update(slope_c_vs_n=float(slope), intercept=float(intercept))
        fits.append(fit)
    return {"science_hash": manifest["science_hash"], "reliability": RELIABILITY,
            "primary_window": [high, low], "rows": rows, "by_architecture_n": groups,
            "within_architecture_fits": fits,
            "missing_value_policy": "未达到节点为 null/CSV 空白；不补 0；四池端点齐全才给数据表主 c",
            "diagnostic_caveat": "接受率、谱系逆集中度、节点前后预测变化均不是 MCMC ESS；自适应和混合偏差不由池间差异覆盖"}


def write_reports(manifest, config, reason="running", error=None):
    jobs = manifest["jobs"]
    completed = sum(job["status"] == "completed" for job in jobs)
    failed = sum(job["status"] == "failed" for job in jobs)
    status = "partial" if completed + failed < len(jobs) else ("completed_with_failures" if failed else "completed")
    result = {
        "status": status, "reason": reason, "science_hash": manifest["science_hash"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total_pools": len(jobs),
        "completed_pools": completed, "failed_pools": failed, "pending_pools": len(jobs) - completed - failed,
        "queue_cursor": manifest["queue_cursor"], "error": error,
        "checkpoint_boundary": "complete_level_only", "source_sha256": source_sha256(),
    }
    summary = summarize(manifest, config)
    summary["status"] = status
    out = config["OUTPUT_DIR"]
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "summary.json", summary)
    buffer = io.StringIO(newline="")
    keys = list(summary["rows"][0]) if summary["rows"] else []
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    for row in summary["rows"]:
        writer.writerow({key: json.dumps(ready(value), ensure_ascii=False) if isinstance(value, (list, dict))
                         else ready(value) for key, value in row.items()})
    atomic_text(out / "summary.csv", buffer.getvalue())
    lines = ["# 随机记忆进度", "", f"- 状态：`{status}`；原因：`{reason}`",
             f"- 独立池：完成 {completed}/{len(jobs)}，失败 {failed}，待续 {len(jobs) - completed - failed}",
             f"- 下次队列游标：{manifest['queue_cursor']}", f"- 科学配置 hash：`{manifest['science_hash']}`",
             f"- 主窗口：{config['PRIMARY_WINDOW']}；先同池 K 差，再四池中位数，再数据表统计。", "",
             "每个池有独立阈值、步长和随机流。n 是共享数据表排列的嵌套前缀。",
             "未达到节点为 null，CSV 标量为空白；不作 0 K，不纳入未齐全数据表的主 c。",
             "只解释同架构内 c 与 n；跨架构绝对 K 不作比较。有限 loss 估计不代表渐近或收敛认证。", "",
             "检查点仅提交完整 level，含当前粒子、谱系、累计 log 体积、节点与全部 RNG。",
             "第一次 Ctrl+C/STOP/时间预算在层边界退出；可能超限一个 level 加 I/O。",
             "第二次强制中断回退最后完整 level。每池只保留一个 CPU checkpoint，不保存 GPU 历史权重。",
             "时间切片不计失败重试。异常累计达到上限/无进展访问达到上限/总层数达到上限则标失败，保留已达节点。",
             "接受率、谱系和节点预测变化不是 ESS；池间标准差不能排除全部混合及自适应偏差。", "",
             "| 架构 | n | 主窗口齐全数据表 | c 均值 |", "|---|---:|---:|---:|"]
    for group in summary["by_architecture_n"]:
        mean = group["c_mean_of_dataset_medians"]
        lines.append(f"| {group['architecture']} | {group['n']} | {group['complete_datasets']}/{config['DATASETS']} | "
                     f"{mean:.6g} |" if mean is not None else
                     f"| {group['architecture']} | {group['n']} | 0/{config['DATASETS']} | 未达到 |")
    if error:
        lines += ["", "最近异常：", "```text", error, "```"]
    atomic_text(out / "PROGRESS.md", "\n".join(lines) + "\n")
    # STATUS 最后提交，外部队列读到的状态对应一组已写出的汇总。
    atomic_json(out / "STATUS.json", result)
    return result


def run(config=None):
    """合并顶部 CONFIG 与显式覆盖；返回 STATUS 字典，不启动任何额外进程。"""
    started = time.perf_counter()
    config = resolve_config(config)
    signature, science_hash = science_signature(config)
    manifest = prepare_manifest(config, signature, science_hash)
    call = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_sha256": source_sha256(), "config": ready(config), "reason": "running"}
    manifest["calls"].append(call)
    stop = StopRequest()
    deadline = started + config["TOTAL_SECONDS"]
    reason, fatal_error = "queue_slice_finished", None

    def progress():
        write_reports(manifest, config, "running")

    progress()
    try:
        with stop_signals(stop), execution_environment(config) as (device, environment, environment_identity):
            if manifest.get("environment_identity") not in (None, environment_identity):
                raise CheckpointMismatch("已存执行环境与当前不一致；请在新 OUTPUT_DIR 运行，避免混合设备结果。")
            manifest["environment_identity"], call["environment"] = environment_identity, environment
            atomic_json(config["OUTPUT_DIR"] / "environment.json", environment)
            jobs, visited = manifest["jobs"], set()
            dispatched = 0
            while len(visited) < len(jobs):
                if stop.requested or (config["OUTPUT_DIR"] / "STOP").exists():
                    reason = stop.reason or "STOP_file"
                    break
                if time.perf_counter() >= deadline:
                    reason = "total_time_budget"
                    break
                if config["MAX_REPLICAS_PER_CALL"] is not None and dispatched >= config["MAX_REPLICAS_PER_CALL"]:
                    reason = "replica_dispatch_limit"
                    break
                index = manifest["queue_cursor"] % len(jobs)
                if index in visited:
                    break
                visited.add(index)
                manifest["queue_cursor"] = (index + 1) % len(jobs)
                job = jobs[index]
                if job["status"] in ("completed", "failed"):
                    continue
                # 先提交下一游标，硬退出也不会让同一 badcase 永久占住队首。
                job["visits"] += 1
                atomic_json(config["OUTPUT_DIR"] / "manifest.json", manifest)
                dispatched += 1
                try:
                    outcome = run_slice(job, config, manifest, device, environment_identity, deadline, stop, progress)
                except CheckpointMismatch:
                    raise
                except KeyboardInterrupt:
                    stop.requested, stop.reason = True, "keyboard_interrupt"
                    outcome = {"error": None, "progressed": False}
                except Exception:
                    outcome = {"error": traceback.format_exc(), "progressed": False}
                if outcome["error"]:
                    job["failure_attempts"] += 1
                    job["last_error"] = outcome["error"]
                    job["status"] = "failed" if job["failure_attempts"] >= config["MAX_FAILURE_ATTEMPTS"] else "partial"
                elif job["status"] not in ("completed", "failed") and not stop.requested:
                    if outcome["progressed"]:
                        job["no_progress_visits"] = 0
                    elif not outcome.get("budget_exhausted", False):
                        job["no_progress_visits"] += 1
                    if job["no_progress_visits"] >= config["MAX_NO_PROGRESS_VISITS"]:
                        job.update(status="failed", last_error="连续切片没有完整 level 进展，达到上限")
                progress()
                if job["status"] == "completed" and not config["KEEP_COMPLETED_CHECKPOINTS"]:
                    (config["OUTPUT_DIR"] / job["key"] / "checkpoint.pt").unlink(missing_ok=True)
                if stop.requested:
                    reason = stop.reason
                    break
            if all(job["status"] in ("completed", "failed") for job in jobs):
                reason = "all_pools_terminal"
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception:
        reason, fatal_error = "fatal_error", traceback.format_exc()
        raise
    finally:
        call.update(reason=reason, elapsed_seconds=time.perf_counter() - started)
        result = write_reports(manifest, config, reason, fatal_error)
    return result


def main():
    result = run(CONFIG)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
