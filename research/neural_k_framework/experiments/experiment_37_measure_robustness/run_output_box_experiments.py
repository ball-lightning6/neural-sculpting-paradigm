"""输出联合分布的平移、幅度与对齐实验。

本脚本专门检验三个容易混在一起的效应：

1. 输出目标整体平移后，中心化体积曲线是否保持不变；
2. 只去掉最后一层output bias时，隐藏层bias能否补偿；
3. 输出二值目标的间距变小时，函数区分度和收缩系数是否消失；
4. 输出偏移或幅度变大后，变化来自真实压缩、测度偏移，还是激活饱和。

脚本不使用 argparse。修改本文件顶部的 ``RUN_PLAN``、``RULE_IDS`` 和预算即可运行。
它复用已经核验过的 3bit 有限宽网络 SMC 内核，但所有新的 MSE 目标、输出测度偏移、
中心化统计和结果目录都在本文件内定义，不会修改旧实验。

建议顺序：

    RUN_PLAN = "translation"  # 先跑同步平移和两个对照
    RUN_PLAN = "bias"         # 只比较有/无最后一层output bias
    RUN_PLAN = "amplitude"    # 再扫目标间距
    RUN_PLAN = "endpoint"     # 比较对称扩张与单侧扩张
    RUN_PLAN = "all"           # 最后合并全部条件

本地 Jupyter/AutoDL 直接运行本文件即可。默认使用第二张 GPU；只有一张 GPU 时自动使用
cuda:0。实验输出在本目录的 ``results`` 下。
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ========================= 用户可直接修改的配置 =========================

RUN_PLAN = "translation"  # "translation"、"bias"、"amplitude"、"endpoint" 或 "all"

# 0x00=常数，0x0F=单输入copy，0x17=三输入多数，0x69=parity3。
# 先用4个代表性规则；若只做快速试跑，可改成 (0x00, 0x0F)。
RULE_IDS = (0x00, 0x0F, 0x17, 0x69)

# 四个副本足以先看方向；若要发表级别的尾部数值，再升到8。
REPLICAS = 4
PARTICLES_PER_REPLICA = 2048
ADAPT_SWEEPS = 4
MUTATION_SWEEPS = 12
MAX_LEVELS = 1500
SECONDS_PER_CASE = 180

# MSE使用绝对输出单位。高阈值只用于让SMC自动进入目标区间，比较时看共同节点。
MSE_THRESHOLDS = (10.0, 2.0, 0.5, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 1e-5)

# 端点实验的目标幅度约为9，必须把高loss节点推到足够高，否则SMC会从错误的截面开始。
ENDPOINT_MSE_THRESHOLDS = (200.0, 100.0, 50.0, 20.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)

# 等幅目标的半间距。(-a,+a) 的两类输出间距为2a。
AMPLITUDES = (0.01, 0.1, 0.5, 1.0, 3.0)

# 同步平移实验：目标中心和输出参考测度一起平移。
BASE_HALF_WIDTH = 1.0
SHIFTED_CENTER = 10.0

# 使用现有的小网络和Gaussian神经坐标。默认自动选择本机空闲GPU。
HIDDEN_WIDTHS = (16,)
ACTIVATION = "tanh"
DEVICE = (
    "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else "cuda:0" if torch.cuda.is_available() else "cpu"
)

# 输出测度偏移为物理输出单位，不是参数坐标单位。
# 例如measure_center=10表示输出bias的先验中心使网络输出整体平移约10。
RESULT_ROOT = Path(__file__).resolve().parent
# translation保留简短的历史目录名；其它运行计划自动分目录，避免互相覆盖。
RESULT_DIR = RESULT_ROOT / ("results" if RUN_PLAN == "translation" else f"results_{RUN_PLAN}")
RESUME = True
OVERWRITE_RESULT_DIR = False


ROOT = Path(__file__).resolve().parent
ENGINE_SOURCE = ROOT / "experiment_3bit_cross_architecture_complexity_order.py"

# 以下变量由每条曲线运行前更新；SMC内核本身仍在中心化参数坐标中工作。
ACTIVE_TARGET: torch.Tensor | None = None
ACTIVE_TARGET_CENTER = 0.0
ACTIVE_MEASURE_CENTER = 0.0
ACTIVE_BIAS_MODE = "with_output_bias"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_engine():
    """动态加载旧内核，避免把原始大文件复制进本实验。"""
    name = "output_box_engine_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ENGINE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载SMC内核：{ENGINE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def configure_engine(engine) -> None:
    """固定为3->16->1 tanh，并把原内核改成适合本轮的预算。"""
    engine.activate_architecture(
        engine.ArchitectureSpec(
            "tanh_w16_d1_output_box",
            HIDDEN_WIDTHS,
            ACTIVATION,
        ),
        0,
    )
    cfg = engine.Config
    cfg.INPUT_BITS = 3
    cfg.POINT_COUNT = 8
    cfg.FUNCTION_COUNT = 256
    cfg.REPLICAS = REPLICAS
    cfg.PARTICLES_PER_REPLICA = PARTICLES_PER_REPLICA
    cfg.ADAPT_SWEEPS = ADAPT_SWEEPS
    cfg.MUTATION_SWEEPS = MUTATION_SWEEPS
    cfg.MAX_LEVELS_PER_TARGET = MAX_LEVELS
    cfg.TARGET_THRESHOLDS = tuple(float(x) for x in thresholds_for_run())
    cfg.LOSS_TOLERANCE = 1e-7
    cfg.EVAL_MICRO_BATCH = 32768
    cfg.DEVICE = DEVICE
    cfg.ALLOW_TF32 = False
    cfg.PRIOR_SEED = 2026091901
    cfg.RESAMPLE_SEED = 2026091902
    cfg.MUTATION_SEED = 2026091903
    cfg.RESULT_DIR = RESULT_DIR
    cfg.PROTOCOL_VERSION = "output_box_translation_v1_mse"
    cfg.RESUME = RESUME
    cfg.OVERWRITE_RESULT_DIR = OVERWRITE_RESULT_DIR
    cfg.SAVE_PARTICLE_SNAPSHOTS = False
    cfg.CHECKPOINT_EVERY_LEVELS = 25
    cfg.LOG_EVERY_LEVELS = 25

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def target_vector(
    engine,
    function_id: int,
    center: float,
    half_width: float,
    device,
    low: float | None = None,
    high: float | None = None,
):
    bits = engine.function_outputs(function_id).astype(np.float32)
    if low is None or high is None:
        values = center + half_width * (2.0 * bits - 1.0)
    else:
        values = float(low) + (float(high) - float(low)) * bits
    return torch.tensor(values, dtype=torch.float32, device=device)


def thresholds_for_run() -> tuple[float, ...]:
    if RUN_PLAN == "endpoint":
        return tuple(float(x) for x in ENDPOINT_MSE_THRESHOLDS)
    if RUN_PLAN == "all":
        return tuple(sorted(
            {float(x) for x in (*ENDPOINT_MSE_THRESHOLDS, *MSE_THRESHOLDS)},
            reverse=True,
        ))
    return tuple(float(x) for x in MSE_THRESHOLDS)


def install_output_box_loss(engine) -> None:
    """把原BCE评估替换为原始输出上的MSE，并支持移除最后一层bias。"""

    original_parameter_blocks = engine.parameter_blocks

    def dynamic_parameter_blocks():
        if ACTIVE_BIAS_MODE == "with_output_bias":
            return original_parameter_blocks()
        if ACTIVE_BIAS_MODE != "without_output_bias":
            raise ValueError(f"未知bias模式：{ACTIVE_BIAS_MODE}")
        cursor = 0
        blocks = []
        fan_in = engine.Config.INPUT_BITS
        for layer_index, width in enumerate(engine.ACTIVE_ARCHITECTURE.hidden_widths):
            layer_size = int(width) * int(fan_in) + int(width)
            blocks.append(engine.ParameterBlock(
                f"hidden_layer_{layer_index + 1}", cursor, cursor + layer_size
            ))
            cursor += layer_size
            fan_in = int(width)
        output = engine.ParameterBlock("output_weight", cursor, cursor + fan_in)
        cursor += fan_in
        all_parameters = engine.ParameterBlock("all_parameters", 0, cursor)
        return [*blocks, output, all_parameters], cursor

    engine.parameter_blocks = dynamic_parameter_blocks

    @torch.no_grad()
    def forward_with_offset(normalized: torch.Tensor, inputs: torch.Tensor):
        count = normalized.shape[0]
        cursor = 0
        hidden = inputs[None].expand(count, -1, -1)
        fan_in = engine.Config.INPUT_BITS
        for width in engine.ACTIVE_ARCHITECTURE.hidden_widths:
            weight_size = int(width) * int(fan_in)
            weight = normalized[:, cursor:cursor + weight_size].reshape(
                count, int(width), int(fan_in)
            ) * (1.0 / math.sqrt(fan_in))
            cursor += weight_size
            bias = normalized[:, cursor:cursor + int(width)] * (
                1.0 / math.sqrt(fan_in)
            )
            cursor += int(width)
            hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
            if engine.ACTIVE_ARCHITECTURE.activation == "tanh":
                hidden = torch.tanh(hidden)
            elif engine.ACTIVE_ARCHITECTURE.activation == "relu":
                hidden = torch.relu(hidden)
            else:
                raise RuntimeError("未知激活函数。")
            fan_in = int(width)
        output_weight = normalized[:, cursor:cursor + fan_in].reshape(
            count, 1, fan_in
        ) * (1.0 / math.sqrt(fan_in))
        cursor += fan_in
        if ACTIVE_BIAS_MODE == "with_output_bias":
            output_bias = normalized[:, cursor:cursor + 1] * (
                1.0 / math.sqrt(fan_in)
            )
            cursor += 1
        else:
            output_bias = torch.zeros(
                count, 1, dtype=normalized.dtype, device=normalized.device
            )
        if cursor != normalized.shape[-1]:
            raise RuntimeError(
                f"参数解包错误：bias模式={ACTIVE_BIAS_MODE}，使用{cursor}，"
                f"实际{normalized.shape[-1]}。"
            )
        return (
            torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
            + output_bias
            + float(ACTIVE_MEASURE_CENTER)
        )

    @torch.no_grad()
    def evaluate_mse(
        particles: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if ACTIVE_TARGET is None:
            raise RuntimeError("ACTIVE_TARGET尚未设置。")
        flat = particles.reshape(-1, particles.shape[-1])
        pieces = []
        for start in range(0, len(flat), engine.Config.EVAL_MICRO_BATCH):
            outputs = forward_with_offset(
                flat[start:start + engine.Config.EVAL_MICRO_BATCH], inputs
            )
            pieces.append((outputs - ACTIVE_TARGET[None]).square().mean(dim=1))
        return torch.cat(pieces).reshape(particles.shape[:-1])

    engine.forward_logits = forward_with_offset
    engine.evaluate_losses = evaluate_mse


def shifted_function_ids(engine, particles: torch.Tensor, inputs: torch.Tensor):
    """按当前目标中心而不是按绝对0切分输出。"""
    flat = particles.reshape(-1, particles.shape[-1])
    powers = torch.bitwise_left_shift(
        torch.ones(engine.Config.POINT_COUNT, dtype=torch.int64, device=particles.device),
        torch.arange(engine.Config.POINT_COUNT, dtype=torch.int64, device=particles.device),
    )
    pieces = []
    for start in range(0, len(flat), engine.Config.EVAL_MICRO_BATCH):
        outputs = engine.forward_logits(
            flat[start:start + engine.Config.EVAL_MICRO_BATCH], inputs
        )
        bits = (outputs >= float(ACTIVE_TARGET_CENTER)).to(torch.int64)
        pieces.append(((bits * powers[None]).sum(dim=1)).cpu().numpy().astype(np.uint16))
    return np.concatenate(pieces).reshape(particles.shape[:-1])


def initialize_state(engine, device, inputs, parameter_count, generator):
    particles = torch.empty(
        engine.Config.REPLICAS,
        engine.Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    ).normal_(mean=0.0, std=1.0, generator=generator)
    losses = engine.evaluate_losses(particles, inputs, inputs)
    total = engine.Config.REPLICAS * engine.Config.PARTICLES_PER_REPLICA
    lineages = torch.arange(total, device=device, dtype=torch.int64).reshape(
        engine.Config.REPLICAS, engine.Config.PARTICLES_PER_REPLICA
    )
    return engine.SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume_fraction=torch.zeros(
            engine.Config.REPLICAS, device=device, dtype=torch.float64
        ),
        proposal_scales=list(engine.Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=float("inf"),
        threshold_index=0,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
        top_rows=[],
    )


def state_payload(state, generators) -> dict[str, Any]:
    return {
        "particles": state.particles.detach().cpu(),
        "losses": state.losses.detach().cpu(),
        "lineages": state.lineages.detach().cpu(),
        "log_volume_fraction": state.log_volume_fraction.detach().cpu(),
        "proposal_scales": list(state.proposal_scales),
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "threshold_rows": state.threshold_rows,
        "replica_rows": state.replica_rows,
        "top_rows": state.top_rows,
        "generator_states": {key: value.get_state() for key, value in generators.items()},
    }


def restore_state(engine, payload, device, generators):
    for key, value in payload["generator_states"].items():
        generators[key].set_state(value)
    return engine.SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume_fraction=payload["log_volume_fraction"].to(device),
        proposal_scales=list(payload["proposal_scales"]),
        current_threshold=float(payload["current_threshold"]),
        threshold_index=int(payload["threshold_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        threshold_rows=list(payload["threshold_rows"]),
        replica_rows=list(payload["replica_rows"]),
        top_rows=list(payload["top_rows"]),
    )


def save_checkpoint(path: Path, state, generators) -> None:
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(state_payload(state, generators), temp)
    # Windows下安全软件或文件索引器可能短暂占用旧checkpoint，允许短暂重试。
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(0.1 * (2 ** min(attempt, 4)), 1.0))


def evaluate_mse64(engine, particles: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if ACTIVE_TARGET is None:
        raise RuntimeError("ACTIVE_TARGET尚未设置。")
    outputs = engine.forward_logits(particles.to(torch.float64), inputs.to(torch.float64))
    return (outputs - ACTIVE_TARGET.to(torch.float64)[None]).square().mean(dim=1)


def summarize_threshold(engine, state, inputs, function_id: int, threshold: float) -> dict[str, Any]:
    ids = shifted_function_ids(engine, state.particles, inputs)
    counts = np.bincount(ids.reshape(-1), minlength=engine.Config.FUNCTION_COUNT)
    replica_logs = state.log_volume_fraction.detach().cpu().numpy()
    outputs = engine.forward_logits(state.particles.reshape(-1, state.particles.shape[-1]), inputs)
    per_network_mean = outputs.mean(dim=1)
    centered = outputs - per_network_mean[:, None]
    centered_rms = float(centered.square().mean().sqrt().item())
    mean_output = float(outputs.mean().item())
    mean_target = float(ACTIVE_TARGET.mean().item())
    target_probability = float(counts[function_id] / max(counts.sum(), 1))
    row = {
        "threshold": float(threshold),
        "level": int(state.level),
        "function_id": int(function_id),
        "function_hex": f"0x{function_id:02X}",
        "target_center": float(ACTIVE_TARGET_CENTER),
        "measure_center": float(ACTIVE_MEASURE_CENTER),
        "target_mean": mean_target,
        "mean_output": mean_output,
        "centered_output_rms": centered_rms,
        "target_probability": target_probability,
        "function_entropy_bits": engine.distribution_entropy(counts),
        "function_support": int(np.count_nonzero(counts)),
        "neural_k_bits_median": float(-np.median(replica_logs) / math.log(2.0)),
        "neural_k_bits_min": float(-np.max(replica_logs) / math.log(2.0)),
        "neural_k_bits_max": float(-np.min(replica_logs) / math.log(2.0)),
        "k_replica_std_bits": float(np.std(-replica_logs / math.log(2.0))),
    }
    return row


def case_list():
    cases = []
    if RUN_PLAN in {"translation", "all"}:
        cases.extend([
            {"name": "center0_base", "center": 0.0, "measure": 0.0, "half_width": BASE_HALF_WIDTH,
             "kind": "translation", "bias_mode": "with_output_bias"},
            {"name": "center10_sync", "center": SHIFTED_CENTER, "measure": SHIFTED_CENTER,
             "half_width": BASE_HALF_WIDTH, "kind": "translation", "bias_mode": "with_output_bias"},
            {"name": "center10_target_only", "center": SHIFTED_CENTER, "measure": 0.0,
             "half_width": BASE_HALF_WIDTH, "kind": "translation_control", "bias_mode": "with_output_bias"},
        ])
    if RUN_PLAN in {"bias", "all"}:
        cases.extend([
            {"name": "output_bias_on", "center": 0.0, "measure": 0.0,
             "half_width": BASE_HALF_WIDTH, "kind": "bias_control",
             "bias_mode": "with_output_bias"},
            {"name": "output_bias_off", "center": 0.0, "measure": 0.0,
             "half_width": BASE_HALF_WIDTH, "kind": "bias_control",
             "bias_mode": "without_output_bias"},
        ])
    if RUN_PLAN in {"amplitude", "all"}:
        for half_width in AMPLITUDES:
            cases.append({
                "name": f"center0_amp_{half_width:g}",
                "center": 0.0,
                "measure": 0.0,
                "half_width": float(half_width),
                "kind": "amplitude",
                "bias_mode": "with_output_bias",
            })
    if RUN_PLAN in {"endpoint", "all"}:
        # 三个条件共享同一个未平移的参考测度，只有目标盒的两端发生变化。
        cases.extend([
            {"name": "endpoint_base_m9_p9", "center": 0.0, "measure": 0.0,
             "half_width": 9.0, "low": -9.0, "high": 9.0,
             "kind": "endpoint_base", "bias_mode": "with_output_bias"},
            {"name": "endpoint_symmetric_m10_p10", "center": 0.0, "measure": 0.0,
             "half_width": 10.0, "low": -10.0, "high": 10.0,
             "kind": "endpoint_symmetric", "bias_mode": "with_output_bias"},
            {"name": "endpoint_high_only_m9_p11", "center": 1.0, "measure": 0.0,
             "half_width": 10.0, "low": -9.0, "high": 11.0,
             "kind": "endpoint_high_only", "bias_mode": "with_output_bias"},
        ])
    if RUN_PLAN not in {"translation", "bias", "amplitude", "endpoint", "all"}:
        raise ValueError("RUN_PLAN必须是translation、bias、amplitude、endpoint或all。")
    return cases


def run_one(engine, case: dict[str, Any], function_id: int) -> dict[str, Any]:
    global ACTIVE_TARGET, ACTIVE_TARGET_CENTER, ACTIVE_MEASURE_CENTER, ACTIVE_BIAS_MODE
    ACTIVE_TARGET_CENTER = float(case["center"])
    ACTIVE_MEASURE_CENTER = float(case["measure"])
    ACTIVE_BIAS_MODE = str(case["bias_mode"])
    device = torch.device(DEVICE)
    inputs = torch.tensor(engine.truth_table_inputs(), dtype=torch.float32, device=device)
    ACTIVE_TARGET = target_vector(
        engine,
        function_id,
        ACTIVE_TARGET_CENTER,
        float(case["half_width"]),
        device,
        low=case.get("low"),
        high=case.get("high"),
    )

    case_name = f"{case['name']}_{ACTIVE_BIAS_MODE}_f{function_id:02X}"
    out = RESULT_DIR / case_name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    if meta_path.exists() and RESUME:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            return saved

    blocks, parameter_count = engine.parameter_blocks()
    generators = engine.make_generators(device, int(function_id))
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists() and RESUME:
        state = restore_state(engine, torch.load(checkpoint, map_location="cpu"), device, generators)
    else:
        state = initialize_state(engine, device, inputs, parameter_count, generators["prior"])

    begin = time.perf_counter()
    last_log = begin
    status = "running"
    thresholds = thresholds_for_run()
    while state.threshold_index < len(thresholds) and state.level < MAX_LEVELS:
        if time.perf_counter() - begin >= SECONDS_PER_CASE:
            status = "time_budget"
            break
        target_threshold = float(thresholds[state.threshold_index])
        quantiles = torch.quantile(state.losses, engine.Config.SURVIVAL_QUANTILE, dim=1)
        adaptive = float(quantiles.max().item())
        threshold = max(target_threshold, adaptive)
        if math.isfinite(state.current_threshold):
            threshold = min(threshold, state.current_threshold)
        if math.isfinite(state.current_threshold) and threshold >= state.current_threshold - engine.Config.MIN_LEVEL_DECREMENT:
            status = "stalled_threshold"
            break
        survival = engine.resample_within_replicas(
            state, threshold, inputs, inputs, generators["resample"]
        )
        acceptance = engine.rejuvenate(
            state, blocks, threshold, inputs, inputs, generators["mutation"]
        )
        state.level += 1
        state.current_threshold = threshold
        row = {
            "level": int(state.level),
            "threshold": float(threshold),
            "survival_min": float(np.min(survival)),
            "survival_max": float(np.max(survival)),
            "neural_k_bits_median": float(
                -torch.median(state.log_volume_fraction).item() / math.log(2.0)
            ),
            **acceptance,
        }
        state.level_rows.append(row)
        reached = threshold <= target_threshold + 1e-12
        if reached:
            threshold_row = summarize_threshold(engine, state, inputs, function_id, threshold)
            state.threshold_rows.append(threshold_row)
            # 用float64重算当前粒子的一小部分，避免MSE边界由float32误差主导。
            sample = state.particles.reshape(-1, parameter_count)[: min(256, state.particles.numel() // parameter_count)]
            loss64 = evaluate_mse64(engine, sample, inputs)
            threshold_row["float64_max_excess"] = float(
                (loss64 - threshold - engine.Config.LOSS_TOLERANCE).max().item()
            )
            state.threshold_index += 1
        if reached or state.level % 25 == 0:
            save_checkpoint(checkpoint, state, generators)
        now = time.perf_counter()
        if reached or now - last_log >= 30.0:
            print(
                f"{case_name} level={state.level} MSE<={threshold:.6g} "
                f"K={row['neural_k_bits_median']:.2f} "
                f"elapsed={now - begin:.1f}s",
                flush=True,
            )
            last_log = now
    if status == "running":
        status = "completed" if state.threshold_index == len(thresholds) else "max_levels"
    save_checkpoint(checkpoint, state, generators)
    metadata = {
        "status": status,
        "case": case,
        "function_id": int(function_id),
        "function_hex": f"0x{function_id:02X}",
        "parameter_count": int(parameter_count),
        "bias_mode": ACTIVE_BIAS_MODE,
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(torch.device(DEVICE)) if device.type == "cuda" else "CPU",
        "replicas": REPLICAS,
        "particles_per_replica": PARTICLES_PER_REPLICA,
        "thresholds_completed": int(state.threshold_index),
        "levels": int(state.level),
        "elapsed_seconds": float(time.perf_counter() - begin),
        "engine_sha256": sha256(ENGINE_SOURCE),
        "script_sha256": sha256(Path(__file__).resolve()),
        "threshold_rows": state.threshold_rows,
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_json(meta_path, metadata)
    return metadata


def summarize_alignment(records: list[dict[str, Any]]) -> dict[str, Any]:
    """对相同条件下的规则K排序做简单的Spearman对齐统计。"""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record["case"]["name"])
        by_case.setdefault(key, []).append(record)
    baseline = None
    rows = []
    for case_name, items in by_case.items():
        # 只在共同的最低阈值上比较，避免把未完成曲线当作结果。
        common = set.intersection(*[
            {float(row["threshold"]) for row in item["threshold_rows"]}
            for item in items
        ]) if items else set()
        if not common:
            continue
        threshold = min(common)
        values = sorted(
            (item["function_id"], next(row for row in item["threshold_rows"] if float(row["threshold"]) == threshold)["neural_k_bits_median"])
            for item in items
        )
        if baseline is None:
            baseline = values
        rank_a = {fid: rank for rank, (fid, _) in enumerate(sorted(values, key=lambda x: x[1]))}
        rank_b = {fid: rank for rank, (fid, _) in enumerate(sorted(baseline, key=lambda x: x[1]))}
        common_ids = sorted(set(rank_a) & set(rank_b))
        if len(common_ids) >= 2:
            a = np.asarray([rank_a[fid] for fid in common_ids], dtype=float)
            b = np.asarray([rank_b[fid] for fid in common_ids], dtype=float)
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
        else:
            corr = None
        rows.append({
            "case": case_name,
            "common_threshold": threshold,
            "spearman_like_rank_correlation_to_first_case": corr,
            "k_bits": {f"0x{fid:02X}": float(k) for fid, k in values},
        })
    return {"rows": rows, "baseline_case": next(iter(by_case), None)}


def summarize_curves(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """估计每条曲线的深尾斜率，不把不完整曲线外推到未测阈值。"""
    rows = []
    for record in records:
        curve = record.get("threshold_rows", [])
        points = [
            (float(row["threshold"]), float(row["neural_k_bits_median"]))
            for row in curve
            if float(row["threshold"]) > 0 and math.isfinite(float(row["neural_k_bits_median"]))
        ]
        if len(points) >= 2:
            x = np.asarray([-math.log10(t) for t, _ in points], dtype=float)
            y = np.asarray([k for _, k in points], dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            deep = points[-min(4, len(points)):]
            dx = np.asarray([-math.log10(t) for t, _ in deep], dtype=float)
            dy = np.asarray([k for _, k in deep], dtype=float)
            deep_slope = float(np.polyfit(dx, dy, 1)[0]) if len(deep) >= 2 else None
        else:
            slope = intercept = deep_slope = None
        rows.append({
            "case": record["case"],
            "function_id": record["function_id"],
            "function_hex": record["function_hex"],
            "bias_mode": record.get("bias_mode", record["case"].get("bias_mode")),
            "status": record["status"],
            "threshold_count": len(points),
            "full_curve_slope_bits_per_decade": None if slope is None else float(slope),
            "full_curve_intercept_bits": None if intercept is None else float(intercept),
            "deep_tail_slope_bits_per_decade": deep_slope,
            "measured_thresholds": [t for t, _ in points],
        })
    return rows


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = load_engine()
    configure_engine(engine)
    install_output_box_loss(engine)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    protocol = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "engine": str(ENGINE_SOURCE),
        "engine_sha256": sha256(ENGINE_SOURCE),
        "run_plan": RUN_PLAN,
        "rules": [f"0x{x:02X}" for x in RULE_IDS],
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(torch.device(DEVICE)) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "loss": "mean squared error on raw scalar outputs",
        "mse_thresholds": list(thresholds_for_run()),
        "replicas": REPLICAS,
        "particles_per_replica": PARTICLES_PER_REPLICA,
        "case_list": case_list(),
    }
    atomic_json(RESULT_DIR / "protocol.json", protocol)
    atomic_text(
        RESULT_DIR / "PROGRESS.md",
        "# 输出联合分布实验进度\n\n"
        f"状态：开始运行，计划 `{RUN_PLAN}`。\n\n"
        f"设备：`{protocol['gpu_name']}`，规则：{protocol['rules']}。\n",
    )

    records = []
    for case in case_list():
        for function_id in RULE_IDS:
            records.append(run_one(engine, case, int(function_id)))
            atomic_json(RESULT_DIR / "records.json", records)
    atomic_json(RESULT_DIR / "alignment.json", summarize_alignment(records))
    atomic_json(RESULT_DIR / "curve_summary.json", summarize_curves(records))
    atomic_text(
        RESULT_DIR / "PROGRESS.md",
        "# 输出联合分布实验进度\n\n"
        f"状态：完成。\n\n设备：`{protocol['gpu_name']}`。\n"
        f"已完成曲线：{len(records)} 条。\n",
    )
    print(json.dumps({"completed_cases": len(records), "result_dir": str(RESULT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
