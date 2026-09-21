"""固定范数球内的欧式体积端点实验。

这是对 ``run_output_box_experiments.py`` 中Gaussian版本的严格测度修正。
参数坐标theta在固定球 ``||theta||_2 <= R`` 内服从均匀Lebesgue测度，
不使用Gaussian密度、不使用pCN核。三个目标盒固定为：

* 端点网格：低端 ``-11,-10,-9``，高端 ``9,10,11``

所有条件使用同一个R、同一个参数坐标、同一个loss节点。脚本不使用argparse，
直接修改顶部配置即可。默认R=72、4个副本、每副本2048粒子、GPU自动选择。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ========================= 配置 =========================

RULE_IDS = (0x00, 0x0F, 0x17, 0x69)
RADIUS = 72.0
REPLICAS = 4
PARTICLES_PER_REPLICA = 2048
SURVIVAL_QUANTILE = 0.5
MUTATION_SWEEPS = 12
MAX_LEVELS = 5000
SECONDS_PER_CASE = 300
MSE_THRESHOLDS = (200.0, 100.0, 50.0, 20.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
LOSS_TOLERANCE = 1e-6
DEVICE = (
    "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else "cuda:0" if torch.cuda.is_available() else "cpu"
)
DTYPE = torch.float32
RESULT_DIR = Path(__file__).resolve().parent / "results_endpoint_uniform_ball"
RESUME = True
BASE_SEED = 2026091900

ROOT = Path(__file__).resolve().parent
ENGINE_SOURCE = ROOT / "run_output_box_experiments.py"
INPUTS_CPU = torch.tensor(
    [[(index >> 2) & 1, (index >> 1) & 1, index & 1] for index in range(8)],
    dtype=DTYPE,
)

# 3->16->1 tanh的标准化参数坐标分块：隐藏层、输出层、全部参数。
PARAMETER_DIM = 16 * 3 + 16 + 16 + 1
BLOCKS = ((0, 64), (64, 81), (0, 81))
ENDPOINT_LOW_VALUES = (-11.0, -10.0, -9.0)
ENDPOINT_HIGH_VALUES = (9.0, 10.0, 11.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_retry(source: Path, target: Path) -> None:
    for attempt in range(40):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(0.1 * (2 ** min(attempt, 4)), 1.0))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_retry(temp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(value, temp)
    replace_retry(temp, path)


def target_values(function_id: int, low: float, high: float, device) -> torch.Tensor:
    bits = torch.tensor(
        [(function_id >> index) & 1 for index in range(8)],
        dtype=DTYPE,
        device=device,
    )
    return low + (high - low) * bits


@torch.no_grad()
def forward(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    count = theta.shape[0]
    cursor = 0
    hidden = inputs[None].expand(count, -1, -1)
    fan_in = 3
    width = 16
    weight = theta[:, cursor:cursor + width * fan_in].reshape(count, width, fan_in)
    weight = weight * (1.0 / math.sqrt(fan_in))
    cursor += width * fan_in
    bias = theta[:, cursor:cursor + width] * (1.0 / math.sqrt(fan_in))
    cursor += width
    hidden = torch.tanh(torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None])
    output_weight = theta[:, cursor:cursor + width].reshape(count, 1, width)
    output_weight = output_weight * (1.0 / math.sqrt(width))
    cursor += width
    output_bias = theta[:, cursor:cursor + 1] * (1.0 / math.sqrt(width))
    cursor += 1
    if cursor != theta.shape[-1]:
        raise RuntimeError(f"参数维度错误：使用{cursor}，实际{theta.shape[-1]}。")
    return torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1) + output_bias


@torch.no_grad()
def score(theta: torch.Tensor, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    outputs = forward(theta.reshape(-1, PARAMETER_DIM), inputs)
    return (outputs - target[None]).square().mean(dim=1).reshape(theta.shape[:-1])


def sample_uniform_ball(shape, radius: float, generator: torch.Generator, device) -> torch.Tensor:
    direction = torch.randn(shape, device=device, dtype=DTYPE, generator=generator)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-30)
    u = torch.rand(shape[:-1] + (1,), device=device, dtype=DTYPE, generator=generator)
    return direction * (radius * u.pow(1.0 / shape[-1]))


def rotate_block(theta: torch.Tensor, start: int, stop: int, angle: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
    current = theta[..., start:stop]
    norm = torch.linalg.vector_norm(current, dim=-1, keepdim=True)
    unit = current / norm.clamp_min(1e-30)
    noise = torch.randn(current.shape, device=theta.device, dtype=theta.dtype, generator=generator)
    tangent = noise - (noise * unit).sum(-1, keepdim=True) * unit
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-30)
    alpha = angle[:, None, None]
    moved = norm * (alpha.cos() * unit + alpha.sin() * tangent)
    proposed = theta.clone()
    proposed[..., start:stop] = moved
    return proposed


@torch.no_grad()
def mutate(
    theta: torch.Tensor,
    values: torch.Tensor,
    thresholds: torch.Tensor,
    target: torch.Tensor,
    inputs: torch.Tensor,
    scales: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rates = torch.zeros_like(scales)
    radial_rate = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
    for _ in range(MUTATION_SWEEPS):
        for block_index, (start, stop) in enumerate(BLOCKS):
            proposed = rotate_block(theta, start, stop, scales[:, block_index], generator)
            proposed_values = score(proposed, inputs, target)
            accepted = proposed_values <= thresholds[:, None] + LOSS_TOLERANCE
            theta = torch.where(accepted[..., None], proposed, theta)
            values = torch.where(accepted, proposed_values, values)
            rates[:, block_index] += accepted.to(theta.dtype).mean(1)

        # 球内均匀测度的径向更新；半径分布不是均匀分布，而是r^(d-1)。
        old_radius = torch.linalg.vector_norm(theta, dim=-1, keepdim=True)
        u = torch.rand(old_radius.shape, device=theta.device, dtype=theta.dtype, generator=generator)
        new_radius = RADIUS * u.pow(1.0 / PARAMETER_DIM)
        proposed = theta * (new_radius / old_radius.clamp_min(1e-30))
        proposed_values = score(proposed, inputs, target)
        accepted = proposed_values <= thresholds[:, None] + LOSS_TOLERANCE
        theta = torch.where(accepted[..., None], proposed, theta)
        values = torch.where(accepted, proposed_values, values)
        radial_rate += accepted.to(theta.dtype).mean(1)

    rates /= MUTATION_SWEEPS
    radial_rate /= MUTATION_SWEEPS
    scales = (scales * (0.30 * (rates - 0.30)).exp()).clamp(1e-5, 0.85)
    return theta, values, rates, radial_rate


def logmeanexp(values: torch.Tensor) -> float:
    maximum = values.max()
    return float(maximum + torch.log(torch.exp(values - maximum).mean()).item())


def state_save(path: Path, state: dict[str, Any], generators: dict[str, torch.Generator]) -> None:
    payload = {}
    for key, value in state.items():
        payload[key] = value.detach().cpu() if torch.is_tensor(value) else value
    payload["generator_states"] = {key: value.get_state() for key, value in generators.items()}
    atomic_torch(path, payload)


def state_load(path: Path, device, generators: dict[str, torch.Generator]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in payload.pop("generator_states").items():
        generators[key].set_state(value)
    for key in ("theta", "values", "log_mass", "scales"):
        payload[key] = payload[key].to(device)
    return payload


def function_ids(theta: torch.Tensor, inputs: torch.Tensor, center: float) -> np.ndarray:
    outputs = forward(theta.reshape(-1, PARAMETER_DIM), inputs)
    powers = torch.tensor([1 << index for index in range(8)], device=theta.device, dtype=torch.int64)
    ids = ((outputs >= center).to(torch.int64) * powers[None]).sum(dim=1)
    return ids.cpu().numpy().reshape(theta.shape[:-1])


def record_node(state: dict[str, Any], target_function: int, target: torch.Tensor,
                inputs: torch.Tensor, case: dict[str, Any], threshold: float) -> dict[str, Any]:
    log_mass = state["log_mass"].detach().cpu()
    k = -logmeanexp(log_mass) / math.log(2.0)
    ids = function_ids(state["theta"], inputs, case["center"])
    counts = np.bincount(ids.reshape(-1), minlength=256)
    outputs = forward(state["theta"].reshape(-1, PARAMETER_DIM), inputs)
    per_mean = outputs.mean(1)
    centered_rms = float((outputs - per_mean[:, None]).square().mean().sqrt().item())
    return {
        "threshold": float(threshold),
        "function_id": int(target_function),
        "function_hex": f"0x{target_function:02X}",
        "low": float(case["low"]),
        "high": float(case["high"]),
        "measure": "uniform_ball",
        "radius": RADIUS,
        "neural_k_bits": k,
        "k_replica_mean_bits": float((-log_mass / math.log(2.0)).mean().item()),
        "k_replica_sd_bits": float((-log_mass / math.log(2.0)).std(unbiased=True).item()),
        "target_probability": float(counts[target_function] / max(counts.sum(), 1)),
        "function_entropy_bits": float(-sum(
            (count / counts.sum()) * math.log2(count / counts.sum())
            for count in counts if count
        )),
        "function_support": int(np.count_nonzero(counts)),
        "mean_output": float(outputs.mean().item()),
        "centered_output_rms": centered_rms,
    }


def cases() -> list[dict[str, Any]]:
    result = []
    for low in ENDPOINT_LOW_VALUES:
        for high in ENDPOINT_HIGH_VALUES:
            low_text = str(int(abs(low))).replace("-", "m")
            high_text = str(int(high)).replace("-", "m")
            result.append({
                "name": f"grid_m{low_text}_p{high_text}",
                "low": low,
                "high": high,
                "center": (low + high) / 2.0,
            })
    return result


def run_case(case: dict[str, Any], function_id: int, case_index: int) -> dict[str, Any]:
    device = torch.device(DEVICE)
    inputs = INPUTS_CPU.to(device)
    target = target_values(function_id, case["low"], case["high"], device)
    name = f"{case['name']}_f{function_id:02X}"
    out = RESULT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.json"
    if RESUME and metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            return saved

    seed = BASE_SEED + case_index * 1009 + function_id
    generators = {
        "initial": torch.Generator(device=device).manual_seed(seed),
        "resample": torch.Generator(device=device).manual_seed(seed + 100003),
        "mutation": torch.Generator(device=device).manual_seed(seed + 200003),
    }
    checkpoint = out / "checkpoint.pt"
    if RESUME and checkpoint.exists():
        state = state_load(checkpoint, device, generators)
    else:
        theta = sample_uniform_ball(
            (REPLICAS, PARTICLES_PER_REPLICA, PARAMETER_DIM),
            RADIUS,
            generators["initial"],
            device,
        )
        state = {
            "theta": theta,
            "values": score(theta, inputs, target),
            "log_mass": torch.zeros(REPLICAS, device=device, dtype=torch.float64),
            "scales": torch.tensor(
                [[0.12, 0.20, 0.06]] * REPLICAS,
                device=device,
                dtype=DTYPE,
            ),
            "threshold": torch.full((REPLICAS,), float("inf"), device=device, dtype=DTYPE),
            "level": 0,
            "node_index": 0,
            "curve": [],
        }

    begin = time.perf_counter()
    last_log = begin
    status = "running"
    while state["node_index"] < len(MSE_THRESHOLDS) and state["level"] < MAX_LEVELS:
        if time.perf_counter() - begin >= SECONDS_PER_CASE:
            status = "time_budget"
            break
        target_threshold = float(MSE_THRESHOLDS[state["node_index"]])
        quantile = torch.quantile(state["values"], SURVIVAL_QUANTILE, dim=1)
        thresholds = torch.minimum(
            state["threshold"], quantile.clamp_min(target_threshold)
        )
        keep = state["values"] <= thresholds[:, None] + LOSS_TOLERANCE
        fractions = keep.to(torch.float64).mean(1)
        if bool((fractions <= 0).any()):
            status = "zero_survival"
            break
        choices = torch.multinomial(
            keep.to(torch.float32), PARTICLES_PER_REPLICA,
            replacement=True, generator=generators["resample"],
        )
        state["theta"] = state["theta"].gather(
            1, choices[..., None].expand_as(state["theta"])
        )
        state["values"] = state["values"].gather(1, choices)
        state["log_mass"] += fractions.log()
        state["threshold"] = thresholds
        theta, values, rates, radial = mutate(
            state["theta"], state["values"], thresholds, target, inputs,
            state["scales"], generators["mutation"],
        )
        state["theta"], state["values"] = theta, values
        state["scales"] = (state["scales"] * (0.30 * (rates - 0.30)).exp()).clamp(1e-5, 0.85)
        state["level"] += 1
        reached = bool(torch.all(thresholds <= target_threshold + 1e-9))
        if reached:
            row = record_node(state, function_id, target, inputs, case, target_threshold)
            row["level"] = int(state["level"])
            row["survival_min"] = float(fractions.min().item())
            row["survival_max"] = float(fractions.max().item())
            state["curve"].append(row)
            state["node_index"] += 1
        if reached or state["level"] % 25 == 0:
            state_save(checkpoint, state, generators)
        now = time.perf_counter()
        if reached or now - last_log >= 30:
            k = -logmeanexp(state["log_mass"]) / math.log(2.0)
            print(f"{name} level={state['level']} MSE<={min(thresholds).item():.6g} K={k:.2f} elapsed={now - begin:.1f}s", flush=True)
            last_log = now

    if status == "running":
        status = "completed" if state["node_index"] == len(MSE_THRESHOLDS) else "max_levels"
    state_save(checkpoint, state, generators)
    metadata = {
        "status": status,
        "case": case,
        "function_id": int(function_id),
        "function_hex": f"0x{function_id:02X}",
        "measure": "uniform_ball",
        "radius": RADIUS,
        "parameter_dimension": PARAMETER_DIM,
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(torch.device(DEVICE)) if device.type == "cuda" else "CPU",
        "replicas": REPLICAS,
        "particles_per_replica": PARTICLES_PER_REPLICA,
        "thresholds_completed": int(state["node_index"]),
        "levels": int(state["level"]),
        "elapsed_seconds": float(time.perf_counter() - begin),
        "curve": state["curve"],
        "script_sha256": sha256(Path(__file__).resolve()),
        "engine_sha256": sha256(ENGINE_SOURCE),
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "3bit_endpoint_grid_uniform_ball_euclidean_volume_v2",
        "measure": "Lebesgue restricted to ||theta||_2 <= R, normalized by ball volume",
        "radius": RADIUS,
        "parameter_dimension": PARAMETER_DIM,
        "rules": [f"0x{x:02X}" for x in RULE_IDS],
        "thresholds": list(MSE_THRESHOLDS),
        "low_values": list(ENDPOINT_LOW_VALUES),
        "high_values": list(ENDPOINT_HIGH_VALUES),
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(torch.device(DEVICE)) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "engine_sha256": sha256(ENGINE_SOURCE),
    }
    atomic_json(RESULT_DIR / "protocol.json", protocol)
    records = []
    for case_index, case in enumerate(cases()):
        for function_id in RULE_IDS:
            records.append(run_case(case, int(function_id), case_index))
            atomic_json(RESULT_DIR / "records.json", records)
    atomic_json(RESULT_DIR / "summary.json", {
        "protocol": protocol,
        "records": len(records),
        "completed": sum(record["status"] == "completed" for record in records),
        "records_path": str(RESULT_DIR / "records.json"),
    })
    print(json.dumps({"records": len(records), "result_dir": str(RESULT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
