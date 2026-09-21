"""8bit随机记忆的有限loss体积收缩：5090独立脚本，无需仓库中的其他文件。

直接运行本文件，或将全文粘贴到Jupyter。仅依赖numpy、torch；matplotlib可选。
默认10档n，每档8份随机数据，4个SMC副本，BCE降至0.001。
网络沿用试采的8->16->16->1 tanh。WIDTH=32时为1377参数，另存结果目录。

V_D(eps) = P_{theta~N(0,I)}[mean_i softplus(-(2y_i-1)z_theta(x_i)) <= eps]
K_D = -log2(V_D)。先对每份数据、每个副本计算指定窗口的K斜率，取副本中位数，
再对独立随机数据集求均值/标准差；不能先平均体积再取对数。

统计对象是有限loss窗口，不声称测到了渐近深尾或排除了容量效应。
较小n档只是过渡参照，不因标签随机就直接称为不可压缩的纯记忆。
随机标签不强制平衡，不重抽看起来简单的数据。不同数据编号独立，同编号的n嵌套。
沿用已审计试采的分块pCN-SMC；本版改用稳定的signed-softplus表达同一个BCE，
并将边界数值容差降至1e-9。共享自适应阈值/步长的副本仅作诊断，不是收敛证明。

Ctrl+C在完整SMC层结束后保存；再次运行跳过已完成项，继续未完成项。
每项每次运行最多20分钟，超时保存并换下一项；可修改预算后继续，无自动recheck。
也可在结果目录创建STOP文件请求停止，续跑前移除该文件。
完整粒子检查点留在结果目录；默认结果zip不含检查点，便于下载分析。
"""

from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
import math
import os
import platform
import signal
import sys
import time
import types
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F


class Config:
    # 只改这里即可；宽度改变会使用不同目录，不与旧架构混合汇总。
    WIDTH = 16
    COUNTS = (32, 48, 64, 80, 96, 128, 160, 192, 224, 256)
    DATASETS_PER_N = 8
    DATA_SEED = 202609112301
    STOP_BCE = 0.001
    THRESHOLDS = (.70, .69, .65, .60, .50, .40, .30, .20, .10,
                  .05, .025, .01, .005, .002, .001, .0005, .0002, .0001)

    REPLICAS = 4
    PARTICLES = 2048
    ADAPT_SWEEPS = 2
    MUTATION_SWEEPS = 12
    EVAL_MICRO_BATCH = 2048
    SURVIVAL_QUANTILE = .5
    INITIAL_PROPOSAL_SCALES = (.08, .06, .08, .025)
    TARGET_ACCEPTANCE = .30
    ADAPT_RATE = .35
    MIN_PROPOSAL_SCALE = 1e-4
    MAX_PROPOSAL_SCALE = .95
    LOSS_TOLERANCE = 1e-9
    PRIOR_SEED = 202609112311
    RESAMPLE_SEED = 202609112312
    MUTATION_SEED = 202609112313

    SECONDS_PER_JOB_PER_RUN = 1200.
    MAX_LEVELS_PER_JOB = 20000
    CHECKPOINT_SECONDS = 60.
    LOG_SECONDS = 30.
    SNAPSHOT_PARTICLES_PER_REPLICA = 8
    PRIOR_CALIBRATION_SAMPLES = 65536
    DEVICE = "cuda:0"
    OUTPUT_ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd() / "tmp"
    RUN_TAG = "v1"
    PACKAGE_CHECKPOINTS = False
    MAKE_PLOTS = True
    # 留空跑全部；可以指定如(256,)先跳过困难档，或(0, 1)先跑两份数据。
    SKIP_COUNTS = ()
    DATASET_IDS = None
    SELF_TEST_ONLY = False


INPUT_BITS = 8
PROTOCOL = "8bit_random_memory_scaling_pcn_v1"
STOP_REQUESTED = False
LOG_PATH = None


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    stop: int


@dataclass
class State:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume_fraction: torch.Tensor
    proposal_scales: list
    current_threshold: float
    threshold_index: int
    level: int
    level_rows: list
    threshold_rows: list


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def log(message):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + str(message)
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def replace_retry(temporary, destination):
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def atomic_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    replace_retry(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(ready(value), ensure_ascii=False, indent=2, allow_nan=False))


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    replace_retry(temporary, path)


def atomic_csv(path, rows):
    buffer = io.StringIO(newline="")
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(buffer, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(ready(v), ensure_ascii=False) if isinstance(v, (list, dict))
                             else ready(v) for k, v in row.items()})
    atomic_text(path, buffer.getvalue())


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, array.dtype.str)).encode("ascii"))
        h.update(array.tobytes())
    return h.hexdigest()


def make_data(dataset_id):
    rng = np.random.default_rng(Config.DATA_SEED + 1_000_003 * dataset_id)
    order = rng.permutation(256).astype(np.int64)
    labels = rng.integers(0, 2, 256, dtype=np.uint8)
    values = np.arange(256, dtype=np.uint16)
    inputs = ((values[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float32)
    return inputs, labels, order


def parameter_blocks(width):
    first = width * INPUT_BITS + width
    middle = first + width * width + width
    total = middle + width + 1
    return [Block("first", 0, first), Block("middle", first, middle),
            Block("output", middle, total), Block("all", 0, total)], total


def forward_logits(theta, inputs):
    batch, width = theta.shape[0], Config.WIDTH
    end = width * INPUT_BITS
    w1 = theta[:, :end].reshape(batch, width, INPUT_BITS) / math.sqrt(INPUT_BITS)
    b1 = theta[:, end:end + width] / math.sqrt(INPUT_BITS)
    start = end + width
    end = start + width * width
    w2 = theta[:, start:end].reshape(batch, width, width) / math.sqrt(width)
    b2 = theta[:, end:end + width] / math.sqrt(width)
    start = end + width
    w3 = theta[:, start:start + width].reshape(batch, 1, width) / math.sqrt(width)
    b3 = theta[:, start + width:start + width + 1] / math.sqrt(width)
    h1 = torch.tanh(torch.bmm(inputs[None].expand(batch, -1, -1), w1.transpose(1, 2)) + b1[:, None])
    h2 = torch.tanh(torch.bmm(h1, w2.transpose(1, 2)) + b2[:, None])
    return torch.bmm(h2, w3.transpose(1, 2)).squeeze(-1) + b3


@torch.no_grad()
def evaluate_losses(particles, inputs, targets):
    flat = particles.reshape(-1, particles.shape[-1])
    values = []
    signs = (2 * targets - 1)[None]
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(flat[start:start + Config.EVAL_MICRO_BATCH], inputs)
        # logaddexp避免高正margin下相减消去；数学上就是原始二分类交叉熵。
        values.append(torch.logaddexp(torch.zeros_like(logits), -signs * logits).mean(-1))
    return torch.cat(values).reshape(particles.shape[:-1])


def make_generators(device, job_id):
    offset = 1_000_003 * int(job_id)
    return {name: torch.Generator(device=device).manual_seed(seed + offset)
            for name, seed in (("prior", Config.PRIOR_SEED), ("resample", Config.RESAMPLE_SEED),
                               ("mutation", Config.MUTATION_SEED))}


def initialize_state(device, x, y, generators):
    _, dimensions = parameter_blocks(Config.WIDTH)
    particles = torch.empty(Config.REPLICAS, Config.PARTICLES, dimensions,
                            dtype=torch.float32, device=device).normal_(generator=generators["prior"])
    return State(particles, evaluate_losses(particles, x, y),
                 torch.arange(Config.REPLICAS * Config.PARTICLES, device=device).reshape(Config.REPLICAS, -1),
                 torch.zeros(Config.REPLICAS, device=device, dtype=torch.float64),
                 list(Config.INITIAL_PROPOSAL_SCALES), float("inf"), 0, 0, [], [])


def choose_next_threshold(state):
    target = float(Config.THRESHOLDS[state.threshold_index])
    adaptive = float(torch.quantile(state.losses, Config.SURVIVAL_QUANTILE, dim=1).max())
    epsilon = min(max(target, adaptive), state.current_threshold)
    return epsilon, epsilon <= target + 1e-12


@torch.no_grad()
def resample(state, epsilon, x, y, generator):
    survival = []
    new_particles, new_lineages = torch.empty_like(state.particles), torch.empty_like(state.lineages)
    for r in range(Config.REPLICAS):
        survivors = torch.nonzero(state.losses[r] <= epsilon + Config.LOSS_TOLERANCE).flatten()
        if not len(survivors):
            raise RuntimeError(f"副本{r}没有幸存粒子；保留上一完整检查点，不伪造体积。")
        choices = torch.randint(len(survivors), (Config.PARTICLES,),
                                device=state.particles.device, generator=generator)
        selected = survivors[choices]
        new_particles[r], new_lineages[r] = state.particles[r, selected], state.lineages[r, selected]
        survival.append(len(survivors) / Config.PARTICLES)
    state.particles, state.lineages = new_particles, new_lineages
    state.losses = evaluate_losses(state.particles, x, y)
    state.log_volume_fraction += torch.tensor(survival, device=state.particles.device, dtype=torch.float64).log()
    return survival


@torch.no_grad()
def mutate_block(state, block, scale, epsilon, x, y, generator):
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
    rho = min(max(float(scale), 0.), .999999)
    proposal[..., block.start:block.stop] = math.sqrt(1 - rho * rho) * current + rho * noise
    losses = evaluate_losses(proposal, x, y)
    accepted = losses <= epsilon + Config.LOSS_TOLERANCE
    state.particles[accepted] = proposal[accepted]
    state.losses[accepted] = losses[accepted]
    return float(accepted.float().mean())


def rejuvenate(state, blocks, epsilon, x, y, generator):
    scales = list(state.proposal_scales)
    for _ in range(Config.ADAPT_SWEEPS):
        for j, block in enumerate(blocks):
            rate = mutate_block(state, block, scales[j], epsilon, x, y, generator)
            scales[j] *= math.exp(Config.ADAPT_RATE * (rate - Config.TARGET_ACCEPTANCE))
            scales[j] = min(max(scales[j], Config.MIN_PROPOSAL_SCALE), Config.MAX_PROPOSAL_SCALE)
    state.proposal_scales = scales
    rates = np.zeros(len(blocks))
    for _ in range(Config.MUTATION_SWEEPS):
        for j, block in enumerate(blocks):
            rates[j] += mutate_block(state, block, scales[j], epsilon, x, y, generator)
    return (rates / Config.MUTATION_SWEEPS).tolist()


def one_level(state, blocks, x, y, generators):
    epsilon, hit = choose_next_threshold(state)
    survival = resample(state, epsilon, x, y, generators["resample"])
    rates = rejuvenate(state, blocks, epsilon, x, y, generators["mutation"])
    if not bool(torch.isfinite(state.losses).all()):
        raise RuntimeError("粒子loss出现非有限值。")
    if float(state.losses.max()) > epsilon + Config.LOSS_TOLERANCE + 2e-7 * max(epsilon, 1e-3):
        raise RuntimeError("采样粒子违反loss约束。")
    state.current_threshold = epsilon
    state.level += 1
    state.threshold_index += int(hit)
    return epsilon, hit, survival, rates


def pack_state(state, generators):
    return {"state": {f.name: getattr(state, f.name).detach().cpu().clone()
                      if isinstance(getattr(state, f.name), torch.Tensor)
                      else json.loads(json.dumps(getattr(state, f.name))) for f in fields(State)},
            "generators": {name: g.get_state().cpu() for name, g in generators.items()}}


def unpack_state(payload, generators, device):
    for name, generator in generators.items():
        generator.set_state(payload["generators"][name])
    return State(**{key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in payload["state"].items()})


def save_checkpoint(path, state, generators, elapsed, signature):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload=pack_state(state, generators), elapsed_s=elapsed, signature=signature), temporary)
    replace_retry(temporary, path)


def load_checkpoint(path, generators, device, signature):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != signature:
        raise ValueError(f"{path}的科学配置/数据不同，拒绝混用检查点；请改RUN_TAG使用新目录。")
    return unpack_state(payload["payload"], generators, device), payload["elapsed_s"]


def code_fingerprint(functions):
    # 不包含路径/行号，脚本搬家或粘贴到Jupyter不会使科学配置签名改变。
    def encode(value):
        if isinstance(value, types.CodeType):
            return [value.co_code.hex(), value.co_names, value.co_varnames,
                    value.co_argcount, value.co_kwonlyargcount, [encode(v) for v in value.co_consts]]
        return repr(value)
    data = [(f.__name__, encode(inspect.unwrap(f).__code__)) for f in functions]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def science_signature():
    names = ("WIDTH", "DATA_SEED", "REPLICAS", "PARTICLES", "ADAPT_SWEEPS", "MUTATION_SWEEPS",
             "EVAL_MICRO_BATCH", "SURVIVAL_QUANTILE", "INITIAL_PROPOSAL_SCALES", "TARGET_ACCEPTANCE",
             "ADAPT_RATE", "MIN_PROPOSAL_SCALE", "MAX_PROPOSAL_SCALE", "LOSS_TOLERANCE", "THRESHOLDS",
             "PRIOR_SEED", "RESAMPLE_SEED", "MUTATION_SEED")
    return ready(dict(protocol=PROTOCOL, **{name: getattr(Config, name) for name in names},
                      parameters=parameter_blocks(Config.WIDTH)[1], input_bits=INPUT_BITS,
                      coordinate_prior="iid N(0,1); weights AND biases scaled by 1/sqrt(fan_in)",
                      loss="mean signed-logaddexp BCE on selected training samples ONLY",
                      dtype="float32; float64 log volume", tf32=False, amp=False,
                      scientific_code_sha256=code_fingerprint((make_data, parameter_blocks, forward_logits,
                          evaluate_losses, make_generators, initialize_state, choose_next_threshold,
                          resample, mutate_block, rejuvenate, one_level))))


@torch.no_grad()
def diagnostics(state, x, y, elapsed, rates):
    correct = torch.zeros(Config.REPLICAS, device=x.device, dtype=torch.float64)
    all_correct = torch.zeros_like(correct)
    loss_min, loss_max = [], []
    for r in range(Config.REPLICAS):
        local_min, local_max = [], []
        for start in range(0, Config.PARTICLES, Config.EVAL_MICRO_BATCH):
            z = forward_logits(state.particles[r, start:start + Config.EVAL_MICRO_BATCH], x)
            ok = (z >= 0) == y.bool()[None]
            correct[r] += ok.sum()
            all_correct[r] += ok.all(-1).sum()
            per_sample = torch.logaddexp(torch.zeros_like(z), -(2 * y - 1)[None] * z)
            local_min.append(per_sample.min(-1).values)
            local_max.append(per_sample.max(-1).values)
        loss_min.append(float(torch.cat(local_min).mean()))
        loss_max.append(float(torch.cat(local_max).mean()))
    k = (-state.log_volume_fraction / math.log(2)).cpu().numpy()
    fractions = (all_correct / Config.PARTICLES).cpu().numpy()
    if state.current_threshold + Config.LOSS_TOLERANCE < math.log(2) / len(x) and np.any(fractions < 1):
        raise RuntimeError("平均BCE已低于ln(2)/n，却存在硬分类错误；需要检查数值/实现。")
    warnings = []
    if np.std(k, ddof=1) > max(10., .03 * abs(np.median(k))):
        warnings.append("REPLICA_K_SPREAD")
    if rates and min(rates) < .03:
        warnings.append("LOW_BLOCK_ACCEPTANCE")
    return dict(bce_limit=state.current_threshold, level=state.level, elapsed_s=elapsed,
                K_bits_by_replica=k.tolist(), K_median_bits=float(np.median(k)),
                K_replica_sd_bits=float(k.std(ddof=1)),
                mean_bce_by_replica=state.losses.mean(1).cpu().tolist(),
                training_bit_accuracy_by_replica=(correct / (Config.PARTICLES * len(x))).cpu().tolist(),
                training_all_correct_fraction_by_replica=fractions.tolist(),
                average_particle_min_sample_bce_by_replica=loss_min,
                average_particle_max_sample_bce_by_replica=loss_max,
                initial_lineages_remaining=[len(torch.unique(row)) for row in state.lineages],
                proposal_scales=state.proposal_scales, acceptance_by_block=rates, warnings=warnings)


def slope_windows():
    nodes = [v for v in Config.THRESHOLDS if v <= .2 and v >= Config.STOP_BCE - 1e-12]
    windows = list(zip(nodes[:-1], nodes[1:]))
    windows += [(hi, lo) for hi, lo in ((.2, .05), (.1, .01), (.05, .005), (.01, .001), (.1, .001))
                if hi in nodes and lo in nodes]
    return list(dict.fromkeys(windows))


def dataset_coefficients(rows, n, dataset_id):
    by_epsilon = {round(row["bce_limit"], 12): row for row in rows}
    result = []
    for high, low in slope_windows():
        a, b = by_epsilon.get(round(high, 12)), by_epsilon.get(round(low, 12))
        if a is None or b is None:
            continue
        coefficients = (np.array(b["K_bits_by_replica"]) - np.array(a["K_bits_by_replica"])) / math.log2(high / low)
        center = float(np.median(coefficients))
        spread = float(np.std(coefficients, ddof=1))
        result.append(dict(n=n, dataset_id=dataset_id, epsilon_high=high, epsilon_low=low,
                           c_by_replica=coefficients.tolist(), c_dataset_median=center,
                           c_replica_sd=spread, c_per_n=center / n,
                           warning=spread > max(5., .15 * abs(center)),
                           endpoint_warnings=sorted(set(a["warnings"] + b["warnings"]))))
    return result


def save_source(out):
    source, origin = None, None
    filename = globals().get("__file__")
    if filename and Path(filename).is_file():
        source, origin = Path(filename).read_text(encoding="utf-8"), str(Path(filename).resolve())
    else:
        getter = globals().get("get_ipython")
        if callable(getter):
            history = getter().history_manager.input_hist_raw
            source = next((cell for cell in reversed(history)
                           if "8bit_random_memory_scaling_pcn_v1" in cell and "def main(" in cell), None)
            origin = "Jupyter input history"
    if source:
        h = hashlib.sha256(source.encode("utf-8")).hexdigest()
        directory = out / "sources"
        directory.mkdir(exist_ok=True)
        atomic_text(directory / f"experiment_{h[:16]}.py", source)
        return dict(origin=origin, sha256=h, saved=True)
    return dict(saved=False, note="未取到完整源码；已保存核心字节码签名和全部配置，不阻止采样。")


def request_stop(signum, frame):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        raise KeyboardInterrupt
    STOP_REQUESTED = True
    print("\n已请求停止：当前SMC层结束后写检查点和结果包。", flush=True)


def should_stop(out):
    return STOP_REQUESTED or (out / "STOP").exists()


def job_signature(signature, inputs, labels, order, n, dataset_id):
    return dict(science=signature, dataset_id=dataset_id, n=n,
                data_sha256=digest(inputs[order[:n]], labels[order[:n]], order[:n]))


def save_job_outputs(directory, state, n, dataset_id, elapsed, status, endpoint, signature):
    coefficients = dataset_coefficients(state.threshold_rows, n, dataset_id)
    summary = dict(status=status, n=n, dataset_id=dataset_id, sampling_seconds=elapsed,
                   reached_bce=state.current_threshold, levels=state.level, signature=signature,
                   endpoint=endpoint, coefficients=coefficients,
                   reliability="FINITE_LOSS_ESTIMATE_NOT_CONVERGENCE_CERTIFIED",
                   note="副本共用自适应阈值/步长；副本标准差不覆盖全部混合偏差。")
    atomic_json(directory / "levels.json", state.level_rows)
    atomic_json(directory / "thresholds.json", state.threshold_rows)
    atomic_json(directory / "summary.json", summary)
    return summary


def run_job(out, n, dataset_id, device, signature):
    inputs, labels, order = make_data(dataset_id)
    directory = out / f"data_{dataset_id:03d}" / f"n{n:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    local_signature = job_signature(signature, inputs, labels, order, n, dataset_id)
    x = torch.tensor(inputs[order[:n]], device=device)
    y = torch.tensor(labels[order[:n]], dtype=torch.float32, device=device)
    atomic_npz(directory / "training_data.npz", input_ids=order[:n], inputs=inputs[order[:n]], labels=labels[order[:n]])
    generators = make_generators(device, dataset_id * 1024 + n)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists():
        state, elapsed_base = load_checkpoint(checkpoint, generators, device, local_signature)
    else:
        state, elapsed_base = initialize_state(device, x, y, generators), 0.
        save_checkpoint(checkpoint, state, generators, elapsed_base, local_signature)
    blocks, _ = parameter_blocks(Config.WIDTH)
    start = time.perf_counter()
    checkpoint_at = log_at = start
    rates = []
    status = "completed" if state.current_threshold <= Config.STOP_BCE else "time_budget"
    inside_level = False
    log(f"START data={dataset_id + 1}/{Config.DATASETS_PER_N} n={n} | level={state.level} | "
        f"BCE<={state.current_threshold:.6g} | 已采{elapsed_base / 60:.1f}分钟")
    try:
        while state.current_threshold > Config.STOP_BCE:
            if should_stop(out):
                status = "paused"
                break
            if time.perf_counter() - start >= Config.SECONDS_PER_JOB_PER_RUN:
                break
            if state.level >= Config.MAX_LEVELS_PER_JOB:
                status = "level_budget"
                break
            inside_level = True
            epsilon, hit, survival, rates = one_level(state, blocks, x, y, generators)
            elapsed = elapsed_base + time.perf_counter() - start
            state.level_rows.append(dict(level=state.level, bce_limit=epsilon, elapsed_s=elapsed,
                K_bits_by_replica=(-state.log_volume_fraction / math.log(2)).cpu().tolist(),
                survival_by_replica=survival, acceptance_by_block=rates, proposal_scales=list(state.proposal_scales)))
            if hit:
                row = diagnostics(state, x, y, elapsed, rates)
                state.threshold_rows.append(row)
                sample = state.particles[:, :Config.SNAPSHOT_PARTICLES_PER_REPLICA]
                z = forward_logits(sample.reshape(-1, sample.shape[-1]), x).reshape(sample.shape[0], sample.shape[1], n)
                atomic_npz(directory / f"snapshot_{epsilon:.6g}.npz",
                           theta=sample.cpu().numpy(), logits=z.cpu().numpy())
                log(f"HIT data={dataset_id + 1} n={n} BCE<={epsilon:g} | K={row['K_median_bits']:.2f} "
                    f"sd={row['K_replica_sd_bits']:.2f}bits | {elapsed / 60:.1f}分钟 | "
                    f"完整拟合粒子={np.mean(row['training_all_correct_fraction_by_replica']):.1%} | "
                    f"{','.join(row['warnings']) or '无内部报警(不等于已收敛)'}")
            inside_level = False
            now = time.perf_counter()
            if now - checkpoint_at >= Config.CHECKPOINT_SECONDS:
                save_checkpoint(checkpoint, state, generators, elapsed, local_signature)
                endpoint = state.threshold_rows[-1] if state.threshold_rows else None
                save_job_outputs(directory, state, n, dataset_id, elapsed, "running", endpoint, local_signature)
                checkpoint_at = time.perf_counter()
            if now - log_at >= Config.LOG_SECONDS:
                log(f"data={dataset_id + 1} n={n} level={state.level} BCE<={epsilon:.6g} | "
                    f"{elapsed / 60:.1f}分钟 | 接受率={'/'.join(f'{r:.2f}' for r in rates)}")
                log_at = time.perf_counter()
        if state.current_threshold <= Config.STOP_BCE:
            status = "completed"
    except KeyboardInterrupt:
        global STOP_REQUESTED
        STOP_REQUESTED = True
        if inside_level:
            # 不把只完成重采样、尚未完成更新的半层写成有效检查点。
            state, elapsed_base = load_checkpoint(checkpoint, generators, device, local_signature)
            start, rates = time.perf_counter(), []
            log("强制中断发生在半层：回退到最后完整检查点，最多丢失一个保存间隔。")
        status = "paused"
    except Exception as exc:
        atomic_json(directory / "error.json", dict(error=repr(exc), last_committed_checkpoint=str(checkpoint),
                    interrupted_level=state.level, note="未覆盖最后完整检查点；无自动重跑。"))
        raise
    elapsed = elapsed_base + time.perf_counter() - start
    endpoint = diagnostics(state, x, y, elapsed, rates)
    save_checkpoint(checkpoint, state, generators, elapsed, local_signature)
    result = save_job_outputs(directory, state, n, dataset_id, elapsed, status, endpoint, local_signature)
    log(f"END data={dataset_id + 1} n={n} | {status} | BCE<={state.current_threshold:.6g} | 累计{elapsed / 60:.1f}分钟")
    return result


def collect_results(out, signature):
    results, point_rows = [], []
    for dataset_id in range(Config.DATASETS_PER_N):
        inputs, labels, order = make_data(dataset_id)
        for n in Config.COUNTS:
            directory = out / f"data_{dataset_id:03d}" / f"n{n:03d}"
            path = directory / "summary.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["signature"] != job_signature(signature, inputs, labels, order, n, dataset_id):
                raise ValueError(f"结果签名不一致：{path}")
            threshold_path = directory / "thresholds.json"
            points = json.loads(threshold_path.read_text(encoding="utf-8")) if threshold_path.exists() else []
            # STOP_BCE变更后重新计算窗口，不使用旧summary内截断过的窗口列表。
            row["coefficients"] = dataset_coefficients(points, n, dataset_id)
            results.append(row)
            point_rows.extend(dict(n=n, dataset_id=dataset_id, **point) for point in points)
    return results, point_rows


def aggregate(out, signature, make_plots=False):
    results, points = collect_results(out, signature)
    coefficients = [c for result in results for c in result["coefficients"]]
    calibration_rows = []
    calibration_path = out / "prior_calibration.json"
    if calibration_path.exists():
        reference = json.loads(calibration_path.read_text(encoding="utf-8"))
        for point in points:
            if point["dataset_id"] != reference["dataset_id"] or point["n"] not in reference["counts"]:
                continue
            if point["bce_limit"] not in reference["thresholds"]:
                continue
            j, k = reference["counts"].index(point["n"]), reference["thresholds"].index(point["bce_limit"])
            p = reference["probability"][j][k]
            estimated = 2 ** (-point["K_median_bits"])
            calibration_rows.append(dict(n=point["n"], bce_limit=point["bce_limit"],
                direct_prior_probability=p, smc_probability_from_median_K=estimated,
                direct_prior_mc_se=math.sqrt(p * (1 - p) / reference["samples"]),
                relative_difference=(estimated - p) / p if p > 0 else None,
                note="只校准浅节点；不能据此认证深尾。"))
    groups = []
    for high, low in slope_windows():
        for n in Config.COUNTS:
            selected = [r for r in coefficients if r["n"] == n and r["epsilon_high"] == high and r["epsilon_low"] == low]
            if not selected:
                continue
            values = np.array([r["c_dataset_median"] for r in selected])
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            groups.append(dict(n=n, epsilon_high=high, epsilon_low=low, datasets_reached=len(values),
                datasets_planned=Config.DATASETS_PER_N, coverage=len(values) / Config.DATASETS_PER_N,
                c_mean=float(values.mean()), c_median=float(np.median(values)), c_dataset_sd=sd,
                c_dataset_sem=sd / math.sqrt(len(values)) if sd is not None else None,
                mean_c_per_n=float(values.mean() / n), dataset_ids=[r["dataset_id"] for r in selected],
                sampler_warning_datasets=sum(bool(r["warning"] or r["endpoint_warnings"]) for r in selected)))
    coverage = []
    for n in Config.COUNTS:
        selected = [r for r in results if r["n"] == n]
        done = sum(r["reached_bce"] is not None and r["reached_bce"] <= Config.STOP_BCE for r in selected)
        coverage.append(dict(n=n, started=len(selected), reached_target=done, planned=Config.DATASETS_PER_N,
                             unfinished_or_unstarted=Config.DATASETS_PER_N - done))
    fits = []
    for high, low in slope_windows():
        # 主线性拟合只用该窗口所有数据集都已完成的n，避免默默平均容易样本。
        rows = [r for r in groups if r["epsilon_high"] == high and r["epsilon_low"] == low and r["coverage"] == 1]
        if len(rows) < 3:
            continue
        x, y = np.array([r["n"] for r in rows]), np.array([r["c_mean"] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        total = np.sum((y - y.mean()) ** 2)
        fits.append(dict(epsilon_high=high, epsilon_low=low, n_values=x.tolist(), slope=float(slope),
                         intercept=float(intercept), r_squared=float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
                         through_origin_slope=float(x @ y / (x @ x)),
                         note="描述性拟合；同编号的不同n嵌套，不把n点当独立重复做显著性检验。"))
    report = dict(protocol=signature, stop_bce=Config.STOP_BCE, coverage=coverage, groups=groups, linear_fits=fits,
                  total_sampling_seconds=sum(r["sampling_seconds"] for r in results),
                  planned_jobs=len(Config.COUNTS) * Config.DATASETS_PER_N,
                  completed_jobs=sum(r["reached_target"] for r in coverage),
                  estimator="每副本差分K -> 每数据集取副本中位数 -> 独立数据集等权平均；未平均体积",
                  caveats=["不同n嵌套；数据集编号才是独立重复。",
                           "误差条为数据集间标准差，不覆盖SMC系统性偏差。",
                           "覆盖不全的均值可能有完成者偏差；逐窗口显示覆盖率，不作为完整结论。",
                           "有限loss窗口，未证明纯记忆/深尾渐近律，也未排除容量或混合瓶颈。"])
    atomic_json(out / "summary.json", report)
    atomic_csv(out / "dataset_coefficients.csv", coefficients)
    atomic_csv(out / "coefficient_summary.csv", groups)
    atomic_csv(out / "threshold_points.csv", points)
    atomic_csv(out / "prior_calibration_comparison.csv", calibration_rows)
    lines = ["# 8bit随机记忆体积收缩", "", f"网络：8->{Config.WIDTH}->{Config.WIDTH}->1 tanh，"
             f"{parameter_blocks(Config.WIDTH)[1]}参数。目标BCE={Config.STOP_BCE:g}。",
             f"已到目标 {report['completed_jobs']}/{report['planned_jobs']}；"
             f"累计采样 {report['total_sampling_seconds'] / 3600:.2f} GPU小时。", "",
             "c = [K(eps_low)-K(eps_high)] / log2(eps_high/eps_low)。",
             "c表示loss缩小一倍时，负log2体积平均增加多少bit；不是直接声明Kolmogorov复杂度。", "",
             "|n|目标完成数|计划数|", "|---:|---:|---:|"]
    lines += [f"|{r['n']}|{r['reached_target']}|{r['planned']}|" for r in coverage]
    lines += ["", "## 指定窗口", "", "|BCE窗口|n|数据数|c均值|数据集间SD|c/n|有采样报警数据数|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        sd = f"{r['c_dataset_sd']:.2f}" if r["c_dataset_sd"] is not None else "NA"
        lines.append(f"|{r['epsilon_high']:g}->{r['epsilon_low']:g}|{r['n']}|"
                     f"{r['datasets_reached']}/{r['datasets_planned']}|{r['c_mean']:.2f}|{sd}|"
                     f"{r['mean_c_per_n']:.3f}|{r['sampler_warning_datasets']}|")
    lines += ["", "## 限制", ""] + ["- " + note for note in report["caveats"]]
    lines += ["", "结果zip默认不含checkpoint.pt；续跑需要保留整个原始结果目录。"]
    atomic_text(out / "report_zh.md", "\n".join(lines) + "\n")
    if make_plots and Config.MAKE_PLOTS:
        plot_results(out, groups, coefficients)
    return report


def plot_results(out, groups, coefficients):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("未安装matplotlib，CSV/JSON已正常生成，跳过绘图。")
        return
    windows = [(h, l) for h, l in ((.2, .1), (.1, .05), (.1, .01), (.01, .001))
               if any(r["epsilon_high"] == h and r["epsilon_low"] == l for r in groups)]
    if not windows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (high, low) in zip(axes.flat, windows):
        rows = [r for r in groups if r["epsilon_high"] == high and r["epsilon_low"] == low]
        for dataset_id in range(Config.DATASETS_PER_N):
            local = [r for r in coefficients if r["dataset_id"] == dataset_id and r["epsilon_high"] == high and r["epsilon_low"] == low]
            if local:
                local.sort(key=lambda r: r["n"])
                ax.plot([r["n"] for r in local], [r["c_dataset_median"] for r in local], color="#9d9d9d", alpha=.35, linewidth=1)
        rows.sort(key=lambda r: r["n"])
        ax.errorbar([r["n"] for r in rows], [r["c_mean"] for r in rows],
                    yerr=[r["c_dataset_sd"] or 0 for r in rows], fmt="o-", capsize=3, color="#14756d")
        for r in rows:
            ax.annotate(f"{r['datasets_reached']}/{r['datasets_planned']}", (r["n"], r["c_mean"]),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=f"BCE {high:g} -> {low:g}", xlabel="Training samples n", ylabel="K slope per loss halving")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(windows):]:
        ax.set_visible(False)
    fig.suptitle("Random-label memory: dataset means +/- between-dataset SD; labels = coverage", fontsize=12)
    fig.savefig(out / "contraction_vs_n.png", dpi=160)
    plt.close(fig)


def package(out):
    destination = out.with_name(out.name + "_package.zip")
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp") or path.name in ("STOP", "RUNNING.lock"):
                continue
            if not Config.PACKAGE_CHECKPOINTS and path.suffix == ".pt":
                continue
            archive.write(path, path.relative_to(out).as_posix())
    replace_retry(temporary, destination)
    return destination


def validate_config():
    if Config.WIDTH < 1 or Config.REPLICAS < 2 or Config.PARTICLES < 16:
        raise ValueError("WIDTH>=1，REPLICAS>=2，PARTICLES>=16。")
    if not Config.COUNTS or len(set(Config.COUNTS)) != len(Config.COUNTS) or any(n < 1 or n > 256 for n in Config.COUNTS):
        raise ValueError("COUNTS必须是不重复的1..256整数。")
    if Config.DATASETS_PER_N < 1 or Config.MUTATION_SWEEPS < 1 or Config.ADAPT_SWEEPS < 0:
        raise ValueError("数据重复数/更新步数设置无效。")
    if Config.STOP_BCE not in Config.THRESHOLDS or any(a <= b for a, b in zip(Config.THRESHOLDS, Config.THRESHOLDS[1:])):
        raise ValueError("THRESHOLDS须严格递减，STOP_BCE须是其中一个节点。")
    if Config.SECONDS_PER_JOB_PER_RUN <= 0 or Config.EVAL_MICRO_BATCH < 1:
        raise ValueError("预算和微批必须为正。")
    ids = range(Config.DATASETS_PER_N) if Config.DATASET_IDS is None else Config.DATASET_IDS
    if len(set(ids)) != len(ids) or any(i < 0 or i >= Config.DATASETS_PER_N for i in ids):
        raise ValueError("DATASET_IDS不合法。")


def preflight(device):
    original = (Config.PARTICLES, Config.ADAPT_SWEEPS, Config.MUTATION_SWEEPS)
    Config.PARTICLES, Config.ADAPT_SWEEPS, Config.MUTATION_SWEEPS = 32, 1, 1
    try:
        inputs, labels, order = make_data(0)
        x = torch.tensor(inputs[order[:13]], device=device)
        y = torch.tensor(labels[order[:13]], device=device, dtype=torch.float32)
        generators = make_generators(device, 999999)
        state = initialize_state(device, x, y, generators)
        theta = state.particles[0, :8].double().cpu().numpy()
        xx, yy = x.double().cpu().numpy(), y.double().cpu().numpy()
        width = Config.WIDTH
        first = width * INPUT_BITS
        h1 = np.tanh(xx[None] @ (theta[:, :first].reshape(-1, width, INPUT_BITS) / math.sqrt(INPUT_BITS)).transpose(0, 2, 1)
                     + theta[:, None, first:first + width] / math.sqrt(INPUT_BITS))
        start, end = first + width, first + width + width * width
        h2 = np.tanh(h1 @ (theta[:, start:end].reshape(-1, width, width) / math.sqrt(width)).transpose(0, 2, 1)
                     + theta[:, None, end:end + width] / math.sqrt(width))
        z = (h2 * theta[:, None, end + width:end + 2 * width]).sum(-1) / math.sqrt(width) + theta[:, end + 2 * width, None] / math.sqrt(width)
        expected = np.logaddexp(0, -(2 * yy - 1)[None] * z).mean(-1)
        error = float(np.max(np.abs(expected - state.losses[0, :8].cpu().numpy())))
        assert error < 3e-6, error
        raw_logits = forward_logits(state.particles[0, :8], x)
        bce_error = float((F.binary_cross_entropy_with_logits(raw_logits, y[None].expand_as(raw_logits), reduction="none").mean(-1)
                           - state.losses[0, :8]).abs().max())
        assert bce_error < 3e-6
        before = pack_state(state, generators)
        blocks, _ = parameter_blocks(Config.WIDTH)
        one_level(state, blocks, x, y, generators)
        after = pack_state(state, generators)
        restored = unpack_state(before, generators, device)
        one_level(restored, blocks, x, y, generators)
        replay = pack_state(restored, generators)
        for key in ("particles", "losses", "lineages", "log_volume_fraction"):
            assert torch.equal(after["state"][key], replay["state"][key]), key
        for key in generators:
            assert torch.equal(after["generators"][key], replay["generators"][key]), key
        return dict(numpy_float64_loss_max_error=error, torch_bce_max_error=bce_error,
                    one_level_checkpoint_replay_bitwise_equal=True,
                    note="这是工程自检，不是深尾混合可靠性认证。")
    finally:
        Config.PARTICLES, Config.ADAPT_SWEEPS, Config.MUTATION_SWEEPS = original


@torch.no_grad()
def prior_calibration(out, device):
    path = out / "prior_calibration.json"
    if path.exists() or Config.PRIOR_CALIBRATION_SAMPLES <= 0:
        return
    inputs, labels, order = make_data(0)
    x = torch.tensor(inputs[order], device=device)
    y = torch.tensor(labels[order], dtype=torch.float32, device=device)
    thresholds = (.70, .69)
    counts = np.zeros((len(Config.COUNTS), len(thresholds)), dtype=np.int64)
    _, dimensions = parameter_blocks(Config.WIDTH)
    generator = torch.Generator(device=device).manual_seed(Config.PRIOR_SEED + 77777777)
    for start in range(0, Config.PRIOR_CALIBRATION_SAMPLES, Config.EVAL_MICRO_BATCH):
        size = min(Config.EVAL_MICRO_BATCH, Config.PRIOR_CALIBRATION_SAMPLES - start)
        theta = torch.randn(size, dimensions, generator=generator, device=device)
        z = forward_logits(theta, x)
        cumulative = torch.logaddexp(torch.zeros_like(z), -(2 * y - 1)[None] * z).cumsum(-1)
        for j, n in enumerate(Config.COUNTS):
            loss = cumulative[:, n - 1] / n
            for k, threshold in enumerate(thresholds):
                counts[j, k] += int((loss <= threshold + Config.LOSS_TOLERANCE).sum())
        if should_stop(out):
            return
    atomic_json(path, dict(dataset_id=0, samples=Config.PRIOR_CALIBRATION_SAMPLES, counts=Config.COUNTS,
                          thresholds=thresholds, hits=counts, probability=counts / Config.PRIOR_CALIBRATION_SAMPLES,
                          note="仅用于直接先验质量核对，不能验证深尾可靠性。"))


def main():
    global STOP_REQUESTED, LOG_PATH
    STOP_REQUESTED = False
    validate_config()
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到CUDA版PyTorch；不自动退回CPU长时间采样。")
    device = torch.device(Config.DEVICE)
    if device.type != "cuda":
        raise ValueError("本脚本正式采样只使用CUDA；请检查Config.DEVICE。")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = Config.OUTPUT_ROOT / f"results_8bit_random_memory_scaling_w{Config.WIDTH}_{Config.RUN_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    signature = science_signature()
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("目录已有不同科学配置的结果。更改RUN_TAG后使用新目录，勿覆盖旧实验。")
    lock = out / "RUNNING.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"发现{lock}。确认旧进程已停止后再移除锁文件；不要让两份脚本同时写此目录。") from None
    os.close(lock_fd)
    previous_handlers = {}
    LOG_PATH = out / "run.log"
    report = None
    try:
        atomic_text(lock, json.dumps(dict(pid=os.getpid(), started=time.time(), host=platform.node())))
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        source = save_source(out)
        atomic_json(protocol_path, dict(signature=signature, scope="有限loss、固定网络的随机标签记忆收缩，不假定每份随机数据不可压缩"))
        env = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                   cuda=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(device),
                   gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                   source=source, configuration={k: getattr(Config, k) for k in vars(Config) if k.isupper()})
        invocations = out / "invocations"
        invocations.mkdir(exist_ok=True)
        atomic_json(invocations / f"{time.time_ns()}.json", env)
        log(f"=== 8bit随机记忆 === {env['gpu']} | 8->{Config.WIDTH}->{Config.WIDTH}->1 | "
            f"{parameter_blocks(Config.WIDTH)[1]}参数 | {len(Config.COUNTS)}档x{Config.DATASETS_PER_N}份数据")
        log(f"每份{Config.REPLICAS}x{Config.PARTICLES}粒子 | 目标BCE={Config.STOP_BCE:g} | "
            f"每项本次预算{Config.SECONDS_PER_JOB_PER_RUN / 60:g}分钟 | 无自动recheck")
        log(f"输出：{out}")
        checks = preflight(device)
        atomic_json(out / "preflight.json", checks)
        log(f"自检通过：NumPy误差{checks['numpy_float64_loss_max_error']:.2g}，检查点重放逐位一致。")
        if Config.SELF_TEST_ONLY:
            return
        for dataset_id in range(Config.DATASETS_PER_N):
            inputs, labels, order = make_data(dataset_id)
            directory = out / f"data_{dataset_id:03d}"
            directory.mkdir(exist_ok=True)
            atomic_npz(directory / "full_random_data.npz", inputs=inputs, labels=labels, order=order,
                       seed=np.array(Config.DATA_SEED + 1_000_003 * dataset_id, dtype=np.int64))
        prior_calibration(out, device)
        ids = range(Config.DATASETS_PER_N) if Config.DATASET_IDS is None else Config.DATASET_IDS
        jobs = [(n, i) for i in ids for n in Config.COUNTS if n not in Config.SKIP_COUNTS]
        for index, (n, dataset_id) in enumerate(jobs):
            if should_stop(out):
                break
            path = out / f"data_{dataset_id:03d}" / f"n{n:03d}" / "summary.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                inputs, labels, order = make_data(dataset_id)
                if existing["signature"] != job_signature(signature, inputs, labels, order, n, dataset_id):
                    raise ValueError(f"结果配置不一致：{path}")
                if existing["reached_bce"] is not None and existing["reached_bce"] <= Config.STOP_BCE:
                    log(f"SKIP {index + 1}/{len(jobs)} data={dataset_id + 1} n={n} 已达目标")
                    continue
            run_job(out, n, dataset_id, device, signature)
            report = aggregate(out, signature)
            log(f"PROGRESS 本次队列{index + 1}/{len(jobs)} | 整个矩阵已达目标 "
                f"{report['completed_jobs']}/{report['planned_jobs']}")
            if (index + 1) % len(Config.COUNTS) == 0 and not should_stop(out):
                log(f"阶段结果包：{package(out)}")
    finally:
        try:
            active_error = sys.exc_info()[1]
            report = aggregate(out, signature, make_plots=True)
            final_status = ("error" if active_error is not None else "self_test_only" if Config.SELF_TEST_ONLY
                            else "paused" if should_stop(out) else "completed" if report["completed_jobs"] == report["planned_jobs"]
                            else "本次队列结束，仍有未完成项")
            atomic_json(out / "last_run_status.json", dict(status=final_status,
                        error=repr(active_error) if active_error is not None else None))
            log(f"FINAL {final_status} | "
                f"已达目标{report['completed_jobs']}/{report['planned_jobs']} | "
                f"未完成项保留检查点，下次继续；没有自动复测。")
            log(f"结果包：{package(out)}")
            if not Config.PACKAGE_CHECKPOINTS:
                log("zip不含粒子检查点；续跑请保留autodl-tmp里的完整结果目录。")
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            lock.unlink(missing_ok=True)
            LOG_PATH = None


if __name__ == "__main__":
    main()
