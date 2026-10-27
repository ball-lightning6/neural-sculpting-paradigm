"""4-bit全部1992严格轨道的独立SMC曲线与质量复测。

运行生成的experiment_4bit_full_curve_scan.py，或整份放进Jupyter一个cell。
不需要argparse、其他代码、旧结果或数据集。只依赖numpy和torch。
首轮预计约一天，不保证所有深尾都可靠；所有警告和复测原样保留。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Config:
    RESULT_DIR = (Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd()) / "results_4bit_full_curve_scan"
    DEVICE = "cuda"
    WIDTH = 16
    REPLICAS = 8
    PARTICLES_PER_REPLICA = 4096
    TARGET_THRESHOLDS = (.7, .5, .3, .2, .12, .08, .06, .04,
                         .025, .016, .01, .0063, .004, .0025, .0016, .001)
    SURVIVAL_QUANTILE = .5
    ADAPT_SWEEPS = 4
    MUTATION_SWEEPS = 12
    TARGET_ACCEPTANCE = .30
    ADAPT_RATE = .35
    INITIAL_PROPOSAL_SCALES = (.10, .10, .04)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = .95
    LOSS_TOLERANCE = 1e-7
    MIN_LEVEL_DECREMENT = 1e-8
    EVAL_MICRO_BATCH = 32768

    # 调大这三个预算可继续未完成工作，不改变采样目标或随机流。
    MAX_SECONDS_PER_TARGET = 300
    MAX_LEVELS_PER_TARGET = 6000
    VERIFY_BUDGET_SECONDS = 7200
    CHECKPOINT_EVERY_LEVELS = 50
    LOG_EVERY_LEVELS = 100
    ARCHIVE_EVERY_TARGETS = 100
    MIN_FREE_DISK_GB = 5
    MAX_CONSECUTIVE_FAILURES = 3

    # 独立新seed、两倍粒子、两倍固定更新；不是从可疑终点接着走。
    VERIFY_PARTICLE_MULTIPLIER = 2
    VERIFY_SWEEP_MULTIPLIER = 2
    AUDIT_RANDOM_ORBITS = 24
    REPLICA_SD_WARN_BITS = 5.
    REPLICA_RANGE_WARN_BITS = 20.
    VOLUME_ESS_WARN = 1.5
    MEAN_MEDIAN_WARN_BITS = 3.
    RECHECK_MIN_TOLERANCE_BITS = 2.
    RECHECK_SE_MULTIPLIER = 4.
    PREFLIGHT_MAX_WARN_FRACTION = .60
    CALIBRATION_MIN_HITS = 200
    CALIBRATION_SIGMAS = 6.
    CALIBRATION_RELATIVE_FLOOR = .15
    NUMERIC_ABSOLUTE_TOLERANCE = 2e-6
    NUMERIC_RELATIVE_TOLERANCE = 2e-4
    AUDIT_PARTICLES_PER_REPLICA = 16
    PRIOR_SAMPLES = 16_777_216
    CALIBRATION_SAMPLES = 1_048_576
    PRIOR_CHECKPOINT_SAMPLES = 1_048_576

    STRATUM_SAMPLE_COUNTS = (1, 2, 2, 2, 3)
    PANEL_SEED = 2026090601
    ORDER_SEED = 2026090670
    AUDIT_SEED = 2026090671
    PRIOR_SEED = 2026090672
    RESAMPLE_SEED = 2026090673
    MUTATION_SEED = 2026090674
    RESUME = True
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


PROTOCOL = "4bit_zero_one_full_scan_v1"
STRATA = ((0, 0), (1, 3), (4, 5), (6, 7), (8, 8))
LOG_PATH = None



CODE_BUILD_SHA256 = '303d90a4a4948c32c7d049f4d47270ef2f4e0b7415e96aa7da3a73582cb8d745'

def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(k for r in records for k in r))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    os.replace(tmp, path)


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "未估计"
    s = max(0, round(seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bit_inputs():
    return (((np.arange(16, dtype=np.uint32)[:, None] >> np.arange(3, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8))


def bits_of(function_id):
    return ((int(function_id) >> np.arange(16)) & 1).astype(np.float32)


def function_id(bits):
    return int(np.sum(np.asarray(bits, dtype=np.uint32) << np.arange(16, dtype=np.uint32)))


def stratum_id(positive_count):
    minority = min(positive_count, 16 - positive_count)
    return next(i for i, (a, b) in enumerate(STRATA) if a <= minority <= b)


def build_orbits():
    x = bit_inputs()
    place = np.array([8, 4, 2, 1], dtype=np.uint32)
    maps = np.array([(x[:, p] * place).sum(1) for p in itertools.permutations(range(4))], dtype=np.uint32)
    representative = np.full(65536, -1, dtype=np.int32)
    orbits = []
    for fid in range(65536):
        if representative[fid] >= 0:
            continue
        transformed = (((np.uint32(fid) >> maps) & 1) << np.arange(16, dtype=np.uint32)).sum(1).astype(np.uint32)
        members = np.unique(np.r_[transformed, transformed ^ 65535])
        assert not np.any(representative[members] >= 0)
        representative[members] = fid
        count = int(fid).bit_count()
        orbits.append({"orbit_index": len(orbits), "function_id": fid, "function_hex": f"0x{fid:04X}",
                       "orbit_size": len(members), "positive_count": count, "stratum": stratum_id(count),
                       "truth_table_x0000_to_x1111": "".join(map(str, bits_of(fid).astype(int)))})
    assert len(orbits) == 1992 and np.all(representative >= 0)
    assert sum(o["orbit_size"] for o in orbits) == 65536
    assert sum(o["positive_count"] == 8 for o in orbits) == 365
    return orbits, representative


def build_panel(orbits, representative):
    rng = np.random.default_rng(Config.PANEL_SEED)
    chosen = {}
    population = []
    for h, bounds in enumerate(STRATA):
        group = [o for o in orbits if o["stratum"] == h]
        n = min(Config.STRATUM_SAMPLE_COUNTS[h], len(group))
        population.append({"stratum": h, "minority_count_range": bounds, "population_orbits": len(group), "sampled_orbits": n})
        for i in rng.choice(len(group), n, replace=False):
            o = group[int(i)]
            chosen[o["function_id"]] = {**o, "random_sample": True, "stress_names": []}
    x = bit_inputs()
    named = {
        "constant0": np.zeros(16, dtype=np.uint32), "parity1": x[:, 0],
        "majority3": (x[:, :3].sum(1) >= 2).astype(np.uint32),
        "mux3": np.where(x[:, 0], x[:, 1], x[:, 2]),
        "parity2": x[:, :2].sum(1) % 2, "parity3": x[:, :3].sum(1) % 2,
        "parity4": x.sum(1) % 2,
    }
    named["parity4_flip0000"] = named["parity4"].copy()
    named["parity4_flip0000"][0] ^= 1
    by_id = {o["function_id"]: o for o in orbits}
    if not Config.SMOKE_TEST:
        for name, values in named.items():
            fid = int(representative[function_id(values)])
            if fid not in chosen:
                chosen[fid] = {**by_id[fid], "random_sample": False, "stress_names": []}
            chosen[fid]["stress_names"].append(name)
    # 随机分层面板先轮流覆盖每组；难例只用于压力检查，不混入总体均值。
    ordered = []
    queues = [[o for o in chosen.values() if o["random_sample"] and o["stratum"] == h] for h in range(len(STRATA))]
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.pop(0))
    ordered.extend(o for o in chosen.values() if not o["random_sample"])
    return ordered, population


def parameter_count():
    return 6 * Config.WIDTH + 1


def blocks():
    return (("hidden", 0, 5 * Config.WIDTH),
            ("output", 5 * Config.WIDTH, parameter_count()),
            ("all", 0, parameter_count()))


def forward_logits(theta, x):
    n, w = theta.shape[0], Config.WIDTH
    weight = theta[:, :4*w].reshape(n, w, 4) / 2
    bias = theta[:, 4*w:5*w] / 2
    hidden = torch.tanh(torch.bmm(x[None].expand(n, -1, -1), weight.transpose(1, 2)) + bias[:, None])
    out_weight = theta[:, 5*w:6*w] / math.sqrt(w)
    out_bias = theta[:, 6*w:] / math.sqrt(w)
    return torch.bmm(hidden, out_weight[:, :, None]).squeeze(-1) + out_bias


@torch.no_grad()
def hard_ids(theta, x):
    flat = theta.reshape(-1, theta.shape[-1])
    powers = 1 << torch.arange(16, device=theta.device, dtype=torch.int64)
    pieces = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        z = forward_logits(flat[start:start + Config.EVAL_MICRO_BATCH], x)
        pieces.append(((z >= 0).to(torch.int64) * powers).sum(1).cpu().numpy())
    return np.concatenate(pieces).reshape(theta.shape[:-1])


def generators(device, fid):
    result = {}
    for key, seed in [("prior", Config.PRIOR_SEED), ("resample", Config.RESAMPLE_SEED), ("mutation", Config.MUTATION_SEED)]:
        result[key] = torch.Generator(device=device).manual_seed(seed + 1_000_003 * int(fid))
    return result


@torch.no_grad()
def resample(state, threshold, x, y, generator):
    old = state["particles"]
    new = torch.empty_like(old)
    lineage = torch.empty_like(state["lineages"])
    survival = []
    for r in range(Config.REPLICAS):
        kept = torch.nonzero(state["losses"][r] <= threshold + Config.LOSS_TOLERANCE).flatten()
        if not len(kept):
            raise RuntimeError(f"副本{r}没有幸存粒子。")
        selected = kept[torch.randint(len(kept), (Config.PARTICLES_PER_REPLICA,), device=old.device, generator=generator)]
        new[r] = old[r, selected]
        lineage[r] = state["lineages"][r, selected]
        survival.append(len(kept) / Config.PARTICLES_PER_REPLICA)
    state["particles"], state["lineages"] = new, lineage
    state["losses"] = losses(new, x, y)
    state["log_volume"] = state["log_volume"] + torch.tensor(survival, device=old.device, dtype=torch.float64).log()
    return survival


@torch.no_grad()
def mutate(state, block, rho, threshold, x, y, generator):
    _, start, stop = block
    proposed = state["particles"].clone()
    current = proposed[..., start:stop]
    noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
    proposed[..., start:stop] = math.sqrt(1-rho*rho) * current + rho * noise
    proposed_loss = losses(proposed, x, y)
    accept = proposed_loss <= threshold + Config.LOSS_TOLERANCE
    state["particles"][accept] = proposed[accept]
    state["losses"][accept] = proposed_loss[accept]
    return float(accept.float().mean())


def rejuvenate(state, threshold, x, y, generator):
    scales = list(state["scales"])
    for _ in range(Config.ADAPT_SWEEPS):
        for j, block in enumerate(blocks()):
            rate = mutate(state, block, scales[j], threshold, x, y, generator)
            scales[j] *= math.exp(Config.ADAPT_RATE * (rate-Config.TARGET_ACCEPTANCE))
            scales[j] = min(Config.MAX_PROPOSAL_SCALE, max(Config.MIN_PROPOSAL_SCALE, scales[j]))
    state["scales"] = scales
    rates = np.zeros(3)
    for _ in range(Config.MUTATION_SWEEPS):
        for j, block in enumerate(blocks()):
            rates[j] += mutate(state, block, scales[j], threshold, x, y, generator)
    return (rates / Config.MUTATION_SWEEPS).tolist()


def boundary_snapshot(state, gens):
    # resample会分配新张量，因此保留旧张量引用即可回滚整层，避免中断后重复累计体积。
    snapshot = dict(state)
    for key in ["scales", "threshold_rows", "replica_rows", "level_rows"]:
        snapshot[key] = list(state[key])
    rng = {k: g.get_state().clone() for k, g in gens.items()}
    return snapshot, rng


def restore_rng(gens, rng):
    for key, value in rng.items():
        gens[key].set_state(value.cpu())


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(ready(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch(path, payload):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextmanager
def exclusive_run(root):
    # 操作系统锁随进程退出自动释放，不会因遗留PID阻止断点恢复。
    path = root / ".run.lock"
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("已有进程正在写这个结果目录；不要同时运行两个cell。") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def make_plan(orbits, representative):
    panel, _ = build_panel(orbits, representative)
    chosen = {t["function_id"] for t in panel}
    rng = np.random.default_rng(Config.ORDER_SEED)
    rest = [o for o in orbits if o["function_id"] not in chosen]
    rng.shuffle(rest)
    targets = panel + [{**o, "random_sample": False, "stress_names": []} for o in rest]
    audit_rng = np.random.default_rng(Config.AUDIT_SEED)
    random_ids = audit_rng.choice([o["function_id"] for o in orbits],
                                 min(Config.AUDIT_RANDOM_ORBITS, len(orbits)), replace=False)
    forced = sorted(set(map(int, random_ids)) | {t["function_id"] for t in panel})
    if Config.SMOKE_TEST:
        targets = targets[:2]
        panel = targets
        forced = [t["function_id"] for t in targets]
    return {"targets": targets, "preflight_ids": [t["function_id"] for t in panel],
            "audit_ids": forced, "random_audit_ids": list(map(int, random_ids)),
            "population_orbits": 1992, "expanded_functions": 65536}


def protocol_config():
    runtime = {"RESULT_DIR", "DEVICE", "MAX_SECONDS_PER_TARGET", "MAX_LEVELS_PER_TARGET",
               "VERIFY_BUDGET_SECONDS", "CHECKPOINT_EVERY_LEVELS", "LOG_EVERY_LEVELS",
               "ARCHIVE_EVERY_TARGETS", "MIN_FREE_DISK_GB", "RESUME", "PACKAGE_RESULTS"}
    return {"protocol": PROTOCOL, "code_build_sha256": CODE_BUILD_SHA256, "input_encoding": "zero_one",
            "prior": "standard_Gaussian; weights_AND_biases_divided_by_sqrt_fan_in",
            "hard_exact_threshold": math.log(2) / 16,
            "settings": {k: ready(v) for k, v in vars(Config).items() if k.isupper() and k not in runtime}}


@contextmanager
def job_settings(phase):
    old_particles, old_sweeps = Config.PARTICLES_PER_REPLICA, Config.MUTATION_SWEEPS
    if phase == "recheck":
        Config.PARTICLES_PER_REPLICA *= Config.VERIFY_PARTICLE_MULTIPLIER
        Config.MUTATION_SWEEPS *= Config.VERIFY_SWEEP_MULTIPLIER
    try:
        yield
    finally:
        Config.PARTICLES_PER_REPLICA, Config.MUTATION_SWEEPS = old_particles, old_sweeps


@torch.no_grad()
def losses(theta, x, y):
    flat = theta.reshape(-1, theta.shape[-1])
    parts = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        z = forward_logits(flat[start:start + Config.EVAL_MICRO_BATCH], x)
        # 避免大正logit处softplus(z)-z的浮点相消。
        parts.append(F.softplus((1 - 2 * y) * z).mean(1))
    result = torch.cat(parts).reshape(theta.shape[:-1])
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("loss出现NaN/Inf。")
    return result


def new_state(device, x, y, gens):
    shape = (Config.REPLICAS, Config.PARTICLES_PER_REPLICA, parameter_count())
    theta = torch.randn(shape, device=device, generator=gens["prior"])
    return {"particles": theta, "losses": losses(theta, x, y),
            "lineages": torch.arange(shape[0] * shape[1], device=device).reshape(shape[:2]),
            "log_volume": torch.zeros(shape[0], device=device, dtype=torch.float64),
            "scales": list(Config.INITIAL_PROPOSAL_SCALES), "threshold": float("inf"),
            "threshold_index": 0, "level": 0, "elapsed_seconds": 0.,
            "algorithm_seconds": 0., "discarded_attempt_seconds": 0.,
            "threshold_rows": [], "replica_rows": [], "level_rows": [], "audit_samples": []}


def statistics(log_volumes):
    lv = np.asarray(log_volumes, dtype=np.float64)
    if lv.ndim != 1 or len(lv) < 2 or not np.isfinite(lv).all() or np.max(lv) > 1e-9:
        raise RuntimeError("逐副本log体积无效。")
    k = -lv / math.log(2)
    p = np.exp(lv - lv.max())
    mean_lv = float(lv.max() + math.log(p.mean()))
    return {"neural_k_bits_median": float(np.median(k)),
            "neural_k_bits_mean_volume": -mean_lv / math.log(2),
            "neural_k_bits_sd": float(k.std(ddof=1)),
            "replica_range_bits": float(np.ptp(k)),
            "volume_weighted_replicas": float(p.sum() ** 2 / (p @ p)),
            "median_mean_gap_bits": abs(float(np.median(k)) + mean_lv / math.log(2))}


def row_warnings(row):
    flags = []
    if row["neural_k_bits_sd"] > Config.REPLICA_SD_WARN_BITS:
        flags.append("REPLICA_SD")
    if row["replica_range_bits"] > Config.REPLICA_RANGE_WARN_BITS:
        flags.append("REPLICA_RANGE")
    if (row["volume_weighted_replicas"] < Config.VOLUME_ESS_WARN
            and row["median_mean_gap_bits"] > Config.MEAN_MEDIAN_WARN_BITS):
        flags.append("VOLUME_DOMINATED_BY_FEW_REPLICAS")
    if row.get("numeric_mismatch", False):
        flags.append("FLOAT32_FLOAT64_MISMATCH")
    if row.get("accept_hidden", 1) < .01 or row.get("accept_all", 1) < .01:
        flags.append("LOW_MOVEMENT_ACCEPTANCE")
    if row.get("scale_min", 1) <= Config.MIN_PROPOSAL_SCALE * 1.001:
        flags.append("PROPOSAL_SCALE_FLOOR")
    return flags


def advance_scan(state, x, y, gens):
    target = Config.TARGET_THRESHOLDS[state["threshold_index"]]
    adaptive = float(torch.quantile(state["losses"], Config.SURVIVAL_QUANTILE, dim=1).max())
    threshold = min(state["threshold"], max(target, adaptive))
    reached = threshold <= target + 1e-12
    if (math.isfinite(state["threshold"]) and not reached
            and threshold >= state["threshold"] - Config.MIN_LEVEL_DECREMENT):
        return None
    before = state["log_volume"].clone()
    survival = resample(state, threshold, x, y, gens["resample"])
    rates = rejuvenate(state, threshold, x, y, gens["mutation"])
    state["level"] += 1
    state["threshold"] = threshold
    expected = before + torch.tensor(survival, device=x.device, dtype=torch.float64).log()
    if not torch.equal(expected, state["log_volume"]):
        raise RuntimeError("存活率乘积与log体积累计不一致。")
    if not bool((state["losses"] <= threshold + Config.LOSS_TOLERANCE).all()):
        raise RuntimeError("粒子越出当前loss约束。")
    return {"reached": reached, "survival": survival, "rates": rates}


def record_threshold(state, target, x, y, rates, elapsed):
    threshold = Config.TARGET_THRESHOLDS[state["threshold_index"]]
    ids = hard_ids(state["particles"], x)
    lv = state["log_volume"].cpu().numpy()
    exact = threshold + Config.LOSS_TOLERANCE < math.log(2) / 16
    if exact and not np.all(ids == target["function_id"]):
        raise RuntimeError("hard-exact门槛下出现错误答案。")
    # 每层保存少量真实权重供离线复核；这些相关粒子不是独立样本。
    stride = max(1, Config.PARTICLES_PER_REPLICA // Config.AUDIT_PARTICLES_PER_REPLICA)
    small = state["particles"][:, ::stride][:, :Config.AUDIT_PARTICLES_PER_REPLICA].detach().cpu().clone()
    check32 = state["losses"][:, ::stride][:, :Config.AUDIT_PARTICLES_PER_REPLICA].detach().cpu()
    fresh32 = losses(small, x.cpu(), y.cpu())
    check64 = losses(small.double(), x.cpu().double(), y.cpu().double())
    err = (check64 - check32.double()).abs()
    mismatch = bool((err > Config.NUMERIC_ABSOLUTE_TOLERANCE
                     + Config.NUMERIC_RELATIVE_TOLERANCE * check64.abs()).any())
    outside64 = bool((check64 > threshold + Config.LOSS_TOLERANCE
                      + Config.NUMERIC_ABSOLUTE_TOLERANCE).any())
    row = {"threshold": threshold, "level": state["level"], "elapsed_seconds": elapsed,
           **statistics(lv), "target_probability": float(np.mean(ids == target["function_id"])),
           "hard_exact_guaranteed": exact, "loss_max": float(state["losses"].max()),
           "numeric_max_abs_error": float(err.max()), "numeric_mismatch": mismatch or outside64,
           "numeric_cached_vs_fresh_error": float((fresh32 - check32).abs().max()),
           "accept_hidden": rates[0], "accept_output": rates[1], "accept_all": rates[2],
           "scale_min": min(state["scales"])}
    row["warnings"] = row_warnings(row)
    replicas = []
    for r in range(Config.REPLICAS):
        p = float(np.mean(ids[r] == target["function_id"]))
        replicas.append({"threshold": threshold, "replica": r, "log_volume": float(lv[r]),
                         "neural_k_bits": float(-lv[r] / math.log(2)), "target_probability": p,
                         "same_hard_log_volume": float(lv[r] + math.log(p)) if p else None,
                         "loss_mean": float(state["losses"][r].mean()),
                         "loss_max": float(state["losses"][r].max()),
                         "coordinate_norm2_mean": float(state["particles"][r].square().sum(-1).mean()),
                         "unique_lineages": int(torch.unique(state["lineages"][r]).numel())})
    state["audit_samples"] = state["audit_samples"] + [{"threshold": threshold, "weights": small,
                                                         "loss_cached": check32, "loss_float64": check64}]
    return row, replicas


def result_from_state(state, target, phase, signature, status, error=None):
    flags = sorted({f for row in state["threshold_rows"] for f in row["warnings"]})
    return {"signature": signature, "protocol": PROTOCOL, "phase": phase,
            "function_id": target["function_id"], "function_hex": target["function_hex"],
            "status": status, "completed_thresholds": state["threshold_index"],
            "level": state["level"], "current_threshold": state["threshold"],
            "elapsed_seconds": state["elapsed_seconds"], "algorithm_seconds": state["algorithm_seconds"],
            "discarded_attempt_seconds": state["discarded_attempt_seconds"],
            "threshold_rows": state["threshold_rows"], "warning_reasons": flags,
            "quality": "INCOMPLETE" if status != "completed" else "WARN" if flags else "NO_INTERNAL_ALARM",
            "error": error}


def save_job(folder, state, target, phase, gens, signature, status, error=None):
    payload = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}
    payload.update({"signature": signature, "function_id": target["function_id"], "phase": phase,
                    "rng": {k: g.get_state().cpu() for k, g in gens.items()}})
    atomic_torch(folder / "checkpoint.pt", payload)
    write_csv(folder / "volume_curve.csv", state["threshold_rows"])
    write_csv(folder / "replica_volume_curve.csv", state["replica_rows"])
    write_csv(folder / "levels.csv", state["level_rows"])
    result = result_from_state(state, target, phase, signature, status, error)
    write_json(folder / "summary.json", result)
    if status == "completed":
        atomic_torch(folder / "audit_samples.pt", {"function_id": target["function_id"],
                     "signature": signature, "samples": state["audit_samples"]})
        names = ("summary.json", "volume_curve.csv", "replica_volume_curve.csv", "levels.csv", "audit_samples.pt")
        write_json(folder / "completed.json", {"signature": signature,
                   "files": {n: file_digest(folder / n) for n in names}})
        # 完成标记最后写入。之后删checkpoint仅为节省约25GB空间，审计权重仍保留。
        (folder / "checkpoint.pt").unlink(missing_ok=True)
    return result


def read_completed(folder, signature):
    marker = folder / "completed.json"
    if not marker.exists():
        return None
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest["signature"] != signature:
        raise RuntimeError("已完成任务的协议不一致。")
    for name, wanted in manifest["files"].items():
        path = folder / name
        if not path.is_file() or file_digest(path) != wanted:
            raise RuntimeError(f"已完成结果缺失或损坏：{path.name}；不会静默跳过或覆盖。")
    result = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if result["status"] != "completed" or result["completed_thresholds"] != len(Config.TARGET_THRESHOLDS):
        raise RuntimeError("完成标记与结果不一致。")
    return result


def load_job(path, target, phase, gens, signature, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.pop("signature") != signature or state.pop("function_id") != target["function_id"]
            or state.pop("phase") != phase):
        raise RuntimeError("checkpoint协议/函数/复测阶段不一致。")
    restore_rng(gens, state.pop("rng"))
    for key in ("particles", "losses", "lineages", "log_volume"):
        state[key] = state[key].to(device)
    return state


def run_job(root, target, phase, device, x, signature, deadline=float("inf")):
    folder = root / "targets" / target["function_hex"] / phase
    folder.mkdir(parents=True, exist_ok=True)
    old = read_completed(folder, signature)
    if old is not None:
        return old
    with job_settings(phase):
        tag = target["function_id"] + (65536 if phase == "recheck" else 0)
        gens = generators(device, tag)
        y = torch.tensor(bits_of(target["function_id"]), device=device)
        checkpoint = folder / "checkpoint.pt"
        state = load_job(checkpoint, target, phase, gens, signature, device) if checkpoint.exists() else new_state(device, x, y, gens)
        started, base_elapsed = time.perf_counter(), state["elapsed_seconds"]
        status, error = "running", None
        log(f"START {target['function_hex']} {phase} | level={state['level']} | particles={Config.REPLICAS}x{Config.PARTICLES_PER_REPLICA}")
        try:
            while state["threshold_index"] < len(Config.TARGET_THRESHOLDS):
                elapsed = base_elapsed + time.perf_counter() - started
                if time.perf_counter() >= deadline:
                    status = "budget_paused"
                    break
                factor = Config.VERIFY_PARTICLE_MULTIPLIER * Config.VERIFY_SWEEP_MULTIPLIER if phase == "recheck" else 1
                if elapsed >= Config.MAX_SECONDS_PER_TARGET * factor:
                    status = "target_budget_paused"
                    break
                if state["level"] >= Config.MAX_LEVELS_PER_TARGET:
                    status = "level_budget_paused"
                    break
                safe, rng = boundary_snapshot(state, gens)
                begin = time.perf_counter()
                try:
                    outcome = advance_scan(state, x, y, gens)
                    if outcome is None:
                        status = "stalled_threshold"
                        break
                    sync(device)
                    state["algorithm_seconds"] += time.perf_counter() - begin
                    elapsed = base_elapsed + time.perf_counter() - started
                    if outcome["reached"]:
                        row, reps = record_threshold(state, target, x, y, outcome["rates"], elapsed)
                        state["threshold_rows"].append(row)
                        state["replica_rows"].extend(reps)
                        state["threshold_index"] += 1
                    state["level_rows"].append({"level": state["level"], "threshold": state["threshold"],
                        "elapsed_seconds": elapsed, "survival_by_replica": json.dumps(outcome["survival"]),
                        "log_volume_by_replica": json.dumps(state["log_volume"].cpu().tolist()),
                        "accept_hidden": outcome["rates"][0], "accept_output": outcome["rates"][1],
                        "accept_all": outcome["rates"][2], "scales": json.dumps(state["scales"])})
                except BaseException:
                    state = safe
                    state["discarded_attempt_seconds"] += time.perf_counter() - begin
                    restore_rng(gens, rng)
                    raise
                if outcome["reached"] or state["level"] % Config.CHECKPOINT_EVERY_LEVELS == 0:
                    state["elapsed_seconds"] = base_elapsed + time.perf_counter() - started
                    save_job(folder, state, target, phase, gens, signature, "running")
                    write_json(root / "progress.json", {"status": "running", "phase": phase,
                        "function_hex": target["function_hex"], "level": state["level"],
                        "threshold": state["threshold"], "completed_thresholds": state["threshold_index"],
                        "target_elapsed_seconds": state["elapsed_seconds"], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                if state["level"] % Config.LOG_EVERY_LEVELS == 0:
                    log(f"{target['function_hex']} {phase} level={state['level']} BCE<={state['threshold']:.6g} elapsed={duration(elapsed)}")
            if status == "running":
                status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            log("收到中断：回滚到上一完整SMC层并保存。")
        except Exception as exc:
            status, error = "failed", str(exc)
            (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"FAILED {target['function_hex']}: {error}")
        finally:
            sync(device)
            state["elapsed_seconds"] = base_elapsed + time.perf_counter() - started
            result = save_job(folder, state, target, phase, gens, signature, status, error)
        last = state["threshold_rows"][-1] if state["threshold_rows"] else None
        detail = f"K={last['neural_k_bits_median']:.2f} sd={last['neural_k_bits_sd']:.2f} bits" if last else "尚无完整门槛"
        log(f"RESULT {target['function_hex']} {phase} | {status} | {detail} | {result['quality']} | {duration(state['elapsed_seconds'])}")
        return result


def numeric_self_check(device, x):
    gen = torch.Generator(device=device).manual_seed(197)
    theta = torch.randn(32, parameter_count(), device=device, generator=gen)
    got = forward_logits(theta, x)
    expected = []
    w = Config.WIDTH
    for p in theta:
        h = torch.tanh(x @ (p[:4*w].reshape(w, 4) / 2).T + p[4*w:5*w] / 2)
        expected.append(h @ (p[5*w:6*w] / math.sqrt(w)) + p[6*w] / math.sqrt(w))
    torch.testing.assert_close(got, torch.stack(expected), atol=2e-6, rtol=2e-5)
    perm = [2, 0, 3, 1]
    transformed = theta.clone()
    transformed[:, :4*w] = theta[:, :4*w].reshape(32, w, 4)[:, :, np.argsort(perm)].reshape(32, 4*w)
    torch.testing.assert_close(forward_logits(theta, x[:, perm]), forward_logits(transformed, x), atol=2e-6, rtol=2e-5)
    transformed = theta.clone()
    transformed[:, 5*w:] *= -1
    torch.testing.assert_close(forward_logits(transformed, x), -got, atol=2e-6, rtol=2e-5)
    theta64 = theta.cpu().double()
    ref64 = forward_logits(theta64, x.cpu().double())
    torch.testing.assert_close(got.cpu().double(), ref64, atol=2e-6, rtol=2e-5)
    return {"status": "PASS", "checks": ["layerwise_forward", "input_permutation",
            "output_complement", "CPU_float64_reference"],
            "max_logit_abs_error": float((got.cpu().double() - ref64).abs().max())}


def run_prior(root, plan, device, x, signature):
    done = root / "prior_completed.json"
    if done.exists():
        meta = json.loads(done.read_text(encoding="utf-8"))
        if (meta["signature"] != signature or file_digest(root / "prior_counts.npz") != meta["data_sha256"]
                or file_digest(root / "prior_summary.json") != meta["summary_sha256"]):
            raise RuntimeError("先验统计文件不完整或协议不一致。")
        return json.loads((root / "prior_summary.json").read_text(encoding="utf-8"))
    path = root / "prior_checkpoint.pt"
    gen = torch.Generator(device=device).manual_seed(Config.PRIOR_SEED + 91)
    thresholds = [e for e in Config.TARGET_THRESHOLDS if e >= .5]
    fids = plan["preflight_ids"]
    labels = torch.tensor(np.stack([bits_of(fid) for fid in fids]), device=device)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise RuntimeError("先验checkpoint协议不一致。")
        gen.set_state(state["rng"])
    else:
        state = {"signature": signature, "done": 0, "counts": np.zeros(65536, dtype=np.int64),
                 "hits": np.zeros((len(fids), len(thresholds)), dtype=np.int64),
                 "calibration_done": 0, "elapsed_seconds": 0., "rng": gen.get_state().cpu()}
    started, base = time.perf_counter(), state["elapsed_seconds"]
    next_save = state["done"] + Config.PRIOR_CHECKPOINT_SAMPLES
    try:
        while state["done"] < Config.PRIOR_SAMPLES:
            n = min(Config.EVAL_MICRO_BATCH, Config.PRIOR_SAMPLES - state["done"])
            theta = torch.randn(n, parameter_count(), device=device, generator=gen)
            with torch.no_grad():
                z = forward_logits(theta, x)
                ids = ((z >= 0).long() * (1 << torch.arange(16, device=device))).sum(1).cpu().numpy()
                count_delta = np.bincount(ids, minlength=65536)
                m = min(n, max(0, Config.CALIBRATION_SAMPLES - state["calibration_done"]))
                hits = state["hits"].copy()
                if m:
                    for j, y in enumerate(labels):
                        bce = F.softplus((1 - 2*y) * z[:m]).mean(1)
                        hits[j] += np.array([int((bce <= e + Config.LOSS_TOLERANCE).sum()) for e in thresholds])
            # 整批作为一个事务提交；中断时只保存最后一个完整批次及对应随机状态。
            state = {**state, "done": state["done"] + n, "counts": state["counts"] + count_delta,
                     "hits": hits, "calibration_done": state["calibration_done"] + m,
                     "rng": gen.get_state().cpu(), "elapsed_seconds": base + time.perf_counter() - started}
            if state["done"] >= next_save:
                atomic_torch(path, state)
                log(f"PRIOR {state['done']:,}/{Config.PRIOR_SAMPLES:,} | {duration(state['elapsed_seconds'])}")
                next_save = state["done"] + Config.PRIOR_CHECKPOINT_SAMPLES
    finally:
        atomic_torch(path, state)
    prior = {"samples": state["done"], "functions_seen": int(np.count_nonzero(state["counts"])),
             "calibration_samples": state["calibration_done"], "calibration_function_ids": fids,
             "calibration_thresholds": thresholds, "calibration_hits": state["hits"].tolist(),
             "elapsed_seconds": state["elapsed_seconds"],
             "warning": "未命中函数仅表示低于本次分辨率，不能视为零概率。"}
    temp = root / "prior_counts.npz.tmp"
    with temp.open("wb") as f:
        np.savez_compressed(f, counts=state["counts"], samples=state["done"])
    os.replace(temp, root / "prior_counts.npz")
    write_json(root / "prior_summary.json", prior)
    write_json(done, {"signature": signature, "data_sha256": file_digest(root / "prior_counts.npz"),
                     "summary_sha256": file_digest(root / "prior_summary.json")})
    path.unlink(missing_ok=True)
    return prior


def calibration_report(root, prior, results):
    rows = []
    n = prior["calibration_samples"]
    for i, fid in enumerate(prior["calibration_function_ids"]):
        result = results.get((fid, "first"))
        if result is None:
            continue
        source = root / "targets" / f"0x{fid:04X}" / "first" / "replica_volume_curve.csv"
        if not source.exists():
            continue
        with source.open(encoding="utf-8-sig", newline="") as f:
            reps = list(csv.DictReader(f))
        for j, threshold in enumerate(prior["calibration_thresholds"]):
            hits = prior["calibration_hits"][i][j]
            volumes = np.exp([float(r["log_volume"]) for r in reps if math.isclose(float(r["threshold"]), threshold)])
            if len(volumes) != Config.REPLICAS:
                continue
            p, estimate = hits / n, float(volumes.mean())
            se = math.sqrt(p * (1-p) / n + float(volumes.var(ddof=1)) / len(volumes))
            eligible = hits >= Config.CALIBRATION_MIN_HITS and n-hits >= Config.CALIBRATION_MIN_HITS
            tolerance = max(Config.CALIBRATION_SIGMAS * se, Config.CALIBRATION_RELATIVE_FLOOR * p)
            flag = bool(eligible and abs(estimate - p) > tolerance)
            rows.append({"function_hex": f"0x{fid:04X}", "threshold": threshold,
                         "prior_hits": hits, "prior_samples": n, "direct_probability": p,
                         "smc_mean_volume": estimate, "combined_se_diagnostic": se,
                         "tolerance": tolerance, "eligible": eligible, "failed": flag})
    checked = sum(r["eligible"] for r in rows)
    failed = sum(r["failed"] for r in rows)
    return {"status": "FAIL" if failed else "PASS" if checked else "INSUFFICIENT_DIRECT_HITS",
            "checked_points": checked, "failed_points": failed, "rows": rows,
            "meaning": "浅层直接先验校准，不认证未命中的稀有函数或深层混合。"}


def preflight_gate(plan, results, calibration):
    checks = [results.get((fid, "first")) for fid in plan["preflight_ids"]]
    incomplete = sum(r is None or r["status"] != "completed" for r in checks)
    warned = sum(r is not None and r["quality"] == "WARN" for r in checks)
    reasons = []
    if calibration["status"] != "PASS":
        reasons.append("SHALLOW_CALIBRATION_NOT_PASSED")
    if incomplete:
        reasons.append("PREFLIGHT_INCOMPLETE")
    if checks and warned / len(checks) > Config.PREFLIGHT_MAX_WARN_FRACTION:
        reasons.append("WIDESPREAD_REPLICA_WARNINGS")
    if any(r and "FLOAT32_FLOAT64_MISMATCH" in r["warning_reasons"] for r in checks):
        reasons.append("NUMERIC_CHECK_FAILED")
    return {"status": "PAUSED_FOR_REVIEW" if reasons else "PROCEED_WITH_LOCAL_WARNINGS",
            "reasons": reasons, "tested": len(checks), "warned": warned, "incomplete": incomplete}


def compare_recheck(first, second):
    rows = []
    by_e = {r["threshold"]: r for r in second["threshold_rows"]}
    for a in first["threshold_rows"]:
        b = by_e.get(a["threshold"])
        if b is None:
            continue
        # 此阈值只是复测筛查容差，不是假定SMC副本独立后的严格置信区间。
        tol = max(Config.RECHECK_MIN_TOLERANCE_BITS, Config.RECHECK_SE_MULTIPLIER * math.sqrt(
            (a["neural_k_bits_sd"]**2 + b["neural_k_bits_sd"]**2) / Config.REPLICAS))
        delta = b["neural_k_bits_median"] - a["neural_k_bits_median"]
        delta_mean = b["neural_k_bits_mean_volume"] - a["neural_k_bits_mean_volume"]
        matched = abs(delta) <= tol and abs(delta_mean) <= tol
        rows.append({"threshold": a["threshold"], "delta_median_k_bits": delta,
                     "delta_mean_volume_k_bits": delta_mean, "screening_tolerance_bits": tol,
                     "within_tolerance": matched,
                     "both_without_internal_alarm": not a["warnings"] and not b["warnings"],
                     "status": "CONSISTENT_RECHECK" if matched and not a["warnings"] and not b["warnings"]
                     else "UNRESOLVED_WARNING"})
    all_done = first["status"] == second["status"] == "completed"
    consistent = all_done and len(rows) == len(Config.TARGET_THRESHOLDS) and all(r["status"] == "CONSISTENT_RECHECK" for r in rows)
    return {"status": "CONSISTENT_RECHECK" if consistent else "UNRESOLVED_WARNING", "rows": rows}


def verify_queue(plan, results):
    targets = {t["function_id"]: t for t in plan["targets"]}
    # 抽查名单事先冻结，包含未报警规则，不能只挑容易测的成功例。
    forced = [fid for fid in plan["audit_ids"] if fid in targets]
    flagged = [fid for fid in targets if fid not in forced and results.get((fid, "first"), {}).get("quality") != "NO_INTERNAL_ALARM"]
    def severity(fid):
        rows = results.get((fid, "first"), {}).get("threshold_rows", [])
        return max([r["replica_range_bits"] for r in rows] or [float("inf")])
    flagged.sort(key=lambda fid: (-severity(fid), fid))
    # 先完成冻结抽查面板，再按警告强度处理；保留未复测队列，不隐去预算截断。
    return [targets[fid] for fid in forced + flagged]


def quality_for(fid, plan, results):
    first, second = results.get((fid, "first")), results.get((fid, "recheck"))
    if first is None or first["status"] != "completed":
        return "FIRST_PASS_INCOMPLETE"
    if second is not None:
        return compare_recheck(first, second)["status"]
    if first["quality"] == "WARN":
        return "WARN_RECHECK_PENDING"
    if fid in plan["audit_ids"]:
        return "AUDIT_PENDING"
    return "NO_INTERNAL_ALARM_NOT_RECHECKED"


def refresh_summary(root, plan, results, environment, status, full_export=False):
    rows = []
    for t in plan["targets"]:
        fid = t["function_id"]
        a, b = results.get((fid, "first"), {}), results.get((fid, "recheck"), {})
        tail = a.get("threshold_rows", [])
        rows.append({"function_hex": t["function_hex"], "orbit_size": t["orbit_size"],
                     "first_status": a.get("status", "not_started"), "quality": quality_for(fid, plan, results),
                     "completed_thresholds": a.get("completed_thresholds", 0),
                     "last_threshold": tail[-1]["threshold"] if tail else None,
                     "last_k_bits": tail[-1]["neural_k_bits_median"] if tail else None,
                     "first_seconds": a.get("elapsed_seconds", 0),
                     "recheck_status": b.get("status", "not_started"), "recheck_seconds": b.get("elapsed_seconds", 0)})
    counts = {q: sum(r["quality"] == q for r in rows) for q in sorted({r["quality"] for r in rows})}
    completed = sum(r["first_status"] == "completed" for r in rows)
    summary = {"protocol": PROTOCOL, "status": status, "environment": environment,
               "planned_orbits": len(rows), "full_population_orbits": 1992,
               "completed_first_pass": completed, "quality_counts": counts,
               "recheck_seconds_spent": sum(r["recheck_seconds"] for r in rows),
               "recheck_budget_seconds": Config.VERIFY_BUDGET_SECONDS,
               "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "global_mixing_certified": False,
               "warning": "无内部报警不等于无偏；复测只是一致性检验。副本共享自适应门槛，误差量不作严格置信区间。"}
    write_json(root / "summary.json", summary)
    write_csv(root / "orbit_status.csv", rows)
    if full_export:
        curve_rows, comparisons = [], []
        for t in plan["targets"]:
            fid = t["function_id"]
            for phase in ("first", "recheck"):
                r = results.get((fid, phase))
                if r:
                    curve_rows.extend({"function_hex": t["function_hex"], "phase": phase,
                                       "job_status": r["status"], **row} for row in r["threshold_rows"])
            if (fid, "first") in results and (fid, "recheck") in results:
                comparisons.extend({"function_hex": t["function_hex"], **r} for r in compare_recheck(
                    results[(fid, "first")], results[(fid, "recheck")])["rows"])
        write_csv(root / "all_volume_curves.csv", curve_rows)
        write_csv(root / "recheck_comparisons.csv", comparisons)
        pending = [{"function_hex": t["function_hex"], "quality": quality_for(t["function_id"], plan, results)}
                   for t in verify_queue(plan, results)
                   if results.get((t["function_id"], "recheck"), {}).get("status") != "completed"]
        write_json(root / "pending_rechecks.json", pending)
    return summary


def archive_scan(root):
    path = root.with_name(root.name + "_package.zip")
    temp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix not in (".pt", ".tmp", ".lock"):
                z.write(f, Path(root.name) / f.relative_to(root))
    os.replace(temp, path)
    return path


def check_disk(root):
    import shutil
    free = shutil.disk_usage(root).free
    if free < Config.MIN_FREE_DISK_GB * 1024**3:
        raise RuntimeError(f"磁盘剩余{free/1024**3:.2f}GB，低于保护门槛；停止并保留已有结果。")


def smoke_overrides():
    if not Config.SMOKE_TEST:
        return
    Config.RESULT_DIR = Path.cwd() / "results_4bit_full_curve_scan_smoke"
    Config.DEVICE = "cpu"
    Config.REPLICAS, Config.PARTICLES_PER_REPLICA = 2, 64
    Config.TARGET_THRESHOLDS = (.72, .70, .68)
    Config.ADAPT_SWEEPS, Config.MUTATION_SWEEPS = 1, 2
    Config.STRATUM_SAMPLE_COUNTS = (1, 1, 1, 1, 1)
    Config.AUDIT_RANDOM_ORBITS = 1
    Config.PRIOR_SAMPLES = 2048
    Config.CALIBRATION_SAMPLES = 2048
    Config.PRIOR_CHECKPOINT_SAMPLES = 512
    Config.EVAL_MICRO_BATCH = 512
    Config.CALIBRATION_MIN_HITS = 5
    Config.MAX_SECONDS_PER_TARGET = 30
    Config.VERIFY_BUDGET_SECONDS = 120
    Config.MAX_LEVELS_PER_TARGET = 100
    Config.MIN_FREE_DISK_GB = .01
    Config.CHECKPOINT_EVERY_LEVELS = 5


def main():
    global LOG_PATH
    smoke_overrides()
    if Config.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("正式运行需要CUDA；本地检查请设置NSP_SMOKE_TEST=1。")
    if (not Config.TARGET_THRESHOLDS or min(Config.TARGET_THRESHOLDS) <= 0
            or any(a <= b for a, b in zip(Config.TARGET_THRESHOLDS, Config.TARGET_THRESHOLDS[1:]))):
        raise ValueError("loss门槛必须为正且严格递减。")
    if Config.REPLICAS < 2 or Config.PARTICLES_PER_REPLICA < 2 or Config.MUTATION_SWEEPS < 1:
        raise ValueError("副本/粒子/更新轮数无效。")
    if not 0 < Config.CALIBRATION_SAMPLES <= Config.PRIOR_SAMPLES:
        raise ValueError("先验校准样本数无效。")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(Config.DEVICE)
    environment = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
                   "parameter_count": parameter_count(), "TF32": False}
    orbits, representative = build_orbits()
    plan = make_plan(orbits, representative)
    config = {"config": protocol_config(), "plan": plan, "environment": environment}
    signature = digest(config)
    root = Config.RESULT_DIR.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        existing = [p for p in root.iterdir() if p.name != ".run.lock"]
        path = root / "config.json"
        if existing:
            if not Config.RESUME or not path.exists() or json.loads(path.read_text(encoding="utf-8"))["signature"] != signature:
                raise RuntimeError("结果目录非空且协议不兼容；请使用新RESULT_DIR，不会覆盖。")
        LOG_PATH = root / "run.log"
        write_json(path, {"signature": signature, **config})
        write_csv(root / "all_1992_orbits.csv", orbits)
        mapping_path = root / "function_to_orbit.npz"
        if not mapping_path.exists():
            with mapping_path.open("wb") as f:
                np.savez_compressed(f, representative=representative, inputs=bit_inputs())
        check_disk(root)
        x = torch.tensor(bit_inputs(), device=device, dtype=torch.float32)
        write_json(root / "numeric_self_check.json", numeric_self_check(device, x))
        log(f"=== 全4-bit独立SMC === {environment['gpu']} | 0/1 | 4->{Config.WIDTH}->1 tanh | {len(plan['targets'])}轨道 | BCE到{Config.TARGET_THRESHOLDS[-1]}")
        log(f"结果目录：{root}；首轮和复测分开保存，未报警不代表已认证全局混合。")
        results = {}
        for target in plan["targets"]:
            for phase in ("first", "recheck"):
                folder = root / "targets" / target["function_hex"] / phase
                r = read_completed(folder, signature)
                if r is None and (folder / "summary.json").exists():
                    r = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                    if r["signature"] != signature:
                        raise RuntimeError("部分结果协议不一致。")
                    if r["status"] == "completed":
                        if not (folder / "checkpoint.pt").exists():
                            raise RuntimeError("完成事务缺少标记和checkpoint，不能静默跳过。")
                        r["status"] = "completion_commit_pending"
                if r:
                    results[(target["function_id"], phase)] = r
        status, started = "running", time.perf_counter()
        prior, consecutive_failed = None, 0
        try:
            prior = run_prior(root, plan, device, x, signature)
            targets = plan["targets"]
            for index, target in enumerate(targets):
                # 前17条先检验整体流程；通过后才进入一天规模的首轮。
                if index == len(plan["preflight_ids"]):
                    calibration = calibration_report(root, prior, results)
                    gate = preflight_gate(plan, results, calibration)
                    write_json(root / "calibration.json", calibration)
                    write_json(root / "preflight_gate.json", gate)
                    log(f"PREFLIGHT {gate['status']} | warnings={gate['warned']}/{gate['tested']} | calibration={calibration['status']}")
                    if gate["reasons"]:
                        status = "preflight_paused_for_review"
                        break
                check_disk(root)
                key = (target["function_id"], "first")
                if results.get(key, {}).get("status") != "completed":
                    results[key] = run_job(root, target, "first", device, x, signature)
                r = results[key]
                consecutive_failed = consecutive_failed + 1 if r["status"] != "completed" else 0
                summary = refresh_summary(root, plan, results, environment, "first_pass", full_export=(index+1) % 25 == 0)
                log(f"SCAN {summary['completed_first_pass']}/{len(targets)} | {target['function_hex']} | quality={r['quality']}")
                if r["status"] == "interrupted":
                    status = "interrupted"
                    break
                if (consecutive_failed >= Config.MAX_CONSECUTIVE_FAILURES
                        or "out of memory" in (r.get("error") or "").lower()
                        or "FLOAT32_FLOAT64_MISMATCH" in r["warning_reasons"]):
                    status = "systemic_failure_paused"
                    break
                if Config.PACKAGE_RESULTS and (index+1) % Config.ARCHIVE_EVERY_TARGETS == 0:
                    refresh_summary(root, plan, results, environment, "first_pass", full_export=True)
                    archive_scan(root)
            if status == "running":
                calibration = calibration_report(root, prior, results)
                gate = preflight_gate(plan, results, calibration)
                write_json(root / "calibration.json", calibration)
                write_json(root / "preflight_gate.json", gate)
                if gate["reasons"]:
                    status = "preflight_paused_for_review"
            if status == "running":
                queue = verify_queue(plan, results)
                write_json(root / "recheck_plan.json", {"targets": [t["function_hex"] for t in queue],
                           "forced_audit_ids": plan["audit_ids"], "particle_multiplier": Config.VERIFY_PARTICLE_MULTIPLIER,
                           "sweep_multiplier": Config.VERIFY_SWEEP_MULTIPLIER})
                spent = sum(r["elapsed_seconds"] for (fid, phase), r in results.items() if phase == "recheck")
                deadline = time.perf_counter() + max(0, Config.VERIFY_BUDGET_SECONDS - spent)
                consecutive_failed = 0
                for target in queue:
                    key = (target["function_id"], "recheck")
                    if results.get(key, {}).get("status") == "completed":
                        continue
                    if time.perf_counter() >= deadline:
                        status = "first_pass_finished_recheck_budget_exhausted"
                        break
                    check_disk(root)
                    results[key] = run_job(root, target, "recheck", device, x, signature, deadline)
                    refresh_summary(root, plan, results, environment, "rechecking", full_export=True)
                    if results[key]["status"] in ("interrupted", "budget_paused"):
                        status = results[key]["status"]
                        break
                    r = results[key]
                    consecutive_failed = consecutive_failed + 1 if r["status"] != "completed" else 0
                    if (consecutive_failed >= Config.MAX_CONSECUTIVE_FAILURES
                            or "out of memory" in (r.get("error") or "").lower()
                            or "FLOAT32_FLOAT64_MISMATCH" in r["warning_reasons"]):
                        status = "recheck_systemic_failure_paused"
                        break
                if status == "running":
                    status = "finished_with_quality_labels"
        except KeyboardInterrupt:
            status = "interrupted"
            log("中断，已提交的曲线和先验批次可恢复。")
        except Exception as exc:
            status = "failed_paused"
            (root / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"停止：{exc}")
        finally:
            summary = refresh_summary(root, plan, results, environment, status, full_export=True)
            summary["elapsed_seconds_this_invocation"] = time.perf_counter() - started
            write_json(root / "summary.json", summary)
            write_json(root / "progress.json", summary)
            log(f"FINAL {status} | {summary['completed_first_pass']}/{len(plan['targets'])} | {summary['quality_counts']}")
            if Config.PACKAGE_RESULTS:
                log(f"结果包：{archive_scan(root)}")
        return summary


if __name__ == "__main__":
    main()
