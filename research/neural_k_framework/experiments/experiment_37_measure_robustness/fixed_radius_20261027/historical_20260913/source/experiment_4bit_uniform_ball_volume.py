"""固定范数球内的四位规则欧式体积；保留旧网络，不使用高斯密度权重。

与 experiment_4bit_full_curve_scan.py 放在同一目录。运行方法见配套 README。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import experiment_4bit_full_curve_scan as legacy


PROTOCOL = "4bit_uniform_ball_euclidean_volume_v1"
LN2 = math.log(2)
DEFAULT_NODES = (.7, .5, .3, .2, .12, .08, .06, .04, .025, .016,
                 .01, .0063, .004, .0025, .0016, .001)
NAMES = {0: "constant", 255: "copy_complement", 4080: "parity2", 27030: "parity4"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_retry(source, target):
    for attempt in range(10):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_retry(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    replace_retry(temp, path)


def atomic_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    replace_retry(temp, path)


def log(message, root=None):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + message
    print(line, flush=True)
    if root is not None:
        with (root / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def sample_prior(shape, radius, measure, device, dtype, rng):
    theta = torch.randn(shape, device=device, dtype=dtype, generator=rng)
    if measure == "uniform_ball":
        direction = theta / torch.linalg.vector_norm(theta, dim=-1, keepdim=True)
        # 半径不能均匀取样；r/R 的 d 次方才服从均匀分布。
        u = torch.rand(shape[:-1] + (1,), device=device, dtype=dtype, generator=rng)
        return direction * (radius * u.pow(1 / shape[-1]))
    outside = theta.square().sum(-1) > radius**2
    attempts = 0
    while bool(outside.any()):
        count = int(outside.sum())
        theta[outside] = torch.randn((count, shape[-1]), device=device, dtype=dtype, generator=rng)
        outside = theta.square().sum(-1) > radius**2
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("截断高斯拒绝采样太难，请增大半径或只运行均匀球。")
    return theta


def rotate_block(theta, start, stop, angle, rng):
    current = theta[..., start:stop]
    norm = torch.linalg.vector_norm(current, dim=-1, keepdim=True)
    unit = current / norm.clamp_min(torch.finfo(theta.dtype).tiny)
    noise = torch.randn(current.shape, device=theta.device, dtype=theta.dtype, generator=rng)
    tangent = noise - (noise * unit).sum(-1, keepdim=True) * unit
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-30)
    # 给定半径的各向同性球面旋转是可逆的，保持均匀方向分布。
    alpha = angle[:, None, None]
    moved = norm * (alpha.cos() * unit + alpha.sin() * tangent)
    moved *= norm / torch.linalg.vector_norm(moved, dim=-1, keepdim=True).clamp_min(1e-30)
    proposed = theta.clone()
    proposed[..., start:stop] = moved
    return proposed


@torch.no_grad()
def mutation(theta, values, threshold, radius, score, rng, scales, sweeps, measure, kernel, blocks):
    rates = torch.zeros_like(scales)
    radial_rate = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
    for _ in range(sweeps):
        for j, (start, stop) in enumerate(blocks):
            if measure == "truncated_gaussian":
                proposed = theta.clone()
                rho = scales[:, j, None, None]
                noise = torch.randn(theta[..., start:stop].shape, device=theta.device,
                                    dtype=theta.dtype, generator=rng)
                proposed[..., start:stop] = (1 - rho.square()).sqrt() * theta[..., start:stop] + rho * noise
            elif kernel == "rotation":
                proposed = rotate_block(theta, start, stop, scales[:, j], rng)
            else:
                proposed = theta.clone()
                sigma = scales[:, j, None, None] * radius / math.sqrt(theta.shape[-1])
                noise = torch.randn(theta[..., start:stop].shape, device=theta.device,
                                    dtype=theta.dtype, generator=rng)
                proposed[..., start:stop] += sigma * noise
            proposed_values = score(proposed)
            accepted = (proposed_values <= threshold[:, None]) & (proposed.square().sum(-1) <= radius**2)
            theta = torch.where(accepted[..., None], proposed, theta)
            values = torch.where(accepted, proposed_values, values)
            rates[:, j] += accepted.to(theta.dtype).mean(1)
        if measure == "uniform_ball":
            old_radius = torch.linalg.vector_norm(theta, dim=-1, keepdim=True)
            u = torch.rand(old_radius.shape, device=theta.device, dtype=theta.dtype, generator=rng)
            new_radius = radius * u.pow(1 / theta.shape[-1])
            proposed = theta * (new_radius / old_radius.clamp_min(1e-30))
            proposed_values = score(proposed)
            accepted = (proposed_values <= threshold[:, None]) & (proposed.square().sum(-1) <= radius**2)
            theta = torch.where(accepted[..., None], proposed, theta)
            values = torch.where(accepted, proposed_values, values)
            radial_rate += accepted.to(theta.dtype).mean(1)
    return theta, values, rates / sweeps, radial_rate / sweeps


def new_state(args, radius, measure, score, dimension, seed):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    rngs = {key: generator(device, seed + offset) for key, offset in
            (("initial", 0), ("resample", 100003), ("mutation", 200003))}
    theta = sample_prior((args.replicas, args.particles, dimension), radius, measure,
                         device, dtype, rngs["initial"])
    state = {"theta": theta, "values": score(theta),
             "log_mass": torch.zeros(args.replicas, device=device, dtype=torch.float64),
             "lineages": torch.arange(args.particles, device=device).expand(args.replicas, -1).clone(),
             "scales": torch.tensor([.12, .20, .06], device=device, dtype=dtype).expand(args.replicas, -1).clone(),
             "threshold": torch.full((args.replicas,), float("inf"), device=device, dtype=dtype),
             "level": 0, "node_index": 0, "elapsed_seconds": 0., "curve": [], "replicas": [],
             "levels": [], "audit": [], "seed": seed}
    return state, rngs


@torch.no_grad()
def smc_step(state, target, args, radius, measure, score, rngs, blocks):
    quantile = torch.quantile(state["values"], args.survival, dim=1)
    threshold = torch.minimum(state["threshold"], quantile.clamp_min(target))
    keep = state["values"] <= threshold[:, None]
    fractions = keep.double().mean(1)
    if bool((fractions <= 0).any()):
        raise RuntimeError("零存活粒子，停止而不伪造体积。")
    index = torch.multinomial(keep.float(), args.particles, replacement=True, generator=rngs["resample"])
    state["theta"] = state["theta"].gather(1, index[..., None].expand_as(state["theta"]))
    state["values"] = state["values"].gather(1, index)
    state["lineages"] = state["lineages"].gather(1, index)
    state["log_mass"] += fractions.log()
    state["threshold"] = threshold
    theta, values, rates, radial = mutation(state["theta"], state["values"], threshold, radius, score,
                                          rngs["mutation"], state["scales"], args.sweeps, measure,
                                          args.kernel, blocks)
    state["theta"], state["values"] = theta, values
    state["level"] += 1
    # 只在完整的一层结束后调整下一层步幅；质量仍须由独立复测验证。
    state["scales"] = (state["scales"] * (.3 * (rates - .3)).exp()).clamp(1e-5, .85)
    if not bool(torch.isfinite(values).all()) or not bool((values <= threshold[:, None]).all()):
        raise RuntimeError("更新后违反 loss 约束或出现非有限值。")
    return rates, radial, fractions


def mass_statistics(log_mass):
    lv = log_mass.detach().cpu().double().numpy()
    normalized = np.exp(lv - lv.max())
    ks = -lv / LN2
    return {"K_bits": float(-(lv.max() + np.log(normalized.mean())) / LN2),
            "K_replica_mean_bits": float(ks.mean()), "K_sd_bits": float(ks.std(ddof=1)),
            "K_range_bits": float(np.ptp(ks)),
            "effective_replicas": float(normalized.sum()**2 / np.square(normalized).sum())}


def checkpoint(folder, state, rngs, signature, status):
    payload = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()}
    payload["rng"] = {key: rng.get_state() for key, rng in rngs.items()}
    payload["signature"] = signature
    atomic_torch(folder / "checkpoint.pt", payload)
    atomic_csv(folder / "volume_curve.csv", state["curve"])
    atomic_csv(folder / "replica_volume_curve.csv", state["replicas"])
    atomic_csv(folder / "levels.csv", state["levels"])
    atomic_torch(folder / "audit_samples.pt", state["audit"])
    summary = {"signature": signature, "status": status, "level": state["level"],
               "completed_nodes": state["node_index"], "elapsed_seconds": state["elapsed_seconds"],
               "last_thresholds": [float(t) if math.isfinite(float(t)) else None for t in state["threshold"]],
               "curve": state["curve"], "error": state.get("error")}
    atomic_json(folder / "summary.json", summary)
    return summary


def restore(folder, signature, device):
    state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    if state.pop("signature") != signature:
        raise RuntimeError("断点源码或科学配置不一致，请使用新的输出目录。")
    saved_rng = state.pop("rng")
    for key in ("theta", "values", "log_mass", "lineages", "scales", "threshold"):
        state[key] = state[key].to(device)
    rngs = {}
    for key, value in saved_rng.items():
        rngs[key] = generator(device, 0)
        rngs[key].set_state(value)
    return state, rngs


def make_score(fid, args):
    x = torch.tensor(legacy.bit_inputs(), device=args.device, dtype=getattr(torch, args.dtype))
    signs = torch.tensor(2 * legacy.bits_of(fid) - 1, device=args.device, dtype=x.dtype)

    def score(theta):
        flat = theta.reshape(-1, theta.shape[-1])
        pieces = []
        for chunk in flat.split(args.micro_batch):
            z = legacy.forward_logits(chunk, x)
            pieces.append(F.softplus(-signs * z).mean(-1))
        return torch.cat(pieces).reshape(theta.shape[:-1])

    return score, x, signs


@torch.no_grad()
def record_node(state, args, radius, measure, fid, x, signs, rates, radial):
    threshold = args.nodes[state["node_index"]]
    stats = mass_statistics(state["log_mass"])
    dimension = state["theta"].shape[-1]
    norm2 = state["theta"].double().square().sum(-1)
    stride = max(1, args.particles // 32)
    small = state["theta"][:, ::stride][:, :32].detach().cpu().clone()
    cached = state["values"][:, ::stride][:, :32].detach().cpu().double()
    z64 = legacy.forward_logits(small.double().reshape(-1, dimension), x.cpu().double())
    margins = z64 * signs.cpu().double()
    score64 = F.softplus(-margins).mean(-1).reshape(cached.shape)
    numeric_error = float((cached - score64).abs().max())
    log2_ball = (dimension / 2 * math.log(math.pi) + dimension * math.log(radius)
                 - math.lgamma(dimension / 2 + 1)) / LN2
    # 旧脚本的实际权重是 theta 经过固定对角线缩放；体积 Jacobian 为常数。
    log2_jacobian = -5 * args.width - (args.width + 1) / 2 * math.log2(args.width)
    warnings = []
    if stats["K_sd_bits"] > 2 or stats["K_range_bits"] > 6:
        warnings.append("REPLICA_SPREAD")
    if stats["effective_replicas"] < args.replicas / 2:
        warnings.append("REPLICA_MASS_IMBALANCE")
    if numeric_error > 2e-6 + 2e-4 * threshold or float(score64.max()) > threshold + 2e-6:
        warnings.append("NUMERIC_AUDIT")
    if float(norm2.max()) > radius**2 * (1 + 2e-6):
        raise RuntimeError("范数越出硬边界。")
    if threshold < LN2 / 16 and not bool((margins > 0).all()):
        raise RuntimeError("严格低 loss 节点出现 hard function 错误。")
    row = {"function_id": fid, "function_hex": f"0x{fid:04X}", "name": NAMES.get(fid, f"rule{fid}"),
           "measure": measure, "radius": radius, "threshold": threshold, "level": state["level"],
           "elapsed_seconds": state["elapsed_seconds"], **stats,
           "mean_norm2": float(norm2.mean()), "norm2_p05": float(torch.quantile(norm2.flatten(), .05)),
           "norm2_p95": float(torch.quantile(norm2.flatten(), .95)), "max_norm": float(norm2.max().sqrt()),
           "mean_loss": float(state["values"].mean()), "audit_mean_margin": float(margins.mean()),
           "audit_min_margin": float(margins.min()),
           "audit_mean_sample_margin_std": float(margins.std(-1).mean()),
           "numeric_max_abs_error": numeric_error, "accept_mean": float(rates.mean()),
           "radial_accept": float(radial.mean()), "warnings": ";".join(warnings)}
    if measure == "uniform_ball":
        row["log2_euclidean_volume_standard_coordinates"] = log2_ball - stats["K_bits"]
        row["log2_euclidean_volume_actual_weights"] = log2_ball + log2_jacobian - stats["K_bits"]
    state["curve"].append(row)
    for r in range(args.replicas):
        state["replicas"].append({"threshold": threshold, "replica": r,
                                  "log_mass": float(state["log_mass"][r]),
                                  "K_bits": float(-state["log_mass"][r] / LN2),
                                  "norm2_mean": float(norm2[r].mean()),
                                  "norm2_sd": float(norm2[r].std()),
                                  "unique_ancestors": int(torch.unique(state["lineages"][r]).numel())})
    state["audit"].append({"threshold": threshold, "theta": small, "loss_float64": score64,
                           "logits_float64": z64.reshape(args.replicas, -1, 16)})
    state["node_index"] += 1
    return row


def scientific_config(args):
    exclude = {"out", "device", "max_seconds", "max_levels", "checkpoint_every", "log_every", "self_test"}
    return {"protocol": PROTOCOL, "settings": {k: v for k, v in vars(args).items() if k not in exclude},
            "source_sha256": digest(__file__), "legacy_source_sha256": digest(legacy.__file__),
            "legacy_build": legacy.CODE_BUILD_SHA256,
            "norm_coordinates": "old_standardized_theta; actual_weights_have_constant_diagonal_scaling",
            "uniform_measure": "Lebesgue restricted to ||theta||_2 <= R, normalized by ball volume",
            "uniform_mutation": "reversible block sphere rotations plus independent prior-radius refresh; loss-only acceptance",
            "boundary_warning": "Fixed finite R may impose a positive minimum BCE. Budget exhaustion is NOT proof of infeasibility."}


def run_job(args, fid, radius, measure, root, signature):
    key = f"{measure}_R{radius:g}_0x{fid:04X}"
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "completed.json"
    if marker.exists():
        commit = json.loads(marker.read_text(encoding="utf-8"))
        if commit["signature"] != signature:
            raise RuntimeError("已完成结果配置不同，拒绝覆盖。")
        for name, expected in commit["files"].items():
            if digest(folder / name) != expected:
                raise RuntimeError("已完成结果校验失败。")
        log("SKIP " + key, root)
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    score, x, signs = make_score(fid, args)
    dimension = legacy.parameter_count()
    seed = args.seed + fid * 1009 + round(radius * 1000) + (0 if measure == "uniform_ball" else 30000001)
    if (folder / "checkpoint.pt").exists():
        state, rngs = restore(folder, signature, args.device)
    else:
        state, rngs = new_state(args, radius, measure, score, dimension, seed)
    blocks = [(0, 5 * args.width), (5 * args.width, dimension), (0, dimension)]
    started, previous = time.perf_counter(), state["elapsed_seconds"]
    status = "running"
    log(f"START {key} | level={state['level']} | {args.replicas}x{args.particles} | kernel={args.kernel}", root)
    try:
        while state["node_index"] < len(args.nodes):
            target = args.nodes[state["node_index"]]
            rates, radial, fractions = smc_step(state, target, args, radius, measure, score, rngs, blocks)
            state["elapsed_seconds"] = previous + time.perf_counter() - started
            state["levels"].append({"level": state["level"], "max_threshold": float(state["threshold"].max()),
                                     "mean_threshold": float(state["threshold"].mean()),
                                     "K_replica_mean_bits": float((-state["log_mass"] / LN2).mean()),
                                     "mean_survival": float(fractions.mean()), "mean_acceptance": float(rates.mean()),
                                     "min_scale": float(state["scales"].min()),
                                     "elapsed_seconds": state["elapsed_seconds"]})
            if bool((state["threshold"] <= target).all()):
                row = record_node(state, args, radius, measure, fid, x, signs, rates, radial)
                log(f"HIT {key} BCE<={target:g} K={row['K_bits']:.3f} sd={row['K_sd_bits']:.3f} "
                    f"norm2={row['mean_norm2']:.1f} | {row['warnings'] or 'NO_INTERNAL_ALARM'}", root)
                checkpoint(folder, state, rngs, signature, "running")
            elif state["level"] % args.checkpoint_every == 0:
                checkpoint(folder, state, rngs, signature, "running")
            if state["level"] % args.log_every == 0:
                log(f"{key} level={state['level']} max_BCE={float(state['threshold'].max()):.6g} "
                    f"elapsed={state['elapsed_seconds']:.1f}s", root)
            if state["node_index"] == len(args.nodes):
                status = "completed"
                break
            if time.perf_counter() - started >= args.max_seconds or state["level"] >= args.max_levels:
                status = "budget_exhausted_not_infeasibility"
                break
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "failed"
        state["error"] = repr(error)
        raise
    finally:
        state["elapsed_seconds"] = previous + time.perf_counter() - started
        summary = checkpoint(folder, state, rngs, signature, status)
        if status == "completed":
            names = ("summary.json", "volume_curve.csv", "replica_volume_curve.csv", "audit_samples.pt")
            atomic_json(marker, {"signature": signature, "files": {name: digest(folder / name) for name in names}})
        log(f"END {key} | {status} | {state['elapsed_seconds']:.1f}s", root)
    if status == "interrupted":
        raise KeyboardInterrupt
    return summary


@torch.no_grad()
def prior_calibration(args, root):
    path = root / "prior_calibration.json"
    if path.exists():
        return
    rows = []
    x = torch.tensor(legacy.bit_inputs(), device=args.device, dtype=getattr(torch, args.dtype))
    signs = torch.tensor(np.stack([2 * legacy.bits_of(f) - 1 for f in args.rules]), device=x.device, dtype=x.dtype)
    for measure in args.measures:
        for radius in args.radii:
            rng = generator(args.device, args.seed + 70000000 + round(radius * 1000))
            counts = np.zeros((len(args.rules), len(args.nodes)), dtype=np.int64)
            for start in range(0, args.prior_samples, args.micro_batch):
                theta = sample_prior((min(args.micro_batch, args.prior_samples-start), legacy.parameter_count()),
                                     radius, measure, x.device, x.dtype, rng)
                z = legacy.forward_logits(theta, x)
                for j in range(len(args.rules)):
                    values = F.softplus(-signs[j] * z).mean(-1)
                    counts[j] += np.array([int((values <= node).sum()) for node in args.nodes])
            for j, fid in enumerate(args.rules):
                for k, node in enumerate(args.nodes):
                    rows.append({"measure": measure, "radius": radius, "function_id": fid,
                                 "threshold": node, "samples": args.prior_samples, "hits": int(counts[j, k])})
    atomic_json(path, rows)


@torch.no_grad()
def self_test(args, root):
    from scipy.special import betainc
    from scipy.stats import kstest

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    dimension, radius, size = 9, 3., 32768
    rng = generator(device, 3917)
    theta = sample_prior((1, size, dimension), radius, "uniform_ball", device, dtype, rng)
    radial_uniform = (torch.linalg.vector_norm(theta.double(), dim=-1) / radius).pow(dimension).cpu().numpy().ravel()
    radial_ks = float(kstest(radial_uniform, "uniform").statistic)
    expected_moment = radius**2 / (dimension + 2)
    moment_error = float((theta.square().mean((0, 1)) / expected_moment - 1).abs().max())
    assert radial_ks < .015 and moment_error < .06
    values = torch.zeros((1, size), device=device, dtype=dtype)
    zero = lambda v: torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype)
    theta, _, _, _ = mutation(theta, values, torch.zeros(1, device=device, dtype=dtype), radius, zero, rng,
                              torch.full((1, 3), .4, device=device, dtype=dtype), 20, "uniform_ball", "rotation",
                              [(0, 4), (4, dimension), (0, dimension)])
    rotation_moment_error = float((theta.square().mean((0, 1)) / expected_moment - 1).abs().max())
    assert rotation_moment_error < .06
    calibration = []
    for kernel in ("rotation", "random_walk"):
        cfg = argparse.Namespace(**vars(args))
        cfg.replicas, cfg.particles, cfg.sweeps, cfg.survival, cfg.kernel = 6, 4096, 12, .5, kernel
        score = lambda v: -v[..., 0]
        state, rngs = new_state(cfg, 1., "uniform_ball", score, dimension, 5733)
        for target in (-.2, -.5, -.8):
            for _ in range(500):
                smc_step(state, target, cfg, 1., "uniform_ball", score, rngs, [(0, 4), (4, dimension), (0, dimension)])
                if bool((state["threshold"] <= target).all()):
                    break
            else:
                raise AssertionError("球冠测试未到达阈值")
            expected = .5 * betainc((dimension + 1) / 2, .5, 1-target**2)
            stats = mass_statistics(state["log_mass"])
            error = stats["K_bits"] + math.log2(expected)
            assert abs(error) < max(.18, 5 * stats["K_sd_bits"] / math.sqrt(cfg.replicas))
            calibration.append({"kernel": kernel, "cap": -target, "analytic_probability": expected,
                                "K_error_bits": error, **stats})
    # 断点必须保留全部随机流，续跑一步与连续执行逐位一致。
    folder = root / "resume_test"
    folder.mkdir(exist_ok=True)
    checkpoint(folder, state, rngs, "test", "running")
    copied, copied_rng = restore(folder, "test", args.device)
    smc_step(state, -.85, cfg, 1., "uniform_ball", score, rngs, [(0, 4), (4, dimension), (0, dimension)])
    smc_step(copied, -.85, cfg, 1., "uniform_ball", score, copied_rng, [(0, 4), (4, dimension), (0, dimension)])
    for key in ("theta", "values", "log_mass", "lineages", "scales", "threshold"):
        assert torch.equal(state[key], copied[key]), key
    score, x, signs = make_score(27030, args)
    theta = sample_prior((4, 256, legacy.parameter_count()), 48., "uniform_ball", device, dtype, rng)
    observed = score(theta).cpu().double()
    reference = F.softplus(-signs.cpu().double() * legacy.forward_logits(
        theta.cpu().double().reshape(-1, legacy.parameter_count()), x.cpu().double())).mean(-1).reshape(observed.shape)
    max_error = float((observed-reference).abs().max())
    assert torch.allclose(observed, reference, atol=2e-5, rtol=2e-4)
    result = {"radial_ks": radial_ks, "coordinate_moment_relative_error": moment_error,
              "rotation_moment_relative_error": rotation_moment_error, "sphere_caps": calibration,
              "resume_bitwise_equal": True, "forward_float64_max_error": max_error, "passed": True}
    atomic_json(root / "self_test.json", result)
    log(json.dumps(result, ensure_ascii=False), root)


def package(root):
    path = root.with_name(root.name + "_package.zip")
    temp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for file in root.rglob("*"):
            if file.is_file() and file.name not in {"checkpoint.pt"} and not file.name.endswith(".tmp"):
                archive.write(file, str(Path(root.name) / file.relative_to(root)))
        for file in (Path(__file__), Path(legacy.__file__)):
            archive.write(file, str(Path(root.name) / "source" / file.name))
    replace_retry(temp, path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results_4bit_uniform_ball_volume"))
    parser.add_argument("--device", default="cuda:1" if os.name == "nt" else "cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--radii", type=float, nargs="+", default=[24., 48., 72.])
    parser.add_argument("--rules", type=lambda v: int(v, 0), nargs="+", default=[0, 255, 4080, 27030])
    parser.add_argument("--nodes", type=float, nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--measures", choices=("uniform_ball", "truncated_gaussian"), nargs="+", default=["uniform_ball"])
    parser.add_argument("--kernel", choices=("rotation", "random_walk"), default="rotation")
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--survival", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=2026091307)
    parser.add_argument("--micro-batch", type=int, default=32768)
    parser.add_argument("--prior-samples", type=int, default=131072)
    parser.add_argument("--max-seconds", type=float, default=180.)
    parser.add_argument("--max-levels", type=int, default=6000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.replicas < 2 or args.particles < 64 or args.width < 2 or args.sweeps < 1:
        parser.error("需要至少2个replica、64个粒子、宽度2和一次更新。")
    if not 0 < args.survival < 1 or min(args.radii) <= 0 or min(args.nodes) <= 0:
        parser.error("半径和阈值必须为正，存活分位数须在0和1之间。")
    if args.nodes != sorted(set(args.nodes), reverse=True) or any(f < 0 or f > 65535 for f in args.rules):
        parser.error("节点须严格递减，规则编号须为16位真值表。")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("请使用GPU版PyTorch并显式选择空闲CUDA设备，不自动退回CPU。")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy.Config.WIDTH = args.width
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = scientific_config(args)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (root / "config.json").exists():
        if json.loads((root / "config.json").read_text(encoding="utf-8")) != config:
            raise RuntimeError("输出目录已有不同协议，拒绝覆盖。")
    else:
        atomic_json(root / "config.json", config)
    atomic_json(root / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                                           "device": args.device, "gpu": torch.cuda.get_device_name(args.device),
                                           "dtype": args.dtype, "tf32": False, "amp": False})
    log(f"=== {PROTOCOL} | {torch.cuda.get_device_name(args.device)} | D={legacy.parameter_count()} ===", root)
    if args.self_test:
        self_test(args, root)
        return
    prior_calibration(args, root)
    results = []
    try:
        for measure in args.measures:
            for radius in args.radii:
                for fid in args.rules:
                    results.append(run_job(args, fid, radius, measure, root, signature))
                    atomic_json(root / "index.json", results)
    except KeyboardInterrupt:
        log("已保存断点，使用相同命令恢复。", root)
    finally:
        log(f"结果包：{package(root)}", root)


if __name__ == "__main__":
    main()
