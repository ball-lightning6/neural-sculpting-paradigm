"""20规则、三种预指定自然测度；复用冻结SMC层，不修改历史脚本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import signal
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import experiment_4bit_uniform_ball_volume as core


FAMILIES = ("wide_gaussian", "spherical_t8", "uniform_cube")
OFFSETS = {"wide_gaussian": 10000000, "spherical_t8": 20000000, "uniform_cube": 30000000}
PROTOCOL = "4bit_natural_reference_measures_v1"
STOP = False
ORIGINAL_STEP = core.smc_step


def scale_for(reference_radius, dimension):
    return reference_radius / math.sqrt(dimension + 2)


def draw_vector(shape, std, measure, device, dtype, rng):
    if measure == "wide_gaussian":
        return std * torch.randn(shape, device=device, dtype=dtype, generator=rng)
    if measure == "uniform_cube":
        return math.sqrt(3) * std * (2*torch.rand(shape, device=device, dtype=dtype, generator=rng)-1)
    if measure == "spherical_t8":
        z = torch.randn(shape, device=device, dtype=dtype, generator=rng)
        chi2 = torch.randn(shape[:-1]+(8,), device=device, dtype=dtype, generator=rng).square().sum(-1, keepdim=True)
        # 所有坐标共享一个随机尺度，才是球对称多元t；不是逐坐标独立t。
        return std * math.sqrt(6) * z / chi2.sqrt()
    raise ValueError(measure)


def sample_prior(shape, reference_radius, measure, device, dtype, rng):
    return draw_vector(shape, scale_for(reference_radius, shape[-1]), measure, device, dtype, rng)


def reflect_box(value, half_width):
    return half_width - ((value + half_width).remainder(4*half_width)-2*half_width).abs()


@torch.no_grad()
def mutation(theta, values, threshold, reference_radius, score, rng, scales, sweeps, measure, kernel, blocks):
    std = scale_for(reference_radius, theta.shape[-1])
    half_width = math.sqrt(3) * std
    rates = torch.zeros_like(scales)
    refresh_rate = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
    for _ in range(sweeps):
        for j, (start, stop) in enumerate(blocks):
            rho = scales[:, j, None, None]
            current = theta[..., start:stop]
            noise = torch.randn(current.shape, device=theta.device, dtype=theta.dtype, generator=rng)
            proposed = theta.clone()
            if measure == "wide_gaussian":
                proposed[..., start:stop] = (1-rho.square()).sqrt()*current + rho*std*noise
                prior_accept = torch.ones_like(values, dtype=torch.bool)
            elif measure == "uniform_cube":
                # 对称高斯步经逐坐标镜面反射，保持立方体内的均匀分布。
                proposed[..., start:stop] = reflect_box(current + rho*std*noise, half_width)
                prior_accept = torch.ones_like(values, dtype=torch.bool)
            elif measure == "spherical_t8":
                proposed = core.rotate_block(theta, start, stop, scales[:, j], rng)
                prior_accept = torch.ones_like(values, dtype=torch.bool)
            else:
                raise ValueError(measure)
            next_values = score(proposed)
            accepted = prior_accept & (next_values <= threshold[:, None])
            theta = torch.where(accepted[..., None], proposed, theta)
            values = torch.where(accepted, next_values, values)
            rates[:, j] += accepted.to(theta.dtype).mean(1)
        if measure == "spherical_t8":
            fresh = draw_vector(theta.shape, std, measure, theta.device, theta.dtype, rng)
            new_radius = torch.linalg.vector_norm(fresh, dim=-1, keepdim=True)
            old_radius = torch.linalg.vector_norm(theta, dim=-1, keepdim=True)
            proposed = theta * new_radius / old_radius.clamp_min(1e-30)
        else:
            # 独立坐标先验的单坐标刷新；多元t不能使用这一更新。
            indices = torch.randint(theta.shape[-1], theta.shape[:-1]+(1,), device=theta.device, generator=rng)
            replacement = draw_vector(indices.shape, std, measure, theta.device, theta.dtype, rng)
            proposed = theta.scatter(-1, indices, replacement)
        next_values = score(proposed)
        accepted = next_values <= threshold[:, None]
        theta = torch.where(accepted[..., None], proposed, theta)
        values = torch.where(accepted, next_values, values)
        refresh_rate += accepted.to(theta.dtype).mean(1)
    return theta, values, rates/sweeps, refresh_rate/sweeps


@torch.no_grad()
def record_node(state, args, reference_radius, measure, fid, x, signs, rates, refresh):
    threshold = args.nodes[state["node_index"]]
    stats = core.mass_statistics(state["log_mass"])
    dimension = state["theta"].shape[-1]
    std = scale_for(reference_radius, dimension)
    norm2 = state["theta"].double().square().sum(-1)
    stride = max(1, args.particles//32)
    small = state["theta"][:, ::stride][:, :32].detach().cpu().clone()
    cached = state["values"][:, ::stride][:, :32].detach().cpu().double()
    z64 = core.legacy.forward_logits(small.double().reshape(-1, dimension), x.cpu().double())
    margins = z64 * signs.cpu().double()
    check64 = F.softplus(-margins).mean(-1).reshape(cached.shape)
    error = float((cached-check64).abs().max())
    flags = []
    if stats["K_sd_bits"] > 2 or stats["K_range_bits"] > 6:
        flags.append("REPLICA_SPREAD")
    if stats["effective_replicas"] < args.replicas/2:
        flags.append("REPLICA_MASS_IMBALANCE")
    if error > 2e-6+2e-4*threshold or float(check64.max()) > threshold+2e-6:
        flags.append("NUMERIC_AUDIT")
    if measure == "uniform_cube" and float(state["theta"].abs().max()) > math.sqrt(3)*std*(1+2e-6):
        raise RuntimeError("立方体硬边界校验失败。")
    if threshold < math.log(2)/16 and not bool((margins > 0).all()):
        raise RuntimeError("低loss节点hard function校验失败。")
    row = {"function_id": fid, "function_hex": f"0x{fid:04X}", "measure": measure,
           "reference_ball_radius": reference_radius, "coordinate_std": std,
           "hard_radius_bound": None, "threshold": threshold, "level": state["level"],
           "elapsed_seconds": state["elapsed_seconds"], **stats, "mean_norm2": float(norm2.mean()),
           "norm2_p05": float(torch.quantile(norm2.flatten(), .05)),
           "norm2_p95": float(torch.quantile(norm2.flatten(), .95)), "max_norm": float(norm2.max().sqrt()),
           "mean_loss": float(state["values"].mean()), "audit_mean_margin": float(margins.mean()),
           "audit_mean_sample_margin_std": float(margins.std(-1).mean()),
           "numeric_max_abs_error": error, "accept_mean": float(rates.mean()),
           "refresh_accept": float(refresh.mean()),
           "refresh_kind": "radius" if measure == "spherical_t8" else "coordinate",
           "warnings": ";".join(flags)}
    if measure == "uniform_cube":
        row["cube_half_width"] = math.sqrt(3)*std
        row["log2_euclidean_volume_standard_coordinates"] = dimension*math.log2(2*math.sqrt(3)*std)-stats["K_bits"]
    state["curve"].append(row)
    for r in range(args.replicas):
        state["replicas"].append({"threshold": threshold, "replica": r, "log_mass": float(state["log_mass"][r]),
                                  "K_bits": float(-state["log_mass"][r]/math.log(2)),
                                  "norm2_mean": float(norm2[r].mean()), "norm2_sd": float(norm2[r].std()),
                                  "unique_ancestors": int(torch.unique(state["lineages"][r]).numel())})
    state["audit"].append({"threshold": threshold, "theta": small, "loss_float64": check64,
                           "logits_float64": z64.reshape(args.replicas, -1, 16)})
    state["node_index"] += 1
    return row


def safe_step(*args, **kwargs):
    if STOP:
        raise KeyboardInterrupt
    return ORIGINAL_STEP(*args, **kwargs)


def install_adapter():
    core.sample_prior = sample_prior
    core.mutation = mutation
    core.record_node = record_node
    core.smc_step = safe_step


def select_panel(source, root):
    dest = root / "panel.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    old = json.loads((source / "panel.json").read_text(encoding="utf-8"))
    ids = old["anchors"] + old["stratified_core"][::2] + [495, 10942, 12276, 2464]
    assert len(ids) == len(set(ids)) == 20
    std = scale_for(72., 97)
    panel = {"protocol": PROTOCOL, "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
             "source_panel": str(source.resolve()), "source_panel_sha256": core.digest(source / "panel.json"),
             "selection": "原4个锚点+原12个分层区间各第一个样本+预指定4个近似/反转案例；不使用新测度结果挑选。",
             "function_ids": ids, "reference_radius": 72., "dimension": 97, "coordinate_std": std,
             "expected_norm2": 97*std**2, "cube_half_width": math.sqrt(3)*std,
             "student_df": 8, "student_scale": std*math.sqrt(6/8), "measures": list(FAMILIES),
             "primary_spherical_measures": ["wide_gaussian", "spherical_t8"],
             "directional_stress_test": ["uniform_cube"],
             "data_independent_definition": True,
             "note": "R72仅作为二阶矩标尺；宽高斯和球对称t没有范数硬边界。立方体是逐坐标边界，另作方向偏置压力测试。"}
    core.atomic_json(dest, panel)
    return panel


@torch.no_grad()
def self_test(args, root):
    from scipy.integrate import quad
    from scipy.stats import chi2, kstest, norm, t
    results = []
    for number, measure in enumerate(FAMILIES):
        dimension, reference_radius = 9, math.sqrt(11)
        rng = core.generator(args.device, 7632+number)
        theta = sample_prior((1, 65536, dimension), reference_radius, measure, args.device, torch.float32, rng)
        moment_error = float((theta.square().mean((0, 1))-1).abs().max())
        assert moment_error < .06, (measure, moment_error)
        zero = lambda t: torch.zeros(t.shape[:-1], device=t.device, dtype=t.dtype)
        moved, _, _, _ = mutation(theta, zero(theta), torch.zeros(1, device=args.device), reference_radius,
                                  zero, rng, torch.full((1, 3), .3, device=args.device), 25,
                                  measure, "native", [(0, 4), (4, 9), (0, 9)])
        stationary_error = float((moved.square().mean((0, 1))-1).abs().max())
        cdf = "norm" if measure == "wide_gaussian" else (lambda v:np.clip((v+math.sqrt(3))/(2*math.sqrt(3)), 0, 1)) if measure == "uniform_cube" else (lambda v:t.cdf(v/math.sqrt(6/8), df=8))
        ks = float(kstest(moved[0, :, 0].cpu().numpy(), cdf).statistic)
        diagonal = moved[0].sum(-1).cpu().numpy()/math.sqrt(dimension)
        diagonal_ks = float(kstest(diagonal, cdf).statistic)
        assert stationary_error < .06 and ks < .02, (measure, stationary_error, ks)
        if measure != "uniform_cube":
            assert diagonal_ks < .02, (measure, diagonal_ks)
        cfg = argparse.Namespace(**vars(args))
        cfg.replicas, cfg.particles, cfg.sweeps = 6, 4096, 16
        score = lambda t:-t[..., :2].min(-1).values
        state, rngs = core.new_state(cfg, reference_radius, measure, score, dimension, 8653+number)
        caps = [0., .75, 1.7 if measure == "uniform_cube" else 2.5]
        errors = []
        for cap in caps:
            for _ in range(500):
                core.smc_step(state, -cap, cfg, reference_radius, measure, score, rngs, [(0,4),(4,9),(0,9)])
                if bool((state["threshold"] <= -cap).all()):
                    break
            else:
                raise AssertionError("解析尾概率未到达")
            if measure == "spherical_t8":
                probability = quad(lambda g: norm.sf(cap*math.sqrt(g/6))**2*chi2.pdf(g,df=8),
                                   0, np.inf, epsabs=1e-12, epsrel=1e-10)[0]
            else:
                p = norm.sf(cap) if measure == "wide_gaussian" else (math.sqrt(3)-cap)/(2*math.sqrt(3))
                probability = p*p
            stats = core.mass_statistics(state["log_mass"])
            error = stats["K_bits"]+math.log2(probability)
            assert abs(error) < max(.2, 5*stats["K_sd_bits"]/math.sqrt(cfg.replicas)), (measure, cap, error)
            errors.append({"cap": cap, "expected_probability": probability, "K_error_bits": error, **stats})
        results.append({"measure": measure, "moment_error": moment_error, "stationary_error": stationary_error,
                        "marginal_KS": ks, "diagonal_KS_against_axis_CDF": diagonal_ks, "tail_checks": errors})
    core.atomic_json(root / "self_test.json", {"passed": True, "results": results})
    core.log(json.dumps(results, ensure_ascii=False), root)


def package(root):
    target = root.with_name(root.name+"_package.zip")
    temp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in root.rglob("*"):
            if path.is_file() and path.name != "checkpoint.pt" and path.suffix not in (".zip", ".tmp", ".pyc"):
                z.write(path, str(path.relative_to(root)))
    core.replace_retry(temp, target)
    return target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tmp/4bit_natural_measures_20260913"))
    parser.add_argument("--source-panel", type=Path, default=Path("tmp/4bit_uniform_ball_rank_panel_20260913"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--measures", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026091367)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.width, args.dtype, args.survival = 16, "float32", .5
    args.micro_batch, args.prior_samples = 32768, 131072
    args.max_levels, args.checkpoint_every, args.log_every = 6000, 50, 100
    args.nodes, args.radii, args.kernel = list(core.DEFAULT_NODES), [72.], "native"
    return args


@torch.no_grad()
def main():
    global STOP
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("必须使用空闲CUDA设备，不回退CPU采样。")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    core.legacy.Config.WIDTH = 16
    install_adapter()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    panel = select_panel(args.source_panel, root)
    args.rules = panel["function_ids"]
    settings = {k:v for k,v in vars(args).items() if k not in {"out", "source_panel", "device", "max_seconds", "max_levels", "self_test"}}
    config = {"protocol": PROTOCOL, "settings": settings, "prior_definitions": panel,
              "seed_offsets": OFFSETS, "adapter_sha256": core.digest(__file__),
              "smc_core_sha256": core.digest(core.__file__), "network_sha256": core.digest(core.legacy.__file__),
              "kernel_definitions": {"wide_gaussian":"scaled pCN", "uniform_cube":"reflected symmetric Gaussian steps",
                                     "spherical_t8":"block spherical rotations, preserving the overall radius"},
              "refresh_update":"Gaussian/cube: one independent coordinate; spherical t8: independent prior radius",
              "directory_R_note":"目录R72是二阶矩标尺，不是新测度的范数界限。"}
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (root/"config.json").exists() and json.loads((root/"config.json").read_text(encoding="utf-8")) != config:
        raise RuntimeError("协议不同，使用新的输出目录。")
    core.atomic_json(root/"config.json", config)
    core.atomic_json(root/"environment.json", {"python":platform.python_version(), "torch":torch.__version__,
                                             "numpy":np.__version__, "gpu":torch.cuda.get_device_name(args.device),
                                             "device":args.device,"dtype":args.dtype,"tf32":False,"amp":False})
    source = root/"source"
    source.mkdir(exist_ok=True)
    for path in (Path(__file__),Path(core.__file__),Path(core.legacy.__file__)):
        shutil.copy2(path,source/path.name)
    core.log(f"=== {PROTOCOL} | 20规则 x {len(args.measures)}测度 | std={panel['coordinate_std']:.6f} ===",root)
    if args.self_test:
        self_test(args,root)
        return
    def request_stop(signum, frame):
        global STOP
        STOP = True
        print("收到停止请求，将在完整SMC层之间保存退出。",flush=True)
    signal.signal(signal.SIGINT,request_stop)
    core.prior_calibration(args,root)
    results = []
    try:
        for fid in args.rules:
            for measure in args.measures:
                cfg = argparse.Namespace(**vars(args))
                cfg.seed = args.seed+OFFSETS[measure]
                results.append(core.run_job(cfg,fid,72.,measure,root,signature))
                core.atomic_json(root/"index.json",results)
    except KeyboardInterrupt:
        core.log("已保存断点。",root)
    finally:
        core.log(f"结果包：{package(root)}",root)


if __name__ == "__main__":
    main()
