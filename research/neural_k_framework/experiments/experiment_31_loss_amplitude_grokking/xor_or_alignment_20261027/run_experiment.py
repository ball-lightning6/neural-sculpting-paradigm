"""4-bit 全规则 SMC：平均 BCE、逐点 BCE 上限与 signed-margin 方差对齐。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEVICE = "cuda:1"
TIER = os.environ.get("XOR_ALIGNMENT_TIER", "primary")
if TIER not in ("primary", "confirm"):
    raise ValueError("未知预算档")
SETTINGS = dict(
    protocol="xor_or_signed_margin_alignment_v1", tier=TIER,
    replicas=4, particles=2048 if TIER == "primary" else 4096,
    adapt_sweeps=4, fixed_sweeps=16 if TIER == "primary" else 32,
    refresh_sweeps=96, seed=202610241701 if TIER == "primary" else 202610241702,
    thresholds=[.7, .5, .3, .2, .12, .08, .06, .04, .025, .016,
                .01, .0063, .004, .0025, .0016, .001],
    alignment_lambda=.1, dtype="float32", log_volume_dtype="float64",
    architecture=[4, 16, 1], activation="tanh", prior="N(0,I_97)",
    layer_weight_and_bias_divisors=[2, 4], input_encoding="zero_one",
    threshold_tolerance=0.0, tf32=False, amp=False,
)
OUT = ROOT / f"results_{TIER}"
TIME_LIMIT_SECONDS = float(os.environ.get("XOR_ALIGNMENT_SECONDS", "2400"))
THRESHOLDS = SETTINGS["thresholds"]
RULES = {"f": 0x0FF0, "g": 0xCFFC}
KINDS = ("mean_bce", "max_bce", "aligned")
BLOCKS = ((0, 80), (80, 97), (0, 97))
STOP = False


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def atomic_checkpoint(path, obj):
    temp = path.with_suffix(".new")
    torch.save(obj, temp)
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(message):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + message
    print(line, flush=True)
    if OUT.exists():
        with (OUT / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def inputs(device="cpu"):
    return ((torch.arange(16, device=device)[:, None] >>
             torch.arange(3, -1, -1, device=device)) & 1).float()


def forward(theta, x):
    shape = theta.shape[:-1]
    t = theta.reshape(-1, 97)
    hidden = torch.tanh(torch.bmm(x[None].expand(len(t), -1, -1),
                                 (t[:, :64].reshape(-1, 16, 4) / 2).transpose(1, 2))
                        + t[:, None, 64:80] / 2)
    z = torch.bmm(hidden, (t[:, 80:96] / 4).unsqueeze(-1)).squeeze(-1) + t[:, 96:97] / 4
    return z.reshape(*shape, 16)


def numpy_forward(theta):
    t = np.asarray(theta, dtype=np.float64).reshape(-1, 97)
    x = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(float)
    h = np.tanh(np.einsum("xi,phi->pxh", x, t[:, :64].reshape(-1, 16, 4) / 2)
                + t[:, None, 64:80] / 2)
    return np.einsum("pxh,ph->px", h, t[:, 80:96] / 4) + t[:, 96:97] / 4


def features(z, signs):
    margins = z * signs
    losses = torch.logaddexp(torch.zeros_like(margins), -margins)
    var = (margins - margins.mean(-1, keepdim=True)).square().mean(-1)
    return losses.mean(-1), losses.amax(-1), var


def score_features(mean, maximum, variance, kind):
    if kind == "mean_bce":
        return mean
    if kind == "max_bce":
        return maximum
    if kind == "aligned":
        return mean + SETTINGS["alignment_lambda"] * variance
    raise ValueError(kind)


def scorer_for(rule, kind, device):
    x = inputs(device)
    signs = torch.tensor([2 * ((RULES[rule] >> i) & 1) - 1 for i in range(16)],
                         dtype=torch.float32, device=device)
    def scorer(theta):
        return score_features(*features(forward(theta, x), signs), kind)
    return scorer, x, signs


def make_generators(device, job_index, count, seed):
    streams = {}
    for j, name in enumerate(("prior", "resample", "mutation")):
        streams[name] = [torch.Generator(device=device).manual_seed(
            seed + job_index * 10000019 + j * 1000003 + r * 10007)
            for r in range(count)]
    return streams


def normal_batch(shape, streams, device):
    return torch.stack([torch.randn(shape[1:], generator=g, device=device) for g in streams])


def initialize(count, particles, device, streams, scorer):
    theta = normal_batch((count, particles, 97), streams["prior"], device)
    return dict(theta=theta, score=scorer(theta),
                ancestry=torch.arange(particles, device=device)[None].expand(count, -1).clone(),
                K=torch.zeros(count, device=device, dtype=torch.float64),
                cutoff=torch.full((count,), float("inf"), device=device),
                scales=torch.tensor([.1, .1, .04], device=device)[None].repeat(count, 1),
                node=[0] * count, levels=[0] * count,
                history=[], nodes=[], elapsed_seconds=0.0)


@torch.no_grad()
def mutate(state, scorer, streams, sweeps, adapt, active=None):
    theta = state["theta"]
    count = len(theta)
    if active is None:
        active = torch.ones(count, device=theta.device, dtype=torch.bool)
    rates = torch.zeros((count, 3), device=theta.device)
    for sweep in range(sweeps):
        for b, (lo, hi) in enumerate(BLOCKS):
            proposal = state["theta"].clone()
            rho = state["scales"][:, b, None, None]
            noise = normal_batch(proposal[..., lo:hi].shape, streams["mutation"], theta.device)
            proposal[..., lo:hi] = torch.sqrt(1 - rho.square()) * proposal[..., lo:hi] + rho * noise
            values = scorer(proposal)
            accept = active[:, None] & torch.isfinite(values) & (values <= state["cutoff"][:, None])
            rate = accept.float().mean(-1)
            state["theta"] = torch.where(accept[..., None], proposal, state["theta"])
            state["score"] = torch.where(accept, values, state["score"])
            if sweep < adapt:
                new_scale = (state["scales"][:, b] * torch.exp(.35 * (rate - .3))).clamp(2e-4, .95)
                state["scales"][:, b] = torch.where(active, new_scale, state["scales"][:, b])
            else:
                rates[:, b] += rate / max(1, sweeps - adapt)
    return rates


@torch.no_grad()
def step(state, scorer, streams, thresholds, adapt, fixed):
    count, n = state["score"].shape
    active = torch.tensor([i < len(thresholds) for i in state["node"]], device=state["theta"].device)
    target = torch.tensor([thresholds[min(i, len(thresholds) - 1)] for i in state["node"]],
                          device=state["theta"].device)
    ordered, order = state["score"].sort(-1)
    median = (ordered[:, n // 2 - 1] + ordered[:, n // 2]) / 2
    cutoff = torch.minimum(state["cutoff"], torch.maximum(target, median))
    cutoff = torch.where(active, cutoff, state["cutoff"])
    kept = (state["score"] <= cutoff[:, None]).sum(-1)
    counts = kept.tolist()
    if min(counts) == 0:
        raise RuntimeError("零幸存粒子，不能生成体积估计")
    select = []
    for r, number in enumerate(counts):
        if number == n or not bool(active[r]):
            select.append(torch.arange(n, device=order.device))
        else:
            choice = torch.randint(number, (n,), generator=streams["resample"][r], device=order.device)
            select.append(order[r, choice])
    indices = torch.stack(select)
    state["theta"] = torch.gather(state["theta"], 1, indices[..., None].expand(-1, -1, 97))
    state["score"] = torch.gather(state["score"], 1, indices)
    state["ancestry"] = torch.gather(state["ancestry"], 1, indices)
    state["K"] -= torch.where(active, torch.log2(kept.double() / n), 0.0)
    state["cutoff"] = cutoff
    rates = mutate(state, scorer, streams, adapt + fixed, adapt, active)
    if bool((state["score"] > cutoff[:, None]).any()):
        raise RuntimeError("更新越过当前事件")
    hit = (active & (cutoff == target)).tolist()
    for r in range(count):
        state["levels"][r] += int(active[r])
    row = dict(levels=state["levels"].copy(), cutoff=cutoff.tolist(), K=state["K"].tolist(),
               kept=counts, acceptance=rates.tolist(), scales=state["scales"].tolist(),
               ancestors=[int(v.unique().numel()) for v in state["ancestry"]], hit=hit)
    state["history"].append(row)
    return hit, row


def numpy_measure(theta, fid, kind):
    z = numpy_forward(theta)
    signs = np.array([2 * ((fid >> i) & 1) - 1 for i in range(16)])
    m = z * signs
    point_loss = np.logaddexp(0, -m)
    bce, maximum, var = point_loss.mean(-1), point_loss.max(-1), m.var(-1)
    score = score_features(bce, maximum, var, kind)
    ids = ((z >= 0).astype(np.int64) << np.arange(16)).sum(-1)
    c = (np.arange(16) >> 1) & 1
    stats = dict(bce_mean=float(bce.mean()), bce_quantiles=np.quantile(bce, [.05, .5, .95]).tolist(),
                 max_bce_mean=float(maximum.mean()), margin_variance_mean=float(var.mean()),
                 hard_exact_fraction=float(np.mean(ids == fid)), score_max=float(score.max()),
                 parameter_norm2_mean=float(np.square(theta.astype(float)).sum(-1).mean()),
                 per_input_loss_mean=point_loss.mean(0).tolist(), per_input_margin_mean=m.mean(0).tolist())
    for branch in (0, 1):
        mask = c == branch
        stats[f"c{branch}_bce_mean"] = float(point_loss[:, mask].mean())
        stats[f"c{branch}_margin_mean"] = float(m[:, mask].mean())
        stats[f"c{branch}_margin_variance_mean"] = float(m[:, mask].var(-1).mean())
    return stats, dict(logits=z, bce=bce, max_bce=maximum, margin_variance=var,
                       score=score, function_ids=ids)


def record_nodes(state, hit, folder, rule, kind, x, streams, scorer):
    before = {}
    for r, yes in enumerate(hit):
        if yes:
            before[r] = numpy_measure(state["theta"][r].cpu().numpy(), RULES[rule], kind)[0]
    active = torch.tensor(hit, device=x.device)
    rates = mutate(state, scorer, streams, SETTINGS["refresh_sweeps"], 0, active)
    for r, yes in enumerate(hit):
        if not yes:
            continue
        node = state["node"][r]
        threshold = THRESHOLDS[node]
        theta = state["theta"][r].cpu().numpy()
        stats, values = numpy_measure(theta, RULES[rule], kind)
        z = forward(state["theta"][r], x).cpu().numpy()
        score = state["score"][r].cpu().numpy()
        max_logit_error = float(np.max(np.abs(z - values["logits"])))
        max_score_error = float(np.max(np.abs(score - values["score"])))
        tolerance = 2e-6 + 2e-4 * threshold
        outside = int(np.count_nonzero(values["score"] > threshold + tolerance))
        if outside:
            raise RuntimeError(f"独立 float64 复算越界：{rule}/{kind}/{r}/{threshold}: {outside}")
        if max_score_error > tolerance:
            raise RuntimeError(f"独立分数误差过大：{max_score_error}")
        snapshot = f"r{r}_node{node:02d}.npz"
        np.savez_compressed(folder / snapshot, theta=theta, **values,
                            ancestry=state["ancestry"][r].cpu().numpy(),
                            torch_logits=z, torch_score=score)
        row = dict(rule=rule, kind=kind, replica=r, threshold=threshold,
                   K_bits=float(state["K"][r]), level=state["levels"][r], snapshot=snapshot,
                   acceptance=rates[r].tolist(), before_refresh=before[r], after_refresh=stats,
                   numeric=dict(max_logit_error=max_logit_error, max_score_error=max_score_error,
                                outside_tolerance=outside,
                                outside_exact=int(np.count_nonzero(values["score"] > threshold))),
                   ancestors=int(state["ancestry"][r].unique().numel()))
        state["nodes"].append(row)
        state["node"][r] += 1
    atomic_json(folder / "nodes.json", state["nodes"])


def move_state(state, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.items()}


def run_job(rule, kind, index, signature, invocation_start, device):
    folder = OUT / f"{rule}__{kind}"
    folder.mkdir(exist_ok=True)
    path = folder / "checkpoint.pt"
    streams = make_generators(device, index, SETTINGS["replicas"], SETTINGS["seed"])
    scorer, x, _ = scorer_for(rule, kind, device)
    if path.exists():
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload["signature"] != signature:
            raise RuntimeError("配置或源码变化，拒绝复用检查点")
        state = move_state(payload["state"], device)
        for name, group in streams.items():
            for g, old in zip(group, payload["rng"][name]):
                g.set_state(old)
        if all(i == len(THRESHOLDS) for i in state["node"]):
            log(f"SKIP {rule}__{kind}")
            return True
    else:
        state = initialize(SETTINGS["replicas"], SETTINGS["particles"], device, streams, scorer)
    start = time.perf_counter()
    old_elapsed = state["elapsed_seconds"]
    def save():
        state["elapsed_seconds"] = old_elapsed + time.perf_counter() - start
        atomic_checkpoint(path, dict(signature=signature, state=move_state(state, "cpu"),
                                     rng={name: [g.get_state() for g in group] for name, group in streams.items()}))
        atomic_json(folder / "levels.json", state["history"])
    save()
    last_log = last_save = start
    log(f"START {rule}__{kind} {SETTINGS['replicas']}池×{SETTINGS['particles']}粒子")
    while any(i < len(THRESHOLDS) for i in state["node"]):
        if STOP or (OUT / "STOP").exists() or time.perf_counter() - invocation_start > TIME_LIMIT_SECONDS:
            save()
            log(f"PAUSED {rule}__{kind}，已保存检查点")
            return False
        hit, row = step(state, scorer, streams, THRESHOLDS,
                        SETTINGS["adapt_sweeps"], SETTINGS["fixed_sweeps"])
        if any(hit):
            record_nodes(state, hit, folder, rule, kind, x, streams, scorer)
            save()
            last_save = time.perf_counter()
        now = time.perf_counter()
        if now - last_log > 25 or all(i == len(THRESHOLDS) for i in state["node"]):
            log(f"{rule}__{kind} level={max(row['levels'])} score<={max(row['cutoff']):.6g} "
                f"K={np.median(row['K']):.2f}，本组{(now-start)/60:.1f}分钟")
            last_log = now
        if now - last_save > 30:
            save()
            last_save = now
        if max(state["levels"]) > 6000:
            save()
            raise RuntimeError("超过预定层数上限；保留结果，不外推")
    save()
    atomic_json(folder / "summary.json", dict(status="completed", rule=rule, kind=kind,
                                              K_bits=state["K"].tolist(), levels=state["levels"],
                                              elapsed_seconds=state["elapsed_seconds"]))
    log(f"DONE {rule}__{kind} K={state['K'].tolist()}")
    return True


def stop_handler(sig, frame):
    global STOP
    STOP = True


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA，不自动退回 CPU")
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = digest(Path(__file__))
    protocol = dict(settings=SETTINGS, source_sha256=source_hash, rules=RULES,
                    score_definitions=dict(mean_bce="mean softplus(-signed_margin)",
                                           max_bce="max softplus(-signed_margin)",
                                           aligned="mean_bce + 0.1 * population_var(signed_margin)"))
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (OUT / "protocol.json").exists():
        if json.loads((OUT / "protocol.json").read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("已有结果的源码或协议不同，请另用新目录")
    else:
        atomic_json(OUT / "protocol.json", protocol)
        (OUT / "frozen_source.py").write_bytes(Path(__file__).read_bytes())
    lock = OUT / "RUNNING.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_handler)
    atomic_json(OUT / f"environment_{time.time_ns()}.json", dict(
        python=platform.python_version(), executable=os.sys.executable, torch=str(torch.__version__),
        numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
        device=str(device), source_sha256=source_hash, time_limit_seconds=TIME_LIMIT_SECONDS))
    selected = set(filter(None, os.environ.get("XOR_ALIGNMENT_JOBS", "").split(",")))
    jobs = [(r, k) for k in KINDS for r in RULES]
    valid = {f"{r}__{k}" for r, k in jobs}
    if selected - valid:
        lock.unlink()
        raise ValueError(f"未知作业：{selected-valid}")
    start = time.perf_counter()
    log(f"GPU={torch.cuda.get_device_name(device)}，device={device}，输出={OUT}")
    try:
        for index, (rule, kind) in enumerate(jobs):
            if selected and f"{rule}__{kind}" not in selected:
                continue
            if not run_job(rule, kind, index, signature, start, device):
                break
    finally:
        lock.unlink(missing_ok=True)
    log(f"本次运行退出，耗时{(time.perf_counter()-start)/60:.2f}分钟")


if __name__ == "__main__":
    main()
