"""5090：先冻结静态预测，再运行SGD；保留直接G&C和hard-fit SMC两种基线。"""

import csv
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# 只需编辑这里；无命令行参数。时间预算允许修改，科学配置/源码改变须另用输出目录。
CONFIG = {
    "device": "cuda:0", "allow_cpu": False,
    "output_dir": str(Path(__file__).resolve().parent / "results_static"),
    "mnist_root": str(Path(__file__).resolve().parent / "data"),
    "networks": [
        {"name": "cnn_w4", "kind": "cnn", "scale": 4},
        {"name": "mlp_256_128", "kind": "mlp", "hidden": [256, 128]},
        {"name": "cnn_w8", "kind": "cnn", "scale": 8},
    ],
    "include_anchor": True, "partitions": 3, "datasets_per_partition": 3,
    "sample_sizes": [32], "include_label_controls": True,
    "data_seed": 202610230, "probe_count": 2048,
    "prior_scale": 0.2, "fixed_zero_bias": True,
    "replicas": 4, "particles": 1024, "nodes": [0.3, 0.1, 0.03, 0.01, 0.003],
    "loss_band_fractions": [0.5, 0.8],
    "model_chunk": 128, "input_chunk": 16, "mutation_chunk": 128,
    "sweeps": 8, "refresh_sweeps": 24,
    "proposal_factors": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
    "gc_target": 512, "gc_max_draws": 2_000_000, "gc_seconds": 90,
    "sgd_models": 128, "sgd_lr": 0.001, "sgd_epochs": 2000,
    "sgd_batch_size": 2, "sgd_model_chunk": 16,
    "total_hours": 4, "replica_seconds": 1200, "sgd_seconds": 900,
    "max_replica_attempts": 3, "max_levels": 20_000,
    "checkpoint_seconds": 120, "audit_weights": 2,
    "keep_completed_checkpoints": False,
    "keep_failed_checkpoints": False,
}

STOP_REQUESTED = False


def request_stop(signum=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def digest_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scientific_hash(cfg):
    operational = {"device", "allow_cpu", "output_dir", "mnist_root", "total_hours", "replica_seconds",
                   "sgd_seconds", "checkpoint_seconds", "keep_completed_checkpoints", "keep_failed_checkpoints"}
    obj = {k: v for k, v in cfg.items() if k not in operational}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def log(out, message):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z") + " | " + message
    print(line, flush=True)
    with (out / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def build_model(spec):
    if spec["kind"] == "cnn":
        s = spec["scale"]
        a, b, c, d = [max(1, round(v * s)) for v in [3, 8, 60, 42]]
        shapes = [(a, 1, 5), (b, a, 5), (c, b * 16, 1), (d, c, 1), (2, d, 1)]
    else:
        dims = [784] + list(spec["hidden"]) + [2]
        shapes = [(b, a, 1) for a, b in zip(dims[:-1], dims[1:])]
    layers, biases, cursor = [], [], 0
    for co, ci, kernel in shapes:
        n = co * ci * kernel * kernel
        layers.append((slice(cursor, cursor + n), slice(cursor + n, cursor + n + co), co, ci, kernel))
        biases.extend(range(cursor + n, cursor + n + co))
        cursor += n + co
    return {"spec": spec, "layers": layers, "biases": biases, "parameters": cursor}


def forward(theta, x, model):
    m = len(theta)
    if model["spec"]["kind"] == "mlp":
        z = x.flatten(1)
        for i, (ws, bs, co, ci, _) in enumerate(model["layers"]):
            w = theta[:, ws].reshape(m, co, ci)
            z = torch.einsum("bi,moi->mbo", z, w) if i == 0 else torch.bmm(z, w.transpose(1, 2))
            z = z + theta[:, bs, None].transpose(1, 2)
            if i + 1 < len(model["layers"]):
                z = F.relu(z)
        return z
    ws, bs, co, ci, k = model["layers"][0]
    z = F.max_pool2d(F.relu(F.conv2d(x, theta[:, ws].reshape(m * co, ci, k, k), theta[:, bs].reshape(-1))), 2)
    ws, bs, co, ci, k = model["layers"][1]
    z = F.max_pool2d(F.relu(F.conv2d(z, theta[:, ws].reshape(m * co, ci, k, k), theta[:, bs].reshape(-1), groups=m)), 2)
    z = z.reshape(len(x), m, model["layers"][2][3]).permute(1, 0, 2)
    for i, (ws, bs, co, ci, _) in enumerate(model["layers"][2:], 2):
        z = torch.bmm(z, theta[:, ws].reshape(m, co, ci).transpose(1, 2)) + theta[:, bs, None].transpose(1, 2)
        if i + 1 < len(model["layers"]):
            z = F.relu(z)
    return z


def logits_chunked(theta, x, model, cfg):
    outputs = []
    for i in range(0, len(theta), cfg["model_chunk"]):
        rows = [forward(theta[i:i + cfg["model_chunk"]], x[j:j + cfg["input_chunk"]], model)
                for j in range(0, len(x), cfg["input_chunk"])]
        outputs.append(torch.cat(rows, 1))
    return torch.cat(outputs)


@torch.no_grad()
def energies(theta, x, y, model, cfg):
    z = logits_chunked(theta, x, model, cfg)
    gap = z[..., 1].double() - z[..., 0].double()
    signed = gap * (2 * y - 1)
    loss = F.softplus(-signed).mean(1)
    hard = (z.argmax(-1) == y).all(1)
    barrier = -signed.min(1).values
    # ReLU可在非零质量的区域产生完全相同logits；错误tie不能混入hard-fit事件。
    barrier = torch.where(hard, barrier, barrier.clamp_min(1e-12))
    return loss, hard, barrier


def sample_prior(n, model, cfg, generator, device):
    theta = torch.empty((n, model["parameters"]), device=device).uniform_(
        -cfg["prior_scale"], cfg["prior_scale"], generator=generator)
    if cfg["fixed_zero_bias"]:
        theta[:, model["biases"]] = 0
    return theta


def reflect(value, scale):
    return scale - torch.abs(torch.remainder(value + scale, 4 * scale) - 2 * scale)


@torch.no_grad()
def mutate(state, x, y, model, cfg, generator, sweeps):
    theta, loss, hard, barrier = [state[k] for k in ["theta", "loss", "hard", "barrier"]]
    accepted, proposed = 0, 0
    layer_slices = [layer[0] for layer in model["layers"]] + [slice(None)]
    for _ in range(sweeps):
        for index in layer_slices:
            # 预定尺度混合独立于粒子状态；每个反射对称核都保留目标均匀条件测度。
            k = int(torch.randint(len(cfg["proposal_factors"]), (), generator=generator, device=theta.device))
            step = cfg["prior_scale"] * cfg["proposal_factors"][k]
            for first in range(0, len(theta), cfg["mutation_chunk"]):
                end = min(first + cfg["mutation_chunk"], len(theta))
                proposal = theta[first:end].clone()
                noise = torch.randn(proposal[:, index].shape, generator=generator, device=theta.device)
                proposal[:, index] = reflect(proposal[:, index] + step * noise, cfg["prior_scale"])
                if cfg["fixed_zero_bias"]:
                    proposal[:, model["biases"]] = 0
                pl, ph, pb = energies(proposal, x, y, model, cfg)
                good = (pb <= state["upper"]) if state["phase"] == "hard" else (ph & (pl <= state["upper"]))
                good &= torch.isfinite(pl) & torch.isfinite(pb)
                theta[first:end][good] = proposal[good]
                loss[first:end][good], hard[first:end][good], barrier[first:end][good] = pl[good], ph[good], pb[good]
                accepted += int(good.sum())
                proposed += len(good)
    return accepted / proposed if proposed else None


def make_cases(cfg):
    cases = []
    if cfg["include_anchor"]:
        cases.append({"name": "anchor_0_7_n16", "kind": "pair", "digits": [0, 7], "n": 16,
                      "seed": cfg["data_seed"] - 1, "anchor": True})
    for p in range(cfg["partitions"]):
        rng = np.random.default_rng(cfg["data_seed"] + p)
        digits = sorted(rng.permutation(10)[:5].tolist())
        for seed in range(cfg["datasets_per_partition"]):
            for n in cfg["sample_sizes"]:
                cases.append({"name": f"partition{p}_seed{seed}_n{n}", "kind": "partition", "digits": digits,
                              "n": n, "seed": cfg["data_seed"] + 1000 + p * 100 + seed, "anchor": False})
    if cfg["include_label_controls"] and cfg["partitions"]:
        base = next(c for c in cases if c["kind"] == "partition")
        for noise in ["flip_quarter", "random_balanced"]:
            cases.append({**base, "name": base["name"] + "_" + noise, "noise": noise})
    return cases


def load_mnist(cfg):
    from torchvision.datasets import MNIST
    train = MNIST(cfg["mnist_root"], train=True, download=True)
    test = MNIST(cfg["mnist_root"], train=False, download=True)
    raw = train.data.float().unsqueeze(1) / 255
    mean, std = float(raw.mean()), float(raw.std(correction=1))
    return {"train_x": (raw - mean) / std, "train_digits": train.targets,
            "test_x": (test.data.float().unsqueeze(1) / 255 - mean) / std,
            "test_digits": test.targets, "mean": mean, "std": std}


def prepare_case(case, raw, cfg, folder):
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "dataset.npz"
    if cached.exists():
        with np.load(cached, allow_pickle=False) as z:
            data = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        manifest = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
        if manifest["case"] != case or manifest["arrays"] != {k: digest_array(v) for k, v in data.items()}:
            raise RuntimeError("已冻结的数据或训练条件发生变化，请使用新结果目录。")
        return data
    if case["kind"] == "pair":
        train_eligible = torch.isin(raw["train_digits"], torch.tensor(case["digits"]))
        test_eligible = torch.isin(raw["test_digits"], torch.tensor(case["digits"]))
        train_y = (raw["train_digits"] == case["digits"][1]).long()
        test_y = (raw["test_digits"] == case["digits"][1]).long()
    else:
        train_eligible = torch.ones(len(raw["train_digits"]), dtype=torch.bool)
        test_eligible = torch.ones(len(raw["test_digits"]), dtype=torch.bool)
        train_y = torch.isin(raw["train_digits"], torch.tensor(case["digits"])).long()
        test_y = torch.isin(raw["test_digits"], torch.tensor(case["digits"])).long()
    g = torch.Generator().manual_seed(case["seed"])
    ids = []
    for label in [0, 1]:
        available = torch.where(train_eligible & (train_y == label))[0]
        ids.append(available[torch.randperm(len(available), generator=g)[:case["n"] // 2]])
    ids = torch.cat(ids)
    ids = ids[torch.randperm(len(ids), generator=g)]
    labels = train_y[ids].clone()
    noise = case.get("noise")
    if noise == "flip_quarter":
        changed = torch.randperm(len(ids), generator=g)[:len(ids) // 4]
        labels[changed] = 1 - labels[changed]
    elif noise == "random_balanced":
        labels = torch.tensor([0] * (len(ids) // 2) + [1] * (len(ids) - len(ids) // 2))
        labels = labels[torch.randperm(len(labels), generator=g)]
    # 所有条件共享预先固定的无标签测试输入抽样；不根据测试正确率选样。
    test_ids = torch.where(test_eligible)[0]
    pg = torch.Generator().manual_seed(cfg["data_seed"] + 900_000)
    test_ids = test_ids[torch.randperm(len(test_ids), generator=pg)[:cfg["probe_count"]]]
    data = {"x": raw["train_x"][ids], "y": labels, "train_ids": ids,
            "probe_x": raw["test_x"][test_ids], "probe_y": test_y[test_ids], "probe_ids": test_ids}
    atomic_npz(cached, **{k: v.numpy() for k, v in data.items()})
    atomic_json(folder / "dataset_manifest.json", {"case": case, "normalization": [raw["mean"], raw["std"]],
                "arrays": {k: digest_array(v) for k, v in data.items()},
                "test_label_policy": "静态采样只读取probe_x；预测冻结后才评分probe_y。"})
    return data


@torch.no_grad()
def predictions(theta, probe, model, cfg):
    return logits_chunked(theta, probe, model, cfg).argmax(-1).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def run_gc(data, model, cfg, folder, deadline):
    path = folder / "gc.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    g = torch.Generator(device=data["x"].device).manual_seed(cfg["data_seed"] + 5001)
    ckpt = folder / "gc_checkpoint.pt"
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        g.set_state(saved["rng"])
        draws, accepted, hits, rows, elapsed = [saved[k] for k in ["draws", "accepted", "hits", "rows", "elapsed"]]
    else:
        draws, accepted, hits, rows, elapsed = 0, 0, 0, [], 0.
    tick = time.monotonic()
    while draws < cfg["gc_max_draws"] and accepted < cfg["gc_target"]:
        if STOP_REQUESTED or time.monotonic() >= deadline or elapsed + time.monotonic() - tick >= cfg["gc_seconds"]:
            break
        count = min(cfg["model_chunk"], cfg["gc_max_draws"] - draws)
        theta = sample_prior(count, model, cfg, g, data["x"].device)
        loss, hard, _ = energies(theta, data["x"], data["y"], model, cfg)
        draws += count
        hits += int(hard.sum())
        if hard.any():
            remaining = cfg["gc_target"] - accepted
            selected = theta[hard][:remaining]
            rows.append(predictions(selected, data["probe_x"], model, cfg))
            accepted += len(selected)
    elapsed += time.monotonic() - tick
    complete = accepted >= cfg["gc_target"] or draws >= cfg["gc_max_draws"] or elapsed >= cfg["gc_seconds"]
    result = {"method": "direct_rejection_G_and_C", "draws": draws, "accepted": accepted, "hard_hits": hits,
              "seconds": elapsed, "status": ("sufficient" if accepted >= cfg["gc_target"] else "budget_limited") if complete else "partial",
              "acceptance_estimate": hits / draws if draws else None,
              "warning": "无命中不等于零质量；预算停止后不以精度或性能理由删条件。"}
    if rows:
        atomic_npz(folder / "gc_forecast.npz", prediction=np.concatenate(rows))
    atomic_json(folder / "gc_progress.json", result)
    if complete:
        atomic_json(path, result)
        ckpt.unlink(missing_ok=True)
    else:
        atomic_torch(ckpt, {"rng": g.get_state().cpu(), "draws": draws, "accepted": accepted,
                           "hits": hits, "rows": rows, "elapsed": elapsed})
    return result


def checkpoint(folder, state, generator):
    payload = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}
    payload["rng"] = generator.get_state().cpu()
    atomic_torch(folder / "checkpoint.pt", payload)


@torch.no_grad()
def save_node(folder, label, state, data, model, cfg, generator):
    before = predictions(state["theta"], data["probe_x"], model, cfg)
    refresh_accept = mutate(state, data["x"], data["y"], model, cfg, generator, cfg["refresh_sweeps"])
    after = predictions(state["theta"], data["probe_x"], model, cfg)
    atomic_npz(folder / (label + ".npz"), prediction=after, train_loss=state["loss"].cpu().numpy(),
               hard_fit=state["hard"].cpu().numpy(), audit_theta=state["theta"][:cfg["audit_weights"]].cpu().numpy(),
               ancestry=state["ancestry"].cpu().numpy())
    record = {"label": label, "phase": state["phase"], "upper": state["upper"], "K_joint_bits": state["K"],
              "K_hard_bits": state.get("K_hard"), "level": state["level"],
              "unique_initial_ancestors": int(state["ancestry"].unique().numel()),
              "train_loss_mean": float(state["loss"].mean()), "train_loss_max": float(state["loss"].max()),
              "hard_fit_count": int(state["hard"].sum()), "refresh_acceptance": refresh_accept,
              "refresh_probability_mean_abs_change": float(np.abs(before.mean(0) - after.mean(0)).mean()),
              "diagnostic_warning": "谱系、接受率、短刷新稳定均不能单独证明全局混合或独立ESS。"}
    atomic_json(folder / (label + ".json"), record)
    return record


def node_label(value):
    return "bce_" + format(value, ".6g").replace(".", "p")


@torch.no_grad()
def run_replica(data, model, cfg, folder, replica, deadline):
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "final.json"
    if final.exists():
        return json.loads(final.read_text(encoding="utf-8"))
    g = torch.Generator(device=data["x"].device).manual_seed(cfg["data_seed"] + 10_000 + replica)
    path = folder / "checkpoint.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        g.set_state(state.pop("rng"))
        for k in ["theta", "loss", "hard", "barrier", "ancestry"]:
            state[k] = state[k].to(data["x"].device)
    else:
        theta = sample_prior(cfg["particles"], model, cfg, g, data["x"].device)
        loss, hard, barrier = energies(theta, data["x"], data["y"], model, cfg)
        state = {"theta": theta, "loss": loss, "hard": hard, "barrier": barrier,
                 "ancestry": torch.arange(len(theta), device=theta.device), "phase": "hard", "upper": float("inf"),
                 "level": 0, "K": 0., "K_hard": None, "node_index": 0, "attempts": 0, "nodes": [], "stalls": 0}
    tick = time.monotonic()
    stop_at = min(deadline, tick + cfg["replica_seconds"])
    last_save, last_log = tick, tick
    while state["node_index"] < len(cfg["nodes"]):
        if STOP_REQUESTED or time.monotonic() >= stop_at:
            break
        target = 0. if state["phase"] == "hard" else cfg["nodes"][state["node_index"]]
        values = state["barrier"] if state["phase"] == "hard" else state["loss"]
        upper = max(target, float(torch.quantile(values, .5)))
        upper = min(state["upper"], upper)
        keep = torch.where(torch.isfinite(values) & (values <= upper))[0]
        if state["phase"] == "bce":
            keep = keep[state["hard"][keep]]
        if state["phase"] == "hard" and upper == 0.:
            keep = keep[state["hard"][keep]]
        if not len(keep):
            checkpoint(folder, state, g)
            result = {"status": "no_survivors", "nodes": state["nodes"], "level": state["level"]}
            atomic_json(final, result)
            if not cfg["keep_failed_checkpoints"]:
                path.unlink(missing_ok=True)
            return result
        survival = len(keep) / cfg["particles"]
        state["K"] -= math.log2(survival)
        state["stalls"] = state["stalls"] + 1 if upper == state["upper"] else 0
        if len(keep) < cfg["particles"]:
            selected = keep[torch.randint(len(keep), (cfg["particles"],), generator=g, device=keep.device)]
            for k in ["theta", "loss", "hard", "barrier", "ancestry"]:
                state[k] = state[k][selected].clone()
        state["upper"] = upper
        state["level"] += 1
        acceptance = mutate(state, data["x"], data["y"], model, cfg, g, cfg["sweeps"])
        if upper == target:
            if state["phase"] == "hard":
                state["K_hard"] = state["K"]
                state["nodes"].append(save_node(folder, "hard_fit", state, data, model, cfg, g))
                state["phase"], state["upper"] = "bce", float("inf")
            else:
                state["nodes"].append(save_node(folder, node_label(target), state, data, model, cfg, g))
                state["node_index"] += 1
            checkpoint(folder, state, g)
            last_save = time.monotonic()
        record = {"level": state["level"], "phase": state["phase"], "upper": upper, "K_joint_bits": state["K"],
                  "survival": survival, "acceptance": acceptance,
                  "ancestors": int(state["ancestry"].unique().numel())}
        with (folder / "levels.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(clean(record), ensure_ascii=False) + "\n")
        if time.monotonic() - last_save >= cfg["checkpoint_seconds"]:
            checkpoint(folder, state, g)
            last_save = time.monotonic()
        if time.monotonic() - last_log >= 30:
            print(f"{folder.parent.name} 池{replica} level={state['level']} {state['phase']} upper={upper:.6g} K={state['K']:.2f} 接受率={acceptance:.3f}", flush=True)
            last_log = time.monotonic()
        if state["level"] >= cfg["max_levels"] or state["stalls"] >= 100:
            break
    checkpoint(folder, state, g)
    completed = state["node_index"] == len(cfg["nodes"])
    # 用户中断或全局轮换不消耗困难条件的重试预算。
    if not completed and not STOP_REQUESTED and time.monotonic() >= tick + cfg["replica_seconds"]:
        state["attempts"] += 1
        checkpoint(folder, state, g)
    exhausted = state["attempts"] >= cfg["max_replica_attempts"] or state["level"] >= cfg["max_levels"] or state["stalls"] >= 100
    result = {"status": "target_reached" if completed else ("unresolved_budget" if exhausted else "partial"),
              "nodes": state["nodes"], "level": state["level"], "attempts": state["attempts"],
              "last_call_seconds": time.monotonic() - tick}
    atomic_json(folder / "progress.json", result)
    if completed or exhausted:
        atomic_json(final, result)
        if ((completed and not cfg["keep_completed_checkpoints"]) or
                (exhausted and not cfg["keep_failed_checkpoints"])):
            path.unlink(missing_ok=True)
    return result


def freeze_forecasts(folder, cfg):
    path = folder / "forecast_frozen.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, h in record["files"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != h:
                raise RuntimeError("冻结预测文件hash已变化：" + name)
        return record
    paths = [folder / "dataset.npz", folder / "dataset_manifest.json", folder / "gc.json"] + sorted(folder.glob("gc_forecast.npz"))
    for r in range(cfg["replicas"]):
        replica = folder / f"replica_{r}"
        if not (replica / "final.json").exists():
            raise RuntimeError("必须先完成或明确停止所有静态副本，才能启动SGD。")
        paths.extend(sorted(replica.glob("*.npz")))
        paths.append(replica / "final.json")
    record = {"source_sha256": source_hash(), "scientific_sha256": scientific_hash(cfg),
              "files": {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "policy": "此后才运行独立初始化SGD和测试标签评分；不得反调静态协议。"}
    atomic_json(path, record)
    return record


def run_sgd(data, model, cfg, folder, deadline):
    final = folder / "sgd.json"
    if final.exists():
        return json.loads(final.read_text(encoding="utf-8"))
    device = data["x"].device
    ckpt = folder / "sgd_checkpoint.pt"
    g = torch.Generator().manual_seed(cfg["data_seed"] + 7001)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        theta = state["theta"].to(device).requires_grad_(True)
        g.set_state(state["batch_rng"])
    else:
        ig = torch.Generator().manual_seed(cfg["data_seed"] + 7000)
        theta = sample_prior(cfg["sgd_models"], model, cfg, ig, "cpu").to(device).requires_grad_(True)
        state = {"epoch": 0, "seen": np.zeros((cfg["sgd_models"], len(cfg["nodes"])), dtype=bool),
                 "node_rows": {node_label(v): [] for v in cfg["nodes"]}}
    opt = torch.optim.SGD([theta], lr=cfg["sgd_lr"])
    tick, saved = time.monotonic(), time.monotonic()
    last = None
    while state["epoch"] < cfg["sgd_epochs"] and not state["seen"].all():
        if STOP_REQUESTED or time.monotonic() >= min(deadline, tick + cfg["sgd_seconds"]):
            break
        order = torch.randperm(len(data["y"]), generator=g)
        for ids in order.split(cfg["sgd_batch_size"]):
            ids = ids.to(device)
            opt.zero_grad(set_to_none=True)
            for first in range(0, len(theta), cfg["sgd_model_chunk"]):
                z = forward(theta[first:first + cfg["sgd_model_chunk"]], data["x"][ids], model)
                loss = F.cross_entropy(z.reshape(-1, 2), data["y"][ids].expand(len(z), -1).reshape(-1), reduction="none")
                loss.reshape(len(z), -1).mean(1).sum().backward()
            if cfg["fixed_zero_bias"]:
                theta.grad[:, model["biases"]] = 0
            opt.step()
        state["epoch"] += 1
        loss, hard, _ = energies(theta, data["x"], data["y"], model, cfg)
        finite = torch.isfinite(loss)
        for j, threshold in enumerate(cfg["nodes"]):
            eligible = (hard & finite & (loss <= threshold)).cpu().numpy() & ~state["seen"][:, j]
            ids = np.flatnonzero(eligible)
            if len(ids):
                pred = predictions(theta[torch.from_numpy(ids).to(device)].detach(), data["probe_x"], model, cfg)
                for index, row in zip(ids, pred):
                    state["node_rows"][node_label(threshold)].append((int(index), state["epoch"], float(loss[index]), row))
                state["seen"][ids, j] = True
        last = {"hard_fit": int(hard.sum()), "finite": int(finite.sum()), "loss_median": float(loss[finite].median()) if finite.any() else None}
        if time.monotonic() - saved >= cfg["checkpoint_seconds"]:
            atomic_torch(ckpt, {**state, "theta": theta.detach().cpu(), "batch_rng": g.get_state()})
            saved = time.monotonic()
            print(f"{folder.name} SGD epoch={state['epoch']} fit={int(hard.sum())}/{len(theta)} 节点覆盖={state['seen'].sum(0).tolist()}", flush=True)
        if not finite.any():
            break
    atomic_torch(ckpt, {**state, "theta": theta.detach().cpu(), "batch_rng": g.get_state()})
    for label, rows in state["node_rows"].items():
        if rows:
            atomic_npz(folder / ("sgd_" + label + ".npz"), init_id=np.array([r[0] for r in rows]),
                       epoch=np.array([r[1] for r in rows]), train_loss=np.array([r[2] for r in rows]), prediction=np.stack([r[3] for r in rows]))
    finished = state["seen"].all() or state["epoch"] >= cfg["sgd_epochs"] or (last is not None and last["finite"] == 0)
    result = {"status": "completed" if finished else "partial", "epoch": state["epoch"],
              "models": cfg["sgd_models"], "last": last, "node_reached": state["seen"].sum(0).tolist(),
              "all_nodes_reached": bool(state["seen"].all()),
              "comparison": "SGD首次过阈值且hard-fit；静态是累计子水平集。二者不是数学上相同的分布。"}
    atomic_json(folder / "sgd_progress.json", result)
    if finished:
        atomic_json(final, result)
        if not cfg["keep_completed_checkpoints"]:
            ckpt.unlink(missing_ok=True)
    return result


def compare_prediction_views(sources, labels, threshold, cfg, sgd=None, gc=None):
    """同时比较累计事件与预定loss区间；区间内SGD人口单独报告。"""
    cumulative = np.concatenate([s["prediction"] for s in sources])
    cumulative_p = cumulative.mean(0)
    cumulative_acc = float((cumulative == labels).mean())
    views = [("cumulative", None)]
    if threshold is not None:
        views.extend((f"band_{fraction:g}_to_1", fraction * threshold)
                     for fraction in cfg["loss_band_fractions"])
    rows = []
    for name, lower in views:
        selected = [s["prediction"] if lower is None else
                    s["prediction"][(s["train_loss"] > lower) & (s["train_loss"] <= threshold)]
                    for s in sources]
        counts = [len(p) for p in selected]
        nonempty = [p for p in selected if len(p)]
        row = {"view": name, "loss_lower_exclusive": lower, "loss_upper_inclusive": threshold,
               "replicas_reached": len(sources), "replicas_with_selected": len(nonempty),
               "static_selected_per_replica": counts, "static_models": sum(counts),
               "static_retained_fraction": sum(counts) / len(cumulative),
               "static_accuracy": None, "static_replica_sd": None,
               "static_probability_replica_max_abs": None, "cumulative_vs_view_probability_mae": None}
        if nonempty:
            p_static = np.concatenate(nonempty).mean(0)
            row.update(static_accuracy=float((np.concatenate(nonempty) == labels).mean()),
                       static_replica_sd=float(np.std([float((p == labels).mean()) for p in nonempty])),
                       static_probability_replica_max_abs=float(np.max(np.ptp(np.stack([p.mean(0) for p in nonempty]), axis=0))),
                       cumulative_vs_view_probability_mae=float(np.abs(cumulative_p - p_static).mean()))
        if sgd is not None:
            p = sgd["prediction"]
            row["sgd_models_reached_node"] = len(p)
            if lower is not None:
                p = p[(sgd["train_loss"] > lower) & (sgd["train_loss"] <= threshold)]
            row["sgd_models_in_view"] = len(p)
            row["sgd_fraction_of_all_initializations"] = len(p) / cfg["sgd_models"]
            if len(p):
                sgd_p = p.mean(0)
                row["sgd_accuracy"] = float((p == labels).mean())
                # 固定同一SGD子集，再比较累计/区间静态预测，避免人口改变冒充改进。
                row["cumulative_probability_mae_same_sgd_subset"] = float(np.abs(cumulative_p - sgd_p).mean())
                row["cumulative_accuracy_gap_same_sgd_subset"] = row["sgd_accuracy"] - cumulative_acc
                if nonempty:
                    row["probability_mae"] = float(np.abs(p_static - sgd_p).mean())
                    row["accuracy_gap_sgd_minus_static"] = row["sgd_accuracy"] - row["static_accuracy"]
                if gc is not None and len(gc):
                    row["gc_probability_mae_same_sgd_subset"] = float(np.abs(gc.mean(0) - sgd_p).mean())
        if gc is not None and len(gc):
            row["gc_accuracy"] = float((gc == labels).mean())
            row["gc_models"] = len(gc)
            row["gc_target_sample_count_met"] = len(gc) >= cfg["gc_target"]
            if nonempty:
                row["static_vs_gc_probability_mae"] = float(np.abs(p_static - gc.mean(0)).mean())
        rows.append(row)
    return rows


def analyze_condition(folder, probe_y, cfg):
    if not (folder / "forecast_frozen.json").exists():
        return []
    freeze_forecasts(folder, cfg)
    labels = probe_y.cpu().numpy()
    rows = []
    for event in ["hard_fit"] + [node_label(v) for v in cfg["nodes"]]:
        sources = []
        for replica in range(cfg["replicas"]):
            path = folder / f"replica_{replica}" / (event + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    sources.append({"prediction": z["prediction"].copy(), "train_loss": z["train_loss"].copy()})
        if not sources:
            continue
        sgd, gc = None, None
        path = folder / ("sgd_" + event + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                sgd = {"prediction": z["prediction"].copy(), "train_loss": z["train_loss"].copy()}
        gc_path = folder / "gc_forecast.npz"
        if gc_path.exists():
            with np.load(gc_path, allow_pickle=False) as z:
                gc = z["prediction"].copy()
        threshold = next((v for v in cfg["nodes"] if node_label(v) == event), None)
        rows.extend({"event": event, **r} for r in compare_prediction_views(sources, labels, threshold, cfg, sgd, gc))
    atomic_json(folder / "analysis.json", rows)
    return rows


def write_progress(out, cfg, jobs, rows, status, current):
    atomic_json(out / "STATUS.json", {"status": status, "current": current, "scientific_sha256": scientific_hash(cfg),
                "conditions_total": len(jobs), "conditions_with_sgd_done": sum((out / j["name"] / "sgd.json").exists() for j in jobs)})
    atomic_json(out / "summary.json", rows)
    if rows:
        keys = sorted({k for row in rows for k in row})
        with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    text = ["# 静态预测实验进度", "", f"状态：{status}；当前：{current}。", "",
            "所有条件预先指定；失败、未收敛、未达节点均保留。工程衔接组不计作新盲预测。", "",
            "| 条件 | G&C | 静态完成副本 | SGD |", "|---|---|---:|---|"]
    for job in jobs:
        folder = out / job["name"]
        g = json.loads((folder / "gc.json").read_text(encoding="utf-8")) if (folder / "gc.json").exists() else {}
        r = sum((folder / f"replica_{i}" / "final.json").exists() for i in range(cfg["replicas"]))
        s = json.loads((folder / "sgd_progress.json").read_text(encoding="utf-8")) if (folder / "sgd_progress.json").exists() else {}
        text.append(f"| {job['name']} | {g.get('accepted', '待运行')} | {r}/{cfg['replicas']} | {s.get('status', '待运行')} |")
    (out / "PROGRESS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run(config=None, raw_data=None):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    cfg = dict(CONFIG if config is None else config)
    if any(n % 2 or n < 2 for n in cfg["sample_sizes"]):
        raise ValueError("分组平衡抽样的样本数须为正偶数。")
    if cfg["nodes"] != sorted(cfg["nodes"], reverse=True) or any(v <= 0 for v in cfg["nodes"]):
        raise ValueError("BCE节点必须为正且从高到低。")
    if any(not 0 < v < 1 for v in cfg["loss_band_fractions"]):
        raise ValueError("loss区间下界比例必须在0与1之间。")
    device = torch.device(cfg["device"])
    if device.type != "cuda" and not cfg["allow_cpu"]:
        raise RuntimeError("正式实验必须显式使用CUDA；CPU仅允许单元测试。")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请检查PyTorch环境，不能悄悄退回CPU。")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type == "cpu":
        torch.set_num_threads(1)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"scientific_sha256": scientific_hash(cfg), "source_sha256": source_hash(), "config": cfg,
                "models": [{"name": s["name"], "parameters": build_model(s)["parameters"]} for s in cfg["networks"]],
                "measure": "所有weight独立U[-a,a]；bias固定0；原始坐标概率质量。",
                "kernel": "独立于状态的预定尺度混合，分层/全局反射对称随机游走，无梯度、无SGD预热。"}
    frozen = out / "protocol.json"
    if frozen.exists():
        previous = json.loads(frozen.read_text(encoding="utf-8"))
        if previous["scientific_sha256"] != protocol["scientific_sha256"] or previous["source_sha256"] != protocol["source_sha256"]:
            raise RuntimeError("科学配置/源码已变化，请使用新的输出目录，不能混合旧结果。")
    else:
        atomic_json(frozen, protocol)
    runtime = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": str(device), "torch": torch.__version__,
               "cuda": torch.version.cuda, "dtype": "float32网络/float64约束评分", "tf32": False,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU_TEST_ONLY",
               "gpu_memory": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None}
    with (out / "runtime_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(runtime, ensure_ascii=False) + "\n")
    deadline = time.monotonic() + cfg["total_hours"] * 3600
    raw = load_mnist(cfg) if raw_data is None else raw_data
    cases = make_cases(cfg)
    jobs = []
    for spec in cfg["networks"]:
        for case in cases:
            if case["anchor"] and spec["name"] != cfg["networks"][0]["name"]:
                continue
            jobs.append({"name": spec["name"] + "__" + case["name"], "network": spec, "case": case})
    rows, partial = [], False
    log(out, f"设备={runtime['gpu']}，预定{len(jobs)}条件；本次预算{cfg['total_hours']}小时。")
    for job in jobs:
        folder = out / job["name"]
        folder.mkdir(parents=True, exist_ok=True)
        if STOP_REQUESTED or time.monotonic() >= deadline:
            partial = True
            break
        model = build_model(job["network"])
        cpu_data = prepare_case(job["case"], raw, cfg, folder)
        data = {k: v.to(device) for k, v in cpu_data.items() if k in {"x", "y", "probe_x"}}
        log(out, f"条件={job['name']}，参数={model['parameters']}，训练样本={len(data['y'])}")
        write_progress(out, cfg, jobs, rows, "partial", job["name"])
        gc = run_gc(data, model, cfg, folder, deadline)
        if gc["status"] == "partial":
            partial = True
            break
        complete_static = True
        for replica in range(cfg["replicas"]):
            if STOP_REQUESTED or time.monotonic() >= deadline:
                complete_static = False
                break
            result = run_replica(data, model, cfg, folder / f"replica_{replica}", replica, deadline)
            complete_static &= result["status"] != "partial"
        if complete_static:
            freeze_forecasts(folder, cfg)
            result = run_sgd(data, model, cfg, folder, deadline)
            partial |= result["status"] == "partial"
            for row in analyze_condition(folder, cpu_data["probe_y"], cfg):
                rows.append({"condition": job["name"], "anchor": job["case"]["anchor"], "parameters": model["parameters"], **row})
        else:
            partial = True
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # 当前条件未终止时先交回队列；避免为所有困难条件同时保存巨型粒子库。
        if not complete_static or result["status"] == "partial":
            break
    # 重建已完成条件的汇总，避免轮换后只保留本次运行的新行。
    rows = []
    for job in jobs:
        path = out / job["name"] / "analysis.json"
        if path.exists():
            rows.extend({"condition": job["name"], "anchor": job["case"]["anchor"],
                         "parameters": build_model(job["network"])["parameters"], **r}
                        for r in json.loads(path.read_text(encoding="utf-8")))
    all_done = all((out / j["name"] / "sgd.json").exists() for j in jobs)
    failures = any(json.loads(p.read_text(encoding="utf-8")).get("status") != "target_reached"
                   for p in out.glob("*/replica_*/final.json"))
    failures |= any(not json.loads(p.read_text(encoding="utf-8")).get("all_nodes_reached", False)
                    for p in out.glob("*/sgd.json"))
    status = ("completed_with_failures" if failures else "completed") if all_done else "partial"
    write_progress(out, cfg, jobs, rows, status, "本次结束，重启同一脚本可续跑")
    log(out, f"本次结束：{status}。")
    return {"status": status, "rows": rows}


if __name__ == "__main__":
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    run()
