"""三层记忆补测：增强SMC与独立拟合诊断，不修改历史参考源码。"""

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import frozen_memory as m
from config import WORK_DIR, EVIDENCE, TASK, DEVICE, RUN_HOURS

ROOT = WORK_DIR / "enhanced"
ORIGINAL_NODE = m.node_diagnostics
COUNTS = (32, 80, 128)
FIT_ARCHITECTURES = ((32,), (240,), (32, 32, 32))
FIT_RATES = (.001, .003)
FIT_NODES = (.1, .01, .001, .0001)


def geometry(theta, x, y, layers):
    """只读少量参数，不调用随机数；归一化margin不是最优margin。"""
    a = theta[:16].detach().double().cpu().numpy()
    values = np.broadcast_to(x.detach().double().cpu().numpy(), (len(a), len(x), x.shape[-1]))
    layer_rows = []
    output_norm = None
    for index, (start, end, stop, fan_in, fan_out) in enumerate(layers):
        raw = a[:, start:stop] / math.sqrt(fan_in)
        weight = raw[:, :end-start].reshape(len(a), fan_out, fan_in)
        bias = raw[:, end-start:]
        values = np.matmul(values, weight.transpose(0, 2, 1)) + bias[:, None, :]
        row = {"layer": index, "mean_standardized_norm_squared": float(np.mean(np.sum(a[:, start:stop]**2, axis=1))),
               "mean_raw_norm_squared": float(np.mean(np.sum(raw**2, axis=1)))}
        if index < len(layers)-1:
            values = np.tanh(values)
            row["saturated_fraction_abs_gt_095"] = float(np.mean(np.abs(values) > .95))
        else:
            output_norm = np.linalg.norm(raw, axis=1)
        layer_rows.append(row)
    margins = values[..., 0] * (2*y.detach().double().cpu().numpy()[None, :] - 1)
    minimum = margins.min(axis=1)
    bce = np.logaddexp(0., -margins)
    return {"particles_observed": len(a), "selection": "前16粒子，只作描述，不视为16个独立样本",
            "layers": layer_rows, "mean_minimum_training_margin": float(minimum.mean()),
            "mean_minimum_margin_divided_by_readout_norm": float(np.mean(minimum / np.maximum(output_norm, 1e-30))),
            "mean_sample_BCE_max_to_mean_ratio": float(np.mean(bce.max(axis=1)/np.maximum(bce.mean(axis=1),1e-300))),
            "scope": "只读内部实现诊断；不是最大margin求解，也不是复杂度或训练可达性的证明。"}


def enriched_node(state, epsilon, x, y, layers, config, rates, change):
    node = ORIGINAL_NODE(state, epsilon, x, y, layers, config, rates, change)
    node["implementation_geometry"] = geometry(state.particles, x, y, layers)
    return node


def smc_config(output, device, seconds):
    return {"OUTPUT_DIR": output, "DEVICE": device, "ARCHITECTURES": ((32, 32, 32),),
            "COUNTS": COUNTS, "DATASETS": 2, "POOLS": 4, "PARTICLES": 4096,
            "MUTATION_SWEEPS": 48, "ADAPT_SWEEPS": 2,
            "THRESHOLDS": (*m.CONFIG["THRESHOLDS"], .0005, .0002, .0001),
            "PRIMARY_WINDOW": (.002, .001), "TOTAL_SECONDS": seconds,
            "SECONDS_PER_REPLICA": 1800., "KEEP_COMPLETED_CHECKPOINTS": True,
            "PRIOR_SEED": m.CONFIG["PRIOR_SEED"] + 10_000_000,
            "RESAMPLE_SEED": m.CONFIG["RESAMPLE_SEED"] + 10_000_000,
            "MUTATION_SEED": m.CONFIG["MUTATION_SEED"] + 10_000_000}


def check_data(config, expected):
    for dataset in range(config["DATASETS"]):
        x, y, order = m.make_data(dataset, config)
        if m.array_digest(x, y, order) != expected[str(dataset)]["raw_dataset_sha256"]:
            raise RuntimeError("原随机表身份不匹配")
        for n in config["COUNTS"]:
            if m.array_digest(x[order[:n]], y[order[:n]], order[:n]) != expected[str(dataset)]["prefix_sha256"][str(n)]:
                raise RuntimeError("原训练前缀身份不匹配")


def run_smc(root, device, seconds):
    m.node_diagnostics = enriched_node
    deadline = time.monotonic() + seconds
    config = smc_config(root / "smc_results", device, seconds)
    check_data(m.resolve_config(config), json.loads((EVIDENCE / "datasets.json").read_text(encoding="utf-8")))
    while time.monotonic() < deadline:
        config["TOTAL_SECONDS"] = max(0., deadline-time.monotonic())
        result = m.run(config)
        if result["completed_pools"] + result["failed_pools"] == result["total_pools"]:
            return "completed_with_unresolved" if result["failed_pools"] else "completed"
        if result.get("reason") in ("keyboard_interrupt", "SIGINT", "SIGTERM", "STOP_file"):
            return "stopped"
        if "signal" in str(result.get("reason", "")).lower():
            return "stopped"
        if (config["OUTPUT_DIR"] / "STOP").exists():
            return "stopped"
    return "partial"


def raw_forward(theta, inputs, layers):
    values = inputs[None].expand(len(theta), -1, -1)
    for index, (start, end, stop, fan_in, fan_out) in enumerate(layers):
        weights = theta[:, start:end].reshape(len(theta), fan_out, fan_in)
        values = torch.bmm(values, weights.transpose(1, 2)) + theta[:, None, end:stop]
        if index < len(layers)-1:
            values = torch.tanh(values)
    return values.squeeze(-1)


def fit_seed(hidden, n, dataset):
    key = json.dumps([list(hidden), n, dataset, "memory_fit_20261025"])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**62)


def fit_one(folder, hidden, n, dataset, lr, device, deadline, stop,
            models=16, max_steps=15000, max_seconds=300., nodes=FIT_NODES):
    folder.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = folder / "RESULT.json", folder / "checkpoint.pt"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    x0, y0, order = m.make_data(dataset, m.CONFIG)
    x = torch.from_numpy(x0[order[:n]]).to(device)
    y = torch.from_numpy(y0[order[:n]].astype(np.float32)).to(device)
    _, layers, size = m.parameter_blocks(hidden)
    seed = fit_seed(hidden, n, dataset)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(models, size, generator=generator)
    for start, _, end, fan_in, _ in layers:
        initial[:, start:end] /= math.sqrt(fan_in)
    theta = torch.nn.Parameter(initial.to(device))
    optimizer = torch.optim.Adam([theta], lr=lr, weight_decay=0.)
    identity = {"hidden": list(hidden), "n": n, "dataset": dataset, "lr": lr, "models": models,
                "max_steps": max_steps, "max_seconds": max_seconds, "nodes": list(nodes),
                "seed": seed, "parameters": size, "raw_parameter_optimization": True,
                "data_hash": m.array_digest(x0[order[:n]], y0[order[:n]], order[:n]),
                "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                "device": str(device), "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None}}
    state = {"identity": identity, "step": 0, "seconds": 0.,
             "hits": {str(eps): [None]*models for eps in nodes}}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["state"]["identity"] != identity:
            raise RuntimeError("拟合诊断断点配置或环境不匹配")
        state = saved["state"]
        with torch.no_grad():
            theta.copy_(saved["theta"].to(device))
        optimizer.load_state_dict(saved["optimizer"])
    base_seconds, start = state["seconds"], time.monotonic()
    last_save, last_log = state["step"], time.monotonic()
    terminal = None
    while True:
        state["seconds"] = base_seconds + time.monotonic()-start
        logits = raw_forward(theta, x, layers)
        losses = torch.nn.functional.softplus((1-2*y[None])*logits).mean(1)
        values = losses.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("拟合诊断出现非有限loss")
        for eps in nodes:
            for i in np.flatnonzero(values <= eps):
                if state["hits"][str(eps)][i] is None:
                    state["hits"][str(eps)][i] = {"step": state["step"], "seconds": state["seconds"], "loss": float(values[i])}
        if time.monotonic()-last_log >= 30:
            reached = {str(eps): sum(v is not None for v in state["hits"][str(eps)]) for eps in nodes}
            print(f"拟合 step={state['step']} 累计{state['seconds']:.1f}秒 到达数={reached}", flush=True)
            last_log = time.monotonic()
        if all(v is not None for v in state["hits"][str(nodes[-1])]):
            terminal = "completed"
        elif state["step"] >= max_steps or state["seconds"] >= max_seconds:
            terminal = "budget_censored"
        paused = stop[0] or time.monotonic() >= deadline
        if terminal or paused or state["step"]-last_save >= 100:
            state["last_losses"] = values.tolist()
            m.atomic_write(checkpoint, lambda f: torch.save({"state": state, "theta": theta.detach().cpu(),
                                                          "optimizer": optimizer.state_dict()}, f), binary=True)
            m.atomic_json(folder / "PROGRESS.json", state)
            last_save = state["step"]
        if terminal:
            result = {**state, "status": terminal,
                      "scope": "达到阈值的成功率/步数；未达到是预算删失，不证明不可表示。不同架构的同编号种子不作严格配对。"}
            m.atomic_json(result_path, result)
            return result
        if paused:
            return {**state, "status": "partial"}
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        state["step"] += 1


def run_fit(root, device, seconds):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = m.resolve_config({"COUNTS": COUNTS, "DATASETS": 2})
    check_data(config, json.loads((EVIDENCE / "datasets.json").read_text(encoding="utf-8")))
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；不自动CPU长跑")
    deadline, stop, results = time.monotonic()+seconds, [False], []
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for n in COUNTS:
            for dataset in range(2):
                for hidden in FIT_ARCHITECTURES:
                    for lr in FIT_RATES:
                        key = f"h{'_'.join(map(str,hidden))}_n{n}_d{dataset}_lr{lr:g}"
                        print("记忆拟合诊断：" + key, flush=True)
                        result = fit_one(root / "fit_results" / key, hidden, n, dataset, lr, device, deadline, stop)
                        results.append({"key": key, "status": result["status"], "identity": result["identity"],
                                        "hits": result["hits"], "seconds": result["seconds"], "step": result["step"]})
                        m.atomic_json(root / "FIT_SUMMARY.json", {"groups": results, "full_complete": False})
                        if result["status"] == "partial":
                            return "stopped" if stop[0] else "partial"
        m.atomic_json(root / "FIT_SUMMARY.json", {"groups": results, "full_complete": True})
        return "completed"
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    if TASK not in ("enhanced", "fit"):
        raise RuntimeError("请通过统一入口显式选择 enhanced 或 fit。")
    settings = {"stage": "memory_smc" if TASK == "enhanced" else "fit",
                "device": DEVICE, "run_hours": RUN_HOURS}
    stage = settings["stage"]
    runner = run_smc if stage == "memory_smc" else run_fit
    status = runner(ROOT, settings["device"], settings["run_hours"]*3600.)
    m.atomic_json(ROOT / f"STATUS_{stage}.json", {"stage": stage, "status": status})


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
