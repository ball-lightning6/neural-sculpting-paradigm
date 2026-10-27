"""仅追加无梯度直接搜索对照；不改历史实验、不连接云端。"""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "peleg_static_recheck_20260919"
CONFIG = {
    "device": "cuda:1",
    "output_dir": str(HERE / "results"),
    "mode": "all",
    "models": 128,
    "nodes": [0.3, 0.1, 0.03, 0.01, 0.003],
    "prior_scale": 0.2,
    "pilot_initial_ids": list(range(504, 512)),
    "pilot_sigmas": [0.001, 0.003, 0.01, 0.03],
    "pilot_steps": 2000,
    "pilot_seconds": 120,
    "stp_steps": 30000,
    "stp_seconds": 900,
    "decay_steps": 2000,
    "pilot_seed": 202610241,
    "stp_seed": 202610242,
    "sgd_lr": 0.01,
    "sgd_epochs": 60,
    "sgd_batch": 2,
    "save_every": 1000,
}
EXPECTED = {
    "experiment.py": "7d0702ef2dcf3ffebf18b780e88721c617d3eead8135ce600683bedad94d8939",
    "results/dataset.npz": "5b9600a00475fb5401dfaaf8774add587105af4f41ae6116d6d7fda16a3a56ad",
    "results/initializations.pt": "09e90d52fbd1f2ae492db7e71a79514a3eecbdcacec84ef0ddc0882e85ebd075",
}
STOP = False


def interrupted(*_):
    global STOP
    STOP = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_npz(path, **obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **obj)
    os.replace(tmp, path)


def node_name(epsilon):
    return "node_" + format(epsilon, "g").replace(".", "p")


def load_reference():
    for name, digest in EXPECTED.items():
        if sha(REFERENCE / name) != digest:
            raise RuntimeError("历史文件hash不同，拒绝继续：" + name)
    spec = importlib.util.spec_from_file_location("peleg_stp_reference", REFERENCE / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(out, stage, message):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z") + " | " + stage + " | " + message
    print(line, flush=True)
    with (out / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    atomic_json(out / "STATUS.json", {"stage": stage, "message": message, "updated_at": time.time()})


def select_three(theta, delta, current, positive, negative, active):
    """只使用目标值排序；相等时留在原点，非有限提议不能获胜。"""
    pos = torch.where(torch.isfinite(positive), positive, torch.inf)
    neg = torch.where(torch.isfinite(negative), negative, torch.inf)
    plus = active & (pos < current) & (pos <= neg)
    minus = active & (neg < current) & ~plus
    sign = plus.to(theta.dtype) - minus.to(theta.dtype)
    updated = theta + sign[:, None] * delta
    loss = torch.where(plus, pos, torch.where(minus, neg, current))
    return updated, loss, plus, minus


def capture_nodes(state, loss, hard, iteration, cfg):
    if "node_weights" not in state:
        return
    for i, threshold in enumerate(cfg["nodes"]):
        ids = (loss <= threshold) & hard & ~state["seen"][:, i]
        if ids.any():
            state["seen"][ids, i] = True
            state["node_weights"][i, ids] = state["theta"][ids]
            state["node_loss"][ids, i] = loss[ids].to(state["node_loss"].dtype)
            state["node_iteration"][ids, i] = iteration


def cpu_state(state):
    return {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in state.items()}


def restore(state, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) and key != "rng" else value
            for key, value in state.items()}


def new_state(initial, cfg, capture):
    state = {"theta": initial.clone(), "iteration": 0, "elapsed": 0.0, "trace": [],
             "accepted": torch.zeros(len(initial), dtype=torch.long, device=initial.device)}
    if capture:
        n, p, k = len(initial), initial.shape[1], len(cfg["nodes"])
        state.update({
            "seen": torch.zeros((n, k), dtype=torch.bool, device=initial.device),
            "node_weights": torch.zeros((k, n, p), dtype=initial.dtype, device=initial.device),
            "node_loss": torch.full((n, k), torch.nan, dtype=torch.float64, device=initial.device),
            "node_iteration": torch.full((n, k), -1, dtype=torch.long, device=initial.device),
        })
    return state


@torch.no_grad()
def run_stp(ref, cfg, out, stage, initial, sigma0, x, y, seed, capture):
    path = out / (stage + "_state.pt")
    if path.exists():
        state = restore(torch.load(path, map_location="cpu", weights_only=True), x.device)
        if state.get("finished"):
            return state
    else:
        state = new_state(initial, cfg, capture)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    if "rng" in state:
        gen.set_state(state["rng"])
    state["loss"], state["hard"], _ = ref.score(ref.forward(state["theta"], x), y)
    if not torch.isfinite(state["loss"]).all():
        raise RuntimeError("非有限初始状态")
    capture_nodes(state, state["loss"], state["hard"], state["iteration"], cfg)
    steps = cfg["stp_steps" if capture else "pilot_steps"]
    seconds = cfg["stp_seconds" if capture else "pilot_seconds"]
    torch.cuda.synchronize(x.device)
    tick, previous = time.perf_counter(), state["elapsed"]
    logged = tick
    sigma0 = torch.as_tensor(sigma0, dtype=initial.dtype, device=x.device).expand(len(initial))
    for iteration in range(state["iteration"] + 1, steps + 1):
        if STOP or previous + time.perf_counter() - tick >= seconds:
            break
        active = ~state["seen"][:, -1] if capture else torch.ones(len(initial), dtype=torch.bool, device=x.device)
        if not active.any():
            break
        delta = torch.randn(state["theta"].shape, device=x.device, generator=gen, dtype=initial.dtype)
        delta *= (sigma0 / math.sqrt(1 + (iteration - 1) / cfg["decay_steps"]))[:, None]
        delta[:, ref.BIAS_INDICES] = 0
        trials = torch.cat((state["theta"] + delta, state["theta"] - delta))
        trial_loss, trial_hard, _ = ref.score(ref.forward(trials, x), y)
        m = len(initial)
        theta, loss, plus, minus = select_three(state["theta"], delta, state["loss"],
                                                trial_loss[:m], trial_loss[m:], active)
        hard = torch.where(plus, trial_hard[:m], torch.where(minus, trial_hard[m:], state["hard"]))
        state.update(theta=theta, loss=loss, hard=hard, iteration=iteration)
        state["accepted"] += plus | minus
        capture_nodes(state, loss, hard, iteration, cfg)
        if iteration % 200 == 0 or iteration == steps:
            torch.cuda.synchronize(x.device)
            state["elapsed"] = previous + time.perf_counter() - tick
            row = {"iteration": iteration, "elapsed": state["elapsed"],
                   "loss_median": float(loss.median()), "loss_p90": float(torch.quantile(loss, .9)),
                   "hard_fit": int(hard.sum())}
            if capture:
                row["node_reached"] = state["seen"].sum(0).tolist()
            state["trace"].append(row)
            if time.perf_counter() - logged >= 20 or iteration == steps:
                log(out, stage, json.dumps(row, ensure_ascii=False))
                logged = time.perf_counter()
        if iteration % cfg["save_every"] == 0:
            state["rng"] = gen.get_state()
            state["elapsed"] = previous + time.perf_counter() - tick
            atomic_torch(path, cpu_state(state))
    torch.cuda.synchronize(x.device)
    state["elapsed"] = previous + time.perf_counter() - tick
    state["rng"] = gen.get_state()
    state["finished"] = not STOP
    state["stop_reason"] = ("interrupt" if STOP else "all_reached" if capture and state["seen"][:, -1].all()
                            else "max_steps" if state["iteration"] >= steps else "time_budget")
    if torch.count_nonzero(state["theta"][:, ref.BIAS_INDICES]):
        raise RuntimeError("固定bias发生变化")
    atomic_torch(path, cpu_state(state))
    log(out, stage, f"结束原因={state['stop_reason']}，步数={state['iteration']}，"
        f"训练BCE中位={float(state['loss'].median()):.7g}，耗时={state['elapsed']:.1f}秒")
    return state


def run_sgd(ref, cfg, out, initial, orders, x, y):
    path = out / "sgd_state.pt"
    state = restore(torch.load(path, map_location="cpu", weights_only=True), x.device) if path.exists() else new_state(initial, cfg, True)
    if state.get("finished"):
        return state
    theta = state["theta"].requires_grad_(True)
    opt = torch.optim.SGD([theta], lr=cfg["sgd_lr"])
    tick, previous = time.perf_counter(), state["elapsed"]
    with torch.no_grad():
        loss, hard, _ = ref.score(ref.forward(theta, x), y)
        capture_nodes(state, loss, hard, state["iteration"], cfg)
    for epoch in range(state["iteration"] + 1, cfg["sgd_epochs"] + 1):
        if STOP:
            break
        for ids in orders[epoch - 1].split(cfg["sgd_batch"]):
            ids = ids.to(x.device)
            opt.zero_grad(set_to_none=True)
            ref.native_ce(ref.forward(theta, x[ids]), y[ids]).sum().backward()
            theta.grad[:, ref.BIAS_INDICES] = 0
            opt.step()
        with torch.no_grad():
            loss, hard, _ = ref.score(ref.forward(theta, x), y)
            if not torch.isfinite(loss).all():
                raise RuntimeError("SGD出现非有限loss，保留失败而不静默删除")
            state["iteration"] = epoch
            capture_nodes(state, loss, hard, epoch, cfg)
            state["trace"].append({"epoch": epoch, "loss_median": float(loss.median()),
                                   "node_reached": state["seen"].sum(0).tolist()})
    torch.cuda.synchronize(x.device)
    state["theta"] = theta.detach()
    state["elapsed"] = previous + time.perf_counter() - tick
    state["loss"], state["hard"] = loss, hard
    state["finished"] = not STOP
    atomic_torch(path, cpu_state(state))
    log(out, "sgd", f"完成{state['iteration']}epoch，节点到达={state['seen'].sum(0).tolist()}")
    return state


def freeze_states(out, cfg):
    names = ["pilot.json", "stp_state.pt", "sgd_state.pt", "manifest.json"]
    freeze = {"files": {name: sha(out / name) for name in names}, "created_at": time.time(),
              "policy": "先冻结训练及步长选择，再导出测试预测；不使用测试标签选择设置。"}
    path = out / "TRAINING_FROZEN.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["files"] != freeze["files"]:
            raise RuntimeError("已冻结训练结果变化")
    else:
        atomic_json(path, freeze)


@torch.no_grad()
def export_predictions(ref, cfg, out, states):
    with np.load(REFERENCE / "results/dataset.npz", allow_pickle=False) as f:
        tx = torch.from_numpy(f["test_x"].copy()).to(cfg["device"])
    manifest = {}
    for name, state in states.items():
        for j, epsilon in enumerate(cfg["nodes"]):
            ids = torch.where(state["seen"][:, j])[0]
            weights = state["node_weights"][j, ids]
            if len(ids):
                logits = ref.evaluate_logits(weights, tx, 32, 128)
                prediction = logits.argmax(-1).cpu().numpy().astype(np.uint8)
            else:
                prediction = np.empty((0, len(tx)), dtype=np.uint8)
            path = out / (name + "_" + node_name(epsilon) + ".npz")
            atomic_npz(path, init_id=ids.cpu().numpy(), prediction=prediction,
                       train_loss=state["node_loss"][ids, j].cpu().numpy(),
                       iteration=state["node_iteration"][ids, j].cpu().numpy())
            manifest[path.name] = sha(path)
    atomic_json(out / "PREDICTIONS_FROZEN.json", {"files": manifest, "training_sha256": sha(out / "TRAINING_FROZEN.json")})


def main(cfg=CONFIG):
    if cfg["mode"] not in ("all", "pilot"):
        raise ValueError("mode必须为all或pilot")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    ref = load_reference()
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("必须使用CUDA，不自动回退CPU")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    oldfiles = [REFERENCE / "results/smc" / f"replica_{r}" / (label + suffix)
                for r in range(4) for label in ["hard_start"] + [node_name(v) for v in cfg["nodes"]]
                for suffix in [".npz", ".json"]]
    scientific = {key: value for key, value in cfg.items() if key not in ("mode", "output_dir")}
    manifest = {"config": scientific, "source": sha(__file__), "reference": EXPECTED,
                "baseline_files": {str(p.relative_to(REFERENCE)): sha(p) for p in oldfiles},
                "parameter_count": ref.PARAMETERS, "trainable_parameters": len(ref.WEIGHT_INDICES),
                "hardware": {"gpu": torch.cuda.get_device_name(device), "device": str(device),
                             "memory": torch.cuda.get_device_properties(device).total_memory,
                             "torch": str(torch.__version__), "cuda": torch.version.cuda,
                             "python": platform.python_version()},
                "dtype": {"network": "float32", "loss": "float64", "tf32": False, "amp": False}}
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("源码/设置/数据/环境不同，请使用新的输出目录")
    if not path.exists():
        atomic_json(path, manifest)
    with np.load(REFERENCE / "results/dataset.npz", allow_pickle=False) as f:
        x = torch.from_numpy(f["train_x"].copy()).to(device)
        y = torch.from_numpy(f["train_y"].copy()).to(device)
    initial = torch.load(REFERENCE / "results/initializations.pt", map_location="cpu", weights_only=True)
    base = initial["base_theta"].to(device) * cfg["prior_scale"]
    if cfg["models"] > min(cfg["pilot_initial_ids"]):
        raise ValueError("pilot与正式初始化发生重叠")
    pilot_path = out / "pilot.json"
    if not pilot_path.exists():
        pilot_base = base[cfg["pilot_initial_ids"]].repeat(len(cfg["pilot_sigmas"]), 1)
        scales = torch.tensor(cfg["pilot_sigmas"], device=device).repeat_interleave(len(cfg["pilot_initial_ids"]))
        log(out, "pilot", f"GPU={manifest['hardware']['gpu']}，{ref.PARAMETERS}参数，4种步长各8个初始化；只看训练loss")
        pilot = run_stp(ref, cfg, out, "pilot", pilot_base, scales, x, y, cfg["pilot_seed"], False)
        if STOP:
            return
        losses = pilot["loss"].view(len(cfg["pilot_sigmas"]), -1)
        scores = losses.clamp_min(1e-300).log().median(1).values
        selected = int(scores.argmin())
        atomic_json(pilot_path, {"sigma": cfg["pilot_sigmas"][selected], "selection": "最小训练BCE中位对数",
                                "losses": losses.cpu().tolist(), "candidates": cfg["pilot_sigmas"],
                                "iterations": pilot["iteration"], "seconds": pilot["elapsed"]})
        del pilot
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    log(out, "pilot_selected", f"训练集选择sigma0={pilot['sigma']}；尚未读取测试标签")
    if cfg["mode"] == "pilot":
        return
    stp = run_stp(ref, cfg, out, "stp", base[:cfg["models"]], pilot["sigma"], x, y, cfg["stp_seed"], True)
    if STOP:
        return
    sgd = run_sgd(ref, cfg, out, base[:cfg["models"]], initial["epoch_permutations"], x, y)
    if STOP:
        return
    freeze_states(out, cfg)
    export_predictions(ref, cfg, out, {"stp": stp, "sgd": sgd})
    log(out, "completed", "训练及预测已冻结；请用analyze.py独立读取测试标签评分。")


if __name__ == "__main__":
    main()
