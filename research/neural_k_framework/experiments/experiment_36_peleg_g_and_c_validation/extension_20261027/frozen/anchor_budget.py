"""复用冻结STP核心，独立保存步长和loss深度对照；不改历史文件。"""

import copy
import json
from pathlib import Path
import platform
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "gradient_free_control_20261024"
sys.path.insert(0, str(PREVIOUS))
import run_experiment as engine
import numpy as np
import torch


CONFIG = {
    "device": "cuda:1", "models": 128,
    "sigmas": [0.01, 0.003, 0.001, 0.0003],
    "nodes": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    "stp_steps": 60000, "stp_seconds": 900,
    "decay_steps": 2000, "stp_seed": 202610242,
    "sgd_epochs": 300, "sgd_lr": 0.01, "sgd_batch": 2,
    "prior_scale": 0.2,
}
CORE_HASH = "06880279e978b4c62b9a4d728f377c28a745c4666652e36c64290c50e8b8d3dc"
OLD_HASHES = {
    "stp_state.pt": "083b077892e3fbe69c4faffe0a90d64109abca130c738e57c6080858df0f7f35",
    "sgd_state.pt": "2a3fcfd228c568b41b2390720a459ada3133badefcc28188b467ac9da6949cb1",
}


def group_name(sigma):
    return "stp_sigma_" + format(sigma, "g").replace(".", "p")


def validate_old():
    if engine.sha(PREVIOUS / "run_experiment.py") != CORE_HASH:
        raise RuntimeError("冻结训练核心已变化")
    for name, expected in OLD_HASHES.items():
        if engine.sha(PREVIOUS / "results" / name) != expected:
            raise RuntimeError("历史训练结果变化：" + name)


def verify_prefix(state, method, out):
    old = torch.load(PREVIOUS / "results" / (method + "_state.pt"), map_location="cpu", weights_only=True)
    weights = state["node_weights"][:5].detach().cpu()
    losses = state["node_loss"][:, :5].detach().cpu()
    iterations = state["node_iteration"][:, :5].detach().cpu()
    report = {
        "method": method,
        "weights_bitwise_equal": torch.equal(weights, old["node_weights"]),
        "loss_bitwise_equal": torch.equal(losses, old["node_loss"]),
        "iteration_equal": torch.equal(iterations, old["node_iteration"]),
        "max_weight_abs_error": float((weights-old["node_weights"]).abs().max()),
        "max_loss_abs_error": float((losses-old["node_loss"]).abs().max()),
    }
    engine.atomic_json(out / (method + "_prefix_audit.json"), report)
    if not all(report[key] for key in ("weights_bitwise_equal", "loss_bitwise_equal", "iteration_equal")):
        raise RuntimeError("原节点路径未精确复现，停止对账：" + json.dumps(report))


def checked_manifest(out, cfg):
    files = {"source": engine.sha(__file__), "core": CORE_HASH,
             "previous": OLD_HASHES, "reference": engine.EXPECTED}
    for r in range(4):
        for label in ["hard_start"] + [engine.node_name(v) for v in cfg["nodes"]]:
            path = engine.REFERENCE / "results/smc" / f"replica_{r}" / (label + ".npz")
            files[str(path.relative_to(engine.REFERENCE))] = engine.sha(path)
    meta = {"config": cfg, "files": files, "hardware": {
        "device": cfg["device"], "gpu": torch.cuda.get_device_name(cfg["device"]),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "python": platform.python_version(), "network_dtype": "float32", "bce_dtype": "float64",
        "tf32": False, "amp": False}}
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != meta:
            raise RuntimeError("运行配置/源码/基线/设备变化；请另用结果目录")
    else:
        engine.atomic_json(path, meta)


def freeze(out, states):
    obj = {"files": {"manifest.json": engine.sha(out / "manifest.json")}, "created_at": time.time()}
    for group, method, _ in states:
        path = group / (method + "_state.pt")
        obj["files"][str(path.relative_to(out))] = engine.sha(path)
    path = out / "TRAINING_FROZEN.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["files"] != obj["files"]:
        raise RuntimeError("已冻结训练状态变化")
    if not path.exists():
        engine.atomic_json(path, obj)
    for group, method, _ in states:
        engine.atomic_json(group / "TRAINING_FROZEN.json", {"files": {
            method + "_state.pt": engine.sha(group / (method + "_state.pt"))},
            "parent_sha256": engine.sha(path)})


def main():
    validate_old()
    ref = engine.load_reference()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, engine.interrupted)
    cfg = {**copy.deepcopy(engine.CONFIG), **CONFIG}
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("需要CUDA，不回退CPU")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    checked_manifest(out, CONFIG)
    with np.load(engine.REFERENCE / "results/dataset.npz", allow_pickle=False) as f:
        x = torch.from_numpy(f["train_x"].copy()).to(device)
        y = torch.from_numpy(f["train_y"].copy()).to(device)
    initial = torch.load(engine.REFERENCE / "results/initializations.pt", map_location="cpu", weights_only=True)
    base = initial["base_theta"][:cfg["models"]].to(device) * cfg["prior_scale"]
    states = []
    for sigma in cfg["sigmas"]:
        folder = out / group_name(sigma)
        folder.mkdir(exist_ok=True)
        engine.log(out, folder.name, f"128个初始化，sigma0={sigma}，目标BCE={cfg['nodes'][-1]}")
        state = engine.run_stp(ref, cfg, folder, "stp", base, sigma, x, y, cfg["stp_seed"], True)
        if engine.STOP:
            return
        if sigma == .01:
            verify_prefix(state, "stp", out)
        states.append((folder, "stp", state))
    orders = torch.stack([torch.randperm(16, generator=torch.Generator().manual_seed(19092027 + epoch))
                          for epoch in range(cfg["sgd_epochs"])])
    if not torch.equal(orders[:60], initial["epoch_permutations"]):
        raise RuntimeError("SGD前60epoch批次顺序不同")
    folder = out / "sgd"
    folder.mkdir(exist_ok=True)
    state = engine.run_sgd(ref, cfg, folder, base, orders, x, y)
    if engine.STOP:
        return
    verify_prefix(state, "sgd", out)
    states.append((folder, "sgd", state))
    freeze(out, states)
    for folder, method, state in states:
        engine.export_predictions(ref, cfg, folder, {method: state})
    engine.log(out, "completed", "全部训练和预测已冻结；尚未读取新测试评分，分析入口analyze_budget.py。")


if __name__ == "__main__":
    main()
