"""4bit copy：在同一n=8数据上补测原始MSE的正负0.5和正负0.3目标。"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).with_name("experiment_4bit_copy_zero_one_mse_local.py")
spec = importlib.util.spec_from_file_location("copy_mse_previous", HELPER)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
OLD = ROOT / "tmp/4bit_copy_zero_one_mse_20260910"


class Config:
    RESULT_DIR = ROOT / "tmp/4bit_copy_centered_mse_20260910"
    GROUPS = (("mse_pm0p5_n8", 0.5), ("mse_pm0p3_n8", 0.3))
    STEPS = 18000
    LR = 0.01
    MILESTONES = (2000, 5000, 9000, 13000, 16000)
    DECAY = 0.3
    ADAM_EPS = 1e-8
    EVAL_EVERY = 200
    CHECKPOINT_EVERY = 1000
    RELATIVE_FIT_THRESHOLD = 1e-4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    old_meta = json.loads((OLD / "model_metadata.json").read_text(encoding="utf-8"))
    bce_ids = [m["model_index"] for m in old_meta if m["group"] == "bce_n8"]
    mse_ids = [m["model_index"] for m in old_meta if m["group"] == "mse01_n8"]
    with np.load(OLD / "results.npz", allow_pickle=False) as p:
        x, y = torch.tensor(p["inputs"]), torch.tensor(p["labels"])
        initial = torch.tensor(p["initial_weights"][bce_ids])
        mask = torch.tensor(p["train_mask"][bce_ids])
        assert np.array_equal(p["initial_weights"][bce_ids], p["initial_weights"][mse_ids])
        assert np.array_equal(p["train_mask"][bce_ids], p["train_mask"][mse_ids])
        train, test = p["train_ids_n8"].copy(), p["test_ids_n8"].copy()
    assert len(bce_ids) == 512 and int(mask.sum()) == 8 * 512
    metadata = []
    for name, amplitude in Config.GROUPS:
        for old_id in bce_ids:
            metadata.append(dict(model_index=len(metadata), group=name, loss_type="mse",
                amplitude=amplitude, center=0., train_count=8,
                split_id=old_meta[old_id]["split_id"], seed=old_meta[old_id]["seed"]))
    amplitudes = torch.tensor([m["amplitude"] for m in metadata], dtype=torch.float64)
    mask = mask.repeat(len(Config.GROUPS), 1)
    targets = torch.where(mask, amplitudes[:, None] * (2 * y[None] - 1), 0.)
    return dict(x=x, y=y, initial=initial.repeat(len(Config.GROUPS), 1), train_mask=mask,
        targets=targets, amplitudes=amplitudes, metadata=metadata, train_ids=train, test_ids=test,
        is_bce=torch.zeros(len(metadata), dtype=torch.bool), counts=mask.sum(1).double(),
        threshold=torch.zeros(len(metadata), dtype=torch.float64),
        goals=amplitudes.square() * Config.RELATIVE_FIT_THRESHOLD)


def objective(z, data):
    # 与此前0/1回归相同，优化原始MSE；相对误差只用于记录和拟合门槛。
    return ((z - data["targets"]).square() * data["train_mask"]).sum(1) / data["counts"]


def evaluate(theta, data):
    with torch.no_grad():
        z = base.forward(theta, data["x"][None].expand(len(theta), -1, -1))
        pred = z > 0
        correct = pred == data["y"].bool()[None]
        mask = data["train_mask"]
        return dict(outputs=z, objective=objective(z, data),
            train_accuracy=(correct * mask).sum(1) / data["counts"],
            unseen_accuracy=(correct * ~mask).sum(1) / (16 - data["counts"]),
            full_recovery=correct.all(1),
            function_ids=(pred.long() * 2 ** torch.arange(16)[None]).sum(1))


def preflight(data):
    p = data["initial"].clone().requires_grad_(True)
    x = data["x"][None].expand(len(p), -1, -1)
    v = objective(base.forward(p, x), data)
    g = torch.autograd.grad(v.sum(), p)[0]
    errors = []
    for i in (0, 511, 512, 1023):
        ids = data["train_mask"][i].numpy()
        value, grad = previous.numpy_value_gradient(p[i].detach().numpy(),
            data["x"].numpy()[ids], data["targets"][i].numpy()[ids], False)
        err = float(abs(g[i].numpy() - grad).max())
        assert err < 1e-12 and abs(value - v[i].item()) < 1e-12
        errors.append(err)
    alt = dict(data)
    alt["targets"] = torch.where(data["train_mask"], data["targets"], 99.)
    alt_v = objective(base.forward(p, x), alt)
    alt_g = torch.autograd.grad(alt_v.sum(), p)[0]
    assert torch.equal(v, alt_v) and torch.equal(g, alt_g)
    for s in range(16):
        tr, te = data["train_ids"][s], data["test_ids"][s]
        assert len(set(tr)) == len(set(te)) == 8 and not set(tr).intersection(te)
    assert torch.equal(data["initial"][:512], data["initial"][512:])
    assert len({m["seed"] for m in data["metadata"][:512]}) == 512

    # 输出bias在forward中除以4；原始bias加2、目标加0.5应给出同一MSE问题。
    translated = data["initial"].clone()
    translated[:, -1] += 2
    translated.requires_grad_(True)
    shifted_data = dict(data)
    shifted_data["targets"] = torch.where(data["train_mask"], data["targets"] + .5, 0.)
    shifted_v = objective(base.forward(translated, x), shifted_data)
    shifted_g = torch.autograd.grad(shifted_v.sum(), translated)[0]
    center_error = float((shifted_v - v).abs().max())
    center_gradient_error = float((shifted_g - g).abs().max())
    assert center_error < 1e-12 and center_gradient_error < 1e-12

    small = {k: (value[:4] if isinstance(value, torch.Tensor) and value.shape[0] == len(p) else value)
             for k, value in data.items()}
    t = torch.nn.Parameter(data["initial"][:4].clone())
    opt = torch.optim.Adam([t], lr=Config.LR, eps=Config.ADAM_EPS)
    def update(weights, optimizer):
        optimizer.zero_grad(set_to_none=True)
        objective(base.forward(weights, data["x"][None].expand(4, -1, -1)), small).sum().backward()
        optimizer.step()
    update(t, opt)
    buffer = io.BytesIO()
    torch.save(dict(weights=t.detach(), optimizer=opt.state_dict()), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    resumed = torch.nn.Parameter(loaded["weights"].clone())
    opt2 = torch.optim.Adam([resumed], lr=Config.LR, eps=Config.ADAM_EPS)
    opt2.load_state_dict(loaded["optimizer"])
    update(t, opt)
    update(resumed, opt2)
    assert torch.equal(t, resumed)
    return dict(independent_gradient_errors=errors, heldout_labels_do_not_affect_loss_or_gradient=True,
        exact_paired_data_and_initialization=True, classification_threshold=0.,
        center_translation_loss_error=center_error, center_translation_gradient_error=center_gradient_error,
        adam_resume_bitwise_equal=True)


def summarize(values, data):
    rows = []
    for name, amplitude in Config.GROUPS:
        ids = [m["model_index"] for m in data["metadata"] if m["group"] == name]
        obj = values["objective"][ids]
        train_acc = values["train_accuracy"][ids]
        rows.append(dict(group=name, amplitude=amplitude, models=len(ids),
            hard_fit=int((train_acc == 1).sum()),
            sufficiently_fitted=int(((train_acc == 1) & (obj <= data["goals"][ids])).sum()),
            maximum_objective=float(obj.max()), median_objective=float(obj.median()),
            maximum_relative_mse=float(obj.max()) / amplitude ** 2,
            unseen_accuracy=float(values["unseen_accuracy"][ids].mean()),
            full_recovery_count=int(values["full_recovery"][ids].sum()),
            full_recovery_fraction=float(values["full_recovery"][ids].double().mean())))
    return rows


def record(theta, state, data):
    values = evaluate(theta, data)
    assert torch.isfinite(values["objective"]).all()
    better = values["objective"] < state["best_objective"]
    state["best_weights"][better] = theta.detach()[better]
    state["best_objective"][better] = values["objective"][better]
    state["best_step"][better] = state["step"]
    fit = ((values["train_accuracy"] == 1) & (values["objective"] <= data["goals"]) & (state["fit_step"] < 0))
    state["fit_weights"][fit] = theta.detach()[fit]
    state["fit_step"][fit] = state["step"]
    state["fit_seconds"][fit] = state["seconds"]
    state["history_steps"].append(state["step"])
    state["history_seconds"].append(state["seconds"])
    for key in ("outputs", "objective", "train_accuracy", "unseen_accuracy", "full_recovery"):
        state["history_" + key].append(values[key].clone())
    progress = dict(status="running", step=state["step"], total_steps=Config.STEPS,
        training_loop_seconds=state["seconds"], groups=summarize(values, data))
    base.write_json(Config.RESULT_DIR / "progress.json", progress)
    if state["step"] % 2000 == 0:
        print(json.dumps(progress, ensure_ascii=False), flush=True)


def checkpoint(theta, optimizer, state, signature):
    temp = Config.RESULT_DIR / "checkpoint.pt.tmp"
    torch.save(dict(signature=signature, weights=theta.detach(), optimizer=optimizer.state_dict(),
        state=state, torch_rng=torch.get_rng_state()), temp)
    temp.replace(Config.RESULT_DIR / "checkpoint.pt")


def save_results(theta, state, data, signature):
    rows, summaries = [], {}
    for stage, weights in (("final", theta.detach()), ("best_train", state["best_weights"]),
                            ("first_fit", state["fit_weights"])):
        valid = torch.isfinite(weights).all(1)
        values = evaluate(torch.nan_to_num(weights), data)
        if stage != "first_fit":
            summaries[stage] = summarize(values, data)
        for i, meta in enumerate(data["metadata"]):
            if not valid[i]:
                continue
            step = state["step"] if stage == "final" else int(state["best_step" if stage == "best_train" else "fit_step"][i])
            obj = float(values["objective"][i])
            rows.append(dict(**meta, stage=stage, step=step, objective=obj,
                relative_mse=obj / meta["amplitude"] ** 2,
                train_accuracy=float(values["train_accuracy"][i]), unseen_accuracy=float(values["unseen_accuracy"][i]),
                function_id=int(values["function_ids"][i]), full_recovery=bool(values["full_recovery"][i]),
                sufficiently_fitted=bool(values["train_accuracy"][i] == 1 and obj <= data["goals"][i])))
    arrays = dict(inputs=data["x"].numpy(), labels=data["y"].numpy(), initial_weights=data["initial"].numpy(),
        train_mask=data["train_mask"].numpy(), targets=data["targets"].numpy(), amplitudes=data["amplitudes"].numpy(),
        decision_thresholds=data["threshold"].numpy(), train_ids_n8=data["train_ids"], test_ids_n8=data["test_ids"],
        final_weights=theta.detach().numpy(), best_train_weights=state["best_weights"].numpy(),
        best_step=state["best_step"].numpy(), first_fit_weights=state["fit_weights"].numpy(),
        first_fit_step=state["fit_step"].numpy(), first_fit_seconds=state["fit_seconds"].numpy(),
        history_steps=np.array(state["history_steps"]), history_seconds=np.array(state["history_seconds"]))
    for key in ("outputs", "objective", "train_accuracy", "unseen_accuracy", "full_recovery"):
        arrays["history_" + key] = torch.stack(state["history_" + key]).numpy()
    np.savez_compressed(Config.RESULT_DIR / "results.npz", **arrays)
    base.write_csv(Config.RESULT_DIR / "per_model_results.csv", rows)
    base.write_json(Config.RESULT_DIR / "model_metadata.json", data["metadata"])
    summary = dict(status="completed", step=state["step"], signature=signature,
        training_loop_seconds=state["seconds"], model_count=len(data["metadata"]), groups=summaries,
        scope="仅新增n=8的原始MSE正负0.5和正负0.3；既有0/1 MSE与BCE不重跑、不改写。")
    base.write_json(Config.RESULT_DIR / "summary.json", summary)
    base.write_json(Config.RESULT_DIR / "progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = setup()
    old_config = json.loads((OLD / "config.json").read_text(encoding="utf-8"))
    assert sha(HELPER) == old_config["script_sha256"]
    assert sha(previous.BASE_SCRIPT) == old_config["base_script_sha256"]
    settings = dict(groups=Config.GROUPS, train_count=8, independent_seeds=512, splits=16, seeds_per_split=32,
        seeds=[m["seed"] for m in data["metadata"][:512]], network="4->16->1 tanh, 97 raw standard Gaussian parameters, all trainable",
        input_encoding="0/1", target="a * (2*y-1)", output="linear, no sigmoid", decision_threshold=0.,
        objective="raw MSE without amplitude normalization", relative_fit_threshold=Config.RELATIVE_FIT_THRESHOLD,
        optimizer="Adam", lr=Config.LR, milestones=Config.MILESTONES, decay=Config.DECAY,
        adam_eps=Config.ADAM_EPS, weight_decay=0., steps=Config.STEPS, eval_every=Config.EVAL_EVERY,
        checkpoint_every=Config.CHECKPOINT_EVERY, batch="full train set per model; model objectives summed",
        dtype="float64", device="cpu", test_usage="evaluation only, no test-based selection",
        old_results_sha256=sha(OLD / "results.npz"), old_config_sha256=sha(OLD / "config.json"),
        helper_script_sha256=sha(HELPER), base_script_sha256=sha(previous.BASE_SCRIPT), script_sha256=sha(Path(__file__)))
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cfg = dict(settings=settings, signature=signature, torch=str(torch.__version__), numpy=np.__version__)
    cfg_path = Config.RESULT_DIR / "config.json"
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text(encoding="utf-8"))["signature"] == signature, "协议或源码变化，拒绝复用检查点"
    else:
        base.write_json(cfg_path, cfg)
    done = Config.RESULT_DIR / "summary.json"
    if done.exists() and json.loads(done.read_text(encoding="utf-8")).get("status") == "completed":
        print(done.read_text(encoding="utf-8"))
        return
    base.write_json(Config.RESULT_DIR / "preflight.json", preflight(data))
    theta = torch.nn.Parameter(data["initial"].clone())
    optimizer = torch.optim.Adam([theta], lr=Config.LR, eps=Config.ADAM_EPS, weight_decay=0.)
    m = len(theta)
    state = dict(step=0, seconds=0., best_weights=theta.detach().clone(),
        best_objective=torch.full((m,), float("inf"), dtype=torch.float64), best_step=torch.full((m,), -1, dtype=torch.long),
        fit_weights=torch.full_like(theta, float("nan")), fit_step=torch.full((m,), -1, dtype=torch.long),
        fit_seconds=torch.full((m,), float("nan"), dtype=torch.float64))
    for key in ("steps", "seconds", "outputs", "objective", "train_accuracy", "unseen_accuracy", "full_recovery"):
        state["history_" + key] = []
    saved = Config.RESULT_DIR / "checkpoint.pt"
    if saved.exists():
        old = torch.load(saved, map_location="cpu", weights_only=True)
        assert old["signature"] == signature
        theta.data.copy_(old["weights"])
        optimizer.load_state_dict(old["optimizer"])
        torch.set_rng_state(old["torch_rng"])
        state = old["state"]
    else:
        record(theta, state, data)
    stop = [False]
    def interrupt_handler(signum, frame):
        stop[0] = True
    original_handler = signal.signal(signal.SIGINT, interrupt_handler)
    started, previous_seconds = time.perf_counter(), state["seconds"]
    inputs = data["x"][None].expand(m, -1, -1)
    try:
        for step in range(state["step"] + 1, Config.STEPS + 1):
            optimizer.param_groups[0]["lr"] = Config.LR * Config.DECAY ** sum(step > k for k in Config.MILESTONES)
            optimizer.zero_grad(set_to_none=True)
            objective(base.forward(theta, inputs), data).sum().backward()
            optimizer.step()
            state["step"] = step
            state["seconds"] = previous_seconds + time.perf_counter() - started
            if step % Config.EVAL_EVERY == 0 or stop[0]:
                record(theta, state, data)
            if step % Config.CHECKPOINT_EVERY == 0 or stop[0]:
                checkpoint(theta, optimizer, state, signature)
            if stop[0]:
                base.write_json(Config.RESULT_DIR / "progress.json", dict(status="paused", step=step, seconds=state["seconds"]))
                print("已在完整更新边界保存，再次运行可继续。", flush=True)
                return
    finally:
        signal.signal(signal.SIGINT, original_handler)
    checkpoint(theta, optimizer, state, signature)
    save_results(theta, state, data, signature)


if __name__ == "__main__":
    main()
