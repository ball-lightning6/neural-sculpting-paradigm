"""4bit copy：BCE加训练集正确方向logit的方差正则，三档系数配对比较。"""
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).with_name("experiment_4bit_copy_zero_one_mse_local.py")
spec = importlib.util.spec_from_file_location("copy_bce_previous", HELPER)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
OLD = ROOT / "tmp/4bit_copy_zero_one_mse_20260910"


class Config:
    RESULT_DIR = ROOT / "tmp/4bit_copy_bce_alignment_20260910"
    GROUPS = (("bce_var0p01", 0.01), ("bce_var0p1", 0.1), ("bce_var1", 1.0))
    STEPS = 18000
    LR = 0.01
    MILESTONES = (2000, 5000, 9000, 13000, 16000)
    DECAY = 0.3
    ADAM_EPS = 1e-8
    EVAL_EVERY = 200
    CHECKPOINT_EVERY = 1000
    BCE_FIT_THRESHOLD = 1e-3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    metadata_old = json.loads((OLD / "model_metadata.json").read_text(encoding="utf-8"))
    ids = [m["model_index"] for m in metadata_old if m["group"] == "bce_n8"]
    with np.load(OLD / "results.npz", allow_pickle=False) as p:
        x, y = torch.tensor(p["inputs"]), torch.tensor(p["labels"])
        initial = torch.tensor(p["initial_weights"][ids])
        mask = torch.tensor(p["train_mask"][ids])
        train, test = p["train_ids_n8"].copy(), p["test_ids_n8"].copy()
    assert initial.shape == (512, 97) and (mask.sum(1) == 8).all()
    metadata = []
    for name, lam in Config.GROUPS:
        for old_id in ids:
            metadata.append(dict(model_index=len(metadata), group=name, regularization=lam,
                loss_type="bce_plus_signed_logit_variance", train_count=8,
                seed=metadata_old[old_id]["seed"], split_id=metadata_old[old_id]["split_id"]))
    mask = mask.repeat(len(Config.GROUPS), 1)
    targets = torch.where(mask, y[None].expand(len(metadata), -1), 0.)
    return dict(x=x, y=y, initial=initial.repeat(len(Config.GROUPS), 1), train_mask=mask,
        targets=targets, counts=mask.sum(1).double(), train_ids=train, test_ids=test,
        metadata=metadata, lambdas=torch.tensor([m["regularization"] for m in metadata], dtype=torch.float64))


def components(z, data):
    mask, count = data["train_mask"], data["counts"]
    point_bce = F.binary_cross_entropy_with_logits(z, data["targets"], reduction="none")
    bce = (point_bce * mask).sum(1) / count
    signed = (2 * data["targets"] - 1) * z
    mean = (signed * mask).sum(1) / count
    # 均值保留梯度；只约束训练样本，采用除以n的总体方差。
    variance = ((signed - mean[:, None]).square() * mask).sum(1) / count
    return dict(objective=bce + data["lambdas"] * variance, raw_bce=bce,
                amplitude_variance=variance, mean_amplitude=mean)


def numpy_value_gradient(theta, x, y, lam):
    hidden = theta[:80].reshape(16, 5)
    h = np.tanh((x @ hidden[:, :4].T + hidden[None, :, 4]) / 2)
    z = (h @ theta[80:96] + theta[96]) / 4
    sign = 2 * y - 1
    m = sign * z
    bce = np.logaddexp(0, -m).mean()
    variance = np.mean((m - m.mean()) ** 2)
    dz = (np.exp(-np.logaddexp(0, -z)) - y + 2 * lam * sign * (m - m.mean())) / len(y)
    grad = np.zeros(97)
    grad[80:96] = h.T @ dz / 4
    grad[96] = dz.sum() / 4
    dh = dz[:, None] * theta[None, 80:96] * (1 - h ** 2) / 8
    grad[:80].reshape(16, 5)[:, :4] = dh.T @ x
    grad[:80].reshape(16, 5)[:, 4] = dh.sum(0)
    return float(bce + lam * variance), grad


def preflight(data):
    p = data["initial"].clone().requires_grad_(True)
    x = data["x"][None].expand(len(p), -1, -1)
    value = components(base.forward(p, x), data)["objective"]
    grad = torch.autograd.grad(value.sum(), p)[0]
    errors = []
    for i in (0, 511, 512, 1023, 1024, 1535):
        mask = data["train_mask"][i].numpy()
        expected_value, expected_grad = numpy_value_gradient(p[i].detach().numpy(), data["x"].numpy()[mask],
            data["targets"][i].numpy()[mask], float(data["lambdas"][i]))
        err = float(abs(grad[i].numpy() - expected_grad).max())
        assert err < 1e-12 and abs(expected_value - float(value[i].detach())) < 1e-12
        errors.append(err)
    alt = dict(data)
    alt["targets"] = torch.where(data["train_mask"], data["targets"], 1 - data["targets"])
    alternate = components(base.forward(p, x), alt)["objective"]
    alternate_grad = torch.autograd.grad(alternate.sum(), p)[0]
    assert torch.equal(value, alternate) and torch.equal(grad, alternate_grad)
    zero = dict(data, lambdas=torch.zeros_like(data["lambdas"]))
    zero_value = components(base.forward(p, x), zero)["objective"]
    zero_grad = torch.autograd.grad(zero_value.sum(), p)[0]
    old_data = dict(data, is_bce=torch.ones(len(p), dtype=torch.bool))
    old_value = previous.objective(base.forward(p, x), old_data)
    old_grad = torch.autograd.grad(old_value.sum(), p)[0]
    assert torch.equal(zero_value, old_value) and torch.equal(zero_grad, old_grad)

    z = base.forward(p.detach(), x)
    sign = 2 * data["targets"] - 1
    original = components(z, data)
    shifted = components(z + 3 * sign, data)
    shift_error = float((original["amplitude_variance"] - shifted["amplitude_variance"]).abs().max())
    assert shift_error < 1e-12
    perfect = components(3 * sign, data)
    assert torch.equal(perfect["amplitude_variance"], torch.zeros_like(data["lambdas"]))
    assert torch.allclose(shifted["mean_amplitude"] - original["mean_amplitude"], torch.full_like(data["lambdas"], 3))
    for start in (512, 1024):
        assert torch.equal(data["initial"][:512], data["initial"][start:start + 512])
        assert torch.equal(data["train_mask"][:512], data["train_mask"][start:start + 512])
    for split in range(16):
        tr, te = data["train_ids"][split], data["test_ids"][split]
        assert len(set(tr)) == len(set(te)) == 8 and not set(tr).intersection(te)

    pick = torch.tensor([0, 512, 1024])
    small = {k: (v[pick] if isinstance(v, torch.Tensor) and v.shape[0] == len(p) else v) for k, v in data.items()}
    t = torch.nn.Parameter(data["initial"][pick].clone())
    opt = torch.optim.Adam([t], lr=Config.LR, eps=Config.ADAM_EPS)
    def update(weights, optimizer):
        optimizer.zero_grad(set_to_none=True)
        components(base.forward(weights, data["x"][None].expand(3, -1, -1)), small)["objective"].sum().backward()
        optimizer.step()
    update(t, opt)
    buffer = io.BytesIO()
    torch.save(dict(weights=t.detach(), optimizer=opt.state_dict()), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    t2 = torch.nn.Parameter(restored["weights"].clone())
    opt2 = torch.optim.Adam([t2], lr=Config.LR, eps=Config.ADAM_EPS)
    opt2.load_state_dict(restored["optimizer"])
    update(t, opt)
    update(t2, opt2)
    assert torch.equal(t, t2)
    return dict(independent_gradient_errors=errors, heldout_labels_do_not_affect_loss_or_gradient=True,
        zero_coefficient_matches_original_bce_bitwise=True, exact_paired_initialization_and_data=True,
        signed_equal_amplitude_has_zero_penalty=True, penalty_invariant_to_common_amplitude_shift_error=shift_error,
        no_fixed_amplitude_target=True, adam_resume_bitwise_equal=True)


def evaluate(theta, data):
    with torch.no_grad():
        z = base.forward(theta, data["x"][None].expand(len(theta), -1, -1))
        pred = z > 0
        correct = pred == data["y"].bool()[None]
        mask = data["train_mask"]
        validation_bce = F.binary_cross_entropy_with_logits(z, data["y"][None].expand_as(z), reduction="none")
        return dict(**components(z, data), outputs=z,
            validation_bce=(validation_bce * ~mask).sum(1) / (16 - data["counts"]),
            train_accuracy=(correct * mask).sum(1) / data["counts"],
            unseen_accuracy=(correct * ~mask).sum(1) / (16 - data["counts"]),
            full_recovery=correct.all(1), function_ids=(pred.long() * 2 ** torch.arange(16)[None]).sum(1))


HISTORY_KEYS = ("outputs", "objective", "raw_bce", "amplitude_variance", "mean_amplitude", "validation_bce",
                "train_accuracy", "unseen_accuracy", "full_recovery")


def summarize(values, data):
    rows = []
    for group, lam in Config.GROUPS:
        ids = [m["model_index"] for m in data["metadata"] if m["group"] == group]
        bce = values["raw_bce"][ids]
        hard_fit = values["train_accuracy"][ids] == 1
        variance = values["amplitude_variance"][ids]
        amplitude = values["mean_amplitude"][ids]
        rows.append(dict(group=group, regularization=lam, models=len(ids), hard_fit=int(hard_fit.sum()),
            sufficiently_fitted=int((hard_fit & (bce <= Config.BCE_FIT_THRESHOLD)).sum()),
            maximum_raw_bce=float(bce.max()), median_raw_bce=float(bce.median()),
            median_objective=float(values["objective"][ids].median()),
            median_amplitude=float(amplitude.median()), median_amplitude_std=float(variance.sqrt().median()),
            median_amplitude_cv=float((variance.sqrt() / amplitude.abs().clamp_min(1e-100)).median()),
            unseen_accuracy=float(values["unseen_accuracy"][ids].mean()),
            full_recovery_count=int(values["full_recovery"][ids].sum()),
            full_recovery_fraction=float(values["full_recovery"][ids].double().mean())))
    return rows


def record(theta, state, data):
    values = evaluate(theta, data)
    assert torch.isfinite(values["objective"]).all()
    better = values["objective"] < state["best_objective"]
    state["best_objective"][better] = values["objective"][better]
    state["best_weights"][better] = theta.detach()[better]
    state["best_step"][better] = state["step"]
    fit = (values["raw_bce"] <= Config.BCE_FIT_THRESHOLD) & (values["train_accuracy"] == 1) & (state["fit_step"] < 0)
    state["fit_weights"][fit] = theta.detach()[fit]
    state["fit_step"][fit] = state["step"]
    state["fit_seconds"][fit] = state["seconds"]
    state["history_steps"].append(state["step"])
    state["history_seconds"].append(state["seconds"])
    for key in HISTORY_KEYS:
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
    for stage, weights in (("final", theta.detach()), ("best_train", state["best_weights"]), ("first_fit", state["fit_weights"])):
        valid = torch.isfinite(weights).all(1)
        values = evaluate(torch.nan_to_num(weights), data)
        if stage != "first_fit":
            summaries[stage] = summarize(values, data)
        for i, meta in enumerate(data["metadata"]):
            if not valid[i]:
                continue
            step = state["step"] if stage == "final" else int(state["best_step" if stage == "best_train" else "fit_step"][i])
            rows.append(dict(**meta, stage=stage, step=step,
                **{k: float(values[k][i]) for k in ("objective", "raw_bce", "amplitude_variance", "mean_amplitude", "validation_bce", "train_accuracy", "unseen_accuracy")},
                full_recovery=bool(values["full_recovery"][i]), function_id=int(values["function_ids"][i]),
                sufficiently_fitted=bool(values["raw_bce"][i] <= Config.BCE_FIT_THRESHOLD and values["train_accuracy"][i] == 1)))
    arrays = dict(inputs=data["x"].numpy(), labels=data["y"].numpy(), initial_weights=data["initial"].numpy(),
        train_mask=data["train_mask"].numpy(), targets=data["targets"].numpy(), lambdas=data["lambdas"].numpy(),
        decision_thresholds=np.zeros(len(theta)), train_ids_n8=data["train_ids"], test_ids_n8=data["test_ids"],
        final_weights=theta.detach().numpy(), best_train_weights=state["best_weights"].numpy(), best_step=state["best_step"].numpy(),
        first_fit_weights=state["fit_weights"].numpy(), first_fit_step=state["fit_step"].numpy(), first_fit_seconds=state["fit_seconds"].numpy(),
        history_steps=np.array(state["history_steps"]), history_seconds=np.array(state["history_seconds"]))
    for key in HISTORY_KEYS:
        arrays["history_" + key] = torch.stack(state["history_" + key]).numpy()
    np.savez_compressed(Config.RESULT_DIR / "results.npz", **arrays)
    base.write_json(Config.RESULT_DIR / "model_metadata.json", data["metadata"])
    base.write_csv(Config.RESULT_DIR / "per_model_results.csv", rows)
    summary = dict(status="completed", step=state["step"], signature=signature,
        training_loop_seconds=state["seconds"], model_count=len(theta), groups=summaries,
        scope="BCE+lambda*Var_D((2*y-1)*z)，方差除以n；仅训练样本参与，不固定共同幅度；原BCE复用。")
    base.write_json(Config.RESULT_DIR / "summary.json", summary)
    base.write_json(Config.RESULT_DIR / "progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = setup()
    old_cfg = json.loads((OLD / "config.json").read_text(encoding="utf-8"))
    assert sha(HELPER) == old_cfg["script_sha256"] and sha(previous.BASE_SCRIPT) == old_cfg["base_script_sha256"]
    settings = dict(groups=Config.GROUPS, train_count=8, independent_seeds=512, splits=16, seeds_per_split=32,
        seeds=[m["seed"] for m in data["metadata"][:512]], network="4->16->1 tanh; all 97 raw standard Gaussian parameters trainable",
        input_encoding="0/1", output="linear logit, classification threshold 0",
        objective="mean BCE + lambda * mean((signed_logit - train_mean_signed_logit)^2)",
        regularization_scope="training inputs only; signs from training labels; population variance; mean not detached",
        optimizer="Adam", lr=Config.LR, milestones=Config.MILESTONES, decay=Config.DECAY, adam_eps=Config.ADAM_EPS,
        weight_decay=0., steps=Config.STEPS, eval_every=Config.EVAL_EVERY, checkpoint_every=Config.CHECKPOINT_EVERY,
        bce_fit_threshold=Config.BCE_FIT_THRESHOLD, batch="full training set per model, objectives summed across models",
        dtype="float64", device="cpu", test_usage="evaluation only; no selection of coefficient or checkpoints by test results",
        source_hashes={str(path.relative_to(ROOT)): sha(path) for path in
            (Path(__file__), HELPER, previous.BASE_SCRIPT, OLD / "results.npz", OLD / "config.json")})
    for key in ("optimizer", "lr", "milestones", "decay", "adam_eps", "weight_decay", "steps", "dtype", "device", "eval_every"):
        assert json.loads(json.dumps(settings[key])) == old_cfg["settings"][key]
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cfg = dict(settings=settings, signature=signature, torch=str(torch.__version__), numpy=np.__version__)
    path = Config.RESULT_DIR / "config.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8"))["signature"] == signature, "协议或源码变化，拒绝复用"
    else:
        base.write_json(path, cfg)
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
    for key in ("steps", "seconds") + HISTORY_KEYS:
        state["history_" + key] = []
    saved = Config.RESULT_DIR / "checkpoint.pt"
    if saved.exists():
        restored = torch.load(saved, map_location="cpu", weights_only=True)
        assert restored["signature"] == signature
        theta.data.copy_(restored["weights"])
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["torch_rng"])
        state = restored["state"]
    else:
        record(theta, state, data)
    stop = [False]
    def interrupt_handler(signum, frame):
        stop[0] = True
    old_handler = signal.signal(signal.SIGINT, interrupt_handler)
    started, previous_seconds = time.perf_counter(), state["seconds"]
    inputs = data["x"][None].expand(m, -1, -1)
    try:
        for step in range(state["step"] + 1, Config.STEPS + 1):
            optimizer.param_groups[0]["lr"] = Config.LR * Config.DECAY ** sum(step > k for k in Config.MILESTONES)
            optimizer.zero_grad(set_to_none=True)
            components(base.forward(theta, inputs), data)["objective"].sum().backward()
            optimizer.step()
            state["step"] = step
            state["seconds"] = previous_seconds + time.perf_counter() - started
            if step % Config.EVAL_EVERY == 0 or stop[0]:
                record(theta, state, data)
            if step % Config.CHECKPOINT_EVERY == 0 or stop[0]:
                checkpoint(theta, optimizer, state, signature)
            if stop[0]:
                base.write_json(Config.RESULT_DIR / "progress.json", dict(status="paused", step=step, seconds=state["seconds"]))
                print("已在完整更新边界保存，可再次运行继续。", flush=True)
                return
    finally:
        signal.signal(signal.SIGINT, old_handler)
    checkpoint(theta, optimizer, state, signature)
    save_results(theta, state, data, signature)


if __name__ == "__main__":
    main()
