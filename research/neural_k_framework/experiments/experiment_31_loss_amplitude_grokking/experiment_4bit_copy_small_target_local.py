"""4bit copy：12个训练输入下，比较BCE与不同幅度的线性输出MSE。"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class Config:
    INPUT_BITS = 4
    WIDTH = 16
    TRAIN_COUNT = 12
    SPLITS = 16
    SEEDS_PER_SPLIT = 32
    INITIALIZATION_SEEDS = tuple(range(71001, 71513))
    SPLIT_SEED = 20260910091
    CONDITIONS = (("bce", 1.0), ("mse_1", 1.0), ("mse_0p001", 0.001))
    DEVICE = "cpu"
    DTYPE = torch.float64
    STEPS = 18000
    LEARNING_RATE = 0.01
    LR_MILESTONES = (2000, 5000, 9000, 13000, 16000)
    LR_DECAY = 0.3
    ADAM_EPS = 1e-8
    EVAL_EVERY = 200
    CHECKPOINT_EVERY = 1000
    BCE_FIT_THRESHOLD = 1e-3
    MSE_RELATIVE_FIT_THRESHOLD = 1e-4
    RESULT_DIR = Path(__file__).resolve().parents[2] / "tmp/4bit_copy_small_target_local_20260910"


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def forward(theta, inputs):
    hidden_parameters = theta[:, :80].reshape(-1, 16, 5)
    h = torch.tanh((torch.bmm(inputs, hidden_parameters[:, :, :4].transpose(1, 2))
                    + hidden_parameters[:, None, :, 4]) / 2)
    return (h * theta[:, None, 80:96]).sum(2) / 4 + theta[:, 96, None] / 4


def numpy_objective_gradient(theta, inputs, labels, condition, amplitude):
    """独立NumPy实现，供梯度核验及训练集上的后续收敛检查使用。"""
    hidden = theta[:80].reshape(16, 5)
    h = np.tanh((inputs @ hidden[:, :4].T + hidden[None, :, 4]) / 2)
    z = (h @ theta[80:96] + theta[96]) / 4
    n = len(inputs)
    if condition == "bce":
        signed = 2 * labels - 1
        value = np.logaddexp(0, -signed * z).mean()
        p = np.exp(-np.logaddexp(0, -z))
        dz = (p - labels) / n
    else:
        residual = z / amplitude - (2 * labels - 1)
        value = (residual ** 2).mean()
        dz = 2 * residual / (amplitude * n)
    grad = np.zeros_like(theta)
    grad[80:96] = h.T @ dz / 4
    grad[96] = dz.sum() / 4
    dh = dz[:, None] * theta[None, 80:96] * (1 - h * h) / 8
    grad[:80].reshape(16, 5)[:, :4] = dh.T @ inputs
    grad[:80].reshape(16, 5)[:, 4] = dh.sum(0)
    return float(value), grad


def setup():
    x = torch.tensor([[i >> j & 1 for j in (3, 2, 1, 0)] for i in range(16)], dtype=Config.DTYPE)
    labels = x[:, -1].clone()
    choices = np.array(list(itertools.combinations(range(16), 4)), dtype=np.int64)
    rng = np.random.default_rng(Config.SPLIT_SEED)
    heldout = choices[rng.choice(len(choices), Config.SPLITS, replace=False)]
    train = np.array([[i for i in range(16) if i not in u] for u in heldout])
    init = torch.stack([torch.randn(97, dtype=Config.DTYPE,
        generator=torch.Generator().manual_seed(seed)) for seed in Config.INITIALIZATION_SEEDS])
    assert len(Config.INITIALIZATION_SEEDS) == Config.SPLITS * Config.SEEDS_PER_SPLIT
    meta = [dict(model_index=condition_id * len(init) + seed_id, condition=name,
                 amplitude=a, split_id=seed_id // Config.SEEDS_PER_SPLIT, seed=seed)
        for condition_id, (name, a) in enumerate(Config.CONDITIONS)
        for seed_id, seed in enumerate(Config.INITIALIZATION_SEEDS)]
    seeds = len(Config.INITIALIZATION_SEEDS)
    initial = init.repeat(len(Config.CONDITIONS), 1)
    train_ids = torch.tensor(np.tile(np.repeat(train, Config.SEEDS_PER_SPLIT, axis=0), (len(Config.CONDITIONS), 1)))
    test_ids = torch.tensor(np.tile(np.repeat(heldout, Config.SEEDS_PER_SPLIT, axis=0), (len(Config.CONDITIONS), 1)))
    amp = torch.tensor([r["amplitude"] for r in meta], dtype=Config.DTYPE)
    is_bce = torch.tensor([r["condition"] == "bce" for r in meta])
    targets = labels[train_ids]
    return dict(x=x, labels=labels, train=train, test=heldout, initial=initial, metadata=meta,
                train_ids=train_ids, test_ids=test_ids, amp=amp, is_bce=is_bce, targets=targets)


def per_model_objective(z, data):
    bce = F.binary_cross_entropy_with_logits(z, data["targets"], reduction="none").mean(1)
    relative_mse = (z / data["amp"][:, None] - (2 * data["targets"] - 1)).square().mean(1)
    return torch.where(data["is_bce"], bce, relative_mse)


def preflight(data):
    errors = []
    probes = (0, len(Config.INITIALIZATION_SEEDS), 2 * len(Config.INITIALIZATION_SEEDS))
    for i in probes:
        point = data["initial"][i:i + 1].clone().requires_grad_(True)
        ids = data["train_ids"][i]
        z = forward(point, data["x"][ids][None])
        meta = data["metadata"][i]
        yy = data["labels"][ids]
        if meta["condition"] == "bce":
            value = F.binary_cross_entropy_with_logits(z[0], yy)
        else:
            value = (z[0] / meta["amplitude"] - (2 * yy - 1)).square().mean()
        gradient = torch.autograd.grad(value, point)[0][0].detach().numpy()
        expected_value, expected_gradient = numpy_objective_gradient(point.detach().numpy()[0],
            data["x"][ids].numpy(), yy.numpy(), meta["condition"], meta["amplitude"])
        error = np.max(abs(gradient - expected_gradient)) / max(1, np.max(abs(expected_gradient)))
        assert error < 1e-12
        assert abs(value.item() - expected_value) / max(1, abs(expected_value)) < 1e-12
        errors.append(float(error))
    # 相同输入与初始化下，不同模型的目标取和，不把梯度除以模型总数。
    batch = data["initial"].clone().requires_grad_(True)
    per_model_objective(forward(batch, data["x"][data["train_ids"]]), data).sum().backward()
    for i in probes:
        m = data["metadata"][i]
        ids = data["train_ids"][i]
        _, expected = numpy_objective_gradient(batch.detach().numpy()[i], data["x"][ids].numpy(),
            data["labels"][ids].numpy(), m["condition"], m["amplitude"])
        assert np.max(abs(batch.grad[i].numpy() - expected)) / max(1, np.max(abs(expected))) < 1e-12
    xaug = np.column_stack((data["x"].numpy(), np.ones(16)))
    linear_error = 0.
    for ids in data["train"]:
        assert len(set(ids)) == 12 and np.linalg.matrix_rank(xaug[ids]) == 5
        for a in (1., .001):
            weights = np.linalg.lstsq(xaug[ids], a * (2 * data["labels"].numpy()[ids] - 1), rcond=None)[0]
            prediction = xaug @ weights
            assert np.array_equal(prediction > 0, data["labels"].numpy().astype(bool))
            linear_error = max(linear_error, float(abs(prediction / a - (2 * data["labels"].numpy() - 1)).max()))
    z = forward(data["initial"][:1], data["x"][None])
    signed = 2 * data["labels"][None] - 1
    original = (z - signed).square().mean()
    rescaled = ((.001 * z - .001 * signed) / .001).square().mean()
    assert abs(original.item() - rescaled.item()) < 1e-12
    return dict(independent_gradient_relative_errors=errors, linear_control_max_relative_error=linear_error,
                linear_control_all_heldout_correct=True, pure_unit_rescaling_objective_error=abs(original.item() - rescaled.item()))


def fit_thresholds(data):
    return torch.where(data["is_bce"], Config.BCE_FIT_THRESHOLD, Config.MSE_RELATIVE_FIT_THRESHOLD)


def metrics(theta, data):
    z = forward(theta, data["x"][None].expand(len(theta), -1, -1))
    pred = z > 0
    correct = pred == data["labels"].bool()[None]
    train_acc = correct.gather(1, data["train_ids"]).double().mean(1)
    test_acc = correct.gather(1, data["test_ids"]).double().mean(1)
    obj = per_model_objective(z.gather(1, data["train_ids"]), data)
    fids = (pred.long() * (2 ** torch.arange(16))[None]).sum(1)
    return dict(outputs=z, objective=obj, train_accuracy=train_acc, unseen_accuracy=test_acc,
                full_recovery=correct.all(1), function_ids=fids)


def summarize_metrics(values, data):
    result = []
    for name, a in Config.CONDITIONS:
        ids = [r["model_index"] for r in data["metadata"] if r["condition"] == name]
        obj = values["objective"][ids]
        sufficient = (obj <= fit_thresholds(data)[ids]) & (values["train_accuracy"][ids] == 1)
        result.append(dict(condition=name, amplitude=a, models=len(ids),
            median_objective=float(obj.median()), maximum_objective=float(obj.max()),
            fully_hard_fit=int((values["train_accuracy"][ids] == 1).sum()),
            sufficiently_fitted=int(sufficient.sum()), mean_unseen_accuracy=float(values["unseen_accuracy"][ids].mean()),
            full_recovery_count=int(values["full_recovery"][ids].sum()),
            recovery_among_sufficiently_fitted=(float(values["full_recovery"][ids][sufficient].double().mean())
                                               if sufficient.any() else None)))
    return result


def checkpoint(state, optimizer, theta, config):
    target = Config.RESULT_DIR / "checkpoint.pt"
    temp = target.with_suffix(".pt.tmp")
    torch.save(dict(signature=config["signature"], weights=theta.detach(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), state=state), temp)
    temp.replace(target)


def record_step(step, seconds, theta, state, data):
    with torch.no_grad():
        values = metrics(theta, data)
        if not torch.isfinite(values["objective"]).all():
            raise FloatingPointError("训练目标出现非有限值")
        better = values["objective"] < state["best_objective"]
        state["best_objective"][better] = values["objective"][better]
        state["best_weights"][better] = theta.detach()[better]
        state["best_step"][better] = step
        hard = (values["train_accuracy"] == 1) & (state["hardfit_step"] < 0)
        fit = (values["train_accuracy"] == 1) & (values["objective"] <= fit_thresholds(data)) & (state["fit_step"] < 0)
        for mask, prefix in ((hard, "hardfit"), (fit, "fit")):
            state[prefix + "_weights"][mask] = theta.detach()[mask]
            state[prefix + "_step"][mask] = step
            state[prefix + "_seconds"][mask] = seconds
        state["history_step"].append(step)
        state["history_seconds"].append(seconds)
        state["history_outputs"].append(values["outputs"].detach().clone())
        state["history_objective"].append(values["objective"].detach().clone())
        state["history_train_accuracy"].append(values["train_accuracy"].detach().clone())
        state["history_unseen_accuracy"].append(values["unseen_accuracy"].detach().clone())
        state["history_recovery"].append(values["full_recovery"].detach().clone())
    progress = dict(status="running", step=step, total_steps=Config.STEPS, training_seconds=seconds,
                    groups=summarize_metrics(values, data))
    write_json(Config.RESULT_DIR / "progress.json", progress)
    if step % 2000 == 0:
        print(json.dumps(clean(progress), ensure_ascii=False), flush=True)
    return values


def save_results(theta, state, data, config):
    stage_weights = dict(final=theta.detach(), best_train=state["best_weights"],
                         first_fit=state["fit_weights"], first_hardfit=state["hardfit_weights"])
    results = []
    stage_values = {}
    with torch.no_grad():
        for stage, weights in stage_weights.items():
            values = metrics(torch.nan_to_num(weights), data)
            valid = torch.isfinite(weights).all(1)
            stage_values[stage] = values
            for i, meta in enumerate(data["metadata"]):
                if not valid[i]:
                    continue
                obj = float(values["objective"][i])
                test_z = values["outputs"][i, data["test_ids"][i]]
                step = state["step"] if stage == "final" else int(state[
                    "best_step" if stage == "best_train" else "fit_step" if stage == "first_fit" else "hardfit_step"][i])
                results.append(dict(**meta, stage=stage, step=step, objective=obj,
                    raw_training_mse=(None if meta["condition"] == "bce" else obj * meta["amplitude"] ** 2),
                    sufficiently_fitted=bool(obj <= float(fit_thresholds(data)[i]) and values["train_accuracy"][i] == 1),
                    train_accuracy=float(values["train_accuracy"][i]), unseen_accuracy=float(values["unseen_accuracy"][i]),
                    full_rule_recovery=bool(values["full_recovery"][i]), function_id=int(values["function_ids"][i]),
                    mean_absolute_unseen_output=float(test_z.abs().mean()),
                    unseen_output_abs_over_target_amplitude=(None if meta["condition"] == "bce" else float(test_z.abs().mean()) / meta["amplitude"])))
    group_results = {}
    for stage in ("final", "best_train"):
        group_results[stage] = summarize_metrics(stage_values[stage], data)
    agreement = []
    seed_count = Config.SEEDS_PER_SPLIT
    for stage in ("final", "best_train"):
        for condition, a in Config.CONDITIONS:
            for split in range(Config.SPLITS):
                ids = [r["model_index"] for r in data["metadata"] if r["condition"] == condition and r["split_id"] == split]
                z = stage_values[stage]["outputs"][ids][:, data["test"][split]].numpy()
                count = (z > 0).sum(0)
                agree = (count * (count - 1) + (seed_count - count) * (seed_count - count - 1)) / (seed_count * (seed_count - 1))
                y = data["labels"].numpy()[data["test"][split]].astype(bool)
                agreement.append(dict(stage=stage, condition=condition, split_id=split,
                    unseen_pair_agreement_excluding_self=float(agree.mean()),
                    mean_output_classifier_accuracy=float(((z.mean(0) > 0) == y).mean()),
                    mean_individual_unseen_accuracy=float(((z > 0) == y[None]).mean())))
    write_csv(Config.RESULT_DIR / "per_model_results.csv", results)
    write_csv(Config.RESULT_DIR / "per_split_agreement.csv", agreement)
    write_json(Config.RESULT_DIR / "model_metadata.json", data["metadata"])
    arrays = dict(inputs=data["x"].numpy(), labels=data["labels"].numpy(), train_indices=data["train"],
        test_indices=data["test"], initial_weights=data["initial"].numpy(), final_weights=theta.detach().numpy(),
        best_train_weights=state["best_weights"].numpy(), best_step=state["best_step"].numpy(),
        first_fit_weights=state["fit_weights"].numpy(), first_fit_step=state["fit_step"].numpy(),
        first_hardfit_weights=state["hardfit_weights"].numpy(), first_hardfit_step=state["hardfit_step"].numpy(),
        history_steps=np.array(state["history_step"]), history_seconds=np.array(state["history_seconds"]))
    for key in ("outputs", "objective", "train_accuracy", "unseen_accuracy", "recovery"):
        arrays["history_" + key] = torch.stack(state["history_" + key]).numpy()
    np.savez_compressed(Config.RESULT_DIR / "results.npz", **arrays)
    summary = dict(status="completed", protocol_signature=config["signature"], step=state["step"],
        training_seconds=state["training_seconds"], models=len(data["metadata"]),
        conditions=group_results, target_function_id=0xAAAA,
        interpretation="16个训练集各用32个不同初始化，共512个独立初始化；同一seed只跨loss配对，另报告逐训练集差异。",
        selection="best_train只按训练目标选择；未使用未见样本调参或选择检查点。",
        scope="Adam真实训练对照，不是静态体积测量；固定n=12，不是完整n50/n90扫描。")
    write_json(Config.RESULT_DIR / "summary.json", summary)
    write_json(Config.RESULT_DIR / "progress.json", summary)
    plot(arrays, data)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2), flush=True)


def plot(arrays, data):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    steps = arrays["history_steps"]
    for j, (name, a) in enumerate(Config.CONDITIONS):
        ids = [r["model_index"] for r in data["metadata"] if r["condition"] == name]
        loss = arrays["history_objective"][:, ids]
        axes[0, j].plot(steps, np.median(loss, axis=1), color="#247f83")
        axes[0, j].fill_between(steps, np.quantile(loss, .1, axis=1), np.quantile(loss, .9, axis=1), color="#247f83", alpha=.18)
        threshold = Config.BCE_FIT_THRESHOLD if name == "bce" else Config.MSE_RELATIVE_FIT_THRESHOLD
        axes[0, j].axhline(threshold, color="#777777", ls="--", lw=.8)
        axes[0, j].set(title="BCE" if name == "bce" else f"线性输出 MSE，目标 ±{a:g}",
                       ylabel="训练 BCE" if name == "bce" else "训练 MSE / 目标幅度²", yscale="log")
        axes[1, j].plot(steps, arrays["history_train_accuracy"][:, ids].mean(1), label="训练准确率", color="#247f83", ls="--")
        axes[1, j].plot(steps, arrays["history_unseen_accuracy"][:, ids].mean(1), label="未见准确率", color="#b24d55")
        axes[1, j].plot(steps, arrays["history_recovery"][:, ids].mean(1), label="完整规则恢复率", color="#566bb0")
        axes[1, j].set(xlabel="Adam 更新步数", ylabel="比例", ylim=(-.02, 1.04))
        axes[1, j].legend(loc="lower right")
        for ax in axes[:, j]:
            ax.grid(alpha=.18)
    fig.suptitle(f"4bit copy：12个训练输入、4个未见输入；每组{Config.SPLITS}个训练集，各{Config.SEEDS_PER_SPLIT}个独立初始化", fontsize=12)
    fig.savefig(Config.RESULT_DIR / "01_training_curves.png", dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = setup()
    settings = dict(network="4->16->1 tanh; 97 independent standard Gaussian raw parameters; all trainable",
        input_encoding="0/1", hidden_fan_in_divisor=2., output_fan_in_divisor=4.,
        rule="copy rightmost input bit", train_count=12, unseen_count=4,
        split_seed=Config.SPLIT_SEED, split_count=Config.SPLITS, seeds_per_split=Config.SEEDS_PER_SPLIT,
        initialization_seeds=Config.INITIALIZATION_SEEDS,
        conditions=Config.CONDITIONS, optimizer="Adam", weight_decay=0., learning_rate=Config.LEARNING_RATE,
        lr_milestones=Config.LR_MILESTONES, lr_decay=Config.LR_DECAY, adam_eps=Config.ADAM_EPS,
        steps=Config.STEPS, evaluation_every=Config.EVAL_EVERY, checkpoint_every=Config.CHECKPOINT_EVERY,
        bce_fit_threshold=Config.BCE_FIT_THRESHOLD, mse_relative_fit_threshold=Config.MSE_RELATIVE_FIT_THRESHOLD,
        mse_definition="mean((raw_output / amplitude - signed_label)^2); linear output, no sigmoid",
        bce_definition="mean binary_cross_entropy_with_logits(raw_output, bit_label)",
        batch="full training set per model; sum independent model objectives", dtype="float64", device="cpu",
        test_usage="evaluation only; not used for scheduling, optimization, fitting threshold or checkpoint choice",
        followup_if_underfit="训练相对误差未达标时，不将之当作拟合后不泛化；可另做训练集上的全参数L-BFGS补充，分别报告。")
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    config = dict(settings=settings, signature=signature, torch=str(torch.__version__),
                  numpy=np.__version__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config_path = Config.RESULT_DIR / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise RuntimeError("结果目录已有不同协议，拒绝覆盖")
    else:
        write_json(config_path, config)
    complete = Config.RESULT_DIR / "summary.json"
    if complete.exists() and json.loads(complete.read_text(encoding="utf-8")).get("status") == "completed":
        print(complete.read_text(encoding="utf-8"))
        return
    checks = preflight(data)
    write_json(Config.RESULT_DIR / "preflight.json", checks)
    np.savez_compressed(Config.RESULT_DIR / "dataset.npz", inputs=data["x"].numpy(),
                        labels=data["labels"].numpy(), train_indices=data["train"], test_indices=data["test"])
    theta = torch.nn.Parameter(data["initial"].clone())
    optimizer = torch.optim.Adam([theta], lr=Config.LEARNING_RATE, eps=Config.ADAM_EPS, weight_decay=0.)
    count = len(theta)
    state = dict(step=0, training_seconds=0., best_objective=torch.full((count,), float("inf"), dtype=Config.DTYPE),
        best_weights=theta.detach().clone(), best_step=torch.full((count,), -1, dtype=torch.int64),
        fit_weights=torch.full_like(theta, float("nan")), fit_step=torch.full((count,), -1, dtype=torch.int64),
        fit_seconds=torch.full((count,), float("nan"), dtype=Config.DTYPE),
        hardfit_weights=torch.full_like(theta, float("nan")), hardfit_step=torch.full((count,), -1, dtype=torch.int64),
        hardfit_seconds=torch.full((count,), float("nan"), dtype=Config.DTYPE))
    for key in ("step", "seconds", "outputs", "objective", "train_accuracy", "unseen_accuracy", "recovery"):
        state["history_" + key] = []
    saved = Config.RESULT_DIR / "checkpoint.pt"
    if saved.exists():
        stored = torch.load(saved, map_location="cpu", weights_only=True)
        if stored["signature"] != signature:
            raise RuntimeError("检查点与当前协议不一致")
        theta.data.copy_(stored["weights"])
        optimizer.load_state_dict(stored["optimizer"])
        torch.set_rng_state(stored["torch_rng"])
        state = stored["state"]
    else:
        record_step(0, 0., theta, state, data)
    batch_inputs = data["x"][data["train_ids"]]
    previous_seconds = state["training_seconds"]
    start = time.perf_counter()
    try:
        for step in range(state["step"] + 1, Config.STEPS + 1):
            optimizer.param_groups[0]["lr"] = Config.LEARNING_RATE * Config.LR_DECAY ** sum(
                step > milestone for milestone in Config.LR_MILESTONES)
            optimizer.zero_grad(set_to_none=True)
            per_model_objective(forward(theta, batch_inputs), data).sum().backward()
            optimizer.step()
            state["step"] = step
            state["training_seconds"] = previous_seconds + time.perf_counter() - start
            if step % Config.EVAL_EVERY == 0:
                record_step(step, state["training_seconds"], theta, state, data)
            if step % Config.CHECKPOINT_EVERY == 0:
                checkpoint(state, optimizer, theta, config)
    except KeyboardInterrupt:
        state["training_seconds"] = previous_seconds + time.perf_counter() - start
        checkpoint(state, optimizer, theta, config)
        write_json(Config.RESULT_DIR / "progress.json", dict(status="paused", step=state["step"],
                   training_seconds=state["training_seconds"], signature=signature))
        print("已保存完整检查点；再次运行同一脚本可继续。", flush=True)
        return
    checkpoint(state, optimizer, theta, config)
    save_results(theta, state, data, config)


if __name__ == "__main__":
    main()
