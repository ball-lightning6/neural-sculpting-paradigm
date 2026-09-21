"""自然0/1回归MSE与BCE：固定4bit copy，比较12与8个训练样本。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).with_name("experiment_4bit_copy_small_target_local.py")
OLD = ROOT / "tmp/4bit_copy_small_target_local_20260910"
spec = importlib.util.spec_from_file_location("copy_original_network", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Config:
    RESULT_DIR = ROOT / "tmp/4bit_copy_zero_one_mse_20260910"
    GROUPS = (("mse01_n12", "mse01", 12), ("bce_n8", "bce", 8), ("mse01_n8", "mse01", 8))
    SUBSET_SEED = 20260910101
    SEEDS = tuple(range(71001, 71513))
    SPLITS = 16
    SEEDS_PER_SPLIT = 32
    STEPS = 18000
    LR = 0.01
    MILESTONES = (2000, 5000, 9000, 13000, 16000)
    DECAY = 0.3
    ADAM_EPS = 1e-8
    EVAL_EVERY = 200
    CHECKPOINT_EVERY = 1000
    BCE_FIT_THRESHOLD = 1e-3
    RAW_MSE_FIT_THRESHOLD = 2.5e-5


def setup():
    with np.load(OLD / "results.npz", allow_pickle=False) as p:
        x = torch.tensor(p["inputs"], dtype=torch.float64)
        y = torch.tensor(p["labels"], dtype=torch.float64)
        initial = torch.tensor(p["initial_weights"][:512], dtype=torch.float64)
        train12 = p["train_indices"].copy()
        test12 = p["test_indices"].copy()
    rng = np.random.default_rng(Config.SUBSET_SEED)
    train8 = np.array([np.sort(rng.permutation(ids)[:8]) for ids in train12])
    test8 = np.array([[i for i in range(16) if i not in ids] for ids in train8])
    train_by_n = {12: train12, 8: train8}
    test_by_n = {12: test12, 8: test8}
    meta, masks = [], []
    for group, loss_type, n in Config.GROUPS:
        for seed_id, seed in enumerate(Config.SEEDS):
            split = seed_id // Config.SEEDS_PER_SPLIT
            mask = np.zeros(16, dtype=bool)
            mask[train_by_n[n][split]] = True
            masks.append(mask)
            meta.append(dict(model_index=len(meta), group=group, loss_type=loss_type,
                             train_count=n, split_id=split, seed=seed))
    mask = torch.tensor(np.array(masks))
    is_bce = torch.tensor([m["loss_type"] == "bce" for m in meta])
    # 未见位置在训练目标中使用占位0，且其权重严格为0。
    targets = torch.where(mask, y[None].expand(len(meta), -1), torch.zeros((len(meta), 16), dtype=torch.float64))
    return dict(x=x, y=y, initial=initial.repeat(len(Config.GROUPS), 1), metadata=meta,
        train_mask=mask, targets=targets, is_bce=is_bce,
        counts=mask.sum(1).double(), threshold=torch.where(is_bce, 0., .5).double(),
        goals=torch.where(is_bce, Config.BCE_FIT_THRESHOLD, Config.RAW_MSE_FIT_THRESHOLD).double(),
        train_by_n=train_by_n, test_by_n=test_by_n)


def objective(outputs, data):
    bce = F.binary_cross_entropy_with_logits(outputs, data["targets"], reduction="none")
    mse = (outputs - data["targets"]).square()
    items = torch.where(data["is_bce"][:, None], bce, mse)
    return (items * data["train_mask"]).sum(1) / data["counts"]


def numpy_value_gradient(theta, x, y, is_bce):
    hpar = theta[:80].reshape(16, 5)
    h = np.tanh((x @ hpar[:, :4].T + hpar[None, :, 4]) / 2)
    z = (h @ theta[80:96] + theta[96]) / 4
    if is_bce:
        value = np.logaddexp(0, -(2 * y - 1) * z).mean()
        dz = (np.exp(-np.logaddexp(0, -z)) - y) / len(y)
    else:
        value = np.mean((z - y) ** 2)
        dz = 2 * (z - y) / len(y)
    grad = np.zeros(97)
    grad[80:96] = h.T @ dz / 4
    grad[96] = dz.sum() / 4
    dh = dz[:, None] * theta[None, 80:96] * (1 - h ** 2) / 8
    grad[:80].reshape(16, 5)[:, :4] = dh.T @ x
    grad[:80].reshape(16, 5)[:, 4] = dh.sum(0)
    return float(value), grad


def preflight(data):
    p = data["initial"].clone().requires_grad_(True)
    x = data["x"][None].expand(len(p), -1, -1)
    values = objective(base.forward(p, x), data)
    grad = torch.autograd.grad(values.sum(), p)[0]
    errors = []
    for i in (0, 511, 512, 1023, 1024, 1535):
        ids = data["train_mask"][i].numpy()
        value, expected = numpy_value_gradient(p[i].detach().numpy(), data["x"].numpy()[ids],
                                               data["y"].numpy()[ids], bool(data["is_bce"][i]))
        error = float(abs(grad[i].numpy() - expected).max() / max(1, abs(expected).max()))
        assert error < 1e-12 and abs(value - values[i].item()) < 1e-12
        errors.append(error)
    flipped = dict(data)
    flipped["targets"] = torch.where(data["train_mask"], data["targets"], 1 - data["targets"])
    alternate = objective(base.forward(p, x), flipped)
    alt_grad = torch.autograd.grad(alternate.sum(), p)[0]
    assert torch.equal(values, alternate) and torch.equal(grad, alt_grad)
    for split in range(16):
        assert set(data["train_by_n"][8][split]).issubset(data["train_by_n"][12][split])
        for n in (8, 12):
            train, test = data["train_by_n"][n][split], data["test_by_n"][n][split]
            assert len(set(train)) == n and len(set(test)) == 16 - n
            assert not set(train).intersection(test)
    original = base.setup()
    for start in (0, 512, 1024):
        assert torch.equal(data["initial"][start:start + 512], original["initial"][:512])
    assert np.array_equal(data["train_by_n"][12], original["train"])
    return dict(independent_gradient_errors=errors, heldout_label_flip_changes_objective=False,
                heldout_label_flip_changes_gradient=False, exact_original_initializations=True,
                exact_original_n12_data=True, n8_nested_in_n12=True,
                mse_decision_threshold=.5, bce_decision_threshold=0.)


def evaluate(theta, data):
    z = base.forward(theta, data["x"][None].expand(len(theta), -1, -1))
    pred = z > data["threshold"][:, None]
    correct = pred == data["y"].bool()[None]
    mask = data["train_mask"]
    train_acc = (correct * mask).sum(1) / data["counts"]
    test_acc = (correct * ~mask).sum(1) / (16 - data["counts"])
    return dict(outputs=z, objective=objective(z, data), train_accuracy=train_acc,
        unseen_accuracy=test_acc, full_recovery=correct.all(1),
        function_ids=(pred.long() * 2 ** torch.arange(16)[None]).sum(1))


def summarize(values, data):
    rows = []
    for group, loss_type, n in Config.GROUPS:
        ids = [m["model_index"] for m in data["metadata"] if m["group"] == group]
        obj = values["objective"][ids]
        good = (obj <= data["goals"][ids]) & (values["train_accuracy"][ids] == 1)
        rows.append(dict(group=group, loss_type=loss_type, train_count=n, models=len(ids),
            train_accuracy=float(values["train_accuracy"][ids].mean()), hard_fit=int((values["train_accuracy"][ids] == 1).sum()),
            sufficiently_fitted=int(good.sum()), median_objective=float(obj.median()), maximum_objective=float(obj.max()),
            maximum_relative_mse=None if loss_type == "bce" else 4 * float(obj.max()),
            unseen_accuracy=float(values["unseen_accuracy"][ids].mean()), full_recovery_count=int(values["full_recovery"][ids].sum()),
            full_recovery_fraction=float(values["full_recovery"][ids].double().mean())))
    return rows


def record(theta, state, data):
    with torch.no_grad():
        values = evaluate(theta, data)
        assert torch.isfinite(values["objective"]).all()
        better = values["objective"] < state["best_objective"]
        state["best_objective"][better] = values["objective"][better]
        state["best_weights"][better] = theta.detach()[better]
        state["best_step"][better] = state["step"]
        fit = (values["objective"] <= data["goals"]) & (values["train_accuracy"] == 1) & (state["fit_step"] < 0)
        state["fit_weights"][fit] = theta.detach()[fit]
        state["fit_step"][fit] = state["step"]
        state["fit_seconds"][fit] = state["seconds"]
        state["history_steps"].append(state["step"])
        state["history_seconds"].append(state["seconds"])
        for key in ("outputs", "objective", "train_accuracy", "unseen_accuracy", "full_recovery"):
            state["history_" + key].append(values[key].detach().clone())
    progress = dict(status="running", step=state["step"], total_steps=Config.STEPS,
                    training_loop_seconds=state["seconds"], groups=summarize(values, data))
    base.write_json(Config.RESULT_DIR / "progress.json", progress)
    if state["step"] % 2000 == 0:
        print(json.dumps(base.clean(progress), ensure_ascii=False), flush=True)


def checkpoint(theta, optimizer, state, signature):
    path = Config.RESULT_DIR / "checkpoint.pt"
    temp = path.with_suffix(".pt.tmp")
    torch.save(dict(signature=signature, weights=theta.detach(), optimizer=optimizer.state_dict(),
                    state=state, torch_rng=torch.get_rng_state()), temp)
    temp.replace(path)


def save_results(theta, state, data, config):
    per_model, summaries = [], {}
    for stage, weights in (("final", theta.detach()), ("best_train", state["best_weights"]), ("first_fit", state["fit_weights"])):
        valid = torch.isfinite(weights).all(1)
        with torch.no_grad():
            values = evaluate(torch.nan_to_num(weights), data)
        if stage != "first_fit":
            summaries[stage] = summarize(values, data)
        for i, m in enumerate(data["metadata"]):
            if not valid[i]:
                continue
            step = state["step"] if stage == "final" else int(state["best_step" if stage == "best_train" else "fit_step"][i])
            obj = float(values["objective"][i])
            per_model.append(dict(**m, stage=stage, step=step, objective=obj,
                relative_mse=None if m["loss_type"] == "bce" else 4 * obj,
                train_accuracy=float(values["train_accuracy"][i]), unseen_accuracy=float(values["unseen_accuracy"][i]),
                function_id=int(values["function_ids"][i]), full_recovery=bool(values["full_recovery"][i]),
                sufficiently_fitted=bool(obj <= float(data["goals"][i]) and values["train_accuracy"][i] == 1)))
    base.write_csv(Config.RESULT_DIR / "per_model_results.csv", per_model)
    base.write_json(Config.RESULT_DIR / "model_metadata.json", data["metadata"])
    arrays = dict(inputs=data["x"].numpy(), labels=data["y"].numpy(), initial_weights=data["initial"].numpy(),
        train_mask=data["train_mask"].numpy(), decision_thresholds=data["threshold"].numpy(),
        final_weights=theta.detach().numpy(), best_train_weights=state["best_weights"].numpy(), best_step=state["best_step"].numpy(),
        first_fit_weights=state["fit_weights"].numpy(), first_fit_step=state["fit_step"].numpy(), first_fit_seconds=state["fit_seconds"].numpy(),
        history_steps=np.array(state["history_steps"]), history_seconds=np.array(state["history_seconds"]))
    for n in (8, 12):
        arrays[f"train_ids_n{n}"] = data["train_by_n"][n]
        arrays[f"test_ids_n{n}"] = data["test_by_n"][n]
    for key in ("outputs", "objective", "train_accuracy", "unseen_accuracy", "full_recovery"):
        arrays["history_" + key] = torch.stack(state["history_" + key]).numpy()
    np.savez_compressed(Config.RESULT_DIR / "results.npz", **arrays)
    summary = dict(status="completed", step=state["step"], training_loop_seconds=state["seconds"],
        signature=config["signature"], model_count=len(data["metadata"]), groups=summaries,
        scope="只新增MSE 0/1 n12、BCE n8、MSE 0/1 n8；BCE n12复用原结果。MSE是原始均方误差，分类阈值0.5。")
    base.write_json(Config.RESULT_DIR / "summary.json", summary)
    base.write_json(Config.RESULT_DIR / "progress.json", summary)
    plot(arrays, data)
    print(json.dumps(base.clean(summary), ensure_ascii=False, indent=2), flush=True)


def plot(arrays, data):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for by_time in (False, True):
        xx = arrays["history_seconds"] if by_time else arrays["history_steps"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
        for j, (group, loss_type, n) in enumerate(Config.GROUPS):
            ids = [m["model_index"] for m in data["metadata"] if m["group"] == group]
            displayed = np.maximum(arrays["history_objective"][:, ids], 1e-30)
            axes[0, j].plot(xx, np.median(displayed, axis=1), color="#247f83")
            axes[0, j].fill_between(xx, np.quantile(displayed, .1, axis=1), np.quantile(displayed, .9, axis=1), color="#247f83", alpha=.18)
            axes[0, j].axhline(Config.BCE_FIT_THRESHOLD if loss_type == "bce" else Config.RAW_MSE_FIT_THRESHOLD, color="#777777", ls="--", lw=.8)
            axes[0, j].set(title=f'{"BCE" if loss_type == "bce" else "MSE 0/1"}，n={n}',
                           ylabel="原始训练目标", yscale="log")
            for key, label, color, style in (("train_accuracy", "训练准确率", "#247f83", "--"),
                                            ("unseen_accuracy", "未见准确率", "#b24d55", "-"),
                                            ("full_recovery", "完整copy恢复率", "#566bb0", "-")):
                axes[1, j].plot(xx, arrays["history_" + key][:, ids].mean(1), label=label, color=color, ls=style)
            axes[1, j].set(xlabel="训练循环墙钟时间 / 秒" if by_time else "Adam 更新步数", ylabel="比例", ylim=(-.02, 1.04))
            for ax in axes[:, j]:
                ax.grid(alpha=.18)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.suptitle("4bit copy：每组512个独立初始化，16个配对训练集\n阴影为10%至90%分位数；低于1e-30仅在绘图中截断", fontsize=11)
        fig.savefig(Config.RESULT_DIR / ("02_training_by_time.png" if by_time else "01_training_curves.png"), dpi=170)
        plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = setup()
    settings = dict(network="4->16->1 tanh; 97 raw standard Gaussian parameters; all trainable",
        groups=Config.GROUPS, subset_seed=Config.SUBSET_SEED, seeds=Config.SEEDS, splits=16, seeds_per_split=32,
        input_encoding="0/1", mse_targets="0 and 1", mse_output_activation="linear, no sigmoid",
        mse_objective="mean squared error, no amplitude normalization", mse_decision_threshold=.5,
        bce_decision_threshold=0., bce_fit_threshold=Config.BCE_FIT_THRESHOLD,
        raw_mse_fit_threshold=Config.RAW_MSE_FIT_THRESHOLD, mse_relative_error_definition="raw MSE / 0.5^2",
        optimizer="Adam", lr=Config.LR, milestones=Config.MILESTONES, decay=Config.DECAY, adam_eps=Config.ADAM_EPS,
        weight_decay=0., steps=Config.STEPS, eval_every=Config.EVAL_EVERY, checkpoint_every=Config.CHECKPOINT_EVERY,
        batch="per-model full training set; explicitly masked heldout positions with dummy labels",
        dtype="float64", device="cpu", test_usage="evaluation only; not used to tune or select checkpoints")
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    config = dict(settings=settings, signature=signature, torch=str(torch.__version__), numpy=np.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_script_sha256=hashlib.sha256(BASE_SCRIPT.read_bytes()).hexdigest(),
        original_results_sha256=hashlib.sha256((OLD / "results.npz").read_bytes()).hexdigest())
    path = Config.RESULT_DIR / "config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise RuntimeError("结果目录协议不一致，拒绝覆盖")
    else:
        base.write_json(path, config)
    done = Config.RESULT_DIR / "summary.json"
    if done.exists() and json.loads(done.read_text(encoding="utf-8")).get("status") == "completed":
        print(done.read_text(encoding="utf-8"))
        return
    base.write_json(Config.RESULT_DIR / "preflight.json", preflight(data))
    theta = torch.nn.Parameter(data["initial"].clone())
    optimizer = torch.optim.Adam([theta], lr=Config.LR, eps=Config.ADAM_EPS, weight_decay=0.)
    m = len(theta)
    state = dict(step=0, seconds=0., best_weights=theta.detach().clone(), best_objective=torch.full((m,), float("inf"), dtype=torch.float64),
        best_step=torch.full((m,), -1, dtype=torch.int64), fit_weights=torch.full_like(theta, float("nan")),
        fit_step=torch.full((m,), -1, dtype=torch.int64), fit_seconds=torch.full((m,), float("nan"), dtype=torch.float64))
    for key in ("steps", "seconds", "outputs", "objective", "train_accuracy", "unseen_accuracy", "full_recovery"):
        state["history_" + key] = []
    saved = Config.RESULT_DIR / "checkpoint.pt"
    if saved.exists():
        old = torch.load(saved, map_location="cpu", weights_only=True)
        if old["signature"] != signature:
            raise RuntimeError("检查点协议不匹配")
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
                print("已在完整更新边界保存，可再次运行继续。", flush=True)
                return
    finally:
        signal.signal(signal.SIGINT, original_handler)
    checkpoint(theta, optimizer, state, signature)
    save_results(theta, state, data, config)


if __name__ == "__main__":
    main()
