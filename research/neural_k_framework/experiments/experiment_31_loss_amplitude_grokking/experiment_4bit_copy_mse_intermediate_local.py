"""复用已核验的copy训练实现，只补充MSE目标幅度0.1和0.01。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).with_name("experiment_4bit_copy_small_target_local.py")
BASE_RESULTS = ROOT / "tmp/4bit_copy_small_target_local_20260910"
RESULT_DIR = ROOT / "tmp/4bit_copy_mse_intermediate_20260910"
CONDITIONS = (("mse_0p1", 0.1), ("mse_0p01", 0.01))

spec = importlib.util.spec_from_file_location("copy_base_training", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.Config.CONDITIONS = CONDITIONS
base.Config.RESULT_DIR = RESULT_DIR


def preflight(data):
    checks = []
    count = len(base.Config.INITIALIZATION_SEEDS)
    probes = [c * count + s for c in range(len(CONDITIONS)) for s in (0, count // 2, count - 1)]
    points = data["initial"].clone().requires_grad_(True)
    value = base.per_model_objective(base.forward(points, data["x"][data["train_ids"]]), data)
    value.sum().backward()
    for i in probes:
        ids, meta = data["train_ids"][i], data["metadata"][i]
        wanted_value, wanted_grad = base.numpy_objective_gradient(points.detach().numpy()[i],
            data["x"][ids].numpy(), data["labels"][ids].numpy(), meta["condition"], meta["amplitude"])
        error = float(np.max(abs(points.grad[i].numpy() - wanted_grad)) / max(1, np.max(abs(wanted_grad))))
        assert error < 1e-12
        assert abs(value[i].item() - wanted_value) / max(1, abs(wanted_value)) < 1e-12
        checks.append(dict(condition=meta["condition"], seed=meta["seed"], gradient_relative_error=error))
    with np.load(BASE_RESULTS / "results.npz", allow_pickle=False) as old:
        assert np.array_equal(data["train"], old["train_indices"])
        assert np.array_equal(data["test"], old["test_indices"])
        assert np.array_equal(data["x"].numpy(), old["inputs"])
        assert np.array_equal(data["labels"].numpy(), old["labels"])
        for c in range(len(CONDITIONS)):
            assert np.array_equal(data["initial"][c * count:(c + 1) * count].numpy(), old["initial_weights"][:count])
    return dict(gradient_checks=checks, exact_match_to_original_data=True,
                exact_match_to_original_512_initializations=True,
                original_result_sha256=hashlib.sha256((BASE_RESULTS / "results.npz").read_bytes()).hexdigest())


def plot(arrays, data):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for by_time in (False, True):
        xx = arrays["history_seconds"] if by_time else arrays["history_steps"]
        fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(10, 7.5), layout="constrained")
        for j, (name, amplitude) in enumerate(CONDITIONS):
            ids = [m["model_index"] for m in data["metadata"] if m["condition"] == name]
            displayed = np.maximum(arrays["history_objective"][:, ids], 1e-30)
            axes[0, j].plot(xx, np.median(displayed, axis=1), color="#247f83")
            axes[0, j].fill_between(xx, np.quantile(displayed, .1, axis=1), np.quantile(displayed, .9, axis=1),
                                   color="#247f83", alpha=.18)
            axes[0, j].axhline(base.Config.MSE_RELATIVE_FIT_THRESHOLD, color="#777777", ls="--", lw=.8)
            axes[0, j].set(title=f"线性输出MSE，目标 ±{amplitude:g}", ylabel="训练 MSE / 目标幅度²", yscale="log")
            for key, name_zh, color, style in (("train_accuracy", "训练准确率", "#247f83", "--"),
                                              ("unseen_accuracy", "未见准确率", "#b24d55", "-"),
                                              ("recovery", "完整copy恢复率", "#566bb0", "-")):
                axes[1, j].plot(xx, arrays["history_" + key][:, ids].mean(1), label=name_zh, color=color, ls=style)
            axes[1, j].set(xlabel="全批训练循环墙钟时间 / 秒" if by_time else "Adam 更新步数", ylabel="比例", ylim=(-.02, 1.04))
            for ax in axes[:, j]:
                ax.grid(alpha=.18)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.suptitle("4bit copy：每组512个独立初始化，与上一轮训练集及权重严格配对\n阴影为10%至90%分位数；低于1e-30仅在绘图中截断", fontsize=11)
        fig.savefig(RESULT_DIR / ("02_training_by_time.png" if by_time else "01_training_curves.png"), dpi=170)
        plt.close(fig)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = dict(added_conditions=CONDITIONS, base_script=str(BASE_SCRIPT),
        base_script_sha256=hashlib.sha256(BASE_SCRIPT.read_bytes()).hexdigest(),
        extension_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_results=str(BASE_RESULTS),
        scope="只新增两个幅度，训练实现、数据、初始化、优化器及步数均复用；不重跑原三组。")
    base.write_json(RESULT_DIR / "extension_protocol.json", provenance)
    base.preflight = preflight
    base.plot = plot
    base.main()


if __name__ == "__main__":
    main()
