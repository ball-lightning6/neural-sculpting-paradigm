"""独立核验远档，再按同一来源/数据编号/初始化配对比较完整规则恢复。"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NEAR = ROOT / "tmp/results_mod97_frozen_function_duels_local_pilot_20260912"
FAR = ROOT / "tmp/results_mod97_frozen_function_duels_far_20260912"
OUT = ROOT / "tmp/mod97_near_far_duel_comparison_20260912"


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def final_arrays(root, cid, dataset, target):
    folder = root / "duels" / f"case_{cid:02d}_data_{dataset:02d}"
    summary = load_json(folder/"summary.json")
    assert summary["step"] == 50000 and summary["models"] == 16
    assert all(row["train_accuracy"] == 1 for row in summary["last"])
    with np.load(folder/"predictions/step_0050000.npz", allow_pickle=False) as z:
        pred = z["predictions"]
    with np.load(folder/"dataset.npz", allow_pickle=False) as z:
        indices = z["train_indices"]
        np.testing.assert_array_equal(z["own_targets"], target)
    assert len(indices) == 6518
    errors = (pred != target).sum(1)
    return errors, indices, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert load_json(NEAR/"protocol.json") == load_json(FAR/"protocol.json"), "网络/优化器/seed/样本量等科学配置不一致"
    assert (NEAR/"frozen_functions.npz").read_bytes() == (FAR/"frozen_functions.npz").read_bytes()
    spec = importlib.util.spec_from_file_location("far_duel_audit", Path(__file__).with_name("analyze_mod97_frozen_function_duels.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.SOURCE = FAR
    audit.OUT = ROOT / "tmp/mod97_far_function_duel_analysis_20260912"
    audit.main()
    panel = load_json(FAR/"panel.json")
    with np.load(NEAR/"frozen_functions.npz", allow_pickle=False) as z:
        target = z["target"]
    rows, near_results, far_results = [], [], []
    for cid in range(3):
        assert panel["cases"][cid]["source_model"] == panel["cases"][cid+3]["source_model"]
        near_group, far_group = [], []
        for dataset in range(2):
            ne, ni, ns = final_arrays(NEAR, cid, dataset, target)
            fe, fi, fs = final_arrays(FAR, cid+3, dataset, target)
            near_group.append(ne)
            far_group.append(fe)
            rows.append(dict(near_case=cid, far_case=cid+3, dataset=dataset, models=len(ne),
                near_exact=int((ne == 0).sum()), far_exact=int((fe == 0).sum()),
                paired_gained_exact=int(((ne > 0) & (fe == 0)).sum()),
                paired_lost_exact=int(((ne == 0) & (fe > 0)).sum()),
                both_exact=int(((ne == 0) & (fe == 0)).sum()),
                near_mean_errors=float(ne.mean()), far_mean_errors=float(fe.mean()),
                training_input_overlap=int(len(np.intersect1d(ni, fi))),
                near_conflict_A=float(np.mean([r["conflict_A"] for r in ns["last"]])),
                far_conflict_A=float(np.mean([r["conflict_A"] for r in fs["last"]])),
                far_conflict_B=float(np.mean([r["conflict_B"] for r in fs["last"]])),
                far_full_B=sum(r["full_B_exact"] for r in fs["last"]),
                near_error_vector=ne.tolist(), far_error_vector=fe.tolist()))
        near_results.append(near_group)
        far_results.append(far_group)
    ne = np.asarray(near_results)
    fe = np.asarray(far_results)
    # 同一初始化在三个来源候选中复用；bootstrap保留这个块，条件于两份已固定的数据划分。
    rng = np.random.default_rng(202609122301)
    deltas = []
    for _ in range(10000):
        index = rng.integers(0, 16, (2, 16))
        differences = [((fe[:, d, index[d]] == 0).mean()-(ne[:, d, index[d]] == 0).mean()) for d in range(2)]
        deltas.append(float(np.mean(differences)))
    summary = dict(models=int(ne.size), near_exact=int((ne == 0).sum()), far_exact=int((fe == 0).sum()),
        near_mean_errors=float(ne.mean()), far_mean_errors=float(fe.mean()),
        gained_exact=int(((ne > 0) & (fe == 0)).sum()), lost_exact=int(((ne == 0) & (fe > 0)).sum()),
        conditional_paired_bootstrap_rate_delta_95=np.quantile(deltas, [.025, .975]).tolist(),
        uncertainty_scope="仅刻画固定来源候选及已选训练集下的初始化波动；不代表跨任务或重新抽数据的总体区间。",
        interpretation="来源loss间隔和候选映射同时改变；训练样本数相同，但一致样本池及抽到的输入不同。不能单独归因为loss间隔。")
    write_json("paired_results.json", rows)
    write_json("summary.json", summary)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    xs = np.arange(3)
    axes[0].bar(xs-.18, (ne == 0).sum((1, 2)), width=.36, label="近档", color="#007C91")
    axes[0].bar(xs+.18, (fe == 0).sum((1, 2)), width=.36, label="更大loss间隔", color="#B84A48")
    axes[0].set(ylabel="完整恢复泛化规则的模型数 / 32", ylim=(0, 34))
    axes[1].bar(xs-.18, ne.mean((1, 2)), width=.36, label="近档", color="#007C91")
    axes[1].bar(xs+.18, fe.mean((1, 2)), width=.36, label="更大loss间隔", color="#B84A48")
    axes[1].set_ylabel("每模型在完整9312点上的平均错误数")
    axes[1].set_ylim(0, max(ne.mean((1, 2)).max(), fe.mean((1, 2)).max())*1.35)
    for ax in axes:
        ax.set_xticks(xs, ["来源seed0", "来源seed1", "来源seed2"])
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f" if ax is axes[0] else "%.2f", padding=3)
    fig.suptitle("同网络、样本数、初始化与五万步预算：近档 / 远档对决", fontsize=14)
    fig.savefig(OUT/"near_far_complete_recovery.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
