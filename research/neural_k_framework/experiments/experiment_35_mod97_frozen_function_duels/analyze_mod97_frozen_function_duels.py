"""独立复算冻结函数对决结果和已存权重；不训练。"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tmp/results_mod97_frozen_function_duels_local_pilot_20260912"
OUT = ROOT / "tmp/mod97_frozen_function_duel_analysis_20260912"


def save_json(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def forward_numpy(weights, operands, member):
    def w(name):
        return weights[name][member].cpu().numpy().astype(float)
    depth = sum(key.startswith("hidden_weights.") for key in weights)
    first = w("hidden_weights.0")
    hidden = first[:, operands[:, 0]].T + first[:, 97+operands[:, 1]].T + w("hidden_biases.0")
    for layer in range(depth):
        if layer:
            hidden = hidden @ w(f"hidden_weights.{layer}").T + w(f"hidden_biases.{layer}")
        hidden = .5*hidden*(1+erf(hidden/np.sqrt(2)))
        hidden = (hidden-hidden.mean(1, keepdims=True))/np.sqrt(hidden.var(1, keepdims=True)+1e-5)
        hidden = hidden*w(f"norm_weights.{layer}")+w(f"norm_biases.{layer}")
    return hidden @ w("output_weight").T + w("output_bias")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE/"frozen_functions.npz", allow_pickle=False) as z:
        x = z["operands"].astype(int)
        target, candidates = z["target"], z["candidates"]
    panel = json.loads((SOURCE/"panel.json").read_text(encoding="utf-8"))
    audited_rows = predictions_checked = 0
    finals, histories, checks = [], {}, []
    for folder in sorted((SOURCE/"duels").glob("case_*")):
        if not (folder/"summary.json").exists():
            continue
        summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        cid, dataset = summary["case_id"], summary["dataset"]
        candidate = candidates[cid]
        conflict = candidate != target
        with np.load(folder/"dataset.npz", allow_pickle=False) as z:
            indices = z["train_indices"]
            np.testing.assert_array_equal(z["own_targets"], target)
        np.testing.assert_array_equal(target[indices], candidate[indices])
        assert len(indices) == len(np.unique(indices)) == 6518
        with (folder/"trajectory.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        lookup = {(int(r["step"]), int(r["seed"])): r for r in rows}
        trajectory = []
        for path in sorted((folder/"predictions").glob("step_*.npz")):
            with np.load(path, allow_pickle=False) as z:
                step = int(z["step"])
                if step > summary["step"]:
                    continue
                pred = z["predictions"]
                loss = z["own_ce"]
                for i in range(len(pred)):
                    row = lookup[(step, i)]
                    assert hashlib.sha256(pred[i].tobytes()).hexdigest() == row["function_sha256"]
                    for name, actual in (("train_accuracy", np.mean(pred[i, indices] == target[indices])),
                                         ("conflict_A", np.mean(pred[i, conflict] == target[conflict])),
                                         ("conflict_B", np.mean(pred[i, conflict] == candidate[conflict]))):
                        assert abs(float(row[name])-actual) < 1e-12
                    assert abs(float(row["train_ce"])-float(loss[i, indices].mean())) < 1e-8
                    assert int(row["full_A_errors"]) == int((pred[i] != target).sum())
                    assert int(row["full_B_errors"]) == int((pred[i] != candidate).sum())
                    audited_rows += 1
                predictions_checked += pred.size
                trajectory.append(dict(step=step, train_accuracy=float((pred[:, indices] == target[None, indices]).mean()),
                    A=float((pred[:, conflict] == target[None, conflict]).mean()),
                    B=float((pred[:, conflict] == candidate[None, conflict]).mean()),
                    full_A=int((pred == target).all(1).sum()), full_B=int((pred == candidate).all(1).sum()),
                    models=len(pred), train_ce=float(np.median(loss[:, indices].mean(1)))))
                if step == summary["step"]:
                    final_pred = pred.copy()
        last = trajectory[-1]
        histories[(cid, dataset)] = trajectory
        full_acc = (final_pred == target).mean(1)
        score = (final_pred[:, conflict] == target[conflict]).mean(1)-(final_pred[:, conflict] == candidate[conflict]).mean(1)
        finals.append(dict(case_id=cid, dataset=dataset, **last,
            conflict_count=int(conflict.sum()), A_winning_seeds=int((score > .05).sum()),
            B_winning_seeds=int((score < -.05).sum()), full_A_accuracy_mean=float(full_acc.mean()),
            source_ce=panel["cases"][cid]["source_ce"],
            elapsed_s=summary["elapsed_s"], source_ce_ratio=panel["cases"][cid]["source_ce_ratio"]))
        state = torch.load(folder/"final_weights.pt", map_location="cpu", weights_only=True)
        for member in (0, len(final_pred)-1):
            z = forward_numpy(state["model"], x, member)
            changed = int((z.argmax(1) != final_pred[member]).sum())
            checks.append(dict(type="duel_final", case_id=cid, dataset=dataset, member=member, prediction_changes=changed))
    for folder in sorted((SOURCE/"capacity").iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("capacity_proof_seed*.pt")):
            state = torch.load(path, map_location="cpu", weights_only=True)
            cid = state["cid"]
            label = target if cid < 0 else candidates[cid]
            z = forward_numpy(state["model"], x, 0)
            errors = int((z.argmax(1) != label).sum())
            checks.append(dict(type="capacity_proof", case_id=cid, file=str(path), errors=errors))
    save_json("final_by_dataset.json", finals)
    save_json("weight_replay.json", checks)
    save_json("audit.json", dict(trajectory_rows=audited_rows, prediction_labels=predictions_checked,
                                weight_replays=len(checks), replay_disagreements=sum(r.get("prediction_changes", r.get("errors", 0)) for r in checks)))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    cids = sorted({r["case_id"] for r in finals})
    fig, axes = plt.subplots(len(cids), 2, figsize=(11, 3.5*len(cids)), squeeze=False, constrained_layout=True)
    for i, cid in enumerate(cids):
        for dataset in sorted(d for c, d in histories if c == cid):
            rows = histories[(cid, dataset)]
            steps = [r["step"] for r in rows]
            for key, color in (("A", "#007C91"), ("B", "#B84A48")):
                axes[i, 0].plot(steps, [r[key] for r in rows], color=color, ls="-" if dataset == 0 else "--", label=f"{key}，数据{dataset}")
            axes[i, 1].plot(steps, [r["full_A"]/r["models"] for r in rows], label=f"完整A，数据{dataset}")
        axes[i, 0].set_title(f"候选{cid}：冲突点上A/B偏好")
        axes[i, 1].set_title(f"候选{cid}：完整9312点映射等于A的比例")
        for ax in axes[i]:
            ax.set_xscale("symlog", linthresh=10)
            ax.set_ylim(-.02, 1.02)
            ax.set_xlabel("训练步数")
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
    fig.suptitle("模97冻结函数对决：A为泛化规则，B为grokking前的实际完整映射", fontsize=14)
    fig.savefig(OUT/"duel_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps(finals, ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
