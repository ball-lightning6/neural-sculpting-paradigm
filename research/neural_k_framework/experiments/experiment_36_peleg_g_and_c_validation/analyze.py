"""离线读取冻结结果，不训练、不采样；NumPy float64独立前向复核。"""

import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SHAPES = [(3, 1, 5), (8, 3, 5), (60, 128, 1), (42, 60, 1), (2, 42, 1)]


def dump(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def independent_forward(theta, images):
    values = np.asarray(theta, dtype=np.float64)
    x = np.asarray(images, dtype=np.float64)
    index = 0
    for layer, (co, ci, kernel) in enumerate(SHAPES):
        count = co * ci * kernel * kernel
        w = values[index:index + count].reshape(co, ci, kernel, kernel)
        index += count
        bias = values[index:index + co]
        index += co
        if layer < 2:
            windows = np.lib.stride_tricks.sliding_window_view(x, (kernel, kernel), axis=(-2, -1))
            x = np.einsum("nchwij,ocij->nohw", windows, w, optimize=True) + bias[None, :, None, None]
            x = np.maximum(x, 0)
            n, c, h, width = x.shape
            x = x.reshape(n, c, h // 2, 2, width // 2, 2).max(axis=(3, 5))
        else:
            x = x.reshape(len(images), ci) @ w.reshape(co, ci).T + bias
            if layer < 4:
                x = np.maximum(x, 0)
    assert index == 11074
    return x


def statistics(train_logits, test_logits, train_y, test_y):
    margins = (train_logits[..., 1].astype(np.float64) - train_logits[..., 0].astype(np.float64)) * (2 * train_y - 1)
    with np.errstate(invalid="ignore", over="ignore"):
        loss = np.logaddexp(0, -margins).mean(1)
    hard = (train_logits.argmax(-1) == train_y).all(1)
    finite = np.isfinite(train_logits).all(axis=(1, 2)) & np.isfinite(test_logits).all(axis=(1, 2))
    acc = (test_logits.argmax(-1) == test_y).mean(1)
    valid_hard = finite & hard
    return {"loss": loss, "hard": hard, "finite": finite, "accuracy": acc,
            "n": len(loss), "hard_count": int(valid_hard.sum()), "finite_count": int(finite.sum()),
            "accuracy_all_finite": float(acc[finite].mean()) if finite.any() else None,
            "accuracy_hard": float(acc[valid_hard].mean()) if valid_hard.any() else None,
            "bce_hard_median": float(np.median(loss[valid_hard])) if valid_hard.any() else None,
            "margin_variance_median": float(np.median(margins[valid_hard].var(1))) if valid_hard.any() else None}


def scalar_stats(stats):
    return {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}


def matched_bce(data, target, width):
    meta = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    cuts = meta["config"]["smc_nodes"]
    edges = np.array(sorted({float(f"{x:.12g}") for x in [*np.logspace(-8, 0, round(8 / width) + 1), *cuts]}))
    bins = len(edges)-1
    counts = np.zeros((4, bins), dtype=int)
    acc_sum, loss_sum = np.zeros((4, bins)), np.zeros((4, bins))
    for replica in range(4):
        folder = OUT / "smc" / f"replica_{replica}"
        final = json.loads((folder / "final.json").read_text(encoding="utf-8"))
        # 分箱边界包含所有源事件阈值，因此每箱内混合系数是常数。
        # 不把同一链的终点复测快照当作额外独立副本。
        for node in final.get("nodes", []):
            if not node["label"].startswith("node_"):
                continue
            with np.load(folder / (node["label"]+".npz")) as saved:
                loss = saved["loss"]
                accuracy = (saved["test_logits"].argmax(-1) == data["test_y"]).mean(1)
            indices = np.searchsorted(edges, loss, side="left")-1
            valid = (indices >= 0) & (indices < bins)
            indices, loss, accuracy = indices[valid], loss[valid], accuracy[valid]
            assert np.all(edges[indices+1] <= node["upper"]+1e-13)
            counts[replica] += np.bincount(indices, minlength=bins)
            acc_sum[replica] += np.bincount(indices, weights=accuracy, minlength=bins)
            loss_sum[replica] += np.bincount(indices, weights=loss, minlength=bins)
    idx = np.searchsorted(edges, target["loss"], side="left")-1
    valid = (idx >= 0) & (idx < bins) & target["finite"] & target["hard"]
    support = (counts.min(0) >= 5) & (counts.sum(0) >= 40)
    retained = valid & support[np.clip(idx, 0, bins-1)]
    target_counts = np.bincount(idx[retained], minlength=bins)
    weights = target_counts / target_counts.sum()
    acc_conditional = np.divide(acc_sum, counts, out=np.zeros_like(acc_sum), where=counts > 0)
    loss_conditional = np.divide(loss_sum, counts, out=np.zeros_like(loss_sum), where=counts > 0)
    sampled_acc = acc_conditional @ weights
    sampled_loss = loss_conditional @ weights
    rng = np.random.default_rng(1909111)
    bootstrap = []
    for _ in range(2000):
        model_ids = rng.integers(0, len(idx), len(idx))
        model_ids = model_ids[retained[model_ids]]
        local = np.bincount(idx[model_ids], minlength=bins)
        local = local / local.sum()
        run_ids = rng.integers(0, 4, 4)
        bootstrap.append(target["accuracy"][model_ids].mean() - (acc_conditional[run_ids] @ local).mean())
    return {
        "log10_bin_width": width, "retained_sgd": int(retained.sum()), "total_sgd": len(idx),
        "coverage": float(retained.mean()), "support_rule": "每箱4副本各至少5粒子、总计至少40；仅按loss决定",
        "sgd_accuracy": float(target["accuracy"][retained].mean()), "smc_matched_accuracy": float(sampled_acc.mean()),
        "smc_matched_accuracy_per_replica": sampled_acc.tolist(),
        "sgd_bce_mean": float(target["loss"][retained].mean()), "smc_bce_mean": float(sampled_loss.mean()),
        "sgd_minus_smc_pp": float((target["accuracy"][retained].mean()-sampled_acc.mean())*100),
        "bootstrap_ci95_pp": (np.quantile(bootstrap, [.025,.975])*100).tolist(),
        "bin_edges": edges.tolist(), "sgd_bin_counts": target_counts.tolist(), "smc_bin_counts": counts.tolist(),
        "scope": "共同支持子集的有限分箱匹配；非全体SGD或精确连续等loss；区间不包含共同混合偏差",
    }


def main():
    dest = HERE / "analysis"
    dest.mkdir(exist_ok=True)
    with np.load(OUT / "dataset.npz") as file:
        data = {key: file[key] for key in file.files}
    meta = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((HERE / "experiment.py").read_bytes()).hexdigest() == meta["source_sha256"]
    source_ok = True
    manifest = json.loads((HERE / "upstream/manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((HERE / "upstream" / name).read_bytes()).hexdigest() == expected
    records, train_arrays, audits = [], {}, []
    probe_ids = np.random.default_rng(190999).choice(len(data["test_x"]), 64, replace=False)
    audit_x = np.concatenate([data["train_x"], data["test_x"][probe_ids]])
    for folder in sorted((OUT / "train").iterdir()):
        info = json.loads((folder / "final.json").read_text(encoding="utf-8"))
        with np.load(folder / "train_trajectory.npz") as saved:
            train_logits = saved["logits"][-1]
            bias_norm = saved["bias_l2"]
            outside = saved["weight_outside_fraction"]
        with np.load(folder / "test_epoch_060.npz") as saved:
            test_logits = saved["logits"]
        stats = statistics(train_logits, test_logits, data["train_y"], data["test_y"])
        train_arrays[info["condition"]] = stats
        rec = {"name": info["condition"], "scale": info["scale"], "lr": info["lr"],
               "train_bias": info["train_bias"], "training_seconds": info["training_seconds"],
               "evaluation_seconds": info["evaluation_seconds"], **scalar_stats(stats),
               "bias_l2_median": float(np.median(bias_norm[stats["finite"]])), "outside_weight_fraction_mean": float(np.mean(outside[stats["finite"]]))}
        assert abs(stats["accuracy_all_finite"] - info["final"]["test"]["accuracy_mean"]) < 1e-12
        weights = torch.load(folder / "final_weights.pt", map_location="cpu", weights_only=True).numpy()
        assert hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest() == info["weights_sha256"]
        refs = np.concatenate([train_logits[:16], test_logits[:16, probe_ids]], axis=1)
        max_difference, max_relative, flips, checked = 0.0, 0.0, 0, 0
        for i in range(16):
            if not np.isfinite(weights[i]).all() or not np.isfinite(refs[i]).all():
                continue
            z = independent_forward(weights[i], audit_x)
            max_difference = max(max_difference, float(np.max(np.abs(z - refs[i]))))
            max_relative = max(max_relative, float(np.linalg.norm(z-refs[i]) / max(1., np.linalg.norm(refs[i].astype(np.float64)))))
            flips += int((z.argmax(-1) != refs[i].argmax(-1)).sum())
            checked += 1
        audits.append({"kind": "sgd", "name": info["condition"], "models": 16,
                       "actually_checked_models": checked, "nonfinite_skipped": 16-checked,
                       "inputs_per_model": len(audit_x), "max_logit_abs_difference": max_difference,
                       "max_relative_l2_difference": max_relative, "class_flips": flips})
        records.append(rec)
    bootstrap = np.random.default_rng(19090000).integers(0, 512, size=(4000, 512))
    effects = []
    for scale in [1.0, .2]:
        for lr in [.001, .01]:
            zero = train_arrays[f"a{scale:g}_lr{lr:g}_biaszero"]
            free = train_arrays[f"a{scale:g}_lr{lr:g}_biastrain"]
            valid = free["finite"] & zero["finite"]
            delta = (free["accuracy"] - zero["accuracy"])[valid]
            draw = bootstrap if len(delta) == 512 else np.random.default_rng(19090001).integers(0, len(delta), size=(4000, len(delta)))
            all_finite = bool(zero["finite"].all() and free["finite"].all())
            effects.append({"scale": scale, "lr": lr, "free_minus_zero_pp": float(delta.mean() * 100),
                            "initialization_bootstrap_ci95_pp": (np.quantile(delta[draw].mean(1), [.025, .975]) * 100).tolist(),
                            "both_fully_fitted": int((free["hard"] & zero["hard"] & valid).sum()), "all_finite": all_finite,
                            "common_finite_models": int(valid.sum())})
    smc, endings = [], []
    for folder in sorted((OUT / "smc").iterdir()):
        final = json.loads((folder / "final.json").read_text(encoding="utf-8"))
        endings.append({"replica": folder.name, "status": final["status"], "gc": final["gc"],
                        "sampling_seconds": final.get("sampling_seconds"), "total_seconds": final.get("total_replica_seconds")})
        node_records = list(final.get("nodes", []))
        check_file = folder / "endpoint_check.json"
        if check_file.exists():
            continuation = json.loads(check_file.read_text(encoding="utf-8"))
            endings[-1]["endpoint_check_seconds"] = continuation["extra_seconds"]
            node_records += continuation["nodes"]
        for record in node_records:
            with np.load(folder / (record["label"] + ".npz")) as saved:
                tr, te, weights = saved["train_logits"], saved["test_logits"], saved["audit_theta"]
                stored_loss = saved["loss"]
            stats = statistics(tr, te, data["train_y"], data["test_y"])
            assert stats["hard_count"] == len(tr)
            assert np.max(np.abs(stored_loss - stats["loss"])) < 1e-12
            assert stats["loss"].max() <= record["upper"] + 1e-12
            refs = np.concatenate([tr[:16], te[:16, probe_ids]], axis=1)
            differences, flips, over = [], 0, 0.0
            for i in range(len(weights)):
                z = independent_forward(weights[i], audit_x)
                differences.append(float(np.max(np.abs(z - refs[i]))))
                flips += int((z.argmax(-1) != refs[i].argmax(-1)).sum())
                signed = (z[:16, 1] - z[:16, 0]) * (2 * data["train_y"] - 1)
                over = max(over, float(np.logaddexp(0, -signed).mean() - record["upper"]))
            audits.append({"kind": "smc", "name": folder.name + "/" + record["label"], "models": len(weights),
                           "inputs_per_model": len(audit_x), "max_logit_abs_difference": max(differences),
                           "class_flips": flips, "float64_loss_excess": over})
            probs = (te.argmax(-1) == 1).mean(0)
            smc.append({"replica": folder.name, **record, **scalar_stats(stats),
                        "mean_pairwise_agreement": float(np.mean(probs**2 + (1 - probs)**2)),
                        "distinct_test_mappings": len(np.unique(np.packbits(te.argmax(-1), axis=1), axis=0))})
    labels = sorted(set(row["label"] for row in smc if row["label"].startswith("node_")))
    shared = []
    for label in labels:
        group = [row for row in smc if row["label"] == label]
        if len(group) != 4:
            continue
        shared.append({"label": label, "upper": group[0]["upper"],
                       "accuracy_mean": float(np.mean([row["accuracy_all_finite"] for row in group])),
                       "accuracy_replica_min": min(row["accuracy_all_finite"] for row in group),
                       "accuracy_replica_max": max(row["accuracy_all_finite"] for row in group),
                       "actual_bce_mean": float(np.mean([row["loss_mean"] for row in group])),
                       "K_given_hard_mean": float(np.mean([row["K_given_hard"] for row in group]))})
    matched = [matched_bce(data, train_arrays["a0.2_lr0.01_biaszero"], width) for width in [.1, .05]]
    endpoints = [r for r in smc if r["label"] == "check_0512"]
    endpoint_summary = {"replicas": len(endpoints), "accuracy_mean": float(np.mean([r["accuracy_all_finite"] for r in endpoints])),
                        "accuracy_min": min(r["accuracy_all_finite"] for r in endpoints),
                        "accuracy_max": max(r["accuracy_all_finite"] for r in endpoints)} if endpoints else None
    result = {"training": records, "paired_bias_effects": effects, "smc_nodes": smc, "smc_endings": endings,
              "shared_smc_nodes": shared, "audit": audits, "source_hashes_unchanged": source_ok,
              "matched_bce": matched, "endpoint_check": endpoint_summary,
              "heldout_score_count": len(data["test_y"]), "dataset_count": 1}
    dump(dest / "summary.json", result)
    dump(dest / "numerical_audit.json", audits)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    configs = [(1, .001), (1, .01), (.2, .001), (.2, .01)]
    positions = np.arange(4)
    for flag, shift, color, label in [(False, -.18, "#3973ac", "bias固定0"), (True, .18, "#d37b34", "bias可训练")]:
        vals = [next(r["accuracy_all_finite"] for r in records if r["scale"] == a and r["lr"] == lr and r["train_bias"] == flag) * 100 for a, lr in configs]
        axes[0].bar(positions + shift, vals, width=.34, color=color, label=label)
    axes[0].set_xticks(positions, [f"幅度{a}\nlr={lr}" for a, lr in configs])
    axes[0].set_ylim(45, 100)
    axes[0].set_ylabel("平均单模型测试准确率（%）")
    axes[0].set_title("60epoch：配对训练对照")
    axes[0].legend()
    for flag, color, label in [(False, "#3973ac", "SGD固定bias"), (True, "#d37b34", "SGD可训练bias")]:
        stats = train_arrays[f"a0.2_lr0.01_bias{'train' if flag else 'zero'}"]
        axes[1].scatter(stats["loss"], stats["accuracy"] * 100, s=8, color=color, alpha=.2, label=label)
    for replica in range(4):
        rs = sorted([r for r in smc if r["replica"] == f"replica_{replica}" and not r["label"].startswith("check_")], key=lambda r: -r["loss_mean"])
        axes[1].plot([r["loss_mean"] for r in rs], [r["accuracy_all_finite"] * 100 for r in rs], ".-", color="#39845d", alpha=.65,
                     label="SMC独立副本" if replica == 0 else None)
        axes[2].plot([r["loss_mean"] for r in rs], [r["K_given_hard"] for r in rs], ".-", label=f"副本{replica}")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("实际平均训练BCE（向右更低）")
    axes[1].set_title("共同参考：初始幅度0.2")
    axes[1].legend(loc="lower right")
    axes[2].set_xscale("log")
    axes[2].invert_xaxis()
    axes[2].set_xlabel("实际平均训练BCE（向右更低）")
    axes[2].set_ylabel("相对硬拟合父事件的K（bit）")
    axes[2].set_title("短预算SMC：非全局混合认证")
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(dest / "overview.png", dpi=170)
    plt.close(fig)
    text = ["# Peleg本地复核：第一轮结果", "", "固定一个16样本训练集；物理GPU1 RTX3080。不是全论文复现。", "", "## SGD对照", "",
            "|初始化幅度|学习率|bias可训练|有限数值模型|训练全对|测试准确率均值|仅训练全对者的测试均值|硬拟合者BCE中位数|训练秒数|",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        hard_acc = "无" if row["accuracy_hard"] is None else f"{row['accuracy_hard'] * 100:.4f}%"
        bce = "无" if row["bce_hard_median"] is None else f"{row['bce_hard_median']:.5g}"
        text.append(f"|{row['scale']}|{row['lr']}|{row['train_bias']}|{row['finite_count']}/{row['n']}|{row['hard_count']}/{row['n']}|{row['accuracy_all_finite']*100:.4f}%|{hard_acc}|{bce}|{row['training_seconds']:.2f}|")
    text += ["", "## bias配对差值", "", "单位为百分点；区间仅描述固定D下初始化差异，不包含重抽数据的不确定性。", ""]
    for row in effects:
        text.append(f"- 幅度{row['scale']}，lr={row['lr']}：可训练减固定={row['free_minus_zero_pp']:.4f}，95%区间{row['initialization_bootstrap_ci95_pp']}。")
    text += ["", "## 共同SMC节点", "", "只列4个独立作业都完成的节点。", "",
             "|BCE上限|实际BCE均值|平均单粒子测试准确率|副本范围|K相对hard父事件|", "|---:|---:|---:|---|---:|"]
    for row in sorted(shared, key=lambda r: -r["upper"]):
        text.append(f"|{row['upper']}|{row['actual_bce_mean']:.6g}|{row['accuracy_mean']*100:.4f}%|{row['accuracy_replica_min']*100:.3f}%至{row['accuracy_replica_max']*100:.3f}%|{row['K_given_hard_mean']:.3f}|")
    text += ["", "## BCE共同支持匹配", "", "只匹配a=0.2、lr=0.01、bias固定0的SGD终点。分箱包含全部SMC源阈值；不按准确率选箱。此分析为pilot结束后的探索性分箱，尚非独立确认。", ""]
    for row in matched:
        text.append(f"- log10箱宽{row['log10_bin_width']}：覆盖{row['retained_sgd']}/{row['total_sgd']}，SGD={row['sgd_accuracy']*100:.4f}%，SMC={row['smc_matched_accuracy']*100:.4f}%；SGD-SMC={row['sgd_minus_smc_pp']:.4f}个百分点，区间{row['bootstrap_ci95_pp']}。平均BCE分别{row['sgd_bce_mean']:.7g}/{row['smc_bce_mean']:.7g}。")
    if endpoint_summary:
        text += ["", "## 固定核终点继续更新", "", f"每副本最多额外512轮，均在原120秒SMC预算剩余时间内。最终平均准确率{endpoint_summary['accuracy_mean']*100:.4f}%，副本范围{endpoint_summary['accuracy_min']*100:.4f}%至{endpoint_summary['accuracy_max']*100:.4f}%。这不是全局混合证明。"]
    text += ["", "## 边界", "", "- SMC终点和SGD终点只有在实际BCE分布有共同支持时才能做matched比较；图上靠近不等于分布已匹配。",
             "- 稳定BCE为两类CE的等价数学表达，训练本身保留原生CrossEntropyLoss；特别低的native CE可能舍入零。",
             "- 不把未拟合SGD、采样时间截断或共同支持不足当作泛化反例。",
             "- NumPy float64抽查见numerical_audit.json，原始权重和逐点输出仍保留。", "",
             "![训练与采样总览](overview.png)"]
    (dest / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"training": records, "paired_bias_effects": effects, "shared_smc_nodes": shared, "smc_endings": endings}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
