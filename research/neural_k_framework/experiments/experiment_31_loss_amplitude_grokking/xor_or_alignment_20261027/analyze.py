"""只读复核新结果、输出表格与图；不调用采样器。"""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
KINDS = ("mean_bce", "max_bce", "aligned")
LABELS = {"mean_bce": "Mean BCE", "max_bce": "Pointwise BCE cap", "aligned": "BCE + 0.1 Var(margin)"}
COLORS = {"mean_bce": "#2777b4", "max_bce": "#b25042", "aligned": "#25876a"}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def volume_pooled_k(k):
    k = np.asarray(k)
    return float(k.min() - np.log2(np.exp2(-(k-k.min())).mean()))


def verify_snapshot(path, row, fid):
    with np.load(path, allow_pickle=False) as d:
        t = d["theta"].astype(np.float64)
        x = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(float)
        w1, b1, w2, b2 = t[:, :64].reshape(-1, 16, 4) / 2, t[:, 64:80] / 2, t[:, 80:96] / 4, t[:, 96] / 4
        z = (np.tanh(x[None] @ w1.transpose(0, 2, 1) + b1[:, None]) * w2[:, None]).sum(-1) + b2[:, None]
        m = z * (2 * ((fid >> np.arange(16)) & 1) - 1)
        ell = np.logaddexp(0, -m)
        score = {"mean_bce": lambda: ell.mean(-1), "max_bce": lambda: ell.max(-1),
                 "aligned": lambda: ell.mean(-1) + .1 * m.var(-1)}[row["kind"]]()
        np.testing.assert_allclose(z, d["logits"], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(score, d["score"], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(ell.mean(), row["after_refresh"]["bce_mean"], atol=1e-10, rtol=1e-10)
        tolerance = 2e-6 + 2e-4 * row["threshold"]
        assert np.all(score <= row["threshold"] + tolerance)
        return len(t), float(np.max(np.abs(score-d["torch_score"]))), hashlib.sha256(path.read_bytes()).hexdigest()


def historical_baseline():
    entry = os.environ.get("XOR_ALIGNMENT_HISTORICAL_PACKAGE")
    if not entry:
        return {"available": False, "note": "历史扫描包未随发布提供；可选环境变量 XOR_ALIGNMENT_HISTORICAL_PACKAGE 指定入口，不影响本轮复算。"}
    package = Path(entry).expanduser()
    if not package.exists():
        raise FileNotFoundError("指定的历史扫描包不存在")
    output = []
    with zipfile.ZipFile(package) as z:
        for rule, representative, phase in (("f", "0x0FF0", "first"), ("g", "0x0099", "first"),
                                            ("f", "0x0FF0", "recheck")):
            folder = f"results_4bit_full_curve_scan/targets/{representative}/{phase}/"
            marker = json.loads(z.read(folder + "completed.json"))
            raw = z.read(folder + "replica_volume_curve.csv")
            assert hashlib.sha256(raw).hexdigest() == marker["files"]["replica_volume_curve.csv"]
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            for threshold in (.01, .001):
                k = [-float(r["log_volume"]) / math.log(2) for r in rows if float(r["threshold"]) == threshold]
                assert len(k) == 8
                output.append(dict(rule=rule, representative=representative, phase=phase,
                                   threshold=threshold, K_median=float(np.median(k)), K_range=[min(k), max(k)]))
    return dict(available=True, package=str(package), original_csv_hashes_verified=True,
                g_mapping="输入变量置换和输出整体取反将 0xCFFC 映到 0x0099；不使用输入位取反。",
                values=output)


def analyze(out):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    jobs = [f"{rule}__{kind}" for kind in KINDS for rule in ("f", "g")]
    if out.name == "results_confirm":
        jobs = json.loads((ROOT / "CONFIRMATION_SCOPE.json").read_text(encoding="utf-8"))["jobs"]
    assert hashlib.sha256((out / "frozen_source.py").read_bytes()).hexdigest() == protocol["source_sha256"]
    records, complete, checks = [], [], []
    total_particles, maximum_error = 0, 0.0
    for kind in KINDS:
        for rule, fid in protocol["rules"].items():
            folder = out / f"{rule}__{kind}"
            if (folder / "summary.json").exists():
                complete.append(json.loads((folder / "summary.json").read_text(encoding="utf-8")))
                levels = json.loads((folder / "levels.json").read_text(encoding="utf-8"))
                accumulated = np.zeros(protocol["settings"]["replicas"])
                for level in levels:
                    kept = np.asarray(level["kept"])
                    assert np.all((kept > 0) & (kept <= protocol["settings"]["particles"]))
                    accumulated -= np.log2(kept / protocol["settings"]["particles"])
                    np.testing.assert_allclose(accumulated, level["K"], atol=2e-9, rtol=0)
                np.testing.assert_allclose(accumulated, complete[-1]["K_bits"], atol=2e-9, rtol=0)
            if not (folder / "nodes.json").exists():
                continue
            nodes = json.loads((folder / "nodes.json").read_text(encoding="utf-8"))
            for row in nodes:
                n, error, digest = verify_snapshot(folder / row["snapshot"], row, fid)
                total_particles += n
                maximum_error = max(maximum_error, error)
                checks.append(dict(file=f"{folder.name}/{row['snapshot']}", sha256=digest, particles=n))
                records.append(row)
    aggregates, differences = [], []
    for kind in KINDS:
        for threshold in protocol["settings"]["thresholds"]:
            pair = {}
            for rule in ("f", "g"):
                rows = sorted([r for r in records if r["kind"] == kind and r["rule"] == rule
                               and r["threshold"] == threshold], key=lambda r: r["replica"])
                if len(rows) != protocol["settings"]["replicas"]:
                    continue
                assert [r["replica"] for r in rows] == list(range(protocol["settings"]["replicas"]))
                k = np.array([r["K_bits"] for r in rows])
                pair[rule] = k
                d = dict(kind=kind, rule=rule, threshold=threshold,
                         K_median=float(np.median(k)), K_min=float(k.min()), K_max=float(k.max()),
                         K_sd=float(k.std(ddof=1)), K_pooled=volume_pooled_k(k))
                for name in ("bce_mean", "max_bce_mean", "margin_variance_mean", "hard_exact_fraction",
                             "c0_bce_mean", "c1_bce_mean", "c0_margin_mean", "c1_margin_mean",
                             "parameter_norm2_mean"):
                    d[name] = float(np.mean([r["after_refresh"][name] for r in rows]))
                d["refresh_bce_change_max"] = float(max(abs(r["after_refresh"]["bce_mean"] -
                                                           r["before_refresh"]["bce_mean"]) for r in rows))
                d["refresh_var_change_max"] = float(max(abs(r["after_refresh"]["margin_variance_mean"] -
                                                           r["before_refresh"]["margin_variance_mean"]) for r in rows))
                aggregates.append(d)
            if len(pair) == 2:
                differences.append(dict(kind=kind, threshold=threshold,
                                        delta_K_median=float(np.median(pair["g"])-np.median(pair["f"])),
                                        delta_K_pooled=volume_pooled_k(pair["g"])-volume_pooled_k(pair["f"]),
                                        cross_pool_min=float(pair["g"].min()-pair["f"].max()),
                                        cross_pool_max=float(pair["g"].max()-pair["f"].min())))
    write_csv(out / "aggregates.csv", aggregates)
    write_csv(out / "differences.csv", differences)
    write_json(out / "INDEPENDENT_VERIFICATION.json", dict(passed=True, snapshots=len(checks),
               particles_recomputed=total_particles, max_float32_float64_score_difference=maximum_error,
               source_hash_verified=True, snapshot_hashes=checks))
    report = dict(tier=out.name, completed_jobs=len(complete), total_jobs=len(jobs), scheduled_jobs=jobs,
                  total_job_seconds=sum(r["elapsed_seconds"] for r in complete),
                  aggregates=aggregates, differences=differences)
    write_json(out / "analysis.json", report)
    write_json(out / "HISTORICAL_BASELINE.json", historical_baseline())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for kind in KINDS:
        for rule, style in (("f", "-"), ("g", "--")):
            rows = [r for r in aggregates if r["kind"] == kind and r["rule"] == rule]
            if not rows:
                continue
            x = [-math.log10(r["threshold"]) for r in rows]
            axes[0, 0].plot(x, [r["K_median"] for r in rows], style, color=COLORS[kind], label=f"{LABELS[kind]}, {rule}")
            axes[1, 0].plot(x, [r["bce_mean"]/r["threshold"] for r in rows], style, color=COLORS[kind])
            axes[1, 1].plot(x, [r["margin_variance_mean"] for r in rows], style, color=COLORS[kind])
        rows = [r for r in differences if r["kind"] == kind]
        if rows:
            x = [-math.log10(r["threshold"]) for r in rows]
            axes[0, 1].plot(x, [r["delta_K_median"] for r in rows], color=COLORS[kind], label=LABELS[kind])
            axes[0, 1].fill_between(x, [r["cross_pool_min"] for r in rows], [r["cross_pool_max"] for r in rows],
                                    color=COLORS[kind], alpha=.13)
    axes[0, 0].set_ylabel("Median pool K (bits)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].axhline(0, color="black", linewidth=.8)
    axes[0, 1].set_ylabel("K(g) - K(f) (bits)")
    if axes[0, 1].get_legend_handles_labels()[0]:
        axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_ylabel("Actual mean BCE / score threshold")
    axes[1, 1].set_ylabel("Mean signed-margin variance")
    axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("-log10(score threshold)")
        ax.grid(alpha=.15)
    fig.suptitle("4-bit XOR and (XOR OR c): fixed Gaussian measure\nShading: all cross-pool differences, not confidence intervals")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# 本地幅度对齐结果", "", f"预算档：{out.name}；已完成 {len(complete)}/{len(jobs)} 个已安排作业。", "",
             "网络为 4→16→1 tanh，固定标准 Gaussian 神经坐标；完整 16 点全规则约束。",
             "f=a XOR b；g=(a XOR b) OR c。表中 K 为独立池 K 中位数，不是平均质量取负对数。", "",
             "## 排序", "", "| 分数阈值 | 普通平均 BCE：K_g-K_f | 逐点最大 BCE：K_g-K_f | BCE+0.1 Var(m)：K_g-K_f |",
             "|---:|---:|---:|---:|"]
    for threshold in (.3, .08, .01, .004, .001):
        vals = []
        for kind in KINDS:
            found = next((r for r in differences if r["kind"] == kind and r["threshold"] == threshold), None)
            missing = "本档未安排" if f"f__{kind}" not in jobs else "尚未完成"
            vals.append(f"{found['delta_K_median']:.3f}" if found else missing)
        lines.append(f"| {threshold} | " + " | ".join(vals) + " |")
    lines.extend(["", "## 实际 BCE 和幅度", "", "| 条件 | 规则 | 阈值 | 实际平均 BCE | c=0 BCE | c=1 BCE | margin 方差 | K 池范围 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in aggregates:
        if r["threshold"] in (.01, .001):
            lines.append(f"| {r['kind']} | {r['rule']} | {r['threshold']} | {r['bce_mean']:.6g} | "
                         f"{r['c0_bce_mean']:.6g} | {r['c1_bce_mean']:.6g} | {r['margin_variance_mean']:.6g} | "
                         f"{r['K_min']:.2f}–{r['K_max']:.2f} |")
    lines.extend(["", "## 核验与边界", "",
                  f"- 独立 float64 逐权重复算：{len(checks)} 个快照，{total_particles} 个粒子通过；最大分数差 {maximum_error:.3g}。",
                  "- 全部交叉池差值范围只描述本批重复波动，不是全局混合证明或正式置信区间。",
                  "- 三个分数定义三个不同事件；aligned 表中的阈值不是其实际 raw BCE。",
                  "- 只有观察到差距缩小，不等于已经发生排序反转；没有交叉也不否定其他预先指定剂量可能发生交叉。",
                  "- 本实验不证明组合逻辑在所有架构上的普遍复杂度排序，也不证明 SMC 与 SGD 分布相等。",
                  "", "![对照曲线](comparison.png)", ""])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"tier":out.name,"completed_jobs":len(complete),"snapshots":len(checks),
                      "deepest": [r for r in differences if r["threshold"] == .001]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    for folder in sorted(ROOT.glob("results_*")):
        if (folder / "protocol.json").exists():
            analyze(folder)
