"""8份随机数据的完整记忆标度分析；复用首轮审计，不修改旧结果或运行新采样。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

import analyze_8bit_random_memory_scaling as audit_core


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(r"E:\Downloads\results_8bit_random_memory_scaling_w16_v1_package (1).zip")
OUT = ROOT / "tmp/8bit_random_memory_scaling_full_analysis_20260912"
OLD = ROOT / "tmp/8bit_random_memory_scaling_w16_analysis_20260911/analysis.json"
BOOTSTRAP_SEED = 202609120101
BOOTSTRAPS = 10000


def write_json(path, value):
    path.write_text(json.dumps(audit_core.clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(audit_core.clean(v), ensure_ascii=False) if isinstance(v, (dict, list))
                         else v for k, v in row.items()} for row in rows)


def data_audit(protocol):
    records, labels, hashes = [], [], []
    with zipfile.ZipFile(PACKAGE) as archive:
        for dataset_id in range(8):
            data = audit_core.read_npz(archive, f"data_{dataset_id:03d}/full_random_data.npz")
            expected_seed = protocol["DATA_SEED"] + 1_000_003 * dataset_id
            assert int(data["seed"]) == expected_seed
            rng = np.random.default_rng(expected_seed)
            assert np.array_equal(rng.permutation(256), data["order"])
            assert np.array_equal(rng.integers(0, 2, 256, dtype=np.uint8), data["labels"])
            h = hashlib.sha256(data["labels"].tobytes() + data["order"].tobytes()).hexdigest()
            records.append(dict(dataset_id=dataset_id, seed=expected_seed, ones=int(data["labels"].sum()), sha256=h))
            hashes.append(h)
            labels.append(data["labels"])
    assert len(set(hashes)) == 8
    distances = [int(np.sum(labels[i] != labels[j])) for i in range(8) for j in range(i + 1, 8)]
    return dict(datasets=records, unique_datasets=8, pairwise_label_hamming_range=[min(distances), max(distances)])


def slope_lookup(jobs):
    return {(job["dataset_id"], job["n"], row["high"], row["low"]): row
            for job in jobs for row in job["slopes"]}


def window_statistics(jobs, lookup, counts):
    groups, fits, individual_fits, matrices = [], [], [], {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 8, (BOOTSTRAPS, 8))
    x = np.column_stack((np.ones(len(counts)), counts))
    x_inverse = np.linalg.pinv(x)
    all_windows = audit_core.WINDOWS + ((.01, .001),)
    job_lookup = {(j["dataset_id"], j["n"]): j for j in jobs}
    for high, low in all_windows:
        matrix = np.array([[lookup[(d, n, high, low)]["median"] for n in counts] for d in range(8)])
        matrices[(high, low)] = matrix
        average = matrix.mean(0)
        median = np.median(matrix, axis=0)
        sd = matrix.std(0, ddof=1)
        full_fit = audit_core.fit_line(counts, average)
        boot_means = matrix[draws].mean(1)
        parameters = boot_means @ x_inverse.T
        ci = np.quantile(parameters, [.025, .975], axis=0)
        bootstrap_fit = parameters @ x.T
        r_squared = 1 - np.sum((boot_means - bootstrap_fit) ** 2, axis=1) / np.sum((boot_means - boot_means.mean(1)[:, None]) ** 2, axis=1)
        early = np.array(counts) <= 192
        restricted = audit_core.fit_line(np.array(counts)[early], average[early])
        late = []
        for n in (224, 256):
            index = counts.index(n)
            expected = restricted["slope"] * n + restricted["intercept"]
            late.append(dict(n=n, actual_mean=float(average[index]), extrapolated_from_n_le192=float(expected),
                             relative_excess=float(average[index] / expected - 1)))
        fits.append(dict(high=high, low=low, **full_fit,
            median_curve_fit=audit_core.fit_line(counts, median),
            dataset_cluster_bootstrap=dict(slope_95_interval=ci[:, 1].tolist(), intercept_95_interval=ci[:, 0].tolist(),
                r_squared_95_interval=np.quantile(r_squared, [.025, .975]).tolist(),
                note="整份嵌套数据轨迹重抽；仅反映这8份数据与本次估计，不覆盖SMC系统性偏差。"),
            n_le192_sensitivity_fit=restricted, late_excess=late,
            sensitivity_note="观察最大两档波动后做的范围敏感性分析，不替代全10档主分析。"))
        for d in range(8):
            individual_fits.append(dict(dataset_id=d, high=high, low=low,
                **audit_core.fit_line(counts, matrix[d]),
                increasing_adjacent_pairs=int(np.sum(np.diff(matrix[d]) > 0)), adjacent_pairs=len(counts) - 1))
        for j, n in enumerate(counts):
            warning_ids, slope_warning_ids = [], []
            for d in range(8):
                points = {p["bce_limit"]: p for p in job_lookup[(d, n)]["points"]}
                row = lookup[(d, n, high, low)]
                slope_warn = row["replica_sd"] > max(5., .15 * abs(row["median"]))
                if slope_warn:
                    slope_warning_ids.append(d)
                if slope_warn or points[high]["warnings"] or points[low]["warnings"]:
                    warning_ids.append(d)
            groups.append(dict(high=high, low=low, n=n, datasets=8, c_mean=float(average[j]),
                c_dataset_sd=float(sd[j]), c_dataset_median=float(median[j]),
                c_per_n=float(average[j] / n), c_by_dataset=matrix[:, j].tolist(),
                median_replica_sd=float(np.median([lookup[(d, n, high, low)]["replica_sd"] for d in range(8)])),
                c_replica_sd_by_dataset=[lookup[(d, n, high, low)]["replica_sd"] for d in range(8)],
                sampler_warning_ids=warning_ids, slope_spread_warning_ids=slope_warning_ids,
                sampler_warning_datasets=len(warning_ids)))
    return groups, fits, individual_fits, matrices


def tail_stability(jobs, lookup):
    rows = []
    for job in jobs:
        d, n = job["dataset_id"], job["n"]
        first = lookup[(d, n, .2, .1)]
        second = lookup[(d, n, .1, .05)]
        penultimate = lookup[(d, n, .005, .002)]
        last = lookup[(d, n, .002, .001)]
        sequence = [lookup[(d, n, h, l)]["median"] for h, l in audit_core.WINDOWS]
        rows.append(dict(dataset_id=d, n=n, early_drop_fraction=1 - second["median"] / first["median"],
                        last_drop_fraction=1 - last["median"] / penultimate["median"],
                        last_c=last["median"], last_c_replica_sd=last["replica_sd"],
                        monotonic_decrease_all_windows=bool(np.all(np.diff(sequence) <= 0)),
                        endpoint_warnings=job["endpoint"]["warnings"]))
    drops = np.array([r["last_drop_fraction"] for r in rows])
    early = np.array([r["early_drop_fraction"] for r in rows])
    return dict(rows=rows, last_drop_median=float(np.median(drops)),
                last_drop_range=[float(drops.min()), float(drops.max())],
                last_drop_10_90_quantiles=np.quantile(drops, [.1, .9]).tolist(),
                early_drop_median=float(np.median(early)),
                last_window_increases=[r for r in rows if r["last_drop_fraction"] < 0],
                fully_decreasing_curves=sum(r["monotonic_decrease_all_windows"] for r in rows),
                last_relative_change_within_5pct=int(np.sum(np.abs(drops) <= .05)),
                last_relative_change_within_10pct=int(np.sum(np.abs(drops) <= .1)))


def compare_hardware(jobs):
    old = json.loads(OLD.read_text(encoding="utf-8"))
    rows = []
    for previous in old["jobs"]:
        current = next(j for j in jobs if j["dataset_id"] == 0 and j["n"] == previous["n"])
        for high, low in ((.2, .1), (.002, .001)):
            before = next(r for r in previous["slopes"] if (r["high"], r["low"]) == (high, low))
            after = next(r for r in current["slopes"] if (r["high"], r["low"]) == (high, low))
            rows.append(dict(n=current["n"], high=high, low=low, old_4090d_c=before["median"],
                             new_5090_c=after["median"], relative_change=after["median"] / before["median"] - 1))
    old_seconds = sum(j["seconds"] for j in old["jobs"])
    new_seconds = sum(j["seconds"] for j in jobs if j["dataset_id"] == 0)
    return dict(rows=rows, old_4090d_seconds=old_seconds, new_5090_seconds=new_seconds,
                speed_ratio=old_seconds / new_seconds,
                note="同数据、同配置、同种子换GPU重跑；不计作第9份独立数据或独立随机流复测。")


def plot(groups, matrices, jobs, lookup, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    windows = ((.2, .1), (.1, .05), (.01, .005), (.002, .001))
    for color, (high, low) in enumerate(windows):
        data = matrices[(high, low)]
        mean, sd = data.mean(0), data.std(0, ddof=1)
        axes[0, 0].errorbar(counts, mean, yerr=sd, fmt="o-", markersize=4, capsize=3, color=f"C{color}", label=f"{high:g}→{low:g}")
        axes[1, 1].errorbar(counts, mean / counts, yerr=sd / counts, fmt="o-", markersize=4, capsize=3, color=f"C{color}", label=f"{high:g}→{low:g}")
    axes[0, 0].set(title="A  各loss窗口：均值与数据集间SD", xlabel="训练样本数 n", ylabel="收缩系数 c")
    axes[0, 0].legend(fontsize=9)
    data = matrices[(.002, .001)]
    for d in range(8):
        axes[0, 1].plot(counts, data[d], ".-", linewidth=1, alpha=.65, label=f"数据{d + 1}")
    axes[0, 1].plot(counts, data.mean(0), "ko-", linewidth=2.3, markersize=4, label="8份均值")
    fit = audit_core.fit_line(counts, data.mean(0))
    axes[0, 1].plot(counts, fit["slope"] * np.array(counts) + fit["intercept"], "k--", linewidth=1.3, label="全范围线性拟合")
    axes[0, 1].set(title="B  末端0.002→0.001：8条独立数据轨迹", xlabel="训练样本数 n", ylabel="收缩系数 c")
    axes[0, 1].legend(fontsize=8, ncol=2)
    centers = [-math.log2(math.sqrt(h * l)) for h, l in audit_core.WINDOWS]
    for n in (32, 64, 128, 192, 224, 256):
        j = counts.index(n)
        values = np.array([matrices[w][:, j] for w in audit_core.WINDOWS])
        axes[1, 0].plot(centers, values.mean(1), "o-", markersize=4, label=f"n={n}")
    axes[1, 0].set(title="C  平均局部斜率随loss收紧变化", xlabel="窗口中心 -log2(BCE)", ylabel="收缩系数 c")
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].set(title="D  除以n后：并非全范围严格正比", xlabel="训练样本数 n", ylabel="c/n")
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("8bit随机记忆：8份随机数据×10档样本量，全部到BCE=0.001\n先逐数据集计算系数再平均；副本诊断不是全局混合认证", fontsize=14)
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)
    relative = np.array([[lookup[(d, n, .002, .001)]["replica_sd"] / lookup[(d, n, .002, .001)]["median"] * 100
                          for n in counts] for d in range(8)])
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    im = ax.imshow(relative, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(counts)), counts)
    ax.set_yticks(range(8), [f"数据{d + 1}" for d in range(8)])
    for d in range(8):
        for j in range(len(counts)):
            ax.text(j, d, f"{relative[d, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if relative[d, j] > relative.max() * .6 else "black")
    ax.set(title="末端收缩系数的副本SD/中位数 (%)，不是严格置信区间", xlabel="训练样本数 n")
    fig.colorbar(im, ax=ax, label="副本相对SD (%)")
    fig.savefig(OUT / "replica_uncertainty.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit_core.PACKAGE = PACKAGE
    summary, protocol, jobs, audit, environments, prior, _ = audit_core.inspect_package()
    assert len(jobs) == 80 and all(j["status"] == "completed" for j in jobs)
    assert all(j["endpoint"]["bce_limit"] == .001 for j in jobs)
    counts = sorted(set(j["n"] for j in jobs))
    assert len(counts) == 10 and len({j["dataset_id"] for j in jobs}) == 8
    lookup = slope_lookup(jobs)
    groups, fits, individual, matrices = window_statistics(jobs, lookup, counts)
    stability = tail_stability(jobs, lookup)
    hardware = compare_hardware(jobs)
    audit["unique_data_check"] = data_audit(protocol)
    audit["all_endpoints_report_every_particle_correct"] = all(all(v == 1 for v in j["endpoint"]["training_all_correct_fraction_by_replica"]) for j in jobs)
    audit["producer_mean_max_difference"] = max(abs(g["c_mean"] - next(r["c_mean"] for r in summary["groups"]
        if r["n"] == g["n"] and r["epsilon_high"] == g["high"] and r["epsilon_low"] == g["low"])) for g in groups)
    result = dict(package=str(PACKAGE), package_sha256=hashlib.sha256(PACKAGE.read_bytes()).hexdigest(),
        protocol=protocol, gpu=[e["gpu"] for e in environments], jobs=jobs, audit=audit,
        sampling_seconds=sum(j["seconds"] for j in jobs), groups=groups, mean_curve_fits=fits,
        individual_dataset_fits=individual, tail_stability=stability, hardware_comparison=hardware,
        prior_calibration=prior, caveat="所有80项保留；原警告未剔除。副本共享自适应阈值/步长，不将SD当严格CI。")
    write_json(OUT / "analysis.json", result)
    write_csv(OUT / "coefficient_statistics.csv", groups)
    write_csv(OUT / "dataset_fits.csv", individual)
    write_csv(OUT / "tail_stability.csv", stability["rows"])
    write_csv(OUT / "hardware_comparison.csv", hardware["rows"])
    plot(groups, matrices, jobs, lookup, counts)
    deep = next(r for r in fits if (r["high"], r["low"]) == (.002, .001))
    deep_individual = [r for r in individual if (r["high"], r["low"]) == (.002, .001)]
    lines = ["# 8bit随机记忆完整矩阵", "", "## 覆盖", "",
        f"80/80项全部达到BCE=0.001；每档8份独立随机数据，同编号各n嵌套。GPU={result['gpu']}。",
        f"累计采样循环{result['sampling_seconds'] / 3600:.3f}小时；不包括全部启动和打包开销。", "",
        "## 末端系数", "", "均值和SD均在8份数据之间计算，不先平均体积。",
        "", "|n|c均值|数据间SD|数据中位数|c/n|有采样报警的数据数|", "|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        if (r["high"], r["low"]) == (.002, .001):
            lines.append(f"|{r['n']}|{r['c_mean']:.2f}|{r['c_dataset_sd']:.2f}|{r['c_dataset_median']:.2f}|"
                         f"{r['c_per_n']:.3f}|{r['sampler_warning_datasets']}/8|")
    lines += ["", "## 线性趋势与限制", "",
        f"全10档均值拟合：c={deep['slope']:.5f}n+{deep['intercept']:.3f}，R²={deep['r_squared']:.6f}。",
        f"8条独立数据轨迹各自拟合的R²范围：{min(r['r_squared'] for r in deep_individual):.5f}至{max(r['r_squared'] for r in deep_individual):.5f}。",
        f"中位数曲线的R²={deep['median_curve_fit']['r_squared']:.6f}，只作稳健性参考，不替代均值主结果。",
        "", "最大两档增大的波动与副本分歧应保留。以下n≤192是事后范围敏感性检查，不当作预注册主结论：",
        f"该范围R²={deep['n_le192_sensitivity_fit']['r_squared']:.6f}。",
        "整轨迹bootstrap只重抽数据编号，保留n之间的相关性；区间不覆盖SMC系统性偏差。", "",
        "## 末端变化", "",
        f"80条曲线末两窗口相对下降的中位数={stability['last_drop_median']:.2%}，"
        f"10%至90%分位={stability['last_drop_10_90_quantiles']}。",
        f"最后两个窗口相对变化不超过5%：{stability['last_relative_change_within_5pct']}/80；"
        f"不超过10%：{stability['last_relative_change_within_10pct']}/80。",
        f"最后窗口系数反而增加的曲线数：{len(stability['last_window_increases'])}。",
        "", "## 核验", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", "",
        "未重新采样，不把完成节点或全部分类正确解释为混合已经充分。",
        "微观原因、容量影响与采样瓶颈尚未拆开；本轮不新增失败seed追查或自动recheck。"]
    (OUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit_core.clean(dict(audit=audit, deep_mean_fit=deep,
        deep_individual_fits=deep_individual,
        deep_groups=[r for r in groups if (r["high"], r["low"]) == (.002, .001)],
        stability={k: v for k, v in stability.items() if k != "rows"}, hardware=hardware,
        sampling_hours=result["sampling_seconds"] / 3600)), ensure_ascii=False, indent=2))
    print(f"分析完成：{OUT}")


if __name__ == "__main__":
    main()
