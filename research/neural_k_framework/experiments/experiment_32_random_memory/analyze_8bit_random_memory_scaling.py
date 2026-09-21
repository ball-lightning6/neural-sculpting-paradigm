"""只读分析8bit随机记忆SMC结果包；不执行包内源码、不启动训练或采样。"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(r"E:\Downloads\results_8bit_random_memory_scaling_w16_v1_package.zip")
OUTPUT = ROOT / "tmp/8bit_random_memory_scaling_w16_analysis_20260911"
PILOT = ROOT / "tmp/8bit_random_memory_sampling_pilot_20260911"
WINDOWS = ((.2, .1), (.1, .05), (.05, .025), (.025, .01), (.01, .005), (.005, .002), (.002, .001))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def read_json(archive, name):
    return json.loads(archive.read(name).decode("utf-8"))


def read_npz(archive, name):
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def numpy_logits(theta, x, width):
    theta = np.asarray(theta, dtype=np.float64).reshape(-1, theta.shape[-1])
    x = np.asarray(x, dtype=np.float64)
    first = width * 8
    h1 = np.tanh(x[None] @ (theta[:, :first].reshape(-1, width, 8) / math.sqrt(8)).transpose(0, 2, 1)
                 + theta[:, None, first:first + width] / math.sqrt(8))
    start, end = first + width, first + width + width * width
    h2 = np.tanh(h1 @ (theta[:, start:end].reshape(-1, width, width) / math.sqrt(width)).transpose(0, 2, 1)
                 + theta[:, None, end:end + width] / math.sqrt(width))
    return ((h2 * theta[:, None, end + width:end + 2 * width]).sum(-1)
            + theta[:, end + 2 * width, None]) / math.sqrt(width)


def get_slopes(points):
    indexed = {round(row["bce_limit"], 12): row for row in points}
    slopes = []
    for high, low in WINDOWS + ((.01, .001),):
        if round(high, 12) not in indexed or round(low, 12) not in indexed:
            continue
        a, b = indexed[round(high, 12)], indexed[round(low, 12)]
        values = (np.array(b["K_bits_by_replica"]) - np.array(a["K_bits_by_replica"])) / math.log2(high / low)
        slopes.append(dict(high=high, low=low, by_replica=values.tolist(), median=float(np.median(values)),
                           replica_sd=float(values.std(ddof=1))))
    return slopes


def fit_line(n, c):
    n, c = np.asarray(n, dtype=float), np.asarray(c, dtype=float)
    slope, intercept = np.polyfit(n, c, 1)
    predicted = slope * n + intercept
    origin = float(n @ c / (n @ n))
    total = np.sum((c - c.mean()) ** 2)
    return dict(n=n.tolist(), slope=float(slope), intercept=float(intercept),
                r_squared=float(1 - np.sum((c - predicted) ** 2) / total),
                rmse=float(np.mean((c - predicted) ** 2) ** .5),
                max_relative_residual=float(np.max(np.abs(predicted - c) / c)),
                through_origin_slope=origin, through_origin_rmse=float(np.mean((c - origin * n) ** 2) ** .5))


def inspect_package():
    audit = dict(snapshot_files=0, parameter_vectors=0, logit_max_abs_error=0.,
                 bce_max_abs_error=0., max_constraint_excess=0., sign_disagreements=0,
                 constraint_violations_beyond_numerical_tolerance=0, endpoint_snapshot_training_errors=0,
                 negative_K_increments=0, negative_level_K_increments=0,
                 saved_coefficient_max_error=0.)
    jobs, environments = [], []
    with zipfile.ZipFile(PACKAGE) as archive:
        summary = read_json(archive, "summary.json")
        protocol = read_json(archive, "protocol.json")["signature"]
        width = protocol["WIDTH"]
        for name in archive.namelist():
            if name.startswith("invocations/") and name.endswith(".json"):
                environments.append(read_json(archive, name))
        for name in sorted(archive.namelist()):
            if not (name.startswith("data_") and "/n" in name and name.endswith("/summary.json")):
                continue
            directory = name.rsplit("/", 1)[0]
            job = read_json(archive, name)
            points = read_json(archive, directory + "/thresholds.json")
            levels = read_json(archive, directory + "/levels.json")
            data = read_npz(archive, directory + "/training_data.npz")
            family_name = name.split("/")[0] + "/full_random_data.npz"
            family = read_npz(archive, family_name)
            n = job["n"]
            assert np.array_equal(data["input_ids"], family["order"][:n])
            assert np.array_equal(data["inputs"], family["inputs"][family["order"][:n]])
            assert np.array_equal(data["labels"], family["labels"][family["order"][:n]])
            assert job["signature"]["science"] == protocol
            assert points[-1]["bce_limit"] == job["reached_bce"]
            for rows, key in ((points, "negative_K_increments"), (levels, "negative_level_K_increments")):
                k = np.array([r["K_bits_by_replica"] for r in rows])
                audit[key] += int(np.sum(np.diff(k, axis=0) < -1e-10))
            slopes = get_slopes(points)
            for row in slopes:
                old = next(v for v in job["coefficients"] if v["epsilon_high"] == row["high"] and v["epsilon_low"] == row["low"])
                audit["saved_coefficient_max_error"] = max(audit["saved_coefficient_max_error"], abs(old["c_dataset_median"] - row["median"]))
            for point in points:
                epsilon = point["bce_limit"]
                snapshot = read_npz(archive, directory + f"/snapshot_{epsilon:.6g}.npz")
                z = numpy_logits(snapshot["theta"], data["inputs"], width)
                stored = snapshot["logits"].astype(np.float64).reshape(z.shape)
                bce = np.logaddexp(0, -(2 * data["labels"].astype(float) - 1)[None] * z).mean(-1)
                recorded_bce = np.logaddexp(0, -(2 * data["labels"].astype(float) - 1)[None] * stored).mean(-1)
                audit["snapshot_files"] += 1
                audit["parameter_vectors"] += len(z)
                audit["logit_max_abs_error"] = max(audit["logit_max_abs_error"], float(np.abs(z - stored).max()))
                audit["bce_max_abs_error"] = max(audit["bce_max_abs_error"], float(np.abs(bce - recorded_bce).max()))
                audit["max_constraint_excess"] = max(audit["max_constraint_excess"], float(np.max(bce - epsilon)))
                audit["sign_disagreements"] += int(np.sum((z >= 0) != (stored >= 0)))
                audit["constraint_violations_beyond_numerical_tolerance"] += int(np.sum(bce > epsilon + max(5e-8, epsilon * 3e-6)))
                if epsilon == summary["stop_bce"]:
                    audit["endpoint_snapshot_training_errors"] += int(np.sum((z >= 0) != data["labels"].astype(bool)[None]))
            jobs.append(dict(n=n, dataset_id=job["dataset_id"], seconds=job["sampling_seconds"],
                             levels=job["levels"], status=job["status"], positive_labels=int(data["labels"].sum()),
                             endpoint=job["endpoint"], points=points, slopes=slopes))
        calibration = read_json(archive, "prior_calibration.json")
        prior_rows = []
        for job in jobs:
            if job["dataset_id"] != calibration["dataset_id"]:
                continue
            for j, epsilon in enumerate(calibration["thresholds"]):
                point = next(p for p in job["points"] if p["bce_limit"] == epsilon)
                p = calibration["probability"][calibration["counts"].index(job["n"])][j]
                estimated = 2 ** (-point["K_median_bits"])
                prior_rows.append(dict(n=job["n"], epsilon=epsilon, prior_probability=p,
                                       smc_probability=estimated, relative_error=(estimated - p) / p))
        first_family = read_npz(archive, "data_000/full_random_data.npz")
    return summary, protocol, jobs, audit, environments, prior_rows, first_family


def plot(jobs, groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "SimSun"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    first = sorted([j for j in jobs if j["dataset_id"] == 0], key=lambda j: j["n"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, job in enumerate(first):
        points = [p for p in job["points"] if p["bce_limit"] <= .2]
        axes[0, 0].plot([-math.log2(p["bce_limit"]) for p in points], [p["K_median_bits"] for p in points],
                        color=colors(index), marker=".", markersize=4, label=f"n={job['n']}")
    axes[0, 0].set(title="A  负对数体积曲线", xlabel="loss深度 -log2(BCE上限)", ylabel="K = -log2 V (bit)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    chosen = ((.2, .1), (.1, .05), (.01, .005), (.002, .001))
    for index, window in enumerate(chosen):
        selected = sorted([r for r in groups if (r["high"], r["low"]) == window], key=lambda r: r["n"])
        n = np.array([r["n"] for r in selected])
        c = np.array([r["mean"] for r in selected])
        errors = [r["dataset_sd"] if r["dataset_sd"] is not None else r["median_replica_sd"] for r in selected]
        axes[0, 1].errorbar(n, c, yerr=errors, color=colors(index), fmt="o-", capsize=3, markersize=4,
                            label=f"BCE {window[0]:g} -> {window[1]:g}")
        axes[1, 0].plot(n, c / n, color=colors(index), marker="o", markersize=4,
                        label=f"BCE {window[0]:g} -> {window[1]:g}")
    axes[0, 1].set(title="B  收缩系数随样本数增加", xlabel="训练样本数 n", ylabel="c：loss减半的K增量")
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].set(title="C  近似线性不等于严格正比", xlabel="训练样本数 n", ylabel="每样本收缩系数 c/n")
    axes[1, 0].legend(fontsize=9)
    for index, job in enumerate(first):
        if job["n"] not in (32, 64, 128, 192, 256):
            continue
        rows = [r for r in job["slopes"] if (r["high"], r["low"]) in WINDOWS]
        axes[1, 1].plot([-math.log2(math.sqrt(r["high"] * r["low"])) for r in rows],
                        [r["median"] for r in rows], color=colors(index), marker="o", markersize=4, label=f"n={job['n']}")
    axes[1, 1].set(title="D  低loss末端变化减小，但仍在变平", xlabel="窗口中心的loss深度", ylabel="局部收缩系数 c")
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    counts = len(set(j["dataset_id"] for j in jobs))
    fig.suptitle(f"8bit随机记忆：宽度16，{counts}份随机标签表的阶段结果\n"
                 "A/D展示第1份数据；单份数据误差线为SMC副本SD，不是独立数据集误差", fontsize=14)
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary, protocol, jobs, audit, environments, prior_rows, family = inspect_package()
    groups = []
    for n in sorted(set(j["n"] for j in jobs)):
        for high, low in WINDOWS + ((.01, .001),):
            rows = [r for job in jobs if job["n"] == n for r in job["slopes"] if r["high"] == high and r["low"] == low]
            if rows:
                values = np.array([r["median"] for r in rows])
                groups.append(dict(n=n, high=high, low=low, data_count=len(values), mean=float(values.mean()),
                                   dataset_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                                   median_replica_sd=float(np.median([r["replica_sd"] for r in rows])),
                                   mean_per_n=float(values.mean() / n)))
    fits, flattening = [], []
    for dataset_id in sorted(set(j["dataset_id"] for j in jobs)):
        for high, low in WINDOWS + ((.01, .001),):
            data = [(job["n"], r["median"]) for job in jobs if job["dataset_id"] == dataset_id
                    for r in job["slopes"] if r["high"] == high and r["low"] == low]
            for n_min in (32, 64, 128):
                selected = [(n, c) for n, c in data if n >= n_min]
                if len(selected) >= 3:
                    fits.append(dict(dataset_id=dataset_id, high=high, low=low, n_min=n_min,
                                     **fit_line(*zip(*selected))))
    for job in jobs:
        rows = {(r["high"], r["low"]): r for r in job["slopes"]}
        if all(pair in rows for pair in ((.2, .1), (.1, .05), (.005, .002), (.002, .001))):
            flattening.append(dict(n=job["n"], dataset_id=job["dataset_id"],
                early_slope_drop_fraction=1 - rows[(.1, .05)]["median"] / rows[(.2, .1)]["median"],
                last_slope_drop_fraction=1 - rows[(.002, .001)]["median"] / rows[(.005, .002)]["median"]))
    pilot_comparison = []
    if (PILOT / "datasets.npz").exists():
        with np.load(PILOT / "datasets.npz", allow_pickle=False) as data:
            same = all(np.array_equal(data[k], family[k]) for k in ("inputs", "labels", "order"))
        if same:
            for n in (64, 128, 192, 256):
                old = json.loads((PILOT / f"n{n:03d}" / "thresholds.json").read_text(encoding="utf-8"))
                a = next(r for r in get_slopes(old) if r["high"] == .2 and r["low"] == .1)
                job = next(j for j in jobs if j["dataset_id"] == 0 and j["n"] == n)
                b = next(r for r in job["slopes"] if r["high"] == .2 and r["low"] == .1)
                pilot_comparison.append(dict(n=n, old=a["median"], new=b["median"],
                    change_fraction=b["median"] / a["median"] - 1,
                    note="同数据；粒子/更新次数、BCE数值实现、设备环境均有变化，不当作同精度独立重复。"))
    output = dict(package=str(PACKAGE), package_sha256=hashlib.sha256(PACKAGE.read_bytes()).hexdigest(),
        protocol=protocol, gpu=[e["gpu"] for e in environments], package_coverage=summary["coverage"],
        planned_jobs=summary["planned_jobs"], completed_jobs=summary["completed_jobs"],
        dataset_ids=sorted(set(j["dataset_id"] for j in jobs)), seconds=sum(j["seconds"] for j in jobs),
        audit=audit, groups=groups, exploratory_fits=fits, flattening=flattening,
        pilot_comparison=pilot_comparison, prior_calibration_comparison=prior_rows, jobs=jobs)
    (OUTPUT / "analysis.json").write_text(json.dumps(clean(output), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot(jobs, groups, OUTPUT)
    first = sorted([j for j in jobs if j["dataset_id"] == 0], key=lambda j: j["n"])
    lines = ["# 8bit随机记忆：第一轮阶段包分析", "", "## 结论", "",
             f"当前包已完成{summary['completed_jobs']}/{summary['planned_jobs']}项，随机数据编号"
             f"{output['dataset_ids']}。不能误读成每档8份随机数据的平均。",
             "当前正面信号：固定loss窗口的收缩系数随n近似线性增加；末端系数继续下降，但变化幅度减小。",
             "近似线性不是严格正比，末端近似稳定不是已经证明极限常数。", "",
             "## 第一份数据", "",
             "|n|到0.001分钟|0.2->0.1的c|0.01->0.005的c|0.002->0.001的c|末端c/n|",
             "|---:|---:|---:|---:|---:|---:|"]
    for job in first:
        c = {(r["high"], r["low"]): r["median"] for r in job["slopes"]}
        lines.append(f"|{job['n']}|{job['seconds'] / 60:.2f}|{c[(.2, .1)]:.2f}|"
                     f"{c[(.01, .005)]:.2f}|{c[(.002, .001)]:.2f}|{c[(.002, .001)] / job['n']:.3f}|")
    lines += ["", "## 描述性线性拟合", "", "以下R²仅描述本条嵌套数据轨迹，不是独立数据重复的显著性检验。",
              "", "|窗口|n下限|c=a*n+b的a|b|R²|强制过原点RMSE/含截距RMSE|", "|---|---:|---:|---:|---:|---:|"]
    for row in fits:
        if row["dataset_id"] == 0 and (row["high"], row["low"]) in ((.2, .1), (.002, .001), (.01, .001)):
            lines.append(f"|{row['high']:g}->{row['low']:g}|{row['n_min']}|{row['slope']:.4f}|"
                         f"{row['intercept']:.2f}|{row['r_squared']:.5f}|{row['through_origin_rmse']:.2f}/{row['rmse']:.2f}|")
    lines += ["", "## 独立核验", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", "",
              "独立NumPy只复核抽存权重的前向和约束，不重新估计稀有事件体积。", "",
              "## 与此前同数据试采比较", "", "|n|旧0.2->0.1系数|新系数|相对变化|", "|---:|---:|---:|---:|"]
    for row in pilot_comparison:
        lines.append(f"|{row['n']}|{row['old']:.2f}|{row['new']:.2f}|{row['change_fraction']:+.1%}|")
    lines += ["", "高n档绝对系数对采样配置仍有可见敏感性。没有内部报警不等于已证明无偏/混合充分。",
              "", "## 范围", "", "- 包中环境记录为" + ", ".join(output["gpu"]) + "，不是按口头计划推定GPU型号。",
              "- n=256的抽存末端权重实现了这张随机真值表；这排除了本张表不可表示的绝对容量障碍，不排除容量压力。",
              "- 433参数相对256样本只是轻度过参数化，不把本结果外推成与宽度无关的定律。",
              "- 还需后续独立数据重复判断典型性，本分析未启动任何新训练或采样。"]
    (OUTPUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(clean(dict(audit=audit, seconds=output["seconds"], gpu=output["gpu"],
          fits=[r for r in fits if r["n_min"] == 32], flattening=flattening, pilot=pilot_comparison,
          final=[dict(n=j["n"], seconds=j["seconds"], endpoint=j["endpoint"],
                      c=next(r for r in j["slopes"] if r["high"] == .002 and r["low"] == .001)) for j in first])),
          ensure_ascii=False, indent=2))
    print(f"分析产物：{OUTPUT}")


if __name__ == "__main__":
    main()
