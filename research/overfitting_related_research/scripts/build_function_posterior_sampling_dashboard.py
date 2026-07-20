import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_function_posterior_sampling"
EXPERIMENT_SCRIPT_DIR = ROOT / "scripts" / "function_posterior_sampling_experiments"


SOURCE_ZIPS = {}

ZIP_CANDIDATES = {
    "tiny_n": ("rule30_layer1_tiny_n_sweep.zip",),
    "untrained": ("untrained_mlp_prior_probe_package.zip",),
    "active_soft": ("active_soft_uncertainty_3runs_package.zip",),
    "active_time": ("active_time_committee_sampling_package.zip",),
    "single_seed_time": ("single_seed_time_sampling_batch.zip",),
    "grokking_time": (
        "grokking_agreement_time_axis_package (2).zip",
        "grokking_agreement_time_axis_package.zip",
    ),
}


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the Function-posterior sampling addendum page."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the raw zip packages. If omitted, the script "
            "tries ./source_packages_function_posterior_sampling. If raw zips are not found, it rebuilds "
            "the HTML from checked-in lightweight data."
        ),
    )
    return parser.parse_args()


def resolve_sources(source_dir=None):
    search_dirs = []
    if source_dir is not None:
        search_dirs.append(source_dir)
    search_dirs.append(ROOT / "source_packages_function_posterior_sampling")

    found = {}
    missing = {}
    for key, candidates in ZIP_CANDIDATES.items():
        for directory in search_dirs:
            for filename in candidates:
                path = directory / filename
                if path.exists():
                    found[key] = path
                    break
            if key in found:
                break
        if key not in found:
            missing[key] = list(candidates)
    return found, missing


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_jsonl_from_zip(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(name).decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_csv_from_zip(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(name).decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fnum(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=None):
    value = fnum(value, None)
    if value is None:
        return default
    return int(value)


def r6(value):
    if value is None:
        return None
    return round(float(value), 6)


def accuracy_baseline(acc):
    if acc is None:
        return None
    return acc * acc + (1.0 - acc) * (1.0 - acc)


def add_derived(row):
    acc = row.get("mean_probe_bit_accuracy")
    agreement = row.get("direct_pairwise_agreement")
    base = accuracy_baseline(acc)
    row["accuracy_baseline_agreement"] = r6(base)
    row["excess_agreement"] = r6(agreement - base) if agreement is not None and base is not None else None
    n = row.get("train_count")
    row["log_train_count_plus_1"] = r6(math.log10(n + 1)) if n is not None else None
    return row


def load_tiny_n_curve():
    rows = []

    untrained_zip = require(SOURCE_ZIPS["untrained"])
    untrained = read_jsonl_from_zip(untrained_zip, "summary.jsonl")[0]
    rows.append(add_derived({
        "series": "rule30_layer1_plus_prior",
        "task_name": "untrained_mlp_prior",
        "train_count": 0,
        "mean_probe_bit_accuracy": None,
        "direct_pairwise_agreement": r6(untrained.get("mean_pairwise_prediction_bit_agreement")),
        "mean_prediction_entropy_bits": r6(untrained.get("mean_prediction_entropy_bits")),
        "unanimously_wrong_bit_fraction": None,
        "source": "untrained_mlp_prior_probe_package.zip",
        "note": "未训练 MLP；无真实标签，accuracy 留空",
    }))

    tiny_zip = require(SOURCE_ZIPS["tiny_n"])
    with zipfile.ZipFile(tiny_zip) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith("/summary.jsonl"))
        for name in names:
            match = re.search(r"_n(\d+)_split", name)
            if not match:
                continue
            summary = json.loads(zf.read(name).decode("utf-8-sig").splitlines()[0])
            rows.append(add_derived({
                "series": "rule30_layer1_plus_prior",
                "task_name": "rule30_layer1",
                "train_count": int(match.group(1)),
                "mean_probe_bit_accuracy": r6(summary.get("mean_probe_bit_accuracy")),
                "mean_probe_exact_accuracy": r6(summary.get("mean_probe_exact_accuracy")),
                "direct_pairwise_agreement": r6(summary.get("mean_pairwise_prediction_bit_agreement")),
                "mean_prediction_entropy_bits": r6(summary.get("mean_prediction_entropy_bits")),
                "unanimously_wrong_bit_fraction": r6(summary.get("unanimously_wrong_bit_fraction")),
                "source": "rule30_layer1_tiny_n_sweep.zip",
                "note": "tiny-n sweep",
            }))

    old_csv = ROOT / "results" / "probe_consistency_dashboard_aggregated.csv"
    if old_csv.exists():
        with old_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("task_name") != "rule30_layer1":
                    continue
                n = inum(row.get("train_count"))
                if n is None or n < 100:
                    continue
                rows.append(add_derived({
                    "series": "rule30_layer1_plus_prior",
                    "task_name": "rule30_layer1",
                    "train_count": n,
                    "mean_probe_bit_accuracy": r6(fnum(row.get("mean_probe_bit_accuracy"))),
                    "mean_probe_exact_accuracy": r6(fnum(row.get("mean_probe_exact_accuracy"))),
                    "direct_pairwise_agreement": r6(fnum(row.get("direct_pairwise_agreement"))),
                    "mean_prediction_entropy_bits": r6(fnum(row.get("mean_prediction_entropy_bits"))),
                    "unanimously_wrong_bit_fraction": None,
                    "unanimously_same_prediction_bit_fraction": r6(fnum(row.get("unanimously_same_prediction_bit_fraction"))),
                    "source": "probe_consistency_dashboard_aggregated.csv",
                    "note": "旧 plateau sweep 聚合点",
                }))

    rows.sort(key=lambda row: row["train_count"])
    return rows


def last_rows_by_train_count(rows, train_key="train_count", eval_key="eval_id"):
    result = {}
    for row in rows:
        train_count = inum(row.get(train_key))
        if train_count is None:
            continue
        eval_id = inum(row.get(eval_key), 0)
        old = result.get(train_count)
        if old is None or eval_id >= inum(old.get(eval_key), 0):
            result[train_count] = row
    return [result[key] for key in sorted(result)]


def threshold_row(rows, key, threshold):
    for row in rows:
        value = fnum(row.get(key))
        if value is not None and value >= threshold:
            return inum(row.get("train_count"))
    return None


def load_active_soft():
    zip_path = require(SOURCE_ZIPS["active_soft"])
    rows = []
    summary = []
    for strategy in ("uncertain", "random", "certain"):
        name = next(
            n for n in zipfile.ZipFile(zip_path).namelist()
            if n.endswith(f"branches/{strategy}/training_curve.csv")
        )
        raw = read_csv_from_zip(zip_path, name)
        branch_rows = []
        for row in last_rows_by_train_count(raw):
            item = {
                "strategy": strategy,
                "train_count": inum(row.get("train_count")),
                "probe_bit_accuracy": r6(fnum(row.get("probe_mean_bit_accuracy"))),
                "probe_exact_accuracy": r6(fnum(row.get("probe_mean_exact_accuracy"))),
                "probe_majority_exact_accuracy": r6(fnum(row.get("probe_majority_exact_accuracy"))),
                "probe_pairwise_agreement": r6(fnum(row.get("probe_pairwise_agreement"))),
                "probe_prediction_entropy_bits": r6(fnum(row.get("probe_prediction_entropy_bits"))),
                "val_bit_accuracy": r6(fnum(row.get("val_mean_bit_accuracy"))),
                "val_pairwise_agreement": r6(fnum(row.get("val_pairwise_agreement"))),
            }
            rows.append(item)
            branch_rows.append(item)
        summary.append({
            "experiment": "active_soft_multiseed",
            "strategy": strategy,
            "final_train_count": branch_rows[-1]["train_count"],
            "final_probe_exact_accuracy": branch_rows[-1]["probe_exact_accuracy"],
            "first_exact_ge_0_8": threshold_row(branch_rows, "probe_exact_accuracy", 0.8),
            "first_exact_ge_0_9": threshold_row(branch_rows, "probe_exact_accuracy", 0.9),
            "first_exact_ge_0_95": threshold_row(branch_rows, "probe_exact_accuracy", 0.95),
            "first_exact_ge_0_99": threshold_row(branch_rows, "probe_exact_accuracy", 0.99),
            "first_exact_ge_0_999": threshold_row(branch_rows, "probe_exact_accuracy", 0.999),
        })
    return rows, summary


def load_active_time():
    zip_path = require(SOURCE_ZIPS["active_time"])
    rows = []
    for strategy in ("uncertain", "random", "certain"):
        raw = read_csv_from_zip(zip_path, f"branches/{strategy}/training_curve.csv")
        for row in raw:
            rows.append({
                "strategy": strategy,
                "train_count": inum(row.get("train_count")),
                "round": inum(row.get("round")),
                "probe_bit_accuracy": r6(fnum(row.get("probe_bit_accuracy"))),
                "probe_exact_accuracy": r6(fnum(row.get("probe_exact_accuracy"))),
                "majority_probe_exact_accuracy": r6(fnum(row.get("majority_probe_exact_accuracy"))),
                "probe_pairwise_agreement": r6(fnum(row.get("probe_agreement"))),
                "probe_prediction_entropy_bits": r6(fnum(row.get("probe_entropy"))),
                "sample_first_step": inum(row.get("sample_first_step")),
                "sample_last_step": inum(row.get("sample_last_step")),
            })
    summary = read_jsonl_from_zip(zip_path, "summary.jsonl")
    for row in summary:
        row["experiment"] = "active_time_committee"
    return rows, summary


def load_single_seed_time_summary():
    zip_path = require(SOURCE_ZIPS["single_seed_time"])
    name = (
        "research/overfitting_related_research/results_single_seed_time_sampling/"
        "single_seed_time_sampling_batch_summary.csv"
    )
    raw = read_csv_from_zip(zip_path, name)
    rows = []
    for row in raw:
        rows.append({
            "task_name": row.get("task_name"),
            "train_count": inum(row.get("train_count")),
            "sample_count": inum(row.get("sample_count")),
            "train_fit_step": inum(row.get("train_fit_step")),
            "plateau_step": inum(row.get("plateau_step")),
            "mean_model_bit_accuracy": r6(fnum(row.get("mean_model_bit_accuracy"))),
            "mean_model_exact_accuracy": r6(fnum(row.get("mean_model_exact_accuracy"))),
            "time_pairwise_agreement": r6(fnum(row.get("time_pairwise_agreement"))),
            "time_prediction_entropy_bits": r6(fnum(row.get("time_prediction_entropy_bits"))),
            "plateau_stop_source": row.get("plateau_stop_source"),
        })

    old_csv = ROOT / "results" / "probe_consistency_dashboard_aggregated.csv"
    old_by_n = {}
    if old_csv.exists():
        with old_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for old in csv.DictReader(f):
                if old.get("task_name") == "rule30_layer1":
                    old_by_n[inum(old.get("train_count"))] = old
    for row in rows:
        old = old_by_n.get(row["train_count"])
        row["multiseed_pairwise_agreement"] = r6(fnum(old.get("direct_pairwise_agreement"))) if old else None
        row["multiseed_mean_bit_accuracy"] = r6(fnum(old.get("mean_probe_bit_accuracy"))) if old else None
        row["agreement_gap_time_minus_multiseed"] = (
            r6(row["time_pairwise_agreement"] - row["multiseed_pairwise_agreement"])
            if row.get("time_pairwise_agreement") is not None and row.get("multiseed_pairwise_agreement") is not None
            else None
        )
    return sorted(rows, key=lambda row: row["train_count"])


def load_grokking_time_curve(limit=900):
    zip_path = require(SOURCE_ZIPS["grokking_time"])
    raw = read_csv_from_zip(zip_path, "agreement_time_curve.csv")
    rows = []
    if len(raw) > limit:
        stride = max(1, len(raw) // limit)
        selected = raw[::stride]
        if selected[-1] is not raw[-1]:
            selected.append(raw[-1])
    else:
        selected = raw
    for row in selected:
        rows.append({
            "step": inum(row.get("step")),
            "train_bit_accuracy": r6(fnum(row.get("mean_train_bit_accuracy"))),
            "probe_bit_accuracy": r6(fnum(row.get("mean_probe_bit_accuracy"))),
            "probe_exact_accuracy": r6(fnum(row.get("mean_probe_exact_accuracy"))),
            "direct_pairwise_agreement": r6(fnum(row.get("direct_pairwise_agreement"))),
            "prediction_entropy_bits": r6(fnum(row.get("prediction_entropy_bits"))),
            "excess_agreement": r6(fnum(row.get("excess_agreement"))),
            "bit_level_excess_agreement": r6(fnum(row.get("bit_level_excess_agreement"))),
        })
    meta = json.loads(zipfile.ZipFile(zip_path).read("metadata.json").decode("utf-8-sig"))
    return rows, meta


def js_data(obj):
    return json.dumps(obj, ensure_ascii=False)


def build_html(data):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>函数后验采样与主动采样实验补充</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f2e8;
      --paper: #fffdf8;
      --ink: #202938;
      --muted: #647084;
      --line: #ded5c3;
      --soft: #efe7d5;
      --blue: #2563eb;
      --green: #059669;
      --red: #dc2626;
      --orange: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "Noto Sans SC", Arial, sans-serif;
      line-height: 1.68;
    }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 56px; }}
    header, section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      margin: 0 0 18px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 32px; line-height: 1.2; }}
    h2 {{ margin: 0 0 10px; font-size: 23px; }}
    h3 {{ margin: 16px 0 6px; font-size: 17px; }}
    p {{ margin: 0 0 10px; }}
    code {{ background: var(--soft); padding: 2px 5px; border-radius: 4px; }}
    a {{ color: var(--blue); }}
    .lead {{ color: var(--muted); max-width: 980px; }}
    .topbar {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 2px; }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 8px 13px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff8ea;
      color: var(--blue);
      font-weight: 600;
      font: inherit;
      cursor: pointer;
      text-decoration: none;
    }}
    .button:hover {{ border-color: var(--blue); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }}
    .stat {{ border: 1px solid var(--line); background: #fff8ea; border-radius: 8px; padding: 14px; }}
    .toc {{
      margin: 18px 0 12px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff8ea;
    }}
    .toc-title {{ font-weight: 800; margin-bottom: 10px; }}
    .toc-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .toc a {{
      display: block;
      min-height: 70px;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      color: var(--ink);
      text-decoration: none;
    }}
    .toc a:hover {{ border-color: var(--blue); box-shadow: 0 4px 16px rgba(37, 99, 235, 0.10); }}
    .toc-num {{ color: var(--blue); font-weight: 800; font-size: 13px; }}
    .toc-main {{ font-weight: 750; margin-top: 2px; }}
    .toc-sub {{ color: var(--muted); font-size: 13px; margin-top: 2px; line-height: 1.45; }}
    .label {{ color: var(--muted); font-size: 13px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 3px; }}
    .chart {{ height: 390px; width: 100%; margin-top: 12px; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .note {{ color: var(--muted); font-size: 14px; }}
    .callout {{ border-left: 4px solid var(--blue); background: #f8fbff; padding: 12px 14px; margin: 12px 0; }}
    .takeaways {{ margin: 8px 0 12px 20px; padding: 0; }}
    .takeaways li {{ margin: 4px 0; }}
    body.lang-en .zh {{ display: none !important; }}
    body:not(.lang-en) .en {{ display: none !important; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    @media (max-width: 860px) {{
      .grid, .two, .toc-grid {{ grid-template-columns: 1fr; }}
      .topbar {{ display: block; }}
      h1 {{ font-size: 27px; }}
      .chart {{ height: 330px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="topbar">
      <div>
        <h1><span class="zh">函数后验采样与主动采样实验补充</span><span class="en">Function-Posterior Sampling and Active Sampling Addendum</span></h1>
        <p class="lead zh">这页补充几组更聚焦的实验：未训练先验、极小训练集、主动采样、多 seed 与单 seed 时间采样、以及 grokking 时间轴上的 agreement 曲线。核心问题是：同一个训练集在不同 seed 下得到的函数，能否近似看作从某种函数后验分布中采样？如果这个视角成立，跨 seed 的 agreement、entropy，以及用不确定性挑选样本的效果，都应该呈现出可解释的结构。</p>
        <p class="lead en">This page adds several focused experiments: the untrained prior, extremely small training sets, active sampling, multi-seed versus single-seed time sampling, and agreement along the grokking time axis. The central question is whether functions learned from the same training set under different random seeds can be treated as approximate samples from a function-space posterior. If this view is useful, cross-seed agreement, entropy, and uncertainty-based sample selection should show interpretable structure.</p>
      </div>
    </div>
    <div class="actions">
      <a class="button" href="index.html"><span class="zh">返回研究主页</span><span class="en">Back to overview</span></a>
      <a class="button" href="results/probe_consistency_dashboard.html"><span class="zh">打开数据量 Dashboard</span><span class="en">Open data-size dashboard</span></a>
      <button class="button" type="button" id="langToggle">English</button>
    </div>
    <nav class="toc" aria-label="Page contents">
      <div class="toc-title"><span class="zh">页面目录</span><span class="en">Page Contents</span></div>
      <div class="toc-grid">
        <a href="#tiny-n">
          <div class="toc-num">01</div>
          <div class="toc-main"><span class="zh">极小训练集</span><span class="en">Tiny training sets</span></div>
          <div class="toc-sub"><span class="zh">n=0 先验、n=1 跃升与伪规则峰。</span><span class="en">n=0 prior, n=1 jump, pseudo-rule peak.</span></div>
        </a>
        <a href="#single-seed-time">
          <div class="toc-num">02</div>
          <div class="toc-main"><span class="zh">单 seed 时间采样</span><span class="en">Single-seed time sampling</span></div>
          <div class="toc-sub"><span class="zh">时间样本与多 seed 系综的 agreement 对照。</span><span class="en">Agreement from time samples versus multi-seed ensembles.</span></div>
        </a>
        <a href="#active-sampling">
          <div class="toc-num">03</div>
          <div class="toc-main"><span class="zh">主动采样</span><span class="en">Active sampling</span></div>
          <div class="toc-sub"><span class="zh">uncertain / random / certain 的严格顺序。</span><span class="en">Strict ordering of uncertain / random / certain.</span></div>
        </a>
        <a href="#grokking-time">
          <div class="toc-num">04</div>
          <div class="toc-main"><span class="zh">Grokking 时间轴</span><span class="en">Grokking time axis</span></div>
          <div class="toc-sub"><span class="zh">固定数据量时，agreement 随训练时间重组。</span><span class="en">Agreement reorganizes over training time.</span></div>
        </a>
        <a href="#script-guide">
          <div class="toc-num">05</div>
          <div class="toc-main"><span class="zh">脚本说明</span><span class="en">Script guide</span></div>
          <div class="toc-sub"><span class="zh">每个复现实验脚本的用途和输出。</span><span class="en">Purpose and output of each reproduction script.</span></div>
        </a>
        <a href="#reproduce">
          <div class="toc-num">06</div>
          <div class="toc-main"><span class="zh">复现入口</span><span class="en">Reproducibility</span></div>
          <div class="toc-sub"><span class="zh">如何重建本页和轻量数据。</span><span class="en">How to rebuild this page and lightweight data.</span></div>
        </a>
      </div>
    </nav>
    <div class="grid">
      <div class="stat"><div class="label"><span class="zh">未训练先验 pairwise agreement</span><span class="en">Untrained prior pairwise agreement</span></div><div class="value">50.01%</div><div class="note"><span class="zh">1000 个随机初始化 MLP，未训练。</span><span class="en">1000 randomly initialized MLPs, no training.</span></div></div>
      <div class="stat"><div class="label"><span class="zh">n=1 后 pairwise agreement</span><span class="en">Pairwise agreement after n=1</span></div><div class="value">100.00%</div><div class="note"><span class="zh">单样本已足以把 seed 拉到几乎同一个表外延拓。</span><span class="en">A single example is enough to pull seeds into almost the same off-training-set extension.</span></div></div>
      <div class="stat"><div class="label"><span class="zh">主动采样顺序</span><span class="en">Active sampling order</span></div><div class="value">uncertain &gt; random &gt; certain</div><div class="note"><span class="zh">多 seed 与单 seed 时间委员会版本都复现。</span><span class="en">Reproduced with both multi-seed and single-seed time committees.</span></div></div>
    </div>
  </header>

  <section id="tiny-n">
    <h2><span class="zh">1. 从未训练先验到极小训练集</span><span class="en">1. From the untrained prior to extremely small datasets</span></h2>
    <div class="zh">
      <p>原来的数据量 sweep 从 <code>n=100</code> 开始，已经能看到 agreement 先下降再上升的 U 形右半边，但看不到最左端：从未训练网络到第一个训练样本之间到底发生了什么。因此这里只补测 <code>rule30_layer1</code> 的极小数据量左端，把未训练先验 <code>n=0</code>、新跑的 <code>n=1..75</code>、以及原有 plateau sweep 的 <code>n>=100</code> 拼成一条完整曲线。</p>
      <p>读图时，上图的 <code>pairwise agreement</code> 是任取两个 seed，它们在同一个 probe bit 上预测相同的概率；<code>entropy (1-H)</code> 是把预测熵反过来画，越高表示越集中。下图同时画 probe bit accuracy 和 excess agreement。后者是在扣除“大家都正确所以自然一致”的部分后，剩下的额外一致性。</p>
      <div class="callout">最关键的现象是：<code>n=0</code> 时 agreement 约为 0.5，说明未训练函数分布本身没有集中；但 <code>n=1</code> 后 agreement 几乎跳到 1，而 accuracy 仍然约为 0.5。这说明少量训练约束会把不同 seed 显影到同一个共享的低复杂度延拓上。它不是正确的规则，却是高度一致的“伪规则”。</div>
    </div>
    <div class="en">
      <p>The original data-size sweep started at <code>n=100</code>. It already showed the right half of a U-shaped agreement curve, but it missed the far-left edge: what happens between an untrained network and the first training example? This addendum only fills that tiny-data region for <code>rule30_layer1</code>, stitching together the untrained prior <code>n=0</code>, the new <code>n=1..75</code> sweep, and the previous plateau sweep for <code>n>=100</code>.</p>
      <p>In the upper plot, <code>pairwise agreement</code> is the probability that two randomly chosen seeds make the same prediction on the same probe bit. <code>entropy (1-H)</code> is the inverse of the prediction entropy, so higher values mean a more concentrated ensemble. The lower plot shows probe bit accuracy and excess agreement, where excess agreement subtracts the part of agreement that is already explained by accuracy.</p>
      <div class="callout">The key observation is sharp: at <code>n=0</code>, agreement is about 0.5, so the untrained function distribution is not concentrated. After <code>n=1</code>, agreement jumps almost to 1 while accuracy remains about 0.5. A tiny training constraint can therefore develop a shared low-complexity extension across seeds. It is not yet the true rule; it is a highly consistent pseudo-rule.</div>
    </div>
    <div id="tinyAgreement" class="chart"></div>
    <div id="tinyAccuracy" class="chart"></div>
    <p class="note"><span class="zh">横轴为了同时放下 <code>n=0</code>、极小 n 和几千样本，实际使用 <code>log10(n+1)</code>。所以左端 <code>0</code> 是未训练先验 <code>n=0</code>，<code>n=1</code> 位于约 <code>0.30</code>，刻度 <code>2</code> 约等于 <code>n=99</code>，刻度 <code>3</code> 约等于 <code>n=999</code>。</span><span class="en">The x-axis uses <code>log10(n+1)</code> so that <code>n=0</code>, tiny n, and thousands of samples fit in one plot. The left edge <code>0</code> is the untrained prior <code>n=0</code>; <code>n=1</code> is around <code>0.30</code>; tick <code>2</code> is about <code>n=99</code>; tick <code>3</code> is about <code>n=999</code>.</span></p>
  </section>

  <section id="single-seed-time">
    <h2><span class="zh">2. 单 seed 时间采样 vs 多 seed 采样</span><span class="en">2. Single-seed time sampling versus multi-seed sampling</span></h2>
    <div class="zh">
      <p>多 seed 系综很贵：每个条件都要训练很多次。这个实验问一个实用问题：如果一个模型已经进入平台期，沿着同一个 seed 的后续训练轨迹每隔一段步数采样一次，这些时间样本能不能近似替代多 seed 样本？</p>
      <p>这里目前只做了一个单任务检查：<code>rule30_layer1</code>，训练样本数为 <code>100..1500</code>。图中把单 seed 时间采样得到的 agreement 与过去多 seed sweep 的 agreement 对齐比较。大多数数据量下，两者相当接近；<code>n=100</code> 偏差最大，符合这个点仍处在很慢的 grokking / 欠稳定区域的直觉。</p>
      <div class="callout">这里的结论要保持克制：它很自然地让人联想到 MCMC / 单链采样，但暂时只是一种类比。实验没有严格证明单条训练轨迹就是独立后验采样器；它给出的实用证据是，在平台态足够稳定时，单 seed 时间采样可以近似复现多 seed 函数系综的统计。</div>
    </div>
    <div class="en">
      <p>Multi-seed ensembles are expensive because every condition must be trained many times. This experiment asks a practical question: once a model is on a plateau, can samples collected along one seed's later trajectory approximate samples from many independent seeds?</p>
      <p>This is currently a single-task check: <code>rule30_layer1</code>, with training sizes <code>100..1500</code>. The plot compares agreement from single-seed time samples with agreement from the previous multi-seed sweep. They are close for most data sizes. The largest deviation is at <code>n=100</code>, consistent with that condition being less stable and having a longer mixing or slow-grokking timescale.</p>
      <div class="callout">The claim is deliberately modest: the result naturally suggests an MCMC / single-chain sampling analogy, but for now this is only an analogy. It is not a proof that one training trajectory is an independent posterior sampler. It is practical evidence that, when the plateau is stable enough, time samples from one seed can approximately reproduce the function-ensemble statistics of many seeds.</div>
    </div>
    <div id="singleSeedTime" class="chart"></div>
  </section>

  <section id="active-sampling">
    <h2><span class="zh">3. 主动采样：分歧就是信息量的可用信号</span><span class="en">3. Active sampling: disagreement is a usable information signal</span></h2>
    <div class="zh">
      <p>在单 seed 时间采样和多 seed 系综给出相近统计之后，我们进一步测试这些“函数样本”的分歧是否真的有预测价值。如果 seed 系综只是一些互不相关的随机错误，那么它们的分歧不应该稳定地告诉我们哪些新样本最有用。反过来，如果它们近似表示当前训练集约束下的函数后验，那么高分歧样本就应该携带更多信息，加入训练集后更容易推动系统跨过泛化相变。</p>
      <p>我们维护同一个任务的三个数据集分支：每轮训练到当前阶段的稳定态，然后在候选池中选择一批新样本加入。<code>uncertain</code> 选择系综最不确定的样本，<code>certain</code> 选择最确定的样本，<code>random</code> 随机选择。选样过程只看模型预测的分歧或置信度，不使用真实标签；真实标签只在样本被选中后加入训练。</p>
      <p>这个设计和主动学习里的 Query-by-Committee 思路直接相连：如果委员会成员可以看作当前假设分布的样本，那么成员之间分歧大的样本就应该更有信息量。这里的“委员会成员”不是手工构造的不同模型，而是同一训练集下的不同 seed，或者单 seed 平台期的时间样本。</p>
      <div class="callout">结果在两个版本里都保持严格顺序：<code>uncertain &gt; random &gt; certain</code>。这说明跨 seed 的分歧不只是一个事后解释指标，而是能提前预测哪些数据更有信息量。这个结果强烈支持“不同 seed 近似函数后验采样”的操作性解释。</div>
    </div>
    <div class="en">
      <p>After single-seed time sampling and multi-seed ensembles showed similar statistics, we tested whether disagreement between these function samples has predictive value. If the seeds were merely unrelated random mistakes, their disagreement should not reliably identify useful new data. If they approximate the current function posterior under the training constraints, then high-disagreement examples should carry more information and should move the system toward generalization faster.</p>
      <p>We maintain three dataset branches for the same task. In each round, the current ensemble is trained to a stable state, then a batch of new examples is acquired from a candidate pool. <code>uncertain</code> selects the most uncertain examples, <code>certain</code> selects the most certain examples, and <code>random</code> samples randomly. The selection rule only uses model predictions; labels are revealed only after examples are selected.</p>
      <p>This is the same logic as Query-by-Committee in active learning: if committee members are samples from the current hypothesis distribution, then examples on which they disagree should be more informative. In this experiment, the committee members are not manually designed models; they are different random seeds for the same training set, or time samples from one seed on a plateau.</p>
      <div class="callout">Both implementations give the same strict ordering: <code>uncertain &gt; random &gt; certain</code>. Cross-seed disagreement is not merely a retrospective diagnostic; it can predict which data points are informative. This is strong behavioral evidence for treating different seeds as approximate function-posterior samples.</div>
    </div>
    <div class="two">
      <div><h3><span class="zh">多 seed soft/BALD 版本</span><span class="en">Multi-seed soft/BALD version</span></h3><div id="activeSoft" class="chart"></div></div>
      <div><h3><span class="zh">单 seed 时间委员会版本</span><span class="en">Single-seed time-committee version</span></h3><div id="activeTime" class="chart"></div></div>
    </div>
    <table id="activeSummary"></table>
  </section>

  <section id="grokking-time">
    <h2><span class="zh">4. Grokking 时间轴上的 agreement</span><span class="en">4. Agreement along the grokking time axis</span></h2>
    <div class="zh">
      <p>前面的数据量实验改变的是训练样本数；这里固定数据量，只改变训练时间。问题是：grokking 过程中，跨 seed 函数分布是否也会经历类似的重组？</p>
      <p>图里同时画出训练集 accuracy、probe/test bit accuracy、probe/test exact accuracy，以及跨 seed 的 pairwise agreement 和 excess agreement。这个区分很重要：probe/test accuracy 在这次实验里基本是单调上升的；真正出现“先升高、再下降、再上升”的是跨 seed agreement。更干净的 excess agreement 则表现为一个早期峰值，随后逐渐回到接近 0。</p>
      <p>这说明时间轴上的非单调结构不是单模型测试准确率的影子，而是函数系综本身的重组：早期不同 seed 共享某种启发式，随后这个共享启发式被训练约束和记忆压力打散，最后随着 grokking 完成，不同 seed 又收缩到同一个真实规则函数附近。</p>
      <div class="callout">这把数据量轴和时间轴连在一起：agreement 测到的不是单个模型的准确率，而是当前函数后验空间的集中或发散。数据量增加和训练时间推进，都可以推动这个函数分布经历“伪规则共识、分歧、真实规则共识”的阶段。</div>
    </div>
    <div class="en">
      <p>The data-size experiment changes the number of training examples. Here the dataset is fixed and only training time changes. The question is whether the function ensemble also reorganizes during grokking.</p>
      <p>The plot shows training accuracy, probe/test bit accuracy, probe/test exact accuracy, cross-seed pairwise agreement, and excess agreement. This distinction matters: in this run, probe/test accuracy is mostly monotone increasing. The rise-fall-rise pattern appears in cross-seed agreement. The cleaner excess-agreement curve forms an early peak and then returns close to zero.</p>
      <p>This means the time-axis non-monotonicity is not just a shadow of single-model test accuracy. It reflects a reorganization of the function ensemble: early seeds share a heuristic, that shared heuristic is disrupted by training constraints and memorization pressure, and after grokking the seeds contract toward the same true rule function.</p>
      <div class="callout">This links the data axis and the time axis. Agreement is not merely single-model accuracy; it measures how concentrated or dispersed the current function-posterior-like ensemble is. Increasing data and continuing training can both drive the function distribution through pseudo-rule consensus, dispersion, and true-rule consensus.</div>
    </div>
    <div id="grokkingTime" class="chart"></div>
    <p class="note"><span class="zh">图中只显示前 400 step；后续很长一段基本已经进入收敛平台，是当时停止较晚留下的冗余记录。全量曲线仍保存在 <code>results_function_posterior_sampling/grokking_agreement_time_axis_curve.csv</code>。</span><span class="en">The plot only shows the first 400 steps. The later segment is mostly a redundant convergence plateau caused by stopping the run late. The full curve remains in <code>results_function_posterior_sampling/grokking_agreement_time_axis_curve.csv</code>.</span></p>
  </section>

  <section id="script-guide">
    <h2><span class="zh">脚本说明</span><span class="en">Script Guide</span></h2>
    <div class="zh">
      <p>这些脚本都是公开复现实验用的 Python 文件，集中放在 <code>scripts/function_posterior_sampling_experiments/</code>。默认路径使用仓库相对目录；如果换机器运行，通常只需要改脚本顶部 <code>Config</code> 里的数据量、任务、设备和输出目录。</p>
    </div>
    <div class="en">
      <p>These are the public Python scripts for reproducing the experiments, stored under <code>scripts/function_posterior_sampling_experiments/</code>. Paths are repository-relative by default. On a new machine, the usual edits are the task, training size, device, and output directory fields in the top-level <code>Config</code>.</p>
    </div>
    <table>
      <thead><tr><th>脚本 / script</th><th>用途 / purpose</th><th>输出 / output</th></tr></thead>
      <tbody>
        <tr>
          <td><code>train_ca_overfit_ensemble.py</code></td>
          <td><span class="zh">对一个任务和一个训练样本数训练多个 seed，保存它们在固定 probe 集上的预测。</span><span class="en">Train multiple seeds for one task and one training size, then save predictions on a fixed probe set.</span></td>
          <td><code>predictions.jsonl</code>, <code>run_statistics.jsonl</code></td>
        </tr>
        <tr>
          <td><code>analyze_ca_overfit_ensemble.py</code></td>
          <td><span class="zh">读取一个 ensemble 结果目录，计算 pairwise agreement、entropy、共同错误等统计。</span><span class="en">Analyze one ensemble result directory and compute pairwise agreement, entropy, and shared-error statistics.</span></td>
          <td><code>summary.jsonl</code>, <code>pairwise_statistics.jsonl</code></td>
        </tr>
        <tr>
          <td><code>sweep_overfit_ensemble_grid.py</code></td>
          <td><span class="zh">批量调用训练和分析脚本，扫描多个训练样本数或任务；用于 tiny-n 和数据量轴相关实验。</span><span class="en">Batch wrapper around training and analysis for multiple training sizes or tasks; used for tiny-n and data-axis experiments.</span></td>
          <td><span class="zh">多个按任务和 <code>n</code> 命名的结果目录。</span><span class="en">Multiple result directories named by task and <code>n</code>.</span></td>
        </tr>
        <tr>
          <td><code>untrained_mlp_prior_probe.py</code></td>
          <td><span class="zh">不训练 MLP，只在固定 probe 上采样随机初始化函数，用作 <code>n=0</code> 先验 baseline。</span><span class="en">Sample randomly initialized MLP functions on a fixed probe set without training; this is the <code>n=0</code> prior baseline.</span></td>
          <td><code>untrained_mlp_prior_probe_package.zip</code></td>
        </tr>
        <tr>
          <td><code>train_single_seed_time_sampling.py</code></td>
          <td><span class="zh">单 seed 训练到平台后按时间间隔采样，检验时间样本能否近似多 seed 函数系综。</span><span class="en">After one seed reaches a plateau, sample along later training steps to test whether time samples approximate a multi-seed function ensemble.</span></td>
          <td><span class="zh">单 seed 时间采样曲线和汇总表。</span><span class="en">Single-seed time-sampling curves and summaries.</span></td>
        </tr>
        <tr>
          <td><code>active_soft_uncertainty_sampling_manual_ca.py</code></td>
          <td><span class="zh">多 seed 主动采样实验，比较 <code>uncertain / random / certain</code> 三种选样策略。</span><span class="en">Multi-seed active-sampling experiment comparing <code>uncertain / random / certain</code> acquisition strategies.</span></td>
          <td><span class="zh">主动采样曲线、选样日志和策略汇总。</span><span class="en">Active-sampling curves, selection logs, and strategy summaries.</span></td>
        </tr>
        <tr>
          <td><code>active_time_committee_sampling_ca.py</code></td>
          <td><span class="zh">主动采样的单 seed 时间委员会版本，用时间样本代替多 seed 委员会。</span><span class="en">Single-seed time-committee version of active sampling, replacing a multi-seed committee with time samples.</span></td>
          <td><span class="zh">主动采样曲线、选样日志和策略汇总。</span><span class="en">Active-sampling curves, selection logs, and strategy summaries.</span></td>
        </tr>
        <tr>
          <td><code>train_grokking_agreement_time_axis.py</code></td>
          <td><span class="zh">固定训练集，沿训练 step 记录跨 seed 的 accuracy、agreement 和 excess agreement。</span><span class="en">For a fixed training set, record cross-seed accuracy, agreement, and excess agreement over training steps.</span></td>
          <td><span class="zh">grokking 时间轴曲线。</span><span class="en">Grokking time-axis curves.</span></td>
        </tr>
        <tr>
          <td><code>build_function_posterior_sampling_dashboard.py</code></td>
          <td><span class="zh">从轻量 CSV/JSON 或原始 zip 包重建本页面。</span><span class="en">Rebuild this page from checked-in lightweight CSV/JSON files or from raw zip packages.</span></td>
          <td><code>function_posterior_sampling_experiments.html</code></td>
        </tr>
      </tbody>
    </table>
  </section>

  <section id="reproduce">
    <h2><span class="zh">复现入口</span><span class="en">Reproducibility entry points</span></h2>
    <div class="zh">
      <p>复现实验脚本保存在 <code>scripts/function_posterior_sampling_experiments/</code>。未训练先验 baseline 也已放在同一目录：<code>scripts/function_posterior_sampling_experiments/untrained_mlp_prior_probe.py</code>。</p>
      <p>这些训练脚本默认读取 <code>research/overfitting_related_research/datasets/</code> 下的 CA 数据集；新机器上先运行 <code>python research/overfitting_related_research/scripts/generate_ca_rule_dataset.py</code>，会生成默认需要的 rule30 layer1/2/3 数据。</p>
      <p>本页可以直接由仓库里的轻量数据重建：<code>python research/overfitting_related_research/scripts/build_function_posterior_sampling_dashboard.py</code>。如果要从原始 zip 包重建轻量数据，把 zip 放到 <code>research/overfitting_related_research/source_packages_function_posterior_sampling/</code>，或使用 <code>--source-dir</code> 指向下载目录。</p>
      <p class="note">大体积 raw predictions 没有直接放入仓库；仓库内保留的是检查页面所需的轻量 CSV/JSON。</p>
    </div>
    <div class="en">
      <p>Experiment scripts are stored in <code>scripts/function_posterior_sampling_experiments/</code>. The untrained-prior baseline is in the same folder: <code>scripts/function_posterior_sampling_experiments/untrained_mlp_prior_probe.py</code>.</p>
      <p>The training scripts read CA datasets from <code>research/overfitting_related_research/datasets/</code> by default. On a fresh machine, first run <code>python research/overfitting_related_research/scripts/generate_ca_rule_dataset.py</code>; it generates the default rule30 layer1/2/3 datasets used by these experiments.</p>
      <p>This page can be rebuilt directly from the checked-in lightweight data with <code>python research/overfitting_related_research/scripts/build_function_posterior_sampling_dashboard.py</code>. To rebuild the lightweight data from raw zip packages, place the zips under <code>research/overfitting_related_research/source_packages_function_posterior_sampling/</code>, or pass a download directory with <code>--source-dir</code>.</p>
      <p class="note">Large raw prediction files are not checked in. The repository keeps the lightweight CSV/JSON files needed to inspect and rebuild this page.</p>
    </div>
  </section>
</main>

<script>
const DATA = {js_data(data)};
const pct = v => v == null ? null : +(v * 100).toFixed(4);
const pctLabel = v => v == null ? "n/a" : (v * 100).toFixed(2) + "%";
const langToggle = document.getElementById("langToggle");
function applyLanguage(lang) {{
  const useEn = lang === "en";
  document.body.classList.toggle("lang-en", useEn);
  document.documentElement.lang = useEn ? "en" : "zh-CN";
  if (langToggle) langToggle.textContent = useEn ? "中文" : "English";
  localStorage.setItem("function-posterior-page-language", useEn ? "en" : "zh");
}}
if (langToggle) {{
  langToggle.addEventListener("click", () => {{
    applyLanguage(document.body.classList.contains("lang-en") ? "zh" : "en");
  }});
}}
applyLanguage(localStorage.getItem("function-posterior-page-language") === "en" ? "en" : "zh");
const chart = (id, option) => {{
  const el = document.getElementById(id);
  const c = echarts.init(el);
  c.setOption(option);
  window.addEventListener("resize", () => c.resize());
  return c;
}};
const grid = {{ left: 70, right: 34, top: 44, bottom: 88, containLabel: true }};
const yPct = {{
  type: "value",
  name: "比例 / fraction",
  nameLocation: "middle",
  nameGap: 48,
  min: 0,
  max: 100,
  axisLabel: {{ formatter: v => v + "%" }}
}};
const logNLabel = v => {{
  const n = Math.round(Math.pow(10, v) - 1);
  if (n <= 0) return "n=0";
  if (n >= 1000) return "n≈" + (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\\.0$/, "") + "k";
  return "n≈" + n;
}};
const xLogN = {{
  type: "value",
  name: "训练样本数 n / training samples n (log10(n+1))",
  nameLocation: "middle",
  nameGap: 48,
  min: 0,
  axisLabel: {{ formatter: logNLabel }}
}};
const xTrainCount = {{
  type: "value",
  name: "训练样本数 / training samples",
  nameLocation: "middle",
  nameGap: 52,
  axisLabel: {{ formatter: v => Number(v).toLocaleString("zh-CN") }}
}};
const xStep = {{
  type: "value",
  name: "训练 step / training step",
  nameLocation: "middle",
  nameGap: 52,
  axisLabel: {{ formatter: v => Number(v).toLocaleString("zh-CN") }}
}};
const line = (name, points, x, y, extra={{}}) => Object.assign({{
  name,
  type: "line",
  showSymbol: points.length < 80,
  symbolSize: 5,
  data: points.map(p => [p[x], pct(p[y]), p.train_count]),
}}, extra);

chart("tinyAgreement", {{
  tooltip: {{ trigger: "axis", formatter: params => params.map(p => {{
    const raw = p.data[2];
    return `${{p.marker}} ${{p.seriesName}} n=${{raw}}: ${{p.value[1].toFixed(2)}}%`;
  }}).join("<br>") }},
  legend: {{ top: 0 }},
  grid,
  xAxis: xLogN,
  yAxis: yPct,
  series: [
    line("pairwise agreement", DATA.tiny_n, "log_train_count_plus_1", "direct_pairwise_agreement"),
    line("entropy (1-H)", DATA.tiny_n.map(p => Object.assign({{}}, p, {{ entropy_comp: p.mean_prediction_entropy_bits == null ? null : 1 - p.mean_prediction_entropy_bits }})), "log_train_count_plus_1", "entropy_comp", {{ lineStyle: {{ type: "dashed" }} }})
  ]
}});
chart("tinyAccuracy", {{
  tooltip: {{ trigger: "axis", formatter: params => params.map(p => `${{p.marker}} ${{p.seriesName}} n=${{p.data[2]}}: ${{p.value[1] == null ? "n/a" : p.value[1].toFixed(2) + "%"}}`).join("<br>") }},
  legend: {{ top: 0 }},
  grid,
  xAxis: xLogN,
  yAxis: yPct,
  series: [
    line("probe bit accuracy", DATA.tiny_n.filter(p => p.mean_probe_bit_accuracy != null), "log_train_count_plus_1", "mean_probe_bit_accuracy"),
    line("excess agreement", DATA.tiny_n.filter(p => p.excess_agreement != null), "log_train_count_plus_1", "excess_agreement", {{ lineStyle: {{ type: "dashed" }} }})
  ]
}});

const byStrategy = rows => rows.reduce((acc, row) => ((acc[row.strategy] ||= []).push(row), acc), {{}});
const activeSoft = byStrategy(DATA.active_soft_curve);
chart("activeSoft", {{
  tooltip: {{ trigger: "axis", valueFormatter: v => v == null ? "n/a" : v.toFixed(2) + "%" }},
  legend: {{ top: 0 }},
  grid,
  xAxis: xTrainCount,
  yAxis: yPct,
  series: Object.entries(activeSoft).map(([name, rows]) => line(name, rows, "train_count", "probe_exact_accuracy"))
}});
const activeTime = byStrategy(DATA.active_time_curve);
chart("activeTime", {{
  tooltip: {{ trigger: "axis", valueFormatter: v => v == null ? "n/a" : v.toFixed(2) + "%" }},
  legend: {{ top: 0 }},
  grid,
  xAxis: xTrainCount,
  yAxis: yPct,
  series: Object.entries(activeTime).map(([name, rows]) => line(name, rows, "train_count", "probe_exact_accuracy"))
}});

const summaryRows = [...DATA.active_soft_summary, ...DATA.active_time_summary];
document.getElementById("activeSummary").innerHTML = `
  <thead><tr><th>实验 / Experiment</th><th>策略 / Strategy</th><th>达到 exact≥0.9 / First exact≥0.9</th><th>达到 exact≥0.99 / First exact≥0.99</th><th>最终样本数 / Final n</th><th>最终 exact / Final exact</th></tr></thead>
  <tbody>${{summaryRows.map(r => `<tr>
    <td>${{r.experiment || ""}}</td>
    <td>${{r.strategy}}</td>
    <td>${{r.first_exact_ge_0_9 ?? r.first_train_count_exact_ge_0_9 ?? "未达到 / not reached"}}</td>
    <td>${{r.first_exact_ge_0_99 ?? r.first_train_count_exact_ge_0_99 ?? "未达到 / not reached"}}</td>
    <td>${{r.final_train_count ?? ""}}</td>
    <td>${{pctLabel(r.final_probe_exact_accuracy)}}</td>
  </tr>`).join("")}}</tbody>`;

chart("singleSeedTime", {{
  title: {{ text: "rule30_layer1：单 seed 时间采样 vs 多 seed", left: "center", top: 0, textStyle: {{ fontSize: 15, fontWeight: 700 }} }},
  tooltip: {{ trigger: "axis", valueFormatter: v => v == null ? "n/a" : v.toFixed(2) + "%" }},
  legend: {{ top: 28 }},
  grid,
  xAxis: xTrainCount,
  yAxis: {{ ...yPct, min: 60, max: 100 }},
  series: [
    line("single-seed time agreement", DATA.single_seed_time, "train_count", "time_pairwise_agreement"),
    line("multi-seed agreement", DATA.single_seed_time.filter(p => p.multiseed_pairwise_agreement != null), "train_count", "multiseed_pairwise_agreement", {{ lineStyle: {{ type: "dashed" }} }}),
    line("single-seed probe bit acc", DATA.single_seed_time, "train_count", "mean_model_bit_accuracy", {{ lineStyle: {{ type: "dotted" }} }})
  ]
}});

const grokkingShown = DATA.grokking_time_curve.filter(p => p.step <= 400);
chart("grokkingTime", {{
  tooltip: {{ trigger: "axis", valueFormatter: v => v == null ? "n/a" : v.toFixed(2) + "%" }},
  legend: {{ top: 0, type: "scroll" }},
  grid,
  xAxis: xStep,
  yAxis: yPct,
  series: [
    line("probe/test bit accuracy", grokkingShown, "step", "probe_bit_accuracy"),
    line("probe/test exact accuracy", grokkingShown, "step", "probe_exact_accuracy", {{ lineStyle: {{ type: "dashed" }} }}),
    line("train bit accuracy", grokkingShown, "step", "train_bit_accuracy", {{ lineStyle: {{ type: "dashed" }} }}),
    line("cross-seed pairwise agreement", grokkingShown, "step", "direct_pairwise_agreement"),
    line("excess agreement", grokkingShown, "step", "excess_agreement", {{ lineStyle: {{ type: "dotted" }} }})
  ]
}});
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    ensure_dirs()

    global SOURCE_ZIPS
    SOURCE_ZIPS, missing = resolve_sources(args.source_dir)
    data_path = OUT_DIR / "dashboard_data.json"

    if missing:
        if not data_path.exists():
            detail = "; ".join(f"{key}: {names}" for key, names in missing.items())
            raise FileNotFoundError(
                "Raw zip packages are missing and no existing dashboard_data.json "
                f"was found. Missing: {detail}"
            )
        data = json.loads(data_path.read_text(encoding="utf-8"))
        output = ROOT / "function_posterior_sampling_experiments.html"
        output.write_text(build_html(data), encoding="utf-8")
        print("raw zip packages not found; rebuilt HTML from existing lightweight data")
        print(f"wrote {output}")
        return

    tiny_n = load_tiny_n_curve()
    active_soft_curve, active_soft_summary = load_active_soft()
    active_time_curve, active_time_summary = load_active_time()
    single_seed_time = load_single_seed_time_summary()
    grokking_time_curve, grokking_meta = load_grokking_time_curve()

    write_csv(OUT_DIR / "tiny_n_rule30_layer1_with_prior.csv", tiny_n)
    write_jsonl(OUT_DIR / "tiny_n_rule30_layer1_with_prior.jsonl", tiny_n)
    write_csv(OUT_DIR / "active_soft_multiseed_curve.csv", active_soft_curve)
    write_csv(OUT_DIR / "active_soft_multiseed_summary.csv", active_soft_summary)
    write_csv(OUT_DIR / "active_time_committee_curve.csv", active_time_curve)
    write_csv(OUT_DIR / "active_time_committee_summary.csv", active_time_summary)
    write_csv(OUT_DIR / "single_seed_time_sampling_summary.csv", single_seed_time)
    write_csv(OUT_DIR / "grokking_agreement_time_axis_curve.csv", grokking_time_curve)
    write_json(OUT_DIR / "grokking_agreement_time_axis_metadata.json", grokking_meta)

    data = {
        "tiny_n": tiny_n,
        "active_soft_curve": active_soft_curve,
        "active_soft_summary": active_soft_summary,
        "active_time_curve": active_time_curve,
        "active_time_summary": active_time_summary,
        "single_seed_time": single_seed_time,
        "grokking_time_curve": grokking_time_curve,
        "grokking_meta": grokking_meta,
    }
    write_json(OUT_DIR / "dashboard_data.json", data)
    html = build_html(data)
    output = ROOT / "function_posterior_sampling_experiments.html"
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote data to {OUT_DIR}")
    print(f"experiment scripts are expected under {EXPERIMENT_SCRIPT_DIR}")


if __name__ == "__main__":
    main()
