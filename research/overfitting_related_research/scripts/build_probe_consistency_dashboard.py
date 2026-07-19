"""
构建 probe 一致性实验的静态可视化网页。

用途：
1. 扫描一个或多个结果根目录。
2. 自动读取根目录中的 sweep_summary.jsonl，以及各实验子目录中的 summary.jsonl。
3. 对同一 task / train_count / stage 的不同 split seed 做聚合。
4. 生成一个可直接上传或本地打开的 ECharts HTML 页面。

这个脚本只依赖 Python 标准库，可以在本地或服务器终端中直接运行。
"""

import csv
import html
import json
import math
import statistics
from pathlib import Path


class Config:
    # 可以放多个目录；脚本会把它们全部扫描后合并。
    RESULT_ROOTS = [
        "research/overfitting_related_research/results",
    ]

    # 输出目录。设为 None 时，写到第一个 RESULT_ROOTS 目录下。
    OUTPUT_DIR = None

    OUTPUT_HTML_NAME = "probe_consistency_dashboard.html"
    CLEAN_ROWS_JSONL_NAME = "probe_consistency_dashboard_rows.jsonl"
    AGGREGATED_JSONL_NAME = "probe_consistency_dashboard_aggregated.jsonl"
    AGGREGATED_CSV_NAME = "probe_consistency_dashboard_aggregated.csv"

    # 根目录下常见的汇总文件名。
    ROOT_SUMMARY_NAMES = [
        "sweep_summary.jsonl",
        "summary_from_subdirectories.jsonl",
    ]

    # If the raw sweep summaries are not present, rebuild the dashboard from the
    # packaged aggregate files shipped with this folder.
    FALLBACK_SUMMARY_NAMES = [
        "probe_consistency_dashboard_aggregated.jsonl",
        "probe_consistency_dashboard_rows.jsonl",
    ]

    # 如果同一个 task/train_count/split_seed/stage 出现多次，默认保留文件修改时间最新的记录。
    KEEP_LATEST_DUPLICATE = True


METRIC_FIELDS = [
    "mean_probe_bit_accuracy",
    "mean_probe_exact_accuracy",
    "majority_vote_bit_accuracy",
    "majority_vote_exact_accuracy",
    "direct_pairwise_agreement",
    "majority_fraction_agreement",
    "unanimously_same_prediction_bit_fraction",
    "mean_prediction_entropy_bits",
    "mean_pairwise_prediction_bit_hamming_distance",
    "mean_pairwise_error_phi_correlation",
    "mean_pairwise_joint_error_lift",
    "mean_pairwise_error_jaccard",
    "prediction_one_rate",
    "mean_train_steps",
    "mean_train_fit_step",
    "pilot_steps",
    "model_count",
    "probe_count",
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def infer_experiment_name(row, source_path):
    if row.get("experiment_name"):
        return str(row["experiment_name"])
    parent = source_path.parent.name
    if parent:
        return parent
    return source_path.stem


def normalize_summary(row, source_path):
    actual_source_mtime = source_path.stat().st_mtime
    task_name = str(row.get("task_name") or row.get("dataset_name") or "unknown_task")
    difficulty_label = str(row.get("difficulty_label") or task_name)
    difficulty_order = safe_float(row.get("difficulty_order"))
    if difficulty_order is None:
        lowered = difficulty_label.lower()
        if "random" in lowered:
            difficulty_order = 0
        elif "layer1" in lowered or "layer 1" in lowered:
            difficulty_order = 1
        elif "layer2" in lowered or "layer 2" in lowered:
            difficulty_order = 2
        elif "layer3" in lowered or "layer 3" in lowered:
            difficulty_order = 3
        else:
            difficulty_order = 999

    train_count = safe_int(row.get("train_count"))
    split_seed = safe_int(row.get("split_seed"))
    stage = str(row.get("stage") or "plateau")
    experiment_name = infer_experiment_name(row, source_path)

    direct_pairwise = (
        safe_float(row.get("mean_direct_pairwise_prediction_bit_agreement"))
        or safe_float(row.get("mean_pairwise_prediction_bit_agreement"))
        or safe_float(row.get("direct_pairwise_agreement"))
    )
    majority_fraction = (
        safe_float(row.get("mean_prediction_bit_agreement"))
        or safe_float(row.get("majority_fraction_agreement"))
    )

    normalized = {
        "record_type": "normalized_summary",
        "experiment_name": experiment_name,
        "task_name": task_name,
        "difficulty_label": difficulty_label,
        "difficulty_order": difficulty_order,
        "task_type": row.get("task_type"),
        "stage": stage,
        "train_count": train_count,
        "split_seed": split_seed,
        "source_file": str(source_path),
        "_source_mtime_sort": actual_source_mtime,
        "direct_pairwise_agreement": direct_pairwise,
        "majority_fraction_agreement": majority_fraction,
    }

    for key in METRIC_FIELDS:
        if key in normalized:
            continue
        normalized[key] = safe_float(row.get(key))

    # 兼容旧脚本里没有 mixed 字段的情况。
    if normalized.get("mean_pairwise_prediction_bit_hamming_distance") is None:
        if direct_pairwise is not None:
            normalized["mean_pairwise_prediction_bit_hamming_distance"] = 1.0 - direct_pairwise

    return normalized


def collect_summary_files(result_roots, cfg):
    files = []
    for root_text in result_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for name in cfg.ROOT_SUMMARY_NAMES:
            path = root / name
            if path.exists() and path.is_file():
                files.append(path)
        for path in root.rglob("summary.jsonl"):
            if path.is_file():
                files.append(path)
    unique = []
    seen = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def collect_rows(cfg):
    summary_files = collect_summary_files(cfg.RESULT_ROOTS, cfg)
    rows = []
    for path in summary_files:
        try:
            for row in read_jsonl(path):
                if row.get("record_type") not in (None, "summary", "normalized_summary"):
                    continue
                normalized = normalize_summary(row, path)
                if normalized["train_count"] is None:
                    continue
                rows.append(normalized)
        except Exception as exc:
            rows.append({
                "record_type": "load_error",
                "source_file": str(path),
                "error": repr(exc),
            })

    valid = [row for row in rows if row.get("record_type") == "normalized_summary"]
    if not valid:
        fallback_loaded = False
        for root_text in cfg.RESULT_ROOTS:
            root = Path(root_text)
            for name in cfg.FALLBACK_SUMMARY_NAMES:
                path = root / name
                if not path.exists():
                    continue
                loaded_count = 0
                try:
                    for row in read_jsonl(path):
                        if row.get("record_type") not in (None, "summary", "normalized_summary", "aggregated_summary"):
                            continue
                        normalized = normalize_summary(row, path)
                        if normalized["train_count"] is None:
                            continue
                        rows.append(normalized)
                        loaded_count += 1
                except Exception as exc:
                    rows.append({
                        "record_type": "load_error",
                        "source_file": str(path),
                        "error": repr(exc),
                    })
                if loaded_count:
                    fallback_loaded = True
                    break
            if fallback_loaded:
                break
        valid = [row for row in rows if row.get("record_type") == "normalized_summary"]
    if cfg.KEEP_LATEST_DUPLICATE:
        deduped = {}
        for row in valid:
            key = (
                row["task_name"],
                row["difficulty_label"],
                row["train_count"],
                row.get("split_seed"),
                row.get("stage"),
            )
            old = deduped.get(key)
            if old is None or row["_source_mtime_sort"] >= old["_source_mtime_sort"]:
                deduped[key] = row
        valid = list(deduped.values())

    valid.sort(key=lambda row: (
        row.get("difficulty_order", 999),
        row.get("difficulty_label", ""),
        row.get("train_count", -1),
        row.get("split_seed") if row.get("split_seed") is not None else -1,
        row.get("stage", ""),
    ))
    for row in valid:
        row.pop("_source_mtime_sort", None)
    return valid, rows


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (
            row["task_name"],
            row["difficulty_label"],
            row["difficulty_order"],
            row["train_count"],
            row["stage"],
        )
        groups.setdefault(key, []).append(row)

    aggregated = []
    for key, group in groups.items():
        task_name, label, order, train_count, stage = key
        record = {
            "record_type": "aggregated_summary",
            "task_name": task_name,
            "difficulty_label": label,
            "difficulty_order": order,
            "train_count": train_count,
            "stage": stage,
            "split_count": len({row.get("split_seed") for row in group}),
            "source_count": len(group),
            "split_seeds": sorted([
                seed for seed in {row.get("split_seed") for row in group}
                if seed is not None
            ]),
        }
        for metric in METRIC_FIELDS:
            values = [row.get(metric) for row in group]
            record[metric] = mean(values)
            record[f"{metric}_std"] = stdev(values)
        aggregated.append(record)

    aggregated.sort(key=lambda row: (
        row.get("difficulty_order", 999),
        row.get("difficulty_label", ""),
        row.get("train_count", -1),
        row.get("stage", ""),
    ))
    return aggregated


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def html_template(raw_rows, aggregated_rows, cfg):
    data = json.dumps({
        "rawRows": raw_rows,
        "aggregatedRows": aggregated_rows,
        "generatedFrom": cfg.RESULT_ROOTS,
    }, ensure_ascii=False, allow_nan=False)
    escaped_data = data.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Probe 一致性实验仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f1e7;
  --panel: #fffaf0;
  --text: #1f2933;
  --muted: #667085;
  --border: #dfd2ba;
  --grid: #e9ddc7;
  --accent: #4f7f2a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 28px;
  font-weight: 600;
  letter-spacing: 0;
}}
.subtitle {{
  margin: 0 0 18px;
  color: var(--muted);
  line-height: 1.6;
}}
.site-nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 0 0 18px;
}}
.site-nav a {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 36px;
  padding: 7px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
  color: #2563eb;
  font-weight: 600;
  text-decoration: none;
}}
.site-nav button {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 36px;
  padding: 7px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
  color: #2563eb;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}}
body.lang-en .zh {{ display: none !important; }}
body:not(.lang-en) .en {{ display: none !important; }}
.explain {{
  margin: 0 0 18px;
  padding: 18px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--panel);
}}
.explain h2 {{
  margin: 0 0 8px;
  font-size: 20px;
}}
.explain p {{
  margin: 0 0 10px;
  line-height: 1.7;
}}
.explain-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}}
.explain-card {{
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #fff;
  padding: 12px;
}}
.explain-card strong {{
  display: block;
  margin-bottom: 4px;
}}
.explain-card span {{
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
}}
.toc {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}}
.toc a {{
  display: block;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #fff;
  color: var(--text);
  text-decoration: none;
}}
.toc a:hover {{
  border-color: #2563eb;
}}
.toc b {{
  display: block;
  color: #2563eb;
}}
.toc small {{
  color: var(--muted);
}}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px 16px;
  align-items: end;
  margin: 0 0 18px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
}}
.field {{
  display: grid;
  gap: 6px;
  min-width: 180px;
}}
.field label {{
  color: var(--muted);
  font-size: 13px;
}}
select, button {{
  font: inherit;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  background: #fff;
  color: var(--text);
}}
button {{ cursor: pointer; }}
.task-list {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  max-width: 100%;
}}
.task-list label {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: #fff;
  color: var(--text);
  font-size: 13px;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  gap: 12px;
  margin: 0 0 18px;
}}
.stat {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
}}
.stat .label {{
  color: var(--muted);
  font-size: 13px;
}}
.stat .value {{
  margin-top: 4px;
  font-size: 24px;
  font-weight: 600;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(360px, 1fr));
  gap: 18px;
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
}}
.chart {{
  width: 100%;
  height: 390px;
}}
.wide {{
  grid-column: 1 / -1;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
th, td {{
  border-bottom: 1px solid var(--border);
  padding: 8px 9px;
  text-align: right;
  white-space: nowrap;
}}
th:first-child, td:first-child {{ text-align: left; }}
th {{
  color: var(--muted);
  font-weight: 600;
}}
.table-wrap {{
  overflow-x: auto;
}}
@media (max-width: 920px) {{
  body {{ padding: 14px; }}
  .grid, .explain-grid, .toc {{ grid-template-columns: 1fr; }}
  .stats {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
}}
@media (max-width: 520px) {{
  .stats {{ grid-template-columns: 1fr; }}
  .field {{ min-width: 100%; }}
  .chart {{ height: 320px; }}
}}
</style>
</head>
<body>
<h1><span class="zh">Probe 一致性实验仪表盘</span><span class="en">Probe Consistency Dashboard</span></h1>
<p class="subtitle">
  <span class="zh">这不是单纯的绘图页。它展示的是：同一个训练集被多个随机 seed 训练到过拟合平台后，这些网络在未见 probe 输入上的函数行为是否收缩到同一个函数，还是分散成许多不同延拓。</span>
  <span class="en">This is not just a plotting page. It asks whether models trained from different random seeds on the same overfitted training set collapse to the same function on unseen probe inputs, or spread into many different extensions.</span>
</p>

<nav class="site-nav">
  <a href="../index.html"><span class="zh">返回研究主页</span><span class="en">Back to overview</span></a>
  <a href="../function_posterior_sampling_experiments.html"><span class="zh">打开函数后验采样补充实验</span><span class="en">Open posterior-sampling addendum</span></a>
  <a href="probe_consistency_dashboard_aggregated.csv"><span class="zh">下载聚合 CSV</span><span class="en">Download aggregate CSV</span></a>
  <button type="button" id="langToggle">English</button>
</nav>

<section class="explain" id="overview">
  <h2><span class="zh">这个实验在测什么？</span><span class="en">What Does This Experiment Measure?</span></h2>
  <p><span class="zh">每个条件固定一个任务和训练样本数 <code>n</code>，训练多个随机 seed。所有模型都先拟合训练集，再在严格未见的 probe 集上输出预测。跨 seed 的 <code>pairwise agreement</code>、预测熵和共同错误结构，被用来观察“训练约束之后的函数分布”有多集中。</span><span class="en">For each condition, a task and a training-set size <code>n</code> are fixed, and many random seeds are trained. Each model first fits the training set and is then evaluated on strictly unseen probe inputs. Cross-seed <code>pairwise agreement</code>, prediction entropy, and shared-error structure are used to inspect how concentrated the function distribution becomes after training constraints are imposed.</span></p>
  <p><span class="zh">任务包括 <code>rule30_layer1/2/3</code> 这三档规则复杂度，以及 <code>random_bits</code> 随机标签对照。规则任务有真实可压缩结构；随机标签没有可泛化结构。两者的 agreement 曲线方向相反，是这个实验最重要的读图线索。</span><span class="en">The tasks include three levels of Rule 30 complexity, <code>rule30_layer1/2/3</code>, plus a <code>random_bits</code> random-label control. Rule tasks contain real compressible structure; random labels do not. The opposite directions of their agreement curves are the central reading clue.</span></p>
  <div class="explain-grid">
    <div class="explain-card">
      <strong><span class="zh">规则任务</span><span class="en">Rule Tasks</span></strong>
      <span class="zh">数据量增加后，函数后验应逐渐收缩到真实规则；任务越复杂，需要的数据越多。</span>
      <span class="en">As data increases, the function posterior should contract toward the true rule. More complex rules require more data.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">随机标签</span><span class="en">Random Labels</span></strong>
      <span class="zh">小数据量下可能出现共享的低复杂度伪规则；数据量增加后随机约束会打碎这种共识。</span>
      <span class="en">With very little data, seeds can share a low-complexity pseudo-rule. As random constraints accumulate, that consensus fragments.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">agreement 的含义</span><span class="en">Meaning of Agreement</span></strong>
      <span class="zh">高 agreement 不等于正确。它表示不同 seed 在 probe 上更像同一个函数；需要和 accuracy 一起读。</span>
      <span class="en">High agreement does not mean correctness. It means different seeds behave more like the same function on the probe set, and must be read together with accuracy.</span>
    </div>
  </div>
  <div class="toc">
    <a href="#controls"><b><span class="zh">筛选器</span><span class="en">Filters</span></b><small><span class="zh">选择任务、阶段和 split 显示方式</span><span class="en">Choose tasks, stage, and split display mode</span></small></a>
    <a href="#summary-stats"><b><span class="zh">汇总统计</span><span class="en">Summary</span></b><small><span class="zh">当前视图的数据点、任务数和 n 范围</span><span class="en">Point count, task count, and n range</span></small></a>
    <a href="#charts"><b><span class="zh">六组曲线</span><span class="en">Six Charts</span></b><small><span class="zh">accuracy、agreement、entropy、共同错误等</span><span class="en">Accuracy, agreement, entropy, shared errors, and more</span></small></a>
    <a href="#table-section"><b><span class="zh">数据表</span><span class="en">Data Table</span></b><small><span class="zh">当前筛选结果的逐行数值</span><span class="en">Rows for the current filtered view</span></small></a>
  </div>
</section>

<section class="explain">
  <h2><span class="zh">如何读这些图？</span><span class="en">How To Read The Charts</span></h2>
  <div class="explain-grid">
    <div class="explain-card">
      <strong><span class="zh">泛化准确率</span><span class="en">Generalization Accuracy</span></strong>
      <span class="zh">单模型 bit accuracy 和多数投票 accuracy。规则任务最终上升到 1，表示后验收缩到真实规则附近。</span>
      <span class="en">Single-model bit accuracy and majority-vote accuracy. In rule tasks, convergence toward 1 indicates contraction toward the true rule.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">函数分布集中度</span><span class="en">Function Concentration</span></strong>
      <span class="zh"><code>direct_pairwise_agreement</code> 是主指标，衡量任意两个 seed 在同一 probe bit 上预测相同的概率。</span>
      <span class="en"><code>direct_pairwise_agreement</code> is the main metric: the probability that two seeds make the same prediction on a probe bit.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">预测熵与分歧</span><span class="en">Entropy And Disagreement</span></strong>
      <span class="zh">entropy 越低、Hamming 距离越低，说明函数系综越集中；随机标签随 n 增大通常更分散。</span>
      <span class="en">Lower entropy and lower Hamming distance mean a more concentrated function ensemble. Random labels usually become more dispersed as n grows.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">共同错误结构</span><span class="en">Shared Error Structure</span></strong>
      <span class="zh">error phi / lift / Jaccard 衡量不同 seed 是否在同一批 probe bit 上共同犯错，可用于观察“集体错误”。</span>
      <span class="en">Error phi, lift, and Jaccard measure whether seeds fail on the same probe bits, exposing shared or collective errors.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">输出偏置</span><span class="en">Output Bias</span></strong>
      <span class="zh"><code>prediction_one_rate</code> 检查模型是否整体偏向输出 0 或 1，帮助排除简单边缘偏置解释。</span>
      <span class="en"><code>prediction_one_rate</code> checks whether outputs are globally biased toward 0 or 1, helping rule out trivial marginal-bias explanations.</span>
    </div>
    <div class="explain-card">
      <strong><span class="zh">训练步数</span><span class="en">Training Steps</span></strong>
      <span class="zh">展示 fit step、pilot steps 和最终训练步数，帮助确认不同任务和数据量的训练协议可比。</span>
      <span class="en">Fit step, pilot steps, and final training steps help check that conditions are compared under a consistent plateau protocol.</span>
    </div>
  </div>
</section>

<div class="toolbar" id="controls">
  <div class="field">
    <label for="view-mode"><span class="zh">显示方式</span><span class="en">View mode</span></label>
    <select id="view-mode">
      <option value="aggregated">合并 split seed</option>
      <option value="raw">显示每个 split seed</option>
    </select>
  </div>
  <div class="field">
    <label for="stage-select"><span class="zh">阶段</span><span class="en">Stage</span></label>
    <select id="stage-select"></select>
  </div>
  <div class="field" style="flex:1; min-width:280px;">
    <label><span class="zh">任务</span><span class="en">Tasks</span></label>
    <div id="task-list" class="task-list"></div>
  </div>
  <button id="select-all" type="button"><span class="zh">全选</span><span class="en">All</span></button>
  <button id="select-rule" type="button"><span class="zh">只看规则</span><span class="en">Rules only</span></button>
  <button id="select-random" type="button"><span class="zh">只看随机</span><span class="en">Random only</span></button>
</div>

<div class="stats" id="summary-stats">
  <div class="stat"><div class="label"><span class="zh">数据点</span><span class="en">Data points</span></div><div id="stat-points" class="value">0</div></div>
  <div class="stat"><div class="label"><span class="zh">任务数</span><span class="en">Tasks</span></div><div id="stat-tasks" class="value">0</div></div>
  <div class="stat"><div class="label"><span class="zh">n 范围</span><span class="en">n range</span></div><div id="stat-range" class="value">-</div></div>
  <div class="stat"><div class="label"><span class="zh">split 数</span><span class="en">Splits</span></div><div id="stat-splits" class="value">-</div></div>
</div>

<div class="grid" id="charts">
  <div class="panel"><div id="accuracy-chart" class="chart"></div></div>
  <div class="panel"><div id="agreement-chart" class="chart"></div></div>
  <div class="panel"><div id="entropy-chart" class="chart"></div></div>
  <div class="panel"><div id="error-chart" class="chart"></div></div>
  <div class="panel"><div id="bias-chart" class="chart"></div></div>
  <div class="panel"><div id="steps-chart" class="chart"></div></div>
  <div class="panel wide" id="table-section">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th><span class="zh">任务</span><span class="en">Task</span></th>
            <th>n</th>
            <th>split</th>
            <th>bit acc</th>
            <th>majority acc</th>
            <th>pairwise agreement</th>
            <th>majority fraction</th>
            <th>entropy</th>
            <th>phi</th>
            <th>lift</th>
            <th>steps</th>
          </tr>
        </thead>
        <tbody id="data-table"></tbody>
      </table>
    </div>
  </div>
</div>

<script id="dashboard-data" type="application/json">{escaped_data}</script>
<script>
const payload = JSON.parse(document.getElementById('dashboard-data').textContent);
const rawRows = payload.rawRows;
const aggregatedRows = payload.aggregatedRows;

const charts = {{
  accuracy: echarts.init(document.getElementById('accuracy-chart')),
  agreement: echarts.init(document.getElementById('agreement-chart')),
  entropy: echarts.init(document.getElementById('entropy-chart')),
  error: echarts.init(document.getElementById('error-chart')),
  bias: echarts.init(document.getElementById('bias-chart')),
  steps: echarts.init(document.getElementById('steps-chart')),
}};

const viewMode = document.getElementById('view-mode');
const stageSelect = document.getElementById('stage-select');
const taskList = document.getElementById('task-list');
const tableBody = document.getElementById('data-table');
const langToggle = document.getElementById('langToggle');

const I18N = {{
  zh: {{
    pageTitle: 'Probe 一致性实验仪表盘',
    mergedSplit: '合并 split seed',
    rawSplit: '显示每个 split seed',
    trainXAxis: '训练样本数 n',
    value: '数值',
    bitAcc: 'bit acc',
    pairwise: 'pairwise',
    entropy: 'entropy',
    phi: 'phi',
    accuracyTitle: '泛化准确率',
    accuracyAxis: 'accuracy',
    singleBitAcc: '单模型 bit acc',
    majorityBitAcc: '多数投票 bit acc',
    agreementTitle: '函数分布集中度',
    agreementAxis: 'agreement',
    majorityFraction: 'majority fraction',
    unanimousBits: 'unanimous bits',
    entropyTitle: '预测熵与分歧',
    valueAxis: 'value',
    errorTitle: '共同错误结构',
    biasTitle: '输出偏置',
    oneRateAxis: 'one rate',
    stepsTitle: '训练步数',
    stepsAxis: 'steps',
  }},
  en: {{
    pageTitle: 'Probe Consistency Dashboard',
    mergedSplit: 'Merge split seeds',
    rawSplit: 'Show each split seed',
    trainXAxis: 'training samples n',
    value: 'value',
    bitAcc: 'bit acc',
    pairwise: 'pairwise',
    entropy: 'entropy',
    phi: 'phi',
    accuracyTitle: 'Generalization Accuracy',
    accuracyAxis: 'accuracy',
    singleBitAcc: 'single-model bit acc',
    majorityBitAcc: 'majority-vote bit acc',
    agreementTitle: 'Function Distribution Concentration',
    agreementAxis: 'agreement',
    majorityFraction: 'majority fraction',
    unanimousBits: 'unanimous bits',
    entropyTitle: 'Prediction Entropy And Disagreement',
    valueAxis: 'value',
    errorTitle: 'Shared Error Structure',
    biasTitle: 'Output Bias',
    oneRateAxis: 'one rate',
    stepsTitle: 'Training Steps',
    stepsAxis: 'steps',
  }},
}};
let currentLang = 'zh';

function t(key) {{
  return (I18N[currentLang] && I18N[currentLang][key]) || key;
}}
function applyLanguage(lang) {{
  currentLang = lang === 'en' ? 'en' : 'zh';
  const useEn = currentLang === 'en';
  document.body.classList.toggle('lang-en', useEn);
  document.documentElement.lang = useEn ? 'en' : 'zh-CN';
  document.title = t('pageTitle');
  if (langToggle) langToggle.textContent = useEn ? '中文' : 'English';
  const aggregatedOption = viewMode.querySelector('option[value="aggregated"]');
  const rawOption = viewMode.querySelector('option[value="raw"]');
  if (aggregatedOption) aggregatedOption.textContent = t('mergedSplit');
  if (rawOption) rawOption.textContent = t('rawSplit');
  localStorage.setItem('probe-consistency-dashboard-language', currentLang);
  renderCharts();
}}

function labelOf(row) {{
  return row.difficulty_label || row.task_name || 'unknown';
}}
function orderOf(row) {{
  return Number.isFinite(row.difficulty_order) ? row.difficulty_order : 999;
}}
function pct(value) {{
  return value == null || Number.isNaN(value) ? '-' : (value * 100).toFixed(2) + '%';
}}
function num(value, digits=4) {{
  return value == null || Number.isNaN(value) ? '-' : Number(value).toFixed(digits);
}}
function taskNames(rows) {{
  const map = new Map();
  rows.forEach(row => {{
    const label = labelOf(row);
    if (!map.has(label)) map.set(label, orderOf(row));
  }});
  return [...map.entries()].sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0], currentLang === 'en' ? 'en' : 'zh-CN')).map(item => item[0]);
}}
function stages() {{
  const values = new Set([...rawRows, ...aggregatedRows].map(row => row.stage || 'plateau'));
  return [...values].sort();
}}
function currentRows() {{
  const rows = viewMode.value === 'raw' ? rawRows : aggregatedRows;
  const selectedTasks = new Set([...taskList.querySelectorAll('input:checked')].map(input => input.value));
  const stage = stageSelect.value;
  return rows
    .filter(row => selectedTasks.has(labelOf(row)))
    .filter(row => (row.stage || 'plateau') === stage)
    .sort((a, b) => orderOf(a) - orderOf(b) || labelOf(a).localeCompare(labelOf(b), currentLang === 'en' ? 'en' : 'zh-CN') || a.train_count - b.train_count || String(a.split_seed || '').localeCompare(String(b.split_seed || '')));
}}

function initControls() {{
  stageSelect.innerHTML = stages().map(stage => `<option value="${{stage}}">${{stage}}</option>`).join('');
  const labels = taskNames([...rawRows, ...aggregatedRows]);
  taskList.innerHTML = labels.map(label => {{
    const id = 'task-' + label.replace(/[^a-zA-Z0-9_-]/g, '-');
    return `<label for="${{id}}"><input id="${{id}}" type="checkbox" value="${{htmlEscape(label)}}" checked> ${{htmlEscape(label)}}</label>`;
  }}).join('');
}}
function htmlEscape(text) {{
  return String(text).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}

function groupedSeries(rows, key, scale=1, nameSuffix='', connectNulls=false) {{
  const labels = taskNames(rows);
  return labels.map(label => {{
    const points = rows
      .filter(row => labelOf(row) === label)
      .sort((a, b) => a.train_count - b.train_count)
      .map(row => [row.train_count, row[key] == null ? null : row[key] * scale, row]);
    return {{
      name: nameSuffix ? `${{label}} ${{nameSuffix}}` : label,
      type: 'line',
      showSymbol: true,
      symbolSize: 7,
      connectNulls,
      data: points,
      emphasis: {{ focus: 'series' }},
    }};
  }});
}}
function multiSeries(rows, specs) {{
  const series = [];
  taskNames(rows).forEach(label => {{
    specs.forEach(spec => {{
      const points = rows
        .filter(row => labelOf(row) === label)
        .sort((a, b) => a.train_count - b.train_count)
        .map(row => [row.train_count, row[spec.key] == null ? null : row[spec.key] * spec.scale, row]);
      series.push({{
        name: `${{label}} - ${{spec.name}}`,
        type: 'line',
        showSymbol: true,
        symbolSize: 6,
        data: points,
        yAxisIndex: spec.yAxisIndex || 0,
        lineStyle: spec.lineStyle || {{}},
        emphasis: {{ focus: 'series' }},
      }});
    }});
  }});
  return series;
}}
function tooltipFormatter(params) {{
  return params.map(param => {{
    const row = param.data[2] || {{}};
    const value = param.value[1];
    const rendered = Math.abs(value) <= 1.2 ? num(value) : num(value, 2);
    return `<b>${{param.seriesName}}</b><br>n=${{param.value[0]}}, ${{t('value')}}=${{rendered}}<br>${{t('bitAcc')}}=${{pct(row.mean_probe_bit_accuracy)}} / ${{t('pairwise')}}=${{pct(row.direct_pairwise_agreement)}}<br>${{t('entropy')}}=${{num(row.mean_prediction_entropy_bits)}} / ${{t('phi')}}=${{num(row.mean_pairwise_error_phi_correlation)}}`;
  }}).join('<hr style="border:none;border-top:1px solid #ddd;margin:6px 0">');
}}
function baseOption(title, yName, yMax=null) {{
  return {{
    title: {{ text: title, left: 6, top: 4, textStyle: {{ fontSize: 16, fontWeight: 600 }} }},
    tooltip: {{ trigger: 'axis', appendToBody: true, formatter: tooltipFormatter }},
    legend: {{ type: 'scroll', top: 32, left: 8, right: 8 }},
    grid: {{ left: 58, right: 24, top: 88, bottom: 62 }},
    dataZoom: [
      {{ type: 'inside', xAxisIndex: 0 }},
      {{ type: 'slider', xAxisIndex: 0, height: 22, bottom: 18 }}
    ],
    xAxis: {{ type: 'value', name: t('trainXAxis'), nameLocation: 'middle', nameGap: 34 }},
    yAxis: {{ type: 'value', name: yName, min: 0, max: yMax, axisLabel: {{ formatter: value => yMax === 100 ? value + '%' : value }} }},
    series: []
  }};
}}

function renderCharts() {{
  const rows = currentRows();
  const allTrainCounts = rows.map(row => row.train_count).filter(value => value != null);
  const splitValues = new Set(rows.flatMap(row => row.split_seeds || [row.split_seed]).filter(value => value != null));
  document.getElementById('stat-points').textContent = rows.length;
  document.getElementById('stat-tasks').textContent = taskNames(rows).length;
  document.getElementById('stat-range').textContent = allTrainCounts.length ? `${{Math.min(...allTrainCounts)}}-${{Math.max(...allTrainCounts)}}` : '-';
  document.getElementById('stat-splits').textContent = splitValues.size || '-';

  let option = baseOption(t('accuracyTitle'), t('accuracyAxis'), 100);
  option.series = multiSeries(rows, [
    {{ key: 'mean_probe_bit_accuracy', name: t('singleBitAcc'), scale: 100 }},
    {{ key: 'majority_vote_bit_accuracy', name: t('majorityBitAcc'), scale: 100, lineStyle: {{ type: 'dashed' }} }},
  ]);
  charts.accuracy.setOption(option, true);

  option = baseOption(t('agreementTitle'), t('agreementAxis'), 100);
  option.series = multiSeries(rows, [
    {{ key: 'direct_pairwise_agreement', name: 'pairwise agreement', scale: 100 }},
    {{ key: 'majority_fraction_agreement', name: t('majorityFraction'), scale: 100, lineStyle: {{ type: 'dashed' }} }},
    {{ key: 'unanimously_same_prediction_bit_fraction', name: t('unanimousBits'), scale: 100, lineStyle: {{ type: 'dotted' }} }},
  ]);
  charts.agreement.setOption(option, true);

  option = baseOption(t('entropyTitle'), t('valueAxis'), null);
  option.yAxis = [
    {{ type: 'value', name: 'entropy bits', min: 0, max: 1 }},
    {{ type: 'value', name: 'hamming', min: 0, max: 0.5 }}
  ];
  option.series = multiSeries(rows, [
    {{ key: 'mean_prediction_entropy_bits', name: 'entropy', scale: 1, yAxisIndex: 0 }},
    {{ key: 'mean_pairwise_prediction_bit_hamming_distance', name: 'pairwise hamming', scale: 1, yAxisIndex: 1, lineStyle: {{ type: 'dashed' }} }},
  ]);
  charts.entropy.setOption(option, true);

  option = baseOption(t('errorTitle'), t('valueAxis'), null);
  option.yAxis = [
    {{ type: 'value', name: 'phi / Jaccard', min: 0 }},
    {{ type: 'value', name: 'lift', min: 0 }}
  ];
  option.series = multiSeries(rows, [
    {{ key: 'mean_pairwise_error_phi_correlation', name: 'error phi', scale: 1, yAxisIndex: 0 }},
    {{ key: 'mean_pairwise_error_jaccard', name: 'error Jaccard', scale: 1, yAxisIndex: 0, lineStyle: {{ type: 'dashed' }} }},
    {{ key: 'mean_pairwise_joint_error_lift', name: 'error lift', scale: 1, yAxisIndex: 1, lineStyle: {{ type: 'dotted' }} }},
  ]);
  charts.error.setOption(option, true);

  option = baseOption(t('biasTitle'), t('oneRateAxis'), 1);
  option.series = groupedSeries(rows, 'prediction_one_rate', 1);
  charts.bias.setOption(option, true);

  option = baseOption(t('stepsTitle'), t('stepsAxis'), null);
  option.yAxis.min = null;
  option.series = multiSeries(rows, [
    {{ key: 'mean_train_steps', name: 'mean train steps', scale: 1 }},
    {{ key: 'mean_train_fit_step', name: 'fit step', scale: 1, lineStyle: {{ type: 'dashed' }} }},
    {{ key: 'pilot_steps', name: 'pilot steps', scale: 1, lineStyle: {{ type: 'dotted' }} }},
  ]);
  charts.steps.setOption(option, true);

  renderTable(rows);
}}
function renderTable(rows) {{
  tableBody.innerHTML = rows.map(row => `
    <tr>
      <td>${{htmlEscape(labelOf(row))}}</td>
      <td>${{row.train_count}}</td>
      <td>${{row.split_count || row.split_seed || '-'}}</td>
      <td>${{pct(row.mean_probe_bit_accuracy)}}</td>
      <td>${{pct(row.majority_vote_bit_accuracy)}}</td>
      <td>${{pct(row.direct_pairwise_agreement)}}</td>
      <td>${{pct(row.majority_fraction_agreement)}}</td>
      <td>${{num(row.mean_prediction_entropy_bits)}}</td>
      <td>${{num(row.mean_pairwise_error_phi_correlation)}}</td>
      <td>${{num(row.mean_pairwise_joint_error_lift, 2)}}</td>
      <td>${{num(row.mean_train_steps, 0)}}</td>
    </tr>
  `).join('');
}}

document.getElementById('select-all').addEventListener('click', () => {{
  taskList.querySelectorAll('input').forEach(input => input.checked = true);
  renderCharts();
}});
document.getElementById('select-rule').addEventListener('click', () => {{
  taskList.querySelectorAll('input').forEach(input => {{
    input.checked = !input.value.toLowerCase().includes('random');
  }});
  renderCharts();
}});
document.getElementById('select-random').addEventListener('click', () => {{
  taskList.querySelectorAll('input').forEach(input => {{
    input.checked = input.value.toLowerCase().includes('random');
  }});
  renderCharts();
}});
viewMode.addEventListener('change', renderCharts);
stageSelect.addEventListener('change', renderCharts);
taskList.addEventListener('change', renderCharts);
if (langToggle) {{
  langToggle.addEventListener('click', () => {{
    applyLanguage(currentLang === 'en' ? 'zh' : 'en');
  }});
}}
window.addEventListener('resize', () => Object.values(charts).forEach(chart => chart.resize()));

initControls();
applyLanguage(localStorage.getItem('probe-consistency-dashboard-language') === 'en' ? 'en' : 'zh');
</script>
</body>
</html>
"""


def build_dashboard(cfg):
    rows, all_rows = collect_rows(cfg)
    if not rows:
        raise RuntimeError("没有找到可用的 summary 记录。请检查 RESULT_ROOTS 是否指向 sweep 结果目录。")
    aggregated = aggregate_rows(rows)

    output_dir = Path(cfg.OUTPUT_DIR) if cfg.OUTPUT_DIR else Path(cfg.RESULT_ROOTS[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / cfg.CLEAN_ROWS_JSONL_NAME
    aggregated_path = output_dir / cfg.AGGREGATED_JSONL_NAME
    csv_path = output_dir / cfg.AGGREGATED_CSV_NAME
    html_path = output_dir / cfg.OUTPUT_HTML_NAME

    write_jsonl(clean_path, rows)
    write_jsonl(aggregated_path, aggregated)
    write_csv(csv_path, aggregated)
    html_path.write_text(html_template(rows, aggregated, cfg), encoding="utf-8", newline="\n")

    load_errors = [row for row in all_rows if row.get("record_type") == "load_error"]
    print(f"读取 summary 记录：{len(rows)}")
    print(f"聚合后数据点：{len(aggregated)}")
    print(f"干净明细：{clean_path}")
    print(f"聚合 JSONL：{aggregated_path}")
    print(f"聚合 CSV：{csv_path}")
    print(f"可视化网页：{html_path}")
    if load_errors:
        print(f"注意：有 {len(load_errors)} 个文件读取失败，已跳过。")


def main():
    build_dashboard(Config())


if __name__ == "__main__":
    main()
