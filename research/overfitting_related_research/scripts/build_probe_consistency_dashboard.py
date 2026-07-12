"""
构建 probe 一致性实验的静态可视化网页。

用途：
1. 扫描一个或多个结果根目录。
2. 自动读取根目录中的 sweep_summary.jsonl，以及各实验子目录中的 summary.jsonl。
3. 对同一 task / train_count / stage 的不同 split seed 做聚合。
4. 生成一个可直接上传或本地打开的 ECharts HTML 页面。

这个脚本只依赖 Python 标准库，方便复制到 AutoDL notebook 或终端中运行。
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
        "source_mtime": source_path.stat().st_mtime,
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
            if old is None or row["source_mtime"] >= old["source_mtime"]:
                deduped[key] = row
        valid = list(deduped.values())

    valid.sort(key=lambda row: (
        row.get("difficulty_order", 999),
        row.get("difficulty_label", ""),
        row.get("train_count", -1),
        row.get("split_seed") if row.get("split_seed") is not None else -1,
        row.get("stage", ""),
    ))
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
  .grid {{ grid-template-columns: 1fr; }}
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
<h1>Probe 一致性实验仪表盘</h1>
<p class="subtitle">合并同一目录下分批运行的 summary 结果，按任务和训练样本数展示函数分布集中度、泛化准确率和共同错误结构。</p>

<div class="toolbar">
  <div class="field">
    <label for="view-mode">显示方式</label>
    <select id="view-mode">
      <option value="aggregated">合并 split seed</option>
      <option value="raw">显示每个 split seed</option>
    </select>
  </div>
  <div class="field">
    <label for="stage-select">阶段</label>
    <select id="stage-select"></select>
  </div>
  <div class="field" style="flex:1; min-width:280px;">
    <label>任务</label>
    <div id="task-list" class="task-list"></div>
  </div>
  <button id="select-all" type="button">全选</button>
  <button id="select-rule" type="button">只看规则</button>
  <button id="select-random" type="button">只看随机</button>
</div>

<div class="stats">
  <div class="stat"><div class="label">数据点</div><div id="stat-points" class="value">0</div></div>
  <div class="stat"><div class="label">任务数</div><div id="stat-tasks" class="value">0</div></div>
  <div class="stat"><div class="label">n 范围</div><div id="stat-range" class="value">-</div></div>
  <div class="stat"><div class="label">split 数</div><div id="stat-splits" class="value">-</div></div>
</div>

<div class="grid">
  <div class="panel"><div id="accuracy-chart" class="chart"></div></div>
  <div class="panel"><div id="agreement-chart" class="chart"></div></div>
  <div class="panel"><div id="entropy-chart" class="chart"></div></div>
  <div class="panel"><div id="error-chart" class="chart"></div></div>
  <div class="panel"><div id="bias-chart" class="chart"></div></div>
  <div class="panel"><div id="steps-chart" class="chart"></div></div>
  <div class="panel wide">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>任务</th>
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
  return [...map.entries()].sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0], 'zh-CN')).map(item => item[0]);
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
    .sort((a, b) => orderOf(a) - orderOf(b) || labelOf(a).localeCompare(labelOf(b), 'zh-CN') || a.train_count - b.train_count || String(a.split_seed || '').localeCompare(String(b.split_seed || '')));
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
    return `<b>${{param.seriesName}}</b><br>n=${{param.value[0]}}, value=${{rendered}}<br>bit acc=${{pct(row.mean_probe_bit_accuracy)}} / pairwise=${{pct(row.direct_pairwise_agreement)}}<br>entropy=${{num(row.mean_prediction_entropy_bits)}} / phi=${{num(row.mean_pairwise_error_phi_correlation)}}`;
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
    xAxis: {{ type: 'value', name: '训练样本数 n', nameLocation: 'middle', nameGap: 34 }},
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

  let option = baseOption('泛化准确率', 'accuracy', 100);
  option.series = multiSeries(rows, [
    {{ key: 'mean_probe_bit_accuracy', name: '单模型 bit acc', scale: 100 }},
    {{ key: 'majority_vote_bit_accuracy', name: '多数投票 bit acc', scale: 100, lineStyle: {{ type: 'dashed' }} }},
  ]);
  charts.accuracy.setOption(option, true);

  option = baseOption('函数分布集中度', 'agreement', 100);
  option.series = multiSeries(rows, [
    {{ key: 'direct_pairwise_agreement', name: 'pairwise agreement', scale: 100 }},
    {{ key: 'majority_fraction_agreement', name: 'majority fraction', scale: 100, lineStyle: {{ type: 'dashed' }} }},
    {{ key: 'unanimously_same_prediction_bit_fraction', name: 'unanimous bits', scale: 100, lineStyle: {{ type: 'dotted' }} }},
  ]);
  charts.agreement.setOption(option, true);

  option = baseOption('预测熵与分歧', 'value', null);
  option.yAxis = [
    {{ type: 'value', name: 'entropy bits', min: 0, max: 1 }},
    {{ type: 'value', name: 'hamming', min: 0, max: 0.5 }}
  ];
  option.series = multiSeries(rows, [
    {{ key: 'mean_prediction_entropy_bits', name: 'entropy', scale: 1, yAxisIndex: 0 }},
    {{ key: 'mean_pairwise_prediction_bit_hamming_distance', name: 'pairwise hamming', scale: 1, yAxisIndex: 1, lineStyle: {{ type: 'dashed' }} }},
  ]);
  charts.entropy.setOption(option, true);

  option = baseOption('共同错误结构', 'value', null);
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

  option = baseOption('输出偏置', 'one rate', 1);
  option.series = groupedSeries(rows, 'prediction_one_rate', 1);
  charts.bias.setOption(option, true);

  option = baseOption('训练步数', 'steps', null);
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
window.addEventListener('resize', () => Object.values(charts).forEach(chart => chart.resize()));

initControls();
renderCharts();
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
