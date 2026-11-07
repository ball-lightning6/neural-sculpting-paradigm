# Mod3 原始训练源码：14-bit 与 16-bit

[English](README_en.md) · [下载原始源码 ZIP](original_sources.zip)

这是 2026-11-02 两个配对训练实验的源码附件。只想读实现，可从 14-bit 的 `experiment.py` 开始；只想查下载是否完整，可直接看末节。本次整理没有运行训练、smoke 或模型前向，也没有使用 GPU。下文的准确率来自原作者已经保存的训练记录。

## 两个实验在比较什么

给定一个高位在前的二进制整数，预测它除以 3 的余数，不是数其中有几个 1。读入第 `j` 位时，前缀余数是 `r[j] = (2*r[j-1] + x[j]) % 3`，初值为 0；余数 0、1、2 分别编码为 `00`、`01`、`10`。两组模型结构相同：`direct` 只监督最后两位；`scaffold` 在同一个末端 BCE 上，加上所有较短前缀标签的平均 BCE，两项系数都是 1。测试时两组都只得到输入位；节省输入样本不等于减少总监督信息。

| 任务 | 输入与输出 | 数据划分（每份随机排列） |
|---|---|---|
| 14-bit：粗扫并补一个中间档 | 14 位左补零到 30 维；28 个输出，辅助监督前 13 个前缀 | 8192 个候选训练输入，另 8192 个固定测试输入 |
| 16-bit：在更长输入上复核 | 16 位左补零到 30 维；32 个输出，辅助监督前 15 个前缀 | 32768 个候选训练输入，另 32768 个固定测试输入 |

每档取训练池的前 `n` 个不同输入，各档嵌套。完整档用两份数据（种子 `202611020`、`202611021`），每份各用初始化 `0/1/2`，共 6 对模型，不是 6 份独立数据。每对共享初始化和 minibatch 序列。骨干为三层宽 1024 的 GELU + LayerNorm MLP；AdamW，weight decay 和 dropout 均为 0，学习率 0.001、预热 10 步，batch 512、有放回抽样。

## 原作者实际跑了哪些档位

表中 exact 指最终余数的两位都正确。正式预算为 4000、8000、16000 步；每个预算取 `T-2000、T-1000、T` 三个记录点的测试 exact 均值，再对运行取平均，不选最高点。“稳定 99%”要求三次各自均达 99%；训练拟合另看末点训练集的主任务 BCE ≤ 0.001。以下例子均取 16000 步窗口，数值顺序为 **direct → scaffold**。

| 历史阶段 | 实际完成的训练输入数 n | 已训模型数 | 测试 exact 示例 |
|---|---|---:|---|
| 14-bit 粗扫及桥接 | 256、1024、4096、8192；后补 2048，全部完整 | 48 + 12 = 60 | 2048：80.16% → 99.14%；稳定 99% 为 0/6 → 5/6 |
| 16-bit 粗扫 | 1024、2048、4096、8192、32768，全部完整 | 60 | 32768：99.98% → 99.99%，两组均有 6/6 稳定达到 99% |
| 16-bit 增补，完整档 | 3000、6000 | 24 | 3000：30.60% → 91.59%；6000：61.38% → 99.86%（稳定 99% 为 0/6 → 6/6） |
| 16-bit 增补，部分档 | 2500 仅数据种子 `202611020`，另一份数据未跑 | 6 | 31.72% → 54.77%，两组均为 0/3 稳定达到 99%；不并入完整档 |

**16-bit 是原作者手动停止的。** 当前组保存后，3500、5000、7000 未启动；12000、16000 **个样本**的提议已取消。16-bit 合计 90 个模型；两项实验所有已启动的正式模型都到了 16000 **步**。这支持已测网格和预算上的输入样本效率改善，不给出精确相变点。查原记录时，14-bit 对应 `results_combined/ANALYSIS.json`；16-bit 分看 `results/ANALYSIS.json`、`results_refined/ANALYSIS.json` 和 `results_refined/STOPPED_SUMMARY.json`。这些不是 ZIP 内的文件。

## 源码地图

解压到**新的空目录**，保留以 `research/` 开头的路径；两个同名的 `validate_results.py` 不可互换。下表的 14-bit、16-bit 分别指 `research/mod3_scaffold_sample_trend_20261102/`、`research/mod3_scaffold_sample_trend_16bit_20261102/`；共享模块均在 `research/ca_phase_transition/`。

| 目录 | 文件 | 作用 |
|---|---|---|
| 14-bit | `launch.py` | 加入 Windows 环境的包路径，设置 cuBLAS 等环境变量，转交参数给训练入口；加载此文件就会启动入口。 |
| 14-bit | `experiment.py` | 两种长度共用的核心：构造数据、核验前缀标签、定义配对损失、预检梯度与 CUDA Graph、训练和恢复分组、保存评估；其 `main` 驱动 14-bit 粗扫。 |
| 共享 | `experiment_mod3_endpoint_scaffold_matched_bce.py` | 提供 `prefix_labels`、AdamW、评估、状态恢复和检查点写入。当前任务**不使用**它的旧损失或独立训练入口。 |
| 共享 | `experiment_mod3_small_data_grokking.py` | 提供整数与余数的位编码、`BatchedIndependentMLP`；沿模型轴批量运算但参数独立。无需运行它的旧实验。 |
| 14-bit | `analyze.py` | 两种长度共用的分析函数 `analyze(out)`，读取协议和逐组记录，写固定窗口、配对差值、CSV 和图；由驱动调用，无独立命令行入口。 |
| 14-bit | `extend_midpoint.py` | 读取原四档协议和汇总，检查源码，只训练 2048 档，再核对初始化和数据排列，合并五档。 |
| 14-bit | `validate_results.py` | 用 NumPy 核验五档的嵌套划分、已存前向差值、窗口和配对统计；写合并审计，不训练。 |
| 16-bit | `run_16bit.py` | 导入 14-bit 的训练与分析核心，配置 16-bit 粗扫或 smoke，单独写结果。若存在 14-bit 正式清单，会核对协议和冻存的分析源码。 |
| 16-bit | `validate_results.py` | 用 NumPy 从种子重建排列，核验粗扫或 smoke 的数据、配对、已存前向和统计；不验证增补档。 |
| 16-bit | `refine_counts.py` | 在粗扫结果上追加过渡档，每个“样本数 × 数据种子”分组开始前检查停止标记。 |
| 16-bit | `audit_refinement.py` | 校验粗扫及**全部**计划增补组，合并并画图；完整性断言不能用于历史上已停止的部分结果。 |

## 环境与运行顺序

只读源码或检查 ZIP 不需要第三方库。训练需要 Python、NumPy、**CUDA 版 PyTorch** 和 Matplotlib（旧模型模块在导入时就加载它，绘图使用 Agg）。历史命令用 Windows 基础解释器 `C:/Python313/python.exe`；入口把解压根下的 `.venv-gpu/Lib/site-packages` 写入搜索路径。需要自行准备与解释器兼容的包和驱动，附件不含环境或锁定文件。正式清单记录 RTX 3080、Torch `2.11.0+cu130`，16-bit 另记录 NumPy `2.4.1`；训练用 float32、关闭 TF32。

**正式训练和两个训练 smoke 都硬编码逻辑设备 `cuda:1`，使用 CUDA Graph，没有 CPU 回退或设备选择参数。** 只有 `cuda:0` 的机器不能原样运行。下面是运行说明，不是构建步骤。从自己解压出的根目录打开 PowerShell；不要在原实验目录执行，也不要让两个进程写同一结果目录。

```powershell
$py = 'C:/Python313/python.exe'
$flags = '-E', '-s', '-X', 'utf8', '-u'
$d14 = 'research/mod3_scaffold_sample_trend_20261102'
$d16 = 'research/mod3_scaffold_sample_trend_16bit_20261102'
```

| 步骤 | PowerShell 命令 | 前置条件与产物（均相对各自实验目录） |
|---|---|---|
| 14-bit smoke，可选 | `& $py @flags "$d14/launch.py" --smoke` | 不依赖历史结果；实际训练 20 步，缩为 6 位输入、宽 32、n=16/32，写 `smoke/`。14-bit 结果审计没有 smoke 模式。 |
| 14-bit 粗扫 | `& $py @flags "$d14/launch.py"` | 无结果前置，默认 `--hours 1.5`，写 `results/`。如用 `--out` 改道，桥接脚本仍只查默认 `results/`。 |
| 14-bit 桥接 | `& $py @flags "$d14/extend_midpoint.py"` | 需默认 `results/` 的清单、冻存分析源码和 8 个已完成组的汇总。固定 30 分钟预算；新组写 `results_midpoint/`，合并表、图及 `PROVENANCE.json` 写 `results_combined/`。 |
| 14-bit 五档审计，CPU | `& $py @flags "$d14/validate_results.py"` | 需粗扫和桥接的协议、冻存源码、预检、全部 10 个组的 JSON/NPZ 及合并分析。生成或改写 `results_combined/AUDIT.json`。 |
| 16-bit smoke，可选 | `& $py @flags "$d16/run_16bit.py" --smoke` | 不依赖历史结果；保留 16 位，宽 32、n=256、20 步，写 `smoke/`。结束后可用 `& $py @flags "$d16/validate_results.py" --smoke` 在 CPU 核验并写 `smoke/AUDIT.json`。 |
| 16-bit 粗扫 | `& $py @flags "$d16/run_16bit.py"` | 需 14-bit **源码**，不强制先跑 14-bit；若有其正式清单，也须有匹配的 `results/source/analyze.py`。默认 `--hours 2`，写本任务的 `results/`。 |
| 16-bit 粗扫审计，CPU | `& $py @flags "$d16/validate_results.py"` | 需本任务粗扫全部 10 组、清单、冻存源码、预检、逐组 JSON/NPZ 和分析。生成或改写 `results/AUDIT.json`。 |

**16-bit 增补不是重现历史停止状态的自动步骤。** 原入口 `& $py @flags "$d16/refine_counts.py"` 需要本任务粗扫 10 组完成的 `results/ANALYSIS.json`、`MANIFEST.json`、匹配的源码及相同 Torch 版本。默认预算 1 小时，按 3000、6000、2500、3500、5000、7000 顺序写 `results_refined/`。原作者留下的 `results_refined/STOP_AFTER_GROUP` 不在源码包内；在全新目录运行会尝试后续组，不会自动复演当时的手动停止。只有粗扫和计划中的 12 个增补组**全部**完成后，`& $py @flags "$d16/audit_refinement.py"` 才能写 `results_combined/` 和 `transition_refined.png`。历史增补仅完成 5 个分组，没有这份完整合并产物。

## 输出与来源核验

每个训练输出目录（如 `results/`）含 `MANIFEST.json`、`source/`、`VERIFICATION.json` 和分组进度 `RUN_STATUS.json`。每个 `data{seed}_n{n}/` 下有 `result.json`（训练和测试轨迹）、`data_ids.npz`、`forward_audit.npz` 及 `checkpoint.pt`（权重、优化器与抽样状态）。驱动调用分析后，同级可查 `ANALYSIS.json`、`summary.csv`、`paired_changes.csv`、`window_results.csv`、`trajectory.csv` 和 `sample_requirement_trend.png`。CPU 结果审计读的是已有记录，不会凭源码生成这些输入。

ZIP 只含上表 11 个原始 `.py`，不含日志、数据、权重、结果目录或第三方包。完整性依据是仓库的 `research/neural_k_framework/supplementary/updates_20261106/scaffolding/source_sha256.json`：其 `files` 键就是可与 ZIP 成员比对的原路径。11 个成员均与清单 SHA-256 相同；其中 10 个取自正式运行的冻存源码，并逐字节核对了四个运行目录下全部 22 份 `source/*.py` 副本。唯一例外是 14-bit 的 `validate_results.py`：没有正式运行时副本，按清单哈希从实验目录原样收录。

在 ZIP 所在目录可做以下只读检查，不需 GPU，也不执行其中的代码。ZIP 为 **44289 字节**，SHA-256 为 `7a0474f6502a6cee70797a6f14771f321d3b454dc0bac143392576db39cbe91f`。

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath original_sources.zip
python -I -S -B -m zipfile -l original_sources.zip
python -I -S -B -m zipfile -t original_sources.zip
```
