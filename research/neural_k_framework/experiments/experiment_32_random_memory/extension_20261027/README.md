# E32 十月扩展复现说明

本目录只补充既有 E32，不新增实验编号，不覆盖历史 README、报告或代码。它是已完成工作的追溯性发布整理，不是事前正式预注册，也没有为发布追加训练或 SMC。

## 安全默认值

从 E32 实验目录执行：

```bash
python -m pip install -r extension_20261027/requirements.txt
python extension_20261027/run.py
python extension_20261027/verify.py
python extension_20261027/plot_evidence.py
```

依赖 Python 3.11 或更新版本、NumPy、PyTorch、SciPy、Matplotlib。实际本地检查环境另见 TEST_RESULTS.json。这里没有安装依赖的自动行为。

config.py 的 TASK 默认是 review。ALLOW_TRAINING、ALLOW_SMC、ALLOW_FULL_ENUMERATION 均为 False。直接执行复制的历史科学模块只打印提示，不调用其主程序。verify.py 只编译、导入、检查小表并执行小规模单元测试；不训练、不采样、不完整枚举。

只有使用者明确修改顶部配置才可启动昂贵复现。DEVICE 固定使用逻辑 cuda:0，CUDA_VISIBLE_DEVICES 指定一张实际显卡；默认映射为物理卡1，不静默回退 CPU。不要与已有 GPU 任务抢占资源。WORK_DIR 默认是本目录下 _runs，重算文件与已发布 evidence 分开，历史半池不能跨设备接续。

## 子主题与任务

| TASK | 入口 | 输入与输出 | 默认许可 |
|---|---|---|---|
| review | review.py | evidence 小表、原始数组 hash；输出 _runs/review_summary.json | 无训练、无 SMC |
| dictionary | structure/audit_data.py | 自动生成原四表；固定有限字典重算至 _runs/structure | CPU 分析 |
| structure | structure/audit.py | 自动生成原四表及128参照表；可选 LOSS_PAIRING_DIR | CPU 分析，最长180秒 |
| agreement | agreement/train_agreement.py | 原四表与投影对照；输出 _runs/agreement | ALLOW_TRAINING |
| agreement_analysis | agreement/analyze_agreement.py | _runs/agreement/AGREEMENT.json 与各组 predictions.npz | 需已有预测文件 |
| loss | loss/experiment.py | 原四表与投影对照；输出 _runs/loss | ALLOW_TRAINING |
| loss_analysis | loss/analyze.py | _runs/loss 的 PLAN.json、STATUS.json、RESULT.json、observations.npz | 需已有观察文件 |
| base_one / base_two / base_three | smc/protocols.py | 同四表八档 n、每条件四池；各任务独立输出目录 | ALLOW_SMC |
| enhanced | smc/memory_followup.py | 同表0/1、n32/80/128、4096粒子、48轮更新；输出 _runs/enhanced | ALLOW_SMC |
| fit | smc/memory_followup.py | 原36组拟合诊断；输出 _runs/enhanced/fit_results | ALLOW_TRAINING |
| calibration | smc/run_calibration.py | 自动生成五规则完整真值表；输出 _runs/calibration | ALLOW_SMC |
| alignment | alignment/run_alignment.py | 重新产生12父池与24桥；输出 _runs/alignment | ALLOW_SMC |
| minimax | minimax/analyze.py | 显式 CATALOGUE_ROOT；输出 _runs/minimax | ALLOW_FULL_ENUMERATION |

预算、dtype、参数化和优化器沿用原科学代码；RUN_HOURS 是调度上限，不是某个池保证完成的承诺。agreement 和 loss 的原始节点、每组步数/时间上限保持不变。协议变更后的结果只能称新复现，不能混入已发布的384主池。base_two 从先验重新生成全部128池；它没有冒充原本地40池加云端88池的原样接续。cloud_memory_remaining 的原身份检查和“不跨硬件接半池”约束保留为来源记录，运行入口不依赖旧 LOCAL_IMPORT 或 MANIFEST。

## 数据与科学核心

data.py 按原生成顺序再生四份 8-bit 随机表：NumPy default_rng(202609112301 + 1000003 * dataset_id)，先 permutation，再 uint8 标签。输入顺序是从高 bit 到低 bit，float32 的0/1坐标。evidence/datasets.json 保存四份完整数组与32个训练前缀的原 hash；未保存重复随机数组。

core/frozen_memory.py 保留分块 pCN、重采样、阈值推进、Gaussian 参数化、float32 前向及 float64 log 质量累计。只有直接执行保护改变，核心函数不变。smc/memory_followup.py 同时供只读几何诊断和原拟合诊断使用；core/memory_geometry.py 只是复用导入，避免重复实现。alignment/alignment_bridge.py 保留原始科学实现。

其余移植只更换路径、数据载入方式和显式入口，不改目标事件、优化器、学习率、节点、随机种子、模型布局或统计计算。structure/audit.py 的可选 loss 配对缺失时跳过，并在输出标注缺失，不显示假零相关。SOURCE_MANIFEST.json 记录来源文件 SHA256、公开文件 SHA256、核心函数 AST 比较及历史文件 hash。config.py 是唯一允许显式修改的配置文件：校验会报告它的新旧 hash；其他已发布文件改变仍会报错，实际计算配置另由采样核心记录。

主要来源模块按主题选自 memory_depth_night_20261026、cloud_followup_memory_depth_20261025、cloud_remaining_suite_20261025、memory_loss_heterogeneity_20261026、memory_dataset_shortcuts_20261026、memory_minimax_20261026。五规则脚本取自 cloud_remaining_suite_20261025 冻结包中的 calibration/run_calibration.py。两层续算的来源与身份约束对应 cloud_memory_remaining_20261026。它们是来源标识，不是运行时路径依赖。

## min-max 的必要外部输入

本包不包含完整旧 SMC 函数目录，也不依赖下载盘 ZIP 或框架外临时目录。默认 CATALOGUE_ROOT=None，缺少输入时明确报错。设置为使用者持有的解包数据根目录后，目录必须包含以下内容：

```text
CATALOGUE_ROOT/
  results_3bit_cross_architecture_complexity_order/
    architectures/
      tanh_w16_d1/
      tanh_w32_d1/
      tanh_w16_d2/
      relu_w16_d1/
  results_4bit_full_curve_scan/
  results_4bit_full_curve_scan_part2/
  results_4bit_full_curve_scan_part3/
```

每个3-bit架构目录需有 all_256_neural_k_curves.npz、representative_definitions.csv、config.json、targets/<name>/replica_volume_curve.csv。NPZ 必须含 function_ids、representative_function_ids、thresholds、neural_k_bits；CSV 的必要列为 function_id、name，以及 threshold、replica、neural_k_bits。完整函数覆盖为256、对称轨道为40、副本为8，节点数组由脚本 THRESHOLDS 明确规定。

每个4-bit分片需有 function_to_orbit.npz（字段 representative）、config.json（config.settings.WIDTH=16）、orbit_status.csv（function_hex、quality）、all_volume_curves.csv（phase、function_hex、threshold、neural_k_bits_median），以及 targets/<0x...>/first/replica_volume_curve.csv（threshold、replica、neural_k_bits）。只使用 first-pass，不按结果挑选复核值；三分片不得重复覆盖同一轨道。联合覆盖65536函数、1992轨道、8副本，保留原始缺失节点与质量警示。

该输入属于既有完整函数目录的外部依赖。本目录没有宣称 E33 的任意默认配置都会自动生成完全相同的3-bit/4-bit分片，也没有把已公开的小表冒充这些必需原始目录。需要获得或按对应公开完整函数扫描协议重建上述表后，才能跑完整 min-max 分析；本次只核验小规模动态规划，不重建旧目录、不启动 SMC。

minimax/analyze.py 使用解包目录适配器，不读取 ZIP。原动态规划、见证、缺失上下界、先求斜率再 min-max 与先取 K 下包络再求斜率的区别均保留。4-bit分析会构造超过4300万个状态，占用可观内存；完整执行须显式 ALLOW_FULL_ENUMERATION=True。

## 轻量证据与核验边界

evidence 仅保存主矩阵、逐池主系数、增强窗口、五规则逐池结果、全部幅度管读数、agreement 各档汇总、loss 各节点汇总、结构统计、字典结果和 min-max 主要曲线。没有完整权重、大日志、全部粒子、完整原始输出或私人来源链接。

review.py 可直接从96个条件的384个逐池系数恢复主矩阵，并计算带截距/过原点拟合及质量合并值。plot_evidence.py 只从这些小表生成四张英文轴图，写入框架 assets/figures/e32_ext_*.png。图像由中英文共用。

原始审计已经完成的 level 账本和 float64 快照核对数属于历史证据；没有附带那些原始快照，因此不能说本次发布检查又独立完成了一遍完整审计。重新训练后才会有 agreement 的逐模型预测及 loss 的参数/观察文件。完整目录 min-max 本次也未执行。

检查记录见 TEST_RESULTS.json；中英文块、公式、数值、表格与图片对应由 check_report_pair.py 复核。网站构建、目录索引和把中英补充追加到原页面由主集成流程负责。

