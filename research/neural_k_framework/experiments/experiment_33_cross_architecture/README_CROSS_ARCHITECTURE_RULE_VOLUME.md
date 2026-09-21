# 跨架构bit规则体积曲线

**9月12日运行中修订：最大BCE路线出现包含关系报警，先暂停。**
已有运行请Ctrl+C等待保存结束，然后使用[兼容续跑版](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/experiment_cross_architecture_mean_mse_resume.py)。
它只继续平均BCE与native-MSE，科学配置/目录不变，不删除或重跑已有结果。
详见[报警诊断](E:/code/clean_github/neural-sculpting-paradigm/research/notes/20260912_cross_architecture_max_bce_sampling_alarm_zh.md)。
下文保留最初全矩阵方案，最大BCE的正式测量当前不再推荐继续。

直接运行[独立脚本](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/experiment_cross_architecture_rule_volume.py)，或将全文放入一个Jupyter单元格。沿用已经测速的5090环境，不下载预训练权重。只需上传这一个`.py`，不需要builder或runner片段。

这是新的正式运行目录，旧`experiment_cross_architecture_output_volume.py`测速版及其结果不变。

## 默认实验

| 项目 | 设置 |
|---|---|
| 架构 | MLP、RNN、LSTM、Transformer、ViT、ConvNeXt、Swin、U-Net、改进U-Net、Swin-Unet |
| 模型大小 | 沿用已测compact模型，9800至145250参数 |
| 规则 | 8位周期元胞自动机演化一步：Rule204恒等映射、Rule90邻位XOR |
| 训练约束输入 | 固定随机排列的前16个8bit输入，两个规则/所有架构完全相同 |
| 符号编码 | 输入0/1，输出8bit；Transformer每bit一个token |
| 图像输入 | 2×4黑白逻辑区域，按位序逐行排列，最近邻放大至32×32，RGB三通道相同；无灰色插值、无网格线、无样本ID水印 |
| 图像输出 | 三种U-Net类模型原生输出32×32图像；其余模型输出8个数 |
| 观测 | 所有模型测格点平均BCE及格点最大BCE；三种图像输出模型另测整图native-MSE |
| SMC | 四个独立副本，各128粒子；所有可学习参数参与，不固定隐藏层 |
| 目标 | BCE到0.001；native-MSE到0.0001 |
| 默认预算 | 每副本每次180秒；一次采样循环最多8小时，保存/打包开销另计 |
| 队列 | 先让各架构都获得第一个副本，再逐轮补齐；不把一个架构无限跑到底 |

格点平均BCE为16×8个元素的平均。最大BCE严格对应这些格点的最小有符号margin下限，**不是全部样本等幅的薄切片**。

普通U-Net沿用原训练文件的raw输出MSE；改进U-Net和Swin-Unet使用sigmoid后MSE。native-MSE约束所有像素，不能与只约束8格点的BCE绝对K混为同一个对象。
图像模型的“完整bit映射”指全部输入上8个指定格点的解码，不是整张图像1024像素的硬映射；整图MSE和像素准确率单独记录。

默认16个输入是fixed-D测量，不是全规则Neural K。每个loss节点额外在全部256个输入上观察输出，但这些额外标签**不进入SMC目标、不用于停机或选择粒子**。
需要全规则约束时，把`POINTS=256`并另设`RUN_TAG`；不能把16输入的测速预算直接沿用到256输入。

## 输出

默认目录：

```text
/root/autodl-tmp/results_cross_architecture_rule_volume_compact_curves_v1
```

- `task_examples.png`：黑白输入及规则输出示例。
- `protocol.json`、`invocation_*.json`、`sources/`：科学配置、环境与源码记录，支持无`__file__`的Jupyter。
- `training_data_rule*.npz`、`full_truth_table_rule*.npz`：原始bit、图像、标签和输入编号。
- `models/*/parameter_prior.json`：完整参数形状及Gaussian参考测度；不声称等于各模型默认训练初始化。
- `models/*/preflight.json`：2048个先验网络的真实统计及浅层直接计数。
- `jobs/*/rule*/*/r*/checkpoint.pt`：全部粒子、RNG、K和进度，始终保留。
- `points.json`、`levels.json`：固定loss节点与全部自适应层的体积/耗时/接受率。
- `node_*.npz`：训练格点logits、全部256输入格点logits、完整bit映射编码、少量权重快照和原始输出图像。
- `coverage.csv`、`points.csv`、`slopes.csv`、`prior_calibration.csv`：覆盖、K曲线、收缩系数和浅层校准。
- `plots/`：分架构K曲线、BCE窗口斜率；副本分歧/未齐副本节点以红叉标记。
- `report_zh.md`、`last_run_status.json`：本次状态、错误和解释边界。

斜率先在同一副本内做K差，再取副本中位数，不把两个节点各自中位数直接相减。最大BCE另给出真实margin坐标`a`及`a²`的窗口系数；平均BCE的对应`a`只是坐标换算，不称为所有样本的共同幅度。

## 暂停与续跑

- Ctrl+C一次：完成当前SMC层后保存并打包；第二次强制中断会恢复最后完整检查点，不保存半层。
- 也可在结果目录创建`STOP`文件；再次运行前移除它。
- 超过单副本预算：保存该项并换下一项。下次运行从断点继续，不从先验重新开始。
- 已达到目标的项直接跳过，不自动recheck。
- 已完成后仍保留检查点；把`STOP_BCE`/`STOP_MSE`改为预设网格中更低的值，可接着向深处采。
- 可以增加`SECONDS_PER_REPLICA`、`MAX_RUN_HOURS`，或用`SKIP_ARCHITECTURES`/`REPLICA_IDS`/`JOB_KEYS`调节本次队列。
- 改粒子数、副本总数、网络、输入数、数据种子、更新轮数或软件版本，需另设`RUN_TAG`。不混合科学配置。

默认zip不含`.pt`，所以下载包用于分析，**服务器原目录用于续跑**。全部暂停粒子检查点量级约5.65GB；节点快照和日志另计。zip从几百MB到数GB取决于到达节点数与数据可压缩性。

## 可靠性

这是一套完整自动化的**有限loss初测**，不是将128粒子称为精密深尾真值。

独立副本、直接先验校准、接受率/步长报警、完整记录和工程测试用于发现问题；没有报警不证明充分混合。SMC粒子间存在相关性，包内agreement不是独立SGD种子的agreement。
更高精度应只对选定规则/架构另开更高粒子或更新预算的复核，不在本版自动让全部曲线再跑一遍。

原型与正式版保持同一模型/目标/参考测度。运行器额外避免全存活时无意义的重采样，并让节点判定与实际float32阈值使用同一舍入口径；名义和有效阈值同时记录。
这些工程改动会改变随机轨迹，不将正式SMC当作测速中三层状态的续跑。

## 本地验证

11项工程测试通过，包括Gaussian可积概率、float32阈值边界、无`__file__`的Jupyter加载、
完整输入诊断不改变采样目标/RNG、断点恢复与不中断运行逐位一致、已完成项跳过及续深、强制半层中断恢复。
另外实际走完十种架构、全部适用观测和两个副本的46项浅层短流程，检查了输出形状、绘图和zip不包含完整检查点。
这些是工程测试，不是正式深尾结论；正式矩阵由用户在5090上运行。
