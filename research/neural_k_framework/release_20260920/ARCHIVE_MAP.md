# 原始材料归档图

发布树是干净的证据层，不是原始结果仓库。以下目录保留在工作区，未复制到发布树。

| 原始位置 | 处理 | 原因 |
|---|---|---|
| `research/addition8_static_prediction_20260914` | 只复制 README、N2048报告、脚本和图 | 包含粒子 checkpoint和服务器临时目录 |
| `research/architecture_bias_theory_20260915` | 只复制 README、报告和必要方法说明 | 大量SMC快照、checkpoint和逐节点原始表 |
| `research/peleg_static_recheck_20260919` | 只复制结论、协议、分析报告和数值审计 | upstream副本与原始结果过大 |
| `research/early_rule_learning_replay_20260914` | 只复制审阅、协议、环境和试跑说明 | 237份代码、测试临时目录和封存包应单独归档 |
| `research/function_information_conservation` | 只复制本发布涉及的脚本/报告 | 该目录是历史结果汇总，含大量重复 smoke 包 |
| `research/ca_phase_transition` | 只复制 E32 的脚本和 Mod3 可达性说明 | 其余任务包含旧实验、checkpoint和失败试跑 |
| `research/rule_ood_generalization` | 复制规则 OOD 的说明和源脚本 | 结果包/训练断点若存在，仍需按独立协议归档 |
| `kimi_experiments` | 不纳入本次发布树，原始脚本保留在工作区 | 输入级/concat输出级互信息实验尚无稳定结论；小任务和网络差异可能产生协议偏差 |
| `scripts` | 只复制 GPU/运行辅助脚本 | 其余下载和临时工具不是科学证据 |
| `research/theory_gap_literature_100` | 只复制综合报告和数学基线 | PDF、网页抓取和审计数据库不作为正文附件 |
| `chatbox_*`、`chatbox_recovered_*` | 不上传 | 本地聊天恢复缓存，不是研究证据 |

如果之后需要提供可完全复算的附加下载包，应按专题分别打包，并在包内加入源码、配置、数据划分、环境 JSON、硬件、随机种子、结果摘要和 SHA256。不要把当前工作区的所有 zip 直接上传。
