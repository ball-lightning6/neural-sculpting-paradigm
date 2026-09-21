# E36 Peleg/G&C 独立验证：raw-BCE 静态复核

首轮已完成。先读[结论](CONCLUSION.md)或[完整表格](analysis/REPORT.md)。不是全论文复现，只有一个16样本训练集。

## 运行

`experiment.py`集中实现训练与SMC，配置在顶部，无argparse。需要PyTorch GPU版、torchvision、NumPy；分析另需Matplotlib。预检还依赖同目录`upstream`中的作者数据类，因此本轮是完整目录可运行，不是已经制作好的单文件迁移包。默认物理GPU1 RTX3080，所有数学和采样设置见[协议](PROTOCOL.md)。

在本仓库根目录运行：

```powershell
& '.venv-gpu/Scripts/python.exe' -E -s -X utf8 release_20260920/experiments/E36_peleg_g_and_c_validation/experiment.py
```

已完成条件会跳过；配置或主源码变化时拒绝覆盖原结果，请指定新目录。数据沿用本地MNIST缓存。跨机器重跑需要修改设备和数据路径并使用新结果目录，不承诺直接续用本地metadata；当前没有制作AutoDL迁移包。

`endpoint_check.py`使用原120秒预算中剩余部分做最多512轮固定核更新，不重新估计K，不把快照当独立副本。`analyze.py`只在CPU离线读取结果，使用独立NumPy float64前向复核。

`results/`保留数据ID、初始权重、batch排列、每epoch训练输出、预定节点测试输出、SGD断点和完整终点权重、SMC断点与抽存审计权重。`upstream/`为冻结的作者代码，只用于核对；没有运行其批量训练入口。

G&C/SMC图示的平均准确率是单模型读数，不是集成。尚未完成高覆盖的BCE匹配及独立训练集确认。
