# E04：在 Mingard 2025协议中延长 post-fit 观察

## 目的

[Mingard 等（2025），*Deep Neural Networks Have an Inbuilt Occam's Razor*](https://www.nature.com/articles/s41467-024-54813-x)中的 Boolean 实验在模型首次达到零训练分类错误时停止，并据此比较初始化函数先验、复杂度分布与训练终点。E04 保持其核心网络、目标、初始化尺度与 advSGD 设置，只把观察窗口延长到 post-fit 100/1,000步，并保存每条轨迹的完整128-bit 函数。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_mingard_2025_postfit_drift.py`](experiment_mingard_2025_postfit_drift.py)

运行：

```bash
python experiment_mingard_2025_postfit_drift.py
```

默认`PROFILE="pilot"`，输出到：

```text
results_mingard_2025_postfit_drift/pilot/
```

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_mingard_2025_postfit_drift.py` | `research/function_information_conservation/experiment_mingard_2025_postfit_drift.py` | `dd3eb8706c2008f6ab6f1c8df51419687016eb3dcb6b60787c18b5781c60d5f8` |

本地结果缓存`results/pilot.zip`，SHA256`72e00af9e6cd91765867afec2e09a55724b4cc9f40f780c5e6f04be40f04a91d`。ZIP 不进入最终发布包。
