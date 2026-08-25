# E08：共享中间表示与有限容量压力

## 目的

E08 检验一个唯象机制：当两个输出任务具有昂贵的共同 CA 前缀计算时，允许网络共享隐藏表示，是否比强制两条独立分支更容易达到同一极低 loss。

实验采用四格交叉设计，使完全相同的四个原子子任务各自在 Shared 与 Separate 组出现一次；另设参数量匹配的 forced-split 架构作为阴性对照，排除标签边际或原子任务固有难度。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_ca_shared_intermediate_pressure.py`](experiment_ca_shared_intermediate_pressure.py)

运行：

```bash
python experiment_ca_shared_intermediate_pressure.py
```

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_ca_shared_intermediate_pressure.py` | `research/computational_pressure/experiment_ca_shared_intermediate_pressure.py` | `ec3b985274222d39dcb62c9dcbf8c115b079e76396a4a5e5618aa2729d52f7ab` |

本地结果包：

```text
E:\Downloads\results_ca_shared_intermediate_pressure.zip
```

SHA256：

```text
97ec4fd2d6342ed62c682109a52cd4c462fb6ff358c8edfdbee1704303b1124f
```

哈希不区分大小写。ZIP 可快速重建，不进入最终发布包。
