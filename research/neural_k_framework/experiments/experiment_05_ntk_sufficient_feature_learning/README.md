# E05：固定 NTK 足够时的特征学习

## 目的

该实验检验一个容易被混淆的命题：若匹配架构的初始化固定 NTK 已经足以学习一层 Rule 110，有限宽网络进行端到端训练时是否仍会改变经验 NTK、隐藏表示和 ReLU 门控。

实验并不试图证明特征学习是泛化的必要条件。恰恰相反，它把两件事拆开：固定 kernel 可以泛化，不等于真实网络必然停留在 lazy regime；发生特征学习，也不自动等于获得额外泛化收益。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`rule110_ntk_sufficient_feature_learning.py`](rule110_ntk_sufficient_feature_learning.py)

运行：

```bash
python rule110_ntk_sufficient_feature_learning.py
```

默认结果写入：

```text
/root/results_rule110_ntk_sufficient_feature_learning_v1/
```

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `rule110_ntk_sufficient_feature_learning.py` | `research/ntk_feature_learning/rule110_ntk_sufficient_feature_learning.py` | `163a13ca0ed1ba5337141cd78603ff3d1b3ce825372ead780e27183693708480` |

本地原始结果包：

```text
E:\Downloads\results_rule110_ntk_sufficient_feature_learning_v1.zip
```

SHA256：

```text
df6b29a4c0d22d77a64877c6d7bf608e6d47c16ef8f984a93e4836857805425d
```

原始 ZIP 约49 MB 且可在消费级 GPU 上快速重建，不进入最终发布包。
