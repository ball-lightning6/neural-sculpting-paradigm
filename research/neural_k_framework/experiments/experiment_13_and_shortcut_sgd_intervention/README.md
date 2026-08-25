# E13：AND shortcut 的 SGD 运输与单样本干预

## 目的

E13 检验 E12 的静态函数换序是否会在真实训练中出现，并用相同初始化、只替换一个训练样本的配对干预判断数据几何能否因果改写长期函数分布。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_and_shortcut_sgd_dynamics.py`](experiment_and_shortcut_sgd_dynamics.py)

运行：

```bash
python experiment_and_shortcut_sgd_dynamics.py
```

发布副本默认同时训练原始缺口与最小平衡条件，各2,048个 paired seeds 到50,000步。

## 冻结来源

| 文件 | 来源 | SHA256 |
|---|---|---|
| `experiment_and_shortcut_sgd_dynamics.py` | 开发版合并条件并冻结50k预算后的副本 | `c65c4c075a2af7eb203c73d10b578bfd5ffcb578eedb9ff63c50d6b3ac6ab73f` |

本地结果：

| 结果包 | SHA256 |
|---|---|
| `results_and_shortcut_sgd_dynamics_package.zip` | `7e46ed6a90786399df388c7c83129ad2d0465f3cf0972097ee6e53b34e84f4eb` |
| `results_and_shortcut_sgd_long_package.zip` | `29e98821174ec378ddf2c69e74bf35750ae43227f55331afe32bf6746d5687a1` |
| `results_and_shortcut_sgd_balanced_n10_long_package.zip` | `2f3d0929938f3d844ccea2c3c7bf11e58cfd081650e2012d12f709a48a8432ea` |

原始 ZIP 位于`E:\Downloads`，包含大模型 checkpoint，故不进入发布包。
