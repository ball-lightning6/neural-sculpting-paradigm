# E03：函数系综动力学与网络参考语言

## 实验组成

- **E03a**：从 ordinary 与 prior-consistent 初始化出发，跟踪1/2/4样本约束下的完整 Boolean 函数分布到 post-fit 5,000步；
- **E03b**：用 Oxford 同族`tanh`网络扫描 width 16/32/64/128，检验 E03a 的定性现象是否依赖1024×3 GELU+LayerNorm 大网络。

## 文件

- [动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`E03a`脚本](experiment_boolean_function_posterior_dynamics.py)
- [`E03b`脚本](experiment_boolean_posterior_width_scan.py)

运行：

```bash
python experiment_boolean_function_posterior_dynamics.py
python experiment_boolean_posterior_width_scan.py
```

E03b 若要自动加载1024参考结果，应把 E03a 结果解压到脚本配置的`REFERENCE_RESULT_DIR`；否则它仍会完成16--128同族扫描，跨架构比较可离线做。

## 冻结来源与 SHA256

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_boolean_function_posterior_dynamics.py` | `research/function_information_conservation/experiment_boolean_function_posterior_dynamics.py` | `05090cf48e83c097178b76f5c591dc8c25d6f018e37c93ac8db36adf53889942` |
| `experiment_boolean_posterior_width_scan.py` | `research/function_information_conservation/experiment_boolean_posterior_width_scan.py` | `f76facc59caa017af58346ae43d675ea0e598bd46c0b1fda017fed0a48c9c7fa` |

本地结果缓存：

- `results/results_boolean_function_posterior_dynamics.zip`，SHA256`7e015ccd6c5951b06cea727c92a03fba2052e4ecc14f03ff596170e629930404`；
- `results/results_boolean_posterior_tanh_width_scan.zip`，SHA256`999ecfe84e6a0ec7d0195bf02ecbeed854ba17d026479c4e47f1c006e33d7a88`。

ZIP 只作本地核验，不进入最终发布包。
