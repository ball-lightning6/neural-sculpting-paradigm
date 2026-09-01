# E30：Weight decay重塑完整静态函数地形

## 目的

说明grokking并不需要weight decay；显式L2之所以能够帮助grokking，是因为它
改变了同一套函数竞争的静态loss地形，而不是引入一种神秘或特殊的学习机制。

## 入口

- [experiment_weight_decay_static_landscape.py](experiment_weight_decay_static_landscape.py)
- [MOTIVATION_AND_PREREGISTRATION.md](MOTIVATION_AND_PREREGISTRATION.md)
- [RESULTS_AND_CONCLUSION.md](RESULTS_AND_CONCLUSION.md)
- [NUMERICAL_TOLERANCE_AUDIT.md](NUMERICAL_TOLERANCE_AUDIT.md)

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| experiment_weight_decay_static_landscape.py | research/function_information_conservation/experiment_8bit_and_gaussian_blind.py | `0EE9E1118EB497997B1557EE49D69A65868CC207B38F8E3379ECA650A1797FC0` |

## 原始结果包

原始结果不进入仓库。三个主包的SHA256为：

```text
D4CB2A6CC1C65399833CB1DEBAD9F9D21CB05078ECF35B89ABE4B925DBB93CEE  lambda=0
39F4814E814A637339D8CF8EADC72E7A2E4B141BD28E5F7F0076A2ECB09C7FFF  lambda=5e-5
178B4A2EB108018E4B37B16DD871AA46DA9FFB83F083AED34406CCE9203255E9  lambda=1e-4
```

更深L2补充包SHA256：

```text
DF0756CA78AB64C83A544298728FCA892AD6ED90A9DB34F815B84F3FCFB6C8F6
```

## 主运行模式

```bash
python experiment_weight_decay_static_landscape.py --mode no_wd_static_matched_bce_n40
python experiment_weight_decay_static_landscape.py --mode l2_static_half_lambda_n40
python experiment_weight_decay_static_landscape.py --mode l2_static_reliable_n40
```

主结果只使用这三个近似matched-BCE条件。`lambda=2e-4`校准和
`J<=0.0186`深层结果只承担边界说明。
