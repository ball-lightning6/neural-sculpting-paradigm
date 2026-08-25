# E16：Parity 终点偏好、全局入口与局部恢复

## 目的

E16 通过 leave-one-out、随机半空间、错误点揭示和 prefix scaffold 四层干预，区分：

1. parity 是否存在并被 endpoint loss 支持；
2. 随机初始化能否进入其低-loss 区域；
3. 已到达的精确解撤去辅助信息后是否稳定；
4. 强扰动破坏 hard function 后能否仅靠 endpoint loss 恢复。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_parity_leave_one_out_dimension_scan.py`](experiment_parity_leave_one_out_dimension_scan.py)
- [`experiment_parity12_half_space_generalization.py`](experiment_parity12_half_space_generalization.py)
- [`experiment_parity16_scaffold_perturb_recovery.py`](experiment_parity16_scaffold_perturb_recovery.py)

运行顺序：

```bash
python experiment_parity_leave_one_out_dimension_scan.py
python experiment_parity12_half_space_generalization.py
python experiment_parity16_scaffold_perturb_recovery.py
```

第二个脚本当前冻结配置为14-bit 半空间加错误揭示；文件名保留其最初12-bit 来源，以维持开发版追踪。

## 冻结脚本 SHA256

```text
4c9498cd825fbf887bc1ffb7ebd035a778d07b8cd139fb9aeb1dfac87a2aa6f2  leave-one-out
5612bc8135377c96b431ca7835c428605080e5f55e01ca31836fc4690fd6f282  half-space/reveal
abb131d68780c441eac692a34720309d5f14554414f67862f7884d31909b0424  scaffold/recovery
```

## 关键结果包 SHA256

```text
41099ca001b64cd4195f16639c1056de0098d6312b23a04d7a3000b7f36adccd  leave-one-out
b503bf464212c59db2bbb14b10d8e5a5d13ff1f48a8ebe5c2b870f6af56dd882  parity14 half-space
675716cafaa260f7a7ed89de2e6693220b54edd6c6facb3cd1edb07c246cd900  error reveal
448072a1678dcdfE28e0d2622dd1070fe2ffee375bf45e0da840d2cb602a3b21  parity16 w256
```

哈希不区分大小写。ZIP 位于`E:\Downloads`，不进入发布包。
