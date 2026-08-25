# E20：Full-rule、固定 D 与 hard-margin 测度闭环

## 目的

E20 检验完整规则 self-loss 体积与固定部分训练集候选函数质量之间的严格关系，避免把不同测量对象凭直觉互换。它由 constant leave-one-out、AND 四候选 full-rule 和 hard-cell margin bridge 三步组成。

## 运行顺序

```bash
python experiment_constant_leave_one_out_smc_consistency.py
python experiment_and_n10_candidate_full_rule_smc.py
python experiment_and_n10_hard_margin_bridge_smc.py
```

- [实验动机与恒等式](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与裁决](RESULTS_AND_CONCLUSION.md)

## 脚本 SHA256

```text
0f89290cf946abdc1d0a11503951479bac5006f12d0f4a1f2a201228d4eb1201  constant LOO
d9df0de87a02a62266d715c3265e3977a2137037917929ae13458eb2841dfb34  AND full-rule
836a9ea573cafebd400700fbfe0203dbbd9314921c474a2ae7a3ef7e27dbe76a  hard-margin bridge
```

关键 ZIP SHA256：

```text
1d8da6adf54480e717c72f1b1399d2d5ea6ef3d7e9edaafb9c211a1c73f3dd2  constant LOO
a29339e94df29e5501e6f9624daa349454d7aad1b8010bba4b6cc61559549ec0  AND full-rule
3c244b05801e1e2e1893fc6bc8b111a536647b80f850d0bf2072c4de7ae5558f  margin bridge
```

哈希不区分大小写。原始 ZIP 不进入发布包。
