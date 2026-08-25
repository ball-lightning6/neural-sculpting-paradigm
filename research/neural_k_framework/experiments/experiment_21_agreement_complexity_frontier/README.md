# E21：Agreement 控制与共识复杂度前沿

## 目的

E21 把 agreement 从被动统计量变为训练集干预工具：反复查询当前 committee 最分歧的未见输入，分别尝试两个标签，选择让函数分布收缩更快或更慢的分支。随后用长 anti-consensus 前缀逐层排除早期简单吸引子，再切换 pro-consensus 使分布重新凝聚，测量稳定终点的符号复杂度怎样变化。

## 文件与运行顺序

```bash
python experiment_adversarial_disagreement_completion_pilot.py
python experiment_consensus_complexity_frontier.py
```

- [实验动机与判据](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

## SHA256

```text
3cbe4dbeeb24aa45215d3710ec85419e117d2317f28aa2316371b855683bf9fe  adversarial pilot script
1a0d1c21c6df3396bf035852ce361eabf29c4709abfa74d35830a3d241a71a82  frontier script
91cb0db9c2d595cb205785af6624f18fc0dbb60abff370db424eee08e276878e  pilot ZIP
8b917a33aa9b2aaf1606e0808e964d2fb4ec6689049353be4de556509506aa87  frontier ZIP
```

原始 ZIP 位于`E:\Downloads`，不进入发布包。
