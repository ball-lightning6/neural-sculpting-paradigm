# 2026-08-31选择性实验更新

本版本在既有E01--E25框架包基础上，只新增四个已经完成定位和结果审计的实验
单元，不收录同期所有探索。

## 新增实验

| ID | 实验 | 主要结论 |
|---|---|---|
| E26 | MNIST平衡无标签划分 | 自然0/1划分在两个盲测panel中随loss从末位升至top1 |
| E27 | Grokking前后的Agreement | hard fit时未见Agreement仍接近随机，随后主要随正确规则恢复而上升 |
| E28 | MNIST 50k整网HMC与Adam | HMC 99.03%、plain Adam 98.95%，达到同一预测水平 |
| E29 | Dead bit loss profile | 深loss静态质量在dead方向保持prior时仍定量预测无衰减Adam |

## 选择边界

- E26只收录5个0、5个1及5:5比例约束的版本；
- E27作为一项Mod97轨迹补充分析收录；
- E28只报告50k HMC与plain/MAP Adam，不纳入8k kernel比较；
- E29收录SMC、Adam、L2、NNGP及温度1posterior的同一机制实验；
- 不收录其他8月30--31日的3-bit、8-bit、Rule110、编辑距离、采样benchmark、
  失败pilot或未完成分支。

每个实验目录只含冻结脚本、动机/协议、结果与边界说明。原始结果ZIP和checkpoint
不进入发布包；其本地文件名与SHA256记录在相应README中。

## 网页

中英文网页已从E01--E25扩展为E01--E29。四个新实验进入：

- 实验索引与独立详情页；
- 研究主文的相关论证段落与证据地图；
- 理论核心中的未见标签预测、grokking/agreement、静态/optimizer分层和dead
  direction段落；
- 中英文证据总账。

网站由build_site.py重建，并由validate_site.py验证全部68个HTML页面、站内
链接、锚点、资源和实验卡片。

