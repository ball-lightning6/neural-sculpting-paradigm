# Neural K 理论框架与证据包

本目录汇总 Neural K 理论框架、正式短论文、研究主文、证据总账、E01--E38 实验说明与复现脚本，以及可直接浏览的中英文静态网站。

## 快速入口

- **当前最准确的核心主张：**[理论核心](theory_core_zh.md)
- **正式短论文：**[中文 Markdown](papers/compression_is_intelligence_zh.md) / [English Markdown](papers/compression_is_intelligence_en.md) / [本地修订 PDF v1.2](papers/compression_is_intelligence_v1.2_en.pdf) / [既有 Zenodo 归档](https://doi.org/10.5281/zenodo.22255552)
- **统计语言逐层教程：**[最小严格统计语言教学版](minimal_strict_statistical_language_tutorial_zh.md)
- **按研究转折展开的说明：**[研究主文](short_paper_zh.md)
- **完整实验与理论审计：**[研究证据总账](evidence_ledger_zh.md)
- **E01--E38 实验索引：**[实验与复现材料](experiments/README.md)
- **引用文献：**[参考文献](references_zh.md)
- **网页入口：**[中文网站](site/index.html) / [英文网站](site/en/index.html)
- **提交后增量发布树：**[2026-09-20整理版](release_20260920/README.md) / [清单](release_20260920/MANIFEST.md)

网站当前包含 E01--E38、严格概率语言、MDL—grokking 长推导、测度鲁棒性复核、架构估计器、双下降解释及两种复杂度测量方案；当前构建为 116 个中英文 HTML 页面，使用 `validate_site.py` 检查链接、资源、双语结构及正式论文源稿对应。

英文源文档分别为 [Formal Research Note](papers/compression_is_intelligence_en.md)、[Core Theory](theory_core_en.md)、[Research Narrative](short_paper_en.md)、[Evidence Ledger](evidence_ledger_en.md) 和 [References](references_en.md)。

## 2026年10月网站增补

2026-10-27的四类补证已合并进原编号：训练方法对应归入[E36](experiments/experiment_36_peleg_g_and_c_validation/VALIDATION_EXTENSION_20261027.md)，复杂度干预/校准归入[E31](experiments/experiment_31_loss_amplitude_grokking/XOR_OR_ALIGNMENT_20261027.md)和[E33](experiments/experiment_33_cross_architecture/COMPLEXITY_CALIBRATION_20261027.md)，测度归入[E37](experiments/experiment_37_measure_robustness/FIXED_RADIUS_20261027.md)，记忆归入[E32](experiments/experiment_32_random_memory/MEMORY_EXTENSION_20261027.md)。原报告保留，实验总数仍为38，中英文完整对应。六个开放问题已加入理论核心；这些更新不改正式论文或PDF，不新增实验。

2026-10-21宽深SMC的95个目标运行全部完成，以E38加入实验索引。双下降解释、完整SMC曲线、跨测度比较边界，以及两种复杂度测量方案已接入中英文网站。本轮这两项新增内容不加入正式论文或PDF，测量方法没有被写成已完成实验，也未因此启动新实验。MNIST训练对照仍缺1个计划条件，仅保留为文献启发的探索记录，不称为严格复现或完整双下降验证。

- **宽度与双下降：**[中文网页](site/width-double-descent.html) / [English page](site/en/width-double-descent.html)；[中文源稿](width_double_descent_zh.md) / [English source](width_double_descent_en.md)。
- **复杂度测量方法：**[中文网页](site/complexity-methods.html) / [English page](site/en/complexity-methods.html)；[中文源稿](complexity_measurement_methods_zh.md) / [English source](complexity_measurement_methods_en.md)。
- **E38复现材料：**[五配置SMC扫描](experiments/experiment_38_width_depth_volume/README.md)。

## 写作与作者参与说明

研究问题、理论判断、实验设计和实验工作来自作者。当前网页及配套说明的大部分具体文字由 AI 根据对话、脚本、结果包和既有文档整理。

**理论核心的既有版本与正式短论文是文字写作例外：作者对理论核心进行了逐段审阅和深度修改，并对正式短论文的中文正文进行了逐段重写和定稿；英文版在中文定稿基础上翻译。2026-10-27理论新增段落由AI根据作者讨论整理，仍待作者审阅；其余页面的具体文字也由AI整理，不应视为作者已经逐句认可。**

这些材料是可核查的研究记录与阶段性理论总纲，不应被视为已经完成同行评审的正式论文。主张应与实验脚本、证据总账和边界说明一起阅读。

## 目录结构

```text
neural_k_framework/
├── theory_core_zh.md       # 中文理论核心
├── minimal_strict_statistical_language_tutorial_zh.md # 当前严格统计语言教程
├── short_paper_zh.md       # 中文研究主文
├── papers/                 # 正式短论文中英文 Markdown 与 DOI 入口
├── evidence_ledger_zh.md   # 中文证据总账
├── references_zh.md        # 中文参考文献
├── experiments/            # E01--E38 文档与复现脚本
├── assets/figures/         # 关键结果图
├── site/                   # 已生成的中英文静态网站
├── build_site.py           # 网站生成器
└── validate_site.py        # 页面、链接与资源校验
```

## 本地浏览与重建

网站不需要服务器，直接打开 `site/index.html` 即可浏览。

如需重新生成：

```powershell
python -m pip install -r web_requirements.txt
python build_site.py
python validate_site.py
```

实验依赖见 `experiments/requirements.txt`。每个实验目录均包含目的、预注册思路、结果解释和独立脚本。

## 上传范围

提交公开源文档、复现脚本、英文对应和已生成网站。保留网页实际使用的图片；可由脚本重建的原始结果、checkpoint和zip不随本轮上传。本地草稿、私有审计、历史副本及未纳入发布的探索目录应留在本地，不应使用全仓批量添加代替范围核对。核心数值保留在实验说明与证据总账中。
