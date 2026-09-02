# Neural K 理论框架与证据包

本目录汇总 Neural K 理论框架、正式短论文、研究主文、证据总账、E01--E30 实验说明与复现脚本，以及可直接浏览的中英文静态网站。

## 快速入口

- **当前最准确的核心主张：**[理论核心](theory_core_zh.md)
- **正式短论文：**[中文 Markdown](papers/compression_is_intelligence_zh.md) / [English Markdown](papers/compression_is_intelligence_en.md) / [Zenodo PDF](https://doi.org/10.5281/zenodo.22255552)
- **统计语言逐层教程：**[从参数测度到函数系综](statistical_language_tutorial_zh.md)
- **按研究转折展开的说明：**[研究主文](short_paper_zh.md)
- **早期短论文工作稿：**[Loss-Resolved 函数选择](short_paper_draft_zh.md)
- **短论文主张与证据材料：**[核心主张、实验链与证据边界](short_paper_claim_evidence_materials_zh.md)
- **短论文写作规划：**[叙事结构、图表、引用与工期](short_paper_writing_blueprint_zh.md)
- **完整实验与理论审计：**[研究证据总账](evidence_ledger_zh.md)
- **E01--E30 实验索引：**[实验与复现材料](experiments/README.md)
- **引用文献：**[参考文献](references_zh.md)
- **网页入口：**[中文网站](site/index.html) / [英文网站](site/en/index.html)

英文源文档分别为 [Formal Research Note](papers/compression_is_intelligence_en.md)、[Core Theory](theory_core_en.md)、[Research Narrative](short_paper_en.md)、[Evidence Ledger](evidence_ledger_en.md) 和 [References](references_en.md)。

## 写作与作者参与说明

研究问题、理论判断、实验设计和实验工作来自作者。当前网页及配套说明的大部分具体文字由 AI 根据对话、脚本、结果包和既有文档整理。

**理论核心与正式短论文是文字写作例外：作者对理论核心进行了逐段审阅和深度修改，并对正式短论文的中文正文进行了逐段重写和定稿；英文版在中文定稿基础上翻译。除这两项外，作者没有参与本目录其他页面的具体文字写作。**

这些材料是可核查的研究记录与阶段性理论总纲，不应被视为已经完成同行评审的正式论文。主张应与实验脚本、证据总账和边界说明一起阅读。

## 目录结构

```text
neural_k_framework/
├── theory_core_zh.md       # 中文理论核心
├── statistical_language_tutorial_zh.md # 统计、信息论与统计物理逐层教程
├── short_paper_zh.md       # 中文研究主文
├── papers/                 # 正式短论文中英文 Markdown 与 DOI 入口
├── evidence_ledger_zh.md   # 中文证据总账
├── references_zh.md        # 中文参考文献
├── experiments/            # E01--E30 文档与复现脚本
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

建议把本目录整体提交到仓库。已生成网站一并保留，方便下载仓库后直接浏览；体积较大的本地结果 zip 由本目录 `.gitignore` 排除，核心数值已经写入实验说明与证据总账，实验也可以由脚本重新运行。
