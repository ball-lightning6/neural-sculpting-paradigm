# Neural K 理论框架与证据包

本目录汇总 Neural K 理论框架、研究主文、证据总账、E01--E25 实验说明与复现脚本，以及可直接浏览的中英文静态网站。

## 快速入口

- **当前最准确的核心主张：**[理论核心](theory_core_zh.md)
- **按研究转折展开的说明：**[研究主文](short_paper_zh.md)
- **完整实验与理论审计：**[研究证据总账](evidence_ledger_zh.md)
- **E01--E25 实验索引：**[实验与复现材料](experiments/README.md)
- **引用文献：**[参考文献](references_zh.md)
- **网页入口：**[中文网站](site/index.html) / [英文网站](site/en/index.html)

英文源文档分别为 [Core Theory](theory_core_en.md)、[Research Narrative](short_paper_en.md)、[Evidence Ledger](evidence_ledger_en.md) 和 [References](references_en.md)。

## 写作与作者参与说明

研究问题、理论判断、实验设计和实验工作来自作者。当前网页及配套说明的大部分具体文字由 AI 根据对话、脚本、结果包和既有文档整理。

**理论核心是唯一的文字写作例外：作者对其进行了逐段审阅和深度修改，因此它是当前对核心研究主张相对更准确的表述。除理论核心外，作者没有参与本目录其他页面的具体文字写作。**

这些材料是可核查的研究记录与阶段性理论总纲，不应被视为已经完成同行评审的正式论文。主张应与实验脚本、证据总账和边界说明一起阅读。

## 目录结构

```text
neural_k_framework/
├── theory_core_zh.md       # 中文理论核心
├── short_paper_zh.md       # 中文研究主文
├── evidence_ledger_zh.md   # 中文证据总账
├── references_zh.md        # 中文参考文献
├── experiments/            # E01--E25 文档与复现脚本
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
