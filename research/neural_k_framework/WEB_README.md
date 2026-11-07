# Neural K 静态网页

## 直接阅读

双击打开：

```text
site/index.html
```

页面组成：

- `site/index.html`：研究主文；
- `site/theory-core.html`：只保留最终命题与必要公式的独立理论核心；
- `site/experiments/index.html`：E01–E38 实验索引；
- `site/experiments/e01.html` 至 `e38.html`：实验详情；
- `site/references.html`：明确引用、完整书目信息与论文原文入口；
- `site/evidence-ledger.html`：完整证据总账。
- `site/post-upload-*.html`：严格概率语言、MDL、架构估计和增量材料；
- `site/width-double-descent.html`：容量轴、时间轴双下降解释与E38证据；
- `site/complexity-methods.html`：给定规则与训练快照的两种复杂度测量方案，尚未新增验证实验；
- `site/ait-connections.html`：神经网络泛化与算法信息论主线，连接复杂度、共享结构、信息增益和数据量相变；
- `site/research-updates.html`：2026-10-27之后的工作总览，连接五组新证据报告；
- `site/interpretation-update.html`：测度、末层缩放、幅度正则、复杂度与预测之间的概念修订；
- `site/update-*.html`：几何、辅助监督、共识、训练控制与相关工作五个专题；
- `site/en/`：对应英文版本；当前共132个中英文HTML页面，每页可切换语言并保持当前页面。

这两份10月理论增补仅接入网站，不加入正式论文或PDF。E38单列在实验索引中；MNIST探索性对照没有被标成已完成的严格文献复现。

2026-10-27更新沿用原编号：E36追加CV/SGD/Adam/STP/G&C对照，E31追加f/g幅度干预，E33追加独立复杂度校准，E37追加固定半径测度，E32追加跨层记忆与诊断。各页保留旧正文并提供`#update-20261027`直达链接；实验索引仍为38项。理论核心新增六个开放问题，主文、增量总览与证据总账同步中英文摘要。新增英文须逐段对应中文，不能只保留摘要。

2026-10-30的AIT主线页以`ait_connections_zh.md`和`ait_connections_en.md`为源，从研究主文、理论核心、增量总览和参考文献进入。两种语言的段落、标题、表格数值和公式纳入校验；该页不并入正式论文，也不新增实验编号。

2026-11-07整理截至11月6日的工作，按作者最新范围保留7对新增页面，回接已有实验编号和主干理论。新增源稿为两个根目录主题稿与`supplementary/updates_20261106/`内五组报告；构建器只复制各组顶层白名单小文件与双语README，生成稳定元数据的ZIP下载包，不递归复制缓存、checkpoint或私有原始结果。相关实验页带`#update-20261106`入口。尚未实施的新设想、近期含噪时间双下降及5bit条件SMC探索暂不公开，原始记录保留本地；入选证据的必要边界仍保留。本轮不更改正式论文或恢复实验。

正文、图片、导航和实验搜索均为本地静态资源。公式使用 MathJax CDN，首次渲染公式时需要联网。

## 重新生成

```bash
pip install -r web_requirements.txt
python format_evidence_ledger.py
python build_site.py
python validate_site.py
```

`build_site.py`以 Markdown 文档为唯一内容源，重新构建 `site/`。不要直接编辑生成的 HTML；内容修改应写回 `short_paper_zh.md` 或对应实验目录中的 Markdown。

`format_evidence_ledger.py`只在完整句子之间增加空行，并在写入前验证去除空白后的全部字符与原文一致；它不会改写总账内容。

`validate_site.py`检查：

- 当前中英文 HTML 页面是否齐全；
- 本地链接、图片、脚本和页内锚点是否存在；
- 实验索引是否包含 38 张卡片；
- 主文是否链接 E01–E38，并包含关键结果图；
- 历史 E01–E30 页是否包含概览、动机和结果分区，E31–E38 是否有完整报告；
- 中英文标题编号、表格、图片、公式和代码块结构是否对应。
- 正式短论文中英文源稿是否有整段缺失、公式不对应，以及行内LaTeX分隔符是否被正确保护。
- 本轮五份增补源稿的中英文段落、标题、表格数值、公式、代码块和图片是否对应，以及旧实验页是否确实包含更新锚点。
- AIT主线双语源稿是否逐段结构对应，三个主要入口是否连到对应语言的新页，以及公开Markdown源稿是否已生成。
- 11月7对主题源稿与页面是否对应、主题资产是否可访问、manifest是否保留38个实验编号，以及相关旧实验是否接入新证据。
