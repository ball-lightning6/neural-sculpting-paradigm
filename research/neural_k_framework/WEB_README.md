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
- `site/en/`：对应英文版本；当前共116个中英文HTML页面，每页可切换语言并保持当前页面。

这两份10月理论增补仅接入网站，不加入正式论文或PDF。E38单列在实验索引中；MNIST探索性对照没有被标成已完成的严格文献复现。

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
