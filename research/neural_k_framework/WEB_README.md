# Neural K 静态网页

## 直接阅读

双击打开：

```text
site/index.html
```

页面组成：

- `site/index.html`：研究主文；
- `site/theory-core.html`：只保留最终命题与必要公式的独立理论核心；
- `site/experiments/index.html`：E01–E29 实验索引；
- `site/experiments/e01.html` 至 `e25.html`：实验详情；
- `site/references.html`：明确引用、完整书目信息与论文原文入口；
- `site/evidence-ledger.html`：完整证据总账。
- `site/en/`：与上述29个中文页面一一对应的英文版本；每页顶部均可切换语言并保持当前页面。

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

- 28 个 HTML 页面是否齐全；
- 本地链接、图片、脚本和页内锚点是否存在；
- 实验索引是否包含 25 张卡片；
- 主文是否链接 E01–E29，并包含四张关键结果图；
- 每个实验页是否包含概览、动机和结果三个主要分区。
