from __future__ import annotations

import html
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from markdown import Markdown
from markdown.extensions.toc import TocExtension

from experiments_en import EXPERIMENTS_EN


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "experiments"
SITE_ROOT = ROOT / "site"
WEB_ASSETS = ROOT / "web_assets"
FIGURE_ROOT = ROOT / "assets" / "figures"


@dataclass(frozen=True)
class Experiment:
    number: int
    code: str
    directory: Path
    slug: str
    title: str
    purpose: str
    scripts: tuple[Path, ...]


def slugify(value: str, separator: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = re.sub(r"[^\w\u3400-\u9fff-]+", separator, value)
    value = re.sub(rf"{re.escape(separator)}+", separator, value)
    return value.strip(separator) or "section"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def first_heading(markdown_text: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown_text)
    return match.group(1).strip() if match else fallback


def strip_first_h1(markdown_text: str) -> str:
    return re.sub(r"\A\s*#\s+.+?\r?\n+", "", markdown_text, count=1)


def section_paragraph(markdown_text: str, heading: str) -> str:
    pattern = rf"(?ms)^##\s+{re.escape(heading)}\s*$\s*(.+?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown_text)
    if not match:
        body = strip_first_h1(markdown_text)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        return plain_text(paragraphs[0]) if paragraphs else "实验说明与复现材料。"
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", match.group(1)) if part.strip()]
    return plain_text(paragraphs[0]) if paragraphs else "实验说明与复现材料。"


def plain_text(markdown_text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown_text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_>#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def experiment_number(path: Path) -> int:
    match = re.match(r"experiment_(\d+)_", path.name)
    if not match:
        raise ValueError(f"无法解析实验编号：{path.name}")
    return int(match.group(1))


def load_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    for directory in sorted(EXPERIMENT_ROOT.glob("experiment_*"), key=experiment_number):
        number = experiment_number(directory)
        readme = read_text(directory / "README.md")
        title = first_heading(readme, f"E{number:02d}")
        purpose = section_paragraph(readme, "目的")
        experiments.append(
            Experiment(
                number=number,
                code=f"E{number:02d}",
                directory=directory,
                slug=f"e{number:02d}.html",
                title=title,
                purpose=purpose,
                scripts=tuple(sorted(directory.glob("*.py"))),
            )
        )
    return experiments


def rewrite_main_links(markdown_text: str, experiments: list[Experiment]) -> str:
    text = markdown_text
    text = text.replace("(experiments/README.md)", "(experiments/index.html)")
    text = text.replace("(evidence_ledger_zh.md)", "(evidence-ledger.html)")
    for experiment in experiments:
        source = f"experiments/{experiment.directory.name}/README.md"
        target = f"experiments/{experiment.slug}"
        text = text.replace(f"({source})", f"({target})")
    return text


def rewrite_ledger_links(markdown_text: str, experiments: list[Experiment]) -> str:
    text = markdown_text.replace("(short_paper_zh.md)", "(index.html)")
    text = text.replace("(experiments/README.md)", "(experiments/index.html)")
    for experiment in experiments:
        source = f"experiments/{experiment.directory.name}/README.md"
        target = f"experiments/{experiment.slug}"
        text = text.replace(f"({source})", f"({target})")
    text = re.sub(
        r"\(experiments/(experiment_\d+_[^/)]+/[^)]+\.py)\)",
        r"(../experiments/\1)",
        text,
    )
    return text


def rewrite_experiment_links(markdown_text: str, experiment: Experiment) -> str:
    text = markdown_text
    text = text.replace("(MOTIVATION_AND_PREREGISTRATION.md)", "(#motivation)")
    text = text.replace("(RESULTS_AND_CONCLUSION.md)", "(#results)")
    text = text.replace("(../../assets/figures/", "(../assets/figures/")
    for document in experiment.directory.glob("*.md"):
        if document.name in {
            "README.md",
            "MOTIVATION_AND_PREREGISTRATION.md",
            "RESULTS_AND_CONCLUSION.md",
        }:
            continue
        text = text.replace(
            f"({document.name})",
            f"(../../experiments/{experiment.directory.name}/{document.name})",
        )
    for script in experiment.scripts:
        text = text.replace(
            f"({script.name})",
            f"(../../experiments/{experiment.directory.name}/{script.name})",
        )
    return text


def make_markdown() -> Markdown:
    return Markdown(
        extensions=[
            "extra",
            "sane_lists",
            "tables",
            "fenced_code",
            TocExtension(slugify=slugify, permalink=False, toc_depth="2-3"),
        ],
        output_format="html5",
    )


def protect_math(markdown_text: str) -> tuple[str, list[tuple[str, str, str]]]:
    tokens: list[tuple[str, str, str]] = []
    output: list[str] = []
    display_buffer: list[str] = []
    in_display = False
    in_fence = False
    fence = ""
    inline_pattern = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

    def protect_inline(line: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = f"MATHINLINETOKEN{len(tokens):05d}X"
            tokens.append((token, "inline", match.group(1)))
            return token

        return inline_pattern.sub(replace, line)

    for line in markdown_text.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^\s*(```|~~~)", line)

        if in_fence:
            output.append(line)
            if stripped.startswith(fence):
                in_fence = False
                fence = ""
            continue

        if fence_match and not in_display:
            in_fence = True
            fence = fence_match.group(1)
            output.append(line)
            continue

        if stripped == "$$":
            if not in_display:
                in_display = True
                display_buffer = []
            else:
                token = f"MATHBLOCKTOKEN{len(tokens):05d}X"
                tokens.append((token, "display", "\n".join(display_buffer)))
                output.append(token)
                in_display = False
                display_buffer = []
            continue

        if in_display:
            display_buffer.append(line)
        else:
            output.append(protect_inline(line))

    if in_display:
        raise ValueError("发现未闭合的 $$ 块公式。")
    return "\n".join(output), tokens


def restore_math(body: str, tokens: list[tuple[str, str, str]]) -> str:
    restored = body
    for token, kind, formula in tokens:
        escaped = html.escape(formula, quote=False)
        if kind == "display":
            replacement = f'<div class="math-display">$$\n{escaped}\n$$</div>'
            restored = restored.replace(f"<p>{token}</p>", replacement)
            restored = restored.replace(token, replacement)
        else:
            restored = restored.replace(token, f'<span class="math-inline">${escaped}$</span>')
    return restored


def render_markdown(markdown_text: str) -> tuple[str, str]:
    protected, math_tokens = protect_math(markdown_text)
    renderer = make_markdown()
    body = renderer.convert(protected)
    body = restore_math(body, math_tokens)
    body = postprocess_html(body)
    return body, renderer.toc


def postprocess_html(body: str) -> str:
    body = re.sub(
        r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>\s*<p><em>((?:图\s*\d+：|Figure\s*\d+\.).*?)</em></p>',
        lambda match: (
            '<figure class="result-figure">'
            f'<a href="{match.group(2)}" target="_blank" rel="noopener">'
            f'<img src="{match.group(2)}" alt="{match.group(1)}" loading="lazy"></a>'
            f'<figcaption>{match.group(3)}</figcaption></figure>'
        ),
        body,
        flags=re.S,
    )
    body = re.sub(
        r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>',
        lambda match: (
            '<figure class="result-figure">'
            f'<a href="{match.group(2)}" target="_blank" rel="noopener">'
            f'<img src="{match.group(2)}" alt="{match.group(1)}" loading="lazy"></a>'
            f'<figcaption>{match.group(1)}</figcaption></figure>'
        ),
        body,
    )
    body = re.sub(
        r'<a href="(https?://[^"]+)"',
        r'<a href="\1" target="_blank" rel="noopener"',
        body,
    )
    return body


def toc_sidebar(toc: str, extra: str = "", language: str = "zh") -> str:
    label = "本页目录" if language == "zh" else "On this page"
    empty = "本页无分节目录" if language == "zh" else "No section outline"
    return f"""
    <aside class="sidebar" id="sidebar" aria-label="{label}">
      <div class="sidebar-inner">
        <div class="sidebar-label">{label}</div>
        <nav class="toc">{toc or f'<p class="muted">{empty}</p>'}</nav>
        {extra}
      </div>
    </aside>
    """


def topbar(prefix: str, active: str, language: str, switch_href: str) -> str:
    if language == "en":
        items = [
            ("paper", f"{prefix}index.html", "Research"),
            ("core", f"{prefix}theory-core.html", "Core Theory"),
            ("experiments", f"{prefix}experiments/index.html", "Experiments"),
            ("references", f"{prefix}references.html", "References"),
            ("ledger", f"{prefix}evidence-ledger.html", "Evidence"),
        ]
        site_name = "Neural K Research Archive"
        home_label = "Back to the research narrative"
        menu_label = "Contents"
        navigation_label = "Site navigation"
        language_switch = (
            f'<nav class="language-switch" aria-label="Language">'
            f'<a href="{switch_href}" lang="zh-CN">中文</a><span class="active" lang="en">EN</span></nav>'
        )
    else:
        items = [
            ("paper", f"{prefix}index.html", "研究主文"),
            ("core", f"{prefix}theory-core.html", "理论核心"),
            ("experiments", f"{prefix}experiments/index.html", "实验索引"),
            ("references", f"{prefix}references.html", "参考文献"),
            ("ledger", f"{prefix}evidence-ledger.html", "证据总账"),
        ]
        site_name = "Neural K 研究档案"
        home_label = "返回研究主文"
        menu_label = "目录"
        navigation_label = "全站导航"
        language_switch = (
            f'<nav class="language-switch" aria-label="语言">'
            f'<span class="active" lang="zh-CN">中文</span><a href="{switch_href}" lang="en">EN</a></nav>'
        )
    links = "".join(
        f'<a class="top-link{" active" if key == active else ""}" href="{href}">{label}</a>'
        for key, href, label in items
    )
    return f"""
    <header class="topbar">
      <a class="site-mark" href="{prefix}index.html" aria-label="{home_label}">
        <span class="mark-block">NK</span>
        <span class="mark-text">{site_name}</span>
      </a>
      <nav class="top-nav" aria-label="{navigation_label}">{links}</nav>
      {language_switch}
      <button class="nav-toggle" type="button" data-nav-toggle aria-controls="sidebar" aria-expanded="false">
        <span aria-hidden="true">☰</span><span>{menu_label}</span>
      </button>
    </header>
    """


def page_template(
    *,
    title: str,
    description: str,
    eyebrow: str,
    body: str,
    sidebar: str,
    prefix: str,
    nav_prefix: str | None = None,
    active: str,
    page_kind: str,
    language: str = "zh",
    switch_href: str = "#",
    article_footer: str = "",
) -> str:
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    document_language = "en" if language == "en" else "zh-CN"
    site_title = "Neural K Research Archive" if language == "en" else "Neural K 研究档案"
    nav_prefix = prefix if nav_prefix is None else nav_prefix
    return f"""<!doctype html>
<html lang="{document_language}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{safe_description}">
  <title>{safe_title} · {site_title}</title>
  <link rel="stylesheet" href="{prefix}assets/site.css">
  <script>
    window.MathJax = {{
      tex: {{inlineMath: [['$', '$']], displayMath: [['$$', '$$']]}},
      options: {{skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']}}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script defer src="{prefix}assets/site.js"></script>
</head>
<body data-page-kind="{page_kind}">
  <div class="reading-progress" aria-hidden="true"><span></span></div>
  {topbar(nav_prefix, active, language, switch_href)}
  <div class="page-shell">
    {sidebar}
    <main class="main-content" id="main-content">
      <article class="article">
        <header class="article-header">
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h1>{safe_title}</h1>
          <p class="article-description">{safe_description}</p>
        </header>
        <div class="article-body">{body}</div>
        {article_footer}
      </article>
    </main>
  </div>
</body>
</html>
"""


def experiment_nav(experiments: list[Experiment], prefix: str, current: int | None = None) -> str:
    links = []
    for experiment in experiments:
        active = " active" if experiment.number == current else ""
        links.append(
            f'<a class="experiment-nav-link{active}" href="{prefix}{experiment.slug}">'
            f'<span>{experiment.code}</span><span>{html.escape(experiment.title.split("：", 1)[-1])}</span></a>'
        )
    return '<div class="sidebar-label experiment-label">全部实验</div><nav class="experiment-nav">' + "".join(links) + "</nav>"


def experiment_nav_en(experiments: list[Experiment], prefix: str, current: int | None = None) -> str:
    links = []
    for experiment in experiments:
        translated = EXPERIMENTS_EN[experiment.number]
        active = " active" if experiment.number == current else ""
        links.append(
            f'<a class="experiment-nav-link{active}" href="{prefix}{experiment.slug}">'
            f'<span>{experiment.code}</span><span>{html.escape(translated["title"])}</span></a>'
        )
    return '<div class="sidebar-label experiment-label">All experiments</div><nav class="experiment-nav">' + "".join(links) + "</nav>"


def build_paper(experiments: list[Experiment]) -> None:
    source = rewrite_main_links(read_text(ROOT / "short_paper_zh.md"), experiments)
    title = first_heading(source, "研究主文")
    body, toc = render_markdown(strip_first_h1(source))
    metrics = """
    <div class="metric-strip" aria-label="文档概览">
      <div><strong>25</strong><span>组证据单元</span></div>
      <div><strong>18</strong><span>组关键实验展开</span></div>
      <div><strong>4</strong><span>张关键结果图</span></div>
    </div>
    """
    disclosure = """
    <aside class="research-disclosure" role="note" aria-labelledby="research-disclosure-title">
      <div class="disclosure-kicker" id="research-disclosure-title">研究状态说明</div>
      <p>本网页汇总的是作者近期密集完成的大量研究与实验。把这些材料整理成严谨、完整的论文，需要投入非常大的额外工作量；限于作者目前的精力，现阶段暂时由 AI 对研究进行总结和网页化整理。</p>
      <p><strong>除“理论核心”外，本站主文、研究证据总账和实验说明的具体文字均由 AI 撰写，仍有不少表达不清、结构不成熟和细节解释不足之处。</strong>研究问题、理论判断和实验工作来自作者；AI 负责整理现有对话、脚本、结果包与文档，不应把当前网页视为已经完成的正式论文。</p>
      <p><strong><a href="theory-core.html">“理论核心”页</a>是唯一的文字写作例外：</strong>作者对该页进行了逐段审阅和深度修改，因此它是目前对核心研究主张相对更准确的表述。除该页外，作者没有参与本站其他页面的具体文字写作。</p>
      <p>需要核查或寻找更详细解释时，请优先查看研究证据总账、各实验详情页、原始实验脚本及其完整说明。作者之后会继续修改当前网页，并推进正式论文写作。</p>
      <div class="disclosure-links">
        <a href="evidence-ledger.html">查看研究证据总账</a>
        <a href="experiments/index.html">查看 E01–E25 实验与脚本</a>
      </div>
    </aside>
    """
    core_entry = """
    <div class="core-entry-banner">
      <div><strong>想直接看核心研究主张？</strong><span>可跳过探索叙事，直接阅读简单性原则、Neural K-profile、统计物理、预测、agreement 与 AGI 理论关系。</span></div>
      <a href="theory-core.html">打开理论核心 →</a>
    </div>
    """
    body = disclosure + core_entry + metrics + body
    page = page_template(
        title=title,
        description="用实验转折讲清训练 loss、函数体积、优化器运输与 Neural K-profile。",
        eyebrow="研究主文 · 可读叙事版",
        body=body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="experiments/index.html">查看全部实验 →</a>'),
        prefix="",
        active="paper",
        page_kind="paper",
        switch_href="en/index.html",
        article_footer='<footer class="article-footer"><a href="experiments/index.html">继续查看 E01–E25 实验索引 →</a></footer>',
    )
    (SITE_ROOT / "index.html").write_text(page, encoding="utf-8")


def build_ledger(experiments: list[Experiment]) -> None:
    source = rewrite_ledger_links(read_text(ROOT / "evidence_ledger_zh.md"), experiments)
    title = first_heading(source, "证据总账")
    body, toc = render_markdown(strip_first_h1(source))
    body = re.sub(
        r'<p>(<a href="experiments/e\d+\.html">E\d+</a>)',
        r'<p class="evidence-experiment">\1',
        body,
    )
    notice = """
    <div class="notice warning"><strong>内部证据总账</strong><span>这里保留完整数字、反例和限定语。第一次阅读请从研究主文开始。</span></div>
    """
    reading_path = """
    <nav class="ledger-path" aria-label="证据总账推荐阅读路径">
      <span>推荐阅读路径</span>
      <a href="#摘要">摘要</a>
      <a href="#3-实验判决链">实验判决链</a>
      <a href="#4-实验链最终收敛出的理论图景">最终理论图景</a>
      <a href="#6-结论边界与尚未解决的问题">证据边界</a>
    </nav>
    """
    page = page_template(
        title=title,
        description="完整实验数字、边界、反例与早期理论版本。",
        eyebrow="内部材料 · 完整证据版",
        body=notice + reading_path + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">返回可读主文 →</a>'),
        prefix="",
        active="ledger",
        page_kind="ledger",
        switch_href="en/evidence-ledger.html",
    )
    (SITE_ROOT / "evidence-ledger.html").write_text(page, encoding="utf-8")


def build_references() -> None:
    source = read_text(ROOT / "references_zh.md")
    title = first_heading(source, "参考文献")
    body, toc = render_markdown(strip_first_h1(source))
    reference_count = len(re.findall(r"(?m)^###\s+R\d+", source))
    summary = f"""
    <div class="reference-summary">
      <strong>{reference_count}</strong>
      <span>条明确书目，按函数先验、Grokking/特征学习、算法信息/MDL、平坦性/SLT和主动学习分组。</span>
    </div>
    """
    page = page_template(
        title=title,
        description="当前网页明确提及的外部研究、正式书目信息、原文入口与本站引用关系。",
        eyebrow="外部研究 · 原始论文入口",
        body=summary + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">返回研究主文 →</a>'),
        prefix="",
        active="references",
        page_kind="references",
        switch_href="en/references.html",
    )
    (SITE_ROOT / "references.html").write_text(page, encoding="utf-8")


def build_theory_core() -> None:
    source = read_text(ROOT / "theory_core_zh.md")
    title = first_heading(source, "理论核心")
    body, toc = render_markdown(strip_first_h1(source))
    page = page_template(
        title=title,
        description="简单性、Neural K-profile、静态/动力学分层、统计物理、预测、信息增益、agreement与符号形成的最小理论总纲。",
        eyebrow="独立总纲 · 只写理论核心",
        body=body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">返回研究主文 →</a>'),
        prefix="",
        active="core",
        page_kind="core",
        switch_href="en/theory-core.html",
    )
    (SITE_ROOT / "theory-core.html").write_text(page, encoding="utf-8")


def combined_experiment_markdown(experiment: Experiment) -> str:
    readme = rewrite_experiment_links(read_text(experiment.directory / "README.md"), experiment)
    motivation_path = experiment.directory / "MOTIVATION_AND_PREREGISTRATION.md"
    results_path = experiment.directory / "RESULTS_AND_CONCLUSION.md"
    motivation = read_text(motivation_path) if motivation_path.exists() else "暂无独立动机文档。"
    results = read_text(results_path) if results_path.exists() else "暂无独立结果文档。"
    motivation = rewrite_experiment_links(motivation, experiment)
    results = rewrite_experiment_links(results, experiment)
    return "\n\n".join(
        [
            "## 实验概览 {#overview}",
            strip_first_h1(readme),
            "## 为什么做、怎样判决 {#motivation}",
            strip_first_h1(motivation),
            "## 结果、解读与边界 {#results}",
            strip_first_h1(results),
        ]
    )


def experiment_footer(experiments: list[Experiment], index: int) -> str:
    previous_link = "<span></span>"
    next_link = "<span></span>"
    if index > 0:
        previous = experiments[index - 1]
        previous_link = f'<a href="{previous.slug}">← {previous.code} {html.escape(previous.title.split("：", 1)[-1])}</a>'
    if index + 1 < len(experiments):
        nxt = experiments[index + 1]
        next_link = f'<a href="{nxt.slug}">{nxt.code} {html.escape(nxt.title.split("：", 1)[-1])} →</a>'
    return f'<footer class="experiment-pager">{previous_link}{next_link}</footer>'


def build_experiment_pages(experiments: list[Experiment]) -> None:
    output = SITE_ROOT / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    sidebar_list = experiment_nav(experiments, "", None)

    cards = []
    for experiment in experiments:
        scripts = f"{len(experiment.scripts)} 个复现脚本"
        search_text = html.escape(f"{experiment.code} {experiment.title} {experiment.purpose}".lower())
        cards.append(
            f"""
            <article class="experiment-card" data-experiment-card data-search-text="{search_text}">
              <div class="experiment-card-top"><span class="experiment-code">{experiment.code}</span><span>{scripts}</span></div>
              <h2><a href="{experiment.slug}">{html.escape(experiment.title.split('：', 1)[-1])}</a></h2>
              <p>{html.escape(experiment.purpose)}</p>
              <a class="card-link" href="{experiment.slug}">查看实验说明 →</a>
            </article>
            """
        )

    index_body = f"""
    <div class="experiment-toolbar">
      <label for="experiment-search">搜索实验</label>
      <input id="experiment-search" type="search" placeholder="输入 E23、parity、MNIST…" data-experiment-search autocomplete="off">
      <span class="search-count" data-search-count>{len(experiments)} / {len(experiments)}</span>
    </div>
    <div class="experiment-grid">{''.join(cards)}</div>
    <p class="empty-search" data-empty-search hidden>没有匹配的实验。</p>
    """
    index_page = page_template(
        title="E01–E25 实验索引",
        description="每页合并实验目的、具体操作、判决标准、结果、边界和复现脚本。",
        eyebrow="实验档案 · 25 组证据单元",
        body=index_body,
        sidebar=toc_sidebar("", sidebar_list),
        prefix="../",
        active="experiments",
        page_kind="experiment-index",
        switch_href="../en/experiments/index.html",
    )
    (output / "index.html").write_text(index_page, encoding="utf-8")

    for index, experiment in enumerate(experiments):
        source = combined_experiment_markdown(experiment)
        body, toc = render_markdown(source)
        script_links = "".join(
            f'<a href="../../experiments/{experiment.directory.name}/{script.name}">{html.escape(script.name)}</a>'
            for script in experiment.scripts
        )
        source_bar = (
            '<div class="source-bar"><span>复现脚本</span><div>'
            + (script_links or "<span>本实验无独立脚本</span>")
            + "</div></div>"
        )
        extra = experiment_nav(experiments, "", experiment.number)
        page = page_template(
            title=experiment.title,
            description=experiment.purpose,
            eyebrow=f"{experiment.code} · 实验详情",
            body=source_bar + body,
            sidebar=toc_sidebar(toc, extra),
            prefix="../",
            active="experiments",
            page_kind="experiment",
            switch_href=f"../en/experiments/{experiment.slug}",
            article_footer=experiment_footer(experiments, index),
        )
        (output / experiment.slug).write_text(page, encoding="utf-8")


def english_experiment_markdown(experiment: Experiment) -> str:
    translated = EXPERIMENTS_EN[experiment.number]
    figure_map = {
        22: ("e22_free_energy.png", "Representative static free-energy endpoint costs"),
        23: ("e23_volume_to_transition.png", "Complete-rule volume and data transitions"),
        24: ("e24_deep_crossing.png", "Shared-parent deep-tail volume crossing"),
        25: ("e25_mnist_static_prediction.png", "MNIST static branch prediction and concentration"),
    }
    results = translated["results"].strip()
    if experiment.number in figure_map:
        filename, caption = figure_map[experiment.number]
        results += f"\n\n![{caption}](../../assets/figures/{filename})"
    if experiment.number == 23:
        results += """

![Five-target Gaussian deep-tail SMC](../../assets/figures/e23_gaussian_deep_tail.png)

![Uniform/cell/conflict optimizer intervention](../../assets/figures/e23_sampling_intervention.png)

![Uniform/cell/conflict fixed-D static SMC](../../assets/figures/e23_fixed_d_static_smc.png)
"""
    return "\n\n".join(
        [
            "## Experiment overview {#overview}",
            translated["overview"].strip(),
            "## Why this experiment was run {#motivation}",
            translated["motivation"].strip(),
            "## Results, interpretation, and limits {#results}",
            results,
        ]
    )


def experiment_footer_en(experiments: list[Experiment], index: int) -> str:
    previous_link = "<span></span>"
    next_link = "<span></span>"
    if index > 0:
        previous = experiments[index - 1]
        previous_link = f'<a href="{previous.slug}">← {previous.code} {html.escape(EXPERIMENTS_EN[previous.number]["title"])}</a>'
    if index + 1 < len(experiments):
        nxt = experiments[index + 1]
        next_link = f'<a href="{nxt.slug}">{nxt.code} {html.escape(EXPERIMENTS_EN[nxt.number]["title"])} →</a>'
    return f'<footer class="experiment-pager">{previous_link}{next_link}</footer>'


def build_paper_en(experiments: list[Experiment], output: Path) -> None:
    source = read_text(ROOT / "short_paper_en.md")
    source = source.replace("(assets/figures/", "(../assets/figures/")
    title = first_heading(source, "Research narrative")
    body, toc = render_markdown(strip_first_h1(source))
    metrics = """
    <div class="metric-strip" aria-label="Document overview">
      <div><strong>25</strong><span>evidence units</span></div>
      <div><strong>18</strong><span>key experiments explained</span></div>
      <div><strong>4</strong><span>result figures</span></div>
    </div>
    """
    disclosure = """
    <aside class="research-disclosure" role="note" aria-labelledby="research-disclosure-title-en">
      <div class="disclosure-kicker" id="research-disclosure-title-en">Research status notice</div>
      <p>This website summarizes a large body of research and experiments completed by the author in a short period. Turning the full record into a rigorous, submission-ready paper is a substantial additional project; given the author’s current time and energy constraints, the present archive has been summarized and organized by AI.</p>
      <p><strong>Except for the Core Theory page, the specific prose of the research narrative, evidence ledger, and experiment descriptions was written by AI and still contains unclear explanations, immature structure, and missing detail.</strong> The research questions, theoretical judgments, and experiments come from the author. The current site should not be treated as a finished paper.</p>
      <p><strong>The <a href="theory-core.html">Core Theory page</a> is the sole authorship exception:</strong> the author reviewed it paragraph by paragraph and participated deeply in revising it, so it is currently the relatively most accurate statement of the core research claims. The author did not participate in writing the specific prose on the other pages.</p>
      <p>For verification or fuller context, consult the evidence ledger, experiment detail pages, original scripts, and preserved result notes. The author intends to continue revising the website and developing a formal paper.</p>
      <div class="disclosure-links">
        <a href="evidence-ledger.html">Open the evidence ledger</a>
        <a href="experiments/index.html">Browse E01–E25 and scripts</a>
      </div>
    </aside>
    """
    core_entry = """
    <div class="core-entry-banner">
      <div><strong>Want the core claims first?</strong><span>Skip the research chronology and open the standalone framework for simplicity, Neural K-profiles, statistical physics, prediction, agreement, and AGI relations.</span></div>
      <a href="theory-core.html">Open Core Theory →</a>
    </div>
    """
    page = page_template(
        title=title,
        description="An experiment-led account of training loss, function volume, optimizer transport, and the Neural K-profile.",
        eyebrow="Research narrative · English edition",
        body=disclosure + core_entry + metrics + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="experiments/index.html">Browse all experiments →</a>', language="en"),
        prefix="../",
        nav_prefix="",
        active="paper",
        page_kind="paper",
        language="en",
        switch_href="../index.html",
        article_footer='<footer class="article-footer"><a href="experiments/index.html">Continue to the E01–E25 experiment index →</a></footer>',
    )
    (output / "index.html").write_text(page, encoding="utf-8")


def build_ledger_en(experiments: list[Experiment], output: Path) -> None:
    source = read_text(ROOT / "evidence_ledger_en.md")
    evidence_sections = []
    for experiment in experiments:
        translated = EXPERIMENTS_EN[experiment.number]
        evidence_sections.append(
            "\n\n".join(
                [
                    f'### {experiment.code} · [{translated["title"]}](experiments/{experiment.slug})',
                    f'**Purpose.** {translated["purpose"]}',
                    translated["overview"].strip(),
                    translated["motivation"].strip(),
                    translated["results"].strip(),
                ]
            )
        )
    source = source.replace("## 4. Final integrated picture", "\n\n".join(evidence_sections) + "\n\n## 4. Final integrated picture")
    title = first_heading(source, "Evidence ledger")
    body, toc = render_markdown(strip_first_h1(source))
    body = re.sub(
        r'<p>(<a href="experiments/e\d+\.html">E\d+</a>)',
        r'<p class="evidence-experiment">\1',
        body,
    )
    notice = '<div class="notice warning"><strong>Internal evidence ledger</strong><span>This page retains detailed measurements, negative results, and scope limits. Start with the research narrative on a first read.</span></div>'
    reading_path = """
    <nav class="ledger-path" aria-label="Recommended evidence-ledger reading path">
      <span>Recommended path</span>
      <a href="#abstract">Abstract</a>
      <a href="#3-experimental-evidence-chain">Evidence chain</a>
      <a href="#4-final-integrated-picture">Integrated picture</a>
      <a href="#6-evidence-boundaries-and-next-falsifiable-tests">Boundaries</a>
    </nav>
    """
    page = page_template(
        title=title,
        description="Measurement definitions, decisive numbers, negative results, integrated interpretation, and scope boundaries.",
        eyebrow="Internal material · English evidence edition",
        body=notice + reading_path + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">Return to the narrative →</a>', language="en"),
        prefix="../",
        nav_prefix="",
        active="ledger",
        page_kind="ledger",
        language="en",
        switch_href="../evidence-ledger.html",
    )
    (output / "evidence-ledger.html").write_text(page, encoding="utf-8")


def build_references_en(output: Path) -> None:
    source = read_text(ROOT / "references_en.md")
    title = first_heading(source, "References")
    body, toc = render_markdown(strip_first_h1(source))
    reference_count = len(re.findall(r"(?m)^###\s+R\d+", source))
    summary = f'<div class="reference-summary"><strong>{reference_count}</strong><span>explicit bibliographic entries grouped by their role in this archive.</span></div>'
    page = page_template(
        title=title,
        description="External research cited by this website, with full metadata, primary-paper links, and the role of each citation.",
        eyebrow="External research · English references",
        body=summary + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">Return to the research narrative →</a>', language="en"),
        prefix="../",
        nav_prefix="",
        active="references",
        page_kind="references",
        language="en",
        switch_href="../references.html",
    )
    (output / "references.html").write_text(page, encoding="utf-8")


def build_theory_core_en(output: Path) -> None:
    source = read_text(ROOT / "theory_core_en.md")
    title = first_heading(source, "Core theory")
    body, toc = render_markdown(strip_first_h1(source))
    page = page_template(
        title=title,
        description="The minimal framework for simplicity, Neural K-profiles, static versus dynamical selection, statistical physics, prediction, information gain, agreement, and symbolic emergence.",
        eyebrow="Standalone framework · Core theory only",
        body=body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">Return to the research narrative →</a>', language="en"),
        prefix="../",
        nav_prefix="",
        active="core",
        page_kind="core",
        language="en",
        switch_href="../theory-core.html",
    )
    (output / "theory-core.html").write_text(page, encoding="utf-8")


def build_experiment_pages_en(experiments: list[Experiment], output: Path) -> None:
    experiment_output = output / "experiments"
    experiment_output.mkdir(parents=True, exist_ok=True)
    sidebar_list = experiment_nav_en(experiments, "", None)
    cards = []
    for experiment in experiments:
        translated = EXPERIMENTS_EN[experiment.number]
        script_label = f"{len(experiment.scripts)} reproduction script" + ("s" if len(experiment.scripts) != 1 else "")
        search_text = html.escape(f'{experiment.code} {translated["title"]} {translated["purpose"]}'.lower())
        cards.append(
            f'<article class="experiment-card" data-experiment-card data-search-text="{search_text}">'
            f'<div class="experiment-card-top"><span class="experiment-code">{experiment.code}</span><span>{script_label}</span></div>'
            f'<h2><a href="{experiment.slug}">{html.escape(translated["title"])}</a></h2>'
            f'<p>{html.escape(translated["purpose"])}</p>'
            f'<a class="card-link" href="{experiment.slug}">Open experiment details →</a></article>'
        )
    index_body = f"""
    <div class="experiment-toolbar">
      <label for="experiment-search-en">Search experiments</label>
      <input id="experiment-search-en" type="search" placeholder="Try E23, parity, MNIST…" data-experiment-search autocomplete="off">
      <span class="search-count" data-search-count>{len(experiments)} / {len(experiments)}</span>
    </div>
    <div class="experiment-grid">{''.join(cards)}</div>
    <p class="empty-search" data-empty-search hidden>No matching experiment.</p>
    """
    index_page = page_template(
        title="E01–E25 Experiment Index",
        description="Each page combines the motivation, actual design, decision criteria, results, limits, and source-script links.",
        eyebrow="Experiment archive · 25 evidence units",
        body=index_body,
        sidebar=toc_sidebar("", sidebar_list, language="en"),
        prefix="../../",
        nav_prefix="../",
        active="experiments",
        page_kind="experiment-index",
        language="en",
        switch_href="../../experiments/index.html",
    )
    (experiment_output / "index.html").write_text(index_page, encoding="utf-8")

    for index, experiment in enumerate(experiments):
        translated = EXPERIMENTS_EN[experiment.number]
        body, toc = render_markdown(english_experiment_markdown(experiment))
        script_links = "".join(
            f'<a href="../../../experiments/{experiment.directory.name}/{script.name}">{html.escape(script.name)}</a>'
            for script in experiment.scripts
        )
        source_bar = '<div class="source-bar"><span>Scripts</span><div>' + (script_links or "<span>No standalone script</span>") + "</div></div>"
        page = page_template(
            title=f'{experiment.code}: {translated["title"]}',
            description=translated["purpose"],
            eyebrow=f"{experiment.code} · Experiment details",
            body=source_bar + body,
            sidebar=toc_sidebar(toc, experiment_nav_en(experiments, "", experiment.number), language="en"),
            prefix="../../",
            nav_prefix="../",
            active="experiments",
            page_kind="experiment",
            language="en",
            switch_href=f"../../experiments/{experiment.slug}",
            article_footer=experiment_footer_en(experiments, index),
        )
        (experiment_output / experiment.slug).write_text(page, encoding="utf-8")


def build_english_site(experiments: list[Experiment]) -> None:
    if set(EXPERIMENTS_EN) != set(range(1, 26)):
        raise RuntimeError("English experiment translations must cover E01–E25 exactly.")
    output = SITE_ROOT / "en"
    output.mkdir(parents=True, exist_ok=True)
    build_paper_en(experiments, output)
    build_ledger_en(experiments, output)
    build_references_en(output)
    build_theory_core_en(output)
    build_experiment_pages_en(experiments, output)


def copy_assets() -> None:
    assets = SITE_ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEB_ASSETS / "site.css", assets / "site.css")
    shutil.copy2(WEB_ASSETS / "site.js", assets / "site.js")
    figure_output = assets / "figures"
    figure_output.mkdir(parents=True, exist_ok=True)
    for figure in FIGURE_ROOT.glob("*.png"):
        shutil.copy2(figure, figure_output / figure.name)


def prepare_site_root() -> None:
    resolved = SITE_ROOT.resolve()
    if resolved.parent != ROOT.resolve():
        raise RuntimeError(f"拒绝清理意外路径：{resolved}")
    if SITE_ROOT.exists():
        shutil.rmtree(SITE_ROOT)
    SITE_ROOT.mkdir(parents=True)


def write_manifest(experiments: list[Experiment]) -> None:
    payload = {
        "title": "Neural K 研究档案",
        "entry": "index.html",
        "languages": {"zh-CN": "index.html", "en": "en/index.html"},
        "core_theory": {"zh-CN": "theory-core.html", "en": "en/theory-core.html"},
        "experiment_count": len(experiments),
        "experiments": [
            {
                "id": item.code,
                "title_zh": item.title,
                "title_en": EXPERIMENTS_EN[item.number]["title"],
                "page_zh": f"experiments/{item.slug}",
                "page_en": f"en/experiments/{item.slug}",
            }
            for item in experiments
        ],
    }
    (SITE_ROOT / "site-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    experiments = load_experiments()
    if len(experiments) != 25:
        raise RuntimeError(f"预期 25 个实验目录，实际得到 {len(experiments)} 个。")
    prepare_site_root()
    copy_assets()
    build_paper(experiments)
    build_ledger(experiments)
    build_references()
    build_theory_core()
    build_experiment_pages(experiments)
    build_english_site(experiments)
    write_manifest(experiments)
    html_count = len(list(SITE_ROOT.rglob("*.html")))
    print(f"Built {html_count} HTML pages in {SITE_ROOT}")


if __name__ == "__main__":
    main()
