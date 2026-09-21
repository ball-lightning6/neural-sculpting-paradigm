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


# 提交后新增的七组公开实验。物理目录仍保留历史名称，网站和公开叙事使用
# 连续的 E31--E37 编号；这样不会把内部试验目录名误当成公开实验编号。
POST_UPLOAD_EXPERIMENTS = (
    (
        "E31",
        "Loss 与幅度对齐",
        "Loss and output amplitude",
        "experiments/e31.html",
        "raw MSE、BCE 幅度对齐和目标幅度对规则恢复与 grokking 的影响。",
        "Raw MSE, BCE amplitude alignment, and target amplitude in rule recovery.",
    ),
    (
        "E32",
        "随机记忆缩放",
        "Random-memory scaling",
        "experiments/e32.html",
        "随机标签记忆复杂度随样本数近似线性增加。",
        "Approximate linear growth of random-memory complexity.",
    ),
    (
        "E33",
        "跨架构联合输出体积",
        "Cross-architecture output volume",
        "experiments/e33.html",
        "不同架构下规则联合输出体积的有限协议测量。",
        "Finite-protocol joint-output volume across architectures.",
    ),
    (
        "E34",
        "条件规则决斗",
        "Conditional rule duel",
        "experiments/e34.html",
        "冻结的 4-bit Neural K 全序预测 75 次 8-bit 决斗。",
        "A frozen 4-bit Neural K ordering predicts 75 conditional duels.",
    ),
    (
        "E35",
        "Mod97 冻结函数决斗",
        "Frozen functions around grokking",
        "experiments/e35.html",
        "grokking 前后真实完整函数在新网络中的竞争偏好。",
        "A real pre-grokking function loses to the completed rule.",
    ),
    (
        "E36",
        "Peleg/G&C 独立验证",
        "Independent Peleg/G&C recheck",
        "experiments/e36.html",
        "复现 Peleg/G&C 的静态插值对照，并检查收紧 raw BCE 后静态采样与 SGD 的差距。",
        "Independent replication of the Peleg/G&C comparison under tightened raw BCE.",
    ),
    (
        "E37",
        "测度鲁棒性与 Gaussian 对应",
        "Measure robustness and Gaussian alignment",
        "experiments/e37.html",
        "Gaussian 与固定范数球欧式测度下的规则排序、端点响应和真实训练对应边界。",
        "Rule ordering, endpoint response, and training correspondence under Gaussian and fixed-ball Euclidean measures.",
    ),
)


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


def rewrite_ledger_links_en(markdown_text: str, experiments: list[Experiment]) -> str:
    """英文证据总账（site/en/ 下）的链接改写：仓库脚本链接比中文页多退一级。"""
    text = markdown_text.replace("(short_paper_zh.md)", "(index.html)")
    text = text.replace("(short_paper_en.md)", "(index.html)")
    text = text.replace("(experiments/README.md)", "(experiments/index.html)")
    for experiment in experiments:
        source = f"experiments/{experiment.directory.name}/README.md"
        target = f"experiments/{experiment.slug}"
        text = text.replace(f"({source})", f"({target})")
    text = re.sub(
        r"\(experiments/(experiment_\d+_[^/)]+/[^)]+\.py)\)",
        r"(../../experiments/\1)",
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


def rewrite_experiment_links_en(markdown_text: str, experiment: Experiment) -> str:
    """英文实验页（site/en/experiments/ 下）的链接改写：仓库文件链接比中文页多退一级。

    图片原始链接 (../../assets/figures/...) 从 site/en/experiments/ 出发恰好解析到
    site/assets/figures/，因此无需像中文页那样改写资源路径。
    """
    text = markdown_text
    text = text.replace("(MOTIVATION_AND_PREREGISTRATION.md)", "(#motivation)")
    text = text.replace("(RESULTS_AND_CONCLUSION.md)", "(#results)")
    for document in experiment.directory.glob("*.md"):
        if document.name in {
            "README.md",
            "MOTIVATION_AND_PREREGISTRATION.md",
            "RESULTS_AND_CONCLUSION.md",
        }:
            continue
        text = text.replace(
            f"({document.name})",
            f"(../../../experiments/{experiment.directory.name}/{document.name})",
        )
    for script in experiment.scripts:
        text = text.replace(
            f"({script.name})",
            f"(../../../experiments/{experiment.directory.name}/{script.name})",
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
    display_delimiter = ""
    in_fence = False
    fence = ""
    inline_pattern = re.compile(
        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)|\\\\\((.+?)\\\\\)"
    )

    def protect_inline(line: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = f"MATHINLINETOKEN{len(tokens):05d}X"
            tokens.append((token, "inline", match.group(1) if match.group(1) is not None else match.group(2)))
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

        if stripped in {"$$", r"\[", r"\]"}:
            if not in_display:
                in_display = True
                display_delimiter = stripped
                display_buffer = []
            else:
                if (display_delimiter == "$$" and stripped != "$$") or (
                    display_delimiter == r"\[" and stripped != r"\]"
                ):
                    output.append(line)
                    continue
                token = f"MATHBLOCKTOKEN{len(tokens):05d}X"
                tokens.append((token, "display", "\n".join(display_buffer)))
                output.append(token)
                in_display = False
                display_delimiter = ""
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
            ("post-upload", f"{prefix}post-upload.html", "Updates"),
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
            ("post-upload", f"{prefix}post-upload.html", "提交后增量"),
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


def post_upload_experiment_nav(prefix: str, language: str = "zh") -> str:
    label = "提交后增量实验" if language == "zh" else "Post-upload experiments"
    links = []
    for code, title_zh, title_en, href, _summary_zh, _summary_en in POST_UPLOAD_EXPERIMENTS:
        title = title_zh if language == "zh" else title_en
        links.append(
            f'<a class="experiment-nav-link" href="{prefix}{href}">'
            f'<span>{code}</span><span>{html.escape(title)}</span></a>'
        )
    return f'<div class="sidebar-label experiment-label">{label}</div><nav class="experiment-nav">' + "".join(links) + "</nav>"


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
      <div><strong>37</strong><span>组已编号证据单元</span></div>
      <div><strong>30</strong><span>组历史档案实验</span></div>
      <div><strong>7</strong><span>组提交后增量实验</span></div>
      <div><strong>22</strong><span>组关键实验展开</span></div>
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
        <a href="experiments/index.html">查看 E01–E37 实验与脚本</a>
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
        article_footer='<footer class="article-footer"><a href="experiments/index.html">继续查看 E01–E37 实验索引 →</a></footer>',
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
    supplement_links = """
    <div class="metric-strip" aria-label="理论补充入口">
      <div><strong><a href="post-upload-language.html">严格概率语言</a></strong><span>联合输出、函数质量和信息更新</span></div>
      <div><strong><a href="post-upload-mdl.html">MDL—grokking 长推导</a></strong><span>竞争质量与样本相变</span></div>
      <div><strong><a href="post-upload-theory-update.html">理论增量全文</a></strong><span>统计语言和周末理论修正</span></div>
      <div><strong><a href="post-upload-estimator.html">Neural K 估计器</a></strong><span>CGF、鞍点与 520 节点回测</span></div>
    </div>
    """
    body = body + supplement_links
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


POST_UPLOAD = ROOT / "release_20260920"
TRANSLATIONS_EN = ROOT / "translations_en"


def rewrite_post_upload_links(markdown_text: str, link_base: str, asset_prefix: str = "assets/post_upload") -> str:
    """把源文档里的相对文件链接改写成站点内可解析的仓库相对路径。"""

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        target, _, fragment = raw.partition("#")
        if fragment:
            fragment = "#" + fragment
        if not target or target.startswith(("http://", "https://", "mailto:", "/")):
            return match.group(0)
        if re.match(r"^[A-Za-z]:[\\/]", target):
            # 报告中遗留的本机绝对路径不能进入公开 HTML。若文件存在且足够
            # 轻量，复制到生成站点的共享资产目录；否则只保留链接文字。
            source_path = Path(target)
            if not source_path.exists() or source_path.is_dir():
                return "]"
            suffix = source_path.suffix.lower()
            try:
                size = source_path.stat().st_size
            except OSError:
                return "]"
            # 公开网页只复制实际展示的图像。JSON/CSV/脚本由实验目录中的
            # 可重跑代码产生，不重复打包成网页附件。
            if suffix not in {".png", ".jpg", ".jpeg", ".svg"} or size > 20 * 1024 * 1024:
                return "]"
            import hashlib

            asset_name = f"{hashlib.sha1(str(source_path).encode('utf-8')).hexdigest()[:10]}_{source_path.name}"
            asset_dir = SITE_ROOT / "assets" / "post_upload"
            asset_dir.mkdir(parents=True, exist_ok=True)
            destination = asset_dir / asset_name
            if not destination.exists():
                shutil.copy2(source_path, destination)
            return f"]({asset_prefix}/{asset_name})"
        if not re.search(r"\.(md|py|json|csv|png|txt|html)$", target, flags=re.IGNORECASE):
            return match.group(0)
        return f"]({link_base}/{target}{fragment})"

    return re.sub(r"\]\(([^)]+)\)", repl, markdown_text)


def post_upload_page(
    source: str,
    fallback: str,
    eyebrow: str,
    description: str,
    outfile: str,
    switch_href: str,
    prefix: str = "",
    link_base: str | None = None,
    language: str = "zh",
    nav_prefix: str | None = None,
) -> None:
    if link_base:
        source = rewrite_post_upload_links(source, link_base, f"{prefix}assets/post_upload")
    body, toc = render_markdown(strip_first_h1(source))
    overview_label = "返回增量总览 →" if language == "zh" else "Back to the increments overview →"
    nav_base = nav_prefix if nav_prefix is not None else prefix
    page = page_template(
        title=first_heading(source, fallback),
        description=description,
        eyebrow=eyebrow,
        body=body,
        sidebar=toc_sidebar(
            toc,
            f'<a class="sidebar-action" href="{nav_base}post-upload.html">{overview_label}</a>',
            language=language,
        ),
        prefix=prefix,
        nav_prefix=nav_prefix,
        active="post-upload",
        page_kind="post-upload",
        language=language,
        switch_href=switch_href,
    )
    (SITE_ROOT / outfile).write_text(page, encoding="utf-8")


def build_post_upload() -> None:
    if not POST_UPLOAD.exists():
        print("release_20260920 不存在，跳过提交后增量页面。")
        return
    source = read_text(POST_UPLOAD / "POST_UPLOAD_EXPERIMENTS_ZH.md")
    notice = """
    <div class="notice warning"><strong>提交后增量</strong><span>本组页面对应 2026-09-02 短论文提交之后新增的理论与实验材料。新增实验使用 E31–E37 公共编号，前三项关联既有 E15/E23 等证据。</span></div>
    """
    links = """
    <div class="metric-strip" aria-label="增量入口">
      <div><strong><a href="post-upload-mdl.html">MDL 与 grokking</a></strong><span>最低可行复杂度与条件推导</span></div>
      <div><strong><a href="post-upload-language.html">严格概率语言</a></strong><span>联合输出、信息更新与统计物理边界</span></div>
      <div><strong><a href="post-upload-theory-update.html">理论增量全文</a></strong><span>统计语言、MDL、复杂度收窄与周末修正</span></div>
      <div><strong><a href="post-upload-estimator.html">架构估计器</a></strong><span>训练前估算 Neural K</span></div>
      <div><strong><a href="experiments/e31.html">E31 Loss 与幅度</a></strong><span>目标幅度与联合输出</span></div>
      <div><strong><a href="experiments/e32.html">E32 随机记忆</a></strong><span>复杂度近线性缩放</span></div>
      <div><strong><a href="experiments/e33.html">E33 跨架构</a></strong><span>联合输出体积</span></div>
      <div><strong><a href="experiments/e34.html">E34 规则决斗</a></strong><span>75 次事前预测全部命中</span></div>
      <div><strong><a href="experiments/e35.html">E35 grokking 决斗</a></strong><span>真实候选函数偏好</span></div>
      <div><strong><a href="experiments/e36.html">E36 Peleg/G&C 验证</a></strong><span>静态 vs SGD 对照</span></div>
      <div><strong><a href="experiments/e37.html">E37 测度鲁棒性</a></strong><span>Gaussian、欧式球与训练对应</span></div>
      <div><strong><a href="post-upload-decision.html">上传裁决</a></strong><span>本批材料取舍清单</span></div>
    </div>
    """
    body, toc = render_markdown(strip_first_h1(source))
    page = page_template(
        title=first_heading(source, "提交后增量"),
        description="2026-09-02 短论文提交之后的新增理论与实验：MDL—grokking、规则对决、幅度对齐、随机记忆、跨架构、架构估计器与 Peleg 复核。",
        eyebrow="提交后增量 · 2026-09",
        body=notice + links + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">返回研究主文 →</a>'),
        prefix="",
        active="post-upload",
        page_kind="post-upload",
        switch_href="en/post-upload.html",
    )
    (SITE_ROOT / "post-upload.html").write_text(page, encoding="utf-8")

    mdl_source = "\n\n".join(
        [
            read_text(POST_UPLOAD / "theory" / "MDL_FRONTIER_AND_GROKKING.md"),
            read_text(POST_UPLOAD / "theory" / "mdl_identifiability_grokking_derivation_zh.md"),
        ]
    )
    post_upload_page(
        mdl_source,
        "MDL 与 grokking",
        "理论增量 · MDL 与 grokking",
        "最低可行复杂度、剩余候选重排与 grokking 数据相变的条件推导。",
        "post-upload-mdl.html",
        "en/post-upload-mdl.html",
        link_base="../release_20260920/theory",
    )
    post_upload_page(
        read_text(ROOT / "supplementary" / "architecture_estimator" / "WEBSITE_ESTIMATOR_FULL_ZH.md"),
        "架构估计器",
        "实验增量 · 架构估计器",
        "从架构与参数先验直接估算 Neural K：有限宽 CGF、鞍点近似与 520 节点回测。",
        "post-upload-estimator.html",
        "en/post-upload-estimator.html",
        link_base="../supplementary/architecture_estimator",
    )
    post_upload_page(
        read_text(POST_UPLOAD / "theory" / "WEBSITE_PROBABILITY_LANGUAGE_ZH.md"),
        "严格概率语言",
        "理论增量 · 严格概率语言",
        "联合输出、函数质量、信息更新和统计物理术语的严格定义。",
        "post-upload-language.html",
        "en/post-upload-language.html",
    )
    post_upload_page(
        read_text(POST_UPLOAD / "theory" / "MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md"),
        "测度鲁棒性与 Gaussian 对应",
        "理论与实验增量 · 测度鲁棒性",
        "Gaussian、固定范数球内欧式测度、输出中心/幅度对照，以及真实训练对应的解释边界。",
        "post-upload-measure.html",
        "en/post-upload-measure.html",
        link_base="../experiments/experiment_37_measure_robustness",
    )
    post_upload_page(
        read_text(POST_UPLOAD / "theory" / "WEEKEND_THEORY_UPDATE.md"),
        "理论增量全文",
        "理论增量 · 2026-09-19 至 2026-09-20",
        "统计语言严格化、MDL—grokking、复杂度分布收窄、幅度对齐和相关边界的完整周末整理。",
        "post-upload-theory-update.html",
        "en/post-upload-theory-update.html",
        link_base="../release_20260920/theory",
    )
    peleg_source = "\n\n".join(
        [
            read_text(EXPERIMENT_ROOT / "experiment_36_peleg_g_and_c_validation" / "README.md"),
            read_text(EXPERIMENT_ROOT / "experiment_36_peleg_g_and_c_validation" / "CONCLUSION.md"),
        ]
    )
    post_upload_page(
        peleg_source,
        "E36 Peleg/G&C 独立验证",
        "实验增量 · E36 Peleg/G&C 独立验证",
        "收紧 raw BCE 后，无梯度静态采样达到与 SGD 相当的 MNIST 泛化；G&C 结论不构成反例。",
        "post-upload-peleg.html",
        "en/post-upload-peleg.html",
        link_base="../experiments/experiment_36_peleg_g_and_c_validation",
    )
    e32_source = "\n\n".join(
        [
            read_text(EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "README.md"),
            read_text(EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "POWER_MOD97_ALIGNMENT_REPORT.md"),
            read_text(EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "POWER_MOD97_MSE_ROUNDS_REPORT.md"),
            read_text(EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "MARGIN_ALIGNMENT_STATIC_VERIFICATION.md"),
            read_text(EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "COPY_4BIT_MSE_AMPLITUDE_SERIES.md"),
        ]
    )
    post_upload_page(
        e32_source,
        "E31 loss 与幅度对齐",
        "实验增量 · E31 loss 与幅度对齐",
        "raw MSE、BCE 幅度对齐和目标幅度如何改变规则恢复与 grokking。",
        "post-upload-e31.html",
        "en/post-upload-e31.html",
        link_base="../experiments/experiment_31_loss_amplitude_grokking",
    )
    e33_source = read_text(EXPERIMENT_ROOT / "experiment_32_random_memory" / "RANDOM_MEMORY_SCALING_REPORT.md")
    post_upload_page(
        e33_source,
        "E32 随机记忆",
        "实验增量 · E32 随机记忆",
        "随机标签记忆复杂度与样本数的近线性关系。",
        "post-upload-e32.html",
        "en/post-upload-e32.html",
        link_base="../experiments/experiment_32_random_memory",
    )
    e34_source = "\n\n".join(
        [
            read_text(EXPERIMENT_ROOT / "experiment_33_cross_architecture" / "README_CROSS_ARCHITECTURE_RULE_VOLUME.md"),
            read_text(EXPERIMENT_ROOT / "experiment_33_cross_architecture" / "CROSS_ARCHITECTURE_RULE_VOLUME_REPORT.md"),
            read_text(EXPERIMENT_ROOT / "experiment_33_cross_architecture" / "CROSS_ARCHITECTURE_SMC_FEASIBILITY.md"),
        ]
    )
    post_upload_page(
        e34_source,
        "E33 跨架构联合输出",
        "实验增量 · E33 跨架构联合输出",
        "不同网络架构下的规则联合输出体积、收缩曲线和采样可行性。",
        "post-upload-e33.html",
        "en/post-upload-e33.html",
        link_base="../experiments/experiment_33_cross_architecture",
    )
    post_upload_page(
        read_text(EXPERIMENT_ROOT / "experiment_34_conditional_rule_duel" / "README.md"),
        "E34 条件规则决斗",
        "实验增量 · E34 条件规则决斗",
        "独立冻结的 4-bit Neural K 全序能否预测 8-bit 条件规则竞争中的函数人口偏好。",
        "post-upload-e34.html",
        "en/post-upload-e34.html",
        link_base="../experiments/experiment_34_conditional_rule_duel",
    )
    post_upload_page(
        read_text(EXPERIMENT_ROOT / "experiment_35_mod97_frozen_function_duels" / "README.md"),
        "E35 Mod97 冻结函数决斗",
        "实验增量 · E35 Mod97 冻结函数决斗",
        "grokking 前后的真实完整函数在新网络中的竞争偏好。",
        "post-upload-e35.html",
        "en/post-upload-e35.html",
        link_base="../experiments/experiment_35_mod97_frozen_function_duels",
    )
    post_upload_page(
        read_text(POST_UPLOAD / "UPLOAD_DECISION.md"),
        "上传裁决",
        "发布说明 · 上传裁决",
        "本次增量包的主包、补充包与不上传材料的取舍清单。",
        "post-upload-decision.html",
        "en/post-upload-decision.html",
        link_base="../release_20260920",
    )


def build_post_upload_en() -> None:
    if not POST_UPLOAD.exists():
        return
    overview_path = TRANSLATIONS_EN / "post_upload" / "overview.md"
    if not overview_path.exists():
        raise RuntimeError(f"缺少英文增量总览完整翻译：{overview_path}")
    source = read_text(overview_path)
    notice = """
    <div class="notice warning"><strong>Post-upload increments</strong><span>This group of pages covers new theory and experimental material completed after the 2026-09-02 preprint. New experiments use the public numbering E31–E37; the first three connect to existing evidence such as E15/E23.</span></div>
    """
    links = """
    <div class="metric-strip" aria-label="Increment entries">
      <div><strong><a href="post-upload-mdl.html">MDL and grokking</a></strong><span>Minimum viable complexity and conditional derivations</span></div>
      <div><strong><a href="post-upload-language.html">Strict probability language</a></strong><span>Joint outputs, updates, and statistical-physics boundaries</span></div>
      <div><strong><a href="post-upload-theory-update.html">Full theory increment</a></strong><span>Strict language, MDL, narrowing, and weekend corrections</span></div>
      <div><strong><a href="post-upload-estimator.html">Architecture estimator</a></strong><span>Estimate Neural K before training</span></div>
      <div><strong><a href="experiments/e31.html">E31 Loss and amplitude</a></strong><span>Target amplitude and the joint output</span></div>
      <div><strong><a href="experiments/e32.html">E32 Random memory</a></strong><span>Near-linear complexity scaling</span></div>
      <div><strong><a href="experiments/e33.html">E33 Cross-architecture</a></strong><span>Joint-output volume</span></div>
      <div><strong><a href="experiments/e34.html">E34 Rule duels</a></strong><span>75 preregistered predictions all confirmed</span></div>
      <div><strong><a href="experiments/e35.html">E35 Grokking duels</a></strong><span>Real candidate-function preference</span></div>
      <div><strong><a href="experiments/e36.html">E36 Peleg/G&amp;C recheck</a></strong><span>Static versus SGD comparison</span></div>
      <div><strong><a href="experiments/e37.html">E37 Measure robustness</a></strong><span>Gaussian, Euclidean balls, and training correspondence</span></div>
      <div><strong><a href="post-upload-decision.html">Upload decision</a></strong><span>What this increment includes</span></div>
    </div>
    """
    body, toc = render_markdown(strip_first_h1(source))
    page = page_template(
        title=first_heading(source, "Post-upload increments"),
        description="Theory and experiments added after the 2026-09-02 preprint: MDL-grokking, rule duels, amplitude alignment, random memory, cross-architecture volume, architecture estimator, and the Peleg/G&C recheck.",
        eyebrow="Post-upload increments · 2026-09",
        body=notice + links + body,
        sidebar=toc_sidebar(toc, '<a class="sidebar-action" href="index.html">Back to the research narrative →</a>', language="en"),
        prefix="../",
        nav_prefix="",
        active="post-upload",
        page_kind="post-upload",
        language="en",
        switch_href="../post-upload.html",
    )
    (SITE_ROOT / "en" / "post-upload.html").write_text(page, encoding="utf-8")

    def en_base(zh_base: str) -> str:
        """英文页面位于 site/en/ 下，指向发布树的链接要比中文页多退一级。"""
        return "../" + zh_base

    mdl_source = "\n\n".join(
        [
            read_text(TRANSLATIONS_EN / "post_upload" / "mdl.md"),
            read_text(TRANSLATIONS_EN / "theory" / "mdl_identifiability_grokking_derivation.md"),
        ]
    )
    post_upload_page(
        mdl_source,
        "MDL and grokking",
        "Theory increment · MDL and grokking",
        "Minimum viable complexity, reallocation among remaining candidates, and a conditional derivation of the grokking data transition.",
        "en/post-upload-mdl.html",
        "../post-upload-mdl.html",
        prefix="../",
        link_base=en_base("../release_20260920/theory"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "architecture_bias" / "website_estimator_full.md"),
        "Architecture estimator",
        "Experiment increment · Architecture estimator",
        "Estimating Neural K before training: finite-width CGF, saddlepoint approximation, and a 520-node backtest.",
        "en/post-upload-estimator.html",
        "../post-upload-estimator.html",
        prefix="../",
        link_base=en_base("../supplementary/architecture_estimator"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "theory" / "probability_language.md"),
        "Strict probability language",
        "Theory increment · Strict probability language",
        "Precise definitions for joint outputs, function mass, information updates, and statistical-physics terminology.",
        "en/post-upload-language.html",
        "../post-upload-language.html",
        prefix="../",
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "theory" / "measure_robustness_and_gaussian_alignment.md"),
        "Measure robustness and Gaussian alignment",
        "Theory and experiment increment · Measure robustness",
        "Gaussian and fixed-ball Euclidean measures, output-center/amplitude controls, and the boundary of the training correspondence.",
        "en/post-upload-measure.html",
        "../post-upload-measure.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_37_measure_robustness"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "theory" / "weekend_theory_update.md"),
        "Full theory increment",
        "Theory increment · 2026-09-19 to 2026-09-20",
        "The full weekend update on strict statistical language, MDL-grokking, complexity narrowing, amplitude alignment, and boundaries.",
        "en/post-upload-theory-update.html",
        "../post-upload-theory-update.html",
        prefix="../",
        link_base=en_base("../release_20260920/theory"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "peleg.md"),
        "E36 Peleg/G&C independent verification",
        "Experiment increment · E36 Peleg/G&C independent verification",
        "With a tightened raw-BCE constraint, gradient-free static sampling reaches SGD-level MNIST generalization; the G&C comparison does not falsify the static picture.",
        "en/post-upload-peleg.html",
        "../post-upload-peleg.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_36_peleg_g_and_c_validation"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "e31.md"),
        "E31 Loss and amplitude alignment",
        "Experiment increment · E31 Loss and amplitude alignment",
        "How raw MSE, BCE amplitude alignment, and target amplitude change rule recovery and grokking.",
        "en/post-upload-e31.html",
        "../post-upload-e31.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_31_loss_amplitude_grokking"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "e32.md"),
        "E32 Random memory",
        "Experiment increment · E32 Random memory",
        "The near-linear relation between random-label memory complexity and sample count.",
        "en/post-upload-e32.html",
        "../post-upload-e32.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_32_random_memory"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "e33.md"),
        "E33 Cross-architecture joint output",
        "Experiment increment · E33 Cross-architecture joint output",
        "Rule joint-output volume, contraction curves, and sampling feasibility across network architectures.",
        "en/post-upload-e33.html",
        "../post-upload-e33.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_33_cross_architecture"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "e34.md"),
        "E34 Conditional rule duel",
        "Experiment increment · E34 Conditional rule duel",
        "Whether an independently frozen 4-bit Neural K ordering predicts function-population preference in 8-bit conditional rule duels.",
        "en/post-upload-e34.html",
        "../post-upload-e34.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_34_conditional_rule_duel"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "e35.md"),
        "E35 Mod97 frozen-function duels",
        "Experiment increment · E35 Mod97 frozen-function duels",
        "The competitive preference between real complete functions from before and after grokking in fresh networks.",
        "en/post-upload-e35.html",
        "../post-upload-e35.html",
        prefix="../",
        link_base=en_base("../experiments/experiment_35_mod97_frozen_function_duels"),
        language="en",
        nav_prefix="",
    )
    post_upload_page(
        read_text(TRANSLATIONS_EN / "post_upload" / "decision.md"),
        "Upload decision",
        "Release note · Upload decision",
        "What is included in this increment package and what is deliberately left out.",
        "en/post-upload-decision.html",
        "../post-upload-decision.html",
        prefix="../",
        link_base=en_base("../release_20260920"),
        language="en",
        nav_prefix="",
    )


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
    sidebar_list = experiment_nav(experiments, "", None) + post_upload_experiment_nav("../")

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

    total_experiments = len(experiments) + len(POST_UPLOAD_EXPERIMENTS)
    supplemental_cards = "".join(
        f'<article class="experiment-card supplemental-card" data-experiment-card '
        f'data-search-text="{html.escape(f"{code} {title_zh} {summary_zh}".lower())}">'
        f'<div class="experiment-card-top"><span class="experiment-code">{code}</span>'
        f'<span>提交后增量</span></div><h2><a href="../{href}">{html.escape(title_zh)}</a></h2>'
        f'<p>{html.escape(summary_zh)}</p><a class="card-link" href="../{href}">查看增量实验 →</a></article>'
        for code, title_zh, _title_en, href, summary_zh, _summary_en in POST_UPLOAD_EXPERIMENTS
    )


    index_body = f"""
    <div class="experiment-toolbar">
      <label for="experiment-search">搜索实验</label>
      <input id="experiment-search" type="search" placeholder="输入 E23、parity、MNIST…" data-experiment-search autocomplete="off">
      <span class="search-count" data-search-count>{total_experiments} / {total_experiments}</span>
    </div>
    <div class="experiment-grid">{''.join(cards)}</div>
    <section class="supplement-experiments">
      <div class="section-label">提交后增量实验</div>
      <h2>新增实验与补充结果</h2>
      <div class="experiment-grid">{supplemental_cards}</div>
    </section>
    <p class="empty-search" data-empty-search hidden>没有匹配的实验。</p>
    """
    index_page = page_template(
        title="E01–E37 实验索引",
        description="每页合并实验目的、具体操作、判决标准、结果、边界和复现脚本。",
        eyebrow="实验档案 · 36 组编号证据单元",
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
        extra = experiment_nav(experiments, "", experiment.number) + post_upload_experiment_nav("../")
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


def build_canonical_increment_pages() -> None:
    """把 E31--E37 生成为统一 experiments/eNN.html 主入口。

    post-upload-eNN.html 仍由 build_post_upload 保留，作为旧链接兼容别名；
    网站索引、主文和 manifest 使用这里的 canonical 页面。
    """

    def joined(*paths: Path) -> str:
        return "\n\n".join(read_text(path) for path in paths)

    def public_title(source: str, title: str) -> str:
        """把历史报告标题替换为公开 E 编号，正文内容保持不变。"""
        if re.search(r"(?m)^#\s+", source):
            return re.sub(r"\A\s*#\s+.+?\r?\n", f"# {title}\n", source, count=1)
        return f"# {title}\n\n{source}"

    release = POST_UPLOAD
    chinese = {
        "e31": joined(
            EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "README.md",
            EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "POWER_MOD97_ALIGNMENT_REPORT.md",
            EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "POWER_MOD97_MSE_ROUNDS_REPORT.md",
            EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "MARGIN_ALIGNMENT_STATIC_VERIFICATION.md",
            EXPERIMENT_ROOT / "experiment_31_loss_amplitude_grokking" / "COPY_4BIT_MSE_AMPLITUDE_SERIES.md",
        ),
        "e32": read_text(EXPERIMENT_ROOT / "experiment_32_random_memory" / "RANDOM_MEMORY_SCALING_REPORT.md"),
        "e33": joined(
            EXPERIMENT_ROOT / "experiment_33_cross_architecture" / "README_CROSS_ARCHITECTURE_RULE_VOLUME.md",
            EXPERIMENT_ROOT / "experiment_33_cross_architecture" / "CROSS_ARCHITECTURE_RULE_VOLUME_REPORT.md",
            EXPERIMENT_ROOT / "experiment_33_cross_architecture" / "CROSS_ARCHITECTURE_SMC_FEASIBILITY.md",
        ),
        "e34": read_text(EXPERIMENT_ROOT / "experiment_34_conditional_rule_duel" / "README.md"),
        "e35": read_text(EXPERIMENT_ROOT / "experiment_35_mod97_frozen_function_duels" / "README.md"),
        "e36": joined(
            EXPERIMENT_ROOT / "experiment_36_peleg_g_and_c_validation" / "README.md",
            EXPERIMENT_ROOT / "experiment_36_peleg_g_and_c_validation" / "CONCLUSION.md",
        ),
        "e37": read_text(EXPERIMENT_ROOT / "experiment_37_measure_robustness" / "README.md"),
    }
    english = {
        "e31": read_text(TRANSLATIONS_EN / "post_upload" / "e31.md"),
        "e32": read_text(TRANSLATIONS_EN / "post_upload" / "e32.md"),
        "e33": read_text(TRANSLATIONS_EN / "post_upload" / "e33.md"),
        "e34": read_text(TRANSLATIONS_EN / "post_upload" / "e34.md"),
        "e35": read_text(TRANSLATIONS_EN / "post_upload" / "e35.md"),
        "e36": read_text(TRANSLATIONS_EN / "post_upload" / "peleg.md"),
        "e37": read_text(TRANSLATIONS_EN / "post_upload" / "e37.md"),
    }
    meta = {
        "e31": ("E31 Loss 与幅度对齐", "E31 Loss and output amplitude", "实验增量 · E31", "Experiment increment · E31"),
        "e32": ("E32 随机记忆", "E32 Random memory", "实验增量 · E32", "Experiment increment · E32"),
        "e33": ("E33 跨架构联合输出", "E33 Cross-architecture joint output", "实验增量 · E33", "Experiment increment · E33"),
        "e34": ("E34 条件规则决斗", "E34 Conditional rule duel", "实验增量 · E34", "Experiment increment · E34"),
        "e35": ("E35 Mod97 冻结函数决斗", "E35 Mod97 frozen-function duels", "实验增量 · E35", "Experiment increment · E35"),
        "e36": ("E36 Peleg/G&C 独立验证", "E36 Peleg/G&C independent verification", "实验增量 · E36", "Experiment increment · E36"),
        "e37": ("E37 测度鲁棒性与 Gaussian 对应", "E37 Measure robustness and Gaussian alignment", "实验增量 · E37", "Experiment increment · E37"),
    }
    link_bases = {
        "e31": ("../../experiments/experiment_31_loss_amplitude_grokking", "../../../experiments/experiment_31_loss_amplitude_grokking"),
        "e32": ("../../experiments/experiment_32_random_memory", "../../../experiments/experiment_32_random_memory"),
        "e33": ("../../experiments/experiment_33_cross_architecture", "../../../experiments/experiment_33_cross_architecture"),
        "e34": ("../../experiments/experiment_34_conditional_rule_duel", "../../../experiments/experiment_34_conditional_rule_duel"),
        "e35": ("../../experiments/experiment_35_mod97_frozen_function_duels", "../../../experiments/experiment_35_mod97_frozen_function_duels"),
        "e36": ("../../experiments/experiment_36_peleg_g_and_c_validation", "../../../experiments/experiment_36_peleg_g_and_c_validation"),
        "e37": ("../../experiments/experiment_37_measure_robustness", "../../../experiments/experiment_37_measure_robustness"),
    }
    for slug, zh_source in chinese.items():
        zh_title, en_title, zh_eyebrow, en_eyebrow = meta[slug]
        post_upload_page(
            public_title(zh_source, zh_title),
            zh_title,
            zh_eyebrow,
            f"{zh_title}的完整协议、结果和边界。",
            f"experiments/{slug}.html",
            f"../en/experiments/{slug}.html",
            prefix="../",
            nav_prefix="../",
            link_base=link_bases[slug][0],
        )
        post_upload_page(
            public_title(english[slug], en_title),
            en_title,
            en_eyebrow,
            f"Full protocol, results, and boundaries for {en_title}.",
            f"en/experiments/{slug}.html",
            f"../../experiments/{slug}.html",
            prefix="../../",
            nav_prefix="../",
            link_base=link_bases[slug][1],
            language="en",
        )
def english_experiment_markdown(experiment: Experiment) -> str:
    """英文实验正文读取 translations_en 的逐节完整翻译，不再使用 experiments_en.py 摘要。"""
    path = TRANSLATIONS_EN / "experiments" / f"e{experiment.number:02d}.md"
    if not path.exists():
        raise RuntimeError(f"缺少英文实验完整翻译：{path}")
    return rewrite_experiment_links_en(read_text(path), experiment)


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
    source = rewrite_main_links(read_text(ROOT / "short_paper_en.md"), experiments)
    source = source.replace("(assets/figures/", "(../assets/figures/")
    title = first_heading(source, "Research narrative")
    body, toc = render_markdown(strip_first_h1(source))
    metrics = """
    <div class="metric-strip" aria-label="Document overview">
      <div><strong>37</strong><span>numbered evidence units</span></div>
      <div><strong>30</strong><span>historical archive experiments</span></div>
      <div><strong>7</strong><span>post-upload experiments</span></div>
      <div><strong>22</strong><span>key experiments explained</span></div>
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
        <a href="experiments/index.html">Browse E01–E37 and scripts</a>
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
        article_footer='<footer class="article-footer"><a href="experiments/index.html">Continue to the E01–E37 experiment index →</a></footer>',
    )
    (output / "index.html").write_text(page, encoding="utf-8")


def build_ledger_en(experiments: list[Experiment], output: Path) -> None:
    source = rewrite_ledger_links_en(read_text(ROOT / "evidence_ledger_en.md"), experiments)
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
      <a href="#3-the-experimental-verdict-chain">Verdict chain</a>
      <a href="#4-the-theoretical-picture-to-which-the-experimental-chain-finally-converged">Integrated picture</a>
      <a href="#6-boundaries-of-the-conclusions-and-open-problems">Boundaries</a>
    </nav>
    """
    page = page_template(
        title=title,
        description="Measurement definitions, decisive numbers, negative results, and scope boundaries.",
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
    supplement_links = """
    <div class="metric-strip" aria-label="Theory supplements">
      <div><strong><a href="post-upload-language.html">Strict probability language</a></strong><span>Joint outputs, function mass, and updates</span></div>
      <div><strong><a href="post-upload-mdl.html">Long MDL–grokking derivation</a></strong><span>Competitor mass and sample transitions</span></div>
      <div><strong><a href="experiments/e37.html">E37 Measure robustness</a></strong><span>Gaussian, Euclidean balls, and training correspondence</span></div>
      <div><strong><a href="post-upload-theory-update.html">Full theory increment</a></strong><span>Strict language and weekend theory corrections</span></div>
      <div><strong><a href="post-upload-estimator.html">Neural K estimator</a></strong><span>CGF, saddlepoint, and 520-node backtest</span></div>
    </div>
    """
    body = body + supplement_links
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
    sidebar_list = experiment_nav_en(experiments, "", None) + post_upload_experiment_nav("../", "en")
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
    total_experiments = len(experiments) + len(POST_UPLOAD_EXPERIMENTS)
    supplemental_cards = "".join(
        f'<article class="experiment-card supplemental-card" data-experiment-card '
        f'data-search-text="{html.escape(f"{code} {title_en} {summary_en}".lower())}">'
        f'<div class="experiment-card-top"><span class="experiment-code">{code}</span>'
        f'<span>post-upload</span></div><h2><a href="../{href}">{html.escape(title_en)}</a></h2>'
        f'<p>{html.escape(summary_en)}</p><a class="card-link" href="../{href}">Open increment →</a></article>'
        for code, _title_zh, title_en, href, _summary_zh, summary_en in POST_UPLOAD_EXPERIMENTS
    )
    index_body = f"""
    <div class="experiment-toolbar">
      <label for="experiment-search-en">Search experiments</label>
      <input id="experiment-search-en" type="search" placeholder="Try E23, parity, MNIST…" data-experiment-search autocomplete="off">
      <span class="search-count" data-search-count>{total_experiments} / {total_experiments}</span>
    </div>
    <div class="experiment-grid">{''.join(cards)}</div>
    <section class="supplement-experiments">
      <div class="section-label">Post-upload experiments</div>
      <h2>New experiments after the preprint</h2>
      <div class="experiment-grid">{supplemental_cards}</div>
    </section>
    <p class="empty-search" data-empty-search hidden>No matching experiment.</p>
    """
    index_page = page_template(
        title="E01–E37 Experiment Index",
        description="Each page combines the motivation, actual design, decision criteria, results, limits, and source-script links.",
        eyebrow="Experiment archive · 36 numbered evidence units",
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
            sidebar=toc_sidebar(toc, experiment_nav_en(experiments, "", experiment.number) + post_upload_experiment_nav("../", "en"), language="en"),
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
    if set(EXPERIMENTS_EN) != set(range(1, 31)):
        raise RuntimeError("English experiment titles must cover E01–E30 exactly.")
    missing_translations = [
        f"e{number:02d}.md"
        for number in range(1, 31)
        if not (TRANSLATIONS_EN / "experiments" / f"e{number:02d}.md").exists()
    ]
    if missing_translations:
        raise RuntimeError(f"缺少英文实验完整翻译：{missing_translations}")
    required_translation_files = [
        TRANSLATIONS_EN / "theory" / "probability_language.md",
        TRANSLATIONS_EN / "theory" / "minimal_statistical_language_audit.md",
        TRANSLATIONS_EN / "theory" / "mdl_identifiability_grokking_derivation.md",
        TRANSLATIONS_EN / "theory" / "weekend_theory_update.md",
        TRANSLATIONS_EN / "theory" / "measure_robustness_and_gaussian_alignment.md",
        TRANSLATIONS_EN / "architecture_bias" / "website_estimator_full.md",
    ]
    missing_theory_translations = [str(path) for path in required_translation_files if not path.exists()]
    if missing_theory_translations:
        raise RuntimeError(f"缺少英文理论补充翻译：{missing_theory_translations}")
    output = SITE_ROOT / "en"
    output.mkdir(parents=True, exist_ok=True)
    build_paper_en(experiments, output)
    build_ledger_en(experiments, output)
    build_references_en(output)
    build_theory_core_en(output)
    build_experiment_pages_en(experiments, output)
    build_post_upload_en()


def copy_assets() -> None:
    assets = SITE_ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEB_ASSETS / "site.css", assets / "site.css")
    shutil.copy2(WEB_ASSETS / "site.js", assets / "site.js")
    figure_output = assets / "figures"
    figure_output.mkdir(parents=True, exist_ok=True)
    for figure in FIGURE_ROOT.glob("*.png"):
        shutil.copy2(figure, figure_output / figure.name)


def copy_public_theory_sources() -> None:
    """把理论源稿放入静态站点，方便从网页直接复查完整版本。"""
    sources = {
        "zh/minimal_strict_statistical_language_tutorial_zh.md": ROOT / "minimal_strict_statistical_language_tutorial_zh.md",
        "zh/minimal_strict_statistical_language_zh.md": POST_UPLOAD / "theory" / "minimal_strict_statistical_language_zh.md",
        "zh/minimal_statistical_language_audit_zh.md": POST_UPLOAD / "theory" / "minimal_statistical_language_audit_zh.md",
        "zh/WEEKEND_THEORY_UPDATE.md": POST_UPLOAD / "theory" / "WEEKEND_THEORY_UPDATE.md",
        "zh/MDL_FRONTIER_AND_GROKKING.md": POST_UPLOAD / "theory" / "MDL_FRONTIER_AND_GROKKING.md",
        "zh/mdl_identifiability_grokking_derivation_zh.md": POST_UPLOAD / "theory" / "mdl_identifiability_grokking_derivation_zh.md",
        "zh/MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md": POST_UPLOAD / "theory" / "MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md",
        "en/probability_language.md": TRANSLATIONS_EN / "theory" / "probability_language.md",
        "en/minimal_statistical_language_audit.md": TRANSLATIONS_EN / "theory" / "minimal_statistical_language_audit.md",
        "en/weekend_theory_update.md": TRANSLATIONS_EN / "theory" / "weekend_theory_update.md",
        "en/mdl_identifiability_grokking_derivation.md": TRANSLATIONS_EN / "theory" / "mdl_identifiability_grokking_derivation.md",
        "en/measure_robustness_and_gaussian_alignment.md": TRANSLATIONS_EN / "theory" / "measure_robustness_and_gaussian_alignment.md",
    }
    output = SITE_ROOT / "source_docs"
    for relative, source in sources.items():
        if not source.exists():
            raise RuntimeError(f"缺少理论源稿：{source}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def prepare_site_root() -> None:
    resolved = SITE_ROOT.resolve()
    if resolved.parent != ROOT.resolve():
        raise RuntimeError(f"拒绝清理意外路径：{resolved}")
    if SITE_ROOT.exists():
        shutil.rmtree(SITE_ROOT)
    SITE_ROOT.mkdir(parents=True)


def write_manifest(experiments: list[Experiment]) -> None:
    experiment_items = [
        {
            "id": item.code,
            "title_zh": item.title,
            "title_en": EXPERIMENTS_EN[item.number]["title"],
            "page_zh": f"experiments/{item.slug}",
            "page_en": f"en/experiments/{item.slug}",
        }
        for item in experiments
    ]
    experiment_items.extend(
        {
            "id": code,
            "title_zh": title_zh,
            "title_en": title_en,
            "page_zh": href,
            "page_en": f"en/{href}",
        }
        for code, title_zh, title_en, href, _summary_zh, _summary_en in POST_UPLOAD_EXPERIMENTS
    )
    payload = {
        "title": "Neural K 研究档案",
        "entry": "index.html",
        "languages": {"zh-CN": "index.html", "en": "en/index.html"},
        "core_theory": {"zh-CN": "theory-core.html", "en": "en/theory-core.html"},
        "experiment_count": len(experiment_items),
        "experiments": experiment_items,
    }
    (SITE_ROOT / "site-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_generated_html() -> None:
    for path in SITE_ROOT.rglob("*.html"):
        source = path.read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in source.splitlines()) + "\n"
        path.write_text(normalized, encoding="utf-8")


def main() -> None:
    experiments = [e for e in load_experiments() if e.number <= 30]
    if len(experiments) != 30:
        raise RuntimeError(f"预期 30 个实验目录，实际得到 {len(experiments)} 个。")
    prepare_site_root()
    copy_assets()
    copy_public_theory_sources()
    build_paper(experiments)
    build_ledger(experiments)
    build_references()
    build_theory_core()
    build_experiment_pages(experiments)
    build_post_upload()
    build_english_site(experiments)
    build_canonical_increment_pages()
    write_manifest(experiments)
    normalize_generated_html()
    html_count = len(list(SITE_ROOT.rglob("*.html")))
    print(f"Built {html_count} HTML pages in {SITE_ROOT}")


if __name__ == "__main__":
    main()
