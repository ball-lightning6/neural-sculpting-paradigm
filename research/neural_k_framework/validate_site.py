from __future__ import annotations

import sys
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.sources: list[str] = []
        self.images: list[str] = []
        self.cards = 0
        self.document_language = ""
        self.title_depth = 0
        self.title_text: list[str] = []
        self.h2_count = 0
        self.h3_count = 0
        self.headings: list[tuple[int, str]] = []
        self.tables = 0
        self.pres = 0
        self.math_display = 0
        self.math_inline = 0
        self._heading_level = 0
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html" and attributes.get("lang"):
            self.document_language = str(attributes["lang"])
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))
        if tag == "a" and attributes.get("href"):
            self.hrefs.append(str(attributes["href"]))
        if tag in {"img", "script"} and attributes.get("src"):
            source = str(attributes["src"])
            self.sources.append(source)
            if tag == "img":
                self.images.append(source)
        if tag == "link" and attributes.get("href"):
            self.sources.append(str(attributes["href"]))
        if "data-experiment-card" in attributes:
            self.cards += 1
        if tag == "title":
            self.title_depth += 1
        if tag in {"h2", "h3"}:
            if tag == "h2":
                self.h2_count += 1
            else:
                self.h3_count += 1
            self._heading_level = int(tag[1])
            self._heading_parts = []
        if tag == "table":
            self.tables += 1
        if tag == "pre":
            self.pres += 1
        classes = str(attributes.get("class", ""))
        if tag == "div" and "math-display" in classes:
            self.math_display += 1
        if tag == "span" and "math-inline" in classes:
            self.math_inline += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag in {"h2", "h3"} and self._heading_level:
            self.headings.append((self._heading_level, "".join(self._heading_parts).strip()))
            self._heading_level = 0
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_text.append(data)
        if self._heading_level:
            self._heading_parts.append(data)


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def local_target(page: Path, url: str) -> tuple[Path | None, str]:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or url.startswith(("mailto:", "javascript:", "data:")):
        return None, ""
    fragment = unquote(parts.fragment)
    if not parts.path:
        return page, fragment
    target = (page.parent / unquote(parts.path)).resolve()
    return target, fragment


def validate_paper_sources(errors: list[str]) -> None:
    """补查不直接渲染为HTML的正式论文，防止整段漏译未被站点校验发现。"""
    paths = [
        ROOT / "papers" / "compression_is_intelligence_zh.md",
        ROOT / "papers" / "compression_is_intelligence_en.md",
    ]
    if not all(path.exists() for path in paths):
        errors.append("缺少正式短论文中英文源稿。")
        return
    texts = [path.read_text(encoding="utf-8") for path in paths]
    signatures = []
    for text in texts:
        displays = re.findall(r"\$\$(.*?)\$\$", text, re.S)
        normalized = [
            re.sub(r"\s+", "", formula).rstrip(".,")
            for formula in displays
        ]
        signatures.append({
            "段落": len(re.split(r"\n\s*\n", text.strip())),
            "标题编号": re.findall(r"(?m)^#{1,4}\s+(\d+(?:\.\d+)*)", text),
            "表格行": len(re.findall(r"(?m)^\|", text)),
            "展示公式": normalized,
            "行内公式分隔符": len(re.findall(r"(?<!\$)\$(?!\$)", text)),
        })
    for label, value in signatures[0].items():
        if value != signatures[1][label]:
            errors.append(f"正式短论文中英不对应：{label}。")
    if "These two processes" not in texts[1]:
        errors.append("英文正式论文缺少两种作用与收窄边界的对应段落。")
    if "both are computable code lengths" in texts[1]:
        errors.append("英文正式论文重新出现无条件可计算码长断言。")


def validate_math_protection(errors: list[str]) -> None:
    from build_site import protect_math

    for example in ["$x_1+1$", r"\(x_1+1\)", r"\\(x_1+1\\)"]:
        _, tokens = protect_math(example)
        if len(tokens) != 1 or tokens[0][1:] != ("inline", "x_1+1"):
            errors.append(f"行内公式保护失败：{example}")
    _, tokens = protect_math("```text\n" + r"\(x_1\)" + "\n```")
    if tokens:
        errors.append("代码块内的公式不应被转换。")


def validate_extension_sources(errors: list[str]) -> None:
    """检查本轮完整译文，避免只凭HTML标题数量放过漏段或数值变化。"""
    from build_site import OCTOBER_EXTENSIONS, october_extension_paths, protect_math

    def signature(source: str) -> dict[str, object]:
        _, formulas = protect_math(source)
        table_rows = re.findall(r"(?m)^\|.*\|\s*$", source)
        numbers = [
            re.findall(r"(?<![0-9.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", row.replace("−", "-"))
            for row in table_rows
        ]
        return {
            "段落块": len(re.split(r"\n\s*\n", source.strip())),
            "标题层级": re.findall(r"(?m)^(#{1,6})\s+", source),
            "表格列数": [len(row.split("|")) for row in table_rows],
            "表格数字": numbers,
            "图片数": len(re.findall(r"!\[[^\]]*\]\(", source)),
            "代码围栏": len(re.findall(r"(?m)^```", source)),
            "公式": [(kind, re.sub(r"\s+", "", formula)) for _, kind, formula in formulas],
        }

    for slug in OCTOBER_EXTENSIONS:
        paths = october_extension_paths(slug)
        if not all(path.is_file() for path in paths):
            errors.append(f"{slug}: 缺少本轮完整双语增补。")
            continue
        sources = [path.read_text(encoding="utf-8") for path in paths]
        for path, source in zip(paths, sources):
            if re.search(r"(?<![A-Za-z0-9_\\])[A-Za-z]:[\\/]", source):
                errors.append(f"{path}: 新公开源稿包含本机绝对路径。")
        signatures = [signature(source) for source in sources]
        for label, value in signatures[0].items():
            if value != signatures[1][label]:
                errors.append(f"{slug}新增中英源稿不对应：{label}。")


def main() -> int:
    errors: list[str] = []
    validate_paper_sources(errors)
    validate_math_protection(errors)
    validate_extension_sources(errors)
    pages = sorted(SITE.rglob("*.html"))
    parsed = {page.resolve(): parse_page(page) for page in pages}

    if len(pages) != 116:
        errors.append(f"预期 116 个 HTML 页面，实际 {len(pages)} 个。")

    required = [
        SITE / "index.html",
        SITE / "evidence-ledger.html",
        SITE / "references.html",
        SITE / "theory-core.html",
        SITE / "width-double-descent.html",
        SITE / "complexity-methods.html",
        SITE / "experiments" / "index.html",
        SITE / "en" / "index.html",
        SITE / "en" / "evidence-ledger.html",
        SITE / "en" / "references.html",
        SITE / "en" / "theory-core.html",
        SITE / "en" / "width-double-descent.html",
        SITE / "en" / "complexity-methods.html",
        SITE / "en" / "experiments" / "index.html",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"缺少页面：{path}")

    for page in pages:
        parser = parsed[page.resolve()]
        raw_html = page.read_text(encoding="utf-8")
        if re.search(r'(?:href|src)="[A-Za-z]:[\\/]', raw_html):
            errors.append(f"{page}: 含本机绝对路径链接或资源")
        duplicates = [item for item, count in Counter(parser.ids).items() if count > 1]
        if duplicates:
            errors.append(f"{page}: 重复 id {duplicates}")
        if not "".join(parser.title_text).strip():
            errors.append(f"{page}: 缺少 title")
        if SITE.joinpath("en").resolve() in page.resolve().parents:
            english_visible = raw_html.replace("中文", "")
            if re.search(r"[\u3400-\u9fff]", english_visible):
                errors.append(f"{page}: 英文页面仍残留非切换按钮中文本")
        if "MATHTOKEN" in raw_html or "MATHBLOCKTOKEN" in raw_html or "MATHINLINETOKEN" in raw_html:
            errors.append(f"{page}: 数学占位符未还原")
        for block in re.findall(r'<div class="math-display">(.*?)</div>', raw_html, flags=re.S):
            if re.search(r"<(?:em|strong|code)\b", block):
                errors.append(f"{page}: 块公式被 Markdown 标签破坏")
        for block in re.findall(r'<span class="math-inline">(.*?)</span>', raw_html, flags=re.S):
            if re.search(r"<(?:em|strong|code)\b", block):
                errors.append(f"{page}: 行内公式被 Markdown 标签破坏")

        for url in parser.hrefs + parser.sources:
            target, fragment = local_target(page, url)
            if target is None:
                continue
            if not target.exists():
                errors.append(f"{page}: 断链 {url} -> {target}")
                continue
            if fragment and target.suffix.lower() == ".html":
                target_parser = parsed.get(target.resolve())
                if target_parser is None:
                    target_parser = parse_page(target)
                    parsed[target.resolve()] = target_parser
                if fragment not in target_parser.ids:
                    errors.append(f"{page}: 锚点不存在 {url}")

    experiment_index = parsed.get((SITE / "experiments" / "index.html").resolve())
    if experiment_index and experiment_index.cards != 38:
        errors.append(f"实验索引卡片应为 38，实际 {experiment_index.cards}。")
    experiment_index_en = parsed.get((SITE / "en" / "experiments" / "index.html").resolve())
    if experiment_index_en and experiment_index_en.cards != 38:
        errors.append(f"英文实验索引卡片应为 38，实际 {experiment_index_en.cards}。")

    main_page = parsed.get((SITE / "index.html").resolve())
    if main_page:
        main_html = (SITE / "index.html").read_text(encoding="utf-8")
        if 'class="research-disclosure"' not in main_html:
            errors.append("主页缺少醒目的研究状态说明。")
        if "本站主文、研究证据总账和实验说明的具体文字均由 AI 撰写" not in main_html:
            errors.append("主页研究状态说明没有明确披露 AI 写作。")
        if "作者对该页进行了逐段审阅和深度修改" not in main_html:
            errors.append("主页没有说明作者对理论核心页的深度修改。")
        if 'class="core-entry-banner"' not in main_html or 'href="theory-core.html"' not in main_html:
            errors.append("主页缺少醒目的理论核心直达入口。")
        if len(main_page.images) != 4:
            errors.append(f"主文应有 4 张结果图，实际 {len(main_page.images)}。")
        experiment_links = {
            Path(urlsplit(url).path).name
            for url in main_page.hrefs
            if "experiments/e" in url or "post-upload-e" in url or "post-upload-peleg.html" in url
        }
        if len(experiment_links) != 38:
            errors.append(f"主文应链接全部 38 个编号实验页，实际 {len(experiment_links)}。")

    main_page_en = parsed.get((SITE / "en" / "index.html").resolve())
    if main_page_en:
        main_html_en = (SITE / "en" / "index.html").read_text(encoding="utf-8")
        if 'class="core-entry-banner"' not in main_html_en or 'href="theory-core.html"' not in main_html_en:
            errors.append("英文主页缺少醒目的理论核心直达入口。")
        experiment_links_en = {
            Path(urlsplit(url).path).name
            for url in main_page_en.hrefs
            if "experiments/e" in url or "post-upload-e" in url or "post-upload-peleg.html" in url
        }
        if len(experiment_links_en) != 38:
            errors.append(f"英文主文应链接全部 38 个编号实验页，实际 {len(experiment_links_en)}。")

    reference_page = parsed.get((SITE / "references.html").resolve())
    if reference_page:
        external_references = [url for url in reference_page.hrefs if url.startswith(("http://", "https://"))]
        if len(external_references) < 24:
            errors.append(f"参考文献页应至少含 24 个原文外链，实际 {len(external_references)}。")
    reference_page_en = parsed.get((SITE / "en" / "references.html").resolve())
    if reference_page_en:
        external_references = [url for url in reference_page_en.hrefs if url.startswith(("http://", "https://"))]
        if len(external_references) < 24:
            errors.append(f"英文参考文献页应至少含 24 个原文外链，实际 {len(external_references)}。")

    citation_minimums = {
        "index.html": 20,
        "evidence-ledger.html": 20,
        "theory-core.html": 15,
        "experiments/e01.html": 3,
        "experiments/e04.html": 1,
        "experiments/e05.html": 3,
        "experiments/e21.html": 1,
        "experiments/e22.html": 2,
        "en/index.html": 20,
        "en/evidence-ledger.html": 20,
        "en/theory-core.html": 15,
        "en/experiments/e01.html": 3,
        "en/experiments/e04.html": 1,
        "en/experiments/e05.html": 3,
        "en/experiments/e21.html": 1,
        "en/experiments/e22.html": 2,
    }
    for relative, minimum in citation_minimums.items():
        page = (SITE / relative).resolve()
        parser = parsed.get(page)
        if parser is None:
            continue
        external = [url for url in parser.hrefs if url.startswith(("http://", "https://"))]
        if len(external) < minimum:
            errors.append(f"{relative}: 外部原文引用至少应有 {minimum} 条，实际 {len(external)}。")

    for number in range(1, 31):
        page = (SITE / "experiments" / f"e{number:02d}.html").resolve()
        parser = parsed.get(page)
        if parser is None:
            errors.append(f"缺少实验页 E{number:02d}")
            continue
        required_ids = {"overview", "motivation", "results"}
        missing = required_ids.difference(parser.ids)
        if missing:
            errors.append(f"E{number:02d}: 缺少分节锚点 {sorted(missing)}")

        page_en = (SITE / "en" / "experiments" / f"e{number:02d}.html").resolve()
        parser_en = parsed.get(page_en)
        if parser_en is None:
            errors.append(f"缺少英文实验页 E{number:02d}")
            continue
        missing_en = required_ids.difference(parser_en.ids)
        if missing_en:
            errors.append(f"英文 E{number:02d}: 缺少分节锚点 {sorted(missing_en)}")

    post_upload_experiment_pages = [
        "post-upload-e31.html",
        "post-upload-e32.html",
        "post-upload-e33.html",
        "post-upload-e34.html",
        "post-upload-e35.html",
        "post-upload-peleg.html",
    ]
    for relative in post_upload_experiment_pages:
        if not (SITE / relative).exists():
            errors.append(f"缺少历史兼容实验页：{relative}")
        if not (SITE / "en" / relative).exists():
            errors.append(f"缺少英文历史兼容实验页：en/{relative}")

    canonical_increment_pages = [f"experiments/e{number:02d}.html" for number in range(31, 39)]
    for relative in canonical_increment_pages:
        if not (SITE / relative).exists():
            errors.append(f"缺少 canonical 实验页：{relative}")
        if not (SITE / "en" / relative).exists():
            errors.append(f"缺少英文 canonical 实验页：en/{relative}")

    for slug in ("e31", "e32", "e33", "e36", "e37"):
        for language in ("", "en"):
            path = (SITE / language / "experiments" / f"{slug}.html").resolve()
            parser = parsed.get(path)
            if parser is None or "update-20261027" not in parser.ids:
                errors.append(f"{path}: 缺少2026-10增补锚点。")

    for language in ("", "en"):
        path = (SITE / language / "theory-core.html").resolve()
        parser = parsed.get(path)
        if parser is None or "open-questions-20261027" not in parser.ids:
            errors.append(f"{path}: 缺少六个开放问题的入口。")

    language_pairs = [
        (SITE / "index.html", SITE / "en" / "index.html"),
        (SITE / "evidence-ledger.html", SITE / "en" / "evidence-ledger.html"),
        (SITE / "references.html", SITE / "en" / "references.html"),
        (SITE / "theory-core.html", SITE / "en" / "theory-core.html"),
        (SITE / "width-double-descent.html", SITE / "en" / "width-double-descent.html"),
        (SITE / "complexity-methods.html", SITE / "en" / "complexity-methods.html"),
        (SITE / "experiments" / "index.html", SITE / "en" / "experiments" / "index.html"),
        (SITE / "post-upload.html", SITE / "en" / "post-upload.html"),
        (SITE / "post-upload-mdl.html", SITE / "en" / "post-upload-mdl.html"),
        (SITE / "post-upload-language.html", SITE / "en" / "post-upload-language.html"),
        (SITE / "post-upload-measure.html", SITE / "en" / "post-upload-measure.html"),
        (SITE / "post-upload-theory-update.html", SITE / "en" / "post-upload-theory-update.html"),
        (SITE / "post-upload-estimator.html", SITE / "en" / "post-upload-estimator.html"),
        (SITE / "post-upload-peleg.html", SITE / "en" / "post-upload-peleg.html"),
        (SITE / "post-upload-e31.html", SITE / "en" / "post-upload-e31.html"),
        (SITE / "post-upload-e32.html", SITE / "en" / "post-upload-e32.html"),
        (SITE / "post-upload-e33.html", SITE / "en" / "post-upload-e33.html"),
        (SITE / "post-upload-e34.html", SITE / "en" / "post-upload-e34.html"),
        (SITE / "post-upload-e35.html", SITE / "en" / "post-upload-e35.html"),
        (SITE / "post-upload-decision.html", SITE / "en" / "post-upload-decision.html"),
    ]
    language_pairs.extend(
        (SITE / "experiments" / f"e{number:02d}.html", SITE / "en" / "experiments" / f"e{number:02d}.html")
        for number in range(1, 31)
    )
    language_pairs.extend(
        (SITE / "experiments" / f"e{number:02d}.html", SITE / "en" / "experiments" / f"e{number:02d}.html")
        for number in range(31, 39)
    )

    def html_targets(page: Path, parser: PageParser) -> set[Path]:
        targets: set[Path] = set()
        for url in parser.hrefs:
            target, _ = local_target(page, url)
            if target is not None and target.suffix.lower() == ".html":
                targets.add(target.resolve())
        return targets

    for chinese, english in language_pairs:
        chinese = chinese.resolve()
        english = english.resolve()
        zh_parser = parsed.get(chinese)
        en_parser = parsed.get(english)
        if zh_parser is None or en_parser is None:
            continue
        if zh_parser.document_language != "zh-CN":
            errors.append(f"中文页面 lang 错误：{chinese}")
        if en_parser.document_language != "en":
            errors.append(f"英文页面 lang 错误：{english}")
        zh_targets = html_targets(chinese, zh_parser)
        en_targets = html_targets(english, en_parser)
        if english not in zh_targets:
            errors.append(f"中文页面缺少对应英文切换：{chinese} -> {english}")
        if chinese not in en_targets:
            errors.append(f"英文页面缺少对应中文切换：{english} -> {chinese}")

        wrong_from_zh = [target for target in zh_targets if SITE.joinpath("en").resolve() in target.parents and target != english]
        if wrong_from_zh:
            errors.append(f"中文页面跳到错误英文页面：{chinese} -> {wrong_from_zh}")
        wrong_from_en = [target for target in en_targets if SITE.resolve() in target.parents and SITE.joinpath("en").resolve() not in target.parents and target != chinese]
        if wrong_from_en:
            errors.append(f"英文页面跳回错误中文页面：{english} -> {wrong_from_en}")

    def structure_signature(parser: PageParser) -> dict[str, int]:
        return {
            "H2": parser.h2_count,
            "H3": parser.h3_count,
            "表格": parser.tables,
            "图片": len(parser.images),
            "块公式": parser.math_display,
            "行内公式": parser.math_inline,
            "代码块": parser.pres,
        }

    def heading_number_sequence(parser: PageParser) -> list[str]:
        sequence: list[str] = []
        chinese_numerals = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
        }
        for _level, heading in parser.headings:
            code = re.match(r"\s*(E\d+|R\d+)(?=[\s:：)\]]|$)", heading)
            if code:
                sequence.append(code.group(1))
                continue
            number = re.match(r"\s*(\d+(?:\.\d+)*)(?=[\s.:：]|$)", heading)
            if number:
                sequence.append(number.group(1))
                continue
            numeral = re.match(r"\s*([一二三四五六七八九十]+)、", heading)
            if numeral and numeral.group(1) in chinese_numerals:
                sequence.append(chinese_numerals[numeral.group(1)])
        return sequence

    for chinese, english in language_pairs:
        zh_parser = parsed.get(chinese.resolve())
        en_parser = parsed.get(english.resolve())
        if zh_parser is None or en_parser is None:
            continue
        zh_signature = structure_signature(zh_parser)
        en_signature = structure_signature(en_parser)
        for label, zh_value in zh_signature.items():
            en_value = en_signature[label]
            if zh_value != en_value:
                errors.append(f"中英结构不一致（{label}）：{chinese.relative_to(SITE)}={zh_value}，{english.relative_to(SITE)}={en_value}")
        zh_numbers = heading_number_sequence(zh_parser)
        en_numbers = heading_number_sequence(en_parser)
        if zh_numbers != en_numbers:
            errors.append(
                f"中英标题编号序列不一致：{chinese.relative_to(SITE)}={zh_numbers}，{english.relative_to(SITE)}={en_numbers}"
            )

    if errors:
        print("Site validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validated {len(pages)} HTML pages: links, anchors, assets, cards, zh/en structural parity and formal-paper source parity all pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
