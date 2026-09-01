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

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_text.append(data)


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


def main() -> int:
    errors: list[str] = []
    pages = sorted(SITE.rglob("*.html"))
    parsed = {page.resolve(): parse_page(page) for page in pages}

    if len(pages) != 70:
        errors.append(f"预期 70 个 HTML 页面，实际 {len(pages)} 个。")

    required = [
        SITE / "index.html",
        SITE / "evidence-ledger.html",
        SITE / "references.html",
        SITE / "theory-core.html",
        SITE / "experiments" / "index.html",
        SITE / "en" / "index.html",
        SITE / "en" / "evidence-ledger.html",
        SITE / "en" / "references.html",
        SITE / "en" / "theory-core.html",
        SITE / "en" / "experiments" / "index.html",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"缺少页面：{path}")

    for page in pages:
        parser = parsed[page.resolve()]
        raw_html = page.read_text(encoding="utf-8")
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
    if experiment_index and experiment_index.cards != 30:
        errors.append(f"实验索引卡片应为 30，实际 {experiment_index.cards}。")
    experiment_index_en = parsed.get((SITE / "en" / "experiments" / "index.html").resolve())
    if experiment_index_en and experiment_index_en.cards != 30:
        errors.append(f"英文实验索引卡片应为 30，实际 {experiment_index_en.cards}。")

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
        experiment_links = {Path(urlsplit(url).path).name for url in main_page.hrefs if "experiments/e" in url}
        if len(experiment_links) != 30:
            errors.append(f"主文应链接全部 30 个实验页，实际 {len(experiment_links)}。")

    main_page_en = parsed.get((SITE / "en" / "index.html").resolve())
    if main_page_en:
        main_html_en = (SITE / "en" / "index.html").read_text(encoding="utf-8")
        if 'class="core-entry-banner"' not in main_html_en or 'href="theory-core.html"' not in main_html_en:
            errors.append("英文主页缺少醒目的理论核心直达入口。")

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

    language_pairs = [
        (SITE / "index.html", SITE / "en" / "index.html"),
        (SITE / "evidence-ledger.html", SITE / "en" / "evidence-ledger.html"),
        (SITE / "references.html", SITE / "en" / "references.html"),
        (SITE / "theory-core.html", SITE / "en" / "theory-core.html"),
        (SITE / "experiments" / "index.html", SITE / "en" / "experiments" / "index.html"),
    ]
    language_pairs.extend(
        (SITE / "experiments" / f"e{number:02d}.html", SITE / "en" / "experiments" / f"e{number:02d}.html")
        for number in range(1, 31)
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

    if errors:
        print("Site validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validated {len(pages)} HTML pages, all local links, anchors, assets and experiment cards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
