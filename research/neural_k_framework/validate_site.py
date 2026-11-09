from __future__ import annotations

import sys
import re
import filecmp
import json
import stat
import zipfile
import zlib
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
NOVEMBER_FILENAMES = {
    "research-updates.html", "interpretation-update.html",
    "update-geometry.html", "update-scaffolding.html", "update-consensus.html",
    "update-training-controls.html", "update-related-work.html",
}
NOVEMBER_ASSET_SUFFIXES = {".json", ".csv", ".py", ".txt", ".png", ".svg", ".md", ".zip"}
NOVEMBER_README_FILENAMES = {"README_zh.md", "README_en.md"}
NOVEMBER_ZIP_TIMESTAMP = (2026, 11, 6, 0, 0, 0)
# 英文句点不是小数位，hidden-16 的连字符不是负号；独立负数和科学计数法保留符号。
NUMBER_PATTERN = r"(?<![0-9.])(?:(?<![A-Za-z0-9_])[-+])?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


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


def html_targets(page: Path, parser: PageParser) -> set[Path]:
    targets: set[Path] = set()
    for url in parser.hrefs:
        target, _ = local_target(page, url)
        if target is not None and target.suffix.lower() == ".html":
            targets.add(target.resolve())
    return targets


def source_signature(source: str) -> dict[str, object]:
    from build_site import protect_math

    _, formulas = protect_math(source)
    table_rows = re.findall(r"(?m)^\|.*\|\s*$", source)
    return {
        "段落块": len(re.split(r"\n\s*\n", source.strip())),
        "标题层级": re.findall(r"(?m)^(#{1,6})\s+", source),
        "表格列数": [len(row.split("|")) for row in table_rows],
        "表格数字": [re.findall(NUMBER_PATTERN, row.replace("−", "-")) for row in table_rows],
        "图片数": len(re.findall(r"!\[[^\]]*\]\(", source)),
        "代码围栏": len(re.findall(r"(?m)^(?:```|~~~)", source)),
        "公式": [(kind, re.sub(r"\s+", "", formula)) for _, kind, formula in formulas],
    }


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
    from build_site import OCTOBER_EXTENSIONS, october_extension_paths

    source_pairs = [(slug, october_extension_paths(slug)) for slug in OCTOBER_EXTENSIONS]
    source_pairs.append(("AIT主线", (ROOT / "ait_connections_zh.md", ROOT / "ait_connections_en.md")))
    for slug, paths in source_pairs:
        if not all(path.is_file() for path in paths):
            errors.append(f"{slug}: 缺少本轮完整双语增补。")
            continue
        sources = [path.read_text(encoding="utf-8") for path in paths]
        for path, source in zip(paths, sources):
            if re.search(r"(?<![A-Za-z0-9_\\])[A-Za-z]:[\\/]", source):
                errors.append(f"{path}: 新公开源稿包含本机绝对路径。")
        signatures = [source_signature(source) for source in sources]
        for label, value in signatures[0].items():
            if value != signatures[1][label]:
                errors.append(f"{slug}新增中英源稿不对应：{label}。")


def validate_november_sources(errors: list[str]) -> None:
    """可独立运行的只读源稿检查，不需要生成站点。"""
    from build_site import NOVEMBER_PAGES, make_markdown, protect_math

    if len(NOVEMBER_PAGES) != 7 or {row[1] for row in NOVEMBER_PAGES} != NOVEMBER_FILENAMES:
        errors.append("NOVEMBER_PAGES 必须包含指定的7对页面，不能替代或新增E编号。")
    if len({row[0] for row in NOVEMBER_PAGES}) != len(NOVEMBER_PAGES):
        errors.append("NOVEMBER_PAGES 存在重复 source stem。")
    known_sources = {
        (ROOT / f"{stem}_{language}.md").resolve(): (filename, language)
        for stem, filename, *_ in NOVEMBER_PAGES for language in ("zh", "en")
    }
    for stem, filename, *_ in NOVEMBER_PAGES:
        paths = [ROOT / f"{stem}_{language}.md" for language in ("zh", "en")]
        if not all(path.is_file() for path in paths):
            errors.append(f"{filename}: 缺少11月完整双语源稿。")
            continue
        sources = [path.read_text(encoding="utf-8") for path in paths]
        signatures = []
        for language, path, source in zip(("zh", "en"), paths, sources):
            try:
                signature = source_signature(source)
                protected, _ = protect_math(source)
            except ValueError as error:
                errors.append(f"{path}: 数学源稿无法解析：{error}")
                signatures.append(None)
                continue
            if re.search(r"(?<![A-Za-z0-9_\\])[A-Za-z]:[\\/]", protected):
                errors.append(f"{path}: 新公开源稿包含本机绝对路径。")
            blocks = re.split(r"\n\s*\n", source.strip())
            # 译文可调整段内语序，但不得更改数字及其重复次数；表格仍按原序逐行核对。
            # 链接地址另行核验，译文路径中的版本目录名不计入正文数字。
            signature["逐段数字"] = [
                sorted(re.findall(NUMBER_PATTERN, re.sub(r"(?<=\])\([^)]*\)", "", block).replace("−", "-")))
                for block in blocks
            ]
            signature["标题编号"] = re.findall(
                r"(?m)^#{1,6}\s+((?:[ER]\d+)|\d+(?:\.\d+)*)(?=[\s.:：、)]|$)", source
            )
            parser = PageParser()
            parser.feed(make_markdown().convert(protected))
            external_links: Counter[str] = Counter()
            november_links: Counter[str] = Counter()
            for url in parser.hrefs + parser.sources:
                parts = urlsplit(url)
                if parts.scheme or parts.netloc:
                    if parts.scheme.lower() == "file":
                        errors.append(f"{path}: 源稿含本机文件链接：{url}")
                    external_links[url] += 1
                    continue
                if not parts.path:
                    if parts.fragment and unquote(parts.fragment) not in parser.ids:
                        errors.append(f"{path}: 源稿页内锚点不存在：{url}")
                    continue
                target = (path.parent / unquote(parts.path)).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    errors.append(f"{path}: 源稿链接到框架外未发布材料：{url}")
                    continue
                if not target.is_file():
                    errors.append(f"{path}: 源稿本地链接不存在：{url}")
                if target in known_sources:
                    target_filename, target_language = known_sources[target]
                    november_links[target_filename] += 1
                    if target_language != language and target_filename != filename:
                        errors.append(f"{path}: 源稿链接到错误语言的11月页面：{url}")
            signature["外部引用与资源"] = external_links
            signature["11月页面链接"] = november_links
            signatures.append(signature)
        if any(signature is None for signature in signatures):
            continue
        for label, value in signatures[0].items():
            if value == signatures[1][label]:
                continue
            if isinstance(value, list) and isinstance(signatures[1][label], list):
                other = signatures[1][label]
                positions = [str(index + 1) for index in range(max(len(value), len(other)))
                             if index >= len(value) or index >= len(other) or value[index] != other[index]]
                detail = f"，差异项 {', '.join(positions)}"
            else:
                detail = ""
            errors.append(f"{filename}: 11月中英源稿不对应：{label}{detail}。")


def validate_november_manifest(errors: list[str]) -> None:
    from build_site import NOVEMBER_EXPERIMENT_LINKS, NOVEMBER_PAGES

    path = SITE / "site-manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"无法读取站点 manifest：{error}")
        return
    if not isinstance(manifest, dict):
        errors.append("站点 manifest 必须为对象。")
        return
    experiments = manifest.get("experiments")
    expected_ids = {f"E{number:02d}" for number in range(1, 39)}
    if manifest.get("experiment_count") != 38:
        errors.append("manifest 的 experiment_count 必须仍为38。")
    if not isinstance(experiments, list) or not all(isinstance(item, dict) for item in experiments):
        errors.append("manifest 缺少合法实验清单。")
    else:
        ids = [item.get("id") for item in experiments]
        if len(ids) != 38 or sorted(str(value) for value in ids) != sorted(expected_ids):
            errors.append("manifest 实验编号必须恰为E01至E38，且各出现一次。")
        for item in experiments:
            code = item.get("id")
            if not isinstance(code, str) or code not in expected_ids:
                continue
            page = f"experiments/{code.lower()}.html"
            if item.get("page_zh") != page or item.get("page_en") != f"en/{page}":
                errors.append(f"manifest 的 {code} 中英文 canonical 路由错误。")
    update = manifest.get("november_2026_update")
    if not isinstance(update, dict):
        errors.append("manifest 缺少 november_2026_update 对象。")
        return
    expected = {
        "baseline": "3834fb3",
        "through": "2026-11-06",
        "entry": {"zh-CN": "research-updates.html", "en": "en/research-updates.html"},
        "pages": [{"source_stem": row[0], "zh-CN": row[1], "en": f"en/{row[1]}"} for row in NOVEMBER_PAGES],
        "experiment_links": {slug: list(targets) for slug, targets in NOVEMBER_EXPERIMENT_LINKS.items()},
        "new_experiment_ids": [],
    }
    for field, value in expected.items():
        if update.get(field) != value:
            errors.append(f"manifest 的 november_2026_update.{field} 与发布协议不符。")
    for slug, targets in NOVEMBER_EXPERIMENT_LINKS.items():
        if slug.upper() not in expected_ids or not set(targets).issubset(NOVEMBER_FILENAMES):
            errors.append(f"11月实验入口配置非法：{slug} -> {targets}")


def november_asset_allowed(path: Path) -> bool:
    if path.parent.name == "training_controls" and path.name == "endpoints.csv.gz":
        return True
    if path.name.startswith(".") or path.suffix.lower() not in NOVEMBER_ASSET_SUFFIXES:
        return False
    if path.suffix.lower() == ".md":
        return path.name in NOVEMBER_README_FILENAMES
    if path.suffix.lower() == ".zip":
        return path.name == "package.zip"
    return True


def validate_november_package(topic: Path, errors: list[str]) -> None:
    """只读检查平铺公开包，不解压、不生成文件。"""
    package = topic / "package.zip"
    if not package.is_file():
        errors.append(f"缺少11月专题下载包：{package}")
        return
    public_files = {path.name: path for path in topic.iterdir()
                    if path.is_file() and path.name != "package.zip"}
    missing_readmes = NOVEMBER_README_FILENAMES.difference(public_files)
    if missing_readmes:
        errors.append(f"{topic}: 下载包目录缺少双语README：{sorted(missing_readmes)}")
    for path in public_files.values():
        if not november_asset_allowed(path):
            errors.append(f"{topic}: 下载包目录含白名单外文件：{path.name}")
    try:
        with zipfile.ZipFile(package) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            if duplicates:
                errors.append(f"{package}: ZIP成员重名：{duplicates}")
            if Counter(names) != Counter(public_files.keys()):
                missing = sorted(set(public_files).difference(names))
                extra = sorted(set(names).difference(public_files))
                errors.append(f"{package}: ZIP成员与顶层公开文件不一致；缺少 {missing}，多出 {extra}。")
            for member in members:
                name = member.filename
                if (not name or name in {".", ".."} or any(char in name for char in "/\\:\x00")
                        or member.orig_filename != name or member.is_dir() or member.external_attr & 0x10):
                    errors.append(f"{package}: ZIP成员必须为无目录的文件名：{name!r}")
                    continue
                if name == "package.zip":
                    errors.append(f"{package}: ZIP不得包含自身。")
                    continue
                mode = stat.S_IFMT(member.external_attr >> 16)
                if mode not in (0, stat.S_IFREG):
                    errors.append(f"{package}: ZIP成员不是普通文件：{name}")
                    continue
                if member.flag_bits & 1:
                    errors.append(f"{package}: 公开ZIP成员不能加密：{name}")
                    continue
                if member.date_time != NOVEMBER_ZIP_TIMESTAMP:
                    errors.append(f"{package}: ZIP成员未使用固定时间戳 {NOVEMBER_ZIP_TIMESTAMP}：{name}")
                published = public_files.get(name)
                if published is None:
                    continue
                if member.file_size != published.stat().st_size:
                    errors.append(f"{package}: ZIP成员长度与公开文件不一致：{name}")
                    continue
                # 完整读取会同时校验ZIP的CRC；分块比较避免一次载入全部附件。
                with archive.open(member) as packed, published.open("rb") as original:
                    while True:
                        chunk = packed.read(1024 * 1024)
                        if chunk != original.read(1024 * 1024):
                            errors.append(f"{package}: ZIP成员内容与公开文件不一致：{name}")
                            break
                        if not chunk:
                            break
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error) as error:
        errors.append(f"无法完整核验专题ZIP {package}：{error}")


def validate_november_publication(errors: list[str], parsed: dict[Path, PageParser]) -> None:
    from build_site import NOVEMBER_EXPERIMENT_LINKS, NOVEMBER_PAGES, NOVEMBER_ROOT, protect_math, strip_first_h1

    expected_assets: set[Path] = set()
    expected_topics: set[Path] = set()
    asset_root = (SITE / "assets/updates_20261106").resolve()
    for stem, filename, *_ in NOVEMBER_PAGES:
        for language in ("zh", "en"):
            source = ROOT / f"{stem}_{language}.md"
            published = SITE / "source_docs" / language / f"{stem}_{language}.md"
            if not published.is_file():
                errors.append(f"缺少按stem路径公开的11月源稿：{published}")
            elif source.is_file() and not filecmp.cmp(source, published, shallow=False):
                errors.append(f"11月公开源稿与当前原稿不一致：{published}")
            output = SITE if language == "zh" else SITE / "en"
            page = (output / filename).resolve()
            parser = parsed.get(page)
            if parser is None:
                continue
            targets = html_targets(page, parser)
            for required in ("research-updates.html", "theory-core.html"):
                if (output / required).resolve() not in targets:
                    errors.append(f"{page}: 缺少同语言的 {required} 导航。")
            if source.is_file():
                try:
                    _, formulas = protect_math(strip_first_h1(source.read_text(encoding="utf-8")))
                except ValueError as error:
                    errors.append(f"{source}: 数学源稿无法解析：{error}")
                    continue
                expected_math = [(kind, re.sub(r"\s+", "", formula)) for _, kind, formula in formulas]
                raw_html = page.read_text(encoding="utf-8")
                actual_math = []
                for match in re.finditer(
                    r'<(?P<tag>div|span) class="math-(?P<kind>display|inline)">(?P<body>.*?)</(?P=tag)>',
                    raw_html, re.S,
                ):
                    kind, body = match["kind"], unescape(match["body"]).strip()
                    delimiter = "$$" if kind == "display" else "$"
                    if not body.startswith(delimiter) or not body.endswith(delimiter):
                        errors.append(f"{page}: 公式容器缺少匹配分隔符。")
                        continue
                    actual_math.append((kind, re.sub(r"\s+", "", body[len(delimiter):-len(delimiter)])))
                if actual_math != expected_math:
                    errors.append(f"{page}: HTML公式内容或顺序与源稿不一致。")
                counts = Counter(kind for kind, _ in expected_math)
                if parser.math_display != counts["display"] or parser.math_inline != counts["inline"]:
                    errors.append(f"{page}: HTML公式容器数量与源稿不一致。")
        directory = (ROOT / stem).parent
        if not directory.is_relative_to(NOVEMBER_ROOT):
            continue
        topic = asset_root / directory.relative_to(NOVEMBER_ROOT)
        expected_topics.add(topic)
        if not directory.is_dir():
            errors.append(f"缺少11月附件源目录：{directory}")
            continue
        # 仅检查专题根目录白名单文件，不进入暂停实验或临时目录。
        for source in directory.iterdir():
            # package.zip是生成产物，不要求源目录提供，也不复制源目录中的旧包。
            if source.name == "package.zip" or not november_asset_allowed(source) or not source.is_file():
                continue
            published = topic / source.name
            expected_assets.add(published.resolve())
            if source.stat().st_size > 20 * 1024 * 1024:
                errors.append(f"11月附件超出20MiB发布上限：{source}")
            if not published.is_file():
                errors.append(f"缺少11月公开附件：{published}")
            elif not filecmp.cmp(source, published, shallow=False):
                errors.append(f"11月公开附件与原文件不一致：{published}")
        package = (topic / "package.zip").resolve()
        expected_assets.add(package)
        validate_november_package(topic, errors)
        for language in ("", "en"):
            page = (SITE / language / filename).resolve()
            parser = parsed.get(page)
            if parser is None:
                continue
            downloads = {target.resolve() for url in parser.hrefs
                         if (target := local_target(page, url)[0]) is not None}
            if package not in downloads:
                errors.append(f"{page}: 缺少本专题package.zip下载链接。")
    if asset_root.is_dir():
        for path in asset_root.rglob("*"):
            resolved = path.resolve()
            if path.is_file() and resolved not in expected_assets:
                errors.append(f"11月附件包含白名单外文件：{path}")
            elif path.is_dir() and resolved not in expected_topics:
                errors.append(f"11月附件包含非专题根目录：{path}")
    for language in ("", "en"):
        output = SITE / language
        overview = (output / "research-updates.html").resolve()
        post_upload = (output / "post-upload.html").resolve()
        parser = parsed.get(post_upload)
        if parser is None or overview not in html_targets(post_upload, parser):
            errors.append(f"{post_upload}: 缺少同语言的11月进展入口。")
        overview_parser = parsed.get(overview)
        if overview_parser is not None:
            expected = {(output / name).resolve() for name in NOVEMBER_FILENAMES}
            missing = expected.difference(html_targets(overview, overview_parser))
            if missing:
                errors.append(f"{overview}: 缺少11月专题入口 {sorted(str(path) for path in missing)}")
        for slug, targets in NOVEMBER_EXPERIMENT_LINKS.items():
            page = (output / "experiments" / f"{slug}.html").resolve()
            parser = parsed.get(page)
            if parser is None or "update-20261106" not in parser.ids:
                errors.append(f"{page}: 缺少11月增补锚点。")
                continue
            expected = {(output / target).resolve() for target in targets}
            if not expected.issubset(html_targets(page, parser)):
                errors.append(f"{page}: 11月增补链接不完整或语言错误。")


def main() -> int:
    from build_site import NOVEMBER_PAGES

    errors: list[str] = []
    validate_paper_sources(errors)
    validate_math_protection(errors)
    validate_extension_sources(errors)
    validate_november_sources(errors)
    pages = sorted(SITE.rglob("*.html"))
    parsed = {page.resolve(): parse_page(page) for page in pages}

    if len(pages) != 132:
        errors.append(f"预期 132 个 HTML 页面，实际 {len(pages)} 个。")
    validate_november_manifest(errors)
    validate_november_publication(errors, parsed)

    required = [
        SITE / "index.html",
        SITE / "evidence-ledger.html",
        SITE / "references.html",
        SITE / "theory-core.html",
        SITE / "width-double-descent.html",
        SITE / "complexity-methods.html",
        SITE / "ait-connections.html",
        SITE / "experiments" / "index.html",
        SITE / "en" / "index.html",
        SITE / "en" / "evidence-ledger.html",
        SITE / "en" / "references.html",
        SITE / "en" / "theory-core.html",
        SITE / "en" / "width-double-descent.html",
        SITE / "en" / "complexity-methods.html",
        SITE / "en" / "ait-connections.html",
        SITE / "en" / "experiments" / "index.html",
    ]
    november_pages = {
        (SITE / language / row[1]).resolve() for row in NOVEMBER_PAGES for language in ("", "en")
    }
    required.extend(sorted(november_pages))
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
            if page.resolve() in november_pages and urlsplit(url).scheme.lower() == "file":
                errors.append(f"{page}: 新页面含本机文件链接 {url}")
                continue
            target, fragment = local_target(page, url)
            if target is None:
                continue
            if page.resolve() in november_pages and not target.resolve().is_relative_to(SITE.resolve()):
                errors.append(f"{page}: 新页面链接到站点外未发布材料 {url}")
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
        "ait-connections.html": 7,
        "experiments/e01.html": 3,
        "experiments/e04.html": 1,
        "experiments/e05.html": 3,
        "experiments/e21.html": 1,
        "experiments/e22.html": 2,
        "en/index.html": 20,
        "en/evidence-ledger.html": 20,
        "en/theory-core.html": 15,
        "en/ait-connections.html": 7,
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
        (SITE / "ait-connections.html", SITE / "en" / "ait-connections.html"),
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
    language_pairs.extend((SITE / row[1], SITE / "en" / row[1]) for row in NOVEMBER_PAGES)

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

    for language in ("", "en"):
        target = (SITE / language / "ait-connections.html").resolve()
        for filename in ("index.html", "theory-core.html", "post-upload.html"):
            page = (SITE / language / filename).resolve()
            parser = parsed.get(page)
            if parser is None or target not in html_targets(page, parser):
                errors.append(f"{page}: 缺少对应语言的AIT主线入口。")
        source = SITE / "source_docs" / ("zh" if not language else "en") / (
            "ait_connections_zh.md" if not language else "ait_connections_en.md"
        )
        if not source.is_file():
            errors.append(f"缺少公开AIT源稿：{source}")

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

    print(f"Validated {len(pages)} HTML pages: links, anchors, assets, 38 experiment IDs, zh/en source and HTML parity, mathematics and November manifest all pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
