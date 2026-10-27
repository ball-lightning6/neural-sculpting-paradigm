"""核验双语增补的块序、结构、数值、公式、代码及公开相对链接。"""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
FRAMEWORK = ROOT.parents[2]
ZH = EXPERIMENT / "MEMORY_EXTENSION_20261027.md"
EN = FRAMEWORK / "translations_en/post_upload/e32_update_20261027.md"


def kind(block):
    if block.startswith("#"):
        return "heading", len(block.split(" ", 1)[0])
    if block.startswith("|"):
        return "table", len(block.splitlines())
    if block.startswith("!["):
        return "image", 0
    if block.startswith("\\["):
        return "formula", 0
    if block.startswith("```"):
        return "code", 0
    if block.startswith("<a"):
        return "anchor", 0
    return "paragraph", 0


def check():
    texts = [p.read_text(encoding="utf-8") for p in (ZH, EN)]
    blocks = [re.split(r"\n\s*\n", s.strip()) for s in texts]
    if len(blocks[0]) != len(blocks[1]):
        raise AssertionError("双语块数量不一致")
    numbers = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    for index, (zh, en) in enumerate(zip(*blocks), 1):
        assert kind(zh) == kind(en), (index, "结构不一致")
        category = kind(zh)[0]
        if category in ("formula", "code", "anchor"):
            assert zh == en, (index, "公式/代码/锚点不一致")
        # 屏蔽相同链接后比较数字，保留每个段落中的顺序。
        clean = [re.sub(r"\]\([^)]*\)", "]", s) for s in (zh, en)]
        assert re.findall(numbers, clean[0]) == re.findall(numbers, clean[1]), (
            index, re.findall(numbers, clean[0]), re.findall(numbers, clean[1]))
        links = [re.findall(r"!?\[[^]]*\]\(([^)]+)\)", s) for s in (zh, en)]
        assert links[0] == links[1], (index, "链接不一致")
        for link in links[0]:
            assert not re.match(r"[A-Za-z]:|https?://|file:", link), link
            assert (EXPERIMENT / link).resolve().exists(), link
        if category == "table":
            assert [line.count("|") for line in zh.splitlines()] == [line.count("|") for line in en.splitlines()]
    return {"paired_blocks": len(blocks[0]), "images": texts[0].count("!["), "anchor": "update-20261027"}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False))

