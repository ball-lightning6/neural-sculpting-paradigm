"""按原 E37 目录解析中英文链接，并检查拼接块、公式、表格与代码一致性。"""

from collections import Counter
import hashlib
import json
import re

from runtime import HERE, read_json

EXPERIMENT = HERE.parent
FRAMEWORK = HERE.parents[2]


def classify(block):
    if block.startswith("#"):
        return "heading"
    if block.startswith("|"):
        return "table"
    if block.startswith("$$"):
        return "formula"
    if block.startswith("```"):
        return "code"
    if block.startswith("!["):
        return "image"
    if block.startswith("<a "):
        return "anchor"
    return "paragraph"


def numbers(text):
    # 链接文字允许翻译，链接目标另行逐一检查。
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.findall(r"0x[0-9A-Fa-f]+|\d+(?:\.\d+)*(?:e[-+]?\d+)?", text)


def check():
    paths = [EXPERIMENT / "FIXED_RADIUS_20261027.md",
             FRAMEWORK / "translations_en/post_upload/e37_update_20261027.md"]
    documents = [path.read_text(encoding="utf-8") for path in paths]
    blocks = [re.split(r"\n\s*\n", doc.strip()) for doc in documents]
    if len(blocks[0]) != len(blocks[1]):
        raise ValueError(f"段落块数量不同：{list(map(len, blocks))}")
    warnings = []
    for index, (zh, en) in enumerate(zip(*blocks), 1):
        kind = classify(zh)
        if kind != classify(en):
            raise ValueError(f"第 {index} 块类型不同。")
        if kind in {"formula", "code", "anchor"} and zh != en:
            raise ValueError(f"第 {index} 块 {kind} 不一致。")
        if kind == "heading":
            if len(zh)-len(zh.lstrip("#")) != len(en)-len(en.lstrip("#")) or numbers(zh) != numbers(en):
                raise ValueError(f"第 {index} 块标题层级或编号不一致。")
        if kind == "table":
            zr, er = zh.splitlines(), en.splitlines()
            if len(zr) != len(er):
                raise ValueError(f"第 {index} 块表格行数不同。")
            for a, b in zip(zr, er):
                if a.count("|") != b.count("|") or numbers(a) != numbers(b):
                    raise ValueError(f"第 {index} 块表格列数或数值不一致：{a} / {b}")
        if kind == "paragraph" and Counter(numbers(zh)) != Counter(numbers(en)):
            warnings.append({"block": index, "zh": numbers(zh), "en": numbers(en)})
    links = [re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc) for doc in documents]
    if links[0] != links[1]:
        raise ValueError("中英文链接目标或顺序不同。")
    for target in links[0]:
        if re.match(r"(?:[A-Za-z]:|/|https?://|file:)", target):
            raise ValueError(f"不是实验内相对链接：{target}")
        path = (EXPERIMENT / target.split("#")[0]).resolve()
        if not path.is_relative_to(FRAMEWORK) or not path.exists():
            raise ValueError(f"公开落点缺失或越界：{target}")
    images = [re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc) for doc in documents]
    if images[0] != images[1] or len(images[0]) != 2:
        raise ValueError("图像数量或目标不一致。")
    if any(doc.count('id="update-20261027"') != 1 for doc in documents):
        raise ValueError("统一锚点缺失或重复。")
    if warnings:
        raise ValueError(json.dumps({"paragraph_number_differences": warnings}, ensure_ascii=False))
    legacy = read_json(HERE / "SOURCE_MANIFEST.json")["legacy_e37_files"]
    for name, expected in legacy.items():
        if hashlib.sha256((EXPERIMENT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"旧 E37 文件 hash 发生变化：{name}")
    return {"blocks_per_language": len(blocks[0]),
            "block_types": dict(Counter(classify(b) for b in blocks[0])),
            "links_per_language": len(links[0]), "images_per_language": len(images[0]),
            "anchor": "update-20261027", "link_base": "experiment_37_measure_robustness",
            "legacy_e37_files_unchanged": len(legacy),
            "paragraph_number_multisets_match": True, "passed": True}


if __name__ == "__main__":
    report = check()
    output = HERE / "validation"
    output.mkdir(exist_ok=True)
    (output / "PUBLICATION_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
