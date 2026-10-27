"""检查发布副本的编译、导入、双语块结构和公开链接；不执行实验入口。"""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import py_compile
import re
import sys


ROOT = Path(__file__).resolve().parent
FRAMEWORK = ROOT.parents[2]
DOCUMENTS = {
    "experiment_31_loss_amplitude_grokking": ("XOR_OR_ALIGNMENT_20261027.md", "e31_update_20261027.md"),
    "experiment_33_cross_architecture": ("COMPLEXITY_CALIBRATION_20261027.md", "e33_update_20261027.md"),
}


def block_kind(block):
    for prefix, kind in (("#", "heading"), ("```", "code"), ("\\[", "math"), ("|", "table"), ("![", "image"), ("<a ", "anchor")):
        if block.startswith(prefix):
            return kind
    return "paragraph"


def numbers(block):
    return re.findall(r"[+-]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)", block)


def main():
    source_name, translation_name = DOCUMENTS[ROOT.parent.name]
    zh = ROOT.parent / source_name
    en = FRAMEWORK / "translations_en/post_upload" / translation_name
    texts = [path.read_text(encoding="utf-8") for path in (zh, en)]
    blocks = [[part.strip() for part in re.split(r"\n\s*\n", text.strip())] for text in texts]
    if len(blocks[0]) != len(blocks[1]):
        raise AssertionError(f"双语块数量不同：{len(blocks[0])} / {len(blocks[1])}")
    for index, (a, b) in enumerate(zip(*blocks), 1):
        kind = block_kind(a)
        if kind != block_kind(b):
            raise AssertionError(f"第{index}块类型不同")
        if kind in ("code", "math", "anchor") and a != b:
            raise AssertionError(f"第{index}块的代码、公式或锚点不对应")
        if kind == "heading" and len(a) - len(a.lstrip("#")) != len(b) - len(b.lstrip("#")):
            raise AssertionError(f"第{index}块标题层级不同")
        if kind == "table":
            rows_a, rows_b = a.splitlines(), b.splitlines()
            if len(rows_a) != len(rows_b):
                raise AssertionError(f"第{index}块表格行数不同")
            for row_a, row_b in zip(rows_a, rows_b):
                if row_a.count("|") != row_b.count("|") or numbers(row_a) != numbers(row_b):
                    raise AssertionError(f"第{index}块表格列或数字不对应：{row_a}")
        if sorted(numbers(a)) != sorted(numbers(b)):
            raise AssertionError(f"第{index}块数字不对应：{numbers(a)} / {numbers(b)}")
    targets = [re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text) for text in texts]
    if targets[0] != targets[1]:
        raise AssertionError("双语链接目标不对应")
    for target in targets[0]:
        if re.match(r"(?:[A-Za-z]:|https?://|file:)", target):
            raise AssertionError("不允许私有或外部链接")
        resolved = (ROOT.parent / target.split("#")[0]).resolve()
        if not resolved.is_relative_to(FRAMEWORK) or not resolved.exists():
            raise AssertionError(f"链接不可公开解析：{target}")
    if any(text.count('id="update-20261027"') != 1 for text in texts):
        raise AssertionError("补充锚点不唯一")
    sys.path.insert(0, str(ROOT))
    modules = []
    imports = set()
    for path in sorted(ROOT.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        if path.name != Path(__file__).name:
            importlib.import_module(path.stem)
        modules.append(path.name)
    provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8-sig"))
    for row in provenance["data_files"]:
        if hashlib.sha256((ROOT / "data" / row["file"]).read_bytes()).hexdigest() != row["sha256"]:
            raise AssertionError(f"数据hash变化：{row['file']}")
    for name in ("run_experiment.py", "bridge_diagnostic.py", "verify_bridge.py", "test_experiment.py", "exact_circuits.py", "test_exact_circuits.py", "verify_small_world.py"):
        if (ROOT / name).exists():
            source = next(r for r in provenance["source_files"] if r["source"].endswith("/" + name))
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != source["sha256"]:
                raise AssertionError(f"冻结源码变化：{name}")
    result = {"passed": True, "compiled_and_imported": modules, "imports": sorted(imports),
              "bilingual_blocks": len(blocks[0]), "block_counts": {kind: sum(block_kind(b) == kind for b in blocks[0]) for kind in sorted({block_kind(b) for b in blocks[0]})},
              "public_links": len(targets[0]), "source_data_hashes_verified": True,
              "scope": "仅编译、导入和静态文档/数据核验；没有调用训练、SMC或证书求解入口。"}
    (ROOT / "release_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
