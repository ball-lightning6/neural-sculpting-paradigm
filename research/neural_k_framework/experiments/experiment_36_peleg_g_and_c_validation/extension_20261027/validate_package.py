"""CPU 轻量核验：编译、导入、前向、数据清单和双语结构；不训练或采样。"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import py_compile
import re
import symtable

from portable import ROOT, load_backend, read_json, torch
import reproduce


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_files():
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "frozen").glob("*.py"))
    folder = (ROOT / "_compile_check").resolve()
    assert folder.is_relative_to(ROOT)
    folder.mkdir(exist_ok=False)
    generated = []
    try:
        for index, path in enumerate(paths):
            target = folder / f"{index}.pyc"
            py_compile.compile(str(path), cfile=str(target), doraise=True)
            generated.append(target)
    finally:
        for path in generated:
            path.unlink()
        folder.rmdir()
    return len(paths)


def check_selected_globals(backend):
    modules = [backend.broad, backend.stp, backend.wide, backend.cloud, backend.adam, backend.recheck,
               backend.anchor, backend.anchor_budget]
    checked = []
    for module in modules:
        tree = symtable.symtable(Path(module.__file__).read_text(encoding="utf-8-sig"), module.__file__, "exec")
        tables = {item.get_name(): item for item in tree.get_children()}
        for name, function in vars(module).items():
            if not inspect.isfunction(function):
                continue
            function = inspect.unwrap(function)
            if function.__globals__ is not vars(module):
                continue
            queue = [tables[name]]
            while queue:
                table = queue.pop()
                for symbol in table.get_symbols():
                    if symbol.is_global() and symbol.is_referenced():
                        key = symbol.get_name()
                        assert key in vars(module) or hasattr(builtins, key), (module.__name__, name, key)
                queue.extend(table.get_children())
            checked.append(module.__name__ + "." + name)
    return checked


def check_models(backend):
    torch.set_num_threads(2)
    errors = []
    generator = torch.Generator().manual_seed(20261027)
    x = torch.randn((3, 1, 28, 28), generator=generator)
    y = torch.tensor([0, 1, 0])
    for spec in backend.core.CONFIG["networks"]:
        model = backend.adapter.LegacyModel(spec)
        theta = backend.core.sample_prior(2, model.old, backend.core.CONFIG, generator, "cpu")
        with torch.no_grad():
            actual = model.logits(theta, x)
            expected = torch.stack([model.reference(row)(x) for row in theta])
            gap = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            loss, hard, _ = backend.core.energies(theta, x, y, model.old, backend.core.CONFIG)
            independent = torch.nn.functional.softplus(-(2 * y - 1) * expected).mean(1)
            torch.testing.assert_close(loss, independent, atol=2e-5, rtol=2e-5)
            assert not torch.count_nonzero(theta[:, model.old["biases"]])
        errors.append({"network": spec["name"], "parameters": model.size, "logit_max_abs_error": gap})
    zero = torch.zeros((1, model.size))
    with torch.no_grad():
        _, correct, barrier0 = backend.core.energies(zero, x[:1], torch.tensor([0]), model.old, backend.core.CONFIG)
        _, incorrect, barrier1 = backend.core.energies(zero, x[:1], torch.tensor([1]), model.old, backend.core.CONFIG)
    assert correct.item() and not incorrect.item() and barrier0.item() <= 0 < barrier1.item()
    return errors


def check_evidence(backend):
    manifest = read_json(ROOT / "EVIDENCE_MANIFEST.json")
    for item in manifest["public_files"]:
        path = (ROOT / item["file"]).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == item["sha256"], item["file"]
    cv = read_json(ROOT / "evidence/cv58.json")
    for protocol in read_json(ROOT / "config/cv_protocols.json").values():
        assert backend.core.scientific_hash(protocol["config"]) == protocol["scientific_sha256"]
        assert backend.core.source_hash() == protocol["source_sha256"]
    assert len(cv["original_static"]["rows"]) == 34 and len(cv["extension"]["rows"]) == 24
    completed = read_json(ROOT / "evidence/completion58.json")
    assert len(completed) == 58 and all(r["sgd"]["all_nodes_reached"] for r in completed)
    assert all(pool["status"] == "target_reached" for r in completed for pool in r["pools"])
    gc = read_json(ROOT / "evidence/direct_gc58.json")
    assert sum(r["record"]["accepted"] == 512 for r in gc) == 13
    assert sum(r["record"]["accepted"] == 0 for r in gc) == 45
    stp = read_json(ROOT / "evidence/stp48.json")
    assert len(stp["rows"]) == 240 and not stp["missing"]
    assert all(r["stp_models"] == r["planned"] == 128 for r in stp["rows"])
    deep = read_json(ROOT / "evidence/wide_deep.json")
    slow = [r for r in deep["rows"] if r["method"] == "sgd" and r["value"] == .0001]
    assert [r["models"] for r in slow if r["epsilon"] == 1e-5] == [0, 0]
    assert [r["models"] for r in slow if r["epsilon"] == 3e-5] == [58, 28]
    return {"public_files": len(manifest["public_files"]), "cv_conditions": 58,
            "direct_gc_sufficient": 13, "direct_gc_zero_hits": 45, "stp_groups": 48,
            "stp_nodes": 240, "wide_slow_sgd_at_1e_5": [0, 0]}


def block_type(block):
    if block.startswith("#"):
        return "heading:" + str(len(block) - len(block.lstrip("#")))
    for prefix, kind in (("```", "code"), ("$$", "formula"), ("|", "table"),
                         ("![", "image"), ("<a ", "anchor")):
        if block.startswith(prefix):
            return kind
    return "paragraph"


def check_documents():
    experiment = ROOT.parent
    framework = ROOT.parents[2]
    zh = (experiment / "VALIDATION_EXTENSION_20261027.md").read_text(encoding="utf-8")
    en = (framework / "translations_en/post_upload/e36_update_20261027.md").read_text(encoding="utf-8")
    blocks = [re.split(r"\n\s*\n", text.strip()) for text in (zh, en)]
    assert len(blocks[0]) == len(blocks[1]), [len(b) for b in blocks]
    numbers = lambda text: re.findall(r"[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    for index, (left, right) in enumerate(zip(*blocks), 1):
        kind = block_type(left)
        assert kind == block_type(right), (index, "块类型不一致")
        assert numbers(left) == numbers(right), (index, "数字不一致", numbers(left), numbers(right))
        assert re.findall(r"\$([^$\n]+)\$", left) == re.findall(r"\$([^$\n]+)\$", right), (index, "行内公式不一致")
        assert re.findall(r"`([^`\n]+)`", left) == re.findall(r"`([^`\n]+)`", right), (index, "代码标识不一致")
        if kind in ("formula", "code", "anchor"):
            assert left == right, (index, kind)
        if kind == "table":
            assert [len(line.split("|")) for line in left.splitlines()] == [len(line.split("|")) for line in right.splitlines()]
    link_pattern = r"!?\[[^\]]*\]\(([^)]+)\)"
    links = re.findall(link_pattern, zh)
    assert links == re.findall(link_pattern, en)
    assert '<a id="update-20261027"></a>' in zh
    for text in (zh, en):
        assert not re.search(r"[A-Za-z]:[/\\]|Downloads|publication_drafts|plugin://|\.zip\)", text)
    for link in links:
        path = (experiment / link).resolve()
        assert path.is_relative_to(framework), link
        assert path.exists() or path == ROOT / "VALIDATION.json", link
    counts = {}
    for block in blocks[0]:
        kind = block_type(block)
        counts[kind] = counts.get(kind, 0) + 1
    return {"blocks": len(blocks[0]), "types": counts, "links": len(links),
            "link_base": "original_experiment_directory", "anchor": "update-20261027"}


def check_figures():
    from PIL import Image
    import numpy as np
    rows = []
    for name in ("e36_ext_cv_mae.png", "e36_ext_stp_steps.png", "e36_ext_wide_boundary.png"):
        path = ROOT.parents[2] / "assets/figures" / name
        with Image.open(path) as picture:
            pixels = np.asarray(picture.convert("RGB"))
            nonwhite = float((pixels.min(axis=2) < 235).mean())
            assert min(picture.size) > 400 and nonwhite > .02
            rows.append({"file": name, "size": list(picture.size), "nonwhite_fraction": nonwhite})
    return rows


def main():
    compiled = compile_files()
    assert reproduce.TASK == "inspect" and reproduce.RUN_COMPUTE is False
    backend = load_backend()
    result = {
        "date": "2026-10-27", "status": "passed", "training_started": False,
        "smc_started": False, "download_started": False, "compiled_python_files": compiled,
        "frozen_source_files": len(backend.sources), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "torch", "torchvision", "matplotlib", "pillow", "sympy")},
        "selected_function_globals_checked": check_selected_globals(backend),
        "forward_checks": check_models(backend), "evidence": check_evidence(backend),
        "bilingual": check_documents(), "figures": check_figures(),
        "limits": ["未运行优化或SMC；前向检查只使用小型合成输入", "未认证全局混合或长任务显存预算",
                   "GPU虚拟环境启动器的_ctypes错误未修复；本轮由同版本系统Python加载现有依赖核验"],
    }
    (ROOT / "VALIDATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "compiled_python_files", "frozen_source_files", "bilingual")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
