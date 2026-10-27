"""增补包的本地路径、来源校验和只读配置；导入不会启动采样。"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_bundle():
    manifest = read_json(HERE / "SOURCE_MANIFEST.json")
    for relative, record in manifest["files"].items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE) or not path.is_file():
            raise RuntimeError(f"缺失或越界文件：{relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"来源校验失败：{relative}")
    return len(manifest["files"])


def load_fixed_core():
    name = "e37_fixed_radius_frozen"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "frozen/measure_diagnostic.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def fixed_config(device=None, hours=None, output=None):
    config = read_json(HERE / "fixed_radius_config.json")
    config["results_dir"] = str(checked_output(output or "runs/fixed_radius"))
    if device is not None:
        config["device"] = device
    if hours is not None:
        config["global_seconds"] = hours * 3600
    return load_fixed_core().normalise_config(config)


def checked_output(value):
    root = (HERE / "runs").resolve()
    target = (HERE / value).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("结果必须写入本增补包 runs/ 下的独立子目录。")
    return target
