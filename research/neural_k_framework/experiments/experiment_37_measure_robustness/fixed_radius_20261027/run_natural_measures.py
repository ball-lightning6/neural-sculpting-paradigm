"""9 月旧面板的独立入口；顶部开关或 --run 可启用采样，默认只检查。"""

RUN_SAMPLING = False
BUNDLE_DIR = "."  # 复制到 Notebook 时指向完整复现目录；脚本模式使用自身目录。
DEVICE = "cuda:0"
MAX_SECONDS = 120.0
OUTPUT_DIR = "runs/natural_measures"

import argparse
import importlib
import json
from pathlib import Path
import sys
from unittest import mock

ENTRY_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
             else Path(BUNDLE_DIR).expanduser().resolve())
if not (ENTRY_DIR / "runtime.py").is_file():
    raise RuntimeError("请将 BUNDLE_DIR 指向含 runtime.py 的完整复现目录。")
if str(ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(ENTRY_DIR))
from runtime import HERE, checked_output, read_json, verify_bundle
if HERE != ENTRY_DIR:
    raise RuntimeError("已导入另一目录的 runtime；请重启 Notebook 内核并确认 BUNDLE_DIR。")


def load_historical_core():
    source = HERE / "historical_20260913/source"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return importlib.import_module("experiment_4bit_natural_measures")


def main(argv=None):
    # Notebook 的 -f 等内核参数不是本实验的命令行参数。
    if argv is None and ("__file__" not in globals() or "ipykernel" in sys.modules):
        argv = []
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", dest="run", action="store_true")
    mode.add_argument("--check", dest="run", action="store_false")
    parser.set_defaults(run=RUN_SAMPLING)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    verified = verify_bundle()
    historical = load_historical_core()
    panel = read_json(HERE / "historical_20260913/panel.json")
    config = read_json(HERE / "historical_20260913/config.json")
    output = checked_output(args.output)
    source_panel = HERE / "historical_20260913/source_panel"
    with mock.patch.object(sys, "argv", ["historical", "--out", str(output),
                                        "--source-panel", str(source_panel),
                                        "--device", args.device,
                                        "--max-seconds", str(args.max_seconds)]):
        parsed = historical.parse_args()
        for key, value in config["settings"].items():
            if key != "rules" and getattr(parsed, key) != value:
                raise RuntimeError(f"历史科学配置不一致：{key}")
        source = read_json(source_panel / "panel.json")
        ids = source["anchors"] + source["stratified_core"][::2] + [495, 10942, 12276, 2464]
        if ids != panel["function_ids"] or ids != config["settings"]["rules"]:
            raise RuntimeError("预选面板不一致。")
        print(json.dumps({"verified_files": verified, "date": "2026-09-13",
                          "rules": len(ids), "measure_rule_jobs": 60,
                          "sampling_requested": args.run,
                          "settings": config["settings"]}, ensure_ascii=False, indent=2))
        if args.run:
            historical.main()


if __name__ == "__main__":
    main()
