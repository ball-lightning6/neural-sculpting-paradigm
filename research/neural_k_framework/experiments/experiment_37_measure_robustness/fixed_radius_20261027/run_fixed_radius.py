"""默认仅核验配置；顶部开关或 --run 可显式启用历史 24 池协议。"""

RUN_SAMPLING = False
BUNDLE_DIR = "."  # 复制到 Notebook 时指向完整复现目录；脚本模式使用自身目录。
DEVICE = "cuda:0"
RUN_HOURS = 4.0
OUTPUT_DIR = "runs/fixed_radius"

import argparse
import json
from pathlib import Path
import sys

ENTRY_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
             else Path(BUNDLE_DIR).expanduser().resolve())
if not (ENTRY_DIR / "runtime.py").is_file():
    raise RuntimeError("请将 BUNDLE_DIR 指向含 runtime.py 的完整复现目录。")
if str(ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(ENTRY_DIR))
from runtime import HERE, fixed_config, load_fixed_core, verify_bundle
if HERE != ENTRY_DIR:
    raise RuntimeError("已导入另一目录的 runtime；请重启 Notebook 内核并确认 BUNDLE_DIR。")


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
    parser.add_argument("--hours", type=float, default=RUN_HOURS)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    verified = verify_bundle()
    core = load_fixed_core()
    config = fixed_config(args.device, args.hours, args.output)
    jobs = core.make_jobs(config)
    if len(jobs) != 24:
        raise RuntimeError("公开协议必须恰有 24 池。")
    print(json.dumps({"verified_files": verified, "protocol": core.PROTOCOL,
                      "pools": len(jobs), "sampling_requested": args.run,
                      "config": config}, ensure_ascii=False, indent=2))
    if args.run:
        core.report_outcome(core.run(config))


if __name__ == "__main__":
    main()
