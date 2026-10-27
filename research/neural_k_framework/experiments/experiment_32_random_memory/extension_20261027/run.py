"""通过顶部配置选择复现任务；默认不启动耗时计算。"""

import importlib
import json
import os
from pathlib import Path
import sys
import config

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "alignment"))

TASKS = {
    "review", "dictionary", "structure", "agreement", "agreement_analysis",
    "loss", "loss_analysis", "base_one", "base_two", "base_three",
    "enhanced", "fit", "calibration", "alignment", "minimax",
}


def main():
    task = config.TASK
    if task not in TASKS:
        raise ValueError(f"未知 TASK：{task}；可用：{sorted(TASKS)}")
    training = task in {"agreement", "loss", "fit"}
    smc = task in {"base_one", "base_two", "base_three", "enhanced", "calibration", "alignment"}
    if training and not config.ALLOW_TRAINING:
        raise SystemExit("训练默认禁用；须明确设置 ALLOW_TRAINING=True。")
    if smc and not config.ALLOW_SMC:
        raise SystemExit("SMC 默认禁用；须明确设置 ALLOW_SMC=True。")
    if task == "minimax" and not config.ALLOW_FULL_ENUMERATION:
        raise SystemExit("完整枚举默认禁用；须设置目录及 ALLOW_FULL_ENUMERATION=True。")
    if training or smc:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        if config.DEVICE != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("正式复现需要显式 CUDA，不自动回退 CPU。")
    from data import verify_data
    verify_data()
    if task == "review":
        from review import main as review
        review()
        return
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    for folder in ("structure", "agreement", "loss", "enhanced", "alignment", "minimax"):
        (config.WORK_DIR / folder).mkdir(parents=True, exist_ok=True)
    if task in {"base_one", "base_two", "base_three", "calibration"}:
        from smc.protocols import run
        run(task)
    elif task == "enhanced" or task == "fit":
        from smc import memory_followup as module
        runner = module.run_smc if task == "enhanced" else module.run_fit
        print(runner(module.ROOT, config.DEVICE, config.RUN_HOURS * 3600))
    else:
        name = {
            "dictionary": "structure.audit_data", "structure": "structure.audit",
            "agreement": "agreement.train_agreement", "agreement_analysis": "agreement.analyze_agreement",
            "loss": "loss.experiment", "loss_analysis": "loss.analyze",
            "alignment": "alignment.run_alignment", "minimax": "minimax.analyze",
        }[task]
        importlib.import_module(name).main()


if __name__ == "__main__":
    main()

