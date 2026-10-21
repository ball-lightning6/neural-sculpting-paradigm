"""4-bit constrained-SMC 宽度×深度两轴扫描驱动（本机 3080 10GB 版）。

修改同目录数值实现的 Config 后调用其 main()，每格独立 RESULT_DIR、
独立打包、单格失败不拖死整体、
RESUME=True。

两组扫描（公共格 W256D2 只在深度组跑一次）：
- 深度对照：W=256 固定，HIDDEN_LAYERS ∈ {1,2}；本轮不跑3、4层
- 宽度扫描：HIDDEN_LAYERS=2 固定，W ∈ {64,128,256,512}

协议阶梯（每格跑前用主模块 estimate_memory_bytes 做 preflight 估算，选
峰值估计 ≤ 7GB 的最高档）：
    full            = 8 replicas × 4096 particles
    reduced_4x2048  = 4 replicas × 2048 particles
    reduced_4x1024  = 4 replicas × 1024 particles
所选协议写进该格 config.json 的 PROTOCOL_NOTE（"full" 或
"reduced_4x2048 因显存" 等）。W>=256 额外把 EVAL_MICRO_BATCH 降到 2048。

NSP_SMOKE_TEST=1 时为 dry run：只打印每格 preflight 估算与所选协议，
不真跑 SMC。
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parent / "experiment_4bit_smc_width_depth.py"
)
OUT_ROOT = Path(__file__).resolve().parent / "results_4bit_smc_width_depth"
# (扫描组, WIDTH, HIDDEN_LAYERS)；公共格 (256,2) 只列一次
# 用户指示：深度组 D=3/4 先不跑，只留宽度组
CELLS = [
    ("depth", 256, 1),
    ("depth", 256, 2),
    ("width", 64, 2),
    ("width", 128, 2),
    ("width", 512, 2),
]

PROTOCOL_LADDER = [
    ("full", 8, 4_096),
    ("reduced_4x2048", 4, 2_048),
    ("reduced_4x1024", 4, 1_024),
]
GPU_PEAK_BUDGET_BYTES = 7 * 1024**3
DRY_RUN = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


def select_protocol(mod, width: int, hidden_layers: int) -> tuple[str, int, int, dict]:
    """按峰值估计 ≤ 7GB 选最高档；都不满足时退回最低档并在 note 里标注。"""
    eval_micro_batch = 2_048 if width >= 256 else mod.Config.EVAL_MICRO_BATCH
    fallback = None
    for name, replicas, particles in PROTOCOL_LADDER:
        report = mod.estimate_memory_bytes(
            width=width,
            hidden_layers=hidden_layers,
            replicas=replicas,
            particles_per_replica=particles,
            eval_micro_batch=eval_micro_batch,
        )
        fallback = (name, replicas, particles, report)
        if report["peak_estimate_bytes"] <= GPU_PEAK_BUDGET_BYTES:
            return name, replicas, particles, report
    name, replicas, particles, report = fallback
    return name, replicas, particles, report


def main() -> None:
    spec = importlib.util.spec_from_file_location("wd_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wd_mod"] = mod  # dataclass 处理需要模块已注册
    spec.loader.exec_module(mod)

    # 每格跑前恢复默认，防上一格的 override 泄漏
    defaults = {
        key: getattr(mod.Config, key)
        for key in (
            "WIDTH",
            "HIDDEN_LAYERS",
            "RESULT_DIR",
            "DEVICE",
            "REPLICAS",
            "PARTICLES_PER_REPLICA",
            "EVAL_MICRO_BATCH",
            "PROTOCOL_NOTE",
        )
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for group, width, hidden_layers in CELLS:
        for key, value in defaults.items():
            setattr(mod.Config, key, value)
        mod.Config.WIDTH = width
        mod.Config.HIDDEN_LAYERS = hidden_layers
        mod.Config.RESULT_DIR = OUT_ROOT / f"w{width}_d{hidden_layers}"
        mod.Config.DEVICE = (
            "cuda:1"
            if __import__("torch").cuda.device_count() > 1
            else "cuda:0"
        )
        mod.Config.RESUME = True
        mod.Config.OVERWRITE_RESULT_DIR = False
        mod.Config.PACKAGE_RESULTS = True
        if width >= 256:
            mod.Config.EVAL_MICRO_BATCH = 2_048

        name, replicas, particles, report = select_protocol(
            mod, width, hidden_layers
        )
        mod.Config.REPLICAS = replicas
        mod.Config.PARTICLES_PER_REPLICA = particles
        fits = report["peak_estimate_bytes"] <= GPU_PEAK_BUDGET_BYTES
        mod.Config.PROTOCOL_NOTE = (
            "full" if name == "full"
            else f"{name} 因显存" + ("" if fits else "(仍超7GB预算，谨慎)")
        )
        print(
            f"[{group}] W={width} D={hidden_layers} | "
            f"params={report['parameter_count']:,} | "
            f"池子={mod.format_gb(report['pool_bytes'])} | "
            f"变异工作集={mod.format_gb(report['mutate_extra_bytes'])} | "
            f"评估激活={mod.format_gb(report['eval_activation_bytes'])} | "
            f"峰值≈{mod.format_gb(report['peak_estimate_bytes'])} | "
            f"协议={mod.Config.PROTOCOL_NOTE} "
            f"({replicas}x{particles})",
            flush=True,
        )
        if DRY_RUN:
            continue

        print(
            f"\n########## [{group}] W={width} D={hidden_layers} 开始 "
            f"{time.strftime('%H:%M:%S')} ##########",
            flush=True,
        )
        t0 = time.time()
        try:
            mod.main()
        except Exception as exc:  # 单格失败不拖死整个扫描，保留已完成数据
            print(
                f"########## [{group}] W={width} D={hidden_layers} "
                f"失败：{exc} ##########",
                flush=True,
            )
            continue
        print(
            f"########## [{group}] W={width} D={hidden_layers} 完成，用时 "
            f"{(time.time()-t0)/60:.1f} 分钟 ##########",
            flush=True,
        )


if __name__ == "__main__":
    main()
