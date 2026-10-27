"""固定主矩阵配置；每种架构独立目录，不混接历史半池。"""

import frozen_memory as m
from config import WORK_DIR, DEVICE, RUN_HOURS


def configuration(task):
    hidden = {"base_one": (32,), "base_two": (32, 32), "base_three": (32, 32, 32)}[task]
    cfg = {
        "OUTPUT_DIR": WORK_DIR / task, "DEVICE": DEVICE, "INPUT_BITS": 8,
        "ARCHITECTURES": (hidden,), "COUNTS": tuple(range(16, 129, 16)),
        "DATASETS": 4, "POOLS": 4, "PARTICLES": 2048,
        "ADAPT_SWEEPS": 2, "MUTATION_SWEEPS": 12, "PRIMARY_WINDOW": (.002, .001),
        "TOTAL_SECONDS": RUN_HOURS * 3600, "SECONDS_PER_REPLICA": 900.,
        "KEEP_COMPLETED_CHECKPOINTS": True,
    }
    if task == "base_two":
        m.PROTOCOL = "memory_depth_night_v1:two_layer"
        for key in ("PRIOR_SEED", "RESAMPLE_SEED", "MUTATION_SEED"):
            cfg[key] = m.CONFIG[key] + 40_000_000
    return cfg


def run(task):
    if task == "calibration":
        from smc.run_calibration import run as calibration
        (WORK_DIR / "calibration").mkdir(parents=True, exist_ok=True)
        calibration({"device": DEVICE, "run_hours": RUN_HOURS, "source_queue": "E32_public_reproduction"})
    else:
        print(m.run(configuration(task)))

