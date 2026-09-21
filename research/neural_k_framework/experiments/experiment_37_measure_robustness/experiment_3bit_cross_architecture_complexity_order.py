"""3-bit -> 1-bit 全函数复杂度排序的真实有限宽跨架构实验。

这个实验不使用NNGP代替有限网络。对每个架构执行两种直接测量：

1. 从该架构的Gaussian初始化分布直接抽取有限宽网络，统计256个完整函数的
   先验概率；
2. 对40个严格对称函数轨道分别运行prior-preserving pCN constrained SMC，
   测量完整真值表BCE不断降低时的绝对参数质量和Neural K曲线。

默认比较四个网络，分别改变宽度、深度和激活函数：

* ``3 -> 16 -> 1 tanh``；
* ``3 -> 32 -> 1 tanh``；
* ``3 -> 16 -> 16 -> 1 tanh``；
* ``3 -> 16 -> 1 relu``。

所有权重和bias都在标准Gaussian神经坐标中抽取，进入每层前按
``1/sqrt(fan_in)``缩放。输入固定为0/1编码，因此严格可合并的对称性只有输入
变量置换和输出整体取反，不能合并输入bit取反。

完成采样后，脚本自动比较：

* 不同有限宽架构的先验函数排序；
* 不同有限宽架构在各个loss阈值下的Neural K排序；
* 先验排序怎样随loss下降保持或重排；
* Neural K与Lempel-Ziv、Boolean表达式、决策树、ANF、输入敏感度和
  Walsh谱复杂度的关系；
* 全部256个函数与固定4个0、4个1的平衡函数，避免相关性只由标签比例造成。

运行：

    python experiment_3bit_cross_architecture_complexity_order.py

快速检查：

    python experiment_3bit_cross_architecture_complexity_order.py --smoke

脚本支持Ctrl+C保存和断点续跑，也可用``--architecture-start/stop``与
``--target-start/stop``分批执行。切片均为Python半开区间。
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import itertools
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    hidden_widths: tuple[int, ...]
    activation: str

    @property
    def display_name(self) -> str:
        hidden = "->".join(str(width) for width in self.hidden_widths)
        return f"3->{hidden}->1 {self.activation}"


ARCHITECTURES = (
    ArchitectureSpec("tanh_w16_d1", (16,), "tanh"),
    ArchitectureSpec("tanh_w32_d1", (32,), "tanh"),
    ArchitectureSpec("tanh_w16_d2", (16, 16), "tanh"),
    ArchitectureSpec("relu_w16_d1", (16,), "relu"),
)

ACTIVE_ARCHITECTURE = ARCHITECTURES[0]
ACTIVE_ARCHITECTURE_INDEX = 0

# PyBDM 0.1.0（https://github.com/sztal/pybdm，MIT）的CTM-B2-D12
# 数据库对全部8-bit字符串给出的CTM值。
# 这里只嵌入本实验需要的256个数，避免运行端额外下载约40 MB数据库。
# function_id的第i位对应真值表第i行，和function_outputs保持一致。
CTM_B2_D12_LENGTH8 = (
    18.527390041484, 19.625216134798, 19.922535674259, 20.740506282542, 20.430511751848, 21.029362235052, 20.909865233072, 21.749226191372,
    19.914020154082, 21.212230212422, 20.879638193258, 21.992170903699, 21.123839319475, 21.385626171451, 21.619450800321, 21.899016998869,
    19.914020154082, 20.898231922780, 21.079919431187, 22.242885700339, 21.312367112387, 21.074909644102, 22.183658668804, 22.679752296797,
    21.302504741813, 22.027864184497, 21.667334844418, 22.111810174929, 22.241428828694, 22.169771791078, 22.101088234553, 21.749226191372,
    20.430511751848, 21.173477342908, 21.037507871441, 22.121730115823, 20.848608927869, 21.225130372488, 21.735681644149, 22.111810174929,
    21.312367112387, 21.905658143158, 20.395154592852, 22.067813587727, 22.027327627276, 21.576570705832, 22.128561804940, 21.992170903699,
    21.123839319475, 22.071028206812, 21.795008189113, 22.616243867598, 22.027327627276, 21.714433807925, 21.552507909898, 22.242885700339,
    22.241428828694, 22.533640059552, 22.003006945849, 22.121730115823, 22.467484907100, 22.030722028485, 22.164133924836, 20.740506282542,
    19.922535674259, 21.279742874309, 20.837796018606, 22.030722028485, 21.037507871441, 21.247815094663, 21.762765127248, 22.169771791078,
    21.079919431187, 20.998446101321, 20.426024537110, 21.576570705832, 21.795008189113, 21.368127807977, 21.824255377951, 21.385626171451,
    20.879638193258, 21.371215591830, 20.426024537110, 21.714433807925, 20.395154592852, 19.836183398903, 20.713091342300, 21.074909644102,
    21.667334844418, 22.025665566915, 20.781825244612, 21.225130372488, 22.003006945849, 21.247815094663, 21.813336017509, 21.029362235052,
    20.909865233072, 22.159340776707, 21.762765127248, 22.533640059552, 21.735681644149, 22.025665566915, 21.898526276251, 22.027864184497,
    22.183658668804, 22.110979951554, 20.713091342300, 21.905658143158, 21.552507909898, 20.998446101321, 21.566573028569, 21.212230212422,
    21.619450800321, 22.316818746894, 21.824255377951, 22.071028206812, 22.128561804940, 21.371215591830, 21.566573028569, 20.898231922780,
    22.101088234553, 22.159340776707, 21.813336017509, 21.173477342908, 22.164133924836, 21.279742874309, 21.352787248778, 19.625216134798,
    19.625216134798, 21.352787248778, 21.279742874309, 22.164133924836, 21.173477342908, 21.813336017509, 22.159340776707, 22.101088234553,
    20.898231922780, 21.566573028569, 21.371215591830, 22.128561804940, 22.071028206812, 21.824255377951, 22.316818746894, 21.619450800321,
    21.212230212422, 21.566573028569, 20.998446101321, 21.552507909898, 21.905658143158, 20.713091342300, 22.110979951554, 22.183658668804,
    22.027864184497, 21.898526276251, 22.025665566915, 21.735681644149, 22.533640059552, 21.762765127248, 22.159340776707, 20.909865233072,
    21.029362235052, 21.813336017509, 21.247815094663, 22.003006945849, 21.225130372488, 20.781825244612, 22.025665566915, 21.667334844418,
    21.074909644102, 20.713091342300, 19.836183398903, 20.395154592852, 21.714433807925, 20.426024537110, 21.371215591830, 20.879638193258,
    21.385626171451, 21.824255377951, 21.368127807977, 21.795008189113, 21.576570705832, 20.426024537110, 20.998446101321, 21.079919431187,
    22.169771791078, 21.762765127248, 21.247815094663, 21.037507871441, 22.030722028485, 20.837796018606, 21.279742874309, 19.922535674259,
    20.740506282542, 22.164133924836, 22.030722028485, 22.467484907100, 22.121730115823, 22.003006945849, 22.533640059552, 22.241428828694,
    22.242885700339, 21.552507909898, 21.714433807925, 22.027327627276, 22.616243867598, 21.795008189113, 22.071028206812, 21.123839319475,
    21.992170903699, 22.128561804940, 21.576570705832, 22.027327627276, 22.067813587727, 20.395154592852, 21.905658143158, 21.312367112387,
    22.111810174929, 21.735681644149, 21.225130372488, 20.848608927869, 22.121730115823, 21.037507871441, 21.173477342908, 20.430511751848,
    21.749226191372, 22.101088234553, 22.169771791078, 22.241428828694, 22.111810174929, 21.667334844418, 22.027864184497, 21.302504741813,
    22.679752296797, 22.183658668804, 21.074909644102, 21.312367112387, 22.242885700339, 21.079919431187, 20.898231922780, 19.914020154082,
    21.899016998869, 21.619450800321, 21.385626171451, 21.123839319475, 21.992170903699, 20.879638193258, 21.212230212422, 19.914020154082,
    21.749226191372, 20.909865233072, 21.029362235052, 20.430511751848, 20.740506282542, 19.922535674259, 19.625216134798, 18.527390041484,
)


class Config:
    INPUT_BITS = 3
    POINT_COUNT = 8
    FUNCTION_COUNT = 256
    WIDTH = 16
    HIDDEN_LAYERS = 1
    ACTIVATION = "tanh"
    ARCHITECTURE_NAME = "tanh_w16_d1"
    ARCHITECTURE_NAMES = tuple(item.name for item in ARCHITECTURES)
    PROTOCOL_VERSION = "3bit_cross_architecture_complexity_order_v1"
    PRIOR_KIND = "independent_standard_gaussian_neural_coordinates"
    INPUT_ENCODING = "zero_one"

    REPLICAS = 8
    PARTICLES_PER_REPLICA = 4_096
    SURVIVAL_QUANTILE = 0.5
    TARGET_THRESHOLDS = (
        0.70,
        0.50,
        0.30,
        0.20,
        0.12,
        0.08,
        0.06,
        0.04,
        0.025,
        0.016,
        0.010,
        0.0063,
        0.0040,
        0.0025,
        0.0016,
        0.0010,
    )
    HARD_EXACT_REFERENCE_THRESHOLD = 0.08
    MAX_LEVELS_PER_TARGET = 5_000
    MIN_LEVEL_DECREMENT = 1e-8

    ADAPT_SWEEPS = 4
    MUTATION_SWEEPS = 12
    TARGET_ACCEPTANCE = 0.30
    ADAPT_RATE = 0.35
    INITIAL_PROPOSAL_SCALES = (0.10, 0.10, 0.04)
    MIN_PROPOSAL_SCALE = 2e-4
    MAX_PROPOSAL_SCALE = 0.95
    LOSS_TOLERANCE = 1e-7

    PRIOR_CALIBRATION_SAMPLES = 67_108_864
    PRIOR_CALIBRATION_REPLICAS = 8
    PRIOR_CALIBRATION_BATCH = 65_536
    PRIOR_CALIBRATION_THRESHOLDS = ()

    EVAL_MICRO_BATCH = 32_768
    CHECKPOINT_EVERY_LEVELS = 10
    LOG_EVERY_LEVELS = 10
    TOP_FUNCTIONS = 8
    SAVE_PARTICLE_SNAPSHOTS = False

    PRIOR_SEED = 2026083101
    RESAMPLE_SEED = 2026083102
    MUTATION_SEED = 2026083103
    CALIBRATION_SEED = 2026083104
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False

    RESULT_DIR = Path(
        "/root/autodl-tmp/results_3bit_cross_architecture_complexity_order"
    )
    RESUME = True
    OVERWRITE_RESULT_DIR = False
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"
    TARGET_START = 0
    TARGET_STOP: int | None = None
    ARCHITECTURE_START = 0
    ARCHITECTURE_STOP: int | None = None


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class TargetSpec:
    target_index: int
    orbit_index: int
    name: str
    outputs: tuple[int, ...]
    function_id: int
    function_hex: str
    function_bits_x000_to_x111: str
    orbit_members: tuple[int, ...]
    orbit_size: int
    positive_count: int
    essential_variables: tuple[int, ...]
    anf_degree: int
    anf_term_count: int
    anf_literal_count: int


@dataclass
class SMCState:
    particles: torch.Tensor
    losses: torch.Tensor
    lineages: torch.Tensor
    log_volume_fraction: torch.Tensor
    proposal_scales: list[float]
    current_threshold: float
    threshold_index: int
    level: int
    level_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    replica_rows: list[dict[str, Any]]
    top_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--architecture-start", type=int, default=None)
    parser.add_argument("--architecture-stop", type=int, default=None)
    parser.add_argument("--target-start", type=int, default=None)
    parser.add_argument("--target-stop", type=int, default=None)
    parser.add_argument("--prior-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    # 在Jupyter/IPython里直接执行脚本时，kernel会自动附加
    # ``-f /path/to/kernel.json``。这些参数不属于本实验，不能让它们阻止
    # notebook运行；已知的实验参数仍由argparse正常解析。
    args, unknown = parser.parse_known_args()
    ignored: list[str] = []
    remaining = list(unknown)
    while len(remaining) >= 2 and remaining[0] == "-f":
        ignored.extend(remaining[:2])
        remaining = remaining[2:]
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if ignored:
        print(
            "忽略Jupyter内核附加的命令行参数：" + " ".join(ignored),
            flush=True,
        )
    return args


def apply_cli(args: argparse.Namespace) -> None:
    if args.result_dir is not None:
        Config.RESULT_DIR = args.result_dir
    if args.device is not None:
        Config.DEVICE = args.device
    if args.architecture_start is not None:
        Config.ARCHITECTURE_START = args.architecture_start
    if args.architecture_stop is not None:
        Config.ARCHITECTURE_STOP = args.architecture_stop
    if args.target_start is not None:
        Config.TARGET_START = args.target_start
    if args.target_stop is not None:
        Config.TARGET_STOP = args.target_stop
    if args.prior_samples is not None:
        Config.PRIOR_CALIBRATION_SAMPLES = args.prior_samples
    if args.overwrite:
        Config.OVERWRITE_RESULT_DIR = True
        Config.RESUME = False
    if args.no_package:
        Config.PACKAGE_RESULTS = False
    if args.smoke:
        Config.SMOKE_TEST = True


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.REPLICAS = 2
    Config.PARTICLES_PER_REPLICA = 64
    Config.TARGET_THRESHOLDS = (0.72, 0.70, 0.68)
    Config.HARD_EXACT_REFERENCE_THRESHOLD = 0.68
    Config.MAX_LEVELS_PER_TARGET = 12
    Config.ADAPT_SWEEPS = 1
    Config.MUTATION_SWEEPS = 2
    Config.PRIOR_CALIBRATION_SAMPLES = 2_048
    Config.PRIOR_CALIBRATION_REPLICAS = 2
    Config.PRIOR_CALIBRATION_BATCH = 256
    Config.PRIOR_CALIBRATION_THRESHOLDS = (0.72, 0.70)
    Config.EVAL_MICRO_BATCH = 256
    Config.CHECKPOINT_EVERY_LEVELS = 1
    Config.LOG_EVERY_LEVELS = 1
    Config.TOP_FUNCTIONS = 4
    Config.DEVICE = "cpu"
    Config.RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_3bit_cross_architecture_complexity_order"
    )
    Config.RESUME = False
    Config.OVERWRITE_RESULT_DIR = True
    Config.PACKAGE_RESULTS = False
    Config.TARGET_START = 0
    Config.TARGET_STOP = 2
    Config.ARCHITECTURE_START = 0
    Config.ARCHITECTURE_STOP = len(ARCHITECTURES)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def config_dict() -> dict[str, Any]:
    return {
        name: json_ready(getattr(Config, name))
        for name in dir(Config)
        if name.isupper()
    }


def immutable_config_keys() -> tuple[str, ...]:
    return (
        "INPUT_BITS",
        "POINT_COUNT",
        "FUNCTION_COUNT",
        "WIDTH",
        "HIDDEN_LAYERS",
        "ACTIVATION",
        "ARCHITECTURE_NAME",
        "PROTOCOL_VERSION",
        "PRIOR_KIND",
        "INPUT_ENCODING",
        "REPLICAS",
        "PARTICLES_PER_REPLICA",
        "SURVIVAL_QUANTILE",
        "ADAPT_SWEEPS",
        "MUTATION_SWEEPS",
        "INITIAL_PROPOSAL_SCALES",
        "PRIOR_SEED",
        "RESAMPLE_SEED",
        "MUTATION_SEED",
        "PRIOR_CALIBRATION_SAMPLES",
        "PRIOR_CALIBRATION_REPLICAS",
    )


def validate_config() -> None:
    if Config.INPUT_BITS != 3 or Config.POINT_COUNT != 8:
        raise ValueError("当前实现固定为3-bit、8点完整真值表。")
    if Config.HIDDEN_LAYERS < 1:
        raise ValueError("至少需要一个隐藏层。")
    if Config.ACTIVATION not in {"tanh", "relu"}:
        raise ValueError(f"不支持的激活函数：{Config.ACTIVATION}")
    thresholds = tuple(float(value) for value in Config.TARGET_THRESHOLDS)
    if tuple(sorted(set(thresholds), reverse=True)) != thresholds:
        raise ValueError("TARGET_THRESHOLDS必须严格从高到低且不重复。")
    if not 0 < Config.SURVIVAL_QUANTILE < 1:
        raise ValueError("SURVIVAL_QUANTILE必须位于(0,1)。")
    if Config.REPLICAS < 2:
        raise ValueError("至少需要两个独立SMC副本。")
    expected_blocks = Config.HIDDEN_LAYERS + 2
    if len(Config.INITIAL_PROPOSAL_SCALES) != expected_blocks:
        raise ValueError(
            "proposal scale必须对应每个隐藏层、输出层和全部参数块。"
        )
    hard_bound = math.log(2.0) / Config.POINT_COUNT
    if (
        not Config.SMOKE_TEST
        and Config.HARD_EXACT_REFERENCE_THRESHOLD >= hard_bound
    ):
        raise ValueError("hard-exact参考阈值必须严格低于ln(2)/8。")


def activate_architecture(spec: ArchitectureSpec, index: int) -> None:
    global ACTIVE_ARCHITECTURE, ACTIVE_ARCHITECTURE_INDEX
    ACTIVE_ARCHITECTURE = spec
    ACTIVE_ARCHITECTURE_INDEX = int(index)
    Config.ARCHITECTURE_NAME = spec.name
    Config.WIDTH = int(spec.hidden_widths[0])
    Config.HIDDEN_LAYERS = len(spec.hidden_widths)
    Config.ACTIVATION = spec.activation
    layer_scales = [0.10]
    layer_scales.extend(0.08 for _ in spec.hidden_widths[1:])
    layer_scales.extend((0.08, 0.04))
    Config.INITIAL_PROPOSAL_SCALES = tuple(layer_scales)


def prepare_result_dir() -> Path:
    output = Path(Config.RESULT_DIR)
    if output.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(output)
    if output.exists() and not Config.RESUME:
        output = output.parent / (
            output.name + "_" + time.strftime("%Y%m%d_%H%M%S")
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "targets").mkdir(exist_ok=True)
    saved_path = output / "config.json"
    current = config_dict()
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        for key in immutable_config_keys():
            if saved.get(key) != current.get(key):
                raise RuntimeError(f"已有结果配置与当前不一致：{key}")
        saved_thresholds = list(saved.get("TARGET_THRESHOLDS", []))
        current_thresholds = list(current["TARGET_THRESHOLDS"])
        if current_thresholds[:len(saved_thresholds)] != saved_thresholds:
            raise RuntimeError("新阈值序列必须以旧阈值序列为前缀。")
    write_json(saved_path, current)
    return output


def truth_table_inputs() -> np.ndarray:
    values = np.arange(Config.POINT_COUNT, dtype=np.uint8)
    shifts = np.arange(Config.INPUT_BITS - 1, -1, -1, dtype=np.uint8)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.uint8)


def function_outputs(function_id: int) -> np.ndarray:
    return np.asarray(
        [
            (int(function_id) >> index) & 1
            for index in range(Config.POINT_COUNT)
        ],
        dtype=np.uint8,
    )


def outputs_to_function_id(outputs: np.ndarray) -> int:
    bits = np.asarray(outputs, dtype=np.uint64).reshape(-1)
    powers = np.left_shift(
        np.uint64(1), np.arange(Config.POINT_COUNT, dtype=np.uint64)
    )
    return int(np.sum(bits * powers, dtype=np.uint64))


def function_bits(function_id: int) -> str:
    return "".join(
        str((int(function_id) >> index) & 1)
        for index in range(Config.POINT_COUNT)
    )


def transform_by_input_permutation(
    outputs: np.ndarray,
    permutation: tuple[int, ...],
) -> np.ndarray:
    inputs = truth_table_inputs()
    transformed = np.empty(Config.POINT_COUNT, dtype=np.uint8)
    place_values = 1 << np.arange(
        Config.INPUT_BITS - 1, -1, -1, dtype=np.int64
    )
    for row, values in enumerate(inputs):
        source = int(np.sum(values[list(permutation)] * place_values))
        transformed[row] = outputs[source]
    return transformed


def exact_symmetry_orbit(function_id: int) -> tuple[int, ...]:
    outputs = function_outputs(function_id)
    members: set[int] = set()
    for permutation in itertools.permutations(range(Config.INPUT_BITS)):
        transformed = transform_by_input_permutation(outputs, permutation)
        local_id = outputs_to_function_id(transformed)
        members.add(local_id)
        members.add(local_id ^ (Config.FUNCTION_COUNT - 1))
    return tuple(sorted(members))


def build_exact_orbits() -> list[tuple[int, ...]]:
    seen: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for function_id in range(Config.FUNCTION_COUNT):
        if function_id in seen:
            continue
        orbit = exact_symmetry_orbit(function_id)
        if function_id not in orbit:
            raise AssertionError("轨道构造没有包含原函数。")
        if seen.intersection(orbit):
            raise AssertionError("严格对称轨道发生非空交叠。")
        seen.update(orbit)
        orbits.append(orbit)
    if seen != set(range(Config.FUNCTION_COUNT)):
        raise AssertionError("严格对称轨道没有覆盖全部256个函数。")
    if len(orbits) != 40:
        raise AssertionError(f"预期40个轨道，实际得到{len(orbits)}个。")
    return orbits


def anf_metrics(outputs: np.ndarray) -> tuple[int, int, int]:
    coefficients = np.asarray(outputs, dtype=np.uint8).copy()
    for bit in range(Config.INPUT_BITS):
        step = 1 << bit
        for mask in range(Config.POINT_COUNT):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    terms = np.flatnonzero(coefficients)
    degrees = np.asarray(
        [int(term).bit_count() for term in terms], dtype=np.int64
    )
    return (
        int(degrees.max()) if len(terms) else 0,
        int(len(terms)),
        int(degrees.sum()),
    )


def essential_variables(outputs: np.ndarray) -> tuple[int, ...]:
    values = np.arange(Config.POINT_COUNT, dtype=np.int64)
    result: list[int] = []
    for bit in range(Config.INPUT_BITS):
        base = values[(values & (1 << bit)) == 0]
        if np.any(outputs[base] != outputs[base | (1 << bit)]):
            result.append(Config.INPUT_BITS - 1 - bit)
    return tuple(sorted(result))


def build_targets() -> tuple[list[TargetSpec], dict[int, int]]:
    targets: list[TargetSpec] = []
    representative_by_function: dict[int, int] = {}
    for orbit_index, orbit in enumerate(build_exact_orbits()):
        representative = min(orbit)
        outputs = function_outputs(representative)
        degree, terms, literals = anf_metrics(outputs)
        target = TargetSpec(
            target_index=orbit_index,
            orbit_index=orbit_index,
            name=f"orbit_{orbit_index:03d}_rep_0x{representative:02X}",
            outputs=tuple(map(int, outputs)),
            function_id=representative,
            function_hex=f"0x{representative:02X}",
            function_bits_x000_to_x111=function_bits(representative),
            orbit_members=orbit,
            orbit_size=len(orbit),
            positive_count=int(outputs.sum()),
            essential_variables=essential_variables(outputs),
            anf_degree=degree,
            anf_term_count=terms,
            anf_literal_count=literals,
        )
        targets.append(target)
        for member in orbit:
            representative_by_function[member] = representative
    return targets, representative_by_function


def parameter_blocks() -> tuple[list[ParameterBlock], int]:
    cursor = 0
    blocks: list[ParameterBlock] = []
    fan_in = Config.INPUT_BITS
    for layer_index, width in enumerate(ACTIVE_ARCHITECTURE.hidden_widths):
        layer_size = int(width) * int(fan_in) + int(width)
        blocks.append(ParameterBlock(
            f"hidden_layer_{layer_index + 1}",
            cursor,
            cursor + layer_size,
        ))
        cursor += layer_size
        fan_in = int(width)
    output_size = fan_in + 1
    output = ParameterBlock("output_layer", cursor, cursor + output_size)
    cursor += output_size
    all_parameters = ParameterBlock("all_parameters", 0, cursor)
    return [*blocks, output, all_parameters], cursor


def forward_logits(normalized: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    count = normalized.shape[0]
    cursor = 0
    hidden = inputs[None].expand(count, -1, -1)
    fan_in = Config.INPUT_BITS
    for width in ACTIVE_ARCHITECTURE.hidden_widths:
        weight_size = int(width) * int(fan_in)
        weight = normalized[:, cursor:cursor + weight_size].reshape(
            count, int(width), int(fan_in)
        ) * (1.0 / math.sqrt(fan_in))
        cursor += weight_size
        bias = normalized[:, cursor:cursor + int(width)] * (
            1.0 / math.sqrt(fan_in)
        )
        cursor += int(width)
        hidden = torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None]
        if ACTIVE_ARCHITECTURE.activation == "tanh":
            hidden = torch.tanh(hidden)
        elif ACTIVE_ARCHITECTURE.activation == "relu":
            hidden = F.relu(hidden)
        else:  # pragma: no cover - validate_config会先拦截
            raise RuntimeError(
                f"未知激活函数：{ACTIVE_ARCHITECTURE.activation}"
            )
        fan_in = int(width)
    output_weight = normalized[:, cursor:cursor + fan_in].reshape(
        count, 1, fan_in
    ) * (1.0 / math.sqrt(fan_in))
    cursor += fan_in
    output_bias = normalized[:, cursor:cursor + 1] * (
        1.0 / math.sqrt(fan_in)
    )
    cursor += 1
    if cursor != normalized.shape[-1]:
        raise RuntimeError(
            f"参数解包错误：使用{cursor}，实际{normalized.shape[-1]}。"
        )
    return (
        torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
        + output_bias
    )


@torch.no_grad()
def evaluate_losses(
    particles: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    flat = particles.reshape(-1, particles.shape[-1])
    pieces: list[torch.Tensor] = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        local_targets = targets[None].expand_as(logits)
        pieces.append(F.binary_cross_entropy_with_logits(
            logits, local_targets, reduction="none"
        ).mean(dim=1))
    return torch.cat(pieces).reshape(particles.shape[:-1])


@torch.no_grad()
def evaluate_function_ids(
    particles: torch.Tensor,
    inputs: torch.Tensor,
) -> np.ndarray:
    flat = particles.reshape(-1, particles.shape[-1])
    powers = torch.bitwise_left_shift(
        torch.ones(Config.POINT_COUNT, dtype=torch.int64, device=particles.device),
        torch.arange(Config.POINT_COUNT, dtype=torch.int64, device=particles.device),
    )
    pieces: list[np.ndarray] = []
    for start in range(0, len(flat), Config.EVAL_MICRO_BATCH):
        logits = forward_logits(
            flat[start:start + Config.EVAL_MICRO_BATCH], inputs
        )
        ids = ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)
        pieces.append(ids.cpu().numpy().astype(np.uint16))
    return np.concatenate(pieces).reshape(particles.shape[:-1])


def distribution_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    probability = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probability * np.log2(probability)))


def js_divergence_from_counts(first: np.ndarray, second: np.ndarray) -> float:
    if first.sum() == 0 or second.sum() == 0:
        return float("nan")
    p = first.astype(np.float64) / first.sum()
    q = second.astype(np.float64) / second.sum()
    middle = 0.5 * (p + q)
    value = 0.0
    for distribution in (p, q):
        mask = distribution > 0
        value += 0.5 * float(np.sum(
            distribution[mask]
            * np.log2(distribution[mask] / middle[mask])
        ))
    return value


def safe_exp(log_value: float) -> float:
    return float(math.exp(log_value)) if log_value > -745.0 else 0.0


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def make_generators(
    device: torch.device,
    target_index: int,
) -> dict[str, torch.Generator]:
    offset = (
        1_000_003 * int(target_index)
        + 100_000_019 * int(ACTIVE_ARCHITECTURE_INDEX)
    )
    generators = {
        "prior": torch.Generator(device=device),
        "resample": torch.Generator(device=device),
        "mutation": torch.Generator(device=device),
    }
    generators["prior"].manual_seed(Config.PRIOR_SEED + offset)
    generators["resample"].manual_seed(Config.RESAMPLE_SEED + offset)
    generators["mutation"].manual_seed(Config.MUTATION_SEED + offset)
    return generators


def initialize_state(
    device: torch.device,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    prior_generator: torch.Generator,
    parameter_count: int,
) -> SMCState:
    particles = torch.empty(
        Config.REPLICAS,
        Config.PARTICLES_PER_REPLICA,
        parameter_count,
        device=device,
    ).normal_(mean=0.0, std=1.0, generator=prior_generator)
    losses = evaluate_losses(particles, inputs, targets)
    total = Config.REPLICAS * Config.PARTICLES_PER_REPLICA
    lineages = torch.arange(
        total, device=device, dtype=torch.int64
    ).reshape(Config.REPLICAS, Config.PARTICLES_PER_REPLICA)
    return SMCState(
        particles=particles,
        losses=losses,
        lineages=lineages,
        log_volume_fraction=torch.zeros(
            Config.REPLICAS, device=device, dtype=torch.float64
        ),
        proposal_scales=list(Config.INITIAL_PROPOSAL_SCALES),
        current_threshold=float("inf"),
        threshold_index=0,
        level=0,
        level_rows=[],
        threshold_rows=[],
        replica_rows=[],
        top_rows=[],
    )


def choose_next_threshold(state: SMCState) -> tuple[float, bool]:
    target = float(Config.TARGET_THRESHOLDS[state.threshold_index])
    quantiles = torch.quantile(
        state.losses, Config.SURVIVAL_QUANTILE, dim=1
    )
    adaptive = float(quantiles.max().item())
    next_threshold = max(target, adaptive)
    if math.isfinite(state.current_threshold):
        next_threshold = min(next_threshold, state.current_threshold)
    return next_threshold, next_threshold <= target + 1e-12


@torch.no_grad()
def resample_within_replicas(
    state: SMCState,
    threshold: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    survival = np.zeros(Config.REPLICAS, dtype=np.float64)
    new_particles = torch.empty_like(state.particles)
    new_lineages = torch.empty_like(state.lineages)
    for replica in range(Config.REPLICAS):
        survivors = torch.nonzero(
            state.losses[replica] <= threshold + Config.LOSS_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if not len(survivors):
            raise RuntimeError(
                f"副本{replica}在阈值{threshold:.8g}没有幸存粒子。"
            )
        survival[replica] = len(survivors) / Config.PARTICLES_PER_REPLICA
        choices = torch.randint(
            len(survivors),
            (Config.PARTICLES_PER_REPLICA,),
            generator=generator,
            device=state.particles.device,
        )
        selected = survivors[choices]
        new_particles[replica] = state.particles[replica, selected]
        new_lineages[replica] = state.lineages[replica, selected]
    state.particles = new_particles
    state.lineages = new_lineages
    state.losses = evaluate_losses(state.particles, inputs, targets)
    state.log_volume_fraction += torch.log(torch.from_numpy(
        survival
    ).to(state.log_volume_fraction))
    return survival


@torch.no_grad()
def mutate_block(
    state: SMCState,
    block: ParameterBlock,
    scale: float,
    threshold: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> float:
    proposal = state.particles.clone()
    current = proposal[..., block.start:block.stop]
    noise = torch.randn(
        current.shape,
        device=current.device,
        generator=generator,
        dtype=current.dtype,
    )
    rho = min(max(float(scale), 0.0), 0.999999)
    proposal[..., block.start:block.stop] = (
        math.sqrt(1.0 - rho * rho) * current + rho * noise
    )
    proposal_losses = evaluate_losses(proposal, inputs, targets)
    accept = proposal_losses <= threshold + Config.LOSS_TOLERANCE
    flat_accept = accept.reshape(-1)
    flat_particles = state.particles.reshape(-1, state.particles.shape[-1])
    flat_proposal = proposal.reshape(-1, proposal.shape[-1])
    flat_particles[flat_accept] = flat_proposal[flat_accept]
    flat_losses = state.losses.reshape(-1)
    flat_losses[flat_accept] = proposal_losses.reshape(-1)[flat_accept]
    return float(accept.float().mean().item())


def rejuvenate(
    state: SMCState,
    blocks: Sequence[ParameterBlock],
    threshold: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, float]:
    scales = list(state.proposal_scales)
    for _ in range(Config.ADAPT_SWEEPS):
        for block_index, block in enumerate(blocks):
            acceptance = mutate_block(
                state,
                block,
                scales[block_index],
                threshold,
                inputs,
                targets,
                generator,
            )
            scales[block_index] *= math.exp(
                Config.ADAPT_RATE * (acceptance - Config.TARGET_ACCEPTANCE)
            )
            scales[block_index] = min(
                max(scales[block_index], Config.MIN_PROPOSAL_SCALE),
                Config.MAX_PROPOSAL_SCALE,
            )
    state.proposal_scales = scales
    acceptance_sum = np.zeros(len(blocks), dtype=np.float64)
    for _ in range(Config.MUTATION_SWEEPS):
        for block_index, block in enumerate(blocks):
            acceptance_sum[block_index] += mutate_block(
                state,
                block,
                scales[block_index],
                threshold,
                inputs,
                targets,
                generator,
            )
    acceptance_mean = acceptance_sum / max(Config.MUTATION_SWEEPS, 1)
    return {
        f"acceptance_{block.name}": float(acceptance_mean[index])
        for index, block in enumerate(blocks)
    }


def append_top_functions(
    rows: list[dict[str, Any]],
    target: TargetSpec,
    threshold: float,
    counts: np.ndarray,
) -> None:
    total = int(counts.sum())
    if total == 0:
        return
    for rank, function_id in enumerate(
        np.argsort(counts)[::-1][:Config.TOP_FUNCTIONS], start=1
    ):
        count = int(counts[function_id])
        if count == 0:
            break
        rows.append({
            "target_name": target.name,
            "threshold": threshold,
            "rank": rank,
            "function_id": int(function_id),
            "function_hex": f"0x{int(function_id):02X}",
            "truth_table_x000_to_x111": function_bits(int(function_id)),
            "count": count,
            "probability": count / total,
            "is_target": int(function_id) == target.function_id,
        })


def record_threshold(
    output_dir: Path,
    state: SMCState,
    threshold: float,
    inputs: torch.Tensor,
    target: TargetSpec,
) -> None:
    ids = evaluate_function_ids(state.particles, inputs)
    losses = state.losses.detach().cpu().numpy()
    counts = np.bincount(ids.reshape(-1), minlength=Config.FUNCTION_COUNT)
    target_probability = float(counts[target.function_id] / counts.sum())
    replica_js: list[float] = []
    replica_target: list[float] = []
    replica_logs = state.log_volume_fraction.detach().cpu().numpy()
    for replica in range(Config.REPLICAS):
        replica_counts = np.bincount(
            ids[replica], minlength=Config.FUNCTION_COUNT
        )
        local_target = float(
            replica_counts[target.function_id] / replica_counts.sum()
        )
        local_js = js_divergence_from_counts(replica_counts, counts)
        replica_target.append(local_target)
        replica_js.append(local_js)
        state.replica_rows.append({
            "target_name": target.name,
            "target_function_id": target.function_id,
            "threshold": threshold,
            "replica": replica,
            "estimated_log_volume_fraction": float(replica_logs[replica]),
            "estimated_log10_volume_fraction": float(
                replica_logs[replica] / math.log(10.0)
            ),
            "neural_k_bits": float(
                -replica_logs[replica] / math.log(2.0)
            ),
            "estimated_volume_fraction": safe_exp(
                float(replica_logs[replica])
            ),
            "target_probability": local_target,
            "function_entropy_bits": distribution_entropy(replica_counts),
            "function_support": int(np.count_nonzero(replica_counts)),
            "js_to_aggregate": local_js,
            "unique_lineages": int(torch.unique(
                state.lineages[replica]
            ).numel()),
        })

    log_median = float(np.median(replica_logs))
    hard_bound = math.log(2.0) / Config.POINT_COUNT
    hard_exact = threshold < hard_bound
    row = {
        "target_name": target.name,
        "target_function_id": target.function_id,
        "target_function_hex": target.function_hex,
        "orbit_index": target.orbit_index,
        "orbit_size": target.orbit_size,
        "threshold": threshold,
        "level": state.level,
        "particle_count": int(counts.sum()),
        "estimated_log_volume_fraction_median": log_median,
        "estimated_log_volume_fraction_min": float(np.min(replica_logs)),
        "estimated_log_volume_fraction_max": float(np.max(replica_logs)),
        "estimated_log10_volume_fraction_median": float(
            log_median / math.log(10.0)
        ),
        "neural_k_bits_median": float(-log_median / math.log(2.0)),
        "estimated_volume_fraction_median": safe_exp(log_median),
        "loss_min": float(losses.min()),
        "loss_median": float(np.median(losses)),
        "loss_max": float(losses.max()),
        "target_probability": target_probability,
        "hard_exact_guaranteed_by_threshold": hard_exact,
        "hard_exact_violation_count": (
            int(counts.sum() - counts[target.function_id])
            if hard_exact else None
        ),
        "function_entropy_bits": distribution_entropy(counts),
        "function_support": int(np.count_nonzero(counts)),
        "top_function_hex": f"0x{int(np.argmax(counts)):02X}",
        "top_function_probability": float(counts.max() / counts.sum()),
        "replica_target_probability_min": float(np.min(replica_target)),
        "replica_target_probability_max": float(np.max(replica_target)),
        "replica_js_to_aggregate_median": float(np.median(replica_js)),
        "replica_js_to_aggregate_max": float(np.max(replica_js)),
        "replica_log_volume_range": float(
            np.max(replica_logs) - np.min(replica_logs)
        ),
        "unique_lineages_min": int(min(
            torch.unique(state.lineages[replica]).numel()
            for replica in range(Config.REPLICAS)
        )),
        "unique_lineages_max": int(max(
            torch.unique(state.lineages[replica]).numel()
            for replica in range(Config.REPLICAS)
        )),
    }
    if hard_exact and row["hard_exact_violation_count"] != 0:
        raise AssertionError(
            f"{target.name}在epsilon={threshold}进入hard-exact区后仍出现"
            f"{row['hard_exact_violation_count']}个非目标函数粒子。"
        )
    state.threshold_rows.append(row)
    append_top_functions(state.top_rows, target, threshold, counts)
    if Config.SAVE_PARTICLE_SNAPSHOTS:
        snapshot_name = f"snapshot_threshold_{threshold:.6f}".replace(".", "p")
        np.savez_compressed(
            output_dir / f"{snapshot_name}.npz",
            target_name=np.asarray(target.name),
            threshold=np.asarray(threshold),
            function_ids=ids,
            losses=losses,
            lineages=state.lineages.detach().cpu().numpy(),
            log_volume_fraction=replica_logs,
        )
    print(
        f"[{target.name}] TARGET eps={threshold:.6g} | "
        f"log10V={row['estimated_log10_volume_fraction_median']:.3f} | "
        f"K={row['neural_k_bits_median']:.2f} bits | "
        f"target={target_probability:.3%} | "
        f"replica_logV_range={row['replica_log_volume_range']:.3f} | "
        f"lineages={row['unique_lineages_min']}-"
        f"{row['unique_lineages_max']}",
        flush=True,
    )


def state_payload(
    state: SMCState,
    target: TargetSpec,
    generators: dict[str, torch.Generator],
) -> dict[str, Any]:
    return {
        "config": config_dict(),
        "target": asdict(target),
        "particles": state.particles.detach().cpu(),
        "losses": state.losses.detach().cpu(),
        "lineages": state.lineages.detach().cpu(),
        "log_volume_fraction": state.log_volume_fraction.detach().cpu(),
        "proposal_scales": state.proposal_scales,
        "current_threshold": state.current_threshold,
        "threshold_index": state.threshold_index,
        "level": state.level,
        "level_rows": state.level_rows,
        "threshold_rows": state.threshold_rows,
        "replica_rows": state.replica_rows,
        "top_rows": state.top_rows,
        "generator_states": {
            name: generator.get_state().cpu()
            for name, generator in generators.items()
        },
    }


def save_checkpoint(
    output_dir: Path,
    state: SMCState,
    target: TargetSpec,
    generators: dict[str, torch.Generator],
) -> None:
    temporary = output_dir / "checkpoint.tmp.pt"
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(state_payload(state, target, generators), temporary)
    temporary.replace(checkpoint)


def load_checkpoint(
    output_dir: Path,
    target: TargetSpec,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> SMCState:
    payload = torch.load(
        output_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    saved_target = payload["target"]
    if (
        saved_target["name"] != target.name
        or int(saved_target["function_id"]) != target.function_id
    ):
        raise RuntimeError(f"{target.name}的checkpoint目标不一致。")
    saved_config = payload["config"]
    current_config = config_dict()
    for key in immutable_config_keys():
        if saved_config.get(key) != current_config.get(key):
            raise RuntimeError(f"checkpoint配置不一致：{key}")
    saved_thresholds = list(saved_config["TARGET_THRESHOLDS"])
    current_thresholds = list(current_config["TARGET_THRESHOLDS"])
    if current_thresholds[:len(saved_thresholds)] != saved_thresholds:
        raise RuntimeError("checkpoint阈值不是当前阈值序列的前缀。")
    for name, generator in generators.items():
        generator.set_state(
            payload["generator_states"][name].to(
                dtype=torch.uint8, device="cpu"
            )
        )
    return SMCState(
        particles=payload["particles"].to(device),
        losses=payload["losses"].to(device),
        lineages=payload["lineages"].to(device),
        log_volume_fraction=payload["log_volume_fraction"].to(
            device=device, dtype=torch.float64
        ),
        proposal_scales=[float(value) for value in payload["proposal_scales"]],
        current_threshold=float(payload["current_threshold"]),
        threshold_index=int(payload["threshold_index"]),
        level=int(payload["level"]),
        level_rows=list(payload["level_rows"]),
        threshold_rows=list(payload["threshold_rows"]),
        replica_rows=list(payload["replica_rows"]),
        top_rows=list(payload["top_rows"]),
    )


def write_target_artifacts(
    output_dir: Path,
    state: SMCState,
    target: TargetSpec,
    status: str,
) -> None:
    write_csv(output_dir / "levels.csv", state.level_rows)
    write_csv(output_dir / "volume_curve.csv", state.threshold_rows)
    write_csv(output_dir / "replica_volume_curve.csv", state.replica_rows)
    write_csv(output_dir / "top_functions.csv", state.top_rows)
    write_json(output_dir / "summary.json", {
        "status": status,
        "protocol": Config.PROTOCOL_VERSION,
        "target": asdict(target),
        "completed_thresholds": state.threshold_index,
        "requested_thresholds": len(Config.TARGET_THRESHOLDS),
        "current_threshold": state.current_threshold,
        "level": state.level,
        "hard_error_lower_bound": math.log(2.0) / Config.POINT_COUNT,
        "threshold_rows": state.threshold_rows,
        "interpretation": {
            "measure": (
                "标准Gaussian神经坐标测度中，完整8点真值表mean raw BCE"
                "不超过阈值的累计参数质量"
            ),
            "not_optimizer": "SMC使用pCN且不读取loss梯度。",
            "symmetry": (
                "输入变量置换和输出整体取反严格共享本曲线；输入位取反"
                "没有被合并。"
            ),
        },
    })


@torch.no_grad()
def run_prior_calibration(
    output_dir: Path,
    targets: Sequence[TargetSpec],
    representative_by_function: dict[int, int],
    device: torch.device,
    inputs: torch.Tensor,
    parameter_count: int,
) -> None:
    hard_path = output_dir / "prior_hard_function_counts.csv"
    replica_path = output_dir / "prior_hard_function_counts_by_replica.csv"
    if hard_path.exists() and replica_path.exists() and Config.RESUME:
        rows = read_csv(hard_path)
        saved_count = sum(int(row["count"]) for row in rows)
        if saved_count == Config.PRIOR_CALIBRATION_SAMPLES:
            print(
                f"direct finite-network prior已有{saved_count:,}个样本，跳过。",
                flush=True,
            )
            return
        raise RuntimeError(
            "已有先验计数的样本量与当前配置不同："
            f"saved={saved_count:,}, requested="
            f"{Config.PRIOR_CALIBRATION_SAMPLES:,}。"
        )

    if Config.PRIOR_CALIBRATION_REPLICAS < 2:
        raise ValueError("先验直接计数至少需要两个独立副本。")
    if (
        Config.PRIOR_CALIBRATION_SAMPLES
        % Config.PRIOR_CALIBRATION_REPLICAS
        != 0
    ):
        raise ValueError("先验样本总量必须能被独立副本数整除。")
    samples_per_replica = (
        Config.PRIOR_CALIBRATION_SAMPLES
        // Config.PRIOR_CALIBRATION_REPLICAS
    )
    replica_counts = np.zeros(
        (Config.PRIOR_CALIBRATION_REPLICAS, Config.FUNCTION_COUNT),
        dtype=np.int64,
    )
    started = time.perf_counter()
    total_completed = 0
    powers = torch.bitwise_left_shift(
        torch.ones(Config.POINT_COUNT, dtype=torch.int64, device=device),
        torch.arange(Config.POINT_COUNT, dtype=torch.int64, device=device),
    )
    for replica in range(Config.PRIOR_CALIBRATION_REPLICAS):
        generator = torch.Generator(device=device)
        generator.manual_seed(
            Config.CALIBRATION_SEED
            + 100_000_019 * int(ACTIVE_ARCHITECTURE_INDEX)
            + 1_000_003 * replica
        )
        completed = 0
        while completed < samples_per_replica:
            batch = min(
                Config.PRIOR_CALIBRATION_BATCH,
                samples_per_replica - completed,
            )
            particles = torch.empty(
                batch, parameter_count, device=device
            ).normal_(mean=0.0, std=1.0, generator=generator)
            logits = forward_logits(particles, inputs)
            ids = ((logits >= 0).to(torch.int64) * powers[None]).sum(dim=1)
            replica_counts[replica] += np.bincount(
                ids.cpu().numpy(), minlength=Config.FUNCTION_COUNT
            )
            completed += batch
            total_completed += batch
        elapsed = time.perf_counter() - started
        eta = elapsed * (
            Config.PRIOR_CALIBRATION_SAMPLES - total_completed
        ) / max(total_completed, 1)
        print(
            f"finite prior replica={replica + 1}/"
            f"{Config.PRIOR_CALIBRATION_REPLICAS} | "
            f"samples={total_completed:,}/"
            f"{Config.PRIOR_CALIBRATION_SAMPLES:,} | "
            f"seen={int(np.sum(replica_counts.sum(axis=0) > 0))}/256 | "
            f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}",
            flush=True,
        )

    hard_counts = replica_counts.sum(axis=0)
    total = int(hard_counts.sum())
    hard_rows: list[dict[str, Any]] = []
    replica_rows: list[dict[str, Any]] = []
    for function_id in range(Config.FUNCTION_COUNT):
        probability = float(hard_counts[function_id] / total)
        local_probabilities = (
            replica_counts[:, function_id].astype(np.float64)
            / samples_per_replica
        )
        hard_rows.append({
            "function_id": function_id,
            "function_hex": f"0x{function_id:02X}",
            "truth_table_x000_to_x111": function_bits(function_id),
            "representative_function_id": (
                representative_by_function[function_id]
            ),
            "sample_count": total,
            "count": int(hard_counts[function_id]),
            "probability": probability,
            "binomial_standard_error": math.sqrt(
                probability * (1.0 - probability) / total
            ),
            "replica_probability_mean": float(np.mean(local_probabilities)),
            "replica_probability_sd": float(
                np.std(local_probabilities, ddof=1)
            ),
            "replica_count_min": int(np.min(replica_counts[:, function_id])),
            "replica_count_max": int(np.max(replica_counts[:, function_id])),
            "nonzero_replicas": int(np.sum(replica_counts[:, function_id] > 0)),
            "zero_count_95pct_upper_bound": (
                1.0 - 0.05 ** (1.0 / total)
                if hard_counts[function_id] == 0 else None
            ),
        })
        for replica in range(Config.PRIOR_CALIBRATION_REPLICAS):
            replica_rows.append({
                "architecture": ACTIVE_ARCHITECTURE.name,
                "replica": replica,
                "function_id": function_id,
                "function_hex": f"0x{function_id:02X}",
                "sample_count": samples_per_replica,
                "count": int(replica_counts[replica, function_id]),
                "probability": float(local_probabilities[replica]),
            })
    write_csv(hard_path, hard_rows)
    write_csv(replica_path, replica_rows)


def target_completed(output_dir: Path) -> bool:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(
        summary.get("status") == "completed"
        and int(summary.get("completed_thresholds", -1))
        == len(Config.TARGET_THRESHOLDS)
    )


def run_target(
    root_dir: Path,
    target: TargetSpec,
    target_ordinal: int,
    target_total: int,
    device: torch.device,
    inputs: torch.Tensor,
    blocks: Sequence[ParameterBlock],
    parameter_count: int,
) -> tuple[str, float]:
    output_dir = root_dir / "targets" / target.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if Config.RESUME and target_completed(output_dir):
        print(f"[{target.name}] 已完成，跳过。", flush=True)
        return "completed", 0.0

    target_tensor = torch.as_tensor(
        np.asarray(target.outputs, dtype=np.float32), device=device
    )
    generators = make_generators(device, target.target_index)
    checkpoint = output_dir / "checkpoint.pt"
    if Config.RESUME and checkpoint.exists():
        state = load_checkpoint(output_dir, target, device, generators)
        print(
            f"[{target.name}] 恢复checkpoint level={state.level} | "
            f"eps={state.current_threshold:.8g} | "
            f"targets={state.threshold_index}/{len(Config.TARGET_THRESHOLDS)}",
            flush=True,
        )
    else:
        state = initialize_state(
            device,
            inputs,
            target_tensor,
            generators["prior"],
            parameter_count,
        )
        save_checkpoint(output_dir, state, target, generators)

    started = time.perf_counter()
    status = "running"
    try:
        while (
            state.threshold_index < len(Config.TARGET_THRESHOLDS)
            and state.level < Config.MAX_LEVELS_PER_TARGET
        ):
            previous = state.current_threshold
            next_threshold, reaches_target = choose_next_threshold(state)
            if (
                math.isfinite(previous)
                and next_threshold >= previous - Config.MIN_LEVEL_DECREMENT
                and not reaches_target
            ):
                status = "stalled_threshold"
                print(
                    f"[{target.name}] 阈值在{previous:.9g}停止下降，"
                    "保存并转入下一目标。",
                    flush=True,
                )
                break
            survival = resample_within_replicas(
                state,
                next_threshold,
                inputs,
                target_tensor,
                generators["resample"],
            )
            mutation = rejuvenate(
                state,
                blocks,
                next_threshold,
                inputs,
                target_tensor,
                generators["mutation"],
            )
            state.level += 1
            state.current_threshold = next_threshold
            replica_logs = state.log_volume_fraction.detach().cpu().numpy()
            elapsed = time.perf_counter() - started
            level_row: dict[str, Any] = {
                "target_name": target.name,
                "target_ordinal": target_ordinal,
                "target_total": target_total,
                "level": state.level,
                "threshold": next_threshold,
                "reaches_requested_target": reaches_target,
                "next_requested_target": float(
                    Config.TARGET_THRESHOLDS[state.threshold_index]
                ),
                "survival_fraction_min": float(np.min(survival)),
                "survival_fraction_median": float(np.median(survival)),
                "survival_fraction_max": float(np.max(survival)),
                "loss_min": float(state.losses.min().item()),
                "loss_median": float(state.losses.median().item()),
                "loss_max": float(state.losses.max().item()),
                "estimated_log_volume_fraction_median": float(
                    np.median(replica_logs)
                ),
                "estimated_log10_volume_fraction_median": float(
                    np.median(replica_logs) / math.log(10.0)
                ),
                "elapsed_seconds": elapsed,
                "proposal_scales": list(state.proposal_scales),
            }
            level_row.update(mutation)
            state.level_rows.append(level_row)

            if reaches_target or state.level % Config.LOG_EVERY_LEVELS == 0:
                acceptance = ",".join(
                    f"{block.name}:{mutation[f'acceptance_{block.name}']:.1%}"
                    for block in blocks
                )
                print(
                    f"[{target_ordinal}/{target_total} {target.name}] "
                    f"level={state.level:>4} | eps={next_threshold:.7g} | "
                    f"requested={state.threshold_index + 1}/"
                    f"{len(Config.TARGET_THRESHOLDS)} | "
                    f"survive={np.min(survival):.1%}/"
                    f"{np.median(survival):.1%}/{np.max(survival):.1%} | "
                    f"log10V={level_row['estimated_log10_volume_fraction_median']:.2f} | "
                    f"accept[{acceptance}] | elapsed={format_duration(elapsed)}",
                    flush=True,
                )

            if reaches_target:
                requested = float(
                    Config.TARGET_THRESHOLDS[state.threshold_index]
                )
                record_threshold(output_dir, state, requested, inputs, target)
                state.threshold_index += 1
                write_target_artifacts(output_dir, state, target, "running")
                save_checkpoint(output_dir, state, target, generators)
            elif state.level % Config.CHECKPOINT_EVERY_LEVELS == 0:
                save_checkpoint(output_dir, state, target, generators)

        if status == "running":
            status = (
                "completed"
                if state.threshold_index == len(Config.TARGET_THRESHOLDS)
                else "stopped_max_levels"
            )
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\n[{target.name}] 收到中断，正在保存checkpoint。", flush=True)
    finally:
        save_checkpoint(output_dir, state, target, generators)
        write_target_artifacts(output_dir, state, target, status)
    if status == "completed" and checkpoint.exists():
        checkpoint.unlink()
    duration = time.perf_counter() - started
    del state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return status, duration


def aggregate_results(
    output_dir: Path,
    targets: Sequence[TargetSpec],
    representative_by_function: dict[int, int],
) -> dict[str, Any]:
    target_by_representative = {
        target.function_id: target for target in targets
    }
    representative_rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for target in targets:
        target_dir = output_dir / "targets" / target.name
        summary_path = target_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            statuses[target.name] = str(summary.get("status", "unknown"))
        for raw in read_csv(target_dir / "volume_curve.csv"):
            row = dict(raw)
            row["target_function_id"] = int(row["target_function_id"])
            row["threshold"] = float(row["threshold"])
            row["estimated_log_volume_fraction_median"] = float(
                row["estimated_log_volume_fraction_median"]
            )
            row["estimated_log_volume_fraction_min"] = float(
                row["estimated_log_volume_fraction_min"]
            )
            row["estimated_log_volume_fraction_max"] = float(
                row["estimated_log_volume_fraction_max"]
            )
            row["neural_k_bits_median"] = float(
                row["neural_k_bits_median"]
            )
            representative_rows.append(row)
    write_csv(output_dir / "representative_volume_curves.csv", representative_rows)

    lookup = {
        (int(row["target_function_id"]), float(row["threshold"])): row
        for row in representative_rows
    }
    all_rows: list[dict[str, Any]] = []
    for function_id in range(Config.FUNCTION_COUNT):
        representative = representative_by_function[function_id]
        target = target_by_representative[representative]
        outputs = function_outputs(function_id)
        degree, terms, literals = anf_metrics(outputs)
        for threshold in Config.TARGET_THRESHOLDS:
            source = lookup.get((representative, float(threshold)))
            if source is None:
                continue
            all_rows.append({
                "function_id": function_id,
                "function_hex": f"0x{function_id:02X}",
                "truth_table_x000_to_x111": function_bits(function_id),
                "positive_count": int(outputs.sum()),
                "essential_variables": essential_variables(outputs),
                "anf_degree": degree,
                "anf_term_count": terms,
                "anf_literal_count": literals,
                "representative_function_id": representative,
                "representative_function_hex": f"0x{representative:02X}",
                "orbit_index": target.orbit_index,
                "orbit_size": target.orbit_size,
                "inherited_by_exact_symmetry": function_id != representative,
                "threshold": float(threshold),
                "estimated_log_volume_fraction_median": source[
                    "estimated_log_volume_fraction_median"
                ],
                "estimated_log_volume_fraction_min": source[
                    "estimated_log_volume_fraction_min"
                ],
                "estimated_log_volume_fraction_max": source[
                    "estimated_log_volume_fraction_max"
                ],
                "neural_k_bits_median": source["neural_k_bits_median"],
                "replica_log_volume_range": float(
                    source["replica_log_volume_range"]
                ),
                "unique_lineages_min": int(source["unique_lineages_min"]),
                "hard_exact_guaranteed_by_threshold": source[
                    "hard_exact_guaranteed_by_threshold"
                ],
            })
    write_csv(output_dir / "all_256_neural_k_curves.csv", all_rows)

    thresholds = np.asarray(Config.TARGET_THRESHOLDS, dtype=np.float64)
    log_volume = np.full(
        (len(thresholds), Config.FUNCTION_COUNT), np.nan, dtype=np.float64
    )
    neural_k = np.full_like(log_volume, np.nan)
    for row in all_rows:
        threshold_index = int(np.flatnonzero(np.isclose(
            thresholds, float(row["threshold"]), atol=1e-12
        ))[0])
        function_id = int(row["function_id"])
        log_volume[threshold_index, function_id] = float(
            row["estimated_log_volume_fraction_median"]
        )
        neural_k[threshold_index, function_id] = float(
            row["neural_k_bits_median"]
        )
    np.savez_compressed(
        output_dir / "all_256_neural_k_curves.npz",
        thresholds=thresholds,
        function_ids=np.arange(Config.FUNCTION_COUNT, dtype=np.int16),
        representative_function_ids=np.asarray([
            representative_by_function[index]
            for index in range(Config.FUNCTION_COUNT)
        ], dtype=np.int16),
        log_volume=log_volume,
        neural_k_bits=neural_k,
    )

    coverage_rows: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(thresholds):
        finite = np.isfinite(log_volume[threshold_index])
        completed_representatives = {
            representative_by_function[index]
            for index in np.flatnonzero(finite)
        }
        coverage_rows.append({
            "threshold": float(threshold),
            "representative_count": len(completed_representatives),
            "function_count": int(finite.sum()),
            "all_40_representatives": len(completed_representatives) == len(targets),
            "all_256_functions": int(finite.sum()) == Config.FUNCTION_COUNT,
        })
    write_csv(output_dir / "coverage_by_threshold.csv", coverage_rows)
    return {
        "representative_count": len(targets),
        "completed_representative_count": sum(
            status == "completed" for status in statuses.values()
        ),
        "expanded_function_count_at_deepest_threshold": int(
            np.isfinite(log_volume[-1]).sum()
        ),
        "run_statuses": statuses,
        "coverage_by_threshold": coverage_rows,
    }


def mingard_lz_word_count(bit_string: str) -> int:
    """Valle-Perez/Mingard使用的LZ76词典计数。"""
    n = len(bit_string)
    source = "0" + bit_string
    complexity = 1
    lookahead = 1
    search = 0
    matched = 1
    matched_max = 1
    while True:
        if source[search + matched] != source[lookahead + matched]:
            matched_max = max(matched_max, matched)
            search += 1
            if search == lookahead:
                complexity += 1
                lookahead += matched_max
                if lookahead + 1 > n:
                    return complexity
                search = 0
                matched = 1
                matched_max = 1
            else:
                matched = 1
        else:
            matched += 1
            if lookahead + matched > n:
                return complexity + 1


def mingard_lz_complexity(outputs: np.ndarray) -> float:
    bits = "".join(str(int(value)) for value in outputs)
    length = len(bits)
    if bits == "0" * length or bits == "1" * length:
        return float(math.log2(length))
    return float(
        math.log2(length)
        * (
            mingard_lz_word_count(bits)
            + mingard_lz_word_count(bits[::-1])
        )
        / 2.0
    )


def lz_symmetry_metrics(outputs: np.ndarray) -> tuple[float, float, float, float]:
    canonical = mingard_lz_complexity(outputs)
    values: list[float] = []
    for permutation in itertools.permutations(range(Config.INPUT_BITS)):
        transformed = transform_by_input_permutation(outputs, permutation)
        values.append(mingard_lz_complexity(transformed))
        values.append(mingard_lz_complexity(1 - transformed))
    return (
        canonical,
        float(np.min(values)),
        float(np.mean(values)),
        float(np.max(values)),
    )


def ctm_symmetry_metrics(outputs: np.ndarray) -> tuple[float, float, float, float]:
    canonical = float(
        CTM_B2_D12_LENGTH8[outputs_to_function_id(outputs)]
    )
    values: list[float] = []
    for permutation in itertools.permutations(range(Config.INPUT_BITS)):
        transformed = transform_by_input_permutation(outputs, permutation)
        values.append(float(
            CTM_B2_D12_LENGTH8[outputs_to_function_id(transformed)]
        ))
        values.append(float(
            CTM_B2_D12_LENGTH8[outputs_to_function_id(1 - transformed)]
        ))
    return (
        canonical,
        float(np.min(values)),
        float(np.mean(values)),
        float(np.max(values)),
    )


def binary_output_entropy(outputs: np.ndarray) -> float:
    probability = float(np.mean(outputs))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * math.log2(probability)
        - (1.0 - probability) * math.log2(1.0 - probability)
    )


@functools.lru_cache(maxsize=None)
def minimum_sop_metrics(function_id: int) -> tuple[int, int, int]:
    """返回最少operation的SOP所需operation、term和literal数量。"""
    positive_mask = int(function_id)
    if positive_mask == 0:
        return 0, 0, 0
    inputs = truth_table_inputs()
    implicants: list[tuple[int, int, int]] = []
    for cube in itertools.product((-1, 0, 1), repeat=Config.INPUT_BITS):
        covered = 0
        literal_count = sum(value != -1 for value in cube)
        negation_count = sum(value == 0 for value in cube)
        for row, values in enumerate(inputs):
            if all(
                required == -1 or int(values[index]) == required
                for index, required in enumerate(cube)
            ):
                covered |= 1 << row
        if covered and (covered & ~positive_mask) == 0:
            internal_operations = (
                negation_count + max(literal_count - 1, 0)
            )
            implicants.append(
                (covered, literal_count, internal_operations)
            )

    infinity = 10_000
    best: list[tuple[int, int, int] | None] = [None] * 256
    best[0] = (0, 0, 0)
    for mask in range(256):
        if best[mask] is None:
            continue
        operations, terms, literals = best[mask]
        for covered, local_literals, local_operations in implicants:
            new_mask = mask | covered
            if new_mask == mask:
                continue
            candidate = (
                operations + local_operations + (1 if terms else 0),
                terms + 1,
                literals + local_literals,
            )
            current = best[new_mask]
            if current is None or candidate < current:
                best[new_mask] = candidate
    result = best[positive_mask]
    if result is None or result[0] >= infinity:
        raise RuntimeError(f"无法构造函数0x{function_id:02X}的SOP。")
    return result


def formula_gate_costs(include_xor: bool) -> np.ndarray:
    """精确求三变量公式树在给定gate语言下的最少operation数量。"""
    infinity = 10_000
    costs = np.full(256, infinity, dtype=np.int32)
    costs[0x00] = 0
    costs[0xFF] = 0
    costs[0xF0] = 0
    costs[0xCC] = 0
    costs[0xAA] = 0
    while True:
        previous = costs.copy()
        finite = np.flatnonzero(costs < infinity)
        for first in finite:
            complement = int(first) ^ 0xFF
            costs[complement] = min(
                costs[complement], int(costs[first]) + 1
            )
        finite = np.flatnonzero(costs < infinity)
        for first in finite:
            first_cost = int(costs[first])
            for second in finite:
                candidate = first_cost + int(costs[second]) + 1
                outputs = (
                    int(first) & int(second),
                    int(first) | int(second),
                )
                if include_xor:
                    outputs = (*outputs, int(first) ^ int(second))
                for output in outputs:
                    if candidate < costs[output]:
                        costs[output] = candidate
        if np.array_equal(previous, costs):
            break
    if np.any(costs >= infinity):
        raise RuntimeError("Boolean公式动态规划没有覆盖全部函数。")
    return costs.astype(np.int64)


@functools.lru_cache(maxsize=None)
def decision_tree_test_count(
    function_id: int,
    rows: tuple[int, ...] = tuple(range(8)),
    remaining: tuple[int, ...] = (0, 1, 2),
) -> int:
    labels = tuple((int(function_id) >> row) & 1 for row in rows)
    if len(set(labels)) <= 1:
        return 0
    if not remaining:
        raise RuntimeError("非恒定子函数没有剩余可测试变量。")
    inputs = truth_table_inputs()
    candidates: list[int] = []
    for variable in remaining:
        zero_rows = tuple(
            row for row in rows if int(inputs[row, variable]) == 0
        )
        one_rows = tuple(
            row for row in rows if int(inputs[row, variable]) == 1
        )
        next_remaining = tuple(
            item for item in remaining if item != variable
        )
        candidates.append(
            1
            + decision_tree_test_count(
                function_id, zero_rows, next_remaining
            )
            + decision_tree_test_count(
                function_id, one_rows, next_remaining
            )
        )
    return min(candidates)


def hamming_complexity_metrics(outputs: np.ndarray) -> dict[str, float]:
    point_count = Config.POINT_COUNT
    changes_by_distance: dict[int, list[int]] = {1: [], 2: [], 3: []}
    has_first_neighbor_change: list[bool] = []
    for row in range(point_count):
        local_first: list[int] = []
        for other in range(point_count):
            distance = int((row ^ other).bit_count())
            if distance == 0:
                continue
            changed = int(outputs[row] != outputs[other])
            changes_by_distance[distance].append(changed)
            if distance == 1:
                local_first.append(changed)
        has_first_neighbor_change.append(any(local_first))
    c1 = float(np.mean(changes_by_distance[1]))
    c2 = float(np.mean(changes_by_distance[2]))
    c3 = float(np.mean(changes_by_distance[3]))
    return {
        "distance1_change_fraction": c1,
        "distance2_change_fraction": c2,
        "distance3_change_fraction": c3,
        "mingard_generalization_complexity_c1_c2": c1 + c2,
        "critical_sample_ratio": float(np.mean(has_first_neighbor_change)),
        "total_influence": Config.INPUT_BITS * c1,
    }


def walsh_complexity_metrics(outputs: np.ndarray) -> dict[str, float]:
    spectrum = (2.0 * outputs.astype(np.float64) - 1.0).copy()
    step = 1
    while step < Config.POINT_COUNT:
        for start in range(0, Config.POINT_COUNT, 2 * step):
            for offset in range(step):
                first = spectrum[start + offset]
                second = spectrum[start + offset + step]
                spectrum[start + offset] = first + second
                spectrum[start + offset + step] = first - second
        step *= 2
    spectrum /= Config.POINT_COUNT
    power = spectrum * spectrum
    degrees = np.asarray(
        [index.bit_count() for index in range(Config.POINT_COUNT)],
        dtype=np.float64,
    )
    nonzero = power > 1e-15
    spectral_entropy = float(
        -np.sum(power[nonzero] * np.log2(power[nonzero]))
    )
    active_degrees = degrees[np.abs(spectrum) > 1e-12]
    return {
        "walsh_mean_degree": float(np.sum(degrees * power)),
        "walsh_spectral_entropy": spectral_entropy,
        "walsh_nonzero_coefficients": int(np.sum(nonzero)),
        "walsh_max_degree": (
            int(np.max(active_degrees)) if len(active_degrees) else 0
        ),
        "walsh_degree3_mass": float(np.sum(power[degrees == 3])),
    }


def build_complexity_panel(
    targets: Sequence[TargetSpec],
) -> list[dict[str, Any]]:
    representative_by_function = {
        member: target.function_id
        for target in targets
        for member in target.orbit_members
    }
    formula_aon = formula_gate_costs(include_xor=False)
    formula_aonx = formula_gate_costs(include_xor=True)
    rows: list[dict[str, Any]] = []
    for function_id in range(Config.FUNCTION_COUNT):
        outputs = function_outputs(function_id)
        lz_canonical, lz_minimum, lz_mean, lz_maximum = (
            lz_symmetry_metrics(outputs)
        )
        ctm_canonical, ctm_minimum, ctm_mean, ctm_maximum = (
            ctm_symmetry_metrics(outputs)
        )
        sop = minimum_sop_metrics(function_id)
        sop_complement = minimum_sop_metrics(function_id ^ 0xFF)
        degree, terms, literals = anf_metrics(outputs)
        row: dict[str, Any] = {
            "function_id": function_id,
            "function_hex": f"0x{function_id:02X}",
            "truth_table_x000_to_x111": function_bits(function_id),
            "representative_function_id": (
                representative_by_function[function_id]
            ),
            "is_orbit_representative": (
                representative_by_function[function_id] == function_id
            ),
            "positive_count": int(outputs.sum()),
            "balanced_4_of_8": int(outputs.sum()) == 4,
            "output_entropy_bits": binary_output_entropy(outputs),
            "mingard_lz_canonical": lz_canonical,
            "mingard_lz_symmetry_min": lz_minimum,
            "mingard_lz_symmetry_mean": lz_mean,
            "mingard_lz_symmetry_max": lz_maximum,
            "ctm_b2_d12_canonical": ctm_canonical,
            "ctm_b2_d12_symmetry_min": ctm_minimum,
            "ctm_b2_d12_symmetry_mean": ctm_mean,
            "ctm_b2_d12_symmetry_max": ctm_maximum,
            "sop_operations_canonical": sop[0],
            "sop_terms_canonical": sop[1],
            "sop_literals_canonical": sop[2],
            "boolean_operations_output_symmetric": min(
                sop[0], sop_complement[0]
            ),
            "boolean_terms_output_symmetric": min(
                sop[1], sop_complement[1]
            ),
            "boolean_literals_output_symmetric": min(
                sop[2], sop_complement[2]
            ),
            "formula_gate_count_and_or_not": int(formula_aon[function_id]),
            "formula_gate_count_and_or_not_xor": int(
                formula_aonx[function_id]
            ),
            "formula_gate_count_aon_output_symmetric": int(min(
                formula_aon[function_id], formula_aon[function_id ^ 0xFF]
            )),
            "formula_gate_count_aonx_output_symmetric": int(min(
                formula_aonx[function_id], formula_aonx[function_id ^ 0xFF]
            )),
            "decision_tree_internal_tests": decision_tree_test_count(
                function_id
            ),
            "anf_degree": degree,
            "anf_term_count": terms,
            "anf_literal_count": literals,
            "anf_term_count_output_symmetric": min(
                terms, anf_metrics(1 - outputs)[1]
            ),
            "anf_literal_count_output_symmetric": min(
                literals, anf_metrics(1 - outputs)[2]
            ),
        }
        row.update(hamming_complexity_metrics(outputs))
        row.update(walsh_complexity_metrics(outputs))
        rows.append(row)
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and math.isclose(
            float(values[order[stop]]),
            float(values[order[start]]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def correlation_statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first)
        & np.isfinite(second)
    )
    count = int(valid.sum())
    if (
        count < 3
        or len(np.unique(first[valid])) < 2
        or len(np.unique(second[valid])) < 2
    ):
        return {
            "n": count,
            "spearman": None,
            "kendall_tau": None,
            "pearson": None,
        }
    return {
        "n": count,
        "spearman": float(spearmanr(first[valid], second[valid]).statistic),
        "kendall_tau": float(kendalltau(first[valid], second[valid]).statistic),
        "pearson": float(pearsonr(first[valid], second[valid]).statistic),
    }


def load_architecture_measurements(
    architecture_dir: Path,
) -> dict[str, Any] | None:
    prior_path = architecture_dir / "prior_hard_function_counts.csv"
    curve_path = architecture_dir / "all_256_neural_k_curves.npz"
    if not prior_path.exists() or not curve_path.exists():
        return None
    prior_rows = read_csv(prior_path)
    if len(prior_rows) != Config.FUNCTION_COUNT:
        return None
    prior_rows.sort(key=lambda row: int(row["function_id"]))
    counts = np.asarray(
        [int(row["count"]) for row in prior_rows], dtype=np.int64
    )
    sample_count = int(counts.sum())
    if sample_count <= 0:
        return None
    # Jeffreys平滑只用于让零计数函数可画图；可靠相关另用count>=20筛选。
    probability_smoothed = (
        (counts.astype(np.float64) + 0.5)
        / (sample_count + 0.5 * Config.FUNCTION_COUNT)
    )
    with np.load(curve_path) as payload:
        thresholds = payload["thresholds"].astype(np.float64)
        neural_k = payload["neural_k_bits"].astype(np.float64)
    return {
        "counts": counts,
        "sample_count": sample_count,
        "prior_probability_smoothed": probability_smoothed,
        "prior_k_bits_smoothed": -np.log2(probability_smoothed),
        "prior_reliable": counts >= 20,
        "thresholds": thresholds,
        "neural_k_bits": neural_k,
    }


def nearest_threshold_index(thresholds: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(thresholds) - float(value))))


def write_analysis_report(
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# 3-bit 全函数复杂度顺序：有限宽跨架构结果",
        "",
        "## 测量口径",
        "",
        "所有先验概率都来自真实有限宽网络的Gaussian随机权重计数；所有低loss ",
        "Neural K都来自prior-preserving constrained SMC。主分析没有使用NNGP。",
        "",
        f"完成架构：{', '.join(summary['completed_architectures']) or '无'}。",
        "",
        "复杂度相关同时报告全部函数、4个0/4个1的平衡函数和40个严格对称轨道代表。",
        "平衡函数结果用于排除仅由输出标签比例造成的相关性。",
        "短字符串CTM值来自PyBDM 0.1.0的CTM-B2-D12数据库，并同时报告固定编码",
        "及对输入变量置换和输出取反后的对称化版本。",
        "",
        "## 自动摘要",
        "",
    ]
    for item in summary.get("headline_findings", []):
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 输出文件",
        "",
        "- `function_complexity_panel.csv`：256个函数的全部外部复杂度指标。",
        "- `architecture_prior_panel.csv`：各有限宽架构的直接先验计数和Neural K。",
        "- `cross_architecture_prior_correlations.csv`：先验排序的跨架构相关。",
        "- `cross_architecture_neural_k_correlations.csv`：各loss层级的跨架构相关。",
        "- `prior_to_neural_k_correlations.csv`：先验排序与低loss曲线的关系。",
        "- `prior_to_neural_k_decomposition.csv`：在hard-exact区把总Neural K分成",
        "  先验函数成本与函数内部继续压低BCE的条件收缩成本。",
        "- `cross_architecture_margin_contraction_correlations.csv`：不同架构是否",
        "  以相似顺序压缩各函数内部的低loss区域。",
        "- `complexity_proxy_correlations.csv`：外部复杂度近似与测量结果的关系。",
        "- `rank_trajectories.csv`：每个函数从先验到各loss层级的排名轨迹。",
        "",
        "## 解释边界",
        "",
        "这里的Neural K是相对于明确网络结构和Gaussian初始化测度的可测量复杂度。",
        "外部指标是用来解释其排序的不同描述语言，不被预设为真实Kolmogorov复杂度。",
    ])
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_complexity_order_plots(
    output_dir: Path,
    measurements: dict[str, dict[str, Any]],
    prior_to_curve_rows: Sequence[dict[str, Any]],
    proxy_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(measurements)
    if not names:
        return
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    prior_matrix = np.eye(len(names), dtype=np.float64)
    deep_matrix = np.eye(len(names), dtype=np.float64)
    balanced = np.asarray([
        int(function_id).bit_count() == 4
        for function_id in range(Config.FUNCTION_COUNT)
    ])
    for first_index, first_name in enumerate(names):
        for second_index, second_name in enumerate(names):
            first = measurements[first_name]
            second = measurements[second_name]
            prior_stats = correlation_statistics(
                first["prior_k_bits_smoothed"],
                second["prior_k_bits_smoothed"],
                first["prior_reliable"]
                & second["prior_reliable"]
                & balanced,
            )
            prior_matrix[first_index, second_index] = (
                prior_stats["spearman"]
                if prior_stats["spearman"] is not None else np.nan
            )
            first_deep = first["neural_k_bits"][-1]
            second_deep = second["neural_k_bits"][-1]
            deep_stats = correlation_statistics(
                first_deep, second_deep, balanced
            )
            deep_matrix[first_index, second_index] = (
                deep_stats["spearman"]
                if deep_stats["spearman"] is not None else np.nan
            )

    for axis, matrix, title in (
        (axes[0, 0], prior_matrix, "Finite-network prior rank (balanced)"),
        (axes[0, 1], deep_matrix, "Deep Neural K rank (balanced)"),
    ):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_title(title)
        for row in range(len(names)):
            for column in range(len(names)):
                value = matrix[row, column]
                axis.text(
                    column, row,
                    "NA" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                )
        figure.colorbar(image, ax=axis, fraction=0.046)

    axis = axes[1, 0]
    for name in names:
        rows = [
            row for row in prior_to_curve_rows
            if row["architecture"] == name
            and row["scope"] == "balanced_reliable"
            and row["spearman"] is not None
        ]
        rows.sort(key=lambda row: float(row["threshold"]), reverse=True)
        if rows:
            axis.plot(
                [float(row["threshold"]) for row in rows],
                [float(row["spearman"]) for row in rows],
                marker="o", label=name,
            )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_ylim(-1.02, 1.02)
    axis.set_xlabel("full-truth BCE threshold")
    axis.set_ylabel("Spearman(prior K, Neural K)")
    axis.set_title("Prior ordering retained as loss decreases")
    if axis.lines:
        axis.legend(fontsize=8)

    axis = axes[1, 1]
    deepest_rows: list[dict[str, Any]] = []
    for name in names:
        candidates = [
            row for row in proxy_rows
            if row["architecture"] == name
            and row["scope"] == "balanced"
            and row["measurement"] == "neural_k"
            and row["spearman"] is not None
        ]
        if candidates:
            deepest = min(float(row["threshold"]) for row in candidates)
            deepest_rows.extend(
                row for row in candidates
                if math.isclose(float(row["threshold"]), deepest)
            )
    proxy_names = sorted({row["proxy"] for row in deepest_rows})
    if proxy_names and deepest_rows:
        matrix = np.full((len(names), len(proxy_names)), np.nan)
        for row in deepest_rows:
            matrix[
                names.index(str(row["architecture"])),
                proxy_names.index(str(row["proxy"])),
            ] = float(row["spearman"])
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        axis.set_xticks(
            range(len(proxy_names)), proxy_names, rotation=55, ha="right",
            fontsize=7,
        )
        axis.set_yticks(range(len(names)), names)
        axis.set_title("Complexity proxies vs deepest Neural K (balanced)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    else:
        axis.text(0.5, 0.5, "insufficient completed data", ha="center")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / "complexity_order_summary.png", dpi=180)
    plt.close(figure)


def analyze_complexity_order(
    root_dir: Path,
    architectures: Sequence[ArchitectureSpec],
    targets: Sequence[TargetSpec],
) -> dict[str, Any]:
    analysis_dir = root_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    complexity_rows = build_complexity_panel(targets)
    write_csv(analysis_dir / "function_complexity_panel.csv", complexity_rows)
    complexity_by_name = {
        key: np.asarray([float(row[key]) for row in complexity_rows])
        for key in complexity_rows[0]
        if key not in {
            "function_id", "function_hex", "truth_table_x000_to_x111"
        }
        and all(isinstance(row[key], (int, float, bool)) for row in complexity_rows)
    }

    measurements: dict[str, dict[str, Any]] = {}
    for spec in architectures:
        loaded = load_architecture_measurements(
            root_dir / "architectures" / spec.name
        )
        if loaded is not None:
            measurements[spec.name] = loaded

    prior_panel_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    hard_exact_bound = math.log(2.0) / Config.POINT_COUNT
    for name, data in measurements.items():
        prior_ranks = average_ranks(data["prior_k_bits_smoothed"])
        for function_id in range(Config.FUNCTION_COUNT):
            prior_panel_rows.append({
                "architecture": name,
                "function_id": function_id,
                "function_hex": f"0x{function_id:02X}",
                "count": int(data["counts"][function_id]),
                "sample_count": int(data["sample_count"]),
                "probability_smoothed": float(
                    data["prior_probability_smoothed"][function_id]
                ),
                "prior_k_bits_smoothed": float(
                    data["prior_k_bits_smoothed"][function_id]
                ),
                "prior_rank": float(prior_ranks[function_id]),
                "reliable_count_ge_20": bool(
                    data["prior_reliable"][function_id]
                ),
            })
            rank_rows.append({
                "architecture": name,
                "function_id": function_id,
                "stage": "prior",
                "threshold": None,
                "complexity_bits": float(
                    data["prior_k_bits_smoothed"][function_id]
                ),
                "rank": float(prior_ranks[function_id]),
            })
        for threshold_index, threshold in enumerate(data["thresholds"]):
            local = data["neural_k_bits"][threshold_index]
            finite = np.isfinite(local)
            local_ranks = np.full(Config.FUNCTION_COUNT, np.nan)
            if finite.sum():
                local_ranks[finite] = average_ranks(local[finite])
            for function_id in np.flatnonzero(finite):
                rank_rows.append({
                    "architecture": name,
                    "function_id": int(function_id),
                    "stage": "neural_k",
                    "threshold": float(threshold),
                    "hard_exact_guaranteed_by_threshold": bool(
                        float(threshold) < hard_exact_bound
                    ),
                    "complexity_bits": float(local[function_id]),
                    "rank": float(local_ranks[function_id]),
                })
                if float(threshold) < hard_exact_bound:
                    decomposition_rows.append({
                        "architecture": name,
                        "function_id": int(function_id),
                        "function_hex": f"0x{int(function_id):02X}",
                        "threshold": float(threshold),
                        "hard_function_prior_count": int(
                            data["counts"][function_id]
                        ),
                        "hard_function_prior_reliable_count_ge_20": bool(
                            data["prior_reliable"][function_id]
                        ),
                        "prior_k_bits_smoothed": float(
                            data["prior_k_bits_smoothed"][function_id]
                        ),
                        "full_target_neural_k_bits": float(
                            local[function_id]
                        ),
                        "conditional_margin_contraction_bits": float(
                            local[function_id]
                            - data["prior_k_bits_smoothed"][function_id]
                        ),
                        "prior_rank": float(prior_ranks[function_id]),
                        "neural_k_rank": float(local_ranks[function_id]),
                        "rank_shift_deep_minus_prior": float(
                            local_ranks[function_id] - prior_ranks[function_id]
                        ),
                    })
    write_csv(analysis_dir / "architecture_prior_panel.csv", prior_panel_rows)
    write_csv(analysis_dir / "rank_trajectories.csv", rank_rows)
    write_csv(
        analysis_dir / "prior_to_neural_k_decomposition.csv",
        decomposition_rows,
    )

    balanced = np.asarray([
        int(function_id).bit_count() == 4
        for function_id in range(Config.FUNCTION_COUNT)
    ])
    representative = np.asarray([
        bool(row["is_orbit_representative"]) for row in complexity_rows
    ])
    prior_cross_rows: list[dict[str, Any]] = []
    neural_cross_rows: list[dict[str, Any]] = []
    contraction_cross_rows: list[dict[str, Any]] = []
    names = list(measurements)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            first = measurements[first_name]
            second = measurements[second_name]
            reliable = first["prior_reliable"] & second["prior_reliable"]
            for scope, mask in (
                ("all_reliable", reliable),
                ("balanced_reliable", reliable & balanced),
                ("orbit_representatives_reliable", reliable & representative),
                (
                    "balanced_orbit_representatives_reliable",
                    reliable & balanced & representative,
                ),
            ):
                prior_cross_rows.append({
                    "architecture_a": first_name,
                    "architecture_b": second_name,
                    "scope": scope,
                    **correlation_statistics(
                        first["prior_k_bits_smoothed"],
                        second["prior_k_bits_smoothed"],
                        mask,
                    ),
                })
            common_thresholds = sorted(
                set(map(float, first["thresholds"])).intersection(
                    map(float, second["thresholds"])
                ),
                reverse=True,
            )
            for threshold in common_thresholds:
                first_values = first["neural_k_bits"][
                    nearest_threshold_index(first["thresholds"], threshold)
                ]
                second_values = second["neural_k_bits"][
                    nearest_threshold_index(second["thresholds"], threshold)
                ]
                for scope, mask in (
                    ("all", np.ones(Config.FUNCTION_COUNT, dtype=bool)),
                    ("balanced", balanced),
                    ("orbit_representatives", representative),
                    (
                        "balanced_orbit_representatives",
                        balanced & representative,
                    ),
                ):
                    neural_cross_rows.append({
                        "architecture_a": first_name,
                        "architecture_b": second_name,
                        "threshold": threshold,
                        "hard_exact_guaranteed_by_threshold": bool(
                            threshold < hard_exact_bound
                        ),
                        "scope": scope,
                        **correlation_statistics(
                            first_values, second_values, mask
                        ),
                    })
                    if threshold < hard_exact_bound:
                        first_contraction = (
                            first_values - first["prior_k_bits_smoothed"]
                        )
                        second_contraction = (
                            second_values - second["prior_k_bits_smoothed"]
                        )
                        contraction_cross_rows.append({
                            "architecture_a": first_name,
                            "architecture_b": second_name,
                            "threshold": threshold,
                            "hard_exact_guaranteed_by_threshold": True,
                            "scope": scope,
                            **correlation_statistics(
                                first_contraction,
                                second_contraction,
                                mask
                                & first["prior_reliable"]
                                & second["prior_reliable"],
                            ),
                        })
    write_csv(
        analysis_dir / "cross_architecture_prior_correlations.csv",
        prior_cross_rows,
    )
    write_csv(
        analysis_dir / "cross_architecture_neural_k_correlations.csv",
        neural_cross_rows,
    )
    write_csv(
        analysis_dir / "cross_architecture_margin_contraction_correlations.csv",
        contraction_cross_rows,
    )

    prior_to_curve_rows: list[dict[str, Any]] = []
    for name, data in measurements.items():
        reliable = data["prior_reliable"]
        for threshold_index, threshold in enumerate(data["thresholds"]):
            values = data["neural_k_bits"][threshold_index]
            for scope, mask in (
                ("all_reliable", reliable),
                ("balanced_reliable", reliable & balanced),
                ("orbit_representatives_reliable", reliable & representative),
                (
                    "balanced_orbit_representatives_reliable",
                    reliable & balanced & representative,
                ),
            ):
                prior_to_curve_rows.append({
                    "architecture": name,
                    "threshold": float(threshold),
                    "hard_exact_guaranteed_by_threshold": bool(
                        float(threshold) < hard_exact_bound
                    ),
                    "scope": scope,
                    **correlation_statistics(
                        data["prior_k_bits_smoothed"], values, mask
                    ),
                })
    write_csv(
        analysis_dir / "prior_to_neural_k_correlations.csv",
        prior_to_curve_rows,
    )

    proxy_names = (
        "output_entropy_bits",
        "mingard_lz_canonical",
        "mingard_lz_symmetry_min",
        "mingard_lz_symmetry_mean",
        "ctm_b2_d12_canonical",
        "ctm_b2_d12_symmetry_min",
        "ctm_b2_d12_symmetry_mean",
        "boolean_operations_output_symmetric",
        "formula_gate_count_aon_output_symmetric",
        "formula_gate_count_aonx_output_symmetric",
        "decision_tree_internal_tests",
        "anf_term_count_output_symmetric",
        "anf_literal_count_output_symmetric",
        "mingard_generalization_complexity_c1_c2",
        "critical_sample_ratio",
        "total_influence",
        "walsh_spectral_entropy",
        "walsh_nonzero_coefficients",
        "walsh_degree3_mass",
    )
    proxy_rows: list[dict[str, Any]] = []
    for name, data in measurements.items():
        for proxy_name in proxy_names:
            proxy = complexity_by_name[proxy_name]
            for scope, mask in (
                ("all_reliable", data["prior_reliable"]),
                ("balanced_reliable", data["prior_reliable"] & balanced),
                (
                    "orbit_representatives_reliable",
                    data["prior_reliable"] & representative,
                ),
                (
                    "balanced_orbit_representatives_reliable",
                    data["prior_reliable"] & balanced & representative,
                ),
            ):
                proxy_rows.append({
                    "architecture": name,
                    "measurement": "prior_k",
                    "threshold": None,
                    "scope": scope,
                    "proxy": proxy_name,
                    **correlation_statistics(
                        proxy, data["prior_k_bits_smoothed"], mask
                    ),
                })
            for threshold_index, threshold in enumerate(data["thresholds"]):
                values = data["neural_k_bits"][threshold_index]
                for scope, mask in (
                    ("all", np.ones(Config.FUNCTION_COUNT, dtype=bool)),
                    ("balanced", balanced),
                    ("orbit_representatives", representative),
                    (
                        "balanced_orbit_representatives",
                        balanced & representative,
                    ),
                ):
                    proxy_rows.append({
                        "architecture": name,
                        "measurement": "neural_k",
                        "threshold": float(threshold),
                        "scope": scope,
                        "proxy": proxy_name,
                        **correlation_statistics(proxy, values, mask),
                    })
                if float(threshold) < hard_exact_bound:
                    contraction = (
                        values - data["prior_k_bits_smoothed"]
                    )
                    for scope, mask in (
                        ("all_reliable", data["prior_reliable"]),
                        (
                            "balanced_reliable",
                            data["prior_reliable"] & balanced,
                        ),
                        (
                            "orbit_representatives_reliable",
                            data["prior_reliable"] & representative,
                        ),
                        (
                            "balanced_orbit_representatives_reliable",
                            data["prior_reliable"] & balanced & representative,
                        ),
                    ):
                        proxy_rows.append({
                            "architecture": name,
                            "measurement": "conditional_margin_contraction",
                            "threshold": float(threshold),
                            "scope": scope,
                            "proxy": proxy_name,
                            **correlation_statistics(
                                proxy, contraction, mask
                            ),
                        })
    write_csv(
        analysis_dir / "complexity_proxy_correlations.csv", proxy_rows
    )

    headline: list[str] = []
    prior_balanced = [
        row for row in prior_cross_rows
        if row["scope"] == "balanced_reliable"
        and row["spearman"] is not None
    ]
    if prior_balanced:
        values = [float(row["spearman"]) for row in prior_balanced]
        headline.append(
            "平衡函数的跨架构先验排序Spearman范围为"
            f"{min(values):.3f}到{max(values):.3f}。"
        )
    neural_balanced = [
        row for row in neural_cross_rows
        if row["scope"] == "balanced" and row["spearman"] is not None
    ]
    if neural_balanced:
        deepest = min(float(row["threshold"]) for row in neural_balanced)
        local = [
            float(row["spearman"]) for row in neural_balanced
            if math.isclose(float(row["threshold"]), deepest)
        ]
        headline.append(
            f"在最深共同阈值epsilon={deepest:g}，平衡函数的跨架构"
            f"Neural K排序Spearman范围为{min(local):.3f}到"
            f"{max(local):.3f}。"
        )
    for name in names:
        rows = [
            row for row in prior_to_curve_rows
            if row["architecture"] == name
            and row["scope"] == "balanced_reliable"
            and row["spearman"] is not None
        ]
        if rows:
            deepest_row = min(rows, key=lambda row: float(row["threshold"]))
            headline.append(
                f"{name}的先验排序与最深Neural K排序在平衡函数上的"
                f"Spearman为{float(deepest_row['spearman']):.3f}。"
            )

    summary = {
        "status": "completed" if measurements else "no_complete_architecture",
        "finite_width_only": True,
        "nngp_used": False,
        "completed_architectures": names,
        "function_count": Config.FUNCTION_COUNT,
        "balanced_function_count": int(balanced.sum()),
        "orbit_representative_count": int(representative.sum()),
        "balanced_orbit_representative_count": int(
            np.sum(balanced & representative)
        ),
        "headline_findings": headline,
    }
    save_complexity_order_plots(
        analysis_dir, measurements, prior_to_curve_rows, proxy_rows
    )
    write_analysis_report(analysis_dir, summary)
    write_json(analysis_dir / "summary.json", summary)
    return summary


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.name not in {
                "checkpoint.pt",
                "checkpoint.tmp.pt",
            }:
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def run_architecture(
    spec: ArchitectureSpec,
    architecture_index: int,
    root_dir: Path,
    device: torch.device,
    targets: Sequence[TargetSpec],
    representative_by_function: dict[int, int],
) -> dict[str, Any]:
    activate_architecture(spec, architecture_index)
    validate_config()
    Config.RESULT_DIR = root_dir / "architectures" / spec.name
    output_dir = prepare_result_dir()
    orbit_rows: list[dict[str, Any]] = []
    for target in targets:
        for member in target.orbit_members:
            orbit_rows.append({
                "function_id": member,
                "function_hex": f"0x{member:02X}",
                "truth_table_x000_to_x111": function_bits(member),
                "orbit_index": target.orbit_index,
                "orbit_size": target.orbit_size,
                "representative_function_id": target.function_id,
                "representative_function_hex": target.function_hex,
                "is_representative": member == target.function_id,
            })
    write_csv(output_dir / "exact_symmetry_orbits.csv", orbit_rows)
    write_csv(output_dir / "representative_definitions.csv", [
        asdict(target) for target in targets
    ])
    panel_payload = {
        "protocol": Config.PROTOCOL_VERSION,
        "created_before_smc": True,
        "architecture": asdict(spec),
        "network": spec.display_name,
        "input_encoding": Config.INPUT_ENCODING,
        "prior": Config.PRIOR_KIND,
        "strict_symmetries_used": [
            "permutation_of_input_variables",
            "global_output_complement",
        ],
        "symmetry_explicitly_not_used": "input_bit_complement",
        "thresholds": list(Config.TARGET_THRESHOLDS),
        "targets": [asdict(target) for target in targets],
    }
    canonical = json.dumps(
        json_ready(panel_payload), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    panel_payload["panel_sha256"] = hashlib.sha256(canonical).hexdigest()
    panel_path = output_dir / "preregistered_target_panel.json"
    if panel_path.exists():
        saved = json.loads(panel_path.read_text(encoding="utf-8"))
        if saved != json_ready(panel_payload):
            raise RuntimeError("已有结果的预注册目标面板不一致。")
    else:
        write_json(panel_path, panel_payload)

    inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    blocks, parameter_count = parameter_blocks()
    hard_bound = math.log(2.0) / Config.POINT_COUNT
    print(
        "=== 3-bit finite-width Neural K | "
        f"architecture={spec.name} ===",
        flush=True,
    )
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"network={spec.display_name} | params={parameter_count} | "
        f"input=0/1 | n=8 | hard-exact-bound={hard_bound:.8f}",
        flush=True,
    )
    print(
        "functions=256 | strict symmetry orbits=40 | "
        "merged=input permutations + output complement only | "
        "input complements are NOT merged",
        flush=True,
    )
    print(
        f"replicas={Config.REPLICAS} | "
        f"particles/replica={Config.PARTICLES_PER_REPLICA:,} | "
        f"thresholds={len(Config.TARGET_THRESHOLDS)} | "
        f"deepest={Config.TARGET_THRESHOLDS[-1]:.3g}",
        flush=True,
    )
    print(f"结果目录：{output_dir}", flush=True)

    overall_started = time.perf_counter()
    run_prior_calibration(
        output_dir,
        targets,
        representative_by_function,
        device,
        inputs,
        parameter_count,
    )

    start = max(0, int(Config.TARGET_START))
    stop = len(targets) if Config.TARGET_STOP is None else min(
        len(targets), int(Config.TARGET_STOP)
    )
    if not 0 <= start <= stop <= len(targets):
        raise ValueError(f"非法目标切片：[{start}, {stop})")
    selected = targets[start:stop]
    completed_durations: list[float] = []
    interrupted = False
    statuses: dict[str, str] = {}
    for local_index, target in enumerate(selected, start=1):
        global_ordinal = target.target_index + 1
        print(
            f"\n=== representative {global_ordinal}/{len(targets)} | "
            f"slice {local_index}/{len(selected)} | {target.name} | "
            f"orbit_size={target.orbit_size} | bits="
            f"{target.function_bits_x000_to_x111} ===",
            flush=True,
        )
        status, duration = run_target(
            output_dir,
            target,
            global_ordinal,
            len(targets),
            device,
            inputs,
            blocks,
            parameter_count,
        )
        statuses[target.name] = status
        if status == "completed" and duration > 0:
            completed_durations.append(duration)
        aggregate = aggregate_results(
            output_dir, targets, representative_by_function
        )
        completed_total = aggregate["completed_representative_count"]
        if completed_durations:
            average = float(np.mean(completed_durations))
            remaining_in_slice = len(selected) - local_index
            eta = average * remaining_in_slice
        else:
            eta = None
        print(
            f"GLOBAL completed={completed_total}/{len(targets)} | "
            f"this_slice={local_index}/{len(selected)} | "
            f"last={format_duration(duration)} | "
            f"slice_ETA={format_duration(eta)}",
            flush=True,
        )
        if status == "interrupted":
            interrupted = True
            break

    aggregate = aggregate_results(
        output_dir, targets, representative_by_function
    )
    summary: dict[str, Any] = {
        "status": "interrupted" if interrupted else (
            "completed"
            if aggregate["completed_representative_count"] == len(targets)
            else "partial"
        ),
        "protocol": Config.PROTOCOL_VERSION,
        "elapsed_seconds_this_invocation": time.perf_counter() - overall_started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "architecture": asdict(spec),
        "network": spec.display_name,
        "parameter_count": parameter_count,
        "input_encoding": Config.INPUT_ENCODING,
        "hard_error_lower_bound": hard_bound,
        "target_slice": [start, stop],
        "invocation_statuses": statuses,
        **aggregate,
        "interpretation_boundary": (
            "这些曲线是完整目标在Gaussian参考测度下的静态累计质量；"
            "它们不是fixed-D函数后验，也不是Adam终点概率。"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print("\n=== 当前架构汇总 ===", flush=True)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    apply_cli(args)
    apply_smoke_overrides()
    root_dir = Path(Config.RESULT_DIR)
    if root_dir.exists() and Config.OVERWRITE_RESULT_DIR:
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "architectures").mkdir(exist_ok=True)

    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch看不到GPU。")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.set_float32_matmul_precision("highest")

    targets, representative_by_function = build_targets()
    architecture_start = max(0, int(Config.ARCHITECTURE_START))
    architecture_stop = (
        len(ARCHITECTURES)
        if Config.ARCHITECTURE_STOP is None
        else min(len(ARCHITECTURES), int(Config.ARCHITECTURE_STOP))
    )
    if not 0 <= architecture_start <= architecture_stop <= len(ARCHITECTURES):
        raise ValueError(
            "非法架构切片："
            f"[{architecture_start}, {architecture_stop})"
        )
    selected_architectures = ARCHITECTURES[
        architecture_start:architecture_stop
    ]
    write_json(root_dir / "experiment_plan.json", {
        "protocol": Config.PROTOCOL_VERSION,
        "finite_width_only": True,
        "nngp_used_in_primary_analysis": False,
        "input_encoding": Config.INPUT_ENCODING,
        "architectures": [asdict(item) for item in ARCHITECTURES],
        "selected_architecture_slice": [
            architecture_start, architecture_stop
        ],
        "thresholds": list(Config.TARGET_THRESHOLDS),
        "function_count": Config.FUNCTION_COUNT,
        "strict_symmetry_orbit_count": len(targets),
        "prior_samples_per_architecture": Config.PRIOR_CALIBRATION_SAMPLES,
    })

    overall_started = time.perf_counter()
    architecture_summaries: list[dict[str, Any]] = []
    interrupted = False
    for local_index, spec in enumerate(selected_architectures, start=1):
        architecture_index = architecture_start + local_index - 1
        print(
            f"\n######## ARCHITECTURE {local_index}/"
            f"{len(selected_architectures)} | {spec.display_name} ########",
            flush=True,
        )
        summary = run_architecture(
            spec,
            architecture_index,
            root_dir,
            device,
            targets,
            representative_by_function,
        )
        architecture_summaries.append(summary)
        if summary["status"] == "interrupted":
            interrupted = True
            break

    analysis_summary = analyze_complexity_order(
        root_dir,
        ARCHITECTURES,
        targets,
    )
    final_summary = {
        "status": "interrupted" if interrupted else (
            "completed"
            if all(
                item.get("status") == "completed"
                for item in architecture_summaries
            ) and len(architecture_summaries) == len(selected_architectures)
            else "partial"
        ),
        "protocol": Config.PROTOCOL_VERSION,
        "elapsed_seconds_this_invocation": (
            time.perf_counter() - overall_started
        ),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else ""
        ),
        "selected_architecture_slice": [
            architecture_start, architecture_stop
        ],
        "architecture_summaries": architecture_summaries,
        "analysis": analysis_summary,
    }
    write_json(root_dir / "summary.json", final_summary)
    archive = (
        create_archive(root_dir)
        if Config.PACKAGE_RESULTS and not interrupted else None
    )
    print("\n=== 全实验汇总 ===", flush=True)
    print(
        json.dumps(json_ready(final_summary), ensure_ascii=False, indent=2),
        flush=True,
    )
    if archive is not None:
        print(f"下载压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
