"""E30：显式L2如何重塑balanced-AND的静态完整函数地形。

上传版主协议使用真实有限宽 ``8->16->16->1`` tanh网络、40个balanced-AND
训练样本和标准Gaussian参考测度。三个近似matched-BCE条件分别运行：

``--mode no_wd_static_matched_bce_n40``
    ``lambda=0``，条件为 ``BCE<=0.00268``。
``--mode l2_static_half_lambda_n40``
    ``lambda=5e-5``，条件为 ``BCE+lambda*R<=0.0160``。
``--mode l2_static_reliable_n40``
    ``lambda=1e-4``，条件为 ``BCE+lambda*R<=0.0211``。

其中 ``R=||theta||^2/2``。每个条件使用16个replica、每个8192粒子的direct
constrained SMC，并在完整256点定义域上重算hard function。默认模式是中间
剂量。结果zip、npz和csv不随仓库提交；本脚本会重新生成它们。

脚本仍保留开发阶段的其他诊断模式，便于复核深层、同J和负校准结果；E30主张
只依赖上面三个模式。命令行和Jupyter ``-f kernel.json``均兼容。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr
from scipy.stats import mannwhitneyu, spearmanr


class Config:
    MODE = "l2_static_half_lambda_n40"
    INPUT_BITS = 8
    WIDTH = 16
    HIDDEN_LAYERS = 2
    AND_TRAIN_PER_PRIMARY = (4, 6, 8, 10, 12)
    NUISANCE_ORDER_SEED = 20261020
    PROBE_SUFFIX_START = 12
    PROBE_SUFFIX_STOP = 20
    MATCHED_LOSS = 0.01
    PROTOCOL_VERSION = "8bit_and_gaussian_blind_v2_bridge_smc"

    KERNEL_QUADRATURE_ORDER = 40
    SMC_REPLICAS = 4
    SMC_PARTICLES = 4_096
    SMC_INITIAL_POOL_FACTOR = 16
    SMC_TARGET_ESS_FRACTION = 0.80
    SMC_GIBBS_SWEEPS_PER_STAGE = 2
    SMC_FINAL_GIBBS_SWEEPS = 16
    SMC_MAX_STAGES = 2_000
    SMC_BETA_TOLERANCE = 1e-8
    SMC_LOG_EVERY_STAGES = 5
    LEGACY_IMPORTANCE_REPLICAS = 4
    LEGACY_IMPORTANCE_SAMPLES_PER_REPLICA = 100_000
    PREDICTION_BATCH_SIZE = 5_000
    PREDICTION_SEED = 2026083141

    VALIDATION_SEEDS = 2_048
    VALIDATION_INITIALIZATION_SEED = 2026083142
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    MAX_STEPS = 20_000
    LOG_EVERY_STEPS = 100
    VALIDATION_SEED_CHUNK = 2_048

    # balanced-AND n=32：显式L2动态/静态闭环。
    L2_AND_COEFFICIENTS = (0.0, 1e-4, 1e-3)
    L2_AND_TRAIN_COUNT = 40
    L2_AND_DYNAMIC_SEEDS = 512
    L2_AND_DYNAMIC_STEPS = 20_000
    L2_AND_DYNAMIC_EVAL_INTERVAL = 100
    L2_AND_STATIC_THRESHOLD_FLOOR = 0.003
    L2_AND_STATIC_THRESHOLD_QUANTILE = 0.50
    L2_AND_STATIC_THRESHOLD_MULTIPLIER = 1.50
    L2_AND_ANCHOR_THRESHOLD_RELAXATION = 1.0
    L2_AND_ANCHOR_MAX_STEPS = 20_000
    L2_AND_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_and_n40_explicit_l2_landscape"
    )
    NORM_TARGET_TRAIN_COUNT = 40
    NORM_TARGET_MATCHED_BCE = 0.01
    NORM_TARGET_REPLICAS = 16
    NORM_TARGET_PARTICLES_PER_REPLICA = 8_192
    NORM_TARGET_SAVED_PARAMETERS_PER_REPLICA = 2_048
    NORM_TARGET_LOSS_STRATA = 8
    NORM_TARGET_REWEIGHT_GAMMAS = (
        0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
    )
    NORM_TARGET_MIN_CLASS_COUNT_PER_REPLICA = 20
    NORM_TARGET_MAX_TARGET_MASS_REPLICA_STD = 0.05
    NORM_TARGET_MAX_STRATIFIED_AUC_REPLICA_STD = 0.08
    NORM_TARGET_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_no_wd_norm_target_smc_v2"
    )
    L2_STATIC_COEFFICIENT = 1e-4
    # 独立动态终点为0.01856772--0.01856787。使用略宽的0.0186包络；
    # 这是更保守的充分性检验，也避免在数百有效维的极小值附近无限下潜。
    L2_STATIC_J_THRESHOLD = 0.0186
    L2_STATIC_TRAIN_COUNT = 40
    L2_STATIC_REPLICAS = 16
    L2_STATIC_PARTICLES_PER_REPLICA = 8_192
    L2_STATIC_SAVED_PARAMETERS_PER_REPLICA = 2_048
    L2_STATIC_TARGET_MASS_REPLICA_STD_MAX = 0.05
    L2_STATIC_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_explicit_l2_static_j1p86e2"
    )
    # 从先前下潜日志中只按采样诊断冻结：J约0.0211时logV replica sd约0.7，
    # 尚未读取该层函数结果。此层用于构造可上传的matched-BCE静态对照。
    L2_RELIABLE_J_THRESHOLD = 0.0211
    L2_RELIABLE_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_explicit_l2_static_j2p11e2"
    )
    L2_HIGHER_COEFFICIENT = 2e-4
    # 0.0396试跑得到BCE=0.0053489；lambda=1e-4深层提供可行点
    # (J_2e-4约0.03493, BCE约0.00223)。仅按BCE线性插值到0.00259，
    # 冻结正式阈值0.03545；0.0396试跑不进入剂量比较。
    L2_HIGHER_J_THRESHOLD = 0.03545
    L2_HIGHER_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_explicit_l2_2e4_static_j3p545e2"
    )
    L2_HALF_COEFFICIENT = 5e-5
    # 0.01645校准点得到BCE=0.0027006、R=273.815。仅按BCE/R校准，
    # 目标BCE约0.00259、预期R约268，对应J约0.01599，冻结为0.0160。
    L2_HALF_J_THRESHOLD = 0.0160
    L2_HALF_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_explicit_l2_5e5_static_j1p60e2"
    )
    NO_WD_STATIC_J_THRESHOLD = 0.0186
    NO_WD_STATIC_TRAIN_COUNT = 40
    NO_WD_STATIC_REPLICAS = 16
    NO_WD_STATIC_PARTICLES_PER_REPLICA = 8_192
    NO_WD_STATIC_SAVED_PARAMETERS_PER_REPLICA = 2_048
    NO_WD_STATIC_TARGET_MASS_REPLICA_STD_MAX = 0.05
    NO_WD_STATIC_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_no_wd_static_j1p86e2"
    )
    NO_WD_MATCHED_BCE_THRESHOLD = 0.00268
    NO_WD_MATCHED_BCE_RESULT_DIR = Path(
        "/root/autodl-tmp/results_8bit_n40_no_wd_static_bce2p68e3"
    )
    # 由当前静态分支设置；0保持旧bridge协议不变。
    FW_EXPLICIT_L2_COEFFICIENT = 0.0

    # 真实有限宽433参数 fixed-D bridge-SMC。
    FW_PROTOCOL_VERSION = "8bit_finite_width_fixed_d_bridge_smc_v1"
    FW_REPLICAS = 4
    FW_PARTICLES = 4_096
    FW_TARGET_ESS_FRACTION = 0.80
    FW_MAX_BRIDGE_STAGES = 1_000
    FW_BETA_TOLERANCE = 1e-8
    FW_ADAPT_SWEEPS = 2
    FW_MUTATION_SWEEPS = 4
    FW_FINAL_MUTATION_SWEEPS = 16
    FW_COMPONENT_SWITCH_SWEEPS = 2
    FW_TARGET_ACCEPTANCE = 0.30
    FW_ADAPT_RATE = 0.35
    FW_INITIAL_PCN_SCALES = (0.08, 0.06, 0.08, 0.025)
    FW_MIN_PCN_SCALE = 1e-4
    FW_MAX_PCN_SCALE = 0.95
    FW_ANCHOR_CANDIDATES_PER_REPLICA = 128
    FW_ANCHORS_PER_REPLICA = 48
    FW_ANCHOR_TARGET_LOSS = 0.003
    FW_ANCHOR_MAX_STEPS = 8_000
    FW_ANCHOR_LOG_EVERY = 500
    FW_SIGMA_TARGET_EVENT_RATE = 0.30
    FW_SIGMA_MIN = 1e-4
    FW_SIGMA_MAX = 1.50
    FW_SIGMA_SEARCH_STEPS = 12
    FW_SIGMA_PILOT_SAMPLES = 8_192
    FW_INITIAL_POOL_FACTOR = 4
    FW_EVAL_BATCH = 4_096
    FW_LOG_EVERY_STAGES = 5
    FW_SAVE_PARAMETER_SUBSET = 256
    FW_CONDITION_START = 0
    FW_CONDITION_STOP: int | None = None
    FW_RESUME = True
    FW_ANCHOR_SEED = 2026083161
    FW_SMC_SEED = 2026083162

    # 从标准Gaussian prior直接下潜的有限宽constrained SMC，无任何anchor。
    DIRECT_REPLICAS = 8
    DIRECT_PARTICLES_PER_REPLICA = 4_096
    DIRECT_SURVIVAL_QUANTILE = 0.25
    DIRECT_MAX_LEVELS = 1_000
    DIRECT_ADAPT_SWEEPS = 4
    DIRECT_MUTATION_SWEEPS = 8
    DIRECT_FINAL_MUTATION_SWEEPS = 24
    DIRECT_INITIAL_PCN_SCALES = (0.08, 0.06, 0.08, 0.025)
    DIRECT_MIN_PCN_SCALE = 1e-4
    DIRECT_MAX_PCN_SCALE = 0.95
    DIRECT_TARGET_ACCEPTANCE = 0.30
    DIRECT_ADAPT_RATE = 0.35
    DIRECT_LOG_EVERY_LEVELS = 5
    DIRECT_SEED = 2026083171

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False
    PREDICTION_DIR = Path(
        "/root/autodl-tmp/"
        "results_8bit_and_gaussian_blind_prediction_v2_bridge_smc"
    )
    VALIDATION_DIR = Path(
        "/root/autodl-tmp/"
        "results_8bit_and_gaussian_blind_validation_v2_bridge_smc"
    )
    PREDICTION_MANIFEST = PREDICTION_DIR / "prediction_manifest.json"
    FINITE_WIDTH_DIR = Path(
        "/root/autodl-tmp/results_8bit_finite_width_fixed_d_bridge_smc"
    )
    FINITE_WIDTH_DIRECT_DIR = Path(
        "/root/autodl-tmp/results_8bit_finite_width_fixed_d_direct_smc"
    )
    PACKAGE_RESULTS = True
    SMOKE_TEST = os.environ.get("NSP_SMOKE_TEST", "0") == "1"


@dataclass(frozen=True)
class ConditionSpec:
    condition_index: int
    name: str
    per_primary: int
    train_indices: tuple[int, ...]
    train_count: int


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "predict", "validate", "finite_width", "finite_width_direct",
            "explicit_l2_and_n32", "explicit_l2_and_boundary",
            "norm_target_n40", "l2_static_n40",
            "no_wd_static_same_j_n40",
            "l2_static_reliable_n40",
            "no_wd_static_matched_bce_n40",
            "l2_static_higher_lambda_n40",
            "l2_static_half_lambda_n40",
        ),
        default=None,
        help="临时覆盖Config.MODE；AutoDL Notebook中通常不需要填写。",
    )
    parser.add_argument("--prediction-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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


def apply_args(args: argparse.Namespace) -> None:
    if args.mode is not None:
        Config.MODE = args.mode
    if Config.MODE not in {
        "predict", "validate", "finite_width", "finite_width_direct",
        "explicit_l2_and_n32", "explicit_l2_and_boundary",
        "norm_target_n40", "l2_static_n40",
        "no_wd_static_same_j_n40",
        "l2_static_reliable_n40",
        "no_wd_static_matched_bce_n40",
        "l2_static_higher_lambda_n40",
        "l2_static_half_lambda_n40",
    }:
        raise ValueError(
            "Config.MODE必须是'predict'、'validate'、'finite_width'"
            "、'finite_width_direct'、'explicit_l2_and_n32'或"
            "'explicit_l2_and_boundary'、'norm_target_n40'或"
            "'l2_static_n40'、'no_wd_static_same_j_n40'或"
            "'l2_static_reliable_n40'或"
            "'no_wd_static_matched_bce_n40'或"
            "'l2_static_higher_lambda_n40'或"
            "'l2_static_half_lambda_n40'。"
        )
    if args.prediction_manifest is not None:
        Config.PREDICTION_MANIFEST = args.prediction_manifest
    if args.device is not None:
        Config.DEVICE = args.device
    if args.no_package:
        Config.PACKAGE_RESULTS = False
    if args.smoke:
        Config.SMOKE_TEST = True
    if args.output_dir is not None:
        if Config.MODE == "predict":
            Config.PREDICTION_DIR = args.output_dir
            Config.PREDICTION_MANIFEST = (
                Config.PREDICTION_DIR / "prediction_manifest.json"
            )
        elif Config.MODE == "validate":
            Config.VALIDATION_DIR = args.output_dir
        elif Config.MODE == "finite_width":
            Config.FINITE_WIDTH_DIR = args.output_dir
        elif Config.MODE == "finite_width_direct":
            Config.FINITE_WIDTH_DIRECT_DIR = args.output_dir
        elif Config.MODE in {"explicit_l2_and_n32", "explicit_l2_and_boundary"}:
            Config.L2_AND_RESULT_DIR = args.output_dir
        elif Config.MODE == "norm_target_n40":
            Config.NORM_TARGET_RESULT_DIR = args.output_dir
        elif Config.MODE == "l2_static_n40":
            Config.L2_STATIC_RESULT_DIR = args.output_dir
        elif Config.MODE == "no_wd_static_same_j_n40":
            Config.NO_WD_STATIC_RESULT_DIR = args.output_dir
        elif Config.MODE == "l2_static_reliable_n40":
            Config.L2_RELIABLE_RESULT_DIR = args.output_dir
        elif Config.MODE == "no_wd_static_matched_bce_n40":
            Config.NO_WD_MATCHED_BCE_RESULT_DIR = args.output_dir
        elif Config.MODE == "l2_static_higher_lambda_n40":
            Config.L2_HIGHER_RESULT_DIR = args.output_dir
        else:
            Config.L2_HALF_RESULT_DIR = args.output_dir


def apply_smoke_overrides() -> None:
    if not Config.SMOKE_TEST:
        return
    Config.KERNEL_QUADRATURE_ORDER = 12
    Config.SMC_REPLICAS = 2
    Config.SMC_PARTICLES = 128
    Config.SMC_INITIAL_POOL_FACTOR = 8
    Config.SMC_TARGET_ESS_FRACTION = 0.50
    Config.SMC_GIBBS_SWEEPS_PER_STAGE = 1
    Config.SMC_FINAL_GIBBS_SWEEPS = 2
    Config.SMC_MAX_STAGES = 128
    Config.SMC_LOG_EVERY_STAGES = 1
    Config.LEGACY_IMPORTANCE_REPLICAS = 2
    Config.LEGACY_IMPORTANCE_SAMPLES_PER_REPLICA = 1_000
    Config.PREDICTION_BATCH_SIZE = 250
    Config.VALIDATION_SEEDS = 8
    Config.MAX_STEPS = 2
    Config.LOG_EVERY_STEPS = 1
    Config.MATCHED_LOSS = 0.70
    Config.DEVICE = "cpu"
    Config.PREDICTION_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_and_gaussian_blind_prediction_v2_bridge_smc"
    )
    Config.PREDICTION_MANIFEST = (
        Config.PREDICTION_DIR / "prediction_manifest.json"
    )
    Config.VALIDATION_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_and_gaussian_blind_validation_v2_bridge_smc"
    )
    Config.FW_REPLICAS = 2
    Config.FW_PARTICLES = 128
    Config.FW_TARGET_ESS_FRACTION = 0.50
    Config.FW_MAX_BRIDGE_STAGES = 64
    Config.FW_ADAPT_SWEEPS = 1
    Config.FW_MUTATION_SWEEPS = 1
    Config.FW_FINAL_MUTATION_SWEEPS = 2
    Config.FW_COMPONENT_SWITCH_SWEEPS = 1
    Config.FW_ANCHOR_CANDIDATES_PER_REPLICA = 12
    Config.FW_ANCHORS_PER_REPLICA = 6
    Config.FW_ANCHOR_TARGET_LOSS = 0.60
    Config.FW_ANCHOR_MAX_STEPS = 100
    Config.FW_ANCHOR_LOG_EVERY = 20
    Config.FW_SIGMA_PILOT_SAMPLES = 256
    Config.FW_SIGMA_SEARCH_STEPS = 5
    Config.FW_INITIAL_POOL_FACTOR = 4
    Config.FW_EVAL_BATCH = 128
    Config.FW_LOG_EVERY_STAGES = 1
    Config.FW_SAVE_PARAMETER_SUBSET = 16
    Config.FW_CONDITION_STOP = 1
    Config.FINITE_WIDTH_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_finite_width_fixed_d_bridge_smc"
    )
    Config.DIRECT_REPLICAS = 2
    Config.DIRECT_PARTICLES_PER_REPLICA = 128
    Config.DIRECT_SURVIVAL_QUANTILE = 0.50
    Config.DIRECT_MAX_LEVELS = 64
    Config.DIRECT_ADAPT_SWEEPS = 1
    Config.DIRECT_MUTATION_SWEEPS = 1
    Config.DIRECT_FINAL_MUTATION_SWEEPS = 2
    Config.DIRECT_LOG_EVERY_LEVELS = 1
    Config.FINITE_WIDTH_DIRECT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_finite_width_fixed_d_direct_smc"
    )
    Config.L2_AND_COEFFICIENTS = (0.0, 1e-4)
    Config.L2_AND_DYNAMIC_SEEDS = 8
    Config.L2_AND_TRAIN_COUNT = 40
    Config.L2_AND_DYNAMIC_STEPS = 2
    Config.L2_AND_DYNAMIC_EVAL_INTERVAL = 1
    Config.L2_AND_STATIC_THRESHOLD_FLOOR = 0.70
    Config.L2_AND_STATIC_THRESHOLD_MULTIPLIER = 1.5
    Config.L2_AND_ANCHOR_THRESHOLD_RELAXATION = 1.0
    Config.L2_AND_ANCHOR_MAX_STEPS = 120
    Config.L2_AND_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_and_n40_explicit_l2_landscape"
    )
    Config.NORM_TARGET_TRAIN_COUNT = 40
    Config.NORM_TARGET_MATCHED_BCE = 0.70
    Config.NORM_TARGET_REPLICAS = 2
    Config.NORM_TARGET_PARTICLES_PER_REPLICA = 128
    Config.NORM_TARGET_SAVED_PARAMETERS_PER_REPLICA = 64
    Config.NORM_TARGET_LOSS_STRATA = 2
    Config.NORM_TARGET_MIN_CLASS_COUNT_PER_REPLICA = 1
    Config.NORM_TARGET_MAX_TARGET_MASS_REPLICA_STD = 1.0
    Config.NORM_TARGET_MAX_STRATIFIED_AUC_REPLICA_STD = 1.0
    Config.NORM_TARGET_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_no_wd_norm_target_smc_v2"
    )
    Config.L2_STATIC_J_THRESHOLD = 0.70
    Config.L2_STATIC_REPLICAS = 2
    Config.L2_STATIC_PARTICLES_PER_REPLICA = 128
    Config.L2_STATIC_SAVED_PARAMETERS_PER_REPLICA = 64
    Config.L2_STATIC_TARGET_MASS_REPLICA_STD_MAX = 1.0
    Config.L2_STATIC_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_explicit_l2_static"
    )
    Config.L2_RELIABLE_J_THRESHOLD = 0.70
    Config.L2_RELIABLE_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_explicit_l2_static_reliable"
    )
    Config.L2_HIGHER_J_THRESHOLD = 0.70
    Config.L2_HIGHER_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_explicit_l2_higher_lambda"
    )
    Config.L2_HALF_J_THRESHOLD = 0.70
    Config.L2_HALF_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_explicit_l2_half_lambda"
    )
    Config.NO_WD_STATIC_J_THRESHOLD = 0.70
    Config.NO_WD_STATIC_REPLICAS = 2
    Config.NO_WD_STATIC_PARTICLES_PER_REPLICA = 128
    Config.NO_WD_STATIC_SAVED_PARAMETERS_PER_REPLICA = 64
    Config.NO_WD_STATIC_TARGET_MASS_REPLICA_STD_MAX = 1.0
    Config.NO_WD_STATIC_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_no_wd_static_same_j"
    )
    Config.NO_WD_MATCHED_BCE_THRESHOLD = 0.70
    Config.NO_WD_MATCHED_BCE_RESULT_DIR = Path(
        "research/function_information_conservation/"
        "_smoke_8bit_n40_no_wd_static_matched_bce"
    )
    Config.PACKAGE_RESULTS = False


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


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        json_ready(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                )
                for key, value in row.items()
            })


def truth_table_inputs() -> np.ndarray:
    values = np.arange(256, dtype=np.uint16)
    shifts = np.arange(7, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None]) & 1).astype(np.float64)


def target_outputs() -> np.ndarray:
    inputs = truth_table_inputs()
    return (inputs[:, 0] * inputs[:, 1]).astype(np.uint8)


def balanced_nuisance_order() -> tuple[int, ...]:
    rng = np.random.default_rng(Config.NUISANCE_ORDER_SEED)
    representatives = np.arange(32, dtype=np.int64)
    rng.shuffle(representatives)
    order: list[int] = []
    for value in representatives:
        order.extend((int(value), int(63 - value)))
    return tuple(order)


def build_protocol_objects() -> tuple[
    list[ConditionSpec], tuple[int, ...], np.ndarray
]:
    nuisance_order = balanced_nuisance_order()
    conditions: list[ConditionSpec] = []
    for condition_index, per_primary in enumerate(
        Config.AND_TRAIN_PER_PRIMARY
    ):
        suffixes = nuisance_order[:per_primary]
        train_indices = tuple(sorted(
            (primary << 6) | suffix
            for primary in range(4)
            for suffix in suffixes
        ))
        conditions.append(ConditionSpec(
            condition_index=condition_index,
            name=f"and_n{len(train_indices)}",
            per_primary=per_primary,
            train_indices=train_indices,
            train_count=len(train_indices),
        ))
    probe_suffixes = nuisance_order[
        Config.PROBE_SUFFIX_START:Config.PROBE_SUFFIX_STOP
    ]
    probe_indices = np.asarray(sorted(
        (primary << 6) | suffix
        for primary in range(4)
        for suffix in probe_suffixes
    ), dtype=np.int64)
    if len(probe_indices) != 32 or len(np.unique(probe_indices)) != 32:
        raise AssertionError("probe数量或唯一性错误。")
    for condition in conditions:
        if np.intersect1d(
            np.asarray(condition.train_indices), probe_indices
        ).size:
            raise AssertionError(f"{condition.name}与probe发生重叠。")
    probes_by_primary = probe_indices >> 6
    if not np.array_equal(np.bincount(
        probes_by_primary, minlength=4
    ), np.full(4, 8)):
        raise AssertionError("四个主语义格的probe数量不平衡。")
    return conditions, probe_suffixes, probe_indices


def protocol_payload() -> dict[str, Any]:
    conditions, probe_suffixes, probe_indices = build_protocol_objects()
    return {
        "protocol_version": Config.PROTOCOL_VERSION,
        "scientific_status": "predictions_must_be_frozen_before_validation",
        "network": {
            "input_bits": Config.INPUT_BITS,
            "width": Config.WIDTH,
            "hidden_layers": Config.HIDDEN_LAYERS,
            "activation": "tanh",
            "parameter_coordinates": "independent_standard_gaussian",
            "fan_in_scaling": True,
            "parameter_count": 433,
        },
        "task": {
            "target": "x0 AND x1",
            "input_encoding": "zero_one",
            "domain_size": 256,
            "nuisance_order_seed": Config.NUISANCE_ORDER_SEED,
            "nuisance_order": balanced_nuisance_order(),
            "conditions": [asdict(condition) for condition in conditions],
            "probe_suffix_slice": [
                Config.PROBE_SUFFIX_START, Config.PROBE_SUFFIX_STOP
            ],
            "probe_suffixes": probe_suffixes,
            "probe_indices": probe_indices,
            "probe_targets": target_outputs()[probe_indices],
        },
        "matched_loss": Config.MATCHED_LOSS,
        "predictor": {
            "kind": "recursive_nngp_adaptive_bridge_smc",
            "kernel_quadrature_order": Config.KERNEL_QUADRATURE_ORDER,
            "smc_replicas": Config.SMC_REPLICAS,
            "smc_particles": Config.SMC_PARTICLES,
            "initial_pool_factor": Config.SMC_INITIAL_POOL_FACTOR,
            "target_ess_fraction": Config.SMC_TARGET_ESS_FRACTION,
            "gibbs_sweeps_per_stage": Config.SMC_GIBBS_SWEEPS_PER_STAGE,
            "final_gibbs_sweeps": Config.SMC_FINAL_GIBBS_SWEEPS,
            "max_stages": Config.SMC_MAX_STAGES,
            "beta_tolerance": Config.SMC_BETA_TOLERANCE,
            "legacy_importance_replicas": (
                Config.LEGACY_IMPORTANCE_REPLICAS
            ),
            "legacy_importance_samples_per_replica": (
                Config.LEGACY_IMPORTANCE_SAMPLES_PER_REPLICA
            ),
            "seed": Config.PREDICTION_SEED,
        },
        "validator": {
            "optimizer": "AdamW",
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "betas": Config.ADAM_BETAS,
            "eps": Config.ADAM_EPS,
            "seed_count": Config.VALIDATION_SEEDS,
            "initialization_seed": Config.VALIDATION_INITIALIZATION_SEED,
            "max_steps": Config.MAX_STEPS,
        },
    }


def tanh_covariance(
    covariance: np.ndarray,
    order: int,
) -> np.ndarray:
    raw_nodes, raw_weights = hermgauss(order)
    nodes = math.sqrt(2.0) * raw_nodes
    weights = raw_weights / math.sqrt(math.pi)
    count = covariance.shape[0]
    output = np.empty_like(covariance, dtype=np.float64)
    for first in range(count):
        for second in range(count):
            first_variance = float(covariance[first, first])
            second_variance = float(covariance[second, second])
            cross = float(covariance[first, second])
            first_value = math.sqrt(first_variance) * nodes[:, None]
            residual = max(
                second_variance - cross * cross / first_variance,
                0.0,
            )
            second_value = (
                cross / math.sqrt(first_variance) * nodes[:, None]
                + math.sqrt(residual) * nodes[None, :]
            )
            output[first, second] = np.sum(
                weights[:, None]
                * weights[None, :]
                * np.tanh(first_value)
                * np.tanh(second_value)
            )
    return output


def recursive_nngp_kernel(inputs: np.ndarray) -> dict[str, np.ndarray]:
    first_preactivation = (inputs @ inputs.T + 1.0) / Config.INPUT_BITS
    first_hidden = tanh_covariance(
        first_preactivation, Config.KERNEL_QUADRATURE_ORDER
    )
    second_preactivation = first_hidden + np.ones_like(first_hidden) / Config.WIDTH
    second_hidden = tanh_covariance(
        second_preactivation, Config.KERNEL_QUADRATURE_ORDER
    )
    output_kernel = second_hidden + np.ones_like(second_hidden) / Config.WIDTH
    output_kernel = 0.5 * (output_kernel + output_kernel.T)
    minimum = float(np.linalg.eigvalsh(output_kernel).min())
    if minimum <= 0:
        output_kernel += np.eye(len(output_kernel)) * (1e-10 - minimum)
    return {
        "first_preactivation": first_preactivation,
        "first_hidden": first_hidden,
        "second_preactivation": second_preactivation,
        "second_hidden": second_hidden,
        "output_kernel": output_kernel,
    }


def solve_gaussian_tilt(
    train_kernel: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    signed = 2.0 * labels.astype(np.float64) - 1.0
    cholesky = np.linalg.cholesky(train_kernel)

    def loss(white: np.ndarray) -> float:
        logits = cholesky @ white
        return float(np.logaddexp(0.0, -signed * logits).mean())

    def constraint_gradient(white: np.ndarray) -> np.ndarray:
        logits = cholesky @ white
        return cholesky.T @ (
            signed
            / (1.0 + np.exp(np.clip(signed * logits, -700.0, 700.0)))
            / len(labels)
        )

    margin = -math.log(math.expm1(epsilon))
    initial = np.linalg.solve(cholesky, signed * margin)
    result = minimize(
        lambda white: float(0.5 * white @ white),
        initial,
        jac=lambda white: white,
        constraints={
            "type": "ineq",
            "fun": lambda white: epsilon - loss(white),
            "jac": constraint_gradient,
        },
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 5_000},
    )
    final_loss = loss(result.x)
    if not result.success and abs(final_loss-epsilon) > 2e-6:
        raise RuntimeError(
            f"Gaussian tilt求解失败：{result.message}; loss={final_loss}"
        )
    train_logits = cholesky @ result.x
    tilt = np.linalg.solve(train_kernel, train_logits)
    log_normalizer = float(0.5 * tilt @ train_kernel @ tilt)
    return tilt, train_logits, log_normalizer


def weighted_probe_metrics(
    probe_probability_one: np.ndarray,
    probe_targets: np.ndarray,
) -> dict[str, Any]:
    probability = np.asarray(probe_probability_one, dtype=np.float64)
    targets = np.asarray(probe_targets, dtype=np.uint8)
    target_probability = np.where(targets == 1, probability, 1.0-probability)
    agreement = np.mean(probability**2 + (1.0-probability)**2)
    modal = (probability >= 0.5).astype(np.uint8)
    return {
        "probe_target_accuracy": float(target_probability.mean()),
        "probe_pairwise_agreement": float(agreement),
        "probe_modal_accuracy": float(np.mean(modal == targets)),
        "probe_modal_bits": "".join(map(str, modal.tolist())),
        "mean_target_probability": float(target_probability.mean()),
    }


def gaussian_tilt_replica(
    kernel: np.ndarray,
    train_positions: np.ndarray,
    probe_positions: np.ndarray,
    train_labels: np.ndarray,
    tilt: np.ndarray,
    log_normalizer: float,
    epsilon: float,
    sample_count: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    local_kernel = torch.as_tensor(kernel, dtype=torch.float64, device=device)
    cholesky = torch.linalg.cholesky(local_kernel)
    mean = torch.as_tensor(
        kernel[:, train_positions] @ tilt,
        dtype=torch.float64,
        device=device,
    )
    local_tilt = torch.as_tensor(tilt, dtype=torch.float64, device=device)
    labels = torch.as_tensor(train_labels, dtype=torch.float64, device=device)
    train_index = torch.as_tensor(train_positions, dtype=torch.long, device=device)
    probe_index = torch.as_tensor(probe_positions, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    log_denominator = -float("inf")
    log_second = -float("inf")
    log_numerator = np.full(len(probe_positions), -np.inf, dtype=np.float64)
    event_count = 0
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count-start)
        noise = torch.randn(
            count, len(kernel),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        logits = mean[None] + noise @ cholesky.T
        train_logits = logits[:, train_index]
        train_loss = F.binary_cross_entropy_with_logits(
            train_logits,
            labels[None].expand_as(train_logits),
            reduction="none",
        ).mean(dim=1)
        event = train_loss <= epsilon
        selected_logits = logits[event]
        selected_train = train_logits[event]
        event_count += int(event.sum().item())
        if not len(selected_logits):
            continue
        log_weight = (
            -selected_train @ local_tilt + log_normalizer
        )
        local_denominator = float(torch.logsumexp(
            log_weight, dim=0
        ).item())
        local_second = float(torch.logsumexp(
            2.0*log_weight, dim=0
        ).item())
        log_denominator = float(np.logaddexp(
            log_denominator, local_denominator
        ))
        log_second = float(np.logaddexp(log_second, local_second))
        probe_positive = selected_logits[:, probe_index] >= 0
        for probe in range(len(probe_positions)):
            positive_weight = log_weight[probe_positive[:, probe]]
            if len(positive_weight):
                log_numerator[probe] = float(np.logaddexp(
                    log_numerator[probe],
                    float(torch.logsumexp(positive_weight, dim=0).item()),
                ))
    if not math.isfinite(log_denominator):
        raise RuntimeError("NNGP importance sampler没有命中low-loss事件。")
    probability_one = np.exp(log_numerator-log_denominator)
    probability_one[~np.isfinite(probability_one)] = 0.0
    return {
        "log_volume": log_denominator-math.log(sample_count),
        "effective_sample_size": math.exp(
            2.0*log_denominator-log_second
        ),
        "event_rate": event_count/sample_count,
        "probe_probability_one": probability_one,
    }


def lower_truncated_standard_normal(
    lower: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """逐元素采样 ``N(0,1) | value >= lower``。

    正尾使用Robert指数拒绝采样，非正阈值使用普通高斯拒绝采样；两支在
    极深尾中都避免直接计算接近1的Gaussian CDF。
    """
    result = torch.empty_like(lower)
    pending = torch.ones_like(lower, dtype=torch.bool)
    tiny = torch.finfo(lower.dtype).tiny
    while bool(torch.any(pending).item()):
        locations = torch.nonzero(pending, as_tuple=False).squeeze(1)
        local_lower = lower[locations]
        positive = local_lower > 0
        candidate = torch.empty_like(local_lower)
        accepted = torch.empty_like(positive)
        if bool(torch.any(positive).item()):
            threshold = local_lower[positive]
            rate = 0.5 * (threshold + torch.sqrt(threshold**2 + 4.0))
            uniform = torch.rand(
                len(threshold),
                dtype=lower.dtype,
                device=lower.device,
                generator=generator,
            ).clamp_min(tiny)
            proposal = threshold - torch.log(uniform) / rate
            check = torch.rand(
                len(threshold),
                dtype=lower.dtype,
                device=lower.device,
                generator=generator,
            ).clamp_min(tiny)
            candidate[positive] = proposal
            accepted[positive] = (
                torch.log(check) <= -0.5 * (proposal-rate) ** 2
            )
        negative = ~positive
        if bool(torch.any(negative).item()):
            proposal = torch.randn(
                int(negative.sum().item()),
                dtype=lower.dtype,
                device=lower.device,
                generator=generator,
            )
            candidate[negative] = proposal
            accepted[negative] = proposal >= local_lower[negative]
        accepted_locations = locations[accepted]
        result[accepted_locations] = candidate[accepted]
        pending[accepted_locations] = False
    return result


def draw_tilted_event_particles(
    mean: torch.Tensor,
    cholesky: torch.Tensor,
    signed_labels: torch.Tensor,
    epsilon: float,
    particle_count: int,
    pool_factor: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, int, int]:
    """从易命中的tilted Gaussian中拒绝采样，得到精确的初始条件粒子。"""
    dimension = len(mean)
    pool_size = max(particle_count * pool_factor, batch_size)
    generated = 0
    accepted_total = 0
    accepted_chunks: list[torch.Tensor] = []
    while sum(len(chunk) for chunk in accepted_chunks) < particle_count:
        remaining = pool_size
        while remaining:
            count = min(batch_size, remaining)
            noise = torch.randn(
                count,
                dimension,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            logits = mean[None] + noise @ cholesky.T
            losses = F.softplus(
                -logits * signed_labels[None]
            ).mean(dim=1)
            event = losses <= epsilon
            selected = logits[event]
            if len(selected):
                accepted_chunks.append(selected)
            accepted_total += int(event.sum().item())
            generated += count
            remaining -= count
        if accepted_total == 0 and generated >= 4 * pool_size:
            raise RuntimeError("tilted Gaussian未命中目标loss事件。")
    accepted = torch.cat(accepted_chunks, dim=0)
    selection = torch.randperm(
        len(accepted), device=mean.device, generator=generator
    )[:particle_count]
    return (
        accepted[selection].clone(),
        accepted_total / generated,
        accepted_total,
        generated,
    )


def constrained_gaussian_gibbs(
    particles: torch.Tensor,
    mean: torch.Tensor,
    precision: torch.Tensor,
    signed_labels: torch.Tensor,
    epsilon: float,
    sweeps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """对Gaussian在BCE亚水平集上的条件分布执行精确坐标Gibbs更新。"""
    if sweeps <= 0:
        losses = F.softplus(-particles * signed_labels[None]).mean(dim=1)
        return particles, {
            "gibbs_rms_displacement": 0.0,
            "mean_loss": float(losses.mean().item()),
            "max_loss": float(losses.max().item()),
        }
    started = particles.clone()
    dimension = particles.shape[1]
    loss_terms = F.softplus(-particles * signed_labels[None])
    loss_sum = loss_terms.sum(dim=1)
    residual = (particles-mean[None]) @ precision
    diagonal = torch.diagonal(precision)
    total_budget = dimension * epsilon
    tiny = torch.finfo(particles.dtype).tiny
    for _ in range(sweeps):
        order = torch.randperm(
            dimension, device=particles.device, generator=generator
        )
        for coordinate_tensor in order:
            coordinate = int(coordinate_tensor.item())
            old_value = particles[:, coordinate].clone()
            old_loss = loss_terms[:, coordinate].clone()
            conditional_mean = (
                old_value-residual[:, coordinate]/diagonal[coordinate]
            )
            conditional_sd = torch.rsqrt(diagonal[coordinate])
            budget = (
                total_budget-(loss_sum-old_loss)
            ).clamp_min(tiny)
            lower_margin = -torch.log(torch.expm1(budget))
            sign = signed_labels[coordinate]
            signed_mean = sign * conditional_mean
            standardized_lower = (
                lower_margin-signed_mean
            ) / conditional_sd
            standard = lower_truncated_standard_normal(
                standardized_lower, generator
            )
            new_margin = signed_mean + conditional_sd * standard
            new_value = sign * new_margin
            new_loss = F.softplus(-new_margin)
            delta = new_value-old_value
            particles[:, coordinate] = new_value
            loss_terms[:, coordinate] = new_loss
            loss_sum += new_loss-old_loss
            residual += delta[:, None] * precision[coordinate][None]
    losses = loss_sum / dimension
    maximum = float(losses.max().item())
    if maximum > epsilon + 5e-10:
        raise RuntimeError(
            f"截断Gaussian Gibbs越过loss边界：{maximum} > {epsilon}"
        )
    return particles, {
        "gibbs_rms_displacement": float(torch.sqrt(torch.mean(
            (particles-started) ** 2
        )).item()),
        "mean_loss": float(losses.mean().item()),
        "max_loss": maximum,
    }


def log_weight_ess(log_weight: torch.Tensor) -> float:
    normalized = torch.softmax(log_weight, dim=0)
    return float((1.0 / torch.sum(normalized**2)).item())


def systematic_resample(
    normalized_weight: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    count = len(normalized_weight)
    start = torch.rand(
        (),
        dtype=normalized_weight.dtype,
        device=normalized_weight.device,
        generator=generator,
    ) / count
    positions = start + torch.arange(
        count,
        dtype=normalized_weight.dtype,
        device=normalized_weight.device,
    ) / count
    cumulative = torch.cumsum(normalized_weight, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False)


def bridge_smc_replica(
    condition_name: str,
    replica_index: int,
    train_kernel: np.ndarray,
    train_labels: np.ndarray,
    tilted_mean: np.ndarray,
    tilt: np.ndarray,
    epsilon: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """从tilted event逐步桥接到原始NNGP event，并返回最终条件粒子。"""
    dtype = torch.float64
    local_kernel = torch.as_tensor(train_kernel, dtype=dtype, device=device)
    cholesky = torch.linalg.cholesky(local_kernel)
    precision = torch.cholesky_inverse(cholesky)
    precision = 0.5 * (precision+precision.T)
    labels = torch.as_tensor(train_labels, dtype=dtype, device=device)
    signed = 2.0 * labels-1.0
    mean_zero = torch.as_tensor(tilted_mean, dtype=dtype, device=device)
    local_tilt = torch.as_tensor(tilt, dtype=dtype, device=device)
    norm_squared = float((mean_zero @ local_tilt).item())
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    particles, initial_rate, accepted, generated = draw_tilted_event_particles(
        mean_zero,
        cholesky,
        signed,
        epsilon,
        Config.SMC_PARTICLES,
        Config.SMC_INITIAL_POOL_FACTOR,
        Config.PREDICTION_BATCH_SIZE,
        generator,
    )
    ancestors = torch.arange(
        Config.SMC_PARTICLES, device=device, dtype=torch.long
    )
    beta = 0.0
    log_volume = math.log(initial_rate)
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    target_ess = Config.SMC_TARGET_ESS_FRACTION * Config.SMC_PARTICLES
    for stage in range(1, Config.SMC_MAX_STAGES+1):
        score = particles @ local_tilt

        def ess_at(candidate: float) -> float:
            return log_weight_ess(-(candidate-beta) * score)

        if ess_at(1.0) >= target_ess:
            next_beta = 1.0
        else:
            lower_beta = beta
            upper_beta = 1.0
            for _ in range(50):
                midpoint = 0.5 * (lower_beta+upper_beta)
                if ess_at(midpoint) >= target_ess:
                    lower_beta = midpoint
                else:
                    upper_beta = midpoint
            next_beta = lower_beta
        if next_beta-beta < Config.SMC_BETA_TOLERANCE:
            raise RuntimeError(
                f"{condition_name} replica={replica_index} bridge beta停滞于"
                f"{beta:.8f}。"
            )
        delta = next_beta-beta
        constant = -0.5 * (
            (1.0-next_beta) ** 2-(1.0-beta) ** 2
        ) * norm_squared
        log_weight = -delta * score + constant
        incremental_ess = log_weight_ess(log_weight)
        log_volume += float(
            torch.logsumexp(log_weight, dim=0).item()
            - math.log(Config.SMC_PARTICLES)
        )
        normalized = torch.softmax(log_weight, dim=0)
        resampled = systematic_resample(normalized, generator)
        particles = particles[resampled].clone()
        ancestors = ancestors[resampled]
        beta = next_beta
        bridge_mean = (1.0-beta) * mean_zero
        particles, gibbs = constrained_gaussian_gibbs(
            particles,
            bridge_mean,
            precision,
            signed,
            epsilon,
            Config.SMC_GIBBS_SWEEPS_PER_STAGE,
            generator,
        )
        elapsed = time.perf_counter()-started
        diagnostic = {
            "condition": condition_name,
            "replica": replica_index,
            "stage": stage,
            "beta": beta,
            "incremental_ess": incremental_ess,
            "incremental_ess_fraction": (
                incremental_ess/Config.SMC_PARTICLES
            ),
            "log_volume": log_volume,
            "unique_ancestor_fraction": float(
                torch.unique(ancestors).numel()/Config.SMC_PARTICLES
            ),
            "elapsed_seconds": elapsed,
            **gibbs,
        }
        diagnostics.append(diagnostic)
        if (
            stage == 1
            or stage % Config.SMC_LOG_EVERY_STAGES == 0
            or beta >= 1.0
        ):
            eta = elapsed * (1.0-beta) / max(beta, 1e-12)
            print(
                f"SMC {condition_name} r={replica_index+1}/"
                f"{Config.SMC_REPLICAS} stage={stage} beta={beta:.5f} | "
                f"ESS={incremental_ess/Config.SMC_PARTICLES:.3f} | "
                f"loss={gibbs['mean_loss']:.5g}/"
                f"{gibbs['max_loss']:.5g} | "
                f"move={gibbs['gibbs_rms_displacement']:.3g} | "
                f"elapsed={elapsed:.1f}s ETA~{eta:.1f}s",
                flush=True,
            )
        if beta >= 1.0:
            break
    else:
        raise RuntimeError(
            f"{condition_name} replica={replica_index}超过最大bridge stages。"
        )
    particles, final_gibbs = constrained_gaussian_gibbs(
        particles,
        torch.zeros_like(mean_zero),
        precision,
        signed,
        epsilon,
        Config.SMC_FINAL_GIBBS_SWEEPS,
        generator,
    )
    return {
        "train_logits": particles.detach().cpu().numpy(),
        "log_volume": log_volume,
        "initial_event_rate": initial_rate,
        "initial_accepted": accepted,
        "initial_generated": generated,
        "stage_count": len(diagnostics),
        "minimum_incremental_ess_fraction": min(
            row["incremental_ess_fraction"] for row in diagnostics
        ),
        "final_unique_ancestor_fraction": diagnostics[-1][
            "unique_ancestor_fraction"
        ],
        "final_gibbs": final_gibbs,
        "diagnostics": diagnostics,
    }


def conditional_probe_probabilities(
    train_kernel: np.ndarray,
    probe_train_kernel: np.ndarray,
    probe_kernel: np.ndarray,
    train_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rao-Blackwell化计算每个probe为正的条件概率。"""
    solve = np.linalg.solve(train_kernel, probe_train_kernel.T)
    conditional_mean = train_logits @ solve
    conditional_covariance = probe_kernel-probe_train_kernel @ solve
    conditional_variance = np.maximum(
        np.diag(conditional_covariance), 1e-15
    )
    probability = ndtr(
        conditional_mean / np.sqrt(conditional_variance)[None]
    ).mean(axis=0)
    return probability, conditional_mean, conditional_variance


def and_swap_symmetrize(probability: np.ndarray) -> tuple[np.ndarray, float]:
    """利用x0/x1交换的严格对称性做Rao-Blackwell化，并返回原始残差。"""
    result = np.asarray(probability, dtype=np.float64).copy()
    first = result[8:16].copy()
    second = result[16:24].copy()
    residual = float(np.max(np.abs(first-second)))
    average = 0.5 * (first+second)
    result[8:16] = average
    result[16:24] = average
    return result, residual


def run_prediction(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    protocol_hash = payload_sha256(protocol)
    conditions, _, probe_indices = build_protocol_objects()
    full_inputs = truth_table_inputs()
    targets = target_outputs()
    union_indices = np.asarray(sorted(set(
        conditions[-1].train_indices
    ).union(map(int, probe_indices))), dtype=np.int64)
    union_lookup = {
        int(index): position for position, index in enumerate(union_indices)
    }
    union_inputs = full_inputs[union_indices]
    kernels = recursive_nngp_kernel(union_inputs)
    kernel = kernels["output_kernel"]
    np.savez_compressed(
        output_dir / "recursive_nngp_kernel.npz",
        union_indices=union_indices,
        **kernels,
    )
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    predictions: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    particle_payload: dict[str, np.ndarray] = {
        "probe_indices": probe_indices,
    }
    for condition in conditions:
        train_positions = np.asarray([
            union_lookup[index] for index in condition.train_indices
        ], dtype=np.int64)
        probe_positions = np.asarray([
            union_lookup[int(index)] for index in probe_indices
        ], dtype=np.int64)
        train_labels = targets[np.asarray(condition.train_indices)]
        train_kernel = kernel[np.ix_(train_positions, train_positions)]
        probe_train_kernel = kernel[np.ix_(probe_positions, train_positions)]
        probe_kernel = kernel[np.ix_(probe_positions, probe_positions)]
        tilt, dominant_logits, log_normalizer = solve_gaussian_tilt(
            train_kernel, train_labels, Config.MATCHED_LOSS
        )
        smc_replicas: list[dict[str, Any]] = []
        smc_probability_replicas: list[np.ndarray] = []
        smc_raw_symmetry_residuals: list[float] = []
        particle_payload[f"{condition.name}_train_indices"] = np.asarray(
            condition.train_indices, dtype=np.int64
        )
        for replica in range(Config.SMC_REPLICAS):
            smc = bridge_smc_replica(
                condition.name,
                replica,
                train_kernel,
                train_labels,
                dominant_logits,
                tilt,
                Config.MATCHED_LOSS,
                Config.PREDICTION_SEED
                + 1_000_003*condition.condition_index
                + 10_007*replica,
                device,
            )
            raw_probability, _, conditional_variance = (
                conditional_probe_probabilities(
                    train_kernel,
                    probe_train_kernel,
                    probe_kernel,
                    smc["train_logits"],
                )
            )
            probability, symmetry_residual = and_swap_symmetrize(
                raw_probability
            )
            smc_probability_replicas.append(probability)
            smc_raw_symmetry_residuals.append(symmetry_residual)
            particle_payload[
                f"{condition.name}_replica_{replica}_train_logits"
            ] = smc["train_logits"]
            stage_rows.extend(smc.pop("diagnostics"))
            smc_replicas.append(smc)

        replica_probability = np.stack(smc_probability_replicas)
        probability_one = replica_probability.mean(axis=0)
        replica_logs = np.asarray([
            replica["log_volume"] for replica in smc_replicas
        ])
        log_volume = float(
            logsumexp(replica_logs)-math.log(len(replica_logs))
        )

        saddle_probability, _, _ = conditional_probe_probabilities(
            train_kernel,
            probe_train_kernel,
            probe_kernel,
            dominant_logits[None],
        )
        saddle_probability, saddle_symmetry_residual = and_swap_symmetrize(
            saddle_probability
        )

        legacy_replicas: list[dict[str, Any]] = []
        for replica in range(Config.LEGACY_IMPORTANCE_REPLICAS):
            legacy_replicas.append(gaussian_tilt_replica(
                kernel,
                train_positions,
                probe_positions,
                train_labels,
                tilt,
                log_normalizer,
                Config.MATCHED_LOSS,
                Config.LEGACY_IMPORTANCE_SAMPLES_PER_REPLICA,
                Config.PREDICTION_BATCH_SIZE,
                Config.PREDICTION_SEED
                + 2_000_003*condition.condition_index
                + 20_011*replica,
                device,
            ))
        legacy_logs = np.asarray([
            replica["log_volume"] for replica in legacy_replicas
        ])
        legacy_probability_replicas = np.stack([
            replica["probe_probability_one"] for replica in legacy_replicas
        ])
        legacy_mass_weight = np.exp(
            legacy_logs-logsumexp(legacy_logs)
        )
        legacy_probability_raw = np.sum(
            legacy_mass_weight[:, None] * legacy_probability_replicas,
            axis=0,
        )
        legacy_probability, legacy_symmetry_residual = and_swap_symmetrize(
            legacy_probability_raw
        )

        metrics = weighted_probe_metrics(
            probability_one, targets[probe_indices]
        )
        saddle_metrics = weighted_probe_metrics(
            saddle_probability, targets[probe_indices]
        )
        legacy_metrics = weighted_probe_metrics(
            legacy_probability, targets[probe_indices]
        )
        prediction = {
            "condition": asdict(condition),
            "matched_loss": Config.MATCHED_LOSS,
            "predicted_log_static_volume": log_volume,
            "smc_log_volume_replicas": replica_logs,
            "dominant_train_logits": dominant_logits,
            "probe_probability_one": probability_one,
            "probe_probability_replica_min": replica_probability.min(axis=0),
            "probe_probability_replica_max": replica_probability.max(axis=0),
            "probe_probability_replica_std": replica_probability.std(axis=0),
            "smc_stage_count_min": min(
                replica["stage_count"] for replica in smc_replicas
            ),
            "smc_stage_count_max": max(
                replica["stage_count"] for replica in smc_replicas
            ),
            "smc_minimum_incremental_ess_fraction": min(
                replica["minimum_incremental_ess_fraction"]
                for replica in smc_replicas
            ),
            "smc_initial_event_rate_min": min(
                replica["initial_event_rate"] for replica in smc_replicas
            ),
            "smc_initial_event_rate_max": max(
                replica["initial_event_rate"] for replica in smc_replicas
            ),
            "smc_raw_swap_symmetry_residual_max": max(
                smc_raw_symmetry_residuals
            ),
            "saddle_probe_probability_one": saddle_probability,
            "saddle_swap_symmetry_residual": saddle_symmetry_residual,
            "saddle_metrics": saddle_metrics,
            "legacy_importance_probe_probability_one": legacy_probability,
            "legacy_importance_log_volume": float(
                logsumexp(legacy_logs)-math.log(len(legacy_logs))
            ),
            "legacy_importance_ess_min": min(
                replica["effective_sample_size"]
                for replica in legacy_replicas
            ),
            "legacy_importance_ess_median": float(np.median([
                replica["effective_sample_size"]
                for replica in legacy_replicas
            ])),
            "legacy_importance_event_rate_min": min(
                replica["event_rate"] for replica in legacy_replicas
            ),
            "legacy_importance_event_rate_max": max(
                replica["event_rate"] for replica in legacy_replicas
            ),
            "legacy_importance_swap_symmetry_residual": (
                legacy_symmetry_residual
            ),
            "legacy_importance_metrics": legacy_metrics,
            "smc_vs_saddle_probability_mae": float(np.mean(np.abs(
                probability_one-saddle_probability
            ))),
            "smc_vs_legacy_probability_mae": float(np.mean(np.abs(
                probability_one-legacy_probability
            ))),
            "conditional_probe_variance": conditional_variance,
            **metrics,
        }
        predictions.append(prediction)
        for probe_offset, input_index in enumerate(probe_indices):
            point_rows.append({
                "condition": condition.name,
                "input_index": int(input_index),
                "input_bits": "".join(map(
                    str, full_inputs[input_index].astype(np.uint8).tolist()
                )),
                "target": int(targets[input_index]),
                "predicted_probability_one": float(
                    probability_one[probe_offset]
                ),
                "replica_min": float(replica_probability[:, probe_offset].min()),
                "replica_max": float(replica_probability[:, probe_offset].max()),
                "replica_std": float(replica_probability[:, probe_offset].std()),
                "saddle_probability_one": float(
                    saddle_probability[probe_offset]
                ),
                "legacy_importance_probability_one": float(
                    legacy_probability[probe_offset]
                ),
            })
        print(
            f"PREDICT {condition.name} | "
            f"accuracy={metrics['probe_target_accuracy']:.4f} | "
            f"agreement={metrics['probe_pairwise_agreement']:.4f} | "
            f"modal_acc={metrics['probe_modal_accuracy']:.4f} | "
            f"replica_max_sd={replica_probability.std(axis=0).max():.3g} | "
            f"SMC/saddle MAE="
            f"{prediction['smc_vs_saddle_probability_mae']:.4g} | "
            f"legacy ESS={prediction['legacy_importance_ess_min']:.1f}/"
            f"{prediction['legacy_importance_ess_median']:.1f}",
            flush=True,
        )
    np.savez_compressed(
        output_dir / "bridge_smc_final_particles.npz",
        **particle_payload,
    )
    write_csv(output_dir / "bridge_smc_stage_diagnostics.csv", stage_rows)
    write_csv(output_dir / "probe_predictions.csv", point_rows)
    prediction_body = {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "created_before_validation": True,
        "predictor_kind": "recursive_nngp_adaptive_bridge_smc",
        "predictions": predictions,
    }
    prediction_hash = payload_sha256(prediction_body)
    manifest = {
        **prediction_body,
        "prediction_sha256": prediction_hash,
    }
    manifest_path = output_dir / "prediction_manifest.json"
    write_json(manifest_path, manifest)
    (output_dir / "PREDICTION_LOCK.sha256").write_text(
        prediction_hash+"  prediction_manifest.json\n",
        encoding="ascii",
    )
    write_json(output_dir / "summary.json", {
        "status": "prediction_frozen",
        "protocol_sha256": protocol_hash,
        "prediction_sha256": prediction_hash,
        "manifest": str(manifest_path),
        "condition_count": len(conditions),
        "probe_count": len(probe_indices),
        "primary_predictor": "adaptive_bridge_smc",
        "comparison_predictors": [
            "saddle_point_conditional_gaussian",
            "legacy_one_step_importance",
        ],
        "validation_has_not_run": True,
    })
    return manifest_path


def parameter_count() -> int:
    return (
        Config.WIDTH*Config.INPUT_BITS+Config.WIDTH
        + Config.WIDTH*Config.WIDTH+Config.WIDTH
        + Config.WIDTH+1
    )


def forward_normalized(
    normalized: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    count = normalized.shape[0]
    cursor = 0
    first_size = Config.WIDTH*Config.INPUT_BITS
    first_weight = normalized[:, cursor:cursor+first_size].reshape(
        count, Config.WIDTH, Config.INPUT_BITS
    ) / math.sqrt(Config.INPUT_BITS)
    cursor += first_size
    first_bias = normalized[:, cursor:cursor+Config.WIDTH] / math.sqrt(
        Config.INPUT_BITS
    )
    cursor += Config.WIDTH
    middle_size = Config.WIDTH*Config.WIDTH
    middle_weight = normalized[:, cursor:cursor+middle_size].reshape(
        count, Config.WIDTH, Config.WIDTH
    ) / math.sqrt(Config.WIDTH)
    cursor += middle_size
    middle_bias = normalized[:, cursor:cursor+Config.WIDTH] / math.sqrt(
        Config.WIDTH
    )
    cursor += Config.WIDTH
    output_weight = normalized[:, cursor:cursor+Config.WIDTH].reshape(
        count, 1, Config.WIDTH
    ) / math.sqrt(Config.WIDTH)
    cursor += Config.WIDTH
    output_bias = normalized[:, cursor:cursor+1] / math.sqrt(Config.WIDTH)
    hidden = torch.tanh(
        torch.bmm(inputs, first_weight.transpose(1, 2))+first_bias[:, None]
    )
    hidden = torch.tanh(
        torch.bmm(hidden, middle_weight.transpose(1, 2))+middle_bias[:, None]
    )
    return (
        torch.bmm(hidden, output_weight.transpose(1, 2)).squeeze(-1)
        + output_bias
    )


def finite_width_parameter_blocks() -> tuple[ParameterBlock, ...]:
    cursor = 0
    first_size = Config.WIDTH*Config.INPUT_BITS+Config.WIDTH
    first = ParameterBlock("first_layer", cursor, cursor+first_size)
    cursor += first_size
    second_size = Config.WIDTH*Config.WIDTH+Config.WIDTH
    second = ParameterBlock("second_layer", cursor, cursor+second_size)
    cursor += second_size
    output_size = Config.WIDTH+1
    output = ParameterBlock("output_layer", cursor, cursor+output_size)
    cursor += output_size
    if cursor != parameter_count():
        raise AssertionError("有限宽参数分块与参数总数不一致。")
    return (
        first,
        second,
        output,
        ParameterBlock("all_parameters", 0, cursor),
    )


def finite_width_objective_losses(
    particles: torch.Tensor,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, len(particles), Config.FW_EVAL_BATCH):
        local = particles[start:start+Config.FW_EVAL_BATCH]
        expanded = train_inputs[None].expand(len(local), -1, -1)
        logits = forward_normalized(local, expanded)
        raw_bce = F.binary_cross_entropy_with_logits(
            logits,
            train_labels[None].expand_as(logits),
            reduction="none",
        ).mean(dim=1)
        if Config.FW_EXPLICIT_L2_COEFFICIENT:
            raw_bce = raw_bce + (
                float(Config.FW_EXPLICIT_L2_COEFFICIENT)
                * 0.5
                * local.square().sum(dim=1)
            )
        pieces.append(raw_bce)
    return torch.cat(pieces)


@torch.no_grad()
def finite_width_losses(
    particles: torch.Tensor,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
) -> torch.Tensor:
    return finite_width_objective_losses(
        particles, train_inputs, train_labels
    )


@torch.no_grad()
def finite_width_probe_logits(
    particles: torch.Tensor,
    probe_inputs: torch.Tensor,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, len(particles), Config.FW_EVAL_BATCH):
        local = particles[start:start+Config.FW_EVAL_BATCH]
        expanded = probe_inputs[None].expand(len(local), -1, -1)
        pieces.append(forward_normalized(local, expanded))
    return torch.cat(pieces)


def probe_function_ids(logits: torch.Tensor) -> np.ndarray:
    bits = (logits >= 0).to(torch.int64)
    powers = torch.bitwise_left_shift(
        torch.ones(32, dtype=torch.int64, device=logits.device),
        torch.arange(32, dtype=torch.int64, device=logits.device),
    )
    return (bits*powers[None]).sum(dim=1).cpu().numpy().astype(np.uint32)


def train_finite_width_anchor_candidates(
    condition: ConditionSpec,
    replica: int,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    probe_inputs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    count = Config.FW_ANCHOR_CANDIDATES_PER_REPLICA
    generator = torch.Generator(device=device)
    generator.manual_seed(
        Config.FW_ANCHOR_SEED
        + 1_000_003*condition.condition_index
        + 10_007*replica
    )
    parameters = torch.nn.Parameter(torch.randn(
        count,
        parameter_count(),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ))
    optimizer = torch.optim.AdamW(
        [parameters],
        lr=Config.LEARNING_RATE,
        betas=Config.ADAM_BETAS,
        eps=Config.ADAM_EPS,
        weight_decay=0.0,
    )
    reached = torch.zeros(count, dtype=torch.bool, device=device)
    saved = torch.empty_like(parameters)
    saved_loss = torch.full(
        (count,), float("nan"), dtype=torch.float32, device=device
    )
    started = time.perf_counter()
    for step in range(Config.FW_ANCHOR_MAX_STEPS+1):
        losses = finite_width_objective_losses(
            parameters, train_inputs, train_labels
        )
        newly = (~reached) & (losses <= Config.FW_ANCHOR_TARGET_LOSS)
        if bool(torch.any(newly).item()):
            saved[newly] = parameters.detach()[newly]
            saved_loss[newly] = losses.detach()[newly]
            reached[newly] = True
        if bool(torch.all(reached).item()):
            break
        if step == Config.FW_ANCHOR_MAX_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        (losses*(~reached).to(losses.dtype)).sum().backward()
        optimizer.step()
        if step and step % Config.FW_ANCHOR_LOG_EVERY == 0:
            print(
                f"ANCHOR {condition.name} r={replica+1}/"
                f"{Config.FW_REPLICAS} step={step:,} | "
                f"reached={reached.float().mean().item():.1%} | "
                f"active_loss_median="
                f"{losses[~reached].median().item() if bool(torch.any(~reached).item()) else 0:.5g} | "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    reached_indices = torch.nonzero(reached, as_tuple=False).flatten()
    if len(reached_indices) < Config.FW_ANCHORS_PER_REPLICA:
        raise RuntimeError(
            f"{condition.name} replica={replica}只有{len(reached_indices)}个"
            f"anchor到达loss<={Config.FW_ANCHOR_TARGET_LOSS}。"
        )
    candidates = saved[reached_indices]
    candidate_losses = saved_loss[reached_indices]
    ids = probe_function_ids(
        finite_width_probe_logits(candidates, probe_inputs)
    )
    rng = np.random.default_rng(
        Config.FW_ANCHOR_SEED
        + 2_000_003*condition.condition_index
        + 20_011*replica
    )
    groups: dict[int, list[int]] = {}
    for index, function_id in enumerate(ids):
        groups.setdefault(int(function_id), []).append(index)
    keys = np.asarray(sorted(groups), dtype=np.uint64)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[int(key)])
    selected: list[int] = []
    depth = 0
    while len(selected) < Config.FW_ANCHORS_PER_REPLICA:
        added = False
        for key in keys:
            local = groups[int(key)]
            if depth < len(local):
                selected.append(local[depth])
                added = True
                if len(selected) == Config.FW_ANCHORS_PER_REPLICA:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != Config.FW_ANCHORS_PER_REPLICA:
        raise RuntimeError("anchor多样性选择没有得到要求的数量。")
    selected_tensor = torch.as_tensor(
        selected, dtype=torch.long, device=device
    )
    anchors = candidates[selected_tensor].detach().clone()
    selected_ids = ids[np.asarray(selected)]
    selected_losses = candidate_losses[selected_tensor].detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    unique_ids, counts = np.unique(ids, return_counts=True)
    count_by_id = dict(zip(unique_ids.tolist(), counts.tolist()))
    for local_index, candidate_index in enumerate(selected):
        function_id = int(selected_ids[local_index])
        rows.append({
            "condition": condition.name,
            "replica": replica,
            "anchor_index": local_index,
            "candidate_index": int(candidate_index),
            "probe_function_id": function_id,
            "probe_function_hex": f"{function_id:08x}",
            "candidate_function_frequency": int(count_by_id[function_id]),
            "anchor_loss": float(selected_losses[local_index]),
            "anchor_parameter_norm": float(torch.linalg.vector_norm(
                anchors[local_index]
            ).item()),
            "reached_candidate_count": int(len(candidates)),
            "unique_candidate_function_count": int(len(unique_ids)),
        })
    return anchors, rows


@torch.no_grad()
def finite_width_mixture_event_rate(
    anchors: torch.Tensor,
    sigma: float,
    components: torch.Tensor,
    noise: torch.Tensor,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
) -> float:
    particles = anchors[components]+float(sigma)*noise
    losses = finite_width_losses(particles, train_inputs, train_labels)
    return float((losses <= Config.MATCHED_LOSS).float().mean().item())


def tune_finite_width_sigma(
    anchors: torch.Tensor,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[float, list[dict[str, float]]]:
    count = Config.FW_SIGMA_PILOT_SAMPLES
    components = torch.randint(
        len(anchors),
        (count,),
        device=anchors.device,
        generator=generator,
    )
    noise = torch.randn(
        count,
        parameter_count(),
        dtype=anchors.dtype,
        device=anchors.device,
        generator=generator,
    )
    rows: list[dict[str, float]] = []

    def evaluate(value: float) -> float:
        rate = finite_width_mixture_event_rate(
            anchors,
            value,
            components,
            noise,
            train_inputs,
            train_labels,
        )
        rows.append({"sigma": float(value), "event_rate": rate})
        return rate

    lower = Config.FW_SIGMA_MIN
    upper = Config.FW_SIGMA_MAX
    lower_rate = evaluate(lower)
    upper_rate = evaluate(upper)
    target = Config.FW_SIGMA_TARGET_EVENT_RATE
    if lower_rate < target:
        return lower, rows
    if upper_rate > target:
        return upper, rows
    for _ in range(Config.FW_SIGMA_SEARCH_STEPS):
        midpoint = math.sqrt(lower*upper)
        rate = evaluate(midpoint)
        if rate >= target:
            lower = midpoint
        else:
            upper = midpoint
    return math.sqrt(lower*upper), rows


@torch.no_grad()
def draw_finite_width_initial_event(
    anchors: torch.Tensor,
    sigma: float,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, int]:
    required = Config.FW_PARTICLES
    pool_size = max(
        required*Config.FW_INITIAL_POOL_FACTOR,
        Config.FW_EVAL_BATCH,
    )
    particle_chunks: list[torch.Tensor] = []
    component_chunks: list[torch.Tensor] = []
    loss_chunks: list[torch.Tensor] = []
    accepted_total = 0
    generated_total = 0
    while accepted_total < required:
        components = torch.randint(
            len(anchors),
            (pool_size,),
            device=anchors.device,
            generator=generator,
        )
        noise = torch.randn(
            pool_size,
            parameter_count(),
            dtype=anchors.dtype,
            device=anchors.device,
            generator=generator,
        )
        particles = anchors[components]+float(sigma)*noise
        losses = finite_width_losses(particles, train_inputs, train_labels)
        event = losses <= Config.MATCHED_LOSS
        if bool(torch.any(event).item()):
            particle_chunks.append(particles[event])
            component_chunks.append(components[event])
            loss_chunks.append(losses[event])
        accepted_total += int(event.sum().item())
        generated_total += pool_size
        if generated_total >= 32*pool_size and accepted_total == 0:
            raise RuntimeError("有限宽anchor mixture没有命中目标loss事件。")
    particles = torch.cat(particle_chunks)
    components = torch.cat(component_chunks)
    losses = torch.cat(loss_chunks)
    selection = torch.randperm(
        len(particles), device=particles.device, generator=generator
    )[:required]
    return (
        particles[selection].clone(),
        components[selection].clone(),
        losses[selection].clone(),
        accepted_total/generated_total,
        accepted_total,
        generated_total,
    )


def finite_width_bridge_scale(beta: float, sigma: float) -> float:
    return math.exp((1.0-beta)*math.log(sigma))


def finite_width_log_q_from_statistics(
    theta_squared: torch.Tensor,
    theta_anchor_dot: torch.Tensor,
    anchor_squared: torch.Tensor,
    beta: float,
    sigma: float,
) -> torch.Tensor:
    coefficient = 1.0-beta
    scale = finite_width_bridge_scale(beta, sigma)
    residual_squared = (
        theta_squared
        - 2.0*coefficient*theta_anchor_dot
        + coefficient**2*anchor_squared
    )
    return (
        -parameter_count()*math.log(scale)
        - 0.5*residual_squared/(scale*scale)
    )


def finite_width_log_q(
    particles: torch.Tensor,
    components: torch.Tensor,
    anchors: torch.Tensor,
    beta: float,
    sigma: float,
) -> torch.Tensor:
    particles64 = particles.to(torch.float64)
    local_anchors = anchors[components].to(torch.float64)
    return finite_width_log_q_from_statistics(
        torch.sum(particles64**2, dim=1),
        torch.sum(particles64*local_anchors, dim=1),
        torch.sum(local_anchors**2, dim=1),
        beta,
        sigma,
    )


@torch.no_grad()
def finite_width_mutate_block(
    particles: torch.Tensor,
    components: torch.Tensor,
    losses: torch.Tensor,
    anchors: torch.Tensor,
    beta: float,
    sigma: float,
    block: ParameterBlock,
    rho: float,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    proposal = particles.clone()
    coefficient = 1.0-beta
    bridge_sd = finite_width_bridge_scale(beta, sigma)
    local_mean = coefficient*anchors[
        components, block.start:block.stop
    ]
    current = particles[:, block.start:block.stop]
    noise = torch.randn(
        current.shape,
        dtype=current.dtype,
        device=current.device,
        generator=generator,
    )
    bounded_rho = min(max(float(rho), 0.0), 0.999999)
    proposed_block = (
        local_mean
        + math.sqrt(1.0-bounded_rho**2)*(current-local_mean)
        + bounded_rho*bridge_sd*noise
    )
    proposal[:, block.start:block.stop] = proposed_block
    proposal_losses = finite_width_losses(
        proposal, train_inputs, train_labels
    )
    accept = proposal_losses <= Config.MATCHED_LOSS+1e-7
    movement = torch.sqrt(torch.mean(
        (proposed_block-current)**2, dim=1
    ))
    particles[accept] = proposal[accept]
    losses[accept] = proposal_losses[accept]
    return (
        particles,
        losses,
        float(accept.float().mean().item()),
        float(movement[accept].mean().item())
        if bool(torch.any(accept).item()) else 0.0,
    )


@torch.no_grad()
def finite_width_switch_components(
    particles: torch.Tensor,
    components: torch.Tensor,
    losses: torch.Tensor,
    anchors: torch.Tensor,
    beta: float,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    proposed_components = torch.randint(
        len(anchors),
        components.shape,
        device=components.device,
        generator=generator,
    )
    same = proposed_components == components
    proposed_components[same] = (
        proposed_components[same]+1
    ) % len(anchors)
    coefficient = 1.0-beta
    proposal = particles + coefficient*(
        anchors[proposed_components]-anchors[components]
    )
    proposal_losses = finite_width_losses(
        proposal, train_inputs, train_labels
    )
    accept = proposal_losses <= Config.MATCHED_LOSS+1e-7
    particles[accept] = proposal[accept]
    components[accept] = proposed_components[accept]
    losses[accept] = proposal_losses[accept]
    return (
        particles,
        components,
        losses,
        float(accept.float().mean().item()),
    )


def finite_width_rejuvenate(
    particles: torch.Tensor,
    components: torch.Tensor,
    losses: torch.Tensor,
    anchors: torch.Tensor,
    beta: float,
    sigma: float,
    blocks: Sequence[ParameterBlock],
    scales: list[float],
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
    adapt_sweeps: int,
    mutation_sweeps: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[float],
    dict[str, float],
]:
    local_scales = list(scales)
    for _ in range(adapt_sweeps):
        for block_index, block in enumerate(blocks):
            particles, losses, acceptance, _ = finite_width_mutate_block(
                particles,
                components,
                losses,
                anchors,
                beta,
                sigma,
                block,
                local_scales[block_index],
                train_inputs,
                train_labels,
                generator,
            )
            local_scales[block_index] *= math.exp(
                Config.FW_ADAPT_RATE
                * (acceptance-Config.FW_TARGET_ACCEPTANCE)
            )
            local_scales[block_index] = min(max(
                local_scales[block_index],
                Config.FW_MIN_PCN_SCALE,
            ), Config.FW_MAX_PCN_SCALE)
    acceptance_sum = np.zeros(len(blocks), dtype=np.float64)
    movement_sum = np.zeros(len(blocks), dtype=np.float64)
    switch_sum = 0.0
    for _ in range(mutation_sweeps):
        for block_index, block in enumerate(blocks):
            particles, losses, acceptance, movement = (
                finite_width_mutate_block(
                    particles,
                    components,
                    losses,
                    anchors,
                    beta,
                    sigma,
                    block,
                    local_scales[block_index],
                    train_inputs,
                    train_labels,
                    generator,
                )
            )
            acceptance_sum[block_index] += acceptance
            movement_sum[block_index] += movement
        for _ in range(Config.FW_COMPONENT_SWITCH_SWEEPS):
            particles, components, losses, switch_acceptance = (
                finite_width_switch_components(
                    particles,
                    components,
                    losses,
                    anchors,
                    beta,
                    train_inputs,
                    train_labels,
                    generator,
                )
            )
            switch_sum += switch_acceptance
    denominator = max(mutation_sweeps, 1)
    diagnostics: dict[str, float] = {
        "component_switch_acceptance": (
            switch_sum
            / max(mutation_sweeps*Config.FW_COMPONENT_SWITCH_SWEEPS, 1)
        ),
        "mean_loss": float(losses.mean().item()),
        "max_loss": float(losses.max().item()),
    }
    for index, block in enumerate(blocks):
        diagnostics[f"acceptance_{block.name}"] = float(
            acceptance_sum[index]/denominator
        )
        diagnostics[f"movement_{block.name}"] = float(
            movement_sum[index]/denominator
        )
        diagnostics[f"scale_{block.name}"] = float(local_scales[index])
    return (
        particles,
        components,
        losses,
        local_scales,
        diagnostics,
    )


def finite_width_bridge_replica(
    condition: ConditionSpec,
    replica: int,
    anchors: torch.Tensor,
    sigma: float,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    probe_inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        Config.FW_SMC_SEED
        + 1_000_003*condition.condition_index
        + 10_007*replica
    )
    (
        particles,
        components,
        losses,
        initial_event_rate,
        initial_accepted,
        initial_generated,
    ) = draw_finite_width_initial_event(
        anchors,
        sigma,
        train_inputs,
        train_labels,
        generator,
    )
    beta = 0.0
    log_volume = math.log(initial_event_rate)
    scales = list(Config.FW_INITIAL_PCN_SCALES)
    blocks = finite_width_parameter_blocks()
    lineages = torch.arange(
        Config.FW_PARTICLES, dtype=torch.long, device=device
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    target_ess = Config.FW_TARGET_ESS_FRACTION*Config.FW_PARTICLES
    for stage in range(1, Config.FW_MAX_BRIDGE_STAGES+1):
        local_anchors64 = anchors[components].to(torch.float64)
        particles64 = particles.to(torch.float64)
        theta_squared = torch.sum(particles64**2, dim=1)
        theta_anchor_dot = torch.sum(
            particles64*local_anchors64, dim=1
        )
        anchor_squared = torch.sum(local_anchors64**2, dim=1)
        current_log_q = finite_width_log_q_from_statistics(
            theta_squared,
            theta_anchor_dot,
            anchor_squared,
            beta,
            sigma,
        )

        def candidate_log_weight(candidate: float) -> torch.Tensor:
            return finite_width_log_q_from_statistics(
                theta_squared,
                theta_anchor_dot,
                anchor_squared,
                candidate,
                sigma,
            )-current_log_q

        full_weight = candidate_log_weight(1.0)
        if log_weight_ess(full_weight) >= target_ess:
            next_beta = 1.0
            log_weight = full_weight
        else:
            lower_beta = beta
            upper_beta = 1.0
            for _ in range(50):
                midpoint = 0.5*(lower_beta+upper_beta)
                if log_weight_ess(
                    candidate_log_weight(midpoint)
                ) >= target_ess:
                    lower_beta = midpoint
                else:
                    upper_beta = midpoint
            next_beta = lower_beta
            log_weight = candidate_log_weight(next_beta)
        if next_beta-beta < Config.FW_BETA_TOLERANCE:
            raise RuntimeError(
                f"{condition.name} replica={replica} finite-width bridge"
                f"停滞于beta={beta:.9f}。"
            )
        incremental_ess = log_weight_ess(log_weight)
        log_volume += float(
            torch.logsumexp(log_weight, dim=0).item()
            - math.log(Config.FW_PARTICLES)
        )
        normalized = torch.softmax(log_weight, dim=0)
        selected = systematic_resample(normalized, generator)
        particles = particles[selected].clone()
        components = components[selected].clone()
        losses = losses[selected].clone()
        lineages = lineages[selected]
        beta = next_beta
        (
            particles,
            components,
            losses,
            scales,
            mutation,
        ) = finite_width_rejuvenate(
            particles,
            components,
            losses,
            anchors,
            beta,
            sigma,
            blocks,
            scales,
            train_inputs,
            train_labels,
            generator,
            Config.FW_ADAPT_SWEEPS,
            Config.FW_MUTATION_SWEEPS,
        )
        elapsed = time.perf_counter()-started
        row = {
            "condition": condition.name,
            "replica": replica,
            "stage": stage,
            "beta": beta,
            "bridge_sd": finite_width_bridge_scale(beta, sigma),
            "incremental_ess": incremental_ess,
            "incremental_ess_fraction": incremental_ess/Config.FW_PARTICLES,
            "log_volume": log_volume,
            "unique_lineage_fraction": float(
                torch.unique(lineages).numel()/Config.FW_PARTICLES
            ),
            "unique_component_count": int(torch.unique(components).numel()),
            "elapsed_seconds": elapsed,
            **mutation,
        }
        rows.append(row)
        if (
            stage == 1
            or stage % Config.FW_LOG_EVERY_STAGES == 0
            or beta >= 1.0
        ):
            eta = elapsed*(1.0-beta)/max(beta, 1e-12)
            print(
                f"FW-SMC {condition.name} r={replica+1}/"
                f"{Config.FW_REPLICAS} stage={stage} beta={beta:.5f} | "
                f"sd={row['bridge_sd']:.4g} | "
                f"ESS={row['incremental_ess_fraction']:.3f} | "
                f"loss={mutation['mean_loss']:.5g}/"
                f"{mutation['max_loss']:.5g} | "
                f"accept_all={mutation['acceptance_all_parameters']:.1%} | "
                f"switch={mutation['component_switch_acceptance']:.1%} | "
                f"logV={log_volume:.2f} | elapsed={elapsed:.1f}s "
                f"ETA~{eta:.1f}s",
                flush=True,
            )
        if beta >= 1.0:
            break
    else:
        raise RuntimeError(
            f"{condition.name} replica={replica}超过最大bridge stages。"
        )
    (
        particles,
        components,
        losses,
        scales,
        final_mutation,
    ) = finite_width_rejuvenate(
        particles,
        components,
        losses,
        anchors,
        1.0,
        sigma,
        blocks,
        scales,
        train_inputs,
        train_labels,
        generator,
        Config.FW_ADAPT_SWEEPS,
        Config.FW_FINAL_MUTATION_SWEEPS,
    )
    probe_logits = finite_width_probe_logits(particles, probe_inputs)
    probe_bits = (probe_logits >= 0).to(torch.uint8)
    probability_one = probe_bits.to(torch.float64).mean(dim=0)
    subset = min(Config.FW_SAVE_PARAMETER_SUBSET, len(particles))
    return {
        "log_volume": log_volume,
        "initial_event_rate": initial_event_rate,
        "initial_accepted": initial_accepted,
        "initial_generated": initial_generated,
        "sigma": sigma,
        "stage_count": len(rows),
        "minimum_incremental_ess_fraction": min(
            row["incremental_ess_fraction"] for row in rows
        ),
        "final_unique_lineage_fraction": rows[-1][
            "unique_lineage_fraction"
        ],
        "final_unique_component_count": rows[-1][
            "unique_component_count"
        ],
        "final_mutation": final_mutation,
        "final_loss_mean": float(losses.mean().item()),
        "final_loss_max": float(losses.max().item()),
        "probe_probability_one": probability_one.cpu().numpy(),
        "probe_logits": probe_logits.cpu().numpy().astype(np.float32),
        "parameter_subset": particles[:subset].cpu().numpy().astype(np.float32),
        "component_subset": components[:subset].cpu().numpy().astype(np.int16),
        "stage_rows": rows,
    }


def finite_width_protocol_payload() -> dict[str, Any]:
    conditions, probe_suffixes, probe_indices = build_protocol_objects()
    return {
        "protocol_version": Config.FW_PROTOCOL_VERSION,
        "measured_object": (
            "iid_standard_Gaussian_mass_of_the_exact_finite_width_"
            "parameter_network_conditioned_on_fixed_D_BCE"
        ),
        "network": {
            "architecture": "8->16->16->1",
            "activation": "tanh",
            "parameter_count": parameter_count(),
            "parameter_coordinates": "iid_standard_Gaussian",
            "fan_in_scaling": True,
            "nngp_used": False,
            "every_loss_and_probe_uses_real_network_forward": True,
        },
        "task": {
            "target": "x0 AND x1",
            "conditions": [asdict(condition) for condition in conditions],
            "probe_suffixes": probe_suffixes,
            "probe_indices": probe_indices,
            "probe_targets": target_outputs()[probe_indices],
            "matched_loss": Config.MATCHED_LOSS,
        },
        "bridge_smc": {
            "replicas": Config.FW_REPLICAS,
            "particles": Config.FW_PARTICLES,
            "target_ess_fraction": Config.FW_TARGET_ESS_FRACTION,
            "anchor_candidates_per_replica": (
                Config.FW_ANCHOR_CANDIDATES_PER_REPLICA
            ),
            "anchors_per_replica": Config.FW_ANCHORS_PER_REPLICA,
            "anchor_target_loss": Config.FW_ANCHOR_TARGET_LOSS,
            "anchor_role": (
                "full_support_importance_proposal_only; exact Gaussian "
                "density ratios remove proposal dependence at beta=1"
            ),
            "proposal_path": (
                "N((1-beta)*anchor, sigma^(2*(1-beta))*I) to N(0,I)"
            ),
            "adapt_sweeps": Config.FW_ADAPT_SWEEPS,
            "mutation_sweeps": Config.FW_MUTATION_SWEEPS,
            "final_mutation_sweeps": Config.FW_FINAL_MUTATION_SWEEPS,
            "component_switch_sweeps": Config.FW_COMPONENT_SWITCH_SWEEPS,
            "initial_pcn_scales": Config.FW_INITIAL_PCN_SCALES,
            "separate_anchor_mixture_per_replica": True,
            "anchor_seed": Config.FW_ANCHOR_SEED,
            "smc_seed": Config.FW_SMC_SEED,
        },
    }


def probe_distribution_summary(
    probe_logits_by_replica: np.ndarray,
    target_bits: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bits = probe_logits_by_replica >= 0
    flat = bits.reshape(-1, bits.shape[-1])
    powers = np.left_shift(
        np.ones(32, dtype=np.uint64),
        np.arange(32, dtype=np.uint64),
    )
    ids = np.sum(flat.astype(np.uint64)*powers[None], axis=1)
    unique, counts = np.unique(ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    masses = counts.astype(np.float64)/len(ids)
    entropy = float(-np.sum(masses*np.log2(masses)))
    target_id = int(np.sum(target_bits.astype(np.uint64)*powers))
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order[:20], start=1):
        function_id = int(unique[index])
        function_bits_text = "".join(
            str((function_id >> bit) & 1) for bit in range(32)
        )
        rows.append({
            "rank": rank,
            "probe_function_id": function_id,
            "probe_function_hex": f"{function_id:08x}",
            "probe_function_bits": function_bits_text,
            "count": int(counts[index]),
            "probability": float(counts[index]/len(ids)),
            "hamming_error_to_target": int(sum(
                first != second
                for first, second in zip(
                    function_bits_text,
                    "".join(map(str, target_bits.tolist())),
                )
            )),
        })
    target_count = int(counts[unique == target_id].sum())
    return {
        "sample_count": int(len(ids)),
        "unique_probe_function_count": int(len(unique)),
        "probe_function_entropy_bits": entropy,
        "effective_probe_function_count": float(2.0**entropy),
        "probe_function_collision_probability": float(np.sum(masses**2)),
        "exact_target_probe_function_fraction": target_count/len(ids),
    }, rows


def run_finite_width_condition(
    output_dir: Path,
    condition: ConditionSpec,
    device: torch.device,
    full_inputs: torch.Tensor,
    full_targets: torch.Tensor,
    probe_indices: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if Config.FW_RESUME and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(f"FW-SMC {condition.name}已完成，跳过。", flush=True)
            return existing
    train_index = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=device
    )
    probe_index = torch.as_tensor(
        probe_indices, dtype=torch.long, device=device
    )
    train_inputs = full_inputs[train_index]
    train_labels = full_targets[train_index]
    probe_inputs = full_inputs[probe_index]
    target_bits = target_outputs()[probe_indices]
    replica_results: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    anchors_by_replica: list[np.ndarray] = []
    started = time.perf_counter()
    for replica in range(Config.FW_REPLICAS):
        anchors, local_anchor_rows = train_finite_width_anchor_candidates(
            condition,
            replica,
            train_inputs,
            train_labels,
            probe_inputs,
            device,
        )
        anchor_rows.extend(local_anchor_rows)
        anchors_by_replica.append(
            anchors.detach().cpu().numpy().astype(np.float32)
        )
        sigma_generator = torch.Generator(device=device)
        sigma_generator.manual_seed(
            Config.FW_SMC_SEED
            + 3_000_017*condition.condition_index
            + 30_013*replica
        )
        sigma, local_sigma_rows = tune_finite_width_sigma(
            anchors,
            train_inputs,
            train_labels,
            sigma_generator,
        )
        if (
            Config.MODE in {
                "explicit_l2_and_n32", "explicit_l2_and_boundary"
            }
            and sigma <= Config.FW_SIGMA_MIN * 1.01
        ):
            raise RuntimeError(
                f"{condition.name} lambda="
                f"{Config.FW_EXPLICIT_L2_COEFFICIENT:g} proposal sigma="
                f"{sigma:.6g}撞到下界；静态event过薄，禁止继续bridge。"
            )
        for row in local_sigma_rows:
            sigma_rows.append({
                "condition": condition.name,
                "replica": replica,
                **row,
            })
        print(
            f"FW-PROPOSAL {condition.name} r={replica+1}/"
            f"{Config.FW_REPLICAS} anchors={len(anchors)} sigma={sigma:.6g}",
            flush=True,
        )
        replica_results.append(finite_width_bridge_replica(
            condition,
            replica,
            anchors,
            sigma,
            train_inputs,
            train_labels,
            probe_inputs,
            device,
        ))
    stage_rows = [
        row
        for result in replica_results
        for row in result.pop("stage_rows")
    ]
    replica_probability = np.stack([
        result["probe_probability_one"] for result in replica_results
    ])
    probability_one = replica_probability.mean(axis=0)
    metrics = weighted_probe_metrics(probability_one, target_bits)
    probe_logits = np.stack([
        result["probe_logits"] for result in replica_results
    ])
    distribution, top_rows = probe_distribution_summary(
        probe_logits, target_bits
    )
    for row in top_rows:
        row["condition"] = condition.name
    log_volumes = np.asarray([
        result["log_volume"] for result in replica_results
    ], dtype=np.float64)
    summary = {
        "status": "completed",
        "condition": asdict(condition),
        "matched_loss": Config.MATCHED_LOSS,
        "network": "exact finite-width 8->16->16->1 tanh",
        "parameter_count": parameter_count(),
        "nngp_used": False,
        "predicted_log_static_volume": float(
            logsumexp(log_volumes)-math.log(len(log_volumes))
        ),
        "replica_log_static_volumes": log_volumes,
        "replica_log_volume_std": float(np.std(log_volumes)),
        "probe_probability_one": probability_one,
        "probe_probability_replica_min": replica_probability.min(axis=0),
        "probe_probability_replica_max": replica_probability.max(axis=0),
        "probe_probability_replica_std": replica_probability.std(axis=0),
        "maximum_probe_replica_std": float(
            replica_probability.std(axis=0).max()
        ),
        "minimum_incremental_ess_fraction": min(
            result["minimum_incremental_ess_fraction"]
            for result in replica_results
        ),
        "stage_count_min": min(
            result["stage_count"] for result in replica_results
        ),
        "stage_count_max": max(
            result["stage_count"] for result in replica_results
        ),
        "sigma_by_replica": [
            result["sigma"] for result in replica_results
        ],
        "final_loss_max": max(
            result["final_loss_max"] for result in replica_results
        ),
        "elapsed_seconds": time.perf_counter()-started,
        **metrics,
        **distribution,
    }
    quality_checks = {
        "replica_log_volume_std_le_1nat": bool(
            summary["replica_log_volume_std"] <= 1.0
        ),
        "maximum_probe_replica_std_le_0p03": bool(
            summary["maximum_probe_replica_std"] <= 0.03
        ),
        "all_particles_respect_loss_threshold": bool(
            summary["final_loss_max"] <= Config.MATCHED_LOSS+1e-6
        ),
    }
    summary["quality_checks"] = quality_checks
    summary["quality_pass"] = bool(all(quality_checks.values()))
    np.savez_compressed(
        output_dir / "finite_width_condition_samples.npz",
        probe_indices=probe_indices,
        target_bits=target_bits,
        probe_logits=probe_logits,
        probe_probability_by_replica=replica_probability,
        parameter_subsets=np.stack([
            result["parameter_subset"] for result in replica_results
        ]),
        component_subsets=np.stack([
            result["component_subset"] for result in replica_results
        ]),
        anchors=np.stack(anchors_by_replica),
        log_volumes=log_volumes,
        sigmas=np.asarray([
            result["sigma"] for result in replica_results
        ]),
    )
    write_csv(output_dir / "bridge_stage_diagnostics.csv", stage_rows)
    write_csv(output_dir / "anchor_diagnostics.csv", anchor_rows)
    write_csv(output_dir / "sigma_search.csv", sigma_rows)
    write_csv(output_dir / "top_probe_functions.csv", top_rows)
    write_json(summary_path, summary)
    print(
        f"FW-RESULT {condition.name} | "
        f"accuracy={metrics['probe_target_accuracy']:.4f} | "
        f"agreement={metrics['probe_pairwise_agreement']:.4f} | "
        f"modal={metrics['probe_modal_accuracy']:.4f} | "
        f"replica_sd={summary['maximum_probe_replica_std']:.4g} | "
        f"quality={'PASS' if summary['quality_pass'] else 'UNCONVERGED'} | "
        f"logV={summary['predicted_log_static_volume']:.2f} | "
        f"elapsed={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return summary


def run_finite_width_smc(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = finite_width_protocol_payload()
    protocol_hash = payload_sha256(protocol)
    write_json(output_dir / "protocol.json", {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
    })
    conditions, _, probe_indices = build_protocol_objects()
    start = max(0, Config.FW_CONDITION_START)
    stop = (
        len(conditions)
        if Config.FW_CONDITION_STOP is None
        else min(Config.FW_CONDITION_STOP, len(conditions))
    )
    selected = conditions[start:stop]
    if not selected:
        raise ValueError("finite-width condition slice为空。")
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    full_inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    full_targets = torch.as_tensor(
        target_outputs().astype(np.float32), device=device
    )
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, condition in enumerate(selected, start=1):
        print(
            f"\n=== FINITE WIDTH {ordinal}/{len(selected)} "
            f"{condition.name} ===",
            flush=True,
        )
        summaries.append(run_finite_width_condition(
            output_dir / "conditions" / condition.name,
            condition,
            device,
            full_inputs,
            full_targets,
            probe_indices,
        ))
    root_summary = {
        "status": "completed",
        "protocol_sha256": protocol_hash,
        "condition_start": start,
        "condition_stop": stop,
        "completed_conditions": [
            summary["condition"]["name"] for summary in summaries
        ],
        "all_quality_pass": bool(all(
            summary.get("quality_pass", False) for summary in summaries
        )),
        "elapsed_seconds": time.perf_counter()-started,
        "results": summaries,
        "interpretation_boundary": (
            "This is exact finite-width parameter-space Gaussian geometry "
            "up to bridge-SMC Monte Carlo and mixing error; no NNGP is used."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, root_summary)
    return summary_path


def finite_width_direct_protocol_payload() -> dict[str, Any]:
    conditions, probe_suffixes, probe_indices = build_protocol_objects()
    return {
        "protocol_version": "8bit_finite_width_fixed_d_direct_smc_v1",
        "measured_object": (
            "exact_finite_width_iid_standard_Gaussian_parameter_mass_"
            "conditioned_on_fixed_D_BCE"
        ),
        "network": {
            "architecture": "8->16->16->1",
            "activation": "tanh",
            "parameter_count": parameter_count(),
            "parameter_coordinates": "iid_standard_Gaussian",
            "fan_in_scaling": True,
            "nngp_used": False,
            "optimizer_or_anchor_used": False,
            "every_loss_and_probe_uses_real_network_forward": True,
        },
        "task": {
            "target": "x0 AND x1",
            "conditions": [asdict(condition) for condition in conditions],
            "probe_suffixes": probe_suffixes,
            "probe_indices": probe_indices,
            "probe_targets": target_outputs()[probe_indices],
            "matched_loss": Config.MATCHED_LOSS,
        },
        "direct_constrained_smc": {
            "replicas": Config.DIRECT_REPLICAS,
            "particles_per_replica": Config.DIRECT_PARTICLES_PER_REPLICA,
            "survival_quantile": Config.DIRECT_SURVIVAL_QUANTILE,
            "adapt_sweeps": Config.DIRECT_ADAPT_SWEEPS,
            "mutation_sweeps": Config.DIRECT_MUTATION_SWEEPS,
            "final_mutation_sweeps": Config.DIRECT_FINAL_MUTATION_SWEEPS,
            "proposal": "prior-preserving blockwise pCN",
            "loss_gradient_used": False,
            "seed": Config.DIRECT_SEED,
        },
    }


def direct_constraint_tolerance(threshold: float) -> float:
    """尺度相关约束容差；深loss时不能让固定1e-7吞掉事件。"""
    float32_roundoff = 16.0*abs(float(np.spacing(np.float32(threshold))))
    return max(
        1e-12,
        abs(float(threshold))*1e-5+float32_roundoff,
    )


@torch.no_grad()
def direct_mutate_block(
    particles: torch.Tensor,
    losses: torch.Tensor,
    block: ParameterBlock,
    rho: float,
    threshold: float,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    proposal = particles.clone()
    # 必须脱离proposal的view；否则下面原位写proposal后movement会恒为0。
    current = particles[..., block.start:block.stop]
    noise = torch.randn(
        current.shape,
        dtype=current.dtype,
        device=current.device,
        generator=generator,
    )
    bounded = min(max(float(rho), 0.0), 0.999999)
    proposed_block = (
        math.sqrt(1.0-bounded**2)*current + bounded*noise
    )
    proposal[..., block.start:block.stop] = proposed_block
    proposal_losses = finite_width_losses(
        proposal.reshape(-1, proposal.shape[-1]),
        train_inputs,
        train_labels,
    ).reshape(losses.shape)
    accept = proposal_losses <= (
        threshold+direct_constraint_tolerance(threshold)
    )
    movement = torch.sqrt(torch.mean(
        (proposed_block-current)**2, dim=-1
    ))
    particles[accept] = proposal[accept]
    losses[accept] = proposal_losses[accept]
    return (
        particles,
        losses,
        float(accept.float().mean().item()),
        float(movement[accept].mean().item())
        if bool(torch.any(accept).item()) else 0.0,
    )


def direct_rejuvenate(
    particles: torch.Tensor,
    losses: torch.Tensor,
    threshold: float,
    blocks: Sequence[ParameterBlock],
    scales: list[float],
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    generator: torch.Generator,
    adapt_sweeps: int,
    mutation_sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float], dict[str, float]]:
    local_scales = list(scales)
    for _ in range(adapt_sweeps):
        for index, block in enumerate(blocks):
            particles, losses, acceptance, _ = direct_mutate_block(
                particles,
                losses,
                block,
                local_scales[index],
                threshold,
                train_inputs,
                train_labels,
                generator,
            )
            local_scales[index] *= math.exp(
                Config.DIRECT_ADAPT_RATE
                * (acceptance-Config.DIRECT_TARGET_ACCEPTANCE)
            )
            local_scales[index] = min(max(
                local_scales[index], Config.DIRECT_MIN_PCN_SCALE
            ), Config.DIRECT_MAX_PCN_SCALE)
    acceptance_sum = np.zeros(len(blocks), dtype=np.float64)
    movement_sum = np.zeros(len(blocks), dtype=np.float64)
    for _ in range(mutation_sweeps):
        for index, block in enumerate(blocks):
            particles, losses, acceptance, movement = direct_mutate_block(
                particles,
                losses,
                block,
                local_scales[index],
                threshold,
                train_inputs,
                train_labels,
                generator,
            )
            acceptance_sum[index] += acceptance
            movement_sum[index] += movement
    denominator = max(mutation_sweeps, 1)
    diagnostics: dict[str, float] = {
        "mean_loss": float(losses.mean().item()),
        "max_loss": float(losses.max().item()),
    }
    for index, block in enumerate(blocks):
        diagnostics[f"acceptance_{block.name}"] = float(
            acceptance_sum[index]/denominator
        )
        diagnostics[f"movement_{block.name}"] = float(
            movement_sum[index]/denominator
        )
        diagnostics[f"scale_{block.name}"] = float(local_scales[index])
    return particles, losses, local_scales, diagnostics


def run_finite_width_direct_condition(
    output_dir: Path,
    condition: ConditionSpec,
    device: torch.device,
    full_inputs: torch.Tensor,
    full_targets: torch.Tensor,
    probe_indices: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if Config.FW_RESUME and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(f"DIRECT-SMC {condition.name}已完成，跳过。", flush=True)
            return existing
    train_index = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=device
    )
    probe_index = torch.as_tensor(
        probe_indices, dtype=torch.long, device=device
    )
    train_inputs = full_inputs[train_index]
    train_labels = full_targets[train_index]
    probe_inputs = full_inputs[probe_index]
    target_bits = target_outputs()[probe_indices]
    generator = torch.Generator(device=device)
    generator.manual_seed(
        Config.DIRECT_SEED+1_000_003*condition.condition_index
    )
    replica_count = Config.DIRECT_REPLICAS
    particle_count = Config.DIRECT_PARTICLES_PER_REPLICA
    particles = torch.randn(
        replica_count,
        particle_count,
        parameter_count(),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    losses = finite_width_losses(
        particles.reshape(-1, parameter_count()),
        train_inputs,
        train_labels,
    ).reshape(replica_count, particle_count)
    lineages = torch.arange(
        replica_count*particle_count,
        dtype=torch.long,
        device=device,
    ).reshape(replica_count, particle_count)
    log_volumes = torch.zeros(
        replica_count, dtype=torch.float64, device=device
    )
    blocks = finite_width_parameter_blocks()
    scales = list(Config.DIRECT_INITIAL_PCN_SCALES)
    current_threshold = float("inf")
    stage_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for level in range(1, Config.DIRECT_MAX_LEVELS+1):
        quantiles = torch.quantile(
            losses, Config.DIRECT_SURVIVAL_QUANTILE, dim=1
        )
        next_threshold = max(
            Config.MATCHED_LOSS, float(quantiles.max().item())
        )
        if math.isfinite(current_threshold):
            next_threshold = min(next_threshold, current_threshold)
            progress_tolerance = max(
                1e-14, abs(current_threshold)*1e-7
            )
            if (
                next_threshold >= current_threshold-progress_tolerance
                and next_threshold > (
                    Config.MATCHED_LOSS
                    + direct_constraint_tolerance(Config.MATCHED_LOSS)
                )
            ):
                raise RuntimeError(
                    f"{condition.name} direct SMC threshold停滞于"
                    f"{current_threshold:.9g}。"
                )
        new_particles = torch.empty_like(particles)
        new_lineages = torch.empty_like(lineages)
        survival = np.zeros(replica_count, dtype=np.float64)
        for replica in range(replica_count):
            survivors = torch.nonzero(
                losses[replica] <= (
                    next_threshold+direct_constraint_tolerance(next_threshold)
                ),
                as_tuple=False,
            ).flatten()
            if not len(survivors):
                raise RuntimeError(
                    f"{condition.name} replica={replica}没有survivor。"
                )
            survival[replica] = len(survivors)/particle_count
            choices = torch.randint(
                len(survivors),
                (particle_count,),
                device=device,
                generator=generator,
            )
            selected = survivors[choices]
            new_particles[replica] = particles[replica, selected]
            new_lineages[replica] = lineages[replica, selected]
        particles = new_particles
        lineages = new_lineages
        losses = finite_width_losses(
            particles.reshape(-1, parameter_count()),
            train_inputs,
            train_labels,
        ).reshape(replica_count, particle_count)
        log_volumes += torch.log(torch.as_tensor(
            survival, dtype=torch.float64, device=device
        ))
        particles, losses, scales, mutation = direct_rejuvenate(
            particles,
            losses,
            next_threshold,
            blocks,
            scales,
            train_inputs,
            train_labels,
            generator,
            Config.DIRECT_ADAPT_SWEEPS,
            Config.DIRECT_MUTATION_SWEEPS,
        )
        current_threshold = next_threshold
        elapsed = time.perf_counter()-started
        replica_logs = log_volumes.detach().cpu().numpy()
        row = {
            "condition": condition.name,
            "level": level,
            "threshold": next_threshold,
            "survival_min": float(np.min(survival)),
            "survival_median": float(np.median(survival)),
            "survival_max": float(np.max(survival)),
            "log_volume_min": float(np.min(replica_logs)),
            "log_volume_median": float(np.median(replica_logs)),
            "log_volume_max": float(np.max(replica_logs)),
            "unique_lineage_fraction_min": min(
                float(torch.unique(lineages[replica]).numel()/particle_count)
                for replica in range(replica_count)
            ),
            "elapsed_seconds": elapsed,
            **mutation,
        }
        stage_rows.append(row)
        if (
            level == 1
            or level % Config.DIRECT_LOG_EVERY_LEVELS == 0
            or next_threshold <= (
                Config.MATCHED_LOSS
                + direct_constraint_tolerance(Config.MATCHED_LOSS)
            )
        ):
            acceptance = ",".join(
                f"{block.name}:{mutation[f'acceptance_{block.name}']:.1%}"
                for block in blocks
            )
            print(
                f"DIRECT-SMC {condition.name} level={level} | "
                f"eps={next_threshold:.6g} | "
                f"logV={np.median(replica_logs):.2f} "
                f"sd={np.std(replica_logs):.2f} | "
                f"accept[{acceptance}] | elapsed={elapsed:.1f}s",
                flush=True,
            )
        if next_threshold <= (
            Config.MATCHED_LOSS
            + direct_constraint_tolerance(Config.MATCHED_LOSS)
        ):
            break
    else:
        raise RuntimeError(f"{condition.name} direct SMC超过最大levels。")
    particles, losses, scales, final_mutation = direct_rejuvenate(
        particles,
        losses,
        Config.MATCHED_LOSS,
        blocks,
        scales,
        train_inputs,
        train_labels,
        generator,
        Config.DIRECT_ADAPT_SWEEPS,
        Config.DIRECT_FINAL_MUTATION_SWEEPS,
    )
    flat_logits = finite_width_probe_logits(
        particles.reshape(-1, parameter_count()), probe_inputs
    )
    probe_logits = flat_logits.reshape(
        replica_count, particle_count, len(probe_indices)
    )
    replica_probability = (
        probe_logits >= 0
    ).to(torch.float64).mean(dim=1).cpu().numpy()
    probability_one = replica_probability.mean(axis=0)
    metrics = weighted_probe_metrics(probability_one, target_bits)
    probe_logits_np = probe_logits.cpu().numpy().astype(np.float32)
    distribution, top_rows = probe_distribution_summary(
        probe_logits_np, target_bits
    )
    for top_row in top_rows:
        top_row["condition"] = condition.name
    replica_logs = log_volumes.cpu().numpy()
    subset = min(Config.FW_SAVE_PARAMETER_SUBSET, particle_count)
    final_lineage_fractions = np.asarray([
        torch.unique(lineages[replica]).numel()/particle_count
        for replica in range(replica_count)
    ], dtype=np.float64)
    summary = {
        "status": "completed",
        "condition": asdict(condition),
        "matched_loss": Config.MATCHED_LOSS,
        "network": "exact finite-width 8->16->16->1 tanh",
        "parameter_count": parameter_count(),
        "nngp_used": False,
        "optimizer_or_anchor_used": False,
        "predicted_log_static_volume": float(
            logsumexp(replica_logs)-math.log(len(replica_logs))
        ),
        "replica_log_static_volumes": replica_logs,
        "replica_log_volume_std": float(np.std(replica_logs)),
        "probe_probability_one": probability_one,
        "probe_probability_replica_min": replica_probability.min(axis=0),
        "probe_probability_replica_max": replica_probability.max(axis=0),
        "probe_probability_replica_std": replica_probability.std(axis=0),
        "maximum_probe_replica_std": float(
            replica_probability.std(axis=0).max()
        ),
        "level_count": len(stage_rows),
        "final_loss_max": float(losses.max().item()),
        "final_unique_lineage_fraction_min": float(
            final_lineage_fractions.min()
        ),
        "elapsed_seconds": time.perf_counter()-started,
        **metrics,
        **distribution,
    }
    quality_checks = {
        "replica_log_volume_std_le_1nat": bool(
            summary["replica_log_volume_std"] <= 1.0
        ),
        "maximum_probe_replica_std_le_0p03": bool(
            summary["maximum_probe_replica_std"] <= 0.03
        ),
        "all_particles_respect_loss_threshold": bool(
            summary["final_loss_max"] <= (
                Config.MATCHED_LOSS
                + direct_constraint_tolerance(Config.MATCHED_LOSS)
            )
        ),
    }
    summary["quality_checks"] = quality_checks
    summary["quality_pass"] = bool(all(quality_checks.values()))
    np.savez_compressed(
        output_dir / "direct_smc_condition_samples.npz",
        probe_indices=probe_indices,
        target_bits=target_bits,
        probe_logits=probe_logits_np,
        probe_probability_by_replica=replica_probability,
        parameter_subsets=particles[:, :subset].cpu().numpy().astype(np.float32),
        log_volumes=replica_logs,
        final_lineage_fractions=final_lineage_fractions,
    )
    write_csv(output_dir / "direct_smc_levels.csv", stage_rows)
    write_csv(output_dir / "top_probe_functions.csv", top_rows)
    write_json(summary_path, summary)
    print(
        f"DIRECT-RESULT {condition.name} | "
        f"probe_bit_acc={metrics['probe_target_accuracy']:.4f} | "
        f"probe_bit_agreement={metrics['probe_pairwise_agreement']:.4f} | "
        f"modal_probe_bit_acc={metrics['probe_modal_accuracy']:.4f} | "
        f"replica_sd={summary['maximum_probe_replica_std']:.4g} | "
        f"quality={'PASS' if summary['quality_pass'] else 'UNCONVERGED'} | "
        f"logV={summary['predicted_log_static_volume']:.2f}",
        flush=True,
    )
    return summary


def run_finite_width_direct_smc(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = finite_width_direct_protocol_payload()
    protocol_hash = payload_sha256(protocol)
    write_json(output_dir / "protocol.json", {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
    })
    conditions, _, probe_indices = build_protocol_objects()
    start = max(0, Config.FW_CONDITION_START)
    stop = (
        len(conditions)
        if Config.FW_CONDITION_STOP is None
        else min(Config.FW_CONDITION_STOP, len(conditions))
    )
    selected = conditions[start:stop]
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    full_inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    full_targets = torch.as_tensor(
        target_outputs().astype(np.float32), device=device
    )
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, condition in enumerate(selected, start=1):
        print(
            f"\n=== DIRECT FINITE WIDTH {ordinal}/{len(selected)} "
            f"{condition.name} ===",
            flush=True,
        )
        summaries.append(run_finite_width_direct_condition(
            output_dir / "conditions" / condition.name,
            condition,
            device,
            full_inputs,
            full_targets,
            probe_indices,
        ))
    root_summary = {
        "status": "completed",
        "protocol_sha256": protocol_hash,
        "all_quality_pass": bool(all(
            summary.get("quality_pass", False) for summary in summaries
        )),
        "elapsed_seconds": time.perf_counter()-started,
        "results": summaries,
        "interpretation_boundary": (
            "Direct prior-initialized finite-width constrained SMC; no NNGP, "
            "optimizer, trained anchor, or revealed validation result is an input."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, root_summary)
    return summary_path


def load_and_validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prediction_hash = manifest.pop("prediction_sha256")
    computed = payload_sha256(manifest)
    manifest["prediction_sha256"] = prediction_hash
    if computed != prediction_hash:
        raise RuntimeError("prediction manifest SHA256校验失败。")
    current_protocol = protocol_payload()
    current_protocol_hash = payload_sha256(current_protocol)
    if manifest.get("protocol_sha256") != current_protocol_hash:
        raise RuntimeError("当前验证协议与冻结预测协议不一致。")
    if manifest.get("protocol") != json_ready(current_protocol):
        raise RuntimeError("prediction manifest中的完整协议内容不一致。")
    if not manifest.get("created_before_validation"):
        raise RuntimeError("预测文件没有声明先于验证创建。")
    return manifest


def train_condition_at_matched_loss(
    condition: ConditionSpec,
    base_initialization: torch.Tensor,
    full_inputs: torch.Tensor,
    targets: torch.Tensor,
    probe_indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    train_indices = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=device
    )
    probe_index = torch.as_tensor(
        probe_indices, dtype=torch.long, device=device
    )
    train_x = full_inputs[train_indices]
    train_y = targets[train_indices]
    probe_x = full_inputs[probe_index]
    seed_count = len(base_initialization)
    parameters = torch.nn.Parameter(base_initialization.clone().to(device))
    optimizer = torch.optim.AdamW(
        [parameters],
        lr=Config.LEARNING_RATE,
        betas=Config.ADAM_BETAS,
        eps=Config.ADAM_EPS,
        weight_decay=Config.WEIGHT_DECAY,
    )
    reached = torch.zeros(seed_count, dtype=torch.bool, device=device)
    recorded_probe = torch.full(
        (seed_count, len(probe_indices)),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    recorded_step = torch.full(
        (seed_count,), -1, dtype=torch.int32, device=device
    )
    recorded_loss = torch.full(
        (seed_count,), float("nan"), dtype=torch.float32, device=device
    )
    expanded_train = train_x[None].expand(seed_count, -1, -1)
    expanded_probe = probe_x[None].expand(seed_count, -1, -1)
    started = time.perf_counter()
    final_step = 0
    for step in range(Config.MAX_STEPS+1):
        final_step = step
        logits = forward_normalized(parameters, expanded_train)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            train_y[None].expand_as(logits),
            reduction="none",
        ).mean(dim=1)
        newly = torch.logical_and(~reached, losses <= Config.MATCHED_LOSS)
        if bool(torch.any(newly).item()):
            probe_logits = forward_normalized(
                parameters[newly], expanded_probe[newly]
            )
            recorded_probe[newly] = probe_logits
            recorded_step[newly] = step
            recorded_loss[newly] = losses[newly]
            reached[newly] = True
        if bool(torch.all(reached).item()) or step == Config.MAX_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        if step and step % Config.LOG_EVERY_STEPS == 0:
            print(
                f"TRAIN {condition.name} step={step:,} | "
                f"loss median={losses.median().item():.5g} | "
                f"reached={reached.float().mean().item():.2%} | "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    valid = reached.detach().cpu().numpy()
    probe_logits = recorded_probe.detach().cpu().numpy()
    probability_one = np.mean(probe_logits[valid] >= 0, axis=0)
    probe_targets = target_outputs()[probe_indices]
    metrics = weighted_probe_metrics(probability_one, probe_targets)
    return {
        "condition": asdict(condition),
        "matched_loss": Config.MATCHED_LOSS,
        "seed_count": seed_count,
        "reached_count": int(valid.sum()),
        "reached_fraction": float(valid.mean()),
        "final_step": final_step,
        "median_crossing_step": (
            float(np.median(recorded_step.detach().cpu().numpy()[valid]))
            if valid.any() else None
        ),
        "probe_probability_one": probability_one,
        "probe_logits_at_crossing": probe_logits,
        "recorded_steps": recorded_step.detach().cpu().numpy(),
        "recorded_losses": recorded_loss.detach().cpu().numpy(),
        **metrics,
    }


def run_validation(output_dir: Path, manifest_path: Path) -> None:
    manifest = load_and_validate_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions, _, probe_indices = build_protocol_objects()
    full_inputs_np = truth_table_inputs().astype(np.float32)
    targets_np = target_outputs().astype(np.float32)
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(Config.VALIDATION_INITIALIZATION_SEED)
    base_initialization = torch.randn(
        Config.VALIDATION_SEEDS,
        parameter_count(),
        generator=generator,
    )
    full_inputs = torch.as_tensor(full_inputs_np, device=device)
    targets = torch.as_tensor(targets_np, device=device)
    actual_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_by_name = {
        item["condition"]["name"]: item
        for item in manifest["predictions"]
    }
    for condition in conditions:
        result = train_condition_at_matched_loss(
            condition,
            base_initialization,
            full_inputs,
            targets,
            probe_indices,
            device,
        )
        actual_results.append(result)
        predicted = prediction_by_name[condition.name]
        predicted_probability = np.asarray(
            predicted["probe_probability_one"], dtype=np.float64
        )
        actual_probability = np.asarray(
            result["probe_probability_one"], dtype=np.float64
        )
        comparison_rows.append({
            "condition": condition.name,
            "reached_fraction": result["reached_fraction"],
            "predicted_accuracy": predicted["probe_target_accuracy"],
            "actual_accuracy": result["probe_target_accuracy"],
            "accuracy_error": (
                result["probe_target_accuracy"]
                - predicted["probe_target_accuracy"]
            ),
            "predicted_agreement": predicted["probe_pairwise_agreement"],
            "actual_agreement": result["probe_pairwise_agreement"],
            "agreement_error": (
                result["probe_pairwise_agreement"]
                - predicted["probe_pairwise_agreement"]
            ),
            "probe_probability_mae": float(np.mean(np.abs(
                actual_probability-predicted_probability
            ))),
            "probe_probability_correlation": float(np.corrcoef(
                actual_probability, predicted_probability
            )[0, 1]),
            "predicted_modal_accuracy": predicted["probe_modal_accuracy"],
            "actual_modal_accuracy": result["probe_modal_accuracy"],
            "predicted_modal_bits": predicted["probe_modal_bits"],
            "actual_modal_bits": result["probe_modal_bits"],
        })
        print(
            f"VALIDATE {condition.name} | "
            f"pred/actual acc={predicted['probe_target_accuracy']:.4f}/"
            f"{result['probe_target_accuracy']:.4f} | "
            f"agreement={predicted['probe_pairwise_agreement']:.4f}/"
            f"{result['probe_pairwise_agreement']:.4f}",
            flush=True,
        )
    np.savez_compressed(
        output_dir / "validation_raw_results.npz",
        condition_names=np.asarray([c.name for c in conditions]),
        probe_indices=probe_indices,
        probe_logits=np.stack([
            result["probe_logits_at_crossing"] for result in actual_results
        ]),
        recorded_steps=np.stack([
            result["recorded_steps"] for result in actual_results
        ]),
        recorded_losses=np.stack([
            result["recorded_losses"] for result in actual_results
        ]),
    )
    write_csv(output_dir / "prediction_vs_validation.csv", comparison_rows)
    write_json(output_dir / "validation_summary.json", {
        "status": "completed",
        "prediction_manifest": str(manifest_path),
        "prediction_sha256": manifest["prediction_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "comparison": comparison_rows,
        "mean_probe_probability_mae": float(np.mean([
            row["probe_probability_mae"] for row in comparison_rows
        ])),
        "mean_absolute_accuracy_error": float(np.mean(np.abs([
            row["accuracy_error"] for row in comparison_rows
        ]))),
        "mean_absolute_agreement_error": float(np.mean(np.abs([
            row["agreement_error"] for row in comparison_rows
        ]))),
    })


def create_archive(result_dir: Path) -> Path:
    archive_path = result_dir.parent / f"{result_dir.name}_package.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(result_dir.parent))
    return archive_path


def l2_coefficient_key(value: float) -> str:
    return "lambda_" + f"{value:.0e}".replace("-", "m").replace("+", "p")


def l2_and_condition() -> tuple[ConditionSpec, np.ndarray]:
    conditions, _, probe_indices = build_protocol_objects()
    matches = [
        condition for condition in conditions
        if condition.train_count == Config.L2_AND_TRAIN_COUNT
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"无法唯一确定balanced-AND n={Config.L2_AND_TRAIN_COUNT}条件。"
        )
    return matches[0], probe_indices


@torch.no_grad()
def l2_dynamic_observation(
    parameters: torch.Tensor,
    coefficient: float,
    step: int,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    full_inputs: torch.Tensor,
    full_targets: torch.Tensor,
    train_indices: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    count = len(parameters)
    train_logits = forward_normalized(
        parameters,
        train_inputs[None].expand(count, -1, -1),
    )
    raw_bce = F.binary_cross_entropy_with_logits(
        train_logits,
        train_labels[None].expand_as(train_logits),
        reduction="none",
    ).mean(dim=1)
    half_norm = 0.5 * parameters.square().sum(dim=1)
    total = raw_bce + float(coefficient) * half_norm
    full_logits = forward_normalized(
        parameters,
        full_inputs[None].expand(count, -1, -1),
    )
    predictions = full_logits >= 0
    targets = full_targets.bool()
    train_exact = torch.all(
        predictions[:, train_indices] == targets[None, train_indices], dim=1
    )
    target_exact = torch.all(predictions == targets[None], dim=1)
    heldout = torch.ones(len(full_targets), dtype=torch.bool, device=parameters.device)
    heldout[train_indices] = False
    heldout_accuracy = (
        predictions[:, heldout] == targets[None, heldout]
    ).float().mean(dim=1)
    full_accuracy = (
        predictions == targets[None]
    ).float().mean(dim=1)
    probability_one = predictions.float().mean(dim=0)
    agreement = float(torch.mean(
        probability_one.square() + (1.0 - probability_one).square()
    ).item())
    row = {
        "coefficient": float(coefficient),
        "step": int(step),
        "raw_bce_mean": float(raw_bce.mean().item()),
        "raw_bce_median": float(raw_bce.median().item()),
        "total_objective_mean": float(total.mean().item()),
        "total_objective_median": float(total.median().item()),
        "half_norm_mean": float(half_norm.mean().item()),
        "train_exact_fraction": float(train_exact.float().mean().item()),
        "target_function_fraction": float(target_exact.float().mean().item()),
        "heldout_accuracy_mean": float(heldout_accuracy.mean().item()),
        "full_accuracy_mean": float(full_accuracy.mean().item()),
        "function_agreement": agreement,
    }
    return row, {
        "raw_bce": raw_bce,
        "total_objective": total,
        "train_exact": train_exact,
        "target_exact": target_exact,
        "full_predictions": predictions,
    }


def run_l2_and_dynamic(
    output_dir: Path,
    condition: ConditionSpec,
    device: torch.device,
    full_inputs: torch.Tensor,
    full_targets: torch.Tensor,
) -> dict[str, Any]:
    summary_path = output_dir / "dynamic_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print("显式L2动态阶段已完成，跳过。", flush=True)
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    train_indices = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=device
    )
    train_inputs = full_inputs[train_indices]
    train_labels = full_targets[train_indices]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(Config.VALIDATION_INITIALIZATION_SEED + 91_003)
    base = torch.randn(
        Config.L2_AND_DYNAMIC_SEEDS,
        parameter_count(),
        generator=generator,
    ).to(device)
    trajectory_rows: list[dict[str, Any]] = []
    coefficient_summaries: list[dict[str, Any]] = []
    final_parameter_subsets: list[np.ndarray] = []
    started_all = time.perf_counter()
    for coefficient in Config.L2_AND_COEFFICIENTS:
        parameters = torch.nn.Parameter(base.clone())
        optimizer = torch.optim.Adam(
            [parameters],
            lr=Config.LEARNING_RATE,
            betas=Config.ADAM_BETAS,
            eps=Config.ADAM_EPS,
            weight_decay=0.0,
        )
        train_fit_step = torch.full(
            (len(parameters),), -1, dtype=torch.int32, device=device
        )
        target_step = torch.full_like(train_fit_step, -1)
        ever_target = torch.zeros(
            len(parameters), dtype=torch.bool, device=device
        )
        final_artifacts: dict[str, torch.Tensor] | None = None
        started = time.perf_counter()
        for step in range(Config.L2_AND_DYNAMIC_STEPS + 1):
            should_evaluate = bool(
                step in {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000}
                or step % Config.L2_AND_DYNAMIC_EVAL_INTERVAL == 0
                or step == Config.L2_AND_DYNAMIC_STEPS
            )
            if should_evaluate:
                row, artifacts = l2_dynamic_observation(
                    parameters,
                    coefficient,
                    step,
                    train_inputs,
                    train_labels,
                    full_inputs,
                    full_targets,
                    train_indices,
                )
                newly_fit = (train_fit_step < 0) & artifacts["train_exact"]
                train_fit_step[newly_fit] = step
                newly_target = (target_step < 0) & artifacts["target_exact"]
                target_step[newly_target] = step
                ever_target |= artifacts["target_exact"]
                row["ever_target_fraction"] = float(
                    ever_target.float().mean().item()
                )
                row["elapsed_seconds"] = time.perf_counter() - started
                trajectory_rows.append(row)
                final_artifacts = artifacts
                if (
                    step <= 1_000
                    or step % max(Config.LOG_EVERY_STEPS, 1) == 0
                    or step == Config.L2_AND_DYNAMIC_STEPS
                ):
                    eta_seconds = (
                        row["elapsed_seconds"]
                        * (Config.L2_AND_DYNAMIC_STEPS - step)
                        / step
                        if step > 0 else None
                    )
                    eta_text = (
                        f"{eta_seconds/60:.1f}m"
                        if eta_seconds is not None else "?"
                    )
                    print(
                        f"DYNAMIC lambda={coefficient:g} step={step:>6,} | "
                        f"BCE={row['raw_bce_median']:.4g} "
                        f"J={row['total_objective_median']:.4g} | "
                        f"train={row['train_exact_fraction']:.1%} "
                        f"target={row['target_function_fraction']:.1%} "
                        f"ever={row['ever_target_fraction']:.1%} | "
                        f"elapsed={row['elapsed_seconds']:.1f}s "
                        f"ETA={eta_text}",
                        flush=True,
                    )
            if step == Config.L2_AND_DYNAMIC_STEPS:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = forward_normalized(
                parameters,
                train_inputs[None].expand(len(parameters), -1, -1),
            )
            raw_bce = F.binary_cross_entropy_with_logits(
                logits,
                train_labels[None].expand_as(logits),
                reduction="none",
            ).mean(dim=1)
            total = raw_bce + (
                float(coefficient)
                * 0.5
                * parameters.square().sum(dim=1)
            )
            total.sum().backward()
            optimizer.step()
        if final_artifacts is None:
            raise RuntimeError("动态阶段没有产生最终观测。")
        train_exact = final_artifacts["train_exact"]
        total_np = final_artifacts["total_objective"].detach().cpu().numpy()
        valid_total = total_np[train_exact.detach().cpu().numpy()]
        if len(valid_total):
            calibrated = float(np.quantile(
                valid_total, Config.L2_AND_STATIC_THRESHOLD_QUANTILE
            ))
            threshold = max(
                Config.L2_AND_STATIC_THRESHOLD_FLOOR,
                calibrated,
            )
            calibration_valid = True
        else:
            threshold = Config.L2_AND_STATIC_THRESHOLD_FLOOR
            calibration_valid = False
        valid_delay = (train_fit_step >= 0) & (target_step >= train_fit_step)
        delays = (
            (target_step[valid_delay] + 1).float()
            / (train_fit_step[valid_delay] + 1).float()
            if bool(torch.any(valid_delay).item()) else torch.empty(0, device=device)
        )
        final_row = next(
            row for row in reversed(trajectory_rows)
            if math.isclose(float(row["coefficient"]), float(coefficient))
        )
        coefficient_summaries.append({
            "coefficient": float(coefficient),
            "final": final_row,
            "train_fit_seed_fraction": float(
                (train_fit_step >= 0).float().mean().item()
            ),
            "ever_target_seed_fraction": float(ever_target.float().mean().item()),
            "median_target_delay_ratio": (
                float(delays.median().item()) if len(delays) else None
            ),
            "static_total_objective_threshold": threshold,
            "threshold_calibration_valid": calibration_valid,
            "threshold_floor": Config.L2_AND_STATIC_THRESHOLD_FLOOR,
        })
        subset = min(64, len(parameters))
        final_parameter_subsets.append(
            parameters[:subset].detach().cpu().numpy().astype(np.float32)
        )
    write_csv(output_dir / "dynamic_trajectory.csv", trajectory_rows)
    np.savez_compressed(
        output_dir / "dynamic_parameter_subsets.npz",
        coefficients=np.asarray(Config.L2_AND_COEFFICIENTS),
        parameters=np.stack(final_parameter_subsets),
    )
    summary = {
        "status": "completed",
        "condition": asdict(condition),
        "network": "8->16->16->1 tanh",
        "parameter_count": parameter_count(),
        "objective": "mean_BCE_D + lambda*||normalized_parameters||^2/2",
        "seed_count": Config.L2_AND_DYNAMIC_SEEDS,
        "max_steps": Config.L2_AND_DYNAMIC_STEPS,
        "elapsed_seconds": time.perf_counter() - started_all,
        "coefficients": coefficient_summaries,
    }
    write_json(summary_path, summary)
    return summary


@torch.no_grad()
def evaluate_l2_static_parameter_subset(
    particles: torch.Tensor,
    coefficient: float,
    condition: ConditionSpec,
    full_inputs: torch.Tensor,
    full_targets: torch.Tensor,
) -> dict[str, Any]:
    train_indices = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=particles.device
    )
    logits = forward_normalized(
        particles,
        full_inputs[None].expand(len(particles), -1, -1),
    )
    predictions = logits >= 0
    targets = full_targets.bool()
    raw_bce = F.binary_cross_entropy_with_logits(
        logits[:, train_indices],
        full_targets[train_indices][None].expand(
            len(particles), -1
        ),
        reduction="none",
    ).mean(dim=1)
    half_norm = 0.5 * particles.square().sum(dim=1)
    total = raw_bce + float(coefficient) * half_norm
    train_exact = torch.all(
        predictions[:, train_indices] == targets[None, train_indices], dim=1
    )
    target_exact = torch.all(predictions == targets[None], dim=1)
    heldout = torch.ones(len(full_targets), dtype=torch.bool, device=particles.device)
    heldout[train_indices] = False
    heldout_accuracy = (
        predictions[:, heldout] == targets[None, heldout]
    ).float().mean(dim=1)
    full_accuracy = (
        predictions == targets[None]
    ).float().mean(dim=1)
    return {
        "sample_count": len(particles),
        "raw_bce_mean": float(raw_bce.mean().item()),
        "raw_bce_max": float(raw_bce.max().item()),
        "half_norm_mean": float(half_norm.mean().item()),
        "total_objective_mean": float(total.mean().item()),
        "total_objective_max": float(total.max().item()),
        "train_exact_mass": float(train_exact.float().mean().item()),
        "target_function_mass": float(target_exact.float().mean().item()),
        "heldout_accuracy_mean": float(heldout_accuracy.mean().item()),
        "full_accuracy_mean": float(full_accuracy.mean().item()),
    }


def run_explicit_l2_and_n32(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition, probe_indices = l2_and_condition()
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    torch.backends.cuda.matmul.allow_tf32 = bool(Config.ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(Config.ALLOW_TF32)
    full_inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    full_targets = torch.as_tensor(
        target_outputs().astype(np.float32), device=device
    )
    dynamic = run_l2_and_dynamic(
        output_dir, condition, device, full_inputs, full_targets
    )
    dynamic_by_coefficient = {
        float(row["coefficient"]): row for row in dynamic["coefficients"]
    }
    protocol = {
        "protocol_version": "8bit_and_boundary_explicit_l2_landscape_v2",
        "task": "balanced AND y=x0*x1",
        "condition": asdict(condition),
        "network": "8->16->16->1 tanh",
        "parameter_count": parameter_count(),
        "parameter_coordinates": (
            "iid_standard_Gaussian with fan-in forward scaling"
        ),
        "objective": (
            "mean_BCE_D + lambda*||normalized_parameters||^2/2"
        ),
        "coefficients": Config.L2_AND_COEFFICIENTS,
        "dynamic_seed_count": Config.L2_AND_DYNAMIC_SEEDS,
        "dynamic_steps": Config.L2_AND_DYNAMIC_STEPS,
        "dynamic_endpoint_thresholds": {
            f"{coefficient:g}": dynamic_by_coefficient[float(coefficient)][
                "static_total_objective_threshold"
            ]
            for coefficient in Config.L2_AND_COEFFICIENTS
        },
        "effective_static_thresholds": {
            f"{coefficient:g}": (
                Config.L2_AND_STATIC_THRESHOLD_MULTIPLIER
                * dynamic_by_coefficient[float(coefficient)][
                    "static_total_objective_threshold"
                ]
            )
            for coefficient in Config.L2_AND_COEFFICIENTS
        },
        "static_threshold_rule": (
            f"{Config.L2_AND_STATIC_THRESHOLD_MULTIPLIER:g} * "
            f"max(floor={Config.L2_AND_STATIC_THRESHOLD_FLOOR:g}, "
            "median final J among train-exact dynamic seeds)"
        ),
        "bridge_replicas": Config.FW_REPLICAS,
        "bridge_particles": Config.FW_PARTICLES,
        "anchor_target_and_event_use_same_explicit_objective": True,
        "anchor_proposal": {
            "target_relaxation": Config.L2_AND_ANCHOR_THRESHOLD_RELAXATION,
            "max_steps": Config.L2_AND_ANCHOR_MAX_STEPS,
            "note": (
                "Anchors are proposal centers only; the beta=1 bridge event and "
                "all final particles still obey the unrelaxed J threshold."
            ),
        },
        "labels_boundary": (
            "All 256 labels are known because the target rule is synthetic; only "
            f"the frozen {condition.train_count} training labels enter dynamic "
            "gradients and static "
            "bridge events. Held-out labels are observables only."
        ),
    }
    protocol_hash = payload_sha256(protocol)
    write_json(output_dir / "protocol.json", {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
    })
    np.savez_compressed(
        output_dir / "frozen_dataset.npz",
        inputs=truth_table_inputs().astype(np.float32),
        targets=target_outputs().astype(np.uint8),
        train_indices=np.asarray(condition.train_indices, dtype=np.int64),
        probe_indices=probe_indices,
    )
    dynamic_sequence = [
        dynamic_by_coefficient[float(coefficient)][
            "ever_target_seed_fraction"
        ]
        for coefficient in Config.L2_AND_COEFFICIENTS
    ]
    dynamic_train_fit = [
        dynamic_by_coefficient[float(coefficient)][
            "train_fit_seed_fraction"
        ]
        for coefficient in Config.L2_AND_COEFFICIENTS
    ]
    dynamic_nondecreasing = all(
        dynamic_sequence[index] <= dynamic_sequence[index+1] + 1e-9
        for index in range(len(dynamic_sequence)-1)
    )
    dynamic_gate_pass = bool(
        all(value >= 0.95 for value in dynamic_train_fit)
        and dynamic_nondecreasing
        and dynamic_sequence[-1] - dynamic_sequence[0] > 0.0
    )
    if Config.SMOKE_TEST:
        # smoke只检查后续代码路径；2步预算不承担科学资格判决。
        dynamic_gate_pass = True
    if not dynamic_gate_pass:
        summary_path = output_dir / "summary.json"
        write_json(summary_path, {
            "status": "dynamic_gate_failed",
            "protocol_sha256": protocol_hash,
            "condition": asdict(condition),
            "coefficients": Config.L2_AND_COEFFICIENTS,
            "dynamic": dynamic,
            "dynamic_target_sequence": dynamic_sequence,
            "dynamic_train_fit_sequence": dynamic_train_fit,
            "dynamic_target_nondecreasing": dynamic_nondecreasing,
            "static_skipped": True,
            "reason": (
                "Static SMC is allowed only when explicit L2 improves dynamic "
                "target entry while all branches still fit the training set."
            ),
        })
        print(
            "DYNAMIC GATE FAILED：不运行静态SMC。target="
            f"{dynamic_sequence} train_fit={dynamic_train_fit}",
            flush=True,
        )
        return summary_path
    static_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for coefficient in Config.L2_AND_COEFFICIENTS:
        calibration = dynamic_by_coefficient[float(coefficient)]
        base_threshold = float(
            calibration["static_total_objective_threshold"]
        )
        threshold = (
            Config.L2_AND_STATIC_THRESHOLD_MULTIPLIER * base_threshold
        )
        Config.FW_EXPLICIT_L2_COEFFICIENT = float(coefficient)
        Config.MATCHED_LOSS = threshold
        Config.FW_ANCHOR_TARGET_LOSS = (
            threshold * Config.L2_AND_ANCHOR_THRESHOLD_RELAXATION
        )
        Config.FW_ANCHOR_MAX_STEPS = Config.L2_AND_ANCHOR_MAX_STEPS
        slack_key = str(Config.L2_AND_STATIC_THRESHOLD_MULTIPLIER).replace(
            ".", "p"
        )
        condition_dir = (
            output_dir
            / f"static_threshold_x{slack_key}"
            / l2_coefficient_key(coefficient)
        )
        print(
            f"\n=== STATIC lambda={coefficient:g} "
            f"J_threshold={threshold:.6g} "
            f"anchor_target={Config.FW_ANCHOR_TARGET_LOSS:.6g} ===",
            flush=True,
        )
        summary = run_finite_width_condition(
            condition_dir,
            condition,
            device,
            full_inputs,
            full_targets,
            probe_indices,
        )
        sample_path = condition_dir / "finite_width_condition_samples.npz"
        with np.load(sample_path) as payload:
            particles_np = payload["parameter_subsets"].reshape(
                -1, parameter_count()
            )
        particles = torch.from_numpy(particles_np).to(device)
        full_metrics = evaluate_l2_static_parameter_subset(
            particles,
            coefficient,
            condition,
            full_inputs,
            full_targets,
        )
        summary["explicit_l2_coefficient"] = float(coefficient)
        summary["dynamic_calibration"] = calibration
        summary["full_function_subset_metrics"] = full_metrics
        write_json(condition_dir / "summary.json", summary)
        static_rows.append({
            "coefficient": float(coefficient),
            "base_dynamic_threshold": base_threshold,
            "total_objective_threshold": threshold,
            "quality_pass": summary["quality_pass"],
            "predicted_log_static_volume": summary[
                "predicted_log_static_volume"
            ],
            "probe_target_accuracy": summary["probe_target_accuracy"],
            "probe_modal_accuracy": summary["probe_modal_accuracy"],
            **full_metrics,
        })
    write_csv(output_dir / "static_order_summary.csv", static_rows)
    target_sequence = [row["target_function_mass"] for row in static_rows]
    full_sequence = [row["full_accuracy_mean"] for row in static_rows]
    raw_bce_sequence = [row["raw_bce_mean"] for row in static_rows]
    static_train_exact_sequence = [
        row["train_exact_mass"] for row in static_rows
    ]
    target_nondecreasing = all(
        target_sequence[index] <= target_sequence[index+1] + 1e-9
        for index in range(len(target_sequence)-1)
    )
    root_summary = {
        "status": "completed",
        "protocol_sha256": protocol_hash,
        "condition": asdict(condition),
        "coefficients": Config.L2_AND_COEFFICIENTS,
        "objective": "mean_BCE_D + lambda*||normalized_parameters||^2/2",
        "dynamic": dynamic,
        "static": static_rows,
        "dynamic_target_sequence": dynamic_sequence,
        "static_target_sequence": target_sequence,
        "static_full_accuracy_sequence": full_sequence,
        "static_raw_bce_sequence": raw_bce_sequence,
        "static_train_exact_sequence": static_train_exact_sequence,
        "all_static_raw_bce_le_0p01": bool(all(
            value <= 0.01 for value in raw_bce_sequence
        )),
        "all_static_train_exact_mass_ge_0p5": bool(all(
            value >= 0.50 for value in static_train_exact_sequence
        )),
        "dynamic_target_nondecreasing": dynamic_nondecreasing,
        "static_target_nondecreasing": target_nondecreasing,
        "order_match": bool(
            dynamic_nondecreasing
            and target_nondecreasing
            and dynamic_sequence[-1] - dynamic_sequence[0] > 0.0
            and target_sequence[-1] - target_sequence[0] > 0.0
            and all(value <= 0.01 for value in raw_bce_sequence)
            and all(value >= 0.50 for value in static_train_exact_sequence)
            and all(row["quality_pass"] for row in static_rows)
        ),
        "elapsed_seconds_static": time.perf_counter() - started,
        "interpretation_boundary": (
            "Dynamic and static branches use the same explicit objective. "
            "Matching order supports a static-landscape contribution but does "
            "not eliminate optimizer accessibility. Static complete-function "
            "mass is estimated from the saved 1024-particle subset."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, root_summary)
    return summary_path


def lower_norm_auc(
    target_norm: np.ndarray,
    other_norm: np.ndarray,
) -> float | None:
    """返回随机target样本范数小于随机other样本范数的概率。"""
    if not len(target_norm) or not len(other_norm):
        return None
    u = float(mannwhitneyu(
        target_norm, other_norm, alternative="two-sided"
    ).statistic)
    return 1.0-u/(len(target_norm)*len(other_norm))


def loss_stratified_lower_norm_auc(
    target_exact: np.ndarray,
    half_norm: np.ndarray,
    raw_bce: np.ndarray,
    stratum_count: int,
) -> tuple[float | None, int, int]:
    """在replica内按raw BCE分位数分层，再汇总范数成对比较。"""
    edges = np.unique(np.quantile(
        raw_bce,
        np.linspace(0.0, 1.0, max(int(stratum_count), 1)+1),
    ))
    if len(edges) < 2:
        return lower_norm_auc(
            half_norm[target_exact], half_norm[~target_exact]
        ), 1, int(target_exact.sum()*(~target_exact).sum())
    weighted_sum = 0.0
    total_pairs = 0
    used = 0
    for index in range(len(edges)-1):
        if index == len(edges)-2:
            selected = (raw_bce >= edges[index]) & (
                raw_bce <= edges[index+1]
            )
        else:
            selected = (raw_bce >= edges[index]) & (
                raw_bce < edges[index+1]
            )
        local_target = half_norm[selected & target_exact]
        local_other = half_norm[selected & ~target_exact]
        local_auc = lower_norm_auc(local_target, local_other)
        if local_auc is None:
            continue
        pair_count = len(local_target)*len(local_other)
        weighted_sum += local_auc*pair_count
        total_pairs += pair_count
        used += 1
    if not total_pairs:
        return None, used, 0
    return weighted_sum/total_pairs, used, total_pairs


def norm_reweighted_target_mass(
    target_exact: np.ndarray,
    half_norm: np.ndarray,
    gamma: float,
) -> tuple[float, float]:
    """在fixed-loss样本内施加exp(-gamma*R)后的target质量与ESS。"""
    log_weight = -float(gamma)*(
        half_norm-float(np.min(half_norm))
    )
    log_weight -= float(np.max(log_weight))
    weight = np.exp(log_weight)
    weight_sum = float(np.sum(weight))
    target_mass = float(np.sum(weight*target_exact)/weight_sum)
    ess = weight_sum**2/float(np.sum(weight**2))
    return target_mass, ess/len(weight)


def run_norm_target_n40(output_dir: Path) -> Path:
    """无WD、matched raw BCE下采样并比较范数与完整AND宏观态。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions, _, probe_indices = build_protocol_objects()
    matches = [
        condition for condition in conditions
        if condition.train_count == Config.NORM_TARGET_TRAIN_COUNT
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"无法唯一确定n={Config.NORM_TARGET_TRAIN_COUNT}条件。"
        )
    condition = matches[0]
    Config.FW_EXPLICIT_L2_COEFFICIENT = 0.0
    Config.MATCHED_LOSS = Config.NORM_TARGET_MATCHED_BCE
    Config.DIRECT_REPLICAS = Config.NORM_TARGET_REPLICAS
    Config.DIRECT_PARTICLES_PER_REPLICA = (
        Config.NORM_TARGET_PARTICLES_PER_REPLICA
    )
    Config.FW_SAVE_PARAMETER_SUBSET = min(
        Config.NORM_TARGET_SAVED_PARAMETERS_PER_REPLICA,
        Config.DIRECT_PARTICLES_PER_REPLICA,
    )
    Config.DIRECT_FINAL_MUTATION_SWEEPS = max(
        Config.DIRECT_FINAL_MUTATION_SWEEPS, 32
    )
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    full_inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    full_targets = torch.as_tensor(
        target_outputs().astype(np.float32), device=device
    )
    condition_dir = output_dir / "condition"
    static_summary = run_finite_width_direct_condition(
        condition_dir,
        condition,
        device,
        full_inputs,
        full_targets,
        probe_indices,
    )
    with np.load(
        condition_dir / "direct_smc_condition_samples.npz"
    ) as payload:
        parameter_subsets = payload["parameter_subsets"]

    train_indices = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=device
    )
    replica_rows: list[dict[str, Any]] = []
    reweight_rows: list[dict[str, Any]] = []
    all_quartile_counts = np.zeros(4, dtype=np.float64)
    all_quartile_targets = np.zeros(4, dtype=np.float64)
    packed_predictions: list[np.ndarray] = []
    target_exact_by_replica: list[np.ndarray] = []
    full_accuracy_by_replica: list[np.ndarray] = []
    raw_bce_by_replica: list[np.ndarray] = []
    half_norm_by_replica: list[np.ndarray] = []
    for replica in range(parameter_subsets.shape[0]):
        parameters = torch.from_numpy(parameter_subsets[replica]).to(device)
        with torch.no_grad():
            logits = forward_normalized(
                parameters,
                full_inputs[None].expand(len(parameters), -1, -1),
            )
            predictions = logits >= 0
            targets = full_targets.bool()
            target_exact = torch.all(
                predictions == targets[None], dim=1
            ).cpu().numpy()
            full_accuracy = (
                predictions == targets[None]
            ).float().mean(dim=1).cpu().numpy()
            raw_bce = F.binary_cross_entropy_with_logits(
                logits[:, train_indices],
                full_targets[train_indices][None].expand(
                    len(parameters), -1
                ),
                reduction="none",
            ).mean(dim=1).cpu().numpy()
            half_norm = (
                0.5 * parameters.square().sum(dim=1)
            ).cpu().numpy()
            train_exact = torch.all(
                predictions[:, train_indices]
                == targets[train_indices][None],
                dim=1,
            ).cpu().numpy()
        packed_predictions.append(np.packbits(
            predictions.cpu().numpy(), axis=1, bitorder="little"
        ))
        target_exact_by_replica.append(target_exact)
        full_accuracy_by_replica.append(full_accuracy)
        raw_bce_by_replica.append(raw_bce)
        half_norm_by_replica.append(half_norm)
        target_norm = half_norm[target_exact]
        other_norm = half_norm[~target_exact]
        lower_auc = lower_norm_auc(target_norm, other_norm)
        stratified_auc, used_strata, stratified_pairs = (
            loss_stratified_lower_norm_auc(
                target_exact,
                half_norm,
                raw_bce,
                Config.NORM_TARGET_LOSS_STRATA,
            )
        )
        if len(target_norm) and len(other_norm):
            nearest_lower: list[bool] = []
            other_indices = np.flatnonzero(~target_exact)
            for target_index in np.flatnonzero(target_exact):
                nearest = other_indices[np.argmin(np.abs(
                    raw_bce[other_indices] - raw_bce[target_index]
                ))]
                nearest_lower.append(
                    half_norm[target_index] < half_norm[nearest]
                )
            matched_lower_fraction = float(np.mean(nearest_lower))
        else:
            matched_lower_fraction = None
        boundary = raw_bce >= np.median(raw_bce)
        boundary_auc = lower_norm_auc(
            half_norm[boundary & target_exact],
            half_norm[boundary & ~target_exact],
        )
        quantiles = np.quantile(half_norm, [0.25, 0.50, 0.75])
        bins = np.digitize(half_norm, quantiles)
        quartile_rates: list[float] = []
        for quartile in range(4):
            selected = bins == quartile
            all_quartile_counts[quartile] += selected.sum()
            all_quartile_targets[quartile] += target_exact[selected].sum()
            quartile_rates.append(float(target_exact[selected].mean()))
        correlation = float(spearmanr(
            half_norm, full_accuracy
        ).statistic)
        replica_rows.append({
            "replica": replica,
            "sample_count": len(parameters),
            "target_count": int(target_exact.sum()),
            "non_target_count": int((~target_exact).sum()),
            "target_mass": float(target_exact.mean()),
            "train_exact_fraction": float(train_exact.mean()),
            "target_half_norm_mean": (
                float(target_norm.mean()) if len(target_norm) else None
            ),
            "target_half_norm_median": (
                float(np.median(target_norm)) if len(target_norm) else None
            ),
            "non_target_half_norm_mean": (
                float(other_norm.mean()) if len(other_norm) else None
            ),
            "non_target_half_norm_median": (
                float(np.median(other_norm)) if len(other_norm) else None
            ),
            "target_raw_bce_mean": (
                float(raw_bce[target_exact].mean())
                if bool(np.any(target_exact)) else None
            ),
            "non_target_raw_bce_mean": (
                float(raw_bce[~target_exact].mean())
                if bool(np.any(~target_exact)) else None
            ),
            "lower_norm_auc": lower_auc,
            "loss_stratified_lower_norm_auc": stratified_auc,
            "loss_strata_used": used_strata,
            "loss_stratified_pair_count": stratified_pairs,
            "boundary_half_lower_norm_auc": boundary_auc,
            "nearest_bce_matched_lower_norm_fraction": (
                matched_lower_fraction
            ),
            "target_rate_norm_q1": quartile_rates[0],
            "target_rate_norm_q2": quartile_rates[1],
            "target_rate_norm_q3": quartile_rates[2],
            "target_rate_norm_q4": quartile_rates[3],
            "spearman_norm_vs_full_accuracy": correlation,
            "raw_bce_mean": float(raw_bce.mean()),
            "raw_bce_max": float(raw_bce.max()),
        })
        for gamma in Config.NORM_TARGET_REWEIGHT_GAMMAS:
            target_mass, ess_fraction = norm_reweighted_target_mass(
                target_exact, half_norm, gamma
            )
            reweight_rows.append({
                "replica": replica,
                "gamma": gamma,
                "target_mass": target_mass,
                "ess_fraction": ess_fraction,
            })
    write_csv(output_dir / "norm_target_by_replica.csv", replica_rows)
    write_csv(output_dir / "norm_reweighting_by_replica.csv", reweight_rows)
    np.savez_compressed(
        output_dir / "norm_target_observables.npz",
        packed_full_predictions=np.stack(packed_predictions),
        target_exact=np.stack(target_exact_by_replica),
        full_accuracy=np.stack(full_accuracy_by_replica),
        raw_bce=np.stack(raw_bce_by_replica),
        half_norm=np.stack(half_norm_by_replica),
        train_indices=np.asarray(condition.train_indices, dtype=np.int64),
    )
    informative = [
        row for row in replica_rows
        if (
            row["loss_stratified_lower_norm_auc"] is not None
            and row["target_count"]
            >= Config.NORM_TARGET_MIN_CLASS_COUNT_PER_REPLICA
            and row["non_target_count"]
            >= Config.NORM_TARGET_MIN_CLASS_COUNT_PER_REPLICA
        )
    ]
    lower_aucs = np.asarray([
        row["lower_norm_auc"] for row in informative
    ], dtype=np.float64)
    stratified_aucs = np.asarray([
        row["loss_stratified_lower_norm_auc"] for row in informative
    ], dtype=np.float64)
    matched = np.asarray([
        row["nearest_bce_matched_lower_norm_fraction"]
        for row in informative
    ], dtype=np.float64)
    target_masses = np.asarray([
        row["target_mass"] for row in replica_rows
    ], dtype=np.float64)
    quartile_rates = (
        all_quartile_targets / np.maximum(all_quartile_counts, 1.0)
    )
    reweight_summary: list[dict[str, Any]] = []
    for gamma in Config.NORM_TARGET_REWEIGHT_GAMMAS:
        selected = [
            row for row in reweight_rows if row["gamma"] == gamma
        ]
        masses = np.asarray([
            row["target_mass"] for row in selected
        ], dtype=np.float64)
        ess = np.asarray([
            row["ess_fraction"] for row in selected
        ], dtype=np.float64)
        reweight_summary.append({
            "gamma": gamma,
            "target_mass_median": float(np.median(masses)),
            "target_mass_min": float(np.min(masses)),
            "target_mass_max": float(np.max(masses)),
            "target_mass_replica_std": float(np.std(masses)),
            "ess_fraction_median": float(np.median(ess)),
            "ess_fraction_min": float(np.min(ess)),
        })
    sample_quality_checks = {
        "direct_smc_quality_pass": bool(
            static_summary.get("quality_pass", False)
        ),
        "all_replicas_have_both_classes": bool(
            len(informative) == Config.NORM_TARGET_REPLICAS
        ),
        "target_mass_replica_std_within_limit": bool(
            np.std(target_masses)
            <= Config.NORM_TARGET_MAX_TARGET_MASS_REPLICA_STD
        ),
        "stratified_auc_replica_std_within_limit": bool(
            len(stratified_aucs)
            and np.std(stratified_aucs)
            <= Config.NORM_TARGET_MAX_STRATIFIED_AUC_REPLICA_STD
        ),
        "all_saved_samples_hard_fit_training_set": bool(all(
            row["train_exact_fraction"] >= 1.0 for row in replica_rows
        )),
    }
    sample_quality_pass = bool(all(sample_quality_checks.values()))
    supports_lower_norm = bool(
        sample_quality_pass
        and float(np.median(stratified_aucs)) >= 0.55
        and float(np.mean(stratified_aucs > 0.50)) >= 0.75
        and float(np.median(matched)) >= 0.55
    )
    supports_higher_norm = bool(
        sample_quality_pass
        and float(np.median(stratified_aucs)) <= 0.45
        and float(np.mean(stratified_aucs < 0.50)) >= 0.75
        and float(np.median(matched)) <= 0.45
    )
    if not sample_quality_pass:
        verdict = "sampler_unresolved"
    elif supports_lower_norm:
        verdict = "target_is_lower_norm_at_matched_loss"
    elif supports_higher_norm:
        verdict = "target_is_higher_norm_at_matched_loss"
    else:
        verdict = "no_stable_scalar_norm_order_detected"
    protocol = {
        "protocol_version": "8bit_n40_no_wd_norm_target_direct_smc_v2",
        "condition": asdict(condition),
        "network": "8->16->16->1 tanh, 433 Gaussian coordinates",
        "sampling": (
            "direct prior-initialized constrained SMC; no optimizer or anchor"
        ),
        "matched_raw_bce": Config.NORM_TARGET_MATCHED_BCE,
        "replicas": Config.DIRECT_REPLICAS,
        "particles_per_replica": Config.DIRECT_PARTICLES_PER_REPLICA,
        "primary_statistic": (
            "within-replica loss-stratified "
            "P(norm_target < norm_non_target), using complete 256-point "
            "function identity"
        ),
        "secondary_statistics": [
            "nearest-raw-BCE matched lower-norm fraction",
            "target mass after exp(-gamma*half_norm) reweighting",
            "target rate by norm quartile",
        ],
    }
    protocol_sha256 = payload_sha256(protocol)
    write_json(output_dir / "protocol.json", {
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
    })
    summary = {
        "status": "completed",
        "protocol_sha256": protocol_sha256,
        "condition": asdict(condition),
        "static_sampler_quality_pass": static_summary.get(
            "quality_pass", False
        ),
        "static_sampler_summary": static_summary,
        "sample_quality_checks": sample_quality_checks,
        "sample_quality_pass": sample_quality_pass,
        "replica_rows": replica_rows,
        "informative_replica_count": len(informative),
        "target_mass_median": float(np.median(target_masses)),
        "target_mass_replica_std": float(np.std(target_masses)),
        "median_lower_norm_auc": (
            float(np.median(lower_aucs)) if len(lower_aucs) else None
        ),
        "median_loss_stratified_lower_norm_auc": (
            float(np.median(stratified_aucs))
            if len(stratified_aucs) else None
        ),
        "loss_stratified_lower_norm_auc_replica_std": (
            float(np.std(stratified_aucs))
            if len(stratified_aucs) else None
        ),
        "fraction_replicas_lower_norm_auc_gt_0p5": (
            float(np.mean(stratified_aucs > 0.50))
            if len(stratified_aucs) else None
        ),
        "median_nearest_bce_matched_lower_norm_fraction": (
            float(np.median(matched)) if len(matched) else None
        ),
        "pooled_target_rate_by_norm_quartile": quartile_rates,
        "norm_reweighting": reweight_summary,
        "supports_lower_norm_target": supports_lower_norm,
        "supports_higher_norm_target": supports_higher_norm,
        "verdict": verdict,
        "interpretation_boundary": (
            "This is a static Gaussian-reference-measure sample inside the "
            "fixed no-WD raw-BCE sublevel set. Loss-stratified comparisons "
            "reduce, but do not turn the sublevel into an infinitesimal shell. "
            "A positive lower-norm result would support an L2 landscape tilt; "
            "it would not establish that Adam or AdamW equilibrates to this "
            "ensemble."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    return summary_path


def run_static_endpoint_n40(
    output_dir: Path,
    *,
    artifact_prefix: str,
    protocol_version: str,
    independent_dynamic_range: tuple[float, float] | None,
    independent_dynamic_target_mass: float | None,
) -> Path:
    """直接采样当前完整目标的静态终点函数分布。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions, _, probe_indices = build_protocol_objects()
    matches = [
        condition for condition in conditions
        if condition.train_count == Config.L2_STATIC_TRAIN_COUNT
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"无法唯一确定n={Config.L2_STATIC_TRAIN_COUNT}条件。"
        )
    condition = matches[0]
    Config.FW_EXPLICIT_L2_COEFFICIENT = Config.L2_STATIC_COEFFICIENT
    Config.MATCHED_LOSS = Config.L2_STATIC_J_THRESHOLD
    Config.DIRECT_REPLICAS = Config.L2_STATIC_REPLICAS
    Config.DIRECT_PARTICLES_PER_REPLICA = (
        Config.L2_STATIC_PARTICLES_PER_REPLICA
    )
    Config.FW_SAVE_PARAMETER_SUBSET = min(
        Config.L2_STATIC_SAVED_PARAMETERS_PER_REPLICA,
        Config.DIRECT_PARTICLES_PER_REPLICA,
    )
    Config.DIRECT_FINAL_MUTATION_SWEEPS = max(
        Config.DIRECT_FINAL_MUTATION_SWEEPS, 64
    )
    Config.DIRECT_MAX_LEVELS = max(Config.DIRECT_MAX_LEVELS, 2_000)
    device = torch.device(Config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但torch看不到GPU。")
    full_inputs = torch.as_tensor(
        truth_table_inputs().astype(np.float32), device=device
    )
    full_targets = torch.as_tensor(
        target_outputs().astype(np.float32), device=device
    )
    condition_dir = output_dir / "condition"
    static_summary = run_finite_width_direct_condition(
        condition_dir,
        condition,
        device,
        full_inputs,
        full_targets,
        probe_indices,
    )
    with np.load(
        condition_dir / "direct_smc_condition_samples.npz"
    ) as payload:
        parameter_subsets = payload["parameter_subsets"]

    train_indices = torch.as_tensor(
        condition.train_indices, dtype=torch.long, device=device
    )
    targets = full_targets.bool()
    rows: list[dict[str, Any]] = []
    target_exact_arrays: list[np.ndarray] = []
    train_exact_arrays: list[np.ndarray] = []
    raw_bce_arrays: list[np.ndarray] = []
    half_norm_arrays: list[np.ndarray] = []
    objective_arrays: list[np.ndarray] = []
    packed_predictions: list[np.ndarray] = []
    for replica in range(parameter_subsets.shape[0]):
        parameters = torch.from_numpy(parameter_subsets[replica]).to(device)
        with torch.no_grad():
            logits = forward_normalized(
                parameters,
                full_inputs[None].expand(len(parameters), -1, -1),
            )
            predictions = logits >= 0
            raw_bce = F.binary_cross_entropy_with_logits(
                logits[:, train_indices],
                full_targets[train_indices][None].expand(
                    len(parameters), -1
                ),
                reduction="none",
            ).mean(dim=1)
            half_norm = 0.5*parameters.square().sum(dim=1)
            objective = raw_bce + (
                float(Config.L2_STATIC_COEFFICIENT)*half_norm
            )
            target_exact = torch.all(
                predictions == targets[None], dim=1
            )
            train_exact = torch.all(
                predictions[:, train_indices]
                == targets[None, train_indices],
                dim=1,
            )
            full_accuracy = (
                predictions == targets[None]
            ).float().mean(dim=1)
        target_np = target_exact.cpu().numpy()
        train_np = train_exact.cpu().numpy()
        raw_np = raw_bce.cpu().numpy()
        norm_np = half_norm.cpu().numpy()
        objective_np = objective.cpu().numpy()
        target_exact_arrays.append(target_np)
        train_exact_arrays.append(train_np)
        raw_bce_arrays.append(raw_np)
        half_norm_arrays.append(norm_np)
        objective_arrays.append(objective_np)
        packed_predictions.append(np.packbits(
            predictions.cpu().numpy(), axis=1, bitorder="little"
        ))
        rows.append({
            "replica": replica,
            "sample_count": len(parameters),
            "target_mass": float(target_np.mean()),
            "train_exact_mass": float(train_np.mean()),
            "raw_bce_mean": float(raw_np.mean()),
            "raw_bce_median": float(np.median(raw_np)),
            "half_norm_mean": float(norm_np.mean()),
            "half_norm_median": float(np.median(norm_np)),
            "objective_mean": float(objective_np.mean()),
            "objective_max": float(objective_np.max()),
            "full_accuracy_mean": float(full_accuracy.mean().item()),
        })
    write_csv(output_dir / f"{artifact_prefix}_by_replica.csv", rows)
    target_exact_np = np.stack(target_exact_arrays)
    train_exact_np = np.stack(train_exact_arrays)
    raw_bce_np = np.stack(raw_bce_arrays)
    half_norm_np = np.stack(half_norm_arrays)
    objective_np = np.stack(objective_arrays)
    packed_np = np.stack(packed_predictions)
    np.savez_compressed(
        output_dir / f"{artifact_prefix}_observables.npz",
        target_exact=target_exact_np,
        train_exact=train_exact_np,
        raw_bce=raw_bce_np,
        half_norm=half_norm_np,
        objective=objective_np,
        packed_full_predictions=packed_np,
        train_indices=np.asarray(condition.train_indices, dtype=np.int64),
    )
    target_masses = target_exact_np.mean(axis=1)
    train_masses = train_exact_np.mean(axis=1)
    full_function_count = len(np.unique(
        packed_np.reshape(-1, packed_np.shape[-1]), axis=0
    ))
    quality_checks = {
        "direct_smc_quality_pass": bool(
            static_summary.get("quality_pass", False)
        ),
        "target_mass_replica_std_within_limit": bool(
            np.std(target_masses)
            <= Config.L2_STATIC_TARGET_MASS_REPLICA_STD_MAX
        ),
        "all_saved_samples_respect_J_threshold": bool(
            objective_np.max()
            <= (
                Config.L2_STATIC_J_THRESHOLD
                + direct_constraint_tolerance(Config.L2_STATIC_J_THRESHOLD)
            )
        ),
    }
    quality_pass = bool(all(quality_checks.values()))
    target_median = float(np.median(target_masses))
    target_min = float(np.min(target_masses))
    pooled_target_mass = float(target_exact_np.mean())
    dynamic_residual = (
        pooled_target_mass-float(independent_dynamic_target_mass)
        if independent_dynamic_target_mass is not None else None
    )
    if not quality_pass:
        verdict = "sampler_unresolved"
    elif (
        independent_dynamic_target_mass is not None
        and independent_dynamic_target_mass >= 0.95
        and target_median >= 0.95
        and target_min >= 0.90
    ):
        verdict = "static_landscape_sufficient_at_dynamic_endpoint"
    elif independent_dynamic_target_mass is None:
        verdict = "static_same_threshold_control_completed"
    elif abs(dynamic_residual) <= 0.10:
        verdict = "static_landscape_consistent_with_dynamic_endpoint"
    elif abs(dynamic_residual) >= 0.20:
        verdict = "static_dynamic_endpoint_mismatch_requires_followup"
    else:
        verdict = "partial_static_explanation"
    protocol = {
        "protocol_version": protocol_version,
        "condition": asdict(condition),
        "network": "8->16->16->1 tanh, 433 Gaussian coordinates",
        "reference_measure": "iid standard Gaussian normalized coordinates",
        "objective": "J = mean_BCE_D + lambda*||theta||^2/2",
        "lambda": Config.L2_STATIC_COEFFICIENT,
        "J_threshold": Config.L2_STATIC_J_THRESHOLD,
        "threshold_frozen_before_control_result": True,
        "independent_dynamic_J_range": independent_dynamic_range,
        "independent_dynamic_target_mass": independent_dynamic_target_mass,
        "sampler": "direct prior-initialized constrained SMC",
        "optimizer_or_anchor_used": False,
        "replicas": Config.DIRECT_REPLICAS,
        "particles_per_replica": Config.DIRECT_PARTICLES_PER_REPLICA,
        "saved_parameters_per_replica": Config.FW_SAVE_PARAMETER_SUBSET,
    }
    protocol_hash = payload_sha256(protocol)
    write_json(output_dir / "protocol.json", {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
    })
    summary = {
        "status": "completed",
        "protocol_sha256": protocol_hash,
        "condition": asdict(condition),
        "lambda": Config.L2_STATIC_COEFFICIENT,
        "J_threshold": Config.L2_STATIC_J_THRESHOLD,
        "static_sampler_summary": static_summary,
        "quality_checks": quality_checks,
        "quality_pass": quality_pass,
        "target_mass_pooled": pooled_target_mass,
        "target_mass_replica_mean": float(target_masses.mean()),
        "target_mass_replica_median": target_median,
        "target_mass_replica_std": float(target_masses.std()),
        "target_mass_replica_min": target_min,
        "target_mass_replica_max": float(target_masses.max()),
        "train_exact_mass_pooled": float(train_exact_np.mean()),
        "train_exact_mass_replica_min": float(train_masses.min()),
        "raw_bce_mean": float(raw_bce_np.mean()),
        "half_norm_mean": float(half_norm_np.mean()),
        "objective_mean": float(objective_np.mean()),
        "objective_max": float(objective_np.max()),
        "unique_complete_function_count": full_function_count,
        "independent_dynamic_target_mass": independent_dynamic_target_mass,
        "static_minus_dynamic_target_mass": dynamic_residual,
        "verdict": verdict,
        "interpretation_boundary": (
            "This directly tests the static low-J function mass of the exact "
            "explicit-L2 objective. Optimizer-specific explanations are only "
            "warranted after a converged static result leaves a substantial "
            "dynamic residual."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    return summary_path


def main() -> None:
    args = parse_args()
    apply_args(args)
    apply_smoke_overrides()
    if Config.MODE == "l2_static_half_lambda_n40":
        Config.L2_STATIC_COEFFICIENT = Config.L2_HALF_COEFFICIENT
        Config.L2_STATIC_J_THRESHOLD = Config.L2_HALF_J_THRESHOLD
        Config.DIRECT_MAX_LEVELS = max(Config.DIRECT_MAX_LEVELS, 2_000)
        summary = run_static_endpoint_n40(
            Config.L2_HALF_RESULT_DIR,
            artifact_prefix="l2_static",
            protocol_version="8bit_n40_explicit_l2_5e5_matched_bce_v2",
            independent_dynamic_range=None,
            independent_dynamic_target_mass=None,
        )
        print(f"L2-STATIC lambda=5e-5 n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包："
                f"{create_archive(Config.L2_HALF_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "l2_static_higher_lambda_n40":
        Config.L2_STATIC_COEFFICIENT = Config.L2_HIGHER_COEFFICIENT
        Config.L2_STATIC_J_THRESHOLD = Config.L2_HIGHER_J_THRESHOLD
        Config.DIRECT_MAX_LEVELS = max(Config.DIRECT_MAX_LEVELS, 2_000)
        summary = run_static_endpoint_n40(
            Config.L2_HIGHER_RESULT_DIR,
            artifact_prefix="l2_static",
            protocol_version="8bit_n40_explicit_l2_2e4_matched_bce_v2",
            independent_dynamic_range=None,
            independent_dynamic_target_mass=None,
        )
        print(f"L2-STATIC lambda=2e-4 n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包："
                f"{create_archive(Config.L2_HIGHER_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "no_wd_static_matched_bce_n40":
        Config.L2_STATIC_COEFFICIENT = 0.0
        Config.L2_STATIC_J_THRESHOLD = Config.NO_WD_MATCHED_BCE_THRESHOLD
        Config.L2_STATIC_TRAIN_COUNT = Config.NO_WD_STATIC_TRAIN_COUNT
        Config.L2_STATIC_REPLICAS = Config.NO_WD_STATIC_REPLICAS
        Config.L2_STATIC_PARTICLES_PER_REPLICA = (
            Config.NO_WD_STATIC_PARTICLES_PER_REPLICA
        )
        Config.L2_STATIC_SAVED_PARAMETERS_PER_REPLICA = (
            Config.NO_WD_STATIC_SAVED_PARAMETERS_PER_REPLICA
        )
        Config.L2_STATIC_TARGET_MASS_REPLICA_STD_MAX = (
            Config.NO_WD_STATIC_TARGET_MASS_REPLICA_STD_MAX
        )
        summary = run_static_endpoint_n40(
            Config.NO_WD_MATCHED_BCE_RESULT_DIR,
            artifact_prefix="no_wd_static",
            protocol_version="8bit_n40_no_wd_static_matched_bce_v1",
            independent_dynamic_range=None,
            independent_dynamic_target_mass=None,
        )
        print(f"NO-WD MATCHED-BCE STATIC n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包："
                f"{create_archive(Config.NO_WD_MATCHED_BCE_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "l2_static_reliable_n40":
        Config.L2_STATIC_COEFFICIENT = 1e-4
        Config.L2_STATIC_J_THRESHOLD = Config.L2_RELIABLE_J_THRESHOLD
        Config.DIRECT_MAX_LEVELS = max(Config.DIRECT_MAX_LEVELS, 2_000)
        summary = run_static_endpoint_n40(
            Config.L2_RELIABLE_RESULT_DIR,
            artifact_prefix="l2_static",
            protocol_version="8bit_n40_explicit_l2_reliable_layer_v1",
            independent_dynamic_range=None,
            independent_dynamic_target_mass=None,
        )
        print(f"L2-STATIC RELIABLE n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包："
                f"{create_archive(Config.L2_RELIABLE_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "no_wd_static_same_j_n40":
        Config.L2_STATIC_COEFFICIENT = 0.0
        Config.L2_STATIC_J_THRESHOLD = Config.NO_WD_STATIC_J_THRESHOLD
        Config.L2_STATIC_TRAIN_COUNT = Config.NO_WD_STATIC_TRAIN_COUNT
        Config.L2_STATIC_REPLICAS = Config.NO_WD_STATIC_REPLICAS
        Config.L2_STATIC_PARTICLES_PER_REPLICA = (
            Config.NO_WD_STATIC_PARTICLES_PER_REPLICA
        )
        Config.L2_STATIC_SAVED_PARAMETERS_PER_REPLICA = (
            Config.NO_WD_STATIC_SAVED_PARAMETERS_PER_REPLICA
        )
        Config.L2_STATIC_TARGET_MASS_REPLICA_STD_MAX = (
            Config.NO_WD_STATIC_TARGET_MASS_REPLICA_STD_MAX
        )
        Config.DIRECT_MAX_LEVELS = max(Config.DIRECT_MAX_LEVELS, 2_000)
        summary = run_static_endpoint_n40(
            Config.NO_WD_STATIC_RESULT_DIR,
            artifact_prefix="no_wd_static",
            protocol_version="8bit_n40_no_wd_static_same_j_v1",
            independent_dynamic_range=None,
            independent_dynamic_target_mass=None,
        )
        print(f"NO-WD STATIC n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包："
                f"{create_archive(Config.NO_WD_STATIC_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "l2_static_n40":
        summary = run_static_endpoint_n40(
            Config.L2_STATIC_RESULT_DIR,
            artifact_prefix="l2_static",
            protocol_version="8bit_n40_explicit_l2_static_endpoint_v2",
            independent_dynamic_range=(0.01856772, 0.01856787),
            independent_dynamic_target_mass=1.0,
        )
        print(f"L2-STATIC n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包：{create_archive(Config.L2_STATIC_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "norm_target_n40":
        summary = run_norm_target_n40(Config.NORM_TARGET_RESULT_DIR)
        print(f"NORM-TARGET n40 COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包：{create_archive(Config.NORM_TARGET_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE in {"explicit_l2_and_n32", "explicit_l2_and_boundary"}:
        summary = run_explicit_l2_and_n32(Config.L2_AND_RESULT_DIR)
        print(
            f"EXPLICIT-L2 AND n={Config.L2_AND_TRAIN_COUNT} COMPLETED: "
            f"{summary}",
            flush=True,
        )
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包：{create_archive(Config.L2_AND_RESULT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "finite_width_direct":
        summary = run_finite_width_direct_smc(
            Config.FINITE_WIDTH_DIRECT_DIR
        )
        print(f"DIRECT FINITE-WIDTH SMC COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                "下载压缩包："
                f"{create_archive(Config.FINITE_WIDTH_DIRECT_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "finite_width":
        summary = run_finite_width_smc(Config.FINITE_WIDTH_DIR)
        print(f"FINITE-WIDTH SMC COMPLETED: {summary}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(
                f"下载压缩包：{create_archive(Config.FINITE_WIDTH_DIR)}",
                flush=True,
            )
        return
    if Config.MODE == "predict":
        manifest = run_prediction(Config.PREDICTION_DIR)
        print(f"PREDICTION FROZEN: {manifest}", flush=True)
        if Config.PACKAGE_RESULTS:
            print(f"下载压缩包：{create_archive(Config.PREDICTION_DIR)}", flush=True)
        return
    if Config.PREDICTION_MANIFEST is None:
        raise ValueError("validate模式必须配置Config.PREDICTION_MANIFEST。")
    run_validation(Config.VALIDATION_DIR, Config.PREDICTION_MANIFEST)
    if Config.PACKAGE_RESULTS:
        print(f"下载压缩包：{create_archive(Config.VALIDATION_DIR)}", flush=True)


if __name__ == "__main__":
    main()
