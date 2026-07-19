# %% cell 1
"""
批量扫描“过拟合态输出稳定性”实验。

夜间挂机版特性：
- 每个数据量使用独立子进程运行，避免 CUDA 内存残留污染后续实验。
- 单个实验失败、超时或中断时，总控进程会记录状态并继续下一个实验。
- 已完成实验会自动跳过，适合断点续跑。
- 每轮结束都会立刻写入 sweep_index.jsonl，方便随时查看进度。

使用时通常只需要改 Config 里的 DATASET_SPECS、SPLIT_SEEDS、
MODEL_SEEDS 和训练超参数。
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


class Config:
    # =========================
    # 总控开关
    # =========================
    RUN_TRAIN = True
    RUN_ANALYSIS = True
    STOP_ON_ERROR = False
    # True 时不训练，只生成很小的假数据和假预测，用来调试总控流程。
    MOCK_RUN = False
    MOCK_PROBE_COUNT = 5
    MOCK_OUTPUT_BITS = 6
    MOCK_SLEEP_SECONDS = 0.2

    # 公开版默认在当前 Python 进程中顺序运行，避免对子进程路径做额外假设。
    # 如果之后需要命令行批量跑 .py 文件，可按需改回 True。
    RUN_EACH_JOB_IN_SUBPROCESS = False
    # 单个数据点最长运行时间。None 表示不限制。
    # 4090 上当前配置通常几十分钟一组；这里给 4 小时兜底。
    JOB_TIMEOUT_SECONDS = 4 * 60 * 60
    SKIP_ALREADY_COMPLETE = True

    # 断点续跑：同名实验已完成的 seed 会由训练脚本自动跳过。
    RESUME_EXISTING_OUTPUT = True
    OVERWRITE_EXISTING_OUTPUT = False

    # =========================
    # 输出位置
    # =========================
    OUTPUT_ROOT = "research/overfitting_related_research/results_overfit_ensemble_sweep"
    SWEEP_INDEX_NAME = "sweep_index.jsonl"

    # =========================
    # 数据集和扫描范围
    # =========================
    DATASET_SPECS = [
        {
            "name": "rule30_layer1",
            "path": (
                "research/overfitting_related_research/datasets/"
                "ca_rule30_layer1_len30_n300000.jsonl"
            ),
            "train_counts": (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75)
        },
        # {
        #     "name": "rule30_layer1",
        #     "path": "research/overfitting_related_research/datasets/ca_rule30_layer1_len30_n300000.jsonl",
        #     "train_counts": list(range(300, 9001, 300)),
        # },
    ]

    # 改这里可以对同一个数据量重复更换训练/monitor/probe 划分。
    SPLIT_SEEDS = [20260709]

    # =========================
    # 数据划分
    # =========================
    MONITOR_COUNT = 3000
    # None 表示 probe 使用训练集和 monitor 集之外的全部样本。
    PROBE_COUNT = None
    DEDUPLICATE_INPUTS = True

    # =========================
    # 模型和训练
    # =========================
    MODEL_SEEDS = tuple(range(20))
    PILOT_SEED = 10000

    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.1

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512

    # None 表示每个实验先跑 pilot 自动找平台期。
    # 如果你想省时间，也可以手动填一个整数，比如 15000。
    COMMON_TRAIN_STEPS = None
    MAX_PILOT_STEPS = 30000
    EVAL_INTERVAL_STEPS = 100

    MIN_PILOT_STEPS = 3000
    MIN_TRAIN_EXACT_FOR_PLATEAU = 1.0
    PLATEAU_METRIC = "monitor_bit_accuracy"
    PLATEAU_WINDOW = 20
    PLATEAU_REQUIRED_WINDOWS = 3
    PLATEAU_MAX_MEAN_SHIFT = 0.001
    PLATEAU_MAX_SLOPE_PER_EVAL = 0.00005

    PILOT_ONLY = False
    VARY_DATA_ORDER_BY_MODEL_SEED = False
    DATA_ORDER_SEED = 314159
    SAVE_MODELS = False

    # 默认从本脚本所在目录导入 train/analyze 脚本；必要时可手动指定目录。
    MODULE_DIR = None


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_single_jsonl(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def count_prediction_records(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") == "prediction":
                count += 1
    return count


def make_experiment_name(dataset_name, train_count, split_seed):
    return f"{dataset_name}_n{train_count}_split{split_seed}"


def experiment_dir_for(output_root, dataset_name, train_count, split_seed):
    return Path(output_root) / make_experiment_name(dataset_name, train_count, split_seed)


def is_experiment_complete(experiment_dir, expected_runs):
    summary_path = experiment_dir / "summary.jsonl"
    predictions_path = experiment_dir / "predictions.jsonl"
    return (
        summary_path.exists()
        and summary_path.stat().st_size > 0
        and count_prediction_records(predictions_path) >= expected_runs
    )


def patch_train_config(train_module, payload):
    cfg = payload["config"]
    dataset_spec = payload["dataset_spec"]
    train_count = payload["train_count"]
    split_seed = payload["split_seed"]

    train_cfg = train_module.Config

    train_cfg.DATASET_PATH = dataset_spec["path"]
    train_cfg.INPUT_KEY = dataset_spec.get("input_key", "input")
    train_cfg.OUTPUT_KEY = dataset_spec.get("output_key", "output")

    train_cfg.TRAIN_COUNT = int(train_count)
    train_cfg.MONITOR_COUNT = int(cfg["MONITOR_COUNT"])
    train_cfg.PROBE_COUNT = cfg["PROBE_COUNT"]
    train_cfg.SPLIT_SEED = int(split_seed)
    train_cfg.DEDUPLICATE_INPUTS = bool(cfg["DEDUPLICATE_INPUTS"])

    train_cfg.HIDDEN_SIZE = int(cfg["HIDDEN_SIZE"])
    train_cfg.HIDDEN_LAYERS = int(cfg["HIDDEN_LAYERS"])
    train_cfg.DROPOUT = float(cfg["DROPOUT"])

    train_cfg.MODEL_SEEDS = tuple(cfg["MODEL_SEEDS"])
    train_cfg.PILOT_SEED = int(cfg["PILOT_SEED"])
    train_cfg.LEARNING_RATE = float(cfg["LEARNING_RATE"])
    train_cfg.WEIGHT_DECAY = float(cfg["WEIGHT_DECAY"])
    train_cfg.BATCH_SIZE = int(cfg["BATCH_SIZE"])

    train_cfg.COMMON_TRAIN_STEPS = dataset_spec.get(
        "common_train_steps",
        cfg["COMMON_TRAIN_STEPS"],
    )
    train_cfg.MAX_PILOT_STEPS = int(cfg["MAX_PILOT_STEPS"])
    train_cfg.EVAL_INTERVAL_STEPS = int(cfg["EVAL_INTERVAL_STEPS"])

    train_cfg.MIN_PILOT_STEPS = int(cfg["MIN_PILOT_STEPS"])
    train_cfg.MIN_TRAIN_EXACT_FOR_PLATEAU = float(
        cfg["MIN_TRAIN_EXACT_FOR_PLATEAU"]
    )
    train_cfg.PLATEAU_METRIC = cfg["PLATEAU_METRIC"]
    train_cfg.PLATEAU_WINDOW = int(cfg["PLATEAU_WINDOW"])
    train_cfg.PLATEAU_REQUIRED_WINDOWS = int(cfg["PLATEAU_REQUIRED_WINDOWS"])
    train_cfg.PLATEAU_MAX_MEAN_SHIFT = float(cfg["PLATEAU_MAX_MEAN_SHIFT"])
    train_cfg.PLATEAU_MAX_SLOPE_PER_EVAL = float(
        cfg["PLATEAU_MAX_SLOPE_PER_EVAL"]
    )

    train_cfg.PILOT_ONLY = bool(cfg["PILOT_ONLY"])
    train_cfg.VARY_DATA_ORDER_BY_MODEL_SEED = bool(
        cfg["VARY_DATA_ORDER_BY_MODEL_SEED"]
    )
    train_cfg.DATA_ORDER_SEED = int(cfg["DATA_ORDER_SEED"])

    train_cfg.EXPERIMENT_NAME = payload["experiment_name"]
    train_cfg.OUTPUT_ROOT = cfg["OUTPUT_ROOT"]
    train_cfg.SAVE_MODELS = bool(cfg["SAVE_MODELS"])
    train_cfg.RESUME_EXISTING_OUTPUT = bool(cfg["RESUME_EXISTING_OUTPUT"])
    train_cfg.OVERWRITE_EXISTING_OUTPUT = bool(cfg["OVERWRITE_EXISTING_OUTPUT"])

    return Path(train_cfg.OUTPUT_ROOT) / train_cfg.EXPERIMENT_NAME


def default_module_dir():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def import_module_from_dir(module_name, module_dir):
    module_dir = Path(module_dir)
    module_path = module_dir / f"{module_name}.py"
    if not module_path.exists():
        raise ModuleNotFoundError(
            f"找不到 {module_name}.py。"
            f"当前查找目录：{module_dir}"
        )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_local_module(module_name, cfg):
    module_dir = cfg.get("MODULE_DIR") or str(default_module_dir())
    try:
        return import_module_from_dir(module_name, module_dir)
    except ModuleNotFoundError:
        # 兜底：如果用户本来就在正确目录运行，也允许普通 import。
        return importlib.import_module(module_name)


def bit_string_from_int(value, width):
    return format(value % (2 ** width), f"0{width}b")


def make_mock_prediction(target_bits, model_seed, train_count):
    chars = list(target_bits)
    for index, char in enumerate(chars):
        # 制造一点稳定错误和一点 seed 相关错误，用来测试分析指标。
        stable_error = (index + train_count) % 17 == 0
        seed_error = (index * 7 + model_seed * 11 + train_count) % 29 == 0
        if stable_error or seed_error:
            chars[index] = "0" if char == "1" else "1"
    return "".join(chars)


def write_mock_experiment(payload):
    cfg = payload["config"]
    if not str(cfg["OUTPUT_ROOT"]).endswith("_mock"):
        cfg["OUTPUT_ROOT"] = str(cfg["OUTPUT_ROOT"]) + "_mock"
    experiment_dir = Path(cfg["OUTPUT_ROOT"]) / payload["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    probe_count = int(cfg["MOCK_PROBE_COUNT"])
    output_bits = int(cfg["MOCK_OUTPUT_BITS"])
    train_count = int(payload["train_count"])

    metadata = {
        "record_type": "metadata",
        "experiment_name": payload["experiment_name"],
        "train_count": train_count,
        "monitor_count": int(cfg["MONITOR_COUNT"]),
        "probe_count": probe_count,
        "input_bits": output_bits,
        "output_bits": output_bits,
        "split_seed": int(payload["split_seed"]),
        "model_seeds": list(cfg["MODEL_SEEDS"]),
        "mock_run": True,
    }
    (experiment_dir / "metadata.jsonl").write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stop_step = {
        "record_type": "stop_step",
        "source": "mock",
        "common_train_steps": 123,
    }
    (experiment_dir / "stop_step.jsonl").write_text(
        json.dumps(stop_step, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    probe_targets = []
    with (experiment_dir / "probe.jsonl").open("w", encoding="utf-8") as f:
        for offset in range(probe_count):
            input_bits = bit_string_from_int(offset + train_count, output_bits)
            target_bits = bit_string_from_int((offset * 5 + train_count), output_bits)
            probe_targets.append(target_bits)
            record = {
                "probe_offset": offset,
                "source_index": offset,
                "input": input_bits,
                "target": target_bits,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    flat_target = "".join(probe_targets)
    with (experiment_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for model_seed in cfg["MODEL_SEEDS"]:
            prediction_bits = make_mock_prediction(
                flat_target,
                int(model_seed),
                train_count,
            )
            record = {
                "record_type": "prediction",
                "model_seed": int(model_seed),
                "train_steps": 123,
                "train_metrics": {
                    "loss": 0.0,
                    "bit_accuracy": 1.0,
                    "exact_accuracy": 1.0,
                },
                "monitor_metrics": {
                    "loss": 0.1,
                    "bit_accuracy": 0.9,
                    "exact_accuracy": 0.5,
                },
                "probe_metrics": {
                    "loss": 0.1,
                    "bit_accuracy": None,
                    "exact_accuracy": None,
                },
                "probe_count": probe_count,
                "output_bits": output_bits,
                "prediction_bits": prediction_bits,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    (experiment_dir / "training_history.jsonl").write_text("", encoding="utf-8")
    if float(cfg["MOCK_SLEEP_SECONDS"]) > 0:
        time.sleep(float(cfg["MOCK_SLEEP_SECONDS"]))

    return experiment_dir


def run_analysis(analyze_module, experiment_dir):
    analyze_module.Config.EXPERIMENT_DIR = str(experiment_dir)
    analyze_module.main()


def collect_brief_summary(experiment_dir):
    summary = load_single_jsonl(experiment_dir / "summary.jsonl")
    stop_step = load_single_jsonl(experiment_dir / "stop_step.jsonl")

    if summary is None:
        return {}

    brief = {
        "model_count": summary.get("model_count"),
        "probe_count": summary.get("probe_count"),
        "mean_probe_bit_accuracy": summary.get("mean_probe_bit_accuracy"),
        "mean_probe_exact_accuracy": summary.get("mean_probe_exact_accuracy"),
        "majority_vote_bit_accuracy": summary.get("majority_vote_bit_accuracy"),
        "majority_vote_exact_accuracy": summary.get("majority_vote_exact_accuracy"),
        "mean_prediction_bit_agreement": summary.get(
            "mean_prediction_bit_agreement"
        ),
        "unanimously_same_prediction_bit_fraction": summary.get(
            "unanimously_same_prediction_bit_fraction"
        ),
        "mean_prediction_entropy_bits": summary.get(
            "mean_prediction_entropy_bits"
        ),
        "mean_pairwise_prediction_bit_hamming_distance": summary.get(
            "mean_pairwise_prediction_bit_hamming_distance"
        ),
        "mean_pairwise_joint_error_lift": summary.get(
            "mean_pairwise_joint_error_lift"
        ),
        "mean_pairwise_error_phi_correlation": summary.get(
            "mean_pairwise_error_phi_correlation"
        ),
    }

    if stop_step is not None:
        brief["common_train_steps"] = stop_step.get("common_train_steps")
        brief["stop_source"] = stop_step.get("source")

    return brief


def config_to_payload_dict(cfg):
    keys = [
        "RUN_TRAIN",
        "RUN_ANALYSIS",
        "MOCK_RUN",
        "MOCK_PROBE_COUNT",
        "MOCK_OUTPUT_BITS",
        "MOCK_SLEEP_SECONDS",
        "RESUME_EXISTING_OUTPUT",
        "OVERWRITE_EXISTING_OUTPUT",
        "OUTPUT_ROOT",
        "MONITOR_COUNT",
        "PROBE_COUNT",
        "DEDUPLICATE_INPUTS",
        "MODEL_SEEDS",
        "PILOT_SEED",
        "HIDDEN_SIZE",
        "HIDDEN_LAYERS",
        "DROPOUT",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "BATCH_SIZE",
        "COMMON_TRAIN_STEPS",
        "MAX_PILOT_STEPS",
        "EVAL_INTERVAL_STEPS",
        "MIN_PILOT_STEPS",
        "MIN_TRAIN_EXACT_FOR_PLATEAU",
        "PLATEAU_METRIC",
        "PLATEAU_WINDOW",
        "PLATEAU_REQUIRED_WINDOWS",
        "PLATEAU_MAX_MEAN_SHIFT",
        "PLATEAU_MAX_SLOPE_PER_EVAL",
        "PILOT_ONLY",
        "VARY_DATA_ORDER_BY_MODEL_SEED",
        "DATA_ORDER_SEED",
        "SAVE_MODELS",
        "MODULE_DIR",
    ]
    result = {}
    for key in keys:
        value = getattr(cfg, key)
        if isinstance(value, tuple):
            value = list(value)
        result[key] = value
    return result


def make_payload(cfg, dataset_spec, train_count, split_seed):
    experiment_name = make_experiment_name(
        dataset_spec["name"],
        train_count,
        split_seed,
    )
    return {
        "config": config_to_payload_dict(cfg),
        "dataset_spec": dataset_spec,
        "train_count": int(train_count),
        "split_seed": int(split_seed),
        "experiment_name": experiment_name,
    }


def run_one_job(payload):
    cfg = payload["config"]

    if cfg["MOCK_RUN"]:
        experiment_dir = write_mock_experiment(payload)
    else:
        train_module = load_local_module("train_ca_overfit_ensemble", cfg)
        experiment_dir = patch_train_config(train_module, payload)

    if cfg["RUN_TRAIN"] and not cfg["MOCK_RUN"]:
        train_module.main()

    expected_runs = len(cfg["MODEL_SEEDS"])
    completed_runs = count_prediction_records(experiment_dir / "predictions.jsonl")

    if completed_runs < expected_runs:
        return {
            "status": "incomplete",
            "experiment_dir": str(experiment_dir),
            "completed_runs": completed_runs,
            "expected_runs": expected_runs,
        }

    if cfg["RUN_ANALYSIS"]:
        analyze_module = load_local_module("analyze_ca_overfit_ensemble", cfg)
        run_analysis(analyze_module, experiment_dir)

    return {
        "status": "ok",
        "experiment_dir": str(experiment_dir),
        "completed_runs": completed_runs,
        "expected_runs": expected_runs,
    }


def run_child_from_config(config_path):
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    result = run_one_job(payload)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "ok" else 2


def run_job_subprocess(payload, output_root, timeout_seconds):
    job_config_dir = Path(output_root) / "_job_configs"
    job_config_dir.mkdir(parents=True, exist_ok=True)
    job_config_path = job_config_dir / f"{payload['experiment_name']}.json"
    job_config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one-job",
        str(job_config_path),
    ]
    return subprocess.run(
        command,
        cwd=str(Path.cwd()),
        env=env,
        timeout=timeout_seconds,
        text=True,
        capture_output=True,
    )


def main():
    cfg = Config()
    if cfg.MOCK_RUN and not str(cfg.OUTPUT_ROOT).endswith("_mock"):
        cfg.OUTPUT_ROOT = str(cfg.OUTPUT_ROOT) + "_mock"
        print(f"MOCK_RUN=True，自动使用 mock 输出目录：{cfg.OUTPUT_ROOT}", flush=True)
    if cfg.RUN_EACH_JOB_IN_SUBPROCESS and "__file__" not in globals():
        cfg.RUN_EACH_JOB_IN_SUBPROCESS = False
        print("未检测到脚本文件路径，自动关闭子进程模式。", flush=True)

    output_root = Path(cfg.OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_index_path = output_root / cfg.SWEEP_INDEX_NAME

    total_jobs = sum(
        len(spec["train_counts"]) * len(cfg.SPLIT_SEEDS)
        for spec in cfg.DATASET_SPECS
    )
    job_index = 0

    print(f"准备运行 {total_jobs} 个 sweep 实验。", flush=True)
    print(f"结果目录：{output_root}", flush=True)

    for dataset_spec in cfg.DATASET_SPECS:
        for split_seed in cfg.SPLIT_SEEDS:
            for train_count in dataset_spec["train_counts"]:
                job_index += 1
                payload = make_payload(cfg, dataset_spec, train_count, split_seed)
                experiment_name = payload["experiment_name"]
                experiment_dir = experiment_dir_for(
                    cfg.OUTPUT_ROOT,
                    dataset_spec["name"],
                    train_count,
                    split_seed,
                )

                print("\n" + "=" * 80, flush=True)
                print(f"[{job_index}/{total_jobs}] {experiment_name}", flush=True)

                start_time = time.time()
                status = "ok"
                error_message = None

                try:
                    if cfg.SKIP_ALREADY_COMPLETE and is_experiment_complete(
                        experiment_dir,
                        len(cfg.MODEL_SEEDS),
                    ):
                        status = "skipped_complete"
                        print("检测到已完成实验，跳过。", flush=True)
                    elif cfg.MOCK_RUN:
                        result = run_one_job(payload)
                        status = result["status"]
                    elif cfg.RUN_EACH_JOB_IN_SUBPROCESS:
                        completed = run_job_subprocess(
                            payload,
                            cfg.OUTPUT_ROOT,
                            cfg.JOB_TIMEOUT_SECONDS,
                        )
                        experiment_dir.mkdir(parents=True, exist_ok=True)
                        (experiment_dir / "subprocess_stdout.log").write_text(
                            completed.stdout or "",
                            encoding="utf-8",
                        )
                        (experiment_dir / "subprocess_stderr.log").write_text(
                            completed.stderr or "",
                            encoding="utf-8",
                        )
                        if completed.returncode == 2:
                            status = "incomplete"
                            error_message = "subprocess_reported_incomplete"
                        elif completed.returncode != 0:
                            status = "error"
                            error_message = (
                                f"subprocess_returncode={completed.returncode}"
                            )
                            if completed.stderr:
                                print("子进程 stderr 最后 20 行：", flush=True)
                                print(
                                    "\n".join(completed.stderr.splitlines()[-20:]),
                                    flush=True,
                                )
                            elif completed.stdout:
                                print("子进程 stdout 最后 20 行：", flush=True)
                                print(
                                    "\n".join(completed.stdout.splitlines()[-20:]),
                                    flush=True,
                                )
                        elif completed.stdout:
                            print(
                                "\n".join(completed.stdout.splitlines()[-10:]),
                                flush=True,
                            )
                    else:
                        result = run_one_job(payload)
                        status = result["status"]

                except subprocess.TimeoutExpired as exc:
                    status = "timeout"
                    error_message = repr(exc)
                    print(f"本轮超时，继续下一个实验：{error_message}", flush=True)
                except Exception as exc:
                    status = "error"
                    error_message = repr(exc)
                    traceback.print_exc()
                    if cfg.STOP_ON_ERROR:
                        raise

                elapsed_seconds = time.time() - start_time
                completed_runs = count_prediction_records(
                    experiment_dir / "predictions.jsonl"
                )
                expected_runs = len(cfg.MODEL_SEEDS)

                if status == "ok" and completed_runs < expected_runs:
                    status = "incomplete"

                record = {
                    "record_type": "sweep_job",
                    "status": status,
                    "dataset_name": dataset_spec["name"],
                    "dataset_path": dataset_spec["path"],
                    "train_count": int(train_count),
                    "split_seed": int(split_seed),
                    "experiment_name": experiment_name,
                    "experiment_dir": str(experiment_dir),
                    "completed_runs": completed_runs,
                    "expected_runs": expected_runs,
                    "elapsed_seconds": elapsed_seconds,
                    "error": error_message,
                }

                if experiment_dir.exists() and status not in {"error", "timeout"}:
                    record.update(collect_brief_summary(experiment_dir))

                append_jsonl(sweep_index_path, record)

                print(
                    f"本轮状态：{status}，"
                    f"完成 seed：{completed_runs}/{expected_runs}，"
                    f"耗时 {elapsed_seconds / 60:.2f} 分钟",
                    flush=True,
                )
                if status in {"ok", "skipped_complete"}:
                    print(
                        "摘要："
                        f" bit={record.get('mean_probe_bit_accuracy')},"
                        f" exact={record.get('mean_probe_exact_accuracy')},"
                        f" stability={record.get('mean_prediction_bit_agreement')},"
                        f" phi={record.get('mean_pairwise_error_phi_correlation')}",
                        flush=True,
                    )

    print("\n全部 sweep 结束。", flush=True)
    print(f"索引文件：{sweep_index_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run-one-job":
        raise SystemExit(run_child_from_config(sys.argv[2]))
    main()


# %% cell 2


