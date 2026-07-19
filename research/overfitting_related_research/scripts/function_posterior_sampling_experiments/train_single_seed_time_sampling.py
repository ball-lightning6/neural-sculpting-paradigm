# %% cell 1
"""
Single-seed post-plateau time-sampling experiment.

Goal:
1. Train one model until a stricter plateau criterion is reached.
2. Continue training the same model for a long time.
3. Save probe predictions every fixed number of steps.
4. Measure agreement/autocorrelation among time samples, and optionally compare
   them with old multi-seed results.

The script is self-contained. Edit Config directly.
"""

import csv
import copy
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Config:
    # =========================
    # Data and task
    # =========================
    TASK_NAME = "rule30_layer1"
    TASK_TYPE = "jsonl"  # "jsonl" or "random"
    DATASET_PATH = (
        "research/overfitting_related_research/datasets/"
        "ca_rule30_layer1_len30_n30000.jsonl"
    )
    INPUT_KEY = "input"
    OUTPUT_KEY = "output"
    DEDUPLICATE_INPUTS = True

    # Used only when TASK_TYPE="random".
    RANDOM_NUM_SAMPLES = 30000
    RANDOM_INPUT_BITS = 30
    RANDOM_OUTPUT_BITS = 30
    RANDOM_DATA_SEED = 20260716

    # Run several training-set sizes in one launch.
    # Leave empty to use TRAIN_COUNT only.
    # These points match the local multi-seed sweep summary.
    TRAIN_COUNTS = (
        100, 200, 300, 400, 600, 800, 1000,
        1200, 1500,)# 1800, 2200, 3000, 4500, 6000,    )
    TRAIN_COUNT = 800
    MONITOR_COUNT = 3000
    PROBE_COUNT = 5000
    SPLIT_SEED = 20260711

    # =========================
    # Model
    # =========================
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.1

    # =========================
    # Training
    # =========================
    MODEL_SEED = 0
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512
    PREDICT_BATCH_SIZE = 2048
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Plateau search. This is intentionally stricter than the older version.
    EVAL_INTERVAL_STEPS = 100
    MAX_PLATEAU_SEARCH_STEPS = 80000
    TRAIN_EXACT_TARGET = 1.0
    PLATEAU_REQUIRE_TRAIN_FIT = True
    PLATEAU_MIN_STEPS_AFTER_TRAIN_FIT = 5000
    PLATEAU_PATIENCE_EVALS = 20
    PLATEAU_MIN_DELTA = 0.0002
    USE_PROBE_LABEL_FOR_PLATEAU = True
    USE_PREDICTION_STABILITY_FOR_PLATEAU = True
    PREDICTION_STABILITY_THRESHOLD = 0.9995
    PREDICTION_STABILITY_PATIENCE_EVALS = 12
    MAX_STEPS_AFTER_TRAIN_FIT = 20000

    # Rolling-window plateau check: adjacent stability is not enough.
    # The recent monitor/probe bit accuracy ranges must also be small.
    USE_ROLLING_PLATEAU_CHECK = True
    PLATEAU_ROLLING_WINDOW_EVALS = 12
    PLATEAU_MAX_MONITOR_BIT_RANGE = 0.0015
    PLATEAU_MAX_PROBE_BIT_RANGE = 0.0015
    PLATEAU_WINDOW_MIN_MONITOR_PREDICTION_AGREEMENT = 0.998
    PLATEAU_WINDOW_MIN_PROBE_PREDICTION_AGREEMENT = 0.998

    # Time sampling after the detected plateau.
    # It is deliberately long so later analysis can use late windows only.
    POST_PLATEAU_TOTAL_STEPS = 120000
    TIME_SAMPLE_INTERVAL_STEPS = 500
    SAVE_PLATEAU_SAMPLE = True

    # Optional reference directory for old multi-seed results.
    # It should contain predictions.jsonl and probe.jsonl.
    MULTI_SEED_REFERENCE_DIR = ""
    # Template for batch mode. Available fields:
    # {task_name}, {train_count}, {split_seed}
    # Example:
    # "research/overfitting_related_research/results_overfit_ensemble_sweep/{task_name}_n{train_count}_split{split_seed}"
    MULTI_SEED_REFERENCE_DIR_TEMPLATE = ""
    EXCLUDE_REFERENCE_PILOT = True

    # Output
    OUTPUT_ROOT = "research/overfitting_related_research/results_single_seed_time_sampling"
    EXPERIMENT_NAME = ""  # Empty means auto-generated.
    OVERWRITE_OUTPUT = True
    PACKAGE_RESULTS = True
    ZIP_NAME = "single_seed_time_sampling_package.zip"

class MLP(nn.Module):
    def __init__(self, input_bits, output_bits, cfg):
        super().__init__()
        layers = [
            nn.Linear(input_bits, cfg.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(cfg.HIDDEN_SIZE),
        ]
        if cfg.DROPOUT > 0:
            layers.append(nn.Dropout(cfg.DROPOUT))
        for _ in range(cfg.HIDDEN_LAYERS):
            layers.extend([
                nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE),
                nn.GELU(),
                nn.LayerNorm(cfg.HIDDEN_SIZE),
            ])
            if cfg.DROPOUT > 0:
                layers.append(nn.Dropout(cfg.DROPOUT))
        layers.append(nn.Linear(cfg.HIDDEN_SIZE, output_bits))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_bits(value):
    if isinstance(value, str):
        text = value.strip()
        if all(ch in "01" for ch in text):
            return [int(ch) for ch in text]
        normalized = text.replace(",", " ").replace("[", " ").replace("]", " ")
        parts = normalized.split()
        if parts and all(part in {"0", "1"} for part in parts):
            return [int(part) for part in parts]
        raise ValueError(f"cannot parse bit string: {value!r}")
    if isinstance(value, (list, tuple)):
        bits = [int(item) for item in value]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"bit list contains values other than 0/1: {value!r}")
        return bits
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return [int(value)]
    raise TypeError(f"unsupported bit format: {type(value)}")


def bits_to_text(bits):
    return "".join("1" if int(bit) else "0" for bit in bits)


def prediction_to_text(prediction):
    flat = prediction.reshape(-1)
    return "".join("1" if int(bit) else "0" for bit in flat)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_jsonl_dataset(cfg):
    path = Path(cfg.DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    records = []
    seen_outputs = {}
    duplicates_removed = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            x_bits = parse_bits(row[cfg.INPUT_KEY])
            y_bits = parse_bits(row[cfg.OUTPUT_KEY])
            x_text = bits_to_text(x_bits)
            y_text = bits_to_text(y_bits)
            if cfg.DEDUPLICATE_INPUTS and x_text in seen_outputs:
                if seen_outputs[x_text] != y_text:
                    raise ValueError(f"{path} line {line_no}: duplicated input with inconsistent output")
                duplicates_removed += 1
                continue
            seen_outputs[x_text] = y_text
            records.append({
                "input_text": x_text,
                "input": x_bits,
                "output": y_bits,
            })

    if not records:
        raise ValueError(f"empty dataset: {path}")
    meta = {
        "task_type": "jsonl",
        "dataset_path": str(path),
        "duplicates_removed": duplicates_removed,
    }
    return records, meta


def load_random_dataset(cfg):
    rng = random.Random(int(cfg.RANDOM_DATA_SEED))
    records = []
    seen = set()
    while len(records) < cfg.RANDOM_NUM_SAMPLES:
        x_bits = [rng.randrange(2) for _ in range(cfg.RANDOM_INPUT_BITS)]
        x_text = bits_to_text(x_bits)
        if x_text in seen:
            continue
        seen.add(x_text)
        y_bits = [rng.randrange(2) for _ in range(cfg.RANDOM_OUTPUT_BITS)]
        records.append({
            "input_text": x_text,
            "input": x_bits,
            "output": y_bits,
        })
    meta = {
        "task_type": "random",
        "num_samples": cfg.RANDOM_NUM_SAMPLES,
        "input_bits": cfg.RANDOM_INPUT_BITS,
        "output_bits": cfg.RANDOM_OUTPUT_BITS,
        "data_seed": cfg.RANDOM_DATA_SEED,
    }
    return records, meta


def load_dataset(cfg):
    if cfg.TASK_TYPE == "jsonl":
        records, meta = load_jsonl_dataset(cfg)
    elif cfg.TASK_TYPE == "random":
        records, meta = load_random_dataset(cfg)
    else:
        raise ValueError("TASK_TYPE must be jsonl or random")

    input_bits = len(records[0]["input"])
    output_bits = len(records[0]["output"])
    for row in records:
        if len(row["input"]) != input_bits or len(row["output"]) != output_bits:
            raise ValueError("dataset contains inconsistent input/output bit lengths")
    meta["input_bits"] = input_bits
    meta["output_bits"] = output_bits
    meta["record_count"] = len(records)
    return records, meta


def make_split(records, cfg):
    total_needed = cfg.TRAIN_COUNT + cfg.MONITOR_COUNT + cfg.PROBE_COUNT
    if total_needed > len(records):
        raise ValueError(
            f"not enough samples: need {total_needed}, got {len(records)}"
        )
    rng = random.Random(int(cfg.SPLIT_SEED))
    indices = list(range(len(records)))
    rng.shuffle(indices)

    train_indices = indices[:cfg.TRAIN_COUNT]
    monitor_indices = indices[cfg.TRAIN_COUNT:cfg.TRAIN_COUNT + cfg.MONITOR_COUNT]
    probe_indices = indices[
        cfg.TRAIN_COUNT + cfg.MONITOR_COUNT:
        cfg.TRAIN_COUNT + cfg.MONITOR_COUNT + cfg.PROBE_COUNT
    ]

    def pick(selected):
        return [records[i] for i in selected]

    return {
        "train_indices": train_indices,
        "monitor_indices": monitor_indices,
        "probe_indices": probe_indices,
        "train": pick(train_indices),
        "monitor": pick(monitor_indices),
        "probe": pick(probe_indices),
    }


def records_to_arrays(records):
    x = np.asarray([row["input"] for row in records], dtype=np.float32)
    y = np.asarray([row["output"] for row in records], dtype=np.float32)
    return x, y


def make_loader(records, cfg, shuffle=True):
    x, y = records_to_arrays(records)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model, records, cfg):
    x, y = records_to_arrays(records)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(
        dataset,
        batch_size=cfg.PREDICT_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(cfg.DEVICE)
    model.eval()
    total_loss = 0.0
    total_bits = 0
    correct_bits = 0
    exact_correct = 0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="sum")
            pred = (torch.sigmoid(logits) >= 0.5).to(yb.dtype)
            correct = pred.eq(yb)
            total_loss += float(loss.item())
            total_bits += int(yb.numel())
            correct_bits += int(correct.sum().item())
            exact_correct += int(correct.all(dim=1).sum().item())
            total_samples += int(yb.shape[0])
    return {
        "loss": total_loss / max(total_bits, 1),
        "bit_accuracy": correct_bits / max(total_bits, 1),
        "exact_accuracy": exact_correct / max(total_samples, 1),
    }


def predict_bits(model, records, cfg):
    x, _ = records_to_arrays(records)
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(
        dataset,
        batch_size=cfg.PREDICT_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(cfg.DEVICE)
    model.eval()
    chunks = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)
            chunks.append(pred.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def bit_agreement(a, b):
    if a is None or b is None:
        return None
    return float((a == b).mean())


def binary_entropy(p):
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    entropy[(p == 0.0) | (p == 1.0)] = 0.0
    return entropy


def prediction_accuracy(prediction, records):
    targets = np.asarray([row["output"] for row in records], dtype=np.uint8)
    correct = prediction == targets
    return {
        "probe_bit_accuracy": float(correct.mean()),
        "probe_exact_accuracy": float(correct.all(axis=1).mean()),
        "prediction_one_rate": float(prediction.mean()),
    }


def metric_row(stage, step, relative_step, train_metrics, monitor_metrics, probe_metrics, extra=None):
    row = {
        "record_type": "metric",
        "stage": stage,
        "step": int(step),
        "relative_step_after_plateau": int(relative_step),
        "train_loss": float(train_metrics["loss"]),
        "train_bit_accuracy": float(train_metrics["bit_accuracy"]),
        "train_exact_accuracy": float(train_metrics["exact_accuracy"]),
        "monitor_loss": float(monitor_metrics["loss"]),
        "monitor_bit_accuracy": float(monitor_metrics["bit_accuracy"]),
        "monitor_exact_accuracy": float(monitor_metrics["exact_accuracy"]),
        "probe_loss": float(probe_metrics["loss"]),
        "probe_bit_accuracy": float(probe_metrics["bit_accuracy"]),
        "probe_exact_accuracy": float(probe_metrics["exact_accuracy"]),
    }
    if extra:
        row.update(extra)
    return row


def train_one_step(model, optimizer, iterator_state, loader, cfg):
    iterator = iterator_state.get("iterator")
    try:
        xb, yb = next(iterator)
    except (StopIteration, TypeError):
        iterator = iter(loader)
        xb, yb = next(iterator)
    iterator_state["iterator"] = iterator

    device = torch.device(cfg.DEVICE)
    model.train()
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def find_plateau(model, optimizer, loader, split, cfg, paths):
    best_monitor = -1.0
    best_probe = -1.0
    best_monitor_step = 0
    best_probe_step = 0
    label_evals_since_best = 0
    prediction_stable_evals = 0
    previous_monitor_prediction = None
    previous_probe_prediction = None
    train_fit_step = None
    final_row = None
    stop_source = "max_steps"
    iterator_state = {"iterator": iter(loader)}
    recent_window = []

    progress = tqdm(
        range(1, cfg.MAX_PLATEAU_SEARCH_STEPS + 1),
        desc="search plateau",
    )
    for step in progress:
        train_one_step(model, optimizer, iterator_state, loader, cfg)
        if step % cfg.EVAL_INTERVAL_STEPS != 0 and step != cfg.MAX_PLATEAU_SEARCH_STEPS:
            continue

        train_metrics = evaluate(model, split["train"], cfg)
        monitor_metrics = evaluate(model, split["monitor"], cfg)
        probe_metrics = evaluate(model, split["probe"], cfg)

        monitor_prediction_stability = None
        probe_prediction_stability = None
        current_monitor_prediction = None
        current_probe_prediction = None
        need_predictions = (
            cfg.USE_PREDICTION_STABILITY_FOR_PLATEAU
            or cfg.USE_ROLLING_PLATEAU_CHECK
        )
        if need_predictions:
            current_monitor_prediction = predict_bits(model, split["monitor"], cfg)
            current_probe_prediction = predict_bits(model, split["probe"], cfg)
            monitor_prediction_stability = bit_agreement(previous_monitor_prediction, current_monitor_prediction)
            probe_prediction_stability = bit_agreement(previous_probe_prediction, current_probe_prediction)
            previous_monitor_prediction = current_monitor_prediction
            previous_probe_prediction = current_probe_prediction
            if (
                monitor_prediction_stability is not None
                and probe_prediction_stability is not None
                and monitor_prediction_stability >= cfg.PREDICTION_STABILITY_THRESHOLD
                and probe_prediction_stability >= cfg.PREDICTION_STABILITY_THRESHOLD
            ):
                prediction_stable_evals += 1
            else:
                prediction_stable_evals = 0

        recent_window.append({
            "step": int(step),
            "monitor_bit": float(monitor_metrics["bit_accuracy"]),
            "probe_bit": float(probe_metrics["bit_accuracy"]),
            "monitor_exact": float(monitor_metrics["exact_accuracy"]),
            "probe_exact": float(probe_metrics["exact_accuracy"]),
            "monitor_prediction": current_monitor_prediction,
            "probe_prediction": current_probe_prediction,
        })
        if len(recent_window) > cfg.PLATEAU_ROLLING_WINDOW_EVALS:
            recent_window.pop(0)

        rolling_plateau_ok = not cfg.USE_ROLLING_PLATEAU_CHECK
        rolling_monitor_bit_range = None
        rolling_probe_bit_range = None
        rolling_monitor_prediction_agreement = None
        rolling_probe_prediction_agreement = None
        if cfg.USE_ROLLING_PLATEAU_CHECK and len(recent_window) >= cfg.PLATEAU_ROLLING_WINDOW_EVALS:
            monitor_bits = [row["monitor_bit"] for row in recent_window]
            probe_bits = [row["probe_bit"] for row in recent_window]
            rolling_monitor_bit_range = max(monitor_bits) - min(monitor_bits)
            rolling_probe_bit_range = max(probe_bits) - min(probe_bits)
            first = recent_window[0]
            last = recent_window[-1]
            rolling_monitor_prediction_agreement = bit_agreement(
                first["monitor_prediction"], last["monitor_prediction"]
            )
            rolling_probe_prediction_agreement = bit_agreement(
                first["probe_prediction"], last["probe_prediction"]
            )
            rolling_plateau_ok = (
                rolling_monitor_bit_range <= cfg.PLATEAU_MAX_MONITOR_BIT_RANGE
                and rolling_probe_bit_range <= cfg.PLATEAU_MAX_PROBE_BIT_RANGE
                and rolling_monitor_prediction_agreement is not None
                and rolling_probe_prediction_agreement is not None
                and rolling_monitor_prediction_agreement >= cfg.PLATEAU_WINDOW_MIN_MONITOR_PREDICTION_AGREEMENT
                and rolling_probe_prediction_agreement >= cfg.PLATEAU_WINDOW_MIN_PROBE_PREDICTION_AGREEMENT
            )

        if train_fit_step is None and train_metrics["exact_accuracy"] >= cfg.TRAIN_EXACT_TARGET:
            train_fit_step = int(step)

        monitor_improved = monitor_metrics["bit_accuracy"] > best_monitor + cfg.PLATEAU_MIN_DELTA
        probe_improved = probe_metrics["bit_accuracy"] > best_probe + cfg.PLATEAU_MIN_DELTA
        if monitor_improved:
            best_monitor = float(monitor_metrics["bit_accuracy"])
            best_monitor_step = int(step)
        if probe_improved:
            best_probe = float(probe_metrics["bit_accuracy"])
            best_probe_step = int(step)

        label_improved = monitor_improved or (cfg.USE_PROBE_LABEL_FOR_PLATEAU and probe_improved)
        if label_improved:
            label_evals_since_best = 0
        else:
            label_evals_since_best += 1

        final_row = metric_row(
            "plateau_search",
            step,
            0,
            train_metrics,
            monitor_metrics,
            probe_metrics,
            extra={
                "best_monitor_step": int(best_monitor_step),
                "best_monitor_bit_accuracy": float(best_monitor),
                "best_probe_step": int(best_probe_step),
                "best_probe_bit_accuracy": float(best_probe),
                "train_fit_step": train_fit_step,
                "label_evals_since_best": int(label_evals_since_best),
                "prediction_stable_evals": int(prediction_stable_evals),
                "monitor_prediction_stability": monitor_prediction_stability,
                "probe_prediction_stability": probe_prediction_stability,
                "rolling_plateau_ok": bool(rolling_plateau_ok),
                "rolling_monitor_bit_range": rolling_monitor_bit_range,
                "rolling_probe_bit_range": rolling_probe_bit_range,
                "rolling_monitor_prediction_agreement": rolling_monitor_prediction_agreement,
                "rolling_probe_prediction_agreement": rolling_probe_prediction_agreement,
            },
        )
        append_jsonl(paths["history"], final_row)

        progress.set_postfix(
            train=f"{train_metrics['exact_accuracy']:.3f}",
            probe=f"{probe_metrics['bit_accuracy']:.3f}",
            adj=("-" if probe_prediction_stability is None else f"{probe_prediction_stability:.4f}"),
            roll=("Y" if rolling_plateau_ok else "N"),
        )

        fit_ok = (train_fit_step is not None) or (not cfg.PLATEAU_REQUIRE_TRAIN_FIT)
        enough_after_fit = (
            train_fit_step is None
            or step - train_fit_step >= cfg.PLATEAU_MIN_STEPS_AFTER_TRAIN_FIT
        )
        label_patience_ok = label_evals_since_best >= cfg.PLATEAU_PATIENCE_EVALS
        prediction_patience_ok = (
            not cfg.USE_PREDICTION_STABILITY_FOR_PLATEAU
            or prediction_stable_evals >= cfg.PREDICTION_STABILITY_PATIENCE_EVALS
        )
        strict_plateau_ok = label_patience_ok and prediction_patience_ok and rolling_plateau_ok
        post_fit_steps = None if train_fit_step is None else step - train_fit_step
        post_fit_budget_ok = (
            post_fit_steps is not None
            and post_fit_steps >= cfg.MAX_STEPS_AFTER_TRAIN_FIT
        )
        if fit_ok and enough_after_fit and (strict_plateau_ok or post_fit_budget_ok):
            stop_source = (
                "strict_rolling_plateau"
                if strict_plateau_ok
                else "postfit_budget_without_strict_plateau"
            )
            break

    if final_row is None:
        train_metrics = evaluate(model, split["train"], cfg)
        monitor_metrics = evaluate(model, split["monitor"], cfg)
        probe_metrics = evaluate(model, split["probe"], cfg)
        final_row = metric_row(
            "plateau_search",
            cfg.MAX_PLATEAU_SEARCH_STEPS,
            0,
            train_metrics,
            monitor_metrics,
            probe_metrics,
        )

    return {
        "plateau_step": int(final_row["step"]),
        "stop_source": stop_source,
        "train_fit_step": train_fit_step,
        "best_monitor_step": int(best_monitor_step),
        "best_monitor_bit_accuracy": float(best_monitor),
        "best_probe_step": int(best_probe_step),
        "best_probe_bit_accuracy": float(best_probe),
        "final_metric": final_row,
        "iterator_state": iterator_state,
    }

def save_prediction_sample(sample_path, summary_path, stage, sample_index, step, relative_step, prediction, split, cfg):
    acc = prediction_accuracy(prediction, split["probe"])
    record = {
        "record_type": "time_prediction",
        "stage": stage,
        "task_name": cfg.TASK_NAME,
        "split_seed": int(cfg.SPLIT_SEED),
        "model_seed": int(cfg.MODEL_SEED),
        "train_count": int(cfg.TRAIN_COUNT),
        "sample_index": int(sample_index),
        "step": int(step),
        "relative_step_after_plateau": int(relative_step),
        "probe_metrics": {
            "bit_accuracy": acc["probe_bit_accuracy"],
            "exact_accuracy": acc["probe_exact_accuracy"],
        },
        "prediction_one_rate": acc["prediction_one_rate"],
        "prediction_bits": prediction_to_text(prediction),
    }
    append_jsonl(sample_path, record)
    summary = dict(record)
    summary.pop("prediction_bits", None)
    append_jsonl(summary_path, summary)
    return record


def continue_time_sampling(model, optimizer, loader, split, cfg, paths, plateau_info):
    iterator_state = plateau_info["iterator_state"]
    plateau_step = int(plateau_info["plateau_step"])
    sample_index = 0

    if cfg.SAVE_PLATEAU_SAMPLE:
        prediction = predict_bits(model, split["probe"], cfg)
        save_prediction_sample(
            paths["time_predictions"],
            paths["time_sample_summary"],
            "plateau",
            sample_index,
            plateau_step,
            0,
            prediction,
            split,
            cfg,
        )
        sample_index += 1

    total_steps = int(cfg.POST_PLATEAU_TOTAL_STEPS)
    interval = int(cfg.TIME_SAMPLE_INTERVAL_STEPS)
    progress = tqdm(range(1, total_steps + 1), desc="post-plateau time sampling")
    for relative_step in progress:
        train_one_step(model, optimizer, iterator_state, loader, cfg)
        if relative_step % interval != 0 and relative_step != total_steps:
            continue

        absolute_step = plateau_step + relative_step
        train_metrics = evaluate(model, split["train"], cfg)
        monitor_metrics = evaluate(model, split["monitor"], cfg)
        probe_metrics = evaluate(model, split["probe"], cfg)
        append_jsonl(paths["history"], metric_row(
            "time_sampling",
            absolute_step,
            relative_step,
            train_metrics,
            monitor_metrics,
            probe_metrics,
        ))
        prediction = predict_bits(model, split["probe"], cfg)
        save_prediction_sample(
            paths["time_predictions"],
            paths["time_sample_summary"],
            "time_sample",
            sample_index,
            absolute_step,
            relative_step,
            prediction,
            split,
            cfg,
        )
        sample_index += 1
        progress.set_postfix(
            probe=f"{probe_metrics['bit_accuracy']:.4f}",
            exact=f"{probe_metrics['exact_accuracy']:.4f}",
            samples=sample_index,
        )


def decode_prediction(bit_text, probe_count, output_bits):
    text = str(bit_text).strip()
    expected = probe_count * output_bits
    if len(text) != expected:
        raise ValueError(f"prediction_bits length is {len(text)}, expected {expected}")
    raw = np.frombuffer(text.encode("ascii"), dtype=np.uint8) - ord("0")
    return raw.reshape(probe_count, output_bits)


def load_prediction_matrix(path, probe_count, output_bits, stage_filter=None, exclude_pilot=True):
    rows = read_jsonl(path)
    matrices = []
    kept_rows = []
    for row in rows:
        if exclude_pilot and row.get("source") == "pilot":
            continue
        if "prediction_bits" not in row:
            continue
        if stage_filter is not None and row.get("stage") not in stage_filter:
            continue
        matrices.append(decode_prediction(row["prediction_bits"], probe_count, output_bits))
        kept_rows.append(row)
    if not matrices:
        return None, []
    return np.stack(matrices, axis=0), kept_rows


def pairwise_stats(predictions, rows, targets, stage_name):
    pairwise_rows = []
    count = int(predictions.shape[0])
    for i in range(count):
        for j in range(i + 1, count):
            same = predictions[i] == predictions[j]
            row_i = rows[i]
            row_j = rows[j]
            pairwise_rows.append({
                "record_type": "pairwise_time_stat",
                "stage": stage_name,
                "sample_index_a": row_i.get("sample_index"),
                "sample_index_b": row_j.get("sample_index"),
                "step_a": row_i.get("step"),
                "step_b": row_j.get("step"),
                "relative_step_a": row_i.get("relative_step_after_plateau"),
                "relative_step_b": row_j.get("relative_step_after_plateau"),
                "lag_samples": int(abs(int(row_j.get("sample_index", j)) - int(row_i.get("sample_index", i)))),
                "lag_steps": int(abs(int(row_j.get("step", 0)) - int(row_i.get("step", 0)))),
                "prediction_bit_agreement": float(same.mean()),
                "prediction_bit_hamming_distance": float((~same).mean()),
                "prediction_exact_agreement": float(same.all(axis=1).mean()),
            })
    return pairwise_rows


def summarize_matrix(predictions, targets):
    p_one = predictions.mean(axis=0)
    majority = (p_one >= 0.5).astype(np.uint8)
    majority_error = majority != targets
    entropy = binary_entropy(p_one)
    model_correct = predictions == targets[None, :, :]

    pairwise_values = []
    for i in range(predictions.shape[0]):
        for j in range(i + 1, predictions.shape[0]):
            pairwise_values.append(float((predictions[i] == predictions[j]).mean()))

    return {
        "model_count": int(predictions.shape[0]),
        "probe_count": int(predictions.shape[1]),
        "output_bits": int(predictions.shape[2]),
        "mean_model_bit_accuracy": float(model_correct.mean()),
        "mean_model_exact_accuracy": float(model_correct.all(axis=2).mean()),
        "majority_vote_bit_accuracy": float(1.0 - majority_error.mean()),
        "majority_vote_exact_accuracy": float((~majority_error).all(axis=1).mean()),
        "mean_pairwise_prediction_bit_agreement": (
            float(np.mean(pairwise_values)) if pairwise_values else None
        ),
        "mean_prediction_entropy_bits": float(entropy.mean()),
        "unanimously_same_prediction_bit_fraction": float(((p_one == 0.0) | (p_one == 1.0)).mean()),
        "prediction_one_rate": float(predictions.mean()),
    }


def aggregate_by_lag(pairwise_rows):
    grouped = {}
    for row in pairwise_rows:
        lag = int(row["lag_samples"])
        grouped.setdefault(lag, []).append(row)
    lag_rows = []
    for lag in sorted(grouped):
        rows = grouped[lag]
        lag_rows.append({
            "record_type": "time_lag_stat",
            "lag_samples": int(lag),
            "lag_steps": int(np.mean([row["lag_steps"] for row in rows])),
            "pair_count": int(len(rows)),
            "mean_prediction_bit_agreement": float(np.mean([row["prediction_bit_agreement"] for row in rows])),
            "mean_prediction_bit_hamming_distance": float(np.mean([row["prediction_bit_hamming_distance"] for row in rows])),
            "mean_prediction_exact_agreement": float(np.mean([row["prediction_exact_agreement"] for row in rows])),
        })
    return lag_rows


def load_probe_targets(path):
    rows = read_jsonl(path)
    targets = []
    for row in rows:
        value = row.get("target")
        if value is None:
            value = row.get("output")
        targets.append(parse_bits(value))
    return np.asarray(targets, dtype=np.uint8)


def resolve_reference_dir(cfg):
    if str(cfg.MULTI_SEED_REFERENCE_DIR).strip():
        return str(cfg.MULTI_SEED_REFERENCE_DIR).format(
            task_name=cfg.TASK_NAME,
            train_count=cfg.TRAIN_COUNT,
            split_seed=cfg.SPLIT_SEED,
        )
    if str(cfg.MULTI_SEED_REFERENCE_DIR_TEMPLATE).strip():
        return str(cfg.MULTI_SEED_REFERENCE_DIR_TEMPLATE).format(
            task_name=cfg.TASK_NAME,
            train_count=cfg.TRAIN_COUNT,
            split_seed=cfg.SPLIT_SEED,
        )
    return ""


def compare_with_reference(paths, cfg, targets, time_predictions, time_rows):
    ref_dir_text = resolve_reference_dir(cfg)
    ref_dir = Path(ref_dir_text)
    if not ref_dir_text.strip():
        return None
    if not ref_dir.exists():
        return {"error": f"reference dir not found: {ref_dir}"}

    ref_pred_path = ref_dir / "predictions.jsonl"
    ref_probe_path = ref_dir / "probe.jsonl"
    if not ref_pred_path.exists():
        return {"error": f"reference predictions.jsonl not found: {ref_pred_path}"}

    probe_count, output_bits = targets.shape
    reference_predictions, reference_rows = load_prediction_matrix(
        ref_pred_path,
        probe_count,
        output_bits,
        stage_filter={"plateau"},
        exclude_pilot=cfg.EXCLUDE_REFERENCE_PILOT,
    )
    if reference_predictions is None:
        return {"error": "reference predictions.jsonl has no compatible plateau predictions"}

    probe_warning = None
    if ref_probe_path.exists():
        try:
            ref_targets = load_probe_targets(ref_probe_path)
            if ref_targets.shape != targets.shape or not np.array_equal(ref_targets, targets):
                probe_warning = "reference probe targets differ from current probe targets"
        except Exception as exc:
            probe_warning = f"failed to compare reference probe: {exc!r}"
    else:
        probe_warning = "reference probe.jsonl not found; shape-only comparison"

    cross_values = []
    for i in range(time_predictions.shape[0]):
        for j in range(reference_predictions.shape[0]):
            cross_values.append(float((time_predictions[i] == reference_predictions[j]).mean()))

    comparison = {
        "record_type": "cross_reference_comparison",
        "reference_dir": str(ref_dir),
        "reference_model_count": int(reference_predictions.shape[0]),
        "time_sample_count": int(time_predictions.shape[0]),
        "probe_warning": probe_warning,
        "time_summary": summarize_matrix(time_predictions, targets),
        "reference_summary": summarize_matrix(reference_predictions, targets),
        "cross_time_reference_bit_agreement": float(np.mean(cross_values)),
        "cross_time_reference_bit_hamming_distance": float(1.0 - np.mean(cross_values)),
    }
    write_json(paths["reference_comparison"], comparison)
    return comparison


def analyze_time_samples(paths, cfg):
    probe_targets = load_probe_targets(paths["probe"])
    probe_count, output_bits = probe_targets.shape
    predictions, rows = load_prediction_matrix(
        paths["time_predictions"],
        probe_count,
        output_bits,
        stage_filter={"plateau", "time_sample"},
        exclude_pilot=False,
    )
    if predictions is None or predictions.shape[0] < 2:
        raise ValueError("fewer than 2 time samples; cannot analyze pairwise structure")

    pairwise_rows = pairwise_stats(predictions, rows, probe_targets, "time_sampling")
    lag_rows = aggregate_by_lag(pairwise_rows)
    summary = summarize_matrix(predictions, probe_targets)
    summary.update({
        "record_type": "time_sampling_summary",
        "task_name": cfg.TASK_NAME,
        "split_seed": int(cfg.SPLIT_SEED),
        "model_seed": int(cfg.MODEL_SEED),
        "train_count": int(cfg.TRAIN_COUNT),
        "sample_count": int(predictions.shape[0]),
        "min_step": int(min(row.get("step", 0) for row in rows)),
        "max_step": int(max(row.get("step", 0) for row in rows)),
        "sample_interval_steps": int(cfg.TIME_SAMPLE_INTERVAL_STEPS),
    })
    write_json(paths["time_summary"], summary)
    write_jsonl(paths["time_pairwise"], pairwise_rows)
    write_jsonl(paths["time_lag"], lag_rows)
    write_csv(paths["time_lag_csv"], lag_rows)

    comparison = compare_with_reference(paths, cfg, probe_targets, predictions, rows)
    return summary, comparison


def write_split_files(paths, split):
    for key in ("train", "monitor", "probe"):
        rows = []
        for row in split[key]:
            rows.append({
                "input": row["input_text"],
                "output": [int(bit) for bit in row["output"]],
                "target": [int(bit) for bit in row["output"]],
            })
        write_jsonl(paths[key], rows)


def prepare_paths(cfg):
    if cfg.EXPERIMENT_NAME:
        experiment_name = cfg.EXPERIMENT_NAME
    else:
        experiment_name = (
            f"{cfg.TASK_NAME}_n{cfg.TRAIN_COUNT}_split{cfg.SPLIT_SEED}_"
            f"seed{cfg.MODEL_SEED}_time"
        )
    root = Path(cfg.OUTPUT_ROOT) / experiment_name
    if root.exists() and cfg.OVERWRITE_OUTPUT:
        # Only clean files generated by this script inside the experiment folder.
        for name in [
            "metadata.json",
            "train.jsonl",
            "monitor.jsonl",
            "probe.jsonl",
            "training_history.jsonl",
            "time_predictions.jsonl",
            "time_sample_summary.jsonl",
            "time_pairwise_statistics.jsonl",
            "time_lag_statistics.jsonl",
            "time_lag_statistics.csv",
            "time_sampling_summary.json",
            "reference_comparison.json",
            cfg.ZIP_NAME,
        ]:
            path = root / name
            if path.exists() and path.is_file():
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "train": root / "train.jsonl",
        "monitor": root / "monitor.jsonl",
        "probe": root / "probe.jsonl",
        "history": root / "training_history.jsonl",
        "time_predictions": root / "time_predictions.jsonl",
        "time_sample_summary": root / "time_sample_summary.jsonl",
        "time_pairwise": root / "time_pairwise_statistics.jsonl",
        "time_lag": root / "time_lag_statistics.jsonl",
        "time_lag_csv": root / "time_lag_statistics.csv",
        "time_summary": root / "time_sampling_summary.json",
        "reference_comparison": root / "reference_comparison.json",
        "zip": root / cfg.ZIP_NAME,
    }


def save_metadata(paths, cfg, dataset_meta, split, plateau_info=None):
    metadata = {
        "record_type": "metadata",
        "task_name": cfg.TASK_NAME,
        "task_type": cfg.TASK_TYPE,
        "dataset": dataset_meta,
        "train_count": int(cfg.TRAIN_COUNT),
        "monitor_count": int(cfg.MONITOR_COUNT),
        "probe_count": int(cfg.PROBE_COUNT),
        "split_seed": int(cfg.SPLIT_SEED),
        "model_seed": int(cfg.MODEL_SEED),
        "train_indices": split["train_indices"],
        "monitor_indices": split["monitor_indices"],
        "probe_indices": split["probe_indices"],
        "model": {
            "hidden_size": int(cfg.HIDDEN_SIZE),
            "hidden_layers": int(cfg.HIDDEN_LAYERS),
            "dropout": float(cfg.DROPOUT),
        },
        "training": {
            "learning_rate": float(cfg.LEARNING_RATE),
            "weight_decay": float(cfg.WEIGHT_DECAY),
            "batch_size": int(cfg.BATCH_SIZE),
            "eval_interval_steps": int(cfg.EVAL_INTERVAL_STEPS),
            "max_plateau_search_steps": int(cfg.MAX_PLATEAU_SEARCH_STEPS),
            "post_plateau_total_steps": int(cfg.POST_PLATEAU_TOTAL_STEPS),
            "time_sample_interval_steps": int(cfg.TIME_SAMPLE_INTERVAL_STEPS),
            "device": cfg.DEVICE,
        },
        "reference": {
            "multi_seed_reference_dir": resolve_reference_dir(cfg),
            "multi_seed_reference_dir_template": cfg.MULTI_SEED_REFERENCE_DIR_TEMPLATE,
            "exclude_reference_pilot": bool(cfg.EXCLUDE_REFERENCE_PILOT),
        },
    }
    if plateau_info is not None:
        metadata["plateau"] = {
            "plateau_step": int(plateau_info["plateau_step"]),
            "stop_source": plateau_info["stop_source"],
            "train_fit_step": plateau_info["train_fit_step"],
            "best_monitor_step": int(plateau_info["best_monitor_step"]),
            "best_monitor_bit_accuracy": float(plateau_info["best_monitor_bit_accuracy"]),
            "best_probe_step": int(plateau_info["best_probe_step"]),
            "best_probe_bit_accuracy": float(plateau_info["best_probe_bit_accuracy"]),
        }
    write_json(paths["metadata"], metadata)


def package_results(paths, cfg):
    if not cfg.PACKAGE_RESULTS:
        return None
    root = paths["root"]
    include_names = {
        "metadata.json",
        "train.jsonl",
        "monitor.jsonl",
        "probe.jsonl",
        "training_history.jsonl",
        "time_predictions.jsonl",
        "time_sample_summary.jsonl",
        "time_pairwise_statistics.jsonl",
        "time_lag_statistics.jsonl",
        "time_lag_statistics.csv",
        "time_sampling_summary.json",
        "reference_comparison.json",
    }
    with zipfile.ZipFile(paths["zip"], "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(include_names):
            path = root / name
            if path.exists():
                zf.write(path, arcname=name)
    return paths["zip"]


def get_train_counts(cfg):
    counts = tuple(getattr(cfg, "TRAIN_COUNTS", ()) or ())
    if not counts:
        return (int(cfg.TRAIN_COUNT),)
    return tuple(int(count) for count in counts)


def make_run_config(base_cfg, train_count, multi_count_mode):
    cfg = copy.copy(base_cfg)
    cfg.TRAIN_COUNT = int(train_count)
    if multi_count_mode and str(base_cfg.EXPERIMENT_NAME).strip():
        cfg.EXPERIMENT_NAME = f"{base_cfg.EXPERIMENT_NAME}_n{int(train_count)}"
    return cfg


def run_once(cfg):
    paths = prepare_paths(cfg)
    set_seed(cfg.MODEL_SEED)

    records, dataset_meta = load_dataset(cfg)
    split = make_split(records, cfg)
    write_split_files(paths, split)
    save_metadata(paths, cfg, dataset_meta, split)

    input_bits = int(dataset_meta["input_bits"])
    output_bits = int(dataset_meta["output_bits"])
    device = torch.device(cfg.DEVICE)
    model = MLP(input_bits, output_bits, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    loader = make_loader(split["train"], cfg, shuffle=True)

    print(f"device: {cfg.DEVICE}")
    print(f"experiment dir: {paths['root']}")
    print(f"task: {cfg.TASK_NAME}, train={cfg.TRAIN_COUNT}, monitor={cfg.MONITOR_COUNT}, probe={cfg.PROBE_COUNT}")

    plateau_info = find_plateau(model, optimizer, loader, split, cfg, paths)
    save_metadata(paths, cfg, dataset_meta, split, plateau_info)
    print(
        "plateau search finished: "
        f"step={plateau_info['plateau_step']}, "
        f"stop={plateau_info['stop_source']}, "
        f"best_probe={plateau_info['best_probe_bit_accuracy']:.4f}"
    )

    continue_time_sampling(model, optimizer, loader, split, cfg, paths, plateau_info)
    summary, comparison = analyze_time_samples(paths, cfg)
    zip_path = package_results(paths, cfg)

    print("time sampling analysis finished:")
    print(f"  samples={summary['sample_count']}")
    print(f"  time pairwise agreement={summary['mean_pairwise_prediction_bit_agreement']}")
    print(f"  entropy={summary['mean_prediction_entropy_bits']}")
    if comparison:
        if comparison.get("error"):
            print(f"  reference comparison skipped: {comparison['error']}")
        else:
            print(f"  reference pairwise agreement={comparison['reference_summary']['mean_pairwise_prediction_bit_agreement']}")
            print(f"  cross time/reference agreement={comparison['cross_time_reference_bit_agreement']}")
            if comparison.get("probe_warning"):
                print(f"  warning: {comparison['probe_warning']}")
    if zip_path:
        print(f"results packaged: {zip_path}")

    result = {
        "record_type": "single_seed_time_sampling_run",
        "status": "ok",
        "task_name": cfg.TASK_NAME,
        "train_count": int(cfg.TRAIN_COUNT),
        "split_seed": int(cfg.SPLIT_SEED),
        "model_seed": int(cfg.MODEL_SEED),
        "experiment_dir": str(paths["root"]),
        "zip_path": str(zip_path) if zip_path else None,
        "plateau_step": int(plateau_info["plateau_step"]),
        "plateau_stop_source": plateau_info["stop_source"],
        "train_fit_step": plateau_info["train_fit_step"],
        "sample_count": int(summary["sample_count"]),
        "time_pairwise_agreement": summary["mean_pairwise_prediction_bit_agreement"],
        "time_prediction_entropy_bits": summary["mean_prediction_entropy_bits"],
        "mean_model_bit_accuracy": summary["mean_model_bit_accuracy"],
        "mean_model_exact_accuracy": summary["mean_model_exact_accuracy"],
    }
    if comparison and not comparison.get("error"):
        result.update({
            "reference_model_count": comparison["reference_model_count"],
            "reference_pairwise_agreement": comparison["reference_summary"]["mean_pairwise_prediction_bit_agreement"],
            "cross_time_reference_bit_agreement": comparison["cross_time_reference_bit_agreement"],
            "probe_warning": comparison.get("probe_warning"),
        })
    elif comparison and comparison.get("error"):
        result["reference_error"] = comparison["error"]
    return result


def main():
    base_cfg = Config()
    train_counts = get_train_counts(base_cfg)
    multi_count_mode = len(train_counts) > 1
    output_root = Path(base_cfg.OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_summary_path = output_root / "single_seed_time_sampling_batch_summary.jsonl"
    batch_summary_csv_path = output_root / "single_seed_time_sampling_batch_summary.csv"

    rows = []
    print(f"will run {len(train_counts)} train counts: {train_counts}")
    for index, train_count in enumerate(train_counts, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(train_counts)}] train_count={train_count}")
        cfg = make_run_config(base_cfg, train_count, multi_count_mode)
        started = time.time()
        try:
            row = run_once(cfg)
        except Exception as exc:
            row = {
                "record_type": "single_seed_time_sampling_run",
                "status": "error",
                "task_name": cfg.TASK_NAME,
                "train_count": int(cfg.TRAIN_COUNT),
                "split_seed": int(cfg.SPLIT_SEED),
                "model_seed": int(cfg.MODEL_SEED),
                "error": repr(exc),
            }
            print(f"run failed: {exc!r}")
        row["elapsed_seconds"] = float(time.time() - started)
        rows.append(row)
        write_jsonl(batch_summary_path, rows)
        write_csv(batch_summary_csv_path, rows)
        print(
            f"run status: {row.get('status')}, "
            f"elapsed {row['elapsed_seconds'] / 60:.2f} min"
        )

    print("\nall done.")
    print(f"batch JSONL: {batch_summary_path}")
    print(f"batch CSV: {batch_summary_csv_path}")

if __name__ == "__main__":
    main()


# %% cell 2


