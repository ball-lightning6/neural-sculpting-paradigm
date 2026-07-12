"""
任务难度 x 数据量的 probe 一致性实验，plateau 停止版。

核心逻辑：
1. 每个任务、每个训练样本数、每个 split seed 先跑一个 pilot。
2. pilot 在训练集已经拟合后，继续观察 monitor 集 bit accuracy。
3. 当 monitor 指标连续若干次没有明显提升，就认为进入平台期。
4. 同一个任务组合下的所有 model seeds 都训练到同一个 plateau step。
5. 最终只在 probe 集上统计准确率、跨 seed 一致性、熵、共同错误结构等指标。

这样避免把“训练集拟合后固定多跑 3000 步”的随机标签逻辑误用于有真实规则的任务。
"""

import csv
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
    # 任务配置
    # =========================
    # task_type:
    # - "jsonl": 从 dataset_path 读取 {"input": ..., "output": ...}
    # - "random": 生成固定随机映射，作为高复杂度负对照
    TASKS = [
        {
            "name": "random_bits",
            "task_type": "random",
            "num_samples": 30000,
            "input_bits": 30,
            "output_bits": 30,
            "data_seed": 20260711,
            "difficulty_label": "fixed random",
            "difficulty_order": 0,
        },
        {
            "name": "rule30_layer1",
            "task_type": "jsonl",
            "dataset_path": (
                "research/overfitting_related_research/datasets/"
                "ca_rule30_layer1_len30_n30000.jsonl"
            ),
            "difficulty_label": "rule30 layer1",
            "difficulty_order": 1,
        },
        {
            "name": "rule30_layer2",
            "task_type": "jsonl",
            "dataset_path": (
                "research/overfitting_related_research/datasets/"
                "ca_rule30_layer2_len30_n30000.jsonl"
            ),
            "difficulty_label": "rule30 layer2",
            "difficulty_order": 2,
        },
        {
            "name": "rule30_layer3",
            "task_type": "jsonl",
            "dataset_path": (
                "research/overfitting_related_research/datasets/"
                "ca_rule30_layer3_len30_n30000.jsonl"
            ),
            "difficulty_label": "rule30 layer3",
            "difficulty_order": 3,
        },
    ]

    INPUT_KEY = "input"
    OUTPUT_KEY = "output"
    DEDUPLICATE_INPUTS = True

    # 可以先只放一个数，例如 (800,)。后面做二维相图时再改成多个。
    TRAIN_COUNTS = (100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 1800, 2200, 3000, 4500, 6000)
    SPLIT_SEEDS = (20260711,)

    # monitor 用来找平台期；probe 只用于最终统计。
    MONITOR_COUNT = 3000
    PROBE_COUNT = 5000

    # =========================
    # 模型
    # =========================
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.1

    # =========================
    # 训练
    # =========================
    MODEL_SEEDS = tuple(range(20))
    PILOT_SEED = 10000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EVAL_INTERVAL_STEPS = 100
    MAX_STEPS = 30000
    TRAIN_EXACT_TARGET = 1.0

    # 平台期判断：
    # 1. 训练集拟合后，monitor/probe label 指标连续若干次没有明显提升。
    # 2. monitor/probe 上相邻 checkpoint 的预测本身也基本不再变化。
    #
    # 对随机标签任务，label 指标天然在 0.5 附近平台，所以预测稳定性会成为主要停机条件。
    # 对 rule30 layer2/layer3 这类慢任务，probe/monitor 只要还在涨或预测还在变，就会继续训练。
    PLATEAU_PATIENCE_EVALS = 8
    PLATEAU_MIN_DELTA = 0.001
    PLATEAU_REQUIRE_TRAIN_FIT = True
    PLATEAU_MIN_STEPS_AFTER_TRAIN_FIT = 1000
    USE_PROBE_LABEL_FOR_PLATEAU = True
    USE_PREDICTION_STABILITY_FOR_PLATEAU = True
    PREDICTION_STABILITY_THRESHOLD = 0.999
    PREDICTION_STABILITY_PATIENCE_EVALS = 5
    # 保险丝：label 已经平台，但预测稳定性因为 random/噪声任务一直达不到时，
    # 不让 pilot 无限等。random 任务用较短预算；普通规则任务用较长预算。
    RANDOM_MAX_STEPS_AFTER_TRAIN_FIT = 3000
    PLATEAU_MAX_STEPS_AFTER_TRAIN_FIT = 10000

    VARY_DATA_ORDER_BY_MODEL_SEED = False
    DATA_ORDER_SEED = 314159

    # =========================
    # 输出
    # =========================
    OUTPUT_ROOT = "research/overfitting_related_research/results"
    SWEEP_SUMMARY_NAME = "sweep_summary.jsonl"
    SWEEP_INDEX_NAME = "sweep_index.jsonl"
    SWEEP_CSV_NAME = "sweep_summary.csv"
    HTML_NAME = "task_difficulty_plateau.html"
    ZIP_NAME = "task_difficulty_plateau_package.zip"

    PACKAGE_RAW_PREDICTIONS = False
    RESUME_EXISTING_OUTPUT = True
    OVERWRITE_EXISTING_OUTPUT = True


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
        raise ValueError(f"无法解析 bit 字符串：{value!r}")
    if isinstance(value, (list, tuple)):
        bits = [int(item) for item in value]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"bit 列表中存在非 0/1 值：{value!r}")
        return bits
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return [int(value)]
    raise TypeError(f"不支持的 bit 格式：{type(value)}")


def bits_to_text(bits):
    return "".join("1" if int(bit) else "0" for bit in bits)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_jsonl_dataset(task, cfg):
    path = Path(task["dataset_path"])
    if not path.exists():
        raise FileNotFoundError(f"找不到数据集：{path}")

    records = []
    seen_outputs = {}
    duplicate_count = 0
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
                    raise ValueError(f"{path} 第 {line_no} 行输入重复但输出不一致")
                duplicate_count += 1
                continue
            seen_outputs[x_text] = y_text
            records.append((x_text, x_bits, y_bits))

    if not records:
        raise ValueError(f"数据集为空：{path}")
    return dataset_from_records(records, {
        "dataset_path": str(path),
        "duplicates_removed": duplicate_count,
        "task_type": "jsonl",
    })


def load_random_dataset(task):
    rng = random.Random(int(task.get("data_seed", 0)))
    num_samples = int(task.get("num_samples", 30000))
    input_bits = int(task.get("input_bits", 30))
    output_bits = int(task.get("output_bits", input_bits))
    records = []
    seen = set()
    while len(records) < num_samples:
        x_bits = [rng.randrange(2) for _ in range(input_bits)]
        x_text = bits_to_text(x_bits)
        if x_text in seen:
            continue
        seen.add(x_text)
        y_bits = [rng.randrange(2) for _ in range(output_bits)]
        records.append((x_text, x_bits, y_bits))
    return dataset_from_records(records, {
        "dataset_path": None,
        "duplicates_removed": 0,
        "task_type": "random",
        "data_seed": int(task.get("data_seed", 0)),
    })


def dataset_from_records(records, extra_meta):
    input_bits = len(records[0][1])
    output_bits = len(records[0][2])
    if any(len(item[1]) != input_bits for item in records):
        raise ValueError("输入 bit 长度不一致")
    if any(len(item[2]) != output_bits for item in records):
        raise ValueError("输出 bit 长度不一致")
    inputs = torch.tensor([item[1] for item in records], dtype=torch.float32)
    targets = torch.tensor([item[2] for item in records], dtype=torch.float32)
    input_texts = [item[0] for item in records]
    metadata = dict(extra_meta)
    metadata.update({
        "unique_samples": len(records),
        "input_bits": input_bits,
        "output_bits": output_bits,
    })
    return inputs, targets, input_texts, metadata


def load_dataset(task, cfg):
    task_type = task.get("task_type", "jsonl")
    if task_type == "random":
        return load_random_dataset(task)
    return load_jsonl_dataset(task, cfg)


def make_split(inputs, targets, input_texts, train_count, monitor_count, probe_count, split_seed):
    total = len(inputs)
    train_count = int(train_count)
    monitor_count = int(monitor_count)
    if train_count + monitor_count >= total:
        raise ValueError(
            f"train_count + monitor_count 不能大于等于总样本数："
            f"{train_count} + {monitor_count} >= {total}"
        )
    remaining = total - train_count - monitor_count
    actual_probe_count = remaining if probe_count is None else min(int(probe_count), remaining)

    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(total, generator=generator)
    train_idx = indices[:train_count]
    monitor_idx = indices[train_count:train_count + monitor_count]
    probe_idx = indices[train_count + monitor_count:train_count + monitor_count + actual_probe_count]

    return {
        "train_x": inputs[train_idx],
        "train_y": targets[train_idx],
        "monitor_x": inputs[monitor_idx],
        "monitor_y": targets[monitor_idx],
        "probe_x": inputs[probe_idx],
        "probe_y": targets[probe_idx],
        "train_indices": train_idx.tolist(),
        "monitor_indices": monitor_idx.tolist(),
        "probe_indices": probe_idx.tolist(),
        "train_inputs": [input_texts[i] for i in train_idx.tolist()],
        "monitor_inputs": [input_texts[i] for i in monitor_idx.tolist()],
        "probe_inputs": [input_texts[i] for i in probe_idx.tolist()],
    }


def make_train_loader(x, y, cfg, model_seed):
    shuffle_seed = int(model_seed) if cfg.VARY_DATA_ORDER_BY_MODEL_SEED else int(cfg.DATA_ORDER_SEED)
    generator = torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )


def evaluate(model, x, y, device, batch_size=4096):
    model.eval()
    total_loss = 0.0
    total_bits = 0
    correct_bits = 0
    exact_samples = 0
    total_samples = 0
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, reduction="sum")
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct = pred.eq(batch_y)
            total_loss += float(loss.item())
            total_bits += int(batch_y.numel())
            correct_bits += int(correct.sum().item())
            exact_samples += int(correct.all(dim=1).sum().item())
            total_samples += int(batch_y.shape[0])
    return {
        "loss": total_loss / max(total_bits, 1),
        "bit_accuracy": correct_bits / max(total_bits, 1),
        "exact_accuracy": exact_samples / max(total_samples, 1),
    }


def predict_bits(model, x, device, batch_size=4096):
    model.eval()
    chunks = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            pred = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()
            chunks.append(pred)
    array = np.concatenate(chunks, axis=0)
    return "".join(str(int(bit)) for bit in array.reshape(-1))


def bit_string_agreement(a, b):
    if a is None or b is None:
        return None
    if len(a) != len(b):
        raise ValueError("两个预测 bit 串长度不一致，无法计算稳定性。")
    if not a:
        return None
    same = sum(ch_a == ch_b for ch_a, ch_b in zip(a, b))
    return same / len(a)


def metric_record(
    task,
    split,
    model_seed,
    step,
    train_metrics,
    monitor_metrics,
    probe_metrics,
    source,
    extra=None,
):
    record = {
        "record_type": "train_eval",
        "task_name": task["name"],
        "split_seed": int(split["split_seed"]),
        "model_seed": int(model_seed),
        "step": int(step),
        "source": source,
        "train_loss": train_metrics["loss"],
        "train_bit_accuracy": train_metrics["bit_accuracy"],
        "train_exact_accuracy": train_metrics["exact_accuracy"],
        "monitor_loss": monitor_metrics["loss"],
        "monitor_bit_accuracy": monitor_metrics["bit_accuracy"],
        "monitor_exact_accuracy": monitor_metrics["exact_accuracy"],
        "probe_loss": probe_metrics["loss"],
        "probe_bit_accuracy": probe_metrics["bit_accuracy"],
        "probe_exact_accuracy": probe_metrics["exact_accuracy"],
    }
    if extra:
        record.update(extra)
    return record


def run_training(task, cfg, split, model_seed, output_dir, target_steps=None, source="ensemble"):
    set_seed(int(model_seed))
    device = torch.device(cfg.DEVICE)
    input_bits = int(split["train_x"].shape[1])
    output_bits = int(split["train_y"].shape[1])
    model = MLP(input_bits, output_bits, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    loader = make_train_loader(split["train_x"], split["train_y"], cfg, model_seed)
    iterator = iter(loader)
    history_path = output_dir / "training_history.jsonl"

    max_steps = int(cfg.MAX_STEPS if target_steps is None else target_steps)
    best_monitor = -1.0
    best_probe = -1.0
    best_step = 0
    best_probe_step = 0
    label_evals_since_best = 0
    prediction_stable_evals = 0
    previous_monitor_prediction = None
    previous_probe_prediction = None
    train_fit_step = None
    stop_source = "fixed_steps" if target_steps is not None else "max_steps"
    final_record = None

    progress = tqdm(
        range(1, max_steps + 1),
        desc=f"{task['name']}, n={split['train_count']}, split={split['split_seed']}, seed={model_seed}",
        leave=False,
    )
    for step in progress:
        model.train()
        try:
            batch_x, batch_y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_x, batch_y = next(iterator)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

        if step % cfg.EVAL_INTERVAL_STEPS != 0 and step != max_steps:
            continue

        train_metrics = evaluate(model, split["train_x"], split["train_y"], device)
        monitor_metrics = evaluate(model, split["monitor_x"], split["monitor_y"], device)
        probe_metrics = evaluate(model, split["probe_x"], split["probe_y"], device)

        monitor_prediction_stability = None
        probe_prediction_stability = None
        if target_steps is None and cfg.USE_PREDICTION_STABILITY_FOR_PLATEAU:
            current_monitor_prediction = predict_bits(model, split["monitor_x"], device)
            current_probe_prediction = predict_bits(model, split["probe_x"], device)
            monitor_prediction_stability = bit_string_agreement(
                previous_monitor_prediction,
                current_monitor_prediction,
            )
            probe_prediction_stability = bit_string_agreement(
                previous_probe_prediction,
                current_probe_prediction,
            )
            previous_monitor_prediction = current_monitor_prediction
            previous_probe_prediction = current_probe_prediction

            prediction_stable = (
                monitor_prediction_stability is not None
                and probe_prediction_stability is not None
                and monitor_prediction_stability >= cfg.PREDICTION_STABILITY_THRESHOLD
                and probe_prediction_stability >= cfg.PREDICTION_STABILITY_THRESHOLD
            )
            if prediction_stable:
                prediction_stable_evals += 1
            else:
                prediction_stable_evals = 0

        if train_fit_step is None and train_metrics["exact_accuracy"] >= cfg.TRAIN_EXACT_TARGET:
            train_fit_step = int(step)

        monitor_improved = monitor_metrics["bit_accuracy"] > best_monitor + cfg.PLATEAU_MIN_DELTA
        probe_improved = probe_metrics["bit_accuracy"] > best_probe + cfg.PLATEAU_MIN_DELTA
        if monitor_improved:
            best_monitor = monitor_metrics["bit_accuracy"]
            best_step = int(step)
        if probe_improved:
            best_probe = probe_metrics["bit_accuracy"]
            best_probe_step = int(step)

        label_improved = monitor_improved or (
            cfg.USE_PROBE_LABEL_FOR_PLATEAU and probe_improved
        )
        if label_improved:
            label_evals_since_best = 0
        else:
            label_evals_since_best += 1

        final_record = metric_record(
            task, split, model_seed, step,
            train_metrics, monitor_metrics, probe_metrics, source,
            extra={
                "best_monitor_step": int(best_step),
                "best_monitor_bit_accuracy": float(best_monitor),
                "best_probe_step": int(best_probe_step),
                "best_probe_bit_accuracy": float(best_probe),
                "label_evals_since_best": int(label_evals_since_best),
                "prediction_stable_evals": int(prediction_stable_evals),
                "monitor_prediction_stability": monitor_prediction_stability,
                "probe_prediction_stability": probe_prediction_stability,
            },
        )
        append_jsonl(history_path, final_record)
        progress.set_postfix(
            train=f"{train_metrics['exact_accuracy']:.3f}",
            monitor=f"{monitor_metrics['bit_accuracy']:.3f}",
            probe=f"{probe_metrics['bit_accuracy']:.3f}",
            stable=(
                "-"
                if probe_prediction_stability is None
                else f"{probe_prediction_stability:.4f}"
            ),
        )

        if target_steps is None:
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
            post_fit_steps = None if train_fit_step is None else step - train_fit_step
            max_post_fit_steps = (
                cfg.RANDOM_MAX_STEPS_AFTER_TRAIN_FIT
                if task.get("task_type") == "random"
                else cfg.PLATEAU_MAX_STEPS_AFTER_TRAIN_FIT
            )
            post_fit_budget_ok = (
                post_fit_steps is not None
                and post_fit_steps >= max_post_fit_steps
            )
            if (
                fit_ok
                and enough_after_fit
                and label_patience_ok
                and (prediction_patience_ok or post_fit_budget_ok)
            ):
                stop_source = (
                    "label_and_prediction_plateau"
                    if prediction_patience_ok
                    else "label_plateau_postfit_budget"
                )
                break

    if final_record is None:
        train_metrics = evaluate(model, split["train_x"], split["train_y"], device)
        monitor_metrics = evaluate(model, split["monitor_x"], split["monitor_y"], device)
        probe_metrics = evaluate(model, split["probe_x"], split["probe_y"], device)
        final_record = metric_record(
            task, split, model_seed, max_steps,
            train_metrics, monitor_metrics, probe_metrics, source,
        )

    prediction_bits = predict_bits(model, split["probe_x"], device)
    return {
        "record_type": "prediction",
        "stage": "plateau",
        "task_name": task["name"],
        "split_seed": int(split["split_seed"]),
        "model_seed": int(model_seed),
        "train_count": int(split["train_count"]),
        "train_steps": int(final_record["step"]),
        "stop_source": stop_source,
        "best_monitor_step": int(best_step),
        "best_monitor_bit_accuracy": float(best_monitor),
        "best_probe_step": int(best_probe_step),
        "best_probe_bit_accuracy": float(best_probe),
        "train_fit_step": train_fit_step,
        "label_evals_since_best": final_record.get("label_evals_since_best"),
        "prediction_stable_evals": final_record.get("prediction_stable_evals"),
        "monitor_prediction_stability": final_record.get("monitor_prediction_stability"),
        "probe_prediction_stability": final_record.get("probe_prediction_stability"),
        "train_metrics": {
            "loss": final_record["train_loss"],
            "bit_accuracy": final_record["train_bit_accuracy"],
            "exact_accuracy": final_record["train_exact_accuracy"],
        },
        "monitor_metrics": {
            "loss": final_record["monitor_loss"],
            "bit_accuracy": final_record["monitor_bit_accuracy"],
            "exact_accuracy": final_record["monitor_exact_accuracy"],
        },
        "probe_metrics": {
            "loss": final_record["probe_loss"],
            "bit_accuracy": final_record["probe_bit_accuracy"],
            "exact_accuracy": final_record["probe_exact_accuracy"],
        },
        "prediction_bits": prediction_bits,
    }


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def binary_entropy(probabilities):
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    entropy[(probabilities == 0) | (probabilities == 1)] = 0.0
    return entropy


def phi_correlation(error_a, error_b):
    a = error_a.reshape(-1).astype(np.float64)
    b = error_b.reshape(-1).astype(np.float64)
    mean_a = a.mean()
    mean_b = b.mean()
    denominator = math.sqrt(mean_a * (1 - mean_a) * mean_b * (1 - mean_b))
    if denominator == 0:
        return None
    return float(((a * b).mean() - mean_a * mean_b) / denominator)


def error_state_kappa(error_a, error_b):
    a = error_a.reshape(-1)
    b = error_b.reshape(-1)
    observed = float((a == b).mean())
    p_a = float(a.mean())
    p_b = float(b.mean())
    expected = (1 - p_a) * (1 - p_b) + p_a * p_b
    if expected >= 1:
        return None
    return float((observed - expected) / (1 - expected))


def decode_prediction(bit_text, probe_count, output_bits):
    expected = probe_count * output_bits
    bit_text = str(bit_text).strip()
    if len(bit_text) != expected:
        raise ValueError(f"prediction_bits 长度为 {len(bit_text)}，预期 {expected}")
    raw = np.frombuffer(bit_text.encode("ascii"), dtype=np.uint8) - ord("0")
    return raw.reshape(probe_count, output_bits)


def summarize_predictions(task, metadata, probe_targets, prediction_records):
    selected = [row for row in prediction_records if row.get("stage") == "plateau"]
    if len(selected) < 2:
        return None, [], []

    probe_count, output_bits = probe_targets.shape
    predictions = np.stack([
        decode_prediction(row["prediction_bits"], probe_count, output_bits)
        for row in selected
    ], axis=0)
    targets = probe_targets.astype(np.uint8)
    errors = predictions != targets[None, :, :]

    run_stats = []
    for i, row in enumerate(selected):
        correct = ~errors[i]
        run_stats.append({
            "record_type": "run_stat",
            "stage": "plateau",
            "task_name": task["name"],
            "split_seed": metadata["split_seed"],
            "train_count": metadata["train_count"],
            "model_seed": row["model_seed"],
            "train_steps": row["train_steps"],
            "train_fit_step": row.get("train_fit_step"),
            "probe_bit_accuracy": float(correct.mean()),
            "probe_exact_accuracy": float(correct.all(axis=1).mean()),
            "prediction_one_rate": float(predictions[i].mean()),
        })

    pairwise = []
    model_count = predictions.shape[0]
    for i in range(model_count):
        for j in range(i + 1, model_count):
            pred_same = predictions[i] == predictions[j]
            err_a = errors[i]
            err_b = errors[j]
            err_any = err_a | err_b
            err_both = err_a & err_b
            error_rate_a = float(err_a.mean())
            error_rate_b = float(err_b.mean())
            expected_joint = error_rate_a * error_rate_b
            joint = float(err_both.mean())
            pairwise.append({
                "record_type": "pairwise_stat",
                "stage": "plateau",
                "task_name": task["name"],
                "split_seed": metadata["split_seed"],
                "train_count": metadata["train_count"],
                "model_seed_a": selected[i]["model_seed"],
                "model_seed_b": selected[j]["model_seed"],
                "prediction_bit_agreement": float(pred_same.mean()),
                "prediction_bit_hamming_distance": float((~pred_same).mean()),
                "prediction_exact_agreement": float(pred_same.all(axis=1).mean()),
                "prediction_exact_disagreement": float((~pred_same).any(axis=1).mean()),
                "error_rate_a": error_rate_a,
                "error_rate_b": error_rate_b,
                "joint_error_rate": joint,
                "joint_error_lift": (
                    joint / expected_joint if expected_joint > 0 else None
                ),
                "error_jaccard": safe_ratio(int(err_both.sum()), int(err_any.sum())),
                "error_phi_correlation": phi_correlation(err_a, err_b),
                "error_state_cohen_kappa": error_state_kappa(err_a, err_b),
            })

    prob_one = predictions.mean(axis=0)
    majority_fraction = np.maximum(prob_one, 1 - prob_one)
    majority_pred = (prob_one >= 0.5).astype(np.uint8)
    tied_vote = prob_one == 0.5
    majority_error = majority_pred != targets
    entropy = binary_entropy(prob_one)
    unanimous = (prob_one == 0) | (prob_one == 1)
    unanimously_correct = unanimous & (~majority_error)
    unanimously_wrong = unanimous & majority_error
    mixed_error_state = (errors.sum(axis=0) > 0) & (errors.sum(axis=0) < model_count)

    def pairwise_mean(key):
        values = [row[key] for row in pairwise if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "record_type": "summary",
        "stage": "plateau",
        "task_name": task["name"],
        "difficulty_label": task.get("difficulty_label", task["name"]),
        "difficulty_order": task.get("difficulty_order"),
        "task_type": task.get("task_type", "jsonl"),
        "split_seed": metadata["split_seed"],
        "train_count": metadata["train_count"],
        "monitor_count": metadata["monitor_count"],
        "probe_count": metadata["probe_count"],
        "model_count": int(model_count),
        "output_bits": int(output_bits),
        "pilot_steps": metadata["pilot_steps"],
        "pilot_stop_source": metadata["pilot_stop_source"],
        "pilot_best_monitor_bit_accuracy": metadata["pilot_best_monitor_bit_accuracy"],
        "pilot_best_probe_bit_accuracy": metadata.get("pilot_best_probe_bit_accuracy"),
        "pilot_monitor_prediction_stability": metadata.get("pilot_monitor_prediction_stability"),
        "pilot_probe_prediction_stability": metadata.get("pilot_probe_prediction_stability"),
        "pilot_label_evals_since_best": metadata.get("pilot_label_evals_since_best"),
        "pilot_prediction_stable_evals": metadata.get("pilot_prediction_stable_evals"),
        "mean_train_steps": float(np.mean([row["train_steps"] for row in selected])),
        "mean_train_fit_step": float(np.mean([
            row["train_fit_step"] for row in selected
            if row.get("train_fit_step") is not None
        ])) if any(row.get("train_fit_step") is not None for row in selected) else None,
        "mean_probe_bit_accuracy": float(np.mean([
            row["probe_metrics"]["bit_accuracy"] for row in selected
        ])),
        "mean_probe_exact_accuracy": float(np.mean([
            row["probe_metrics"]["exact_accuracy"] for row in selected
        ])),
        "majority_vote_bit_accuracy": float(1 - majority_error.mean()),
        "majority_vote_exact_accuracy": float((~majority_error).all(axis=1).mean()),
        "majority_vote_tied_bit_fraction": float(tied_vote.mean()),
        "mean_prediction_bit_agreement": float(majority_fraction.mean()),
        "mean_pairwise_prediction_bit_agreement": pairwise_mean("prediction_bit_agreement"),
        "mean_pairwise_prediction_bit_hamming_distance": pairwise_mean(
            "prediction_bit_hamming_distance"
        ),
        "mean_prediction_entropy_bits": float(entropy.mean()),
        "unanimously_same_prediction_bit_fraction": float(unanimous.mean()),
        "unanimously_correct_bit_fraction": float(unanimously_correct.mean()),
        "unanimously_wrong_bit_fraction": float(unanimously_wrong.mean()),
        "mixed_error_state_bit_fraction": float(mixed_error_state.mean()),
        "prediction_one_rate": float(predictions.mean()),
        "mean_pairwise_joint_error_lift": pairwise_mean("joint_error_lift"),
        "mean_pairwise_error_jaccard": pairwise_mean("error_jaccard"),
        "mean_pairwise_error_phi_correlation": pairwise_mean("error_phi_correlation"),
        "mean_pairwise_error_state_cohen_kappa": pairwise_mean("error_state_cohen_kappa"),
    }
    return summary, run_stats, pairwise


def prepare_experiment(task, cfg, dataset_cache, train_count, split_seed):
    inputs, targets, input_texts, dataset_meta = dataset_cache[task["name"]]
    split = make_split(
        inputs,
        targets,
        input_texts,
        train_count,
        cfg.MONITOR_COUNT,
        cfg.PROBE_COUNT,
        split_seed,
    )
    split["split_seed"] = int(split_seed)
    split["train_count"] = int(train_count)

    safe_task_name = task["name"].replace("/", "_").replace("\\", "_")
    experiment_name = f"{safe_task_name}_n{int(train_count)}_split{int(split_seed)}"
    output_dir = Path(cfg.OUTPUT_ROOT) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metadata": output_dir / "metadata.jsonl",
        "train": output_dir / "train.jsonl",
        "monitor": output_dir / "monitor.jsonl",
        "probe": output_dir / "probe.jsonl",
        "history": output_dir / "training_history.jsonl",
        "predictions": output_dir / "predictions.jsonl",
        "run_statistics": output_dir / "run_statistics.jsonl",
        "pairwise_statistics": output_dir / "pairwise_statistics.jsonl",
        "summary": output_dir / "summary.jsonl",
    }

    metadata = {
        "record_type": "metadata",
        "experiment_name": experiment_name,
        "task_name": task["name"],
        "task_type": task.get("task_type", "jsonl"),
        "difficulty_label": task.get("difficulty_label", task["name"]),
        "difficulty_order": task.get("difficulty_order"),
        "dataset_path": task.get("dataset_path"),
        "split_seed": int(split_seed),
        "train_count": int(train_count),
        "monitor_count": int(len(split["monitor_x"])),
        "probe_count": int(len(split["probe_x"])),
        "input_bits": int(dataset_meta["input_bits"]),
        "output_bits": int(dataset_meta["output_bits"]),
        "unique_samples": int(dataset_meta["unique_samples"]),
        "dataset_meta": dataset_meta,
        "model_seeds": list(cfg.MODEL_SEEDS),
        "pilot_seed": int(cfg.PILOT_SEED),
        "config": {
            "hidden_size": cfg.HIDDEN_SIZE,
            "hidden_layers": cfg.HIDDEN_LAYERS,
            "dropout": cfg.DROPOUT,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE,
            "eval_interval_steps": cfg.EVAL_INTERVAL_STEPS,
            "max_steps": cfg.MAX_STEPS,
            "plateau_patience_evals": cfg.PLATEAU_PATIENCE_EVALS,
            "plateau_min_delta": cfg.PLATEAU_MIN_DELTA,
            "plateau_min_steps_after_train_fit": cfg.PLATEAU_MIN_STEPS_AFTER_TRAIN_FIT,
            "use_probe_label_for_plateau": cfg.USE_PROBE_LABEL_FOR_PLATEAU,
            "use_prediction_stability_for_plateau": cfg.USE_PREDICTION_STABILITY_FOR_PLATEAU,
            "prediction_stability_threshold": cfg.PREDICTION_STABILITY_THRESHOLD,
            "prediction_stability_patience_evals": cfg.PREDICTION_STABILITY_PATIENCE_EVALS,
            "random_max_steps_after_train_fit": cfg.RANDOM_MAX_STEPS_AFTER_TRAIN_FIT,
            "plateau_max_steps_after_train_fit": cfg.PLATEAU_MAX_STEPS_AFTER_TRAIN_FIT,
        },
    }

    if cfg.OVERWRITE_EXISTING_OUTPUT:
        for path in paths.values():
            path.write_text("", encoding="utf-8")
    elif not cfg.RESUME_EXISTING_OUTPUT and paths["predictions"].exists():
        raise FileExistsError(f"结果已存在：{output_dir}")

    if not paths["metadata"].exists() or cfg.OVERWRITE_EXISTING_OUTPUT:
        write_jsonl(paths["metadata"], [metadata])
        write_split_file(paths["train"], "train", split["train_indices"], split["train_inputs"], split["train_y"])
        write_split_file(paths["monitor"], "monitor", split["monitor_indices"], split["monitor_inputs"], split["monitor_y"])
        write_split_file(paths["probe"], "probe", split["probe_indices"], split["probe_inputs"], split["probe_y"])

    for key in ("history", "predictions", "run_statistics", "pairwise_statistics", "summary"):
        if not paths[key].exists() or cfg.OVERWRITE_EXISTING_OUTPUT:
            paths[key].write_text("", encoding="utf-8")
    return output_dir, paths, metadata, split


def write_split_file(path, name, indices, inputs, targets):
    write_jsonl(path, [
        {
            f"{name}_offset": i,
            "dataset_index": int(index),
            "input": text,
            "target": bits_to_text(target),
        }
        for i, (index, text, target) in enumerate(zip(
            indices,
            inputs,
            targets.to(torch.uint8).numpy(),
        ))
    ])


def load_existing_prediction_seeds(path):
    completed = set()
    for row in read_jsonl(path):
        if row.get("record_type") == "prediction" and row.get("stage") == "plateau":
            completed.add(int(row["model_seed"]))
    return completed


def run_one_experiment(task, cfg, dataset_cache, train_count, split_seed):
    output_dir, paths, metadata, split = prepare_experiment(
        task, cfg, dataset_cache, train_count, split_seed
    )
    start = time.time()

    existing_predictions = read_jsonl(paths["predictions"])
    existing_pilot = [
        row for row in existing_predictions
        if row.get("record_type") == "prediction" and row.get("source") == "pilot"
    ]
    if existing_pilot and cfg.RESUME_EXISTING_OUTPUT and not cfg.OVERWRITE_EXISTING_OUTPUT:
        pilot_record = existing_pilot[-1]
        pilot_steps = int(pilot_record["train_steps"])
        print(f"{metadata['experiment_name']} 已有 pilot step={pilot_steps}，跳过 pilot。")
    else:
        print(f"{metadata['experiment_name']} 运行 pilot，寻找 monitor 平台期。")
        pilot_record = run_training(
            task, cfg, split, cfg.PILOT_SEED, output_dir,
            target_steps=None, source="pilot",
        )
        pilot_record["source"] = "pilot"
        append_jsonl(paths["predictions"], pilot_record)
        pilot_steps = int(pilot_record["train_steps"])
        print(
            f"pilot 完成：step={pilot_steps}, stop={pilot_record['stop_source']}, "
            f"monitor={pilot_record['monitor_metrics']['bit_accuracy']:.6f}, "
            f"probe={pilot_record['probe_metrics']['bit_accuracy']:.6f}"
        )

    metadata.update({
        "pilot_steps": int(pilot_steps),
        "pilot_stop_source": pilot_record["stop_source"],
        "pilot_best_monitor_bit_accuracy": pilot_record["best_monitor_bit_accuracy"],
        "pilot_best_monitor_step": pilot_record["best_monitor_step"],
        "pilot_best_probe_bit_accuracy": pilot_record.get("best_probe_bit_accuracy"),
        "pilot_best_probe_step": pilot_record.get("best_probe_step"),
        "pilot_label_evals_since_best": pilot_record.get("label_evals_since_best"),
        "pilot_prediction_stable_evals": pilot_record.get("prediction_stable_evals"),
        "pilot_monitor_prediction_stability": pilot_record.get("monitor_prediction_stability"),
        "pilot_probe_prediction_stability": pilot_record.get("probe_prediction_stability"),
    })
    write_jsonl(paths["metadata"], [metadata])

    completed = load_existing_prediction_seeds(paths["predictions"])
    for seed in cfg.MODEL_SEEDS:
        if int(seed) in completed and cfg.RESUME_EXISTING_OUTPUT:
            print(f"{metadata['experiment_name']} seed={seed} 已完成，跳过。")
            continue
        record = run_training(
            task, cfg, split, int(seed), output_dir,
            target_steps=pilot_steps, source="ensemble",
        )
        append_jsonl(paths["predictions"], record)
        print(
            f"{metadata['experiment_name']} seed={seed} 完成："
            f"step={record['train_steps']}, "
            f"monitor={record['monitor_metrics']['bit_accuracy']:.6f}, "
            f"probe={record['probe_metrics']['bit_accuracy']:.6f}"
        )

    all_prediction_records = read_jsonl(paths["predictions"])
    all_prediction_records = [
        row for row in all_prediction_records
        if row.get("source") != "pilot"
    ]
    probe_targets = split["probe_y"].to(torch.uint8).numpy()

    summary, run_stats, pairwise = summarize_predictions(
        task, metadata, probe_targets, all_prediction_records
    )
    paths["run_statistics"].write_text("", encoding="utf-8")
    paths["pairwise_statistics"].write_text("", encoding="utf-8")
    paths["summary"].write_text("", encoding="utf-8")
    summaries = []
    if summary is not None:
        for row in run_stats:
            append_jsonl(paths["run_statistics"], row)
        for row in pairwise:
            append_jsonl(paths["pairwise_statistics"], row)
        append_jsonl(paths["summary"], summary)
        summaries.append(summary)

    index_record = {
        "record_type": "task_difficulty_plateau_job",
        "status": "ok" if summaries else "no_summary",
        "task_name": task["name"],
        "task_type": task.get("task_type", "jsonl"),
        "difficulty_label": task.get("difficulty_label", task["name"]),
        "difficulty_order": task.get("difficulty_order"),
        "dataset_path": task.get("dataset_path"),
        "train_count": int(train_count),
        "split_seed": int(split_seed),
        "experiment_name": metadata["experiment_name"],
        "experiment_dir": str(output_dir),
        "pilot_steps": int(pilot_steps),
        "pilot_stop_source": pilot_record["stop_source"],
        "completed_model_count": len(load_existing_prediction_seeds(paths["predictions"])),
        "expected_model_count": len(cfg.MODEL_SEEDS),
        "elapsed_seconds": time.time() - start,
    }
    return index_record, summaries


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


def write_html(path, rows):
    data = json.dumps(rows, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>任务难度 x 数据量 probe 一致性</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
body {{
  margin: 0;
  padding: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  background: #f7f3e8;
  color: #233142;
}}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
.sub {{ color: #5b677a; margin: 0 0 20px; line-height: 1.6; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(360px, 1fr));
  gap: 18px;
}}
.panel {{
  background: #fffaf0;
  border: 1px solid #e0d4bb;
  border-radius: 14px;
  padding: 14px;
}}
.chart {{ height: 390px; min-width: 0; }}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: 1fr; }}
  body {{ padding: 14px; }}
}}
</style>
</head>
<body>
<h1>任务难度 x 数据量 probe 一致性</h1>
<p class="sub">每个任务/数据量先用 pilot 找 monitor 平台期，然后所有 seed 训练同样步数，最终在 probe 上统计。</p>
<div class="grid">
  <div class="panel"><div id="acc" class="chart"></div></div>
  <div class="panel"><div id="agree" class="chart"></div></div>
  <div class="panel"><div id="entropy" class="chart"></div></div>
  <div class="panel"><div id="error" class="chart"></div></div>
</div>
<script>
const rows = {data};
const charts = {{
  acc: echarts.init(document.getElementById('acc')),
  agree: echarts.init(document.getElementById('agree')),
  entropy: echarts.init(document.getElementById('entropy')),
  error: echarts.init(document.getElementById('error'))
}};
const tasks = [...new Set(rows.map(r => r.difficulty_label || r.task_name))];
const byTask = label => rows
  .filter(r => (r.difficulty_label || r.task_name) === label)
  .sort((a, b) => a.train_count - b.train_count);
const pct = v => (v == null || Number.isNaN(v)) ? '-' : (v * 100).toFixed(2) + '%';
const val = v => (v == null || Number.isNaN(v)) ? '-' : Number(v).toFixed(4);

function seriesFor(key, percent=false) {{
  return tasks.map(label => {{
    const data = byTask(label).map(r => [r.train_count, percent ? r[key] * 100 : r[key]]);
    return {{ name: label, type: 'line', showSymbol: true, smooth: false, data }};
  }});
}}
function option(title, key, percent=false, max=null) {{
  return {{
    title: {{ text: title, left: 8, top: 4, textStyle: {{ fontSize: 16 }} }},
    tooltip: {{
      trigger: 'axis',
      formatter: params => params.map(p => {{
        const raw = percent ? p.data[1] / 100 : p.data[1];
        return `${{p.seriesName}} n=${{p.data[0]}}: ${{percent ? pct(raw) : val(raw)}}`;
      }}).join('<br>')
    }},
    legend: {{ top: 30, type: 'scroll' }},
    grid: {{ left: 64, right: 24, top: 76, bottom: 52 }},
    xAxis: {{ type: 'value', name: '训练样本数 n' }},
    yAxis: {{ type: 'value', min: 0, max, axisLabel: {{ formatter: percent ? '{{value}}%' : '{{value}}' }} }},
    series: seriesFor(key, percent)
  }};
}}
charts.acc.setOption(option('probe 准确率', 'mean_probe_bit_accuracy', true, 100));
charts.agree.setOption(option('跨 seed bit agreement', 'mean_prediction_bit_agreement', true, 100));
charts.entropy.setOption(option('预测熵 bits', 'mean_prediction_entropy_bits', false, 1));
charts.error.setOption(option('共同错误 phi', 'mean_pairwise_error_phi_correlation', false, null));
window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def package_results(root, cfg):
    zip_path = root / cfg.ZIP_NAME
    include_names = {
        cfg.SWEEP_SUMMARY_NAME,
        cfg.SWEEP_INDEX_NAME,
        cfg.SWEEP_CSV_NAME,
        cfg.HTML_NAME,
        "metadata.jsonl",
        "training_history.jsonl",
        "summary.jsonl",
        "run_statistics.jsonl",
        "pairwise_statistics.jsonl",
    }
    if cfg.PACKAGE_RAW_PREDICTIONS:
        include_names.add("predictions.jsonl")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            if path.name in include_names:
                zf.write(path, path.relative_to(root))
    return zip_path


def load_all_datasets(cfg):
    cache = {}
    for task in cfg.TASKS:
        task_type = task.get("task_type", "jsonl")
        print(f"加载任务：{task['name']} ({task_type})")
        cache[task["name"]] = load_dataset(task, cfg)
        meta = cache[task["name"]][3]
        print(
            f"  unique={meta['unique_samples']}, "
            f"input_bits={meta['input_bits']}, output_bits={meta['output_bits']}"
        )
    return cache


def main():
    cfg = Config()
    root = Path(cfg.OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / cfg.SWEEP_INDEX_NAME
    summary_path = root / cfg.SWEEP_SUMMARY_NAME
    csv_path = root / cfg.SWEEP_CSV_NAME
    html_path = root / cfg.HTML_NAME

    if cfg.OVERWRITE_EXISTING_OUTPUT:
        index_path.write_text("", encoding="utf-8")
        summary_path.write_text("", encoding="utf-8")

    dataset_cache = load_all_datasets(cfg)
    all_summaries = []
    if summary_path.exists() and cfg.RESUME_EXISTING_OUTPUT and not cfg.OVERWRITE_EXISTING_OUTPUT:
        all_summaries = read_jsonl(summary_path)

    total_jobs = len(cfg.TASKS) * len(cfg.TRAIN_COUNTS) * len(cfg.SPLIT_SEEDS)
    print(f"准备运行 {total_jobs} 个任务/数据量/划分组合。设备：{cfg.DEVICE}")
    for task in cfg.TASKS:
        for train_count in cfg.TRAIN_COUNTS:
            for split_seed in cfg.SPLIT_SEEDS:
                print("\n" + "=" * 80)
                print(
                    f"任务：{task['name']}，n={train_count}, split_seed={split_seed}"
                )
                try:
                    index_record, summaries = run_one_experiment(
                        task, cfg, dataset_cache, train_count, split_seed
                    )
                    append_jsonl(index_path, index_record)
                    all_summaries = [
                        row for row in all_summaries
                        if not (
                            row.get("task_name") == task["name"]
                            and row.get("train_count") == int(train_count)
                            and row.get("split_seed") == int(split_seed)
                        )
                    ]
                    all_summaries.extend(summaries)
                    write_jsonl(summary_path, all_summaries)
                    write_csv(csv_path, all_summaries)
                    write_html(html_path, all_summaries)
                    zip_path = package_results(root, cfg)
                    print(f"完成：{index_record['experiment_name']}")
                    for row in summaries:
                        print(
                            f"  bit={row['mean_probe_bit_accuracy']:.4f}, "
                            f"exact={row['mean_probe_exact_accuracy']:.4f}, "
                            f"agreement={row['mean_prediction_bit_agreement']:.4f}, "
                            f"entropy={row['mean_prediction_entropy_bits']:.4f}, "
                            f"pilot_steps={row['pilot_steps']}"
                        )
                    print(f"打包：{zip_path}")
                except Exception as exc:
                    error_record = {
                        "record_type": "task_difficulty_plateau_job",
                        "status": "error",
                        "task_name": task["name"],
                        "train_count": int(train_count),
                        "split_seed": int(split_seed),
                        "error": repr(exc),
                    }
                    append_jsonl(index_path, error_record)
                    print(f"出错：{repr(exc)}")

    if all_summaries:
        write_csv(csv_path, all_summaries)
        write_html(html_path, all_summaries)
        zip_path = package_results(root, cfg)
        print("\n全部完成。")
        print(f"汇总：{summary_path}")
        print(f"CSV：{csv_path}")
        print(f"HTML：{html_path}")
        print(f"ZIP：{zip_path}")


if __name__ == "__main__":
    main()
