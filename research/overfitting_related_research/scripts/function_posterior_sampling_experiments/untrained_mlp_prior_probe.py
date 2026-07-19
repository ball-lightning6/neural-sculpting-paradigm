"""
未训练 MLP 的函数先验 probe 统计。

用途：
1. 随机初始化多个 MLP，不进行任何训练。
2. 在固定 probe 输入上记录输出分布。
3. 计算跨 seed 的 agreement、entropy、one-rate、unanimity 等指标。

这个脚本用于回答一个基础问题：
低 n random 训练后出现的高 probe agreement，到底来自未训练初始化先验，
还是来自少量训练样本诱导出的训练后函数偏置？

配置方式：
直接修改 Config，不使用 argparse，可以在本地或服务器 Python 环境中运行。
"""

import csv
import json
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Config:
    # =========================
    # probe 来源
    # =========================
    # task_type:
    # - "jsonl": 读取 {"input": ..., "output": ...} 数据集，按 split_seed 划出 probe。
    # - "random_probe": 只随机生成 probe 输入，不使用真实标签。
    TASKS = [
        {
            "name": "random_probe_30bit",
            "task_type": "random_probe",
            "num_samples": 30000,
            "input_bits": 30,
            "output_bits": 30,
            "data_seed": 20260712,
        },
        # 如果想直接复用 rule30 数据集的 probe，可以打开下面这些任务。
        # {
        #     "name": "rule30_layer1",
        #     "task_type": "jsonl",
        #     "dataset_path": (
        #         "research/overfitting_related_research/datasets/"
        #         "ca_rule30_layer1_len30_n300000.jsonl"
        #     ),
        # },
        # {
        #     "name": "rule30_layer2",
        #     "task_type": "jsonl",
        #     "dataset_path": (
        #         "research/overfitting_related_research/datasets/"
        #         "ca_rule30_layer2_len30_n300000.jsonl"
        #     ),
        # },
        # {
        #     "name": "rule30_layer3",
        #     "task_type": "jsonl",
        #     "dataset_path": (
        #         "research/overfitting_related_research/datasets/"
        #         "ca_rule30_layer3_len30_n300000.jsonl"
        #     ),
        # },
    ]

    INPUT_KEY = "input"
    OUTPUT_KEY = "output"
    DEDUPLICATE_INPUTS = True

    # 对 jsonl 任务，probe 按和 sweep 脚本相同的方式从数据集中切出来。
    # 对 random_probe，TRAIN_COUNT/MONITOR_COUNT 只用于保留同样的“跳过区间”语义。
    TRAIN_COUNTS = (800,)
    SPLIT_SEEDS = (20260711,)
    MONITOR_COUNT = 3000
    PROBE_COUNT = 5000

    # =========================
    # 模型结构：默认对齐 sweep_task_difficulty_plateau.py
    # =========================
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # 随机初始化采样
    # =========================
    MODEL_SEEDS = tuple(range(1000))
    PREDICT_BATCH_SIZE = 4096

    # 保存每个 seed 的预测会比较大；默认只保存统计。
    SAVE_RAW_PREDICTIONS = False
    # 为了计算 error phi / lift，可保留预测矩阵到内存。
    # 1000 seeds * 5000 probe * 30 bits 约 150 MB，通常可以接受。
    KEEP_PREDICTION_MATRIX_IN_MEMORY = True
    PAIRWISE_SAMPLE_COUNT = 20000
    PAIRWISE_SAMPLE_SEED = 20260712

    # =========================
    # 输出
    # =========================
    OUTPUT_ROOT = "research/overfitting_related_research/results_untrained_mlp_prior"
    ZIP_NAME = "untrained_mlp_prior_probe_package.zip"
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


def bits_to_text(bits):
    return "".join("1" if int(bit) else "0" for bit in bits)


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


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        "task_type": "jsonl",
        "dataset_path": str(path),
        "duplicates_removed": duplicate_count,
    })


def load_random_probe_dataset(task):
    rng = np.random.default_rng(int(task.get("data_seed", 0)))
    num_samples = int(task.get("num_samples", 30000))
    input_bits = int(task.get("input_bits", 30))
    output_bits = int(task.get("output_bits", input_bits))

    seen = set()
    records = []
    while len(records) < num_samples:
        x_bits = rng.integers(0, 2, size=input_bits, dtype=np.uint8)
        x_text = bits_to_text(x_bits)
        if x_text in seen:
            continue
        seen.add(x_text)
        # random_probe 没有真实标签；这里填零，只用于保持数组形状。
        y_bits = np.zeros(output_bits, dtype=np.uint8)
        records.append((x_text, x_bits.tolist(), y_bits.tolist()))

    return dataset_from_records(records, {
        "task_type": "random_probe",
        "dataset_path": None,
        "duplicates_removed": 0,
        "data_seed": int(task.get("data_seed", 0)),
        "has_targets": False,
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
        "has_targets": bool(extra_meta.get("has_targets", True)),
    })
    return inputs, targets, input_texts, metadata


def load_dataset(task, cfg):
    if task.get("task_type") == "random_probe":
        return load_random_probe_dataset(task)
    return load_jsonl_dataset(task, cfg)


def make_probe_split(inputs, targets, input_texts, train_count, monitor_count, probe_count, split_seed):
    total = len(inputs)
    train_count = int(train_count)
    monitor_count = int(monitor_count)
    if train_count + monitor_count >= total:
        raise ValueError(
            f"train_count + monitor_count 不能大于等于总样本数："
            f"{train_count} + {monitor_count} >= {total}"
        )
    remaining = total - train_count - monitor_count
    actual_probe_count = min(int(probe_count), remaining)

    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(total, generator=generator)
    probe_idx = indices[
        train_count + monitor_count:
        train_count + monitor_count + actual_probe_count
    ]
    return {
        "probe_x": inputs[probe_idx],
        "probe_y": targets[probe_idx],
        "probe_indices": probe_idx.tolist(),
        "probe_inputs": [input_texts[i] for i in probe_idx.tolist()],
    }


def predict_untrained_seed(seed, task_meta, probe_x, cfg):
    set_seed(seed)
    device = torch.device(cfg.DEVICE)
    model = MLP(task_meta["input_bits"], task_meta["output_bits"], cfg).to(device)
    model.eval()

    chunks = []
    loader = DataLoader(TensorDataset(probe_x), batch_size=cfg.PREDICT_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            pred = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()
            chunks.append(pred)
    return np.concatenate(chunks, axis=0)


def binary_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    out[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
    return out


def summarize_prediction_matrix(predictions, targets, has_targets, cfg):
    model_count, probe_count, output_bits = predictions.shape
    flat = predictions.reshape(model_count, -1).astype(np.uint8)
    ones = flat.sum(axis=0)
    p = ones / float(model_count)
    majority_fraction = np.maximum(p, 1 - p)
    entropy = binary_entropy(p)
    unanimous = (ones == 0) | (ones == model_count)
    tied = ones * 2 == model_count

    if model_count > 1:
        pair_same = (
            ones * (ones - 1)
            + (model_count - ones) * (model_count - ones - 1)
        ) / float(model_count * (model_count - 1))
    else:
        pair_same = np.ones_like(p, dtype=np.float64)

    summary = {
        "model_count": int(model_count),
        "probe_count": int(probe_count),
        "output_bits": int(output_bits),
        "total_probe_bits": int(probe_count * output_bits),
        "prediction_one_rate": float(p.mean()),
        "mean_prediction_bit_agreement": float(majority_fraction.mean()),
        "mean_pairwise_prediction_bit_agreement": float(pair_same.mean()),
        "mean_pairwise_prediction_bit_hamming_distance": float(1 - pair_same.mean()),
        "mean_prediction_entropy_bits": float(entropy.mean()),
        "unanimously_same_prediction_bit_fraction": float(unanimous.mean()),
        "majority_vote_tied_bit_fraction": float(tied.mean()),
    }

    if not has_targets:
        return summary, [], []

    target_flat = targets.to(torch.uint8).numpy().reshape(-1)
    errors = flat != target_flat[None, :]
    bit_accs = 1.0 - errors.mean(axis=1)
    exact_accs = 1.0 - errors.reshape(model_count, probe_count, output_bits).any(axis=2).mean(axis=1)

    majority_bits = (ones * 2 >= model_count).astype(np.uint8)
    majority_error = majority_bits != target_flat
    unanimous_correct = unanimous & (~majority_error)
    unanimous_wrong = unanimous & majority_error
    mixed_error_state = (~unanimous) & (ones > 0) & (ones < model_count)

    summary.update({
        "mean_probe_bit_accuracy": float(bit_accs.mean()),
        "mean_probe_exact_accuracy": float(exact_accs.mean()),
        "majority_vote_bit_accuracy": float((~majority_error).mean()),
        "majority_vote_exact_accuracy": float((~majority_error.reshape(probe_count, output_bits)).all(axis=1).mean()),
        "unanimously_correct_bit_fraction": float(unanimous_correct.mean()),
        "unanimously_wrong_bit_fraction": float(unanimous_wrong.mean()),
        "mixed_error_state_bit_fraction": float(mixed_error_state.mean()),
    })

    run_stats = []
    for i, seed in enumerate(cfg.MODEL_SEEDS):
        run_stats.append({
            "record_type": "run_statistics",
            "model_seed": int(seed),
            "probe_bit_accuracy": float(bit_accs[i]),
            "probe_exact_accuracy": float(exact_accs[i]),
            "prediction_one_rate": float(flat[i].mean()),
        })

    pairwise = sample_pairwise_error_stats(errors, cfg)
    if pairwise:
        for key in pairwise[0]:
            if key in {"record_type", "seed_a", "seed_b"}:
                continue
            values = [row[key] for row in pairwise if row[key] is not None and not math.isnan(row[key])]
            summary["mean_pairwise_" + key] = float(np.mean(values)) if values else None

    return summary, run_stats, pairwise


def sample_pairwise_error_stats(errors, cfg):
    model_count = errors.shape[0]
    if model_count < 2:
        return []

    rng = np.random.default_rng(int(cfg.PAIRWISE_SAMPLE_SEED))
    max_pairs = model_count * (model_count - 1) // 2
    sample_count = min(int(cfg.PAIRWISE_SAMPLE_COUNT), max_pairs)
    seen = set()
    pairs = []
    while len(pairs) < sample_count:
        a = int(rng.integers(0, model_count))
        b = int(rng.integers(0, model_count - 1))
        if b >= a:
            b += 1
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    rows = []
    for a, b in pairs:
        ea = errors[a].astype(bool)
        eb = errors[b].astype(bool)
        pa = float(ea.mean())
        pb = float(eb.mean())
        joint = float((ea & eb).mean())
        union = float((ea | eb).mean())
        expected = pa * pb
        lift = joint / expected if expected > 0 else None
        jaccard = joint / union if union > 0 else None

        denom = math.sqrt(pa * (1 - pa) * pb * (1 - pb))
        phi = (joint - expected) / denom if denom > 0 else None

        rows.append({
            "record_type": "pairwise_statistics",
            "seed_a": int(cfg.MODEL_SEEDS[a]),
            "seed_b": int(cfg.MODEL_SEEDS[b]),
            "error_phi_correlation": phi,
            "joint_error_lift": lift,
            "error_jaccard": jaccard,
            "joint_error_rate": joint,
        })
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


def package_results(root, cfg):
    zip_path = root / cfg.ZIP_NAME
    include_names = {
        "summary.jsonl",
        "summary.csv",
        "metadata.jsonl",
        "run_statistics.jsonl",
        "pairwise_statistics_sample.jsonl",
        "majority_prediction.jsonl",
    }
    if cfg.SAVE_RAW_PREDICTIONS:
        include_names.add("predictions.jsonl")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            if path.name in include_names:
                zf.write(path, path.relative_to(root))
    return zip_path


def run_one(task, cfg, dataset_cache, train_count, split_seed):
    inputs, targets, input_texts, dataset_meta = dataset_cache[task["name"]]
    split = make_probe_split(
        inputs,
        targets,
        input_texts,
        train_count,
        cfg.MONITOR_COUNT,
        cfg.PROBE_COUNT,
        split_seed,
    )

    task_name = task["name"].replace("/", "_").replace("\\", "_")
    exp_name = f"{task_name}_untrained_n{int(train_count)}_split{int(split_seed)}"
    out_dir = Path(cfg.OUTPUT_ROOT) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.OVERWRITE_EXISTING_OUTPUT:
        for name in [
            "metadata.jsonl",
            "summary.jsonl",
            "summary.csv",
            "run_statistics.jsonl",
            "pairwise_statistics_sample.jsonl",
            "majority_prediction.jsonl",
            "predictions.jsonl",
        ]:
            (out_dir / name).write_text("", encoding="utf-8")

    metadata = {
        "record_type": "metadata",
        "experiment_name": exp_name,
        "task_name": task["name"],
        "task_type": task.get("task_type"),
        "dataset_path": task.get("dataset_path"),
        "train_count_for_probe_split": int(train_count),
        "monitor_count_for_probe_split": int(cfg.MONITOR_COUNT),
        "probe_count": int(len(split["probe_x"])),
        "split_seed": int(split_seed),
        "input_bits": int(dataset_meta["input_bits"]),
        "output_bits": int(dataset_meta["output_bits"]),
        "has_targets": bool(dataset_meta["has_targets"]),
        "model_seeds": list(cfg.MODEL_SEEDS),
        "model": {
            "hidden_size": int(cfg.HIDDEN_SIZE),
            "hidden_layers": int(cfg.HIDDEN_LAYERS),
            "dropout": float(cfg.DROPOUT),
        },
        "device": cfg.DEVICE,
    }
    write_jsonl(out_dir / "metadata.jsonl", [metadata])

    predictions = []
    for seed in tqdm(cfg.MODEL_SEEDS, desc=exp_name):
        pred = predict_untrained_seed(seed, dataset_meta, split["probe_x"], cfg)
        if cfg.KEEP_PREDICTION_MATRIX_IN_MEMORY:
            predictions.append(pred)
        if cfg.SAVE_RAW_PREDICTIONS:
            append_jsonl(out_dir / "predictions.jsonl", {
                "record_type": "prediction",
                "model_seed": int(seed),
                "prediction": "".join(bits_to_text(row) for row in pred),
            })

    if not predictions:
        raise RuntimeError("没有预测矩阵，无法统计。请打开 KEEP_PREDICTION_MATRIX_IN_MEMORY。")
    prediction_matrix = np.stack(predictions, axis=0)

    summary, run_stats, pairwise = summarize_prediction_matrix(
        prediction_matrix,
        split["probe_y"],
        bool(dataset_meta["has_targets"]),
        cfg,
    )
    summary.update({
        "record_type": "summary",
        "experiment_name": exp_name,
        "task_name": task["name"],
        "task_type": task.get("task_type"),
        "train_count_for_probe_split": int(train_count),
        "split_seed": int(split_seed),
    })

    write_jsonl(out_dir / "summary.jsonl", [summary])
    write_csv(out_dir / "summary.csv", [summary])
    write_jsonl(out_dir / "run_statistics.jsonl", run_stats)
    write_jsonl(out_dir / "pairwise_statistics_sample.jsonl", pairwise)

    flat = prediction_matrix.reshape(len(cfg.MODEL_SEEDS), -1)
    majority_bits = (flat.sum(axis=0) * 2 >= len(cfg.MODEL_SEEDS)).astype(np.uint8)
    append_jsonl(out_dir / "majority_prediction.jsonl", {
        "record_type": "majority_prediction",
        "prediction": "".join(str(int(bit)) for bit in majority_bits),
    })

    return summary


def main():
    cfg = Config()
    root = Path(cfg.OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    dataset_cache = {}
    for task in cfg.TASKS:
        print(f"加载 probe 来源：{task['name']}")
        dataset_cache[task["name"]] = load_dataset(task, cfg)
        meta = dataset_cache[task["name"]][3]
        print(
            f"  unique={meta['unique_samples']}, "
            f"input_bits={meta['input_bits']}, output_bits={meta['output_bits']}, "
            f"has_targets={meta['has_targets']}"
        )

    all_summaries = []
    for task in cfg.TASKS:
        for train_count in cfg.TRAIN_COUNTS:
            for split_seed in cfg.SPLIT_SEEDS:
                print("\n" + "=" * 80)
                print(
                    f"未训练先验采样：task={task['name']}, "
                    f"probe split n={train_count}, split_seed={split_seed}"
                )
                summary = run_one(task, cfg, dataset_cache, train_count, split_seed)
                all_summaries.append(summary)
                print(
                    f"one_rate={summary['prediction_one_rate']:.4f}, "
                    f"agreement={summary['mean_prediction_bit_agreement']:.4f}, "
                    f"pair_agree={summary['mean_pairwise_prediction_bit_agreement']:.4f}, "
                    f"entropy={summary['mean_prediction_entropy_bits']:.4f}, "
                    f"unanim={summary['unanimously_same_prediction_bit_fraction']:.4f}"
                )

    write_jsonl(root / "summary.jsonl", all_summaries)
    write_csv(root / "summary.csv", all_summaries)
    zip_path = package_results(root, cfg)
    print("\n完成。")
    print(f"汇总：{root / 'summary.jsonl'}")
    print(f"CSV：{root / 'summary.csv'}")
    print(f"打包：{zip_path}")


if __name__ == "__main__":
    main()
