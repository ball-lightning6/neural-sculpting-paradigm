# %% cell 1
"""
训练多个过拟合态元胞自动机模型，并保存它们在固定 probe 集上的输出。

实验目的：
1. 固定同一训练集、monitor 集和 probe 集。
2. 先用一个 pilot seed 确定统一训练步数，也可以手动指定。
3. 用多个模型随机种子训练到完全相同的步数。
4. 将每个模型在 probe 集上的离散预测保存为 JSONL，供独立分析脚本使用。

预测采用行主序 bit 字符串编码。例如 2 个样本、每个输出 3 bit：
[[0, 1, 0], [1, 1, 0]] 会保存为 "010110"。
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Config:
    # =========================
    # 数据与划分
    # =========================
    DATASET_PATH = (
        "research/overfitting_related_research/datasets/"
        "ca_rule30_layer1_len30_n300000.jsonl"
    )
    INPUT_KEY = "input"
    OUTPUT_KEY = "output"

    TRAIN_COUNT = 900
    MONITOR_COUNT = 3000
    # None 表示使用除训练集和 monitor 集以外的全部样本。
    PROBE_COUNT = None
    SPLIT_SEED = 20260709
    DEDUPLICATE_INPUTS = True

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

    # 设为整数时，所有 seed 直接训练这么多 optimizer steps。
    # 设为 None 时，先运行 pilot 自动寻找 monitor 指标的平台期。
    COMMON_TRAIN_STEPS = None
    MAX_PILOT_STEPS = 30000
    EVAL_INTERVAL_STEPS = 100

    # pilot 至少训练这么多步，并且训练集 exact accuracy 达标后才检测平台。
    MIN_PILOT_STEPS = 3000
    MIN_TRAIN_EXACT_FOR_PLATEAU = 1.0
    # 推荐使用较平滑的 monitor_bit_accuracy。
    # 也可改为 monitor_exact_accuracy 或 monitor_loss。
    PLATEAU_METRIC = "monitor_bit_accuracy"
    PLATEAU_WINDOW = 20
    PLATEAU_REQUIRED_WINDOWS = 3
    PLATEAU_MAX_MEAN_SHIFT = 0.001
    PLATEAU_MAX_SLOPE_PER_EVAL = 0.00005

    # True 时只运行 pilot，便于人工查看历史后再填写 COMMON_TRAIN_STEPS。
    PILOT_ONLY = False

    # False 表示不同模型 seed 只改变初始化、dropout 等模型随机性，
    # mini-batch 顺序保持一致。True 表示数据顺序也随模型 seed 改变。
    VARY_DATA_ORDER_BY_MODEL_SEED = False
    DATA_ORDER_SEED = 314159

    # =========================
    # 输出
    # =========================
    EXPERIMENT_NAME = "rule30_layer1_overfit_n900"
    OUTPUT_ROOT = "research/overfitting_related_research/results_overfit_ensemble"
    SAVE_MODELS = False
    # 同名实验中断后可继续：配置一致时复用停止步数并跳过已完成 seed。
    RESUME_EXISTING_OUTPUT = True
    # True 会清空同名实验的 JSONL 结果后重新开始。
    OVERWRITE_EXISTING_OUTPUT = False


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def bits_to_string(bits):
    return "".join("1" if int(bit) else "0" for bit in bits)


def load_dataset(cfg):
    path = Path(cfg.DATASET_PATH)
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
            x_string = bits_to_string(x_bits)

            y_string = bits_to_string(y_bits)
            if cfg.DEDUPLICATE_INPUTS and x_string in seen_outputs:
                if seen_outputs[x_string] != y_string:
                    raise ValueError(
                        f"第 {line_no} 行的输入重复，但标签与此前记录不一致："
                        f"{x_string}"
                    )
                duplicate_count += 1
                continue

            seen_outputs[x_string] = y_string
            records.append((x_string, x_bits, y_bits))

    if not records:
        raise ValueError("数据集为空。")

    input_bits = len(records[0][1])
    output_bits = len(records[0][2])
    if any(len(row[1]) != input_bits for row in records):
        raise ValueError("输入 bit 长度不一致。")
    if any(len(row[2]) != output_bits for row in records):
        raise ValueError("输出 bit 长度不一致。")

    inputs = torch.tensor([row[1] for row in records], dtype=torch.float32)
    targets = torch.tensor([row[2] for row in records], dtype=torch.float32)
    input_strings = [row[0] for row in records]

    metadata = {
        "dataset_path": str(path),
        "unique_samples": len(records),
        "duplicates_removed": duplicate_count,
        "input_bits": input_bits,
        "output_bits": output_bits,
    }
    return inputs, targets, input_strings, metadata


def split_dataset(inputs, targets, input_strings, cfg):
    total = len(inputs)
    required = cfg.TRAIN_COUNT + cfg.MONITOR_COUNT
    if required >= total:
        raise ValueError(
            f"训练集和 monitor 集共需 {required} 条，但数据集只有 {total} 条。"
        )

    generator = torch.Generator().manual_seed(cfg.SPLIT_SEED)
    indices = torch.randperm(total, generator=generator)

    train_end = cfg.TRAIN_COUNT
    monitor_end = train_end + cfg.MONITOR_COUNT

    if cfg.PROBE_COUNT is None:
        probe_end = total
    else:
        probe_end = monitor_end + cfg.PROBE_COUNT
        if probe_end > total:
            raise ValueError(
                f"probe 样本不足：需要 {cfg.PROBE_COUNT} 条，"
                f"实际只剩 {total - monitor_end} 条。"
            )

    train_idx = indices[:train_end]
    monitor_idx = indices[train_end:monitor_end]
    probe_idx = indices[monitor_end:probe_end]

    split = {
        "train_x": inputs[train_idx],
        "train_y": targets[train_idx],
        "monitor_x": inputs[monitor_idx],
        "monitor_y": targets[monitor_idx],
        "probe_x": inputs[probe_idx],
        "probe_y": targets[probe_idx],
        "probe_indices": probe_idx.tolist(),
        "probe_inputs": [input_strings[index] for index in probe_idx.tolist()],
    }
    return split


def make_train_loader(train_x, train_y, cfg, model_seed):
    order_seed = cfg.DATA_ORDER_SEED
    if cfg.VARY_DATA_ORDER_BY_MODEL_SEED:
        order_seed += model_seed

    return DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(order_seed),
    )


def evaluate(model, x, y, device, batch_size):
    model.eval()
    total_loss = 0.0
    total_bits = 0
    correct_bits = 0
    correct_rows = 0

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            total_loss += F.binary_cross_entropy_with_logits(
                logits,
                batch_y,
                reduction="sum",
            ).item()
            pred = logits > 0
            labels = batch_y > 0.5
            matches = pred == labels
            correct_bits += matches.sum().item()
            total_bits += matches.numel()
            correct_rows += matches.all(dim=1).sum().item()

    return {
        "loss": total_loss / total_bits,
        "bit_accuracy": correct_bits / total_bits,
        "exact_accuracy": correct_rows / len(x),
    }


def predict_bit_string(model, x, device, batch_size):
    model.eval()
    chunks = []
    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for (batch_x,) in loader:
            pred = (model(batch_x.to(device)) > 0).to(torch.uint8).cpu().numpy()
            chunks.append(pred.reshape(-1))

    flat = np.concatenate(chunks)
    ascii_bits = np.where(flat == 1, ord("1"), ord("0")).astype(np.uint8)
    return ascii_bits.tobytes().decode("ascii")


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_probe_jsonl(path, split):
    probe_y = split["probe_y"].to(torch.uint8).numpy()
    with path.open("w", encoding="utf-8") as f:
        for offset, (source_index, input_bits, target_bits) in enumerate(
            zip(
                split["probe_indices"],
                split["probe_inputs"],
                probe_y,
            )
        ):
            record = {
                "probe_offset": offset,
                "source_index": source_index,
                "input": input_bits,
                "target": bits_to_string(target_bits),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def plateau_detected(history, cfg):
    if len(history) < cfg.PLATEAU_WINDOW:
        return False
    if history[-1]["step"] < cfg.MIN_PILOT_STEPS:
        return False
    if history[-1]["train_exact_accuracy"] < cfg.MIN_TRAIN_EXACT_FOR_PLATEAU:
        return False

    recent = history[-cfg.PLATEAU_WINDOW:]
    allowed_metrics = {
        "monitor_bit_accuracy",
        "monitor_exact_accuracy",
        "monitor_loss",
    }
    if cfg.PLATEAU_METRIC not in allowed_metrics:
        raise ValueError(
            f"PLATEAU_METRIC={cfg.PLATEAU_METRIC!r} 不受支持，"
            f"可选值为 {sorted(allowed_metrics)}。"
        )
    values = np.array(
        [row[cfg.PLATEAU_METRIC] for row in recent],
        dtype=np.float64,
    )
    half = len(values) // 2
    mean_shift = abs(values[half:].mean() - values[:half].mean())
    slope = abs(np.polyfit(np.arange(len(values)), values, 1)[0])

    return (
        mean_shift <= cfg.PLATEAU_MAX_MEAN_SHIFT
        and slope <= cfg.PLATEAU_MAX_SLOPE_PER_EVAL
    )


def train_one_model(
    cfg,
    split,
    model_seed,
    target_steps,
    history_path,
    phase,
    detect_plateau=False,
):
    set_seed(model_seed)
    device = torch.device(cfg.DEVICE)
    input_bits = split["train_x"].shape[1]
    output_bits = split["train_y"].shape[1]

    model = MLP(input_bits, output_bits, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    train_loader = make_train_loader(
        split["train_x"],
        split["train_y"],
        cfg,
        model_seed,
    )

    step = 0
    history = []
    plateau_hits = 0
    iterator = iter(train_loader)

    progress = tqdm(
        total=target_steps,
        desc=f"{phase}, seed={model_seed}",
    )

    while step < target_steps:
        try:
            batch_x, batch_y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch_x, batch_y = next(iterator)

        model.train()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        loss.backward()
        optimizer.step()

        step += 1
        progress.update(1)

        should_evaluate = (
            step == 1
            or step % cfg.EVAL_INTERVAL_STEPS == 0
            or step == target_steps
        )
        if not should_evaluate:
            continue

        train_metrics = evaluate(
            model,
            split["train_x"],
            split["train_y"],
            device,
            cfg.BATCH_SIZE,
        )
        monitor_metrics = evaluate(
            model,
            split["monitor_x"],
            split["monitor_y"],
            device,
            cfg.BATCH_SIZE,
        )
        row = {
            "record_type": "history",
            "phase": phase,
            "model_seed": model_seed,
            "step": step,
            "train_loss": train_metrics["loss"],
            "train_bit_accuracy": train_metrics["bit_accuracy"],
            "train_exact_accuracy": train_metrics["exact_accuracy"],
            "monitor_loss": monitor_metrics["loss"],
            "monitor_bit_accuracy": monitor_metrics["bit_accuracy"],
            "monitor_exact_accuracy": monitor_metrics["exact_accuracy"],
        }
        history.append(row)
        append_jsonl(history_path, row)

        progress.set_postfix({
            "train_exact": f"{train_metrics['exact_accuracy']:.4f}",
            "monitor_bit": f"{monitor_metrics['bit_accuracy']:.4f}",
            "monitor_exact": f"{monitor_metrics['exact_accuracy']:.4f}",
        })

        if detect_plateau:
            if plateau_detected(history, cfg):
                plateau_hits += 1
            else:
                plateau_hits = 0

            if plateau_hits >= cfg.PLATEAU_REQUIRED_WINDOWS:
                print(
                    f"\npilot 在 step={step} 检测到 monitor 平台期，"
                    f"连续命中 {plateau_hits} 个窗口。"
                )
                break

    progress.close()

    train_metrics = evaluate(
        model,
        split["train_x"],
        split["train_y"],
        device,
        cfg.BATCH_SIZE,
    )
    monitor_metrics = evaluate(
        model,
        split["monitor_x"],
        split["monitor_y"],
        device,
        cfg.BATCH_SIZE,
    )
    probe_metrics = evaluate(
        model,
        split["probe_x"],
        split["probe_y"],
        device,
        cfg.BATCH_SIZE,
    )

    return {
        "model": model,
        "final_step": step,
        "train_metrics": train_metrics,
        "monitor_metrics": monitor_metrics,
        "probe_metrics": probe_metrics,
    }


def config_record(cfg, dataset_meta, split):
    record = {
        "record_type": "metadata",
        "dataset": dataset_meta,
        "experiment_name": cfg.EXPERIMENT_NAME,
        "train_count": len(split["train_x"]),
        "monitor_count": len(split["monitor_x"]),
        "probe_count": len(split["probe_x"]),
        "input_bits": int(split["train_x"].shape[1]),
        "output_bits": int(split["train_y"].shape[1]),
        "split_seed": cfg.SPLIT_SEED,
        "model_seeds": list(cfg.MODEL_SEEDS),
        "pilot_seed": cfg.PILOT_SEED,
        "hidden_size": cfg.HIDDEN_SIZE,
        "hidden_layers": cfg.HIDDEN_LAYERS,
        "dropout": cfg.DROPOUT,
        "learning_rate": cfg.LEARNING_RATE,
        "weight_decay": cfg.WEIGHT_DECAY,
        "batch_size": cfg.BATCH_SIZE,
        "plateau_metric": cfg.PLATEAU_METRIC,
        "vary_data_order_by_model_seed": cfg.VARY_DATA_ORDER_BY_MODEL_SEED,
        "prediction_encoding": "row_major_concatenated_bits",
    }
    fingerprint_source = dict(record)
    # 允许续跑时扩充 MODEL_SEEDS，不把 seed 列表纳入配置指纹。
    fingerprint_source.pop("model_seeds")
    serialized = json.dumps(
        fingerprint_source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    record["configuration_fingerprint"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return record


def read_jsonl_records(path):
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"警告：忽略 {path} 第 {line_no} 行的不完整 JSON，"
                    "它可能来自一次中断写入。"
                )
    return records


def main():
    cfg = Config()
    inputs, targets, input_strings, dataset_meta = load_dataset(cfg)
    split = split_dataset(inputs, targets, input_strings, cfg)

    output_dir = Path(cfg.OUTPUT_ROOT) / cfg.EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"
    probe_path = output_dir / "probe.jsonl"
    history_path = output_dir / "training_history.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    stop_step_path = output_dir / "stop_step.jsonl"

    current_metadata = config_record(cfg, dataset_meta, split)
    existing_metadata_records = read_jsonl_records(metadata_path)
    has_existing_experiment = bool(existing_metadata_records)

    if has_existing_experiment and cfg.OVERWRITE_EXISTING_OUTPUT:
        has_existing_experiment = False
    elif has_existing_experiment and not cfg.RESUME_EXISTING_OUTPUT:
        raise FileExistsError(
            f"实验结果已存在：{output_dir}\n"
            "如需续跑，请设置 RESUME_EXISTING_OUTPUT=True；"
            "如需重跑，请设置 OVERWRITE_EXISTING_OUTPUT=True。"
        )

    if has_existing_experiment:
        existing_metadata = existing_metadata_records[0]
        old_fingerprint = existing_metadata.get("configuration_fingerprint")
        new_fingerprint = current_metadata["configuration_fingerprint"]
        if old_fingerprint != new_fingerprint:
            raise ValueError(
                "同名实验的配置与当前配置不一致，拒绝续写。\n"
                "请修改 EXPERIMENT_NAME，或明确设置 "
                "OVERWRITE_EXISTING_OUTPUT=True。"
            )
        if not probe_path.exists():
            raise FileNotFoundError(f"续跑所需的 probe 文件不存在：{probe_path}")
        print(f"检测到配置一致的已有实验，将从中断位置续跑：{output_dir}")
    else:
        metadata_path.write_text(
            json.dumps(current_metadata, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_probe_jsonl(probe_path, split)
        history_path.write_text("", encoding="utf-8")
        predictions_path.write_text("", encoding="utf-8")
        stop_step_path.write_text("", encoding="utf-8")

    print(f"设备：{cfg.DEVICE}")
    print(
        f"数据划分：train={len(split['train_x'])}, "
        f"monitor={len(split['monitor_x'])}, "
        f"probe={len(split['probe_x'])}"
    )
    print(f"输出目录：{output_dir}")

    existing_stop_records = read_jsonl_records(stop_step_path)
    if existing_stop_records:
        saved_steps = int(existing_stop_records[0]["common_train_steps"])
        if (
            cfg.COMMON_TRAIN_STEPS is not None
            and int(cfg.COMMON_TRAIN_STEPS) != saved_steps
        ):
            raise ValueError(
                f"已有实验使用 {saved_steps} steps，"
                f"当前配置要求 {cfg.COMMON_TRAIN_STEPS} steps，无法续写。"
            )
        common_steps = saved_steps
        print(f"复用已有实验的统一停止步数：{common_steps}")
    else:
        common_steps = cfg.COMMON_TRAIN_STEPS

    if common_steps is None:
        pilot_result = train_one_model(
            cfg=cfg,
            split=split,
            model_seed=cfg.PILOT_SEED,
            target_steps=cfg.MAX_PILOT_STEPS,
            history_path=history_path,
            phase="pilot",
            detect_plateau=True,
        )
        common_steps = pilot_result["final_step"]
        stop_record = {
            "record_type": "stop_step",
            "source": "pilot_plateau_or_max_steps",
            "pilot_seed": cfg.PILOT_SEED,
            "common_train_steps": common_steps,
            "pilot_train_metrics": pilot_result["train_metrics"],
            "pilot_monitor_metrics": pilot_result["monitor_metrics"],
            "pilot_probe_metrics": pilot_result["probe_metrics"],
        }
        del pilot_result["model"]
        del pilot_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        stop_record = {
            "record_type": "stop_step",
            "source": "manual_config",
            "common_train_steps": int(common_steps),
        }

    if not existing_stop_records:
        stop_step_path.write_text(
            json.dumps(stop_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"所有正式模型统一训练 {common_steps} steps。")

    if cfg.PILOT_ONLY:
        print("PILOT_ONLY=True，已完成 pilot，不继续训练正式模型。")
        return

    completed_records = read_jsonl_records(predictions_path)
    completed_seeds = {
        int(row["model_seed"])
        for row in completed_records
        if row.get("record_type") == "prediction"
    }
    if completed_seeds:
        print(f"已完成的 seed 将被跳过：{sorted(completed_seeds)}")

    for model_seed in cfg.MODEL_SEEDS:
        if model_seed in completed_seeds:
            continue
        result = train_one_model(
            cfg=cfg,
            split=split,
            model_seed=model_seed,
            target_steps=common_steps,
            history_path=history_path,
            phase="ensemble",
            detect_plateau=False,
        )
        prediction_bits = predict_bit_string(
            result["model"],
            split["probe_x"],
            torch.device(cfg.DEVICE),
            cfg.BATCH_SIZE,
        )

        record = {
            "record_type": "prediction",
            "model_seed": model_seed,
            "train_steps": result["final_step"],
            "train_metrics": result["train_metrics"],
            "monitor_metrics": result["monitor_metrics"],
            "probe_metrics": result["probe_metrics"],
            "probe_count": len(split["probe_x"]),
            "output_bits": int(split["probe_y"].shape[1]),
            "prediction_bits": prediction_bits,
        }
        append_jsonl(predictions_path, record)

        if cfg.SAVE_MODELS:
            torch.save(
                result["model"].state_dict(),
                output_dir / f"model_seed_{model_seed}.pt",
            )

        print(
            f"seed={model_seed} 完成："
            f"train exact={result['train_metrics']['exact_accuracy']:.6f}, "
            f"probe bit={result['probe_metrics']['bit_accuracy']:.6f}, "
            f"probe exact={result['probe_metrics']['exact_accuracy']:.6f}"
        )
        del result["model"]
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n预测结果已保存：{predictions_path}")
    print(
        "下一步运行 analyze_ca_overfit_ensemble.py，"
        "计算跨种子函数相似度与共同错误统计。"
    )


if __name__ == "__main__":
    main()


# %% cell 2


