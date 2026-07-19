# %% cell 1
"""
手动主动采样实验（soft uncertainty 版）：用 seed ensemble 的不确定性选择新增样本。

这个脚本默认不同时跑三种策略，而是让你手动选择一条策略，然后自动推进：

1. ACTION = "run_selected"
   只运行 BRANCH_NAME / ACQUIRE_STRATEGY 指定的一条策略。
   脚本会自动循环：训练到验证集准确率和 agreement 平台期 -> 加一批数据 -> 继续训练。

2. ACTION = "init"
   创建固定 split，并为 uncertain / certain / random 三个分支建立相同初始训练集。

3. ACTION = "train"
   只训练 BRANCH_NAME 指定的分支。脚本会持续保存 checkpoint、验证集/测试集曲线、
   agreement、entropy，并在检测到平台期后停止。

4. ACTION = "acquire"
   在你确认稳定后，按 ACQUIRE_STRATEGY 从 pool 中选一批样本加入当前分支训练集。
   可选策略只有三种：uncertain / certain / random。

本版相对 active_uncertainty_sampling_manual_ca.py 的升级：
- 选样不只看 hard prediction 投票，也会保存每个 seed 的 sigmoid 概率。
- 默认用 BALD = H(mean p) - mean(H(p)) 作为不确定性分数。
- selection_log 会记录 hard entropy、soft entropy、BALD、variance、margin、
  certain-wrong 比例、累计信息量等，方便事后检查“熵的账本”。

推荐用法：
- 把 ACTION 设为 run_selected。
- 把 BRANCH_NAME 和 ACQUIRE_STRATEGY 同时设为 uncertain / certain / random 中的一个。
- 分别跑三次配置，即可得到三条加数据策略曲线。
"""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Config:
    # =========================
    # 手动控制区
    # =========================
    # 可选："init" / "train" / "acquire" / "run_selected"
    # run_selected：只跑 BRANCH_NAME 指定的一条策略，
    # 自动循环“训练到平台期 -> 加数据 -> 继续训练到平台期”。
    ACTION = "run_selected"

    # 可选："uncertain" / "certain" / "random"
    BRANCH_NAME = "certain"

    # ACTION="acquire" 或 "run_selected" 时生效。
    # 可选："uncertain" / "certain" / "random"
    ACQUIRE_STRATEGY = "certain"

    # 每一轮训练的平台期搜索上限；脚本会在稳定性条件满足时提前停止。
    TRAIN_STEPS_THIS_RUN = 50000
    STOP_WHEN_STABLE = True
    ACTIVE_ROUNDS = 2000

    # 如果希望一旦 probe 完整准确率足够高就停止整条策略，把它设为 True。
    STOP_WHEN_PROBE_EXACT_REACHED = False
    TARGET_PROBE_EXACT = 0.999

    # =========================
    # 数据
    # =========================
    DATASET_PATH = (
        "research/overfitting_related_research/datasets/ca_rule30_layer1_len30_n300000.jsonl"
    )
    INPUT_KEY = "input"
    OUTPUT_KEYS = ("output", "target")
    DEDUPLICATE_INPUTS = True

    SPLIT_SEED = 20260714
    INITIAL_TRAIN_COUNT = 200
    POOL_COUNT = 100000
    VAL_COUNT = 5000
    PROBE_COUNT = 5000

    ACQUIRE_BATCH_SIZE = 50
    RANDOM_BRANCH_SEED = 910246

    # 选样分数，可选：
    # - "bald"：推荐。区分“模型们都不确定”和“模型彼此分歧”。
    # - "soft_entropy"：H(mean sigmoid probability)。
    # - "hard_entropy"：旧版 hard vote entropy。
    # - "variance"：seed 间 sigmoid probability 方差。
    # - "soft_margin"：0.5 - |mean probability - 0.5|。
    ACQUIRE_SCORE = "bald"
    SAVE_SELECTED_DETAILS = True

    # =========================
    # 稳定性判定提示
    # =========================
    # 平台期判定不再要求窗口内完全不抖，而是比较前后两个窗口的均值变化。
    # 这样可以容忍 agreement 这类指标在平台期附近自然振荡。
    STABILITY_WINDOW = 5
    STABILITY_MIN_VAL_EXACT = 0.0
    STABILITY_MAX_BIT_MEAN_SHIFT = 0.006
    STABILITY_MAX_EXACT_MEAN_SHIFT = 0.02
    STABILITY_MAX_AGREEMENT_MEAN_SHIFT = 0.02

    # =========================
    # 模型
    # =========================
    MODEL_SEEDS = (0, 1, 2, 3, 4)
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.1

    # =========================
    # 训练
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512
    EVAL_INTERVAL_STEPS = 200
    PREDICT_BATCH_SIZE = 2048

    # =========================
    # 输出
    # =========================
    OUTPUT_ROOT = "research/overfitting_related_research/results_active_soft_uncertainty_manual"
    EXPERIMENT_NAME = "rule30_layer1_soft_active_sampling"
    LIVE_PLOT = True


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
        raise ValueError(f"无法解析 bit 字符串：{value!r}")
    if isinstance(value, (list, tuple)):
        bits = [int(item) for item in value]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"bit 列表中存在非 0/1 值：{value!r}")
        return bits
    raise TypeError(f"不支持的 bit 格式：{type(value)}")


def bits_to_text(bits):
    return "".join("1" if int(bit) else "0" for bit in bits)


def read_dataset(cfg):
    path = Path(cfg.DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据集：{path}")

    records = []
    seen_inputs = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output_key = None
            for key in cfg.OUTPUT_KEYS:
                if key in row:
                    output_key = key
                    break
            if output_key is None:
                raise KeyError(f"第 {line_no} 行找不到 output/target 字段。")

            x_bits = parse_bits(row[cfg.INPUT_KEY])
            y_bits = parse_bits(row[output_key])
            input_text = bits_to_text(x_bits)
            if cfg.DEDUPLICATE_INPUTS and input_text in seen_inputs:
                continue
            seen_inputs.add(input_text)
            records.append({
                "input": input_text,
                "target": bits_to_text(y_bits),
                "x": x_bits,
                "y": y_bits,
                "source_line": line_no,
            })

    if not records:
        raise ValueError(f"数据集为空：{path}")
    input_bits = len(records[0]["x"])
    output_bits = len(records[0]["y"])
    return records, input_bits, output_bits


def experiment_dir(cfg):
    return Path(cfg.OUTPUT_ROOT) / cfg.EXPERIMENT_NAME


def branch_dir(cfg, branch):
    return experiment_dir(cfg) / "branches" / branch


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def initialize_experiment(cfg):
    records, input_bits, output_bits = read_dataset(cfg)
    need = cfg.INITIAL_TRAIN_COUNT + cfg.POOL_COUNT + cfg.VAL_COUNT + cfg.PROBE_COUNT
    if need > len(records):
        raise ValueError(f"需要 {need} 条样本，但数据集只有 {len(records)} 条。")

    root = experiment_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.SPLIT_SEED)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    train = indices[:cfg.INITIAL_TRAIN_COUNT]
    pool_start = cfg.INITIAL_TRAIN_COUNT
    pool_end = pool_start + cfg.POOL_COUNT
    val_end = pool_end + cfg.VAL_COUNT
    probe_end = val_end + cfg.PROBE_COUNT
    split = {
        "train": train,
        "pool": indices[pool_start:pool_end],
        "val": indices[pool_end:val_end],
        "probe": indices[val_end:probe_end],
    }

    metadata = {
        "dataset_path": cfg.DATASET_PATH,
        "input_bits": input_bits,
        "output_bits": output_bits,
        "split_seed": cfg.SPLIT_SEED,
        "initial_train_count": cfg.INITIAL_TRAIN_COUNT,
        "pool_count": cfg.POOL_COUNT,
        "val_count": cfg.VAL_COUNT,
        "probe_count": cfg.PROBE_COUNT,
        "model_seeds": list(cfg.MODEL_SEEDS),
        "note": "Manual active sampling. Branches share the same initial split.",
    }
    json_write(root / "metadata.json", metadata)
    json_write(root / "split.json", split)

    branch_seed_offsets = {"uncertain": 11, "certain": 23, "random": 37}
    for branch in ("uncertain", "certain", "random"):
        state = {
            "branch": branch,
            "round": 0,
            "train_indices": list(train),
            "pool_indices": list(split["pool"]),
            "rng_seed": cfg.RANDOM_BRANCH_SEED + branch_seed_offsets[branch],
            "model_steps": {str(seed): 0 for seed in cfg.MODEL_SEEDS},
            "cumulative_selected_hard_entropy": 0.0,
            "cumulative_selected_soft_entropy": 0.0,
            "cumulative_selected_bald": 0.0,
            "cumulative_selected_variance": 0.0,
            "cumulative_selected_soft_margin": 0.0,
        }
        bdir = branch_dir(cfg, branch)
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        json_write(bdir / "state.json", state)

    print(f"初始化完成：{root}")
    print("下一步：设置 ACTION='train'，BRANCH_NAME='uncertain'/'certain'/'random'。")


def get_records(records, indices):
    return [records[i] for i in indices]


def tensorize_records(records):
    x = torch.tensor([row["x"] for row in records], dtype=torch.float32)
    y = torch.tensor([row["y"] for row in records], dtype=torch.float32)
    return x, y


def make_loader(records, cfg):
    return DataLoader(
        TensorDataset(*tensorize_records(records)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def checkpoint_path(cfg, branch, seed):
    return branch_dir(cfg, branch) / "checkpoints" / f"model_seed{seed}.pt"


def load_or_create_model(cfg, input_bits, output_bits, branch, seed):
    set_seed(seed)
    model = MLP(input_bits, output_bits, cfg).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    path = checkpoint_path(cfg, branch, seed)
    if path.exists():
        ckpt = torch.load(path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return model, optimizer, int(ckpt.get("steps", 0))
    return model, optimizer, 0


def save_checkpoint(cfg, branch, seed, model, optimizer, steps):
    path = checkpoint_path(cfg, branch, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": int(steps),
    }, path)


def predict_prob(model, records, cfg):
    model.eval()
    loader = DataLoader(
        TensorDataset(*tensorize_records(records)),
        batch_size=cfg.PREDICT_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(cfg.DEVICE, non_blocking=True)
            logits = model(xb)
            prob = torch.sigmoid(logits).to(torch.float32).cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0)


def predict_binary(model, records, cfg):
    return (predict_prob(model, records, cfg) >= 0.5).astype(np.uint8)


def ensemble_predict(models, records, cfg):
    return np.stack([predict_binary(model, records, cfg) for model in models], axis=0)


def ensemble_predict_prob(models, records, cfg):
    return np.stack([predict_prob(model, records, cfg) for model in models], axis=0)


def target_array(records):
    return np.array([row["y"] for row in records], dtype=np.uint8)


def binary_entropy(p):
    # 这里必须转成 float64，并使用比 float32 epsilon 大得多的裁剪值。
    # 否则 sigmoid 概率饱和到 1.0 时，1 - 1e-9 在 float32 中仍会被舍入成 1.0，
    # 从而出现 0 * log2(0) = nan，污染 BALD/soft entropy 选样。
    eps = 1e-6
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def ensemble_metrics(preds, target):
    model_bit_acc = (preds == target[None, :, :]).mean(axis=(1, 2))
    model_exact_acc = (preds == target[None, :, :]).all(axis=2).mean(axis=1)
    p = preds.mean(axis=0)
    majority = (p >= 0.5).astype(np.uint8)
    pairwise = None
    if preds.shape[0] >= 2:
        pairwise_values = []
        for i in range(preds.shape[0]):
            for j in range(i + 1, preds.shape[0]):
                pairwise_values.append((preds[i] == preds[j]).mean())
        pairwise = float(np.mean(pairwise_values))
    return {
        "mean_bit_accuracy": float(model_bit_acc.mean()),
        "mean_exact_accuracy": float(model_exact_acc.mean()),
        "majority_bit_accuracy": float((majority == target).mean()),
        "majority_exact_accuracy": float((majority == target).all(axis=1).mean()),
        "pairwise_agreement": pairwise,
        "prediction_entropy_bits": float(binary_entropy(p).mean()),
    }


def uncertainty_scores_from_probs(probs):
    """返回每个样本的多种不确定性分数。

    probs: [model, sample, bit]，是 sigmoid 后的概率。
    """
    hard_preds = (probs >= 0.5).astype(np.uint8)
    hard_p = hard_preds.mean(axis=0)
    mean_p = probs.mean(axis=0)

    hard_entropy = binary_entropy(hard_p).mean(axis=1)
    hard_disagreement = (2.0 * hard_p * (1.0 - hard_p)).mean(axis=1)

    soft_entropy_bits = binary_entropy(mean_p)
    expected_entropy_bits = binary_entropy(probs).mean(axis=0)
    bald_bits = soft_entropy_bits - expected_entropy_bits

    soft_entropy = soft_entropy_bits.mean(axis=1)
    expected_entropy = expected_entropy_bits.mean(axis=1)
    bald = bald_bits.mean(axis=1)
    variance = probs.var(axis=0).mean(axis=1)
    soft_margin = (0.5 - np.abs(mean_p - 0.5)).mean(axis=1)

    return {
        "hard_entropy": hard_entropy,
        "hard_disagreement": hard_disagreement,
        "soft_entropy": soft_entropy,
        "expected_entropy": expected_entropy,
        "bald": bald,
        "variance": variance,
        "soft_margin": soft_margin,
        "mean_probability": mean_p.mean(axis=1),
        "hard_one_rate": hard_p.mean(axis=1),
    }


def uncertainty_scores(preds):
    # 兼容旧逻辑：只基于 hard prediction。
    p = preds.mean(axis=0)
    entropy = binary_entropy(p).mean(axis=1)
    disagreement = (2.0 * p * (1.0 - p)).mean(axis=1)
    return entropy, disagreement


def train_branch(cfg):
    if cfg.BRANCH_NAME not in ("uncertain", "certain", "random"):
        raise ValueError("BRANCH_NAME 只能是 uncertain / certain / random")

    records, input_bits, output_bits = read_dataset(cfg)
    root = experiment_dir(cfg)
    split = json_read(root / "split.json")
    bdir = branch_dir(cfg, cfg.BRANCH_NAME)
    state_path = bdir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"找不到分支状态：{state_path}，请先 ACTION='init'")
    state = json_read(state_path)

    train_records = get_records(records, state["train_indices"])
    val_records = get_records(records, split["val"])
    probe_records = get_records(records, split["probe"])
    loader = make_loader(train_records, cfg)

    models = []
    optimizers = []
    model_steps = {}
    for seed in cfg.MODEL_SEEDS:
        model, optimizer, steps = load_or_create_model(cfg, input_bits, output_bits, cfg.BRANCH_NAME, seed)
        models.append(model)
        optimizers.append(optimizer)
        model_steps[str(seed)] = steps

    curve_jsonl = bdir / "training_curve.jsonl"
    curve_csv = bdir / "training_curve.csv"
    status_path = bdir / "stability_status.json"
    plot_path = bdir / "training_curve.png"

    start_time = time.time()
    trained_steps = 0
    eval_id = 0
    while trained_steps < cfg.TRAIN_STEPS_THIS_RUN:
        interval = min(cfg.EVAL_INTERVAL_STEPS, cfg.TRAIN_STEPS_THIS_RUN - trained_steps)
        for model_index, (model, optimizer) in enumerate(zip(models, optimizers)):
            iterator = iter(loader)
            seed = cfg.MODEL_SEEDS[model_index]
            pbar = tqdm(total=interval, desc=f"{cfg.BRANCH_NAME} seed={seed}", leave=False)
            for _ in range(interval):
                batch, iterator = next_batch(iterator, loader)
                xb, yb = batch
                xb = xb.to(cfg.DEVICE, non_blocking=True)
                yb = yb.to(cfg.DEVICE, non_blocking=True)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                optimizer.step()
                model_steps[str(seed)] = int(model_steps[str(seed)]) + 1
                pbar.update(1)
            pbar.close()
            save_checkpoint(cfg, cfg.BRANCH_NAME, seed, model, optimizer, model_steps[str(seed)])

        trained_steps += interval
        eval_id += 1

        val_preds = ensemble_predict(models, val_records, cfg)
        probe_preds = ensemble_predict(models, probe_records, cfg)
        val_metrics = ensemble_metrics(val_preds, target_array(val_records))
        probe_metrics = ensemble_metrics(probe_preds, target_array(probe_records))
        row = {
            "branch": cfg.BRANCH_NAME,
            "round": state["round"],
            "eval_id": eval_id,
            "train_count": len(state["train_indices"]),
            "pool_count": len(state["pool_indices"]),
            "steps_this_action": trained_steps,
            "min_model_steps_total": min(model_steps.values()),
            "max_model_steps_total": max(model_steps.values()),
            "elapsed_seconds": time.time() - start_time,
        }
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        for key, value in probe_metrics.items():
            row[f"probe_{key}"] = value

        append_jsonl(curve_jsonl, row)
        append_csv_row(curve_csv, row)
        state["model_steps"] = model_steps
        json_write(state_path, state)

        status = compute_stability(load_jsonl(curve_jsonl), cfg)
        json_write(status_path, status)
        if cfg.LIVE_PLOT:
            plot_training_curve(load_jsonl(curve_jsonl), plot_path)

        print(
            f"[{cfg.BRANCH_NAME}] round={state['round']} n={len(state['train_indices'])} "
            f"val_bit={row['val_majority_bit_accuracy']:.4f} "
            f"val_exact={row['val_majority_exact_accuracy']:.4f} "
            f"val_agree={row['val_pairwise_agreement']:.4f} "
            f"probe_bit={row['probe_majority_bit_accuracy']:.4f} "
            f"probe_exact={row['probe_majority_exact_accuracy']:.4f} "
            f"stable={status['is_stable']}"
        )
        if cfg.STOP_WHEN_STABLE and status["is_stable"]:
            print("已检测到验证集准确率和 agreement 进入平台期，本轮训练自动停止。")
            break

    print(f"训练动作完成。曲线：{curve_csv}")
    print(f"稳定性状态：{status_path}")
    print(f"图：{plot_path}")
    return json_read(status_path)


def compute_stability(rows, cfg):
    branch_rows = [row for row in rows if row.get("branch") == cfg.BRANCH_NAME]
    if branch_rows:
        current_round = branch_rows[-1].get("round")
        branch_rows = [row for row in branch_rows if row.get("round") == current_round]
    need = cfg.STABILITY_WINDOW * 2
    if len(branch_rows) < need:
        return {
            "is_stable": False,
            "reason": f"当前 round 记录数不足 {need}",
            "window": len(branch_rows),
        }
    prev = branch_rows[-need:-cfg.STABILITY_WINDOW]
    recent = branch_rows[-cfg.STABILITY_WINDOW:]
    val_bits = [row["val_majority_bit_accuracy"] for row in recent]
    prev_bits = [row["val_majority_bit_accuracy"] for row in prev]
    val_exact = [row["val_majority_exact_accuracy"] for row in recent]
    prev_exact = [row["val_majority_exact_accuracy"] for row in prev]
    agrees = [row["val_pairwise_agreement"] for row in recent]
    prev_agrees = [row["val_pairwise_agreement"] for row in prev]
    bit_span = max(val_bits) - min(val_bits)
    exact_span = max(val_exact) - min(val_exact)
    agreement_span = max(agrees) - min(agrees)
    bit_mean_shift = abs(float(np.mean(val_bits) - np.mean(prev_bits)))
    exact_mean_shift = abs(float(np.mean(val_exact) - np.mean(prev_exact)))
    agreement_mean_shift = abs(float(np.mean(agrees) - np.mean(prev_agrees)))
    min_exact = min(val_exact)
    is_stable = (
        min_exact >= cfg.STABILITY_MIN_VAL_EXACT
        and bit_mean_shift <= cfg.STABILITY_MAX_BIT_MEAN_SHIFT
        and exact_mean_shift <= cfg.STABILITY_MAX_EXACT_MEAN_SHIFT
        and agreement_mean_shift <= cfg.STABILITY_MAX_AGREEMENT_MEAN_SHIFT
    )
    return {
        "is_stable": bool(is_stable),
        "window": len(recent),
        "val_bit_span": float(bit_span),
        "val_exact_span": float(exact_span),
        "val_agreement_span": float(agreement_span),
        "val_bit_mean_shift": float(bit_mean_shift),
        "val_exact_mean_shift": float(exact_mean_shift),
        "val_agreement_mean_shift": float(agreement_mean_shift),
        "min_val_exact": float(min_exact),
        "thresholds": {
            "min_val_exact": cfg.STABILITY_MIN_VAL_EXACT,
            "max_bit_mean_shift": cfg.STABILITY_MAX_BIT_MEAN_SHIFT,
            "max_exact_mean_shift": cfg.STABILITY_MAX_EXACT_MEAN_SHIFT,
            "max_agreement_mean_shift": cfg.STABILITY_MAX_AGREEMENT_MEAN_SHIFT,
        },
    }


def plot_training_curve(rows, output_path):
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"无法导入 matplotlib，跳过绘图：{exc}")
        return

    xs = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    specs = [
        ("val_majority_bit_accuracy", "val majority bit accuracy"),
        ("val_pairwise_agreement", "val pairwise agreement"),
        ("probe_majority_bit_accuracy", "probe majority bit accuracy"),
        ("probe_pairwise_agreement", "probe pairwise agreement"),
    ]
    for ax, (key, title) in zip(axes, specs):
        ax.plot(xs, [row[key] for row in rows], marker="o")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if "accuracy" in key or "agreement" in key:
            ax.set_ylim(0.0, 1.02)
    axes[-1].set_xlabel("eval checkpoint")
    fig.suptitle(f"Manual active sampling branch: {rows[-1]['branch']}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def acquire_samples(cfg):
    if cfg.BRANCH_NAME not in ("uncertain", "certain", "random"):
        raise ValueError("BRANCH_NAME 只能是 uncertain / certain / random")
    if cfg.ACQUIRE_STRATEGY not in ("uncertain", "certain", "random"):
        raise ValueError("ACQUIRE_STRATEGY 只能是 uncertain / certain / random")
    valid_scores = {"hard_entropy", "soft_entropy", "bald", "variance", "soft_margin"}
    if cfg.ACQUIRE_SCORE not in valid_scores:
        raise ValueError(f"ACQUIRE_SCORE 只能是 {sorted(valid_scores)}")

    records, input_bits, output_bits = read_dataset(cfg)
    root = experiment_dir(cfg)
    bdir = branch_dir(cfg, cfg.BRANCH_NAME)
    state_path = bdir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"找不到分支状态：{state_path}，请先 ACTION='init'")
    state = json_read(state_path)

    status_path = bdir / "stability_status.json"
    if status_path.exists():
        status = json_read(status_path)
        print(f"当前稳定性状态：{status}")
    else:
        print("警告：没有 stability_status.json，建议先 ACTION='train'。")

    models = []
    for seed in cfg.MODEL_SEEDS:
        model, _, _ = load_or_create_model(cfg, input_bits, output_bits, cfg.BRANCH_NAME, seed)
        models.append(model)

    pool_indices = list(state["pool_indices"])
    if not pool_indices:
        print("pool 已空，没有可选样本。")
        return
    pool_records = get_records(records, pool_indices)
    pool_probs = ensemble_predict_prob(models, pool_records, cfg)
    pool_preds = (pool_probs >= 0.5).astype(np.uint8)
    scores = uncertainty_scores_from_probs(pool_probs)
    acquire_score = scores[cfg.ACQUIRE_SCORE]
    if not np.all(np.isfinite(acquire_score)):
        bad_count = int((~np.isfinite(acquire_score)).sum())
        raise FloatingPointError(
            f"ACQUIRE_SCORE={cfg.ACQUIRE_SCORE} 出现 {bad_count} 个非有限值。"
            "请检查 sigmoid/logits 或 binary_entropy。"
        )
    count = min(cfg.ACQUIRE_BATCH_SIZE, len(pool_indices))

    if cfg.ACQUIRE_STRATEGY == "uncertain":
        selected_pos = np.argsort(-acquire_score)[:count]
    elif cfg.ACQUIRE_STRATEGY == "certain":
        selected_pos = np.argsort(acquire_score)[:count]
    else:
        rng = random.Random(int(state.get("rng_seed", cfg.RANDOM_BRANCH_SEED)) + int(state["round"]))
        selected_pos = np.array(rng.sample(range(len(pool_indices)), count))

    selected = [pool_indices[int(pos)] for pos in selected_pos]
    selected_targets = target_array(get_records(records, selected))
    hard_vote_p = pool_preds.mean(axis=0)
    selected_majority = (hard_vote_p[selected_pos] >= 0.5).astype(np.uint8)
    selected_bit_correct = selected_majority == selected_targets
    selected_exact_correct = selected_bit_correct.all(axis=1)
    selected_details = []
    if cfg.SAVE_SELECTED_DETAILS:
        for local_i, pos in enumerate(selected_pos):
            pos = int(pos)
            selected_details.append({
                "dataset_index": int(pool_indices[pos]),
                "score": float(acquire_score[pos]),
                "hard_entropy": float(scores["hard_entropy"][pos]),
                "hard_disagreement": float(scores["hard_disagreement"][pos]),
                "soft_entropy": float(scores["soft_entropy"][pos]),
                "expected_entropy": float(scores["expected_entropy"][pos]),
                "bald": float(scores["bald"][pos]),
                "variance": float(scores["variance"][pos]),
                "soft_margin": float(scores["soft_margin"][pos]),
                "mean_probability": float(scores["mean_probability"][pos]),
                "hard_one_rate": float(scores["hard_one_rate"][pos]),
                "majority": bits_to_text(selected_majority[local_i]),
                "target": bits_to_text(selected_targets[local_i]),
                "bit_accuracy": float(selected_bit_correct[local_i].mean()),
                "exact_correct": bool(selected_exact_correct[local_i]),
            })

    selected_sums = {
        "hard_entropy": float(np.sum(scores["hard_entropy"][selected_pos])),
        "soft_entropy": float(np.sum(scores["soft_entropy"][selected_pos])),
        "bald": float(np.sum(scores["bald"][selected_pos])),
        "variance": float(np.sum(scores["variance"][selected_pos])),
        "soft_margin": float(np.sum(scores["soft_margin"][selected_pos])),
    }
    state["cumulative_selected_hard_entropy"] = float(
        state.get("cumulative_selected_hard_entropy", 0.0) + selected_sums["hard_entropy"]
    )
    state["cumulative_selected_soft_entropy"] = float(
        state.get("cumulative_selected_soft_entropy", 0.0) + selected_sums["soft_entropy"]
    )
    state["cumulative_selected_bald"] = float(
        state.get("cumulative_selected_bald", 0.0) + selected_sums["bald"]
    )
    state["cumulative_selected_variance"] = float(
        state.get("cumulative_selected_variance", 0.0) + selected_sums["variance"]
    )
    state["cumulative_selected_soft_margin"] = float(
        state.get("cumulative_selected_soft_margin", 0.0) + selected_sums["soft_margin"]
    )

    selected_set = set(selected)
    state["train_indices"].extend(selected)
    state["pool_indices"] = [idx for idx in pool_indices if idx not in selected_set]
    state["round"] = int(state["round"]) + 1
    json_write(state_path, state)

    record = {
        "new_round": state["round"],
        "branch": cfg.BRANCH_NAME,
        "strategy": cfg.ACQUIRE_STRATEGY,
        "acquire_score": cfg.ACQUIRE_SCORE,
        "selected_count": len(selected),
        "train_count_after": len(state["train_indices"]),
        "pool_count_after": len(state["pool_indices"]),
        "selected_score_mean": float(np.mean(acquire_score[selected_pos])),
        "selected_score_min": float(np.min(acquire_score[selected_pos])),
        "selected_score_max": float(np.max(acquire_score[selected_pos])),
        "selected_hard_entropy_mean": float(np.mean(scores["hard_entropy"][selected_pos])),
        "selected_hard_entropy_sum": selected_sums["hard_entropy"],
        "selected_hard_entropy_min": float(np.min(scores["hard_entropy"][selected_pos])),
        "selected_hard_entropy_max": float(np.max(scores["hard_entropy"][selected_pos])),
        "selected_hard_disagreement_mean": float(np.mean(scores["hard_disagreement"][selected_pos])),
        "selected_soft_entropy_mean": float(np.mean(scores["soft_entropy"][selected_pos])),
        "selected_soft_entropy_sum": selected_sums["soft_entropy"],
        "selected_expected_entropy_mean": float(np.mean(scores["expected_entropy"][selected_pos])),
        "selected_bald_mean": float(np.mean(scores["bald"][selected_pos])),
        "selected_bald_sum": selected_sums["bald"],
        "selected_variance_mean": float(np.mean(scores["variance"][selected_pos])),
        "selected_variance_sum": selected_sums["variance"],
        "selected_soft_margin_mean": float(np.mean(scores["soft_margin"][selected_pos])),
        "selected_soft_margin_sum": selected_sums["soft_margin"],
        "selected_mean_probability_mean": float(np.mean(scores["mean_probability"][selected_pos])),
        "selected_hard_one_rate_mean": float(np.mean(scores["hard_one_rate"][selected_pos])),
        "selected_majority_bit_accuracy": float(np.mean(selected_bit_correct)),
        "selected_majority_exact_accuracy": float(np.mean(selected_exact_correct)),
        "selected_majority_bit_error_rate": float(1.0 - np.mean(selected_bit_correct)),
        "selected_majority_exact_wrong_rate": float(1.0 - np.mean(selected_exact_correct)),
        "selected_certain_wrong_bit_fraction": float(1.0 - np.mean(selected_bit_correct)),
        "selected_certain_wrong_sample_fraction": float(1.0 - np.mean(selected_exact_correct)),
        "cumulative_selected_hard_entropy": state["cumulative_selected_hard_entropy"],
        "cumulative_selected_soft_entropy": state["cumulative_selected_soft_entropy"],
        "cumulative_selected_bald": state["cumulative_selected_bald"],
        "cumulative_selected_variance": state["cumulative_selected_variance"],
        "cumulative_selected_soft_margin": state["cumulative_selected_soft_margin"],
        "selected_indices": selected,
        "selected_details": selected_details,
    }
    append_jsonl(bdir / "selection_log.jsonl", record)
    print(
        "选样完成："
        f"round={record['new_round']}, strategy={record['strategy']}, "
        f"score={record['acquire_score']}, selected={record['selected_count']}, "
        f"score_mean={record['selected_score_mean']:.6f}, "
        f"bald_mean={record['selected_bald_mean']:.6f}, "
        f"soft_entropy_mean={record['selected_soft_entropy_mean']:.6f}, "
        f"wrong_bit={record['selected_majority_bit_error_rate']:.4f}, "
        f"wrong_sample={record['selected_majority_exact_wrong_rate']:.4f}"
    )
    print("下一步：把 ACTION 改成 'train'，继续训练该分支。")
    return record


def run_selected_strategy(cfg):
    if cfg.BRANCH_NAME not in ("uncertain", "certain", "random"):
        raise ValueError("BRANCH_NAME 只能是 uncertain / certain / random")
    if cfg.ACQUIRE_STRATEGY not in ("uncertain", "certain", "random"):
        raise ValueError("ACQUIRE_STRATEGY 只能是 uncertain / certain / random")

    root = experiment_dir(cfg)
    state_path = branch_dir(cfg, cfg.BRANCH_NAME) / "state.json"
    if not state_path.exists():
        if root.exists() and any(root.iterdir()):
            raise FileNotFoundError(
                f"找不到当前分支状态：{state_path}。如果这是旧目录，请先检查，不要自动覆盖。"
            )
        print("未发现实验状态，先执行初始化。")
        initialize_experiment(cfg)

    print(
        f"开始单策略自动循环：branch={cfg.BRANCH_NAME}, "
        f"strategy={cfg.ACQUIRE_STRATEGY}, max_rounds={cfg.ACTIVE_ROUNDS}"
    )
    for loop_id in range(1, cfg.ACTIVE_ROUNDS + 1):
        state = json_read(state_path)
        print(
            f"\n========== 自动轮次 {loop_id}/{cfg.ACTIVE_ROUNDS} | "
            f"round={state['round']} | train={len(state['train_indices'])} | "
            f"pool={len(state['pool_indices'])} =========="
        )

        status = train_branch(cfg)
        if not status.get("is_stable", False):
            print("本轮没有在训练上限内达到平台期，停止自动循环。")
            print("你可以增大 TRAIN_STEPS_THIS_RUN 后继续运行同一策略。")
            break

        rows = load_jsonl(branch_dir(cfg, cfg.BRANCH_NAME) / "training_curve.jsonl")
        last_row = rows[-1] if rows else {}
        probe_exact = float(last_row.get("probe_majority_exact_accuracy", 0.0))
        if cfg.STOP_WHEN_PROBE_EXACT_REACHED and probe_exact >= cfg.TARGET_PROBE_EXACT:
            print(
                f"probe exact={probe_exact:.6f} 已达到目标 "
                f"{cfg.TARGET_PROBE_EXACT:.6f}，停止自动循环。"
            )
            break

        state = json_read(state_path)
        if not state["pool_indices"]:
            print("pool 已空，停止自动循环。")
            break

        acquire_samples(cfg)


def main():
    cfg = Config()
    if cfg.ACTION == "init":
        initialize_experiment(cfg)
    elif cfg.ACTION == "train":
        train_branch(cfg)
    elif cfg.ACTION == "acquire":
        acquire_samples(cfg)
    elif cfg.ACTION == "run_selected":
        run_selected_strategy(cfg)
    else:
        raise ValueError("ACTION 只能是 init / train / acquire / run_selected")


if __name__ == "__main__":
    main()


# %% cell 2


