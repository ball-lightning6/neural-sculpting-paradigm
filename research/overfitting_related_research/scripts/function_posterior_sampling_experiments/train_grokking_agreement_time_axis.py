# %% cell 1
"""
Time-axis agreement experiment for grokking / underdetermined learning.

Core idea:
For a fixed training set, train several independent seeds. At the same training
step, collect their probe predictions and compute agreement / entropy / excess
agreement. This measures how the function posterior changes over training time.

The script is self-contained. Edit Config directly and run it as a normal
Python script.
"""

import csv
import json
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
    # Data
    # =========================
    TASK_NAME = "rule30_layer2"
    DATASET_PATH = (
        "research/overfitting_related_research/datasets/"
        "ca_rule30_layer2_len30_n300000.jsonl"
    )
    DATASET_FALLBACK_PATHS = (
        "research/overfitting_related_research/datasets/ca_rule30_layer2_len30_n300000.jsonl",
        "research/overfitting_related_research/datasets/ca_rule30_layer2_len30_n30000.jsonl",
    )
    INPUT_KEY = "input"
    OUTPUT_KEYS = ("output", "target")
    DEDUPLICATE_INPUTS = True

    SPLIT_SEED = 20260711
    TRAIN_COUNTS = (3000,)
    PROBE_COUNT = 5000

    # =========================
    # Model
    # =========================
    MODEL_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.1

    # =========================
    # Training
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512
    PREDICT_BATCH_SIZE = 4096

    MAX_STEPS = 3000
    # Dense evaluations are useful because a pseudo-rule peak may be short-lived.
    # The main cost is output size, not GPU compute.
    EARLY_EVAL_STEPS = (
        0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50, 75,
        100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000,
    )
    EVAL_INTERVAL_AFTER_EARLY = 50

    SAVE_PREDICTIONS = True

    # =========================
    # Output
    # =========================
    OUTPUT_ROOT = (
        "research/overfitting_related_research/"
        "results_grokking_agreement_time_axis"
    )
    EXPERIMENT_NAME = "rule30_layer2_n3000_time_axis"
    OVERWRITE_OUTPUT = False
    PACKAGE_RESULTS = True
    ZIP_NAME = "grokking_agreement_time_axis_package.zip"


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
            raise ValueError(f"bit list contains non-binary values: {value!r}")
        return bits
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return [int(value)]
    raise TypeError(f"unsupported bit format: {type(value)}")


def bits_to_text(bits):
    return "".join("1" if int(bit) else "0" for bit in bits)


def prediction_to_text(prediction):
    return "".join("1" if int(bit) else "0" for bit in prediction.reshape(-1))


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def resolve_dataset_path(cfg):
    candidates = [Path(cfg.DATASET_PATH)]
    candidates.extend(Path(item) for item in cfg.DATASET_FALLBACK_PATHS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(item) for item in candidates)
    raise FileNotFoundError(f"none of the dataset paths exists:\n{joined}")


def read_dataset(cfg):
    path = resolve_dataset_path(cfg)
    inputs = []
    targets = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            x_bits = parse_bits(record[cfg.INPUT_KEY])
            output_value = None
            for key in cfg.OUTPUT_KEYS:
                if key in record:
                    output_value = record[key]
                    break
            if output_value is None:
                raise KeyError(f"none of output keys found: {cfg.OUTPUT_KEYS}")
            y_bits = parse_bits(output_value)
            key = bits_to_text(x_bits)
            if cfg.DEDUPLICATE_INPUTS and key in seen:
                continue
            seen.add(key)
            inputs.append(x_bits)
            targets.append(y_bits)
    if not inputs:
        raise ValueError(f"empty dataset: {path}")
    return (
        path,
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def make_split(num_samples, train_count, cfg):
    rng = np.random.default_rng(int(cfg.SPLIT_SEED))
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    need = train_count + cfg.PROBE_COUNT
    if need > num_samples:
        raise ValueError(f"dataset too small: need {need}, got {num_samples}")
    train = indices[:train_count].tolist()
    probe = indices[train_count:train_count + cfg.PROBE_COUNT].tolist()
    return train, probe


def make_loader(x_array, y_array, indices, cfg):
    x_tensor = torch.tensor(x_array[indices], dtype=torch.float32)
    y_tensor = torch.tensor(y_array[indices], dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=min(cfg.BATCH_SIZE, len(indices)),
        shuffle=True,
        drop_last=False,
    )


def endless_batches(loader):
    while True:
        for batch in loader:
            yield batch


def make_eval_steps(cfg):
    steps = {int(step) for step in cfg.EARLY_EVAL_STEPS if int(step) <= cfg.MAX_STEPS}
    for step in range(
        cfg.EVAL_INTERVAL_AFTER_EARLY,
        cfg.MAX_STEPS + 1,
        cfg.EVAL_INTERVAL_AFTER_EARLY,
    ):
        steps.add(int(step))
    steps.add(int(cfg.MAX_STEPS))
    return sorted(steps)


def train_one_step(model, optimizer, batch_iter, cfg):
    model.train()
    xb, yb = next(batch_iter)
    xb = xb.to(cfg.DEVICE)
    yb = yb.to(cfg.DEVICE)
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.binary_cross_entropy_with_logits(logits, yb)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def predict_logits(model, x_array, indices, cfg):
    model.eval()
    outputs = []
    for start in range(0, len(indices), cfg.PREDICT_BATCH_SIZE):
        batch_idx = indices[start:start + cfg.PREDICT_BATCH_SIZE]
        xb = torch.tensor(x_array[batch_idx], dtype=torch.float32, device=cfg.DEVICE)
        logits = model(xb).detach().cpu().numpy().astype(np.float32)
        outputs.append(logits)
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def evaluate(model, x_array, y_array, indices, cfg):
    logits = predict_logits(model, x_array, indices, cfg)
    probs = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
    pred = (probs >= 0.5).astype(np.uint8)
    y = y_array[indices].astype(np.uint8)
    bit_acc = float((pred == y).mean())
    exact_acc = float((pred == y).all(axis=1).mean())
    loss = float(
        F.binary_cross_entropy_with_logits(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(y.astype(np.float32), dtype=torch.float32),
        ).item()
    )
    return {
        "loss": loss,
        "bit_accuracy": bit_acc,
        "exact_accuracy": exact_acc,
        "pred": pred,
    }


def binary_entropy(p):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def pairwise_agreement(bits):
    m = bits.shape[0]
    if m < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += float((bits[i] == bits[j]).mean())
            count += 1
    return total / count


def aggregate_step(step, seed_rows, y_probe):
    seeds = sorted(seed_rows)
    pred = np.stack([seed_rows[seed]["probe_pred"] for seed in seeds], axis=0)
    train_metrics = [seed_rows[seed]["train_metrics"] for seed in seeds]
    probe_metrics = [seed_rows[seed]["probe_metrics"] for seed in seeds]

    y = y_probe.astype(np.uint8)
    correct = (pred == y[None, :, :])
    bit_acc = float(correct.mean())
    exact_acc = float(correct.all(axis=2).mean())
    observed_agreement = float(pairwise_agreement(pred))

    vote_p = pred.mean(axis=0)
    majority = (vote_p >= 0.5).astype(np.uint8)
    majority_bit_acc = float((majority == y).mean())
    majority_exact_acc = float((majority == y).all(axis=1).mean())
    entropy = float(binary_entropy(vote_p).mean())
    unanimous = ((vote_p == 0.0) | (vote_p == 1.0))
    unanimously_wrong = unanimous & (majority != y)

    global_base = bit_acc * bit_acc + (1.0 - bit_acc) * (1.0 - bit_acc)
    # Bit-level baseline is included because it is a cheap guard against
    # output-bit heterogeneity.
    bit_acc_by_output = correct.mean(axis=(0, 1))
    bit_base = float(np.mean(bit_acc_by_output ** 2 + (1.0 - bit_acc_by_output) ** 2))

    return {
        "step": int(step),
        "model_count": int(len(seeds)),
        "mean_train_loss": float(np.mean([m["loss"] for m in train_metrics])),
        "mean_train_bit_accuracy": float(np.mean([m["bit_accuracy"] for m in train_metrics])),
        "mean_train_exact_accuracy": float(np.mean([m["exact_accuracy"] for m in train_metrics])),
        "mean_probe_loss": float(np.mean([m["loss"] for m in probe_metrics])),
        "mean_probe_bit_accuracy": bit_acc,
        "mean_probe_exact_accuracy": exact_acc,
        "majority_probe_bit_accuracy": majority_bit_acc,
        "majority_probe_exact_accuracy": majority_exact_acc,
        "direct_pairwise_agreement": observed_agreement,
        "prediction_entropy_bits": entropy,
        "unanimously_same_prediction_bit_fraction": float(unanimous.mean()),
        "unanimously_wrong_bit_fraction": float(unanimously_wrong.mean()),
        "accuracy_baseline_agreement": global_base,
        "excess_agreement": observed_agreement - global_base,
        "bit_level_baseline_agreement": bit_base,
        "bit_level_excess_agreement": observed_agreement - bit_base,
    }


def prepare_output_dir(cfg, train_count):
    root = Path(cfg.OUTPUT_ROOT)
    base_name = f"{cfg.EXPERIMENT_NAME}_n{train_count}_split{cfg.SPLIT_SEED}"
    out_dir = root / base_name
    if cfg.OVERWRITE_OUTPUT:
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in out_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        return out_dir
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    for i in range(1, 1000):
        candidate = root / f"{base_name}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError("cannot create output directory")


def run_one_train_count(train_count, dataset_path, x_array, y_array, cfg):
    out_dir = prepare_output_dir(cfg, train_count)
    train_indices, probe_indices = make_split(len(x_array), train_count, cfg)
    eval_steps = make_eval_steps(cfg)
    y_probe = y_array[probe_indices]

    metadata = {
        "task_name": cfg.TASK_NAME,
        "dataset_path": str(dataset_path),
        "train_count": train_count,
        "probe_count": cfg.PROBE_COUNT,
        "split_seed": cfg.SPLIT_SEED,
        "model_seeds": list(cfg.MODEL_SEEDS),
        "max_steps": cfg.MAX_STEPS,
        "eval_steps": eval_steps,
        "device": cfg.DEVICE,
    }
    write_json(out_dir / "metadata.json", metadata)
    write_jsonl(out_dir / "train_indices.jsonl", [{"index": int(i)} for i in train_indices])
    write_jsonl(out_dir / "probe.jsonl", [
        {
            "probe_offset": int(offset),
            "dataset_index": int(index),
            "input": bits_to_text(x_array[index].astype(np.uint8)),
            "target": bits_to_text(y_array[index].astype(np.uint8)),
        }
        for offset, index in enumerate(probe_indices)
    ])

    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "seed_metrics.jsonl"
    all_step_rows = {step: {} for step in eval_steps}
    seed_summaries = []

    for seed in cfg.MODEL_SEEDS:
        set_seed(seed)
        model = MLP(x_array.shape[1], y_array.shape[1], cfg).to(cfg.DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        loader = make_loader(x_array, y_array, train_indices, cfg)
        batch_iter = endless_batches(loader)

        current_step = 0
        fit_step = None
        pbar = tqdm(total=cfg.MAX_STEPS, desc=f"n={train_count} seed={seed}", leave=False)
        last_loss = None
        for eval_step in eval_steps:
            while current_step < eval_step:
                last_loss = train_one_step(model, optimizer, batch_iter, cfg)
                current_step += 1
                pbar.update(1)

            train_eval = evaluate(model, x_array, y_array, train_indices, cfg)
            probe_eval = evaluate(model, x_array, y_array, probe_indices, cfg)
            if fit_step is None and train_eval["exact_accuracy"] >= 1.0:
                fit_step = current_step

            row = {
                "record_type": "seed_step",
                "task_name": cfg.TASK_NAME,
                "train_count": int(train_count),
                "split_seed": int(cfg.SPLIT_SEED),
                "model_seed": int(seed),
                "step": int(current_step),
                "last_train_batch_loss": last_loss,
                "fit_step": fit_step,
                "train_loss": train_eval["loss"],
                "train_bit_accuracy": train_eval["bit_accuracy"],
                "train_exact_accuracy": train_eval["exact_accuracy"],
                "probe_loss": probe_eval["loss"],
                "probe_bit_accuracy": probe_eval["bit_accuracy"],
                "probe_exact_accuracy": probe_eval["exact_accuracy"],
            }
            append_jsonl(metrics_path, row)
            if cfg.SAVE_PREDICTIONS:
                pred_row = dict(row)
                pred_row["prediction_bits"] = prediction_to_text(probe_eval["pred"])
                append_jsonl(predictions_path, pred_row)

            all_step_rows[eval_step][seed] = {
                "probe_pred": probe_eval["pred"],
                "train_metrics": {
                    "loss": train_eval["loss"],
                    "bit_accuracy": train_eval["bit_accuracy"],
                    "exact_accuracy": train_eval["exact_accuracy"],
                },
                "probe_metrics": {
                    "loss": probe_eval["loss"],
                    "bit_accuracy": probe_eval["bit_accuracy"],
                    "exact_accuracy": probe_eval["exact_accuracy"],
                },
            }
            pbar.set_postfix({
                "train": f"{train_eval['exact_accuracy']:.3f}",
                "probe": f"{probe_eval['bit_accuracy']:.4f}",
            })
        pbar.close()
        seed_summaries.append({
            "model_seed": int(seed),
            "fit_step": fit_step,
            "final_probe_bit_accuracy": probe_eval["bit_accuracy"],
            "final_probe_exact_accuracy": probe_eval["exact_accuracy"],
        })

    curve_rows = []
    for step in eval_steps:
        seed_rows = all_step_rows[step]
        if len(seed_rows) != len(cfg.MODEL_SEEDS):
            continue
        curve_rows.append(aggregate_step(step, seed_rows, y_probe))
    write_jsonl(out_dir / "agreement_time_curve.jsonl", curve_rows)
    write_csv(out_dir / "agreement_time_curve.csv", curve_rows)
    write_jsonl(out_dir / "seed_summaries.jsonl", seed_summaries)
    write_csv(out_dir / "seed_summaries.csv", seed_summaries)
    make_plot(out_dir, curve_rows, train_count)

    if cfg.PACKAGE_RESULTS:
        zip_path = out_dir / cfg.ZIP_NAME
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in out_dir.rglob("*"):
                if path == zip_path or path.is_dir():
                    continue
                zf.write(path, path.relative_to(out_dir))
        print(f"package: {zip_path}")
    print(f"output: {out_dir}")
    print("last row:", curve_rows[-1] if curve_rows else None)
    return out_dir


def make_plot(out_dir, rows, train_count):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip plot: {exc}")
        return
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)

    axes[0, 0].plot(steps, [row["mean_train_exact_accuracy"] for row in rows], label="train exact")
    axes[0, 0].plot(steps, [row["mean_probe_exact_accuracy"] for row in rows], label="probe exact")
    axes[0, 0].plot(steps, [row["majority_probe_exact_accuracy"] for row in rows], label="majority exact")
    axes[0, 0].set_title("exact accuracy")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, [row["mean_probe_bit_accuracy"] for row in rows], label="probe bit")
    axes[0, 1].plot(steps, [row["majority_probe_bit_accuracy"] for row in rows], label="majority bit")
    axes[0, 1].set_title("bit accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, [row["direct_pairwise_agreement"] for row in rows], label="raw agreement")
    axes[1, 0].plot(steps, [row["accuracy_baseline_agreement"] for row in rows], label="accuracy baseline")
    axes[1, 0].plot(steps, [row["bit_level_baseline_agreement"] for row in rows], label="bit baseline")
    axes[1, 0].set_title("agreement")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, [row["excess_agreement"] for row in rows], label="global excess")
    axes[1, 1].plot(steps, [row["bit_level_excess_agreement"] for row in rows], label="bit-level excess")
    axes[1, 1].plot(steps, [row["prediction_entropy_bits"] for row in rows], label="entropy")
    axes[1, 1].set_title("excess agreement / entropy")
    axes[1, 1].legend()

    for ax in axes.reshape(-1):
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{Config.TASK_NAME}, n={train_count}")
    fig.tight_layout()
    fig.savefig(out_dir / "agreement_time_axis.png")
    plt.close(fig)


def main():
    cfg = Config()
    start = time.time()
    dataset_path, x_array, y_array = read_dataset(cfg)
    output_dirs = []
    for train_count in cfg.TRAIN_COUNTS:
        output_dirs.append(run_one_train_count(train_count, dataset_path, x_array, y_array, cfg))
    print(f"done in {(time.time() - start) / 60:.2f} min")
    print("outputs:")
    for out_dir in output_dirs:
        print(out_dir)


if __name__ == "__main__":
    main()


# %% cell 2


