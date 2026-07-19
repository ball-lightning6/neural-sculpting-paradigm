# %% cell 1
"""
Active sampling with a single-seed time committee.

Hypothesis:
After one model reaches a loose plateau, predictions sampled from the same
training trajectory at separated time steps can act like a small function
posterior committee. If this is useful, active sampling by committee
uncertainty should still show:

    uncertain > random > certain

The script is self-contained. Edit Config directly and run it as a normal
Python script. It runs all three strategies by default.
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
    TASK_NAME = "rule30_layer1"
    DATASET_PATH = (
        "research/overfitting_related_research/datasets/"
        "ca_rule30_layer1_len30_n300000.jsonl"
    )
    DATASET_FALLBACK_PATHS = (
        "research/overfitting_related_research/datasets/ca_rule30_layer1_len30_n300000.jsonl",
        "research/overfitting_related_research/datasets/ca_rule30_layer1_len30_n30000.jsonl",
    )
    INPUT_KEY = "input"
    OUTPUT_KEYS = ("output", "target")
    DEDUPLICATE_INPUTS = True

    SPLIT_SEED = 20260711
    INITIAL_TRAIN_COUNT = 200
    PROBE_COUNT = 5000
    POOL_COUNT = 20000

    # =========================
    # Active sampling
    # =========================
    STRATEGIES = ("uncertain", "random", "certain")
    INITIAL_MODEL_SEED = 0
    RANDOM_BRANCH_SEED = 910246
    ACTIVE_ROUNDS = 45
    ACQUIRE_BATCH_SIZE = 50

    # Score used by uncertain/certain.
    # Options: "hard_entropy", "soft_entropy", "bald", "variance",
    # "soft_margin".
    ACQUIRE_SCORE = "hard_entropy"

    STOP_WHEN_TARGET_REACHED = True
    TARGET_PROBE_EXACT = 0.999

    # =========================
    # Model
    # =========================
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

    # Loose plateau. We only need enough stability to choose the next batch.
    EVAL_INTERVAL_STEPS = 100
    MAX_TRAIN_STEPS_PER_ROUND = 6000
    TRAIN_EXACT_TARGET = 1.0
    MIN_STEPS_AFTER_TRAIN_FIT = 400
    RECENT_WINDOW = 5
    MAX_RECENT_PROBE_BIT_RANGE = 0.008
    MAX_RECENT_PROBE_AGREEMENT_RANGE = 0.04

    # Time committee sampled after the loose plateau.
    TIME_COMMITTEE_SIZE = 8
    TIME_COMMITTEE_INTERVAL_STEPS = 300

    # =========================
    # Output
    # =========================
    OUTPUT_ROOT = (
        "research/overfitting_related_research/"
        "results_active_time_committee_sampling"
    )
    EXPERIMENT_NAME = "rule30_layer1_time_committee_active_sampling"
    OVERWRITE_OUTPUT = True
    PACKAGE_RESULTS = True
    ZIP_NAME = "active_time_committee_sampling_package.zip"


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


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


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


def read_dataset(path, cfg):
    path = resolve_dataset_path(path, cfg)
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
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def resolve_dataset_path(path, cfg):
    candidates = [Path(path)]
    candidates.extend(Path(item) for item in getattr(cfg, "DATASET_FALLBACK_PATHS", ()))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(item) for item in candidates)
    raise FileNotFoundError(f"none of the dataset paths exists:\n{joined}")


def make_split(num_samples, cfg):
    rng = np.random.default_rng(int(cfg.SPLIT_SEED))
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    need = cfg.INITIAL_TRAIN_COUNT + cfg.PROBE_COUNT + cfg.POOL_COUNT
    if need > num_samples:
        raise ValueError(
            f"dataset too small: need {need}, got {num_samples}"
        )
    train = indices[:cfg.INITIAL_TRAIN_COUNT].tolist()
    probe = indices[cfg.INITIAL_TRAIN_COUNT:cfg.INITIAL_TRAIN_COUNT + cfg.PROBE_COUNT].tolist()
    pool_start = cfg.INITIAL_TRAIN_COUNT + cfg.PROBE_COUNT
    pool = indices[pool_start:pool_start + cfg.POOL_COUNT].tolist()
    return train, probe, pool


def make_model(input_bits, output_bits, cfg):
    model = MLP(input_bits, output_bits, cfg).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    return model, optimizer


def make_loader(x_array, y_array, indices, cfg, shuffle=True):
    x_tensor = torch.tensor(x_array[indices], dtype=torch.float32)
    y_tensor = torch.tensor(y_array[indices], dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=min(cfg.BATCH_SIZE, len(indices)),
        shuffle=shuffle,
        drop_last=False,
    )


def endless_batches(loader):
    while True:
        for batch in loader:
            yield batch


def train_steps(model, optimizer, batch_iter, steps, cfg):
    model.train()
    last_loss = None
    for _ in range(int(steps)):
        xb, yb = next(batch_iter)
        xb = xb.to(cfg.DEVICE)
        yb = yb.to(cfg.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
    return last_loss


@torch.no_grad()
def predict_probs(model, x_array, indices, cfg):
    model.eval()
    out = []
    for start in range(0, len(indices), cfg.PREDICT_BATCH_SIZE):
        batch_idx = indices[start:start + cfg.PREDICT_BATCH_SIZE]
        xb = torch.tensor(x_array[batch_idx], dtype=torch.float32, device=cfg.DEVICE)
        probs = torch.sigmoid(model(xb)).detach().cpu().numpy().astype(np.float32)
        out.append(probs)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def evaluate_single(model, x_array, y_array, indices, cfg):
    probs = predict_probs(model, x_array, indices, cfg)
    pred = (probs >= 0.5).astype(np.uint8)
    y = y_array[indices].astype(np.uint8)
    bit_acc = float((pred == y).mean())
    exact_acc = float((pred == y).all(axis=1).mean())
    return {
        "bit_accuracy": bit_acc,
        "exact_accuracy": exact_acc,
        "probs": probs,
        "pred": pred,
    }


def binary_entropy(p):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def committee_scores(probs):
    hard = (probs >= 0.5).astype(np.float32)
    hard_p = hard.mean(axis=0)
    mean_p = probs.mean(axis=0)

    hard_entropy_bits = binary_entropy(hard_p)
    soft_entropy_bits = binary_entropy(mean_p)
    expected_entropy_bits = binary_entropy(probs).mean(axis=0)

    scores = {
        "hard_entropy": hard_entropy_bits.mean(axis=1),
        "soft_entropy": soft_entropy_bits.mean(axis=1),
        "bald": (soft_entropy_bits - expected_entropy_bits).mean(axis=1),
        "variance": probs.var(axis=0).mean(axis=1),
        "soft_margin": (0.5 - np.abs(mean_p - 0.5)).mean(axis=1),
    }
    return scores


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


def committee_summary(probs, y):
    bits = (probs >= 0.5).astype(np.uint8)
    y_uint = y.astype(np.uint8)
    model_bit_acc = float((bits == y_uint[None, :, :]).mean())
    model_exact_acc = float((bits == y_uint[None, :, :]).all(axis=2).mean())
    vote_p = bits.mean(axis=0)
    majority = (vote_p >= 0.5).astype(np.uint8)
    majority_bit_acc = float((majority == y_uint).mean())
    majority_exact_acc = float((majority == y_uint).all(axis=1).mean())
    entropy = float(binary_entropy(vote_p).mean())
    agreement = float(pairwise_agreement(bits))
    unanimous = float(((vote_p == 0.0) | (vote_p == 1.0)).mean())
    return {
        "model_bit_accuracy": model_bit_acc,
        "model_exact_accuracy": model_exact_acc,
        "majority_bit_accuracy": majority_bit_acc,
        "majority_exact_accuracy": majority_exact_acc,
        "direct_pairwise_agreement": agreement,
        "mean_prediction_entropy_bits": entropy,
        "unanimous_bit_fraction": unanimous,
    }


def recent_stable(history, cfg):
    if len(history) < cfg.RECENT_WINDOW:
        return False
    recent = history[-cfg.RECENT_WINDOW:]
    probe_bits = [row["probe_bit_accuracy"] for row in recent]
    probe_agrees = [row["probe_agreement"] for row in recent]
    bit_range = max(probe_bits) - min(probe_bits)
    agree_range = max(probe_agrees) - min(probe_agrees)
    return (
        bit_range <= cfg.MAX_RECENT_PROBE_BIT_RANGE
        and agree_range <= cfg.MAX_RECENT_PROBE_AGREEMENT_RANGE
    )


def train_to_loose_plateau(
    model,
    optimizer,
    batch_iter,
    x_array,
    y_array,
    train_indices,
    probe_indices,
    cfg,
    total_steps,
    branch_name,
    round_index,
):
    fit_step = None
    after_fit_steps = 0
    history = []
    pbar = tqdm(
        total=cfg.MAX_TRAIN_STEPS_PER_ROUND,
        desc=f"{branch_name} round={round_index} n={len(train_indices)}",
        leave=False,
    )
    for local_step in range(0, cfg.MAX_TRAIN_STEPS_PER_ROUND, cfg.EVAL_INTERVAL_STEPS):
        loss = train_steps(model, optimizer, batch_iter, cfg.EVAL_INTERVAL_STEPS, cfg)
        total_steps += cfg.EVAL_INTERVAL_STEPS
        pbar.update(cfg.EVAL_INTERVAL_STEPS)

        train_eval = evaluate_single(model, x_array, y_array, train_indices, cfg)
        probe_eval = evaluate_single(model, x_array, y_array, probe_indices, cfg)

        # A single model has agreement 1.0; here we use it only for a loose
        # consistency placeholder before the time committee is sampled.
        row = {
            "step": total_steps,
            "local_step": local_step + cfg.EVAL_INTERVAL_STEPS,
            "loss": loss,
            "train_bit_accuracy": train_eval["bit_accuracy"],
            "train_exact_accuracy": train_eval["exact_accuracy"],
            "probe_bit_accuracy": probe_eval["bit_accuracy"],
            "probe_exact_accuracy": probe_eval["exact_accuracy"],
            "probe_agreement": 1.0,
        }
        history.append(row)

        if fit_step is None and train_eval["exact_accuracy"] >= cfg.TRAIN_EXACT_TARGET:
            fit_step = total_steps
            after_fit_steps = 0
        if fit_step is not None:
            after_fit_steps = total_steps - fit_step

        pbar.set_postfix({
            "train": f"{train_eval['exact_accuracy']:.3f}",
            "probe": f"{probe_eval['bit_accuracy']:.4f}",
        })

        if fit_step is not None and after_fit_steps >= cfg.MIN_STEPS_AFTER_TRAIN_FIT:
            if recent_stable(history, cfg) or after_fit_steps >= cfg.MIN_STEPS_AFTER_TRAIN_FIT:
                pbar.close()
                return total_steps, {
                    "fit_step": fit_step,
                    "stop_reason": "loose_fit_stable",
                    "last_eval": row,
                    "local_history": history,
                }
    pbar.close()
    return total_steps, {
        "fit_step": fit_step,
        "stop_reason": "max_round_steps",
        "last_eval": history[-1] if history else {},
        "local_history": history,
    }


def sample_time_committee(
    model,
    optimizer,
    batch_iter,
    x_array,
    pool_indices,
    probe_indices,
    cfg,
    total_steps,
):
    pool_probs = []
    probe_probs = []
    sample_steps = []
    for _ in range(cfg.TIME_COMMITTEE_SIZE):
        train_steps(model, optimizer, batch_iter, cfg.TIME_COMMITTEE_INTERVAL_STEPS, cfg)
        total_steps += cfg.TIME_COMMITTEE_INTERVAL_STEPS
        pool_probs.append(predict_probs(model, x_array, pool_indices, cfg))
        probe_probs.append(predict_probs(model, x_array, probe_indices, cfg))
        sample_steps.append(total_steps)
    return (
        np.stack(pool_probs, axis=0),
        np.stack(probe_probs, axis=0),
        sample_steps,
        total_steps,
    )


def choose_indices(strategy, scores, cfg, rng):
    n = len(next(iter(scores.values())))
    k = min(cfg.ACQUIRE_BATCH_SIZE, n)
    if strategy == "random":
        return np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64)
    score = np.asarray(scores[cfg.ACQUIRE_SCORE])
    if strategy == "uncertain":
        order = np.argsort(-score)
    elif strategy == "certain":
        order = np.argsort(score)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return order[:k].astype(np.int64)


def selected_stats(local_indices, scores, pool_probs, pool_targets, cfg):
    selected_probs = pool_probs[:, local_indices, :]
    selected_targets = pool_targets[local_indices].astype(np.uint8)
    selected_bits = (selected_probs >= 0.5).astype(np.uint8)
    vote = (selected_bits.mean(axis=0) >= 0.5).astype(np.uint8)
    wrong_bits = (vote != selected_targets)
    row = {
        "selected_count": int(len(local_indices)),
        "selected_wrong_bit_fraction": float(wrong_bits.mean()),
        "selected_wrong_sample_fraction": float(wrong_bits.any(axis=1).mean()),
    }
    for name, values in scores.items():
        chosen = np.asarray(values)[local_indices]
        row[f"selected_{name}_mean"] = float(np.mean(chosen))
        row[f"selected_{name}_min"] = float(np.min(chosen))
        row[f"selected_{name}_max"] = float(np.max(chosen))
    return row


def run_branch(strategy, x_array, y_array, initial_train, probe, pool, cfg, out_dir):
    branch_dir = out_dir / "branches" / strategy
    branch_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.INITIAL_MODEL_SEED)
    model, optimizer = make_model(x_array.shape[1], y_array.shape[1], cfg)

    train_indices = list(initial_train)
    pool_indices = list(pool)
    strategy_offsets = {"uncertain": 11, "random": 29, "certain": 47}
    rng = np.random.default_rng(cfg.RANDOM_BRANCH_SEED + strategy_offsets[strategy])
    total_steps = 0
    curve_rows = []
    selection_rows = []

    for round_index in range(cfg.ACTIVE_ROUNDS + 1):
        loader = make_loader(x_array, y_array, train_indices, cfg, shuffle=True)
        batch_iter = endless_batches(loader)
        total_steps, plateau_info = train_to_loose_plateau(
            model,
            optimizer,
            batch_iter,
            x_array,
            y_array,
            train_indices,
            probe,
            cfg,
            total_steps,
            strategy,
            round_index,
        )

        pool_probs, probe_probs, sample_steps, total_steps = sample_time_committee(
            model,
            optimizer,
            batch_iter,
            x_array,
            pool_indices,
            probe,
            cfg,
            total_steps,
        )
        probe_summary = committee_summary(probe_probs, y_array[probe])

        curve_row = {
            "strategy": strategy,
            "round": int(round_index),
            "train_count": int(len(train_indices)),
            "pool_count": int(len(pool_indices)),
            "total_steps": int(total_steps),
            "fit_step": plateau_info.get("fit_step"),
            "stop_reason": plateau_info.get("stop_reason"),
            "sample_first_step": int(sample_steps[0]),
            "sample_last_step": int(sample_steps[-1]),
            "probe_bit_accuracy": probe_summary["model_bit_accuracy"],
            "probe_exact_accuracy": probe_summary["model_exact_accuracy"],
            "majority_probe_bit_accuracy": probe_summary["majority_bit_accuracy"],
            "majority_probe_exact_accuracy": probe_summary["majority_exact_accuracy"],
            "probe_agreement": probe_summary["direct_pairwise_agreement"],
            "probe_entropy": probe_summary["mean_prediction_entropy_bits"],
            "probe_unanimous_bit_fraction": probe_summary["unanimous_bit_fraction"],
        }
        append_jsonl(branch_dir / "training_curve.jsonl", curve_row)
        curve_rows.append(curve_row)

        print(
            f"[{strategy}] round={round_index} n={len(train_indices)} "
            f"probe_bit={curve_row['probe_bit_accuracy']:.4f} "
            f"probe_exact={curve_row['probe_exact_accuracy']:.4f} "
            f"agree={curve_row['probe_agreement']:.4f}"
        )

        if (
            cfg.STOP_WHEN_TARGET_REACHED
            and probe_summary["model_exact_accuracy"] >= cfg.TARGET_PROBE_EXACT
        ):
            break
        if round_index >= cfg.ACTIVE_ROUNDS or len(pool_indices) == 0:
            break

        scores = committee_scores(pool_probs)
        local_selected = choose_indices(strategy, scores, cfg, rng)
        pool_targets = y_array[pool_indices]
        stats = selected_stats(local_selected, scores, pool_probs, pool_targets, cfg)

        selected_global = [pool_indices[int(i)] for i in local_selected]
        selected_set = set(selected_global)
        pool_indices = [idx for idx in pool_indices if idx not in selected_set]
        train_indices.extend(selected_global)

        selection_row = {
            "strategy": strategy,
            "round": int(round_index),
            "train_count_before": int(len(train_indices) - len(selected_global)),
            "train_count_after": int(len(train_indices)),
            "acquire_score": cfg.ACQUIRE_SCORE,
            "selected_global_indices": selected_global,
        }
        selection_row.update(stats)
        append_jsonl(branch_dir / "selection_log.jsonl", selection_row)
        selection_rows.append(selection_row)

    write_json(branch_dir / "final_state.json", {
        "strategy": strategy,
        "final_train_count": len(train_indices),
        "final_pool_count": len(pool_indices),
        "total_steps": total_steps,
        "curve_rows": len(curve_rows),
        "selection_rounds": len(selection_rows),
    })
    write_csv(branch_dir / "training_curve.csv", curve_rows)
    write_csv(branch_dir / "selection_log.csv", selection_rows)
    return curve_rows, selection_rows


def make_plot(out_dir, all_curve_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip plot: {exc}")
        return

    colors = {
        "uncertain": "#4C78A8",
        "random": "#F58518",
        "certain": "#E45756",
    }
    metrics = [
        ("probe_exact_accuracy", "probe exact accuracy"),
        ("probe_bit_accuracy", "probe bit accuracy"),
        ("probe_agreement", "time-committee agreement"),
        ("probe_entropy", "time-committee entropy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.reshape(-1)
    for ax, (key, title) in zip(axes, metrics):
        for strategy in sorted({row["strategy"] for row in all_curve_rows}):
            rows = [row for row in all_curve_rows if row["strategy"] == strategy]
            rows.sort(key=lambda r: r["train_count"])
            x = [row["train_count"] for row in rows]
            y = [row[key] for row in rows]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors.get(strategy),
                label=strategy,
            )
        ax.set_title(title)
        ax.set_xlabel("train count")
        ax.grid(True, alpha=0.25)
        if "accuracy" in key or "agreement" in key:
            ax.set_ylim(0, 1.02)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "active_time_committee_curves.png")
    plt.close(fig)


def package_results(out_dir, cfg):
    zip_path = out_dir / cfg.ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(out_dir))
    return zip_path


def main():
    cfg = Config()
    out_dir = Path(cfg.OUTPUT_ROOT) / cfg.EXPERIMENT_NAME
    if cfg.OVERWRITE_OUTPUT and out_dir.exists():
        # Keep this conservative: remove only files produced by this script.
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)

    x_array, y_array = read_dataset(cfg.DATASET_PATH, cfg)
    initial_train, probe, pool = make_split(len(x_array), cfg)

    metadata = {
        "task_name": cfg.TASK_NAME,
        "dataset_path": cfg.DATASET_PATH,
        "split_seed": cfg.SPLIT_SEED,
        "initial_train_count": cfg.INITIAL_TRAIN_COUNT,
        "probe_count": cfg.PROBE_COUNT,
        "pool_count": cfg.POOL_COUNT,
        "strategies": list(cfg.STRATEGIES),
        "acquire_score": cfg.ACQUIRE_SCORE,
        "time_committee_size": cfg.TIME_COMMITTEE_SIZE,
        "time_committee_interval_steps": cfg.TIME_COMMITTEE_INTERVAL_STEPS,
        "initial_model_seed": cfg.INITIAL_MODEL_SEED,
        "device": cfg.DEVICE,
    }
    write_json(out_dir / "metadata.json", metadata)
    write_jsonl(out_dir / "initial_train.jsonl", [{"index": int(i)} for i in initial_train])
    write_jsonl(out_dir / "probe.jsonl", [{"index": int(i)} for i in probe])

    all_curve_rows = []
    all_selection_rows = []
    start = time.time()
    for strategy in cfg.STRATEGIES:
        curve_rows, selection_rows = run_branch(
            strategy,
            x_array,
            y_array,
            initial_train,
            probe,
            pool,
            cfg,
            out_dir,
        )
        all_curve_rows.extend(curve_rows)
        all_selection_rows.extend(selection_rows)

    write_jsonl(out_dir / "training_curves_all.jsonl", all_curve_rows)
    write_csv(out_dir / "training_curves_all.csv", all_curve_rows)
    write_jsonl(out_dir / "selection_log_all.jsonl", all_selection_rows)
    write_csv(out_dir / "selection_log_all.csv", all_selection_rows)

    summary_rows = []
    for strategy in cfg.STRATEGIES:
        rows = [row for row in all_curve_rows if row["strategy"] == strategy]
        rows.sort(key=lambda r: r["train_count"])
        if not rows:
            continue
        final = rows[-1]
        row = {
            "strategy": strategy,
            "final_train_count": final["train_count"],
            "final_probe_bit_accuracy": final["probe_bit_accuracy"],
            "final_probe_exact_accuracy": final["probe_exact_accuracy"],
            "final_probe_agreement": final["probe_agreement"],
            "final_probe_entropy": final["probe_entropy"],
        }
        for threshold in (0.8, 0.9, 0.95, 0.99, 0.999):
            hit = next(
                (r for r in rows if r["probe_exact_accuracy"] >= threshold),
                None,
            )
            row[f"first_train_count_exact_ge_{threshold}"] = (
                hit["train_count"] if hit else None
            )
        summary_rows.append(row)
    write_jsonl(out_dir / "summary.jsonl", summary_rows)
    write_csv(out_dir / "summary.csv", summary_rows)

    make_plot(out_dir, all_curve_rows)
    if cfg.PACKAGE_RESULTS:
        zip_path = package_results(out_dir, cfg)
        print(f"package: {zip_path}")
    print(f"done in {(time.time() - start) / 60:.2f} min")
    print(f"output: {out_dir}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()


# %% cell 2


