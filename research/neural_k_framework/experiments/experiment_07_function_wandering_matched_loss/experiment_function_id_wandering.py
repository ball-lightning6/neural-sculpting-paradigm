"""
4-bit Boolean 条件函数的 function-ID 时间游走实验。

固定一个包含 6 个训练点的平衡训练集。训练集首次 hard-fit 后，剩余 10 个
未见输入的 hard predictions 恰好构成一个 10-bit state ID，因此条件函数空间
只有 2^10=1024 个状态，可以直接观察：

- 单个 seed 在哪些具体函数之间迁移；
- 多 seed 函数分布是否继续漂移、分叉或达到稳定；
- 单个函数是否仍在运动，而总体分布已经近似平稳；
- 状态的流入率、逃逸率、停留时间与转移边。

主实验使用 full-batch Adam 和无 weight decay，避免把函数运动简单归因于
minibatch 噪声。所有轨迹按每个 seed 首次 hard-fit 的时刻对齐。
"""

from __future__ import annotations

import csv
import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def script_directory() -> Path:
    source = globals().get("__file__")
    if source:
        return Path(source).resolve().parent
    return Path.cwd()


class Config:
    PROFILE = "pilot"  # "full" / "pilot" / "smoke"
    RESULT_DIR = script_directory() / "results_function_id_wandering"
    CREATE_ZIP = True

    INPUT_BITS = 4
    TRAIN_INDICES = (0, 3, 5, 10, 12, 15)
    TRAIN_TARGETS = (0, 1, 0, 1, 0, 1)

    # width 16 只适合链路 pilot；既有宽度扫描显示 width 64 的精细后验
    # 已明显更接近 width 128，因此主实验使用 64 x 3。
    HIDDEN_SIZE = 64
    HIDDEN_LAYERS = 3
    LEARNING_RATE = 3e-3
    WEIGHT_DECAY = 0.0

    PILOT_MODELS = 1_024
    PILOT_MAX_PREFIT_STEPS = 10_000
    PILOT_MAX_POSTFIT_AGE = 20_000

    FULL_MODELS = 4_096
    FULL_MAX_PREFIT_STEPS = 20_000
    FULL_MAX_POSTFIT_AGE = 50_000

    GLOBAL_SEED = 20260818
    LOG_INTERVAL_STEPS = 500
    MAX_EVENT_ROWS = 2_000_000

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOW_TF32 = False


@dataclass(frozen=True)
class EffectiveConfig:
    profile: str
    result_dir: Path
    input_bits: int
    train_indices: tuple[int, ...]
    train_targets: tuple[int, ...]
    heldout_indices: tuple[int, ...]
    hidden_size: int
    hidden_layers: int
    learning_rate: float
    weight_decay: float
    model_count: int
    max_prefit_steps: int
    max_postfit_age: int
    record_ages: tuple[int, ...]
    logit_ages: tuple[int, ...]
    device: str
    allow_tf32: bool
    smoke_test: bool


def build_record_ages(max_age: int) -> tuple[int, ...]:
    values = set(range(0, min(max_age, 100) + 1))
    values.update(range(110, min(max_age, 1_000) + 1, 10))
    values.update(range(1_100, min(max_age, 10_000) + 1, 100))
    values.update(range(11_000, max_age + 1, 1_000))
    values.add(max_age)
    return tuple(sorted(value for value in values if value <= max_age))


def resolve_config() -> EffectiveConfig:
    profile = str(Config.PROFILE).strip().lower()
    if profile == "full":
        model_count = Config.FULL_MODELS
        max_prefit = Config.FULL_MAX_PREFIT_STEPS
        max_postfit = Config.FULL_MAX_POSTFIT_AGE
        hidden_size = Config.HIDDEN_SIZE
        hidden_layers = Config.HIDDEN_LAYERS
        smoke = False
    elif profile == "pilot":
        model_count = Config.PILOT_MODELS
        max_prefit = Config.PILOT_MAX_PREFIT_STEPS
        max_postfit = Config.PILOT_MAX_POSTFIT_AGE
        hidden_size = Config.HIDDEN_SIZE
        hidden_layers = Config.HIDDEN_LAYERS
        smoke = False
    elif profile == "smoke":
        model_count = 64
        max_prefit = 500
        max_postfit = 500
        hidden_size = 16
        hidden_layers = 2
        smoke = True
    else:
        raise ValueError("PROFILE 只能是 full/pilot/smoke。")

    domain = 1 << Config.INPUT_BITS
    train_indices = tuple(int(value) for value in Config.TRAIN_INDICES)
    train_targets = tuple(int(value) for value in Config.TRAIN_TARGETS)
    if len(train_indices) != len(train_targets):
        raise ValueError("TRAIN_INDICES 与 TRAIN_TARGETS 长度不一致。")
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("TRAIN_INDICES 不能重复。")
    if any(value < 0 or value >= domain for value in train_indices):
        raise ValueError("TRAIN_INDICES 超出输入空间。")
    heldout = tuple(value for value in range(domain) if value not in train_indices)
    logit_ages = tuple(
        value
        for value in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000,
                      5_000, 10_000, 20_000, 50_000)
        if value <= max_postfit
    )
    return EffectiveConfig(
        profile=profile,
        result_dir=Path(Config.RESULT_DIR),
        input_bits=Config.INPUT_BITS,
        train_indices=train_indices,
        train_targets=train_targets,
        heldout_indices=heldout,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        model_count=model_count,
        max_prefit_steps=max_prefit,
        max_postfit_age=max_postfit,
        record_ages=build_record_ages(max_postfit),
        logit_ages=logit_ages,
        device=Config.DEVICE,
        allow_tf32=Config.ALLOW_TF32,
        smoke_test=smoke,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key, "")) for key in fields})


def truth_table_inputs(input_bits: int) -> torch.Tensor:
    values = torch.arange(1 << input_bits, dtype=torch.int64)
    shifts = torch.arange(input_bits, dtype=torch.int64)
    return ((values[:, None] >> shifts) & 1).to(torch.float32)


def entropy_bits(probability: np.ndarray) -> float:
    values = probability[probability > 0]
    return float(-(values * np.log2(values)).sum()) if values.size else 0.0


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = first / max(float(first.sum()), 1e-300)
    second = second / max(float(second.sum()), 1e-300)
    middle = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        valid = left > 0
        return float(np.sum(left[valid] * np.log2(left[valid] / right[valid])))

    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


class EnsembleLinear(nn.Module):
    def __init__(self, count: int, in_features: int, out_features: int, generator: torch.Generator):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        weight = torch.empty(count, out_features, in_features)
        bias = torch.empty(count, out_features)
        weight.uniform_(-bound, bound, generator=generator)
        bias.uniform_(-bound, bound, generator=generator)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.bmm(inputs, self.weight.transpose(1, 2)) + self.bias[:, None, :]


class BatchedTanhMLP(nn.Module):
    def __init__(self, cfg: EffectiveConfig):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(Config.GLOBAL_SEED)
        layers: list[nn.Module] = []
        width = cfg.input_bits
        for _ in range(cfg.hidden_layers):
            layers.append(EnsembleLinear(cfg.model_count, width, cfg.hidden_size, generator))
            width = cfg.hidden_size
        self.layers = nn.ModuleList(layers)
        self.output = EnsembleLinear(cfg.model_count, width, 1, generator)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.output(hidden)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    all_inputs: torch.Tensor,
    train_indices: torch.Tensor,
    train_targets: torch.Tensor,
    heldout_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits = model(all_inputs).squeeze(-1)
    train_logits = logits[:, train_indices]
    signed_targets = train_targets * 2.0 - 1.0
    margins = train_logits * signed_targets[None, :]
    loss = F.binary_cross_entropy_with_logits(
        train_logits,
        train_targets[None, :].expand_as(train_logits),
        reduction="none",
    ).mean(dim=1)
    domain_powers = 2 ** torch.arange(logits.shape[1], device=logits.device, dtype=torch.int64)
    full_ids = ((logits >= 0).to(torch.int64) * domain_powers[None, :]).sum(dim=1)
    heldout_logits = logits[:, heldout_indices]
    state_powers = 2 ** torch.arange(
        heldout_logits.shape[1], device=logits.device, dtype=torch.int64
    )
    state_ids = (
        (heldout_logits >= 0).to(torch.int64) * state_powers[None, :]
    ).sum(dim=1)
    return {
        "logits": logits,
        "full_ids": full_ids,
        "state_ids": state_ids,
        "loss": loss,
        "min_margin": margins.min(dim=1).values,
        "exact": (margins > 0).all(dim=1),
    }


def run_training(cfg: EffectiveConfig) -> dict[str, np.ndarray]:
    device = torch.device(cfg.device)
    all_cpu = truth_table_inputs(cfg.input_bits)
    all_inputs = all_cpu.to(device)[None, :, :].expand(cfg.model_count, -1, -1)
    train_indices = torch.tensor(cfg.train_indices, dtype=torch.int64, device=device)
    heldout_indices = torch.tensor(cfg.heldout_indices, dtype=torch.int64, device=device)
    train_targets = torch.tensor(cfg.train_targets, dtype=torch.float32, device=device)

    model = BatchedTanhMLP(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    ages = np.asarray(cfg.record_ages, dtype=np.int64)
    age_to_position = {int(age): position for position, age in enumerate(ages)}
    logit_ages = np.asarray(cfg.logit_ages, dtype=np.int64)
    logit_age_to_position = {int(age): position for position, age in enumerate(logit_ages)}

    shape = (len(ages), cfg.model_count)
    recorded = np.zeros(shape, dtype=bool)
    full_ids_out = np.zeros(shape, dtype=np.uint16)
    state_ids_out = np.zeros(shape, dtype=np.uint16)
    train_loss_out = np.full(shape, np.nan, dtype=np.float32)
    min_margin_out = np.full(shape, np.nan, dtype=np.float32)
    train_exact_out = np.zeros(shape, dtype=bool)
    logits_out = np.full(
        (len(logit_ages), cfg.model_count, 1 << cfg.input_bits),
        np.nan,
        dtype=np.float16,
    )

    first_fit_steps = np.full(cfg.model_count, -1, dtype=np.int64)
    transition_count = np.zeros(cfg.model_count, dtype=np.int32)
    previous_state = np.zeros(cfg.model_count, dtype=np.uint16)
    previous_state_valid = np.zeros(cfg.model_count, dtype=bool)
    last_change_age = np.zeros(cfg.model_count, dtype=np.int64)
    state_count = 1 << len(cfg.heldout_indices)
    transition_matrix = np.zeros((state_count, state_count), dtype=np.int64)
    event_rows: list[dict[str, Any]] = []
    last_log_step = 0
    last_log_transition_count = transition_count.copy()

    max_total_steps = cfg.max_prefit_steps + cfg.max_postfit_age
    started = time.perf_counter()
    print(
        f"训练：models={cfg.model_count:,} | full batch={len(cfg.train_indices)} | "
        f"max prefit={cfg.max_prefit_steps:,} | max postfit age={cfg.max_postfit_age:,}"
    )

    for step in range(max_total_steps + 1):
        snapshot = evaluate(
            model, all_inputs, train_indices, train_targets, heldout_indices
        )
        exact_np = snapshot["exact"].cpu().numpy()
        state_np = snapshot["state_ids"].cpu().numpy().astype(np.uint16)
        newly_fitted = (first_fit_steps < 0) & exact_np
        first_fit_steps[newly_fitted] = step
        previous_state[newly_fitted] = state_np[newly_fitted]
        previous_state_valid[newly_fitted] = True
        last_change_age[newly_fitted] = 0

        fitted = first_fit_steps >= 0
        current_age = np.where(fitted, step - first_fit_steps, -1)

        comparable = previous_state_valid & fitted & (current_age > 0)
        changed = comparable & (state_np != previous_state)
        if np.any(changed):
            indices = np.flatnonzero(changed)
            np.add.at(
                transition_matrix,
                (previous_state[indices].astype(np.int64), state_np[indices].astype(np.int64)),
                1,
            )
            transition_count[indices] += 1
            if len(event_rows) < Config.MAX_EVENT_ROWS:
                remaining = Config.MAX_EVENT_ROWS - len(event_rows)
                for index in indices[:remaining]:
                    event_rows.append(
                        {
                            "model_index": int(index),
                            "global_step": step,
                            "post_fit_age": int(current_age[index]),
                            "from_state_id": int(previous_state[index]),
                            "to_state_id": int(state_np[index]),
                            "dwell_steps": int(current_age[index] - last_change_age[index]),
                            "train_exact": bool(exact_np[index]),
                        }
                    )
            last_change_age[indices] = current_age[indices]
            previous_state[indices] = state_np[indices]

        positions_needed = np.flatnonzero(
            fitted & np.isin(current_age, ages)
        )
        if positions_needed.size:
            full_np = snapshot["full_ids"].cpu().numpy().astype(np.uint16)
            loss_np = snapshot["loss"].cpu().numpy().astype(np.float32)
            margin_np = snapshot["min_margin"].cpu().numpy().astype(np.float32)
            logits_np: np.ndarray | None = None
            for index in positions_needed:
                age = int(current_age[index])
                position = age_to_position[age]
                recorded[position, index] = True
                full_ids_out[position, index] = full_np[index]
                state_ids_out[position, index] = state_np[index]
                train_loss_out[position, index] = loss_np[index]
                min_margin_out[position, index] = margin_np[index]
                train_exact_out[position, index] = exact_np[index]
                if age in logit_age_to_position:
                    if logits_np is None:
                        logits_np = snapshot["logits"].cpu().numpy().astype(np.float16)
                    logits_out[logit_age_to_position[age], index] = logits_np[index]

        fitted_count = int(fitted.sum())
        minimum_age = int(current_age[fitted].min()) if fitted_count else -1
        if step % Config.LOG_INTERVAL_STEPS == 0 or step == max_total_steps:
            elapsed = time.perf_counter() - started
            mean_transitions = float(transition_count[fitted].mean()) if fitted_count else 0.0
            window_delta = transition_count - last_log_transition_count
            window_transitions = int(window_delta.sum())
            window_changed_models = int(np.count_nonzero(window_delta))
            window_steps = max(step - last_log_step, 1)
            hazard_per_1k = (
                1_000.0 * window_transitions / max(cfg.model_count * window_steps, 1)
            )
            print(
                f"  step={step:6d} | fitted={fitted_count:,}/{cfg.model_count:,} | "
                f"youngest_age={minimum_age:,} | transitions/model={mean_transitions:.2f} | "
                f"window=+{window_transitions} ({window_changed_models} models) | "
                f"hazard={hazard_per_1k:.4f}/1k-model-step | "
                f"{step/max(elapsed,1e-9):.1f} step/s"
            )
            last_log_step = step
            last_log_transition_count = transition_count.copy()

        if fitted_count == cfg.model_count and minimum_age >= cfg.max_postfit_age:
            break
        if step == max_total_steps:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(all_inputs[:, train_indices, :]).squeeze(-1)
        targets = train_targets[None, :].expand_as(train_logits)
        per_model_loss = F.binary_cross_entropy_with_logits(
            train_logits, targets, reduction="none"
        ).mean(dim=1)
        per_model_loss.sum().backward()
        optimizer.step()

    rows, cols = np.nonzero(transition_matrix)
    edge_rows = [
        {
            "from_state_id": int(row),
            "to_state_id": int(col),
            "count": int(transition_matrix[row, col]),
            "hamming_distance": int((int(row) ^ int(col)).bit_count()),
        }
        for row, col in zip(rows, cols)
    ]
    write_csv(cfg.result_dir / "analysis" / "transition_events.csv", event_rows)
    write_csv(cfg.result_dir / "analysis" / "transition_edges.csv", edge_rows)

    arrays = {
        "post_fit_ages": ages,
        "logit_ages": logit_ages,
        "recorded": recorded,
        "full_function_ids": full_ids_out,
        "heldout_state_ids": state_ids_out,
        "train_loss": train_loss_out,
        "train_min_margin": min_margin_out,
        "train_exact": train_exact_out,
        "first_fit_steps": first_fit_steps,
        "transition_count": transition_count,
        "logits": logits_out,
        "heldout_indices": np.asarray(cfg.heldout_indices, dtype=np.int64),
    }
    np.savez_compressed(cfg.result_dir / "trajectories.npz", **arrays)
    return arrays


def analyze(cfg: EffectiveConfig, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    ages = arrays["post_fit_ages"]
    state_count = 1 << len(cfg.heldout_indices)
    truth = (
        (np.arange(state_count)[:, None] >> np.arange(len(cfg.heldout_indices))) & 1
    ).astype(np.float64)
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    previous_probability: np.ndarray | None = None
    previous_ids: np.ndarray | None = None
    previous_valid: np.ndarray | None = None

    for position, age_value in enumerate(ages):
        valid = arrays["recorded"][position]
        ids = arrays["heldout_state_ids"][position, valid].astype(np.int64)
        if ids.size == 0:
            continue
        counts = np.bincount(ids, minlength=state_count).astype(np.int64)
        probability = counts / counts.sum()
        point_probability = probability @ truth
        agreement = float(
            np.mean(np.square(point_probability) + np.square(1.0 - point_probability))
        )
        change_rate = math.nan
        if previous_ids is not None and previous_valid is not None:
            paired = valid & previous_valid
            change_rate = float(
                np.mean(
                    arrays["heldout_state_ids"][position, paired]
                    != previous_ids[paired]
                )
            )
        row = {
            "post_fit_age": int(age_value),
            "model_count": int(valid.sum()),
            "train_exact_fraction": float(arrays["train_exact"][position, valid].mean()),
            "mean_train_loss": float(np.nanmean(arrays["train_loss"][position, valid])),
            "mean_min_margin": float(np.nanmean(arrays["train_min_margin"][position, valid])),
            "unique_state_ids": int(np.count_nonzero(counts)),
            "state_entropy_bits": entropy_bits(probability),
            "effective_state_support": float(1.0 / np.square(probability).sum()),
            "top_state_id": int(np.argmax(probability)),
            "top_state_mass": float(probability.max()),
            "heldout_pairwise_agreement": agreement,
            "js_from_previous_age_bits": js_divergence(probability, previous_probability)
            if previous_probability is not None
            else math.nan,
            "hard_change_rate_from_previous_age": change_rate,
            "mean_transition_count": float(
                arrays["transition_count"][valid].mean()
            ),
        }
        summary_rows.append(row)
        for state_id, count in enumerate(counts):
            if count:
                distribution_rows.append(
                    {
                        "post_fit_age": int(age_value),
                        "state_id": state_id,
                        "count": int(count),
                        "probability": float(probability[state_id]),
                    }
                )
        previous_probability = probability
        previous_ids = arrays["heldout_state_ids"][position].copy()
        previous_valid = valid.copy()

    write_csv(cfg.result_dir / "analysis" / "posterior_by_age.csv", summary_rows)
    write_csv(cfg.result_dir / "analysis" / "state_distributions.csv", distribution_rows)
    create_plots(cfg, summary_rows, distribution_rows, arrays)

    fitted = arrays["first_fit_steps"] >= 0
    final_row = summary_rows[-1] if summary_rows else {}
    summary = {
        "protocol_version": "function_id_wandering_v1",
        "model_count": cfg.model_count,
        "fitted_count": int(fitted.sum()),
        "fitted_fraction": float(fitted.mean()),
        "first_fit_step_median": float(np.median(arrays["first_fit_steps"][fitted]))
        if np.any(fitted)
        else math.nan,
        "models_with_any_transition": int(
            np.count_nonzero(arrays["transition_count"][fitted])
        ),
        "transition_fraction": float(
            np.mean(arrays["transition_count"][fitted] > 0)
        )
        if np.any(fitted)
        else math.nan,
        "mean_transition_count": float(arrays["transition_count"][fitted].mean())
        if np.any(fitted)
        else math.nan,
        "median_transition_count": float(np.median(arrays["transition_count"][fitted]))
        if np.any(fitted)
        else math.nan,
        "final": final_row,
    }
    save_json(cfg.result_dir / "summary.json", summary)
    return summary


def create_plots(
    cfg: EffectiveConfig,
    summary_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"跳过作图：{error}")
        return

    plot_dir = cfg.result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ages = np.asarray([row["post_fit_age"] for row in summary_rows])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("state_entropy_bits", "state entropy (bits)"),
        ("heldout_pairwise_agreement", "heldout agreement"),
        ("hard_change_rate_from_previous_age", "ID change rate"),
        ("mean_transition_count", "mean transitions / model"),
    ]
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        axis.plot(ages, [row[metric] for row in summary_rows], marker="o", markersize=3)
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("post-fit age")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "wandering_overview.png", dpi=170)
    plt.close(figure)

    frame = distribution_rows
    max_by_state: dict[int, float] = {}
    for row in frame:
        state_id = int(row["state_id"])
        max_by_state[state_id] = max(max_by_state.get(state_id, 0.0), row["probability"])
    top_states = sorted(max_by_state, key=max_by_state.get, reverse=True)[:10]
    figure, axis = plt.subplots(figsize=(10, 6))
    for state_id in top_states:
        rows = [row for row in frame if int(row["state_id"]) == state_id]
        rows.sort(key=lambda row: row["post_fit_age"])
        axis.plot(
            [row["post_fit_age"] for row in rows],
            [row["probability"] for row in rows],
            marker="o",
            markersize=3,
            label=f"ID {state_id}",
        )
    axis.set_xscale("symlog", linthresh=10)
    axis.set_xlabel("post-fit age")
    axis.set_ylabel("state probability")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "top_state_probability_paths.png", dpi=170)
    plt.close(figure)

    # 展示前 64 个成功模型相对于最终众数的 Hamming 距离时间线。
    final_valid = arrays["recorded"][-1]
    final_ids = arrays["heldout_state_ids"][-1, final_valid].astype(np.int64)
    if final_ids.size:
        modal = int(np.bincount(final_ids).argmax())
        selected = np.flatnonzero(arrays["first_fit_steps"] >= 0)[:64]
        matrix = np.full((len(selected), len(arrays["post_fit_ages"])), np.nan)
        for row_index, model_index in enumerate(selected):
            valid = arrays["recorded"][:, model_index]
            values = arrays["heldout_state_ids"][valid, model_index].astype(np.int64)
            matrix[row_index, np.flatnonzero(valid)] = [
                (int(value) ^ modal).bit_count() for value in values
            ]
        figure, axis = plt.subplots(figsize=(14, 7))
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_xlabel("recorded post-fit age index")
        axis.set_ylabel("model")
        axis.set_title(f"Hamming distance to final modal state ID {modal}")
        figure.colorbar(image, ax=axis, label="Hamming distance")
        figure.tight_layout()
        figure.savefig(plot_dir / "seed_trajectory_raster.png", dpi=170)
        plt.close(figure)


def create_zip(cfg: EffectiveConfig) -> Path:
    archive = cfg.result_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(cfg.result_dir.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=path.relative_to(cfg.result_dir))
    return archive


def main() -> None:
    cfg = resolve_config()
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["result_dir"] = str(cfg.result_dir)
    payload["protocol_version"] = "function_id_wandering_v1"
    save_json(cfg.result_dir / "config.json", payload)

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)

    print("=== Function-ID Wandering ===")
    print(f"设备：{cfg.device}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"结果目录：{cfg.result_dir}")
    print(
        f"配置：4->{cfg.hidden_size}x{cfg.hidden_layers}->1 tanh | "
        f"models={cfg.model_count:,} | train={list(zip(cfg.train_indices, cfg.train_targets))} | "
        f"heldout states=2^{len(cfg.heldout_indices)}={1<<len(cfg.heldout_indices):,}"
    )

    arrays = run_training(cfg)
    summary = analyze(cfg, arrays)
    archive = create_zip(cfg) if Config.CREATE_ZIP else None

    print("\n=== Function-ID 游走实验完成 ===")
    print(
        f"fitted={summary['fitted_count']:,}/{summary['model_count']:,} | "
        f"transition_fraction={summary['transition_fraction']:.6f} | "
        f"mean/median transitions={summary['mean_transition_count']:.2f}/"
        f"{summary['median_transition_count']:.1f}"
    )
    print(f"汇总：{cfg.result_dir / 'summary.json'}")
    if archive is not None:
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
