"""
Function-ID 游走后验与同 loss 静态先验的直接比较。

脚本读取 experiment_function_id_wandering.py 的结果配置，为完全相同的网络
架构和训练约束重新大规模采样未训练权重。对每个 post-fit age 比较：

1. SGD 经验函数分布 Q_t(f)；
2. hard-exact 先验中满足 L <= mean_loss(t) 的微正则子水平集；
3. exp(-beta L) 退火族中平均 loss 与 SGD 匹配的正则分布；
4. 整个可靠 beta 网格中与 SGD 的 JS 最接近分布。

由此可以直接判断：SGD 的函数迁移是否只是“进入相同 loss 的静态区域”，
还是仍存在无法由 loss 水平解释的定向概率运输。
"""

from __future__ import annotations

import csv
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def script_directory() -> Path:
    source = globals().get("__file__")
    if source:
        return Path(source).resolve().parent
    return Path.cwd()


class Config:
    PROFILE = "pilot"  # "full" / "pilot" / "smoke"
    WANDERING_RESULT_DIR = Path("/root/results_function_id_wandering")
    RESULT_DIR = Path("/root/results_wandering_matched_loss_prior")
    CREATE_ZIP = True

    PILOT_PRIOR_MODELS = 4_194_304
    FULL_PRIOR_MODELS = 16_777_216
    MICRO_BATCH = 32_768
    SHARD_SIZE = 262_144
    GLOBAL_SEED = 20260819
    MIN_RELIABLE_COUNT = 200
    MIN_RELIABLE_ESS = 200.0

    BETAS = (
        0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0,
        48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0,
        1_024.0, 1_536.0, 2_048.0, 3_072.0, 4_096.0,
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth_inputs(input_bits: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(1 << input_bits, dtype=torch.int64, device=device)
    shifts = torch.arange(input_bits, dtype=torch.int64, device=device)
    return ((values[:, None] >> shifts) & 1).to(torch.float32)


@torch.inference_mode()
def sample_logits(
    count: int,
    inputs: torch.Tensor,
    hidden_size: int,
    hidden_layers: int,
    generator: torch.Generator,
) -> torch.Tensor:
    hidden = inputs[None, :, :].expand(count, -1, -1)
    width = inputs.shape[1]
    for _ in range(hidden_layers):
        bound = 1.0 / math.sqrt(width)
        weight = torch.empty(count, hidden_size, width, device=inputs.device).uniform_(
            -bound, bound, generator=generator
        )
        bias = torch.empty(count, hidden_size, device=inputs.device).uniform_(
            -bound, bound, generator=generator
        )
        hidden = torch.tanh(torch.bmm(hidden, weight.transpose(1, 2)) + bias[:, None, :])
        del weight, bias
        width = hidden_size
    bound = 1.0 / math.sqrt(width)
    weight = torch.empty(count, 1, width, device=inputs.device).uniform_(
        -bound, bound, generator=generator
    )
    bias = torch.empty(count, 1, device=inputs.device).uniform_(
        -bound, bound, generator=generator
    )
    return torch.bmm(hidden, weight.transpose(1, 2)).squeeze(-1) + bias


def sample_prior(
    wandering_config: dict[str, Any],
    model_count: int,
    result_dir: Path,
) -> dict[str, np.ndarray]:
    aggregate = result_dir / "prior_samples.npz"
    metadata_path = result_dir / "prior_samples.json"
    signature = json.dumps(
        {
            "model_count": model_count,
            "input_bits": wandering_config["input_bits"],
            "hidden_size": wandering_config["hidden_size"],
            "hidden_layers": wandering_config["hidden_layers"],
            "train_indices": wandering_config["train_indices"],
            "train_targets": wandering_config["train_targets"],
            "global_seed": Config.GLOBAL_SEED,
        },
        sort_keys=True,
    )
    if aggregate.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            print("复用 matched prior。")
            with np.load(aggregate, allow_pickle=False) as loaded:
                return {key: loaded[key] for key in loaded.files}

    device = torch.device(Config.DEVICE)
    input_bits = int(wandering_config["input_bits"])
    hidden_size = int(wandering_config["hidden_size"])
    hidden_layers = int(wandering_config["hidden_layers"])
    train_indices = np.asarray(wandering_config["train_indices"], dtype=np.int64)
    heldout_indices = np.asarray(wandering_config["heldout_indices"], dtype=np.int64)
    targets = np.asarray(wandering_config["train_targets"], dtype=np.float64)
    signed = targets * 2.0 - 1.0
    inputs = truth_inputs(input_bits, device)
    state_powers = 2 ** np.arange(len(heldout_indices), dtype=np.uint16)

    shard_dir = result_dir / "prior_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_size = min(Config.SHARD_SIZE, model_count)
    shard_count = model_count // shard_size
    if model_count % shard_size:
        raise ValueError("PRIOR_MODELS 必须能被 SHARD_SIZE 整除。")

    print(
        f"采样 matched prior：models={model_count:,} | network="
        f"{input_bits}->{hidden_size}x{hidden_layers}->1 tanh"
    )
    started = time.perf_counter()
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        end = start + shard_size
        path = shard_dir / f"prior_{start:09d}_{end:09d}.npz"
        if path.exists():
            continue
        state_parts: list[np.ndarray] = []
        loss_parts: list[np.ndarray] = []
        normalized_parts: list[np.ndarray] = []
        rms_parts: list[np.ndarray] = []
        hard_parts: list[np.ndarray] = []
        for local_start in range(0, shard_size, Config.MICRO_BATCH):
            global_start = start + local_start
            generator = torch.Generator(device=device)
            generator.manual_seed(Config.GLOBAL_SEED + global_start * 1_000_003)
            logits_t = sample_logits(
                Config.MICRO_BATCH, inputs, hidden_size, hidden_layers, generator
            )
            logits = logits_t.cpu().numpy().astype(np.float32)
            train_logits = logits[:, train_indices].astype(np.float64)
            margins = train_logits * signed[None, :]
            loss = np.logaddexp(0.0, -margins).mean(axis=1)
            rms = np.sqrt(np.mean(np.square(logits.astype(np.float64)), axis=1))
            normalized_loss = np.logaddexp(
                0.0, -(margins / np.maximum(rms[:, None], 1e-12))
            ).mean(axis=1)
            state = (
                (logits[:, heldout_indices] >= 0).astype(np.uint16)
                * state_powers[None, :]
            ).sum(axis=1, dtype=np.uint16)
            state_parts.append(state)
            loss_parts.append(loss.astype(np.float32))
            normalized_parts.append(normalized_loss.astype(np.float32))
            rms_parts.append(rms.astype(np.float32))
            hard_parts.append(np.all(margins > 0, axis=1))
            del logits_t
        np.savez_compressed(
            path,
            state_ids=np.concatenate(state_parts),
            train_loss=np.concatenate(loss_parts),
            normalized_loss=np.concatenate(normalized_parts),
            logit_rms=np.concatenate(rms_parts),
            hard_exact=np.concatenate(hard_parts),
        )
        if shard_index == 0 or (shard_index + 1) % max(1, shard_count // 10) == 0:
            completed = end
            rate = completed / max(time.perf_counter() - started, 1e-9)
            print(
                f"  {completed:,}/{model_count:,} | {rate:,.0f} models/s | "
                f"ETA={(model_count-completed)/max(rate,1e-9):.1f}s"
            )

    arrays: dict[str, list[np.ndarray]] = {
        "state_ids": [], "train_loss": [], "normalized_loss": [],
        "logit_rms": [], "hard_exact": []
    }
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        end = start + shard_size
        with np.load(shard_dir / f"prior_{start:09d}_{end:09d}.npz", allow_pickle=False) as loaded:
            for key in arrays:
                arrays[key].append(loaded[key])
    result = {key: np.concatenate(parts) for key, parts in arrays.items()}
    np.savez_compressed(aggregate, **result)
    save_json(
        metadata_path,
        {
            "signature": signature,
            "model_count": model_count,
            "hard_exact_count": int(result["hard_exact"].sum()),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return result


def distribution(ids: np.ndarray, weights: np.ndarray, state_count: int) -> np.ndarray:
    values = np.bincount(ids.astype(np.int64), weights=weights, minlength=state_count)
    return values / max(float(values.sum()), 1e-300)


def entropy_bits(probability: np.ndarray) -> float:
    values = probability[probability > 0]
    return float(-(values * np.log2(values)).sum()) if values.size else 0.0


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    middle = 0.5 * (first + second)
    def kl(left: np.ndarray, right: np.ndarray) -> float:
        valid = left > 0
        return float(np.sum(left[valid] * np.log2(left[valid] / right[valid])))
    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


def distribution_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    point = probability @ truth
    return {
        "entropy_bits": entropy_bits(probability),
        "effective_support": float(1.0 / np.square(probability).sum()),
        "top_state_id": int(np.argmax(probability)),
        "top_state_mass": float(probability.max()),
        "pairwise_agreement": float(
            np.mean(np.square(point) + np.square(1.0 - point))
        ),
    }


def beta_weights(loss: np.ndarray, beta: float) -> tuple[np.ndarray, float, float]:
    shifted = loss - float(loss.min())
    log_weights = -float(beta) * shifted
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    total = float(weights.sum())
    normalized = weights / total
    ess = total**2 / max(float(np.square(weights).sum()), 1e-300)
    mean_loss = float(np.sum(normalized * loss))
    return normalized, ess, mean_loss


def analyze(
    wandering_dir: Path,
    result_dir: Path,
    wandering_config: dict[str, Any],
    prior: dict[str, np.ndarray],
) -> dict[str, Any]:
    posterior_rows = read_csv(wandering_dir / "analysis" / "posterior_by_age.csv")
    state_rows = read_csv(wandering_dir / "analysis" / "state_distributions.csv")
    heldout_count = len(wandering_config["heldout_indices"])
    state_count = 1 << heldout_count
    truth = (
        (np.arange(state_count)[:, None] >> np.arange(heldout_count)) & 1
    ).astype(np.float64)
    q_by_age: dict[int, np.ndarray] = {}
    for row in state_rows:
        age = int(row["post_fit_age"])
        q_by_age.setdefault(age, np.zeros(state_count, dtype=np.float64))
        q_by_age[age][int(row["state_id"])] = float(row["probability"])

    hard = prior["hard_exact"].astype(bool)
    prior_ids = prior["state_ids"][hard]
    raw_loss = prior["train_loss"][hard].astype(np.float64)
    normalized_loss = prior["normalized_loss"][hard].astype(np.float64)
    hard_probability = distribution(
        prior_ids, np.ones(len(prior_ids)), state_count
    )

    beta_cache: dict[tuple[str, float], tuple[np.ndarray, float, float]] = {}
    for family, loss in (("raw", raw_loss), ("normalized", normalized_loss)):
        for beta in Config.BETAS:
            weights, ess, mean_loss = beta_weights(loss, float(beta))
            beta_cache[(family, float(beta))] = (
                distribution(prior_ids, weights, state_count), ess, mean_loss
            )

    comparison_rows: list[dict[str, Any]] = []
    for posterior_row in posterior_rows:
        age = int(posterior_row["post_fit_age"])
        q = q_by_age[age]
        target_loss = float(posterior_row["mean_train_loss"])

        threshold_mask = raw_loss <= target_loss
        threshold_count = int(threshold_mask.sum())
        threshold_probability = (
            distribution(
                prior_ids[threshold_mask],
                np.ones(threshold_count),
                state_count,
            )
            if threshold_count
            else np.zeros(state_count)
        )

        row: dict[str, Any] = {
            "post_fit_age": age,
            "sgd_mean_loss": target_loss,
            "sgd_entropy_bits": float(posterior_row["state_entropy_bits"]),
            "sgd_top_state_id": int(posterior_row["top_state_id"]),
            "sgd_top_state_mass": float(posterior_row["top_state_mass"]),
            "threshold_count": threshold_count,
            "threshold_reliable": threshold_count >= Config.MIN_RELIABLE_COUNT,
            "threshold_js_to_sgd": js_divergence(q, threshold_probability)
            if threshold_count
            else math.nan,
        }
        if threshold_count:
            for key, value in distribution_metrics(threshold_probability, truth).items():
                row[f"threshold_{key}"] = value

        for family, loss in (("raw", raw_loss), ("normalized", normalized_loss)):
            reliable_candidates: list[tuple[float, np.ndarray, float, float]] = []
            for beta in Config.BETAS:
                probability, ess, mean_loss = beta_cache[(family, float(beta))]
                if ess >= Config.MIN_RELIABLE_ESS:
                    reliable_candidates.append((float(beta), probability, ess, mean_loss))

            best_js = min(
                reliable_candidates,
                key=lambda item: js_divergence(q, item[1]),
            )
            beta, probability, ess, mean_loss = best_js
            row[f"{family}_best_js_beta"] = beta
            row[f"{family}_best_js"] = js_divergence(q, probability)
            row[f"{family}_best_js_ess"] = ess
            row[f"{family}_best_js_mean_loss"] = mean_loss

            # 在可靠网格内选平均 loss 最接近 SGD 的 beta。
            matched = min(
                reliable_candidates,
                key=lambda item: abs(math.log(max(item[3], 1e-300)) - math.log(max(target_loss, 1e-300))),
            )
            beta, probability, ess, mean_loss = matched
            row[f"{family}_loss_match_beta"] = beta
            row[f"{family}_loss_match_mean_loss"] = mean_loss
            row[f"{family}_loss_match_ratio"] = mean_loss / max(target_loss, 1e-300)
            row[f"{family}_loss_match_ess"] = ess
            row[f"{family}_loss_match_js"] = js_divergence(q, probability)
            metrics = distribution_metrics(probability, truth)
            for key, value in metrics.items():
                row[f"{family}_loss_match_{key}"] = value

        comparison_rows.append(row)

    write_csv(result_dir / "analysis" / "matched_loss_comparison.csv", comparison_rows)
    create_plots(result_dir, comparison_rows)
    summary = {
        "protocol_version": "wandering_matched_loss_prior_v1",
        "prior_models": len(prior["state_ids"]),
        "hard_exact_count": int(hard.sum()),
        "hard_prior": distribution_metrics(hard_probability, truth),
        "comparable_age_count": int(
            sum(row["threshold_reliable"] for row in comparison_rows)
        ),
        "last_threshold_comparable_age": max(
            (row["post_fit_age"] for row in comparison_rows if row["threshold_reliable"]),
            default=None,
        ),
    }
    save_json(result_dir / "summary.json", summary)
    return summary


def create_plots(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"跳过作图：{error}")
        return
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ages = [row["post_fit_age"] for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(ages, [row["threshold_js_to_sgd"] for row in rows], label="L <= SGD loss")
    axes[0, 0].plot(ages, [row["raw_loss_match_js"] for row in rows], label="raw beta loss-match")
    axes[0, 0].plot(ages, [row["normalized_loss_match_js"] for row in rows], label="normalized beta loss-match")
    axes[0, 0].set_ylabel("JS to SGD (bits)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(ages, [row["sgd_top_state_mass"] for row in rows], label="SGD")
    axes[0, 1].plot(ages, [row.get("threshold_top_state_mass", np.nan) for row in rows], label="L threshold")
    axes[0, 1].plot(ages, [row["raw_loss_match_top_state_mass"] for row in rows], label="raw beta")
    axes[0, 1].set_ylabel("top state mass")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(ages, [row["sgd_mean_loss"] for row in rows], label="SGD")
    axes[1, 0].plot(ages, [row["raw_loss_match_mean_loss"] for row in rows], label="raw beta matched")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("mean train loss")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(ages, [row["threshold_count"] for row in rows], label="threshold samples")
    axes[1, 1].plot(ages, [row["raw_loss_match_ess"] for row in rows], label="raw beta ESS")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("support / ESS")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("post-fit age")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "matched_loss_overview.png", dpi=170)
    plt.close(figure)


def create_zip(result_dir: Path) -> Path:
    archive = result_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and "prior_shards" not in path.parts:
                handle.write(path, arcname=path.relative_to(result_dir))
    return archive


def main() -> None:
    profile = str(Config.PROFILE).strip().lower()
    if profile == "full":
        model_count = Config.FULL_PRIOR_MODELS
    elif profile == "pilot":
        model_count = Config.PILOT_PRIOR_MODELS
    elif profile == "smoke":
        model_count = 65_536
    else:
        raise ValueError("PROFILE 只能是 full/pilot/smoke。")

    wandering_dir = Path(Config.WANDERING_RESULT_DIR)
    result_dir = Path(Config.RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)
    wandering_config = json.loads(
        (wandering_dir / "config.json").read_text(encoding="utf-8")
    )

    print("=== Wandering vs Matched-Loss Prior ===")
    print(f"设备：{Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"wandering：{wandering_dir}")
    print(f"结果目录：{result_dir}")
    print(f"prior models={model_count:,}")

    prior = sample_prior(wandering_config, model_count, result_dir)
    summary = analyze(wandering_dir, result_dir, wandering_config, prior)
    save_json(
        result_dir / "config.json",
        {
            "profile": profile,
            "wandering_result_dir": wandering_dir,
            "result_dir": result_dir,
            "model_count": model_count,
            "betas": Config.BETAS,
            "min_reliable_count": Config.MIN_RELIABLE_COUNT,
            "min_reliable_ess": Config.MIN_RELIABLE_ESS,
        },
    )
    archive = create_zip(result_dir) if Config.CREATE_ZIP else None
    print("\n=== 同 loss 对照完成 ===")
    print(
        f"prior={summary['prior_models']:,} | hard={summary['hard_exact_count']:,} | "
        f"last threshold-comparable age={summary['last_threshold_comparable_age']}"
    )
    if archive is not None:
        print(f"下载压缩包：{archive}")


if __name__ == "__main__":
    main()
