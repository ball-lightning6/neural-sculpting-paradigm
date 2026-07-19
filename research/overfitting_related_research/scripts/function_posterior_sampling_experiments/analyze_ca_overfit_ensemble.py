# %% cell 1
"""
分析 train_ca_overfit_ensemble.py 保存的 JSONL 结果。

输出：
- run_statistics.jsonl：每个模型的准确率与错误率。
- pairwise_statistics.jsonl：每对模型的函数距离、共同错误和独立基线。
- summary.jsonl：整个实验的跨种子一致性摘要。

这里最重要的不是普通预测相似度，而是：
不同模型是否在相同 probe bit 上共同犯错，并且共同错误是否显著超过
各自错误率所对应的独立错误基线。
"""

import json
import math
from pathlib import Path

import numpy as np


class Config:
    EXPERIMENT_DIR = (
        "research/overfitting_related_research/results_overfit_ensemble/"
        "rule30_layer1_overfit_n900"
    )


def load_single_jsonl_record(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"文件中没有记录：{path}")


def load_probe(path):
    inputs = []
    targets = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inputs.append(row["input"])
            targets.append(row["target"])

    if not targets:
        raise ValueError(f"probe 文件为空：{path}")

    output_bits = len(targets[0])
    if any(len(target) != output_bits for target in targets):
        raise ValueError("probe target 长度不一致。")

    flat = "".join(targets)
    target_array = np.frombuffer(flat.encode("ascii"), dtype=np.uint8) - ord("0")
    return inputs, target_array.reshape(len(targets), output_bits)


def load_predictions(path, probe_count, output_bits):
    records = []
    arrays = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") != "prediction":
                continue

            bit_string = row["prediction_bits"]
            expected_length = probe_count * output_bits
            if len(bit_string) != expected_length:
                raise ValueError(
                    f"seed={row.get('model_seed')} 的预测长度为 {len(bit_string)}，"
                    f"预期为 {expected_length}。"
                )

            array = (
                np.frombuffer(bit_string.encode("ascii"), dtype=np.uint8)
                - ord("0")
            )
            arrays.append(array.reshape(probe_count, output_bits))
            records.append(row)

    if len(records) < 2:
        raise ValueError("至少需要两个模型 seed 才能计算跨模型统计。")

    return records, np.stack(arrays, axis=0)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    denominator = math.sqrt(
        mean_a * (1 - mean_a) * mean_b * (1 - mean_b)
    )
    if denominator == 0:
        return None
    return float(((a * b).mean() - mean_a * mean_b) / denominator)


def cohen_kappa_for_error_state(error_a, error_b):
    a = error_a.reshape(-1)
    b = error_b.reshape(-1)
    observed = float((a == b).mean())
    p_a = float(a.mean())
    p_b = float(b.mean())
    expected = (1 - p_a) * (1 - p_b) + p_a * p_b
    if expected >= 1:
        return None, observed, expected
    return (observed - expected) / (1 - expected), observed, expected


def main():
    cfg = Config()
    experiment_dir = Path(cfg.EXPERIMENT_DIR)
    metadata_path = experiment_dir / "metadata.jsonl"
    probe_path = experiment_dir / "probe.jsonl"
    predictions_path = experiment_dir / "predictions.jsonl"

    metadata = load_single_jsonl_record(metadata_path)
    probe_inputs, targets = load_probe(probe_path)
    probe_count, output_bits = targets.shape
    run_records, predictions = load_predictions(
        predictions_path,
        probe_count,
        output_bits,
    )

    run_stats_path = experiment_dir / "run_statistics.jsonl"
    pairwise_path = experiment_dir / "pairwise_statistics.jsonl"
    summary_path = experiment_dir / "summary.jsonl"
    run_stats_path.write_text("", encoding="utf-8")
    pairwise_path.write_text("", encoding="utf-8")

    errors = predictions != targets[None, :, :]
    run_statistics = []

    for index, row in enumerate(run_records):
        error = errors[index]
        sample_exact = (~error).all(axis=1)
        record = {
            "record_type": "run_statistics",
            "model_seed": row["model_seed"],
            "train_steps": row["train_steps"],
            "probe_bit_accuracy": float(1 - error.mean()),
            "probe_exact_accuracy": float(sample_exact.mean()),
            "probe_bit_error_rate": float(error.mean()),
            "probe_sample_error_rate": float((~sample_exact).mean()),
            "saved_probe_metrics": row.get("probe_metrics"),
        }
        run_statistics.append(record)
        append_jsonl(run_stats_path, record)

    pairwise_statistics = []
    model_count = len(run_records)

    for i in range(model_count):
        for j in range(i + 1, model_count):
            pred_a = predictions[i]
            pred_b = predictions[j]
            error_a = errors[i]
            error_b = errors[j]

            error_rate_a = float(error_a.mean())
            error_rate_b = float(error_b.mean())
            joint_error = float((error_a & error_b).mean())
            expected_joint_error = error_rate_a * error_rate_b
            error_union = float((error_a | error_b).mean())

            kappa, observed_error_state_agreement, expected_error_state_agreement = (
                cohen_kappa_for_error_state(error_a, error_b)
            )

            record = {
                "record_type": "pairwise_statistics",
                "seed_a": run_records[i]["model_seed"],
                "seed_b": run_records[j]["model_seed"],
                "prediction_bit_agreement": float((pred_a == pred_b).mean()),
                "prediction_bit_hamming_distance": float(
                    (pred_a != pred_b).mean()
                ),
                "prediction_exact_agreement": float(
                    (pred_a == pred_b).all(axis=1).mean()
                ),
                "prediction_exact_disagreement": float(
                    (pred_a != pred_b).any(axis=1).mean()
                ),
                "error_rate_a": error_rate_a,
                "error_rate_b": error_rate_b,
                "joint_error_rate": joint_error,
                "expected_joint_error_if_independent": expected_joint_error,
                "joint_error_lift": safe_ratio(
                    joint_error,
                    expected_joint_error,
                ),
                "error_jaccard": safe_ratio(joint_error, error_union),
                "error_phi_correlation": phi_correlation(error_a, error_b),
                "error_state_agreement": observed_error_state_agreement,
                "expected_error_state_agreement_if_independent": (
                    expected_error_state_agreement
                ),
                "error_state_cohen_kappa": kappa,
            }
            pairwise_statistics.append(record)
            append_jsonl(pairwise_path, record)

    prediction_one_probability = predictions.mean(axis=0)
    prediction_entropy = binary_entropy(prediction_one_probability)
    majority_prediction_fraction = np.maximum(
        prediction_one_probability,
        1 - prediction_one_probability,
    )
    unanimously_same_prediction = (
        (prediction_one_probability == 0) | (prediction_one_probability == 1)
    )
    mixed_prediction = ~unanimously_same_prediction
    sample_unanimously_same_prediction = (
        (predictions == predictions[0:1]).all(axis=0).all(axis=1)
    )
    error_probability = errors.mean(axis=0)
    majority_prediction = (prediction_one_probability >= 0.5).astype(np.uint8)
    majority_error = majority_prediction != targets
    tied_vote = prediction_one_probability == 0.5
    non_tied_vote = ~tied_vote

    unanimously_correct = (error_probability == 0)
    unanimously_wrong = (error_probability == 1)
    mixed_error_state = ~(unanimously_correct | unanimously_wrong)

    def pairwise_mean(key):
        values = [
            row[key]
            for row in pairwise_statistics
            if row[key] is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    summary = {
        "record_type": "summary",
        "experiment_name": metadata.get("experiment_name"),
        "model_count": model_count,
        "model_seeds": [row["model_seed"] for row in run_records],
        "probe_count": probe_count,
        "output_bits": output_bits,
        "total_probe_bits": probe_count * output_bits,
        "mean_probe_bit_accuracy": float(
            np.mean([row["probe_bit_accuracy"] for row in run_statistics])
        ),
        "std_probe_bit_accuracy": float(
            np.std([row["probe_bit_accuracy"] for row in run_statistics])
        ),
        "mean_probe_exact_accuracy": float(
            np.mean([row["probe_exact_accuracy"] for row in run_statistics])
        ),
        "std_probe_exact_accuracy": float(
            np.std([row["probe_exact_accuracy"] for row in run_statistics])
        ),
        "majority_vote_bit_accuracy": float(1 - majority_error.mean()),
        "majority_vote_exact_accuracy": float(
            (~majority_error).all(axis=1).mean()
        ),
        "majority_vote_tied_bit_fraction": float(tied_vote.mean()),
        "majority_vote_non_tied_bit_accuracy": (
            float((~majority_error[non_tied_vote]).mean())
            if non_tied_vote.any()
            else None
        ),
        "mean_prediction_bit_agreement": float(
            majority_prediction_fraction.mean()
        ),
        "unanimously_same_prediction_bit_fraction": float(
            unanimously_same_prediction.mean()
        ),
        "mixed_prediction_bit_fraction": float(mixed_prediction.mean()),
        "unanimously_same_prediction_sample_fraction": float(
            sample_unanimously_same_prediction.mean()
        ),
        "mean_prediction_entropy_bits": float(prediction_entropy.mean()),
        "unanimously_correct_bit_fraction": float(unanimously_correct.mean()),
        "unanimously_wrong_bit_fraction": float(unanimously_wrong.mean()),
        "mixed_error_state_bit_fraction": float(mixed_error_state.mean()),
        "mean_pairwise_prediction_bit_agreement": pairwise_mean(
            "prediction_bit_agreement"
        ),
        "mean_pairwise_prediction_bit_hamming_distance": pairwise_mean(
            "prediction_bit_hamming_distance"
        ),
        "mean_pairwise_prediction_exact_agreement": pairwise_mean(
            "prediction_exact_agreement"
        ),
        "mean_pairwise_prediction_exact_disagreement": pairwise_mean(
            "prediction_exact_disagreement"
        ),
        "mean_pairwise_joint_error_lift": pairwise_mean("joint_error_lift"),
        "mean_pairwise_error_jaccard": pairwise_mean("error_jaccard"),
        "mean_pairwise_error_phi_correlation": pairwise_mean(
            "error_phi_correlation"
        ),
        "mean_pairwise_error_state_cohen_kappa": pairwise_mean(
            "error_state_cohen_kappa"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== 过拟合态函数一致性分析 ===")
    print(f"模型数量：{model_count}")
    print(
        f"probe bit accuracy："
        f"{summary['mean_probe_bit_accuracy']:.6f} "
        f"± {summary['std_probe_bit_accuracy']:.6f}"
    )
    print(
        f"probe exact accuracy："
        f"{summary['mean_probe_exact_accuracy']:.6f} "
        f"± {summary['std_probe_exact_accuracy']:.6f}"
    )
    print(
        f"平均两两预测 bit 汉明距离："
        f"{summary['mean_pairwise_prediction_bit_hamming_distance']:.6f}"
    )
    print(
        f"平均共同错误 lift："
        f"{summary['mean_pairwise_joint_error_lift']:.6f}"
    )
    print(
        f"平均错误 phi 相关："
        f"{summary['mean_pairwise_error_phi_correlation']:.6f}"
    )
    print(
        f"所有模型一致预测错误的 bit 比例："
        f"{summary['unanimously_wrong_bit_fraction']:.6f}"
    )
    print(f"\n逐模型结果：{run_stats_path}")
    print(f"两两比较结果：{pairwise_path}")
    print(f"总体汇总：{summary_path}")


if __name__ == "__main__":
    main()


# %% cell 2


