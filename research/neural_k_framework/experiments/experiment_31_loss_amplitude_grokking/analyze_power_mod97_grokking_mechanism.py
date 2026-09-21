"""离线分析用户返回的n70结果包；重算原始输出并用已存权重前向核验，不训练。"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(r"E:\Downloads\results_power_mod97_grokking_mechanism_n70_package.zip")
OUT = ROOT / "tmp" / "power_mod97_n70_analysis_20260911"
MODEL_SOURCE = Path(__file__).with_name("experiment_power_mod97_grokking_mechanism.py")
DEVICE = "cuda:1"


def read_csv(z, name):
    rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))
    strings = {"condition", "kind", "answer_function_sha256", "status", "classification"}
    integers = {"model_id", "seed", "step", "train_answer_error_count", "test_answer_error_count", "last_step"}
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif key in strings:
                continue
            elif value in {"True", "False"}:
                row[key] = value == "True"
            elif key in integers:
                row[key] = int(value)
            else:
                row[key] = float(value)
    return rows


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_metrics(predictions, target, n):
    result = {"full_rule_exact": bool(np.array_equal(predictions[:, 0], target[:, 0])),
              "answer_function_sha256": hashlib.sha256(np.ascontiguousarray(predictions[:, 0]).tobytes()).hexdigest()}
    for split, sl in (("train", slice(0, n)), ("test", slice(n, None))):
        good = predictions[sl] == target[sl]
        result.update({f"{split}_answer_accuracy": float(good[:, 0].mean()),
                       f"{split}_answer_error_count": int((~good[:, 0]).sum()),
                       f"{split}_eos_accuracy": float(good[:, 1].mean()),
                       f"{split}_official_row_accuracy": float(good.all(-1).mean())})
    return result


def numeric_metrics(raw, target, spec):
    z = raw.astype(np.float64)
    chosen = np.take_along_axis(z, target[..., None], -1)
    # 独立使用logaddexp归约，不复用训练脚本的log1p实现。
    ce = np.logaddexp.reduce(z - chosen, axis=-1)
    answer = z[:, 0]
    labels = target[:, 0]
    true = answer[np.arange(len(answer)), labels]
    wrong = answer[np.arange(answer.shape[-1])[None] != labels[:, None]].reshape(len(answer), -1)
    gaps = true[:, None] - wrong
    base = float(ce.mean())
    addition = spec["coefficient"] * float(np.var(gaps)) if spec["kind"] == "gap_var" else 0.
    return {"raw_answer_ce": float(ce[:, 0].mean()), "raw_eos_ce": float(ce[:, 1].mean()),
            "raw_rhs_ce": base, "diagnostic_total_objective": base + addition,
            "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
            "gap_variance": float(gaps.var()),
            "between_sample_gap_variance": float(gaps.mean(-1).var()),
            "within_sample_gap_variance": float(gaps.var(-1).mean()),
            "true_logit_mean": float(true.mean()), "true_logit_std": float(true.std()),
            "wrong_logit_mean": float(wrong.mean()), "wrong_logit_std": float(wrong.std()),
            "min_top_gap": float((true - wrong.max(-1)).min())}


def first_hit(history, field, threshold):
    return next((r for r in history if r[field] >= threshold), None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with zipfile.ZipFile(PACKAGE) as z:
        cfg = json.loads(z.read("config.json"))
        env = json.loads(z.read("environment.json"))
        summary = json.loads(z.read("summary.json"))
        manifest = json.loads(z.read("protocol.json"))["model_manifest"]
        artifact_manifest = json.loads(z.read("artifact_manifest.json"))["artifacts"]
        assert all(name in z.namelist() for name in artifact_manifest)
        captured = z.read("executed_cell.py")
        assert hashlib.sha256(captured).hexdigest() == env["source_sha256"]
        assert ast.dump(ast.parse(captured.decode("utf-8")), include_attributes=False) == ast.dump(
            ast.parse(MODEL_SOURCE.read_text(encoding="utf-8")), include_attributes=False)
        print("源码hash及AST核验通过。", flush=True)
        spec = importlib.util.spec_from_file_location("audited_power", MODEL_SOURCE)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)

        with np.load(io.BytesIO(z.read("dataset.npz")), allow_pickle=False) as packed:
            sequences = packed["sequences"].copy()
            semantic = packed["semantics"].copy()
            n = int(packed["train_count"])
        fresh = m.make_dataset(cfg)
        np.testing.assert_array_equal(sequences, fresh["sequences"])
        np.testing.assert_array_equal(semantic, fresh["semantics"])
        assert n == fresh["train_count"] == 6518
        assert len(np.unique(semantic[:, :2], axis=0)) == len(sequences) == 9312
        assert np.all(semantic[:, 1] * semantic[:, 2] % 97 == semantic[:, 0])
        assert hashlib.sha256(sequences.tobytes()).hexdigest() == env["dataset_sha256"]
        target = sequences[:, 5:7]
        rows = read_csv(z, "trajectory.csv")
        lookup = {(r["model_id"], r["step"]): r for r in rows}
        assert len(lookup) == len(rows)
        groups = {s["model_id"]: sorted((r for r in rows if r["model_id"] == s["model_id"]), key=lambda r: r["step"])
                  for s in manifest}

        checked_predictions = 0
        observed_by_id = {s["model_id"]: [] for s in manifest}
        for name in sorted(name for name in z.namelist() if name.startswith("predictions/") and name.endswith(".npz")):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                ids = packed["active_ids"]
                predictions = packed["rhs_predictions"]
                step = int(packed["step"])
            assert predictions.shape == (len(ids), len(sequences), 2)
            for i, gid in enumerate(ids):
                actual = prediction_metrics(predictions[i], target, n)
                row = lookup[(int(gid), step)]
                for key, value in actual.items():
                    if isinstance(value, (str, bool, int)):
                        assert value == row[key], (gid, step, key)
                    else:
                        assert abs(value - row[key]) < 1e-12, (gid, step, key)
                observed_by_id[int(gid)].append((step, actual["answer_function_sha256"]))
                checked_predictions += 1
        assert checked_predictions == len(rows)
        print(f"{checked_predictions}条完整映射及准确率核验通过。", flush=True)

        initialization_checks = []
        recorded_initial = json.loads(z.read("pairing_audit.json"))["initialization"]
        for item in recorded_initial:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(item["seed"])
                initial_model = m.single_model(cfg, len(data_tokens := fresh["tokens"]))
            hasher = hashlib.sha256()
            for p in initial_model.parameters():
                hasher.update(p.detach().numpy().tobytes())
            equal = hasher.hexdigest() == item["initialization_sha256"]
            initialization_checks.append({"seed": item["seed"], "initializer_hash_matches_local_environment": equal,
                "producer_cross_condition_hash_assertion_passed": item["identical_across_conditions"]})
            initial_rows = [lookup[(s["model_id"], 0)] for s in manifest if s["seed"] == item["seed"]]
            assert len({r["answer_function_sha256"] for r in initial_rows}) == 1

        weight_checks = []
        maximum_numeric_error = 0.
        raw_snapshots = sorted(name for name in z.namelist() if name.startswith("snapshots/") and name.endswith(".npz"))
        device = torch.device(DEVICE)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("权重前向核验需要CUDA；没有自动启动CPU长任务。")
        model = m.single_model(cfg, len(fresh["tokens"])).to(device)
        x = torch.from_numpy(sequences[:, :-1]).to(device)
        for number, name in enumerate(raw_snapshots, 1):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                raw = packed["raw_rhs_outputs"].copy()
                gid, step = int(packed["model_id"]), int(packed["step"])
            row = lookup[(gid, step)]
            spec_row = manifest[gid]
            assert raw.shape == (len(sequences), 2, len(fresh["tokens"])) and raw.dtype == np.float32
            independent_pred = prediction_metrics(raw.argmax(-1).astype(np.uint16), target, n)
            assert all(independent_pred[k] == row[k] if isinstance(independent_pred[k], (str, bool, int))
                       else abs(independent_pred[k] - row[k]) < 1e-12 for k in independent_pred)
            for split, sl in (("train", slice(0, n)), ("test", slice(n, None))):
                derived = numeric_metrics(raw[sl], target[sl], spec_row)
                for key, value in derived.items():
                    reference = row[f"{split}_{key}"]
                    error = abs(value - reference)
                    maximum_numeric_error = max(maximum_numeric_error, error)
                    assert np.isclose(value, reference, rtol=1e-10, atol=1e-10), (name, key, value, reference)
            payload = torch.load(io.BytesIO(z.read(name[:-4] + ".pt")), map_location="cpu", weights_only=True)
            assert payload["step"] == step and payload["spec"]["model_id"] == gid
            model.load_state_dict(payload["model"])
            model.eval()
            reconstructed = np.empty_like(raw)
            with torch.no_grad():
                for start in range(0, len(x), 256):
                    end = min(start + 256, len(x))
                    reconstructed[start:end] = model(x[start:end])[:, 4:6].cpu().numpy()
            error = float(np.max(np.abs(reconstructed - raw)))
            changed_answers = int(np.sum(reconstructed[:, 0].argmax(-1) != raw[:, 0].argmax(-1)))
            changed_eos = int(np.sum(reconstructed[:, 1].argmax(-1) != raw[:, 1].argmax(-1)))
            verified = prediction_metrics(reconstructed.argmax(-1).astype(np.uint16), target, n)
            weight_checks.append({"snapshot": name, "model_id": gid, "condition": spec_row["condition"],
                                  "seed": spec_row["seed"], "step": step, "max_abs_output_difference": error,
                                  "changed_answer_predictions": changed_answers, "changed_eos_predictions": changed_eos,
                                  "reconstructed_full_rule_exact": verified["full_rule_exact"],
                                  "reconstructed_test_accuracy": verified["test_answer_accuracy"]})
            print(f"权重核验 {number}/{len(raw_snapshots)}：{spec_row['condition']} seed={spec_row['seed']} "
                  f"step={step}，答案变化={changed_answers}，最大输出差={error:.3g}", flush=True)
        write_csv("weight_verification.csv", weight_checks)

        thresholds = []
        per_seed = []
        checkpoints = []
        for spec_row in manifest:
            gid = spec_row["model_id"]
            history = groups[gid]
            fit = next((r for r in history if r["train_answer_error_count"] == 0), None)
            record = {**spec_row, "first_train_fit": fit["step"] if fit else None,
                      "last_step": history[-1]["step"], "last_test_accuracy": history[-1]["test_answer_accuracy"],
                      "last_train_answer_ce": history[-1]["train_raw_answer_ce"],
                      "last_gap_mean": history[-1]["train_gap_mean"], "last_gap_std": history[-1]["train_gap_std"]}
            for label, threshold in (("50", .5), ("90", .9), ("95", .95), ("99", .99), ("99p9", .999), ("100", 1.)):
                hit = first_hit(history, "test_answer_accuracy", threshold)
                record[f"first_test_{label}"] = hit["step"] if hit else None
                thresholds.append({**spec_row, "threshold": threshold, "hit": hit is not None,
                                   "step": hit["step"] if hit else None,
                                   "train_answer_ce": hit["train_raw_answer_ce"] if hit else None,
                                   "gap_mean": hit["train_gap_mean"] if hit else None,
                                   "gap_std": hit["train_gap_std"] if hit else None})
            # 同时记录最后一次跌破99%的后续恢复，避免把短暂越线当作稳定结果。
            first_suffix = next((r for i, r in enumerate(history) if r["test_answer_accuracy"] >= .99
                                 and all(v["test_answer_accuracy"] >= .99 for v in history[i:])), None)
            record["last_99pct_upcrossing_through_observed_end"] = first_suffix["step"] if first_suffix else None
            values = observed_by_id[gid]
            changes = [(s, h) for i, (s, h) in enumerate(values) if i and h != values[i-1][1]]
            record["last_observed_function_change_step"] = changes[-1][0] if changes else 0
            record["test_accuracy_at_10k"] = next((r["test_answer_accuracy"] for r in history if r["step"] == 10000), None)
            per_seed.append(record)
            for point in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 150550):
                r = next((r for r in history if r["step"] == point), None)
                if r:
                    checkpoints.append({**spec_row, **{k:r[k] for k in ("step", "train_answer_accuracy", "test_answer_accuracy", "train_raw_answer_ce", "train_gap_mean", "train_gap_std")}})
        write_csv("first_thresholds.csv", thresholds)
        write_csv("per_seed_results.csv", per_seed)
        write_csv("selected_steps.csv", checkpoints)

        matched = read_csv(z, "matched_answer_ce.csv")
        matched_aggregate = []
        for name, _, _, _ in cfg["conditions"]:
            for level in sorted(set(r["answer_ce_threshold"] for r in matched), reverse=True):
                group = [r for r in matched if r["condition"] == name and r["answer_ce_threshold"] == level and r["reached"]]
                matched_aggregate.append({"condition": name, "threshold": level, "reached_count": len(group),
                                          "mean_test_accuracy": float(np.mean([r["test_answer_accuracy"] for r in group])) if group else None,
                                          "median_actual_train_ce": float(np.median([r["train_raw_answer_ce"] for r in group])) if group else None,
                                          "median_gap_std": float(np.median([r["train_gap_std"] for r in group])) if group else None})
        write_csv("matched_ce_aggregate.csv", matched_aggregate)

        condition_rows = []
        for name, _, _, _ in cfg["conditions"]:
            g = [r for r in per_seed if r["condition"] == name]
            condition_rows.append({"condition": name, "seeds": len(g), "complete": sum(r["first_test_100"] is not None for r in g),
                "first_99_steps": [r["first_test_99"] for r in g], "first_100_steps": [r["first_test_100"] for r in g],
                "mean_last_test_accuracy": float(np.mean([r["last_test_accuracy"] for r in g])),
                "median_last_train_answer_ce": float(np.median([r["last_train_answer_ce"] for r in g])),
                "median_last_gap_std": float(np.median([r["last_gap_std"] for r in g]))})
        verification = {"source_ast_matches_local": True, "source_sha256_matches_metadata": True,
            "dataset_matches_original_protocol": True, "initializer_hashes": initialization_checks,
            "initialization_note": "本机torch2.11与结果包torch2.8环境不同，未逐位复现seed初始化哈希；没有把这当作已通过。组间配对由已核对源码、生产端哈希断言和独立核验的step0完整预测支持。已存权重直接载入，不依赖重新初始化。",
            "full_prediction_rows_checked": checked_predictions, "saved_raw_output_snapshots_checked": len(raw_snapshots),
            "max_independent_metric_absolute_error": maximum_numeric_error,
            "weight_snapshots_checked": len(weight_checks),
            "max_weight_forward_output_difference": max(r["max_abs_output_difference"] for r in weight_checks),
            "total_changed_answer_predictions": sum(r["changed_answer_predictions"] for r in weight_checks),
            "total_changed_eos_predictions": sum(r["changed_eos_predictions"] for r in weight_checks),
            "successful_final_weight_snapshots_still_exact": all(r["reconstructed_full_rule_exact"] for r in weight_checks if "test_perfect" in r["snapshot"]),
            "last_baseline_weight_available_in_package": all(any(
                r["model_id"] == gid and r["step"] == groups[gid][-1]["step"] for r in weight_checks)
                for gid in groups if groups[gid][0]["condition"] == cfg["baseline_condition"]),
            "verification_device": torch.cuda.get_device_name(device), "verification_torch": str(torch.__version__)}
        result = {"package": str(PACKAGE), "status": summary["status"], "last_step": summary["step"],
                  "recorded_gpu": env["gpu"], "elapsed_minutes": summary["elapsed_active_s"] / 60,
                  "train_count": n, "test_count": len(sequences)-n, "conditions": condition_rows,
                  "verification": verification}
        write_json("analysis.json", result)
        write_json("per_seed_results.json", per_seed)
        write_json("verification.json", verification)
        for name in ("per_seed_accuracy.png", "recovery_and_matched_ce.png"):
            (OUT / ("original_" + name)).write_bytes(z.read(name))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r[0] for r in cfg["conditions"]]
    fig, axes = plt.subplots(3, len(names), figsize=(16, 10), sharex="col")
    colors = ("#2675ad", "#c44e52", "#428a52")
    for col, name in enumerate(names):
        for seed, color in zip(cfg["model_seeds"], colors):
            history = [r for r in rows if r["condition"] == name and r["seed"] == seed]
            xvalues = np.array([r["step"] for r in history])/1000
            axes[0,col].plot(xvalues, [r["test_answer_accuracy"]*100 for r in history], color=color, label=f"seed {seed}")
            axes[0,col].plot(xvalues, [r["train_answer_accuracy"]*100 for r in history], "--", color=color, alpha=.35)
            axes[1,col].plot(xvalues, [r["train_raw_answer_ce"] for r in history], color=color)
            axes[2,col].plot(xvalues, [r["train_gap_std"] for r in history], color=color)
        axes[0,col].set_title(name)
        axes[0,col].set_ylim(-2,102)
        axes[1,col].set_yscale("log")
        axes[1,col].set_ylim(1e-12,10)
        axes[2,col].set_yscale("log")
        axes[2,col].set_ylim(.01,20)
        axes[2,col].set_xlabel("Updates (thousands); curves end at stopping")
        axes[0,col].legend(fontsize=8)
        for ax in axes[:,col]:
            ax.set_xlim(0,155)
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Answer accuracy (%)\nsolid: test; dashed: train")
    axes[1,0].set_ylabel("Train answer CE")
    axes[2,0].set_ylabel("Train correct-minus-other gap std")
    fig.suptitle("Power Transformer, mod97 division, 70% data: paired alignment intervention")
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_overview.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
