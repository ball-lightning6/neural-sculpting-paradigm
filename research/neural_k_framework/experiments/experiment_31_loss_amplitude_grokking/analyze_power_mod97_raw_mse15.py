"""核验0/15 MSE的规则恢复结果并汇总已有对照，不分析失败机制或继续训练。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import analyze_power_mod97_grokking_mechanism as shared


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(r"E:\Downloads\results_power_mod97_raw_mse15_n70_package.zip")
MSE01 = Path(r"E:\Downloads\results_power_mod97_raw_mse01_n70_package.zip")
CE = Path(r"E:\Downloads\results_power_mod97_grokking_mechanism_n70_package.zip")
SOURCE = Path(__file__).with_name("experiment_power_mod97_raw_mse15.py")
OUT = ROOT / "tmp" / "power_mod97_mse15_analysis_20260911"
DEVICE = "cuda:1"


def validate_prediction(row, prediction, target, n):
    for key, value in shared.prediction_metrics(prediction, target, n).items():
        if isinstance(value, float):
            assert abs(value - row[key]) < 1e-12, key
        else:
            assert value == row[key], key


def independent_metrics(raw, target, spec):
    result = shared.numeric_metrics(raw, target, spec)
    amplitude = spec["target_amplitude"]
    error = raw.astype(np.float64) - amplitude * np.eye(raw.shape[-1])[target]
    point_mse = np.square(error).mean(-1)
    for suffix, value in (("answer", point_mse[:, 0].mean()), ("eos", point_mse[:, 1].mean()),
                          ("rhs", point_mse.mean())):
        result[f"raw_{suffix}_mse"] = float(value)
        result[f"relative_{suffix}_mse"] = float(value / amplitude**2)
    result["diagnostic_total_objective"] = float(point_mse.mean())
    return result


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    shared.OUT = OUT
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with zipfile.ZipFile(PACKAGE) as z:
        cfg, env, summary, protocol, pairing = (
            json.loads(z.read(name + ".json"))
            for name in ("config", "environment", "summary", "protocol", "pairing_audit"))
        captured = z.read("executed_cell.py")
        assert hashlib.sha256(captured).hexdigest() == env["source_sha256"]
        assert ast.dump(ast.parse(captured.decode("utf-8"))) == ast.dump(ast.parse(SOURCE.read_text(encoding="utf-8")))
        assert all(name in z.namelist() for name in json.loads(z.read("artifact_manifest.json"))["artifacts"])
        module_spec = importlib.util.spec_from_file_location("audited_mse15", SOURCE)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        with np.load(io.BytesIO(z.read("dataset.npz")), allow_pickle=False) as data:
            sequences, semantics = data["sequences"].copy(), data["semantics"].copy()
            n = int(data["train_count"])
        fresh = module.make_dataset(cfg)
        np.testing.assert_array_equal(sequences, fresh["sequences"])
        np.testing.assert_array_equal(semantics, fresh["semantics"])
        assert n == 6518 and len(sequences) == 9312
        assert np.all(semantics[:, 1] * semantics[:, 2] % 97 == semantics[:, 0])
        assert hashlib.sha256(sequences.tobytes()).hexdigest() == env["dataset_sha256"]
        target = sequences[:, 5:7]
        rows = shared.read_csv(z, "trajectory.csv")
        lookup = {(r["model_id"], r["step"]): r for r in rows}
        assert len(lookup) == len(rows)
        manifest = protocol["model_manifest"]
        checked = 0
        for name in sorted(p for p in z.namelist() if p.startswith("predictions/") and p.endswith(".npz")):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                step = int(packed["step"])
                ids, prediction = packed["active_ids"], packed["rhs_predictions"]
                assert prediction.shape == (len(ids), 9312, 2)
                for gid, pred in zip(ids, prediction):
                    validate_prediction(lookup[(int(gid), step)], pred, target, n)
                    checked += 1
        assert checked == len(rows)
        print(f"源码、数据和{checked}条全空间预测核验通过。", flush=True)

        device = torch.device(DEVICE)
        if not torch.cuda.is_available():
            raise RuntimeError("成功权重核验需要CUDA，未启动CPU长任务。")
        model = module.single_model(cfg, 239).to(device).eval()
        x = torch.from_numpy(sequences[:, :-1]).to(device)
        numeric_count, max_metric_error, success_checks = 0, 0.0, []
        for name in sorted(p for p in z.namelist() if p.startswith("snapshots/") and p.endswith(".npz")):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                raw = packed["raw_rhs_outputs"].copy()
                gid, step = int(packed["model_id"]), int(packed["step"])
            assert raw.shape == (9312, 2, 239) and raw.dtype == np.float32
            row = lookup[(gid, step)]
            validate_prediction(row, raw.argmax(-1).astype(np.uint16), target, n)
            for split, sl in (("train", slice(0, n)), ("test", slice(n, None))):
                for key, value in independent_metrics(raw[sl], target[sl], manifest[gid]).items():
                    expected = row[f"{split}_{key}"]
                    max_metric_error = max(max_metric_error, abs(value - expected))
                    assert np.isclose(value, expected, rtol=1e-10, atol=1e-12), (name, key)
            numeric_count += 1
            if "test_perfect" not in name:
                continue
            payload = torch.load(io.BytesIO(z.read(name[:-4] + ".pt")), map_location="cpu", weights_only=True)
            assert payload["step"] == step and payload["spec"]["model_id"] == gid
            model.load_state_dict(payload["model"])
            rebuilt = np.empty_like(raw)
            with torch.no_grad():
                for start in range(0, len(x), 256):
                    rebuilt[start:start + 256] = model(x[start:start + 256])[:, 4:6].cpu().numpy()
            changes = int(np.sum(rebuilt.argmax(-1) != raw.argmax(-1)))
            assert changes == 0
            assert np.array_equal(rebuilt[:, 0].argmax(-1), target[:, 0])
            success_checks.append({"seed": gid, "step": step, "changed_predictions": changes,
                                   "max_abs_output_difference": float(np.max(np.abs(rebuilt - raw)))})
            print(f"成功权重重算：seed {gid}, step {step}, 全部9312个答案正确。", flush=True)

        all_rows = list(rows)
        summaries = [summary]
        for previous in (MSE01, CE):
            with zipfile.ZipFile(previous) as old:
                old_cfg = json.loads(old.read("config.json"))
                assert json.loads(old.read("environment.json"))["dataset_sha256"] == env["dataset_sha256"]
                initial = json.loads(old.read("pairing_audit.json"))["initialization"]
                for item in pairing["initialization"]:
                    other = next(i for i in initial if i["seed"] == item["seed"])
                    assert item["initialization_sha256"] == other["initialization_sha256"]
                for key in ("prime", "train_fraction", "data_seed", "model_seeds", "d_model", "n_layers", "n_heads",
                            "max_context_len", "learning_rate", "warmup_steps", "adam_betas", "adam_eps",
                            "batch_order_seed", "allow_tf32"):
                    assert cfg[key] == old_cfg[key], key
                all_rows.extend(shared.read_csv(old, "trajectory.csv"))
                summaries.append(json.loads(old.read("summary.json")))
        per_seed, conditions, matched = [], [], []
        for condition in ("ce_stable", "gap_var_0p1", "gap_var_1", "raw_mse_01", "raw_mse_015"):
            successes = []
            for seed in range(3):
                history = sorted((r for r in all_rows if r["condition"] == condition and r["seed"] == seed),
                                 key=lambda r: r["step"])
                assert history
                end = history[-1]
                hit = next((r for r in history if r["test_answer_accuracy"] == 1), None)
                if hit:
                    successes.append(hit["step"])
                per_seed.append({"condition": condition, "seed": seed,
                    "first_test_100_step": hit["step"] if hit else None, "last_step": end["step"],
                    "last_train_accuracy": end["train_answer_accuracy"], "last_test_accuracy": end["test_answer_accuracy"],
                    "last_test_errors": end["test_answer_error_count"], "last_answer_ce": end["train_raw_answer_ce"]})
                # 停止的满分模型保留末值，其余模型要求在同一步确有记录，不做插值。
                row = max((r for r in history if r["step"] <= 100000), key=lambda r: r["step"])
                assert row["step"] == 100000 or row["test_answer_accuracy"] == 1
                matched.append({"condition": condition, "seed": seed, "budget": 100000,
                                "test_accuracy": row["test_answer_accuracy"], "snapshot_step": row["step"],
                                "test_perfect_stopped": row["test_answer_accuracy"] == 1})
            owner = next(s for s in summaries if any(c["condition"] == condition for c in s["conditions"]))
            recorded = next(c for c in owner["conditions"] if c["condition"] == condition)
            assert len(successes) == recorded["test_perfect_count"]
            fixed = [r for r in matched if r["condition"] == condition]
            conditions.append({"condition": condition, "successes": len(successes), "model_count": 3,
                "success_steps": sorted(successes), "package_end_step": owner["step"],
                "test_100_count_by_100k": sum(r["test_perfect_stopped"] for r in fixed),
                "mean_test_accuracy_at_100k": float(np.mean([r["test_accuracy"] for r in fixed]))})

        result = {"summary": summary, "training_environment": env, "per_seed": per_seed, "conditions": conditions,
                  "checks": {"source_hash_and_ast": True, "data_and_initialization_match": True,
                             "prediction_records": checked, "numeric_snapshots": numeric_count,
                             "max_metric_error": max_metric_error, "success_weight_checks": success_checks},
                  "audit_environment": {"torch": torch.__version__, "numpy": np.__version__, "device": DEVICE,
                      "gpu": torch.cuda.get_device_name(device), "forward_dtype": "float32", "statistics_dtype": "float64",
                      "new_training_steps": 0, "elapsed_s": time.perf_counter() - started}}
        shared.write_json("analysis.json", result)
        shared.write_csv("per_seed_comparison.csv", per_seed)
        shared.write_csv("matched_100k.csv", matched)
        names = {"ce_stable": "普通CE", "gap_var_0p1": "CE+0.1R", "gap_var_1": "CE+R",
                 "raw_mse_01": "0/1 MSE", "raw_mse_015": "0/15 MSE"}
        lines = ["# 模97规则恢复：0/15 MSE结果与已有对照", "",
            "本次只核验结果，不分析失败seed的微观原因，不追加训练或采样。", "",
            "## 0/15 MSE结果", "",
            "同一70%训练集、2层宽128 Transformer、239类输出。答案与EOS都直接拟合0/15；"
            "无额外正则、WD或CE梯度，相对MSE只作记录。",
            f"在{summary['step']}步暂停，活动训练累计{summary['elapsed_active_s'] / 60:.2f}分钟。", "",
            "|seed|首次测试100%|末次测试准确率|末次步数|", "|---|---:|---:|---:|"]
        for r in per_seed:
            if r["condition"] == "raw_mse_015":
                lines.append(f"|{r['seed']}|{r['first_test_100_step'] or '本次未达到'}|{r['last_test_accuracy']:.4%}|{r['last_step']}|")
        lines += ["", "## 已有条件对照", "",
            "各包暂停预算不同，保留实际终止步数。两个正则强度复用相同三个seed，合计6次成功不是6个独立初始化。",
            "首次满分后即停，不代表继续训练下的稳定性。", "",
            "|条件|截至各包结束恢复数|成功步数|各包结束步数|统一10万步恢复数|", "|---|---:|---|---:|---:|"]
        for r in conditions:
            steps = ", ".join(map(str, r["success_steps"])) or "无"
            lines.append(f"|{names[r['condition']]}|{r['successes']}/3|{steps}|{r['package_end_step']}|{r['test_100_count_by_100k']}/3|")
        lines += ["", "## 结果层面的结论", "",
            "- 正则CE两种强度各3/3；0/1 MSE为1/3；0/15 MSE在本次较短预算下达到2/3。",
            "- 统一到10万步，普通CE与0/1 MSE均0/3，两种正则CE和0/15 MSE均2/3。",
            "- 0/15的seed 1在14500步恢复完整规则，seed 2在70500步恢复；幅度改动带来实质结果变化。",
            "- 不据此声称每个seed均改善、幅度单调越大越好，或未恢复模型永远无法恢复；本次不继续追因。",
            "", "## 核验", "",
            f"源码hash与AST、数据及初始化指纹核验通过；{checked}条全空间预测一致；"
            f"{numeric_count}份原始输出独立复算指标，最大差{max_metric_error:.3g}。"
            "两个成功模型的保存权重在本地GPU重算全部9312个输入，答案均正确。",
            "训练环境RTX4090、PyTorch2.8.0+cu128、float32；核验环境RTX3080、PyTorch2.11.0+cu130、"
            "float32前向与float64统计。"]
        (OUT / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
