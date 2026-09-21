"""离线核验0/1 MSE结果包，比较逐seed曲线；只做权重前向，不训练。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import analyze_power_mod97_grokking_mechanism as shared


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(r"E:\Downloads\results_power_mod97_raw_mse01_n70_package.zip")
CE_PACKAGE = Path(r"E:\Downloads\results_power_mod97_grokking_mechanism_n70_package.zip")
SOURCE = Path(__file__).with_name("experiment_power_mod97_raw_mse01.py")
OUT = ROOT / "tmp" / "power_mod97_mse01_analysis_20260911"
DEVICE = "cuda:1"


def numeric_metrics(raw, target, spec):
    result = shared.numeric_metrics(raw, target, spec)
    z = raw.astype(np.float64)
    residual = z - np.eye(z.shape[-1], dtype=np.float64)[target]
    point_mse = np.square(residual).mean(-1)
    labels = target[:, 0]
    answer = z[:, 0]
    true = answer[np.arange(len(answer)), labels]
    wrong = answer[np.arange(answer.shape[-1])[None] != labels[:, None]].reshape(len(answer), -1)
    result.update(
        raw_answer_mse=float(point_mse[:, 0].mean()),
        raw_eos_mse=float(point_mse[:, 1].mean()),
        raw_rhs_mse=float(point_mse.mean()),
        diagnostic_total_objective=float(point_mse.mean()),
        answer_mse_vs_zero_output=float(point_mse[:, 0].mean() * z.shape[-1]),
        rhs_mse_vs_zero_output=float(point_mse.mean() * z.shape[-1]),
        true_output_rmse_to_one=float(np.sqrt(np.square(true - 1).mean())),
        wrong_output_rms_to_zero=float(np.sqrt(np.square(wrong).mean())),
        max_answer_abs_target_error=float(np.abs(residual[:, 0]).max()),
    )
    return result


def validate_prediction(row, prediction, target, train_count):
    actual = shared.prediction_metrics(prediction, target, train_count)
    for key, value in actual.items():
        if isinstance(value, float):
            assert abs(row[key] - value) < 1e-12, (key, row[key], value)
        else:
            assert row[key] == value, (key, row[key], value)


def make_plot(groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for seed, history in groups.items():
        steps = np.array([r["step"] for r in history]) / 1000
        axes[0, seed].plot(steps, [100 * r["train_answer_accuracy"] for r in history],
                           color="#666666", label="训练集")
        axes[0, seed].plot(steps, [100 * r["test_answer_accuracy"] for r in history],
                           color="#147D92", label="测试集", linewidth=1.8)
        axes[0, seed].set(title=f"seed {seed}", ylim=(-2, 102))
        axes[0, seed].legend(loc="lower right")
        axes[1, seed].semilogy(steps, [r["train_raw_answer_mse"] for r in history],
                              color="#666666", label="训练集")
        axes[1, seed].semilogy(steps, [r["test_raw_answer_mse"] for r in history],
                              color="#B24572", label="测试集")
        axes[1, seed].legend(loc="upper right")
        axes[2, seed].semilogy(steps, [r["train_gap_std"] for r in history], color="#477A32")
        axes[2, seed].set_xlabel("训练步数（千步）")
        for row in axes:
            row[seed].grid(alpha=0.2)
            row[seed].set_xlim(0, 210)
    axes[0, 0].set_ylabel("答题准确率（%）")
    axes[1, 0].set_ylabel("答题 MSE")
    axes[2, 0].set_ylabel("训练集 logit 差距标准差")
    fig.suptitle("模97除法，70%训练集，原始输出0/1 MSE：三个seed并未停在同一平台\n"
                 "seed 2在首次测试100%时停止，其余在206654步暂停", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "per_seed_overview.png", dpi=150)
    plt.close(fig)


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
            for name in ("config", "environment", "summary", "protocol", "pairing_audit")
        )
        captured = z.read("executed_cell.py")
        assert hashlib.sha256(captured).hexdigest() == env["source_sha256"]
        assert ast.dump(ast.parse(captured.decode("utf-8"))) == ast.dump(
            ast.parse(SOURCE.read_text(encoding="utf-8")))
        assert all(n in z.namelist() for n in json.loads(z.read("artifact_manifest.json"))["artifacts"])
        module_spec = importlib.util.spec_from_file_location("audited_mse01", SOURCE)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        with np.load(io.BytesIO(z.read("dataset.npz")), allow_pickle=False) as data:
            sequences = data["sequences"].copy()
            semantics = data["semantics"].copy()
            n = int(data["train_count"])
        fresh = module.make_dataset(cfg)
        np.testing.assert_array_equal(sequences, fresh["sequences"])
        np.testing.assert_array_equal(semantics, fresh["semantics"])
        assert n == fresh["train_count"] == 6518
        assert len(sequences) == len(np.unique(semantics[:, :2], axis=0)) == 9312
        assert np.all(semantics[:, 1] * semantics[:, 2] % 97 == semantics[:, 0])
        assert hashlib.sha256(sequences.tobytes()).hexdigest() == env["dataset_sha256"]
        target = sequences[:, 5:7]
        rows = shared.read_csv(z, "trajectory.csv")
        lookup = {(r["model_id"], r["step"]): r for r in rows}
        assert len(rows) == len(lookup)
        manifest = protocol["model_manifest"]
        groups = {s["model_id"]: sorted((r for r in rows if r["model_id"] == s["model_id"]),
                                       key=lambda r: r["step"]) for s in manifest}
        predictions = {gid: {} for gid in groups}
        checked = 0
        for name in sorted(n for n in z.namelist() if n.startswith("predictions/") and n.endswith(".npz")):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                step = int(packed["step"])
                ids, pred = packed["active_ids"], packed["rhs_predictions"]
                assert pred.shape == (len(ids), 9312, 2)
                for gid, p in zip(ids, pred):
                    validate_prediction(lookup[(int(gid), step)], p, target, n)
                    predictions[int(gid)][step] = p[:, 0].copy()
                    checked += 1
        assert checked == len(rows)
        print(f"源码、数据和{checked}条全空间预测核验通过。", flush=True)

        snapshot_checks = []
        max_metric_error = 0.0
        device = torch.device(DEVICE)
        if not torch.cuda.is_available():
            raise RuntimeError("只做GPU权重前向核验，不回退长时间CPU计算。")
        model = module.single_model(cfg, len(fresh["tokens"])).to(device).eval()
        x = torch.from_numpy(sequences[:, :-1]).to(device)
        for name in sorted(n for n in z.namelist() if n.startswith("snapshots/") and n.endswith(".npz")):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                raw = packed["raw_rhs_outputs"].copy()
                gid, step = int(packed["model_id"]), int(packed["step"])
            assert raw.shape == (9312, 2, 239) and raw.dtype == np.float32
            row = lookup[(gid, step)]
            validate_prediction(row, raw.argmax(-1).astype(np.uint16), target, n)
            for split, sl in (("train", slice(0, n)), ("test", slice(n, None))):
                metrics = numeric_metrics(raw[sl], target[sl], manifest[gid])
                for key, value in metrics.items():
                    expected = row[f"{split}_{key}"]
                    max_metric_error = max(max_metric_error, abs(value - expected))
                    assert np.isclose(value, expected, rtol=1e-10, atol=1e-12), (name, key, value, expected)
            payload = torch.load(io.BytesIO(z.read(name[:-4] + ".pt")), map_location="cpu", weights_only=True)
            assert payload["step"] == step and payload["spec"]["model_id"] == gid
            model.load_state_dict(payload["model"])
            rebuilt = np.empty_like(raw)
            with torch.no_grad():
                for start in range(0, len(x), 256):
                    rebuilt[start:start + 256] = model(x[start:start + 256])[:, 4:6].cpu().numpy()
            changed = int(np.sum(rebuilt.argmax(-1) != raw.argmax(-1)))
            assert changed == 0, (name, changed)
            validate_prediction(row, rebuilt.argmax(-1).astype(np.uint16), target, n)
            snapshot_checks.append({"snapshot": name, "seed": manifest[gid]["seed"], "step": step,
                                    "max_abs_output_difference": float(np.max(np.abs(rebuilt - raw))),
                                    "changed_predictions": changed,
                                    "train_accuracy": row["train_answer_accuracy"],
                                    "test_accuracy": row["test_answer_accuracy"]})
            print(f"权重重算：seed {gid}, step {step}, 分类变化 {changed}。", flush=True)

        per_seed, selected, thresholds = [], [], []
        for gid, history in groups.items():
            end = history[-1]
            fit = next((r for r in history if r["train_answer_error_count"] == 0), None)
            best = max(history, key=lambda r: r["test_answer_accuracy"])
            record = {"seed": manifest[gid]["seed"], "last_step": end["step"],
                      "first_train_fit_step": fit["step"] if fit else None,
                      "test_at_first_train_fit": fit["test_answer_accuracy"] if fit else None,
                      "best_test_accuracy": best["test_answer_accuracy"], "best_test_step": best["step"],
                      "last_train_accuracy": end["train_answer_accuracy"],
                      "last_test_accuracy": end["test_answer_accuracy"],
                      "last_test_errors": end["test_answer_error_count"],
                      "last_train_mse": end["train_raw_answer_mse"],
                      "last_test_mse": end["test_raw_answer_mse"],
                      "last_train_ce": end["train_raw_answer_ce"],
                      "last_gap_mean": end["train_gap_mean"], "last_gap_std": end["train_gap_std"]}
            for threshold in (0.5, 0.9, 0.95, 0.99, 1.0):
                hit = next((r for r in history if r["test_answer_accuracy"] >= threshold), None)
                record[f"first_test_{threshold:g}"] = hit["step"] if hit else None
            # 这些是观测节点的比较，不声称每一步都保持给定MSE或准确率。
            for level in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
                low = [r for r in history if r["train_answer_accuracy"] == 1 and r["train_raw_answer_mse"] <= level]
                thresholds.append({"seed": gid, "train_mse_threshold": level, "record_count": len(low),
                    "first_step": low[0]["step"] if low else None,
                    "first_test_accuracy": low[0]["test_answer_accuracy"] if low else None,
                    "test_min": min(r["test_answer_accuracy"] for r in low) if low else None,
                    "test_max": max(r["test_answer_accuracy"] for r in low) if low else None})
            for step in (12000, 20000, 38500, 50000, 90000, 100000, 110000, 130000, 140000,
                         150000, 170000, 190000, 200000, 206654):
                row = lookup.get((gid, step))
                if row is not None:
                    selected.append({key: row[key] for key in ("seed", "step", "train_answer_accuracy",
                        "test_answer_accuracy", "train_raw_answer_mse", "train_raw_answer_ce", "train_gap_std")})
            if fit:
                postfit = [r for r in history if r["step"] >= fit["step"]]
                record["distinct_postfit_answer_functions"] = len({r["answer_function_sha256"] for r in postfit})
                record["postfit_train_fit_violations_at_evaluations"] = sum(r["train_answer_accuracy"] < 1 for r in postfit)
            per_seed.append(record)
        assert sum(r["last_test_accuracy"] == 1 for r in per_seed) == summary["conditions"][0]["test_perfect_count"]
        np.testing.assert_allclose(np.mean([r["last_test_accuracy"] for r in per_seed]),
                                   summary["conditions"][0]["mean_last_test_answer_accuracy"], atol=1e-12)

        comparisons = []
        with zipfile.ZipFile(CE_PACKAGE) as old:
            old_cfg = json.loads(old.read("config.json"))
            old_env = json.loads(old.read("environment.json"))
            assert old_env["dataset_sha256"] == env["dataset_sha256"]
            old_pairing = json.loads(old.read("pairing_audit.json"))["initialization"]
            for item in pairing["initialization"]:
                matched = next(i for i in old_pairing if i["seed"] == item["seed"])
                assert item["initialization_sha256"] == matched["initialization_sha256"]
            for key in ("prime", "train_fraction", "data_seed", "model_seeds", "d_model", "n_layers", "n_heads",
                        "max_context_len", "learning_rate", "warmup_steps", "adam_betas", "adam_eps",
                        "batch_order_seed", "allow_tf32"):
                assert cfg[key] == old_cfg[key], key
            old_rows = shared.read_csv(old, "trajectory.csv")
            for step in (20000, 50000, 100000, 140000, 150000):
                for condition in [c[0] for c in old_cfg["conditions"]] + ["raw_mse_01"]:
                    population = []
                    for seed in range(3):
                        h = groups[seed] if condition == "raw_mse_01" else [
                            r for r in old_rows if r["condition"] == condition and r["seed"] == seed]
                        candidates = [r for r in h if r["step"] <= step]
                        assert candidates, (condition, seed, step)
                        row = max(candidates, key=lambda r: r["step"])
                        # 只允许停止的满分模型前值保持，不插值尚在训练的模型。
                        assert row["step"] == step or row["test_answer_accuracy"] == 1
                        population.append(row)
                    if population:
                        assert len(population) == 3
                        comparisons.append({"step": step, "condition": condition,
                            "mean_test_accuracy": float(np.mean([r["test_answer_accuracy"] for r in population])),
                            "full_test_models": sum(r["test_answer_accuracy"] == 1 for r in population),
                            "stopped_successes_carried_forward": sum(r["step"] < step for r in population)})

        shared.write_csv("per_seed_results.csv", per_seed)
        shared.write_csv("selected_steps.csv", selected)
        shared.write_csv("low_mse_observations.csv", thresholds)
        shared.write_csv("weight_verification.csv", snapshot_checks)
        shared.write_csv("matched_budget_comparison.csv", comparisons)
        make_plot(groups)
        result = {"package": str(PACKAGE), "summary": summary, "training_environment": env,
                  "checks": {"source_ast_and_hash": True, "dataset_and_semantics": True,
                             "prediction_records": checked, "snapshot_count": len(snapshot_checks),
                             "max_metric_recompute_error": max_metric_error,
                             "snapshot_changed_classes": sum(r["changed_predictions"] for r in snapshot_checks),
                             "previous_ce_data_and_recorded_initialization_match": True},
                  "per_seed": per_seed, "matched_budget_comparison": comparisons,
                  "audit_environment": {"python": platform.python_version(), "torch": torch.__version__,
                      "numpy": np.__version__, "device": DEVICE, "gpu": torch.cuda.get_device_name(device),
                      "dtype": "保存的float32权重前向；独立统计float64", "new_training_steps": 0,
                      "elapsed_seconds": time.perf_counter() - started}}
        shared.write_json("analysis.json", result)
        lines = ["# 原始0/1 MSE：模97除法结果核验", "",
            "## 设置与结局", "",
            "70%训练集，6518训练/2794测试，2层宽128 Transformer，239类输出。"
            "答案和EOS均为原始输出0/1 MSE，无CE梯度、额外正则或weight decay。",
            f"运行到{summary['step']}步暂停，活动训练耗时{summary['elapsed_active_s'] / 60:.2f}分钟；"
            "seed 2在14万步首次测试满分后单独停止，没有进行满分后稳定性检验。", "",
            "|seed|停止/暂停步数|最终测试准确率|错题数/2794|训练答题MSE|",
            "|---|---:|---:|---:|---:|"]
        for r in per_seed:
            lines.append(f"|{r['seed']}|{r['last_step']}|{r['last_test_accuracy']:.4%}|"
                         f"{r['last_test_errors']}|{r['last_train_mse']:.6g}|")
        lines += ["", "## 关键解读", "",
            "- 汇总准确率为均值，汇总MSE为中位数。约90%的汇总平台不等于三个模型停在90%。",
            "- seed 0长时间停在99.3%左右；seed 2继续上升到100%；seed 1从最佳77.70%下降到66.21%。",
            "- seed 1所有训练全对的记录中，答题MSE都高于1e-5。它与另外两个约1e-9的端点并非同等拟合精度。",
            "- seed 2在12000步训练99.8006%/测试21.9757%，之后38500步训练首次100%/测试68.5039%，"
            "140000步测试100%。存在拟合与泛化的时间分离；自动诊断因严格门槛未判为典型延迟，"
            "不能把诊断字符串当成没有延迟的证据。",
            "- 0/1 MSE允许恢复完整规则，且无需诊断CE趋于0；成功端点训练CE约4.48358，"
            "符合239类、logit差距为1的输出几何。不能仅凭训练轨迹把这些端点解释为静态体积分布。",
            "- 接下来的0/15实验应比较逐seed成功率、恢复时间、精度和相对MSE，而不是只比较一条汇总曲线。", "",
            "## 与已有CE结果对齐步数", "",
            "停止的满分模型仅以冻结结果计入汇总；各组在同一数据划分、相同初始化指纹与minibatch协议下比较。",
            "", "|步数|条件|平均测试准确率|首次满分模型数|", "|---|---|---:|---:|"]
        for r in comparisons:
            if r["step"] == 150000:
                lines.append(f"|{r['step']}|{r['condition']}|{r['mean_test_accuracy']:.4%}|{r['full_test_models']}/3|")
        lines += ["", "## 独立核验", "",
            f"源码hash与AST、数据划分及模除法语义通过核验；{checked}条全空间预测均重算一致。"
            f"{len(snapshot_checks)}份原始输出独立复算，最大指标差{max_metric_error:.3g}。"
            "6份保存权重在本地RTX3080重新计算全部9312个输入，分类输出无变化。",
            "训练环境为RTX4090、PyTorch2.8.0+cu128、float32；离线权重核验为PyTorch2.11.0+cu130、"
            "float32前向及float64独立统计。没有新增训练。",
            "", "图：`per_seed_overview.png`；详细数值：`analysis.json`、`per_seed_results.csv`、"
            "`selected_steps.csv`、`low_mse_observations.csv`、`matched_budget_comparison.csv`。"]
        (OUT / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
