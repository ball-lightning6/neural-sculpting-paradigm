"""核验公开增补包或复算本地原始记录；不启动训练、采样或 GPU。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
ROOT = P14 = P16 = None
REFERENCE = "3834fb3f6555c7c58218462502635b16b9b377f9"
PACKAGE_PREFIX = "research/neural_k_framework/supplementary/updates_20261106/scaffolding/"
FINGERPRINTS = {}
HASH_CHECKS = []
MAX_ERRORS = {"mod3_statistics": 0.0}
PACKAGE_FILES = {"README_zh.md", "README_en.md", "recompute.py",
                 "key_results.json", "audit.json", "source_sha256.json"}
HISTORICAL_SOURCES = [
    f"research/notes/20260913_mod3_{name}_zh.md"
    for name in ["endpoint_scaffold_matched_bce", "weight_perturbation_generalization",
                 "scaffold_withdrawal"]
]


def fingerprint(path, expected=None):
    path = path.resolve()
    key = path.relative_to(ROOT).as_posix()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    FINGERPRINTS[key] = value
    if expected is not None:
        assert value == expected, (key, value, expected)
        HASH_CHECKS.append(key)
    return value


def read(path):
    fingerprint(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def close(actual, expected, category="mod3_statistics", tolerance=1e-10):
    error = abs(actual - expected)
    MAX_ERRORS[category] = max(MAX_ERRORS[category], error)
    assert math.isfinite(actual) and error <= tolerance, (actual, expected)


def mean(values):
    return statistics.fmean(values)


def check_sources(base, roots):
    for path in sorted(base.glob("*.py")):
        fingerprint(path)
    for name in ["README.md", "REPORT_ZH.md"]:
        fingerprint(base / name)
    for folder in roots:
        manifest = read(folder / "MANIFEST.json")
        for path in sorted((folder / "source").glob("*.py")):
            fingerprint(path)
        cfg = manifest["config"]
        if "source_sha256" in cfg:
            fingerprint(folder / "source/experiment.py", cfg["source_sha256"])
            fingerprint(P14 / "experiment.py", cfg["source_sha256"])
            for name, key in [
                ("experiment_mod3_endpoint_scaffold_matched_bce.py", "legacy_sha256"),
                ("experiment_mod3_small_data_grokking.py", "model_source_sha256"),
            ]:
                fingerprint(folder / "source" / name, cfg[key])
                fingerprint(ROOT / "research/ca_phase_transition" / name, cfg[key])
        for name, expected in manifest.get("frozen_sources", {}).items():
            fingerprint(folder / "source" / name, expected)


def mod3(bits, roots, analyses):
    cfg = read(roots[0] / "MANIFEST.json")["config"]
    assert cfg["bits"] == bits and cfg["max_steps"] == 16000
    expected_protocol = {
        "model_bits": 30, "width": 1024, "layers": 3,
        "data_seeds": [202611020, 202611021], "model_seeds": [0, 1, 2],
        "batch_size": 512, "learning_rate": .001, "warmup_steps": 10,
        "aux_lambda": 1.0, "stages": [4000, 8000, 16000],
        "summary_window_points": 3, "eval_every": 1000,
        "success_accuracy": .99, "fit_bce": .001,
        "weight_decay": 0.0, "dropout": 0.0, "dtype": "float32",
        "objective": "L_endpoint + lambda * mean_BCE(auxiliary_prefixes); lambda=0 or 1",
    }
    for name, expected in expected_protocol.items():
        assert cfg[name] == expected, (name, cfg[name], expected)
    jobs, lookup, by_n = [], {}, {}
    for folder in roots:
        other = read(folder / "MANIFEST.json")["config"]
        for key in ["bits", "model_bits", "width", "layers", "data_seeds", "model_seeds",
                    "batch_seed", "batch_size", "learning_rate", "warmup_steps", "aux_lambda",
                    "max_steps", "stages", "summary_window_points", "eval_every",
                    "success_accuracy", "fit_bce", "weight_decay", "dropout", "dtype"]:
            assert cfg[key] == other[key], key
        verification = read(folder / "VERIFICATION.json")
        assert verification["graph_eager_max_weight_error"] < 2e-5
        assert verification["restored_step_max_weight_error"] < 1e-7
        for path in sorted(folder.glob("data*_n*/result.json")):
            job = read(path)
            assert job["status"] == "complete" and job["step"] == cfg["max_steps"]
            assert job["n"] in other["counts"] and job["data_seed"] in cfg["data_seeds"]
            jobs.append(job)
            by_n.setdefault(job["n"], []).append(job["data_seed"])
            for row in job["rows"]:
                key = tuple(row[k] for k in ["n", "data_seed", "model_seed", "arm", "step"])
                assert key not in lookup, key
                lookup[key] = row
    assert len({j["initial_weights_sha256"] for j in jobs}) == 1
    for ds in cfg["data_seeds"]:
        assert len({j["data_ids_sha256"] for j in jobs if j["data_seed"] == ds}) == 1
    parameters = (30 * 1024 + 1024 + 2 * 1024) + 2 * (1024**2 + 3 * 1024) + 1025 * 2 * bits
    assert {j["parameters_per_model"] for j in jobs} == {parameters}
    windows, summaries, pairs = {}, {}, {}
    for n, data_seeds in sorted(by_n.items()):
        assert len(data_seeds) == len(set(data_seeds))
        for stage in cfg["stages"]:
            for arm in ["direct", "scaffold"]:
                group, ends = [], []
                for ds in sorted(data_seeds):
                    for ms in cfg["model_seeds"]:
                        end = lookup[n, ds, ms, arm, stage]
                        values = [lookup[n, ds, ms, arm, stage - i * cfg["eval_every"]]
                                  ["test_endpoint_exact"] for i in range(cfg["summary_window_points"])]
                        item = {"mean": mean(values), "stable": min(values) >= cfg["success_accuracy"]}
                        windows[n, ds, ms, arm, stage] = item
                        group.append(item)
                        ends.append(end)
                summaries[n, stage, arm] = {
                    "n": n, "stage": stage, "arm": arm, "trials": len(group),
                    "complete_panel": len(data_seeds) == len(cfg["data_seeds"]),
                    "test_mean": mean([r["mean"] for r in group]),
                    "test_min": min(r["mean"] for r in group),
                    "test_max": max(r["mean"] for r in group),
                    "stable_success_count": sum(r["stable"] for r in group),
                    "endpoint_fit_count": sum(r["train_endpoint_bce"] <= cfg["fit_bce"] for r in ends),
                    "train_exact_100_count": sum(r["train_endpoint_exact"] == 1 for r in ends),
                    "train_bce_mean": mean([r["train_endpoint_bce"] for r in ends]),
                    "train_aux_bce_mean": mean([(bits * r["train_all_output_bce"] -
                                                  r["train_endpoint_bce"]) / (bits - 1) for r in ends]),
                }
            for ds in sorted(data_seeds):
                for ms in cfg["model_seeds"]:
                    d, s = [windows[n, ds, ms, arm, stage]["mean"] for arm in ["direct", "scaffold"]]
                    pairs[n, ds, ms, stage] = {
                        "direct_accuracy": d, "scaffold_accuracy": s,
                        "test_difference_pp": 100 * (s - d),
                        "both_endpoint_fit": all(lookup[n, ds, ms, arm, stage]["train_endpoint_bce"] <=
                                                 cfg["fit_bce"] for arm in ["direct", "scaffold"]),
                    }
    seen_summaries, seen_pairs = set(), set()
    for path in analyses:
        data = read(path)
        for row in data["summary"]:
            key = tuple(row[k] for k in ["n", "stage", "arm"])
            assert key not in seen_summaries
            seen_summaries.add(key)
            for name, expected in row.items():
                actual = summaries[key][name]
                if isinstance(expected, float):
                    close(actual, expected)
                else:
                    assert actual == expected, (key, name)
        for row in data["pairs"]:
            key = tuple(row[k] for k in ["n", "data_seed", "model_seed", "stage"])
            assert key not in seen_pairs
            seen_pairs.add(key)
            for name, actual in pairs[key].items():
                close(actual, row[name])
    assert seen_summaries == set(summaries) and seen_pairs == set(pairs)
    final = []
    for n in sorted(by_n):
        d, s = [summaries[n, 16000, arm] for arm in ["direct", "scaffold"]]
        final.append({"n": n, "data_seeds": sorted(by_n[n]), "trials_per_arm": d["trials"],
                      "complete_panel": d["complete_panel"],
                      "direct_test_percent": 100 * d["test_mean"],
                      "scaffold_test_percent": 100 * s["test_mean"],
                      "paired_change_pp": 100 * (s["test_mean"] - d["test_mean"]),
                      "stable_99_counts": [d["stable_success_count"], s["stable_success_count"]],
                      "fit_bce_counts": [d["endpoint_fit_count"], s["endpoint_fit_count"]],
                      "train_exact_100_counts": [d["train_exact_100_count"], s["train_exact_100_count"]],
                      "train_bce_means": [d["train_bce_mean"], s["train_bce_mean"]]})
    focal_counts = [2048] if bits == 14 else [3000, 4096, 6000, 8192]
    budgets, datasets = [], []
    for n in focal_counts:
        for stage in cfg["stages"]:
            d, s = [summaries[n, stage, arm]["test_mean"] for arm in ["direct", "scaffold"]]
            budgets.append({"n": n, "steps": stage, "direct_percent": 100 * d,
                            "scaffold_percent": 100 * s, "change_pp": 100 * (s - d)})
        for ds in sorted(by_n[n]):
            d, s = [mean([windows[n, ds, ms, arm, 16000]["mean"] for ms in cfg["model_seeds"]])
                    for arm in ["direct", "scaffold"]]
            datasets.append({"n": n, "data_seed": ds, "direct_percent": 100 * d,
                             "scaffold_percent": 100 * s, "change_pp": 100 * (s - d)})
    final_rows = [r for key, r in lookup.items() if key[-1] == 16000]
    return {"bits": bits, "parameters_per_model": parameters, "final": final,
            "selected_budget_comparisons": budgets, "selected_by_dataset": datasets,
            "audit": {"completed_groups": len(jobs), "models": len(final_rows),
                      "summary_rows_recomputed": len(summaries), "paired_rows_recomputed": len(pairs),
                      "endpoint_fit_count": sum(r["train_endpoint_bce"] <= .001 for r in final_rows),
                      "train_exact_100_count": sum(r["train_endpoint_exact"] == 1 for r in final_rows),
                      "nonexact_final_rows": [{k: r[k] for k in ["n", "data_seed", "model_seed", "arm",
                                                                "train_endpoint_bce", "train_endpoint_exact"]}
                                              for r in final_rows if r["train_endpoint_exact"] != 1],
                      "saved_forward_max_error": max(j["numpy_final_forward_max_error"] for j in jobs)}}


def check_bilingual(results):
    result14, result16 = results["mod3_14bit"], results["mod3_16bit"]
    paths = [OUT / "README_zh.md", OUT / "README_en.md"]
    assert all(p.is_file() for p in paths), "缺少中英文正文"
    documents = [p.read_text(encoding="utf-8") for p in paths]
    blocks = []
    for doc in documents:
        parts = re.split(r"<!-- sync:([a-z0-9_-]+) -->", doc)
        assert not parts[0].strip()
        blocks.append(list(zip(parts[1::2], parts[2::2])))
    assert [k for k, _ in blocks[0]] == [k for k, _ in blocks[1]]
    assert len({k for k, _ in blocks[0]}) == len(blocks[0])
    for (key, zh), (_, en) in zip(*blocks):
        number = r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
        assert sorted(re.findall(number, zh)) == sorted(re.findall(number, en)), key
        zh_tables = [line for line in zh.splitlines() if line.startswith("|")]
        en_tables = [line for line in en.splitlines() if line.startswith("|")]
        assert [re.findall(number, line) for line in zh_tables] == [re.findall(number, line) for line in en_tables], key
        assert re.findall(r"\$\$([\s\S]*?)\$\$", zh) == re.findall(r"\$\$([\s\S]*?)\$\$", en), key
        assert re.findall(r"\]\(([^)]+)\)", zh) == re.findall(r"\]\(([^)]+)\)", en), key
        for link in re.findall(r"\]\(([^)]+)\)", zh):
            if not link.startswith("#"):
                target = (OUT / link.split("#")[0]).resolve()
                assert target.is_relative_to(OUT) and target.is_file(), f"非包内有效链接：{link}"
    zh_blocks = dict(blocks[0])
    for name, result in [("table-14", result14), ("table-16", result16)]:
        for row in result["final"]:
            if not row["complete_panel"]:
                continue
            d, s = row["stable_99_counts"]
            digits = 3 if result["bits"] == 14 and row["n"] == 8192 else 2
            expected = (f"| {row['n']} | {row['direct_test_percent']:.2f}% | "
                        f"{row['scaffold_test_percent']:.2f}% | {row['paired_change_pp']:+.{digits}f} | "
                        f"{d}/6 -> {s}/6 |")
            assert expected in zh_blocks[name], expected
    for row in result14["selected_by_dataset"]:
        expected = (f"| {row['data_seed']} | {row['direct_percent']:.2f}% | "
                    f"{row['scaffold_percent']:.2f}% | {row['change_pp']:+.2f} |")
        assert expected in zh_blocks["table-14-datasets"], expected
    for name, result in [("table-14-budgets", result14), ("table-16-budgets", result16)]:
        for row in result["selected_budget_comparisons"]:
            if result["bits"] == 16 and row["n"] not in [3000, 6000]:
                continue
            prefix = f"{row['n']} | " if result["bits"] == 16 else ""
            expected = (f"| {prefix}{row['steps']} | {row['direct_percent']:.2f}% | "
                        f"{row['scaffold_percent']:.2f}% | {row['change_pp']:+.2f} |")
            assert expected in zh_blocks[name], expected
    return {"status": "passed", "aligned_blocks": len(blocks[0]),
            "checks": ["block_order", "numeric_tokens", "display_formulas", "package_links",
                       "mod3_main_budget_dataset_tables_vs_key_results"],
            "semantic_translation_review": "manual"}


def build():
    global ROOT, P14, P16
    if len(OUT.parents) < 5:
        raise FileNotFoundError("完整复算需要仓库布局和本地原始结果；仅有公开包时请使用 --public-only。")
    ROOT = OUT.parents[4]
    P14 = ROOT / "research/mod3_scaffold_sample_trend_20261102"
    P16 = ROOT / "research/mod3_scaffold_sample_trend_16bit_20261102"
    for base in [P14, P16]:
        if not (base / "results/MANIFEST.json").is_file():
            raise FileNotFoundError(f"完整复算缺少本地原始结果：{base}；仅检查公开包请使用 --public-only。")
    FINGERPRINTS.clear()
    HASH_CHECKS.clear()
    for key in MAX_ERRORS:
        MAX_ERRORS[key] = 0.0
    roots14 = [P14 / "results", P14 / "results_midpoint"]
    roots16 = [P16 / "results", P16 / "results_refined"]
    check_sources(P14, roots14)
    check_sources(P16, roots16)
    result14 = mod3(14, roots14, [P14 / "results_combined/ANALYSIS.json"])
    result16 = mod3(16, roots16, [p / "ANALYSIS.json" for p in roots16])
    stopped = read(P16 / "results_refined/STOPPED_SUMMARY.json")
    assert stopped["stopped_at_user_request"] and stopped["saved_models"] == 30
    marker = P16 / "results_refined/STOP_AFTER_GROUP"
    assert marker.is_file()
    fingerprint(marker)
    result16["queue"] = {"status": "stopped_at_user_request", "stop_marker_present": True,
                         "complete_refinement_counts": stopped["full_data_counts"],
                         "partial_counts": stopped["partial_data_counts"],
                         "unstarted_counts": stopped["unstarted_counts"], "cancelled_counts": [12000, 16000]}
    assert [r["n"] for r in result16["final"] if not r["complete_panel"]] == [2500]
    for path in [P14 / "results_combined/AUDIT.json", P16 / "results/AUDIT.json"]:
        read(path)
    for name in HISTORICAL_SOURCES:
        fingerprint(ROOT / name)
    for path in [P14 / "results_combined/summary.csv", P16 / "results/summary.csv",
                 P16 / "results_refined/summary.csv"]:
        fingerprint(path)
    key_results = {"schema_version": 2, "review_date": "2026-11-06", "reference_commit": REFERENCE,
                   "publication_scope": "mod3_input_sample_efficiency_with_historical_accessibility_context",
                   "units": {"test_accuracy": "percent", "change": "percentage_points"},
                   "protocol": {
                       "data_seeds": [202611020, 202611021], "model_seeds_per_dataset": [0, 1, 2],
                       "complete_panel_trials_per_arm": 6, "independent_datasets": 2,
                       "training_steps": 16000, "budget_steps": [4000, 8000, 16000],
                       "paired_initial_weights": True, "paired_minibatch_sequence": True,
                       "input_budget_unit": "distinct_training_inputs",
                       "endpoint_label_bits_per_input": 2,
                       "auxiliary_label_bits_per_input": {"14": 26, "16": 30},
                       "endpoint_loss_coefficient": 1, "auxiliary_mean_loss_coefficient": 1,
                       "test_time_auxiliary_labels": False,
                       "total_supervision_budget_matched": False},
                   "interpretation": {
                       "finding": "解耦辅助监督能帮助优化，并在已测固定预算下减少达到高泛化所需的输入样本。",
                       "distinct_axes": "优化可达性与规则/样本困难不是同一轴；较低训练损失不等于较少输入样本需求。",
                       "supervision_boundary": "前缀标签增加了监督信息；输入样本减少不代表总监督预算减少。"},
                   "mod3_14bit": result14, "mod3_16bit": result16,
                   "historical_30bit_mod3": {
                       "evidence_level": "dated_2026-09-13_notes_not_raw_checkpoint_recomputed",
                       "role": "optimization_accessibility_context_not_new_sample_measurements",
                       "new_measurements": False, "source_paths": HISTORICAL_SOURCES},
                   "unresolved": ["relative_static_Neural_K_reduction", "exact_sample_threshold",
                                  "infinite_budget_gap", "mechanism_contribution_separation",
                                  "total_supervision_budget_reduction", "cross_architecture_generality"]}
    public_counts = check_public_results(key_results)
    audit = {"scope": "只读JSON复算与源码哈希；未加载权重，未重新执行前向、数据划分或训练。",
             "framework_imports": [], "gpu_used": False, "training_started": False, "sampling_started": False,
             "historical_sources_written": False, "stop_marker_sha256": FINGERPRINTS[marker.relative_to(ROOT).as_posix()],
             "mod3_models": result14["audit"]["models"] + result16["audit"]["models"],
             "mod3_summary_rows": result14["audit"]["summary_rows_recomputed"] + result16["audit"]["summary_rows_recomputed"],
             "mod3_paired_rows": result14["audit"]["paired_rows_recomputed"] + result16["audit"]["paired_rows_recomputed"],
             "maximum_absolute_errors": MAX_ERRORS, "bilingual": check_bilingual(key_results),
             "public_summary_checks": public_counts,
             "mode": "full_local_recomputation", "requires_local_original_results": True,
             "hash_assertions": len(HASH_CHECKS), "source_files_unchanged_during_audit": True}
    # 再读哈希，检查汇总期间来源未被并发改写。
    for name, expected in FINGERPRINTS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    fingerprint(Path(__file__))
    hashes = {"algorithm": "SHA256", "reference_commit": REFERENCE,
              "scope": "Mod3所用源码、冻结副本、JSON/CSV与历史引用笔记；不包括大型权重。",
              "files": dict(sorted(FINGERPRINTS.items())),
              "package_files": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                for name in ["README_zh.md", "README_en.md", "recompute.py"]}}
    for name, value in [("key_results.json", key_results), ("audit.json", audit)]:
        hashes["package_files"][name] = hashlib.sha256(artifact_bytes(name, value)).hexdigest()
    return {"key_results.json": key_results, "source_sha256.json": hashes, "audit.json": audit}


def render(content):
    return json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def artifact_bytes(name, content):
    target = OUT / name
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) == content:
        return target.read_bytes()
    return render(content).encode("utf-8")


def check_public_results(results):
    """检查已发布摘要的内部关系，不将其视为原始逐运行记录的独立复算。"""
    def near(a, b):
        assert math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-9, (a, b)

    def finite(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    finite(results)
    assert set(results) == {"schema_version", "review_date", "reference_commit", "publication_scope",
                            "units", "protocol", "interpretation", "mod3_14bit", "mod3_16bit",
                            "historical_30bit_mod3", "unresolved"}
    assert results["schema_version"] == 2 and results["reference_commit"] == REFERENCE
    assert results["publication_scope"] == "mod3_input_sample_efficiency_with_historical_accessibility_context"
    assert results["units"] == {"test_accuracy": "percent", "change": "percentage_points"}
    protocol = results["protocol"]
    assert protocol["data_seeds"] == [202611020, 202611021]
    assert protocol["model_seeds_per_dataset"] == [0, 1, 2]
    assert protocol["complete_panel_trials_per_arm"] == 6 and protocol["independent_datasets"] == 2
    assert protocol["training_steps"] == 16000 and protocol["budget_steps"] == [4000, 8000, 16000]
    assert protocol["paired_initial_weights"] and protocol["paired_minibatch_sequence"]
    assert protocol["input_budget_unit"] == "distinct_training_inputs"
    assert protocol["endpoint_label_bits_per_input"] == 2
    assert protocol["auxiliary_label_bits_per_input"] == {"14": 26, "16": 30}
    assert protocol["endpoint_loss_coefficient"] == protocol["auxiliary_mean_loss_coefficient"] == 1
    assert protocol["test_time_auxiliary_labels"] is False
    assert protocol["total_supervision_budget_matched"] is False
    history = results["historical_30bit_mod3"]
    assert history["evidence_level"] == "dated_2026-09-13_notes_not_raw_checkpoint_recomputed"
    assert history["role"] == "optimization_accessibility_context_not_new_sample_measurements"
    assert history["new_measurements"] is False and history["source_paths"] == HISTORICAL_SOURCES
    counts = {"mod3_final_rows": 0, "budget_rows": 0, "dataset_rows": 0}
    for bits in [14, 16]:
        data = results[f"mod3_{bits}bit"]
        assert data["bits"] == bits
        assert data["parameters_per_model"] == 2137088 + 2050 * bits
        rows = {r["n"]: r for r in data["final"]}
        assert len(rows) == len(data["final"])
        for n, row in rows.items():
            assert 0 < n <= 2 ** (bits - 1)
            seeds = row["data_seeds"]
            assert len(seeds) == len(set(seeds)) and set(seeds) <= {202611020, 202611021}
            trials = row["trials_per_arm"]
            assert trials == 3 * len(seeds)
            assert row["complete_panel"] == (len(seeds) == 2)
            for key in ["direct_test_percent", "scaffold_test_percent"]:
                assert 0 <= row[key] <= 100
            near(row["paired_change_pp"], row["scaffold_test_percent"] - row["direct_test_percent"])
            for key in ["stable_99_counts", "fit_bce_counts", "train_exact_100_counts"]:
                assert len(row[key]) == 2 and all(isinstance(v, int) and 0 <= v <= trials for v in row[key])
            assert len(row["train_bce_means"]) == 2 and min(row["train_bce_means"]) >= 0
        counts["mod3_final_rows"] += len(rows)
        for row in data["selected_budget_comparisons"]:
            assert row["n"] in rows and row["steps"] in [4000, 8000, 16000]
            near(row["change_pp"], row["scaffold_percent"] - row["direct_percent"])
            if row["steps"] == 16000:
                for arm in ["direct", "scaffold"]:
                    near(row[f"{arm}_percent"], rows[row["n"]][f"{arm}_test_percent"])
        counts["budget_rows"] += len(data["selected_budget_comparisons"])
        by_dataset = {}
        for row in data["selected_by_dataset"]:
            assert row["data_seed"] in rows[row["n"]]["data_seeds"]
            near(row["change_pp"], row["scaffold_percent"] - row["direct_percent"])
            by_dataset.setdefault(row["n"], []).append(row)
        for n, group in by_dataset.items():
            assert sorted(r["data_seed"] for r in group) == rows[n]["data_seeds"]
            for arm in ["direct", "scaffold"]:
                near(mean([r[f"{arm}_percent"] for r in group]), rows[n][f"{arm}_test_percent"])
        counts["dataset_rows"] += len(data["selected_by_dataset"])
        audit = data["audit"]
        models = sum(2 * r["trials_per_arm"] for r in rows.values())
        assert audit["models"] == models
        assert audit["completed_groups"] == sum(len(r["data_seeds"]) for r in rows.values())
        assert audit["summary_rows_recomputed"] == 6 * len(rows)
        assert audit["paired_rows_recomputed"] == 3 * models // 2
        assert audit["endpoint_fit_count"] == sum(sum(r["fit_bce_counts"]) for r in rows.values())
        assert audit["train_exact_100_count"] == sum(sum(r["train_exact_100_counts"]) for r in rows.values())
        exceptions = audit["nonexact_final_rows"]
        assert len(exceptions) == models - audit["train_exact_100_count"]
        assert len({(r["n"], r["data_seed"], r["model_seed"], r["arm"]) for r in exceptions}) == len(exceptions)
        assert all(0 <= r["train_endpoint_exact"] < 1 for r in exceptions)
        assert sum(r["train_endpoint_bce"] > .001 for r in exceptions) == models - audit["endpoint_fit_count"]
    data16 = results["mod3_16bit"]
    queue = data16["queue"]
    assert queue["status"] == "stopped_at_user_request" and queue["stop_marker_present"]
    assert queue["partial_counts"] == [r["n"] for r in data16["final"] if not r["complete_panel"]]
    observed = {r["n"] for r in data16["final"]}
    assert set(queue["complete_refinement_counts"]) <= observed
    assert not observed.intersection(queue["unstarted_counts"] + queue["cancelled_counts"])
    return counts


def public_only():
    """只读取包内文件；原始来源哈希仅作为出处，不检查缺省的研究目录。"""
    def local(name):
        return json.loads((OUT / name).read_text(encoding="utf-8-sig"))

    results, hashes, audit = [local(name) for name in ["key_results.json", "source_sha256.json", "audit.json"]]
    assert hashes["algorithm"] == "SHA256" and hashes["reference_commit"] == REFERENCE
    expected_names = PACKAGE_FILES - {"source_sha256.json"}
    assert {p.name for p in OUT.iterdir() if p.is_file() and not p.name.startswith(".")} == PACKAGE_FILES
    assert set(hashes["package_files"]) == expected_names
    for name, expected in hashes["package_files"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, f"包内文件哈希不符：{name}"
    assert hashes["files"][PACKAGE_PREFIX + "recompute.py"] == hashes["package_files"]["recompute.py"]
    for name, expected in hashes["files"].items():
        assert name.startswith("research/") and ".." not in Path(name).parts
        assert (name.startswith(("research/mod3_scaffold_sample_trend_20261102/",
                                 "research/mod3_scaffold_sample_trend_16bit_20261102/"))
                or name in HISTORICAL_SOURCES
                or name in {PACKAGE_PREFIX + "recompute.py",
                            "research/ca_phase_transition/experiment_mod3_endpoint_scaffold_matched_bce.py",
                            "research/ca_phase_transition/experiment_mod3_small_data_grokking.py"}), name
        assert re.fullmatch(r"[0-9a-f]{64}", expected), name
    counts = check_public_results(results)
    bilingual = check_bilingual(results)
    assert audit["bilingual"] == bilingual
    for published, key in [("mod3_models", "models"), ("mod3_summary_rows", "summary_rows_recomputed"),
                           ("mod3_paired_rows", "paired_rows_recomputed")]:
        assert audit[published] == sum(results[f"mod3_{bits}bit"]["audit"][key] for bits in [14, 16])
    assert audit["public_summary_checks"] == counts
    assert audit["framework_imports"] == []
    assert all(audit[key] is False for key in ["gpu_used", "training_started", "sampling_started",
                                              "historical_sources_written"])
    assert set(audit["maximum_absolute_errors"]) == {"mod3_statistics"}
    assert 0 <= audit["maximum_absolute_errors"]["mod3_statistics"] <= 1e-10
    assert audit["stop_marker_sha256"] == hashes["files"][
        "research/mod3_scaffold_sample_trend_16bit_20261102/results_refined/STOP_AFTER_GROUP"]
    return {"status": "passed", "mode": "public_only", "scope": "公开包内部一致性；未读取或独立复算原始逐运行记录。",
            "raw_results_read": False, "training_started": False, "sampling_started": False, "gpu_used": False,
            "package_hashes_checked": len(expected_names), "checked_rows": counts, "bilingual": bilingual,
            "not_verified": ["original_source_hashes", "per_run_metrics", "forward_passes",
                             "historical_measurements"]}


def main():
    if not __debug__:
        raise SystemExit("核验依赖断言，请不要使用 -O 或 -OO。")
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--public-only", action="store_true", help="仅检查公开包数据、表格、双语与文件哈希，不需要原始结果")
    modes.add_argument("--write", action="store_true", help="依赖本地原始结果，完整复算并写出本目录的三个 JSON")
    args = parser.parse_args()
    if args.public_only:
        print(render(public_only()), end="")
        return
    artifacts = build()
    for name, content in artifacts.items():
        target = OUT / name
        if args.write:
            target.write_bytes(artifact_bytes(name, content))
        else:
            assert json.loads(target.read_text(encoding="utf-8")) == content, f"产物不同：{name}"
    print(json.dumps(artifacts["audit.json"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
