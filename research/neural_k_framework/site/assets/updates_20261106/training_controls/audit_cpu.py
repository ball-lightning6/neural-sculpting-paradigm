"""--public-only 仅核对本包汇总；默认模式依赖本地原始结果，两者都不运行实验。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import zipfile

HERE = Path(__file__).resolve().parent
BASE_COMMIT = "3834fb3f6555c7c58218462502635b16b9b377f9"
PACKAGE_TABLES = ("copy_summary.csv", "rule30_summary.csv", "rule30_paired.csv", "sources_sha256.csv")
AGREEMENT_FILES = ("check_agreement.py", "endpoints.csv.gz", "seed_pairs.csv", "group_counts.csv")
PACKAGE_FILES = (*PACKAGE_TABLES, *AGREEMENT_FILES, "audit_summary.json", "audit_cpu.py", "README_zh.md", "README_en.md")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

SOURCES = {}
MANIFEST_CHECKS = []


def configure_local_sources():
    global REPO, COPY, HIGH, LOW, OLD, E31, E30, BASE, BASE_MODEL, np
    relative_base = Path("research/ca_phase_transition/experiment_rule30_bce_raw_mse_local.py")
    REPO = next((parent for parent in HERE.parents if (parent / relative_base).is_file()), None)
    if REPO is None:
        raise RuntimeError("默认模式需要未随本包提供的研究仓库和原始结果；仅有公开包时请使用 --public-only。")
    COPY = REPO / "research/copy_mse_amplitude_sgd_20261103"
    HIGH = REPO / "research/rule30_high_norm_penalty_20261103"
    LOW = REPO / "research/rule30_low_norm_penalty_20261103"
    OLD = REPO / "tmp/4bit_copy_small_target_local_20260910"
    E31 = REPO / "research/neural_k_framework/experiments/experiment_31_loss_amplitude_grokking"
    E30 = REPO / "research/neural_k_framework/experiments/experiment_30_weight_decay_static_landscape"
    BASE = REPO / relative_base
    BASE_MODEL = BASE.with_name("experiment_capacity_pressure_width_scan.py")
    try:
        import numpy as np
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / ".venv-gpu/Lib/site-packages"))
        import numpy as np


def package_digest(name):
    assert name in PACKAGE_FILES
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def register(path, expected=None, source_id=None):
    path = Path(path).resolve()
    key = source_id or path.relative_to(REPO).as_posix()
    if key not in SOURCES:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        SOURCES[key] = dict(path=key, bytes=path.stat().st_size,
                            sha256=digest.hexdigest(), recorded_sha256="", matches_recorded="")
    if expected:
        row = SOURCES[key]
        assert row["sha256"] == expected.lower(), f"来源哈希不符: {key}"
        row.update(recorded_sha256=expected.lower(), matches_recorded=True)
        MANIFEST_CHECKS.append(key)
    return path


def read_json(path):
    return json.loads(register(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with register(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-11):
    assert abs(float(a) - float(b)) <= tolerance * max(1, abs(float(b))), (a, b)


def audit_manifests(archive):
    manifests = [read_json(root / "MANIFEST.json") for root in (COPY, HIGH, LOW)]
    for root, manifest in zip((COPY, HIGH, LOW), manifests):
        for name, expected in manifest["sources"].items():
            register(root / name, expected)
        # 对目录内的文本配置和落盘汇总建立完整指纹；不执行其中的脚本。
        for path in sorted(root.rglob("*")):
            if path.suffix in {".json", ".csv", ".py", ".md"}:
                register(path)
                if path.suffix == ".json":
                    read_json(path)
    cm, hm, lm = manifests
    register(E31 / "experiment_4bit_copy_small_target_local.py", cm["original_script_sha256"])
    register(OLD / "results.npz", cm["original_archive_sha256"])
    register(COPY / "PILOT.json", cm["pilot_sha256"])
    for manifest in (hm, lm):
        register(BASE, manifest["base_source"])
        register(BASE_MODEL, manifest["model_source"])
        register(REPO / "tmp/rule30_bce_raw_mse_local_20260910/initial_weights.pt", manifest["old_initial_sha256"])
        register(REPO / "tmp/rule30_bce_raw_mse_local_20260910/final_weights.pt", manifest["reference"]["sha256"])
        register(archive, manifest["data_archive_sha256"], "external/results_capacity_pressure_width_scan_package.zip")
        ref = manifest["reference"]
        close(statistics.median(ref["radii"]) * ref["multiplier"], ref["cutoff"])
        close(ref["cutoff"] ** 2, ref["cutoff2"])
    keys = ("base_source", "model_source", "data_archive_sha256", "old_initial_sha256",
            "counts", "seeds", "steps", "batch", "learning_rate", "weight_decay", "dtype", "tf32")
    assert all(hm[k] == lm[k] for k in keys)
    assert hm["counts"] == [600, 750, 900, 1200] and hm["seeds"] == [0, 1, 2, 3]
    assert hm["lambda_value"] == lm["lambda_value"] == .1
    return manifests


def forward(weights, inputs):
    hidden = weights[:, :80].reshape(-1, 16, 5)
    h = np.tanh((np.einsum("nd,bhd->bnh", inputs, hidden[:, :, :4]) + hidden[:, None, :, 4]) / 2)
    return (np.einsum("bnh,bh->bn", h, weights[:, 80:96]) + weights[:, 96, None]) / 4, h


def audit_copy():
    metadata = read_json(COPY / "MODELS.json")
    summary = read_json(COPY / "SUMMARY.json")
    assert summary["status"] == "completed_step_cap" and summary["step"] == 100000
    pilot = read_json(COPY / "PILOT.json")
    assert pilot["training_only"] and all(v == .3 for v in pilot["selected"].values())
    assert pilot["steps"] == 2000 and pilot["models_per_candidate"] == 32
    for name, selected in pilot["selected"].items():
        candidates = [r for r in pilot["candidates"] if r["condition"] == name and r["valid_all"]]
        assert selected == min(candidates, key=lambda r: (r["q90_relative_objective"], r["raw_equivalent_lr"]))["raw_equivalent_lr"]
    with np.load(register(OLD / "results.npz"), allow_pickle=False) as a:
        x, y, train, test, initial = (a[k] for k in ("inputs", "labels", "train_indices", "test_indices", "initial_weights"))
    assert np.array_equal(y, x[:, -1]) and train.shape == (16, 12) and test.shape == (16, 4)
    assert np.array_equal(initial[:512], initial[512:1024]) and np.array_equal(initial[:512], initial[1024:])
    assert all(set(a).isdisjoint(b) and len(set(a) | set(b)) == 16 for a, b in zip(train, test))
    with np.load(register(COPY / "RESULTS.npz"), allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    assert len(metadata) == 3840
    assert all(np.array_equal(arrays["initial"][i], initial[m["seed"] - 71001]) for i, m in enumerate(metadata))
    ids = np.array([m["split"] for m in metadata])
    amplitudes = np.array([m["amplitude"] for m in metadata])
    bce = np.array([m["condition"] == "bce" for m in metadata])
    thresholds = np.where(bce, 1e-3, 1e-4)
    _, h0 = forward(arrays["initial"], x)
    h0c = h0 - h0.mean(1, keepdims=True)
    gram0 = h0c @ h0c.transpose(0, 2, 1)
    expected = {(r["stage"], r["condition"], r["arm"]): r for r in read_csv(COPY / "INDEPENDENT_GROUPS.csv")}
    rows, errors = [], []
    for stage in ("final", "best_train", "first_fit"):
        valid = np.isfinite(arrays[stage]).all(1)
        weights = np.nan_to_num(arrays[stage])
        z, h = forward(weights, x)
        correct = (z > 0) == y[None].astype(bool)
        ztrain = np.take_along_axis(z, train[ids], axis=1)
        signed = 2 * y[train[ids]] - 1
        loss = np.where(bce, np.logaddexp(0, -signed * ztrain).mean(1),
                        ((ztrain / amplitudes[:, None] - signed) ** 2).mean(1))
        train_acc = np.take_along_axis(correct, train[ids], axis=1).mean(1)
        unseen = np.take_along_axis(correct, test[ids], axis=1).mean(1)
        full = correct.all(1)
        fitted = valid & (train_acc == 1) & (loss <= thresholds)
        fids = ((z > 0) * (2 ** np.arange(16))[None]).sum(1)
        error = np.max(np.abs(loss[valid] - arrays[f"{stage}_objective"][valid]) /
                       np.maximum(1, np.abs(arrays[f"{stage}_objective"][valid])))
        assert error < 1e-9
        for key, values in (("unseen_accuracy", unseen), ("train_accuracy", train_acc),
                            ("full_recovery", full), ("function_id", fids)):
            assert np.array_equal(values[valid], arrays[f"{stage}_{key}"][valid]), (stage, key)
        hc = h - h.mean(1, keepdims=True)
        gram = hc @ hc.transpose(0, 2, 1)
        displacement = np.linalg.norm(h-h0, axis=(1, 2)) / np.linalg.norm(h0, axis=(1, 2))
        denominator = np.sqrt((gram**2).sum((1, 2)) * (gram0**2).sum((1, 2)))
        cka = (gram*gram0).sum((1, 2)) / np.maximum(denominator, 1e-30)
        diagnostics = dict(hidden_norm=np.linalg.norm(weights[:, :80], axis=1),
                           readout_norm=np.linalg.norm(weights[:, 80:], axis=1),
                           feature_displacement=displacement, feature_cka=cka)
        diagnostic_error = max(float(np.max(np.abs(v[valid]-arrays[f"{stage}_{k}"][valid])))
                               for k, v in diagnostics.items())
        assert diagnostic_error < 1e-11
        errors.append(dict(stage=stage, max_normalized_loss_error=float(error), predictions_match=True,
                           max_norm_feature_absolute_error=diagnostic_error))
        for condition, arm in dict.fromkeys((m["condition"], m["arm"]) for m in metadata):
            selected = np.array([i for i, m in enumerate(metadata) if (m["condition"], m["arm"]) == (condition, arm)])
            accepted, fit = selected[valid[selected]], selected[fitted[selected]]
            row = dict(stage=stage, condition=condition, arm=arm, models=len(selected), valid=len(accepted),
                       fitted=len(fit), unseen_all=float(unseen[accepted].mean()), unseen_fitted=float(unseen[fit].mean()),
                       full_all=int(full[accepted].sum()), full_fitted=int(full[fit].sum()),
                       median_fitted_feature_displacement=float(np.median(displacement[fit])),
                       median_fitted_feature_cka=float(np.median(cka[fit])))
            old = expected[stage, condition, arm]
            for ours, theirs in (("models", "n"), ("fitted", "fit_count"), ("full_fitted", "full_recovery_among_fitted"),
                                ("unseen_all", "mean_unseen_accuracy"), ("unseen_fitted", "mean_unseen_among_fitted")):
                close(row[ours], old[theirs])
            rows.append(row)
    history = read_json(COPY / "HISTORY.json")
    assert history[-1]["step"] == 100000 and history[-1]["groups"] == summary["stages"]["final"]
    snapshot = read_json(COPY / "SUMMARY_step018000.json")
    assert snapshot["status"] == "paused_at_checkpoint" and snapshot["step"] == 18000
    snapshot_checks = []
    with np.load(register(COPY / "RESULTS_step018000.npz"), allow_pickle=False) as a:
        for stage in ("final", "best_train"):
            z, _ = forward(a[stage], x)
            correct = (z > 0) == y[None].astype(bool)
            ztrain = np.take_along_axis(z, train[ids], axis=1)
            loss = np.where(bce, np.logaddexp(0, -(2*y[train[ids]]-1)*ztrain).mean(1),
                            ((ztrain/amplitudes[:, None]-(2*y[train[ids]]-1))**2).mean(1))
            fitted = (np.take_along_axis(correct, train[ids], axis=1).mean(1) == 1) & (loss <= thresholds)
            error = float(np.max(np.abs(loss-a[f"{stage}_objective"])/np.maximum(1, np.abs(a[f"{stage}_objective"]))))
            assert error < 1e-9
            for saved in snapshot["stages"][stage]:
                mask = np.array([(m["condition"], m["arm"]) == (saved["condition"], saved["arm"]) for m in metadata])
                assert int(fitted[mask].sum()) == saved["sufficiently_fitted"]
            snapshot_checks.append(dict(stage=stage, max_normalized_loss_error=error))
    old_config = read_json(OLD / "config.json")["settings"]
    assert old_config["optimizer"] == "Adam" and old_config["steps"] == 18000
    old_groups = [r for r in read_json(OLD / "analysis.json")["groups"] if r["stage"] == "final"]
    register(OLD / "verification.json")
    return rows, dict(status=summary["status"], step=summary["step"], models=3840,
                      protocol=dict(optimizer="full_batch_SGD", momentum=0, weight_decay=0,
                                    amplitudes=[.001, .01, .1, 1., 10.], raw_equivalent_rates=[.3, .15],
                                    actual_mse_rate="raw_equivalent_rate * amplitude**2",
                                    train_points=12, unseen_points=4, splits=16, primary_models_per_condition=512,
                                    half_rate_models_per_condition=128, eval_every=200, minimum_steps=18000,
                                    maximum_steps=100000, mse_relative_fit_threshold=1e-4, bce_fit_threshold=1e-3),
                      forward_checks=errors, exact_initial_pairing=True,
                      intermediate_snapshot=dict(status=snapshot["status"], step=18000, checks=snapshot_checks,
                          primary_fitted={r["condition"]: r["sufficiently_fitted"] for r in snapshot["stages"]["best_train"] if r["arm"] == "primary"}),
                      old_adam=dict(steps=old_config["steps"], lr=old_config["learning_rate"],
                                    lr_milestones=old_config["lr_milestones"], lr_decay=old_config["lr_decay"], groups=old_groups),
                      raw_mse_fit_limits={str(a): a*a*1e-4 for a in (.001, .01, .1, 1., 10.)})


def audit_ca(root):
    summary, history = read_json(root / "SUMMARY.json"), read_json(root / "HISTORY.json")
    assert summary["status"] == "completed" and summary["step"] == 20000
    assert summary["last_rows"] == history[-1]
    assert [rows[0]["step"] for rows in history[-3:]] == [19000, 19500, 20000]
    with np.load(register(root / "DATA.npz"), allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    all_ids = np.concatenate(list(data.values()))
    assert len(np.unique(all_ids)) == len(all_ids)
    packed = {}
    max_error = 0.
    with np.load(register(root / "PREDICTIONS.npz"), allow_pickle=False) as predictions:
        assert list(predictions["steps"]) == [rows[0]["step"] for rows in history]
        for split in ("monitor", "probe"):
            assert len(data[split + "_ids"]) == 4096
            bits = ((data[split + "_ids"][:, None] >> np.arange(29, -1, -1)) & 1).astype(bool)
            target = np.roll(bits, 1, axis=1) ^ (bits | np.roll(bits, -1, axis=1))
            packed[split] = predictions[split]
            for i, saved in enumerate(packed[split]):
                correct = np.unpackbits(saved, axis=-1, count=30, bitorder="little").astype(bool) == target[None]
                for j, row in enumerate(history[i]):
                    max_error = max(max_error, abs(correct[j].mean() - row[split + "_bit"]),
                                    abs(correct[j].all(1).mean() - row[split + "_exact"]))
    assert max_error < 1e-12
    crossing_expected = read_json(root / "CROSSINGS.json")
    crossing_actual = {}
    for rows in history:
        for r in rows:
            assert r["fitted"] == (r["train_exact"] == 1 and r["train_bce"] <= 1e-4)
            close(r["total_objective"], r["train_bce"] + r["penalty"])
            for threshold in (1e-2, 1e-3, 1e-4):
                key = f"{r['model_index']}:{threshold}"
                if r["train_exact"] == 1 and r["train_bce"] <= threshold and key not in crossing_actual:
                    crossing_actual[key] = {**r, "threshold": threshold}
    assert crossing_actual == crossing_expected
    csv_crossings = read_csv(root / "MATCHED_THRESHOLDS.csv")
    assert len(csv_crossings) == len(crossing_actual) == 96
    for row in csv_crossings:
        original = crossing_actual[f"{row['model_index']}:{float(row['threshold'])}"]
        for key in ("step", "threshold", "train_bce", "probe_exact"):
            close(row[key], original[key])
    per_model = read_csv(root / "PER_MODEL.csv")
    for r, saved in zip(history[-1], per_model):
        for key in ("train_bce", "probe_bit", "probe_exact", "radius", "effective_radius", "penalty"):
            close(r[key], saved[key])
    return history, data, packed, dict(evaluation_nodes=len(history), max_prediction_accuracy_error=float(max_error),
                                      first_threshold_crossings=96, train_fit_verified_from_saved_metrics=True,
                                      weight_forward_recomputed=False)


def combine_ca(high, low):
    hh, hd, hp, ha = high
    lh, ld, lp, la = low
    assert hd.keys() == ld.keys() and all(np.array_equal(hd[k], ld[k]) for k in hd)
    baseline_h = [r for rows in hh for r in rows if r["loss"] == "baseline"]
    baseline_l = [r for rows in lh for r in rows if r["loss"] == "baseline"]
    fields = ("train_bce", "train_exact", "train_bit", "probe_bce", "probe_exact", "probe_bit",
              "monitor_bce", "monitor_exact", "monitor_bit", "radius", "effective_radius", "raw_full_norm")
    differences = {key: max(abs(a[key] - b[key]) for a, b in zip(baseline_h, baseline_l)) for key in fields}
    assert all(value == 0 for value in differences.values())
    base_indices = [r["model_index"] for r in hh[-1] if r["loss"] == "baseline"]
    assert all(np.array_equal(hp[k][:, base_indices], lp[k][:, base_indices]) for k in hp)
    rows = []
    old = {(r["arm"], int(r["n"])): r for r in read_csv(HIGH / "THREE_ARM_GROUPS.csv")}
    for arm, history in (("baseline", hh), ("high_norm", hh), ("low_norm", lh)):
        for n in (600, 750, 900, 1200):
            group = [r for r in history[-1] if r["loss"] == arm and r["n"] == n]
            stable = sum(r["fitted"] and all(node[r["model_index"]]["probe_exact"] >= .99 for node in history[-3:]) for r in group)
            row = dict(arm=arm, n=n, models=4, fitted=sum(r["fitted"] for r in group), stable_successes=stable,
                       mean_probe_exact=statistics.mean(r["probe_exact"] for r in group),
                       mean_probe_bit=statistics.mean(r["probe_bit"] for r in group),
                       max_train_bce=max(r["train_bce"] for r in group),
                       mean_radius=statistics.mean(r["radius"] for r in group),
                       min_radius=min(r["radius"] for r in group), max_radius=max(r["radius"] for r in group),
                       mean_effective_radius=statistics.mean(r["effective_radius"] for r in group),
                       max_penalty=max(r["penalty"] for r in group))
            for key in ("models", "fitted", "stable_successes", "mean_probe_exact", "max_train_bce", "mean_radius", "mean_effective_radius"):
                close(row[key], old[arm, n][key])
            rows.append(row)
    paired = []
    for arm, history in (("high_norm", hh), ("low_norm", lh)):
        for r in history[-1]:
            if r["loss"] != arm:
                continue
            b = next(b for b in hh[-1] if b["loss"] == "baseline" and (b["n"], b["seed"]) == (r["n"], r["seed"]))
            paired.append(dict(arm=arm, n=r["n"], seed=r["seed"],
                               probe_difference_pp=100*(r["probe_exact"]-b["probe_exact"]),
                               radius_ratio=r["radius"]/b["radius"], effective_radius_ratio=r["effective_radius"]/b["effective_radius"]))
    minima = {arm: min(r["n"] for r in rows if r["arm"] == arm and r["stable_successes"] >= 3)
              for arm in ("baseline", "high_norm", "low_norm")}
    return rows, paired, dict(high=ha, low=la, data_identical=True, baseline_metric_max_differences=differences,
                              baseline_predictions_identical=True, unique_paired_conditions=48, physical_runs=64,
                              smallest_grid_n_with_three_of_four_successes=minima)


def bilingual_check(copy_rows, ca_rows):
    docs = [(HERE / name).read_text(encoding="utf-8") for name in ("README_zh.md", "README_en.md")]
    pattern = r"<!-- block:([a-z0-9_]+) -->\s*(.*?)(?=<!-- block:|\Z)"
    blocks = [dict(re.findall(pattern, doc, re.S)) for doc in docs]
    assert blocks[0].keys() == blocks[1].keys() and blocks[0], "双语段落不对应"
    number = r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    for key in blocks[0]:
        a, b = blocks[0][key], blocks[1][key]
        # 忽略链接地址和标题编号之外的语言差异，逐段比较数值多重集及展示公式。
        a, b = (re.sub(r"\]\([^)]*\)", "]", t) for t in (a, b))
        assert Counter(re.findall(number, a)) == Counter(re.findall(number, b)), f"双语数字不一致: {key}"
        assert re.findall(r"\$\$(.*?)\$\$", a, re.S) == re.findall(r"\$\$(.*?)\$\$", b, re.S), f"双语公式不一致: {key}"
    def table(block):
        return [[cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in block.splitlines() if line.startswith("|")][2:]

    def percent(text, value):
        assert text == ("100%" if value == 1 else f"{100*value:.4f}%"), (text, value)

    with (HERE / "group_counts.csv").open(encoding="utf-8", newline="") as stream:
        agreement = {(int(r["n"]), r["arm"]): int(r["agreeing_pair_rings"]) / (6 * 4096)
                     for r in csv.DictReader(stream)}
    assert set(agreement) == {(r["n"], r["arm"]) for r in ca_rows}
    for document in blocks:
        for block_name, arm in (("copy_primary", "primary"), ("copy_half", "half_lr")):
            selected = [r for r in copy_rows if r["stage"] == "best_train" and r["arm"] == arm]
            cells = table(document[block_name])
            assert len(cells) == len(selected) == 6
            for cell, row in zip(cells, selected):
                assert cell[1] == f"{row['fitted']}/{row['models']}"
                if arm == "primary":
                    percent(cell[2], row["unseen_all"])
                    percent(cell[3], row["unseen_fitted"])
                    assert cell[4] == f"{row['full_fitted']}/{row['fitted']}"
                else:
                    percent(cell[2], row["unseen_fitted"])
        for cell in table(document["copy_thresholds"]):
            a = float(cell[0])
            for text, expected in zip(cell[1:], (a*a*1e-4, .3*a*a, .15*a*a)):
                assert abs(float(text)-expected) <= 1e-12*expected
        ca_cells = table(document["ca_table"])
        assert [(int(c[0]), c[1]) for c in ca_cells] == [(n, arm) for n in (600, 750, 900, 1200)
                                                          for arm in ("baseline", "high_norm", "low_norm")]
        for cell in ca_cells:
            row = next(r for r in ca_rows if r["n"] == int(cell[0]) and r["arm"] == cell[1])
            percent(cell[2], row["mean_probe_exact"])
            percent(cell[3], agreement[(row["n"], row["arm"])])
            percent(cell[4], row["mean_probe_bit"])
            assert cell[5] == f"{row['stable_successes']}/4" and cell[6] == f"{row['fitted']}/4"
    for doc in docs:
        for target in re.findall(r"\]\(([^)]+)\)", doc):
            assert target in PACKAGE_FILES, f"公开稿含包外链接: {target}"
            assert (HERE / target).is_file(), f"本包链接失效: {target}"
    return dict(aligned_blocks=len(blocks[0]), numeric_tokens_identical=True, display_formulas_identical=True,
                tables_match_supplied_values=True, package_links_only=True, package_links_exist=True)


def audit_public_package():
    # 只读取发布包；来源清单中的路径是记录，不是本模式的读取目标。
    report = json.loads((HERE / "audit_summary.json").read_text(encoding="utf-8"))
    assert set(report["package_sha256"]) == set(PACKAGE_TABLES)
    for name, expected in report["package_sha256"].items():
        assert package_digest(name) == expected, f"本包表格哈希不符: {name}"
    assert report["agreement_20261109_sha256"] == {name: package_digest(name) for name in AGREEMENT_FILES}
    assert package_digest("audit_cpu.py") == report["audit_script_sha256"]
    assert set(report["document_sha256"]) == {"README_zh.md", "README_en.md"}
    for name, expected in report["document_sha256"].items():
        assert package_digest(name) == expected, f"本包文档哈希不符: {name}"

    def table(name, integers=(), labels=()):
        with (HERE / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key, value in row.items():
                if key not in labels:
                    row[key] = int(value) if key in integers else float(value)
                    assert math.isfinite(row[key]), (name, key)
        return rows

    copy_rows = table("copy_summary.csv", ("models", "valid", "fitted", "full_all", "full_fitted"),
                      ("stage", "condition", "arm"))
    ca_rows = table("rule30_summary.csv", ("n", "models", "fitted", "stable_successes"), ("arm",))
    paired = table("rule30_paired.csv", ("n", "seed"), ("arm",))
    conditions = ("mse_0p001", "mse_0p01", "mse_0p1", "mse_1", "mse_10", "bce")
    stages, arms = ("final", "best_train", "first_fit"), ("primary", "half_lr")
    copy_index = {(r["stage"], r["arm"], r["condition"]): r for r in copy_rows}
    assert len(copy_rows) == len(copy_index) == 36
    assert set(copy_index) == {(stage, arm, condition) for stage in stages for arm in arms for condition in conditions}
    for row in copy_rows:
        assert row["models"] == (512 if row["arm"] == "primary" else 128)
        assert 0 < row["fitted"] <= row["valid"] <= row["models"]
        assert row["valid"] == (row["fitted"] if row["stage"] == "first_fit" else row["models"])
        assert 0 <= row["full_fitted"] <= min(row["fitted"], row["full_all"]) <= row["valid"]
        assert row["full_all"] <= row["valid"]
        for metric, count in (("unseen_all", row["valid"]), ("unseen_fitted", row["fitted"])):
            assert 0 <= row[metric] <= 1
            numerator = row[metric] * count * 4
            close(numerator, round(numerator))
    for arm in arms:
        for condition in conditions:
            final, best, first = (copy_index[stage, arm, condition] for stage in stages)
            for key in ("fitted", "full_all", "full_fitted", "unseen_all", "unseen_fitted"):
                close(final[key], best[key])
            for key in ("fitted", "full_fitted", "unseen_fitted"):
                close(first[key], best[key])
    primary = {condition: dict(fitted=copy_index["best_train", "primary", condition]["fitted"],
                              not_sufficiently_fitted=512-copy_index["best_train", "primary", condition]["fitted"],
                              unseen_fitted=copy_index["best_train", "primary", condition]["unseen_fitted"])
               for condition in conditions}
    assert report["copy"]["models"] == sum(r["models"] for r in copy_rows if r["stage"] == "final")
    for amplitude in report["copy"]["protocol"]["amplitudes"]:
        assert math.isclose(report["copy"]["raw_mse_fit_limits"][str(amplitude)],
                            amplitude**2 * report["copy"]["protocol"]["mse_relative_fit_threshold"],
                            rel_tol=1e-12, abs_tol=0)

    counts, ca_arms = (600, 750, 900, 1200), ("baseline", "high_norm", "low_norm")
    ca_index = {(r["arm"], r["n"]): r for r in ca_rows}
    assert len(ca_rows) == len(ca_index) == 12
    assert set(ca_index) == {(arm, n) for arm in ca_arms for n in counts}
    for row in ca_rows:
        assert row["models"] == row["fitted"] == 4 and 0 <= row["stable_successes"] <= 4
        assert 0 <= row["mean_probe_exact"] <= row["mean_probe_bit"] <= 1
        assert 0 <= row["max_train_bce"] <= 1e-4
        assert 0 < row["min_radius"] <= row["mean_radius"] <= row["max_radius"]
    paired_keys = {(r["arm"], r["n"], r["seed"]) for r in paired}
    assert len(paired) == len(paired_keys) == 32
    assert paired_keys == {(arm, n, seed) for arm in ca_arms[1:] for n in counts for seed in range(4)}
    mean_differences = []
    for arm in ca_arms[1:]:
        for n in counts:
            group = [r for r in paired if r["arm"] == arm and r["n"] == n]
            difference = statistics.mean(r["probe_difference_pp"] for r in group)
            close(difference, 100*(ca_index[arm, n]["mean_probe_exact"]-ca_index["baseline", n]["mean_probe_exact"]))
            mean_differences.append(dict(arm=arm, n=n, difference_pp=difference))
    minima = {arm: min(r["n"] for r in ca_rows if r["arm"] == arm and r["stable_successes"] >= 3)
              for arm in ca_arms}
    assert minima == report["rule30"]["smallest_grid_n_with_three_of_four_successes"]
    ref = report["rule30_reference"]
    close(ref["high_cutoff"], .85*ref["median"])
    close(ref["low_cutoff"], 1.5*ref["median"])

    with (HERE / "sources_sha256.csv").open(encoding="utf-8", newline="") as stream:
        sources = list(csv.DictReader(stream))
    assert len(sources) == len({r["path"] for r in sources}) == report["source_files"]
    recorded = 0
    for row in sources:
        assert int(row["bytes"]) >= 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert not Path(row["path"]).is_absolute() and not re.match(r"[A-Za-z]:", row["path"])
        if row["recorded_sha256"]:
            assert row["recorded_sha256"] == row["sha256"] and row["matches_recorded"] == "True"
            recorded += 1
    assert recorded == report["recorded_hashes_verified"]
    bilingual = bilingual_check(copy_rows, ca_rows)
    assert bilingual == report["bilingual"]
    assert "numpy" not in sys.modules and "torch" not in sys.modules
    return dict(status="PASS", mode="public_only", scope="published_summary_consistency_only",
                copy_primary=primary, rule30_mean_differences=mean_differences,
                rule30_grid_from_published_success_counts=minima, source_manifest_rows=len(sources),
                bilingual=bilingual, package_hashes_match=True, original_source_hashes_rechecked=False,
                raw_results_recomputed=False, experiments_rerun=False, numpy_imported=False,
                torch_imported=False, gpu_used=False)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-only", action="store_true", help="仅用标准库只读核对本包 CSV/JSON，不读取原始结果")
    parser.add_argument("--write", action="store_true", help="仅更新本目录的派生数据；默认只读核验")
    parser.add_argument("--archive", type=Path, help="旧Rule30数据包位置；默认从历史源码只读解析")
    if sys.flags.optimize:
        parser.error("核验依赖断言，不能使用 -O 或 -OO")
    args = parser.parse_args()
    if args.public_only:
        if args.write or args.archive is not None:
            parser.error("--public-only 不能与 --write 或 --archive 合用")
        print(json.dumps(audit_public_package(), ensure_ascii=False, indent=2))
        return
    configure_local_sources()
    archive = args.archive
    if archive is None:
        tree = ast.parse(BASE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OLD_PACKAGE" for t in node.targets):
                archive = Path(ast.literal_eval(node.value.args[0]))
    assert archive is not None
    manifests = audit_manifests(archive)
    copy_rows, copy_audit = audit_copy()
    high, low = audit_ca(HIGH), audit_ca(LOW)
    with zipfile.ZipFile(archive) as package:
        with np.load(io.BytesIO(package.read("dataset_split_ids.npz")), allow_pickle=False) as original:
            assert set(original.files) == set(high[1])
            assert all(np.array_equal(original[k], high[1][k]) for k in original.files)
    ca_rows, paired, ca_audit = combine_ca(high, low)
    ca_audit["data_ids_match_historical_archive"] = True
    for path in (E30 / "README.md", E30 / "RESULTS_AND_CONCLUSION.md", E30 / "MOTIVATION_AND_PREREGISTRATION.md",
                 E31 / "COPY_4BIT_MSE_AMPLITUDE_SERIES.md", E31 / "RULE30_CENTERED_MSE_AMPLITUDES.md",
                 E31 / "RULE30_MSE_PM01.md", REPO / "research/notes/20260910_4bit_copy_small_mse_training_zh.md"):
        register(path)
    bilingual = bilingual_check(copy_rows, ca_rows)
    assert "torch" not in sys.modules
    tables = {"copy_summary.csv": csv_text(copy_rows), "rule30_summary.csv": csv_text(ca_rows),
              "rule30_paired.csv": csv_text(paired), "sources_sha256.csv": csv_text([SOURCES[k] for k in sorted(SOURCES)])}
    report = dict(schema_version=2, audit_date="2026-11-06", base_commit=BASE_COMMIT,
                  verification_modes=dict(public_only="published_summary_consistency_only",
                                          default="requires_unbundled_local_sources; no_experiment_rerun"),
                  package_sha256={name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in tables.items()},
                  agreement_20261109_sha256={name: package_digest(name) for name in AGREEMENT_FILES},
                  training_started=False, smc_started=False, torch_imported=False, gpu_used=False,
                  numpy_version=np.__version__, source_files=len(SOURCES),
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  document_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                   for name in ("README_zh.md", "README_en.md")},
                  recorded_hashes_verified=len(set(MANIFEST_CHECKS)), copy=copy_audit, rule30=ca_audit,
                  rule30_reference=dict(median=statistics.median(manifests[1]["reference"]["radii"]),
                                        high_cutoff=manifests[1]["reference"]["cutoff"], low_cutoff=manifests[2]["reference"]["cutoff"],
                                        linear_parameters=2161694, total_parameters=2167838, lambda_values=[0, .1]),
                  bilingual=bilingual, historical_unit_tests_rerun=False,
                  source_hash_scope="text_metadata_and_consumed_archives; not_all_checkpoints",
                  ca_bce_scope="saved_metrics_only; packed_predictions_verify_accuracy_not_logits")
    outputs = {**tables, "audit_summary.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    for name, content in outputs.items():
        destination = (HERE / name).resolve()
        assert destination.parent == HERE and destination.name in outputs
        if args.write:
            destination.write_text(content, encoding="utf-8", newline="")
        else:
            assert destination.read_text(encoding="utf-8") == content, f"派生文件已变化: {name}"
    print(json.dumps(dict(status="PASS", mode="write" if args.write else "read_only",
                          source_files=len(SOURCES), recorded_hashes_verified=len(set(MANIFEST_CHECKS)),
                          copy_checks=copy_audit["forward_checks"], rule30_grid=ca_audit["smallest_grid_n_with_three_of_four_successes"],
                          bilingual=bilingual, torch_imported=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
