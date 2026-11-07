"""公开包检查或本地原始数组 CPU 复算；不导入训练器或 Torch。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sys
import time
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
REPO = None
PUBLIC_ONLY = False
XOR = "research/xor_face_agreement_20261104"
FOUR = "research/four_sample_shortcuts_20261104"
CNN = "research/wide_cnn_time_ensemble_20261101"
EXPERIMENTS = "research/neural_k_framework/experiments"
BACKGROUND = [
    f"{EXPERIMENTS}/experiment_18_consensus_symbolicity/README.md",
    f"{EXPERIMENTS}/experiment_18_consensus_symbolicity/RESULTS_AND_CONCLUSION.md",
    f"{EXPERIMENTS}/experiment_21_agreement_complexity_frontier/README.md",
    f"{EXPERIMENTS}/experiment_21_agreement_complexity_frontier/RESULTS_AND_CONCLUSION.md",
    f"{EXPERIMENTS}/experiment_29_dead_bit_loss_resolved/RESULTS_AND_CONCLUSION.md",
    f"{EXPERIMENTS}/experiment_36_peleg_g_and_c_validation/VALIDATION_EXTENSION_20261027.md",
    "research/notes/20261030_ait_correspondence_discussion_zh.md",
    "research/notes/20261030_function_mass_accuracy_bridge_zh.md",
]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
np = None
INPUTS = None

SOURCES: dict[str, dict] = {}


def initialize_numerics():
    global np, INPUTS
    try:
        import numpy as np
    except ImportError:
        if REPO is None:
            raise RuntimeError("此模式需要已有 NumPy；公开包检查请使用 --public-only") from None
        sys.path.insert(0, str(REPO / ".venv-gpu/Lib/site-packages"))
        import numpy as np
    INPUTS = bits(np.arange(16), [3, 2, 1, 0])


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(relative):
    if PUBLIC_ONLY:
        raise RuntimeError("--public-only 禁止读取包外来源")
    if REPO is None:
        raise RuntimeError("此模式需要原仓库布局和本地来源；仅有公开包时请使用 --public-only")
    relative = str(relative).replace("\\", "/")
    path = (REPO / relative).resolve()
    allowed = relative in BACKGROUND or relative == "research/AGENTS.md"
    allowed |= any(path.is_relative_to((REPO / prefix).resolve()) for prefix in (XOR, FOUR, CNN))
    if not allowed:
        raise ValueError(f"超出只读来源范围: {relative}")
    if relative not in SOURCES:
        SOURCES[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
    return path


def read_json(relative):
    return json.loads(source(relative).read_text(encoding="utf-8-sig"))


def read_csv(relative):
    with source(relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def arrays(relative):
    with np.load(source(relative), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def check_hash(relative, expected):
    assert digest(source(relative)) == expected, relative


def bits(ids, positions):
    return (np.asarray(ids, dtype=np.int64)[..., None] >> np.asarray(positions)) & 1


def mlp_forward(weights):
    w = weights.astype(np.float64)
    h = np.tanh((INPUTS @ w[:, :64].reshape(-1, 16, 4).transpose(0, 2, 1) + w[:, None, 64:80]) / 2)
    return (h @ w[:, 80:96, None]).squeeze(-1) / 4 + w[:, 96:] / 4


def agreement(predictions, axis=0):
    n = predictions.shape[axis]
    assert n >= 2
    ones = predictions.sum(axis=axis, dtype=np.int64)
    return (ones * (ones - 1) + (n - ones) * (n - ones - 1)) / (n * (n - 1))


def metrics(logits, train, labels, teacher=None):
    held = np.setdiff1d(np.arange(16), train)
    assert len(train) == 4 and len(held) == 12
    hard = logits >= 0
    assert np.all(hard[:, train] == labels)
    n = len(hard)
    ids = hard.astype(np.int64) @ (1 << np.arange(16))
    unique, counts = np.unique(ids, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    result = dict(n=n, heldout=held.tolist(), agreement=float(agreement(hard[:, held]).mean()),
                  collision=float(np.sum(counts * (counts - 1)) / (n * (n - 1))),
                  distinct=len(unique), mode=f"0x{unique[order[0]]:04X}",
                  mode_count=int(counts[order[0]]), ones=hard[:, held].sum(0).tolist(),
                  zero_logits=int(np.count_nonzero(logits == 0)))
    if teacher is not None:
        correct = hard[:, held] == bits(teacher, held)
        result.update(accuracy=float(correct.mean()), teacher_count=int(correct.all(1).sum()))
        close(correct.mean(0).mean(), correct.mean(1).mean())
    return result


def audit_xor():
    comparison = read_json(f"{XOR}/FACE_COMPARISON.json")
    catalogue = read_json(f"{XOR}/CATALOGUE.json")
    cache = arrays(f"{XOR}/CATALOGUE.npz")
    ranks = read_csv(f"{XOR}/FACE_WINDOW_RANKS.csv")
    result = {}
    primary = {}
    first_init = None
    raw = {}
    for face, folder in [("11", XOR), ("00", f"{XOR}/zero_face")]:
        manifest = read_json(f"{folder}/TRAIN_MANIFEST.json")
        check_hash(f"{folder}/TRAINING.npz", manifest["predictions_sha256"])
        check_hash(f"{folder}/PROTOCOL_ZH.md", manifest["protocol_sha256"])
        trainer = "train.py" if face == "11" else "run.py"
        check_hash(f"{folder}/{trainer}", manifest["source_sha256"])
        data = arrays(f"{folder}/TRAINING.npz")
        raw[face] = data
        if first_init is not None:
            np.testing.assert_array_equal(data["initial"], first_init)
        first_init = data["initial"]
        assert first_init.shape == (2048, 97)
        train = np.array(manifest["train"])
        expected = np.arange(12, 16) if face == "11" else np.arange(4)
        np.testing.assert_array_equal(train, expected)
        index = int(np.flatnonzero(data["thresholds"] == .004)[0])
        assert np.all(data["hit_steps"][index] >= 0)
        for task, (name, teacher) in enumerate([("xor", 0x6666), ("copy", 0xAAAA)]):
            key = f"{face}_{name}"
            labels = bits(teacher, train)
            logits = data["hit_logits"][index, task]
            bce = np.logaddexp(0., -(2 * labels - 1) * logits[:, train].astype(float)).mean(1)
            close(bce, data["hit_bce"][index, task])
            assert bce.max() <= .004
            hit = metrics(logits, train, labels, teacher)
            hit["steps_min_median_max"] = np.quantile(data["hit_steps"][index, task], [0, .5, 1]).tolist()
            reference = comparison["cases"][key]
            for a, b in [("agreement", "agreement"), ("collision", "collision"), ("accuracy", "teacher_accuracy"),
                         ("teacher_count", "teacher_count"), ("distinct", "distinct")]:
                close(hit[a], reference["hits"]["0.004"][b])
            final_logits = data["snapshot_logits"][-1, task]
            independent = mlp_forward(data["final"][task])
            error = float(np.abs(independent - final_logits).max())
            assert error < 1e-5
            np.testing.assert_array_equal(independent >= 0, final_logits >= 0)
            final = metrics(final_logits, train, labels, teacher)
            close(final["agreement"], reference["final"]["agreement"])
            assert final["teacher_count"] == reference["final"]["teacher_count"]
            delta = data["final"][task].astype(float) - data["initial"].astype(float)
            fixed = delta[:, :64].reshape(-1, 16, 4)[:, :, :2]
            bias = delta[:, 64:80, None]
            changes = dict(fixed_cols_max=float(abs(fixed).max()), fixed_cols_RMS=float(np.mean(fixed ** 2) ** .5),
                           bias_RMS=float(np.mean(bias ** 2) ** .5),
                           fixed_col_change_minus_bias_RMS=float(np.mean((fixed - bias) ** 2) ** .5))
            for k, v in changes.items():
                close(v, reference["first_layer_change"][k])
            if face == "00":
                assert changes["fixed_cols_max"] == 0
            proxy = reference["full_cost_proxy"]
            full_ids = (logits >= 0).astype(np.int64) @ (1 << np.arange(16))
            proxy_mode_training_count = int(np.count_nonzero(full_ids == int(proxy["top"][0]["id"], 16)))
            point_mae = float(np.abs(np.array(hit["ones"]) / 2048 - proxy["one_prob"]).mean())
            close(point_mae, reference["full_point_MAE"])
            # 11 面缓存含完整候选权重；00 面只在授权来源中保留了汇总，不能伪称重新积分。
            if face == "11":
                ids = cache[f"{name}_ids"]
                k = cache[f"{name}_measured_K_0.004"]
                assert len(ids) == 4096 and np.isfinite(k).all()
                w = np.exp2(k.min() - k)
                w /= w.sum()
                close(w, cache[f"{name}_0.004_weights"])
                close(w @ bits(ids, hit["heldout"]), proxy["one_prob"])
                close(w[ids == teacher][0], proxy["teacher_mass"])
                empirical_ids = (logits >= 0).astype(np.int64) @ (1 << np.arange(16))
                hist = np.bincount(empirical_ids, minlength=65536)[ids] / 2048
                close(np.abs(w - hist).sum() / 2, reference["full_TV"])
            result[key] = dict(train_inputs=[format(int(i), "04b") for i in train], labels=labels.tolist(),
                               teacher=f"0x{teacher:04X}", primary=hit, final=final,
                               first_layer_change=changes, final_numpy_logit_max_error=error,
                               proxy={k: proxy[k] for k in ("min_K", "agreement", "teacher_mass", "all_one_mass", "one_prob")},
                               proxy_mode=proxy["top"][0]["id"], point_frequency_mae=point_mae,
                               proxy_mode_training_count=proxy_mode_training_count,
                               proxy_TV_reported=reference["full_TV"],
                               anchored_proxy_default1_mass=reference["slope_anchored_proxy"]["all_one_mass"])
            correct = (logits[:, hit["heldout"]] >= 0) == bits(teacher, hit["heldout"])
            primary[key] = (correct.mean(1), correct.all(1))
    for name in ("xor", "copy"):
        for metric in range(2):
            difference = primary[f"00_{name}"][metric].astype(float) - primary[f"11_{name}"][metric].astype(float)
            close(difference.mean(), comparison["paired"][name]["accuracy_and_exact_changes"][metric])
    return dict(cases=result, paired=comparison["paired"],
                primary_slope_rows=[r for r in ranks if r["case"] == "xor" and float(r["high"]) == .01 and float(r["low"]) == .0063],
                reported_archive_hashes=[dict(filename=PureWindowsPath(row["path"]).name,
                                             bytes=row["bytes"], sha256=row["sha256"])
                                         for row in catalogue["sources"]]), raw


def independent_combinatorics(rows):
    ids = np.arange(65536)
    table = bits(ids, np.arange(16))
    support = np.zeros(65536, dtype=np.int8)
    for bit in range(4):
        support += np.any(table != table[:, np.arange(16) ^ (1 << bit)], axis=1)
    simple = ids[support <= 2]
    xor_ids = np.array([i for i in simple if support[i] == 2 and int(table[i].sum()) == 8])
    assert len(simple) == 70 and len(xor_ids) == 12
    families = {"constant": ids[support == 0], "literal_or_constant": ids[support <= 1],
                "nonxor_cost1_gate_or_simpler": simple[~np.isin(simple, xor_ids)],
                "two_or_fewer_bits": simple, "other_two_bit_than_xor_teacher": simple[simple != 0x6666]}
    support_families = [ids[support <= size] for size in range(4)]
    keyed = {(int(r["mask"]), int(r["positive"])): r for r in rows}
    assert len(keyed) == 29120
    rectangle = {2: 0, 3: 0, 4: 0}
    for points in itertools.combinations(range(16), 4):
        mask = sum(1 << i for i in points)
        x = INPUTS[list(points)]
        for labels in itertools.product((0, 1), repeat=4):
            positive = sum(y << i for i, y in zip(points, labels))
            row = keyed[(mask, positive)]
            for key, family in families.items():
                assert bool(np.any((family & mask) == positive)) == (row[key] == "True")
            hits = int(np.sum((simple & mask) == positive))
            assert hits == int(row["cheap_count"])
            minimum_support = next(size for size, family in enumerate(support_families)
                                   if np.any((family & mask) == positive))
            assert minimum_support == int(row["min_support"])
            labels = np.array(labels)
            nonsep = bool(labels.sum() == 2 and np.all(x[labels == 0].sum(0) == x[labels == 1].sum(0)))
            assert nonsep == (row["linear_separable"] == "False")
            if nonsep:
                rectangle[int(np.sum(np.ptp(x, axis=0) > 0))] += 1
    assert rectangle == {2: 48, 3: 96, 4: 56}
    return rectangle


def audit_four(raw_faces):
    census = read_json(f"{FOUR}/CENSUS.json")
    training = read_json(f"{FOUR}/TRAINING.json")
    old_audit = read_json(f"{FOUR}/AUDIT.json")
    check_hash(f"{FOUR}/CENSUS.json", training["census_sha256"])
    check_hash(f"{FOUR}/PROTOCOL_ZH.md", training["protocol_sha256"])
    check_hash(f"{FOUR}/optimizer_probe.py", training["training_source_sha256"])
    rows = read_csv(f"{FOUR}/ALL_N4.csv")
    rectangle = independent_combinatorics(rows)
    for name, selector in [("all", lambda r: True), ("balanced", lambda r: int(r["positive_n"]) == 2),
                           ("xor_teacher", lambda r: (0x6666 & int(r["mask"])) == int(r["positive"]))]:
        selected = [r for r in rows if selector(r)]
        reference = census["coverage"][name]
        assert len(selected) == reference["n"]
        for key, value in reference["shortcut_counts"].items():
            assert sum(r[key] == "True" for r in selected) == value
    k = np.array([float(r["min_K"]) for r in rows])
    close(np.quantile(k, [.5, .9, .95, .99, 1]), [census["coverage"]["all"]["min_K_quantiles"][s]
                                               for s in ("median", "p90", "p95", "p99", "max")])
    for row in rows:
        close(-math.log2(int(row["prior_hits"]) / census["prior_samples"]), float(row["hard_K"]))
    separable = np.array([r["linear_separable"] == "True" for r in rows])
    for key, expected in old_audit["gap_checks"].items():
        values = np.array([float(r[key]) for r in rows])
        close(values[separable].max(), expected["largest_separable"])
        close(values[~separable].min(), expected["smallest_nonseparable"])
        close(values.max(), expected["maximum"])
        assert int(np.count_nonzero(np.isclose(values, values.max(), rtol=0, atol=1e-8))) == expected["max_ties"]
    details = []
    max_error = 0.
    for row in training["results"]:
        key = f"{row['mask']:04X}_{row['positive']:04X}"
        relative = f"{FOUR}/matched/{key}.npz"
        check_hash(relative, row["raw_sha256"])
        data = arrays(relative)
        points, labels, logits = data["points"], data["labels"], data["logits"]
        assert len(logits) == 512 and np.all(data["hit_steps"] >= 0)
        assert int(data["hit_steps"].max()) <= 30000
        assert sum(1 << int(i) for i in points) == row["mask"]
        assert sum(int(y) << int(i) for i, y in zip(points, labels)) == row["positive"]
        m = metrics(logits, points, labels)
        close(m["agreement"], row["agreement"])
        close(m["collision"], row["full_collision"])
        assert m["distinct"] == row["distinct"]
        loss = np.logaddexp(0., -(2 * labels - 1) * logits[:, points].astype(float)).mean(1)
        assert loss.max() <= .004
        independent = mlp_forward(data["weights"])
        error = float(np.abs(independent - logits).max())
        max_error = max(max_error, error)
        assert error < 2e-6
        np.testing.assert_array_equal(independent >= 0, logits >= 0)
        for face, tag in [("11", "prior_11_control"), ("00", "prior_00_control")]:
            if tag in row["tags"]:
                old = raw_faces[face]
                ix = int(np.flatnonzero(old["thresholds"] == .004)[0])
                np.testing.assert_array_equal(old["hit_steps"][ix, 0, :512], data["hit_steps"])
                np.testing.assert_array_equal(old["hit_logits"][ix, 0, :512] >= 0, logits >= 0)
        fixed = {name: int(INPUTS[points[0], j]) for j, name in enumerate("abcd")
                 if np.ptp(INPUTS[points, j]) == 0}
        details.append(dict(key=key, points=row["points"], fixed_bits=fixed, min_K=row["min_K"],
                            hard_K=row["hard_K"], tags=row["tags"], orbit_size=row["orbit_size"],
                            last_step=int(data["hit_steps"].max()), **m))
    top = [r for r in details if any(t.startswith("top_orbit_") for t in r["tags"])]
    random = [r for r in details if any(t.startswith("uniform_") for t in r["tags"])]
    assert len(details) == 30 and len(top) == 12 and len(random) == 16
    weighted = sum(r["agreement"] * r["orbit_size"] for r in top) / sum(r["orbit_size"] for r in top)
    close(weighted, old_audit["top_orbit_weighted_agreement"])
    close(np.mean([r["agreement"] for r in random]), old_audit["random16_agreement_mean"])
    # 对称证书逐点验证，不将 50:50 边际误说成整表均匀。
    for record in old_audit["details"]:
        row = next(r for r in details if r["key"] == record["key"])
        pairs = [t.split(":") for t in row["points"].split()]
        labels = {int(x, 2): int(y) for x, y in pairs}
        certified = set()
        for permutation in itertools.permutations(range(4)):
            transformed = INPUTS[:, permutation] @ np.array([8, 4, 2, 1])
            if all(int(transformed[x]) in labels and labels[int(transformed[x])] == 1 - y for x, y in labels.items()):
                certified.update(i for i in row["heldout"] if transformed[i] == i)
        assert sorted(certified) == record["forced_inputs"]
    return dict(count=29120, orbit_count=census["orbit_count"], coverage=census["coverage"],
                prior_draws=census["prior_samples"], gap_checks=old_audit["gap_checks"],
                rectangle_labelings=rectangle, training=details, top_weighted_agreement=weighted,
                random16_agreement_mean=float(np.mean([r["agreement"] for r in random])),
                max_numpy_logit_error=max_error,
                symmetry_forced_counts={r["key"]: r["forced_count"] for r in old_audit["details"] if r["forced_count"]})


def audit_cnn():
    reference = read_json(f"{CNN}/results_extended/ANALYSIS.json")
    result = []
    for ref in reference:
        case = ref["case"]
        folder = f"{CNN}/results_extended/{case}"
        protocol = read_json(f"{folder}/PROTOCOL.json")
        data = arrays(f"{folder}/trajectory.npz")
        parent = arrays(f"{CNN}/results/{case}/trajectory.npz")
        labels = arrays(f"{folder}/probe_labels.npz")
        for key, expected in [("labels", "probe_y"), ("probe_ids", "probe_ids")]:
            value = np.ascontiguousarray(labels[key])
            hashed = hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()
            assert hashed == protocol["data"][expected]
        for k, v in parent.items():
            np.testing.assert_array_equal(data[k][:len(v)], v)
        step, gap = data["step"], data["gap"]
        assert gap.shape == (129, 16, 2048) and step[-1] == 32768
        assert np.all(np.diff(step) == 256) and len(np.unique(labels["probe_ids"])) == 2048
        late = step >= 24576
        p = gap[late] > 0
        y = labels["labels"]
        assert p.shape == (33, 16, 2048)
        acc = (p == y).mean(2)
        close((p == y).mean(1).mean(1), acc.mean(1))
        loss = data["loss"][late]
        flip = (p[1:] != p[:-1]).sum(0)
        seed_frequency = p.mean(0)
        within = float((seed_frequency * (1 - seed_frequency)).mean())
        between = float(seed_frequency.var(0).mean())
        pooled = p.mean((0, 1))
        total = float((pooled * (1 - pooled)).mean())
        close(within + between, total)
        values = dict(final_accuracy_mean=float(acc[-1].mean()), final_accuracy_seed_sd=float(acc[-1].std(ddof=1)),
                      mean_within_seed_accuracy_sd=float(acc.std(0, ddof=1).mean()),
                      time_agreement_mean=float(agreement(p, 0).mean()), seed_agreement_mean=float(agreement(p, 1).mean()),
                      temporal_ever_flip_fraction=float((flip > 0).mean()),
                      temporal_repeated_flip_fraction=float((flip >= 2).mean()),
                      late_loss_first_median=float(np.median(loss[0])), final_loss_median=float(np.median(loss[-1])),
                      late_loss_final_over_first_median=float(np.median(loss[-1] / loss[0])),
                      final_loss_min_max=[float(loss[-1].min()), float(loss[-1].max())])
        for k, value in values.items():
            close(value, ref[k])
        close(between / total, ref["variance_decomposition"]["between_fraction"])
        for k, expected in protocol["source"].items():
            if k.replace("\\", "/").startswith(CNN):
                check_hash(k, expected)
        crossings = []
        for expected in ref["crossings"]:
            found, hit_loss, hit_acc = [], [], []
            for seed in range(16):
                ids = np.flatnonzero((data["loss"][:, seed] <= expected["epsilon"]) & data["hard_fit"][:, seed])
                if len(ids):
                    i = ids[0]
                    found.append(int(step[i]))
                    hit_loss.append(float(data["loss"][i, seed]))
                    hit_acc.append(float(((gap[i, seed] > 0) == y).mean()))
            assert found == expected["steps"] and len(found) == expected["reached"]
            close(np.mean(hit_loss), expected["actual_loss_mean"])
            close(np.mean(hit_acc), expected["accuracy"])
            crossings.append({k: expected[k] for k in ("epsilon", "reached", "accuracy", "actual_loss_mean")})
        # 显式枚举小样本模型对，独立验证计数公式。
        a, b = np.triu_indices(16, 1)
        close(np.mean(p[-1, a] == p[-1, b]), agreement(p[-1], 0).mean())
        result.append(dict(case=case, seeds=16, probe_count=2048, snapshots=129, late_snapshots=33,
                           final_below_1e5=int((loss[-1] <= 1e-5).sum()),
                           late_ever_changed_images_per_seed=float((flip > 0).sum(1).mean()),
                           between_seed_variance_fraction=between / total, coarse_crossings=crossings,
                           probe_zero_gaps=int(np.count_nonzero(gap == 0)),
                           data_hashes_reported=protocol["data"], prefix_8192_exact=True, **values))
    return result


def render(results):
    # 字体缓存仅写入当前包目录，完成后清理本次生成的缓存文件。
    existing_caches = set(HERE.glob("fontlist-v*.json"))
    with nullcontext(str(HERE)) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "savefig.facecolor": "white"})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        ax = axes[0]
        for tag, color, label, marker in [("top_orbit_", "#176f78", "Top 12 orbit representatives", "o"),
                                         ("uniform_", "#a64b61", "16 fixed random controls", "x")]:
            rows = [r for r in results["four_point"]["training"] if any(t.startswith(tag) for t in r["tags"])]
            ax.scatter([r["min_K"] for r in rows], [100 * r["agreement"] for r in rows], color=color, marker=marker, label=label)
        ax.set(xlabel="Minimum full-target cost (bits; proxy)", ylabel="Unseen pairwise agreement (%)", ylim=(45, 104),
               title="A. Four-point constraints, 512 paired starts")
        ax.legend(loc="lower left", fontsize=8)
        ax = axes[1]
        face = results["xor_faces"]["cases"]["11_xor"]
        x = np.arange(12)
        ax.plot(x, np.array(face["primary"]["ones"]) / 2048, "o-", color="#176f78", label="Adam: 2048 starts")
        ax.plot(x, face["proxy"]["one_prob"], "s--", color="#a64b61", label="Full-target volume proxy")
        ax.set(xticks=x, xticklabels=[format(i, "04b") for i in range(12)], xlabel="Unseen input (11 face training)",
               ylabel="Probability of predicting 1", ylim=(-.05, 1.05), title="B. Similar agreement, different answers")
        ax.tick_params(axis="x", rotation=60)
        ax.legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.15)
        fig.savefig(HERE / "small_data_consensus.png", dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        rows = results["wide_cnn"]
        x = np.arange(2)
        axes[0].bar(x - .18, [r["time_agreement_mean"] * 100 for r in rows], .36, color="#176f78", label="Within-seed time")
        axes[0].bar(x + .18, [r["seed_agreement_mean"] * 100 for r in rows], .36, color="#a64b61", label="Across seeds")
        for i, r in enumerate(rows):
            axes[0].text(i - .18, r["time_agreement_mean"] * 100 + 1, f"{100*r['time_agreement_mean']:.3f}", ha="center", fontsize=9)
            axes[0].text(i + .18, r["seed_agreement_mean"] * 100 + 1, f"{100*r['seed_agreement_mean']:.3f}", ha="center", fontsize=9)
        axes[0].set(xticks=x, xticklabels=["partition0", "partition2"], ylim=(0, 110), ylabel="Pairwise agreement (%)",
                    title="A. Late checkpoints are not new seeds")
        axes[0].legend(loc="lower right", fontsize=9)
        for r, color in zip(rows, ("#176f78", "#a64b61")):
            axes[1].plot([24576, 32768], [r["late_loss_first_median"], r["final_loss_median"]], "o-", color=color, label=r["case"].split("_")[0])
        axes[1].set(yscale="log", xticks=[24576, 32768], xlabel="SGD update", ylabel="Median training BCE",
                    title="B. Stable hard predictions, decreasing loss")
        axes[1].legend()
        for ax in axes:
            ax.grid(axis="y", alpha=.15)
        fig.savefig(HERE / "wide_cnn_ensembles.png", dpi=170)
        plt.close(fig)
    for path in set(HERE.glob("fontlist-v*.json")) - existing_caches:
        path.unlink()


def bilingual_check():
    docs = [(HERE / f"README_{language}.md").read_text(encoding="utf-8") for language in ("zh", "en")]
    markers = [re.findall(r"<!-- ([PFT]\d+) -->", doc) for doc in docs]
    assert markers[0] == markers[1] and len(markers[0]) == len(set(markers[0]))
    formulas = [re.findall(r"\$\$(.*?)\$\$", doc, re.S) for doc in docs]
    assert formulas[0] == formulas[1]
    for doc in docs:
        assert not any(ord(c) < 32 and c not in "\n\r\t" for c in doc)
    tables = [[line for line in doc.splitlines() if line.startswith("|")] for doc in docs]
    assert len(tables[0]) == len(tables[1])
    for left, right in zip(*tables):
        # 数字、函数 ID 与输入位串必须逐行相同，表头文字可翻译。
        token = r"0x[0-9A-F]+|(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
        assert re.findall(token, left) == re.findall(token, right), (left, right)
    image_sets = []
    for doc in docs:
        assert not re.search(r"dirty|main\s+agent|主\s*agent|待整合|唯一写入范围|\.plot-cache", doc, re.I)
        for target in re.findall(r"\]\(([^)]+)\)", doc):
            parsed = urlsplit(target)
            if parsed.scheme in ("http", "https") or target.startswith("#"):
                continue
            assert not parsed.scheme and not parsed.netloc, target
            path = unquote(parsed.path)
            assert not PureWindowsPath(path).drive and not PurePosixPath(path).is_absolute(), target
            assert ".." not in PurePosixPath(path).parts, target
            assert (HERE / path).resolve().is_relative_to(HERE), target
            assert path in ("VALIDATION.json", "PUBLIC_VALIDATION.json") or (HERE / path).is_file(), target
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc)
        assert sorted(images) == ["small_data_consensus.png", "wide_cnn_ensembles.png"]
        for name in images:
            assert (HERE / name).is_file()
        image_sets.append(images)
    assert image_sets[0] == image_sets[1]
    return dict(aligned_blocks=len(markers[0]), identical_formula_blocks=len(formulas[0]),
                matched_table_rows=len(tables[0]), package_local_images=2, external_source_links=0)


def table_value_check(results):
    doc = (HERE / "README_zh.md").read_text(encoding="utf-8")

    def table(name):
        block = doc.split(f"<!-- {name} -->", 1)[1].split("\n\n", 1)[0]
        return [[cell.strip() for cell in line.strip("|").split("|")]
                for line in block.splitlines() if line.startswith("|")][2:]

    for name, count in {"T01": 2, "T02": 4, "T03": 2, "T04": 5, "T05": 7, "T07": 11}.items():
        assert len(table(name)) == count
    for line, face in zip(table("T01"), ("11", "00")):
        r = results["xor_faces"]["cases"][face + "_xor"]
        c = results["xor_faces"]["cases"][face + "_copy"]
        assert line[1:4] == [", ".join(r["train_inputs"]), ", ".join(map(str, r["labels"])), ", ".join(map(str, c["labels"]))]
        assert line[-1] == str(len(r["primary"]["heldout"]))
    for line, key in zip(table("T02"), ("11_xor", "00_xor", "11_copy", "00_copy")):
        r = results["xor_faces"]["cases"][key]["primary"]
        assert line[1:] == [f"{100*r['agreement']:.2f}", f"{100*r['accuracy']:.2f}", f"{100*r['collision']:.2f}",
                            f"{r['teacher_count']}/{r['n']}", str(r["distinct"])]
    for line, key in zip(table("T03"), ("11_xor", "00_xor")):
        r = results["xor_faces"]["cases"][key]
        p = r["proxy"]
        assert line[1:] == [f"{p['min_K']:.2f}", r["proxy_mode"], f"{100*p['all_one_mass']:.2f}",
                            f"{p['teacher_mass']:.2e}", f"{r['proxy_mode_training_count']}/2048", f"{100*r['point_frequency_mae']:.2f}"]
    family = ("constant", "literal_or_constant", "nonxor_cost1_gate_or_simpler", "two_or_fewer_bits", "linear_separable")
    for line, key in zip(table("T04"), family):
        n = results["four_point"]["coverage"]["all"]["shortcut_counts"][key]
        assert line[1:] == [str(n), f"{100*n/29120:.2f}"]
    for line in table("T05"):
        r = next(row for row in results["four_point"]["training"] if row["points"] == line[0])
        fixed = ",".join(f"{key}={v}" for key, v in r["fixed_bits"].items()) or "无"
        assert line[1:] == [fixed, f"{r['min_K']:.2f}", f"{100*r['agreement']:.2f}", f"{100*r['collision']:.2f}"]
    specs = [("final_accuracy_mean", 100, ".3f"), ("final_accuracy_seed_sd", 100, ".3f"),
             ("mean_within_seed_accuracy_sd", 100, ".4f"), ("time_agreement_mean", 100, ".3f"),
             ("seed_agreement_mean", 100, ".3f"), ("temporal_ever_flip_fraction", 100, ".4f"),
             ("temporal_repeated_flip_fraction", 100, ".0f"), ("late_loss_first_median", 1, ".3e"),
             ("final_loss_median", 1, ".3e"), ("late_loss_final_over_first_median", 1, ".4f"),
             ("between_seed_variance_fraction", 100, ".2f")]
    for line, (key, scale, fmt) in zip(table("T07"), specs):
        expected = [format(r[key] * scale, fmt).replace("e-0", "e-") for r in results["wide_cnn"]]
        assert line[1:] == expected, (line, expected)
    return {"results_verified_data_tables": 6}


def public_data_check(results):
    """只核验精简数据能支持的代数关系，不从边际票数推断联合碰撞。"""
    def equal(actual, expected):
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), (actual, expected)

    profiles = 0

    def votes(row, train, teacher=None):
        nonlocal profiles
        n, held, ones = row["n"], row["heldout"], row["ones"]
        assert n >= 2 and len(held) == len(ones) == 12
        assert sorted(train + held) == list(range(16))
        assert all(isinstance(k, int) and 0 <= k <= n for k in ones)
        estimate = sum(k*(k-1)+(n-k)*(n-k-1) for k in ones) / (12*n*(n-1))
        equal(estimate, row["agreement"])
        assert 0 <= row["collision"] <= row["agreement"] <= 1
        assert 1 <= row["distinct"] <= min(n, 4096)
        assert 1 <= row["mode_count"] <= n
        if teacher is not None:
            correct = sum(k if (teacher >> i) & 1 else n-k for i, k in zip(held, ones))
            equal(correct / (12*n), row["accuracy"])
            assert 0 <= row["teacher_count"] <= n
        profiles += 1

    cases = results["xor_faces"]["cases"]
    assert set(cases) == {"11_xor", "00_xor", "11_copy", "00_copy"}
    for key, row in cases.items():
        train = [int(x, 2) for x in row["train_inputs"]]
        assert train == (list(range(12, 16)) if key.startswith("11") else list(range(4)))
        teacher = int(row["teacher"], 16)
        assert row["labels"] == [(teacher >> i) & 1 for i in train]
        for stage in ("primary", "final"):
            assert row[stage]["n"] == 2048
            votes(row[stage], train, teacher)
        frequency = [k/2048 for k in row["primary"]["ones"]]
        proxy = row["proxy"]["one_prob"]
        equal(sum(abs(a-b) for a, b in zip(frequency, proxy))/12, row["point_frequency_mae"])
        equal(sum(p*p+(1-p)**2 for p in proxy)/12, row["proxy"]["agreement"])
    for name in ("xor", "copy"):
        left, right = cases["00_"+name]["primary"], cases["11_"+name]["primary"]
        paired = results["xor_faces"]["paired"][name]["accuracy_and_exact_changes"]
        equal(left["accuracy"]-right["accuracy"], paired[0])
        equal((left["teacher_count"]-right["teacher_count"])/2048, paired[1])
    census = results["four_point"]
    assert census["count"] == math.comb(16, 4)*16 == 29120
    for size, n in census["rectangle_labelings"].items():
        d = int(size)
        assert n == math.comb(4, d)*2**(4-d)*math.comb(2**(d-1), 2)*2
    for group in census["coverage"].values():
        assert sum(group["fewest_bits_hist"].values()) == group["n"]
        assert all(0 <= count <= group["n"] for count in group["shortcut_counts"].values())
    panel = census["training"]
    assert len(panel) == len({row["key"] for row in panel}) == 30
    for row in panel:
        pairs = [pair.split(":") for pair in row["points"].split()]
        points = [int(x, 2) for x, _ in pairs]
        mask = sum(1 << x for x in points)
        positive = sum(int(y) << int(x, 2) for x, y in pairs)
        assert row["key"] == f"{mask:04X}_{positive:04X}"
        assert row["n"] == 512 and row["last_step"] <= 30000
        votes(row, points)
        fixed = {name: (points[0] >> (3-j)) & 1 for j, name in enumerate("abcd")
                 if len({(x >> (3-j)) & 1 for x in points}) == 1}
        assert fixed == row["fixed_bits"]
    top = [r for r in panel if any(t.startswith("top_orbit_") for t in r["tags"])]
    random = [r for r in panel if any(t.startswith("uniform_") for t in r["tags"])]
    assert len(top) == 12 and len(random) == 16
    equal(sum(r["agreement"]*r["orbit_size"] for r in top)/sum(r["orbit_size"] for r in top), census["top_weighted_agreement"])
    equal(sum(r["agreement"] for r in random)/16, census["random16_agreement_mean"])
    cnn = results["wide_cnn"]
    assert len(cnn) == 2
    protocol = results["protocols"]["wide_cnn"]
    assert protocol["steps"] // protocol["probe_stride"] + 1 == 129
    assert (protocol["late_window"][1]-protocol["late_window"][0]) // protocol["probe_stride"] + 1 == 33
    for row in cnn:
        assert row["seeds"] == 16 and row["probe_count"] == 2048
        assert row["snapshots"] == 129 and row["late_snapshots"] == 33
        for key in ("time_agreement_mean", "seed_agreement_mean", "final_accuracy_mean", "between_seed_variance_fraction"):
            assert 0 <= row[key] <= 1
        equal(row["temporal_ever_flip_fraction"]*2048, row["late_ever_changed_images_per_seed"])
        assert row["final_loss_min_max"][0] <= row["final_loss_median"] <= row["final_loss_min_max"][1]
        assert 0 <= row["final_below_1e5"] <= 16
        for crossing in row["coarse_crossings"]:
            assert 0 <= crossing["reached"] <= 16
            assert crossing["actual_loss_mean"] <= crossing["epsilon"]
    return dict(vote_profiles_recalculated=profiles, cnn_summary_cases_checked=len(cnn),
                full_collision_recalculated=False, cnn_raw_time_seed_statistics_recalculated=False)


PUBLIC_FILES = ("README_zh.md", "README_en.md", "results.json", "SOURCE_SHA256.json", "SOURCE_SHA256SUMS.txt",
                "recompute.py", "VALIDATION.json", "small_data_consensus.png", "wide_cnn_ensembles.png")


def public_manifest_check(results):
    manifest = json.loads((HERE / "SOURCE_SHA256.json").read_text(encoding="utf-8"))
    assert manifest["path_base"] == "repository_root"
    paths = set()
    for row in manifest["files"]:
        path = row["path"]
        assert path not in paths and path.startswith("research/") and ".." not in PurePosixPath(path).parts
        assert not PureWindowsPath(path).drive
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) and row["bytes"] >= 0
        paths.add(path)
    expected = "".join(f"{r['sha256']}  {r['path']}\n" for r in manifest["files"])
    assert (HERE / "SOURCE_SHA256SUMS.txt").read_text(encoding="utf-8") == expected
    archives = manifest["upstream_archives_reported_not_rehashed"]
    assert archives == results["xor_faces"]["reported_archive_hashes"] and len(archives) == 3
    for row in archives:
        assert set(row) == {"filename", "bytes", "sha256"}
        assert PureWindowsPath(row["filename"]).name == row["filename"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    for name in ("small_data_consensus.png", "wide_cnn_ensembles.png"):
        with (HERE / name).open("rb") as stream:
            header = stream.read(24)
        assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
        assert int.from_bytes(header[16:20], "big") > 0 and int.from_bytes(header[20:24], "big") > 0
    return dict(source_files_listed=len(paths), source_files_read=0, source_hashes_revalidated=False,
                source_manifest_format_checked=True, png_files_checked=2)


def check_public_package(started):
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    report = bilingual_check() | table_value_check(results) | public_data_check(results) | public_manifest_check(results)
    report.update(status="passed", mode="public-only", python=platform.python_version(),
                  elapsed_seconds=time.perf_counter()-started, raw_recomputation_in_this_run=False,
                  numpy_imported="numpy" in sys.modules, matplotlib_imported="matplotlib" in sys.modules,
                  torch_imported="torch" in sys.modules, new_training=False, new_smc=False,
                  package_sha256={name: digest(HERE / name) for name in PUBLIC_FILES})
    assert not SOURCES and not any(report[key] for key in ("numpy_imported", "matplotlib_imported", "torch_imported"))
    write_json("PUBLIC_VALIDATION.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def verify_sources(manifest):
    for row in manifest["files"]:
        path = source(row["path"])
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], row["path"]


def main():
    global REPO, PUBLIC_ONLY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["recompute", "check", "render"], default="check", nargs="?")
    parser.add_argument("--public-only", action="store_true", help="仅用标准库检查公开包，不访问本地原始来源")
    args = parser.parse_args()
    started = time.perf_counter()
    PUBLIC_ONLY = args.public_only
    if PUBLIC_ONLY:
        if args.mode != "check":
            parser.error("--public-only 不能与 recompute 或 render 同时使用")
        check_public_package(started)
        return
    REPO = next((parent for parent in HERE.parents if (parent / XOR).is_dir()), None)
    if args.mode in ("recompute", "render"):
        initialize_numerics()
    if args.mode == "render":
        render(json.loads((HERE / "results.json").read_text(encoding="utf-8")))
        return
    frozen = HERE / "SOURCE_SHA256.json"
    if frozen.exists():
        verify_sources(json.loads(frozen.read_text(encoding="utf-8")))
    if args.mode == "recompute":
        for relative in BACKGROUND + ["research/AGENTS.md"]:
            source(relative)
        for directory in (XOR, FOUR, CNN):
            for extension in ("*.py", "*.md"):
                for path in (REPO / directory).glob(extension):
                    source(path.relative_to(REPO).as_posix())
        for relative in [f"{XOR}/FACE_VOTES.csv", f"{FOUR}/TRAINING_PANEL.csv", f"{CNN}/PRECHECK.json",
                         f"{CNN}/EXECUTION_PLAN.json", f"{CNN}/results_extended/VALIDATION.json",
                         f"{CNN}/results_extended/RESUME_AND_FORWARD_AUDIT.json"]:
            source(relative)
        xor, raw = audit_xor()
        four = audit_four(raw)
        cnn = audit_cnn()
        result = dict(schema_version=1, baseline_commit="3834fb3f6555c7c58218462502635b16b9b377f9",
                      xor_faces=xor, four_point=four, wide_cnn=cnn,
                      protocols=dict(
                          boolean=dict(encoding="0/1, abcd, integer order 0..15", architecture=[4, 16, 1],
                                       activation="tanh", raw_parameters=97, raw_prior="independent N(0,1)",
                                       hidden_scale_divisor=2, output_scale_divisor=4, train_points=4, unseen_points=12,
                                       face_initializations=2048, panel_initializations=512, panel_conditions=30,
                                       streams=[2026110401, 2026110402, 2026110403, 2026110404],
                                       optimizer="full-batch Adam", learning_rate=.01, weight_decay=0,
                                       max_steps=30000, check_stride=10, primary_bce=.004,
                                       historical_dtype="float32", historical_tf32=False),
                          wide_cnn=dict(parameter_slots_including_fixed_bias=693378, bias="fixed zero",
                                        convolution_channels=[24, 64], fully_connected=[1024, 480, 336, 2],
                                        activation="ReLU", dropout=False, train_points=32, probe_points=2048,
                                        initializations_per_condition=16, initialization="first 16 of historical U[-0.2,0.2] stream",
                                        optimizer="SGD", learning_rate=.001, batch_size=2, momentum=0, weight_decay=0,
                                        steps=32768, epochs=2048, loss_stride=16, probe_stride=256,
                                        checkpoint_stride=4096, late_window=[24576, 32768], late_checkpoints=33,
                                        historical_dtype="float32", historical_tf32=False)),
                      audit_scope=dict(cpu_only=True, torch_imported="torch" in sys.modules,
                                       new_training=False, new_smc=False, upstream_archives_reopened=False,
                                       cnn_weight_forward_repeated=False, raw_probe_statistics_recomputed=True))
        assert not result["audit_scope"]["torch_imported"]
        manifest = dict(schema_version=1, path_base="repository_root", files=sorted(SOURCES.values(), key=lambda x: x["path"]),
                        upstream_archives_reported_not_rehashed=xor["reported_archive_hashes"])
        verify_sources(manifest)
        write_json("results.json", result)
        write_json("SOURCE_SHA256.json", manifest)
        (HERE / "SOURCE_SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in manifest["files"]), encoding="utf-8")
        render(result)
    else:
        assert frozen.exists(), "请先运行 recompute"
    qa = bilingual_check()
    qa.update(table_value_check(json.loads((HERE / "results.json").read_text(encoding="utf-8"))))
    qa.update(status="passed", source_files=len(SOURCES), python=platform.python_version(), numpy=np.__version__ if np is not None else None,
              cpu_only=True, torch_imported="torch" in sys.modules, elapsed_seconds=time.perf_counter() - started,
              mode=args.mode, raw_recomputation_in_this_run=args.mode == "recompute",
              script_sha256=digest(__file__))
    qa["output_sha256"] = {name: digest(HERE / name) for name in
                           ["README_zh.md", "README_en.md", "results.json", "SOURCE_SHA256.json", "SOURCE_SHA256SUMS.txt",
                            "small_data_consensus.png", "wide_cnn_ensembles.png", ".gitignore"]}
    write_json("VALIDATION.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
