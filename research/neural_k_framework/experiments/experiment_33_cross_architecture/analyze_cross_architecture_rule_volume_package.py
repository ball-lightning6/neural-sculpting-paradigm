"""分析正式跨架构结果包；流式核验快照，不执行包内源码或启动新采样。"""

import ast
import hashlib
import io
import json
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import experiment_cross_architecture_mean_mse_resume as m


PACKAGE = Path(r"E:\Downloads\results_cross_architecture_rule_volume_compact_curves_v1_package.zip")
OUT = Path("tmp/cross_architecture_rule_volume_analysis_20260912")
RULE_LABELS = {204: "Rule204", 90: "Rule90"}


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def decode_key(key):
    name, rule, track, rep = key.split("/")
    return name, int(rule[4:]), track, int(rep[1:])


def compare_sources(archive, root):
    local = ast.parse((Path(__file__).parent / "experiment_cross_architecture_rule_volume.py").read_text(encoding="utf-8"))
    reference = {n.name: ast.dump(n, include_attributes=False) for n in local.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    important = ("dataset", "SymbolMLP", "Recurrent", "SymbolTransformer", "conv_pair", "GlobalAttention",
                 "UNet", "make_swin", "SwinUNet", "make_model", "score_raw", "parameter_prior", "Evaluator",
                 "SamplingEvaluator", "generator_for", "initialize_state", "smc_level", "record_node", "output_statistics")
    results = []
    for name in archive.namelist():
        if name.startswith(root+"sources/") and name.endswith(".py"):
            raw = archive.read(name)
            nodes = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw.decode("utf-8")).body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            differences = [key for key in important if nodes.get(key) != reference[key]]
            results.append(dict(file=name, sha256=hashlib.sha256(raw).hexdigest(), physical_ast_differences=differences))
    return results


def summarize_groups(jobs):
    groups = defaultdict(list)
    for key, job in jobs.items():
        name, rule, track, _ = decode_key(key)
        if track != "cells_max_bce":
            groups[(name, rule, track)].append(job)
    points, slopes, group_rows, fits, pair_comparisons = [], [], [], [], []
    group_points = {}
    for (name, rule, track), members in sorted(groups.items()):
        by_rep = {decode_key(j["key"])[3]: {p["cutoff"]: p for p in j["points"]} for j in members}
        common = set.intersection(*(set(v) for v in by_rep.values()))
        deepest = min(common) if common else None
        grid = sorted(set.union(*(set(v) for v in by_rep.values())), reverse=True)
        collection = []
        for cutoff in grid:
            subset = {rep: d[cutoff] for rep, d in by_rep.items() if cutoff in d}
            k = np.array([p["K_bits"] for p in subset.values()])
            stats = [p["output_statistics"] for p in subset.values()]
            row = dict(architecture=name, rule=rule, track=track, cutoff=cutoff, replicas=len(k),
                       K_by_replica={rep: p["K_bits"] for rep, p in subset.items()}, K_median=float(np.median(k)),
                       K_sd=float(k.std(ddof=1)) if len(k)>1 else None,
                       heldout_mean=float(np.mean([s["held_out"]["bit_accuracy"] for s in stats])),
                       heldout_by_replica=[s["held_out"]["bit_accuracy"] for s in stats],
                       full_target_fraction=float(np.mean([s["full_input_space"]["full_mapping_fraction"] for s in stats])),
                       full_target_by_replica=[s["full_input_space"]["full_mapping_fraction"] for s in stats],
                       entropy_mean=float(np.mean([s["full_input_space"]["empirical_function_entropy_bits"] for s in stats])),
                       full_agreement_mean=float(np.mean([s["full_input_space"]["agreement_majority"] for s in stats])),
                       training_full_fit=float(np.mean([s["full_mapping_fraction"] for s in stats])))
            row["relative_replica_sd"] = row["K_sd"]/row["K_median"] if row["K_sd"] is not None and row["K_median"] else None
            collection.append(row)
            points.append(row)
        group_points[(name, rule, track)] = {r["cutoff"]: r for r in collection}
        complete = sum(j["summary"]["cutoff"] <= m.stop_at(track) for j in members)
        selected = next((r for r in collection if r["cutoff"] == deepest), {})
        group_rows.append(dict(architecture=name, rule=rule, track=track, completed=complete,
                              deepest_all4=deepest, endpoint_losses=[j["summary"]["cutoff"] for j in members],
                              wall_seconds=sum(j["summary"]["elapsed_s"] for j in members),
                              common_K_median=selected.get("K_median"), common_K_sd=selected.get("K_sd"),
                              common_relative_sd=selected.get("relative_replica_sd"),
                              common_heldout=selected.get("heldout_mean"), common_target_fraction=selected.get("full_target_fraction")))
        for high, low in zip(m.target_grid(track)[:-1], m.target_grid(track)[1:]):
            delta = {rep: d[low]["K_bits"]-d[high]["K_bits"] for rep, d in by_rep.items() if high in d and low in d}
            if not delta:
                continue
            arr = np.array(list(delta.values())) / math.log2(high/low)
            row = dict(architecture=name, rule=rule, track=track, high=high, low=low, replicas=len(arr),
                       c_median=float(np.median(arr)), c_sd=float(arr.std(ddof=1)) if len(arr)>1 else None,
                       c_by_replica={rep: d/math.log2(high/low) for rep, d in delta.items()})
            if "bce" in track:
                a0, a1 = -math.log(math.expm1(high)), -math.log(math.expm1(low))
                row["c_a_median"] = float(np.median(list(delta.values()))/(a1-a0))
                if a0 >= 0:
                    row["c_a_squared_median"] = float(np.median(list(delta.values()))/(a1*a1-a0*a0))
            slopes.append(row)
        # 固定区间的描述拟合；不把高R²当作收敛证明或统一规律。
        if track == "cells_mean_bce" and .001 in common:
            nodes = sorted([e for e in common if .001 <= e <= .01], reverse=True)
            for rep, values in by_rep.items():
                y = np.array([values[e]["K_bits"] for e in nodes])
                for coordinate in ("log2_inverse_loss", "equivalent_a_squared"):
                    x = np.log2(1/np.array(nodes)) if coordinate == "log2_inverse_loss" else (-np.log(np.expm1(nodes)))**2
                    coef, intercept = np.polyfit(x, y, 1)
                    residual = y-(coef*x+intercept)
                    fits.append(dict(architecture=name, rule=rule, replica=rep, coordinate=coordinate,
                                     high=.01, low=.001, slope=float(coef), intercept=float(intercept),
                                     r2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
                                     rmse=float(np.sqrt(np.mean(residual**2)))))
    for name in m.Config.ARCHITECTURES:
        a = group_points[(name, 204, "cells_mean_bce")]
        b = group_points[(name, 90, "cells_mean_bce")]
        for cutoff in sorted(set(a)&set(b), reverse=True):
            if a[cutoff]["replicas"] != 4 or b[cutoff]["replicas"] != 4:
                continue
            d = np.array(list(b[cutoff]["K_by_replica"].values()))-np.array(list(a[cutoff]["K_by_replica"].values()))
            pair_comparisons.append(dict(architecture=name, cutoff=cutoff, delta_K_90_minus_204=float(np.median(d)),
                                         all_90_estimates_exceed_all_204=min(b[cutoff]["K_by_replica"].values()) > max(a[cutoff]["K_by_replica"].values())))
    return group_rows, points, slopes, fits, pair_comparisons


def numeric_snapshot_check(saved, point, name, rule, track, reference, training):
    labels = reference[rule]["labels"]
    train_ids = training[rule]["input_ids"]
    train_labels = training[rule]["labels"]
    np.testing.assert_array_equal(saved["train_input_ids"], train_ids)
    np.testing.assert_array_equal(saved["train_labels"], train_labels)
    np.testing.assert_array_equal(saved["full_input_ids"], np.arange(256))
    np.testing.assert_array_equal(saved["full_labels"], labels)
    threshold = .5 if name == "unet" and track == "native_mse" else 0.
    z = saved["train_cell_logits"].astype(np.float64)
    full = saved["full_cell_logits"]
    assert z.shape == (m.Config.PARTICLES, m.Config.POINTS, 8)
    assert full.shape == (m.Config.PARTICLES, 256, 8)
    assert np.isfinite(z).all() and np.isfinite(full).all()
    field_diff = float(np.abs(full[:, train_ids].astype(np.float64)-z).max())
    train_sign_diff = int(((full[:, train_ids] >= threshold) != (z >= threshold)).sum())
    expected_held = np.ones(256, dtype=bool)
    expected_held[train_ids] = False
    np.testing.assert_array_equal(saved["held_out_mask"], expected_held)
    predicted = full >= threshold
    packed = np.packbits(predicted.reshape(len(full), -1), axis=1, bitorder="little")
    np.testing.assert_array_equal(saved["hard_functions_packed"], packed)
    correct = predicted == labels[None].astype(bool)
    stats = point["output_statistics"]
    metric_errors = [abs(float(correct.mean())-stats["full_input_space"]["bit_accuracy"]),
                     abs(float(correct.all((1, 2)).mean())-stats["full_input_space"]["full_mapping_fraction"]),
                     abs(float(correct[:, expected_held].mean())-stats["held_out"]["bit_accuracy"])]
    vote = predicted.mean(0)
    metric_errors.append(abs(float(np.maximum(vote, 1-vote).mean())-stats["full_input_space"]["agreement_majority"]))
    _, counts = np.unique(packed, axis=0, return_counts=True)
    p = counts/len(full)
    entropy = float(-(p*np.log2(p)).sum())
    metric_errors.append(abs(entropy-stats["full_input_space"]["empirical_function_entropy_bits"]))
    score_error = 0.
    scores = saved["scores"].astype(np.float64)
    assert np.isfinite(scores).all()
    effective = float(np.float32(point["cutoff"]))
    constraint_excess = float(max(0., scores.max()-effective))
    if track in ("cells_mean_bce", "cells_max_bce"):
        losses = np.logaddexp(0, -(2*train_labels[None]-1)*z)
        estimate = losses.mean((1, 2)) if track == "cells_mean_bce" else losses.max((1, 2))
        score_error = float(np.abs(estimate-scores).max())
        np.testing.assert_allclose(estimate, scores, rtol=2e-5, atol=2e-7)
    else:
        np.testing.assert_allclose(saved["native_mse_by_particle"], scores, rtol=2e-5, atol=2e-7)
        raw = saved["raw_outputs"].astype(np.float64)
        prediction = raw if name == "unet" else np.exp(-np.logaddexp(0, -raw))
        mse = ((prediction-training[rule]["targets"][None])**2).reshape(len(raw), -1).mean(1)
        score_error = float(np.abs(mse-scores[:len(raw)]).max())
        np.testing.assert_allclose(mse, scores[:len(raw)], rtol=2e-5, atol=2e-7)
    assert max(metric_errors) < 1e-12
    assert constraint_excess == 0.
    return dict(score_max_abs_error=score_error, constraint_excess=constraint_excess,
                train_full_logit_max_difference=field_diff, train_full_sign_disagreements=train_sign_diff,
                metric_max_abs_error=max(metric_errors), full_target_particles=int(correct.all((1, 2)).sum()))


def make_figures(points, slopes, group_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {204: "#126B9C", 90: "#BD4A28"}
    fig, axes = plt.subplots(4, 3, figsize=(14, 13))
    for ax, name in zip(axes.flat, m.Config.ARCHITECTURES):
        for rule in (204, 90):
            rows = sorted([r for r in points if r["architecture"] == name and r["rule"] == rule and r["track"] == "cells_mean_bce"], key=lambda r:-r["cutoff"])
            full = [r for r in rows if r["replicas"] == 4]
            partial = [r for r in rows if r["replicas"] < 4]
            if full:
                x = [math.log2(1/r["cutoff"]) for r in full]
                y = [r["K_median"] for r in full]
                sd = [r["K_sd"] for r in full]
                ax.plot(x, y, color=colors[rule], marker=".", markersize=3, label=RULE_LABELS[rule])
                ax.fill_between(x, np.maximum(0, np.array(y)-sd), np.array(y)+sd, color=colors[rule], alpha=.15)
            if partial:
                ax.scatter([math.log2(1/r["cutoff"]) for r in partial], [r["K_median"] for r in partial],
                           marker="x", color=colors[rule], s=25)
        ax.set(title=name, xlabel="log2(1 / mean BCE limit)", ylabel="K_D (bits)")
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[10:]:
        ax.set_visible(False)
    fig.suptitle("Fixed 16-input constraints | lines: 4 replicas; x: incomplete | bands: replica SD", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT/"01_bce_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, ("unet", "unet_global", "swin_unet")):
        for rule in (204, 90):
            rows = sorted([r for r in points if r["architecture"] == name and r["rule"] == rule and r["track"] == "native_mse"], key=lambda r:-r["cutoff"])
            full, partial = [r for r in rows if r["replicas"] == 4], [r for r in rows if r["replicas"] < 4]
            if full:
                x = [math.log2(1/r["cutoff"]) for r in full]
                y = np.array([r["K_median"] for r in full])
                sd = np.array([r["K_sd"] for r in full])
                ax.plot(x, y, ".-", color=colors[rule], label=RULE_LABELS[rule])
                ax.fill_between(x, np.maximum(0, y-sd), y+sd, alpha=.15, color=colors[rule])
            if partial:
                ax.scatter([math.log2(1/r["cutoff"]) for r in partial], [r["K_median"] for r in partial], marker="x", color=colors[rule])
        ax.set(title=name+ (" | raw MSE" if name == "unet" else " | sigmoid MSE"), xlabel="log2(1 / whole-image MSE)", ylabel="K_D (bits)")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/"02_native_mse_curves.png", dpi=160)
    plt.close(fig)

    selected = [r for r in points if r["track"] == "cells_mean_bce" and r["rule"] == 204 and r["cutoff"] == .001 and r["replicas"] == 4]
    selected.sort(key=lambda r:r["K_median"])
    names = [r["architecture"] for r in selected]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    y = np.arange(len(names))
    axes[0].barh(y, [r["K_median"] for r in selected], color="#126B9C")
    axes[0].set_xscale("log")
    axes[0].set(xlabel="K_D (bits; log axis)", yticks=y, yticklabels=names)
    axes[0].invert_yaxis()
    axes[1].barh(y, [100*r["heldout_mean"] for r in selected], color="#438B59")
    axes[1].set(xlim=(0, 104), xlabel="Held-out bit accuracy (%)")
    axes[2].barh(y, [100*r["full_target_fraction"] for r in selected], color="#A35F9D")
    axes[2].set(xlim=(0, 104), xlabel="Exact 256-input bit-map recovery (%)")
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Rule204 | mean BCE <= 0.001 | 4 x 128 correlated SMC particles, not SGD models")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT/"03_copy_volume_and_function_recovery.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.flat, ("mlp", "rnn", "convnext", "transformer", "unet_global", "swin_unet")):
        for rule in (204, 90):
            rows = [r for r in slopes if r["track"] == "cells_mean_bce" and r["architecture"] == name and r["rule"] == rule and r["replicas"] == 4 and r["high"] <= .1]
            rows.sort(key=lambda r:-r["high"])
            if rows:
                ax.errorbar([math.log2(1/math.sqrt(r["high"]*r["low"])) for r in rows],
                            [r["c_median"] for r in rows], yerr=[r["c_sd"] for r in rows],
                            fmt=".-", capsize=2, color=colors[rule], label=RULE_LABELS[rule])
        ax.set(title=name, xlabel="log2(1 / window midpoint)", ylabel="delta K / delta log2(1/loss)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Finite-window slopes; only windows reached by all four replicas")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT/"04_local_contraction_slopes.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    start = time.perf_counter()
    print("读取元数据与源码AST", flush=True)
    with zipfile.ZipFile(PACKAGE) as archive:
        root = archive.namelist()[0].split("/")[0]+"/"
        load = lambda path: json.loads(archive.read(root+path))
        protocol = load("protocol.json")
        original_analysis = load("analysis.json")
        for k, v in protocol["signature"]["config"].items():
            setattr(m.Config, k, tuple(v) if isinstance(v, list) else v)
        assert m.Config.POINTS == 16 and m.Config.PARTICLES == 128 and m.Config.REPLICAS == 4
        sources = compare_sources(archive, root)
        assert sources and all(not s["physical_ast_differences"] for s in sources)
        reference, training = {}, {}
        for rule in (204, 90):
            with np.load(io.BytesIO(archive.read(root+f"full_truth_table_rule{rule}.npz")), allow_pickle=False) as npz:
                reference[rule] = {k: v.copy() for k, v in npz.items()}
            with np.load(io.BytesIO(archive.read(root+f"training_data_rule{rule}.npz")), allow_pickle=False) as npz:
                training[rule] = {k: v.copy() for k, v in npz.items()}
            truth = m.full_data(rule, "cpu")
            np.testing.assert_array_equal(reference[rule]["inputs"], truth["bits"].numpy())
            np.testing.assert_array_equal(reference[rule]["labels"], truth["labels"].numpy())
            expected = m.dataset(16, rule, "cpu")
            for k, value in training[rule].items():
                np.testing.assert_array_equal(value, expected[k].numpy() if isinstance(expected[k], torch.Tensor) else expected[k])
        jobs, level_failures, snapshot_refs = {}, [], {}
        for name in archive.namelist():
            if "/jobs/" not in name or not name.endswith("/summary.json"):
                continue
            directory = name[:-len("summary.json")]
            summary = json.loads(archive.read(name))
            key = summary["signature"]["job"]
            assert summary["signature"]["science"] == protocol["signature"]
            points = json.loads(archive.read(directory+"points.json"))
            levels = json.loads(archive.read(directory+"levels.json"))
            accumulated, prior_cut = 0., float("inf")
            for row in levels:
                accumulated -= math.log2(row["survival"])
                if abs(accumulated-row["K_bits"]) > 1e-7 or row["cutoff"] > prior_cut:
                    level_failures.append(dict(job=key, level=row["level"]))
                prior_cut = row["cutoff"]
            if levels:
                assert abs(summary["K_bits"]-levels[-1]["K_bits"]) < 1e-7
            job = dict(key=key, directory=directory, summary=summary, points=points,
                       level_count=len(levels), level_last=levels[-1] if levels else None)
            jobs[key] = job
            for point in points:
                snapshot_refs[directory+point["snapshot"]] = (key, point)
        assert len(jobs) == 106 and not level_failures
        print(f"{len(jobs)}个副本记录，{len(snapshot_refs)}个已提交快照", flush=True)
        group_rows, points, slopes, fits, pairs = summarize_groups(jobs)
        m.write_csv(OUT/"coverage_by_curve.csv", group_rows)
        m.write_csv(OUT/"points_recomputed.csv", points)
        m.write_csv(OUT/"slopes_recomputed.csv", slopes)
        m.write_csv(OUT/"tail_fits.csv", fits)
        m.write_csv(OUT/"rule_pair_comparisons.csv", pairs)
        # 确认生产端汇总算法与本轮独立重算一致。
        root_points = {(r["architecture"], int(r["rule"][4:]), r["track"], r["cutoff"]): r for r in original_analysis["points"]}
        root_slopes = {(r["architecture"], int(r["rule"][4:]), r["track"], r["high"], r["low"]): r for r in original_analysis["slopes"]}
        for row in points:
            ref = root_points[(row["architecture"], row["rule"], row["track"], row["cutoff"])]
            assert abs(row["K_median"]-ref["K_median_bits"]) < 1e-8
        for row in slopes:
            ref = root_slopes[(row["architecture"], row["rule"], row["track"], row["high"], row["low"])]
            assert abs(row["c_median"]-ref["c_log2_inverse_loss_median"]) < 1e-8
        # 每副本最后节点做权重重放；未齐副本还需保留其各自更深的末点。
        replay_names = {j["directory"]+j["points"][-1]["snapshot"] for j in jobs.values()
                        if j["points"] and "/cells_max_bce/" not in j["key"]}
        replay, numeric_audit, cache, orphan = [], [], {}, []
        for index, (filename, (key, point)) in enumerate(sorted(snapshot_refs.items())):
            name, rule, track, rep = decode_key(key)
            raw = archive.read(filename)
            assert hashlib.sha256(raw).hexdigest() == point["snapshot_sha256"], filename
            with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                check = numeric_snapshot_check(saved, point, name, rule, track, reference, training)
                numeric_audit.append(dict(job=key, cutoff=point["cutoff"], **check))
                if filename in replay_names:
                    if name not in cache:
                        model = m.make_model(name).double().eval()
                        cache[name] = model
                        manifest = load(f"models/{name}/parameter_prior.json")
                        assert m.parameter_prior(model)[0] == manifest
                    data = {k: torch.tensor(v, dtype=torch.float64) if k != "input_ids" else v for k, v in training[rule].items()}
                    ev = m.SamplingEvaluator(name, cache[name], data, track)
                    theta = saved["theta"].copy()
                    observed = saved["raw_outputs"].astype(np.float64)
                    with torch.no_grad():
                        predicted = ev.raw_batch(torch.tensor(theta, dtype=torch.float64)).numpy()
                    diff = float(np.abs(predicted-observed).max())
                    threshold = .5 if track == "native_mse" and name == "unet" else 0.
                    old_cells = m.cell_outputs(torch.tensor(observed[0])).numpy()
                    new_cells = m.cell_outputs(torch.tensor(predicted[0])).numpy()
                    sign_diff = int(((old_cells >= threshold) != (new_cells >= threshold)).sum())
                    replay.append(dict(job=key, cutoff=point["cutoff"], weights=len(theta),
                                       raw_logit_max_error=diff, raw_output_relative_l2=float(np.linalg.norm(predicted-observed)/max(np.linalg.norm(observed), 1e-12)),
                                       first_particle_observed_sign_disagreements=sign_diff))
            if (index+1) % 100 == 0:
                print(f"快照核验 {index+1}/{len(snapshot_refs)}，累计{time.perf_counter()-start:.1f}s", flush=True)
        for filename in archive.namelist():
            if "/jobs/" in filename and filename.endswith(".npz") and filename not in snapshot_refs:
                orphan.append(filename)
        m.write_csv(OUT/"snapshot_audit.csv", numeric_audit)
        m.write_csv(OUT/"weight_replay_audit.csv", replay)
        report = dict(package=str(PACKAGE), package_sha256=sha_file(PACKAGE), package_bytes=PACKAGE.stat().st_size,
                      protocol=protocol, sources=sources, active_replicas=104, completed=54,
                      active_wall_seconds=sum(j["summary"]["elapsed_s"] for j in jobs.values() if "/cells_max_bce/" not in j["key"]),
                      quarantined=[j["summary"] for j in jobs.values() if "/cells_max_bce/" in j["key"]],
                      group_rows=group_rows, points=points, slopes=slopes, tail_fits=fits, rule_pairs=pairs,
                      audit=dict(committed_snapshots=len(numeric_audit), parameter_replays=replay,
                                 numerical_max_score_error=max(r["score_max_abs_error"] for r in numeric_audit),
                                 max_constraint_excess=max(r["constraint_excess"] for r in numeric_audit),
                                 max_train_full_logit_difference=max(r["train_full_logit_max_difference"] for r in numeric_audit),
                                 total_train_full_sign_disagreements=sum(r["train_full_sign_disagreements"] for r in numeric_audit),
                                 max_metric_error=max(r["metric_max_abs_error"] for r in numeric_audit),
                                 orphan_snapshots=orphan, level_arithmetic_failures=level_failures),
                      prior_calibration=original_analysis["prior_calibration"])
        m.atomic_json(OUT/"analysis.json", report)
    make_figures(points, slopes, group_rows)
    print(f"完成：{OUT.resolve()}，总耗时{time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
