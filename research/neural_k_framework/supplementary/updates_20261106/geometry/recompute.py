"""CPU-only audit of saved JSON/CSV records; never imports experiment code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import re
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = CROSS = RADIAL = SLT = None
LN2 = math.log(2)
SOURCES = {}
ERRORS = {}


def original_sources():
    global ROOT, CROSS, RADIAL, SLT
    if len(HERE.parents) < 5:
        raise FileNotFoundError("Full audit requires the original repository; use --public-only for this package")
    ROOT = HERE.parents[4]
    CROSS = ROOT / "research/cross_architecture_contraction_20261101"
    RADIAL = ROOT / "research/radial_mixture_diagnostic_20261101"
    SLT = ROOT / "research/slt_local_probe_20261101"
    if not all(path.is_dir() for path in (CROSS, RADIAL, SLT)):
        raise FileNotFoundError("Original result directories are absent; use --public-only for this package")


def source(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = {
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)
    }
    return data


def read(path):
    return json.loads(source(path).decode("utf-8-sig"))


def close(key, actual, expected, tolerance=1e-8):
    error = abs(actual - expected)
    ERRORS[key] = max(ERRORS.get(key, 0.0), error)
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError((key, actual, expected, tolerance))


def hash_match(path, expected):
    source(path)
    assert SOURCES[path.relative_to(ROOT).as_posix()]["sha256"] == expected, path


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def recorded_path(value):
    path = PureWindowsPath(value)
    assert not path.is_absolute()
    return Path(*path.parts)


def linear(x, y):
    xm, ym = st.mean(x), st.mean(y)
    b = sum((a-xm)*(v-ym) for a, v in zip(x, y)) / sum((a-xm)**2 for a in x)
    return ym-b*xm, b


def rmse(a, b):
    return math.sqrt(st.mean((x-y)**2 for x, y in zip(a, b)))


def quadratic(x, y):
    x = [v-st.mean(x) for v in x]
    matrix = [[sum(v**(i+j) for v in x) for j in range(3)] +
              [sum(v**i*w for v, w in zip(x, y))] for i in range(3)]
    for k in range(3):
        pivot = max(range(k, 3), key=lambda i: abs(matrix[i][k]))
        matrix[k], matrix[pivot] = matrix[pivot], matrix[k]
        scale = matrix[k][k]
        matrix[k] = [v/scale for v in matrix[k]]
        for i in range(3):
            if i != k:
                scale = matrix[i][k]
                matrix[i] = [a-scale*b for a, b in zip(matrix[i], matrix[k])]
    return [row[-1] for row in matrix]


def pooled_k(values):
    minimum = min(values)
    return minimum - math.log2(st.mean(2**(-(v-minimum)) for v in values))


def shape(eps, values):
    pairs = [(e, v) for e, v in zip(eps, values) if .001 <= e <= .01]
    x = [-math.log(math.expm1(e)) for e, _ in pairs]
    y = [v for _, v in pairs]
    return {"linear_slope_bit_per_a": linear(x, y)[1],
            "quadratic_curvature_bit_per_a2": 2*quadratic(x, y)[2]}


def ledger(record, label):
    replicas = len(record["levels"][0]["K_bits"])
    total = [0.0]*replicas
    levels = {}
    for row in record["levels"]:
        assert len(row["survival"]) == replicas
        for j, survival in enumerate(row["survival"]):
            assert 0 < survival <= 1
            total[j] -= math.log2(survival)
            close(label + "_ledger_bits", total[j], row["K_bits"][j])
        levels[row["level"]] = row
    for node in record["nodes"]:
        assert node["K_bits"] == levels[node["level"]]["K_bits"]
    assert record["status"] == "complete"
    return replicas


def cross_audit():
    manifest = read(CROSS / "results_v2/manifest.json")
    for relative, digest in manifest["sources"].items():
        hash_match(CROSS / "source_snapshot_v2" / recorded_path(relative), digest)
    stored = read(CROSS / "analysis_v2/fits.json")
    historical = read(CROSS / "analysis_v2/RESULT_AUDIT.json")
    for relative, digest in historical["input_hashes"].items():
        hash_match(CROSS / recorded_path(relative), digest)
    rows, curves = [], []
    for path in sorted((CROSS / "results_v2").glob("*__*/main_x*/result.json")):
        raw = read(path)
        replicas = ledger(raw, "cross")
        identity = raw["identity"]
        pilot = read(path.parent.parent / "pilot.json")
        assert pilot["status"] == "complete" and pilot["identity"] == identity
        assert [r["epsilon"] for r in raw["levels"]] == [s["epsilon"] for s in pilot["stages"]]
        multiplier = int(path.parent.name.removeprefix("main_x"))
        offset = int(json_digest({"name": identity["architecture"], "rule": identity["rule"]})[:8], 16)
        config = dict(manifest["config"], seed=manifest["config"]["seed"]+offset)
        assert raw["signature"] == json_digest({"identity": identity, "config": config,
                                           "schedule": pilot["stages"], "multiplier": multiplier})
        for relative, digest in identity["sources"].items():
            assert digest == manifest["sources"][relative]
        count = sum(math.prod(p["shape"]) for p in identity["parameterization"])
        assert replicas == raw["replicas"] == 3
        assert len(raw["nodes"]) == 13
        assert [n["epsilon"] for n in raw["nodes"]] == manifest["nodes"]
        key = path.parent.parent.name + "/" + path.parent.name
        nodes = [n for n in raw["nodes"] if n["epsilon"] <= .03]
        x = [math.log(1/n["epsilon"]) for n in nodes]
        y = [st.mean(n["K_bits"]) for n in nodes]
        a, b = linear(x, y)
        residual = rmse([a+b*t for t in x], y)
        r2 = 1 - len(y)*residual**2/sum((v-st.mean(y))**2 for v in y)
        ta, tb = linear(x[:-3], y[:-3])
        holdout = rmse([ta+tb*t for t in x[-3:]], y[-3:])
        def power_error(p):
            xx = [v**p for v in x[:-3]]
            aa, bb = linear(xx, y[:-3])
            return rmse([aa+bb*v for v in xx], y[:-3])
        p = min((.15+i*.01 for i in range(336)), key=power_error)
        early = (y[3]-y[0])/(x[3]-x[0])
        late = (y[-1]-y[7])/(x[-1]-x[7])
        old = stored["conditions"][key]
        close("cross_fit_R2", r2, old["models"]["幂律_p1"]["full_R2_descriptive"])
        close("cross_holdout_bits", holdout, old["models"]["幂律_p1"]["holdout_rmse_bits"])
        close("cross_free_p", p, old["models"]["自由指数_仅描述"]["p"])
        close("cross_slope", early, old["window_slopes"][0]["mean"])
        close("cross_slope", late, old["window_slopes"][-1]["mean"])
        for node in raw["nodes"]:
            if node["epsilon"] < LN2/16:
                assert min(node["min_margin"]) > 0
            assert max(node["max_bce"]) <= node["epsilon"]
            curves.append({"condition": key, "epsilon": node["epsilon"],
                           "K_pool_0": node["K_bits"][0], "K_pool_1": node["K_bits"][1],
                           "K_pool_2": node["K_bits"][2]})
        rows.append({"condition": key, "parameters": count, "spec": identity["spec"],
                     "particles_per_pool": raw["particles_per_replica"], "sweeps": raw["sweeps"],
                     "endpoint_K_mean": y[-1], "endpoint_K_pool_sd": st.stdev(nodes[-1]["K_bits"]),
                     "window_delta_K": y[-1]-y[0], "early_bit_per_nat": early,
                     "late_bit_per_nat": late, "linear_R2": r2,
                     "linear_RMSE_fraction_of_K_span": residual/(y[-1]-y[0]),
                     "linear_holdout_RMSE_bits": holdout, "effective_p_descriptive": p})
    assert len(rows) == 18
    by_key = {r["condition"]: r for r in rows}
    budgets = []
    for r in rows:
        if r["condition"].endswith("main_x2"):
            first = by_key[r["condition"].replace("main_x2", "main_x1")]
            budgets.append({"architecture": r["condition"].split("__")[0],
                            "deep_K_change_bits": r["endpoint_K_mean"]-first["endpoint_K_mean"],
                            "window_delta_K_change_bits": r["window_delta_K"]-first["window_delta_K"]})
    progress = read(CROSS / "results_v2/progress.json")
    extra = read(CROSS / "results_v2/additional_budget_checks.json")
    v1 = read(CROSS / "results_v1/progress.json")
    assert len(extra["jobs"]) == 6 and all(r["status"] == "complete" for r in extra["jobs"])
    return {"complete_curves": len(rows), "complete_pools": 3*len(rows),
            "failed_final_curves": 0, "v2_prelaunch_errors_retained": sum(r["status"] == "failed" for r in progress),
            "v1_recorded_io_errors_excluded": sum(r["status"] == "failed" for r in v1),
            "rows": rows, "budget_comparisons": budgets}, curves


def radial_audit():
    versions, curves = {}, []
    for version, expected in (("results_v1", 30), ("results_v2_enhanced", 12)):
        base = RADIAL / version
        manifest = read(base / "manifest.json")
        snapshot = RADIAL / "results_v1/source"
        for filename, key in (("experiment.py", "runner_sha256"),
                              ("experiment_4bit_full_curve_scan.py", "network_sha256"),
                              ("sampler.py", "helper_sha256"),
                              ("launch_local.py", "local_launcher_sha256")):
            hash_match(snapshot / filename, manifest[key])
        results = {}
        for path in sorted(base.glob("*__*/result.json")):
            raw = read(path)
            identity = read(path.with_name("identity.json"))
            assert identity["dimension"] == 97 and identity["sigma"] == manifest["sigma"]
            assert identity["rule"] == raw["rule"] and identity["method"] == raw["method"]
            assert identity["config"] == raw["config"]
            for key in ("runner", "network", "helper"):
                assert identity[key+"_hash"] == manifest[key+"_sha256"]
            pilot = read(path.with_name("pilot.json"))
            assert pilot["status"] == "complete" and pilot["signature"] == raw["signature"]
            assert [r["epsilon"] for r in raw["levels"]] == [s["epsilon"] for s in pilot["stages"]]
            assert ledger(raw, "radial") == 4
            assert [r["epsilon"] for r in raw["nodes"]] == manifest["config"]["nodes"]
            assert raw["config"]["particles"] == manifest["config"]["particles"]
            results[raw["rule"], raw["method"]] = raw
            for node in raw["nodes"]:
                curves.append({"version": version, "rule": raw["rule"], "method": raw["method"],
                               "epsilon": node["epsilon"], **{f"K_pool_{i}": v for i, v in enumerate(node["K_bits"])}})
        assert len(results) == expected
        stored = read(base / "analysis/summary.json")
        audit = read(base / "AUDIT.json")
        assert audit["passed"] and audit["counts"]["complete"] == expected
        decompositions = {}
        bands = ["annulus_0_72", "annulus_72_80", "annulus_80_88", "annulus_88_inf"]
        maxdiff = 0.0
        for rule in sorted({key[0] for key in results}):
            cells = [results[rule, band] for band in bands]
            eps = [n["epsilon"] for n in cells[0]["nodes"]]
            masses = [c["prior_bin_mass"] for c in cells]
            close("radial_prior_mass_sum", sum(masses), 1.0)
            def chi_cdf(radius):
                x = (radius/manifest["sigma"])**2/2
                corrections = [math.exp((j+.5)*math.log(x)-x-math.lgamma(j+1.5)) for j in range(48)]
                return math.erf(math.sqrt(x))-math.fsum(corrections)
            cdfs = [0.0]+[chi_cdf(r) for r in (72, 80, 88)]+[1.0]
            for j, p in enumerate(masses):
                close("radial_prior_chi_cdf", p, cdfs[j+1]-cdfs[j], 1e-8)
            ks = [[pooled_k(n["K_bits"]) for n in c["nodes"]] for c in cells]
            mix, weights = [], []
            for i in range(len(eps)):
                terms = [p*2**(-k[i]) for p, k in zip(masses, ks)]
                mix.append(-math.log2(sum(terms)))
                weights.append([v/sum(terms) for v in terms])
            start = eps.index(.01)
            frozen = [mix[start]+sum(weights[start][j]*(ks[j][i]-ks[j][start]) for j in range(4))
                      for i in range(len(eps))]
            for i in range(len(eps)):
                kl = sum(p*math.log2(p/q) for p, q in zip(weights[start], weights[i]))
                close("radial_KL_bits", frozen[i]-mix[i], kl)
            direct = [pooled_k(n["K_bits"]) for n in results[rule, "gaussian"]["nodes"]]
            maxdiff = max(maxdiff, max(abs(a-b) for a, b in zip(mix, direct)))
            old = stored["decompositions"][rule]
            mixshape, frozenshape = shape(eps, mix), shape(eps, frozen)
            close("radial_mix_curvature", mixshape["quadratic_curvature_bit_per_a2"], old["mixture_shape"]["quadratic_curvature"])
            close("radial_frozen_curvature", frozenshape["quadratic_curvature_bit_per_a2"], old["frozen_weight_shape"]["quadratic_curvature"])
            close("radial_frozen_gap_bits", frozen[-1]-mix[-1], old["frozen_minus_mixture_at_001_bits"])
            x = [-math.log(math.expm1(e)) for e in eps[start:]]
            poly = [quadratic(x, k[start:]) for k in ks]
            terms = [p*2**(-q[0]) for p, q in zip(masses, poly)]
            midw = [v/sum(terms) for v in terms]
            mean_curv = sum(w*2*q[2] for w, q in zip(midw, poly))
            mid_slope = sum(w*q[1] for w, q in zip(midw, poly))
            variance = LN2*sum(w*(q[1]-mid_slope)**2 for w, q in zip(midw, poly))
            close("radial_mean_curvature", mean_curv, old["midpoint_fitted_mean_stratum_curvature"])
            close("radial_variance_term", variance, old["midpoint_fitted_reallocation_variance_term"])
            for i, oldrow in enumerate(old["posterior_bin_weights"]):
                for a, b in zip(weights[i], oldrow["weights"]):
                    close("radial_weights", a, b)
            decompositions[rule] = {"mixture_curvature": mixshape["quadratic_curvature_bit_per_a2"],
                "frozen_curvature": frozenshape["quadratic_curvature_bit_per_a2"],
                "midpoint_mean_curvature": mean_curv, "midpoint_variance_term": variance,
                "endpoint_frozen_gap_bits": frozen[-1]-mix[-1], "prior_bin_masses": masses,
                "outer_bin_weight_at_epsilon_0_01": weights[start][-1], "outer_bin_weight_at_epsilon_0_001": weights[-1][-1],
                "bootstrap_95_from_saved_summary_not_rerun": old["bootstrap_pool_95_intervals"],
                "reconstruction_95_from_saved_summary_not_rerun": old["reconstruction_bootstrap_95"]}
        close("radial_reconstruction_max_bits", maxdiff, stored["max_reconstruction_difference_bits"])
        spheres = []
        for (rule, method), raw in results.items():
            if method.startswith("sphere"):
                fit = shape([r["epsilon"] for r in raw["nodes"]], [pooled_k(r["K_bits"]) for r in raw["nodes"]])
                spheres.append({"rule": rule, "radius": int(method[6:]), **fit})
        versions[version] = {"complete_conditions": expected, "complete_pools": expected*4,
            "condition_nodes": expected*11, "failed_final_conditions": 0,
            "max_reconstruction_difference_bits": maxdiff, "decompositions": decompositions,
            "fixed_sphere_shapes": spheres,
            "radial_TV_from_saved_audit_not_rerun": read(base / "analysis/radial_posterior_check.json"),
            "historical_parameter_audit_not_rerun": audit["counts"]}
    historical = read(RADIAL / "historical_analysis/summary.json")
    rows = list(csv.DictReader(io.StringIO(source(RADIAL / "historical_analysis/paired_ball_gaussian.csv").decode("utf-8-sig"))))
    assert len(rows) == historical["matched_rules"] == 20
    assert sum(abs(float(r["gaussian_relative_drift"])) < abs(float(r["ball_relative_drift"])) for r in rows) == 20
    close("historical_gaussian_drift", st.median(abs(float(r["gaussian_relative_drift"])) for r in rows), historical["median_gaussian_abs_drift"])
    close("historical_ball_drift", st.median(abs(float(r["ball_relative_drift"])) for r in rows), historical["median_ball_abs_drift"])
    versions["historical_panel"] = historical
    return versions, curves


def slt_audit():
    manifest = read(SLT / "results_v2/manifest.json")
    for relative, digest in manifest["source_hashes"].items():
        path = SLT / "experiment_v2_snapshot.py" if relative.endswith("experiment.py") else ROOT / recorded_path(relative)
        hash_match(path, digest)
    v1 = read(SLT / "results_v1/manifest.json")
    hash_match(SLT / "experiment_v1_snapshot.py", v1["source_hashes"]["research\\slt_local_probe_20261101\\experiment.py"])
    source(SLT / "results_v1/FAILURE_NOTE.md")
    v1_status = [read(p) for p in sorted((SLT / "results_v1").glob("*__*/status.json"))]
    assert len(v1_status) == 2 and all(r["status"] == "complete" for r in v1_status)
    failed_pcn = read(SLT / "results_v1/relu32x3__copy/pcn_mh.json")
    assert min(failed_pcn["sampler"]["acceptance_per_chain"]) == 0
    assert read(SLT / "results_v2/completion.json")["status"] == "complete"
    stored = read(SLT / "analysis_v2/summary.json")
    geom, probes, anchors, chains = [], [], 0, 0
    min_accept = 1.0
    for condition in manifest["conditions"]:
        base = SLT / "results_v2" / condition
        assert read(base / "status.json")["status"] == "complete"
        geometry = read(base / "geometry.json")
        anchors += len(geometry)
        for r in geometry:
            assert set(r["rank_relative"].values()) == {16}
            for spectrum, key in (("logit_gram_eigenvalues", "jacobian_participation_ratio"),
                                  ("bce_ggn_eigenvalues", "bce_ggn_participation_ratio")):
                values = r[spectrum]
                close("slt_spectral_participation", sum(values)**2/sum(v*v for v in values), r[key])
        deepest = [r for r in geometry if r["target_bce"] == .0001]
        assert len(deepest) == 6
        geom.append({"condition": condition, "rank": 16,
                     "logit_participation_mean": st.mean(r["jacobian_participation_ratio"] for r in deepest),
                     "bce_ggn_participation_mean": st.mean(r["bce_ggn_participation_ratio"] for r in deepest)})
        for method in ("pcn_mh", "sgld", "sgld_halfstep"):
            raw = read(base / (method + ".json"))
            chains += len(raw["rows"])
            accept = raw["sampler"]["acceptance_per_chain"]
            if accept:
                min_accept = min(min_accept, min(accept))
            for target in (.03, .003, .0001):
                for gamma in (1e6, 1e7):
                    for ratio in (100., 10000.):
                        rows = [r for r in raw["rows"] if (r["target_bce"], r["gamma"], r["beta_gamma_ratio"]) == (target, gamma, ratio)]
                        assert len(rows) == 18
                        mean = st.mean(r["finite_scale_cost"] for r in rows)
                        old = next(r for r in stored["probes"] if (r["condition"], r["method"], r["bce"], r["gamma"], r["beta_gamma_ratio"]) == (condition, method, target, gamma, ratio))
                        close("slt_cost_summary", mean, old["cost_mean"])
                        probes.append({"condition": condition, "method": method, "bce": target,
                                       "gamma": gamma, "ratio": ratio, "cost_mean": mean,
                                       "linear_reference": st.mean(r["linear_finite_scale_reference"] for r in rows)})
        read(base / "unit_checks.json")
    assert anchors == 72 and chains == 2592
    comparisons = {}
    for method in ("sgld", "sgld_halfstep"):
        differences = []
        for r in probes:
            if r["method"] != method:
                continue
            pcn = next(p for p in probes if p["method"] == "pcn_mh" and all(p[k] == r[k] for k in ("condition", "bce", "gamma", "ratio")))
            differences.append(abs(r["cost_mean"]-pcn["cost_mean"])/pcn["cost_mean"])
        comparisons[method + "_max_group_relative_difference_from_pcn"] = max(differences)
    identity = read(SLT / "readout_identity_results.json")
    hash_match(SLT / "readout_identity.py", identity["source_sha256"])
    identity_rows = []
    for r in identity["results"]:
        raw = read(CROSS / "results_v2" / r["condition"] / f"main_x{r['budget_multiplier']}/result.json")
        nodes = {n["epsilon"]: n["K_bits"] for n in raw["nodes"]}
        for i, point in enumerate(r["point_kappa_by_pool"]):
            close("readout_decomposition", point, sum(r[k][i] for k in ("dimension_term_by_pool", "norm_term_by_pool", "derivative_term_by_pool")))
            close("readout_secant", r["previous_interval_kappa_by_pool"][i], nodes[.0001][i]-nodes[.0002][i])
        point, secant = st.mean(r["point_kappa_by_pool"]), st.mean(r["previous_interval_kappa_by_pool"])
        identity_rows.append({"condition": r["condition"], "budget_multiplier": r["budget_multiplier"],
                              "head_dimension": r["head_dimension"], "point_kappa": point, "secant_kappa": secant,
                              "relative_point_minus_secant": (point-secant)/secant,
                              "dimension_term": st.mean(r["dimension_term_by_pool"]),
                              "norm_term": st.mean(r["norm_term_by_pool"]),
                              "derivative_term": st.mean(r["derivative_term_by_pool"]),
                              "checkpoint_sha256_reported_not_rehashed": r["checkpoint_sha256"]})
    calibration = read(SLT / "results_v2/calibration.json")
    assert all(r["passed"] for r in calibration)
    read(SLT / "results_v2/unit_checks.json")
    read(SLT / "AUDIT.json")
    return {"complete_conditions": 4, "failed_final_conditions": 0, "prototypes": 24,
            "controlled_anchors": anchors, "local_chain_runs_not_datasets": chains,
            "v1_complete_conditions_excluded": len(v1_status), "v1_failed_conditions": 1, "v1_not_started_conditions": 1,
            "pcn_min_acceptance": min_accept, "deep_geometry": geom,
            "deep_pcn_costs": [r for r in probes if r["bce"] == .0001 and r["gamma"] == 1e7 and r["method"] == "pcn_mh"],
            "sampler_comparisons": comparisons,
            "pcn_linear_reference_max_difference": max(abs(r["cost_mean"]-r["linear_reference"]) for r in probes if r["method"] == "pcn_mh"),
            "readout_identity": identity_rows,
            "historical_identity_check_max_relative_error_not_rerun": max(r["relative_error"] for r in identity["checks"])}


def csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def public_audit():
    """Recompute published numbers without opening any original source path."""
    package_hashes = {}

    def package_text(name):
        assert name in {"audit_summary.json", "source_manifest.json", "cross_architecture_curves.csv", "radial_curves.csv"}
        path = (HERE / name).resolve()
        assert path.parent == HERE
        data = path.read_bytes()
        package_hashes[name] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig")

    summary = json.loads(package_text("audit_summary.json"))
    manifest = json.loads(package_text("source_manifest.json"))
    assert summary["passed"] is True
    assert manifest["path_base"] == "repository_root" and manifest["hash_algorithm"] == "SHA256"
    for name, metadata in manifest["entries"].items():
        path = PureWindowsPath(name)
        assert not path.is_absolute() and not path.drive and ".." not in path.parts
        assert re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])
        assert isinstance(metadata["bytes"], int) and metadata["bytes"] > 0

    def curves(name, identities, pools, expected_nodes):
        reader = csv.DictReader(io.StringIO(package_text(name)))
        fields = [*identities, "epsilon", *(f"K_pool_{i}" for i in range(pools))]
        assert reader.fieldnames == fields, (name, "CSV columns differ")
        groups = {}
        for row in reader:
            assert None not in row and all(value is not None for value in row.values())
            key = tuple(row[column] for column in identities)
            epsilon = float(row["epsilon"])
            values = [float(row[f"K_pool_{i}"]) for i in range(pools)]
            assert all(math.isfinite(v) and v >= 0 for v in values)
            group = groups.setdefault(key, {})
            assert epsilon not in group, (name, key, "Duplicate node")
            group[epsilon] = values
        for key, group in groups.items():
            assert sorted(group, reverse=True) == expected_nodes, (name, key, "Missing or unexpected node")
            for first, last in zip(expected_nodes, expected_nodes[1:]):
                assert all(a <= b+1e-10 for a, b in zip(group[first], group[last])), (name, key, "Nonmonotonic K")
        return groups

    cross_nodes = [.3, .1, .03, .02, .01, .006, .003, .002, .001, .0006, .0003, .0002, .0001]
    cross = curves("cross_architecture_curves.csv", ["condition"], 3, cross_nodes)
    names = ("tanh16", "tanh32x3", "relu32x3", "cnn8x2", "gru16", "transformer8")
    expected = {f"{name}__{rule}/main_x1" for name in names for rule in ("copy", "parity2")}
    expected |= {f"{name}__parity2/main_x2" for name in names}
    assert {key[0] for key in cross} == expected
    saved_cross = summary["cross_architecture"]
    indexed = {row["condition"]: row for row in saved_cross["rows"]}
    assert len(indexed) == len(saved_cross["rows"]) == len(cross) == 18
    assert set(indexed) == expected
    assert saved_cross["complete_curves"] == 18 and saved_cross["complete_pools"] == 54
    stats = {}
    for (key,), nodes in cross.items():
        eps = cross_nodes[2:]
        x = [math.log(1/e) for e in eps]
        y = [st.mean(nodes[e]) for e in eps]
        a, b = linear(x, y)
        residual = rmse([a+b*t for t in x], y)
        ta, tb = linear(x[:-3], y[:-3])
        def power_error(p):
            xx = [v**p for v in x[:-3]]
            aa, bb = linear(xx, y[:-3])
            return rmse([aa+bb*v for v in xx], y[:-3])
        stats[key] = {
            "endpoint_K_mean": y[-1], "endpoint_K_pool_sd": st.stdev(nodes[eps[-1]]),
            "window_delta_K": y[-1]-y[0],
            "early_bit_per_nat": (y[3]-y[0])/(x[3]-x[0]),
            "late_bit_per_nat": (y[-1]-y[7])/(x[-1]-x[7]),
            "linear_R2": 1-len(y)*residual**2/sum((v-st.mean(y))**2 for v in y),
            "linear_RMSE_fraction_of_K_span": residual/(y[-1]-y[0]),
            "linear_holdout_RMSE_bits": rmse([ta+tb*t for t in x[-3:]], y[-3:]),
            "effective_p_descriptive": min((.15+i*.01 for i in range(336)), key=power_error),
        }
        for field, value in stats[key].items():
            close("public_cross_"+field, value, indexed[key][field])
        multiplier = int(key[-1])
        assert indexed[key]["particles_per_pool"] == 768*multiplier
        assert indexed[key]["sweeps"] == 12*multiplier
    budgets = {}
    assert len(saved_cross["budget_comparisons"]) == 6
    for old in saved_cross["budget_comparisons"]:
        name = old["architecture"]
        assert name in names and name not in budgets
        first, extra = [stats[f"{name}__parity2/main_x{i}"] for i in (1, 2)]
        budgets[name] = {"deep_K_change_bits": extra["endpoint_K_mean"]-first["endpoint_K_mean"],
                         "window_delta_K_change_bits": extra["window_delta_K"]-first["window_delta_K"]}
        for field, value in budgets[name].items():
            close("public_cross_"+field, value, old[field])

    radial_nodes = [.7, .3, .1, .03, .016, .01, .0063, .004, .0025, .0016, .001]
    radial = curves("radial_curves.csv", ["version", "rule", "method"], 4, radial_nodes)
    bands = ["annulus_0_72", "annulus_72_80", "annulus_80_88", "annulus_88_inf"]
    versions = {}
    expected_keys = set()
    for version, rules, methods in (
        ("results_v1", ("copy", "parity2", "parity4"), ["gaussian", *bands, "ball72", "sphere72", "sphere80", "sphere88", "sphere96"]),
        ("results_v2_enhanced", ("parity2", "parity4"), ["gaussian", *bands, "sphere72"]),
    ):
        expected_keys.update((version, rule, method) for rule in rules for method in methods)
        old_version = summary["radial_mixture"][version]
        count = len(rules)*len(methods)
        assert old_version["complete_conditions"] == count
        assert old_version["complete_pools"] == count*4 and old_version["condition_nodes"] == count*len(radial_nodes)
        maxdiff, decompositions = 0.0, {}
        for rule in rules:
            old = old_version["decompositions"][rule]
            masses = old["prior_bin_masses"]
            assert len(masses) == 4 and all(0 < p < 1 for p in masses)
            close("public_prior_mass_sum", sum(masses), 1.0)
            sigma = 72/math.sqrt(99)
            cdfs = [0.0]
            for radius in (72, 80, 88):
                t = (radius/sigma)**2/2
                terms = [math.exp((j+.5)*math.log(t)-t-math.lgamma(j+1.5)) for j in range(48)]
                cdfs.append(math.erf(math.sqrt(t))-math.fsum(terms))
            cdfs.append(1.0)
            for j, p in enumerate(masses):
                close("public_prior_chi_cdf", p, cdfs[j+1]-cdfs[j], 1e-8)
            ks = [[pooled_k(radial[version, rule, band][e]) for e in radial_nodes] for band in bands]
            mix, weights = [], []
            for i in range(len(radial_nodes)):
                terms = [p*2**(-k[i]) for p, k in zip(masses, ks)]
                mix.append(-math.log2(sum(terms)))
                weights.append([value/sum(terms) for value in terms])
            start = radial_nodes.index(.01)
            frozen = [mix[start]+sum(weights[start][j]*(ks[j][i]-ks[j][start]) for j in range(4))
                      for i in range(len(radial_nodes))]
            for i in range(len(radial_nodes)):
                kl = sum(p*math.log2(p/q) for p, q in zip(weights[start], weights[i]))
                close("public_radial_KL_bits", frozen[i]-mix[i], kl)
            direct = [pooled_k(radial[version, rule, "gaussian"][e]) for e in radial_nodes]
            maxdiff = max(maxdiff, max(abs(a-b) for a, b in zip(mix, direct)))
            x = [-math.log(math.expm1(e)) for e in radial_nodes[start:]]
            poly = [quadratic(x, k[start:]) for k in ks]
            terms = [p*2**(-q[0]) for p, q in zip(masses, poly)]
            midw = [v/sum(terms) for v in terms]
            midslope = sum(w*q[1] for w, q in zip(midw, poly))
            result = {
                "mixture_curvature": shape(radial_nodes, mix)["quadratic_curvature_bit_per_a2"],
                "frozen_curvature": shape(radial_nodes, frozen)["quadratic_curvature_bit_per_a2"],
                "midpoint_mean_curvature": sum(w*2*q[2] for w, q in zip(midw, poly)),
                "midpoint_variance_term": LN2*sum(w*(q[1]-midslope)**2 for w, q in zip(midw, poly)),
                "endpoint_frozen_gap_bits": frozen[-1]-mix[-1],
                "outer_bin_weight_at_epsilon_0_01": weights[start][-1],
                "outer_bin_weight_at_epsilon_0_001": weights[-1][-1],
            }
            for field, value in result.items():
                close("public_radial_"+field, value, old[field])
            decompositions[rule] = result
        close("public_radial_reconstruction_bits", maxdiff, old_version["max_reconstruction_difference_bits"])
        seen_spheres = set()
        for old in old_version["fixed_sphere_shapes"]:
            key = (version, old["rule"], f"sphere{old['radius']}")
            assert key not in seen_spheres
            seen_spheres.add(key)
            values = [pooled_k(radial[key][e]) for e in radial_nodes]
            for field, value in shape(radial_nodes, values).items():
                close("public_sphere_"+field, value, old[field])
        assert seen_spheres == {(version, rule, method) for rule in rules for method in methods if method.startswith("sphere")}
        versions[version] = {"conditions_in_csv": count, "pool_curves_in_csv": count*4,
                             "max_reconstruction_difference_bits": maxdiff, "decompositions": decompositions}
    assert set(radial) == expected_keys

    readout = summary["slt_readout"]["readout_identity"]
    expected_readout = {(f"{name}__{rule}", budget) for name in ("tanh16", "relu32x3")
                        for rule, budget in (("copy", 1), ("parity2", 1), ("parity2", 2))}
    assert len(readout) == 6 and {(r["condition"], r["budget_multiplier"]) for r in readout} == expected_readout
    for row in readout:
        nodes = cross[(f"{row['condition']}/main_x{row['budget_multiplier']}",)]
        secant = st.mean(b-a for a, b in zip(nodes[.0002], nodes[.0001]))
        close("public_readout_secant", secant, row["secant_kappa"])
        close("public_readout_component_sum", row["point_kappa"], sum(row[k] for k in ("dimension_term", "norm_term", "derivative_term")))
        close("public_readout_relative_gap", (row["point_kappa"]-secant)/secant, row["relative_point_minus_secant"])
    return {"passed": True, "mode": "public_only_package_arithmetic_not_smc_replication",
            "package_file_sha256": package_hashes, "source_manifest_entries_metadata_only": len(manifest["entries"]),
            "cross_curve_nodes": len(cross)*len(cross_nodes), "cross_pool_curves": len(cross)*3,
            "cross_budget_comparisons": budgets, "radial_condition_nodes": len(radial)*len(radial_nodes),
            "radial_results": versions, "readout_summary_rows_checked": len(readout),
            "max_arithmetic_differences": dict(ERRORS),
            "limitations": ["Uses packaged node-level K values and saved summary terms, not raw SMC runs.",
                            "Does not recheck original source hashes, per-level survival ledgers, run status or failure logs.",
                            "Does not regenerate SLT spectra or chain statistics, particle forward passes, bootstrap intervals, TV or the historical panel.",
                            "Agreement between packaged numbers does not establish original-data authenticity or global mixing."]}


def check_documents():
    docs = [(HERE / f"README_{lang}.md").read_text(encoding="utf-8") for lang in ("zh", "en")]
    def structure(doc):
        blocks = re.split(r"\n\s*\n", doc.strip())
        kinds = []
        for block in blocks:
            if block.startswith("#"):
                kinds.append(("heading", len(block.split(" ")[0])))
            elif block.startswith("|"):
                kinds.append(("table", len(block.splitlines()), [len(line.split("|")) for line in block.splitlines()]))
            elif block.startswith("$$"):
                kinds.append(("math", block))
            elif block.startswith("```"):
                kinds.append(("code", block))
            else:
                kinds.append(("paragraph",))
        return kinds
    assert structure(docs[0]) == structure(docs[1]), "Bilingual block structure differs"
    for doc in docs:
        for target in re.findall(r"\]\(([^)]+)\)", doc):
            if not target.startswith(("https:", "http:", "#")):
                path = (HERE / target.split("#")[0]).resolve()
                assert path.is_relative_to(HERE) and path.exists(), target
    tables = [re.findall(r"^\|.*$", doc, re.M) for doc in docs]
    number = r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    assert [re.findall(number, row) for row in tables[0]] == [re.findall(number, row) for row in tables[1]], "Bilingual table numbers differ"
    blocks = [re.split(r"\n\s*\n", doc.strip()) for doc in docs]
    assert all(sorted(re.findall(number, a)) == sorted(re.findall(number, b))
               for a, b in zip(*blocks)), "Bilingual paragraph numbers differ"
    return {"matched_blocks": len(structure(docs[0])), "matched_table_rows": len(tables[0]), "links_exist": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write derived files only next to this script")
    parser.add_argument("--documents", action="store_true", help="Check bilingual structure, formulas, numbers and links")
    parser.add_argument("--public-only", action="store_true", help="Read packaged CSV/JSON only; no original sources or sampling")
    args = parser.parse_args()
    if args.public_only:
        if args.write:
            parser.error("--public-only is read-only and cannot be combined with --write")
        report = public_audit()
        if args.documents:
            report["documents"] = check_documents()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    original_sources()
    cross, cross_curves = cross_audit()
    radial, radial_curves = radial_audit()
    slt = slt_audit()
    context = [
        CROSS / "REPORT_ZH.md", CROSS / "README.md", CROSS / "analyze.py", CROSS / "audit_results.py",
        CROSS / "source_snapshot_v2/README.md", CROSS / "analysis_v2/ANALYSIS_MANIFEST.json",
        RADIAL / "REPORT_ZH.md", RADIAL / "README.md", RADIAL / "analyze.py", RADIAL / "audit.py",
        RADIAL / "check_radial_posterior.py", RADIAL / "historical_analysis/independent_archive_audit.json",
        SLT / "REPORT_ZH.md", SLT / "README.md", SLT / "READOUT_IDENTITY_ZH.md", SLT / "analyze.py",
        ROOT / "research/notes/20261105_amplitude_alignment_mechanism_revision_zh.md",
        ROOT / "research/notes/20261105_core_volume_generalization_literature_review_zh.md",
        ROOT / "research/neural_k_framework/experiments/experiment_33_cross_architecture/README.md",
        ROOT / "research/neural_k_framework/experiments/experiment_33_cross_architecture/OUTPUT_CONTRACTION_DERIVATION.md",
        ROOT / "research/neural_k_framework/experiments/experiment_37_measure_robustness/README.md",
        ROOT / "research/neural_k_framework/experiments/experiment_37_measure_robustness/FIXED_RADIUS_20261027.md",
    ]
    for path in context:
        source(path)
    summary = {"passed": True, "audit_date": "2026-11-06", "baseline_commit": "3834fb3f6555c7c58218462502635b16b9b377f9",
               "mode": "stdlib_cpu_saved_records_no_training_no_smc_no_gpu",
               "limitations": ["No checkpoint or particle array forward recomputation in this audit.",
                               "Saved bootstrap intervals and radial TV diagnostics were not regenerated.",
                               "Numerical consistency does not certify global mixing."],
               "cross_architecture": cross, "radial_mixture": radial, "slt_readout": slt,
               "max_recomputation_errors": ERRORS}
    outputs = {
        "audit_summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "cross_architecture_curves.csv": csv_text(cross_curves),
        "radial_curves.csv": csv_text(radial_curves),
        "source_manifest.json": json.dumps({"path_base": "repository_root", "hash_algorithm": "SHA256",
            "entries": dict(sorted(SOURCES.items()))}, ensure_ascii=False, indent=2) + "\n",
    }
    for name, data in outputs.items():
        path = HERE / name
        if args.write:
            path.write_text(data, encoding="utf-8", newline="\n")
        else:
            assert path.read_text(encoding="utf-8") == data, f"Derived file differs: {name}"
    report = {"passed": True, "sources": len(SOURCES), "cross_pools": cross["complete_pools"],
              "radial_pools": sum(radial[v]["complete_pools"] for v in ("results_v1", "results_v2_enhanced")),
              "slt_anchors": slt["controlled_anchors"], "max_errors": ERRORS}
    if args.documents:
        report["documents"] = check_documents()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
