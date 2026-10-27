"""利用已有完整函数目录，精确枚举部分训练集的最低延拓成本；不运行 SMC。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from config import WORK_DIR, CATALOGUE_ROOT


ROOT = WORK_DIR / "minimax"
ARCHITECTURES = ("tanh_w16_d1", "tanh_w32_d1", "tanh_w16_d2", "relu_w16_d1")
THRESHOLDS = np.array(
    [.7, .5, .3, .2, .12, .08, .06, .04, .025, .016, .01, .0063,
     .004, .0025, .0016, .001], dtype=np.float64
)


def read_csv(archive, name):
    return list(csv.DictReader(io.StringIO(
        archive.read(name).decode("utf-8-sig"), newline=None
    )))


def source_info(path):
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and (
            p.suffix in (".csv", ".json") or p.name in ("all_256_neural_k_curves.npz", "function_to_orbit.npz")))
        manifest = {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        return {"directory": path.name, "files_sha256": manifest}
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}


class DirectoryArchive:
    """只读取显式提供的解包目录；不查找本机 ZIP 或临时文件。"""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        if not self.path.is_dir():
            raise FileNotFoundError(f"缺少完整函数目录：{self.path.name}，见复现说明")
        return self

    def __exit__(self, *_):
        return False

    def read(self, name):
        parts = Path(name).parts
        if not parts or parts[0] != self.path.name or ".." in parts:
            raise ValueError("目录成员路径不符合输入约定")
        return self.path.joinpath(*parts[1:]).read_bytes()

    def namelist(self):
        return [self.path.name + "/" + p.relative_to(self.path).as_posix()
                for p in self.path.rglob("*") if p.is_file() and p.suffix in (".csv", ".npz", ".json")]


def input_directory(name):
    if CATALOGUE_ROOT is None:
        raise RuntimeError("需外部完整函数目录；先设置 config.CATALOGUE_ROOT。不是零依赖复现。")
    path = Path(CATALOGUE_ROOT) / name
    if not path.is_dir():
        raise FileNotFoundError(f"缺少完整函数目录：{name}，见复现说明")
    return path


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def finite_number(value):
    return float(value) if np.isfinite(value) else None


@dataclass
class Catalogue:
    bits: int
    architecture: str
    representative: np.ndarray
    representatives: np.ndarray
    replica_k: np.ndarray
    qualities: dict
    metadata: dict

    def costs(self, epsilon, aggregation="median"):
        ti = int(np.flatnonzero(np.isclose(THRESHOLDS, epsilon, atol=1e-12, rtol=0))[0])
        data = self.replica_k[ti]
        if aggregation == "median":
            cost = np.median(data, axis=1)
        elif aggregation == "first_half":
            cost = np.median(data[:, :4], axis=1)
        elif aggregation == "second_half":
            cost = np.median(data[:, 4:], axis=1)
        elif aggregation == "mean_volume":
            minimum = np.min(data, axis=1)
            cost = minimum - np.log2(np.mean(np.exp2(-(data - minimum[:, None])), axis=1))
        else:
            raise ValueError(aggregation)
        return cost[np.searchsorted(self.representatives, self.representative)]


def load_three_bit():
    path = input_directory("results_3bit_cross_architecture_complexity_order")
    source = source_info(path)
    catalogues = []
    with DirectoryArchive(path) as archive:
        for architecture in ARCHITECTURES:
            prefix = f"results_3bit_cross_architecture_complexity_order/architectures/{architecture}/"
            with np.load(io.BytesIO(archive.read(prefix + "all_256_neural_k_curves.npz")),
                         allow_pickle=False) as stored:
                mapping = stored["representative_function_ids"].astype(np.int64)
                stored_k = stored["neural_k_bits"].copy()
                np.testing.assert_array_equal(stored["function_ids"], np.arange(256))
                np.testing.assert_allclose(stored["thresholds"], THRESHOLDS)
            definitions = read_csv(archive, prefix + "representative_definitions.csv")
            representatives = np.array(sorted(int(row["function_id"]) for row in definitions))
            raw = np.full((len(THRESHOLDS), len(representatives), 8), np.nan)
            for row in definitions:
                ri = int(np.searchsorted(representatives, int(row["function_id"])))
                for record in read_csv(archive, prefix + "targets/" + row["name"]
                                       + "/replica_volume_curve.csv"):
                    ti = int(np.flatnonzero(np.isclose(THRESHOLDS, float(record["threshold"]),
                                                     atol=1e-12, rtol=0))[0])
                    replica = int(record["replica"])
                    assert np.isnan(raw[ti, ri, replica])
                    raw[ti, ri, replica] = float(record["neural_k_bits"])
            assert np.isfinite(raw).all()
            reconstructed = np.median(raw, axis=2)[:, np.searchsorted(representatives, mapping)]
            np.testing.assert_allclose(stored_k, reconstructed, atol=1e-9, rtol=0)
            config = json.loads(archive.read(prefix + "config.json"))
            catalogues.append(Catalogue(3, architecture, mapping, representatives, raw, {}, {
                "source": source, "configuration": config,
                "complete_functions": 256, "measured_orbits": len(representatives),
                "replicas": 8, "independence_caveat": "副本共享自适应阈值/尺度，不能当成完全独立复测。"
            }))
    return catalogues


def load_four_bit():
    records, qualities, sources, configs, audits = {}, {}, [], [], []
    mapping = None
    for part in (1, 2, 3):
        name = "results_4bit_full_curve_scan" if part == 1 else f"results_4bit_full_curve_scan_part{part}"
        path = input_directory(name)
        sources.append(source_info(path))
        prefix = "results_4bit_full_curve_scan/" if part == 1 else f"results_4bit_full_curve_scan_part{part}/"
        with DirectoryArchive(path) as archive:
            with np.load(io.BytesIO(archive.read(prefix + "function_to_orbit.npz")),
                         allow_pickle=False) as stored:
                candidate = stored["representative"].astype(np.int64)
                if mapping is None:
                    mapping = candidate
                else:
                    np.testing.assert_array_equal(mapping, candidate)
            config = json.loads(archive.read(prefix + "config.json"))
            configs.append(config["config"])
            assert config["config"]["settings"]["WIDTH"] == 16
            for row in read_csv(archive, prefix + "orbit_status.csv"):
                qualities[int(row["function_hex"], 16)] = row["quality"]
            aggregate = {}
            for row in read_csv(archive, prefix + "all_volume_curves.csv"):
                if row["phase"] == "first":
                    aggregate[(int(row["function_hex"], 16), float(row["threshold"]))] = row
            error = 0.0
            for name in archive.namelist():
                match = re.fullmatch(re.escape(prefix) + r"targets/(0x[0-9A-Fa-f]+)/first/replica_volume_curve.csv", name)
                if not match:
                    continue
                representative = int(match.group(1), 16)
                assert representative not in records, "三个分片不得重复覆盖 first-pass。"
                raw = np.full((len(THRESHOLDS), 8), np.nan)
                for row in read_csv(archive, name):
                    ti = int(np.flatnonzero(np.isclose(THRESHOLDS, float(row["threshold"]),
                                                     atol=1e-12, rtol=0))[0])
                    replica = int(row["replica"])
                    assert np.isnan(raw[ti, replica])
                    raw[ti, replica] = float(row["neural_k_bits"])
                for ti, epsilon in enumerate(THRESHOLDS):
                    if np.isfinite(raw[ti]).all():
                        row = aggregate[(representative, float(epsilon))]
                        error = max(error, abs(float(np.median(raw[ti]))
                                               - float(row["neural_k_bits_median"])))
                records[representative] = raw
            assert error < 1e-8
            audits.append({"part": part, "replica_to_exported_median_max_error_bits": error})
    representatives = np.array(sorted(records), dtype=np.int64)
    np.testing.assert_array_equal(representatives, np.unique(mapping))
    assert len(mapping) == 65536 and len(representatives) == 1992
    raw = np.stack([records[int(rep)] for rep in representatives], axis=1)
    missing = []
    for ti, epsilon in enumerate(THRESHOLDS):
        bad = ~np.isfinite(raw[ti]).all(axis=1)
        missing.append({"epsilon": float(epsilon), "missing_orbits": representatives[bad].tolist(),
                        "missing_functions": int(np.isin(mapping, representatives[bad]).sum())})
    return Catalogue(4, "tanh_w16_d1", mapping, representatives, raw, qualities, {
        "sources": sources, "configurations": configs, "audits": audits,
        "function_count": 65536, "measured_orbits": 1992,
        "missing_by_threshold": missing, "aggregation_source": "统一使用 first-pass，不选择性替换复核结果。",
        "quality_counts": {quality: list(qualities.values()).count(quality) for quality in set(qualities.values())},
        "independence_caveat": "副本共享自适应阈值/尺度；原始资料也不声称全局混合已认证。"
    })


def partial_minima(costs):
    """每个三进制位为已知 0、已知 1、未知 2；未知轴上对完整函数取最小值。"""
    costs = np.asarray(costs, dtype=np.float64)
    m = int(round(math.log2(len(costs))))
    assert len(costs) == 2 ** m and not np.isnan(costs).any()
    table = costs.reshape((2,) * m)
    for axis in range(m):
        table = np.concatenate((table, table.min(axis=axis, keepdims=True)), axis=axis)
    return table.ravel()


def observed_counts(m):
    counts = np.array([0], dtype=np.uint8)
    for _ in range(m):
        counts = (np.array([1, 1, 0], dtype=np.uint8)[:, None] + counts[None, :]).ravel()
    return counts


def decode_dataset(index, m):
    mask, labels, observations = 0, 0, []
    for point in range(m):
        digit, index = index % 3, index // 3
        if digit != 2:
            mask |= 1 << point
            labels |= digit << point
            observations.append({"input_index": point, "label": digit})
    return mask, labels, observations


def complete_witness(index, costs, catalogue):
    m = 2 ** catalogue.bits
    mask, labels, observations = decode_dataset(index, m)
    functions = np.arange(len(costs), dtype=np.int64)
    compatible = functions[(functions & mask) == labels]
    local = costs[compatible]
    best = float(np.min(local))
    cheapest = compatible[local == best]
    digits = (m + 3) // 4
    representatives = np.unique(catalogue.representative[cheapest])
    return {
        "ternary_index": int(index), "observed_mask": int(mask), "label_mask": int(labels),
        "observations": observations, "compatible_functions": int(len(compatible)),
        "minimum_cost": finite_number(best), "minimum_is_infinite": bool(np.isinf(best)),
        "cheapest_function_count": int(len(cheapest)),
        "cheapest_functions_first_16": [f"0x{int(f):0{digits}X}" for f in cheapest[:16]],
        "cheapest_representatives": [f"0x{int(f):0{digits}X}" for f in representatives],
        "representative_quality": {f"0x{int(f):0{digits}X}": catalogue.qualities.get(int(f), "未分级")
                                   for f in representatives},
    }


def linear_description(y):
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all() or np.ptp(y) == 0:
        return None
    x = np.arange(len(y), dtype=np.float64)
    # 只作描述，不用拟合值替换原始曲线；同时报告固定零点后的线性近似。
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    delta = y - y[0]
    anchored_slope = float(x @ delta / (x @ x))
    return {
        "range_n": [0, len(y) - 1], "slope_with_intercept": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1 - np.sum((y - fitted) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "max_residual_as_fraction_of_total_rise": float(np.max(np.abs(y - fitted)) / np.ptp(y)),
        "baseline_subtracted_origin_slope": anchored_slope,
        "baseline_subtracted_max_residual_fraction": float(np.max(np.abs(delta - anchored_slope * x)) / np.ptp(y)),
    }


def summarize_table(table, counts, *, costs=None, catalogue=None, witness=True, monotonic=True):
    m = int(np.max(counts))
    maxima = np.full(m + 1, -np.inf)
    np.maximum.at(maxima, counts, table)
    if monotonic:
        assert np.all(maxima[1:] >= maxima[:-1])
    rows = []
    for n in range(m + 1):
        selected = counts == n
        values = table[selected]
        expected = math.comb(m, n) * 2 ** n
        assert len(values) == expected
        finite = bool(np.isfinite(values).all())
        row = {
            "n": n, "datasets": expected, "maximum": finite_number(maxima[n]),
            "maximum_is_infinite": bool(np.isinf(maxima[n])),
            "maximum_minus_n0": finite_number(maxima[n] - maxima[0]),
            "increment": None if n == 0 else finite_number(maxima[n] - maxima[n - 1]),
            "mean": float(np.mean(values)) if finite else None,
            "median": float(np.median(values)) if finite else None,
            "p90": float(np.quantile(values, .9)) if finite else None,
            "p99": float(np.quantile(values, .99)) if finite else None,
            "maximizing_datasets": int(np.count_nonzero(values == maxima[n])),
        }
        if witness:
            index = int(np.flatnonzero(selected & (table == maxima[n]))[0])
            row["witness"] = complete_witness(index, costs, catalogue)
        rows.append(row)
    return {"rows": rows, "linear_fit_description_only": linear_description(maxima),
            "enumerated_datasets_including_empty": int(len(table))}


def run_case(catalogue, counts, name, costs, meaning, **extra):
    start = time.perf_counter()
    assert not np.isnan(costs).any()
    table = partial_minima(costs)
    result = summarize_table(table, counts, costs=costs, catalogue=catalogue)
    assert result["rows"][0]["maximum"] == finite_number(np.min(costs))
    assert result["rows"][-1]["maximum"] == finite_number(np.max(costs))
    result.update({"id": name, "bits": catalogue.bits, "architecture": catalogue.architecture,
                   "meaning": meaning, "seconds": time.perf_counter() - start, **extra})
    write_json(ROOT / "cases" / f"{name}.json", result)
    print(f"完成 {name}: {result['seconds']:.2f}s; M_n="
          + str([round(row["maximum"], 3) if row["maximum"] is not None else "inf"
                 for row in result["rows"]]), flush=True)
    return result


def cross_evaluate(winners, heldout_costs):
    functions = np.arange(len(heldout_costs), dtype=np.int64)
    result = []
    for row in winners["rows"]:
        witness = row["witness"]
        compatible = (functions & witness["observed_mask"]) == witness["label_mask"]
        result.append(float(np.min(heldout_costs[compatible])))
    return result


def run_envelope_slope(catalogue, counts, high, low):
    start = time.perf_counter()
    lower = partial_minima(catalogue.costs(low))
    upper = partial_minima(catalogue.costs(high))
    lower -= upper
    lower /= math.log2(high / low)
    assert np.min(lower) >= -1e-8
    result = summarize_table(lower, counts, witness=False, monotonic=False)
    name = f"b{catalogue.bits}_{catalogue.architecture}_slope_of_minimum_K_{high}_{low}"
    result.update({"id": name, "bits": catalogue.bits, "architecture": catalogue.architecture,
                   "meaning": "对每个固定 D 先取最低完整规则 K，再对该下包络求有限窗口斜率，最后对 D 取最大；不同于先对规则斜率取 min。",
                   "high_epsilon": high, "low_epsilon": low,
                   "seconds": time.perf_counter() - start})
    write_json(ROOT / "cases" / f"{name}.json", result)
    print(f"完成 {name}: {result['seconds']:.2f}s", flush=True)
    return result


def main():
    started = time.perf_counter()
    (ROOT / "cases").mkdir(parents=True, exist_ok=True)
    print("仅 CPU 目录枚举；不训练、不采样、不改历史结果。", flush=True)
    catalogues = load_three_bit() + [load_four_bit()]
    audit = {"catalogues": [dict(bits=c.bits, architecture=c.architecture, **c.metadata) for c in catalogues],
             "python": platform.python_version(), "numpy": np.__version__, "analysis_dtype": "float64",
             "script_sha256": source_info(Path(__file__))["sha256"]}
    write_json(ROOT / "SOURCE_AUDIT.json", audit)
    cases, sensitivity = [], []
    for catalogue in catalogues:
        counts = observed_counts(2 ** catalogue.bits)
        prefix = f"b{catalogue.bits}_{catalogue.architecture}"
        epsilons = [.04, .01, .004, .001] if catalogue.bits == 3 else [.04, .01, .004]
        for epsilon in epsilons:
            cases.append(run_case(catalogue, counts, f"{prefix}_K_{epsilon}", catalogue.costs(epsilon),
                                  "M_n = max_{|D|=n} min_{f compatible D} K_full(f; epsilon)",
                                  epsilon=epsilon, aggregation="median", quantity="K_minimax"))
        windows = [(.0063, .004), (.0016, .001)] if catalogue.bits == 3 else [(.0063, .004)]
        for high, low in windows:
            costs = (catalogue.costs(low) - catalogue.costs(high)) / math.log2(high / low)
            cases.append(run_case(catalogue, counts, f"{prefix}_c_{high}_{low}", costs,
                                  "max_D min_f c_f；c_f = (K_full(f; low)-K_full(f; high))/log2(high/low)",
                                  high_epsilon=high, low_epsilon=low, quantity="coefficient_minimax"))
        cases.append(run_envelope_slope(catalogue, counts, *windows[-1]))
        epsilon = .001 if catalogue.bits == 3 else .004
        for aggregate in ("mean_volume", "first_half", "second_half"):
            cases.append(run_case(catalogue, counts, f"{prefix}_K_{epsilon}_{aggregate}",
                                  catalogue.costs(epsilon, aggregate), "聚合方式/副本分组敏感性，不是独立采样置信区间。",
                                  epsilon=epsilon, aggregation=aggregate, quantity="K_sensitivity"))
        half1, half2 = cases[-2], cases[-1]
        sensitivity.append({"bits": catalogue.bits, "architecture": catalogue.architecture,
                            "epsilon": epsilon,
                            "first_half_winners_evaluated_on_second_half": cross_evaluate(half1, catalogue.costs(epsilon, "second_half")),
                            "second_half_winners_evaluated_on_first_half": cross_evaluate(half2, catalogue.costs(epsilon, "first_half")),
                            "caveat": "共享自适应参数，分组复核只能查部分敏感性，不能排除共同混合偏差。"})
        if catalogue.bits == 4:
            costs = catalogue.costs(.001)
            missing = np.isnan(costs)
            lower = costs.copy()
            upper = costs.copy()
            lower[missing] = catalogue.costs(.004)[missing]
            upper[missing] = np.inf
            for label, bound in (("lower", lower), ("upper", upper)):
                cases.append(run_case(catalogue, counts, f"{prefix}_K_0.001_{label}_bound", bound,
                                      "缺失轨道的目录补全界；仅假定缺失 K(.001)>=已测 K(.004)，其余测量值固定。不含 SMC 统计误差。",
                                      epsilon=.001, quantity="catalogue_completion_bound", bound=label,
                                      missing_functions=int(missing.sum())))
    result = {"definition": "M_n(C)=max_{S subset U, |S|=n} max_{y in {0,1}^S} min_{f|S=y} C(f)",
              "scope": "完整函数成本目录的组合枚举；不是对每个部分训练集重新测量 fixed-D BCE 体积。",
              "cases": cases, "split_sensitivity": sensitivity, "seconds": time.perf_counter() - started}
    write_json(ROOT / "RESULTS.json", result)
    with (ROOT / "CURVES.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "bits", "architecture", "n", "maximum", "maximum_minus_n0",
                                                    "increment", "mean", "median", "p90", "p99", "datasets"])
        writer.writeheader()
        for case in cases:
            for row in case["rows"]:
                writer.writerow({key: (case[key] if key in case else row.get(key)) for key in writer.fieldnames})
    print(f"全部完成，CPU 耗时 {result['seconds']:.1f}s。", flush=True)


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
