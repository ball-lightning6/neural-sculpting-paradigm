"""复核 2026-11-01 的只读径向重加权诊断；默认仅检查公开包，不运行采样。"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "radial_reweighting.json"
BASE = "research/radial_mixture_diagnostic_20261101"
D, SIGMA0 = 97, 72 / math.sqrt(99)
V0, LN2 = SIGMA0**2, math.log(2)
EPS = [0.01, 0.0063, 0.004, 0.0025, 0.0016, 0.001]
POWER, VAR = "radial_power_p", "gaussian_variance_ratio"
GRID = {POWER: [1, 1.5, 2, 2.5, 3], VAR: [0.9, 1, 1.1]}
RULES = {
    "copy": ("results_v1", 768),
    "parity2": ("results_v2_enhanced", 1536),
    "parity4": ("results_v2_enhanced", 1536),
}
CONFIG = {
    "diagnostic_date": "2026-11-01", "recheck_date": "2026-11-07",
    "dimension": D, "sigma0": SIGMA0, "epsilon": EPS, "families": GRID,
    "coordinate": "-ln(expm1(epsilon))", "power_second_moment": "d*sigma0^2",
    "gaussian_ratio": "v/v0", "pooling": "mean_probability_before_K",
    "ess": "min_over_nodes_and_pools((sum w)^2/(N*sum w^2))", "pools": 4,
    "rules": {r: {"source": f"{BASE}/{v}/{r}__gaussian", "particles_per_pool": n}
              for r, (v, n) in RULES.items()},
}
SHAPE = ("linear_slope", "first_slope", "last_slope", "drift_pct")
ESS = "min_weight_ess_fraction"
ATOL = 1e-10

# 对照值来自找回的命令输出，不是磁盘上的旧实验报告。没有历史值的项不补造。
HIST_POWER = {
    "copy": (1.9138189110671515, 1.9347614960156263, 1.9478729756514, 1.9561463585626537),
    "parity2": (4.893147122359415, 5.073501582063071, 5.193365667240075, 5.278958117625177),
    "parity4": (8.279344420246597, 8.851091033384977, 9.28365172835163, 9.62393582290316),
}
HIST_VAR = {
    "copy": (2.062263568505373, 1.9347614960156405, 1.8161667174174827),
    "parity2": (5.417375315042584, 5.073501582063108, 4.783652232501238),
    "parity4": (9.464293752753235, 8.851091033385066, 8.351027975355057),
}
HIST_P4_DRIFT = (-2.883129501638859, -2.250088577952025, -2.3721761156638044, -1.9786377763994834)
HIST_P4_ESS = (0.008240670743843258, 0.2156411969070681, 1, 0.6782778419572862, 0.35338571285034653)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare(errors, metric, value, reference, where):
    diff = abs(value - reference)
    require(math.isfinite(diff) and diff <= ATOL, f"{where}: {metric}: {value} != {reference}")
    errors[metric] = max(errors.get(metric, 0.0), diff)


def shape(x, k, slope):
    first = float((k[1] - k[0]) / (x[1] - x[0]))
    last = float((k[-1] - k[-2]) / (x[-1] - x[-2]))
    return dict(linear_slope=float(slope), first_slope=first, last_slope=last,
                drift_pct=100 * (last / first - 1))


def public_shape(k):
    x = [-math.log(math.expm1(e)) for e in EPS]
    xm, ym = math.fsum(x) / len(x), math.fsum(k) / len(k)
    slope = math.fsum((a-xm)*(b-ym) for a, b in zip(x, k)) / math.fsum((a-xm)**2 for a in x)
    return shape(x, k, slope)


def historical_check(rows):
    errors = {}
    for family, params, ref in ((POWER, GRID[POWER][1:], HIST_POWER), (VAR, GRID[VAR], HIST_VAR)):
        for rule, slopes in ref.items():
            for param, expected in zip(params, slopes):
                key = (rule, family, param)
                compare(errors, family + "_slope", rows[key]["linear_slope"], expected, key)
    for metric, params, ref in (("drift_pct", GRID[POWER][1:], HIST_P4_DRIFT), (ESS, GRID[POWER], HIST_P4_ESS)):
        for param, expected in zip(params, ref):
            key = ("parity4", POWER, param)
            compare(errors, metric, rows[key][metric], expected, key)
    return errors


def check_tables(rows):
    # 只比较本节两张实测表；公开包不声称能重新生成旧跨测度表。
    expected = []
    for family, params in GRID.items():
        table = []
        for param in params:
            curves = [rows[rule, family, param] for rule in RULES]
            cells = [f"{param:g}", *(f"{r['linear_slope']:.4f}" for r in curves)]
            if family == POWER:
                cells += [f"{curves[-1]['drift_pct']:+.2f}%", f"{curves[-1][ESS]:.5f}"]
            else:
                cells.append(" / ".join(f"{r[ESS]:.5f}" for r in curves))
            table.append(cells)
        expected.append(table)
    for lang in ("zh", "en"):
        doc = (HERE / f"README_{lang}.md").read_text(encoding="utf-8")
        section = re.search(r"^### 1\.3 [^\n]*\n(.*?)(?=^#{1,3} |\Z)", doc, re.M | re.S)
        require(section, f"README_{lang}: 缺少 §1.3")
        tables = re.findall(r"(?:^\|[^\n]*\n?)+", section[1], re.M)
        require(len(tables) == len(expected), f"README_{lang}: §1.3 表数不符")
        for table, header, wanted in zip(tables, ("p", "v/v0"), expected):
            cells = [[c.strip() for c in line.strip("|").split("|")] for line in table.splitlines()]
            labels = [c.strip("`") for c in cells[0]]
            copy = "copy 取反" if lang == "zh" else "Complemented copy"
            drift = "parity4 漂移" if lang == "zh" else "parity4 drift"
            require(labels[:4] == [header, copy, "parity2", "parity4"] and len(labels) == len(wanted[0])
                    and "ESS" in labels[-1] and (header != "p" or labels[4] == drift) and cells[2:] == wanted,
                    f"README_{lang}: §1.3 {header} 表与节点汇总不符")
    return sum(map(len, expected))


def node_name(epsilon):
    return f"node_{epsilon:.8g}.npz"


def validate(record):
    require(record["config"] == CONFIG, "公开配置不符")
    files = {f"{cfg['source']}/{name}" for cfg in CONFIG["rules"].values()
             for name in ("identity.json", "result.json", *map(node_name, EPS))}
    hashes = record["source_sha256"]
    require(hashes.keys() == files and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()),
            "来源文件列表或 SHA256 格式不符")
    rows = {(r["rule"], r["family"], r["parameter"]): r for r in record["curves"]}
    expected = {(rule, fam, p) for rule in RULES for fam, params in GRID.items() for p in params}
    require(rows.keys() == expected and len(rows) == len(record["curves"]), "曲线有缺失、重复或多余")
    arithmetic, baseline = {}, {}
    for key, row in rows.items():
        k = row["K_bits"]
        require(len(k) == len(EPS) and all(math.isfinite(v) and v >= 0 for v in k), f"{key}: 节点 K 不符")
        require(0 < row[ESS] <= 1, f"{key}: 权重 ESS 占比越界")
        for metric, value in public_shape(k).items():
            compare(arithmetic, metric, value, row[metric], key)
    for rule in RULES:
        power, gaussian = rows[rule, POWER, 2], rows[rule, VAR, 1]
        for a, b in zip(power["K_bits"], gaussian["K_bits"]):
            compare(baseline, "K_bits", a, b, rule)
        for row in (power, gaussian):
            compare(baseline, ESS, row[ESS], 1, rule)
    return rows, {"curves": len(rows), "curve_nodes": len(rows)*len(EPS),
                  "public_shape_max_abs_diff": arithmetic, "baseline_max_abs_diff": baseline,
                  "historical_max_abs_diff": historical_check(rows),
                  "readme_rows_checked_per_language": check_tables(rows)}


def read_sources():
    if len(HERE.parents) < 5 or not (HERE.parents[4] / BASE).is_dir():
        raise FileNotFoundError("缺少原始径向目录；公开包请使用 --public-only")
    import numpy as np  # 仅来源模式需要 NumPy；不导入实验模块。

    hashes, curves = {}, []
    root = HERE.parents[4]

    def source(path):
        data = (root / path).read_bytes()
        hashes[path] = hashlib.sha256(data).hexdigest()
        return data

    log_surface = LN2 + D/2 * math.log(math.pi) - math.lgamma(D/2)
    log_c0 = -D/2 * math.log(2*math.pi*V0)

    def logweights(family, p, radius):
        if family == VAR:
            return -D/2 * math.log(p) + (1-1/p) * radius**2 / (2*V0)
        log_a = (math.log(D*V0) - math.lgamma((D+2)/p) + math.lgamma(D/p)) / 2
        log_c = math.log(p) - D*log_a - log_surface - math.lgamma(D/p)
        return log_c - log_c0 - np.exp(p*(np.log(radius)-log_a)) + radius**2/(2*V0)

    x = -np.log(np.expm1(EPS))
    x -= x[0]
    for rule, cfg in CONFIG["rules"].items():
        directory = cfg["source"]
        identity = json.loads(source(f"{directory}/identity.json"))
        result = json.loads(source(f"{directory}/result.json"))
        config = identity["config"]
        require(identity["rule"] == result["rule"] == rule and identity["method"] == result["method"] == "gaussian"
                and identity["dimension"] == D and identity["sigma"] == SIGMA0
                and config["replicas"] == CONFIG["pools"] and config["particles"] == cfg["particles_per_pool"]
                and config == result["config"] and result["status"] == "complete"
                and result["signature"] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                f"{rule}: 原始配置或身份不符")
        nodes = [n for n in result["nodes"] if EPS[-1] <= n["epsilon"] <= EPS[0]]
        require([n["epsilon"] for n in nodes] == EPS, f"{rule}: 原始节点窗口不符")
        group = {(fam, p): dict(rule=rule, family=fam, parameter=p, K_bits=[], min_weight_ess_fraction=1.0)
                 for fam, params in GRID.items() for p in params}
        for node in nodes:
            path = f"{directory}/{node_name(node['epsilon'])}"
            with np.load(io.BytesIO(source(path)), allow_pickle=False) as archive:
                r = archive["radius"]
                k0 = np.asarray(node["K_bits"], dtype=np.float64)
                require(r.dtype == np.float64 and r.shape == (config["replicas"], config["particles"])
                        and np.isfinite(r).all() and (r > 0).all() and k0.shape == (config["replicas"],)
                        and np.array_equal(k0, archive["K_bits"]), f"{path}: 半径或逐池 K 不符")
            for (family, p), row in group.items():
                lw = logweights(family, p, r)
                mx = lw.max(axis=1)
                w = np.exp(lw - mx[:, None])
                k_new = k0 - (mx + np.log(w.mean(axis=1))) / LN2
                # 池内的权重归一化不能跨池；合池时平均的是概率而不是 K。
                lp = -LN2 * k_new
                k = -(lp.max() + np.log(np.exp(lp-lp.max()).mean())) / LN2
                ess = w.sum(axis=1)**2 / (r.shape[1] * (w*w).sum(axis=1))
                row["K_bits"].append(float(k))
                row[ESS] = min(row[ESS], float(ess.min()))
        for row in group.values():
            k = row["K_bits"]
            row.update(shape(x, k, np.polyfit(x, k, 1)[0]))
            curves.append(row)
    return {"config": CONFIG, "curves": curves, "source_sha256": hashes}, np.__version__


def compare_published(record, rows, published):
    saved, _ = validate(published)
    for path, digest in record["source_sha256"].items():
        require(digest == published["source_sha256"][path], f"来源 SHA256 不符：{path}")
    errors = {}
    for key, row in rows.items():
        for a, b in zip(row["K_bits"], saved[key]["K_bits"]):
            compare(errors, "K_bits", a, b, key)
        for metric in (*SHAPE, ESS):
            compare(errors, metric, row[metric], saved[key][metric], key)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--public-only", action="store_true", help="仅用标准库核对公开节点、配置及双语表格（默认）")
    mode.add_argument("--from-sources", action="store_true", help="用 NumPy 从原半径短 CPU 复算，对照已发布记录")
    parser.add_argument("--write", action="store_true", help="仅配合 --from-sources 生成轻量 JSON")
    args = parser.parse_args()
    if args.write and not args.from_sources:
        parser.error("--write 必须与 --from-sources 同用")
    start = perf_counter()
    report = {"mode": "from-sources" if args.from_sources else "public-only", "written": False,
              "weight_ess_recomputed": args.from_sources, "source_sha256_matched": 0}
    if args.from_sources:
        record, report["numpy_version"] = read_sources()
    else:
        record = json.loads(OUTPUT.read_bytes())
    rows, checks = validate(record)
    if args.write:
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        report["written"] = True
    elif args.from_sources:
        report["source_vs_published_max_abs_diff"] = compare_published(record, rows, json.loads(OUTPUT.read_bytes()))
        report["source_sha256_matched"] = len(record["source_sha256"])
    print(json.dumps(dict(report, **checks, passed=True, elapsed_seconds=round(perf_counter()-start, 3)), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as exc:
        sys.exit(f"核验失败：{exc}")
