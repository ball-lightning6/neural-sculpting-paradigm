"""为全部 40 个严格轨道冻结新预测，再由独立分析脚本读取 SMC K。"""

import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from finite_cgf import FiniteCGF, truth_inputs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "three_bit_v1"
EPSILONS = (.5, .2, .08, .01, .001)
BASE_ORDERS = (32, 48)
MAX_ORDER = 80
TOLERANCE_BITS = .10


def representatives(bits):
    x = truth_inputs(bits).astype(int)
    maps = []
    for perm in itertools.permutations(range(bits)):
        maps.append((x[:, perm] * (2**np.arange(bits - 1, -1, -1))).sum(1))
    seen, result = set(), []
    for f in range(2**(2**bits)):
        if f in seen:
            continue
        y = (f >> np.arange(2**bits)) & 1
        orbit = set()
        for index in maps:
            transformed = sum(int(v) << i for i, v in enumerate(y[index]))
            orbit.update((transformed, transformed ^ (2**(2**bits) - 1)))
        seen.update(orbit)
        result.append(min(orbit))
    return result


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()),
                    encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    if not json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))["all_passed"]:
        raise RuntimeError("先通过独立技术核验。")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    fids = representatives(3)
    assert len(fids) == 40
    protocol_hash = hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()
    source_hash = hashlib.sha256((ROOT / "finite_cgf.py").read_bytes()).hexdigest()
    latest, previous = {}, {}
    begin = time.perf_counter()
    pending = [(f, e) for f in fids for e in EPSILONS]
    for order in (32, 48, 64, 80):
        if not pending:
            break
        m = FiniteCGF(order=order)
        next_pending = []
        print(f"开始 q={order}，节点 {m.node_count}，待算 {len(pending)}", flush=True)
        for index, key in enumerate(pending):
            f, e = key
            path = OUT / f"q{order}_f{f:02x}_e{e}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["source_sha256"] != source_hash:
                    raise RuntimeError("既有预测源码不同，不覆盖。")
            else:
                initial = latest[key]["dual_variables"] if key in latest else None
                row = m.solve(f, e, initial=initial)
                row["source_sha256"] = source_hash
                write(path, row)
            if key in latest:
                previous[key] = latest[key]
            latest[key] = row
            change = abs(row["saddle_K_bits"] - previous[key]["saddle_K_bits"]) if key in previous else float("inf")
            if order == 32 or change > TOLERANCE_BITS or not row["numerically_converged"]:
                next_pending.append(key)
            if (index + 1) % 25 == 0 or index == 0:
                print(f"q={order} {index+1}/{len(pending)} f={f:02x} e={e} "
                      f"K={row['saddle_K_bits']:.3f} 阶数差={change:.3g} "
                      f"累计{time.perf_counter()-begin:.1f}s", flush=True)
        pending = next_pending
        del m
        torch.cuda.empty_cache()

    result = []
    for key in sorted(latest):
        row = dict(latest[key])
        change = abs(row["saddle_K_bits"] - previous[key]["saddle_K_bits"])
        row.update(quadrature_change_bits=change, quadrature_stable=change <= TOLERANCE_BITS,
                   previous_order=previous[key]["order"])
        result.append(row)
    write(OUT / "predictions_frozen.json", result)
    write(OUT / "prediction_manifest.json", {"source_sha256": source_hash, "protocol_sha256": protocol_hash,
        "predictions_sha256": hashlib.sha256((OUT / "predictions_frozen.json").read_bytes()).hexdigest(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "events": len(result),
        "stable_events": sum(r["quadrature_stable"] and r["numerically_converged"] for r in result),
        "seconds": time.perf_counter()-begin,
        "note": "本脚本不读取 SMC K，不使用 K 选择阶数或初值。"})
    print(f"完成 {len(result)} 个预测，稳定 {sum(r['quadrature_stable'] and r['numerically_converged'] for r in result)}", flush=True)


if __name__ == "__main__":
    main()
