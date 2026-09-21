"""固定 64 个 4bit 轨道，先冻结预测，不读取其旧 SMC 结果。"""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from finite_cgf import FiniteCGF, truth_inputs
from run_three_bit import representatives, write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "four_bit_v1"
EPSILONS = (.5, .2, .08, .01, .001)
SEED = 2026091501
ORDERS = (24, 32)
TOLERANCE = .1


def essential_variables(f):
    y = (f >> np.arange(16)) & 1
    return sum(np.any(y != y[np.arange(16) ^ (1 << bit)]) for bit in range(4))


def canonical(y):
    x = truth_inputs(4).astype(int)
    ids = []
    for p in itertools.permutations(range(4)):
        indices = (x[:, p] * (2**np.arange(3, -1, -1))).sum(1)
        f = sum(int(v) << i for i, v in enumerate(y[indices]))
        ids.extend((f, f ^ 65535))
    return min(ids)


def select_functions():
    x = truth_inputs(4).astype(int)
    pop = x.sum(1)
    named = {"constant": np.zeros(16, dtype=int), "copy": x[:, 0],
             "and4": pop == 4, "or4": pop >= 1, "parity4": pop % 2,
             "majority4_ge3": pop >= 3, "parity3_embedded": x[:, :3].sum(1) % 2,
             "majority3_embedded": x[:, :3].sum(1) >= 2}
    anchors = {name: canonical(y) for name, y in named.items()}
    all_f = representatives(4)
    assert len(all_f) == 1992
    pool = [f for f in all_f if essential_variables(f) == 4 and f not in anchors.values()]
    chosen = np.random.default_rng(SEED).choice(pool, 56, replace=False).tolist()
    selected = sorted(set(chosen) | set(anchors.values()))
    assert len(selected) == 64
    return {"seed": SEED, "anchors": anchors, "function_ids": selected,
            "four_variable_functions": sum(essential_variables(f) == 4 for f in selected),
            "selection": "8 个预定命名锚点 + 56 个均匀随机真正 4 变量轨道，不按 K 选择。"}


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest3 = json.loads((ROOT / "three_bit_v1/prediction_manifest.json").read_text(encoding="utf-8"))
    if manifest3["stable_events"] != 200:
        raise RuntimeError("3bit 技术核验未全部通过，不开始迁移。")
    selection = select_functions()
    source_hash = hashlib.sha256((ROOT / "finite_cgf.py").read_bytes()).hexdigest()
    selection.update(source_sha256=source_hash,
        three_bit_calibration_sha256=hashlib.sha256((ROOT / "frozen_three_bit_calibration.json").read_bytes()).hexdigest(),
        epsilons=list(EPSILONS), quadrature_orders=list(ORDERS), quadrature_tolerance_bits=TOLERANCE,
        primary="未标定 CGF+鞍点近似，原样使用 3bit 的推导", secondary="只沿用 3bit 标定系数，无 4bit 重标定")
    selection_path = OUT / "selection_before_prediction.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise RuntimeError("既有选择与当前协议不一致，不覆盖。")
    else:
        write(selection_path, selection)
    latest, previous = {}, {}
    begin = time.perf_counter()
    for order in ORDERS:
        m = FiniteCGF(bits=4, order=order)
        print(f"4bit q={order} 积分节点={m.node_count} 64规则x5节点", flush=True)
        for i, fid in enumerate(selection["function_ids"]):
            for eps in EPSILONS:
                path = OUT / f"q{order}_f{fid:04x}_e{eps}.json"
                key = (fid, eps)
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row["source_sha256"] != source_hash:
                        raise RuntimeError("预测核心源码不同，不覆盖。")
                else:
                    row = m.solve(fid, eps, initial=latest[key]["dual_variables"] if key in latest else None)
                    row["source_sha256"] = source_hash
                    write(path, row)
                if key in latest:
                    previous[key] = latest[key]
                latest[key] = row
            if (i + 1) % 4 == 0 or i == 0:
                print(f"4bit q={order} {i+1}/64 f={fid:04x} "
                      f"末节点 K={row['saddle_K_bits']:.3f} "
                      f"累计{time.perf_counter()-begin:.1f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    results = []
    for key in sorted(latest):
        row = dict(latest[key])
        change = abs(row["saddle_K_bits"] - previous[key]["saddle_K_bits"])
        row.update(quadrature_change_bits=change, quadrature_stable=change <= TOLERANCE,
                   previous_order=previous[key]["order"])
        results.append(row)
    write(OUT / "predictions_frozen.json", results)
    write(OUT / "prediction_manifest.json", {"events": len(results),
        "predictions_sha256": hashlib.sha256((OUT / "predictions_frozen.json").read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "source_sha256": source_hash, "seconds_current_invocation": time.perf_counter()-begin,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stable_events": sum(r["quadrature_stable"] and r["numerically_converged"] for r in results),
        "max_quadrature_change_bits": max(r["quadrature_change_bits"] for r in results),
        "note": "尚未读取所选 4bit 节点的 SMC 参考值；不按拟合程度筛除节点。"})
    print(f"4bit 完成，稳定 {sum(r['quadrature_stable'] and r['numerically_converged'] for r in results)}/320", flush=True)


if __name__ == "__main__":
    main()
