"""以已有 mean-BCE 条件粒子作父集合，检查新增约束的条件体积。

这不是第三套从先验独立初始化的重复；它继承原父池误差，只改变后续收紧路径。
"""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import run_experiment as exp


OUT = exp.ROOT / "bridge_results"
PARENT = exp.ROOT / "results_primary"
THRESHOLDS = (.01, .001)


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("本诊断需要 CUDA")
    device = torch.device(exp.DEVICE)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    exp.OUT = OUT
    protocol = dict(kind="conditional_bridge_diagnostic", parent=str(PARENT), thresholds=THRESHOLDS,
                    particles=2048, replicas=4, adapt=4, fixed=64, refresh=128, seed=202610241905,
                    backend_sha256=exp.digest(Path(exp.__file__)), source_sha256=exp.digest(Path(__file__)),
                    scope="继承父 SMC 的误差及相关性；不是新的独立先验重复或全局混合证明。")
    exp.atomic_json(OUT / "protocol.json", protocol)
    output = []
    start = time.perf_counter()
    for node, threshold in enumerate(THRESHOLDS):
        for rule in exp.RULES:
            folder = PARENT / f"{rule}__mean_bce"
            rows = json.loads((folder / "nodes.json").read_text(encoding="utf-8"))
            parents = sorted([r for r in rows if r["threshold"] == threshold], key=lambda r:r["replica"])
            assert len(parents) == 4
            thetas, hashes = [], []
            for row in parents:
                path = folder / row["snapshot"]
                hashes.append(exp.digest(path))
                with np.load(path, allow_pickle=False) as data:
                    thetas.append(data["theta"].copy())
            theta = torch.tensor(np.stack(thetas), device=device)
            parent_k = np.array([r["K_bits"] for r in parents])
            baseline, x, _ = exp.scorer_for(rule, "mean_bce", device)
            assert bool((baseline(theta) <= float(np.float32(threshold))).all())
            for kind in ("max_bce", "aligned"):
                label = f"{rule}__{kind}__{threshold}"
                path = OUT / f"{label}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    assert saved["protocol"] == json.loads(json.dumps(protocol))
                    output.append(saved)
                    continue
                objective, _, _ = exp.scorer_for(rule, kind, device)
                limit = float(np.float32(threshold))
                def scorer(t):
                    loss = baseline(t)
                    return torch.where(loss <= limit, objective(t), float("inf"))
                index = node * 4 + int(rule == "g") * 2 + int(kind == "aligned")
                streams = exp.make_generators(device, index, 4, protocol["seed"])
                state = exp.initialize(4, 2048, device, streams, baseline)
                state["theta"] = theta.clone()
                state["score"] = scorer(theta)
                state["K"] = torch.tensor(parent_k, device=device)
                assert bool(torch.isfinite(state["score"]).all())
                local_start = time.perf_counter()
                exp.log(f"BRIDGE START {label}")
                for _ in range(1000):
                    hit, level = exp.step(state, scorer, streams, [threshold], 4, 64)
                    for r, yes in enumerate(hit):
                        state["node"][r] += int(yes)
                    if all(n == 1 for n in state["node"]):
                        break
                else:
                    raise RuntimeError("桥接层数超过诊断预算")
                rates = exp.mutate(state, scorer, streams, 128, 0)
                diagnostics = []
                for r in range(4):
                    t = state["theta"][r].cpu().numpy()
                    stats, values = exp.numpy_measure(t, exp.RULES[rule], kind)
                    assert float(values["score"].max()) <= threshold + 2e-6 + 2e-4 * threshold
                    assert float(values["bce"].max()) <= threshold + 2e-6 + 2e-4 * threshold
                    np.savez_compressed(OUT / f"{label}_r{r}.npz", theta=t, **values)
                    diagnostics.append(stats)
                total_k = np.array(state["K"].tolist())
                result = dict(protocol=protocol, rule=rule, kind=kind, threshold=threshold,
                              parent_snapshot_hashes=hashes, parent_K_bits=parent_k.tolist(),
                              conditional_K_bits=(total_k-parent_k).tolist(), total_K_bits=total_k.tolist(),
                              K_median=float(np.median(total_k)), acceptance=rates.tolist(),
                              levels=state["levels"], elapsed_seconds=time.perf_counter()-local_start,
                              diagnostics=diagnostics)
                exp.atomic_json(path, result)
                output.append(result)
                exp.log(f"BRIDGE DONE {label} K={total_k.tolist()}")
    exp.atomic_json(OUT / "summary.json", dict(completed_jobs=len(output), jobs=output,
                                              seconds=time.perf_counter()-start))
    exp.log(f"BRIDGE 全部完成，{(time.perf_counter()-start)/60:.2f}分钟")


if __name__ == "__main__":
    main()
