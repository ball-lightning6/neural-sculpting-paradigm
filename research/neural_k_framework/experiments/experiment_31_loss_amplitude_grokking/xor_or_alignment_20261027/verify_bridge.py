"""独立 float64 复算条件桥接终点及父快照身份。"""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bridge_results"


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_jobs"] == 8
    x = np.array([[int(c) for c in format(i, "04b")] for i in range(16)], dtype=float)
    count = 0
    maximum_stored_logit_error = 0.0
    differences = []
    for job in summary["jobs"]:
        rule, kind, eps = job["rule"], job["kind"], job["threshold"]
        fid = 0x0FF0 if rule == "f" else 0xCFFC
        parent = ROOT / "results_primary" / f"{rule}__mean_bce"
        nodes = json.loads((parent / "nodes.json").read_text(encoding="utf-8"))
        nodes = sorted([r for r in nodes if r["threshold"] == eps], key=lambda r:r["replica"])
        for r, row in enumerate(nodes):
            assert hashlib.sha256((parent / row["snapshot"]).read_bytes()).hexdigest() == job["parent_snapshot_hashes"][r]
            assert abs(row["K_bits"] - job["parent_K_bits"][r]) < 1e-10
            assert abs(job["total_K_bits"][r] - job["parent_K_bits"][r] - job["conditional_K_bits"][r]) < 1e-10
            assert job["conditional_K_bits"][r] >= 0
            with np.load(OUT / f"{rule}__{kind}__{eps}_r{r}.npz", allow_pickle=False) as data:
                t = data["theta"].astype(float)
                hidden = np.tanh(x[None] @ (t[:, :64].reshape(-1, 16, 4).transpose(0, 2, 1)/2)
                                 + t[:, None, 64:80]/2)
                z = (hidden * t[:, None, 80:96]/4).sum(-1) + t[:, 96, None]/4
                signs = np.array([1 if (fid >> i) & 1 else -1 for i in range(16)])
                margin = z * signs
                ell = np.logaddexp(0, -margin)
                objective = ell.max(-1) if kind == "max_bce" else ell.mean(-1) + .1*margin.var(-1)
                np.testing.assert_allclose(z, data["logits"], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(objective, data["score"], atol=1e-10, rtol=1e-10)
                tolerance = 2e-6 + 2e-4*eps
                assert np.all(objective <= eps + tolerance)
                assert np.all(ell.mean(-1) <= eps + tolerance)
                assert np.all(margin > 0)
                maximum_stored_logit_error = max(maximum_stored_logit_error, float(np.abs(z-data["logits"]).max()))
                count += len(t)
    for eps in (.01, .001):
        for kind in ("max_bce", "aligned"):
            f, g = [next(j for j in summary["jobs"] if j["rule"] == r and j["kind"] == kind
                         and j["threshold"] == eps) for r in ("f", "g")]
            differences.append(dict(threshold=eps, kind=kind, K_f=f["K_median"], K_g=g["K_median"],
                delta_K=g["K_median"]-f["K_median"],
                cross_pool_min=min(g["total_K_bits"])-max(f["total_K_bits"]),
                cross_pool_max=max(g["total_K_bits"])-min(f["total_K_bits"]),
                conditional_K_f=float(np.median(f["conditional_K_bits"])),
                conditional_K_g=float(np.median(g["conditional_K_bits"]))))
    result = dict(passed=True, jobs=8, snapshots=32, particles_recomputed=count,
                  maximum_stored_logit_error=maximum_stored_logit_error,
                  note="父快照身份、终点事件及加法账本通过；没有独立重建桥接的逐层存活计数，不是混合认证。",
                  differences=differences)
    (OUT / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
