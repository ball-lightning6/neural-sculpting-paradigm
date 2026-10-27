"""仅复算已发布小表及 hash；不调用训练或 SMC。"""

import hashlib
import json
import numpy as np
from config import ROOT, EVIDENCE, WORK_DIR
from data import verify_data


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def verify_sources():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    edited_configuration = []
    for row in manifest["published_files"]:
        path = ROOT / row["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"] and row["file"] == "config.py":
            edited_configuration.append({"file": row["file"], "release_sha256": row["sha256"], "current_sha256": actual})
        elif actual != row["sha256"]:
            raise RuntimeError("发布文件 hash 不符：" + row["file"])
    for row in manifest["historical_e32_files"]:
        path = ROOT.parent / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("历史文件 hash 不符：" + row["file"])
    return {"files_checked": len(manifest["published_files"]),
            "historical_files_checked": len(manifest["historical_e32_files"]),
            "edited_configuration": edited_configuration, "training_started": False}


def fits(curve):
    x = np.array([r["n"] for r in curve], dtype=float)
    out = {}
    for key in ("one", "two", "three"):
        y = np.array([r[key] for r in curve])
        a, b = np.polyfit(x, y, 1)
        q = x @ y / (x @ x)
        total = np.sum((y - y.mean()) ** 2)
        out[key] = {
            "a": float(a), "b": float(b), "R2_intercept": float(1 - np.sum((y-a*x-b)**2)/total),
            "q": float(q), "R2_origin_centered": float(1 - np.sum((y-q*x)**2)/total),
            "c_per_n_16": float(y[0]/x[0]), "c_per_n_128": float(y[-1]/x[-1]),
        }
    return out


def main():
    result = {"data": verify_data(), "sources": verify_sources()}
    curve = read("main_curve.json")["curve"]
    pools = read("main_pools.json")
    assert len(pools) == 96 and sum(len(r["c_by_pool"]) for r in pools) == 384
    for arch, column in (("h32", "one"), ("h32_32", "two"), ("h32_32_32", "three")):
        for row in curve:
            selected = [r for r in pools if r["architecture"] == arch and r["n"] == row["n"]]
            assert sorted(r["dataset_id"] for r in selected) == list(range(4))
            value = np.mean([np.median(r["c_by_pool"]) for r in selected])
            np.testing.assert_allclose(value, row[column], rtol=0, atol=1e-10)
    result["main_pools"] = 384
    result["fits"] = fits(curve)
    controls = read("calibration_alignment.json")
    assert [r["rule"] for r in sorted(controls["calibration"], key=lambda r:r["c_median"])] == [
        "parity1", "majority3", "mux3", "parity2", "parity4"]
    for row in controls["calibration"]:
        np.testing.assert_allclose(np.median(row["c_by_pool"]), row["c_median"], rtol=0, atol=1e-10)
        k = np.array(row["K001_by_pool"])
        mass = k.min() - np.log2(np.exp2(-(k-k.min())).mean())
        np.testing.assert_allclose(mass, row["K001_pooled_mass"], rtol=0, atol=1e-10)
    assert read("agreement.json")["verified_groups"] == 120
    assert read("dictionary.json")["dictionary_size"] == 25888
    result.update(training_started=False, smc_started=False, full_enumeration_started=False)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "review_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

