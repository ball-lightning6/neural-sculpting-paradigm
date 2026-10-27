"""独立复核：原数据/模型/测度不变，所有层增加粒子与混合；不重训SGD。"""

import math
import time
from pathlib import Path

import torch

import analysis
import common as c
import engine as e
from legacy_adapter import CASES, LegacyModel, import_reference, original_case


HERE = Path(__file__).resolve().parent
CONFIG = {
    "mode": "run", "device": "cuda:0", "allow_cpu": False,
    "original_results": "/root/autodl-tmp/smc_scale_suite_20261023/results_static",
    "output_dir": str(HERE / "results_recheck"), "run_hours": 12.,
    "cases": [name for name, _ in CASES], "replicas": 4,
    "tiers": ["baseline", "enhanced"], "slice_seconds": 1800.,
    "model_chunk": 128, "input_chunk": 16, "mutation_chunk": 128,
    "keep_final_checkpoint": False,
}
TIERS = {
    "baseline": {"particles": 1024, "sweeps": 8, "refresh_sweeps": 24},
    "enhanced": {"particles": 2048, "sweeps": 32, "refresh_sweeps": 96},
}


def _sampler(cfg, tier):
    return {**e.DEFAULTS, **TIERS[tier], "replicas": cfg["replicas"], **{k: cfg[k] for k in (
        "slice_seconds", "model_chunk", "input_chunk", "mutation_chunk", "keep_final_checkpoint")}}


def _inside(root, path):
    path = Path(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("输出目录含越界链接：" + str(path))
    return path


def _finalize(root, name, tier, arrays, cfg, protocol_hash):
    case = _inside(root, root / "cases" / f"{name}__{tier}")
    pools = [_inside(root, case / f"replica_{r}") for r in range(cfg["replicas"])]
    frozen_path = case / "forecast_frozen.json"
    anchor_path = case / "forecast_frozen_identity.json"
    if anchor_path.exists() and not frozen_path.exists():
        raise RuntimeError("已提交的冻结身份缺少清单，不允许重建覆盖")
    if frozen_path.exists() and anchor_path.exists():
        c.verify_freeze(case)
    if not all((p / "final.json").exists() for p in pools):
        if frozen_path.exists():
            raise RuntimeError("已冻结条件缺少终态池，不允许恢复写入")
        return False
    for pool in pools:
        final = c.read_json(pool / "final.json")
        if final.get("status") not in {"completed", "unresolved"}:
            raise RuntimeError("final.json 不是合法终态：" + str(pool))
        if final["status"] == "completed":
            for node in ["hard_fit", *map(e.label, _sampler(cfg, tier)["nodes"])]:
                if not all((pool / (node + suffix)).is_file() for suffix in (".npz", ".json")):
                    raise RuntimeError("已完成静态池缺少节点预测/诊断：" + str(pool / node))
    reference = root / "references" / name
    paths = [_inside(root, p) for pool in pools for p in pool.iterdir()
             if p.suffix in (".json", ".npz", ".jsonl")]
    record = {"files": {p.relative_to(case).as_posix(): c.file_hash(p) for p in sorted(paths)},
              "extra": {"dataset_hashes": {k: c.array_hash(v) for k, v in arrays.items()},
                        "tier": tier, "protocol_sha256": protocol_hash,
                        "reference_sha256": c.file_hash(reference / "reference_identity.json")},
              "policy": "历史SGD已可见；本次只冻结采样可靠性复核，不重训SGD，也不声称新的盲预测。"}
    c.guard_json(frozen_path, record)
    # 清单写入后中断时，必须先与完整预期 record 相等，才可补交身份摘要。
    c.guard_json(anchor_path, {"manifest_sha256": c.file_hash(frozen_path), "files": sorted(record["files"])})
    c.verify_freeze(case)
    total = c.read_json(reference / "sgd.json")["models"] if (reference / "sgd.json").exists() else None
    analysis.analyze_case(case, arrays["probe_y"], _sampler(cfg, tier)["nodes"], pools,
                          {"original_sgd": reference / "optimizer"}, reference / "gc", total)
    return True


def run(config=None):
    cfg = {**CONFIG, **(config or {})}
    if set(cfg) - set(CONFIG):
        raise ValueError("未知复核配置：" + ", ".join(sorted(set(cfg) - set(CONFIG))))
    if cfg["mode"] not in ("run", "preflight") or set(cfg["tiers"]) - set(TIERS):
        raise ValueError("复核 mode/tier 配置不合法")
    if set(cfg["cases"]) - {name for name, _ in CASES}:
        raise ValueError("未知原始条件")
    if (not cfg["cases"] or not cfg["tiers"] or len(set(cfg["cases"])) != len(cfg["cases"])
            or len(set(cfg["tiers"])) != len(cfg["tiers"])):
        raise ValueError("条件与 tier 必须非空且无重复")
    for key in ("replicas", "model_chunk", "input_chunk", "mutation_chunk"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(key + " 必须是正整数")
    if (not math.isfinite(cfg["run_hours"]) or cfg["run_hours"] < 0
            or not math.isfinite(cfg["slice_seconds"]) or cfg["slice_seconds"] <= 0):
        raise ValueError("总预算必须非负、切片预算必须为正，且均有限")
    root, original = Path(cfg["output_dir"]).resolve(), Path(cfg["original_results"]).resolve()
    if root.is_relative_to(original) or original.is_relative_to(root):
        raise RuntimeError("新输出与原结果目录必须完全分离；旧结果只读")
    env = c.environment(cfg["device"], cfg["allow_cpu"])
    root.mkdir(parents=True, exist_ok=True)
    provenance = {"code": c.code_hashes(HERE), "role": "已见SGD的采样可靠性复核，不是新的盲预测"}
    deadline = time.monotonic() + cfg["run_hours"] * 3600
    prepared = []
    with c.RunLock(root):
        c.guard_json(_inside(root, root / "protocol.json"), {"provenance": provenance,
            "cases": cfg["cases"], "tiers": {name: e.scientific_config(_sampler(cfg, name)) for name in cfg["tiers"]},
            "replicas": cfg["replicas"]})
        protocol_hash = c.file_hash(root / "protocol.json")
        _inside(root, root / "runtime.jsonl")
        c.record_runtime(root, env, cfg)
        for name, spec in CASES:
            if name not in cfg["cases"]:
                continue
            source, arrays, metadata = original_case(cfg["original_results"], name)
            data_folder = _inside(root, root / "data" / name)
            _inside(root, data_folder / "manifest.json")
            _inside(root, data_folder / "dataset.npz")
            identity = {"arrays": {k: c.array_hash(v) for k, v in arrays.items()}, "metadata": metadata}
            c.guard_json(data_folder / "manifest.json", identity)
            if not (data_folder / "dataset.npz").exists():
                c.atomic_npz(data_folder / "dataset.npz", **arrays)
            else:
                saved = c.read_npz(data_folder / "dataset.npz")
                if {k: c.array_hash(v) for k, v in saved.items()} != identity["arrays"]:
                    raise RuntimeError("本地复核数据已改变")
            import_reference(source, _inside(root, root / "references" / name))
            prepared.append((name, LegacyModel(spec), data_folder))
        if cfg["mode"] == "preflight":
            import validate_gpu
            return validate_gpu.check_models([(name, model, c.read_npz(path / "dataset.npz")) for name, model, path in prepared], root, cfg["device"], 32)
        planned = [root / "cases" / f"{name}__{tier}" / f"replica_{pool}" / "final.json"
                   for name, _, _ in prepared for tier in cfg["tiers"] for pool in range(cfg["replicas"])]
        # 即使预算为零，已完成的重启仍须校验冻结；不再调用采样器重投影。
        while True:
            for name, model, data_folder in prepared:
                arrays = c.read_npz(data_folder / "dataset.npz")
                ready = {tier: _finalize(root, name, tier, arrays, cfg, protocol_hash) for tier in cfg["tiers"]}
                if all(ready.values()) or c.expired(deadline):
                    continue
                data = {k: torch.from_numpy(arrays[k]).to(cfg["device"]) for k in ("x", "y", "probe_x")}
                for pool in range(cfg["replicas"]):
                    for tier in cfg["tiers"]:
                        case = _inside(root, root / "cases" / f"{name}__{tier}")
                        folder = _inside(root, case / f"replica_{pool}")
                        if c.expired(deadline):
                            break
                        if ready[tier] or (folder / "final.json").exists():
                            continue
                        sampler = _sampler(cfg, tier)
                        c.atomic_json(root / "STATUS.json", {"status": "running", "condition": name, "tier": tier, "pool": pool})
                        result = e.run_smc(model, data, sampler, folder, f"recheck:{name}:{tier}:{pool}", env, deadline, provenance)
                        if result.get("status") not in {"partial", "completed", "unresolved"}:
                            raise RuntimeError("采样器返回未知状态")
                        if result["status"] != "partial" and not (folder / "final.json").exists():
                            raise RuntimeError("采样器声称终态但缺少 final.json")
                for tier in cfg["tiers"]:
                    _finalize(root, name, tier, arrays, cfg, protocol_hash)
                del data
                torch.cuda.empty_cache() if torch.device(cfg["device"]).type == "cuda" else None
            analysis.export_summary(root)
            if all(path.exists() for path in planned) or c.expired(deadline):
                break
        finals = [path for path in planned if path.exists()]
        expected = len(planned)
        status = "completed" if len(finals) == expected else "partial"
        failed = sum(c.read_json(p)["status"] != "completed" for p in finals)
        if failed and status == "completed":
            status = "completed_with_unresolved"
        result = {"status": status, "terminal_pools": len(finals), "planned_pools": expected, "unresolved_pools": failed,
                  "note": "不自动关闭租用实例；重复同一入口续跑。统计误差与预算稳定性仍需分析。"}
        c.atomic_json(root / "STATUS.json", result)
        return result


if __name__ == "__main__":
    c.install_signals()
    print(c.clean(run(c.settings(CONFIG, HERE))), flush=True)
