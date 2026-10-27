"""E36 新增验证的自包含入口。默认只核验配置，不训练、不采样、不下载。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import signal
import time

from portable import ROOT, load_backend, read_json, source_identity, torch

# 只修改本区；重跑科学设置改变时应换 OUTPUT_NAME，不能混入历史结果。
TASK = "inspect"  # inspect, cv34, cv24, recheck, anchor_steps, stp48, wide_data, wide_deep, adam
RUN_COMPUTE = False
DEVICE = "cuda:1"
OUTPUT_NAME = "replication_v1"
HOURS = 12.0
JOB_IDS = []  # 空列表运行该子任务全部条件；填写完整作业名可分批运行。


def configuration(task, output):
    protocols = read_json(ROOT / "config/cv_protocols.json")
    key = "extension" if task == "cv24" else "original_static"
    cfg = copy.deepcopy(protocols[key]["config"])
    cfg.update(device=DEVICE, allow_cpu=False, output_dir=str(output / task),
               mnist_root=str(ROOT / "runs/mnist"), total_hours=HOURS)
    return cfg


def datasets(core, raw, cfg, plan, output):
    all_cases = {c["name"]: c for c in core.make_cases(cfg)}
    result = {}
    for name, metadata in plan["datasets"].items():
        case = metadata.get("case", all_cases.get(name))
        if case is None:
            raise RuntimeError("缺少可重建的数据定义：" + name)
        data = core.prepare_case(case, raw, cfg, output / "datasets" / name)
        arrays = {key: value.numpy() for key, value in data.items()}
        if {key: core.digest_array(value) for key, value in arrays.items()} != metadata["arrays"]:
            raise RuntimeError("MNIST 数组与历史记录不一致：" + name)
        result[name] = arrays
    return result


def chosen(jobs):
    if not JOB_IDS:
        return jobs
    known = {job["id"] for job in jobs}
    if set(JOB_IDS) - known:
        raise ValueError("JOB_IDS 含未知作业")
    return [job for job in jobs if job["id"] in JOB_IDS]


def check_cv_data(core, case, data, stage):
    candidates = [item["record"] for item in read_json(ROOT / "config/cv_datasets.json")
                  if item["stage"] == stage and item["record"]["case"] == case]
    if not candidates or any(record["arrays"] != candidates[0]["arrays"] for record in candidates):
        raise RuntimeError("历史数据身份缺失或不一致：" + case["name"])
    if {key: core.digest_array(value) for key, value in data.items()} != candidates[0]["arrays"]:
        raise RuntimeError("重建数据与公开历史 hash 不一致：" + case["name"])


def run_recheck(b, cfg, raw, output, deadline, env):
    names = [
        "cnn_w4__anchor_0_7_n16", "cnn_w4__partition0_seed1_n32",
        "mlp_256_128__partition0_seed0_n32_flip_quarter",
    ]
    cases = {c["name"]: c for c in b.core.make_cases(cfg)}
    networks = {n["name"]: n for n in cfg["networks"]}
    valid = {name + "__" + tier for name in names for tier in b.recheck.TIERS}
    if set(JOB_IDS) - valid:
        raise ValueError("JOB_IDS 含未知复核作业")
    for name in names:
        net, case = name.split("__", 1)
        cpu = b.core.prepare_case(cases[case], raw, cfg, output / "datasets" / case)
        check_cv_data(b.core, cases[case], cpu, "original_static")
        data = {k: v.to(DEVICE) for k, v in cpu.items() if k in ("x", "y", "probe_x")}
        for tier in b.recheck.TIERS:
            job_id = name + "__" + tier
            if JOB_IDS and job_id not in JOB_IDS:
                continue
            sampler = b.recheck._sampler(dict(
                replicas=4, slice_seconds=1800., model_chunk=128, input_chunk=16,
                mutation_chunk=128, keep_final_checkpoint=False,
            ), tier)
            provenance = {"condition": name, "tier": tier, "source_hashes": source_identity()}
            model = b.adapter.LegacyModel(networks[net])
            for pool in range(4):
                folder = output / "recheck" / job_id / f"replica_{pool}"
                while not (folder / "final.json").exists() and time.monotonic() < deadline and not b.common.STOP:
                    status = b.engine.run_smc(
                        model, data, sampler, folder, f"recheck:{name}:{tier}:{pool}",
                        env, deadline, provenance,
                    )
                    if status["status"] != "partial":
                        break


def run_stp(b, cfg, raw, output, deadline, env):
    plan = read_json(ROOT / "config/stp_plan.json")
    arrays = datasets(b.core, raw, cfg, plan, output)
    plan["source_hashes"] = source_identity()
    plan["reference_config"] = cfg
    stop = [False]

    def request_stop(*args):
        stop[0] = True
        b.common.request_stop(*args)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    for job in chosen(plan["jobs"]):
        if time.monotonic() >= deadline or b.common.STOP:
            break
        b.broad.run_group(
            output / "stp48" / job["id"], job, plan, arrays[job["case"]],
            torch.device(DEVICE), max(0., deadline - time.monotonic()), stop, env,
        )


def deep_jobs(plan):
    jobs = []
    for case in ("partition0_n32_s0", "partition2_n32_s0"):
        jobs.append(dict(id=case + "__smc_enhanced", case=case, method="smc", enhanced=True))
        for method, value in (("sgd", .001), ("sgd", .0001), ("stp", .003), ("stp", .001), ("stp", .0003)):
            jobs.append(dict(id=case + f"__deep_{method}_{value:g}", case=case,
                             method=method, value=value, deep=True, max_seconds=1200.))
    return jobs


def run_wide(b, cfg, raw, output, deadline, env):
    plan = read_json(ROOT / "config/wide_plan.json")
    arrays = datasets(b.core, raw, cfg, plan, output)
    plan.update(source_hashes=source_identity(), reference_config=cfg, smoke=False,
                optimizer_seed_offset=1000000)
    jobs = deep_jobs(plan) if TASK == "wide_deep" else [
        job for job in plan["jobs"] if job["case"] in {
            "pair01_n32_s0", "pair01_n32_s1", "partition0_n16_s0", "partition0_n64_s0"}
    ]
    for job in chosen(jobs):
        if time.monotonic() >= deadline or b.common.STOP:
            break
        folder = output / TASK / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "STATUS.json").exists():
            status = read_json(folder / "STATUS.json")["status"]
            if status in ("completed", "unresolved"):
                continue
        if TASK == "wide_data":
            if job["method"] == "smc":
                b.wide.sample(plan, job, arrays[job["case"]], folder, DEVICE, deadline, env)
            else:
                b.wide.optimize(plan, job, arrays[job["case"]], folder, DEVICE,
                                min(deadline, time.monotonic() + job["max_seconds"]))
        elif job["method"] != "smc":
            b.cloud.optimize(plan, job, arrays[job["case"]], folder, DEVICE,
                             min(deadline, time.monotonic() + job["max_seconds"]))
        else:
            model = b.adapter.LegacyModel(plan["network"])
            data = {k: torch.from_numpy(arrays[job["case"]][k]).to(DEVICE) for k in ("x", "y", "probe_x")}
            provenance = dict(job=job, dataset=plan["datasets"][job["case"]], sources=source_identity())
            for pool in range(4):
                # 历史 partition0 的前两池用内存安全核，其余池用云端原核。
                engine = b.local_engine if job["case"] == "partition0_n32_s0" and pool < 2 else b.engine
                sampler = dict(engine.DEFAULTS, particles=2048, replicas=4, sweeps=32,
                               refresh_sweeps=96, nodes=plan["deep_nodes"], model_chunk=32,
                               input_chunk=16, mutation_chunk=32, slice_seconds=600.,
                               keep_final_checkpoint=False)
                target = folder / f"replica_{pool}"
                while not (target / "final.json").exists() and time.monotonic() < deadline and not b.common.STOP:
                    result = engine.run_smc(model, data, sampler, target,
                        f"local20261025:{job['id']}:{pool}", env, deadline, provenance)
                    if result["status"] != "partial":
                        break


def run_adam(b, cfg, raw, output, deadline):
    plan = read_json(ROOT / "config/wide_plan.json")
    arrays = datasets(b.core, raw, cfg, plan, output)
    model = b.core.build_model(plan["network"])
    cfg = {**cfg, "model_chunk": 1}
    generator = torch.Generator().manual_seed(cfg["data_seed"] + 7000)
    initial = b.core.sample_prior(128, model, cfg, generator, "cpu")
    torch.set_num_threads(b.adam.CONFIG["cpu_threads"])
    for init_id in b.adam.CONFIG["init_ids"]:
        for case in b.adam.CONFIG["cases"]:
            for lr in b.adam.CONFIG["rates"]:
                if time.monotonic() >= deadline or b.common.STOP:
                    return
                b.adam.run_model(case, lr, init_id, initial[init_id:init_id + 1].clone(),
                                 arrays[case], model, cfg, deadline)


def run_anchor(b, output):
    ref = b.anchor_reference
    cfg = dict(ref.CONFIG, mnist_root=str(ROOT / "runs/mnist"))
    folder = output / "anchor_reference/results"
    folder.mkdir(parents=True, exist_ok=True)
    data = ref.prepare_data(cfg, folder)
    expected = read_json(ROOT / "config/anchor_dataset.json")
    if any(ref.digest(value) != expected[key]["sha256"] for key, value in data.items()):
        raise RuntimeError("原 0/7 数据数组身份不符")
    initial = ref.prepare_initializations(cfg, folder)
    cfg = dict(b.anchor.CONFIG, **b.anchor_budget.CONFIG)
    cfg["device"] = DEVICE
    base = initial["base_theta"][:cfg["models"]].to(DEVICE) * cfg["prior_scale"]
    x, y = data["train_x"].to(DEVICE), data["train_y"].to(DEVICE)
    states = []
    for sigma in cfg["sigmas"]:
        target = output / "anchor_steps" / b.anchor_budget.group_name(sigma)
        target.mkdir(parents=True, exist_ok=True)
        state = b.anchor.run_stp(ref, cfg, target, "stp", base, sigma, x, y, cfg["stp_seed"], True)
        states.append((target, "stp", state))
        if b.anchor.STOP:
            return
    orders = torch.stack([torch.randperm(16, generator=torch.Generator().manual_seed(19092027 + epoch))
                          for epoch in range(cfg["sgd_epochs"])])
    target = output / "anchor_steps/sgd"
    target.mkdir(parents=True, exist_ok=True)
    states.append((target, "sgd", b.anchor.run_sgd(ref, cfg, target, base, orders, x, y)))
    for target, method, state in states:
        b.common.atomic_json(target / "TRAINING_FROZEN.json", {
            "files": {method + "_state.pt": b.common.file_hash(target / (method + "_state.pt"))},
            "sources": source_identity(),
        })
        b.anchor.export_predictions(ref, cfg, target, {method: state})


def main():
    tasks = ("inspect", "cv34", "cv24", "recheck", "anchor_steps", "stp48", "wide_data", "wide_deep", "adam")
    if TASK not in tasks:
        raise ValueError("未知 TASK")
    output = (ROOT / "runs" / OUTPUT_NAME).resolve()
    if Path(OUTPUT_NAME).name != OUTPUT_NAME or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("OUTPUT_NAME 必须是单个目录名")
    b = load_backend(output)
    cfg = configuration(TASK, output)
    if TASK == "inspect" or not RUN_COMPUTE:
        print(json.dumps({"mode": "只读检查", "compute_started": False,
                          "frozen_files": len(b.sources), "tasks": tasks,
                          "original_conditions": 34, "extension_conditions": 24,
                          "stp_groups": 48, "wide_deep_optimizer_seed_offset": 1000000}, ensure_ascii=False))
        return
    if TASK != "adam" and (not DEVICE.startswith("cuda:") or not torch.cuda.is_available()):
        raise RuntimeError("科学复现需要指定 CUDA；不自动回退 CPU")
    b.common.install_signals()
    if TASK == "anchor_steps":
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, b.anchor.interrupted)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True, exist_ok=True)
    identity = {"sources": source_identity(), "task": TASK, "device": "cpu" if TASK == "adam" else DEVICE,
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "dtype": "float32 parameters / float64 constraint scores"}
    b.common.guard_json(output / (TASK + "_IDENTITY.json"), identity)
    deadline = time.monotonic() + HOURS * 3600
    if TASK == "adam":
        deadline = min(deadline, time.monotonic() + b.adam.CONFIG["total_seconds"])
    env = b.common.environment("cpu" if TASK == "adam" else DEVICE, allow_cpu=TASK == "adam")
    if TASK == "anchor_steps":
        run_anchor(b, output)
        return
    if TASK in ("cv34", "cv24"):
        if JOB_IDS:
            raise ValueError("主矩阵使用冻结全矩阵，不支持 JOB_IDS 筛选")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, b.core.request_stop)
        original_prepare = b.core.prepare_case

        def checked_prepare(case, raw, config, folder):
            data = original_prepare(case, raw, config, folder)
            check_cv_data(b.core, case, data, "extension" if TASK == "cv24" else "original_static")
            return data

        b.core.prepare_case = checked_prepare
        try:
            b.core.run(cfg)
        finally:
            b.core.prepare_case = original_prepare
        return
    raw = b.core.load_mnist(cfg)
    if TASK == "recheck":
        run_recheck(b, cfg, raw, output, deadline, env)
    elif TASK == "stp48":
        run_stp(b, cfg, raw, output, deadline, env)
    elif TASK in ("wide_data", "wide_deep"):
        run_wide(b, cfg, raw, output, deadline, env)
    else:
        run_adam(b, cfg, raw, output, deadline)


if __name__ == "__main__":
    main()
