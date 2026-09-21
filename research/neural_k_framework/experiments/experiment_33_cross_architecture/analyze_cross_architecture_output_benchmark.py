"""读取5090测速包，复核数据/参数化/前向与SMC算术；不新增采样。"""

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
import experiment_cross_architecture_output_volume as m


PACKAGE = Path(r"E:\Downloads\results_cross_architecture_output_volume_compact_v1_package.zip")
LOCAL = Path("tmp/results_cross_architecture_output_volume_compact_batch128_feasibility_v1/run_summary.json")
OUTPUT = Path("tmp/cross_architecture_output_volume_5090_benchmark_analysis_20260912")


def main():
    import hashlib
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    package_hash = hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    with zipfile.ZipFile(PACKAGE) as archive:
        assert archive.testzip() is None, "ZIP CRC校验失败"
        roots = {name.split("/")[0] for name in archive.namelist()}
        assert len(roots) == 1
        root = next(iter(roots)) + "/"

        def read_json(name):
            return json.loads(archive.read(root + name))

        protocol = read_json("protocol.json")
        summary = read_json("run_summary.json")
        curves = read_json("curves.json")
        invocations = [json.loads(archive.read(name)) for name in archive.namelist() if "/invocation_" in name]
        assert summary["mode"] == "benchmark"
        assert not summary["errors"] and not summary["stopped"]
        assert curves["points"] == [] and curves["slopes"] == []
        assert not any("/jobs/" in name for name in archive.namelist())
        for key, value in protocol["signature"]["config"].items():
            setattr(m.Config, key, tuple(value) if isinstance(value, list) else value)

        data_checks = []
        for rule in protocol["configuration"]["RULES"]:
            expected = m.dataset(m.Config.POINTS, rule, "cpu")
            with np.load(io.BytesIO(archive.read(root + f"data_rule{rule}.npz")), allow_pickle=False) as stored:
                for name, array in stored.items():
                    ref = expected[name].numpy() if isinstance(expected[name], torch.Tensor) else expected[name]
                    np.testing.assert_array_equal(array, ref, err_msg=f"数据{rule}/{name}")
            data_checks.append(dict(rule=rule, input_ids=expected["input_ids"].tolist(), all_arrays_equal=True))

        local_by_name = {row["architecture"]: row for row in json.loads(LOCAL.read_text(encoding="utf-8"))["benchmarks"]}
        rows, checks = [], []
        data = m.dataset(m.Config.POINTS, protocol["configuration"]["RULES"][0], "cpu")
        data64 = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in data.items()}
        for item in summary["benchmarks"]:
            name = item["architecture"]
            assert item == read_json(f"models/{name}/benchmark.json")
            manifest = read_json(f"models/{name}/parameter_prior.json")
            model = m.make_model(name).double().eval()
            evaluator = m.Evaluator(name, model, data64, "cells_mean_bce")
            assert manifest == evaluator.metadata, f"参数名/形状/先验不一致: {name}"
            assert item["parameters"] == evaluator.dimensions
            with np.load(io.BytesIO(archive.read(root + f"models/{name}/prior_probe.npz")), allow_pickle=False) as saved:
                theta = saved["theta"].copy()
                observed = saved["raw_outputs"].copy()
                assert theta.shape == (1, evaluator.dimensions)
                with torch.no_grad():
                    predicted = evaluator.raw(torch.tensor(theta[0], dtype=torch.float64)).numpy()
                error = float(np.abs(predicted - observed).max())
                np.testing.assert_allclose(predicted, observed, rtol=2e-4, atol=2e-5, err_msg=name)
                # 中心格点的硬输出可额外核对，但不把初始快照视为SMC终点。
                z_ref = m.cell_outputs(torch.tensor(observed)).numpy()
                z_new = m.cell_outputs(torch.tensor(predicted)).numpy()
                sign_changes = int(((z_ref >= 0) != (z_new >= 0)).sum())
                assert np.isfinite(saved["scores"]).all()
            ks = 0.
            previous = float("inf")
            for level in item["full_smc_levels"]:
                survival = level["survival"]
                assert 0 < survival <= 1
                assert abs(survival * m.Config.PARTICLES - round(survival * m.Config.PARTICLES)) < 1e-8
                ks -= math.log2(survival)
                assert abs(ks - level["K_bits"]) < 1e-10
                assert level["cutoff"] <= previous
                assert all(0 <= v <= 1 for v in level["acceptance"])
                previous = level["cutoff"]
            times = [level["wall_seconds"] for level in item["full_smc_levels"]]
            seconds = float(np.median(times))
            assert abs(seconds - item["full_smc_level_seconds_median"]) < 1e-12
            local_seconds = local_by_name[name]["full_smc_level_seconds_median"]
            rows.append(dict(architecture=name, parameters=item["parameters"],
                             level_seconds_5090=seconds, level_seconds_local_3080=local_seconds,
                             measured_environment_speed_ratio=local_seconds / seconds,
                             minutes_per_curve_4rep_1000levels=seconds * 4000 / 60,
                             peak_allocated_gb=item["full_smc_peak_allocated_gb"],
                             peak_reserved_gb=item["full_smc_peak_reserved_gb"],
                             first_level_seconds=times[0], last_two_median_seconds=float(np.median(times[1:]))))
            checks.append(dict(architecture=name, parameter_manifest_equal=True, raw_output_shape=observed.shape,
                               cpu_float64_max_abs_error=error, cell_sign_disagreements=sign_changes,
                               smc_levels_checked=len(times), smc_mass_arithmetic_equal=True))

        total_seconds = sum(row["level_seconds_5090"] for row in rows)
        image_seconds = sum(row["level_seconds_5090"] for row in rows if row["architecture"] in m.IMAGE_MODELS)
        selected_seconds = sum(row["level_seconds_5090"] for row in rows
                               if row["architecture"] in ("mlp", "transformer", "swin_unet"))
        estimates = dict(
            all10_2rules_2bce_4rep_1000levels_hours=total_seconds * 16000 / 3600,
            all10_plus_3image_native_mse_4rep_1000levels_hours=(total_seconds * 16000 + image_seconds * 8000) / 3600,
            focused3_2rules_2bce_4rep_1000levels_minutes=selected_seconds * 16000 / 60,
            default_46_curves_4rep_60s_nominal_budget_hours=46 * 4 * 60 / 3600,
            note="1000层是预算单位，不是已知到指定loss所需层数。其它约束沿用平均BCE层速仅作估算；不含额外IO/混合预算。")
        report = dict(package=str(PACKAGE), sha256=package_hash, package_bytes=PACKAGE.stat().st_size,
                      protocol=protocol, invocations=invocations, timing=rows, estimates=estimates,
                      audit=checks, datasets=data_checks,
                      caveats=["只测速rule204的cells_mean_bce，16个输入，不是全规则或真实完整图像MSE采样。",
                               "只有三个浅层/模型，没有独立副本的完整曲线，不能验证深尾收缩或混合。",
                               "两台机器还使用不同操作系统及torch/torchvision版本，速度比不只是GPU硬件比。",
                               "无源码副本；跨Python版本字节码签名不可直接比较。本轮以参数清单和数值复算核对。",
                               "PRIOR_SAMPLES=EVAL_PARTICLES=128，此版prior_rms_mean只抽查一个网络，不能用它比较架构先验尺度。"])
        m.atomic_json(OUTPUT / "analysis.json", report)
        m.write_csv(OUTPUT / "timing.csv", rows)
        m.write_csv(OUTPUT / "audit.csv", checks)
        print(json.dumps(m.ready(dict(estimates=estimates, checks=checks, timing=rows)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
