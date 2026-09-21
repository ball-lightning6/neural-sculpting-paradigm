"""使用原协议剩余时间做固定核终点更新；不覆盖首轮SMC。"""

import json
import time

import numpy as np
import torch

import experiment as exp


def main():
    cfg = exp.CONFIG
    out, device = exp.setup(cfg)
    data = exp.prepare_data(cfg, out)
    x, y = data["train_x"].to(device), data["train_y"].to(device)
    for replica in range(cfg["smc_replicas"]):
        folder = out / "smc" / f"replica_{replica}"
        if (folder / "endpoint_check.json").exists():
            continue
        final = json.loads((folder / "final.json").read_text(encoding="utf-8"))
        if final["status"] != "target_reached":
            continue
        allowance = max(0., cfg["smc_seconds"] - final["sampling_seconds"])
        state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        theta, loss, ancestry = state["theta"].to(device), state["loss"].to(device), state["ancestry"].to(device)
        generator = torch.Generator(device=device)
        generator.set_state(state["rng"])
        steps, upper = state["scales"], state["upper"]
        records, acceptances = [], []
        tick = time.perf_counter()
        completed = 0
        with torch.no_grad():
            for sweep in range(1, 513):
                if time.perf_counter() - tick >= allowance:
                    break
                for block in range(6):
                    index = exp.LAYERS[block][0] if block < 5 else exp.WEIGHT_INDICES
                    proposal = theta.clone()
                    values = theta[:, index]
                    proposal[:, index] = exp.reflect(values + steps[block] * torch.randn(values.shape, generator=generator, device=device), cfg["smc_scale"])
                    candidate_loss, hard = exp.network_energy(proposal, x, y, cfg)
                    accepted = hard & torch.isfinite(candidate_loss) & (candidate_loss <= upper)
                    theta[accepted] = proposal[accepted]
                    loss[accepted] = candidate_loss[accepted]
                    acceptances.append(float(accepted.double().mean()))
                completed = sweep
                if sweep in [32, 128, 512]:
                    records.append(exp.save_smc_node(folder, f"check_{sweep:04d}", theta, loss, upper, state["k"], state["level"], ancestry,
                                                     cfg, data, device, {"extra_sweeps": sweep, "extra_seconds": time.perf_counter()-tick}))
                    exp.log(f"终点检查 r{replica} 固定核{sweep}轮，{time.perf_counter()-tick:.1f}s", out)
            if not records or records[-1]["extra_sweeps"] != completed:
                records.append(exp.save_smc_node(folder, "check_endpoint", theta, loss, upper, state["k"], state["level"], ancestry,
                                                 cfg, data, device, {"extra_sweeps": completed, "extra_seconds": time.perf_counter()-tick}))
        result = {"allowance_seconds": allowance, "extra_seconds": time.perf_counter()-tick,
                  "max_sweeps": 512, "completed_sweeps": completed, "frozen_steps": steps,
                  "mean_acceptance": float(np.mean(acceptances)), "nodes": records,
                  "policy": "原120秒SMC预算未用完部分；固定512轮上限；不依据测试表现停止。"}
        exp.atomic_json(folder / "endpoint_check.json", result)
        exp.atomic_torch(folder / "endpoint_check_weights.pt", theta.cpu())
        del theta, loss, ancestry
        torch.cuda.empty_cache()
    exp.progress(out, "终点稳定性检查完成", "固定提议核继续更新，使用原SMC预算剩余时间；等待统一离线复核。")


if __name__ == "__main__":
    main()
