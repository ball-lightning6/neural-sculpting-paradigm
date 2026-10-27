"""均匀盒条件 SMC、独立拒绝采样与优化器系综。节点前后诊断不参与选样。"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as c


DEFAULTS = {
    "particles": 1024, "replicas": 4, "sweeps": 32, "refresh_sweeps": 96, "refresh_blocks": 4,
    "factors": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003], "nodes": [0.3, 0.1, 0.03, 0.01, 0.003],
    "model_chunk": 32, "input_chunk": 32, "mutation_chunk": 32, "audit_weights": 2,
    "max_levels": 20000, "max_stalls": 100, "max_pool_seconds": 21600.,
    "slice_seconds": 1800., "checkpoint_seconds": 120., "keep_final_checkpoint": False,
    "max_levels_this_call": None,
}


def label(value):
    return "hard_fit" if value is None else "bce_" + format(value, ".8g").replace(".", "p")


def reflect(value, bounds):
    safe = torch.where(bounds > 0, bounds, torch.ones_like(bounds))
    folded = safe - (torch.remainder(value + safe, 4 * safe) - 2 * safe).abs()
    return torch.where(bounds > 0, folded, torch.zeros_like(folded))


def sample_prior(count, model, generator, device):
    return (torch.rand((count, model.size), generator=generator, device=device) * 2 - 1) * model.bounds.to(device)


def logits(theta, x, model, cfg):
    return torch.cat([torch.cat([model.logits(theta[p:p + cfg["model_chunk"]], x[j:j + cfg["input_chunk"]])
                                 for j in range(0, len(x), cfg["input_chunk"])], 1)
                      for p in range(0, len(theta), cfg["model_chunk"])], 0)


@torch.no_grad()
def energies(theta, x, y, model, cfg):
    z = logits(theta, x, model, cfg).double()
    parameter_finite = torch.cat([torch.isfinite(theta[i:i + cfg["model_chunk"]]).all(1)
                                 for i in range(0, len(theta), cfg["model_chunk"])])
    valid = parameter_finite & torch.isfinite(z).all(1)
    margin = z * (2 * y - 1)
    loss = F.softplus(-margin).mean(1)
    hard = ((z > 0).long() == y).all(1)
    barrier = -margin.min(1).values
    barrier = torch.where(hard, barrier, barrier.clamp_min(1e-12))
    hard &= valid
    loss = torch.where(valid, loss, torch.full_like(loss, float("inf")))
    barrier = torch.where(valid, barrier, torch.full_like(barrier, float("inf")))
    return loss, hard, barrier


@torch.no_grad()
def predictions(theta, x, model, cfg):
    for i in range(0, len(theta), cfg["model_chunk"]):
        if not bool(torch.isfinite(theta[i:i + cfg["model_chunk"]]).all()):
            raise RuntimeError("预测参数含非有限值，拒绝提交")
    z = logits(theta, x, model, cfg)
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError("探针logits含非有限值，不能把NaN转换为类别0")
    return (z > 0).cpu().numpy().astype(np.uint8)


def model_identity(model):
    return {"spec": model.spec, "parameters": model.size, "bounds": c.array_hash(model.bounds),
            "blocks": [(name, index.start, index.stop) for name, index in model.blocks]}


def scientific_config(cfg):
    return {k: v for k, v in cfg.items() if k not in {
        "slice_seconds", "checkpoint_seconds", "keep_final_checkpoint", "max_levels_this_call"}}


def merge_stats(target, extra):
    for key, value in extra.items():
        row = target.setdefault(key, {"proposed": 0, "accepted": 0, "normalized_esjd_sum": 0.})
        for field in row:
            row[field] += value[field]


@torch.no_grad()
def mutate(state, data, model, cfg, generator, sweeps):
    theta = state["theta"]
    bounds = model.bounds.to(theta.device)
    counts = {}
    for _ in range(sweeps):
        for name, index in model.blocks:
            factor = cfg["factors"][int(torch.randint(len(cfg["factors"]), (), generator=generator, device=theta.device))]
            row = counts.setdefault(f"{name}@{factor:g}", {"proposed": 0, "accepted": 0, "normalized_esjd_sum": 0.})
            active = bounds[index] > 0
            for first in range(0, len(theta), cfg["mutation_chunk"]):
                end = min(first + cfg["mutation_chunk"], len(theta))
                proposal = theta[first:end].clone()
                noise = torch.randn(proposal[:, index].shape, generator=generator, device=theta.device)
                proposal[:, index] = reflect(proposal[:, index] + factor * bounds[index] * noise, bounds[index])
                pl, ph, pb = energies(proposal, data["x"], data["y"], model, cfg)
                good = (pb <= state["upper"]) if state["phase"] == "hard" else (ph & (pl <= state["upper"]))
                good &= torch.isfinite(pl) & torch.isfinite(pb)
                if active.any():
                    delta = (proposal[:, index][:, active] - theta[first:end, index][:, active]) / bounds[index][active]
                    row["normalized_esjd_sum"] += float(delta[good].square().sum() / int(active.sum()))
                row["proposed"] += len(good)
                row["accepted"] += int(good.sum())
                theta[first:end][good] = proposal[good]
                for key, values in (("loss", pl), ("hard", ph), ("barrier", pb)):
                    state[key][first:end][good] = values[good]
    return counts


@torch.no_grad()
def snapshot(state, data, model, cfg, generator, name):
    before = predictions(state["theta"], data["probe_x"], model, cfg)
    probabilities, flips, stages, moves = [before.mean(0)], [], [0], {}
    blocks = min(cfg["refresh_blocks"], cfg["refresh_sweeps"]) if cfg["refresh_sweeps"] else 0
    for i in range(blocks):
        count = cfg["refresh_sweeps"] // blocks + int(i < cfg["refresh_sweeps"] % blocks)
        merge_stats(moves, mutate(state, data, model, cfg, generator, count))
        after = predictions(state["theta"], data["probe_x"], model, cfg)
        probabilities.append(after.mean(0))
        flips.append(float((after != before).mean()))
        stages.append(stages[-1] + count)
        before = after
    finite = torch.isfinite(state["loss"]) & torch.isfinite(state["barrier"])
    valid = finite & state["hard"]
    if name != "hard_fit":
        valid &= state["loss"] <= state["upper"]
    if not bool(valid.all()):
        raise RuntimeError("节点约束自检失败，拒绝提交预测")
    record = {
        "label": name, "level": state["level"], "phase": state["phase"], "upper": state["upper"],
        "K_joint_bits": state["K"], "ancestors": int(state["ancestry"].unique().numel()),
        "loss_max": float(state["loss"].max()), "loss_mean": float(state["loss"].mean()),
        "refresh_sweeps": stages, "refresh_label_flip_fraction": flips, "refresh_move_stats": moves,
        "warning": "初始祖先不是ESS；接受率、位移、短刷新变化均不是全局收敛证明。",
    }
    arrays = {"prediction": before, "train_loss": state["loss"].cpu().numpy().copy(),
              "hard_fit": state["hard"].cpu().numpy().copy(), "ancestry": state["ancestry"].cpu().numpy().copy(),
              "refresh_probability": np.stack(probabilities),
              "audit_theta": state["theta"][:cfg["audit_weights"]].cpu().numpy().copy()}
    return {"record": record, "arrays": arrays}


def project_smc(folder, state):
    for name, node in state["nodes"].items():
        c.atomic_json(folder / (name + ".json"), node["record"])
        c.atomic_npz(folder / (name + ".npz"), **node["arrays"])
    # 断点是权威来源，重建日志可避免崩溃后的重复/半层记录。
    text = "".join(__import__("json").dumps(c.clean(row), ensure_ascii=False) + "\n" for row in state["history"])
    temp = folder / "levels.jsonl.tmp"
    temp.write_text(text, encoding="utf-8")
    c.atomic_replace(temp, folder / "levels.jsonl")
    c.atomic_json(folder / "progress.json", {
        "status": state["status"], "reason": state.get("reason"), "level": state["level"],
        "phase": state["phase"], "upper": state["upper"], "K_joint_bits": state["K"],
        "seconds": state["seconds"], "nodes": list(state["nodes"]),
    })


def save_smc(folder, state, generators):
    payload = {**state, "rng": {k: g.get_state().cpu() for k, g in generators.items()}}
    c.atomic_torch(folder / "checkpoint.pt", payload)
    project_smc(folder, state)


def validate_nodes(nodes):
    if not nodes or any(not math.isfinite(v) or v <= 0 for v in nodes):
        raise ValueError("节点不能为空，且须为正有限数")
    if nodes != sorted(set(nodes), reverse=True):
        raise ValueError("节点须严格降序且不能重复")


@torch.no_grad()
def validate_smc_state(state, data, model, cfg):
    theta = state["theta"]
    if theta.shape != (cfg["particles"], model.size):
        raise RuntimeError("断点参数形状不匹配")
    bounds = model.bounds.to(theta.device)
    for first in range(0, len(theta), cfg["model_chunk"]):
        chunk = theta[first:first + cfg["model_chunk"]]
        if not bool(torch.isfinite(chunk).all()) or not bool((chunk.abs() <= bounds).all()):
            raise RuntimeError("断点含非有限参数或参数越过先验盒")
    current = energies(theta, data["x"], data["y"], model, cfg)
    for key, actual in zip(("loss", "hard", "barrier"), current):
        cached = state[key]
        if cached.shape != actual.shape:
            raise RuntimeError("断点缓存形状不匹配")
        if key == "hard":
            valid = torch.equal(cached, actual)
        else:
            valid = bool(torch.isfinite(cached).all() and torch.isfinite(actual).all())
            valid = valid and torch.allclose(cached, actual, rtol=2e-5, atol=2e-6)
        if not valid:
            raise RuntimeError("断点缓存与重新前向不一致：" + key)
    if state["phase"] not in ("hard", "bce") or math.isnan(state["upper"]):
        raise RuntimeError("断点阶段/门槛非法")
    if state["phase"] == "hard":
        valid = state["barrier"] <= state["upper"]
    else:
        valid = state["hard"] & (state["loss"] <= state["upper"])
    if not bool(valid.all()):
        raise RuntimeError("断点粒子违反已提交阶段约束")
    if not math.isfinite(state["K"]) or state["K"] < 0 or not 0 <= state["node_index"] <= len(cfg["nodes"]):
        raise RuntimeError("断点K或节点序号非法")
    ancestry = state["ancestry"]
    if ancestry.shape != (len(theta),) or not bool(((ancestry >= 0) & (ancestry < len(theta))).all()):
        raise RuntimeError("断点祖先标识非法")


@torch.no_grad()
def run_smc(model, data, cfg, folder, key, env, deadline, provenance):
    cfg = {**DEFAULTS, **cfg}
    validate_nodes(cfg["nodes"])
    if cfg["particles"] < 2 or cfg["refresh_blocks"] < 1 or cfg["sweeps"] < 1:
        raise ValueError("粒子和更新配置非法")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"version": 1, "model": model_identity(model), "config": scientific_config(cfg), "key": key,
                "environment": env, "data": {k: c.array_hash(data[k]) for k in ("x", "y", "probe_x")},
                "provenance": provenance}
    c.guard_json(folder / "identity.json", identity)
    if (folder / "final.json").exists():
        return c.read_json(folder / "final.json")
    device = data["x"].device
    generators = {name: torch.Generator(device=device).manual_seed(c.seed(key, name))
                  for name in ("prior", "resample", "mutation")}
    if (folder / "checkpoint.pt").exists():
        state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        for name, value in state.pop("rng").items():
            generators[name].set_state(value)
        for name in ("theta", "loss", "hard", "barrier", "ancestry"):
            state[name] = state[name].to(device)
        validate_smc_state(state, data, model, cfg)
        project_smc(folder, state)
    else:
        theta = sample_prior(cfg["particles"], model, generators["prior"], device)
        loss, hard, barrier = energies(theta, data["x"], data["y"], model, cfg)
        if not bool((torch.isfinite(loss) & torch.isfinite(barrier)).all()):
            raise RuntimeError("先验含非有限输出，拒绝改变先验后继续")
        state = {"theta": theta, "loss": loss, "hard": hard, "barrier": barrier,
                 "ancestry": torch.arange(len(theta), device=device), "phase": "hard", "upper": float("inf"),
                 "level": 0, "K": 0., "node_index": 0, "nodes": {}, "stalls": 0,
                 "seconds": 0., "history": [], "status": "partial"}
        save_smc(folder, state, generators)
    start, last_save, last_log, initial_level = time.monotonic(), time.monotonic(), time.monotonic(), state["level"]
    previous_seconds = state["seconds"]
    stop_at = min(deadline, start + cfg["slice_seconds"])
    while state["node_index"] < len(cfg["nodes"]):
        if c.expired(stop_at):
            break
        if cfg["max_levels_this_call"] is not None and state["level"] - initial_level >= cfg["max_levels_this_call"]:
            break
        if state["level"] >= cfg["max_levels"] or state["stalls"] >= cfg["max_stalls"] or previous_seconds + time.monotonic() - start >= cfg["max_pool_seconds"]:
            state.update(status="unresolved", reason="level_stall_or_compute_budget")
            break
        phase = state["phase"]
        target = 0. if phase == "hard" else cfg["nodes"][state["node_index"]]
        values = state["barrier"] if phase == "hard" else state["loss"]
        upper = min(state["upper"], max(target, float(torch.quantile(values, .5))))
        mask = torch.isfinite(values) & (values <= upper)
        if phase == "bce" or upper == 0.:
            mask &= state["hard"]
        keep = torch.where(mask)[0]
        if not len(keep):
            state.update(status="unresolved", reason="no_survivors_not_zero_probability")
            break
        survival = len(keep) / len(values)
        state["K"] -= math.log2(survival)
        state["stalls"] = state["stalls"] + 1 if upper == state["upper"] else 0
        if len(keep) != len(values):
            selected = keep[torch.randint(len(keep), (len(values),), generator=generators["resample"], device=device)]
            for name in ("theta", "loss", "hard", "barrier", "ancestry"):
                state[name] = state[name][selected].clone()
        state["upper"], state["level"] = upper, state["level"] + 1
        moves = mutate(state, data, model, cfg, generators["mutation"], cfg["sweeps"])
        reached = upper == target
        if reached:
            name = label(None if phase == "hard" else target)
            state["nodes"][name] = snapshot(state, data, model, cfg, generators["mutation"], name)
            if phase == "hard":
                state["phase"], state["upper"] = "bce", float("inf")
            else:
                state["node_index"] += 1
        state["history"].append({"level": state["level"], "phase_completed": phase, "phase_next": state["phase"],
            "upper": upper, "survival": survival, "K_joint_bits": state["K"],
            "ancestors": int(state["ancestry"].unique().numel()), "move_stats": moves})
        state["seconds"] = previous_seconds + time.monotonic() - start
        if reached or time.monotonic() - last_save >= cfg["checkpoint_seconds"]:
            save_smc(folder, state, generators)
            last_save = time.monotonic()
        if reached or time.monotonic() - last_log >= 30:
            print(f"{folder.parent.name}/{folder.name} level={state['level']} {phase} <= {upper:.6g} K={state['K']:.3f} ancestors={int(state['ancestry'].unique().numel())}", flush=True)
            last_log = time.monotonic()
    if state["node_index"] == len(cfg["nodes"]):
        state.update(status="completed", reason="target_reached")
    state["seconds"] = previous_seconds + time.monotonic() - start
    save_smc(folder, state, generators)
    result = c.read_json(folder / "progress.json")
    if state["status"] != "partial":
        c.atomic_json(folder / "final.json", result)
        if state["status"] == "completed" and not cfg["keep_final_checkpoint"]:
            (folder / "checkpoint.pt").unlink(missing_ok=True)
    return result


@torch.no_grad()
def run_gc(model, data, cfg, folder, key, env, deadline, provenance):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    c.guard_json(folder / "identity.json", {"model": model_identity(model), "key": key, "environment": env,
        "data": {k: c.array_hash(data[k]) for k in ("x", "y", "probe_x")}, "config": cfg, "provenance": provenance})
    if (folder / "final.json").exists():
        return c.read_json(folder / "final.json")
    generator = torch.Generator(device=data["x"].device).manual_seed(c.seed(key, "direct_prior"))
    path = folder / "checkpoint.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        generator.set_state(state.pop("rng"))
    else:
        state = {"draws": 0, "hits": 0, "accepted": 0, "seconds": 0., "predictions": [], "losses": []}
    start = last_log = last_save = time.monotonic()
    while state["draws"] < cfg["max_draws"] and state["accepted"] < cfg["target"]:
        if c.expired(deadline) or state["seconds"] + time.monotonic() - start >= cfg["max_seconds"]:
            break
        count = min(cfg["model_chunk"], cfg["max_draws"] - state["draws"])
        theta = sample_prior(count, model, generator, data["x"].device)
        loss, hard, barrier = energies(theta, data["x"], data["y"], model, cfg)
        if not bool((torch.isfinite(loss) & torch.isfinite(barrier)).all()):
            raise RuntimeError("G&C 非有限值不能当作普通拒绝处理")
        state["draws"] += count
        state["hits"] += int(hard.sum())
        selected = torch.where(hard)[0][:cfg["target"] - state["accepted"]]
        if len(selected):
            state["predictions"].append(predictions(theta[selected], data["probe_x"], model, cfg))
            state["losses"].append(loss[selected].cpu().numpy())
            state["accepted"] += len(selected)
        if time.monotonic() - last_save >= 120:
            c.atomic_torch(path, {**state, "seconds": state["seconds"] + time.monotonic() - start,
                                 "rng": generator.get_state()})
            last_save = time.monotonic()
        if time.monotonic() - last_log >= 30:
            print(f"{folder.parent.name}/G&C draws={state['draws']} hits={state['hits']} accepted={state['accepted']}/{cfg['target']}", flush=True)
            last_log = time.monotonic()
    state["seconds"] += time.monotonic() - start
    c.atomic_torch(path, {**state, "rng": generator.get_state()})
    if state["predictions"]:
        c.atomic_npz(folder / "forecast.npz", prediction=np.concatenate(state["predictions"]), train_loss=np.concatenate(state["losses"]))
    done = state["draws"] >= cfg["max_draws"] or state["accepted"] >= cfg["target"] or state["seconds"] >= cfg["max_seconds"]
    result = {k: state[k] for k in ("draws", "hits", "accepted", "seconds")}
    result.update(status=("sufficient" if state["accepted"] >= cfg["target"] else "budget_limited") if done else "partial",
                  probability_estimate=state["hits"] / state["draws"] if state["draws"] else None,
                  note="零命中不等于零准确率；low-loss子集仅在实际保留足量时可比较。")
    c.atomic_json(folder / "progress.json", result)
    if done:
        c.atomic_json(folder / "final.json", result)
        path.unlink(missing_ok=True)
    return result


def project_training(folder, state, cfg):
    for name, rows in state["rows"].items():
        if rows:
            c.atomic_npz(folder / (name + ".npz"), init_id=np.array([r[0] for r in rows], np.int64),
                         step=np.array([r[1] for r in rows], np.int64),
                         train_loss=np.array([r[2] for r in rows]), prediction=np.stack([r[3] for r in rows]))
    progress = {"status": state["status"], "step": state["step"], "models": cfg["models"],
                "node_counts": state["seen"].sum(0).tolist(), "failed_initializations": int(state["failed"].sum()),
                "seconds": state["seconds"], "all_nodes_reached": bool(state["seen"].all()),
                "recording": "每个optimizer更新之后首次观测H且loss达标；含初始化step0。"}
    c.atomic_json(folder / "progress.json", progress)
    return progress


def save_training(folder, state, theta, optimizer, generator, cfg):
    c.atomic_torch(folder / "checkpoint.pt", {**state, "theta": theta, "optimizer": optimizer.state_dict(),
                                             "batch_rng": generator.get_state()})
    return project_training(folder, state, cfg)


def run_training(model, data, cfg, folder, key, env, deadline, provenance, require_frozen=None):
    validate_nodes(cfg["nodes"])
    if require_frozen is not None:
        c.verify_freeze(require_frozen)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scientific = {k: v for k, v in cfg.items() if k not in ("checkpoint_seconds", "max_steps_this_call")}
    c.guard_json(folder / "identity.json", {"model": model_identity(model), "config": scientific, "key": key,
        "frozen_prediction_sha256": c.file_hash(Path(require_frozen) / "forecast_frozen.json") if require_frozen is not None else None,
        "environment": env, "data": {k: c.array_hash(data[k]) for k in ("x", "y", "probe_x")}, "provenance": provenance})
    if (folder / "final.json").exists():
        return c.read_json(folder / "final.json")
    device = data["x"].device
    generator = torch.Generator().manual_seed(c.seed(key, "minibatch"))
    path = folder / "checkpoint.pt"
    saved_optimizer = None
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        theta = state.pop("theta").to(device).requires_grad_(True)
        saved_optimizer = state.pop("optimizer")
        generator.set_state(state.pop("batch_rng"))
    else:
        init = torch.Generator(device=device).manual_seed(c.seed(key, "initialization"))
        theta = sample_prior(cfg["models"], model, init, device).requires_grad_(True)
        state = {"step": 0, "seen": np.zeros((len(theta), len(cfg["nodes"])), bool),
                 "rows": {label(v): [] for v in cfg["nodes"]}, "failed": np.zeros(len(theta), bool),
                 "orders": None, "offset": 0, "observed_step": -1, "seconds": 0., "status": "partial"}
    if cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([theta], lr=cfg["lr"])
    elif cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam([theta], lr=cfg["lr"], weight_decay=0.)
    else:
        raise ValueError("只支持显式SGD/Adam，无weight decay")
    if saved_optimizer is not None:
        optimizer.load_state_dict(saved_optimizer)
    start, saved_at, log_at, initial_step = time.monotonic(), time.monotonic(), time.monotonic(), state["step"]
    previous_seconds = state["seconds"]
    n = len(data["y"])
    while True:
        if state["observed_step"] != state["step"]:
            loss, hard, _ = energies(theta.detach(), data["x"], data["y"], model, cfg)
            nonfinite = ~torch.isfinite(loss)
            state["failed"] |= nonfinite.cpu().numpy()
            for j, threshold in enumerate(cfg["nodes"]):
                eligible = (hard & torch.isfinite(loss) & (loss <= threshold)).cpu().numpy()
                ids = np.flatnonzero(eligible & ~state["seen"][:, j] & ~state["failed"])
                if len(ids):
                    pred = predictions(theta[torch.as_tensor(ids, device=device)].detach(), data["probe_x"], model, cfg)
                    for idx, row in zip(ids, pred):
                        state["rows"][label(threshold)].append((int(idx), state["step"], float(loss[idx]), row))
                    state["seen"][ids, j] = True
            state["observed_step"] = state["step"]
            # 失败初始化仍计入总分母；清零仅隔离NaN，不允许重新成为有效模型。
            invalid = torch.as_tensor(state["failed"], device=device)
            if invalid.any():
                with torch.no_grad():
                    theta[invalid] = 0
                for value in optimizer.state.get(theta, {}).values():
                    if isinstance(value, torch.Tensor) and value.shape == theta.shape:
                        value[invalid] = 0
        terminal = (state["seen"].all(1) | state["failed"]).all() or state["step"] >= cfg["max_steps"]
        if terminal:
            state["status"] = "completed" if state["seen"].all() else "completed_with_unreached"
            break
        if c.expired(deadline):
            break
        if cfg.get("max_steps_this_call") is not None and state["step"] - initial_step >= cfg["max_steps_this_call"]:
            break
        if state["orders"] is None or state["offset"] >= n:
            state["orders"] = torch.stack([torch.randperm(n, generator=generator) for _ in range(len(theta))])
            state["offset"] = 0
        ids = state["orders"][:, state["offset"]:state["offset"] + cfg["batch_size"]].to(device)
        state["offset"] += cfg["batch_size"]
        optimizer.zero_grad(set_to_none=True)
        for first in range(0, len(theta), cfg["model_chunk"]):
            end = min(first + cfg["model_chunk"], len(theta))
            z = model.logits(theta[first:end], data["x"]).double()
            chosen = ids[first:end]
            margins = z.gather(1, chosen) * (2 * data["y"][chosen] - 1)
            each = F.softplus(-margins).mean(1)
            active = torch.as_tensor(~state["failed"][first:end], device=device)
            each[active].sum().backward()
        with torch.no_grad():
            theta.grad[torch.as_tensor(state["failed"], device=device)] = 0
        optimizer.step()
        state["step"] += 1
        state["seconds"] = previous_seconds + time.monotonic() - start
        if time.monotonic() - saved_at >= cfg["checkpoint_seconds"]:
            save_training(folder, state, theta, optimizer, generator, cfg)
            saved_at = time.monotonic()
        if time.monotonic() - log_at >= 30:
            print(f"{folder.parent.name}/{folder.name} step={state['step']} reached={state['seen'].sum(0).tolist()}/{len(theta)}", flush=True)
            log_at = time.monotonic()
    state["seconds"] = previous_seconds + time.monotonic() - start
    result = save_training(folder, state, theta, optimizer, generator, cfg)
    if state["status"] != "partial":
        c.atomic_json(folder / "final.json", result)
        if not cfg.get("keep_final_checkpoint", False):
            path.unlink(missing_ok=True)
    return result
