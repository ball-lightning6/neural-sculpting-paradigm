"""固定标准化半径的角向可行质量诊断，自包含且不读取历史实验结果。

直接运行本文件，或调用 run({...})；不使用 argparse。默认 CUDA:0，4 个独立
replica，各 1024 个正式粒子，另有独立 pilot 选择中间门槛。输入为完整 4bit
0/1 真值表，BCE 为 16 项均值。theta = raw / sigma，sigma=1/sqrt(fan_in)，
bias 也采用相同的高斯标准化约定。[16] 为 97 维，[16,16] 为 369 维。

目标 A_f(r,eps)=P(L_f(r U)<=eps)，U 在 S^(d-1) 上均匀。提案
normalize(u+eta Z) 的转移密度只依赖 u·v，故相对于均匀球面可逆；再按硬
BCE 拒绝保持条件球面测度。eta 来自固定循环，不根据正式粒子在线调整。
独立 pilot 决定门槛，正式粒子只累计实际存活比例。有限粒子、有限更新仍有
谱系退化和非凸混合风险；通过自检不等于证明网络深尾已充分混合。

V_mu=integral A_f(r,eps) rho_mu(r) dr 中 rho 必须包含 r^(d-1) Jacobian。
本文件只给出完整径向密度与预定离散半径测度的混合，不对连续小体积插值。
未达到节点写 unreached/空估计，不把零存活、预算停止当成真实质量为零。
预算在完整 level 边界检查；每次调用全局 4h，每 replica 累计计算 30min。
全局预算或 Ctrl+C 可续；replica 的停滞/预算/零存活为保留的终止失败。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import signal
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F


PROTOCOL = "fixed_radius_angular_smc_independent_pilot_v2_cpu_kthvalue"
SOURCE_PROVENANCE = {
    "original_path": "research/smc_scale_suite_20261023/measure_diagnostic.py",
    "original_sha256": "5befe724b94e48a7d705cfd5dc45e1bc94ead41abe708da8db69707dc99af2f5",
    "original_protocol": "fixed_radius_angular_smc_independent_pilot_v1",
    "fix": "torch.kthvalue(pilot_losses.detach().cpu(), order); preserve input dtype",
    "determinism_disabled": False,
    "historical_checkpoints_accepted": False,
}
SCIENTIFIC_UNREACHED_REASONS = frozenset({
    "analytic_loss_lower_bound", "replica_time_budget", "level_budget",
    "stalled_levels_budget", "zero_evaluation_survivors",
})
CONFIG = {
    "device": "cuda:0",
    "dtype": "float64",
    "allow_cpu": False,  # 仅短测试显式打开；绝不自动回退 CPU。
    "architectures": [[16]],
    "radii": [10.0, 20.0, 40.0, 80.0],
    "targets": ["constant0", "projection", "majority3", "parity2", "parity4",
                "random_balanced_2d69"],
    "nodes": [2.0, 1.0, 0.7, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    "replicas": 4,
    "particles": 1024,
    "pilot_particles": 256,
    "survival": 0.5,
    "mutation_steps": 56,
    # 实际 eta=列表元素/sqrt(d)，正式与 pilot 使用独立随机流。
    "eta_multipliers": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    "micro_batch": 2048,
    "audit_particles": 16,
    "max_levels_per_replica": 2048,
    "max_stalled_levels": 32,
    "min_relative_progress": 1e-10,
    "global_seconds": 4 * 3600.0,
    "replica_seconds": 30 * 60.0,
    "seed": 2026102301,
    "results_dir": str(Path(__file__).resolve().parent / "results_measure_fixed"),
    "resume": True,
    # 只用于短烟测或主动分批，每次调用至多完成这些 level；None 不限制。
    "max_levels_per_invocation": None,
    # 权重必须预定、非负、和为 1；不由径向密度在稀疏节点上临时归一化。
    "discrete_radial_weights": {},
}
TARGET_MASKS = {
    "constant0": 0x0000,
    "projection": 0xFF00,  # 第一输入 bit，输入行按 0000 ... 1111。
    "majority3": 0xFCC0,
    "parity2": 0x0FF0,
    "parity4": 0x6996,
    "random_balanced_2d69": 0x2D69,  # 固定、恰有 8 个 1，不随运行重抽。
}
_OPERATIONAL_KEYS = {"results_dir", "resume", "global_seconds", "replica_seconds",
                     "max_levels_per_invocation"}


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def normalise_config(config=None):
    result = copy.deepcopy(CONFIG)
    if config is not None:
        unknown = set(config) - set(CONFIG)
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        result.update(copy.deepcopy(config))
    result["architectures"] = [list(item) for item in result["architectures"]]
    for arch in result["architectures"]:
        if not arch or any(type(width) is not int or width < 1 for width in arch):
            raise ValueError("architectures 必须为非空正整数宽度列表。")
    if not result["architectures"]:
        raise ValueError("至少指定一个架构。")
    if len({tuple(a) for a in result["architectures"]}) != len(result["architectures"]):
        raise ValueError("架构不得重复。")
    for field in ("radii", "nodes", "eta_multipliers"):
        result[field] = [float(item) for item in result[field]]
        if not result[field] or any(not math.isfinite(x) or x <= 0 for x in result[field]):
            raise ValueError(f"{field} 必须为非空有限正数列表。")
    if len(set(result["radii"])) != len(result["radii"]):
        raise ValueError("半径不得重复。")
    if any(a <= b for a, b in zip(result["nodes"], result["nodes"][1:])):
        raise ValueError("BCE 节点必须严格递减。")
    if not result["targets"] or len(set(result["targets"])) != len(result["targets"]):
        raise ValueError("规则列表不能为空或重复。")
    if any(name not in TARGET_MASKS for name in result["targets"]):
        raise ValueError("规则名不在固定面板中。")
    for key in ("replicas", "particles", "pilot_particles", "mutation_steps", "micro_batch",
                "audit_particles", "max_levels_per_replica", "max_stalled_levels"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} 必须为正整数。")
    if not 0 < result["survival"] < 1:
        raise ValueError("survival 必须在 (0,1)。")
    if not 0 <= result["min_relative_progress"] < 1:
        raise ValueError("min_relative_progress 必须在 [0,1)。")
    for key in ("global_seconds", "replica_seconds"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} 必须为有限正数。")
    cap = result["max_levels_per_invocation"]
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("max_levels_per_invocation 必须为 None 或正整数。")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed 必须为非负整数。")
    if result["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype 只支持 float32/float64，不使用 AMP。")
    for name, weights in result["discrete_radial_weights"].items():
        weights = [float(weight) for weight in weights]
        if (len(weights) != len(result["radii"]) or
                any(not math.isfinite(w) or w < 0 for w in weights) or
                not math.isclose(sum(weights), 1.0, abs_tol=1e-12, rel_tol=0)):
            raise ValueError(f"离散径向权重 {name} 必须与半径一一对应且和为 1。")
        result["discrete_radial_weights"][name] = weights
    result["results_dir"] = str(Path(result["results_dir"]).resolve())
    return result


def bit_inputs(device="cpu", dtype=torch.float64):
    rows = torch.arange(16, device=device)
    shifts = torch.tensor([3, 2, 1, 0], device=device)
    return ((rows[:, None] >> shifts) & 1).to(dtype)


def target_labels(name, device="cpu", dtype=torch.float64):
    mask = TARGET_MASKS[name]
    return torch.tensor([(mask >> row) & 1 for row in range(16)],
                        device=device, dtype=dtype)


def parameter_layout(hidden):
    sizes = [4, *hidden, 1]
    layout, cursor = [], 0
    for layer, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
        for kind, shape in (("weight", [fan_out, fan_in]), ("bias", [fan_out])):
            count = math.prod(shape)
            layout.append({"name": f"layers.{layer}.{kind}", "shape": shape,
                           "start": cursor, "stop": cursor + count,
                           "fan_in": fan_in, "sigma": 1 / math.sqrt(fan_in)})
            cursor += count
    return layout


def parameter_count(hidden):
    return parameter_layout(hidden)[-1]["stop"]


def sigma_vector(hidden, device="cpu", dtype=torch.float64):
    return torch.cat([torch.full((part["stop"] - part["start"],), part["sigma"],
                                 device=device, dtype=dtype)
                      for part in parameter_layout(hidden)])


def forward_logits(theta, x, hidden):
    """theta 为标准化坐标，权重和 bias 都乘各自 fan_in 的 sigma。"""
    if theta.ndim != 2 or theta.shape[1] != parameter_count(hidden):
        raise ValueError("参数形状不符合当前架构。")
    output = x.unsqueeze(0).expand(theta.shape[0], -1, -1)
    layout = parameter_layout(hidden)
    for index in range(0, len(layout), 2):
        weight, bias = layout[index:index + 2]
        w = theta[:, weight["start"]:weight["stop"]].reshape(
            theta.shape[0], *weight["shape"]) * weight["sigma"]
        b = theta[:, bias["start"]:bias["stop"]] * bias["sigma"]
        output = torch.bmm(output, w.transpose(1, 2)) + b[:, None, :]
        if index + 2 < len(layout):
            output = torch.tanh(output)
    return output.squeeze(-1)


class NetworkScore:
    def __init__(self, hidden, target, config):
        self.hidden = list(hidden)
        self.target = target
        self.x = bit_inputs(config["device"], getattr(torch, config["dtype"]))
        self.labels = target_labels(target, self.x.device, self.x.dtype)
        self.micro_batch = config["micro_batch"]

    @torch.no_grad()
    def __call__(self, theta):
        values = []
        for batch in theta.split(self.micro_batch):
            logits = forward_logits(batch, self.x, self.hidden)
            values.append(F.softplus((1 - 2 * self.labels) * logits).mean(-1))
        return torch.cat(values)


def unit_normalise(values):
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise RuntimeError("非有限方向或零向量，不能归一化。")
    return values / norms


def sample_sphere(count, dimension, device, dtype, rng):
    if dimension < 2:
        raise ValueError("球面诊断要求 d>=2。")
    return unit_normalise(torch.randn((count, dimension), device=device,
                                      dtype=dtype, generator=rng))


def sphere_proposal(u, eta, rng):
    """q(u,v)=q(v,u)：各向同性投影正态只依赖内积，不更新半径。"""
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta 必须为有限正数。")
    noise = torch.randn(u.shape, device=u.device, dtype=u.dtype, generator=rng)
    return unit_normalise(u + eta * noise)


def radius_diagnostics(u, radius):
    relative = torch.linalg.vector_norm((u * radius).double(), dim=-1) / radius - 1
    return {"standardized_radius": radius,
            "max_abs_relative_radius_error": float(relative.abs().max()),
            "rms_relative_radius_error": float(relative.square().mean().sqrt())}


def genealogy(lineage):
    _, counts = torch.unique(lineage, return_counts=True)
    weights = counts.double() / lineage.numel()
    return {"unique_initial_ancestors": len(counts),
            "initial_ancestor_ess": float(1 / weights.square().sum()),
            "largest_family_fraction": float(weights.max())}


@torch.no_grad()
def mutate(u, losses, threshold, radius, score, config, rng):
    initial = u
    moved = torch.zeros(len(u), device=u.device, dtype=torch.bool)
    accepted_total, jump_total, proposal_error = 0, 0.0, 0.0
    per_eta = {}
    for step in range(config["mutation_steps"]):
        multiplier = config["eta_multipliers"][step % len(config["eta_multipliers"])]
        eta = multiplier / math.sqrt(u.shape[-1])
        proposed = sphere_proposal(u, eta, rng)
        diagnostics = radius_diagnostics(proposed, radius)
        proposal_error = max(proposal_error, diagnostics["max_abs_relative_radius_error"])
        proposed_loss = score(proposed * radius)
        if not bool(torch.isfinite(proposed_loss).all()):
            raise RuntimeError("提案 BCE 出现非有限值。")
        accepted = proposed_loss <= threshold
        count = int(accepted.sum())
        chord = (proposed.double() - u.double()).square().sum(-1)
        jump_total += float(chord[accepted].sum())
        accepted_total += count
        moved |= accepted
        u = torch.where(accepted[:, None], proposed, u)
        losses = torch.where(accepted, proposed_loss, losses)
        record = per_eta.setdefault(str(multiplier), {"eta": eta, "accepted": 0, "proposed": 0})
        record["accepted"] += count
        record["proposed"] += len(u)
    if not bool((losses <= threshold).all()):
        raise RuntimeError("更新后违反当前硬门槛。")
    tolerance = 2e-6 if u.dtype == torch.float32 else 1e-12
    if proposal_error > tolerance:
        raise RuntimeError("提案违反固定标准化半径。")
    proposals = config["mutation_steps"] * len(u)
    return u, losses, {
        "acceptance": accepted_total / proposals,
        "accepted_squared_chord_per_proposal": jump_total / proposals,
        "fraction_ever_accepted": float(moved.double().mean()),
        "mean_start_end_cosine": float((initial.double() * u.double()).sum(-1).mean()),
        "per_eta": per_eta,
        "max_proposal_relative_radius_error": proposal_error,
    }


def make_rngs(device, seed):
    streams = ("initial", "pilot_initial", "resample", "pilot_resample", "mutation", "pilot_mutation")
    return {name: torch.Generator(device=device).manual_seed((seed + 100003 * index) % (2**63 - 1))
            for index, name in enumerate(streams)}


@torch.no_grad()
def initial_state(dimension, radius, score, config, seed):
    rngs = make_rngs(config["device"], seed)
    dtype = getattr(torch, config["dtype"])
    u = sample_sphere(config["particles"], dimension, config["device"], dtype, rngs["initial"])
    pilot = sample_sphere(config["pilot_particles"], dimension, config["device"], dtype,
                          rngs["pilot_initial"])
    losses, pilot_losses = score(u * radius), score(pilot * radius)
    if not bool(torch.isfinite(losses).all()) or not bool(torch.isfinite(pilot_losses).all()):
        raise RuntimeError("初始化 BCE 非有限。")
    nodes = [{"epsilon": node, "status": "unreached", "reason": "not_yet_attempted",
              "log2_angular_mass": None, "initial_direct_hits": int((losses <= node).sum()),
              "initial_direct_samples": len(losses)} for node in config["nodes"]]
    return {"u": u, "losses": losses, "pilot_u": pilot, "pilot_losses": pilot_losses,
            "lineage": torch.arange(len(u), device=u.device), "threshold": None,
            "log2_mass": 0.0, "level": 0, "target_index": 0, "stalled_levels": 0,
            "phase": "partial", "reason": None, "elapsed_seconds": 0.0,
            "nodes": nodes, "levels": [], "audits": [], "seed": seed}, rngs


def mark_unreached(state, reason):
    state = dict(state)
    state["nodes"] = copy.deepcopy(state["nodes"])
    for node in state["nodes"][state["target_index"]:]:
        node.update(status="unreached", reason=reason, log2_angular_mass=None, K_angular_bits=None,
                    last_completed_support_threshold=state["threshold"],
                    last_completed_support_log2_mass_estimate=state["log2_mass"])
    state["phase"], state["reason"] = "failed", reason
    return state


def analytic_bce_lower_bound(hidden, target, radius):
    """单隐藏层tanh的配对Lipschitz下界，仅用于可严格识别的逐bit反转规则。"""
    if hidden is None or len(hidden) != 1 or target not in TARGET_MASKS:
        return 0.0
    labels = target_labels(target).tolist()
    if not any(all(labels[i] != labels[i ^ bit] for i in range(16)) for bit in (1, 2, 4, 8)):
        return 0.0
    # 两层物理权重的范数积 <= r^2/(4 sqrt(width))；配对平均margin再除以2。
    paired_margin_bound = radius ** 2 / (8 * math.sqrt(hidden[0]))
    return math.log1p(math.exp(-paired_margin_bound))


def cpu_pilot_kthvalue(pilot_losses, order):
    """只迁移门槛选择到 CPU，保持 dtype、k 值及所有随机流不变。"""
    return torch.kthvalue(pilot_losses.detach().cpu(), order).values


@torch.no_grad()
def smc_level(state, radius, score, config, rngs):
    """完整的一层事务；pilot 门槛不读取正式群体，零存活不输出零质量。"""
    if state["phase"] != "partial":
        raise ValueError("只能推进未终止的 replica。")
    next_state = dict(state)
    next_state["nodes"] = copy.deepcopy(state["nodes"])
    next_state["levels"] = list(state["levels"])
    next_state["audits"] = list(state["audits"])
    target = config["nodes"][state["target_index"]]
    bound = analytic_bce_lower_bound(getattr(score, "hidden", None), getattr(score, "target", None), radius)
    if target < bound - 1e-12:
        next_state["analytic_bce_lower_bound"] = bound
        return mark_unreached(next_state, "analytic_loss_lower_bound")
    order = max(1, math.ceil(config["survival"] * len(state["pilot_losses"])))
    threshold = max(target, float(cpu_pilot_kthvalue(state["pilot_losses"], order)))
    if state["threshold"] is not None:
        threshold = min(threshold, state["threshold"])
    keep = state["losses"] <= threshold
    pilot_keep = state["pilot_losses"] <= threshold
    hits = int(keep.sum())
    target_hits = int((state["losses"] <= target).sum())
    record = {"level": state["level"] + 1, "requested_epsilon": target,
              "proposed_threshold": threshold, "previous_support_threshold": state["threshold"],
              "survivors": hits, "particles": len(keep), "survival_fraction": hits / len(keep),
              "pilot_survivors": int(pilot_keep.sum()),
              "lookahead_hits_at_requested_node": target_hits,
              "lookahead_log2_mass_estimate": (state["log2_mass"] + math.log2(target_hits / len(keep))
                                                if target_hits else None),
              "lookahead_is_not_official_node_estimate": True}
    next_state["level"] += 1
    if hits == 0:
        record["outcome"] = "zero_survivors_unreached_not_zero_mass"
        next_state["levels"].append(record)
        return mark_unreached(next_state, "zero_evaluation_survivors")
    if not bool(pilot_keep.any()):
        raise RuntimeError("pilot 分位门槛意外零存活。")
    parents = torch.multinomial(keep.double(), len(keep), replacement=True, generator=rngs["resample"])
    pilot_parents = torch.multinomial(pilot_keep.double(), len(pilot_keep), replacement=True,
                                      generator=rngs["pilot_resample"])
    u, losses, mix = mutate(state["u"][parents], state["losses"][parents], threshold,
                            radius, score, config, rngs["mutation"])
    pilot, pilot_losses, pilot_mix = mutate(state["pilot_u"][pilot_parents],
                                            state["pilot_losses"][pilot_parents], threshold,
                                            radius, score, config, rngs["pilot_mutation"])
    next_state.update(u=u, losses=losses, pilot_u=pilot, pilot_losses=pilot_losses,
                      lineage=state["lineage"][parents], threshold=threshold,
                      log2_mass=state["log2_mass"] + math.log2(hits / len(keep)))
    old_threshold = state["threshold"]
    stalled = (old_threshold is not None and
               old_threshold - threshold <= config["min_relative_progress"] * max(1, abs(old_threshold)))
    next_state["stalled_levels"] = state["stalled_levels"] + 1 if stalled else 0
    record.update(outcome="complete", log2_angular_mass_at_support=next_state["log2_mass"],
                  mix=mix, pilot_mix=pilot_mix, genealogy=genealogy(next_state["lineage"]),
                  distinct_immediate_parents=int(torch.unique(parents).numel()),
                  radius=radius_diagnostics(u, radius),
                  loss_min=float(losses.min()), loss_max=float(losses.max()),
                  loss_mean=float(losses.double().mean()))
    next_state["levels"].append(record)
    if threshold == target:
        node = next_state["nodes"][state["target_index"]]
        node.update(status="reached", reason=None, log2_angular_mass=next_state["log2_mass"],
                    K_angular_bits=-next_state["log2_mass"], level=next_state["level"],
                    radius=record["radius"], mix=mix, genealogy=record["genealogy"])
        next_state["target_index"] += 1
        next_state["stalled_levels"] = 0
        if next_state["target_index"] == len(config["nodes"]):
            next_state["phase"] = "completed"
    if next_state["phase"] == "partial":
        if next_state["stalled_levels"] >= config["max_stalled_levels"]:
            next_state = mark_unreached(next_state, "stalled_levels_budget")
        elif next_state["level"] >= config["max_levels_per_replica"]:
            next_state = mark_unreached(next_state, "level_budget")
    return next_state


@torch.no_grad()
def add_audit(state, radius, score, config, reason):
    """保存标准化/实际权重、预测和 CPU float64 独立重算，不消费任何随机流。"""
    count = min(config["audit_particles"], len(state["u"]))
    index = torch.linspace(0, len(state["u"]) - 1, count, device=state["u"].device).long()
    theta = (state["u"][index] * radius).detach().cpu().double()
    x = bit_inputs()
    labels = target_labels(score.target)
    logits = forward_logits(theta, x, score.hidden)
    losses = F.softplus((1 - 2 * labels) * logits).mean(-1)
    cached = state["losses"][index].detach().cpu().double()
    error = float((losses - cached).abs().max())
    tolerance = 2e-5 if config["dtype"] == "float32" else 1e-10
    warnings = []
    if error > tolerance:
        warnings.append("CPU_FLOAT64_LOSS_MISMATCH")
    if state["threshold"] is not None and float(losses.max()) > state["threshold"] + tolerance:
        warnings.append("CPU_FLOAT64_THRESHOLD_VIOLATION")
    audit = {"level": state["level"], "reason": reason, "support_threshold": state["threshold"],
             "requested_nodes": list(config["nodes"]), "indices": index.detach().cpu(),
             "theta_standardized": theta, "raw_parameters": theta * sigma_vector(score.hidden),
             "parameter_layout": parameter_layout(score.hidden), "labels": labels,
             "logits_float64": logits, "probabilities_float64": logits.sigmoid(),
             "predicted_bits": (logits > 0).to(torch.int64), "BCE_float64": losses,
             "BCE_cached": cached, "max_abs_loss_error": error,
             "feasible_at_requested_nodes": losses[:, None] <= torch.tensor(config["nodes"], dtype=torch.float64),
             "initial_ancestor": state["lineage"][index].detach().cpu(),
             "particle_normalized_weight": 1 / len(state["u"]),
             "particle_log2_unnormalized_mass_weight": state["log2_mass"] - math.log2(len(state["u"])),
             "audit_subset_is_not_reweighted_to_sum_one": True,
             "radius": radius_diagnostics(state["u"], radius), "warnings": warnings}
    state["audits"] = [*state["audits"], audit]
    audit_index = len(state["audits"]) - 1
    for node in state["nodes"]:
        if node.get("level") == state["level"] or (state["phase"] == "failed" and node["status"] == "unreached"):
            node["audit_index"] = audit_index
            node["audit_warnings"] = warnings
    if warnings:
        # 本层刚到达的节点若审计失败，不能继续作为有效质量参与合并。
        invalid = [index for index, node in enumerate(state["nodes"])
                   if node.get("level") == state["level"]]
        if invalid:
            state["target_index"] = min(state["target_index"], min(invalid))
        state = mark_unreached(state, "numeric_audit_failure")
        state["audit_failure"] = warnings
    return state


def capture_rng(rngs):
    return {"streams": {key: value.get_state().clone() for key, value in rngs.items()},
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda_all": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                               if torch.cuda.is_initialized() else [])}


def rng_hash(snapshot):
    digest = hashlib.sha256(repr(snapshot["python"]).encode("ascii"))
    numpy_state = snapshot["numpy"]
    digest.update(repr((numpy_state[0], *numpy_state[2:])).encode("ascii"))
    digest.update(numpy_state[1].tobytes())
    for key in sorted(snapshot["streams"]):
        digest.update(key.encode("ascii"))
        digest.update(snapshot["streams"][key].cpu().numpy().tobytes())
    for tensor in [snapshot["torch_cpu"], *snapshot["torch_cuda_all"]]:
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def restore_rng(snapshot, device):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if snapshot["torch_cuda_all"]:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda_all"])
    rngs = make_rngs(device, 0)
    if set(rngs) != set(snapshot["streams"]):
        raise RuntimeError("断点 RNG 流集合不匹配。")
    for key, value in rngs.items():
        value.set_state(snapshot["streams"][key])
    return rngs


def state_summary(state, job):
    return {"job": job, "phase": state["phase"], "reason": state["reason"],
            "level": state["level"], "completed_nodes": state["target_index"],
            "elapsed_seconds": state["elapsed_seconds"], "nodes": state["nodes"],
            "last_support_threshold": state["threshold"],
            "last_support_log2_mass_estimate": state["log2_mass"],
            "genealogy": genealogy(state["lineage"]),
            "radius": radius_diagnostics(state["u"], job["radius"]),
            "error": state.get("error"), "audit_failure": state.get("audit_failure"),
            "analytic_bce_lower_bound": state.get("analytic_bce_lower_bound")}


def save_checkpoint(folder, state, rngs, signature, job):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = capture_rng(rngs)
    payload = {"protocol": PROTOCOL, "signature": signature,
               "boundary": "complete_level", "state": to_cpu(state), "rng": snapshot,
               "rng_sha256": rng_hash(snapshot), "job": job}
    atomic_torch(folder / "checkpoint.pt", payload)
    return write_checkpoint_projection(folder, state, signature, job, payload["rng_sha256"])


def write_checkpoint_projection(folder, state, signature, job, rng_sha256):
    """重建派生文件；保持权威checkpoint及其随机状态原封不动。"""
    summary = state_summary(state, job)
    summary.update(signature=signature, checkpoint_sha256=file_hash(folder / "checkpoint.pt"),
                   rng_sha256=rng_sha256)
    # checkpoint 是唯一权威状态；这些投影文件在恢复时会重新生成。
    atomic_json(folder / "summary.json", summary)
    atomic_json(folder / "levels.json", state["levels"])
    atomic_torch(folder / "node_audits.pt", state["audits"])
    return summary


def load_checkpoint(folder, signature, device, job=None):
    # 仅加载自己生成的可信本地断点；pickle 文件不应来自不可信第三方。
    payload = torch.load(Path(folder) / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (payload["protocol"] != PROTOCOL or payload["signature"] != signature or
            payload["boundary"] != "complete_level" or (job is not None and payload["job"] != job)):
        raise RuntimeError("断点科学配置、源码、环境或作业身份不匹配，拒绝续跑。")
    if rng_hash(payload["rng"]) != payload["rng_sha256"]:
        raise RuntimeError("断点 RNG 校验失败。")
    state = payload["state"]
    for key in ("u", "losses", "pilot_u", "pilot_losses", "lineage"):
        state[key] = state[key].to(device)
    rngs = restore_rng(payload["rng"], device)
    return state, rngs


def radial_log_density(radius, dimension, kind, scale=1.0, df=8.0, ball_radius=None):
    """完整半径密度的自然对数，含球面面积和 r^(d-1)，不是裸位置密度。"""
    if dimension < 2 or type(dimension) is not int or not math.isfinite(radius) or radius < 0:
        raise ValueError("要求整数 d>=2、有限 r>=0。")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须为有限正数。")
    if kind not in ("gaussian", "student_t", "uniform_ball"):
        raise ValueError("未知径向测度。")
    if kind == "student_t" and (not math.isfinite(df) or df <= 0):
        raise ValueError("df 必须为有限正数。")
    if kind == "uniform_ball" and (ball_radius is None or not math.isfinite(ball_radius) or ball_radius <= 0):
        raise ValueError("均匀球要求有限 ball_radius>0。")
    if radius == 0 or (kind == "uniform_ball" and radius > ball_radius):
        return -math.inf
    jacobian = (dimension - 1) * math.log(radius)
    if kind == "gaussian":
        return ((1 - dimension / 2) * math.log(2) - math.lgamma(dimension / 2)
                - dimension * math.log(scale) + jacobian - (radius / scale)**2 / 2)
    if kind == "student_t":
        return (math.log(2) + math.lgamma((dimension + df) / 2)
                - math.lgamma(dimension / 2) - math.lgamma(df / 2)
                - dimension / 2 * math.log(df) - dimension * math.log(scale) + jacobian
                - (dimension + df) / 2 * math.log1p((radius / scale)**2 / df))
    return math.log(dimension) + jacobian - dimension * math.log(ball_radius)


def discrete_log2_mass(log2_angular_masses, weights):
    """仅对 sum_j w_j delta(r-r_j) 严格；缺失正权重节点时不重归一化。"""
    if len(log2_angular_masses) != len(weights) or not weights:
        raise ValueError("质量与权重必须非空且等长。")
    if (any(not math.isfinite(w) or w < 0 for w in weights) or
            not math.isclose(sum(weights), 1.0, abs_tol=1e-12, rel_tol=0)):
        raise ValueError("离散权重必须非负、和为 1。")
    terms = []
    for mass, weight in zip(log2_angular_masses, weights):
        if not weight:
            continue
        if mass is None:
            return None
        if not math.isfinite(mass):
            raise ValueError("未达质量应用 None，不能用 -inf 假装已知为零。")
        terms.append(mass + math.log2(weight))
    maximum = max(terms)
    return maximum + math.log2(sum(2**(term - maximum) for term in terms))


def mean_log2_mass(values):
    if not values or any(value is None for value in values):
        return None
    return discrete_log2_mass(values, [1 / len(values)] * len(values))


def sphere_cap_mass(dimension, cutoff):
    """短解析自检用：圆周或二维球面 U_0>=cutoff 的确切质量。"""
    if dimension not in (2, 3):
        raise ValueError("解析校验仅实现 d=2,3，不进行数值特殊函数近似。")
    if cutoff <= -1:
        return 1.0
    if cutoff >= 1:
        return 0.0
    return math.acos(cutoff) / math.pi if dimension == 2 else (1 - cutoff) / 2


def environment_info(config):
    device = torch.device(config["device"])
    if device.type == "cpu":
        if not config["allow_cpu"]:
            raise RuntimeError("CPU 只供显式 allow_cpu=True 的短测试，默认不回退。")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请核对 GPU 解释器；未回退 CPU。")
        if device.index is None:
            raise ValueError("请显式指定 CUDA 编号，例如 cuda:0。")
        torch.cuda.set_device(device)
    else:
        raise ValueError("只支持明确的 CUDA 或短测试 CPU。")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "dtype": config["dtype"], "tf32": False, "amp": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_num_threads": torch.get_num_threads()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = {"name": properties.name, "total_memory": properties.total_memory,
                       "compute_capability": [properties.major, properties.minor],
                       "multiprocessors": properties.multi_processor_count,
                       "cudnn": torch.backends.cudnn.version()}
    return info


def make_jobs(config):
    jobs = []
    for hidden in config["architectures"]:
        arch = "4_" + "_".join(map(str, hidden)) + "_1"
        for radius in config["radii"]:
            for target in config["targets"]:
                for replica in range(config["replicas"]):
                    identity = {"architecture": hidden, "radius": radius,
                                "target": target, "replica": replica}
                    token = canonical_hash(identity)[:12]
                    jobs.append({**identity, "id": f"{arch}_{target}_{token}",
                                 "dimension": parameter_count(hidden),
                                 "target_bitmask_hex": f"0x{TARGET_MASKS[target]:04x}",
                                 "seed": int(canonical_hash([config["seed"], identity])[:15], 16)})
    return jobs


def _tail_differences(rows):
    for previous, current in zip(rows, rows[1:]):
        left, right = previous["log2_mass"], current["log2_mass"]
        if left is not None and right is not None:
            da = -math.log(math.expm1(current["epsilon"])) + math.log(math.expm1(previous["epsilon"]))
            current["finite_window_dK_da_from_previous"] = (left - right) / da


def aggregate_results(config, jobs, summaries):
    angular, discrete = [], []
    by_identity = {(tuple(j["architecture"]), j["target"], j["radius"], j["replica"]):
                   summaries.get(j["id"]) for j in jobs}
    for hidden in config["architectures"]:
        for target in config["targets"]:
            for radius in config["radii"]:
                rows = []
                for index, epsilon in enumerate(config["nodes"]):
                    values, reasons = [], []
                    for replica in range(config["replicas"]):
                        summary = by_identity[(tuple(hidden), target, radius, replica)]
                        node = summary["nodes"][index] if summary is not None else None
                        values.append(node["log2_angular_mass"] if node is not None else None)
                        reasons.append(node["reason"] if node is not None else "not_attempted")
                    rows.append({"epsilon": epsilon, "replica_log2_angular_masses": values,
                                 "replica_unreached_reasons": reasons,
                                 "log2_mass": mean_log2_mass(values),
                                 "all_replicas_reached": all(v is not None for v in values)})
                _tail_differences(rows)
                angular.append({"architecture": hidden, "dimension": parameter_count(hidden),
                                "target": target, "radius": radius, "nodes": rows})
            for name, weights in config["discrete_radial_weights"].items():
                rows = []
                for index, epsilon in enumerate(config["nodes"]):
                    values = []
                    for replica in range(config["replicas"]):
                        components = []
                        for radius in config["radii"]:
                            summary = by_identity[(tuple(hidden), target, radius, replica)]
                            components.append(summary["nodes"][index]["log2_angular_mass"]
                                              if summary is not None else None)
                        values.append(discrete_log2_mass(components, weights))
                    rows.append({"epsilon": epsilon, "replica_log2_masses": values,
                                 "log2_mass": mean_log2_mass(values)})
                _tail_differences(rows)
                discrete.append({"architecture": hidden, "target": target, "name": name,
                                 "radii": config["radii"], "weights": weights, "nodes": rows,
                                 "measure": "预定离散径向测度，不是连续积分近似"})
    return {"angular": angular, "discrete_radial": discrete,
            "no_cross_architecture_K_comparison": True,
            "incomplete_replicas_are_not_dropped": True,
            "replica_spread_is_not_a_mixing_guarantee_or_formal_CI": True}


def failure_summary(status):
    """科学未达保留为 failed；仅程序/数值故障导致入口非零退出。"""
    expected, errors = {}, {}
    for row in status["job_summaries"]:
        if row["phase"] != "failed":
            continue
        reason = row.get("reason") or "unknown_failure"
        counts = (expected if reason in SCIENTIFIC_UNREACHED_REASONS
                  and not row.get("error") and not row.get("audit_failure") else errors)
        counts[reason] = counts.get(reason, 0) + 1
    return {"jobs_scientific_unreached": sum(expected.values()),
            "jobs_errors": sum(errors.values()), "scientific_unreached_reasons": expected,
            "error_reasons": errors}


def report_outcome(status):
    summary = failure_summary(status)
    reached = sum(node["status"] == "reached" for row in status["job_summaries"] for node in row["nodes"])
    print(json.dumps({"status": status["status"], "stop_reason": status["stop_reason"],
                      "jobs_failed": status["jobs_failed"], "reached_nodes": reached,
                      **summary}, ensure_ascii=False), flush=True)
    if summary["jobs_errors"]:
        for row in status["job_summaries"]:
            if row.get("error"):
                print(row["error"], flush=True)
        raise RuntimeError(f"jobs_errors={summary['jobs_errors']} reason={summary['error_reasons']}；详见 STATUS.json。")
    return status


def write_status(root, manifest, config, jobs, summaries, session, stop_reason):
    terminal = sum(s.get("phase") in ("completed", "failed") for s in summaries.values())
    failures = sum(s.get("phase") == "failed" for s in summaries.values())
    status = ("completed_with_failures" if failures else "completed") if terminal == len(jobs) else "partial"
    timing = {"invocation_started_utc": session["started_utc"],
              "invocation_elapsed_seconds": time.monotonic() - session["started_monotonic"],
              "total_replica_charged_seconds": sum(s["elapsed_seconds"] for s in summaries.values()),
              "levels_this_invocation": session["levels"], "stop_reason": stop_reason}
    payload = {"status": status, "stop_reason": stop_reason, "jobs_total": len(jobs),
               "protocol": PROTOCOL, "source_provenance": SOURCE_PROVENANCE,
               "jobs_terminal": terminal, "jobs_failed": failures,
               "jobs_uninitialised": len(jobs) - len(summaries),
               "scientific_hash": manifest["scientific_hash"],
               "source_sha256": manifest["source_sha256"],
               "environment_hash": manifest["environment_hash"],
               "timing": timing, "timing_sha256": canonical_hash(timing),
               "job_summaries": list(summaries.values()),
               "budget_semantics": "全局含 I/O；replica 累计加载/计算/审计，不含本次序列化；完整 level 边界软预算"}
    payload["failure_summary"] = failure_summary(payload)
    atomic_json(root / "STATUS.json", payload)
    atomic_json(root / "angular_and_discrete_summary.json", aggregate_results(config, jobs, summaries))
    lines = ["# 固定半径角向诊断", "", f"- 状态：`{status}`；停止原因：`{stop_reason}`。",
             f"- 完成/终止作业：{terminal}/{len(jobs)}；失败：{failures}；未初始化：{len(jobs)-len(summaries)}。",
             f"- 本次完成 level：{session['levels']}；耗时：{timing['invocation_elapsed_seconds']:.2f}s。",
             f"- 科学配置 hash：`{manifest['scientific_hash']}`。",
             "- 只估计固定标准化半径的 A_f(r,eps)，不重复旧 Gaussian/球/t8 面板作为新发现。",
             "- unreached 不代表体积为零；失败 replica 不删除、不只平均成功者。",
             "- 密度公式包含径向 Jacobian；离散混合只对预定原子测度严格，不重构连续极小体积。",
             "- 核可逆不等于非凸条件下混合充分；acceptance、位移、谱系和审计全部保留。",
             "- 不跨架构硬比 K；节点间斜率只描述当前有限窗口。", "",
             "| 作业 | 状态 | level | 已达节点 | 累计秒 | 原因 |",
             "|---|---|---:|---:|---:|---|"]
    for job in jobs:
        summary = summaries.get(job["id"])
        if summary is None:
            lines.append(f"| {job['id']} | unreached | 0 | 0 | 0 | not_attempted |")
        else:
            lines.append(f"| {job['id']} | {summary['phase']} | {summary['level']} | "
                         f"{summary['completed_nodes']} | {summary['elapsed_seconds']:.2f} | {summary['reason']} |")
    temporary = root / "PROGRESS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, root / "PROGRESS.md")
    return payload


def _synchronise(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run(config=None):
    """执行/续跑，返回 STATUS 字典。Ctrl+C 只请求在完整 level 边界退出。"""
    config = normalise_config(config)
    environment = environment_info(config)  # CUDA 错误时不创建伪结果目录。
    source_sha256 = file_hash(__file__)
    science = {key: value for key, value in config.items() if key not in _OPERATIONAL_KEYS}
    signature = canonical_hash({"protocol": PROTOCOL, "config": science,
                                "source": source_sha256, "environment": environment})
    root = Path(config["results_dir"])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "MANIFEST.json"
    jobs = make_jobs(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL:
            raise RuntimeError("旧协议结果禁止复用，包括原 96 个失败终态；请使用全新 results_measure_fixed 目录。")
        if not config["resume"] or manifest["scientific_hash"] != signature:
            raise RuntimeError("已有输出与科学配置/源码/环境不匹配，或未允许 resume；请用新目录。")
    else:
        if any(root.iterdir()):
            raise RuntimeError("输出目录非空且无 MANIFEST，拒绝混入其他实验。")
        density_rows = []
        for hidden in config["architectures"]:
            dimension = parameter_count(hidden)
            for radius in config["radii"]:
                density_rows.append({"architecture": hidden, "dimension": dimension, "radius": radius,
                                     "radius_over_sqrt_d": radius / math.sqrt(dimension),
                                     "log_rho_standard_gaussian": radial_log_density(radius, dimension, "gaussian"),
                                     "log_rho_t8_scale1": radial_log_density(radius, dimension, "student_t"),
                                     "log_rho_uniform_ball_Rmax": radial_log_density(
                                         radius, dimension, "uniform_ball", ball_radius=max(config["radii"]))})
        manifest = {"protocol": PROTOCOL, "scientific_hash": signature, "source_sha256": source_sha256,
                    "source_provenance": SOURCE_PROVENANCE,
                    "environment_hash": canonical_hash(environment), "environment": environment,
                    "scientific_config": science, "jobs": jobs, "target_masks": TARGET_MASKS,
                    "coordinate_convention": "theta=raw/sigma, sigma=1/sqrt(fan_in)，所有 bias 同样标准化",
                    "parameter_layouts": [{"architecture": h, "layout": parameter_layout(h)}
                                          for h in config["architectures"]],
                    "radial_density_diagnostics_not_quadrature_weights": density_rows,
                    "student_t_convention": "球对称多元 t8，公共 chi-square 尺度；scale=1，坐标方差=8/6",
                    "pilot_schedule": "仅由独立 pilot 决定；正式粒子不用于阈值或 eta 调参",
                    "checkpoint_policy": "初始化或完整 level 边界；完整 RNG；checkpoint.pt 为权威状态"}
        atomic_json(manifest_path, manifest)
    lock = root / "RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "scientific_hash": signature}))
    except FileExistsError as error:
        raise RuntimeError("存在 RUNNING.lock，拒绝并发写入；强制结束后须先确认原进程退出。") from error
    stop = {"requested": False}
    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.getsignal(signal.SIGINT)

        def request_stop(_signum, _frame):
            stop["requested"] = True
            print("收到 Ctrl+C，将在完整 level 保存后退出。", flush=True)

        signal.signal(signal.SIGINT, request_stop)
    session = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "started_monotonic": time.monotonic(), "levels": 0}
    summaries = {}
    score_cache = {}
    stop_reason = "running"
    try:
        # 从权威断点重建投影，允许上次在 checkpoint 与 JSON 投影之间中断。
        for job in jobs:
            folder = root / job["id"]
            if (folder / "checkpoint.pt").exists():
                state, rngs = load_checkpoint(folder, signature, config["device"], job)
                summaries[job["id"]] = write_checkpoint_projection(
                    folder, state, signature, job, rng_hash(capture_rng(rngs)))
        write_status(root, manifest, config, jobs, summaries, session, stop_reason)
        while True:
            active = [job for job in jobs if summaries.get(job["id"], {}).get("phase", "partial") == "partial"]
            if not active:
                stop_reason = "all_jobs_terminal"
                break
            for job in active:
                if stop["requested"]:
                    stop_reason = "ctrl_c_complete_level_boundary"
                    break
                if time.monotonic() - session["started_monotonic"] >= config["global_seconds"]:
                    stop_reason = "global_budget_resumable"
                    break
                cap = config["max_levels_per_invocation"]
                if cap is not None and session["levels"] >= cap:
                    stop_reason = "invocation_level_cap_resumable"
                    break
                started = time.monotonic()
                cache_key = (tuple(job["architecture"]), job["target"])
                if cache_key not in score_cache:
                    score_cache[cache_key] = NetworkScore(job["architecture"], job["target"], config)
                score = score_cache[cache_key]
                folder = root / job["id"]
                if (folder / "checkpoint.pt").exists():
                    state, rngs = load_checkpoint(folder, signature, config["device"], job)
                else:
                    state, rngs = initial_state(job["dimension"], job["radius"], score, config, job["seed"])
                    _synchronise(config["device"])
                    state["elapsed_seconds"] += time.monotonic() - started
                    summaries[job["id"]] = save_checkpoint(folder, state, rngs, signature, job)
                    started = time.monotonic()
                previous = state
                if state["elapsed_seconds"] >= config["replica_seconds"]:
                    state = mark_unreached(state, "replica_time_budget")
                    state = add_audit(state, job["radius"], score, config, state["reason"])
                else:
                    try:
                        state = smc_level(state, job["radius"], score, config, rngs)
                        _synchronise(config["device"])
                        session["levels"] += 1
                        if state["target_index"] > previous["target_index"] or state["phase"] == "failed":
                            state = add_audit(state, job["radius"], score, config, "node_or_terminal")
                    except KeyboardInterrupt:
                        # 非信号处理器来源的中断丢弃未提交层，保留原始 RNG 断点。
                        stop["requested"] = True
                        stop_reason = "keyboard_interrupt_rollback_to_checkpoint"
                        break
                    except Exception:
                        # 出错层不提交其随机流，回滚到上个完整层后记录失败。
                        state, rngs = load_checkpoint(folder, signature, config["device"], job)
                        state = mark_unreached(state, "internal_error")
                        state["error"] = traceback.format_exc()
                        state = add_audit(state, job["radius"], score, config, "internal_error")
                _synchronise(config["device"])
                state["elapsed_seconds"] += time.monotonic() - started
                if state["phase"] == "partial" and state["elapsed_seconds"] >= config["replica_seconds"]:
                    state = mark_unreached(state, "replica_time_budget")
                    state = add_audit(state, job["radius"], score, config, state["reason"])
                summaries[job["id"]] = save_checkpoint(folder, state, rngs, signature, job)
                write_status(root, manifest, config, jobs, summaries, session, "running")
                print(f"{job['id']} level={state['level']} reached={state['target_index']}/{len(config['nodes'])} "
                      f"support={state['threshold']} phase={state['phase']} reason={state['reason']}", flush=True)
                if state.get("error"):
                    print(state["error"], flush=True)
            if stop_reason != "running":
                break
        return write_status(root, manifest, config, jobs, summaries, session, stop_reason)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
        lock.unlink(missing_ok=True)


def main():
    """独立入口；默认 4h，科学未达正常返回，程序/数值故障非零退出。"""
    return report_outcome(run())


if __name__ == "__main__":
    main()

