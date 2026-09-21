"""模97除法：给 CE 加入正确方向 logit 差距的幅度对齐正则。

单文件，AutoDL 上直接 python 本文件，也可以粘入 Jupyter 单元格运行。
不需要下载数据或使用 argparse。
沿用已有成功记录的改版：194 -> 512 x 3 -> 97，GELU + LayerNorm。
这是原论文的模除法任务，但不是原论文的 Transformer 复现。

默认 90% 训练集、每组32个配对seed、20万步、无 WD/Dropout。
五组：原CE、CE+0.01R、CE+0.1R、CE+R、原始0/1 MSE，共160个独立模型。
R = Var_{训练样本i, 错误类别c}(z[i,y[i]] - z[i,c])，总体方差，不含正确类自身。
这是一种多分类推广：两类时退化为正确方向logit差距的跨样本方差。
正则同时限制样本间平均差距和同一样本不同错误类的差距，不固定共同幅度。
全部正确方向差距一起从1变成10时，R不变；正则不是weight decay或logit范数。
训练使用同一minibatch上的方差，评估另外计算整个训练集的精确方差。
MSE 对原始输出和0/1 one-hot求平均，不使用sigmoid/softmax，不放大标签。
各组逐步使用相同minibatch和初始权重。全部模型跑同样步数，不按测试结果早停。
测试标签只参与评估，不参与梯度、调参或训练停止。

结果放在 /root/autodl-tmp。Ctrl+C/SIGTERM 在当前完整更新后保存并暂停，
重新运行自动续训；增加 MAX_STEPS 可延长预算。不要强制关闭保存中的进程。
下载 *_package.zip 用于分析；大号 latest_checkpoint.pt 留在机器上用于续训。
CPU 仅允许 SMOKE_TEST=True 的工程自检；默认必须有支持当前显卡的 CUDA PyTorch。
"""

from __future__ import annotations

import csv
import hashlib
import json
import linecache
import math
import os
import platform
import random
import shutil
import signal
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Config:
    PRIME = 97
    TRAIN_FRACTION = 0.90
    DATA_SEED = 20260820
    MODEL_SEEDS = tuple(range(32))
    # 名称、目标类型、正则系数。三档沿用先前copy实验，不根据本轮测试结果选取。
    CONDITIONS = (("ce", "ce", 0.0),
                  ("ce_align_0p01", "ce_align", 0.01),
                  ("ce_align_0p1", "ce_align", 0.1),
                  ("ce_align_1", "ce_align", 1.0),
                  ("raw_mse_01", "raw_mse", 0.0))
    HIDDEN_SIZE = 512
    HIDDEN_LAYERS = 3
    LAYERNORM_EPS = 1e-5
    LEARNING_RATE = 1e-3
    ADAM_BETAS = (0.9, 0.999)  # 沿用已有改版脚本的实际 AdamW 默认值。
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 0.0
    WARMUP_STEPS = 10
    BATCH_SIZE = 512
    TRAIN_ORDER_SEED = 314159
    MAX_STEPS = 200_000

    DEVICE = "cuda:0"
    ALLOW_TF32 = True  # 与已有改版一致；最后额外用关闭 TF32 的前向核验。
    EVAL_BATCH_SIZE = 512
    EVAL_INTERVAL = 1000
    CHECKPOINT_INTERVAL = 5000
    LOG_INTERVAL = 1000
    SNAPSHOT_STEPS = (0, 12_500, 50_000, 200_000)
    # 上面这些节点保存所有输入上的 float32 输出及权重；其余评估保存完整类别映射。
    RESULT_DIR = Path("/root/autodl-tmp/results_mod97_ce_amplitude_alignment")
    RESUME = True
    PACKAGE_RESULTS = True

    # 只筛选值得人工看曲线的延迟泛化，不把自动标签当作机制证明。
    FIT_CONSECUTIVE_EVALS = 3
    GENERALIZATION_ACCURACY = 0.99
    PLATEAU_ACCURACY_MAX = 0.50
    MIN_PLATEAU_STEPS = 2000
    MATCHED_CE_LEVELS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.001, 1e-4, 1e-6, 1e-8)
    SMOKE_TEST = False


def config_dict() -> dict:
    cfg = {k.lower(): v for k, v in vars(Config).items() if k.isupper()}
    cfg["result_dir"] = str(cfg["result_dir"])
    if cfg["smoke_test"]:
        cfg.update(prime=11, model_seeds=(0, 1), hidden_size=32,
                   hidden_layers=2, batch_size=16, max_steps=12,
                   eval_batch_size=32, eval_interval=2,
                   checkpoint_interval=4, log_interval=4,
                   snapshot_steps=(0, 6, 12), min_plateau_steps=2)
        cfg["result_dir"] += "_smoke"
    return cfg


def signature(cfg: dict) -> dict:
    # 只允许延长预算、改变保存位置/频率和显示参数，不允许续跑中改科学配置。
    keys = ("prime", "train_fraction", "data_seed", "model_seeds", "hidden_size",
            "hidden_layers", "layernorm_eps", "learning_rate", "adam_betas",
            "adam_eps", "weight_decay", "warmup_steps", "batch_size",
            "train_order_seed", "allow_tf32", "smoke_test", "conditions")
    result = {k: cfg[k] for k in keys}
    result["protocol_version"] = 1
    result["experiment"] = "mod97_ce_pairwise_gap_variance"
    result["alignment"] = "population_variance_over_minibatch_and_all_incorrect_classes"
    result["batch_protocol"] = "shared_uniform_with_replacement"
    result["dtype"] = "float32"
    return json.loads(json.dumps(result))


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_torch(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dataset(cfg: dict) -> dict[str, np.ndarray]:
    p = cfg["prime"]
    pairs = np.array([(a, b) for a in range(p) for b in range(1, p)], np.int64)
    inverse = np.array([0] + [pow(b, -1, p) for b in range(1, p)], np.int64)
    labels = pairs[:, 0] * inverse[pairs[:, 1]] % p
    inputs = np.zeros((len(pairs), 2 * p), np.float32)
    indices = np.arange(len(pairs))
    inputs[indices, pairs[:, 0]] = 1
    inputs[indices, p + pairs[:, 1]] = 1
    permutation = np.random.default_rng(cfg["data_seed"]).permutation(len(pairs))
    n = int(round(len(pairs) * cfg["train_fraction"]))
    return {"inputs": inputs[permutation], "targets": labels[permutation],
            "pairs": pairs[permutation], "permutation": permutation,
            "train_mask": np.arange(len(pairs)) < n}


class PairedMLP(nn.Module):
    """第一维是独立模型，按CONDITIONS排列，各组初始化逐seed相同。"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.eps = cfg["layernorm_eps"]
        self.depth = cfg["hidden_layers"]
        dimensions = [2 * cfg["prime"]] + [cfg["hidden_size"]] * self.depth
        states = []
        for seed in cfg["model_seeds"]:
            state = {k: [] for k in ("w", "b", "nw", "nb")}
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
                    linear = nn.Linear(fan_in, fan_out)
                    norm = nn.LayerNorm(fan_out, eps=self.eps)
                    for key, value in (("w", linear.weight), ("b", linear.bias),
                                       ("nw", norm.weight), ("nb", norm.bias)):
                        state[key].append(value.detach().clone())
                output = nn.Linear(dimensions[-1], cfg["prime"])
                state["ow"] = output.weight.detach().clone()
                state["ob"] = output.bias.detach().clone()
            states.append(state)
        states = states * len(cfg["conditions"])
        for key in ("w", "b", "nw", "nb"):
            setattr(self, key, nn.ParameterList([
                nn.Parameter(torch.stack([s[key][layer] for s in states]))
                for layer in range(self.depth)]))
        self.ow = nn.Parameter(torch.stack([s["ow"] for s in states]))
        self.ob = nn.Parameter(torch.stack([s["ob"] for s in states]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(0).expand(self.ow.shape[0], -1, -1)
        for layer in range(self.depth):
            h = F.gelu(torch.bmm(h, self.w[layer].transpose(1, 2))
                       + self.b[layer][:, None, :])
            mean = h.mean(-1, keepdim=True)
            variance = h.var(-1, unbiased=False, keepdim=True)
            h = (h - mean) * torch.rsqrt(variance + self.eps)
            h = h * self.nw[layer][:, None, :] + self.nb[layer][:, None, :]
        return torch.bmm(h, self.ow.transpose(1, 2)) + self.ob[:, None, :]


def stable_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    y = targets.expand(logits.shape[:-1])
    chosen = logits.gather(-1, y.unsqueeze(-1))
    relative = (logits - chosen).scatter(-1, y.unsqueeze(-1), -torch.inf)
    # 数学上与普通 CE 相同，避免 softmax 真类概率舍入成 1 后梯度消失。
    return F.softplus(torch.logsumexp(relative, dim=-1))


def gap_variance(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """每个独立模型的一项方差；不混合不同模型，也不把真类自身的0差距混入。"""
    target_index = y[None, :, None].expand(z.shape[0], -1, 1)
    chosen = z.gather(-1, target_index)
    indices = torch.arange(z.shape[-1] - 1, device=z.device)[None, :]
    wrong_indices = indices + (indices >= y[:, None]).to(torch.long)
    others = z.gather(-1, wrong_indices[None].expand(z.shape[0], -1, -1))
    gap = chosen - others
    mean = gap.mean(dim=(1, 2), keepdim=True)
    # 均值保留梯度，用中心化差的平方避免大幅度、极小方差时的消减误差。
    return (gap - mean).square().mean(dim=(1, 2))


def losses_per_model(z: torch.Tensor, y: torch.Tensor, cfg: dict) -> torch.Tensor:
    size = len(cfg["model_seeds"])
    if z.shape[0] != size * len(cfg["conditions"]):
        raise ValueError("模型数与CONDITIONS不一致。")
    losses = []
    for index, (_, kind, coefficient) in enumerate(cfg["conditions"]):
        part = z[index * size:(index + 1) * size]
        if kind == "raw_mse":
            target = F.one_hot(y, z.shape[-1]).to(z.dtype)
            value = (part - target).square().mean(dim=(-1, -2))
        else:
            value = stable_ce(part, y).mean(-1)
            if coefficient != 0:
                value = value + coefficient * gap_variance(part, y)
        losses.append(value)
    return torch.cat(losses)


def create_optimizer(model: nn.Module, cfg: dict):
    return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"],
                            weight_decay=cfg["weight_decay"])


def train_step(model, optimizer, generator, x, y, n, cfg, step):
    for group in optimizer.param_groups:
        group["lr"] = cfg["learning_rate"] * min(step / max(cfg["warmup_steps"], 1), 1)
    idx = torch.randint(n, (cfg["batch_size"],), generator=generator, device=x.device)
    loss = losses_per_model(model(x[idx]), y[idx], cfg)
    optimizer.zero_grad(set_to_none=True)
    # 求和而不是跨模型求平均，保证每个模型都等价于单独训练。
    loss.sum().backward()
    optimizer.step()
    return loss.detach()


def cpu_weights(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def initial_hashes(model, cfg) -> list[dict]:
    arrays = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    count = len(cfg["model_seeds"])
    rows = []
    for i, seed in enumerate(cfg["model_seeds"]):
        fingerprints = {}
        for arm_index, (name, _, _) in enumerate(cfg["conditions"]):
            j = i + arm_index * count
            h = hashlib.sha256()
            for array in arrays:
                h.update(np.ascontiguousarray(array[j]).tobytes())
            fingerprints[name] = h.hexdigest()
        if len(set(fingerprints.values())) != 1:
            raise AssertionError("配对初始化不一致。")
        rows.append({"seed": seed, "arm_sha256": fingerprints, "identical": True})
    return rows


def evaluation_steps(cfg) -> set[int]:
    steps = {0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500, 750,
             1000, 1500, 2000, 3000, 4000, 5000, 7500, 10_000, 12_500,
             15_000, 17_500, 20_000, cfg["max_steps"]}
    steps.update(range(cfg["eval_interval"], cfg["max_steps"] + 1, cfg["eval_interval"]))
    steps.update(cfg["snapshot_steps"])
    return {s for s in steps if 0 <= s <= cfg["max_steps"]}


@torch.no_grad()
def collect_outputs(model, x, batch_size) -> np.ndarray:
    # 全部原始输出保存在 CPU，显存占用由 EVAL_BATCH_SIZE 控制。
    result = np.empty((model.ow.shape[0], len(x), model.ob.shape[-1]), np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        z = model(x[start:stop])
        if not torch.isfinite(z).all().item():
            raise FloatingPointError("检测到非有限输出，不能把数值故障当成不泛化。")
        result[:, start:stop] = z.cpu().numpy()
    return result


def output_metrics(z: np.ndarray, y: np.ndarray) -> dict:
    # 每个模型单独用 float64 统计，避免很低 MSE/CE 的汇总消减误差。
    z = z.astype(np.float64)
    selected = z[np.arange(len(y)), y]
    wrong = z.copy()
    wrong[np.arange(len(y)), y] = -np.inf
    relative = wrong - selected[:, None]
    maximum = np.maximum(0.0, relative.max(-1))
    # log1p 保留极低 CE 的有效数字，同时兼容真类不是最大值的情况。
    ce = np.log1p(np.exp(relative).sum(-1)) if relative.max() < 500 else (
        maximum + np.log(np.exp(-maximum) + np.exp(relative - maximum[:, None]).sum(-1)))
    errors = z.copy()
    errors[np.arange(len(y)), y] -= 1.0
    negative = z[np.arange(z.shape[-1])[None, :] != y[:, None]]
    gaps = selected[:, None] - negative.reshape(len(y), z.shape[-1] - 1)
    gap_var = float(gaps.var())
    margin = selected - wrong.max(-1)
    error = z.argmax(-1) != y
    return {
        "accuracy": float(1 - error.mean()), "error_count": int(error.sum()),
        "raw_ce": float(ce.mean()), "raw_mse_01": float(np.mean(errors ** 2)),
        "mse_vs_zero_output": float(z.shape[-1] * np.mean(errors ** 2)),
        "max_target_abs_error": float(np.abs(errors).max()),
        "true_output_mean": float(selected.mean()), "true_output_std": float(selected.std()),
        "true_output_rmse_to_1": float(np.sqrt(np.mean((selected - 1) ** 2))),
        "wrong_output_mean": float(negative.mean()), "wrong_output_std": float(negative.std()),
        "wrong_output_rms_to_0": float(np.sqrt(np.mean(negative ** 2))),
        "all_gaps_mean": float(gaps.mean()), "all_gaps_variance": gap_var,
        "all_gaps_std": math.sqrt(gap_var),
        "between_sample_gap_variance": float(gaps.mean(axis=1).var()),
        "within_sample_gap_variance": float(gaps.var(axis=1).mean()),
        "margin_mean": float(margin.mean()), "margin_std": float(margin.std()),
        "margin_min": float(margin.min()), "margin_q10": float(np.quantile(margin, .1)),
        "margin_median": float(np.median(margin)), "margin_q90": float(np.quantile(margin, .9)),
    }


def population_metrics(predictions, targets, p) -> dict:
    models, samples = predictions.shape
    counts = np.zeros((samples, p), np.int32)
    for pred in predictions:
        np.add.at(counts, (np.arange(samples), pred), 1)
    agreement = float(np.mean(np.sum(counts * (counts - 1), axis=1) / (models * (models - 1))))
    _, frequencies = np.unique(predictions, axis=0, return_counts=True)
    probabilities = frequencies / models
    return {"point_agreement": agreement,
            "function_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
            "function_count": len(frequencies),
            "exact_target_fraction": float(np.all(predictions == targets, axis=1).mean())}


def evaluate(model, x, data, cfg, step, elapsed_s, out, artifact_names):
    z = collect_outputs(model, x, cfg["eval_batch_size"])
    n = int(data["train_mask"].sum())
    seeds = cfg["model_seeds"]
    rows = []
    populations = []
    predictions = z.argmax(-1).astype(np.uint8)
    for arm_index, (arm, kind, coefficient) in enumerate(cfg["conditions"]):
        subset = predictions[arm_index * len(seeds):(arm_index + 1) * len(seeds)]
        pop = {"objective": arm, "regularization_coefficient": coefficient,
               "step": step, "elapsed_active_s": elapsed_s}
        for split, sl in (("full", slice(None)), ("test", slice(n, None))):
            pop.update({f"{split}_{k}": v for k, v in population_metrics(
                subset[:, sl], data["targets"][sl], cfg["prime"]).items()})
        populations.append(pop)
        for i, seed in enumerate(seeds):
            j = arm_index * len(seeds) + i
            row = {"objective": arm, "objective_kind": kind,
                   "regularization_coefficient": coefficient, "seed": seed, "step": step,
                   "elapsed_active_s": elapsed_s,
                   "full_accuracy": float(np.mean(predictions[j] == data["targets"])),
                   "full_rule_exact": bool(np.array_equal(predictions[j], data["targets"])),
                   "function_sha256": digest(predictions[j])}
            for split, sl in (("train", slice(0, n)), ("test", slice(n, None))):
                row.update({f"{split}_{k}": v for k, v in output_metrics(z[j, sl], data["targets"][sl]).items()})
            for split in ("train", "test"):
                penalty = coefficient * row[f"{split}_all_gaps_variance"]
                row[f"{split}_alignment_penalty"] = penalty
                row[f"{split}_objective_loss"] = (row[f"{split}_raw_mse_01"] if kind == "raw_mse"
                                                    else row[f"{split}_raw_ce"] + penalty)
            rows.append(row)
        group = rows[-len(seeds):]
        log(f"{arm} step={step:,} | train/test准确率="
            f"{np.mean([r['train_accuracy'] for r in group]):.3%}/"
            f"{np.mean([r['test_accuracy'] for r in group]):.3%} | "
            f"完整规则={sum(r['full_rule_exact'] for r in group)}/{len(seeds)} | "
            f"训练CE={np.median([r['train_raw_ce'] for r in group]):.3g} | "
            f"训练MSE={np.median([r['train_raw_mse_01'] for r in group]):.3g} | "
            f"差距均值/标准差={np.median([r['train_all_gaps_mean'] for r in group]):.3g}/"
            f"{np.median([r['train_all_gaps_std'] for r in group]):.3g} | "
            f"总目标={np.median([r['train_objective_loss'] for r in group]):.3g}")
    name = f"predictions/step_{step:07d}.npz"
    atomic_npz(out / name, predictions=predictions, step=np.array(step))
    artifact_names.add(name)
    if step in cfg["snapshot_steps"] or step == cfg["max_steps"]:
        name = f"snapshots/outputs_{step:07d}.npz"
        atomic_npz(out / name, raw_outputs=z, step=np.array(step))
        artifact_names.add(name)
        name = f"snapshots/weights_{step:07d}.pt"
        atomic_torch(out / name, {"step": step, "model": cpu_weights(model), "signature": signature(cfg)})
        artifact_names.add(name)
    return rows, populations


def save_checkpoint(path, model, optimizer, generator, cfg, state):
    atomic_torch(path, {"signature": signature(cfg), "state": state,
                       "model": cpu_weights(model), "optimizer": optimizer.state_dict(),
                       "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})


def restore_checkpoint(path, model, optimizer, generator, cfg):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["signature"] != signature(cfg):
        raise ValueError("续跑配置发生改变。需要新实验时请改 RESULT_DIR，不要覆盖旧结果。")
    if payload["state"]["step"] > cfg["max_steps"]:
        raise ValueError("MAX_STEPS 小于已完成步数。")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["batch_rng"].cpu())
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return payload["state"]


def first_sustained(rows, predicate, checks):
    for i in range(len(rows) - checks + 1):
        if all(predicate(r) for r in rows[i:i + checks]):
            return rows[i]["step"]
    return None


def summarize(cfg, state):
    result = []
    for arm, kind, coefficient in cfg["conditions"]:
        for seed in cfg["model_seeds"]:
            rows = sorted((r for r in state["rows"] if r["objective"] == arm and r["seed"] == seed),
                          key=lambda r: r["step"])
            if not rows:
                continue
            checks = cfg["fit_consecutive_evals"]
            fit = first_sustained(rows, lambda r: r["train_error_count"] == 0, checks)
            recovery = first_sustained(rows, lambda r: r["test_accuracy"] >= cfg["generalization_accuracy"], checks)
            plateau_span = 0
            start = None
            if fit is not None:
                for r in rows:
                    if r["step"] < fit or (recovery is not None and r["step"] >= recovery):
                        continue
                    if r["train_error_count"] == 0 and r["test_accuracy"] <= cfg["plateau_accuracy_max"]:
                        start = r["step"] if start is None else start
                        plateau_span = max(plateau_span, r["step"] - start)
                    else:
                        start = None
            passed = (fit is not None and recovery is not None and
                      recovery - fit >= cfg["min_plateau_steps"] and
                      plateau_span >= cfg["min_plateau_steps"])
            if passed:
                classification = "delayed_generalization_screen_pass"
            elif fit is None:
                classification = "training_fit_not_confirmed"
            elif recovery is None:
                classification = "no_sustained_99pct_within_budget"
            else:
                classification = "generalized_without_long_low_accuracy_plateau"
            row = {"objective": arm, "objective_kind": kind,
                   "regularization_coefficient": coefficient,
                   "seed": seed, "last_step": rows[-1]["step"],
                   "train_exact_sustained_step": fit, "test_99pct_sustained_step": recovery,
                   "post_fit_low_accuracy_plateau_steps": plateau_span,
                   "classification": classification,
                   "last_train_accuracy": rows[-1]["train_accuracy"],
                   "last_test_accuracy": rows[-1]["test_accuracy"],
                   "last_full_rule_exact": rows[-1]["full_rule_exact"],
                   "last_train_raw_ce": rows[-1]["train_raw_ce"],
                   "last_train_raw_mse_01": rows[-1]["train_raw_mse_01"],
                   "last_train_all_gaps_mean": rows[-1]["train_all_gaps_mean"],
                   "last_train_all_gaps_std": rows[-1]["train_all_gaps_std"],
                   "last_train_objective_loss": rows[-1]["train_objective_loss"]}
            for threshold in (.1, .5, .9, .95, .99, 1.):
                row[f"first_test_{threshold:g}"] = next((r["step"] for r in rows if r["test_accuracy"] >= threshold), None)
            result.append(row)
    return result


def condition_colors(cfg):
    palette = ("#2374ab", "#bc6c25", "#36864c", "#8956a8", "#c54250")
    return [(name, palette[i % len(palette)]) for i, (name, _, _) in enumerate(cfg["conditions"])]


def make_plots(out, rows, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("未安装 matplotlib，CSV/权重/输出仍完整保存；安装后重跑可补图。")
        return
    for xkey, xlabel, filename in (("step", "Training step", "curves_steps.png"),
                                  ("elapsed_active_s", "Active elapsed time (s)", "curves_time.png")):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fields = ("accuracy", "objective_loss", "raw_ce", "raw_mse_01", "all_gaps_mean", "all_gaps_std")
        labels = ("Accuracy", "Optimized objective", "CE of raw outputs", "Raw 0/1 MSE",
                  "Mean true-minus-other gap (not fixed)", "Std of true-minus-other gaps (alignment)")
        for arm, color in condition_colors(cfg):
            group = [r for r in rows if r["objective"] == arm]
            steps = sorted(set(r["step"] for r in group))
            chunks = [[r for r in group if r["step"] == step] for step in steps]
            x = [g[0][xkey] for g in chunks]
            for ax, field, label in zip(axes.flat, fields, labels):
                for split, style in (("train", "--"), ("test", "-")):
                    values = [np.array([r[f"{split}_{field}"] for r in g]) for g in chunks]
                    center = [float(np.mean(v)) for v in values]
                    ax.plot(x, center, style, color=color, label=f"{arm} {split}", lw=1.4)
                    if split == "test":
                        ax.fill_between(x, [np.quantile(v, .1) for v in values],
                                        [np.quantile(v, .9) for v in values], color=color, alpha=.1)
                ax.set_title(label, fontsize=10)
                ax.set_xlabel(xlabel)
                ax.grid(alpha=.2)
                if field in ("objective_loss", "raw_ce", "raw_mse_01", "all_gaps_std"):
                    ax.set_yscale("log")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle("Mod97 division: CE + amplitude alignment / raw MSE (mean, 10-90% band)")
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)

    seeds = sorted(set(r["seed"] for r in rows))
    columns = min(4, len(seeds))
    fig, axes = plt.subplots(math.ceil(len(seeds) / columns), columns,
                             figsize=(4 * columns, 2.8 * math.ceil(len(seeds) / columns)),
                             squeeze=False)
    for ax, seed in zip(axes.flat, seeds):
        for arm, color in condition_colors(cfg):
            group = sorted((r for r in rows if r["objective"] == arm and r["seed"] == seed),
                           key=lambda r: r["step"])
            for split, style in (("train", "--"), ("test", "-")):
                ax.plot([r["step"] for r in group], [r[f"{split}_accuracy"] for r in group],
                        style, color=color, lw=1.1, label=f"{arm} {split}")
        ax.set(title=f"Seed {seed}", ylim=(-.02, 1.02), xlabel="Step", ylabel="Accuracy")
        ax.set_xscale("symlog", linthresh=100)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(seeds):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Per-seed accuracy: delayed transitions must be checked individually")
    fig.tight_layout()
    fig.savefig(out / "per_seed_accuracy.png", dpi=150)
    plt.close(fig)


def matched_ce_summary(cfg, rows):
    result = []
    for name, kind, coefficient in cfg["conditions"]:
        for seed in cfg["model_seeds"]:
            history = sorted((r for r in rows if r["objective"] == name and r["seed"] == seed),
                             key=lambda r: r["step"])
            for level in cfg["matched_ce_levels"]:
                crossed = next((r for r in history if r["train_error_count"] == 0
                                and r["train_raw_ce"] <= level), None)
                row = {"objective": name, "objective_kind": kind,
                       "regularization_coefficient": coefficient, "seed": seed,
                       "raw_ce_threshold": level, "reached": crossed is not None}
                for key in ("step", "elapsed_active_s", "train_raw_ce", "train_raw_mse_01",
                            "train_all_gaps_mean", "train_all_gaps_std", "test_accuracy", "full_rule_exact"):
                    row[key] = crossed[key] if crossed is not None else None
                result.append(row)
    return result


def persist_tables(out, cfg, state, status):
    csv_write(out / "training_curves.csv", state["rows"])
    csv_write(out / "function_population.csv", state["populations"])
    summaries = summarize(cfg, state)
    csv_write(out / "seed_summary.csv", summaries)
    csv_write(out / "matched_raw_ce.csv", matched_ce_summary(cfg, state["rows"]))
    atomic_json(out / "summary.json", {
        "status": status, "last_step": state["step"], "elapsed_active_s": state["elapsed_active_s"],
        "smoke_test": cfg["smoke_test"], "seed_results": summaries,
        "说明": ["自动标签只是延迟泛化曲线筛选，须结合逐seed曲线判断跃升形状。",
                 "未出现只表示当前设置和预算内未出现，不等于任何MSE都不可能grokking。",
                 "两种训练目标的数值不可直接比较；两组均另行记录原始CE和原始MSE。",
                 "没有静态采样，本实验不能单独证明参数体积占比发生了什么变化。",
                 "正则是训练样本真类对所有错误类的logit差距方差，共同幅度不固定。",
                 "训练用minibatch方差，日志中的方差在完整训练/测试集上分别计算。",
                 "matched_raw_ce是首次通过共同门槛的已记录评估，不是精确相同CE。未达到也列出。",
                 "elapsed_active_s包含训练、评估和保存，不包含暂停期间；不是单模型GPU时间。"]})
    atomic_json(out / "artifact_manifest.json", {"step": state["step"], "artifacts": state["artifacts"]})


def package(out, state):
    archive = out.parent / (out.name + "_package.zip")
    temporary = archive.with_name(archive.name + ".tmp")
    names = set(state["artifacts"])
    names.update(p.name for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".csv", ".png", ".py", ".md"))
    names.add("dataset_split.npz")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for name in sorted(names):
            path = out / name
            if path.is_file():
                handle.write(path, arcname=name)
    temporary.replace(archive)
    return archive


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def handler(self, signum, frame):
        # 不在 optimizer.step 中抛异常，避免保存只更新了一半的 Adam 状态。
        self.requested = True

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self.handler)

    def restore(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def set_precision(allow_tf32):
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def check_result_directory(out: Path, cfg: dict) -> None:
    if not out.exists() or not any(out.iterdir()):
        return
    if cfg["resume"] and (out / "latest_checkpoint.pt").is_file():
        return
    # 旧版在读取 __file__ 前只生成这些文件；此时尚未评估或更新任何模型。
    startup_files = {"config.json", "dataset_split.npz", "pairing_audit.json"}
    entries = list(out.iterdir())
    startup_only = all(
        (p.is_file() and p.name in startup_files)
        or (p.is_dir() and p.name in {"predictions", "snapshots"} and not any(p.iterdir()))
        for p in entries
    )
    if cfg["resume"] and startup_only and (out / "config.json").is_file():
        try:
            previous = json.loads((out / "config.json").read_text(encoding="utf-8"))
            same = signature(previous) == signature(cfg)
        except (OSError, ValueError, KeyError, TypeError):
            same = False
        if same:
            log("检测到上次启动失败留下的初始化文件，尚无训练结果；使用相同配置重新启动。")
            return
    raise FileExistsError(f"已有结果且无法续跑，请换一个 RESULT_DIR：{out}")


def archive_source(out: Path) -> dict:
    filename = globals().get("__file__")
    runtime_filename = main.__code__.co_filename
    if filename and not runtime_filename.startswith("<"):
        source = Path(filename).resolve()
        if source.is_file():
            if source != (out / source.name).resolve():
                shutil.copy2(source, out / source.name)
            return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "source_capture": "script_file", "source_snapshot": source.name}
    # IPython 通常把当前代码单元格放入 linecache，不依赖不存在的 __file__。
    text = "".join(linecache.getlines(runtime_filename))
    if text:
        snapshot = out / "executed_cell.py"
        snapshot.write_text(text, encoding="utf-8")
        return {"source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "source_capture": "runtime_source_cache", "source_snapshot": snapshot.name}
    log("当前环境未提供可读取的脚本/单元格源码；配置照常保存，源码hash记为null，不影响训练。")
    return {"source_sha256": None, "source_capture": "unavailable", "source_snapshot": None}


def main(cfg=None):
    cfg = config_dict() if cfg is None else cfg
    device = torch.device(cfg["device"])
    if (device.type != "cuda" or not torch.cuda.is_available()) and not cfg["smoke_test"]:
        raise RuntimeError("正式实验需要 CUDA PyTorch；不会静默改用 CPU。")
    if len(cfg["model_seeds"]) < 2 or len(set(cfg["model_seeds"])) != len(cfg["model_seeds"]):
        raise ValueError("至少两个不同的 model seed。")
    names = [row[0] for row in cfg["conditions"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("CONDITIONS不能为空且名称不能重复。")
    for name, kind, coefficient in cfg["conditions"]:
        if kind not in {"ce", "ce_align", "raw_mse"} or not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"无效条件：{name}")
        if kind in {"ce", "raw_mse"} and coefficient != 0:
            raise ValueError("原CE/MSE基线的正则系数必须为0。")
    if not 0 < cfg["train_fraction"] < 1 or not 2 < cfg["prime"] < 256:
        raise ValueError("需 0<TRAIN_FRACTION<1、2<PRIME<256。")
    if min(cfg[k] for k in ("max_steps", "batch_size", "eval_batch_size", "eval_interval",
                           "checkpoint_interval", "log_interval")) < 1:
        raise ValueError("步数、批量和间隔必须为正。")
    if cfg["weight_decay"] != 0:
        raise ValueError("本实验固定 WD=0；需要正则对照时应另设协议。")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_precision(cfg["allow_tf32"])
    random.seed(cfg["data_seed"])
    np.random.seed(cfg["data_seed"])
    torch.manual_seed(cfg["data_seed"])
    out = Path(cfg["result_dir"])
    checkpoint = out / "latest_checkpoint.pt"
    check_result_directory(out, cfg)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "snapshots"):
        (out / name).mkdir(exist_ok=True)
    data = dataset(cfg)
    n = int(data["train_mask"].sum())
    if not 0 < n < len(data["targets"]):
        raise ValueError("训练/测试集不能为空。")
    model = PairedMLP(cfg).to(device)
    paired_hashes = initial_hashes(model, cfg)
    optimizer = create_optimizer(model, cfg)
    generator = torch.Generator(device=device).manual_seed(cfg["train_order_seed"])
    state = {"step": 0, "rows": [], "populations": [], "elapsed_active_s": 0.0, "artifacts": []}
    if cfg["resume"] and checkpoint.exists():
        state = restore_checkpoint(checkpoint, model, optimizer, generator, cfg)
        log(f"断点续跑：step={state['step']:,}，已累计 {state['elapsed_active_s']/60:.1f} 分钟")
    atomic_json(out / "config.json", cfg)
    atomic_npz(out / "dataset_split.npz", **data)
    atomic_json(out / "pairing_audit.json", {"initialization": paired_hashes,
                "dataset_sha256": digest(np.column_stack((data["pairs"], data["targets"]))),
                "train_count": n, "test_count": len(data["targets"]) - n,
                "batch": "每一步所有模型使用同一批均匀有放回抽取的训练样本。"})
    source_metadata = archive_source(out)
    metadata = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                "numpy": np.__version__, "cuda_runtime": torch.version.cuda, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU工程自检",
                "parameters_per_model": sum(p.numel() for p in model.parameters()) // (len(cfg["conditions"]) * len(cfg["model_seeds"])),
                **source_metadata,
                "onehot_perfect_fit_raw_ce": math.log(math.e + cfg["prime"] - 1) - 1}
    atomic_json(out / "environment.json", metadata)
    (out / "说明.md").write_text(
        "# CE 幅度对齐正则的 grokking 对照\n\n"
        "模97除法，输入为两个符号的one-hot拼接，使用以前的MLP改版，不是原论文Transformer。\n"
        "MSE直接拟合0/1 one-hot，不经过sigmoid/softmax；分类由原始输出argmax决定。\n"
        "97类完全拟合one-hot时，原始输出的CE仍约3.59；这不是训练失败。\n"
        "默认每组32个配对初始化，同一训练/测试划分、相同逐步batch；独立seed不等于独立数据划分。\n"
        "五组为原CE、CE+0.01R、CE+0.1R、CE+R、原始0/1 MSE，完整报告所有系数。\n"
        "R=Var(z[i,y[i]]-z[i,c])，i遍历训练样本，c遍历不等于y[i]的错误类别。\n"
        "总体方差只在单个模型内计算，均值保留梯度。两类时退化为正确方向logit差距方差。\n"
        "正则同时约束不同样本的平均差距、同一样本对不同错误类的差距，不仅是前一项。\n"
        "所有差距一起增加同一个数，R不变；每个输入的全部logit一起平移，R也不变。\n"
        "R不固定幅度、不等于权重L2或logit范数惩罚，也不直接把输出拉到0和1。\n"
        "每次更新只使用同一minibatch内的方差，评估的方差另外在完整集合上精确计算。\n"
        "延迟泛化需要在训练集已拟合后观察测试平台和后续改善，不只看终点是否准确。\n"
        "seed_summary仅自动筛选，训练未拟合、预算内未出现、直接泛化分别记录。\n"
        "training_curves保存每seed完整统计；function_population保存有限seed的经验分布，不是静态体积。\n"
        "predictions保存每次评估的完整离散映射；snapshots保存指定步的原始float32输出与权重。\n"
        "所有输入顺序见dataset_split.npz，前train_count个是训练集；模型按CONDITIONS顺序，每组按MODEL_SEEDS排列。\n"
        "matched_raw_ce.csv记录首次达到共同原始CE门槛的快照；保留实际CE，不称作精确matched-loss。\n"
        "checkpoint保存Adam状态、随机状态和已提交记录。程序退出后的空闲时间不计入曲线耗时。\n"
        "结果包不含latest_checkpoint.pt，断点文件请保留在AutoDL实例；中间权重仍包含在结果包。\n"
        "正式训练使用与旧改版相同的TF32设置；最后重新关闭TF32前向评估，结果单列，不改变训练。\n",
        encoding="utf-8")
    x = torch.from_numpy(data["inputs"]).to(device)
    y = torch.from_numpy(data["targets"]).to(device)
    ideal_ce = metadata["onehot_perfect_fit_raw_ce"]
    log(f"=== CE 幅度对齐对照 | {metadata['gpu']} | {len(cfg['conditions'])*len(cfg['model_seeds'])} 个独立模型 ===")
    log(f"网络 {2*cfg['prime']} -> {cfg['hidden_size']}x{cfg['hidden_layers']} -> {cfg['prime']} | "
        f"每模型参数={metadata['parameters_per_model']:,} | train/test={n}/{len(y)-n} | WD=0")
    log(f"MSE零输出基线={1/cfg['prime']:.6g}；完美one-hot的原始CE={ideal_ce:.6g}")
    log(f"预定条件：{cfg['conditions']}；正则不固定共同幅度。")
    log(f"结果目录：{out}；Ctrl+C/SIGTERM安全暂停，重跑自动续训。")
    if cfg["smoke_test"]:
        log("工程冒烟模式：不是科学实验结果。")
    events = evaluation_steps(cfg)
    artifacts = set(state["artifacts"])
    evaluated = {r["step"] for r in state["rows"]}
    base_elapsed = state["elapsed_active_s"]
    started = time.perf_counter()
    initial_step = state["step"]
    stop = StopRequest()
    stop.install()
    in_update = False
    status = "running"

    def elapsed():
        return base_elapsed + time.perf_counter() - started

    def record():
        rows, pops = evaluate(model, x, data, cfg, state["step"], elapsed(), out, artifacts)
        state["rows"].extend(rows)
        state["populations"].extend(pops)
        evaluated.add(state["step"])

    def commit():
        state["artifacts"] = sorted(artifacts)
        state["elapsed_active_s"] = elapsed()
        save_checkpoint(checkpoint, model, optimizer, generator, cfg, state)
        persist_tables(out, cfg, state, status)

    try:
        if state["step"] not in evaluated:
            record()
        commit()
        model.train()
        for step in range(state["step"] + 1, cfg["max_steps"] + 1):
            in_update = True
            loss = train_step(model, optimizer, generator, x, y, n, cfg, step)
            if device.type == "cuda" and (step in events or step % cfg["log_interval"] == 0):
                torch.cuda.synchronize(device)
            state["step"] = step
            in_update = False
            if step in events:
                record()
            if step % cfg["log_interval"] == 0:
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError("训练loss非有限，已停止，使用此前完整断点排查。")
                seconds = time.perf_counter() - started
                speed = (step - initial_step) / max(seconds, 1e-9)
                remaining = (cfg["max_steps"] - step) / max(speed, 1e-9)
                log(f"PROGRESS {step:,}/{cfg['max_steps']:,} | {speed:.2f} step/s | "
                    f"累计{elapsed()/60:.1f}分钟 | 当前速度预计剩余{remaining/60:.1f}分钟")
            if step % cfg["checkpoint_interval"] == 0:
                commit()
                log(f"断点已保存 step={step:,}")
            if stop.requested:
                status = "paused"
                break
        if state["step"] not in evaluated:
            record()
        status = "paused" if stop.requested else "completed"
        commit()
    except KeyboardInterrupt:
        if in_update:
            log("中断发生在更新内部，不保存可能不完整的参数；下次从已有周期断点恢复。")
            return
        status = "paused"
        commit()
    except Exception:
        # 不把损坏/不完整状态覆盖到最后一个正常断点。
        log("运行异常，最后完整断点未被覆盖；排查后可直接续跑。")
        raise
    finally:
        stop.restore()

    if status == "completed":
        set_precision(False)
        strict = collect_outputs(model, x, cfg["eval_batch_size"])
        last = np.load(out / f"predictions/step_{state['step']:07d}.npz")["predictions"]
        audit = []
        for j in range(len(strict)):
            pred = strict[j].argmax(-1)
            audit.append({"objective": cfg["conditions"][j // len(cfg["model_seeds"])][0],
                          "seed": cfg["model_seeds"][j % len(cfg["model_seeds"])] ,
                          "changed_class_count_tf32_off": int((pred != last[j]).sum()),
                          "train": output_metrics(strict[j, :n], data["targets"][:n]),
                          "test": output_metrics(strict[j, n:], data["targets"][n:])})
        atomic_json(out / "final_tf32_off_audit.json", audit)
        name = f"snapshots/final_outputs_tf32_off_{state['step']:07d}.npz"
        atomic_npz(out / name, raw_outputs=strict, step=np.array(state["step"]))
        artifacts.add(name)
        set_precision(cfg["allow_tf32"])
        commit()
    make_plots(out, state["rows"], cfg)
    log(f"{status} | step={state['step']:,} | 结果已保存。")
    for arm, _, _ in cfg["conditions"]:
        summaries = [r for r in summarize(cfg, state) if r["objective"] == arm]
        log(f"{arm}：延迟泛化筛选通过 "
            f"{sum(r['classification']=='delayed_generalization_screen_pass' for r in summaries)}/{len(summaries)}，"
            f"最终完整规则 {sum(r['last_full_rule_exact'] for r in summaries)}/{len(summaries)}")
    if cfg["package_results"]:
        log("正在打包（原始输出和中间权重较大，断点文件不会放进下载包）……")
        log(f"结果包：{package(out, state)}")


if __name__ == "__main__":
    main()
