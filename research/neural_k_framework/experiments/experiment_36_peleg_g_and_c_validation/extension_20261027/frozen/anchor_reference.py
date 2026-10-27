"""Peleg协议的独立本地复核。修改CONFIG即可，无需argparse。"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "peleg_recheck"
CONFIG = {
    "device": "cuda:1",
    "output_dir": str(HERE / "results"),
    "mnist_root": str(HERE.parent / "overfitting_related_research" / "_smoke_mnist_dataset"),
    "data_seed": 202,
    "models": 512,
    "initialization_seed": 19092026,
    "batch_seed": 19092027,
    "epochs": 60,
    "batch_size": 2,
    "test_epochs": [0, 1, 5, 20, 60],
    "scales": [1.0, 0.2],
    "learning_rates": [0.001, 0.01],
    "model_chunk": 512,
    "test_model_chunk": 32,
    "test_input_chunk": 128,
    "prior_draws": 65536,
    "prior_seconds": 120.0,
    "smc_particles": 1024,
    "smc_replicas": 4,
    "smc_scale": 0.2,
    "smc_seed": 19094000,
    "gc_seconds": 120.0,
    "gc_max_draws": 4194304,
    "smc_seconds": 120.0,
    "smc_adapt_sweeps": 2,
    "smc_fixed_sweeps": 6,
    "smc_nodes": [0.69, 0.68, 0.65, 0.6, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
}

# 每层依次存放权重与bias；最后三层等价于1x1分组卷积。
SHAPES = [(3, 1, 5), (8, 3, 5), (60, 128, 1), (42, 60, 1), (2, 42, 1)]
LAYERS = []
cursor = 0
BIAS_INDICES = []
for co, ci, kernel in SHAPES:
    size = co * ci * kernel * kernel
    LAYERS.append((slice(cursor, cursor + size), slice(cursor + size, cursor + size + co), co, ci, kernel))
    BIAS_INDICES.extend(range(cursor + size, cursor + size + co))
    cursor += size + co
PARAMETERS = cursor
WEIGHT_INDICES = sorted(set(range(PARAMETERS)) - set(BIAS_INDICES))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace(temporary, path)


def replace(source, destination):
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.15 * (attempt + 1))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    replace(temporary, path)


def digest(array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def log(message, out):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + message
    print(line, flush=True)
    with (Path(out) / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def progress(out, stage, details):
    value = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "details": details}
    atomic_json(Path(out) / "status.json", value)
    text = "# Peleg本地复核进度\n\n更新时间：" + value["updated_at"] + "\n\n阶段：" + stage + "\n\n" + details + "\n"
    temporary = HERE / "PROGRESS.md.tmp"
    temporary.write_text(text, encoding="utf-8")
    replace(temporary, HERE / "PROGRESS.md")


def setup(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本协议需要CUDA；不自动转到CPU训练。")
    torch.cuda.set_device(device)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__) if "__file__" in globals() else None
    meta = {
        "config": cfg,
        "parameter_count": PARAMETERS,
        "bias_count": len(BIAS_INDICES),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_memory": torch.cuda.get_device_properties(device).total_memory,
        "tf32": False,
        "amp": False,
        "network_dtype": "float32",
        "score_dtype": "float64",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        "upstream_commit": "74c221cec9ae288534312aa837fa9fe9f44db117",
    }
    frozen = out / "metadata.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf-8"))
        if old != meta:
            raise RuntimeError("配置/源码已变化，必须另用结果目录，不能覆盖旧科学结果。")
    else:
        atomic_json(frozen, meta)
    return out, device


def prepare_data(cfg, out):
    path = out / "dataset.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {key: torch.from_numpy(data[key].copy()) for key in data.files}
    train = torchvision.datasets.MNIST(cfg["mnist_root"], train=True, download=True)
    test = torchvision.datasets.MNIST(cfg["mnist_root"], train=False, download=True)
    all_x = train.data.float().unsqueeze(1) / 255.0
    mean = all_x.mean()
    std = all_x.std(correction=1)
    generator = torch.Generator().manual_seed(cfg["data_seed"])
    classes = torch.randperm(10, generator=generator)[:2]
    ids, val_ids, test_ids, train_y, test_y = [], [], [], [], []
    for label, digit in enumerate(classes.tolist()):
        available = torch.where(train.targets == digit)[0]
        available = available[torch.randperm(len(available), generator=generator)]
        val_ids.append(available[:128])
        ids.append(available[128:136])
        train_y.extend([label] * 8)
        unseen = torch.where(test.targets == digit)[0]
        unseen = unseen[torch.randperm(len(unseen), generator=generator)]
        test_ids.append(unseen)
        test_y.extend([label] * len(unseen))
    ids, val_ids, test_ids = torch.cat(ids), torch.cat(val_ids), torch.cat(test_ids)
    data = {
        "train_x": ((all_x[ids] - mean) / std).contiguous(),
        "train_y": torch.tensor(train_y, dtype=torch.long),
        "test_x": ((test.data[test_ids].float().unsqueeze(1) / 255.0 - mean) / std).contiguous(),
        "test_y": torch.tensor(test_y, dtype=torch.long),
        "train_ids": ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "classes": classes,
        "mean": mean,
        "std": std,
    }
    assert set(classes.tolist()) == {0, 7}
    assert len(set(ids.tolist())) == 16 and not set(ids.tolist()) & set(val_ids.tolist())
    assert len(test_ids) == 2008
    atomic_npz(path, **{key: value.numpy() for key, value in data.items()})
    atomic_json(out / "dataset_manifest.json", {key: {"shape": list(value.shape), "sha256": digest(value)} for key, value in data.items()})
    return data


def forward_grouped(theta, x):
    models = theta.shape[0]
    z = x.repeat(1, models, 1, 1)
    for index, (weights, biases, co, ci, kernel) in enumerate(LAYERS):
        if index == 2:
            z = z.reshape(x.shape[0], models * 128, 1, 1)
        w = theta[:, weights].reshape(models * co, ci, kernel, kernel)
        b = theta[:, biases].reshape(-1)
        z = F.conv2d(z, w, b, groups=models)
        if index < 2:
            z = F.max_pool2d(F.relu(z), 2)
        elif index < 4:
            z = F.relu(z)
    return z.reshape(x.shape[0], models, 2).permute(1, 0, 2)


def forward(theta, x):
    models = theta.shape[0]
    # 首层输入对全部模型相同，可以直接合并输出滤波器，不必复制输入通道。
    ws, bs, co, ci, kernel = LAYERS[0]
    z = F.conv2d(x, theta[:, ws].reshape(models * co, ci, kernel, kernel), theta[:, bs].reshape(-1))
    z = F.max_pool2d(F.relu(z), 2)
    ws, bs, co, ci, kernel = LAYERS[1]
    z = F.conv2d(z, theta[:, ws].reshape(models * co, ci, kernel, kernel), theta[:, bs].reshape(-1), groups=models)
    z = F.max_pool2d(F.relu(z), 2)
    z = z.reshape(len(x), models, 128).permute(1, 0, 2)
    for index in range(2, 5):
        ws, bs, co, ci, _ = LAYERS[index]
        z = torch.bmm(z, theta[:, ws].reshape(models, co, ci).transpose(1, 2)) + theta[:, bs, None].transpose(1, 2)
        if index < 4:
            z = F.relu(z)
    return z


def native_ce(logits, labels):
    return F.cross_entropy(logits.reshape(-1, 2), labels.expand(logits.shape[0], -1).reshape(-1), reduction="none").reshape(logits.shape[:2]).mean(1)


def score(logits, labels):
    gap = logits[..., 1].double() - logits[..., 0].double()
    margins = gap * (2 * labels - 1)
    loss = F.softplus(-margins).mean(1)
    correct = (logits.argmax(-1) == labels).all(1)
    return loss, correct, margins


@torch.no_grad()
def evaluate_logits(theta, x, model_chunk=128, input_chunk=128):
    outputs = []
    for first in range(0, len(theta), model_chunk):
        rows = []
        for start in range(0, len(x), input_chunk):
            rows.append(forward(theta[first:first + model_chunk], x[start:start + input_chunk]))
        outputs.append(torch.cat(rows, dim=1))
    return torch.cat(outputs)


def sample_prior(count, scale, generator, device):
    theta = torch.empty((count, PARAMETERS), device=device).uniform_(-scale, scale, generator=generator)
    theta[:, BIAS_INDICES] = 0
    return theta


def parameter_summary(theta, scale):
    value = theta.detach().double()
    return {
        "bias_l2": value[:, BIAS_INDICES].norm(dim=1).cpu().numpy(),
        "weight_outside_fraction": (value[:, WEIGHT_INDICES].abs() > scale).double().mean(1).cpu().numpy(),
        "weight_l2": value[:, WEIGHT_INDICES].norm(dim=1).cpu().numpy(),
    }


def summary(logits, labels):
    loss, hard, margins = score(logits, labels)
    finite = torch.isfinite(logits).all(dim=(1, 2))
    acc = (logits.argmax(-1) == labels).double().mean(1)
    return {
        "models": len(logits), "finite": int(finite.sum()), "hard_fit": int((hard & finite).sum()),
        "accuracy_mean": float(acc[finite].mean()) if finite.any() else None,
        "bce_median": float(loss[finite].median()) if finite.any() else None,
        "bce_p10": float(torch.quantile(loss[finite], .1)) if finite.any() else None,
        "bce_p90": float(torch.quantile(loss[finite], .9)) if finite.any() else None,
        "margin_variance_median": float(margins[finite].var(1, correction=0).median()) if finite.any() else None,
    }


def prepare_initializations(cfg, out):
    path = out / "initializations.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(cfg["initialization_seed"])
    theta = sample_prior(cfg["models"], 1.0, generator, "cpu")
    batches = torch.stack([torch.randperm(16, generator=torch.Generator().manual_seed(cfg["batch_seed"] + epoch)) for epoch in range(cfg["epochs"])])
    value = {"base_theta": theta, "epoch_permutations": batches}
    atomic_torch(path, value)
    atomic_json(out / "initialization_manifest.json", {key: digest(v) for key, v in value.items()})
    return value


def run_training(cfg=CONFIG):
    out, device = setup(cfg)
    data = prepare_data(cfg, out)
    initial = prepare_initializations(cfg, out)
    x, y = data["train_x"].to(device), data["train_y"].to(device)
    tx, ty = data["test_x"].to(device), data["test_y"].to(device)
    completed = []
    for scale in cfg["scales"]:
        for lr in cfg["learning_rates"]:
            for train_bias in [False, True]:
                name = f"a{scale:g}_lr{lr:g}_bias{'train' if train_bias else 'zero'}"
                folder = out / "train" / name
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / "final.json").exists():
                    completed.append(name)
                    continue
                theta = torch.nn.Parameter((initial["base_theta"] * scale).to(device))
                optimizer = torch.optim.SGD([theta], lr=lr, momentum=0)
                history, train_outputs, elapsed_train, elapsed_eval, first_epoch = [], [], 0.0, 0.0, 0
                checkpoint = folder / "checkpoint.pt"
                if checkpoint.exists():
                    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    theta.data.copy_(state["theta"].to(device))
                    optimizer.load_state_dict(state["optimizer"])
                    history, train_outputs = state["history"], state["train_outputs"]
                    elapsed_train, elapsed_eval = state["elapsed_train"], state["elapsed_eval"]
                    first_epoch = state["epoch"] + 1
                log(f"TRAIN {name}，{cfg['models']}模型，从epoch {first_epoch}开始", out)
                progress(out, "SGD对照运行", f"已完成{len(completed)}/8条件；当前{name}。")
                started = time.perf_counter()
                for epoch in range(first_epoch, cfg["epochs"] + 1):
                    if epoch:
                        torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        for indices in initial["epoch_permutations"][epoch - 1].split(cfg["batch_size"]):
                            logits = forward(theta, x[indices])
                            optimizer.zero_grad(set_to_none=True)
                            native_ce(logits, y[indices]).sum().backward()
                            if not train_bias:
                                theta.grad[:, BIAS_INDICES] = 0
                            optimizer.step()
                        torch.cuda.synchronize(device)
                        elapsed_train += time.perf_counter() - tick
                    tick = time.perf_counter()
                    with torch.no_grad():
                        logits = evaluate_logits(theta, x, cfg["model_chunk"], 16)
                        row = {"epoch": epoch, **summary(logits, y)}
                        row["native_ce_mean"] = float(native_ce(logits, y).double().mean())
                        train_outputs.append(logits.cpu())
                        if epoch in cfg["test_epochs"]:
                            test_logits = evaluate_logits(theta, tx, cfg["test_model_chunk"], cfg["test_input_chunk"])
                            row["test"] = summary(test_logits, ty)
                            atomic_npz(folder / f"test_epoch_{epoch:03d}.npz", logits=test_logits.cpu().numpy())
                            del test_logits
                        history.append(row)
                    elapsed_eval += time.perf_counter() - tick
                    if epoch % 10 == 0 or epoch == cfg["epochs"]:
                        atomic_torch(checkpoint, {
                            "theta": theta.detach().cpu(), "optimizer": optimizer.state_dict(),
                            "history": history, "train_outputs": train_outputs, "epoch": epoch,
                            "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval,
                        })
                        log(f"{name} epoch={epoch} fit={row['hard_fit']}/{cfg['models']} BCE中位={row['bce_median']}，训练{elapsed_train:.1f}s", out)
                weights = theta.detach()
                atomic_npz(folder / "train_trajectory.npz", logits=torch.stack(train_outputs).numpy(), **parameter_summary(weights, scale))
                atomic_torch(folder / "final_weights.pt", weights.cpu())
                result = {"condition": name, "scale": scale, "lr": lr, "train_bias": train_bias,
                          "training_seconds": elapsed_train, "evaluation_seconds": elapsed_eval,
                          "invocation_seconds": time.perf_counter() - started, "history": history,
                          "weights_sha256": digest(weights), "final": history[-1]}
                atomic_json(folder / "final.json", result)
                log(f"DONE {name} 测试均值={history[-1]['test']['accuracy_mean']:.5f}", out)
                completed.append(name)
                del theta, optimizer, logits, weights
                torch.cuda.empty_cache()
    progress(out, "SGD对照完成", "8条件×512模型已完成；下一步为独立先验和短预算SMC。")


def reflect(value, scale):
    return scale - torch.abs(torch.remainder(value + scale, 4 * scale) - 2 * scale)


@torch.no_grad()
def network_energy(theta, x, y, cfg):
    z = evaluate_logits(theta, x, cfg["model_chunk"], 16)
    loss, hard, _ = score(z, y)
    return loss, hard


def run_prior(cfg=CONFIG):
    out, device = setup(cfg)
    if (out / "prior_calibration.json").exists():
        return json.loads((out / "prior_calibration.json").read_text(encoding="utf-8"))
    data = prepare_data(cfg, out)
    x, y = data["train_x"].to(device), data["train_y"].to(device)
    generator = torch.Generator(device=device).manual_seed(cfg["smc_seed"] - 1)
    draws, hits, histogram, samples = 0, 0, [], []
    tick = time.perf_counter()
    while draws < cfg["prior_draws"] and time.perf_counter() - tick < cfg["prior_seconds"]:
        count = min(cfg["model_chunk"], cfg["prior_draws"] - draws)
        theta = sample_prior(count, cfg["smc_scale"], generator, device)
        loss, hard = network_energy(theta, x, y, cfg)
        hits += int(hard.sum())
        draws += count
        histogram.append(loss.cpu())
        samples.append(hard.cpu())
    result = {"draws": draws, "hard_hits": hits, "p_hard": hits / draws,
              "seconds": time.perf_counter() - tick, "draws_per_second": draws / (time.perf_counter() - tick)}
    atomic_npz(out / "prior_calibration.npz", loss=torch.cat(histogram).numpy(), hard=torch.cat(samples).numpy())
    atomic_json(out / "prior_calibration.json", result)
    log(f"PRIOR {draws}次独立先验，硬拟合{hits}次，{result['seconds']:.1f}s", out)
    return result


@torch.no_grad()
def save_smc_node(folder, label, theta, loss, upper, k_conditional, level, ancestry, cfg, data, device, extra):
    tx = data["test_x"].to(device)
    logits = evaluate_logits(theta, tx, cfg["test_model_chunk"], cfg["test_input_chunk"])
    train_logits = evaluate_logits(theta, data["train_x"].to(device), cfg["model_chunk"], 16)
    atomic_npz(folder / (label + ".npz"), train_logits=train_logits.cpu().numpy(), test_logits=logits.cpu().numpy(),
               loss=loss.cpu().numpy(), audit_theta=theta[:16].cpu().numpy(), ancestry=ancestry.cpu().numpy())
    # 此处只导出预测，不读取测试标签；由独立分析阶段评分。
    record = {"label": label, "upper": upper, "K_given_hard": k_conditional, "level": level,
              "unique_ancestors": int(ancestry.unique().numel()), "loss_mean": float(loss.mean()),
              "loss_min": float(loss.min()), "loss_max": float(loss.max()), **extra}
    atomic_json(folder / (label + ".json"), record)
    return record


def run_smc(cfg=CONFIG):
    out, device = setup(cfg)
    data = prepare_data(cfg, out)
    prior = run_prior(cfg)
    x, y = data["train_x"].to(device), data["train_y"].to(device)
    scale, n = cfg["smc_scale"], cfg["smc_particles"]
    for replica in range(cfg["smc_replicas"]):
        replica_started = time.perf_counter()
        folder = out / "smc" / f"replica_{replica}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "final.json").exists():
            continue
        generator = torch.Generator(device=device).manual_seed(cfg["smc_seed"] + replica)
        checkpoint = folder / "checkpoint.pt"
        rows, level, k, node_index = [], 0, 0.0, 0
        upper = math.log(2)
        scales = [0.05 * scale] * 6
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            theta, loss, ancestry = state["theta"].to(device), state["loss"].to(device), state["ancestry"].to(device)
            generator.set_state(state["rng"])
            rows, level, k, node_index = state["rows"], state["level"], state["k"], state["node_index"]
            upper, scales, gc = state["upper"], state["scales"], state["gc"]
        else:
            accepted, count, draws = [], 0, 0
            tick = time.perf_counter()
            while count < n and draws < cfg["gc_max_draws"] and time.perf_counter() - tick < cfg["gc_seconds"]:
                batch = sample_prior(cfg["model_chunk"], scale, generator, device)
                _, hard = network_energy(batch, x, y, cfg)
                selected = batch[hard]
                accepted.append(selected)
                count += len(selected)
                draws += len(batch)
            gc = {"draws": draws, "hits": count, "seconds": time.perf_counter() - tick}
            atomic_json(folder / "gc.json", gc)
            if count < n:
                atomic_json(folder / "final.json", {"status": "gc_budget", "gc": gc})
                log(f"SMC r{replica} G&C预算停止，{count}/{n}起点", out)
                continue
            theta = torch.cat(accepted)[:n].clone()
            loss, hard = network_energy(theta, x, y, cfg)
            assert bool(hard.all())
            ancestry = torch.arange(n, device=device)
            rows.append(save_smc_node(folder, "hard_start", theta, loss, upper, k, level, ancestry, cfg, data, device, {"gc": gc}))
            atomic_torch(folder / "hard_start_weights.pt", theta.cpu())
            del accepted, selected, batch
        progress(out, "短预算SMC", f"副本{replica + 1}/{cfg['smc_replicas']}；SGD矩阵已完成。")
        tick, last_log = time.perf_counter(), time.perf_counter()
        accept_history = []
        stop = "time_budget"
        while node_index < len(cfg["smc_nodes"]) and time.perf_counter() - tick < cfg["smc_seconds"]:
            target = cfg["smc_nodes"][node_index]
            median = float(torch.quantile(loss, .5))
            next_upper = max(target, median)
            if next_upper >= upper:
                next_upper = max(target, float(loss.min()))
            keep = torch.where(loss <= next_upper)[0]
            if not len(keep):
                stop = "empty_survivors"
                break
            fraction = len(keep) / n
            k -= math.log2(fraction)
            if len(keep) < n:
                chosen = keep[torch.randint(len(keep), (n,), generator=generator, device=device)]
                theta = theta[chosen].clone()
                loss = loss[chosen].clone()
                ancestry = ancestry[chosen].clone()
            upper = next_upper
            level += 1
            accepts = []
            for sweep in range(cfg["smc_adapt_sweeps"] + cfg["smc_fixed_sweeps"]):
                for block in range(6):
                    proposal = theta.clone()
                    if block < 5:
                        index = LAYERS[block][0]
                    else:
                        index = WEIGHT_INDICES
                    old = theta[:, index]
                    noise = torch.randn(old.shape, generator=generator, device=device)
                    proposal[:, index] = reflect(old + scales[block] * noise, scale)
                    candidate_loss, candidate_hard = network_energy(proposal, x, y, cfg)
                    good = candidate_hard & torch.isfinite(candidate_loss) & (candidate_loss <= upper)
                    theta[good] = proposal[good]
                    loss[good] = candidate_loss[good]
                    acceptance = float(good.double().mean())
                    accepts.append(acceptance)
                    if sweep < cfg["smc_adapt_sweeps"]:
                        scales[block] = min(scale, max(1e-5 * scale, scales[block] * math.exp(0.6 * (acceptance - 0.35))))
            accept_history.append({"level": level, "upper": upper, "survival": fraction, "K_given_hard": k,
                                   "acceptance": float(np.mean(accepts)), "steps": scales.copy()})
            if upper == target:
                label = "node_" + format(target, ".6g").replace(".", "p")
                rows.append(save_smc_node(folder, label, theta, loss, upper, k, level, ancestry, cfg, data, device,
                                         {"sampling_seconds": time.perf_counter() - tick, "acceptance": float(np.mean(accepts))}))
                node_index += 1
                log(f"SMC r{replica} HIT BCE<={target:g} K|hard={k:.2f}，祖先{int(ancestry.unique().numel())}", out)
            if time.perf_counter() - last_log >= 30:
                log(f"SMC r{replica} level={level} BCE<={upper:.6g} K|hard={k:.2f} acc={np.mean(accepts):.3f}", out)
                last_log = time.perf_counter()
                atomic_torch(checkpoint, {"theta": theta.cpu(), "loss": loss.cpu(), "ancestry": ancestry.cpu(),
                                         "rng": generator.get_state(), "rows": rows, "level": level, "k": k,
                                         "node_index": node_index, "upper": upper, "scales": scales, "gc": gc})
        if node_index == len(cfg["smc_nodes"]):
            stop = "target_reached"
        rows.append(save_smc_node(folder, "endpoint", theta, loss, upper, k, level, ancestry, cfg, data, device,
                                 {"sampling_seconds": time.perf_counter() - tick}))
        atomic_torch(checkpoint, {"theta": theta.cpu(), "loss": loss.cpu(), "ancestry": ancestry.cpu(),
                                 "rng": generator.get_state(), "rows": rows, "level": level, "k": k,
                                 "node_index": node_index, "upper": upper, "scales": scales, "gc": gc})
        atomic_json(folder / "levels.json", accept_history)
        atomic_json(folder / "final.json", {"status": stop, "gc": gc, "nodes": rows,
                                            "sampling_seconds": time.perf_counter() - tick,
                                            "total_replica_seconds": time.perf_counter() - replica_started,
                                            "p_hard_calibration": prior["p_hard"]})
        log(f"SMC r{replica} END {stop} BCE<={upper:.6g} 用时{time.perf_counter()-tick:.1f}s", out)
        del theta, loss, ancestry
        torch.cuda.empty_cache()
    progress(out, "运行完成，待独立分析", "SGD对照、独立先验及4个短预算SMC作业已记录；尚未认证matched-BCE结论。")


def preflight(cfg=CONFIG):
    test_cfg = copy.deepcopy(cfg)
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    test_cfg["output_dir"] = str(HERE / "preflight" / version)
    out, device = setup(test_cfg)
    data = prepare_data(test_cfg, out)
    checks = {}
    assert PARAMETERS == 11074 and len(BIAS_INDICES) == 115
    checks["parameter_count"] = True
    g = torch.Generator().manual_seed(91803)
    theta = sample_prior(2, 0.2, g, "cpu").double().requires_grad_()
    selected = torch.tensor([0, 8, 1, 9])
    x, y = data["train_x"][selected].double(), data["train_y"][selected]
    refs = []
    for row in theta:
        modules = []
        for i, (_, _, co, ci, kernel) in enumerate(LAYERS):
            if i == 2:
                modules.append(torch.nn.Flatten())
                layer = torch.nn.Linear(ci, co, dtype=torch.float64)
            elif i > 2:
                layer = torch.nn.Linear(ci, co, dtype=torch.float64)
            else:
                layer = torch.nn.Conv2d(ci, co, kernel, dtype=torch.float64)
            w_slice, b_slice = LAYERS[i][:2]
            layer.weight.data.copy_(row[w_slice].reshape_as(layer.weight))
            layer.bias.data.copy_(row[b_slice])
            modules.append(layer)
            if i < 4:
                modules.append(torch.nn.ReLU())
            if i < 2:
                modules.append(torch.nn.MaxPool2d(2))
        refs.append(torch.nn.Sequential(*modules))
    z = forward(theta, x)
    expected = torch.stack([model(x) for model in refs])
    checks["forward_max_abs"] = float((z - expected).abs().max().detach())
    assert checks["forward_max_abs"] < 1e-12
    checks["original_grouped_max_abs"] = float((z-forward_grouped(theta, x)).abs().max().detach())
    assert checks["original_grouped_max_abs"] < 1e-12
    native_ce(z, y).sum().backward()
    grads = []
    for model in refs:
        F.cross_entropy(model(x), y).backward()
        grads.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()]))
    checks["gradient_max_abs"] = float((theta.grad - torch.stack(grads)).abs().max())
    assert checks["gradient_max_abs"] < 1e-12
    paired = torch.nn.Parameter(theta.detach().clone())
    batched_optimizer = torch.optim.SGD([paired], lr=.01)
    reference_optimizers = [torch.optim.SGD(model.parameters(), lr=.01) for model in refs]
    for _ in range(10):
        batched_optimizer.zero_grad(set_to_none=True)
        native_ce(forward(paired, x), y).sum().backward()
        batched_optimizer.step()
        for model, optimizer in zip(refs, reference_optimizers):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
    reference_weights = torch.stack([torch.cat([p.detach().reshape(-1) for p in model.parameters()]) for model in refs])
    checks["ten_sgd_steps_max_abs"] = float((paired.detach() - reference_weights).abs().max())
    assert checks["ten_sgd_steps_max_abs"] < 1e-11
    checks["stable_loss_max_abs"] = float((native_ce(z, y) - score(z, y)[0]).abs().max().detach())
    assert checks["stable_loss_max_abs"] < 1e-12
    changed = theta.detach().clone()
    changed[1] += 0.02
    assert torch.equal(forward(changed, x)[0], z[0].detach())
    checks["model_independence"] = True
    scaled = forward(theta.detach() * 5, x)
    checks["homogeneity_error"] = float((scaled - z.detach() * (5**5)).abs().max())
    assert checks["homogeneity_error"] < 1e-9
    for freeze in [True, False]:
        value = torch.nn.Parameter(theta.detach().float().to(device))
        opt = torch.optim.SGD([value], lr=.01)
        native_ce(forward(value, x.float().to(device)), y.to(device)).sum().backward()
        if freeze:
            value.grad[:, BIAS_INDICES] = 0
        opt.step()
        bias_changed = bool((value[:, BIAS_INDICES] != 0).any())
        assert bias_changed != freeze
    checks["bias_control"] = True
    toy = torch.rand(100000, generator=torch.Generator().manual_seed(920))*2-1
    before = toy.square().mean()
    for _ in range(5):
        toy = reflect(toy + torch.randn(toy.shape, generator=torch.Generator().manual_seed(930+_))*.4, 1.)
    checks["reflection_second_moment"] = float(toy.square().mean())
    assert float(toy.abs().max()) <= 1 and abs(float(toy.square().mean()) - 1/3) < .006
    checks["reflection_mean"] = float(toy.mean())
    assert abs(checks["reflection_mean"]) < .01 and abs(float(before)-1/3) < .006
    source = HERE / "upstream/utils/datasets_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Dataset_Mnist_Cifar10")
    namespace = {"torch": torch, "torchvision": torchvision, "transforms": torchvision.transforms,
                 "DataLoader": torch.utils.data.DataLoader, "DATA_DIR": cfg["mnist_root"]}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    author = namespace["Dataset_Mnist_Cifar10"](16, threads=0, train_count=16, classes_seed=202, permutation_seed=202)
    assert torch.equal(author.train.dataset.indices, data["train_ids"])
    assert torch.equal(author.test.dataset.indices, data["test_ids"])
    checks["author_test_data_max_abs"] = float((author.test_all_data - data["test_x"]).abs().max())
    assert checks["author_test_data_max_abs"] < 1e-6
    assert torch.equal(author.test_all_labels, data["test_y"])
    checks["author_dataset_reconstruction"] = True
    check_theta = sample_prior(8, .2, torch.Generator(device=device).manual_seed(99), device)
    a = forward(check_theta, data["train_x"].to(device))
    grouped = forward_grouped(check_theta, data["train_x"].to(device))
    checks["cuda_fast_grouped_max_abs"] = float((a-grouped).abs().max())
    assert checks["cuda_fast_grouped_max_abs"] < 2e-5
    b = forward(check_theta.double(), data["train_x"].double().to(device))
    checks["cuda_float32_float64_max_abs"] = float((a.double()-b).abs().max())
    assert checks["cuda_float32_float64_max_abs"] < 2e-5
    checks["cuda_float32_float64_sign_changes"] = int((a.argmax(-1) != b.argmax(-1)).sum())
    benches = []
    for count in [32, 128, 512]:
        value = sample_prior(count, .2, torch.Generator(device=device).manual_seed(119), device)
        for _ in range(2):
            forward(value, data["train_x"].to(device))
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        for _ in range(10):
            forward(value, data["train_x"].to(device))
        torch.cuda.synchronize(device)
        benches.append({"models": count, "seconds_per_forward": (time.perf_counter()-tick)/10})
    checks["benchmarks"] = benches
    checks["status"] = "PASS"
    atomic_json(out / "checks.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return checks


def main(cfg=CONFIG):
    preflight(cfg)
    run_training(cfg)
    run_smc(cfg)


if __name__ == "__main__":
    main()
