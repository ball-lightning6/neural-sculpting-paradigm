"""原版Power模97除法Transformer：多种正则的配对比较。

单文件，命令行或直接粘入Jupyter均可，不使用argparse，不下载数据。
原版头图设置：50%数据，两层decoder、宽128、4头、ReLU，Adam(.9,.98)，
lr=1e-3，10步warmup，无WD，最多100万步。保留官方239-token词表与EOS训练。
每个seed/条件首次评估到测试答案100%即单独保存并停止，其余继续；这不是稳定性检验。
默认8个配对seed、10个条件共80模型；已停止模型从参数批次和Adam状态中移除。

普通CE为原版数值基线；其余条件用数学等价的稳定CE，另有独立稳定CE基线。
幅度正则只使用预测答案位置，避免把简单的EOS预测和规则输出混合对齐。
EOS的原始CE保留。训练、评估、所有系数和未成功结果全部记录。
结果位于/root/autodl-tmp，Ctrl+C安全保存，重新运行自动续训。

上游模型/数据：openai/grok，commit 3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3。
下面将模型数学运算向量化并改用当前PyTorch循环，不声称逐位重演旧硬件/Lightning。

MIT License
Copyright (c) 2021 OpenAI
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
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
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Config:
    PRIME = 97
    TRAIN_FRACTION = 0.50
    DATA_SEED = 0
    MODEL_SEEDS = tuple(range(8))
    # 名称、干预种类、系数、decoupled weight decay。
    CONDITIONS = (
        ("ce_official", "official", 0.0, 0.0),
        ("ce_stable", "stable", 0.0, 0.0),
        ("gap_var_0p01", "gap_var", 0.01, 0.0),
        ("gap_var_0p1", "gap_var", 0.1, 0.0),
        ("gap_var_1", "gap_var", 1.0, 0.0),
        ("squentropy", "squentropy", 1.0, 0.0),
        ("logit_l2_0p01", "logit_l2", 0.01, 0.0),
        ("label_smoothing_0p1", "label_smoothing", 0.1, 0.0),
        ("focal_gamma2", "focal", 2.0, 0.0),
        ("adamw_wd1", "stable", 0.0, 1.0),
    )
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4
    MAX_CONTEXT_LEN = 50
    LEARNING_RATE = 1e-3
    WARMUP_STEPS = 10
    ADAM_BETAS = (0.9, 0.98)
    ADAM_EPS = 1e-8
    BATCH_ORDER_SEED = 20260901
    MAX_STEPS = 1_000_000

    DEVICE = "cuda:0"
    ALLOW_TF32 = False
    MODEL_CHUNK_SIZE = 16  # 只分块独立模型，不拆分一个模型的minibatch或改变目标。
    EVAL_BATCH_SIZE = 256
    EVAL_INTERVAL_EARLY = 100  # 前1万步。
    EVAL_INTERVAL_MIDDLE = 500  # 1万至10万步。
    EVAL_INTERVAL_LATE = 5000
    LOG_INTERVAL = 1000
    CHECKPOINT_INTERVAL = 5000
    STOP_AT_TEST_PERFECT = True
    SAVE_FIRST_TRAIN_FIT = True
    RESULT_DIR = Path("/root/autodl-tmp/results_power_mod97_regularizer_panel")
    RESUME = True
    PACKAGE_RESULTS = True
    SMOKE_TEST = False


def config_dict():
    cfg = {k.lower(): v for k, v in vars(Config).items() if k.isupper()}
    cfg["result_dir"] = str(cfg["result_dir"])
    if cfg["smoke_test"]:
        cfg.update(d_model=16, n_layers=1, n_heads=2, model_seeds=(0, 1),
                   max_steps=6, eval_batch_size=64, eval_interval_early=2,
                   log_interval=2, checkpoint_interval=2)
        cfg["result_dir"] += "_smoke"
    return cfg


def signature(cfg):
    keys = ("prime", "train_fraction", "data_seed", "model_seeds", "conditions",
            "d_model", "n_layers", "n_heads", "max_context_len", "learning_rate",
            "warmup_steps", "adam_betas", "adam_eps", "batch_order_seed", "allow_tf32",
            "stop_at_test_perfect", "eval_interval_early", "eval_interval_middle", "eval_interval_late",
            "smoke_test")
    result = {k: cfg[k] for k in keys}
    result.update(protocol="power_mod97_regularizer_panel_v1", dtype="float32",
                  upstream_commit="3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3",
                  stopping="first_observed_test_answer_perfect", regularizer_scope="answer_only",
                  batch_protocol="shuffle_without_replacement_keep_short_last_batch")
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


EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
DIV_TOKEN = "/"

VALID_OPERATOR_TOKENS = {
    "+", "-", "*", "/", "**2+", "**3+",
    "x**2+y**2_mod_97",
    "x**2+y**2+x*y_mod_97",
    "x**2+y**2+x*y+x_mod_97",
    "x**3+x*y_mod_97",
    "x**3+x*y**2+y_mod_97",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97",
    "s5", "s5conj", "s5aba", "+*", "+-", "sort", "reverse", "copy",
}


def official_vocabulary(prime: int) -> tuple[list[str], dict[str, int]]:
    permutation_tokens = ["".join(map(str, p)) for p in itertools.permutations(range(5))]
    tokens = (
        [EOS_TOKEN, EQ_TOKEN]
        + sorted(VALID_OPERATOR_TOKENS)
        + [str(i) for i in range(prime)]
        + permutation_tokens
    )
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("官方token列表意外出现重复。")
    return tokens, {token: i for i, token in enumerate(tokens)}


def make_official_division_data(
    prime: int,
    data_seed: int,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """精确复现openai/grok ArithmeticDataset.make_data('/')的顺序与shuffle。"""
    tokens, stoi = official_vocabulary(prime)
    rows: list[list[int]] = []
    semantic_rows: list[tuple[int, int, int]] = []

    # 上游先遍历变量a,b；division分支令c=a，再令a=b*c mod p。
    for original_a in range(prime):
        for b in range(prime):
            if b == 0:
                continue
            c = original_a
            a = (b * c) % prime
            equation = [EOS_TOKEN, str(a), DIV_TOKEN, str(b), EQ_TOKEN, str(c), EOS_TOKEN]
            rows.append([stoi[token] for token in equation])
            semantic_rows.append((a, b, c))

    order = np.arange(len(rows), dtype=np.int64)
    rng = np.random.RandomState(seed=data_seed)
    rng.shuffle(order)
    encoded = torch.tensor(np.asarray(rows, dtype=np.int64)[order], dtype=torch.long)
    semantics = np.asarray(semantic_rows, dtype=np.int64)[order]
    return encoded, semantics, tokens


# ---------------------------------------------------------------------------
# OpenAI/grok Transformer（MIT；去掉未使用的weight_noise分支）
# ---------------------------------------------------------------------------


class OfficialAttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int) -> None:
        super().__init__()
        self.d_key = d_key
        self.wq = nn.Linear(d_model, d_key, bias=False)
        self.wk = nn.Linear(d_model, d_key, bias=False)
        self.wv = nn.Linear(d_model, d_key, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)
        attn = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_key)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        return torch.matmul(torch.softmax(attn, dim=-1), values)


class OfficialMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        d_key = d_model // heads
        self.heads = nn.ModuleList(
            [OfficialAttentionHead(d_model, d_key) for _ in range(heads)]
        )
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.wo(torch.cat([head(x, mask) for head in self.heads], dim=-1))


class OfficialFFN(nn.Module):
    def __init__(self, d_model: int, non_linearity: str) -> None:
        super().__init__()
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if non_linearity not in activations:
            raise ValueError(f"不支持activation={non_linearity}")
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            activations[non_linearity](),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OfficialDecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, non_linearity: str) -> None:
        super().__init__()
        self.attn = OfficialMultiHeadAttention(d_model, heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = OfficialFFN(d_model, non_linearity)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x + self.attn(x, mask))
        return self.ffn_norm(x + self.ffn_drop(self.ffn(x)))


class OfficialTransformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_model: int,
        dropout: float,
        max_context_len: int,
        vocab_len: int,
        non_linearity: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, d_model)
        self.blocks = nn.ModuleList(
            [
                OfficialDecoderBlock(d_model, n_heads, dropout, non_linearity)
                for _ in range(n_layers)
            ]
        )
        self.output = nn.Linear(d_model, vocab_len, bias=False)
        self.register_buffer(
            "position_encoding",
            self._position_encoding(max_context_len, d_model),
        )
        self.register_buffer(
            "causal_mask",
            torch.ones(max_context_len, max_context_len).tril(),
        )

    @staticmethod
    def _position_encoding(context_len: int, d_model: int) -> torch.Tensor:
        rows = [
            torch.tensor(
                [
                    math.sin(pos / (10000 ** (i / d_model)))
                    if i % 2 == 0
                    else math.cos(pos / (10000 ** ((i - 1) / d_model)))
                    for i in range(d_model)
                ],
                dtype=torch.float32,
            )
            for pos in range(context_len)
        ]
        return torch.stack(rows, dim=0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        context_len = indices.shape[-1]
        x = self.embedding(indices) + self.position_encoding[:context_len]
        mask = self.causal_mask[:context_len, :context_len]
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x)


class PairedBatchStream:
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.permutation = torch.randperm(self.size, generator=self.generator)
        self.cursor = 0
        self.epoch = 0

    def next_indices(self) -> torch.Tensor:
        if self.cursor >= self.size:
            self.epoch += 1
            self.permutation = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
        end = min(self.cursor + self.batch_size, self.size)
        indices = self.permutation[self.cursor:end]
        self.cursor = end
        return indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation,
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["size"]) != self.size or int(state["batch_size"]) != self.batch_size:
            raise ValueError("checkpoint中的batch stream与当前协议不一致。")
        self.generator.set_state(state["generator_state"])
        self.permutation = state["permutation"].clone()
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])


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


def make_dataset(cfg):
    encoded, semantics, tokens = make_official_division_data(cfg["prime"], cfg["data_seed"])
    if cfg["smoke_test"]:
        encoded, semantics = encoded[:128], semantics[:128]
    n = round(len(encoded) * cfg["train_fraction"])
    return {"sequences": encoded.numpy(), "semantics": semantics, "tokens": tokens,
            "train_count": n, "batch_size": min(512, math.ceil(n / 2))}


def model_manifest(cfg):
    return [{"model_id": i * len(cfg["model_seeds"]) + j, "condition": name,
             "kind": kind, "coefficient": strength, "weight_decay": wd, "seed": seed}
            for i, (name, kind, strength, wd) in enumerate(cfg["conditions"])
            for j, seed in enumerate(cfg["model_seeds"])]


def single_model(cfg, vocabulary_size):
    return OfficialTransformer(cfg["n_layers"], cfg["n_heads"], cfg["d_model"],
                               0.0, cfg["max_context_len"], vocabulary_size, "relu")


class BatchedTransformer(nn.Module):
    """每个参数的第一维对应一个独立网络；共享的只有输入，不共享可训练参数。"""

    def __init__(self, cfg, vocabulary_size, active_ids):
        super().__init__()
        self.cfg = cfg
        self.manifest = model_manifest(cfg)
        self.active_ids = list(active_ids)
        states = {}
        for gid in self.active_ids:
            seed = self.manifest[gid]["seed"]
            if seed not in states:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    reference = single_model(cfg, vocabulary_size)
                states[seed] = {name: p.detach().clone() for name, p in reference.named_parameters()}
        self.names = list(next(iter(states.values())))
        self.params = nn.ParameterDict({
            self.key(name): nn.Parameter(torch.stack([states[self.manifest[i]["seed"]][name] for i in active_ids]))
            for name in self.names})
        self.register_buffer("position_encoding", reference.position_encoding.clone())
        self.register_buffer("causal_mask", reference.causal_mask.clone())
        self.register_buffer("decay_coefficients", torch.tensor(
            [self.manifest[i]["weight_decay"] for i in active_ids], dtype=torch.float32))
        self.refresh_layout()

    @staticmethod
    def key(name):
        return name.replace(".", "__")

    def p(self, name, selection=None):
        value = self.params[self.key(name)]
        return value if selection is None else value[selection]

    def refresh_layout(self):
        self.layout = []
        start = 0
        while start < len(self.active_ids):
            spec = self.manifest[self.active_ids[start]]
            end = start + 1
            while end < len(self.active_ids) and self.manifest[self.active_ids[end]]["condition"] == spec["condition"]:
                end += 1
            self.layout.append((start, end, spec))
            start = end

    @staticmethod
    def linear(x, weight):
        result = torch.bmm(x.reshape(x.shape[0], -1, x.shape[-1]), weight.transpose(1, 2))
        return result.reshape(*x.shape[:-1], weight.shape[1])

    def norm(self, x, prefix, selection=None):
        z = F.layer_norm(x, (self.cfg["d_model"],), eps=1e-5)
        return z * self.p(prefix + ".weight", selection)[:, None, None] + self.p(prefix + ".bias", selection)[:, None, None]

    def forward(self, indices, all_positions=False, model_slice=None):
        h = self.p("embedding.weight", model_slice)[:, indices] + self.position_encoding[:indices.shape[-1]]
        models, batch, length, width = h.shape
        heads = self.cfg["n_heads"]
        key_size = width // heads
        mask = self.causal_mask[:length, :length] == 0
        for layer in range(self.cfg["n_layers"]):
            prefix = f"blocks.{layer}"
            projections = []
            for which in ("wq", "wk", "wv"):
                weight = torch.cat([self.p(f"{prefix}.attn.heads.{head}.{which}.weight", model_slice)
                                    for head in range(heads)], dim=1)
                projected = self.linear(h, weight).reshape(models, batch, length, heads, key_size)
                projections.append(projected.permute(0, 1, 3, 2, 4))
            q, k, v = projections
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(key_size)
            attention = scores.masked_fill(mask, -torch.inf).softmax(-1)
            context = torch.matmul(attention, v).permute(0, 1, 3, 2, 4).reshape(models, batch, length, width)
            h = self.norm(h + self.linear(context, self.p(prefix + ".attn.wo.weight", model_slice)), prefix + ".attn_norm", model_slice)
            feedforward = self.linear(F.relu(self.linear(h, self.p(prefix + ".ffn.net.0.weight", model_slice))),
                                      self.p(prefix + ".ffn.net.2.weight", model_slice))
            h = self.norm(h + feedforward, prefix + ".ffn_norm", model_slice)
        return self.linear(h if all_positions else h[:, :, 4:6], self.p("output.weight", model_slice))

    def export_single(self, local_index):
        state = {name: self.p(name)[local_index].detach().cpu().clone() for name in self.names}
        state.update(position_encoding=self.position_encoding.detach().cpu().clone(),
                     causal_mask=self.causal_mask.detach().cpu().clone())
        return state


def create_optimizer(model, cfg):
    # 不手写Adam；不同模型的decoupled WD在step前分别施加，梯度仍由同一库实现更新。
    return torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                            betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"], foreach=False)


def stable_ce(z, y):
    target = y.expand(z.shape[:-1])
    chosen = z.gather(-1, target[..., None])
    relative = (z - chosen).scatter(-1, target[..., None], -torch.inf)
    return F.softplus(torch.logsumexp(relative, dim=-1))


def objective_per_model(z, rhs_target, layout):
    values = []
    for start, end, spec in layout:
        part = z[start:end]
        y = rhs_target[None].expand(end - start, -1, -1)
        kind, strength = spec["kind"], spec["coefficient"]
        if kind == "official":
            loss = F.cross_entropy(part.reshape(-1, part.shape[-1]), y.reshape(-1), reduction="none")
            values.append(loss.reshape(end - start, -1).mean(-1))
            continue
        point_ce = stable_ce(part, y)
        answer_ce, eos_ce = point_ce[:, :, 0], point_ce[:, :, 1]
        answer = part[:, :, 0]
        chosen = answer.gather(-1, rhs_target[None, :, 0, None].expand(end - start, -1, 1))
        base = (answer_ce.mean(-1) + eos_ce.mean(-1)) / 2
        if kind == "gap_var":
            base = base + strength * gap_variance(answer, rhs_target[:, 0])
        elif kind == "squentropy":
            other = answer.scatter(-1, rhs_target[None, :, 0, None].expand(end - start, -1, 1), 0.)
            base = base + strength * other.square().sum(-1).mean(-1) / (answer.shape[-1] - 1)
        elif kind == "logit_l2":
            base = base + strength * answer.square().mean((1, 2))
        elif kind == "label_smoothing":
            # 只平滑答案标签，EOS仍保留原版硬标签CE。
            base = base + .5 * strength * (chosen.squeeze(-1) - answer.mean(-1)).mean(-1)
        elif kind == "focal":
            focal = (-torch.expm1(-answer_ce)).pow(strength) * answer_ce
            base = .5 * (focal.mean(-1) + eos_ce.mean(-1))
        elif kind != "stable":
            raise ValueError(f"未知目标：{kind}")
        values.append(base)
    return torch.cat(values)


def train_step(model, optimizer, stream, sequences, cfg, step):
    lr = cfg["learning_rate"] * min(step / max(cfg["warmup_steps"], 1), 1.)
    for group in optimizer.param_groups:
        group["lr"] = lr
    indices = stream.next_indices().to(sequences.device)
    batch = sequences[indices]
    optimizer.zero_grad(set_to_none=True)
    pieces = []
    for start in range(0, len(model.active_ids), cfg["model_chunk_size"]):
        end = min(start + cfg["model_chunk_size"], len(model.active_ids))
        layout = [(max(a, start) - start, min(b, end) - start, spec)
                  for a, b, spec in model.layout if max(a, start) < min(b, end)]
        logits = model(batch[:, :-1], model_slice=slice(start, end))
        loss = objective_per_model(logits, batch[:, 5:7], layout)
        loss.sum().backward()
        pieces.append(loss.detach())
    with torch.no_grad():
        for p in model.parameters():
            factor = (1 - lr * model.decay_coefficients).reshape(-1, *([1] * (p.ndim - 1)))
            p.mul_(factor)
    optimizer.step()
    return torch.cat(pieces)


def compact_active(model, optimizer, keep, cfg):
    """连同Adam的一二阶矩移除已停止模型，不能只清零梯度后任其动量继续更新。"""
    index = torch.tensor(keep, dtype=torch.long, device=model.position_encoding.device)
    saved = []
    for key in list(model.params.keys()):
        old = model.params[key]
        new = nn.Parameter(old.detach().index_select(0, index))
        state = {}
        for name, value in optimizer.state.get(old, {}).items():
            if torch.is_tensor(value):
                state[name] = value.index_select(0, index).clone() if value.shape == old.shape else value.clone()
            else:
                state[name] = value
        model.params[key] = new
        saved.append((new, state))
    model.decay_coefficients = model.decay_coefficients.index_select(0, index)
    model.active_ids = [model.active_ids[i] for i in keep]
    model.refresh_layout()
    new_optimizer = create_optimizer(model, cfg)
    new_optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
    for p, state in saved:
        new_optimizer.state[p] = state
    return new_optimizer


@torch.no_grad()
def collect_outputs(model, sequences, batch_size):
    count = len(model.active_ids)
    vocabulary = model.p("output.weight").shape[1]
    output = np.empty((count, len(sequences), 2, vocabulary), np.float32)
    for first in range(0, count, model.cfg["model_chunk_size"]):
        last = min(first + model.cfg["model_chunk_size"], count)
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            z = model(sequences[start:end, :-1], model_slice=slice(first, last))
            if not torch.isfinite(z).all().item():
                raise FloatingPointError("评估出现非有限输出，保留此前的完整断点，不将其当成实验阴性。")
            output[first:last, start:end] = z.cpu().numpy()
    return output


def numpy_ce(z, target):
    chosen = np.take_along_axis(z, target[..., None], -1)
    relative = z - chosen
    np.put_along_axis(relative, target[..., None], -np.inf, -1)
    maximum = relative.max(-1)
    if maximum.max() < 500:
        return np.log1p(np.exp(relative).sum(-1))
    maximum = np.maximum(maximum, 0.)
    return maximum + np.log(np.exp(-maximum) + np.exp(relative - maximum[..., None]).sum(-1))


def metrics(raw, target, spec):
    z = raw.astype(np.float64)
    ce = numpy_ce(z, target)
    prediction = z.argmax(-1)
    answer = z[:, 0]
    label = target[:, 0]
    chosen = answer[np.arange(len(answer)), label]
    other_mask = np.arange(answer.shape[-1])[None] != label[:, None]
    others = answer[other_mask].reshape(len(answer), -1)
    gaps = chosen[:, None] - others
    mean_rhs_ce = float(ce.mean())
    kind, strength = spec["kind"], spec["coefficient"]
    added = 0.
    if kind == "gap_var":
        added = strength * float(gaps.var())
    elif kind == "squentropy":
        added = strength * float(np.mean(others ** 2))
    elif kind == "logit_l2":
        added = strength * float(np.mean(answer ** 2))
    elif kind == "label_smoothing":
        added = .5 * strength * float(np.mean(chosen - answer.mean(-1)))
    elif kind == "focal":
        added = .5 * float(np.mean(((-np.expm1(-ce[:, 0])) ** strength - 1) * ce[:, 0]))
    native = F.cross_entropy(torch.from_numpy(np.ascontiguousarray(raw)).reshape(-1, raw.shape[-1]),
                             torch.from_numpy(np.ascontiguousarray(target)).reshape(-1),
                             reduction="none").numpy().reshape(len(target), 2)
    answer_wrong = prediction[:, 0] != target[:, 0]
    eos_wrong = prediction[:, 1] != target[:, 1]
    return {
        "answer_accuracy": float(1 - answer_wrong.mean()),
        "answer_error_count": int(answer_wrong.sum()),
        "eos_accuracy": float(1 - eos_wrong.mean()),
        "official_row_accuracy": float(np.all(prediction == target, axis=-1).mean()),
        "raw_answer_ce": float(ce[:, 0].mean()), "raw_eos_ce": float(ce[:, 1].mean()),
        "raw_rhs_ce": mean_rhs_ce, "native_fp32_rhs_ce": float(native.mean()),
        "native_fp32_answer_ce_zero_fraction": float((native[:, 0] == 0).mean()),
        "diagnostic_total_objective": mean_rhs_ce + added,
        "answer_objective_adjustment": added,
        "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
        "gap_variance": float(gaps.var()),
        "between_sample_gap_variance": float(gaps.mean(-1).var()),
        "within_sample_gap_variance": float(gaps.var(-1).mean()),
        "true_logit_mean": float(chosen.mean()), "true_logit_std": float(chosen.std()),
        "wrong_logit_mean": float(others.mean()), "wrong_logit_std": float(others.std()),
        "min_top_gap": float((chosen - others.max(-1)).min()),
    }


def evaluation_due(step, cfg):
    if step in {0, 1, 2, 5, 10, 20, 50, cfg["max_steps"]}:
        return True
    interval = (cfg["eval_interval_early"] if step <= 10_000 else
                cfg["eval_interval_middle"] if step <= 100_000 else cfg["eval_interval_late"])
    return step % interval == 0


def save_snapshot(out, model, local, spec, raw, row, reason, cfg, state):
    stem = f"model_{spec['model_id']:03d}_{reason}_step_{state['step']:07d}"
    weight_file = f"snapshots/{stem}.pt"
    output_file = f"snapshots/{stem}.npz"
    atomic_torch(out / weight_file, {"model": model.export_single(local), "spec": spec,
                                   "step": state["step"], "config": cfg, "metrics": row,
                                   "model_type": "OfficialTransformer", "reason": reason})
    atomic_npz(out / output_file, raw_rhs_outputs=raw, model_id=np.array(spec["model_id"]),
               step=np.array(state["step"]))
    state["artifacts"].extend((weight_file, output_file))
    return weight_file


def evaluate_and_select(model, optimizer, sequences, data, cfg, state, out, elapsed, final_budget=False):
    if not model.active_ids:
        return optimizer
    output = collect_outputs(model, sequences, cfg["eval_batch_size"])
    n = data["train_count"]
    targets = data["sequences"][:, 5:7]
    prediction = output.argmax(-1).astype(np.uint16)
    step = state["step"]
    name = f"predictions/step_{step:07d}.npz"
    atomic_npz(out / name, active_ids=np.asarray(model.active_ids), rhs_predictions=prediction,
               step=np.array(step))
    state["artifacts"].append(name)
    keep = []
    for local, gid in enumerate(model.active_ids):
        spec = model.manifest[gid]
        row = {**spec, "step": step, "elapsed_active_s": elapsed,
               "full_rule_exact": bool(np.array_equal(prediction[local, :, 0], targets[:, 0])),
               "answer_function_sha256": digest(prediction[local, :, 0])}
        for split, sl in (("train", slice(0, n)), ("test", slice(n, None))):
            row.update({f"{split}_{k}": v for k, v in metrics(output[local, sl], targets[sl], spec).items()})
        state["rows"].append(row)
        if row["train_answer_error_count"] == 0 and gid not in state["first_fit"]:
            state["first_fit"][gid] = step
            if cfg["save_first_train_fit"]:
                save_snapshot(out, model, local, spec, output[local], row, "train_fit", cfg, state)
        if row["test_answer_error_count"] == 0 and cfg["stop_at_test_perfect"]:
            filename = save_snapshot(out, model, local, spec, output[local], row, "test_perfect", cfg, state)
            state["finished"][gid] = {"step": step, "reason": "first_observed_test_answer_100pct",
                                      "snapshot": filename, "row": row}
            log(f"STOP {spec['condition']} seed={spec['seed']} step={step:,} | 测试答案100% | "
                f"训练答案={row['train_answer_accuracy']:.3%} | 答题CE={row['train_raw_answer_ce']:.3g}")
        else:
            keep.append(local)
            if final_budget:
                save_snapshot(out, model, local, spec, output[local], row, "budget_end", cfg, state)
    if len(keep) < len(model.active_ids):
        optimizer = compact_active(model, optimizer, keep, cfg)
    state["active_ids"] = list(model.active_ids)
    state["last_evaluation"] = step
    latest = {r["model_id"]: r for r in state["rows"]}
    for name, _, _, _ in cfg["conditions"]:
        group = [r for r in latest.values() if r["condition"] == name]
        stopped = sum(r["model_id"] in state["finished"] for r in group)
        log(f"{name} step={step:,} | 已满分停止={stopped}/{len(group)} | "
            f"train/test答案={np.mean([r['train_answer_accuracy'] for r in group]):.3%}/"
            f"{np.mean([r['test_answer_accuracy'] for r in group]):.3%} | "
            f"答题CE={np.median([r['train_raw_answer_ce'] for r in group]):.3g} | "
            f"差距均值/std={np.median([r['train_gap_mean'] for r in group]):.3g}/"
            f"{np.median([r['train_gap_std'] for r in group]):.3g}")
    return optimizer


def checkpoint_save(path, model, optimizer, stream, cfg, state):
    atomic_torch(path, {"signature": signature(cfg), "state": state,
                       "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "optimizer": optimizer.state_dict(), "batch_stream": stream.state_dict(),
                       "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})


def checkpoint_restore(payload, model, optimizer, stream, cfg):
    if payload["signature"] != signature(cfg):
        raise ValueError("断点科学配置不同，不能混用结果；需要新实验时请换RESULT_DIR。")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    stream.load_state_dict(payload["batch_stream"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])


def summary_tables(cfg, state):
    last = {r["model_id"]: r for r in state["rows"]}
    runs = []
    for spec in model_manifest(cfg):
        gid = spec["model_id"]
        row = last.get(gid)
        finished = state["finished"].get(gid)
        runs.append({**spec, "first_train_fit_step": state["first_fit"].get(gid),
                     "first_test_perfect_step": finished["step"] if finished else None,
                     "status": "test_perfect_stopped" if finished else "active_or_budget_limited",
                     "last_step": row["step"] if row else 0,
                     "test_answer_accuracy": row["test_answer_accuracy"] if row else None,
                     "train_answer_accuracy": row["train_answer_accuracy"] if row else None,
                     "train_raw_answer_ce": row["train_raw_answer_ce"] if row else None,
                     "train_raw_rhs_ce": row["train_raw_rhs_ce"] if row else None,
                     "train_gap_std": row["train_gap_std"] if row else None,
                     "full_rule_exact": row["full_rule_exact"] if row else None})
    conditions = []
    for name, _, _, _ in cfg["conditions"]:
        group = [r for r in runs if r["condition"] == name]
        successes = [r for r in group if r["first_test_perfect_step"] is not None]
        conditions.append({"condition": name, "model_count": len(group), "test_perfect_count": len(successes),
                           "median_step_among_successes": float(np.median([r["first_test_perfect_step"] for r in successes])) if successes else None,
                           "mean_last_test_answer_accuracy": float(np.mean([r["test_answer_accuracy"] for r in group])) if all(r["test_answer_accuracy"] is not None for r in group) else None})
    return runs, conditions


def persist(out, cfg, state, status):
    csv_write(out / "trajectory.csv", state["rows"])
    runs, conditions = summary_tables(cfg, state)
    csv_write(out / "run_summary.csv", runs)
    csv_write(out / "condition_summary.csv", conditions)
    matched = []
    for spec in model_manifest(cfg):
        rows = [r for r in state["rows"] if r["model_id"] == spec["model_id"]]
        for level in (1., .3, .1, .03, .01, .001, 1e-4, 1e-6, 1e-8):
            crossed = next((r for r in rows if r["train_answer_error_count"] == 0 and r["train_raw_answer_ce"] <= level), None)
            entry = {**spec, "answer_ce_threshold": level, "reached": crossed is not None}
            for key in ("step", "train_raw_answer_ce", "train_raw_rhs_ce", "test_answer_accuracy", "train_gap_std"):
                entry[key] = crossed[key] if crossed else None
            matched.append(entry)
    csv_write(out / "matched_answer_ce.csv", matched)
    atomic_json(out / "summary.json", {"status": status, "step": state["step"],
                "elapsed_active_s": state["elapsed_active_s"], "active_count": len(state["active_ids"]),
                "conditions": conditions, "notes": [
                    "测试集用于首次满分停止，结果是首次命中，不代表继续训练仍稳定。",
                    "停止看语义答案，不把EOS作为一个容易样本抬高答题准确率；原版含EOS的整行准确率另行报告。",
                    "各组同一数据划分、逐seed同一初始化和batch顺序；seed不是独立数据划分。",
                    "普通CE与稳定CE数学等价但浮点梯度可能不同，保留两条数值基线。",
                    "方差、Squentropy和logit-L2只作用于答案位置，并加到原版RHS平均CE上。",
                    "标签平滑和Focal只改变答案项，EOS保持硬标签CE。WD是decoupled参数衰减，不冒充loss中的L2。",
                    "停止模型从参数及Adam状态中移除。汇总保留其冻结后的末次结果，逐模型轨迹在停止点截断。",
                    "未恢复的模型保留到预算末，不能用只统计成功seed的中位步数代表整个条件。",
                    "共同CE门槛只取已记录评估，不是精确相同CE，也不证明静态参数体积。"]})
    atomic_json(out / "artifact_manifest.json", {"step": state["step"], "artifacts": sorted(set(state["artifacts"]))})


def make_plots(out, cfg, state):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("没有matplotlib，CSV和模型已保存，之后安装matplotlib再运行可以补图。")
        return
    conditions = cfg["conditions"]
    fig, axes = plt.subplots(math.ceil(len(conditions) / 2), 2,
                             figsize=(14, 3 * math.ceil(len(conditions) / 2)), squeeze=False)
    for ax, (name, _, _, _) in zip(axes.flat, conditions):
        for seed in cfg["model_seeds"]:
            history = [r for r in state["rows"] if r["condition"] == name and r["seed"] == seed]
            x = [r["step"] for r in history]
            ax.plot(x, [r["train_answer_accuracy"] for r in history], "--", color="#999999", alpha=.5, lw=.8)
            ax.plot(x, [r["test_answer_accuracy"] for r in history], lw=1.2, label=f"seed {seed}")
        ax.set(title=name, xlabel="Updates (curves end when stopped)", ylabel="Answer accuracy", ylim=(-.02, 1.02))
        ax.set_xscale("symlog", linthresh=100)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(conditions):]:
        ax.set_visible(False)
    fig.suptitle("Power mod97 Transformer: dashed train / colored test; first-perfect stopping")
    fig.tight_layout()
    fig.savefig(out / "per_seed_accuracy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = sorted(set([0, state["step"]] + [r["step"] for r in state["rows"]]))
    for name, _, _, _ in conditions:
        specs = [s for s in model_manifest(cfg) if s["condition"] == name]
        hits = [state["finished"][s["model_id"]]["step"] for s in specs if s["model_id"] in state["finished"]]
        axes[0].step(steps, [sum(h <= t for h in hits) / len(specs) for t in steps], where="post", label=name)
        selected = [r for r in state["rows"] if r["condition"] == name]
        axes[1].scatter([r["train_raw_answer_ce"] for r in selected],
                        [r["test_answer_accuracy"] for r in selected], s=6, alpha=.25, label=name)
    axes[0].set(xlabel="Updates", ylabel="Fraction with first perfect test result", ylim=(-.02, 1.02))
    axes[0].set_xscale("symlog", linthresh=100)
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Train answer CE (float64 diagnostic)", ylabel="Test answer accuracy", ylim=(-.02, 1.02))
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "recovery_and_matched_ce.png", dpi=160)
    plt.close(fig)


def package_results(out, state):
    archive = out.parent / (out.name + "_package.zip")
    temporary = archive.with_name(archive.name + ".tmp")
    names = set(state["artifacts"])
    names.update(p.name for p in out.iterdir() if p.is_file() and p.suffix in {".csv", ".json", ".md", ".py", ".png"})
    names.add("dataset.npz")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for name in sorted(names):
            if (out / name).is_file():
                handle.write(out / name, arcname=name)
    temporary.replace(archive)
    return archive


def main(cfg=None):
    cfg = config_dict() if cfg is None else cfg
    device = torch.device(cfg["device"])
    if not cfg["smoke_test"] and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("正式实验要求CUDA，不会静默退回CPU。")
    if cfg["prime"] != 97 or not 0 < cfg["train_fraction"] < 1:
        raise ValueError("本脚本固定模97，训练比例需在0和1之间。")
    if not cfg["model_seeds"] or len(set(cfg["model_seeds"])) != len(cfg["model_seeds"]):
        raise ValueError("seed列表不能为空或重复。")
    if cfg["d_model"] % cfg["n_heads"] or cfg["d_model"] % 2:
        raise ValueError("宽度需为偶数且被头数整除。")
    if cfg["max_context_len"] < 6:
        raise ValueError("context至少为6。")
    names = set()
    for name, kind, strength, wd in cfg["conditions"]:
        if name in names or kind not in {"official", "stable", "gap_var", "squentropy", "logit_l2", "label_smoothing", "focal"}:
            raise ValueError("条件名重复或目标类型非法。")
        names.add(name)
        if not math.isfinite(strength) or not math.isfinite(wd) or min(strength, wd) < 0:
            raise ValueError("正则参数应非负且有限。")
        if kind == "label_smoothing" and not 0 <= strength < 1:
            raise ValueError("标签平滑系数必须小于1。")
    if not names or min(cfg[k] for k in ("max_steps", "model_chunk_size", "eval_batch_size", "eval_interval_early",
                                        "eval_interval_middle", "eval_interval_late", "log_interval", "checkpoint_interval")) < 1:
        raise ValueError("条件不能为空，步数和间隔必须为正。")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_precision(cfg["allow_tf32"])
    out = Path(cfg["result_dir"])
    checkpoint = out / "latest_checkpoint.pt"
    payload = None
    if out.exists() and any(out.iterdir()):
        if cfg["resume"] and checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["signature"] != signature(cfg):
                raise ValueError("断点配置不同，需要新实验时请更换RESULT_DIR。")
        else:
            raise FileExistsError(f"结果目录已有文件但无可续跑断点，请换一个RESULT_DIR：{out}")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("predictions", "snapshots"):
        (out / sub).mkdir(exist_ok=True)
    data = make_dataset(cfg)
    all_specs = model_manifest(cfg)
    state = payload["state"] if payload else {"step": 0, "active_ids": list(range(len(all_specs))),
        "rows": [], "finished": {}, "first_fit": {}, "elapsed_active_s": 0., "artifacts": [], "last_evaluation": -1}
    if state["step"] > cfg["max_steps"]:
        raise ValueError("MAX_STEPS小于已有步数。")
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "protocol.json", {"signature": signature(cfg), "model_manifest": all_specs,
        "tokens": data["tokens"], "train_count": data["train_count"], "batch_size": data["batch_size"],
        "objective": {
            "base": "mean(CE_answer + CE_EOS)/2",
            "gap_var": "base + coefficient * Var_{i,c!=y_i}(answer_logit_y - answer_logit_c)",
            "squentropy": "base + coefficient * mean(incorrect_answer_logits**2)",
            "logit_l2": "base + coefficient * mean(all_answer_logits**2)",
            "label_smoothing": "answer target=(1-eps)*onehot + eps/vocab; EOS unchanged",
            "focal": "mean((1-p_answer)**gamma * CE_answer + CE_EOS)/2; no alpha weighting",
            "weight_decay": "decoupled AdamW decay of all weights, not a scalar loss penalty"},
        "scope_note": "除WD外，新增正则/目标变化只作用于答案预测。Squentropy等是该位置上的移植，不声称复现其原任务。",
        "original_note": "原版头图无WD、100万步；WD=1是论文多数其他实验的强基线。"})
    atomic_npz(out / "dataset.npz", sequences=data["sequences"], semantics=data["semantics"],
               train_count=np.array(data["train_count"]))
    source_meta = archive_source(out)
    atomic_json(out / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
        "numpy": np.__version__, "cuda": torch.version.cuda, "platform": platform.platform(),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU工程自检",
        "dataset_sha256": digest(data["sequences"]), **source_meta})
    (out / "说明.md").write_text(
        "# 原版Transformer多正则对照\n\n"
        "模型、token及模97除法生成遵循openai/grok公开版本；保留50%训练数据、宽128、两层、4头。\n"
        "向量化只批量执行独立网络，官方模型类同时保留，单个快照可直接载入OfficialTransformer。\n"
        "每种条件默认8个配对seed，共同训练上限100万步；测试答案首次评估100%就停止该模型。\n"
        "因此报告首次达到所需步数，不代表持续训练后仍保持100%，未恢复模型保留到预算末。\n"
        "停止模型会真正从权重和Adam状态移除；停止后的汇总保留冻结权重的末次结果。\n"
        "训练CE按原版包含答案和EOS，答案单独的CE、准确率及完整规则恢复也单列。\n"
        "新方差正则只用于答案，不混入EOS；其余类型的精确定义见protocol.json。\n"
        "普通CE和数学等价稳定CE两条基线用于区分极低loss浮点问题。默认关闭TF32、无混合精度。\n"
        "测试标签只用于评估和用户指定的停止，不用于梯度；每种条件系数均预先指定。\n"
        "matched_answer_ce.csv保留已记录的共同门槛快照；停止后的更深CE不可比较，未达到会明确列出。\n"
        "trajectory.csv为逐模型曲线，run_summary.csv记录每个模型首次训练全对及首次测试满分。\n"
        "snapshots保存首次训练全对、首次测试满分或预算终点的权重与全输入raw RHS输出。\n"
        "中断或恢复不清空结果；下载包不含大号latest_checkpoint.pt，请把断点留在AutoDL实例。\n",
        encoding="utf-8")
    if not state["active_ids"]:
        persist(out, cfg, state, "all_test_perfect_stopped")
        make_plots(out, cfg, state)
        log("全部模型已在先前运行中达到测试满分，不重复训练。")
        if cfg["package_results"]:
            log(f"结果包：{package_results(out, state)}")
        return
    model = BatchedTransformer(cfg, len(data["tokens"]), state["active_ids"]).to(device)
    optimizer = create_optimizer(model, cfg)
    stream = PairedBatchStream(data["train_count"], data["batch_size"], cfg["batch_order_seed"])
    if payload:
        checkpoint_restore(payload, model, optimizer, stream, cfg)
        del payload
        log(f"续跑step={state['step']:,}，活动模型={len(model.active_ids)}")
    else:
        audits = []
        for seed in cfg["model_seeds"]:
            local = [i for i, gid in enumerate(model.active_ids) if all_specs[gid]["seed"] == seed]
            hashes = []
            for i in local:
                hasher = hashlib.sha256()
                for name in model.names:
                    hasher.update(model.p(name)[i].detach().cpu().contiguous().numpy().tobytes())
                hashes.append(hasher.hexdigest())
            if len(set(hashes)) != 1:
                raise AssertionError("跨条件初始化不一致。")
            audits.append({"seed": seed, "initialization_sha256": hashes[0], "identical_across_conditions": True})
        atomic_json(out / "pairing_audit.json", {"initialization": audits, "batch_order_seed": cfg["batch_order_seed"],
                    "batch_note": "所有活动模型逐步使用同一minibatch，停止后不再更新，不改变其余模型的batch。"})
    per_model = sum(p.numel() for p in model.parameters()) // len(model.active_ids)
    log(f"=== Power模97原版架构 | {cfg['n_layers']}层/宽{cfg['d_model']}/{cfg['n_heads']}头 | 参数={per_model:,} ===")
    log(f"train/test={data['train_count']}/{len(data['sequences'])-data['train_count']} | "
        f"条件={len(cfg['conditions'])} x seeds={len(cfg['model_seeds'])} | 预算={cfg['max_steps']:,}步")
    log("每个模型首次评估测试答案100%即保存停止；其他继续；Ctrl+C请求安全暂停。")
    sequences = torch.from_numpy(data["sequences"]).long().to(device)
    base_elapsed = state["elapsed_active_s"]
    started = time.perf_counter()
    initial_step = state["step"]
    status = "running"
    in_mutation = False
    stop = StopRequest()
    stop.install()

    def elapsed():
        return base_elapsed + time.perf_counter() - started

    def commit():
        state["elapsed_active_s"] = elapsed()
        state["active_ids"] = list(model.active_ids)
        state["artifacts"] = sorted(set(state["artifacts"]))
        checkpoint_save(checkpoint, model, optimizer, stream, cfg, state)
        persist(out, cfg, state, status)

    try:
        commit()  # 在首次完整评估之前也建立可恢复的零步断点。
        if state["last_evaluation"] != state["step"]:
            in_mutation = True
            optimizer = evaluate_and_select(model, optimizer, sequences, data, cfg, state, out, elapsed())
            in_mutation = False
            commit()
        for step in range(state["step"] + 1, cfg["max_steps"] + 1):
            if not model.active_ids or stop.requested:
                break
            in_mutation = True
            losses = train_step(model, optimizer, stream, sequences, cfg, step)
            state["step"] = step
            in_mutation = False
            if evaluation_due(step, cfg):
                in_mutation = True
                before = len(model.active_ids)
                optimizer = evaluate_and_select(model, optimizer, sequences, data, cfg, state, out,
                                                 elapsed(), final_budget=step == cfg["max_steps"])
                in_mutation = False
                if len(model.active_ids) != before:
                    commit()
            if step % cfg["log_interval"] == 0:
                if not torch.isfinite(losses).all().item():
                    raise FloatingPointError("训练loss出现非有限值。")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                rate = (step - initial_step) / max(time.perf_counter() - started, 1e-9)
                log(f"PROGRESS {step:,}/{cfg['max_steps']:,} | 活动={len(model.active_ids)}/{len(all_specs)} | "
                    f"累计{elapsed()/60:.1f}分钟 | 本次平均{rate:.2f} step/s（模型停止后会改变）")
            if step % cfg["checkpoint_interval"] == 0:
                commit()
            if not model.active_ids:
                break
        status = "all_test_perfect_stopped" if not model.active_ids else "paused" if stop.requested else "budget_exhausted"
        if model.active_ids and state["last_evaluation"] != state["step"]:
            in_mutation = True
            optimizer = evaluate_and_select(model, optimizer, sequences, data, cfg, state, out, elapsed())
            in_mutation = False
            if not model.active_ids:
                status = "all_test_perfect_stopped"
        commit()
    except KeyboardInterrupt:
        if in_mutation:
            log("中断落在更新/模型移除内部，不覆盖已有完整断点。重跑从该断点恢复。")
            return
        status = "paused"
        commit()
    except Exception:
        log("发生异常；此前的完整断点仍保留，不将异常状态覆盖进去。")
        raise
    finally:
        stop.restore()
    make_plots(out, cfg, state)
    log(f"{status} | step={state['step']:,} | 测试满分停止={len(state['finished'])}/{len(all_specs)}")
    if cfg["package_results"]:
        log("正在打包，包含单模型快照，不包含续跑所需的大号checkpoint。")
        log(f"结果包：{package_results(out, state)}")


if __name__ == "__main__":
    main()
