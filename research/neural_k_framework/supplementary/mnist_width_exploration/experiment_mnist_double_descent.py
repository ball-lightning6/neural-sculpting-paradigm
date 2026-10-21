"""MNIST 双下降直接验证 + 简单性偏置指纹测量。

教科书现象（不构造合成任务）：MNIST 10 类、15% 标签噪声、双隐层 ReLU MLP、
宽度扫描 {64..4096}、Adam lr=1e-3、batch 128、300 epochs、无 wd/增广/早停。
参照 Nakkiran et al. 2019 的设置族：峰值应出现在"刚好能记住含噪声训练集"
的宽度附近（训练准确率→100% 的宽度）。

我们的增量（文献没有直接测过的指纹）：沿宽度轴测学到的函数有多"简单"——
1. 跨种子一致性 agreement：两个初始化在 1 万测试点 argmax 的一致率
   （系综越集中于单一简单函数，越高）。两个 seed 共用同一份噪声标签。
2. 预测压缩长度：测试集 argmax 标签串的 gzip 字节数（粗糙的描述长度代理）。
3. 敏感度：固定 1000 个测试点加 σ=0.1 像素噪声后 logits 的平均 L2 变动
   （函数越"尖峰"越大）。
4. 权重 L2 范数、测试 margin 统计。

预测（neural K 图景）：
- 测试误差在插值阈值宽度附近出峰（无噪声则峰弱/无）。
- 过峰后随宽度增加：agreement 上升、敏感度下降、压缩长度下降
  ——大网络更偏向简单函数。

输出：kimi_experiments/results_mnist_double_descent/
无 argparse，配置在顶部。环境变量：NSP_NOISE_RATE（默认 0.15，0 = 无噪声对照臂）、
NSP_DD_OUT（输出目录名后缀，对照臂用 _nonoise）。
"""

import gzip
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 配置区 =====================

WIDTHS = [64, 128, 256, 512, 1024, 2048, 4096]
SEEDS = [0, 1]
NOISE_RATE = float(os.environ.get("NSP_NOISE_RATE", "0.15"))
NOISE_SEED = 20261013          # 两个 seed 共用同一份噪声标签

LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 300
LOG_EVERY = 5                  # 每多少 epoch 记一次标量
PRED_EVERY = 25                # 每多少 epoch 存一次测试集预测

MNIST_ROOT = os.environ.get(
    "NSP_MNIST_ROOT", str(Path(__file__).resolve().parent / "data")
)
OUT_DIR = Path(__file__).resolve().parent / ("results_mnist_double_descent" + os.environ.get("NSP_DD_OUT", ""))
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))

# ===================== 数据 =====================

def load_mnist():
    """直接读 idx 文件，不依赖 torchvision 下载。"""
    raw = Path(MNIST_ROOT) / "MNIST" / "raw"

    def read_images(path):
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def read_labels(path):
        return np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=8).astype(np.int64)

    x_tr = read_images(raw / "train-images-idx3-ubyte")
    y_tr = read_labels(raw / "train-labels-idx1-ubyte")
    x_te = read_images(raw / "t10k-images-idx3-ubyte")
    y_te = read_labels(raw / "t10k-labels-idx1-ubyte")
    # 标准归一化
    x_tr = (x_tr - 0.1307) / 0.3081
    x_te = (x_te - 0.1307) / 0.3081
    return (torch.from_numpy(x_tr), torch.from_numpy(y_tr),
            torch.from_numpy(x_te), torch.from_numpy(y_te))


def make_noisy_labels(y_tr: torch.Tensor) -> torch.Tensor:
    """以 NOISE_RATE 概率把标签换成均匀随机类（可与真类相同，Nakkiran 式）。"""
    rng = np.random.default_rng(NOISE_SEED)
    y = y_tr.numpy().copy()
    flip = rng.random(len(y)) < NOISE_RATE
    y[flip] = rng.integers(0, 10, size=int(flip.sum()))
    return torch.from_numpy(y)

# ===================== 模型 =====================

class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 10),
        )

    def forward(self, x):
        return self.net(x)

# ===================== 指纹指标 =====================

@torch.no_grad()
def full_eval(model, x, y):
    model.eval()
    errs, losses = [], []
    for start in range(0, len(x), 4096):
        logits = model(x[start:start + 4096].to(DEVICE))
        yy = y[start:start + 4096].to(DEVICE)
        losses.append(F.cross_entropy(logits, yy).item() * len(yy))
        errs.append(int((logits.argmax(-1) != yy).sum().item()))
    return sum(losses) / len(x), sum(errs) / len(x)


@torch.no_grad()
def sensitivity(model, probe_x, delta):
    """固定扰动下 logits 的平均 L2 变动（函数尖峰度代理）。"""
    model.eval()
    base = model(probe_x)
    pert = model(probe_x + delta)
    return float((pert - base).norm(dim=-1).mean().item())


def weight_l2(model):
    return float(sum(p.pow(2).sum().item() for p in model.parameters()) ** 0.5)


def gzip_size(labels: np.ndarray) -> int:
    return len(gzip.compress(labels.tobytes()))

# ===================== 单点训练 =====================

def train_one(width, seed, data, noisy_y, probe_x, probe_delta):
    x_tr, y_tr_clean, x_te, y_te = data
    torch.manual_seed(seed)
    model = MLP(width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(y_tr_clean)
    gen = torch.Generator().manual_seed(314159 + seed)

    history = []
    pred_snaps = {}
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(n, generator=gen)
        for start in range(0, n, BATCH_SIZE):
            bidx = idx[start:start + BATCH_SIZE]  # CPU 索引 CPU 张量
            logits = model(x_tr[bidx].to(DEVICE))
            loss = F.cross_entropy(logits, noisy_y[bidx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % LOG_EVERY == 0 or epoch == EPOCHS:
            tr_loss, tr_err = full_eval(model, x_tr, noisy_y)      # 噪声标签（优化对象）
            _, tr_err_clean = full_eval(model, x_tr, y_tr_clean)   # 干净标签
            te_loss, te_err = full_eval(model, x_te, y_te)
            history.append({
                "epoch": epoch, "train_loss": tr_loss,
                "train_err_noisy": tr_err, "train_err_clean": tr_err_clean,
                "test_loss": te_loss, "test_err": te_err,
                "weight_l2": weight_l2(model),
                "sensitivity": sensitivity(model, probe_x, probe_delta),
            })
        if epoch % PRED_EVERY == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                preds = torch.cat([
                    model(x_te[s:s + 4096].to(DEVICE)).argmax(-1).cpu()
                    for s in range(0, len(x_te), 4096)
                ])
            pred_snaps[epoch] = preds.numpy().astype(np.uint8)

    return {
        "width": width, "seed": seed,
        "history": history, "pred_snaps": pred_snaps,
        "wall_seconds": round(time.time() - t0, 1),
    }

# ===================== 汇总与图 =====================

def summarize_and_plot(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 按宽度聚合终态
    rows = []
    for width in WIDTHS:
        recs = [r for r in all_results if r["width"] == width]
        finals = [r["history"][-1] for r in recs]
        row = {"width": width}
        for key in ("train_err_noisy", "train_err_clean", "test_err", "test_loss",
                    "weight_l2", "sensitivity"):
            row[key] = float(np.mean([f[key] for f in finals]))
        # 跨种子 agreement 与压缩长度（终态）
        if len(recs) >= 2:
            final_epoch = max(r["pred_snaps"] and max(r["pred_snaps"].keys()) for r in recs)
            preds = [r["pred_snaps"][final_epoch] for r in recs]
            agree = float(np.mean([
                (preds[a] == preds[b]).mean()
                for a in range(len(preds)) for b in range(a + 1, len(preds))
            ]))
            row["agreement"] = agree
            row["gzip_bytes"] = float(np.mean([gzip_size(p) for p in preds]))
        rows.append(row)

    print("\n===== 双下降汇总（终态，按宽度） =====")
    print(f"{'W':>5} | {'训练错(噪声)':>10} | {'测试错误率':>9} | {'agreement':>9} | {'敏感度':>7} | {'gzip字节':>8}")
    for row in rows:
        print(f"{row['width']:>5} | {row['train_err_noisy']:>11.4f} | {row['test_err']:>10.4f} | "
              f"{row.get('agreement', float('nan')):>9.4f} | {row['sensitivity']:>7.2f} | "
              f"{row.get('gzip_bytes', float('nan')):>8.0f}")

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6), dpi=150)
    ws = [r["width"] for r in rows]
    axes[0].plot(ws, [r["test_err"] for r in rows], "o-", label="测试错误率")
    axes[0].plot(ws, [r["train_err_noisy"] for r in rows], "s--", label="训练错误率(噪声标签)")
    axes[0].set_title("双下降曲线"); axes[0].legend()
    axes[1].plot(ws, [r.get("agreement", np.nan) for r in rows], "o-", color="#13795b")
    axes[1].set_title("跨种子 agreement（越高越集中）")
    axes[2].plot(ws, [r["sensitivity"] for r in rows], "o-", color="#c23b3b")
    axes[2].set_title("敏感度（越低越平滑）")
    axes[3].plot(ws, [r.get("gzip_bytes", np.nan) for r in rows], "o-", color="#3455a3")
    axes[3].set_title("测试预测 gzip 字节数（越低越简单）")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("隐藏宽度")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "double_descent_fingerprints.png")
    plt.close(fig)

    # 训练过程曲线：每宽度 test_err vs epoch（两个 seed 细线）
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(WIDTHS)))
    for color, width in zip(cmap, WIDTHS):
        for r in all_results:
            if r["width"] != width:
                continue
            ax.plot([h["epoch"] for h in r["history"]],
                    [h["test_err"] for h in r["history"]],
                    color=color, alpha=0.5 if r["seed"] else 0.95, linewidth=1.3,
                    label=f"W={width}" if r["seed"] == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("测试错误率")
    ax.set_title("逐 epoch 测试错误率（15% 标签噪声）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "double_descent_epoch_curves.png")
    plt.close(fig)
    return rows

# ===================== 主流程 =====================

def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("此长训练入口需要CUDA；请先配置PyTorch GPU环境，不自动退回CPU。")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_mnist()
    x_te = data[2].to(DEVICE)
    noisy_y = make_noisy_labels(data[1])
    print(f"噪声标签翻转率（含撞回真类）：{(noisy_y != data[1]).float().mean().item():.4f}")

    # 固定敏感度探针
    probe_rng = torch.Generator(device=DEVICE).manual_seed(777)
    probe_x = x_te[:1000]
    probe_delta = torch.randn(probe_x.shape, generator=probe_rng, device=DEVICE) * 0.1

    results_path = OUT_DIR / "results.jsonl"
    done_keys = set()
    if results_path.exists():
        for line in results_path.open(encoding="utf-8"):
            rec = json.loads(line)
            done_keys.add((rec["width"], rec["seed"]))
        print(f"已有 {len(done_keys)} 格，跳过")

    all_results = []
    with results_path.open("a", encoding="utf-8") as handle:
        for width in WIDTHS:
            for seed in SEEDS:
                if (width, seed) in done_keys:
                    continue
                rec = train_one(width, seed, data, noisy_y, probe_x, probe_delta)
                slim = {  # jsonl 里不存预测快照（太大），快照进 npz
                    "width": rec["width"], "seed": rec["seed"],
                    "history": rec["history"], "wall_seconds": rec["wall_seconds"],
                }
                np.savez_compressed(
                    OUT_DIR / f"preds_w{width}_s{seed}.npz",
                    **{str(k): v for k, v in rec["pred_snaps"].items()},
                )
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                handle.flush()
                final = rec["history"][-1]
                print(f"W={width} seed={seed}: 训练错误(噪声)={final['train_err_noisy']:.4f} "
                      f"测试错误={final['test_err']:.4f} 敏感度={final['sensitivity']:.2f} "
                      f"({rec['wall_seconds'] / 60:.1f}分钟)", flush=True)
                all_results.append(rec)

    # 重读全部（含先前完成的）做汇总
    full = []
    for line in results_path.open(encoding="utf-8"):
        full.append(json.loads(line))
    # 汇总需要预测快照，从 npz 读回
    for rec in full:
        snaps = np.load(OUT_DIR / f"preds_w{rec['width']}_s{rec['seed']}.npz")
        rec["pred_snaps"] = {int(k): snaps[k] for k in snaps.files}
    rows = summarize_and_plot(full)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path(__file__).resolve().parent / f"results_kimi_mnist_double_descent_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in OUT_DIR.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(OUT_DIR.parent))
        zf.write(Path(__file__), Path(__file__).name)
    print(f"\n结果已打包：{zip_path}")


if __name__ == "__main__":
    main()
