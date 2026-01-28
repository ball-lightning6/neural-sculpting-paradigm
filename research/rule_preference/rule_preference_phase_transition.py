import os
# 如果需要指定 GPU，请取消下面这行的注释
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# ==========================================
# 0. 实验配置
# ==========================================
class Config:
    # --- 关键变量：修改这里 (1, 2, 3) 观察相变 ---
    CA_LAYERS = 3

    INPUT_DIM = 30  # 15 bit front + 15 bit back
    CORE_DIM = 15  # 输出 15 bit

    # --- 模型参数 ---
    HIDDEN_SIZE = 1024  # 大模型容量
    # HIDDEN_SIZE = 64    # 如果想测试小模型逼迫 XOR，可以改小

    LR = 1e-4
    WEIGHT_DECAY = 1e-4  # L2 正则，用于打破叠加态

    EPOCHS = 1000
    VAL_INTERVAL = 10  # 每多少轮评估一次
    SEED = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 规则定义 (纯 Numpy 整数位运算 - 绝对精确)
# ==========================================
def rule110_step_numpy(x):
    """
    一维元胞自动机 Rule 110 的单步演化 (Numpy版)
    输入 x: shape [Batch, Dim], dtype=int (0/1)
    """
    # 循环填充 (Periodic Boundary)
    # roll(-1) 是向左移 (取右边邻居)，roll(1) 是向右移 (取左边邻居)
    # 注意：np.roll 的 axis=1 对 [Batch, Dim]
    l = np.roll(x, 1, axis=1)
    c = x
    r = np.roll(x, -1, axis=1)

    # Rule 110 逻辑: (center & ~left) | (center ^ right) & (left | ~center)
    # 在位运算中: & 是 AND, | 是 OR, ^ 是 XOR, ~ 是 NOT
    # 这里的 ~ 对于 0/1 需要小心，我们用 (1-x) 代替 ~x
    return ((c * (1 - l)) | (c ^ r) & (l | (1 - c)))


def apply_ca_numpy(x_front, layers=3):
    """执行多层 CA 演化"""
    res = x_front.copy()
    for _ in range(layers):
        res = rule110_step_numpy(res)
    return res


# ==========================================
# 2. 数据集生成器 (逻辑修正版)
# ==========================================
def generate_datasets(cfg):
    print(f"正在生成数据... (CA Layers={cfg.CA_LAYERS})")

    # --- A. 训练集 (ID: In-Distribution) ---
    # 约束：Rule A (CA) 和 Rule B (XOR) 结果必须一致
    train_size = 10000

    # 1. 随机生成前半段 (0/1 整数)
    f_train_np = np.random.randint(0, 2, size=(train_size, cfg.CORE_DIM))

    # 2. 计算 CA 结果 (Rule A 的真理)
    y_train_np = apply_ca_numpy(f_train_np, layers=cfg.CA_LAYERS)

    # 3. 逆推后半段 (使得 front ^ back = y)
    # 逻辑：a ^ b = c  =>  b = a ^ c
    b_train_np = f_train_np ^ y_train_np

    # 4. 拼接输入
    x_train_np = np.concatenate([f_train_np, b_train_np], axis=1)

    # 5. 转为 Tensor
    x_train = torch.tensor(x_train_np, dtype=torch.float32).to(cfg.DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).to(cfg.DEVICE)

    # --- B. 验证集 (ID: In-Distribution) ---
    # 同分布，但未见过的数据
    val_size = 2000
    f_val_np = np.random.randint(0, 2, size=(val_size, cfg.CORE_DIM))
    y_val_np = apply_ca_numpy(f_val_np, layers=cfg.CA_LAYERS)
    b_val_np = f_val_np ^ y_val_np
    x_val_np = np.concatenate([f_val_np, b_val_np], axis=1)

    x_val = torch.tensor(x_val_np, dtype=torch.float32).to(cfg.DEVICE)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).to(cfg.DEVICE)

    # --- C. 测试集 (OOD: Out-of-Distribution / Conflict) ---
    # front 和 back 完全独立随机。用于测试"模型偏好"
    test_size = 2000
    f_test_np = np.random.randint(0, 2, size=(test_size, cfg.CORE_DIM))
    b_test_np = np.random.randint(0, 2, size=(test_size, cfg.CORE_DIM))  # 完全随机的 back
    x_test_np = np.concatenate([f_test_np, b_test_np], axis=1)

    # 规则 A 的真理 (只看 front)
    y_target_a_np = apply_ca_numpy(f_test_np, layers=cfg.CA_LAYERS)

    # 规则 B 的真理 (只看 front ^ back)
    y_target_b_np = f_test_np ^ b_test_np

    x_test = torch.tensor(x_test_np, dtype=torch.float32).to(cfg.DEVICE)
    y_target_a = torch.tensor(y_target_a_np, dtype=torch.float32).to(cfg.DEVICE)
    y_target_b = torch.tensor(y_target_b_np, dtype=torch.float32).to(cfg.DEVICE)

    return x_train, y_train, x_val, y_val, x_test, y_target_a, y_target_b


# ==========================================
# 3. 训练与全方位监控
# ==========================================
def train_experiment():
    cfg = Config()

    # 设置随机种子
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    print(f"=== 开始实验: CA Layers = {cfg.CA_LAYERS} ===")
    print(f"设备: {cfg.DEVICE}, Hidden: {cfg.HIDDEN_SIZE}, Weight Decay: {cfg.WEIGHT_DECAY}")

    # 准备数据
    xt, yt, xv, yv, xtest, yta, ytb = generate_datasets(cfg)

    # 模型
    model = nn.Sequential(
        nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.CORE_DIM)
    ).to(cfg.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    # 记录器
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss_id': [],
        'val_acc_id': [],  # 分布内准确率
        'loss_ood_a': [],  # OOD下，针对规则A的Loss
        'loss_ood_b': [],  # OOD下，针对规则B的Loss
        'acc_ood_a': [],  # OOD下，匹配规则A的比例
        'acc_ood_b': []  # OOD下，匹配规则B的比例
    }

    pbar = tqdm(range(cfg.EPOCHS), desc="Training")

    for epoch in pbar:
        # --- 训练步 ---
        model.train()
        optimizer.zero_grad()
        out = model(xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()

        # 记录平滑的 Train Loss
        history['train_loss'].append(loss.item())

        # --- 评估步 ---
        if (epoch + 1) % cfg.VAL_INTERVAL==0:
            model.eval()
            with torch.no_grad():
                # 1. 分布内 (ID) 验证
                out_v = model(xv)
                loss_v = criterion(out_v, yv).item()
                pred_v = (torch.sigmoid(out_v) > 0.5).float()
                acc_v = (pred_v==yv).all(dim=1).float().mean().item()

                # 2. 分布外 (OOD) 冲突测试
                out_t = model(xtest)
                pred_t = (torch.sigmoid(out_t) > 0.5).float()

                # 针对规则 A (CA)
                loss_a = criterion(out_t, yta).item()
                acc_a = (pred_t==yta).all(dim=1).float().mean().item()

                # 针对规则 B (XOR)
                loss_b = criterion(out_t, ytb).item()
                acc_b = (pred_t==ytb).all(dim=1).float().mean().item()

            # 存入历史
            history['epochs'].append(epoch + 1)
            history['val_loss_id'].append(loss_v)
            history['val_acc_id'].append(acc_v)
            history['loss_ood_a'].append(loss_a)
            history['loss_ood_b'].append(loss_b)
            history['acc_ood_a'].append(acc_a)
            history['acc_ood_b'].append(acc_b)

            # 更新进度条显示核心指标
            pbar.set_postfix({
                'L_Tr': f"{loss.item():.4f}",
                'ID': f"{acc_v * 100:.1f}%",
                'CA': f"{acc_a * 100:.1f}%",
                'XOR': f"{acc_b * 100:.1f}%"
            })

    return history, cfg


# ==========================================
# 4. 绘图与分析
# ==========================================
def plot_results(history, cfg):
    epochs = history['epochs']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图 1: Loss 动力学 ---
    ax1.set_title(f"Loss Dynamics (CA Layers={cfg.CA_LAYERS})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("BCE Loss (Log Scale)")

    # Train Loss (取每 VAL_INTERVAL 的点以便对齐)
    train_loss_sampled = [history['train_loss'][i - 1] for i in epochs]
    ax1.plot(epochs, train_loss_sampled, 'k-', alpha=0.3, label="Train Loss")
    ax1.plot(epochs, history['val_loss_id'], 'k--', label="Val Loss (In-Dist)")

    # OOD Losses
    ax1.plot(epochs, history['loss_ood_a'], 'b-', linewidth=2, label="Test Loss vs Rule A (CA)")
    ax1.plot(epochs, history['loss_ood_b'], 'orange', linewidth=2, label="Test Loss vs Rule B (XOR)")

    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 图 2: 准确率竞争 ---
    ax2.set_title(f"Accuracy Competition (CA Layers={cfg.CA_LAYERS})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Exact Match Accuracy")

    # ID Accuracy (基准线)
    ax2.plot(epochs, history['val_acc_id'], 'k--', linewidth=2, label="In-Distribution Acc (Fitted)")

    # OOD Preferences
    ax2.plot(epochs, history['acc_ood_a'], 'b.-', markersize=4, label="Follows Rule A (CA)")
    ax2.plot(epochs, history['acc_ood_b'], 'orange', marker='d', markersize=4, label="Follows Rule B (XOR)")

    ax2.axhline(y=0.0, color='gray', alpha=0.5)
    ax2.axhline(y=1.0, color='gray', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"duel_layers_{cfg.CA_LAYERS}_fixed.png")
    plt.show()


if __name__=="__main__":
    # 执行
    hist, config = train_experiment()
    plot_results(hist, config)