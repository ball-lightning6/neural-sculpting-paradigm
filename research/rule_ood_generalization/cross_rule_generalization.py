#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
元学习基础实验：跨规则泛化
Meta-Learning Basic: Cross-Rule Generalization

本实验验证神经网络能否从有限规则的示例中，学习元胞自动机（CA）规则的通用结构，
并泛化到完全未见过的256种规则中的新规则。
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json
import time


# ==========================================
# 0. 实验配置
# ==========================================
class Config:
    RULE_BITS = 8
    STATE_BITS = 30
    INPUT_DIM = RULE_BITS + STATE_BITS  # 38 bits
    CORE_DIM = STATE_BITS               # 30 bits
    
    CA_LAYERS = 2  # 演化层数
    TRAIN_RULES_RATIO = 0.9  # 90% 的规则用于训练

    HIDDEN_SIZE = 4096
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.1

    # --- 训练控制 ---
    BATCH_SIZE = 1024 
    EPOCHS = 20000    
    VAL_INTERVAL = 5  # 每多少个 Epoch 评估一次
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 通用 CA 引擎
# ==========================================
def apply_any_rule(state, rule_number, width):
    """应用任意CA规则的单步演化"""
    next_state = np.zeros_like(state)
    rule_bin = np.array([int(b) for b in format(rule_number, '08b')])
    for i in range(width):
        prev = state[(i - 1 + width) % width]
        curr = state[i]
        nxt  = state[(i + 1) % width]
        idx = 7 - ((prev << 2) | (curr << 1) | nxt)
        next_state[i] = rule_bin[idx]
    return next_state


def get_ca_output(rule_num, state, layers):
    """执行多层CA演化"""
    res = state.copy()
    for _ in range(layers):
        res = apply_any_rule(res, rule_num, len(state))
    return res


# ==========================================
# 2. 数据生成器
# ==========================================
def generate_meta_datasets(cfg):
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成 2,000,000 个元学习样本...")
    all_rules = list(range(256))
    random.seed(cfg.SEED)
    random.shuffle(all_rules)
    
    split_idx = int(256 * cfg.TRAIN_RULES_RATIO)
    train_rule_list = all_rules[:split_idx]
    ood_rule_list = all_rules[split_idx:]
    
    print(f"| 训练集规则: {len(train_rule_list)} 种 | OOD测试集规则: {len(ood_rule_list)} 种 |")

    def create_batch(rule_list, n_samples):
        inputs, outputs = [], []
        for _ in range(n_samples):
            r = random.choice(rule_list)
            s = np.random.randint(0, 2, size=cfg.STATE_BITS)
            y = get_ca_output(r, s, cfg.CA_LAYERS)
            r_bin = [int(b) for b in format(r, '08b')]
            inputs.append(np.concatenate([r_bin, s]))
            outputs.append(y)
        return torch.tensor(np.array(inputs), dtype=torch.float32), \
               torch.tensor(np.array(outputs), dtype=torch.float32)

    xt_train, yt_train = create_batch(train_rule_list, 2000000)
    xt_val_id, yt_val_id = create_batch(train_rule_list, 10000)
    xt_val_ood, yt_val_ood = create_batch(ood_rule_list, 10000)
    
    return xt_train, yt_train, xt_val_id, yt_val_id, xt_val_ood, yt_val_ood


# ==========================================
# 3. 训练与实时同步监控
# ==========================================
def run_meta_experiment():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    
    xt, yt, xvid, yvid, xvoud, yvoud = generate_meta_datasets(cfg)
    train_loader = DataLoader(
        TensorDataset(xt, yt), 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True
    )
    
    xvid, yvid = xvid.to(cfg.DEVICE), yvid.to(cfg.DEVICE)
    xvoud, yvoud = xvoud.to(cfg.DEVICE), yvoud.to(cfg.DEVICE)

    # 5 层 MLP
    model = nn.Sequential(
        nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE), nn.Dropout(cfg.DROPOUT_RATE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE), nn.Dropout(cfg.DROPOUT_RATE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE), nn.Dropout(cfg.DROPOUT_RATE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE), nn.Dropout(cfg.DROPOUT_RATE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(cfg.HIDDEN_SIZE), nn.Dropout(cfg.DROPOUT_RATE),
        nn.Linear(cfg.HIDDEN_SIZE, cfg.CORE_DIM)
    ).to(cfg.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        'epochs': [], 'train_loss': [], 'val_loss_id': [], 'val_loss_ood': [],
        'acc_id': [], 'acc_ood': []
    }

    print(f"\n[{time.strftime('%H:%M:%S')}] 启动训练动力学监控...")
    best_ood = 0.0

    epoch_pbar = range(cfg.EPOCHS)
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}", leave=False)
        for batch_idx, (batch_x, batch_y) in enumerate(batch_pbar):
            batch_x, batch_y = batch_x.to(cfg.DEVICE), batch_y.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                cur_loss = running_loss / (batch_idx + 1)
                batch_pbar.set_postfix({
                    'TrainLoss': f"{cur_loss:.6f}",
                    'BestOOD': f"{best_ood*100:.2f}%"
                })
        
        avg_epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_epoch_loss)

        if (epoch + 1) % cfg.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                # ID 评估
                out_id = model(xvid)
                loss_id = criterion(out_id, yvid).item()
                acc_id = (torch.all((torch.sigmoid(out_id) > 0.5).float() == yvid, dim=1)).float().mean().item()
                
                # OOD 评估
                out_ood = model(xvoud)
                loss_ood = criterion(out_ood, yvoud).item()
                acc_ood = (torch.all((torch.sigmoid(out_ood) > 0.5).float() == yvoud, dim=1)).float().mean().item()

            if acc_ood > best_ood: best_ood = acc_ood

            history['epochs'].append(epoch + 1)
            history['val_loss_id'].append(loss_id)
            history['val_loss_ood'].append(loss_ood)
            history['acc_id'].append(acc_id)
            history['acc_ood'].append(acc_ood)

            tqdm.write(f"\n>>> [Epoch {epoch+1}] 状态快照 <<<")
            tqdm.write(f"    训练 Loss: {avg_epoch_loss:.8f}")
            tqdm.write(f"    分布内(ID) - Loss: {loss_id:.6f} | Acc: {acc_id*100:.2f}%")
            tqdm.write(f"    分布外(OOD) - Loss: {loss_ood:.6f} | Acc: {acc_ood*100:.2f}%")
            tqdm.write(f"    距离全规则顿悟进度: {'█' * int(acc_ood*20)}{'░' * (20-int(acc_ood*20))} {acc_ood*100:.2f}%\n")

    return history, cfg


def plot_meta_results(history, cfg):
    epochs = history['epochs']
    if not epochs: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_title(f"Meta-Learning Loss Dynamics (H={cfg.HIDDEN_SIZE})")
    train_l_sampled = [history['train_loss'][i-1] for i in epochs]
    ax1.plot(epochs, train_l_sampled, 'k-', alpha=0.3, label="Train Loss")
    ax1.plot(epochs, history['val_loss_id'], 'b--', label="Val Loss (Seen Rules)")
    ax1.plot(epochs, history['val_loss_ood'], 'r-', label="Test Loss (Unseen Rules)")
    ax1.set_yscale('log')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("Generalization Path (Grokking Unseen Rules)")
    ax2.plot(epochs, history['acc_id'], 'b.-', label="Seen Rules Acc")
    ax2.plot(epochs, history['acc_ood'], 'r.-', label="Unseen Rules Acc")
    ax2.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"meta_grokking_realtime_{int(time.time())}.png")
    plt.show()


if __name__ == "__main__":
    hist, config = run_meta_experiment()
    plot_meta_results(hist, config)
