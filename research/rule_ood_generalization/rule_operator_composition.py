#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
元学习高级实验：多算子组合泛化
Meta-Learning with Operators: Compositional Generalization

本实验测试神经网络学习多种算子（CA演化、层数控制、空间偏移）的组合能力。
模型需要理解不同控制位对应不同算子，并能零样本组合未见过的新算子序列。
这是通向通用程序执行器的关键一步。
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
    # --- 规则与维度配置 ---
    BASE_RULE_BITS = 8   # 前8位：CA规则 (0-255)
    LAYER_CONTROL_BIT = 1  # 第9位：层数控制 (0->1层, 1->2层)
    SHIFT_CONTROL_BIT = 1  # 第10位：偏移控制 (0->原样, 1->右移15位)
    RULE_BITS = BASE_RULE_BITS + LAYER_CONTROL_BIT + SHIFT_CONTROL_BIT  # 总规则位：10位
    
    STATE_BITS = 30
    INPUT_DIM = RULE_BITS + STATE_BITS  # 10 + 30 = 40 bits
    CORE_DIM = STATE_BITS               # 30 bits
    
    TRAIN_RULES_RATIO = 0.7  # 70% 的"规则-层数-偏移组合"用于训练

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
# 1. 通用 CA 引擎与算子
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


def get_ca_output(rule_num, state, layers, shift_enabled):
    """执行CA演化并可选空间偏移"""
    res = state.copy()
    # 执行 CA 演化
    for _ in range(layers):
        res = apply_any_rule(res, rule_num, len(res))
    
    # 执行空间偏移算子 (右移 15 位)
    if shift_enabled:
        res = np.roll(res, 15)
    return res


# ==========================================
# 2. 数据生成器 (适配 10 位规则编码)
# ==========================================
def generate_meta_datasets(cfg):
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成 2,000,000 个元学习样本 (40-bit 复杂组合模式)...")
    
    # 构造所有可能的组合 (256 * 2 * 2 = 1024 种任务)
    all_tasks = []
    for r in range(256):
        for l_bit in [0, 1]:
            for s_bit in [0, 1]:
                all_tasks.append((r, l_bit, s_bit))
    
    random.seed(cfg.SEED)
    random.shuffle(all_tasks)
    
    split_idx = int(len(all_tasks) * cfg.TRAIN_RULES_RATIO)
    train_task_list = all_tasks[:split_idx]
    ood_task_list = all_tasks[split_idx:]
    
    print(f"| 训练任务组合: {len(train_task_list)} | OOD测试任务组合: {len(ood_task_list)} |")

    def create_batch(task_list, n_samples):
        inputs, outputs = [], []
        for _ in range(n_samples):
            rule_num, layer_bit, shift_bit = random.choice(task_list)
            actual_layers = 1 if layer_bit == 0 else 2
            do_shift = True if shift_bit == 1 else False
            
            s = np.random.randint(0, 2, size=cfg.STATE_BITS)
            y = get_ca_output(rule_num, s, actual_layers, do_shift)
            
            # 构造 10 位规则码
            r_bin = [int(b) for b in format(rule_num, '08b')]
            full_rule_code = np.concatenate([r_bin, [layer_bit], [shift_bit]])
            
            # 拼接 40 位输入
            full_input = np.concatenate([full_rule_code, s])
            
            inputs.append(full_input)
            outputs.append(y)
            
        return torch.tensor(np.array(inputs), dtype=torch.float32), \
               torch.tensor(np.array(outputs), dtype=torch.float32)

    xt_train, yt_train = create_batch(train_task_list, 2000000)
    xt_val_id, yt_val_id = create_batch(train_task_list, 10000)
    xt_val_ood, yt_val_ood = create_batch(ood_task_list, 10000)
    
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

    print(f"\n[{time.strftime('%H:%M:%S')}] 启动 40-bit 逻辑全解耦监控...")
    best_ood = 0.0

    for epoch in range(cfg.EPOCHS):
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
                batch_pbar.set_postfix({'L_Tr': f"{cur_loss:.6f}", 'BestOOD': f"{best_ood*100:.2f}%"})
        
        avg_epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_epoch_loss)

        if (epoch + 1) % cfg.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                out_id = model(xvid)
                acc_id = (torch.all((torch.sigmoid(out_id) > 0.5).float() == yvid, dim=1)).float().mean().item()
                out_ood = model(xvoud)
                acc_ood = (torch.all((torch.sigmoid(out_ood) > 0.5).float() == yvoud, dim=1)).float().mean().item()
            

            if acc_ood > best_ood: best_ood = acc_ood

            history['epochs'].append(epoch + 1)
            history['acc_id'].append(acc_id)
            history['acc_ood'].append(acc_ood)

            tqdm.write(f"\n>>> [Epoch {epoch+1}] 状态快照 (40-bit 组合逻辑)")
            tqdm.write(f"    训练 Loss: {avg_epoch_loss:.8f}")
            tqdm.write(f"    分布内(716个任务) Acc: {acc_id*100:.2f}%")
            tqdm.write(f"    分布外(308个任务) Acc: {acc_ood*100:.2f}%")
            bar = '█' * int(acc_ood*20) + '░' * (20-int(acc_ood*20))
            tqdm.write(f"    多算子组合泛化进度: [{bar}] {acc_ood*100:.2f}%\n")

    return history, cfg


def plot_meta_results(history, cfg):
    """绘图函数 (与meta_with_layers.py相同)"""
    epochs = history['epochs']
    if not epochs: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_title(f"40-bit Meta-Learning Loss (H={cfg.HIDDEN_SIZE})")
    train_l_sampled = [history['train_loss'][i-1] for i in epochs]
    ax1.plot(epochs, train_l_sampled, 'k-', alpha=0.3, label="Train Loss")
    ax1.plot(epochs, history['val_loss_id'], 'b--', label="Val Loss (Seen)")
    ax1.plot(epochs, history['val_loss_ood'], 'r-', label="Test Loss (Unseen)")
    ax1.set_yscale('log'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Meta-Generalization to Unseen [Rule, Layer, Shift] Combinations")
    ax2.plot(epochs, history['acc_id'], 'b.-', label="Seen Tasks Acc")
    ax2.plot(epochs, history['acc_ood'], 'r.-', label="Unseen Tasks Acc")
    ax2.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"meta_40bit_results_{int(time.time())}.png"); plt.show()


if __name__ == "__main__":
    hist, config = run_meta_experiment()
    plot_meta_results(hist, config)
