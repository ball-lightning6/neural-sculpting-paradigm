# train_mlp_multitask.py
"""
多任务/多头MLP训练脚本 - 第八幕猜想验证实验

核心发现：多任务训练会涌现共享的中间解耦表示，这种表示反过来加速了各个任务的学习。
实验结果：几乎在每个任务上，多任务训练都比单任务训练收敛更快。

这一发现验证了第八幕的核心猜想：通过共享的底层表示，神经网络能够发现跨任务的通用模式，
从而实现知识的迁移和加速学习。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os
import json
import random
from typing import Dict, List

class Config:
    # --- 模式选择 ---
    # 'single' 或 'multi'
    TRAINING_MODE = 'single'

    # --- 数据集路径配置 ---
    # 如果是 'multi' 模式, 这里应该是你的多任务数据集
    DATASET_PATH = "multitask_prefixed_ca110x4.jsonl"

    # --- 任务定义 (至关重要!) ---
    # 定义了每个任务的输出标签键和维度
    # 即使在 'single' 模式下，也请定义你要训练的那个任务
    TASKS: Dict[str, Dict] = {
        "add":      {"key": "output_add",  "bits": 16},
        "rain":     {"key": "output_rain", "bits": 30},
        "mod3":     {"key": "output_mod3", "bits": 60},
        "ca30":     {"key": "output_ca30", "bits": 30},
    }

    # 如果是 'single' 模式, 在这里指定要训练哪一个任务
    SINGLE_TASK_NAME = "mod3"

    # --- 模型参数 ---
    INPUT_BITS = 30
    HIDDEN_SIZE = 4096
    NUM_HIDDEN_LAYERS = 4
    DROPOUT_RATE = 0.1

    # --- 训练参数 ---
    EPOCHS = 500
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和验证配置 ---
    LOG_FILE = "training_log_mlp_multitask.log"
    EVAL_INTERVAL_STEPS = 500

def setup_logger(log_file):
    logger = logging.getLogger()
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

class MultiTaskDataset(Dataset):
    def __init__(self, metadata_list: List[Dict], config: Config):
        self.metadata_list = metadata_list
        self.config = config

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_str = row['input']
        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)

        # 根据模式准备输出标签
        if self.config.TRAINING_MODE == 'single':
            task_key = self.config.TASKS[self.config.SINGLE_TASK_NAME]['key']
            output_list = row[task_key]
            output_tensor = torch.tensor(output_list, dtype=torch.float32)
            return input_tensor, output_tensor
        else: # multi-task mode
            # 返回一个字典，键是任务名，值是标签张量
            output_tensors = {}
            for task_name, task_info in self.config.TASKS.items():
                task_key = task_info['key']
                output_list = row[task_key]
                output_tensors[task_name] = torch.tensor(output_list, dtype=torch.float32)
            return input_tensor, output_tensors

class MultiHeadMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # --- 主干网络 (Backbone) ---
        backbone_layers = [
            nn.Linear(config.INPUT_BITS, config.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_SIZE),
            nn.Dropout(config.DROPOUT_RATE)
        ]
        for _ in range(config.NUM_HIDDEN_LAYERS):
            backbone_layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
            backbone_layers.append(nn.GELU())
            backbone_layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
            backbone_layers.append(nn.Dropout(config.DROPOUT_RATE))
        self.backbone = nn.Sequential(*backbone_layers)

        # --- 输出头 (Heads) ---
        self.heads = nn.ModuleDict()
        if config.TRAINING_MODE == 'single':
            task_bits = config.TASKS[config.SINGLE_TASK_NAME]['bits']
            self.heads[config.SINGLE_TASK_NAME] = nn.Linear(config.HIDDEN_SIZE, task_bits)
        else: # multi-task mode
            for task_name, task_info in self.config.TASKS.items():
                self.heads[task_name] = nn.Linear(config.HIDDEN_SIZE, task_info['bits'])

    def forward(self, x):
        features = self.backbone(x)
        outputs = {task_name: head(features) for task_name, head in self.heads.items()}

        # 为了兼容单任务模式，如果只有一个头，直接返回值
        if len(self.heads) == 1:
            return list(outputs.values())[0]

        return outputs

def validate(model, dataloader, criterion, device, config: Config, logger, epoch, current_step):
    model.eval()

    # 初始化一个字典来存储每个任务的统计数据
    task_stats = {name: {'loss': 0.0, 'correct_bits': 0.0, 'total_bits': 0.0, 'exact_matches': 0} for name in model.heads.keys()}
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # 将标签也移动到device
            if isinstance(labels, dict):
                labels = {k: v.to(device) for k, v in labels.items()}
            else: # single task
                labels = labels.to(device)

            outputs = model(inputs)

            # --- 计算每个任务的损失和指标 ---
            if config.TRAINING_MODE == 'single':
                task_name = config.SINGLE_TASK_NAME
                loss = criterion(outputs, labels)
                task_stats[task_name]['loss'] += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                task_stats[task_name]['correct_bits'] += (preds == labels).sum().item()
                task_stats[task_name]['total_bits'] += labels.numel()
                task_stats[task_name]['exact_matches'] += torch.all(preds == labels, dim=1).sum().item()
            else: # multi-task mode
                for task_name, task_output in outputs.items():
                    task_label = labels[task_name]
                    loss = criterion(task_output, task_label)
                    task_stats[task_name]['loss'] += loss.item() * inputs.size(0)

                    preds = (torch.sigmoid(task_output) > 0.5).float()
                    task_stats[task_name]['correct_bits'] += (preds == task_label).sum().item()
                    task_stats[task_name]['total_bits'] += task_label.numel()
                    task_stats[task_name]['exact_matches'] += torch.all(preds == task_label, dim=1).sum().item()

    # --- 打印所有任务的结果 ---
    logger.info(f"--- Validation @ Epoch {epoch+1}, Step {current_step} ---")
    for task_name, stats in task_stats.items():
        avg_loss = stats['loss'] / total_samples
        bit_acc = 100 * stats['correct_bits'] / stats['total_bits']
        exact_match = 100 * stats['exact_matches'] / total_samples

        log_message = (f"  - Task [{task_name.upper()}]: "
                       f"Loss: {avg_loss:.6f}, "
                       f"Bit Acc: {bit_acc:.2f}%, "
                       f"Exact Match: {exact_match:.2f}%")
        logger.info(log_message)

def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config: Config, logger):
    logger.info("\n[3/3] 开始MLP训练循环...")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for step, (inputs, labels) in progress_bar:
            inputs = inputs.to(device)
            if isinstance(labels, dict):
                labels = {k: v.to(device) for k, v in labels.items()}
            else:
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # --- 计算总损失 ---
            total_loss = 0
            if config.TRAINING_MODE == 'single':
                total_loss = criterion(outputs, labels)
            else: # multi-task mode
                for task_name, task_output in outputs.items():
                    total_loss += criterion(task_output, labels[task_name])

            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

            current_step_total = epoch * len(train_loader) + step + 1
            if current_step_total % config.EVAL_INTERVAL_STEPS == 0:
                validate(model, val_loader, criterion, device, config, logger, epoch, current_step_total)
                model.train()

        # 每个epoch结束时也验证一次
        validate(model, val_loader, criterion, device, config, logger, epoch, (epoch + 1) * len(train_loader))

def main():
    config = Config()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)

    logger.info(f"使用设备: {device}")
    logger.info(f"--- 训练模式: {config.TRAINING_MODE.upper()} ---")

    # --- 加载和划分数据集 ---
    logger.info(f"\n[1/3] 正在从 {config.DATASET_PATH} 加载数据集...")
    try:
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            full_metadata = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"无法读取或解析JSONL文件! 错误: {e}")
        exit()

    random.seed(42)
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]

    train_dataset = MultiTaskDataset(train_metadata, config)
    val_dataset = MultiTaskDataset(val_metadata, config)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    logger.info(f"数据集加载和划分完毕: {len(train_dataset)}个训练样本, {len(val_dataset)}个验证样本。")

    # --- 初始化模型和训练组件 ---
    logger.info("\n[2/3] 正在初始化多头MLP模型...")
    model = MultiHeadMLP(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- 启动训练循环 ---
    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)

    logger.info("\n训练完成！")