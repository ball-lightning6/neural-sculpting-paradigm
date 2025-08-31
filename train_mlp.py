"""
使用MLP（多层感知机）对一维元胞自动机进行训练的脚本 (V2 - 轻量化版)。

核心改动:
- 移除了pandas依赖，改用Python内置的json和random库处理数据。
- 脚本更轻量，部署更方便。

实验目的:
- 建立性能基准的"地板"，测试一个没有结构性归纳偏置的纯粹全连接网络的能力。
- 通过设计一个参数量与SOTA模型相当的“巨型”MLP，确保对比的公平性。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os
import json  # 引入[json]库
import random  # 引入[random]库


# --- 1. 配置区域 ---
class Config:
    # --- 数据集路径配置 ---
    DATASET_PATH = "ca_rule110_layer6_36.jsonl"  # 请将您的jsonl文件名放在这里

    # --- 模型参数 ---
    BITS = 36
    HIDDEN_SIZE = 4096
    NUM_HIDDEN_LAYERS = 3
    DROPOUT_RATE = 0.1

    # --- 训练参数 ---
    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01  # 10%的数据用于验证

    # --- 日志和验证配置 ---
    LOG_FILE = "training_log_mlp.log"
    EVAL_INTERVAL_STEPS = 500


# --- 2. 日志系统设置 ---
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


# --- 3. 自定义数据集 (已更新) ---
class CASymbolicDataset(Dataset):
    """从数据列表加载符号形式的元胞自动机数据。"""

    def __init__(self, metadata_list):
        self.metadata_list = metadata_list  # 现在接收一个列表

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]  # 直接通过索引访问
        input_str = row['input']
        output_str = row['output']

        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)
        output_tensor = torch.tensor([int(bit) for bit in output_str], dtype=torch.float32)

        return input_tensor, output_tensor


# --- 4. MLP模型定义 (与之前一致) ---
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        layers.append(nn.Linear(config.BITS, config.HIDDEN_SIZE))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
        layers.append(nn.Dropout(config.DROPOUT_RATE))
        for _ in range(config.NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
            layers.append(nn.Dropout(config.DROPOUT_RATE))
        layers.append(nn.Linear(config.HIDDEN_SIZE, config.BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- 5. 验证与训练循环 (与之前一致) ---
def validate(model, dataloader, criterion, device, logger, epoch, current_step):
    model.eval()
    total_loss, total_correct_bits, exact_matches, total_bits = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct_bits += (preds==labels).sum().item()
            exact_matches += torch.all(preds==labels, dim=1).sum().item()
            total_bits += labels.numel()
    avg_loss = total_loss / len(dataloader)
    bit_accuracy = 100 * total_correct_bits / total_bits
    exact_match_ratio = 100 * exact_matches / len(dataloader.dataset)
    logger.info(f"--- Validation @ Epoch {epoch + 1}, Step {current_step} ---")
    logger.info(
        f"    Validation Loss: {avg_loss:.4f}, Bit Acc: {bit_accuracy:.2f}%, Exact Match: {exact_match_ratio:.2f}%")


def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    logger.info("\n[3/3] 开始MLP训练循环...")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for step, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if (step + 1) % config.EVAL_INTERVAL_STEPS==0 or (step + 1)==len(train_loader):
                validate(model, val_loader, criterion, device, logger, epoch, step + 1)
                model.train()


# --- 6. 主执行函数 (已更新) ---
if __name__=='__main__':
    config = Config()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)

    logger.info(f"使用设备: {device}")
    logger.info("--- 正在为MLP基准测试做准备 (轻量化版) ---")

    # --- 加载和划分数据集 (已更新) ---
    logger.info(f"\n[1/3] 正在从 {config.DATASET_PATH} 加载符号数据集...")
    try:
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            full_metadata = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"无法读取或解析JSONL文件! 请确保文件存在且每行都是一个有效的JSON。错误: {e}")
        exit()

    # 使用 random.shuffle 替代 pandas.sample
    random.seed(42)  # 为了可复现性
    random.shuffle(full_metadata)

    # 通过列表切片进行划分
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]
    logger.info(f"数据集划分完毕: {len(train_metadata)}个训练样本, {len(val_metadata)}个验证样本。")

    train_dataset = CASymbolicDataset(train_metadata)
    val_dataset = CASymbolicDataset(val_metadata)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 初始化模型和训练组件 ---
    logger.info("\n[2/3] 正在初始化“巨型”MLP模型...")
    model = MLP(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MLP模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- 启动训练循环 ---
    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)

    logger.info("\n训练完成！")