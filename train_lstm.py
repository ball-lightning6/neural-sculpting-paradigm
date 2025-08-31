"""
使用RNN（LSTM）对一维元胞自动机进行训练的脚本 (V1 - 时序演化版)。

实验目的:
- 测试一个为“时序和记忆”而生的RNN架构，在学习CA规则上的能力。
- 采用“一次性输入，内部演化，最终输出”的模式，精确地模拟一个封闭系统的
  逐步演化过程，这是对RNN记忆能力最纯粹的考验。
- 作为整个对比实验的收官之战，将RNN与MLP、CNN、Diffusion的性能进行比较。

数据集:
- 与MLP实验相同的jsonl格式，每行包含{'input': '...', 'output': '...'}
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


# --- 1. 配置区域 ---
class Config:
    # --- 数据集与任务配置 ---
    DATASET_PATH = "ca_rule110_layer2_36.jsonl"  # 使用与MLP相同的6层演化数据集
    EVOLUTION_STEPS = 9  # 【重要】必须与数据集的演化层数匹配
    BITS = 36

    # --- 模型参数 ---
    RNN_TYPE = 'LSTM'
    HIDDEN_SIZE = 1024  # 一个强大的RNN需要足够大的隐藏状态
    NUM_LAYERS = 3  # 堆叠多层以增加深度

    # --- 训练参数 ---
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和验证 ---
    LOG_FILE = "training_log_rnn.log"
    EVAL_INTERVAL_STEPS = 500


# --- 2. 日志系统设置 (与之前一致) ---
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


# --- 3. 数据集定义 (与MLP脚本一致) ---
class CASymbolicDataset(Dataset):
    def __init__(self, metadata_list): self.metadata_list = metadata_list

    def __len__(self): return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_tensor = torch.tensor([int(bit) for bit in row['input']], dtype=torch.float32)
        output_tensor = torch.tensor([int(bit) for bit in row['output']], dtype=torch.float32)
        return input_tensor, output_tensor


# --- 4. RNN 模型定义 ---
class RNNModel(nn.Module):
    """一个为执行逐步演化而设计的RNN模型。"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 输入编码器：将36位的输入映射到高维隐藏空间
        self.input_encoder = nn.Linear(config.BITS, config.HIDDEN_SIZE)

        # RNN核心
        if config.RNN_TYPE.upper()=='LSTM':
            self.rnn = nn.LSTM(config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, batch_first=True)
        elif config.RNN_TYPE.upper()=='GRU':
            self.rnn = nn.GRU(config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, batch_first=True)
        else:
            self.rnn = nn.RNN(config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, batch_first=True)

        # 输出解码器：将最终的隐藏状态映射回36位的输出
        self.output_decoder = nn.Linear(config.HIDDEN_SIZE, config.BITS)

    def forward(self, x, num_steps):
        batch_size = x.size(0)
        device = x.device

        # 1. 编码初始状态，并增加一个“时间步”维度 (seq_len=1)
        x_encoded = self.input_encoder(x).unsqueeze(1)

        # 2. 初始化隐藏状态
        h0 = torch.zeros(self.config.NUM_LAYERS, batch_size, self.config.HIDDEN_SIZE).to(device)
        # LSTM需要两个隐藏状态 (hidden state, cell state)
        hidden = (h0, h0.clone()) if isinstance(self.rnn, nn.LSTM) else h0

        # 3. 第一个时间步：输入真实的初始状态
        output, hidden = self.rnn(x_encoded, hidden)

        # 4. 后续时间步：输入全零张量，强迫模型依赖内部状态进行演化
        if num_steps > 1:
            # 创建一个与输出形状相同但内容为零的输入
            dummy_input = torch.zeros_like(output)
            for _ in range(num_steps - 1):
                output, hidden = self.rnn(dummy_input, hidden)

        # 5. 解码最终时间步的输出
        # a. 移除“时间步”维度
        final_hidden_state = output.squeeze(1)
        # b. 通过解码器得到最终的36位输出logits
        logits = self.output_decoder(final_hidden_state)

        return logits


# --- 5. 验证与训练循环 ---
def validate(model, dataloader, criterion, device, logger, epoch, current_step, config):
    model.eval()
    total_loss, total_correct_bits, exact_matches, total_bits = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 在验证时也使用配置中定义的演化步数
            outputs = model(inputs, num_steps=config.EVOLUTION_STEPS)
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
    logger.info("\n[3/3] 开始RNN训练循环...")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for step, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 使用配置中定义的演化步数进行前向传播
            outputs = model(inputs, num_steps=config.EVOLUTION_STEPS)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if (step + 1) % config.EVAL_INTERVAL_STEPS==0 or (step + 1)==len(train_loader):
                validate(model, val_loader, criterion, device, logger, epoch, step + 1, config)
                model.train()


# --- 6. 主执行函数 ---
if __name__=='__main__':
    config = Config()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)

    logger.info(f"使用设备: {device}")
    logger.info("--- 正在为RNN对比实验做准备 ---")

    # 加载数据
    logger.info(f"\n[1/3] 正在从 {config.DATASET_PATH} 加载符号数据集...")
    with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
        full_metadata = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]
    logger.info(f"数据集划分完毕: {len(train_metadata)}训练, {len(val_metadata)}验证。")

    train_dataset = CASymbolicDataset(train_metadata)
    val_dataset = CASymbolicDataset(val_metadata)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型
    logger.info("\n[2/3] 正在初始化RNN模型...")
    model = RNNModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型创建成功! 类型: {config.RNN_TYPE}, 总参数量: {num_params / 1_000_000:.2f} M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 开始训练
    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)

    logger.info("\n训练完成！")