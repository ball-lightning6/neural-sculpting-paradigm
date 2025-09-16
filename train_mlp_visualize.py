import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import os
import json
import random
import textwrap  # 用于格式化输出


# --- 1. 配置区域 ---
class Config:
    # --- 数据集配置 ---
    # !! 使用我们为二进制加法准备的数据集 !!
    DATASET_PATH = "ca_rule110_layer6_30.jsonl"
    NUM_BITS = 30  # 输入的每个二进制数的位数

    # --- 模型架构 ---
    INPUT_SIZE = NUM_BITS  # * 2
    # 我们将训练模型输出最终的和 (N+1位)
    OUTPUT_SIZE = NUM_BITS  # + 1
    HIDDEN_SIZE = 4096
    NUM_HIDDEN_LAYERS = 3
    DROPOUT_RATE = 0.1

    # --- 训练参数 ---
    EPOCHS = 50  # 对于加法，可能不需要太久
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和验证 ---
    LOG_FILE = "training_log_learning_dynamics.log"
    EVAL_INTERVAL_STEPS = 200  # 增加评估频率，以观察动态变化


# --- 2. 日志系统设置 (保持不变) ---
def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler();
    stream_handler.setFormatter(formatter);
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file, mode='w');
    file_handler.setFormatter(formatter);
    logger.addHandler(file_handler)
    return logger


# --- 3. 自定义数据集 (加载二进制加法数据) ---
class AdditionDataset(Dataset):
    def __init__(self, metadata_list):
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_str = row['input']
        # 目标是最终的和 (sum_output)
        output_data = row['output']

        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)
        output_tensor = torch.tensor(output_data, dtype=torch.float32)

        return input_tensor, output_tensor


# --- 4. MLP模型定义 ---
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        layers.append(nn.Linear(config.INPUT_SIZE, config.HIDDEN_SIZE))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
        layers.append(nn.Dropout(config.DROPOUT_RATE))
        for _ in range(config.NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
            layers.append(nn.Dropout(config.DROPOUT_RATE))
        layers.append(nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- 5. 核心改动：全新的、带“逐比特分析”的验证函数 ---
def validate_and_analyze_dynamics(model, dataloader, criterion, device, logger, epoch, current_step):
    model.eval()

    # 存储每个比特的loss和正确数
    per_bit_loss = torch.zeros(config.OUTPUT_SIZE, device=device)
    per_bit_correct = torch.zeros(config.OUTPUT_SIZE, device=device)

    total_loss, exact_matches = 0.0, 0
    num_samples = len(dataloader.dataset)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 1. 计算总损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # 2. 逐比特计算损失和准确率
            # criterion with reduction='none' returns loss for each element
            bce_per_element = nn.BCEWithLogitsLoss(reduction='none')(outputs, labels)
            per_bit_loss += bce_per_element.sum(dim=0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            per_bit_correct += (preds == labels).sum(dim=0)

            exact_matches += torch.all(preds == labels, dim=1).sum().item()

    # --- 格式化并打印详细报告 ---
    avg_loss = total_loss / num_samples
    exact_match_ratio = 100 * exact_matches / num_samples

    # 计算平均逐比特loss和acc
    avg_per_bit_loss = per_bit_loss / num_samples
    avg_per_bit_acc = 100 * per_bit_correct / num_samples

    logger.info(f"--- Validation @ Epoch {epoch + 1}, Step {current_step} ---")
    logger.info(f"    Overall -> Loss: {avg_loss:.6f}, Exact Match: {exact_match_ratio:.2f}%")

    # 格式化逐比特准确率字符串
    # (MSB is bit 0, LSB is bit N)
    # acc_header = "    Bit Acc -> | Overflow | " + " | ".join([f"Bit_{config.NUM_BITS-1-i:<2}" for i in range(config.NUM_BITS)]) + " |"
    acc_header = "    Bit Acc -> " + " | ".join(
        [f"Bit_{config.NUM_BITS - 1 - i:<2}" for i in range(config.NUM_BITS)]) + " |"
    acc_values = f"               | {avg_per_bit_acc[0]:>6.2f}% | " + " | ".join(
        [f"{acc:>6.2f}%" for acc in avg_per_bit_acc[1:]]) + " |"

    logger.info(acc_header)
    logger.info(acc_values)

    model.train()


# --- 6. 训练循环 (保持不变) ---
def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    logger.info("\n--- 🚀 开始训练并观察学习动力学 ---")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for step, (inputs, labels) in progress_bar:
            current_step = epoch * len(train_loader) + step + 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            if current_step % config.EVAL_INTERVAL_STEPS == 0:
                validate_and_analyze_dynamics(model, val_loader, criterion, device, logger, epoch, current_step)


# --- 7. 主执行函数 ---
if __name__ == '__main__':
    config = Config()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)

    logger.info(f"--- 🚀 实验: 观察二进制加法的逐比特学习动力学 ---")
    logger.info(f"使用设备: {device}")

    with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
        full_metadata = [json.loads(line) for line in f]
    random.seed(42);
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]

    train_dataset = AdditionDataset(train_metadata)
    val_dataset = AdditionDataset(val_metadata)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MLP(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)
    logger.info("\n✅ 训练完成！")