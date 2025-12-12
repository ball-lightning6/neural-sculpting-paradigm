# train_mlp_dual_head_supervision.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os
import json
import random
import math

# ==============================================================================
# --- 1. 配置中心 (引入混合损失权重) ---
# ==============================================================================
class Config:
    # --- 数据集配置 ---
    DATASET_PATH = "multiplier_12bit_decoupled_train.jsonl"
    NUM_BITS = 12
    
    # --- 模型架构 ---
    INPUT_SIZE = NUM_BITS * 2
    HIDDEN_SIZE = 4096
    
    # 将网络明确分解为两部分
    NUM_LAYERS_PART1 = 4  # 前半段网络层数
    NUM_LAYERS_PART2 = 4  # 后半段网络层数
    DROPOUT_RATE = 0.1
    
    # --- 标签维度 (根据数据集脚本计算) ---
    BITS_PER_COUNTER = math.ceil(math.log2(NUM_BITS + 1))
    INTERMEDIATE_LABEL_SIZE = (NUM_BITS * 2 - 1) * BITS_PER_COUNTER
    FINAL_LABEL_SIZE = NUM_BITS * 2
    
    # --- 训练参数 ---
    EPOCHS = 200  # 混合监督可能会加速收敛
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    
    # --- 核心：混合损失权重 ---
    LOSS_WEIGHT_INTERMEDIATE = 0.5  # alpha
    LOSS_WEIGHT_FINAL = 0.5        # beta
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01
    LOG_FILE = "training_log_dual_head_supervision.log"
    EVAL_INTERVAL_STEPS = 500


# ==============================================================================
# --- 2. 双分支监督MLP模型 ---
# ==============================================================================
class HybridSupervisionMLP(nn.Module):
    """
    双分支监督架构：
    - Part1的输出 → intermediate_head → 预测中间解释
    - Part1的输出 → Part2 → final_head → 预测最终答案
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- Part1: 前半段网络 ---
        part1_layers = []
        part1_layers.append(nn.Linear(config.INPUT_SIZE, config.HIDDEN_SIZE))
        part1_layers.append(nn.GELU())
        part1_layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
        part1_layers.append(nn.Dropout(config.DROPOUT_RATE))
        for _ in range(config.NUM_LAYERS_PART1 - 1):
            part1_layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
            part1_layers.append(nn.GELU())
            part1_layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
            part1_layers.append(nn.Dropout(config.DROPOUT_RATE))
        self.part1 = nn.Sequential(*part1_layers)
        
        # --- 中间分支：预测无进位计数器 ---
        self.intermediate_head = nn.Linear(config.HIDDEN_SIZE, config.INTERMEDIATE_LABEL_SIZE)
        
        # --- Part2: 后半段网络 ---
        part2_layers = []
        for _ in range(config.NUM_LAYERS_PART2):
            part2_layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
            part2_layers.append(nn.GELU())
            part2_layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
            part2_layers.append(nn.Dropout(config.DROPOUT_RATE))
        self.part2 = nn.Sequential(*part2_layers)
        
        # --- 最终分支：预测乘积 ---
        self.final_head = nn.Linear(config.HIDDEN_SIZE, config.FINAL_LABEL_SIZE)

    def forward(self, x):
        # Part1处理
        intermediate_hidden_state = self.part1(x)
        
        # 中间分支输出
        intermediate_logits = self.intermediate_head(intermediate_hidden_state)
        
        # Part2处理
        final_hidden_state = self.part2(intermediate_hidden_state)
        
        # 最终分支输出
        final_logits = self.final_head(final_hidden_state)
        
        return intermediate_logits, final_logits


# ==============================================================================
# --- 3. 双重标签数据集 ---
# ==============================================================================
class HybridDataset(Dataset):
    def __init__(self, metadata_list, config):
        self.metadata_list = metadata_list
        self.config = config

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_tensor = torch.tensor([int(bit) for bit in row['input']], dtype=torch.float32)
        
        full_label = row['output']
        intermediate_label = torch.tensor(full_label[:self.config.INTERMEDIATE_LABEL_SIZE], dtype=torch.float32)
        final_label = torch.tensor(full_label[self.config.INTERMEDIATE_LABEL_SIZE:], dtype=torch.float32)
        
        return input_tensor, intermediate_label, final_label


# ==============================================================================
# --- 4. 辅助函数 ---
# ==============================================================================
def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def validate(model, dataloader, criterion, device, config, logger, epoch, current_step):
    model.eval()
    total_loss, total_im_em, total_final_em = 0.0, 0, 0
    with torch.no_grad():
        for inputs, im_labels, final_labels in dataloader:
            inputs = inputs.to(device)
            im_labels, final_labels = im_labels.to(device), final_labels.to(device)
            
            im_logits, final_logits = model(inputs)
            
            loss_im = criterion(im_logits, im_labels)
            loss_final = criterion(final_logits, final_labels)
            loss = config.LOSS_WEIGHT_INTERMEDIATE * loss_im + config.LOSS_WEIGHT_FINAL * loss_final
            total_loss += loss.item()

            im_preds = (torch.sigmoid(im_logits) > 0.5).float()
            final_preds = (torch.sigmoid(final_logits) > 0.5).float()
            
            total_im_em += torch.all(im_preds == im_labels, dim=1).sum().item()
            total_final_em += torch.all(final_preds == final_labels, dim=1).sum().item()
            
    avg_loss = total_loss / len(dataloader)
    im_em_ratio = 100 * total_im_em / len(dataloader.dataset)
    final_em_ratio = 100 * total_final_em / len(dataloader.dataset)
    
    logger.info(f"--- Validation @ Epoch {epoch+1}, Step {current_step} ---")
    logger.info(f"    Total Loss: {avg_loss:.6f} | Intermediate EM: {im_em_ratio:.2f}% | Final EM: {final_em_ratio:.2f}%")
    model.train()


def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    logger.info("\n--- 🚀 开始双分支监督训练 ---")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for step, (inputs, im_labels, final_labels) in progress_bar:
            current_step = epoch * len(train_loader) + step + 1
            inputs = inputs.to(device)
            im_labels, final_labels = im_labels.to(device), final_labels.to(device)
            
            optimizer.zero_grad()
            
            im_logits, final_logits = model(inputs)
            
            loss_im = criterion(im_logits, im_labels)
            loss_final = criterion(final_logits, final_labels)
            total_loss = config.LOSS_WEIGHT_INTERMEDIATE * loss_im + config.LOSS_WEIGHT_FINAL * loss_final
            
            total_loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{total_loss.item():.6f}")
            if current_step % config.EVAL_INTERVAL_STEPS == 0:
                validate(model, val_loader, criterion, device, config, logger, epoch, current_step)


# ==============================================================================
# --- 5. 主执行函数 ---
# ==============================================================================
if __name__ == '__main__':
    config = Config()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)
    
    logger.info(f"--- 🚀 开始双分支监督实验: {config.DATASET_PATH} ---")
    logger.info(f"使用设备: {device}")
    
    with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
        full_metadata = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]
    
    train_dataset = HybridDataset(train_metadata, config)
    val_dataset = HybridDataset(val_metadata, config)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = HybridSupervisionMLP(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)
    logger.info("\n✅ 双分支监督训练完成！")
