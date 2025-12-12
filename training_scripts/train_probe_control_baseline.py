"""
探针实验的对照组基准测试脚本 (Control Baseline)

实验目的:
- 验证"探针成功解码是因为主模型内部真的形成了可解释表征，
  还是因为探针头本身足够强大能直接学会输入→解释的映射？"

实验设计:
- 创建一个独立的浅层MLP（不使用主模型的隐藏层表征）
- 直接学习 输入 → 解释标签 的映射
- 如果对照组也能成功，说明探针实验的成功可能只是因为探针头足够强大

使用方法:
1. 先运行主实验（如train_mlp_probe_add_binary.py）获得探针性能基准
2. 运行本脚本作为对照实验
3. 比较两者的性能：
   - 如果对照组性能远低于探针，说明主模型确实学会了可解释表征
   - 如果对照组性能接近探针，说明探针成功可能只是因为头部足够强大
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import json
import random


# --- 1. 配置中心 ---
class ControlConfig:
    # --- 数据和任务配置 ---
    DATASET_PATH = "binary_addition_explainable.jsonl"
    NUM_BITS = 15  # 输入的每个二进制数的位数
    
    # --- 模型架构 ---
    # 这是关键！我们定义一个与之前Probe Head结构类似的简单模型
    INPUT_SIZE = NUM_BITS * 2
    OUTPUT_SIZE = NUM_BITS * 2
    # 调整HIDDEN_SIZE来匹配probe阶段Head的复杂度
    # 如果之前的Head只是一个线性层，可以将NUM_HIDDEN_LAYERS设为0
    HIDDEN_SIZE = 2048  # 保持和主模型最后一层维度一致，模拟最接近的条件
    NUM_HIDDEN_LAYERS = 1  # 0层代表纯线性模型，1层代表浅MLP
    
    # --- 训练参数 ---
    EPOCHS = 5000  # 给予和probe阶段相同的训练时间
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01
    
    # --- 日志配置 ---
    LOG_FILE = "control_experiment_log.log"
    EVAL_INTERVAL_STEPS = 200


# --- 2. 极简的"探针"模型 ---
class ProbeHeadOnly(nn.Module):
    """
    对照组模型：一个独立的浅层MLP，直接从输入学习解释标签。
    不使用任何来自主模型的隐藏表征。
    """
    def __init__(self, config):
        super().__init__()
        
        layers = []
        
        # 如果没有隐藏层，就是一个简单的线性模型
        if config.NUM_HIDDEN_LAYERS == 0:
            layers.append(nn.Linear(config.INPUT_SIZE, config.OUTPUT_SIZE))
        else:
            # 否则，构建一个浅层MLP
            layers.append(nn.Linear(config.INPUT_SIZE, config.HIDDEN_SIZE))
            layers.append(nn.GELU())
            for _ in range(config.NUM_HIDDEN_LAYERS - 1):
                layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
                layers.append(nn.GELU())
            layers.append(nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- 3. 数据集类 (只加载输入和可解释性标签) ---
class ControlDataset(Dataset):
    """加载输入和解释标签，用于对照实验。"""
    def __init__(self, metadata_list):
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_str = row['input']
        explain_data = row['output']  # 使用解释标签作为目标
        
        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)
        output_tensor = torch.tensor(explain_data, dtype=torch.float32)
        
        return input_tensor, output_tensor


# --- 4. 辅助函数 ---
def setup_logger(log_file):
    logger = logging.getLogger(__name__)
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


def validate(model, dataloader, criterion, device, logger, epoch, current_step):
    model.eval()
    total_loss, total_correct_bits, exact_matches, total_bits = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct_bits += (preds == labels).sum().item()
            exact_matches += torch.all(preds == labels, dim=1).sum().item()
            total_bits += labels.numel()
    avg_loss = total_loss / len(dataloader)
    bit_accuracy = 100 * total_correct_bits / total_bits
    exact_match_ratio = 100 * exact_matches / len(dataloader.dataset)
    logger.info(f"--- CTRL Validation @ Epoch {epoch+1}, Step {current_step} ---")
    logger.info(f"    Validation Loss: {avg_loss:.6f}, Bit Acc: {bit_accuracy:.2f}%, Exact Match: {exact_match_ratio:.2f}%")
    return exact_match_ratio


def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    logger.info(f"\n--- 🚀 开始对照组训练 ---")
    best_exact_match = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"CTRL Epoch {epoch+1}/{config.EPOCHS}")
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
                em = validate(model, val_loader, criterion, device, logger, epoch, current_step)
                if em > best_exact_match:
                    best_exact_match = em
                model.train()
    
    return best_exact_match


# --- 5. 主执行函数 ---
if __name__ == '__main__':
    config = ControlConfig()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)
    
    logger.info(f"=" * 60)
    logger.info(f"🔬 探针实验对照组基准测试 (Control Baseline)")
    logger.info(f"=" * 60)
    logger.info(f"目标: 测试一个浅层模型能否独立学习 输入 → 解释 的映射")
    logger.info(f"模型结构: {config.NUM_HIDDEN_LAYERS} 个隐藏层, 隐藏维度 {config.HIDDEN_SIZE}")
    logger.info(f"使用设备: {device}")

    # 加载数据
    try:
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            full_metadata = [json.loads(line) for line in f]
    except FileNotFoundError:
        logger.error(f"数据集文件不存在: {config.DATASET_PATH}")
        logger.info("请先运行数据生成脚本：symbolic_math_logic/generate_add_binary_explainable.py")
        exit(1)
        
    random.seed(42)
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]
    
    train_dataset = ControlDataset(train_metadata)
    val_dataset = ControlDataset(val_metadata)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数、优化器
    model = ProbeHeadOnly(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"对照模型创建成功! 总参数量: {num_params / 1_000_000:.4f} M")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 启动训练
    best_em = train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"✅ 对照实验完成！")
    logger.info(f"对照组最佳Exact Match: {best_em:.2f}%")
    logger.info(f"=" * 60)
    logger.info(f"\n📊 结果解读:")
    logger.info(f"- 将此结果与探针实验(train_mlp_probe_add_binary.py)的结果对比")
    logger.info(f"- 如果探针性能 >> 对照组性能：主模型确实学会了可解释表征")
    logger.info(f"- 如果探针性能 ≈ 对照组性能：探针成功可能只是因为头部足够强大")
