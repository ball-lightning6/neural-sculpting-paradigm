import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import os
import json
import random

# --- 1. 配置中心 ---
class Config:
    # --- 数据和任务配置 ---
    DATASET_PATH = "binary_addition_explainable.jsonl"
    NUM_BITS = 50  # 输入的每个二进制数的位数

    # --- 任务模式 ---
    # 'train_sum': 阶段一，只训练最终答案 (sum)
    # 'probe_serial': 阶段二.A，冻结主体，探测串行解释
    # 'probe_parallel': 阶段二.B，冻结主体，探测并行解释
    # 'finetune_explain': 阶段三 (可选), 微调整个模型输出解释
    TASK_MODE = 'train_sum'  # <--- 修改这里来切换实验阶段

    # --- 模型架构 ---
    # 输入位数 = 2 * NUM_BITS
    # 输出位数会根据 TASK_MODE 自动设置
    HIDDEN_SIZE = 4096
    NUM_HIDDEN_LAYERS = 3
    DROPOUT_RATE = 0.1
    
    # --- 探针头架构 (可配置更复杂的探针) ---
    PROBE_HEAD_HIDDEN_SIZE = 2048
    PROBE_HEAD_NUM_HIDDEN_LAYERS = 1  # 0代表线性，>0代表MLP
    
    # --- 训练参数 ---
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01
    
    # --- 日志和保存 ---
    LOG_FILE = "training_log_mlp_explainable.log"
    EVAL_INTERVAL_STEPS = 200
    OUTPUT_DIR = "./checkpoints_mlp_explainable"
    BODY_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "mlp_body_weights.pth")

# --- 2. 模块化MLP模型 ---
class MLPBody(nn.Module):
    """MLP的核心隐藏层部分"""
    def __init__(self, config):
        super().__init__()
        input_size = config.NUM_BITS * 2
        
        layers = []
        layers.append(nn.Linear(input_size, config.HIDDEN_SIZE))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
        layers.append(nn.Dropout(config.DROPOUT_RATE))
        
        for _ in range(config.NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(config.HIDDEN_SIZE))
            layers.append(nn.Dropout(config.DROPOUT_RATE))
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class ExplainableMLP(nn.Module):
    """完整的MLP模型，包含body和head"""
    def __init__(self, config):
        super().__init__()
        self.body = MLPBody(config)
        
        # 根据任务模式确定输出头的维度
        if config.TASK_MODE == 'train_sum':
            output_size = config.NUM_BITS + 1  # N位结果 + 1位溢出
        else: # 'train_probe' or 'finetune_explain'
            output_size = config.NUM_BITS * 2 # N位结果 + N位进位
            
        self.head = nn.Linear(config.HIDDEN_SIZE, output_size)

    def forward(self, x):
        hidden_representation = self.body(x)
        logits = self.head(hidden_representation)
        return logits

# --- 3. 动态数据集 ---
class CASymbolicDataset(Dataset):
    def __init__(self, metadata_list, task_mode, logger):
        self.metadata_list = metadata_list
        self.task_mode = task_mode
        
        # 根据任务模式选择标签字段
        if self.task_mode == 'train_sum':
            self.label_key = "sum_output"
        elif self.task_mode == 'probe_serial':
            self.label_key = "output"  # 串行解释：[结果位] + [进位]
        elif self.task_mode == 'probe_parallel':
            self.label_key = "output_parallel"  # 并行解释：[XOR] + [AND]
        else:
            self.label_key = "output"  # 默认使用串行解释
            
        logger.info(f"Dataset created in '{self.task_mode}' mode. Using label key: '{self.label_key}'")

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_str = row['input']
        # 注意：MLP输入是 [0, 1] 的浮点数列表
        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)
        
        # 确保输出标签也是浮点数
        output_data = row[self.label_key]
        output_tensor = torch.tensor(output_data, dtype=torch.float32)
        
        return input_tensor, output_tensor

def setup_logger(log_file):
    logger = logging.getLogger(__name__)
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
    logger.info(f"--- Validation @ Epoch {epoch+1}, Step {current_step} ---")
    logger.info(f"    Validation Loss: {avg_loss:.6f}, Bit Acc: {bit_accuracy:.2f}%, Exact Match: {exact_match_ratio:.2f}%")

def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    logger.info(f"\n--- 🚀 开始训练阶段: {config.TASK_MODE} ---")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.EPOCHS}")
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
                validate(model, val_loader, criterion, device, logger, epoch, current_step)
                model.train()

# --- 6. 主执行函数 ---
def run_experiment(config: Config):
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"使用设备: {device}")
    logger.info(f"--- 实验模式: {config.TASK_MODE} ---")

    # --- 加载和划分数据集 ---
    try:
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            full_metadata = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"无法读取或解析JSONL文件! 错误: {e}")
        return
        
    random.seed(42)
    random.shuffle(full_metadata)

    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]
    
    train_dataset = CASymbolicDataset(train_metadata, config.TASK_MODE, logger)
    val_dataset = CASymbolicDataset(val_metadata, config.TASK_MODE, logger)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 初始化模型和训练组件 ---
    model = ExplainableMLP(config).to(device)
    
    # 根据任务模式，加载或冻结权重
    if config.TASK_MODE in ['probe_serial', 'probe_parallel', 'train_probe']:
        logger.info(f"加载 Body 权重从: {config.BODY_WEIGHTS_PATH}")
        if not os.path.exists(config.BODY_WEIGHTS_PATH):
            logger.error("🚨 错误: Body 权重文件未找到! 请先运行 'train_sum' 模式。")
            return
        model.body.load_state_dict(torch.load(config.BODY_WEIGHTS_PATH))
        logger.info("冻结 Body 权重，只训练 Head...")
        for param in model.body.parameters():
            param.requires_grad = False
    elif config.TASK_MODE == 'finetune_explain':
        logger.info(f"加载 Body 权重从: {config.BODY_WEIGHTS_PATH}")
        if not os.path.exists(config.BODY_WEIGHTS_PATH):
            logger.error("🚨 错误: Body 权重文件未找到! 请先运行 'train_sum' 模式。")
            return
        model.body.load_state_dict(torch.load(config.BODY_WEIGHTS_PATH))
        logger.info("所有权重将参与训练 (微调模式)。")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型初始化成功! 可训练参数量: {num_params / 1_000_000:.2f} M")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)

    # --- 启动训练循环 ---
    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)
    
    # --- 训练结束后的操作 ---
    if config.TASK_MODE == 'train_sum':
        logger.info(f"✅ 阶段一训练完成。保存 Body 权重至: {config.BODY_WEIGHTS_PATH}")
        torch.save(model.body.state_dict(), config.BODY_WEIGHTS_PATH)
    else:
        logger.info(f"✅ 阶段 '{config.TASK_MODE}' 训练完成。")

if __name__ == '__main__':
    """
    运行完整的双重解释探测实验流程:
    1. 训练主模型 (只输出最终和)
    2. 探测串行解释 (模拟人类逐位计算)
    3. 探测并行解释 (模拟并行XOR/AND计算)
    
    通过对比两种解释的探测成功率，可以推断神经网络内部表征更接近哪种计算方式。
    
    注意: 需要先运行 symbolic_math_logic/generate_add_binary_explainable.py
          并设置 DUAL_EXPLANATION = True 来生成包含双重解释的数据集。
    """
    
    # --- 阶段一: 训练主模型 (只输出最终和) ---
    print("\n" + "#"*20 + " 阶段一: 训练主模型 " + "#"*20)
    config_stage1 = Config()
    config_stage1.TASK_MODE = 'train_sum'
    config_stage1.EPOCHS = 3
    run_experiment(config_stage1)
    
    print("\n" + "="*60 + "\n")
    
    # --- 阶段二.A: 冻结主模型，探测"串行解释" ---
    print("\n" + "#"*20 + " 阶段二.A: 探测串行解释 " + "#"*20)
    config_stage2a = Config()
    config_stage2a.TASK_MODE = 'probe_serial'
    config_stage2a.EPOCHS = 50
    run_experiment(config_stage2a)
    
    print("\n" + "="*60 + "\n")
    
    # --- 阶段二.B: 冻结主模型，探测"并行解释" ---
    print("\n" + "#"*20 + " 阶段二.B: 探测并行解释 " + "#"*20)
    config_stage2b = Config()
    config_stage2b.TASK_MODE = 'probe_parallel'
    config_stage2b.EPOCHS = 50
    run_experiment(config_stage2b)
    
    print("\n" + "="*60)
    print("🎉 双重解释探测实验完成！")
    print("请对比 probe_serial 和 probe_parallel 阶段的 Exact Match 性能")
    print("- 如果串行解释更高：说明网络更倾向于串行计算方式")
    print("- 如果并行解释更高：说明网络更倾向于并行计算方式")
    print("="*60)
