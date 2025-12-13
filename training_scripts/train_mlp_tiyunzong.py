# train_mlp_tiyunzong.py
"""
“梯云纵”课程学习 - 元胞自动机长程推理训练

实验目的：验证通过渐进式增加推理深度的课程学习，是否可以训练出能执行超长程推理的模型

核心思想：
既然已经训练了n层元胞自动机，那根据同样隐藏层训练n+1层应该不难，
这样肯定比直接训练很高层数的元胞自动机快。这可以看成一种解耦方式。

关键发现：
到了一定层数就很难再增加层数了，这充分说明这个范式并非魔法，
很可能不能通过任何办法任意增加元胞自动机的层数。

理论意义：
1. 存在无法通过简单技巧突破的能力天花板
2. 验证了计算不可约性在神经网络学习中的体现
3. 课程学习有其固有的局限性
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
    # --- 数据集路径配置 ---
    # 使用包含完整轨迹的数据集
    DATASET_PATH = "autodl-tmp/ca_rule110_n30_l100_full_trace.jsonl"

    # --- 模型参数 ---
    INPUT_BITS = 30
    HIDDEN_SIZE = 4096
    NUM_HIDDEN_LAYERS = 4
    DROPOUT_RATE = 0.1

    # --- 训练参数 ---
    # EPOCHS 现在代表每个课程阶段的训练轮数
    EPOCHS_PER_STAGE = 5
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和验证配置 ---
    LOG_FILE = "training_log_mlp_tiyunzong.log"
    EVAL_INTERVAL_STEPS = 500
    BEST_MODEL_PATH = "autodl-tmp/best_model_mlp_tiyunzong.pth"

    # --- 输出位数 (会被课程动态修改) ---
    # 这个值只是个占位符，实际值由 CurriculumConfig 和 Dataset 动态决定
    OUTPUT_BITS = -1

    GATE_METRIC = 'exact_match' # 'eval_loss', 'bit_accuracy', 'exact_match'
    GATE_THRESHOLD = 100.00      # 触发进入下一阶段的阈值

    MAX_EPOCHS_PER_STAGE = 5000   # 每个阶段最多训练的epoch数，防止卡死

# =========================================================================
# --- 新增: “梯云纵”课程配置 ---
# =========================================================================
class CurriculumConfig:
    """配置“梯云纵”课程学习的参数"""
    CA_WIDTH = 30
    TOTAL_LAYERS = 100  # 数据集中的总层数

    # 标签始终由 WINDOW_SIZE 个CA状态构成
    WINDOW_SIZE = 5
    START_LAYERS=5

    # 课程的总阶段数，即“拉伸”的总次数
    # 最终能预测到 WINDOW_SIZE + TOTAL_STAGES 层
    # 例如 5 + 85 = 90
    TOTAL_STAGES = TOTAL_LAYERS - WINDOW_SIZE

# --- 2. 日志系统设置 (无变化) ---
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

# =========================================================================
# --- 3. 修改: 自定义数据集 ---
# =========================================================================
class TiYunZongCADataset(Dataset):
    """
    一个能够根据课程阶段动态构建标签的数据集。
    """
    def __init__(self, metadata_list, curriculum_config):
        self.metadata_list = metadata_list
        self.cfg = curriculum_config
        self.current_stage = 0  # 初始阶段为0
        self.max_predict_layer = self.cfg.START_LAYERS

        # 预先计算好每个阶段的标签索引，避免在 __getitem__ 中重复计算
        self._precompute_label_indices()

    def _precompute_label_indices(self):
        self.stage_indices = {}

        # 阶段 0: 初始阶段
        # 标签长度由 WINDOW_SIZE 决定
        # 索引从 0 开始，到 WINDOW_SIZE-1
        # 例如 WINDOW_SIZE=4, 初始索引是 [0, 1, 2, 3]
        initial_indices = list(range(self.cfg.WINDOW_SIZE))
        self.stage_indices[0] = initial_indices

        # 后续阶段
        current_indices = initial_indices
        # 我们总共要拉伸的次数是 TOTAL_LAYERS - WINDOW_SIZE
        # 例如 90层 - 4个窗口 = 86次拉伸
        num_stretches = self.cfg.TOTAL_STAGES # 我们在Config里已经算好了

        for stage in range(1, num_stretches + 1):
            # 要被替换的旧索引在窗口中的位置
            replace_idx_in_window = (stage - 1) % self.cfg.WINDOW_SIZE

            # 新的层索引是窗口覆盖范围之外的下一个
            # 例如 stage=1, new_layer_idx = 4
            # stage=2, new_layer_idx = 5
            new_layer_idx = self.cfg.WINDOW_SIZE + stage - 1

            # 复制并替换
            new_indices = list(current_indices)
            new_indices[replace_idx_in_window] = new_layer_idx

            self.stage_indices[stage] = new_indices
            current_indices = new_indices

    def set_stage(self, stage):
        """外部调用此方法来推进课程"""
        if stage > self.cfg.TOTAL_STAGES:
            raise ValueError(f"Stage {stage} exceeds total stages {self.cfg.TOTAL_STAGES}")
        self.current_stage = stage
        # 更新当前阶段需要预测的最远层
        indices = self.stage_indices[self.current_stage]
        self.max_predict_layer = max(indices) + 1
        logging.info(f"Dataset curriculum advanced to Stage {self.current_stage}. "
                     f"Max layer prediction: {self.max_predict_layer}. "
                     f"Label indices: {indices}")

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        row = self.metadata_list[index]
        input_str = row['input']
        full_trace = row['output'] # 这是包含所有层演化的长列表

        # 根据当前阶段，动态构建输出标签
        label_indices = self.stage_indices[self.current_stage]

        output_list = []
        for layer_idx in label_indices:
            start = layer_idx * self.cfg.CA_WIDTH
            end = start + self.cfg.CA_WIDTH
            output_list.extend(full_trace[start:end])

        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)
        output_tensor = torch.tensor(output_list, dtype=torch.float32)

        return input_tensor, output_tensor

# --- 4. MLP模型定义 (无变化) ---
class MLP(nn.Module):
    def __init__(self, input_bits, output_bits, hidden_size, num_hidden, dropout):
        super().__init__()
        # 为了动态修改输出层，我们将参数直接传入
        layers = []
        layers.append(nn.Linear(input_bits, hidden_size))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, output_bits))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def validate(model, dataloader, criterion, device, logger, stage, epoch, current_step):
    model.eval()
    total_loss, total_correct_bits, exact_matches, total_bits = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            exact_matches += torch.all(preds == labels, dim=1).sum().item()
            total_correct_bits += (preds == labels).sum().item()
            total_bits += labels.numel()

    avg_loss = total_loss / len(dataloader)
    bit_accuracy = 100 * total_correct_bits / total_bits
    exact_match_ratio = 100 * exact_matches / len(dataloader.dataset)

    logger.info(f"--- Validation @ Stage {stage}, Epoch {epoch}, Step {current_step} ---")
    logger.info(f"Loss: {avg_loss:.12f}, Bit Acc: {bit_accuracy:.6f}%, Exact Match: {exact_match_ratio:.2f}%")
    logger.info(f"Current Max Prediction Layer: {dataloader.dataset.max_predict_layer}")

    # 返回一个包含所有指标的字典
    return {
        'eval_loss': avg_loss,
        'bit_accuracy': bit_accuracy,
        'exact_match': exact_match_ratio
    }

# =========================================================================
# --- 6. 修改: 训练循环引入“收敛门控” ---
# =========================================================================
def train_loop(model, train_dataset, val_dataset, criterion, optimizer, device, config, curriculum_cfg, logger):
    logger.info("\n[3/3] 开始“梯云纵”课程学习训练循环（带收敛门控）...")
    best_val_loss = float('inf')

    # 外层循环: 控制课程阶段
    for stage in range(curriculum_cfg.TOTAL_STAGES + 1):
        logger.info("="*70)
        logger.info(f"🚀🚀🚀 STARTING CURRICULUM STAGE {stage}/{curriculum_cfg.TOTAL_STAGES} 🚀🚀🚀")
        logger.info("="*70)

        # 1. 推进数据集到当前阶段
        train_dataset.set_stage(stage)
        val_dataset.set_stage(stage)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # 2. 内层循环: 改为 while 循环，直到满足条件
        stage_converged = False
        for epoch in range(1, config.MAX_EPOCHS_PER_STAGE + 1):
            model.train()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Stage {stage} - Epoch {epoch}/{config.MAX_EPOCHS_PER_STAGE}")
            for step, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=f"{loss.item():.12f}")

                # 验证逻辑保持不变
                if (step + 1) % config.EVAL_INTERVAL_STEPS == 0 or (step + 1) == len(train_loader):
                    metrics = validate(model, val_loader, criterion, device, logger, stage, epoch, step + 1)
                    model.train()

                    # 保存最佳模型
                    if metrics['eval_loss'] < best_val_loss:
                        best_val_loss = metrics['eval_loss']
                        logger.info(f"🎉 新的全局最低验证损失: {best_val_loss:.12f}. 正在保存模型...")
                        torch.save(model.state_dict(), config.BEST_MODEL_PATH)

            # 3. 每轮 epoch 结束后，检查收敛门控
            metrics = validate(model, val_loader, criterion, device, logger, stage, epoch, len(train_loader))
            current_metric_val = metrics[config.GATE_METRIC]

            gate_passed = False
            if config.GATE_METRIC == 'eval_loss':
                if current_metric_val < config.GATE_THRESHOLD:
                    gate_passed = True
            else: # bit_accuracy or exact_match
                if current_metric_val >= config.GATE_THRESHOLD:
                    gate_passed = True

            if gate_passed:
                logger.info(f"✅ CONVERGENCE GATE PASSED at Stage {stage}!")
                logger.info(f"Metric '{config.GATE_METRIC}' ({current_metric_val:.4f}) passed threshold ({config.GATE_THRESHOLD}).")
                stage_converged = True
                break # 跳出当前阶段的 epoch 循环

        # 4. 检查当前阶段是否真的收敛了
        if not stage_converged:
            logger.warning(f"⚠️ STAGE {stage} FAILED TO CONVERGE within {config.MAX_EPOCHS_PER_STAGE} epochs.")
            logger.warning(f"Training will stop. The model might be struggling with the current complexity.")
            break # 彻底跳出外层的 stage 循环

# --- 7. 主执行函数 ---
if __name__ == '__main__':
    config = Config()
    curriculum_cfg = CurriculumConfig() # 初始化课程配置
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)

    logger.info(f"使用设备: {device}")
    logger.info("--- 正在为“梯云纵”课程学习做准备 ---")

    # --- 加载数据集 ---
    logger.info(f"\n[1/3] 正在从 {config.DATASET_PATH} 加载完整轨迹数据集...")
    # ... (加载逻辑不变) ...
    try:
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            full_metadata = [json.loads(line) for line in tqdm(f, desc="Loading data")]
    except Exception as e:
        logger.error(f"无法读取或解析JSONL文件! 错误: {e}")
        exit()

    random.seed(42)
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]

    # 使用新的动态数据集
    train_dataset = TiYunZongCADataset(train_metadata, curriculum_cfg)
    val_dataset = TiYunZongCADataset(val_metadata, curriculum_cfg)
    logger.info(f"数据集初始化完毕: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本。")

    # --- 初始化模型 ---
    logger.info("\n[2/3] 正在初始化MLP模型...")
    # 计算模型的输出位数，它在整个训练中是固定的
    fixed_output_bits = curriculum_cfg.WINDOW_SIZE * curriculum_cfg.CA_WIDTH
    config.OUTPUT_BITS = fixed_output_bits

    model = MLP(
        input_bits=config.INPUT_BITS,
        output_bits=config.OUTPUT_BITS,
        hidden_size=config.HIDDEN_SIZE,
        num_hidden=config.NUM_HIDDEN_LAYERS,
        dropout=config.DROPOUT_RATE
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MLP模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")
    logger.info(f"初始输出维度: {config.OUTPUT_BITS}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- 启动训练 ---
    train_loop(model, train_dataset, val_dataset, criterion, optimizer, device, config, curriculum_cfg, logger)

    logger.info("\n“梯云纵”课程学习完成！")"file_path":"e:\code\neural-sculpting-paradigm\training_scripts/train_mlp_tiyunzong.py