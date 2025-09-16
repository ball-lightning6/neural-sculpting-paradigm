import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import os
import json
import random
import numpy as np


# --- 1. 配置中心 ---
class Config:
    # --- 数据集配置 ---
    DATASET_PATH = "ca_rule110_n30_l4_full_trace.jsonl"
    NUM_BITS = 30
    TOTAL_LAYERS = 4

    # --- 主模型 (Body) 架构 ---
    BODY_INPUT_SIZE = NUM_BITS
    BODY_OUTPUT_SIZE = NUM_BITS  # 主模型只预测最后一层 S_6
    BODY_HIDDEN_SIZE = 4096
    BODY_NUM_HIDDEN_LAYERS = 3

    # --- 探针模型 (Probe Head) 架构 ---
    # 这个探针需要足够强大，以解码超长的轨迹
    PROBE_HEAD_HIDDEN_SIZE = 4096
    PROBE_HEAD_NUM_HIDDEN_LAYERS = 3

    # --- 训练参数 ---
    EPOCHS_MAIN = 10
    EPOCHS_PROBE = 150  # 探测任务更难，需要更多训练
    BATCH_SIZE = 512
    LEARNING_RATE = 5e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和保存 ---
    LOG_FILE = "log_probe_ca_full_trace.log"
    OUTPUT_DIR = "./autodl-tmp/checkpoints_ca_full_trace_6"
    BODY_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "ca_main_body_weights.pth")
    HIDDEN_STATES_PATH = os.path.join(OUTPUT_DIR, "ca_hidden_states.npy")


# (此处省略了所有辅助类和函数的定义，它们将在主函数中完整提供)

# --- 完整版的辅助函数 ---
def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler();
    stream_handler.setFormatter(formatter);
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file, mode='w');
    file_handler.setFormatter(formatter);
    logger.addHandler(file_handler)
    return logger


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_elements, exact_matches = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_elements += labels.numel()
            exact_matches += torch.all(preds == labels, dim=1).sum().item()
    model.train()
    return {
        "loss": total_loss / len(dataloader),
        "bit_acc": 100 * total_correct / total_elements,
        "exact_match": 100 * exact_matches / len(dataloader.dataset)
    }


def train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, log_prefix="", eval_interval=500):
    best_exact_match = -1
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"[{log_prefix}] Epoch {epoch + 1}/{epochs}")
        for step, (inputs, labels) in progress_bar:
            current_step = epoch * len(train_loader) + step + 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            if current_step % eval_interval == 0:
                metrics = validate(model, val_loader, criterion, device)
                logger.info(f"--- [{log_prefix}] Validation @ Step {current_step} ---")
                logger.info(
                    f"    Loss: {metrics['loss']:.6f}, Bit Acc: {metrics['bit_acc']:.2f}%, Exact Match: {metrics['exact_match']:.2f}%")
                if metrics['exact_match'] > best_exact_match:
                    best_exact_match = metrics['exact_match']

    final_metrics = validate(model, val_loader, criterion, device)
    if final_metrics['exact_match'] > best_exact_match:
        best_exact_match = final_metrics['exact_match']
    logger.info(f"--- [{log_prefix}] Final Validation: Best EM = {best_exact_match:.2f}% ---")
    return best_exact_match


# --- 2. 主执行流程 ---
if __name__ == '__main__':
    config = Config()


    # (为了粘贴方便和保证完整性，所有辅助函数和类都在这里定义)
    # --- 模型定义 ---
    class MLPBody(nn.Module):
        def __init__(self, config):
            super().__init__()
            layers = []
            layers.append(nn.Linear(config.BODY_INPUT_SIZE, config.BODY_HIDDEN_SIZE))
            layers.append(nn.GELU());
            layers.append(nn.LayerNorm(config.BODY_HIDDEN_SIZE))
            for _ in range(config.BODY_NUM_HIDDEN_LAYERS):
                layers.append(nn.Linear(config.BODY_HIDDEN_SIZE, config.BODY_HIDDEN_SIZE))
                layers.append(nn.GELU());
                layers.append(nn.LayerNorm(config.BODY_HIDDEN_SIZE))
            self.net = nn.Sequential(*layers)

        def forward(self, x): return self.net(x)


    class MainMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.body = MLPBody(config)
            self.head = nn.Linear(config.BODY_HIDDEN_SIZE, config.BODY_OUTPUT_SIZE)

        def forward(self, x):
            hidden = self.body(x)
            return self.head(hidden)

        def extract_hidden(self, x):
            return self.body(x)


    class ProbeMLP(nn.Module):
        def __init__(self, output_size, config):
            super().__init__()
            layers = []
            layers.append(nn.Linear(config.BODY_HIDDEN_SIZE, config.PROBE_HEAD_HIDDEN_SIZE))
            layers.append(nn.GELU())
            for _ in range(config.PROBE_HEAD_NUM_HIDDEN_LAYERS - 1):
                layers.append(nn.Linear(config.PROBE_HEAD_HIDDEN_SIZE, config.PROBE_HEAD_HIDDEN_SIZE))
                layers.append(nn.GELU())
            layers.append(nn.Linear(config.PROBE_HEAD_HIDDEN_SIZE, output_size))
            self.net = nn.Sequential(*layers)

        def forward(self, x): return self.net(x)


    # --- 数据集定义 ---
    class MainDataset(Dataset):
        def __init__(self, metadata_list, config):
            self.metadata = metadata_list
            self.config = config

        def __len__(self): return len(self.metadata)

        def __getitem__(self, idx):
            row = self.metadata[idx]
            input_tensor = torch.tensor([int(b) for b in row['input']], dtype=torch.float32)
            label = torch.tensor(row['output'][-self.config.NUM_BITS:], dtype=torch.float32)  # 只取S_6
            return input_tensor, label


    class FullTraceDataset(Dataset):  # 用于探针训练
        def __init__(self, hidden_states, full_metadata, config):
            self.hidden_states = hidden_states
            self.metadata = full_metadata
            self.config = config

        def __len__(self): return len(self.metadata)

        def __getitem__(self, idx):
            row = self.metadata[idx]
            hidden_state = self.hidden_states[idx]
            full_trace_label = torch.tensor(row['output'], dtype=torch.float32)  # 使用完整的S1..S6轨迹
            return hidden_state, full_trace_label


    # --- 正式开始执行 ---
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- 阶段一: 训练主模型 Body ---
    logger.info("\n" + "#" * 20 + " 阶段一: 训练主模型 Body (S0 -> S6) " + "#" * 20)
    with open(config.DATASET_PATH, 'r') as f:
        full_metadata = [json.loads(line) for line in f]
    random.seed(42);
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_meta, val_meta = full_metadata[val_size:], full_metadata[:val_size]

    if os.path.exists(config.BODY_WEIGHTS_PATH):
        logger.info(f"主模型权重已存在，跳过训练。")
    else:
        main_train_dataset = MainDataset(train_meta, config)
        main_val_dataset = MainDataset(val_meta, config)
        main_train_loader = DataLoader(main_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        main_val_loader = DataLoader(main_val_dataset, batch_size=config.BATCH_SIZE)
        main_model = MainMLP(config).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(main_model.parameters(), lr=config.LEARNING_RATE)
        train_loop(main_model, main_train_loader, main_val_loader, criterion, optimizer, config.EPOCHS_MAIN,
                   log_prefix="[MainModel]")
        torch.save(main_model.body.state_dict(), config.BODY_WEIGHTS_PATH)
        logger.info(f"✅ 主模型Body训练完成并已保存。")

    # --- 阶段二: 提取最终隐藏态 ---
    logger.info("\n" + "#" * 20 + " 阶段二: 提取最终隐藏态 " + "#" * 20)
    if os.path.exists(config.HIDDEN_STATES_PATH):
        logger.info("隐藏态文件已存在，跳过提取。")
        hidden_states_data = torch.from_numpy(np.load(config.HIDDEN_STATES_PATH)).float()
    else:
        main_model = MainMLP(config).to(device)
        main_model.body.load_state_dict(torch.load(config.BODY_WEIGHTS_PATH))
        main_model.eval()
        # 注意：这里我们仍然使用MainDataset，因为它只关心输入
        full_loader = DataLoader(MainDataset(full_metadata, config), batch_size=config.BATCH_SIZE, shuffle=False)
        all_hidden_states = []
        with torch.no_grad():
            for inputs, _ in tqdm(full_loader, desc="提取隐藏态"):
                inputs = inputs.to(device)
                hidden_states = main_model.extract_hidden(inputs)
                all_hidden_states.append(hidden_states.cpu().numpy())
        hidden_states_data_np = np.concatenate(all_hidden_states, axis=0)
        np.save(config.HIDDEN_STATES_PATH, hidden_states_data_np)
        hidden_states_data = torch.from_numpy(hidden_states_data_np).float()
        logger.info(f"✅ 隐藏态提取完成并已保存。Shape: {hidden_states_data.shape}")

    # --- 阶段三: 探测思维轨迹完整性 ---
    logger.info("\n" + "#" * 20 + " 阶段三: 探测思维轨迹完整性 " + "#" * 20)

    # 准备探针的数据集 (输入是隐藏态，标签是完整的S1..S6轨迹)
    probe_val_size = int(len(full_metadata) * config.VAL_SPLIT)
    probe_train_size = len(full_metadata) - probe_val_size
    probe_train_meta, probe_val_meta = full_metadata[probe_val_size:], full_metadata[:probe_val_size]
    probe_train_hs, probe_val_hs = hidden_states_data[probe_val_size:], hidden_states_data[:probe_val_size]

    probe_train_dataset = FullTraceDataset(probe_train_hs, probe_train_meta, config)
    probe_val_dataset = FullTraceDataset(probe_val_hs, probe_val_meta, config)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    probe_val_loader = DataLoader(probe_val_dataset, batch_size=config.BATCH_SIZE)

    # 初始化探针模型
    probe_output_size = config.TOTAL_LAYERS * config.NUM_BITS
    probe_model = ProbeMLP(probe_output_size, config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(probe_model.parameters(), lr=config.LEARNING_RATE)

    logger.info(f"探针模型创建成功! 目标解码维度: {probe_output_size}")

    # 训练探针
    best_exact_match = train_loop(
        probe_model, probe_train_loader, probe_val_loader, criterion, optimizer,
        config.EPOCHS_PROBE, log_prefix="[FullTraceProbe]"
    )

    logger.info("\n" + "#" * 40)
    logger.info("#" + " " * 10 + "最终“思维轨迹”探测报告" + " " * 10 + "#")
    logger.info("#" * 40)
    logger.info(f"从最终隐藏层解码完整演化轨迹 (S1..S6) 的最佳性能:")
    logger.info(f"  >>> Best Exact Match: {best_exact_match:.2f}% <<<")
    logger.info("\n--- 实验结论 ---")
    if best_exact_match > 95:
        logger.info("结论：惊人成功！这强有力地证明，一个只为最终结果而训练的网络，")
        logger.info("其最终隐藏层中，确实涌现并保留了通往该结果的、几乎全部的中间步骤信息。")
    elif best_exact_match > 50:
        logger.info("结论：显著成功。结果表明最终隐藏层包含了大量的过程信息，")
        logger.info("尽管可能存在信息压缩导致的解码困难。")
    else:
        logger.info("结论：结果不理想。这可能表明对于此复杂任务，信息是分布式存储在")
        logger.info("网络各层中的，仅靠最后一层信息不足以完美重构完整轨迹。")
