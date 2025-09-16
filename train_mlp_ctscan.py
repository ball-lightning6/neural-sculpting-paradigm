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
import textwrap


# --- 1. 配置中心 ---
class Config:
    # --- 数据集配置 ---
    DATASET_PATH = "ca_rule110_n30_l8_full_trace.jsonl"
    NUM_BITS = 30
    TOTAL_LAYERS = 8

    # --- 主模型 (Body) 架构 ---
    BODY_INPUT_SIZE = NUM_BITS
    BODY_OUTPUT_SIZE = NUM_BITS  # 主模型只预测最后一层 S_6
    BODY_HIDDEN_SIZE = 4096
    BODY_NUM_HIDDEN_LAYERS = 3  # 7个隐藏层 + 1个输入层 = 8个可探测的表征层

    # --- 探针模型 (Probe Head) 架构 ---
    PROBE_HIDDEN_SIZE = 512
    PROBE_NUM_HIDDEN_LAYERS = 1

    # --- 训练参数 ---
    EPOCHS_MAIN = 15  # 主模型训练轮数
    EPOCHS_PROBE = 3  # 每个探针的训练轮数
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和保存 ---
    LOG_FILE = "log_mind_scanner.log"
    OUTPUT_DIR = "./autodl-tmp/checkpoints_mind_scanner_b"
    BODY_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "main_body_weights.pth")
    HIDDEN_STATES_DIR = os.path.join(OUTPUT_DIR, "hidden_states")


# --- 2. 可“解剖”的MLP模型 ---
class ScannableMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_layer = nn.Sequential(
            nn.Linear(config.BODY_INPUT_SIZE, config.BODY_HIDDEN_SIZE),
            nn.GELU(), nn.LayerNorm(config.BODY_HIDDEN_SIZE)
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.BODY_HIDDEN_SIZE, config.BODY_HIDDEN_SIZE),
                nn.GELU(), nn.LayerNorm(config.BODY_HIDDEN_SIZE)
            ) for _ in range(config.BODY_NUM_HIDDEN_LAYERS)
        ])

        self.output_head = nn.Linear(config.BODY_HIDDEN_SIZE, config.BODY_OUTPUT_SIZE)

    def forward(self, x, extract_hidden_states=False):
        # 如果要提取隐藏态，我们需要一个列表来存储它们
        if extract_hidden_states:
            hidden_states_list = []

        x = self.input_layer(x)
        if extract_hidden_states:
            # .detach()可以防止在提取过程中计算梯度，节省内存
            hidden_states_list.append(x.detach().cpu().numpy())

        for layer in self.hidden_layers:
            x = layer(x)
            if extract_hidden_states:
                hidden_states_list.append(x.detach().cpu().numpy())

        logits = self.output_head(x)

        if extract_hidden_states:
            return logits, hidden_states_list
        return logits


# --- 3. 探针模型 ---
class ProbeMLP(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, config.PROBE_HIDDEN_SIZE))
        layers.append(nn.GELU())
        for _ in range(config.PROBE_NUM_HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(config.PROBE_HIDDEN_SIZE, config.PROBE_HIDDEN_SIZE))
            layers.append(nn.GELU())
        layers.append(nn.Linear(config.PROBE_HIDDEN_SIZE, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


# --- 4. 数据集类 ---
class CATraceDataset(Dataset):
    def __init__(self, metadata_list, config, mode='main', task_layer_idx=None):
        self.metadata = metadata_list;
        self.config = config
        self.mode = mode;
        self.task_layer_idx = task_layer_idx  # 0-indexed for S_1

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        input_tensor = torch.tensor([int(bit) for bit in row['input']], dtype=torch.float32)

        if self.mode == 'main':
            label = row['output'][-self.config.NUM_BITS:]  # 主模型只学最后一层 S_6
            return input_tensor, torch.tensor(label, dtype=torch.float32)
        else:  # probe mode
            start = self.task_layer_idx * self.config.NUM_BITS
            end = start + self.config.NUM_BITS
            label = row['output'][start:end]
            return input_tensor, torch.tensor(label, dtype=torch.float32)


class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = hidden_states;
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return self.hidden_states[idx], self.labels[idx]


# --- 5. 辅助函数 (完整版) ---
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


def validate(model, dataloader, criterion, device, log_prefix=""):
    model.eval()
    total_loss, total_correct, total_elements, exact_matches = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_elements += labels.numel()
            exact_matches += torch.all(preds == labels, dim=1).sum().item()

    avg_loss = total_loss / len(dataloader)
    bit_acc = 100 * total_correct / total_elements
    exact_match_ratio = 100 * exact_matches / len(dataloader.dataset)

    # 返回字典，方便记录和打印
    return {"loss": avg_loss, "bit_acc": bit_acc, "exact_match": exact_match_ratio}


def train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, log_prefix="", eval_interval=500):
    logger.info(f"--- 🚀 开始训练: {log_prefix} ---")
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
                metrics = validate(model, val_loader, criterion, device, log_prefix)
                logger.info(f"--- [{log_prefix}] Validation @ Step {current_step} ---")
                logger.info(
                    f"    Loss: {metrics['loss']:.6f}, Bit Acc: {metrics['bit_acc']:.2f}%, Exact Match: {metrics['exact_match']:.2f}%")
                if metrics['exact_match'] > best_exact_match:
                    best_exact_match = metrics['exact_match']

    # 返回训练结束时最好的Exact Match，用于填充热力图
    final_metrics = validate(model, val_loader, criterion, device, log_prefix)
    logger.info(f"--- [{log_prefix}] Final Validation ---")
    logger.info(
        f"    Loss: {final_metrics['loss']:.6f}, Bit Acc: {final_metrics['bit_acc']:.2f}%, Exact Match: {final_metrics['exact_match']:.2f}%")
    return final_metrics['exact_match']


# --- 6. 主执行流程 ---
if __name__ == '__main__':
    config = Config()
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.HIDDEN_STATES_DIR, exist_ok=True)

    # --- 阶段一: 训练并保存主模型 ---
    logger.info("\n" + "#" * 20 + " 阶段一: 训练主模型 " + "#" * 20)

    # 检查主模型权重是否存在，如果存在则跳过训练
    if os.path.exists(config.BODY_WEIGHTS_PATH):
        logger.info(f"主模型权重 {config.BODY_WEIGHTS_PATH} 已存在，跳过阶段一训练。")
    else:
        with open(config.DATASET_PATH, 'r') as f:
            full_metadata = [json.loads(line) for line in f]
        random.seed(42);
        random.shuffle(full_metadata)
        val_size = int(len(full_metadata) * config.VAL_SPLIT)
        train_meta, val_meta = full_metadata[val_size:], full_metadata[:val_size]

        main_train_dataset = CATraceDataset(train_meta, config, mode='main')
        main_val_dataset = CATraceDataset(val_meta, config, mode='main')
        main_train_loader = DataLoader(main_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        main_val_loader = DataLoader(main_val_dataset, batch_size=config.BATCH_SIZE)

        main_model = ScannableMLP(config).to(device)
        logger.info(f"主模型创建成功! 总参数量: {sum(p.numel() for p in main_model.parameters()) / 1e6:.2f} M")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(main_model.parameters(), lr=config.LEARNING_RATE)

        train_loop(main_model, main_train_loader, main_val_loader, criterion, optimizer, config.EPOCHS_MAIN,
                   log_prefix="[MainModel]")
        torch.save(main_model.state_dict(), config.BODY_WEIGHTS_PATH)
        logger.info(f"✅ 主模型训练完成并已保存至 {config.BODY_WEIGHTS_PATH}")

    # --- 阶段二: 提取所有隐藏态 ---
    logger.info("\n" + "#" * 20 + " 阶段二: 提取所有隐藏态 " + "#" * 20)
    # 重新加载完整数据集以确保顺序一致
    with open(config.DATASET_PATH, 'r') as f:
        full_metadata = [json.loads(line) for line in f]

    # 检查隐藏态文件是否存在，如果存在则跳过提取
    all_hidden_states_exist = all(
        [os.path.exists(os.path.join(config.HIDDEN_STATES_DIR, f"hidden_states_layer_{i}.npy")) for i in
         range(config.BODY_NUM_HIDDEN_LAYERS + 1)])

    if all_hidden_states_exist:
        logger.info("所有隐藏态文件已存在，跳过阶段二提取。")
    else:
        main_model = ScannableMLP(config).to(device)
        main_model.load_state_dict(torch.load(config.BODY_WEIGHTS_PATH))
        main_model.eval()

        full_dataset = CATraceDataset(full_metadata, config, mode='main')
        full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # +2: 1 for input_layer output, 1 for last hidden layer
        num_probe_layers = config.BODY_NUM_HIDDEN_LAYERS + 1
        all_hidden_states = [[] for _ in range(num_probe_layers)]

        with torch.no_grad():
            for inputs, _ in tqdm(full_loader, desc="提取隐藏态"):
                inputs = inputs.to(device)
                _, hidden_states_batch = main_model(inputs, extract_hidden_states=True)
                for i in range(len(all_hidden_states)):
                    all_hidden_states[i].append(hidden_states_batch[i])

        for i in range(len(all_hidden_states)):
            all_hidden_states[i] = np.concatenate(all_hidden_states[i], axis=0)
            state_path = os.path.join(config.HIDDEN_STATES_DIR, f"hidden_states_layer_{i}.npy")
            np.save(state_path, all_hidden_states[i])
            logger.info(f"Layer {i} hidden states ({all_hidden_states[i].shape}) saved to {state_path}")

    # --- 阶段三: 自动化“CT扫描” ---
    logger.info("\n" + "#" * 20 + " 阶段三: 自动化探针扫描 " + "#" * 20)
    # +1 for the input layer's output
    num_probe_layers = config.BODY_NUM_HIDDEN_LAYERS + 1
    results_matrix = np.zeros((num_probe_layers, config.TOTAL_LAYERS))

    for h_layer_idx in range(num_probe_layers):
        logger.info(f"\n--- 扫描主模型隐藏层 H_{h_layer_idx} ---")
        h_states_path = os.path.join(config.HIDDEN_STATES_DIR, f"hidden_states_layer_{h_layer_idx}.npy")
        h_states_data = torch.from_numpy(np.load(h_states_path)).float()

        for task_layer_idx in range(config.TOTAL_LAYERS):
            task_name = f"S_{task_layer_idx + 1}"
            log_prefix = f"[H_{h_layer_idx}->{task_name}]"
            logger.info(f"  -- 探测任务: {log_prefix} --")

            probe_labels = []
            for row in full_metadata:
                start = task_layer_idx * config.NUM_BITS;
                end = start + config.NUM_BITS
                probe_labels.append(torch.tensor(row['output'][start:end], dtype=torch.float32))
            probe_labels = torch.stack(probe_labels)

            probe_full_dataset = HiddenStateDataset(h_states_data, probe_labels)
            val_size = int(len(probe_full_dataset) * config.VAL_SPLIT)
            train_size = len(probe_full_dataset) - val_size
            probe_train_ds, probe_val_ds = random_split(probe_full_dataset, [train_size, val_size])

            probe_train_loader = DataLoader(probe_train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
            probe_val_loader = DataLoader(probe_val_ds, batch_size=config.BATCH_SIZE)

            probe_model = ProbeMLP(
                input_size=config.BODY_HIDDEN_SIZE,
                output_size=config.NUM_BITS,
                config=config
            ).to(device)
            probe_optimizer = optim.AdamW(probe_model.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.BCEWithLogitsLoss()

            final_exact_match = train_loop(
                probe_model, probe_train_loader, probe_val_loader,
                criterion, probe_optimizer, config.EPOCHS_PROBE, log_prefix=log_prefix,
                eval_interval=500  # 探针训练可以评估得更频繁
            )
            results_matrix[h_layer_idx, task_layer_idx] = final_exact_match

    # --- 阶段四: 生成并打印最终报告 ---
    logger.info("\n" + "#" * 40)
    logger.info("#" + " " * 9 + "最终扫描报告: 信息热力图 (Exact Match %)" + " " * 8 + "#")
    logger.info("#" * 40)

    header = "          | " + " | ".join([f"  Decode S_{i + 1:<2}" for i in range(config.TOTAL_LAYERS)])
    logger.info(header)
    logger.info("-" * len(header))

    for h_layer_idx in range(num_probe_layers):
        row_str = f" From H_{h_layer_idx:<2}   | " + " | ".join(
            [f"  {results_matrix[h_layer_idx, j]:>6.2f} " for j in range(config.TOTAL_LAYERS)])
        logger.info(row_str)