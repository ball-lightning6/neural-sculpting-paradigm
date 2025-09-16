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
import math


# --- 1. 配置中心 (完全适配您的最终版解耦) ---
class Config:
    # --- 数据集配置 ---
    DATASET_PATH = "rain_water_10n_4b_final_showdown.jsonl"  # <--- 指向您最终版的数据集
    NUM_COLUMNS = 10
    BITS_PER_HEIGHT = 4

    # --- 主模型 (Body) 架构 ---
    BODY_INPUT_SIZE = NUM_COLUMNS * BITS_PER_HEIGHT
    BODY_OUTPUT_SIZE = NUM_COLUMNS * BITS_PER_HEIGHT  # 主模型目标是 final_answer
    BODY_HIDDEN_SIZE = 4096
    BODY_NUM_HIDDEN_LAYERS = 4

    # --- 探针模型 (Probe Head) 架构 ---
    PROBE_HEAD_HIDDEN_SIZE = 2048
    PROBE_HEAD_NUM_HIDDEN_LAYERS = 2

    # --- 训练参数 ---
    EPOCHS_MAIN = 15
    EPOCHS_PROBE = 15
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和保存 ---
    LOG_FILE = "log_final_cognitive_preference.log"
    OUTPUT_DIR = "./autodl-tmp/checkpoints_final_preference"
    BODY_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "main_body_weights.pth")
    HIDDEN_STATES_PATH = os.path.join(OUTPUT_DIR, "hidden_states.npy")


# (此处省略了所有辅助类和函数的定义，它们将在主函数中完整提供)

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
        def __init__(self, metadata_list): self.metadata = metadata_list

        def __len__(self): return len(self.metadata)

        def __getitem__(self, idx):
            row = self.metadata[idx]
            input_tensor = torch.tensor([int(b) for b in row['input']], dtype=torch.float32)
            label = torch.tensor(row['final_answer'], dtype=torch.float32)
            return input_tensor, label


    class HiddenStateDataset(Dataset):
        def __init__(self, hidden_states, labels): self.hidden_states, self.labels = hidden_states, labels

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx): return self.hidden_states[idx], self.labels[idx]


    # --- 辅助函数定义 ---
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


    # --- 正式开始执行 ---
    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- 阶段一: 训练主模型 Body ---
    logger.info("\n" + "#" * 20 + " 阶段一: 训练主模型 Body " + "#" * 20)

    with open(config.DATASET_PATH, 'r') as f:
        full_metadata = [json.loads(line) for line in f]
    random.seed(42);
    random.shuffle(full_metadata)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_meta, val_meta = full_metadata[val_size:], full_metadata[:val_size]

    if os.path.exists(config.BODY_WEIGHTS_PATH):
        logger.info(f"主模型权重已存在，跳过训练。")
    else:
        main_train_dataset = MainDataset(train_meta)
        main_val_dataset = MainDataset(val_meta)
        main_train_loader = DataLoader(main_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4,
                                       pin_memory=True)
        main_val_loader = DataLoader(main_val_dataset, batch_size=config.BATCH_SIZE)
        main_model = MainMLP(config).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(main_model.parameters(), lr=config.LEARNING_RATE)
        train_loop(main_model, main_train_loader, main_val_loader, criterion, optimizer, config.EPOCHS_MAIN,
                   log_prefix="[MainModel]")
        torch.save(main_model.body.state_dict(), config.BODY_WEIGHTS_PATH)
        logger.info(f"✅ 主模型Body训练完成并已保存。")

    # --- 阶段二: 提取隐藏态 ---
    logger.info("\n" + "#" * 20 + " 阶段二: 提取隐藏态 " + "#" * 20)
    if os.path.exists(config.HIDDEN_STATES_PATH):
        logger.info("隐藏态文件已存在，跳过提取。")
        hidden_states_data = torch.from_numpy(np.load(config.HIDDEN_STATES_PATH)).float()
    else:
        main_model = MainMLP(config).to(device)
        main_model.body.load_state_dict(torch.load(config.BODY_WEIGHTS_PATH))
        main_model.eval()
        full_dataset = MainDataset(full_metadata)
        full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4,
                                 pin_memory=True)
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

    # --- 阶段三: 自动化“认知赛马” ---
    logger.info("\n" + "#" * 20 + " 阶段三: 开始认知风格探测 " + "#" * 20)

    probe_tasks = {
        "DP": "explain_dp",
        "Stack": "explain_stack",
        "TP": "explain_tp"
    }
    results = {}

    for task_name, label_key in probe_tasks.items():
        logger.info(f"\n--- 探测任务: {task_name} (标签: {label_key}) ---")

        probe_labels = torch.stack([torch.tensor(row[label_key], dtype=torch.float32) for row in full_metadata])
        output_size = probe_labels.shape[1]

        probe_full_dataset = HiddenStateDataset(hidden_states_data, probe_labels)
        val_size = int(len(probe_full_dataset) * config.VAL_SPLIT)
        train_size = len(probe_full_dataset) - val_size
        probe_train_ds, probe_val_ds = random_split(probe_full_dataset, [train_size, val_size])
        probe_train_loader = DataLoader(probe_train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4,
                                        pin_memory=True)
        probe_val_loader = DataLoader(probe_val_ds, batch_size=config.BATCH_SIZE)

        probe_model = ProbeMLP(output_size, config).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(probe_model.parameters(), lr=config.LEARNING_RATE)

        best_exact_match = train_loop(
            probe_model, probe_train_loader, probe_val_loader, criterion, optimizer,
            config.EPOCHS_PROBE, log_prefix=f"[Probe-{task_name}]"
        )
        results[task_name] = best_exact_match

    # --- 阶段四: 最终报告 ---
    logger.info("\n" + "#" * 40)
    logger.info("#" + " " * 8 + "最终“认知偏好”探测报告" + " " * 8 + "#")
    logger.info("#" * 40)
    logger.info("从通用隐藏层解码不同解释的最终性能 (Best Exact Match %):")

    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    for task_name, em_score in sorted_results:
        logger.info(f"  - {task_name:<10}: {em_score:.2f}%")

    logger.info("\n--- 实验结论 ---")
    best_method = sorted_results[0][0]
    logger.info(f"结果表明，神经网络的内在表征结构，与“{best_method}”解法的认知风格最为亲和。")
    logger.info("这个结果可以与“目标引导法”的收敛速度进行最终对比。")