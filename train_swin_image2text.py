import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score
import json

# ==============================================================================
# --- 1. 配置区域 (已整合和更新) ---
# ==============================================================================

# --- 模型配置 ---
# MODEL_NAME = "microsoft/swin-base-patch4-window7-224-in22k"
MODEL_NAME = r"swin"
NUM_LABELS = 36

# --- 数据集配置 ---
# IMAGE_DIR = "autodl-tmp/cnn_rainwater_dataset_mp/initial_images"
# METADATA_PATH = "autodl-tmp/cnn_rainwater_dataset_mp/metadata.csv"
IMAGE_DIR = "autodl-tmp/line_angle/images"
LABEL_DIR = "autodl-tmp/line_angle"

# --- 训练参数 ---
OUTPUT_DIR = "./checkpoints_swin_classifier_full"
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log_full.txt")
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  # Full fine-tuning might benefit from a slightly smaller LR
NUM_EPOCHS = 50  # Adjust as needed
FP16 = torch.cuda.is_available()
EVAL_EVERY_N_STEPS = 1000  # 每N步评估一次


# ==============================================================================
# --- 2. 自定义图像数据集 (保持不变) ---
# ==============================================================================

class CountingDataset(Dataset):
    """
    一个专门用于读取形状计数图像和12-bit标签的Dataset类。
    """

    def __init__(self, image_dir, label_dir, image_processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_processor = image_processor
        self.image_files = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_file_path = os.path.join(self.label_dir, 'labels.json')
        with open(self.label_file_path, 'r') as f:
            self.labels = json.loads(f.read())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        base_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, f"{base_name}.png")
        image = Image.open(img_path).convert("RGB")
        label = self.labels[f"{base_name}.png"]

        # ImageProcessor 会自动处理Swin Transformer需要的输入格式
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)

        label_tensor = torch.tensor(label, dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "labels": label_tensor
        }


class RainwaterImageDataset(Dataset):
    """
    一个PyTorch Dataset类。
    用于从一个元数据CSV文件和对应的图像文件夹中为“接雨水”问题加载样本。
    """

    def __init__(self, metadata_path, images_dir, image_processor):
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.metadata_df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_filename = row['initial_image']
        img_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(img_path).convert("RGB")

        label_str = str(row['final_label'])
        label_list = [int(bit) for bit in label_str]
        label_tensor = torch.tensor(label_list, dtype=torch.float32)

        # Swin Transformer的处理器会处理归一化、尺寸调整和到张量的转换
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # 返回的字典键名与训练循环中的使用保持一致
        return {"pixel_values": pixel_values, "labels": label_tensor}


# ==============================================================================
# --- 3. 评估函数 (手动) ---
# ==============================================================================

def evaluate(model, dataloader, criterion, device):
    """手动执行评估"""
    model.eval()
    total_loss, total_correct_bits, exact_matches, total_samples = 0.0, 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=FP16):
                outputs = model(inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # 计算准确率指标
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct_bits += (preds==labels).sum().item()
            exact_matches += torch.all(preds==labels, dim=1).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    bit_accuracy = 100 * total_correct_bits / (total_samples * NUM_LABELS)
    exact_match_ratio = 100 * exact_matches / total_samples

    return avg_loss, bit_accuracy, exact_match_ratio


# ==============================================================================
# --- 4. 主训练流程 (全新：完全手动的全量微调) ---
# ==============================================================================

def main():
    # --- 准备环境 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载 Image Processor 和 模型 (不再使用LoRA) ---
    print(f"加载 Image Processor 和 Swin Transformer 模型: {MODEL_NAME}...")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    ).to(device)

    print("模型已切换为全量微调模式。")

    # --- 准备数据集和Dataloaders ---
    print("准备数据集...")
    full_dataset = CountingDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        image_processor=image_processor
    )
    # full_dataset = RainwaterImageDataset(
    #     metadata_path=METADATA_PATH,
    #     images_dir=IMAGE_DIR,
    #     image_processor=image_processor
    # )

    train_size = int(0.995 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"数据集划分: {len(train_dataset)} 训练样本, {len(eval_dataset)} 验证样本。")

    # --- 初始化训练组件 ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    # --- 开始训练循环 ---
    print(f"🚀 开始Swin Transformer的全量微调训练... 每 {EVAL_EVERY_N_STEPS} 步评估一次。")
    log_file = open(LOG_FILE, "w")
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=FP16):
                outputs = model(pixel_values=inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress_bar.set_postfix({"train_loss": f"{loss.item():.6f}", "step": global_step})

            # --- 按步数进行评估 ---
            if global_step % EVAL_EVERY_N_STEPS==0:
                val_loss, bit_acc, exact_match = evaluate(model, eval_loader, criterion, device)

                log_message = (f"Step: {global_step} | Val Loss: {val_loss:.6f} | "
                               f"Bit Acc: {bit_acc:.2f}% | Exact Match: {exact_match:.2f}%")
                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()

                if val_loss < best_eval_loss:
                    best_eval_loss = val_loss
                    save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
                    # torch.save(model.state_dict(), save_path)
                    print(f"🎉 新的最佳模型已保存至 {save_path}")

                model.train()  # 评估后切回训练模式

    # --- 保存最终模型 ---
    print("✅ 训练完成，保存最终模型...")
    final_save_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_save_path)
    log_file.close()
    print(f"模型已保存至: {final_save_path}")


if __name__=="__main__":
    main()