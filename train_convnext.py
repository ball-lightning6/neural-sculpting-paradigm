"""
使用ConvNeXt模型对一维元胞自动机（CA）进行训练的脚本 (V2 - 从磁盘加载数据)。

任务描述：
- 输入: 一个从磁盘加载的、代表CA初始状态的240x240图像。
- 标签: 一个从metadata.csv加载的、包含36个'0'或'1'的符号字符串。
- 数据集结构:
    - a_img2img_dataset_240/
        - initial_images/
            - sample_000000.png
            - ...
        - final_images/ (此脚本不使用)
            - ...
        - metadata.csv (包含 'initial_image', 'final_image', 'final_label' 列)
- 模型: ConvNeXt (Tiny)，一个SOTA的CNN架构，旨在测试其符号推理能力。
- 目标: 验证强大的CNN在多大程度上可以学习抽象的、非局部的CA演化规则。
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

global_step = 0


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# --- 1. 配置区域 ---
class Config:
    # --- 数据集路径配置 ---
    DATASET_DIR = "ca_img2img_dataset_240"
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
    INITIAL_IMAGES_DIR = os.path.join(DATASET_DIR, "initial_images")

    # --- 模型和训练参数 ---
    BITS = 36
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 数据划分 ---
    VAL_SPLIT = 0.01  # 20%的数据用于验证


# --- 2. 自定义数据集 (从磁盘加载) ---
class CAImageDataset(Dataset):
    """
    从预生成的磁盘文件中加载元胞自动机图像和标签的PyTorch数据集。
    """

    def __init__(self, metadata_df, images_dir, transform=None):
        self.metadata_df = metadata_df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, index):
        # 1. 获取该索引对应的元数据行
        row = self.metadata_df.iloc[index]

        # 2. 加载输入图像
        image_filename = row['initial_image']
        image_path = os.path.join(self.images_dir, image_filename)
        # 确保以RGB格式打开，因为ConvNeXt需要3通道输入
        image = Image.open(image_path).convert("RGB")

        # 3. 解析符号标签
        label_str = row['final_label']
        label = torch.tensor([int(bit) for bit in label_str], dtype=torch.float32)

        # 4. 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, label


# --- 3. 模型定义 (与之前一致) ---
def get_model(config):
    """加载ConvNeXt模型并修改其分类头"""
    # 使用在ImageNet-1K上预训练的权重。这通常能提供更好的初始特征提取能力。
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)

    num_in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_in_features, config.BITS)

    print(f"模型加载成功: ConvNeXt Tiny (使用预训练权重)")
    return model


# # --- 4. 训练和验证循环 (与之前一致) ---
# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     global global_step
#     model.train()
#     total_loss = 0.0
#     progress_bar = tqdm(dataloader, desc="Training")

#     for inputs, labels in progress_bar:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         progress_bar.set_postfix(loss=f"{loss.item():.4f}")

#         global_step+=1
#         if global_step %1000==0:


#     return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct_bits, exact_matches, total_bits = 0.0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct_bits += (preds==labels).sum().item()
            exact_matches += torch.all(preds==labels, dim=1).sum().item()
            total_bits += labels.numel()

    avg_loss = total_loss / len(dataloader)
    bit_accuracy = 100 * total_correct_bits / total_bits
    exact_match_ratio = 100 * exact_matches / len(dataloader.dataset)

    return avg_loss, bit_accuracy, exact_match_ratio


# --- 5. 主执行函数 (已更新) ---
if __name__=='__main__':
    global global_step
    config = Config()
    device = torch.device(config.DEVICE)
    print(f"使用设备: {device}")

    # --- 数据预处理 ---
    # 这是使用ImageNet预训练模型时的标准变换流程
    # ToTensor() 会将PIL图像[0, 255]转换为Tensor[0, 1]
    # Normalize() 会将Tensor从[0, 1]归一化到[-1, 1]附近，使用ImageNet的均值和标准差
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 加载和划分数据集 ---
    print(f"\n[1/3] 正在从 {config.METADATA_PATH} 加载元数据...")
    full_metadata = pd.read_csv(config.METADATA_PATH)

    # 随机打乱数据集
    full_metadata = full_metadata.sample(frac=1, random_state=42).reset_index(drop=True)

    # 划分训练集和验证集
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata = full_metadata[val_size:]
    val_metadata = full_metadata[:val_size]

    print(f"数据集划分完毕: {len(train_metadata)}个训练样本, {len(val_metadata)}个验证样本。")

    # 创建Dataset实例
    train_dataset = CAImageDataset(train_metadata, config.INITIAL_IMAGES_DIR, transform=data_transforms)
    val_dataset = CAImageDataset(val_metadata, config.INITIAL_IMAGES_DIR, transform=data_transforms)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 初始化模型和训练组件 ---
    print("\n[2/3] 正在初始化模型...")
    model = get_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    total_train_loss = 0.0
    # --- 开始训练循环 ---
    print("\n[3/3] 开始训练循环...")

    log_file = open('ca_image_log.txt', 'w')

    for epoch in range(config.EPOCHS):
        print(f"--- Epoch {epoch + 1}/{config.EPOCHS} ---")

        model.train()

        progress_bar = tqdm(train_loader, desc="Training")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            global_step += 1
            if global_step % 1000==0:
                val_loss, bit_acc, exact_match = validate(model, val_loader, criterion, device)

                print(f"Epoch {epoch + 1} | 训练损失: {total_train_loss / 1000.:.8f} | 验证损失: {val_loss:.8f}")
                print(f"           | 验证位准确率: {bit_acc:.8f}% | 验证完全匹配率: {exact_match:.8f}%")

                log_file.write(f"Epoch {epoch + 1} | 训练损失: {total_train_loss / 1000.:.8f} | 验证损失: {val_loss:.8f}\n")
                log_file.write(f"           | 验证位准确率: {bit_acc:.8f}% | 验证完全匹配率: {exact_match:.8f}%\n")
                log_file.flush()

                total_train_loss = 0.0

        # train_loss = total_loss / len(dataloader)

        # train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    print("\n训练完成！")