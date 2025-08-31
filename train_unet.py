"""
使用UNET模型进行图像到图像元胞自动机演化预测的训练脚本 (V2 - 可视化评估版)。

核心特性:
- 可视化验证: 在每次验证时，自动保存第一个批次的样本对比图。
- 三联图设计: 为每个样本生成 (输入图像 | 目标图像 | 预测图像) 的并列图，
              极其方便地对比模型表现。
- 反归一化: 保存的输入图像会进行反归一化，以显示原始视觉效果。
- 模块化设计: 整个脚本依然保持清晰、专业的结构。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image  # 引入保存图像的工具
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import logging
import numpy as np


# --- 1. 配置区域 ---
class Config:
    # --- 路径配置 ---
    DATASET_DIR = "ca_img2img_dataset_240"
    OUTPUT_DIR = "unet_training_results"  # 一个总的输出目录
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
    INITIAL_IMAGES_DIR = os.path.join(DATASET_DIR, "initial_images")
    FINAL_IMAGES_DIR = os.path.join(DATASET_DIR, "final_images")
    # 【新增】评估图像保存目录
    EVAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "eval_images")

    # --- 训练参数 ---
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01

    # --- 日志和验证 ---
    LOG_FILE = os.path.join(OUTPUT_DIR, "training_log_unet.log")
    EVAL_INTERVAL_STEPS = 1000


# --- 2. 日志系统设置 (与之前一致) ---
def setup_logger(log_file):
    # (代码与之前版本一致)
    log_dir = os.path.dirname(log_file)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
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


# --- 3. 数据集定义 (与之前一致) ---
class CAImageToImageDataset(Dataset):
    # (代码与之前版本一致)
    def __init__(self, metadata_df, initial_dir, final_dir, transform=None):
        self.metadata_df, self.initial_dir, self.final_dir, self.transform = metadata_df, initial_dir, final_dir, transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, index):
        row = self.metadata_df.iloc[index]
        initial_img_path = os.path.join(self.initial_dir, row['initial_image'])
        initial_image = Image.open(initial_img_path).convert("RGB")
        final_img_path = os.path.join(self.final_dir, row['final_image'])
        final_image = Image.open(final_img_path).convert("L")
        if self.transform:
            initial_image = self.transform['input'](initial_image)
            final_image = self.transform['target'](final_image)
        return initial_image, final_image


# --- 4. UNET模型定义 (与之前一致) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x): return self.double_conv(x)


class UNet(nn.Module):
    # (代码与之前版本一致，为简洁省略)
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1);
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1);
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1);
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1);
        x = self.conv4(x)
        return self.outc(x)


# --- 5. 验证循环 (已集成图像保存功能) ---
def validate(model, dataloader, criterion, device, logger, epoch, current_step, config):
    model.eval()
    total_loss, total_correct_pixels, perfect_matches, total_pixels = 0.0, 0, 0, 0

    # 反归一化所需的均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        # 我们只保存第一个验证批次的图像
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            total_correct_pixels += (preds==targets).sum().item()
            perfect_matches += torch.all(preds==targets, dim=(1, 2, 3)).sum().item()
            total_pixels += targets.numel()

            # --- 【核心改动】图像保存逻辑 ---
            if batch_idx==0:  # 只处理第一个batch
                # 遍历batch中的每个样本
                for i in range(inputs.size(0)):
                    # 反归一化输入图像以供可视化
                    input_vis = inputs[i] * std + mean
                    # 将单通道的目标和预测图扩展为3通道以进行拼接
                    target_vis = targets[i].repeat(3, 1, 1)
                    pred_vis = preds[i].repeat(3, 1, 1)

                    # 水平拼接三张图
                    combined_image = torch.cat([input_vis, target_vis, pred_vis], dim=2)

                    # 定义保存路径
                    save_path = os.path.join(config.EVAL_IMAGES_DIR,
                        f"epoch_{epoch + 1}_step_{current_step}_sample_{i}.png")
                    save_image(combined_image, save_path)
                logger.info(f"已保存第一个验证批次的对比图到: {config.EVAL_IMAGES_DIR}")

    avg_loss = total_loss / len(dataloader)
    pixel_accuracy = 100 * total_correct_pixels / total_pixels
    perfect_match_ratio = 100 * perfect_matches / len(dataloader.dataset)

    logger.info(f"--- Validation @ Epoch {epoch + 1}, Step {current_step} ---")
    logger.info(
        f"    Validation Loss: {avg_loss:.4f}, Pixel Acc: {pixel_accuracy:.2f}%, Perfect Image Match: {perfect_match_ratio:.2f}%")


# --- 6. 训练循环 (已更新对validate的调用) ---
def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    logger.info("\n[3/3] 开始UNET训练循环...")
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for step, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if (step + 1) % config.EVAL_INTERVAL_STEPS==0 or (step + 1)==len(train_loader):
                # 传入config以访问保存路径
                validate(model, val_loader, criterion, device, logger, epoch, step + 1, config)
                model.train()


# --- 7. 主执行函数 ---
if __name__=='__main__':
    config = Config()
    # 【新增】创建所有需要的输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.EVAL_IMAGES_DIR, exist_ok=True)

    logger = setup_logger(config.LOG_FILE)
    device = torch.device(config.DEVICE)

    logger.info(f"使用设备: {device}")
    logger.info("--- 正在为UNET(可视化版)对比实验做准备 ---")

    data_transforms = {
        'input': transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'target': transforms.Compose([transforms.ToTensor()])
    }

    logger.info(f"\n[1/3] 正在从 {config.METADATA_PATH} 加载元数据...")
    full_metadata = pd.read_csv(config.METADATA_PATH)
    full_metadata = full_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(full_metadata) * config.VAL_SPLIT)
    train_metadata, val_metadata = full_metadata[val_size:], full_metadata[:val_size]
    logger.info(f"数据集划分完毕: {len(train_metadata)}训练, {len(val_metadata)}验证。")

    train_dataset = CAImageToImageDataset(train_metadata, config.INITIAL_IMAGES_DIR, config.FINAL_IMAGES_DIR,
        transform=data_transforms)
    val_dataset = CAImageToImageDataset(val_metadata, config.INITIAL_IMAGES_DIR, config.FINAL_IMAGES_DIR,
        transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    logger.info("\n[2/3] 正在初始化UNET模型...")
    model = UNet(n_channels=3, n_classes=1).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"UNET模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")

    criterion = nn.MSELoss()  # BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger)

    logger.info("\n训练完成！")