"""
高性能 UNET 模型 (V3 - 逻辑对齐版)
核心优化：
- Upsample (Nearest) + Conv2d: 镜像 Swin-UNet 的成功经验，彻底消除点状伪影。
- Strided Convolution: 替代 MaxPool2d，通过学习保留格点间的空间位置关系。
- GroupNorm: 适合高维度逻辑拟合，比 BatchNorm 更稳定。
- 保持 MSELoss: 以维持与 Swin-UNet 实验的变量一致性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import logging


# --- 1. 配置区域 ---
class Config:
    DATASET_DIR = "ca_img2img_dataset_240"
    OUTPUT_DIR = "unet_v3_logic_results"
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
    INITIAL_IMAGES_DIR = os.path.join(DATASET_DIR, "initial_images")
    FINAL_IMAGES_DIR = os.path.join(DATASET_DIR, "final_images")
    EVAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "eval_images")

    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.01
    LOG_FILE = os.path.join(OUTPUT_DIR, "training_log_unet_v3.log")
    EVAL_INTERVAL_STEPS = 500


# --- 2. 日志与数据加载 ---
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    sh = logging.StreamHandler();
    sh.setFormatter(formatter);
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file, mode='w');
    fh.setFormatter(formatter);
    logger.addHandler(fh)
    return logger


class CAImageToImageDataset(Dataset):
    def __init__(self, metadata_df, initial_dir, final_dir, transform=None):
        self.metadata_df, self.initial_dir, self.final_dir, self.transform = metadata_df, initial_dir, final_dir, transform

    def __len__(self): return len(self.metadata_df)

    def __getitem__(self, index):
        row = self.metadata_df.iloc[index]
        initial_image = Image.open(os.path.join(self.initial_dir, row['initial_image'])).convert("RGB")
        final_image = Image.open(os.path.join(self.final_dir, row['final_image'])).convert("L")
        if self.transform:
            initial_image = self.transform['input'](initial_image)
            final_image = self.transform['target'](final_image)
        return initial_image, final_image


# --- 新增：极简全局自注意力模块 ---
class GlobalAttention(nn.Module):
    """
    一个极简的空间自注意力模块，用于在 Bottleneck 层建立全局逻辑关联。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始为0，让模型平滑过渡

    def forward(self, x):
        batch, c, h, w = x.size()
        # 投影并展平
        proj_query = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)  # [B, N, C']
        proj_key = self.key(x).view(batch, -1, h * w)  # [B, C', N]

        # 计算注意力地图
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)

        # 加权求和
        proj_value = self.value(x).view(batch, -1, h * w)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)

        # 残差连接：x + gamma * out
        return x + self.gamma * out


# --- 3. 核心模型定义 (V3 Logic-Aligned) ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),  # 8组 GroupNorm，稳定逻辑梯度
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
    """使用步长卷积实现下采样，不丢失空间精度"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x): return self.down_conv(x)


class Up(nn.Module):
    """
    使用最近邻插值 + 卷积，彻底解决反卷积带来的点状伪影。
    Nearest 插值最符合元胞自动机的格点特性。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 尺寸对齐处理
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetV3(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetV3, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 【核心改动】：在最深处加入全局大脑
        self.global_brain = GlobalAttention(1024)

        # 最后一层：卷积 + Sigmoid
        self.outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.global_brain(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# --- 4. 训练与验证 ---

def validate(model, dataloader, criterion, device, logger, epoch, current_step, config):
    model.eval()
    total_loss, total_correct_pixels, perfect_matches, total_pixels = 0.0, 0, 0, 0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            preds = (outputs > 0.5).float()  # 由于末端是 Sigmoid，直接阈值判定

            total_correct_pixels += (preds==targets).sum().item()
            perfect_matches += torch.all(preds==targets, dim=(1, 2, 3)).sum().item()
            total_pixels += targets.numel()

            if batch_idx==0:  # 存图观察
                for i in range(min(inputs.size(0), 4)):
                    in_v = inputs[i] * std + mean
                    ta_v = targets[i].repeat(3, 1, 1)
                    pr_v = preds[i].repeat(3, 1, 1)
                    combined = torch.cat([in_v, ta_v, pr_v], dim=2)
                    save_image(combined, os.path.join(config.EVAL_IMAGES_DIR, f"E{epoch + 1}_S{current_step}_I{i}.png"))

    avg_loss = total_loss / len(dataloader)
    pixel_acc = 100 * total_correct_pixels / total_pixels
    perfect_match = 100 * perfect_matches / len(dataloader.dataset)
    logger.info(
        f"Step {current_step}: Loss: {avg_loss:.6f}, Pixel Acc: {pixel_acc:.2f}%, PERFECT: {perfect_match:.2f}%")
    return perfect_match


def train_loop(model, train_loader, val_loader, criterion, optimizer, device, config, logger):
    for epoch in range(config.EPOCHS):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
        for step, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if (step + 1) % config.EVAL_INTERVAL_STEPS==0:
                perf = validate(model, val_loader, criterion, device, logger, epoch, step + 1, config)
                if perf > 99.9: return
                model.train()


if __name__=='__main__':
    cfg = Config()
    os.makedirs(cfg.EVAL_IMAGES_DIR, exist_ok=True)
    logger = setup_logger(cfg.LOG_FILE)

    transform = {
        'input': transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'target': transforms.Compose([transforms.ToTensor()])
    }

    full_metadata = pd.read_csv(cfg.METADATA_PATH)
    train_metadata = full_metadata.sample(frac=0.99, random_state=42)
    val_metadata = full_metadata.drop(train_metadata.index)

    train_loader = DataLoader(
        CAImageToImageDataset(train_metadata, cfg.INITIAL_IMAGES_DIR, cfg.FINAL_IMAGES_DIR, transform),
        batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        CAImageToImageDataset(val_metadata, cfg.INITIAL_IMAGES_DIR, cfg.FINAL_IMAGES_DIR, transform),
        batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = UNetV3(n_channels=3, n_classes=1).to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loop(model, train_loader, val_loader, criterion, optimizer, cfg.DEVICE, cfg, logger)