import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import timm
import sys

# ==============================================================================
# --- 1. 配置区域 (已更新) ---
# ==============================================================================

# --- 模型配置 ---
# MODEL_NAME = "swin_base_patch4_window7_224_in22k"
MODEL_NAME = "swin_base_patch4_window7_224.ms_in22k"
OUTPUT_CHANNELS = 3

# --- 数据集配置 ---
# DATASET_DIR = "rainwater_image_dataset"
# DATASET_DIR = "maze_path_dataset"
DATASET_DIR = "incircle_dataset"
# DATASET_DIR = "centroid_dataset"
# DATASET_DIR = "symmetry_axis_dataset_FULLY_CONTAINED"
# DATASET_DIR = "catenary_dataset_v3.2_DEFINITIVE_FIX"
# DATASET_DIR = "catenary_dataset_v6_PHYSICALLY_CONSTRAINED"
# DATASET_DIR = "catenary_dataset_v5_CONSTRUCTIVE"
# DATASET_DIR = "refraction_dataset_v1.4_USER_LOGIC"
# DATASET_DIR = "bouncing_ball_dataset_v0.4_variable_bounces"
# DATASET_DIR = "tessellation_dataset_256"
# DATASET_DIR = "orbital_dataset_256_separated_v3"
# DATASET_DIR = "triangle_dataset_pil"
# DATASET_DIR = "arc_final_dataset_mp"
# DATASET_DIR = "arc_recursive_4x4_dataset"
# DATASET_DIR = "arc_conditional_bbox_dataset"
# DATASET_DIR = "arc_final_projection_dataset"
# DATASET_DIR = "arc_conditional_projection_dataset_v2"
# DATASET_DIR = "arc_ultimate_plus_dataset"
# DATASET_DIR = "arc_priority_final_dataset_v2"
# DATASET_DIR = "arc_dynamic_swap_dataset"
# DATASET_DIR = "arc_ultimate_final_v4_dataset"#arc-agi-2
# DATASET_DIR = "arc_ultimate_graduation_final_dataset"#arc-agi-2
# DATASET_DIR = "arc_cross_pattern_dataset"#arc-agi-2
# DATASET_DIR = "arc_periodic_pattern_dataset"#arc-agi-2
# DATASET_DIR = "arc_final_sorting_dataset"#arc-agi-2
# DATASET_DIR = "arc_meta_reasoning_dataset"#arc-agi-2
# DATASET_DIR = "arc_fluid_final_dataset"#arc-agi-2
# DATASET_DIR = "arc_jigsaw_puzzle_masterpiece_dataset"#arc-agi-2
# DATASET_DIR = "arc_jigsaw_puzzle_mine_dataset"  # arc-agi-2，和上面同一个任务，数据集更好
IMAGE_SIZE = (224, 224)
# IMAGE_SIZE = (256, 256)

# --- 训练参数 ---
OUTPUT_DIR = "./checkpoints_incircle"
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt")
BATCH_SIZE = 16
# LEARNING_RATE = 1e-4#通用，nan后改为下面
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1800
FP16 = torch.cuda.is_available()

PRETRAINED_MODEL_PATH = "./swin-base-patch4-window7-224-in22k1/pytorch_model.bin"

# --- 全新：按步数评估的配置 ---
EVAL_EVERY_N_STEPS = 200  # <--- 每500个训练步数，进行一次验证
SAVE_IMAGE_EVERY_N_STEPS = 200  # <--- 每1000个训练步数，保存一次对比图


# ==============================================================================
# --- 2. 自定义图像数据集 (保持不变) ---
# ==============================================================================

class ImageToImageDataset(Dataset):
    """一个专门用于读取 Image-to-Image 任务数据的Dataset。"""

    def __init__(self, root_dir, image_size=(224, 224)):
        self.input_dir = os.path.join(root_dir, "input")
        self.output_dir = os.path.join(root_dir, "output")
        self.image_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.png')])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_img_path = os.path.join(self.input_dir, img_name)
        output_img_path = os.path.join(self.output_dir, img_name)
        input_image = Image.open(input_img_path).convert("RGB")
        output_image = Image.open(output_img_path).convert("RGB")
        input_tensor = self.transform(input_image)
        output_tensor = self.transform(output_image)
        return {"input_pixel_values": input_tensor, "output_pixel_values": output_tensor}


class ImageToImageDataset_maze(Dataset):
    """
    一个专门用于读取 Image-to-Image 任务数据的Dataset。
    能处理输入和输出图像在同一目录下的情况 (例如 '0_input.png', '0_output.png')。
    """

    def __init__(self, root_dir, mode='train', image_size=(224, 224)):
        """
        Args:
            root_dir (string): 数据集根目录 (例如 'maze_path_dataset')。
            mode (string): 'train' 或 'eval'，决定加载哪个子目录。
            image_size (tuple): 图像将被调整到的尺寸。
        """
        assert mode in ['train', 'eval'], "mode 必须是 'train' 或 'eval'"

        # 1. 【修改】根据 mode 确定数据目录
        self.data_dir = os.path.join(root_dir, mode)

        # 2. 【修改】智能地查找所有输入文件
        # 我们只查找 '_input.png' 文件，然后推断出对应的输出文件名
        self.input_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('_input.png')])

        # 3. 【改进】保持你原有的 transform 逻辑
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),  # 使用最近邻插值，避免颜色模糊
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # 4. 【修改】构建输入和输出文件的完整路径
        input_filename = self.input_files[idx]
        # 通过替换字符串来获得输出文件名，这比假设它们在不同目录中更可靠
        output_filename = input_filename.replace('_input.png', '_output.png')

        input_img_path = os.path.join(self.data_dir, input_filename)
        output_img_path = os.path.join(self.data_dir, output_filename)

        try:
            input_image = Image.open(input_img_path).convert("RGB")
            output_image = Image.open(output_img_path).convert("RGB")
        except FileNotFoundError:
            # 增加一个健壮性检查，如果找不到对应的输出文件，给出清晰的错误提示
            raise FileNotFoundError(f"错误：找到了输入文件 '{input_img_path}' 但找不到对应的输出文件 '{output_img_path}'。请检查数据集是否完整。")

        input_tensor = self.transform(input_image)
        output_tensor = self.transform(output_image)

        # 返回的字典键名保持不变，以便与你的训练循环兼容
        return {"input_pixel_values": input_tensor, "output_pixel_values": output_tensor}
        # 注意：我将键名从 "input_pixel_values" 和 "output_pixel_values"
        # 改为 "pixel_values" 和 "labels"。这更符合 Hugging Face Transformers 的习惯，
        # 如果你的训练循环使用的是前者，请改回即可。
        # return {"input_pixel_values": input_tensor, "output_pixel_values": output_tensor}


# ==============================================================================
# --- 3. Swin-Unet 模型 (保持不变) ---
# ==============================================================================

class UpsampleBlock(nn.Module):
    """一个U-Net的上采样块，包含上采样、拼接和卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 上采样层：输入in_channels, 输出out_channels
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # ★★★ 关键修正 ★★★
        # 卷积块接收的是拼接后的张量，其通道数是 skip_connection的通道数 + 上采样后的通道数。
        # skip_connection的通道数就是out_channels，上采样后的通道数也是out_channels。
        # 所以，这里的输入通道数是 out_channels * 2。
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class SwinUnet(nn.Module):
    def __init__(self, model_name=MODEL_NAME, output_channels=OUTPUT_CHANNELS, pretrained=True):
        super().__init__()

        # 编码器部分保持不变
        self.encoder = timm.create_model(
            model_name,
            # 注意：在离线加载的场景下，这里应该设为 False
            # pretrained=False, # pretrained 参数在主函数中处理
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        encoder_channels = self.encoder.feature_info.channels()

        # 解码器部分也保持不变
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[3], encoder_channels[3] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = UpsampleBlock(encoder_channels[3] * 2, encoder_channels[2])
        self.up2 = UpsampleBlock(encoder_channels[2], encoder_channels[1])
        self.up3 = UpsampleBlock(encoder_channels[1], encoder_channels[0])
        self.up4 = nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0] // 2, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0] // 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 从编码器获取特征
        features = self.encoder(x)

        # ★★★★★★★★★★★  关键修正：添加 Permute 操作 ★★★★★★★★★★★
        # 将所有特征图从 (N, H, W, C) 转换为 (N, C, H, W)
        f0, f1, f2, f3 = [feat.permute(0, 3, 1, 2) for feat in features]
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # 2. 解码过程 (现在可以正常工作了)
        b = self.bottleneck(f3)
        d1 = self.up1(b, f2)
        d2 = self.up2(d1, f1)
        d3 = self.up3(d2, f0)

        d4 = self.up4(d3)
        out = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)

        return self.final_conv(out)


# ==============================================================================
# --- 4. 主训练流程 (已修改为按步数评估) ---
# ==============================================================================

def evaluate_and_save(model, eval_loader, criterion, device, best_eval_loss, global_step):
    """独立的评估函数"""
    model.eval()
    eval_loss = 0.0
    # eval_pbar = tqdm(eval_loader, desc=f"Step {global_step} [验证]", file=sys.stdout)

    with torch.no_grad():
        for batch in eval_loader:  # eval_pbar:
            inputs = batch["input_pixel_values"].to(device)
            targets = batch["output_pixel_values"].to(device)

            with torch.cuda.amp.autocast(enabled=FP16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            eval_loss += loss.item()
            # eval_pbar.set_postfix({"loss": loss.item()})

    avg_eval_loss = eval_loss / len(eval_loader)

    log_message = f"Step {global_step} | Eval Loss: {avg_eval_loss:.6f}"
    print(log_message)
    with open(LOG_FILE, "a") as f:
        f.write(log_message + "\n")

    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"🎉 新的最佳模型已保存至 {save_path} (Step: {global_step}, Eval Loss: {best_eval_loss:.6f})")

    return best_eval_loss


def save_comparison_images(model, eval_loader, device, global_step):
    """独立的保存对比图函数"""
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(eval_loader))
        sample_input = sample_batch["input_pixel_values"][:20].to(device)
        sample_output = model(sample_input)

        to_pil = transforms.ToPILImage()
        for i in range(sample_input.size(0)):
            input_img = to_pil(sample_batch["input_pixel_values"][i])
            target_img = to_pil(sample_batch["output_pixel_values"][i])
            pred_img = to_pil(sample_output[i].cpu())

            comparison_img = Image.new('RGB', (IMAGE_SIZE[0] * 3, IMAGE_SIZE[1]))
            comparison_img.paste(input_img, (0, 0))
            comparison_img.paste(pred_img, (IMAGE_SIZE[0], 0))
            comparison_img.paste(target_img, (IMAGE_SIZE[0] * 2, 0))

            comparison_img.save(os.path.join(OUTPUT_DIR, f"step_{global_step}_sample_{i}.png"))
    print(f"已保存对比图像至 {OUTPUT_DIR}")


def main():
    # --- 准备环境 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 准备数据集 ---
    print("准备数据集...")
    full_dataset = ImageToImageDataset(root_dir=DATASET_DIR, image_size=IMAGE_SIZE)
    # full_dataset = ImageToImageDataset_maze(root_dir=DATASET_DIR, image_size=IMAGE_SIZE)
    train_size = int(0.998 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1,
        pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"数据集划分: {len(train_dataset)} 训练样本, {len(eval_dataset)} 验证样本。")

    # --- 初始化模型、损失函数和优化器 ---
    print("初始化 Swin-Unet 模型...")

    # ★★★ 关键改动：分两步创建和加载模型 ★★★

    # 1. 创建模型结构，但设置 pretrained=False，避免在线下载
    print(f"创建模型结构: {MODEL_NAME}")
    model = SwinUnet(model_name=MODEL_NAME, pretrained=False).to(device)

    # 2. 从本地文件加载预训练权重
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"从本地路径加载预训练权重: {PRETRAINED_MODEL_PATH}")
        # 加载整个模型的状态字典
        state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)

        # SwinUnet包含编码器和解码器，而我们的权重文件只包含编码器部分。
        # 我们需要智能地只加载编码器的权重。

        # 提取只属于编码器的权重
        encoder_state_dict = {}
        for k, v in state_dict.items():
            # 我们的SwinUnet模型中，编码器被命名为 self.encoder
            # timm下载的权重键名没有这个前缀，所以我们要加上
            encoder_key = f"encoder.{k}"
            if encoder_key in model.state_dict():
                encoder_state_dict[encoder_key] = v

        # 加载权重，strict=False允许只加载部分权重（只加载编码器，不加载解码器）
        model.load_state_dict(encoder_state_dict, strict=False)
        print("编码器权重加载成功！解码器权重将随机初始化。")

    else:
        print(f"警告: 未找到预训练模型文件于 '{PRETRAINED_MODEL_PATH}'。将从头开始训练。")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    # --- 开始训练 ---
    print(f"🚀 开始 Image-to-Image 训练，每 {EVAL_EVERY_N_STEPS} 步评估一次...")
    best_eval_loss = float('inf')
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in train_pbar:
            inputs = batch["input_pixel_values"].to(device)
            targets = batch["output_pixel_values"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=FP16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            train_pbar.set_postfix({"train_loss": f"{loss.item():.6f}", "step": global_step})

            # --- 按步数进行评估和保存 ---
            if global_step % EVAL_EVERY_N_STEPS==0:
                # 调用评估函数
                best_eval_loss = evaluate_and_save(
                    model, eval_loader, criterion, device, best_eval_loss, global_step
                )
                # 评估后切回训练模式
                model.train()

            if global_step % SAVE_IMAGE_EVERY_N_STEPS==0:
                # 调用保存图像函数
                save_comparison_images(model, eval_loader, device, global_step)
                # 评估后切回训练模式
                model.train()

    print("✅ 训练完成！")


if __name__=="__main__":
    main()
