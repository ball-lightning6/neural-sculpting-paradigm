import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
from dataclasses import dataclass


# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 8  # Diffusion模型显存占用较大
    eval_batch_size = 8
    num_epochs = 150
    learning_rate = 1e-4
    lr_warmup_steps = 500

    validation_split_ratio = 0.1
    eval_steps = 2000
    save_image_steps = 2000
    save_model_steps = 10000

    mixed_precision = "fp16"
    # 【已修改】指向我们的新数据集
    dataset_dir = "ca_img2img_dataset_240"
    output_dir = "diffusion_ca_results_v1"

    num_train_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    beta_schedule = "linear"


config = TrainingConfig()

# ==============================================================================
# --- 2. 模型与调度器 (已修改) ---
# ==============================================================================
model = UNet2DModel(
    sample_size=config.image_size,
    # 【修改】输入通道数为6 (3 for noisy_image + 3 for condition_image)
    in_channels=6,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "AttnDownBlock2D", "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D",
    ),
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=config.num_train_timesteps,
    beta_start=config.beta_start,
    beta_end=config.beta_end,
    beta_schedule=config.beta_schedule,
)


# ==============================================================================
# --- 3. 数据集类 (已修改) ---
# ==============================================================================
class CADiffusionDataset(Dataset):
    """一个为条件生成CA图像而设计的Dataset类。"""

    def __init__(self, metadata_df, initial_dir, final_dir, transform=None):
        self.metadata_df = metadata_df
        self.initial_dir = initial_dir
        self.final_dir = final_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        # 加载初始图像 (条件)
        initial_image_path = os.path.join(self.initial_dir, row['initial_image'])
        initial_image = Image.open(initial_image_path).convert("RGB")

        # 加载最终图像 (目标)
        final_image_path = os.path.join(self.final_dir, row['final_image'])
        final_image = Image.open(final_image_path).convert("RGB")

        if self.transform:
            initial_image = self.transform(initial_image)
            final_image = self.transform(final_image)

        return initial_image, final_image


# ==============================================================================
# --- 4. 评估/采样函数 (已重构) ---
# ==============================================================================
@torch.no_grad()
def evaluate_and_generate(config, model, scheduler, val_dataloader, accelerator, step):
    model.eval()

    # 从验证集中取一个批次的数据用于评估
    initial_images, target_images = next(iter(val_dataloader))

    # 准备生成过程的初始噪声
    generated_images = torch.randn(
        (initial_images.shape[0], model.out_channels, model.sample_size, model.sample_size),
        device=accelerator.device,
    )

    scheduler.set_timesteps(config.num_train_timesteps)

    # 条件生成循环
    for t in tqdm(scheduler.timesteps, desc="Conditional Generation", disable=not accelerator.is_main_process):
        model_input = torch.cat([generated_images, initial_images], dim=1)
        noise_pred = model(model_input, t, return_dict=False)[0]
        generated_images = scheduler.step(noise_pred, t, generated_images).prev_sample

    # --- 计算指标 ---
    # 将生成图像和目标图像二值化以进行精确比较
    preds_binary = (generated_images > 0.0).float()  # [-1, 1]范围, 0.0是分界线
    targets_binary = (target_images > 0.0).float()

    pixel_accuracy = 100 * (preds_binary==targets_binary).sum().item() / targets_binary.numel()
    perfect_matches = 100 * torch.all(preds_binary==targets_binary, dim=(1, 2, 3)).sum().item() / targets_binary.shape[
        0]

    accelerator.log({"val_pixel_accuracy": pixel_accuracy, "val_perfect_match_ratio": perfect_matches}, step=step)

    if accelerator.is_main_process:
        print(f"Step {step}: Pixel Acc: {pixel_accuracy:.2f}%, Perfect Match: {perfect_matches:.2f}%")

        # --- 保存三联图 ---
        # 将图像从[-1, 1]反归一化到[0, 1]
        initial_images_vis = (initial_images / 2 + 0.5).clamp(0, 1)
        target_images_vis = (target_images / 2 + 0.5).clamp(0, 1)
        generated_images_vis = (generated_images / 2 + 0.5).clamp(0, 1)

        # 拼接并保存
        for i in range(initial_images_vis.shape[0]):
            combined_image = torch.cat([initial_images_vis[i], target_images_vis[i], generated_images_vis[i]], dim=2)
            save_path = os.path.join(config.output_dir, "samples", f"step_{step:07d}_sample_{i}.png")
            transforms.ToPILImage()(combined_image.cpu()).save(save_path)

        print(f"评估图像已保存至: {os.path.join(config.output_dir, 'samples')}")

    model.train()  # 切换回训练模式


# ==============================================================================
# --- 5. 训练主循环 (已修改) ---
# ==============================================================================
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )

    if accelerator.is_main_process:
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        accelerator.init_trackers("diffusion_ca_conditional")

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (initial_images, final_images) in enumerate(train_dataloader):
            clean_images = final_images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],),
                device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # 【修改】拼接条件图像和噪声图像作为模型输入
                model_input = torch.cat([noisy_images, initial_images], dim=1)
                noise_pred = model(model_input, timesteps, return_dict=False)[0]

                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())
            accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

            if accelerator.is_main_process:
                if (global_step + 1) % config.eval_steps==0:
                    evaluate_and_generate(config, accelerator.unwrap_model(model), noise_scheduler, val_dataloader,
                        accelerator, global_step + 1)

                    if (global_step + 1) % config.save_model_steps==0:
                        pipeline_model = accelerator.unwrap_model(model)
                        save_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step + 1}")
                        pipeline_model.save_pretrained(save_path)
                        print(f"模型检查点已保存至: {save_path}")

            global_step += 1

    accelerator.end_training()
    print("训练完成！")


# ==============================================================================
# --- 6. 运行入口 (已修改) ---
# ==============================================================================
if __name__=="__main__":
    # 图像预处理，归一化到[-1, 1]
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    full_metadata = pd.read_csv(os.path.join(config.dataset_dir, "metadata.csv"))

    val_size = int(len(full_metadata) * config.validation_split_ratio)
    train_size = len(full_metadata) - val_size

    # 为了保证验证集固定，我们先分割metadata
    train_meta, val_meta = random_split(full_metadata, [train_size, val_size])

    train_dataset = CADiffusionDataset(
        metadata_df=train_meta.dataset.iloc[train_meta.indices],
        initial_dir=os.path.join(config.dataset_dir, "initial_images"),
        final_dir=os.path.join(config.dataset_dir, "final_images"),
        transform=transform
    )
    val_dataset = CADiffusionDataset(
        metadata_df=val_meta.dataset.iloc[val_meta.indices],
        initial_dir=os.path.join(config.dataset_dir, "initial_images"),
        final_dir=os.path.join(config.dataset_dir, "final_images"),
        transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)