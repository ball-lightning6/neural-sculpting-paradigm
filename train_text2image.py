import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# --- Tokenizer 配置 ---
TOKENIZER_NAME = "qwen2_0.5b"

# --- TinyTransformer 配置 ---
TINY_TRANSFORMER_CONFIG = {
    'hidden_size': 896,
    'num_hidden_layers': 6,
    'num_attention_heads': 8,
    'intermediate_size': 1536,
    'max_position_embeddings': 256,
    'dropout_prob': 0.1,
}

# --- 图像配置 ---
IMAGE_SIZE = (256, 256)
OUTPUT_CHANNELS = 3  # RGB

# --- 数据集配置 ---
DATASET_DIR = "./cube_dataset_final_highlight"
CAPTIONS_FILE = "metadata.csv"

# --- 训练参数 ---
OUTPUT_DIR = "autodl-tmp/checkpoints_transformer_text2image_cube"
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1800
FP16 = torch.cuda.is_available()
EVAL_EVERY_N_STEPS = 100
LOGGING_STEPS = 20


# ==============================================================================
# --- 2. 核心模型定义 ---
# ==============================================================================

@dataclass
class TinyTransformerConfig:
    vocab_size: int = 320
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 256
    dropout_prob: float = 0.1

    def to_dict(self): return asdict(self)

    def save_json(self, path):
        with open(path, 'w') as f: json.dump(self.to_dict(), f, indent=4)


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_prob,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob)
        )
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, attention_mask=None):
        residual = x
        x_norm = self.ln_1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=attention_mask, need_weights=False)
        x = residual + self.dropout(attn_output)
        residual = x
        x_norm = self.ln_2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)
        return x


@dataclass
class CausalLMOutputWithHiddenStates:
    logits: torch.Tensor
    hidden_states: tuple = None


class TinyTransformerForCausalLM(nn.Module):
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

        token_embeds = self.tok_embeddings(input_ids)
        pos_embeds = self.pos_embeddings(position_ids)
        hidden_states = self.dropout(token_embeds + pos_embeds)

        key_padding_mask = (attention_mask==0) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=key_padding_mask)

        final_hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(final_hidden_states)

        return CausalLMOutputWithHiddenStates(
            logits=logits,
            hidden_states=(final_hidden_states,)
        )


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        if x.shape[2:]!=skip_connection.shape[2:]:
            x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class ImageDecoder(nn.Module):
    def __init__(self, text_feature_dim, image_size=224, output_channels=3):
        super().__init__()
        self.image_size = image_size
        decoder_channels = [128, 256, 512, 1024]
        feature_map_sizes = [
            (image_size // 4, image_size // 4),
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16),
            (image_size // 32, image_size // 32)
        ]
        self.skip_projections = nn.ModuleList([
            self._create_projection(text_feature_dim, ch, size)
            for ch, size in zip(decoder_channels, feature_map_sizes)
        ])
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = UpsampleBlock(decoder_channels[3], decoder_channels[2])
        self.up2 = UpsampleBlock(decoder_channels[2], decoder_channels[1])
        self.up3 = UpsampleBlock(decoder_channels[1], decoder_channels[0])
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[0] // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[0] // 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_map_sizes = feature_map_sizes
        self.decoder_channels = decoder_channels

    def _create_projection(self, text_dim, out_channels, size):
        return nn.Sequential(
            nn.Linear(text_dim, text_dim // 2), nn.ReLU(),
            nn.Linear(text_dim // 2, out_channels * size[0] * size[1])
        )

    def forward(self, text_feature):
        fakes = [
            proj(text_feature).view(
                text_feature.size(0), ch, size[0], size[1]
            ) for proj, ch, size in zip(self.skip_projections, self.decoder_channels, self.feature_map_sizes)
        ]
        f0_fake, f1_fake, f2_fake, f3_fake = fakes
        b = self.bottleneck_conv(f3_fake)
        d1 = self.up1(b, f2_fake)
        d2 = self.up2(d1, f1_fake)
        d3 = self.up3(d2, f0_fake)
        d4 = self.final_up(d3)
        out = nn.functional.interpolate(d4, size=(self.image_size, self.image_size), mode='bilinear',
            align_corners=False)
        return self.final_conv(out)


class TextToImageModel(nn.Module):
    def __init__(self, tokenizer, transformer_config, image_size, output_channels):
        super().__init__()
        self.tokenizer = tokenizer
        print("🤖 初始化 TinyTransformer 文本编码器...")
        self.text_encoder = TinyTransformerForCausalLM(transformer_config)
        print("TinyTransformer 模型已创建。")
        text_feature_dim = transformer_config.hidden_size
        self.decoder = ImageDecoder(text_feature_dim, image_size, output_channels)
        print("U-Net 图像解码器已创建。")

    def forward(self, input_ids, attention_mask, **kwargs):
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = encoder_outputs.hidden_states[-1]
        sequence_lengths = torch.eq(attention_mask, 1).sum(-1) - 1
        text_feature = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            sequence_lengths
        ]
        generated_image = self.decoder(text_feature)
        return generated_image


# ==============================================================================
# --- 3. 数据集与辅助工具 ---
# ==============================================================================

class TextToImageDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, image_size):
        self.image_dir = os.path.join(root_dir, "images")
        self.captions_df = pd.read_csv(os.path.join(root_dir, captions_file))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        caption = row.get('label') or row.get('caption')
        img_name = row.get('filename') or row.get('image_file')
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            target_pixels = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image not found {img_path}, returning None.")
            return None

        # 使用字符级编码
        input_ids = [ord(char) for char in caption]
        input_ids.append(319)  # 自定义结束符
        # print(min(input_ids),max(input_ids))
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
            "labels": target_pixels
        }


class ImageSavingManager:
    """辅助类，用于在评估期间保存预测图像"""

    def __init__(self, eval_dataset, output_dir, tokenizer, num_samples=4):
        self.output_dir = os.path.join(output_dir, "eval_predictions")
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_samples = min(num_samples, len(eval_dataset))
        self.samples = [eval_dataset[i] for i in range(self.num_samples)]
        self.tokenizer = tokenizer
        self.collate_fn = self.create_collate_fn()
        print(f"✅ [Image Saver] 已初始化，将在每次评估时保存 {self.num_samples} 张对比图至 {self.output_dir}")

    def create_collate_fn(self):
        """创建一个collator，用于处理样本批次，特别是填充"""

        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if not batch: return None

            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            labels = torch.stack([item['labels'] for item in batch])

            max_len = TINY_TRANSFORMER_CONFIG['max_position_embeddings']
            padded_input_ids, padded_attention_mask = [], []

            # 假设 pad_token_id 为 0 或其他
            pad_token_id = 319  # self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            for ids, mask in zip(input_ids, attention_mask):
                seq_len = len(ids)
                padding_length = max_len - seq_len

                padded_ids = ids[:max_len]
                padded_mask = mask[:max_len]

                if padding_length > 0:
                    padded_ids = torch.cat(
                        [padded_ids, torch.tensor([pad_token_id] * padding_length, dtype=torch.long)])
                    padded_mask = torch.cat([padded_mask, torch.tensor([0] * padding_length, dtype=torch.long)])

                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)

            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": labels
            }

        return collate_fn

    def save_images(self, model, global_step):
        device = next(model.parameters()).device
        valid_samples = [s for s in self.samples if s is not None]
        if not valid_samples:
            print("[Image Saver] 警告：没有有效的样本用于生成图像。")
            return

        batch = self.collate_fn(valid_samples)
        if batch is None: return

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        model.eval()
        with torch.no_grad():
            generated_images = model(input_ids=input_ids, attention_mask=attention_mask)

        to_pil = transforms.ToPILImage()
        gen_pils = [to_pil(img.cpu()) for img in generated_images]
        target_pils = [to_pil(s['labels']) for s in valid_samples]

        captions_raw = [s['input_ids'].tolist() for s in valid_samples]
        captions = [''.join(map(chr, filter(lambda x: 32 <= x < 319, ids))) for ids in captions_raw]

        for i in range(len(valid_samples)):
            target_img, gen_img = target_pils[i], gen_pils[i]
            comparison_img = Image.new('RGB', (target_img.width * 2, target_img.height))
            comparison_img.paste(target_img, (0, 0))
            comparison_img.paste(gen_img, (target_img.width, 0))

            caption_slug = captions[i][:30].replace(" ", "_").replace("/", "").replace("\\", "")
            save_path = os.path.join(self.output_dir, f"step_{global_step}_sample_{i}_{caption_slug}.png")
            comparison_img.save(save_path)

        print(f"💾 [Image Saver] 已在步骤 {global_step} 保存对比图像。")
        model.train()


# ==============================================================================
# --- 4. 主执行流程 ---
# ==============================================================================

def main():
    # --- 1. 初始化 Tokenizer 和 Config ---
    print("🚀 初始化 Tokenizer 和模型配置...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    TINY_TRANSFORMER_CONFIG['vocab_size'] = 320
    transformer_config = TinyTransformerConfig(**TINY_TRANSFORMER_CONFIG)

    # --- 2. 初始化复合模型 ---
    model = TextToImageModel(
        tokenizer=tokenizer,
        transformer_config=transformer_config,
        image_size=IMAGE_SIZE[0],
        output_channels=OUTPUT_CHANNELS
    )

    # --- 3. 设置可训练参数 ---
    for param in model.parameters(): param.requires_grad = True
    print("\n✅ 可训练参数统计:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数: {total_params / 1e6:.2f}M")

    # --- 4. 准备数据集 ---
    print("\n📦 准备数据集中...")
    if not os.path.exists(DATASET_DIR) or not os.path.exists(os.path.join(DATASET_DIR, CAPTIONS_FILE)):
        print(f"❌ 错误: 数据集目录 '{DATASET_DIR}' 或 captions 文件 '{CAPTIONS_FILE}' 不存在。")
        return

    full_dataset = TextToImageDataset(
        root_dir=DATASET_DIR,
        captions_file=CAPTIONS_FILE,
        tokenizer=tokenizer,
        image_size=IMAGE_SIZE
    )

    if len(full_dataset)==0:
        print("❌ 数据集为空，请检查你的数据集路径和内容。")
        return

    train_size = int(0.99 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    print(f"数据集划分: {len(train_dataset)} 训练, {len(eval_dataset)} 验证")

    # --- 5. 设置 PyTorch 训练环境 ---
    print("\n🔧 正在设置 PyTorch 训练环境...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已移至 {device}")

    # 创建 Collate Function
    image_saver = ImageSavingManager(eval_dataset=eval_dataset, output_dir=OUTPUT_DIR, tokenizer=tokenizer)
    collate_fn = image_saver.create_collate_fn()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, num_workers=2,
        pin_memory=True, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=FP16)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file_path = os.path.join(OUTPUT_DIR, 'log.txt')
    log_f = open(log_file_path, 'w', buffering=1)  # 使用行缓冲
    print(f"日志将保存到: {log_file_path}")

    # --- 6. 开始训练 ---
    print("\n🔥 开始 PyTorch 训练循环！")
    global_step = 0
    train_losses = []
    # model.to(cpu)
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in pbar:
            if batch is None: continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with autocast(enabled=FP16):
                generated_images = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(generated_images, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            if (global_step + 1) % LOGGING_STEPS==0:
                avg_train_loss = np.mean(train_losses[-LOGGING_STEPS:])
                pbar.set_postfix(train_loss=f"{avg_train_loss:.5f}")

            if (global_step + 1) % EVAL_EVERY_N_STEPS==0:
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        if eval_batch is None: continue
                        input_ids_eval = eval_batch['input_ids'].to(device, non_blocking=True)
                        attention_mask_eval = eval_batch['attention_mask'].to(device, non_blocking=True)
                        labels_eval = eval_batch['labels'].to(device, non_blocking=True)

                        with autocast(enabled=FP16):
                            generated_images_eval = model(input_ids=input_ids_eval, attention_mask=attention_mask_eval)
                            eval_loss = loss_fn(generated_images_eval, labels_eval)
                        eval_losses.append(eval_loss.item())

                avg_train_loss_eval_step = np.mean(train_losses[-EVAL_EVERY_N_STEPS:])
                avg_eval_loss = np.mean(eval_losses)

                log_msg = f"Step: {global_step + 1}, Train Loss: {avg_train_loss_eval_step:.6f}, Val Loss: {avg_eval_loss:.6f}\n"
                print(f"\n{log_msg.strip()}")
                log_f.write(log_msg)

                image_saver.save_images(model, global_step + 1)
                model.train()

            global_step += 1

    log_f.close()

    # --- 7. 保存最终模型 ---
    print("\n✅ 训练完成，保存最终模型...")
    final_save_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_save_dir, exist_ok=True)

    torch.save(model.text_encoder.state_dict(), os.path.join(final_save_dir, "final_text_encoder.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(final_save_dir, "final_decoder.pth"))
    model.tokenizer.save_pretrained(final_save_dir)
    print(f"模型组件已保存至 {final_save_dir}")


if __name__=="__main__":
    main()