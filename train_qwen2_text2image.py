import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import time
from transformers import TrainerState, TrainerControl


# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# --- 模型配置 ---
# 文本编码器，使用你的目标模型
TEXT_MODEL_NAME = "qwen2_0.5b"
# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
# 你的U-Net解码器需要的输入通道数，等于Qwen2的隐藏层维度
# Qwen2-0.5B 的 hidden_size 是 896
DECODER_INPUT_CHANNELS = 896

# --- 图像配置 ---
# IMAGE_SIZE = (224, 224)
# IMAGE_SIZE = (256, 256)
IMAGE_SIZE = (240, 240)
OUTPUT_CHANNELS = 3  # RGB

# --- 数据集配置 ---
# 你的数据集根目录，里面应该包含一个"images"文件夹和一个"captions.csv"文件
# DATASET_DIR = "./checkerboard_dataset"
# DATASET_DIR = "./triangle_dataset"
# DATASET_DIR = "./cube_dataset_final_highlight"
DATASET_DIR = "./ca_render_dataset_240"
CAPTIONS_FILE = "metadata.csv"  # csv文件应包含 'image_file' 和 'caption' 两列

# --- 训练参数 ---
OUTPUT_DIR = "./autodl-tmp/checkpoints_qwen2_text2image_ca"
BATCH_SIZE = 32  # 图像生成任务通常需要更小的batch size
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1800
FP16 = torch.cuda.is_available()
EVAL_EVERY_N_STEPS = 100
LOGGING_STEPS = 20


# ==============================================================================
# --- 2. 核心模型定义 (TextToImageModel) ---
# ==============================================================================
class SaveImagePredictionCallback(TrainerCallback):
    """一个在评估时保存模型预测图像对比的回调函数。"""

    def __init__(self, eval_dataset, output_dir, num_samples=32):
        super().__init__()
        self.output_dir = os.path.join(output_dir, "eval_predictions")
        os.makedirs(self.output_dir, exist_ok=True)

        # 从评估数据集中固定取几个样本，确保每次评估都用同样的样本进行对比
        self.num_samples = min(num_samples, len(eval_dataset))
        self.samples = [eval_dataset[i] for i in range(self.num_samples)]

        print(f"✅ [Callback] 已初始化，将在每次评估时保存 {self.num_samples} 张对比图至 {self.output_dir}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """在评估事件结束时被调用。"""
        # 从kwargs中获取模型和分词器
        model = kwargs["model"]
        tokenizer = model.tokenizer  # 从复合模型中获取tokenizer
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # 如果模型没有任何参数（虽然不太可能），可以从TrainingArguments获取
            device = args.device

        # 准备输入数据
        input_ids = torch.stack([s['input_ids'] for s in self.samples]).to(device)
        attention_mask = torch.stack([s['attention_mask'] for s in self.samples]).to(device)

        # 将模型置于评估模式，并关闭梯度计算
        model.eval()
        with torch.no_grad():
            # 使用当前模型生成图像
            generated_images = model(input_ids=input_ids, attention_mask=attention_mask)

        # 将生成的图像和目标图像转换回PIL Image格式
        to_pil = transforms.ToPILImage()
        gen_pils = [to_pil(img.cpu()) for img in generated_images]
        target_pils = [to_pil(s['labels']) for s in self.samples]  # Dataset返回'labels'

        # 将输入文本也解码出来，作为文件名的一部分
        captions = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in input_ids
        ]

        # 拼接并保存图像
        for i in range(self.num_samples):
            target_img = target_pils[i]
            gen_img = gen_pils[i]

            # 创建一个宽度为两倍的大图
            comparison_img = Image.new('RGB', (target_img.width * 2, target_img.height))
            comparison_img.paste(target_img, (0, 0))
            comparison_img.paste(gen_img, (target_img.width, 0))

            # 用global_step和caption来命名文件
            step = state.global_step
            # 清理caption，使其适合做文件名
            caption_slug = captions[i][:30].replace(" ", "_").replace("/", "")
            save_path = os.path.join(self.output_dir, f"step_{step}_sample_{i}_{caption_slug}.png")
            comparison_img.save(save_path)

        print(f"💾 [Callback] 已在步骤 {state.global_step} 保存对比图像。")

        # 记得将模型切回训练模式，以便训练继续
        model.train()


class UpsampleBlock(nn.Module):
    """你的U-Net上采样块，保持不变"""

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
    """将你的U-Net解码器部分封装成一个独立的模块"""

    def __init__(self, text_feature_dim, image_size=224, output_channels=3):
        super().__init__()
        self.image_size = image_size

        # 模仿Swin-Unet的结构，定义解码器需要的特征图尺寸和通道数
        decoder_channels = [128, 256, 512, 1024]
        feature_map_sizes = [
            (image_size // 4, image_size // 4),  # f0
            (image_size // 8, image_size // 8),  # f1
            (image_size // 16, image_size // 16),  # f2
            (image_size // 32, image_size // 32)  # f3
        ]

        # 核心：将文本向量投影成不同尺寸的2D特征图，作为"假的"跳跃连接
        self.skip_projections = nn.ModuleList([
            self._create_projection(text_feature_dim, ch, size)
            for ch, size in zip(decoder_channels, feature_map_sizes)
        ])

        # U-Net解码器结构
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = UpsampleBlock(decoder_channels[3], decoder_channels[2])
        self.up2 = UpsampleBlock(decoder_channels[2], decoder_channels[1])
        self.up3 = UpsampleBlock(decoder_channels[1], decoder_channels[0])

        # 最终输出层
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[0] // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[0] // 2, output_channels, kernel_size=1),
            nn.Sigmoid()  # 或 nn.Tanh()
        )

        self.feature_map_sizes = feature_map_sizes
        self.decoder_channels = decoder_channels

    def _create_projection(self, text_dim, out_channels, size):
        return nn.Sequential(
            nn.Linear(text_dim, text_dim // 2), nn.ReLU(),
            nn.Linear(text_dim // 2, out_channels * size[0] * size[1])
        )

    def forward(self, text_feature):
        # 制造_所有_跳跃连接
        fakes = [
            proj(text_feature).view(
                text_feature.size(0), ch, size[0], size[1]
            ) for proj, ch, size in zip(self.skip_projections, self.decoder_channels, self.feature_map_sizes)
        ]
        f0_fake, f1_fake, f2_fake, f3_fake = fakes

        # 运行解码器
        b = self.bottleneck_conv(f3_fake)
        d1 = self.up1(b, f2_fake)
        d2 = self.up2(d1, f1_fake)
        d3 = self.up3(d2, f0_fake)

        # 生成最终图像
        d4 = self.final_up(d3)
        out = nn.functional.interpolate(d4, size=(self.image_size, self.image_size), mode='bilinear',
            align_corners=False)
        return self.final_conv(out)


class TextToImageModel(nn.Module):
    """复合模型：封装Qwen2 (PEFT) 和 图像解码器"""

    def __init__(self, text_model_name, image_size, output_channels):
        super().__init__()
        # 加载基础的Qwen2模型和Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(text_model_name, trust_remote_code=True)

        # 应用LoRA
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Qwen2的常用目标模块
            lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.text_encoder = get_peft_model(base_model, lora_config)
        print("Qwen2 LoRA模型已创建。")
        self.text_encoder.print_trainable_parameters()

        # 初始化图像解码器
        text_feature_dim = self.text_encoder.config.hidden_size
        self.decoder = ImageDecoder(text_feature_dim, image_size, output_channels)

    def forward(self, input_ids, attention_mask, **kwargs):
        # 1. 从Qwen2获取文本特征
        # 我们需要的是最后一个隐藏层状态，而不是lm_head的输出
        # .model 会绕过lm_head，直接调用底层的transformer模块
        encoder_outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # encoder_outputs 是一个 BaseModelOutputWithPast 对象，
        # 它包含 .last_hidden_state

        # 获取所有token的隐藏状态，形状为 [batch_size, sequence_length, hidden_size]
        hidden_states = encoder_outputs.hidden_states

        # 获取最后一层的隐藏状态，其形状为 [batch_size, sequence_length, hidden_size]
        last_hidden_state = hidden_states[-1]

        # 2. 提取代表整个句子语义的向量
        # (这部分逻辑保持不变)
        sequence_lengths = torch.eq(attention_mask, 1).sum(-1) - 1

        # text_feature 的形状是 [batch_size, hidden_size]
        # 例如：[32, 896]
        text_feature = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            sequence_lengths
        ]

        # 3. 用正确的文本特征驱动图像解码器
        generated_image = self.decoder(text_feature)
        return generated_image


# ==============================================================================
# --- 3. 数据集与训练器定义 ---
# ==============================================================================

class TextToImageDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, image_size):
        self.image_dir = os.path.join(root_dir, "images") # root_dir
        self.captions_df = pd.read_csv(os.path.join(root_dir, captions_file))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        caption = row['label']
        img_name = row['filename']

        # 加载和转换目标图片
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            target_pixels = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image not found {img_path}, skipping.")
            # 返回一个dummy数据，collate_fn会处理
            return None

        # 分词文本
        # tokenized_caption = self.tokenizer(
        #     caption, padding="max_length", truncation=True,
        #     max_length=128, return_tensors="pt"
        # )
        input_ids = []
        for char in caption:
            input_ids.append(self.tokenizer.convert_tokens_to_ids(str(char)))
        if self.tokenizer.eos_token_id is not None:
            input_ids.append(151643)  # tokenizer.eos_token_id)
        else:
            print("警告: Tokenizer没有定义eos_token_id。")
        attention_mask = [1] * len(input_ids)
        attention_mask[-1] = 0
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": target_pixels  # 使用 'pixel_values' 作为key
        }
        # return {
        #     "input_ids": tokenized_caption['input_ids'].squeeze(0),
        #     "attention_mask": tokenized_caption['attention_mask'].squeeze(0),
        #     "pixel_values": target_pixels # 使用 'pixel_values' 作为key
        # }


class ImageGenTrainer(Trainer):
    """自定义Trainer，用于计算图像重建损失"""

    def compute_loss(self, model, inputs, return_outputs=False):
        # 从输入中分离出目标图像
        # print(list(inputs.keys()))
        labels = inputs.pop("labels")

        # 获取模型生成的图像
        # **inputs 包含 'input_ids' 和 'attention_mask'
        generated_images = model(**inputs)

        # 使用MSE或L1作为图像重建损失
        loss_fct = nn.MSELoss()
        loss = loss_fct(generated_images, labels)

        return (loss, {"outputs": generated_images}) if return_outputs else loss

    def evalua1te(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix='eval',
    ):
        """
        重写评估方法，手动计算并注入eval_loss。
        """
        # 1. 获取评估数据加载器
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # 2. 调用父类的 evaluation_loop 来获取基础指标和模型输出
        #    注意：这个loop本身不会计算我们的自定义loss
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True,  # 设为True，因为它不知道怎么算loss，可以跳过
            metric_key_prefix=metric_key_prefix,
        )

        # 3. 手动计算我们自己的评估损失
        total_eval_loss = 0.0
        num_eval_samples = 0

        # 将模型置于评估模式
        model = self._wrap_model(self.model, training=False)
        model.eval()

        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                # 直接调用compute_loss来得到损失值
                loss, _ = self.compute_loss(model, inputs, return_outputs=True)

            # 累加损失 (注意多GPU情况下的聚合)
            total_eval_loss += loss.item() * len(inputs["input_ids"])
            num_eval_samples += len(inputs["input_ids"])

        # 4. 计算平均损失
        avg_eval_loss = total_eval_loss / num_eval_samples

        # 5. 将我们的eval_loss添加到指标字典中
        #    output.metrics 是父类方法返回的基础指标（如runtime）
        #    我们必须使用 'eval_loss' 这个键名！
        output.metrics[f"{metric_key_prefix}_loss"] = avg_eval_loss
        print('eval loss: ', avg_eval_loss)

        end_time = time.time()
        # 补充其他缺失的指标
        output.metrics[f"{metric_key_prefix}_runtime"] = end_time - start_time
        output.metrics[f"{metric_key_prefix}_samples_per_second"] = num_eval_samples / (end_time - start_time)

        # 6. 记录日志并返回
        # self.log(output.metrics) # 这行由trainer在外部调用，我们不用自己调

        return output.metrics


class SaveDecoderCallback(TrainerCallback):
    """一个回调，用于定期保存解码器部分的权重"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        # on_save 在每次 checkpoint 保存时触发
        model = kwargs["model"]
        save_path = os.path.join(self.save_dir, f"decoder_step_{state.global_step}.pth")
        if state.global_step % 1000000==0:
            torch.save(model.decoder.state_dict(), save_path)
            print(f"💾 单独保存图像解码器权重 -> {save_path}")


# ==============================================================================
# --- 4. 主执行流程 ---
# ==============================================================================

def main():
    # --- 1. 初始化复合模型 ---
    print("🚀 初始化Text-to-Image模型...")
    model = TextToImageModel(
        text_model_name=TEXT_MODEL_NAME,
        image_size=IMAGE_SIZE[0],
        output_channels=OUTPUT_CHANNELS
    )

    # --- 2. 关键：设置可训练参数 ---
    # 默认 LoRA 已经设置好了 text_encoder 的可训练参数
    # 我们需要手动解冻下游的解码器
    for param in model.decoder.parameters():
        param.requires_grad = True

    print("\n✅ 可训练参数设置完成:")
    model.text_encoder.print_trainable_parameters()

    # 统计解码器的可训练参数
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"图像解码器 (ImageDecoder) 可训练参数: {decoder_params / 1e6:.2f}M")

    # --- 3. 准备数据集 ---
    print("\n📦 准备数据集中...")
    # 检查数据集是否存在
    if not os.path.exists(DATASET_DIR) or not os.path.exists(os.path.join(DATASET_DIR, CAPTIONS_FILE)):
        print(f"❌ 错误: 数据集目录 '{DATASET_DIR}' 或 captions 文件 '{CAPTIONS_FILE}' 不存在。")
        print("请创建一个目录，包含 'images' 子目录和 'captions.csv' 文件。")
        # 创建一个示例以供参考
        os.makedirs(os.path.join(DATASET_DIR, 'images'), exist_ok=True)
        sample_df = pd.DataFrame([
            {'image_file': 'sample1.png', 'caption': 'a red circle on a white background'},
            {'image_file': 'sample2.png', 'caption': 'a blue square fading to green'}
        ])
        sample_df.to_csv(os.path.join(DATASET_DIR, CAPTIONS_FILE), index=False)
        print("已创建示例 captions.csv。请放入你的数据。")
        return

    full_dataset = TextToImageDataset(
        root_dir=DATASET_DIR,
        captions_file=CAPTIONS_FILE,
        tokenizer=model.tokenizer,
        image_size=IMAGE_SIZE
    )

    # 过滤掉加载失败的None项
    # full_dataset.samples = [s for s in full_dataset.samples if s is not None]

    if len(full_dataset)==0:
        print("❌ 数据集为空，请检查你的数据集路径和内容。")
        return

    train_size = int(0.995 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    print(f"数据集划分: {len(train_dataset)} 训练, {len(eval_dataset)} 验证")

    save_image_callback = SaveImagePredictionCallback(
        eval_dataset=eval_dataset,
        output_dir=OUTPUT_DIR,
        num_samples=32  # 你想保存的样本数量
    )

    # --- 4. 设置训练参数和训练器 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="steps",
        eval_steps=EVAL_EVERY_N_STEPS,
        save_strategy="steps",
        save_steps=EVAL_EVERY_N_STEPS*10000,
        fp16=FP16,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        # save_total_limit=3,
        # report_to="none", # 或者 "tensorboard"
        remove_unused_columns=False,
        dataloader_num_workers=2,  # <-- 设置为你CPU核心数的一半或更多，根据实际情况调整
        dataloader_prefetch_factor=1,
        dataloader_pin_memory=True,  # <-- 强烈建议开启
        save_safetensors=False,
    )

    trainer = ImageGenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[save_image_callback, SaveDecoderCallback(save_dir=OUTPUT_DIR)]  # 添加自定义回调
    )

    # --- 5. 开始训练 ---
    print("\n🔥 开始训练！")
    trainer.train()

    # --- 6. 保存最终模型 ---
    print("\n✅ 训练完成，保存最终模型...")
    # 保存LoRA适配器
    model.text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_adapter"))
    # 保存解码器
    torch.save(model.decoder.state_dict(), os.path.join(OUTPUT_DIR, "final_decoder.pth"))
    # 保存tokenizer
    model.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_tokenizer"))
    print(f"模型已保存至 {OUTPUT_DIR}")


if __name__=="__main__":
    main()