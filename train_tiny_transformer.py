import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn import BCEWithLogitsLoss, Module, ModuleList, LayerNorm, Linear, Embedding, Dropout, Sequential, GELU, \
    MultiheadAttention
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tqdm import tqdm  # 引入tqdm来显示进度条
from torch.nn import CrossEntropyLoss

# 确保 CUDA 可用
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 使用设备: {device} ---")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(412)


# --- 模型和数据类的定义 (完全保持不变) ---
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


class TransformerBlock(Module):
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads,
            dropout=config.dropout_prob, batch_first=True)
        self.ffn = Sequential(Linear(config.hidden_size, config.intermediate_size), GELU(),
            Linear(config.intermediate_size, config.hidden_size), Dropout(config.dropout_prob))
        self.ln_1 = LayerNorm(config.hidden_size);
        self.ln_2 = LayerNorm(config.hidden_size);
        self.dropout = Dropout(config.dropout_prob)

    def forward(self, x, attention_mask=None):
        residual = x;
        x_norm = self.ln_1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=attention_mask, need_weights=False)
        x = residual + self.dropout(attn_output);
        residual = x;
        x_norm = self.ln_2(x);
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output);
        return x


class TinyTransformerForCausalLM(Module):
    @dataclass
    class CausalLMOutput: logits: torch.Tensor

    def __init__(self, config: TinyTransformerConfig):
        super().__init__();
        self.config = config
        self.tok_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = LayerNorm(config.hidden_size);
        self.dropout = Dropout(config.dropout_prob)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_length = input_ids.shape;
        device = input_ids.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
        token_embeds = self.tok_embeddings(input_ids);
        pos_embeds = self.pos_embeddings(position_ids)
        hidden_states = self.dropout(token_embeds + pos_embeds)
        key_padding_mask = (attention_mask==0) if attention_mask is not None else None
        for layer in self.layers: hidden_states = layer(hidden_states, attention_mask=key_padding_mask)
        hidden_states = self.final_layernorm(hidden_states);
        logits = self.lm_head(hidden_states)
        return self.CausalLMOutput(logits=logits)

    def save_pretrained(self, save_directory):  # 这个保存函数很有用，我们保留它
        return
        # os.makedirs(save_directory, exist_ok=True); config_path = os.path.join(save_directory, "config.json")
        # self.config.save_json(config_path); model_path = os.path.join(save_directory, "pytorch_model.bin")
        # torch.save(self.state_dict(), model_path); print(f"✅ 模型已保存至 {save_directory}")


@dataclass
class CustomDataCollator:  # Collator 依然是必要的
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        ########################## 根本原因修复点 ##########################
        # 错误的方式： labels = [feature.pop("labels") for feature in features]
        # 这会永久性地从数据集中移除 'labels'，导致第二次评估时出错。

        # 正确的方式：
        # 1. 安全地提取 'labels'，不改变原始的 features 列表
        labels = [feature["labels"] for feature in features]

        # 2. 创建一份不含 'labels' 的副本，用于 padding
        features_for_padding = [{k: v for k, v in feature.items() if k!='labels'} for feature in features]

        # 使用副本进行 padding
        batch = self.tokenizer.pad(features_for_padding, padding=True, return_tensors="pt")

        # 将提取出的 labels 堆叠并添加到批次中
        batch["labels"] = torch.stack(labels)
        return batch


class LightsOutDataset(Dataset):  # Dataset 定义也保持不变
    def __init__(self, path, tokenizer):
        self.samples = [];
        self.tokenizer = tokenizer
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    input_binary_str = item['input'];
                    output_binary_str = item['output']
                    # input_ids = [int(char) for char in input_binary_str]
                    input_ids = [ord(char) for char in input_binary_str]
                    # if self.tokenizer.eos_token_id is not None:
                    input_ids.append(319)
                    attention_mask = [1] * len(input_ids)
                    solution = [int(i) for i in output_binary_str]
                    # solution = solution[0]*2+solution[1]
                    # solution = solution[0]*8+solution[1]*4+solution[2]*2+solution[3]
                    # print(solution)
                    label_tensor = torch.tensor(solution, dtype=torch.float32)

                    self.samples.append(
                        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_tensor})
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告: 第 {line_idx + 1} 行处理失败，错误: {e}，跳过。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- 手动定义超参数 (替代 TrainingArguments) ---
hyperparams = {
    "num_epochs": 1800,
    "train_batch_size": 256,
    "eval_batch_size": 256,
    "learning_rate": 5e-4,
    "log_interval": 20,  # 每20步打印一次日志
    "eval_interval": 1000,  # 每1000步评估一次
    "save_interval": 1000,  # 每1000步保存一次
    "output_dir": "./checkpoints_manual",
}
os.makedirs(hyperparams["output_dir"], exist_ok=True)

# --- 初始化模型、Tokenizer、数据集 (与之前相同) ---
base_model_path_for_tokenizer = "qwen2_0.5b"
num_labels = 33
tokenizer = AutoTokenizer.from_pretrained(base_model_path_for_tokenizer, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_config = TinyTransformerConfig(vocab_size=320)  # tokenizer.vocab_size)
model = TinyTransformerForCausalLM(model_config)
model.lm_head = torch.nn.Linear(model.config.hidden_size, num_labels, bias=False)
model.to(device)  # 将模型移动到GPU

dataset = LightsOutDataset("adder_8bit_base16_sem_shuffled_train.jsonl", tokenizer)
# eval_dataset = LightsOutDataset("arc_agi_val.jsonl", tokenizer)
# ca_rule110_layer30_30.jsonl
# edit_distance_path_unique_final.jsonl
# navball_dataset.jsonl
# arc_agi_train.jsonl;arc_agi_val.jsonl
# edit_distance_15bit_train.jsonl
# modulo_20bit_by_2bit_train.jsonl

train_size = int(0.99 * len(dataset))
eval_size = len(dataset) - train_size
if eval_size==0 and train_size > 0: train_size -= 1; eval_size = 1
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

data_collator = CustomDataCollator(tokenizer=tokenizer)

# --- 创建 DataLoader (替代 Trainer 的数据处理) ---
train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["train_batch_size"], collate_fn=data_collator,
    shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=hyperparams["eval_batch_size"], collate_fn=data_collator)

# --- 初始化优化器、损失函数、混合精度和学习率调度器 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"])
loss_fn = BCEWithLogitsLoss()
# loss_fn = CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()  # 用于FP16混合精度

# 计算总训练步数以设置学习率调度器
num_training_steps = hyperparams["num_epochs"] * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


# --- 评估函数 (清晰地分离评估逻辑) ---
def evaluate(model, dataloader, loss_fn, device):
    model.eval()  # 进入评估模式
    total_loss = 0
    with torch.no_grad():  # 不计算梯度，节省资源
        for batch in dataloader:
            # 将数据移动到GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits

            # 计算损失
            sequence_lengths = torch.eq(attention_mask, 1).sum(-1) - 1
            last_token_logits = logits[torch.arange(logits.shape[0]), sequence_lengths]
            loss = loss_fn(last_token_logits, labels)
            total_loss += loss.item()

    model.train()  # 切换回训练模式
    return total_loss / len(dataloader)


# --- 核心训练循环 ---
global_step = 0
print("🚀 开始从零训练 (手动循环)...")

for epoch in range(hyperparams["num_epochs"]):
    model.train()  # 确保模型处于训练模式

    # 使用tqdm创建进度条
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{hyperparams['num_epochs']}")

    for batch in progress_bar:
        optimizer.zero_grad()  # 清空梯度

        # 将数据移动到GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # FP16混合精度前向传播
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            sequence_lengths = torch.eq(attention_mask, 1).sum(-1) - 1
            last_token_logits = logits[torch.arange(logits.shape[0]), sequence_lengths]
            loss = loss_fn(last_token_logits, labels)

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        # --- 日志、评估、保存 ---
        if global_step % hyperparams["log_interval"]==0:
            # 更新进度条上的损失显示
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        if global_step % hyperparams["eval_interval"]==0:
            eval_loss = evaluate(model, eval_dataloader, loss_fn, device)
            print(f"\n--- Step {global_step} ---")
            print(f"  Train Loss: {loss.item():.6f}")
            print(f"  Eval Loss:  {eval_loss:.6f}")
            print("-----------------------")

        if global_step % hyperparams["save_interval"]==0:
            save_path = os.path.join(hyperparams["output_dir"], f"step_{global_step}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)  # 同时保存tokenizer

# --- 训练结束 ---
print("✅ 训练完成，保存最终模型...")
final_save_path = "merged_tiny_transformer_manual"
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"🎉 最终模型和Tokenizer已保存至 {final_save_path}")