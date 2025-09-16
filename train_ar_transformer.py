import os
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any


# --- 1. 配置中心 (保持不变) ---
class Config:
    DATASET_PATH = "ca_rule110_n30_l2_text_format.jsonl"
    LOCAL_TOKENIZER_PATH = './qwen2_0.5b'
    CONTEXT_LENGTH = 512  # 确保足够长
    HIDDEN_SIZE = 384
    NUM_LAYERS = 6
    NUM_HEADS = 8
    OUTPUT_DIR = "./ca_autoregressive_fair_model"
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-5
    VAL_SPLIT = 0.01


# --- 2. 自定义Data Collator (保持不变) ---
@dataclass
class CausalInferenceDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ... (此处代码与您提供的完全相同，为简洁省略)
        texts = [ex['text'] for ex in examples]
        prompts = [text.split(" -> ")[0] + " -> " for text in texts]
        answers = [text.split(" -> ")[1] for text in texts]
        prompt_tokenized = self.tokenizer(prompts, padding=False, truncation=False)
        answer_tokenized = self.tokenizer(answers, padding=False, truncation=False, add_special_tokens=False)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        pad_token_id = self.tokenizer.pad_token_id
        max_length = 0
        for i in range(len(prompts)):
            length = len(prompt_tokenized['input_ids'][i]) + len(answer_tokenized['input_ids'][i])
            if length > max_length: max_length = length
        max_length = min(max_length, self.tokenizer.model_max_length or 1024)
        for i in range(len(prompts)):
            prompt_ids, answer_ids = prompt_tokenized['input_ids'][i], answer_tokenized['input_ids'][i]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            padding_len = max_length - len(input_ids)
            input_ids += [pad_token_id] * padding_len
            labels += [-100] * padding_len
            attention_mask = [1] * (len(prompt_ids) + len(answer_ids)) + [0] * padding_len
            batch_input_ids.append(torch.tensor(input_ids));
            batch_attention_mask.append(torch.tensor(attention_mask));
            batch_labels.append(torch.tensor(labels))
        return {"input_ids": torch.stack(batch_input_ids), "attention_mask": torch.stack(batch_attention_mask),
                "labels": torch.stack(batch_labels)}


# --- 3. 数据集 (保持不变) ---
class TextDataset(Dataset):
    def __init__(self, metadata_list):
        self.texts = [item['text'] for item in metadata_list]

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx): return {"text": self.texts[idx]}


# --- 4. 核心修改: 自定义 Trainer 子类 ---
class GenerativeEvalTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # 调用父类的evaluate方法，获取常规的eval_loss等指标
        eval_output = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # # --- "侦察兵"功能区 ---
        # print("\n--- 🕵️‍♂️ Running Generative Evaluation... ---")

        # # 从验证集中随机选择一个样本
        # if eval_dataset is None:
        #     eval_dataset = self.eval_dataset
        # random_idx = random.randint(0, len(eval_dataset) - 1)
        # test_example = eval_dataset[random_idx]['text']

        # prompt_text = test_example.split(" -> ")[0] + " -> "
        # ground_truth = test_example

        # print(f"Input Prompt: '{prompt_text}'")
        # print(f"Ground Truth: '{ground_truth}'")

        # # 使用当前模型进行生成
        # inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.args.device)

        # output_sequences = self.model.generate(
        #     **inputs,
        #     max_new_tokens=self.model.config.n_positions - inputs['input_ids'].shape[1], # 动态计算最大长度
        #     num_return_sequences=1,
        #     do_sample=False,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id
        # )

        # generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        # print(f"Model Output: '{generated_text}'")
        # print("--- 🕵️‍♂️ Generative Evaluation Finished ---")

        return eval_output


# --- 5. 主执行流程 ---
if __name__ == "__main__":
    config = Config()

    # --- 准备 Tokenizer & 数据集 (与之前相同) ---
    tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_TOKENIZER_PATH, trust_remote_code=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['Evolve', 'this', ':', '->']})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    VOCAB_SIZE = len(tokenizer)
    print(f"Tokenizer准备完毕，词汇表大小: {VOCAB_SIZE}")
    with open(config.DATASET_PATH, 'r') as f:
        full_metadata = [json.loads(line) for line in f]
    train_size = int((1 - config.VAL_SPLIT) * len(full_metadata))
    val_size = len(full_metadata) - train_size
    train_meta, val_meta = random_split(full_metadata, [train_size, val_size])
    train_dataset = TextDataset(list(train_meta))
    eval_dataset = TextDataset(list(val_meta))
    data_collator = CausalInferenceDataCollator(tokenizer=tokenizer)

    # --- 准备模型 (与之前相同) ---
    model_config = GPT2Config(
        vocab_size=VOCAB_SIZE, n_positions=config.CONTEXT_LENGTH, n_embd=config.HIDDEN_SIZE,
        n_layer=config.NUM_LAYERS, n_head=config.NUM_HEADS, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2LMHeadModel(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建成功! 总参数量: {num_params / 1_000_000:.2f} M")

    # --- 准备训练参数 (与之前相同) ---
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR, overwrite_output_dir=True, num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE, per_device_eval_batch_size=config.BATCH_SIZE,
        evaluation_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=1500,
        learning_rate=config.LEARNING_RATE, logging_steps=100, remove_unused_columns=False,
        load_best_model_at_end=True,  # 开启这个可以在最后加载最好的模型
        metric_for_best_model="eval_loss",  # 以eval_loss为标准
        report_to="none", fp16=torch.cuda.is_available()
    )

    # --- 核心：实例化我们自定义的 Trainer ---
    trainer = GenerativeEvalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer  # 将tokenizer传递给自定义Trainer
    )

    # --- 开始训练 ---
    print("\n--- 🚀 开始自回归训练 (带实时生成侦察) ---")
    trainer.train()

    # --- 训练结束 ---
    print("\n--- ✅ 训练完成! ---")
    # 因为开启了load_best_model_at_end，现在的trainer.model就是最好的模型
    print("使用性能最好的模型进行最终测试...")

    # 从验证集随机选几个样本进行最终展示
    for i in range(5):
        test_example = eval_dataset[random.randint(0, len(eval_dataset) - 1)]['text']
        prompt_text = test_example.split(" -> ")[0] + " -> "
        ground_truth = test_example

        print("\n" + "=" * 20 + f" 测试样本 {i + 1} " + "=" * 20)
        print(f"输入 Prompt: '{prompt_text}'")

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        output_sequences = model.generate(
            **inputs, max_new_tokens=config.CONTEXT_LENGTH, num_return_sequences=1,
            do_sample=False, pad_token_id=tokenizer.eos_token_idSSS
        )
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        print(f"模型生成结果: '{generated_text}'")
        print(f"真实答案: '{ground_truth}'")
        # 检查是否完全匹配
        if generated_text == ground_truth:
            print("✅ 结果完全匹配！")
        else:
            print("❌ 结果不匹配。")