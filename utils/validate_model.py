# validate_model.py
# 独立模型验证脚本 - 用于验证训练好的模型在大型验证集上的性能

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import logging
import time
import os
import argparse

class ValidationConfig:
    """验证配置类"""
    def __init__(self, model_path, dataset_path, input_bits, output_bits, hidden_size=4096, num_layers=4):
        self.MODEL_PATH = model_path
        self.DATASET_PATH = dataset_path
        self.INPUT_BITS = input_bits
        self.OUTPUT_BITS = output_bits
        self.HIDDEN_SIZE = hidden_size
        self.NUM_HIDDEN_LAYERS = num_layers
        self.BATCH_SIZE = 4096
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.LOG_FILE = f"validation_log_{os.path.basename(model_path).replace('.pth', '')}.log"

class SimpleDataset(Dataset):
    """简单数据集类"""
    def __init__(self, metadata_list):
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        row = self.metadata_list[idx]
        input_tensor = torch.tensor([int(bit) for bit in row['input']], dtype=torch.float32)
        output_tensor = torch.tensor(row['output'], dtype=torch.float32)
        return input_tensor, output_tensor

class MLP(nn.Module):
    """标准MLP模型"""
    def __init__(self, config):
        super().__init__()
        layers = [
            nn.Linear(config.INPUT_BITS, config.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_SIZE),
            nn.Dropout(0.1)
        ]

        for _ in range(config.NUM_HIDDEN_LAYERS):
            layers.extend([
                nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
                nn.GELU(),
                nn.LayerNorm(config.HIDDEN_SIZE),
                nn.Dropout(0.1)
            ])

        layers.append(nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def setup_logger(log_file):
    """设置日志系统"""
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 控制台输出
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 文件输出
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def validate_model(model, dataloader, device, logger):
    """验证模型性能"""
    model.eval()
    total_samples = 0
    total_exact_matches = 0
    total_correct_bits = 0
    total_bits = 0

    start_time = time.time()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证进度")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 计算精确匹配
            preds = (torch.sigmoid(outputs) > 0.5).float()
            matches = torch.all(preds == labels, dim=1).sum().item()
            total_exact_matches += matches
            total_samples += inputs.size(0)

            # 计算位准确率
            total_correct_bits += (preds == labels).sum().item()
            total_bits += labels.numel()

            # 更新进度条
            current_exact_acc = 100 * total_exact_matches / total_samples
            current_bit_acc = 100 * total_correct_bits / total_bits
            progress_bar.set_postfix(
                Exact_Match=f"{current_exact_acc:.6f}%",
                Bit_Acc=f"{current_bit_acc:.6f}%"
            )

    eval_time = time.time() - start_time

    # 计算最终指标
    final_exact_acc = 100 * total_exact_matches / total_samples
    final_bit_acc = 100 * total_correct_bits / total_bits
    total_errors = total_samples - total_exact_matches

    return {
        "exact_match_rate": final_exact_acc,
        "bit_accuracy": final_bit_acc,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "eval_time": eval_time
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="验证训练好的模型性能")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (.pth)")
    parser.add_argument("--dataset", type=str, required=True, help="验证数据集路径 (.jsonl)")
    parser.add_argument("--input-bits", type=int, required=True, help="输入位数")
    parser.add_argument("--output-bits", type=int, required=True, help="输出位数")
    parser.add_argument("--hidden-size", type=int, default=4096, help="隐藏层大小")
    parser.add_argument("--num-layers", type=int, default=4, help="隐藏层数量")
    parser.add_argument("--batch-size", type=int, default=4096, help="批次大小")

    args = parser.parse_args()

    # 创建配置
    config = ValidationConfig(
        model_path=args.model,
        dataset_path=args.dataset,
        input_bits=args.input_bits,
        output_bits=args.output_bits,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )

    # 设置日志
    logger = setup_logger(config.LOG_FILE)
    logger.info("="*60)
    logger.info("           模型性能验证工具")
    logger.info("="*60)
    logger.info(f"模型路径: {config.MODEL_PATH}")
    logger.info(f"数据集: {config.DATASET_PATH}")
    logger.info(f"输入维度: {config.INPUT_BITS}")
    logger.info(f"输出维度: {config.OUTPUT_BITS}")
    logger.info(f"隐藏层: {config.HIDDEN_SIZE} x {config.NUM_HIDDEN_LAYERS}")
    logger.info(f"设备: {config.DEVICE}")
    logger.info("="*60)

    # 加载数据集
    logger.info(f"正在加载验证数据集: {config.DATASET_PATH}")
    try:
        with open(config.DATASET_PATH, 'r') as f:
            metadata = [json.loads(line) for line in tqdm(f, desc="读取数据")]
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return

    logger.info(f"成功加载 {len(metadata)} 条验证样本")

    # 创建数据加载器
    dataset = SimpleDataset(metadata)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型
    logger.info("正在初始化模型...")
    model = MLP(config).to(config.DEVICE)

    # 加载模型权重
    logger.info(f"正在加载模型权重: {config.MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except Exception as e:
        logger.error(f"加载模型权重失败: {e}")
        return

    # 开始验证
    logger.info("\n开始验证模型性能...")
    results = validate_model(model, dataloader, config.DEVICE, logger)

    # 输出结果
    logger.info("\n" + "="*60)
    logger.info("              验证结果报告")
    logger.info("="*60)
    logger.info(f"总样本数: {results['total_samples']:,}")
    logger.info(f"错误样本数: {results['total_errors']:,}")
    logger.info(f"精确匹配率: {results['exact_match_rate']:.12f}%")
    logger.info(f"位准确率: {results['bit_accuracy']:.12f}%")
    logger.info(f"验证耗时: {results['eval_time']:.2f} 秒")
    logger.info("="*60)

    # 保存结果到文件
    result_file = config.LOG_FILE.replace('.log', '_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n详细结果已保存至: {result_file}")

if __name__ == "__main__":
    main()