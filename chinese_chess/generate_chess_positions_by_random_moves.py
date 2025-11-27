#!/usr/bin/env python3
"""
中国象棋随机走法局面生成器
通过模拟随机对局生成合法象棋局面
"""

import json
import os
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from cchess import Board

# ==============================================================================
# 配置参数
# ==============================================================================
NUM_SAMPLES = 1000  # 默认生成样本数
NUM_PROCESSES = cpu_count()
MAX_STEPS = 100     # 每局最大步数
MAX_CAPTURE = 5     # 每局最大吃子数
OUTPUT_DIR = "chess_random_moves_dataset"

# ==============================================================================
# 核心生成函数
# ==============================================================================
def generate_random_fen_and_moves(max_steps=MAX_STEPS, max_capture=MAX_CAPTURE):
    """
    通过随机走法生成象棋局面
    返回: (fen, legal_moves) 元组
    """
    capture_count = 0
    board = Board()  # 标准起始局面
    
    for step in range(max_steps):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        move = random.choice(legal_moves)
        
        # 吃子计数和控制
        if board.is_capture(move):
            capture_count += 1
            if capture_count > max_capture:
                # 重新选择非吃子走法
                non_capture_moves = [m for m in legal_moves if not board.is_capture(m)]
                if non_capture_moves:
                    move = random.choice(non_capture_moves)
                else:
                    break
        
        board.push(move)
    
    # 获取最终局面的FEN和合法走法
    final_fen = board.fen()
    legal_uci_moves = [m.uci() for m in board.legal_moves]
    
    return final_fen, legal_uci_moves

# ==============================================================================
# 单样本生成任务（用于多进程）
# ==============================================================================
def generate_sample_task(args):
    """多进程任务函数"""
    idx, max_steps, max_capture = args
    try:
        fen, legal_moves = generate_random_fen_and_moves(max_steps, max_capture)
        
        # 构建标准输出格式
        sample = {
            "input": fen,  # FEN格式局面
            "output": {
                "fen": fen,
                "legal_moves": legal_moves,
                "num_legal_moves": len(legal_moves)
            },
            "metadata": {
                "max_steps": max_steps,
                "max_capture": max_capture,
                "generation_method": "random_moves"
            }
        }
        
        return sample
    except Exception as e:
        print(f"生成样本 {idx} 失败: {e}")
        return None

# ==============================================================================
# 主生成函数
# ==============================================================================
def generate_dataset(num_samples=NUM_SAMPLES, max_steps=MAX_STEPS, max_capture=MAX_CAPTURE, output_dir=OUTPUT_DIR):
    """
    生成完整的数据集
    """
    print("=" * 60)
    print("中国象棋随机走法局面数据集生成器")
    print(f"参数: num_samples={num_samples}, max_steps={max_steps}, max_capture={max_capture}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备多进程任务
    tasks = [(i, max_steps, max_capture) for i in range(num_samples)]
    
    print(f"开始生成 {num_samples} 个样本，使用 {NUM_PROCESSES} 个进程...")
    
    # 多进程生成
    dataset = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_task, tasks), 
                          total=num_samples, desc="生成进度"):
            if result is not None:
                dataset.append(result)
    
    print(f"成功生成 {len(dataset)} 个样本")
    
    # 保存训练集 (90%)
    train_size = int(len(dataset) * 0.9)
    train_data = dataset[:train_size]
    
    # 保存评估集 (10%)
    eval_data = dataset[train_size:]
    
    # 写入文件
    train_file = os.path.join(output_dir, f"chess_random_moves_train.jsonl")
    eval_file = os.path.join(output_dir, f"chess_random_moves_eval.jsonl")
    
    print(f"写入训练集: {len(train_data)} 条记录 -> {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"写入评估集: {len(eval_data)} 条记录 -> {eval_file}")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for record in eval_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 生成元数据
    metadata = {
        "dataset_name": "中国象棋随机走法局面数据集",
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "max_steps": max_steps,
        "max_capture": max_capture,
        "generation_method": "random_moves",
        "output_format": "jsonl",
        "train_file": train_file,
        "eval_file": eval_file
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("✅ 数据集生成完成!")
    print(f"总计: {len(dataset)} 个样本")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"评估集: {len(eval_data)} 个样本")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

# ==============================================================================
# 主执行部分
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中国象棋随机走法局面数据集生成器")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, 
                       help=f"生成样本数量 (默认: {NUM_SAMPLES})")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                       help=f"每局最大步数 (默认: {MAX_STEPS})")
    parser.add_argument("--max_capture", type=int, default=MAX_CAPTURE,
                       help=f"每局最大吃子数 (默认: {MAX_CAPTURE})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                       help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES,
                       help=f"进程数 (默认: {NUM_PROCESSES})")
    
    args = parser.parse_args()
    
    # 设置进程数
    if args.num_processes > 0:
        NUM_PROCESSES = args.num_processes
    
    # 生成数据集
    generate_dataset(
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        max_capture=args.max_capture,
        output_dir=args.output_dir
    )