#!/usr/bin/env python3
"""
Nim游戏（齐肯多夫表示法）数据集生成器
基于斐波那契数列和数位DP的博弈论问题
"""

import json
import os
import random
import math
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================================
# 配置参数
# ==============================================================================
NUM_SAMPLES = 1000  # 默认生成样本数
NUM_PROCESSES = cpu_count()
K_BITS = 1          # k的位数
N_BITS = 30         # N的位数
OUTPUT_BITS = N_BITS + 1  # 输出位数
MAX_ATTEMPTS = 100  # 每个样本的最大尝试次数
OUTPUT_DIR = "nim_game_zeckendorf_dataset"

# ==============================================================================
# 核心算法：基于齐肯多夫表示法的Nim博弈求解器
# ==============================================================================
# 预计算斐波那契数，作为全局常量
FIB = [0] * 95
FIB[1], FIB[2] = 1, 2
for i in range(3, 91):
    FIB[i] = FIB[i - 1] + FIB[i - 2]

# dp数组和v数组的缓存
DP_CACHE = {}
V_CACHE = [False] * 95

def DP_solver(p, bound, lst, k_fib_idx):
    """
    数位DP函数，基于齐肯多夫表示法的Nim博弈求解
    """
    if p < k_fib_idx:
        return 1

    state = (p, bound, lst)
    if not lst and not bound and state in DP_CACHE:
        return DP_CACHE[state]

    res = 0
    # case 1: 第p位取0
    res += DP_solver(p - 1, bound and (not V_CACHE[p]), False, k_fib_idx)

    # case 2: 第p位取1
    if not lst and (not bound or V_CACHE[p]):
        res += DP_solver(p - 1, bound and V_CACHE[p], True, k_fib_idx)

    if not lst and not bound:
        DP_CACHE[state] = res

    return res

def solve_nim_game_from_solution(k, n):
    """
    使用齐肯多夫表示法求解Nim博弈
    """
    # 题解中n是0-indexed的
    n -= 1

    # 将n转换为齐肯多夫表示法
    global V_CACHE
    V_CACHE = [False] * 95
    temp_n = n
    for i in range(90, 0, -1):
        if temp_n >= FIB[i]:
            temp_n -= FIB[i]
            V_CACHE[i] = True

    # 找到第一个大于k的斐波那契数的下标
    k_fib_idx = 0
    for i in range(90, 1, -1):
        if FIB[i] > k:
            k_fib_idx = i

    # 重置DP缓存
    global DP_CACHE
    DP_CACHE = {}

    # 最终答案公式
    return (n + 1) - (DP_solver(90, True, False, k_fib_idx) - 1)

def generate_single_nim_position(k, n):
    """
    生成单个Nim博弈局面
    """
    try:
        # 计算答案
        answer = solve_nim_game_from_solution(k, n) - solve_nim_game_from_solution(k, n-1)
        
        # 编码输入和输出
        k_bin = format(k, f'0{K_BITS}b')
        n_bin = format(n, f'0{N_BITS}b')
        input_str = n_bin  # 简化：只使用n作为输入
        
        # 输出是单个数值（简化版）
        output_value = answer
        
        return {
            "input": input_str,
            "output": [output_value],
            "metadata": {
                "k": k,
                "n": n,
                "answer": answer,
                "generation_method": "nim_game_zeckendorf",
                "mathematical_concept": "zeckendorf_representation"
            }
        }
    except Exception as e:
        print(f"生成失败: {e}")
        return None

# ==============================================================================
# 单样本生成任务（用于多进程）
# ==============================================================================
def generate_sample_task(args):
    """多进程任务函数"""
    idx, max_attempts = args
    try:
        # 随机生成k和N
        max_k = 2**K_BITS - 1
        max_n = 2**N_BITS - 1
        max_n = min(max_n, 10**18)  # 限制大小
        
        for _ in range(max_attempts):
            k = random.randint(1, max_k)
            n = random.randint(k, max_n)
            
            result = generate_single_nim_position(k, n)
            if result is not None:
                return result
        
        return None
    except Exception as e:
        print(f"生成样本 {idx} 失败: {e}")
        return None

# ==============================================================================
# 主生成函数
# ==============================================================================
def generate_dataset(num_samples=NUM_SAMPLES, max_attempts=100, output_dir=OUTPUT_DIR):
    """
    生成完整的数据集
    """
    print("=" * 60)
    print("Nim游戏（齐肯多夫表示法）数据集生成器")
    print(f"参数: num_samples={num_samples}, k_bits={K_BITS}, n_bits={N_BITS}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备多进程任务
    tasks = [(i, max_attempts) for i in range(num_samples)]
    
    print(f"开始生成 {num_samples} 个Nim博弈局面，使用 {NUM_PROCESSES} 个进程...")
    print("正在生成基于齐肯多夫表示法的博弈局面...")
    
    # 多进程生成
    dataset = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_task, tasks), 
                          total=num_samples, desc="生成进度"):
            if result is not None:
                dataset.append(result)
    
    print(f"成功生成 {len(dataset)} 个Nim博弈局面")
    
    # 保存训练集 (90%)
    train_size = int(len(dataset) * 0.9)
    train_data = dataset[:train_size]
    
    # 保存评估集 (10%)
    eval_data = dataset[train_size:]
    
    # 写入文件
    train_file = os.path.join(output_dir, f"nim_game_zeckendorf_train.jsonl")
    eval_file = os.path.join(output_dir, f"nim_game_zeckendorf_eval.jsonl")
    
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
        "dataset_name": "Nim游戏（齐肯多夫表示法）数据集",
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "k_bits": K_BITS,
        "n_bits": N_BITS,
        "output_bits": OUTPUT_BITS,
        "generation_method": "nim_game_zeckendorf",
        "mathematical_concept": "zeckendorf_representation",
        "output_format": "jsonl",
        "train_file": train_file,
        "eval_file": eval_file
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("✅ Nim博弈数据集生成完成!")
    print(f"总计: {len(dataset)} 个博弈局面")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"评估集: {len(eval_data)} 个样本")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

# ==============================================================================
# 主执行部分
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nim游戏（齐肯多夫表示法）数据集生成器")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, 
                       help=f"生成样本数量 (默认: {NUM_SAMPLES})")
    parser.add_argument("--k_bits", type=int, default=K_BITS,
                       help=f"k的位数 (默认: {K_BITS})")
    parser.add_argument("--n_bits", type=int, default=N_BITS,
                       help=f"N的位数 (默认: {N_BITS})")
    parser.add_argument("--max_attempts", type=int, default=100,
                       help=f"每个样本最大尝试次数 (默认: 100)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                       help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES,
                       help=f"进程数 (默认: {NUM_PROCESSES})")
    
    args = parser.parse_args()
    
    # 更新全局参数
    global K_BITS, N_BITS, OUTPUT_BITS, NUM_PROCESSES
    K_BITS = args.k_bits
    N_BITS = args.n_bits
    OUTPUT_BITS = N_BITS + 1
    NUM_PROCESSES = args.num_processes
    
    # 生成数据集
    generate_dataset(
        num_samples=args.num_samples,
        max_attempts=args.max_attempts,
        output_dir=args.output_dir
    )