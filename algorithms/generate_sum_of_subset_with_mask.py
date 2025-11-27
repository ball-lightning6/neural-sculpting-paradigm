"""
子集和问题数据集生成器 - 双模式版本
-----------------------------------
本脚本提供两种问题格式，用于对比研究任务结构对模型可学习性的影响：

模式1 - 逆向问题（组合优化）:
  输入: numbers + target
  输出: mask (哪个子集的和等于target)
  难度: 极高（需要搜索、决策、组合推理）
  用途: 测试模型学习NP-hard组合优化问题的能力

模式2 - 顺向问题（纯计算）:
  输入: numbers + mask (已经告诉你选哪个子集)
  输出: sum (这些数的和是多少)
  难度: 低（只需要加法运算）
  用途: 验证任务结构本身（而非模型容量）对可学习性的关键影响
"""

import json
import random
from itertools import combinations

# ========== 配置区域 ==========
# 选择生成模式: "reverse" = 逆向问题, "forward" = 顺向问题
MODE = "forward"  # 当前使用的模式

# ========== 模式1: 逆向问题 (组合优化) ==========
def generate_subset_sum_reverse(n_items=4, value_range=(0, 15), min_target=10, max_target=63, min_solution_len=2):
    """
    逆向问题：给定数字和目标值，输出应该选哪个子集
    这是一个组合优化问题，需要搜索和决策，学习难度极高
    """
    while True:
        numbers = random.sample(range(value_range[0], value_range[1] + 1), n_items)

        # 枚举所有子集求和
        subset_by_sum = {}
        for r in range(1, n_items + 1):
            for idx in combinations(range(n_items), r):
                s = sum(numbers[i] for i in idx)
                if min_target <= s <= max_target:
                    if s not in subset_by_sum:
                        subset_by_sum[s] = []
                    subset_by_sum[s].append(idx)

        for target, idx_list in subset_by_sum.items():
            # 找出最短解
            min_len = min(len(idx) for idx in idx_list)
            if min_len < min_solution_len:
                continue

            min_sets = [idx for idx in idx_list if len(idx) == min_len]

            # 要求唯一最短解
            if len(min_sets) == 1:
                best_idx = min_sets[0]
                output_mask = [1 if i in best_idx else 0 for i in range(n_items)]
                return {
                    "input": ''.join([f'{n:04b}' for n in numbers]) + f'{target:06b}',
                    "output": ''.join(map(str, output_mask)),
                    "mode": "reverse",
                    "difficulty": "high"
                }


# ========== 模式2: 顺向问题 (纯计算) ==========
def generate_subset_sum_forward(n_items=6, value_range=(0, 15), min_target=10, max_target=63):
    """
    顺向问题：给定数字和子集掩码，计算这些数的和
    这是一个纯计算问题，只需要加法，学习难度极低
    """
    while True:
        numbers = random.sample(range(value_range[0], value_range[1] + 1), n_items)

        # 枚举所有子集求和
        subset_by_sum = {}
        for r in range(1, n_items + 1):
            for idx in combinations(range(n_items), r):
                s = sum(numbers[i] for i in idx)
                if min_target <= s <= max_target:
                    if s not in subset_by_sum:
                        subset_by_sum[s] = []
                    subset_by_sum[s].append(idx)

        # 只保留 target 只有一个合法解的情况
        for target, idx_list in subset_by_sum.items():
            if len(idx_list) == 1:
                idx = idx_list[0]
                output_mask = [1 if i in idx else 0 for i in range(n_items)]
                return {
                    "input": ''.join([f'{n:04b}' for n in numbers]) + ''.join(map(str, output_mask)),
                    "output": f'{target:06b}',
                    "mode": "forward",
                    "difficulty": "low"
                }


# ========== 主生成函数 ==========
def generate_dataset(num_samples=300000, mode="forward", filename=None):
    """
    生成数据集
    mode: "forward" 或 "reverse"
    """
    if filename is None:
        filename = f"subset_sum_{mode}_dataset.jsonl"

    print(f"生成模式: {mode} ({'顺向问题' if mode == 'forward' else '逆向问题'})")
    print(f"目标样本数: {num_samples}")
    print(f"输出文件: {filename}")

    with open(filename, "w") as f:
        for i in range(num_samples):
            if mode == "forward":
                example = generate_subset_sum_forward()
            else:
                example = generate_subset_sum_reverse()

            f.write(json.dumps(example) + "\n")

            if (i + 1) % 10000 == 0:
                print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"数据集创建成功！文件已保存至: {filename}")
    print(f"模式: {mode}")
    print(f"样本数: {num_samples}")


# ========== 执行生成 ==========
if __name__ == "__main__":
    # 配置参数
    MODE = "forward"  # 可选择 "forward" 或 "reverse"
    NUM_SAMPLES = 300000
    OUTPUT_FILE = None  # 自动根据模式命名

    generate_dataset(num_samples=NUM_SAMPLES, mode=MODE, filename=OUTPUT_FILE)