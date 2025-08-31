import json
import random
import math
import numpy as np

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
MATRIX_M = 5
MATRIX_N = 6

DATASET_SIZE = 300000

TRAIN_FILE = f'matrix_flip_strategy_{MATRIX_M}x{MATRIX_N}_train.jsonl'
EVAL_FILE = f'matrix_flip_strategy_{MATRIX_M}x{MATRIX_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = MATRIX_M * MATRIX_N
# --- 关键改动：输出是M+N个翻转标志 ---
OUTPUT_BITS = MATRIX_M + MATRIX_N

print("=" * 70)
print(f"     翻转矩阵策略 - 数据集生成器 (解耦版)")
print("=" * 70)
print(f"矩阵大小: {MATRIX_M}x{MATRIX_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (行翻转标志 + 列翻转标志)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (找到最优翻转策略) ---
# ==============================================================================
all_set=set()
def generate_matrix(m, n):
    """随机生成一个 M x N 的 0/1 矩阵"""
    while True:
        matrix=[[random.choice([0, 1]) for _ in range(n)] for _ in range(m)]
        s=''.join(''.join(str(x)) for x in matrix)

        if s not in all_set:
            all_set.add(s)
            return matrix


def solve_optimal_flips(grid):
    """
    找到能得到最大得分的最优翻转策略。
    返回一个长度为 M+N 的0/1列表。
    """
    m, n = len(grid), len(grid[0])

    row_flips = [0] * m
    col_flips = [0] * n

    # 步骤1：确定行翻转策略 (贪心：让第一列的1最多)
    # 我们需要一个临时grid来模拟翻转
    temp_grid = [row[:] for row in grid]

    for i in range(m):
        if temp_grid[i][0]==0:
            row_flips[i] = 1
            # 翻转这一行
            for j in range(n):
                temp_grid[i][j] = 1 - temp_grid[i][j]

    # 步骤2：确定列翻转策略 (贪心：让每一列的1比0多)
    for j in range(n):
        count_ones = sum(temp_grid[i][j] for i in range(m))
        count_zeros = m - count_ones
        if count_zeros > count_ones:
            col_flips[j] = 1

    return row_flips + col_flips


def process_sample(m, n, output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入矩阵
    grid = generate_matrix(m, n)
    input_str = "".join(str(cell) for row in grid for cell in row)

    # 2. 计算最优翻转策略
    optimal_flips_vector = solve_optimal_flips(grid)

    return {"input": input_str, "output": optimal_flips_vector}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, m, n, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        records.append(process_sample(m, n, output_bits))
        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")
    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略)
    random.shuffle(records)
    train_size = int(len(records) * 1)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in data: f.write(json.dumps(record) + '\n')
        print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        with open(path[1], 'w') as f:
            for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, MATRIX_M, MATRIX_N, OUTPUT_BITS)