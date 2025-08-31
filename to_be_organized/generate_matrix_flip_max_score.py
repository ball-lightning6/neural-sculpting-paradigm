import json
import random
import math
import numpy as np

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 矩阵的行数 M 和列数 N
MATRIX_M = 5
MATRIX_N = 6

DATASET_SIZE = 300000

TRAIN_FILE = f'matrix_score_{MATRIX_M}x{MATRIX_N}_train.jsonl'
EVAL_FILE = f'matrix_score_{MATRIX_M}x{MATRIX_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = MATRIX_M * MATRIX_N

# 输出是最大得分，是一个整数。我们需要计算其最大值来确定输出位数。
# 所有位都为1时，单行最大值为 2^N - 1
# 总得分最大值是 M * (2^N - 1)
MAX_SCORE = MATRIX_M * (2 ** MATRIX_N - 1)
OUTPUT_BITS = math.ceil(math.log2(MAX_SCORE + 1))

print("=" * 70)
print(f"     翻转矩阵后的得分 - 数据集生成器")
print("=" * 70)
print(f"矩阵大小: {MATRIX_M}x{MATRIX_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表最大得分)")
print(f"最大可能得分: {MAX_SCORE}")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (高效的 O(M*N) 贪心算法) ---
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

    # return [[random.choice([0, 1]) for _ in range(n)] for _ in range(m)]


def solve_max_score(grid):
    """
    使用高效的 O(M*N) 贪心算法计算最大得分。
    """
    m, n = len(grid), len(grid[0])

    # 步骤1：行翻转 (贪心策略：确保最高位为1)
    for i in range(m):
        if grid[i][0]==0:
            # 翻转这一行
            for j in range(n):
                grid[i][j] = 1 - grid[i][j]

    # 步骤2：列翻转 (贪心策略：确保每一列的1比0多)
    total_score = 0
    for j in range(n):
        count_ones = sum(grid[i][j] for i in range(m))
        count_zeros = m - count_ones

        # 如果0更多，翻转这一列更有利
        # 权重是 2^(n-1-j)
        weight = 2 ** (n - 1 - j)
        total_score += max(count_ones, count_zeros) * weight

    return total_score


def process_sample(m, n, output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入矩阵
    grid = generate_matrix(m, n)

    # 将输入矩阵转换为扁平化的字符串
    input_str = "".join(str(cell) for row in grid for cell in row)

    # 2. 计算最大得分
    # 需要传递grid的副本，因为solve_max_score会修改它
    max_score = solve_max_score([row[:] for row in grid])

    # 3. 编码输出
    output_binary_str = format(max_score, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


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
    train_size = int(len(records) * 1)#0.9)
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