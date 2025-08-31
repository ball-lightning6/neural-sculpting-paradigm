import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_COLUMNS_N = 10
BITS_PER_HEIGHT = 3

DATASET_SIZE = 300000

TRAIN_FILE = f'trapping_rain_water_decoupled_n{NUM_COLUMNS_N}_b{BITS_PER_HEIGHT}_train.jsonl'
EVAL_FILE = f'trapping_rain_water_decoupled_n{NUM_COLUMNS_N}_b{BITS_PER_HEIGHT}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 (全新输出格式) ---
# ==============================================================================
INPUT_LEN = NUM_COLUMNS_N * BITS_PER_HEIGHT

# --- 关键改动：输出是每个格子的雨水量 ---
# 每个格子最多能接的雨水量，是MAX_HEIGHT
MAX_HEIGHT = 2 ** BITS_PER_HEIGHT - 1
# 表示这个最大值需要多少个bit (与输入位数相同)
BITS_PER_OUTPUT_CELL = BITS_PER_HEIGHT
# 总输出比特数
TOTAL_OUTPUT_BITS = NUM_COLUMNS_N * BITS_PER_OUTPUT_CELL

print("=" * 70)
print(f"     “接雨水”问题 - 数据集生成器 (解耦版)")
print("=" * 70)
print(f"柱子数量: {NUM_COLUMNS_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'")
print(f"输出格式: {TOTAL_OUTPUT_BITS}个多标签二分类")
print(f"  (共有{NUM_COLUMNS_N}个格子, 每个格子的雨水量用{BITS_PER_OUTPUT_CELL} bits表示)")
print("=" * 70)

all_set=set()
# ==============================================================================
# --- 3. 核心逻辑：求解器 (高效的 O(N) 双指针算法) ---
# ==============================================================================
def generate_heights(num_columns, max_height):
    """随机生成一个高度图数组"""
    while True:
        heights = [random.randint(0, max_height) for _ in range(num_columns)]
        if tuple(heights) not in all_set:
            all_set.add(tuple(heights))
            return heights
    # return [random.randint(0, max_height) for _ in range(num_columns)]

def solve_trapping_rain_water_per_cell(height):
    """
    计算每个格子能接的雨水量。
    返回一个长度为N的列表。
    """
    n = len(height)
    if n==0:
        return []

    water_per_cell = [0] * n

    # 从左到右计算left_max
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # 从右到左计算right_max
    right_max = [0] * n
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # 计算每个格子的雨水
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > height[i]:
            water_per_cell[i] = water_level - height[i]

    return water_per_cell


def process_sample(num_columns, bits_per_height, bits_per_output_cell):
    """生成一个完整的 (输入, 输出) 数据对。"""
    max_height = 2 ** bits_per_height - 1
    heights = generate_heights(num_columns, max_height)

    # 1. 编码输入
    input_str_list = [format(h, f'0{bits_per_height}b') for h in heights]
    input_str = "".join(input_str_list)

    # 2. 计算每个格子的雨水量
    water_per_cell = solve_trapping_rain_water_per_cell(heights)

    # 3. 编码输出
    output_str_list = [format(w, f'0{bits_per_output_cell}b') for w in water_per_cell]
    output_str = "".join(output_str_list)
    output_multilabel = [int(bit) for bit in output_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_columns, bits_per_height, bits_per_output_cell):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        records.append(process_sample(num_columns, bits_per_height, bits_per_output_cell))
        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略) ...
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
    generate_datasets(DATASET_SIZE, NUM_COLUMNS_N, BITS_PER_HEIGHT, BITS_PER_OUTPUT_CELL)