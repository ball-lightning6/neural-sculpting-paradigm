# generate_rain_water_summation.py

import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 这些参数需要和你第一步的脚本保持一致，以确保数据可以对接
NUM_COLUMNS_N = 10
BITS_PER_CELL = 3

DATASET_SIZE = 500000

TRAIN_FILE = f'rain_water_summation_n{NUM_COLUMNS_N}_b{BITS_PER_CELL}_train.jsonl'
# EVAL_FILE = f'rain_water_summation_n{NUM_COLUMNS_N}_b{BITS_PER_CELL}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是N个格子的雨水量，每个格子用B个bit表示
INPUT_LEN = NUM_COLUMNS_N * BITS_PER_CELL
MAX_CELL_WATER = 2 ** BITS_PER_CELL - 1

# 输出是总和。我们需要计算总和的最大可能值，来确定需要多少个bit
MAX_TOTAL_WATER = NUM_COLUMNS_N * MAX_CELL_WATER
# 使用 math.ceil(math.log2(...)) 来计算表示这个最大值至少需要多少个bit
# 为了安全和对齐，我们取一个比理论值稍大的2的幂次，或者直接计算
# 如果 MAX_TOTAL_WATER 是 0, log2会出错，所以加个max(1, ...)
TOTAL_OUTPUT_BITS = math.ceil(math.log2(max(1, MAX_TOTAL_WATER + 1))) if MAX_TOTAL_WATER > 0 else 1


print("=" * 70)
print(f"     “雨水加和”问题 - 数据集生成器 (任务B)")
print("=" * 70)
print(f"格子数量: {NUM_COLUMNS_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1' (每格雨水)")
print(f"输出格式: {TOTAL_OUTPUT_BITS}个多标签二分类 (总雨水量)")
print(f"最大单格雨水: {MAX_CELL_WATER}, 最大总雨水量: {MAX_TOTAL_WATER}")
print("=" * 70)

# ==============================================================================
# --- 3. 核心逻辑 ---
# ==============================================================================
def generate_water_levels(num_columns, max_water):
    """随机生成一个每格雨水量的数组"""
    return [random.randint(0, max_water) for _ in range(num_columns)]

def process_sample():
    """生成一个完整的 (输入, 输出) 数据对。"""
    water_per_cell = generate_water_levels(NUM_COLUMNS_N, MAX_CELL_WATER)

    # 1. 编码输入
    input_str_list = [format(w, f'0{BITS_PER_CELL}b') for w in water_per_cell]
    input_str = "".join(input_str_list)

    # 2. 计算总和
    total_water = sum(water_per_cell)

    # 3. 编码输出
    output_str = format(total_water, f'0{TOTAL_OUTPUT_BITS}b')
    output_multilabel = [int(bit) for bit in output_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 (与你的脚本结构保持一致) ---
# ==============================================================================
def generate_datasets():
    print("\n--- 开始生成数据集 ---")

    # 为了避免重复，我们使用set来存储已经生成过的输入
    records_dict = {}
    while len(records_dict) < DATASET_SIZE:
        sample = process_sample()
        if sample['input'] not in records_dict:
            records_dict[sample['input']] = sample['output']
        
        if (len(records_dict)) % 10000 == 0:
            # 避免重复打印
            if len(records_dict) > (len(records_dict) - 10):
                 print(f"已生成 {len(records_dict)} / {DATASET_SIZE} 条不重复数据...")


    records = [{"input": k, "output": v} for k, v in records_dict.items()]
    random.shuffle(records)
    print(f"生成完毕。共 {len(records)} 条数据。")

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}数据到 '{path}'...")
        with open(path, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')

    write_to_file(records, TRAIN_FILE, "训练")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
