import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的高度图的柱子数量
NUM_COLUMNS_N = 10
# 每个柱子的高度，用多少个bit表示 (0 <= height <= 10^5)
# 10^5 约等于 2^16.6，所以需要17位
BITS_PER_HEIGHT = 3

DATASET_SIZE = 300000

TRAIN_FILE = f'trapping_rain_water_n{NUM_COLUMNS_N}_b{BITS_PER_HEIGHT}_train.jsonl'
EVAL_FILE = f'trapping_rain_water_n{NUM_COLUMNS_N}_b{BITS_PER_HEIGHT}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = NUM_COLUMNS_N * BITS_PER_HEIGHT

# 输出是雨水总量
# 计算最大可能的雨水量，以确定输出位数
MAX_HEIGHT = 2 ** BITS_PER_HEIGHT - 1
# 最极端情况，两边是最高，中间是0
MAX_WATER = (NUM_COLUMNS_N - 2) * MAX_HEIGHT
OUTPUT_BITS = math.ceil(math.log2(MAX_WATER + 1))

print("=" * 70)
print(f"     “接雨水”问题 - 数据集生成器")
print("=" * 70)
print(f"柱子数量: {NUM_COLUMNS_N}")
print(f"每个柱子高度用 {BITS_PER_HEIGHT} bits表示")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表雨水总量)")
print(f"最大可能雨水量: {MAX_WATER}")
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


def solve_trapping_rain_water(height):
    """
    使用高效的 O(N) 双指针算法计算雨水总量。
    """
    if not height:
        return 0

    n = len(height)
    left, right = 0, n - 1
    left_max, right_max = 0, 0
    total_water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                total_water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                total_water += right_max - height[right]
            right -= 1

    return total_water


def process_sample(num_columns, bits_per_height, output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    max_height = 2 ** bits_per_height - 1
    heights = generate_heights(num_columns, max_height)

    # 1. 编码输入
    input_str_list = [format(h, f'0{bits_per_height}b') for h in heights]
    input_str = "".join(input_str_list)

    # 2. 计算雨水总量
    total_water = solve_trapping_rain_water(heights)

    # 3. 编码输出
    output_binary_str = format(total_water, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_columns, bits_per_height, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(num_columns, bits_per_height, output_bits))

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
    generate_datasets(DATASET_SIZE, NUM_COLUMNS_N, BITS_PER_HEIGHT, OUTPUT_BITS)