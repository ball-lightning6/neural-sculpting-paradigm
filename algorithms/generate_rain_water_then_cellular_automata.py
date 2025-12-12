# generate_rain_water_then_cellular_automata.py

import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# --- 接雨水部分参数 ---
NUM_COLUMNS_N = 10
BITS_PER_HEIGHT = 3

# --- 元胞自动机部分参数 ---
# CA的宽度必须与接雨水第一步的输出位数一致
CA_WIDTH = NUM_COLUMNS_N * BITS_PER_HEIGHT
CA_RULE_NUMBER = 110
# 你可以调整演化层数，这是一个关键的实验变量
CA_EVOLUTION_LAYERS = 3 

DATASET_SIZE = 300000

# 文件名反映了这是一个组合任务
TRAIN_FILE = f'rain_water_then_ca_l{CA_EVOLUTION_LAYERS}_train.jsonl'
EVAL_FILE = f'rain_water_then_ca_l{CA_EVOLUTION_LAYERS}_eval.jsonl'


# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = NUM_COLUMNS_N * BITS_PER_HEIGHT
MAX_HEIGHT = 2 ** BITS_PER_HEIGHT - 1

# 输出格式与输入格式完全相同（结构保持）
OUTPUT_LEN = CA_WIDTH

print("=" * 70)
print(f"     “接雨水 -> CA演化”问题 - 数据集生成器 (任务D)")
print("=" * 70)
print(f"输入格式: {INPUT_LEN}个'0'/'1' (地形)")
print(f"中间状态: {CA_WIDTH}个'0'/'1' (每格雨水，作为CA初始状态)")
print(f"输出格式: {OUTPUT_LEN}个多标签二分类 (CA演化 {CA_EVOLUTION_LAYERS} 层后的状态)")
print(f"CA规则: Rule {CA_RULE_NUMBER}")
print("=" * 70)

# ==============================================================================
# --- 3. 核心逻辑 ---
# ==============================================================================

# --- 组件1: 接雨水求解器 (来自你的脚本) ---
def solve_trapping_rain_water_per_cell(height):
    n = len(height)
    if n == 0: return []
    water_per_cell = [0] * n
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])
    right_max = [0] * n
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > height[i]:
            water_per_cell[i] = water_level - height[i]
    return water_per_cell

# --- 组件2: 元胞自动机演化器 ---
def get_rule_map(rule_number):
    """根据规则编号生成一个8位的规则映射字典"""
    rule_binary = format(rule_number, '08b')
    rule_map = {}
    for i in range(8):
        pattern = format(7 - i, '03b')
        rule_map[pattern] = int(rule_binary[i])
    return rule_map

# 预先计算好规则映射，避免重复计算
CA_RULE_MAP = get_rule_map(CA_RULE_NUMBER)

def evolve_ca(state, rule_map):
    """对CA状态进行单步演化"""
    width = len(state)
    new_state = [0] * width
    for i in range(width):
        # 处理周期性边界条件
        left = state[(i - 1 + width) % width]
        center = state[i]
        right = state[(i + 1) % width]
        pattern = f"{left}{center}{right}"
        new_state[i] = rule_map[pattern]
    return new_state

# --- 样本生成器 ---
all_set = set()
def generate_heights(num_columns, max_height):
    """随机生成一个高度图数组，并确保不重复"""
    while True:
        heights = [random.randint(0, max_height) for _ in range(num_columns)]
        if tuple(heights) not in all_set:
            all_set.add(tuple(heights))
            return heights

def process_sample():
    """生成一个完整的 (输入, 输出) 数据对。"""
    # 1. 生成地形输入
    heights = generate_heights(NUM_COLUMNS_N, MAX_HEIGHT)
    input_str_list = [format(h, f'0{BITS_PER_HEIGHT}b') for h in heights]
    input_str = "".join(input_str_list)

    # 2. 计算中间状态：每格雨水
    water_per_cell = solve_trapping_rain_water_per_cell(heights)
    intermediate_state_str_list = [format(w, f'0{BITS_PER_HEIGHT}b') for w in water_per_cell]
    intermediate_state_str = "".join(intermediate_state_str_list)
    
    # 3. 将中间状态作为CA初始状态，并进行多步演化
    current_ca_state = [int(bit) for bit in intermediate_state_str]
    for _ in range(CA_EVOLUTION_LAYERS):
        current_ca_state = evolve_ca(current_ca_state, CA_RULE_MAP)

    # 4. 最终的CA状态就是我们的输出
    output_multilabel = current_ca_state

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 (与你的脚本结构保持一致) ---
# ==============================================================================
def generate_datasets():
    print("\n--- 开始生成数据集 ---")
    
    records = []
    for i in range(DATASET_SIZE):
        records.append(process_sample())
        if (i + 1) % 10000 == 0:
            print(f"已生成 {i + 1} / {DATASET_SIZE} 条数据...")

    random.shuffle(records)
    print(f"生成完毕。共 {len(records)} 条数据。")

    # 拆分训练集和评估集
    eval_size = min(int(len(records) * 0.1), 5000)
    train_data, eval_data = records[:-eval_size], records[-eval_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}数据到 '{path}'...")
        with open(path, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')

    write_to_file(records, TRAIN_FILE, "训练")
    # write_to_file(eval_data, EVAL_FILE, "评估")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
