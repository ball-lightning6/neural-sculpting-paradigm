# generate_multitask_prefixed_ca110.py

import json
import random
import math
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
INPUT_LEN = 30
DATASET_SIZE = 500000

# --- 前置任务参数 ---
PRE_CA_RULE_NUMBER = 110
PRE_CA_EVOLUTION_LAYERS = 4

# --- 后置任务A：二进制加法 ---
ADD_NUM_BITS = 15  # 输入是30位，所以每个数是15位

# --- 后置任务B：接雨水 ---
RAIN_NUM_COLUMNS_N = 10
RAIN_BITS_PER_HEIGHT = 3

# --- 后置任务C：mod 3 ---
MOD3_NUM_BITS = 30

# --- 后置任务D：CA Rule 30 ---
POST_CA_RULE_NUMBER = 30
POST_CA_EVOLUTION_LAYERS = 3
POST_CA_WIDTH = 30

# --- 文件名 ---
OUTPUT_FILE = f'multitask_prefixed_ca{PRE_CA_RULE_NUMBER}x{PRE_CA_EVOLUTION_LAYERS}.jsonl'

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
print("=" * 80)
print(f" \"前置CA + 4个后置任务\" 的多任务数据集生成器")
print("=" * 80)
print(f"共享前置任务: Rule {PRE_CA_RULE_NUMBER} 演化 {PRE_CA_EVOLUTION_LAYERS} 次")
print(f"输入格式: {INPUT_LEN}位二进制字符串")
print(f"输出格式: 包含4个独立任务结果的JSON对象")
print(f"数据集大小: {DATASET_SIZE}")
print("=" * 80)


# ==============================================================================
# --- 3. 所有任务的核心求解器 ---
# ==============================================================================

# --- 组件0: 前置CA演化器 ---
def get_ca_rule_map(rule_number):
    rule_binary = format(rule_number, '08b')
    return {format(7 - i, '03b'): int(rule_binary[i]) for i in range(8)}


PRE_CA_RULE_MAP = get_ca_rule_map(PRE_CA_RULE_NUMBER)


def evolve_ca(state_list, rule_map):
    width = len(state_list)
    new_state = [0] * width
    state_str = "".join(map(str, state_list))
    # 循环边界条件
    padded_state = state_str[-1] + state_str + state_str[0]
    for i in range(width):
        pattern = padded_state[i:i+3]
        new_state[i] = rule_map[pattern]
    return new_state


# --- 组件A: 二进制加法求解器 ---
def solve_binary_addition(binary_str: str):
    num1_str = binary_str[:ADD_NUM_BITS]
    num2_str = binary_str[ADD_NUM_BITS:]
    num1_int = int(num1_str, 2)
    num2_int = int(num2_str, 2)
    sum_int = num1_int + num2_int
    # 输出长度是 num_bits + 1 以处理可能的进位
    sum_str = format(sum_int, f'0{ADD_NUM_BITS + 1}b')
    return [int(bit) for bit in sum_str]


# --- 组件B: 接雨水求解器 ---
def solve_trapping_rain_water_per_cell(binary_str: str):
    heights = [int(binary_str[i:i+RAIN_BITS_PER_HEIGHT], 2) for i in range(0, len(binary_str), RAIN_BITS_PER_HEIGHT)]
    n = len(heights)
    if n == 0:
        return []
    water_per_cell = [0] * n
    left_max = [0] * n
    left_max[0] = heights[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], heights[i])
    right_max = [0] * n
    right_max[n - 1] = heights[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], heights[i])
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > heights[i]:
            water_per_cell[i] = water_level - heights[i]
    output_str = "".join([format(w, f'0{RAIN_BITS_PER_HEIGHT}b') for w in water_per_cell])
    return [int(bit) for bit in output_str]


# --- 组件C: Mod 3 DFA求解器 ---
MOD3_DFA_TRANSITIONS = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 0}, 2: {0: 1, 1: 2}}


def solve_mod3_dfa_trace(binary_str: str):
    current_state = 0
    state_trace = []
    for bit_char in binary_str:
        current_state = MOD3_DFA_TRANSITIONS[current_state][int(bit_char)]
        if current_state == 0:
            state_trace.extend([0, 0])
        elif current_state == 1:
            state_trace.extend([0, 1])
        else:
            state_trace.extend([1, 0])
    return state_trace


# --- 组件D: 后置CA演化器 ---
POST_CA_RULE_MAP = get_ca_rule_map(POST_CA_RULE_NUMBER)


def solve_post_ca(binary_str: str):
    current_state = [int(bit) for bit in binary_str]
    for _ in range(POST_CA_EVOLUTION_LAYERS):
        current_state = evolve_ca(current_state, POST_CA_RULE_MAP)
    return current_state


# ==============================================================================
# --- 4. 单个样本处理与主生成函数 ---
# ==============================================================================
def process_sample():
    # 1. 生成原始输入
    input_int = random.randint(0, 2 ** INPUT_LEN - 1)
    input_str = format(input_int, f'0{INPUT_LEN}b')

    # 2. 执行共享的前置任务
    prefixed_state = [int(bit) for bit in input_str]
    for _ in range(PRE_CA_EVOLUTION_LAYERS):
        prefixed_state = evolve_ca(prefixed_state, PRE_CA_RULE_MAP)
    
    prefixed_str = "".join(map(str, prefixed_state))

    # 3. 并行计算四个后置任务的输出
    output_A = solve_binary_addition(prefixed_str)
    output_B = solve_trapping_rain_water_per_cell(prefixed_str)
    output_C = solve_mod3_dfa_trace(prefixed_str)
    output_D = solve_post_ca(prefixed_str)

    return {
        "input": input_str,
        "inter": list(map(int, prefixed_str)),
        "output_add": output_A,
        "output_rain": output_B,
        "output_mod3": output_C,
        "output_ca30": output_D,
    }


def generate_datasets():
    print("\n--- 开始生成多任务数据集 ---")
    
    # 为了确保输入的多样性，我们使用set来避免原始输入重复
    all_inputs = set()
    records = []
    with tqdm(total=DATASET_SIZE, desc="生成样本") as pbar:
        while len(records) < DATASET_SIZE:
            sample = process_sample()
            if sample['input'] not in all_inputs:
                all_inputs.add(sample['input'])
                records.append(sample)
                pbar.update(1)

    with open(OUTPUT_FILE, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"\n✅ 数据集生成完成！共 {len(records)} 条数据已保存至 '{OUTPUT_FILE}'")
    
    print("\n--- 样本数据结构验证 ---")
    sample = records[0]
    for key, value in sample.items():
        if isinstance(value, list):
            print(f"字段 '{key}': 长度为 {len(value)} 的列表。 [验证通过]")
        else:
            print(f"字段 '{key}': 字符串。 [验证通过]")
    print("-" * 80)


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
