# generate_convergent_abstraction.py

import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# --- 中间表示的参数 ---
# 这是两个任务汇合的"枢纽"，它的位数是固定的
INTERMEDIATE_BITS = 30 

# --- 任务A (接雨水) 的参数 ---
# 我们需要调整接雨水的参数，使其输出恰好是 INTERMEDIATE_BITS 位
RAIN_NUM_COLUMNS_N = 10
RAIN_BITS_PER_HEIGHT = 3  # 10 * 3 = 30, 完美匹配

# --- 任务B (二进制加法) 的参数 ---
# 我们需要调整加法的参数，使其输入是 INTERMEDIATE_BITS 位
ADD_NUM_BITS = INTERMEDIATE_BITS # 每个加数是30位

# --- 公共后置任务的参数 ---
POST_CA_RULE_NUMBER = 110
POST_CA_EVOLUTION_LAYERS = 6

# --- 数据集参数 ---
DATASET_SIZE = 500000
OUTPUT_FILE = f'convergent_abstraction_exp.jsonl'

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
print("=" * 80)
print(f" '收敛式抽象' (多输入->单枢纽) - 数据集生成器")
print("=" * 80)
print(f"任务A (接雨水): {RAIN_NUM_COLUMNS_N}x{RAIN_BITS_PER_HEIGHT}bit 地形 -> {INTERMEDIATE_BITS}bit 每格雨水")
print(f"任务B (加法):  2x{ADD_NUM_BITS}bit 数字 -> {INTERMEDIATE_BITS+1}bit 和 (我们将截取低位)")
print(f"共享中间表示 (枢纽): {INTERMEDIATE_BITS} bits")
print(f"公共后置任务: 对枢纽进行 {POST_CA_EVOLUTION_LAYERS} 次 Rule {POST_CA_RULE_NUMBER} 演化")
print(f"数据集大小: {DATASET_SIZE}")
print("=" * 80)

# ==============================================================================
# --- 3. 核心求解器与生成器 ---
# ==============================================================================

# --- 组件A: 接雨水求解器 ---
def solve_trapping_rain_water_per_cell(heights):
    n = len(heights)
    if n == 0: return []
    water_per_cell = [0] * n
    left_max = [0] * n; left_max[0] = heights[0]
    for i in range(1, n): left_max[i] = max(left_max[i - 1], heights[i])
    right_max = [0] * n; right_max[n - 1] = heights[n - 1]
    for i in range(n - 2, -1, -1): right_max[i] = max(right_max[i + 1], heights[i])
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > heights[i]: water_per_cell[i] = water_level - heights[i]
    return water_per_cell

# --- 组件B: 逆向加法生成器 ---
def reverse_engineer_addition(target_sum_int, num_bits):
    """
    给定目标和，逆向生成两个加数。
    确保两个加数都是 num_bits 位。
    """
    max_addend = 2**num_bits - 1
    
    # 确保 A 和 B 都不超过最大值
    if target_sum_int > 2 * max_addend:
        # 这种情况几乎不会发生，但作为保护
        num1_int = max_addend
        num2_int = max_addend
    elif target_sum_int > max_addend:
        num1_int = random.randint(target_sum_int - max_addend, max_addend)
        num2_int = target_sum_int - num1_int
    else:
        num1_int = random.randint(0, target_sum_int)
        num2_int = target_sum_int - num1_int
        
    num1_str = format(num1_int, f'0{num_bits}b')
    num2_str = format(num2_int, f'0{num_bits}b')
    return num1_str, num2_str

# --- 组件C: 公共CA演化器 ---
CA_RULE_MAP = {format(7 - i, '08b')[5:]: int(format(POST_CA_RULE_NUMBER, '08b')[i]) for i in range(8)}
CA_RULE_MAP_30 = {format(7 - i, '08b')[5:]: int(format(30, '08b')[i]) for i in range(8)}

def evolve_ca_n_times(initial_state_str, layers, rule_map):
    current_state = [int(bit) for bit in initial_state_str]
    for _ in range(layers):
        width = len(current_state)
        new_state = [0] * width
        state_str_temp = "".join(map(str, current_state))
        padded_state = state_str_temp[-1] + state_str_temp + state_str_temp[0]
        for i in range(width):
            pattern = padded_state[i:i+3]
            new_state[i] = rule_map[pattern]
        current_state = new_state
    return current_state

# ==============================================================================
# --- 4. 单个样本处理与主生成函数 ---
# ==============================================================================

def process_sample():
    # 1. 以任务A为起点，生成 Input_1 和 中间表示
    max_height = 2 ** RAIN_BITS_PER_HEIGHT - 1
    heights = [random.randint(0, max_height) for _ in range(RAIN_NUM_COLUMNS_N)]
    input_1_str = "".join([format(h, f'0{RAIN_BITS_PER_HEIGHT}b') for h in heights])
    
    water_per_cell = solve_trapping_rain_water_per_cell(heights)
    intermediate_representation_str = "".join([format(w, f'0{RAIN_BITS_PER_HEIGHT}b') for w in water_per_cell])
    
    # 2. 基于中间表示，逆向生成 Input_2
    intermediate_int = int(intermediate_representation_str, 2)
    num1_str, num2_str = reverse_engineer_addition(intermediate_int, ADD_NUM_BITS)
    input_2_str = num1_str + num2_str

    # 3. 对共享的中间表示，执行公共的后置任务，得到最终输出
    inter = evolve_ca_n_times(intermediate_representation_str, 4, CA_RULE_MAP)
    
    final_output_list = evolve_ca_n_times(inter, 4, CA_RULE_MAP)
    final_output_list_30 = evolve_ca_n_times(inter, 2, CA_RULE_MAP_30)

    return {
        "input_rain": input_1_str,
        "input_add": input_2_str,
        "intermediate_shared": [int(b) for b in intermediate_representation_str],
        "final_output": final_output_list,
        "final_output_30": final_output_list_30
    }

def generate_datasets():
    print("\n--- 开始生成'收敛式抽象'数据集 ---")
    
    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            record = process_sample()
            f.write(json.dumps(record) + '\n')
            
    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")
    
    print("\n--- 样本数据结构验证 ---")
    sample = process_sample()
    print(json.dumps(sample, indent=2))
    print("-" * 80)

# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
