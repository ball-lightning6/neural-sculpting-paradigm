import os
import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 (请在此处定制您的任务) ---
# ==============================================================================
CA_WIDTH = 30  # 元胞自动机的位数长度
NUM_SAMPLES = 50000  # 您希望生成的数据集大小
MAX_ITERATION_LAYERS = 7  # <-- 新增：最大迭代层数 (例如，7代表层数在1到7之间随机)
OUTPUT_FILE = "ca_dynamic_rule_dataset.jsonl"  # 输出文件名

# ==============================================================================
# --- 2. 派生配置 (脚本自动计算) ---
# ==============================================================================
# 自动计算表示迭代层数需要的二进制位数
# 例如，MAX_ITERATION_LAYERS = 7, (7).bit_length() = 3, 需要3位 (可以表示0-7)
ITERATION_BITS = (MAX_ITERATION_LAYERS-1).bit_length()
TOTAL_OUTPUT_BITS = 8 + ITERATION_BITS  # 8位规则 + k位层数


# ==============================================================================
# --- 3. 核心逻辑 (与之前相同，但会在循环中被多次调用) ---
# ==============================================================================

def apply_rule(state: list, rule_binary: list) -> list:
    """对给定的状态应用一次元胞自动机规则（周期性边界条件）。"""
    width = len(state)
    next_state = [0] * width
    for i in range(width):
        left, center, right = state[i - 1], state[i], state[(i + 1) % width]
        neighborhood_index = left * 4 + center * 2 + right * 1
        rule_index = 7 - neighborhood_index
        next_state[i] = rule_binary[rule_index]
    return next_state


def generate_state_with_all_patterns(width: int) -> list:
    """生成一个保证包含所有8种3位邻域模式的状态。"""
    base_sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    if width < len(base_sequence):
        raise ValueError(f"宽度必须至少为 {len(base_sequence)} 才能包含所有模式。")
    remaining_len = width - len(base_sequence)
    random_padding = [random.randint(0, 1) for _ in range(remaining_len)]
    insert_pos = random.randint(0, remaining_len)
    final_state = random_padding[:insert_pos] + base_sequence + random_padding[insert_pos:]
    return final_state


# ==============================================================================
# --- 4. 主生成函数 (已为动态任务更新) ---
# ==============================================================================

def generate_dataset():
    """
    主函数，生成包含动态迭代层数的元胞自动机规则预测数据集。
    """
    print(f"🚀 开始生成动态数据集...")
    print(f"   - 元胞自动机宽度: {CA_WIDTH}")
    print(f"   - 最大迭代层数: {MAX_ITERATION_LAYERS}")
    print(f"   - 样本数量: {NUM_SAMPLES}")
    print(f"   - 输出文件: {OUTPUT_FILE}")
    print(f"   - 迭代层数将使用 {ITERATION_BITS} 位二进制表示。")
    print(f"   - 总输出标签长度: 8 (规则) + {ITERATION_BITS} (层数) = {TOTAL_OUTPUT_BITS} 位。")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(NUM_SAMPLES), desc="Generating Samples"):
            # 1. 随机选择一个规则 (0-255)
            rule_num = random.randint(0, 255)
            rule_binary_label = [int(bit) for bit in format(rule_num, '08b')]

            # --- 核心修改：随机选择迭代层数 ---
            iteration_count = random.randint(1, MAX_ITERATION_LAYERS)
            iteration_binary_label = [int(bit) for bit in format(iteration_count-1, f'0{ITERATION_BITS}b')]

            # 2. 构造复合式的最终输出标签
            final_output_label = rule_binary_label + iteration_binary_label

            # 3. 生成一个保证包含所有模式的初始状态
            initial_state = generate_state_with_all_patterns(CA_WIDTH)

            # 4. 应用规则 `iteration_count` 次，得到输出状态
            current_state = initial_state
            for _ in range(iteration_count):
                current_state = apply_rule(current_state, rule_binary_label)
            output_state = current_state

            # 5. 格式化 'input' 字符串
            input_string = "".join(map(str, initial_state)) + "".join(map(str, output_state))

            # 6. 构造最终的JSON对象并写入文件
            data_point = {
                "input": input_string,
                "output": final_output_label  # <-- 使用复合标签
            }
            f.write(json.dumps(data_point) + '\n')

    print("\n✅ 数据集生成完毕！")
    # 打印一个样本看看
    with open(OUTPUT_FILE, 'r') as f:
        first_line = f.readline()
        print("\n示例数据 (第一条):")
        print(json.dumps(json.loads(first_line), indent=2))


if __name__=="__main__":
    generate_dataset()
