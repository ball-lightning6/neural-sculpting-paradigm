import os
import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 (请在此处定制您的任务) ---
# ==============================================================================
CA_WIDTH = 30  # 元胞自动机的位数长度
NUM_SAMPLES = 300000  # 您希望生成的数据集大小
OUTPUT_FILE = "ca_rule_dataset.jsonl"  # 输出文件名

# 注意：迭代层数固定为1，因为任务是根据单步演化来预测规则。
# 如果需要多步，需要重新定义问题（例如，输入是初始状态，输出是N步后的状态）。
ITERATION_LAYERS = 2


# ==============================================================================
# --- 2. 核心逻辑：元胞自动机模拟器与数据生成器 ---
# ==============================================================================

def apply_rule(state: list, rule_binary: list) -> list:
    """
    对给定的状态应用一次元胞自动机规则（周期性边界条件）。

    Args:
        state: 当前状态，一个由0和1组成的列表。
        rule_binary: 8位的规则，一个由0和1组成的列表。

    Returns:
        演化一次后的新状态列表。
    """
    width = len(state)
    next_state = [0] * width
    for i in range(width):
        # 获取邻域 (周期性边界)
        left = state[i - 1]
        center = state[i]
        right = state[(i + 1) % width]

        # 将邻域模式转换为十进制索引 (0-7)
        # 例如 '111' -> 7, '110' -> 6, ..., '000' -> 0
        neighborhood_index = left * 4 + center * 2 + right * 1

        # 规则的索引通常与邻域索引相反
        # rule_binary[0] 对应 '111', rule_binary[7] 对应 '000'
        rule_index = 7 - neighborhood_index

        next_state[i] = rule_binary[rule_index]

    return next_state


def generate_state_with_all_patterns(width: int) -> list:
    """
    生成一个保证包含所有8种3位邻域模式的状态。

    方法：使用一个已知的包含所有8个模式的De Bruijn序列 B(2,3) 作为核心，
    然后用随机位填充到指定宽度。

    Args:
        width: 目标状态的宽度。

    Returns:
        一个满足条件的初始状态列表。
    """
    # De Bruijn B(2,3) 序列 '00011101' 包含了所有8个3-bit模式（当循环看待时）
    # 例如：000, 001, 011, 111, 110, 101, 010, 100
    base_sequence = [0, 0, 0, 1, 1, 1, 0, 1]

    if width < len(base_sequence):
        raise ValueError(f"宽度必须至少为 {len(base_sequence)} 才能包含所有模式。")

    # 填充剩余部分
    remaining_len = width - len(base_sequence)
    random_padding = [random.randint(0, 1) for _ in range(remaining_len)]

    # 将核心序列随机插入到填充中
    insert_pos = random.randint(0, remaining_len)
    final_state = random_padding[:insert_pos] + base_sequence + random_padding[insert_pos:]

    return final_state


def generate_dataset():
    """
    主函数，生成完整的元胞自动机规则预测数据集。
    """
    print(f"🚀 开始生成数据集...")
    print(f"   - 元胞自动机宽度: {CA_WIDTH}")
    print(f"   - 样本数量: {NUM_SAMPLES}")
    print(f"   - 输出文件: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(NUM_SAMPLES), desc="Generating Samples"):
            # 1. 随机选择一个规则 (0-255)
            rule_num = random.randint(0, 255)
            # 将规则转换为8位二进制列表，作为我们的'output'标签
            rule_binary_label = [int(bit) for bit in format(rule_num, '08b')]

            # 2. 生成一个保证包含所有模式的初始状态
            initial_state = generate_state_with_all_patterns(CA_WIDTH)

            # 3. 应用规则，得到输出状态
            current_state = initial_state
            for _ in range(ITERATION_LAYERS):
                current_state = apply_rule(current_state, rule_binary_label)
            output_state = current_state

            # 4. 格式化 'input' 字符串
            # 输入状态 + 输出状态
            input_string = "".join(map(str, initial_state)) + "".join(map(str, output_state))

            # 5. 构造最终的JSON对象并写入文件
            data_point = {
                "input": input_string,
                "output": rule_binary_label
            }
            f.write(json.dumps(data_point) + '\n')

    print("\n✅ 数据集生成完毕！")
    # 打印一个样本看看
    with open(OUTPUT_FILE, 'r') as f:
        first_line = f.readline()
        print("\n示例数据 (第一条):")
        print(json.dumps(json.loads(first_line), indent=2))


# ==============================================================================
# --- 3. 执行入口 ---
# ==============================================================================

if __name__=="__main__":
    generate_dataset()
