import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_BITS = 30  # 输入的二进制数的位数
DATASET_SIZE = 500000  # 要生成的数据样本总数

# --- 文件名 ---
OUTPUT_FILE = f'binary_mod3_dfa_explain_n{NUM_BITS}.jsonl'

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
print("=" * 70)
print(f" 二进制数 mod 3 - DFA解释性数据集生成器")
print("=" * 70)
print(f"输入位数 (N): {NUM_BITS}")
print(f"将为每个样本生成2种标签:")
print(f"  1. final_mod_result (主模型目标, 2 bits)")
print(f"  2. dfa_state_trace (DFA状态轨迹解释, {NUM_BITS * 2} bits)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：DFA模拟器/解释器 ---
# ==============================================================================

def solve_and_explain_mod3_dfa(binary_str: str):
    """
    使用有限状态自动机(DFA)计算二进制数的mod 3，并记录状态转移轨迹。

    Args:
        binary_str (str): 输入的二进制字符串。

    Returns:
        tuple: (最终结果[0,1,2], 状态轨迹列表[[s_bit1, s_bit0], ...])
    """
    n = len(binary_str)

    # DFA状态定义: S0=0, S1=1, S2=2
    current_state = 0
    state_trace = []

    # 状态转移表: transitions[current_state][input_bit] -> next_state
    transitions = {
        0: {0: 0, 1: 1},  # S0: 读0->S0, 读1->S1
        1: {0: 2, 1: 0},  # S1: 读0->S2, 读1->S0
        2: {0: 1, 1: 2}  # S2: 读0->S1, 读1->S2
    }

    # 从最高位(MSB, 字符串左边)开始处理
    for bit_char in binary_str:
        bit = int(bit_char)
        current_state = transitions[current_state][bit]

        # 将当前状态编码为2位二进制并记录
        # S0 -> [0,0], S1 -> [0,1], S2 -> [1,0]
        if current_state == 0:
            state_trace.extend([0, 0])
        elif current_state == 1:
            state_trace.extend([0, 1])
        else:  # current_state == 2
            state_trace.extend([1, 0])

    # 最终结果编码
    final_result = [0, 0]
    if current_state == 1:
        final_result = [0, 1]
    elif current_state == 2:
        final_result = [1, 0]

    return final_result, state_trace


# ==============================================================================
# --- 4. 单个样本处理与主生成函数 ---
# ==============================================================================

def process_sample(num_bits):
    """生成一个包含输入和所有两种标签的完整数据对。"""
    # 生成一个随机的N位二进制数
    num_int = random.randint(0, 2 ** num_bits - 1)
    input_str = format(num_int, f'0{num_bits}b')

    # 使用DFA求解并生成解释
    final_result_label, dfa_trace_label = solve_and_explain_mod3_dfa(input_str)

    return {
        "input": input_str,
        "final_mod_result": final_result_label,
        "dfa_state_trace": dfa_trace_label
    }


def generate_datasets():
    print("\n--- 开始生成DFA解释性数据集 ---")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            record = process_sample(NUM_BITS)
            f.write(json.dumps(record) + '\n')

    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample = process_sample(NUM_BITS)
    for key, value in sample.items():
        if isinstance(value, list):
            print(f"字段 '{key}': 是一个长度为 {len(value)} 的扁平整数列表。 [验证通过]")
        else:
            print(f"字段 '{key}': 是一个字符串。 [验证通过]")
    print("-" * 70)


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()