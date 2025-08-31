import random
import json
from tqdm import tqdm

# ==============================================================================
# --------------------------- CONFIGURATION FLAGS ------------------------------
# ==============================================================================

# --- 基本参数 ---
NUM_SAMPLES = 500000  # 生成的样本数量
LENGTH = 30  # 元胞自动机的宽度
EVOLUTION_LAYERS = 2  # 演化的层数（步数）

# --- 扰动控制 (Perturbation Control) ---
# 设置为 0.0 表示不进行扰动
# 设置为 0.01 表示 1% 的扰动率 (即每个比特有1%的概率被翻转)

# 输入扰动率: 模拟观测噪声
INPUT_PERTURBATION_RATE = 0.1  # 例如: 5% 的输入噪声

# 输出/标签扰动率: 模拟标签噪声
OUTPUT_PERTURBATION_RATE = 0.00  # 例如: 1% 的输出/标签噪声

# --- 规则定义 ---
# 您可以轻松切换规则，例如使用 rule_30
SELECTED_RULE = "rule_110"

# ==============================================================================
# ------------------------------ SCRIPT LOGIC ----------------------------------
# ==============================================================================

# 规则转换表
rules = {
    "rule_110": {
        "111": "0", "110": "1", "101": "1", "100": "0",
        "011": "1", "010": "1", "001": "1", "000": "0"
    },
    "rule_30": {
        "111": "0", "110": "0", "101": "0", "100": "1",
        "011": "1", "010": "1", "001": "1", "000": "0"
    }
}

current_rule_map = rules[SELECTED_RULE]


def evolve(state):
    """根据选定的规则演化一步元胞自动机"""
    # 循环边界条件：左右首尾相连
    padded = state[-1] + state + state[0]
    return "".join(current_rule_map[padded[i:i + 3]] for i in range(len(state)))


def perturb_sequence(sequence, rate):
    """
    以给定的概率翻转序列中的每一个比特。
    支持字符串和整数列表。
    """
    if rate==0.0:
        return sequence

    if isinstance(sequence, str):
        perturbed = list(sequence)
        for i in range(len(perturbed)):
            if random.random() < rate:
                perturbed[i] = "1" if perturbed[i]=="0" else "0"
        return "".join(perturbed)

    elif isinstance(sequence, list):
        perturbed = list(sequence)
        for i in range(len(perturbed)):
            if random.random() < rate:
                perturbed[i] = 1 - perturbed[i]  # 翻转 0 和 1
        return perturbed

    else:
        raise TypeError("Unsupported sequence type for perturbation.")


def generate_dataset(num_samples, length, l, input_perturb_rate, output_perturb_rate):
    """
    生成带有可控扰动的数据集。
    """
    # --- 动态生成文件名 ---
    filename_parts = [
        f"ca_{SELECTED_RULE}",
        f"layer{l}",
        f"len{length}"
    ]
    if input_perturb_rate > 0:
        filename_parts.append(f"inp{int(input_perturb_rate * 1000)}p")
    if output_perturb_rate > 0:
        filename_parts.append(f"out{int(output_perturb_rate * 1000)}p")

    output_path = "_".join(filename_parts) + ".jsonl"

    print(f"--- Experiment Configuration ---")
    print(f"Rule: {SELECTED_RULE}, Layers: {l}, Length: {length}")
    print(f"Input Perturbation: {input_perturb_rate * 100:.1f}%")
    print(f"Output Perturbation: {output_perturb_rate * 100:.1f}%")
    print(f"Number of Samples: {num_samples}")
    print(f"Generating dataset at: {output_path}")
    print("---------------------------------")

    with open(output_path, "w") as f:
        for _ in tqdm(range(num_samples), desc="Generating Samples"):
            # 1. 生成原始输入
            original_input_seq = "".join(random.choice("01") for _ in range(length))

            # 2. 对输入进行扰动
            perturbed_input_seq = perturb_sequence(original_input_seq, input_perturb_rate)

            # 3. 基于 *扰动后* 的输入进行演化，得到正确的输出
            correct_output_seq = original_input_seq
            for _ in range(l):
                correct_output_seq = evolve(correct_output_seq)

            # 4. 将正确输出转换为整数列表
            correct_output_int_list = list(map(int, correct_output_seq))

            # 5. 对 *正确* 的输出进行扰动，得到最终的标签
            final_output_label = perturb_sequence(correct_output_int_list, output_perturb_rate)

            # 6. 写入文件
            sample = {
                "input": perturbed_input_seq,  # 使用扰动后的输入
                "output": final_output_label  # 使用扰动后的输出/标签
            }
            f.write(json.dumps(sample) + "\n")

    print(f"\nDataset generation complete. File saved to {output_path}")


# --- 主程序入口 ---
if __name__=="__main__":
    generate_dataset(
        num_samples=NUM_SAMPLES,
        length=LENGTH,
        l=EVOLUTION_LAYERS,
        input_perturb_rate=INPUT_PERTURBATION_RATE,
        output_perturb_rate=OUTPUT_PERTURBATION_RATE
    )