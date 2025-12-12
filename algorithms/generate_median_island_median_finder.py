# generate_median_island_median_finder.py
import json
import random
from tqdm import tqdm
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000
    INPUT_WIDTH = 30

    # --- 文件名 ---
    OUTPUT_FILE = f"median_island_median_finder_w{INPUT_WIDTH}.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: "七重炼狱"算法 ---
# ==============================================================================
def find_median_of_median_island(input_str: str):
    """找到中位孤岛的中位'1'的最终绝对位置。"""

    # 步骤 1 & 2: 孤岛分割与计数
    islands = []
    in_island = False
    for i, char in enumerate(input_str):
        if char == '1' and not in_island:
            in_island = True
            current_island_start = i
        elif char == '0' and in_island:
            in_island = False
            islands.append((current_island_start, i - 1)) # 记录孤岛的起止位置
    # 处理末尾的孤岛
    if in_island:
        islands.append((current_island_start, len(input_str) - 1))

    island_count = len(islands)

    # 如果没有孤岛，定义特殊输出
    if island_count == 0:
        return -1

    # 步骤 3 & 4: 中位孤岛的定位
    median_island_index = math.floor((island_count - 1) / 2)
    median_island_start, median_island_end = islands[median_island_index]

    # 步骤 5 & 6: 中位'1'的内部序号计算
    count_in_island = median_island_end - median_island_start + 1
    median_k_in_island = math.floor((count_in_island - 1) / 2)

    # 步骤 7: 最终绝对位置计算
    final_position = median_island_start + median_k_in_island

    return final_position

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个样本。"""

    input_str = "".join(random.choice("01") for _ in range(cfg.INPUT_WIDTH))
    position = find_median_of_median_island(input_str)

    position_to_encode = cfg.INPUT_WIDTH if position == -1 else position
    output_bits_len = cfg.INPUT_WIDTH.bit_length()
    output_list = [int(bit) for bit in format(position_to_encode, f'0{output_bits_len}b')]

    return {
        "input": input_str,
        "output": output_list
    }

# ==============================================================================
# --- 4. 主生成函数与验证 ---
# ==============================================================================
def main():
    cfg = Config()

    print("=" * 70)
    print(f"""“中位孤岛的中位'1'定位” - 数据集生成器""")
    print("=" * 70)
    print(f"输入维度: {cfg.INPUT_WIDTH}, 输出维度: {cfg.INPUT_WIDTH.bit_length()}")
    print("=" * 70)

    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

    # 验证一个复杂的例子
    print("\n--- 样本逻辑验证 ---")
    test_input = "011010111001100011110101000111"
    # 孤岛们: (1,2), (4,4), (6,8), (11,12), (16,19), (21,21), (23,23), (27,29)
    # 共8个孤岛。中位孤岛是第4个，即 (11,12)
    # (11,12)这个孤岛是'11'，长度为2。中位'1'是第1个'1' (0-indexed)。
    # 它的绝对位置是 11 + 0 = 11。
    expected_pos = find_median_of_median_island(test_input)
    print(f"测试输入: {test_input}")
    print(f"预期位置: {expected_pos}")
    assert expected_pos == 11
    print("逻辑验证通过！")

if __name__ == "__main__":
    main()