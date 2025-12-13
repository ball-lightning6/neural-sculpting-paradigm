# generate_median_one_finder.py
# Median One Finder - 找到输入字符串中所有'1'的中间位置

import json
import random
import math
import os
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000
    INPUT_WIDTH = 30
    
    # --- 文件名 ---
    OUTPUT_FILE = f"median_one_finder_w{INPUT_WIDTH}.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: 中位数'1'定位算法 (用于生成标签) ---
# ==============================================================================
def find_median_one_position(input_str: str):
    """找到输入字符串中，所有'1'的中间位置。"""
    
    # 1. 找到所有'1'的索引
    one_indices = [i for i, char in enumerate(input_str) if char == '1']
    
    count = len(one_indices)
    
    # 如果没有'1'，我们可以定义一个特殊输出，比如返回-1或最大宽度
    if count == 0:
        return -1 # 或者 input_width，表示未找到
        
    # 2. 计算中位数的序号 k (1-indexed)
    # 偶数个1，例如8个，我们取第4个 (index 3)
    # 奇数个1，例如5个，我们取第3个 (index 2)
    median_k_index = math.floor((count -1) / 2)
    
    # 3. 从索引列表中，根据序号找到最终的位置
    median_position = one_indices[median_k_index]
    
    return median_position

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个 (二进制输入, 中位数'1'的位置) 的数据对。"""
    
    # 1. 随机生成二进制输入字符串
    input_str = "".join(random.choice("01") for _ in range(cfg.INPUT_WIDTH))
    
    # 2. 计算正确答案
    position = find_median_one_position(input_str)
    
    # 3. 将输出位置转换为固定长度的二进制表示
    #    位置范围是 0 to INPUT_WIDTH-1，或者-1。我们需要能表示 INPUT_WIDTH 种可能。
    #    我们用 (INPUT_WIDTH).bit_length() 来表示 0 到 INPUT_WIDTH-1。
    #    为了表示-1，我们可以用全1，或者增加一位。简单的做法是用一个能表示到 INPUT_WIDTH 的位数。
    if position == -1:
        position_to_encode = cfg.INPUT_WIDTH # 用最大值+1代表“未找到”
    else:
        position_to_encode = position
        
    output_bits_len = cfg.INPUT_WIDTH.bit_length()
    output_list = [int(bit) for bit in format(position_to_encode, f'0{output_bits_len}b')]

    return {
        "input": input_str,
        "output": output_list
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"中位数'1'定位 - 数据集生成器")
    print("=" * 70)
    
    input_dim = cfg.INPUT_WIDTH
    output_dim = cfg.INPUT_WIDTH.bit_length()
    
    print(f"输入维度: {input_dim}")
    print(f"输出维度 (位置的二进制): {output_dim}")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")
    
    # 验证一个例子
    print("\n--- 样本逻辑验证 ---")
    test_input = "010110100100000000000000000000" # 5个'1'
    expected_pos = find_median_one_position(test_input)
    print(f"测试输入: {test_input}")
    print(f"'1'的位置: {[i for i, char in enumerate(test_input) if char == '1']}")
    print(f"中位数'1'的位置 (0-indexed): {expected_pos} (预期是第3个'1'的位置, 即8)")
    assert expected_pos == 8, f"Expected 8 but got {expected_pos}"

    test_input_2 = "111111110000000000000000000000" # 8个'1'
    expected_pos_2 = find_median_one_position(test_input_2)
    print(f"\n测试输入: {test_input_2}")
    print(f"'1'的位置: {[i for i, char in enumerate(test_input_2) if char == '1']}")
    print(f"中位数'1'的位置 (0-indexed): {expected_pos_2} (预期是第4个'1'的位置, 即3)")
    assert expected_pos_2 == 3, f"Expected 3 but got {expected_pos_2}"
    
    print("逻辑验证通过!")

if __name__ == "__main__":
    main()
