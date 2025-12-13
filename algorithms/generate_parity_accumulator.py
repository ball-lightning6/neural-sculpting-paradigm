# generate_parity_accumulator.py
# Parity Accumulator - 奇偶指令累加器

import json
import random
import re
import os
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000
    INPUT_WIDTH = 30
    INITIAL_ACCUMULATOR = 32
    
    # --- 文件名 ---
    OUTPUT_FILE = f"parity_accumulator_w{INPUT_WIDTH}.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: “神经CPU”模拟器 ---
# ==============================================================================
def run_parity_accumulator(input_str: str, initial_value: int):
    """
    模拟奇偶指令累加器的计算过程。
    
    算法：
    1. 解析输入字符串为连续的 '0' 块和 '1' 块。
    2. 第一个 '1' 块（如果是开头）总是做加法。
    3. 后续 '1' 块根据前面所有 '0' 块的总长度的奇偶性决定操作：
       - '0' 总数为偶数 -> 加法
       - '0' 总数为奇数 -> 减法
    """
    accumulator = initial_value
    
    # 使用正则表达式高效地找到所有 '0' 块和 '1' 块
    tokens = [token for token in re.split(r'([0]+|[1]+)', input_str) if token]
    
    total_zeros_count = 0
    
    # 检查第一个token是否是'1'孤岛
    if tokens and tokens[0][0] == '1':
        # 前面有0个'0'，是偶数，执行加法
        island_length = len(tokens[0])
        accumulator += island_length

    # 遍历后续的token对 ('0'块, '1'块)
    for i in range(len(tokens) - 1):
        if tokens[i][0] == '0' and tokens[i+1][0] == '1':
            zeros_block = tokens[i]
            ones_block = tokens[i+1]
            
            total_zeros_count += len(zeros_block)
            island_length = len(ones_block)
            
            # 根据'0'的总数的奇偶性决定操作
            if total_zeros_count % 2 == 0: # 偶数
                accumulator += island_length
            else: # 奇数
                accumulator -= island_length
                
    return accumulator

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个样本。"""
    
    input_str = "".join(random.choice("01") for _ in range(cfg.INPUT_WIDTH))
    final_value = run_parity_accumulator(input_str, cfg.INITIAL_ACCUMULATOR)
    
    # 使用6位无符号二进制表示输出 (0-63)
    output_bits_len = 6
    
    # 确保值在可表示范围内
    # 由于初始值32，宽度30，理论最大值62，最小值2，范围安全
    final_value = max(0, min(final_value, 2**output_bits_len - 1))

    output_str = format(final_value, f'0{output_bits_len}b')
    output_list = [int(bit) for bit in output_str]

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
    print(f"“奇偶指令累加器” - 数据集生成器")
    print("=" * 70)
    
    input_dim = cfg.INPUT_WIDTH
    output_dim = 6 
    
    print(f"输入维度: {input_dim}")
    print(f"输出维度: {output_dim} (6-bit unsigned value)")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        pbar = tqdm(total=cfg.NUM_SAMPLES, desc="生成样本")
        count = 0
        while count < cfg.NUM_SAMPLES:
            sample = generate_sample(cfg)
            if sample:
                f.write(json.dumps(sample) + "\n")
                count += 1
                pbar.update(1)
        pbar.close()
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")
    
    # 逻辑验证
    print("\n--- 样本逻辑验证 ---")
    # Case: 0011 -> 0s=2(even) -> +2. start=32 -> 34. 
    # 01 -> 0s=2+1=3(odd) -> -1. start=34 -> 33.
    # 00011 -> 0s=3+3=6(even) -> +2. start=33 -> 35.
    # Total input: 00110100011 + padding
    test_input = "00110100011" + "0" * 19 
    expected_value = run_parity_accumulator(test_input, cfg.INITIAL_ACCUMULATOR)
    print(f"测试输入: {test_input}")
    print(f"预期最终值: {expected_value}")
    # 32 + 2(11, pre0=2) - 1(1, pre0=3) + 2(11, pre0=6) = 35
    assert expected_value == 35, f"Expected 35, got {expected_value}"
    print("逻辑验证通过！")

if __name__ == "__main__":
    main()
