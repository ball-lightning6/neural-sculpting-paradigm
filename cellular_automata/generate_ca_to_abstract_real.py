
import random
import json
import os
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000 # 鉴于任务的极端难度，需要大量样本
    
    # --- 元胞自动机配置 ---
    CA_WIDTH = 30
    CA_LAYERS = 5 # 先从一个相对简单的层数开始
    
    # --- 输出编码配置 (实验核心) ---
    # 输出的30位二进制，将被两两分组，编码成15个实数
    REAL_SYMBOL_MAP = {
        (0, 0): 7.6,
        (0, 1): 1.3,
        (1, 0): 5.9,
        (1, 1): 3.0,
    }
    
    # --- 自动计算的参数 ---
    INPUT_DIM = CA_WIDTH
    OUTPUT_DIM = CA_WIDTH // 2
    
    # --- 文件名 ---
    OUTPUT_FILE = f"ca_l{CA_LAYERS}_w{CA_WIDTH}_to_abstract_real.jsonl"

# --- Rule 110 (无变化) ---
rule_110_map = {
    (1,1,1): 0, (1,1,0): 1, (1,0,1): 1, (1,0,0): 0,
    (0,1,1): 1, (0,1,0): 1, (0,0,1): 1, (0,0,0): 0
}

def evolve(state_list, layers):
    n = len(state_list)
    current_state = list(state_list)
    for _ in range(layers):
        next_state = [0] * n
        for i in range(n):
            left = current_state[(i - 1 + n) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            next_state[i] = rule_110_map.get((left, center, right), 0)
        current_state = next_state
    return current_state

# ==============================================================================
# --- 2. 核心逻辑与样本生成 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个 (二进制输入, 任意实数输出) 的数据对。"""
    
    # 1. 生成CA初始状态 (S_0)
    initial_state_str = "".join(random.choice("01") for _ in range(cfg.CA_WIDTH))
    initial_state_list = [int(bit) for bit in initial_state_str]

    # 2. 计算完整的最终二进制状态 (S_final)
    final_state_binary = evolve(initial_state_list, cfg.CA_LAYERS)

    # 3. 将最终的二进制状态，按两位一组，编码为任意实数
    output_real_list = []
    if len(final_state_binary) % 2 != 0:
        raise ValueError("CA宽度必须是偶数才能进行2-bit分组！")
        
    for i in range(0, len(final_state_binary), 2):
        two_bits = tuple(final_state_binary[i:i+2])
        real_value = cfg.REAL_SYMBOL_MAP[two_bits]
        output_real_list.append(real_value)
            
    assert len(output_real_list) == cfg.OUTPUT_DIM

    return {
        "input": initial_state_str,
        "output": output_real_list
    }

def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"CA -> 任意实数符号 - 数据集生成器")
    print("=" * 70)
    print(f"CA宽度: {cfg.CA_WIDTH}, 演化层数: {cfg.CA_LAYERS}")
    print(f"输入格式: {cfg.INPUT_DIM} bit (二进制)")
    print(f"输出格式: {cfg.OUTPUT_DIM} 个实数符号")
    print(f"输出符号映射: {cfg.REAL_SYMBOL_MAP}")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
