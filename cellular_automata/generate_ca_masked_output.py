# generate_ca_masked.py
import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 1000_000
    CA_WIDTH = 30
    CA_LAYERS = 3
    
    # --- Masking 配置 ---
    MASK_VISIBLE_BITS = 15 # Mask中'1'的数量
    
    # --- 文件名 ---
    OUTPUT_FILE = f"ca_rule110_l{CA_LAYERS}_w{CA_WIDTH}_masked{MASK_VISIBLE_BITS}.jsonl"

# --- Rule 110 (无变化) ---
rule_110_map = {
    (1,1,1): 0, (1,1,0): 1, (1,0,1): 1, (1,0,0): 0,
    (0,1,1): 1, (0,1,0): 1, (0,0,1): 1, (0,0,0): 0
}

def evolve(state_list):
    n = len(state_list)
    next_state = [0] * n
    for i in range(n):
        left = state_list[(i - 1 + n) % n]
        center = state_list[i]
        right = state_list[(i + 1) % n]
        next_state[i] = rule_110_map.get((left, center, right), 0)
    return next_state

# ==============================================================================
# --- 2. 核心逻辑与样本生成 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个 (输入+Mask, 部分输出) 的数据对。"""
    
    # 1. 生成CA初始状态 (S_0)
    initial_state_str = "".join(random.choice("01") for _ in range(cfg.CA_WIDTH))
    initial_state_list = [int(bit) for bit in initial_state_str]

    # 2. 计算完整的最终状态 (S_3)
    final_state_list = initial_state_list
    for _ in range(cfg.CA_LAYERS):
        final_state_list = evolve(final_state_list)

    # 3. 生成随机的Mask (M)
    mask_list = [1] * cfg.MASK_VISIBLE_BITS + [0] * (cfg.CA_WIDTH - cfg.MASK_VISIBLE_BITS)
    random.shuffle(mask_list)
    mask_str = "".join(map(str, mask_list))

    # 4. 组合成最终的输入
    # 输入格式: 30位S_0 + 30位Mask M
    input_str = initial_state_str + mask_str
    
    # 5. 应用Mask来获取部分输出
    masked_output_list = []
    for i in range(cfg.CA_WIDTH):
        if mask_list[i] == 1:
            masked_output_list.append(final_state_list[i])
            
    # 确保输出长度正确
    assert len(masked_output_list) == cfg.MASK_VISIBLE_BITS

    return {
        "input": input_str,
        "output": masked_output_list
    }

def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"部分可观测元胞自动机 - 数据集生成器")
    print("=" * 70)
    print(f"CA宽度: {cfg.CA_WIDTH}, 演化层数: {cfg.CA_LAYERS}")
    print(f"输入格式: {cfg.CA_WIDTH} bit (S_0) + {cfg.CA_WIDTH} bit (Mask)")
    print(f"总输入维度: {cfg.CA_WIDTH * 2}")
    print(f"输出格式: {cfg.MASK_VISIBLE_BITS} bit (被Mask选择的部分S_{cfg.CA_LAYERS})")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
