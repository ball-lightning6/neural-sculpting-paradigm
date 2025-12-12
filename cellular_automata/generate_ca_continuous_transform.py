
import random
import json
import os
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000   # 建议使用较大的数据集
    
    # --- 元胞自动机配置 ---
    CA_WIDTH = 30
    CA_LAYERS = 5           # 演化层数
    
    # --- 输入编码配置 ---
    # 输入实数在 [0, 0.25] 区间代表逻辑 '0'
    # 输入实数在 [0.75, 1] 区间代表逻辑 '1'
    
    # --- 自动计算的参数 ---
    INPUT_DIM = CA_WIDTH
    OUTPUT_DIM = CA_WIDTH
    
    # --- 文件名 ---
    OUTPUT_FILE = f"ca_l{CA_LAYERS}_w{CA_WIDTH}_continuous_transform.jsonl"

# --- Rule 110 (与之前一致) ---
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
    """
    生成一个 (模糊符号输入, 条件变换实数输出) 的数据对。
    """
    
    input_real_list = []
    input_logic_list = []

    # 1. 生成30个随机的、代表'0'或'1'的实数作为输入
    for _ in range(cfg.CA_WIDTH):
        if random.random() > 0.5:
            # 代表 '1'
            val = random.uniform(0.75, 1.0)
            logic = 1
        else:
            # 代表 '0'
            val = random.uniform(0.0, 0.25)
            logic = 0
        input_real_list.append(val)
        input_logic_list.append(logic)
        
    # 2. 对逻辑列表进行CA演化，得到输出的逻辑状态
    output_logic_list = evolve(input_logic_list, cfg.CA_LAYERS)
    
    # 3. 根据核心规则，生成最终的输出实数列表
    output_real_list = []
    for i in range(cfg.CA_WIDTH):
        if output_logic_list[i] == input_logic_list[i]:
            # 逻辑符号相同，输出值 = 输入值
            output_real_list.append(input_real_list[i])
        else:
            # 逻辑符号不同，输出值 = 1.0 - 输入值
            output_real_list.append(1.0 - input_real_list[i])
            
    assert len(input_real_list) == cfg.INPUT_DIM
    assert len(output_real_list) == cfg.OUTPUT_DIM

    return {
        "input": input_real_list,
        "output": output_real_list
    }

def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"CA 条件数值变换 - 数据集生成器")
    print("=" * 70)
    print(f"CA宽度: {cfg.CA_WIDTH}, 演化层数: {cfg.CA_LAYERS}")
    print(f"输入格式: {cfg.INPUT_DIM} 个在 [0,0.25] U [0.75,1] 的实数")
    print(f"输出格式: {cfg.OUTPUT_DIM} 个条件变换后的实数")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")
    
    # 打印一个样本进行验证
    print("\n--- 样本数据结构验证 ---")
    sample = generate_sample(cfg)
    # 为了可读性，对浮点数进行格式化
    formatted_sample = {
        "input": [f"{x:.3f}" for x in sample["input"][:5]] + ["..."],
        "output": [f"{y:.3f}" for y in sample["output"][:5]] + ["..."]
    }
    print(json.dumps(formatted_sample, indent=2))
    print(f"输入向量长度: {len(sample['input'])}")
    print(f"输出向量长度: {len(sample['output'])}")
    
    print("-" * 70)
    print("注意：该数据集包含连续实数，请使用MLP或支持浮点输入输出的模型进行训练。")

if __name__ == "__main__":
    main()
