# generate_ca_meta_dataset_final.py
import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 5_000_000 # 在歧义性存在的情况下，需要更大的数据集
    CA_WIDTH = 30
    
    # --- 文件名 ---
    OUTPUT_FILE = f"ca_meta_l1_or_l2_w{CA_WIDTH}_ambiguous.jsonl"

# ==============================================================================
# --- 2. 核心逻辑 (与之前版本一致) ---
# ==============================================================================
def generate_rule_map(rule_index):
    """根据给定的规则索引，生成确定性的 CA 规则映射表。"""
    rule_binary_str = format(rule_index, '08b')
    rule_map = {}
    patterns = [(i, j, k) for i in (1, 0) for j in (1, 0) for k in (1, 0)]
    for i, pattern in enumerate(patterns):
        rule_map[pattern] = int(rule_binary_str[i])
    return rule_map

def precompute_rule_fingerprints_v2(cfg):
    """
    为所有512个(Rule, Layer)组合，生成一个理论完备的、基于“5-bit邻域真值表”的
    行为“指纹”，用于精确识别等价规则。
    """
    print("正在预计算所有规则的“理论完备”指纹以识别等价规则...")
    fingerprints = {}
    
    # 1. 遍历所有 2^5 = 32 种可能的5-bit邻域输入
    five_bit_inputs = []
    for i in range(32):
        # 将其转换为一个5-bit的列表 [0,1,0,1,1]
        five_bit_inputs.append([int(b) for b in format(i, '05b')])

    # 2. 遍历所有 512 种可能的“宇宙定律”
    for rule_idx in range(256):
        rule_map = generate_rule_map(rule_idx)
        for layer in [1, 2]:
            truth_table_output = []
            
            # 3. 对每种定律，计算其完整的32位真值表
            for s0_5bit in five_bit_inputs:
                # 演化1层
                s1_3bit = [
                    rule_map.get(tuple(s0_5bit[0:3]), 0),
                    rule_map.get(tuple(s0_5bit[1:4]), 0),
                    rule_map.get(tuple(s0_5bit[2:5]), 0)
                ]
                
                if layer == 1:
                    # 对于1层演化，我们只关心中心点的输出
                    # S1[i] 依赖于 S0[i-1, i, i+1]，对应5-bit输入的中间3位
                    final_bit = s1_3bit[1]
                else: # layer == 2
                    # 对于2层演化，S2[i] 依赖于 S1[i-1, i, i+1]
                    # 这对应于我们刚计算出的 s1_3bit
                    final_bit = rule_map.get(tuple(s1_3bit), 0)
                
                truth_table_output.append(str(final_bit))
            
            # 4. 将32位的真值表，作为该定律的唯一“指纹”
            fingerprint = "".join(truth_table_output)
            
            key = (rule_idx, layer)
            if fingerprint not in fingerprints:
                fingerprints[fingerprint] = []
            fingerprints[fingerprint].append(key)
            
    # 5. 从等价的规则组中，只保留第一个作为代表 (逻辑不变)
    unique_rules = []
    # 为了结果的确定性，对fingerprint进行排序
    for fp in sorted(fingerprints.keys()):
        # 同样为了确定性，对等价的规则也进行排序
        unique_rules.append(sorted(fingerprints[fp])[0])
        
    print(f"预计算完成！发现 {512 - len(unique_rules)} 组等价规则。")
    print(f"将只使用 {len(unique_rules)} 个唯一的、可区分的规则进行训练。")
    return unique_rules

def evolve(state_list, rule_map, layers):
    """根据给定的规则表演化状态。"""
    n = len(state_list)
    current_state = list(state_list)
    for _ in range(layers):
        next_state = [0] * n
        for i in range(n):
            left = current_state[(i - 1 + n) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            next_state[i] = rule_map.get((left, center, right), 0)
        current_state = next_state
    return current_state

# ==============================================================================
# --- 3. 样本生成函数 (格式已修改) ---
# ==============================================================================
def generate_sample(cfg, rule_layer_pair):
    # (这个函数现在接收一个确定的 rule_layer_pair)
    rule_index, layers = rule_layer_pair
    rule_map = generate_rule_map(rule_index)
    
    s0_str = "".join(random.choice("01") for _ in range(cfg.CA_WIDTH))
    s0_list = [int(bit) for bit in s0_str]
    s_final_list = evolve(s0_list, rule_map, layers)
    
    # 编码标签 (9-bit)
    rule_bits = format(rule_index, '08b')
    layer_bit = '0' if layers == 1 else '1'
    label_str = rule_bits + layer_bit
    label_list = [int(bit) for bit in label_str]

    return {
        "input": s0_str,
        "output": s_final_list,
        "rule_and_layer_label": label_list
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()
    
    # --- 【核心修正】: 先过滤出唯一的规则 ---
    unique_rule_layer_pairs = precompute_rule_fingerprints_v2(cfg)
    
    print("=" * 70)
    print(f"最终挑战 (无歧义版) - 数据集生成器")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            # 在每一次生成时，从“唯一规则列表”中随机抽取一个
            chosen_rule_pair = random.choice(unique_rule_layer_pairs)
            sample = generate_sample(cfg, chosen_rule_pair)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
