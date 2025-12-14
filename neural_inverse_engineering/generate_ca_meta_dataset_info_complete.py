# generate_ca_meta_dataset.py
import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000
    CA_WIDTH = 30
    CA_LAYERS = 1
    
    # --- 新增：是否强制每个样本都“信息完备” ---
    # 设置为 True，进行我们讨论的“简化版”第一步实验
    # 设置为 False，则生成通用的、信息不完备的数据集，用于真正的RNN训练
    FORCE_INFO_COMPLETE = True
    
    # --- 文件名 ---
    # 文件名将反映是否为“信息完备”版
    mode_str = "info_complete" if FORCE_INFO_COMPLETE else "standard"
    OUTPUT_FILE = f"ca_meta_l{CA_LAYERS}_w{CA_WIDTH}_{mode_str}.jsonl"

# ==============================================================================
# --- 2. 核心逻辑 (与之前版本一致) ---
# ==============================================================================
def generate_rule_map():
    """随机生成一个 CA 规则及其映射表。"""
    rule_index = random.randint(0, 255)
    rule_binary_str = format(rule_index, '08b')
    rule_map = {}
    patterns = [(i, j, k) for i in (1, 0) for j in (1, 0) for k in (1, 0)]
    for i, pattern in enumerate(patterns):
        rule_map[pattern] = int(rule_binary_str[i])
    return rule_index, rule_map

def evolve(state_list, rule_map, layers=1):
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

def has_all_patterns(s: str) -> bool:
    """检查字符串是否包含所有8种3-bit组合。"""
    required_patterns = {"000", "001", "010", "011", "100", "101", "110", "111"}
    found_patterns = set()
    padded = s[-1] + s + s[0]
    for i in range(len(s)):
        found_patterns.add(padded[i:i+3])
    return required_patterns.issubset(found_patterns)

# ==============================================================================
# --- 3. 样本生成函数 (格式已修改) ---
# ==============================================================================
def generate_sample(cfg):
    """
    生成一个 (S0, S1, RuleIndex) 的样本。
    """
    # 1. 随机生成一个规则 R
    rule_index, rule_map = generate_rule_map()
    
    # 2. 生成 S0
    if cfg.FORCE_INFO_COMPLETE:
        # 循环直到找到一个“信息完备”的 S0
        while True:
            s0_str = "".join(random.choice("01") for _ in range(cfg.CA_WIDTH))
            if has_all_patterns(s0_str):
                break
    else:
        # 标准模式：完全随机的 S0
        s0_str = "".join(random.choice("01") for _ in range(cfg.CA_WIDTH))
        
    s0_list = [int(bit) for bit in s0_str]
    
    # 3. 计算 S1
    s1_list = evolve(s0_list, rule_map, cfg.CA_LAYERS)
    
    # 4. 构建符合元学习格式的样本
    return {
        "input": s0_str,       # S0, 字符串格式
        "output": s1_list,     # S1, 0/1列表格式
        "rule_index": rule_index # 规则的整数索引
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"元学习CA规则归纳 - 数据集生成器")
    print(f"模式: {'信息完备样本' if cfg.FORCE_INFO_COMPLETE else '标准随机样本'}")
    print("=" * 70)
    print(f"CA宽度: {cfg.CA_WIDTH}, 演化层数: {cfg.CA_LAYERS}")
    print(f"输出格式: (input: S0, output: S1, rule_index: 0-255)")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
