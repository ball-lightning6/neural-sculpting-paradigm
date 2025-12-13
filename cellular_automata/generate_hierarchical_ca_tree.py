# generate_hierarchical_ca_tree.py
"""
层次化抽象CA树 - 数据集生成器

实验目的：更进一步验证中间表示的涌现现象
核心思想：通过父节点的两个孙节点和另一个子节点的组合来检查涌现现象

理论背景：
如果说前面的多任务实验验证了共有子节点的训练是否会涌现父节点表示，
那么这个实验通过更复杂的层次化结构（祖父-父亲-孙子三代关系）来检查：
1. 神经网络能否从多个子节点的状态推断出父节点的状态？
2. 这种层次化推理是否会自然涌现出对中间抽象层的理解？

输入格式: 30位二进制字符串 (Root)
树状结构 (演化层数): Base(4) -> Branch(5) -> Leaf(3)
输出格式: 包含树上所有7个节点状态的JSON对象
"""

import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
INPUT_LEN = 30
DATASET_SIZE = 500000

# --- 树状演化结构配置 ---
# 每一段演化的层数，可根据难度调整
BASE_LAYERS = 4
BRANCH_LAYERS = 5
LEAF_LAYERS = 3

# --- 使用的CA规则 ---
RULE_BASE = 110
RULE_OTHER = 30
RULE_L = 110
RULE_R = 30
RULE_LL = 110
RULE_LR = 30
RULE_RL = 110
RULE_RR = 30

# --- 文件名 ---
OUTPUT_FILE = f'ca_tree_l{BASE_LAYERS}-{BRANCH_LAYERS}-{LEAF_LAYERS}_1.jsonl'

print("=" * 80)
print(f" “层次化抽象”CA树 - 数据集生成器")
print("=" * 80)
print(f"输入格式: {INPUT_LEN}位二进制字符串 (Root)")
print(f"树状结构 (演化层数): Base({BASE_LAYERS}) -> Branch({BRANCH_LAYERS}) -> Leaf({LEAF_LAYERS})")
print(f"输出格式: 包含树上所有7个节点状态的JSON对象")
print(f"数据集大小: {DATASET_SIZE}")
print("=" * 80)

# ==============================================================================
# --- 2. 核心逻辑：CA演化器 ---
# ==============================================================================

def get_ca_rule_map(rule_number):
    """根据规则编号生成一个8位的规则映射字典"""
    rule_binary = format(rule_number, '08b')
    return {format(7 - i, '03b'): int(rule_binary[i]) for i in range(8)}

# 预先计算好所有需要的规则映射
RULE_MAPS = {
    "base": get_ca_rule_map(RULE_BASE),
    "other_base": get_ca_rule_map(RULE_OTHER),
    "L": get_ca_rule_map(RULE_L),
    "R": get_ca_rule_map(RULE_R),
    "LL": get_ca_rule_map(RULE_LL),
    "LR": get_ca_rule_map(RULE_LR),
    "RL": get_ca_rule_map(RULE_RL),
    "RR": get_ca_rule_map(RULE_RR),
}

def evolve_ca_n_times(initial_state_list, layers, rule_map):
    """对CA状态进行N次演化"""
    current_state = initial_state_list[:]
    for _ in range(layers):
        width = len(current_state)
        new_state = [0] * width
        state_str = "".join(map(str, current_state))
        padded_state = state_str[-1] + state_str + state_str[0]
        for i in range(width):
            pattern = padded_state[i:i+3]
            new_state[i] = rule_map[pattern]
        current_state = new_state
    return current_state

# ==============================================================================
# --- 3. 单个样本处理与主生成函数 ---
# ==============================================================================

def process_sample():
    """生成一个包含树上所有节点状态的完整数据对。"""

    # --- 节点 0: Root (原始输入) ---
    root_int = random.randint(0, 2 ** INPUT_LEN - 1)
    root_str = format(root_int, f'0{INPUT_LEN}b')
    root_list = [int(bit) for bit in root_str]

    # --- 节点 1: Node_Base ---
    node_base_list = evolve_ca_n_times(root_list, BASE_LAYERS, RULE_MAPS["base"])
    node_other_base_list = evolve_ca_n_times(root_list, BASE_LAYERS, RULE_MAPS["other_base"])

    # --- 节点 2 & 3: Node_L 和 Node_R ---
    node_L_list = evolve_ca_n_times(node_base_list, BRANCH_LAYERS, RULE_MAPS["L"])
    node_R_list = evolve_ca_n_times(node_base_list, BRANCH_LAYERS-3, RULE_MAPS["R"])

    node_OL_list = evolve_ca_n_times(node_other_base_list, BRANCH_LAYERS, RULE_MAPS["L"])

    # --- 节点 4, 5, 6, 7: 四个叶子节点 ---
    leaf_LL_list = evolve_ca_n_times(node_L_list, LEAF_LAYERS, RULE_MAPS["LL"])
    leaf_LR_list = evolve_ca_n_times(node_L_list, LEAF_LAYERS, RULE_MAPS["LR"])
    leaf_RL_list = evolve_ca_n_times(node_R_list, LEAF_LAYERS, RULE_MAPS["RL"])
    leaf_RR_list = evolve_ca_n_times(node_R_list, LEAF_LAYERS, RULE_MAPS["RR"])

    leaf_OLL_list = evolve_ca_n_times(node_OL_list, LEAF_LAYERS, RULE_MAPS["LL"])
    leaf_OLR_list = evolve_ca_n_times(node_OL_list, LEAF_LAYERS, RULE_MAPS["LR"])

    return {
        "input_root": root_str,
        "node_base": node_base_list,
        "node_L": node_L_list,
        "node_R": node_R_list,
        "leaf_LL": leaf_LL_list,
        "leaf_LR": leaf_LR_list,
        "leaf_RL": leaf_RL_list,
        "leaf_RR": leaf_RR_list,
        "leaf_OLL": leaf_OLL_list,
        "leaf_OLR": leaf_OLR_list
    }

def generate_datasets():
    print("\n--- 开始生成层次化CA树数据集 ---")

    # 使用set来避免输入重复，确保数据多样性
    all_inputs = set()
    records = []
    with tqdm(total=DATASET_SIZE, desc="生成样本") as pbar:
        while len(records) < DATASET_SIZE:
            sample = process_sample()
            if sample['input_root'] not in all_inputs:
                all_inputs.add(sample['input_root'])
                records.append(sample)
                pbar.update(1)

    with open(OUTPUT_FILE, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"\n✅ 数据集生成完成！共 {len(records)} 条数据已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample = records[0]
    print(json.dumps(sample, indent=2))
    print("-" * 80)
    for key, value in sample.items():
        print(f"字段 '{key}': 类型 {type(value).__name__}, 长度 {len(value)}")
    print("-" * 80)

# ==============================================================================
# --- 4. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()