import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 100000  # 数据集大小
DEPTH = 5  # 推理链深度
NUM_ATTRS = 16  # 属性总数

# 编码长度定义
FACT_MASK_BITS = NUM_ATTRS  # 事实掩码位数
QUERY_ENCODE_BITS = 4  # 查询目标编码位数（足够编码0-15）
INPUT_BITS = FACT_MASK_BITS + QUERY_ENCODE_BITS
OUTPUT_BITS = 1  # 二分类输出

TRAIN_FILE = f'deduction_fixed_depth_{DEPTH}_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'deduction_fixed_depth_{DEPTH}_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_single_sample(depth, num_attrs):
    """生成单个固定深度推理样本"""
    attr_bits = [0] * num_attrs
    rules = {}
    used_targets = set()
    used_sources = set()
    
    # 生成推理链
    chain = []
    available_attrs = list(range(num_attrs))
    
    while len(chain) < depth:
        remaining = list(set(available_attrs) - used_targets)
        if len(remaining) < 3:
            break
        a1, a2 = random.sample(remaining, 2)
        candidate_targets = list(set(available_attrs) - set([a1, a2]) - used_targets)
        if not candidate_targets:
            break
        target = random.choice(candidate_targets)
        rules[target] = (a1, a2)
        chain.append(target)
        used_targets.add(target)
        used_sources.update([a1, a2])
    
    if not chain:
        raise ValueError("Failed to generate valid rule chain")
    
    target_attr = chain[-1]  # 最终目标
    
    # 反向链：找出所有必要事实
    facts = set()
    def backchain(attr):
        if attr in rules:
            a1, a2 = rules[attr]
            backchain(a1)
            backchain(a2)
        facts.add(attr)
    backchain(target_attr)
    
    # 设置事实掩码
    for f in facts:
        attr_bits[f] = 1
    
    # 正样本
    input_bits = ''.join(str(b) for b in attr_bits)
    query_bits = format(target_attr, '04b')
    pos_input = input_bits + query_bits
    pos_output = [1]
    
    # 负样本：查询无效目标
    non_facts = list(set(range(num_attrs)) - facts)
    if not non_facts:
        # 如果没有负样本，复制正样本但标记为负（特殊情况）
        neg_input, neg_output = pos_input, [0]
    else:
        neg_query = random.choice(non_facts)
        neg_input = input_bits + format(neg_query, '04b')
        neg_output = [0]
    
    return (pos_input, pos_output), (neg_input, neg_output)

def generate_fixed_depth_dataset(num_samples, depth, num_attrs):
    """生成固定深度推理数据集"""
    samples = []
    seen_inputs = set()
    count = 0
    
    while len(samples) < num_samples:
        try:
            pos, neg = generate_single_sample(depth, num_attrs)
            
            # 去重并添加正样本
            if pos[0] not in seen_inputs:
                seen_inputs.add(pos[0])
                samples.append({"input": pos[0], "output": pos[1]})
            
            # 去重并添加负样本
            if neg[0] not in seen_inputs:
                seen_inputs.add(neg[0])
                samples.append({"input": neg[0], "output": neg[1]})
            
        except ValueError:
            continue
    
    return samples[:num_samples]  # 精确返回所需数量

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成固定深度推理数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 (深度: {DEPTH}, 样本数: {DATASET_SIZE}) ---")
    print(f"属性总数: {NUM_ATTRS}")
    print(f"输入长度: {INPUT_BITS} bits (事实掩码[{FACT_MASK_BITS}] + 查询编码[{QUERY_ENCODE_BITS}])")
    
    samples = generate_fixed_depth_dataset(DATASET_SIZE, DEPTH, NUM_ATTRS)
    print(f"生成完毕。共 {len(samples)} 条数据（含正负样本）。")
    
    # 打乱并分割为训练集和验证集
    random.shuffle(samples)
    train_size = int(len(samples) * 0.9)
    train_data = samples[:train_size]
    eval_data = samples[train_size:]
    
    # 写入文件
    print(f"\n正在写入 {len(train_data)} 条训练数据到 '{TRAIN_FILE}'...")
    with open(TRAIN_FILE, 'w') as f:
        for record in train_data:
            f.write(json.dumps(record) + '\n')
    
    print(f"正在写入 {len(eval_data)} 条评估数据到 '{EVAL_FILE}'...")
    with open(EVAL_FILE, 'w') as f:
        for record in eval_data:
            f.write(json.dumps(record) + '\n')
    
    print("\n所有数据集生成完成！")

# ==============================================================================
# --- 4. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
