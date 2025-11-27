import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 10000  # 数据集大小
NUM_FACT_BITS = 8  # 事实掩码位数
QUERY_BITS = 1  # 查询目标位数

# 定义两套独立规则（二进制编码版本）
# 规则1: (3, 4) -> 5
# 规则2: (1, 2) -> 6
TARGET_RULES = {
    5: (3, 4),  # 查询5需要前提3和4
    6: (1, 2)   # 查询6需要前提1和2
}

# 输入总长度 = 事实掩码 + 查询编码
INPUT_BITS = NUM_FACT_BITS + QUERY_BITS
OUTPUT_BITS = 1  # 二分类输出

TRAIN_FILE = f'deduction_multirule_binary_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'deduction_multirule_binary_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_binary_multirule_sample(query, target_rule, num_fact_bits):
    """生成二进制格式的多规则推理样本"""
    rule_premises = target_rule
    all_positions = list(range(num_fact_bits))
    
    is_positive = random.random() < 0.5
    fact_bits = ['0'] * num_fact_bits
    
    if is_positive:
        # 正样本：设置所有必要条件
        for pos in rule_premises:
            fact_bits[pos] = '1'
        # 添加随机干扰
        extra_pool = list(set(all_positions) - set(rule_premises) - {query})
        for pos in random.sample(extra_pool, random.randint(0, 3)):
            fact_bits[pos] = '1'
        label = 1
    else:
        # 负样本：少一个前提或全缺
        n_present = random.choice([0, 1])
        present = random.sample(rule_premises, n_present)
        for pos in present:
            fact_bits[pos] = '1'
        # 添加随机噪音
        noise_pool = list(set(all_positions) - set(rule_premises) - {query})
        for pos in random.sample(noise_pool, random.randint(1, 3)):
            fact_bits[pos] = '1'
        label = 0
    
    # 查询目标编码：5->0, 6->1
    query_encoded = '0' if query == 5 else '1'
    input_str = ''.join(fact_bits) + query_encoded
    
    return {
        "input": input_str,
        "output": [label]  # 多标签二分类格式
    }

def generate_binary_multirule_dataset(num_samples, num_fact_bits):
    """生成二进制格式的多规则推理数据集"""
    samples = []
    queries = list(TARGET_RULES.keys())
    seen_inputs = set()  # 局部去重集合
    
    while len(samples) < num_samples:
        query = random.choice(queries)
        target_rule = TARGET_RULES[query]
        sample = generate_binary_multirule_sample(query, target_rule, num_fact_bits)
        
        # 去重
        if sample["input"] not in seen_inputs:
            seen_inputs.add(sample["input"])
            samples.append(sample)
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成二进制多规则推理数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 (样本数: {DATASET_SIZE}) ---")
    print(f"事实位数: {NUM_FACT_BITS}, 查询位数: {QUERY_BITS}")
    print(f"输入总长度: {INPUT_BITS} bits")
    print(f"目标规则: {TARGET_RULES}")
    
    samples = generate_binary_multirule_dataset(DATASET_SIZE, NUM_FACT_BITS)
    print(f"生成完毕。共 {len(samples)} 条不重复数据。")
    
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
