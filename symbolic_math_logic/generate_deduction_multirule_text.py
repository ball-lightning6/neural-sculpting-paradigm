import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 10000  # 数据集大小
NUM_ATTRIBUTES = 10  # 属性编号范围 0~9

# 定义两套独立规则
# 规则1: (3, 4) -> 5
# 规则2: (3, 6) -> 7
TARGET_RULES = {
    5: (3, 4),  # 查询5需要前提3和4
    7: (3, 6)   # 查询7需要前提3和6
}

TRAIN_FILE = f'deduction_multirule_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'deduction_multirule_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_multirule_sample(query, target_rule, num_attributes):
    """生成单个多规则推理样本"""
    target_label = query
    rule_premises = target_rule
    
    is_positive = random.random() < 0.5
    
    if is_positive:
        # 正样本：提供所有必要条件
        facts = list(rule_premises)
        # 添加干扰属性（不包括目标标签和规则前提）
        noise_pool = list(set(range(num_attributes)) - set(rule_premises) - {target_label})
        noise = random.sample(noise_pool, random.randint(0, 2))
        facts += noise
        label = 1
    else:
        # 负样本：缺少至少一个必要条件
        noise_pool = list(set(range(num_attributes)) - set(rule_premises) - {target_label})
        facts = random.sample(noise_pool, random.randint(1, 3))
        # 50%概率只包含一条前提（更难）
        if random.random() < 0.5:
            facts.append(random.choice(rule_premises))
        label = 0
    
    random.shuffle(facts)
    fact_str = ", ".join(str(f) for f in facts)
    input_text = f"Facts: {fact_str}\nQuery: {query}"
    
    return {
        "input": input_text,
        "output": [label]  # 多标签二分类格式
    }

def generate_multirule_dataset(num_samples, num_attributes):
    """生成多规则推理数据集"""
    samples = []
    queries = list(TARGET_RULES.keys())
    
    for _ in range(num_samples):
        # 随机选择一个查询目标
        query = random.choice(queries)
        target_rule = TARGET_RULES[query]
        sample = generate_multirule_sample(query, target_rule, num_attributes)
        samples.append(sample)
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成多规则推理数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 (样本数: {DATASET_SIZE}) ---")
    print(f"属性范围: 0~{NUM_ATTRIBUTES-1}")
    print(f"目标规则: {TARGET_RULES}")
    
    samples = generate_multirule_dataset(DATASET_SIZE, NUM_ATTRIBUTES)
    print(f"生成完毕。共 {len(samples)} 条数据。")
    
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
