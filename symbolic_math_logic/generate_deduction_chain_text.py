import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 数据集大小和配置
DATASET_SIZE = 10000  # 总样本数
DEPTH = 2  # 推理链深度 (1, 2, 或 5)
ATTR_RANGE = (2, 40)  # 属性值范围

# 文件名配置
TRAIN_FILE = f'deduction_depth{DEPTH}_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'deduction_depth{DEPTH}_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_depth1_samples(num_samples, attr_range):
    """生成深度1（单步）推理样本"""
    attrs = list(range(attr_range[0], attr_range[1] + 1))
    samples = []
    
    for _ in range(num_samples):
        label = random.choice([0, 1])
        a, b, c = random.sample(attrs, 3)
        rule = f"({a},{b}|{c})"
        
        if label == 1:
            facts = f"{a}, {b}"
        else:
            facts = f"{random.choice([a, b])}"
        
        query = str(c)
        input_text = f"Facts: {facts}\nRules: {rule}\nQuery: {query}"
        
        samples.append({"input": input_text, "output": [label]})
    
    return samples

def generate_depth2_samples(num_samples, attr_range):
    """生成深度2（两步）推理样本"""
    attrs = list(range(attr_range[0], attr_range[1] + 1))
    samples = []
    
    for _ in range(num_samples):
        label = random.choice([0, 1])
        a, b, d, e = random.sample(attrs, 4)
        c = random.choice([i for i in attrs if i not in [a, b, d, e]])
        
        rule1 = f"({a},{b}|{c})"
        rule2 = f"({c},{d}|{e})"
        rules = f"{rule1}; {rule2}"
        
        if label == 1:
            facts = f"{a}, {b}, {d}"
        else:
            remove = random.choice([[a], [b], [d], [a, d], [b, d]])
            fact_set = {a, b, d} - set(remove)
            facts = ", ".join(str(x) for x in fact_set)
        
        query = str(e)
        input_text = f"Facts: {facts}\nRules: {rules}\nQuery: {query}"
        
        samples.append({"input": input_text, "output": [label]})
    
    return samples

def generate_deduction_dataset(num_samples, depth, attr_range):
    """根据深度生成相应的推理数据集"""
    if depth == 1:
        return generate_depth1_samples(num_samples, attr_range)
    elif depth == 2:
        return generate_depth2_samples(num_samples, attr_range)
    else:
        raise ValueError("Depth must be 1 or 2")

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成推理数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 (深度: {DEPTH}, 样本数: {DATASET_SIZE}) ---")
    print(f"属性范围: {ATTR_RANGE}")
    
    samples = generate_deduction_dataset(DATASET_SIZE, DEPTH, ATTR_RANGE)
    print(f"生成完毕。共 {len(samples)} 条数据。")
    
    random.shuffle(samples)
    train_size = int(len(samples) * 0.9)
    train_data = samples[:train_size]
    eval_data = samples[train_size:]
    
    with open(TRAIN_FILE, 'w') as f:
        for record in train_data:
            f.write(json.dumps(record) + '\n')
    
    with open(EVAL_FILE, 'w') as f:
        for record in eval_data:
            f.write(json.dumps(record) + '\n')
    
    print("\n所有数据集生成完成！")

# ==============================================================================
# --- 4. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
