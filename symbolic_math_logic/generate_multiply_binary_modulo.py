import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 8192  # 数据集大小
BITS = 8  # 操作数位数

# 文件名配置
TRAIN_FILE = f'mul_mod_{BITS}bit_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'mul_mod_{BITS}bit_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def int_to_bitstring(n, bits):
    """将整数转换为固定位数的二进制字符串"""
    return format(n, f'0{bits}b')

def int_to_multilabel(n, bits):
    """将整数转换为多标签二分类格式"""
    return [int(bit) for bit in format(n, f'0{bits}b')]

def generate_modulo_mul_sample(bits):
    """生成单个模乘样本"""
    max_val = 2 ** bits
    a = random.randint(0, max_val - 1)
    b = random.randint(0, max_val - 1)
    product = (a * b) % max_val  # 截断乘法（模乘）
    
    return {
        "input": int_to_bitstring(a, bits) + int_to_bitstring(b, bits),
        "output": int_to_multilabel(product, bits)
    }

def generate_modulo_mul_dataset(num_samples, bits):
    """生成模乘数据集"""
    samples = []
    seen_inputs = set()
    
    while len(samples) < num_samples:
        sample = generate_modulo_mul_sample(bits)
        if sample["input"] not in seen_inputs:
            seen_inputs.add(sample["input"])
            samples.append(sample)
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成模乘数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 ({DATASET_SIZE} 条样本) ---")
    print(f"操作数位数: {BITS}, 模数: 2^{BITS} = {2 ** BITS}")
    print(f"输入格式: 2x{BITS} bits (a + b)")
    print(f"输出格式: {BITS} bits 多标签二分类 (a*b mod 2^{BITS})")
    
    samples = generate_modulo_mul_dataset(DATASET_SIZE, BITS)
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
