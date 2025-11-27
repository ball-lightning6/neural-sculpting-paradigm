import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 5429  # 数据集大小（使用所有可能的m值）
BITS = 13  # 编码位数（足够表示n-1）

# RSA公钥参数
E = 17  # 加密指数
eN = 5429  # 模数

# 文件名配置
TRAIN_FILE = f'rsa_encrypt_{BITS}bit_train.jsonl'
EVAL_FILE = f'rsa_encrypt_{BITS}bit_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def int_to_bin_multilabel(x, bits):
    """将整数转换为二进制多标签格式"""
    return [int(bit) for bit in format(x, f'0{bits}b')]

def rsa_encrypt(m, e, n):
    """RSA加密：c = m^e mod n"""
    return pow(m, e, n)

def generate_rsa_dataset(num_samples, bits, e, n):
    """生成RSA加密数据集"""
    # 由于RSA在固定公钥下，输入空间是0到n-1，我们使用全部
    actual_size = min(num_samples, n)
    
    samples = []
    for m in range(actual_size):
        c = rsa_encrypt(m, e, n)
        
        # 输入和输出都转为多标签二分类格式
        input_multilabel = int_to_bin_multilabel(m, bits)
        output_multilabel = int_to_bin_multilabel(c, bits)
        
        samples.append({
            "input": input_multilabel,
            "output": output_multilabel
        })
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成RSA加密数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 ({DATASET_SIZE} 条样本) ---")
    print(f"RSA参数: e={E}, n={N}")
    print(f"编码位数: {BITS}")
    
    samples = generate_rsa_dataset(DATASET_SIZE, BITS, E, N)
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
