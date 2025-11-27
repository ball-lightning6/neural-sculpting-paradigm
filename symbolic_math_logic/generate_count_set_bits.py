import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 100000  # 数据集大小
INPUT_BITS = 31        # 输入二进制位数
OUTPUT_BITS = 5        # 输出位数（足够表示最大计数值）
BALANCED = True        # 是否均衡采样（每个汉明重量出现频率相同）

TRAIN_FILE = f'count_set_bits_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'count_set_bits_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def int_to_bits(n, width):
    """将整数转换为固定位数的二进制字符串"""
    return bin(n)[2:].zfill(width)

def generate_count_set_bits_dataset(num_samples, input_bits, output_bits, balanced):
    """生成计算二进制字符串中1的个数的样本"""
    samples = []
    
    if balanced:
        max_weight = input_bits
        per_weight = num_samples // (max_weight + 1)
        for weight in range(max_weight + 1):
            for _ in range(per_weight):
                # 随机生成指定数量1的二进制字符串
                bits = ['0'] * input_bits
                ones_indices = random.sample(range(input_bits), weight)
                for i in ones_indices:
                    bits[i] = '1'
                
                input_str = ''.join(bits)
                output_val = weight  # 1的个数
                output_multilabel = [int(bit) for bit in int_to_bits(output_val, output_bits)]
                
                samples.append({"input": input_str, "output": output_multilabel})
    else:
        # 随机采样模式
        for _ in range(num_samples):
            bits = [random.choice('01') for _ in range(input_bits)]
            input_str = ''.join(bits)
            weight = input_str.count('1')
            output_multilabel = [int(bit) for bit in int_to_bits(weight, output_bits)]
            
            samples.append({"input": input_str, "output": output_multilabel})
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成训练集和验证集"""
    print("\n--- 开始生成数据集 ---")
    print(f"输入位数: {INPUT_BITS}, 输出位数: {OUTPUT_BITS}")
    print(f"均衡采样: {BALANCED}")
    
    # 生成所有样本
    samples = generate_count_set_bits_dataset(DATASET_SIZE, INPUT_BITS, OUTPUT_BITS, BALANCED)
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
