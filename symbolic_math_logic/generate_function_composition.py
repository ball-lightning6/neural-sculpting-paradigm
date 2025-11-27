import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 100000  # 数据集大小
NUM_FUNCS = 4  # 函数调用序列长度
FUNC_ENCODE_BITS = 2  # 每个函数的编码位数
VALUE_BITS = 16  # 数值编码位数（0-65535）

# 函数列表与实现
FUNC_LIST = ["double", "increment", "square", "decrement"]
FUNC_IMPL = {
    "double": lambda x: 2 * x,
    "increment": lambda x: x + 1,
    "square": lambda x: x * x,
    "decrement": lambda x: x - 1
}

# 输入总长度 = 函数指令 + 初始值
INPUT_BITS = (NUM_FUNCS * FUNC_ENCODE_BITS) + VALUE_BITS
OUTPUT_BITS = VALUE_BITS  # 输出与初始值同长度

TRAIN_FILE = f'func_compose_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'func_compose_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def func_to_bits(func_names):
    """将函数序列编码为二进制字符串"""
    return ''.join(format(FUNC_LIST.index(f), f'0{FUNC_ENCODE_BITS}b') for f in func_names)

def apply_funcs_strict(func_names, x):
    """严格应用函数序列，检查中间结果是否越界"""
    for f in func_names:
        x = FUNC_IMPL[f](x)
        if x < 0 or x > 65535:
            return None  # 一旦中间越界，返回None
    return x

def generate_example():
    """生成单个有效样本"""
    while True:
        func_seq = random.choices(FUNC_LIST, k=NUM_FUNCS)
        x = random.randint(0, 65535)
        y = apply_funcs_strict(func_seq, x)
        if y is not None:
            input_bits = func_to_bits(func_seq) + format(x, f'0{VALUE_BITS}b')
            output_multilabel = [int(bit) for bit in format(y, f'0{VALUE_BITS}b')]
            return {
                "input": input_bits,
                "output": output_multilabel  # 多标签二分类格式
            }

def generate_function_dataset(num_samples):
    """生成函数组合数据集"""
    samples = []
    seen_inputs = set()
    
    while len(samples) < num_samples:
        example = generate_example()
        if example["input"] not in seen_inputs:
            seen_inputs.add(example["input"])
            samples.append(example)
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成函数组合数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 ({DATASET_SIZE} 条样本) ---")
    print(f"函数序列长度: {NUM_FUNCS}, 每个函数编码: {FUNC_ENCODE_BITS} bits")
    print(f"初始值位数: {VALUE_BITS} bits")
    print(f"输入总长度: {INPUT_BITS} bits")
    print(f"输出长度: {OUTPUT_BITS} bits")
    
    samples = generate_function_dataset(DATASET_SIZE)
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
