import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 100000  # 数据集大小
NUM_OPERANDS = 3  # 操作数个数（目前支持3个）
OPERAND_BITS = 8  # 每个操作数的位数（0-255）
NUM_OPS = 2  # 运算符个数（目前支持2个）
OP_ENCODE_BITS = 1  # 运算符编码位数（+/-用1位足够）

# 输入长度 = 操作数 + 运算符编码
INPUT_BITS = (NUM_OPERANDS * OPERAND_BITS) + (NUM_OPS * OP_ENCODE_BITS)
# 输出长度 = 中间结果 + 最终结果
OUTPUT_BITS = OPERAND_BITS * 2

TRAIN_FILE = f'explainable_calc_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'explainable_calc_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def to_bin8(n):
    """将整数转换为8位二进制字符串"""
    return format(n, '08b')

def safe_op(a, b, op):
    """安全执行运算，检查越界"""
    if op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '*':
        result = a * b
    else:
        raise ValueError("未知运算符")
    
    if 0 <= result <= 255:
        return result
    else:
        return None

def encode_op(op):
    """编码运算符（+ -> 0, - -> 1）"""
    return '0' if op == '+' else '1'

def generate_explainable_sample():
    """生成单个可解释的两步计算样本"""
    while True:
        # 生成三个8位数
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        c = random.randint(0, 255)
        
        # 生成两个运算符（目前只用+/-）
        op1 = random.choice(['+', '-'])#, '*'])
        op2 = random.choice(['+', '-'])#, '*'])
        
        # 第一步计算
        step1 = safe_op(a, b, op1)
        if step1 is None:
            continue
        
        # 第二步计算
        step2 = safe_op(step1, c, op2)
        if step2 is None:
            continue
        
        # 构建输入：a(8) + op1(1) + b(8) + op2(1) + c(8)
        input_bits = to_bin8(a) + encode_op(op1) + to_bin8(b) + encode_op(op2) + to_bin8(c)
        
        # 构建输出：step1(8) + step2(8) 的多标签二分类格式
        output_multilabel = [int(bit) for bit in (to_bin8(step1) + to_bin8(step2))]
        
        return {
            "input": input_bits,
            "output": output_multilabel  # 多标签二分类格式
        }

def generate_explainable_dataset(num_samples):
    """生成可解释计算数据集"""
    samples = []
    seen_inputs = set()
    
    while len(samples) < num_samples:
        sample = generate_explainable_sample()
        if sample["input"] not in seen_inputs:
            seen_inputs.add(sample["input"])
            samples.append(sample)
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成可解释计算数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 ({DATASET_SIZE} 条样本) ---")
    print(f"操作数: {NUM_OPERANDS}, 每个位数: {OPERAND_BITS}")
    print(f"运算符: {NUM_OPS}, 每个编码: {OP_ENCODE_BITS} bit")
    print(f"输入长度: {INPUT_BITS} bits")
    print(f"输出长度: {OUTPUT_BITS} bits (中间结果[{OPERAND_BITS}] + 最终结果[{OPERAND_BITS}])")
    
    samples = generate_explainable_dataset(DATASET_SIZE)
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
