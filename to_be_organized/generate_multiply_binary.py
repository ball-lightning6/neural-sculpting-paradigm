import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 12-bit二进制整数
NUM_BITS = 10

DATASET_SIZE = 200000

TRAIN_FILE = f'multiplier_{NUM_BITS}bit_binary_train.jsonl'
EVAL_FILE = f'multiplier_{NUM_BITS}bit_binary_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 2 * N 位 0/1 的字符
#INPUT_LEN = NUM_BITS +4
INPUT_LEN = NUM_BITS *2

# 输出是结果的二进制表示
# 两个N-bit数相乘，结果最大是 2*N bit
#OUTPUT_BITS = NUM_BITS +4
OUTPUT_BITS = NUM_BITS *2

print("=" * 70)
print(f"{NUM_BITS}-bit 二进制整数乘法 - 数据集生成器")
print("=" * 70)
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_multiplication_pair(num_bits):
    """随机生成一对N-bit的整数"""
    max_val = 2 ** num_bits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    # b = random.randint(0, 2**4-1)
    return a, b


def process_pair(a, b, num_bits, output_bits):
    """
    为一对整数(a, b)生成二进制输入和二进制输出。
    """
    # 1. 生成二进制输入字符串
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    # b_bin = format(b, f'04b')
    input_str = a_bin + b_bin

    # 2. 生成二进制多标签输出
    product = a * b
    output_binary_str = format(product, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]#[-NUM_BITS:]

    return {"input": input_str, "output": output_multilabel}

# --- 写入文件 ---
def write_to_file(records, train_path, eval_path):
    random.shuffle(records)
    train_size = int(len(records) *1  )
    train_data, eval_data = records[:train_size], records[train_size:]

    print(f"\n正在写入 {len(train_data)} 条训练数据到 '{train_path}'...")
    with open(train_path, 'w') as f:
        for record in train_data:
            f.write(json.dumps(record) + '\n')

    print(f"正在写入 {len(eval_data)} 条评估数据到 '{eval_path}'...")
    with open(eval_path, 'w') as f:
        for record in eval_data:
            f.write(json.dumps(record) + '\n')
# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    seen_pairs = set()

    while len(seen_pairs) < num_samples:
        a, b = generate_multiplication_pair(num_bits)
        if (a, b) in seen_pairs or (b, a) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        records.append(process_pair(a, b, num_bits, output_bits))

        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的乘法问题...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")



    write_to_file(records, TRAIN_FILE, EVAL_FILE)

    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS, OUTPUT_BITS)