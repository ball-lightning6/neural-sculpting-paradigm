import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 10位10进制整数
NUM_DIGITS = 3

DATASET_SIZE = 200000

TRAIN_FILE = f'multiplier_{NUM_DIGITS}digit_train.jsonl'
EVAL_FILE = f'multiplier_{NUM_DIGITS}digit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 2 * N 位 0-9 的字符
INPUT_LEN = NUM_DIGITS * 2

# 输出是结果的二进制表示
# 计算最大可能值
MAX_OPERAND = 10 ** NUM_DIGITS - 1
MAX_PRODUCT = MAX_OPERAND ** 2
# 计算表示这个最大值需要多少个二进制位
OUTPUT_BITS = MAX_PRODUCT.bit_length()

print("=" * 70)
print(f"{NUM_DIGITS}位 十进制整数乘法 - 数据集生成器")
print("=" * 70)
print(f"输入格式: {INPUT_LEN}个'0'-'9'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类")
print(f"最大操作数: {MAX_OPERAND}")
print(f"最大乘积:   {MAX_PRODUCT}")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_multiplication_pair(num_digits):
    """随机生成一对N位10进制的整数"""
    max_val = 10 ** num_digits - 1
    # 为了增加多样性，我们可以让位数也随机
    a_digits = random.randint(1, num_digits)
    b_digits = random.randint(1, num_digits)
    a = random.randint(0, 10 ** a_digits - 1)
    b = random.randint(0, 10 ** b_digits - 1)
    return a, b


def process_pair(a, b, num_digits, output_bits):
    """
    为一对整数(a, b)生成输入和输出。
    """
    # 1. 生成输入字符串，左侧补零
    a_str = str(a).zfill(num_digits)
    b_str = str(b).zfill(num_digits)
    input_str = a_str + b_str

    # 2. 生成二进制多标签输出
    product = a * b
    output_binary_str = format(product, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_digits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    seen_pairs = set()

    # 使用set去重，确保我们生成的乘法问题是唯一的
    while len(seen_pairs) < num_samples:
        a, b = generate_multiplication_pair(num_digits)
        # 为了对称性，可以交换a,b，但对于乘法意义不大
        if (a, b) in seen_pairs or (b, a) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        records.append(process_pair(a, b, num_digits, output_bits))

        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的乘法问题...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")

    # --- 写入文件 ---
    def write_to_file(records, path, name):
        random.shuffle(records)
        train_size = int(len(records) * 1)#0.9)
        train_data, eval_data = records[:train_size], records[train_size:]

        print(f"\n正在写入 {len(train_data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in train_data:
                f.write(json.dumps(record) + '\n')

        print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        with open(path[1], 'w') as f:
            for record in eval_data:
                f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")

    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_DIGITS, OUTPUT_BITS)