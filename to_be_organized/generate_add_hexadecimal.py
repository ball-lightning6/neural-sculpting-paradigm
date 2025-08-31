import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 我们要进行16-bit整数的加法
NUM_BITS = 16

DATASET_SIZE = 100000  # 生成一个足够大的数据集

# 文件名清晰地反映了实验内容
BINARY_TRAIN_FILE = f'adder_{NUM_BITS}bit_binary_train.jsonl'
BINARY_EVAL_FILE = f'adder_{NUM_BITS}bit_binary_eval.jsonl'

HEX_TRAIN_FILE = f'adder_{NUM_BITS}bit_hex_train.jsonl'
HEX_EVAL_FILE = f'adder_{NUM_BITS}bit_hex_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输出编码：多标签二分类
# 两个16-bit数相加，结果最大是 17-bit
OUTPUT_LEN = NUM_BITS + 1

# 十六进制字符的数量
HEX_CHARS = int(NUM_BITS / 4)  # 16-bit / 4 = 4个十六进制字符/数

print("=" * 60)
print(f"{NUM_BITS}-bit 整数加法对比实验数据集生成器")
print(f"二进制输入长度: {NUM_BITS * 2} chars")
print(f"十六进制输入长度: {HEX_CHARS * 2} chars")
print(f"输出格式: {OUTPUT_LEN} 个多标签二分类")
print("=" * 60)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_addition_pair(num_bits):
    """随机生成一对N-bit的整数"""
    max_val = 2 ** num_bits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    return a, b


def process_pair(a, b, num_bits, output_len):
    """
    为一对整数(a, b)生成两种输入格式和统一的输出格式。
    """
    # 1. 生成二进制输入 (长度 2*N)
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    input_binary = a_bin + b_bin

    # 2. 生成十六进制输入 (长度 N/2)
    # 每个数需要 N/4 个十六进制字符
    hex_chars_per_num = num_bits // 4
    a_hex = format(a, f'0{hex_chars_per_num}x').upper()  # 使用大写 A-F
    b_hex = format(b, f'0{hex_chars_per_num}x').upper()
    input_hex = a_hex + b_hex

    # 3. 生成统一的二进制多标签输出 (长度 N+1)
    sum_val = a + b
    output_binary_str = format(sum_val, f'0{output_len}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {
        "binary": {"input": input_binary, "output": output_multilabel},
        "hex": {"input": input_hex, "output": output_multilabel}
    }


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits):
    print("\n--- 开始生成数据集 ---")

    binary_records = []
    hex_records = []

    # 使用set去重，确保我们生成的加法问题是唯一的
    seen_pairs = set()

    while len(seen_pairs) < num_samples:
        a, b = generate_addition_pair(num_bits)
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        processed_data = process_pair(a, b, num_bits, OUTPUT_LEN)

        binary_records.append(processed_data["binary"])
        hex_records.append(processed_data["hex"])

        if len(seen_pairs) % 10000==0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的加法问题...")

    print(f"生成完毕。共 {len(binary_records)} 条不重复数据。")

    # --- 写入文件 ---
    def write_to_file(records, train_path, eval_path):
        random.shuffle(records)
        train_size = int(len(records) * 1)
        train_data = records[:train_size]
        eval_data = records[train_size:]

        print(f"正在写入 {len(train_data)} 条训练数据到 '{train_path}'...")
        with open(train_path, 'w') as f:
            for record in train_data:
                f.write(json.dumps(record) + '\n')

        print(f"正在写入 {len(eval_data)} 条评估数据到 '{eval_path}'...")
        with open(eval_path, 'w') as f:
            for record in eval_data:
                f.write(json.dumps(record) + '\n')

    print("\n--- 正在处理'二进制'数据集 ---")
    write_to_file(binary_records, BINARY_TRAIN_FILE, BINARY_EVAL_FILE)

    print("\n--- 正在处理'十六进制'数据集 ---")
    write_to_file(hex_records, HEX_TRAIN_FILE, HEX_EVAL_FILE)

    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS)