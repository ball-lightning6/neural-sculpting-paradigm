import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 这个参数必须与你的“模块一”脚本完全一致
NUM_BITS = 10

DATASET_SIZE = 200000

TRAIN_FILE = f'carry_adder_{NUM_BITS}bit_train.jsonl'
EVAL_FILE = f'carry_adder_{NUM_BITS}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# --- 输入编码：无进位计数向量 ---
# 这是模块一的输出，也是模块二的输入
NUM_INPUT_COUNTERS = NUM_BITS * 2
# 每个计数器需要多少bit来表示？最大值是N
BITS_PER_COUNTER = math.ceil(math.log2(NUM_BITS + 1))
# 总输入比特数
TOTAL_INPUT_BITS = NUM_INPUT_COUNTERS * BITS_PER_COUNTER

# --- 输出编码：最终的标准二进制结果 ---
OUTPUT_BITS = NUM_BITS * 2

print("=" * 70)
print(f"模块二：{NUM_BITS}-bit '神经进位加法器' - 数据集生成器")
print("=" * 70)
print(f"输入格式: {NUM_INPUT_COUNTERS}个'计数器', 每个{BITS_PER_COUNTER} bits, 共{TOTAL_INPUT_BITS} bits")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (最终带进位结果)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_multiplication_pair(num_bits):
    """随机生成一对N-bit的整数"""
    max_val = 2 ** num_bits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    return a, b


def process_pair_for_module2(a, b, num_bits):
    """
    为一对整数(a, b)生成模块二所需的输入和输出。
    """
    # 1. 计算“无进位计数向量”，这将作为模块二的输入
    counters = [0] * (2 * num_bits)
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')

    for i in range(num_bits):
        if b_bin[num_bits - 1 - i]=='1':
            for j in range(num_bits):
                if a_bin[num_bits - 1 - j]=='1':
                    counters[i + j] += 1

    # 将计数器列表转换为输入二进制字符串
    input_str = ""
    for count in counters:
        input_str += format(count, f'0{BITS_PER_COUNTER}b')

    # 2. 计算最终的标准二进制结果，这将作为模块二的输出
    product = a * b
    output_binary_str = format(product, f'0{2 * num_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    seen_pairs = set()

    while len(seen_pairs) < num_samples:
        a, b = generate_multiplication_pair(num_bits)
        if (a, b) in seen_pairs or (b, a) in seen_pairs: continue
        seen_pairs.add((a, b))

        records.append(process_pair_for_module2(a, b, num_bits))

        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复问题...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")

    # --- 写入文件 ---
    random.shuffle(records)
    train_size = int(len(records) *1)# 0.9)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in data:
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
    generate_datasets(DATASET_SIZE, NUM_BITS)