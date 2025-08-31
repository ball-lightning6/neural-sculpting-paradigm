import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 让我们从一个稍小，但依然很有挑战性的位数开始
NUM_BITS = 10

DATASET_SIZE = 200000

TRAIN_FILE = f'multiplier_{NUM_BITS}bit_carryless_train.jsonl'
EVAL_FILE = f'multiplier_{NUM_BITS}bit_carryless_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 (全新输出格式) ---
# ==============================================================================
# 输入是 2 * N 位 0/1 的字符
INPUT_LEN = NUM_BITS * 2

# --- 输出编码：无进位多计数器 ---
# 输出的总“位数”（或叫“计数器数”）是 2*N
NUM_OUTPUT_COUNTERS = NUM_BITS * 2
# 每个计数器需要多少bit来表示？
# 在N-bit乘法中，任何一个输出位，最多是N个1相加，所以最大值是N
BITS_PER_COUNTER = math.ceil(math.log2(NUM_BITS + 1))
# 总输出比特数
TOTAL_OUTPUT_BITS = NUM_OUTPUT_COUNTERS * BITS_PER_COUNTER

print("=" * 70)
print(f"{NUM_BITS}-bit '无进位'乘法 - 数据集生成器")
print("=" * 70)
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {NUM_OUTPUT_COUNTERS}个'计数器', 每个计数器{BITS_PER_COUNTER} bits")
print(f"总输出比特数: {TOTAL_OUTPUT_BITS}")
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


def process_pair_carryless(a, b, num_bits):
    """
    为一对整数(a, b)生成二进制输入和“无进位”的输出。
    """
    # 1. 生成二进制输入字符串
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    input_str = a_bin + b_bin

    # 2. 生成“无进位”输出
    # 这模拟了手算乘法的过程，但只记录每一列的1的个数，不进位

    # 初始化一个长度为 2*N 的计数器列表
    counters = [0] * (2 * num_bits)

    # 逐位相乘，并将结果累加到对应的计数器上
    for i in range(num_bits):
        if b_bin[num_bits - 1 - i]=='1':  # 如果b的第i位是1
            for j in range(num_bits):
                if a_bin[num_bits - 1 - j]=='1':  # 如果a的第j位是1
                    # 那么在最终结果的第 i+j 位，就多了一个1
                    counters[i + j] += 1

    # 将每个计数器的值，转换为固定位数的二进制字符串
    output_str = ""
    for count in counters:
        output_str += format(count, f'0{BITS_PER_COUNTER}b')

    # 转换为多标签二分类的列表
    output_multilabel = [int(bit) for bit in output_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 (与之前类似) ---
# ==============================================================================
def generate_datasets(num_samples, num_bits):
    print("\n--- 开始生成数据集 ---")

    records, seen_pairs = [], set()
    while len(seen_pairs) < num_samples:
        a, b = generate_multiplication_pair(num_bits)
        if (a, b) in seen_pairs or (b, a) in seen_pairs: continue
        seen_pairs.add((a, b))
        records.append(process_pair_carryless(a, b, num_bits))
        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的乘法问题...")
    print(f"生成完毕。共 {len(records)} 条不重复数据。")

    # ... (写入文件的逻辑，与上一版完全相同，此处省略)
    random.shuffle(records)
    train_size = int(len(records) * 1)#0.9)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in data: f.write(json.dumps(record) + '\n')
        print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        with open(path[1], 'w') as f:
            for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS)