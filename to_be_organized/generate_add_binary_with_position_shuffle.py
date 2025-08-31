import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 你可以在这里轻松调整，比如改成4-bit或16-bit
NUM_BITS = 32

DATASET_SIZE = 500000  # 生成一个足够大的数据集

# 文件名将清晰地反映实验内容
ORDERED_TRAIN_FILE = f'adder_{NUM_BITS}bit_ordered_train.jsonl'
ORDERED_EVAL_FILE = f'adder_{NUM_BITS}bit_ordered_eval.jsonl'

SHUFFLED_TRAIN_FILE = f'adder_{NUM_BITS}bit_shuffled_train.jsonl'
SHUFFLED_EVAL_FILE = f'adder_{NUM_BITS}bit_shuffled_eval.jsonl'

# ==============================================================================
# --- 2. 编码与排列定义 ---
# ==============================================================================
INPUT_BITS = NUM_BITS * 2
# 两个N-bit整数相加，结果最大可能是 N+1 bit。我们给足空间。
OUTPUT_BITS = NUM_BITS + 1

# --- 生成并固定一个随机排列 (Shuffle Permutation) ---
# 这非常重要：所有被打乱的样本，都必须遵循同一个打乱顺序
# 这样模型才有规律可循
ORIGINAL_INDICES = list(range(INPUT_BITS))
SHUFFLE_MAP = list(range(INPUT_BITS))
random.shuffle(SHUFFLE_MAP)

# 打印出这个固定的打乱映射，以便记录和复现
print("=" * 60)
print(f"N-bit 整数加法对比实验数据集生成器 (N={NUM_BITS})")
print(f"输入长度: {INPUT_BITS} bits, 输出长度: {OUTPUT_BITS} bits")
print("-" * 60)
print("固定的随机打乱映射 (SHUFFLE_MAP):")
# 为了方便阅读，我们打印出 from_idx -> to_idx 的形式
shuffle_dict = {from_idx: to_idx for from_idx, to_idx in enumerate(SHUFFLE_MAP)}
print(shuffle_dict)
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


def numbers_to_binary(a, b, num_bits, output_bits):
    """将两个数转换为输入和输出的二进制字符串"""
    # 将数字格式化为固定位数的二进制字符串
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')

    # 输入：直接拼接
    input_ordered = a_bin + b_bin

    # 输出：计算和并格式化
    sum_val = a + b
    output_bin = format(sum_val, f'0{output_bits}b')

    return input_ordered, output_bin


def shuffle_input(input_ordered, shuffle_map):
    """根据固定的映射打乱输入字符串"""
    input_list = list(input_ordered)
    shuffled_list = [''] * len(input_list)
    for i in range(len(input_list)):
        shuffled_list[shuffle_map[i]] = input_list[i]
    return "".join(shuffled_list)


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits):
    print("\n--- 开始生成数据集 ---")

    ordered_records = []
    shuffled_records = []

    # 使用set去重，确保我们生成的加法问题是唯一的
    seen_pairs = set()

    while len(seen_pairs) < num_samples:
        a, b = generate_addition_pair(num_bits)
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        input_ordered, output_bin = numbers_to_binary(a, b, num_bits, OUTPUT_BITS)
        input_shuffled = shuffle_input(input_ordered, SHUFFLE_MAP)

        ordered_records.append({"input": input_ordered, "output": output_bin})
        shuffled_records.append({"input": input_shuffled, "output": output_bin})

    print(f"已生成 {len(ordered_records)} 条不重复的加法问题。")

    # --- 写入文件 ---
    def write_to_file(records, train_path, eval_path):
        random.shuffle(records)
        train_size = int(len(records) * 1.)
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

    print("\n--- 正在处理'有序'数据集 ---")
    write_to_file(ordered_records, ORDERED_TRAIN_FILE, ORDERED_EVAL_FILE)

    print("\n--- 正在处理'打乱'数据集 ---")
    write_to_file(shuffled_records, SHUFFLED_TRAIN_FILE, SHUFFLED_EVAL_FILE)

    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS)