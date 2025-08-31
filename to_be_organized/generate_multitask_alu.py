import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_BITS = 10  # 从8-bit开始，这是一个 manageable 但不平凡的起点

DATASET_SIZE = 200000

TRAIN_FILE = f'neural_alu_{NUM_BITS}bit_all_train.jsonl'
EVAL_FILE = f'neural_alu_{NUM_BITS}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码与输出分区定义 ---
# ==============================================================================
# --- 输入定义 ---
INPUT_LEN = NUM_BITS * 2

# --- 输出分区定义 ---
# 我们为每个操作的输出，都分配固定的“地址段”
# 1. 加法 (ADD): N+1 bits
# 2. 减法 (SUB): N bits (结果不会超过N位)
# 3. 异或 (XOR): N bits
# 4. 与 (AND): N bits
# 5. 或 (OR): N bits
# 6. 比较 (CMP): 1 bit (a > b ? 1 : 0)
OUTPUT_PARTITIONS = {
    'ADD': NUM_BITS + 1,
    'SUB': NUM_BITS,
    'XOR': NUM_BITS,
    'AND': NUM_BITS,
    'OR': NUM_BITS,
    'CMP': 1
}
TOTAL_OUTPUT_BITS = sum(OUTPUT_PARTITIONS.values())
PARTITION_ORDER = ['ADD', 'SUB', 'XOR', 'AND', 'OR', 'CMP']  # 固定输出顺序

print("=" * 70)
print(f"     神经算术逻辑单元 (ALU) - 数据集生成器 (N={NUM_BITS})")
print("=" * 70)
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列 ([a] + [b])")
print(f"输出格式: {TOTAL_OUTPUT_BITS}个多标签二分类，分区如下:")
offset = 0
for op in PARTITION_ORDER:
    size = OUTPUT_PARTITIONS[op]
    print(f"  - {op}: bits {offset} - {offset + size - 1} (长度 {size})")
    offset += size
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_op_pair(num_bits):
    """随机生成一对N-bit的整数"""
    max_val = 2 ** num_bits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    return a, b


def process_pair_alu(a, b, num_bits):
    """
    为一对整数(a, b)生成并行的、多任务的输入和输出。
    """
    # 1. 生成统一的二进制输入字符串
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    input_str = a_bin + b_bin

    # 2. 并行计算所有任务的结果
    results = {
        'ADD': a + b,
        'SUB': abs(a - b),
        'XOR': a ^ b,
        'AND': a & b,
        'OR': a | b,
        'CMP': 1 if a > b else 0
    }

    # 3. 将所有结果编码并拼接成一个大的多标签列表
    final_output_multilabel = []
    for op in PARTITION_ORDER:
        result_val = results[op]
        output_len = OUTPUT_PARTITIONS[op]
        # 将结果转换为固定长度的二进制，然后转为list of int
        output_binary_str = format(result_val, f'0{output_len}b')
        final_output_multilabel.extend([int(bit) for bit in output_binary_str])
    #print('output_len:', len(final_output_multilabel))

    return {"input": input_str, "output": final_output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    seen_pairs = set()

    while len(seen_pairs) < num_samples:
        a, b = generate_op_pair(num_bits)
        if (a, b) in seen_pairs: continue
        seen_pairs.add((a, b))

        records.append(process_pair_alu(a, b, num_bits))

        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的问题对...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")

    # --- 写入文件 ---
    random.shuffle(records)
    train_size = int(len(records) * 1)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in tqdm(data): f.write(json.dumps(record) + '\n')
        # print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        # with open(path[1], 'w') as f:
        #     for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS)