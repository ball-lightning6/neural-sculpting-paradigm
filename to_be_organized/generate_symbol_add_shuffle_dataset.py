import json
import random
import string
import math
import time

# ==============================================================================
# --- 1. 核心实验参数配置 ---
# ==============================================================================
NUM_BITS = 8
BASE = 16
SHUFFLE_SEMANTICS = True
SHUFFLE_POSITIONS = False#True  # True

DATASET_SIZE = 50

# --- 文件名和路径 ---
exp_name = f'adder_{NUM_BITS}bit_base{BASE}'
if SHUFFLE_SEMANTICS: exp_name += '_sem_shuffled'
if SHUFFLE_POSITIONS: exp_name += '_pos_shuffled'
TRAIN_FILE = f'{exp_name}_train.jsonl'
EVAL_FILE = f'{exp_name}_eval.jsonl'

# ==============================================================================
# --- 2. 编码与排列定义 (自动生成) ---
# ==============================================================================

# 创建通用字符集
PRINTABLE_CHARS = list(set([chr(i) for i in range(33, 127)]) - set(r'"\?'))

if BASE > len(PRINTABLE_CHARS):
    raise ValueError(f"错误：最大支持进制为 {len(PRINTABLE_CHARS)}，当前设置为 {BASE}")

if SHUFFLE_SEMANTICS:
    random.shuffle(PRINTABLE_CHARS)
else:
    PRINTABLE_CHARS = list('0123456789abcdef')

STANDARD_CHAR_MAP = {i: PRINTABLE_CHARS[i] for i in range(BASE)}
shuffled_chars = list(STANDARD_CHAR_MAP.values())
SEMANTIC_MAP = STANDARD_CHAR_MAP

# 2. 结构层 (Positional Layer)
chars_per_num = NUM_BITS  # math.ceil(NUM_BITS / math.log2(BASE))
INPUT_LEN = chars_per_num * 2
POSITION_SHUFFLE_MAP = None  # 默认为不洗牌
if SHUFFLE_POSITIONS:
    position_map = list(range(INPUT_LEN))
    random.shuffle(position_map)
    POSITION_SHUFFLE_MAP = {from_idx: to_idx for from_idx, to_idx in enumerate(position_map)}

# 3. 输出层 (Output Layer)
OUTPUT_LEN = NUM_BITS * int(math.log2(BASE)) + 1

# --- 打印实验配置 ---
print("=" * 70)
print("     终极符号学习能力探针 - 数据集生成器")
print("=" * 70)
print(f"计算任务: {NUM_BITS}-bit 整数加法")
print(f"输入进制: {BASE}-based")
print(f"语义洗牌 (Shuffle Semantics): {'是' if SHUFFLE_SEMANTICS else '否'}")
print(f"位置洗牌 (Shuffle Positions): {'是' if SHUFFLE_POSITIONS else '否'}")
print("-" * 70)
print(f"输入字符长度: {INPUT_LEN}")
print(f"输出比特长度: {OUTPUT_LEN}")
print(f"使用的字符集 (前10个): {list(STANDARD_CHAR_MAP.values())[:10]}...")

# --- 关键修正：修复打印部分的语法和逻辑错误 ---
if SHUFFLE_SEMANTICS:
    print("语义映射 (数值 -> 字符):")
    # 使用正确的语法打印字典
    print({k: v for k, v in list(SEMANTIC_MAP.items())[:5]}, "...")
if POSITION_SHUFFLE_MAP:
    print("位置洗牌映射 (原位置 -> 新位置):")
    # 使用正确的语法并确保此代码块在if内部
    print({k: v for k, v in list(POSITION_SHUFFLE_MAP.items())[:5]}, "...")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================

def number_to_base_str(n, base, chars_per_num, semantic_map):
    if n==0:
        return semantic_map[0] * chars_per_num
    base_n_digits = []
    temp_n = n
    while temp_n > 0:
        base_n_digits.append(temp_n % base)
        temp_n //= base
    while len(base_n_digits) < chars_per_num:
        base_n_digits.append(0)
    return "".join([semantic_map[digit] for digit in reversed(base_n_digits)])


def shuffle_positions(input_str, pos_map):
    if not pos_map:
        return input_str
    input_list = list(input_str)
    shuffled_list = [''] * len(input_list)
    for i in range(len(input_list)):
        shuffled_list[pos_map[i]] = input_list[i]
    return "".join(shuffled_list)


def process_pair(a, b, num_bits, base, output_len):
    a_str = number_to_base_str(a, base, chars_per_num, SEMANTIC_MAP)
    b_str = number_to_base_str(b, base, chars_per_num, SEMANTIC_MAP)
    ordered_input = a_str + b_str
    final_input = shuffle_positions(ordered_input, POSITION_SHUFFLE_MAP)
    sum_val = a + b
    output_binary_str = format(sum_val, f'0{output_len}b')
    output_multilabel = [int(bit) for bit in output_binary_str]
    return {"input": final_input, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits, base, output_len):
    print("\n--- 开始生成数据集 ---")
    records, seen_pairs = [], set()
    while len(seen_pairs) < num_samples:
        max_val = base ** num_bits - 1
        a, b = random.randint(0, max_val), random.randint(0, max_val)
        if (a, b) in seen_pairs: continue
        seen_pairs.add((a, b))
        records.append(process_pair(a, b, num_bits, base, output_len))
        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的加法问题...")
    print(f"生成完毕。共 {len(records)} 条不重复数据。")

    random.shuffle(records)
    train_size = int(len(records) * 1)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}数据到 '{path}'...")
        with open(path, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
                # print(record)

    write_to_file(train_data, TRAIN_FILE, "训练")
    write_to_file(eval_data, EVAL_FILE, "评估")

    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS, BASE, OUTPUT_LEN)