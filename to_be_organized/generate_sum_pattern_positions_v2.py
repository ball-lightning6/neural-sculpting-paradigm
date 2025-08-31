import json
import random
import math
from collections import defaultdict

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 每个“子模式”的位数
BITS_PER_PATTERN = 1  # p值
# 输入中包含多少个这样的模式（数）
NUM_PATTERNS = 30  # q值
# 最终，输入的总长度是 p * q
NUM_INPUT_BITS = BITS_PER_PATTERN * NUM_PATTERNS

# --- 数据集参数 ---
DATASET_SIZE = 300000
exp_name = f'pattern_pos_sum_p{BITS_PER_PATTERN}_q{NUM_PATTERNS}'
TRAIN_FILE = f'{exp_name}_train.jsonl'
EVAL_FILE = f'{exp_name}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 可能的模式有 2^p 种
NUM_PATTERN_TYPES = 2 ** BITS_PER_PATTERN

# 每个计数器需要多少bit？
# 位置编号从1到q。一个模式最多出现q次。
# 最大可能的和，是所有位置都相加：1+2+...+q = q*(q+1)/2
MAX_SUM = NUM_PATTERNS * (NUM_PATTERNS + 1) // 2
BITS_PER_SUM = math.ceil(math.log2(MAX_SUM + 1))

# 总输出比特数
TOTAL_OUTPUT_BITS = NUM_PATTERN_TYPES * BITS_PER_SUM

print("=" * 70)
print(f"     神经模式位置和计算器 - 数据集生成器")
print("=" * 70)
print(f"任务: 计算输入中, 按{BITS_PER_PATTERN}-bit分割的模式, 其出现位置编号的和")
print(f"输入格式: {NUM_INPUT_BITS}个'0'/'1'的字符序列")
print(f"输出格式: {TOTAL_OUTPUT_BITS}个多标签二分类")
print(f"  (共有{NUM_PATTERN_TYPES}个模式, 每个和用{BITS_PER_SUM} bits表示)")
print(f"  (最大可能和: {MAX_SUM})")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_binary_string(num_bits):
    """随机生成一个N-bit的二进制字符串"""
    return "".join(random.choice('01') for _ in range(num_bits))


def process_sample(num_input_bits, bits_per_pattern, num_output_bits_per_sum):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入字符串
    input_str = generate_binary_string(num_input_bits)

    # 2. 按k位分割，并计算每个模式出现的位置编号的和
    # 使用 defaultdict 会很方便
    position_sums = defaultdict(int)

    for i in range(0, num_input_bits, bits_per_pattern):
        pattern = input_str[i:i + bits_per_pattern]
        # 位置编号从1开始
        position_index = (i // bits_per_pattern) + 1
        position_sums[pattern] += position_index

    # 3. 生成输出多标签
    # 我们需要一个确定的顺序来排列所有可能的模式
    possible_patterns = [format(i, f'0{bits_per_pattern}b') for i in range(2 ** bits_per_pattern)]

    output_multilabel = []
    for pattern in possible_patterns:
        total_sum = position_sums.get(pattern, 0)  # 如果模式不存在，和为0
        # 将和值转换为固定长度的二进制，并加入最终列表
        sum_bin = format(total_sum, f'0{num_output_bits_per_sum}b')
        output_multilabel.extend([int(bit) for bit in sum_bin])

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_input_bits, bits_per_pattern, num_output_bits_per_sum):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        records.append(process_sample(num_input_bits, bits_per_pattern, num_output_bits_per_sum))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略与之前版本相同的写入逻辑) ...
    random.shuffle(records)
    train_size = int(len(records) * 1)#0.9)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, train_path, eval_path):
        print(f"\n正在写入 {len(train_data)} 条训练数据到 '{train_path}'...")
        with open(train_path, 'w') as f:
            for record in data: f.write(json.dumps(record) + '\n')
        print(f"正在写入 {len(eval_data)} 条评估数据到 '{eval_path}'...")
        with open(eval_path, 'w') as f:
            for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, TRAIN_FILE, EVAL_FILE)
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_INPUT_BITS, BITS_PER_PATTERN, BITS_PER_SUM)