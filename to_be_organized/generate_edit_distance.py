import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 定义两个字符串的位数
NUM_BITS_PER_STRING = 15

DATASET_SIZE = 2000

TRAIN_FILE = f'edit_distance_{NUM_BITS_PER_STRING}bit_train1.jsonl'
EVAL_FILE = f'edit_distance_{NUM_BITS_PER_STRING}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = NUM_BITS_PER_STRING * 2
# 编辑距离最大为两个字符串中较长者的长度
OUTPUT_BITS = math.ceil(math.log2(NUM_BITS_PER_STRING + 1))

print("=" * 70)
print(f"     神经编辑距离计算器 - 数据集生成器")
print("=" * 70)
print(f"任务: 计算两个{NUM_BITS_PER_STRING}-bit字符串的编辑距离")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表距离)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_binary_string(num_bits):
    """随机生成一个N-bit的二进制字符串"""
    return "".join(random.choice('01') for _ in range(num_bits))


def levenshtein_distance(s1, s2):
    """一个标准的、计算两个字符串编辑距离的动态规划算法"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2)==0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1!=c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

all_set=set()

def process_pair(num_bits, output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    while True:
        s1 = generate_binary_string(num_bits)
        s2 = generate_binary_string(num_bits)

        # 输入是两个字符串的拼接
        input_str = s1 + s2
        if input_str not in all_set:
            all_set.add(input_str)
            break

    # 计算编辑距离作为“真值”
    distance = levenshtein_distance(s1, s2)

    # 生成输出多标签
    output_binary_str = format(distance, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_pair(num_bits, output_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略与之前版本相同的写入逻辑) ...
    random.shuffle(records)
    train_size = int(len(records) * 1)
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
    import math

    output_bits = math.ceil(math.log2(NUM_BITS_PER_STRING + 1))
    generate_datasets(DATASET_SIZE, NUM_BITS_PER_STRING, output_bits)