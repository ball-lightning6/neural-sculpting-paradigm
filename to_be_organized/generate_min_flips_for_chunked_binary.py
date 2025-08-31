import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制字符串的总位数
INPUT_BITS = 30  # 必须是偶数
# 按多少位进行分割
BITS_PER_CHUNK = 2  # 在这个问题中，固定为2

DATASET_SIZE = 300000

exp_name = f'beautiful_string_p{BITS_PER_CHUNK}_n{INPUT_BITS}'
TRAIN_FILE = f'{exp_name}_train.jsonl'
EVAL_FILE = f'{exp_name}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输出是修改次数，是一个整数
# 最多需要修改的次数是 INPUT_BITS / 2
MAX_FLIPS = INPUT_BITS // BITS_PER_CHUNK
# 表示这个最大值需要多少个bit
OUTPUT_BITS = math.ceil(math.log2(MAX_FLIPS + 1))

print("=" * 70)
print(f"     “美丽字符串”最小修改次数 - 数据集生成器")
print("=" * 70)
print(f"任务: 对一个{INPUT_BITS}-bit字符串，计算使其“美丽”的最小修改数")
print(f"输入格式: {INPUT_BITS}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表修改次数)")
print("=" * 70)

all_set=set()
# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_binary_string(num_bits):
    """随机生成一个N-bit的二进制字符串"""
    while True:
        s= "".join(random.choice('01') for _ in range(num_bits))
        if s not in all_set:
            all_set.add(s)
            return s


def calculate_min_flips(binary_str, chunk_size):
    """计算最小修改次数"""
    flips = 0
    for i in range(0, len(binary_str), chunk_size):
        chunk = binary_str[i:i + chunk_size]
        # 只要块内的字符不完全相同，就需要一次修改
        if chunk[0]!=chunk[1]:
            flips += 1
    return flips


def process_sample(num_input_bits, chunk_size, num_output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入字符串
    input_str = generate_binary_string(num_input_bits)

    # 2. 计算最小修改次数
    min_flips = calculate_min_flips(input_str, chunk_size)

    # 3. 生成二进制多标签输出
    output_binary_str = format(min_flips, f'0{num_output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_input_bits, chunk_size, num_output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(num_input_bits, chunk_size, num_output_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略写入逻辑) ...
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
    generate_datasets(DATASET_SIZE, INPUT_BITS, BITS_PER_CHUNK, OUTPUT_BITS)