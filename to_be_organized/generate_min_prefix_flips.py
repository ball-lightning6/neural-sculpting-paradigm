import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制数组的长度
STRING_LENGTH_N = 30
DATASET_SIZE = 300000

TRAIN_FILE = f'flip_to_ones_binary_n{STRING_LENGTH_N}_train.jsonl'
EVAL_FILE = f'flip_to_ones_binary_n{STRING_LENGTH_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = STRING_LENGTH_N
# 输出是操作次数的二进制表示
# 最多需要N次操作
MAX_FLIPS = STRING_LENGTH_N
# 表示这个最大值需要多少个bit
OUTPUT_BITS = math.ceil(math.log2(MAX_FLIPS + 1))

print("=" * 70)
print(f"     “翻转至全1”最小操作次数 - 数据集生成器 (二进制输出版)")
print("=" * 70)
print(f"任务: 对一个{INPUT_LEN}-bit数组，计算使其全为1的最小操作次数")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 ---
# ==============================================================================
def generate_binary_array(num_bits):
    """随机生成一个N-bit的二进制数组"""
    return [random.choice([0, 1]) for _ in range(num_bits)]


def solve_min_flips_to_ones(nums):
    """
    使用O(N)的贪心算法计算最小操作次数。
    """
    flips = 0
    for num in nums:
        current_state = num if (flips % 2==0) else 1 - num
        if current_state==0:
            flips += 1
    return flips


def process_sample(num_input_bits, num_output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    input_array = generate_binary_array(num_input_bits)
    min_flips = solve_min_flips_to_ones(input_array)

    # 将输入数组转换为字符串
    input_str = "".join(map(str, input_array))

    # --- 关键改动：将输出转换为二进制多标签 ---
    output_binary_str = format(min_flips, f'0{num_output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_input_bits, num_output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(num_input_bits, num_output_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略写入逻辑) ...
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
    generate_datasets(DATASET_SIZE, STRING_LENGTH_N, OUTPUT_BITS)