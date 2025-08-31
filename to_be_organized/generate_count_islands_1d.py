import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制字符串的位数
NUM_INPUT_BITS = 30
# 输出是最多15个岛屿，需要4-bit来表示
NUM_OUTPUT_BITS = 4

DATASET_SIZE = 300000

TRAIN_FILE = f'island_counter_{NUM_INPUT_BITS}bit_train.jsonl'
EVAL_FILE = f'island_counter_{NUM_INPUT_BITS}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
print("=" * 70)
print(f"     神经孤岛计数器 - 数据集生成器")
print("=" * 70)
print(f"任务: 计算一个{NUM_INPUT_BITS}-bit二进制字符串中'1'的孤岛数量")
print(f"输入格式: {NUM_INPUT_BITS}个'0'/'1'的字符序列")
print(f"输出格式: {NUM_OUTPUT_BITS}个多标签二分类 (代表孤岛数量)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_binary_string(num_bits):
    """随机生成一个N-bit的二进制字符串"""
    # 为了生成更多样化的孤岛，我们不完全随机
    # 而是随机生成一些0和1的“块”并拼接
    s = []
    while len(s) < num_bits:
        if random.random() < 0.5:  # 生成一串0
            s.extend([0] * random.randint(1, 5))
        else:  # 生成一串1
            s.extend([1] * random.randint(1, 5))
    return "".join(map(str, s[:num_bits]))


def count_islands(binary_str):
    """计算二进制字符串中'1'的孤岛数量"""
    count = 0
    in_island = False
    for char in binary_str:
        if char=='1' and not in_island:
            count += 1
            in_island = True
        elif char=='0':
            in_island = False
    return count


def process_sample(num_input_bits, num_output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入字符串
    input_str = generate_binary_string(num_input_bits)

    # 2. 计算孤岛数量
    island_count = count_islands(input_str)

    # 3. 生成二进制多标签输出
    output_binary_str = format(island_count, f'0{num_output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_input_bits, num_output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复，我们可以简化去重
    for i in range(num_samples):
        records.append(process_sample(num_input_bits, num_output_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    random.shuffle(records)
    train_size = int(len(records) * 1)
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
    generate_datasets(DATASET_SIZE, NUM_INPUT_BITS, NUM_OUTPUT_BITS)