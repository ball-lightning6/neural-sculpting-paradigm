import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 我们要处理多少个数？
NUM_ITEMS = 3
# 每个数用多少个bit表示？
BITS_PER_ITEM = 10

DATASET_SIZE = 300000

TRAIN_FILE = f'hamming_distance_sum_{NUM_ITEMS}items_{BITS_PER_ITEM}bit_train.jsonl'
EVAL_FILE = f'hamming_distance_sum_{NUM_ITEMS}items_{BITS_PER_ITEM}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = NUM_ITEMS * BITS_PER_ITEM

# 输出是汉明距离总和
# 计算最大可能的总和，以确定输出位数
# 最大距离发生在一半是全0，一半是全1时
# 两两配对数：C(N, 2) = N * (N-1) / 2
# 每对最大距离：BITS_PER_ITEM
MAX_SUM = (NUM_ITEMS * (NUM_ITEMS - 1) // 2) * BITS_PER_ITEM
OUTPUT_BITS = math.ceil(math.log2(MAX_SUM + 1))

print("=" * 70)
print(f"     汉明距离总和 - 数据集生成器")
print("=" * 70)
print(f"任务: 计算 {NUM_ITEMS} 个 {BITS_PER_ITEM}-bit 数之间的汉明距离总和")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表总和)")
print(f"最大可能和: {MAX_SUM}")
print("=" * 70)

all_set = set()
# ==============================================================================
# --- 3. 核心逻辑：求解器 (高效的 O(N*L) 算法) ---
# ==============================================================================
def generate_numbers(num_items, num_bits):
    """随机生成一串N-bit整数"""
    max_val = 2 ** num_bits - 1
    while True:
        nums = tuple([random.randint(0, max_val) for _ in range(num_items)])
        if nums not in all_set:
            all_set.add(nums)
            return nums


def solve_total_hamming_distance(nums, num_bits):
    """
    使用高效的按位统计方法，在O(N*L)时间内解决问题。
    """
    total_distance = 0
    num_count = len(nums)

    for i in range(num_bits):
        # 统计在第i位上，有多少个1
        count_of_ones = 0
        for num in nums:
            if (num >> i) & 1:
                count_of_ones += 1

        count_of_zeros = num_count - count_of_ones

        # 在这一位上产生的距离，是(0的个数) * (1的个数)
        distance_at_bit_i = count_of_zeros * count_of_ones
        total_distance += distance_at_bit_i

    return total_distance


def process_sample(num_items, num_bits, output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入数组
    numbers = generate_numbers(num_items, num_bits)

    # 2. 计算汉明距离总和
    total_distance = solve_total_hamming_distance(numbers, num_bits)

    # 3. 编码输入和输出
    input_str_list = [format(n, f'0{num_bits}b') for n in numbers]
    input_str = "".join(input_str_list)

    output_binary_str = format(total_distance, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_items, num_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(num_items, num_bits, output_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略) ...
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
    generate_datasets(DATASET_SIZE, NUM_ITEMS, BITS_PER_ITEM, OUTPUT_BITS)