import json
import random

# ==============================================================================
# --- 1. 核心参数配置 (你的“宇宙编辑器”) ---
# ==============================================================================
# 你想让模型排序多少个数？(从2开始)
NUM_ITEMS = 5
# 每个数用多少个bit表示？
NUM_BITS_PER_ITEM = 6

DATASET_SIZE = 500000

TRAIN_FILE = f'sorter_{NUM_ITEMS}items_{NUM_BITS_PER_ITEM}bit_train.jsonl'
EVAL_FILE = f'sorter_{NUM_ITEMS}items_{NUM_BITS_PER_ITEM}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = NUM_ITEMS * NUM_BITS_PER_ITEM
OUTPUT_LEN = NUM_ITEMS * NUM_BITS_PER_ITEM

print("=" * 70)
print(f"     神经排序网络 - 数据集生成器")
print("=" * 70)
print(f"任务: 对 {NUM_ITEMS} 个 {NUM_BITS_PER_ITEM}-bit 的无重复正整数进行排序")
print(f"输入格式: {INPUT_LEN} 个 '0'/'1' 的字符序列")
print(f"输出格式: {OUTPUT_LEN} 个多标签二分类 (排序后的结果)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_unique_numbers(num_items, num_bits):
    """随机生成一串不重复的N-bit整数"""
    max_val = 2 ** num_bits - 1
    if num_items > max_val + 1:
        raise ValueError("数的个数不能超过可能的数字总数")

    numbers = set()
    while len(numbers) < num_items:
        numbers.add(random.randint(0, max_val))
    #print(list(numbers))
    return list(numbers)


def process_pair(numbers, num_bits):
    """
    为一组数，生成未排序的输入和排好序的输出。
    """
    # 1. 对原始数字列表进行排序，得到“真值”
    sorted_numbers = sorted(numbers)

    # 2. 生成输入字符串 (使用原始、未排序的列表)
    input_str_list = [format(n, f'0{num_bits}b') for n in numbers]
    input_str = "".join(input_str_list)

    # 3. 生成输出多标签 (使用排好序的列表)
    output_str_list = [format(n, f'0{num_bits}b') for n in sorted_numbers]
    output_str = "".join(output_str_list)
    output_multilabel = [int(bit) for bit in output_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_items, num_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 使用set去重，确保我们生成的数字组合是唯一的
    seen_sets = set()

    while len(seen_sets) < num_samples:
        # 生成一组不重复的数字
        numbers = generate_unique_numbers(num_items, num_bits)
        # 将其转换为可哈希的frozenset用于去重
        num_set_key = frozenset(numbers)

        if num_set_key in seen_sets:
            continue
        seen_sets.add(num_set_key)

        # 打乱原始顺序以确保输入是无序的
        random.shuffle(numbers)
        records.append(process_pair(numbers, num_bits))

        if len(seen_sets) % 10000==0 and len(seen_sets) > 0:
            print(f"已生成 {len(seen_sets)} / {num_samples} 组不重复的数字...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")

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
    generate_datasets(DATASET_SIZE, NUM_ITEMS, NUM_BITS_PER_ITEM)