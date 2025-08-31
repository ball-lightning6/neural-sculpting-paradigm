import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制数的位数
NUM_DATA_BITS = 24
# 用于表示滚动次数的位数
NUM_SHIFT_BITS = 6  # 2^6 = 64，可以表示0-63位的滚动

DATASET_SIZE = 500000

TRAIN_FILE = f'shifter_{NUM_DATA_BITS}bit_by_{NUM_SHIFT_BITS}bit_train.jsonl'
EVAL_FILE = f'shifter_{NUM_DATA_BITS}bit_by_{NUM_SHIFT_BITS}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = NUM_DATA_BITS + NUM_SHIFT_BITS
OUTPUT_LEN = NUM_DATA_BITS

print("=" * 70)
print(f"     神经位移寄存器 - 数据集生成器")
print("=" * 70)
print(f"任务: 将一个{NUM_DATA_BITS}-bit数，向右循环滚动k位({NUM_SHIFT_BITS}-bit表示)")
print(f"输入格式: {INPUT_LEN}个'0'/'1' ([数据] + [滚动位数])")
print(f"输出格式: {OUTPUT_LEN}个多标签二分类 (滚动后的结果)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_shift_problem(num_data_bits, num_shift_bits):
    """随机生成一个数和一个滚动位数"""
    data_max = 2 ** num_data_bits - 1
    shift_max = 2 ** num_shift_bits - 1

    data = random.randint(0, data_max)
    shift_amount = random.randint(0, shift_max)

    return data, shift_amount


def process_pair(data, shift_amount, num_data_bits, num_shift_bits):
    """
    为给定的数据和滚动位数，生成输入和输出。
    """
    # 1. 生成输入字符串
    data_bin = format(data, f'0{num_data_bits}b')
    shift_bin = format(shift_amount, f'0{num_shift_bits}b')
    input_str = data_bin + shift_bin

    # 2. 计算循环右移的结果
    # Python中没有内置的循环移位，我们手动实现
    # 将字符串看作一个列表，进行切片和拼接
    shift_amount = shift_amount % num_data_bits  # 确保滚动位数在范围内
    result_str = data_bin[-shift_amount:] + data_bin[:-shift_amount]

    # 3. 生成二进制多标签输出
    output_multilabel = [int(bit) for bit in result_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_data_bits, num_shift_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机生成的(data, shift)对几乎不可能重复，所以我们简化去重
    for i in range(num_samples):
        data, shift_amount = generate_shift_problem(num_data_bits, num_shift_bits)

        records.append(process_pair(data, shift_amount, num_data_bits, num_shift_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
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
    generate_datasets(DATASET_SIZE, NUM_DATA_BITS, NUM_SHIFT_BITS)