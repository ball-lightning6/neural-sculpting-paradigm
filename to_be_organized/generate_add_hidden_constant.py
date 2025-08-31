import json
import random

# ==============================================================================
# --- 1. 核心参数与“隐藏的宇宙常数” ---
# ==============================================================================
NUM_BITS = 20  # 操作数x和常数C的位数
DATASET_SIZE = 200000

TRAIN_FILE = f'hidden_adder_{NUM_BITS}bit_train.jsonl'
EVAL_FILE = f'hidden_adder_{NUM_BITS}bit_eval.jsonl'

# --- 在这里，我们定义那个“隐藏的幽灵” ---
# 生成一个一次性的、固定的、隐藏的常数 C
# 为了可复现性，我们使用固定的随机种子
random.seed(42)
MAX_VAL = 2 ** NUM_BITS - 1
HIDDEN_CONSTANT_C = random.randint(0, MAX_VAL)

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_BITS = NUM_BITS
OUTPUT_BITS = NUM_BITS + 1

print("=" * 70)
print(f"     “幽灵加法器”实验 - 数据集生成器")
print("=" * 70)
print(f"任务: 学习 y = x + C，其中 C 是一个隐藏的 {NUM_BITS}-bit 常数。")
print(f"隐藏的常数C的值 (十进制): {HIDDEN_CONSTANT_C}")
print(f"输入格式: {INPUT_BITS}个'0'/'1'的字符序列 (代表x)")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表y)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def process_pair(x, hidden_c, num_bits, output_bits):
    """
    为输入x和隐藏常数c，生成输入和输出。
    """
    # 1. 生成二进制输入字符串 (只有x)
    input_str = format(x, f'0{num_bits}b')

    # 2. 计算结果 y = x + C
    y = x + hidden_c

    # 3. 生成二进制多标签输出
    output_binary_str = format(y, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits, hidden_c, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    seen_x = set()

    # 使用set去重，确保我们生成的输入x是唯一的
    while len(seen_x) < num_samples:
        x = random.randint(0, 2 ** num_bits - 1)
        if x in seen_x:
            continue
        seen_x.add(x)

        records.append(process_pair(x, hidden_c, num_bits, output_bits))

        if len(seen_x) % 10000==0 and len(seen_x) > 0:
            print(f"已生成 {len(seen_x)} / {num_samples} 条不重复的输入x...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")

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
    generate_datasets(DATASET_SIZE, NUM_BITS, HIDDEN_CONSTANT_C, OUTPUT_BITS)