import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# a % n
# 被除数 a 的位数
A_BITS = 20
# 模数 n 的位数 (更小，更自然)
N_BITS = 2

DATASET_SIZE = 500000

# 是否固定模数为3（用于mod3实验，在论文中提到需要解耦才能学习）
# 如果设为False，则n会随机生成，输入也会包含n的信息
FIX_MOD_TO_THREE = True

TRAIN_FILE = f'modulo_{A_BITS}bit_by_{N_BITS}bit_train.jsonl'
EVAL_FILE = f'modulo_{A_BITS}bit_by_{N_BITS}bit_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 a_bits + n_bits 长度的0/1字符
INPUT_LEN = A_BITS + N_BITS
# 输出是结果y，最大是n-1，所以用 n_bits 来表示就足够了
OUTPUT_BITS = N_BITS

print("=" * 70)
print(f"非对称取模运算 ({A_BITS}-bit % {N_BITS}-bit) - 数据集生成器")
print("=" * 70)
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列 ([a] + [n])")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表结果y)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def generate_modulo_pair(a_bits, n_bits):
    """随机生成一对用于取模的整数 (a, n)"""
    a_max = 2 ** a_bits - 1
    n_max = 2 ** n_bits - 1

    a = random.randint(0, a_max)
    # 模数n不能为0或1，从2开始
    n = random.randint(2, n_max)
    if FIX_MOD_TO_THREE:
        n = 3  # 固定为3，用于mod3实验
    else:
        n = random.randint(2, n_max)  # 随机生成模数
    return a, n


def process_pair(a, n, a_bits, n_bits, output_bits):
    """
    为一对整数(a, n)生成输入和输出。
    """
    # 1. 生成二进制输入字符串
    a_bin = format(a, f'0{a_bits}b')
    n_bin = format(n, f'0{n_bits}b')
    if FIX_MOD_TO_THREE:
        # 当n固定为3时，模型无法从输入中得知模数，这是论文中提到的难点
        input_str = a_bin
    else:
        # 当n不固定时，输入包含a和n，让模型知道当前模数
        input_str = a_bin + n_bin

    # 2. 计算结果 y = a % n
    result = a % n

    # print(a,n,result)  # 调试打印，注释以提升性能

    # 3. 生成二进制多标签输出
    output_binary_str = format(result, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, a_bits, n_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    seen_pairs = set()

    while len(seen_pairs) < num_samples:
        a, n = generate_modulo_pair(a_bits, n_bits)
        if (a, n) in seen_pairs:
            continue
        seen_pairs.add((a, n))

        records.append(process_pair(a, n, a_bits, n_bits, output_bits))

        if len(seen_pairs) % 10000==0 and len(seen_pairs) > 0:
            print(f"已生成 {len(seen_pairs)} / {num_samples} 条不重复的取模问题...")

    print(f"生成完毕。共 {len(records)} 条不重复数据。")

    # --- 写入文件 ---
    random.shuffle(records)
    train_size = int(len(records) * 0.9)
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
    generate_datasets(DATASET_SIZE, A_BITS, N_BITS, OUTPUT_BITS)