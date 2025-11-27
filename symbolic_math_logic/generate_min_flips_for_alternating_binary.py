import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制字符串的总位数
STRING_LENGTH_N = 30  # 必须是偶数
DATASET_SIZE = 300000

TRAIN_FILE = f'alternating_flips_n{STRING_LENGTH_N}_train.jsonl'
EVAL_FILE = f'alternating_flips_n{STRING_LENGTH_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = STRING_LENGTH_N
# 输出是最小修改次数，最大可能是 N/2
MAX_FLIPS = STRING_LENGTH_N // 2
OUTPUT_BITS = math.ceil(math.log2(MAX_FLIPS + 1))

print("=" * 70)
print(f"     “美丽字符串”最小修改次数 - 数据集生成器")
print("=" * 70)
print(f"任务: 对一个{INPUT_LEN}-bit字符串，计算使其交替的最小修改次数")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表修改次数)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (高效的滑动窗口算法) ---
# ==============================================================================
def generate_binary_string(num_bits):
    """随机生成一个N-bit的二进制字符串"""
    return "".join(random.choice('01') for _ in range(num_bits))


def solve_min_flips_for_alternating(s):
    """
    使用高效的 O(N) 滑动窗口算法计算最小修改次数。
    """
    n = len(s)
    # 构造两种目标交替字符串
    target0 = "".join(['0' if i % 2==0 else '1' for i in range(n)])
    target1 = "".join(['1' if i % 2==0 else '0' for i in range(n)])

    # 核心技巧：通过拼接字符串，将循环问题转化为滑动窗口问题
    s_double = s + s

    diff0, diff1 = 0, 0
    min_flips = n

    # 初始化第一个窗口的差异
    for i in range(n):
        if s[i]!=target0[i]:
            diff0 += 1
        if s[i]!=target1[i]:
            diff1 += 1
    min_flips = min(diff0, diff1)

    # 开始滑动窗口
    for i in range(1, n):
        # 移出窗口的第一个字符
        if s[i - 1]!=target0[i - 1]: diff0 -= 1
        if s[i - 1]!=target1[i - 1]: diff1 -= 1

        # 移入窗口的最后一个字符
        # 注意：比较的对象是原始target字符串的相应位置
        new_char_idx_in_s = i + n - 1
        new_char_idx_in_target = n - 1
        if s_double[new_char_idx_in_s]!=target0[new_char_idx_in_target]: diff0 += 1
        if s_double[new_char_idx_in_s]!=target1[new_char_idx_in_target]: diff1 += 1

        min_flips = min(min_flips, diff0, diff1)

    return min_flips


def process_sample(num_input_bits, num_output_bits):
    """生成一个完整的 (输入, 输出) 数据对。"""
    input_str = generate_binary_string(num_input_bits)
    min_flips = solve_min_flips_for_alternating(input_str)
    output_binary_str = format(min_flips, f'0{num_output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]
    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_input_bits, num_output_bits):
    print("\n--- 开始生成数据集 ---")
    records = []
    for i in range(num_samples):
        records.append(process_sample(num_input_bits, num_output_bits))
        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")
    print(f"生成完毕。共 {len(records)} 条数据。")

    # ... (省略写入文件的逻辑) ...
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
    generate_datasets(DATASET_SIZE, INPUT_LEN, OUTPUT_BITS)
