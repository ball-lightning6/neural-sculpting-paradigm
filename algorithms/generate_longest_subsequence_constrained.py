import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制字符串的长度
STRING_LENGTH_N = 30
# k的最大值，用位数来表示
K_BITS = 10  # 2^10 = 1024

DATASET_SIZE = 300000

TRAIN_FILE = f'longest_subsequence_binary_n{STRING_LENGTH_N}_k{K_BITS}_train.jsonl'
EVAL_FILE = f'longest_subsequence_binary_n{STRING_LENGTH_N}_k{K_BITS}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 [s] + [k]
INPUT_LEN = STRING_LENGTH_N + K_BITS
# 输出是子序列长度的二进制表示
# 最大长度是N
OUTPUT_BITS = math.ceil(math.log2(STRING_LENGTH_N + 1))

print("=" * 70)
print(f"     最长子序列长度 - 数据集生成器 (二进制输出版)")
print("=" * 70)
print(f"任务: 寻找s({STRING_LENGTH_N}-bit)中值<=k({K_BITS}-bit)的最长子序列长度")
print(f"输入格式: {INPUT_LEN}个'0'/'1' ([s] + [k])")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类")
print("=" * 70)

all_set = set()
# ==============================================================================
# --- 3. 核心逻辑：求解器 (高效的 O(N) DP解法) ---
# ==============================================================================
def generate_problem(n_bits, k_bits):
    """随机生成一个二进制字符串s和一个整数k"""
    while True:
        s = "".join(random.choice('01') for _ in range(n_bits))
        k = random.randint(1, 2 ** k_bits - 1)
        if (s,k) not in all_set:
            all_set.add((s,k))
            return s,k
    # return s, k


def solve(s: str, k: int) -> int:
    """
    解决最长子序列问题：找到s的一个子序列，其二进制值 <= k，使得子序列长度最大。
    贪心策略：
    1. 所有的'0'都可以选（因为不增加数值）。
    2. 对于'1'，我们希望选尽可能多的'1'，且这些'1'应该尽可能在低位（右边），这样数值才小。
    """
    sm = 0
    cnt = 0
    bits = k.bit_length()
    # 从右往左遍历（低位到高位）
    for i, ch in enumerate(s[::-1]):
        if ch == '1':
            # 如果当前位 i 小于 k 的位数，且加上这个 1 后总值 <= k
            if i < bits and sm + (1 << i) <= k:
                sm += 1 << i
                cnt += 1
        else:
            cnt += 1
    return cnt


def process_sample(n_bits, k_bits, output_bits):
    """生成一个完整的 (输入, 输出) 数据对。"""
    s, k = generate_problem(n_bits, k_bits)
    length = solve(s, k)

    # 编码输入和输出
    k_bin = format(k, f'0{k_bits}b')
    input_str = s + k_bin

    # --- 关键改动：将输出转换为二进制多标签 ---
    output_binary_str = format(length, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, n_bits, k_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(n_bits, k_bits, output_bits))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略)
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
    generate_datasets(DATASET_SIZE, STRING_LENGTH_N, K_BITS, OUTPUT_BITS)