import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制数组的长度
NUMS_LENGTH_N = 27
# 翻转的子数组长度K的最大值 (k会从1到K_MAX_N之间随机)
K_MAX_N = 7

DATASET_SIZE = 300000

TRAIN_FILE = f'min_k_flips_binary_n{NUMS_LENGTH_N}_k{K_MAX_N}_train.jsonl'
EVAL_FILE = f'min_k_flips_binary_n{NUMS_LENGTH_N}_k{K_MAX_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 [nums] + [k]
K_BITS = math.ceil(math.log2(K_MAX_N + 1))
INPUT_LEN = NUMS_LENGTH_N + K_BITS

# 输出是最小翻转次数的二进制表示
# 最多需要N次翻转，加上0(代表-1)，所以最大值是N+1
MAX_FLIPS = NUMS_LENGTH_N
OUTPUT_BITS = math.ceil(math.log2(MAX_FLIPS + 2))

print("=" * 70)
print(f"     K连续位的最小翻转次数 - 数据集生成器 (二进制输出最终版)")
print("=" * 70)
print(f"数组长度N: {NUMS_LENGTH_N}, 翻转长度K最大为: {K_MAX_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1' ([nums] + [k])")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (0代表不可行)")
print("=" * 70)

all_set = set()
# ==============================================================================
# --- 3. 核心逻辑：求解器 (高效的 O(N) 差分数组算法) ---
# ==============================================================================
def generate_problem(n, k_max):
    """随机生成一个二进制数组和翻转长度k"""
    while True:
        nums = [random.choice([0, 1]) for _ in range(n)]
        k = random.randint(2, k_max)

        nums_tuple = tuple(nums)
        if (nums_tuple, k) not in all_set:
            all_set.add((nums_tuple, k))

            return nums, k


def solve_min_k_flips(nums, k):
    """
    使用O(N)的贪心算法计算最小操作次数。
    这是C语言题解的直接Python翻译。
    """
    n = len(nums)
    diff = [0] * (n + 1)
    ans = 0
    revCnt = 0

    for i in range(n):
        revCnt += diff[i]
        if (nums[i] + revCnt) % 2==0:
            if i + k > n:
                return -1
            ans += 1
            revCnt += 1
            if i + k <= n:
                diff[i + k] -= 1
    return ans


def process_sample(n, k_max, k_bits, output_bits):
    """生成一个完整的 (输入, 输出) 数据对。"""
    nums, k = generate_problem(n, k_max)

    # 编码输入
    k_bin = format(k, f'0{k_bits}b')
    nums_str = "".join(map(str, nums))
    input_str = nums_str + k_bin

    # 计算答案
    min_flips = solve_min_k_flips(nums, k)

    # 编码输出
    output_label = min_flips + 1 if min_flips!=-1 else 0
    output_binary_str = format(output_label, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, n, k_max, k_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        records.append(process_sample(n, k_max, k_bits, output_bits))
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
    generate_datasets(DATASET_SIZE, NUMS_LENGTH_N, K_MAX_N, K_BITS, OUTPUT_BITS)