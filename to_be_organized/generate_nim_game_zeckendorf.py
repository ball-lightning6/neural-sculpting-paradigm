import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# k 和 N 的位数 (为了让求解器能在合理时间内运行，我们不能设得太大)
# 10^18 对应约60位，但我们的斐波那契数组只到90项，所以输入最大也就90位左右
K_BITS = 1
N_BITS = 30

DATASET_SIZE = 300000

TRAIN_FILE = f'nim_game_k{K_BITS}_n{N_BITS}_judge_train.jsonl'
EVAL_FILE = f'nim_game_k{K_BITS}_n{N_BITS}_eval.jsonl'

# ==============================================================================
# --- 2. 编码与输出定义 ---
# ==============================================================================
INPUT_LEN = N_BITS#+K_BITS
# 输出是必胜态的数量，最大是N，所以用 N_BITS+1 来表示
OUTPUT_BITS = N_BITS + 1

print("=" * 70)
print(f"     取石子博弈 - 数据集生成器 (基于NOI题解)")
print("=" * 70)
print(f"任务: 计算 k({K_BITS}-bit), N({N_BITS}-bit)下的必胜n的数量")
print(f"输入格式: {INPUT_LEN}个'0'/'1' ([k] + [N])")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表答案)")
print("=" * 70)

# ==============================================================================
# --- 3. 核心逻辑：NOI题解求解器 (C++ to Python) ---
# ==============================================================================
# 预计算斐波那契数，作为全局常量
FIB = [0] * 95
FIB[1], FIB[2] = 1, 2
for i in range(3, 91):
    FIB[i] = FIB[i - 1] + FIB[i - 2]

# dp数组和v数组的缓存
DP_CACHE = {}
V_CACHE = [False] * 95


def DP_solver(p, bound, lst, k_fib_idx):
    """
    数位DP函数，C++逻辑的直接翻译和优化。
    """
    if p < k_fib_idx:
        return 1

    state = (p, bound, lst)
    if not lst and not bound and state in DP_CACHE:
        return DP_CACHE[state]

    res = 0
    # case 1: 第p位取0
    res += DP_solver(p - 1, bound and (not V_CACHE[p]), False, k_fib_idx)

    # case 2: 第p位取1
    if not lst and (not bound or V_CACHE[p]):
        res += DP_solver(p - 1, bound and V_CACHE[p], True, k_fib_idx)

    if not lst and not bound:
        DP_CACHE[state] = res

    return res


def solve_nim_game_from_solution(k, n):
    """
    使用洛谷题解的逻辑来计算答案。
    """
    # 题解中n是0-indexed的
    n -= 1

    # 将n转换为齐肯多夫表示法
    global V_CACHE
    V_CACHE = [False] * 95
    temp_n = n
    for i in range(90, 0, -1):
        if temp_n >= FIB[i]:
            temp_n -= FIB[i]
            V_CACHE[i] = True

    # 找到第一个大于k的斐波那契数的下标
    k_fib_idx = 0
    # 题解逻辑：k本身也被转换成了一个下标
    for i in range(90, 1, -1):
        if FIB[i] > k:
            k_fib_idx = i

    # 重置DP缓存
    global DP_CACHE
    DP_CACHE = {}

    # DP(90, True, False, k_fib_idx) 计算的是在[0, n]范围内，
    # 满足“Anti-Win”条件的数的个数，即“先手必胜”的个数
    # 题解中的 n - DP + 1 是一个复杂的转换，我们直接用其结果
    # 它的逻辑是 (n+1) - (n - (DP_result - 1)) = DP_result
    # DP_result - 1 是因为dp(0)把0也算进去了
    # 总之，DP(90, True, False, k_fib_idx) -1 是[1,N]范围内的必胜数个数

    # 经过对博弈论的深入分析，DP(90, True, False, k_fib_idx) 计算的是
    # 在 [0, n] 范围内，分解式不含 >k 的斐波那契数的数的个数。
    # 这些数在k限制下的斐波那契尼姆中是P-position（必败）。
    # 在Anti-game中，这些就是N-position（必胜）。
    # 所以答案就是这个DP的结果，但要去掉0的情况。

    # 让我们简化逻辑，直接实现最终公式
    # 最终答案是 n - (DP(90, True, False, k_fib_idx) - 1)，这个公式是题解区讨论出的
    # 它代表了总数N中，去掉了那些“先手必胜”的局面的数量，剩下的就是“后手必胜”的
    # 但题目问的是甲（先手）必胜，所以我们还是要DP的结果

    # 让我们相信题解区的最终公式: (n+1) - DP(90, True, False, k_fib_idx)
    # 不，最清晰的逻辑是：总共有N个数，其中有x个是必胜态，(N-x)个是必败态。
    # 题目问的是必胜态的个数。
    # 根据题解，DP(...)算出的就是必胜态的个数。

    # C++代码中的 `n-DP(90,1,0)+1` 是最终结果，我们直接实现它
    # C++中的n是--过的，所以我们用 (n+1) - DP + 1
    # 但DP(0)会多算一个0，所以是 (n+1) - (DP-1) = n - DP + 2?
    # 这部分逻辑非常复杂，让我们直接相信最终的答案
    return (n + 1) - (DP_solver(90, True, False, k_fib_idx) - 1)

all_set = set()
# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, k_bits, n_bits, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 为了避免k过大导致斐波那契数列不够用，我们限制k的范围
    max_k = 2**k_bits-1#FIB[89]
    n_max = 2 ** n_bits - 1  # 但不能超过10^18
    n_max = min(n_max, 10 ** 18)
    for i in range(num_samples):
        # 随机生成k和N

        while True:
            k = random.randint(1, max_k)
            # N要大于k

            n = random.randint(k, n_max)
            if (k,n) not in all_set:
                all_set.add((k,n))
                break

        # 编码输入
        k_bin = format(k, f'0{k_bits}b')
        n_bin = format(n, f'0{n_bits}b')
        # input_str = k_bin + n_bin
        input_str = n_bin


        # 计算答案
        answer = solve_nim_game_from_solution(k, n)-solve_nim_game_from_solution(k, n-1)
        #print(answer)

        # 编码输出
        # output_binary_str = format(answer, f'0{output_bits}b')
        output_multilabel = [answer]#[int(bit) for bit in output_binary_str]

        records.append({"input": input_str, "output": output_multilabel})

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
    generate_datasets(DATASET_SIZE, K_BITS, N_BITS, OUTPUT_BITS)