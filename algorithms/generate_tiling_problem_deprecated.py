import json
import random
import math
import sys

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 矩阵的最大尺寸 N x M
# LeetCode 限制是 13，这是一个计算上可行的上限
N_MAX = 13
M_MAX = 13

DATASET_SIZE = 100000  # 生成一个足够大的数据集

TRAIN_FILE = f'tiling_rectangle_n{N_MAX}_m{M_MAX}_train.jsonl'
EVAL_FILE = f'tiling_rectangle_n{N_MAX}_m{M_MAX}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 n 和 m
# 1 <= n, m <= 13，所以每个数最多需要 4 bits
BITS_PER_DIM = 4
INPUT_LEN = BITS_PER_DIM * 2

# 输出是最小正方形数量
# 对于n,m<=13，已知最大结果是8。为保险起见，我们用4位表示(0-15)
OUTPUT_BITS = 4

print("=" * 70)
print(f"     铺瓷砖问题 - 数据集生成器")
print("=" * 70)
print(f"最大尺寸: {N_MAX}x{M_MAX}")
print(f"输入格式: {INPUT_LEN}个'0'/'1' ([n] + [m])")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表最小数量)")
print("=" * 70)

# ==============================================================================
# --- 3. 核心逻辑：求解器 (带剪枝的回溯搜索) ---
# ==============================================================================
# 使用一个全局变量来缓存已计算的结果，因为 solve(n,m) == solve(m,n)
solution_cache = {}


def solve_tiling(n, m):
    """
    使用带剪枝的回溯搜索来解决铺瓷砖问题。
    """
    if n > m:
        n, m = m, n  # 确保 n <= m

    if (n, m) in solution_cache:
        return solution_cache[(n, m)]

    if n==0 or m==0:
        return 0
    if n==m:
        return 1

    ans = m * n  # 一个简单的上界：用1x1的瓷砖

    # sky_line 是一个表示当前矩形顶部轮廓的数组
    sky_line = [0] * n

    def backtrack(count):
        nonlocal ans

        # 剪枝：如果当前数量已经超过已知最优解，则返回
        if count >= ans:
            return

        # 找到第一个未被覆盖的最低点
        min_h = m
        start_pos = -1
        for i in range(n):
            if sky_line[i] < min_h:
                min_h = sky_line[i]
                start_pos = i

        # 如果所有点都已被覆盖，说明找到一个解
        if start_pos==-1:
            ans = min(ans, count)
            return

        # 寻找可以放置的最大正方形的右边界
        end_pos = start_pos
        while end_pos + 1 < n and sky_line[end_pos + 1]==min_h:
            end_pos += 1

        # 尝试放置从大到小的正方形
        max_square_size = min(end_pos - start_pos + 1, m - min_h)
        for size in range(max_square_size, 0, -1):

            # 更新天际线
            for i in range(start_pos, start_pos + size):
                sky_line[i] += size

            backtrack(count + 1)

            # 撤销操作，回溯
            for i in range(start_pos, start_pos + size):
                sky_line[i] -= size

    backtrack(0)
    solution_cache[(n, m)] = ans
    return ans


# ==============================================================================
# --- 4. 数据集生成主函数 ---
# ==============================================================================
def generate_datasets(num_samples, n_max, m_max):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        n = random.randint(1, n_max)
        m = random.randint(1, m_max)

        # 编码输入
        n_bin = format(n, f'0{BITS_PER_DIM}b')
        m_bin = format(m, f'0{BITS_PER_DIM}b')
        input_str = n_bin + m_bin

        # 计算答案
        min_squares = solve_tiling(n, m)

        # 编码输出
        output_binary_str = format(min_squares, f'0{OUTPUT_BITS}b')
        output_multilabel = [int(bit) for bit in output_binary_str]

        records.append({"input": input_str, "output": output_multilabel})

        if (i + 1) % 1000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略) ...
    random.shuffle(records)
    train_size = int(len(records) * 1)# 0.9)
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
    generate_datasets(DATASET_SIZE, N_MAX, M_MAX)