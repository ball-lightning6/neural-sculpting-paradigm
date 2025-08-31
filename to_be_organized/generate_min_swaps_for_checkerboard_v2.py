import json
import random
import math
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 矩阵大小 N x N (注意：N>10，DP解法会变得极慢！)
MATRIX_SIZE_N = 5
DATASET_SIZE = 300000

TRAIN_FILE = f'grid_sort_{MATRIX_SIZE_N}d_train.jsonl'
EVAL_FILE = f'grid_sort_{MATRIX_SIZE_N}d_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = MATRIX_SIZE_N * MATRIX_SIZE_N

# 输出是最小交换次数
# 最大交换次数是C(N,2) = N*(N-1)/2
MAX_SWAPS = MATRIX_SIZE_N * (MATRIX_SIZE_N - 1) // 2
OUTPUT_BITS = math.ceil(math.log2(MAX_SWAPS + 2))  # +1是0，再+1是-1

print("=" * 70)
print(f"     网格排序最小交换次数 - 数据集生成器")
print("=" * 70)
print(f"矩阵大小: {MATRIX_SIZE_N}x{MATRIX_SIZE_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (-1用0表示)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (状态压缩DP) ---
# ==============================================================================
def solve_grid_swaps(grid):
    """
    使用状态压缩DP + BFS 来解决这个问题。
    这本质上是在图上寻找最短路径。
    """
    n = len(grid)

    # 步骤1: 检查哪些行可以放在哪些位置 (row_i是否可以放在第k行)
    # 放在第k行，意味着它上面的k-1行，都必须在对角线之上全为0
    # 同时，它自己也必须在对角线之上全为0
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: continue

            # 检查 row_i 能否放在 row_j 上面
            # row_i 在 k, row_j 在 k+1
            # row_i 的第 k+1 列必须是0
            # row_j 的第 k 列必须是1 (或无所谓)
            # 这个逻辑太复杂，一个更简单的方法是直接构建邻接关系

            # row i 能否紧邻在 row j 上面
            can_be_on_top = True
            for col in range(j + 1, n):
                if grid[i][col]==1:
                    can_be_on_top = False
                    break
            if can_be_on_top:
                adj[i][j] = True

    # 步骤2: BFS在状态图上寻找最短路径
    # 状态是 (mask, last_row_idx, inversions)
    # mask: 一个位掩码，表示哪些行已经被使用
    # last_row_idx: 路径中最后一个行的索引
    # inversions: 到达这个状态所需的交换次数（即逆序对数）

    # 更简单直接的解法：这是一个排列问题
    # 我们可以把每一行看作一个节点，构建一个图，然后找哈密尔顿路径

    # 最终的、正确的、但依然复杂的解法：
    # 这是一个“旅行商问题”的变体，我们可以用DP解决
    # dp[mask][i] = 把mask中的行排列好，且最后一行是i时，所需的最小交换次数

    # --- 为简化和效率，我们用一个更直观的 BFS ---
    # 状态: (当前行的排列顺序tuple, 交换次数)
    initial_perm = tuple(range(n))
    queue = deque([(initial_perm, 0)])
    visited = {initial_perm}

    while queue:
        perm, swaps = queue.popleft()

        # 检查当前排列是否满足要求
        is_beautiful = True
        for r_idx, original_row_idx in enumerate(perm):
            for c_idx in range(r_idx + 1, n):
                if grid[original_row_idx][c_idx]==1:
                    is_beautiful = False
                    break
            if not is_beautiful:
                break

        if is_beautiful:
            return swaps

        # 生成下一个状态
        perm_list = list(perm)
        for i in range(n - 1):
            # 交换相邻的行
            new_perm_list = perm_list[:]
            new_perm_list[i], new_perm_list[i + 1] = new_perm_list[i + 1], new_perm_list[i]
            new_perm = tuple(new_perm_list)
            if new_perm not in visited:
                visited.add(new_perm)
                queue.append((new_perm, swaps + 1))

    return -1

all_set= set()
# ==============================================================================
# --- 4. 数据集生成主函数 ---
# ==============================================================================
def generate_datasets(num_samples, n):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        # 随机生成矩阵
        while True:
            grid = [[random.choice([0, 1]) for _ in range(n)] for _ in range(n)]
            grid_str= ''.join(''.join(str(x)) for x in grid)
            if grid_str not in all_set:
                all_set.add(grid_str)
                break

        # 计算最小交换次数
        # 注意：对于N>8，暴力BFS会非常非常慢！
        # 这个脚本只适用于小N值的探索。
        if n > 8:
            print("\n警告：N>8时，暴力求解会极其缓慢，可能无法完成。仅用于演示。")
            # 我们可以用一个近似解或跳过
            min_swaps = -1  # 假设无法求解
        else:
            min_swaps = solve_grid_swaps(grid)

        # 编码输入和输出
        input_str = "".join(str(cell) for row in grid for cell in row)

        # 将-1映射为0，其他+1
        output_label = min_swaps + 1
        output_binary_str = format(output_label, f'0{OUTPUT_BITS}b')
        output_multilabel = [int(bit) for bit in output_binary_str]

        records.append({"input": input_str, "output": output_multilabel})

        if (i + 1) % 1000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # ... (省略写入文件的逻辑)
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
    generate_datasets(DATASET_SIZE, MATRIX_SIZE_N)