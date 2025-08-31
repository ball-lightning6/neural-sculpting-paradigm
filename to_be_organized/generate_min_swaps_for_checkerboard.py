import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
MATRIX_SIZE_N = 6
DATASET_SIZE = 500000
# 控制“打乱”的程度，即从完美棋盘开始，交换多少次
MAX_SCRAMBLE_STEPS = 2 * (MATRIX_SIZE_N - 1)

TRAIN_FILE = f'chessboard_smart_gen_n{MATRIX_SIZE_N}_train.jsonl'
EVAL_FILE = f'chessboard_smart_gen_n{MATRIX_SIZE_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 (与之前相同) ---
# ==============================================================================
INPUT_LEN = MATRIX_SIZE_N * MATRIX_SIZE_N
MAX_MOVES = 2 * (MATRIX_SIZE_N - 1)
OUTPUT_BITS = math.ceil(math.log2(MAX_MOVES + 2))

print("=" * 70)
print(f"     “变为棋盘” - 智能数据集生成器")
print("=" * 70)
print(f"矩阵大小: {MATRIX_SIZE_N}x{MATRIX_SIZE_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (0代表不可行)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：智能数据生成 ---
# ==============================================================================
all_set=set()
def generate_solvable_grid_and_solution(n, max_scramble_steps):
    """
    智能生成一个可解的grid，并直接知道其最优解之一。
    """
    # 1. 创建一个完美的棋盘
    board = [[(i + j) % 2 for j in range(n)] for i in range(n)]

    # 随机决定初始棋盘是0开头还是1开头
    if random.random() < 0.5:
        board = [[1 - cell for cell in row] for row in board]

    # 2. 随机打乱行和列的顺序，这保证了矩阵依然是可解的
    row_perm = list(range(n))
    col_perm = list(range(n))
    random.shuffle(row_perm)
    random.shuffle(col_perm)

    scrambled_board = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            scrambled_board[i][j] = board[row_perm[i]][col_perm[j]]

    # 3. 计算这个打乱的排列，恢复到有序状态，需要多少次“相邻交换”
    # 这就是计算“逆序对”的数量
    def count_swaps(perm):
        inversions = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]:
                    inversions += 1
        return inversions

    # 注意：这里的最优解是“任意交换”，而不是“相邻交换”。
    # 将一个排列恢复成有序，最少交换次数是 n - 环的数量。
    # 这依然很复杂。让我们用一个更简单、更直接的逻辑。
    # 我们直接生成一个随机的、但满足“行列数量约束”的矩阵。

    while True:
        while True:
            grid = [[random.choice([0, 1]) for _ in range(n)] for _ in range(n)]
            grid_tuple = tuple(tuple(x) for x in grid)
            if grid_tuple not in all_set:
                all_set.add(grid_tuple)
                break

        # 检查基本的数量约束
        if n % 2==0:
            # 每行每列的1都必须是 n/2
            valid = True
            for i in range(n):
                if sum(grid[i])!=n / 2: valid = False; break
                if sum(row[i] for row in grid)!=n / 2: valid = False; break
            if valid: return grid  # 找到了一个可能的解
        else:  # n is odd
            # 每行每列的1和0数量差1
            valid = True
            for i in range(n):
                if abs(sum(grid[i]) - (n / 2)) > 0.5: valid = False; break
                if abs(sum(row[i] for row in grid) - (n / 2)) > 0.5: valid = False; break
            if valid: return grid


def solve_chessboard_puzzle_leetcode(board):
    # ... (省略，与上一版完全相同的 LeetCode 求解器)
    n = len(board)
    for i in range(n):
        for j in range(n):
            if board[0][0] ^ board[i][0] ^ board[0][j] ^ board[i][j]==1: return -1
    row_sum, col_sum = sum(board[0]), sum(board[i][0] for i in range(n))
    if not (n // 2 <= row_sum <= (n + 1) // 2) or not (n // 2 <= col_sum <= (n + 1) // 2): return -1
    row_mask, col_mask = sum(board[0][i] << i for i in range(n)), sum(board[i][0] << i for i in range(n))
    row_cnt = sum(1 for i in range(n) if sum(board[i][j] << j for j in range(n))==row_mask)

    def get_moves(mask, count, n):
        ones = bin(mask).count('1')
        if n & 1:
            if abs(n - 2 * ones)!=1 or abs(n - 2 * count)!=1: return -1
            if ones==n // 2:
                return (n // 2 - bin(mask & int('101010101010101010101010101010', 2) & ((1 << n) - 1)).count('1'))
            else:
                return ((n + 1) // 2 - bin(mask & int('010101010101010101010101010101', 2) & ((1 << n) - 1)).count('1'))
        else:
            if ones!=n / 2 or count!=n / 2: return -1
            swaps1 = n // 2 - bin(mask & int('101010101010101010101010101010', 2) & ((1 << n) - 1)).count('1')
            swaps2 = n // 2 - bin(mask & int('010101010101010101010101010101', 2) & ((1 << n) - 1)).count('1')
            return min(swaps1, swaps2)

    row_moves = get_moves(row_mask, row_cnt, n)
    col_moves = get_moves(col_mask, n - row_cnt, n)
    if row_moves==-1 or col_moves==-1: return -1
    return row_moves + col_moves


# ==============================================================================
# --- 4. 主生成函数 (使用智能生成) ---
# ==============================================================================
def generate_datasets(num_samples, n):
    print("\n--- 开始生成高质量数据集 ---")

    records = []
    attempts = 0
    while len(records) < num_samples and attempts < num_samples * 20:  # 增加尝试上限
        attempts += 1
        # 智能地生成一个更可能“有解”的矩阵
        grid = generate_solvable_grid(n)

        min_swaps = solve_chessboard_puzzle_leetcode(grid)

        # 我们只保留那些真正有解的，和少数无解的，让数据分布更健康
        if min_swaps==-1 and random.random() < 0.8:  # 大幅降低无解样本的比例
            continue

        input_str = "".join(str(cell) for row in grid for cell in row)
        output_label = min_swaps + 1 if min_swaps!=-1 else 0
        output_binary_str = format(output_label, f'0{OUTPUT_BITS}b')
        output_multilabel = [int(bit) for bit in output_binary_str]

        records.append({"input": input_str, "output": output_multilabel})

        if len(records) % 1000==0 and len(records) > 0:
            print(f"已生成 {len(records)} / {num_samples} 条有效数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # ... (省略写入文件的逻辑)
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
# --- 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    # 需要一个更智能的`generate_solvable_grid`
    def generate_solvable_grid(n):
        # 从一个完美棋盘开始，进行随机交换
        board = [[(i + j) % 2 for j in range(n)] for i in range(n)]
        if random.random() < 0.5: board = [[1 - c for c in r] for r in board]

        # 随机交换几行
        for _ in range(random.randint(0, n * 2)):
            r1, r2 = random.randint(0, n - 1), random.randint(0, n - 1)
            board[r1], board[r2] = board[r2], board[r1]

        # 随机交换几列
        for _ in range(random.randint(0, n * 2)):
            c1, c2 = random.randint(0, n - 1), random.randint(0, n - 1)
            for i in range(n):
                board[i][c1], board[i][c2] = board[i][c2], board[i][c1]
        return board


    generate_datasets(DATASET_SIZE, MATRIX_SIZE_N)