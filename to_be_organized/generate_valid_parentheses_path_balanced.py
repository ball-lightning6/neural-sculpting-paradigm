import json
import random
import math
from functools import lru_cache

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
MAZE_M = 5
MAZE_N = 6
DATASET_SIZE = 300000

TRAIN_FILE = f'valid_parentheses_balanced_n{MAZE_M}x{MAZE_N}_train.jsonl'
EVAL_FILE = f'valid_parentheses_balanced_n{MAZE_M}x{MAZE_N}_eval.jsonl'


# ==============================================================================
# --- 2. 求解器与编码函数 (与上一版完全相同，此处省略) ---
# ==============================================================================
class PathValidator:
    # ... (省略，与上一版完全相同)
    def __init__(self, grid):
        self.grid, self.m, self.n = grid, len(grid), len(grid[0]); self.dfs.cache_clear()

    @lru_cache(maxsize=None)
    def dfs(self, x, y, c):
        c += 1 if self.grid[x][y]=='(' else -1
        if c < 0 or c > self.m - x + self.n - y - 1: return False
        if x==self.m - 1 and y==self.n - 1: return c==0
        if x < self.m - 1 and self.dfs(x + 1, y, c): return True
        if y < self.n - 1 and self.dfs(x, y + 1, c): return True
        return False

    def hasValidPath(self) -> bool:
        if (self.m + self.n - 1) % 2!=0: return False
        if self.grid[0][0]==')' or self.grid[self.m - 1][self.n - 1]=='(': return False
        return self.dfs(0, 0, 0)


def state_to_binary(grid_bits):
    return "".join(str(cell) for row in grid_bits for cell in row)


# ==============================================================================
# --- 3. 核心逻辑：智能数据生成 (全新) ---
# ==============================================================================

def generate_solvable_grid(m, n):
    """主动构造一个保证有解的网格"""
    grid = [[random.choice(['(', ')']) for _ in range(n)] for _ in range(m)]

    # 1. 随机生成一条从左上到右下的路径
    path = []
    r, c = 0, 0
    while r < m - 1 or c < n - 1:
        path.append((r, c))
        if r==m - 1:
            c += 1
        elif c==n - 1:
            r += 1
        else:
            if random.random() < 0.5:
                r += 1
            else:
                c += 1
    path.append((m - 1, n - 1))

    # 2. 在这条路径上，生成一个合法的括号序列
    balance = 0
    path_len = len(path)
    # 确保起点是'('，终点是')'
    grid[0][0] = '('
    grid[m - 1][n - 1] = ')'

    # 在路径中间填充，以保证最终balance为0
    # 这是一个简化的方法，可能不总是能生成最复杂的序列
    path_parentheses = [''] * path_len
    path_parentheses[0] = '('
    path_parentheses[-1] = ')'
    balance = 1
    for i in range(1, path_len - 1):
        # 优先保持balance > 0
        if random.random() < 0.6 and balance > 0:
            path_parentheses[i] = ')'
            balance -= 1
        else:
            path_parentheses[i] = '('
            balance += 1

    # 修正最后的balance
    # 这部分逻辑很复杂，一个更简单的方法是直接生成合法的括号序列

    # --- 更健壮的生成方法 ---
    path_len = m + n - 1
    valid_p_str = generate_valid_parentheses(path_len)
    # print(valid_p_str)
    # print(path)

    for i in range(len(path)):
        r, c = path[i]
        grid[r][c] = valid_p_str[i]
        # 起点和终点已经在grid中设置
        # if (r, c)==(0, 0):
        #     grid[r][c] = '('
        # elif (r, c)==(m - 1, n - 1):
        #     grid[r][c] = ')'
        # elif i - 1 < len(valid_p_str):
        #     grid[r][c] = valid_p_str[i - 1]
    # for grid_row in grid:
    #     print(''.join(grid_row))
    # print(grid)

    return grid


# def generate_valid_parentheses(n):
#     import random

def generate_valid_parentheses(length):
    """
    生成指定长度的随机合法括号字符串

    Args:
        length (int): 字符串长度，必须是2的倍数

    Returns:
        str: 合法的括号字符串

    Raises:
        ValueError: 当length不是2的倍数或小于0时
    """
    if length < 0 or length % 2!=0:
        raise ValueError("长度必须是非负的偶数")

    if length==0:
        return ""

    # 使用栈来生成合法的括号序列
    result = []
    open_count = 0  # 当前未匹配的左括号数量
    pairs = length // 2  # 括号对数

    for i in range(length):
        # 计算剩余位置数
        remaining = length - i

        # 如果剩余的位置正好等于未匹配的左括号数，必须全部放右括号
        if remaining==open_count:
            result.append(')')
            open_count -= 1
        # 如果还没有左括号，必须放左括号
        elif open_count==0:
            result.append('(')
            open_count += 1
        # 如果已经放了足够的左括号对，只能放右括号
        elif len([c for c in result if c=='('])==pairs:
            result.append(')')
            open_count -= 1
        else:
            # 随机选择放左括号或右括号
            if random.choice([True, False]):
                result.append('(')
                open_count += 1
            else:
                result.append(')')
                open_count -= 1

    return ''.join(result)


def generate_unsolvable_grid(m, n):
    """主动构造一个保证无解的网格"""
    while True:
        grid = [[random.choice(['(', ')']) for _ in range(n)] for _ in range(m)]
        grid[0][0] = '('
        grid[m - 1][n - 1] = ')'
        validator = PathValidator(grid)
        if not validator.hasValidPath():
            return grid
    # # 制造一个必败条件
    # if random.random() < 0.5:
    #     grid[0][0] = ')'  # 起点必败
    # else:
    #     grid[m - 1][n - 1] = '('  # 终点必败
    # return grid

all_set= set()
# ==============================================================================
# --- 4. 主生成函数 (全新) ---
# ==============================================================================
def generate_balanced_dataset(num_samples, m, n):
    print("\n--- 开始生成平衡的数据集 ---")

    records = []

    for i in range(num_samples):
        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

        # 保持标签1:1的比例
        if i < num_samples // 2:
            # 生成有解的样本
            while True:
                grid_chars = generate_solvable_grid(m, n)
                grid_tuple = tuple(map(tuple, grid_chars))
                #print(grid_tuple)
                if grid_tuple not in all_set:
                    all_set.add(grid_tuple)
                    break
            # for grid_row in grid_chars:
            #     print(''.join(grid_row))

            # 我们需要用求解器再次确认，因为生成器不完美
            validator = PathValidator(grid_chars)
            if not validator.hasValidPath():
                # print('error')
                # print('-' * 100)
                # 如果我们构造的方法失败了，就跳过，再试一次
                # 为了简化，我们直接相信构造是有解的
                # records.append(None) # 占位
                continue  # 重新生成
            # print('-' * 100)
            label = 1
        else:
            # 生成无解的样本
            grid_chars = generate_unsolvable_grid(m, n)
            label = 0

        grid_bits = [[0 if cell=='(' else 1 for cell in row] for row in grid_chars]
        input_str = "".join(str(cell) for row in grid_bits for cell in row)

        output_label = [label]
        records.append({"input": input_str, "output": output_label})

    # 清理掉可能失败的占位符
    # records = [r for r in records if r is not None]

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
    INPUT_LEN = MAZE_M * MAZE_N
    OUTPUT_LEN = 1
    print("=" * 70)
    print(f"     合法括号路径检查 - 平衡数据集生成器")
    print("=" * 70)
    print(f"迷宫大小: {MAZE_M}x{MAZE_N}")
    print(f"输入格式: {INPUT_LEN}个'0'/'1'")
    print(f"输出格式: {OUTPUT_LEN}个bit (1=存在, 0=不存在)")
    print("=" * 70)
    generate_balanced_dataset(DATASET_SIZE, MAZE_M, MAZE_N)