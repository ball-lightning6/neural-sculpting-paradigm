import json
import random
import math
from functools import lru_cache

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
MAZE_M = 5
MAZE_N = 6
DATASET_SIZE = 100000

TRAIN_FILE = f'valid_parentheses_path_n{MAZE_M}x{MAZE_N}_final_train.jsonl'
EVAL_FILE = f'valid_parentheses_path_n{MAZE_M}x{MAZE_N}_final_eval.jsonl'


# ==============================================================================
# --- 2. 核心逻辑：求解器 (LeetCode官方题解的Python实现) ---
# ==============================================================================
class PathValidator:
    def __init__(self, grid):
        self.grid = grid
        self.m = len(grid)
        self.n = len(grid[0])
        # 使用Python内置的lru_cache来实现记忆化
        self.dfs.cache_clear()

    @lru_cache(maxsize=None)
    def dfs(self, x, y, c):
        # c: 当前路径上，未配对的左括号的数量

        # 1. 剪枝：如果剩余的格子数，比未配对的左括号还少，
        # 那么即使剩下的全是右括号，也不可能完全配对成功。
        remaining_steps = (self.m - 1 - x) + (self.n - 1 - y)
        if c > remaining_steps:
            return False

        # 2. 更新当前括号数
        c += 1 if self.grid[x][y]=='(' else -1

        # 3. 检查括号数是否合法（不能为负）
        if c < 0:
            return False

        # 4. 到达终点
        if x==self.m - 1 and y==self.n - 1:
            # 必须恰好配对完
            return c==0

        # 5. 递归探索
        # 向下走
        if x < self.m - 1:
            if self.dfs(x + 1, y, c):
                return True
        # 向右走
        if y < self.n - 1:
            if self.dfs(x, y + 1, c):
                return True

        return False

    def hasValidPath(self) -> bool:
        # 初始剪枝
        if (self.m + self.n - 1) % 2!=0: return False
        if self.grid[0][0]==')' or self.grid[self.m - 1][self.n - 1]=='(':
            return False

        return self.dfs(0, 0, 0)

all_set=set()
# ==============================================================================
# --- 3. 数据集生成主函数 ---
# ==============================================================================
def generate_datasets(num_samples, m, n):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        while True:
            grid_chars = [[random.choice(['(', ')']) for _ in range(n)] for _ in range(m)]
            grid_tuple = tuple(map(tuple, grid_chars))
            # print(grid_tuple)
            if grid_tuple not in all_set:
                all_set.add(grid_tuple)
                break
        # 编码输入
        grid_bits = [[0 if cell=='(' else 1 for cell in row] for row in grid_chars]
        input_str = "".join(str(cell) for row in grid_bits for cell in row)

        # 计算答案
        validator = PathValidator(grid_chars)
        result = validator.hasValidPath()
        print(result)

        # 输出是单个bit
        output_label = [1 if result else 0]

        records.append({"input": input_str, "output": output_label})

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
            for record in data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 4. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    # 打印参数
    INPUT_LEN = MAZE_M * MAZE_N
    OUTPUT_LEN = 1
    print("=" * 70)
    print(f"     合法括号路径检查 - 数据集生成器 (LeetCode解法最终版)")
    print("=" * 70)
    print(f"迷宫大小: {MAZE_M}x{MAZE_N}")
    print(f"输入格式: {INPUT_LEN}个'0'/'1'")
    print(f"输出格式: {OUTPUT_LEN}个bit (1=存在, 0=不存在)")
    print("=" * 70)

    generate_datasets(DATASET_SIZE, MAZE_M, MAZE_N)