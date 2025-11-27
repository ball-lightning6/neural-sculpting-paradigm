import json
from collections import deque
import math
import sys
import time

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 你可以在这里指定汉诺塔的盘子数
HANOI_N = 12

# 文件名反映了这是一个特殊的、困难的路径
TRAIN_FILE = f'hanoi_n{HANOI_N}_true_twisted_path_train.jsonl'
EVAL_FILE = f'hanoi_n{HANOI_N}_true_twisted_path_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 ---
# ==============================================================================
# 输出动作编码：为6种可能的移动(from, to)分配唯一的ID
ACTION_MAP = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}
NUM_ACTIONS = len(ACTION_MAP)

# 输入状态编码
TOKEN_MAP = {i: i for i in range(HANOI_N + 2)}  # 1-N是盘子, N+1是SEP, 0是PAD
SEP_TOKEN_ID = HANOI_N + 1
PAD_TOKEN_ID = 0
# 输入序列的固定Token长度 = N个盘子 + (3-1)个分隔符
INPUT_TOKEN_LEN = HANOI_N + 2


# ==============================================================================
# --- 3. 核心逻辑：环境类 ---
# ==============================================================================
class HanoiEnv:
    """一个健壮的、用于求解和模拟汉诺塔问题的环境。"""

    def __init__(self, n):
        self.n = n
        self.num_pegs = 3
        # 标准起点：所有盘子在柱子0
        self.initial_state = self.get_state_tuple([list(range(n, 0, -1)), [], []])
        # 标准终点：所有盘子在柱子2
        self.goal_state = self.get_state_tuple([[], [], list(range(n, 0, -1))])

    def get_state_tuple(self, pegs_list):
        """将列表形式的盘面 [[3,2,1],[],[]] 转换为可哈希的元组 ((3,2,1),(),())"""
        return tuple(tuple(p) for p in pegs_list)

    def get_state_list(self, pegs_tuple):
        """将元组形式的盘面转回列表"""
        return [list(p) for p in pegs_tuple]

    def get_valid_moves(self, state):
        """获取当前状态下的所有合法移动 (from_peg, to_peg)"""
        moves = []
        pegs = self.get_state_list(state)
        for i in range(self.num_pegs):
            if not pegs[i]: continue  # 源柱子不能为空

            disk_to_move = pegs[i][0]  # 能移动的只有顶部的盘子

            for j in range(self.num_pegs):
                if i==j: continue  # 不能移动到自己身上

                # 目标柱子要么是空的，要么顶部的盘子比要移动的盘子大
                if not pegs[j] or pegs[j][0] > disk_to_move:
                    moves.append((i, j))
        #print(moves)
        return moves

    def apply_move(self, state, move):
        """执行一个移动，并返回新的状态元组"""
        from_p, to_p = move
        pegs = self.get_state_list(state)
        disk = pegs[from_p].pop(0)
        #pegs[to_p].insert(0, disk)
        pegs[to_p].append(disk)
        #print(self.get_state_tuple(pegs))
        return self.get_state_tuple(pegs)


# ==============================================================================
# --- 4. 编码与路径查找函数 ---
# ==============================================================================
def state_to_input_ids(state, env):
    """将盘面状态转换为固定长度的、填充过的整数ID列表"""
    tokens = []
    state_list = env.get_state_list(state)
    for peg_idx in range(env.num_pegs):
        # 我们从下到上编码，更符合堆栈的直觉
        tokens.extend(list(reversed(state_list[peg_idx])))
        if peg_idx < env.num_pegs - 1:
            tokens.append(SEP_TOKEN_ID)

    padded_tokens = tokens + [PAD_TOKEN_ID] * (INPUT_TOKEN_LEN - len(tokens))
    return padded_tokens


def find_path(env, start, end):
    """一个通用的BFS求解器，寻找从任意起点到任意终点的最短路径"""
    if start==end: return []
    queue = deque([(start, [])])  # (state, path_of_moves)
    visited = {start}

    while queue:
        current_state, path = queue.popleft()
        for move in env.get_valid_moves(current_state):
            next_state = env.apply_move(current_state, move)
            if next_state==end:
                return path + [move]
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
    return None


# ==============================================================================
# --- 5. 数据集生成主函数 ---
# ==============================================================================
def generate_twisted_path_dataset(env):
    print("=" * 60)
    print(f"汉诺塔 N={env.n} '扭曲路径' 数据集生成器")
    print("=" * 60)

    # 1. 精确地构建你所定义的、正确的“扭曲起点”
    # “n-1的盘子在最右侧柱子(peg 2)，最大盘子(n)在最左侧柱子(peg 0)”
    twisted_start_list = [[] for _ in range(env.num_pegs)]
    twisted_start_list[0] = [env.n]
    twisted_start_list[2] = list(range(env.n - 1, 0, -1))
    twisted_start_state = env.get_state_tuple(twisted_start_list)

    # 目标依然是标准终点
    goal_state = env.goal_state

    print(f"已定义正确的'扭曲起点': {twisted_start_state}")
    print(f"标准终点: {goal_state}")

    # 2. 找到从这个起点到标准终点的唯一最优路径
    print("\n正在搜索路径...")
    start_time = time.time()
    path_moves = find_path(env, twisted_start_state, goal_state)
    end_time = time.time()

    if path_moves is None:
        print("严重错误：未能找到路径！请检查环境或求解器逻辑。")
        return

    print(f"路径搜索成功！该'扭曲路径'的长度为 {len(path_moves)} 步。耗时: {end_time - start_time:.4f}秒")

    # 3. 将这条路径上的所有 (State, Action) 对转换为数据集
    print("\n正在生成数据集...")
    dataset = []
    current_state = twisted_start_state
    for move in path_moves:
        dataset.append({
            "input": state_to_input_ids(current_state, env),
            "output": ACTION_MAP[move]
        })
        current_state = env.apply_move(current_state, move)

    # 4. 写入文件
    print(f"\n正在将 {len(dataset)} 条路径数据写入文件...")
    # 训练集和评估集使用相同的数据，因为我们是想看模型能否“背下”并“理解”这条困难的路径
    with open(TRAIN_FILE, 'w') as f:
        for rec in dataset: f.write(json.dumps(rec) + '\n')
    with open(EVAL_FILE, 'w') as f:
        for rec in dataset: f.write(json.dumps(rec) + '\n')

    print(f"数据集 '{TRAIN_FILE}' 和 '{EVAL_FILE}' 生成完成！")


# ==============================================================================
# --- 6. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    env = HanoiEnv(HANOI_N)
    generate_twisted_path_dataset(env)