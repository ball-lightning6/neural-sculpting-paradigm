import json
import random
import sys
import time
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
CHECKERS_N = 4  # 我们从N=3开始，因为它的状态空间足够小，可以进行完整分析
# 对于N=3，最优路径长度是15

# ==============================================================================
# --- 2. 环境类与编码函数 (与之前相同，为完整性保留) ---
# ==============================================================================
TOKEN_MAP = {'R': 0, 'B': 1, '_': 2}
BOARD_SIZE = 2 * CHECKERS_N + 1


class CheckersEnv:
    # ... (省略，与上一版完全相同) ...
    def __init__(self, n):
        self.n, self.board_size, self.initial_state, self.goal_state = n, 2 * n + 1, tuple(
            ['R'] * n + ['_'] + ['B'] * n), tuple(['B'] * n + ['_'] + ['R'] * n)

    def get_valid_moves(self, state_tuple):
        board, moves, size = list(state_tuple), [], self.board_size
        for i in range(size):
            if board[i]=='R' and i + 1 < size and board[i + 1]=='_': moves.append((i, i + 1))
            if board[i]=='R' and i + 2 < size and board[i + 1]=='B' and board[i + 2]=='_': moves.append((i, i + 2))
            if board[i]=='B' and i - 1 >= 0 and board[i - 1]=='_': moves.append((i, i - 1))
            if board[i]=='B' and i - 2 >= 0 and board[i - 1]=='R' and board[i - 2]=='_': moves.append((i, i - 2))
        return moves

    def apply_move(self, state_tuple, move):
        from_pos, to_pos = move
        new_board = list(state_tuple)
        new_board[to_pos], new_board[from_pos] = new_board[from_pos], '_'
        return tuple(new_board)


def state_seq_to_input(state_seq):
    return [[TOKEN_MAP[token] for token in state] for state in state_seq]


def move_seq_to_output(move_seq):
    # 输出是移动的棋子位置索引序列
    return [move[0] for move in move_seq]


# ==============================================================================
# --- 3. 核心逻辑：全局路径求解器 (全新，极其关键) ---
# ==============================================================================
def find_all_paths_and_layers(env):
    """
    通过一次BFS，找到所有可达状态，并记录到每个状态的最短路径。
    这是整个脚本的核心引擎。
    """
    print("开始进行全局状态空间探索...")
    start_time = time.time()

    # 记录从起点到每个状态的最短路径
    paths_from_start = {env.initial_state: []}
    queue = deque([env.initial_state])

    # 我们用一个列表来存储所有探索到的状态，以保持顺序
    # 这对于后续分层和找到叶子节点很重要
    discovered_states = [env.initial_state]

    head = 0
    while head < len(discovered_states):
        current_state = discovered_states[head];
        head += 1

        for move in env.get_valid_moves(current_state):
            next_state = env.apply_move(current_state, move)
            if next_state not in paths_from_start:
                # 记录路径并添加到队列
                new_path = paths_from_start[current_state] + [move]
                paths_from_start[next_state] = new_path
                discovered_states.append(next_state)

    print(f"探索完成。共发现 {len(paths_from_start)} 个可达状态。耗时: {time.time() - start_time:.4f}秒。")
    return paths_from_start


# ==============================================================================
# --- 4. 数据集生成主函数 (实现你的所有要求) ---
# ==============================================================================
def generate_all_datasets(env):
    # 首先，获取包含所有最优路径的“全局地图”
    all_optimal_paths_map = find_all_paths_and_layers(env)

    # 1. 数据集A: 创世史诗 (从标准起点到终点)
    print("\n--- [A] 生成'创世史诗'数据集 ---")
    optimal_path_moves = all_optimal_paths_map.get(env.goal_state)
    if not optimal_path_moves:
        print("错误：未能找到从起点到终点的最优路径！")
        return

    path_a_states = []
    current_state = env.initial_state
    for move in optimal_path_moves:
        path_a_states.append(current_state)
        current_state = env.apply_move(current_state, move)

    dataset_a = [{
        "input": state_seq_to_input(path_a_states),
        "output": move_seq_to_output(optimal_path_moves)
    }]
    print(f"已生成，路径长度: {len(optimal_path_moves)} 步。")
    with open(f'checkers_n{env.n}_path_optimal.jsonl', 'w') as f:
        f.write(json.dumps(dataset_a[0]) + '\n')

    # 2. 数据集B: 英雄列传 (从中间点到终点)
    print("\n--- [B] 生成'英雄列传'数据集 ---")
    mid_point_idx = len(optimal_path_moves) // 2
    mid_state = path_a_states[mid_point_idx]
    path_b_moves = all_optimal_paths_map[env.goal_state][mid_point_idx:]
    path_b_states = path_a_states[mid_point_idx:]

    dataset_b = [{
        "input": state_seq_to_input(path_b_states),
        "output": move_seq_to_output(path_b_moves)
    }]
    print(f"已生成，路径长度: {len(path_b_moves)} 步。")
    with open(f'checkers_n{env.n}_path_sub_optimal.jsonl', 'w') as f:
        f.write(json.dumps(dataset_b[0]) + '\n')

    # 3. 数据集C: 凡人悲歌 (所有非最优路径的混合)
    print("\n--- [C] 生成'凡人悲歌'数据集 (混合长度) ---")
    optimal_path_states = set(path_a_states)
    fork_states = set(all_optimal_paths_map.keys()) - optimal_path_states - {env.goal_state}
    dataset_c = []
    for start_state in fork_states:
        # 我们需要一个能从任意点找到目标路径的求解器
        # 注意：这里的all_optimal_paths_map是从initial_state出发的，不能直接用
        # 我们需要重新为每个fork_state求解
        path = get_path_from_a_to_b(env, start_state, env.goal_state)
        if path:
            path_states = []
            current = start_state
            for move in path:
                path_states.append(current)
                current = env.apply_move(current, move)
            dataset_c.append({
                "input": state_seq_to_input(path_states),
                "output": move_seq_to_output(path)
            })
    print(f"已生成 {len(dataset_c)} 条非最优路径。")
    with open(f'checkers_n{env.n}_path_forked_mixed.jsonl', 'w') as f:
        for rec in dataset_c: f.write(json.dumps(rec) + '\n')

    # 4. 数据集D: 精英悲歌 (与最优路径等长的非最优路径)
    print("\n--- [D] 生成'精英悲歌'数据集 (固定长度) ---")
    optimal_len = len(optimal_path_moves)
    dataset_d = [rec for rec in dataset_c if len(rec['output'])==optimal_len]
    print(f"从混合路径中，筛选出 {len(dataset_d)} 条长度为 {optimal_len} 的路径。")
    with open(f'checkers_n{env.n}_path_forked_fixed_len.jsonl', 'w') as f:
        for rec in dataset_d: f.write(json.dumps(rec) + '\n')


# ==============================================================================
# --- 5. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    env = CheckersEnv(CHECKERS_N)
    print("=" * 60)
    print("跳棋交换 - 序列学习对比实验数据集生成器")
    print(f"参数: N={CHECKERS_N}")
    print("=" * 60)


    # 需要一个通用的求解器
    def get_path_from_a_to_b(env, start_state, goal_state):
        queue = deque([(start_state, [])])
        visited = {start_state}
        while queue:
            current_state, path = queue.popleft()
            if current_state==goal_state: return path
            for move in env.get_valid_moves(current_state):
                next_state = env.apply_move(current_state, move)
                if next_state not in visited:
                    visited.add(next_state)
                    new_path = path + [move]
                    queue.append((next_state, new_path))
        return None


    generate_all_datasets(env)