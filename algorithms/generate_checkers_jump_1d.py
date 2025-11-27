import json
import random
import sys
import time
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
CHECKERS_N = 15  # 每种颜色3个棋子
TRAIN_FILE = f'checkers_n{CHECKERS_N}_full_policy_train.jsonl'
EVAL_FILE = f'checkers_n{CHECKERS_N}_full_policy_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 ---
# ==============================================================================
# 输入编码
TOKEN_MAP = {'R': 0, 'B': 1, '_': 2}
BOARD_SIZE = 2 * CHECKERS_N + 1

# 输出编码：7分类，标签是0到6，代表要移动的棋子的位置索引
NUM_ACTIONS = BOARD_SIZE


# ==============================================================================
# --- 3. 核心逻辑：环境类 ---
# ==============================================================================
class CheckersEnv:
    """一个健壮的、用于求解和模拟“跳棋交换”问题的环境。"""

    def __init__(self, n):
        self.n = n
        self.board_size = 2 * n + 1
        self.initial_state = tuple(['R'] * n + ['_'] + ['B'] * n)
        self.goal_state = tuple(['B'] * n + ['_'] + ['R'] * n)

    def apply_move(self, state_tuple, move):
        """执行移动并返回新状态"""
        from_pos, to_pos = move
        new_board = list(state_tuple)
        # 交换位置
        new_board[to_pos], new_board[from_pos] = new_board[from_pos], new_board[to_pos]
        return tuple(new_board)


# ==============================================================================
# --- 4. 数据集生成主函数 (高效反向BFS版) ---
# ==============================================================================
def generate_full_policy_dataset(env, output_path, is_eval=False):
    log_prefix = f"[{'评估集' if is_eval else '训练集'}]"
    print(f"\n{log_prefix} --- 开始为 N={env.n} 生成纯净策略数据集 ---")
    print(f"{log_prefix} 输出文件: {output_path}")

    # 1. 从目标状态开始，进行反向广度优先搜索
    print(f"{log_prefix} 步骤 1: 通过反向搜索构建最优策略地图...")
    start_time = time.time()

    # policy_map 将存储: state -> optimal_action_from_this_state
    policy_map = {}

    queue = deque([env.goal_state])
    # visited 集合是去重的核心，确保每个状态只被处理一次
    visited = {env.goal_state}

    while queue:
        current_state = queue.popleft()

        # 核心逻辑：寻找所有能一步“正向”移动到达 current_state 的“前驱状态”
        for i in range(env.board_size):
            if current_state[i]!='_':
                piece = current_state[i]

                # 检查反向滑动
                # 如果是红棋(R)，它只能从左边(i-1)滑过来
                if piece=='R' and i > 0 and current_state[i - 1]=='_':
                    prev_state = env.apply_move(current_state, (i, i - 1))
                    if prev_state not in visited:
                        visited.add(prev_state)
                        queue.append(prev_state)
                        policy_map[prev_state] = (i - 1, i)  # 记录正向移动

                # 如果是蓝棋(B)，它只能从右边(i+1)滑过来
                if piece=='B' and i < env.board_size - 1 and current_state[i + 1]=='_':
                    prev_state = env.apply_move(current_state, (i, i + 1))
                    if prev_state not in visited:
                        visited.add(prev_state)
                        queue.append(prev_state)
                        policy_map[prev_state] = (i + 1, i)

                # 检查反向跳跃
                # 如果是红棋(R)，它只能从左边(i-2)跳过来
                if piece=='R' and i > 1 and current_state[i - 2]=='_' and current_state[i - 1]=='B':
                    prev_state = env.apply_move(current_state, (i, i - 2))
                    if prev_state not in visited:
                        visited.add(prev_state)
                        queue.append(prev_state)
                        policy_map[prev_state] = (i - 2, i)

                # 如果是蓝棋(B)，它只能从右边(i+2)跳过来
                if piece=='B' and i < env.board_size - 2 and current_state[i + 2]=='_' and current_state[i + 1]=='R':
                    prev_state = env.apply_move(current_state, (i, i + 2))
                    if prev_state not in visited:
                        visited.add(prev_state)
                        queue.append(prev_state)
                        policy_map[prev_state] = (i + 2, i)

    end_time = time.time()
    print(f"{log_prefix} 策略地图生成完毕，找到 {len(policy_map)} 条唯一策略。耗时: {end_time - start_time:.4f} 秒。")

    # 2. 将策略地图转换为数据集
    print(f"{log_prefix} 步骤 2: 将策略地图转换为数据集...")
    dataset = []
    for state, optimal_move in policy_map.items():
        from_pos, _ = optimal_move
        action_label = from_pos  # 输出就是要移动的棋子的位置索引
        input_ids = [TOKEN_MAP[token] for token in state]
        dataset.append({"input": input_ids, "output": action_label})

    # 3. 写入文件
    if is_eval:
        random.shuffle(dataset)
        # 评估集使用大约20%的数据，或者至少3条
        eval_size = max(3, len(dataset) // 5)
        dataset = dataset[:eval_size]

    print(f"{log_prefix} 正在将 {len(dataset)} 条不重复记录写入 '{output_path}'...")
    with open(output_path, 'w') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')
    print(f"{log_prefix} 数据集生成完成。")


# ==============================================================================
# --- 5. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    env = CheckersEnv(CHECKERS_N)

    print("=" * 60)
    print("跳棋交换数据集生成器 (全局策略 - 最终性能版)")
    print(f"参数: N={CHECKERS_N} (每种颜色棋子数)")
    print(f"输入格式: 长度为 {BOARD_SIZE} 的整数序列 (0=R, 1=B, 2=_)")
    print(f"输出格式: {NUM_ACTIONS} 分类问题 (要移动的棋子位置索引 0-6)")
    print("=" * 60)

    # 生成训练集，包含所有可解状态的最优策略
    generate_full_policy_dataset(env, TRAIN_FILE, is_eval=False)

    # 生成评估集
    # generate_full_policy_dataset(env, EVAL_FILE, is_eval=True)