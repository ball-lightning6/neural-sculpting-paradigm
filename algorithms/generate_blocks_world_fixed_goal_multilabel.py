import json
import random
import math
import sys
import time
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
BLOCKS_N = 3
NUM_STACKS = 3
# 我们的数据集将包含所有可解状态，所以大小是固定的
# DATASET_SIZE 这个参数变得没有意义了
TRAIN_FILE = f'blocks_n{BLOCKS_N}_multihot_policy_train.jsonl'
EVAL_FILE = f'blocks_n{BLOCKS_N}_multihot_policy_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 (输出是多标签) ---
# ==============================================================================
ACTIONS = [(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i!=j]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)  # 6

print(ACTIONS)

# 输入编码部分保持不变
BITS_PER_TOKEN = math.ceil(math.log2(BLOCKS_N + 2))
SEP_TOKEN_ID = BLOCKS_N + 1
PAD_TOKEN_ID = 0
SINGLE_STATE_TOKEN_LEN = BLOCKS_N + (NUM_STACKS - 1)
TOTAL_INPUT_BITS = SINGLE_STATE_TOKEN_LEN * 1  # 输入只有一个状态


# ==============================================================================
# --- 3. 核心逻辑：环境类 (无变化) ---
# ==============================================================================
class BlocksWorld:
    # ... (省略完整代码，与上一版完全相同)
    def __init__(self, num_blocks, num_stacks):
        self.num_blocks, self.num_stacks = num_blocks, num_stacks
        goal_stack = tuple(range(num_blocks, 0, -1))
        goal_list = [[] for _ in range(num_stacks)]
        goal_list[0] = list(goal_stack)
        self.goal_state = self.state_to_tuple(goal_list)

    def get_valid_moves(self, state):
        valid_moves, state_list = [], self.state_from_tuple(state)
        for i in range(self.num_stacks):
            if state_list[i]:
                for j in range(self.num_stacks):
                    if i!=j: valid_moves.append((i, j))
        return valid_moves

    def apply_move(self, state, move):
        from_s, to_s = move
        new_state_list = self.state_from_tuple(state)
        if not new_state_list[from_s]: return None
        block = new_state_list[from_s].pop(0)
        new_state_list[to_s].insert(0, block)
        return self.state_to_tuple(new_state_list)

    def state_to_tuple(self, state_list):
        return tuple(tuple(stack) for stack in state_list)

    def state_from_tuple(self, state_tuple):
        return [list(stack) for stack in state_tuple]


# ==============================================================================
# --- 4. 编码与数据集生成 (全新核心逻辑) ---
# ==============================================================================
def state_to_binary(state_tuple, num_blocks, num_stacks):
    # ... (省略完整代码，与上一版相同)
    tokens, target_len = [], num_blocks + (num_stacks - 1)
    for i, stack in enumerate(state_tuple): tokens.extend(list(reversed(stack)));
    if i < num_stacks - 1: tokens.append(SEP_TOKEN_ID)


    binary_string, padded_tokens = "", tokens + [PAD_TOKEN_ID] * (target_len - len(tokens))
    for token_id in padded_tokens: binary_string += format(token_id, f'0{BITS_PER_TOKEN}b')
    return binary_string


def generate_multihot_policy_dataset(env, output_path, is_eval=False):
    log_prefix = f"[{'评估集' if is_eval else '训练集'}]"
    print(f"\n{log_prefix} --- 开始生成多重最优解策略数据集 ---")
    print(f"{log_prefix} 输出文件: {output_path}")

    # 1. 反向BFS，计算所有可达状态到终点的最短距离
    print(f"{log_prefix} 步骤 1: 反向搜索并计算所有状态到终点的距离...")
    start_time = time.time()

    # dist_map 存储: state -> distance_from_goal
    dist_map = {env.goal_state: 0}
    queue = deque([env.goal_state])

    while queue:
        current_state = queue.popleft()
        current_dist = dist_map[current_state]

        # 寻找前驱状态
        current_state_list = env.state_from_tuple(current_state)
        for from_s in range(env.num_stacks):
            for to_s in range(env.num_stacks):
                if from_s==to_s: continue
                if current_state_list[to_s]:
                    prev_state = env.apply_move(env.state_to_tuple(current_state_list), (to_s, from_s))
                    if prev_state not in dist_map:
                        dist_map[prev_state] = current_dist + 1
                        queue.append(prev_state)

    end_time = time.time()
    print(f"{log_prefix} 距离地图构建完毕。共找到 {len(dist_map)} 个可达状态。耗时: {time.time() - start_time:.4f} 秒。")

    # 2. 遍历所有状态，找到所有最优动作，生成多标签输出
    print(f"{log_prefix} 步骤 2: 生成多标签最优策略...")
    dataset = []
    for state, dist in dist_map.items():
        if state==env.goal_state: continue

        optimal_actions = []
        for move in env.get_valid_moves(state):
            next_state = env.apply_move(state, move)
            # 如果下一步的状态存在于我们的地图中，并且距离比当前状态近1，那它就是最优动作
            if next_state in dist_map and dist_map[next_state]==dist - 1:
                optimal_actions.append(move)

        if optimal_actions:
            # 创建一个长度为 NUM_ACTIONS 的、全为0的列表
            output_labels = [0] * NUM_ACTIONS
            # 将所有最优动作对应的位置设为1
            for move in optimal_actions:
                output_labels[ACTION_MAP[move]] = 1

            input_binary = state_to_binary(state, env.num_blocks, env.num_stacks)
            print(state, env.num_blocks, env.num_stacks)
            for l,a in zip(output_labels,ACTIONS):
                if l:
                    print(a, end=' ')
            print()
            dataset.append({"input": input_binary, "output": output_labels})

    print(f"{log_prefix} 策略生成完毕。共找到 {len(dataset)} 条有效策略。")

    # 3. 写入文件
    if is_eval:
        random.shuffle(dataset)
        eval_size = max(10, len(dataset) // 10)
        dataset = dataset[:eval_size]

    print(f"{log_prefix} 正在将 {len(dataset)} 条记录写入 '{output_path}'...")
    with open(output_path, 'w') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')
    print(f"{log_prefix} 数据集生成完成。")


# ==============================================================================
# --- 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    env = BlocksWorld(BLOCKS_N, NUM_STACKS)

    print("=" * 60)
    print("积木世界数据集生成器 (多重最优解 - 多标签版)")
    print(f"参数: N={BLOCKS_N}, Stacks={NUM_STACKS}")
    print(f"输入格式: {TOTAL_INPUT_BITS} bits (仅当前状态)")
    print(f"输出格式: {NUM_ACTIONS} 个多标签二分类")
    print("=" * 60)

    generate_multihot_policy_dataset(env, TRAIN_FILE, is_eval=False)
    # generate_multihot_policy_dataset(env, EVAL_FILE, is_eval=True)