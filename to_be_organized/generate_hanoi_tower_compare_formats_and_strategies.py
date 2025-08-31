import json
import random
import math
import sys
import time
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 警告：N > 8 会让全局策略生成变得非常慢，因为状态空间是3^N！
HANOI_N = 14
NUM_STACKS = 3

# 对于路径数据，我们可以生成大量重复采样
PATH_DATASET_SIZE = 200000
# 对于全局策略数据，其大小由状态空间决定 (3^N - 1)
POLICY_DATASET_SIZE = (3 ** HANOI_N) - 1

# --- 文件名定义 ---
# 1A: 路径数据 + SEP表示
PATH_SEP_TRAIN = f'hanoi_n{HANOI_N}_path_sep_train.jsonl'
PATH_SEP_EVAL = f'hanoi_n{HANOI_N}_path_sep_eval.jsonl'
# 1B: 路径数据 + 固定槽位表示
PATH_SLOTS_TRAIN = f'hanoi_n{HANOI_N}_path_slots_train.jsonl'
PATH_SLOTS_EVAL = f'hanoi_n{HANOI_N}_path_slots_eval.jsonl'
# 2A: 全局策略数据 + SEP表示
POLICY_SEP_TRAIN = f'hanoi_n{HANOI_N}_policy_sep_train.jsonl'
POLICY_SEP_EVAL = f'hanoi_n{HANOI_N}_policy_sep_eval.jsonl'
# 2B: 全局策略数据 + 固定槽位表示
POLICY_SLOTS_TRAIN = f'hanoi_n{HANOI_N}_policy_slots_train.jsonl'
POLICY_SLOTS_EVAL = f'hanoi_n{HANOI_N}_policy_slots_eval.jsonl'

# ==============================================================================
# --- 2. 编码与环境类 ---
# ==============================================================================
ACTIONS = [(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i!=j]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}


class HanoiEnv:
    def __init__(self, n, s=3):
        self.n, self.s = n, s
        goal_list = [[] for _ in range(s)]
        goal_list[-1] = list(range(n, 1, -1)) + [1]  # 正确的顺序
        self.goal_state = tuple(map(tuple, goal_list))

    def is_valid_state(self, state_tuple):
        for stack in state_tuple:
            for i in range(len(stack) - 1):
                if stack[i] < stack[i + 1]:
                    return False
        return True

    def apply_move(self, state_tuple, move):
        from_s, to_s = move
        state_list = [list(s) for s in state_tuple]
        if not state_list[from_s]: return None
        disk = state_list[from_s].pop()
        state_list[to_s].append(disk)
        return tuple(map(tuple, state_list))


# ==============================================================================
# --- 3. 编码函数 ---
# ==============================================================================
def state_to_binary_A(state_tuple, n, s):
    bits_per_token = math.ceil(math.log2(n + 2))
    sep_id, pad_id = n + 1, 0
    target_len = n + s - 1

    tokens = []
    for i in range(s):
        tokens.extend(list(reversed(state_tuple[i])))
        if i < s - 1: tokens.append(sep_id)

    padded_tokens = tokens + [pad_id] * (target_len - len(tokens))
    return "".join(format(tok_id, f'0{bits_per_token}b') for tok_id in padded_tokens)


def state_to_slots_B(state_tuple, n, s):
    slots = [0] * (n * s)
    for i in range(s):
        stack = state_tuple[i]
        for j in range(len(stack)):
            slots[i * n + j] = stack[j]
    #slots = ''.join([str(x).zfill(2) for x in slots])
    slots = ''.join([chr(x+97) for x in slots])
    return slots


# ==============================================================================
# --- 4. 数据集生成核心逻辑 ---
# ==============================================================================

def get_hanoi_solution_path(env):
    """生成最优解路径上的所有 (state, action) 对"""
    data_pairs = []
    pegs = [list(range(env.n, 0, -1)), [], []]

    def solver(k, src, dest, aux):
        if k > 0:
            solver(k - 1, src, aux, dest)
            current_state = tuple(map(tuple, [p[:] for p in pegs]))
            action = (src, dest)
            data_pairs.append({'state': current_state, 'action': action})
            disk = pegs[src].pop()
            pegs[dest].append(disk)
            solver(k - 1, aux, dest, src)

    solver(env.n, 0, 2, 1)
    return data_pairs


def get_hanoi_full_policy(env):
    """通过反向BFS生成所有可解状态的最优策略"""
    policy_map = {}
    queue = deque([env.goal_state])
    visited = {env.goal_state}

    while queue:
        current_state = queue.popleft()
        for from_s in range(env.s):
            for to_s in range(env.s):
                if from_s==to_s: continue

                state_list = [list(s) for s in current_state]
                if state_list[to_s]:
                    disk = state_list[to_s].pop()
                    state_list[from_s].append(disk)
                    prev_state = tuple(map(tuple, state_list))

                    if env.is_valid_state(prev_state) and prev_state not in visited:
                        visited.add(prev_state)
                        queue.append(prev_state)
                        policy_map[prev_state] = (from_s, to_s)
    return policy_map


def write_datasets(data_pairs, name, size):
    """将数据对转换为两种格式并写入文件"""
    train_A_path = globals()[f"{name}_SEP_TRAIN"]
    eval_A_path = globals()[f"{name}_SEP_EVAL"]
    train_B_path = globals()[f"{name}_SLOTS_TRAIN"]
    eval_B_path = globals()[f"{name}_SLOTS_EVAL"]

    dataset_A, dataset_B = [], []
    for pair in data_pairs:
        state, move = pair['state'], pair['action']
        action_label = ACTION_MAP[move]
        dataset_A.append({"input": state_to_binary_A(state, HANOI_N, NUM_STACKS), "output": action_label})
        dataset_B.append({"input": state_to_slots_B(state, HANOI_N, NUM_STACKS), "output": action_label})

    # 按需采样
    if len(dataset_A) > size:
        sampled_indices = random.choices(range(len(dataset_A)), k=size)
        sampled_A = [dataset_A[i] for i in sampled_indices]
        sampled_B = [dataset_B[i] for i in sampled_indices]
    else:
        sampled_A, sampled_B = dataset_A, dataset_B

    # 分割训练集和评估集并写入
    def write(data, train_path, eval_path):
        random.shuffle(data)
        eval_size = max(100, len(data) // 10)
        train_data, eval_data = data, data[:eval_size]  # 评估集是训练集的子集
        print(f"正在写入 {len(train_data)} 条训练数据到 {train_path}...")
        with open(train_path, 'w') as f:
            for r in train_data: f.write(json.dumps(r) + '\n')
        print(f"正在写入 {len(eval_data)} 条评估数据到 {eval_path}...")
        with open(eval_path, 'w') as f:
            for r in eval_data: f.write(json.dumps(r) + '\n')

    print(f"\n--- 写入 {name} 数据集 ---")
    write(sampled_A, train_A_path, eval_A_path)
    write(sampled_B, train_B_path, eval_B_path)


# ==============================================================================
# --- 5. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    print("=" * 60)
    print(f"汉诺塔终极对比数据集生成器 (N={HANOI_N})")
    print("=" * 60)

    env = HanoiEnv(HANOI_N)

    # --- 生成并写入“最优路径”数据集 ---
    print("\n[1] 开始生成“最优路径”数据...")
    start_time = time.time()
    path_data_pairs = get_hanoi_solution_path(env)
    print(f"“最优路径”数据对生成完毕 (共 {len(path_data_pairs)} 条)，耗时 {time.time() - start_time:.2f} 秒。")
    write_datasets(path_data_pairs, "PATH", PATH_DATASET_SIZE)

    # --- 生成并写入“全局策略”数据集 ---
    print("\n[2] 开始生成“全局策略”数据...")
    start_time = time.time()
    policy_map = get_hanoi_full_policy(env)
    policy_data_pairs = [{'state': state, 'action': action} for state, action in policy_map.items()]
    print(f"“全局策略”数据对生成完毕 (共 {len(policy_data_pairs)} 条)，耗时 {time.time() - start_time:.2f} 秒。")
    write_datasets(policy_data_pairs, "POLICY", POLICY_DATASET_SIZE)

    print("\n所有数据集生成完成！")