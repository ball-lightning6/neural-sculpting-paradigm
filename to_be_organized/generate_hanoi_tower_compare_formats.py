import json
import random
import math
import sys
import time
from collections import deque
import itertools

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
HANOI_N = 3
NUM_STACKS = 3
DATASET_SIZE = (3**HANOI_N) - 1

FORMAT_A_TRAIN_FILE = f'hanoi_n{HANOI_N}_sep_full_train.jsonl'
FORMAT_A_EVAL_FILE = f'hanoi_n{HANOI_N}_sep_full_eval.jsonl'
FORMAT_B_TRAIN_FILE = f'hanoi_n{HANOI_N}_slots_full_train.jsonl'
FORMAT_B_EVAL_FILE = f'hanoi_n{HANOI_N}_slots_full_eval.jsonl'


# ==============================================================================
# --- 2. 编码与表示定义 ---
# ==============================================================================
# ... (无变化，省略) ...
ACTIONS,ACTION_MAP,NUM_ACTIONS = ([(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i != j], {(i, j): idx for idx, (i, j) in enumerate([(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i != j])}, len([(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i != j]))
BITS_PER_TOKEN_A,SEP_TOKEN_ID_A,PAD_TOKEN_ID_A,INPUT_TOKEN_LEN_A,TOTAL_INPUT_BITS_A = (math.ceil(math.log2(HANOI_N + 2)), HANOI_N + 1, 0, HANOI_N + (NUM_STACKS - 1), (HANOI_N + (NUM_STACKS - 1)) * math.ceil(math.log2(HANOI_N + 2)))
INPUT_LEN_B = HANOI_N * NUM_STACKS

# ==============================================================================
# --- 3. 核心逻辑：环境类 (关键修正) ---
# ==============================================================================
class HanoiEnv:
    def __init__(self, n, s=3):
        self.n, self.s = n, s
        goal_list = [[] for _ in range(s)]
        goal_list[-1] = list(range(n, 0, -1))
        self.goal_state = tuple(map(tuple, goal_list))

    def is_valid_state(self, state_tuple):
        """检查一个状态是否合法（大盘不能压小盘）"""
        for stack in state_tuple:
            for i in range(len(stack) - 1):
                # 栈底(i)的盘子必须比它上面的盘子(i+1)大
                if stack[i] < stack[i+1]: # <--- 致命BUG已修正！
                    return False
        return True

    def apply_move(self, state_tuple, move):
        from_s, to_s = move
        state_list = [list(s) for s in state_tuple]
        disk = state_list[from_s].pop()
        state_list[to_s].append(disk)
        return tuple(map(tuple, state_list))

# ==============================================================================
# --- 4. 编码与数据集生成 (无变化) ---
# ==============================================================================
# ... (省略所有其他函数，它们都是正确的)
def state_to_tokens_A(state_tuple):
    tokens = [];
    for i in range(NUM_STACKS):
        tokens.extend(list(reversed(state_tuple[i])))
        if i < NUM_STACKS - 1: tokens.append(SEP_TOKEN_ID_A)
    return tokens

def tokens_to_binary_A(tokens):
    binary_string, padded_tokens = "", tokens + [PAD_TOKEN_ID_A] * (INPUT_TOKEN_LEN_A - len(tokens));
    for token_id in padded_tokens: binary_string += format(token_id, f'0{BITS_PER_TOKEN_A}b')
    return binary_string
def state_to_slots_B(state_tuple, num_disks, num_stacks):
    slots = [0] * (num_disks * num_stacks);
    for i in range(num_stacks):
        stack, stack_len = state_tuple[i], len(state_tuple[i]);
        for j in range(stack_len): slots[i * num_disks + j] = stack[j]
    slots = ''.join(map(str,slots))
    return slots

def generate_hanoi_full_dataset(env):
    print("步骤 1: 通过反向BFS构建覆盖所有3^N个状态的最优策略地图..."); start_time = time.time()
    policy_map, queue, visited = {}, deque([env.goal_state]), {env.goal_state}
    while queue:
        current_state = queue.popleft()
        for from_s in range(env.s):
            for to_s in range(env.s):
                if from_s == to_s: continue
                state_list = [list(s) for s in current_state]
                if state_list[to_s]:
                    disk = state_list[to_s].pop(); state_list[from_s].append(disk)
                    prev_state = tuple(map(tuple, state_list))
                    if env.is_valid_state(prev_state) and prev_state not in visited:
                        visited.add(prev_state); queue.append(prev_state); policy_map[prev_state] = (from_s, to_s)
    print(f"策略地图构建完毕，共找到 {len(policy_map)} 个状态的策略。耗时: {time.time() - start_time:.2f} 秒。")
    dataset_A, dataset_B = [], []
    for state, move in policy_map.items():
        tokens_A = state_to_tokens_A(state); input_A = tokens_to_binary_A(tokens_A)
        dataset_A.append({"input": input_A, "output": ACTION_MAP[move]})
        #print(tokens_A,move)
        input_B = state_to_slots_B(state, env.n, env.s)
        dataset_B.append({"input": input_B, "output": ACTION_MAP[move]})
        print(input_B, move)
    return dataset_A, dataset_B

def write_dataset(dataset, train_path, eval_path):
    print(f"\n为 {train_path} 写入数据... (总数: {len(dataset)})"); random.shuffle(dataset)
    train_data, eval_data = dataset, dataset[:max(100, len(dataset) // 10)]
    with open(train_path, 'w') as f:
        for record in train_data: f.write(json.dumps(record) + '\n')
    with open(eval_path, 'w') as f:
        for record in eval_data: f.write(json.dumps(record) + '\n')

# ==============================================================================
# --- 5. 主执行部分 ---
# ==============================================================================
if __name__ == "__main__":
    env = HanoiEnv(HANOI_N, NUM_STACKS)
    print(f"汉诺塔 N={HANOI_N}, 状态空间大小: 3^{HANOI_N} = {3**HANOI_N}")
    dataset_A, dataset_B = generate_hanoi_full_dataset(env)
    write_dataset(dataset_A, FORMAT_A_TRAIN_FILE, FORMAT_A_EVAL_FILE)
    write_dataset(dataset_B, FORMAT_B_TRAIN_FILE, FORMAT_B_EVAL_FILE)
    print("\n所有数据集生成完成！")