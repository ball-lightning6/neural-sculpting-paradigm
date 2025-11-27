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
TRAIN_FILE = f'blocks_n{BLOCKS_N}_slots_multilabel_train.jsonl'
EVAL_FILE = f'blocks_n{BLOCKS_N}_slots_multilabel_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 ---
# ==============================================================================
ACTIONS = [(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i!=j]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)  # 输出向量的维度，例如6

INPUT_LEN = BLOCKS_N * NUM_STACKS

print("=" * 60)
print("积木世界数据集生成器 (固定槽位 + 多标签终极版)")
print(f"参数: N={BLOCKS_N}, Stacks={NUM_STACKS}")
print(f"输入格式: 长度为 {INPUT_LEN} 的整数序列")
print(f"输出格式: {NUM_ACTIONS} 个多标签二分类")
print("=" * 60)


# ==============================================================================
# --- 3. 核心逻辑：环境类 (无变化) ---
# ==============================================================================
class BlocksWorld:
    # ... (省略完整代码，与之前完全相同)
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
def state_to_fixed_slots(state_tuple, num_blocks, num_stacks):
    slots = [0] * (num_blocks * num_stacks)
    state_list = [list(stack) for stack in state_tuple]
    for i in range(num_stacks):
        stack = state_list[i]
        for j in range(len(stack)):
            slot_index = i * num_blocks + j
            block_id = stack[len(stack) - 1 - j]
            slots[slot_index] = block_id
    slots = ''.join(map(str,slots))
    return slots


def generate_multihot_policy_dataset(env, output_path, is_eval=False):
    log_prefix = f"[{'评估集' if is_eval else '训练集'}]"
    print(f"\n{log_prefix} --- 开始生成多重最优解策略数据集 ---")
    print(f"{log_prefix} 输出文件: {output_path}")

    # 1. 反向BFS，计算所有状态到终点的最短距离
    print(f"{log_prefix} 步骤 1: 反向搜索并计算所有状态到终点的距离...")
    start_time = time.time()
    dist_map = {env.goal_state: 0}
    queue = deque([env.goal_state])
    while queue:
        current_state = queue.popleft()
        current_dist = dist_map[current_state]
        current_state_list = env.state_from_tuple(current_state)
        for from_s in range(env.num_stacks):
            for to_s in range(env.num_stacks):
                if from_s==to_s: continue
                if current_state_list[to_s]:
                    prev_state = env.apply_move(env.state_to_tuple(current_state_list), (to_s, from_s))
                    if prev_state is not None and prev_state not in dist_map:
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
            if next_state in dist_map and dist_map[next_state]==dist - 1:
                optimal_actions.append(move)

        if optimal_actions:
            output_labels = [0] * NUM_ACTIONS
            for move in optimal_actions:
                output_labels[ACTION_MAP[move]] = 1

            input_ids = state_to_fixed_slots(state, env.num_blocks, env.num_stacks)
            print(state, input_ids)
            for l, a in zip(output_labels, ACTIONS):
                if l:
                    print(a, end=' ')
            print()
            dataset.append({"input": input_ids, "output": output_labels})

    print(f"{log_prefix} 策略生成完毕。共找到 {len(dataset)} 条有效策略。")

    # 3. 写入文件
    if is_eval:
        random.shuffle(dataset)
        eval_size = max(10, int(len(dataset) * 0.2))
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
    generate_multihot_policy_dataset(env, TRAIN_FILE, is_eval=False)
    # generate_multihot_policy_dataset(env, EVAL_FILE, is_eval=True)