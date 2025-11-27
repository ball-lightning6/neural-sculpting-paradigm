import json
import random
import math
import sys
import time
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
BLOCKS_N = 6
NUM_STACKS = 3
# 对于N=6，不重复策略数是4682。我们可以生成一个更大的数据集以进行充分训练。
DATASET_SIZE = 200000
TRAIN_FILE = f'blocks_n{BLOCKS_N}_fixed_goal_final_train.jsonl'
EVAL_FILE = f'blocks_n{BLOCKS_N}_fixed_goal_final_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 ---
# ==============================================================================
ACTIONS = [(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i!=j]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
BITS_PER_TOKEN = math.ceil(math.log2(BLOCKS_N + 2))
SEP_TOKEN_ID = BLOCKS_N + 1
PAD_TOKEN_ID = 0
SINGLE_STATE_TOKEN_LEN = BLOCKS_N + (NUM_STACKS - 1)
TOTAL_INPUT_BITS = SINGLE_STATE_TOKEN_LEN * BITS_PER_TOKEN


# ==============================================================================
# --- 3. 核心逻辑：环境类 ---
# ==============================================================================
class BlocksWorld:
    def __init__(self, num_blocks, num_stacks):
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks
        goal_stack = tuple(range(num_blocks, 0, -1))
        goal_list = [[] for _ in range(num_stacks)]
        goal_list[0] = list(goal_stack)
        self.goal_state = self.state_to_tuple(goal_list)

    def apply_move(self, state, move):
        from_s, to_s = move
        new_state_list = self.state_from_tuple(state)
        if not new_state_list[from_s]: return None  # 非法移动：从空栈移动
        block = new_state_list[from_s].pop(0)
        new_state_list[to_s].insert(0, block)
        return self.state_to_tuple(new_state_list)

    def state_to_tuple(self, state_list):
        return tuple(tuple(stack) for stack in state_list)

    def state_from_tuple(self, state_tuple):
        return [list(stack) for stack in state_tuple]


# ==============================================================================
# --- 4. 编码与数据集生成 (高效反向BFS) ---
# ==============================================================================
def state_to_binary(state_tuple, num_blocks, num_stacks):
    tokens, target_len = [], num_blocks + (num_stacks - 1)
    for i, stack in enumerate(state_tuple):
        tokens.extend(list(reversed(stack)))
        if i < num_stacks - 1: tokens.append(SEP_TOKEN_ID)
    binary_string, padded_tokens = "", tokens + [PAD_TOKEN_ID] * (target_len - len(tokens))
    for token_id in padded_tokens:
        binary_string += format(token_id, f'0{BITS_PER_TOKEN}b')
    return binary_string


def generate_fixed_goal_policy_perf(env, output_path, dataset_size, is_eval=False):
    log_prefix = f"[{'评估集' if is_eval else '训练集'}]"
    print(f"\n{log_prefix} --- 开始生成数据集 (固定目标 + 反向BFS) ---")
    print(f"{log_prefix} 输出文件: {output_path}")

    # 1. 从固定的目标状态开始，进行反向广度优先搜索
    print(f"{log_prefix} 步骤 1: 通过反向搜索构建最优策略地图...")
    start_time = time.time()

    policy_map = {}
    queue = deque([env.goal_state])
    # visited 集合是去重的核心，确保每个状态只被处理一次
    visited = {env.goal_state}

    while queue:
        current_state = queue.popleft()

        # 核心逻辑：寻找所有能一步“正向”移动到达 current_state 的“前驱状态”
        current_state_list = env.state_from_tuple(current_state)
        for from_s in range(env.num_stacks):
            for to_s in range(env.num_stacks):
                if from_s==to_s: continue

                # 尝试一个“反向”移动：即把一个积木从 to_s 栈顶挪回 from_s 栈顶
                if current_state_list[to_s]:
                    # 构造假想的前驱状态
                    prev_state_list = env.state_from_tuple(current_state)
                    block_to_move_back = prev_state_list[to_s].pop(0)
                    prev_state_list[from_s].insert(0, block_to_move_back)
                    prev_state = env.state_to_tuple(prev_state_list)

                    if prev_state not in visited:
                        visited.add(prev_state)
                        queue.append(prev_state)
                        # 记录最优策略：从 prev_state 出发，就应该执行正向移动 (from_s, to_s)
                        policy_map[prev_state] = (from_s, to_s)

    end_time = time.time()
    print(f"{log_prefix} 策略地图生成完毕。找到 {len(policy_map)} 个状态的最优策略。耗时: {time.time() - start_time:.4f} 秒。")

    # 2. 将策略地图转换为数据集
    print(f"{log_prefix} 步骤 2: 将策略地图转换为数据集...")
    dataset = []
    for state, optimal_move in policy_map.items():
        input_binary = state_to_binary(state, env.num_blocks, env.num_stacks)
        action_label = ACTION_MAP[optimal_move]
        dataset.append({"input": input_binary, "output": action_label})

    # 3. 按需进行采样和写入文件
    if dataset_size>len(dataset):
        dataset_size=len(dataset)
    final_dataset = []
    if len(dataset) > 0:
        while len(final_dataset) < dataset_size:
            random.shuffle(dataset)
            final_dataset.extend(dataset)
    final_dataset = final_dataset[:dataset_size]

    if is_eval:
        random.shuffle(final_dataset)
        eval_size = max(10, len(final_dataset) // 10)
        final_dataset = final_dataset[:eval_size]

    print(f"{log_prefix} 正在将 {len(final_dataset)} 条记录写入 '{output_path}'...")
    with open(output_path, 'w') as f:
        for record in final_dataset:
            f.write(json.dumps(record) + '\n')
    print(f"{log_prefix} 数据集生成完成。")


# ==============================================================================
# --- 5. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    env = BlocksWorld(BLOCKS_N, NUM_STACKS)

    print("=" * 60)
    print("积木世界数据集生成器 (固定目标 - 最终性能版)")
    print(f"参数: N={BLOCKS_N} (积木数), Stacks={NUM_STACKS}")
    print(f"输入格式: {TOTAL_INPUT_BITS} bits (仅当前状态)")
    print(f"输出格式: {NUM_ACTIONS} 分类问题 (from, to)")
    print("=" * 60)

    generate_fixed_goal_policy_perf(env, TRAIN_FILE, DATASET_SIZE, is_eval=False)
    # generate_fixed_goal_policy_perf(env, EVAL_FILE, DATASET_SIZE, is_eval=True)