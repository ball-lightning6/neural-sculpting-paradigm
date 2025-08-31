import json
import random
import math
import sys
from collections import deque

# --- 参数配置 ---
BLOCKS_N = 6
NUM_STACKS = 3
DATASET_SIZE = 100000
TRAIN_FILE = f'blocks_n{BLOCKS_N}_final_train.jsonl'
EVAL_FILE = f'blocks_n{BLOCKS_N}_final_eval.jsonl'

# --- 编码定义 ---
ACTIONS = [(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i!=j]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
BITS_PER_TOKEN = math.ceil(math.log2(BLOCKS_N + 2))
SEP_TOKEN_ID = BLOCKS_N + 1
PAD_TOKEN_ID = 0
SINGLE_STATE_TOKEN_LEN = BLOCKS_N + (NUM_STACKS - 1)
TOTAL_INPUT_BITS = SINGLE_STATE_TOKEN_LEN * 2 * BITS_PER_TOKEN


# --- 环境与求解器 ---
class BlocksWorld:
    # ... (此处省略，与之前版本相同)
    def __init__(self, num_blocks, num_stacks):
        self.num_blocks, self.num_stacks = num_blocks, num_stacks

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
        block = new_state_list[from_s].pop(0)
        new_state_list[to_s].insert(0, block)
        return self.state_to_tuple(new_state_list)

    def state_to_tuple(self, state_list):
        return tuple(tuple(stack) for stack in state_list)

    def state_from_tuple(self, state_tuple):
        return [list(stack) for stack in state_tuple]


def bfs_solver(env, initial_state, goal_state):
    # ... (此处省略，与之前版本相同)
    queue = deque([(initial_state, [])])
    visited = {initial_state}
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


def generate_random_state(num_blocks, num_stacks):
    # ... (此处省略，与之前版本相同)
    blocks = list(range(1, num_blocks + 1))
    random.shuffle(blocks)
    state = [[] for _ in range(num_stacks)]
    for block in blocks:
        state[random.randint(0, num_stacks - 1)].insert(0, block)
    return tuple(tuple(s) for s in state)


# --- 编码函数 ---
def state_to_binary(state_tuple, num_blocks, num_stacks):
    # ... (此处省略，与之前版本相同)
    tokens, target_len = [], num_blocks + (num_stacks - 1)
    for i, stack in enumerate(state_tuple):
        tokens.extend(list(stack))
        if i < num_stacks - 1: tokens.append(SEP_TOKEN_ID)
    binary_string, padded_tokens = "", tokens + [PAD_TOKEN_ID] * (target_len - len(tokens))
    for token_id in padded_tokens: binary_string += format(token_id, f'0{BITS_PER_TOKEN}b')
    return binary_string


# --- 主生成函数 (最终版) ---
def generate_blocks_dataset_final(n_samples, blocks_n, num_stacks, output_path):
    print(f"--- Generating dataset for N={blocks_n} ---")
    print(f"Target samples: {n_samples}, Output file: {output_path}")
    env = BlocksWorld(blocks_n, num_stacks)
    seen_inputs = set()

    with open(output_path, 'w') as f:
        count = 0
        total_attempts = 0
        max_attempts = n_samples * 100  # 留出足够的尝试空间

        while count < n_samples and total_attempts < max_attempts:
            total_attempts += 1
            if total_attempts % 100==0:
                progress_percent = (count / n_samples) * 100
                sys.stdout.write(
                    f"\rAttempts: {total_attempts}, Generated: {count}/{n_samples} ({progress_percent:.2f}%)")
                sys.stdout.flush()

            initial_state = generate_random_state(blocks_n, num_stacks)
            goal_state = generate_random_state(blocks_n, num_stacks)
            if initial_state==goal_state: continue

            path = bfs_solver(env, initial_state, goal_state)

            if path:
                current_state = initial_state
                for move in path:
                    current_state_bin = state_to_binary(current_state, blocks_n, num_stacks)
                    goal_state_bin = state_to_binary(goal_state, blocks_n, num_stacks)
                    input_binary = current_state_bin + goal_state_bin

                    if input_binary not in seen_inputs:
                        seen_inputs.add(input_binary)

                        action_label = ACTION_MAP[move]
                        record = {"input": input_binary, "output": action_label}
                        f.write(json.dumps(record) + '\n')

                        count += 1
                        if count >= n_samples: break

                    current_state = env.apply_move(current_state, move)

        if total_attempts >= max_attempts and count < n_samples:
            print(f"\nWarning: Max attempts reached. Generated {count} unique samples.")
        else:
            sys.stdout.write(f"\rGenerated: {count}/{n_samples} (100.00%)\n")
            sys.stdout.flush()

    print(f"Dataset '{output_path}' generation complete. Total unique records: {count}.")


# --- 执行 ---
if __name__=="__main__":
    generate_blocks_dataset_final(DATASET_SIZE, BLOCKS_N, NUM_STACKS, TRAIN_FILE)
    # generate_blocks_dataset_final(max(1000, DATASET_SIZE // 10), BLOCKS_N, NUM_STACKS, EVAL_FILE)