import json
import sys
from collections import deque
import itertools
import random
import time

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
PAIRS_N = 3
BOAT_CAPACITY_K = 3
TRAIN_FILE = f'river_n{PAIRS_N}_k{BOAT_CAPACITY_K}_final_v4_train.jsonl'
EVAL_FILE = f'river_n{PAIRS_N}_k{BOAT_CAPACITY_K}_final_v4_eval.jsonl'


# ==============================================================================
# --- 2. 核心逻辑：环境类 (无变化) ---
# ==============================================================================
class RiverCrossingEnv:
    def __init__(self, num_pairs, boat_capacity):
        self.N, self.k = num_pairs, boat_capacity
        self.CLIENTS = frozenset({f'C{i + 1}' for i in range(self.N)})
        self.AGENTS = frozenset({f'A{i + 1}' for i in range(self.N)})
        self.PEOPLE = sorted(list(self.CLIENTS.union(self.AGENTS)))
        self.PEOPLE_FS = frozenset(self.PEOPLE)
        client_to_agent = {f'C{i + 1}': f'A{i + 1}' for i in range(self.N)}
        agent_to_client = {f'A{i + 1}': f'C{i + 1}' for i in range(self.N)}
        self.PAIRS = {**client_to_agent, **agent_to_client}
        self.ACTION_SPACE = self._generate_action_space()
        self.ACTION_MAP = {action: i for i, action in enumerate(self.ACTION_SPACE)}
        self.initial_state = (self.PEOPLE_FS, 'left')
        self.goal_state = (frozenset(), 'right')

    def _generate_action_space(self):
        actions = []
        for i in range(1, self.k + 1):
            for group in itertools.combinations(self.PEOPLE, i):
                if self.is_safe(frozenset(group)):
                    actions.append(frozenset(group))
        return sorted(list(actions), key=lambda x: (len(x), sorted(list(x))))

    def is_safe(self, people_set):
        clients, agents = self.CLIENTS.intersection(people_set), self.AGENTS.intersection(people_set)
        if not clients or not agents: return True
        for client in clients:
            if self.PAIRS[client] not in agents: return False
        return True

    def get_valid_moves(self, state):
        left_bank, boat_loc = state
        right_bank = self.PEOPLE_FS - left_bank
        bank_to_move_from = left_bank if boat_loc=='left' else right_bank
        for move_group in self.ACTION_SPACE:
            if move_group.issubset(bank_to_move_from):
                remaining_bank = bank_to_move_from - move_group
                if self.is_safe(remaining_bank):
                    if boat_loc=='left':
                        if self.is_safe(right_bank.union(move_group)): yield move_group
                    else:
                        if self.is_safe(left_bank.union(move_group)): yield move_group

    def apply_move(self, state, move_group):
        left_bank, boat_loc = state
        return (left_bank - move_group, 'right') if boat_loc=='left' else (left_bank.union(move_group), 'left')


# ==============================================================================
# --- 3. 编码函数 (关键修正) ---
# ==============================================================================
def state_to_binary(state, all_people_sorted):
    left_bank, boat_loc = state
    binary_list = ['0' if person in left_bank else '1' for person in all_people_sorted]
    binary_list.append('0' if boat_loc=='left' else '1')
    return "".join(binary_list)


def action_to_multilabel(action_group, all_people_sorted):
    """修正了逻辑，确保遍历的是总人员列表"""
    return [1 if person in action_group else 0 for person in all_people_sorted]


# ==============================================================================
# --- 4. 全局最优策略生成器 (无变化) ---
# ==============================================================================
def generate_unique_policy_dataset(env, output_path, is_eval=False):
    log_prefix = f"[{'评估集' if is_eval else '训练集'}]"
    print(f"\n{log_prefix} --- 开始生成纯净策略数据集 ---")
    print(f"{log_prefix} 输出文件: {output_path}")

    print(f"{log_prefix} 步骤 1: 构建最优策略地图...")
    start_time = time.time()
    policy_map, queue, visited = {}, deque([env.goal_state]), {env.goal_state}
    while queue:
        current_state = queue.popleft()
        prev_boat_loc = 'left' if current_state[1]=='right' else 'right'
        bank_to_take_from = env.PEOPLE_FS - current_state[0] if prev_boat_loc=='left' else current_state[0]
        for move_group in env.ACTION_SPACE:
            if move_group.issubset(bank_to_take_from):
                prev_state_guess = env.apply_move(current_state, move_group)
                if prev_state_guess not in visited:
                    prev_left_bank, _ = prev_state_guess
                    if env.is_safe(prev_left_bank) and env.is_safe(env.PEOPLE_FS - prev_left_bank):
                        visited.add(prev_state_guess)
                        queue.append(prev_state_guess)
                        policy_map[prev_state_guess] = move_group
    end_time = time.time()
    print(f"{log_prefix} 策略地图生成完毕，找到 {len(policy_map)} 条唯一策略。耗时: {time.time() - start_time:.4f} 秒。")

    print(f"{log_prefix} 步骤 2: 将策略地图转换为数据集...")
    dataset = []
    goal_state_bin = state_to_binary(env.goal_state, env.PEOPLE)
    for state, optimal_action_group in policy_map.items():
        current_state_bin = state_to_binary(state, env.PEOPLE)
        input_binary = current_state_bin# + goal_state_bin
        action_labels = action_to_multilabel(optimal_action_group, env.PEOPLE)
        dataset.append({"input": input_binary, "output": action_labels})

    if is_eval:
        random.shuffle(dataset)
        eval_size = max(1, len(dataset) // 5)
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
    env = RiverCrossingEnv(PAIRS_N, BOAT_CAPACITY_K)
    print("=" * 60)
    print("过河问题数据集生成器 (多标签 - 终极无暇版)")
    print(f"参数: N={PAIRS_N}, k={BOAT_CAPACITY_K}")
    TOTAL_INPUT_BITS = 2 * (len(env.PEOPLE) + 1)
    NUM_OUTPUTS = len(env.PEOPLE)
    print(f"输入格式: {TOTAL_INPUT_BITS} bits ([当前状态] + [目标状态])")
    print(f"输出格式: {NUM_OUTPUTS} 个多标签 (每个人是否上船)")
    print("=" * 60)
    generate_unique_policy_dataset(env, TRAIN_FILE, is_eval=False)
    # generate_unique_policy_dataset(env, EVAL_FILE, is_eval=True)