import json
import random
import math
import sys
import time

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
HANOI_N = 16
NUM_STACKS = 3
# 对于N=16，最优解有 2^16-1 = 65535 步。我们可以从中采样。
DATASET_SIZE = 300000
TRAIN_FILE = f'hanoi_n{HANOI_N}_slots_train.jsonl'
EVAL_FILE = f'hanoi_n{HANOI_N}_slots_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 ---
# ==============================================================================
ACTIONS = [(i, j) for i in range(NUM_STACKS) for j in range(NUM_STACKS) if i!=j]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# 输入编码：每个槽位一个token，0代表空，1-N代表积木
INPUT_LEN = HANOI_N * NUM_STACKS

print("=" * 60)
print("汉诺塔数据集生成器 (固定槽位 - 终极版)")
print(f"参数: N={HANOI_N}, Stacks={NUM_STACKS}")
print(f"输入格式: 长度为 {INPUT_LEN} 的整数序列 (0=空, 1-{HANOI_N}=盘子)")
print(f"输出格式: {NUM_ACTIONS} 分类问题 (from, to)")
print("=" * 60)


# ==============================================================================
# --- 3. 核心逻辑：求解器与编码函数 ---
# ==============================================================================
def hanoi_solver_recursive(n, source, target, auxiliary, moves_list):
    """标准的汉诺塔递归求解器"""
    if n > 0:
        hanoi_solver_recursive(n - 1, source, auxiliary, target, moves_list)
        moves_list.append({'disk': n, 'from': source, 'to': target})
        hanoi_solver_recursive(n - 1, auxiliary, target, source, moves_list)


def get_hanoi_solution(n):
    """获取N盘汉诺塔的完整解法步骤列表"""
    moves = []
    print(f"正在计算 N={n} 的汉诺塔解法 (共 {2 ** n - 1} 步)...")
    start_time = time.time()
    hanoi_solver_recursive(n, 0, 2, 1, moves)
    print(f"解法计算完成，耗时: {time.time() - start_time:.2f} 秒。")
    return moves


def state_to_fixed_slots(pegs, num_disks, num_stacks):
    """将盘面状态 [[3,2,1], [], []] 转换为固定槽位的整数列表"""
    slots = [0] * (num_disks * num_stacks)
    for i in range(num_stacks):
        stack = pegs[i]
        # 从下到上填充槽位
        for j in range(len(stack)):
            slot_index = i * num_disks + j
            disk_id = stack[len(stack) - 1 - j]
            slots[slot_index] = disk_id
    return slots


# ==============================================================================
# --- 4. 数据集生成主函数 ---
# ==============================================================================
def generate_hanoi_dataset(n_samples, hanoi_n, num_stacks, output_path):
    log_prefix = f"[{'评估集' if 'eval' in output_path else '训练集'}]"
    print(f"\n{log_prefix} --- 开始为 N={hanoi_n} 生成数据集 ---")

    # 1. 首先生成完整的解法序列
    solution_moves = get_hanoi_solution(hanoi_n)

    # 2. 从解法序列中创建 (状态, 下一步动作) 的数据对
    print(f"{log_prefix} 正在从解法路径中构建 (状态, 动作) 数据对...")
    data_pairs = []
    pegs = [list(range(hanoi_n, 0, -1)), [], []]  # 初始状态

    for move in solution_moves:
        current_pegs_state = [p[:] for p in pegs]  # 深拷贝
        action = (move['from'], move['to'])
        action_label = ACTION_MAP[action]

        # 使用全新的固定槽位编码
        input_ids = state_to_fixed_slots(current_pegs_state, hanoi_n, num_stacks)
        data_pairs.append({'input': input_ids, 'output': action_label})

        # 更新状态到下一步
        disk_to_move = pegs[move['from']].pop(-1)  # 汉诺塔是从顶部拿
        pegs[move['to']].append(disk_to_move)

    print(f"{log_prefix} 数据对构建完成，共 {len(data_pairs)} 个不重复步骤。")

    # 3. 从所有步骤中随机抽样来构成最终数据集
    print(f"{log_prefix} 正在从 {len(data_pairs)} 个步骤中随机采样 {n_samples} 条数据...")
    final_dataset = random.choices(data_pairs, k=n_samples)

    print(f"{log_prefix} 正在将 {len(final_dataset)} 条记录写入 '{output_path}'...")
    with open(output_path, 'w') as f:
        for record in final_dataset:
            f.write(json.dumps(record) + '\n')

    print(f"{log_prefix} 数据集 '{output_path}' 生成完成！")


# ==============================================================================
# --- 5. 主执行部分 ---
# ==============================================================================
if __name__=="__main__":
    generate_hanoi_dataset(DATASET_SIZE, HANOI_N, NUM_STACKS, TRAIN_FILE)
    # generate_hanoi_dataset(max(1000, DATASET_SIZE // 10), HANOI_N, NUM_STACKS, EVAL_FILE)