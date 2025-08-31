import random
import json
from tqdm import tqdm

file_name = 'all_states_n16.json'

all_states= []
best_moves = []
next_nodes = []

HANOI_N = 16

ACTIONS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)  # 输出是 6 分类

with open(file_name,'r') as f:
    # lines =
    for line in tqdm(f.readlines()):
        node_idx, state, best_move, next_node_idx = line.strip().split('|')
        node_idx = int(node_idx.strip())
        state = eval(state.strip())
        best_move = eval(best_move.strip())
        if next_node_idx.strip().lower()=='none':
            next_node_idx = None
        else:
            next_node_idx = int(next_node_idx.strip())

        all_states.append(state)
        best_moves.append(best_move)
        next_nodes.append(next_node_idx)

def state_to_slots_B(state_tuple, n=HANOI_N, s=3):
    slots = [0] * (n * s)
    for i in range(s):
        stack = state_tuple[i]
        for j in range(len(stack)):
            slots[i * n + j] = stack[j]
    #slots = ''.join([str(x).zfill(2) for x in slots])
    slots = ''.join([chr(x+97) for x in slots])
    return slots

def calc_twist_node(start_node_idx, end_node_idx):
    return (start_node_idx+end_node_idx-1)//2

start_idx = 3**HANOI_N-2**HANOI_N
end_idx = 3**HANOI_N-1

twist_idx = calc_twist_node(start_idx,end_idx)
twist_idx_1= twist_idx+1

def calc_twistest_node(start_node_idx, end_node_idx, direction):
    if direction==-1:
        if start_node_idx == end_node_idx -1:
            return start_node_idx
        twist_idx = calc_twist_node(start_node_idx, end_node_idx)
        return calc_twistest_node(start_node_idx,twist_idx,1)
    else:
        if start_node_idx == end_node_idx -1:
            return end_node_idx
        twist_idx = calc_twist_node(start_node_idx, end_node_idx)+1
        return calc_twistest_node(twist_idx,end_node_idx,-1)

twistest_idx = calc_twistest_node(start_idx,end_idx,-1)
twistest_idx_1 = calc_twistest_node(start_idx,end_idx,1)

gene_idxs = [start_idx, end_idx, twist_idx, twist_idx_1, twistest_idx, twistest_idx_1]
print(gene_idxs)

def generate_dataset_from_node_idx(idx):
    start_i = idx
    dataset = []
    state = all_states[idx]
    best_move= best_moves[idx]
    action_label = ACTION_MAP[best_move]
    dataset.append({"input": state_to_slots_B(state), "output": action_label})
    while idx!=0:
        idx = next_nodes[idx]
        state = all_states[idx]
        best_move = best_moves[idx]
        action_label = ACTION_MAP[best_move]
        dataset.append({"input": state_to_slots_B(state), "output": action_label})

    def write(data, train_path):
        random.shuffle(data)
        print(f"正在写入 {len(data)} 条训练数据到 {train_path}...")
        with open(train_path, 'w') as f:
            for r in data: f.write(json.dumps(r) + '\n')

    write(dataset, f'hanoi_n16_path_slots_train_idx_{start_i}.jsonl')

def generate_dataset_from_region(start_region, end_region):
    dataset = []
    for idx in range(start_region,end_region+1):
        state = all_states[idx]
        best_move = best_moves[idx]
        action_label = ACTION_MAP[best_move]
        dataset.append({"input": state_to_slots_B(state), "output": action_label})

    def write(data, train_path):
        random.shuffle(data)
        print(f"正在写入 {len(data)} 条训练数据到 {train_path}...")
        with open(train_path, 'w') as f:
            for r in data: f.write(json.dumps(r) + '\n')

    write(dataset, f'hanoi_n16_path_slots_train_idx_{start_region}_{end_region}.jsonl')

for gene_idx in gene_idxs:
    generate_dataset_from_node_idx(gene_idx)

generate_dataset_from_region(start_idx,end_idx)