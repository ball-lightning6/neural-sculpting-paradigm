from functools import lru_cache
import copy
import time
import random
from tqdm import tqdm

HANOI_N = 10

ACTIONS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)  # 输出是 6 分类

@lru_cache(maxsize=None)
def calc_mlist(n):
    if n==1:
        return [1]
    else:
        return calc_mlist(n-1)+[n]+calc_mlist(n-1)

def calc_end_state(n):
    return [[],[],list(range(1,n+1))]

def init_all_states(n):
    all_state = [0]*(3**n)
    all_state[0] = calc_end_state(n)
    return all_state

def init_best_moves(n):
    all_best_moves = [0] * (3 ** n)
    all_best_moves[0] = None
    return all_best_moves

def init_next_nodes(n):
    next_nodes = [0] * (3 ** n)
    next_nodes[0] = None
    return next_nodes

@lru_cache(maxsize=None)
def calc_region_length_list(n):
    if n==1:
        return [1,2]
    else:
        last = calc_region_length_list(n-1)
        last.extend([x*2 for x in last])
        return last

def calc_region(n):
    region_length_list=calc_region_length_list(n)
    region=[0]
    for length in region_length_list:
        region.append(region[-1]+length)
    return region, region_length_list

def calc_process_idx_list(region, length):
    if length==2:
        return [region[0],region[1]-1]
    else:
        region_mid_idx = (region[0]+region[1]+1)//2
        x=calc_process_idx_list([region[0],region_mid_idx], length//2)
        x.extend(calc_process_idx_list([region_mid_idx,region[1]], length//2))
        return x


def find_x(state,x):
    for idx,s in enumerate(state):
        if x in s and s[0]==x:
            return idx
    return None

def move_1(state,direction,idx):
    new_state = copy.deepcopy(state)
    if direction == -1:
        if idx==0:
            new_idx = 2
        else:
            new_idx = idx-1
    else:
        if idx==2:
            new_idx = 0
        else:
            new_idx = idx+1
    new_state[idx] = new_state[idx][1:]
    new_state[new_idx] = [1]+new_state[new_idx]
    return new_state,(new_idx, idx)

def move_x(state,m,idx):
    new_state = copy.deepcopy(state)
    for new_idx,s in enumerate(state):
        if len(s)==0 or s[0]>m:
            break
    else:
        return None, None

    new_state[idx] = new_state[idx][1:]
    new_state[new_idx] = [m]+new_state[new_idx]
    return new_state, (new_idx, idx)


def calc_one_layer(all_states,best_moves,next_nodes,region,mlist,region_length_list,l,tq):
    # print('l:',l)

    m = mlist[l]

    # print(region[l],region[l+1])
    if m==1:
        for i,state_idx in enumerate(range(region[l],region[l+1])):
            # print('m: 1 state_idx:',state_idx)
            state = all_states[state_idx]
            # print(state)
            idx = find_x(state, 1)
            right_state, r_best_move = move_1(state, 1, idx)
            tq.update()
            left_state, l_best_move = move_1(state, -1, idx)
            tq.update()
            all_states[region[l + 1] + i * 2] = right_state
            all_states[region[l + 1] + i * 2 + 1] = left_state

            next_nodes[region[l + 1] + i * 2] = state_idx
            next_nodes[region[l + 1] + i * 2 + 1] = state_idx

            # print(region[l + 1] + i * 2, right_state)
            # print(region[l + 1] + i * 2 + 1, left_state)

            best_moves[region[l + 1] + i * 2] = r_best_move
            best_moves[region[l + 1] + i * 2 + 1] = l_best_move
    else:
        i = 0
        process_idxs = calc_process_idx_list([region[l],region[l+1]], region_length_list[l+1])
        for state_idx in process_idxs:
            # print(f'm: {m} state_idx:', state_idx)
            state = all_states[state_idx]
            # print(state)
            idx = find_x(state, m)
            if idx is None:
                continue

            new_state, best_move = move_x(state,m,idx)
            if new_state is None:
                continue
            # print('----',new_state,best_move)
            tq.update()
            all_states[region[l + 1] + i] = new_state
            next_nodes[region[l + 1] + i] = state_idx
            # print(region[l + 1] + i, new_state)
            best_moves[region[l + 1] + i] = best_move
            i +=1
import json
import tqdm

def calc_all(n):
    tq = tqdm.tqdm(total=3 ** n)
    tq.update()
    all_states = init_all_states(n)
    best_moves = init_best_moves(n)
    next_nodes = init_next_nodes(n)
    region, region_length_list = calc_region(n)
    mlist = calc_mlist(n)
    for l in range(2**n-1):
        calc_one_layer(all_states,best_moves,next_nodes,region,mlist,region_length_list,l,tq)
    return all_states,best_moves,next_nodes
    # with open(f'all_states_n{n}.json', 'w') as f:
    #     for i,(state,best_move, next_node) in enumerate(zip(all_states,best_moves,next_nodes)):
    #         f.write(str(i)+' | '+str(state)+' | '+str(best_move)+' | '+str(next_node)+'\n')
    # with open(f'best_moves_n{n}.json', 'w') as f:
    #     for i, best_move in enumerate(best_moves):
    #         f.write(str(i) + ' | ' + str(best_move) + '\n')

n=HANOI_N
s= time.time()
all_states,best_moves,next_nodes=calc_all(n)
t= time.time()-s
nlf = 3**n
print(t,t/nlf)

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
    while True:
        idx = next_nodes[idx]
        if idx == 0:
            break
        state = all_states[idx]
        best_move = best_moves[idx]
        #print(best_move)
        action_label = ACTION_MAP[best_move]
        dataset.append({"input": state_to_slots_B(state), "output": action_label})

    def write(data, train_path):
        random.shuffle(data)
        print(f"正在写入 {len(data)} 条训练数据到 {train_path}...")
        with open(train_path, 'w') as f:
            for r in data: f.write(json.dumps(r) + '\n')

    write(dataset, f'hanoi_n{HANOI_N}_path_slots_train_idx_{start_i}.jsonl')

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

    write(dataset, f'hanoi_n{HANOI_N}_path_slots_train_idx_{start_region}_{end_region}.jsonl')

def generate_all_dataset():
    dataset = []
    for idx in range(1,end_idx+1):
        state = all_states[idx]
        best_move = best_moves[idx]
        action_label = ACTION_MAP[best_move]
        dataset.append({"input": state_to_slots_B(state), "output": action_label})

    def write(data, train_path):
        random.shuffle(data)
        print(f"正在写入 {len(data)} 条训练数据到 {train_path}...")
        with open(train_path, 'w') as f:
            for r in data: f.write(json.dumps(r) + '\n')

    write(dataset, f'hanoi_n{HANOI_N}_path_slots_train_all.jsonl')


for gene_idx in gene_idxs:
    generate_dataset_from_node_idx(gene_idx)


generate_dataset_from_region(start_idx,end_idx)

generate_all_dataset()