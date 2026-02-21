import json
import random
import itertools

def generate_hanoi_data(num_disks, num_samples=100000, save_path="hanoi_decoupled.jsonl", target_peg=1):
    """
    生成汉诺塔任意合法状态到固定目标柱(target_peg)的最优解数据集。
    包含解耦标签(每个盘子的阶段性目标)和最终动作(one-hot格式)。
    """
    
    # 动作映射表: (起始柱子, 目标柱子) -> 动作索引 (0~5)
    action_to_idx = {
        (-1, 0): 0, (-1, 1): 1,
        (0, -1): 2, (0, 1): 3,
        (1, -1): 4, (1, 0): 5
    }

    def get_buffer(p1, p2):
        """计算中转柱：给定两个柱子，返回第三个。柱子集合为{-1, 0, 1}，总和为0"""
        return -(p1 + p2)

    def get_recursive_goals(state):
        """
        核心逻辑：计算每个盘子在当前状态下的“阶段性目标柱”
        state: [p1, p2, ..., pn] (p1最小, pn最大)
        """
        n = len(state)
        goals = [0] * n
        
        # 最大的盘子目标永远是 target_peg
        goals[n-1] = target_peg
        
        # 自顶向下（从大盘子到小盘子）推导目标
        for i in range(n-2, -1, -1):
            parent_disk_idx = i + 1
            if state[parent_disk_idx] == goals[parent_disk_idx]:
                # 如果更大的盘子已经在它的目标上了，我的目标和它一样
                goals[i] = goals[parent_disk_idx]
            else:
                # 如果更大的盘子还没归位，我必须去它当前位置和它目标的“中转柱”腾地方
                goals[i] = get_buffer(state[parent_disk_idx], goals[parent_disk_idx])
        return goals

    def get_next_move(state, goals):
        """寻找当前需要移动的盘子（从小到大寻找第一个不在目标位置的盘子）"""
        for i in range(len(state)):
            if state[i] != goals[i]:
                return action_to_idx[(state[i], goals[i])]
        return -1 # 已完成状态（所有盘子都在 target_peg）

    def process_state_to_dict(state):
        """将单个状态处理为模型需要的字典格式"""
        goals = get_recursive_goals(state)
        action_idx = get_next_move(state, goals)
        
        if action_idx == -1: 
            return None # 终点状态没有下一步动作
            
        # Scaffold: 每组 3 个 one-hot，表示目标柱子 {-1, 0, 1}
        scaffold_label = []
        for g in goals:
            one_hot = [0, 0, 0]
            one_hot[g + 1] = 1 # 将 -1, 0, 1 映射到索引 0, 1, 2
            scaffold_label.extend(one_hot)
        
        # Action: one-hot 编码 (6维向量)
        action_one_hot = [0] * 6
        action_one_hot[action_idx] = 1
            
        return {
            "input": state,
            "scaffold": scaffold_label,
            "action": action_one_hot  # 改回 one-hot 格式
        }

    dataset = []
    total_states = 3**num_disks
    is_full_enumeration = total_states <= num_samples

    print(f"开始生成 N={num_disks} 层汉诺塔数据...")
    print(f"总状态空间大小: {total_states}")

    if is_full_enumeration:
        print(f"状态空间较小，启动全量遍历生成...")
        for p in itertools.product([-1, 0, 1], repeat=num_disks):
            data_entry = process_state_to_dict(list(p))
            if data_entry:
                dataset.append(data_entry)
        print(f"全量遍历完成，共生成 {len(dataset)} 条有效数据（去除了终点状态）。")
        
    else:
        print(f"状态空间巨大，启动随机采样生成，目标 {num_samples} 条...")
        seen_states = set()
        count = 0
        
        while count < num_samples:
            # 随机生成一个合法状态
            state = [random.choice([-1, 0, 1]) for _ in range(num_disks)]
            state_tuple = tuple(state)
            
            if state_tuple in seen_states: 
                continue
            seen_states.add(state_tuple)
            
            data_entry = process_state_to_dict(state)
            if data_entry:
                dataset.append(data_entry)
                count += 1
                
                if count % 50000 == 0: 
                    print(f"已生成 {count} 条...")

    # 写入文件
    with open(save_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
    
    print(f"数据生成完美结束！已保存至 {save_path}")

# 示例：生成 16 层数据
generate_hanoi_data(num_disks=16, num_samples=1000000, save_path="hanoi_decoupled_N16.jsonl")