import random
import json
import argparse
from typing import List, Optional

def rule90_next(state: List[int]) -> List[int]:
    """应用Rule 90演化一次。
    
    Args:
        state: N位二进制列表
    Returns:
        N位二进制列表
    """
    # 左右补0边界
    padded = [0] + state + [0]
    # C_{i}^{t+1} = C_{i-1}^t ^ C_{i+1}^t
    return [padded[i-1] ^ padded[i+1] for i in range(1, len(padded)-1)]

def solve_rule90_inverse_linear(target_state: List[int]) -> List[List[int]]:
    """利用Rule 90的线性性质求解所有可能的前像(Only O(N) complexity)。
    
    Rule 90的逆向关系导致奇数位置和偶数位置的解耦：
    y[i] = x[i-1] ^ x[i+1]  (x[-1]=0, x[N]=0)
    
    这可以分解为两个独立的链条：
    1. 奇数链: x[1] 由 y[0] 确定 (y[0] = x[-1]^x[1] = 0^x[1]), x[3] 由 x[1]和y[2]确定...
    2. 偶数链: x[0] 是自由变量(除非受右边界约束), x[2] 由 x[0]和y[1]确定...
    
    Returns:
        List of valid input states (each is a list of 0/1)
    """
    N = len(target_state)
    # 目标状态 y
    y = target_state
    
    def solve_chain(indices, start_val=None):
        # 如果start_val给定，则链被确定
        # 如果start_val未给定，则有两种可能
        valid_chains = []
        
        possible_starts = [0, 1] if start_val is None else [start_val]
        
        for s in possible_starts:
            chain_vals = {}
            current = s
            valid = True
            
            # 如果indices是 [0, 2, 4...]
            # 则 x[0] = current
            # 下一个值 x[2] 由 y[1] 决定: y[1] = x[0] ^ x[2] => x[2] = y[1] ^ x[0]
            
            # 将值填入map
            idx_ptr = 0
            # 处理第一个位置
            first_idx = indices[0]
            chain_vals[first_idx] = current
            
            # 逐步推导后续值
            # 关系: y[k] = x[k-1] ^ x[k+1]
            # 所以如果知道 x[k-1] 和 y[k], 就可以求 x[k+1]
            # 对于偶链: x[0]->y[1]->x[2]->y[3]->x[4]...
            # 对于奇链: x[-1]=0->y[0]->x[1]->y[2]->x[3]...
            
            for i in range(len(indices) - 1):
                prev_x_idx = indices[i]
                next_x_idx = indices[i+1]
                # 中间的y索引是 (prev_x_idx + next_x_idx) // 2
                # 注意：Rule 90是 y[i] = x[i-1] ^ x[i+1]
                # 即 x[i+1] = y[i] ^ x[i-1]
                
                mid_y_idx = (prev_x_idx + next_x_idx) // 2
                
                if mid_y_idx >= N:
                    # 超出范围，理论不应发生，除非N极小
                    valid = False
                    break
                    
                next_val = y[mid_y_idx] ^ chain_vals[prev_x_idx]
                chain_vals[next_x_idx] = next_val
            
            if valid:
                valid_chains.append(chain_vals)
        return valid_chains

    # -----------------------------------------------------
    # 1. 奇数位置链 (Odd indices: 1, 3, 5...)
    # -----------------------------------------------------
    # 边界约束1: x[-1] = 0 (隐式)
    # 关系: y[0] = x[-1] ^ x[1] => x[1] = y[0] ^ 0 = y[0]
    # 这意味着 x[1] 是确定的！
    odd_indices = list(range(1, N, 2))
    if not odd_indices:
        pass # N=1时可能没有odd indices? Range(1,1) is empty. x[0] is even.
    else:
        # x[1] 必须等于 y[0]
        start_odd = y[0]
    
    # 但这里如果 N 很小，要注意。用通用逻辑：
    # 我们可以把虚构的 x[-1]=0 加入推导，或者直接设定 x[1]=y[0]
    # 让我们用更通用的方式：
    # 奇数链实际上是从 x[1] 开始，由 y[0] 决定
    odd_chains = []
    if odd_indices:
        # x[1] is fixed by y[0]
        odd_chains = solve_chain(odd_indices, start_val=y[0])
    else:
        # N=1的情况，没有奇数索引
        odd_chains = [{}]

    # -----------------------------------------------------
    # 2. 偶数位置链 (Even indices: 0, 2, 4...)
    # -----------------------------------------------------
    # x[0] 是自由变量，有两种可能 (0 或 1)
    even_indices = list(range(0, N, 2))
    even_chains = solve_chain(even_indices, start_val=None)
    
    # -----------------------------------------------------
    # 3. 验证右边界条件: x[N] = 0 (隐式)
    # -----------------------------------------------------
    # 这意味着最后一个 y[N-1] = x[N-2] ^ x[N]
    # 也就是 y[N-1] = x[N-2] ^ 0 = x[N-2]
    # 所有的解必须满足这个约束
    
    valid_states = []
    
    for o_chain in odd_chains:
        for e_chain in even_chains:
            # 合并
            full_state = [0] * N
            # 填充值
            for k, v in o_chain.items():
                full_state[k] = v
            for k, v in e_chain.items():
                full_state[k] = v
                
            # 检查右边界一致性
            # Rule 90 at last position N-1: 
            # next[N-1] = current[N-2] ^ current[N] (current[N] is 0 boundary)
            # => next[N-1] should equal current[N-2]
            if len(full_state) >= 2:
                if y[N-1] != full_state[N-2]:
                    continue
            elif len(full_state) == 1:
                # N=1: y[0] = x[-1] ^ x[1] = 0 ^ 0 = 0. 
                # x[0] is free but unconnected to y[0]? 
                # Wait, for N=1, padded is 0, x0, 0
                # next[0] = 0 ^ 0 = 0. So y[0] must be 0.
                # The above logic: 
                # odd indices: empty.
                # even indices: [0]. x[0] can be 0 or 1.
                # check: y[0]==full[N-2]? N-2 = -1 out of bounds.
                # Special check for N=1
                if y[0] != 0:
                    continue
                    
            valid_states.append(full_state)
            
    return valid_states

def generate_unique_sparse_dataset(num_samples=1000, length=32, save_path="rule90.jsonl"):
    """Generates dataset where each output y has a UNIQUE, SPARM-EST input x."""
    print(f"开始生成数据集: {num_samples} 样本, 长度 {length}...")
    
    count = 0
    total_attempts = 0
    
    with open(save_path, "w") as f:
        while count < num_samples:
            total_attempts += 1
            
            # 1. 随机生成一个 '目标' 状态 (即 next state)
            # 注意：不是所有的随机状态作为 Rule 90 的输出都有解
            # 但我们的算法 solve_rule90_inverse_linear 会返回空列表如果是无解的
            y = [random.randint(0, 1) for _ in range(length)]
            
            # 2. 求解所有可能的原像 (Pre-images)
            candidates = solve_rule90_inverse_linear(y)
            
            if not candidates:
                continue
                
            # 3. 寻找最稀疏的解
            # 计算每个解的汉明重量 (Hamming weight)
            candidates_with_weight = []
            min_weight = length + 1
            
            for cand in candidates:
                w = sum(cand)
                if w < min_weight:
                    min_weight = w
                candidates_with_weight.append((cand, w))
            
            # 4. 检查最稀疏解是否唯一
            best_candidates = [c for c, w in candidates_with_weight if w == min_weight]
            
            if len(best_candidates) == 1:
                x_opt = best_candidates[0]
                
                # 构造样本
                sample = {
                    "input": ''.join(str(b) for b in y),         # N bits
                    "output": ''.join(str(b) for b in x_opt)     # N bits
                }
                f.write(json.dumps(sample) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"已生成 {count}/{num_samples}...")
    
    print(f"✅ 完成! 生成 {count} 样本, 尝试 {total_attempts} 次 (效率: {count/total_attempts:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--length", type=int, default=32, help="Length of binary string")
    parser.add_argument("--save_path", type=str, default="rule90_sparse_unique.jsonl")
    args = parser.parse_args()
    
    generate_unique_sparse_dataset(args.num_samples, args.length, args.save_path)
