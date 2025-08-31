import random
import json
from itertools import product

def rule90_next(state):
    """应用Rule 90演化一次"""
    padded = [0] + state + [0]
    return [padded[i-1] ^ padded[i+1] for i in range(1, len(padded)-1)]

def random_binary_list(length):
    return [random.randint(0, 1) for _ in range(length)]

def to_str(lst):
    return ''.join(str(x) for x in lst)

def generate_unique_sparse_inverse(next_state):
    n = len(next_state)# + 2
    candidates = []
    for bits in product([0, 1], repeat=n):
        if rule90_next(list(bits)) == next_state:
            candidates.append(bits)
    # 统计1的个数作为稀疏度指标
    sparsity_map = {}
    for c in candidates:
        ones = sum(c)
        sparsity_map.setdefault(ones, []).append(c)
    # 最小稀疏度是否唯一
    min_ones = min(sparsity_map)
    if len(sparsity_map[min_ones]) == 1:
        return list(sparsity_map[min_ones][0])
    else:
        return None

def generate_dataset(num_samples=1000, length=15, save_path="rule90_sparse_unique.jsonl"):
    with open(save_path, "w") as f:
        count = 0
        attempts = 0
        while count < num_samples:
            original = random_binary_list(length + 2)
            next_state = rule90_next(original)
            optimal = generate_unique_sparse_inverse(next_state)
            attempts += 1
            if optimal:
                sample = {
                    "input": to_str(next_state),
                    "output": to_str(optimal)
                }
                f.write(json.dumps(sample) + "\n")
                count += 1
        print(f"✅ 生成完成：{count}个样本，共尝试了{attempts}次（保留率约{count/attempts:.2%}）")

# 示例调用
if __name__ == "__main__":
    generate_dataset(num_samples=100, length=15)
