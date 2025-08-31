import json
import random
from itertools import combinations

def generate_subset_sum_example(n_items=4, value_range=(0, 15), min_target=10, max_target=63, min_solution_len=2):
    while True:
        numbers = random.sample(range(value_range[0], value_range[1] + 1), n_items)

        # 枚举所有子集求和
        subset_by_sum = {}
        for r in range(1, n_items + 1):
            for idx in combinations(range(n_items), r):
                s = sum(numbers[i] for i in idx)
                if min_target <= s <= max_target:
                    if s not in subset_by_sum:
                        subset_by_sum[s] = []
                    subset_by_sum[s].append(idx)

        for target, idx_list in subset_by_sum.items():
            # 找出最短解
            min_len = min(len(idx) for idx in idx_list)
            if min_len < min_solution_len:
                continue  # 跳过最短解太短的样本

            min_sets = [idx for idx in idx_list if len(idx) == min_len]
            longer_sets = [idx for idx in idx_list if len(idx) > min_len]

            # 要求唯一最短解，且有其他更长的合法解
            if len(min_sets) == 1 and len(longer_sets) > 0:
                best_idx = min_sets[0]
                output_mask = [1 if i in best_idx else 0 for i in range(n_items)]
                return {
                    "input": ''.join([f'{n:04b}' for n in numbers]) + f'{target:06b}',
                    "output": ''.join(map(str,output_mask))
                }
import random
import json
from itertools import combinations

def generate_unique_subset_example(n_items=6, value_range=(0, 15), min_target=10, max_target=63):
    while True:
        numbers = random.sample(range(value_range[0], value_range[1] + 1), n_items)

        # 枚举所有子集求和
        subset_by_sum = {}
        for r in range(1, n_items + 1):
            for idx in combinations(range(n_items), r):
                s = sum(numbers[i] for i in idx)
                if min_target <= s <= max_target:
                    if s not in subset_by_sum:
                        subset_by_sum[s] = []
                    subset_by_sum[s].append(idx)

        # 只保留 target 只有一个合法解的情况
        for target, idx_list in subset_by_sum.items():
            if len(idx_list) == 1:
                idx = idx_list[0]
                output_mask = [1 if i in idx else 0 for i in range(n_items)]
                return {
                    "input": ''.join([f'{n:04b}' for n in numbers]) + ''.join(map(str,output_mask)),
                    "output": f'{target:06b}'
                }
                # return {
                #     "input": ''.join([f'{n:04b}' for n in numbers]) + f'{target:06b}',
                #     "output": ''.join(map(str,output_mask))
                # }

# 保存多个样本到 JSONL
def write_dataset(filename="subset_sum_min_solution.jsonl", num_samples=300000):
    with open(filename, "w") as f:
        for _ in range(num_samples):
            example = generate_unique_subset_example()
            #print(example)
            f.write(json.dumps(example) + "\n")
write_dataset()