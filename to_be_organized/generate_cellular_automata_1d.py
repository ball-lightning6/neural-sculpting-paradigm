import random
import json

# Rule 110 转换表
rule_110 = {
    "111": "0",
    "110": "1",
    "101": "1",
    "100": "0",
    "011": "1",
    "010": "1",
    "001": "1",
    "000": "0"
}

rule_30 = {
    "111": "0",
    "110": "0",
    "101": "0",
    "100": "1",
    "011": "1",
    "010": "1",
    "001": "1",
    "000": "0"
}

def evolve(state):
    # 循环边界条件：左右首尾相连
    padded = state[-1] + state + state[0]
    return "".join(rule_110[padded[i:i+3]] for i in range(len(state)))

from tqdm import tqdm

def generate_dataset(num_samples=300000, length=36,l=6):
    output_path = f"ca_rule110_layer{l}_{length}.jsonl"
    with open(output_path, "w") as f:
        for _ in tqdm(range(num_samples)):
            input_seq = "".join(random.choice("01") for _ in range(length))
            output_seq = input_seq
            for _ in range(l):
                output_seq = evolve(output_seq)
            output_seq = list(map(int,output_seq))
            sample = {
                "input": input_seq,
                "output": output_seq
            }
            f.write(json.dumps(sample) + "\n")

# 生成数据集
generate_dataset()