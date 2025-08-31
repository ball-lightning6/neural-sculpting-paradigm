import random
import json

n_samples = 10000
rules = {
    5: (3, 4),  # rule1: 3+4 => 5
    6: (1, 2)   # rule2: 1+2 => 6
}

all_attrs = list(range(8))

data = []
s= set()
for _ in range(n_samples):
    query_attr = random.choice(list(rules.keys()))
    antecedents = list(rules[query_attr])

    is_positive = random.random() < 0.5
    bitstring = ['0'] * 8

    if is_positive:
        for a in antecedents:
            bitstring[a] = '1'
        # 加入额外干扰项
        extra_attrs = list(set(all_attrs) - set(antecedents + [query_attr]))
        for i in random.sample(extra_attrs, random.randint(0, 3)):
            bitstring[i] = '1'
        label = "1"
    else:
        # 缺一个前提或两个都缺
        n_present = random.choice([0, 1])
        present = random.sample(antecedents, n_present)
        for a in present:
            bitstring[a] = '1'
        noise_pool = list(set(all_attrs) - set(antecedents + [query_attr]))
        for i in random.sample(noise_pool, random.randint(1, 3)):
            bitstring[i] = '1'
        label = "0"

    input_str = ''.join(bitstring) + f"{int(query_attr)-5}"
    s.add(input_str)
    data.append({"input": input_str, "output": label})

print(len(s))

with open("bitstring_rule_learning.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
