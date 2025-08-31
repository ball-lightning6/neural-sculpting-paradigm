import random
import json

# 参数
n_samples = 10000
rules = {
    5: (3, 4),   # rule1: 3+4 -> 5
    6: (1, 2)    # rule2: 7+8 -> 9
}
all_attrs = set(range(10))

data = []

for _ in range(n_samples):
    query_attr = random.choice(list(rules.keys()))
    antecedents = rules[query_attr]

    is_positive = random.random() < 0.5
    if is_positive:
        facts = list(antecedents)
        noise_pool = list(all_attrs - set(antecedents) - {query_attr})
        noise = random.sample(noise_pool, random.randint(0, 2))
        facts += noise
        label = "1"
    else:
        # 随机缺失前提之一
        missing = random.choice(antecedents)
        present = [x for x in antecedents if x != missing]
        noise_pool = list(all_attrs - set(antecedents) - {query_attr})
        facts = present + random.sample(noise_pool, random.randint(1, 3))
        label = "0"

    random.shuffle(facts)
    input_str = f"Facts: {', '.join(str(f) for f in facts)}\nQuery: {query_attr}"
    data.append({"input": input_str, "output": label})

# 写入文件
with open("rule_learning_double_rule.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
