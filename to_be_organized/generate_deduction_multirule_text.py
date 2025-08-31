import random
import json

# 参数配置
n_samples = 10000
num_attributes = 10  # 属性编号 0~9
target_rule = (3, 4)  # 如果同时有 3 和 4 → 推出 5
target_label = 5

data = []

for _ in range(n_samples):
    is_positive = random.random() < 0.5

    if is_positive:
        # 正样本：包含前提 3 和 4
        facts = list(target_rule)
        # 加些干扰属性
        noise_pool = list(set(range(num_attributes)) - set(target_rule) - {target_label})
        noise = random.sample(noise_pool, random.randint(0, 2))
        facts += noise
        label = "1"
    else:
        # 负样本：不包含 3 和 4 同时
        noise_pool = list(set(range(num_attributes)) - set(target_rule) - {target_label})
        facts = random.sample(noise_pool, random.randint(1, 3))
        # 50% 概率包含一条前提（更难）
        if random.random() < 0.5:
            facts.append(random.choice(target_rule))
        label = "0"

    random.shuffle(facts)
    fact_str = ", ".join(str(f) for f in facts)
    input_str = f"Facts: {fact_str}\nQuery: {target_label}"
    data.append({"input": input_str, "output": label})

# 保存为 JSONL 文件
with open("rule_learning_pure_numeric.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
