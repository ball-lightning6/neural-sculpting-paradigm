import random

import random
import json


def generate_depth1_jsonl(num_samples=10000, attr_range=(2, 30), seed=42, filename="depth1_samples.jsonl"):
    random.seed(seed)
    attrs = list(range(attr_range[0], attr_range[1] + 1))

    with open(filename, "w") as f:
        for _ in range(num_samples):
            label = random.choice([0, 1])  # 正负样本平衡
            a, b, c = random.sample(attrs, 3)
            rule = f"({a},{b}|{c})"

            if label==1:
                facts = f"{a}, {b}"
            else:
                facts = f"{random.choice([a, b])}"  # 漏掉一个前提，变负样本

            query = str(c)

            input_text = f"Facts: {facts}\nRules: {rule}\nQuery: {query}"
            output_text = str(label)

            f.write(json.dumps({"input": input_text, "output": output_text}) + "\n")


def generate_depth2_jsonl(num_samples=10000, attr_range=(2, 40), seed=42, filename="depth2_samples.jsonl"):
    random.seed(seed)
    attrs = list(range(attr_range[0], attr_range[1] + 1))

    with open(filename, "w") as f:
        for _ in range(num_samples):
            label = random.choice([0, 1])

            # 三步的中转变量：a,b → c；c,d → e
            a, b, d, e = random.sample(attrs, 4)
            c = random.choice([i for i in attrs if i not in [a, b, d, e]])

            rule1 = f"({a},{b}|{c})"
            rule2 = f"({c},{d}|{e})"
            rules = f"{rule1}; {rule2}"

            if label==1:
                facts = f"{a}, {b}, {d}"
            else:
                # 随机漏掉一个关键前提，使无法两步推出 e
                remove = random.choice([[a], [b], [d], [a, d], [b, d]])
                fact_set = {a, b, d} - set(remove)
                facts = ", ".join(str(x) for x in fact_set)

            input_text = f"Facts: {facts}\nRules: {rules}\nQuery: {e}"
            output_text = str(label)

            f.write(json.dumps({"input": input_text, "output": output_text}) + "\n")


def generate_depth5_jsonl(num_samples=10000, attr_range=(2, 100), seed=42, filename="depth5.jsonl"):
    random.seed(seed)
    attrs = list(range(attr_range[0], attr_range[1] + 1))

    with open(filename, "w") as f:
        for _ in range(num_samples):
            label = random.choice([0, 1])
            # 需要 7 个不同属性用于串联：a,b → c → d → e → f → g
            x = random.sample(attrs, 7)  # x[0]~x[6]
            rules = [f"({x[0]},{x[1]}|{x[2]})", f"({x[2]},{x[3]}|{x[4]})",
                     f"({x[4]},{x[5]}|{x[6]})", f"({x[6]},{x[1]}|{x[5] + 1})",
                     f"({x[5] + 1},{x[3]}|{x[5] + 2})"]
            rules_str = "; ".join(rules)

            if label==1:
                facts = [x[0], x[1], x[3], x[5]]
            else:
                remove = random.choice([[x[0]], [x[1]], [x[3]], [x[5]], [x[1], x[3]]])
                facts = list(set([x[0], x[1], x[3], x[5]]) - set(remove))
            facts_str = ", ".join(str(f) for f in sorted(facts))

            query = str(x[5] + 2)
            input_text = f"Facts: {facts_str}\nRules: {rules_str}\nQuery: {query}"
            f.write(json.dumps({"input": input_text, "output": str(label)}) + "\n")
# 生成样本并保存
samples = generate_depth5_jsonl(num_samples=30000)  # 你可以改为更多
