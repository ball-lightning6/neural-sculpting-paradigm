import random
import json

def generate_single_sample(depth=5, num_attrs=16):
    attr_bits = [0] * num_attrs
    rules = {}
    used_targets = set()
    used_sources = set()

    chain = []
    available_attrs = list(range(num_attrs))

    while len(chain) < depth:
        remaining = list(set(available_attrs) - used_targets)
        if len(remaining) < 3:
            break
        a1, a2 = random.sample(remaining, 2)
        candidate_targets = list(set(available_attrs) - set([a1, a2]) - used_targets)
        if not candidate_targets:
            break
        target = random.choice(candidate_targets)
        rules[target] = (a1, a2)
        chain.append(target)
        used_targets.add(target)
        used_sources.update([a1, a2])

    if not chain:
        raise ValueError("Failed to generate valid rule chain")

    target_attr = chain[-1]

    facts = set()
    def backchain(attr):
        if attr in rules:
            a1, a2 = rules[attr]
            backchain(a1)
            backchain(a2)
        facts.add(attr)
    backchain(target_attr)

    for f in facts:
        attr_bits[f] = 1

    input_bits = ''.join(str(b) for b in attr_bits)
    query_bits = format(target_attr, '04b')
    input_str = input_bits + query_bits
    output_str = '1'

    non_facts = list(set(range(num_attrs)) - facts)
    if not non_facts:
        neg_input_str, neg_output_str = input_str, '1'
    else:
        neg_query = random.choice(non_facts)
        neg_input_str = input_bits + format(neg_query, '04b')
        neg_output_str = '0'

    return (input_str, output_str), (neg_input_str, neg_output_str)

def generate_dataset(num_samples=100000, depth=5):
    dataset = []
    for _ in range(num_samples):
        try:
            pos, neg = generate_single_sample(depth=depth)
            dataset.append({"input": pos[0], "output": pos[1]})
            dataset.append({"input": neg[0], "output": neg[1]})
        except ValueError:
            continue
    return dataset

dataset = generate_dataset()
with open("structured_reasoning_depth5.jsonl", "w") as f:
    for d in dataset:
        f.write(json.dumps(d) + "\n")
