import pycosat
import random
import json

def generate_clause(num_vars):
    return [random.choice([1, -1]) * random.randint(1, num_vars) for _ in range(3)]

def generate_formula(num_vars, num_clauses):
    return [generate_clause(num_vars) for _ in range(num_clauses)]

def generate_balanced_dataset(n_samples, num_vars=6, num_clauses=6):
    data = []
    count_1 = count_0 = 0
    while len(data) < n_samples:
        formula = generate_formula(num_vars, num_clauses)
        solution = pycosat.solve(formula)
        label = 1 if isinstance(solution, list) else 0

        if (label == 1 and count_1 >= n_samples // 2) or (label == 0 and count_0 >= n_samples // 2):
            continue

        encoded = []
        for clause in formula:
            for literal in clause:
                var_id = abs(literal) - 1
                sign = 1 if literal > 0 else 0
                #binary = f'{var_id:03b}' + str(sign)  # 3 bit var + 1 bit sign
                binary = str(sign)+chr(var_id+97)
                encoded.append(binary)
        input_str = ''.join(encoded)

        data.append({'input': input_str, 'label': label})
        if label == 1:
            count_1 += 1
        else:
            count_0 += 1
        print(count_1, count_0)

    return data

# 写入 jsonl
dataset = generate_balanced_dataset(10000)
with open("balanced_sat_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
