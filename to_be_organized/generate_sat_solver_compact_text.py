import json
import random
from z3 import Solver, Bool, Or, Not, sat

# 参数设置
VAR_COUNT = 6
NUM_SAMPLES_PER_CLASS = 50  # 每个类别的样本数
NUM_CLAUSES = 20
OUTPUT_PATH = "sat_toy_balanced_100k.jsonl"

# 字符编码映射：小写正 literal，大写负 literal
LITERAL2CHAR = { (i, True): chr(ord('a') + i) for i in range(VAR_COUNT) }
LITERAL2CHAR.update({ (i, False): chr(ord('A') + i) for i in range(VAR_COUNT) })
z3_vars = [Bool(f"x{i}") for i in range(VAR_COUNT)]

def random_clause():
    """随机生成一个子句（三个不同变量）"""
    indices = random.sample(range(VAR_COUNT), 3)
    return [(i, random.choice([True, False])) for i in indices]

def clause_to_z3(clause):
    return Or(*[z3_vars[i] if sign else Not(z3_vars[i]) for (i, sign) in clause])

def clause_to_string(clause):
    return ''.join([LITERAL2CHAR[(i, sign)] for (i, sign) in clause])

def generate_formula():
    clauses = [random_clause() for _ in range(NUM_CLAUSES)]
    formula_str = ''.join([clause_to_string(c) for c in clauses])
    formula_z3 = [clause_to_z3(c) for c in clauses]
    return formula_str, formula_z3

def generate_balanced_dataset(path, num_per_class):
    solver = Solver()
    pos, neg = 0, 0
    with open(path, 'w') as f:
        while pos < num_per_class or neg < num_per_class:
            formula_str, formula_z3 = generate_formula()
            solver.reset()
            solver.add(formula_z3)
            is_sat = solver.check() == sat
            label = "1" if is_sat else "0"
            if is_sat and pos < num_per_class:
                f.write(json.dumps({"input": formula_str, "label": label}) + '\n')
                pos += 1
            elif not is_sat and neg < num_per_class:
                f.write(json.dumps({"input": formula_str, "label": label}) + '\n')
                neg += 1
            print(pos,neg)

if __name__ == "__main__":
    generate_balanced_dataset(OUTPUT_PATH, NUM_SAMPLES_PER_CLASS)
