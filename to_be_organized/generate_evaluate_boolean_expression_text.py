import json
import random

def random_expr(vars):
    ops = ['&', '|']
    terms = [f'x{i}' for i in range(len(vars))]
    expr = terms[0]
    for i in range(1, len(vars)):
        op = random.choice(ops)
        expr = f'({expr}{op}{terms[i]})'
    return expr

def eval_expr(expr, values):
    env = {f'x{i}': values[i] for i in range(len(values))}
    safe_expr = expr.replace('&', ' and ').replace('|', ' or ').replace('!', ' not ')
    return int(eval(safe_expr, {}, env))

with open("bool_eval_dataset.jsonl", "w") as f:
    for _ in range(20000):
        num_vars = random.randint(2, 4)
        values = [random.randint(0, 1) for _ in range(num_vars)]
        x_str = ''.join(str(v) for v in values)
        expr = random_expr(values)
        result = eval_expr(expr, values)
        item = {
            "input": f"x={x_str};expr=({expr})",
            "output": str(result)
        }
        f.write(json.dumps(item) + "\n")
