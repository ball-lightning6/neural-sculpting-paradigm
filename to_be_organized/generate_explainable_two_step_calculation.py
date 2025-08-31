import random
import json

def to_bin8(n):
    return format(n, '08b')

def safe_op(a, b, op):
    if op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '*':
        result = a * b
    else:
        raise ValueError("Unknown operator")
    if 0 <= result <= 255:
        return result
    else:
        return None

def generate_sample():
    while True:
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        c = random.randint(0, 255)
        # d = random.randint(0, 255)
        op1 = random.choice(['+', '-'])#, '*'])
        op2 = random.choice(['+', '-'])#, '*'])
        # op3 = random.choice(['+', '-'])#, '*'])

        step1 = safe_op(a, b, op1)
        if step1 is None:
            continue
        step2 = safe_op(step1, c, op2)
        if step2 is None:
            continue
        # step3 = safe_op(step2, d, op3)
        # if step3 is None:
        #     continue

        input_str = to_bin8(a) + op1 + to_bin8(b) + op2 + to_bin8(c)# + op3 + to_bin8(d)
        output_str = to_bin8(step1) + to_bin8(step2)# + to_bin8(step3)
        return {"input": input_str, "output": output_str}

def generate_dataset(filename, count=100000):
    with open(filename, 'w') as f:
        for _ in range(count):
            sample = generate_sample()
            f.write(json.dumps(sample) + '\n')

# 用法示例
generate_dataset('math_trace_dataset.jsonl', count=100000)
