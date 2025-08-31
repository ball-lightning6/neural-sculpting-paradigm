import random
import json

def int_to_bitstring(n, bits=8):
    return format(n, f'0{bits}b')

def generate_mul_dataset_jsonl(filename='mul_8bit_dataset.jsonl', num_samples=2048, bits=8):
    max_val = 2 ** bits
    with open(filename, 'w') as f:
        for _ in range(num_samples):
            a = random.randint(0, max_val - 1)
            b = random.randint(0, max_val - 1)
            product = (a * b) % max_val  # 截断结果
            sample = {
                # 'input1': int_to_bitstring(a, bits),
                # 'input2': int_to_bitstring(b, bits),
                'input': int_to_bitstring(a, bits) + int_to_bitstring(b, bits),
                'output': list(int(x) for x in int_to_bitstring(product, bits))
            }
            f.write(json.dumps(sample) + '\n')

# 执行生成
generate_mul_dataset_jsonl('mul_8bit_dataset.jsonl', num_samples=8192)
