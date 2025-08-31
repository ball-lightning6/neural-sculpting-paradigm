import random
import json

def int_to_bits(n, width):
    return bin(n)[2:].zfill(width)

def generate_hamming_weight_dataset(path, num_samples=10000, input_bits=31, output_bits=5, balanced=False):
    dataset = []

    if balanced:
        max_weight = input_bits
        per_weight = num_samples // (max_weight + 1)
        for weight in range(max_weight + 1):
            for _ in range(per_weight):
                bits = ['0'] * input_bits
                ones_indices = random.sample(range(input_bits), weight)
                for i in ones_indices:
                    bits[i] = '1'
                input_str = ''.join(bits)
                output_str = int_to_bits(weight, output_bits)
                dataset.append({"input": input_str, "output": output_str})
    else:
        for _ in range(num_samples):
            bits = [random.choice('01') for _ in range(input_bits)]
            input_str = ''.join(bits)
            weight = input_str.count('1')
            output_str = int_to_bits(weight, output_bits)
            dataset.append({"input": input_str, "output": output_str})

    with open(path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')

# 用法示例：
generate_hamming_weight_dataset("hamming_dataset.jsonl", num_samples=100000, balanced=True)
