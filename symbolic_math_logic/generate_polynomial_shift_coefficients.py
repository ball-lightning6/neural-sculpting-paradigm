import json
from itertools import product
from math import comb

def encode_int_to_bits(n, bits):
    return format(n, f'0{bits}b')

def poly_eval_at_shifted(coeffs, shift=1):
    """Evaluate polynomial g(x + shift), return new coefficients"""
    deg = len(coeffs) - 1
    new_coeffs = [0] * (deg + 1)
    for k in range(deg + 1):
        for j in range(k + 1):
            new_coeffs[j] += coeffs[k] * comb(k, j) * (shift ** (k - j))
    return new_coeffs

def generate_dataset_jsonl(filename, max_samples=None):
    count = 0
    with open(filename, 'w') as f:
        for coeffs in product(range(8), repeat=6):  # a_0 ~ a_5 ∈ [0,7]
            input_bits = ''.join(encode_int_to_bits(c, 3) for c in coeffs)
            result_coeffs = poly_eval_at_shifted(coeffs, shift=1)
            output_bits = ''.join(encode_int_to_bits(c, 8) for c in result_coeffs)
            json.dump({'input': input_bits, 'output': output_bits}, f)
            f.write('\n')
            count += 1
            if max_samples and count >= max_samples:
                break

# 使用方式：生成前 10000 个样本
generate_dataset_jsonl('poly_shift_train.jsonl', max_samples=32768)
