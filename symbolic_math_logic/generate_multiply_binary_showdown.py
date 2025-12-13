import json
import random
from tqdm import tqdm
import math

# 核心参数配置
NUM_BITS = 16
DATASET_SIZE = 500000

OUTPUT_FILE = f'multiplier_{NUM_BITS}bit_showdown.jsonl'

# 标签维度计算
INPUT_LEN = NUM_BITS * 2
EXPLAIN_A_LEN = (NUM_BITS - 1) * (NUM_BITS * 2)
NUM_COUNTERS = NUM_BITS * 2 - 1
BITS_PER_COUNTER = math.ceil(math.log2(NUM_BITS + 1))
EXPLAIN_B_LEN = NUM_COUNTERS * BITS_PER_COUNTER
EXPLAIN_C_LEN = (NUM_BITS) + (NUM_BITS + 1) + (NUM_BITS)
FINAL_PRODUCT_LEN = NUM_BITS * 2

print("=" * 70)
print(f"{NUM_BITS}-bit 乘法 - 三算法对决数据集生成器")
print("=" * 70)
print(f"输入长度: {INPUT_LEN}")
print(f"输出标签:")
print(f"  - final_product: {FINAL_PRODUCT_LEN} bits")
print(f"  - explain_progressive_sum: {EXPLAIN_A_LEN} bits")
print(f"  - explain_carryless_counters: {EXPLAIN_B_LEN} bits")
print(f"  - explain_karatsuba: {EXPLAIN_C_LEN} bits")
print("=" * 70)


def explain_progressive_sum(a, b, n):
    """逐行累加部分积"""
    partial_products = []
    for i in range(n):
        if (b >> i) & 1:
            partial_products.append(a << i)
        else:
            partial_products.append(0)
    
    partial_sums_trace = []
    current_sum = partial_products[0]
    for i in range(1, n - 1):
        current_sum += partial_products[i]
        sum_str = format(current_sum, f'0{n * 2}b')
        partial_sums_trace.extend([int(bit) for bit in sum_str])
    
    assert len(partial_sums_trace) == EXPLAIN_A_LEN
    return partial_sums_trace


def explain_carryless_counters(a_bin, b_bin, n):
    """无进位列和"""
    counters = [0] * (2 * n - 1)
    for i in range(n):
        if b_bin[n - 1 - i] == '1':
            for j in range(n):
                if a_bin[n - 1 - j] == '1':
                    counters[i + j] += 1
    
    counter_bits_str = "".join([format(c, f'0{BITS_PER_COUNTER}b') for c in counters])
    flat_list = [int(bit) for bit in counter_bits_str]
    assert len(flat_list) == EXPLAIN_B_LEN
    return flat_list


def karatsuba(a, b):
    """Karatsuba分解"""
    n = max(a.bit_length(), b.bit_length())
    if n <= 2:
        return a * b, None, None, None
    
    n_half = (n + 1) // 2
    mask = (1 << n_half) - 1
    
    a_high, a_low = a >> n_half, a & mask
    b_high, b_low = b >> n_half, b & mask
    
    z2 = a_high * b_high
    z0 = a_low * b_low
    z1 = (a_high + a_low) * (b_high + b_low)
    
    return a * b, z0, z1, z2


def explain_karatsuba(a, b, n):
    """Karatsuba递归分解"""
    _, z0, z1, z2 = karatsuba(a, b)
    if z0 is None:
        z0, z1, z2 = 0, 0, 0
    
    z0_str = format(z0, f'0{n}b')
    z1_str = format(z1, f'0{n+1}b')
    z2_str = format(z2, f'0{n}b')
    
    flat_list = [int(bit) for bit in (z0_str + z1_str + z2_str)]
    assert len(flat_list) == EXPLAIN_C_LEN
    return flat_list


def generate_datasets():
    print("\n--- 开始生成数据集 ---")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            max_val = 2 ** NUM_BITS - 1
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            a_bin = format(a, f'0{NUM_BITS}b')
            b_bin = format(b, f'0{NUM_BITS}b')
            
            product = a * b
            final_product_label = [int(bit) for bit in format(product, f'0{FINAL_PRODUCT_LEN}b')]
            
            exp_a = explain_progressive_sum(a, b, NUM_BITS)
            exp_b = explain_carryless_counters(a_bin, b_bin, NUM_BITS)
            exp_c = explain_karatsuba(a, b, NUM_BITS)
            
            record = {
                "input": a_bin + b_bin,
                "final_product": final_product_label,
                "explain_progressive_sum": exp_a,
                "explain_carryless_counters": exp_b,
                "explain_karatsuba": exp_c
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")


if __name__ == "__main__":
    generate_datasets()
