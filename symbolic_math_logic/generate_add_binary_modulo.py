import random
import json

def int_to_bitstring(x: int, bit_length: int = 16) -> str:
    return format(x, f'0{bit_length}b')

def generate_addition_dataset(n_samples: int, bit_length: int = 8, seed: int = 42):
    random.seed(seed)
    data = []

    max_val = 2 ** bit_length
    for _ in range(n_samples):
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)
        c = (a + b) % max_val  # 截断加法
        input_str = int_to_bitstring(a, bit_length) + int_to_bitstring(b, bit_length)
        output_str = int_to_bitstring(c, bit_length)
        data.append((input_str, output_str))

    return data

def save_dataset(dataset, file_path: str):
    with open(file_path, 'w') as f:
        for input_str, output_str in dataset:
            # 合并input并转换output为多标签二分类格式
            f.write(json.dumps({'input': input_str, 'output': [int(bit) for bit in output_str]}) + '\n')

if __name__ == "__main__":
    dataset = generate_addition_dataset(n_samples=2048)
    save_dataset(dataset, "add_binary_modulo_dataset.jsonl")
    print("数据集已保存为 add_binary_modulo_dataset.jsonl")
