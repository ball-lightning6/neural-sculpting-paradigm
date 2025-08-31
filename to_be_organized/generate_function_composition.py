import json
import random

# 函数列表与实现
FUNC_LIST = ["double", "increment", "square", "decrement"]
FUNC_IMPL = {
    "double": lambda x: 2 * x,
    "increment": lambda x: x + 1,
    "square": lambda x: x * x,
    "decrement": lambda x: x - 1
}

# 2-bit 函数编码
def func_to_bits(func_names):
    return ''.join(format(FUNC_LIST.index(f), '02b') for f in func_names)

# 检查中间步骤是否越界（每一步都必须在 0~65535）
def apply_funcs_strict(func_names, x):
    for f in func_names:
        x = FUNC_IMPL[f](x)
        if x < 0 or x > 65535:
            return None  # 一旦中间越界，就丢弃
    return x

# 生成样本，确保中间值也合法
def generate_example():
    while True:
        func_seq = random.choices(FUNC_LIST, k=4)
        x = random.randint(0, 65535)
        y = apply_funcs_strict(func_seq, x)
        if y is not None:
            input_bits = func_to_bits(func_seq) + format(x, '016b')
            output_bits = format(y, '016b')
            return {
                "input": input_bits,
                "output": output_bits
            }

# 写入 JSONL 文件
with open("function_compose_strict.jsonl", "w") as f:
    for _ in range(100000):  # 根据需求调整
        example = generate_example()
        json.dump(example, f)
        f.write("\n")
