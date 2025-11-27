import numpy as np
import struct
import json

# 数据量和范围
N = 1_000_000
x_range = (-10 * np.pi, 10 * np.pi)

# 生成数据
x_values = np.random.uniform(x_range[0], x_range[1], size=N).astype(np.float32)
y_values = np.sin(x_values)

# 转 float32 -> 32位二进制
def float32_to_bits(f):
    [d] = struct.unpack(">I", struct.pack(">f", f))
    return [int(b) for b in f"{d:032b}"]

# sin(x) -> int in [-8388607, +8388607] -> 24位有符号二进制
def sin_to_24bit(y):
    max_int = (1 << 23) - 1  # 8388607
    y_int = int(np.clip(round(y * max_int), -max_int, max_int))
    if y_int < 0:
        y_int = (1 << 24) + y_int  # 补码表示
    return [int(b) for b in f"{y_int:024b}"]

# 转换所有样本
with open("float32_input_sin24bit_output.jsonl", "w") as f:
    for x, y in zip(x_values, y_values):
        x_bits = float32_to_bits(x)
        y_bits = sin_to_24bit(y)
        json_line = json.dumps({"input_bits": x_bits, "output_bits": y_bits})
        f.write(json_line + "\n")
