import numpy as np
import json
import struct
import math


def float_to_bits(f):
    """将 float32 转换为 32 位二进制列表"""
    [d] = struct.unpack(">I", struct.pack(">f", f))
    return [int(b) for b in format(d, "032b")]


def bits_to_float(bits):
    """将 32 位二进制列表转换为 float32"""
    bstr = "".join(str(b) for b in bits)
    i = int(bstr, 2)
    return struct.unpack(">f", struct.pack(">I", i))[0]


def generate_dataset(num_samples=1000000, min_val=-math.pi, max_val=math.pi, output_file="sin_float_bits_dataset.jsonl"):
    data = []
    for _ in range(num_samples):
        x = np.random.uniform(min_val, max_val)
        y = math.sin(x)
        input_bits = float_to_bits(np.float32(x))
        output_bits = float_to_bits(np.float32(y))
        data.append({"input_bits": input_bits, "output_bits": output_bits})

    # 保存为 JSONL 文件
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ 数据集已生成，共 {num_samples} 条，保存为 {output_file}")


# 示例调用
if __name__=="__main__":
    generate_dataset(
        num_samples=10000,
        min_val=math.pi-0.1,
        max_val=math.pi,
        output_file="sin_float_bits_dataset_test_pi.jsonl"
    )
