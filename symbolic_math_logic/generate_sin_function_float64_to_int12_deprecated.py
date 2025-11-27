import numpy as np
import struct
import json

def float64_to_bits(f):
    """Convert float64 to 64-bit binary string"""
    return format(struct.unpack('>Q', struct.pack('>d', f))[0], '064b')

def float32_to_bits(f):
    """Convert float32 to 32-bit binary string"""
    return format(struct.unpack('>I', struct.pack('>f', f))[0], '032b')

def bits_to_float64(bits):
    """将长度为64的0/1列表还原为float64数值"""
    assert len(bits) == 64
    bitstring = ''.join(str(b) for b in bits)
    int_val = int(bitstring, 2)
    bytes_val = struct.pack('>Q', int_val)
    return struct.unpack('>d', bytes_val)[0]



def partial_bits_to_float32(bits8, tail_fill='0'):
    """
    将8位的float32前缀（符号位 + 7位尾数）补全为32位，再转换为浮点数
    默认指数位为127（即2^0），尾数其他位用 tail_fill 补齐
    """
    assert len(bits8)==8
    sign = bits8[0]
    mantissa_prefix = bits8[1:]

    # 指数固定为127（二进制 01111111）
    exponent = '01111111'

    # 尾数补齐：7位前缀 + (23-7)位填充
    mantissa = mantissa_prefix + tail_fill * (23 - len(mantissa_prefix))

    full_bits = sign + exponent + mantissa
    int_val = int(full_bits, 2)
    bytes_val = struct.pack('>I', int_val)
    return struct.unpack('>f', bytes_val)[0]

def generate_sin_dataset(num_samples=10, range_min=-10000, range_max=10000, output_bits=8):
    dataset = []
    for _ in range(num_samples):
        x = np.random.uniform(range_min, range_max)
        x_bits = float64_to_bits(x)

        y = int(32768*np.sin(x))

        dataset.append({
            "input": [int(b) for b in x_bits],
            "label": ''.join(map(str,label))
        })
        print(x,y)
        print(bits_to_float64([int(b) for b in x_bits]),partial_bits_to_float32(''.join(map(str,label))))


    with open("sin_dataset_float64bits_to_float32head.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

generate_sin_dataset()
