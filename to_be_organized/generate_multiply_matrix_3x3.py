import random
import json

def matmul_binary(a, b):
    # 3x3 矩阵乘法，输入 a, b 是长度为 9 的 0/1 列表
    result = []
    for i in range(3):  # 行
        for j in range(3):  # 列
            sum_val = 0
            for k in range(3):
                sum_val += a[i * 3 + k] * b[k * 3 + j]
            result.append(sum_val)
    return result  # 长度 9，元素为 0~3

def int_to_2bit(x):
    return format(x, "02b")

t_set = set()
def generate_binary_matmul_jsonl(filename="binary_matmul_3x3.jsonl", num_samples=131072):
    with open(filename, "w") as f:
        #for _ in range(num_samples):
        while len(t_set)<num_samples:
            a = [random.randint(0, 1) for _ in range(9)]
            b = [random.randint(0, 1) for _ in range(9)]
            s = "".join(map(str, a+b))
            if s not in t_set:
                t_set.add(s)
            else:
                continue
            c = matmul_binary(a, b)  # 得到 9 个 0~3 的值
            c_bits = "".join(int_to_2bit(x) for x in c)  # 转为 18 位输出
            sample = {
                # "input1": "".join(map(str, a)),  # 9位
                # "input2": "".join(map(str, b)),  # 9位
                "input": "".join(map(str, a+b)),
                "output": list(int(x) for x in c_bits)                 # 18位，01字符串
            }
            f.write(json.dumps(sample) + "\n")

# 运行脚本
generate_binary_matmul_jsonl()