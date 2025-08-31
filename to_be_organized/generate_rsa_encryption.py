import json

def int_to_bin_str(x, bits):
    return format(x, f'013b')

def rsa_encrypt(m, e, n):
    return pow(m, e, n)

def generate_rsa_dataset(e=17, n=5429, bits=13, output_file='rsa_16bit_dataset.jsonl'):
    with open(output_file, 'w') as f:
        for m in range(n):
            c = rsa_encrypt(m, e, n)
            input_bin = int_to_bin_str(m, bits)
            output_bin = int_to_bin_str(c, bits)  # 加密后最多为 n-1，仍然能容纳在16位
            sample = {
                "input": input_bin,
                "output": output_bin
            }
            f.write(json.dumps(sample) + '\n')
    print(f"✅ Dataset saved to {output_file}")
# 公钥：(e = 17, n = 5429)
#
# 私钥：(d = 1241, n = 5429)
# def generate_rsa_dataset_13(e=3, n=8388607, bits=12, output_file='rsa_12bit_dataset.jsonl'):
#     dataset = []
#     max_m = 2**bits
#     for m in range(max_m):
#         c = pow(m, e, n)
#         input_str = bin(m)[2:].zfill(bits)
#         output_str = bin(c)[2:].zfill(13)  # 固定输出为13位
#         dataset.append({'input': input_str, 'output': output_str})
#
#     with open(output_file, 'w') as f:
#         for item in dataset:
#             f.write(json.dumps(item) + '\n')
#
#     print(f"Generated {len(dataset)} samples to {output_file}")

if __name__ == "__main__":
    generate_rsa_dataset()