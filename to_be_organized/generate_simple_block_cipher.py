import json
import random
import numpy as np

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
INPUT_BITS = 64
NUM_ROUNDS = 32  # 加密的轮数，轮数越多越“混沌”，越难学
DATASET_SIZE = 1000000  # 这是一个困难任务，需要更多数据

TRAIN_FILE = f'tcipher_{INPUT_BITS}bit_r{NUM_ROUNDS}_train.jsonl'
EVAL_FILE = f'tcipher_{INPUT_BITS}bit_r{NUM_ROUNDS}_eval.jsonl'

# ==============================================================================
# --- 2. “隐藏的宇宙法则”：T-Cipher的固定密钥 ---
# ==============================================================================
random.seed(42)
# 生成一个一次性的、固定的、隐藏的64-bit轮密钥
HIDDEN_KEY = random.getrandbits(INPUT_BITS)

print("=" * 70)
print(f"     “神经密码分析师”实验 - T-Cipher 数据集生成器")
print("=" * 70)
print(f"输入/输出长度: {INPUT_BITS} bits")
print(f"加密轮数: {NUM_ROUNDS}")
# 我们不打印密钥，因为它是“宇宙的秘密”
# print(f"隐藏的密钥 (十进制): {HIDDEN_KEY}")
print("=" * 70)


# ==============================================================================
# --- 3. T-Cipher 核心加密逻辑 ---
# ==============================================================================

def sub_bytes(state_int):
    """字节代换层：对每个字节进行非线性变换"""
    new_state_int = 0
    # 将64位的整数看作8个8位的字节
    for i in range(8):
        byte = (state_int >> (i * 8)) & 0xFF
        # 简单的非线性变换
        new_byte = (byte * 5 + 1) & 0xFF  # 使用 & 0xFF 确保结果在8位内
        new_state_int |= (new_byte << (i * 8))
    return new_state_int


def shift_rows(state_int):
    """行移位层：将字节看作2x4矩阵并循环移位"""
    # 提取8个字节
    b = [(state_int >> i) & 0xFF for i in range(0, 64, 8)]
    # 将其视为 2x4 矩阵:
    # b0 b1 b2 b3
    # b4 b5 b6 b7
    # 第二行 (b4,b5,b6,b7) 左移1位 -> (b5,b6,b7,b4)
    new_b = b[:4] + [b[5], b[6], b[7], b[4]]

    # 重新组合成64位整数
    new_state_int = 0
    for i in range(8):
        new_state_int |= (new_b[i] << (i * 8))
    return new_state_int


def encrypt_t_cipher(plaintext_int, key_int, num_rounds):
    """完整的T-Cipher加密流程"""
    state = plaintext_int
    for _ in range(num_rounds):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = state ^ key_int  # 轮密钥加 (XOR)
    return state


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits, num_rounds, hidden_key):
    print("\n--- 开始生成 (明文 -> 密文) 数据集 ---")

    records = []
    # 在这里，我们不需要去重，因为随机生成的明文几乎不可能重复
    for i in range(num_samples):
        # 随机生成一个明文
        plaintext = random.getrandbits(num_bits)

        # 用T-Cipher加密得到密文
        ciphertext = encrypt_t_cipher(plaintext, hidden_key, num_rounds)

        # 编码为二进制字符串和多标签列表
        input_str = format(plaintext, f'0{num_bits}b')
        output_str = format(ciphertext, f'0{num_bits}b')
        output_multilabel = [int(bit) for bit in output_str]

        records.append({"input": input_str, "output": output_multilabel})

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    random.shuffle(records)
    train_size = int(len(records) * 1)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in data: f.write(json.dumps(record) + '\n')
        print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        with open(path[1], 'w') as f:
            for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, INPUT_BITS, NUM_ROUNDS, HIDDEN_KEY)