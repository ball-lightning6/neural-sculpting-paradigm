import random
import json
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
NUM_SAMPLES = 2000000  # 200万样本，保证足够的覆盖率
SEQ_LEN = 30           # 每一层的比特数
NUM_LAYERS = 3         # 迭代层数（程序深度）
RULE_BITS = 3          # 表示一个规则需要的位数 (2^3 = 8 rules)
OUTPUT_FILE = "reverse_engineering_5layers.jsonl"

# ================= HELPER FUNCTIONS =================

def int_to_bits(n, length):
    """将整数转换为01列表"""
    return [int(x) for x in f"{n:0{length}b}"]

def bits_to_int(bits):
    """将01列表转换为整数"""
    return int("".join(map(str, bits)), 2)

def cyclic_shift(bits, k):
    """循环左移 k 位"""
    k = k % len(bits)
    return bits[k:] + bits[:k]

def get_neighbors(bits, i):
    """获取元胞自动机的邻居 (Left, Center, Right)"""
    l = bits[i - 1] # Python handles -1 as last element (cyclic)
    c = bits[i]
    r = bits[(i + 1) % len(bits)]
    return l, c, r

# ================= RULE DEFINITIONS =================

# 规则 0-3: 元胞自动机 (CA)
def apply_ca(bits, rule_number):
    """应用一维元胞自动机规则"""
    # 规则表：例如 rule 30 -> [0,0,0,1,1,1,1,0]
    rule_map = int_to_bits(rule_number, 8) 
    new_bits = []
    for i in range(len(bits)):
        l, c, r = get_neighbors(bits, i)
        # 将邻居组合成索引 (e.g. 101 -> 5)
        idx = 7 - (l * 4 + c * 2 + r) 
        new_bits.append(rule_map[idx])
    return new_bits

# 规则 4: 位取反 (NOT)
def apply_not(bits):
    return [1 - b for b in bits]

# 规则 5: 移位加法 (Add with Shift)
# 逻辑：(x + (x << 15)) % 2^30
def apply_add_shift(bits):
    val_x = bits_to_int(bits)
    val_shifted = bits_to_int(cyclic_shift(bits, 15))
    # 模拟30位整数加法溢出
    res_val = (val_x + val_shifted) % (2**SEQ_LEN)
    return int_to_bits(res_val, SEQ_LEN)

# 规则 6: 动态移位 (Dynamic Shift / Control Flow)
# 逻辑：取前5位作为参数 k，循环左移 k 位
def apply_dynamic_shift(bits):
    # 取前5位，最大移动 31 位
    k = bits_to_int(bits[:5])
    return cyclic_shift(bits, k)
def apply_majority(bits):
    new_bits = []
    for i in range(len(bits)):
        l, c, r = get_neighbors(bits, i)
        if (l + c + r) >= 2:
            new_bits.append(1)
        else:
            new_bits.append(0)
    return new_bits


# 规则 7: 长程异或 (Long-range XOR)
# 逻辑：x ^ (x << 10)
def apply_long_or(bits):
    shifted = cyclic_shift(bits, 10)
    return [b | s for b, s in zip(bits, shifted)]
def apply_shift_part_reverse(bits):
    return bits[1::3]+[1 - x for x in bits[2::3]]+bits[0::3][::-1]
# 规则映射表
RULES = {
    0: lambda b: apply_ca(b, 30),   # Chaos
    1: lambda b: apply_ca(b, 110),  # Turing Complete
    2: lambda b: apply_ca(b, 167),   # Fractal / Linear
    3: lambda b: apply_ca(b, 184),  # Traffic Flow
    4: apply_majority,                   # Bitwise NOT
    5: apply_add_shift,             # Global Arithmetic
    6: apply_dynamic_shift,         # Data-dependent Control Flow (Hard!)
    7: apply_shift_part_reverse               # Long-range Logic
}

# ================= MAIN GENERATION =================

def generate_sample():
    # 1. 生成随机初始状态 X
    x_bits = [random.randint(0, 1) for _ in range(SEQ_LEN)]
    
    # 2. 生成随机规则序列 (5层，每层3bit)
    # 例如: [2, 0, 5, 7, 1] 代表先后执行 Rule 2 -> Rule 0 ...
    rule_indices = [random.randint(0, 7) for _ in range(NUM_LAYERS)]
    
    # 3. 将规则序列转换为二进制标签 (Label)
    # 5 * 3 = 15 bits
    label_bits = []
    for r in rule_indices:
        label_bits.extend(int_to_bits(r, RULE_BITS))
        
    # 4. 执行程序
    current_bits = x_bits[:]
    # 逐步应用规则
    for rule_idx in rule_indices:
        current_bits = RULES[rule_idx](current_bits)
        
    y_bits = current_bits
    
    # 5. 构造模型输入：X + Y (拼接)
    # 输入是 60 位，输出是 15 位
    model_input = x_bits + y_bits
    
    return {
        "input": "".join(map(str, model_input)),
        "output": label_bits,
        "debug_rule_seq": rule_indices # 方便人工检查
    }

if __name__ == "__main__":
    print(f"Generating {NUM_SAMPLES} samples...")
    print(f"Input Length: {SEQ_LEN * 2} bits (Input State + Output State)")
    print(f"Target Length: {NUM_LAYERS * RULE_BITS} bits ({NUM_LAYERS} steps program)")
    
    # 预览一个样本
    sample = generate_sample()
    print("\nSample Preview:")
    print(f"Input (X+Y): {sample['input']}")
    print(f"Target (Rules): {sample['output']}")
    print(f"Actual Rules: {sample['debug_rule_seq']}")
    
    with open(OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(NUM_SAMPLES)):
            s = generate_sample()
            # 只保存模型需要的字段
            json_line = json.dumps({
                "input": s["input"],
                "output": s["output"]
            })
            f.write(json_line + "\n")
            
    print(f"\nDone! Dataset saved to {OUTPUT_FILE}")
