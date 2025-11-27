import random
import json
import sys

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 10000  # 数据集大小
OPS = {'+': '01', '-': '10'}

VAL_RANGE = (1, 127)
X_VAL_RANGE = (0, 255)
X_APPEAR_PROB = 0.3  # x出现的概率

# 位数定义
OP_BITS_RAW = 2
VAL_BITS_RAW = 7
UNIFIED_DATA_BITS = max(OP_BITS_RAW, VAL_BITS_RAW)
X_BITS = 8
OUTPUT_BITS = 11

# 类型定义
TYPE_OP = '1'
TYPE_VAL = '0'
TYPE_BITS = 1

# 最终token长度
TOKEN_LEN = TYPE_BITS + UNIFIED_DATA_BITS

# 文件名配置
TRAIN_FILE = f'expr_addsub_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'expr_addsub_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
def encode_token(is_op, val):
    """编码单个token，强制使用统一的token长度"""
    if is_op:
        raw_bits = OPS[val]
        padded_bits = raw_bits.ljust(UNIFIED_DATA_BITS, '0')
        return TYPE_OP + padded_bits
    else:
        raw_bits = bin(val)[2:]
        padded_bits = raw_bits.zfill(UNIFIED_DATA_BITS)
        return TYPE_VAL + padded_bits

def gen_operand():
    if random.random() < X_APPEAR_PROB:
        return 'x'
    else:
        return str(random.randint(*VAL_RANGE))

def gen_op():
    return random.choice(list(OPS.keys()))

def gen_expr_tree(op_count=2, val_count=3):
    if op_count==0 and val_count==1:
        return gen_operand()
    if op_count > 0 and val_count > 1:
        op = gen_op()
        left_vals = random.randint(1, val_count - 1)
        right_vals = val_count - left_vals
        left_ops = random.randint(0, op_count - 1) if left_vals > 1 else 0
        right_ops = op_count - 1 - left_ops
        if (left_vals==1 and left_ops > 0) or \
           (right_vals==1 and right_ops > 0) or \
           (left_vals > 1 and left_ops >= left_vals) or \
           (right_vals > 1 and right_ops >= right_vals):
            return gen_expr_tree(op_count, val_count)
        return (op, gen_expr_tree(left_ops, left_vals), gen_expr_tree(right_ops, right_vals))
    raise RuntimeError("Invalid tree state")

def flatten(expr):
    if isinstance(expr, str):
        return [expr]
    else:
        op, left, right = expr
        return [op] + flatten(left) + flatten(right)

def evaluate(expr, x_val):
    if isinstance(expr, str):
        return x_val if expr=='x' else int(expr)
    else:
        op, a, b = expr
        a_val = evaluate(a, x_val)
        b_val = evaluate(b, x_val)
        if op=='+': return a_val + b_val
        if op=='-': return a_val - b_val
        raise ValueError(f"未知操作: {op}")

def to_twos_complement(value, bits):
    if value >= 0:
        return bin(value)[2:].zfill(bits)
    else:
        return bin((1 << bits) + value)[2:]

def encode_expr(prefix_tokens):
    bits = ''
    for tok in prefix_tokens:
        if tok in OPS:
            bits += encode_token(True, tok)
        else:
            val = 0 if tok=='x' else int(tok)
            bits += encode_token(False, val)
    return bits

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, output_bits):
    """生成训练集和验证集"""
    print(f"\n--- 开始生成数据集 ({num_samples} 条样本) ---")
    print(f"每个Token长度: {TOKEN_LEN} bits, X值编码: {X_BITS} bits, 输出编码: {output_bits} bits")
    
    # 使用局部set去重，避免全局变量污染
    input_set = set()
    records = []
    count = 0
    attempts = 0
    
    max_attempts = num_samples * 100 if num_samples > 100 else num_samples * 1000
    
    while count < num_samples:
        if attempts > max_attempts:
            print(f"\n尝试次数过多 ({attempts})，已生成 {count} 条，提前退出。")
            break
        
        try:
            expr = gen_expr_tree()
            tokens = flatten(expr)
            
            if 'x' not in tokens or len(tokens)!=5:
                attempts += 1
                continue
            
            bits = encode_expr(tokens)
            x_val = random.randint(*X_VAL_RANGE)
            val = evaluate(expr, x_val)
            
            x_bin = bin(x_val)[2:].zfill(X_BITS)
            out_bin = to_twos_complement(val, output_bits)
            input_str = bits + x_bin
            
            if input_str in input_set:
                attempts += 1
                continue
            
            input_set.add(input_str)
            
            # 输出改为多标签二分类格式
            records.append({
                "input": input_str,
                "output": [int(bit) for bit in out_bin]  # 改为列表格式
            })
            
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(f"\r已生成: {count}/{num_samples}")
                sys.stdout.flush()
                
        except (ValueError, RuntimeError):
            attempts += 1
            continue
        except Exception as e:
            print(f"\n发生未知错误: {e}")
            attempts += 1
            continue
    
    print(f"\n生成完毕。共 {len(records)} 条不重复数据。")
    
    # 打乱并分割为训练集和验证集
    random.shuffle(records)
    train_size = int(len(records) * 0.9)
    train_data = records[:train_size]
    eval_data = records[train_size:]
    
    # 写入文件
    print(f"\n正在写入 {len(train_data)} 条训练数据到 '{TRAIN_FILE}'...")
    with open(TRAIN_FILE, 'w') as f:
        for record in train_data:
            f.write(json.dumps(record) + '\n')
    
    print(f"正在写入 {len(eval_data)} 条评估数据到 '{EVAL_FILE}'...")
    with open(EVAL_FILE, 'w') as f:
        for record in eval_data:
            f.write(json.dumps(record) + '\n')
    
    print("\n所有数据集生成完成！")

# ==============================================================================
# --- 4. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets(DATASET_SIZE, OUTPUT_BITS)
