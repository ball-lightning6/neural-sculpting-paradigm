import random
import json

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 10000  # 数据集大小
VAL_RANGE = 15        # 数字常量的范围 (1-15)
X_VAL_RANGE = 15      # 变量x的值范围 (1-15)
EXPR_MAX_OPS = 2      # 表达式最大操作符数
EXPR_MAX_VALS = 3     # 表达式最大操作数值

# 计算输出位宽 (基于最大可能值)
# 最坏情况: (VAL_RANGE + X_VAL_RANGE) * (EXPR_MAX_OPS + 1) < 256
# 所以使用8位足够
OUTPUT_BITS = 8

# 文件名配置
TRAIN_FILE = f'expr_eval_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'expr_eval_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
OPS = {
    '+': '0000',
    '-': '0001',
    '*': '0010'
}

VAL_BITS = 4
TYPE_OP = '1'
TYPE_VAL = '0'

def encode_token(is_op, val_bits):
    return TYPE_OP + val_bits if is_op else TYPE_VAL + val_bits

def gen_operand():
    if random.random() < 0.3:
        return 'x'
    else:
        return str(random.randint(1, VAL_RANGE))

def gen_op():
    return random.choice(['+', '-', '*'])

def gen_expr_tree(op_count=EXPR_MAX_OPS, val_count=EXPR_MAX_VALS):
    """递归生成合法的表达式树"""
    if op_count == 0:
        return gen_operand()
    else:
        op = gen_op()
        left_ops = random.randint(0, op_count - 1)
        right_ops = op_count - 1 - left_ops
        left_vals = random.randint(1, val_count - 1)
        right_vals = val_count - left_vals
        return (op,
                gen_expr_tree(left_ops, left_vals),
                gen_expr_tree(right_ops, right_vals))

def flatten(expr):
    """前缀展开表达式树"""
    if isinstance(expr, str):
        return [expr]
    else:
        op, left, right = expr
        return [op] + flatten(left) + flatten(right)

def evaluate(expr, x_val):
    """求值表达式，x替换为具体数值"""
    if isinstance(expr, str):
        return x_val if expr == 'x' else int(expr)
    else:
        op, a, b = expr
        a_val = evaluate(a, x_val)
        b_val = evaluate(b, x_val)
        # 使用参数化边界检查
        max_val = 2**OUTPUT_BITS - 1
        if not (0 <= a_val <= max_val and 0 <= b_val <= max_val):
            raise ValueError("中间越界")
        if op == '+':
            res = a_val + b_val
        elif op == '-':
            res = a_val - b_val
        elif op == '*':
            res = a_val * b_val
        else:
            raise ValueError("未知操作")
        if not (0 <= res <= max_val):
            raise ValueError("结果越界")
        return res

def encode_expr(prefix_tokens):
    """编码表达式为二进制串"""
    bits = ''
    for tok in prefix_tokens:
        if tok in OPS:
            bits += encode_token(True, OPS[tok])
        else:
            val = 0 if tok == 'x' else int(tok)
            bits += encode_token(False, format(val, '04b'))
    return bits

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, output_bits):
    """生成训练集和验证集"""
    print(f"\n--- 开始生成数据集 ({num_samples} 条样本) ---")
    
    # 使用局部set去重，确保我们生成的样本是唯一的
    input_set = set()
    records = []
    count = 0
    
    while count < num_samples:
        try:
            expr = gen_expr_tree()
            tokens = flatten(expr)

            if 'x' not in tokens:
                continue

            bits = encode_expr(tokens)
            x_val = random.randint(1, X_VAL_RANGE)
            val = evaluate(expr, x_val)

            x_bin = format(x_val, '04b')
            out_bin = format(val, f'0{output_bits}b')

            input_key = bits + x_bin
            if input_key in input_set:
                continue
            input_set.add(input_key)
            
            # 输出改为多标签二分类格式
            records.append({
                "input": bits + x_bin,
                "output": [int(bit) for bit in out_bin]  # 改为列表格式
            })
            
            count += 1
            if count % 1000 == 0:
                print(f"已生成 {count} / {num_samples} 条样本...")

        except Exception:
            continue

    print(f"生成完毕。共 {len(records)} 条不重复数据。")
    
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
