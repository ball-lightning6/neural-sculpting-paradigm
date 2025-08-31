import random
import json
import sys

# --- 1. 全局参数修改 ---
# 目标：统一所有token的编码长度

OPS = {
    '+': '001',  # 用3位就够了，但我们会补齐
    '-': '010'
}
OP_BITS_RAW = 3  # 操作符原始编码位数

VAL_RANGE = (1, 127)
VAL_BITS_RAW = 7  # 操作数原始编码位数

X_VAL_RANGE = (0, 255)
X_BITS = 8

OUTPUT_BITS = 11

TYPE_OP = '1'
TYPE_VAL = '0'
TYPE_BITS = 1

# --- 关键改动在这里 ---
# 决定所有token的统一数据位长度，取最长者
UNIFIED_DATA_BITS = max(OP_BITS_RAW, VAL_BITS_RAW)  # 结果是 7
# 最终每个token的长度 = 类型位 + 统一数据位
TOKEN_LEN = TYPE_BITS + UNIFIED_DATA_BITS  # 结果是 1 + 7 = 8


# --- 修改后的编码函数 ---
def encode_token(is_op, val):
    """
    编码单个token，强制使用统一的token长度。
    """
    if is_op:
        type_prefix = TYPE_OP
        # 获取原始编码
        raw_bits = OPS[val]
        # 补零到统一数据位长度
        padded_bits = raw_bits.ljust(UNIFIED_DATA_BITS, '0')  # 右侧补零
        return type_prefix + padded_bits
    else:  # 是操作数
        type_prefix = TYPE_VAL
        # 获取原始编码
        raw_bits = bin(val)[2:]
        # 补零到统一数据位长度
        padded_bits = raw_bits.zfill(UNIFIED_DATA_BITS)  # 左侧补零
        return type_prefix + padded_bits


# ... gen_op, gen_operand, gen_expr_tree, flatten, evaluate, to_twos_complement 保持不变 ...

# --- 只需要修改 encode_expr 来使用新的 encode_token 函数 ---
def encode_expr(prefix_tokens):
    bits = ''
    for tok in prefix_tokens:
        if tok in OPS:
            # 新的 encode_token 函数更简单
            bits += encode_token(True, tok)
        else:
            val = 0 if tok=='x' else int(tok)
            bits += encode_token(False, val)
    return bits


# ... 主生成循环 generate_dataset 保持不变，因为它调用的是高层函数 ...

# ==========================================================
# 为了完整性，我将整个脚本的最终版本放在下面
# ==========================================================

# 最终版脚本

import random
import json
import sys

# --- 参数配置 ---
OPS = {'+': '01', '-': '10'}
VAL_RANGE = (1, 3)
X_VAL_RANGE = (0, 7)
X_APPEAR_PROB = 0.3  # x出现的概率

# 位数定义
OP_BITS_RAW = 2
VAL_BITS_RAW = 2
UNIFIED_DATA_BITS = max(OP_BITS_RAW, VAL_BITS_RAW)  # 7
X_BITS = 3
OUTPUT_BITS = 6

# 类型定义
TYPE_OP = '1'
TYPE_VAL = '0'
TYPE_BITS = 1

# 最终token长度
TOKEN_LEN = TYPE_BITS + UNIFIED_DATA_BITS  # 1 + 7 = 8


def encode_token(is_op, val):
    if is_op:
        raw_bits = OPS[val]
        # 右侧补零以达到统一长度
        padded_bits = raw_bits.ljust(UNIFIED_DATA_BITS, '0')
        return TYPE_OP + padded_bits
    else:
        raw_bits = bin(val)[2:]
        # 左侧补零以达到统一长度
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
        # 确保左右子树至少有一个操作数
        left_vals = random.randint(1, val_count - 1)
        right_vals = val_count - left_vals
        # 分配操作符
        left_ops = random.randint(0, op_count - 1) if left_vals > 1 else 0
        right_ops = op_count - 1 - left_ops

        # 简单校验，避免无效递归
        if (left_vals==1 and left_ops > 0) or \
                (right_vals==1 and right_ops > 0) or \
                (left_vals > 1 and left_ops >= left_vals) or \
                (right_vals > 1 and right_ops >= right_vals):
            return gen_expr_tree(op_count, val_count)  # 无效分配，重试

        return (op,
                gen_expr_tree(left_ops, left_vals),
                gen_expr_tree(right_ops, right_vals))
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


input_set = set()


def generate_dataset(n, path):
    print(f"开始生成数据集，目标数量: {n}, 输出到: {path}")
    print(f"每个Token长度: {TOKEN_LEN} bits, X值编码: {X_BITS} bits, 输出编码: {OUTPUT_BITS} bits")

    with open(path, 'w') as f:
        count = 0
        attempts = 0
        while count < n:
            if attempts > n * 100 and n > 100:
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
                out_bin = to_twos_complement(val, OUTPUT_BITS)
                input_str = bits + x_bin

                if input_str in input_set:
                    attempts += 1
                    continue
                #print(tokens,x_val,val)
                input_set.add(input_str)
                f.write(json.dumps({"input": input_str, "output": out_bin}) + '\n')
                count += 1

                if count % 1000==0:
                    sys.stdout.write(f"\r已生成: {count}/{n}")
                    sys.stdout.flush()

            except (ValueError, RuntimeError):
                attempts += 1
                continue
            except Exception as e:
                print(f"\n发生未知错误: {e}")
                attempts += 1
                continue
    print(f"\n数据集 '{path}' 生成完成，共 {count} 条。")


# --- 使用示例 ---
generate_dataset(2048, 'expr_addsub_train_tiny.jsonl')
# 重新初始化set，确保eval集和train集可能重叠（更真实的评估）
#input_set = set()
#generate_dataset(100000, 'expr_addsub_eval.jsonl')