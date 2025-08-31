import random
import json

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
        return str(random.randint(1, 15))

def gen_op():
    return random.choice(['+', '-', '*'])

# 递归生成合法的表达式树，确保3操作4值
def gen_expr_tree(op_count=2, val_count=3):
    if op_count == 0:
        val = gen_operand()
        return val
    else:
        op = gen_op()
        left_ops = random.randint(0, op_count - 1)
        right_ops = op_count - 1 - left_ops
        left_vals = random.randint(1, val_count - 1)
        right_vals = val_count - left_vals
        return (op,
                gen_expr_tree(left_ops, left_vals),
                gen_expr_tree(right_ops, right_vals))

# 前缀展开
def flatten(expr):
    if isinstance(expr, str):
        return [expr]
    else:
        op, left, right = expr
        return [op] + flatten(left) + flatten(right)

# 求值，x 替换为具体数值
def evaluate(expr, x_val):
    if isinstance(expr, str):
        return x_val if expr == 'x' else int(expr)
    else:
        op, a, b = expr
        a_val = evaluate(a, x_val)
        b_val = evaluate(b, x_val)
        if not (0 <= a_val <= 255 and 0 <= b_val <= 255):
            raise ValueError("中间越界")
        if op == '+':
            res = a_val + b_val
        elif op == '-':
            res = a_val - b_val
        elif op == '*':
            res = a_val * b_val
        else:
            raise ValueError("未知操作")
        if not (0 <= res <= 255):
            raise ValueError("结果越界")
        return res

# 编码为 5bit 串
def encode_expr(prefix_tokens):
    bits = ''
    for tok in prefix_tokens:
        if tok in OPS:
            bits += encode_token(True, OPS[tok])
        else:
            val = 0 if tok == 'x' else int(tok)
            bits += encode_token(False, format(val, '04b'))
    return bits

input_set=set()
# 生成训练集
def generate_dataset(n, path):
    with open(path, 'w') as f:
        count = 0
        while count < n:
            try:
                expr = gen_expr_tree()
                tokens = flatten(expr)

                if 'x' not in tokens:
                    continue

                bits = encode_expr(tokens)
                x_val = random.randint(1, 15)
                val = evaluate(expr, x_val)
                #print(tokens)
                #print(x_val,val)
                x_bin = format(x_val, '04b')
                out_bin = format(val, '08b')
                #print(bits+x_bin)
                if bits+x_bin in input_set:
                    continue
                input_set.add(bits+x_bin)
                f.write(json.dumps({"input": bits + x_bin, "output": out_bin}) + '\n')
                count += 1
                if count%1000==0:
                    print(count)
            except Exception:
                continue

generate_dataset(10000,'expr_eval_1m_2_3_eval_dup_1.jsonl')