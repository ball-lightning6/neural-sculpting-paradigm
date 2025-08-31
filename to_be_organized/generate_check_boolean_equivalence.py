import itertools
import random
import json

# 可用变量
vars = ['a', 'b', 'c', 'd']

# 可用运算符构建表达式模板
def generate_random_expr(variables):
    """生成一个合法的布尔表达式字符串"""
    expr = random.choice(variables)
    for _ in range(random.randint(1, 3)):
        op = random.choice(['&', '|'])
        term = random.choice(variables + [f'~{v}' for v in variables])
        expr = f'({expr}{op}{term})'
    return expr

def is_equivalent(expr1, expr2, variables):
    """枚举所有布尔组合判断等价"""
    for vals in itertools.product([False, True], repeat=len(variables)):
        env = dict(zip(variables, vals))
        try:
            v1 = eval(expr1, {}, env)
            v2 = eval(expr2, {}, env)
        except Exception:
            return False  # 不合法表达式
        if v1 != v2:
            return False
    return True

# 数据集生成
def generate_dataset(n=1000, seed=42, filename="bool_eq_dataset.jsonl"):
    random.seed(seed)
    with open(filename, 'w') as f:
        for _ in range(n):
            while True:
                expr1 = generate_random_expr(vars)
                expr2 = generate_random_expr(vars)
                try:
                    eq = is_equivalent(expr1, expr2, vars)
                    break
                except:
                    continue  # 有非法表达式则重试
            label = "1" if eq else "0"
            item = {
                "input": f"expr1={expr1};expr2={expr2}",
                "label": label
            }
            f.write(json.dumps(item) + '\n')
    print(f"✅ Generated {n} samples to {filename}")

# 调用主函数
if __name__ == "__main__":
    generate_dataset(n=10000)
