# generate_edit_nextstep_mlp.py

import json
import random
from tqdm import tqdm

random.seed(42)
OUTPUT_FILE = "edit_nextstep_mlp_train.jsonl"
NUM_SAMPLES = 500_000
LEN = 15              # 固定长度 15+15=30 位输入
DP_BIT_PER_CELL = 3   # 每格操作类型占 3bit
DP_CELLS = 31 * 31    # DP 表总格数

# 操作编码表
OP_NONE, OP_INS, OP_DEL, OP_SUB = 0, 1, 2, 3
OP_NAME = {OP_NONE: 'N', OP_INS: 'I', OP_DEL: 'D', OP_SUB: 'S'}

# 把操作编码成 3bit 字符串
def op2bin(op):
    return f"{op:03b}"

# 将字符串切成定长 LEN，不足补 0，超出截断
def fix(s):
    s = s[:LEN]
    return s + '0' * (LEN - len(s))

# 标准 DP 求最小编辑距离，并记录"前驱操作类型"
def dp_with_ops(a: str, b: str):
    m, n = len(a), len(b)
    # dist[i][j] = 编辑距离
    dist = [[0] * (n + 1) for _ in range(m + 1)]
    # ops[i][j] = 到达 (i,j) 所用的操作类型（OP_XXX）
    ops = [[OP_NONE] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dist[i][0] = i
        ops[i][0] = OP_DEL if i > 0 else OP_NONE
    for j in range(n + 1):
        dist[0][j] = j
        ops[0][j] = OP_INS if j > 0 else OP_NONE
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            candidates = [
                (dist[i][j - 1] + 1, OP_INS),   # 插入
                (dist[i - 1][j] + 1, OP_DEL),   # 删除
                (dist[i - 1][j - 1] + cost, OP_SUB if cost else OP_NONE)
            ]
            # 按距离+操作序保留唯一最优
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_dist, best_op = candidates[0]
            dist[i][j] = best_dist
            ops[i][j] = best_op
    return dist, ops

# 回溯得到"最优下一步操作"集合（允许多条）
def backtrace_next_ops(a: str, b: str, dist, ops):
    m, n = len(a), len(b)
    i, j = m, n
    next_ops = set()
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == OP_INS:
            # 对应 (i, j-1) -> (i,j) 插入 b[j-1]
            next_ops.add(('I', j - 1))  # 在 A 的 i 位置前插入 b[j-1]
            j -= 1
        elif op == OP_DEL:
            next_ops.add(('D', i - 1))  # 删除 A[i-1]
            i -= 1
        elif op == OP_SUB:
            next_ops.add(('S', i - 1))  # 替换 A[i-1] 为 b[j-1]
            i -= 1
            j -= 1
        else:  # NONE
            i -= 1
            j -= 1
        if len(next_ops) >= 3:  # 防止回溯太长，取前 3 个即可
            break
        if i == 0 and j == 0:
            break
    return list(next_ops)

# 90 标签：3 操作 × 30 位置
def ops2binmask(ops_list):
    mask = [0] * 90
    for op, pos in ops_list:
        idx = {'I': 0, 'D': 30, 'S': 60}[op] + pos
        mask[idx] = 1
    return ''.join(str(x) for x in mask)

# 把 ops 表展平成 31×31×3 bit 串
def ops_table2bin(ops):
    bits = []
    for row in ops:
        for cell in row:
            bits.append(op2bin(cell))
    return ''.join(bits)

# 生成一条样本
def sample_one():
    a = ''.join(random.choices('01', k=LEN))
    b = ''.join(random.choices('01', k=LEN))
    a, b = fix(a), fix(b)
    dist, ops = dp_with_ops(a, b)
    next_ops = backtrace_next_ops(a, b, dist, ops)
    # 若回溯为空（极少），强制给一个无操作
    if not next_ops:
        next_ops = [('S', 0)]
    op_mask = ops2binmask(next_ops)
    dp_bits = ops_table2bin(ops)
    return {
        'input': a + b,                          # 30 bit
        'op_mask': list(map(int, op_mask)),      # 90 bit
        'dp_ops': list(map(int, dp_bits)),       # 2883 bit
    }

# 主流程
def main():
    print("=" * 70)
    print("编辑距离下一步操作预测 - 数据集生成器")
    print("=" * 70)
    print(f"字符串长度: {LEN}")
    print(f"输入: {LEN * 2} bits")
    print(f"操作掩码输出: 90 bits (3操作 × 30位置)")
    print(f"DP表输出: {DP_CELLS * DP_BIT_PER_CELL} bits")
    print(f"数据集大小: {NUM_SAMPLES}")
    print("=" * 70)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(NUM_SAMPLES), desc="生成样本"):
            rec = sample_one()
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"\n✅ 完成！已生成 {NUM_SAMPLES} 条样本 -> {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
