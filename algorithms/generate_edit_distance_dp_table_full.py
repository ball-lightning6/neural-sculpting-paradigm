import json
import random
from tqdm import tqdm

# 核心参数配置
LEN = 20
NUM_SAMPLES = 500000
OUTPUT_FILE = f"edit_distance_dp_table_full_len{LEN}.jsonl"

# 操作编码表 (3-bit one-hot)
OP_NONE = [0, 0, 0]  # Match
OP_INS = [1, 0, 0]   # Insert
OP_DEL = [0, 1, 0]   # Delete
OP_SUB = [0, 0, 1]   # Substitute


def solve_dp_with_ops(s1: str, s2: str):
    """
    计算编辑距离DP表并记录每个单元格的最优操作
    Returns: 完整的操作表 ops_table
    """
    m, n = len(s1), len(s2)
    ops_table = [[OP_NONE] * (n + 1) for _ in range(m + 1)]
    dist_table = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界
    for i in range(1, m + 1):
        dist_table[i][0] = i
        ops_table[i][0] = OP_DEL
    for j in range(1, n + 1):
        dist_table[0][j] = j
        ops_table[0][j] = OP_INS
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 1 if s1[i - 1] != s2[j - 1] else 0
            
            # 按优先级排序确保唯一解
            candidates = [
                (dist_table[i][j - 1] + 1, OP_INS),
                (dist_table[i - 1][j] + 1, OP_DEL),
                (dist_table[i - 1][j - 1] + cost, OP_SUB if cost else OP_NONE)
            ]
            
            candidates.sort(key=lambda x: (x[0], tuple(x[1])))
            
            best_dist, best_op = candidates[0]
            dist_table[i][j] = best_dist
            ops_table[i][j] = best_op
    
    return ops_table


def sample_one():
    """生成一条包含完整DP表的数据样本"""
    s1 = ''.join(random.choices('01', k=LEN))
    s2 = ''.join(random.choices('01', k=LEN))
    
    # 运行求解器获取完整DP操作表
    ops_table = solve_dp_with_ops(s1, s2)
    
    # 准备输入
    input_str = s1 + s2
    
    # 完整DP表作为解耦标签
    explanation_label = [bit for row in ops_table for op_code in row for bit in op_code]
    
    # 第一行作为预测标签
    prediction_label = [bit for op_code in ops_table[1] for bit in op_code]
    
    return {
        "input": input_str,
        "prediction_label": prediction_label,
        "explanation_label": explanation_label
    }


def main():
    print("=" * 70)
    print(f"编辑距离 - 完整DP表解耦实验")
    print("=" * 70)
    print(f"字符串长度: {LEN}")
    print(f"预测标签 (第一行): {(LEN + 1) * 3} bits")
    print(f"解耦标签 (完整表): {(LEN + 1) * (LEN + 1) * 3} bits")
    print(f"数据集大小: {NUM_SAMPLES}")
    print("=" * 70)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(NUM_SAMPLES), desc="生成样本"):
            f.write(json.dumps(sample_one(), ensure_ascii=False) + '\n')
    
    print(f"\n✅ 完成！已生成 {NUM_SAMPLES} 条样本 -> {OUTPUT_FILE}")
    
    print("\n--- 样本示例 ---")
    sample = sample_one()
    print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
