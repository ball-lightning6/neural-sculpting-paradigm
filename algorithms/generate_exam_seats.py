# generate_exam_seats.py
# LeetCode 1349. Maximum Students Taking Exam - 状态压缩 DP 数据集生成器
# https://leetcode.cn/problems/maximum-students-taking-exam/

import json
import random
from functools import reduce
from typing import List
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500000
    
    # 固定矩阵尺寸以保证生成效率和一致性
    # 复杂度是 O(m * 2^n * 2^n)，n=6时状态数为64，转移开销可控
    ROWS = 8
    COLS = 6
    
    # --- 文件名 ---
    OUTPUT_FILE = f"exam_seats_{ROWS}x{COLS}.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: LeetCode 题解算法 (用于生成标签) ---
# ==============================================================================
class LeetCodeSolution:
    """
    用于生成正确答案的参考实现，基于状态压缩动态规划。
    """
    def maxStudents(self, seats: List[List[str]]) -> int:
        m, n = len(seats), len(seats[0])
        # dp[i][mask] 表示第i行座位状态为mask时，前i行（包括第i行）能坐的最大人数
        # 初始多加一行作为padding，简化边界处理
        dp = [[0] * (1 << n) for _ in range(m + 1)]
        
        # 将坏座位 '#' 预处理成二进制掩码，broken_masks[i]的第j位为1表示第i行第j列是坏座位
        broken_masks = [0] * m
        for i in range(m):
            mask = 0
            for j in range(n):
                if seats[i][j] == '#':
                    mask |= (1 << j)
            broken_masks[i] = mask

        # 这里的row是实际的行索引，对应dp数组的索引需要注意
        # 我们定义 dp[i] 为处理完第 i-1 行后的状态
        # 为了逻辑清晰，我们从第0行开始递推到第m-1行
        
        # 初始化第0行的前置状态（可以认为是-1行），全0
        # dp[0][0] = 0，其他默认为0
        
        for row in range(1, m + 1):
            current_row_idx = row - 1
            broken = broken_masks[current_row_idx]
            
            for j in range(1 << n):
                # 检查当前行状态 j 的行内合法性：
                # 1. 不能坐在坏座位上: (j & broken) == 0
                # 2. 左右不能相邻: (j & (j << 1)) == 0
                if (j & broken) == 0 and (j & (j << 1)) == 0:
                    
                    count_j = bin(j).count('1')
                    
                    # 遍历上一行的所有合法状态 k
                    for k in range(1 << n):
                        # 检查当前行j与上一行k是否冲突：
                        # 考试作弊规则：左上和右上不能有人
                        # j的第bit位对应当前行，k对应上一行
                        # 左上角冲突: (j & (k >> 1))
                        # 右上角冲突: (j & (k << 1))
                        if (j & (k >> 1)) == 0 and (j & (k << 1)) == 0:
                            dp[row][j] = max(dp[row][j], dp[row - 1][k] + count_j)
                            
        return max(dp[m])

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg, solver):
    """生成一个 (输入矩阵, 最大学生数) 的数据对。"""
    
    # 随机生成座位图，'.'为好座位，'#'为坏座位
    # 稍微倾向于生成好座位，增加问题的复杂性
    matrix = [[random.choices(['.', '#'], weights=[0.7, 0.3])[0] for _ in range(cfg.COLS)] for _ in range(cfg.ROWS)]
    
    max_students = solver.maxStudents(matrix)
    
    # 将输入矩阵压平为0/1字符串 (' ' -> 1, '#' -> 0)
    # 注意题目输入是二维字符数组，这里展平提供给神经网络
    input_flat = ['1' if char == '.' else '0' for row in matrix for char in row]
    input_str = "".join(input_flat)
    
    # 将输出（最大学生数）编码为二进制
    output_bits_len = (cfg.ROWS * cfg.COLS).bit_length()
    output_list = [int(bit) for bit in format(max_students, f'0{output_bits_len}b')]

    return {
        "input": input_str,
        "output": output_list
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()
    solver = LeetCodeSolution()
    
    print("=" * 70)
    print(f"LeetCode 1349. 参加考试的最大学生数 - 数据集生成器")
    print("=" * 70)
    
    input_dim = cfg.ROWS * cfg.COLS
    output_dim = (cfg.ROWS * cfg.COLS).bit_length()
    
    print(f"固定矩阵尺寸: {cfg.ROWS}x{cfg.COLS}")
    print(f"输入维度: {input_dim}")
    print(f"输出维度 (最大人数的二进制): {output_dim}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg, solver)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")
    
    # 验证一个简单Case
    # . #
    # # .
    # 最优解：(0,0)和(1,1)各坐一人，共2人
    print("\n--- 简单逻辑验证 ---")
    test_matrix = [['.', '#'], ['#', '.']]
    ans = solver.maxStudents(test_matrix)
    print(f"Test Matrix: {test_matrix}")
    print(f"Max Students: {ans} (Expected 2)")
    assert ans == 2

if __name__ == "__main__":
    main()
