# generate_lights_out_gaussian_elimination.py
import json
import random
from tqdm import tqdm
import numpy as np

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 1000_000 # DP解耦任务信息量很大，可能不需要超大数据集
    ROWS = 5
    COLS = 5

    GRID_SIZE = ROWS * COLS
    INPUT_DIM = GRID_SIZE

    # 输出维度 = 最终解(25) + 中间步骤(25x26)
    OUTPUT_DIM_SOLUTION = GRID_SIZE
    OUTPUT_DIM_ECHELON = GRID_SIZE * (GRID_SIZE + 1)
    OUTPUT_DIM_TOTAL = OUTPUT_DIM_SOLUTION + OUTPUT_DIM_ECHELON

    OUTPUT_FILE = f"lights_out_{ROWS}x{COLS}_inverse_decoupled.jsonl"

# ==============================================================================
# --- 2. 核心算法: "带记录"的模2高斯消元求解器 ---
# ==============================================================================

def get_lights_out_matrix(rows, cols):
    """构建 Lights Out 问题的变换矩阵 A (可逆规则版)"""
    n = rows * cols
    A = np.zeros((n, n), dtype=int)
    directions = [(0, 0), (-1, 1), (1, 1), (-1, 0), (1,-1)]

    for r in range(rows):
        for c in range(cols):
            press_idx = r * cols + c
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    affected_idx = nr * cols + nc
                    A[affected_idx, press_idx] = 1
    return A

def solve_linear_system_mod2(A, b):
    """
    在模2域上，使用高斯消元法求解线性方程组 Ax = b。
    返回 (解x, 中间的行简化阶梯矩阵)。
    """
    n = A.shape[0]
    # 1. 构建增广矩阵 [A|b]
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])

    pivot_row = 0
    for col in range(n):
        if pivot_row < n:
            # 找到一个主元为1的行
            i = pivot_row
            while i < n and augmented_matrix[i, col] == 0:
                i += 1

            if i < n: # 找到了主元
                # 将主元行换到 pivot_row
                augmented_matrix[[pivot_row, i]] = augmented_matrix[[i, pivot_row]]

                # 将主元下方的所有行消为0
                for j in range(pivot_row + 1, n):
                    if augmented_matrix[j, col] == 1:
                        # 行j = 行j XOR 行pivot_row
                        augmented_matrix[j, :] = (augmented_matrix[j, :] + augmented_matrix[pivot_row, :]) % 2

                pivot_row += 1

    # 到这里，我们得到了一个"行阶梯形矩阵"，这就是我们的解耦标签！
    # 为了简化，我们直接用这个，而不是更复杂的"行最简形"
    echelon_form = augmented_matrix.copy()

    # 2. 回代求解 x
    x = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        # 计算 Ai,i+1*xi+1 + ... + Ai,n-1*xn-1
        row_sum = np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n]) % 2

        if augmented_matrix[i, i] == 1:
            # x[i] = (b[i] - row_sum) mod 2
            x[i] = (augmented_matrix[i, n] + row_sum) % 2
        else:
            # 自由变量或无解，对于可逆矩阵不会发生
            pass

    return x, echelon_form

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg, matrix_A):
    # 1. 随机生成一个"按压方案"，这是我们最终要预测的"真实解"
    true_solution_x = np.random.randint(0, 2, size=cfg.GRID_SIZE)

    # 2. 通过正向计算，得到对应的"灯光局面 b"
    output_lights_b = np.dot(matrix_A, true_solution_x) % 2

    # 3. 使用高斯消元法，计算中间步骤（解耦标签）和最终解（用于验证）
    #    在实际生成中，我们信任求解器，不需要每次都验证解
    _, echelon_matrix = solve_linear_system_mod2(matrix_A, output_lights_b)

    # 4. 编码输入输出
    input_list = output_lights_b.tolist()

    solution_list = true_solution_x.tolist()
    echelon_list = echelon_matrix.flatten().tolist()

    output_list = solution_list + echelon_list

    return {
        "input": "".join(map(str, input_list)),
        "output": output_list
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()

    print("=" * 70)
    print(f"Lights Out (逆问题-高斯消元解耦版) - 数据集生成器")
    print("=" * 70)
    print(f"输入维度 (灯光局面): {cfg.INPUT_DIM}")
    print(f"输出维度 (最终解 + 阶梯矩阵): {cfg.OUTPUT_DIM_TOTAL}")
    print("=" * 70)

    # 预先计算好固定的变换矩阵A
    matrix_A = get_lights_out_matrix(cfg.ROWS, cfg.COLS)

    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg, matrix_A)
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

if __name__ == "__main__":
    main()