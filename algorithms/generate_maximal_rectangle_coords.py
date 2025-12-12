# generate_maximal_rectangle_coords_fixed_size.py
import json
import random
from tqdm import tqdm
import numpy as np
from typing import List

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500000
    
    # --- 【核心修改】: 固定矩阵尺寸 ---
    ROWS = 8
    COLS = 8
    
    # --- 文件名 ---
    OUTPUT_FILE = f"maximal_rectangle_{ROWS}x{COLS}_coords_fixed.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: LeetCode 题解算法 (无变化) ---
# ==============================================================================
class LeetCodeSolution:
    """
    修改版题解，不仅返回最大面积，还返回所有最大矩形的坐标。
    """
    def largestRectangleArea_with_coords(self, heights: List[int], row_idx: int):
        heights_padded = heights + [0]
        st = [-1]
        max_area = 0
        rects = []
        
        for right, h in enumerate(heights_padded):
            while len(st) > 1 and heights_padded[st[-1]] >= h:
                i = st.pop()
                height = heights_padded[i]
                left = st[-1]
                width = right - left - 1
                area = height * width
                
                if area > max_area:
                    max_area = area
                    rects = []
                
                if area == max_area and area > 0:
                    r1 = row_idx - height + 1
                    c1 = left + 1
                    r2 = row_idx
                    c2 = right - 1
                    rects.append((r1, c1, r2, c2))
            st.append(right)
        return max_area, rects

    def maximalRectangle_with_coords(self, matrix: List[List[str]]):
        if not matrix or not matrix[0]:
            return 0, []
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        max_rects = []

        for r_idx, row in enumerate(matrix):
            for j, c in enumerate(row):
                heights[j] = heights[j] + 1 if c == '1' else 0

            area, rects_in_row = self.largestRectangleArea_with_coords(heights, r_idx)
            
            if area > max_area:
                max_area = area
                max_rects = rects_in_row
            elif area == max_area and area > 0:
                max_rects.extend(rects_in_row)
        
        return max_area, list(set(max_rects))

# ==============================================================================
# --- 3. 样本生成函数 (已修改) ---
# ==============================================================================
def generate_sample(cfg, solver):
    """
    生成一个 (输入矩阵, 最大矩形坐标) 的数据对，确保只有一个最大矩形。
    """
    while True: # 循环直到找到一个合格的样本
        # 使用固定的尺寸
        matrix = [[random.choice(['0', '1']) for _ in range(cfg.COLS)] for _ in range(cfg.ROWS)]
        
        max_area, max_rects = solver.maximalRectangle_with_coords(matrix)
        
        # 核心约束：只接受只有一个最大矩形的样本
        if len(max_rects) == 1:
            r1, c1, r2, c2 = max_rects[0]
            
            # --- 【核心修改】: 输入直接压平，不再有填充和尺寸信息 ---
            input_flat = [item for sublist in matrix for item in sublist]
            input_str = "".join(input_flat)
            
            # 将4个坐标编码为二进制
            # 使用固定尺寸的 bit_length()
            rows_bits = cfg.ROWS.bit_length()
            cols_bits = cfg.COLS.bit_length()
            
            r1_bin = format(r1, f'0{rows_bits}b')
            c1_bin = format(c1, f'0{cols_bits}b')
            r2_bin = format(r2, f'0{rows_bits}b')
            c2_bin = format(c2, f'0{cols_bits}b')
            
            output_str = r1_bin + c1_bin + r2_bin + c2_bin
            output_list = [int(bit) for bit in output_str]

            # 返回 matrix 用于可视化
            return {
                "input": input_str,
                "output": output_list
            }, matrix, max_rects[0]

# ==============================================================================
# --- 4. 可视化函数 (新增) ---
# ==============================================================================
def visualize_sample(matrix, rect_coords):
    """在终端用文本可视化一个样本及其最大矩形。"""
    rows = len(matrix)
    cols = len(matrix[0])
    r1, c1, r2, c2 = rect_coords
    
    print("-" * (cols * 2 + 3))
    print("输入矩阵:")
    for r in range(rows):
        print(f"| {' '.join(matrix[r])} |")
    print("-" * (cols * 2 + 3))
    
    print(f"最大矩形坐标: (r1,c1)=({r1},{c1}), (r2,c2)=({r2},{c2})")
    print("最大矩形可视化:")
    
    # 创建一个用于高亮的副本
    highlight_matrix = [list(row) for row in matrix]
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            # 用特殊字符高亮
            highlight_matrix[r][c] = '*'
            
    print("-" * (cols * 2 + 3))
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            val = highlight_matrix[r][c]
            # 根据内容添加 ANSI 颜色代码
            if val == '*':
                row_str += f"\033[91m{val}\033[0m " # 红色
            elif val == '1':
                 row_str += f"\033[92m{val}\033[0m " # 绿色
            else:
                row_str += f"{val} "
        print(f"| {row_str}|")
    print("-" * (cols * 2 + 3))


# ==============================================================================
# --- 5. 主生成函数 (已修改) ---
# ==============================================================================
def main():
    cfg = Config()
    solver = LeetCodeSolution()
    
    print("=" * 70)
    print(f"LeetCode 85. 最大矩形 (固定尺寸坐标输出) - 数据集生成器")
    print("=" * 70)
    # 计算维度
    input_dim = cfg.ROWS * cfg.COLS
    output_dim = 2 * (cfg.ROWS.bit_length() + cfg.COLS.bit_length())
    print(f"固定矩阵尺寸: {cfg.ROWS}x{cfg.COLS}")
    print(f"输入维度: {input_dim}")
    print(f"输出维度: {output_dim}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample, _, _ = generate_sample(cfg, solver)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

    # --- 【新增】生成结束后，可视化一个样本 ---
    print("\n" + "="*30 + " 样本可视化检查 " + "="*30)
    _, matrix, rect_coords = generate_sample(cfg, solver)
    visualize_sample(matrix, rect_coords)
    print("="* (74))


if __name__ == "__main__":
    main()
