import json
import random
from tqdm import tqdm
from typing import List

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 5000_000
    ROWS = 8
    COLS = 8
    OUTPUT_FILE = f"bricks_falling_{ROWS}x{COLS}_stable.jsonl"

# ==============================================================================
# --- 2. 核心算法: 并查集与物理模拟器 ---
# ==============================================================================

class UnionFind:
    """一个优化的并查集实现，用于处理连通性问题。"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        # 路径压缩
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # 按大小合并（秩合并优化）
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]

def get_stable_grid(grid: List[List[int]]) -> List[List[int]]:
    """
    计算给定网格的最终稳定状态（即移除所有悬空的砖块）。
    这是整个脚本逻辑修正的核心。
    """
    rows, cols = len(grid), len(grid[0])
    
    # 尺寸+1，最后一个节点代表“天花板”
    uf_size = rows * cols + 1
    ceiling_node = uf_size - 1
    uf = UnionFind(uf_size)

    # 将所有砖块与其邻居合并
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                node = r * cols + c
                # 与天花板连接
                if r == 0:
                    uf.union(node, ceiling_node)
                # 只需与上方和左方的邻居合并，即可覆盖所有连接
                if r > 0 and grid[r - 1][c] == 1:
                    uf.union(node, (r - 1) * cols + c)
                if c > 0 and grid[r][c - 1] == 1:
                    uf.union(node, r * cols + (c - 1))

    # 构建最终的稳定局面
    stable_grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                node = r * cols + c
                # 如果这个砖块最终与天花板连通，它就保留
                if uf.find(node) == uf.find(ceiling_node):
                    stable_grid[r][c] = 1
    
    return stable_grid

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================

def generate_sample(cfg):
    """
    生成一个 (稳定的初始局面+打击点 -> 最终局面) 的数据对。
    """
    while True:
        # 1. 随机生成一个“原始”的、可能不稳定的矩阵
        raw_matrix = [[random.choice([0, 1]) for _ in range(cfg.COLS)] for _ in range(cfg.ROWS)]
        
        # 2. 【关键】: 计算这个原始矩阵的“稳定形态”作为我们真正的“初始局面”
        stable_grid = get_stable_grid(raw_matrix)
        
        # 3. 在这个稳定的局面上，找到所有存在的砖块，作为可能的打击目标
        brick_locations = []
        for r in range(cfg.ROWS):
            for c in range(cfg.COLS):
                if stable_grid[r][c] == 1:
                    brick_locations.append([r, c])
        
        # 如果这个稳定局面里一块砖都没有，就重新生成一个
        if not brick_locations:
            continue
            
        # 4. 从存在的砖块中，随机选择一个进行“打击”
        hit_coord = random.choice(brick_locations)
        
        # 5. “打击”操作：创建一个被打击后的局面
        hit_grid = [row[:] for row in stable_grid]
        hit_grid[hit_coord[0]][hit_coord[1]] = 0
        
        # 6. 计算打击后的“最终稳定局面”，作为标签
        final_grid = get_stable_grid(hit_grid)
        
        # 7. 编码输入和输出
        input_grid_str = "".join(map(str, [bit for row in stable_grid for bit in row]))
        
        rows_bits = (cfg.ROWS - 1).bit_length()
        cols_bits = (cfg.COLS - 1).bit_length()
        
        hit_r_bin = format(hit_coord[0], f'0{rows_bits}b')
        hit_c_bin = format(hit_coord[1], f'0{cols_bits}b')
        
        input_str = input_grid_str + hit_r_bin + hit_c_bin
        
        output_str = "".join(map(str, [bit for row in final_grid for bit in row]))
        output_list = [int(bit) for bit in output_str]

        # 返回所有需要的信息，用于生成和可视化
        return {
            "input": input_str,
            "output": output_list
        }, stable_grid, hit_coord, final_grid

# ==============================================================================
# --- 4. 可视化函数 ---
# ==============================================================================

def visualize_sample(rows, cols, stable_grid, hit_coord, final_grid):
    """更清晰地可视化整个过程"""
    print("-" * (cols * 2 + 3))
    print("初始稳定局面 (敲击点用 'X' 标记):")
    for r in range(rows):
        line = ""
        for c in range(cols):
            char = str(stable_grid[r][c])
            # 使用 ANSI 颜色代码
            if r == hit_coord[0] and c == hit_coord[1]:
                line += f"\033[91mX\033[0m "  # 红色 X
            elif char == '1':
                line += f"\033[92m1\033[0m "  # 绿色 1
            else:
                line += "0 "
        print(f"| {line}|")
    print("-" * (cols * 2 + 3))
        
    print("\n最终稳定局面:")
    for r in range(rows):
        line = ""
        for c in range(cols):
            char = str(final_grid[r][c])
            if char == '1':
                line += f"\033[92m1\033[0m " # 绿色 1
            else:
                line += "0 "
        print(f"| {line}|")
    print("-" * (cols * 2 + 3))

# ==============================================================================
# --- 5. 主生成函数 ---
# ==============================================================================

def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"LeetCode 803. 打砖块 (稳定局面预测版) - 数据集生成器")
    print("=" * 70)
    
    input_dim = cfg.ROWS * cfg.COLS + (cfg.ROWS-1).bit_length() + (cfg.COLS-1).bit_length()
    output_dim = cfg.ROWS * cfg.COLS
    
    print(f"固定矩阵尺寸: {cfg.ROWS}x{cfg.COLS}")
    print(f"输入维度: {input_dim}")
    print(f"输出维度: {output_dim}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample, _, _, _ = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")
    
    print("\n" + "="*30 + " 样本可视化检查 " + "="*30)
    # 生成一个用于可视化的新样本
    _, stable_grid, hit_coord, final_grid = generate_sample(cfg)
    visualize_sample(cfg.ROWS, cfg.COLS, stable_grid, hit_coord, final_grid)
    print("="* (74))

if __name__ == "__main__":
    main()
