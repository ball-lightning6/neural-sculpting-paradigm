import json
import random
from tqdm import tqdm
import collections

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
DATASET_SIZE = 50000  # 数据集大小（减小规模，避免生成困难）
GRID_SIZE = 5  # 网格尺寸（n×n）

# 输入输出长度计算
INPUT_BITS = GRID_SIZE * GRID_SIZE  # n*n 的网格
OUTPUT_GRID_BITS = GRID_SIZE * GRID_SIZE  # 热力图
OUTPUT_COORD_BITS = (GRID_SIZE - 1).bit_length() * 2 if GRID_SIZE > 1 else 2  # x+y 坐标

TRAIN_FILE = f'largest_island_{GRID_SIZE}x{GRID_SIZE}_{DATASET_SIZE}_train.jsonl'
EVAL_FILE = f'largest_island_{GRID_SIZE}x{GRID_SIZE}_{DATASET_SIZE}_eval.jsonl'

# ==============================================================================
# --- 2. 核心逻辑：数据生成与编码 ---
# ==============================================================================
class LargestIslandSolver:
    """
    一个高效的“最大人工岛”问题求解器。
    使用两遍扫描法（迭代式DFS+评估）来找到最佳翻转位置。
    """
    
    def __init__(self, grid):
        self.grid = grid
        self.n = len(grid)
        self.island_labels = [[0] * self.n for _ in range(self.n)]
        self.island_areas = {0: 0}
    
    def _dfs_iterative(self, r_start, c_start, island_id):
        stack = [(r_start, c_start)]
        self.island_labels[r_start][c_start] = island_id
        area = 0
        while stack:
            r, c = stack.pop()
            area += 1
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.n and 0 <= nc < self.n and
                        self.grid[nr][nc] == 1 and
                        self.island_labels[nr][nc] == 0):
                    self.island_labels[nr][nc] = island_id
                    stack.append((nr, nc))
        return area
    
    def solve_for_optimal_position(self):
        island_id = 2
        has_zero = False
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c] == 0:
                    has_zero = True
                elif self.grid[r][c] == 1 and self.island_labels[r][c] == 0:
                    area = self._dfs_iterative(r, c, island_id)
                    self.island_areas[island_id] = area
                    island_id += 1
        
        if not has_zero:
            return None, None
        
        max_area, best_pos = 0, None
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c] == 0:
                    neighbor_ids = set()
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.n and 0 <= nc < self.n:
                            neighbor_ids.add(self.island_labels[nr][nc])
                    
                    current_area = 1 + sum(self.island_areas.get(i_id, 0) for i_id in neighbor_ids)
                    
                    if current_area > max_area:
                        max_area = current_area
                        best_pos = (r, c)
        
        if best_pos is None and has_zero:
            return (0, 0), max_area
        return best_pos, max_area

def generate_puzzle_and_solve(n):
    """生成谜题并求解"""
    # 使用权重确保有足够的水域和陆地
    grid = [[random.choices([0, 1], weights=[0.4, 0.6])[0] for _ in range(n)] for _ in range(n)]
    solver = LargestIslandSolver(grid)
    solution_pos, max_area = solver.solve_for_optimal_position()
    return grid, solution_pos, max_area

def generate_island_samples(num_samples, n):
    """生成岛屿样本"""
    samples = []
    seen_inputs = set()
    
    # 计算坐标编码所需的位数
    bit_count = (n - 1).bit_length() if n > 1 else 1
    print(f"坐标编码将使用 {bit_count} 位 (共 {bit_count * 2} 位)。")
    
    while len(samples) < num_samples:
        grid, solution_pos, max_area = generate_puzzle_and_solve(n)
        
        # 输入：扁平化的网格
        input_str = "".join(map(str, [cell for row in grid for cell in row]))
        
        if input_str in seen_inputs:
            continue
        seen_inputs.add(input_str)
        
        # 输出1：热力图（最佳位置为1，其余为0）
        output_grid_2d = [[0] * n for _ in range(n)]
        if solution_pos is not None:
            r, c = solution_pos
            output_grid_2d[r][c] = 1
        output_grid_str = "".join(map(str, [cell for row in output_grid_2d for cell in row]))
        output_grid_multilabel = [int(bit) for bit in output_grid_str]
        
        # 输出2：最大面积的二进制表示
        max_area_binary = format(max_area, f'0{n * n}b')
        output_area_multilabel = [int(bit) for bit in max_area_binary]
        
        # 输出3：最佳位置的坐标编码
        output_coords_multilabel = []
        if solution_pos is not None:
            r, c = solution_pos
            x_bin = format(c, f'0{bit_count}b')
            y_bin = format(r, f'0{bit_count}b')
            output_coords_multilabel = [int(bit) for bit in (x_bin + y_bin)]
        else:
            # 如果没有解决方案，编码为全0
            output_coords_multilabel = [0] * (bit_count * 2)
        
        samples.append({
            "input": input_str,
            "output_grid": output_grid_multilabel,
            "output_area": output_area_multilabel,
            "output_coords": output_coords_multilabel
        })
    
    return samples

# ==============================================================================
# --- 3. 数据集生成函数 ---
# ==============================================================================
def generate_datasets():
    """生成最大岛屿数据集并分割训练/验证集"""
    print(f"\n--- 开始生成数据集 ({DATASET_SIZE} 条样本) ---")
    print(f"网格尺寸: {GRID_SIZE}x{GRID_SIZE}")
    print(f"输入长度: {INPUT_BITS} bits")
    print(f"输出1长度: {OUTPUT_GRID_BITS} bits (热力图)")
    print(f"输出2长度: {GRID_SIZE * GRID_SIZE} bits (最大面积)")
    print(f"输出3长度: {OUTPUT_COORD_BITS} bits (坐标编码)")
    
    samples = generate_island_samples(DATASET_SIZE, GRID_SIZE)
    print(f"生成完毕。共 {len(samples)} 条不重复数据。")
    
    # 打乱并分割为训练集和验证集
    random.shuffle(samples)
    train_size = int(len(samples) * 0.9)
    train_data = samples[:train_size]
    eval_data = samples[train_size:]
    
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
    generate_datasets()
