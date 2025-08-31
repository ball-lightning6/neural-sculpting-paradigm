import json
import random
from tqdm import tqdm
import collections


# (LargestIslandSolver 类与上一版完全相同，此处为了简洁省略)
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
                        self.grid[nr][nc]==1 and
                        self.island_labels[nr][nc]==0):
                    self.island_labels[nr][nc] = island_id
                    stack.append((nr, nc))
        return area

    def solve_for_optimal_position(self):
        island_id = 2
        has_zero = False
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c]==0:
                    has_zero = True
                elif self.grid[r][c]==1 and self.island_labels[r][c]==0:
                    area = self._dfs_iterative(r, c, island_id)
                    self.island_areas[island_id] = area
                    island_id += 1

        if not has_zero: return None

        max_area, best_pos = 0, None
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c]==0:
                    neighbor_ids = set()
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.n and 0 <= nc < self.n:
                            neighbor_ids.add(self.island_labels[nr][nc])

                    current_area = 1 + sum(self.island_areas.get(i_id, 0) for i_id in neighbor_ids)

                    if current_area > max_area:
                        max_area = current_area
                        best_pos = (r, c)

        if best_pos is None and has_zero: return (0, 0)
        return best_pos


def generate_puzzle_and_solve(n):
    grid = [[random.choices([0, 1], weights=[0.4, 0.6])[0] for _ in range(n)] for _ in range(n)]
    solver = LargestIslandSolver(grid)
    solution_pos = solver.solve_for_optimal_position()
    return grid, solution_pos


def create_dataset(num_samples, n, output_path):
    """
    创建完整的数据集文件，每个样本包含三个字段：
    1. input: 扁平化的输入网格字符串。
    2. output_grid: 扁平化的热力图，最佳位置为1，其余为0。
    3. output_coords: 最佳位置的(x,y)坐标的二进制编码字符串。
    """
    print(f"--- 开始生成 '最大人工岛' 多模态数据集 ---")
    print(f"网格尺寸: {n}x{n}, 目标样本数量: {num_samples}")

    # <<< 新增：计算坐标编码所需的位数
    # (n-1).bit_length() 是一个高效且准确的方法
    bit_count = (n - 1).bit_length() if n > 1 else 1
    print(f"坐标编码将使用 {bit_count} 位 (共 {bit_count * 2} 位)。")

    all_samples = []
    seen_inputs = set()

    with tqdm(total=num_samples, desc="生成不重复样本") as pbar:
        while len(all_samples) < num_samples:
            grid, solution_pos = generate_puzzle_and_solve(n)

            input_str = "".join(map(str, [cell for row in grid for cell in row]))

            if input_str in seen_inputs: continue
            seen_inputs.add(input_str)

            # --- 生成第一个输出：热力图 ---
            output_grid_2d = [[0] * n for _ in range(n)]
            if solution_pos is not None:
                r, c = solution_pos
                output_grid_2d[r][c] = 1
            output_grid_str = "".join(map(str, [cell for row in output_grid_2d for cell in row]))

            # --- 新增：生成第二个输出：坐标编码 ---
            output_coords_str = ""
            if solution_pos is not None:
                r, c = solution_pos  # r 是 y 坐标, c 是 x 坐标
                # 格式化为固定长度的二进制字符串，前面补0
                x_bin = f'{c:0{bit_count}b}'
                y_bin = f'{r:0{bit_count}b}'
                output_coords_str = x_bin + y_bin

            # <<< 修改：将所有输出字段添加到样本中
            all_samples.append({
                "input": input_str,
                "output_grid": output_grid_str,
                "output_coords": output_coords_str
            })
            pbar.update(1)

    print("\n样本生成完毕，开始打乱顺序...")
    random.shuffle(all_samples)
    print("打乱完成！")

    print("开始写入文件...")
    with open(output_path, 'w') as f:
        for sample in tqdm(all_samples, desc="写入文件"):
            f.write(json.dumps(sample) + '\n')

    print(f"数据集创建成功！文件已保存至: {output_path}")


if __name__=='__main__':
    # --- 配置参数 ---
    NUM_SAMPLES = 500000
    GRID_SIZE = 5
    OUTPUT_FILE = 'largest_island_multimodal_5_5_dataset.jsonl'

    create_dataset(
        num_samples=NUM_SAMPLES,
        n=GRID_SIZE,
        output_path=OUTPUT_FILE
    )
