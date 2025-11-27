import json
import random
from tqdm import tqdm
import math


class LargestIslandSolver:
    """
    一个高效的“最大人工岛”问题求解器。
    这个版本会返回最佳翻转位置以及形成的最大面积。
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

    def solve(self):
        """
        [修改] 主求解函数，现在返回 (best_pos, max_area)
        """
        # Pass 1: Label all islands
        island_id = 2
        has_zero = False
        initial_max_area = 0

        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c]==0:
                    has_zero = True
                elif self.grid[r][c]==1 and self.island_labels[r][c]==0:
                    area = self._dfs_iterative(r, c, island_id)
                    self.island_areas[island_id] = area
                    initial_max_area = max(initial_max_area, area)
                    island_id += 1

        # 如果全是1，最大面积就是n*n，没有可移动的位置
        if not has_zero:
            return (None, self.n * self.n)

        # Pass 2: Evaluate each '0'
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

        # 最终的最大面积是 “翻转一个0能得到的最大面积” 和 “原图中本就存在的最大岛屿面积” 中的较大者
        final_max_area = max(max_area, initial_max_area)

        # 如果全是0，翻转任意一个得到面积1
        if best_pos is None and has_zero:
            return ((0, 0), 1)

        return (best_pos, final_max_area)


def generate_puzzle_and_solve(n):
    grid = [[random.choices([0, 1], weights=[0.4, 0.6])[0] for _ in range(n)] for _ in range(n)]
    solver = LargestIslandSolver(grid)
    solution_pos, max_area = solver.solve()
    return grid, solution_pos, max_area


def create_dataset(num_samples, n, output_path):
    """
    创建完整的数据集文件，每个样本包含：
    1. input: 扁平化的输入网格字符串。
    2. output_class: 最佳位置的类别标签 (0-n*n-1)。
    3. output_area: 最大面积的二进制编码字符串。
    """
    print(f"--- 开始生成 '最大人工岛' 数据集 (位置+面积) ---")
    print(f"网格尺寸: {n}x{n}, 目标样本数量: {num_samples}")

    # <<< 新增：计算面积编码所需的位数
    max_possible_area = n * n
    area_bit_count = max_possible_area.bit_length()
    print(f"面积编码将使用 {area_bit_count} 位。")

    all_samples = []
    seen_inputs = set()

    with tqdm(total=num_samples, desc="生成不重复样本") as pbar:
        while len(all_samples) < num_samples:
            grid, solution_pos, max_area = generate_puzzle_and_solve(n)

            input_str = "".join(map(str, [cell for row in grid for cell in row]))

            if input_str in seen_inputs: continue
            seen_inputs.add(input_str)

            # --- 输出1：位置类别 ---
            output_class = -1  # 用-1表示没有可翻转的位置
            if solution_pos is not None:
                r, c = solution_pos
                output_class = r * n + c

            # --- 新增：输出2：面积编码 ---
            area_bin = f'{max_area:0{area_bit_count}b}'

            all_samples.append({
                "input": input_str,
                "output_class": output_class,
                "output_area": area_bin
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
    NUM_SAMPLES = 500000
    GRID_SIZE = 6  # 为了让分类任务 manageable，我们用10x10，总共100个类别
    OUTPUT_FILE = 'largest_island_pos_area_6_6_dataset.jsonl'

    create_dataset(
        num_samples=NUM_SAMPLES,
        n=GRID_SIZE,
        output_path=OUTPUT_FILE
    )
