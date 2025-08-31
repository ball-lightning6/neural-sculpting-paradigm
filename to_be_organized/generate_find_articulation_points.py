import json
import random
from tqdm import tqdm
from collections import deque


class MinDaysSolver:
    """
    算法预言机：解决“使陆地分离的最少天数”问题。
    它不仅返回天数，还返回最终的网格状态。
    """

    def __init__(self, grid):
        self.initial_grid = [row[:] for row in grid]
        self.m = len(grid)
        self.n = len(grid[0])

    def _count_islands(self, grid):
        """使用DFS计算给定网格中的岛屿数量"""
        if not grid:
            return 0

        visited = set()
        islands = 0
        for r in range(self.m):
            for c in range(self.n):
                if grid[r][c]==1 and (r, c) not in visited:
                    islands += 1
                    stack = [(r, c)]
                    visited.add((r, c))
                    while stack:
                        row, col = stack.pop()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            if (0 <= nr < self.m and 0 <= nc < self.n and
                                    grid[nr][nc]==1 and (nr, nc) not in visited):
                                visited.add((nr, nc))
                                stack.append((nr, nc))
        return islands

    def solve(self):
        """
        执行三步决策，返回 (天数, 最终网格)
        """
        # 步骤 1: 检查初始状态 (0天)
        cut_grid = [[0]*len(row) for row in self.initial_grid]
        initial_islands = self._count_islands(self.initial_grid)
        if initial_islands!=1:
            return (0, self.initial_grid)

        # 步骤 2: 检查移除1个点的情况 (1天)
        land_cells = []
        for r in range(self.m):
            for c in range(self.n):
                if self.initial_grid[r][c]==1:
                    land_cells.append((r, c))

        for r, c in land_cells:
            temp_grid = [row[:] for row in self.initial_grid]
            temp_grid[r][c] = 0
            if self._count_islands(temp_grid)!=1:
                # 找到了一个割点，返回修改后的网格
                return (1, temp_grid)

        # 步骤 3: 如果以上都不成立，答案必然是2天
        # 我们需要一个确定性的方式来移除两个点。
        # 一个简单的方式是移除前两个找到的陆地单元格。
        # 只要岛屿大小 > 1，这就能保证分离（或完全移除）
        final_grid = [row[:] for row in self.initial_grid]
        if len(land_cells) > 0:
            r1, c1 = land_cells[0]
            final_grid[r1][c1] = 0
            cut_grid[r1][c1] = 1
        if len(land_cells) > 1:
            r2, c2 = land_cells[1]
            final_grid[r2][c2] = 0
            cut_grid[r2][c2] = 1


        return (2, cut_grid)


def create_dataset(num_samples, m, n, output_path):
    """
    创建完整的数据集文件。
    输入: 扁平化的0/1字符串。
    输出: 一个代表最终网格状态的0/1整数列表。
    """
    print(f"--- 开始生成 '使陆地分离的最少天数' 数据集 ---")
    print(f"网格尺寸: {m}x{n}, 目标样本数量: {num_samples}")

    all_samples = []
    seen_inputs = set()

    with tqdm(total=num_samples, desc="生成不重复样本") as pbar:
        while len(all_samples) < num_samples:
            # 生成随机网格，让水稍微多一点以产生更有趣的形状
            grid = [[random.choices([0, 1], weights=[0.6, 0.4])[0] for _ in range(n)] for _ in range(m)]

            # 去重
            input_key = tuple(cell for row in grid for cell in row)
            if input_key in seen_inputs:
                continue
            seen_inputs.add(input_key)

            # 使用算法预言机求解
            solver = MinDaysSolver(grid)
            _days, final_grid = solver.solve()

            # 格式化输入和输出
            input_str = "".join(map(str, input_key))
            output_list = [cell for row in final_grid for cell in row]

            all_samples.append({
                "input": input_str,
                "output": output_list
            })
            pbar.update(1)

    print("\n样本生成完毕，开始打乱顺序...")
    random.shuffle(all_samples)
    print("打乱完成！")

    print("开始写入文件...")
    with open(output_path, 'w') as f:
        for sample in tqdm(all_samples, desc="写入文件"):
            # json.dumps 会自动处理列表
            f.write(json.dumps(sample) + '\n')

    print(f"数据集创建成功！文件已保存至: {output_path}")


if __name__=='__main__':
    NUM_SAMPLES = 500000
    GRID_M, GRID_N = 8, 8
    OUTPUT_FILE = 'disconnect_island_dataset.jsonl'

    create_dataset(
        num_samples=NUM_SAMPLES,
        m=GRID_M,
        n=GRID_N,
        output_path=OUTPUT_FILE
    )
