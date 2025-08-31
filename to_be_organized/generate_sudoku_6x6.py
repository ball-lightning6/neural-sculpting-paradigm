import random
import json
import argparse
from tqdm import tqdm
import copy


class Sudoku6x6Generator:
    """
    生成并验证 6x6 数独谜题的类。
    网格为 6x6，分为 6 个 2x3 的子区域。
    """

    def __init__(self):
        self.grid_size = 6
        self.box_rows = 2
        self.box_cols = 3
        self.numbers = list(range(1, self.grid_size + 1))
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.solution_count = 0

    def _find_empty(self):
        """找到网格中第一个空格子"""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c]==0:
                    return r, c
        return None

    def _is_valid(self, row, col, num):
        """检查在 (row, col) 位置放置 num 是否合法"""
        # 检查行
        if num in self.grid[row]:
            return False
        # 检查列
        if num in [self.grid[i][col] for i in range(self.grid_size)]:
            return False
        # 检查 2x3 子区域
        box_start_row, box_start_col = row - row % self.box_rows, col - col % self.box_cols
        for r in range(self.box_rows):
            for c in range(self.box_cols):
                if self.grid[box_start_row + r][box_start_col + c]==num:
                    return False
        return True

    def _solve_grid(self):
        """使用回溯算法填充网格，用于生成完整解"""
        find = self._find_empty()
        if not find:
            return True
        else:
            row, col = find

        # 随机化数字顺序以生成不同的数独
        random.shuffle(self.numbers)
        for num in self.numbers:
            if self._is_valid(row, col, num):
                self.grid[row][col] = num
                if self._solve_grid():
                    return True
                self.grid[row][col] = 0  # 回溯
        return False

    def _count_solutions(self):
        """使用回溯算法计算当前网格的解的数量"""
        find = self._find_empty()
        if not find:
            self.solution_count += 1
            return

        row, col = find
        for num in self.numbers:
            if self._is_valid(row, col, num):
                self.grid[row][col] = num
                self._count_solutions()
                # 找到一个解后，必须回溯以寻找其他解
                self.grid[row][col] = 0
            # 如果已经找到超过1个解，可以提前终止以提高效率
            if self.solution_count > 1:
                return

    def generate_full_solution(self):
        """生成一个完整的、随机的6x6数独解"""
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self._solve_grid()
        return copy.deepcopy(self.grid)

    def create_puzzle(self, solution, difficulty=0.6):
        """
        从完整解中“挖洞”生成谜题。
        difficulty: 移除数字的比例 (0.1 - 0.8)，越高越难。
        """
        puzzle = copy.deepcopy(solution)
        cells = list(range(self.grid_size * self.grid_size))
        random.shuffle(cells)

        # 根据难度决定要移除的格子数量
        cells_to_remove = int(len(cells) * difficulty)

        for cell_index in cells[:cells_to_remove]:
            row = cell_index // self.grid_size
            col = cell_index % self.grid_size

            # 暂存被移除的数字
            temp = puzzle[row][col]
            puzzle[row][col] = 0

            # 检查移除后是否仍有唯一解
            self.grid = copy.deepcopy(puzzle)
            self.solution_count = 0
            self._count_solutions()

            # 如果解不唯一（0个或多个），则恢复该数字
            if self.solution_count!=1:
                puzzle[row][col] = temp

        return puzzle


# --- 格式化辅助函数 ---

def number_to_binary_list(n):
    """将数字 1-6 转换为一个3位的二进制列表 [0,0,1] 到 [1,1,0]"""
    if not 1 <= n <= 6:
        raise ValueError("Number must be between 1 and 6")
    # format(n, '03b') -> '001', '010', ...
    return [int(bit) for bit in format(n, '03b')]


def format_puzzle(puzzle_grid):
    """将谜题网格转换为 '1_3__6...' 格式的字符串"""
    return "".join(str(cell) if cell!=0 else '_' for row in puzzle_grid for cell in row)


def format_solution(solution_grid):
    """将解网格转换为 108 位的 0/1 列表"""
    output_list = []
    for row in solution_grid:
        for cell in row:
            output_list.extend(number_to_binary_list(cell))
    return output_list


# --- 主程序 ---

def main(num_puzzles, output_file, difficulty):
    """主函数，生成并写入数独数据"""
    generator = Sudoku6x6Generator()
    seen_puzzles = set()

    print(f"开始生成 {num_puzzles} 个 6x6 数独谜题...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用 tqdm 创建进度条
        pbar = tqdm(total=num_puzzles)
        while len(seen_puzzles) < num_puzzles:
            # 1. 生成完整解
            solution = generator.generate_full_solution()

            # 2. 从解创建谜题并验证唯一性
            puzzle = generator.create_puzzle(solution, difficulty)

            # 3. 格式化
            input_str = format_puzzle(puzzle)

            # 4. 检查重复并写入
            if input_str not in seen_puzzles:
                seen_puzzles.add(input_str)

                output_list = format_solution(solution)

                data = {
                    "input": input_str,
                    "output": output_list
                }

                f.write(json.dumps(data) + '\n')
                pbar.update(1)
        pbar.close()

    print(f"\n成功生成 {len(seen_puzzles)} 个不重复的谜题，已保存至 {output_file}")


# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description="生成 6x6 数独谜题数据集")
#     parser.add_argument(
#         "num_puzzles",
#         type=int,
#         help="要生成的数据集条数"
#     )
#     parser.add_argument(
#         "-o", "--output",
#         default="sudoku_6x6_dataset.jsonl",
#         help="输出的 jsonl 文件名"
#     )
#     parser.add_argument(
#         "-d", "--difficulty",
#         type=float,
#         default=0.6,
#         help="谜题难度（移除格子的比例），范围 0.1 到 0.8"
#     )
#
#     args = parser.parse_args()
#
#     if not 0.1 <= args.difficulty <= 0.8:
#         raise ValueError("Difficulty must be between 0.1 and 0.8")
difficulty = 0.6
num_puzzles = 1000000
output = 'sudoku_6_6.jsonl'
main(num_puzzles, output, difficulty)
