import json
import random
from collections import deque
from typing import List, Tuple, Set, Optional
import copy


class SokobanGenerator:
    def __init__(self, m: int, n: int):
        self.m = m  # 行数
        self.n = n  # 列数
        self.directions = [0, -1, 0, 1, 0]  # 方向数组：上左下右
        self.dir_names = ['up', 'left', 'down', 'right']

    def minPushBox(self, grid: List[List[str]]) -> Tuple[int, List[int]]:
        """
        计算推箱子最短步数，并返回所有最优的第一步方向
        返回 (最短步数, 最优第一步方向列表)
        """
        m, n = self.m, self.n
        sx, sy, bx, by, tx, ty = None, None, None, None, None, None

        # 找到人(S)、箱子(B)、目标(T)的位置
        for x in range(m):
            for y in range(n):
                if grid[x][y]=='S':
                    sx, sy = x, y
                elif grid[x][y]=='B':
                    bx, by = x, y
                elif grid[x][y]=='T':
                    tx, ty = x, y

        # 不越界且不在墙上
        def ok(x, y):
            return (0 <= x < m and 0 <= y < n and grid[x][y]!='#')

        d = self.directions

        # dp[人的位置][箱子的位置] = 最少推动次数
        dp = [[float('inf')] * (m * n) for _ in range(m * n)]
        # parent[人的位置][箱子的位置] = (上一步的人位置, 上一步的箱子位置, 移动方向)
        parent = [[None] * (m * n) for _ in range(m * n)]

        initial_s = sx * n + sy
        initial_b = bx * n + by
        dp[initial_s][initial_b] = 0

        q = deque([(initial_s, initial_b)])
        target_b = tx * n + ty

        while q:
            q1 = deque()
            while q:
                s1, b1 = q.popleft()
                sx1, sy1 = s1 // n, s1 % n
                bx1, by1 = b1 // n, b1 % n

                if b1==target_b:  # 箱子到达目标
                    # 回溯找到所有最优的第一步
                    return self.backtrack_first_moves(dp, parent, initial_s, initial_b, s1, b1)

                for i in range(4):  # 人向四个方向移动
                    sx2, sy2 = sx1 + d[i], sy1 + d[i + 1]
                    s2 = sx2 * n + sy2

                    if not ok(sx2, sy2):  # 人的新位置不合法
                        continue

                    if sx2==bx1 and sy2==by1:  # 推动箱子
                        bx2, by2 = bx1 + d[i], by1 + d[i + 1]
                        b2 = bx2 * n + by2

                        if not ok(bx2, by2) or dp[s2][b2] <= dp[s1][b1] + 1:
                            continue

                        dp[s2][b2] = dp[s1][b1] + 1
                        parent[s2][b2] = (s1, b1, i)
                        q1.append((s2, b2))
                    else:  # 只是人移动
                        if dp[s2][b1] <= dp[s1][b1]:
                            continue

                        dp[s2][b1] = dp[s1][b1]
                        parent[s2][b1] = (s1, b1, i)
                        q.append((s2, b1))

            q, q1 = q1, q

        return -1, []

    def backtrack_first_moves(self, dp, parent, initial_s, initial_b, final_s, final_b):
        """回溯找到所有最优的第一步移动方向"""
        min_steps = dp[final_s][final_b]

        # 找到所有能达到最优解的第一步
        first_moves = set()

        # BFS找到所有最优路径的第一步
        def find_all_first_moves():
            # 从终点回溯到起点
            stack = [(final_s, final_b)]
            visited = set()

            while stack:
                curr_s, curr_b = stack.pop()
                if (curr_s, curr_b) in visited:
                    continue
                visited.add((curr_s, curr_b))

                if curr_s==initial_s and curr_b==initial_b:
                    continue

                # 查找所有能到达当前状态的前一步
                for prev_s in range(self.m * self.n):
                    for prev_b in range(self.m * self.n):
                        if parent[curr_s][curr_b] and parent[curr_s][curr_b][0]==prev_s and parent[curr_s][curr_b][
                            1]==prev_b:
                            if prev_s==initial_s and prev_b==initial_b:
                                # 这是第一步
                                move_dir = parent[curr_s][curr_b][2]
                                first_moves.add(move_dir)
                            else:
                                stack.append((prev_s, prev_b))

        # 简化版本：直接从初始状态开始，找到所有最优的第一步
        for i in range(4):
            sx1, sy1 = initial_s // self.n, initial_s % self.n
            sx2, sy2 = sx1 + self.directions[i], sy1 + self.directions[i + 1]

            if not (0 <= sx2 < self.m and 0 <= sy2 < self.n):
                continue

            s2 = sx2 * self.n + sy2

            # 检查这一步是否能导向最优解
            if sx2==initial_b // self.n and sy2==initial_b % self.n:
                # 第一步就推箱子
                bx2 = (initial_b // self.n) + self.directions[i]
                by2 = (initial_b % self.n) + self.directions[i + 1]
                if 0 <= bx2 < self.m and 0 <= by2 < self.n:
                    b2 = bx2 * self.n + by2
                    if dp[s2][b2]==min_steps:
                        first_moves.add(i)
            else:
                # 第一步只是人移动
                if dp[s2][initial_b] + self.count_min_pushes_from(s2, initial_b, dp)==min_steps:
                    first_moves.add(i)

        return min_steps, list(first_moves)

    def count_min_pushes_from(self, start_s, start_b, dp):
        """从给定状态开始的最少推动次数"""
        min_pushes = float('inf')
        for s in range(self.m * self.n):
            for b in range(self.m * self.n):
                if dp[s][b] < float('inf'):
                    min_pushes = min(min_pushes, dp[s][b])
        return min_pushes if min_pushes!=float('inf') else 0

    def generate_valid_map(self) -> Optional[Tuple[List[List[str]], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """生成一个有效的推箱子地图"""
        max_attempts = 200

        for attempt in range(max_attempts):
            # 创建基础地图框架
            grid = [['#' for _ in range(self.n)] for _ in range(self.m)]

            # 在内部创建开放空间
            for i in range(1, self.m - 1):
                for j in range(1, self.n - 1):
                    if random.random() < 0.75:  # 75%概率是空地
                        grid[i][j] = '.'

            # 确保有足够的空地
            empty_cells = [(i, j) for i in range(1, self.m - 1) for j in range(1, self.n - 1) if grid[i][j]=='.']
            if len(empty_cells) < 8:  # 需要足够的空地
                continue

            # 随机选择人、箱子、目标位置
            attempts_inner = 0
            while attempts_inner < 30:
                attempts_inner += 1
                selected = random.sample(empty_cells, 3)
                person_pos = selected[0]
                box_pos = selected[1]
                target_pos = selected[2]

                # 确保箱子和目标有足够距离，但不要太远
                dist = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                if dist < 2 or dist > min(self.m, self.n):
                    continue

                # 确保箱子不在角落
                bx, by = box_pos
                if self.is_corner(grid, bx, by):
                    continue

                # 创建临时地图
                temp_grid = copy.deepcopy(grid)
                temp_grid[person_pos[0]][person_pos[1]] = 'S'
                temp_grid[box_pos[0]][box_pos[1]] = 'B'
                temp_grid[target_pos[0]][target_pos[1]] = 'T'

                # 测试是否可解
                min_steps, first_moves = self.minPushBox(temp_grid)
                if min_steps!=-1 and min_steps > 0 and len(first_moves) > 0:
                    return temp_grid, person_pos, box_pos, target_pos

        return None

    def is_corner(self, grid: List[List[str]], x: int, y: int) -> bool:
        """检查位置是否在角落"""
        # 检查四个角落模式
        corner_patterns = [
            [(-1, 0), (0, -1)],  # 左上角
            [(-1, 0), (0, 1)],  # 右上角
            [(1, 0), (0, -1)],  # 左下角
            [(1, 0), (0, 1)]  # 右下角
        ]

        for pattern in corner_patterns:
            blocked_count = 0
            for dx, dy in pattern:
                nx, ny = x + dx, y + dy
                if (nx < 0 or nx >= self.m or ny < 0 or ny >= self.n or
                        grid[nx][ny]=='#'):
                    blocked_count += 1
            if blocked_count==2:  # 两个方向都被阻挡
                return True
        return False

    def solve_sokoban(self, initial_state: str) -> Optional[List[int]]:
        """求解推箱子，返回最优第一步方向"""
        grid = self.string_to_grid(initial_state)
        min_steps, first_moves = self.minPushBox(grid)

        if min_steps==-1 or len(first_moves)==0:
            return None

        # 转换为输出格式 [上, 下, 左, 右]
        result = [0, 0, 0, 0]
        direction_mapping = {0: 0, 2: 1, 1: 2, 3: 3}  # 上左下右 -> 上下左右

        for move in first_moves:
            mapped_move = direction_mapping[move]
            result[mapped_move] = 1

        return result

    def string_to_grid(self, state: str) -> List[List[str]]:
        """将字符串状态转换为网格"""
        grid = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                char = state[i * self.n + j]
                # 转换字符格式：s->S, b->B, *->T
                if char=='s':
                    char = 'S'
                elif char=='b':
                    char = 'B'
                elif char=='*':
                    char = 'T'
                row.append(char)
            grid.append(row)
        return grid

    def grid_to_string(self, grid: List[List[str]]) -> str:
        """将网格转换为字符串"""
        result = ""
        for i in range(self.m):
            for j in range(self.n):
                char = grid[i][j]
                # 转换字符格式：S->s, B->b, T->*
                if char=='S':
                    char = 's'
                elif char=='B':
                    char = 'b'
                elif char=='T':
                    char = '*'
                result += char
        return result

    def generate_dataset(self, num_samples: int) -> List[dict]:
        """生成训练数据集"""
        dataset = []
        seen_states = set()

        attempts = 0
        max_attempts = num_samples * 10

        print("Starting dataset generation...")

        while len(dataset) < num_samples and attempts < max_attempts:
            attempts += 1

            if attempts % 100==0:
                print(f"Attempts: {attempts}, Generated: {len(dataset)}")

            try:
                # 生成地图
                result = self.generate_valid_map()
                if result is None:
                    continue

                grid, person_pos, box_pos, target_pos = result
                state_str = self.grid_to_string(grid)

                # 避免重复
                if state_str in seen_states:
                    continue
                seen_states.add(state_str)

                # 求解
                optimal_moves = self.solve_sokoban(state_str)

                # 确保有解
                if optimal_moves and any(optimal_moves):
                    dataset.append({
                        "input": state_str,
                        "output": optimal_moves
                    })

                    if len(dataset) % 50==0:
                        print(f"Generated {len(dataset)} valid samples...")

            except Exception as e:
                if attempts % 500==0:  # 只偶尔打印错误
                    print(f"Error generating sample: {e}")
                continue

        # 打乱数据集
        random.shuffle(dataset)
        print(f"Dataset generation complete. Total valid samples: {len(dataset)}")
        return dataset


def main():
    # 参数设置
    M = 5  # 行数 (稍微小一点，更容易生成有效样本)
    N = 5  # 列数
    NUM_SAMPLES = 500  # 生成样本数 (先生成少一点测试)
    OUTPUT_FILE = "sokoban_train.jsonl"

    print(f"Generating Sokoban dataset: {M}x{N}, {NUM_SAMPLES} samples")

    # 设置随机种子
    random.seed(42)

    # 生成数据集
    generator = SokobanGenerator(M, N)
    dataset = generator.generate_dataset(NUM_SAMPLES)

    print(f"Successfully generated {len(dataset)} unique samples")

    if len(dataset) > 0:
        # 保存为JSONL格式
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Dataset saved to {OUTPUT_FILE}")

        # 显示一些样例
        print("\nSample data:")
        for i in range(min(3, len(dataset))):
            print(f"Sample {i + 1}:")
            state = dataset[i]["input"]
            output = dataset[i]["output"]

            # 格式化显示地图
            for row in range(M):
                print(''.join(state[row * N:(row + 1) * N]))
            print(f"Optimal moves: {output} (up:{output[0]}, down:{output[1]}, left:{output[2]}, right:{output[3]})")
            print()
    else:
        print("Failed to generate any valid samples. Try adjusting parameters.")


if __name__=="__main__":
    main()
