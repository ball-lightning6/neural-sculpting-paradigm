import json
import random
import collections
import heapq
from tqdm import tqdm
import os
import time

# --- ANSI颜色代码，用于美化终端输出 ---
COLOR_PLAYER = '\033[92m'  # 亮绿色
COLOR_BOX = '\033[93m'  # 亮黄色
COLOR_TARGET = '\033[91m'  # 亮红色
COLOR_WALL = '\033[90m'  # 深灰色
COLOR_RESET = '\033[0m'


class SokobanSolver:
    """
    一个高效的推箱子求解器，融合了A*搜索和Tarjan算法。
    包含两个求解方法：一个为速度优化，一个为获取完整路径。
    """

    def __init__(self, grid):
        self.m, self.n = len(grid), len(grid[0])
        self.grid = grid
        self.walls = set()
        self.floor = set()
        self.player_start, self.box_start, self.target_pos = None, None, None

        for r in range(self.m):
            for c in range(self.n):
                pos = (r, c)
                if grid[r][c]=='#':
                    self.walls.add(pos)
                else:
                    self.floor.add(pos)
                    if grid[r][c]=='S':
                        self.player_start = pos
                    elif grid[r][c]=='B':
                        self.box_start = pos
                    elif grid[r][c]=='T':
                        self.target_pos = pos

        self.directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.move_map = {v: k for k, v in self.directions.items()}
        self._tarjan_preprocess()

    def _tarjan_preprocess(self):
        """使用Tarjan算法计算双连通分量，用于O(1)寻路检查。"""
        self.low = {pos: 0 for pos in self.floor}
        self.dfn = self.low.copy()
        self.count = 0

        def tarjan(u, parent):
            self.count += 1
            self.dfn[u] = self.low[u] = self.count
            for _, (dr, dc) in self.directions.items():
                v = (u[0] + dr, u[1] + dc)
                if v in self.floor and v!=parent:
                    if not self.dfn.get(v):  # Use .get for safety
                        tarjan(v, u)
                        self.low[u] = min(self.low[u], self.low.get(v, float('inf')))
                    else:
                        self.low[u] = min(self.low[u], self.dfn.get(v, float('inf')))

        if self.player_start in self.floor:
            tarjan(self.player_start, None)

    def _find_player_path(self, p_start, p_end, box_at):
        """BFS寻找玩家的完整路径"""
        q = collections.deque([(p_start, [])])
        visited = {p_start}
        while q:
            curr_pos, path = q.popleft()
            if curr_pos==p_end:
                return path
            for move_name, (dr, dc) in self.directions.items():
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                if next_pos in self.floor and next_pos!=box_at and next_pos not in visited:
                    visited.add(next_pos)
                    q.append((next_pos, path + [move_name]))
        return None

    def _find_player_first_move(self, p_start, p_end, box_at):
        """一个简单的BFS，只为了找到玩家路径的第一步"""
        q = collections.deque([(p_start, None)])  # (pos, first_move_dir)
        visited = {p_start}
        while q:
            curr_pos, first_move = q.popleft()
            for move_name, (dr, dc) in self.directions.items():
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                if next_pos==p_end:
                    return first_move if first_move is not None else move_name
                if next_pos in self.floor and next_pos!=box_at and next_pos not in visited:
                    visited.add(next_pos)
                    q.append((next_pos, first_move if first_move is not None else move_name))
        return None

    def solve_for_next_move(self):
        """[快速版] 求解器，只返回最优第一步"""

        def heuristic(box_pos, pushes):
            dist = abs(box_pos[0] - self.target_pos[0]) + abs(box_pos[1] - self.target_pos[1])
            return dist + pushes, dist

        pq = [(heuristic(self.box_start, 0), 0, self.player_start, self.box_start, None)]
        visited_states = set()

        while pq:
            _, pushes, p_pos, b_pos, first_move = heapq.heappop(pq)
            if b_pos==self.target_pos: return first_move
            if (p_pos, b_pos) in visited_states: continue
            visited_states.add((p_pos, b_pos))

            for move_name, (dr, dc) in self.directions.items():
                player_needed_pos, new_box_pos = (b_pos[0] - dr, b_pos[1] - dc), (b_pos[0] + dr, b_pos[1] + dc)
                if not (new_box_pos in self.floor and player_needed_pos in self.floor): continue

                if self.low.get(p_pos)==self.low.get(player_needed_pos):
                    current_first_move = first_move
                    if current_first_move is None:
                        if p_pos==player_needed_pos:
                            current_first_move = move_name
                        else:
                            current_first_move = self._find_player_first_move(p_pos, player_needed_pos, b_pos)

                    if current_first_move and (player_needed_pos, new_box_pos) not in visited_states:
                        new_pushes, new_player_pos = pushes + 1, b_pos
                        heapq.heappush(pq, (heuristic(new_box_pos, new_pushes), new_pushes, new_player_pos, new_box_pos,
                                            current_first_move))
        return None

    def solve_for_full_path(self):
        """[完整版] 求解器，返回完整的最优路径用于可视化"""

        def heuristic(box_pos, pushes):
            dist = abs(box_pos[0] - self.target_pos[0]) + abs(box_pos[1] - self.target_pos[1])
            return dist + pushes, dist

        pq = [(heuristic(self.box_start, 0), 0, self.player_start, self.box_start, [])]
        visited_states = set()

        while pq:
            _, pushes, p_pos, b_pos, path_history = heapq.heappop(pq)
            if b_pos==self.target_pos: return path_history
            if (p_pos, b_pos) in visited_states: continue
            visited_states.add((p_pos, b_pos))

            for move_name, (dr, dc) in self.directions.items():
                player_needed_pos, new_box_pos = (b_pos[0] - dr, b_pos[1] - dc), (b_pos[0] + dr, b_pos[1] + dc)
                if not (new_box_pos in self.floor and player_needed_pos in self.floor): continue

                if self.low.get(p_pos)==self.low.get(player_needed_pos):
                    player_path = self._find_player_path(p_pos, player_needed_pos, b_pos)
                    if player_path is not None:
                        new_path_history = path_history + [('player', move) for move in player_path] + [
                            ('push', move_name)]
                        new_pushes, new_player_pos = pushes + 1, b_pos
                        heapq.heappush(pq, (
                        heuristic(new_box_pos, new_pushes), new_pushes, new_player_pos, new_box_pos, new_path_history))
        return None


def generate_open_puzzle_and_solve(m, n):
    """生成一个开放式谜题，并使用高效求解器求解"""
    while True:
        grid = [['.' for _ in range(n)] for _ in range(m)]
        for _ in range(int(m * n * 0.25)):
            r, c = random.randint(0, m - 1), random.randint(0, n - 1)
            grid[r][c] = '#'

        floor_tiles = [(r, c) for r in range(m) for c in range(n) if grid[r][c]=='.']
        if len(floor_tiles) < 5: continue

        random.shuffle(floor_tiles)
        try:
            p, b, t = floor_tiles.pop(), floor_tiles.pop(), floor_tiles.pop()
            grid[p[0]][p[1]], grid[b[0]][b[1]], grid[t[0]][t[1]] = 'S', 'B', 'T'
        except IndexError:
            continue

        solver = SokobanSolver(grid)
        if not (solver.low.get(p) and solver.low.get(b) and solver.low.get(t)): continue

        solution_move = solver.solve_for_next_move()
        if solution_move is not None:
            return grid, solution_move


def create_dataset(num_samples, output_path, m, n):
    """创建完整的数据集文件"""
    print(f"开始生成 {num_samples} 个开放式推箱子样本 (使用A*+Tarjan优化)...")
    print(f"固定尺寸: M={m}, N={n}")

    all_samples = []
    seen_inputs = set()
    move_to_class = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

    with tqdm(total=num_samples, desc="生成可解谜题") as pbar:
        while len(all_samples) < num_samples:
            result = generate_open_puzzle_and_solve(m, n)
            if result is None: continue
            grid, solution_move = result

            input_str = "".join("".join(row) for row in grid)

            if input_str not in seen_inputs:
                seen_inputs.add(input_str)
                output_class = move_to_class[solution_move]
                all_samples.append({"input": input_str, "output": output_class})
                pbar.update(1)

    print("样本生成完毕，开始打乱顺序...")
    random.shuffle(all_samples)

    print(f"开始写入到文件: {output_path}")
    with open(output_path, 'w') as f:
        for sample in tqdm(all_samples, desc="写入文件"):
            f.write(json.dumps(sample) + '\n')

    print(f"数据集创建成功！文件已保存至: {output_path}")


def print_grid_visual(grid, player_pos, box_pos, target_pos):
    """在终端打印当前的游戏状态，带颜色。"""
    os.system('cls' if os.name=='nt' else 'clear')
    m, n = len(grid), len(grid[0])
    print("+" + "---" * n + "+")
    for r in range(m):
        print("|", end="")
        for c in range(n):
            pos = (r, c)
            char_at_pos = grid[r][c]
            display_char = ' '
            color = COLOR_RESET
            if pos==player_pos:
                display_char, color = 'S', COLOR_PLAYER
            elif pos==box_pos:
                display_char, color = 'B', COLOR_BOX
                if pos==target_pos:
                    color = COLOR_TARGET
            elif char_at_pos=='T':
                display_char, color = 'T', COLOR_TARGET
            elif char_at_pos=='#':
                display_char, color = '#', COLOR_WALL
            else:
                display_char = '.'
            print(f" {color}{display_char}{COLOR_RESET} ", end="")
        print("|")
    print("+" + "---" * n + "+")


def visualize_solution(initial_grid):
    """生成一个谜题，求解并用动画展示完整解法。"""
    print("\n" + "=" * 40)
    print("--- 启动可视化演示 ---")
    print("=" * 40)

    solver = SokobanSolver(initial_grid)
    p_pos, b_pos, t_pos = solver.player_start, solver.box_start, solver.target_pos

    print("1. 计算'最佳第一步' (用于数据集)...")
    correct_first_move = solver.solve_for_next_move()
    if not correct_first_move:
        print("此谜题无法找到解，可视化中止。")
        return

    print(f"   > 算法计算出的最佳第一步是: {COLOR_PLAYER}{correct_first_move}{COLOR_RESET}")

    print("2. 计算'完整最佳路径' (用于动画)...")
    full_path = solver.solve_for_full_path()

    print("3. 开始播放动画...(按 Ctrl+C 中止)")
    time.sleep(2)

    for i, (action_type, move) in enumerate(full_path):
        dr, dc = solver.directions[move]

        if action_type=='player':
            p_pos = (p_pos[0] + dr, p_pos[1] + dc)
            status = f"玩家移动: {move}"
        elif action_type=='push':
            p_pos = (p_pos[0] + dr, p_pos[1] + dc)
            b_pos = (b_pos[0] + dr, b_pos[1] + dc)
            status = f"玩家推动箱子: {move}"

        print_grid_visual(initial_grid, p_pos, b_pos, t_pos)
        print(f"步骤 {i + 1}/{len(full_path)}: {status}")

        if b_pos==t_pos:
            print(f"\n{COLOR_TARGET}恭喜！箱子已到达目标位置！{COLOR_RESET}")
            break

        time.sleep(0.4)

    print("\n--- 可视化演示结束 ---")


if __name__=='__main__':
    # --- 模式选择 ---
    mode = "GENERATE_DATASET"
    # mode = "VISUALIZE"

    if mode=="GENERATE_DATASET":
        NUM_SAMPLES = 5000
        OUTPUT_FILE = 'sokoban_optimized_dataset.jsonl'
        M_FIXED, N_FIXED = 10, 10
        create_dataset(NUM_SAMPLES, OUTPUT_FILE, M_FIXED, N_FIXED)

    elif mode=="VISUALIZE":
        while True:
            try:
                print("正在寻找一个用于演示的谜题...")
                m_vis, n_vis = 8, 8
                result = generate_open_puzzle_and_solve(m_vis, n_vis)
                if result:
                    grid, _ = result
                    visualize_solution(grid)
                    break
            except Exception as e:
                print(f"生成或演示时出错: {e}, 正在重试...")
                continue
