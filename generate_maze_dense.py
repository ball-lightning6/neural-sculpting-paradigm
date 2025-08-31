import json
import random
from collections import deque
from tqdm import tqdm

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================

# --- 数据集设置 ---
TARGET_NUM_SAMPLES = 2000000
OUTPUT_FILE = "maze_optimized_13_13_dataset1.jsonl"

# --- 迷宫几何设置 ---
MAZE_HEIGHT = 15
MAZE_WIDTH = 15

# --- 可视化设置 ---
NUM_SAMPLES_TO_VISUALIZE = 5


# ==============================================================================
# --- 核心代码 ---
# ==============================================================================

class OptimizedMazeGenerator:
    def __init__(self, height, width):
        if height % 2==0 or width % 2==0:
            print(f"警告: 迷宫尺寸 ({height}x{width}) 最好是奇数，以获得最佳的墙壁结构。")
        self.height = height
        self.width = width
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.move_order = ['U', 'D', 'L', 'R']
        self.move_map = {move: i for i, move in enumerate(self.move_order)}

    def generate_maze(self):
        maze = [[1] * self.width for _ in range(self.height)]
        start_r, start_c = random.randrange(1, self.height, 2), random.randrange(1, self.width, 2)
        maze[start_r][start_c] = 0
        stack = [(start_r, start_c)]
        while stack:
            current_r, current_c = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = current_r + dr, current_c + dc
                if 0 < nr < self.height and 0 < nc < self.width and maze[nr][nc]==1:
                    neighbors.append((nr, nc))
            if neighbors:
                next_r, next_c = random.choice(neighbors)
                wall_r, wall_c = (current_r + next_r) // 2, (current_c + next_c) // 2
                maze[wall_r][wall_c] = 0
                maze[next_r][next_c] = 0
                stack.append((next_r, next_c))
            else:
                stack.pop()
        path_cells = []
        for r in range(self.height):
            for c in range(self.width):
                if maze[r][c]==0:
                    path_cells.append((r, c))
        if len(path_cells) < 2: return self.generate_maze()
        start_pos, target_pos = random.sample(path_cells, 2)
        return maze, start_pos, target_pos

    def solve_with_bfs(self, maze, target_pos):
        q = deque([(target_pos, 0)])
        distances = {target_pos: 0}
        while q:
            (r, c), dist = q.popleft()
            # 【BUG已修复】之前这里错误地使用了 self.move_map.keys()
            for dr, dc in self.moves.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and \
                        maze[nr][nc]==0 and (nr, nc) not in distances:
                    distances[(nr, nc)] = dist + 1
                    q.append(((nr, nc), dist + 1))
        return distances

    def generate_dataset(self):
        print("=" * 60)
        print("高效迷宫数据集生成器 (v3.1 - Bug修复版)")
        print(f"目标唯一样本数: {TARGET_NUM_SAMPLES}, 内部尺寸: {self.height - 2}x{self.width - 2}")
        print("=" * 60)

        all_data_points, seen_inputs = [], set()
        mazes_generated, duplicates_skipped = 0, 0

        with tqdm(total=TARGET_NUM_SAMPLES, desc="收集唯一样本") as pbar:
            while len(all_data_points) < TARGET_NUM_SAMPLES:
                mazes_generated += 1
                maze, _, target_pos = self.generate_maze()
                distances = self.solve_with_bfs(maze, target_pos)

                potential_points = list(distances.items())
                random.shuffle(potential_points)

                for pos, dist in potential_points:
                    if pos==target_pos: continue

                    optimal_move_index = -1
                    # 注意：这里的循环是正确的，因为它需要move_name
                    for move_name, (dr, dc) in self.moves.items():
                        next_pos = (pos[0] + dr, pos[1] + dc)
                        if next_pos in distances and distances[next_pos] < dist:
                            optimal_move_index = self.move_map[move_name]
                            break

                    if optimal_move_index==-1: continue

                    temp_maze = [row[:] for row in maze]
                    temp_maze[pos[0]][pos[1]] = 's'
                    temp_maze[target_pos[0]][target_pos[1]] = 't'

                    inner_maze = [row[1:-1] for row in temp_maze[1:-1]]
                    input_str = "".join(map(str, [cell for row in inner_maze for cell in row]))

                    optimal_move_str = f'{optimal_move_index:02b}'
                    output_list = list(map(int, optimal_move_str))
                    if input_str not in seen_inputs:
                        seen_inputs.add(input_str)
                        all_data_points.append({"input": input_str, "output": output_list})
                        pbar.update(1)
                    else:
                        duplicates_skipped += 1

                    pbar.set_postfix_str(f"迷宫: {mazes_generated}, 重复: {duplicates_skipped}")
                    if len(all_data_points) >= TARGET_NUM_SAMPLES: break
                if len(all_data_points) >= TARGET_NUM_SAMPLES: break

        print(f"\n生成了 {len(all_data_points)} 个唯一数据点。正在进行全局洗牌...")
        random.shuffle(all_data_points)
        print(f"正在写入文件: {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            for data_point in tqdm(all_data_points, desc="写入文件"):
                f.write(json.dumps(data_point) + '\n')
        print("\n🎉🎉🎉 数据集生成完毕！ 🎉🎉🎉")
        return all_data_points


def visualize_samples(dataset, num_samples, height, width):
    if not dataset: return
    print("\n" + "=" * 20 + " 数据集样本可视化 " + "=" * 20)
    VIZ_MAP = {'0': ' ', '1': '█', 's': 'S', 't': 'T'}
    move_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    samples_to_show = random.sample(dataset, min(num_samples, len(dataset)))
    for i, sample in enumerate(samples_to_show):
        print(f"\n--- 样本 {i + 1}/{len(samples_to_show)} ---")
        input_str = sample["input"]
        output_int = sample["output"]

        print(f"Input (内部尺寸: {height}x{width}):")
        print("+" + "---" * width + "+")
        for r in range(height):
            print("|", end="")
            for c in range(width):
                char = input_str[r * width + c]
                viz_char = VIZ_MAP.get(char, '?')
                print(f" {viz_char} ", end="")
            print("|")
        print("+" + "---" * width + "+")

        print(f"Output (最优移动): {output_int}  =>  {move_names[output_int]}")
    print("\n" + "=" * 58)


if __name__=='__main__':
    generator = OptimizedMazeGenerator(height=MAZE_HEIGHT, width=MAZE_WIDTH)
    full_dataset = generator.generate_dataset()

    # visualize_samples(
    #     dataset=full_dataset,
    #     num_samples=NUM_SAMPLES_TO_VISUALIZE,
    #     height=MAZE_HEIGHT - 2,
    #     width=MAZE_WIDTH - 2
    # )
