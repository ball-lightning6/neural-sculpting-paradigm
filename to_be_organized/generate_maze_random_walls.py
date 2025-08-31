import json
import random
from collections import deque
from tqdm import tqdm

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================

# --- 数据集设置 ---
# 注意：这是目标样本数。如果去重率高，实际生成的迷宫数会更多。
TARGET_NUM_SAMPLES = 50
OUTPUT_FILE = "maze_shortest_path_dataset_unique.jsonl"

# --- 迷宫几何设置 ---
MAZE_HEIGHT = 12
MAZE_WIDTH = 12

# --- 墙壁密度 ---
WALL_DENSITY = 0.4

# --- 可视化设置 ---
NUM_SAMPLES_TO_VISUALIZE = 50


# ==============================================================================
# --- 核心代码 (大部分与之前相同) ---
# ==============================================================================

class MazeDatasetGenerator:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.move_order = ['U', 'D', 'L', 'R']
        self.move_map = {move: i for i, move in enumerate(self.move_order)}

    def generate_maze(self):
        # ... 此函数无变化 ...
        while True:
            maze = [[0] * self.width for _ in range(self.height)]
            for r in range(self.height):
                for c in range(self.width):
                    if random.random() < WALL_DENSITY:
                        maze[r][c] = 1
            possible_starts = []
            for r in range(self.height):
                for c in range(self.width):
                    if maze[r][c]==0:
                        possible_starts.append((r, c))
            if len(possible_starts) < 2: continue
            start_pos, target_pos = random.sample(possible_starts, 2)
            q = deque([start_pos])
            visited = {start_pos}
            path_exists = False
            while q:
                r, c = q.popleft()
                if (r, c)==target_pos:
                    path_exists = True
                    break
                for dr, dc in self.moves.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width and \
                            maze[nr][nc]==0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            if path_exists:
                return maze, start_pos, target_pos

    def solve_with_bfs(self, maze, target_pos):
        # ... 此函数无变化 ...
        q = deque([(target_pos, 0)])
        distances = {target_pos: 0}
        while q:
            (r, c), dist = q.popleft()
            for dr, dc in self.moves.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and \
                        maze[nr][nc]==0 and (nr, nc) not in distances:
                    distances[(nr, nc)] = dist + 1
                    q.append(((nr, nc), dist + 1))
        return distances

    def generate_dataset(self):
        """【已修改】主函数，增加了显式的去重逻辑。"""
        print("=" * 60)
        print("迷宫最短路径数据集生成器 (带去重和可视化)")
        print(f"目标唯一样本数: {TARGET_NUM_SAMPLES}, 尺寸: {self.height}x{self.width}")
        print("=" * 60)

        all_data_points = []
        seen_inputs = set()  # 用于存储所有见过的输入字符串，实现去重
        mazes_generated = 0
        duplicates_skipped = 0

        with tqdm(total=TARGET_NUM_SAMPLES, desc="收集唯一样本") as pbar:
            while len(all_data_points) < TARGET_NUM_SAMPLES:
                mazes_generated += 1
                maze, _, target_pos = self.generate_maze()
                distances = self.solve_with_bfs(maze, target_pos)

                # 将该迷宫产生的所有潜在数据点随机化，增加多样性
                potential_points = list(distances.items())
                random.shuffle(potential_points)

                for pos, dist in potential_points:
                    if pos==target_pos: continue

                    optimal_moves = [0] * 4
                    current_dist = distances[pos]
                    for move_name, (dr, dc) in self.moves.items():
                        next_pos = (pos[0] + dr, pos[1] + dc)
                        if next_pos in distances and distances[next_pos]==current_dist - 1:
                            optimal_moves[self.move_map[move_name]] = 1

                    if sum(optimal_moves)==0: continue

                    temp_maze = [row[:] for row in maze]
                    temp_maze[pos[0]][pos[1]] = 's'
                    temp_maze[target_pos[0]][target_pos[1]] = 't'
                    input_str = "".join(map(str, [cell for row in temp_maze for cell in row]))

                    # 【核心去重逻辑】
                    if input_str not in seen_inputs:
                        seen_inputs.add(input_str)
                        all_data_points.append({"input": input_str, "output": optimal_moves})
                        pbar.update(1)  # 只有在找到新样本时才更新进度条
                    else:
                        duplicates_skipped += 1

                    # 实时更新进度条的后缀信息
                    pbar.set_postfix_str(f"迷宫: {mazes_generated}, 重复: {duplicates_skipped}")

                    if len(all_data_points) >= TARGET_NUM_SAMPLES:
                        break

                if len(all_data_points) >= TARGET_NUM_SAMPLES:
                    break

        print(f"\n生成了 {len(all_data_points)} 个唯一数据点。正在进行全局洗牌...")
        random.shuffle(all_data_points)

        print(f"正在写入文件: {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            for data_point in tqdm(all_data_points, desc="写入文件"):
                f.write(json.dumps(data_point) + '\n')

        print("\n🎉🎉🎉 数据集生成完毕！ 🎉🎉🎉")
        print(f"总共生成迷宫数: {mazes_generated}")
        print(f"期间跳过重复样本数: {duplicates_skipped} (这通常是一个非常小的数字)")

        return all_data_points


# --- 可视化函数 (无变化) ---
def visualize_samples(dataset, num_samples, height, width):
    if not dataset:
        print("\n数据集为空，无法进行可视化。")
        return
    print("\n" + "=" * 20 + " 数据集样本可视化 " + "=" * 20)
    VIZ_MAP = {'0': ' ', '1': '█', 's': 'S', 't': 'T'}
    samples_to_show = random.sample(dataset, min(num_samples, len(dataset)))
    for i, sample in enumerate(samples_to_show):
        print(f"\n--- 样本 {i + 1}/{len(samples_to_show)} ---")
        input_str = sample["input"]
        output_arr = sample["output"]
        print("Input (迷宫状态):")
        print("+" + "---" * width + "+")
        for r in range(height):
            print("|", end="")
            for c in range(width):
                char = input_str[r * width + c]
                viz_char = VIZ_MAP.get(char, '?')
                print(f" {viz_char} ", end="")
            print("|")
        print("+" + "---" * width + "+")
        output_str_parts = []
        move_order = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for i, move_val in enumerate(output_arr):
            if move_val==1:
                output_str_parts.append(move_order[i])
        print(f"Output (最优移动): {output_arr}  =>  {', '.join(output_str_parts)}")
    print("\n" + "=" * 58)


if __name__=='__main__':
    generator = MazeDatasetGenerator(height=MAZE_HEIGHT, width=MAZE_WIDTH)
    full_dataset = generator.generate_dataset()
    visualize_samples(
        dataset=full_dataset,
        num_samples=NUM_SAMPLES_TO_VISUALIZE,
        height=MAZE_HEIGHT,
        width=MAZE_WIDTH
    )
