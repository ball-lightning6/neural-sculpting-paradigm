# generate_maze_decoupled.py

import json
import random
from collections import deque
from tqdm import tqdm
import math

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================
TARGET_NUM_SAMPLES = 500000
OUTPUT_FILE = "maze_decoupled_9_9_dataset.jsonl"
MAZE_HEIGHT = 11
MAZE_WIDTH = 11

# ==============================================================================
# --- 核心代码 ---
# ==============================================================================

class DecoupledMazeGenerator:
    def __init__(self, height, width):
        if height % 2 == 0 or width % 2 == 0:
            print(f"警告: 迷宫尺寸 ({height}x{width}) 最好是奇数。")
        self.height = height
        self.width = width
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.move_order = ['U', 'D', 'L', 'R']
        self.move_map = {move: i for i, move in enumerate(self.move_order)}
        
        # 内部迷宫的尺寸和单元格总数
        self.inner_height = height - 2
        self.inner_width = width - 2
        self.inner_cells = self.inner_height * self.inner_width
        
        # 计算表示距离值需要的位数
        max_possible_dist = self.inner_cells
        self.bits_per_distance = math.ceil(math.log2(max_possible_dist + 1))

    def generate_maze(self):
        """生成稠密迷宫，返回迷宫、起点和终点"""
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
        path_cells = [ (r, c) for r in range(self.height) for c in range(self.width) if maze[r][c]==0 ]
        if len(path_cells) < 2: return self.generate_maze()
        start_pos, target_pos = random.sample(path_cells, 2)
        return maze, start_pos, target_pos

    def solve_with_bfs(self, maze, target_pos):
        """使用BFS从终点反向计算所有点到终点的距离"""
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

    def generate_dataset_sample(self):
        """生成单个包含所有标签的样本"""
        maze, start_pos, target_pos = self.generate_maze()
        distances = self.solve_with_bfs(maze, target_pos)

        # 确保起点是可达的
        if start_pos not in distances:
            return None

        # --- 1. 生成最终答案标签 (Prediction Label) ---
        optimal_move = -1
        start_dist = distances[start_pos]
        for move_name, (dr, dc) in self.moves.items():
            next_pos = (start_pos[0] + dr, start_pos[1] + dc)
            if next_pos in distances and distances[next_pos] < start_dist:
                optimal_move = self.move_map[move_name]
                break
        
        # 如果起点就在终点旁边，可能没有更优的移动了，这种情况我们跳过
        if optimal_move == -1:
            return None
            
        output_prediction = optimal_move # 这是一个整数 (0, 1, 2, or 3)

        # --- 2. 生成输入字符串 ---
        maze_with_s_t = [row[:] for row in maze]
        maze_with_s_t[start_pos[0]][start_pos[1]] = 's'
        maze_with_s_t[target_pos[0]][target_pos[1]] = 't'
        inner_maze = [row[1:-1] for row in maze_with_s_t[1:-1]]
        input_str = "".join(map(str, [cell for row in inner_maze for cell in row]))

        # --- 3. 生成解耦标签 (Explain Label) - 完整BFS距离图 ---
        distance_map_flat = []
        for r in range(self.inner_height):
            for c in range(self.inner_width):
                # 将内部坐标映射回完整迷宫坐标
                full_r, full_c = r + 1, c + 1
                dist = distances.get((full_r, full_c), -1) # 如果是墙或不可达，距离为-1
                
                # +1 是为了让 -1 变成 0，所有真实距离都 > 0
                dist_to_encode = dist + 1
                dist_bin_str = format(dist_to_encode, f'0{self.bits_per_distance}b')
                distance_map_flat.extend([int(b) for b in dist_bin_str])
        
        output_explanation = distance_map_flat

        return {
            "input": input_str,
            "prediction_label": output_prediction,
            "explanation_label": output_explanation
        }

    def generate_and_save(self):
        print("=" * 60)
        print("稠密迷宫解耦实验 - 数据集生成器")
        print(f"目标唯一样本数: {TARGET_NUM_SAMPLES}")
        print(f"解耦格式: 完整BFS距离图 ({self.bits_per_distance} bits/cell)")
        print("=" * 60)

        all_data_points, seen_inputs = [], set()
        with tqdm(total=TARGET_NUM_SAMPLES, desc="生成唯一样本") as pbar:
            while len(all_data_points) < TARGET_NUM_SAMPLES:
                sample = self.generate_dataset_sample()
                if sample and sample["input"] not in seen_inputs:
                    seen_inputs.add(sample["input"])
                    all_data_points.append(sample)
                    pbar.update(1)

        print(f"\n生成了 {len(all_data_points)} 个唯一数据点。正在写入文件: {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            for data_point in tqdm(all_data_points, desc="写入文件"):
                f.write(json.dumps(data_point) + '\n')
        print("\n🎉 数据集生成完毕！ 🎉")

if __name__ == '__main__':
    generator = DecoupledMazeGenerator(height=MAZE_HEIGHT, width=MAZE_WIDTH)
    generator.generate_and_save()
