import json
import random
import heapq
from collections import deque
from tqdm import tqdm
import sys
import time

# ==============================================================================
# ===                      参数配置区域 (直接在此修改)                      ===
# ==============================================================================

# --- 数据集设置 ---
NUM_SAMPLES = 500000
OUTPUT_FILE = "sokoban_dataset_shuffled_8.jsonl"

# --- 谜题几何设置 ---
M_DIMENSION = 10
N_DIMENSION = 10

# --- 难度控制设置 ---
MIN_DIFFICULTY = 12
MAX_DIFFICULTY = 48


# ==============================================================================


class SokobanDatasetGenerator:
    """
    最终版推箱子数据集生成器 v2.0。
    功能: 全局洗牌, 实时去重, 边界去除, 难度过滤。
    """

    def __init__(self, m, n):
        if m < 5 or n < 5:
            raise ValueError("迷宫尺寸(m, n)必须至少为 5x5 以确保可玩性。")
        self.m = m
        self.n = n
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.move_order = ['U', 'D', 'L', 'R']
        self.move_map = {move: i for i, move in enumerate(self.move_order)}

    # ... (_generate_puzzle, _solve_sokoban, _state_to_string_optimized 方法与之前相同) ...
    def _generate_puzzle(self, complexity=30):
        while True:
            board = [['.' for _ in range(self.n)] for _ in range(self.m)]
            for i in range(self.m):
                board[i][0] = '#'
                board[i][self.n - 1] = '#'
            for j in range(self.n):
                board[0][j] = '#'
                board[self.m - 1][j] = '#'
            for _ in range(int(self.m * self.n * 0.15)):
                row, col = random.randint(1, self.m - 2), random.randint(1, self.n - 2)
                board[row][col] = '#'
            goal_pos = (random.randint(1, self.m - 2), random.randint(1, self.n - 2))
            if board[goal_pos[0]][goal_pos[1]]=='#': continue
            board[goal_pos[0]][goal_pos[1]] = '*'
            box_pos = goal_pos
            possible_player_starts = []
            for dr, dc in self.moves.values():
                pr, pc = box_pos[0] + dr, box_pos[1] + dc
                if 1 <= pr < self.m - 1 and 1 <= pc < self.n - 1 and board[pr][pc]!='#':
                    possible_player_starts.append((pr, pc))
            if not possible_player_starts: continue
            player_pos = random.choice(possible_player_starts)
            state_history = set([(player_pos, box_pos)])
            for _ in range(complexity):
                move_dir = random.choice(list(self.moves.values()))
                prev_box_pos = player_pos
                prev_player_pos = (player_pos[0] - move_dir[0], player_pos[1] - move_dir[1])
                new_player_pos = (prev_player_pos[0] - move_dir[0], prev_player_pos[1] - move_dir[1])
                if not (1 <= new_player_pos[0] < self.m - 1 and 1 <= new_player_pos[1] < self.n - 1): continue
                if board[new_player_pos[0]][new_player_pos[1]]=='#' or board[prev_player_pos[0]][
                    prev_player_pos[1]]=='#': continue
                new_state = (new_player_pos, prev_box_pos)
                if new_state in state_history: continue
                player_pos, box_pos = new_player_pos, prev_box_pos
                state_history.add(new_state)
            if box_pos==goal_pos: continue
            final_board = [row[:] for row in board]
            final_board[player_pos[0]][player_pos[1]] = 's'
            final_board[box_pos[0]][box_pos[1]] = 'b'
            return final_board, (player_pos, box_pos), goal_pos

    def _solve_sokoban(self, board, initial_state, goal_pos):
        player_pos, box_pos = initial_state
        initial_state_tuple = (player_pos, box_pos)

        def heuristic(b_pos):
            return abs(b_pos[0] - goal_pos[0]) + abs(b_pos[1] - goal_pos[1])

        frontier = [(heuristic(box_pos), 0, initial_state_tuple)]
        cost_so_far = {initial_state_tuple: 0}
        while frontier:
            _, g_score, current_state = heapq.heappop(frontier)
            current_player_pos, current_box_pos = current_state
            if current_box_pos==goal_pos:
                return cost_so_far
            for move_key, (dr, dc) in self.moves.items():
                next_player_pos = (current_player_pos[0] + dr, current_player_pos[1] + dc)
                if not (0 <= next_player_pos[0] < self.m and 0 <= next_player_pos[1] < self.n) or \
                        board[next_player_pos[0]][next_player_pos[1]]=='#': continue
                next_box_pos = current_box_pos
                if next_player_pos==current_box_pos:
                    next_box_pos = (current_box_pos[0] + dr, current_box_pos[1] + dc)
                    if not (0 <= next_box_pos[0] < self.m and 0 <= next_box_pos[1] < self.n) or \
                            board[next_box_pos[0]][next_box_pos[1]]=='#': continue
                new_state = (next_player_pos, next_box_pos)
                new_g_score = g_score + 1
                if new_state not in cost_so_far or new_g_score < cost_so_far[new_state]:
                    cost_so_far[new_state] = new_g_score
                    f_score = new_g_score + heuristic(next_box_pos)
                    heapq.heappush(frontier, (f_score, new_g_score, new_state))
        return None

    def _state_to_string_optimized(self, board_template, state):
        player_pos, box_pos = state
        temp_board = [row[:] for row in board_template]
        if temp_board[box_pos[0]][box_pos[1]]=='*':
            temp_board[box_pos[0]][box_pos[1]] = 'B'
        else:
            temp_board[box_pos[0]][box_pos[1]] = 'b'
        if temp_board[player_pos[0]][player_pos[1]]=='*':
            temp_board[player_pos[0]][player_pos[1]] = 'S'
        else:
            temp_board[player_pos[0]][player_pos[1]] = 's'
        inner_board = [row[1:-1] for row in temp_board[1:-1]]
        flat_list = [char.replace('B', 'b').replace('S', 's') for row in inner_board for char in row]
        return "".join(flat_list)

    def generate_dataset(self, num_samples, output_file, min_difficulty, max_difficulty):
        """【最终版 v2.0】主函数，集成全局洗牌。"""
        start_time = time.time()
        print("=" * 60)
        print("Sokoban 高质量数据集生成器 v2.0 (集成全局洗牌)")
        print(f"目标样本数: {num_samples}")
        print(f"谜题尺寸 (内部): {(self.m - 2)}x{(self.n - 2)}, 难度范围: [{min_difficulty}, {max_difficulty}] 步")
        print("=" * 60)

        # 步骤 1: 在内存中收集所有数据
        print("\n--- 步骤 1/3: 正在生成并收集唯一的样本... ---")
        all_data = []
        seen_inputs = set()
        puzzles_processed = 0
        duplicates_skipped = 0

        with tqdm(total=num_samples, desc="收集样本") as pbar:
            while len(all_data) < num_samples:
                puzzles_processed += 1
                complexity = random.randint(min_difficulty, max_difficulty * 2)
                board_template, initial_state, goal_pos = self._generate_puzzle(complexity=complexity)
                cost_so_far = self._solve_sokoban(board_template, initial_state, goal_pos)
                if not cost_so_far: continue

                final_states_costs = [g for state, g in cost_so_far.items() if state[1]==goal_pos]
                if not final_states_costs: continue
                solution_length = min(final_states_costs)
                if not (min_difficulty <= solution_length <= max_difficulty): continue

                potential_datapoints = list(cost_so_far.items())
                random.shuffle(potential_datapoints)

                for state, g_score in potential_datapoints:
                    if state[1]==goal_pos: continue
                    optimal_moves = [0] * 4
                    is_valid_datapoint = False
                    for move_key, (dr, dc) in self.moves.items():
                        next_player_pos = (state[0][0] + dr, state[0][1] + dc)
                        if not (0 <= next_player_pos[0] < self.m and 0 <= next_player_pos[1] < self.n) or \
                                board_template[next_player_pos[0]][next_player_pos[1]]=='#': continue
                        next_box_pos = state[1]
                        if next_player_pos==state[1]:
                            next_box_pos = (state[1][0] + dr, state[1][1] + dc)
                            if not (0 <= next_box_pos[0] < self.m and 0 <= next_box_pos[1] < self.n) or \
                                    board_template[next_box_pos[0]][next_box_pos[1]]=='#': continue
                        next_state = (next_player_pos, next_box_pos)
                        if next_state in cost_so_far and cost_so_far[next_state]==g_score + 1:
                            optimal_moves[self.move_map[move_key]] = 1
                            is_valid_datapoint = True

                    if not is_valid_datapoint: continue

                    input_str = self._state_to_string_optimized(board_template, state)
                    if input_str in seen_inputs:
                        duplicates_skipped += 1
                        continue

                    seen_inputs.add(input_str)
                    data_entry = {"input": input_str, "output": optimal_moves}
                    all_data.append(data_entry)
                    pbar.update(1)
                    pbar.set_postfix_str(f"谜题: {puzzles_processed}, 重复: {duplicates_skipped}")

                    if len(all_data) >= num_samples: break
                if len(all_data) >= num_samples: break

        # 步骤 2: 全局洗牌
        print("\n--- 步骤 2/3: 正在对所有样本进行全局随机洗牌... ---")
        random.shuffle(all_data)
        print("洗牌完成！")

        # 步骤 3: 写入文件
        print(f"\n--- 步骤 3/3: 正在将 {len(all_data)} 个已洗牌的样本写入文件... ---")
        with open(output_file, 'w') as f:
            for data_entry in tqdm(all_data, desc="写入文件"):
                f.write(json.dumps(data_entry) + '\n')

        end_time = time.time()
        duration = end_time - start_time

        print("\n\n🎉🎉🎉 全部流程完成！ 🎉🎉🎉")
        print("=" * 60)
        print(f"最终生成纯净、已洗牌的样本数: {len(all_data)}")
        print(f"高质量数据集已保存至: {OUTPUT_FILE}")
        print("------------------------------------------------------------")
        print(f"总共处理谜题数: {puzzles_processed}")
        print(f"期间跳过重复数: {duplicates_skipped}")
        print(f"总耗时: {duration:.2f} 秒")
        print("=" * 60)


if __name__=='__main__':
    generator = SokobanDatasetGenerator(M_DIMENSION, N_DIMENSION)
    generator.generate_dataset(
        num_samples=NUM_SAMPLES,
        output_file=OUTPUT_FILE,
        min_difficulty=MIN_DIFFICULTY,
        max_difficulty=MAX_DIFFICULTY
    )
