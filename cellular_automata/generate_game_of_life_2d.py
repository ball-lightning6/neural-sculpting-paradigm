import random
import json

def neighbors(grid, i, j, n):
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == dy == 0:
                continue
            ni, nj = i + dx, j + dy
            if 0 <= ni < n and 0 <= nj < n:
                count += grid[ni][nj]
    return count

def next_state(grid):
    n = len(grid)
    new_grid = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cnt = neighbors(grid, i, j, n)
            if grid[i][j] == 1 and cnt in (2, 3):
                new_grid[i][j] = 1
            elif grid[i][j] == 0 and cnt == 3:
                new_grid[i][j] = 1
            else:
                new_grid[i][j] = 0
    return new_grid

def grid_to_str(grid):
    return ''.join(str(cell) for row in grid for cell in row)
from tqdm import tqdm

def generate_dataset_jsonl(num_samples=1000, n=5, d=1):
    filename = f'game_of_life_dataset_{n}_{n}_layer{d}.jsonl'
    with open(filename, 'w') as f:
        for _ in tqdm(range(num_samples)):
            grid = [[random.randint(0, 1) for _ in range(n)] for _ in range(n)]
            next_grid = grid
            for _ in range(d):
                next_grid = next_state(grid)
            input_str = grid_to_str(grid)
            output_label = list(map(int,grid_to_str(next_grid)))
            json_line = json.dumps({"input": input_str, "output": output_label})
            f.write(json_line + '\n')

# 示例用法
if __name__ == "__main__":
    generate_dataset_jsonl(num_samples=1000000, n=32, d =1)
    print("✅ Done. JSONL dataset saved.")
