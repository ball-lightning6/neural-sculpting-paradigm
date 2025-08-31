import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 配置参数 ---
# 移到主函数外部，方便多进程访问
DATASET_DIR = "arc_final_dataset_mp"
GRID_DIM = 14
CELL_SIZE = 16
IMG_SIZE = GRID_DIM * CELL_SIZE
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (255, 255, 0)

C_BLACK = 0
C_GREEN = 1
C_YELLOW = 2

NOISE_WALL_RATIO = 0.1
WALL_RATIO_MIN = 0.2
WALL_RATIO_MAX = 0.4


# --- 核心算法函数 (保持不变) ---

def get_neighbors(r, c, dim):
    neighbors = []
    if r > 0: neighbors.append((r - 1, c))
    if r < dim - 1: neighbors.append((r + 1, c))
    if c > 0: neighbors.append((r, c - 1))
    if c < dim - 1: neighbors.append((r, c + 1))
    return neighbors


def generate_yellow_region(dim):
    grid = np.zeros((dim, dim), dtype=int)
    num_yellow_cells = random.randint(dim, dim * dim // 3)
    r_start, c_start = random.randint(1, dim - 2), random.randint(1, dim - 2)
    q = [(r_start, c_start)]
    grid[r_start, c_start] = C_YELLOW
    count = 1
    while q and count < num_yellow_cells:
        r, c = q.pop(0)
        neighbors = get_neighbors(r, c, dim)
        random.shuffle(neighbors)
        for nr, nc in neighbors:
            if grid[nr, nc]==C_BLACK and random.random() < 0.7:
                if 1 <= nr < dim - 1 and 1 <= nc < dim - 1:
                    grid[nr, nc] = C_YELLOW
                    q.append((nr, nc))
                    count += 1
                    if count >= num_yellow_cells: break
    return grid


def build_minimal_walls(grid):
    dim = grid.shape[0]
    wall_grid = grid.copy()
    yellow_coords = np.argwhere(grid==C_YELLOW)
    for r_y, c_y in yellow_coords:
        for nr, nc in get_neighbors(r_y, c_y, dim):
            if wall_grid[nr, nc]==C_BLACK:
                wall_grid[nr, nc] = C_GREEN
    return wall_grid


def add_safe_noise_walls(grid, noise_ratio):
    dim = grid.shape[0]
    outside_air = np.zeros_like(grid, dtype=bool)
    q = []
    for i in range(dim):
        if grid[0, i]==C_BLACK: q.append((0, i))
        if grid[dim - 1, i]==C_BLACK: q.append((dim - 1, i))
        if grid[i, 0]==C_BLACK: q.append((i, 0))
        if grid[i, dim - 1]==C_BLACK: q.append((i, dim - 1))

    visited = set(q)
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        outside_air[r, c] = True
        for nr, nc in get_neighbors(r, c, dim):
            if (nr, nc) not in visited and grid[nr, nc]==C_BLACK:
                visited.add((nr, nc))
                q.append((nr, nc))

    background_mask = (grid==C_BLACK)
    safe_noise_area = background_mask & outside_air

    num_noise_walls = int(noise_ratio * dim * dim)
    safe_coords = np.argwhere(safe_noise_area).tolist()
    if not safe_coords: return grid

    for _ in range(num_noise_walls):
        if not safe_coords: break
        idx = random.randrange(len(safe_coords))
        r, c = safe_coords.pop(idx)
        grid[r, c] = C_GREEN

    return grid


def grid_to_image(grid, cell_size, color_map):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


# --- 新增的、用于多进程的单个任务封装函数 ---
def generate_single_sample(sample_index):
    """
    生成并保存单个样本，此函数将被并行调用。
    """
    # 1. 生成黄色区域
    yellow_grid = generate_yellow_region(GRID_DIM)

    # 2. 建造最小围墙
    walled_grid = build_minimal_walls(yellow_grid)

    # 3. 智能地添加噪音墙
    final_grid = add_safe_noise_walls(walled_grid.copy(), NOISE_WALL_RATIO)
    final_grid[yellow_grid==C_YELLOW] = C_YELLOW  # 恢复黄色区域

    # 4. 创建输入和输出图像
    input_logic_grid = final_grid.copy()
    input_logic_grid[input_logic_grid==C_YELLOW] = C_BLACK

    input_image = grid_to_image(input_logic_grid, CELL_SIZE, {C_BLACK: COLOR_BLACK, C_GREEN: COLOR_GREEN})
    output_image = grid_to_image(final_grid, CELL_SIZE,
        {C_BLACK: COLOR_BLACK, C_GREEN: COLOR_GREEN, C_YELLOW: COLOR_YELLOW})

    # 5. 保存
    filename = f"{sample_index:06d}.png"
    input_path = os.path.join(DATASET_DIR, "input", filename)
    output_path = os.path.join(DATASET_DIR, "output", filename)
    input_image.save(input_path)
    output_image.save(output_path)

    # 这个返回值是可选的，主要为了tqdm能计数
    return True


# --- 主程序入口 (修改为使用多进程) ---
if __name__=="__main__":
    # 创建目录
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")

    # 要生成的样本总数
    NUM_SAMPLES = 1000  # 我们可以自信地增加数量了！

    # 设置进程数，通常设置为CPU核心数或稍少一些
    # os.cpu_count() 会自动获取你的CPU核心数
    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本...")

    # 使用进程池来并行执行任务
    with mp.Pool(processes=num_processes) as pool:
        # 使用imap_unordered可以让我们在任务完成时立即更新进度条，体验更好
        # list(...) 在这里是为了确保所有任务都完成后再结束tqdm
        # total=NUM_SAMPLES 告诉tqdm总共有多少个任务
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个高保真图像对已保存到 '{DATASET_DIR}' 目录。")