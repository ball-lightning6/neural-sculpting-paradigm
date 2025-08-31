import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp
from collections import deque

### 1. 配置参数 ###
DATASET_DIR = "arc_ultimate_graduation_final_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_BLACK = (0, 0, 0)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (128, 0, 128), (255, 165, 0), (0, 255, 255), (192, 192, 192),
    (240, 128, 128), (255, 0, 255)
]
NUM_SAMPLES = 1000

C_BLACK = 0


### 2. 辅助函数 ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def get_regions(grid, bg_color_code):
    dim = grid.shape[0]
    visited = (grid!=bg_color_code)
    regions = []

    for r in range(dim):
        for c in range(dim):
            if not visited[r, c]:
                region = set()  # Use a set for faster lookups
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    curr_r, curr_c = q.popleft()
                    region.add((curr_r, curr_c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < dim and 0 <= nc < dim and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                if region and len(region)>3:
                    regions.append(list(region))  # Convert back to list
    return regions


### 3. 核心生成逻辑 ###
def generate_logic_grids():
    for _ in range(100):
        dim = GRID_DIM
        grid = np.zeros((dim, dim), dtype=int)

        avail_colors = list(range(1, len(COLOR_PALETTE) + 1))
        random.shuffle(avail_colors)

        bg_color_code = avail_colors.pop()
        divider_color_code = avail_colors.pop()
        grid[:, :] = bg_color_code

        num_lines = random.randint(2, 4)
        for _ in range(num_lines):
            if random.random() < 0.5 and num_lines > 1:
                r1, c1, r2, c2 = [random.randint(1, dim - 2) for _ in range(4)]
                grid[r1, min(c1, c2):max(c1, c2) + 1] = divider_color_code
                grid[min(r1, r2):max(r1, r2) + 1, c2] = divider_color_code
            else:
                if random.random() < 0.5:
                    c = random.randint(1, dim - 2)
                    grid[:, c] = divider_color_code
                else:
                    r = random.randint(1, dim - 2)
                    grid[r, :] = divider_color_code

        regions = get_regions(grid, bg_color_code)
        if not regions or len(regions) > 6 or len(regions) < 2: continue

        output_grid = grid.copy()

        # 确保有足够的颜色用于指令
        if len(avail_colors) < len(regions) * 2: continue

        instruction_colors_pool = list(avail_colors)
        random.shuffle(instruction_colors_pool)

        for region_set in regions:
            if len(region_set) < 4: continue

            c1_code = instruction_colors_pool.pop()

            corners = [
                min(region_set, key=lambda p: (p[0], p[1])), min(region_set, key=lambda p: (p[0], -p[1])),
                min(region_set, key=lambda p: (-p[0], p[1])), min(region_set, key=lambda p: (-p[0], -p[1]))]
            corner_r, corner_c = random.choice(corners)

            grid[corner_r, corner_c] = c1_code

            has_c2 = random.choice([True, False])

            if has_c2 and instruction_colors_pool:
                c2_code = instruction_colors_pool.pop()
                dr = 1 if corner_r < dim // 2 else -1
                dc = 1 if corner_c < dim // 2 else -1
                if (corner_r + dr, corner_c + dc) in region_set:
                    grid[corner_r + dr, corner_c + dc] = c2_code

                    # --- 【核心修正】洋葱式填充，使用8邻接判断边界 ---
                    dist_grid = np.full((dim, dim), -1)
                    q = deque()

                    for r_p, c_p in region_set:
                        is_boundary = False
                        # 使用8邻接来判断是否是边界点
                        for dr_n, dc_n in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            if (r_p + dr_n, c_p + dc_n) not in region_set:
                                is_boundary = True
                                break
                        if is_boundary:
                            q.append(((r_p, c_p), 0))
                            dist_grid[r_p, c_p] = 0

                    head = 0
                    while head < len(q):
                        (r, c), d = q[head];
                        head += 1
                        # BFS也用8邻接来传播距离
                        for dr_n, dc_n in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            nr, nc = r + dr_n, c + dc_n
                            if (nr, nc) in region_set and dist_grid[nr, nc]==-1:
                                dist_grid[nr, nc] = d + 1
                                q.append(((nr, nc), d + 1))

                    for r_p, c_p in region_set:
                        dist = dist_grid[r_p, c_p]
                        if dist!=-1:
                            if dist % 2==0:  # 距离0, 2, 4... -> 颜色1
                                output_grid[r_p, c_p] = c1_code
                            else:  # 距离1, 3, 5... -> 颜色2
                                output_grid[r_p, c_p] = c2_code
                else:
                    has_c2 = False

            if not has_c2:
                for r_p, c_p in region_set:
                    output_grid[r_p, c_p] = c1_code

        return grid, output_grid

    return None


def grid_to_image(grid, cell_size):
    # ... (此函数无需修改)
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {C_BLACK: COLOR_BLACK}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 1] = color

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], C_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    try:
        result = None
        while result is None:
            result = generate_logic_grids()
        input_grid, output_grid = result

        input_image = grid_to_image(input_grid, CELL_SIZE)
        output_image = grid_to_image(output_grid, CELL_SIZE)

        filename = f"{sample_index:06d}.png"
        input_path = os.path.join(DATASET_DIR, "input", filename)
        output_path = os.path.join(DATASET_DIR, "output", filename)
        input_image.save(input_path)
        output_image.save(output_path)
        return True
    except Exception as e:
        print(f"在生成样本 {sample_index} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__=="__main__":
    create_directories()
    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (毕业设计最终修正版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终版图像对已保存到 '{DATASET_DIR}' 目录。")