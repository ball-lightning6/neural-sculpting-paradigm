import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 ---
DATASET_DIR = "arc_priority_final_dataset_v2"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_BLACK = (0, 0, 0)
COLOR_PALETTE = [
    (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0),
    (255, 165, 0), (128, 0, 128), (0, 255, 255), (240, 128, 128)
]
NUM_SAMPLES = 1000

C_BLACK = 0


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def check_overlap(r1, c1, h1, w1, rects):
    # 这个函数在最终逻辑中不再需要，因为我们直接检查单元格占用
    pass


def generate_logic_grids():
    """【已修正】核心算法"""
    dim = GRID_DIM

    num_horizontal = random.randint(2, dim // 2 - 2)
    num_vertical = random.randint(2, dim // 2 - 2)
    num_total_pairs = num_horizontal + num_vertical

    # 确保颜色足够用
    if num_total_pairs > len(COLOR_PALETTE):
        num_total_pairs = len(COLOR_PALETTE)
        # 按比例重新分配 horizontal 和 vertical 的数量
        ratio = num_horizontal / (num_horizontal + num_vertical)
        num_horizontal = int(num_total_pairs * ratio)
        num_vertical = num_total_pairs - num_horizontal

    used_colors_indices = random.sample(range(len(COLOR_PALETTE)), num_total_pairs)

    occupied_cells = set()
    horizontal_pairs = []
    vertical_pairs = []

    color_counter = 0

    # 生成横向对
    available_rows = list(range(dim))
    random.shuffle(available_rows)
    for r in available_rows:
        if len(horizontal_pairs) >= num_horizontal: break

        c1 = random.randint(0, dim - 3)
        c2 = random.randint(c1 + 2, dim - 1)
        if (r, c1) not in occupied_cells and (r, c2) not in occupied_cells:
            occupied_cells.add((r, c1))
            occupied_cells.add((r, c2))
            horizontal_pairs.append({'r': r, 'c1': c1, 'c2': c2, 'color_idx': used_colors_indices[color_counter]})
            color_counter += 1

    # 生成纵向对
    available_cols = list(range(dim))
    random.shuffle(available_cols)

    # 获取横向对的端点行和列，用于精确的粘连检查
    h_endpoints = {(p['r'], p['c1']) for p in horizontal_pairs} | {(p['r'], p['c2']) for p in horizontal_pairs}

    for c in available_cols:
        if len(vertical_pairs) >= num_vertical: break

        r1 = random.randint(0, dim - 3)
        r2 = random.randint(r1 + 2, dim - 1)

        p1 = (r1, c)
        p2 = (r2, c)

        # 检查是否与任何点重叠
        if p1 in occupied_cells or p2 in occupied_cells: continue
        # 检查是否与横向对的端点粘连
        if p1 in h_endpoints or p2 in h_endpoints: continue

        occupied_cells.add(p1)
        occupied_cells.add(p2)
        vertical_pairs.append({'c': c, 'r1': r1, 'r2': r2, 'color_idx': used_colors_indices[color_counter]})
        color_counter += 1

    # ---- 绘制网格 ----
    input_logic_grid = np.zeros((dim, dim), dtype=int)
    # 使用 color_idx + 1 作为颜色代码
    for pair in horizontal_pairs:
        input_logic_grid[pair['r'], pair['c1']] = pair['color_idx'] + 1
        input_logic_grid[pair['r'], pair['c2']] = pair['color_idx'] + 1
    for pair in vertical_pairs:
        input_logic_grid[pair['r1'], pair['c']] = pair['color_idx'] + 1
        input_logic_grid[pair['r2'], pair['c']] = pair['color_idx'] + 1

    output_logic_grid = input_logic_grid.copy()

    # 画横线
    for pair in horizontal_pairs:
        output_logic_grid[pair['r'], pair['c1'] + 1:pair['c2']] = pair['color_idx'] + 1

    # 画竖线 (覆盖)
    for pair in vertical_pairs:
        output_logic_grid[pair['r1'] + 1:pair['r2'], pair['c']] = pair['color_idx'] + 1

    return input_logic_grid, output_logic_grid


# --- 后续函数保持不变 ---
def grid_to_image(grid, cell_size):
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
        input_grid, output_grid = generate_logic_grids()
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (已修复最终版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终版图像对已保存到 '{DATASET_DIR}' 目录。")