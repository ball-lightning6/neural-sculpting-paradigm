import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 配置参数 (保持不变) ---
DATASET_DIR = "arc_final_projection_dataset"
GRID_DIM = 20
CELL_SIZE = 12
IMG_SIZE = GRID_DIM * CELL_SIZE
COLOR_BLACK = (0, 0, 0)
COLOR_PALETTE = [
    (255, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 165, 0), (128, 0, 128)
]
NUM_SAMPLES = 1000

# --- 颜色编码 ---
C_BLACK = 0
# 使用不同的编码范围来区分角色
ARROW_CODE_START = 1
DOT_CODE_START = 1 + len(COLOR_PALETTE)
PROJECTION_CODE_START = 1 + 2 * len(COLOR_PALETTE)


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def get_arrow_mask():
    arrow = np.zeros((4, 7), dtype=int)
    arrow[0, 3] = 1
    arrow[1, 2:5] = 1
    arrow[2, [1, 2, 4, 5]] = 1
    arrow[3, [0, 6]] = 1
    return arrow


def generate_logic_grids(arrow_mask):
    """
    【最终修正版核心算法】
    """
    dim = GRID_DIM
    grid = np.zeros((dim, dim), dtype=int)

    # 1. 随机选择颜色
    arrow_color_idx, dot_color_idx = random.sample(range(len(COLOR_PALETTE)), 2)
    arrow_code = ARROW_CODE_START + arrow_color_idx
    dot_code = DOT_CODE_START + dot_color_idx

    # 2. 随机放置箭头
    arrow_h, arrow_w = arrow_mask.shape
    r_arrow_start = random.randint(0, dim - arrow_h - 3)
    c_arrow_start = random.randint(0, dim - arrow_w)

    grid[r_arrow_start:r_arrow_start + arrow_h, c_arrow_start:c_arrow_start + arrow_w][arrow_mask==1] = arrow_code

    # 3. 随机散布散点
    num_dots = random.randint(dim, int(dim * 2.5))  # 增加散点密度以确保有足够的过滤目标
    for _ in range(num_dots):
        r, c = random.randint(0, dim - 1), random.randint(0, dim - 1)
        if grid[r, c]==C_BLACK:
            grid[r, c] = dot_code

    input_grid = grid.copy()
    output_grid = grid.copy()

    # 4. 【修正逻辑】执行核心的“局部投影”
    arrow_bottom_row = r_arrow_start + arrow_h
    arrow_left_col = c_arrow_start
    arrow_right_col = c_arrow_start + arrow_w

    arrow_down_area = [[r_arrow_start+2,c_arrow_start+3],
                       [r_arrow_start+3,c_arrow_start+1],
                       [r_arrow_start+3,c_arrow_start+2],
                       [r_arrow_start+3,c_arrow_start+3],
                       [r_arrow_start+3,c_arrow_start+4],
                       [r_arrow_start+3,c_arrow_start+5]]

    # --- Step A: 数据过滤 (第一个错误修正) ---
    # 找到所有 **同时满足行和列条件** 的散点
    dots_to_process = []
    all_dots_coords = np.argwhere(grid==dot_code)
    for r, c in all_dots_coords:
        if r >= arrow_bottom_row and arrow_left_col <= c < arrow_right_col or [r,c] in arrow_down_area:
            dots_to_process.append((r, c))

    # --- Step B: 信息提取 ---
    # 获取这些散点所在的列 (现在已经是局部列了)
    columns_to_draw = np.unique([c for r, c in dots_to_process])

    # --- Step C: 条件性投影 (第二个错误修正) ---
    if columns_to_draw.size > 0:
        projection_color_code = PROJECTION_CODE_START + dot_color_idx
        for c in columns_to_draw:
            # 只在箭头下方区域进行涂色
            for r in range(arrow_bottom_row, dim):
                # 只对黑色背景进行涂色
                if output_grid[r, c]==C_BLACK:
                    output_grid[r, c] = projection_color_code
        for r,c in arrow_down_area:
            if c in columns_to_draw.tolist():
                output_grid[r, c] = projection_color_code


    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()

    color_map = {C_BLACK: COLOR_BLACK}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[ARROW_CODE_START + i] = color
        color_map[DOT_CODE_START + i] = color
        color_map[PROJECTION_CODE_START + i] = color

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(args):
    """封装的单个任务，用于多进程"""
    sample_index, arrow_mask = args
    input_logic_grid, output_logic_grid = generate_logic_grids(arrow_mask)
    input_image = grid_to_image(input_logic_grid, CELL_SIZE)
    output_image = grid_to_image(output_logic_grid, CELL_SIZE)

    filename = f"{sample_index:06d}.png"
    input_path = os.path.join(DATASET_DIR, "input", filename)
    output_path = os.path.join(DATASET_DIR, "output", filename)
    input_image.save(input_path)
    output_image.save(output_path)
    return True


# --- 主程序入口 ---
if __name__=="__main__":
    create_directories()
    arrow_mask = get_arrow_mask()
    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (最终修正版)...")
    args_list = [(i, arrow_mask) for i in range(NUM_SAMPLES)]

    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, args_list), total=NUM_SAMPLES))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个局部坐标系投影图像对已保存到 '{DATASET_DIR}' 目录。")