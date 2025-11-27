import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 (保持不变) ---
DATASET_DIR = "arc_program_final_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (128, 0, 128), (255, 165, 0), (192, 192, 192)
]
NUM_SAMPLES = 1000

C_BLACK = 0
C_BLUE = 1


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_logic_grids():
    """【最终王者版核心算法】"""
    dim = GRID_DIM
    input_grid = np.zeros((dim, dim), dtype=int)

    color_a_idx, color_b_idx = random.sample(range(len(COLOR_PALETTE)), 2)
    color_a_code = color_a_idx + 2
    color_b_code = color_b_idx + 2

    input_grid[0, 0] = color_a_code
    input_grid[0, 1] = color_b_code

    while True:
        r_start = random.randint(0, dim - 1)
        c_start = random.randint(0, dim - 1)
        if (r_start, c_start)!=(0, 0) and (r_start, c_start)!=(0, 1):
            input_grid[r_start, c_start] = C_BLUE
            break

    output_grid = np.zeros((dim, dim), dtype=int)
    output_grid[r_start, c_start] = C_BLUE

    r_curr, c_curr = r_start, c_start

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 左, 下, 右, 上
    colors = [color_a_code, color_b_code, color_a_code, color_b_code]
    current_length = 2

    # --- 【核心修正】 while循环实现运行时截断 ---
    while True:
        direction_idx = (current_length - 2) % 4
        dr, dc = directions[direction_idx]
        color_code = colors[direction_idx]

        # 尝试绘制当前长度的线段
        final_step = 0
        for step in range(1, current_length + 1):
            next_r = r_curr + dr * step
            next_c = c_curr + dc * step

            # 如果碰壁，记录下最后一步，并跳出内层循环
            if not (0 <= next_r < dim and 0 <= next_c < dim):
                final_step = step - 1
                break

            # 如果没碰壁，就正常画点
            output_grid[next_r, next_c] = color_code
            final_step = step

        # 更新画笔位置到实际画到的最后一点
        r_curr += dr * final_step
        c_curr += dc * final_step

        # 如果实际画的步数小于期望的步数，说明碰壁了，终止外层循环
        if final_step < current_length:
            break

        current_length += 1

    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()

    color_map = {C_BLACK: COLOR_BLACK, C_BLUE: COLOR_BLUE}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 2] = color

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    """【最终版】生成并保存单个样本"""
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (终极王者版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个终极版图像对已保存到 '{DATASET_DIR}' 目录。")