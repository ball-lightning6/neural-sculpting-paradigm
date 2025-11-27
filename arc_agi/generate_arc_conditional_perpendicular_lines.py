import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 配置参数 ---
DATASET_DIR = "arc_conditional_projection_dataset_v2"
GRID_DIM = 14
CELL_SIZE = 16
IMG_SIZE = GRID_DIM * CELL_SIZE  # 14 * 16 = 224

# 颜色定义 (R, G, B)
COLOR_BLACK = (0, 0, 0)
COLOR_GREY = (128, 128, 128)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)
NUM_SAMPLES = 1000

# --- 颜色编码 ---
C_BLACK = 0
C_GREY = 1
C_BLUE = 2
C_RED = 3


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_logic_grids():
    """
    核心算法：生成输入和输出的逻辑网格
    """
    dim = GRID_DIM

    # 1. 随机确定灰色横线的位置
    grey_row_index = random.randint(1, dim - 2)

    # 2. 生成基础网格（全黑+灰色线）
    grid = np.zeros((dim, dim), dtype=int)
    grid[grey_row_index, :] = C_GREY

    # 3. 放置红色和蓝色散点
    upper_cols = list(range(dim))
    lower_cols = list(range(dim))
    random.shuffle(upper_cols)
    random.shuffle(lower_cols)

    num_upper_dots = random.randint(dim // 4, dim - 2)
    num_lower_dots = random.randint(dim // 4, dim - 2)

    # 放置上方散点
    for i in range(num_upper_dots):
        c = upper_cols[i]
        r = random.randint(0, grey_row_index - 1)
        grid[r, c] = random.choice([C_BLUE, C_RED])

    # 放置下方散点
    for i in range(num_lower_dots):
        c = lower_cols[i]
        r = random.randint(grey_row_index + 1, dim - 1)
        grid[r, c] = random.choice([C_BLUE, C_RED])

    # 在这里，grid就是我们最终的输入逻辑网格
    input_logic_grid = grid.copy()
    output_logic_grid = grid.copy()

    # 4. 执行核心变换规则，生成输出
    dot_coords = np.argwhere((grid==C_BLUE) | (grid==C_RED))

    for r, c in dot_coords:
        dot_color = grid[r, c]

        if dot_color==C_RED:
            if r < grey_row_index:
                for y in range(r + 1, grey_row_index):
                    output_logic_grid[y, c] = C_RED
            else:
                for y in range(grey_row_index + 1, r):
                    output_logic_grid[y, c] = C_RED

        elif dot_color==C_BLUE:
            if r < grey_row_index:
                for y in range(0, r):
                    output_logic_grid[y, c] = C_BLUE
            else:
                for y in range(r + 1, dim):
                    output_logic_grid[y, c] = C_BLUE

    return input_logic_grid, output_logic_grid


def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {
        C_BLACK: COLOR_BLACK,
        C_GREY: COLOR_GREY,
        C_BLUE: COLOR_BLUE,
        C_RED: COLOR_RED
    }
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    """
    【已修正】生成并保存单个样本
    """
    try:
        # 1. 生成输入和输出的逻辑网格
        # 现在，我们正确地接收了两个返回的网格
        final_input_grid, final_output_grid = generate_logic_grids()

        # 2. 渲染成图像，使用正确的变量名
        input_image = grid_to_image(final_input_grid, CELL_SIZE)
        output_image = grid_to_image(final_output_grid, CELL_SIZE)

        # 3. 保存
        filename = f"{sample_index:06d}.png"
        input_path = os.path.join(DATASET_DIR, "input", filename)
        output_path = os.path.join(DATASET_DIR, "output", filename)
        input_image.save(input_path)
        output_image.save(output_path)
        return True
    except Exception as e:
        # 保持错误捕获，以防万一
        print(f"在生成样本 {sample_index} 时出错: {e}")
        return False


# --- 主程序入口 (多进程部分无需修改) ---
if __name__=="__main__":
    create_directories()

    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (修正版)...")

    with mp.Pool(processes=num_processes) as pool:
        # 这里不需要传递arrow_mask，所以简化了
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个条件")