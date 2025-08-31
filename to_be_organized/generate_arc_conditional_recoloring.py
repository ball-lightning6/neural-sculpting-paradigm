import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 配置参数 ---
DATASET_DIR = "arc_conditional_bbox_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 139)  # 深蓝色 (初始随机点)
COLOR_LIGHT_BLUE = (173, 216, 230)  # 浅蓝色 (矩形内的黑色变成的)
COLOR_GREEN = (0, 255, 0)  # 绿色 (输出中，矩形内的深蓝色变成的)
NUM_SAMPLES = 1000

# --- 颜色编码 ---
C_BLACK = 0
C_BLUE = 1
C_LIGHT_BLUE = 2
C_GREEN = 3

# --- 任务参数 ---
INITIAL_BLUE_RATIO = 0.4  # 初始深蓝色散点的比例


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_logic_grids():
    """
    核心算法：严格按照您的三步逻辑生成输入和输出网格
    """
    dim = GRID_DIM

    # 步骤 1: 首先随机产生蓝色格点
    # 创建一个基础网格，其中一些是蓝色，一些是黑色
    grid = np.random.choice(
        [C_BLACK, C_BLUE],
        size=(dim, dim),
        p=[1 - INITIAL_BLUE_RATIO, INITIAL_BLUE_RATIO]
    )

    # 步骤 2: 然后随机一个矩形区域，把这个矩形区域里的黑色变成淡蓝色
    # 随机确定矩形大小和位置
    w = random.randint(3, dim - 2)
    h = random.randint(3, dim - 2)
    r_start = random.randint(0, dim - h)
    c_start = random.randint(0, dim - w)

    # 创建输入网格
    input_grid = grid.copy()
    for r in range(r_start, r_start + h):
        for c in range(c_start, c_start + w):
            if input_grid[r, c]==C_BLACK:
                input_grid[r, c] = C_LIGHT_BLUE

    # 步骤 3: 输出是把这个矩形区域里的蓝色再变成绿色
    output_grid = input_grid.copy()
    for r in range(r_start, r_start + h):
        for c in range(c_start, c_start + w):
            # 注意：这里要用原始的grid来判断是否是蓝色
            if grid[r, c]==C_BLUE:
                output_grid[r, c] = C_GREEN

    return input_grid, output_grid


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


def generate_single_sample(sample_index):
    """生成并保存单个样本"""
    # 1. 生成输入和输出的逻辑网格
    input_logic_grid, output_logic_grid = generate_logic_grids()

    # 2. 渲染成图像
    color_map_in = {C_BLACK: COLOR_BLACK, C_BLUE: COLOR_BLUE, C_LIGHT_BLUE: COLOR_LIGHT_BLUE}
    color_map_out = {
        C_BLACK: COLOR_BLACK,
        C_BLUE: COLOR_BLUE,  # 输出图中，框外的蓝色保持不变
        C_LIGHT_BLUE: COLOR_LIGHT_BLUE,  # 输出图中，框内的浅蓝色保持不变
        C_GREEN: COLOR_GREEN  # 输出图中，框内的深蓝色变为绿色
    }

    input_image = grid_to_image(input_logic_grid, CELL_SIZE, color_map_in)
    output_image = grid_to_image(output_logic_grid, CELL_SIZE, color_map_out)

    # 3. 保存
    filename = f"{sample_index:06d}.png"
    input_path = os.path.join(DATASET_DIR, "input", filename)
    output_path = os.path.join(DATASET_DIR, "output", filename)
    input_image.save(input_path)
    output_image.save(output_path)

    return True


# --- 主程序入口 ---
if __name__=="__main__":
    create_directories()

    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本...")

    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个条件矩形变换图像对已保存到 '{DATASET_DIR}' 目录。")