import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 配置参数 (根据您的新设定) ---
DATASET_DIR = "arc_recursive_4x4_dataset"
GRID_DIM_IN = 4  # 输入网格维度
GRID_DIM_OUT = 16  # 输出网格维度
CELL_SIZE_OUT = 14  # 输出图像中，每个逻辑格点的像素大小
IMG_SIZE = GRID_DIM_OUT * CELL_SIZE_OUT  # 16 * 14 = 224

# 颜色定义 (R, G, B)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)
NUM_SAMPLES = 1000  # 设定一个较大的样本量

# --- 颜色编码 ---
C_BLACK = 0
C_BLUE = 1
C_RED = 2


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_input_grid(dim, colors):
    """随机生成一个输入网格"""
    return np.random.choice(colors, size=(dim, dim))


def apply_recursive_rule(input_grid):
    """
    核心算法：根据4x4输入网格，生成16x16逻辑输出网格
    """
    dim_in = input_grid.shape[0]
    dim_out = dim_in * dim_in
    output_grid = np.zeros((dim_out, dim_out), dtype=int)

    # 找到所有红色的“指令”格点
    red_coords = np.argwhere(input_grid==C_RED)

    # 遍历所有指令
    for r_instruction, c_instruction in red_coords:
        # 计算在大画布上要“盖章”的起始位置
        start_row = r_instruction * dim_in
        start_col = c_instruction * dim_in
        end_row = start_row + dim_in
        end_col = start_col + dim_in

        # “盖章”操作：将完整的4x4输入图案，复制到对应位置
        output_grid[start_row:end_row, start_col:end_col] = input_grid

    return output_grid


def grid_to_image(grid, cell_size, color_map):
    """将一个逻辑网格转换为PIL图像"""
    dim = grid.shape[0]
    image_size = dim * cell_size
    image = Image.new("RGB", (image_size, image_size))
    pixels = image.load()

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    """
    生成并保存单个样本，用于多进程调用
    """
    # 1. 生成随机的4x4输入逻辑网格
    input_logic_grid = generate_input_grid(GRID_DIM_IN, [C_BLACK, C_BLUE, C_RED])

    # 2. 应用递归规则，生成16x16输出逻辑网格
    output_logic_grid = apply_recursive_rule(input_logic_grid)

    # 3. 将逻辑网格渲染成224x224的图像
    # 输入图像是4x4网格放大而来，每个逻辑格点是 224/4 = 56 像素
    cell_size_in = IMG_SIZE // GRID_DIM_IN
    input_image = grid_to_image(input_logic_grid, cell_size_in,
        {C_BLACK: COLOR_BLACK, C_BLUE: COLOR_BLUE, C_RED: COLOR_RED})

    # 输出图像是16x16网格放大而来，每个逻辑格点是 224/16 = 14 像素
    output_image = grid_to_image(output_logic_grid, CELL_SIZE_OUT,
        {C_BLACK: COLOR_BLACK, C_BLUE: COLOR_BLUE, C_RED: COLOR_RED})

    # 4. 保存文件
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

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个4x4递归任务图像对已保存到 '{DATASET_DIR}' 目录。")