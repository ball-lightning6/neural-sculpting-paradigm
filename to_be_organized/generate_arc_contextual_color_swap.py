import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 ---
DATASET_DIR = "arc_dynamic_swap_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE  # 16 * 14 = 224

# 颜色定义 (R, G, B)
COLOR_BLACK = (0, 0, 0)
# 我们需要一个至少有4种颜色的调色板
COLOR_PALETTE = [
    (255, 0, 0),  # Red
    (0, 0, 255),  # Blue
    (0, 255, 0),  # Green
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
]
NUM_SAMPLES = 1000

# 颜色编码
C_BLACK = 0


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_logic_grids():
    """核心算法：生成输入和输出的逻辑网格"""
    dim = GRID_DIM

    # 1. 随机选择4种不同的颜色作为规则
    color_indices = random.sample(range(len(COLOR_PALETTE)), 4)
    # 颜色代码从1开始
    color_a, color_b, color_c, color_d = [c + 1 for c in color_indices]

    # 初始化输入网格
    input_grid = np.zeros((dim, dim), dtype=int)

    # 2. 设置规则定义区
    input_grid[0, 0] = color_a
    input_grid[0, 1] = color_b
    input_grid[1, 0] = color_c
    input_grid[1, 1] = color_d

    # 3. 在数据操作区随机散布这四种颜色的点
    # 根据您的指示，行号和列号都从4开始 (索引从3开始)
    data_region_start = 4  # 行/列号
    num_data_dots = random.randint(dim, (dim - data_region_start) ** 2 // 2)

    for _ in range(num_data_dots):
        r = random.randint(data_region_start, dim - 1)
        c = random.randint(data_region_start, dim - 1)
        if input_grid[r, c]==C_BLACK:
            input_grid[r, c] = random.choice([color_a, color_b, color_c, color_d])

    # 4. 根据规则生成输出网格
    output_grid = input_grid.copy()

    # 创建交换映射
    swap_map = {
        color_a: color_b,
        color_b: color_a,
        color_c: color_d,
        color_d: color_c
    }

    # 遍历整个网格进行替换
    for r in range(dim):
        for c in range(dim):
            original_color = input_grid[r, c]
            if original_color in swap_map:
                output_grid[r, c] = swap_map[original_color]

    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()

    color_map = {C_BLACK: COLOR_BLACK}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 1] = color  # 颜色代码从1开始

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    """生成并保存单个样本"""
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


# --- 主程序入口 ---
if __name__=="__main__":
    create_directories()

    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本...")

    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个动态规则交换任务图像对已保存到 '{DATASET_DIR}' 目录。")