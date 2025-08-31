import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 ---
DATASET_DIR = "arc_cross_pattern_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE  # 16 * 14 = 224

# 颜色定义 (R, G, B)
COLOR_RED_BG = (255, 0, 0)  # 背景色
COLOR_YELLOW = (255, 255, 0)
COLOR_BLUE = (0, 191, 255)  # 输出的十字形颜色
NUM_SAMPLES = 1000

# 颜色编码
C_RED = 1
C_YELLOW = 2
C_BLUE = 3


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def get_cross_mask():
    """定义一个3x3的十字形掩码"""
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    mask[:, 1] = True
    return mask


def generate_logic_grids(cross_mask):
    """核心算法：生成输入和输出的逻辑网格"""
    dim = GRID_DIM

    # 1. 创建红色背景
    input_grid = np.full((dim, dim), C_RED, dtype=int)

    # 2. 智能地放置“十字形”黄色图案
    num_crosses = random.randint(1, 5)
    cross_centers = []
    # 使用一个标记已占用区域的网格，包括边距
    occupied_with_margin = np.zeros((dim, dim), dtype=bool)

    for _ in range(num_crosses):
        for attempt in range(50):  # 尝试50次找到一个不冲突的位置
            # 中心点不能在最边缘，因为十字形是3x3
            r = random.randint(1, dim - 2)
            c = random.randint(1, dim - 2)

            # 检查3x3外包围框，以及其外一圈的“安全距离”
            # 即检查5x5区域是否空闲
            if not np.any(occupied_with_margin[r - 1:r + 2, c - 1:c + 2]):  # 简化：检查3x3是否空闲

                # 再次检查，确保5x5区域不出界
                if 1 <= r - 1 and r + 1 < dim - 1 and 1 <= c - 1 and c + 1 < dim - 1:
                    # 检查更严格的5x5安全区
                    if not np.any(occupied_with_margin[r - 2:r + 3, c - 2:c + 3]):
                        cross_centers.append((r, c))
                        # 标记3x3区域为已占用
                        occupied_with_margin[r - 1:r + 2, c - 1:c + 2] = True
                        break

    # 在确定的中心点绘制黄色十字
    for r, c in cross_centers:
        input_grid[r - 1:r + 2, c - 1:c + 2][cross_mask] = C_YELLOW

    # 3. 随机散布一些“噪音”黄色点
    num_noise_dots = random.randint(dim, dim * 3)
    for _ in range(num_noise_dots):
        r, c = random.randint(0, dim - 1), random.randint(0, dim - 1)
        # 只在红色背景上添加噪音
        if input_grid[r, c]==C_RED:
            input_grid[r, c] = C_YELLOW

    # 4. 生成输出网格
    output_grid = input_grid.copy()
    # 将所有识别出的十字形，在输出中变为蓝色
    for r, c in cross_centers:
        output_grid[r - 1:r + 2, c - 1:c + 2][cross_mask] = C_BLUE

    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    """将一个逻辑网格转换为PIL图像"""
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {
        C_RED: COLOR_RED_BG,
        C_YELLOW: COLOR_YELLOW,
        C_BLUE: COLOR_BLUE
    }
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_RED_BG)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(args):
    """生成并保存单个样本"""
    sample_index, cross_mask = args
    try:
        input_grid, output_grid = generate_logic_grids(cross_mask)
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

    # 预先计算十字形掩码
    cross_mask = get_cross_mask()

    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (最终加冕版)...")

    args_list = [(i, cross_mask) for i in range(NUM_SAMPLES)]

    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, args_list), total=NUM_SAMPLES, desc="生成数据集"))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终加冕版图像对已保存到 '{DATASET_DIR}' 目录。")