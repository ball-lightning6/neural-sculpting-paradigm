import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 (保持不变) ---
DATASET_DIR = "arc_final_sorting_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_BG_BLUE = (173, 216, 230)
COLOR_PURPLE_AXIS = (128, 0, 128)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 165, 0), (0, 255, 255), (192, 192, 192)
]
NUM_SAMPLES = 1000

C_BG_BLUE = 1
C_PURPLE = 2


### 2. 辅助函数 (保持不变) ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


### 3. 核心生成逻辑 ###
def generate_logic_grids():
    """【最终毕业典礼版】核心算法"""
    dim = GRID_DIM

    # --- 步骤1 & 2: 生成输入网格 (与之前版本逻辑相同) ---
    grid = np.full((dim, dim), C_BG_BLUE, dtype=int)
    axis_row = dim // 2 - 1
    grid[axis_row, :] = C_PURPLE

    possible_lengths = [i for i in range(1, dim // 2, 2) if i > 1]
    num_columns = random.randint(3, 10)

    columns_data = []

    column_indices = sorted(random.sample(range(dim), num_columns))
    colors = random.choices(range(len(COLOR_PALETTE)), k=num_columns)
    lengths = random.choices(possible_lengths, k=num_columns)

    for i in range(num_columns):
        columns_data.append({
            "col": column_indices[i],
            "len": lengths[i],
            "color_code": colors[i] + 3
        })

    input_grid = grid.copy()
    for col_info in columns_data:
        c, length, color = col_info["col"], col_info["len"], col_info["color_code"]
        start_offset = (length - 1) // 2
        start_row, end_row = axis_row - start_offset, axis_row + start_offset
        input_grid[start_row: end_row + 1, c] = color

    # --- 步骤3: 【核心修正】生成输出网格 ---
    output_grid = grid.copy()

    # a. 提取所有柱子的属性
    original_columns_attrs = []
    for col_info in columns_data:
        original_columns_attrs.append({
            "len": col_info["len"],
            "color_code": col_info["color_code"]
        })

    # b. 对属性列表进行排序
    sorted_attrs = sorted(original_columns_attrs, key=lambda x: x["len"])

    # c. 获取原始的列位置，并保持其从左到右的顺序
    original_col_positions = sorted([col_info["col"] for col_info in columns_data])

    # d. 【新】按原始列位置，用排序后的属性，重新绘制柱子
    for i in range(len(original_col_positions)):
        c = original_col_positions[i]  # 列的位置不变

        # 获取排序后的新属性
        new_length = sorted_attrs[i]["len"]
        new_color = original_columns_attrs[i]["color_code"]

        # 用新属性绘制新的柱子
        start_offset = (new_length - 1) // 2
        start_row, end_row = axis_row - start_offset, axis_row + start_offset
        output_grid[start_row: end_row + 1, c] = new_color

    return input_grid, output_grid


# --- 后续函数 (grid_to_image, generate_single_sample, main) 保持不变 ---
def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {C_BG_BLUE: COLOR_BG_BLUE, C_PURPLE: COLOR_PURPLE_AXIS}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 3] = color

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BG_BLUE)
            for i_ in range(cell_size):
                for j_ in range(cell_size):
                    pixels[c * cell_size + j_, r * cell_size + i_] = color
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (毕业典礼版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终毕业典礼版图像对已保存到 '{DATASET_DIR}' 目录。")