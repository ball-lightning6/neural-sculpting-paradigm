import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

### 1. 配置参数 ###
DATASET_DIR = "arc_meta_reasoning_dataset"
# 输入配置
GRID_DIM_IN = 25
CELL_SIZE_IN = 8
IMG_SIZE_IN = GRID_DIM_IN * CELL_SIZE_IN  # 200
# 输出配置
GRID_DIM_OUT = 13
CELL_SIZE_OUT = 16
IMG_SIZE_OUT = GRID_DIM_OUT * CELL_SIZE_OUT  # 208
# 画布配置
CANVAS_SIZE = 224

# 颜色定义
COLOR_BLACK = (0, 0, 0)  # 未使用，但保留
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128),
    (255, 165, 0), (0, 255, 255), (192, 192, 192), (240, 128, 128),
    (124, 252, 0), (255, 255, 255), (100, 100, 100)
]
NUM_SAMPLES = 1000

# 颜色编码
C_BLACK = 0


### 2. 辅助函数 ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_random_3x3_pattern(c_b, c_c):
    """随机生成一个3x3的图案"""
    return np.random.choice([c_b, c_c], size=(3, 3))


def embed_in_canvas(grid_img, canvas_size):
    """将生成的图像居中放置在一个更大的画布上"""
    canvas = Image.new("RGB", (canvas_size, canvas_size), grid_img.getpixel((0, 0)))
    offset_x = (canvas_size - grid_img.width) // 2
    offset_y = (canvas_size - grid_img.height) // 2
    canvas.paste(grid_img, (offset_x, offset_y))
    return canvas


### 3. 核心生成逻辑 ###
def generate_logic_grids():
    # --- 步骤1：初始化和颜色选择 ---
    dim = GRID_DIM_IN
    input_grid = np.zeros((dim, dim), dtype=int)

    #
    # 随机选择所有需要的颜色，确保互不相同
    # 背景色a, 分割线(也是a), 4x(背景b, 图案c) -> 2 + 4*2 = 10种
    try:
        colors_needed = 2 + 8
        chosen_colors = random.sample(range(len(COLOR_PALETTE)), colors_needed)
    except ValueError:
        print("调色板颜色不足，请增加COLOR_PALETTE中的颜色数量。")
        return None

    C_COLOR_A = chosen_colors[0] + 1
    # 分割线也用背景色a

    input_grid[:, :] = C_COLOR_A
    for i in range(0, dim + 1, 6):
        input_grid[i, :] = C_COLOR_A
        input_grid[:, i] = C_COLOR_A

    # --- 步骤2：为每一行生成图案 ---
    special_patterns = []  # 存储每一行的特殊图案

    for r_idx in range(4):  # 遍历4个宏观行
        # 为这一行选择独特的背景b和图案c
        C_COLOR_B = chosen_colors[2 + r_idx * 2] + 1
        C_COLOR_C = chosen_colors[2 + r_idx * 2 + 1] + 1

        # 生成两种图案：一种是主流，一种是特殊的
        pattern_common = generate_random_3x3_pattern(C_COLOR_B, C_COLOR_C)
        pattern_special = generate_random_3x3_pattern(C_COLOR_B, C_COLOR_C)
        # 确保特殊图案与主流图案不同
        while np.array_equal(pattern_common, pattern_special):
            pattern_special = generate_random_3x3_pattern(C_COLOR_B, C_COLOR_C)

        special_patterns.append(pattern_special)

        # 决定哪个位置是特殊的
        special_col_idx = random.randint(0, 3)

        # 将图案填充到输入网格中
        for c_idx in range(4):
            # 定位到5x5空洞的左上角
            r_start = 1 + r_idx * 6
            c_start = 1 + c_idx * 6

            # 先用b颜色填充边框
            input_grid[r_start, c_start:c_start + 5] = C_COLOR_B
            input_grid[r_start + 4, c_start:c_start + 5] = C_COLOR_B
            input_grid[r_start:r_start + 5, c_start] = C_COLOR_B
            input_grid[r_start:r_start + 5, c_start + 4] = C_COLOR_B

            # 填充内部3x3图案
            inner_r_start, inner_c_start = r_start + 1, c_start + 1
            if c_idx==special_col_idx:
                input_grid[inner_r_start:inner_r_start + 3, inner_c_start:inner_c_start + 3] = pattern_special
            else:
                input_grid[inner_r_start:inner_r_start + 3, inner_c_start:inner_c_start + 3] = pattern_common

    # --- 步骤3：生成输出网格 ---
    output_grid = np.full((GRID_DIM_OUT, GRID_DIM_OUT), C_COLOR_A, dtype=int)
    for i in range(0, GRID_DIM_OUT + 1, 6):
        output_grid[i, :] = C_COLOR_A
        output_grid[:, i] = C_COLOR_A

    output_positions = [(1, 1), (1, 7), (7, 1), (7, 7)]  # 输出的4个5x5空洞的左上角

    for i in range(4):
        r_out_start, c_out_start = output_positions[i]
        pattern_to_draw = special_patterns[i]  # 将第i行的特殊图案画到第i个位置
        # 同样需要先画边框，再画内容
        output_grid[r_out_start, c_out_start:c_out_start + 5] = chosen_colors[2 + i * 2] + 1#pattern_to_draw[0, 0]  # 用图案的背景色画边框
        output_grid[r_out_start + 4, c_out_start:c_out_start + 5] = chosen_colors[2 + i * 2] + 1#pattern_to_draw[0, 0]
        output_grid[r_out_start:r_out_start + 5, c_out_start] = chosen_colors[2 + i * 2] + 1#pattern_to_draw[0, 0]
        output_grid[r_out_start:r_out_start + 5, c_out_start + 4] = chosen_colors[2 + i * 2] + 1#pattern_to_draw[0, 0]

        output_grid[r_out_start + 1:r_out_start + 4, c_out_start + 1:c_out_start + 4] = pattern_to_draw

    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    # ...
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {C_BLACK: COLOR_BLACK}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 1] = color

    #
    # Find background color
    vals, counts = np.unique(grid, return_counts=True)
    bg_code = vals[np.argmax(counts)]

    for r in range(dim):
        for c in range(dim):
            code = grid[r, c]
            color = color_map.get(code, color_map[bg_code])
            for i_ in range(cell_size):
                for j_ in range(cell_size):
                    pixels[c * cell_size + j_, r * cell_size + i_] = color
    return image


def generate_single_sample(sample_index):
    try:
        result = None
        while result is None:
            result = generate_logic_grids()
        input_grid, output_grid = result

        input_image = grid_to_image(input_grid, CELL_SIZE_IN)
        output_image = grid_to_image(output_grid, CELL_SIZE_OUT)

        # 嵌入到最终画布
        final_input_image = embed_in_canvas(input_image, CANVAS_SIZE)
        final_output_image = embed_in_canvas(output_image, CANVAS_SIZE)

        filename = f"{sample_index:06d}.png"
        input_path = os.path.join(DATASET_DIR, "input", filename)
        output_path = os.path.join(DATASET_DIR, "output", filename)
        final_input_image.save(input_path)
        final_output_image.save(output_path)
        return True
    except Exception as e:
        print(f"在生成样本 {sample_index} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__=="__main__":
    create_directories()
    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (终极毕业设计版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终版图像对已保存到 '{DATASET_DIR}' 目录。")