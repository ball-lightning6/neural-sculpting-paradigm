import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 ---
DATASET_DIR = "arc_periodic_pattern_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE

# 颜色定义
COLOR_BLACK = (0, 0, 0)  # 背景
COLOR_YELLOW = (255, 255, 0)  # 指令条
COLOR_GREEN = (0, 255, 0)  # 特殊生成色
COLOR_PALETTE = [  # 用于背景色a和散点色b
    (255, 0, 0), (0, 0, 255), (128, 0, 128),
    (255, 165, 0), (0, 255, 255), (192, 192, 192)
]
NUM_SAMPLES = 1000

# 颜色编码
C_BLACK = 0
C_YELLOW = 1
C_GREEN = 2


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def generate_logic_grids():
    dim = GRID_DIM

    # 1. 选择背景色a和散点色b
    color_a_idx, color_b_idx = random.sample(range(len(COLOR_PALETTE)), 2)
    # 颜色代码从3开始
    C_COLOR_A = color_a_idx + 3
    C_COLOR_B = color_b_idx + 3

    # 2. 创建输入网格
    input_grid = np.full((dim, dim), C_COLOR_A, dtype=int)  # 全局背景为a

    # 3. 放置黄色指令条
    yellow_len = random.randint(3, dim // 2)
    m = random.randint(0, dim - yellow_len)
    n = m + yellow_len - 1
    input_grid[dim - 1, m:n + 1] = C_YELLOW

    # 4. 随机散布b色格点（不能在最下面一行）
    num_b_dots = random.randint(dim, dim * dim // 4)
    for _ in range(num_b_dots):
        r = random.randint(0, dim - 2)
        c = random.randint(0, dim - 1)
        input_grid[r, c] = C_COLOR_B

    # --- 5. 生成输出网格 ---
    output_grid = input_grid.copy()

    base_row = dim - 2  # 倒数第二行

    for r in range(dim - 2, -1, -1):  # 从下往上遍历
        distance = base_row - r
        mod_6 = distance % 6

        if mod_6==0:  # 倒数2, 8...行 -> 涂成背景色a
            output_grid[r, m:n + 1] = C_COLOR_A
        elif mod_6==1:  # 倒数3, 9...行 -> 涂成黄色
            output_grid[r, m:n + 1] = C_YELLOW
        elif mod_6==2:  # 倒数4, 10...行 -> 涂成背景色a
            output_grid[r, m:n + 1] = C_COLOR_A
        elif mod_6==3:  # 倒数5, 11...行 -> 复合规则
            # 步骤A: m-n区域涂成绿色
            output_grid[r, m:n + 1] = C_GREEN
            # 步骤B: 整行的b点也变成绿色
            for c in range(dim):
                if input_grid[r, c]==C_COLOR_B:
                    output_grid[r, c] = C_GREEN
        elif mod_6==4:  # 倒数6, 12...行 -> 涂成背景色a
            output_grid[r, m:n + 1] = C_COLOR_A
        elif mod_6 == 5:
            output_grid[r, m:n + 1] = C_YELLOW

    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()

    # 构建完整的颜色映射
    color_map = {
        C_BLACK: COLOR_BLACK,
        C_YELLOW: COLOR_YELLOW,
        C_GREEN: COLOR_GREEN
    }
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 3] = color

    for r in range(dim):
        for c in range(dim):
            # 如果找不到颜色，默认为黑色
            color = color_map.get(grid[r, c], COLOR_BLACK)
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (周期模式版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个周期模式任务图像对已保存到 '{DATASET_DIR}' 目录。")