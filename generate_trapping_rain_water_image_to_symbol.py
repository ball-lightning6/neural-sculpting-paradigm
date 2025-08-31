import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing


# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    # --- 数据集相关 ---
    NUM_COLUMNS_N = 12
    BITS_PER_HEIGHT = 3
    DATASET_SIZE = 300000

    # --- 图像参数 ---
    GRID_SIZE = 6
    CELL_SIZE = 40
    IMAGE_SIZE = GRID_SIZE * CELL_SIZE  # 240

    # --- 输出目录和文件配置 ---
    MAIN_DATASET_DIR = "autodl-tmp/cnn_rainwater_dataset_mp"
    IMAGES_SUBDIR = "initial_images"
    METADATA_FILENAME = "metadata.csv"


# ==============================================================================
# --- 2. 工作函数 (由每个子进程执行) ---
# ==============================================================================
def worker_process(args):
    """
    单个工作进程执行的函数。
    接收一个任务元组，生成图像并返回元数据。
    """
    index, heights, config = args

    # --- a. 核心逻辑：计算输入和输出 ---
    input_str_list = [format(h, f'0{config.BITS_PER_HEIGHT}b') for h in heights]
    input_str = "".join(input_str_list)

    # 精确计算每个格子的雨水量
    n = len(heights)
    water_per_cell = [0] * n
    if n > 0:
        left_max = [0] * n;
        left_max[0] = heights[0]
        for i in range(1, n): left_max[i] = max(left_max[i - 1], heights[i])
        right_max = [0] * n;
        right_max[n - 1] = heights[n - 1]
        for i in range(n - 2, -1, -1): right_max[i] = max(right_max[i + 1], heights[i])
        for i in range(n):
            water_level = min(left_max[i], right_max[i])
            if water_level > heights[i]:
                water_per_cell[i] = water_level - heights[i]

    output_str_list = [format(w, f'0{config.BITS_PER_HEIGHT}b') for w in water_per_cell]
    output_str = "".join(output_str_list)

    # --- b. 图像生成 ---
    img = Image.new('L', (config.IMAGE_SIZE, config.IMAGE_SIZE))
    pixels = img.load()
    for i in range(len(input_str)):
        row, col = i // config.GRID_SIZE, i % config.GRID_SIZE
        color = 0 if input_str[i]=='1' else 255
        for x in range(config.CELL_SIZE):
            for y in range(config.CELL_SIZE):
                pixels[col * config.CELL_SIZE + x, row * config.CELL_SIZE + y] = color

    # --- c. 保存图像 ---
    image_filename = f"sample_{index:06d}.png"
    images_path = os.path.join(config.MAIN_DATASET_DIR, config.IMAGES_SUBDIR)
    full_image_path = os.path.join(images_path, image_filename)
    img.save(full_image_path)

    # --- d. 返回元数据 ---
    return {
        'initial_image': image_filename,
        'final_label': output_str,
        'final_image': 'placeholder'
    }


# ==============================================================================
# --- 3. 主生成函数 (协调多进程) ---
# ==============================================================================
def generate_cnn_dataset_multiprocess(config):
    """
    使用多进程并行生成数据集。
    """
    print("=" * 70)
    print(f" 开始生成CNN“接雨水”数据集 (多进程版)")
    print("=" * 70)

    # --- 步骤 1: 创建目录结构 (主进程) ---
    images_path = os.path.join(config.MAIN_DATASET_DIR, config.IMAGES_SUBDIR)
    os.makedirs(images_path, exist_ok=True)
    print(f"目录 '{images_path}' 已准备就绪。")

    # --- 步骤 2: 生成所有唯一的任务定义 (主进程，快速) ---
    print("\n--- 正在生成唯一的任务定义 ---")
    all_generated_heights = set()
    max_height = 2 ** config.BITS_PER_HEIGHT - 1
    tasks = []

    for i in tqdm(range(config.DATASET_SIZE), desc="定义任务"):
        while True:
            heights_tuple = tuple(random.randint(0, max_height) for _ in range(config.NUM_COLUMNS_N))
            if heights_tuple not in all_generated_heights:
                all_generated_heights.add(heights_tuple)
                tasks.append((i, list(heights_tuple), config))
                break

    # 释放内存
    del all_generated_heights

    # --- 步骤 3: 使用进程池并行处理任务 (多进程，耗时) ---
    num_processes = multiprocessing.cpu_count()
    print(f"\n--- 将使用 {num_processes} 个CPU核心并行生成图像 ---")

    metadata_list = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用imap_unordered来获得更好的性能和tqdm的即时更新
        results_iterator = pool.imap_unordered(worker_process, tasks)

        for result in tqdm(results_iterator, total=len(tasks), desc="生成图像和数据"):
            metadata_list.append(result)

    # --- 步骤 4: 创建并保存metadata.csv (主进程) ---
    print("\n--- 正在创建并保存 metadata.csv ---")
    metadata_df = pd.DataFrame(metadata_list)
    # 按文件名排序，确保元数据文件顺序一致
    metadata_df = metadata_df.sort_values(by='initial_image').reset_index(drop=True)
    metadata_path = os.path.join(config.MAIN_DATASET_DIR, config.METADATA_FILENAME)
    metadata_df.to_csv(metadata_path, index=False)

    print(f"元数据文件已保存到: {metadata_path}")
    print("=" * 70)
    print("数据集生成成功！")
    print("=" * 70)


# ==============================================================================
# --- 4. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    # 在Windows或macOS上，需要把多进程的启动放在 `if __name__ == "__main__":` 块内
    multiprocessing.freeze_support()

    cfg = Config()
    generate_cnn_dataset_multiprocess(cfg)