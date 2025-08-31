"""
数据集生成脚本 (版本 2.0 - Img2Img & Img2Label)

功能:
1.  生成一维元胞自动机的随机初始状态。
2.  将【初始状态】渲染成一张输入图像 (input image)。
3.  根据指定规则对初始状态进行演化，得到最终状态。
4.  将【最终状态】渲染成一张目标图像 (target image)。
5.  将【最终状态】的符号形式作为标签 (target label)。
6.  使用多进程并行处理，高效生成大规模数据集。
7.  生成一个metadata.csv文件，关联所有输入、输出和标签，以供后续训练使用。

此脚本生成的数据可同时用于：
- 图像到图像任务 (如 UNet, Diffusion): 使用 initial_image -> final_image
- 图像到分类任务 (如 ConvNeXt): 使用 initial_image -> final_label
"""
import os
import random
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time


# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
class Config:
    # --- 基本参数 ---
    CA_WIDTH = 36  # 元胞自动机的位数长度
    NUM_SAMPLES = 300000  # 您希望生成的数据集大小
    IMG_SIZE = 240  # 输出图像尺寸
    GRID_DIM = 6  # 网格维度 (6x6 = 36)

    # --- 任务特定参数 ---
    RULE_NUMBER = 110  # 固定的演化规则
    ITERATIONS = 2  # 固定的演化步数

    # --- 输出目录配置 ---
    OUTPUT_DIR = "ca_img2img_dataset_240"
    INITIAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "initial_images")
    FINAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "final_images")
    METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

    # --- 多进程配置 ---
    # 使用所有可用的CPU核心，如果机器负载高可以适当减小
    NUM_WORKERS = os.cpu_count()

    # --- 颜色配置 ---
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)


# ==============================================================================
# --- 2. 核心逻辑 (与原脚本保持一致) ---
# ==============================================================================
def apply_rule(state: list, rule_binary: list) -> list:
    """对给定的状态应用一次元胞自动机规则（周期性边界条件）。"""
    width = len(state)
    next_state = [0] * width
    for i in range(width):
        left, center, right = state[i - 1], state[i], state[(i + 1) % width]
        neighborhood_index = left * 4 + center * 2 + right * 1
        rule_index = 7 - neighborhood_index
        next_state[i] = rule_binary[rule_index]
    return next_state


def run_ca_simulation(initial_state: list, rule_num: int, iterations: int) -> list:
    """执行完整的元胞自动机模拟。"""
    rule_binary = [int(bit) for bit in format(rule_num, '08b')]
    current_state = initial_state
    for _ in range(iterations):
        current_state = apply_rule(current_state, rule_binary)
    return current_state


def draw_ca_grid(state: list, img_size: int, grid_dim: int) -> Image.Image:
    """将给定的状态 (初始或最终) 渲染成一个网格图像。"""
    image = Image.new("RGB", (img_size, img_size), Config.COLOR_WHITE)
    draw = ImageDraw.Draw(image)

    if len(state)!=grid_dim * grid_dim:
        raise ValueError(f"状态长度({len(state)})与网格尺寸({grid_dim * grid_dim})不匹配!")

    cell_size = img_size // grid_dim
    for i, bit in enumerate(state):
        row, col = i // grid_dim, i % grid_dim
        x0, y0 = col * cell_size, row * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        cell_color = Config.COLOR_BLACK if bit==1 else Config.COLOR_WHITE
        draw.rectangle([x0, y0, x1, y1], fill=cell_color, outline=None)
    return image


# ==============================================================================
# --- 3. 并行工作单元 (已更新) ---
# ==============================================================================
def generate_sample_worker(task_data):
    """
    一个独立的工作单元，负责生成一对图像和一个标签并返回元数据。
    """
    i, initial_state_str = task_data
    try:
        # 1. 解析初始状态
        initial_state_list = [int(bit) for bit in initial_state_str]

        # 2. 【新增】渲染初始状态图像
        initial_image = draw_ca_grid(initial_state_list, Config.IMG_SIZE, Config.GRID_DIM)

        # 3. 执行推理，得到最终状态
        final_state_list = run_ca_simulation(initial_state_list, Config.RULE_NUMBER, Config.ITERATIONS)
        final_state_str = "".join(map(str, final_state_list))  # 转换为字符串标签

        # 4. 渲染最终状态图像
        final_image = draw_ca_grid(final_state_list, Config.IMG_SIZE, Config.GRID_DIM)

        # 5. 保存两张图像到不同目录，使用统一的文件名
        base_filename = f"sample_{i:06d}.png"
        initial_save_path = os.path.join(Config.INITIAL_IMAGES_DIR, base_filename)
        final_save_path = os.path.join(Config.FINAL_IMAGES_DIR, base_filename)

        initial_image.save(initial_save_path)
        final_image.save(final_save_path)

        # 6. 返回完整的元数据
        return (base_filename, base_filename, final_state_str)
    except Exception as e:
        print(f"Worker for task {i} failed with error: {e}")
        return None


# ==============================================================================
# --- 4. 主执行流程 (已更新) ---
# ==============================================================================
def main():
    """主函数，负责初始化、调度和后处理。"""
    print("🚀 开始生成“图像到图像 & 图像到标签”通用数据集...")
    print(f"   - CA宽度: {Config.CA_WIDTH}, 规则: {Config.RULE_NUMBER}, 迭代: {Config.ITERATIONS}")
    print(f"   - 目标样本数: {Config.NUM_SAMPLES}")
    print(f"   - 使用 {Config.NUM_WORKERS} 个CPU核心进行并行处理。")

    start_time = time.time()

    # --- 步骤1: 创建输出目录 ---
    print("\n[1/5] 正在创建目录结构...")
    os.makedirs(Config.INITIAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(Config.FINAL_IMAGES_DIR, exist_ok=True)

    # --- 步骤2: 生成唯一的初始状态 (去重) ---
    print("\n[2/5] 正在生成唯一的初始状态...")
    unique_states = set()
    pbar_unique = tqdm(total=Config.NUM_SAMPLES, desc="Generating unique states")
    while len(unique_states) < Config.NUM_SAMPLES:
        state_str = "".join(random.choice('01') for _ in range(Config.CA_WIDTH))
        if state_str not in unique_states:
            unique_states.add(state_str)
            pbar_unique.update(1)
    pbar_unique.close()
    initial_states_list = list(unique_states)

    # --- 步骤3: 并行生成图像和元数据 ---
    print("\n[3/5] 正在并行生成图像数据对...")
    metadata = []
    with multiprocessing.Pool(processes=Config.NUM_WORKERS) as pool:
        tasks = enumerate(initial_states_list)
        results_iterator = pool.imap_unordered(generate_sample_worker, tasks)

        for result in tqdm(results_iterator, total=Config.NUM_SAMPLES, desc="Rendering images"):
            if result:
                metadata.append(result)

    # --- 步骤4: 对元数据进行随机洗牌 (乱序) ---
    print("\n[4/5] 正在对元数据进行随机洗牌...")
    random.shuffle(metadata)

    # --- 步骤5: 保存元数据到CSV文件 ---
    print("\n[5/5] 正在写入 metadata.csv...")
    df = pd.DataFrame(metadata, columns=['initial_image', 'final_image', 'final_label'])
    df.to_csv(Config.METADATA_FILE, index=False)

    end_time = time.time()

    print("\n✅ 数据集生成完毕！")
    print(f"   - 成功生成 {len(df)} 个样本对。")
    print(f"   - 数据保存在: {Config.OUTPUT_DIR}")
    print(f"   - 总耗时: {end_time - start_time:.2f} 秒")
    print("\n元数据文件 (metadata.csv) 示例 (洗牌后):")
    print(df.head())


if __name__=="__main__":
    main()