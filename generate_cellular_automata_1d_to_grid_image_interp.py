"""
数据集生成脚本 (版本 3.0 - 逻辑/感知混合)

功能:
1.  支持两种模式:
    a. [默认] 纯逻辑模式: 生成纯黑白的图像。
    b. [新] 内插模式: 在输入图像的黑白块中加入RGB噪声，并要求输出图像
       根据逻辑规则和输入的RGB值进行颜色变换。
2.  在一个任务中，同时考验模型的“离散规则学习”和“连续值内插”能力。
3.  其余功能与版本2.0保持一致 (多进程、元数据生成等)。
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time


# ==============================================================================
# --- 1. 配置区域 (已更新) ---
# ==============================================================================
class Config:
    # --- 基本参数 ---
    CA_WIDTH = 36
    NUM_SAMPLES = 300000
    IMG_SIZE = 240
    GRID_DIM = 6

    # --- 任务特定参数 ---
    RULE_NUMBER = 110
    ITERATIONS = 2

    # ★★★ 新增：实验模式开关 ★★★
    # 设置为 True 来启用我们设计的“逻辑/感知混合”实验
    # 设置为 False 则恢复为生成纯黑白图像的原有行为
    ENABLE_INTERPOLATION_MODE = True

    # --- 输出目录配置 (已更新) ---
    # 根据模式自动添加后缀
    BASE_OUTPUT_DIR = "ca_img2img_dataset_240"
    OUTPUT_DIR = f"{BASE_OUTPUT_DIR}_interp" if ENABLE_INTERPOLATION_MODE else BASE_OUTPUT_DIR
    INITIAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "initial_images")
    FINAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "final_images")
    METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

    # --- 多进程配置 ---
    NUM_WORKERS = os.cpu_count()

    # ★★★ 新增：颜色扰动范围 (仅在内插模式下生效) ★★★
    # 黑色(逻辑0)的RGB值将在 [0, 63] 范围内随机选取
    # 白色(逻辑1)的RGB值将在 [192, 255] 范围内随机选取
    # 注意：为了简化，我们使用灰度值，即R=G=B
    COLOR_BLACK_RANGE = (0, 63)
    COLOR_WHITE_RANGE = (192, 255)

    # --- 基础颜色 (在纯逻辑模式下使用) ---
    COLOR_BLACK_PURE = (0, 0, 0)
    COLOR_WHITE_PURE = (255, 255, 255)


# ==============================================================================
# --- 2. 核心逻辑 (与原脚本保持一致) ---
# ==============================================================================
# apply_rule 和 run_ca_simulation 函数与原脚本完全相同，故此处省略以保持简洁
def apply_rule(state: list, rule_binary: list) -> list:
    width = len(state)
    next_state = [0] * width
    for i in range(width):
        left, center, right = state[i - 1], state[i], state[(i + 1) % width]
        neighborhood_index = left * 4 + center * 2 + right * 1
        rule_index = 7 - neighborhood_index
        next_state[i] = rule_binary[rule_index]
    return next_state


def run_ca_simulation(initial_state: list, rule_num: int, iterations: int) -> list:
    rule_binary = [int(bit) for bit in format(rule_num, '08b')]
    current_state = initial_state
    for _ in range(iterations):
        current_state = apply_rule(current_state, rule_binary)
    return current_state


# ==============================================================================
# --- 3. 核心改造：图像绘制函数 (已更新) ---
# ==============================================================================
def draw_ca_grid_mixed(logical_state: list, img_size: int, grid_dim: int, base_colors: dict = None):
    """
    将给定的逻辑状态渲染成一个网格图像。
    支持纯逻辑模式和内插模式。

    Args:
        logical_state (list): 模型的逻辑状态 [0, 1, 0, ...]。
        img_size (int): 图像尺寸。
        grid_dim (int): 网格维度。
        base_colors (dict, optional):
            - 如果为 None (生成输入图): 将会生成随机颜色并返回它们。
            - 如果提供 (生成输出图): 将使用这些颜色和变换规则来绘制。

    Returns:
        tuple: (PIL.Image.Image, dict)
               返回生成的图像和该图像的颜色映射 (仅在生成输入图时有意义)。
    """
    image = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    if len(logical_state)!=grid_dim * grid_dim:
        raise ValueError(f"状态长度({len(logical_state)})与网格尺寸({grid_dim * grid_dim})不匹配!")

    cell_size = img_size // grid_dim

    # 如果是生成输入图，则创建新的随机颜色
    is_generating_input = base_colors is None
    if is_generating_input:
        generated_colors = {}

    for i, bit in enumerate(logical_state):
        row, col = i // grid_dim, i % grid_dim
        x0, y0 = col * cell_size, row * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size

        # --- 核心颜色逻辑 ---
        if Config.ENABLE_INTERPOLATION_MODE:
            if is_generating_input:
                # 生成输入图：创建并记录随机颜色
                if bit==0:  # 逻辑黑
                    val = random.randint(*Config.COLOR_BLACK_RANGE)
                else:  # 逻辑白
                    val = random.randint(*Config.COLOR_WHITE_RANGE)
                cell_color = (val, val, val)
                generated_colors[i] = val  # 只记录灰度值即可
            else:
                # 生成输出图：使用base_colors和变换规则
                input_val = base_colors[i]
                if bit==1:  # 输出逻辑为白 -> 保持输入颜色
                    val = input_val
                else:  # 输出逻辑为黑 -> 反转输入颜色
                    val = 255 - input_val
                cell_color = (val, val, val)
        else:
            # 纯逻辑模式：使用纯黑纯白
            cell_color = Config.COLOR_BLACK_PURE if bit==0 else Config.COLOR_WHITE_PURE

        draw.rectangle([x0, y0, x1, y1], fill=cell_color, outline=None)

    return image, (generated_colors if is_generating_input else None)


# ==============================================================================
# --- 4. 并行工作单元 (已更新) ---
# ==============================================================================
def generate_sample_worker(task_data):
    """
    独立工作单元，已更新以支持内插模式。
    """
    i, initial_state_str = task_data
    try:
        # 1. 解析初始状态
        initial_state_list = [int(bit) for bit in initial_state_str]

        # 2. 渲染初始状态图像，并捕获其生成的随机颜色
        initial_image, initial_colors = draw_ca_grid_mixed(initial_state_list, Config.IMG_SIZE, Config.GRID_DIM,
            base_colors=None)

        # 3. 执行CA逻辑推理，得到最终的逻辑状态
        final_state_list = run_ca_simulation(initial_state_list, Config.RULE_NUMBER, Config.ITERATIONS)
        final_state_str = "".join(map(str, final_state_list))

        # 4. 渲染最终状态图像，这次传入 initial_colors 来进行颜色变换
        final_image, _ = draw_ca_grid_mixed(final_state_list, Config.IMG_SIZE, Config.GRID_DIM,
            base_colors=initial_colors)

        # 5. 保存图像
        base_filename = f"sample_{i:06d}.png"
        initial_save_path = os.path.join(Config.INITIAL_IMAGES_DIR, base_filename)
        final_save_path = os.path.join(Config.FINAL_IMAGES_DIR, base_filename)

        initial_image.save(initial_save_path)
        final_image.save(final_save_path)

        # 6. 返回元数据
        return (base_filename, base_filename, final_state_str)
    except Exception as e:
        print(f"Worker for task {i} failed with error: {e}")
        return None


# ==============================================================================
# --- 5. 主执行流程 (微调) ---
# ==============================================================================
def main():
    """主函数，负责初始化、调度和后处理。"""
    mode_str = "逻辑/感知混合 (内插模式)" if Config.ENABLE_INTERPOLATION_MODE else "纯逻辑 (黑白模式)"
    print(f"🚀 开始生成数据集...")
    print(f"   - 实验模式: {mode_str}")
    print(f"   - CA宽度: {Config.CA_WIDTH}, 规则: {Config.RULE_NUMBER}, 迭代: {Config.ITERATIONS}")
    print(f"   - 目标样本数: {Config.NUM_SAMPLES}")
    print(f"   - 使用 {Config.NUM_WORKERS} 个CPU核心进行并行处理。")

    start_time = time.time()

    # --- 步骤1: 创建输出目录 ---
    print("\n[1/5] 正在创建目录结构...")
    os.makedirs(Config.INITIAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(Config.FINAL_IMAGES_DIR, exist_ok=True)

    # --- 步骤2: 生成唯一的初始状态 ---
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

    # --- 步骤4: 随机洗牌元数据 ---
    print("\n[4/5] 正在对元数据进行随机洗牌...")
    random.shuffle(metadata)

    # --- 步骤5: 保存元数据到CSV ---
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