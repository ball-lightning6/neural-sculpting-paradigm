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
CA_WIDTH = 36  # 元胞自动机的位数长度
NUM_SAMPLES = 1000  # 您希望生成的数据集大小
IMG_SIZE = 240  # 输出图像尺寸
OUTPUT_DIR = "ca_render_dataset_240"  # 输出目录

# --- 任务特定参数 ---
RULE_NUMBER = 110  # 固定的演化规则
ITERATIONS = 3  # 固定的演化层数
GRID_DIM = 6  # 网格维度 (6x6 = 36)

# --- 多进程配置 ---
NUM_WORKERS = os.cpu_count()

# --- 颜色配置 ---
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


# ==============================================================================
# --- 2. 核心逻辑 (推理 & 渲染) ---
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


def draw_ca_grid(final_state: list, img_size: int, grid_dim: int) -> Image.Image:
    """将最终状态渲染成一个网格图像。"""
    image = Image.new("RGB", (img_size, img_size), COLOR_WHITE)
    draw = ImageDraw.Draw(image)

    if len(final_state)!=grid_dim * grid_dim:
        raise ValueError("最终状态的长度与网格尺寸不匹配!")

    cell_size = img_size // grid_dim

    for i in range(len(final_state)):
        row = i // grid_dim
        col = i % grid_dim

        x0 = col * cell_size
        y0 = row * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        # 1 = black, 0 = white
        cell_color = COLOR_BLACK if final_state[i]==1 else COLOR_WHITE
        draw.rectangle([x0, y0, x1, y1], fill=cell_color, outline=None)

    return image


# ==============================================================================
# --- 3. 并行工作单元 ---
# ==============================================================================

def generate_sample_worker(task_data):
    """
    一个独立的工作单元，负责完整的“推理->渲染->保存”流程。
    """
    i, initial_state_str = task_data
    try:
        # 1. 解析输入
        initial_state_list = [int(bit) for bit in initial_state_str]

        # 2. 执行推理
        final_state = run_ca_simulation(initial_state_list, RULE_NUMBER, ITERATIONS)

        # 3. 执行渲染
        output_image = draw_ca_grid(final_state, IMG_SIZE, GRID_DIM)

        # 4. 保存图像
        filename = f"ca_110_{i:06d}.png"
        save_path = os.path.join(OUTPUT_DIR, "images", filename)
        output_image.save(save_path)

        # 5. 返回元数据 (文件名, 原始输入标签)
        return (filename, initial_state_str)
    except Exception as e:
        print(f"Worker for task {i} failed with error: {e}")
        return None


# ==============================================================================
# --- 4. 主执行流程 (调度、去重、乱序) ---
# ==============================================================================

def main():
    """主函数，负责初始化、调度和后处理。"""
    print(f"🚀 开始生成“推理+渲染”数据集...")
    print(f"   - CA宽度: {CA_WIDTH}, 规则: {RULE_NUMBER}, 迭代: {ITERATIONS}")
    print(f"   - 目标样本数: {NUM_SAMPLES}")
    print(f"   - 使用 {NUM_WORKERS} 个CPU核心。")

    start_time = time.time()

    # --- 步骤1: 生成唯一的初始状态 (去重) ---
    print("\n[1/4] 正在生成唯一的初始状态...")
    unique_states = set()
    pbar_unique = tqdm(total=NUM_SAMPLES, desc="Generating unique states")
    while len(unique_states) < NUM_SAMPLES:
        state_str = "".join(random.choice('01') for _ in range(CA_WIDTH))
        if state_str not in unique_states:
            unique_states.add(state_str)
            pbar_unique.update(1)
    pbar_unique.close()
    initial_states_list = list(unique_states)

    # --- 步骤2: 并行生成图像和元数据 ---
    print("\n[2/4] 正在并行生成图像...")
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        tasks = enumerate(initial_states_list)
        results_iterator = pool.imap_unordered(generate_sample_worker, tasks)

        for result in tqdm(results_iterator, total=NUM_SAMPLES, desc="Rendering images"):
            if result:
                metadata.append(result)

    # --- 步骤3: 对元数据进行随机洗牌 (乱序) ---
    print("\n[3/4] 正在对元数据进行随机洗牌...")
    random.shuffle(metadata)

    # --- 步骤4: 保存元数据到CSV文件 ---
    print("\n[4/4] 正在写入 metadata.csv...")
    df = pd.DataFrame(metadata, columns=['filename', 'label'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)

    end_time = time.time()

    print("\n✅ 数据集生成完毕！")
    print(f"   - 成功生成 {len(df)} 个样本。")
    print(f"   - 总耗时: {end_time - start_time:.2f} 秒")
    print("\n示例数据 (洗牌后):")
    print(df.head())


if __name__=="__main__":
    main()
