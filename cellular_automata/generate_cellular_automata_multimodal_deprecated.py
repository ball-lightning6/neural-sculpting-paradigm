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
NUM_SAMPLES = 100  # 您希望生成的数据集大小
IMG_SIZE = 240  # 输出图像尺寸
GRID_DIM = 6  # 网格维度 (6x6 = 36)
CA_WIDTH = GRID_DIM * GRID_DIM  # 元胞自动机宽度
ITERATIONS = 3  # 固定的演化层数
OUTPUT_DIR = "multimodal_ca_dataset"  # 输出目录
NUM_WORKERS = os.cpu_count()

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
        # 将一维状态视为环形进行邻域计算
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
    """将状态渲染成一个网格图像。"""
    image = Image.new("RGB", (img_size, img_size), COLOR_WHITE)
    draw = ImageDraw.Draw(image)
    cell_size = img_size // grid_dim
    for i in range(len(state)):
        row, col = i // grid_dim, i % grid_dim
        x0, y0 = col * cell_size, row * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        cell_color = COLOR_BLACK if state[i]==1 else COLOR_WHITE
        draw.rectangle([x0, y0, x1, y1], fill=cell_color)
    return image


# ==============================================================================
# --- 3. 并行工作单元 ---
# ==============================================================================

def generate_sample_worker(i):
    """一个独立的工作单元，负责生成一个完整的多模态样本。"""
    try:
        # 1. 生成随机初始状态 (用于输入图像)
        initial_state_str = "".join(random.choice('01') for _ in range(CA_WIDTH))
        initial_state_list = [int(bit) for bit in initial_state_str]

        # 2. 生成随机规则 (用于文本指令)
        rule_num = random.randint(0, 255)
        rule_str = format(rule_num, '08b')  # 8位01字符串

        # 3. 执行推理，得到最终状态
        final_state_list = run_ca_simulation(initial_state_list, rule_num, ITERATIONS)

        # 4. 渲染输入和输出图像
        input_image = draw_ca_grid(initial_state_list, IMG_SIZE, GRID_DIM)
        output_image = draw_ca_grid(final_state_list, IMG_SIZE, GRID_DIM)

        # 5. 保存图像到各自的文件夹
        filename = f"sample_{i:06d}.png"
        input_save_path = os.path.join(OUTPUT_DIR, "input", filename)
        output_save_path = os.path.join(OUTPUT_DIR, "output", filename)
        input_image.save(input_save_path)
        output_image.save(output_save_path)

        # 6. 返回元数据
        return (filename, filename, rule_str)  # (input_file, output_file, caption)
    except Exception as e:
        print(f"Worker {i} failed with error: {e}")
        return None


# ==============================================================================
# --- 4. 主执行流程 ---
# ==============================================================================

def main():
    print(f"🚀 开始生成多模态元胞自动机数据集...")
    start_time = time.time()

    # 创建目录结构
    os.makedirs(os.path.join(OUTPUT_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "output"), exist_ok=True)

    print(f"\n[1/3] 正在使用 {NUM_WORKERS} 个核心并行生成 {NUM_SAMPLES} 个样本...")
    metadata = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        tasks = range(NUM_SAMPLES)
        results_iterator = pool.imap_unordered(generate_sample_worker, tasks)
        for result in tqdm(results_iterator, total=NUM_SAMPLES, desc="Generating samples"):
            if result:
                metadata.append(result)

    print("\n[2/3] 正在对元数据进行随机洗牌...")
    random.shuffle(metadata)

    print("\n[3/3] 正在写入 metadata.csv...")
    df = pd.DataFrame(metadata, columns=['input_file', 'output_file', 'caption'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)

    end_time = time.time()
    print(f"\n✅ 数据集生成完毕！总耗时: {end_time - start_time:.2f} 秒")
    print(f"   - 目录: {OUTPUT_DIR}")
    print(f"   - 样本数: {len(df)}")
    print("\n示例元数据:")
    print(df.head())


if __name__=="__main__":
    main()
