import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
NUM_INITIAL_STATES = 350  # 初始状态的数量
ITERATIONS = 3  # 我们仍然生成3步后的目标，但只使用初始状态和最终状态
IMG_WIDTH, IMG_HEIGHT = 240, 240
GRID_DIM = 6
OUTPUT_DIR = "image_encoded_ca_dataset"  # ★★★ 新的数据集目录
NUM_WORKERS = os.cpu_count()

# ★★★ 全新的颜色和布局配置 ★★★
# 规则区域颜色
COLOR_RULE_0 = (255, 0, 0)  # Red for '0'
COLOR_RULE_1 = (0, 255, 0)  # Green for '1'
# 状态区域颜色
COLOR_STATE_0 = (255, 255, 255)  # White for '0'
COLOR_STATE_1 = (0, 0, 0)  # Black for '1'

# 布局尺寸
RULE_AREA_HEIGHT = 30
STATE_AREA_HEIGHT = IMG_HEIGHT - RULE_AREA_HEIGHT  # 210
STATE_CELL_HEIGHT = STATE_AREA_HEIGHT // GRID_DIM  # 35
STATE_CELL_WIDTH = IMG_WIDTH // GRID_DIM  # 40


# ==============================================================================
# --- 2. 核心绘图函数 ---
# ==============================================================================

def apply_rule(state, rule_binary):
    """元胞自动机规则应用函数 (与之前相同)"""
    width = len(state)
    next_state = [0] * width
    for i in range(width):
        neighborhood_index = state[i - 1] * 4 + state[i] * 2 + state[(i + 1) % width] * 1
        next_state[i] = rule_binary[7 - neighborhood_index]
    return next_state


def draw_combined_input_image(rule_binary, state):
    """
    ★★★ 核心函数：绘制包含规则和状态的统一输入图像 ★★★
    """
    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "gray")
    draw = ImageDraw.Draw(image)

    # --- 1. 绘制顶部的规则区域 ---
    rule_bit_width = IMG_WIDTH // len(rule_binary)
    for i, bit in enumerate(rule_binary):
        color = COLOR_RULE_1 if bit==1 else COLOR_RULE_0
        x0 = i * rule_bit_width
        x1 = x0 + rule_bit_width
        draw.rectangle([x0, 0, x1, RULE_AREA_HEIGHT], fill=color)

    # --- 2. 绘制下方的状态区域 ---
    y_offset = RULE_AREA_HEIGHT
    for i, cell in enumerate(state):
        row, col = i // GRID_DIM, i % GRID_DIM
        x0 = col * STATE_CELL_WIDTH
        y0 = row * STATE_CELL_HEIGHT + y_offset
        x1 = x0 + STATE_CELL_WIDTH
        y1 = y0 + STATE_CELL_HEIGHT

        color = COLOR_STATE_1 if cell==1 else COLOR_STATE_0
        draw.rectangle([x0, y0, x1, y1], fill=color)

    return image


def draw_simple_output_image(state):
    """
    绘制简单的、只包含最终状态的目标图像 (黑白)
    """
    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), COLOR_STATE_0)
    draw = ImageDraw.Draw(image)
    cell_height_out = IMG_HEIGHT // GRID_DIM
    cell_width_out = IMG_WIDTH // GRID_DIM
    for i, cell in enumerate(state):
        row, col = i // GRID_DIM, i % GRID_DIM
        x0, y0 = col * cell_width_out, row * cell_height_out
        x1, y1 = x0 + cell_width_out, y0 + cell_height_out
        if cell==1:
            draw.rectangle([x0, y0, x1, y1], fill=COLOR_STATE_1)
    return image


# ==============================================================================
# --- 3. 并行工作与主流程 ---
# ==============================================================================

def generate_sample_worker(task_id):
    """为单个任务生成一个输入-输出对"""
    initial_state = [random.choice([0, 1]) for _ in range(GRID_DIM * GRID_DIM)]
    rule_num = random.randint(0, 255)
    rule_str = format(rule_num, '08b')
    rule_binary = [int(bit) for bit in rule_str]

    # 计算3步后的最终状态
    current_state = initial_state
    for _ in range(ITERATIONS):
        current_state = apply_rule(current_state, rule_binary)
    final_state = current_state

    # 创建输入和输出图像
    input_image = draw_combined_input_image(rule_binary, initial_state)
    output_image = draw_simple_output_image(final_state)

    # 保存图像
    filename_in = f"sample_{task_id}_input.png"
    filename_out = f"sample_{task_id}_output.png"
    input_image.save(os.path.join(OUTPUT_DIR, "input", filename_in))
    output_image.save(os.path.join(OUTPUT_DIR, "output", filename_out))

    # 元数据现在非常简单
    return (filename_in, filename_out)


def main():
    print(f"🚀 开始生成【空间条件化】数据集...")
    start_time = time.time()
    os.makedirs(os.path.join(OUTPUT_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "output"), exist_ok=True)

    tasks = list(range(NUM_INITIAL_STATES))

    print(f"并行生成 {len(tasks)} 个训练样本...")
    metadata = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_worker, tasks), total=len(tasks)):
            if result:
                metadata.append(result)

    df = pd.DataFrame(metadata, columns=['input_file', 'output_file'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)

    print(f"\n✅ 数据集生成完毕！总耗时: {time.time() - start_time:.2f} 秒")
    print(f"   - 目录: {OUTPUT_DIR}")
    print(f"   - 样本数: {len(df)}")
    print("\n下一步：使用一个纯图像到图像的模型（如U-Net）来训练这个数据集。")


if __name__=="__main__":
    main()
