import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
import random
from pathlib import Path


# ==============================================================================
# --- 配置区域 (Config Section) ---
# ==============================================================================

class Config:
    DATA_DIR = "catenary_dataset_v5_CONSTRUCTIVE"
    TRAIN_DIR = DATA_DIR#os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 150000
    NUM_SAMPLES_EVAL = 1000

    IMG_SIZE = 224
    BG_COLOR = (255, 255, 255)
    CURVE_COLOR = (0, 255, 0)
    ANCHOR_POINT_COLOR = (255, 0, 0)
    PASS_POINT_COLOR = (0, 0, 255)

    POINT_RADIUS = 5
    CURVE_WIDTH = 10


# ==============================================================================
# --- 核心物理与绘图函数 (Core Physics and Drawing Functions) ---
# ==============================================================================

def catenary_func(x, a, b, c):
    """悬链线的基础方程"""
    return c - a * np.cosh((x - b) / (a + 1e-9))


def draw_image(config, p1, p2, p3, full_curve_points=None):
    img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BG_COLOR)
    draw = ImageDraw.Draw(img)
    if full_curve_points:
        draw.line(full_curve_points, fill=config.CURVE_COLOR, width=config.CURVE_WIDTH)
    for p in [p1, p2]:
        draw.ellipse((p[0] - config.POINT_RADIUS, p[1] - config.POINT_RADIUS,
                      p[0] + config.POINT_RADIUS, p[1] + config.POINT_RADIUS), fill=config.ANCHOR_POINT_COLOR)
    draw.ellipse((p3[0] - config.POINT_RADIUS, p3[1] - config.POINT_RADIUS,
                  p3[0] + config.POINT_RADIUS, p3[1] + config.POINT_RADIUS), fill=config.PASS_POINT_COLOR)


    return img


def create_single_sample(config):
    """
    通过'正向构造'的方法高效生成样本。
    1. 随机生成悬链线参数 a, b, c。
    2. 在曲线上随机取三个点 P1, P2, P3。
    3. 检查所有点是否在图像内。
    """
    size = config.IMG_SIZE
    padding = 20

    while True:
        # 1. 随机生成悬链线参数 (这是新方法的核心)
        # 'a' 控制曲线的平坦度: a越大越平坦
        a = random.uniform(20, 80)
        # 'b' 控制曲线的水平位置 (最低点x坐标)
        b = random.uniform(size * 0.02, size * 0.8)
        # 'c' 控制曲线的垂直位置。我们希望最低点y_min = c - a 在图像内
        y_min = random.uniform(padding, size * 0.4)
        c = y_min + a

        # 2. 在这条随机的悬链线上，随机选取P1, P2, P3
        # 确定P1, P2的x坐标范围，确保它们不会太近
        x_range_width = random.uniform(size * 0.2, size * 0.8)
        x_start = random.uniform(padding, size - padding - x_range_width)
        x_end = x_start + x_range_width

        x1 = x_start
        x2 = x_end
        # 在 P1, P2 之间选取 P3 的 x 坐标
        x3 = random.uniform(x1 + (x2 - x1) * 0.2, x1 + (x2 - x1) * 0.8)

        # 计算三个点的y坐标
        y1 = catenary_func(x1, a, b, c)
        y2 = catenary_func(x2, a, b, c)
        y3 = catenary_func(x3, a, b, c)

        p1, p2, p3 = (x1, y1), (x2, y2), (x3, y3)

        # 3. 检查所有点和曲线是否在图像内
        all_points = np.array([p1, p2, p3])
        if np.any(all_points < padding) or np.any(all_points >= size - padding):
            continue

        # 生成完整的曲线用于绘制
        x_coords = np.linspace(x1, x2, 200)
        y_coords = catenary_func(x_coords, a, b, c)

        # 再次检查整条曲线是否越界
        if np.any(y_coords < 0) or np.any(y_coords >= size):
            continue

        full_curve_points = list(zip(x_coords, y_coords))

        # 所有检查通过，这是一个完美的样本
        input_image = draw_image(config, p1, p2, p3)
        output_image = draw_image(config, p1, p2, p3, full_curve_points)
        return input_image, output_image


def generate_dataset(num_samples, output_dir, name, config):
    print(f"\n正在生成 {name} 数据集 ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)
    # 这个循环会比之前快得多！
    for i in tqdm(range(num_samples), desc=f"生成 {name}", unit="个世界"):
        input_img, output_img = create_single_sample(config)
        input_img.save(Path(output_dir) /"input"/ f"{i:06d}.png")
        output_img.save(Path(output_dir) /"output"/ f"{i:06d}.png")


# ==============================================================================
# --- 程序入口 (Main Execution Block) ---
# ==============================================================================

if __name__=="__main__":
    cfg = Config()
    if os.path.exists(cfg.DATA_DIR):
        shutil.rmtree(cfg.DATA_DIR)
    print("=" * 60)
    print("开始生成物理推理数据集：悬链线问题 v5.0 (正向构造版)")
    print("方法: 先生成随机悬链线，再从线上取点。高效、优雅、无求解器。")
    print("=" * 60)
    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
    # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)
    print(f"\n🎉🎉🎉 数据集 '{cfg.DATA_DIR}' 生成完毕！ 🎉🎉🎉")
