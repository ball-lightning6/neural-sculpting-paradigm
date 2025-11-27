import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
import random
from pathlib import Path


# ==============================================================================
# --- 配置区域 ---
# ==============================================================================

class Config:
    # --- 目录和样本数量 ---
    DATA_DIR = "centroid_dataset"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 500
    NUM_SAMPLES_EVAL = 500

    # --- 图像属性 ---
    IMG_SIZE = 224
    BG_COLOR = (255, 255, 255)  # 背景 - 白色
    TRIANGLE_COLOR = (0, 255, 0)  # 三角形 - 绿色
    CENTROID_CIRCLE_COLOR = (255, 0, 0)  # 重心圆 - 红色

    # --- 几何约束 ---
    MIN_TRIANGLE_AREA = IMG_SIZE * IMG_SIZE * 0.1
    # 核心修改：定义重心圆的半径
    CENTROID_CIRCLE_RADIUS = 10


# ==============================================================================
# --- 核心几何与绘图代码 ---
# ==============================================================================

def get_triangle_area(p1, p2, p3):
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def calculate_centroid(p1, p2, p3):
    """根据三个顶点计算重心坐标"""
    # 公式: Gx = (Ax + Bx + Cx) / 3
    center_x = (p1[0] + p2[0] + p3[0]) / 3
    center_y = (p1[1] + p2[1] + p3[1]) / 3
    return (center_x, center_y)


def draw_image(config, vertices, centroid_params=None):
    img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 1. 绘制实心绿色三角形
    draw.polygon(vertices, fill=config.TRIANGLE_COLOR, outline=None)

    # 2. 如果提供了重心参数，绘制红色实心圆
    if centroid_params:
        center = centroid_params
        radius = config.CENTROID_CIRCLE_RADIUS
        x, y = center
        bounding_box = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bounding_box, fill=config.CENTROID_CIRCLE_COLOR, outline=None)

    return img


def create_single_sample(config):
    while True:
        padding = 10
        vertices = [
            (random.randint(padding, config.IMG_SIZE - padding), random.randint(padding, config.IMG_SIZE - padding))
            for _ in range(3)
        ]
        if get_triangle_area(vertices[0], vertices[1], vertices[2]) > config.MIN_TRIANGLE_AREA:
            break

    # 计算重心的精确坐标
    centroid_coords = calculate_centroid(vertices[0], vertices[1], vertices[2])

    input_image = draw_image(config, vertices)
    output_image = draw_image(config, vertices, centroid_coords)

    return input_image, output_image


def generate_dataset(num_samples, output_dir, name, config):
    print(f"\n正在生成 {name} ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_samples), desc=f"生成 {name}"):
        input_img, output_img = create_single_sample(config)
        input_img.save(Path(output_dir) / f"{i}_input.png")
        output_img.save(Path(output_dir) / f"{i}_output.png")


if __name__=="__main__":
    cfg = Config()

    if os.path.exists(cfg.DATA_DIR):
        print(f"发现旧数据目录 '{cfg.DATA_DIR}', 正在删除...")
        shutil.rmtree(cfg.DATA_DIR)

    print("=" * 60)
    print("开始生成几何推理（重心）数据集...")
    print(f"图像尺寸: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
    print(f"重心圆半径: {cfg.CENTROID_CIRCLE_RADIUS}")
    print("=" * 60)

    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
    # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)

    print("\n🎉🎉🎉 重心数据集生成完毕！ 🎉🎉🎉")

