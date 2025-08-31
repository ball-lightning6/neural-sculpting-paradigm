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
    DATA_DIR = "incircle_dataset"
    TRAIN_DIR = DATA_DIR#os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 150000  # 几何任务可能需要更多样本来学习
    NUM_SAMPLES_EVAL = 500

    # --- 图像属性 ---
    IMG_SIZE = 224  # 使用224x224，与常见模型输入尺寸一致
    BG_COLOR = (255, 255, 255)  # 背景 - 白色
    TRIANGLE_COLOR = (0, 255, 0)  # 三角形 - 绿色
    INCIRCLE_COLOR = (255, 0, 0)  # 内切圆 - 红色

    # --- 几何约束 ---
    MIN_TRIANGLE_AREA = IMG_SIZE * IMG_SIZE * 0.1  # 确保三角形不会太小


# ==============================================================================
# --- 核心几何与绘图代码 ---
# ==============================================================================

def get_triangle_area(p1, p2, p3):
    """使用行列式计算三角形面积"""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def calculate_incircle(p1, p2, p3):
    """
    根据三个顶点计算内切圆的圆心和半径。
    Args:
        p1, p2, p3: (x, y) 格式的元组.
    Returns:
        (center_x, center_y), radius
    """
    # 计算三边长
    a = np.linalg.norm(np.array(p2) - np.array(p3))
    b = np.linalg.norm(np.array(p1) - np.array(p3))
    c = np.linalg.norm(np.array(p1) - np.array(p2))

    # 计算内心坐标 (Incenter)
    # 公式: Ix = (a*Ax + b*Bx + c*Cx) / (a+b+c)
    incenter_x = (a * p1[0] + b * p2[0] + c * p3[0]) / (a + b + c)
    incenter_y = (a * p1[1] + b * p2[1] + c * p3[1]) / (a + b + c)

    # 计算半径 (Inradius)
    # 公式: r = 2 * Area / Perimeter
    perimeter = a + b + c
    area = get_triangle_area(p1, p2, p3)
    radius = 2 * area / perimeter

    return (incenter_x, incenter_y), radius


def draw_image(config, vertices, incircle_params=None):
    """
    绘制图像：输入（只有三角形）或输出（三角形+内切圆）
    """
    img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 1. 绘制实心绿色三角形
    draw.polygon(vertices, fill=config.TRIANGLE_COLOR, outline=None)

    # 2. 如果提供了内切圆参数，绘制红色实心内切圆
    if incircle_params:
        center, radius = incircle_params
        x, y = center
        # Ellipse takes a bounding box [x0, y0, x1, y1]
        bounding_box = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bounding_box, fill=config.INCIRCLE_COLOR, outline=None)

    return img


def create_single_sample(config):
    """生成一个有效的几何样本"""
    while True:
        # 1. 在图像边界内随机生成三个顶点
        padding = 10  # 确保顶点不会紧贴边缘
        vertices = [
            (random.randint(padding, config.IMG_SIZE - padding), random.randint(padding, config.IMG_SIZE - padding))
            for _ in range(3)
        ]

        # 2. 检查三角形是否有效（不是一条线，面积足够大）
        if get_triangle_area(vertices[0], vertices[1], vertices[2]) > config.MIN_TRIANGLE_AREA:
            break  # 这是一个好三角形，跳出循环

    # 3. 计算内切圆的精确参数
    incircle_params = calculate_incircle(vertices[0], vertices[1], vertices[2])

    # 4. 生成输入和输出图像
    input_image = draw_image(config, vertices)
    output_image = draw_image(config, vertices, incircle_params)

    return input_image, output_image


def generate_dataset(num_samples, output_dir, name, config):
    """生成数据集的主函数"""
    print(f"\n正在生成 {name} ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)
    for i in tqdm(range(num_samples), desc=f"生成 {name}"):
        input_img, output_img = create_single_sample(config)
        input_img.save(Path(output_dir) /"input"/ f"{i:06d}.png")
        output_img.save(Path(output_dir) /"output"/ f"{i:06d}.png")


if __name__=="__main__":
    cfg = Config()

    if os.path.exists(cfg.DATA_DIR):
        print(f"发现旧数据目录 '{cfg.DATA_DIR}', 正在删除...")
        shutil.rmtree(cfg.DATA_DIR)

    print("=" * 60)
    print("开始生成几何推理（内切圆）数据集...")
    print(f"图像尺寸: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
    print("=" * 60)

    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
    # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)

    print("\n🎉🎉🎉 几何数据集生成完毕！ 🎉🎉🎉")

