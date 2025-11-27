import os
import random
import csv
import math
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# --- 配置参数 ---
IMG_SIZE = 256  # 图像尺寸 (宽和高)
NUM_SAMPLES = 150000  # 生成的样本数量
OUTPUT_DIR = "tessellation_coords_dataset_256"  # 输出目录

# 定义颜色 (R, G, B)
COLOR_BACKGROUND = (255, 255, 255)  # 白色
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)  # 备用颜色，用于更复杂的镶嵌
COLOR_BLACK = (0, 0, 0)   # 基准三角形颜色

# --- 核心函数 ---

def draw_triangle(draw, points, color):
    """在给定的draw对象上绘制一个实心三角形"""
    # Pillow的polygon需要一个扁平的点列表
    draw.polygon([(p[0], p[1]) for p in points], fill=color)


def generate_perfect_tessellation(img_size, base_triangle_pts):
    """
    根据一个基准三角形生成完美的平面镶嵌图。
    这个函数是整个脚本的核心，它实现了“晶体生长”算法。
    """
    # 创建一张白色背景的输出图像
    image = Image.new("RGB", (img_size, img_size), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(image)

    # 从输入的三个点，计算出平行四边形的第四个点
    # A, B, C -> D = A + (C - B)
    p_a, p_b, p_c = base_triangle_pts
    p_d = p_a + (p_c - p_b)

    # 定义两个平移向量，它们构成了我们的晶格基础
    # Vector 1: 从A到B
    # Vector 2: 从A到D (等价于从B到C)
    vec1 = p_b - p_a
    vec2 = p_d - p_a

    # 确定需要平铺的范围
    # 为了覆盖整个画布，我们需要计算在每个方向上需要多少个“晶胞”
    # 我们从中心向外扩展，所以范围要比img_size / 晶胞尺寸 稍大一些
    # 这是一个经验性的范围，确保能覆盖到角落
    range_x = int(img_size / np.linalg.norm(vec1))*2 + 2 if np.linalg.norm(vec1) > 0 else 1
    range_y = int(img_size / np.linalg.norm(vec2))*2 + 2 if np.linalg.norm(vec2) > 0 else 1

    # 开始平铺
    for i in range(-range_x, range_x + 1):
        for j in range(-range_y, range_y + 1):
            # 计算当前晶胞的平移向量
            translation = i * vec1 + j * vec2

            # 计算平移后的两个三角形的顶点
            # 第一个三角形 (类似初始的绿色三角形)
            tri1_pts = [p_a + translation, p_b + translation, p_c + translation]

            # 第二个三角形 (类似初始的红色三角形)
            # 它的顶点是 B, D, C
            tri2_pts = [p_a + translation, p_d + translation, p_c + translation]

            # 简单的染色逻辑 (可以根据需要变得更复杂)
            # 这里我们让 (i+j) 的奇偶性决定颜色，形成棋盘格模式
            if 1:#(i + j) % 2==0:
                color1 = COLOR_GREEN
                color2 = COLOR_RED
            else:
                color1 = COLOR_RED
                color2 = COLOR_GREEN  # 或者使用 COLOR_BLUE

            draw_triangle(draw, tri1_pts, color1)
            draw_triangle(draw, tri2_pts, color2)

    # --- 关键修改：最后绘制基准三角形并涂黑 ---
    # 基准三角形就是 i=0, j=0 时的 tri1 (A, B, C)
    # 也就是传入的 base_triangle_pts
    draw_triangle(draw, base_triangle_pts, COLOR_BLACK)

    return image


# --- 主生成逻辑 ---
def generate_base_triangle():
    """
    按照您的新规范，生成一个“良好形态”的三角形。
    1. 在原点附近构建一个锐角三角形。
    2. 随机旋转。
    3. 随机平移到图像中心区域。
    """

    # 1. 在原点坐标系 (0,0) 构建三角形

    # 点B固定在原点
    p_b_local = np.array([0.0, 0.0])

    # 点A在x轴正半轴上随机选择，控制底边长度
    base_length_range = (60, IMG_SIZE // 3)  # (min, max) 长度
    base_length = random.uniform(*base_length_range)
    p_a_local = np.array([base_length, 0.0])

    # 点C的生成是关键，要确保是锐角三角形
    # 为了保证锐角，点C的x坐标必须在A和B的x坐标之间，即 (0, base_length)
    # 并且角度不能太小或太大
    c_x_range = (base_length * 0.2, base_length * 0.8)  # 限制x坐标，避免直角或钝角
    c_y_range = (base_length * 0.5, base_length * 0.8)  # 限制y坐标，避免细长

    p_c_local_x = random.uniform(*c_x_range)
    p_c_local_y = random.uniform(*c_y_range)
    # y坐标随机为正或负，增加多样性
    if random.choice([True, False]):
        p_c_local_y *= -1

    p_c_local = np.array([p_c_local_x, p_c_local_y])

    # 此时我们有了一个局部坐标下的良好三角形 (p_a_local, p_b_local, p_c_local)
    local_points = np.array([p_a_local, p_b_local, p_c_local])

    # 2. 随机旋转
    angle = random.uniform(0, 2 * math.pi)  # 旋转0到360度
    rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])
    rotated_points = np.dot(local_points, rotation_matrix.T)

    # 3. 随机平移到图像中心区域
    # 在中心附近的一个小方框内随机选择一个最终的锚点
    center_margin = IMG_SIZE // 8
    center_x = IMG_SIZE // 2 + random.randint(-center_margin, center_margin)
    center_y = IMG_SIZE // 2 + random.randint(-center_margin, center_margin)
    translation_vector = np.array([center_x, center_y])

    final_points = rotated_points + translation_vector

    # 4. 将浮点数坐标转换为整数坐标，准备绘图
    base_triangle_pts = final_points.astype(int)

    return base_triangle_pts

def coords_to_binary(points):
    """将坐标转换为48位二进制字符串 (3点 * 2坐标 * 8位)"""
    binary_str = ""
    for point in points:
        x, y = point
        # 裁剪到 0-255 范围
        x = max(0, min(255, int(x)))
        y = max(0, min(255, int(y)))
        binary_str += f"{x:08b}{y:08b}"
    return binary_str

def generate_dataset():
    """生成整个数据集"""
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []

    print(f"开始生成 {NUM_SAMPLES} 个样本到目录 '{OUTPUT_DIR}'...")

    for i in tqdm(range(NUM_SAMPLES)):
        # --- 随机生成一个“良好形态”的基准三角形 ---
        base_triangle_pts = generate_base_triangle()

        # --- 生成输入标签 (48-bit string) ---
        input_binary = coords_to_binary(base_triangle_pts)

        # --- 生成输出图像 (镶嵌图案，基准三角形染黑) ---
        output_image = generate_perfect_tessellation(IMG_SIZE, base_triangle_pts)

        # --- 保存图像 ---
        filename = f"sample_{i:06d}.png"
        output_image.save(os.path.join(images_dir, filename))

        # 记录元数据
        metadata.append({
            'input_text': input_binary,
            'output_image': os.path.join("images", filename)
        })

    # 保存元数据
    with open(os.path.join(OUTPUT_DIR, 'metadata.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['input_text', 'output_image'])
        writer.writeheader()
        writer.writerows(metadata)

    print("数据集生成完毕！")


if __name__=="__main__":
    generate_dataset()