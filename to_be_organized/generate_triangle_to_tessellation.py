import os
import random
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import math

# --- 配置参数 ---
IMG_SIZE = 224  # 图像尺寸 (宽和高)
NUM_SAMPLES = 150000  # 生成的样本数量
OUTPUT_DIR = "tessellation_dataset_256"  # 输出目录

# 定义颜色 (R, G, B)
COLOR_BACKGROUND = (255, 255, 255)  # 白色
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)  # 备用颜色，用于更复杂的镶嵌


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

    return image


def create_input_image(img_size, triangle_pts):
    """创建只包含单个绿色三角形的输入图像"""
    image = Image.new("RGB", (img_size, img_size), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_triangle(draw, triangle_pts, COLOR_GREEN)
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

def generate_dataset():
    """生成整个数据集"""
    input_dir = os.path.join(OUTPUT_DIR, "input")
    output_dir = os.path.join(OUTPUT_DIR, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始生成 {NUM_SAMPLES} 个样本到目录 '{OUTPUT_DIR}'...")

    for i in tqdm(range(NUM_SAMPLES)):
        # --- 随机生成一个“良好形态”的基准三角形 ---
        # 我们在图像中心附近的一个区域内随机选择顶点
        # 确保所有坐标都是整数！
        # center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        #
        # # 限制随机范围，避免三角形过大或过小
        # margin = IMG_SIZE // 4
        # w_range = (30, IMG_SIZE // 3)
        # h_range = (30, IMG_SIZE // 3)
        # x_skew_range = (-IMG_SIZE // 4, IMG_SIZE // 4)
        # y_skew_range = (-IMG_SIZE // 8, IMG_SIZE // 8)
        #
        # # 定义平行四边形的几何参数 (整数)
        # w = random.randint(*w_range)
        # h = random.randint(*h_range)
        # x_skew = random.randint(*x_skew_range)
        # y_skew = random.randint(*y_skew_range)
        #
        # # 定义平行四边形的四个顶点相对于中心的位置
        # # A, B, C, D
        # # 我们让B点作为锚点，位于中心
        # p_b_local = np.array([0, 0])
        # p_a_local = p_b_local - np.array([w, y_skew])
        # p_c_local = p_b_local + np.array([x_skew, h])
        #
        # # 将整个平行四边形移动到图像中心
        # offset = np.array([center_x, center_y])
        #
        # p_a = p_a_local + offset
        # p_b = p_b_local + offset
        # p_c = p_c_local + offset

        base_triangle_pts = generate_base_triangle()#[p_a, p_b, p_c]

        # --- 生成输入和输出图像 ---
        input_image = create_input_image(IMG_SIZE, base_triangle_pts)
        output_image = generate_perfect_tessellation(IMG_SIZE, base_triangle_pts)

        # --- 保存图像 ---
        filename = f"sample_{i:05d}.png"
        input_image.save(os.path.join(input_dir, filename))
        output_image.save(os.path.join(output_dir, filename))

    print("数据集生成完毕！")


if __name__=="__main__":
    generate_dataset()

