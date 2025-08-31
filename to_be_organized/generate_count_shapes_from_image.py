import os
import random
import math
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# --- 配置参数 ---
# 图像设置
IMG_SIZE = 224
BACKGROUND_COLOR = (255, 255, 255)  # 白色

# 形状和颜色定义
SHAPES = ['square', 'circle', 'triangle']
COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

# 数量和大小控制
MIN_COUNT_PER_CATEGORY = 0
MAX_COUNT_PER_CATEGORY = 3
MIN_SHAPE_SIZE = 20  # 形状的最小尺寸 (像素)
MAX_SHAPE_SIZE = 50  # 形状的最大尺寸 (像素)

# 生成设置
TOTAL_SAMPLES = 1000  # 你可以根据需要调整数据集大小
OUTPUT_DIR = 'counting_dataset'
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
LABEL_DIR = os.path.join(OUTPUT_DIR, 'labels')


# --- 辅助函数 ---

def draw_square(draw, position, size, color):
    """在指定位置绘制一个正方形"""
    cx, cy = position
    half_size = size / 2
    draw.rectangle(
        [cx - half_size, cy - half_size, cx + half_size, cy + half_size],
        fill=color
    )


def draw_circle(draw, position, size, color):
    """在指定位置绘制一个圆形, size为直径"""
    cx, cy = position
    half_size = size / 2
    draw.ellipse(
        [cx - half_size, cy - half_size, cx + half_size, cy + half_size],
        fill=color
    )


def draw_triangle(draw, position, size, color):
    """在指定位置绘制一个等边三角形, size为边长, 一个角朝上"""
    cx, cy = position
    height = (math.sqrt(3) / 2) * size
    p1 = (cx, cy - (2 / 3) * height)
    p2 = (cx - size / 2, cy + (1 / 3) * height)
    p3 = (cx + size / 2, cy + (1 / 3) * height)
    draw.polygon([p1, p2, p3], fill=color)


def check_overlap(new_bbox, existing_bboxes):
    """检查新的边界框是否与已存在的边界框重叠"""
    for bbox in existing_bboxes:
        # AABB aabb collision detection
        if (new_bbox[0] < bbox[2] and new_bbox[2] > bbox[0] and
                new_bbox[1] < bbox[3] and new_bbox[3] > bbox[1]):
            return True
    return False


def get_bbox(position, size):
    """获取形状的边界框"""
    cx, cy = position
    half_size = size / 2
    return (cx - half_size, cy - half_size, cx + half_size, cy + half_size)


# --- 主生成逻辑 ---

def create_sample():
    """创建一个图像和对应的标签"""
    # 1. 初始化空白图像和计数器
    image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    shape_counts = {shape: 0 for shape in SHAPES}
    color_counts = {color: 0 for color in COLORS}

    # 2. 随机决定每种形状和颜色的目标数量
    target_shape_counts = {shape: random.randint(MIN_COUNT_PER_CATEGORY, MAX_COUNT_PER_CATEGORY) for shape in SHAPES}

    all_shapes_to_draw = []
    for shape, count in target_shape_counts.items():
        for _ in range(count):
            all_shapes_to_draw.append(shape)

    random.shuffle(all_shapes_to_draw)

    drawn_bboxes = []

    # 3. 绘制形状
    for shape_type in all_shapes_to_draw:
        # 为当前形状随机选择一个颜色
        color_name = random.choice(list(COLORS.keys()))
        color_rgb = COLORS[color_name]

        # 尝试最多100次来放置形状，避免死循环
        for _ in range(100):
            size = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
            # 确保形状完整地在图像内
            margin = int(size / 2) + 2
            position = (
                random.randint(margin, IMG_SIZE - margin),
                random.randint(margin, IMG_SIZE - margin)
            )

            bbox = get_bbox(position, size)

            if not check_overlap(bbox, drawn_bboxes):
                # 找到不重叠的位置，开始绘制
                if shape_type=='square':
                    draw_square(draw, position, size, color_rgb)
                elif shape_type=='circle':
                    draw_circle(draw, position, size, color_rgb)
                elif shape_type=='triangle':
                    draw_triangle(draw, position, size, color_rgb)

                # 更新计数器和已绘制列表
                shape_counts[shape_type] += 1
                color_counts[color_name] += 1
                drawn_bboxes.append(bbox)
                break  # 成功放置，跳出尝试循环

    # 4. 生成12-bit标签
    # 形状部分 (6-bit)
    square_bits = format(shape_counts['square'], '02b')
    circle_bits = format(shape_counts['circle'], '02b')
    triangle_bits = format(shape_counts['triangle'], '02b')

    # 颜色部分 (6-bit)
    red_bits = format(color_counts['red'], '02b')
    green_bits = format(color_counts['green'], '02b')
    blue_bits = format(color_counts['blue'], '02b')

    label = f"{square_bits}{circle_bits}{triangle_bits}{red_bits}{green_bits}{blue_bits}"

    return image, label


# --- 运行脚本 ---
if __name__=='__main__':
    # 创建输出目录
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    print(f"开始生成 {TOTAL_SAMPLES} 个样本到 '{OUTPUT_DIR}' 目录...")

    for i in tqdm(range(TOTAL_SAMPLES), desc="生成数据"):
        image, label = create_sample()

        # 保存图像和标签
        image.save(os.path.join(IMG_DIR, f"{i}.png"))
        with open(os.path.join(LABEL_DIR, f"{i}.txt"), 'w') as f:
            f.write(label)

    print("\n数据生成完毕!")

    # 打印一个样本以供检查
    print("\n--- 样本检查 ---")
    img, lbl = create_sample()
    img.save(os.path.join(OUTPUT_DIR, "sample_check.png"))
    print(f"已生成一个检查样本: 'sample_check.png'")
    print(f"对应的标签是: {lbl}")
    print("\n标签格式解读 (12-bit):")
    print(f" [0:2] 方块数: {int(lbl[0:2], 2)} ({lbl[0:2]})")
    print(f" [2:4] 圆形数: {int(lbl[2:4], 2)} ({lbl[2:4]})")
    print(f" [4:6] 三角形数: {int(lbl[4:6], 2)} ({lbl[4:6]})")
    print(f" [6:8] 红色数: {int(lbl[6:8], 2)} ({lbl[6:8]})")
    print(f" [8:10] 绿色数: {int(lbl[8:10], 2)} ({lbl[8:10]})")
    print(f"[10:12] 蓝色数: {int(lbl[10:12], 2)} ({lbl[10:12]})")

