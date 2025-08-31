import os
import random
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 (已更新) ---
# ==============================================================================
IMAGE_SIZE = (256, 256)
NUM_SAMPLES = 200
DATASET_DIR = "./triangle_dataset"
BACKGROUND_COLOR = "white"
TRIANGLE_COLOR = (0,255,0)

# --- 新增：最小面积阈值 ---
# 一个 256x256 的图像，总面积为 65536。
# 500 大约是总面积的 0.76%，可以避免生成特别细小的三角形。
MIN_TRIANGLE_AREA = 8000  # <-- 新增：面积过滤器


# ==============================================================================
# --- 2. 辅助函数 (已更新) ---
# ==============================================================================

def is_clockwise(p1, p2, p3):
    """使用向量叉积的z分量判断三点是否为顺时针。"""
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    return val < 0


def calculate_area(p1, p2, p3):  # <-- 新增：面积计算函数
    """使用海伦公式或行列式（Shoelace formula）计算三角形面积。"""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def generate_clockwise_points(max_coord, min_area):  # <-- 修改：增加min_area参数
    """生成三个随机点，确保它们是顺时针且面积大于阈值。"""
    while True:
        points = [(random.randint(0, max_coord), random.randint(0, max_coord)) for _ in range(3)]
        p1, p2, p3 = points

        # 避免三点共线或过于接近的情况
        if len(set(points)) < 3:
            continue

        # --- 核心修改：检查面积 ---
        if calculate_area(p1, p2, p3) < min_area:  # <-- 修改：如果面积太小，则跳过
            continue

        if is_clockwise(p1, p2, p3):
            return points
        elif is_clockwise(p1, p3, p2):
            return [p1, p3, p2]
        # 如果共线(val=0)，则重新生成


def coord_to_binary(coord, bits=8):
    """将一个坐标值转换为指定位数的二进制字符串。"""
    return format(coord, f'0{bits}b')


# ==============================================================================
# --- 3. 主生成函数 (已更新) ---
# ==============================================================================

def generate_dataset():
    """主函数，生成完整的数据集。"""
    images_dir = os.path.join(DATASET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []
    print(f"🚀 开始生成 {NUM_SAMPLES} 个三角形样本 (最小面积: {MIN_TRIANGLE_AREA} 像素)...")

    for i in tqdm(range(NUM_SAMPLES), desc="Generating Data"):
        # 1. 生成三个满足条件的点
        points = generate_clockwise_points(IMAGE_SIZE[0] - 1, MIN_TRIANGLE_AREA)  # <-- 修改：传入最小面积

        # 2. 创建二进制标签
        label_parts = []
        for p in points:
            x_bin = coord_to_binary(p[0])
            y_bin = coord_to_binary(p[1])
            label_parts.extend([x_bin, y_bin])
        label = "".join(label_parts)

        # 3. 绘制图像
        image = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        draw.polygon(points, fill=TRIANGLE_COLOR)

        # 4. 保存图像和元数据
        filename = f"triangle_{i}.png"
        image_path = os.path.join(images_dir, filename)
        image.save(image_path)

        metadata.append({
            "filename": os.path.join("images", filename),
            "label": label
        })

    # 5. 保存metadata.csv
    df = pd.DataFrame(metadata)
    captions_path = os.path.join(DATASET_DIR, "metadata.csv")
    df.to_csv(captions_path, index=False)

    print("\n✅ 数据集生成完毕！")
    print(f"   -> 图像保存在: {images_dir}")
    print(f"   -> 标签保存在: {captions_path}")
    print("\n示例数据:")
    print(df.head())


if __name__=="__main__":
    generate_dataset()
