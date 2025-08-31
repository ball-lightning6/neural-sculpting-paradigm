import os
import numpy as np
import random
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- 全局配置 ---
# 您可以在这里微调所有几何和外观参数
CONFIG = {
    "image_size": 256,
    "min_triangle_area": 1200,
    "min_angle_deg": 20,
    "min_distance": 15,
    "line_thickness": 2,
    "triangle_color": (0, 200, 0),  # 绿色 (RGB)
    "line_color": (255, 0, 0),  # 红色 (RGB)
    "background_color": (255, 255, 255)  # 白色
}


def get_triangle_angles(p1, p2, p3):
    """计算三角形三个角的度数"""
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.degrees(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
        beta = np.degrees(np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
        gamma = np.degrees(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
    return [angle for angle in [alpha, beta, gamma] if not np.isnan(angle)]


def generate_valid_triangle(config):
    """通过拒绝采样生成一个满足所有约束的有效三角形。"""
    while True:
        points = np.random.randint(
            int(config["image_size"] * 0.05),
            int(config["image_size"] * 0.95),
            size=(3, 2)
        )
        triangle_poly = Polygon(points)
        if triangle_poly.area < config["min_triangle_area"]:
            continue
        angles = get_triangle_angles(points[0], points[1], points[2])
        if not angles or min(angles) < config["min_angle_deg"]:
            continue
        return triangle_poly


def draw_image_with_pil(triangles, line_segment=None, config=None):
    """根据给定的几何体，使用 Pillow 绘制图像"""
    img = Image.new(
        'RGB',
        (config["image_size"], config["image_size"]),
        config["background_color"]
    )
    draw = ImageDraw.Draw(img)

    for tri in triangles:
        coords = list(tri.exterior.coords)
        draw.polygon(coords, fill=config["triangle_color"])

    if line_segment:
        p1, p2 = line_segment
        line_coords = [(p1.x, p1.y), (p2.x, p2.y)]
        draw.line(
            line_coords,
            fill=config["line_color"],
            width=config["line_thickness"]
        )

    return img


def worker_task(sample_idx, output_dir, config):
    """一个工作进程的任务：生成一对有效的 (输入, 输出) 图像并保存。"""
    while True:
        tri1 = generate_valid_triangle(config)
        tri2 = generate_valid_triangle(config)

        if tri1.intersects(tri2):
            continue
        if tri1.distance(tri2) < config["min_distance"]:
            continue

        break

    p1, p2 = nearest_points(tri1, tri2)

    input_image = draw_image_with_pil([tri1, tri2], line_segment=None, config=config)
    output_image = draw_image_with_pil([tri1, tri2], line_segment=(p1, p2), config=config)

    input_filename = os.path.join(output_dir, "inputs", f"{sample_idx:06d}.png")
    output_filename = os.path.join(output_dir, "outputs", f"{sample_idx:06d}.png")

    input_image.save(input_filename)
    output_image.save(output_filename)


def generate_dataset(num_samples, output_dir, config):
    """主函数，用于并行生成整个数据集"""
    print("--- 开始生成几何推理数据集 (使用 Pillow) ---")
    print(f"  - 目标样本数: {num_samples}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 图像尺寸: {config['image_size']}x{config['image_size']}")

    input_path = os.path.join(output_dir, "inputs")
    output_path = os.path.join(output_dir, "outputs")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # 在某些系统上，为了让多进程正常工作，最好保留 if __name__ == '__main__': 的结构
    # 但我们可以在这个结构内直接设置参数
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    print(f"  - 使用 {num_processes} 个工作进程。")

    task = partial(worker_task, output_dir=output_dir, config=config)

    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(task, range(num_samples)), total=num_samples, desc="Generating Samples"))

    print("\n✅ 数据集生成完毕！")


# ★★★ 主执行块：直接在此处修改参数 ★★★
if __name__=='__main__':
    # 为了在多进程环境中安全运行，我们保留 if __name__ == '__main__'
    # 但将参数配置移到这里，方便在 Notebook 中直接修改和运行。

    # --- 在这里直接设置参数 ---
    NUM_SAMPLES = 1000  # 要生成的样本数量
    OUTPUT_DIR = "triangle_dataset_pil"  # 数据集的输出目录

    # --- 执行生成任务 ---
    generate_dataset(
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR,
        config=CONFIG
    )
