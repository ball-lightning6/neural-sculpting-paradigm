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
    """配置类，集中管理所有生成参数。"""
    # --- 目录和样本数量 ---
    DATA_DIR = "symmetry_axis_dataset_FULLY_CONTAINED"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 500
    NUM_SAMPLES_EVAL = 500

    # --- 图像属性 ---
    IMG_SIZE = 224
    BG_COLOR = (255, 255, 255)
    SHAPE_COLOR = (0, 255, 0)
    AXIS_COLOR = (255, 0, 0)

    # --- 几何约束 ---
    MIN_POLYGON_VERTICES_HALF = 3
    MAX_POLYGON_VERTICES_HALF = 5  # 减少最大值以更容易生成
    AXIS_WIDTH = 10
    CONCAVE_PROBABILITY = 0.5
    MIN_SHAPE_AREA_RATIO = 0.03  # 图形最小面积占总面积的比例


# ==============================================================================
# --- 核心几何与绘图函数 (Core Geometry and Drawing Functions) ---
# ==============================================================================

def reflect_point(point, line_p1, line_p2):
    p = np.array(point, dtype=float)
    p1 = np.array(line_p1, dtype=float)
    p2 = np.array(line_p2, dtype=float)
    line_vec = p2 - p1
    p_vec = p - p1
    line_vec_sq_mag = np.dot(line_vec, line_vec)
    if line_vec_sq_mag < 1e-9: return tuple(p)
    t = np.dot(p_vec, line_vec) / line_vec_sq_mag
    projection_point = p1 + t * line_vec
    reflected_point = 2 * projection_point - p
    return tuple(reflected_point)


def get_line_endpoints_on_boundary(p1, p2, size):
    p1 = np.array(p1, dtype=float);
    p2 = np.array(p2, dtype=float)
    direction = p2 - p1
    if np.linalg.norm(direction) < 1e-9: return None
    intersections = []
    for boundary in [0, size - 1]:
        if abs(direction[0]) > 1e-9:
            t = (boundary - p1[0]) / direction[0]
            y = p1[1] + t * direction[1]
            if -1e-9 <= y < size + 1e-9: intersections.append((boundary, y))
        if abs(direction[1]) > 1e-9:
            t = (boundary - p1[1]) / direction[1]
            x = p1[0] + t * direction[0]
            if -1e-9 <= x < size + 1e-9: intersections.append((x, boundary))
    unique_intersections = []
    for p in intersections:
        if not any(np.linalg.norm(np.array(p) - np.array(up)) < 1e-5 for up in unique_intersections):
            unique_intersections.append(p)
    return tuple(unique_intersections[:2]) if len(unique_intersections) >= 2 else None


def draw_image(config, polygon_vertices, axis_line_endpoints=None):
    img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BG_COLOR)
    draw = ImageDraw.Draw(img)
    if polygon_vertices:
        draw.polygon(polygon_vertices, fill=config.SHAPE_COLOR, outline=None)
    if axis_line_endpoints:
        draw.line(axis_line_endpoints, fill=config.AXIS_COLOR, width=config.AXIS_WIDTH)
    return img


def polygon_area(vertices):
    """使用鞋带公式计算多边形面积"""
    if len(vertices) < 3: return 0
    x = np.array([v[0] for v in vertices])
    y = np.array([v[1] for v in vertices])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def create_single_sample(config):
    """生成一个独立的、图形完全在边界内的样本。"""
    size = config.IMG_SIZE

    sample_generation_attempts = 0
    while sample_generation_attempts < 500:  # 如果500次都找不到合适的轴和点，就报错
        sample_generation_attempts += 1

        # 1. 随机定义对称轴
        axis_p1 = (random.randint(0, size - 1), random.randint(0, size - 1))
        axis_p2 = (random.randint(0, size - 1), random.randint(0, size - 1))
        if np.linalg.norm(np.array(axis_p1) - np.array(axis_p2)) < size / 4: continue

        boundary_endpoints = get_line_endpoints_on_boundary(axis_p1, axis_p2, size)
        if not boundary_endpoints: continue

        # 2. 生成半边顶点 (核心修正逻辑)
        num_verts_half = random.randint(config.MIN_POLYGON_VERTICES_HALF, config.MAX_POLYGON_VERTICES_HALF)
        half_vertices = []
        generation_successful = True

        for _ in range(num_verts_half):
            point_generation_attempts = 0
            found_valid_point = False
            while point_generation_attempts < 200:  # 尝试200次找到一个有效的点
                point_generation_attempts += 1
                p = (random.randint(0, size - 1), random.randint(0, size - 1))

                cross_product = (axis_p2[0] - axis_p1[0]) * (p[1] - axis_p1[1]) - (axis_p2[1] - axis_p1[1]) * (
                            p[0] - axis_p1[0])
                if cross_product > 0:  # 确保点在轴的一侧
                    p_reflected = reflect_point(p, axis_p1, axis_p2)
                    rx, ry = p_reflected

                    # *** 双重约束检查 ***
                    if 0 <= rx < size and 0 <= ry < size:
                        half_vertices.append(p)
                        found_valid_point = True
                        break  # 找到了一个有效点，跳出内层循环

            if not found_valid_point:
                generation_successful = False
                break  # 找不到有效点，放弃这个样本

        if not generation_successful or len(half_vertices) < num_verts_half:
            continue  # 重新开始生成一个新样本

        # 3. 构造完整的多边形
        if random.random() < config.CONCAVE_PROBABILITY:
            random.shuffle(half_vertices)
        else:
            centroid = np.mean(half_vertices, axis=0)
            half_vertices.sort(key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))

        reflected_vertices = [reflect_point(p, axis_p1, axis_p2) for p in half_vertices]
        full_polygon = half_vertices + reflected_vertices[::-1]

        # 4. 最终检查面积
        area = polygon_area(full_polygon)
        if area < (size * size * config.MIN_SHAPE_AREA_RATIO):
            continue

        # 所有检查通过，这是一个完美的样本
        input_image = draw_image(config, full_polygon)
        output_image = draw_image(config, full_polygon, axis_line_endpoints=boundary_endpoints)
        return input_image, output_image

    raise RuntimeError("无法在合理尝试次数内生成有效样本，请检查几何约束。")


def generate_dataset(num_samples, output_dir, name, config):
    print(f"\n正在生成 {name} 数据集 ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_samples), desc=f"生成 {name}", unit="张"):
        input_img, output_img = create_single_sample(config)
        input_img.save(Path(output_dir) / f"{i}_input.png")
        output_img.save(Path(output_dir) / f"{i}_output.png")


# ==============================================================================
# --- 程序入口 (Main Execution Block) ---
# ==============================================================================

if __name__=="__main__":
    cfg = Config()
    if os.path.exists(cfg.DATA_DIR):
        print(f"发现旧数据目录 '{cfg.DATA_DIR}', 正在删除...")
        shutil.rmtree(cfg.DATA_DIR)

    print("=" * 60)
    print("开始生成几何推理数据集：完全内含的对称图形 v4.0")
    print(f"数据集目录: {cfg.DATA_DIR}")
    print("约束: 所有生成的对称图形将100%位于图像边界之内。")
    print("=" * 60)

    try:
        generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
        # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)
        print(f"\n🎉🎉🎉 数据集 '{cfg.DATA_DIR}' 生成完毕！ 🎉🎉🎉")
    except RuntimeError as e:
        print(f"\n❌ 生成失败: {e}")
