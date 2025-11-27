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
    DATA_DIR = "refraction_dataset_v1.3_USER_LOGIC"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 500
    NUM_SAMPLES_EVAL = 1000

    IMG_SIZE = 224

    AIR_COLOR = (0, 0, 255)
    MEDIUM_COLOR = (255, 255, 255)
    INCIDENT_RAY_COLOR = (0, 0, 0)
    REFRACTED_RAY_COLOR = (255, 0, 0)
    LINE_WIDTH = 10

    N_AIR = 1.0
    N_MEDIUM = 1.5


# ==============================================================================
# --- 几何与物理计算辅助函数 (部分复用) ---
# ==============================================================================

def get_intersection_with_box(p1, p2, box_size):
    v = p2 - p1
    intersections = []
    for edge_val, axis_idx in [(0, 1), (box_size, 1), (0, 0), (box_size, 0)]:
        if abs(v[axis_idx]) < 1e-6: continue
        t = (edge_val - p1[axis_idx]) / v[axis_idx]
        if t > 1e-4:
            intersection_point = p1 + t * v
            other_axis_idx = 1 - axis_idx
            if -1e-3 <= intersection_point[other_axis_idx] <= box_size + 1e-3:
                intersections.append(intersection_point)
    if not intersections: return p2
    intersections.sort(key=lambda p: np.linalg.norm(p - p2))
    return intersections[0]


def get_boundary_polygon(boundary_normal, boundary_point, size):
    A, B = boundary_normal
    C = -np.dot(boundary_normal, boundary_point)
    corners = [np.array([0, 0]), np.array([size, 0]), np.array([size, size]), np.array([0, size])]
    polygon_vertices = [p for p in corners if np.dot(A, p[0]) + np.dot(B, p[1]) + C >= 0]
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        val1 = np.dot(A, p1[0]) + np.dot(B, p1[1]) + C
        val2 = np.dot(A, p2[0]) + np.dot(B, p2[1]) + C
        if val1 * val2 < 0:
            t = -val1 / (val2 - val1)
            intersection = p1 + t * (p2 - p1)
            polygon_vertices.append(intersection)
    center = np.mean(polygon_vertices, axis=0)
    polygon_vertices.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return [tuple(p) for p in polygon_vertices]


# ==============================================================================
# --- 核心生成逻辑 (完全按照你的思路重写) ---
# ==============================================================================
def create_single_sample(config):
    size = config.IMG_SIZE

    # --- 步骤 1: 生成分割直线，确定空气和介质区域 ---
    # 为了保证边界线不会太靠边，我们让它穿过中心的一个小区域
    center_point = np.array([size / 2, size / 2]) + np.random.uniform(-size / 4, size / 4, 2)
    angle = np.random.uniform(0, 2 * np.pi)
    normal_vec = np.array([np.cos(angle), np.sin(angle)])
    # 介质区域定义为 normal_vec 指向的一侧

    # --- 步骤 2: 在分割线上挑一个点，在空气区域在图像的边缘随机选一个点，连成入射线 ---
    # 在分割线上挑一个点 (point_of_incidence)
    # 我们直接使用之前定义的 center_point 作为入射点，这保证了它在图像中心附近
    point_of_incidence = center_point

    # 在空气区域的图像边缘随机选一个点 (incident_start)
    air_side_normal = -normal_vec  # 空气在法线的反方向
    A, B = air_side_normal
    C = -np.dot(air_side_normal, point_of_incidence)  # Ax+By+C > 0 是空气侧

    valid_edges = []
    corners = [np.array([0, 0]), np.array([size, 0]), np.array([size, size]), np.array([0, size])]
    edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for i in range(4):
        p1, p2 = corners[edge_indices[i][0]], corners[edge_indices[i][1]]
        val1 = np.dot(A, p1[0]) + np.dot(B, p1[1]) + C
        val2 = np.dot(A, p2[0]) + np.dot(B, p2[1]) + C
        if val1 >= 0 or val2 >= 0:  # 只要有一个端点在空气侧，这条边就有效
            valid_edges.append((p1, p2))

    # 从有效边中随机选一条，并在这条边上随机取一个点
    edge_p1, edge_p2 = random.choice(valid_edges)
    incident_start = edge_p1 + random.uniform(0, 1) * (edge_p2 - edge_p1)

    # --- 步骤 3: 计算折射 ---
    incident_vec = point_of_incidence - incident_start
    if np.linalg.norm(incident_vec) < 1: return None  # 避免距离太近
    incident_vec /= np.linalg.norm(incident_vec)

    cos_theta1 = -np.dot(incident_vec, normal_vec)
    if cos_theta1 < 1e-6: return None  # 避免光线几乎平行于边界

    sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)
    sin_theta2 = (config.N_AIR / config.N_MEDIUM) * sin_theta1

    if abs(sin_theta2) > 1: return None  # 全反射，理论上不会发生

    theta2 = np.arcsin(sin_theta2)
    cos_theta2 = np.cos(theta2)
    ratio = config.N_AIR / config.N_MEDIUM
    refracted_vec = ratio * incident_vec + (ratio * cos_theta1 - cos_theta2) * normal_vec
    refracted_vec /= np.linalg.norm(refracted_vec)

    refracted_end = get_intersection_with_box(point_of_incidence, point_of_incidence + refracted_vec, size)

    # --- 步骤 4: 绘图 ---
    polygon_to_fill = get_boundary_polygon(normal_vec, point_of_incidence, size)

    input_img = Image.new('RGB', (size, size), config.AIR_COLOR)
    draw_input = ImageDraw.Draw(input_img)
    draw_input.polygon(polygon_to_fill, fill=config.MEDIUM_COLOR)
    draw_input.line([tuple(incident_start), tuple(point_of_incidence)], fill=config.INCIDENT_RAY_COLOR,
        width=config.LINE_WIDTH)

    output_img = input_img.copy()
    draw_output = ImageDraw.Draw(output_img)
    draw_output.line([tuple(point_of_incidence), tuple(refracted_end)], fill=config.REFRACTED_RAY_COLOR,
        width=config.LINE_WIDTH)

    return input_img, output_img


def generate_dataset(num_samples, output_dir, name, config):
    print(f"\n正在生成 {name} 数据集 ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    pbar = tqdm(total=num_samples, desc=f"生成 {name}", unit="个世界")
    while count < num_samples:
        sample = create_single_sample(config)
        if sample:
            input_img, output_img = sample
            input_img.save(Path(output_dir) / f"{count}_input.png")
            output_img.save(Path(output_dir) / f"{count}_output.png")
            count += 1
            pbar.update(1)
    pbar.close()


# ==============================================================================
# --- 程序入口 ---
# ==============================================================================
if __name__=="__main__":
    cfg = Config()
    if os.path.exists(cfg.DATA_DIR):
        print(f"删除旧的数据集: {cfg.DATA_DIR}")
        shutil.rmtree(cfg.DATA_DIR)

    print("=" * 60)
    print("开始生成物理推理数据集：光的折射 v1.3 (用户逻辑优化版)")
    print("方法: 完全遵循用户设计的构造法，清晰、健壮、优雅。")
    print("=" * 60)

    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
    # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)

    print(f"\n🎉🎉🎉 数据集 '{cfg.DATA_DIR}' 生成完毕！ 🎉🎉🎉")

