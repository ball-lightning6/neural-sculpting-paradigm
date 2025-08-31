import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import math
import random

# --- 全局配置 ---
IMG_SIZE = 224
NUM_SAMPLES = 150000
OUTPUT_DIR = "orbital_dataset_256_separated_v3"
INPUT_SUBDIR = os.path.join(OUTPUT_DIR, "input")
OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "output")
os.makedirs(INPUT_SUBDIR, exist_ok=True)
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

# --- 物理和视觉参数 ---
G = 5000
MIN_AREA_RATIO = 0.15  # [FIX 2] 轨道最小面积占总面积的比例
COLOR_BACKGROUND = (0, 0, 0)
COLOR_STAR = (255, 220, 0)
COLOR_PLANET_START = (0, 150, 255)
COLOR_SLOW = np.array([0, 150, 255])  # [FIX 1] 低速颜色 (蓝色)
COLOR_FAST = np.array([255, 50, 50])  # [FIX 1] 高速颜色 (红色)
COLOR_ORBIT_PATH = (255, 255, 255)


def interpolate_color(value, color1, color2):
    """在线性颜色空间中插值"""
    return tuple((color1 * (1 - value) + color2 * value).astype(int))


def generate_orbital_sample(sample_id):
    # 1. 生成随机椭圆参数，并施加更严格的约束
    margin = 30
    center_x = random.uniform(margin, IMG_SIZE - margin)
    center_y = random.uniform(margin, IMG_SIZE - margin)
    max_a = min(center_x, IMG_SIZE - center_x, center_y, IMG_SIZE - center_y) - 5
    if max_a <= 40: return None  # 避免过小的椭圆

    # [FIX 2] 提高相对尺寸下限
    a = random.uniform(max_a * 0.6, max_a)
    b = random.uniform(a * 0.4, a * 0.95)  # 同样可以略微提高偏心率下限

    # [FIX 2] 增加绝对面积约束
    if (np.pi * a * b) < (IMG_SIZE * IMG_SIZE * MIN_AREA_RATIO):
        return None

    angle_rad = random.uniform(0, 2 * np.pi)

    # 2. 定位恒星
    eccentricity = np.sqrt(1 - (b ** 2 / a ** 2))
    focal_distance = a * eccentricity
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    star_pos_local = random.choice([np.array([-focal_distance, 0]), np.array([focal_distance, 0])])
    star_pos = np.dot(rotation_matrix, star_pos_local) + np.array([center_x, center_y])
    star_mass = a * 1.5
    star_radius = np.clip(a / 15, 4, 12)

    # 3. 放置行星并计算方向
    theta = random.uniform(0, 2 * np.pi)
    planet_pos_local = np.array([a * np.cos(theta), b * np.sin(theta)])
    planet_pos = np.dot(rotation_matrix, planet_pos_local) + np.array([center_x, center_y])
    tangent_local = np.array([-a * np.sin(theta), b * np.cos(theta)])
    tangent_direction = np.dot(rotation_matrix, tangent_local)
    velocity_direction = tangent_direction / np.linalg.norm(tangent_direction)

    # 4. 计算速度大小和颜色
    r = np.linalg.norm(planet_pos - star_pos)
    v_squared = G * star_mass * (2 / r - 1 / a)
    if v_squared < 0: return None
    v_magnitude = np.sqrt(v_squared)

    # [FIX 1] 计算速度范围并进行颜色插值
    r_aphelion = a * (1 + eccentricity)
    r_perihelion = a * (1 - eccentricity)
    v_aphelion_sq = G * star_mass * (2 / r_aphelion - 1 / a)
    v_perihelion_sq = G * star_mass * (2 / r_perihelion - 1 / a)
    if v_aphelion_sq < 0 or v_perihelion_sq < 0: return None
    v_min = np.sqrt(v_aphelion_sq)
    v_max = np.sqrt(v_perihelion_sq)

    # 归一化速度
    norm_v = (v_magnitude - v_min) / (v_max - v_min + 1e-6)  # 加一个极小量避免除零
    velocity_color = interpolate_color(np.clip(norm_v, 0, 1), COLOR_SLOW, COLOR_FAST)

    # --- 5. 生成图像 ---
    input_image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), COLOR_BACKGROUND)
    draw_input = ImageDraw.Draw(input_image)
    draw_input.ellipse([(star_pos[0] - star_radius, star_pos[1] - star_radius),
                        (star_pos[0] + star_radius, star_pos[1] + star_radius)], fill=COLOR_STAR)
    planet_radius = 10
    draw_input.ellipse([(planet_pos[0] - planet_radius, planet_pos[1] - planet_radius),
                        (planet_pos[0] + planet_radius, planet_pos[1] + planet_radius)], fill=COLOR_PLANET_START)
    vel_indicator_length = 40
    vel_indicator_end = planet_pos + velocity_direction * vel_indicator_length
    draw_input.line([(planet_pos[0], planet_pos[1]), (vel_indicator_end[0], vel_indicator_end[1])],
        fill=velocity_color, width=10)

    output_image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), COLOR_BACKGROUND)#input_image.copy()
    draw_output = ImageDraw.Draw(output_image)
    path_points = []
    num_path_points = 200
    for t in np.linspace(0, 2 * np.pi, num_path_points):
        point_local = np.array([a * np.cos(t), b * np.sin(t)])
        point_world = np.dot(rotation_matrix, point_local) + np.array([center_x, center_y])
        path_points.append(tuple(point_world))
    draw_output.line(path_points, fill=COLOR_ORBIT_PATH, width=10, joint='curve')

    draw_output.ellipse([(star_pos[0] - star_radius, star_pos[1] - star_radius),
                        (star_pos[0] + star_radius, star_pos[1] + star_radius)], fill=COLOR_STAR)
    planet_radius = 10
    draw_output.ellipse([(planet_pos[0] - planet_radius, planet_pos[1] - planet_radius),
                        (planet_pos[0] + planet_radius, planet_pos[1] + planet_radius)], fill=COLOR_PLANET_START)
    vel_indicator_length = 40
    vel_indicator_end = planet_pos + velocity_direction * vel_indicator_length
    draw_output.line([(planet_pos[0], planet_pos[1]), (vel_indicator_end[0], vel_indicator_end[1])],
        fill=velocity_color, width=10)

    # 6. 分别保存图像
    filename = f"orbit_{sample_id:05d}.png"
    input_image.save(os.path.join(INPUT_SUBDIR, filename))
    output_image.save(os.path.join(OUTPUT_SUBDIR, filename))
    return True


# --- 主循环 ---
if __name__=="__main__":
    print(f"开始生成 {NUM_SAMPLES} 个轨道样本到 '{OUTPUT_DIR}'...")
    success_count = 0
    pbar = tqdm(total=NUM_SAMPLES)
    while success_count < NUM_SAMPLES:
        if generate_orbital_sample(success_count):
            success_count += 1
            pbar.update(1)
    pbar.close()
    print("轨道数据集生成完毕！")
