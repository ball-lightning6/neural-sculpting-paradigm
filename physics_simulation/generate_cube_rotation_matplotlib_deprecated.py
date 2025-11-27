import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random

# --- 1. 配置参数 ---
NUM_SAMPLES = 100
NUM_PROCESSES = cpu_count()
IMAGE_SIZE_PX = 256
OUTPUT_DIR = "cube_dataset"

# --- 2. 立方体和绘图常量定义 ---
VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
]) * 0.5

FACE_INDICES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]
])

FACE_COLORS = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
EDGE_COLOR = 'black'
LINE_WIDTH_EDGE = 3
LINE_WIDTH_MID = 2.5


# --- 3. 辅助函数 ---
def angles_to_binary_label(q_roll, q_pitch, q_yaw):
    return f"{q_roll:08b}{q_pitch:08b}{q_yaw:08b}"


def generate_unique_angle_triplets(num_samples):
    print(f"正在生成 {num_samples} 组不重复的角度...")
    angle_triplets = set()
    max_val = 2 ** 8
    if num_samples > max_val ** 3:
        raise ValueError(f"请求的样本数 {num_samples} 超过了可能组合的总数 {max_val ** 3}")
    while len(angle_triplets) < num_samples:
        q_roll = random.randint(0, max_val - 1)
        q_pitch = random.randint(0, max_val - 1)
        q_yaw = random.randint(0, max_val - 1)
        angle_triplets.add((q_roll, q_pitch, q_yaw))
    print("角度生成完毕。")
    return list(angle_triplets)


# --- 4. 核心图像生成函数（为多进程设计） ---
def generate_image_task(args):
    idx, (q_roll, q_pitch, q_yaw), output_dir_path = args

    angle_scale = 360.0 / 256.0
    roll_deg, pitch_deg, yaw_deg = q_roll * angle_scale, q_pitch * angle_scale, q_yaw * angle_scale
    roll_rad, pitch_rad, yaw_rad = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([[1, 0, 0], [0, np.cos(roll_rad), -np.sin(roll_rad)], [0, np.sin(roll_rad), np.cos(roll_rad)]])
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)], [0, 1, 0], [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    rotated_vertices = VERTICES @ R.T

    fig = plt.figure(figsize=(IMAGE_SIZE_PX / 100, IMAGE_SIZE_PX / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    rotated_faces = rotated_vertices[FACE_INDICES]

    # 添加面集合
    ax.add_collection3d(Poly3DCollection(
        rotated_faces, facecolors=FACE_COLORS, linewidths=LINE_WIDTH_EDGE, edgecolors=EDGE_COLOR
    ))

    # 绘制每个面上的白色中线
    for face_verts in rotated_faces:
        v0, v1, v2, v3 = face_verts
        mid01, mid23 = (v0 + v1) / 2, (v2 + v3) / 2
        mid12, mid30 = (v1 + v2) / 2, (v3 + v0) / 2

        # 【关键修正】: 添加 zorder 参数，强制白线在顶层渲染
        ax.plot([mid01[0], mid23[0]], [mid01[1], mid23[1]], [mid01[2], mid23[2]],
            color='white', linewidth=LINE_WIDTH_MID, zorder=5)
        ax.plot([mid12[0], mid30[0]], [mid12[1], mid30[1]], [mid12[2], mid30[2]],
            color='white', linewidth=LINE_WIDTH_MID, zorder=5)

    ax.set_xlim([-0.7, 0.7]);
    ax.set_ylim([-0.7, 0.7]);
    ax.set_zlim([-0.7, 0.7])
    ax.view_init(elev=20, azim=30)
    ax.axis('off')

    filename = f"{idx:06d}.png"
    filepath = os.path.join(output_dir_path, "images", filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return filename, angles_to_binary_label(q_roll, q_pitch, q_yaw)


# --- 5. 主执行逻辑 ---
if __name__=='__main__':
    print("--- 开始生成立方体旋转数据集 ---")

    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"数据集将保存在: '{OUTPUT_DIR}' 目录")

    unique_angles = generate_unique_angle_triplets(NUM_SAMPLES)

    tasks = [(i, angle_tuple, OUTPUT_DIR) for i, angle_tuple in enumerate(unique_angles)]

    print(f"使用 {NUM_PROCESSES} 个CPU核心开始生成 {NUM_SAMPLES} 张图像...")
    metadata = []

    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(generate_image_task, tasks), total=NUM_SAMPLES):
            metadata.append(result)

    print("所有图像生成完毕！")

    print("正在创建 metadata.csv...")
    df = pd.DataFrame(metadata, columns=['filename', 'label'])
    df = df.sample(frac=1).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    df.to_csv(csv_path, index=False)

    print("-" * 30)
    print("🎉 数据集生成成功！ 🎉")
    print(f"总计: {len(df)} 张图像")
    print(f"图像目录: {images_dir}")
    print(f"元数据文件: {csv_path}")
    print("-" * 30)