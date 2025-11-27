import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random
from PIL import Image, ImageDraw
from operator import itemgetter

# --- 1. 配置参数 ---
NUM_SAMPLES = 1000
NUM_PROCESSES = cpu_count()
IMAGE_SIZE_PX = 256
OUTPUT_DIR = "cube_dataset_final_highlight"

# --- 2. 立方体和绘图常量定义 ---
VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
])

FACE_INDICES = [
    (0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
    (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)
]

FACE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]
EDGE_COLOR = (0, 0, 0)
MID_LINE_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (255, 255, 255)

LINE_WIDTH_UNIFIED = 4

# 【新增要求】: 特殊顶点高亮的参数
# 我们固定选择顶点索引为6（即原始坐标(1,1,1)的那个角）作为特殊顶点
SPECIAL_VERTEX_INDEX = 6
SPECIAL_VERTEX_COLOR = (255, 128, 0)  # 醒目的橙色
SPECIAL_VERTEX_RADIUS = 12  # 圆的半径


# --- 3. 辅助函数 --- (无变动)
def angles_to_binary_label(q_roll, q_pitch, q_yaw):
    return f"{q_roll:08b}{q_pitch:08b}{q_yaw:08b}"


def generate_unique_angle_triplets(num_samples):
    print(f"正在生成 {num_samples} 组不重复的角度...")
    angle_triplets = set()
    max_val = 2 ** 8
    if num_samples > max_val ** 3:
        raise ValueError(f"请求的样本数 {num_samples} 超过了可能组合的总数 {max_val ** 3}")
    while len(angle_triplets) < num_samples:
        q_roll, q_pitch, q_yaw = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        angle_triplets.add((q_roll, q_pitch, q_yaw))
    print("角度生成完毕。")
    return list(angle_triplets)


# --- 4. 核心图像生成函数 ---
def generate_image_task(args):
    idx, (q_roll, q_pitch, q_yaw), output_dir_path = args

    angle_scale = 360.0 / 256.0
    roll_deg, pitch_deg, yaw_deg = q_roll * angle_scale, q_pitch * angle_scale, q_yaw * angle_scale
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    rotated_vertices = VERTICES @ R.T

    scale = IMAGE_SIZE_PX * 0.28
    projected_vertices = []
    for v in rotated_vertices:
        x = v[0] * scale + IMAGE_SIZE_PX / 2
        y = -v[1] * scale + IMAGE_SIZE_PX / 2
        projected_vertices.append((x, y))

    faces_to_render = []
    for i, face_indices in enumerate(FACE_INDICES):
        face_3d_verts = rotated_vertices[list(face_indices)]
        face_2d_verts = [projected_vertices[j] for j in face_indices]
        v0, v1, v2 = face_2d_verts[0], face_2d_verts[1], face_2d_verts[2]
        cull_check = (v1[0] - v0[0]) * (v2[1] - v1[1]) - (v1[1] - v0[1]) * (v2[0] - v1[0])
        if cull_check > 0:
            avg_depth = np.mean(face_3d_verts[:, 2])
            faces_to_render.append({
                'depth': avg_depth, 'vertices': face_2d_verts, 'color': FACE_COLORS[i]
            })

    faces_to_render.sort(key=itemgetter('depth'))

    img = Image.new('RGB', (IMAGE_SIZE_PX, IMAGE_SIZE_PX), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Pass 1: 绘制所有带颜色的填充面
    for face in faces_to_render:
        draw.polygon(face['vertices'], fill=face['color'])

    # Pass 2: 绘制所有白色中线
    for face in faces_to_render:
        verts = face['vertices']
        mid01 = ((verts[0][0] + verts[1][0]) / 2, (verts[0][1] + verts[1][1]) / 2)
        mid23 = ((verts[2][0] + verts[3][0]) / 2, (verts[2][1] + verts[3][1]) / 2)
        mid12 = ((verts[1][0] + verts[2][0]) / 2, (verts[1][1] + verts[2][1]) / 2)
        mid30 = ((verts[3][0] + verts[0][0]) / 2, (verts[3][1] + verts[0][1]) / 2)
        draw.line([mid01, mid23], fill=MID_LINE_COLOR, width=LINE_WIDTH_UNIFIED)
        draw.line([mid12, mid30], fill=MID_LINE_COLOR, width=LINE_WIDTH_UNIFIED)

    # Pass 3: 绘制所有黑色边框
    for face in faces_to_render:
        verts = face['vertices']
        draw.line(verts + [verts[0]], fill=EDGE_COLOR, width=LINE_WIDTH_UNIFIED, joint="curve")

    # 【新增要求】 Pass 4: 最后绘制特殊的顶点，确保它在最顶层
    special_vertex_2d = projected_vertices[SPECIAL_VERTEX_INDEX]
    x, y = special_vertex_2d
    radius = SPECIAL_VERTEX_RADIUS
    # draw.ellipse需要一个边界框 (x0, y0, x1, y1)
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=SPECIAL_VERTEX_COLOR)

    filename = f"{idx:06d}.png"
    filepath = os.path.join(output_dir_path, "images", filename)
    img.save(filepath)

    return filename, angles_to_binary_label(q_roll, q_pitch, q_yaw)


# --- 5. 主执行逻辑 --- (无变动)
if __name__=='__main__':
    print("--- 开始使用Pillow生成立方体旋转数据集 ---")

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

    print("-" * 30);
    print("🎉 数据集生成成功！ 🎉");
    print(f"总计: {len(df)} 张图像");
    print(f"图像目录: {images_dir}");
    print(f"元数据文件: {csv_path}");
    print("-" * 30)