import os
import random
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 1. Configuration ---
# 您可以在这里轻松调整所有参数
# ----------------------------------------------------
NUM_SAMPLES = 100      # 要生成的训练/验证样本总数
OUTPUT_DIR = "checkerboard_dataset"  # 数据集根目录
IMAGE_SUBDIR = "images"  # 存放图像的子目录
METADATA_FILE = "metadata.csv"  # 记录图像文件名和对应标签的CSV文件

IMAGE_SIZE = 224         # 输出图像的尺寸 (像素)
GRID_DIM = 8             # 网格的维度 (8x8)

# 定义 '0' 和 '1' 对应的颜色 (0=黑色, 255=白色)
# 我们使用灰度图，因为颜色本身不重要，结构才是关键
COLOR_MAP = {
    '0': 255,
    '1': 0
}
# ----------------------------------------------------


def generate_dataset():
    """
    主函数，用于生成整个数据集。
    """
    # --- 2. Setup ---
    # 创建输出目录
    images_path = os.path.join(OUTPUT_DIR, IMAGE_SUBDIR)
    os.makedirs(images_path, exist_ok=True)
    print(f"数据集将被保存在: '{os.path.abspath(OUTPUT_DIR)}'")

    # 计算每个单元格的像素尺寸
    assert IMAGE_SIZE % GRID_DIM == 0, "图像尺寸必须能被网格维度整除"
    cell_size = IMAGE_SIZE // GRID_DIM
    grid_total_cells = GRID_DIM * GRID_DIM

    metadata = []

    # --- 3. Generation Loop ---
    # 使用 tqdm 创建一个漂亮的进度条
    for i in tqdm(range(NUM_SAMPLES), desc="正在生成样本"):
        # a. 生成一个随机的64位二进制字符串作为标签
        binary_string_label = ''.join(random.choice('01') for _ in range(grid_total_cells))

        # b. 创建一个黑色的画布 (NumPy 数组)
        image_array = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        # c. 根据二进制字符串填充图像
        for idx, char in enumerate(binary_string_label):
            # 计算当前字符在8x8网格中的行和列
            row = idx // GRID_DIM
            col = idx % GRID_DIM

            # 计算该单元格在图像中的像素起止坐标
            start_y = row * cell_size
            end_y = start_y + cell_size
            start_x = col * cell_size
            end_x = start_x + cell_size

            # 获取对应的颜色并填充该区域
            color = COLOR_MAP[char]
            image_array[start_y:end_y, start_x:end_x] = color

        # d. 将 NumPy 数组转换为 PIL 图像对象
        img = Image.fromarray(image_array, 'L')  # 'L' 模式代表8位灰度图

        # e. 保存图像文件
        # 使用补零的方式命名，方便文件排序 (e.g., 00001, 00002, ...)
        filename = f"checkerboard_{i:05d}.png"
        filepath = os.path.join(images_path, filename)
        img.save(filepath)

        # f. 记录元数据
        relative_filepath = os.path.join(IMAGE_SUBDIR, filename)
        metadata.append({'filename': relative_filepath, 'label': binary_string_label})

    # --- 4. Save Metadata ---
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
        writer.writeheader()
        writer.writerows(metadata)

    print("\n数据集生成完毕！")
    print(f"总共生成了 {len(metadata)} 个样本。")
    print(f"元数据文件保存在: '{metadata_path}'")


if __name__ == "__main__":
    generate_dataset()

