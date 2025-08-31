import json
import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# --- 配置参数 ---

# 输入的JSONL文件路径 (由上一个脚本生成)
INPUT_JSONL_PATH = 'sokoban_optimized_dataset.jsonl'

# 输出的图像数据集根目录
OUTPUT_DIR = 'sokoban_image_dataset'

# --- 图像规格 ---
GRID_SIZE = 10  # 我们的网格是 10x10
CELL_PIXELS = 22  # 每个单元格的像素大小
PADDING = 2  # 上下左右的内边距
IMAGE_SIZE = GRID_SIZE * CELL_PIXELS + PADDING * 2  # 10*22 + 2*2 = 224

# --- 颜色映射 (R, G, B) ---
# 精心挑选的高对比度颜色
COLOR_MAP = {
    '#': (50, 50, 50),  # 墙壁: 深灰色
    '.': (200, 200, 200),  # 地板: 浅灰色
    'S': (0, 255, 0),  # 玩家: 亮绿色
    'B': (255, 165, 0),  # 箱子: 亮橙色
    'T': (255, 0, 0),  # 目标: 亮红色
}
PADDING_COLOR = (0, 0, 0)  # 内边距: 黑色


def string_to_image(grid_string: str) -> Image.Image:
    """
    将1D的网格字符串转换为一个Pillow图像对象。

    Args:
        grid_string: 包含所有网格信息的1D字符串。

    Returns:
        一个 224x224 的Pillow图像对象。
    """
    # 1. 创建一个黑色的背景画布 (用于padding)
    image_array = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), PADDING_COLOR, dtype=np.uint8)

    # 2. 遍历字符串，绘制每一个单元格
    for i, char in enumerate(grid_string):
        # 将1D索引转换为2D坐标
        r = i // GRID_SIZE
        c = i % GRID_SIZE

        # 获取该单元格的颜色
        color = COLOR_MAP.get(char, (255, 255, 255))  # 如果有未知字符，用白色

        # 计算该单元格在图像上的像素起止位置
        y_start = r * CELL_PIXELS + PADDING
        y_end = y_start + CELL_PIXELS
        x_start = c * CELL_PIXELS + PADDING
        x_end = x_start + CELL_PIXELS

        # 使用Numpy切片高效地填充颜色矩形
        image_array[y_start:y_end, x_start:x_end] = color

    # 3. 将Numpy数组转换为Pillow图像
    return Image.fromarray(image_array)


def create_image_dataset():
    """
    主函数，读取JSONL文件并生成图像数据集。
    """
    print("--- 开始将符号数据集转换为图像数据集 ---")

    # 1. 检查输入文件是否存在
    if not os.path.exists(INPUT_JSONL_PATH):
        print(f"错误: 输入文件 '{INPUT_JSONL_PATH}' 未找到。")
        print("请先运行上一个脚本生成该文件。")
        return

    # 2. 创建输出目录结构
    images_dir = os.path.join(OUTPUT_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)
    print(f"输出目录 '{OUTPUT_DIR}' 已准备就绪。")

    # 3. 准备写入labels.csv文件
    labels_csv_path = os.path.join(OUTPUT_DIR, 'labels.csv')

    with open(labels_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        csv_writer.writerow(['image_filename', 'label'])
        print(f"标签文件 '{labels_csv_path}' 已创建。")

        # 4. 遍历JSONL文件，生成每个样本
        print("开始生成图像和标签...")
        with open(INPUT_JSONL_PATH, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="生成图像")):
                data = json.loads(line)
                grid_string = data['input']
                label = data['output']

                # 将字符串转换为图像
                image = string_to_image(grid_string)

                # 生成文件名并保存图像
                filename = f"sample_{i + 1:06d}.png"  # e.g., sample_000001.png
                image_path = os.path.join(images_dir, filename)
                image.save(image_path)

                # 将文件名和标签写入CSV
                csv_writer.writerow([filename, label])

    print("\n--- 任务完成！---")
    print(f"图像数据集已成功生成在 '{OUTPUT_DIR}' 目录下。")
    print(f"总计生成了 {len(lines)} 张图像。")

    # 随机展示一张生成的图片以供验证
    print("\n显示一张随机生成的样本以供验证...")
    try:
        random_image_file = random.choice(os.listdir(images_dir))
        img_to_show = Image.open(os.path.join(images_dir, random_image_file))
        img_to_show.show()
    except Exception as e:
        print(f"无法显示图片 (可能在无GUI的环境中): {e}")


if __name__=='__main__':
    create_image_dataset()
