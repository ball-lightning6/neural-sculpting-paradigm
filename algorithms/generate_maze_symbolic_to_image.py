import os
import json
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================

# --- 输入文件 ---
INPUT_JSONL_FILE = "maze_optimized_dataset.jsonl"

# --- 输出设置 ---
OUTPUT_IMAGE_DIR = "maze_images"  # 存放所有生成图像的文件夹
OUTPUT_LABELS_FILE = "labels.csv"  # 存放图像路径和对应标签的CSV文件

# --- 图像参数 ---
MAZE_DIM = 13  # 我们的迷宫内部尺寸是 13x13
CELL_SIZE = 17  # 每个格子的像素尺寸
IMAGE_SIZE = MAZE_DIM * CELL_SIZE  # 最终图像尺寸: 13 * 17 = 221

# --- 颜色映射 (你可以自定义这些颜色) ---
# 使用醒目的颜色以帮助模型区分
COLOR_MAP = {
    '0': (255, 255, 255),  # 0: 通路 (白色)
    '1': (0, 0, 0),  # 1: 墙壁 (黑色)
    's': (0, 255, 0),  # s: 玩家 (亮绿色)
    't': (255, 0, 0)  # t: 目标 (亮红色)
}

# --- JPG 图像质量 ---
JPG_QUALITY = 95  # 1-100, 95是高质量和高压缩率的良好平衡


# ==============================================================================
# --- 核心转换代码 ---
# ==============================================================================

def convert_jsonl_to_images():
    """
    读取JSONL文件，并将其转换为图像数据集和标签文件。
    """
    print("=" * 60)
    print("开始将 JSONL 数据集转换为图像格式...")
    print(f"输入文件: {INPUT_JSONL_FILE}")
    print(f"输出图像文件夹: {OUTPUT_IMAGE_DIR}")
    print(f"输出标签文件: {OUTPUT_LABELS_FILE}")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    print("=" * 60)

    # 1. 确保输出文件夹存在
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # 2. 打开文件准备读写
    try:
        with open(INPUT_JSONL_FILE, 'r') as f_in, \
                open(OUTPUT_LABELS_FILE, 'w', newline='') as f_out:

            # 创建CSV写入器并写入表头
            csv_writer = csv.writer(f_out)
            csv_writer.writerow(['image_path', 'label'])

            # 使用tqdm来显示进度条
            # 先计算总行数以正确显示进度
            total_lines = sum(1 for line in open(INPUT_JSONL_FILE, 'r'))
            f_in.seek(0)  # 重置文件指针

            for line_num, line in tqdm(enumerate(f_in), total=total_lines, desc="正在转换"):
                data = json.loads(line)
                input_str = data['input']
                label = data['output']
                
                # 验证输入长度
                expected_length = MAZE_DIM * MAZE_DIM
                if len(input_str) != expected_length:
                    print(f"警告: 第{line_num}行输入长度{len(input_str)}不符合预期{expected_length}")
                    continue

                # 3. 创建空白的图像数组
                # 使用uint8类型，这是图像的标准格式 (0-255)
                image_array = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

                # 4. 填充图像颜色块
                for r in range(MAZE_DIM):
                    for c in range(MAZE_DIM):
                        char_index = r * MAZE_DIM + c
                        char = input_str[char_index]
                        color = COLOR_MAP.get(char, (128, 128, 128))  # 如果有未知字符，用灰色表示

                        # 计算该色块在图像中的像素坐标范围
                        y_start, x_start = r * CELL_SIZE, c * CELL_SIZE
                        y_end, x_end = y_start + CELL_SIZE, x_start + CELL_SIZE

                        # 使用NumPy的切片功能高效填充颜色
                        image_array[y_start:y_end, x_start:x_end] = color

                # 5. 从NumPy数组创建Pillow图像对象并保存
                image = Image.fromarray(image_array, 'RGB')
                image_filename = f"{line_num}.jpg"
                image_path = os.path.join(OUTPUT_IMAGE_DIR, image_filename)

                # 保存为JPG格式
                image.save(image_path, 'jpeg', quality=JPG_QUALITY)

                # 6. 将图像路径和标签写入CSV文件
                csv_writer.writerow([image_path, label])

        print("\n🎉🎉🎉 转换成功！🎉🎉🎉")
        print(f"总共生成了 {total_lines} 张图像。")
        print(f"图像保存在: '{OUTPUT_IMAGE_DIR}' 文件夹中。")
        print(f"标签保存在: '{OUTPUT_LABELS_FILE}' 文件中。")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_JSONL_FILE}' 未找到。请确保文件名正确且文件存在。")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")


if __name__=='__main__':
    convert_jsonl_to_images()
