import json
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

# --- 配置 ---
CONFIG = {
    "input_file": "trapping_rain_water_decoupled_n10_b3_train.jsonl",  # <--- 请将这里替换为你的jsonl文件名
    "output_dir": "rainwater_image_dataset",
    "image_size": (224, 224),
    "num_pillars": 10,
    "bits_per_number": 3,
    "colors": {
        "background": "white",
        "pillar": "black",
        "water": "blue"
    }
}


# --- 核心功能函数 ---

def parse_line(line_str, num_pillars, bits_per_number):
    """解析单行jsonl数据，将其从符号格式转换为整数数组。"""
    data = json.loads(line_str)
    input_str = data["input"]
    output_bits = data["output"]

    # 1. 解析高度
    heights = []
    for i in range(num_pillars):
        start_idx = i * bits_per_number
        end_idx = start_idx + bits_per_number
        binary_str = input_str[start_idx:end_idx]
        heights.append(int(binary_str, 2))

    # 2. 解析接水量
    water_amounts = []
    for i in range(num_pillars):
        start_idx = i * bits_per_number
        end_idx = start_idx + bits_per_number
        # 将bit列表 [1, 1, 0] 转换为字符串 "110"
        binary_str = "".join(map(str, output_bits[start_idx:end_idx]))
        water_amounts.append(int(binary_str, 2))

    return heights, water_amounts


def render_image(heights, water_amounts=None, config=CONFIG):
    """
    将高度数组和可选的接水量数组渲染成一张图片。
    """
    img_width, img_height = config["image_size"]
    colors = config["colors"]

    # 创建一个白色背景的画布
    image = Image.new('RGB', (img_width, img_height), colors["background"])
    draw = ImageDraw.Draw(image)

    num_pillars = len(heights)
    if num_pillars==0:
        return image

    # 使用固定的最大高度来确保所有图片的垂直比例尺一致
    # 最大可能高度是 2^bits - 1, e.g., 2^3 - 1 = 7
    max_possible_height = (2 ** config["bits_per_number"]-1)# * 2  # 乘以2确保水有足够的空间

    # 计算每个柱子的宽度和高度缩放比例
    bar_width = img_width / num_pillars
    scale_h = (img_height * 1) / max_possible_height  # 留出10%的上边距

    for i in range(num_pillars):
        pillar_h = heights[i]

        # 计算柱子的坐标
        x0 = i * bar_width
        y1 = img_height  # 底部
        y0_pillar = img_height - (pillar_h * scale_h)  # 柱子顶部
        x1 = (i + 1) * bar_width

        # 1. 画柱子
        draw.rectangle([x0, y0_pillar, x1, y1], fill=colors["pillar"])

        # 2. 如果提供了接水量，则画水
        if water_amounts is not None:
            water_h = water_amounts[i]
            if water_h > 0:
                # 水位是柱子高度 + 接水量
                water_level = pillar_h + water_h
                y0_water = img_height - (water_level * scale_h)  # 水面顶部
                # 水的底部就是柱子的顶部
                y1_water = y0_pillar

                draw.rectangle([x0, y0_water, x1, y1_water], fill=colors["water"])

    return image


# --- 主执行逻辑 ---

def main():
    """主函数，执行整个转换过程。"""
    print("开始将符号数据集转换为图像数据集...")

    # 创建输出目录结构
    input_img_dir = os.path.join(CONFIG["output_dir"], "train", "input")
    output_img_dir = os.path.join(CONFIG["output_dir"], "train", "output")
    os.makedirs(input_img_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    print(f"图像将保存在: {CONFIG['output_dir']}")

    try:
        with open(CONFIG["input_file"], 'r') as f:
            lines = f.readlines()

            for idx, line in enumerate(tqdm(lines, desc="正在生成图像对")):
                line = line.strip()
                if not line:
                    continue

                # 1. 解析数据
                heights, water_amounts = parse_line(
                    line, CONFIG["num_pillars"], CONFIG["bits_per_number"]
                )

                # 2. 渲染输入图像 (只有柱子)
                input_image = render_image(heights)

                # 3. 渲染输出图像 (柱子 + 水)
                output_image = render_image(heights, water_amounts=water_amounts)

                # 4. 保存图像
                filename = f"{idx:06d}.png"  # e.g., 000001.png
                input_image.save(os.path.join(input_img_dir, filename))
                output_image.save(os.path.join(output_img_dir, filename))

    except FileNotFoundError:
        print(f"错误: 输入文件 '{CONFIG['input_file']}' 未找到。")
        print("请确保脚本和jsonl文件在同一目录下，或者修改脚本中的'input_file'路径。")
        return

    print(f"\n转换完成！成功生成 {len(lines)} 对训练图像。")


if __name__=="__main__":
    main()
