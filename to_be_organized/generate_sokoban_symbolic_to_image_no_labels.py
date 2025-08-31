import os
import json
from PIL import Image, ImageDraw

# ==============================================================================
# --- 配置区域: 你只需要修改这里 ---
# ==============================================================================

# 1. 输入/输出文件设置
INPUT_JSONL_FILE = "sokoban_dataset_shuffled_8.jsonl"  # 你的训练集文件名
OUTPUT_IMAGE_DIR = "sokoban_images_224"  # 图片保存目录

# 2. 棋盘尺寸 (m行, n列)
GRID_ROWS = 8  # 你的棋盘的行数 (m)
GRID_COLS = 8  # 你的棋盘的列数 (n)

# 3. 数据格式
#    你的jsonl文件中，代表棋盘状态字符串的那个键名
JSON_STATE_KEY = "input"

# 4. 视觉风格
#    最终输出的目标图片尺寸 (ViT-Large-224 需要 224)
TARGET_IMAGE_SIZE = 224

#    定义你的符号和对应的颜色
COLOR_MAP = {
    '#': (40, 40, 40),  # 墙壁 (Wall)
    's': (0, 102, 204),  # 玩家 (Player)
    'b': (153, 102, 51),  # 箱子 (Box)
    '*': (128, 255, 128),  # 目标点 (Goal)
    '.': (211, 211, 211),  # 地板/空白区域 (Floor)
}
# 背景色，用于填充多余的区域
PADDING_COLOR = (0, 0, 0)  # 黑色


# ==============================================================================
# --- 核心代码: 通常不需要修改以下部分 ---
# ==============================================================================

def main():
    """主执行函数"""

    # --- 步骤1: 自动计算最佳的色块大小 (TILE_SIZE) ---
    # 为了让 m x n 的棋盘能放进 TARGET_IMAGE_SIZE 的正方形里，
    # 我们需要找到最大的整数TILE_SIZE
    tile_size_w = TARGET_IMAGE_SIZE // GRID_COLS
    tile_size_h = TARGET_IMAGE_SIZE // GRID_ROWS
    TILE_SIZE = min(tile_size_w, tile_size_h)

    if TILE_SIZE==0:
        print("错误: 棋盘尺寸相对于目标图片尺寸过大，无法计算有效的色块大小。")
        return

    print(f"棋盘尺寸: {GRID_ROWS}x{GRID_COLS}")
    print(f"目标图片尺寸: {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}")
    print(f"自动计算出的最佳色块大小: {TILE_SIZE}x{TILE_SIZE} 像素")

    # --- 步骤2: 创建输出目录 ---
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    print(f"\n图片将保存在: {OUTPUT_IMAGE_DIR}")

    # --- 步骤3: 读取并处理jsonl文件 ---
    try:
        with open(INPUT_JSONL_FILE, 'r') as f:
            lines = f.readlines()

        total_lines = len(lines)
        print(f"找到 {total_lines} 个样本。开始转换...")

        # --- 步骤4: 循环生成图片 ---
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                flat_state = data[JSON_STATE_KEY]

                # 完整性检查
                if len(flat_state)!=GRID_ROWS * GRID_COLS:
                    print(f"  [警告] 第 {i} 行的状态字符串长度 ({len(flat_state)}) 与配置的棋盘尺寸 ({GRID_ROWS * GRID_COLS}) 不匹配。已跳过。")
                    continue

                # 构建输出文件名 (行号.png)
                output_filename = f"{i}.png"
                output_path = os.path.join(OUTPUT_IMAGE_DIR, output_filename)

                # 调用渲染函数
                render_and_save_image(flat_state, TILE_SIZE, output_path)

                # 打印进度
                if (i + 1) % 500==0 or (i + 1)==total_lines:
                    print(f"  已处理: {i + 1} / {total_lines}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [警告] 处理第 {i} 行时发生错误 ({e})。已跳过。")

        print("\n转换完成！")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {INPUT_JSONL_FILE}")


def render_and_save_image(flat_state, tile_size, output_path):
    """根据扁平字符串，渲染并保存一张居中、填充的图片"""

    # 渲染棋盘本身（可能不是正方形）
    board_width = GRID_COLS * tile_size
    board_height = GRID_ROWS * tile_size
    board_image = Image.new('RGB', (board_width, board_height))
    draw = ImageDraw.Draw(board_image)

    for i, char in enumerate(flat_state):
        if char in COLOR_MAP:
            row = i // GRID_COLS
            col = i % GRID_COLS
            color = COLOR_MAP[char]

            top_left_x = col * tile_size
            top_left_y = row * tile_size

            draw.rectangle(
                [top_left_x, top_left_y, top_left_x + tile_size, top_left_y + tile_size],
                fill=color
            )

    # 创建一个最终的、标准尺寸的画布
    final_image = Image.new('RGB', (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), PADDING_COLOR)

    # 计算粘贴位置，使其居中
    paste_x = (TARGET_IMAGE_SIZE - board_width) // 2
    paste_y = (TARGET_IMAGE_SIZE - board_height) // 2

    # 将渲染好的棋盘粘贴到最终画布的中央
    final_image.paste(board_image, (paste_x, paste_y))

    # 保存最终的图片
    final_image.save(output_path)


if __name__=="__main__":
    main()
