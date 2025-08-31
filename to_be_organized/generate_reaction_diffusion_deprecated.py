import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from scipy.signal import convolve2d

# --- 全局配置 ---
IMG_SIZE = 128  # 考虑到计算量，使用128x128，可以按需调整
NUM_SAMPLES = 500
OUTPUT_DIR = "snowflake_dataset_combined_128"
NUM_STEPS = 100  # 物理过程演化的总步数

# --- 物理参数 ---
# 这个扩散核是关键，它定义了养分如何扩散
# 中心值为负，代表该点养分的流失；周围为正，代表从邻居获得
DIFFUSION_KERNEL = np.array([[0.05, 0.2, 0.05],
                             [0.2, -1.0, 0.2],
                             [0.05, 0.2, 0.05]])
FEED_RATE = 0.055  # 每一步，向系统中添加的基础养分率 (模拟开放系统)
KILL_RATE = 0.062  # 养分消耗率的参数
SOLIDIFICATION_THRESHOLD = 0.1  # 养分浓度达到多少时，会发生凝固

# --- 颜色定义 ---
COLOR_LIQUID = (10, 10, 40)  # 深蓝色背景 (液态)
COLOR_SOLID = (255, 255, 255)  # 白色 (固态晶体)


# --- 核心物理引擎 ---

def run_simulation(initial_matter, num_steps):
    """
    运行确定性的反应-扩散模拟。
    """
    # 初始化物质场和养分场
    matter = initial_matter.copy()
    nutrient = np.ones_like(matter, dtype=np.float32)  # 养分场初始为1

    for _ in range(num_steps):
        # 1. 扩散 (Diffusion)
        # 使用卷积模拟养分的扩散，'same'模式保持尺寸不变，'wrap'处理边界
        laplacian = convolve2d(nutrient, DIFFUSION_KERNEL, mode='same', boundary='wrap')

        # 2. 反应 (Reaction)
        # 根据Gray-Scott模型方程更新养分场
        # dN/dt = D * laplacian(N) - N * M^2 + F * (1-N)
        #   D: 扩散系数 (已在卷积核中体现)
        #   N * M^2: 养分消耗项 (nutrient * matter^2)
        #   F * (1-N): 养分补给项
        nutrient_change = laplacian - (nutrient * matter ** 2) + FEED_RATE * (1 - nutrient)
        nutrient += nutrient_change

        # 3. 凝固 (Solidification)
        # 找到所有与固体相邻的液体点 (候选点)
        # 通过卷积找到固体的邻居
        solid_neighbors = convolve2d(matter, np.ones((3, 3)), mode='same', boundary='wrap') > 0
        candidates = solid_neighbors & (matter==0)  # 是邻居，但本身不是固体

        # 检查候选点的养分浓度
        solidifying_points = candidates & (nutrient > SOLIDIFICATION_THRESHOLD)

        # 更新物质场
        matter[solidifying_points] = 1

        # 消耗凝固点的养分
        nutrient[solidifying_points] = 0

    return matter


def to_image(matter_grid):
    """将物质状态矩阵转换为PIL图像"""
    h, w = matter_grid.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)

    # 根据物质状态填充颜色
    img_array[matter_grid==0] = COLOR_LIQUID
    img_array[matter_grid==1] = COLOR_SOLID

    return Image.fromarray(img_array)


# --- 数据集生成主逻辑 ---

def generate_dataset():
    """生成整个数据集"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"开始生成 {NUM_SAMPLES} 个雪花样本到 '{OUTPUT_DIR}'...")

    for i in tqdm(range(NUM_SAMPLES)):
        # 1. 创建初始条件 (输入)
        # 在一个空白的物质场中，放置一个或多个晶核
        initial_matter = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # 在中心放置一个晶核
        center = IMG_SIZE // 2
        initial_matter[center, center] = 1

        # (可选) 增加一些随机性，比如放置几个随机的初始晶核
        num_extra_seeds = random.randint(0, 2)
        for _ in range(num_extra_seeds):
            x, y = random.randint(center - 5, center + 5), random.randint(center - 5, center + 5)
            initial_matter[x, y] = 1

        # 2. 运行模拟得到最终状态 (输出)
        final_matter = run_simulation(initial_matter, NUM_STEPS)

        # 3. 将输入和输出转换为图像
        input_image = to_image(initial_matter)
        output_image = to_image(final_matter)

        # 4. 合并并保存
        combined_image = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))
        combined_image.paste(input_image, (0, 0))
        combined_image.paste(output_image, (IMG_SIZE, 0))

        filename = f"snowflake_{i:05d}.png"
        combined_image.save(os.path.join(OUTPUT_DIR, filename))

    print("雪花数据集生成完毕！")
    print(f"一张预览图已保存为: {os.path.join(OUTPUT_DIR, 'snowflake_00000.png')}")


if __name__=="__main__":
    generate_dataset()
