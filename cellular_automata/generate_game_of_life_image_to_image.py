import os
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from tqdm import tqdm

# --- 配置参数 ---
# 逻辑网格的大小
GRID_SIZE = 8
# 最终输出图像的尺寸 (像素)
IMAGE_SIZE = 224
# 要生成的样本数量
NUM_SAMPLES = 10000
# 数据集输出的根目录
OUTPUT_DIR = "life_game_dataset"


# --- 核心逻辑 ---

def calculate_next_state(grid: np.ndarray) -> np.ndarray:
    """
    使用标准生命游戏规则 (B3/S23) 计算下一个状态。
    这是一个高效的实现，使用2D卷积来计算邻居数量。

    Args:
        grid (np.ndarray): 当前状态的二维Numpy数组 (值为0或1)。

    Returns:
        np.ndarray: 下一个状态的二维Numpy数组。
    """
    # B3/S23 规则:
    # 1. 对于活细胞 (1): 如果邻居少于2个或多于3个，则死亡 (变为0)。(Underpopulation/Overpopulation)
    # 2. 对于活细胞 (1): 如果邻居有2个或3个，则存活 (保持为1)。(Survival)
    # 3. 对于死细胞 (0): 如果邻居恰好有3个，则诞生 (变为1)。(Birth)

    # 创建一个3x3的卷积核来计算8个邻居的和
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # 使用卷积计算每个细胞的邻居数量。'wrap'模式实现了环形边界条件。
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')

    # 应用规则
    # 条件1: 活细胞 -> 存活 (邻居为2或3)
    survivors = (grid==1) & ((neighbor_count==2) | (neighbor_count==3))

    # 条件2: 死细胞 -> 诞生 (邻居为3)
    newborns = (grid==0) & (neighbor_count==3)

    # 新的网格是存活者和诞生者的并集
    new_grid = (survivors | newborns).astype(np.uint8)

    return new_grid


def grid_to_image(grid: np.ndarray, image_size: int) -> Image.Image:
    """
    将逻辑网格转换为指定尺寸的黑白图像。
    使用Numpy的kron函数进行高效的图像放大。

    Args:
        grid (np.ndarray): 二维Numpy数组 (值为0或1)。
        image_size (int): 输出图像的边长 (像素)。

    Returns:
        Image.Image: PIL图像对象。
    """
    grid_h, grid_w = grid.shape
    block_size_h = image_size // grid_h
    block_size_w = image_size // grid_w

    if image_size % grid_h!=0 or image_size % grid_w!=0:
        raise ValueError("Image size must be a multiple of grid size.")

    # 将逻辑值 (0, 1) 映射到颜色值 (白色, 黑色)
    # 0 (死) -> 255 (白)
    # 1 (活) -> 0 (黑)
    color_grid = 255 - grid * 255

    # 使用Kronecker product高效地将每个单元格放大成一个块
    image_array = np.kron(color_grid, np.ones((block_size_h, block_size_w))).astype(np.uint8)

    return Image.fromarray(image_array, 'L')


def generate_dataset():
    """
    主函数，生成并保存整个数据集。
    """
    print("--- 开始生成生命游戏数据集 ---")
    print(f"逻辑网格: {GRID_SIZE}x{GRID_SIZE}")
    print(f"输出图像: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"样本数量: {NUM_SAMPLES}")

    # 创建输出目录
    input_path = os.path.join(OUTPUT_DIR, "input")
    output_path = os.path.join(OUTPUT_DIR, "output")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"数据将保存到: {os.path.abspath(OUTPUT_DIR)}")

    for i in tqdm(range(NUM_SAMPLES), desc="正在生成图像对"):
        # 1. 生成一个随机的初始状态
        # 使用0.2的概率初始化，避免棋盘过满或过空，产生更有趣的模式
        initial_density = 0.2
        state_t = (np.random.rand(GRID_SIZE, GRID_SIZE) < initial_density).astype(np.uint8)

        # 2. 计算下一个状态
        state_t_plus_1 = calculate_next_state(state_t)

        # 3. 将两个状态都转换成图像
        image_t = grid_to_image(state_t, IMAGE_SIZE)
        image_t_plus_1 = grid_to_image(state_t_plus_1, IMAGE_SIZE)

        # 4. 保存图像对
        # 使用6位数字格式化文件名，方便排序 (e.g., 000001.png)
        filename = f"{i + 1:06d}.png"
        image_t.save(os.path.join(input_path, filename))
        image_t_plus_1.save(os.path.join(output_path, filename))

    print("\n--- 数据集生成完毕！---")


if __name__=="__main__":
    generate_dataset()