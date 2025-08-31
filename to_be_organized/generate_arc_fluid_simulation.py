import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp
from collections import deque

# --- 1. 配置参数 ---
DATASET_DIR = "arc_fluid_final_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_ORANGE_BG = (255, 165, 0)
COLOR_RED_BARRIER = (255, 0, 0)
COLOR_PURPLE = (128, 0, 128)
NUM_SAMPLES = 150000

C_ORANGE = 1
C_RED = 2
C_PURPLE = 3


### 2. 辅助函数 ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def check_barrier_overlap(new_r, new_c, new_len, existing_barriers):
    """【新】检查挡板是否与现有挡板（包括边和角）粘连"""
    for r, c, length in existing_barriers:
        # 检查新挡板的每个点
        for i in range(new_len):
            nc = new_c + i
            # 检查周围8个邻居+自己
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # 如果邻居点是现有挡板的一部分
                    if r <= new_r + dr < r + 1 and c <= nc + dc < c + length:
                        return True
    return False


### 3. 核心生成逻辑 ###
def generate_logic_grids():
    """【最终救赎版】核心算法"""
    dim = GRID_DIM

    # 1. 创建橙色背景
    input_grid = np.full((dim, dim), C_ORANGE, dtype=int)

    # 2. 【修正】生成互不接触的红色挡板
    num_barriers = random.randint(4, 8)
    existing_barriers = []
    for _ in range(num_barriers):
        for attempt in range(20):
            length = random.randint(1, dim // 2)
            r = random.randint(3, dim - 2)  # 不在最顶和最底
            c_start = random.randint(0, dim - length)

            if not check_barrier_overlap(r, c_start, length, existing_barriers):
                existing_barriers.append((r, c_start, length))
                input_grid[r, c_start:c_start + length] = C_RED
                break

    # 3. 随机生成1或2个水龙头
    num_faucets = random.randint(1, 2)
    faucet_cols = sorted(random.sample(range(dim), num_faucets))
    if num_faucets==2 and faucet_cols[1] <= faucet_cols[0] + 1:
        return None  # 生成失败，重试

    faucet_starts = []
    for c in faucet_cols:
        # 确保水龙头不生成在挡板上
        if input_grid[0, c]==C_RED or input_grid[1, c]==C_RED:
            return None
        input_grid[0:2, c] = C_PURPLE
        faucet_starts.append((1, c))

    if len(faucet_starts)!=num_faucets: return None

    output_grid = input_grid.copy()

    # 4. 【核心修正】模拟水流
    q = deque()
    for r_start, c_start in faucet_starts:
        q.append((r_start, c_start, 1, 0))

    visited_states = set()

    while q:
        r, c, dr, dc = q.popleft()

        current_state_key = (r, c, dr, dc)
        if current_state_key in visited_states or len(visited_states) > dim * dim * 4:  # 防止无限循环
            continue
        visited_states.add(current_state_key)

        # 渲染当前位置
        if output_grid[r, c]==C_ORANGE:
            output_grid[r, c] = C_PURPLE

        # --- 【核心修正】所有碰撞检测，都在原始的、不变的 input_grid 上进行 ---
        # 状态转移逻辑
        if dr==1:  # 向下流
            next_r = r + 1
            if next_r < dim and input_grid[next_r, c]!=C_RED:
                # 下方不是挡板，继续向下
                q.append((next_r, c, 1, 0))
            elif next_r < dim and input_grid[next_r, c]==C_RED:
                # 下方是挡板，分叉
                q.append((r, c, 0, -1))  # 从当前点开始向左
                q.append((r, c, 0, 1))  # 从当前点开始向右

        elif dc!=0:  # 水平流
            next_c = c + dc
            next_r_down = r + 1
            if 0 <= next_c < dim and input_grid[r, next_c]!=C_RED:
                # 前方不是挡板，继续水平，并检查下方
                q.append((r, next_c, 0, dc))


                if next_r_down < dim and input_grid[next_r_down, c]!=C_RED:
                    # 下方无支撑，产生新的向下分支
                    #q.append((r, c, 1, 0))
                    q[-1] = (r, c, 1, 0)
            #else: 撞墙或撞挡板侧面，该分支停止
            elif next_r_down < dim and input_grid[next_r_down, c]!=C_RED:
                q.append((r, c, 1, 0))

    return input_grid, output_grid


def grid_to_image(grid, cell_size):
    """将逻辑网格转换为PIL图像"""
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {
        C_ORANGE: COLOR_ORANGE_BG,
        C_RED: COLOR_RED_BARRIER,
        C_PURPLE: COLOR_PURPLE,
    }
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_ORANGE_BG)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    try:
        result = None
        while result is None:
            # 循环直到成功生成一个有效的样本
            result = generate_logic_grids()
        input_grid, output_grid = result

        input_image = grid_to_image(input_grid, CELL_SIZE)
        output_image = grid_to_image(output_grid, CELL_SIZE)

        filename = f"{sample_index:06d}.png"
        input_path = os.path.join(DATASET_DIR, "input", filename)
        output_path = os.path.join(DATASET_DIR, "output", filename)
        input_image.save(input_path)
        output_image.save(output_path)
        return True
    except Exception as e:
        print(f"在生成样本 {sample_index} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__=="__main__":
    create_directories()

    num_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (最终救赎版)...")

    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))

    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终救赎版图像对已保存到 '{DATASET_DIR}' 目录。")