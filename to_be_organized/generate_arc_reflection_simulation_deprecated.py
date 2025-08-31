import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp
from collections import deque

### 1. 配置参数 ###
DATASET_DIR = "arc_ray_optics_dataset"
GRID_DIM = 24  # 24x24 for more space
CELL_SIZE = 10  # 240x240 image
IMG_SIZE = GRID_DIM * CELL_SIZE

# 颜色定义
COLOR_BLACK = (0, 0, 0)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (128, 0, 128), (255, 165, 0), (0, 255, 255), (192, 192, 192)
]
NUM_SAMPLES = 1000

# 颜色编码
C_BLACK = 0


### 2. 辅助函数 ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def get_emitter_shape(dr, dc):
    """根据发射方向获取2x2的L形发射器"""
    shape = np.ones((2, 2), dtype=bool)
    if dr==1 and dc==1:
        shape[0, 0] = False  # 右下
    elif dr==1 and dc==-1:
        shape[0, 1] = False  # 左下
    elif dr==-1 and dc==1:
        shape[1, 0] = False  # 右上
    elif dr==-1 and dc==-1:
        shape[1, 1] = False  # 左上
    return shape


### 3. 核心生成逻辑 ###
def generate_logic_grids():
    for _ in range(100):  # 尝试100次生成一个合法样本
        dim = GRID_DIM
        occupied = np.zeros((dim, dim), dtype=bool)

        input_grid = np.zeros((dim, dim), dtype=int)
        output_grid = np.zeros((dim, dim), dtype=int)

        num_rays = random.randint(2, 4)
        colors_in_use = random.sample(range(len(COLOR_PALETTE)), num_rays)

        all_paths_ok = True
        for i in range(num_rays):
            emitter_color_code = colors_in_use[i] + 1

            # --- 逆向工程：先生成轨迹 ---
            path = []
            reflect_points = []

            # 随机起点
            r, c = random.randint(2, dim - 3), random.randint(2, dim - 3)
            # 随机初始方向
            dr, dc = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])

            current_color_code = emitter_color_code

            path_len = 0
            max_path_len = dim * 3
            num_reflections = random.randint(0, 2)

            temp_path = [(r, c)]

            while path_len < max_path_len:
                r, c = r + dr, c + dc

                if not (0 <= r < dim and 0 <= c < dim):  # 撞墙
                    break

                temp_path.append((r, c))
                path_len += 1

                if num_reflections > 0 and random.random() < 0.2 and len(temp_path) > 3:  # 随机反射
                    num_reflections -= 1

                    # 定义反射
                    # 从(dr, dc) 反射到 (new_dr, new_dc)
                    # 挡板位置在 (r,c) 的某个邻居
                    # 挡板颜色是反射后的新颜色

                    # 确定新方向
                    if dr==1 and dc==1:  # 右下 ->
                        new_dr, new_dc = random.choice([(-1, 1), (1, -1)])  # 右上或左下
                    elif dr==1 and dc==-1:  # 左下 ->
                        new_dr, new_dc = random.choice([(-1, -1), (1, 1)])  # 左上或右下
                    elif dr==-1 and dc==1:  # 右上 ->
                        new_dr, new_dc = random.choice([(1, 1), (-1, -1)])  # 右下或左上
                    else:  # 左上 ->
                        new_dr, new_dc = random.choice([(1, -1), (-1, 1)])  # 左下或右上

                    # 确定挡板位置
                    # e.g. 右下->右上: 挡板在下方 (r+1,c)
                    # e.g. 右下->左下: 挡板在右方 (r,c+1)
                    # 这是一个复杂的几何关系，我们简化一下
                    # 我们只关心反射点和新方向

                    new_color_code = random.choice(range(len(COLOR_PALETTE))) + 1
                    reflect_points.append({
                        'pos': (r, c),
                        'old_dir': (dr, dc),
                        'new_dir': (new_dr, new_dc),
                        'new_color': new_color_code
                    })
                    dr, dc = new_dr, new_dc
                    # 这部分逻辑太复杂，简化版无法保证无歧义，我们采用更直接的生成方式

            # 由于逆向工程的逻辑极其复杂，我们再次简化，采用“正向生成”并检查有效性

        # --- 正向生成，但保证逻辑清晰 ---

        objects = []  # 存储所有发射器和挡板 {'type', 'pos', 'shape/dir', 'color'}

        # 1. 放置发射器
        for i in range(num_rays):
            color_code = colors_in_use[i] + 1
            for _ in range(20):
                r = random.randint(1, dim - 3)
                c = random.randint(1, dim - 3)
                dr, dc = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                emitter_shape = get_emitter_shape(dr, dc)

                #
                coords = [(r + ro, c + co) for ro in range(2) for co in range(2) if emitter_shape[ro, co]]
                if not any(occupied[r_p, c_p] for r_p, c_p in coords):
                    objects.append({'type': 'emitter', 'pos': (r, c), 'dir': (dr, dc), 'color': color_code})
                    for r_p, c_p in coords: occupied[r_p, c_p] = True
                    break

        # 2. 放置挡板
        num_barriers = random.randint(num_rays, num_rays * 2)
        for _ in range(num_barriers):
            color_code = random.choice(range(len(COLOR_PALETTE))) + 1
            for _ in range(20):
                r, c = random.randint(1, dim - 2)
                if not occupied[r, c]:
                    objects.append({'type': 'barrier', 'pos': (r, c), 'color': color_code})
                    occupied[r, c] = True
                    break

        # 3. 绘制输入
        for obj in objects:
            if obj['type']=='emitter':
                r, c = obj['pos']
                shape = get_emitter_shape(*obj['dir'])
                input_grid[r:r + 2, c:c + 2][shape] = obj['color']
            else:
                r, c = obj['pos']
                input_grid[r, c] = obj['color']

        output_grid = input_grid.copy()

        # 4. 计算射线路径
        for obj in objects:
            if obj['type']=='emitter':
                r_start, c_start = obj['pos']
                dr_start, dc_start = obj['dir']

                # 定位发射点
                # e.g., 向右下(1,1)发射, 发射器缺(0,0), 发射点是(0,0)的对角(1,1)
                r, c = r_start + (1 - dr_start) // -2, c_start + (1 - dc_start) // -2

                dr, dc = obj['dir']
                current_color = obj['color']

                for _ in range(dim * 2):  # 限制最长路径
                    r, c = r + dr, c + dc
                    if not (0 <= r < dim and 0 <= c < dim): break  # 撞墙

                    if output_grid[r, c]==C_BLACK:
                        output_grid[r, c] = current_color
                    else:  # 撞到东西了
                        # 检查是不是挡板
                        hit_obj = None
                        for barrier in objects:
                            if barrier['type']=='barrier' and barrier['pos']==(r, c):
                                hit_obj = barrier
                                break

                        if hit_obj:  # 是挡板，发生反射
                            # 简化反射逻辑：90度偏转
                            if dr!=0 and dc!=0:  # 对角线运动
                                new_dr, new_dc = random.choice([(dr, -dc), (-dr, dc)])
                                dr, dc = new_dr, new_dc
                                current_color = hit_obj['color']
                            # 其他复杂情况忽略
                        else:  # 撞到其他射线或发射器，停止
                            break

        # 检查是否成功生成了有意义的图像，如果太简单就重试
        if np.sum(output_grid > 1) > np.sum(input_grid > 0) + 5:  # 至少画了5个点
            return input_grid, output_grid

    return None  # 最终失败


def grid_to_image(grid, cell_size):
    # ...
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {C_BLACK: COLOR_BLACK}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 1] = color
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], C_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    # ...
    try:
        result = None
        while result is None:
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (终极BOSS战)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个光学模拟任务图像对已保存到 '{DATASET_DIR}' 目录。")