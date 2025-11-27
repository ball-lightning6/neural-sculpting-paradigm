import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp
from collections import deque

### 1. 配置参数 ###
DATASET_DIR = "arc_ultimate_final_v4_dataset"
GRID_DIM = 28
CELL_SIZE = 8
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_BLACK = (0, 0, 0)
COLOR_BG_BLUE = (135, 206, 250)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 0, 128),
    (255, 165, 0), (255, 192, 203), (0, 255, 255), (165, 42, 42)
]
NUM_SAMPLES = 1000

C_BLACK = 0
C_BG_BLUE = 1


### 2. 辅助函数 ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def create_path_block(m, n, color_code):
    block = np.full((m, m), C_BG_BLUE)
    offset = (m - n) // 2
    block[offset:offset + n, offset:offset + n] = color_code
    return block


def find_path_on_black(grid, start_node, end_node):
    dim = grid.shape[0]
    q = deque([(start_node, [start_node])])
    visited = {start_node}

    while q:
        (r, c), path = q.popleft()
        if (r, c)==end_node:
            return path

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < dim and 0 <= nc < dim and grid[nr, nc]==C_BLACK and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                q.append(((nr, nc), new_path))
    return None


def sample_with_distance(p, i, j):
    """
    从 1 到 p 的范围内（包括1和p），抽取 i 个正整数，
    保证任意两个相邻的正整数 a, b (a < b)，都有 b - a >= j。

    Args:
        p (int): 范围的上限 (e.g., 28)。
        i (int): 需要抽取的整数个数 (e.g., 4)。
        j (int): 相邻整数间的最小距离 (e.g., 5)。

    Returns:
        list: 一个排序好的、满足条件的整数列表。
              如果无法满足条件，则返回一个空列表。
    """
    # --- 1. 可行性检查 ---
    # i个数字本身需要i个位置
    # i-1个间隔，每个间隔至少占用j-1个位置
    min_required_space = i + (i - 1) * (j - 1)

    if i==0:
        return []
    if i==1:
        return [random.randint(1, p)]

    if p < min_required_space:
        # print(f"错误：无法在范围 [1, {p}] 内找到 {i} 个距离至少为 {j} 的数。")
        # print(f"至少需要 {min_required_space} 的空间。")
        return []

    # --- 2. 核心算法：映射到压缩空间 ---

    # 压缩后的上限。我们从这个更小的范围里，无约束地抽i个数。
    # 想象一下，我们先把所有强制的间隔(j-1)都抽掉
    compressed_p = p - (i - 1) * (j - 1)

    # 从[1, compressed_p]这个压缩空间中，随机抽取i个不重复的数
    # range的第二个参数是 exclusive，所以要 +1
    compressed_samples = random.sample(range(1, compressed_p + 1), i)

    # --- 3. 映射回原始空间 ---

    # 先对抽取的样本排序
    compressed_samples.sort()

    # 将每个样本“拉伸”回原始空间
    # 第k个样本，它的前面有k-1个间隔，所以要加上 (k-1)*(j-1) 的偏移量
    result = [sample + k * (j - 1) for k, sample in enumerate(compressed_samples)]

    return result

### 3. 核心生成逻辑 ###
def generate_logic_grids():
    for _ in range(50):  # 主循环，尝试50次生成一个合法样本
        dim = GRID_DIM
        block_configs = [(4, 2), (5, 3), (6,4), (6,2)]
        m, n = random.choice(block_configs)
        path_len = random.choice([6, 7, 8])

        # Step 1: 定义宏观坐标系
        margin = m + 1
        max_slots_per_dim = (dim - 2) // margin

        if max_slots_per_dim < 2: continue

        num_rows = random.randint(2, max_slots_per_dim)
        num_cols = random.randint(2, max_slots_per_dim)

        row_pixels = sample_with_distance(25-m,num_rows,m+1)
        col_pixels = sample_with_distance(27-m,num_cols,m+1)

        if num_rows * num_cols < path_len: continue

        row_indices = sorted(random.sample(range(max_slots_per_dim), num_rows))
        col_indices = sorted(random.sample(range(max_slots_per_dim), num_cols))

        # Step 2: 生成路径骨架
        slots_grid = np.full((num_rows, num_cols), -1, dtype=int)

        # 随机游走生成一个保证连通的路径
        path_in_slots = []
        # r, c = random.randint(0, num_rows - 1), random.randint(0, num_cols - 1)
        r, c = random.randint(0, 0), random.randint(0, 0)
        path_in_slots.append((r, c))

        for _ in range(path_len - 1):
            neighbors = []
            if r > 0 and (r - 1, c) not in path_in_slots: neighbors.append((r - 1, c))
            if r < num_rows - 1 and (r + 1, c) not in path_in_slots: neighbors.append((r + 1, c))
            if c > 0 and (r, c - 1) not in path_in_slots: neighbors.append((r, c - 1))
            if c < num_cols - 1 and (r, c + 1) not in path_in_slots: neighbors.append((r, c + 1))

            if not neighbors: break
            r, c = random.choice(neighbors)
            path_in_slots.append((r, c))

        if len(path_in_slots)!=path_len: continue

        # Step 3: 实例化路径块
        path_colors = random.sample(range(len(COLOR_PALETTE)), path_len)
        blocks = []
        for i in range(path_len):
            r_idx, c_idx = path_in_slots[i]
            # r_pixel = 1 + row_indices[r_idx] * margin
            # c_pixel = 1 + col_indices[c_idx] * margin
            r_pixel = row_pixels[r_idx]
            c_pixel = col_pixels[c_idx]

            # 【新】严格的边界检查
            if not (1 <= r_pixel and r_pixel + m < dim - 3 and 1 <= c_pixel and c_pixel + m < dim - 1):
                continue  # 如果此槽位导致越界，则此次生成失败

            blocks.append({'r': r_pixel, 'c': c_pixel, 'm': m, 'n': n, 'color_code': path_colors[i] + 2})

        if len(blocks)!=path_len: continue

        # Step 4: 绘制输入和输出
        input_grid = np.zeros((dim, dim), dtype=int)
        instruction_colors = [b['color_code'] for b in blocks]

        for i, color_code in enumerate(instruction_colors):
            input_grid[dim - 2, 2 + i * 2] = color_code

        for block in blocks:
            block_pattern = create_path_block(m, n, block['color_code'])
            input_grid[block['r']:block['r'] + m, block['c']:block['c'] + m] = block_pattern

        output_grid = input_grid.copy()

        # Step 5:【新】执行非破坏性连线
        for i in range(path_len - 1):
            current_block = blocks[i]
            next_block = blocks[i + 1]  # 路径是按顺序生成的

            line_color = current_block['color_code']
            line_width = current_block['n']
            offset = m // 2

            start_node = (current_block['r'] + offset, current_block['c'] + offset)
            end_node = (next_block['r'] + offset, next_block['c'] + offset)

            # 【新】在一个纯黑色的背景上寻找路径
            path_finding_grid = np.zeros((dim, dim), dtype=int)
            path = find_path_on_black(path_finding_grid, start_node, end_node)

            if path:
                for r_p, c_p in path:
                    # 绘制线宽
                    start_r_fill = r_p - line_width // 2
                    end_r_fill = start_r_fill + line_width
                    start_c_fill = c_p - line_width // 2
                    end_c_fill = start_c_fill + line_width

                    if line_width > 1:  # 如果线宽大于1，就画一个方块
                        output_grid[start_r_fill:end_r_fill, start_c_fill:end_c_fill][
                            output_grid[start_r_fill:end_r_fill, start_c_fill:end_c_fill]==C_BLACK] = line_color
                    else:  # 如果线宽为1，只画一个点
                        if output_grid[r_p, c_p]==C_BLACK:
                            output_grid[r_p, c_p] = line_color

        return input_grid, output_grid  # 成功，返回结果

    return None  # 尝试多次后仍然失败


def grid_to_image(grid, cell_size):
    # ... (此函数无需修改)
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {C_BLACK: COLOR_BLACK, C_BG_BLUE: COLOR_BG_BLUE}
    for i, color in enumerate(COLOR_PALETTE):
        color_map[i + 2] = color

    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], C_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (最终勘误大结局版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终版图像对已保存到 '{DATASET_DIR}' 目录。")