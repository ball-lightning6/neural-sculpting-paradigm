import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

# --- 1. 配置参数 ---
DATASET_DIR = "arc_ultimate_plus_dataset"
GRID_DIM = 16
CELL_SIZE = 14
IMG_SIZE = GRID_DIM * CELL_SIZE  # 16 * 14 = 224

COLOR_BLACK = (0, 0, 0)
COLOR_GREY = (128, 128, 128)
COLOR_RED = (255, 0, 0)
NUM_SAMPLES = 1000

C_BLACK = 0
C_GREY = 1
C_RED = 2

# --- 新增任务参数 ---
HOLE_AREA_MIN_RATIO = 0.15  # 空洞面积至少占内部空间的15%


def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def check_overlap(r1, c1, h1, w1, rects):
    for r2, c2, h2, w2 in rects:
        if not (r1 >= r2 + h2 + 1 or r1 + h1 + 1 <= r2 or \
                c1 >= c2 + w2 + 1 or c1 + w1 + 1 <= c2):
            return True
    return False


def generate_hole(inner_h, inner_w, is_square):
    """【最终修正版】生成正方形，或按比例生成长方形/不规则形状的空洞"""
    hole_mask = np.zeros((inner_h, inner_w), dtype=int)
    inner_area = inner_h * inner_w
    min_hole_area = int(inner_area * HOLE_AREA_MIN_RATIO)

    if is_square:
        max_size = min(inner_h, inner_w)
        # 确保正方形面积也满足阈值
        min_size = int(np.sqrt(max(1, min_hole_area)))
        if min_size > max_size: min_size = max_size  # 防止无法生成

        size = random.randint(min_size, max_size)
        h, w = size, size
        r_start = random.randint(0, inner_h - h)
        c_start = random.randint(0, inner_w - w)
        hole_mask[r_start:r_start + h, c_start:c_start + w] = 1
    else:  # 生成非正方形负例
        # 按比例决定是生成长方形，还是不规则形状
        if random.random() < 0.7:  # 70%的概率生成长方形
            # 确保面积够大，且长宽不等
            for _ in range(20):  # 尝试20次
                h = random.randint(1, inner_h)
                w = random.randint(1, inner_w)
                if h!=w and h * w >= min_hole_area:
                    r_start = random.randint(0, inner_h - h)
                    c_start = random.randint(0, inner_w - w)
                    hole_mask[r_start:r_start + h, c_start:c_start + w] = 1
                    return hole_mask  # 成功生成，直接返回

        # 如果上面没成功，或30%的概率，生成不规则形状
        for _ in range(20):  # 尝试20次
            num_cells = random.randint(max(2, min_hole_area), inner_area - 1)
            r, c = random.randint(0, inner_h - 1), random.randint(0, inner_w - 1)
            points = {(r, c)}
            for _ in range(num_cells - 1):
                frontier = []
                for pr, pc in points:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in points:
                            frontier.append((nr, nc))
                if not frontier: break
                new_point = random.choice(frontier)
                points.add(new_point)

            if len(points) >= min_hole_area:
                rows, cols = zip(*points)
                h_bbox = max(rows) - min(rows) + 1
                w_bbox = max(cols) - min(cols) + 1
                if h_bbox!=w_bbox:  # 确保包围盒不是正方形，增加难度
                    for r_p, c_p in points:
                        hole_mask[r_p, c_p] = 1
                    return hole_mask  # 成功生成，直接返回

    return hole_mask  # 如果都没成功，返回一个可能为空的掩码


# --- 后续函数与主程序入口与上一版完全相同，这里省略以保持简洁 ---
# 您只需将上面的 generate_hole 函数，替换掉上一版脚本中的同名函数即可。
# 为了完整性，我还是将完整脚本放在下面。

def generate_logic_grids():
    dim = GRID_DIM
    grid = np.zeros((dim, dim), dtype=int)
    num_rects = random.randint(2, 6)
    existing_rects = []

    for _ in range(num_rects):
        for attempt in range(50):
            w = random.randint(5, dim // 2)
            h = random.randint(5, dim // 2)
            r = random.randint(0, dim - h)
            c = random.randint(0, dim - w)
            if not check_overlap(r, c, h, w, existing_rects):
                existing_rects.append((r, c, h, w))

                grid[r, c:c + w] = C_GREY
                grid[r + h - 1, c:c + w] = C_GREY
                grid[r:r + h, c] = C_GREY
                grid[r:r + h, c + w - 1] = C_GREY

                inner_r, inner_c = r + 1, c + 1
                inner_h, inner_w = h - 2, w - 2
                grid[inner_r:inner_r + inner_h, inner_c:inner_c + inner_w] = C_GREY

                is_square_hole = random.choice([True, False])
                hole_mask = generate_hole(inner_h, inner_w, is_square_hole)

                grid[inner_r:inner_r + inner_h, inner_c:inner_c + inner_w][hole_mask==1] = C_BLACK
                break
    return grid


def apply_rule_and_get_output(input_grid):
    from scipy.ndimage import label, binary_fill_holes
    output_grid = input_grid.copy()
    grey_mask = (input_grid==C_GREY)
    all_holes_mask = binary_fill_holes(grey_mask) & ~grey_mask
    labeled_holes, num_holes = label(all_holes_mask)
    for i in range(1, num_holes + 1):
        hole_mask = (labeled_holes==i)
        hole_coords = np.argwhere(hole_mask)
        if hole_coords.shape[0] > 0:
            min_r, min_c, max_r, max_c = *np.min(hole_coords, axis=0), *np.max(hole_coords, axis=0)
            hole_h = max_r - min_r + 1
            hole_w = max_c - min_c + 1
            if hole_h==hole_w:
                output_grid[hole_mask] = C_RED
    return output_grid


def grid_to_image(grid, cell_size):
    dim = grid.shape[0]
    image = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = image.load()
    color_map = {C_BLACK: COLOR_BLACK, C_GREY: COLOR_GREY, C_RED: COLOR_RED}
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], COLOR_BLACK)
            for i in range(cell_size):
                for j in range(cell_size):
                    pixels[c * cell_size + j, r * cell_size + i] = color
    return image


def generate_single_sample(sample_index):
    try:
        input_logic_grid = generate_logic_grids()
        output_logic_grid = apply_rule_and_get_output(input_logic_grid)
        input_image = grid_to_image(input_logic_grid, CELL_SIZE)
        output_image = grid_to_image(output_logic_grid, CELL_SIZE)
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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (终极完美版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个终极版图像对已保存到 '{DATASET_DIR}' 目录。")