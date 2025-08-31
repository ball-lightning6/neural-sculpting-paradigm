import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import multiprocessing as mp

### 1. 配置参数 ###
DATASET_DIR = "arc_jigsaw_puzzle_mine_dataset"
GRID_DIM = 28
CELL_SIZE = 8
IMG_SIZE = GRID_DIM * CELL_SIZE

COLOR_BLACK = (0, 0, 0)
COLOR_PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128), (255, 165, 0), (0, 255, 255),
                 (192, 192, 192), (240, 128, 128), (255, 0, 255), (255, 255, 255), (100, 100, 100)]
NUM_SAMPLES = 200000

C_BLACK = 0


### 2. 辅助函数 ###
def create_directories():
    os.makedirs(os.path.join(DATASET_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    print(f"目录 '{DATASET_DIR}/input' 和 '{DATASET_DIR}/output' 已准备就绪。")


def get_neighbors(r, c, h, w):
    neighbors = []
    if r > 0: neighbors.append((r - 1, c))
    if r < h - 1: neighbors.append((r + 1, c))
    if c > 0: neighbors.append((r, c - 1))
    if c < w - 1: neighbors.append((r, c + 1))
    return neighbors

def generate_piece_mine(min_size,max_size,exclude_set=None):
    all_pieces={
        1: [np.array([[1]],dtype=bool)],
        2: [np.array([[1,1]],dtype=bool)],
        3: [np.array([[1,1,1]],dtype=bool),
            np.array([[1,1],[1,0]],dtype=bool)],
        4: [np.array([[1,1,1,1]],dtype=bool),
            np.array([[1,1,1],[1,0,0]],dtype=bool),
            np.array([[1,1,1],[0,1,0]], dtype=bool),
            np.array([[1,1],[1,1]],dtype=bool),
            np.array([[0,1,1],[1,1,0]],dtype=bool)],
        5: [np.array([[1,1,1,1,1]],dtype=bool),
            np.array([[1,1,1,1],[1,0,0,0]],dtype=bool),
            np.array([[1,1,1,1],[0,1,0,0]],dtype=bool),
            np.array([[1,1,1],[1,0,0],[1,0,0]],dtype=bool),
            np.array([[1,1,1],[0,1,0],[0,1,0]],dtype=bool),
            np.array([[1,1,1],[1,0,1]], dtype=bool),
            np.array([[1,1,1],[1,1,0]],dtype=bool),
            np.array([[1,1,1,0],[0,0,1,1]],dtype=bool),
            np.array([[1,0,0],[1,1,1],[0,1,0]],dtype=bool),
            np.array([[1,0,0],[1,1,0],[0,1,1]],dtype=bool),
            np.array([[1,0,0],[1,1,1],[0,0,1]],dtype=bool),
            np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool),
        ]
    }
    while True:
        size=random.randint(min_size, max_size)
        size_idx=random.randint(0,len(all_pieces[size])-1)
        if exclude_set is None or (size,size_idx) not in exclude_set:
            break
    if exclude_set is not None:
        exclude_set.add((size,size_idx))
    return all_pieces[size][size_idx]

def generate_piece(min_size,max_size,exclude_set=None):
    # if exclude_set is None:
    #     exclude_set = set()
    while True:
        size = random.randint(min_size, max_size)
        if exclude_set is None or size not in exclude_set:
            break
    if exclude_set is not None:
        exclude_set.add(size)
    temp_dim = size * 2
    grid = np.zeros((temp_dim, temp_dim), dtype=bool)
    r, c = temp_dim // 2, temp_dim // 2
    points = {(r, c)}
    for _ in range(size - 1):
        frontier = [n for p in points for n in get_neighbors(p[0], p[1], temp_dim, temp_dim) if n not in points]
        if not frontier: break
        points.add(random.choice(frontier))

    if not points: return None
    min_r, min_c = min(p[0] for p in points), min(p[1] for p in points)
    shape_points = [(r - min_r, c - min_c) for r, c in points]
    h = max(p[0] for p in shape_points) + 1
    w = max(p[1] for p in shape_points) + 1
    shape = np.zeros((h, w), dtype=bool)
    for r, c in shape_points: shape[r, c] = True



    return shape


def place_object(grid, obj_mask, color_code, occupied_mask, avoid_area=None):
    dim_h, dim_w = grid.shape
    h, w = obj_mask.shape
    for _ in range(100):
        r = random.randint(0, dim_h - h)
        c = random.randint(0, dim_w - w)
        # 检查是否与需要避开的区域重叠
        if avoid_area is not None:
            r_avoid_s, c_avoid_s, h_avoid, w_avoid = avoid_area
            if not (r >= r_avoid_s + h_avoid or r + h <= r_avoid_s or \
                    c >= c_avoid_s + w_avoid or c + w <= c_avoid_s):
                continue

        # 检查带边距的占用情况
        if not np.any(occupied_mask[max(0, r - 1):min(dim_h, r + h + 1), max(0, c - 1):min(dim_w, c + w + 1)]):
            if color_code is not None:
                grid[r:r + h, c:c + w][obj_mask] = color_code
            occupied_mask[r:r + h, c:c + w] = True
            return True, (r, c)
    return False, None


def rotate_and_scale_and_mirror(piece_mask, rotation, mirror=False):
    rotated = np.rot90(piece_mask, k=rotation)
    h, w = rotated.shape
    scaled = np.zeros((h * 2, w * 2), dtype=bool)
    for r in range(h):
        for c in range(w):
            if rotated[r, c]:
                scaled[r * 2:r * 2 + 2, c * 2:c * 2 + 2] = True
    x_mirror_flag = random.randint(0,1)
    y_mirror_flag = random.randint(0,1)
    if mirror:
        if x_mirror_flag:
            scaled = scaled[::-1]
        if y_mirror_flag:
            scaled = scaled[:,::-1]
    return scaled


### 3. 核心生成逻辑 ###
def generate_logic_grids():
    for _ in range(100):
        dim, t_dim, t_inner_dim = GRID_DIM, 14, 12
        input_grid = np.full((dim, dim), C_BLACK, dtype=int)
        occupied_mask = np.zeros((dim, dim), dtype=bool)

        num_source_pieces = random.randint(3, 6)
        num_noise_pieces = random.randint(8, 15)
        colors_needed = 1 + num_source_pieces + 1
        if len(COLOR_PALETTE) < colors_needed: continue

        colors = random.sample(range(len(COLOR_PALETTE)), colors_needed)
        color_iter = iter(colors)

        template_bg_color = next(color_iter) + 1
        template_pos = random.choice([(0, 0), (0, dim - t_dim), (dim - t_dim, 0), (dim - t_dim, dim - t_dim)])
        r_t, c_t = template_pos

        input_grid[r_t:r_t + t_dim, c_t:c_t + t_dim] = template_bg_color
        occupied_mask[r_t:r_t + t_dim, c_t:c_t + t_dim] = True

        source_pieces_info = []
        # 【核心修正】为模板内部创建独立的占用掩码
        template_inner_grid = input_grid[r_t + 1:r_t + t_inner_dim + 1, c_t + 1:c_t + t_inner_dim + 1]
        occupied_inner = np.zeros((t_inner_dim, t_inner_dim), dtype=bool)

        ONE_COLOR = 0#next(color_iter) + 1

        piece_size_exclude_set = set()
        for _ in range(num_source_pieces):
            piece = generate_piece_mine(1,5,piece_size_exclude_set)
            if piece is None or piece.shape[0] > t_inner_dim or piece.shape[1] > t_inner_dim: continue

            if list(piece.shape)==[1,1]:
                place_times = random.randint(1,2)
                ONE_COLOR = next(color_iter) + 1
            else:
                place_times = 1

            for xxx in range(place_times):
                # 在内部放置时，使用内部的占用掩码
                for _ in range(100):
                    placed, pos = place_object(template_inner_grid, piece, C_BLACK, occupied_inner)
                    if placed:
                        break
                if placed:
                    source_pieces_info.append({'shape': piece, 'pos_in_tpl': pos})

        if not source_pieces_info: continue

        transformed_pieces_info = []
        for info in source_pieces_info:
            rotation = random.randint(1, 3)
            scaled_piece = rotate_and_scale_and_mirror(info['shape'], rotation, True)
            if list(info['shape'].shape) == [1,1]:
                color = ONE_COLOR
            else:
                color = next(color_iter) + 1
            placed, _ = place_object(input_grid, scaled_piece, color, occupied_mask, (r_t, c_t, t_dim, t_dim))
            if placed:
                transformed_pieces_info.append({'original_info': info, 'color': color})

        if len(transformed_pieces_info)!=len(source_pieces_info): continue

        noise_color = next(color_iter) + 1
        for _ in range(num_noise_pieces):
            noise_piece = generate_piece_mine(1,2)
            if noise_piece is None: continue
            # color = next(color_iter) + 1
            place_object(input_grid, noise_piece, noise_color, occupied_mask, (r_t, c_t, t_dim, t_dim))

        # 生成输出图像
        output_grid = np.full((dim, dim), template_bg_color, dtype=int)
        for info in transformed_pieces_info:
            original_piece = info['original_info']['shape']
            scaled_original = rotate_and_scale_and_mirror(original_piece, 0)
            color = info['color']
            pr, pc = info['original_info']['pos_in_tpl']
            r_out_start, c_out_start = (pr * 2) + 2, (pc * 2) + 2  # 加上边框(1)和放大(2)的偏移

            h_s, w_s = scaled_original.shape
            if r_out_start + h_s <= dim and c_out_start + w_s <= dim:
                output_grid[r_out_start:r_out_start + h_s, c_out_start:c_out_start + w_s][scaled_original] = color

        return input_grid, output_grid

    return None


def grid_to_image(grid, cell_size):
    # ...
    dim = grid.shape[0]
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), COLOR_BLACK)
    grid_img = Image.new("RGB", (dim * cell_size, dim * cell_size))
    pixels = grid_img.load()
    color_map = {C_BLACK: COLOR_BLACK}
    for i, color in enumerate(COLOR_PALETTE): color_map[i + 1] = color
    for r in range(dim):
        for c in range(dim):
            color = color_map.get(grid[r, c], (0, 0, 0))
            for i_ in range(cell_size):
                for j_ in range(cell_size):
                    pixels[c * cell_size + j_, r * cell_size + i_] = color

    offset_x = (IMG_SIZE - grid_img.width) // 2
    offset_y = (IMG_SIZE - grid_img.height) // 2
    canvas.paste(grid_img, (offset_x, offset_y))
    return canvas


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
    print(f"将使用 {num_processes} 个进程并行生成 {NUM_SAMPLES} 个样本 (最终典藏版)...")
    with mp.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES, desc="生成数据集"))
    print(f"\n数据集生成完毕！ {NUM_SAMPLES} 个最终版图像对已保存到 '{DATASET_DIR}' 目录。")