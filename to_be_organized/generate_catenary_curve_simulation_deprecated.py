import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
import random
from pathlib import Path
from scipy.optimize import fsolve


# ==============================================================================
# --- 配置区域 (Config Section) ---
# ==============================================================================

class Config:
    DATA_DIR = "catenary_dataset_v3.2_DEFINITIVE_FIX"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 500
    NUM_SAMPLES_EVAL = 500

    IMG_SIZE = 224
    BG_COLOR = (255, 255, 255)
    CURVE_COLOR = (0, 255, 0)
    POINT_COLOR = (255, 0, 0)

    POINT_RADIUS = 5
    CURVE_WIDTH = 10

    FIXED_ROPE_LENGTH = 200.0
    MIN_HORIZONTAL_DISTANCE = 15


# ==============================================================================
# --- 核心物理与绘图函数 (Core Physics and Drawing Functions) ---
# ==============================================================================

def solve_catenary(p1, p2, L):
    # ... (求解器函数保持不变)
    x1, y1 = p1;
    x2, y2 = p2

    def equations(vars):
        a, b, c = vars
        if a <= 1e-6: return [1e9, 1e9, 1e9]
        eq1 = c - a * np.cosh((x1 - b) / a) - y1
        eq2 = c - a * np.cosh((x2 - b) / a) - y2
        eq3 = abs(a * (np.sinh((x2 - b) / a) - np.sinh((x1 - b) / a))) - L
        return [eq1, eq2, eq3]

    initial_guess = [max(abs(x1 - x2), abs(y1 - y2)), (x1 + x2) / 2, min(y1, y2) + L / 2]
    solution, _, ier, _ = fsolve(equations, initial_guess, full_output=True, xtol=1e-6)
    return solution if ier==1 and solution[0] > 0 else None


def get_curve_points(p1, p2, params):
    """根据参数计算曲线上的一系列点。"""
    a, b, c = params
    x_start, x_end = sorted([p1[0], p2[0]])
    x_coords = np.linspace(x_start, x_end, 200)  # 增加点数以提高精度
    y_coords = c - a * np.cosh((x_coords - b) / (a + 1e-9))
    return list(zip(x_coords, y_coords))


# *** 新增: 物理现实检验器 (关键修正) ***
def is_solution_valid(p1, p2, params, target_length):
    """
    验证一个解是否物理上和几何上都正确。
    """
    curve_points = get_curve_points(p1, p2, params)

    # 检验1: 终点连接检验
    # 确保曲线的终点与锚点p2足够接近
    # 我们需要根据p1和p2的x坐标大小来确定哪个是终点
    end_point_of_curve = curve_points[-1] if p1[0] < p2[0] else curve_points[0]
    target_end_point = p2 if p1[0] < p2[0] else p1

    endpoint_distance = np.linalg.norm(np.array(end_point_of_curve) - np.array(target_end_point))
    if endpoint_distance > 5.0:  # 允许5个像素的误差
        return False

    # 检验2: 弧长守恒检验
    # 计算生成曲线的实际弧长
    arc_length = np.sum(np.linalg.norm(np.diff(curve_points, axis=0), axis=1))
    if abs(arc_length - target_length) > target_length * 0.05:  # 允许5%的误差
        return False

    return True  # 只有通过所有检验的才是好解


def draw_image(config, p1, p2, curve_points=None):
    img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BG_COLOR)
    draw = ImageDraw.Draw(img)
    for p in [p1, p2]:
        x, y = p
        draw.ellipse((x - config.POINT_RADIUS, y - config.POINT_RADIUS,
                      x + config.POINT_RADIUS, y + config.POINT_RADIUS), fill=config.POINT_COLOR)
    if curve_points:
        draw.line(curve_points, fill=config.CURVE_COLOR, width=config.CURVE_WIDTH)
    return img


def create_single_sample(config):
    size = config.IMG_SIZE;
    padding = 20
    while True:
        p1 = (random.randint(padding, size - padding), random.randint(padding, size - padding))
        p2 = (random.randint(padding, size - padding), random.randint(padding, size - padding))

        if np.linalg.norm(np.array(p1) - np.array(p2)) >= config.FIXED_ROPE_LENGTH - 1: continue
        if abs(p1[0] - p2[0]) < config.MIN_HORIZONTAL_DISTANCE: continue

        params = solve_catenary(p1, p2, config.FIXED_ROPE_LENGTH)
        if params is None: continue

        # *** 在这里使用我们的现实检验器 ***
        if not is_solution_valid(p1, p2, params, config.FIXED_ROPE_LENGTH):
            continue

        # 边界检查
        curve_points = get_curve_points(p1, p2, params)
        all_points = np.array(curve_points)
        if np.any(all_points < 0) or np.any(all_points >= size): continue

        # 所有检查通过，这是一个完美的样本
        input_image = draw_image(config, p1, p2)
        output_image = draw_image(config, p1, p2, curve_points)
        return input_image, output_image


def generate_dataset(num_samples, output_dir, name, config):
    print(f"\n正在生成 {name} 数据集 ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_samples), desc=f"生成 {name}", unit="个世界"):
        input_img, output_img = create_single_sample(config)
        input_img.save(Path(output_dir) / f"{i}_input.png")
        output_img.save(Path(output_dir) / f"{i}_output.png")


# ==============================================================================
# --- 程序入口 (Main Execution Block) ---
# ==============================================================================

if __name__=="__main__":
    cfg = Config()
    if os.path.exists(cfg.DATA_DIR):
        shutil.rmtree(cfg.DATA_DIR)
    print("=" * 60)
    print("开始生成物理推理数据集：悬链线问题 v3.2 (最终版)")
    print("新增约束: 引入'物理现实检验器'，对每个解进行交叉验证。")
    print("=" * 60)
    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
    # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)
    print(f"\n🎉🎉🎉 数据集 '{cfg.DATA_DIR}' 生成完毕！ 🎉🎉🎉")
