import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
import random
from pathlib import Path


# ==============================================================================
# --- 配置区域 (Config Section) ---
# ==============================================================================

class Config:
    DATA_DIR = "bouncing_ball_dataset_v0.4_variable_bounces"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 500  # 增加样本量以覆盖更多情况
    NUM_SAMPLES_EVAL = 1000

    IMG_SIZE = 256
    BACKGROUND_COLOR = (255, 255, 255)
    TRAJECTORY_COLOR = (255, 0, 0)

    INPUT_LINE_WIDTH = 10
    TRAJECTORY_WIDTH = 10

    BALL_RADIUS = 3
    GRAVITY = 9.8 * 20
    TIME_STEP = 0.05
    ELASTICITY_FACTOR = 0.9

    INITIAL_SPEED_RANGE = [100.0, 450.0]
    SPEED_TO_BLUE_SLOPE = (0 - 220) / (INITIAL_SPEED_RANGE[1] - INITIAL_SPEED_RANGE[0])
    SPEED_TO_BLUE_INTERCEPT = 220 - SPEED_TO_BLUE_SLOPE * INITIAL_SPEED_RANGE[0]


# ==============================================================================
# --- 物理模拟器 (无变化) ---
# ==============================================================================

def simulate_trajectory(initial_pos, initial_vel, config):
    # ... (此函数与 v0.3 完全相同)
    pos = np.array(initial_pos, dtype=float)
    vel = np.array(initial_vel, dtype=float)
    trajectory_points = [tuple(pos)]
    max_steps = 3000  # 增加步数以模拟更多弹跳
    bb= 0
    for _ in range(max_steps):
        vel[1] += config.GRAVITY * config.TIME_STEP
        pos += vel * config.TIME_STEP
        if pos[0] + config.BALL_RADIUS >= config.IMG_SIZE or pos[0] - config.BALL_RADIUS <= 0:
            break
        if pos[1] + config.BALL_RADIUS >= config.IMG_SIZE and vel[1] > 0:
            pos[1] = config.IMG_SIZE - config.BALL_RADIUS
            vel[1] *= -config.ELASTICITY_FACTOR
            bb+=1
        if pos[1] - config.BALL_RADIUS <= 0 and vel[1] < 0:
            pos[1] = config.BALL_RADIUS
            vel[1] *= -1
        is_on_floor = pos[1] + config.BALL_RADIUS >= config.IMG_SIZE - 1
        if is_on_floor and abs(vel[1]) < 1.0:
            break
        trajectory_points.append(tuple(pos))
    return trajectory_points, bb


# ==============================================================================
# --- 渲染器与主生成函数 (核心逻辑修改处) ---
# ==============================================================================
def create_single_sample(config):
    # --- 核心修改：先决定目标弹跳次数，再设计初始条件 ---
    target_bounces = random.randint(0, 3)

    while True:
        # 设定通用的起点
        x_start = config.BALL_RADIUS
        y_start = random.uniform(config.IMG_SIZE * 0.2, config.IMG_SIZE * 0.7)

        if target_bounces==0:
            # **情况一：生成 "0次弹跳" 的轨迹 (纯抛物线)**
            # 目标：让球在落地前飞出屏幕
            # 方法：设定一个较快的水平速度
            vx = random.uniform(100, 200)
            # 设定一个初始向上的垂直速度，保证是抛物线
            vy = random.uniform(-200, -50)

            # 验证：轨迹顶点是否在界内
            y_peak = y_start + (vy ** 2) / (2 * config.GRAVITY)  # vy<0, so this is subtraction
            if y_peak < config.BALL_RADIUS: continue

            # 验证：落地前是否能飞出屏幕
            time_to_peak = abs(vy) / config.GRAVITY
            y_at_peak = y_start - (vy ** 2) / (2 * config.GRAVITY)
            time_from_peak_to_floor = np.sqrt(2 * (config.IMG_SIZE - y_at_peak) / config.GRAVITY)
            total_time_to_floor = time_to_peak + time_from_peak_to_floor

            if vx * total_time_to_floor > config.IMG_SIZE:
                initial_pos = [x_start, y_start]
                initial_vel = [vx, vy]
                break  # 找到了一个有效的0弹跳轨迹
            else:
                continue  # 否则重新尝试

        else:
            # **情况二：生成 "1到5次弹跳" 的轨迹**
            # 目标：通过控制首次落点来创造多次弹跳的条件
            # 首次落点越靠前，弹跳次数越可能多
            bounce_pos_factor = random.uniform(0.2, 0.5) * (1 - target_bounces / 7.0)
            x_bounce = bounce_pos_factor * config.IMG_SIZE
            y_bounce = config.IMG_SIZE - config.BALL_RADIUS

            time_to_bounce = random.uniform(0.8, 1.5)

            vx = (x_bounce - x_start) / time_to_bounce
            vy = ((y_bounce - y_start) - 0.5 * config.GRAVITY * time_to_bounce ** 2) / time_to_bounce

            if vy < 0:
                y_peak = y_start - (vy ** 2) / (2 * config.GRAVITY)
                if y_peak < config.BALL_RADIUS:
                    continue

            initial_pos = [x_start, y_start]
            initial_vel = [vx, vy]
            break

    # 后续流程不变
    trajectory,bb = simulate_trajectory(initial_pos, initial_vel, config)
    if len(trajectory) < 10 or bb>4: return None

    input_img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BACKGROUND_COLOR)
    draw_input = ImageDraw.Draw(input_img)
    line_start = tuple(initial_pos)
    norm_vel = initial_vel / np.linalg.norm(initial_vel)
    line_end = tuple(np.array(initial_pos) + norm_vel * 80)
    initial_speed = np.linalg.norm(initial_vel)
    color_val = int(config.SPEED_TO_BLUE_SLOPE * initial_speed + config.SPEED_TO_BLUE_INTERCEPT)
    color_val = max(0, min(255, color_val))
    line_color = (0, 255-color_val, color_val)
    draw_input.line([line_start, line_end], fill=line_color, width=config.INPUT_LINE_WIDTH)

    output_img = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), config.BACKGROUND_COLOR)
    draw_output = ImageDraw.Draw(output_img)
    draw_output.line(trajectory, fill=config.TRAJECTORY_COLOR, width=config.TRAJECTORY_WIDTH)

    return input_img, output_img


# ==============================================================================
# --- 程序入口 (无变化) ---
# ==============================================================================
def generate_dataset(num_samples, output_dir, name, config):
    # ... (此函数与 v0.3 完全相同)
    print(f"\n正在生成 {name} 数据集 ({num_samples} 个样本)...")
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    pbar = tqdm(total=num_samples, desc=f"生成 {name}", unit="个世界")
    while count < num_samples:
        sample = create_single_sample(config)
        if sample:
            input_img, output_img = sample
            input_img.save(Path(output_dir) / f"{count}_input.png")
            output_img.save(Path(output_dir) / f"{count}_output.png")
            count += 1
            pbar.update(1)
    pbar.close()


if __name__=="__main__":
    cfg = Config()
    if os.path.exists(cfg.DATA_DIR):
        print(f"删除旧的数据集: {cfg.DATA_DIR}")
        shutil.rmtree(cfg.DATA_DIR)

    print("=" * 60)
    print("开始生成物理推理数据集：弹跳小球 v0.4 (可控弹跳次数版)")
    print("方法: 通过反向设计，生成包含 0 到 5 次弹跳的均衡数据集。")
    print("=" * 60)

    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, "训练集", cfg)
    # generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, "验证集", cfg)

    print(f"\n🎉🎉🎉 数据集 '{cfg.DATA_DIR}' 生成完毕！ 🎉🎉🎉")
