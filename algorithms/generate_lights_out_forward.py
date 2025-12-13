# generate_lights_out.py
import json
import random
from tqdm import tqdm
import numpy as np

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 1_000_000
    ROWS = 5
    COLS = 5

    # --- 自动计算的参数 ---
    GRID_SIZE = ROWS * COLS

    # --- 文件名 ---
    OUTPUT_FILE = f"lights_out_{ROWS}x{COLS}_forward.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: "Lights Out" 物理模拟器 ---
# ==============================================================================
def run_lights_out_forward(press_grid: np.ndarray):
    """
    根据按压方案，计算最终的灯光效果。
    """
    rows, cols = press_grid.shape
    final_lights = np.zeros_like(press_grid, dtype=int)

    # 定义"十字形"影响的相对坐标
    # (dx, dy) = (0,0) -> 当前点
    # (0,1), (0,-1), (1,0), (-1,0) -> 邻居
    # directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # 遍历所有可能的按压点
    for r in range(rows):
        for c in range(cols):
            # 如果这个点被按压了
            if press_grid[r, c] == 1:
                # 对其影响范围内的所有灯，进行状态翻转 (XOR)
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc

                    # 检查是否在边界内
                    if 0 <= nr < rows and 0 <= nc < cols:
                        final_lights[nr, nc] = 1 - final_lights[nr, nc] # 1-x 等价于 x XOR 1

    return final_lights

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个 (按压方案 -> 最终灯效) 的数据对。"""

    # 1. 随机生成一个按压方案 (输入)
    press_grid = np.random.randint(0, 2, size=(cfg.ROWS, cfg.COLS))
    input_str = "".join(map(str, press_grid.flatten()))

    # 2. 计算最终的灯光效果 (输出)
    final_lights_grid = run_lights_out_forward(press_grid)
    output_list = final_lights_grid.flatten().tolist()

    return {
        "input": input_str,
        "output": output_list
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()

    print("=" * 70)
    print(f"Lights Out (正向问题) - {cfg.ROWS}x{cfg.COLS} - 数据集生成器")
    print("=" * 70)

    input_dim = cfg.GRID_SIZE
    output_dim = cfg.GRID_SIZE

    print(f"输入维度 (按压方案): {input_dim}")
    print(f"输出维度 (最终灯效): {output_dim}")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)

    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

    # 验证一个例子
    print("\n--- 样本逻辑验证 ---")
    test_press = np.zeros((cfg.ROWS, cfg.COLS), dtype=int)
    test_press[2, 2] = 1 # 只按压中心点
    expected_lights = run_lights_out_forward(test_press)

    print("测试输入 (只按压中心点):")
    print(test_press)
    print("\n预期输出 (中心十字亮起):")
    print(expected_lights)

    # 注意：根据当前规则，按压点本身不翻转
    # 所以中心点(2,2)保持为0，只有邻居会翻转

if __name__ == "__main__":
    main()