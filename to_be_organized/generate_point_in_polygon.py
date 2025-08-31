import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 多边形的顶点数
NUM_VERTICES_N = 7
# 每个坐标值（x或y）用多少个bit表示
BITS_PER_COORD = 4  # 坐标范围 0 到 1023

DATASET_SIZE = 500000

TRAIN_FILE = f'point_in_polygon_n{NUM_VERTICES_N}_b{BITS_PER_COORD}_train.jsonl'
EVAL_FILE = f'point_in_polygon_n{NUM_VERTICES_N}_b{BITS_PER_COORD}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入是 (n+1)个点，每个点2个坐标，每个坐标m位
INPUT_LEN = (NUM_VERTICES_N + 1) * 2 * BITS_PER_COORD
# 输出是单个bit (True/False)
OUTPUT_LEN = 1

print("=" * 70)
print(f"     点是否在多边形内 - 数据集生成器")
print("=" * 70)
print(f"多边形顶点数: {NUM_VERTICES_N}, 坐标位数: {BITS_PER_COORD}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'")
print(f"输出格式: {OUTPUT_LEN}个bit (1=在内部, 0=在外部)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (射线法) ---
# ==============================================================================
all_set=set()
zero_count = 0

def generate_polygon_and_point(n, max_coord):
    global zero_count
    """随机生成一个不自交的多边形和一个测试点"""
    # 为了保证生成的多边形不那么“怪异”，我们可以用一些策略
    # 这里我们用一个简单的方法：在一个圆上随机取点，然后略微扰动
    center_x, center_y = max_coord / 2, max_coord / 2
    radius = max_coord / 3
    while True:

        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n)])

        polygon = []
        for angle in angles:
            x = int(center_x + radius * math.cos(angle) + random.uniform(-radius / 4, radius / 4))
            y = int(center_y + radius * math.sin(angle) + random.uniform(-radius / 4, radius / 4))
            # 确保坐标在范围内
            x = max(0, min(max_coord, x))
            y = max(0, min(max_coord, y))
            polygon.append((x, y))

        # 生成一个随机测试点
        test_point = (random.randint(0, max_coord), random.randint(0, max_coord))
        polygon_tuple = tuple(map(tuple, polygon))
        if (polygon_tuple, test_point) not in all_set:
            all_set.add((polygon_tuple, test_point))
            if not is_inside_polygon(polygon, test_point):
                zero_count+=1
                if zero_count>=DATASET_SIZE//2:
                    continue
            return polygon, test_point


def is_inside_polygon(polygon, point):
    """
    使用射线法（Ray Casting Algorithm）判断点是否在多边形内部。
    """
    n = len(polygon)
    x, y = point
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y!=p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x==p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def process_sample(n_vertices, bits_per_coord):
    """生成一个完整的 (输入, 输出) 数据对。"""
    max_coord = 2 ** bits_per_coord - 1
    polygon, test_point = generate_polygon_and_point(n_vertices, max_coord)

    # 编码输入
    all_points = polygon + [test_point]
    input_str_list = []
    for px, py in all_points:
        input_str_list.append(format(px, f'0{bits_per_coord}b'))
        input_str_list.append(format(py, f'0{bits_per_coord}b'))
    input_str = "".join(input_str_list)

    # 计算答案
    result = is_inside_polygon(polygon, test_point)

    # 输出是单个bit
    output_label = [1 if result else 0]

    return {"input": input_str, "output": output_label}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, n_vertices, bits_per_coord):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(n_vertices, bits_per_coord))

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略)
    random.shuffle(records)
    train_size = int(len(records) * 1)#0.9)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in data: f.write(json.dumps(record) + '\n')
        print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        with open(path[1], 'w') as f:
            for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_VERTICES_N, BITS_PER_COORD)