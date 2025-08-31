import json
import random
from collections import deque
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
MATRIX_SIZE_N = 6
INPUT_FORMAT = 'binary'  # 'binary' or 'hex'

DATASET_SIZE = 500000

exp_name = f'shortest_path_n{MATRIX_SIZE_N}_{INPUT_FORMAT}'
TRAIN_FILE = f'{exp_name}_train.jsonl'
EVAL_FILE = f'{exp_name}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
if INPUT_FORMAT=='binary':
    INPUT_LEN = MATRIX_SIZE_N * MATRIX_SIZE_N
else:  # hex
    if MATRIX_SIZE_N % 4!=0:
        raise ValueError("使用16进制时，矩阵大小N必须是4的倍数")
    INPUT_LEN = (MATRIX_SIZE_N * MATRIX_SIZE_N) // 4

# 输出是路径长度的二进制表示
# 路径长度最大是 N*N (虽然几乎不可能达到)
# 所以输出的位数，是表示 N*N 所需的位数
OUTPUT_BITS = math.ceil(math.log2(MATRIX_SIZE_N * MATRIX_SIZE_N + 1))

print("=" * 70)
print(f"     二进制矩阵最短路径 - 数据集生成器 (最终版)")
print("=" * 70)
print(f"矩阵大小: {MATRIX_SIZE_N}x{MATRIX_SIZE_N}")
print(f"输入格式: {INPUT_FORMAT} (长度 {INPUT_LEN})")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (0代表不连通)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 ---
# ==============================================================================
def bfs_shortest_path(grid):
    n = len(grid)
    if n > 0 and (grid[0][0]==1 or grid[n - 1][n - 1]==1):
        return 0  # 按照新定义，不连通返回0

    queue = deque([((0, 0), 1)])
    visited = set([(0, 0)])
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        (r, c), length = queue.popleft()
        if r==n - 1 and c==n - 1:
            return length
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited and grid[nr][nc]==0:
                visited.add((nr, nc))
                queue.append(((nr, nc), length + 1))

    return 0  # 按照新定义，不连通返回0

all_set=set()
# ==============================================================================
# --- 4. 数据集生成主函数 ---
# ==============================================================================
def generate_datasets(num_samples, n, input_format, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        # 随机生成矩阵，可以调整0的概率来生成更多可解的路径
        while True:
            p_zero = random.uniform(0.6, 0.9)
            grid = [[0 if random.random() < p_zero else 1 for _ in range(n)] for _ in range(n)]
            input_str = "".join(str(cell) for row in grid for cell in row)
            if input_str not in all_set:
                all_set.add(input_str)
                break
        # 计算最短路径
        path_length = bfs_shortest_path(grid)

        # 编码输入
        if input_format=='binary':
            input_str = "".join(str(cell) for row in grid for cell in row)
        else:  # hex
            binary_str = "".join(str(cell) for row in grid for cell in row)
            hex_str = hex(int(binary_str, 2))[2:].upper().zfill(INPUT_LEN)
            input_str = hex_str

        # 编码输出 (多标签二分类)
        output_binary_str = format(path_length, f'0{output_bits}b')
        output_multilabel = [int(bit) for bit in output_binary_str]

        records.append({"input": input_str, "output": output_multilabel})

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
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
    generate_datasets(DATASET_SIZE, MATRIX_SIZE_N, INPUT_FORMAT, OUTPUT_BITS)