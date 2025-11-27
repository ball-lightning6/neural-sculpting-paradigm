import json
import random
import heapq
import numpy as np
from tqdm import tqdm


def solve_trapping_rain_water_3d(height_map):
    """
    使用最小堆（优先队列）算法解决“接雨水 II”问题。
    返回一个与height_map同样大小的矩阵，其中每个元素是该位置最终的水面高度。

    Args:
        height_map (list of list of int): 代表地形高度的二维矩阵。

    Returns:
        list of list of int: 代表每个位置最终水面高度（地面+水）的二维矩阵。
    """
    if not height_map or not height_map[0]:
        return []

    m, n = len(height_map), len(height_map[0])
    visited = [[False for _ in range(n)] for _ in range(m)]
    pq = []

    water_levels = [row[:] for row in height_map]

    for r in range(m):
        for c in range(n):
            if r==0 or r==m - 1 or c==0 or c==n - 1:
                heapq.heappush(pq, (height_map[r][c], r, c))
                visited[r][c] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while pq:
        height, r, c = heapq.heappop(pq)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                visited[nr][nc] = True
                new_water_level = max(height, height_map[nr][nc])
                water_levels[nr][nc] = new_water_level
                heapq.heappush(pq, (new_water_level, nr, nc))

    return water_levels


def generate_one_sample(m, n, k):
    """
    生成一个单独的、格式化的训练样本，尺寸固定。

    Args:
        m (int): 固定的矩阵行数。
        n (int): 固定的矩阵列数。
        k (int): 用于表示高度的比特位数。

    Returns:
        dict: 包含 "input" 和 "output" 字段的字典。
    """
    max_height = 2 ** k - 1
    height_map = [[random.randint(0, max_height) for _ in range(n)] for _ in range(m)]

    water_levels = solve_trapping_rain_water_3d(height_map)

    # --- 格式化输入 ---
    input_bits = []
    for r in range(m):
        for c in range(n):
            height = height_map[r][c]
            input_bits.append(format(height, f'0{k}b'))
    input_str = "".join(input_bits)

    # --- 格式化输出 ---
    # 这里我们直接生成每个格子的蓄水量，而不是像之前那样生成一个0/1 mask
    # 这更符合题目要求 "输出仍然是对应的每个格子乘的雨水量"
    # 但由于你的范式是预测每个(cell, height)是否有水，我们还是按原来的方式生成
    output_list = []
    for r in range(m):
        for c in range(n):
            ground_height = height_map[r][c]
            water_surface = water_levels[r][c]

            # 每个格子有k个bit，代表k个高度层
            column_bits = [0] * k
            # 从地面高度到水面高度（不含水面）的区间，就是水
            for h in range(ground_height, water_surface):
                if h < k:
                    column_bits[h] = 1  # 第h层有水
            output_list.extend(column_bits)

    return {"input": input_str, "output": output_list}


def create_dataset(num_samples, output_path, m, n, k):
    """
    创建完整的数据集文件，尺寸固定。

    Args:
        num_samples (int): 要生成的数据集样本数量。
        output_path (str): 输出的 .jsonl 文件路径。
        m (int): 固定的矩阵行数。
        n (int): 固定的矩阵列数。
        k (int): 高度的比特位数。
    """
    total_len = m * n * k
    print(f"开始生成 {num_samples} 个样本...")
    print(f"固定尺寸: m={m}, n={n}, 高度比特数 k={k}")
    print(f"每个样本的输入/输出向量长度将是: {m} * {n} * {k} = {total_len}")

    all_samples = []
    seen_inputs = set()

    with tqdm(total=num_samples, desc="生成样本") as pbar:
        while len(all_samples) < num_samples:
            sample = generate_one_sample(m, n, k)
            if sample["input"] not in seen_inputs:
                seen_inputs.add(sample["input"])
                all_samples.append(sample)
                pbar.update(1)

    print("样本生成完毕，开始打乱顺序...")
    random.shuffle(all_samples)

    print(f"开始写入到文件: {output_path}")
    with open(output_path, 'w') as f:
        for sample in tqdm(all_samples, desc="写入文件"):
            f.write(json.dumps(sample) + '\n')

    print("数据集创建成功！")
    print(f"总计 {len(all_samples)} 条不重复数据已写入 {output_path}")


if __name__=='__main__':
    # --- 配置参数 ---
    NUM_SAMPLES_TO_GENERATE = 500000
    OUTPUT_FILE = 'trapping_rain_water_3d_fixed_size_dataset.jsonl'

    # --- 固定尺寸 ---
    # 所有样本都将是 8x8 的矩阵
    M_FIXED = 8
    N_FIXED = 8

    # 用多少个bit来表示高度。k=8意味着最大高度是 2^8 - 1 = 255
    # 这也决定了输入/输出向量的第三个维度
    K_BITS = 3

    # --- 开始执行 ---
    create_dataset(
        num_samples=NUM_SAMPLES_TO_GENERATE,
        output_path=OUTPUT_FILE,
        m=M_FIXED,
        n=N_FIXED,
        k=K_BITS
    )

    # --- 验证生成的文件 (可选) ---
    print("\n--- 验证文件第一行 ---")
    with open(OUTPUT_FILE, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
        expected_len = M_FIXED * N_FIXED * K_BITS
        print(f"输入 (input) 长度: {len(data['input'])}")
        print(f"输出 (output) 长度: {len(data['output'])}")
        print(f"期望长度: {expected_len}")
        assert len(data['input'])==expected_len, "输入长度与期望不符！"
        assert len(data['output'])==expected_len, "输出长度与期望不符！"
    print("验证通过！")

