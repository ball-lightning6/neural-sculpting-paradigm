import json
import random
from tqdm import tqdm
import math


class MaxHeightSolver:
    """
    算法预言机：使用双向动态规划解决“最高建筑高度”问题。
    返回每栋建筑在最优方案下的确切高度。
    """

    def __init__(self, n, restrictions):
        self.n = n
        self.restrictions_map = {r[0] - 1: r[1] for r in restrictions}

    def solve(self):
        heights = [self.n - 1] * self.n
        heights[0] = 0

        for idx, max_h in self.restrictions_map.items():
            if 0 <= idx < self.n:
                heights[idx] = min(heights[idx], max_h)

        for i in range(1, self.n):
            heights[i] = min(heights[i], heights[i - 1] + 1)

        for i in range(self.n - 2, -1, -1):
            heights[i] = min(heights[i], heights[i + 1] + 1)

        return heights


def create_dataset_fixed_n(num_samples, n, max_h, output_path):
    """
    【修改版】创建数据集，具有固定的建筑数量n，并预测最终的最高高度。

    输入: 由n个建筑的限高（二进制表示）拼接成的字符串。
    输出: 一个代表最终峰值高度的二进制编码的0/1整数列表。
    """
    print(f"--- 开始生成 '最高建筑高度' 数据集 (固定n, 预测峰值) ---")
    print(f"固定建筑数 (n): {n}, 最大限制高度 (max_h): {max_h}, 目标样本数量: {num_samples}")

    all_samples = []
    seen_inputs = set()

    # --- 修改点 1: 预先计算编码位数 ---
    # 输入编码位数 (m)
    input_bit_count = max(n - 1, max_h).bit_length()
    # 输出编码位数 (峰值最高为n-1)
    output_bit_count = (n - 1).bit_length()
    print(f"输入高度编码将使用 {input_bit_count} 位。")
    print(f"输出峰值编码将使用 {output_bit_count} 位。")

    with tqdm(total=num_samples, desc="生成不重复样本") as pbar:
        while len(all_samples) < num_samples:
            # 1. 生成随机限制
            # --- 修改点 2: 不限制restriction数量 ---
            num_restrictions = random.randint(0, n - 1)
            restricted_ids = random.sample(range(2, n + 1), num_restrictions)
            restrictions = [[idx, random.randint(0, max_h)] for idx in restricted_ids]

            # 2. 构建输入字符串
            initial_heights = [n - 1] * n
            initial_heights[0] = 0
            for idx, h in restrictions:
                initial_heights[idx - 1] = min(initial_heights[idx - 1], h)

            input_parts = [f'{h_limit:0{input_bit_count}b}' for h_limit in initial_heights]
            input_str = "".join(input_parts)

            if input_str in seen_inputs:
                continue
            seen_inputs.add(input_str)

            # 3. 使用算法预言机求解
            solver = MaxHeightSolver(n, restrictions)
            solution_heights = solver.solve()
            peak_height = max(solution_heights) if solution_heights else 0

            # 4. 构建输出
            # --- 修改点 3: 输出为峰值的二进制编码 ---
            output_bin_str = f'{peak_height:0{output_bit_count}b}'
            # 转为0/1整数列表
            output_list = [int(bit) for bit in output_bin_str]

            all_samples.append({
                "input": input_str,
                "output": output_list
            })
            pbar.update(1)

    print("\n样本生成完毕，开始打乱顺序...")
    random.shuffle(all_samples)
    print("打乱完成！")

    print("开始写入文件...")
    with open(output_path, 'w') as f:
        for sample in tqdm(all_samples, desc="写入文件"):
            f.write(json.dumps(sample) + '\n')

    print(f"数据集创建成功！文件已保存至: {output_path}")


if __name__=='__main__':
    # --- 配置参数 ---
    NUM_SAMPLES = 500000
    FIXED_N = 16  # 【修改】固定建筑数量
    MAX_HEIGHT = 15  # restrictions中的最大高度限制
    OUTPUT_FILE = 'max_building_peak_fixed_n_dataset.jsonl'

    create_dataset_fixed_n(
        num_samples=NUM_SAMPLES,
        n=FIXED_N,
        max_h=MAX_HEIGHT,
        output_path=OUTPUT_FILE
    )
