import json
import random
from tqdm import tqdm
import math


class MaxHeightSolver:
    """
    算法预言机：使用双向动态规划解决“最高建筑高度”问题。
    这个类保持不变，因为它本来就计算出了完整的高度剖面。
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


def create_dataset_full_profile(num_samples, n, max_h, output_path):
    """
    【最终版】创建数据集，具有固定的建筑数量n，并输出完整的、最优的高度剖面。

    输入: 由n个建筑的限高（二进制表示）拼接成的字符串。
    输出: 一个代表最终n个建筑高度的二进制编码的0/1整数列表。
          输入和输出的长度将完全相同。
    """
    print(f"--- 开始生成 '最高建筑高度' 数据集 (Seq2Seq, 完整剖面) ---")
    print(f"固定建筑数 (n): {n}, 最大限制高度 (max_h): {max_h}, 目标样本数量: {num_samples}")

    all_samples = []
    seen_inputs = set()

    # --- 关键点: 输入和输出使用相同的编码位数，确保长度一致 ---
    bit_count = max(n - 1, max_h).bit_length()
    print(f"输入和输出的高度编码都将使用 {bit_count} 位。")
    print(f"输入/输出序列总长度将为: {n} * {bit_count} = {n * bit_count}")

    with tqdm(total=num_samples, desc="生成不重复样本") as pbar:
        while len(all_samples) < num_samples:
            # 1. 生成随机限制 (逻辑不变)
            num_restrictions = random.randint(0, n - 1)
            restricted_ids = random.sample(range(2, n + 1), num_restrictions)
            restrictions = [[idx, random.randint(0, max_h)] for idx in restricted_ids]

            # 2. 构建输入字符串 (逻辑不变)
            initial_heights = [n - 1] * n
            initial_heights[0] = 0
            for idx, h in restrictions:
                initial_heights[idx - 1] = min(initial_heights[idx - 1], h)

            input_parts = [f'{h_limit:0{bit_count}b}' for h_limit in initial_heights]
            input_str = "".join(input_parts)

            if input_str in seen_inputs:
                continue
            seen_inputs.add(input_str)

            # 3. 使用算法预言机求解 (逻辑不变)
            solver = MaxHeightSolver(n, restrictions)
            solution_heights = solver.solve()

            # 4. 构建输出 (这是核心修改点)
            # --- 修改点: 输出整个高度剖面，而非仅仅是峰值 ---
            output_parts = []
            for h in solution_heights:
                output_parts.append(f'{h:0{bit_count}b}')

            output_bin_str = "".join(output_parts)
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
    # --- 沿用上次成功的配置 ---
    NUM_SAMPLES = 2000000  # 增加样本量以应对更复杂的任务
    FIXED_N = 32  # 固定16个建筑
    MAX_HEIGHT = 31  # 最大限制高度15
    OUTPUT_FILE = 'max_building_full_profile_big_32_dataset.jsonl'

    create_dataset_full_profile(
        num_samples=NUM_SAMPLES,
        n=FIXED_N,
        max_h=MAX_HEIGHT,
        output_path=OUTPUT_FILE
    )
