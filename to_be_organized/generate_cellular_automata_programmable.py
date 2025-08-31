import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 配置区域 ---
# ==============================================================================

# --- 数据集设置 ---
TARGET_NUM_SAMPLES = 300000  # 任务更复杂，可以适当增加样本量
OUTPUT_FILE = "programmable_ca_dataset.jsonl"

# --- 元胞自动机参数 ---
CA_WIDTH = 30  # 状态长度
EVOLUTION_STEPS = 3  # 演化层数

# --- 【新】规则设置 ---
# 包含一些著名的、行为各异的规则
RULES_TO_INCLUDE = [30, 54, 60, 90, 102, 110, 150, 184]


# ==============================================================================
# --- 核心代码 ---
# ==============================================================================

class ProgrammableCASimulator:
    def __init__(self, width, steps):
        self.width = width
        self.steps = steps

    def apply_rule(self, state, rule_number):
        """
        对给定的状态应用一次指定的规则。
        """
        next_state = [0] * self.width
        # 将规则编号转换为8位二进制，用于查找
        rule_bits = format(rule_number, '08b')

        for i in range(self.width):
            # 获取邻居状态 (使用循环边界)
            left = state[(i - 1 + self.width) % self.width]
            center = state[i]
            right = state[(i + 1) % self.width]

            # 将邻居模式 (e.g., 1,1,0) 转换为十进制索引 (e.g., 6)
            pattern_index = (left << 2) + (center << 1) + right

            # 从规则的二进制表示中查找输出
            # rule_bits是'11101000'这样的字符串，所以索引是反过来的 (7-index)
            output_bit = int(rule_bits[7 - pattern_index])

            next_state[i] = output_bit

        return next_state

    def generate_one_instance(self, rule_number):
        """
        为单个规则生成一个输入-输出对。
        """
        # 1. 生成随机初始状态
        current_state = [random.randint(0, 1) for _ in range(self.width)]
        initial_state_str = "".join(map(str, current_state))

        # 2. 演化N步
        for _ in range(self.steps):
            current_state = self.apply_rule(current_state, rule_number)

        final_state_str = "".join(map(str, current_state))

        # 3. 格式化输入
        # 将规则编号转为8位二进制字符串
        rule_binary_str = format(rule_number, '08b')

        # 拼接成最终的输入字符串
        # 格式: 8位规则 + 100位状态
        input_str = rule_binary_str + initial_state_str

        return {"input": input_str, "output": final_state_str}

    def generate_dataset(self):
        """
        生成包含多种规则的混合数据集。
        """
        print("=" * 60)
        print("开始生成“可编程”元胞自动机数据集...")
        print(f"包含规则: {RULES_TO_INCLUDE}")
        print(f"输入长度: 8 (rule) + {self.width} (state) = {8 + self.width}")
        print("=" * 60)

        all_data = []
        # 为了数据集平衡，我们为每个规则生成等量的样本
        samples_per_rule = TARGET_NUM_SAMPLES // len(RULES_TO_INCLUDE)

        with tqdm(total=TARGET_NUM_SAMPLES, desc="生成数据") as pbar:
            for rule in RULES_TO_INCLUDE:
                for _ in range(samples_per_rule):
                    all_data.append(self.generate_one_instance(rule))
                    pbar.update(1)

        # 补齐可能因整除丢失的样本
        while len(all_data) < TARGET_NUM_SAMPLES:
            rule = random.choice(RULES_TO_INCLUDE)
            all_data.append(self.generate_one_instance(rule))
            pbar.update(1)

        print("\n数据生成完毕，正在进行全局洗牌...")
        random.shuffle(all_data)

        print(f"正在写入文件: {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            for data_point in tqdm(all_data, desc="写入文件"):
                f.write(json.dumps(data_point) + '\n')

        print("\n🎉🎉🎉 可编程数据集生成完毕！ 🎉🎉🎉")
        # 可视化一个样本
        print("\n--- 样本示例 ---")
        sample = all_data[0]
        print(f"Input: {sample['input']}")
        print(f"  - Rule (bits 0-7):   {sample['input'][:8]}")
        print(f"  - State (bits 8-): {sample['input'][8:]}")
        print(f"Output: {sample['output']}")


if __name__=='__main__':
    simulator = ProgrammableCASimulator(width=CA_WIDTH, steps=EVOLUTION_STEPS)
    simulator.generate_dataset()
