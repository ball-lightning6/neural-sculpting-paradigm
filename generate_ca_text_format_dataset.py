import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_BITS = 30  # 元胞自动机的宽度
TOTAL_LAYERS = 2  # 总演化层数
DATASET_SIZE = 500000

# --- 文件名 ---
OUTPUT_FILE = f'ca_rule110_n{NUM_BITS}_l{TOTAL_LAYERS}_text_format.jsonl'

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
print("=" * 70)
print(f"元胞自动机 Rule 110 - 自回归文本格式数据集生成器")
print("=" * 70)
print(f"输入宽度: {NUM_BITS}")
print(f"总演化层数: {TOTAL_LAYERS}")
print(f"输出格式: 'Evolve this: [S_0] -> [S_6]'")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：元胞自动机模拟器 ---
# ==============================================================================
def apply_rule110(left, center, right):
    """应用Rule 110规则"""
    pattern = (left << 2) | (center << 1) | right
    rule110_map = {7: 0, 6: 1, 5: 1, 4: 0, 3: 1, 2: 1, 1: 1, 0: 0}
    return rule110_map.get(pattern, 0)


def get_final_state(initial_state_list, total_layers):
    """
    只计算并返回最终的演化状态。
    """
    n = len(initial_state_list)
    current_state = list(initial_state_list)

    for _ in range(total_layers):
        next_state = [0] * n
        for i in range(n):
            left = current_state[(i - 1 + n) % n]
            center = current_state[i]
            right = current_state[(i + 1 + n) % n]
            next_state[i] = apply_rule110(left, center, right)
        current_state = next_state

    return current_state


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets():
    print("\n--- 开始生成自回归文本数据集 ---")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            # 生成随机的初始状态
            initial_state = [random.randint(0, 1) for _ in range(NUM_BITS)]

            # 计算最终状态
            final_state = get_final_state(initial_state, TOTAL_LAYERS)

            # 将状态转换为字符串
            initial_str = "".join(map(str, initial_state))
            final_str = "".join(map(str, final_state))

            # 构建适合自回归模型学习的文本格式
            # 使用特殊的分隔符，帮助模型区分问题和答案
            record_text = f"Evolve this: {initial_str} -> {final_str}"

            # 写入jsonl文件
            f.write(json.dumps({"text": record_text}) + '\n')

    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample_state = [random.randint(0, 1) for _ in range(NUM_BITS)]
    sample_final_state = get_final_state(sample_state, TOTAL_LAYERS)
    sample_text = f"Evolve this: {''.join(map(str, sample_state))} -> {''.join(map(str, sample_final_state))}"
    sample_text = f"{''.join(map(str, sample_state))} -> {''.join(map(str, sample_final_state))}"

    print(json.dumps({"text": sample_text}, indent=2))


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()