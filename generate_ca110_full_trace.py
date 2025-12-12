import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_BITS = 30  # 元胞自动机的宽度
TOTAL_LAYERS = 90  # 总演化层数
DATASET_SIZE = 500000

# --- 文件名 ---
# 文件名明确表示了其内容：包含了完整轨迹
OUTPUT_FILE = f'autodl-tmp/ca_rule110_n{NUM_BITS}_l{TOTAL_LAYERS}_full_trace.jsonl'

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
# 计算总输出标签的长度
# (总层数 * 每个状态的位数)
# 注意：我们每3层记录一次 (t%3==2)
TOTAL_OUTPUT_BITS = (TOTAL_LAYERS // 3) * NUM_BITS

print("=" * 70)
print(f"元胞自动机 Rule 110 - “完整演化轨迹”数据集生成器")
print(f"(为 '神经心智扫描仪' 实验准备)")
print("=" * 70)
print(f"输入宽度: {NUM_BITS}")
print(f"总演化层数: {TOTAL_LAYERS}")
print(f"输出标签: 拼接 S_1 到 S_{TOTAL_LAYERS} 的所有状态")
print(f"输出标签长度: {TOTAL_OUTPUT_BITS} bits")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：元胞自动机模拟器 ---
# ==============================================================================
def apply_rule110(left, center, right):
    """应用Rule 110规则"""
    pattern = (left << 2) | (center << 1) | right
    # Rule 110: 01101110 in binary for patterns 7 down to 0
    rule110_map = {7: 0, 6: 1, 5: 1, 4: 0, 3: 1, 2: 1, 1: 1, 0: 0}
    return rule110_map.get(pattern, 0)


def generate_ca_full_trace(initial_state, total_layers):
    """
    模拟元胞自动机的演化，并记录从S_1到S_N的完整状态轨迹。
    """
    n = len(initial_state)
    current_state = list(initial_state)
    full_trace = []

    for t in range(total_layers):
        next_state = [0] * n
        for i in range(n):
            left = current_state[(i - 1 + n) % n]  # Periodic boundary conditions
            center = current_state[i]
            right = current_state[(i + 1) % n]
            next_state[i] = apply_rule110(left, center, right)

        current_state = next_state
        # 将当前层（演化结果）的状态加入到轨迹中
        if t % 3 == 2:
            full_trace.extend(current_state)

    return full_trace


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets():
    print("\n--- 开始生成完整轨迹数据集 ---")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            # 生成随机的初始状态
            initial_state = [random.randint(0, 1) for _ in range(NUM_BITS)]

            # 模拟演化并获取所有中间状态的拼接列表
            full_trace_label = generate_ca_full_trace(initial_state, TOTAL_LAYERS)

            # 将初始状态转换为字符串输入
            input_str = "".join(map(str, initial_state))

            record = {
                "input": input_str,
                "output": full_trace_label
            }
            f.write(json.dumps(record) + '\n')

    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample_state = [random.randint(0, 1) for _ in range(NUM_BITS)]
    sample_label = generate_ca_full_trace(sample_state, TOTAL_LAYERS)
    print(f"Input: len={len(sample_state)}")
    print(f"Output: len={len(sample_label)}")
    assert len(sample_label) == TOTAL_OUTPUT_BITS
    print("标签长度验证通过！")
    print("-" * 70)
    print("样本示例:")
    print(json.dumps({
        "input": "".join(map(str, sample_state)),
        "output": sample_label[:10] + ["..."]  # 只显示部分标签
    }, indent=2))


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
