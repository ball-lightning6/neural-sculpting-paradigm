import json
import random
from tqdm import tqdm
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_COLUMNS_N = 10
BITS_PER_HEIGHT = 4  # 决定了最大高度 (0-7)
DATASET_SIZE = 500000

# --- 文件名 ---
OUTPUT_FILE = f'rain_water_{NUM_COLUMNS_N}n_{BITS_PER_HEIGHT}b_final_showdown.jsonl'

# ==============================================================================
# --- 2. 标签维度计算与脚本信息打印 ---
# ==============================================================================
MAX_HEIGHT = 2 ** BITS_PER_HEIGHT - 1
BITS_PER_INDEX = math.ceil(math.log2(NUM_COLUMNS_N)) if NUM_COLUMNS_N > 1 else 1

# --- 解释A (DP) ---
EXPLAIN_A_LEN = 2 * NUM_COLUMNS_N * BITS_PER_HEIGHT

# --- 解释B (Stack Trace) ---
# 最多有 N-1 次有效的雨水计算。每次计算记录 left_idx, right_idx, top_height
# (N-1) * (BITS_PER_INDEX + BITS_PER_INDEX + BITS_PER_HEIGHT)
# 我们用一个固定长度 N 的列表来存储这些事件
EXPLAIN_B_LEN = NUM_COLUMNS_N * (2 * BITS_PER_INDEX + BITS_PER_HEIGHT)

# --- 解释C (Two Pointers Trace) ---
# 最多有 N-1 次迭代。每次记录 leftMax 或 rightMax
# 我们记录每一次迭代时，被用来计算ans的那个max值
EXPLAIN_C_LEN = NUM_COLUMNS_N * BITS_PER_HEIGHT

# --- 主模型目标 ---
FINAL_ANSWER_LEN = NUM_COLUMNS_N * BITS_PER_HEIGHT

print("=" * 70)
print(f"接雨水 - “终极对决”数据集生成器")
print("=" * 70)
print(f"柱子数量: {NUM_COLUMNS_N}, 高度位数: {BITS_PER_HEIGHT}")
print(f"将为每个样本生成4种标签:")
print(f"  - 1. final_answer (主模型目标, {FINAL_ANSWER_LEN} bits)")
print(f"  - 2. explain_dp (DP解释, {EXPLAIN_A_LEN} bits)")
print(f"  - 3. explain_stack (单调栈解释, {EXPLAIN_B_LEN} bits)")
print(f"  - 4. explain_tp (双指针解释, {EXPLAIN_C_LEN} bits)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：三种不同的信息完备解释器 ---
# ==============================================================================

def generate_heights(n, max_h):
    return [random.randint(0, max_h) for _ in range(n)]


def explain_dp(height, n, bits_h):
    left_max = [0] * n;
    left_max[0] = height[0]
    for i in range(1, n): left_max[i] = max(left_max[i - 1], height[i])
    right_max = [0] * n;
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1): right_max[i] = max(right_max[i + 1], height[i])

    l_str = "".join([format(h, f'0{bits_h}b') for h in left_max])
    r_str = "".join([format(h, f'0{bits_h}b') for h in right_max])
    return [int(bit) for bit in (l_str + r_str)]


def explain_stack(height, n, bits_h, bits_idx):
    trace = []  # 存储 (left_idx, right_idx, top_height) 元组
    stack = []
    for i, h in enumerate(height):
        while stack and h > height[stack[-1]]:
            top_idx = stack.pop()
            if not stack: break
            left_idx = stack[-1]
            trace.append((left_idx, i, height[top_idx]))
        stack.append(i)

    # 将trace编码为定长扁平列表
    flat_list = []
    for i in range(n):
        if i < len(trace):
            l_idx, r_idx, t_h = trace[i]
            l_str = format(l_idx, f'0{bits_idx}b')
            r_str = format(r_idx, f'0{bits_idx}b')
            h_str = format(t_h, f'0{bits_h}b')
            flat_list.extend([int(b) for b in (l_str + r_str + h_str)])
        else:  # 用0填充
            flat_list.extend([0] * (2 * bits_idx + bits_h))

    assert len(flat_list) == EXPLAIN_B_LEN
    return flat_list


def explain_two_pointers(height, n, bits_h):
    trace = [0] * n  # 记录每一步用于计算ans的max值
    left, right = 0, n - 1
    left_max = right_max = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                trace[left] = left_max  # 记录
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                trace[right] = right_max  # 记录
            right -= 1

    flat_list_str = "".join([format(h, f'0{bits_h}b') for h in trace])
    flat_list = [int(bit) for bit in flat_list_str]
    assert len(flat_list) == EXPLAIN_C_LEN
    return flat_list


def solve_per_cell(height, n):  # 主模型目标
    if n == 0: return []
    water = [0] * n
    left_max = [0] * n;
    left_max[0] = height[0]
    for i in range(1, n): left_max[i] = max(left_max[i - 1], height[i])
    right_max = [0] * n;
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1): right_max[i] = max(right_max[i + 1], height[i])
    for i in range(n):
        water[i] = min(left_max[i], right_max[i]) - height[i]
    return water


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets():
    print("\n--- 开始生成“接雨水终极对决”数据集 ---")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            # 1. 生成随机高度
            heights = generate_heights(NUM_COLUMNS_N, MAX_HEIGHT)
            input_str = "".join([format(h, f'0{BITS_PER_HEIGHT}b') for h in heights])

            # 2. 计算主模型目标
            water_per_cell = solve_per_cell(heights, NUM_COLUMNS_N)
            final_answer_label = [int(b) for b in "".join([format(w, f'0{BITS_PER_HEIGHT}b') for w in water_per_cell])]

            # 3. 生成三种信息完备的解释
            exp_a = explain_dp(heights, NUM_COLUMNS_N, BITS_PER_HEIGHT)
            exp_b = explain_stack(heights, NUM_COLUMNS_N, BITS_PER_HEIGHT, BITS_PER_INDEX)
            exp_c = explain_two_pointers(heights, NUM_COLUMNS_N, BITS_PER_HEIGHT)

            # 4. 写入文件
            record = {
                "input": input_str,
                "final_answer": final_answer_label,
                "explain_dp": exp_a,
                "explain_stack": exp_b,
                "explain_tp": exp_c
            }
            f.write(json.dumps(record) + '\n')

    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()