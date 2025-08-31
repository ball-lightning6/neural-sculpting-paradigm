import json
import random
import numpy as np

# ==============================================================================
# --- 1. 核心实验参数配置 ---
# ==============================================================================
# --- 卷积任务定义 ---
MAP_SIZE = 5
KERNEL_SIZE = 3
PADDING = 1

# --- “宇宙法则”定义 ---
# 生成一个一次性的、固定的卷积核
# 这个Kernel将作为“隐藏”模式下的“宇宙常数”
random.seed(42)
HIDDEN_KERNEL = np.random.randint(0, 2, size=(KERNEL_SIZE, KERNEL_SIZE))

# --- 数据集参数 ---
DATASET_SIZE = 200000

# --- 文件名 ---
VISIBLE_KERNEL_TRAIN = f'conv_map{MAP_SIZE}_kernel{KERNEL_SIZE}_visible_train.jsonl'
VISIBLE_KERNEL_EVAL = f'conv_map{MAP_SIZE}_kernel{KERNEL_SIZE}_visible_eval.jsonl'

HIDDEN_KERNEL_TRAIN = f'conv_map{MAP_SIZE}_kernel{KERNEL_SIZE}_hidden_train.jsonl'
HIDDEN_KERNEL_EVAL = f'conv_map{MAP_SIZE}_kernel{KERNEL_SIZE}_hidden_eval.jsonl'

# ==============================================================================
# --- 2. 编码与卷积计算定义 ---
# ==============================================================================
# 输入编码
INPUT_BITS_MAP = MAP_SIZE * MAP_SIZE
INPUT_BITS_KERNEL = KERNEL_SIZE * KERNEL_SIZE

# 输出编码
OUTPUT_SIZE = MAP_SIZE * MAP_SIZE
# 每个输出元素最大值是 kernel中1的个数，最坏情况是9
# ceil(log2(9+1)) = 4 bits
BITS_PER_OUTPUT_ELEMENT = 4
TOTAL_OUTPUT_BITS = OUTPUT_SIZE * BITS_PER_OUTPUT_ELEMENT

# --- 打印实验配置 ---
print("=" * 70)
print(f"     “神经卷积”实验 - 数据集生成器")
print("=" * 70)
print(f"Feature Map: {MAP_SIZE}x{MAP_SIZE}, Kernel: {KERNEL_SIZE}x{KERNEL_SIZE}, Padding: {PADDING}")
print(f"隐藏的卷积核 (Hidden Kernel):\n{HIDDEN_KERNEL}")
print("-" * 70)
print(f"可见模式输入: {INPUT_BITS_MAP + INPUT_BITS_KERNEL} bits ([Map] + [Kernel])")
print(f"隐藏模式输入: {INPUT_BITS_MAP} bits ([Map] only)")
print(f"输出格式: {OUTPUT_SIZE}个'计数器', 每个{BITS_PER_OUTPUT_ELEMENT} bits, 共{TOTAL_OUTPUT_BITS} bits")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：卷积计算与数据生成 ---
# ==============================================================================

def perform_convolution(feature_map, kernel, padding):
    """执行一次标准的2D卷积操作"""
    padded_map = np.pad(feature_map, pad_width=padding, mode='constant', constant_values=0)
    output_map = np.zeros_like(feature_map, dtype=int)

    map_h, map_w = feature_map.shape
    ker_h, ker_w = kernel.shape

    for i in range(map_h):
        for j in range(map_w):
            # 提取与kernel对应的区域
            region = padded_map[i:i + ker_h, j:j + ker_w]
            # 逐元素相乘并求和
            output_map[i, j] = np.sum(region * kernel)

    return output_map


def process_case(num_bits_map, hidden_kernel):
    """
    生成一个独立的 (输入, 输出) 案例，包含两种模式。
    """
    # 1. 随机生成一个输入特征图
    feature_map = np.random.randint(0, 2, size=(MAP_SIZE, MAP_SIZE))

    # 2. 使用隐藏的kernel计算标准输出
    output_map = perform_convolution(feature_map, hidden_kernel, PADDING)

    # 3. 编码
    # 将矩阵展平为一维字符串
    map_str = "".join(map(str, feature_map.flatten()))
    kernel_str = "".join(map(str, hidden_kernel.flatten()))

    # 为输出编码
    output_str = ""
    for val in output_map.flatten():
        output_str += format(val, f'0{BITS_PER_OUTPUT_ELEMENT}b')
    output_multilabel = [int(bit) for bit in output_str]

    return {
        "visible": {"input": map_str + kernel_str, "output": output_multilabel},
        "hidden": {"input": map_str, "output": output_multilabel}
    }


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples):
    print("\n--- 开始生成数据集 ---")

    visible_records = []
    hidden_records = []

    # 我们只需要生成不同的feature_map即可，因为kernel是固定的
    for i in range(num_samples):
        record = process_case(INPUT_BITS_MAP, HIDDEN_KERNEL)
        visible_records.append(record["visible"])
        hidden_records.append(record["hidden"])

        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(visible_records)} 条数据。")

    # --- 写入文件 ---
    def write_to_file(records, train_path, eval_path, name):
        print(f"\n--- 正在处理'{name}'数据集 ---")
        random.shuffle(records)
        train_size = int(len(records) * 1)
        train_data, eval_data = records[:train_size], records[train_size:]

        print(f"正在写入 {len(train_data)} 条训练数据到 '{train_path}'...")
        with open(train_path, 'w') as f:
            for rec in train_data: f.write(json.dumps(rec) + '\n')

        print(f"正在写入 {len(eval_data)} 条评估数据到 '{eval_path}'...")
        with open(eval_path, 'w') as f:
            for rec in eval_data: f.write(json.dumps(rec) + '\n')

    write_to_file(visible_records, VISIBLE_KERNEL_TRAIN, VISIBLE_KERNEL_EVAL, "可见Kernel")
    write_to_file(hidden_records, HIDDEN_KERNEL_TRAIN, HIDDEN_KERNEL_EVAL, "隐藏Kernel")

    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE)