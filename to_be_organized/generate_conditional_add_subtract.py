import json
import random

# ==============================================================================
# --- 1. 核心实验参数配置 ---
# ==============================================================================
# --- 计算任务定义 ---
NUM_BITS = 12  # 每个操作数的位数

# --- 数据集规模 ---
DATASET_SIZE = 200000

# ==============================================================================
# --- 实验模式选择 (在这里切换) ---
# 'PROBABILITY_MIX': 概率p混合模式
# 'INDICATOR_BIT': 指示位模式
EXPERIMENT_MODE = 'PROBABILITY_MIX'

# --- 特定模式的参数 ---
# 仅在 'PROBABILITY_MIX' 模式下有效
PROBABILITY_ADD = 0.5  # p值，从0.0到1.0

# --- 文件名 ---
if EXPERIMENT_MODE=='PROBABILITY_MIX':
    exp_name = f'adder_{NUM_BITS}bit_prob_p{str(PROBABILITY_ADD).replace(".", "")}'
else:  # INDICATOR_BIT
    exp_name = f'adder_{NUM_BITS}bit_indicator'

TRAIN_FILE = f'{exp_name}_train.jsonl'
EVAL_FILE = f'{exp_name}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输出是 N+1 位
OUTPUT_BITS = NUM_BITS + 1

print("=" * 70)
print(f"     加减法对比实验数据集生成器 (模式: {EXPERIMENT_MODE})")
print("=" * 70)
if EXPERIMENT_MODE=='PROBABILITY_MIX':
    print(f"概率混合模式: p(加法) = {PROBABILITY_ADD}")
else:
    print("指示位模式: 第一个bit决定操作 (0=加, 1=减)")
print(f"操作数位数: {NUM_BITS} bits")
print(f"输出格式: {OUTPUT_BITS} 个多标签二分类")
print("-" * 70)


# ==============================================================================
# --- 3. 核心逻辑：数据生成与编码 ---
# ==============================================================================

def generate_op_pair(num_bits):
    """随机生成一对N-bit的整数"""
    max_val = 2 ** num_bits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    return a, b


def process_and_write(records, train_path, eval_path):
    """通用的写入文件函数"""
    random.shuffle(records)
    train_size = int(len(records))# * 0.9)
    train_data, eval_data = records[:train_size], records[train_size:]

    print(f"\n正在写入 {len(train_data)} 条训练数据到 '{train_path}'...")
    with open(train_path, 'w') as f:
        for record in train_data:
            f.write(json.dumps(record) + '\n')

    print(f"正在写入 {len(eval_data)} 条评估数据到 '{eval_path}'...")
    with open(eval_path, 'w') as f:
        for record in eval_data:
            f.write(json.dumps(record) + '\n')

    print("所有文件写入完成！")


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, num_bits):
    records = []
    seen_pairs = set()

    print(f"--- 开始生成 {num_samples} 条数据 ---")

    while len(seen_pairs) < num_samples:
        a, b = generate_op_pair(num_bits)

        # 为了确保加法和减法的问题对都见过，我们用一个技巧
        # 我们可以强制a>=b，或者允许a,b任意，这里选择后者更通用
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        a_bin = format(a, f'0{num_bits}b')
        b_bin = format(b, f'0{num_bits}b')

        if EXPERIMENT_MODE=='PROBABILITY_MIX':
            if random.random() < PROBABILITY_ADD:
                # 执行加法
                op = '+'
                result = a + b
            else:
                # 执行减法 (为避免负数，我们做大数减小数)
                op = '-'
                result = abs(a - b)  # 或者 max(a,b) - min(a,b)

            input_binary = a_bin + b_bin
            output_multilabel = [int(bit) for bit in format(result, f'0{OUTPUT_BITS}b')]
            records.append({"input": input_binary, "output": output_multilabel})

        elif EXPERIMENT_MODE=='INDICATOR_BIT':
            # --- 生成加法样本 ---
            result_add = a + b
            input_add = '0' + a_bin + b_bin  # 0 指示加法
            output_add = [int(bit) for bit in format(result_add, f'0{OUTPUT_BITS}b')]
            records.append({"input": input_add, "output": output_add})

            # --- 生成减法样本 ---
            result_sub = abs(a - b)
            input_sub = '1' + a_bin + b_bin  # 1 指示减法
            output_sub = [int(bit) for bit in format(result_sub, f'0{OUTPUT_BITS}b')]
            records.append({"input": input_sub, "output": output_sub})

            # 因为每个(a,b)对生成了两个样本，所以我们只需要循环 num_samples/2 次
            if len(seen_pairs) >= num_samples / 2:
                break

    print(f"生成完毕。共 {len(records)} 条数据。")
    process_and_write(records, TRAIN_FILE, EVAL_FILE)


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, NUM_BITS)