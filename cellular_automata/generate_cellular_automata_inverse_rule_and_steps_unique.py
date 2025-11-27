import os
import random
import json
import time
from tqdm import tqdm
import multiprocessing

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
CA_WIDTH = 30
NUM_SAMPLES = 300000
MAX_ITERATION_LAYERS = 4
OUTPUT_FILE = "ca_dynamic_unique_mp_dataset.jsonl"

# --- 多进程配置 ---
# 默认使用所有可用的CPU核心，您也可以手动指定一个数字，例如 4
NUM_WORKERS = os.cpu_count()

# ==============================================================================
# --- 2. 派生配置 ---
# ==============================================================================
ITERATION_BITS = (MAX_ITERATION_LAYERS-1).bit_length()
TOTAL_OUTPUT_BITS = 8 + ITERATION_BITS


# ==============================================================================
# --- 3. 核心逻辑 (这些函数将被每个工作进程独立调用) ---
# ==============================================================================

def apply_rule(state: list, rule_binary: list) -> list:
    width = len(state)
    next_state = [0] * width
    for i in range(width):
        left, center, right = state[i - 1], state[i], state[(i + 1) % width]
        neighborhood_index = left * 4 + center * 2 + right * 1
        rule_index = 7 - neighborhood_index
        next_state[i] = rule_binary[rule_index]
    return next_state


def generate_state_with_all_patterns(width: int) -> list:
    base_sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    if width < len(base_sequence):
        raise ValueError(f"宽度必须至少为 {len(base_sequence)}")
    remaining_len = width - len(base_sequence)
    random_padding = [random.randint(0, 1) for _ in range(remaining_len)]
    insert_pos = random.randint(0, remaining_len)
    final_state = random_padding[:insert_pos] + base_sequence + random_padding[insert_pos:]
    return final_state


def is_unique(initial_state, target_output_state, true_rule_num, true_iter_count):
    for r_num in range(256):
        for i_count in range(1, MAX_ITERATION_LAYERS + 1):
            if r_num==true_rule_num and i_count==true_iter_count:
                continue

            rule_binary = [int(bit) for bit in format(r_num, '08b')]
            current_state = initial_state
            for _ in range(i_count):
                current_state = apply_rule(current_state, rule_binary)

            if current_state==target_output_state:
                return False
    return True


# --- 新增：工作单元函数 ---
def generate_one_unique_sample(_):  # 接受一个虚拟参数以适配 imap
    """
    这个函数代表一个独立的工作单元，其目标是生成并返回一个唯一样本。
    它将被并行执行。
    """
    while True:
        rule_num = random.randint(0, 255)
        iteration_count = random.randint(1, MAX_ITERATION_LAYERS)
        initial_state = generate_state_with_all_patterns(CA_WIDTH)

        rule_binary_label = [int(bit) for bit in format(rule_num, '08b')]
        current_state = list(initial_state)  # 确保是副本
        for _ in range(iteration_count):
            current_state = apply_rule(current_state, rule_binary_label)
        output_state = current_state

        if is_unique(initial_state, output_state, rule_num, iteration_count):
            # 找到了一个唯一样本，构造并返回它
            iteration_binary_label = [int(bit) for bit in format(iteration_count-1, f'0{ITERATION_BITS}b')]
            final_output_label = rule_binary_label + iteration_binary_label
            input_string = "".join(map(str, initial_state)) + "".join(map(str, output_state))

            return {"input": input_string, "output": final_output_label}


# ==============================================================================
# --- 4. 主执行流程 (多进程调度器) ---
# ==============================================================================

def main():
    print(f"🚀 开始并行生成数据集...")
    print(f"   - 使用 {NUM_WORKERS} 个CPU核心进行工作。")
    print(f"   - 目标样本数: {NUM_SAMPLES}")

    start_time = time.time()

    # 使用 with 语句确保进程池被正确关闭
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with open(OUTPUT_FILE, 'w') as f:
            # 使用 imap_unordered 来获得最佳性能，它会一有结果就返回，无需等待其他任务
            # 用 tqdm 包装来显示进度条
            results_iterator = pool.imap_unordered(generate_one_unique_sample, [None] * NUM_SAMPLES)

            for data_point in tqdm(results_iterator, total=NUM_SAMPLES, desc="Generating Unique Samples"):
                f.write(json.dumps(data_point) + '\n')

    end_time = time.time()

    print("\n✅ 数据集生成完毕！")
    print(f"   - 文件保存在: {OUTPUT_FILE}")
    print(f"   - 总耗时: {end_time - start_time:.2f} 秒")


if __name__=="__main__":
    # 这行代码在多进程中至关重要，它能防止子进程重复执行主模块代码
    main()
