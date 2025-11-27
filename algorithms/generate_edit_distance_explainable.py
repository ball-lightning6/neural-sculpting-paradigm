import json
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import time


# --- 核心算法函数 ---

def generate_random_binary_string(length=15):
    return "".join(random.choice("01") for _ in range(length))


# ★★★ 新增：判断并计算最优路径数量的函数 ★★★
def count_optimal_paths(s1: str, s2: str):
    """
    计算从s1到s2的最优编辑路径数量。
    返回 (距离, 路径数量)。
    """
    n, m = len(s1), len(s2)
    dist_dp = np.zeros((n + 1, m + 1), dtype=int)
    count_dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dist_dp[i, 0] = i
        count_dp[i, 0] = 1
    for j in range(m + 1):
        dist_dp[0, j] = j
        count_dp[0, j] = 1

    count_dp[0, 0] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1]==s2[j - 1] else 1

            # 计算最小距离
            options = {
                'replace': dist_dp[i - 1, j - 1] + cost,
                'delete': dist_dp[i - 1, j] + 1,
                'insert': dist_dp[i, j - 1] + 1
            }
            min_dist = min(options.values())
            dist_dp[i, j] = min_dist

            # 累加路径数量
            if options['replace']==min_dist:
                count_dp[i, j] += count_dp[i - 1, j - 1]
            if options['delete']==min_dist:
                count_dp[i, j] += count_dp[i - 1, j]
            if options['insert']==min_dist:
                count_dp[i, j] += count_dp[i, j - 1]

    return dist_dp[n, m], count_dp[n, m]


# get_levenshtein_path 函数保持我们上一版修复后的状态，它是正确的
def get_levenshtein_path(s1: str, s2: str):
    if s1==s2: return [s1]
    n, m = len(s1), len(s2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1]==s2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        cost = 0 if i > 0 and j > 0 and s1[i - 1]==s2[j - 1] else 1
        if i > 0 and j > 0 and dp[i, j]==dp[i - 1, j - 1] + cost:
            alignment.append(('align', s1[i - 1], s2[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i, j]==dp[i - 1, j] + 1:
            alignment.append(('delete', s1[i - 1], '')); i -= 1
        elif j > 0 and dp[i, j]==dp[i, j - 1] + 1:
            alignment.append(('insert', '', s2[j - 1])); j -= 1
        else:
            break
    alignment.reverse()
    path = [s1];
    current_s_list = list(s1)
    s1_ptr = 0
    for op, c1, c2 in alignment:
        if op=='delete':
            del current_s_list[s1_ptr]; path.append("".join(current_s_list))
        elif op=='align':
            if c1!=c2: current_s_list[s1_ptr] = c2; path.append("".join(current_s_list))
            s1_ptr += 1
    insert_ptr = 0
    for op, c1, c2 in alignment:
        if op=='insert': current_s_list.insert(insert_ptr, c2); path.append("".join(current_s_list))
        insert_ptr += 1
    final_path = [];
    if not path: return [s1] if s1==s2 else []
    final_path.append(path[0])
    for state in path[1:]:
        if state!=final_path[-1]: final_path.append(state)
    return final_path


# format_output_vector 函数保持不变
def format_output_vector(path, max_steps=15, m=15):
    output_bits = [];
    edit_states = path[1:]
    for i in range(max_steps):
        if i < len(edit_states):
            state_str = edit_states[i];
            str_len = len(state_str)
            state_bits = [int(char) for char in state_str]
            state_bits.extend([0] * (m - str_len))
            mask_bits = [1] * str_len + [0] * (m - str_len)
            output_bits.extend(state_bits);
            output_bits.extend(mask_bits)
        else:
            output_bits.extend([0] * (m * 2))
    return output_bits


# ★★★ 重构：工作进程现在只产出路径唯一的样本 ★★★
def worker_task(batch_size, str_len, max_edits):
    """生成一批路径唯一的样本"""
    lines = []
    attempts = 0
    while len(lines) < batch_size:
        attempts += 1
        s1 = generate_random_binary_string(str_len)
        s2 = generate_random_binary_string(str_len)
        if s1==s2: continue

        # 核心筛选逻辑
        _, path_count = count_optimal_paths(s1, s2)
        if path_count==1:
            edit_path = get_levenshtein_path(s1, s2)
            output_vector = format_output_vector(edit_path, max_steps=max_edits, m=str_len)
            input_str = s1 + s2
            sample = {"input": input_str, "output": output_vector}
            lines.append(json.dumps(sample))

    # 返回这批合格的样本和为了找到它们所做的尝试次数
    return lines, attempts


# ★★★ 重构：主函数采用迭代式生成，保证最终数量 ★★★
def generate_dataset_mp(num_samples, filename, str_len=15, max_edits=15):
    print(f"▶️ 开始生成高纯度数据集 (路径唯一 & 数量保证)...")
    print(f"  - 目标唯一样本数: {num_samples}")

    num_processes = multiprocessing.cpu_count() - 2 or 1
    print(f"  - 使用 {num_processes} 个工作进程。")

    unique_lines = set()
    total_attempts = 0

    with tqdm(total=num_samples, desc="Unique Samples") as pbar:
        while len(unique_lines) < num_samples:
            remaining = num_samples - len(unique_lines)
            # 动态计算批次大小，避免最后生成过多
            batch_size_per_worker = max(1, remaining // num_processes // 2 + 1)
            chunks = [batch_size_per_worker] * num_processes

            with multiprocessing.Pool(processes=num_processes) as pool:
                task = partial(worker_task, str_len=str_len, max_edits=max_edits)
                # 并行执行，收集合格的样本和尝试次数
                for result_lines, attempts in pool.imap_unordered(task, chunks):
                    total_attempts += attempts
                    # 更新进度条前，先记录当前set的大小
                    prev_count = len(unique_lines)
                    unique_lines.update(result_lines)
                    # 更新进度条，增量为新加入的唯一样本数
                    pbar.update(len(unique_lines) - prev_count)

    print("\n--- 生成完毕，开始后处理 ---")

    # 截取精确数量的样本并打乱
    final_lines = list(unique_lines)[:num_samples]
    print(f"  - 生成唯一样本: {len(final_lines)}")
    print(f"  - 总尝试次数: {total_attempts} (效率: {len(final_lines) / total_attempts:.2%})")
    print("  - 正在进行全局随机打乱...")
    random.shuffle(final_lines)

    print(f"--- 写入文件 {filename} ---")
    with open(filename, 'w') as f:
        for line in tqdm(final_lines, desc="Writing to file"):
            f.write(line + '\n')

    print(f"\n✅ 高质量数据集生成完毕！")


if __name__=='__main__':
    NUM_SAMPLES = 500000
    FILENAME = "edit_distance_path_unique_final.jsonl"
    STRING_LENGTH = 15
    MAX_EDIT_STEPS = 15
    generate_dataset_mp(
        num_samples=NUM_SAMPLES,
        filename=FILENAME,
        str_len=STRING_LENGTH,
        max_edits=MAX_EDIT_STEPS
    )
