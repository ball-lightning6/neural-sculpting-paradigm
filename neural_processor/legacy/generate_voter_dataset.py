# generate_neural_voter_dataset.py

import json
import random
from tqdm import tqdm
from collections import Counter
import os

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
OUTPUT_BITS = 1  # 每个独立预测结果的位数
NUM_VOTERS = 3   # 我们使用3个投票者
INPUT_BITS = OUTPUT_BITS * NUM_VOTERS # 1 * 3 = 3 bits

DATASET_SIZE = 100000
OUTPUT_FILE = "neural_voter_dataset.jsonl"

# --- 噪声参数 ---
# 模拟单个模型犯错的概率，我们可以设置得比真实情况高一点，以便生成足够的“争议”样本
SINGLE_BIT_ERROR_RATE = 0.01 # 1% 的比特位错误率

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
print("=" * 80)
print(f" “神经投票器” - 数据集生成器")
print("=" * 80)
print("Note: User feedback indicates that errors from neural CPUs are often not independent,")
print("so simple voting may not improve accuracy significantly. Better training is preferred.")
print("This script is preserved for historical context and potential future use.")
print("=" * 80)
print(f"输入格式: {NUM_VOTERS}个独立预测结果拼接 (共 {INPUT_BITS} bits)")
print(f"输出格式: 经过“智能投票”后的最终结果 ({OUTPUT_BITS} bits)")
print(f"数据集大小: {DATASET_SIZE}")
print("=" * 80)

# ==============================================================================
# --- 3. 核心逻辑：模拟“有瑕疵”的预测器 ---
# ==============================================================================

def generate_ground_truth():
    """生成一个随机的、100%正确的“真值”输出向量"""
    return [random.choice([0, 1]) for _ in range(OUTPUT_BITS)]

def introduce_noise(vector, error_rate):
    """以一定概率，翻转向量中的某些比特，模拟一个“有瑕疵”的预测"""
    noisy_vector = vector[:]
    for i in range(len(noisy_vector)):
        if random.random() < error_rate:
            noisy_vector[i] = 1 - noisy_vector[i] # 翻转 0 -> 1, 1 -> 0
    return noisy_vector

def hard_vote(vectors):
    """执行经典的“硬投票”逻辑，作为我们生成标签的依据"""
    voted_vector = []
    # 对每一个比特位进行投票
    for i in range(len(vectors[0])):
        bits_at_position_i = [v[i] for v in vectors]
        # Counter会统计每个元素出现的次数，most_common(1)返回出现最多次的元素和它的次数
        most_common_bit = Counter(bits_at_position_i).most_common(1)[0][0]
        voted_vector.append(most_common_bit)
    return voted_vector

# ==============================================================================
# --- 4. 单个样本处理与主生成函数 ---
# ==============================================================================

def sample_one():
    """
    生成一个 (输入=[pred1, pred2, pred3], 输出=voted_result) 的样本
    并控制“一致”和“有争议”样本的比例
    """
    ground_truth = generate_ground_truth()
    
    # --- 核心逻辑：控制样本类型 ---
    if random.random() < 0.5:
        # --- 生成“三者一致”的样本 ---
        # 这种情况，我们假设所有预测器都做对了
        pred1 = ground_truth[:]
        pred2 = ground_truth[:]
        pred3 = ground_truth[:]
    else:
        # --- 生成“两者一致，一者不同”的样本 ---
        # 我们随机选一个预测器让它犯错
        pred1 = ground_truth[:]
        pred2 = ground_truth[:]
        pred3 = introduce_noise(ground_truth, SINGLE_BIT_ERROR_RATE)
        
        # 为了让错误随机分布在三个预测器中，我们再随机打乱它们
        predictions = [pred1, pred2, pred3]
        random.shuffle(predictions)
        pred1, pred2, pred3 = predictions[0], predictions[1], predictions[2]

    # --- 构建输入和输出 ---
    input_vector = pred1 + pred2 + pred3
    
    # 输出标签是“硬投票”的结果。
    # 在我们这个生成逻辑下，硬投票的结果永远等于 ground_truth
    # 但我们仍然调用 hard_vote 函数，以保证逻辑的普适性
    output_vector = hard_vote([pred1, pred2, pred3])
    
    return {
        "input": input_vector,
        "output": output_vector
    }

def main():
    print("\n--- 开始生成神经投票器数据集 ---")
    
    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            record = sample_one()
            f.write(json.dumps(record) + '\n')
            
    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条样本已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample = sample_one()
    print(json.dumps(sample, indent=2))
    print("-" * 80)
    print(f"输入向量长度: {len(sample['input'])} (预期: {INPUT_BITS})")
    print(f"输出向量长度: {len(sample['output'])} (预期: {OUTPUT_BITS})")
    print("-" * 80)

if __name__ == "__main__":
    main()
