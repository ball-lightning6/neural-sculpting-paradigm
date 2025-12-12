# generate_ca_rule110_minimalist_trace.py

import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
NUM_BITS = 30       # 元胞自动机的宽度
TOTAL_LAYERS = 2    # 总演化层数
DATASET_SIZE = 500000

# --- 文件名 ---
OUTPUT_FILE = f'ca_rule110_n{NUM_BITS}_l{TOTAL_LAYERS}_minimalist_trace.jsonl'

# ==============================================================================
# --- 2. 脚本信息打印 ---
# ==============================================================================
print("=" * 70)
print(f"元胞自动机 Rule 110 - '极简解释'数据集生成器")
print("=" * 70)
print(f"输入宽度: {NUM_BITS}")
print(f"总演化层数: {TOTAL_LAYERS}")
print(f"输出格式: 包含每层 '邻域>结果' 轨迹和该层最终状态")
print("=" * 70)

# ==============================================================================
# --- 3. 核心逻辑：模拟器与"极简解释"生成器 ---
# ==============================================================================
def apply_rule110(left, center, right):
    """应用Rule 110规则"""
    pattern = (left << 2) | (center << 1) | right
    rule110_map = {7: 0, 6: 1, 5: 1, 4: 0, 3: 1, 2: 1, 1: 1, 0: 0}
    return rule110_map.get(pattern, 0)

def generate_ca_minimalist_trace(initial_state, total_layers):
    """
    模拟元胞自动机的演化，并生成极简的、包含计算过程的文本。
    """
    n = len(initial_state)
    current_state = list(initial_state)
    explanation_parts = []

    for _ in range(total_layers):
        next_state = [0] * n
        layer_trace_parts = []
        
        # 逐比特计算并生成 '邻域>结果' 的解释
        for i in range(n):
            left = current_state[(i - 1 + n) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            
            result_bit = apply_rule110(left, center, right)
            next_state[i] = result_bit
            
            # 生成当前比特的解释字符串，例如 "101>1"
            neighbor_str = f"{left}{center}{right}"
            layer_trace_parts.append(f"{neighbor_str}>{result_bit}")
        
        # 将当前层的计算轨迹用空格连接
        layer_trace = " ".join(layer_trace_parts)
        
        # 得到当前层的演化结果
        current_state = next_state
        current_state_str = "".join(map(str, current_state))
        
        # 将轨迹和结果组合，用换行符分隔
        explanation_parts.append(f"{layer_trace}\n{current_state_str}")

    # 将所有层的解释用换行符拼接
    full_explanation = "\n".join(explanation_parts)
    
    return full_explanation

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets():
    print("\n--- 开始生成极简解释数据集 ---")

    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            initial_state = [random.randint(0, 1) for _ in range(NUM_BITS)]
            
            # 生成完整的、极简的解释文本
            explanation_text = generate_ca_minimalist_trace(initial_state, TOTAL_LAYERS)
            
            # 构建 "prompt -> answer" 格式
            prompt = "".join(map(str, initial_state))
            # 在prompt和answer之间加一个换行符，让模型更容易学习分界
            record_text = f"Evolve Rule 110:\n{prompt} -> \n{explanation_text}"

            f.write(json.dumps({"text": record_text}) + '\n')
    
    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")
    
    # --- 验证样本格式和长度 ---
    print("\n--- 样本数据结构验证 ---")
    sample_state = [random.randint(0, 1) for _ in range(NUM_BITS)]
    prompt = "".join(map(str, sample_state))
    sample_explanation = generate_ca_minimalist_trace(sample_state, TOTAL_LAYERS)
    sample_text = f"Evolve Rule 110:\n{prompt} -> \n{sample_explanation}"
    
    print("样本示例:")
    print(json.dumps({"text": sample_text}, indent=2))
    print("-" * 70)
    print(f"注意: 估算的单个样本总长度可能很长。请确保您的模型 CONTEXT_LENGTH 足够大。")
    print(f"例如，对于{NUM_BITS}位{TOTAL_LAYERS}层，总字符数约为 (5*{NUM_BITS}+1)*{TOTAL_LAYERS} + {NUM_BITS} + 20")

# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__ == "__main__":
    generate_datasets()
