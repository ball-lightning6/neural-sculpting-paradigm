import json
import random
from tqdm import tqdm

# 核心参数配置
NUM_BITS = 30
TOTAL_LAYERS = 2
DATASET_SIZE = 500000

# 规则库：8种著名的基本元胞自动机规则
RULES_DATABASE = {
    30:  "00011110",  # Class 3: 混沌
    54:  "00110110",  # Class 4: 复杂
    60:  "01101100",  # Class 2: XOR
    90:  "01011010",  # Class 2: Sierpinski triangle
    110: "01101110",  # Class 4: 图灵完备
    126: "01111110",  # Class 2: 简单重复
    150: "10010110",  # Class 2: 分形
    184: "10111000"   # Class 2: 平移
}

# 配置要包含在训练集中的规则
RULES_TO_INCLUDE = [30, 90, 110, 184]

rules_str = '_'.join(map(str, sorted(RULES_TO_INCLUDE)))
OUTPUT_FILE = f'ca_multirule_n{NUM_BITS}_l{TOTAL_LAYERS}_rules_{rules_str}.jsonl'


def get_rule_map(rule_number):
    """将规则编号转换为查找表"""
    binary_representation = format(rule_number, '08b')
    return {i: int(bit) for i, bit in enumerate(reversed(binary_representation))}


def apply_rule(left, center, right, rule_map):
    """应用CA规则"""
    pattern = (left << 2) | (center << 1) | right
    return rule_map.get(pattern, 0)


def generate_ca_multirule_cot_trace(initial_state, total_layers, rule_number, rule_map):
    """生成CA演化的CoT解释文本"""
    n = len(initial_state)
    current_state = list(initial_state)
    
    explanation_parts = [f"Received instruction to use Rule {rule_number}."]
    
    for layer in range(1, total_layers + 1):
        explanation_parts.append(f"\nThinking about Layer {layer}:")
        
        next_state = [0] * n
        layer_trace_parts = []
        
        for i in range(n):
            left = current_state[(i - 1 + n) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            
            result_bit = apply_rule(left, center, right, rule_map)
            next_state[i] = result_bit
            
            neighbor_str = f"{left}{center}{right}"
            layer_trace_parts.append(f"{neighbor_str}>{result_bit}")
        
        layer_trace = " ".join(layer_trace_parts)
        current_state = next_state
        current_state_str = "".join(map(str, current_state))
        
        explanation_parts.append(f"\nTrace: {layer_trace}\nResult: {current_state_str}")
    
    explanation_parts.append("\nEvolution complete.")
    return "".join(explanation_parts)


def generate_datasets():
    """生成数据集"""
    print("=" * 70)
    print(f"元胞自动机 - 多规则CoT解释数据集生成器")
    print("=" * 70)
    print(f"输入宽度: {NUM_BITS}")
    print(f"总演化层数: {TOTAL_LAYERS}")
    print(f"本次训练包含的规则: {RULES_TO_INCLUDE}")
    print(f"数据集大小: {DATASET_SIZE}")
    print("=" * 70)
    
    rule_maps_to_use = {num: get_rule_map(num) for num in RULES_TO_INCLUDE}
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            rule_number = random.choice(RULES_TO_INCLUDE)
            rule_map = rule_maps_to_use[rule_number]
            
            initial_state = [random.randint(0, 1) for _ in range(NUM_BITS)]
            initial_str = "".join(map(str, initial_state))
            
            explanation_text = generate_ca_multirule_cot_trace(
                initial_state, TOTAL_LAYERS, rule_number, rule_map
            )
            
            prompt = f"Rule: {rule_number}, State: {initial_str}"
            record_text = f"{prompt} -> \n{explanation_text}"
            
            f.write(json.dumps({"text": record_text}, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条数据已保存至 '{OUTPUT_FILE}'")
    
    # 输出样本
    print("\n--- 样本示例 ---")
    sample_rule = random.choice(RULES_TO_INCLUDE)
    sample_map = rule_maps_to_use[sample_rule]
    sample_state = [random.randint(0, 1) for _ in range(NUM_BITS)]
    sample_initial_str = "".join(map(str, sample_state))
    sample_explanation = generate_ca_multirule_cot_trace(
        sample_state, TOTAL_LAYERS, sample_rule, sample_map
    )
    sample_prompt = f"Rule: {sample_rule}, State: {sample_initial_str}"
    print(json.dumps(
        {"text": f"{sample_prompt} -> \n{sample_explanation}"}, 
        indent=2, 
        ensure_ascii=False
    ).replace('\\n', '\n'))


if __name__ == "__main__":
    generate_datasets()
