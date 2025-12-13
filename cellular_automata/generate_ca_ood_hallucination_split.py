import json
import random
from tqdm import tqdm

# 核心参数配置
CA_WIDTH = 30
EVOLUTION_STEPS = 2

# OOD分割配置
TRAIN_RULES_RATIO = 0.95  # 95%规则用于训练，5%用于OOD测试
SAMPLES_PER_RULE = 3000

# 输出文件
TRAIN_FILE = "ca_ood_train_dataset.jsonl"
OOD_EVAL_FILE = "ca_ood_eval_dataset.jsonl"


def apply_rule(state, rule_number, width):
    """执行CA演化一步"""
    next_state = [0] * width
    rule_bits = format(rule_number, '08b')
    for i in range(width):
        left = state[(i - 1 + width) % width]
        center = state[i]
        right = state[(i + 1) % width]
        pattern_index = (left << 2) | (center << 1) | right
        output_bit = int(rule_bits[7 - pattern_index])
        next_state[i] = output_bit
    return next_state


def generate_ca_instance(rule_number, ca_width, evolution_steps):
    """生成单个CA演化样本"""
    current_state = [random.randint(0, 1) for _ in range(ca_width)]
    initial_state_str = "".join(map(str, current_state))
    
    for _ in range(evolution_steps):
        current_state = apply_rule(current_state, rule_number, ca_width)
    
    final_state_str = "".join(map(str, current_state))
    rule_binary_str = format(rule_number, '08b')
    input_str = rule_binary_str + initial_state_str
    
    return {"input": input_str, "output": final_state_str}


def main():
    print("=" * 70)
    print("CA OOD/幻觉测试数据集生成器")
    print("=" * 70)
    print(f"CA宽度: {CA_WIDTH}")
    print(f"演化步数: {EVOLUTION_STEPS}")
    print(f"训练规则比例: {TRAIN_RULES_RATIO * 100:.0f}%")
    print(f"每规则样本数: {SAMPLES_PER_RULE}")
    print("=" * 70)
    
    # 分割规则集
    all_rules = list(range(256))
    random.seed(42)
    random.shuffle(all_rules)
    
    num_train_rules = int(len(all_rules) * TRAIN_RULES_RATIO)
    train_rules = all_rules[:num_train_rules]
    ood_rules = all_rules[num_train_rules:]
    
    print(f"\n总规则数: 256")
    print(f"训练集规则数: {len(train_rules)}")
    print(f"OOD测试集规则数: {len(ood_rules)}")
    
    # 生成训练数据集
    print("\n--- 生成训练数据集 ---")
    train_metadata = []
    for rule in tqdm(train_rules, desc="训练数据"):
        for _ in range(SAMPLES_PER_RULE):
            train_metadata.append(generate_ca_instance(rule, CA_WIDTH, EVOLUTION_STEPS))
    
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for record in train_metadata:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ 训练集: {len(train_metadata)} 样本 -> {TRAIN_FILE}")
    
    # 生成OOD评估数据集
    print("\n--- 生成OOD评估数据集 ---")
    ood_metadata = []
    for rule in tqdm(ood_rules, desc="OOD数据"):
        for _ in range(SAMPLES_PER_RULE // 5):
            ood_metadata.append(generate_ca_instance(rule, CA_WIDTH, EVOLUTION_STEPS))
    
    with open(OOD_EVAL_FILE, 'w', encoding='utf-8') as f:
        for record in ood_metadata:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ OOD测试集: {len(ood_metadata)} 样本 -> {OOD_EVAL_FILE}")
    
    print("\n" + "=" * 70)
    print("数据集生成完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
