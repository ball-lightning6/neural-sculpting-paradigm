import random
from tqdm import tqdm
import itertools

# ================= 规则定义 (请确保和生成脚本一致) =================
SEQ_LEN = 30 
MOD = 2**SEQ_LEN

def int_to_bits(n, length): return [int(x) for x in f"{n:0{length}b}"]
def bits_to_int(bits): return int("".join(map(str, bits)), 2)
def cyclic_shift(bits, k): k = k % len(bits); return bits[k:] + bits[:k]
def get_neighbors(bits, i): return bits[i-1], bits[i], bits[(i+1)%len(bits)]

def apply_ca(bits, rule_number):
    rule_map = int_to_bits(rule_number, 8)
    return [rule_map[7 - (get_neighbors(bits, i)[0]*4 + get_neighbors(bits, i)[1]*2 + get_neighbors(bits, i)[2])] for i in range(len(bits))]

def apply_add_shift(bits):
    val = (bits_to_int(bits) + bits_to_int(cyclic_shift(bits, 15))) % MOD
    return int_to_bits(val, SEQ_LEN)

def apply_long_or(bits):
    shifted = cyclic_shift(bits, 10)
    return [b | s for b, s in zip(bits, shifted)]

def apply_dynamic_shift(bits):
    k = bits_to_int(bits[:5])
    return cyclic_shift(bits, k)

def apply_majority(bits):
    new_bits = []
    for i in range(len(bits)):
        l, c, r = get_neighbors(bits, i)
        if (l + c + r) >= 2:
            new_bits.append(1)
        else:
            new_bits.append(0)
    return new_bits

def apply_not(bits):
    return [1 - x for x in bits]

def apply_shift_part_reverse(bits):
    return bits[1::3]+[1 - x for x in bits[2::3]]+bits[0::3][::-1]

# 这里的规则 2 已经改成了 Rule 167
RULES_FUNC = {
    0: lambda b: apply_ca(b, 30),
    1: lambda b: apply_ca(b, 110),
    2: lambda b: apply_ca(b, 167), # 新规则
    3: lambda b: apply_ca(b, 184),
    4: apply_majority,
    5: apply_add_shift,
    6: apply_dynamic_shift,
    7: apply_shift_part_reverse
}

# ================= 核心逻辑 =================

class RuleDeduplicator:
    def __init__(self, num_layers=2, num_probes=1000):
        self.num_layers = num_layers
        self.num_probes = num_probes
        self.test_inputs = self._generate_probes()
        
    def _generate_probes(self):
        """生成一组固定的随机输入作为指纹提取器"""
        print(f"Generating {self.num_probes} probe inputs...")
        return [[random.randint(0, 1) for _ in range(SEQ_LEN)] for _ in range(self.num_probes)]
    
    def get_fingerprint(self, rule_seq):
        """计算规则序列的功能指纹"""
        fingerprint = []
        # 为了加速，其实不需要存所有 bits，存 hash 或者 int 值也可以
        # 但为了绝对安全，我们还是存转换后的 tuple
        for x in self.test_inputs:
            current = x[:]
            for r_idx in rule_seq:
                current = RULES_FUNC[r_idx](current)
            fingerprint.extend(current)
        return tuple(fingerprint)

    def run(self):
        """运行去重逻辑"""
        print(f"Checking equivalence for {self.num_layers} layers ({8**self.num_layers} combinations)...")
        
        # 1. 生成所有可能的规则组合
        all_combinations = itertools.product(range(8), repeat=self.num_layers)
        
        fingerprint_map = {} # Fingerprint -> [Rule Seq 1, Rule Seq 2...]
        total_combinations = 8**self.num_layers
        
        # 2. 遍历并计算指纹
        for rule_seq in tqdm(all_combinations, total=total_combinations):
            fp = self.get_fingerprint(rule_seq)
            if fp not in fingerprint_map:
                fingerprint_map[fp] = []
            fingerprint_map[fp].append(rule_seq)
            
        # 3. 统计结果
        unique_behaviors = len(fingerprint_map)
        print(f"\n=== Result Analysis ===")
        print(f"Total Combinations: {total_combinations}")
        print(f"Unique Behaviors: {unique_behaviors}")
        print(f"Redundancy Rate: {100 * (1 - unique_behaviors/total_combinations):.2f}%")
        
        # 4. 提取Canonical Rules (只取每组的第一个)
        canonical_rules = []
        ambiguous_cases = 0
        
        print("\n--- Ambiguity Examples ---")
        for fp, rules in fingerprint_map.items():
            canonical_rules.append(rules[0]) # 取字典序最小的那个
            if len(rules) > 1:
                ambiguous_cases += 1
                if ambiguous_cases <= 50: # 只打印前5个例子
                    print(f"Equivalence Group: {rules}")
        
        print(f"\nFound {ambiguous_cases} groups with equivalent rules.")
        
        return canonical_rules

# ================= 导出工具 =================
def export_safe_rules(canonical_rules, filename="safe_rules.json"):
    import json
    # 将 tuple 转为 list 方便 JSON 序列化
    rules_list = [list(r) for r in canonical_rules]
    with open(filename, 'w') as f:
        json.dump(rules_list, f)
    print(f"Saved {len(rules_list)} unique rule sequences to {filename}")

if __name__ == "__main__":
    # 你可以在这里修改层数，比如 2 或 3
    # 注意：层数多了穷举会很慢 (8^5 是 32768，跑得动；8^6 就有点慢了)
    deduplicator = RuleDeduplicator(num_layers=4) 
    
    safe_rules = deduplicator.run()
    
    # 导出这份"安全"的规则列表
    # 你在生成训练数据时，应该直接从这个列表里 random.choice()
    export_safe_rules(safe_rules, "safe_rule_combinations_L2.json")
