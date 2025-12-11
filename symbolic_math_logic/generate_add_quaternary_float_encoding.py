import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 可在此处调整参数
class TaskConfig:
    NUM_BITS_PER_ADDEND = 20
    DATASET_SIZE = 500000

class EncodingConfig:
    ENCODING_MODE = 'quaternary_float' # 'binary', 'quaternary_float'
    QUATERNARY_MAP = {
        '00': 0.0,
        '01': 5.5,
        '10': 1.3,
        '11': 10.0,
    }

# --- 自动计算的参数 ---
# 输入维度是符号数
BITS_PER_SYMBOL_IN = 2 if 'quaternary' in EncodingConfig.ENCODING_MODE else 1
INPUT_DIM = TaskConfig.NUM_BITS_PER_ADDEND * 2 // BITS_PER_SYMBOL_IN

# 输出维度是标准的二进制位数
OUTPUT_DIM = TaskConfig.NUM_BITS_PER_ADDEND + 1

# --- 文件名 ---
OUTPUT_FILE = f"add_{TaskConfig.NUM_BITS_PER_ADDEND}bit_{EncodingConfig.ENCODING_MODE}_dataset.jsonl"


# ==============================================================================
# --- 2. 核心逻辑: 编码器 (只用于输入) ---
# ==============================================================================
class Encoder:
    def __init__(self, config):
        self.mode = config.ENCODING_MODE
        self.map = config.QUATERNARY_MAP
        self.bits_per_symbol = 2 if 'quaternary' in self.mode else 1

    def encode_input(self, binary_string):
        """将二进制字符串编码为输入的浮点数列表"""
        if self.mode == 'binary':
            return [float(bit) for bit in binary_string]
        
        if 'quaternary' in self.mode:
            # 确保长度是2的倍数，如果不是，前面补0 (虽然对于固定位数加法通常是偶数)
            if len(binary_string) % 2 != 0:
                binary_string = '0' + binary_string
            
            symbols = []
            for i in range(0, len(binary_string), 2):
                two_bits = binary_string[i:i+2]
                symbols.append(self.map[two_bits])
            return symbols
        
        raise ValueError(f"未知的编码模式: {self.mode}")


# ==============================================================================
# --- 3. 样本生成 ---
# ==============================================================================
def generate_sample(num_bits, encoder):
    """
    生成一个 (任意符号输入, 标准二进制输出) 的数据对。
    """
    # 1. 生成两个n位二进制数
    max_val = 2**num_bits - 1
    num1_int = random.randint(0, max_val)
    num2_int = random.randint(0, max_val)

    # 2. 计算和
    sum_int = num1_int + num2_int

    # 3. 获取标准的二进制字符串表示
    num1_str = format(num1_int, f'0{num_bits}b')
    num2_str = format(num2_int, f'0{num_bits}b')
    
    # 4. 输入使用指定的编码器进行编码
    input_str_combined = num1_str + num2_str
    input_list = encoder.encode_input(input_str_combined)
    
    # 5. 输出保持标准的二进制形式 (0/1整数列表)
    output_bits_len = num_bits + 1
    output_str = format(sum_int, f'0{output_bits_len}b')
    output_list = [int(bit) for bit in output_str]

    return {
        "input": input_list,
        "output": output_list
    }


# ==============================================================================
# --- 主函数 ---
# ==============================================================================
def main():
    task_cfg = TaskConfig()
    encoding_cfg = EncodingConfig()
    encoder = Encoder(encoding_cfg)

    # --- 打印脚本信息 ---
    print("=" * 70)
    print(f"抽象加法 ({encoding_cfg.ENCODING_MODE} -> binary) - 数据集生成器")
    print("=" * 70)
    print(f"原始加数位数: {task_cfg.NUM_BITS_PER_ADDEND}")
    print(f"输入维度 (符号数): {INPUT_DIM}")
    print(f"输出维度 (二进制位数): {OUTPUT_DIM}")
    if 'quaternary' in encoding_cfg.ENCODING_MODE:
        print(f"输入符号映射: {encoding_cfg.QUATERNARY_MAP}")
    print(f"数据集大小: {task_cfg.DATASET_SIZE:,}")
    print("=" * 70)
    
    print(f"\n--- 开始生成 {task_cfg.DATASET_SIZE:,} 条样本 ---")
    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(task_cfg.DATASET_SIZE), desc="生成样本"):
            sample = generate_sample(task_cfg.NUM_BITS_PER_ADDEND, encoder)
            f.write(json.dumps(sample) + '\n')
    print(f"\n✅ 数据集生成完成！已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample = generate_sample(task_cfg.NUM_BITS_PER_ADDEND, encoder)
    # print(json.dumps(sample, indent=2)) 
    print(f"Sample Input: {sample['input'][:5]} ...")
    print("-" * 70)
    print(f"输入向量长度: {len(sample['input'])} (预期: {INPUT_DIM})")
    print(f"输出向量长度: {len(sample['output'])} (预期: {OUTPUT_DIM})")
    print("-" * 70)


if __name__ == "__main__":
    main()
