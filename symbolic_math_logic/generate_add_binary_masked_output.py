import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 500_000   # 鉴于任务难度，建议使用较大的数据集
    BITS_PER_ADDEND = 20      # 每个加数的位数
    
    # --- Masking 配置 ---
    MASK_VISIBLE_BITS = 15   # Mask中'1'的数量，可以调节
    
    # --- 自动计算的参数 ---
    INPUT_DIM_ADDENDS = BITS_PER_ADDEND * 2
    OUTPUT_DIM_FULL = BITS_PER_ADDEND + 1
    INPUT_DIM_MASK = OUTPUT_DIM_FULL
    
    INPUT_DIM_TOTAL = INPUT_DIM_ADDENDS + INPUT_DIM_MASK
    OUTPUT_DIM_MASKED = MASK_VISIBLE_BITS
    
    # --- 文件名 ---
    OUTPUT_FILE = f"add_{BITS_PER_ADDEND}bit_masked_output_{MASK_VISIBLE_BITS}.jsonl"


# ==============================================================================
# --- 2. 核心逻辑与样本生成 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个 (A+B+Mask, 部分Sum) 的数据对。"""
    
    # 1. 随机生成两个n位二进制数 (A, B)
    max_val = 2**cfg.BITS_PER_ADDEND - 1
    num1_int = random.randint(0, max_val)
    num2_int = random.randint(0, max_val)

    # 2. 计算完整的和 (Sum)
    sum_int = num1_int + num2_int

    # 3. 将 A, B, Sum 转换为标准二进制字符串
    num1_str = format(num1_int, f'0{cfg.BITS_PER_ADDEND}b')
    num2_str = format(num2_int, f'0{cfg.BITS_PER_ADDEND}b')
    sum_str = format(sum_int, f'0{cfg.OUTPUT_DIM_FULL}b')
    
    # 4. 生成随机的Mask (M)
    # Mask的长度应与完整的Sum长度一致
    mask_list = [1] * cfg.MASK_VISIBLE_BITS + [0] * (cfg.OUTPUT_DIM_FULL - cfg.MASK_VISIBLE_BITS)
    random.shuffle(mask_list)
    mask_str = "".join(map(str, mask_list))

    # 5. 组合成最终的输入
    # 输入格式: 20位A + 20位B + 21位Mask M
    input_str = num1_str + num2_str + mask_str
    
    # 6. 应用Mask来获取部分输出
    masked_output_list = []
    for i in range(cfg.OUTPUT_DIM_FULL):
        if mask_list[i] == 1:
            masked_output_list.append(int(sum_str[i]))
            
    assert len(masked_output_list) == cfg.MASK_VISIBLE_BITS

    return {
        "input": input_str,
        "output": masked_output_list
    }


def main():
    cfg = Config()
    
    print("=" * 70)
    print(f"部分可观测二进制加法 - 数据集生成器")
    print("=" * 70)
    print(f"任务: {cfg.BITS_PER_ADDEND}bit + {cfg.BITS_PER_ADDEND}bit 加法")
    print(f"输入格式: {cfg.INPUT_DIM_ADDENDS} bit (A+B) + {cfg.INPUT_DIM_MASK} bit (Mask)")
    print(f"总输入维度: {cfg.INPUT_DIM_TOTAL}")
    print(f"输出格式: {cfg.OUTPUT_DIM_MASKED} bit (被Mask选择的部分Sum)")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)
    
    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
