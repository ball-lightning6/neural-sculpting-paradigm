# generate_neural_turing_machine.py
import json
import random
from tqdm import tqdm

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
class Config:
    NUM_SAMPLES = 2000_000
    TAPE_LENGTH = 32
    NUM_INSTRUCTIONS = 8
    BITS_PER_INSTRUCTION = 4

    # --- 【关键】: 选择写入模式 ---
    # 'overwrite' 或 'xor'
    WRITE_MODE = 'xor' # <--- 在这里切换实验模式！

    # --- 自动计算的参数 ---
    INPUT_DIM = NUM_INSTRUCTIONS * BITS_PER_INSTRUCTION
    OUTPUT_DIM = TAPE_LENGTH

    # --- 文件名 ---
    OUTPUT_FILE = f"autodl-tmp/tape_writer_{WRITE_MODE}_w{TAPE_LENGTH}.jsonl"

# ==============================================================================
# --- 2. 核心逻辑: "神经图灵机"模拟器 ---
# ==============================================================================
def run_tape_writer(input_str: str, cfg: Config):
    """
    模拟一个在循环磁带上进行条件写入的计算过程。
    """
    # 1. 初始化
    tape = [0] * cfg.TAPE_LENGTH
    pointer = 0

    # 2. 主循环：处理8个指令
    for i in range(cfg.NUM_INSTRUCTIONS):
        # a. 解码指令
        chunk = input_str[i * cfg.BITS_PER_INSTRUCTION : (i + 1) * cfg.BITS_PER_INSTRUCTION]

        n_binary = chunk[:3]
        a_binary = chunk[3]

        n = int(n_binary, 2) + 1
        a = int(a_binary)

        # c. 执行写入子循环
        for _ in range(n):
            # i. 写入
            if cfg.WRITE_MODE == 'overwrite':
                tape[pointer] = a
            elif cfg.WRITE_MODE == 'xor':
                tape[pointer] = tape[pointer] ^ a

            # ii. & iii. 更新指针并处理循环边界
            pointer = (pointer + 1) % cfg.TAPE_LENGTH

    return tape

# ==============================================================================
# --- 3. 样本生成函数 ---
# ==============================================================================
def generate_sample(cfg):
    """生成一个样本。"""

    # 随机生成32位的输入指令字符串
    input_str = "".join(random.choice("01") for _ in range(cfg.INPUT_DIM))

    # 计算最终的磁带状态作为标签
    output_list = run_tape_writer(input_str, cfg)

    return {
        "input": input_str,
        "output": output_list
    }

# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def main():
    cfg = Config()

    print("=" * 70)
    print(f""神经图灵机"模拟 - {cfg.WRITE_MODE} 模式 - 数据集生成器")
    print("=" * 70)
    print(f"输入维度: {cfg.INPUT_DIM} (8组4-bit指令)")
    print(f"输出维度: {cfg.OUTPUT_DIM} (最终磁带状态)")
    print(f"数据集大小: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)

    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="生成样本"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ 数据集生成完成！已保存至 '{cfg.OUTPUT_FILE}'")

    # 验证一个例子
    print("\n--- 样本逻辑验证 ---")
    # 指令1: '001' (1) -> n=2, a=1. 写入 '11' 到 tape[0], tape[1]. pointer=2.
    # 指令2: '111' (7) -> n=8, a=0. 写入8个'0'. tape[2]到tape[9]变为0. pointer=10.
    # ...
    test_input = '0011' + '1110' + '0' * 24
    expected_output = run_tape_writer(test_input, cfg)
    print(f"测试输入: {test_input}")
    print(f"预期输出 ({cfg.WRITE_MODE}): {''.join(map(str, expected_output))}")

if __name__ == "__main__":
    main()