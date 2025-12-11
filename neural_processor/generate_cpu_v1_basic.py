# generate_neural_cpu_v1.py

import json
import random
from tqdm import tqdm
import os

# ==============================================================================
# --- 1. “球状闪电v1”指令集架构 (ISA) 定义 ---
# ==============================================================================

# --- 机器状态定义 ---
NUM_REGISTERS = 4
BITS_PER_REGISTER = 8
STATE_BITS = NUM_REGISTERS * BITS_PER_REGISTER  # 4 * 8 = 32 bits

# --- 指令格式定义 (16 bits) ---
# [ 4 bit OpCode | 2 bit DstReg | 2 bit SrcReg | 8 bit Immediate ]
# (为了简化，我们将SrcReg2的位置用作SrcReg1)
OPCODE_BITS = 4
REG_BITS = 2
IMMEDIATE_BITS = 8
INSTRUCTION_BITS = OPCODE_BITS + REG_BITS + REG_BITS + IMMEDIATE_BITS # 4+2+2+8 = 16 bits

# --- OpCode 定义 ---
OPCODES = {
    "NOP":  0b0000,
    "MOVI": 0b0001,
    "ADD":  0b0010,
    "XOR":  0b0011,
}

# --- 数据集参数 ---
DATASET_SIZE = 2000000
OUTPUT_FILE = "neural_cpu_v1_dataset.jsonl"

# ==============================================================================
# --- 2. 核心逻辑：Python模拟的“完美CPU” ---
# ==============================================================================

class PerfectCPU:
    """一个用Python实现的、绝对正确的CPU模拟器，用于生成真值(ground truth)"""
    
    def parse_instruction(self, instruction_int):
        """从16位整数中解析出指令的各个部分"""
        opcode = (instruction_int >> (REG_BITS * 2 + IMMEDIATE_BITS)) & 0b1111
        dst_reg_idx = (instruction_int >> (REG_BITS + IMMEDIATE_BITS)) & 0b11
        src_reg_idx = (instruction_int >> IMMEDIATE_BITS) & 0b11
        immediate = instruction_int & 0xFF # 取后8位
        return opcode, dst_reg_idx, src_reg_idx, immediate

    def execute(self, instruction_int, initial_state_list):
        """
        执行单条指令。
        Args:
            instruction_int (int): 16位的指令整数。
            initial_state_list (list): 32位的当前寄存器状态列表 [R0, R1, R2, R3]。
        Returns:
            list: 32位的更新后寄存器状态列表。
        """
        opcode, dst, src, imm = self.parse_instruction(instruction_int)
        
        # 复制一份当前状态，避免修改原始列表
        new_state = initial_state_list[:]

        if opcode == OPCODES["NOP"]:
            # 无操作，状态不变
            pass
        
        elif opcode == OPCODES["MOVI"]:
            # MOVI Dst, Imm
            new_state[dst] = imm
        
        elif opcode == OPCODES["ADD"]:
            # ADD Dst, Src, Imm
            src_val = initial_state_list[src]
            result = (src_val + imm) & 0xFF # 8位加法，自动处理溢出
            new_state[dst] = result

        elif opcode == OPCODES["XOR"]:
            # XOR Dst, Src, Imm
            src_val = initial_state_list[src]
            result = src_val ^ imm
            new_state[dst] = result
        
        return new_state

# ==============================================================================
# --- 3. 单个样本处理与主生成函数 ---
# ==============================================================================

def sample_one(cpu):
    """生成一个完整的 (输入, 输出) 数据对。"""
    
    # 1. 随机生成一个指令
    instruction_int = random.randint(0, 2**INSTRUCTION_BITS - 1)
    instruction_str = format(instruction_int, f'0{INSTRUCTION_BITS}b')

    # 2. 随机生成一个初始状态
    initial_state_regs = [random.randint(0, 2**BITS_PER_REGISTER - 1) for _ in range(NUM_REGISTERS)]
    
    # 3. 用“完美CPU”计算出正确的下一状态
    final_state_regs = cpu.execute(instruction_int, initial_state_regs)

    # 4. 编码成最终的输入输出格式
    initial_state_str = "".join([format(r, f'0{BITS_PER_REGISTER}b') for r in initial_state_regs])
    final_state_str = "".join([format(r, f'0{BITS_PER_REGISTER}b') for r in final_state_regs])

    input_vector = instruction_str + initial_state_str
    output_vector = [int(bit) for bit in final_state_str]
    
    return {
        "input": input_vector,
        "output": output_vector
    }

def main():
    cpu = PerfectCPU()
    
    print("\n--- 开始生成神经CPU v1数据集 ---")
    print("=" * 80)
    print(f"状态空间: {NUM_REGISTERS}个 {BITS_PER_REGISTER}-bit 寄存器 (共 {STATE_BITS} bits)")
    print(f"指令集: NOP, MOVI, ADD, XOR ({INSTRUCTION_BITS} bits)")
    print(f"训练格式: (指令 + 前一状态) -> (当前状态)")
    print(f"输入总位数: {INSTRUCTION_BITS + STATE_BITS} bits")
    print(f"输出总位数: {STATE_BITS} bits")
    print("=" * 80)
    
    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            record = sample_one(cpu)
            f.write(json.dumps(record) + '\n')
            
    print(f"\n✅ 数据集生成完成！共 {DATASET_SIZE} 条样本已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample = sample_one(cpu)
    print(json.dumps(sample, indent=2))
    print("-" * 80)
    print(f"输入向量长度: {len(sample['input'])} (预期: {INSTRUCTION_BITS + STATE_BITS})")
    print(f"输出向量长度: {len(sample['output'])} (预期: {STATE_BITS})")
    print("-" * 80)

if __name__ == "__main__":
    main()
