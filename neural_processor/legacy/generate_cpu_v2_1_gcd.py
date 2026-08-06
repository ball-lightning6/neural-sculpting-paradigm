# generate_neural_alu_gcd.py

import json
import random
from tqdm import tqdm
import os

# ==============================================================================
# --- 1. “球状闪电v2.1-GCD” 指令集架构 (ISA) 定义 ---
# ==============================================================================

# --- 机器状态定义 ---
NUM_REGISTERS = 4
BITS_PER_REGISTER = 8
NUM_FLAGS = 2  # 1 bit for Zero Flag (ZF), 1 bit for Greater-than Flag (GF)
STATE_BITS = NUM_REGISTERS * BITS_PER_REGISTER + NUM_FLAGS # 4 * 8 + 2 = 34 bits

# --- 指令格式 (16 bits, for simplicity) ---
# [ 4b OpCode | 2b Dst | 2b Src1 | 2b Src2 | 6b Unused/Imm_low ]
# 立即数将使用 Src1和Src2字段组合，或直接使用一个字段
INSTRUCTION_BITS = 16

# --- OpCode 定义 (只包含计算指令) ---
OPCODES = {
    # Key: Name, Value: OpCode
    "MOVI": 0b0001,
    "MOV":  0b0010,
    "SUB":  0b0011,
    "CMP":  0b0100,
}

# --- 数据集参数 ---
DATASET_SIZE = 20000000
OUTPUT_FILE = "neural_alu_gcd_dataset_val.jsonl"

# ==============================================================================
# --- 2. 核心逻辑：Python模拟的“完美ALU” ---
# ==============================================================================

class PerfectALU:
    
    def parse_instruction(self, instr_int):
        opcode = (instr_int >> 12) & 0xF
        dst    = (instr_int >> 10) & 0x3
        src1   = (instr_int >> 8)  & 0x3
        src2   = (instr_int >> 6)  & 0x3
        imm    = instr_int & 0xFF  # Allow full 8-bit immediate for MOVI
        return opcode, dst, src1, src2, imm

    def state_to_vector(self, regs, zf, gf):
        vec = [zf, gf]
        for r_val in regs:
            vec.extend([int(b) for b in format(r_val, f'0{BITS_PER_REGISTER}b')])
        return vec

    def execute(self, instruction_int, initial_regs, initial_zf, initial_gf):
        regs = initial_regs[:]
        zf, gf = initial_zf, initial_gf
        
        opcode, dst, src1, src2, imm = self.parse_instruction(instruction_int)
        
        if opcode == OPCODES["MOVI"]:
            regs[dst] = imm
        elif opcode == OPCODES["MOV"]:
            regs[dst] = regs[src1]
        elif opcode == OPCODES["SUB"]:
            result = (regs[src1] - regs[src2]) & 0xFF
            regs[dst] = result
        elif opcode == OPCODES["CMP"]:
            zf = 1 if regs[src1] == regs[src2] else 0
            gf = 1 if regs[src1] > regs[src2] else 0
            # CMP不修改通用寄存器
        
        return regs, zf, gf

# ==============================================================================
# --- 3. 单个样本处理与主生成函数 ---
# ==============================================================================

def sample_one(cpu):
    # 1. 随机选择一个要测试的指令
    op_name = random.choice(list(OPCODES.keys()))
    opcode = OPCODES[op_name]

    # 2. 随机生成操作数
    dst_reg = random.randint(0, NUM_REGISTERS - 1)
    src1_reg = random.randint(0, NUM_REGISTERS - 1)
    src2_reg = random.randint(0, NUM_REGISTERS - 1)
    immediate = random.randint(0, 2**BITS_PER_REGISTER - 1)
    
    # 3. 构建指令整数
    instruction_int = (opcode << 12) | (dst_reg << 10) | (src1_reg << 8) | (src2_reg << 6) | immediate

    # 4. 随机生成一个初始状态
    initial_zf = random.randint(0, 1)
    initial_gf = random.randint(0, 1)
    initial_regs = [random.randint(0, 255) for _ in range(NUM_REGISTERS)]
    
    # 5. 用“完美ALU”计算出正确的下一状态
    final_regs, final_zf, final_gf = cpu.execute(instruction_int, initial_regs, initial_zf, initial_gf)
    
    # 6. 编码成最终的输入输出格式
    instruction_str = format(instruction_int, f'0{INSTRUCTION_BITS}b')
    initial_state_vector = cpu.state_to_vector(initial_regs, initial_zf, initial_gf)
    final_state_vector = cpu.state_to_vector(final_regs, final_zf, final_gf)

    input_vector_str = instruction_str + "".join(map(str, initial_state_vector))
    
    return {
        "input": input_vector_str,
        "output": final_state_vector
    }

def main():
    cpu = PerfectALU()
    
    print("\n--- 开始生成神经ALU (GCD专用) 数据集 ---")
    print("=" * 80)
    print(f"状态空间: {NUM_REGISTERS}x{BITS_PER_REGISTER}b regs, {NUM_FLAGS}b flags (共 {STATE_BITS} bits)")
    print(f"指令集 (仅计算): MOVI, MOV, SUB, CMP")
    print(f"训练格式: (指令 + 前一状态) -> (下一状态)")
    print(f"输入总位数: {INSTRUCTION_BITS + STATE_BITS}")
    print(f"输出总位数: {STATE_BITS}")
    print("=" * 80)
    
    with open(OUTPUT_FILE, 'w') as f:
        # 使用set确保指令和状态的多样性，防止模式过于单一
        seen_samples = set()
        with tqdm(total=DATASET_SIZE, desc="生成样本") as pbar:
            while len(seen_samples) < DATASET_SIZE:
                record = sample_one(cpu)
                # 只用输入做key，因为输入决定输出
                if record['input'] not in seen_samples:
                    f.write(json.dumps(record) + '\n')
                    seen_samples.add(record['input'])
                    pbar.update(1)

    print(f"\n✅ 数据集生成完成！共 {len(seen_samples)} 条不重复样本已保存至 '{OUTPUT_FILE}'")

    print("\n--- 样本数据结构验证 ---")
    sample = sample_one(cpu)
    print(json.dumps(sample, indent=2))
    print("-" * 80)
    print(f"输入向量长度: {len(sample['input'])} (预期: {INSTRUCTION_BITS + STATE_BITS})")
    print(f"输出向量长度: {len(sample['output'])} (预期: {STATE_BITS})")
    print("-" * 80)

if __name__ == "__main__":
    main()
