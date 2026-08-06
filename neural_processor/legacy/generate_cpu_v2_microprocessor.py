# generate_neural_microprocessor_v2.py

import json
import random
from tqdm import tqdm
import os

# ==============================================================================
# --- 1. “球状闪电v2-RainWater” 指令集架构 (ISA) 定义 ---
# ==============================================================================

# --- 机器状态定义 ---
NUM_REGISTERS = 8
# Note: Original notebook said BITS_PER_REGISTER=4 but code utilized 8-bit values (0-255).
# Using 8 bits here to be consistent with the logic and standard byte size.
BITS_PER_REGISTER = 8 
NUM_FLAGS = 1  # 1 bit for Less-than Flag (LF)
MEM_SIZE = 16  # 16字节内存，足够放下10个柱子的高度和一些变量

STATE_BITS = NUM_REGISTERS * BITS_PER_REGISTER + NUM_FLAGS 
FULL_STATE_BITS = STATE_BITS + MEM_SIZE * BITS_PER_REGISTER 

# --- 指令格式 (24 bits) ---
# [ 4b OpCode | 3b Dst | 3b Src1 | 3b Src2 | 8b Imm/Addr ]
INSTRUCTION_BITS = 24
INPUT_BITS = INSTRUCTION_BITS + FULL_STATE_BITS
OUTPUT_BITS = FULL_STATE_BITS

# --- OpCode 定义 (增加了STORE) ---
OPCODES = {
    "NOP": 0, "MOVI": 1, "LOAD": 2, "STORE": 3, 
    "ADD": 4, "SUB": 5, "CMP": 6, "JMP": 7, 
    "JLT": 8, "PRINT": 9
    # JMP/JLT/PRINT 是特殊指令，它们的输出格式不同
}

# --- 数据集参数 ---
DATASET_SIZE = 4000000
OUTPUT_FILE = "neural_microprocessor_v2_dataset.jsonl"

# ==============================================================================
# --- 2. 核心逻辑：Python模拟的“完美微处理器” ---
# ==============================================================================

class PerfectMicroprocessor:
    
    def parse_instruction(self, instr_int):
        opcode = (instr_int >> 20) & 0xF
        dst    = (instr_int >> 17) & 0x7
        src1   = (instr_int >> 14) & 0x7
        src2   = (instr_int >> 11) & 0x7
        imm_or_addr = instr_int & 0xFF
        return opcode, dst, src1, src2, imm_or_addr

    def state_to_vector(self, regs, flag, mem):
        """将机器状态编码为一个二进制列表"""
        vec = [flag]
        for r_val in regs:
            vec.extend([int(b) for b in format(r_val, f'0{BITS_PER_REGISTER}b')])
        for m_val in mem:
            vec.extend([int(b) for b in format(m_val, f'0{BITS_PER_REGISTER}b')])
        return vec

    def execute(self, instruction_int, initial_regs, initial_flag, initial_mem):
        """执行单条指令，返回更新后的 (regs, flag, mem)"""
        
        # 复制状态以避免副作用
        regs = initial_regs[:]
        flag = initial_flag
        mem = initial_mem[:]
        
        opcode, dst, src1, src2, imm_or_addr = self.parse_instruction(instruction_int)
        
        # JMP/JLT/PRINT 是特殊指令，它们不改变状态，但我们会在这里忽略它们的特殊输出
        # 因为训练数据只关心状态转移
        if opcode == OPCODES["NOP"]:
            pass
        elif opcode == OPCODES["MOVI"]:
            regs[dst] = imm_or_addr
        elif opcode == OPCODES["LOAD"]:
            # LOAD Dst, [Src1] (Src1寄存器里的值作为地址)
            addr = regs[src1]
            if 0 <= addr < MEM_SIZE:
                regs[dst] = mem[addr]
        elif opcode == OPCODES["STORE"]:
            # STORE [Src1], Dst (Dst寄存器里的值存入Src1寄存器值指定的地址)
            addr = regs[src1]
            if 0 <= addr < MEM_SIZE:
                mem[addr] = regs[dst]
        elif opcode == OPCODES["ADD"]:
            regs[dst] = (regs[src1] + regs[src2]) & 0xFF
        elif opcode == OPCODES["SUB"]:
            regs[dst] = (regs[src1] - regs[src2]) & 0xFF
        elif opcode == OPCODES["CMP"]:
            flag = 1 if regs[src1] < regs[src2] else 0
        
        # JMP, JLT, PRINT 指令不改变寄存器、标志位或内存状态
        
        return regs, flag, mem

# ==============================================================================
# --- 3. 单个样本处理与主生成函数 ---
# ==============================================================================

def sample_one(cpu):
    # 1. 随机生成一个指令
    instruction_int = random.randint(0, 2**INSTRUCTION_BITS - 1)
    
    # 2. 随机生成一个初始状态
    initial_flag = random.randint(0, 1)
    initial_regs = [random.randint(0, 255) for _ in range(NUM_REGISTERS)]
    initial_mem = [random.randint(0, 255) for _ in range(MEM_SIZE)]

    # 3. 用“完美CPU”计算出正确的下一状态
    final_regs, final_flag, final_mem = cpu.execute(instruction_int, initial_regs, initial_flag, initial_mem)
    
    # 4. 编码成最终的输入输出格式
    instruction_str = format(instruction_int, f'0{INSTRUCTION_BITS}b')
    initial_state_vector = cpu.state_to_vector(initial_regs, initial_flag, initial_mem)
    final_state_vector = cpu.state_to_vector(final_regs, final_flag, final_mem)

    input_vector_str = instruction_str + "".join(map(str, initial_state_vector))
    
    return {
        "input": input_vector_str,
        "output": final_state_vector
    }

def main():
    cpu = PerfectMicroprocessor()
    
    print("\n--- 开始生成神经微处理器 v2 数据集 ---")
    print("=" * 80)
    print(f"状态空间: {NUM_REGISTERS}x{BITS_PER_REGISTER}b regs, {NUM_FLAGS}b flags, {MEM_SIZE} bytes mem (总共 {FULL_STATE_BITS} bits)")
    print(f"指令集: NOP, MOVI, LOAD, STORE, ADD, SUB, CMP, JMP, JLT, PRINT")
    print(f"训练格式: (指令 + 前一完整状态) -> (下一完整状态)")
    print(f"输入总位数: {INPUT_BITS}")
    print(f"输出总位数: {OUTPUT_BITS}")
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
    print(f"输入向量长度: {len(sample['input'])} (预期: {INPUT_BITS})")
    print(f"输出向量长度: {len(sample['output'])} (预期: {OUTPUT_BITS})")
    print("-" * 80)

if __name__ == "__main__":
    main()
