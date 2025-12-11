# generate_neural_alu_v3_pi_calculator.py

import json
import random
from tqdm import tqdm
import os

# ==============================================================================
# --- 1. “球状闪电v3.1-Pi-Calculator” ISA 定义 ---
# ==============================================================================

# --- 机器状态定义 (16位极简版) ---
NUM_REGISTERS = 2
BITS_PER_REGISTER = 16
NUM_FLAGS = 3  # ZF, GF, CF (进位/借位标志)
STATE_BITS = NUM_REGISTERS * BITS_PER_REGISTER + NUM_FLAGS # 2 * 16 + 3 = 35 bits

# --- 指令格式 (16 bits) ---
# [ 4b OpCode | 1b Dst | 1b Src1 | 1b Src2 | 9b Unused ]
INSTRUCTION_BITS = 16

# --- OpCode 定义 (多精度计算核心指令集) ---
OPCODES = {
    "MOV":  0b0001, # 寄存器间移动
    "ADD":  0b0010, # 16位加法 (设置CF)
    "ADC":  0b0011, # Add with Carry (带进位加法)
    "SUB":  0b0100, # 16位减法 (设置CF/Borrow)
    "SBC":  0b0101, # Subtract with Borrow (带借位减法)
    "CMP":  0b0110, # 16位比较 (设置ZF, GF)
}
# 我们暂时移除了MOVI，因为可以用MOV+另一个寄存器预置来实现，简化ALU的学习任务

# --- 数据集参数 ---
DATASET_SIZE = 5000000
OUTPUT_FILE = "neural_alu_v3_pi_calculator_dataset.jsonl"
INPUT_BITS = INSTRUCTION_BITS + STATE_BITS
OUTPUT_BITS = STATE_BITS

# ==============================================================================
# --- 2. 核心逻辑：Python模拟的“完美16位多精度ALU” ---
# ==============================================================================

class PerfectPiALU:
    
    def parse_instruction(self, instr_int):
        opcode = (instr_int >> 12) & 0xF
        p1 = (instr_int >> 11) & 0x1 
        p2 = (instr_int >> 10) & 0x1
        p3 = (instr_int >> 9) & 0x1
        return opcode, p1, p2, p3

    def state_to_vector(self, regs, zf, gf, cf):
        vec = [zf, gf, cf]
        for r_val in regs:
            vec.extend([int(b) for b in format(r_val, f'0{BITS_PER_REGISTER}b')])
        return vec

    def execute(self, instruction_int, initial_regs, initial_zf, initial_gf, initial_cf):
        regs, zf, gf, cf = initial_regs[:], initial_zf, initial_gf, initial_cf
        opcode, p1, p2, p3 = self.parse_instruction(instruction_int)
        
        # 统一操作数地址
        dst, src1, src2 = p1, p2, p3
        
        # 为了代码清晰，我们将寄存器别名
        val1 = regs[src1]
        val2 = regs[src2]
        
        # --- 执行指令 ---
        if opcode == OPCODES["MOV"]:    # MOV Dst, Src1
            regs[dst] = val1
        elif opcode == OPCODES["ADD"]:  # ADD Dst, Src1, Src2
            res = val1 + val2
            regs[dst] = res & 0xFFFF # 取低16位
            cf = 1 if res > 0xFFFF else 0
        elif opcode == OPCODES["ADC"]:  # ADC Dst, Src1, Src2
            res = val1 + val2 + cf
            regs[dst] = res & 0xFFFF
            cf = 1 if res > 0xFFFF else 0
        elif opcode == OPCODES["SUB"]:  # SUB Dst, Src1, Src2
            res = val1 - val2
            regs[dst] = res & 0xFFFF
            cf = 1 if val1 < val2 else 0
        elif opcode == OPCODES["SBC"]:  # SBC Dst, Src1, Src2
            res = val1 - val2 - cf
            regs[dst] = res & 0xFFFF
            cf = 1 if (val1 - cf) < val2 else 0
        elif opcode == OPCODES["CMP"]:  # CMP Src1, Src2
            zf = 1 if val1 == val2 else 0
            gf = 1 if val1 > val2 else 0
        
        return regs, zf, gf, cf

# ==============================================================================
# --- 3. 样本生成 ---
# ==============================================================================

def sample_one(cpu):
    op_name = random.choice(list(OPCODES.keys()))
    opcode = OPCODES[op_name]

    p1 = random.randint(0, NUM_REGISTERS - 1)
    p2 = random.randint(0, NUM_REGISTERS - 1)
    p3 = random.randint(0, NUM_REGISTERS - 1)
    
    instruction_int = (opcode << 12) | (p1 << 11) | (p2 << 10) | (p3 << 9)

    initial_zf, initial_gf, initial_cf = random.randint(0,1), random.randint(0,1), random.randint(0,1)
    initial_regs = [random.randint(0, 2**16 - 1) for _ in range(NUM_REGISTERS)]
    
    final_regs, final_zf, final_gf, final_cf = cpu.execute(instruction_int, initial_regs, initial_zf, initial_gf, initial_cf)
    
    instruction_str = format(instruction_int, f'0{INSTRUCTION_BITS}b')
    initial_state_vector = cpu.state_to_vector(initial_regs, initial_zf, initial_gf, initial_cf)
    final_state_vector = cpu.state_to_vector(final_regs, final_zf, final_gf, final_cf)

    input_vector_str = instruction_str + "".join(map(str, initial_state_vector))
    
    return { "input": input_vector_str, "output": final_state_vector }

def main():
    cpu = PerfectPiALU()
    print("\n--- 开始生成神经高精度计算ALU (Pi专用) 数据集 ---")
    
    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(DATASET_SIZE), desc="生成样本"):
            f.write(json.dumps(sample_one(cpu)) + '\n')
            
    print(f"\n✅ 数据集生成完成！已保存至 '{OUTPUT_FILE}'")
    print("\n--- 样本数据结构验证 ---")
    sample = sample_one(cpu)
    print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
