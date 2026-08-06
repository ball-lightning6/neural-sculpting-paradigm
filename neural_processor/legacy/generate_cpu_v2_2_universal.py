# generate_neural_alu_v2_universal.py

import json
import random
from tqdm import tqdm
import os

# ==============================================================================
# --- 1. “球状闪电v2.2-通用ALU” ISA 定义 ---
# ==============================================================================

# --- 机器状态定义 ---
NUM_REGISTERS = 4
BITS_PER_REGISTER = 8
NUM_FLAGS = 2  # ZF, GF
STATE_BITS = NUM_REGISTERS * BITS_PER_REGISTER + NUM_FLAGS # 34 bits

# --- 指令格式 (16 bits) ---
# [ 4b OpCode | 2b Dst/Src1 | 2b Src1/Src2 | 8b Imm ] 
# 字段的含义会根据指令类型变化
INSTRUCTION_BITS = 16
INPUT_BITS = INSTRUCTION_BITS + STATE_BITS
OUTPUT_BITS = STATE_BITS

# --- OpCode 定义 (扩展版) ---
OPCODES = {
    "NOP":  0b0000, "MOVI": 0b0001, "MOV":  0b0010, "ADD":  0b0011,
    "SUB":  0b0100, "INC":  0b0101, "DEC":  0b0110, "AND":  0b0111,
    "OR":   0b1000, "XOR":  0b1001, "NOT":  0b1010, "SHL":  0b1011,
    "CMP":  0b1100, "HALT": 0b1111
}

# --- 数据集参数 ---
DATASET_SIZE = 20000000 # 既然是通用ALU，我们可以生成更多的样本
OUTPUT_FILE = "neural_alu_v2_universal_dataset_val_20m.jsonl"

# ==============================================================================
# --- 2. 核心逻辑：Python模拟的“完美通用ALU” ---
# ==============================================================================

class PerfectUniversalALU:
    
    def parse_instruction(self, instr_int):
        opcode = (instr_int >> 12) & 0xF
        p1     = (instr_int >> 10) & 0x3 # Dst or Src1
        p2     = (instr_int >> 8)  & 0x3 # Src1 or Src2
        p3     = (instr_int >> 6)  & 0x3 # Src2
        imm    = instr_int & 0xFF
        return opcode, p1, p2, p3, imm

    def state_to_vector(self, regs, zf, gf):
        vec = [zf, gf]
        for r_val in regs:
            vec.extend([int(b) for b in format(r_val, f'0{BITS_PER_REGISTER}b')])
        return vec

    def execute(self, instruction_int, initial_regs, initial_zf, initial_gf):
        regs, zf, gf = initial_regs[:], initial_zf, initial_gf
        opcode, p1, p2, p3, imm = self.parse_instruction(instruction_int)
        
        # 为了代码清晰，我们用 dst 和 src 命名
        dst, src1, src2 = p1, p2, p3

        if opcode == OPCODES["NOP"] or opcode == OPCODES["HALT"]:
            pass
        elif opcode == OPCODES["MOVI"]: # MOVI Dst, Imm
            regs[dst] = imm
        elif opcode == OPCODES["MOV"]:  # MOV Dst, Src1
            regs[dst] = regs[src1]
        elif opcode == OPCODES["ADD"]:  # ADD Dst, Src1, Src2
            regs[dst] = (regs[src1] + regs[src2]) & 0xFF
        elif opcode == OPCODES["SUB"]:  # SUB Dst, Src1, Src2
            regs[dst] = (regs[src1] - regs[src2]) & 0xFF
        elif opcode == OPCODES["INC"]:  # INC Dst
            regs[dst] = (regs[dst] + 1) & 0xFF
        elif opcode == OPCODES["DEC"]:  # DEC Dst
            regs[dst] = (regs[dst] - 1) & 0xFF
        elif opcode == OPCODES["AND"]:  # AND Dst, Src1, Src2
            regs[dst] = regs[src1] & regs[src2]
        elif opcode == OPCODES["OR"]:   # OR Dst, Src1, Src2
            regs[dst] = regs[src1] | regs[src2]
        elif opcode == OPCODES["XOR"]:  # XOR Dst, Src1, Src2
            regs[dst] = regs[src1] ^ regs[src2]
        elif opcode == OPCODES["NOT"]:  # NOT Dst, Src1
            regs[dst] = ~regs[src1] & 0xFF # 按位取反并限制在8位
        elif opcode == OPCODES["SHL"]:  # SHL Dst, Src1, Imm(low 3 bits)
            shift_amount = imm & 0x7 # 只取立即数的低3位作为移位数 (0-7)
            regs[dst] = (regs[src1] << shift_amount) & 0xFF
        elif opcode == OPCODES["CMP"]:  # CMP Src1, Src2
            zf = 1 if regs[src1] == regs[src2] else 0
            gf = 1 if regs[src1] > regs[src2] else 0
        
        return regs, zf, gf

# ==============================================================================
# --- 3. 样本生成 ---
# ==============================================================================

def sample_one(cpu):
    op_name = random.choice(list(OPCODES.keys()))
    opcode = OPCODES[op_name]

    # 随机生成操作数
    p1 = random.randint(0, NUM_REGISTERS - 1)
    p2 = random.randint(0, NUM_REGISTERS - 1)
    p3 = random.randint(0, NUM_REGISTERS - 1)
    imm = random.randint(0, 255)
    
    # 构建指令
    instruction_int = (opcode << 12) | (p1 << 10) | (p2 << 8) | (p3 << 6) | imm

    # 随机生成初始状态
    initial_zf = random.randint(0, 1)
    initial_gf = random.randint(0, 1)
    initial_regs = [random.randint(0, 255) for _ in range(NUM_REGISTERS)]

    # 计算正确结果
    final_regs, final_zf, final_gf = cpu.execute(instruction_int, initial_regs, initial_zf, initial_gf)
    
    # 编码输入输出
    instruction_str = format(instruction_int, f'0{INSTRUCTION_BITS}b')
    initial_state_vector = cpu.state_to_vector(initial_regs, initial_zf, initial_gf)
    final_state_vector = cpu.state_to_vector(final_regs, final_zf, final_gf)

    input_vector_str = instruction_str + "".join(map(str, initial_state_vector))
    
    return { "input": input_vector_str, "output": final_state_vector }

def main():
    cpu = PerfectUniversalALU()
    
    print("\n--- 开始生成神经通用ALU v2.2 数据集 ---")
    
    with open(OUTPUT_FILE, 'w') as f:
        # 使用set确保指令和状态的多样性
        seen_inputs = set()
        with tqdm(total=DATASET_SIZE, desc="生成样本") as pbar:
            while len(seen_inputs) < DATASET_SIZE:
                record = sample_one(cpu)
                if record['input'] not in seen_inputs:
                    f.write(json.dumps(record) + '\n')
                    seen_inputs.add(record['input'])
                    pbar.update(1)

    print(f"\n✅ 数据集生成完成！共 {len(seen_inputs)} 条不重复样本已保存至 '{OUTPUT_FILE}'")
    
    print("\n--- 样本数据结构验证 ---")
    sample = sample_one(cpu)
    print(json.dumps(sample, indent=2))
    print("-" * 80)

if __name__ == "__main__":
    main()
