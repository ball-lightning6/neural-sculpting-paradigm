# debug_neural_cpu_gcd.py

import torch
import torch.nn as nn
import json
import random
from tqdm import tqdm
import logging
import time
import os

# ==============================================================================
# --- 1. 从训练/生成脚本复制过来的核心定义 ---
# ==============================================================================
# (保持与训练时完全一致)

class Config:
    INPUT_BITS = 50
    OUTPUT_BITS = 34
    HIDDEN_SIZE = 4096
    NUM_HIDDEN_LAYERS = 4
    DROPOUT_RATE = 0.1

class MLP(nn.Module):
    # ... (从你训练脚本中复制的、保证100%一致的MLP类) ...
    def __init__(self, config):
        super().__init__()
        layers = [
            nn.Linear(config.INPUT_BITS, config.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_SIZE),
            nn.Dropout(config.DROPOUT_RATE)
        ]
        for _ in range(config.NUM_HIDDEN_LAYERS):
            layers.extend([
                nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
                nn.GELU(),
                nn.LayerNorm(config.HIDDEN_SIZE),
                nn.Dropout(config.DROPOUT_RATE)
            ])
        layers.append(nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_BITS))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# --- 机器和指令集定义 ---
NUM_REGISTERS = 4
BITS_PER_REGISTER = 8
NUM_FLAGS = 2
STATE_BITS = 34
INSTRUCTION_BITS = 16
OPCODES = { "MOVI": 0b0001, "MOV": 0b0010, "SUB": 0b0011, "CMP": 0b0100 }

# --- “完美ALU”模拟器 (从生成脚本复制过来，作为我们的真值参考) ---
class PerfectALU:
    def parse_instruction(self, instr_int):
        opcode = (instr_int >> 12) & 0xF
        dst    = (instr_int >> 10) & 0x3
        src1   = (instr_int >> 8)  & 0x3
        src2   = (instr_int >> 6)  & 0x3
        imm    = instr_int & 0xFF
        return opcode, dst, src1, src2, imm

    def execute(self, instruction_int, initial_regs, initial_zf, initial_gf):
        regs, zf, gf = initial_regs[:], initial_zf, initial_gf
        opcode, dst, src1, src2, imm = self.parse_instruction(instruction_int)
        if opcode == OPCODES["MOVI"]: regs[dst] = imm
        elif opcode == OPCODES["MOV"]: regs[dst] = regs[src1]
        elif opcode == OPCODES["SUB"]: regs[dst] = (regs[src1] - regs[src2]) & 0xFF
        elif opcode == OPCODES["CMP"]:
            zf = 1 if regs[src1] == regs[src2] else 0
            gf = 1 if regs[src1] > regs[src2] else 0
        return regs, zf, gf

# ==============================================================================
# --- 2. 带有“单步调试”功能的神经CPU模拟器 ---
# ==============================================================================

class Debuggable_NeuralCPU_Emulator:
    def __init__(self, model_path, config):
        # ... (与之前相同) ...
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLP(config).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model file {model_path} not found. Please train the model first.")
        self.model.eval()
        self.perfect_alu = PerfectALU() # 内置一个“真值”引擎
        
        # ... (GCD程序和assemble函数保持不变) ...
        self.gcd_program = [
            # Addr 00: LOOP_START
            {"op": "CMP", "src1": 0, "src2": 1},      # 比较 R0 和 R1
            # Addr 01:
            {"op": "JZ", "target": 7},               # 如果相等(ZF=1), 跳转到 END_PROGRAM
            # Addr 02:
            {"op": "JG", "target": 5},               # 如果 R0 > R1 (GF=1), 跳转到 A_IS_GREATER
            
            # --- B_IS_GREATER 逻辑块 (b > a) ---
            # Addr 03:
            {"op": "SUB", "dst": 1, "src1": 1, "src2": 0}, # R1 = R1 - R0 (b = b - a)
            # Addr 04:
            {"op": "JMP", "target": 0},               # 无条件跳回 LOOP_START
            
            # --- A_IS_GREATER 逻辑块 (a > b) ---
            # Addr 05:
            {"op": "SUB", "dst": 0, "src1": 0, "src2": 1}, # R0 = R0 - R1 (a = a - b)
            # Addr 06:
            {"op": "JMP", "target": 0},               # 无条件跳回 LOOP_START
        
            # --- 结束 ---
            # Addr 07: END_PROGRAM
            {"op": "NOP"}
        ]
        self.gcd_machine_code = [self.assemble(instr) for instr in self.gcd_program]

    # ... (assemble, state_to_vector, vector_to_state 保持不变) ...
    def assemble(self, instr):
        if instr["op"] == "NOP": return 0
        opcode = OPCODES.get(instr["op"], 0)
        dst = instr.get("dst", 0)
        src1 = instr.get("src1", 0)
        src2 = instr.get("src2", 0)
        imm = instr.get("imm", 0)
        return (opcode << 12) | (dst << 10) | (src1 << 8) | (src2 << 6) | imm
    
    def state_to_vector(self, regs, zf, gf):
        vec = [zf, gf]
        for r_val in regs:
            vec.extend([int(b) for b in format(r_val, f'0{BITS_PER_REGISTER}b')])
        return torch.tensor(vec, dtype=torch.float32).to(self.device)

    def vector_to_state(self, vector):
        zf = int(vector[0].item())
        gf = int(vector[1].item())
        regs = []
        for i in range(NUM_REGISTERS):
            start = 2 + i * BITS_PER_REGISTER
            end = start + BITS_PER_REGISTER
            val = int("".join(map(str, vector[start:end].int().tolist())), 2)
            regs.append(val)
        return regs, zf, gf
        
    def run_program_with_debug(self, initial_regs, max_steps=1000):
        """执行GCD程序，并打印每一步的详细调试信息"""
        regs = initial_regs[:]
        zf, gf = 0, 0
        pc = 0
        
        print("\n" + "="*20 + f" 开始执行 GCD({regs[0]}, {regs[1]}) " + "="*20)

        for step in range(max_steps):
            if pc >= len(self.gcd_program): break
            
            instr = self.gcd_program[pc]
            opcode_name = instr["op"]

            print(f"\n--- 步骤 {step+1}: PC={pc:02d}, 指令: {opcode_name} ---")
            print(f"当前状态 (前): R=[{', '.join(map(str, regs))}], ZF={zf}, GF={gf}")

            if opcode_name in ["JMP", "JZ", "JG"]:
                # 控制流由Python处理
                print(">>> 控制流指令 (Python处理)")
                if opcode_name == "JMP":
                    pc = instr["target"]
                elif opcode_name == "JZ" and zf == 1:
                    pc = instr["target"]
                elif opcode_name == "JG" and gf == 1:
                    pc = instr["target"]
                else:
                    pc += 1
                print(f"跳转结果: 新 PC={pc:02d}")
            elif opcode_name == "NOP":
                print(">>> NOP: 停机")
                break
            else:
                # 计算由神经网络处理
                with torch.no_grad():
                    machine_code = self.gcd_machine_code[pc]
                    
                    # --- 神经网络预测 ---
                    instr_tensor = torch.tensor([int(b) for b in format(machine_code, f'0{INSTRUCTION_BITS}b')], dtype=torch.float32).to(self.device)
                    state_tensor = self.state_to_vector(regs, zf, gf)
                    input_vector = torch.cat((instr_tensor, state_tensor)).unsqueeze(0)
                    output_vector = self.model(input_vector).squeeze(0)
                    preds_vector = (torch.sigmoid(output_vector) > 0.5)
                    
                    # --- 真值计算 ---
                    true_final_regs, true_final_zf, true_final_gf = self.perfect_alu.execute(machine_code, regs, zf, gf)
                    true_vector = self.state_to_vector(true_final_regs, true_final_zf, true_final_gf)
                    true_vector = torch.tensor(true_vector).int()

                    # --- 对比 ---
                    is_correct = torch.all(preds_vector.int() == true_vector.to(self.device))
                    print(f">>> 计算指令 (神经网络处理)")
                    print(f"    - 神经网络输出: {self.vector_to_state(preds_vector)}")
                    print(f"    -   真值输出:   ({true_final_regs}, {true_final_zf}, {true_final_gf})")
                    if not is_correct:
                        print(f"    - 🔴🔴🔴 错误! 🔴🔴🔴")
                        # 找到第一个不匹配的比特位
                        diff = torch.ne(preds_vector.int(), true_vector.to(self.device))
                        first_error_idx = diff.nonzero(as_tuple=True)[0][0].item()
                        print(f"    - 第一个错误发生在比特位 {first_error_idx}")
                        # 如果出错了，程序无法继续，因为状态已经污染
                        return -1 # 返回一个错误代码
                    else:
                        print(f"    - ✅ 正确")

                    regs, zf, gf = self.vector_to_state(preds_vector) # 使用神经网络的结果继续
                pc += 1
        
        print("\n" + "="*20 + " 程序执行完毕 " + "="*20)
        return regs[0]

# ==============================================================================
# --- 主函数 ---
# ==============================================================================

if __name__ == '__main__':
    config = Config()
    
    # 我们只测试一个例子，来观察详细的执行过程
    a, b = 21, 14 # 一个简单的例子，gcd(21, 14) = 7
    
    print("--- 正在初始化模拟器 ---")
    # Note: Using consistent model name
    emulator = Debuggable_NeuralCPU_Emulator("best_model_mlp_cpu2.pth", config)
    
    initial_regs = [0] * NUM_REGISTERS
    initial_regs[0] = a
    initial_regs[1] = b
    
    neural_result = emulator.run_program_with_debug(initial_regs)
    def gcd_python(a, b):
        while b: a, b = b, a % b
        return a
        
    ground_truth = gcd_python(a, b) 

    print("\n=================================================")
    print("                最终结果对比")
    print("=================================================")
    print(f"  - 输入: gcd({a}, {b})")
    print(f"  - 神经CPU计算结果: {neural_result}")
    print(f"  - Python真值结果:   {ground_truth}")
    print(f"  - 程序是否正确执行: {'是' if neural_result == ground_truth else '否'}")
    print("=================================================")
