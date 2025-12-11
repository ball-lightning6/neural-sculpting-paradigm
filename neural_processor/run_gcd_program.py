# run_gcd_program.py

import torch
import torch.nn as nn
import json
import random
from tqdm import tqdm
import time
import os

# ==============================================================================
# --- 1. “球状闪电v2.2” 架构定义 (与训练时100%同步) ---
# ==============================================================================
class Config:
    INPUT_BITS = 50; OUTPUT_BITS = 34; HIDDEN_SIZE = 4096; NUM_HIDDEN_LAYERS = 4

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [ nn.Linear(config.INPUT_BITS, config.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(config.HIDDEN_SIZE), nn.Dropout(0.1) ]
        for _ in range(config.NUM_HIDDEN_LAYERS):
            layers.extend([ nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE), nn.GELU(), nn.LayerNorm(config.HIDDEN_SIZE), nn.Dropout(0.1) ])
        layers.append(nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_BITS))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

NUM_REGISTERS, BITS_PER_REGISTER, NUM_FLAGS = 4, 8, 2
STATE_BITS, INSTRUCTION_BITS = 34, 16
OPCODES = {
    "NOP": 0, "MOVI": 1, "MOV": 2, "ADD": 3, "SUB": 4, "INC": 5, "DEC": 6, 
    "AND": 7, "OR": 8, "XOR": 9, "NOT": 10, "SHL": 11, "CMP": 12, "HALT": 15
}

class PerfectALU: # 真值引擎
    def parse_instruction(self, i): return ((i>>12)&0xF, (i>>10)&0x3, (i>>8)&0x3, (i>>6)&0x3, i&0xFF)
    def execute(self, i, r, z, g):
        regs, zf, gf = r[:], z, g
        op, p1, p2, p3, imm = self.parse_instruction(i)
        dst, src1, src2 = p1, p2, p3
        if op in [OPCODES["NOP"], OPCODES["HALT"]]: pass
        elif op == OPCODES["MOVI"]: regs[dst] = imm
        elif op == OPCODES["MOV"]: regs[dst] = regs[src1]
        elif op == OPCODES["ADD"]: regs[dst] = (regs[src1] + regs[src2]) & 0xFF
        elif op == OPCODES["SUB"]: regs[dst] = (regs[src1] - regs[src2]) & 0xFF
        elif op == OPCODES["INC"]: regs[dst] = (regs[dst] + 1) & 0xFF
        elif op == OPCODES["DEC"]: regs[dst] = (regs[dst] - 1) & 0xFF
        elif op == OPCODES["AND"]: regs[dst] = regs[src1] & regs[src2]
        elif op == OPCODES["OR"]: regs[dst] = regs[src1] | regs[src2]
        elif op == OPCODES["XOR"]: regs[dst] = regs[src1] ^ regs[src2]
        elif op == OPCODES["NOT"]: regs[dst] = ~regs[src1] & 0xFF
        elif op == OPCODES["SHL"]: regs[dst] = (regs[src1] << (imm & 0x7)) & 0xFF
        elif op == OPCODES["CMP"]:
            zf = 1 if regs[src1] == regs[src2] else 0
            gf = 1 if regs[src1] > regs[src2] else 0
        return regs, zf, gf

# ==============================================================================
# --- 2. 神经CPU模拟器，加载GCD程序 ---
# ==============================================================================
class Verifiable_NeuralCPU_v2:
    def __init__(self, model_path, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLP(config).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model file {model_path} not found. Please train the model first.")
        self.model.eval()
        self.perfect_alu = PerfectALU()
        
        self.gcd_program = [ {"op": "CMP", "src1": 0, "src2": 1}, 
                            {"op": "JZ", "target": 7}, 
                            {"op": "JG", "target": 5}, 
                            {"op": "SUB", "dst": 1, "src1": 1, "src2": 0}, 
                            {"op": "JMP", "target": 0}, 
                            {"op": "SUB", "dst": 0, "src1": 0, "src2": 1}, 
                            {"op": "JMP", "target": 0}, 
                            {"op": "NOP"} ]
        self.gcd_machine_code = [self.assemble(instr) for instr in self.gcd_program]

        self.total_steps_executed = 0
        self.correct_steps_executed = 0
    
    def assemble(self, i):
        if i["op"]=="NOP": 
            return 0
        return (OPCODES.get(i["op"],0)<<12)|(i.get("dst",0)<<10)|(i.get("src1",0)<<8)|(i.get("src2",0)<<6)|i.get("imm",0)
    
    def state_to_vector(self, r, z, g):
        v = [z,g]; [v.extend(list(map(int,f'{val:08b}'))) for val in r]; return torch.tensor(v,dtype=torch.float32).to(self.device)

    def vector_to_state(self, vec):
        z,g=int(vec[0].item()),int(vec[1].item()); r=[]; [r.append(int("".join(map(str,vec[2+i*8:10+i*8].int().tolist())),2)) for i in range(4)]; return r,z,g
    
    def run_program(self, initial_regs, max_steps=10000):
        regs, zf, gf, pc = initial_regs[:], 0, 0, 0
        
        for _ in range(max_steps):
            if pc >= len(self.gcd_program): break
            instr = self.gcd_program[pc]
            op_name = instr["op"]

            if op_name in ["JMP", "JZ", "JG"]:
                if op_name=="JMP" or (op_name=="JZ" and zf==1) or (op_name=="JG" and gf==1):
                    pc = instr["target"]
                else: 
                    pc += 1
            elif op_name == "NOP": 
                break
            else:
                self.total_steps_executed += 1
                machine_code = self.gcd_machine_code[pc]
                
                with torch.no_grad():
                    # --- 神经网络预测 ---
                    instr_tensor = torch.tensor([int(b) for b in format(machine_code,f'0{INSTRUCTION_BITS}b')],dtype=torch.float32).to(self.device)
                    state_tensor = self.state_to_vector(regs, zf, gf)
                    input_vector = torch.cat((instr_tensor, state_tensor)).unsqueeze(0)
                    output_vector = self.model(input_vector).squeeze(0)
                    preds_vector = (torch.sigmoid(output_vector) > 0.5)
                        
                    # --- 真值计算 ---
                    true_regs, true_zf, true_gf = self.perfect_alu.execute(machine_code, regs, zf, gf)
                    true_vector = self.state_to_vector(true_regs, true_zf, true_gf).int()
                     
                    # --- 单步对比 ---
                    if torch.all(preds_vector.int() == true_vector):
                        self.correct_steps_executed += 1
                    
                    # 无论对错，都使用神经网络的结果继续执行
                    regs, zf, gf = self.vector_to_state(preds_vector)
                pc += 1
        return regs[0]

# ==============================================================================
# --- 3. 最终验证流程 ---
# ==============================================================================

def gcd_python(a,b):
    while b: a, b = b, a % b
    return a

def final_validation(num_tests=1):
    print("\n--- 开始进行端到端GCD程序执行与单步准确率验证 ---")
    
    config = Config()
    # Note: Requires a trained model. Path assumes default from training script.
    emulator = Verifiable_NeuralCPU_v2("best_model_mlp_cpu2.pth", config)
    
    program_correct_count = 0
    
    with tqdm(total=num_tests, desc="测试GCD程序") as pbar:
        for i in range(num_tests):
            a, b = random.randint(1, 255), random.randint(1, 255)
            initial_regs = [0]*NUM_REGISTERS; initial_regs[0]=a; initial_regs[1]=b
            
            neural_result = emulator.run_program(initial_regs)
            ground_truth = gcd_python(a, b)
            
            if neural_result == ground_truth:
                program_correct_count += 1
            
            pbar.update(1)
            pbar.set_postfix_str(f"程序准确率: {100*program_correct_count/(i+1):.4f}%")

    total_steps = emulator.total_steps_executed if emulator.total_steps_executed > 0 else 1
    program_accuracy = 100 * program_correct_count / num_tests
    step_accuracy = 100 * emulator.correct_steps_executed / total_steps
    
    print("\n=================================================")
    print("                最终验证结果报告")
    print("=================================================")
    print(f"  - 总测试程序数: {num_tests:,}")
    print(f"  - 总执行计算指令数: {emulator.total_steps_executed:,}")
    print("---")
    print(f"  - 单步执行准确率 (Per-Step Accuracy): {step_accuracy:.12f}%")
    print(f"  - 完整程序执行准确率 (End-to-End Accuracy): {program_accuracy:.6f}%")
    print("=================================================")

if __name__ == '__main__':
    final_validation()
