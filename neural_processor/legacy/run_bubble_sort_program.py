# run_bubble_sort_on_neural_cpu.py

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
# --- 2. 神经CPU模拟器，现在加载“冒泡排序”程序 ---
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
        
        assembly_program = [
            # Pass 1
            {"op": "CMP", "src1": 0, "src2": 1},
            {"op": "JG", "target": "SWAP_0_1_P1"},
            "CONTINUE_1_2_P1:",
            {"op": "CMP", "src1": 1, "src2": 2},
            {"op": "JG", "target": "SWAP_1_2_P1"},
            "END_PASS_1:",
            # Pass 2
            {"op": "CMP", "src1": 0, "src2": 1},
            {"op": "JG", "target": "SWAP_0_1_P2"},
            "END_PASS_2:",
            {"op": "HALT"},

            # --- 子程序 ---
            "SWAP_0_1_P1:",
            {"op": "MOV", "dst": 3, "src1": 0},
            {"op": "MOV", "dst": 0, "src1": 1},
            {"op": "MOV", "dst": 1, "src1": 3},
            {"op": "JMP", "target": "CONTINUE_1_2_P1"},

            "SWAP_1_2_P1:",
            {"op": "MOV", "dst": 3, "src1": 1},
            {"op": "MOV", "dst": 1, "src1": 2},
            {"op": "MOV", "dst": 2, "src1": 3},
            {"op": "JMP", "target": "END_PASS_1"},

            "SWAP_0_1_P2:",
            {"op": "MOV", "dst": 3, "src1": 0},
            {"op": "MOV", "dst": 0, "src1": 1},
            {"op": "MOV", "dst": 1, "src1": 3},
            {"op": "JMP", "target": "END_PASS_2"},
        ]

        # --- 简单的汇编器 ---
        labels = {}
        program = []
        for line in assembly_program:
            if isinstance(line, str) and line.endswith(":"):
                labels[line[:-1]] = len(program)
            else:
                program.append(line)
        
        for instr in program:
            if "target" in instr:
                instr["target"] = labels[instr["target"]]
        
        self.final_sort_program = program
        self.machine_code = [self.assemble(instr) for instr in self.final_sort_program]

        self.total_steps_executed = 0
        self.correct_steps_executed = 0
    
    def assemble(self,i): op_name=i.get("op","NOP");op=OPCODES.get(op_name,0);return(op<<12)|(i.get("dst",0)<<10)|(i.get("src1",0)<<8)|(i.get("src2",0)<<6)|i.get("imm",0)
    def state_to_vector(self,r,z,g): v=[z,g]; [v.extend(list(map(int,f'{val:08b}'))) for val in r]; return torch.tensor(v,dtype=torch.float32).to(self.device)
    def vector_to_state(self,v): z,g=int(v[0].item()),int(v[1].item()); r=[]; [r.append(int("".join(map(str,v[2+i*8:10+i*8].int().tolist())),2)) for i in range(4)]; return r,z,g

    def run_program(self, initial_regs):
        regs, zf, gf, pc = initial_regs[:], 0, 0, 0
        for _ in range(100): # 排序最多几十步
            if pc >= len(self.final_sort_program): break
            instr = self.final_sort_program[pc]
            op_name = instr["op"]

            if op_name in ["JMP", "JZ", "JG", "HALT"]:
                if op_name == "HALT": break
                if op_name == "JMP": pc = instr["target"]
                elif op_name == "JZ" and zf == 1: pc = instr["target"]
                elif op_name == "JG" and gf == 1: pc = instr["target"]
                else: pc += 1
            else: # 计算指令
                self.total_steps_executed += 1
                machine_code = self.machine_code[pc]
                with torch.no_grad():
                    instr_tensor = torch.tensor([int(b) for b in format(machine_code,f'0{INSTRUCTION_BITS}b')],dtype=torch.float32).to(self.device)
                    state_tensor = self.state_to_vector(regs, zf, gf)
                    input_vector = torch.cat((instr_tensor, state_tensor)).unsqueeze(0)
                    output_vector = self.model(input_vector).squeeze(0)
                    preds_vector = (torch.sigmoid(output_vector) > 0.5)
                    
                    true_regs, true_zf, true_gf = self.perfect_alu.execute(machine_code, regs, zf, gf)
                    true_vector = self.state_to_vector(true_regs, true_zf, true_gf).int()

                    if not torch.all(preds_vector.int() == true_vector):
                        # 如果单步出错，整个程序就算失败了
                        return [-1, -1, -1] # 返回错误代码
                    
                    self.correct_steps_executed += 1
                    regs, zf, gf = self.vector_to_state(preds_vector)
                pc += 1
        return regs[:3] # 返回排序后的R0, R1, R2

# ==============================================================================
# --- 3. 最终验证流程 ---
# ==============================================================================
def final_validation(num_tests=10000):
    print("\n--- 开始端到端'冒泡排序'程序执行验证 ---")
    config = Config()
    # Note: Requires a trained model. Using likely path.
    emulator = Verifiable_NeuralCPU_v2("best_model_mlp_cpu2.pth", config)
    
    program_correct_count = 0
    with tqdm(total=num_tests, desc="测试排序程序") as pbar:
        for i in range(num_tests):
            # 生成3个不重复的随机数
            data_to_sort = random.sample(range(256), 3)
            ground_truth = sorted(data_to_sort)
            
            initial_regs = [0] * NUM_REGISTERS
            initial_regs[0], initial_regs[1], initial_regs[2] = data_to_sort[0], data_to_sort[1], data_to_sort[2]

            neural_result = emulator.run_program(initial_regs)
            
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
    print(f"  - 测试程序: 3个8位整数冒泡排序")
    print(f"  - 总测试程序数: {num_tests:,}")
    print(f"  - 总执行计算指令数: {emulator.total_steps_executed:,}")
    print("---")
    print(f"  - 单步执行准确率 (Per-Step Accuracy): {step_accuracy:.12f}%")
    print(f"  - 完整程序执行准确率 (End-to-End Accuracy): {program_accuracy:.6f}%")
    print("=================================================")

if __name__ == '__main__':
    final_validation()
