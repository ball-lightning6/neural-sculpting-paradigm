"""让 Neural CPU v3 用机器码寻找第 10,000,000 个质数。

程序使用分段奇数筛，默认目标答案为 179,424,673。基础质数、``p^2``、
筛缓冲区地址、跨分段的下一个倍数、质数计数和停止条件都由 Neural CPU
的 16-bit 指令流计算。宿主只维护 PC、取指和外部 64-KiB RAM，并对神经核
做大 batch 逐操作审计。

所有运行参数都集中在文件顶部的 ``Config`` 中。快速自检时可将目标名次、
预期质数、搜索上限和执行模式分别改为 1000、7919、10000 和 ``"exact"``。
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 配置
# =============================================================================


def script_directory() -> Path:
    source_file = globals().get("__file__")
    if not source_file:
        return Path.cwd().resolve()
    source_dir = Path(source_file).resolve().parent
    return source_dir.parent if source_dir.name == "scripts" else source_dir


class Config:
    BASE_DIR = script_directory()
    MODEL_PATH = BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    RESULT_DIR = BASE_DIR / "results" / "primes"

    TARGET_RANK = 10_000_000
    EXPECTED_PRIME = 179_424_673
    SEARCH_LIMIT = 180_000_000
    EXECUTION_MODE = "batch"
    EXACT_ONLY = EXECUTION_MODE == "exact"
    USE_CUDA_GRAPH = True
    _MAX_OPS = 0
    MAX_NEURAL_OPERATIONS: int | None = _MAX_OPS if _MAX_OPS > 0 else None
    PROGRESS_EVERY_OPERATIONS = 50_000_000

    ALLOW_TF32 = False
    TRACE_BATCH = 262_144
    INFERENCE_BATCH = 262_144
    BATCH_RECHECK_MARGIN = 1e-4


# =============================================================================
# Neural CPU v3 ISA
# =============================================================================


NUM_REGISTERS = 4
REGISTER_BITS = 8
NUM_FLAGS = 3
INSTRUCTION_BITS = 16
STATE_BITS = NUM_FLAGS + NUM_REGISTERS * REGISTER_BITS
CONTROL_BITS = 4
INPUT_BITS = INSTRUCTION_BITS + STATE_BITS
OUTPUT_BITS = STATE_BITS + CONTROL_BITS

FLAG_ZF = 0
FLAG_GF = 1
FLAG_CF = 2

CTRL_MEMORY_READ = 0
CTRL_MEMORY_WRITE = 1
CTRL_BRANCH_TAKEN = 2
CTRL_HALT = 3

OPCODES = {
    "NOP": 0,
    "HALT": 1,
    "MOV": 2,
    "MOVI": 3,
    "LOAD": 4,
    "STORE": 5,
    "ADD": 6,
    "ADC": 7,
    "SUB": 8,
    "SBC": 9,
    "INC": 10,
    "DEC": 11,
    "AND": 12,
    "OR": 13,
    "XOR": 14,
    "NOT": 15,
    "SHL": 16,
    "SHR": 17,
    "CMP": 18,
    "ADDI": 19,
    "SUBI": 20,
    "CMPI": 21,
    "JMP": 22,
    "JZ": 23,
    "JNZ": 24,
    "JG": 25,
    "JL": 26,
    "JC": 27,
    "JNC": 28,
    "JMPR": 29,
    "RESERVED": 30,
    "TRAP": 31,
}
OPCODE_NAMES = {value: key for key, value in OPCODES.items()}

RRR_NAMES = {
    "MOV",
    "LOAD",
    "STORE",
    "ADD",
    "ADC",
    "SUB",
    "SBC",
    "INC",
    "DEC",
    "AND",
    "OR",
    "XOR",
    "NOT",
    "SHL",
    "SHR",
    "CMP",
}
RI8_NAMES = {"MOVI", "ADDI", "SUBI", "CMPI"}
BRANCH_NAMES = {"JMP", "JZ", "JNZ", "JG", "JL", "JC", "JNC"}


def output_bit_name(index: int) -> str:
    if index < NUM_FLAGS:
        return ("ZF", "GF", "CF")[index]
    if index < STATE_BITS:
        offset = index - NUM_FLAGS
        register = offset // REGISTER_BITS
        bit_from_msb = offset % REGISTER_BITS
        return f"R{register}.bit{REGISTER_BITS - 1 - bit_from_msb}"
    return (
        "memory_read",
        "memory_write",
        "branch_taken",
        "halt",
    )[index - STATE_BITS]


def instruction_fields(instruction: int) -> tuple[int, int, int, int, int, int]:
    opcode = (instruction >> 11) & 0x1F
    p1 = (instruction >> 9) & 0x03
    p2 = (instruction >> 7) & 0x03
    p3 = (instruction >> 5) & 0x03
    immediate = instruction & 0xFF
    raw_offset = instruction & 0x7FF
    offset = raw_offset - 0x800 if raw_offset & 0x400 else raw_offset
    return opcode, p1, p2, p3, immediate, offset


def scalar_reference(
    opcode: int,
    p1: int,
    p2: int,
    p3: int,
    immediate: int,
    registers: list[int],
    flags: list[int],
) -> tuple[list[int], list[int], list[int]]:
    regs = registers.copy()
    out_flags = flags.copy()
    controls = [0, 0, 0, 0]
    value1 = registers[p2]
    value2 = registers[p3]
    destination_value = registers[p1]
    carry = flags[FLAG_CF]

    def write(value: int) -> int:
        regs[p1] = value & 0xFF
        return regs[p1]

    if opcode == OPCODES["HALT"]:
        controls[CTRL_HALT] = 1
    elif opcode == OPCODES["MOV"]:
        write(value1)
    elif opcode == OPCODES["MOVI"]:
        write(immediate)
    elif opcode == OPCODES["LOAD"]:
        controls[CTRL_MEMORY_READ] = 1
    elif opcode == OPCODES["STORE"]:
        controls[CTRL_MEMORY_WRITE] = 1
    elif opcode in (OPCODES["ADD"], OPCODES["ADC"]):
        result = value1 + value2 + (carry if opcode == OPCODES["ADC"] else 0)
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
        out_flags[FLAG_CF] = int(result > 0xFF)
    elif opcode in (OPCODES["SUB"], OPCODES["SBC"]):
        borrow = carry if opcode == OPCODES["SBC"] else 0
        result = value1 - value2 - borrow
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
        out_flags[FLAG_CF] = int(value1 < value2 + borrow)
    elif opcode == OPCODES["INC"]:
        result = value1 + 1
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
        out_flags[FLAG_CF] = int(result > 0xFF)
    elif opcode == OPCODES["DEC"]:
        result = value1 - 1
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
        out_flags[FLAG_CF] = int(value1 == 0)
    elif opcode in (OPCODES["AND"], OPCODES["OR"], OPCODES["XOR"]):
        if opcode == OPCODES["AND"]:
            result = value1 & value2
        elif opcode == OPCODES["OR"]:
            result = value1 | value2
        else:
            result = value1 ^ value2
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
    elif opcode == OPCODES["NOT"]:
        written = write(~value1)
        out_flags[FLAG_ZF] = int(written == 0)
    elif opcode in (OPCODES["SHL"], OPCODES["SHR"]):
        shift = value2 & 7
        if opcode == OPCODES["SHL"]:
            result = value1 << shift
            new_carry = carry if shift == 0 else (value1 >> (8 - shift)) & 1
        else:
            result = value1 >> shift
            new_carry = carry if shift == 0 else (value1 >> (shift - 1)) & 1
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
        out_flags[FLAG_CF] = new_carry
    elif opcode == OPCODES["CMP"]:
        out_flags[FLAG_ZF] = int(value1 == value2)
        out_flags[FLAG_GF] = int(value1 > value2)
        out_flags[FLAG_CF] = int(value1 < value2)
    elif opcode in (OPCODES["ADDI"], OPCODES["SUBI"]):
        if opcode == OPCODES["ADDI"]:
            result = destination_value + immediate
            new_carry = int(result > 0xFF)
        else:
            result = destination_value - immediate
            new_carry = int(destination_value < immediate)
        written = write(result)
        out_flags[FLAG_ZF] = int(written == 0)
        out_flags[FLAG_CF] = new_carry
    elif opcode == OPCODES["CMPI"]:
        out_flags[FLAG_ZF] = int(destination_value == immediate)
        out_flags[FLAG_GF] = int(destination_value > immediate)
        out_flags[FLAG_CF] = int(destination_value < immediate)
    elif opcode == OPCODES["JMP"]:
        controls[CTRL_BRANCH_TAKEN] = 1
    elif opcode == OPCODES["JZ"]:
        controls[CTRL_BRANCH_TAKEN] = flags[FLAG_ZF]
    elif opcode == OPCODES["JNZ"]:
        controls[CTRL_BRANCH_TAKEN] = 1 - flags[FLAG_ZF]
    elif opcode == OPCODES["JG"]:
        controls[CTRL_BRANCH_TAKEN] = flags[FLAG_GF]
    elif opcode in (OPCODES["JL"], OPCODES["JC"]):
        controls[CTRL_BRANCH_TAKEN] = flags[FLAG_CF]
    elif opcode == OPCODES["JNC"]:
        controls[CTRL_BRANCH_TAKEN] = 1 - flags[FLAG_CF]
    elif opcode == OPCODES["JMPR"]:
        controls[CTRL_BRANCH_TAKEN] = 1
    elif opcode == OPCODES["TRAP"]:
        controls[CTRL_HALT] = 1

    return regs, out_flags, controls


def state_and_controls_to_bits(
    registers: list[int], flags: list[int], controls: list[int]
) -> list[int]:
    bits = flags.copy()
    for value in registers:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    bits.extend(controls)
    return bits


def decode_output_bits(bits: list[int]) -> dict[str, Any]:
    flags = bits[:NUM_FLAGS]
    registers = []
    for register in range(NUM_REGISTERS):
        start = NUM_FLAGS + register * REGISTER_BITS
        value = 0
        for bit in bits[start : start + REGISTER_BITS]:
            value = (value << 1) | int(bit)
        registers.append(value)
    controls = bits[STATE_BITS:]
    return {
        "flags": flags,
        "registers": registers,
        "controls": controls,
    }


# =============================================================================
# 网络与权重加载
# =============================================================================


class NeuralCPUCore(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(INPUT_BITS, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, OUTPUT_BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def find_model_path() -> Path:
    if Config.MODEL_PATH.exists():
        return Config.MODEL_PATH
    raise FileNotFoundError(
        f"找不到 Neural CPU v3 权重：{Config.MODEL_PATH}\n"
        "请修改 Config.MODEL_PATH。"
    )


class CoreRunner:
    def __init__(self, device: torch.device):
        self.device = device
        self.model: NeuralCPUCore | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.static_input: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.model_path: Path | None = None

        if Config.EXACT_ONLY:
            return

        model_path = find_model_path()
        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        first_weight = next(
            value
            for key, value in state_dict.items()
            if key.endswith("weight") and value.ndim == 2 and value.shape[1] == INPUT_BITS
        )
        linear_weights = [
            value
            for key, value in state_dict.items()
            if key.endswith("weight") and value.ndim == 2
        ]
        hidden_size = int(config.get("HIDDEN_SIZE", first_weight.shape[0]))
        hidden_layers = int(config.get("HIDDEN_LAYERS", len(linear_weights) - 2))
        dropout = float(config.get("DROPOUT", 0.0))

        self.model = NeuralCPUCore(hidden_size, hidden_layers, dropout).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.checkpoint = checkpoint if isinstance(checkpoint, dict) else None

        print(f"加载模型：{model_path}")
        print(
            f"模型结构：51 -> {hidden_size} x {hidden_layers + 1} -> 39；"
            f"checkpoint step={checkpoint.get('step', 'unknown')}"
        )

        if device.type == "cuda" and Config.USE_CUDA_GRAPH:
            try:
                self._prepare_cuda_graph()
            except Exception as error:
                self.static_input = None
                self.static_output = None
                self.graph = None
                print(f"CUDA Graph 初始化失败，退回普通推理：{error}")

    def _prepare_cuda_graph(self) -> None:
        assert self.model is not None
        self.static_input = torch.zeros((1, INPUT_BITS), device=self.device)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(5):
                self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_output = self.model(self.static_input)
        print("已启用 CUDA Graph 单步推理。")

    @torch.inference_mode()
    def predict(
        self,
        instruction: int,
        registers: list[int],
        flags: list[int],
        exact_bits: list[int],
    ) -> tuple[list[int], list[float], list[float]]:
        if Config.EXACT_ONLY:
            logits = [20.0 if bit else -20.0 for bit in exact_bits]
            probabilities = [1.0 if bit else 0.0 for bit in exact_bits]
            return exact_bits.copy(), logits, probabilities

        assert self.model is not None
        input_bits = [(instruction >> shift) & 1 for shift in range(15, -1, -1)]
        input_bits.extend(flags)
        for value in registers:
            input_bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
        input_tensor = torch.tensor(
            input_bits, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        if self.graph is not None:
            assert self.static_input is not None and self.static_output is not None
            self.static_input.copy_(input_tensor)
            self.graph.replay()
            output = self.static_output[0]
        else:
            output = self.model(input_tensor)[0]

        predicted = (output >= 0).to(torch.int64)
        probabilities = torch.sigmoid(output)
        return (
            predicted.cpu().tolist(),
            output.float().cpu().tolist(),
            probabilities.float().cpu().tolist(),
        )

    @torch.inference_mode()
    def find_first_batch_mismatch(
        self,
        input_bits: np.ndarray,
        target_bits: np.ndarray,
    ) -> dict[str, Any] | None:
        """批量筛选差异和低 margin 行，再用原 batch=1 路径逐个复核。"""
        if self.model is None:
            raise RuntimeError("batch 轨迹审计必须加载神经 CPU 权重。")

        total = int(input_bits.shape[0])
        batch_size = max(1, Config.INFERENCE_BATCH)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            inputs = torch.from_numpy(input_bits[start:end]).to(
                self.device, dtype=torch.float32, non_blocking=False
            )
            expected = torch.from_numpy(target_bits[start:end]).to(
                self.device, dtype=torch.bool, non_blocking=False
            )
            logits = self.model(inputs)
            predicted = logits >= 0
            row_mismatch = torch.any(predicted != expected, dim=1)
            low_margin = torch.amin(torch.abs(logits), dim=1) < Config.BATCH_RECHECK_MARGIN
            candidate_indices = torch.nonzero(
                row_mismatch | low_margin, as_tuple=False
            )
            if candidate_indices.numel() == 0:
                continue

            for candidate in candidate_indices[:, 0].cpu().tolist():
                row = start + int(candidate)
                input_row = input_bits[row]
                instruction = 0
                for bit in input_row[:INSTRUCTION_BITS]:
                    instruction = (instruction << 1) | int(bit)
                cursor = INSTRUCTION_BITS
                flags = input_row[cursor : cursor + NUM_FLAGS].astype(int).tolist()
                cursor += NUM_FLAGS
                registers: list[int] = []
                for _ in range(NUM_REGISTERS):
                    value = 0
                    for bit in input_row[cursor : cursor + REGISTER_BITS]:
                        value = (value << 1) | int(bit)
                    registers.append(value)
                    cursor += REGISTER_BITS
                exact_bits = target_bits[row].astype(int).tolist()
                serial_predicted, serial_logits, serial_probabilities = self.predict(
                    instruction, registers, flags, exact_bits
                )
                if serial_predicted != exact_bits:
                    return {
                        "index": row,
                        "logits": serial_logits,
                        "probabilities": serial_probabilities,
                        "predicted_bits": serial_predicted,
                        "expected_bits": exact_bits,
                        "batch_candidate_was_mismatch": bool(
                            row_mismatch[int(candidate)].item()
                        ),
                    }
        return None


# =============================================================================
# 汇编器
# =============================================================================


@dataclass
class AssemblyInstruction:
    op: str
    rd: int = 0
    ra: int = 0
    rb: int = 0
    imm: int = 0
    target: str | None = None
    comment: str = ""


class Assembler:
    def __init__(self):
        self.instructions: list[AssemblyInstruction] = []
        self.labels: dict[str, int] = {}
        self._unique_id = 0

    def unique(self, prefix: str) -> str:
        self._unique_id += 1
        return f"{prefix}_{self._unique_id}"

    def label(self, name: str) -> None:
        if name in self.labels:
            raise ValueError(f"重复汇编标签：{name}")
        self.labels[name] = len(self.instructions)

    def emit(
        self,
        op: str,
        rd: int = 0,
        ra: int = 0,
        rb: int = 0,
        imm: int = 0,
        target: str | None = None,
        comment: str = "",
    ) -> None:
        self.instructions.append(
            AssemblyInstruction(op, rd, ra, rb, imm, target, comment)
        )

    def branch(self, op: str, target: str, comment: str = "") -> None:
        self.emit(op, target=target, comment=comment)

    def assemble(self) -> tuple[list[int], list[dict[str, Any]]]:
        machine_code: list[int] = []
        listing: list[dict[str, Any]] = []
        for pc, item in enumerate(self.instructions):
            opcode = OPCODES[item.op]
            if item.op in RRR_NAMES:
                word = (
                    (opcode << 11)
                    | ((item.rd & 3) << 9)
                    | ((item.ra & 3) << 7)
                    | ((item.rb & 3) << 5)
                )
            elif item.op in RI8_NAMES:
                word = (opcode << 11) | ((item.rd & 3) << 9) | (item.imm & 0xFF)
            elif item.op in BRANCH_NAMES:
                if item.target is None or item.target not in self.labels:
                    raise ValueError(f"未定义的跳转标签：{item.target}")
                offset = self.labels[item.target] - (pc + 1)
                if not -1024 <= offset <= 1023:
                    raise ValueError(f"跳转距离超出 BR11 范围：pc={pc}, offset={offset}")
                word = (opcode << 11) | (offset & 0x7FF)
            elif item.op == "JMPR":
                word = (opcode << 11) | ((item.ra & 3) << 9)
            else:
                word = opcode << 11
            machine_code.append(word)
            listing.append(
                {
                    "pc": pc,
                    "word": f"0x{word:04X}",
                    **asdict(item),
                }
            )
        return machine_code, listing


# =============================================================================
# pi spigot 汇编程序生成器
# =============================================================================


class MemoryMap:
    I = 0x00  # uint16
    LENGTH = 0x02  # uint16
    Q = 0x04  # uint8
    X = 0x05  # uint16 low word
    A = 0x07  # uint16
    DEN = 0x09  # uint16
    COUNTER = 0x0B  # uint8
    REMAIN = 0x0C  # uint16
    NINES = 0x0E  # uint16
    PREDIGIT = 0x10  # uint8
    NEW_PREDIGIT = 0x11  # uint8
    ARRAY_POINTER = 0x12  # uint16
    X_HIGH = 0x14  # uint8；X 的第 17..24 bit

    ARRAY_BASE = 0x1000
    OUTPUT_PORT = 0xFF00


class PiAssemblyBuilder:
    def __init__(self, decimal_digits: int):
        self.n = decimal_digits
        self.length = (10 * decimal_digits) // 3
        self.asm = Assembler()

    def load_abs(self, dst: int, address: int) -> None:
        if dst not in (0, 1):
            raise ValueError("绝对 LOAD 宏只把数据装入 R0/R1。")
        self.asm.emit("MOVI", rd=2, imm=(address >> 8) & 0xFF)
        self.asm.emit("MOVI", rd=3, imm=address & 0xFF)
        self.asm.emit("LOAD", rd=dst, ra=2, rb=3)

    def store_abs(self, src: int, address: int) -> None:
        if src not in (0, 1):
            raise ValueError("绝对 STORE 宏只从 R0/R1 写数据。")
        self.asm.emit("MOVI", rd=2, imm=(address >> 8) & 0xFF)
        self.asm.emit("MOVI", rd=3, imm=address & 0xFF)
        self.asm.emit("STORE", rd=src, ra=2, rb=3)

    def write_u8(self, address: int, value: int) -> None:
        self.asm.emit("MOVI", rd=0, imm=value)
        self.store_abs(0, address)

    def write_u16(self, address: int, value: int) -> None:
        self.write_u8(address, value & 0xFF)
        self.write_u8(address + 1, (value >> 8) & 0xFF)

    def copy_u16(self, source: int, destination: int) -> None:
        self.load_abs(0, source)
        self.store_abs(0, destination)
        self.load_abs(0, source + 1)
        self.store_abs(0, destination + 1)

    def increment_u16(self, address: int) -> None:
        done = self.asm.unique("inc16_done")
        self.load_abs(0, address)
        self.asm.emit("ADDI", rd=0, imm=1)
        self.store_abs(0, address)
        self.asm.branch("JNC", done)
        self.load_abs(0, address + 1)
        self.asm.emit("ADDI", rd=0, imm=1)
        self.store_abs(0, address + 1)
        self.asm.label(done)

    def decrement_u16(self, address: int) -> None:
        done = self.asm.unique("dec16_done")
        self.load_abs(0, address)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, address)
        self.asm.branch("JNC", done)
        self.load_abs(0, address + 1)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, address + 1)
        self.asm.label(done)

    def branch_if_u16_zero(self, address: int, target: str) -> None:
        nonzero = self.asm.unique("u16_nonzero")
        self.load_abs(0, address + 1)
        self.asm.emit("CMPI", rd=0, imm=0)
        self.asm.branch("JNZ", nonzero)
        self.load_abs(0, address)
        self.asm.emit("CMPI", rd=0, imm=0)
        self.asm.branch("JZ", target)
        self.asm.label(nonzero)

    def add_u16(self, destination: int, source: int) -> None:
        # 地址设置、LOAD 和 STORE 都不改 CF，因此低字节进位可安全传给 ADC。
        self.load_abs(0, destination)
        self.load_abs(1, source)
        self.asm.emit("ADD", rd=0, ra=0, rb=1)
        self.store_abs(0, destination)
        self.load_abs(0, destination + 1)
        self.load_abs(1, source + 1)
        self.asm.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, destination + 1)

    def subtract_u16(self, destination: int, source: int) -> None:
        self.load_abs(0, destination)
        self.load_abs(1, source)
        self.asm.emit("SUB", rd=0, ra=0, rb=1)
        self.store_abs(0, destination)
        self.load_abs(0, destination + 1)
        self.load_abs(1, source + 1)
        self.asm.emit("SBC", rd=0, ra=0, rb=1)
        self.store_abs(0, destination + 1)

    def clear_x24(self) -> None:
        self.write_u16(MemoryMap.X, 0)
        self.write_u8(MemoryMap.X_HIGH, 0)

    def add_u16_to_x24(self, source: int) -> None:
        """X24 += uint16(source)，把高字节进位保留下来。"""
        self.load_abs(0, MemoryMap.X)
        self.load_abs(1, source)
        self.asm.emit("ADD", rd=0, ra=0, rb=1)
        self.store_abs(0, MemoryMap.X)
        self.load_abs(0, MemoryMap.X + 1)
        self.load_abs(1, source + 1)
        self.asm.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, MemoryMap.X + 1)
        self.load_abs(0, MemoryMap.X_HIGH)
        self.asm.emit("MOVI", rd=1, imm=0)
        self.asm.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, MemoryMap.X_HIGH)

    def subtract_u16_from_x24(self, source: int) -> None:
        """X24 -= uint16(source)，把高字节借位传播完整。"""
        self.load_abs(0, MemoryMap.X)
        self.load_abs(1, source)
        self.asm.emit("SUB", rd=0, ra=0, rb=1)
        self.store_abs(0, MemoryMap.X)
        self.load_abs(0, MemoryMap.X + 1)
        self.load_abs(1, source + 1)
        self.asm.emit("SBC", rd=0, ra=0, rb=1)
        self.store_abs(0, MemoryMap.X + 1)
        self.load_abs(0, MemoryMap.X_HIGH)
        self.asm.emit("MOVI", rd=1, imm=0)
        self.asm.emit("SBC", rd=0, ra=0, rb=1)
        self.store_abs(0, MemoryMap.X_HIGH)

    def compare_x24_to_u16(
        self, right: int, less: str, greater: str, equal: str
    ) -> None:
        """无符号比较 X24 与 uint16(right)。"""
        self.load_abs(0, MemoryMap.X_HIGH)
        self.asm.emit("CMPI", rd=0, imm=0)
        self.asm.branch("JNZ", greater)
        self.compare_u16(MemoryMap.X, right, less, greater, equal)

    def compare_u16(
        self, left: int, right: int, less: str, greater: str, equal: str
    ) -> None:
        self.load_abs(0, left + 1)
        self.load_abs(1, right + 1)
        self.asm.emit("CMP", ra=0, rb=1)
        self.asm.branch("JL", less)
        self.asm.branch("JG", greater)
        self.load_abs(0, left)
        self.load_abs(1, right)
        self.asm.emit("CMP", ra=0, rb=1)
        self.asm.branch("JL", less)
        self.asm.branch("JG", greater)
        self.asm.branch("JMP", equal)

    def compute_array_pointer(self) -> None:
        no_borrow = self.asm.unique("array_i_minus_one")
        self.load_abs(0, MemoryMap.I)
        self.load_abs(1, MemoryMap.I + 1)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.asm.branch("JNC", no_borrow)
        self.asm.emit("SUBI", rd=1, imm=1)
        self.asm.label(no_borrow)
        # 2 * (i - 1)
        self.asm.emit("ADD", rd=0, ra=0, rb=0)
        self.asm.emit("ADC", rd=1, ra=1, rb=1)
        # 加 ARRAY_BASE。MOVI 不改变前一条 ADDI 产生的 CF。
        self.asm.emit("ADDI", rd=0, imm=MemoryMap.ARRAY_BASE & 0xFF)
        self.asm.emit("MOVI", rd=2, imm=(MemoryMap.ARRAY_BASE >> 8) & 0xFF)
        self.asm.emit("ADC", rd=1, ra=1, rb=2)
        self.store_abs(0, MemoryMap.ARRAY_POINTER)
        self.store_abs(1, MemoryMap.ARRAY_POINTER + 1)

    def restore_array_pointer(self) -> None:
        # 保留 R0 中待 LOAD/STORE 的数据，R1 作为地址装载辅助寄存器。
        self.asm.emit("MOVI", rd=1, imm=0)
        self.asm.emit("MOVI", rd=2, imm=MemoryMap.ARRAY_POINTER)
        self.asm.emit("LOAD", rd=3, ra=1, rb=2)
        self.asm.emit("MOVI", rd=2, imm=MemoryMap.ARRAY_POINTER + 1)
        self.asm.emit("LOAD", rd=2, ra=1, rb=2)

    def load_array_element_to_a(self) -> None:
        self.compute_array_pointer()
        self.restore_array_pointer()
        self.asm.emit("LOAD", rd=0, ra=2, rb=3)
        self.store_abs(0, MemoryMap.A)
        self.increment_u16(MemoryMap.ARRAY_POINTER)
        self.restore_array_pointer()
        self.asm.emit("LOAD", rd=0, ra=2, rb=3)
        self.store_abs(0, MemoryMap.A + 1)

    def store_x_to_array_element(self) -> None:
        self.compute_array_pointer()
        self.load_abs(0, MemoryMap.X)
        self.restore_array_pointer()
        self.asm.emit("STORE", rd=0, ra=2, rb=3)
        self.increment_u16(MemoryMap.ARRAY_POINTER)
        self.load_abs(0, MemoryMap.X + 1)
        self.restore_array_pointer()
        self.asm.emit("STORE", rd=0, ra=2, rb=3)

    def initialize_array_element_to_two(self) -> None:
        self.compute_array_pointer()
        self.asm.emit("MOVI", rd=0, imm=2)
        self.restore_array_pointer()
        self.asm.emit("STORE", rd=0, ra=2, rb=3)
        self.increment_u16(MemoryMap.ARRAY_POINTER)
        self.asm.emit("MOVI", rd=0, imm=0)
        self.restore_array_pointer()
        self.asm.emit("STORE", rd=0, ra=2, rb=3)

    def emit_digit_from_r0(self) -> None:
        self.asm.emit("MOVI", rd=2, imm=0xFF)
        self.asm.emit("MOVI", rd=3, imm=0x00)
        self.asm.emit("STORE", rd=0, ra=2, rb=3)

    def emit_digit_from_memory(self, address: int, add_one: bool = False) -> None:
        self.load_abs(0, address)
        if add_one:
            self.asm.emit("ADDI", rd=0, imm=1)
        self.emit_digit_from_r0()

    def build(self) -> tuple[list[int], list[dict[str, Any]]]:
        if not 1 <= self.n <= 5000:
            raise ValueError("当前 spigot 汇编实现要求数字数在 1..5000 之间。")
        if self.length >= 0x7800:
            raise ValueError("spigot 数组超出预留 RAM。")

        a = self.asm
        self.write_u16(MemoryMap.LENGTH, self.length)
        self.write_u16(MemoryMap.REMAIN, self.n)
        self.write_u16(MemoryMap.NINES, 0)
        self.write_u8(MemoryMap.PREDIGIT, 0)
        self.copy_u16(MemoryMap.LENGTH, MemoryMap.I)

        a.label("init_array")
        self.branch_if_u16_zero(MemoryMap.I, "init_done")
        self.initialize_array_element_to_two()
        self.decrement_u16(MemoryMap.I)
        a.branch("JMP", "init_array")

        a.label("init_done")
        a.label("outer_loop")
        self.write_u8(MemoryMap.Q, 0)
        self.copy_u16(MemoryMap.LENGTH, MemoryMap.I)

        a.label("inner_loop")
        self.load_array_element_to_a()

        # X = 10 * A：通过十次 16-bit 加法实现，避免引入未训练的新操作。
        self.clear_x24()
        self.write_u8(MemoryMap.COUNTER, 10)
        a.label("multiply_a_by_10")
        self.load_abs(0, MemoryMap.COUNTER)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JZ", "multiply_q_by_i_setup")
        self.add_u16_to_x24(MemoryMap.A)
        self.load_abs(0, MemoryMap.COUNTER)
        a.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, MemoryMap.COUNTER)
        a.branch("JMP", "multiply_a_by_10")

        # X += Q * I。Q 很小，重复加法比通用乘法更省机器指令。
        a.label("multiply_q_by_i_setup")
        self.load_abs(0, MemoryMap.Q)
        self.store_abs(0, MemoryMap.COUNTER)
        a.label("multiply_q_by_i")
        self.load_abs(0, MemoryMap.COUNTER)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JZ", "prepare_denominator")
        self.add_u16_to_x24(MemoryMap.I)
        self.load_abs(0, MemoryMap.COUNTER)
        a.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, MemoryMap.COUNTER)
        a.branch("JMP", "multiply_q_by_i")

        # DEN = 2 * I - 1。
        a.label("prepare_denominator")
        self.copy_u16(MemoryMap.I, MemoryMap.DEN)
        self.add_u16(MemoryMap.DEN, MemoryMap.I)
        self.decrement_u16(MemoryMap.DEN)
        self.write_u8(MemoryMap.Q, 0)

        # divmod(X, DEN)：本算法的商很小，使用重复减法。
        a.label("division_loop")
        self.compare_x24_to_u16(
            MemoryMap.DEN,
            less="division_done",
            greater="division_subtract",
            equal="division_subtract",
        )
        a.label("division_subtract")
        self.subtract_u16_from_x24(MemoryMap.DEN)
        self.load_abs(0, MemoryMap.Q)
        a.emit("ADDI", rd=0, imm=1)
        self.store_abs(0, MemoryMap.Q)
        a.branch("JMP", "division_loop")

        a.label("division_done")
        self.store_x_to_array_element()
        self.decrement_u16(MemoryMap.I)
        self.branch_if_u16_zero(MemoryMap.I, "inner_done")
        a.branch("JMP", "inner_loop")

        # A[1] = Q mod 10；NEW_PREDIGIT = Q div 10。
        a.label("inner_done")
        self.write_u8(MemoryMap.NEW_PREDIGIT, 0)
        a.label("q_div_10")
        self.load_abs(0, MemoryMap.Q)
        a.emit("CMPI", rd=0, imm=10)
        a.branch("JL", "q_div_10_done")
        a.emit("SUBI", rd=0, imm=10)
        self.store_abs(0, MemoryMap.Q)
        self.load_abs(0, MemoryMap.NEW_PREDIGIT)
        a.emit("ADDI", rd=0, imm=1)
        self.store_abs(0, MemoryMap.NEW_PREDIGIT)
        a.branch("JMP", "q_div_10")

        a.label("q_div_10_done")
        # 数组第一项是固定地址 ARRAY_BASE，写低字节余数和高字节 0。
        self.load_abs(0, MemoryMap.Q)
        self.store_abs(0, MemoryMap.ARRAY_BASE)
        self.write_u8(MemoryMap.ARRAY_BASE + 1, 0)
        self.load_abs(0, MemoryMap.NEW_PREDIGIT)
        self.store_abs(0, MemoryMap.Q)

        self.load_abs(0, MemoryMap.Q)
        a.emit("CMPI", rd=0, imm=9)
        a.branch("JZ", "predigit_is_9")
        a.emit("CMPI", rd=0, imm=10)
        a.branch("JZ", "predigit_is_10")

        # 普通 predigit：释放旧 predigit，再释放队列中的 9。
        self.emit_digit_from_memory(MemoryMap.PREDIGIT)
        self.load_abs(0, MemoryMap.Q)
        self.store_abs(0, MemoryMap.PREDIGIT)
        a.label("emit_held_nines")
        self.branch_if_u16_zero(MemoryMap.NINES, "finish_outer_iteration")
        a.emit("MOVI", rd=0, imm=9)
        self.emit_digit_from_r0()
        self.decrement_u16(MemoryMap.NINES)
        a.branch("JMP", "emit_held_nines")

        a.label("predigit_is_9")
        self.increment_u16(MemoryMap.NINES)
        a.branch("JMP", "finish_outer_iteration")

        # predigit=10：旧 predigit 加一，所有暂存的 9 变成 0。
        a.label("predigit_is_10")
        self.emit_digit_from_memory(MemoryMap.PREDIGIT, add_one=True)
        self.write_u8(MemoryMap.PREDIGIT, 0)
        a.label("emit_held_zeros")
        self.branch_if_u16_zero(MemoryMap.NINES, "finish_outer_iteration")
        a.emit("MOVI", rd=0, imm=0)
        self.emit_digit_from_r0()
        self.decrement_u16(MemoryMap.NINES)
        a.branch("JMP", "emit_held_zeros")

        a.label("finish_outer_iteration")
        self.decrement_u16(MemoryMap.REMAIN)
        self.branch_if_u16_zero(MemoryMap.REMAIN, "program_finish")
        a.branch("JMP", "outer_loop")

        a.label("program_finish")
        self.emit_digit_from_memory(MemoryMap.PREDIGIT)
        a.emit("HALT")
        return a.assemble()


# =============================================================================
# 参考 pi 与执行器
# =============================================================================


def reference_pi_digits(digits: int) -> str:
    """用 Python Decimal/Chudnovsky 生成显示与验收用参考值，不参与 CPU 运算。"""
    with localcontext() as context:
        context.prec = digits + 40
        terms = digits // 14 + 3
        m = 1
        l = 13_591_409
        x = 1
        k = 6
        total = Decimal(l)
        for i in range(1, terms):
            m = (m * (k**3 - 16 * k)) // (i**3)
            l += 545_140_134
            x *= -262_537_412_640_768_000
            total += Decimal(m * l) / Decimal(x)
            k += 12
        pi = (Decimal(426_880) * Decimal(10_005).sqrt()) / total
        text = format(+pi, "f").replace(".", "")
        return text[:digits]


def format_pi(digits: str) -> str:
    if not digits:
        return "(尚未产生数字)"
    if len(digits) == 1:
        return digits[0] + "."
    return digits[0] + "." + digits[1:]


@dataclass
class MachineState:
    pc: int
    registers: list[int]
    flags: list[int]


# =============================================================================
# batch 轨迹审计：原生真值轨迹生成器
# =============================================================================


NATIVE_TRACE_SOURCE = r"""
#include <cstdint>
#include <cstring>

namespace {
constexpr int NUM_REGISTERS = 4;
constexpr int NUM_FLAGS = 3;
constexpr int INPUT_BITS = 51;
constexpr int OUTPUT_BITS = 39;

constexpr int FLAG_ZF = 0;
constexpr int FLAG_GF = 1;
constexpr int FLAG_CF = 2;
constexpr int CTRL_MEMORY_READ = 0;
constexpr int CTRL_MEMORY_WRITE = 1;
constexpr int CTRL_BRANCH_TAKEN = 2;
constexpr int CTRL_HALT = 3;

constexpr int OP_HALT = 1;
constexpr int OP_MOV = 2;
constexpr int OP_MOVI = 3;
constexpr int OP_LOAD = 4;
constexpr int OP_STORE = 5;
constexpr int OP_ADD = 6;
constexpr int OP_ADC = 7;
constexpr int OP_SUB = 8;
constexpr int OP_SBC = 9;
constexpr int OP_INC = 10;
constexpr int OP_DEC = 11;
constexpr int OP_AND = 12;
constexpr int OP_OR = 13;
constexpr int OP_XOR = 14;
constexpr int OP_NOT = 15;
constexpr int OP_SHL = 16;
constexpr int OP_SHR = 17;
constexpr int OP_CMP = 18;
constexpr int OP_ADDI = 19;
constexpr int OP_SUBI = 20;
constexpr int OP_CMPI = 21;
constexpr int OP_JMP = 22;
constexpr int OP_JZ = 23;
constexpr int OP_JNZ = 24;
constexpr int OP_JG = 25;
constexpr int OP_JL = 26;
constexpr int OP_JC = 27;
constexpr int OP_JNC = 28;
constexpr int OP_JMPR = 29;
constexpr int OP_TRAP = 31;

inline void fields(
    uint16_t instruction,
    int& opcode,
    int& p1,
    int& p2,
    int& p3,
    int& immediate,
    int& offset
) {
    opcode = (instruction >> 11) & 0x1F;
    p1 = (instruction >> 9) & 0x03;
    p2 = (instruction >> 7) & 0x03;
    p3 = (instruction >> 5) & 0x03;
    immediate = instruction & 0xFF;
    offset = instruction & 0x7FF;
    if (offset & 0x400) offset -= 0x800;
}

inline void encode_input(
    uint8_t* destination,
    uint16_t instruction,
    const uint8_t* registers,
    const uint8_t* flags
) {
    int cursor = 0;
    for (int shift = 15; shift >= 0; --shift) {
        destination[cursor++] = (instruction >> shift) & 1;
    }
    for (int i = 0; i < NUM_FLAGS; ++i) destination[cursor++] = flags[i];
    for (int reg = 0; reg < NUM_REGISTERS; ++reg) {
        for (int shift = 7; shift >= 0; --shift) {
            destination[cursor++] = (registers[reg] >> shift) & 1;
        }
    }
}

inline void encode_target(
    uint8_t* destination,
    const uint8_t* registers,
    const uint8_t* flags,
    const uint8_t* controls
) {
    int cursor = 0;
    for (int i = 0; i < NUM_FLAGS; ++i) destination[cursor++] = flags[i];
    for (int reg = 0; reg < NUM_REGISTERS; ++reg) {
        for (int shift = 7; shift >= 0; --shift) {
            destination[cursor++] = (registers[reg] >> shift) & 1;
        }
    }
    for (int i = 0; i < 4; ++i) destination[cursor++] = controls[i];
}

inline void transition(
    uint16_t instruction,
    uint8_t* registers,
    uint8_t* flags,
    uint8_t* controls
) {
    int opcode, p1, p2, p3, immediate, offset;
    fields(instruction, opcode, p1, p2, p3, immediate, offset);
    controls[0] = controls[1] = controls[2] = controls[3] = 0;
    const int value1 = registers[p2];
    const int value2 = registers[p3];
    const int destination_value = registers[p1];
    const int carry = flags[FLAG_CF];

    auto write = [&](int value) -> int {
        registers[p1] = static_cast<uint8_t>(value & 0xFF);
        return registers[p1];
    };

    if (opcode == OP_HALT) {
        controls[CTRL_HALT] = 1;
    } else if (opcode == OP_MOV) {
        write(value1);
    } else if (opcode == OP_MOVI) {
        write(immediate);
    } else if (opcode == OP_LOAD) {
        controls[CTRL_MEMORY_READ] = 1;
    } else if (opcode == OP_STORE) {
        controls[CTRL_MEMORY_WRITE] = 1;
    } else if (opcode == OP_ADD || opcode == OP_ADC) {
        const int result = value1 + value2 + (opcode == OP_ADC ? carry : 0);
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
        flags[FLAG_CF] = result > 0xFF;
    } else if (opcode == OP_SUB || opcode == OP_SBC) {
        const int borrow = opcode == OP_SBC ? carry : 0;
        const int result = value1 - value2 - borrow;
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
        flags[FLAG_CF] = value1 < value2 + borrow;
    } else if (opcode == OP_INC) {
        const int result = value1 + 1;
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
        flags[FLAG_CF] = result > 0xFF;
    } else if (opcode == OP_DEC) {
        const int result = value1 - 1;
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
        flags[FLAG_CF] = value1 == 0;
    } else if (opcode == OP_AND || opcode == OP_OR || opcode == OP_XOR) {
        int result = 0;
        if (opcode == OP_AND) result = value1 & value2;
        else if (opcode == OP_OR) result = value1 | value2;
        else result = value1 ^ value2;
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
    } else if (opcode == OP_NOT) {
        const int written = write(~value1);
        flags[FLAG_ZF] = written == 0;
    } else if (opcode == OP_SHL || opcode == OP_SHR) {
        const int shift = value2 & 7;
        int result = 0;
        int new_carry = carry;
        if (opcode == OP_SHL) {
            result = value1 << shift;
            if (shift != 0) new_carry = (value1 >> (8 - shift)) & 1;
        } else {
            result = value1 >> shift;
            if (shift != 0) new_carry = (value1 >> (shift - 1)) & 1;
        }
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
        flags[FLAG_CF] = new_carry;
    } else if (opcode == OP_CMP) {
        flags[FLAG_ZF] = value1 == value2;
        flags[FLAG_GF] = value1 > value2;
        flags[FLAG_CF] = value1 < value2;
    } else if (opcode == OP_ADDI || opcode == OP_SUBI) {
        int result = 0;
        int new_carry = 0;
        if (opcode == OP_ADDI) {
            result = destination_value + immediate;
            new_carry = result > 0xFF;
        } else {
            result = destination_value - immediate;
            new_carry = destination_value < immediate;
        }
        const int written = write(result);
        flags[FLAG_ZF] = written == 0;
        flags[FLAG_CF] = new_carry;
    } else if (opcode == OP_CMPI) {
        flags[FLAG_ZF] = destination_value == immediate;
        flags[FLAG_GF] = destination_value > immediate;
        flags[FLAG_CF] = destination_value < immediate;
    } else if (opcode == OP_JMP) {
        controls[CTRL_BRANCH_TAKEN] = 1;
    } else if (opcode == OP_JZ) {
        controls[CTRL_BRANCH_TAKEN] = flags[FLAG_ZF];
    } else if (opcode == OP_JNZ) {
        controls[CTRL_BRANCH_TAKEN] = 1 - flags[FLAG_ZF];
    } else if (opcode == OP_JG) {
        controls[CTRL_BRANCH_TAKEN] = flags[FLAG_GF];
    } else if (opcode == OP_JL || opcode == OP_JC) {
        controls[CTRL_BRANCH_TAKEN] = flags[FLAG_CF];
    } else if (opcode == OP_JNC) {
        controls[CTRL_BRANCH_TAKEN] = 1 - flags[FLAG_CF];
    } else if (opcode == OP_JMPR) {
        controls[CTRL_BRANCH_TAKEN] = 1;
    } else if (opcode == OP_TRAP) {
        controls[CTRL_HALT] = 1;
    }
}
}  // namespace

#if defined(_WIN32)
#define NEURAL_CPU_EXPORT __declspec(dllexport)
#else
#define NEURAL_CPU_EXPORT __attribute__((visibility("default")))
#endif

extern "C" NEURAL_CPU_EXPORT int generate_trace_chunk(
    const uint16_t* program,
    uint32_t program_length,
    uint8_t* ram,
    uint16_t* pc,
    uint8_t* registers,
    uint8_t* flags,
    uint64_t* fetched_instructions,
    uint8_t* halted,
    uint64_t max_records,
    uint8_t* input_bits,
    uint8_t* target_bits,
    uint16_t* record_instructions,
    uint16_t* record_pcs,
    uint8_t* record_micro_ops,
    uint64_t* record_fetch_indices,
    uint64_t* records_written,
    uint64_t* output_record_indices,
    uint8_t* output_digits,
    uint32_t output_capacity,
    uint32_t* output_count
) {
    uint64_t count = 0;
    uint32_t outputs = 0;
    while (count < max_records && !*halted) {
        if (*pc >= program_length) return -1;
        const uint16_t instruction = program[*pc];
        int opcode, p1, p2, p3, immediate, offset;
        fields(instruction, opcode, p1, p2, p3, immediate, offset);
        if (opcode == OP_LOAD && count + 2 > max_records) break;

        const uint16_t current_pc = *pc;
        uint8_t before_registers[NUM_REGISTERS];
        std::memcpy(before_registers, registers, NUM_REGISTERS);
        ++(*fetched_instructions);

        encode_input(input_bits + count * INPUT_BITS, instruction, registers, flags);
        uint8_t controls[4];
        transition(instruction, registers, flags, controls);
        encode_target(target_bits + count * OUTPUT_BITS, registers, flags, controls);
        record_instructions[count] = instruction;
        record_pcs[count] = current_pc;
        record_micro_ops[count] = 0;
        record_fetch_indices[count] = *fetched_instructions;
        ++count;

        if (controls[CTRL_MEMORY_WRITE]) {
            const uint16_t address =
                (static_cast<uint16_t>(before_registers[p2]) << 8) |
                before_registers[p3];
            const uint8_t value = before_registers[p1];
            if (address == 0xFF00) {
                if (outputs >= output_capacity) return -2;
                output_record_indices[outputs] = count - 1;
                output_digits[outputs] = value;
                ++outputs;
            } else {
                ram[address] = value;
            }
        }

        if (controls[CTRL_MEMORY_READ]) {
            const uint16_t address =
                (static_cast<uint16_t>(before_registers[p2]) << 8) |
                before_registers[p3];
            const uint8_t value = ram[address];
            const uint16_t movi =
                (static_cast<uint16_t>(OP_MOVI) << 11) |
                (static_cast<uint16_t>(p1 & 3) << 9) |
                value;
            encode_input(input_bits + count * INPUT_BITS, movi, registers, flags);
            uint8_t movi_controls[4];
            transition(movi, registers, flags, movi_controls);
            encode_target(
                target_bits + count * OUTPUT_BITS,
                registers,
                flags,
                movi_controls
            );
            record_instructions[count] = movi;
            record_pcs[count] = current_pc;
            record_micro_ops[count] = 1;
            record_fetch_indices[count] = *fetched_instructions;
            ++count;
        }

        if (controls[CTRL_HALT]) {
            *halted = 1;
            break;
        }
        if (controls[CTRL_BRANCH_TAKEN]) {
            if (opcode == OP_JMPR) {
                *pc = before_registers[p1];
            } else {
                *pc = static_cast<uint16_t>(current_pc + 1 + offset);
            }
        } else {
            *pc = static_cast<uint16_t>(current_pc + 1);
        }
    }
    *records_written = count;
    *output_count = outputs;
    return 0;
}
"""


class NativeTraceGenerator:
    """用编译后的精确解释器流式生成神经 CPU 的真实逐操作轨迹。"""

    def __init__(self, program: list[int]):
        if os.name == "nt":
            raise RuntimeError("原生 batch 轨迹生成器当前面向 Linux 环境。")
        self.library = self._load_library()
        self.function = self.library.generate_trace_chunk
        self.function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.function.restype = ctypes.c_int

        self.program = np.ascontiguousarray(program, dtype=np.uint16)
        self.ram = np.zeros(65_536, dtype=np.uint8)
        self.pc = np.zeros(1, dtype=np.uint16)
        self.registers = np.zeros(NUM_REGISTERS, dtype=np.uint8)
        self.flags = np.zeros(NUM_FLAGS, dtype=np.uint8)
        self.fetched = np.zeros(1, dtype=np.uint64)
        self.halted = np.zeros(1, dtype=np.uint8)

    @staticmethod
    def _pointer(array: np.ndarray) -> ctypes.c_void_p:
        return ctypes.c_void_p(array.ctypes.data)

    @classmethod
    def _load_library(cls) -> ctypes.CDLL:
        compiler = next(
            (shutil.which(name) for name in ("c++", "g++", "clang++") if shutil.which(name)),
            None,
        )
        if compiler is None:
            raise RuntimeError(
                "找不到 C++ 编译器；请安装 g++，并用 `which g++` 检查。"
            )
        digest = hashlib.sha256(NATIVE_TRACE_SOURCE.encode("utf-8")).hexdigest()[:16]
        cache_dir = Path.home() / ".cache" / "neural_cpu_pi_trace"
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_path = cache_dir / f"trace_{digest}.cpp"
        library_path = cache_dir / f"trace_{digest}.so"
        if not library_path.exists():
            source_path.write_text(NATIVE_TRACE_SOURCE, encoding="utf-8")
            command = [
                compiler,
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(source_path),
                "-o",
                str(library_path),
            ]
            print("首次运行 batch 模式：正在编译原生真值轨迹生成器……")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    "原生轨迹生成器编译失败：\n"
                    + result.stdout
                    + "\n"
                    + result.stderr
                )
            print(f"原生轨迹生成器已缓存：{library_path}")
        return ctypes.CDLL(str(library_path))

    def next_chunk(self, max_records: int) -> dict[str, Any]:
        capacity = max(2, int(max_records))
        inputs = np.empty((capacity, INPUT_BITS), dtype=np.uint8)
        targets = np.empty((capacity, OUTPUT_BITS), dtype=np.uint8)
        instructions = np.empty(capacity, dtype=np.uint16)
        pcs = np.empty(capacity, dtype=np.uint16)
        micro_ops = np.empty(capacity, dtype=np.uint8)
        fetch_indices = np.empty(capacity, dtype=np.uint64)
        records_written = np.zeros(1, dtype=np.uint64)
        output_indices = np.empty(1024, dtype=np.uint64)
        output_digits = np.empty(1024, dtype=np.uint8)
        output_count = np.zeros(1, dtype=np.uint32)

        code = self.function(
            self._pointer(self.program),
            ctypes.c_uint32(self.program.size),
            self._pointer(self.ram),
            self._pointer(self.pc),
            self._pointer(self.registers),
            self._pointer(self.flags),
            self._pointer(self.fetched),
            self._pointer(self.halted),
            ctypes.c_uint64(capacity),
            self._pointer(inputs),
            self._pointer(targets),
            self._pointer(instructions),
            self._pointer(pcs),
            self._pointer(micro_ops),
            self._pointer(fetch_indices),
            self._pointer(records_written),
            self._pointer(output_indices),
            self._pointer(output_digits),
            ctypes.c_uint32(output_indices.size),
            self._pointer(output_count),
        )
        if code != 0:
            messages = {
                -1: "PC 越界",
                -2: "单块输出数字缓存不足",
            }
            raise RuntimeError(f"原生轨迹生成失败：{messages.get(code, code)}")
        count = int(records_written[0])
        outputs = int(output_count[0])
        return {
            "inputs": inputs[:count],
            "targets": targets[:count],
            "instructions": instructions[:count],
            "pcs": pcs[:count],
            "micro_ops": micro_ops[:count],
            "fetch_indices": fetch_indices[:count],
            "output_indices": output_indices[:outputs],
            "output_digits": output_digits[:outputs],
            "halted": bool(self.halted[0]),
            "pc": int(self.pc[0]),
            "registers": self.registers.astype(int).tolist(),
            "flags": self.flags.astype(int).tolist(),
            "fetched": int(self.fetched[0]),
        }


class NeuralPiMachine:
    def __init__(
        self,
        program: list[int],
        listing: list[dict[str, Any]],
        runner: CoreRunner,
        reference_digits: str,
    ):
        self.program = program
        self.listing = listing
        self.runner = runner
        self.reference_digits = reference_digits
        self.ram = bytearray(65_536)
        self.state = MachineState(0, [0, 0, 0, 0], [0, 0, 0])
        self.neural_operations = 0
        self.fetched_instructions = 0
        self.emitted_digits = ""
        self.ignored_guard_digit = False
        self.started_at = time.perf_counter()
        self.last_progress_operation = 0
        self.last_instruction: dict[str, Any] | None = None
        self.neural_error_operations = 0
        self.neural_bit_errors = 0
        self.error_by_opcode: Counter[str] = Counter()
        self.error_by_pc: Counter[str] = Counter()
        self.error_by_output_bit: Counter[str] = Counter()
        self.error_by_micro_op: Counter[str] = Counter()
        self.error_by_state_hamming_weight: Counter[str] = Counter()
        Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self.error_log_path = Config.RESULT_DIR / "neural_errors.jsonl"
        if self.error_log_path.exists():
            self.error_log_path.unlink()

    def _input_address(self, registers: list[int], ra: int, rb: int) -> int:
        return ((registers[ra] & 0xFF) << 8) | (registers[rb] & 0xFF)

    def _save_json(self, filename: str, payload: dict[str, Any]) -> Path:
        Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path = Config.RESULT_DIR / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def _summary(self, status: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started_at
        valid = 0
        for observed, expected in zip(self.emitted_digits, self.reference_digits):
            if observed != expected:
                break
            valid += 1
        return {
            "status": status,
            "model_path": str(self.runner.model_path or Config.MODEL_PATH),
            "target_decimal_digits": Config.TARGET_DECIMAL_DIGITS,
            "exact_only": Config.EXACT_ONLY,
            "teacher_force_on_error": Config.TEACHER_FORCE_ON_ERROR,
            "max_neural_operations": Config.MAX_NEURAL_OPERATIONS,
            "neural_operations": self.neural_operations,
            "neural_error_operations": self.neural_error_operations,
            "neural_bit_errors": self.neural_bit_errors,
            "neural_operation_error_rate": (
                self.neural_error_operations / max(self.neural_operations, 1)
            ),
            "neural_bit_error_rate": (
                self.neural_bit_errors / max(self.neural_operations * OUTPUT_BITS, 1)
            ),
            "error_by_opcode": dict(self.error_by_opcode.most_common()),
            "error_by_pc": dict(self.error_by_pc.most_common()),
            "error_by_output_bit": dict(self.error_by_output_bit.most_common()),
            "error_by_micro_op": dict(self.error_by_micro_op.most_common()),
            "error_by_state_hamming_weight": dict(
                sorted(self.error_by_state_hamming_weight.items(), key=lambda item: int(item[0]))
            ),
            "fetched_instructions": self.fetched_instructions,
            "elapsed_seconds": elapsed,
            "operations_per_second": self.neural_operations / max(elapsed, 1e-9),
            "pc": self.state.pc,
            "registers": self.state.registers,
            "flags": self.state.flags,
            "pi_digits": self.emitted_digits,
            "pi_decimal": format_pi(self.emitted_digits),
            "valid_significant_digits": valid,
            "valid_decimal_places": max(valid - 1, 0),
        }

    def _append_error_event(self, event: dict[str, Any]) -> None:
        with self.error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_audit_summary(self, status: str) -> Path:
        summary = self._summary(status)
        path = self._save_json("audit_summary.json", summary)
        self._save_json("summary.json", summary)
        return path

    def _print_audit_summary(self, status: str, path: Path) -> None:
        print("\n=== Neural CPU 程序轨迹审计汇总 ===")
        print(
            f"status={status} | operations={self.neural_operations:,} | "
            f"error_operations={self.neural_error_operations:,} "
            f"({self.neural_error_operations / max(self.neural_operations, 1):.12e}) | "
            f"bit_errors={self.neural_bit_errors:,}"
        )
        if self.error_by_opcode:
            print(
                "错误 opcode："
                + ", ".join(
                    f"{name}={count}" for name, count in self.error_by_opcode.most_common()
                )
            )
        print(f"逐错误现场：{self.error_log_path}")
        print(f"聚合统计：{path}")

    def _print_pi_progress(self) -> None:
        summary = self._summary("running")
        count = len(self.emitted_digits)
        if count % Config.PRINT_EVERY_DIGITS != 0:
            return
        print(
            f"[pi] CPU操作={self.neural_operations:,} | 输出={count} 位 | "
            f"有效={summary['valid_significant_digits']} 位 "
            f"(小数点后 {summary['valid_decimal_places']} 位)\n"
            f"     pi ~= {summary['pi_decimal']}"
        )
        if count % Config.SAVE_PROGRESS_EVERY_DIGITS == 0:
            self._save_json("progress.json", summary)

    def _accept_output_digit(self, value: int) -> None:
        if not 0 <= value <= 9:
            raise RuntimeError(f"pi 程序向输出端口写入了非十进制数字：{value}")
        # Rabinowitz-Wagon 算法第一次释放的是初始 guard predigit=0。
        if not self.ignored_guard_digit:
            if value != 0:
                raise RuntimeError(f"首个 guard predigit 应为 0，实际为 {value}")
            self.ignored_guard_digit = True
            return
        self.emitted_digits += str(value)
        self._print_pi_progress()
        expected_prefix = self.reference_digits[: len(self.emitted_digits)]
        if self.emitted_digits != expected_prefix:
            payload = self._summary("pi_program_mismatch")
            payload["expected_prefix"] = expected_prefix
            path = self._save_json("pi_program_mismatch.json", payload)
            raise RuntimeError(f"CPU 无单步错误，但 pi 程序输出与参考值不符：{path}")

    def _execute_neural_operation(
        self,
        instruction: int,
        pc: int,
        micro_op: str | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        opcode, p1, p2, p3, immediate, offset = instruction_fields(instruction)
        before_registers = self.state.registers.copy()
        before_flags = self.state.flags.copy()
        exact_registers, exact_flags, exact_controls = scalar_reference(
            opcode,
            p1,
            p2,
            p3,
            immediate,
            before_registers,
            before_flags,
        )
        exact_bits = state_and_controls_to_bits(
            exact_registers, exact_flags, exact_controls
        )
        predicted_bits, logits, probabilities = self.runner.predict(
            instruction, before_registers, before_flags, exact_bits
        )
        self.neural_operations += 1

        differing = [
            index
            for index, (predicted, expected) in enumerate(
                zip(predicted_bits, exact_bits)
            )
            if predicted != expected
        ]
        if differing:
            predicted = decode_output_bits(predicted_bits)
            expected = decode_output_bits(exact_bits)
            opcode_name = OPCODE_NAMES.get(opcode, str(opcode))
            state_hamming_weight = sum(before_flags) + sum(
                value.bit_count() for value in before_registers
            )
            event = {
                "operation_index": self.neural_operations,
                "fetched_instruction_index": self.fetched_instructions,
                "pc": pc,
                "micro_op": micro_op,
                "instruction": f"0x{instruction:04X}",
                "opcode": opcode_name,
                "state_hamming_weight": state_hamming_weight,
                "pi_digits_emitted": len(self.emitted_digits),
                "fields": {
                    "p1": p1,
                    "p2": p2,
                    "p3": p3,
                    "immediate": immediate,
                    "offset": offset,
                },
                "before": {
                    "registers": before_registers,
                    "flags": before_flags,
                },
                "predicted": predicted,
                "expected": expected,
                "differing_bits": [
                    {
                        "index": index,
                        "name": output_bit_name(index),
                        "predicted": predicted_bits[index],
                        "expected": exact_bits[index],
                        "logit": logits[index],
                        "probability": probabilities[index],
                    }
                    for index in differing
                ],
            }

            self.neural_error_operations += 1
            self.neural_bit_errors += len(differing)
            self.error_by_opcode[opcode_name] += 1
            self.error_by_pc[str(pc)] += 1
            self.error_by_micro_op[micro_op or "instruction"] += 1
            self.error_by_state_hamming_weight[str(state_hamming_weight)] += 1
            for index in differing:
                self.error_by_output_bit[output_bit_name(index)] += 1
            self._append_error_event(event)

            if self.neural_error_operations == 1:
                payload = self._summary("first_neural_error")
                payload["error"] = event
                self._save_json("first_neural_error.json", payload)

            should_print = (
                self.neural_error_operations <= Config.PRINT_FIRST_ERRORS
                or self.neural_error_operations % Config.PRINT_EVERY_ERRORS == 0
            )
            if should_print:
                names = ", ".join(
                    item["name"] for item in event["differing_bits"]
                )
                mode = "真值替代后继续" if Config.TEACHER_FORCE_ON_ERROR else "立即停止"
                print(
                    f"\n[神经错误 #{self.neural_error_operations}] "
                    f"operation={self.neural_operations:,}, pc={pc}, "
                    f"opcode={opcode_name}, bits={names}, state_hw={state_hamming_weight}; "
                    f"{mode}。"
                )

            if not Config.TEACHER_FORCE_ON_ERROR:
                raise StopIteration

        self.state.registers = exact_registers
        self.state.flags = exact_flags
        return exact_registers, exact_flags, exact_controls

    def run(self) -> dict[str, Any]:
        print(
            f"开始执行 pi 汇编程序：目标={Config.TARGET_DECIMAL_DIGITS} 位，"
            f"机器指令={len(self.program):,} 条，EXACT_ONLY={Config.EXACT_ONLY}，"
            f"TEACHER_FORCE={Config.TEACHER_FORCE_ON_ERROR}，"
            f"MAX_OPS={Config.MAX_NEURAL_OPERATIONS}"
        )
        try:
            while True:
                if Config.MAX_NEURAL_OPERATIONS is not None:
                    if self.neural_operations >= Config.MAX_NEURAL_OPERATIONS:
                        status = "audit_operation_limit"
                        path = self._save_audit_summary(status)
                        self._print_audit_summary(status, path)
                        result = self._summary(status)
                        return result
                if not 0 <= self.state.pc < len(self.program):
                    raise RuntimeError(f"PC 越界：{self.state.pc}")

                pc = self.state.pc
                instruction = self.program[pc]
                opcode, p1, p2, p3, immediate, offset = instruction_fields(instruction)
                before_registers = self.state.registers.copy()
                self.fetched_instructions += 1
                self.last_instruction = self.listing[pc]

                _, _, controls = self._execute_neural_operation(instruction, pc)

                if controls[CTRL_MEMORY_WRITE]:
                    address = self._input_address(before_registers, p2, p3)
                    value = before_registers[p1]
                    if address == MemoryMap.OUTPUT_PORT:
                        self._accept_output_digit(value)
                    else:
                        self.ram[address] = value

                if controls[CTRL_MEMORY_READ]:
                    address = self._input_address(before_registers, p2, p3)
                    value = self.ram[address]
                    # LOAD 的第二拍也必须通过同一个神经核心，不能由 Python 直接写寄存器。
                    movi = (OPCODES["MOVI"] << 11) | ((p1 & 3) << 9) | value
                    self._execute_neural_operation(movi, pc, micro_op="LOAD->MOVI")

                if controls[CTRL_HALT]:
                    result = self._summary("completed")
                    path = self._save_audit_summary("completed")
                    print("\n=== pi 程序正常结束 ===")
                    print(
                        f"pi ~= {result['pi_decimal']}\n"
                        f"有效数字={result['valid_significant_digits']}，"
                        f"CPU操作={result['neural_operations']:,}，"
                        f"速度={result['operations_per_second']:,.1f} op/s"
                    )
                    print(f"汇总：{path}")
                    self._print_audit_summary("completed", path)
                    return result

                if controls[CTRL_BRANCH_TAKEN]:
                    if opcode == OPCODES["JMPR"]:
                        self.state.pc = before_registers[p1]
                    else:
                        self.state.pc = (pc + 1 + offset) & 0xFFFF
                else:
                    self.state.pc = (pc + 1) & 0xFFFF

                if (
                    self.neural_operations - self.last_progress_operation
                    >= Config.PROGRESS_EVERY_OPERATIONS
                ):
                    self.last_progress_operation = self.neural_operations
                    elapsed = time.perf_counter() - self.started_at
                    print(
                        f"[运行] CPU操作={self.neural_operations:,} | "
                        f"取指={self.fetched_instructions:,} | pc={self.state.pc} | "
                        f"速度={self.neural_operations / max(elapsed, 1e-9):,.1f} op/s | "
                        f"已输出={len(self.emitted_digits)} 位 | "
                        f"累计错误操作={self.neural_error_operations:,}"
                    )
                    self._save_audit_summary("running")

        except StopIteration:
            result = self._summary("first_neural_error")
            path = self._save_audit_summary("first_neural_error")
            self._print_audit_summary("first_neural_error", path)
            return result
        except KeyboardInterrupt:
            result = self._summary("interrupted")
            path = self._save_audit_summary("interrupted")
            print(f"\n收到中断，进度已保存：{path}")
            self._print_audit_summary("interrupted", path)
            return result


class BatchedNeuralPiAuditor:
    """在精确轨迹上大批量审计神经核，并严格定位第一处错误。"""

    def __init__(
        self,
        program: list[int],
        listing: list[dict[str, Any]],
        runner: CoreRunner,
        reference_digits: str,
    ):
        self.program = program
        self.listing = listing
        self.runner = runner
        self.reference_digits = reference_digits
        self.generator = NativeTraceGenerator(program)
        self.neural_operations = 0
        self.emitted_digits = ""
        self.ignored_guard_digit = False
        self.started_at = time.perf_counter()
        self.trace_seconds = 0.0
        self.gpu_seconds = 0.0
        self.last_progress_operation = 0
        self.current_pc = 0
        self.current_registers = [0, 0, 0, 0]
        self.current_flags = [0, 0, 0]
        Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self.error_log_path = Config.RESULT_DIR / "neural_errors.jsonl"
        if self.error_log_path.exists():
            self.error_log_path.unlink()

    @staticmethod
    def _decode_input_state(row: np.ndarray) -> tuple[list[int], list[int]]:
        flags = row[INSTRUCTION_BITS : INSTRUCTION_BITS + NUM_FLAGS].astype(int).tolist()
        registers: list[int] = []
        cursor = INSTRUCTION_BITS + NUM_FLAGS
        for _ in range(NUM_REGISTERS):
            value = 0
            for bit in row[cursor : cursor + REGISTER_BITS]:
                value = (value << 1) | int(bit)
            registers.append(value)
            cursor += REGISTER_BITS
        return registers, flags

    def _save_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = Config.RESULT_DIR / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def _summary(self, status: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started_at
        valid = 0
        for observed, expected in zip(self.emitted_digits, self.reference_digits):
            if observed != expected:
                break
            valid += 1
        return {
            "status": status,
            "audit_mode": "batched_exact_trajectory",
            "model_path": str(self.runner.model_path or Config.MODEL_PATH),
            "target_decimal_digits": Config.TARGET_DECIMAL_DIGITS,
            "trace_batch": Config.TRACE_BATCH,
            "inference_batch": Config.INFERENCE_BATCH,
            "neural_operations": self.neural_operations,
            "fetched_instructions": int(self.generator.fetched[0]),
            "elapsed_seconds": elapsed,
            "operations_per_second": self.neural_operations / max(elapsed, 1e-9),
            "trace_generation_seconds": self.trace_seconds,
            "gpu_audit_seconds": self.gpu_seconds,
            "pc": self.current_pc,
            "registers": self.current_registers,
            "flags": self.current_flags,
            "pi_digits": self.emitted_digits,
            "pi_decimal": format_pi(self.emitted_digits),
            "valid_significant_digits": valid,
            "valid_decimal_places": max(valid - 1, 0),
            "induction_note": (
                "每条记录来自同一精确轨迹；第一处 batch 差异等于自主执行的第一处差异。"
            ),
        }

    def _save_summary(self, status: str) -> Path:
        summary = self._summary(status)
        self._save_json("summary.json", summary)
        return self._save_json("audit_summary.json", summary)

    def _print_pi_progress(self) -> None:
        count = len(self.emitted_digits)
        if count % Config.PRINT_EVERY_DIGITS != 0:
            return
        summary = self._summary("running")
        print(
            f"[pi] 已验证操作={self.neural_operations:,} | 输出={count} 位 | "
            f"有效={summary['valid_significant_digits']} 位 "
            f"(小数点后 {summary['valid_decimal_places']} 位)\n"
            f"     pi ~= {summary['pi_decimal']}"
        )
        if count % Config.SAVE_PROGRESS_EVERY_DIGITS == 0:
            self._save_json("progress.json", summary)

    def _accept_output_digit(self, value: int) -> None:
        if not 0 <= value <= 9:
            raise RuntimeError(f"pi 程序向输出端口写入了非十进制数字：{value}")
        if not self.ignored_guard_digit:
            if value != 0:
                raise RuntimeError(f"首个 guard predigit 应为 0，实际为 {value}")
            self.ignored_guard_digit = True
            return
        self.emitted_digits += str(value)
        expected_prefix = self.reference_digits[: len(self.emitted_digits)]
        if self.emitted_digits != expected_prefix:
            payload = self._summary("pi_program_mismatch")
            payload["expected_prefix"] = expected_prefix
            path = self._save_json("pi_program_mismatch.json", payload)
            raise RuntimeError(f"精确 pi 程序输出与参考值不符：{path}")
        self._print_pi_progress()

    def _build_error_event(
        self,
        chunk: dict[str, Any],
        mismatch: dict[str, Any],
        base_operation: int,
    ) -> dict[str, Any]:
        index = int(mismatch["index"])
        instruction = int(chunk["instructions"][index])
        pc = int(chunk["pcs"][index])
        micro_op = int(chunk["micro_ops"][index])
        registers, flags = self._decode_input_state(chunk["inputs"][index])
        opcode, p1, p2, p3, immediate, offset = instruction_fields(instruction)
        predicted_bits = mismatch["predicted_bits"]
        expected_bits = mismatch["expected_bits"]
        differing = [
            bit
            for bit, (predicted, expected) in enumerate(
                zip(predicted_bits, expected_bits)
            )
            if predicted != expected
        ]
        return {
            "operation_index": base_operation + index + 1,
            "fetched_instruction_index": int(chunk["fetch_indices"][index]),
            "pc": pc,
            "micro_op": "LOAD->MOVI" if micro_op else None,
            "instruction": f"0x{instruction:04X}",
            "opcode": OPCODE_NAMES.get(opcode, str(opcode)),
            "state_hamming_weight": sum(flags)
            + sum(value.bit_count() for value in registers),
            "pi_digits_emitted": len(self.emitted_digits),
            "fields": {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "immediate": immediate,
                "offset": offset,
            },
            "before": {"registers": registers, "flags": flags},
            "predicted": decode_output_bits(predicted_bits),
            "expected": decode_output_bits(expected_bits),
            "differing_bits": [
                {
                    "index": bit,
                    "name": output_bit_name(bit),
                    "predicted": predicted_bits[bit],
                    "expected": expected_bits[bit],
                    "logit": mismatch["logits"][bit],
                    "probability": mismatch["probabilities"][bit],
                }
                for bit in differing
            ],
        }

    def _process_verified_outputs(
        self,
        chunk: dict[str, Any],
        verified_record_count: int,
        base_operation: int,
    ) -> None:
        for relative_index, digit in zip(
            chunk["output_indices"], chunk["output_digits"]
        ):
            index = int(relative_index)
            if index >= verified_record_count:
                break
            self.neural_operations = base_operation + index + 1
            self._accept_output_digit(int(digit))

    def _print_progress(self) -> None:
        if (
            self.neural_operations - self.last_progress_operation
            < Config.PROGRESS_EVERY_OPERATIONS
        ):
            return
        self.last_progress_operation = self.neural_operations
        elapsed = time.perf_counter() - self.started_at
        print(
            f"[batch 审计] 操作={self.neural_operations:,} | "
            f"取指={int(self.generator.fetched[0]):,} | pc={self.current_pc} | "
            f"总速度={self.neural_operations / max(elapsed, 1e-9):,.1f} op/s | "
            f"轨迹生成={self.trace_seconds:.1f}s | GPU={self.gpu_seconds:.1f}s | "
            f"已输出={len(self.emitted_digits)} 位 | 错误=0"
        )
        self._save_summary("running")

    def run(self) -> dict[str, Any]:
        print(
            f"开始 batch 轨迹审计：目标={Config.TARGET_DECIMAL_DIGITS} 位，"
            f"机器指令={len(self.program):,} 条，TRACE_BATCH={Config.TRACE_BATCH:,}，"
            f"INFERENCE_BATCH={Config.INFERENCE_BATCH:,}，MAX_OPS={Config.MAX_NEURAL_OPERATIONS}"
        )
        print(
            "判定规则：若前 k-1 个真值状态上的预测均正确，则第 k 个 batch 差异"
            "就是自主执行的第一处差异。"
        )
        try:
            while True:
                if Config.MAX_NEURAL_OPERATIONS is None:
                    capacity = Config.TRACE_BATCH
                else:
                    remaining = Config.MAX_NEURAL_OPERATIONS - self.neural_operations
                    if remaining <= 0:
                        path = self._save_summary("audit_operation_limit")
                        print(f"达到操作上限，结果已保存：{path}")
                        return self._summary("audit_operation_limit")
                    capacity = min(Config.TRACE_BATCH, remaining)

                trace_started = time.perf_counter()
                chunk = self.generator.next_chunk(capacity)
                self.trace_seconds += time.perf_counter() - trace_started
                count = int(chunk["inputs"].shape[0])
                if count == 0 and not chunk["halted"]:
                    raise RuntimeError("轨迹生成器未前进；请确保 TRACE_BATCH >= 2。")

                base_operation = self.neural_operations
                gpu_started = time.perf_counter()
                mismatch = self.runner.find_first_batch_mismatch(
                    chunk["inputs"], chunk["targets"]
                )
                self.gpu_seconds += time.perf_counter() - gpu_started

                verified_count = count if mismatch is None else int(mismatch["index"])
                self._process_verified_outputs(
                    chunk, verified_count, base_operation
                )

                if mismatch is not None:
                    event = self._build_error_event(chunk, mismatch, base_operation)
                    self.neural_operations = int(event["operation_index"])
                    self.current_pc = int(event["pc"])
                    self.current_registers = event["before"]["registers"]
                    self.current_flags = event["before"]["flags"]
                    with self.error_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    payload = self._summary("first_neural_error")
                    payload["error"] = event
                    first_path = self._save_json("first_neural_error.json", payload)
                    summary_path = self._save_summary("first_neural_error")
                    names = ", ".join(
                        item["name"] for item in event["differing_bits"]
                    )
                    print(
                        f"\n[第一处神经错误] operation={self.neural_operations:,} | "
                        f"pc={event['pc']} | opcode={event['opcode']} | bits={names}"
                    )
                    print(f"错误现场：{first_path}")
                    print(f"汇总：{summary_path}")
                    return payload

                self.neural_operations = base_operation + count
                self.current_pc = int(chunk["pc"])
                self.current_registers = chunk["registers"]
                self.current_flags = chunk["flags"]
                self._print_progress()

                if chunk["halted"]:
                    status = "completed"
                    summary = self._summary(status)
                    if len(self.emitted_digits) != Config.TARGET_DECIMAL_DIGITS:
                        raise RuntimeError(
                            f"程序已 HALT，但只输出 {len(self.emitted_digits)} / "
                            f"{Config.TARGET_DECIMAL_DIGITS} 位。"
                        )
                    path = self._save_summary(status)
                    print("\n=== batch 轨迹审计完成：全程零错误 ===")
                    print(
                        f"pi ~= {summary['pi_decimal']}\n"
                        f"有效数字={summary['valid_significant_digits']}，"
                        f"神经操作={self.neural_operations:,}，"
                        f"总速度={summary['operations_per_second']:,.1f} op/s"
                    )
                    print(f"汇总：{path}")
                    return summary
        except KeyboardInterrupt:
            path = self._save_summary("interrupted")
            print(f"\n收到中断，batch 审计进度已保存：{path}")
            return self._summary("interrupted")


class PrimeMemoryMap:
    """64-KiB RAM 布局；所有多字节整数均为 little-endian。"""

    SEGMENT_BUFFER = 0x0000
    SEGMENT_SIZE = 0x8000
    BASE_FLAGS = 0x8000
    RECORD_BASE = 0xA000
    RECORD_SIZE = 8
    RECORD_LIMIT = 0xF000

    P = 0xF000                 # uint16，当前基础筛候选
    P_SQUARE = 0xF002          # uint32，当前候选平方
    BASE_INDEX = 0xF006        # uint16，(p - 3) / 2
    RECORD_POINTER = 0xF008    # uint16，基础质数记录写指针
    PRIME_COUNT = 0xF00A       # uint16，奇基础质数个数
    ACTIVE_COUNT = 0xF00C      # uint16，当前已激活记录个数
    SEGMENT = 0xF00E           # uint16，当前 65536 整数分段编号
    NEXT_ACTIVE_POINTER = 0xF010
    ACTIVE_POINTER = 0xF012
    ACTIVE_REMAIN = 0xF014
    MARK_OFFSET = 0xF016       # uint16，分段缓冲区中的奇数索引
    CURRENT_P = 0xF018
    SCAN_INDEX = 0xF01A
    PRIME_RANK = 0xF01C        # uint24，已发现质数个数
    RESULT = 0xF01F            # uint32，最终质数
    BASE_MARK_INDEX = 0xF023
    TEMP16 = 0xF025


def read_ram_uint(ram: Any, address: int, size: int) -> int:
    return sum(int(ram[address + index]) << (8 * index) for index in range(size))


class PrimeAssemblyBuilder:
    """生成完全由 Neural CPU 执行的分段奇数筛。"""

    def __init__(self, target_rank: int, search_limit: int):
        self.target_rank = target_rank
        self.search_limit = search_limit
        self.base_limit = math.isqrt(search_limit)
        self.base_odd_limit = self.base_limit if self.base_limit & 1 else self.base_limit - 1
        self.base_odd_count = max(0, (self.base_odd_limit - 3) // 2 + 1)
        self.max_segment = search_limit >> 16
        self.asm = Assembler()

    def load_abs(self, dst: int, address: int) -> None:
        self.asm.emit("MOVI", rd=2, imm=(address >> 8) & 0xFF)
        self.asm.emit("MOVI", rd=3, imm=address & 0xFF)
        self.asm.emit("LOAD", rd=dst, ra=2, rb=3)

    def store_abs(self, src: int, address: int) -> None:
        self.asm.emit("MOVI", rd=2, imm=(address >> 8) & 0xFF)
        self.asm.emit("MOVI", rd=3, imm=address & 0xFF)
        self.asm.emit("STORE", rd=src, ra=2, rb=3)

    def write_u8(self, address: int, value: int) -> None:
        self.asm.emit("MOVI", rd=0, imm=value)
        self.store_abs(0, address)

    def write_uint(self, address: int, value: int, size: int) -> None:
        for byte in range(size):
            self.write_u8(address + byte, (value >> (8 * byte)) & 0xFF)

    def copy_uint(self, source: int, destination: int, size: int) -> None:
        for byte in range(size):
            self.load_abs(0, source + byte)
            self.store_abs(0, destination + byte)

    def increment_u16(self, address: int, amount: int = 1) -> None:
        done = self.asm.unique("inc16_done")
        self.load_abs(0, address)
        self.asm.emit("ADDI", rd=0, imm=amount)
        self.store_abs(0, address)
        self.asm.branch("JNC", done)
        self.load_abs(0, address + 1)
        self.asm.emit("ADDI", rd=0, imm=1)
        self.store_abs(0, address + 1)
        self.asm.label(done)

    def decrement_u16(self, address: int) -> None:
        done = self.asm.unique("dec16_done")
        self.load_abs(0, address)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, address)
        self.asm.branch("JNC", done)
        self.load_abs(0, address + 1)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, address + 1)
        self.asm.label(done)

    def subtract_u16_constant(self, address: int, amount: int) -> None:
        done = self.asm.unique("sub16_const_done")
        self.load_abs(0, address)
        self.asm.emit("SUBI", rd=0, imm=amount)
        self.store_abs(0, address)
        self.asm.branch("JNC", done)
        self.load_abs(0, address + 1)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, address + 1)
        self.asm.label(done)

    def branch_if_u16_zero(self, address: int, target: str) -> None:
        nonzero = self.asm.unique("u16_nonzero")
        self.load_abs(0, address + 1)
        self.asm.emit("CMPI", rd=0, imm=0)
        self.asm.branch("JNZ", nonzero)
        self.load_abs(0, address)
        self.asm.emit("CMPI", rd=0, imm=0)
        self.asm.branch("JZ", target)
        self.asm.label(nonzero)

    def compare_uint_constant(
        self,
        address: int,
        value: int,
        size: int,
        less: str,
        greater: str,
        equal: str,
    ) -> None:
        for byte in reversed(range(size)):
            self.load_abs(0, address + byte)
            self.asm.emit("CMPI", rd=0, imm=(value >> (8 * byte)) & 0xFF)
            self.asm.branch("JL", less)
            self.asm.branch("JG", greater)
        self.asm.branch("JMP", equal)

    def compare_u16(
        self, left: int, right: int, less: str, greater: str, equal: str
    ) -> None:
        for byte in (1, 0):
            self.load_abs(0, left + byte)
            self.load_abs(1, right + byte)
            self.asm.emit("CMP", ra=0, rb=1)
            self.asm.branch("JL", less)
            self.asm.branch("JG", greater)
        self.asm.branch("JMP", equal)

    def increment_u24(self, address: int) -> None:
        done = self.asm.unique("inc24_done")
        for byte in range(3):
            self.load_abs(0, address + byte)
            self.asm.emit("ADDI", rd=0, imm=1)
            self.store_abs(0, address + byte)
            self.asm.branch("JNC", done)
        self.asm.label(done)

    def add_u16_to_u32(self, destination: int, source: int) -> None:
        self.load_abs(0, destination)
        self.load_abs(1, source)
        self.asm.emit("ADD", rd=0, ra=0, rb=1)
        self.store_abs(0, destination)
        self.load_abs(0, destination + 1)
        self.load_abs(1, source + 1)
        self.asm.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, destination + 1)
        for byte in (2, 3):
            self.load_abs(0, destination + byte)
            self.asm.emit("MOVI", rd=1, imm=0)
            self.asm.emit("ADC", rd=0, ra=0, rb=1)
            self.store_abs(0, destination + byte)

    def add_constant_u32(self, address: int, amount: int) -> None:
        done = self.asm.unique("add32_const_done")
        self.load_abs(0, address)
        self.asm.emit("ADDI", rd=0, imm=amount)
        self.store_abs(0, address)
        self.asm.branch("JNC", done)
        for byte in (1, 2, 3):
            self.load_abs(0, address + byte)
            self.asm.emit("ADDI", rd=0, imm=1)
            self.store_abs(0, address + byte)
            self.asm.branch("JNC", done)
        self.asm.label(done)

    def shift_right_u16_one(self, address: int) -> None:
        self.load_abs(0, address)
        self.load_abs(1, address + 1)
        self.asm.emit("MOV", rd=3, ra=1)
        self.asm.emit("MOVI", rd=2, imm=1)
        self.asm.emit("SHR", rd=0, ra=0, rb=2)
        self.asm.emit("AND", rd=3, ra=3, rb=2)
        self.asm.emit("MOVI", rd=2, imm=7)
        self.asm.emit("SHL", rd=3, ra=3, rb=2)
        self.asm.emit("OR", rd=0, ra=0, rb=3)
        self.asm.emit("MOVI", rd=2, imm=1)
        self.asm.emit("SHR", rd=1, ra=1, rb=2)
        self.store_abs(0, address)
        self.store_abs(1, address + 1)

    def setup_indirect_address(self, pointer: int, offset: int = 0) -> None:
        no_carry = self.asm.unique("ptr_no_carry")
        self.load_abs(1, pointer + 1)
        self.load_abs(3, pointer)
        self.asm.emit("MOV", rd=2, ra=1)
        if offset:
            self.asm.emit("ADDI", rd=3, imm=offset)
            self.asm.branch("JNC", no_carry)
            self.asm.emit("ADDI", rd=2, imm=1)
            self.asm.label(no_carry)

    def load_indirect_to_memory(self, pointer: int, offset: int, destination: int) -> None:
        self.setup_indirect_address(pointer, offset)
        self.asm.emit("LOAD", rd=0, ra=2, rb=3)
        self.store_abs(0, destination)

    def store_memory_indirect(self, source: int, pointer: int, offset: int) -> None:
        self.load_abs(0, source)
        self.setup_indirect_address(pointer, offset)
        self.asm.emit("STORE", rd=0, ra=2, rb=3)

    def load_record_u16(self, pointer: int, offset: int, destination: int) -> None:
        self.load_indirect_to_memory(pointer, offset, destination)
        self.load_indirect_to_memory(pointer, offset + 1, destination + 1)

    def store_record_u16(self, source: int, pointer: int, offset: int) -> None:
        self.store_memory_indirect(source, pointer, offset)
        self.store_memory_indirect(source + 1, pointer, offset + 1)

    def append_prime_record(self) -> None:
        self.store_record_u16(PrimeMemoryMap.P, PrimeMemoryMap.RECORD_POINTER, 0)
        for byte in range(4):
            self.store_memory_indirect(
                PrimeMemoryMap.P_SQUARE + byte,
                PrimeMemoryMap.RECORD_POINTER,
                2 + byte,
            )
        self.write_uint(PrimeMemoryMap.TEMP16, 0, 2)
        self.store_record_u16(
            PrimeMemoryMap.TEMP16, PrimeMemoryMap.RECORD_POINTER, 6
        )
        self.increment_u16(PrimeMemoryMap.RECORD_POINTER, PrimeMemoryMap.RECORD_SIZE)
        self.increment_u16(PrimeMemoryMap.PRIME_COUNT)

    def load_base_flag(self) -> None:
        self.load_abs(1, PrimeMemoryMap.BASE_INDEX + 1)
        self.load_abs(3, PrimeMemoryMap.BASE_INDEX)
        self.asm.emit("MOV", rd=2, ra=1)
        self.asm.emit("ADDI", rd=2, imm=0x80)
        self.asm.emit("LOAD", rd=0, ra=2, rb=3)

    def mark_base_flag(self) -> None:
        self.load_abs(1, PrimeMemoryMap.BASE_MARK_INDEX + 1)
        self.load_abs(3, PrimeMemoryMap.BASE_MARK_INDEX)
        self.asm.emit("MOV", rd=2, ra=1)
        self.asm.emit("ADDI", rd=2, imm=0x80)
        self.asm.emit("MOVI", rd=0, imm=1)
        self.asm.emit("STORE", rd=0, ra=2, rb=3)

    def save_scan_registers(self) -> None:
        self.asm.emit("MOV", rd=0, ra=3)
        self.asm.emit("MOV", rd=1, ra=2)
        self.store_abs(0, PrimeMemoryMap.SCAN_INDEX)
        self.store_abs(1, PrimeMemoryMap.SCAN_INDEX + 1)

    def restore_scan_registers(self) -> None:
        self.load_abs(1, PrimeMemoryMap.SCAN_INDEX + 1)
        self.load_abs(0, PrimeMemoryMap.SCAN_INDEX)
        self.asm.emit("MOV", rd=3, ra=0)
        self.asm.emit("MOV", rd=2, ra=1)

    def emit_found_result(self) -> None:
        # 当前奇数低 16 bit = 2 * scan_index + 1。
        self.load_abs(0, PrimeMemoryMap.SCAN_INDEX)
        self.load_abs(1, PrimeMemoryMap.SCAN_INDEX + 1)
        self.asm.emit("ADD", rd=0, ra=0, rb=0)
        self.asm.emit("ADC", rd=1, ra=1, rb=1)
        self.asm.emit("ADDI", rd=0, imm=1)
        self.asm.emit("MOVI", rd=2, imm=0)
        self.asm.emit("ADC", rd=1, ra=1, rb=2)
        self.store_abs(0, PrimeMemoryMap.RESULT)
        self.store_abs(1, PrimeMemoryMap.RESULT + 1)
        self.load_abs(0, PrimeMemoryMap.SEGMENT)
        self.store_abs(0, PrimeMemoryMap.RESULT + 2)
        self.load_abs(0, PrimeMemoryMap.SEGMENT + 1)
        self.store_abs(0, PrimeMemoryMap.RESULT + 3)
        self.asm.emit("HALT")

    def build(self) -> tuple[list[int], list[dict[str, Any]]]:
        if self.target_rank < 1 or self.target_rank >= 1 << 24:
            raise ValueError("当前程序的目标序号必须在 1..16,777,215。")
        if self.search_limit < 3 or self.search_limit >= 1 << 32:
            raise ValueError("搜索上界必须位于 3..2^32-1。")
        if self.base_odd_count > PrimeMemoryMap.RECORD_BASE - PrimeMemoryMap.BASE_FLAGS:
            raise ValueError("基础筛超出 RAM 预留区域。")
        record_capacity = (
            PrimeMemoryMap.RECORD_LIMIT - PrimeMemoryMap.RECORD_BASE
        ) // PrimeMemoryMap.RECORD_SIZE
        if self.base_odd_count > record_capacity * 6:
            raise ValueError("搜索上界太大，基础质数记录可能超出 RAM。")

        a = self.asm
        if self.target_rank == 1:
            self.write_uint(PrimeMemoryMap.PRIME_RANK, 1, 3)
            self.write_uint(PrimeMemoryMap.RESULT, 2, 4)
            a.emit("HALT")
            return a.assemble()

        # 第一阶段：CPU 自行筛出 sqrt(search_limit) 内的基础质数。
        self.write_uint(PrimeMemoryMap.P, 3, 2)
        self.write_uint(PrimeMemoryMap.P_SQUARE, 9, 4)
        self.write_uint(PrimeMemoryMap.BASE_INDEX, 0, 2)
        self.write_uint(PrimeMemoryMap.RECORD_POINTER, PrimeMemoryMap.RECORD_BASE, 2)
        self.write_uint(PrimeMemoryMap.PRIME_COUNT, 0, 2)

        a.label("base_candidate_loop")
        self.compare_uint_constant(
            PrimeMemoryMap.P,
            self.base_odd_limit,
            2,
            "base_candidate_body",
            "base_generation_done",
            "base_candidate_body",
        )
        a.label("base_candidate_body")
        self.load_base_flag()
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JNZ", "base_candidate_advance")
        self.append_prime_record()

        self.compare_uint_constant(
            PrimeMemoryMap.P_SQUARE,
            self.base_odd_limit,
            4,
            "base_prepare_mark",
            "base_candidate_advance",
            "base_prepare_mark",
        )
        a.label("base_prepare_mark")
        self.copy_uint(PrimeMemoryMap.P_SQUARE, PrimeMemoryMap.BASE_MARK_INDEX, 2)
        self.subtract_u16_constant(PrimeMemoryMap.BASE_MARK_INDEX, 3)
        self.shift_right_u16_one(PrimeMemoryMap.BASE_MARK_INDEX)

        a.label("base_mark_loop")
        self.compare_uint_constant(
            PrimeMemoryMap.BASE_MARK_INDEX,
            self.base_odd_count,
            2,
            "base_mark_body",
            "base_candidate_advance",
            "base_candidate_advance",
        )
        a.label("base_mark_body")
        self.mark_base_flag()
        # mark_index += p
        self.load_abs(0, PrimeMemoryMap.BASE_MARK_INDEX)
        self.load_abs(1, PrimeMemoryMap.P)
        a.emit("ADD", rd=0, ra=0, rb=1)
        self.store_abs(0, PrimeMemoryMap.BASE_MARK_INDEX)
        self.load_abs(0, PrimeMemoryMap.BASE_MARK_INDEX + 1)
        self.load_abs(1, PrimeMemoryMap.P + 1)
        a.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, PrimeMemoryMap.BASE_MARK_INDEX + 1)
        a.branch("JMP", "base_mark_loop")

        a.label("base_candidate_advance")
        # (p + 2)^2 = p^2 + 4p + 4。
        for _ in range(4):
            self.add_u16_to_u32(PrimeMemoryMap.P_SQUARE, PrimeMemoryMap.P)
        self.add_constant_u32(PrimeMemoryMap.P_SQUARE, 4)
        self.increment_u16(PrimeMemoryMap.P, 2)
        self.increment_u16(PrimeMemoryMap.BASE_INDEX)
        a.branch("JMP", "base_candidate_loop")

        # 第二阶段：65536 个整数为一段，缓冲区只保存其中 32768 个奇数。
        a.label("base_generation_done")
        self.write_uint(PrimeMemoryMap.ACTIVE_COUNT, 0, 2)
        self.write_uint(PrimeMemoryMap.SEGMENT, 0, 2)
        self.write_uint(
            PrimeMemoryMap.NEXT_ACTIVE_POINTER, PrimeMemoryMap.RECORD_BASE, 2
        )
        self.write_uint(PrimeMemoryMap.PRIME_RANK, 1, 3)  # 已计入质数 2

        a.label("segment_loop")
        # 清空当前段；R2:R3 本身就是 CPU 计算出的 RAM 地址。
        a.emit("MOVI", rd=2, imm=0)
        a.emit("MOVI", rd=3, imm=0)
        a.emit("MOVI", rd=0, imm=0)
        a.label("clear_segment_loop")
        a.emit("STORE", rd=0, ra=2, rb=3)
        a.emit("ADDI", rd=3, imm=1)
        clear_no_carry = a.unique("clear_no_carry")
        a.branch("JNC", clear_no_carry)
        a.emit("ADDI", rd=2, imm=1)
        a.label(clear_no_carry)
        a.emit("CMPI", rd=2, imm=0x80)
        a.branch("JL", "clear_segment_loop")

        # 数字 1 不是质数；只需在第 0 段标记一次。
        segment_nonzero = a.unique("segment_nonzero")
        self.branch_if_u16_zero(PrimeMemoryMap.SEGMENT, "mark_number_one")
        a.branch("JMP", segment_nonzero)
        a.label("mark_number_one")
        a.emit("MOVI", rd=0, imm=1)
        a.emit("MOVI", rd=2, imm=0)
        a.emit("MOVI", rd=3, imm=0)
        a.emit("STORE", rd=0, ra=2, rb=3)
        a.label(segment_nonzero)

        # 当 p^2 进入当前段时激活记录，并由 CPU 计算首次局部地址。
        a.label("activate_loop")
        self.compare_u16(
            PrimeMemoryMap.ACTIVE_COUNT,
            PrimeMemoryMap.PRIME_COUNT,
            "activate_check_square",
            "fatal_trap",
            "activate_done",
        )
        a.label("activate_check_square")
        self.load_record_u16(
            PrimeMemoryMap.NEXT_ACTIVE_POINTER, 4, PrimeMemoryMap.TEMP16
        )
        self.compare_u16(
            PrimeMemoryMap.TEMP16,
            PrimeMemoryMap.SEGMENT,
            "fatal_trap",
            "activate_done",
            "activate_record",
        )
        a.label("activate_record")
        self.load_record_u16(
            PrimeMemoryMap.NEXT_ACTIVE_POINTER, 2, PrimeMemoryMap.MARK_OFFSET
        )
        self.subtract_u16_constant(PrimeMemoryMap.MARK_OFFSET, 1)
        self.shift_right_u16_one(PrimeMemoryMap.MARK_OFFSET)
        self.store_record_u16(
            PrimeMemoryMap.MARK_OFFSET, PrimeMemoryMap.NEXT_ACTIVE_POINTER, 6
        )
        self.increment_u16(PrimeMemoryMap.ACTIVE_COUNT)
        self.increment_u16(
            PrimeMemoryMap.NEXT_ACTIVE_POINTER, PrimeMemoryMap.RECORD_SIZE
        )
        a.branch("JMP", "activate_loop")

        a.label("activate_done")
        self.write_uint(PrimeMemoryMap.ACTIVE_POINTER, PrimeMemoryMap.RECORD_BASE, 2)
        self.copy_uint(
            PrimeMemoryMap.ACTIVE_COUNT, PrimeMemoryMap.ACTIVE_REMAIN, 2
        )

        a.label("active_prime_loop")
        self.branch_if_u16_zero(PrimeMemoryMap.ACTIVE_REMAIN, "scan_segment_setup")
        self.load_record_u16(
            PrimeMemoryMap.ACTIVE_POINTER, 0, PrimeMemoryMap.CURRENT_P
        )
        self.load_record_u16(
            PrimeMemoryMap.ACTIVE_POINTER, 6, PrimeMemoryMap.MARK_OFFSET
        )

        # 小于 256 的筛质数承担绝大多数标记工作。对它们采用寄存器常驻热循环：
        # R0=p，R1=0，R2:R3=局部筛地址。STORE 写入任意非零值都表示合数，
        # 因而可直接写 R0；每个标记只需 STORE/ADD/ADC/CMPI/JL 五条指令。
        self.load_abs(0, PrimeMemoryMap.CURRENT_P + 1)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JZ", "segment_mark_fast_setup")

        a.label("segment_mark_slow_loop")
        self.load_abs(0, PrimeMemoryMap.MARK_OFFSET + 1)
        a.emit("CMPI", rd=0, imm=0x80)
        a.branch("JL", "segment_mark_slow_body")
        # 首个越界位置减去 32768，就是下一段中的首次位置。
        a.emit("SUBI", rd=0, imm=0x80)
        self.store_abs(0, PrimeMemoryMap.MARK_OFFSET + 1)
        a.branch("JMP", "segment_mark_finished")

        a.label("segment_mark_slow_body")
        self.load_abs(1, PrimeMemoryMap.MARK_OFFSET + 1)
        self.load_abs(3, PrimeMemoryMap.MARK_OFFSET)
        a.emit("MOV", rd=2, ra=1)
        a.emit("MOVI", rd=0, imm=1)
        a.emit("STORE", rd=0, ra=2, rb=3)
        # 局部地址 += p；地址的产生完全发生在 CPU 中。
        self.load_abs(0, PrimeMemoryMap.MARK_OFFSET)
        self.load_abs(1, PrimeMemoryMap.CURRENT_P)
        a.emit("ADD", rd=0, ra=0, rb=1)
        self.store_abs(0, PrimeMemoryMap.MARK_OFFSET)
        self.load_abs(0, PrimeMemoryMap.MARK_OFFSET + 1)
        self.load_abs(1, PrimeMemoryMap.CURRENT_P + 1)
        a.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, PrimeMemoryMap.MARK_OFFSET + 1)
        a.branch("JMP", "segment_mark_slow_loop")

        a.label("segment_mark_fast_setup")
        self.load_abs(0, PrimeMemoryMap.CURRENT_P)
        self.load_abs(1, PrimeMemoryMap.MARK_OFFSET + 1)
        self.load_abs(3, PrimeMemoryMap.MARK_OFFSET)
        a.emit("MOV", rd=2, ra=1)
        a.emit("MOVI", rd=1, imm=0)

        a.label("segment_mark_fast_loop")
        a.emit("STORE", rd=0, ra=2, rb=3)
        a.emit("ADD", rd=3, ra=3, rb=0)
        a.emit("ADC", rd=2, ra=2, rb=1)
        a.emit("CMPI", rd=2, imm=0x80)
        a.branch("JL", "segment_mark_fast_loop")

        # 把首个越界位置折回下一段，再写回该质数的滚动 offset。
        a.emit("MOV", rd=0, ra=3)
        a.emit("MOV", rd=1, ra=2)
        a.emit("SUBI", rd=1, imm=0x80)
        self.store_abs(0, PrimeMemoryMap.MARK_OFFSET)
        self.store_abs(1, PrimeMemoryMap.MARK_OFFSET + 1)

        a.label("segment_mark_finished")
        self.store_record_u16(
            PrimeMemoryMap.MARK_OFFSET, PrimeMemoryMap.ACTIVE_POINTER, 6
        )
        self.increment_u16(PrimeMemoryMap.ACTIVE_POINTER, PrimeMemoryMap.RECORD_SIZE)
        self.decrement_u16(PrimeMemoryMap.ACTIVE_REMAIN)
        a.branch("JMP", "active_prime_loop")

        a.label("scan_segment_setup")
        a.emit("MOVI", rd=2, imm=0)
        a.emit("MOVI", rd=3, imm=0)
        a.label("scan_segment_loop")
        a.emit("CMPI", rd=2, imm=0x80)
        a.branch("JZ", "segment_complete")
        a.emit("LOAD", rd=0, ra=2, rb=3)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JNZ", "scan_next")

        self.save_scan_registers()
        self.increment_u24(PrimeMemoryMap.PRIME_RANK)
        rank_not_target = a.unique("rank_not_target")
        for byte in reversed(range(3)):
            self.load_abs(0, PrimeMemoryMap.PRIME_RANK + byte)
            a.emit(
                "CMPI",
                rd=0,
                imm=(self.target_rank >> (8 * byte)) & 0xFF,
            )
            a.branch("JNZ", rank_not_target)
        self.emit_found_result()
        a.label(rank_not_target)
        self.restore_scan_registers()

        a.label("scan_next")
        a.emit("ADDI", rd=3, imm=1)
        scan_no_carry = a.unique("scan_no_carry")
        a.branch("JNC", scan_no_carry)
        a.emit("ADDI", rd=2, imm=1)
        a.label(scan_no_carry)
        a.branch("JMP", "scan_segment_loop")

        a.label("segment_complete")
        self.increment_u16(PrimeMemoryMap.SEGMENT)
        self.compare_uint_constant(
            PrimeMemoryMap.SEGMENT,
            self.max_segment,
            2,
            "segment_loop",
            "fatal_trap",
            "segment_loop",
        )
        a.label("fatal_trap")
        a.emit("TRAP")
        return a.assemble()


def prime_runtime_state(ram: Any) -> dict[str, int]:
    return {
        "segment": read_ram_uint(ram, PrimeMemoryMap.SEGMENT, 2),
        "prime_rank": read_ram_uint(ram, PrimeMemoryMap.PRIME_RANK, 3),
        "result": read_ram_uint(ram, PrimeMemoryMap.RESULT, 4),
        "base_prime_count": read_ram_uint(ram, PrimeMemoryMap.PRIME_COUNT, 2),
        "active_prime_count": read_ram_uint(ram, PrimeMemoryMap.ACTIVE_COUNT, 2),
    }


class ExactPrimeMachine:
    """用于小规模自检的纯 Python 精确 ISA 执行器。"""

    def __init__(self, program: list[int]):
        self.program = program
        self.ram = bytearray(65_536)
        self.pc = 0
        self.registers = [0, 0, 0, 0]
        self.flags = [0, 0, 0]
        self.operations = 0
        self.fetched = 0

    def execute_operation(self, instruction: int) -> list[int]:
        opcode, p1, p2, p3, immediate, _ = instruction_fields(instruction)
        regs, flags, controls = scalar_reference(
            opcode, p1, p2, p3, immediate, self.registers, self.flags
        )
        self.registers, self.flags = regs, flags
        self.operations += 1
        return controls

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        last_progress = 0
        while True:
            if Config.MAX_NEURAL_OPERATIONS is not None and self.operations >= Config.MAX_NEURAL_OPERATIONS:
                status = "operation_limit"
                break
            if not 0 <= self.pc < len(self.program):
                raise RuntimeError(f"PC 越界：{self.pc}")
            pc = self.pc
            instruction = self.program[pc]
            opcode, p1, p2, p3, _, offset = instruction_fields(instruction)
            before = self.registers.copy()
            self.fetched += 1
            controls = self.execute_operation(instruction)
            if controls[CTRL_MEMORY_WRITE]:
                address = (before[p2] << 8) | before[p3]
                self.ram[address] = before[p1]
            if controls[CTRL_MEMORY_READ]:
                address = (before[p2] << 8) | before[p3]
                movi = (OPCODES["MOVI"] << 11) | ((p1 & 3) << 9) | self.ram[address]
                self.execute_operation(movi)
            if controls[CTRL_HALT]:
                status = "completed" if opcode == OPCODES["HALT"] else "trap"
                break
            self.pc = (
                (pc + 1 + offset) & 0xFFFF
                if controls[CTRL_BRANCH_TAKEN]
                else (pc + 1) & 0xFFFF
            )
            if self.operations - last_progress >= Config.PROGRESS_EVERY_OPERATIONS:
                last_progress = self.operations
                state = prime_runtime_state(self.ram)
                print(
                    f"[exact] operations={self.operations:,} | segment={state['segment']:,} | "
                    f"rank={state['prime_rank']:,}"
                )
        elapsed = time.perf_counter() - started
        state = prime_runtime_state(self.ram)
        return {
            "status": status,
            "mode": "exact",
            "operations": self.operations,
            "fetched_instructions": self.fetched,
            "elapsed_seconds": elapsed,
            "operations_per_second": self.operations / max(elapsed, 1e-9),
            **state,
        }


class BatchedPrimeAuditor:
    """在精确程序轨迹上并行审计 Neural CPU，首个差异立即停止。"""

    def __init__(self, program: list[int], runner: CoreRunner):
        self.program = program
        self.runner = runner
        self.generator = NativeTraceGenerator(program)
        self.operations = 0
        self.trace_seconds = 0.0
        self.gpu_seconds = 0.0
        self.started = time.perf_counter()
        self.last_progress = 0
        Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def summary(self, status: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started
        return {
            "status": status,
            "mode": "batched_exact_trajectory",
            "target_rank": Config.TARGET_RANK,
            "search_limit": Config.SEARCH_LIMIT,
            "expected_prime": Config.EXPECTED_PRIME,
            "model_path": str(self.runner.model_path or Config.MODEL_PATH),
            "operations": self.operations,
            "fetched_instructions": int(self.generator.fetched[0]),
            "elapsed_seconds": elapsed,
            "operations_per_second": self.operations / max(elapsed, 1e-9),
            "trace_generation_seconds": self.trace_seconds,
            "gpu_audit_seconds": self.gpu_seconds,
            "pc": int(self.generator.pc[0]),
            "registers": self.generator.registers.astype(int).tolist(),
            "flags": self.generator.flags.astype(int).tolist(),
            **prime_runtime_state(self.generator.ram),
        }

    def save(self, filename: str, payload: dict[str, Any]) -> Path:
        path = Config.RESULT_DIR / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def decode_input(row: np.ndarray) -> tuple[list[int], list[int]]:
        flags = row[INSTRUCTION_BITS : INSTRUCTION_BITS + NUM_FLAGS].astype(int).tolist()
        registers = []
        cursor = INSTRUCTION_BITS + NUM_FLAGS
        for _ in range(NUM_REGISTERS):
            value = 0
            for bit in row[cursor : cursor + REGISTER_BITS]:
                value = (value << 1) | int(bit)
            registers.append(value)
            cursor += REGISTER_BITS
        return registers, flags

    def mismatch_event(
        self, chunk: dict[str, Any], mismatch: dict[str, Any], base: int
    ) -> dict[str, Any]:
        index = int(mismatch["index"])
        instruction = int(chunk["instructions"][index])
        pc = int(chunk["pcs"][index])
        opcode, p1, p2, p3, immediate, offset = instruction_fields(instruction)
        registers, flags = self.decode_input(chunk["inputs"][index])
        predicted = mismatch["predicted_bits"]
        expected = mismatch["expected_bits"]
        differing = [
            bit for bit, pair in enumerate(zip(predicted, expected)) if pair[0] != pair[1]
        ]
        return {
            "operation_index": base + index + 1,
            "pc": pc,
            "micro_op": "LOAD->MOVI" if int(chunk["micro_ops"][index]) else None,
            "instruction": f"0x{instruction:04X}",
            "opcode": OPCODE_NAMES.get(opcode, str(opcode)),
            "fields": {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "immediate": immediate,
                "offset": offset,
            },
            "before": {"registers": registers, "flags": flags},
            "predicted": decode_output_bits(predicted),
            "expected": decode_output_bits(expected),
            "differing_bits": [
                {
                    "index": bit,
                    "name": output_bit_name(bit),
                    "logit": mismatch["logits"][bit],
                    "probability": mismatch["probabilities"][bit],
                }
                for bit in differing
            ],
        }

    def run(self) -> dict[str, Any]:
        print(
            f"开始质数程序 batch 审计：target_rank={Config.TARGET_RANK:,} | "
            f"search_limit={Config.SEARCH_LIMIT:,} | instructions={len(self.program):,}"
        )
        while True:
            if Config.MAX_NEURAL_OPERATIONS is None:
                capacity = Config.TRACE_BATCH
            else:
                remaining = Config.MAX_NEURAL_OPERATIONS - self.operations
                if remaining <= 0:
                    payload = self.summary("operation_limit")
                    self.save("summary.json", payload)
                    return payload
                capacity = min(Config.TRACE_BATCH, remaining)

            trace_started = time.perf_counter()
            chunk = self.generator.next_chunk(capacity)
            self.trace_seconds += time.perf_counter() - trace_started
            count = int(chunk["inputs"].shape[0])
            base = self.operations
            gpu_started = time.perf_counter()
            mismatch = self.runner.find_first_batch_mismatch(
                chunk["inputs"], chunk["targets"]
            )
            self.gpu_seconds += time.perf_counter() - gpu_started
            if mismatch is not None:
                event = self.mismatch_event(chunk, mismatch, base)
                self.operations = int(event["operation_index"])
                payload = self.summary("first_neural_error")
                payload["error"] = event
                self.save("first_neural_error.json", payload)
                self.save("summary.json", payload)
                names = ", ".join(item["name"] for item in event["differing_bits"])
                print(
                    f"\n首个 Neural CPU 错误：operation={self.operations:,} | "
                    f"pc={event['pc']} | opcode={event['opcode']} | bits={names}"
                )
                return payload

            self.operations = base + count
            if self.operations - self.last_progress >= Config.PROGRESS_EVERY_OPERATIONS:
                self.last_progress = self.operations
                state = prime_runtime_state(self.generator.ram)
                elapsed = time.perf_counter() - self.started
                print(
                    f"[batch] operations={self.operations:,} | segment={state['segment']:,} | "
                    f"rank={state['prime_rank']:,} | active={state['active_prime_count']:,} | "
                    f"speed={self.operations / max(elapsed, 1e-9):,.1f} op/s | errors=0"
                )
                self.save("progress.json", self.summary("running"))

            if chunk["halted"]:
                payload = self.summary("completed")
                if payload["result"] != Config.EXPECTED_PRIME:
                    payload["status"] = "program_result_mismatch"
                    self.save("summary.json", payload)
                    raise RuntimeError(
                        f"程序输出 {payload['result']:,}，预期 {Config.EXPECTED_PRIME:,}。"
                    )
                self.save("summary.json", payload)
                print(
                    f"\n=== 第 {Config.TARGET_RANK:,} 个质数："
                    "全轨迹 Neural CPU 零错误 ==="
                )
                print(
                    f"prime({payload['prime_rank']:,}) = {payload['result']:,} | "
                    f"operations={self.operations:,} | "
                    f"speed={payload['operations_per_second']:,.1f} op/s"
                )
                print(f"汇总：{Config.RESULT_DIR / 'summary.json'}")
                return payload


def main() -> None:
    if Config.EXECUTION_MODE not in {"exact", "batch"}:
        raise ValueError("Config.EXECUTION_MODE 只能是 exact 或 batch。")
    if Config.TRACE_BATCH < 2 or Config.INFERENCE_BATCH < 1:
        raise ValueError("TRACE_BATCH 必须 >= 2，INFERENCE_BATCH 必须 >= 1。")
    if Config.EXPECTED_PRIME > Config.SEARCH_LIMIT:
        raise ValueError("EXPECTED_PRIME 不能大于 SEARCH_LIMIT。")

    builder = PrimeAssemblyBuilder(Config.TARGET_RANK, Config.SEARCH_LIMIT)
    program, listing = builder.build()
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (Config.RESULT_DIR / "prime_program_listing.json").write_text(
        json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (Config.RESULT_DIR / "prime_program.bin").write_bytes(
        b"".join(word.to_bytes(2, "big") for word in program)
    )
    build_info = {
        "target_rank": Config.TARGET_RANK,
        "search_limit": Config.SEARCH_LIMIT,
        "expected_prime": Config.EXPECTED_PRIME,
        "base_limit": builder.base_limit,
        "base_odd_count": builder.base_odd_count,
        "program_instructions": len(program),
        "ram_layout": {
            "segment_buffer": [0, PrimeMemoryMap.SEGMENT_SIZE - 1],
            "base_flags": [PrimeMemoryMap.BASE_FLAGS, PrimeMemoryMap.RECORD_BASE - 1],
            "prime_records": [PrimeMemoryMap.RECORD_BASE, PrimeMemoryMap.RECORD_LIMIT - 1],
            "variables": [0xF000, 0xF0FF],
        },
    }
    (Config.RESULT_DIR / "config.json").write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"质数程序：第 {Config.TARGET_RANK:,} 个 | 搜索上界={Config.SEARCH_LIMIT:,} | "
        f"sqrt 上界={builder.base_limit:,} | machine instructions={len(program):,}"
    )
    print(
        "完整性边界：筛地址、倍数推进、质数计数和停止条件均由 Neural CPU 执行；"
        "宿主仅维护外部 RAM/PC 并审计。"
    )

    if Config.EXECUTION_MODE == "exact":
        result = ExactPrimeMachine(program).run()
        if result["status"] == "completed" and result["result"] != Config.EXPECTED_PRIME:
            raise RuntimeError(
                f"精确程序输出 {result['result']:,}，预期 {Config.EXPECTED_PRIME:,}。"
            )
        (Config.RESULT_DIR / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result["status"] == "completed":
            print(
                f"精确执行完成：prime({result['prime_rank']:,})={result['result']:,} | "
                f"operations={result['operations']:,}"
            )
        else:
            print(
                f"精确执行停止：status={result['status']} | "
                f"operations={result['operations']:,}"
            )
        return

    torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    else:
        raise RuntimeError(
            f"第 {Config.TARGET_RANK:,} 个质数的 Neural CPU batch 审计需要 CUDA。"
        )
    BatchedPrimeAuditor(program, CoreRunner(device)).run()


if __name__ == "__main__":
    main()

# %%
