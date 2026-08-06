"""让 Neural CPU v3 自主执行一个通用 9x9 Sudoku 求解器。

求解算法完全写成项目 ISA 的机器码：

- 棋盘、MRV 搜索状态和显式回溯栈均位于 64-KiB RAM；
- 每个候选数字都由程序扫描该格的 20 个 peer 后判定；
- Python 只负责装入题目、提供固定的 Sudoku peer 表、显示和验算答案；
- serial 模式逐步自主执行神经核心；
- batch 模式先生成唯一的精确轨迹，再用 GPU 大 batch 审计每个神经操作；
- exact 模式不加载权重，只验证汇编程序和求解算法。

所有运行参数都集中在文件顶部的 ``Config`` 中。默认随机抽取 10 道题，
逐题生成轨迹并用 GPU 大 batch 审计。
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 自包含的 Neural CPU v3 ISA、网络和推理器
# =============================================================================


def script_directory() -> Path:
    source = globals().get("__file__")
    if not source:
        return Path.cwd().resolve()
    source_dir = Path(source).resolve().parent
    return source_dir.parent if source_dir.name == "scripts" else source_dir


class _CPUConfig:
    MODEL_PATH = (
        script_directory()
        / "weights"
        / "neural_cpu_v3_best_balanced_model.pt"
    )
    EXACT_ONLY = False
    USE_CUDA_GRAPH = True
    INFERENCE_BATCH = 262144
    BATCH_RECHECK_MARGIN = 1e-4


_NUM_REGISTERS = 4
_REGISTER_BITS = 8
_NUM_FLAGS = 3
_INSTRUCTION_BITS = 16
_STATE_BITS = _NUM_FLAGS + _NUM_REGISTERS * _REGISTER_BITS
_CONTROL_BITS = 4
_INPUT_BITS = _INSTRUCTION_BITS + _STATE_BITS
_OUTPUT_BITS = _STATE_BITS + _CONTROL_BITS

_FLAG_ZF = 0
_FLAG_GF = 1
_FLAG_CF = 2

_CTRL_MEMORY_READ = 0
_CTRL_MEMORY_WRITE = 1
_CTRL_BRANCH_TAKEN = 2
_CTRL_HALT = 3

_OPCODES = {
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
_OPCODE_NAMES = {value: key for key, value in _OPCODES.items()}
_RRR_NAMES = {
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
_RI8_NAMES = {"MOVI", "ADDI", "SUBI", "CMPI"}
_BRANCH_NAMES = {"JMP", "JZ", "JNZ", "JG", "JL", "JC", "JNC"}


def _output_bit_name(index: int) -> str:
    if index < _NUM_FLAGS:
        return ("ZF", "GF", "CF")[index]
    if index < _STATE_BITS:
        offset = index - _NUM_FLAGS
        register = offset // _REGISTER_BITS
        bit_from_msb = offset % _REGISTER_BITS
        return f"R{register}.bit{_REGISTER_BITS - 1 - bit_from_msb}"
    return (
        "memory_read",
        "memory_write",
        "branch_taken",
        "halt",
    )[index - _STATE_BITS]


def _instruction_fields(
    instruction: int,
) -> tuple[int, int, int, int, int, int]:
    opcode = (instruction >> 11) & 0x1F
    p1 = (instruction >> 9) & 0x03
    p2 = (instruction >> 7) & 0x03
    p3 = (instruction >> 5) & 0x03
    immediate = instruction & 0xFF
    raw_offset = instruction & 0x7FF
    offset = raw_offset - 0x800 if raw_offset & 0x400 else raw_offset
    return opcode, p1, p2, p3, immediate, offset


def _scalar_reference(
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
    carry = flags[_FLAG_CF]

    def write(value: int) -> int:
        regs[p1] = value & 0xFF
        return regs[p1]

    if opcode == _OPCODES["HALT"]:
        controls[_CTRL_HALT] = 1
    elif opcode == _OPCODES["MOV"]:
        write(value1)
    elif opcode == _OPCODES["MOVI"]:
        write(immediate)
    elif opcode == _OPCODES["LOAD"]:
        controls[_CTRL_MEMORY_READ] = 1
    elif opcode == _OPCODES["STORE"]:
        controls[_CTRL_MEMORY_WRITE] = 1
    elif opcode in (_OPCODES["ADD"], _OPCODES["ADC"]):
        result = value1 + value2 + (carry if opcode == _OPCODES["ADC"] else 0)
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
        out_flags[_FLAG_CF] = int(result > 0xFF)
    elif opcode in (_OPCODES["SUB"], _OPCODES["SBC"]):
        borrow = carry if opcode == _OPCODES["SBC"] else 0
        result = value1 - value2 - borrow
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
        out_flags[_FLAG_CF] = int(value1 < value2 + borrow)
    elif opcode == _OPCODES["INC"]:
        result = value1 + 1
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
        out_flags[_FLAG_CF] = int(result > 0xFF)
    elif opcode == _OPCODES["DEC"]:
        result = value1 - 1
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
        out_flags[_FLAG_CF] = int(value1 == 0)
    elif opcode in (_OPCODES["AND"], _OPCODES["OR"], _OPCODES["XOR"]):
        if opcode == _OPCODES["AND"]:
            result = value1 & value2
        elif opcode == _OPCODES["OR"]:
            result = value1 | value2
        else:
            result = value1 ^ value2
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
    elif opcode == _OPCODES["NOT"]:
        written = write(~value1)
        out_flags[_FLAG_ZF] = int(written == 0)
    elif opcode in (_OPCODES["SHL"], _OPCODES["SHR"]):
        shift = value2 & 7
        if opcode == _OPCODES["SHL"]:
            result = value1 << shift
            new_carry = carry if shift == 0 else (value1 >> (8 - shift)) & 1
        else:
            result = value1 >> shift
            new_carry = carry if shift == 0 else (value1 >> (shift - 1)) & 1
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
        out_flags[_FLAG_CF] = new_carry
    elif opcode == _OPCODES["CMP"]:
        out_flags[_FLAG_ZF] = int(value1 == value2)
        out_flags[_FLAG_GF] = int(value1 > value2)
        out_flags[_FLAG_CF] = int(value1 < value2)
    elif opcode in (_OPCODES["ADDI"], _OPCODES["SUBI"]):
        if opcode == _OPCODES["ADDI"]:
            result = destination_value + immediate
            new_carry = int(result > 0xFF)
        else:
            result = destination_value - immediate
            new_carry = int(destination_value < immediate)
        written = write(result)
        out_flags[_FLAG_ZF] = int(written == 0)
        out_flags[_FLAG_CF] = new_carry
    elif opcode == _OPCODES["CMPI"]:
        out_flags[_FLAG_ZF] = int(destination_value == immediate)
        out_flags[_FLAG_GF] = int(destination_value > immediate)
        out_flags[_FLAG_CF] = int(destination_value < immediate)
    elif opcode == _OPCODES["JMP"]:
        controls[_CTRL_BRANCH_TAKEN] = 1
    elif opcode == _OPCODES["JZ"]:
        controls[_CTRL_BRANCH_TAKEN] = flags[_FLAG_ZF]
    elif opcode == _OPCODES["JNZ"]:
        controls[_CTRL_BRANCH_TAKEN] = 1 - flags[_FLAG_ZF]
    elif opcode == _OPCODES["JG"]:
        controls[_CTRL_BRANCH_TAKEN] = flags[_FLAG_GF]
    elif opcode in (_OPCODES["JL"], _OPCODES["JC"]):
        controls[_CTRL_BRANCH_TAKEN] = flags[_FLAG_CF]
    elif opcode == _OPCODES["JNC"]:
        controls[_CTRL_BRANCH_TAKEN] = 1 - flags[_FLAG_CF]
    elif opcode == _OPCODES["JMPR"]:
        controls[_CTRL_BRANCH_TAKEN] = 1
    elif opcode == _OPCODES["TRAP"]:
        controls[_CTRL_HALT] = 1

    return regs, out_flags, controls


def _state_and_controls_to_bits(
    registers: list[int], flags: list[int], controls: list[int]
) -> list[int]:
    bits = flags.copy()
    for value in registers:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    bits.extend(controls)
    return bits


class _NeuralCPUCore(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(_INPUT_BITS, hidden_size),
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
        layers.append(nn.Linear(hidden_size, _OUTPUT_BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _find_model_path() -> Path:
    if _CPUConfig.MODEL_PATH.exists():
        return _CPUConfig.MODEL_PATH
    raise FileNotFoundError(
        f"找不到 Neural CPU v3 权重：{_CPUConfig.MODEL_PATH}\n"
        "请修改 Config.MODEL_PATH。"
    )


class _CoreRunner:
    def __init__(self, device: torch.device):
        self.device = device
        self.model: _NeuralCPUCore | None = None
        self.static_input: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

        if _CPUConfig.EXACT_ONLY:
            return
        model_path = _find_model_path()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        first_weight = next(
            value
            for key, value in state_dict.items()
            if key.endswith("weight")
            and value.ndim == 2
            and value.shape[1] == _INPUT_BITS
        )
        linear_weights = [
            value
            for key, value in state_dict.items()
            if key.endswith("weight") and value.ndim == 2
        ]
        hidden_size = int(config.get("HIDDEN_SIZE", first_weight.shape[0]))
        hidden_layers = int(config.get("HIDDEN_LAYERS", len(linear_weights) - 2))
        dropout = float(config.get("DROPOUT", 0.0))
        self.model = _NeuralCPUCore(hidden_size, hidden_layers, dropout).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"加载模型：{model_path}")
        print(
            f"模型结构：51 -> {hidden_size} x {hidden_layers + 1} -> 39；"
            f"checkpoint step={checkpoint.get('step', 'unknown')}"
        )
        if device.type == "cuda" and _CPUConfig.USE_CUDA_GRAPH:
            try:
                self._prepare_cuda_graph()
            except Exception as error:
                self.static_input = None
                self.static_output = None
                self.graph = None
                print(f"CUDA Graph 初始化失败，退回普通推理：{error}")

    def _prepare_cuda_graph(self) -> None:
        assert self.model is not None
        self.static_input = torch.zeros((1, _INPUT_BITS), device=self.device)
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
        assert self.model is not None
        input_bits = [(instruction >> shift) & 1 for shift in range(15, -1, -1)]
        input_bits.extend(flags)
        for value in registers:
            input_bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
        inputs = torch.tensor(
            input_bits, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        if self.graph is not None:
            assert self.static_input is not None and self.static_output is not None
            self.static_input.copy_(inputs)
            self.graph.replay()
            output = self.static_output[0]
        else:
            output = self.model(inputs)[0]
        predicted = (output >= 0).to(torch.int64)
        probabilities = torch.sigmoid(output)
        return (
            predicted.cpu().tolist(),
            output.float().cpu().tolist(),
            probabilities.float().cpu().tolist(),
        )

    @torch.inference_mode()
    def find_first_batch_mismatch(
        self, input_bits: np.ndarray, target_bits: np.ndarray
    ) -> dict[str, Any] | None:
        if self.model is None:
            raise RuntimeError("batch 轨迹审计必须加载 Neural CPU 权重。")
        total = int(input_bits.shape[0])
        batch_size = max(1, _CPUConfig.INFERENCE_BATCH)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            inputs = torch.from_numpy(input_bits[start:end]).to(
                self.device, dtype=torch.float32
            )
            expected = torch.from_numpy(target_bits[start:end]).to(
                self.device, dtype=torch.bool
            )
            logits = self.model(inputs)
            predicted = logits >= 0
            row_mismatch = torch.any(predicted != expected, dim=1)
            low_margin = (
                torch.amin(torch.abs(logits), dim=1)
                < _CPUConfig.BATCH_RECHECK_MARGIN
            )
            candidates = torch.nonzero(
                row_mismatch | low_margin, as_tuple=False
            )
            for candidate in candidates[:, 0].cpu().tolist():
                row = start + int(candidate)
                input_row = input_bits[row]
                instruction = 0
                for bit in input_row[:_INSTRUCTION_BITS]:
                    instruction = (instruction << 1) | int(bit)
                cursor = _INSTRUCTION_BITS
                flags = input_row[cursor : cursor + _NUM_FLAGS].astype(int).tolist()
                cursor += _NUM_FLAGS
                registers: list[int] = []
                for _ in range(_NUM_REGISTERS):
                    value = 0
                    for bit in input_row[cursor : cursor + _REGISTER_BITS]:
                        value = (value << 1) | int(bit)
                    registers.append(value)
                    cursor += _REGISTER_BITS
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
                    }
        return None


@dataclass
class _AssemblyInstruction:
    op: str
    rd: int = 0
    ra: int = 0
    rb: int = 0
    imm: int = 0
    target: str | None = None
    comment: str = ""


class _Assembler:
    def __init__(self):
        self.instructions: list[_AssemblyInstruction] = []
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
            _AssemblyInstruction(op, rd, ra, rb, imm, target, comment)
        )

    def branch(self, op: str, target: str, comment: str = "") -> None:
        self.emit(op, target=target, comment=comment)

    def assemble(self) -> tuple[list[int], list[dict[str, Any]]]:
        machine_code: list[int] = []
        listing: list[dict[str, Any]] = []
        for pc, item in enumerate(self.instructions):
            opcode = _OPCODES[item.op]
            if item.op in _RRR_NAMES:
                word = (
                    (opcode << 11)
                    | ((item.rd & 3) << 9)
                    | ((item.ra & 3) << 7)
                    | ((item.rb & 3) << 5)
                )
            elif item.op in _RI8_NAMES:
                word = (opcode << 11) | ((item.rd & 3) << 9) | (item.imm & 0xFF)
            elif item.op in _BRANCH_NAMES:
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
                    "op": item.op,
                    "rd": item.rd,
                    "ra": item.ra,
                    "rb": item.rb,
                    "imm": item.imm,
                    "target": item.target,
                    "comment": item.comment,
                }
            )
        return machine_code, listing


cpu = SimpleNamespace(
    Config=_CPUConfig,
    CoreRunner=_CoreRunner,
    Assembler=_Assembler,
    INPUT_BITS=_INPUT_BITS,
    OUTPUT_BITS=_OUTPUT_BITS,
    CTRL_MEMORY_READ=_CTRL_MEMORY_READ,
    CTRL_MEMORY_WRITE=_CTRL_MEMORY_WRITE,
    CTRL_BRANCH_TAKEN=_CTRL_BRANCH_TAKEN,
    CTRL_HALT=_CTRL_HALT,
    OPCODES=_OPCODES,
    OPCODE_NAMES=_OPCODE_NAMES,
    output_bit_name=_output_bit_name,
    instruction_fields=_instruction_fields,
    scalar_reference=_scalar_reference,
    state_and_controls_to_bits=_state_and_controls_to_bits,
)


class Config:
    BASE_DIR = script_directory()
    MODEL_PATH = BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    RESULT_DIR = BASE_DIR / "results" / "sudoku"
    MODE = "batch"
    TEACHER_FORCE_ON_ERROR = False
    MAX_NEURAL_OPERATIONS = 1_000_000_000
    TRACE_BATCH = 262_144
    INFERENCE_BATCH = 262_144
    PROGRESS_EVERY = 100_000
    USE_CUDA_GRAPH = True
    DATASET_PATH = BASE_DIR / "assets" / "project_euler_096_sudoku.txt"
    PUZZLE = ""
    PUZZLE_COUNT = 10
    PUZZLE_INDEXES = ""
    RANDOM_SEED = ""
    STOP_ON_FIRST_ERROR = True


class Memory:
    DEPTH = 0x0000
    BEST_CELL = 0x0001
    BEST_COUNT = 0x0002
    SCAN_CELL = 0x0003
    CANDIDATE_COUNT = 0x0004
    DIGIT = 0x0005
    PEER_INDEX = 0x0006
    POINTER_LOW = 0x0007
    POINTER_HIGH = 0x0008
    CHECK_CELL = 0x0009
    PEER_CELL = 0x000A
    PEER_VALUE = 0x000B
    ADDRESS_LOW = 0x000C
    ADDRESS_HIGH = 0x000D
    BOARD_VALUE = 0x000E
    STATUS = 0x000F  # 1=solved, 2=unsatisfiable

    STACK_CELL = 0x0100
    STACK_NEXT = 0x0200
    BOARD = 0x1000
    PEER_POINTER_LOW = 0x1100
    PEER_POINTER_HIGH = 0x1200
    PEERS = 0x2000


def parse_puzzle(text: str) -> list[int]:
    values = [character for character in text if character in "0123456789."]
    if len(values) != 81:
        raise ValueError(f"Sudoku 必须恰好包含 81 格，当前为 {len(values)} 格。")
    board = [0 if value in "0." else int(value) for value in values]
    validate_clues(board)
    return board


@dataclass(frozen=True)
class PuzzleRecord:
    name: str
    source_index: int | None
    clues: list[int]


def resolve_dataset_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(
        f"找不到 Sudoku 题库：{path}\n请修改 Config.DATASET_PATH。"
    )


def load_project_euler_dataset(path: Path) -> tuple[Path, list[PuzzleRecord]]:
    resolved = resolve_dataset_path(path)
    lines = [line.strip() for line in resolved.read_text(encoding="utf-8").splitlines()]
    records: list[PuzzleRecord] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor]:
            cursor += 1
            continue
        name = lines[cursor]
        if not name.startswith("Grid "):
            raise ValueError(f"题库第 {cursor + 1} 行应为 Grid 标题，实际为：{name!r}")
        rows = lines[cursor + 1 : cursor + 10]
        if len(rows) != 9 or any(len(row) != 9 or not row.isdigit() for row in rows):
            raise ValueError(f"{name} 后没有 9 行合法的 9 位数字。")
        records.append(
            PuzzleRecord(
                name=name.replace(" ", "_"),
                source_index=len(records) + 1,
                clues=parse_puzzle("".join(rows)),
            )
        )
        cursor += 10
    if not records:
        raise ValueError(f"题库为空：{resolved}")
    return resolved, records


def select_puzzles() -> tuple[list[PuzzleRecord], dict[str, Any]]:
    if Config.PUZZLE:
        return [PuzzleRecord("custom", None, parse_puzzle(Config.PUZZLE))], {
            "source": "Config.PUZZLE",
            "selection": "custom",
            "seed": None,
        }

    dataset_path, records = load_project_euler_dataset(Config.DATASET_PATH)
    if Config.PUZZLE_INDEXES:
        indexes = [int(token.strip()) for token in Config.PUZZLE_INDEXES.split(",")]
        if not indexes or any(index < 1 or index > len(records) for index in indexes):
            raise ValueError(f"题目编号必须位于 1..{len(records)}：{indexes}")
        selected = [records[index - 1] for index in indexes]
        selection = "fixed_indexes"
        seed: int | None = None
    else:
        count = max(1, min(Config.PUZZLE_COUNT, len(records)))
        seed = int(Config.RANDOM_SEED) if Config.RANDOM_SEED else time.time_ns() & 0xFFFFFFFF
        selected = random.Random(seed).sample(records, count)
        selection = "random_without_replacement"
    return selected, {
        "source": str(dataset_path),
        "source_url": "https://projecteuler.net/resources/documents/0096_sudoku.txt",
        "dataset_size": len(records),
        "selection": selection,
        "seed": seed,
        "selected_indexes": [record.source_index for record in selected],
    }


def validate_clues(board: list[int]) -> None:
    if len(board) != 81 or any(not 0 <= value <= 9 for value in board):
        raise ValueError("棋盘必须由 81 个 0..9 整数组成。")
    groups: list[list[int]] = []
    groups.extend([[row * 9 + column for column in range(9)] for row in range(9)])
    groups.extend([[row * 9 + column for row in range(9)] for column in range(9)])
    for box_row in range(3):
        for box_column in range(3):
            groups.append(
                [
                    (box_row * 3 + row) * 9 + box_column * 3 + column
                    for row in range(3)
                    for column in range(3)
                ]
            )
    for group in groups:
        nonzero = [board[index] for index in group if board[index] != 0]
        if len(nonzero) != len(set(nonzero)):
            raise ValueError("输入题目的初始数字已经违反 Sudoku 约束。")


def is_solution(board: list[int], clues: list[int]) -> bool:
    if any(value == 0 for value in board):
        return False
    if any(clue and board[index] != clue for index, clue in enumerate(clues)):
        return False
    expected = set(range(1, 10))
    for row in range(9):
        if set(board[row * 9 : row * 9 + 9]) != expected:
            return False
    for column in range(9):
        if {board[row * 9 + column] for row in range(9)} != expected:
            return False
    for box_row in range(3):
        for box_column in range(3):
            values = {
                board[(box_row * 3 + row) * 9 + box_column * 3 + column]
                for row in range(3)
                for column in range(3)
            }
            if values != expected:
                return False
    return True


def format_board(board: list[int]) -> str:
    rows: list[str] = []
    for row in range(9):
        values = [str(value) if value else "." for value in board[row * 9 : row * 9 + 9]]
        rows.append(
            " ".join(values[:3]) + " | " + " ".join(values[3:6]) + " | " + " ".join(values[6:])
        )
        if row in (2, 5):
            rows.append("------+-------+------")
    return "\n".join(rows)


def build_peer_table() -> list[list[int]]:
    table: list[list[int]] = []
    for cell in range(81):
        row, column = divmod(cell, 9)
        peers = {row * 9 + other for other in range(9)}
        peers.update(other * 9 + column for other in range(9))
        box_row, box_column = (row // 3) * 3, (column // 3) * 3
        peers.update(
            (box_row + delta_row) * 9 + box_column + delta_column
            for delta_row in range(3)
            for delta_column in range(3)
        )
        peers.remove(cell)
        ordered = sorted(peers)
        if len(ordered) != 20:
            raise AssertionError(f"cell={cell} 的 peer 数不是 20。")
        table.append(ordered)
    return table


def initial_ram(board: list[int]) -> bytearray:
    ram = bytearray(65_536)
    ram[Memory.BOARD : Memory.BOARD + 81] = bytes(board)
    peers = build_peer_table()
    cursor = Memory.PEERS
    for cell, cell_peers in enumerate(peers):
        ram[Memory.PEER_POINTER_LOW + cell] = cursor & 0xFF
        ram[Memory.PEER_POINTER_HIGH + cell] = (cursor >> 8) & 0xFF
        ram[cursor : cursor + 20] = bytes(cell_peers)
        cursor += 20
    return ram


class SudokuAssemblyBuilder:
    def __init__(self) -> None:
        self.asm = cpu.Assembler()

    def load_abs(self, destination: int, address: int) -> None:
        if destination not in (0, 1):
            raise ValueError("绝对 LOAD 宏只使用 R0/R1 作为目标。")
        self.asm.emit("MOVI", rd=2, imm=(address >> 8) & 0xFF)
        self.asm.emit("MOVI", rd=3, imm=address & 0xFF)
        self.asm.emit("LOAD", rd=destination, ra=2, rb=3)

    def store_abs(self, source: int, address: int) -> None:
        if source not in (0, 1):
            raise ValueError("绝对 STORE 宏只使用 R0/R1 作为数据源。")
        self.asm.emit("MOVI", rd=2, imm=(address >> 8) & 0xFF)
        self.asm.emit("MOVI", rd=3, imm=address & 0xFF)
        self.asm.emit("STORE", rd=source, ra=2, rb=3)

    def write_u8(self, address: int, value: int) -> None:
        self.asm.emit("MOVI", rd=0, imm=value)
        self.store_abs(0, address)

    def copy_var(self, source: int, destination: int) -> None:
        self.load_abs(0, source)
        self.store_abs(0, destination)

    def increment_var(self, address: int) -> None:
        self.load_abs(0, address)
        self.asm.emit("ADDI", rd=0, imm=1)
        self.store_abs(0, address)

    def decrement_var(self, address: int) -> None:
        self.load_abs(0, address)
        self.asm.emit("SUBI", rd=0, imm=1)
        self.store_abs(0, address)

    def load_indexed_to_var(self, base: int, index_address: int, output: int) -> None:
        if base & 0xFF:
            raise ValueError("当前 indexed 宏要求数组按 256 字节对齐。")
        self.load_abs(1, index_address)
        self.asm.emit("MOVI", rd=2, imm=(base >> 8) & 0xFF)
        self.asm.emit("MOV", rd=3, ra=1)
        self.asm.emit("LOAD", rd=0, ra=2, rb=3)
        self.store_abs(0, output)

    def store_var_to_indexed(self, value: int, base: int, index_address: int) -> None:
        if base & 0xFF:
            raise ValueError("当前 indexed 宏要求数组按 256 字节对齐。")
        self.load_abs(0, value)
        self.load_abs(1, index_address)
        self.asm.emit("MOVI", rd=2, imm=(base >> 8) & 0xFF)
        self.asm.emit("MOV", rd=3, ra=1)
        self.asm.emit("STORE", rd=0, ra=2, rb=3)

    def emit_candidate_check(
        self, valid_target: str, invalid_target: str, prefix: str
    ) -> None:
        """检查 Memory.CHECK_CELL 能否填入 Memory.DIGIT。"""
        loop = self.asm.unique(f"{prefix}_peer_loop")
        self.write_u8(Memory.PEER_INDEX, 0)
        self.asm.label(loop)
        self.load_abs(0, Memory.PEER_INDEX)
        self.asm.emit("CMPI", rd=0, imm=20)
        self.asm.branch("JZ", valid_target)

        # 取该 cell 的 peer 列表起始地址。
        self.load_indexed_to_var(
            Memory.PEER_POINTER_LOW, Memory.CHECK_CELL, Memory.POINTER_LOW
        )
        self.load_indexed_to_var(
            Memory.PEER_POINTER_HIGH, Memory.CHECK_CELL, Memory.POINTER_HIGH
        )

        # pointer + peer_index，得到真正的 16-bit RAM 地址。
        self.load_abs(0, Memory.POINTER_LOW)
        self.load_abs(1, Memory.PEER_INDEX)
        self.asm.emit("ADD", rd=0, ra=0, rb=1)
        self.store_abs(0, Memory.ADDRESS_LOW)
        self.load_abs(0, Memory.POINTER_HIGH)
        self.asm.emit("MOVI", rd=1, imm=0)
        self.asm.emit("ADC", rd=0, ra=0, rb=1)
        self.store_abs(0, Memory.ADDRESS_HIGH)

        # 先加载 peer 的 cell index，再加载 board[peer_cell]。
        # load_abs 会把 R2/R3 当作地址临时寄存器。先把高低字节都读入
        # R0/R1，再复制到动态地址寄存器，否则第二次读取会覆盖高字节。
        self.load_abs(0, Memory.ADDRESS_HIGH)
        self.load_abs(1, Memory.ADDRESS_LOW)
        self.asm.emit("MOV", rd=2, ra=0)
        self.asm.emit("MOV", rd=3, ra=1)
        self.asm.emit("LOAD", rd=0, ra=2, rb=3)
        self.store_abs(0, Memory.PEER_CELL)

        self.load_indexed_to_var(Memory.BOARD, Memory.PEER_CELL, Memory.PEER_VALUE)
        self.load_abs(0, Memory.PEER_VALUE)
        self.load_abs(1, Memory.DIGIT)
        self.asm.emit("CMP", ra=0, rb=1)
        self.asm.branch("JZ", invalid_target)
        self.increment_var(Memory.PEER_INDEX)
        self.asm.branch("JMP", loop)

    def build(self) -> tuple[list[int], list[dict[str, Any]]]:
        a = self.asm
        enter_level = "enter_level"
        scan_loop = "scan_loop"
        scan_skip = "scan_skip"
        scan_done = "scan_done"
        digit_loop = "digit_loop"
        count_valid = "count_valid"
        count_invalid = "count_invalid"
        digits_done = "digits_done"
        update_best = "update_best"
        after_best = "after_best"
        dead_end = "dead_end"
        try_candidate = "try_candidate"
        try_valid = "try_valid"
        try_invalid = "try_invalid"
        assign = "assign"
        backtrack = "backtrack"
        solved = "solved"
        unsatisfiable = "unsatisfiable"

        self.write_u8(Memory.DEPTH, 0)
        self.write_u8(Memory.STATUS, 0)
        a.branch("JMP", enter_level)

        # 为新深度做 MRV 扫描。
        a.label(enter_level)
        self.write_u8(Memory.BEST_CELL, 255)
        self.write_u8(Memory.BEST_COUNT, 10)
        self.write_u8(Memory.SCAN_CELL, 0)

        a.label(scan_loop)
        self.load_abs(0, Memory.SCAN_CELL)
        a.emit("CMPI", rd=0, imm=81)
        a.branch("JZ", scan_done)
        self.load_indexed_to_var(Memory.BOARD, Memory.SCAN_CELL, Memory.BOARD_VALUE)
        self.load_abs(0, Memory.BOARD_VALUE)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JNZ", scan_skip)

        self.copy_var(Memory.SCAN_CELL, Memory.CHECK_CELL)
        self.write_u8(Memory.CANDIDATE_COUNT, 0)
        self.write_u8(Memory.DIGIT, 1)

        a.label(digit_loop)
        self.load_abs(0, Memory.DIGIT)
        a.emit("CMPI", rd=0, imm=10)
        a.branch("JZ", digits_done)
        self.emit_candidate_check(count_valid, count_invalid, "count")

        a.label(count_valid)
        self.increment_var(Memory.CANDIDATE_COUNT)
        a.label(count_invalid)
        self.increment_var(Memory.DIGIT)
        a.branch("JMP", digit_loop)

        a.label(digits_done)
        self.load_abs(0, Memory.CANDIDATE_COUNT)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JZ", dead_end)
        self.load_abs(0, Memory.CANDIDATE_COUNT)
        self.load_abs(1, Memory.BEST_COUNT)
        a.emit("CMP", ra=0, rb=1)
        a.branch("JL", update_best)
        a.branch("JMP", after_best)

        a.label(update_best)
        self.copy_var(Memory.CANDIDATE_COUNT, Memory.BEST_COUNT)
        self.copy_var(Memory.SCAN_CELL, Memory.BEST_CELL)
        self.load_abs(0, Memory.BEST_COUNT)
        a.emit("CMPI", rd=0, imm=1)
        a.branch("JZ", scan_done)

        a.label(after_best)
        a.label(scan_skip)
        self.increment_var(Memory.SCAN_CELL)
        a.branch("JMP", scan_loop)

        a.label(scan_done)
        self.load_abs(0, Memory.BEST_CELL)
        a.emit("CMPI", rd=0, imm=255)
        a.branch("JZ", solved)
        self.store_var_to_indexed(Memory.BEST_CELL, Memory.STACK_CELL, Memory.DEPTH)
        self.write_u8(Memory.DIGIT, 1)
        self.store_var_to_indexed(Memory.DIGIT, Memory.STACK_NEXT, Memory.DEPTH)
        a.branch("JMP", try_candidate)

        a.label(dead_end)
        a.branch("JMP", backtrack)

        # 尝试当前深度尚未测试的下一个数字。
        a.label(try_candidate)
        self.load_indexed_to_var(Memory.STACK_CELL, Memory.DEPTH, Memory.CHECK_CELL)
        self.load_indexed_to_var(Memory.STACK_NEXT, Memory.DEPTH, Memory.DIGIT)
        self.load_abs(0, Memory.DIGIT)
        a.emit("CMPI", rd=0, imm=10)
        a.branch("JZ", backtrack)
        self.increment_var(Memory.DIGIT)
        self.store_var_to_indexed(Memory.DIGIT, Memory.STACK_NEXT, Memory.DEPTH)
        self.decrement_var(Memory.DIGIT)
        self.emit_candidate_check(try_valid, try_invalid, "try")

        a.label(try_invalid)
        a.branch("JMP", try_candidate)

        a.label(try_valid)
        a.branch("JMP", assign)

        a.label(assign)
        self.store_var_to_indexed(Memory.DIGIT, Memory.BOARD, Memory.CHECK_CELL)
        self.increment_var(Memory.DEPTH)
        a.branch("JMP", enter_level)

        # 弹出父节点，清除它写入的格子，再继续尝试剩余候选。
        a.label(backtrack)
        self.load_abs(0, Memory.DEPTH)
        a.emit("CMPI", rd=0, imm=0)
        a.branch("JZ", unsatisfiable)
        self.decrement_var(Memory.DEPTH)
        self.load_indexed_to_var(Memory.STACK_CELL, Memory.DEPTH, Memory.CHECK_CELL)
        self.write_u8(Memory.DIGIT, 0)
        self.store_var_to_indexed(Memory.DIGIT, Memory.BOARD, Memory.CHECK_CELL)
        a.branch("JMP", try_candidate)

        a.label(solved)
        self.write_u8(Memory.STATUS, 1)
        a.emit("HALT")

        a.label(unsatisfiable)
        self.write_u8(Memory.STATUS, 2)
        a.emit("HALT")
        return a.assemble()


@dataclass
class ExactState:
    pc: int
    registers: list[int]
    flags: list[int]


class ExactTraceGenerator:
    """流式生成 Sudoku 程序的唯一精确执行轨迹。"""

    def __init__(self, program: list[int], ram: bytearray):
        self.program = program
        self.ram = ram
        self.state = ExactState(0, [0, 0, 0, 0], [0, 0, 0])
        self.neural_operations = 0
        self.fetched_instructions = 0
        self.halted = False

    @staticmethod
    def input_bits(instruction: int, registers: list[int], flags: list[int]) -> list[int]:
        bits = [(instruction >> shift) & 1 for shift in range(15, -1, -1)]
        bits.extend(flags)
        for value in registers:
            bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
        return bits

    def _address(self, registers: list[int], high_register: int, low_register: int) -> int:
        return ((registers[high_register] & 0xFF) << 8) | (
            registers[low_register] & 0xFF
        )

    def _record_operation(
        self, instruction: int, pc: int, micro_op: bool
    ) -> tuple[list[int], list[int], list[int]]:
        opcode, p1, p2, p3, immediate, _ = cpu.instruction_fields(instruction)
        before_registers = self.state.registers.copy()
        before_flags = self.state.flags.copy()
        next_registers, next_flags, controls = cpu.scalar_reference(
            opcode,
            p1,
            p2,
            p3,
            immediate,
            before_registers,
            before_flags,
        )
        inputs = self.input_bits(instruction, before_registers, before_flags)
        targets = cpu.state_and_controls_to_bits(next_registers, next_flags, controls)
        self.state.registers = next_registers
        self.state.flags = next_flags
        self.neural_operations += 1
        return inputs, targets, controls

    def next_chunk(self, capacity: int) -> dict[str, Any]:
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        instructions: list[int] = []
        pcs: list[int] = []
        micro_ops: list[int] = []

        while len(inputs) < capacity and not self.halted:
            if self.neural_operations >= Config.MAX_NEURAL_OPERATIONS:
                break
            if not 0 <= self.state.pc < len(self.program):
                raise RuntimeError(f"PC 越界：{self.state.pc}")
            pc = self.state.pc
            instruction = self.program[pc]
            opcode, p1, p2, p3, _, offset = cpu.instruction_fields(instruction)
            if opcode == cpu.OPCODES["LOAD"] and capacity - len(inputs) < 2:
                break
            before_registers = self.state.registers.copy()
            self.fetched_instructions += 1
            row_input, row_target, controls = self._record_operation(
                instruction, pc, False
            )
            inputs.append(row_input)
            targets.append(row_target)
            instructions.append(instruction)
            pcs.append(pc)
            micro_ops.append(0)

            if controls[cpu.CTRL_MEMORY_WRITE]:
                address = self._address(before_registers, p2, p3)
                self.ram[address] = before_registers[p1] & 0xFF

            if controls[cpu.CTRL_MEMORY_READ]:
                address = self._address(before_registers, p2, p3)
                value = self.ram[address]
                movi = (
                    (cpu.OPCODES["MOVI"] << 11)
                    | ((p1 & 3) << 9)
                    | value
                )
                row_input, row_target, _ = self._record_operation(movi, pc, True)
                inputs.append(row_input)
                targets.append(row_target)
                instructions.append(movi)
                pcs.append(pc)
                micro_ops.append(1)

            if controls[cpu.CTRL_HALT]:
                self.halted = True
            elif controls[cpu.CTRL_BRANCH_TAKEN]:
                if opcode == cpu.OPCODES["JMPR"]:
                    self.state.pc = before_registers[p1]
                else:
                    self.state.pc = (pc + 1 + offset) & 0xFFFF
            else:
                self.state.pc = (pc + 1) & 0xFFFF

        return {
            "inputs": np.asarray(inputs, dtype=np.uint8).reshape(-1, cpu.INPUT_BITS),
            "targets": np.asarray(targets, dtype=np.uint8).reshape(-1, cpu.OUTPUT_BITS),
            "instructions": np.asarray(instructions, dtype=np.uint16),
            "pcs": np.asarray(pcs, dtype=np.uint16),
            "micro_ops": np.asarray(micro_ops, dtype=np.uint8),
            "halted": self.halted,
        }


def save_json(filename: str, payload: dict[str, Any]) -> Path:
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = Config.RESULT_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def result_summary(
    mode: str,
    generator: ExactTraceGenerator,
    clues: list[int],
    started: float,
    status: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board = list(generator.ram[Memory.BOARD : Memory.BOARD + 81])
    solver_status_byte = int(generator.ram[Memory.STATUS])
    solver_status = {0: "running", 1: "solved", 2: "unsatisfiable"}.get(
        solver_status_byte, "unknown"
    )
    payload: dict[str, Any] = {
        "status": status,
        "solver_status": solver_status,
        "mode": mode,
        "model_path": str(Config.MODEL_PATH),
        "program_neural_operations": generator.neural_operations,
        "fetched_instructions": generator.fetched_instructions,
        "elapsed_seconds": time.perf_counter() - started,
        "operations_per_second": generator.neural_operations
        / max(time.perf_counter() - started, 1e-9),
        "solver_status_byte": solver_status_byte,
        "solution_valid": is_solution(board, clues),
        "board": board,
        "board_text": format_board(board),
    }
    if error is not None:
        payload["error"] = error
    return payload


def mismatch_event(chunk: dict[str, Any], mismatch: dict[str, Any], base: int) -> dict[str, Any]:
    index = int(mismatch["index"])
    instruction = int(chunk["instructions"][index])
    opcode, p1, p2, p3, immediate, offset = cpu.instruction_fields(instruction)
    differing = [
        bit
        for bit, (predicted, expected) in enumerate(
            zip(mismatch["predicted_bits"], mismatch["expected_bits"])
        )
        if predicted != expected
    ]
    return {
        "operation_index": base + index + 1,
        "pc": int(chunk["pcs"][index]),
        "micro_op": "LOAD->MOVI" if int(chunk["micro_ops"][index]) else None,
        "instruction": f"0x{instruction:04X}",
        "opcode": cpu.OPCODE_NAMES.get(opcode, str(opcode)),
        "fields": {
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "immediate": immediate,
            "offset": offset,
        },
        "differing_bits": [
            {
                "index": bit,
                "name": cpu.output_bit_name(bit),
                "predicted": mismatch["predicted_bits"][bit],
                "expected": mismatch["expected_bits"][bit],
                "logit": mismatch["logits"][bit],
                "probability": mismatch["probabilities"][bit],
            }
            for bit in differing
        ],
    }


def create_core_runner(batch_mode: bool) -> cpu.CoreRunner:
    cpu.Config.MODEL_PATH = Config.MODEL_PATH
    cpu.Config.EXACT_ONLY = False
    cpu.Config.USE_CUDA_GRAPH = Config.USE_CUDA_GRAPH and not batch_mode
    cpu.Config.INFERENCE_BATCH = Config.INFERENCE_BATCH
    cpu.Config.BATCH_RECHECK_MARGIN = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    return cpu.CoreRunner(device)


def run_exact_or_batch(
    program: list[int],
    ram: bytearray,
    clues: list[int],
    batch_audit: bool,
    runner: cpu.CoreRunner | None = None,
) -> dict[str, Any]:
    generator = ExactTraceGenerator(program, ram)
    started = time.perf_counter()
    if batch_audit:
        if runner is None:
            runner = create_core_runner(batch_mode=True)

    verified = 0
    last_progress = 0
    while not generator.halted:
        chunk = generator.next_chunk(Config.TRACE_BATCH)
        count = int(chunk["inputs"].shape[0])
        if count == 0:
            if generator.neural_operations >= Config.MAX_NEURAL_OPERATIONS:
                payload = result_summary(
                    "batch" if batch_audit else "exact",
                    generator,
                    clues,
                    started,
                    "operation_limit",
                )
                save_json("summary.json", payload)
                return payload
            raise RuntimeError("精确轨迹生成器没有前进。")

        if runner is not None:
            mismatch = runner.find_first_batch_mismatch(chunk["inputs"], chunk["targets"])
            if mismatch is not None:
                event = mismatch_event(chunk, mismatch, verified)
                payload = result_summary(
                    "batch", generator, clues, started, "first_neural_error", event
                )
                payload["neural_error_operations"] = 1
                save_json("first_neural_error.json", payload)
                save_json("summary.json", payload)
                print(
                    f"首次神经错误：operation={event['operation_index']:,}, "
                    f"pc={event['pc']}, opcode={event['opcode']}"
                )
                return payload
        verified += count
        if verified - last_progress >= Config.PROGRESS_EVERY:
            last_progress = verified
            elapsed = time.perf_counter() - started
            print(
                f"[{'batch' if batch_audit else 'exact'}] 已验证={verified:,} | "
                f"取指={generator.fetched_instructions:,} | pc={generator.state.pc} | "
                f"速度={verified / max(elapsed, 1e-9):,.1f} op/s"
            )

    payload = result_summary(
        "batch" if batch_audit else "exact",
        generator,
        clues,
        started,
        "completed",
    )
    payload["neural_error_operations"] = 0
    save_json("summary.json", payload)
    return payload


def run_serial(
    program: list[int],
    ram: bytearray,
    clues: list[int],
    runner: cpu.CoreRunner | None = None,
) -> dict[str, Any]:
    if runner is None:
        runner = create_core_runner(batch_mode=False)
    state = ExactState(0, [0, 0, 0, 0], [0, 0, 0])
    started = time.perf_counter()
    operations = 0
    fetched = 0
    errors = 0
    error_by_opcode: Counter[str] = Counter()

    def address(registers: list[int], high: int, low: int) -> int:
        return ((registers[high] & 0xFF) << 8) | (registers[low] & 0xFF)

    def neural_operation(
        instruction: int, pc: int, micro_op: str | None = None
    ) -> tuple[list[int], list[int], list[int], dict[str, Any] | None]:
        nonlocal operations, errors
        opcode, p1, p2, p3, immediate, offset = cpu.instruction_fields(instruction)
        before_registers = state.registers.copy()
        before_flags = state.flags.copy()
        exact_registers, exact_flags, exact_controls = cpu.scalar_reference(
            opcode, p1, p2, p3, immediate, before_registers, before_flags
        )
        exact_bits = cpu.state_and_controls_to_bits(
            exact_registers, exact_flags, exact_controls
        )
        predicted_bits, logits, probabilities = runner.predict(
            instruction, before_registers, before_flags, exact_bits
        )
        operations += 1
        differing = [
            index
            for index, (predicted, expected) in enumerate(
                zip(predicted_bits, exact_bits)
            )
            if predicted != expected
        ]
        event = None
        if differing:
            errors += 1
            opcode_name = cpu.OPCODE_NAMES.get(opcode, str(opcode))
            error_by_opcode[opcode_name] += 1
            event = {
                "operation_index": operations,
                "pc": pc,
                "micro_op": micro_op,
                "instruction": f"0x{instruction:04X}",
                "opcode": opcode_name,
                "before": {"registers": before_registers, "flags": before_flags},
                "differing_bits": [
                    {
                        "name": cpu.output_bit_name(index),
                        "predicted": predicted_bits[index],
                        "expected": exact_bits[index],
                        "logit": logits[index],
                        "probability": probabilities[index],
                    }
                    for index in differing
                ],
            }
            if not Config.TEACHER_FORCE_ON_ERROR:
                return exact_registers, exact_flags, exact_controls, event
        state.registers = exact_registers
        state.flags = exact_flags
        return exact_registers, exact_flags, exact_controls, event

    first_error: dict[str, Any] | None = None
    halted = False
    while not halted and operations < Config.MAX_NEURAL_OPERATIONS:
        if not 0 <= state.pc < len(program):
            raise RuntimeError(f"PC 越界：{state.pc}")
        pc = state.pc
        instruction = program[pc]
        opcode, p1, p2, p3, _, offset = cpu.instruction_fields(instruction)
        before_registers = state.registers.copy()
        fetched += 1
        _, _, controls, event = neural_operation(instruction, pc)
        if event is not None and first_error is None:
            first_error = event
            save_json("first_neural_error.json", event)
            if not Config.TEACHER_FORCE_ON_ERROR:
                break

        if controls[cpu.CTRL_MEMORY_WRITE]:
            ram[address(before_registers, p2, p3)] = before_registers[p1] & 0xFF
        if controls[cpu.CTRL_MEMORY_READ]:
            value = ram[address(before_registers, p2, p3)]
            movi = (cpu.OPCODES["MOVI"] << 11) | ((p1 & 3) << 9) | value
            _, _, _, event = neural_operation(movi, pc, "LOAD->MOVI")
            if event is not None and first_error is None:
                first_error = event
                save_json("first_neural_error.json", event)
                if not Config.TEACHER_FORCE_ON_ERROR:
                    break

        if controls[cpu.CTRL_HALT]:
            halted = True
        elif controls[cpu.CTRL_BRANCH_TAKEN]:
            state.pc = (
                before_registers[p1]
                if opcode == cpu.OPCODES["JMPR"]
                else (pc + 1 + offset) & 0xFFFF
            )
        else:
            state.pc = (pc + 1) & 0xFFFF

        if operations % Config.PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[serial] 操作={operations:,} | 取指={fetched:,} | pc={state.pc} | "
                f"速度={operations / max(elapsed, 1e-9):,.1f} op/s | 错误={errors}"
            )

    generator_view = ExactTraceGenerator(program, ram)
    generator_view.neural_operations = operations
    generator_view.fetched_instructions = fetched
    payload = result_summary(
        "serial",
        generator_view,
        clues,
        started,
        "completed" if halted else ("first_neural_error" if first_error else "operation_limit"),
        first_error,
    )
    payload["neural_error_operations"] = errors
    payload["error_by_opcode"] = dict(error_by_opcode)
    save_json("summary.json", payload)
    return payload


def main() -> None:
    if Config.MODE not in {"exact", "serial", "batch"}:
        raise ValueError("Config.MODE 只能是 exact、serial 或 batch。")
    root_result_dir = Config.RESULT_DIR
    root_result_dir.mkdir(parents=True, exist_ok=True)
    selected, selection = select_puzzles()
    builder = SudokuAssemblyBuilder()
    program, listing = builder.build()
    save_json(
        "config.json",
        {
            "mode": Config.MODE,
            "model_path": str(Config.MODEL_PATH),
            "program_instructions": len(program),
            "selection": selection,
            "selected_puzzles": [record.name for record in selected],
            "trace_batch": Config.TRACE_BATCH,
            "inference_batch": Config.INFERENCE_BATCH,
            "teacher_force_on_error": Config.TEACHER_FORCE_ON_ERROR,
        },
    )
    save_json("program_listing.json", {"listing": listing})
    (root_result_dir / "sudoku_program.bin").write_bytes(
        b"".join(word.to_bytes(2, "little") for word in program)
    )

    print("=== Neural CPU v3 Sudoku ===")
    print(f"模式：{Config.MODE}")
    print(f"机器指令：{len(program):,} 条")
    print(f"题库：{selection['source']}")
    print(f"选择：{selection['selected_indexes']} | seed={selection['seed']}")

    runner: cpu.CoreRunner | None = None
    if Config.MODE == "batch":
        runner = create_core_runner(batch_mode=True)
    elif Config.MODE == "serial":
        runner = create_core_runner(batch_mode=False)

    suite_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for ordinal, record in enumerate(selected, start=1):
        Config.RESULT_DIR = root_result_dir / record.name.lower()
        Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        clues = record.clues
        ram = initial_ram(clues)
        print(
            f"\n=== 题目 {ordinal}/{len(selected)}：{record.name} | "
            f"线索={sum(value != 0 for value in clues)} ==="
        )
        print(format_board(clues))

        if Config.MODE == "exact":
            result = run_exact_or_batch(program, ram, clues, batch_audit=False)
        elif Config.MODE == "batch":
            result = run_exact_or_batch(
                program, ram, clues, batch_audit=True, runner=runner
            )
        else:
            result = run_serial(program, ram, clues, runner=runner)

        result["puzzle_name"] = record.name
        result["puzzle_index"] = record.source_index
        result["clues"] = clues
        result["clue_count"] = sum(value != 0 for value in clues)
        save_json("summary.json", result)
        results.append(result)

        print(f"status={result['status']} | solver={result['solver_status']}")
        print(
            f"neural_operations={result['program_neural_operations']:,} | "
            f"solution_valid={result['solution_valid']}"
        )
        print(result["board_text"])
        if (
            Config.STOP_ON_FIRST_ERROR
            and result["status"] == "first_neural_error"
        ):
            print("检测到首个神经错误，按配置停止后续题目。")
            break

    Config.RESULT_DIR = root_result_dir
    total_operations = sum(item["program_neural_operations"] for item in results)
    total_errors = sum(int(item.get("neural_error_operations", 0)) for item in results)
    aggregate = {
        "status": (
            "first_neural_error"
            if any(item["status"] == "first_neural_error" for item in results)
            else "completed"
        ),
        "mode": Config.MODE,
        "selection": selection,
        "requested_puzzles": len(selected),
        "completed_puzzles": len(results),
        "valid_solutions": sum(bool(item["solution_valid"]) for item in results),
        "total_neural_operations": total_operations,
        "total_neural_error_operations": total_errors,
        "elapsed_seconds": time.perf_counter() - suite_started,
        "results": results,
    }
    save_json("suite_summary.json", aggregate)
    print("\n=== 题库汇总 ===")
    print(
        f"完成={len(results)}/{len(selected)} | 有效解={aggregate['valid_solutions']} | "
        f"总 Neural CPU 操作={total_operations:,} | 神经错误={total_errors:,}"
    )
    print(f"结果目录：{root_result_dir}")


if __name__ == "__main__":
    main()

# %%
