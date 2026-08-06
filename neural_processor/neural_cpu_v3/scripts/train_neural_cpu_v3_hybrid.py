"""
训练 Neural CPU v3 混合架构的神经执行核心。

设计边界：
- 神经网络负责：4 个 8-bit 寄存器、ZF/GF/CF、ALU 状态转移及控制信号。
- 精确控制器负责：PC、GPU RAM、取指，以及 LOAD 的第二个 MOVI 微操作。
- 输入：16-bit 指令 + 35-bit 神经状态。
- 输出：35-bit 下一状态 + memory_read/memory_write/branch_taken/halt。

脚本完全自包含，不生成大型离线数据集；训练样本在 GPU 上实时生成，验证集则
由固定随机种子重复生成。默认不设置最大训练步数，按 Ctrl+C 会保存断点。

所有运行参数都集中在文件顶部的 ``Config`` 中。将 ``SMOKE_TEST`` 改为
``True`` 可执行快速自检。
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    SEED = 20260803
    BASE_DIR = script_directory()

    # 从原最佳权重继续训练，并在原始在线分布中混入少量 hard cases。
    HARD_REPLAY_MODE = True
    PRETRAINED_MODEL_PATH = (
        BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    )
    RESULT_DIR = BASE_DIR / "results" / "training"

    # HIDDEN_LAYERS 表示“第一隐藏层之后追加的层数”。因此这里实际共有 3 层。
    HIDDEN_SIZE = 1024
    HIDDEN_LAYERS = 2
    DROPOUT = 0.0

    # 2.2M 参数的 MLP 在 5090 上可安全使用更大的 batch；低 loss 修补阶段也会更平滑。
    BATCH_SIZE = 16_384
    LEARNING_RATE = 3e-6 if HARD_REPLAY_MODE else 1e-3
    WEIGHT_DECAY = 0.0
    GRAD_CLIP_NORM = None

    # hard batch 只占 1/8，剩余 7/8 继续来自原始生成分布。
    HARD_REPLAY_FRACTION = 0.125
    # replay 内只保留少量已知算术边界样本，主体由全量审计的 opcode loss 动态分配。
    STRUCTURED_HARD_SHARE_OF_REPLAY = 0.25
    HARD_REPLAY_SEED = 20260806

    # 这份审计集会影响后续采样，因此它是训练控制集，不是最终测试集。
    DYNAMIC_OPCODE_REPLAY = HARD_REPLAY_MODE
    # 6,250 * 16,384 = 102,400,000 条训练样本，约等于旧配置每 25,000 步一次。
    DYNAMIC_AUDIT_INTERVAL = 6_250
    DYNAMIC_AUDIT_SAMPLES = 33_554_432
    DYNAMIC_AUDIT_BATCH_SIZE = 262_144
    DYNAMIC_AUDIT_SEED = 20260807
    # 除了按 opcode 审计，还显式覆盖程序运行中常见的低汉明重量状态。
    # 高汉明重量状态作为对称对照；若它本身很容易，动态权重会自动降低。
    STRUCTURAL_VALIDATION_SAMPLES_PER_GROUP = 262_144
    DYNAMIC_STRUCTURAL_AUDIT_SAMPLES_PER_GROUP = 4_194_304
    # EMA 中该次审计所占比例。
    DYNAMIC_LOSS_EMA_ALPHA = 0.5
    # opcode 风险 = 平均 bit BCE + 该 opcode 的 sample exact error rate。
    # 两者均为无量纲概率尺度；后者可捕获 MOVI/JZ 这类均值很低但仍有孤立错误的长尾。
    DYNAMIC_EXACT_ERROR_RATE_WEIGHT = 1.0
    # 1.0 表示严格按 loss 比例；0.5 使用平方根降温，减少采样权重振荡。
    DYNAMIC_LOSS_POWER = 0.5
    # 动态 replay 本身仍保留 20% 均匀分布，防止某些 opcode 在 replay 中完全消失。
    DYNAMIC_UNIFORM_MIX = 0.2
    # 第一次从静态 hard replay 切到动态调度时，从既有 best_model.pt 而非已退化的 latest 重新出发。
    DYNAMIC_FIRST_RUN_FROM_BEST = True

    # 在线训练没有 epoch。None 表示持续训练，直到用户中断。
    MAX_STEPS = None
    LOG_INTERVAL = 25
    EVAL_INTERVAL = 125
    CHECKPOINT_INTERVAL = 125

    # 固定验证流。1,048,576 条样本恰好是 32 个 opcode 的整数倍。
    VALIDATION_SAMPLES = 1_048_576
    VALIDATION_BATCH_SIZE = 32_768
    VALIDATION_SEED = 20260804

    # replay 使用固定的新学习率，不继承旧 scheduler 已经降得很低的状态。
    USE_LR_SCHEDULER = not HARD_REPLAY_MODE
    LR_REDUCE_FACTOR = 0.5
    LR_PATIENCE_EVALS = 8
    MIN_LEARNING_RATE = 1e-7

    AUTO_RESUME = True
    SAVE_LATEST_EVERY_EVAL = True

    # 将此项改为 True 后，只加载 best_model.pt 做大规模错误诊断，不继续训练。
    ANALYZE_BEST_MODEL_ONLY = False
    # 动态 replay 模式优先诊断兼顾最坏 opcode 的权重；旧目录没有该文件时自动退回 best_model.pt。
    ANALYSIS_MODEL_FILENAME = "best_balanced_model.pt"
    ANALYSIS_SAMPLES = 33_554_432
    # 5090 上的纯推理批量；若特定环境 OOM，可降为 131_072。
    ANALYSIS_BATCH_SIZE = 262_144
    ANALYSIS_SEED = 20260805
    ANALYSIS_MAX_SAVED_ERROR_CASES = 10_000
    ANALYSIS_PROGRESS_INTERVAL = 1_048_576

    # 关闭 TF32 和 AMP，保留最后几个 9 所需的 FP32 数值精度。
    ALLOW_TF32 = False

    # 同时兼容普通 .py、%run 和直接粘贴到 Jupyter cell 三种运行方式。
    SMOKE_TEST = False


# =============================================================================
# ISA 与位布局
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
NUM_OPCODES = 32

# 这两个名字会作为“虚拟 opcode”进入同一套动态风险与 replay 调度。
# 状态包含 3 个 flag bit 和 4x8 个寄存器 bit，共 35 bit。边缘区定义为
# 少数 bit 不超过 8 个，即至少四分之三的状态位取同一个值。
STRUCTURAL_EDGE_MAX_MINORITY_BITS = 8
STRUCTURAL_STATE_GROUPS: dict[str, tuple[int, int]] = {
    "SPARSE_STATE": (0, STRUCTURAL_EDGE_MAX_MINORITY_BITS),
    "DENSE_STATE": (STATE_BITS - STRUCTURAL_EDGE_MAX_MINORITY_BITS, STATE_BITS),
}


def output_bit_name(index: int) -> str:
    if index < NUM_FLAGS:
        return ("ZF", "GF", "CF")[index]
    if index < STATE_BITS:
        register_offset = index - NUM_FLAGS
        register = register_offset // REGISTER_BITS
        bit_from_msb = register_offset % REGISTER_BITS
        return f"R{register}.bit{REGISTER_BITS - 1 - bit_from_msb}"
    return (
        "memory_read",
        "memory_write",
        "branch_taken",
        "halt",
    )[index - STATE_BITS]

RRR_OPCODES = {
    OPCODES[name]
    for name in (
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
    )
}
RI8_OPCODES = {OPCODES[name] for name in ("MOVI", "ADDI", "SUBI", "CMPI")}
RELATIVE_BRANCH_OPCODES = {
    OPCODES[name] for name in ("JMP", "JZ", "JNZ", "JG", "JL", "JC", "JNC")
}


ISA_DESCRIPTION = {
    "architecture": "Neural CPU v3 hybrid",
    "registers": "R0-R3, each uint8",
    "flags": ["ZF", "GF", "CF"],
    "external_precise_state": ["PC16", "64-KiB uint8 RAM"],
    "input_layout": "instruction[16] + ZF/GF/CF[3] + R0..R3[32], MSB first",
    "output_layout": (
        "next ZF/GF/CF[3] + next R0..R3[32] + "
        "memory_read/memory_write/branch_taken/halt[4]"
    ),
    "instruction_formats": {
        "RRR": "opcode[5] rd[2] ra[2] rb[2] reserved[5]",
        "RI8": "opcode[5] rd[2] reserved[1] imm8[8]",
        "BR11": "opcode[5] signed_relative_offset[11]",
        "JMPR": "opcode[5] ra[2] reserved[9]",
    },
    "load_micro_op": (
        "LOAD first emits memory_read without changing neural state; the external controller "
        "reads RAM and feeds an internal MOVI rd,value through the same neural core."
    ),
    "opcodes": OPCODES,
}


# =============================================================================
# 网络
# =============================================================================


class NeuralCPUCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(INPUT_BITS, Config.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(Config.HIDDEN_SIZE),
        ]
        if Config.DROPOUT > 0:
            layers.append(nn.Dropout(Config.DROPOUT))

        for _ in range(Config.HIDDEN_LAYERS):
            layers.extend(
                [
                    nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE),
                    nn.GELU(),
                    nn.LayerNorm(Config.HIDDEN_SIZE),
                ]
            )
            if Config.DROPOUT > 0:
                layers.append(nn.Dropout(Config.DROPOUT))

        layers.append(nn.Linear(Config.HIDDEN_SIZE, OUTPUT_BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 真值生成器
# =============================================================================


@dataclass
class BatchMetadata:
    opcodes: torch.Tensor
    instructions: torch.Tensor
    registers: torch.Tensor
    flags: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor
    p3: torch.Tensor
    immediate: torch.Tensor
    offset: torch.Tensor


def integer_to_bits(values: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width - 1, -1, -1, device=values.device, dtype=torch.int64)
    return ((values.unsqueeze(1) >> shifts) & 1).to(torch.float32)


def registers_to_bits(registers: torch.Tensor) -> torch.Tensor:
    bits = integer_to_bits(registers.reshape(-1), REGISTER_BITS)
    return bits.reshape(registers.shape[0], NUM_REGISTERS * REGISTER_BITS)


def _balanced_state_hamming_weights(
    batch_size: int,
    minimum: int,
    maximum: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0 <= minimum <= maximum <= STATE_BITS:
        raise ValueError(
            f"非法状态汉明重量范围：[{minimum}, {maximum}]，STATE_BITS={STATE_BITS}"
        )
    values = torch.arange(minimum, maximum + 1, device=device, dtype=torch.int64)
    repeats = math.ceil(batch_size / values.numel())
    weights = values.repeat(repeats)[:batch_size]
    permutation = torch.randperm(batch_size, device=device, generator=generator)
    return weights[permutation]


def _states_with_hamming_weights(
    hamming_weights: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在给定汉明重量下均匀随机选择 35-bit 神经状态。"""
    batch_size = hamming_weights.numel()
    weights = hamming_weights.to(device=device, dtype=torch.int64)
    if bool(((weights < 0) | (weights > STATE_BITS)).any().item()):
        raise ValueError("forced_state_hamming_weights 超出合法范围")

    minority_counts = torch.minimum(weights, STATE_BITS - weights)
    maximum_minority = int(minority_counts.max().item()) if batch_size else 0
    dense_rows = weights > STATE_BITS // 2

    if maximum_minority <= STRUCTURAL_EDGE_MAX_MINORITY_BITS:
        # sparse/dense 专项组至多只有少量少数 bit。直接无放回抽位置，
        # 避免为每一行对 35 个随机数排序。拒绝重复后，每个有序位置组等概率，
        # 因而忽略顺序后的 bit 子集也严格均匀。
        state_bits = dense_rows.to(torch.int64).unsqueeze(1).expand(
            -1, STATE_BITS
        ).clone()
        if maximum_minority > 0:
            positions = torch.empty(
                (batch_size, maximum_minority), device=device, dtype=torch.int64
            )
            for column in range(maximum_minority):
                candidate = torch.randint(
                    0,
                    STATE_BITS,
                    (batch_size,),
                    device=device,
                    generator=generator,
                )
                if column > 0:
                    duplicate = (candidate.unsqueeze(1) == positions[:, :column]).any(
                        dim=1
                    )
                    while bool(duplicate.any().item()):
                        candidate[duplicate] = torch.randint(
                            0,
                            STATE_BITS,
                            (int(duplicate.sum().item()),),
                            device=device,
                            generator=generator,
                        )
                        duplicate = (
                            candidate.unsqueeze(1) == positions[:, :column]
                        ).any(dim=1)
                positions[:, column] = candidate

            active = (
                torch.arange(maximum_minority, device=device).unsqueeze(0)
                < minority_counts.unsqueeze(1)
            )
            base_values = dense_rows.to(torch.int64).unsqueeze(1).expand(
                -1, maximum_minority
            )
            scatter_values = torch.where(active, 1 - base_values, base_values)
            state_bits.scatter_(1, positions, scatter_values)
    else:
        # 保留任意中间汉明重量的通用路径；当前专项训练不会走到这里。
        random_scores = torch.rand(
            (batch_size, STATE_BITS), device=device, generator=generator
        )
        positions = random_scores.argsort(dim=1)
        selected = (
            torch.arange(STATE_BITS, device=device).unsqueeze(0)
            < weights.unsqueeze(1)
        ).to(torch.int64)
        state_bits = torch.zeros(
            (batch_size, STATE_BITS), device=device, dtype=torch.int64
        )
        state_bits.scatter_(1, positions, selected)

    flags = state_bits[:, :NUM_FLAGS]
    register_bits = state_bits[:, NUM_FLAGS:].reshape(-1, REGISTER_BITS)
    powers = 2 ** torch.arange(
        REGISTER_BITS - 1, -1, -1, device=device, dtype=torch.int64
    )
    registers = (register_bits * powers).sum(dim=1).reshape(
        batch_size, NUM_REGISTERS
    )
    return registers, flags


def _balanced_opcodes(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    # 每个连续 batch 对 32 个 opcode 严格均衡（尾数最多相差 1）。
    opcodes = torch.arange(batch_size, device=device, dtype=torch.int64) % NUM_OPCODES
    permutation = torch.randperm(batch_size, device=device, generator=generator)
    return opcodes[permutation]


def _set_destination(
    output_registers: torch.Tensor,
    destination: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> None:
    rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if rows.numel() > 0:
        output_registers[rows, destination[rows]] = values[rows] & 0xFF


def _set_flag(
    output_flags: torch.Tensor,
    flag_index: int,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> None:
    output_flags[mask, flag_index] = values[mask].to(torch.int64)


def generate_batch(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    return_metadata: bool = False,
    forced_opcodes: torch.Tensor | None = None,
    forced_state_hamming_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata | None]:
    if forced_opcodes is None:
        opcodes = _balanced_opcodes(batch_size, device, generator)
    else:
        if forced_opcodes.shape != (batch_size,):
            raise ValueError(
                f"forced_opcodes 形状必须是 ({batch_size},)，"
                f"实际是 {tuple(forced_opcodes.shape)}"
            )
        opcodes = forced_opcodes.to(device=device, dtype=torch.int64)
        if bool(((opcodes < 0) | (opcodes >= NUM_OPCODES)).any().item()):
            raise ValueError("forced_opcodes 包含非法 opcode")
    p1 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    p2 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    p3 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    immediate = torch.randint(0, 256, (batch_size,), device=device, generator=generator)
    offset = torch.randint(-1024, 1024, (batch_size,), device=device, generator=generator)
    registers = torch.randint(
        0,
        256,
        (batch_size, NUM_REGISTERS),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    flags = torch.randint(
        0,
        2,
        (batch_size, NUM_FLAGS),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    if forced_state_hamming_weights is not None:
        if forced_state_hamming_weights.shape != (batch_size,):
            raise ValueError(
                f"forced_state_hamming_weights 形状必须是 ({batch_size},)，"
                f"实际是 {tuple(forced_state_hamming_weights.shape)}"
            )
        requested_weights = forced_state_hamming_weights.to(
            device=device, dtype=torch.int64
        )
        if bool(((requested_weights < -1) | (requested_weights > STATE_BITS)).any().item()):
            raise ValueError("状态汉明重量只允许 -1（保持 IID）或 0..STATE_BITS")
        structural_mask = requested_weights >= 0
        if bool(structural_mask.any().item()):
            structural_registers, structural_flags = _states_with_hamming_weights(
                requested_weights[structural_mask], device, generator
            )
            registers[structural_mask] = structural_registers
            flags[structural_mask] = structural_flags

    # 所有未使用位固定为 0，避免网络学习无意义的多种编码。
    instructions = opcodes << 11
    rrr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    ri8_mask = torch.zeros_like(rrr_mask)
    branch_mask = torch.zeros_like(rrr_mask)
    for opcode in RRR_OPCODES:
        rrr_mask |= opcodes == opcode
    for opcode in RI8_OPCODES:
        ri8_mask |= opcodes == opcode
    for opcode in RELATIVE_BRANCH_OPCODES:
        branch_mask |= opcodes == opcode

    instructions[rrr_mask] |= (
        (p1[rrr_mask] << 9)
        | (p2[rrr_mask] << 7)
        | (p3[rrr_mask] << 5)
    )
    instructions[ri8_mask] |= (p1[ri8_mask] << 9) | immediate[ri8_mask]
    instructions[branch_mask] |= offset[branch_mask] & 0x7FF
    jmpr_mask = opcodes == OPCODES["JMPR"]
    instructions[jmpr_mask] |= p1[jmpr_mask] << 9

    output_registers = registers.clone()
    output_flags = flags.clone()
    controls = torch.zeros(
        (batch_size, CONTROL_BITS), device=device, dtype=torch.int64
    )

    rows = torch.arange(batch_size, device=device)
    value1 = registers[rows, p2]
    value2 = registers[rows, p3]
    destination_value = registers[rows, p1]
    carry_in = flags[:, FLAG_CF]

    def operation_mask(name: str) -> torch.Tensor:
        return opcodes == OPCODES[name]

    mask = operation_mask("MOV")
    _set_destination(output_registers, p1, mask, value1)

    mask = operation_mask("MOVI")
    _set_destination(output_registers, p1, mask, immediate)

    controls[operation_mask("LOAD"), CTRL_MEMORY_READ] = 1
    controls[operation_mask("STORE"), CTRL_MEMORY_WRITE] = 1

    mask = operation_mask("ADD")
    result = value1 + value2
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("ADC")
    result = value1 + value2 + carry_in
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("SUB")
    result = value1 - value2
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, value1 < value2)

    mask = operation_mask("SBC")
    result = value1 - value2 - carry_in
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, value1 < (value2 + carry_in))

    mask = operation_mask("INC")
    result = value1 + 1
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("DEC")
    result = value1 - 1
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, value1 == 0)

    mask = operation_mask("AND")
    result = value1 & value2
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, result == 0)

    mask = operation_mask("OR")
    result = value1 | value2
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, result == 0)

    mask = operation_mask("XOR")
    result = value1 ^ value2
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, result == 0)

    mask = operation_mask("NOT")
    result = (~value1) & 0xFF
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, result == 0)

    shift = value2 & 0x7
    mask = operation_mask("SHL")
    result = (value1 << shift) & 0xFF
    shift_left_carry = torch.where(
        shift == 0,
        carry_in,
        (value1 >> (8 - shift).clamp(min=1)) & 1,
    )
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, result == 0)
    _set_flag(output_flags, FLAG_CF, mask, shift_left_carry)

    mask = operation_mask("SHR")
    result = value1 >> shift
    shift_right_carry = torch.where(
        shift == 0,
        carry_in,
        (value1 >> (shift - 1).clamp(min=0)) & 1,
    )
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, result == 0)
    _set_flag(output_flags, FLAG_CF, mask, shift_right_carry)

    mask = operation_mask("CMP")
    _set_flag(output_flags, FLAG_ZF, mask, value1 == value2)
    _set_flag(output_flags, FLAG_GF, mask, value1 > value2)
    _set_flag(output_flags, FLAG_CF, mask, value1 < value2)

    mask = operation_mask("ADDI")
    result = destination_value + immediate
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("SUBI")
    result = destination_value - immediate
    _set_destination(output_registers, p1, mask, result)
    _set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    _set_flag(output_flags, FLAG_CF, mask, destination_value < immediate)

    mask = operation_mask("CMPI")
    _set_flag(output_flags, FLAG_ZF, mask, destination_value == immediate)
    _set_flag(output_flags, FLAG_GF, mask, destination_value > immediate)
    _set_flag(output_flags, FLAG_CF, mask, destination_value < immediate)

    controls[operation_mask("JMP"), CTRL_BRANCH_TAKEN] = 1
    controls[operation_mask("JZ"), CTRL_BRANCH_TAKEN] = flags[
        operation_mask("JZ"), FLAG_ZF
    ]
    controls[operation_mask("JNZ"), CTRL_BRANCH_TAKEN] = 1 - flags[
        operation_mask("JNZ"), FLAG_ZF
    ]
    controls[operation_mask("JG"), CTRL_BRANCH_TAKEN] = flags[
        operation_mask("JG"), FLAG_GF
    ]
    controls[operation_mask("JL"), CTRL_BRANCH_TAKEN] = flags[
        operation_mask("JL"), FLAG_CF
    ]
    controls[operation_mask("JC"), CTRL_BRANCH_TAKEN] = flags[
        operation_mask("JC"), FLAG_CF
    ]
    controls[operation_mask("JNC"), CTRL_BRANCH_TAKEN] = 1 - flags[
        operation_mask("JNC"), FLAG_CF
    ]
    controls[operation_mask("JMPR"), CTRL_BRANCH_TAKEN] = 1
    controls[operation_mask("HALT"), CTRL_HALT] = 1
    controls[operation_mask("TRAP"), CTRL_HALT] = 1

    instruction_bits = integer_to_bits(instructions, INSTRUCTION_BITS)
    state_bits = torch.cat([flags.to(torch.float32), registers_to_bits(registers)], dim=1)
    output_state_bits = torch.cat(
        [output_flags.to(torch.float32), registers_to_bits(output_registers)], dim=1
    )
    x = torch.cat([instruction_bits, state_bits], dim=1)
    y = torch.cat([output_state_bits, controls.to(torch.float32)], dim=1)

    metadata = None
    if return_metadata:
        metadata = BatchMetadata(
            opcodes=opcodes,
            instructions=instructions,
            registers=registers,
            flags=flags,
            p1=p1,
            p2=p2,
            p3=p3,
            immediate=immediate,
            offset=offset,
        )
    return x, y, metadata


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


def bits_to_integers(bits: torch.Tensor, width: int) -> torch.Tensor:
    powers = 2 ** torch.arange(width - 1, -1, -1, device=bits.device, dtype=torch.int64)
    return (bits.to(torch.int64) * powers).sum(dim=1)


def run_semantics_self_test(device: torch.device) -> None:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(123456)
    x, y, metadata = generate_batch(4096, device, generator, return_metadata=True)
    assert metadata is not None
    assert x.shape == (4096, INPUT_BITS)
    assert y.shape == (4096, OUTPUT_BITS)

    decoded_instruction = bits_to_integers(x[:, :INSTRUCTION_BITS], INSTRUCTION_BITS)
    if not torch.equal(decoded_instruction, metadata.instructions):
        raise AssertionError("指令编码自检失败。")

    output_flags = y[:, :NUM_FLAGS].to(torch.int64)
    output_register_bits = y[:, NUM_FLAGS:STATE_BITS].reshape(-1, REGISTER_BITS)
    output_registers = bits_to_integers(output_register_bits, REGISTER_BITS).reshape(
        -1, NUM_REGISTERS
    )
    controls = y[:, STATE_BITS:].to(torch.int64)

    # 独立的标量实现逐条核对一部分样本。
    cpu_metadata = BatchMetadata(
        **{
            field: getattr(metadata, field).detach().cpu()
            for field in metadata.__dataclass_fields__
        }
    )
    cpu_output_flags = output_flags.detach().cpu()
    cpu_output_registers = output_registers.detach().cpu()
    cpu_controls = controls.detach().cpu()
    for index in range(1024):
        expected_regs, expected_flags, expected_controls = scalar_reference(
            opcode=int(cpu_metadata.opcodes[index]),
            p1=int(cpu_metadata.p1[index]),
            p2=int(cpu_metadata.p2[index]),
            p3=int(cpu_metadata.p3[index]),
            immediate=int(cpu_metadata.immediate[index]),
            registers=cpu_metadata.registers[index].tolist(),
            flags=cpu_metadata.flags[index].tolist(),
        )
        if cpu_output_registers[index].tolist() != expected_regs:
            raise AssertionError(f"寄存器真值自检失败，样本 {index}。")
        if cpu_output_flags[index].tolist() != expected_flags:
            raise AssertionError(f"标志位真值自检失败，样本 {index}。")
        if cpu_controls[index].tolist() != expected_controls:
            raise AssertionError(f"控制信号真值自检失败，样本 {index}。")

    print("ISA 语义自检通过：向量化真值与独立标量实现一致。")


def encode_hard_arithmetic_xy(
    opcode: int,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    immediate: torch.Tensor,
    registers: torch.Tensor,
    flags: torch.Tensor,
    result: torch.Tensor,
    new_cf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把定向构造的算术边界样本编码成标准 Neural CPU I/O。"""
    batch_size = registers.shape[0]
    rows = torch.arange(batch_size, device=registers.device)
    if opcode in (OPCODES["ADDI"], OPCODES["SUBI"]):
        instructions = (torch.full_like(p1, opcode) << 11) | (p1 << 9) | immediate
    else:
        instructions = (
            (torch.full_like(p1, opcode) << 11)
            | (p1 << 9)
            | (p2 << 7)
            | (p3 << 5)
        )
    output_registers = registers.clone()
    output_registers[rows, p1] = result & 0xFF
    output_flags = flags.clone()
    output_flags[:, FLAG_ZF] = ((result & 0xFF) == 0).to(torch.int64)
    output_flags[:, FLAG_CF] = new_cf.to(torch.int64)
    controls = torch.zeros(
        (batch_size, CONTROL_BITS), device=registers.device, dtype=torch.int64
    )
    x = torch.cat(
        [
            integer_to_bits(instructions, INSTRUCTION_BITS),
            flags.to(torch.float32),
            registers_to_bits(registers),
        ],
        dim=1,
    )
    y = torch.cat(
        [
            output_flags.to(torch.float32),
            registers_to_bits(output_registers),
            controls.to(torch.float32),
        ],
        dim=1,
    )
    return x, y


def generate_hard_replay_batch(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成新的 SBC/ADDI/SUBI 边界样本，不复用固定错误样本。"""
    # 50% SBC、25% ADDI、25% SUBI；SBC 是当前主要错误来源。
    suite_token = torch.randint(
        0, 4, (batch_size,), device=device, generator=generator
    )
    suite = torch.where(suite_token < 2, 0, suite_token - 1)
    p1 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    p2 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    source_offset = torch.randint(
        1, 4, (batch_size,), device=device, generator=generator
    )
    p3 = (p2 + source_offset) & 3
    flags = torch.randint(
        0, 2, (batch_size, NUM_FLAGS), device=device, generator=generator
    )
    registers = torch.randint(
        0,
        256,
        (batch_size, NUM_REGISTERS),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )

    make_zero = torch.randint(
        0, 2, (batch_size,), device=device, generator=generator
    ).to(torch.bool)
    hard_nonzero_values = torch.tensor(
        [1, 8, 16, 32, 64, 84, 88, 100, 127, 128, 187, 191, 192, 200, 212, 223, 224, 255],
        device=device,
        dtype=torch.int64,
    )
    nonzero_choice = torch.randint(
        0,
        hard_nonzero_values.numel(),
        (batch_size,),
        device=device,
        generator=generator,
    )
    desired_result = torch.where(
        make_zero,
        torch.zeros(batch_size, device=device, dtype=torch.int64),
        hard_nonzero_values[nonzero_choice],
    )
    x = torch.empty((batch_size, INPUT_BITS), device=device, dtype=torch.float32)
    y = torch.empty((batch_size, OUTPUT_BITS), device=device, dtype=torch.float32)

    sbc_mask = suite == 0
    if sbc_mask.any():
        indices = torch.nonzero(sbc_mask, as_tuple=False).squeeze(1)
        count = indices.numel()
        carry = (
            torch.randint(0, 4, (count,), device=device, generator=generator) != 0
        ).to(torch.int64)
        value1 = torch.randint(0, 256, (count,), device=device, generator=generator)
        value2 = (value1 - desired_result[indices] - carry) & 0xFF
        local_registers = registers[indices]
        local_rows = torch.arange(count, device=device)
        local_registers[local_rows, p2[indices]] = value1
        local_registers[local_rows, p3[indices]] = value2
        local_flags = flags[indices]
        local_flags[:, FLAG_CF] = carry
        result = value1 - value2 - carry
        new_cf = value1 < (value2 + carry)
        immediate = torch.zeros(count, device=device, dtype=torch.int64)
        local_x, local_y = encode_hard_arithmetic_xy(
            OPCODES["SBC"],
            p1[indices],
            p2[indices],
            p3[indices],
            immediate,
            local_registers,
            local_flags,
            result,
            new_cf,
        )
        x[indices] = local_x
        y[indices] = local_y

    for suite_index, opcode_name in ((1, "ADDI"), (2, "SUBI")):
        mask = suite == suite_index
        if not mask.any():
            continue
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        count = indices.numel()
        immediate = torch.randint(0, 256, (count,), device=device, generator=generator)
        if opcode_name == "ADDI":
            destination = (desired_result[indices] - immediate) & 0xFF
            result = destination + immediate
            new_cf = result > 0xFF
        else:
            destination = (desired_result[indices] + immediate) & 0xFF
            result = destination - immediate
            new_cf = destination < immediate
        local_registers = registers[indices]
        local_registers[
            torch.arange(count, device=device), p1[indices]
        ] = destination
        local_x, local_y = encode_hard_arithmetic_xy(
            OPCODES[opcode_name],
            p1[indices],
            p2[indices],
            p3[indices],
            immediate,
            local_registers,
            flags[indices],
            result,
            new_cf,
        )
        x[indices] = local_x
        y[indices] = local_y
    return x, y


def dynamic_group_names() -> list[str]:
    return [OPCODE_NAMES[opcode] for opcode in range(NUM_OPCODES)] + list(
        STRUCTURAL_STATE_GROUPS
    )


def dynamic_group_metrics(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    if name in STRUCTURAL_STATE_GROUPS:
        return metrics["structural_groups"][name]
    return metrics["per_opcode"][name]


def default_dynamic_opcode_state() -> dict[str, Any]:
    names = dynamic_group_names()
    uniform_probability = 1.0 / len(names)
    return {
        "sampling_schema": 2,
        "audit_count": 0,
        "last_audit_step": None,
        "ema_loss_by_group": {},
        "ema_risk_by_group": {},
        "replay_probability_by_group": {
            name: uniform_probability for name in names
        },
        # 保留 opcode 子视图，兼容既有分析文件。
        "ema_loss_by_opcode": {},
        "ema_risk_by_opcode": {},
        "replay_probability_by_opcode": {
            OPCODE_NAMES[opcode]: uniform_probability
            for opcode in range(NUM_OPCODES)
        },
    }


def load_dynamic_opcode_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_dynamic_opcode_state()
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    state = default_dynamic_opcode_state()
    state.update(loaded)
    if int(loaded.get("sampling_schema", 1)) < 2:
        # 旧状态没有结构组风险，必须先重新审计，不能凭空继承 opcode 权重。
        state["sampling_schema"] = 2
        state["audit_count"] = 0
        state["last_audit_step"] = None
    return state


def update_dynamic_opcode_state(
    state: dict[str, Any],
    metrics: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    names = dynamic_group_names()
    previous_ema_loss = state.get("ema_loss_by_group", {})
    previous_ema_risk = state.get("ema_risk_by_group", {})
    alpha = Config.DYNAMIC_LOSS_EMA_ALPHA
    ema_loss_by_group: dict[str, float] = {}
    ema_risk_by_group: dict[str, float] = {}
    scores: list[float] = []

    for name in names:
        group_metrics = dynamic_group_metrics(metrics, name)
        current_loss = float(group_metrics["loss"])
        current_exact_error_rate = int(group_metrics["exact_errors"]) / max(
            int(group_metrics["samples"]), 1
        )
        current_risk = (
            current_loss
            + Config.DYNAMIC_EXACT_ERROR_RATE_WEIGHT * current_exact_error_rate
        )
        if name in previous_ema_loss:
            ema_loss = alpha * current_loss + (1.0 - alpha) * float(
                previous_ema_loss[name]
            )
        else:
            ema_loss = current_loss
        if name in previous_ema_risk:
            ema_risk = alpha * current_risk + (1.0 - alpha) * float(
                previous_ema_risk[name]
            )
        else:
            ema_risk = current_risk
        ema_loss_by_group[name] = ema_loss
        ema_risk_by_group[name] = ema_risk
        scores.append(max(ema_risk, 1e-30) ** Config.DYNAMIC_LOSS_POWER)

    score_sum = sum(scores)
    weighted = [score / score_sum for score in scores]
    uniform = 1.0 / len(names)
    mix = Config.DYNAMIC_UNIFORM_MIX
    probabilities = [
        (1.0 - mix) * probability + mix * uniform for probability in weighted
    ]

    return {
        "sampling_schema": 2,
        "audit_count": int(state.get("audit_count", 0)) + 1,
        "last_audit_step": step,
        "ema_loss_by_group": ema_loss_by_group,
        "ema_risk_by_group": ema_risk_by_group,
        "replay_probability_by_group": {
            name: probabilities[index] for index, name in enumerate(names)
        },
        "ema_loss_by_opcode": {
            name: ema_loss_by_group[name]
            for name in names
            if name not in STRUCTURAL_STATE_GROUPS
        },
        "ema_risk_by_opcode": {
            name: ema_risk_by_group[name]
            for name in names
            if name not in STRUCTURAL_STATE_GROUPS
        },
        "replay_probability_by_opcode": {
            OPCODE_NAMES[opcode]: probabilities[opcode]
            for opcode in range(NUM_OPCODES)
        },
    }


def generate_dynamic_opcode_replay_batch(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    names = dynamic_group_names()
    probability_map = state["replay_probability_by_group"]
    probabilities = torch.tensor(
        [float(probability_map[name]) for name in names],
        device=device,
        dtype=torch.float32,
    )
    selected_groups = torch.multinomial(
        probabilities,
        batch_size,
        replacement=True,
        generator=generator,
    )
    # 真实 opcode group 的索引恰好等于 opcode。虚拟状态组先放一个占位
    # opcode，随后在各自 mask 内改成均衡 opcode；整批只调用一次真值生成器。
    forced_opcodes = selected_groups.clamp(max=NUM_OPCODES - 1).to(torch.int64)
    requested_weights = torch.full(
        (batch_size,), -1, device=device, dtype=torch.int64
    )
    for offset, (name, hamming_range) in enumerate(
        STRUCTURAL_STATE_GROUPS.items()
    ):
        group_index = NUM_OPCODES + offset
        mask = selected_groups == group_index
        count = int(mask.sum().item())
        if count == 0:
            continue
        forced_opcodes[mask] = _balanced_opcodes(count, device, generator)
        requested_weights[mask] = _balanced_state_hamming_weights(
            count,
            hamming_range[0],
            hamming_range[1],
            device,
            generator,
        )

    x, y, _ = generate_batch(
        batch_size,
        device,
        generator,
        return_metadata=False,
        forced_opcodes=forced_opcodes,
        forced_state_hamming_weights=requested_weights,
    )
    return x, y


def append_dynamic_opcode_audit_csv(
    step: int,
    metrics: dict[str, Any],
    state: dict[str, Any],
    path: Path,
) -> None:
    fieldnames = [
        "step",
        "audit_count",
        "group",
        "group_type",
        "state_hamming_weight_range",
        "loss",
        "bit_errors",
        "exact_errors",
        "exact_error_rate",
        "ema_loss",
        "ema_risk",
        "replay_probability",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for name in dynamic_group_names():
            group_metrics = dynamic_group_metrics(metrics, name)
            hamming_range = STRUCTURAL_STATE_GROUPS.get(name)
            writer.writerow(
                {
                    "step": step,
                    "audit_count": state["audit_count"],
                    "group": name,
                    "group_type": "state_structure" if hamming_range else "opcode",
                    "state_hamming_weight_range": (
                        f"{hamming_range[0]}..{hamming_range[1]}"
                        if hamming_range
                        else ""
                    ),
                    "loss": group_metrics["loss"],
                    "bit_errors": group_metrics["bit_errors"],
                    "exact_errors": group_metrics["exact_errors"],
                    "exact_error_rate": int(group_metrics["exact_errors"])
                    / max(int(group_metrics["samples"]), 1),
                    "ema_loss": state["ema_loss_by_group"][name],
                    "ema_risk": state["ema_risk_by_group"][name],
                    "replay_probability": state["replay_probability_by_group"][name],
                }
            )


def run_dynamic_opcode_audit(
    model: nn.Module,
    device: torch.device,
    step: int,
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = time.time()
    # 每轮使用新的、但可复现的随机审计流，避免控制集逐渐变成固定训练集。
    audit_seed = Config.DYNAMIC_AUDIT_SEED + int(state.get("audit_count", 0))
    metrics = evaluate(
        model,
        device,
        Config.DYNAMIC_AUDIT_SAMPLES,
        Config.DYNAMIC_AUDIT_BATCH_SIZE,
        validation_seed=audit_seed,
        structural_samples_per_group=Config.DYNAMIC_STRUCTURAL_AUDIT_SAMPLES_PER_GROUP,
    )
    metrics["step"] = step
    metrics["audit_seed"] = audit_seed
    metrics["audit_seconds"] = time.time() - started_at
    new_state = update_dynamic_opcode_state(state, metrics, step)

    ranked_names = sorted(
        dynamic_group_names(),
        key=lambda name: float(new_state["ema_risk_by_group"][name]),
        reverse=True,
    )
    print(
        f"\n[动态 group 审计 step={step:,}] "
        f"seed={audit_seed} | samples={metrics['samples']:,} | "
        f"loss={metrics['loss']:.15e} | "
        f"exact_errors={metrics['output_exact_errors']:,} | "
        f"耗时={metrics['audit_seconds']:.1f}s"
    )
    print("下一阶段 replay 权重最高的真实/虚拟 op group：")
    for name in ranked_names[:10]:
        group_metrics = dynamic_group_metrics(metrics, name)
        probability = float(new_state["replay_probability_by_group"][name])
        exact_error_rate = int(group_metrics["exact_errors"]) / max(
            int(group_metrics["samples"]), 1
        )
        print(
            f"  {name:<13} loss={float(group_metrics['loss']):.15e} | "
            f"exact_errors={int(group_metrics['exact_errors']):,} "
            f"({exact_error_rate:.3e}) | "
            f"EMA_risk={float(new_state['ema_risk_by_group'][name]):.15e} | "
            f"dynamic_replay={probability:.4%}"
        )

    payload = {
        "step": step,
        "audit_count": new_state["audit_count"],
        "audit_seconds": metrics["audit_seconds"],
        "metrics": metrics,
        "sampling_state": new_state,
    }
    append_jsonl(payload, Config.RESULT_DIR / "dynamic_opcode_audit_history.jsonl")
    append_dynamic_opcode_audit_csv(
        step,
        metrics,
        new_state,
        Config.RESULT_DIR / "dynamic_group_audit_history_v2.csv",
    )
    atomic_json_save(
        payload,
        Config.RESULT_DIR / "dynamic_opcode_audit_latest.json",
    )
    atomic_json_save(
        new_state,
        Config.RESULT_DIR / "dynamic_opcode_sampling_state.json",
    )
    return metrics, new_state


# =============================================================================
# 验证与保存
# =============================================================================


def accuracy_from_errors(errors: int, total: int) -> float:
    return 1.0 - errors / total if total else float("nan")


@torch.inference_mode()
def evaluate_structural_state_group(
    model: nn.Module,
    device: torch.device,
    samples: int,
    batch_size: int,
    seed: int,
    minimum_hamming_weight: int,
    maximum_hamming_weight: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    total_loss = 0.0
    total_bit_errors = 0
    total_exact_errors = 0
    total_samples = 0

    remaining = samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        weights = _balanced_state_hamming_weights(
            current_batch,
            minimum_hamming_weight,
            maximum_hamming_weight,
            device,
            generator,
        )
        forced_opcodes = _balanced_opcodes(current_batch, device, generator)
        x, y, _ = generate_batch(
            current_batch,
            device,
            generator,
            return_metadata=False,
            forced_opcodes=forced_opcodes,
            forced_state_hamming_weights=weights,
        )
        logits = model(x)
        element_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        errors = (logits >= 0) != y.to(torch.bool)
        total_loss += float(element_loss.sum().item())
        total_bit_errors += int(errors.sum().item())
        total_exact_errors += int(errors.any(dim=1).sum().item())
        total_samples += current_batch
        remaining -= current_batch

    return {
        "samples": total_samples,
        "loss": total_loss / (total_samples * OUTPUT_BITS),
        "bit_errors": total_bit_errors,
        "bit_accuracy": accuracy_from_errors(
            total_bit_errors, total_samples * OUTPUT_BITS
        ),
        "exact_errors": total_exact_errors,
        "exact_accuracy": accuracy_from_errors(total_exact_errors, total_samples),
        "minimum_hamming_weight": minimum_hamming_weight,
        "maximum_hamming_weight": maximum_hamming_weight,
        "seed": seed,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    device: torch.device,
    validation_samples: int,
    validation_batch_size: int,
    validation_seed: int | None = None,
    structural_samples_per_group: int | None = None,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device.type)
    generator.manual_seed(
        Config.VALIDATION_SEED if validation_seed is None else validation_seed
    )

    total_loss = 0.0
    total_state_loss = 0.0
    total_control_loss = 0.0
    total_samples = 0
    total_bit_errors = 0
    total_state_bit_errors = 0
    total_control_bit_errors = 0
    total_output_exact_errors = 0
    total_state_exact_errors = 0
    total_control_exact_errors = 0

    opcode_stats: dict[int, dict[str, float | int]] = {
        opcode: {
            "samples": 0,
            "loss_sum": 0.0,
            "bit_errors": 0,
            "exact_errors": 0,
        }
        for opcode in range(NUM_OPCODES)
    }

    remaining = validation_samples
    while remaining > 0:
        current_batch = min(validation_batch_size, remaining)
        x, y, metadata = generate_batch(
            current_batch, device, generator, return_metadata=True
        )
        assert metadata is not None
        logits = model(x)
        element_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        predictions = logits >= 0
        truth = y.to(torch.bool)
        errors = predictions != truth

        total_loss += float(element_loss.sum().item())
        total_state_loss += float(element_loss[:, :STATE_BITS].sum().item())
        total_control_loss += float(element_loss[:, STATE_BITS:].sum().item())
        total_samples += current_batch
        total_bit_errors += int(errors.sum().item())
        total_state_bit_errors += int(errors[:, :STATE_BITS].sum().item())
        total_control_bit_errors += int(errors[:, STATE_BITS:].sum().item())
        total_output_exact_errors += int(errors.any(dim=1).sum().item())
        total_state_exact_errors += int(errors[:, :STATE_BITS].any(dim=1).sum().item())
        total_control_exact_errors += int(errors[:, STATE_BITS:].any(dim=1).sum().item())

        sample_loss = element_loss.sum(dim=1)
        sample_exact_error = errors.any(dim=1)
        for opcode in range(NUM_OPCODES):
            mask = metadata.opcodes == opcode
            count = int(mask.sum().item())
            if count == 0:
                continue
            stats = opcode_stats[opcode]
            stats["samples"] = int(stats["samples"]) + count
            stats["loss_sum"] = float(stats["loss_sum"]) + float(
                sample_loss[mask].sum().item()
            )
            stats["bit_errors"] = int(stats["bit_errors"]) + int(
                errors[mask].sum().item()
            )
            stats["exact_errors"] = int(stats["exact_errors"]) + int(
                sample_exact_error[mask].sum().item()
            )

        remaining -= current_batch

    opcode_metrics: dict[str, dict[str, float | int]] = {}
    for opcode, stats in opcode_stats.items():
        samples = int(stats["samples"])
        bits = samples * OUTPUT_BITS
        name = OPCODE_NAMES[opcode]
        opcode_metrics[name] = {
            "opcode": opcode,
            "samples": samples,
            "loss": float(stats["loss_sum"]) / bits,
            "bit_errors": int(stats["bit_errors"]),
            "bit_accuracy": accuracy_from_errors(int(stats["bit_errors"]), bits),
            "exact_errors": int(stats["exact_errors"]),
            "exact_accuracy": accuracy_from_errors(int(stats["exact_errors"]), samples),
        }

    worst_by_loss = max(opcode_metrics, key=lambda name: float(opcode_metrics[name]["loss"]))
    worst_by_exact = min(
        opcode_metrics,
        key=lambda name: float(opcode_metrics[name]["exact_accuracy"]),
    )

    base_seed = Config.VALIDATION_SEED if validation_seed is None else validation_seed
    if structural_samples_per_group is None:
        structural_samples_per_group = Config.STRUCTURAL_VALIDATION_SAMPLES_PER_GROUP
    structural_metrics: dict[str, dict[str, Any]] = {}
    if structural_samples_per_group > 0:
        for index, (name, hamming_range) in enumerate(
            STRUCTURAL_STATE_GROUPS.items()
        ):
            structural_metrics[name] = evaluate_structural_state_group(
                model,
                device,
                structural_samples_per_group,
                validation_batch_size,
                base_seed + 100_000 + index,
                hamming_range[0],
                hamming_range[1],
            )

    model.train()
    return {
        "samples": total_samples,
        "loss": total_loss / (total_samples * OUTPUT_BITS),
        "state_loss": total_state_loss / (total_samples * STATE_BITS),
        "control_loss": total_control_loss / (total_samples * CONTROL_BITS),
        "bit_errors": total_bit_errors,
        "bit_accuracy": accuracy_from_errors(
            total_bit_errors, total_samples * OUTPUT_BITS
        ),
        "state_bit_errors": total_state_bit_errors,
        "state_bit_accuracy": accuracy_from_errors(
            total_state_bit_errors, total_samples * STATE_BITS
        ),
        "control_bit_errors": total_control_bit_errors,
        "control_bit_accuracy": accuracy_from_errors(
            total_control_bit_errors, total_samples * CONTROL_BITS
        ),
        "output_exact_errors": total_output_exact_errors,
        "output_exact_accuracy": accuracy_from_errors(
            total_output_exact_errors, total_samples
        ),
        "state_exact_errors": total_state_exact_errors,
        "state_exact_accuracy": accuracy_from_errors(
            total_state_exact_errors, total_samples
        ),
        "control_exact_errors": total_control_exact_errors,
        "control_exact_accuracy": accuracy_from_errors(
            total_control_exact_errors, total_samples
        ),
        "worst_opcode_by_loss": worst_by_loss,
        "worst_opcode_by_exact": worst_by_exact,
        "per_opcode": opcode_metrics,
        "structural_groups": structural_metrics,
    }


def alias_pattern_name(code: int) -> str:
    relations = []
    if code & 1:
        relations.append("dst=src1")
    if code & 2:
        relations.append("dst=src2")
    if code & 4:
        relations.append("src1=src2")
    return "+".join(relations) if relations else "all_distinct"


def decode_output_vector(bits: list[int]) -> dict[str, Any]:
    flags = {"ZF": bits[0], "GF": bits[1], "CF": bits[2]}
    registers: list[int] = []
    for register in range(NUM_REGISTERS):
        start = NUM_FLAGS + register * REGISTER_BITS
        value = 0
        for bit in bits[start : start + REGISTER_BITS]:
            value = (value << 1) | int(bit)
        registers.append(value)
    controls = {
        "memory_read": bits[STATE_BITS + CTRL_MEMORY_READ],
        "memory_write": bits[STATE_BITS + CTRL_MEMORY_WRITE],
        "branch_taken": bits[STATE_BITS + CTRL_BRANCH_TAKEN],
        "halt": bits[STATE_BITS + CTRL_HALT],
    }
    return {"flags": flags, "registers": registers, "controls": controls}


def add_bincount(target: list[int], values: torch.Tensor) -> None:
    counts = values.detach().cpu().tolist()
    for index, count in enumerate(counts):
        if index < len(target):
            target[index] += int(count)


def group_rows(
    group_type: str,
    totals: list[int],
    errors: list[int],
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, total in enumerate(totals):
        if total == 0:
            continue
        label = labels[index] if labels is not None else str(index)
        rows.append(
            {
                "group_type": group_type,
                "group_value": label,
                "samples": total,
                "exact_errors": errors[index],
                "exact_error_rate": errors[index] / total,
            }
        )
    return rows


@torch.inference_mode()
def analyze_model_errors(
    model: nn.Module,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    """在新的超大固定样本流上定位模型的系统性错误。"""

    analysis_dir = Config.RESULT_DIR / "large_validation_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    error_cases_path = analysis_dir / "error_cases.jsonl"
    error_cases_handle = error_cases_path.open("w", encoding="utf-8")

    generator = torch.Generator(device=device.type)
    generator.manual_seed(Config.ANALYSIS_SEED)
    model.eval()

    total_samples = 0
    total_loss = 0.0
    total_bit_errors = 0
    total_exact_errors = 0
    total_state_bit_errors = 0
    total_control_bit_errors = 0
    output_bit_errors = [0] * OUTPUT_BITS
    stored_error_cases = 0

    opcode_samples = [0] * NUM_OPCODES
    opcode_loss_sums = [0.0] * NUM_OPCODES
    opcode_bit_errors = [0] * NUM_OPCODES
    opcode_exact_errors = [0] * NUM_OPCODES

    alias_totals = [0] * 8
    alias_errors = [0] * 8
    carry_totals = [0] * 2
    carry_errors = [0] * 2

    sbc_carry_totals = [0] * 2
    sbc_carry_errors = [0] * 2
    sbc_borrow_totals = [0] * 2
    sbc_borrow_errors = [0] * 2
    sbc_alias_totals = [0] * 8
    sbc_alias_errors = [0] * 8
    sbc_distance_totals = [0] * 257
    sbc_distance_errors = [0] * 257
    sbc_result_totals = [0] * 256
    sbc_result_errors = [0] * 256

    started_at = time.time()
    next_progress = Config.ANALYSIS_PROGRESS_INTERVAL
    remaining = Config.ANALYSIS_SAMPLES

    try:
        while remaining > 0:
            current_batch = min(Config.ANALYSIS_BATCH_SIZE, remaining)
            x, y, metadata = generate_batch(
                current_batch, device, generator, return_metadata=True
            )
            assert metadata is not None
            logits = model(x)
            element_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            predictions = logits >= 0
            truth = y.to(torch.bool)
            errors = predictions != truth
            sample_exact_error = errors.any(dim=1)
            sample_bit_errors = errors.sum(dim=1)
            sample_loss = element_loss.sum(dim=1)

            total_loss += float(element_loss.sum().item())
            total_bit_errors += int(errors.sum().item())
            total_exact_errors += int(sample_exact_error.sum().item())
            total_state_bit_errors += int(errors[:, :STATE_BITS].sum().item())
            total_control_bit_errors += int(errors[:, STATE_BITS:].sum().item())
            add_bincount(output_bit_errors, errors.sum(dim=0).to(torch.int64))

            for opcode in range(NUM_OPCODES):
                mask = metadata.opcodes == opcode
                count = int(mask.sum().item())
                opcode_samples[opcode] += count
                opcode_loss_sums[opcode] += float(sample_loss[mask].sum().item())
                opcode_bit_errors[opcode] += int(sample_bit_errors[mask].sum().item())
                opcode_exact_errors[opcode] += int(sample_exact_error[mask].sum().item())

            alias_code = (
                (metadata.p1 == metadata.p2).to(torch.int64)
                + 2 * (metadata.p1 == metadata.p3).to(torch.int64)
                + 4 * (metadata.p2 == metadata.p3).to(torch.int64)
            )
            input_carry = metadata.flags[:, FLAG_CF]
            add_bincount(
                alias_totals,
                torch.bincount(alias_code, minlength=8),
            )
            add_bincount(
                alias_errors,
                torch.bincount(alias_code[sample_exact_error], minlength=8),
            )
            add_bincount(
                carry_totals,
                torch.bincount(input_carry, minlength=2),
            )
            add_bincount(
                carry_errors,
                torch.bincount(input_carry[sample_exact_error], minlength=2),
            )

            rows = torch.arange(current_batch, device=device)
            value1 = metadata.registers[rows, metadata.p2]
            value2 = metadata.registers[rows, metadata.p3]
            sbc_mask = metadata.opcodes == OPCODES["SBC"]
            sbc_error_mask = sbc_mask & sample_exact_error
            sbc_raw_result = value1 - value2 - input_carry
            sbc_result = sbc_raw_result & 0xFF
            sbc_borrow = (sbc_raw_result < 0).to(torch.int64)
            sbc_distance = torch.abs(value1 - (value2 + input_carry))

            add_bincount(
                sbc_carry_totals,
                torch.bincount(input_carry[sbc_mask], minlength=2),
            )
            add_bincount(
                sbc_carry_errors,
                torch.bincount(input_carry[sbc_error_mask], minlength=2),
            )
            add_bincount(
                sbc_borrow_totals,
                torch.bincount(sbc_borrow[sbc_mask], minlength=2),
            )
            add_bincount(
                sbc_borrow_errors,
                torch.bincount(sbc_borrow[sbc_error_mask], minlength=2),
            )
            add_bincount(
                sbc_alias_totals,
                torch.bincount(alias_code[sbc_mask], minlength=8),
            )
            add_bincount(
                sbc_alias_errors,
                torch.bincount(alias_code[sbc_error_mask], minlength=8),
            )
            add_bincount(
                sbc_distance_totals,
                torch.bincount(sbc_distance[sbc_mask], minlength=257),
            )
            add_bincount(
                sbc_distance_errors,
                torch.bincount(sbc_distance[sbc_error_mask], minlength=257),
            )
            add_bincount(
                sbc_result_totals,
                torch.bincount(sbc_result[sbc_mask], minlength=256),
            )
            add_bincount(
                sbc_result_errors,
                torch.bincount(sbc_result[sbc_error_mask], minlength=256),
            )

            available_slots = Config.ANALYSIS_MAX_SAVED_ERROR_CASES - stored_error_cases
            if available_slots > 0:
                error_rows = torch.nonzero(sample_exact_error, as_tuple=False).squeeze(1)
                selected = error_rows[:available_slots]
                if selected.numel() > 0:
                    selected_cpu = selected.detach().cpu().tolist()
                    selected_logits = logits[selected].detach().cpu()
                    selected_probabilities = torch.sigmoid(selected_logits)
                    selected_targets = truth[selected].detach().cpu().to(torch.int64)
                    selected_predictions = predictions[selected].detach().cpu().to(torch.int64)
                    selected_opcodes = metadata.opcodes[selected].detach().cpu()
                    selected_instructions = metadata.instructions[selected].detach().cpu()
                    selected_registers = metadata.registers[selected].detach().cpu()
                    selected_flags = metadata.flags[selected].detach().cpu()
                    selected_p1 = metadata.p1[selected].detach().cpu()
                    selected_p2 = metadata.p2[selected].detach().cpu()
                    selected_p3 = metadata.p3[selected].detach().cpu()
                    selected_immediate = metadata.immediate[selected].detach().cpu()
                    selected_offset = metadata.offset[selected].detach().cpu()

                    for stored_index, local_index in enumerate(selected_cpu):
                        opcode = int(selected_opcodes[stored_index])
                        p1 = int(selected_p1[stored_index])
                        p2 = int(selected_p2[stored_index])
                        p3 = int(selected_p3[stored_index])
                        registers = selected_registers[stored_index].tolist()
                        flags = selected_flags[stored_index].tolist()
                        target_bits = selected_targets[stored_index].tolist()
                        predicted_bits = selected_predictions[stored_index].tolist()
                        wrong_indices = [
                            index
                            for index, (target, predicted) in enumerate(
                                zip(target_bits, predicted_bits)
                            )
                            if target != predicted
                        ]
                        wrong_outputs = []
                        for output_index in wrong_indices:
                            wrong_outputs.append(
                                {
                                    "index": output_index,
                                    "name": output_bit_name(output_index),
                                    "target": target_bits[output_index],
                                    "prediction": predicted_bits[output_index],
                                    "logit": float(
                                        selected_logits[stored_index, output_index]
                                    ),
                                    "probability": float(
                                        selected_probabilities[stored_index, output_index]
                                    ),
                                }
                            )

                        value_a = registers[p2]
                        value_b = registers[p3]
                        case: dict[str, Any] = {
                            "sample_index": total_samples + local_index,
                            "opcode": opcode,
                            "opcode_name": OPCODE_NAMES[opcode],
                            "instruction_int": int(selected_instructions[stored_index]),
                            "instruction_hex": f"0x{int(selected_instructions[stored_index]):04X}",
                            "instruction_binary": f"{int(selected_instructions[stored_index]):016b}",
                            "p1_dst": p1,
                            "p2_src1": p2,
                            "p3_src2": p3,
                            "immediate": (
                                int(selected_immediate[stored_index])
                                if opcode in RI8_OPCODES
                                else None
                            ),
                            "offset": (
                                int(selected_offset[stored_index])
                                if opcode in RELATIVE_BRANCH_OPCODES
                                else None
                            ),
                            "input_registers": registers,
                            "input_flags": {"ZF": flags[0], "GF": flags[1], "CF": flags[2]},
                            "alias_pattern": alias_pattern_name(
                                int(
                                    (p1 == p2)
                                    + 2 * (p1 == p3)
                                    + 4 * (p2 == p3)
                                )
                            ),
                            "operands": {
                                "src1": value_a,
                                "src2": value_b,
                                "carry_in": flags[FLAG_CF],
                            },
                            "target": decode_output_vector(target_bits),
                            "prediction": decode_output_vector(predicted_bits),
                            "wrong_outputs": wrong_outputs,
                        }
                        if opcode == OPCODES["SBC"]:
                            raw_result = value_a - value_b - flags[FLAG_CF]
                            case["sbc_diagnostics"] = {
                                "raw_result": raw_result,
                                "uint8_result": raw_result & 0xFF,
                                "borrow": int(raw_result < 0),
                                "distance_to_borrow_boundary": abs(
                                    value_a - (value_b + flags[FLAG_CF])
                                ),
                            }
                        error_cases_handle.write(
                            json.dumps(case, ensure_ascii=False) + "\n"
                        )
                        stored_error_cases += 1

            total_samples += current_batch
            remaining -= current_batch
            if total_samples >= next_progress or remaining == 0:
                elapsed = time.time() - started_at
                print(
                    f"分析进度：{total_samples:,}/{Config.ANALYSIS_SAMPLES:,} | "
                    f"exact_errors={total_exact_errors:,} | "
                    f"bit_errors={total_bit_errors:,} | "
                    f"samples/s={total_samples / max(elapsed, 1e-9):,.1f}"
                )
                while next_progress <= total_samples:
                    next_progress += Config.ANALYSIS_PROGRESS_INTERVAL
                error_cases_handle.flush()
    finally:
        error_cases_handle.close()

    opcode_metrics = []
    for opcode in range(NUM_OPCODES):
        samples = opcode_samples[opcode]
        opcode_metrics.append(
            {
                "opcode": opcode,
                "opcode_name": OPCODE_NAMES[opcode],
                "samples": samples,
                "loss": opcode_loss_sums[opcode] / (samples * OUTPUT_BITS),
                "bit_errors": opcode_bit_errors[opcode],
                "bit_error_rate": opcode_bit_errors[opcode] / (samples * OUTPUT_BITS),
                "exact_errors": opcode_exact_errors[opcode],
                "exact_error_rate": opcode_exact_errors[opcode] / samples,
                "exact_accuracy": accuracy_from_errors(opcode_exact_errors[opcode], samples),
            }
        )

    alias_labels = [alias_pattern_name(index) for index in range(8)]
    groups = []
    groups.extend(group_rows("all.alias_pattern", alias_totals, alias_errors, alias_labels))
    groups.extend(group_rows("all.input_CF", carry_totals, carry_errors, ["0", "1"]))
    groups.extend(
        group_rows("SBC.alias_pattern", sbc_alias_totals, sbc_alias_errors, alias_labels)
    )
    groups.extend(group_rows("SBC.input_CF", sbc_carry_totals, sbc_carry_errors, ["0", "1"]))
    groups.extend(group_rows("SBC.borrow", sbc_borrow_totals, sbc_borrow_errors, ["0", "1"]))
    groups.extend(
        group_rows(
            "SBC.distance_to_borrow_boundary",
            sbc_distance_totals,
            sbc_distance_errors,
        )
    )
    groups.extend(group_rows("SBC.uint8_result", sbc_result_totals, sbc_result_errors))

    bit_metrics = [
        {
            "index": index,
            "name": output_bit_name(index),
            "errors": errors,
            "error_rate": errors / total_samples,
        }
        for index, errors in enumerate(output_bit_errors)
    ]
    summary = {
        "model_path": str(model_path),
        "analysis_seed": Config.ANALYSIS_SEED,
        "samples": total_samples,
        "output_bits_per_sample": OUTPUT_BITS,
        "loss": total_loss / (total_samples * OUTPUT_BITS),
        "bit_errors": total_bit_errors,
        "bit_error_rate": total_bit_errors / (total_samples * OUTPUT_BITS),
        "bit_accuracy": accuracy_from_errors(
            total_bit_errors, total_samples * OUTPUT_BITS
        ),
        "state_bit_errors": total_state_bit_errors,
        "control_bit_errors": total_control_bit_errors,
        "exact_errors": total_exact_errors,
        "exact_error_rate": total_exact_errors / total_samples,
        "exact_accuracy": accuracy_from_errors(total_exact_errors, total_samples),
        "saved_error_cases": stored_error_cases,
        "error_cases_truncated": total_exact_errors > stored_error_cases,
        "elapsed_seconds": time.time() - started_at,
        "per_opcode": opcode_metrics,
        "per_output_bit": bit_metrics,
        "groups": groups,
    }

    atomic_json_save(summary, analysis_dir / "summary.json")
    with (analysis_dir / "opcode_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(opcode_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(opcode_metrics)
    with (analysis_dir / "error_groups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0].keys()))
        writer.writeheader()
        writer.writerows(groups)

    print("\n=== 大规模验证结果 ===")
    print(
        f"samples={total_samples:,} | loss={summary['loss']:.15e} | "
        f"bit_errors={total_bit_errors:,} | bit_acc={summary['bit_accuracy']:.12f}"
    )
    print(
        f"exact_errors={total_exact_errors:,} | "
        f"exact_acc={summary['exact_accuracy']:.12f}"
    )
    failing_opcodes = [row for row in opcode_metrics if row["exact_errors"] > 0]
    if failing_opcodes:
        print("有错误的 opcode：")
        for row in failing_opcodes:
            print(
                f"  {row['opcode_name']:8s} exact_errors={row['exact_errors']:,}/"
                f"{row['samples']:,}, bit_errors={row['bit_errors']:,}, "
                f"loss={row['loss']:.15e}"
            )
    else:
        print("本轮大规模验证没有发现错误。")

    sbc_error_groups = [
        row
        for row in groups
        if row["group_type"].startswith("SBC.") and row["exact_errors"] > 0
    ]
    sbc_error_groups.sort(key=lambda row: row["exact_error_rate"], reverse=True)
    if sbc_error_groups:
        print("SBC 错误率最高的分组（前 12 项）：")
        for row in sbc_error_groups[:12]:
            print(
                f"  {row['group_type']}={row['group_value']}: "
                f"{row['exact_errors']}/{row['samples']} "
                f"({row['exact_error_rate']:.12e})"
            )
    print(f"错误明细：{error_cases_path}")
    print(f"汇总报告：{analysis_dir / 'summary.json'}")
    return summary


def config_as_dict() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in dir(Config):
        if not name.isupper():
            continue
        value = getattr(Config, name)
        result[name] = str(value) if isinstance(value, Path) else value
    result.update(
        {
            "INPUT_BITS": INPUT_BITS,
            "OUTPUT_BITS": OUTPUT_BITS,
            "STATE_BITS": STATE_BITS,
            "CONTROL_BITS": CONTROL_BITS,
            "TOTAL_HIDDEN_LINEAR_LAYERS": Config.HIDDEN_LAYERS + 1,
        }
    )
    return result


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def append_jsonl(payload: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


CSV_FIELDS = [
    "step",
    "elapsed_seconds",
    "learning_rate",
    "train_loss",
    "val_loss",
    "val_state_loss",
    "val_control_loss",
    "val_bit_accuracy",
    "val_bit_errors",
    "val_state_bit_accuracy",
    "val_state_bit_errors",
    "val_control_bit_accuracy",
    "val_control_bit_errors",
    "val_output_exact_accuracy",
    "val_output_exact_errors",
    "val_state_exact_accuracy",
    "val_state_exact_errors",
    "val_control_exact_accuracy",
    "val_control_exact_errors",
    "worst_opcode_by_loss",
    "worst_opcode_by_exact",
    "best_val_loss",
]


def append_csv(payload: dict[str, Any], path: Path) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: payload.get(field) for field in CSV_FIELDS})


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    train_generator: torch.Generator,
    step: int,
    best_val_loss: float,
    latest_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_generator_state": train_generator.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_rng_state": random.getstate(),
        "config": config_as_dict(),
        "isa": ISA_DESCRIPTION,
        "latest_metrics": latest_metrics,
    }


def load_checkpoint_if_available(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    train_generator: torch.Generator,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[int, float]:
    if not Config.AUTO_RESUME or not checkpoint_path.exists():
        return 0, math.inf

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("train_generator_state") is not None:
        train_generator.set_state(checkpoint["train_generator_state"].cpu())
    if checkpoint.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["torch_cuda_rng_state_all"]]
        )
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])

    step = int(checkpoint.get("step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    print(
        f"从 {checkpoint_path} 恢复：step={step:,}, "
        f"best_val_loss={best_val_loss:.15e}"
    )
    return step, best_val_loss


def print_validation(step: int, metrics: dict[str, Any], best_val_loss: float) -> None:
    worst_loss_name = str(metrics["worst_opcode_by_loss"])
    worst_exact_name = str(metrics["worst_opcode_by_exact"])
    worst_loss = metrics["per_opcode"][worst_loss_name]
    worst_exact = metrics["per_opcode"][worst_exact_name]
    print(
        f"\n[验证 step={step:,}] "
        f"loss={metrics['loss']:.15e} | "
        f"state_loss={metrics['state_loss']:.15e} | "
        f"control_loss={metrics['control_loss']:.15e}"
    )
    print(
        f"bit_acc={metrics['bit_accuracy']:.12f} "
        f"({metrics['bit_errors']:,}/{metrics['samples'] * OUTPUT_BITS:,} bit errors) | "
        f"state_bit={metrics['state_bit_accuracy']:.12f} | "
        f"control_bit={metrics['control_bit_accuracy']:.12f}"
    )
    print(
        f"output_exact={metrics['output_exact_accuracy']:.12f} "
        f"({metrics['output_exact_errors']:,}/{metrics['samples']:,} sample errors) | "
        f"state_exact={metrics['state_exact_accuracy']:.12f} | "
        f"control_exact={metrics['control_exact_accuracy']:.12f}"
    )
    print(
        f"worst_loss={worst_loss_name}:{float(worst_loss['loss']):.15e} | "
        f"worst_exact={worst_exact_name}:{float(worst_exact['exact_accuracy']):.12f} | "
        f"best={best_val_loss:.15e}"
    )
    for name, group_metrics in metrics.get("structural_groups", {}).items():
        print(
            f"  virtual_op={name:<13} "
            f"HW={group_metrics['minimum_hamming_weight']}.."
            f"{group_metrics['maximum_hamming_weight']} | "
            f"loss={float(group_metrics['loss']):.15e} | "
            f"bit_errors={int(group_metrics['bit_errors']):,} | "
            f"exact_errors={int(group_metrics['exact_errors']):,}/"
            f"{int(group_metrics['samples']):,}"
        )


def balanced_checkpoint_key(metrics: dict[str, Any]) -> tuple[int, float, float]:
    """先保 IID/结构组离散正确性，再压最弱 group，最后优化 IID 均值。"""
    all_groups = list(metrics["per_opcode"].values()) + list(
        metrics.get("structural_groups", {}).values()
    )
    worst_loss = max(float(group_metrics["loss"]) for group_metrics in all_groups)
    structural_errors = sum(
        int(group_metrics["exact_errors"])
        for group_metrics in metrics.get("structural_groups", {}).values()
    )
    return (
        int(metrics["output_exact_errors"]) + structural_errors,
        worst_loss,
        float(metrics["loss"]),
    )


# =============================================================================
# 训练
# =============================================================================


def main() -> None:
    if Config.SMOKE_TEST:
        Config.BATCH_SIZE = 128
        Config.VALIDATION_SAMPLES = 1024
        Config.VALIDATION_BATCH_SIZE = 256
        Config.MAX_STEPS = 3
        Config.LOG_INTERVAL = 1
        Config.EVAL_INTERVAL = 1
        Config.CHECKPOINT_INTERVAL = 1
        Config.DYNAMIC_AUDIT_INTERVAL = 2
        Config.DYNAMIC_AUDIT_SAMPLES = 1024
        Config.DYNAMIC_AUDIT_BATCH_SIZE = 256
        Config.STRUCTURAL_VALIDATION_SAMPLES_PER_GROUP = 256
        Config.DYNAMIC_STRUCTURAL_AUDIT_SAMPLES_PER_GROUP = 256
        Config.RESULT_DIR = Config.RESULT_DIR.with_name(Config.RESULT_DIR.name + "_smoke")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32

    print(f"设备：{device}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"结果目录：{Config.RESULT_DIR}")
    print(
        f"模型：width={Config.HIDDEN_SIZE}, "
        f"hidden_linear_layers={Config.HIDDEN_LAYERS + 1}, dropout={Config.DROPOUT}"
    )
    print(
        f"I/O：{INPUT_BITS} -> {OUTPUT_BITS}; batch={Config.BATCH_SIZE:,}; "
        f"max_steps={Config.MAX_STEPS}"
    )

    atomic_json_save(config_as_dict(), Config.RESULT_DIR / "config.json")
    atomic_json_save(ISA_DESCRIPTION, Config.RESULT_DIR / "isa.json")
    run_semantics_self_test(device)

    model = NeuralCPUCore().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"模型参数：{parameter_count:,}")

    latest_checkpoint_path = Config.RESULT_DIR / "latest_checkpoint.pt"
    best_model_path = Config.RESULT_DIR / "best_model.pt"
    best_balanced_model_path = Config.RESULT_DIR / "best_balanced_model.pt"
    dynamic_state_path = Config.RESULT_DIR / "dynamic_opcode_sampling_state.json"
    if Config.ANALYZE_BEST_MODEL_ONLY:
        analysis_model_path = Config.RESULT_DIR / Config.ANALYSIS_MODEL_FILENAME
        if not analysis_model_path.exists():
            analysis_model_path = best_model_path
        if not analysis_model_path.exists():
            raise FileNotFoundError(f"找不到最佳模型：{analysis_model_path}")
        best_checkpoint = torch.load(
            analysis_model_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print(
            f"加载诊断模型：{analysis_model_path.name} | "
            f"step={int(best_checkpoint.get('step', 0)):,}, "
            f"val_loss={float(best_checkpoint.get('val_loss', math.nan)):.15e}"
        )
        print(
            f"开始大规模诊断：samples={Config.ANALYSIS_SAMPLES:,}, "
            f"batch={Config.ANALYSIS_BATCH_SIZE:,}"
        )
        analyze_model_errors(model, device, analysis_model_path)
        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.LR_REDUCE_FACTOR,
            patience=Config.LR_PATIENCE_EVALS,
            min_lr=Config.MIN_LEARNING_RATE,
        )

    train_generator = torch.Generator(device=device.type)
    train_generator.manual_seed(Config.SEED + 1)

    first_dynamic_run_from_best = (
        Config.DYNAMIC_OPCODE_REPLAY
        and Config.DYNAMIC_FIRST_RUN_FROM_BEST
        and not dynamic_state_path.exists()
        and best_model_path.exists()
    )
    if first_dynamic_run_from_best:
        initial_best = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(initial_best["model_state_dict"])
        step = int(initial_best.get("step", 0))
        best_val_loss = float(
            initial_best.get("val_loss", initial_best.get("best_val_loss", math.inf))
        )
        print(
            "首次启用动态 opcode replay：从现有 best_model.pt 重新出发，"
            f"step={step:,} | best_val_loss={best_val_loss:.15e}"
        )
        print(f"优化器已重置，固定 lr={Config.LEARNING_RATE:.9e}")
    elif (
        Config.HARD_REPLAY_MODE
        and Config.AUTO_RESUME
        and not latest_checkpoint_path.exists()
    ):
        if not Config.PRETRAINED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"找不到用于继续训练的最佳模型：{Config.PRETRAINED_MODEL_PATH}"
            )
        pretrained = torch.load(
            Config.PRETRAINED_MODEL_PATH,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(pretrained["model_state_dict"])
        step = int(pretrained.get("step", 0))
        best_val_loss = float(
            pretrained.get("val_loss", pretrained.get("best_val_loss", math.inf))
        )
        print(
            f"从旧最佳权重继续训练：{Config.PRETRAINED_MODEL_PATH} | "
            f"step={step:,} | best_val_loss={best_val_loss:.15e}"
        )
        print(
            f"优化器已重置，固定 lr={Config.LEARNING_RATE:.9e}；"
            f"hard 比例={Config.HARD_REPLAY_FRACTION:.3f}"
        )
        atomic_torch_save(
            {
                "format_version": 1,
                "step": step,
                "val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "config": config_as_dict(),
                "isa": ISA_DESCRIPTION,
                "metrics": pretrained.get("metrics"),
                "source_model": str(Config.PRETRAINED_MODEL_PATH),
            },
            best_model_path,
        )
    else:
        step, best_val_loss = load_checkpoint_if_available(
            model,
            optimizer,
            scheduler,
            train_generator,
            device,
            latest_checkpoint_path,
        )

    if best_balanced_model_path.exists():
        balanced_checkpoint = torch.load(
            best_balanced_model_path,
            map_location="cpu",
            weights_only=False,
        )
        stored_key = balanced_checkpoint.get("balanced_key")
        if balanced_checkpoint.get("balanced_schema") == 2 and stored_key is not None:
            best_balanced_key = tuple(stored_key)
            print(
                "已有 balanced 最佳模型（含结构状态验证）："
                f"errors={best_balanced_key[0]}, "
                f"worst_loss={float(best_balanced_key[1]):.15e}, "
                f"mean_loss={float(best_balanced_key[2]):.15e}"
            )
        else:
            best_balanced_key = (math.inf, math.inf, math.inf)
            print("旧 balanced 模型未包含结构状态口径；保留文件并重新建立排名。")
    elif best_model_path.exists():
        mean_best_checkpoint = torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        mean_best_metrics = mean_best_checkpoint.get("metrics")
        if (
            mean_best_metrics is not None
            and "per_opcode" in mean_best_metrics
            and "structural_groups" in mean_best_metrics
        ):
            best_balanced_key = balanced_checkpoint_key(mean_best_metrics)
            initialized_checkpoint = dict(mean_best_checkpoint)
            initialized_checkpoint["balanced_key"] = list(best_balanced_key)
            initialized_checkpoint["balanced_schema"] = 2
            atomic_torch_save(initialized_checkpoint, best_balanced_model_path)
            print(
                "用现有 best_model.pt 初始化 balanced 候选："
                f"errors={best_balanced_key[0]}, "
                f"worst_loss={best_balanced_key[1]:.15e}, "
                f"mean_loss={best_balanced_key[2]:.15e}"
            )
        else:
            best_balanced_key = (math.inf, math.inf, math.inf)
    else:
        best_balanced_key = (math.inf, math.inf, math.inf)

    dynamic_opcode_state = load_dynamic_opcode_state(dynamic_state_path)
    if Config.DYNAMIC_OPCODE_REPLAY and int(dynamic_opcode_state["audit_count"]) == 0:
        print("首次启动动态 replay：先运行独立的全量 opcode 审计以初始化权重。")
        _, dynamic_opcode_state = run_dynamic_opcode_audit(
            model,
            device,
            step,
            dynamic_opcode_state,
        )

    model.train()
    started_at = time.time()
    accumulated_train_loss = 0.0
    accumulated_train_steps = 0
    latest_metrics: dict[str, Any] | None = None

    print("开始在线训练；默认持续运行，按 Ctrl+C 保存并退出。")
    try:
        while Config.MAX_STEPS is None or step < Config.MAX_STEPS:
            step += 1
            if Config.HARD_REPLAY_MODE:
                replay_count = max(
                    1, round(Config.BATCH_SIZE * Config.HARD_REPLAY_FRACTION)
                )
                if Config.DYNAMIC_OPCODE_REPLAY:
                    structured_hard_count = round(
                        replay_count * Config.STRUCTURED_HARD_SHARE_OF_REPLAY
                    )
                    dynamic_count = replay_count - structured_hard_count
                else:
                    structured_hard_count = replay_count
                    dynamic_count = 0
                uniform_count = (
                    Config.BATCH_SIZE - structured_hard_count - dynamic_count
                )
                x_uniform, y_uniform, _ = generate_batch(
                    uniform_count,
                    device,
                    train_generator,
                    return_metadata=False,
                )
                x_parts = [x_uniform]
                y_parts = [y_uniform]
                if structured_hard_count > 0:
                    x_hard, y_hard = generate_hard_replay_batch(
                        structured_hard_count, device, train_generator
                    )
                    x_parts.append(x_hard)
                    y_parts.append(y_hard)
                if dynamic_count > 0:
                    x_dynamic, y_dynamic = generate_dynamic_opcode_replay_batch(
                        dynamic_count,
                        device,
                        train_generator,
                        dynamic_opcode_state,
                    )
                    x_parts.append(x_dynamic)
                    y_parts.append(y_dynamic)
                x = torch.cat(x_parts, dim=0)
                y = torch.cat(y_parts, dim=0)
                permutation = torch.randperm(
                    Config.BATCH_SIZE, device=device, generator=train_generator
                )
                x = x[permutation]
                y = y[permutation]
            else:
                x, y, _ = generate_batch(
                    Config.BATCH_SIZE,
                    device,
                    train_generator,
                    return_metadata=False,
                )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            if Config.GRAD_CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            accumulated_train_loss += float(loss.item())
            accumulated_train_steps += 1

            if step % Config.LOG_INTERVAL == 0:
                average_train_loss = accumulated_train_loss / accumulated_train_steps
                learning_rate = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - started_at
                samples_per_second = (
                    Config.BATCH_SIZE * accumulated_train_steps
                    / max(elapsed, 1e-9)
                )
                print(
                    f"step={step:,} | train_loss={average_train_loss:.15e} | "
                    f"lr={learning_rate:.9e} | samples/s={samples_per_second:,.1f}"
                )
                accumulated_train_loss = 0.0
                accumulated_train_steps = 0
                started_at = time.time()

            if step % Config.EVAL_INTERVAL == 0:
                evaluation_started = time.time()
                latest_metrics = evaluate(
                    model,
                    device,
                    Config.VALIDATION_SAMPLES,
                    Config.VALIDATION_BATCH_SIZE,
                )
                latest_metrics["step"] = step
                latest_metrics["eval_seconds"] = time.time() - evaluation_started
                latest_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

                is_best = float(latest_metrics["loss"]) < best_val_loss
                if is_best:
                    best_val_loss = float(latest_metrics["loss"])

                if scheduler is not None:
                    scheduler.step(float(latest_metrics["loss"]))

                print_validation(step, latest_metrics, best_val_loss)
                if is_best:
                    print(f"保存新的最佳模型：val_loss={best_val_loss:.15e}")
                    atomic_torch_save(
                        {
                            "format_version": 1,
                            "step": step,
                            "val_loss": best_val_loss,
                            "model_state_dict": model.state_dict(),
                            "config": config_as_dict(),
                            "isa": ISA_DESCRIPTION,
                            "metrics": latest_metrics,
                        },
                        best_model_path,
                    )

                current_balanced_key = balanced_checkpoint_key(latest_metrics)
                if current_balanced_key < best_balanced_key:
                    best_balanced_key = current_balanced_key
                    print(
                        "保存新的 balanced 最佳模型："
                        f"errors={best_balanced_key[0]}, "
                        f"worst_loss={best_balanced_key[1]:.15e}, "
                        f"mean_loss={best_balanced_key[2]:.15e}"
                    )
                    atomic_torch_save(
                        {
                            "format_version": 1,
                            "balanced_schema": 2,
                            "step": step,
                            "val_loss": float(latest_metrics["loss"]),
                            "balanced_key": list(best_balanced_key),
                            "model_state_dict": model.state_dict(),
                            "config": config_as_dict(),
                            "isa": ISA_DESCRIPTION,
                            "metrics": latest_metrics,
                        },
                        best_balanced_model_path,
                    )

                history_row = {
                    "step": step,
                    "elapsed_seconds": latest_metrics["eval_seconds"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss": loss.item(),
                    "val_loss": latest_metrics["loss"],
                    "val_state_loss": latest_metrics["state_loss"],
                    "val_control_loss": latest_metrics["control_loss"],
                    "val_bit_accuracy": latest_metrics["bit_accuracy"],
                    "val_bit_errors": latest_metrics["bit_errors"],
                    "val_state_bit_accuracy": latest_metrics["state_bit_accuracy"],
                    "val_state_bit_errors": latest_metrics["state_bit_errors"],
                    "val_control_bit_accuracy": latest_metrics["control_bit_accuracy"],
                    "val_control_bit_errors": latest_metrics["control_bit_errors"],
                    "val_output_exact_accuracy": latest_metrics["output_exact_accuracy"],
                    "val_output_exact_errors": latest_metrics["output_exact_errors"],
                    "val_state_exact_accuracy": latest_metrics["state_exact_accuracy"],
                    "val_state_exact_errors": latest_metrics["state_exact_errors"],
                    "val_control_exact_accuracy": latest_metrics["control_exact_accuracy"],
                    "val_control_exact_errors": latest_metrics["control_exact_errors"],
                    "worst_opcode_by_loss": latest_metrics["worst_opcode_by_loss"],
                    "worst_opcode_by_exact": latest_metrics["worst_opcode_by_exact"],
                    "sparse_state_loss": latest_metrics["structural_groups"][
                        "SPARSE_STATE"
                    ]["loss"],
                    "sparse_state_exact_errors": latest_metrics["structural_groups"][
                        "SPARSE_STATE"
                    ]["exact_errors"],
                    "dense_state_loss": latest_metrics["structural_groups"][
                        "DENSE_STATE"
                    ]["loss"],
                    "dense_state_exact_errors": latest_metrics["structural_groups"][
                        "DENSE_STATE"
                    ]["exact_errors"],
                    "best_val_loss": best_val_loss,
                }
                append_csv(history_row, Config.RESULT_DIR / "history.csv")
                append_jsonl(latest_metrics, Config.RESULT_DIR / "history.jsonl")
                atomic_json_save(
                    latest_metrics, Config.RESULT_DIR / "validation_latest.json"
                )

                if Config.SAVE_LATEST_EVERY_EVAL:
                    atomic_torch_save(
                        checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            train_generator,
                            step,
                            best_val_loss,
                            latest_metrics,
                        ),
                        latest_checkpoint_path,
                    )

            if (
                Config.DYNAMIC_OPCODE_REPLAY
                and step
                - int(dynamic_opcode_state.get("last_audit_step") or 0)
                >= Config.DYNAMIC_AUDIT_INTERVAL
            ):
                _, dynamic_opcode_state = run_dynamic_opcode_audit(
                    model,
                    device,
                    step,
                    dynamic_opcode_state,
                )
            if (
                step % Config.EVAL_INTERVAL != 0
                and step % Config.CHECKPOINT_INTERVAL == 0
            ):
                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        train_generator,
                        step,
                        best_val_loss,
                        latest_metrics,
                    ),
                    latest_checkpoint_path,
                )

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在保存最新断点……")
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                train_generator,
                step,
                best_val_loss,
                latest_metrics,
            ),
            latest_checkpoint_path,
        )
        print(f"断点已保存：{latest_checkpoint_path}")
        return

    atomic_torch_save(
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            train_generator,
            step,
            best_val_loss,
            latest_metrics,
        ),
        latest_checkpoint_path,
    )
    print(f"训练完成，最新断点：{latest_checkpoint_path}")
    print(f"最佳模型：{best_model_path}")
    print(f"balanced 最佳模型：{best_balanced_model_path}")


if __name__ == "__main__":
    main()

# %%



# %%
