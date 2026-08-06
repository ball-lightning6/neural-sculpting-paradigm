"""Neural CPU v3 独立、只读的大规模随机验证器。

本脚本内嵌最终网络结构、ISA 和真值生成器，只加载 checkpoint 做推理；
不创建 optimizer，也不会改写权重。

默认验证 10_737_418_240 个随机操作。若要调整规模，直接修改
``Config.SAMPLES``。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 内嵌 Neural CPU v3 定义
# =============================================================================


class CPUConfig:
    """只保留构造最终推理网络所需的配置。"""

    HIDDEN_SIZE = 1024
    # 第一隐藏层之后再追加两层，因此总计三个 1024-wide 隐藏层。
    HIDDEN_LAYERS = 2
    DROPOUT = 0.0


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


class NeuralCPUCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(INPUT_BITS, CPUConfig.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(CPUConfig.HIDDEN_SIZE),
        ]
        for _ in range(CPUConfig.HIDDEN_LAYERS):
            layers.extend(
                [
                    nn.Linear(CPUConfig.HIDDEN_SIZE, CPUConfig.HIDDEN_SIZE),
                    nn.GELU(),
                    nn.LayerNorm(CPUConfig.HIDDEN_SIZE),
                ]
            )
        layers.append(nn.Linear(CPUConfig.HIDDEN_SIZE, OUTPUT_BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


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
    shifts = torch.arange(
        width - 1, -1, -1, device=values.device, dtype=torch.int64
    )
    return ((values.unsqueeze(1) >> shifts) & 1).to(torch.float32)


def registers_to_bits(registers: torch.Tensor) -> torch.Tensor:
    bits = integer_to_bits(registers.reshape(-1), REGISTER_BITS)
    return bits.reshape(registers.shape[0], NUM_REGISTERS * REGISTER_BITS)


def balanced_opcodes(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    opcodes = torch.arange(batch_size, device=device, dtype=torch.int64) % NUM_OPCODES
    permutation = torch.randperm(batch_size, device=device, generator=generator)
    return opcodes[permutation]


def set_destination(
    output_registers: torch.Tensor,
    destination: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> None:
    rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if rows.numel() > 0:
        output_registers[rows, destination[rows]] = values[rows] & 0xFF


def set_flag(
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
) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata | None]:
    """生成与最终训练脚本普通 IID 部分完全相同的随机批次。"""

    opcodes = balanced_opcodes(batch_size, device, generator)
    p1 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    p2 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    p3 = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    immediate = torch.randint(
        0, 256, (batch_size,), device=device, generator=generator
    )
    offset = torch.randint(
        -1024, 1024, (batch_size,), device=device, generator=generator
    )
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
        (p1[rrr_mask] << 9) | (p2[rrr_mask] << 7) | (p3[rrr_mask] << 5)
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
    set_destination(output_registers, p1, mask, value1)
    mask = operation_mask("MOVI")
    set_destination(output_registers, p1, mask, immediate)
    controls[operation_mask("LOAD"), CTRL_MEMORY_READ] = 1
    controls[operation_mask("STORE"), CTRL_MEMORY_WRITE] = 1

    mask = operation_mask("ADD")
    result = value1 + value2
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("ADC")
    result = value1 + value2 + carry_in
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("SUB")
    result = value1 - value2
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, value1 < value2)

    mask = operation_mask("SBC")
    result = value1 - value2 - carry_in
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, value1 < (value2 + carry_in))

    mask = operation_mask("INC")
    result = value1 + 1
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("DEC")
    result = value1 - 1
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, value1 == 0)

    for name, result in (
        ("AND", value1 & value2),
        ("OR", value1 | value2),
        ("XOR", value1 ^ value2),
        ("NOT", (~value1) & 0xFF),
    ):
        mask = operation_mask(name)
        set_destination(output_registers, p1, mask, result)
        set_flag(output_flags, FLAG_ZF, mask, result == 0)

    shift = value2 & 0x7
    mask = operation_mask("SHL")
    result = (value1 << shift) & 0xFF
    shift_left_carry = torch.where(
        shift == 0, carry_in, (value1 >> (8 - shift).clamp(min=1)) & 1
    )
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, result == 0)
    set_flag(output_flags, FLAG_CF, mask, shift_left_carry)

    mask = operation_mask("SHR")
    result = value1 >> shift
    shift_right_carry = torch.where(
        shift == 0, carry_in, (value1 >> (shift - 1).clamp(min=0)) & 1
    )
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, result == 0)
    set_flag(output_flags, FLAG_CF, mask, shift_right_carry)

    mask = operation_mask("CMP")
    set_flag(output_flags, FLAG_ZF, mask, value1 == value2)
    set_flag(output_flags, FLAG_GF, mask, value1 > value2)
    set_flag(output_flags, FLAG_CF, mask, value1 < value2)

    mask = operation_mask("ADDI")
    result = destination_value + immediate
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, result > 0xFF)

    mask = operation_mask("SUBI")
    result = destination_value - immediate
    set_destination(output_registers, p1, mask, result)
    set_flag(output_flags, FLAG_ZF, mask, (result & 0xFF) == 0)
    set_flag(output_flags, FLAG_CF, mask, destination_value < immediate)

    mask = operation_mask("CMPI")
    set_flag(output_flags, FLAG_ZF, mask, destination_value == immediate)
    set_flag(output_flags, FLAG_GF, mask, destination_value > immediate)
    set_flag(output_flags, FLAG_CF, mask, destination_value < immediate)

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
    state_bits = torch.cat(
        [flags.to(torch.float32), registers_to_bits(registers)], dim=1
    )
    output_state_bits = torch.cat(
        [output_flags.to(torch.float32), registers_to_bits(output_registers)], dim=1
    )
    inputs = torch.cat([instruction_bits, state_bits], dim=1)
    targets = torch.cat([output_state_bits, controls.to(torch.float32)], dim=1)
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
    return inputs, targets, metadata


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
    decoded_flags = {"ZF": bits[0], "GF": bits[1], "CF": bits[2]}
    decoded_registers = []
    for register in range(NUM_REGISTERS):
        start = NUM_FLAGS + register * REGISTER_BITS
        value = 0
        for bit in bits[start : start + REGISTER_BITS]:
            value = (value << 1) | int(bit)
        decoded_registers.append(value)
    decoded_controls = {
        "memory_read": bits[STATE_BITS + CTRL_MEMORY_READ],
        "memory_write": bits[STATE_BITS + CTRL_MEMORY_WRITE],
        "branch_taken": bits[STATE_BITS + CTRL_BRANCH_TAKEN],
        "halt": bits[STATE_BITS + CTRL_HALT],
    }
    return {
        "flags": decoded_flags,
        "registers": decoded_registers,
        "controls": decoded_controls,
    }


def embedded_cpu_module() -> ModuleType:
    module = ModuleType("embedded_neural_cpu_v3")
    exported = (
        "NeuralCPUCore",
        "generate_batch",
        "OPCODE_NAMES",
        "ISA_DESCRIPTION",
        "INPUT_BITS",
        "OUTPUT_BITS",
        "STATE_BITS",
        "CONTROL_BITS",
        "NUM_OPCODES",
        "NUM_FLAGS",
        "REGISTER_BITS",
        "output_bit_name",
        "decode_output_vector",
        "alias_pattern_name",
        "RI8_OPCODES",
        "RELATIVE_BRANCH_OPCODES",
    )
    namespace = globals()
    for name in exported:
        setattr(module, name, namespace[name])
    module.Config = CPUConfig
    return module


def script_directory() -> Path:
    source = globals().get("__file__")
    if not source:
        return Path.cwd().resolve()
    source_dir = Path(source).resolve().parent
    return source_dir.parent if source_dir.name == "scripts" else source_dir


class Config:
    BASE_DIR = script_directory()
    MODEL_PATH = BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    RESULT_DIR = BASE_DIR / "results" / "validation"

    # 最终全量随机验证规模：10 × 2^30，约 107 亿次单步操作。
    SAMPLES = 10_737_418_240
    SEED = 20260901
    # 约每 1.34 亿条打印并保存一次；5090 上通常间隔 15～20 秒。
    PROGRESS_INTERVAL = 134_217_728
    MAX_SAVED_ERROR_CASES = 10_000

    # 5090 启动时实测这些 batch，自动选择端到端吞吐最高的一档。
    AUTO_TUNE_BATCH = True
    BATCH_CANDIDATES = (262_144, 524_288, 1_048_576, 2_097_152)
    BATCH_TUNE_REPEATS = 2
    FALLBACK_BATCH_SIZE = 262_144

    ALLOW_TF32 = False
    SMOKE_TEST = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def configure_numerics() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    return device


def check_checkpoint(checkpoint: dict[str, Any], cpu: ModuleType) -> dict[str, Any]:
    stored = checkpoint.get("config") or {}
    expected = {
        "HIDDEN_SIZE": int(cpu.Config.HIDDEN_SIZE),
        "HIDDEN_LAYERS": int(cpu.Config.HIDDEN_LAYERS),
        "DROPOUT": float(cpu.Config.DROPOUT),
        "INPUT_BITS": int(cpu.INPUT_BITS),
        "OUTPUT_BITS": int(cpu.OUTPUT_BITS),
    }
    mismatches = {
        key: {"checkpoint": stored[key], "source": value}
        for key, value in expected.items()
        if key in stored and stored[key] != value
    }
    stored_isa = checkpoint.get("isa")
    isa_matches = stored_isa is None or stored_isa == cpu.ISA_DESCRIPTION
    if mismatches or not isa_matches:
        raise RuntimeError(
            "checkpoint 与训练脚本不兼容："
            f"config={mismatches}, isa_matches={isa_matches}"
        )
    return {
        "expected": expected,
        "mismatches": mismatches,
        "checkpoint_has_isa": stored_isa is not None,
        "isa_matches": isa_matches,
    }


def wilson_interval(errors: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = errors / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def rate_summary(errors: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(errors, total)
    result: dict[str, Any] = {
        "errors": errors,
        "trials": total,
        "error_rate": errors / total if total else None,
        "accuracy": 1.0 - errors / total if total else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }
    if errors:
        result["observed_trials_per_error"] = total / errors
    elif total:
        # 0 次错误时的单侧 95% 二项/Poisson 近似上界。
        upper = -math.log(0.05) / total
        result["zero_error_one_sided_95_upper"] = upper
        result["zero_error_trials_per_error_95_lower"] = 1.0 / upper
    return result


@torch.inference_mode()
def benchmark_batch_sizes(
    model: torch.nn.Module, cpu: ModuleType, device: torch.device
) -> tuple[int, list[dict[str, Any]]]:
    if device.type != "cuda" or not Config.AUTO_TUNE_BATCH:
        return Config.FALLBACK_BATCH_SIZE, []

    rows: list[dict[str, Any]] = []
    print("\n自动测试验证 batch 吞吐：")
    for batch_size in Config.BATCH_CANDIDATES:
        generator = torch.Generator(device=device.type)
        generator.manual_seed(20260900 + int(math.log2(batch_size)))
        try:
            torch.cuda.empty_cache()
            # 一轮 warmup；计时轮包含随机真值生成、前向和错误统计。
            inputs, truth, _ = cpu.generate_batch(
                batch_size, device, generator, return_metadata=False
            )
            logits = model(inputs)
            _ = ((logits >= 0.0) != (truth >= 0.5)).sum()
            torch.cuda.synchronize()

            started = time.perf_counter()
            for _ in range(Config.BATCH_TUNE_REPEATS):
                inputs, truth, _ = cpu.generate_batch(
                    batch_size, device, generator, return_metadata=False
                )
                logits = model(inputs)
                _ = ((logits >= 0.0) != (truth >= 0.5)).sum()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            throughput = batch_size * Config.BATCH_TUNE_REPEATS / elapsed
            row = {
                "batch_size": batch_size,
                "samples_per_second": throughput,
                "status": "ok",
            }
            rows.append(row)
            print(f"  batch={batch_size:,}: {throughput:,.1f} samples/s")
            del inputs, truth, logits
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            rows.append(
                {
                    "batch_size": batch_size,
                    "samples_per_second": 0.0,
                    "status": "oom",
                }
            )
            print(f"  batch={batch_size:,}: OOM，跳过")
        finally:
            torch.cuda.empty_cache()

    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("所有候选 batch 均 OOM")
    best = max(successful, key=lambda row: row["samples_per_second"])
    print(f"选择 batch={best['batch_size']:,}")
    return int(best["batch_size"]), rows


def empty_metrics(cpu: ModuleType) -> dict[str, Any]:
    return {
        "samples": 0,
        "loss_sum": 0.0,
        "state_loss_sum": 0.0,
        "control_loss_sum": 0.0,
        "bit_errors": 0,
        "state_bit_errors": 0,
        "control_bit_errors": 0,
        "exact_errors": 0,
        "state_exact_errors": 0,
        "control_exact_errors": 0,
        "opcode_samples": [0] * cpu.NUM_OPCODES,
        "opcode_loss_sums": [0.0] * cpu.NUM_OPCODES,
        "opcode_bit_errors": [0] * cpu.NUM_OPCODES,
        "opcode_exact_errors": [0] * cpu.NUM_OPCODES,
        "output_bit_loss_sums": [0.0] * cpu.OUTPUT_BITS,
        "output_bit_errors": [0] * cpu.OUTPUT_BITS,
        "saved_error_cases": 0,
        "elapsed_seconds": 0.0,
        "status": "running",
    }


def add_list(target: list[Any], values: torch.Tensor, cast: type) -> None:
    for index, value in enumerate(values.detach().cpu().tolist()):
        target[index] += cast(value)


def update_metrics(
    metrics: dict[str, Any],
    truth: torch.Tensor,
    logits: torch.Tensor,
    opcodes: torch.Tensor,
    cpu: ModuleType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    truth_bool = truth >= 0.5
    predictions = logits >= 0.0
    errors = predictions != truth_bool
    sample_bit_errors = errors.sum(dim=1, dtype=torch.int64)
    sample_exact_error = sample_bit_errors > 0
    loss_matrix = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    sample_loss = loss_matrix.mean(dim=1)

    batch_size = truth.shape[0]
    metrics["samples"] += batch_size
    metrics["loss_sum"] += float(loss_matrix.sum(dtype=torch.float64).item())
    metrics["state_loss_sum"] += float(
        loss_matrix[:, : cpu.STATE_BITS].sum(dtype=torch.float64).item()
    )
    metrics["control_loss_sum"] += float(
        loss_matrix[:, cpu.STATE_BITS :].sum(dtype=torch.float64).item()
    )
    metrics["bit_errors"] += int(errors.sum(dtype=torch.int64).item())
    metrics["state_bit_errors"] += int(
        errors[:, : cpu.STATE_BITS].sum(dtype=torch.int64).item()
    )
    metrics["control_bit_errors"] += int(
        errors[:, cpu.STATE_BITS :].sum(dtype=torch.int64).item()
    )
    metrics["exact_errors"] += int(sample_exact_error.sum(dtype=torch.int64).item())
    metrics["state_exact_errors"] += int(
        errors[:, : cpu.STATE_BITS].any(dim=1).sum(dtype=torch.int64).item()
    )
    metrics["control_exact_errors"] += int(
        errors[:, cpu.STATE_BITS :].any(dim=1).sum(dtype=torch.int64).item()
    )

    add_list(
        metrics["output_bit_loss_sums"],
        loss_matrix.sum(dim=0, dtype=torch.float64),
        float,
    )
    add_list(
        metrics["output_bit_errors"],
        errors.sum(dim=0, dtype=torch.int64),
        int,
    )
    add_list(
        metrics["opcode_samples"],
        torch.bincount(opcodes, minlength=cpu.NUM_OPCODES),
        int,
    )
    add_list(
        metrics["opcode_loss_sums"],
        torch.bincount(
            opcodes, weights=sample_loss, minlength=cpu.NUM_OPCODES
        ),
        float,
    )
    opcode_bit_errors = torch.zeros(
        cpu.NUM_OPCODES, device=logits.device, dtype=torch.int64
    )
    opcode_bit_errors.scatter_add_(0, opcodes, sample_bit_errors)
    add_list(metrics["opcode_bit_errors"], opcode_bit_errors, int)
    opcode_exact_errors = torch.zeros_like(opcode_bit_errors)
    opcode_exact_errors.scatter_add_(
        0, opcodes, sample_exact_error.to(torch.int64)
    )
    add_list(metrics["opcode_exact_errors"], opcode_exact_errors, int)
    return sample_exact_error, sample_loss, predictions


def save_error_cases(
    handle: TextIO,
    metrics: dict[str, Any],
    sample_offset: int,
    sample_exact_error: torch.Tensor,
    sample_loss: torch.Tensor,
    truth: torch.Tensor,
    logits: torch.Tensor,
    predictions: torch.Tensor,
    metadata: Any,
    cpu: ModuleType,
) -> None:
    available = Config.MAX_SAVED_ERROR_CASES - metrics["saved_error_cases"]
    if available <= 0:
        return
    selected = torch.nonzero(sample_exact_error, as_tuple=False).squeeze(1)[:available]
    if selected.numel() == 0:
        return

    logits_cpu = logits[selected].detach().cpu()
    probabilities_cpu = torch.sigmoid(logits_cpu)
    truth_cpu = truth[selected].detach().cpu().to(torch.int64)
    predictions_cpu = predictions[selected].detach().cpu().to(torch.int64)
    loss_cpu = sample_loss[selected].detach().cpu()
    opcodes_cpu = metadata.opcodes[selected].detach().cpu()
    instructions_cpu = metadata.instructions[selected].detach().cpu()
    registers_cpu = metadata.registers[selected].detach().cpu()
    flags_cpu = metadata.flags[selected].detach().cpu()
    p1_cpu = metadata.p1[selected].detach().cpu()
    p2_cpu = metadata.p2[selected].detach().cpu()
    p3_cpu = metadata.p3[selected].detach().cpu()
    immediate_cpu = metadata.immediate[selected].detach().cpu()
    offset_cpu = metadata.offset[selected].detach().cpu()

    for stored_index, local_index in enumerate(selected.detach().cpu().tolist()):
        opcode = int(opcodes_cpu[stored_index])
        p1 = int(p1_cpu[stored_index])
        p2 = int(p2_cpu[stored_index])
        p3 = int(p3_cpu[stored_index])
        registers = [int(value) for value in registers_cpu[stored_index].tolist()]
        flags = [int(value) for value in flags_cpu[stored_index].tolist()]
        targets = truth_cpu[stored_index].tolist()
        predicted = predictions_cpu[stored_index].tolist()
        wrong_outputs = []
        for output_index, (target, prediction) in enumerate(zip(targets, predicted)):
            if target == prediction:
                continue
            logit = float(logits_cpu[stored_index, output_index])
            wrong_outputs.append(
                {
                    "index": output_index,
                    "name": cpu.output_bit_name(output_index),
                    "target": int(target),
                    "prediction": int(prediction),
                    "logit": logit,
                    "probability": float(
                        probabilities_cpu[stored_index, output_index]
                    ),
                }
            )

        alias_code = int(p1 == p2) + 2 * int(p1 == p3) + 4 * int(p2 == p3)
        case = {
            "sample_index": sample_offset + local_index,
            "sample_loss": float(loss_cpu[stored_index]),
            "opcode": opcode,
            "opcode_name": cpu.OPCODE_NAMES[opcode],
            "instruction_int": int(instructions_cpu[stored_index]),
            "instruction_hex": f"0x{int(instructions_cpu[stored_index]):04X}",
            "instruction_binary": f"{int(instructions_cpu[stored_index]):016b}",
            "p1_dst": p1,
            "p2_src1": p2,
            "p3_src2": p3,
            "immediate": (
                int(immediate_cpu[stored_index])
                if opcode in cpu.RI8_OPCODES
                else None
            ),
            "offset": (
                int(offset_cpu[stored_index])
                if opcode in cpu.RELATIVE_BRANCH_OPCODES
                else None
            ),
            "input_registers": registers,
            "input_flags": {"ZF": flags[0], "GF": flags[1], "CF": flags[2]},
            "state_hamming_weight": sum(flags)
            + sum(value.bit_count() for value in registers),
            "alias_pattern": cpu.alias_pattern_name(alias_code),
            "target": cpu.decode_output_vector(targets),
            "prediction": cpu.decode_output_vector(predicted),
            "wrong_outputs": wrong_outputs,
        }
        handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        metrics["saved_error_cases"] += 1


def build_summary(
    metrics: dict[str, Any],
    cpu: ModuleType,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    samples = metrics["samples"]
    output_trials = samples * cpu.OUTPUT_BITS
    state_trials = samples * cpu.STATE_BITS
    control_trials = samples * cpu.CONTROL_BITS
    return {
        **run_metadata,
        "status": metrics["status"],
        "requested_samples": Config.SAMPLES,
        "samples": samples,
        "completion_fraction": samples / Config.SAMPLES if Config.SAMPLES else 1.0,
        "seed": Config.SEED,
        "loss": metrics["loss_sum"] / output_trials if output_trials else None,
        "state_loss": (
            metrics["state_loss_sum"] / state_trials if state_trials else None
        ),
        "control_loss": (
            metrics["control_loss_sum"] / control_trials if control_trials else None
        ),
        "bit": rate_summary(metrics["bit_errors"], output_trials),
        "state_bit": rate_summary(metrics["state_bit_errors"], state_trials),
        "control_bit": rate_summary(metrics["control_bit_errors"], control_trials),
        "output_exact": rate_summary(metrics["exact_errors"], samples),
        "state_exact": rate_summary(metrics["state_exact_errors"], samples),
        "control_exact": rate_summary(metrics["control_exact_errors"], samples),
        "saved_error_cases": metrics["saved_error_cases"],
        "error_cases_truncated": metrics["exact_errors"]
        > metrics["saved_error_cases"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "samples_per_second": (
            samples / metrics["elapsed_seconds"]
            if metrics["elapsed_seconds"] > 0
            else None
        ),
    }


def save_reports(
    metrics: dict[str, Any],
    cpu: ModuleType,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = build_summary(metrics, cpu, run_metadata)
    atomic_json_save(summary, Config.RESULT_DIR / "summary.json")

    opcode_rows = []
    for opcode in range(cpu.NUM_OPCODES):
        samples = metrics["opcode_samples"][opcode]
        if not samples:
            continue
        opcode_rows.append(
            {
                "opcode": opcode,
                "opcode_name": cpu.OPCODE_NAMES[opcode],
                "samples": samples,
                "loss": metrics["opcode_loss_sums"][opcode] / samples,
                "bit_errors": metrics["opcode_bit_errors"][opcode],
                "bit_error_rate": metrics["opcode_bit_errors"][opcode]
                / (samples * cpu.OUTPUT_BITS),
                "exact_errors": metrics["opcode_exact_errors"][opcode],
                "exact_error_rate": metrics["opcode_exact_errors"][opcode] / samples,
            }
        )
    write_csv(Config.RESULT_DIR / "opcode_metrics.csv", opcode_rows)

    bit_rows = []
    for index in range(cpu.OUTPUT_BITS):
        bit_rows.append(
            {
                "index": index,
                "name": cpu.output_bit_name(index),
                "samples": metrics["samples"],
                "loss": (
                    metrics["output_bit_loss_sums"][index] / metrics["samples"]
                    if metrics["samples"]
                    else None
                ),
                "errors": metrics["output_bit_errors"][index],
                "error_rate": (
                    metrics["output_bit_errors"][index] / metrics["samples"]
                    if metrics["samples"]
                    else None
                ),
            }
        )
    write_csv(Config.RESULT_DIR / "output_bit_metrics.csv", bit_rows)
    return summary


@torch.inference_mode()
def run_validation(
    model: torch.nn.Module,
    cpu: ModuleType,
    device: torch.device,
    batch_size: int,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = empty_metrics(cpu)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(Config.SEED)
    next_progress = min(Config.PROGRESS_INTERVAL, Config.SAMPLES)
    started = time.time()
    error_path = Config.RESULT_DIR / "error_cases.jsonl"

    print(
        f"\n开始全量随机验证：samples={Config.SAMPLES:,} | "
        f"batch={batch_size:,} | seed={Config.SEED}",
        flush=True,
    )
    with error_path.open("w", encoding="utf-8") as error_handle:
        try:
            while metrics["samples"] < Config.SAMPLES:
                current_batch = min(batch_size, Config.SAMPLES - metrics["samples"])
                sample_offset = metrics["samples"]
                inputs, truth, metadata = cpu.generate_batch(
                    current_batch,
                    device,
                    generator,
                    return_metadata=True,
                )
                if metadata is None:
                    raise RuntimeError("真值生成器没有返回 metadata")
                logits = model(inputs)
                sample_exact_error, sample_loss, predictions = update_metrics(
                    metrics, truth, logits, metadata.opcodes, cpu
                )
                save_error_cases(
                    error_handle,
                    metrics,
                    sample_offset,
                    sample_exact_error,
                    sample_loss,
                    truth,
                    logits,
                    predictions,
                    metadata,
                    cpu,
                )

                if metrics["samples"] >= next_progress or metrics["samples"] == Config.SAMPLES:
                    metrics["elapsed_seconds"] = time.time() - started
                    current_loss = metrics["loss_sum"] / (
                        metrics["samples"] * cpu.OUTPUT_BITS
                    )
                    print(
                        f"{metrics['samples']:,}/{Config.SAMPLES:,} | "
                        f"loss={current_loss:.15e} | "
                        f"exact_errors={metrics['exact_errors']:,} | "
                        f"bit_errors={metrics['bit_errors']:,} | "
                        f"samples/s={metrics['samples'] / metrics['elapsed_seconds']:,.1f}",
                        flush=True,
                    )
                    error_handle.flush()
                    save_reports(metrics, cpu, run_metadata)
                    while next_progress <= metrics["samples"]:
                        next_progress += Config.PROGRESS_INTERVAL
            metrics["status"] = "completed"
        except KeyboardInterrupt:
            metrics["status"] = "interrupted"
            print("\n收到 Ctrl+C，保存已完成的验证结果。")

    metrics["elapsed_seconds"] = time.time() - started
    return save_reports(metrics, cpu, run_metadata)


def main() -> None:
    if Config.SMOKE_TEST:
        Config.SAMPLES = 4_096
        Config.PROGRESS_INTERVAL = 4_096
        Config.AUTO_TUNE_BATCH = False
        Config.FALLBACK_BATCH_SIZE = 4_096
        Config.RESULT_DIR = Config.BASE_DIR / "results_neural_cpu_v3_validation_smoke"

    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到权重：{Config.MODEL_PATH}\n"
            "请修改 Config.MODEL_PATH。"
        )
    cpu = embedded_cpu_module()
    device = configure_numerics()
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"模型：{Config.MODEL_PATH}")
    print("真值源：验证器内嵌 Neural CPU v3 定义（不依赖训练脚本）")
    print(f"结果目录：{Config.RESULT_DIR}")
    print(f"TF32={Config.ALLOW_TF32} | dtype=FP32")

    checkpoint = torch.load(
        Config.MODEL_PATH, map_location="cpu", weights_only=False
    )
    compatibility = check_checkpoint(checkpoint, cpu)
    model = cpu.NeuralCPUCore().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch_size, batch_benchmark = benchmark_batch_sizes(model, cpu, device)
    validator_source = globals().get("__file__")
    validator_path = Path(validator_source).resolve() if validator_source else None
    validator_is_file = validator_path is not None and validator_path.is_file()
    run_metadata = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "model_path": str(Config.MODEL_PATH.resolve()),
        "truth_implementation": "embedded_neural_cpu_v3",
        "validator_path": str(validator_path) if validator_is_file else None,
        "model_sha256": sha256_file(Config.MODEL_PATH),
        "validator_sha256": (
            sha256_file(validator_path) if validator_is_file else None
        ),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_val_loss": checkpoint.get(
            "val_loss", checkpoint.get("best_val_loss")
        ),
        "checkpoint_balanced_key": checkpoint.get("balanced_key"),
        "architecture": {
            "input_bits": cpu.INPUT_BITS,
            "hidden_size": cpu.Config.HIDDEN_SIZE,
            "hidden_linear_layers": cpu.Config.HIDDEN_LAYERS + 1,
            "output_bits": cpu.OUTPUT_BITS,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "compatibility": compatibility,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "tf32": Config.ALLOW_TF32,
        },
        "validation": {
            "distribution": (
                "training generate_batch distribution: uniform opcodes stratified "
                "within each batch; random canonical instruction fields/registers/flags"
            ),
            "samples": Config.SAMPLES,
            "seed": Config.SEED,
            "batch_size": batch_size,
            "batch_autotune": batch_benchmark,
        },
    }
    atomic_json_save(run_metadata, Config.RESULT_DIR / "run_manifest.json")

    summary = run_validation(model, cpu, device, batch_size, run_metadata)
    exact = summary["output_exact"]
    bit = summary["bit"]
    print("\n=== Neural CPU v3 全量随机验证完成 ===")
    print(
        f"samples={summary['samples']:,} | loss={summary['loss']:.15e} | "
        f"bit_errors={bit['errors']:,} | bit_acc={bit['accuracy']:.12f}"
    )
    print(
        f"exact_errors={exact['errors']:,} | exact_acc={exact['accuracy']:.12f} | "
        f"samples/s={summary['samples_per_second']:,.1f}"
    )
    if exact["errors"] == 0:
        print(
            "0-error 单侧 95% 错误率上界："
            f"{exact['zero_error_one_sided_95_upper']:.3e}/operation"
        )
    print(f"汇总：{Config.RESULT_DIR / 'summary.json'}")
    print(f"错误现场：{Config.RESULT_DIR / 'error_cases.jsonl'}")


if __name__ == "__main__":
    main()

# %%
