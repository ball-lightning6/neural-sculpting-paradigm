"""用训练好的 Neural CPU v3 渲染 Fast 2D Clouds 的一帧。

本脚本忠实保留原 shader 的双 fBm、8 个 octave、curl 旋转、smoothstep 和颜色混合
结构，只把浮点运算编译为 Q17.15 坐标 / Q0.8 颜色定点指令。外部只承担真实 GPU/CPU 本来也
需要的系统组件：像素坐标、只读纹理 RAM、SIMD lane 调度和 framebuffer。

纯度约束：
1. 纹理读取严格走 ``LOAD -> RAM -> MOVI``，repeat、mipmap 坐标和双线性插值由
   Neural CPU 指令计算；
2. 坐标变换、fBm 累加、smoothstep 与颜色混合只使用训练过的
   ADD/ADC/SUB/SBC/AND/OR/XOR/SHL/SHR/CMP；
3. 不使用乘法 LUT、三角函数 LUT、由 PyTorch 代算的 shader 中间量或降分辨率放大；
4. ``torch.where`` 只承担 SIMD 分支路由，其条件来自 Neural CPU 的 CMP flags。

原 shader：Sinuosity 的 ShaderToy《2D Fast Clouds》（2014）：
https://www.shadertoy.com/view/XsjSRt 。输入纹理为 ShaderToy 的
Gray Noise Medium；仓库内附一份用户提供的 256x256 原始纹理。

所有运行参数都集中在文件顶部的 ``Config`` 中。将 ``REFERENCE_ONLY`` 改为
``True`` 可只生成精确整数参考并核算操作量。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


INPUT_BITS = 51
OUTPUT_BITS = 39
STATE_BITS = 35
NUM_FLAGS = 3
NUM_REGISTERS = 4
REGISTER_BITS = 8

FLAG_ZF = 0
FLAG_GF = 1
FLAG_CF = 2

OPCODES = {
    "MOVI": 3,
    "LOAD": 4,
    "ADD": 6,
    "ADC": 7,
    "SUB": 8,
    "SBC": 9,
    "AND": 12,
    "OR": 13,
    "XOR": 14,
    "SHL": 16,
    "SHR": 17,
    "CMP": 18,
}

RRR_OPCODES = {
    OPCODES[name]
    for name in ("LOAD", "ADD", "ADC", "SUB", "SBC", "AND", "OR", "XOR", "SHL", "SHR", "CMP")
}
RI8_OPCODES = {OPCODES["MOVI"]}

CTRL_MEMORY_READ = 0


def script_directory() -> Path:
    source = globals().get("__file__")
    if not source:
        return Path.cwd().resolve()
    source_dir = Path(source).resolve().parent
    return source_dir.parent if source_dir.name == "scripts" else source_dir


class Config:
    BASE_DIR = script_directory()
    MODEL_PATH = BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    RESULT_DIR = BASE_DIR / "results" / "clouds"
    TEXTURE_PATH = BASE_DIR / "assets" / "gray_noise_medium.png"
    WIDTH = 1280
    HEIGHT = 720
    TIME_SECONDS = 32.95
    TILE_PIXELS = 65_536
    INTERPOLATION_BITS = 7
    TRILINEAR_PRIMARY = True
    TRILINEAR_CURL = True
    REFERENCE_ONLY = False
    STOP_ON_ERROR = False
    ALLOW_TF32 = False
    MAX_LANE_OPERATIONS = 12_000_000_000
    ALLOW_OVER_BUDGET = False


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def load_model(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 Neural CPU 模型：{Config.MODEL_PATH}")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    linear_weights = [
        value
        for key, value in state_dict.items()
        if key.endswith("weight") and value.ndim == 2
    ]
    first = next(value for value in linear_weights if value.shape[1] == INPUT_BITS)
    hidden_size = int(config.get("HIDDEN_SIZE", first.shape[0]))
    hidden_layers = int(config.get("HIDDEN_LAYERS", len(linear_weights) - 2))
    dropout = float(config.get("DROPOUT", 0.0))
    model = NeuralCPUCore(hidden_size, hidden_layers, dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {
        "path": str(Config.MODEL_PATH),
        "step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
        "hidden_size": hidden_size,
        "hidden_linear_layers": hidden_layers + 1,
        "dropout": dropout,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def encode_instruction(
    opcode: int,
    rd: int = 0,
    ra: int = 1,
    rb: int = 2,
    immediate: int | torch.Tensor = 0,
) -> int | torch.Tensor:
    base = opcode << 11
    if opcode in RRR_OPCODES:
        return base | ((rd & 3) << 9) | ((ra & 3) << 7) | ((rb & 3) << 5)
    if opcode in RI8_OPCODES:
        prefix = base | ((rd & 3) << 9)
        if isinstance(immediate, torch.Tensor):
            return prefix | (immediate.to(torch.int64) & 0xFF)
        return prefix | (int(immediate) & 0xFF)
    raise ValueError(f"未定义 opcode={opcode} 的指令编码格式。")


def verify_instruction_encoding() -> None:
    """防止寄存器字段污染立即数等静默编码错误。"""
    immediate = torch.arange(256, dtype=torch.int64)
    movi = encode_instruction(OPCODES["MOVI"], rd=0, immediate=immediate)
    if not torch.equal(movi & 0xFF, immediate):
        raise RuntimeError("MOVI 编码自检失败：立即数字段被其他字段污染。")
    add = int(encode_instruction(OPCODES["ADD"], rd=0, ra=1, rb=2))
    if ((add >> 9) & 3, (add >> 7) & 3, (add >> 5) & 3) != (0, 1, 2):
        raise RuntimeError("RRR 指令编码自检失败。")


@dataclass
class AuditSummary:
    neural_calls: int = 0
    lane_operations: int = 0
    register_bit_errors: int = 0
    flag_bit_errors: int = 0
    control_bit_errors: int = 0

    @property
    def total_errors(self) -> int:
        return self.register_bit_errors + self.flag_bit_errors + self.control_bit_errors


class ExactALU:
    """精确 8-bit ISA；同时统计等价指令 lane-op。"""

    def __init__(self, device: torch.device):
        self.device = device
        self.audit = AuditSummary()
        self.texture_mips: list[torch.Tensor] = []

    def install_texture_mips(self, texture_mips: list[np.ndarray]) -> None:
        self.texture_mips = []
        for mip in texture_mips:
            size = int(mip.shape[0])
            table = torch.zeros((256, 256), device=self.device, dtype=torch.uint8)
            table[:size, :size] = torch.from_numpy(
                np.array(mip, dtype=np.uint8, copy=True)
            ).to(self.device, torch.uint8)
            self.texture_mips.append(table.reshape(-1))

    def _count(self, value: torch.Tensor) -> None:
        self.audit.neural_calls += 1
        self.audit.lane_operations += value.numel()

    def binary(
        self,
        name: str,
        left: torch.Tensor,
        right: torch.Tensor,
        carry: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = left.to(self.device, torch.uint8).reshape(-1)
        right = right.to(self.device, torch.uint8).reshape(-1)
        self._count(left)
        carry_u8 = (
            torch.zeros_like(left)
            if carry is None
            else carry.to(self.device, torch.uint8).reshape(-1)
        )
        l16 = left.to(torch.int16)
        r16 = right.to(torch.int16)
        c16 = carry_u8.to(torch.int16)
        flags = torch.zeros((left.numel(), NUM_FLAGS), device=self.device, dtype=torch.uint8)
        if name in ("ADD", "ADC"):
            total = l16 + r16 + (c16 if name == "ADC" else 0)
            result = (total & 255).to(torch.uint8)
            flags[:, FLAG_ZF] = (result == 0).to(torch.uint8)
            flags[:, FLAG_CF] = (total > 255).to(torch.uint8)
        elif name in ("SUB", "SBC"):
            borrow = c16 if name == "SBC" else 0
            total = l16 - r16 - borrow
            result = (total & 255).to(torch.uint8)
            flags[:, FLAG_ZF] = (result == 0).to(torch.uint8)
            flags[:, FLAG_CF] = (l16 < r16 + borrow).to(torch.uint8)
        elif name == "AND":
            result = torch.bitwise_and(left, right)
            flags[:, FLAG_ZF] = (result == 0).to(torch.uint8)
        elif name == "OR":
            result = torch.bitwise_or(left, right)
            flags[:, FLAG_ZF] = (result == 0).to(torch.uint8)
        elif name == "XOR":
            result = torch.bitwise_xor(left, right)
            flags[:, FLAG_ZF] = (result == 0).to(torch.uint8)
        elif name in ("SHL", "SHR"):
            shift = (right & 7).to(torch.int16)
            if name == "SHL":
                result = ((l16 << shift) & 255).to(torch.uint8)
                shifted_carry = torch.where(
                    shift == 0,
                    c16,
                    (l16 >> (8 - shift)) & 1,
                )
            else:
                result = (l16 >> shift).to(torch.uint8)
                shifted_carry = torch.where(
                    shift == 0,
                    c16,
                    (l16 >> (shift - 1)) & 1,
                )
            flags[:, FLAG_ZF] = (result == 0).to(torch.uint8)
            flags[:, FLAG_CF] = shifted_carry.to(torch.uint8)
        else:
            raise ValueError(f"不支持的指令：{name}")
        return result, flags

    def greater(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.to(self.device, torch.uint8).reshape(-1)
        right = right.to(self.device, torch.uint8).reshape(-1)
        self._count(left)
        return left > right

    def texture_load(
        self, level: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if not self.texture_mips:
            raise RuntimeError("尚未安装纹理 mip pyramid。")
        x = x.to(self.device, torch.uint8).reshape(-1)
        y = y.to(self.device, torch.uint8).reshape(-1)
        self._count(x)  # LOAD
        self._count(x)  # RAM 返回后执行 MOVI
        address = (y.to(torch.int64) << 8) | x.to(torch.int64)
        return self.texture_mips[level][address]


class NeuralALU:
    """在每个像素 lane 上执行 Neural CPU，并与精确 ISA 同步审计。"""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.audit = AuditSummary()
        self._shift16 = torch.arange(15, -1, -1, device=device)
        self._shift8 = torch.arange(7, -1, -1, device=device)
        self._register_error_terms: list[torch.Tensor] = []
        self._flag_error_terms: list[torch.Tensor] = []
        self._control_error_terms: list[torch.Tensor] = []
        self.texture_mips: list[torch.Tensor] = []

    def install_texture_mips(self, texture_mips: list[np.ndarray]) -> None:
        self.texture_mips = []
        for mip in texture_mips:
            size = int(mip.shape[0])
            table = torch.zeros((256, 256), device=self.device, dtype=torch.uint8)
            table[:size, :size] = torch.from_numpy(
                np.array(mip, dtype=np.uint8, copy=True)
            ).to(self.device, torch.uint8)
            self.texture_mips.append(table.reshape(-1))

    def _instruction_bits(
        self, instruction: int | torch.Tensor, size: int
    ) -> torch.Tensor:
        if isinstance(instruction, torch.Tensor):
            values = instruction.to(self.device, torch.int64).reshape(-1)
            if values.numel() == 1 and size != 1:
                values = values.expand(size)
        else:
            values = torch.full(
                (size,), instruction, device=self.device, dtype=torch.int64
            )
        return ((values[:, None] >> self._shift16[None, :]) & 1).to(torch.float32)

    def _state_bits(self, registers: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        bits = (
            (registers.to(torch.int64)[:, :, None] >> self._shift8[None, None, :]) & 1
        ).reshape(registers.shape[0], -1)
        return torch.cat((flags.to(torch.float32), bits.to(torch.float32)), dim=1)

    @torch.inference_mode()
    def _execute(
        self,
        name: str,
        registers: torch.Tensor,
        flags: torch.Tensor,
        instruction: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = registers.shape[0]
        if instruction is None:
            instruction = encode_instruction(OPCODES[name])
        inputs = torch.cat(
            (self._instruction_bits(instruction, size), self._state_bits(registers, flags)),
            dim=1,
        )
        bits = self.model(inputs) >= 0
        out_flags = bits[:, :NUM_FLAGS].to(torch.uint8)
        register_bits = bits[:, NUM_FLAGS:STATE_BITS].reshape(size, 4, 8)
        weights = (1 << self._shift8).to(torch.int64)
        out_registers = torch.sum(
            register_bits.to(torch.int64) * weights[None, None, :], dim=2
        ).to(torch.uint8)
        controls = bits[:, STATE_BITS:OUTPUT_BITS].to(torch.uint8)
        self.audit.neural_calls += 1
        self.audit.lane_operations += size
        return out_registers, out_flags, controls

    def _record(
        self,
        actual_registers: torch.Tensor,
        expected_registers: torch.Tensor,
        actual_flags: torch.Tensor,
        expected_flags: torch.Tensor,
        controls: torch.Tensor,
        name: str,
        expected_controls: torch.Tensor | None = None,
    ) -> None:
        register_xor = torch.bitwise_xor(actual_registers, expected_registers).to(torch.int64)
        register_bits = sum((register_xor >> bit) & 1 for bit in range(8))
        register_errors = torch.sum(register_bits)
        flag_errors = torch.count_nonzero(actual_flags != expected_flags)
        if expected_controls is None:
            expected_controls = torch.zeros_like(controls)
        control_errors = torch.count_nonzero(controls != expected_controls)
        self._register_error_terms.append(register_errors)
        self._flag_error_terms.append(flag_errors)
        self._control_error_terms.append(control_errors)
        if Config.STOP_ON_ERROR:
            total = int((register_errors + flag_errors + control_errors).item())
            if total:
                raise RuntimeError(f"Neural CPU 指令 {name} 出现 {total} 个 bit/control 错误。")

    def binary(
        self,
        name: str,
        left: torch.Tensor,
        right: torch.Tensor,
        carry: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = left.to(self.device, torch.uint8).reshape(-1)
        right = right.to(self.device, torch.uint8).reshape(-1)
        size = left.numel()
        registers = torch.zeros((size, 4), device=self.device, dtype=torch.uint8)
        registers[:, 1] = left
        registers[:, 2] = right
        flags = torch.zeros((size, 3), device=self.device, dtype=torch.uint8)
        if carry is not None:
            flags[:, FLAG_CF] = carry.to(self.device, torch.uint8).reshape(-1)
        out_registers, out_flags, controls = self._execute(name, registers, flags)

        exact = ExactALU(self.device)
        expected, expected_flags = exact.binary(name, left, right, flags[:, FLAG_CF])
        expected_registers = registers.clone()
        expected_registers[:, 0] = expected
        self._record(
            out_registers,
            expected_registers,
            out_flags,
            expected_flags,
            controls,
            name,
        )
        return out_registers[:, 0], out_flags

    def greater(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.to(self.device, torch.uint8).reshape(-1)
        right = right.to(self.device, torch.uint8).reshape(-1)
        size = left.numel()
        registers = torch.zeros((size, 4), device=self.device, dtype=torch.uint8)
        registers[:, 1] = left
        registers[:, 2] = right
        flags = torch.zeros((size, 3), device=self.device, dtype=torch.uint8)
        out_registers, out_flags, controls = self._execute("CMP", registers, flags)
        expected_flags = flags.clone()
        expected_flags[:, FLAG_ZF] = (left == right).to(torch.uint8)
        expected_flags[:, FLAG_GF] = (left > right).to(torch.uint8)
        expected_flags[:, FLAG_CF] = (left < right).to(torch.uint8)
        self._record(
            out_registers,
            registers,
            out_flags,
            expected_flags,
            controls,
            "CMP",
        )
        return out_flags[:, FLAG_GF].bool()

    def texture_load(
        self, level: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if not self.texture_mips:
            raise RuntimeError("尚未安装纹理 mip pyramid。")
        x = x.to(self.device, torch.uint8).reshape(-1)
        y = y.to(self.device, torch.uint8).reshape(-1)
        size = x.numel()
        registers = torch.zeros((size, 4), device=self.device, dtype=torch.uint8)
        registers[:, 1] = y
        registers[:, 2] = x
        flags = torch.zeros((size, 3), device=self.device, dtype=torch.uint8)

        after_load, load_flags, controls = self._execute("LOAD", registers, flags)
        expected_controls = torch.zeros_like(controls)
        expected_controls[:, CTRL_MEMORY_READ] = 1
        self._record(
            after_load,
            registers,
            load_flags,
            flags,
            controls,
            "LOAD.texture",
            expected_controls,
        )

        address = (y.to(torch.int64) << 8) | x.to(torch.int64)
        loaded = self.texture_mips[level][address]
        movi = encode_instruction(OPCODES["MOVI"], rd=0, immediate=loaded)
        after_movi, movi_flags, movi_controls = self._execute(
            "MOVI", after_load, load_flags, movi
        )
        expected_registers = after_load.clone()
        expected_registers[:, 0] = loaded
        self._record(
            after_movi,
            expected_registers,
            movi_flags,
            load_flags,
            movi_controls,
            "MOVI.texture",
        )
        return after_movi[:, 0]

    def finalize_audit(self) -> AuditSummary:
        if self._register_error_terms:
            self.audit.register_bit_errors = int(
                torch.stack(self._register_error_terms).sum().item()
            )
            self.audit.flag_bit_errors = int(torch.stack(self._flag_error_terms).sum().item())
            self.audit.control_bit_errors = int(
                torch.stack(self._control_error_terms).sum().item()
            )
        return self.audit


def byte_constant(reference: torch.Tensor, value: int) -> torch.Tensor:
    return torch.full_like(reference, value & 255, dtype=torch.uint8)


def add8(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result, flags = alu.binary("ADD", a, b)
    return result, flags[:, FLAG_CF]


def adc8(
    alu: ExactALU | NeuralALU,
    a: torch.Tensor,
    b: torch.Tensor,
    carry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result, flags = alu.binary("ADC", a, b, carry)
    return result, flags[:, FLAG_CF]


def sub8(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return alu.binary("SUB", a, b)[0]


def bit_and(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return alu.binary("AND", a, b)[0]


def bit_or(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return alu.binary("OR", a, b)[0]


def bit_xor(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return alu.binary("XOR", a, b)[0]


def shl8(alu: ExactALU | NeuralALU, value: torch.Tensor, amount: int) -> torch.Tensor:
    return alu.binary("SHL", value, byte_constant(value, amount))[0]


def shr8(alu: ExactALU | NeuralALU, value: torch.Tensor, amount: int) -> torch.Tensor:
    return alu.binary("SHR", value, byte_constant(value, amount))[0]


def sat_add(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result, carry = add8(alu, a, b)
    return torch.where(carry.bool(), byte_constant(result, 255), result)


def absdiff(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_gt_b = alu.greater(a, b)
    ab = sub8(alu, a, b)
    ba = sub8(alu, b, a)
    return torch.where(a_gt_b, ab, ba)


def minmax(
    alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a_gt_b = alu.greater(a, b)
    return torch.where(a_gt_b, b, a), torch.where(a_gt_b, a, b)


U16 = tuple[torch.Tensor, torch.Tensor]  # low byte, high byte
Coord32 = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
COORD_FRACTION_BITS = 15


def u16_from_int(reference: torch.Tensor, value: int) -> U16:
    return byte_constant(reference, value), byte_constant(reference, value >> 8)


def add16(alu: ExactALU | NeuralALU, a: U16, b: U16) -> U16:
    low, carry = add8(alu, a[0], b[0])
    high, _ = adc8(alu, a[1], b[1], carry)
    return low, high


def coord_from_float(reference: torch.Tensor, value: float) -> Coord32:
    raw = int(round(value * (1 << COORD_FRACTION_BITS))) & 0xFFFFFFFF
    return tuple(byte_constant(reference, raw >> (8 * index)) for index in range(4))  # type: ignore[return-value]


def coord_from_u16(value: U16) -> Coord32:
    zero = byte_constant(value[0], 0)
    return value[0], value[1], zero, zero


def add32(
    alu: ExactALU | NeuralALU, a: Coord32, b: Coord32
) -> Coord32:
    out0, carry = add8(alu, a[0], b[0])
    out1, carry = adc8(alu, a[1], b[1], carry)
    out2, carry = adc8(alu, a[2], b[2], carry)
    out3, _ = adc8(alu, a[3], b[3], carry)
    return out0, out1, out2, out3


def sub32(
    alu: ExactALU | NeuralALU, a: Coord32, b: Coord32
) -> Coord32:
    out0, flags = alu.binary("SUB", a[0], b[0])
    out1, flags = alu.binary("SBC", a[1], b[1], flags[:, FLAG_CF])
    out2, flags = alu.binary("SBC", a[2], b[2], flags[:, FLAG_CF])
    out3, _ = alu.binary("SBC", a[3], b[3], flags[:, FLAG_CF])
    return out0, out1, out2, out3


def neg32(alu: ExactALU | NeuralALU, value: Coord32) -> Coord32:
    zero = byte_constant(value[0], 0)
    zeros: Coord32 = (zero, zero, zero, zero)
    return sub32(alu, zeros, value)


def asr32(
    alu: ExactALU | NeuralALU, value: Coord32, amount: int
) -> Coord32:
    if amount <= 0:
        return value
    sign = alu.greater(value[3], byte_constant(value[3], 127))
    fill = torch.where(
        sign,
        byte_constant(value[3], 255),
        byte_constant(value[3], 0),
    )
    parts = list(value)
    while amount >= 8:
        parts = [parts[1], parts[2], parts[3], fill]
        amount -= 8
    if amount:
        shifted: list[torch.Tensor] = []
        for index in range(3):
            shifted.append(
                bit_or(
                    alu,
                    shr8(alu, parts[index], amount),
                    shl8(alu, parts[index + 1], 8 - amount),
                )
            )
        shifted.append(
            bit_or(
                alu,
                shr8(alu, parts[3], amount),
                shl8(alu, fill, 8 - amount),
            )
        )
        parts = shifted
    return parts[0], parts[1], parts[2], parts[3]


def coord_to_texture_u16(
    alu: ExactALU | NeuralALU, value: Coord32
) -> U16:
    """Q17.15 归一化坐标转为周期纹理的 Q0.16 地址。"""
    low = shl8(alu, value[0], 1)
    high = bit_or(alu, shl8(alu, value[1], 1), shr8(alu, value[0], 7))
    return low, high


def shl16_1(alu: ExactALU | NeuralALU, value: U16) -> U16:
    low, flags = alu.binary("SHL", value[0], byte_constant(value[0], 1))
    high, _ = adc8(alu, value[1], value[1], flags[:, FLAG_CF])
    return low, high


def shl16(alu: ExactALU | NeuralALU, value: U16, amount: int) -> U16:
    result = value
    for _ in range(amount):
        result = shl16_1(alu, result)
    return result


def mul_u8_small(
    alu: ExactALU | NeuralALU, value: torch.Tensor, factor: torch.Tensor
) -> U16:
    """用移位、CMP 路由和 ADD/ADC 计算 value * factor，factor 必须在 [0,16]。"""
    zero = byte_constant(value, 0)
    acc: U16 = (zero, zero)
    for bit in range(5):
        flag = bit_and(alu, factor, byte_constant(factor, 1 << bit))
        present = alu.greater(flag, zero)
        if bit == 0:
            term = (value, zero)
        else:
            term = (shl8(alu, value, bit), shr8(alu, value, 8 - bit))
        selected = (
            torch.where(present, term[0], zero),
            torch.where(present, term[1], zero),
        )
        acc = add16(alu, acc, selected)
    return acc


def lerp_u8(
    alu: ExactALU | NeuralALU,
    a: torch.Tensor,
    b: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """定点插值：a ± abs(b-a)*t/(2**bits)，仅需一次小乘法。"""
    bits = Config.INTERPOLATION_BITS
    if not 4 <= bits <= 8:
        raise ValueError("Config.INTERPOLATION_BITS 必须在 [4, 8]。")
    t = fraction if bits == 8 else shr8(alu, fraction, 8 - bits)
    b_above_a = alu.greater(b, a)
    a_minus_b = sub8(alu, a, b)
    b_minus_a = sub8(alu, b, a)
    difference = torch.where(b_above_a, b_minus_a, a_minus_b)

    zero = byte_constant(difference, 0)
    product: U16 = (zero, zero)
    term: U16 = (difference, zero)
    for bit in range(bits):
        bit_value = bit_and(alu, t, byte_constant(t, 1 << bit))
        present = alu.greater(bit_value, zero)
        selected = (
            torch.where(present, term[0], zero),
            torch.where(present, term[1], zero),
        )
        product = add16(alu, product, selected)
        if bit != bits - 1:
            term = shl16_1(alu, term)
    if bits == 8:
        step = product[1]
    else:
        step = bit_or(
            alu,
            shr8(alu, product[0], bits),
            shl8(alu, product[1], 8 - bits),
        )
    upward = add8(alu, a, step)[0]
    downward = sub8(alu, a, step)
    return torch.where(b_above_a, upward, downward)


def sub16(alu: ExactALU | NeuralALU, a: U16, b: U16) -> U16:
    low, flags = alu.binary("SUB", a[0], b[0])
    high, _ = alu.binary("SBC", a[1], b[1], flags[:, FLAG_CF])
    return low, high


def shr16_logical(
    alu: ExactALU | NeuralALU, value: U16, amount: int
) -> U16:
    if amount <= 0:
        return value
    zero = byte_constant(value[0], 0)
    if amount >= 16:
        return zero, zero
    if amount >= 8:
        return shr8(alu, value[1], amount - 8), zero
    low = bit_or(
        alu,
        shr8(alu, value[0], amount),
        shl8(alu, value[1], 8 - amount),
    )
    return low, shr8(alu, value[1], amount)


def asr16(alu: ExactALU | NeuralALU, value: U16, amount: int) -> U16:
    """16-bit two's-complement 算术右移；符号判断来自 CMP。"""
    if amount <= 0:
        return value
    sign = alu.greater(value[1], byte_constant(value[1], 127))
    sign_mask = torch.where(
        sign,
        byte_constant(value[1], 255),
        byte_constant(value[1], 0),
    )
    if amount >= 16:
        return sign_mask, sign_mask
    if amount >= 8:
        shift = amount - 8
        if shift == 0:
            low = value[1]
        else:
            fill_bits = ((1 << shift) - 1) << (8 - shift)
            low = bit_or(
                alu,
                shr8(alu, value[1], shift),
                bit_and(alu, sign_mask, byte_constant(sign_mask, fill_bits)),
            )
        return low, sign_mask
    low = bit_or(
        alu,
        shr8(alu, value[0], amount),
        shl8(alu, value[1], 8 - amount),
    )
    fill_bits = ((1 << amount) - 1) << (8 - amount)
    high = bit_or(
        alu,
        shr8(alu, value[1], amount),
        bit_and(alu, sign_mask, byte_constant(sign_mask, fill_bits)),
    )
    return low, high


def neg16(alu: ExactALU | NeuralALU, value: U16) -> U16:
    zero = u16_from_int(value[0], 0)
    return sub16(alu, zero, value)


def mul_u8_constant_u16(
    alu: ExactALU | NeuralALU, value: torch.Tensor, factor: int
) -> U16:
    """把固定常数乘法编译为移位和 ADD，不使用乘法表。"""
    zero = byte_constant(value, 0)
    acc: U16 = (zero, zero)
    term: U16 = (value, zero)
    remaining = int(factor)
    bit = 0
    while remaining:
        if remaining & 1:
            acc = add16(alu, acc, term)
        remaining >>= 1
        bit += 1
        if remaining:
            term = shl16_1(alu, term)
    return acc


def mul_u8_q8(
    alu: ExactALU | NeuralALU, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """用 8 次 bit 选择实现 round(left*right/255)。"""
    zero = byte_constant(left, 0)
    acc: U16 = (zero, zero)
    term: U16 = (left, zero)
    for bit in range(8):
        present_bits = bit_and(alu, right, byte_constant(right, 1 << bit))
        present = alu.greater(present_bits, zero)
        selected = (
            torch.where(present, term[0], zero),
            torch.where(present, term[1], zero),
        )
        acc = add16(alu, acc, selected)
        if bit != 7:
            term = shl16_1(alu, term)
    acc = add16(alu, acc, u16_from_int(left, 128))
    acc = add16(alu, acc, shr16_logical(alu, acc, 8))
    return acc[1]


def sat_sub(alu: ExactALU | NeuralALU, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    keep = alu.greater(a, b)
    value = sub8(alu, a, b)
    return torch.where(keep, value, byte_constant(value, 0))


def smoothstep_u8(
    alu: ExactALU | NeuralALU,
    value: torch.Tensor,
    edge0: int,
    edge1: int,
) -> torch.Tensor:
    """Q0.8 smoothstep；当前 shader 的两个区间恰好可用乘 5 编译。"""
    above0 = alu.greater(value, byte_constant(value, edge0))
    at_or_above1 = alu.greater(value, byte_constant(value, edge1 - 1))
    delta = sub8(alu, value, byte_constant(value, edge0))
    scaled = mul_u8_constant_u16(alu, delta, 5)
    if edge1 - edge0 == 102:
        scaled = shr16_logical(alu, scaled, 1)
    elif edge1 - edge0 != 51:
        raise ValueError("smoothstep_u8 只为当前 shader 的 102/51 两种跨度编译。")
    t = torch.where(above0, scaled[0], byte_constant(value, 0))
    t = torch.where(at_or_above1, byte_constant(value, 255), t)
    t2 = mul_u8_q8(alu, t, t)
    t3 = mul_u8_q8(alu, t2, t)
    three_t2 = add16(
        alu,
        (t2, byte_constant(t2, 0)),
        shl16_1(alu, (t2, byte_constant(t2, 0))),
    )
    two_t3 = shl16_1(alu, (t3, byte_constant(t3, 0)))
    return sub16(alu, three_t2, two_t3)[0]


def texture_bilinear(
    alu: ExactALU | NeuralALU,
    x: Coord32,
    y: Coord32,
    level: int,
) -> torch.Tensor:
    """指定 mip 内的 repeat + 定点双线性；每个 texel 都经 LOAD/MOVI。"""
    sx = shr16_logical(alu, coord_to_texture_u16(alu, x), level)
    sy = shr16_logical(alu, coord_to_texture_u16(alu, y), level)
    ix, iy = sx[1], sy[1]
    fx, fy = sx[0], sy[0]
    mask = (256 >> level) - 1
    ix1 = bit_and(alu, add8(alu, ix, byte_constant(ix, 1))[0], byte_constant(ix, mask))
    iy1 = bit_and(alu, add8(alu, iy, byte_constant(iy, 1))[0], byte_constant(iy, mask))
    p00 = alu.texture_load(level, ix, iy)
    p10 = alu.texture_load(level, ix1, iy)
    p01 = alu.texture_load(level, ix, iy1)
    p11 = alu.texture_load(level, ix1, iy1)
    top = lerp_u8(alu, p00, p10, fx)
    bottom = lerp_u8(alu, p01, p11, fx)
    return lerp_u8(alu, top, bottom, fy)


def texture_trilinear(
    alu: ExactALU | NeuralALU,
    x: Coord32,
    y: Coord32,
    lower_level: int,
    upper_level: int,
    upper_fraction: int,
) -> torch.Tensor:
    """先在两个 mip 内做双线性采样，再通过 ISA 在层间插值。"""
    lower = texture_bilinear(alu, x, y, lower_level)
    if lower_level == upper_level or upper_fraction <= 0:
        return lower
    upper = texture_bilinear(alu, x, y, upper_level)
    return lerp_u8(
        alu,
        lower,
        upper,
        byte_constant(lower, min(upper_fraction, 255)),
    )


def scale_coord_point2(
    alu: ExactALU | NeuralALU, value: Coord32
) -> Coord32:
    """0.19921875*x，作为 shader 中 uv*0.2 的定点近似。"""
    result = add32(alu, asr32(alu, value, 3), asr32(alu, value, 4))
    result = add32(alu, result, asr32(alu, value, 7))
    return add32(alu, result, asr32(alu, value, 8))


def rotate_curl(
    alu: ExactALU | NeuralALU,
    x: Coord32,
    y: Coord32,
    noise_profile: tuple[int, int, int],
) -> tuple[U16, U16]:
    # GLSL: uv += noise(uv*.2)*.005
    sample_x = scale_coord_point2(alu, x)
    sample_y = scale_coord_point2(alu, y)
    lower_lod, upper_lod, lod_fraction = noise_profile
    if Config.TRILINEAR_CURL:
        noise = texture_trilinear(
            alu,
            sample_x,
            sample_y,
            lower_lod,
            upper_lod,
            lod_fraction,
        )
    else:
        nearest_lod = upper_lod if lod_fraction >= 128 else lower_lod
        noise = texture_bilinear(alu, sample_x, sample_y, nearest_lod)
    # noise/255*0.005 在 Q17.15 中约为 noise*0.6425。
    displacement = add8(alu, shr8(alu, noise, 1), shr8(alu, noise, 3))[0]
    displacement = add8(alu, displacement, shr8(alu, noise, 6))[0]
    delta = coord_from_u16((displacement, byte_constant(displacement, 0)))
    x = add32(alu, x, delta)
    y = add32(alu, y, delta)

    # rot=3.0：cos≈-1+1/128，sin≈1/8+1/64。
    x3, x6, x7 = asr32(alu, x, 3), asr32(alu, x, 6), asr32(alu, x, 7)
    y3, y6, y7 = asr32(alu, y, 3), asr32(alu, y, 6), asr32(alu, y, 7)
    out_x = add32(alu, neg32(alu, x), x7)
    out_x = sub32(alu, sub32(alu, out_x, y3), y6)
    out_y = add32(alu, neg32(alu, y), y7)
    out_y = add32(alu, add32(alu, out_y, x3), x6)
    return out_x, out_y


def triple32(alu: ExactALU | NeuralALU, value: Coord32) -> Coord32:
    return add32(alu, value, add32(alu, value, value))


def fbm8(
    alu: ExactALU | NeuralALU,
    x: Coord32,
    y: Coord32,
    time_seconds: float,
) -> torch.Tensor:
    # 由纹理坐标对屏幕像素的导数估计隐式 LOD。最后三个 octave
    # 分别落在 1.64、3.22、4.80，原版 GPU texture() 会在相邻 mip 间插值。
    primary_lod_profiles = (
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (1, 2, 164),
        (3, 4, 56),
        (4, 5, 205),
    )
    # curl 内部先乘 3、再缩放约 0.2；末三层隐式 LOD 约为
    # 0.90、2.48、4.06，也需要跨 mip 连续过渡。
    curl_lod_profiles = (
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 1, 230),
        (2, 3, 124),
        (4, 5, 18),
    )
    total = byte_constant(x[0], 0)
    for octave in range(8):
        mul = 0.5 ** (octave + 1)
        offset = time_seconds * 0.0015 * (1.0 - mul)
        sample_x = add32(alu, x, coord_from_float(x[0], offset))
        sample_y = add32(alu, y, coord_from_float(y[0], offset))
        lower_lod, upper_lod, lod_fraction = primary_lod_profiles[octave]
        if Config.TRILINEAR_PRIMARY:
            noise = texture_trilinear(
                alu,
                sample_x,
                sample_y,
                lower_lod,
                upper_lod,
                lod_fraction,
            )
        else:
            nearest_lod = upper_lod if lod_fraction >= 128 else lower_lod
            noise = texture_bilinear(alu, sample_x, sample_y, nearest_lod)
        if octave < 7:
            weighted = shr8(alu, noise, octave + 1)
        else:
            present = alu.greater(noise, byte_constant(noise, 127))
            weighted = torch.where(
                present, byte_constant(noise, 1), byte_constant(noise, 0)
            )
        total = add8(alu, total, weighted)[0]
        if octave != 7:
            x, y = triple32(alu, x), triple32(alu, y)
            x, y = rotate_curl(alu, x, y, curl_lod_profiles[octave])
    return total


def scale_pixel_to_uv(alu: ExactALU | NeuralALU, pixel: U16) -> Coord32:
    """近似 pixel*32768/20000，生成 Q17.15 坐标。"""
    result = pixel
    for shift in (1, 3, 7, 8, 10, 11, 13):
        result = add16(alu, result, shr16_logical(alu, pixel, shift))
    return coord_from_u16(result)


def render_cloud_program(
    alu: ExactALU | NeuralALU,
    pixel_x: U16,
    pixel_y: U16,
    time_seconds: float,
) -> torch.Tensor:
    uv_x = scale_pixel_to_uv(alu, pixel_x)
    uv_y = scale_pixel_to_uv(alu, pixel_y)
    color1_offset = -0.5 + time_seconds * 0.0004
    color2_offset = -10.5 + time_seconds * 0.0002
    c1x = add32(alu, uv_x, coord_from_float(uv_x[0], color1_offset))
    c1y = add32(alu, uv_y, coord_from_float(uv_y[0], color1_offset))
    c2x = add32(alu, uv_x, coord_from_float(uv_x[0], color2_offset))
    c2y = add32(alu, uv_y, coord_from_float(uv_y[0], color2_offset))
    color1 = fbm8(alu, c1x, c1y, time_seconds)
    color2 = fbm8(alu, c2x, c2y, time_seconds)

    clouds1 = smoothstep_u8(alu, color1, 128, 230)
    clouds2 = smoothstep_u8(alu, color2, 128, 179)
    cloud_form = sat_add(alu, clouds1, clouds2)

    # cloudCol = saturate((1 - color1*0.2) * 1.3)
    dark = add8(alu, shr8(alu, color1, 3), shr8(alu, color1, 4))[0]
    dark = add8(alu, dark, shr8(alu, color1, 7))[0]
    cloud = sub8(alu, byte_constant(color1, 255), dark)
    cloud = sat_add(alu, cloud, shr8(alu, cloud, 2))
    cloud = sat_add(alu, cloud, shr8(alu, cloud, 5))
    cloud = sat_add(alu, cloud, shr8(alu, cloud, 6))

    sky = (153, 204, 255)
    quarter = byte_constant(cloud, 64)
    cloud2_rgb = tuple(
        lerp_u8(alu, cloud, byte_constant(cloud, channel), quarter)
        for channel in sky
    )
    cloud_mix = sat_sub(alu, clouds2, clouds1)
    combined = tuple(
        lerp_u8(alu, cloud, channel, cloud_mix) for channel in cloud2_rgb
    )
    result = tuple(
        lerp_u8(alu, byte_constant(cloud, sky_channel), channel, cloud_form)
        for sky_channel, channel in zip(sky, combined)
    )
    return torch.stack(result, dim=1)


def current_directory() -> Path:
    source = globals().get("__file__")
    return Path(source).resolve().parent if source else Path.cwd()


def locate_texture() -> Path:
    if Config.TEXTURE_PATH.exists():
        return Config.TEXTURE_PATH.resolve()
    raise FileNotFoundError(
        f"找不到 Gray Noise Medium 纹理：{Config.TEXTURE_PATH}\n"
        "请修改 Config.TEXTURE_PATH。"
    )


def load_texture_mips(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.shape != (256, 256):
        raise ValueError(f"纹理必须为 256x256，实际为 {gray.shape[::-1]}。")
    channel_spread = int(
        np.max(rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16))
    )
    mips = [gray]
    current = Image.fromarray(gray)
    while current.width > 1:
        size = current.width // 2
        current = current.resize((size, size), Image.Resampling.BOX)
        mips.append(np.asarray(current, dtype=np.uint8))
    return mips, {
        "path": str(path),
        "width": 256,
        "height": 256,
        "source_mode": "RGB",
        "max_rgb_channel_spread": channel_spread,
        "mip_sizes": [int(mip.shape[0]) for mip in mips],
        "sampling": (
            "repeat + "
            f"{'trilinear primary mip' if Config.TRILINEAR_PRIMARY else 'nearest primary mip'} + "
            f"{'trilinear curl mip' if Config.TRILINEAR_CURL else 'nearest curl mip'} + "
            f"{Config.INTERPOLATION_BITS}-bit bilinear"
        ),
    }


def make_coordinates(start: int, end: int, device: torch.device) -> tuple[U16, U16]:
    """外部只编码整数像素坐标；归一化由 render_cloud_program 的 ISA 完成。"""
    indices = torch.arange(start, end, device=device, dtype=torch.int64)
    px = indices % Config.WIDTH
    py = indices // Config.WIDTH
    return (
        ((px & 255).to(torch.uint8), ((px >> 8) & 255).to(torch.uint8)),
        ((py & 255).to(torch.uint8), ((py >> 8) & 255).to(torch.uint8)),
    )


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)


def render_image(
    alu: ExactALU | NeuralALU,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, float]:
    total = Config.WIDTH * Config.HEIGHT
    output = np.empty((total, 3), dtype=np.uint8)
    started = time.perf_counter()
    last_report = 0
    for start in range(0, total, Config.TILE_PIXELS):
        end = min(total, start + Config.TILE_PIXELS)
        x, y = make_coordinates(start, end, device)
        tile = render_cloud_program(alu, x, y, Config.TIME_SECONDS)
        output[start:end] = tile.detach().cpu().numpy()
        if end == total or end - last_report >= max(Config.TILE_PIXELS, total // 10):
            elapsed = time.perf_counter() - started
            print(
                f"[{label}] pixels={end:,}/{total:,} | "
                f"lane_ops={alu.audit.lane_operations:,} | {end / max(elapsed, 1e-9):,.0f} px/s"
            )
            last_report = end
    elapsed = time.perf_counter() - started
    return output.reshape(Config.HEIGHT, Config.WIDTH, 3), elapsed


def main() -> None:
    verify_instruction_encoding()
    if Config.WIDTH <= 0 or Config.HEIGHT <= 0 or Config.TILE_PIXELS <= 0:
        raise ValueError("宽、高和 tile pixels 必须为正数。")
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    texture_path = locate_texture()
    texture_mips, texture_info = load_texture_mips(texture_path)
    print(
        f"结果目录：{Config.RESULT_DIR}\n"
        f"配置：{Config.WIDTH}x{Config.HEIGHT} | time={Config.TIME_SECONDS:.3f}s | "
        f"tile={Config.TILE_PIXELS:,} | interpolation={Config.INTERPOLATION_BITS}-bit | "
        f"trilinear_primary={Config.TRILINEAR_PRIMARY} | "
        f"trilinear_curl={Config.TRILINEAR_CURL} | "
        f"reference_only={Config.REFERENCE_ONLY}\n"
        f"纹理：{texture_path} | mip={texture_info['mip_sizes']}"
    )

    exact = ExactALU(device)
    exact.install_texture_mips(texture_mips)
    reference, reference_seconds = render_image(exact, device, "精确整数参考")
    reference_path = Config.RESULT_DIR / "fast_clouds_integer_reference.png"
    save_rgb(reference_path, reference)
    exact_ops = exact.audit.lane_operations
    print(
        f"精确参考完成：{reference_path} | lane_ops={exact_ops:,} | "
        f"{exact_ops / (Config.WIDTH * Config.HEIGHT):,.1f} op/pixel"
    )

    summary: dict[str, Any] = {
        "status": "reference_only" if Config.REFERENCE_ONLY else "running",
        "width": Config.WIDTH,
        "height": Config.HEIGHT,
        "time_seconds": Config.TIME_SECONDS,
        "interpolation_bits": Config.INTERPOLATION_BITS,
        "trilinear_primary": Config.TRILINEAR_PRIMARY,
        "trilinear_curl": Config.TRILINEAR_CURL,
        "texture": texture_info,
        "integer_lane_operations": exact_ops,
        "operations_per_pixel": exact_ops / (Config.WIDTH * Config.HEIGHT),
        "integer_render_seconds": reference_seconds,
        "reference_path": str(reference_path),
        "purity_contract": {
            "external": [
                "integer pixel coordinates",
                "time constant",
                "read-only texture RAM and mip storage",
                "SIMD scheduling",
                "framebuffer",
            ],
            "forbidden_and_unused": [
                "multiplication LUT",
                "trigonometric LUT",
                "externally computed shader intermediate",
                "low-resolution render followed by resize",
            ],
            "texture_contract": "address/filter math runs through ISA; RAM only returns bytes",
        },
    }
    summary_path = Config.RESULT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if exact_ops > Config.MAX_LANE_OPERATIONS and not Config.ALLOW_OVER_BUDGET:
        raise RuntimeError(
            f"预计 Neural lane-op={exact_ops:,}，超过预算 {Config.MAX_LANE_OPERATIONS:,}。"
            "可降低真实输出分辨率，或将 Config.ALLOW_OVER_BUDGET 改为 True。"
        )
    if Config.REFERENCE_ONLY:
        print(f"参考预览：{reference_path}\n汇总：{summary_path}")
        return

    model, model_info = load_model(device)
    print(
        f"Neural CPU：{model_info['path']} | step={model_info['step']} | "
        f"51 -> {model_info['hidden_size']} x {model_info['hidden_linear_layers']} -> 39"
    )
    neural = NeuralALU(model, device)
    neural.install_texture_mips(texture_mips)
    neural_image, neural_seconds = render_image(neural, device, "Neural CPU")
    audit = neural.finalize_audit()
    neural_path = Config.RESULT_DIR / "fast_clouds_neural.png"
    difference_path = Config.RESULT_DIR / "fast_clouds_difference.png"
    save_rgb(neural_path, neural_image)
    difference = np.abs(neural_image.astype(np.int16) - reference.astype(np.int16))
    save_rgb(difference_path, np.clip(difference * 4, 0, 255).astype(np.uint8))
    differing_pixels = int(np.count_nonzero(np.any(difference != 0, axis=2)))
    max_channel_difference = int(difference.max())
    summary.update(
        {
            "status": "ok" if audit.total_errors == 0 else "neural_errors",
            "model": model_info,
            "neural_audit": asdict(audit),
            "neural_render_seconds": neural_seconds,
            "neural_path": str(neural_path),
            "difference_path": str(difference_path),
            "differing_pixels": differing_pixels,
            "max_channel_difference": max_channel_difference,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Neural CPU Fast 2D Clouds 完成 ===")
    print(f"Neural 图：{neural_path}")
    print(f"精确参考：{reference_path}")
    print(f"差异图：{difference_path}")
    print(
        f"lane_ops={audit.lane_operations:,} | register_bit_errors={audit.register_bit_errors:,} | "
        f"flag_errors={audit.flag_bit_errors:,} | control_errors={audit.control_bit_errors:,}"
    )
    print(
        f"差异像素={differing_pixels:,}/{Config.WIDTH * Config.HEIGHT:,} | "
        f"max_channel_diff={max_channel_difference} | 耗时={neural_seconds:.1f}s"
    )
    print(f"汇总：{summary_path}")


if __name__ == "__main__":
    main()
