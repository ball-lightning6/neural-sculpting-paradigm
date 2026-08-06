"""
使用训练好的 Neural CPU v3 渲染一个逐步复原的 3x3 魔方。

实验边界：
- 默认 pure_flat 模式不使用光照、阴影、纹理或 OpenCV 多边形填充。
- 相机/魔方旋转、16-bit 坐标投影表读取，以及每个候选像素的四条边函数，
  均由 Neural CPU 通过 LOAD/MOVI、ADD/ADC、SUB/SBC 和 8-bit 乘法表执行。
- 外部固定功能只保留动作调度、帧缓冲存储和视频编码；画面按 Neural CPU
  实际完成覆盖判断的原生分辨率直接写入视频，不进行 resize；
  它们不决定面覆盖关系，也不计算颜色。
- 复原序列是预先打乱序列的严格逆序，因此每一步都是合法魔方动作。

所有运行参数都集中在文件顶部的 ``Config`` 中。将 ``REFERENCE_ONLY`` 改为
``True`` 可只检查画面而不加载 Neural CPU。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Neural CPU v3
# =============================================================================

INPUT_BITS = 51
STATE_BITS = 35
OUTPUT_BITS = 39
NUM_FLAGS = 3
NUM_REGISTERS = 4
REGISTER_BITS = 8
CTRL_MEMORY_READ = 0
CTRL_MEMORY_WRITE = 1
ROTATION_SCALE = 127.0

OPCODES = {
    "MOVI": 3,
    "LOAD": 4,
    "STORE": 5,
    "ADD": 6,
    "ADC": 7,
    "SUB": 8,
    "SBC": 9,
}


def script_directory() -> Path:
    source = globals().get("__file__")
    if not source:
        return Path.cwd().resolve()
    source_dir = Path(source).resolve().parent
    return source_dir.parent if source_dir.name == "scripts" else source_dir


class Config:
    BASE_DIR = script_directory()
    MODEL_PATH = BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    RESULT_DIR = BASE_DIR / "results" / "rubiks"

    # pure_flat 直接输出 Neural CPU 实际覆盖判定的原生帧缓冲，不再把低分辨率
    # 画面最近邻放大成“伪 1080p”。默认 640x360 约需 60--70 亿 lane-op。
    WIDTH = 640
    HEIGHT = 360
    FPS = 24
    MOVE_FRAMES = 12
    PAUSE_FRAMES = 3
    START_HOLD_FRAMES = 30
    END_HOLD_FRAMES = 48

    # pure_flat 是默认路径：Neural CPU 负责逐像素四边形覆盖判定，
    # 外部控制器只保存帧缓冲、放大画面并编码视频。
    RENDER_MODE = "pure_flat"
    LOGICAL_WIDTH = WIDTH
    LOGICAL_HEIGHT = HEIGHT
    PURE_BACKGROUND = (10, 13, 20)

    CAMERA_DISTANCE = 275.0
    VIEW_SCALE = 4.65
    CAMERA_YAW_DEGREES = -35.0
    CAMERA_PITCH_DEGREES = 24.0
    CAMERA_ORBIT_DEGREES = 16.0

    REFERENCE_ONLY = False
    STOP_ON_ERROR = False
    ALLOW_TF32 = False
    PRINT_EVERY = 12
    MAX_NEURAL_LANE_OPS = 10_000_000_000

    # 一组不太短、但画面节奏仍然清晰的合法打乱。
    SCRAMBLE = "R U F' L2 D B' R2 U' F L' D2 B U2 R' F2".split()


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


def encode_instruction(
    opcode: int,
    rd: int = 0,
    ra: int = 0,
    rb: int = 0,
    immediate: int | torch.Tensor = 0,
) -> int | torch.Tensor:
    prefix = (opcode << 11) | ((rd & 3) << 9) | ((ra & 3) << 7) | ((rb & 3) << 5)
    if isinstance(immediate, torch.Tensor):
        return prefix | immediate.to(torch.int64)
    return prefix | (int(immediate) & 0xFF)


def load_model(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 Neural CPU 模型：{Config.MODEL_PATH}")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device, weights_only=False)
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
    model = NeuralCPUCore(hidden_size, hidden_layers, dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {
        "path": str(Config.MODEL_PATH),
        "step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
        "hidden_size": hidden_size,
        "hidden_linear_layers": hidden_layers + 1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_signed_product_luts(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(256, device=device, dtype=torch.int16)
    signed = torch.where(values < 128, values, values - 256).to(torch.int32)
    products_u16 = torch.bitwise_and(signed[:, None] * signed[None, :], 0xFFFF)
    low = torch.bitwise_and(products_u16, 0xFF).to(torch.uint8).reshape(-1)
    high = torch.bitwise_and(torch.bitwise_right_shift(products_u16, 8), 0xFF)
    return low, high.to(torch.uint8).reshape(-1)


def build_flat_projection_luts(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if Config.LOGICAL_WIDTH > 65535 or Config.LOGICAL_HEIGHT > 65535:
        raise ValueError("pure_flat 的逻辑帧缓冲宽高不能超过 65535。")
    unsigned = np.arange(65536, dtype=np.int32)
    signed = np.where(unsigned < 32768, unsigned, unsigned - 65536)
    # pure_flat 不使用透视缩放；这里留出边缘，避免旋转时裁掉顶角。
    scale = Config.VIEW_SCALE * Config.LOGICAL_HEIGHT / 880.0 / ROTATION_SCALE
    x = np.clip(
        np.rint(Config.LOGICAL_WIDTH * 0.5 + signed * scale), 0, Config.LOGICAL_WIDTH - 1
    ).astype(np.uint16)
    y = np.clip(
        np.rint(Config.LOGICAL_HEIGHT * 0.48 - signed * scale), 0, Config.LOGICAL_HEIGHT - 1
    ).astype(np.uint16)
    x_low = (x & 0xFF).astype(np.uint8)
    x_high = (x >> 8).astype(np.uint8)
    y_low = (y & 0xFF).astype(np.uint8)
    y_high = (y >> 8).astype(np.uint8)
    return tuple(
        torch.from_numpy(values).to(device)
        for values in (x_low, x_high, y_low, y_high)
    )


@dataclass
class ErrorCounters:
    neural_calls: int = 0
    neural_lane_operations: int = 0
    result_errors: int = 0
    load_control_errors: int = 0
    store_control_errors: int = 0


class NeuralSIMDShader:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.product_low_lut, self.product_high_lut = build_signed_product_luts(device)
        (
            self.project_x_low_lut,
            self.project_x_high_lut,
            self.project_y_low_lut,
            self.project_y_high_lut,
        ) = build_flat_projection_luts(device)
        self.errors = ErrorCounters()
        self._bit_shifts_16 = torch.arange(15, -1, -1, device=device)
        self._bit_shifts_8 = torch.arange(7, -1, -1, device=device)

    def _instruction_bits(
        self, instruction: int | torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if isinstance(instruction, torch.Tensor):
            values = instruction.to(device=self.device, dtype=torch.int64).reshape(-1)
            if values.numel() == 1 and batch_size != 1:
                values = values.expand(batch_size)
        else:
            values = torch.full(
                (batch_size,), int(instruction), device=self.device, dtype=torch.int64
            )
        return torch.bitwise_and(
            torch.bitwise_right_shift(values[:, None], self._bit_shifts_16[None, :]),
            1,
        ).to(torch.float32)

    def _state_bits(
        self, registers: torch.Tensor, flags: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = registers.shape[0]
        if flags is None:
            flags = torch.zeros(
                (batch_size, NUM_FLAGS), device=self.device, dtype=torch.uint8
            )
        register_bits = torch.bitwise_and(
            torch.bitwise_right_shift(
                registers.to(torch.int64)[:, :, None],
                self._bit_shifts_8[None, None, :],
            ),
            1,
        ).reshape(batch_size, -1)
        return torch.cat((flags.to(torch.float32), register_bits.to(torch.float32)), dim=1)

    @torch.inference_mode()
    def execute(
        self,
        instruction: int | torch.Tensor,
        registers: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(registers.shape[0])
        inputs = torch.cat(
            (
                self._instruction_bits(instruction, batch_size),
                self._state_bits(registers, flags),
            ),
            dim=1,
        )
        bits = self.model(inputs) >= 0
        output_flags = bits[:, :NUM_FLAGS].to(torch.uint8)
        register_bits = bits[:, NUM_FLAGS:STATE_BITS].reshape(
            batch_size, NUM_REGISTERS, REGISTER_BITS
        )
        weights = (1 << self._bit_shifts_8).to(torch.int64)
        output_registers = torch.sum(
            register_bits.to(torch.int64) * weights[None, None, :], dim=2
        ).to(torch.uint8)
        controls = bits[:, STATE_BITS:OUTPUT_BITS]
        self.errors.neural_calls += 1
        self.errors.neural_lane_operations += batch_size
        return output_registers, output_flags, controls

    def _load_external_byte(
        self, left: torch.Tensor, right: torch.Tensor, lut: torch.Tensor
    ) -> torch.Tensor:
        left = left.to(device=self.device, dtype=torch.uint8).reshape(-1)
        right = right.to(device=self.device, dtype=torch.uint8).reshape(-1)
        registers = torch.zeros(
            (left.numel(), NUM_REGISTERS), device=self.device, dtype=torch.uint8
        )
        registers[:, 1] = left
        registers[:, 2] = right
        after_load, _, controls = self.execute(
            encode_instruction(OPCODES["LOAD"], rd=0, ra=1, rb=2), registers
        )
        read_errors = int(torch.count_nonzero(~controls[:, CTRL_MEMORY_READ]).item())
        self.errors.load_control_errors += read_errors
        address = (left.to(torch.int64) << 8) | right.to(torch.int64)
        loaded = lut[address]
        after_movi, _, _ = self.execute(
            encode_instruction(OPCODES["MOVI"], rd=0, immediate=loaded), after_load
        )
        result = after_movi[:, 0]
        value_errors = int(torch.count_nonzero(result != loaded).item())
        self.errors.result_errors += value_errors
        if Config.STOP_ON_ERROR and (read_errors or value_errors):
            raise RuntimeError(
                f"外部乘法表读取失败：LOAD={read_errors}, MOVI={value_errors}"
            )
        return result

    def signed_product16(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._load_external_byte(left, right, self.product_low_lut),
            self._load_external_byte(left, right, self.product_high_lut),
        )

    def add16(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_low, left_high = left
        right_low, right_high = right
        batch_size = left_low.numel()
        registers = torch.zeros(
            (batch_size, NUM_REGISTERS), device=self.device, dtype=torch.uint8
        )
        registers[:, 1] = left_low
        registers[:, 2] = right_low
        low_output, low_flags, _ = self.execute(
            encode_instruction(OPCODES["ADD"], rd=0, ra=1, rb=2), registers
        )
        registers[:, 1] = left_high
        registers[:, 2] = right_high
        high_output, _, _ = self.execute(
            encode_instruction(OPCODES["ADC"], rd=0, ra=1, rb=2),
            registers,
            flags=low_flags,
        )
        expected = (
            (left_low.to(torch.int32) | (left_high.to(torch.int32) << 8))
            + (right_low.to(torch.int32) | (right_high.to(torch.int32) << 8))
        ) & 0xFFFF
        actual = low_output[:, 0].to(torch.int32) | (
            high_output[:, 0].to(torch.int32) << 8
        )
        errors = int(torch.count_nonzero(actual != expected).item())
        self.errors.result_errors += errors
        if Config.STOP_ON_ERROR and errors:
            raise RuntimeError(f"ADD/ADC 出现 {errors} 个双字节结果错误")
        return low_output[:, 0], high_output[:, 0]

    def sub8(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """用 Neural CPU 的 SUB 计算一个批次的 8-bit 模减法。"""
        left = left.to(device=self.device, dtype=torch.uint8).reshape(-1)
        right = right.to(device=self.device, dtype=torch.uint8).reshape(-1)
        registers = torch.zeros(
            (left.numel(), NUM_REGISTERS), device=self.device, dtype=torch.uint8
        )
        registers[:, 1] = left
        registers[:, 2] = right
        output, _, _ = self.execute(
            encode_instruction(OPCODES["SUB"], rd=0, ra=1, rb=2), registers
        )
        result = output[:, 0]
        expected = (left.to(torch.int16) - right.to(torch.int16)).to(torch.uint8)
        errors = int(torch.count_nonzero(result != expected).item())
        self.errors.result_errors += errors
        if Config.STOP_ON_ERROR and errors:
            raise RuntimeError(f"SUB 出现 {errors} 个结果错误")
        return result

    def sub16(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """用 SUB/SBC 完成小端 16-bit 模减法。"""
        left_low, left_high = left
        right_low, right_high = right
        batch_size = left_low.numel()
        registers = torch.zeros(
            (batch_size, NUM_REGISTERS), device=self.device, dtype=torch.uint8
        )
        registers[:, 1] = left_low
        registers[:, 2] = right_low
        low_output, low_flags, _ = self.execute(
            encode_instruction(OPCODES["SUB"], rd=0, ra=1, rb=2), registers
        )
        registers[:, 1] = left_high
        registers[:, 2] = right_high
        high_output, _, _ = self.execute(
            encode_instruction(OPCODES["SBC"], rd=0, ra=1, rb=2),
            registers,
            flags=low_flags,
        )
        expected = (
            (left_low.to(torch.int32) | (left_high.to(torch.int32) << 8))
            - (right_low.to(torch.int32) | (right_high.to(torch.int32) << 8))
        ) & 0xFFFF
        actual = low_output[:, 0].to(torch.int32) | (
            high_output[:, 0].to(torch.int32) << 8
        )
        errors = int(torch.count_nonzero(actual != expected).item())
        self.errors.result_errors += errors
        if Config.STOP_ON_ERROR and errors:
            raise RuntimeError(f"SUB/SBC 出现 {errors} 个双字节结果错误")
        return low_output[:, 0], high_output[:, 0]

    def framebuffer_store(
        self,
        x: np.ndarray,
        y: np.ndarray,
        palette_index: np.ndarray,
    ) -> np.ndarray:
        """让 Neural CPU 发出一批 STORE；外部帧缓冲只接收有效写使能。"""
        count = int(x.size)
        if count == 0:
            return np.empty(0, dtype=bool)
        registers = torch.zeros(
            (count, NUM_REGISTERS), device=self.device, dtype=torch.uint8
        )
        registers[:, 0] = _as_u8_tensor(palette_index, self.device)
        registers[:, 1] = _as_u8_tensor(x, self.device)
        registers[:, 2] = _as_u8_tensor(y, self.device)
        output_registers, output_flags, controls = self.execute(
            encode_instruction(OPCODES["STORE"], rd=0, ra=1, rb=2), registers
        )
        write_enable = controls[:, CTRL_MEMORY_WRITE]
        control_errors = int(torch.count_nonzero(~write_enable).item())
        state_errors = int(
            torch.count_nonzero(torch.any(output_registers != registers, dim=1)).item()
            + torch.count_nonzero(torch.any(output_flags != 0, dim=1)).item()
        )
        self.errors.store_control_errors += control_errors
        self.errors.result_errors += state_errors
        if Config.STOP_ON_ERROR and (control_errors or state_errors):
            raise RuntimeError(
                f"STORE 出错：write_enable={control_errors}, state={state_errors}"
            )
        return write_enable.cpu().numpy()


def signed_to_u8(values: np.ndarray) -> np.ndarray:
    return np.bitwise_and(values.astype(np.int16), 0xFF).astype(np.uint8)


def combine_int16_bytes(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    unsigned = low.to(torch.int32) | (high.to(torch.int32) << 8)
    return torch.where(unsigned < 0x8000, unsigned, unsigned - 0x10000)


def quantize_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(matrix * ROTATION_SCALE), -127, 127).astype(np.int16)


def rotate_vectors_neural(
    shader: NeuralSIMDShader,
    vectors: np.ndarray,
    matrix_q7: np.ndarray,
) -> np.ndarray:
    if vectors.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    device = shader.device
    values = [
        torch.from_numpy(signed_to_u8(vectors[:, axis])).to(device)
        for axis in range(3)
    ]
    coefficients = [
        torch.full(
            (vectors.shape[0],),
            int(matrix_q7[row, col]) & 0xFF,
            device=device,
            dtype=torch.uint8,
        )
        for row in range(3)
        for col in range(3)
    ]
    outputs: list[torch.Tensor] = []
    for row in range(3):
        accumulator: tuple[torch.Tensor, torch.Tensor] | None = None
        for col in range(3):
            product = shader.signed_product16(values[col], coefficients[row * 3 + col])
            accumulator = product if accumulator is None else shader.add16(
                accumulator, product
            )
        assert accumulator is not None
        outputs.append(combine_int16_bytes(*accumulator))
    return torch.stack(outputs, dim=1).cpu().numpy().astype(np.int32)


def rotate_vectors_exact(vectors: np.ndarray, matrix_q7: np.ndarray) -> np.ndarray:
    return vectors.astype(np.int32) @ matrix_q7.astype(np.int32).T


# =============================================================================
# 魔方状态与几何
# =============================================================================

FACE_AXES = (
    (0, -1, "L"),
    (0, 1, "R"),
    (1, -1, "D"),
    (1, 1, "U"),
    (2, -1, "B"),
    (2, 1, "F"),
)

STICKER_COLORS = {
    "U": np.asarray([242, 242, 235], dtype=np.uint8),
    "D": np.asarray([247, 205, 36], dtype=np.uint8),
    "R": np.asarray([205, 42, 39], dtype=np.uint8),
    "L": np.asarray([242, 115, 28], dtype=np.uint8),
    "F": np.asarray([35, 158, 79], dtype=np.uint8),
    "B": np.asarray([39, 92, 190], dtype=np.uint8),
}

PURE_PALETTE = np.asarray(
    [
        Config.PURE_BACKGROUND,
        [28, 30, 35],
        STICKER_COLORS["U"],
        STICKER_COLORS["D"],
        STICKER_COLORS["R"],
        STICKER_COLORS["L"],
        STICKER_COLORS["F"],
        STICKER_COLORS["B"],
    ],
    dtype=np.uint8,
)
PURE_PALETTE_LOOKUP = {
    tuple(int(channel) for channel in color): index
    for index, color in enumerate(PURE_PALETTE)
}

MOVE_SPECS = {
    "R": (0, 1, -1),
    "L": (0, -1, 1),
    "U": (1, 1, -1),
    "D": (1, -1, 1),
    "F": (2, 1, -1),
    "B": (2, -1, 1),
}


def axis_rotation(axis: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def parse_move(move: str) -> tuple[int, int, int]:
    base = move[0]
    axis, layer, base_turn = MOVE_SPECS[base]
    if move.endswith("2"):
        turns = 2
    elif move.endswith("'"):
        turns = -base_turn
    else:
        turns = base_turn
    return axis, layer, turns


def inverse_move(move: str) -> str:
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[0]
    return move + "'"


def discrete_rotation(axis: int, turns: int) -> np.ndarray:
    return np.rint(axis_rotation(axis, turns * math.pi / 2)).astype(np.int8)


@dataclass
class RubiksState:
    positions: np.ndarray
    orientations: np.ndarray

    @classmethod
    def solved(cls) -> "RubiksState":
        positions = np.asarray(
            [
                (x, y, z)
                for x in (-1, 0, 1)
                for y in (-1, 0, 1)
                for z in (-1, 0, 1)
                if (x, y, z) != (0, 0, 0)
            ],
            dtype=np.int8,
        )
        orientations = np.repeat(np.eye(3, dtype=np.int8)[None, :, :], 26, axis=0)
        return cls(positions, orientations)

    def apply(self, move: str) -> None:
        axis, layer, turns = parse_move(move)
        rotation = discrete_rotation(axis, turns)
        selected = self.positions[:, axis] == layer
        self.positions[selected] = self.positions[selected] @ rotation.T
        self.orientations[selected] = rotation[None, :, :] @ self.orientations[selected]


@dataclass
class RubiksGeometry:
    face_cubies: np.ndarray
    local_vertices: np.ndarray
    local_normals: np.ndarray
    base_colors: np.ndarray
    sticker_mask: np.ndarray


def face_vertices(axis: int, sign: int, half: int, tangent: int | None = None) -> np.ndarray:
    t = half if tangent is None else tangent
    free = [index for index in range(3) if index != axis]
    vertices = np.zeros((4, 3), dtype=np.int16)
    vertices[:, axis] = sign * half
    corners = [(-t, -t), (t, -t), (t, t), (-t, t)]
    if sign < 0:
        corners = list(reversed(corners))
    for index, (u, v) in enumerate(corners):
        vertices[index, free[0]] = u
        vertices[index, free[1]] = v
    return vertices


def build_geometry() -> RubiksGeometry:
    face_cubies: list[int] = []
    local_vertices: list[np.ndarray] = []
    local_normals: list[np.ndarray] = []
    base_colors: list[np.ndarray] = []
    sticker_mask: list[bool] = []
    solved_positions = RubiksState.solved().positions

    for cubie, position in enumerate(solved_positions):
        # 小块间只留 2 个坐标单位的真实缝隙。主体六面必须完整封闭；
        # 不能只把平面内缩来伪装倒角，否则边棱处会真的漏出背景。
        for axis, sign, face_name in FACE_AXES:
            normal = np.zeros(3, dtype=np.int16)
            normal[axis] = sign * 127
            face_cubies.append(cubie)
            local_vertices.append(face_vertices(axis, sign, half=19))
            local_normals.append(normal)
            base_colors.append(np.asarray([28, 30, 35], dtype=np.uint8))
            sticker_mask.append(False)

            if int(position[axis]) == sign:
                face_cubies.append(cubie)
                local_vertices.append(face_vertices(axis, sign, half=20, tangent=15))
                local_normals.append(normal)
                base_colors.append(STICKER_COLORS[face_name])
                sticker_mask.append(True)

    return RubiksGeometry(
        face_cubies=np.asarray(face_cubies, dtype=np.int16),
        local_vertices=np.asarray(local_vertices, dtype=np.int16),
        local_normals=np.asarray(local_normals, dtype=np.int16),
        base_colors=np.asarray(base_colors, dtype=np.uint8),
        sticker_mask=np.asarray(sticker_mask, dtype=bool),
    )


def object_space_faces(
    state: RubiksState, geometry: RubiksGeometry
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.empty_like(geometry.local_vertices)
    normals = np.empty_like(geometry.local_normals)
    spacing = 40
    for face_index, cubie in enumerate(geometry.face_cubies):
        orientation = state.orientations[cubie].astype(np.int16)
        center = state.positions[cubie].astype(np.int16) * spacing
        vertices[face_index] = geometry.local_vertices[face_index] @ orientation.T + center
        normals[face_index] = geometry.local_normals[face_index] @ orientation.T
    return vertices, normals


def camera_matrix(yaw: float, pitch: float) -> np.ndarray:
    return axis_rotation(0, pitch) @ axis_rotation(1, yaw)


def transform_frame_geometry(
    state: RubiksState,
    geometry: RubiksGeometry,
    camera: np.ndarray,
    active_move: str | None,
    progress: float,
    shader: NeuralSIMDShader | None,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, normals = object_space_faces(state, geometry)
    transformed_vertices = np.empty((vertices.shape[0] * 4, 3), dtype=np.int32)
    transformed_normals = np.empty((normals.shape[0], 3), dtype=np.int32)

    selected_cubies = np.zeros(26, dtype=bool)
    active_matrix = camera
    if active_move is not None:
        axis, layer, turns = parse_move(active_move)
        selected_cubies = state.positions[:, axis] == layer
        partial = axis_rotation(axis, turns * math.pi / 2 * progress)
        active_matrix = camera @ partial

    face_selected = selected_cubies[geometry.face_cubies]
    for selected, matrix in ((False, camera), (True, active_matrix)):
        face_indices = np.flatnonzero(face_selected == selected)
        if face_indices.size == 0:
            continue
        vectors = np.concatenate(
            (vertices[face_indices].reshape(-1, 3), normals[face_indices]), axis=0
        )
        matrix_q7 = quantize_matrix(matrix)
        if shader is None:
            rotated = rotate_vectors_exact(vectors, matrix_q7)
        else:
            rotated = rotate_vectors_neural(shader, vectors, matrix_q7)
        vertex_count = face_indices.size * 4
        transformed_vertices.reshape(-1, 4, 3)[face_indices] = rotated[:vertex_count].reshape(
            -1, 4, 3
        )
        transformed_normals[face_indices] = rotated[vertex_count:]

    return transformed_vertices.reshape(-1, 4, 3), transformed_normals


# =============================================================================
# 光照与光栅化
# =============================================================================


def make_background() -> np.ndarray:
    height, width = Config.HEIGHT, Config.WIDTH
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top = np.asarray([31, 38, 52], dtype=np.float32)[None, None, :]
    bottom = np.asarray([8, 11, 17], dtype=np.float32)[None, None, :]
    rgb = top * (1.0 - y) + bottom * y
    rgb = np.repeat(rgb, width, axis=1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def add_floor_and_shadow(frame: np.ndarray) -> np.ndarray:
    import cv2

    result = frame.copy()
    horizon = int(Config.HEIGHT * 0.78)
    overlay = result.copy()
    cv2.rectangle(
        overlay,
        (0, horizon),
        (Config.WIDTH, Config.HEIGHT),
        (13, 16, 22),
        thickness=-1,
    )
    result = cv2.addWeighted(result, 0.36, overlay, 0.64, 0)
    shadow = np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8)
    cv2.ellipse(
        shadow,
        (Config.WIDTH // 2, int(Config.HEIGHT * 0.79)),
        (int(Config.HEIGHT * 0.29), int(Config.HEIGHT * 0.055)),
        0,
        0,
        360,
        190,
        thickness=-1,
    )
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=28, sigmaY=12)
    alpha = (shadow.astype(np.float32) / 255.0 * 0.62)[..., None]
    return np.clip(result.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def shade_faces(
    geometry: RubiksGeometry, transformed_normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normals = transformed_normals.astype(np.float32) / ROTATION_SCALE
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    key = np.asarray([-0.45, 0.76, 0.66], dtype=np.float32)
    key /= np.linalg.norm(key)
    fill = np.asarray([0.72, 0.12, 0.40], dtype=np.float32)
    fill /= np.linalg.norm(fill)
    diffuse = np.clip(normals @ key, 0.0, 1.0)
    fill_diffuse = np.clip(normals @ fill, 0.0, 1.0)
    sky = np.clip(0.5 + 0.5 * normals[:, 1], 0.0, 1.0)
    half_vector = key + view
    half_vector /= np.linalg.norm(half_vector)
    specular = np.power(np.clip(normals @ half_vector, 0.0, 1.0), 38.0)

    light = 0.24 + 0.76 * diffuse + 0.20 * fill_diffuse + 0.13 * sky
    colors = geometry.base_colors.astype(np.float32) * light[:, None]
    sticker_spec = np.where(geometry.sticker_mask, 46.0, 68.0)
    colors += specular[:, None] * sticker_spec[:, None]
    return np.clip(colors, 0, 255).astype(np.uint8), normals[:, 2]


def project_vertices(transformed_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = transformed_vertices.astype(np.float32) / ROTATION_SCALE
    depth = coordinates[:, :, 2]
    perspective = Config.CAMERA_DISTANCE / np.maximum(
        Config.CAMERA_DISTANCE - depth, 80.0
    )
    pixel_scale = Config.VIEW_SCALE * Config.HEIGHT / 720.0
    x = Config.WIDTH * 0.5 + coordinates[:, :, 0] * pixel_scale * perspective
    y = Config.HEIGHT * 0.47 - coordinates[:, :, 1] * pixel_scale * perspective
    return np.stack((x, y), axis=2), depth


def rasterize(
    projected: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    normal_z: np.ndarray,
    geometry: RubiksGeometry,
    background: np.ndarray,
) -> tuple[np.ndarray, int]:
    import cv2

    frame = add_floor_and_shadow(background)
    area = (
        (projected[:, 1, 0] - projected[:, 0, 0])
        * (projected[:, 2, 1] - projected[:, 0, 1])
        - (projected[:, 1, 1] - projected[:, 0, 1])
        * (projected[:, 2, 0] - projected[:, 0, 0])
    )
    visible = (normal_z > 0.006) & (np.abs(area) > 0.25)
    indices = np.flatnonzero(visible)
    depth_order = indices[np.argsort(depth[indices].mean(axis=1), kind="stable")]
    # 平均深度 painter 对几乎共面的塑料主体与贴纸会偶尔排反，形成孤立黑点。
    # 先画全部塑料，再按深度画全部外露贴纸，保证贴纸覆盖自己的承载面。
    order = np.concatenate(
        (
            depth_order[~geometry.sticker_mask[depth_order]],
            depth_order[geometry.sticker_mask[depth_order]],
        )
    )
    polygons = np.rint(projected).astype(np.int32)
    for face_index in order:
        polygon = np.ascontiguousarray(polygons[face_index])
        color = tuple(int(value) for value in colors[face_index])
        cv2.fillConvexPoly(frame, polygon, color, lineType=cv2.LINE_AA)
        if geometry.sticker_mask[face_index]:
            edge = tuple(max(0, int(value * 0.58)) for value in color)
            cv2.polylines(frame, [polygon], True, edge, 1, lineType=cv2.LINE_AA)
    return frame, int(order.size)


def project_vertices_flat(
    transformed_vertices: np.ndarray,
    shader: NeuralSIMDShader | None,
) -> tuple[np.ndarray, np.ndarray]:
    """由 Neural CPU 读取低/高字节投影表，得到 16-bit 帧缓冲坐标。"""
    flat = transformed_vertices.reshape(-1, 3).astype(np.int32)
    unsigned_x = np.bitwise_and(flat[:, 0], 0xFFFF)
    unsigned_y = np.bitwise_and(flat[:, 1], 0xFFFF)
    high_x = (unsigned_x >> 8).astype(np.uint8)
    low_x = (unsigned_x & 0xFF).astype(np.uint8)
    high_y = (unsigned_y >> 8).astype(np.uint8)
    low_y = (unsigned_y & 0xFF).astype(np.uint8)

    if shader is None:
        x_low_lut, x_high_lut, y_low_lut, y_high_lut = build_flat_projection_luts(
            torch.device("cpu")
        )
        address_x = (high_x.astype(np.int32) << 8) | low_x.astype(np.int32)
        address_y = (high_y.astype(np.int32) << 8) | low_y.astype(np.int32)
        x = (
            x_low_lut.numpy()[address_x].astype(np.int32)
            | (x_high_lut.numpy()[address_x].astype(np.int32) << 8)
        )
        y = (
            y_low_lut.numpy()[address_y].astype(np.int32)
            | (y_high_lut.numpy()[address_y].astype(np.int32) << 8)
        )
    else:
        x_low = shader._load_external_byte(
            torch.from_numpy(high_x).to(shader.device),
            torch.from_numpy(low_x).to(shader.device),
            shader.project_x_low_lut,
        )
        x_high = shader._load_external_byte(
            torch.from_numpy(high_x).to(shader.device),
            torch.from_numpy(low_x).to(shader.device),
            shader.project_x_high_lut,
        )
        y_low = shader._load_external_byte(
            torch.from_numpy(high_y).to(shader.device),
            torch.from_numpy(low_y).to(shader.device),
            shader.project_y_low_lut,
        )
        y_high = shader._load_external_byte(
            torch.from_numpy(high_y).to(shader.device),
            torch.from_numpy(low_y).to(shader.device),
            shader.project_y_high_lut,
        )
        x = (x_low.to(torch.int32) | (x_high.to(torch.int32) << 8)).cpu().numpy()
        y = (y_low.to(torch.int32) | (y_high.to(torch.int32) << 8)).cpu().numpy()
    projected = np.stack((x, y), axis=1).reshape(-1, 4, 2).astype(np.float32)
    return projected, transformed_vertices[:, :, 2].astype(np.float32)


def _as_u8_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    encoded = np.bitwise_and(values.astype(np.int64), 0xFF).astype(np.uint8)
    return torch.from_numpy(encoded).to(device=device)


def _edge_cross_neural(
    shader: NeuralSIMDShader,
    px: np.ndarray,
    py: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """由 Neural CPU 计算 cross(edge, pixel - vertex)，返回 signed int16。"""
    device = shader.device
    px_u8 = _as_u8_tensor(px, device)
    py_u8 = _as_u8_tensor(py, device)
    x0_u8 = _as_u8_tensor(x0, device)
    y0_u8 = _as_u8_tensor(y0, device)
    x1_u8 = _as_u8_tensor(x1, device)
    y1_u8 = _as_u8_tensor(y1, device)

    edge_x = shader.sub8(x1_u8, x0_u8)
    edge_y = shader.sub8(y1_u8, y0_u8)
    delta_x = shader.sub8(px_u8, x0_u8)
    delta_y = shader.sub8(py_u8, y0_u8)
    left = shader.signed_product16(edge_x, delta_y)
    right = shader.signed_product16(edge_y, delta_x)
    return combine_int16_bytes(*shader.sub16(left, right)).cpu().numpy()


def rasterize_neural_flat(
    projected: np.ndarray,
    depth: np.ndarray,
    normal_z: np.ndarray,
    geometry: RubiksGeometry,
    shader: NeuralSIMDShader | None,
) -> tuple[np.ndarray, int, int]:
    """在逻辑帧缓冲中用 Neural CPU 做逐像素凸四边形覆盖判定。"""
    height, width = Config.LOGICAL_HEIGHT, Config.LOGICAL_WIDTH
    proposed_palette = np.zeros((height, width), dtype=np.uint8)

    area = (
        (projected[:, 1, 0] - projected[:, 0, 0])
        * (projected[:, 2, 1] - projected[:, 0, 1])
        - (projected[:, 1, 1] - projected[:, 0, 1])
        * (projected[:, 2, 0] - projected[:, 0, 0])
    )
    visible = (normal_z > 0.006) & (np.abs(area) > 0.08)
    indices = np.flatnonzero(visible)
    depth_order = indices[np.argsort(depth[indices].mean(axis=1), kind="stable")]
    order = np.concatenate(
        (
            depth_order[~geometry.sticker_mask[depth_order]],
            depth_order[geometry.sticker_mask[depth_order]],
        )
    )
    polygons = np.rint(projected).astype(np.int32)

    records: list[tuple[int, int, int, int, int, int, int]] = []
    face_ids: list[np.ndarray] = []
    pixel_x: list[np.ndarray] = []
    pixel_y: list[np.ndarray] = []
    offset = 0
    for face_index in order:
        polygon = polygons[face_index].copy()
        signed_area = int(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        if signed_area < 0:
            polygon = polygon[::-1]
            polygons[face_index] = polygon
        x_min = max(0, int(polygon[:, 0].min()))
        x_max = min(width - 1, int(polygon[:, 0].max()))
        y_min = max(0, int(polygon[:, 1].min()))
        y_max = min(height - 1, int(polygon[:, 1].max()))
        if x_min > x_max or y_min > y_max:
            continue
        box_width = x_max - x_min + 1
        box_height = y_max - y_min + 1
        if box_width > 127 or box_height > 127:
            raise RuntimeError(
                "单个面的逻辑包围盒超过 signed 8-bit 范围；请减小 "
                "Config.LOGICAL_WIDTH/HEIGHT 或 Config.VIEW_SCALE。"
            )
        yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
        count = int(xx.size)
        records.append((face_index, x_min, x_max, y_min, y_max, offset, count))
        face_ids.append(np.full(count, face_index, dtype=np.int32))
        pixel_x.append(xx.reshape(-1).astype(np.int32))
        pixel_y.append(yy.reshape(-1).astype(np.int32))
        offset += count

    if not records:
        return PURE_PALETTE[proposed_palette], 0, 0

    all_face_ids = np.concatenate(face_ids)
    all_x = np.concatenate(pixel_x)
    all_y = np.concatenate(pixel_y)
    inside = np.ones(all_x.size, dtype=bool)
    for edge in range(4):
        next_edge = (edge + 1) % 4
        x0 = polygons[all_face_ids, edge, 0]
        y0 = polygons[all_face_ids, edge, 1]
        x1 = polygons[all_face_ids, next_edge, 0]
        y1 = polygons[all_face_ids, next_edge, 1]
        if shader is None:
            cross = (x1 - x0) * (all_y - y0) - (y1 - y0) * (all_x - x0)
        else:
            cross = _edge_cross_neural(shader, all_x, all_y, x0, y0, x1, y1)
        inside &= cross >= 0

    for face_index, x_min, x_max, y_min, y_max, start, count in records:
        local_mask = inside[start : start + count].reshape(
            y_max - y_min + 1, x_max - x_min + 1
        )
        region = proposed_palette[y_min : y_max + 1, x_min : x_max + 1]
        color_key = tuple(int(channel) for channel in geometry.base_colors[face_index])
        region[local_mask] = PURE_PALETTE_LOOKUP[color_key]

    if shader is None:
        committed_palette = proposed_palette
    else:
        write_y, write_x = np.nonzero(proposed_palette)
        values = proposed_palette[write_y, write_x]
        write_enable = shader.framebuffer_store(write_x, write_y, values)
        committed_palette = np.zeros_like(proposed_palette)
        committed_palette[write_y[write_enable], write_x[write_enable]] = values[write_enable]
    return PURE_PALETTE[committed_palette], len(records), int(all_x.size)


def upscale_logical_frame(frame: np.ndarray) -> np.ndarray:
    """pure_flat 不做任何伪分辨率放大，视频尺寸就是实际覆盖分辨率。"""
    if (Config.WIDTH, Config.HEIGHT) != (
        Config.LOGICAL_WIDTH,
        Config.LOGICAL_HEIGHT,
    ):
        raise ValueError(
            "pure_flat 禁止 resize：请令 Config.WIDTH/HEIGHT 与 "
            "Config.LOGICAL_WIDTH/HEIGHT 完全相同。"
        )
    return frame


def draw_overlay(
    frame: np.ndarray,
    current_step: int,
    total_steps: int,
    move: str | None,
    phase: str,
    solution: list[str],
) -> np.ndarray:
    import cv2

    result = frame.copy()
    cv2.putText(
        result,
        "NEURAL CPU / RUBIK'S CUBE",
        (44, 58),
        cv2.FONT_HERSHEY_DUPLEX,
        0.86,
        (235, 239, 247),
        1,
        cv2.LINE_AA,
    )
    if phase == "scrambled":
        status = "SCRAMBLED"
    elif phase == "solved":
        status = "SOLVED"
    else:
        status = f"SOLVING  {current_step:02d}/{total_steps:02d}    MOVE {move}"
    cv2.putText(
        result,
        status,
        (45, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (126, 198, 255) if phase != "solved" else (113, 231, 151),
        2,
        cv2.LINE_AA,
    )

    bar_x, bar_y, bar_w, bar_h = 45, Config.HEIGHT - 52, 390, 7
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 71, 84), -1)
    ratio = current_step / max(total_steps, 1)
    cv2.rectangle(
        result,
        (bar_x, bar_y),
        (bar_x + int(bar_w * ratio), bar_y + bar_h),
        (72, 169, 238),
        -1,
    )
    sequence = "  ".join(solution)
    cv2.putText(
        result,
        sequence,
        (45, Config.HEIGHT - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (160, 168, 184),
        1,
        cv2.LINE_AA,
    )
    return result


class VideoWriter:
    def __init__(self, path: Path):
        import cv2

        self.path = path
        self.ffmpeg = shutil.which("ffmpeg")
        self.temporary_path = (
            path.with_name(f"{path.stem}.mp4v-temporary.mp4")
            if self.ffmpeg
            else path
        )
        self.temporary_path.unlink(missing_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            Config.FPS,
            (Config.WIDTH, Config.HEIGHT),
        )
        if not self.writer.isOpened():
            raise RuntimeError("OpenCV mp4v 编码器不可用")

    def append(self, rgb: np.ndarray) -> None:
        self.writer.write(rgb[..., ::-1])

    def close(self) -> None:
        self.writer.release()
        if not self.ffmpeg:
            print("警告：未找到 ffmpeg，视频保留为兼容性较弱的 MPEG-4 Part 2。")
            return

        command = [
            self.ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(self.temporary_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.temporary_path.unlink(missing_ok=True)
            print("视频已转码为网页兼容的 H.264/yuv420p MP4。")
        except (OSError, subprocess.CalledProcessError) as error:
            # 转码失败时仍保留原始产物，避免整段渲染结果丢失。
            self.temporary_path.replace(self.path)
            details = getattr(error, "stderr", "") or str(error)
            print(f"警告：H.264 转码失败，已保留 MPEG-4 Part 2 视频：{details}")


def save_png(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        import cv2

        cv2.imwrite(str(path), rgb[..., ::-1])


# =============================================================================
# 主流程
# =============================================================================


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solution = [inverse_move(move) for move in reversed(Config.SCRAMBLE)]
    total_frames = (
        Config.START_HOLD_FRAMES
        + len(solution) * (Config.MOVE_FRAMES + Config.PAUSE_FRAMES)
        + Config.END_HOLD_FRAMES
    )
    print("=== Neural CPU 3x3 魔方复原 ===")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(f"结果目录：{Config.RESULT_DIR}")
    print(f"分辨率：{Config.WIDTH}x{Config.HEIGHT} | frames={total_frames} | fps={Config.FPS}")
    print("打乱：" + " ".join(Config.SCRAMBLE))
    print("复原：" + " ".join(solution))

    print(
        f"渲染模式：{Config.RENDER_MODE} | "
        f"逻辑帧缓冲={Config.LOGICAL_WIDTH}x{Config.LOGICAL_HEIGHT}"
    )

    geometry = build_geometry()
    state = RubiksState.solved()
    for move in Config.SCRAMBLE:
        state.apply(move)

    shader: NeuralSIMDShader | None = None
    model_info: dict[str, Any] = {"mode": "exact_reference"}
    if not Config.REFERENCE_ONLY:
        model, model_info = load_model(device)
        shader = NeuralSIMDShader(model, device)
        print(
            f"Neural CPU：{model_info['path']} | step={model_info['step']} | "
            f"51 -> {model_info['hidden_size']} x {model_info['hidden_linear_layers']} -> 39"
        )

    background = make_background() if Config.RENDER_MODE != "pure_flat" else None
    video_name = (
        "neural_cpu_rubiks_cube_flat.mp4"
        if Config.RENDER_MODE == "pure_flat"
        else "neural_gpu_rubiks_cube_solving.mp4"
    )
    video_path = Config.RESULT_DIR / video_name
    writer = VideoWriter(video_path)
    frame_index = 0
    started = time.perf_counter()
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None

    def render_one(
        active_move: str | None,
        progress: float,
        current_step: int,
        phase: str,
    ) -> None:
        nonlocal frame_index, first_frame, last_frame
        orbit_phase = frame_index / max(total_frames - 1, 1)
        yaw = math.radians(
            Config.CAMERA_YAW_DEGREES
            + Config.CAMERA_ORBIT_DEGREES * math.sin(orbit_phase * 2.0 * math.pi)
        )
        pitch = math.radians(
            Config.CAMERA_PITCH_DEGREES
            + 2.5 * math.sin(orbit_phase * 2.0 * math.pi + 0.8)
        )
        camera = camera_matrix(yaw, pitch)
        transformed_vertices, transformed_normals = transform_frame_geometry(
            state, geometry, camera, active_move, progress, shader
        )
        candidate_pixels = 0
        if Config.RENDER_MODE == "pure_flat":
            normal_z = transformed_normals[:, 2].astype(np.float32) / ROTATION_SCALE
            projected, depth = project_vertices_flat(transformed_vertices, shader)
            logical_frame, visible_faces, candidate_pixels = rasterize_neural_flat(
                projected, depth, normal_z, geometry, shader
            )
            frame = upscale_logical_frame(logical_frame)
        else:
            assert background is not None
            colors, normal_z = shade_faces(geometry, transformed_normals)
            projected, depth = project_vertices(transformed_vertices)
            frame, visible_faces = rasterize(
                projected, depth, colors, normal_z, geometry, background
            )
            frame = draw_overlay(
                frame,
                current_step,
                len(solution),
                active_move,
                phase,
                solution,
            )
        writer.append(frame)
        if first_frame is None:
            first_frame = frame.copy()
        last_frame = frame.copy()
        frame_index += 1
        if frame_index == 1 or frame_index % Config.PRINT_EVERY == 0:
            elapsed = time.perf_counter() - started
            lane_ops = shader.errors.neural_lane_operations if shader else 0
            result_errors = shader.errors.result_errors if shader else 0
            load_errors = shader.errors.load_control_errors if shader else 0
            store_errors = shader.errors.store_control_errors if shader else 0
            print(
                f"frame={frame_index:4d}/{total_frames} | render_fps={frame_index / elapsed:7.3f} | "
                f"visible_faces={visible_faces:3d} | candidates={candidate_pixels:,} | "
                f"lane_ops={lane_ops:,} | "
                f"result_errors={result_errors} | load_errors={load_errors} | "
                f"store_errors={store_errors}"
            )
            if lane_ops > Config.MAX_NEURAL_LANE_OPS:
                raise RuntimeError("Neural lane-op 超过配置上限")

    try:
        for _ in range(Config.START_HOLD_FRAMES):
            render_one(None, 0.0, 0, "scrambled")

        for move_index, move in enumerate(solution):
            for local_frame in range(Config.MOVE_FRAMES):
                progress = smoothstep((local_frame + 1) / Config.MOVE_FRAMES)
                render_one(move, progress, move_index + 1, "solving")
            state.apply(move)
            for _ in range(Config.PAUSE_FRAMES):
                render_one(None, 0.0, move_index + 1, "solving")

        for _ in range(Config.END_HOLD_FRAMES):
            render_one(None, 0.0, len(solution), "solved")
    finally:
        writer.close()

    elapsed = time.perf_counter() - started
    if first_frame is not None:
        save_png(Config.RESULT_DIR / "rubiks_scrambled.png", first_frame)
    if last_frame is not None:
        save_png(Config.RESULT_DIR / "rubiks_solved.png", last_frame)

    errors = shader.errors if shader else ErrorCounters()
    summary = {
        "status": "ok",
        "video": str(video_path),
        "width": Config.WIDTH,
        "height": Config.HEIGHT,
        "fps": Config.FPS,
        "frames": total_frames,
        "duration_seconds": total_frames / Config.FPS,
        "elapsed_seconds": elapsed,
        "render_fps": total_frames / elapsed,
        "render_mode": Config.RENDER_MODE,
        "logical_width": Config.LOGICAL_WIDTH,
        "logical_height": Config.LOGICAL_HEIGHT,
        "lighting": Config.RENDER_MODE != "pure_flat",
        "opencv_polygon_rasterizer": Config.RENDER_MODE != "pure_flat",
        "external_fixed_function": [
            "rubiks_move_scheduler",
            "framebuffer_storage",
            "mp4_encoder",
        ],
        "neural_render_stages": [
            "camera_and_layer_rotation",
            "projection_lut_loads",
            "per_pixel_quad_edge_tests",
            "framebuffer_store_enable",
        ] if Config.RENDER_MODE == "pure_flat" else ["camera_and_layer_rotation"],
        "scramble": Config.SCRAMBLE,
        "solution": solution,
        "neural_lane_operations": errors.neural_lane_operations,
        "neural_calls": errors.neural_calls,
        "result_errors": errors.result_errors,
        "load_control_errors": errors.load_control_errors,
        "store_control_errors": errors.store_control_errors,
        "model": model_info,
    }
    with (Config.RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n=== 魔方复原视频完成 ===")
    print(f"视频：{video_path}")
    print(f"时长：{total_frames / Config.FPS:.2f}s | 离线渲染速度：{total_frames / elapsed:.3f} FPS")
    print(
        f"Neural lane-op={errors.neural_lane_operations:,} | "
        f"result_errors={errors.result_errors} | load_errors={errors.load_control_errors} | "
        f"store_errors={errors.store_control_errors}"
    )


if __name__ == "__main__":
    main()

# %%
