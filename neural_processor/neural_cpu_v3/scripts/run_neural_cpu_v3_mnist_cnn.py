"""用 Neural CPU v3 执行量化 MNIST CNN。

这个脚本包含三个彼此独立的检查层：

1. QAT 浮点前向：复现训练时的 fake-quantized 网络。
2. 精确整数前向：验证导出的定点 CNN 与 QAT 网络一致。
3. Neural CPU 前向：把常量权重乘法编译成 CSD/NAF 移位加减，并把
   32-bit 累加、缩放、ReLU、max-pooling 和最终分类全部拆成 8-bit ISA
   操作，由训练得到的 Neural CPU 权重执行每一次数值操作。

外部控制器只负责与普通计算机相同的取指、张量寻址、lane 重排、内存写回
和分支调度；卷积与全连接中的乘法、加法、量化、选择和分类均不由 PyTorch
算术核替代。
脚本会用精确整数实现审计每一个 Neural CPU lane-op，默认遇到首个错误即停止。

所有运行参数都集中在文件顶部的 ``Config`` 中。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 配置
# =============================================================================


def script_directory() -> Path:
    source = globals().get("__file__")
    if not source:
        return Path.cwd().resolve()
    source_dir = Path(source).resolve().parent
    return source_dir.parent if source_dir.name == "scripts" else source_dir


def _first_existing(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到{description}，尝试过：\n"
        + "\n".join(f"  {candidate}" for candidate in candidates)
    )


class Config:
    BASE_DIR = script_directory()
    CNN_MODEL = BASE_DIR / "weights" / "mnist_quantized_cnn_best_model.pt"
    CPU_MODEL = BASE_DIR / "weights" / "neural_cpu_v3_best_balanced_model.pt"
    DATA_DIR = BASE_DIR / "assets" / "mnist_dataset"
    MNIST_MIRRORS = (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    )
    MNIST_TEST_FILES = {
        "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    }
    DOWNLOAD_TIMEOUT_SECONDS = 60
    RESULT_DIR = BASE_DIR / "results" / "mnist"
    IMAGE_INDEX = 0
    RUN_INTEGER_TEST_SET = True
    RUN_NEURAL = True
    STOP_ON_NEURAL_ERROR = True
    STOP_AFTER_LAYER = ""
    # 每个 chunk 内按 CSD bit 分组发射 Neural CPU 指令。
    LANE_CHUNK = 131_072
    TEST_BATCH_SIZE = 256
    PRINT_EVERY_CHUNKS = 8
    ALLOW_TF32 = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_mnist_test_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha256 in Config.MNIST_TEST_FILES.items():
        destination = data_dir / filename
        if destination.exists() and _sha256(destination) == expected_sha256:
            continue

        errors: list[str] = []
        for mirror in Config.MNIST_MIRRORS:
            url = mirror + filename
            temporary = destination.with_name(destination.name + ".part")
            try:
                print(f"下载 MNIST 测试文件：{url}")
                with urllib.request.urlopen(
                    url, timeout=Config.DOWNLOAD_TIMEOUT_SECONDS
                ) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                actual_sha256 = _sha256(temporary)
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"SHA-256 不匹配：期望 {expected_sha256}，实际 {actual_sha256}"
                    )
                temporary.replace(destination)
                break
            except Exception as error:
                temporary.unlink(missing_ok=True)
                errors.append(f"{url}: {error}")
        else:
            raise RuntimeError(
                f"无法下载 {filename}。也可以手动放入 {data_dir}。\n"
                + "\n".join(errors)
            )


def resolve_paths() -> tuple[Path, Path, Path]:
    cnn_path = _first_existing([Config.CNN_MODEL], "MNIST CNN 权重")
    cpu_path = _first_existing([Config.CPU_MODEL], "Neural CPU v3 权重")
    ensure_mnist_test_data(Config.DATA_DIR)
    data_dir = Config.DATA_DIR
    return cnn_path, cpu_path, data_dir


# =============================================================================
# MNIST 与量化 CNN
# =============================================================================


@dataclass(frozen=True)
class CNNConfig:
    seed: int = 20260804
    conv1_channels: int = 32
    conv1b_channels: int = 32
    conv2_channels: int = 64
    conv2b_channels: int = 64
    fc_hidden: int = 256
    batch_size: int = 512
    epochs: int = 30
    float_warmup_epochs: int = 20
    learning_rate: float = 1e-3
    mid_learning_rate: float = 3e-4
    qat_learning_rate: float = 3e-5
    final_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    input_scale: int = 255
    weight_scale: int = 4096
    weight_qmax: int = 32767
    hidden_scale: int = 256
    hidden_qmax: int = 65535
    dropout: float = 0.3
    target_accuracy: float = 0.995
    num_workers: int = 4


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object, tensor: torch.Tensor, scale: int, lower: int, upper: int
    ) -> torch.Tensor:
        del ctx
        return torch.clamp(torch.round(tensor * scale), lower, upper) / scale

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        del ctx
        return gradient, None, None, None


quantize_ste = QuantizeSTE.apply


class QuantizedCNN(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, config.conv1_channels, 3, padding=1)
        self.conv1b = nn.Conv2d(
            config.conv1_channels, config.conv1b_channels, 3, padding=1
        )
        self.conv2 = nn.Conv2d(
            config.conv1b_channels, config.conv2_channels, 3, padding=1
        )
        self.conv2b = nn.Conv2d(
            config.conv2_channels, config.conv2b_channels, 3, padding=1
        )
        self.fc1 = nn.Linear(config.conv2b_channels * 7 * 7, config.fc_hidden)
        self.fc2 = nn.Linear(config.fc_hidden, 10)
        self.quantization_enabled = True

    def _weight(self, value: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return value
        cfg = self.config
        return quantize_ste(value, cfg.weight_scale, -cfg.weight_qmax, cfg.weight_qmax)

    def _bias(self, value: torch.Tensor, input_scale: int) -> torch.Tensor:
        if not self.quantization_enabled:
            return value
        return quantize_ste(
            value,
            input_scale * self.config.weight_scale,
            -(2**31),
            2**31 - 1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x = quantize_ste(inputs, cfg.input_scale, 0, 255)
        x = F.conv2d(
            x,
            self._weight(self.conv1.weight),
            self._bias(self.conv1.bias, cfg.input_scale),
            padding=1,
        )
        x = quantize_ste(F.relu(x), cfg.hidden_scale, 0, cfg.hidden_qmax)
        x = F.conv2d(
            x,
            self._weight(self.conv1b.weight),
            self._bias(self.conv1b.bias, cfg.hidden_scale),
            padding=1,
        )
        x = F.max_pool2d(F.relu(x), 2)
        x = quantize_ste(x, cfg.hidden_scale, 0, cfg.hidden_qmax)
        x = F.conv2d(
            x,
            self._weight(self.conv2.weight),
            self._bias(self.conv2.bias, cfg.hidden_scale),
            padding=1,
        )
        x = quantize_ste(F.relu(x), cfg.hidden_scale, 0, cfg.hidden_qmax)
        x = F.conv2d(
            x,
            self._weight(self.conv2b.weight),
            self._bias(self.conv2b.bias, cfg.hidden_scale),
            padding=1,
        )
        x = F.max_pool2d(F.relu(x), 2)
        x = quantize_ste(x, cfg.hidden_scale, 0, cfg.hidden_qmax)
        x = x.flatten(1)
        x = F.linear(
            x,
            self._weight(self.fc1.weight),
            self._bias(self.fc1.bias, cfg.hidden_scale),
        )
        x = quantize_ste(F.relu(x), cfg.hidden_scale, 0, cfg.hidden_qmax)
        return F.linear(
            x,
            self._weight(self.fc2.weight),
            self._bias(self.fc2.bias, cfg.hidden_scale),
        )


def read_idx_gzip(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    image_path = data_dir / "t10k-images-idx3-ubyte.gz"
    label_path = data_dir / "t10k-labels-idx1-ubyte.gz"
    with gzip.open(image_path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if (magic, rows, columns) != (2051, 28, 28):
            raise ValueError(f"无效的 MNIST 图像文件：{image_path}")
        images = torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8).clone()
        images = images.reshape(count, 1, 28, 28)
    with gzip.open(label_path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"无效的 MNIST 标签文件：{label_path}")
        labels = torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8).clone()
        labels = labels.to(torch.long)
    return images, labels


def load_cnn(path: Path, device: torch.device) -> tuple[QuantizedCNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = CNNConfig(**checkpoint["config"])
    model = QuantizedCNN(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.quantization_enabled = True
    return model, checkpoint


def quantized_parameters(model: QuantizedCNN) -> dict[str, torch.Tensor]:
    cfg = model.config
    result: dict[str, torch.Tensor] = {}
    for name, layer, input_scale in (
        ("conv1", model.conv1, cfg.input_scale),
        ("conv1b", model.conv1b, cfg.hidden_scale),
        ("conv2", model.conv2, cfg.hidden_scale),
        ("conv2b", model.conv2b, cfg.hidden_scale),
        ("fc1", model.fc1, cfg.hidden_scale),
        ("fc2", model.fc2, cfg.hidden_scale),
    ):
        result[f"{name}.weight"] = torch.clamp(
            torch.round(layer.weight.detach() * cfg.weight_scale),
            -cfg.weight_qmax,
            cfg.weight_qmax,
        ).to(torch.int64)
        result[f"{name}.bias"] = torch.round(
            layer.bias.detach() * input_scale * cfg.weight_scale
        ).to(torch.int64)
    return result


def round_divide_ties_to_even(value: torch.Tensor, denominator: int) -> torch.Tensor:
    quotient = torch.div(value, denominator, rounding_mode="floor")
    remainder = value - quotient * denominator
    twice = remainder * 2
    increment = (twice > denominator) | (
        (twice == denominator) & ((quotient & 1) == 1)
    )
    return quotient + increment.to(value.dtype)


def exact_integer_conv2d(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # 当前模型的最大累加值远低于 2^53，因此 float64 卷积在整数域中精确。
    result = F.conv2d(
        inputs.to(torch.float64),
        weight.to(torch.float64),
        bias.to(torch.float64),
        padding=1,
    )
    rounded = torch.round(result)
    if int(rounded.abs().amax().item()) >= 2**53:
        raise ArithmeticError("整数卷积越过 float64 的精确整数区间。")
    return rounded.to(torch.int64)


def exact_integer_linear(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """在 CUDA 不支持 int64 GEMM 时，用 float64 精确完成整数全连接。"""
    result = F.linear(
        inputs.to(torch.float64),
        weight.to(torch.float64),
        bias.to(torch.float64),
    )
    rounded = torch.round(result)
    if int(rounded.abs().amax().item()) >= 2**53:
        raise ArithmeticError("整数全连接越过 float64 的精确整数区间。")
    return rounded.to(torch.int64)


def integer_forward_trace(
    model: QuantizedCNN, inputs_u8: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = model.config
    p = quantized_parameters(model)
    trace: dict[str, torch.Tensor] = {}
    x = inputs_u8.to(torch.int64)
    x = exact_integer_conv2d(x, p["conv1.weight"], p["conv1.bias"])
    x = round_divide_ties_to_even(
        torch.clamp_min(x, 0) * cfg.hidden_scale,
        cfg.input_scale * cfg.weight_scale,
    ).clamp(0, cfg.hidden_qmax)
    trace["conv1"] = x
    x = exact_integer_conv2d(x, p["conv1b.weight"], p["conv1b.bias"])
    x = round_divide_ties_to_even(
        torch.clamp_min(x, 0), cfg.weight_scale
    ).clamp(0, cfg.hidden_qmax)
    x = F.max_pool2d(x.to(torch.float64), 2).to(torch.int64)
    trace["conv1b_pool"] = x
    x = exact_integer_conv2d(x, p["conv2.weight"], p["conv2.bias"])
    x = round_divide_ties_to_even(
        torch.clamp_min(x, 0), cfg.weight_scale
    ).clamp(0, cfg.hidden_qmax)
    trace["conv2"] = x
    x = exact_integer_conv2d(x, p["conv2b.weight"], p["conv2b.bias"])
    x = round_divide_ties_to_even(
        torch.clamp_min(x, 0), cfg.weight_scale
    ).clamp(0, cfg.hidden_qmax)
    x = F.max_pool2d(x.to(torch.float64), 2).to(torch.int64)
    trace["conv2b_pool"] = x
    x = x.flatten(1)
    x = exact_integer_linear(x, p["fc1.weight"], p["fc1.bias"])
    x = round_divide_ties_to_even(
        torch.clamp_min(x, 0), cfg.weight_scale
    ).clamp(0, cfg.hidden_qmax)
    trace["fc1"] = x
    logits = exact_integer_linear(x, p["fc2.weight"], p["fc2.bias"])
    trace["fc2_logits"] = logits
    return logits, trace


def naf_terms(value: int) -> tuple[tuple[int, int], ...]:
    """把有符号常量编译成非相邻形式 ``(bit, sign)``。"""
    if value == 0:
        return ()
    sign = -1 if value < 0 else 1
    remaining = abs(value)
    bit = 0
    terms: list[tuple[int, int]] = []
    while remaining:
        if remaining & 1:
            digit = 2 - (remaining & 3)
            terms.append((bit, sign * digit))
            remaining -= digit
        remaining //= 2
        bit += 1
    return tuple(terms)


def naf_multiply_lane_ops(value: int) -> int:
    """估算一次 ``uint16 * 常量 int16`` 的字节 ISA 操作数。"""
    # 每个 NAF 项需要一次 32-bit ADD/SUB。非整字节位移还需要两次 SHL、
    # 两次 SHR 和一次 OR；整字节位移只是编译期字节重排。
    return sum(4 + (0 if bit % 8 == 0 else 5) for bit, _ in naf_terms(value))


def estimate_neural_lane_operations(
    model: QuantizedCNN, trace: dict[str, torch.Tensor]
) -> int:
    """按 CSD/NAF 常量乘法方案估算一张图像的 Neural CPU lane-op。"""
    config = model.config
    conv1_outputs = config.conv1_channels * 28 * 28
    conv1b_outputs = config.conv1b_channels * 28 * 28
    conv2_outputs = config.conv2_channels * 14 * 14
    conv2b_outputs = config.conv2b_channels * 14 * 14
    fc1_outputs = config.fc_hidden
    fc2_outputs = 10
    macs = (
        conv1_outputs * 9
        + conv1b_outputs * config.conv1_channels * 9
        + conv2_outputs * config.conv1b_channels * 9
        + conv2b_outputs * config.conv2_channels * 9
        + fc1_outputs * config.conv2b_channels * 7 * 7
        + fc2_outputs * config.fc_hidden
    )
    dot_outputs = (
        conv1_outputs
        + conv1b_outputs
        + conv2_outputs
        + conv2b_outputs
        + fc1_outputs
        + fc2_outputs
    )
    parameters = quantized_parameters(model)
    layer_positions = {
        "conv1.weight": 28 * 28,
        "conv1b.weight": 28 * 28,
        "conv2.weight": 14 * 14,
        "conv2b.weight": 14 * 14,
        "fc1.weight": 1,
        "fc2.weight": 1,
    }
    multiply_ops = 0
    for name, positions in layer_positions.items():
        costs = sum(
            naf_multiply_lane_ops(int(value))
            for value in parameters[name].detach().cpu().reshape(-1).tolist()
        )
        multiply_ops += positions * costs
    # 树归约每个 MAC 渐近需要一次 32-bit ADD；bias 的四次 MOVI 和一次
    # 32-bit ADD 抵消归约树少掉的最后一次 ADD，合计 4*MAC+4*输出。
    dot_ops = multiply_ops + 4 * macs + 4 * dot_outputs
    conv1_quotient_max = int(trace["conv1"].amax().item())
    conv1_requant_ops = conv1_outputs * (10 * conv1_quotient_max + 13)
    power_of_two_requant_outputs = (
        conv1b_outputs + conv2_outputs + conv2b_outputs + fc1_outputs
    )
    requant_ops = conv1_requant_ops + 23 * power_of_two_requant_outputs
    pool_outputs = config.conv1b_channels * 14 * 14 + config.conv2b_channels * 7 * 7
    pool_ops = 30 * pool_outputs
    argmax_ops = 235
    return dot_ops + requant_ops + pool_ops + argmax_ops


# =============================================================================
# Neural CPU v3 ISA 与向量执行器
# =============================================================================


INPUT_BITS = 51
NUM_FLAGS = 3
NUM_REGISTERS = 4
REGISTER_BITS = 8
STATE_BITS = 35
OUTPUT_BITS = 39
FLAG_ZF = 0
FLAG_GF = 1
FLAG_CF = 2
CTRL_MEMORY_READ = 0

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


def load_cpu(path: Path, device: torch.device) -> tuple[NeuralCPUCore, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {})
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
    return model, checkpoint


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


def i32_to_bytes(values: torch.Tensor) -> torch.Tensor:
    values = torch.bitwise_and(values.to(torch.int64), 0xFFFFFFFF)
    return torch.stack(
        [torch.bitwise_and(torch.bitwise_right_shift(values, 8 * i), 255) for i in range(4)],
        dim=-1,
    ).to(torch.uint8)


def bytes_to_i32(values: torch.Tensor) -> torch.Tensor:
    wide = values.to(torch.int64)
    unsigned = wide[..., 0]
    unsigned |= wide[..., 1] << 8
    unsigned |= wide[..., 2] << 16
    unsigned |= wide[..., 3] << 24
    return torch.where(unsigned >= 2**31, unsigned - 2**32, unsigned)


def u16_to_bytes(values: torch.Tensor) -> torch.Tensor:
    values = torch.bitwise_and(values.to(torch.int64), 0xFFFF)
    return torch.stack((values & 255, (values >> 8) & 255), dim=-1).to(torch.uint8)


def bytes_to_u16(values: torch.Tensor) -> torch.Tensor:
    return values[..., 0].to(torch.int64) | (values[..., 1].to(torch.int64) << 8)


@dataclass
class AuditCounters:
    neural_calls: int = 0
    neural_lane_operations: int = 0
    result_bit_errors: int = 0
    flag_bit_errors: int = 0
    control_bit_errors: int = 0
    first_error: dict[str, Any] | None = None


class NeuralExecutionError(RuntimeError):
    pass


class NeuralByteMachine:
    """把 GPU batch 中的每一行当作一颗 Neural CPU lane。"""

    def __init__(self, model: NeuralCPUCore, device: torch.device):
        self.model = model
        self.device = device
        self.audit = AuditCounters()
        self.bit_shifts_16 = torch.arange(15, -1, -1, device=device)
        self.bit_shifts_8 = torch.arange(7, -1, -1, device=device)

    def _instruction_bits(
        self, instruction: int | torch.Tensor, size: int
    ) -> torch.Tensor:
        if isinstance(instruction, torch.Tensor):
            values = instruction.to(self.device, torch.int64).reshape(-1)
        else:
            values = torch.full((size,), instruction, device=self.device, dtype=torch.int64)
        return ((values[:, None] >> self.bit_shifts_16[None, :]) & 1).to(torch.float32)

    def _state_bits(self, registers: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        register_bits = (
            (registers.to(torch.int64)[:, :, None] >> self.bit_shifts_8[None, None, :])
            & 1
        ).reshape(registers.shape[0], -1)
        return torch.cat((flags.to(torch.float32), register_bits.to(torch.float32)), 1)

    @torch.inference_mode()
    def execute(
        self,
        instruction: int | torch.Tensor,
        registers: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        registers = registers.to(self.device, torch.uint8).reshape(-1, NUM_REGISTERS)
        size = registers.shape[0]
        if flags is None:
            flags = torch.zeros((size, NUM_FLAGS), device=self.device, dtype=torch.uint8)
        inputs = torch.cat(
            (self._instruction_bits(instruction, size), self._state_bits(registers, flags)),
            1,
        )
        bits = self.model(inputs) >= 0
        out_flags = bits[:, :NUM_FLAGS].to(torch.uint8)
        register_bits = bits[:, NUM_FLAGS:STATE_BITS].reshape(size, 4, 8)
        weights = (1 << self.bit_shifts_8).to(torch.int64)
        out_registers = torch.sum(
            register_bits.to(torch.int64) * weights[None, None, :], dim=2
        ).to(torch.uint8)
        controls = bits[:, STATE_BITS:OUTPUT_BITS].to(torch.uint8)
        self.audit.neural_calls += 1
        self.audit.neural_lane_operations += size
        return out_registers, out_flags, controls

    def _record_error(
        self,
        operation: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        category: str,
    ) -> None:
        difference = actual != expected
        count = int(torch.count_nonzero(difference).item())
        if count == 0:
            return
        if category == "result":
            self.audit.result_bit_errors += count
        elif category == "flag":
            self.audit.flag_bit_errors += count
        else:
            self.audit.control_bit_errors += count
        if self.audit.first_error is None:
            first = torch.nonzero(difference, as_tuple=False)[0]
            row = int(first[0].item())
            column = int(first[1].item()) if first.numel() > 1 else 0
            self.audit.first_error = {
                "operation": operation,
                "category": category,
                "row": row,
                "column": column,
                "actual": int(actual.reshape(actual.shape[0], -1)[row, column].item()),
                "expected": int(expected.reshape(expected.shape[0], -1)[row, column].item()),
                "lane_operations": self.audit.neural_lane_operations,
            }
        if Config.STOP_ON_NEURAL_ERROR:
            raise NeuralExecutionError(
                f"{operation} 出现 {count} 个 {category} 错误；"
                f"首错={self.audit.first_error}"
            )

    def _rrr(
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
        instruction = encode_instruction(OPCODES[name], rd=0, ra=1, rb=2)
        out, out_flags, _ = self.execute(instruction, registers, flags)
        l16 = left.to(torch.int16)
        r16 = right.to(torch.int16)
        carry16 = flags[:, FLAG_CF].to(torch.int16)
        expected_flags = flags.clone()
        if name in ("ADD", "ADC"):
            total = l16 + r16 + (carry16 if name == "ADC" else 0)
            expected = (total & 255).to(torch.uint8)
            expected_flags[:, FLAG_ZF] = (expected == 0).to(torch.uint8)
            expected_flags[:, FLAG_CF] = (total > 255).to(torch.uint8)
        elif name in ("SUB", "SBC"):
            borrow = carry16 if name == "SBC" else 0
            total = l16 - r16 - borrow
            expected = (total & 255).to(torch.uint8)
            expected_flags[:, FLAG_ZF] = (expected == 0).to(torch.uint8)
            expected_flags[:, FLAG_CF] = (l16 < r16 + borrow).to(torch.uint8)
        elif name == "AND":
            expected = torch.bitwise_and(left, right)
            expected_flags[:, FLAG_ZF] = (expected == 0).to(torch.uint8)
        elif name == "OR":
            expected = torch.bitwise_or(left, right)
            expected_flags[:, FLAG_ZF] = (expected == 0).to(torch.uint8)
        elif name == "XOR":
            expected = torch.bitwise_xor(left, right)
            expected_flags[:, FLAG_ZF] = (expected == 0).to(torch.uint8)
        elif name in ("SHL", "SHR"):
            shift = (right & 7).to(torch.int16)
            if name == "SHL":
                expected = ((l16 << shift) & 255).to(torch.uint8)
                shifted_carry = torch.where(
                    shift == 0,
                    carry16,
                    (l16 >> (8 - shift)) & 1,
                )
            else:
                expected = (l16 >> shift).to(torch.uint8)
                shifted_carry = torch.where(
                    shift == 0,
                    carry16,
                    (l16 >> (shift - 1)) & 1,
                )
            expected_flags[:, FLAG_ZF] = (expected == 0).to(torch.uint8)
            expected_flags[:, FLAG_CF] = shifted_carry.to(torch.uint8)
        elif name == "CMP":
            expected = torch.zeros_like(left)
            expected_flags[:, FLAG_ZF] = (left == right).to(torch.uint8)
            expected_flags[:, FLAG_GF] = (left > right).to(torch.uint8)
            expected_flags[:, FLAG_CF] = (left < right).to(torch.uint8)
            # CMP 不写寄存器，R0 的期望仍为零。
        else:
            raise ValueError(name)
        self._record_error(name, out[:, 0:1], expected[:, None], "result")
        self._record_error(name, out_flags, expected_flags, "flag")
        return out[:, 0], out_flags

    def movi(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(self.device, torch.uint8).reshape(-1)
        size = values.numel()
        registers = torch.zeros((size, 4), device=self.device, dtype=torch.uint8)
        instruction = encode_instruction(OPCODES["MOVI"], rd=0, immediate=values)
        out, out_flags, controls = self.execute(instruction, registers)
        self._record_error("MOVI", out[:, 0:1], values[:, None], "result")
        self._record_error(
            "MOVI", out_flags, torch.zeros_like(out_flags), "flag"
        )
        self._record_error(
            "MOVI", controls, torch.zeros_like(controls), "control"
        )
        return out[:, 0]

    def add_u32(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        carry = None
        for byte in range(4):
            name = "ADD" if byte == 0 else "ADC"
            value, flags = self._rrr(name, left[:, byte], right[:, byte], carry)
            outputs.append(value)
            carry = flags[:, FLAG_CF]
        return torch.stack(outputs, dim=1)

    def sub_u32(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        borrow = None
        for byte in range(4):
            name = "SUB" if byte == 0 else "SBC"
            value, flags = self._rrr(name, left[:, byte], right[:, byte], borrow)
            outputs.append(value)
            borrow = flags[:, FLAG_CF]
        return torch.stack(outputs, dim=1)

    def negate_u32(self, values: torch.Tensor) -> torch.Tensor:
        return self.sub_u32(torch.zeros_like(values), values)

    def add_u16_increment(
        self, values: torch.Tensor, increment: torch.Tensor
    ) -> torch.Tensor:
        low, flags = self._rrr("ADD", values[:, 0], increment.to(torch.uint8))
        high, _ = self._rrr(
            "ADC", values[:, 1], torch.zeros_like(values[:, 1]), flags[:, FLAG_CF]
        )
        return torch.stack((low, high), dim=1)

    def compare_unsigned(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = torch.ones(left.shape[0], device=self.device, dtype=torch.bool)
        greater = torch.zeros_like(active)
        less = torch.zeros_like(active)
        for byte in range(left.shape[1] - 1, -1, -1):
            _, flags = self._rrr("CMP", left[:, byte], right[:, byte])
            greater |= active & flags[:, FLAG_GF].bool()
            less |= active & flags[:, FLAG_CF].bool()
            active &= flags[:, FLAG_ZF].bool()
        return greater, active, less

    def compare_signed_i32(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """比较两个小端 int32；符号和逐字节比较均由 Neural CPU 产生。"""
        size = left.shape[0]
        shift7 = torch.full((size,), 7, device=self.device, dtype=torch.uint8)
        left_sign, _ = self._rrr("SHR", left[:, 3], shift7)
        right_sign, _ = self._rrr("SHR", right[:, 3], shift7)
        unsigned_greater, equal, unsigned_less = self.compare_unsigned(left, right)
        left_negative = left_sign.bool()
        right_negative = right_sign.bool()
        signs_differ = left_negative != right_negative
        greater = (signs_differ & torch.logical_not(left_negative)) | (
            torch.logical_not(signs_differ) & unsigned_greater
        )
        less = (signs_differ & left_negative) | (
            torch.logical_not(signs_differ) & unsigned_less
        )
        return greater, equal, less

    def _condition_mask(
        self, condition: torch.Tensor, *, invert: bool = False
    ) -> torch.Tensor:
        """用 SBC 把 CPU 条件标志扩展成 0x00/0xFF 字节掩码。"""
        carry = condition.to(self.device, torch.bool).reshape(-1)
        if invert:
            carry = torch.logical_not(carry)
        zero = torch.zeros(carry.numel(), device=self.device, dtype=torch.uint8)
        mask, _ = self._rrr("SBC", zero, zero, carry.to(torch.uint8))
        return mask

    def select_bytes(
        self,
        condition: torch.Tensor,
        when_true: torch.Tensor,
        when_false: torch.Tensor,
    ) -> torch.Tensor:
        """完全用 Neural CPU 的 SBC/AND/OR 实现逐 lane 条件选择。"""
        when_true = when_true.to(self.device, torch.uint8)
        when_false = when_false.to(self.device, torch.uint8)
        if when_true.shape != when_false.shape:
            raise ValueError("条件选择两侧的字节张量形状必须一致。")
        flat_true = when_true.reshape(-1, when_true.shape[-1])
        flat_false = when_false.reshape(-1, when_false.shape[-1])
        mask = self._condition_mask(condition)
        inverse = self._condition_mask(condition, invert=True)
        columns: list[torch.Tensor] = []
        for byte in range(flat_true.shape[1]):
            true_part, _ = self._rrr("AND", flat_true[:, byte], mask)
            false_part, _ = self._rrr("AND", flat_false[:, byte], inverse)
            selected, _ = self._rrr("OR", true_part, false_part)
            columns.append(selected)
        return torch.stack(columns, 1).reshape(when_true.shape)

    def zero_when(self, condition: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """条件为真时清零；数值选择由 Neural CPU 掩码完成。"""
        values = values.to(self.device, torch.uint8)
        flat = values.reshape(-1, values.shape[-1])
        keep_mask = self._condition_mask(condition, invert=True)
        columns = [self._rrr("AND", flat[:, byte], keep_mask)[0] for byte in range(flat.shape[1])]
        return torch.stack(columns, 1).reshape(values.shape)

    def fill_ff_when(self, condition: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """条件为真时饱和到全 1；数值选择由 Neural CPU 掩码完成。"""
        values = values.to(self.device, torch.uint8)
        flat = values.reshape(-1, values.shape[-1])
        fill_mask = self._condition_mask(condition)
        columns = [self._rrr("OR", flat[:, byte], fill_mask)[0] for byte in range(flat.shape[1])]
        return torch.stack(columns, 1).reshape(values.shape)

    def shift_u16_to_u32(self, values: torch.Tensor, bit: int) -> torch.Tensor:
        """用 Neural CPU 把 uint16 左移常量位，结果写入四个小端字节。"""
        if not 0 <= bit <= 15:
            raise ValueError(f"CSD 位移超出 int16 常量范围：{bit}")
        values = values.to(self.device, torch.int64).reshape(-1)
        source = u16_to_bytes(values)
        result = torch.zeros((values.numel(), 4), device=self.device, dtype=torch.uint8)
        byte_offset, residual = divmod(bit, 8)
        if residual == 0:
            result[:, byte_offset : byte_offset + 2] = source
            return result

        left_shift = torch.full(
            (values.numel(),), residual, device=self.device, dtype=torch.uint8
        )
        right_shift = torch.full(
            (values.numel(),), 8 - residual, device=self.device, dtype=torch.uint8
        )
        low_left, _ = self._rrr("SHL", source[:, 0], left_shift)
        high_left, _ = self._rrr("SHL", source[:, 1], left_shift)
        low_carry, _ = self._rrr("SHR", source[:, 0], right_shift)
        high_carry, _ = self._rrr("SHR", source[:, 1], right_shift)
        middle, _ = self._rrr("OR", high_left, low_carry)
        result[:, byte_offset] = low_left
        result[:, byte_offset + 1] = middle
        result[:, byte_offset + 2] = high_carry
        return result

    def multiply_u16_constant_csd(
        self, values: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """把 int16 常量权重编译成 NAF 移位加减并由 Neural CPU 执行。"""
        values = values.to(self.device, torch.int64).reshape(-1)
        weights = weights.to(self.device, torch.int64).reshape(-1)
        if values.shape != weights.shape:
            raise ValueError("常量乘法的输入与权重形状必须一致。")

        product = torch.zeros((values.numel(), 4), device=self.device, dtype=torch.uint8)
        remaining = weights.abs().clone()
        weight_sign = torch.ones_like(weights)
        weight_sign[weights < 0] = -1
        bit = 0
        while bool((remaining != 0).any().item()):
            odd = (remaining & 1) != 0
            digit = torch.zeros_like(remaining)
            digit[odd] = 2 - (remaining[odd] & 3)
            signed_digit = digit * weight_sign
            active = signed_digit != 0
            if bool(active.any().item()):
                indices = torch.nonzero(active, as_tuple=False).flatten()
                shifted = self.shift_u16_to_u32(values[indices], bit)
                positive = signed_digit[indices] > 0
                if bool(positive.any().item()):
                    target = indices[positive]
                    product[target] = self.add_u32(product[target], shifted[positive])
                negative = torch.logical_not(positive)
                if bool(negative.any().item()):
                    target = indices[negative]
                    product[target] = self.sub_u32(product[target], shifted[negative])
            remaining = (remaining - digit) // 2
            bit += 1
            if bit > 16:
                raise RuntimeError("CSD 编译异常：int16 权重超过 16 位。")

        expected = values * weights
        self._record_error(
            "MUL16_CSD",
            bytes_to_i32(product)[:, None],
            expected[:, None],
            "result",
        )
        return product


class NeuralFixedPointBackend:
    def __init__(self, machine: NeuralByteMachine):
        self.machine = machine
        self.device = machine.device

    def _chunks(self, size: int):
        for start in range(0, size, Config.LANE_CHUNK):
            yield start, min(start + Config.LANE_CHUNK, size)

    def multiply_many(self, values: torch.Tensor, weights: torch.Tensor, label: str) -> torch.Tensor:
        result = torch.empty((values.numel(), 4), device=self.device, dtype=torch.uint8)
        chunk_count = (values.numel() + Config.LANE_CHUNK - 1) // Config.LANE_CHUNK
        started = time.perf_counter()
        for chunk_index, (start, end) in enumerate(self._chunks(values.numel()), 1):
            result[start:end] = self.machine.multiply_u16_constant_csd(
                values[start:end], weights[start:end]
            )
            if chunk_index % Config.PRINT_EVERY_CHUNKS == 0 or chunk_index == chunk_count:
                elapsed = time.perf_counter() - started
                print(
                    f"  [{label}] CSD 常量乘法 {end:,}/{values.numel():,} | "
                    f"chunk={chunk_index}/{chunk_count} | {end / max(elapsed, 1e-9):,.0f} MAC/s"
                )
        return result

    def add_many(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start, end in self._chunks(left.shape[0]):
            outputs.append(self.machine.add_u32(left[start:end], right[start:end]))
        return torch.cat(outputs, 0)

    def sub_many(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start, end in self._chunks(left.shape[0]):
            outputs.append(self.machine.sub_u32(left[start:end], right[start:end]))
        return torch.cat(outputs, 0)

    def compare_many(self, left: torch.Tensor, right: torch.Tensor):
        greater_parts = []
        equal_parts = []
        less_parts = []
        for start, end in self._chunks(left.shape[0]):
            greater, equal, less = self.machine.compare_unsigned(
                left[start:end], right[start:end]
            )
            greater_parts.append(greater)
            equal_parts.append(equal)
            less_parts.append(less)
        return torch.cat(greater_parts), torch.cat(equal_parts), torch.cat(less_parts)

    def argmax_i32(self, values: torch.Tensor) -> int:
        """用 Neural CPU 的有符号比较和掩码选择完成最终分类。"""
        values = values.to(self.device, torch.uint8).reshape(-1, 4)
        if values.shape[0] == 0:
            raise ValueError("argmax 至少需要一个候选值。")
        current_value = values[0:1].clone()
        current_index = self.machine.movi(
            torch.zeros(1, device=self.device, dtype=torch.uint8)
        ).reshape(1, 1)
        for index in range(1, values.shape[0]):
            candidate_value = values[index : index + 1]
            candidate_index = self.machine.movi(
                torch.full((1,), index, device=self.device, dtype=torch.uint8)
            ).reshape(1, 1)
            greater, _, _ = self.machine.compare_signed_i32(
                candidate_value, current_value
            )
            current_value = self.machine.select_bytes(
                greater, candidate_value, current_value
            )
            current_index = self.machine.select_bytes(
                greater, candidate_index, current_index
            )
        return int(current_index[0, 0].item())

    def movi_u32(self, values: torch.Tensor) -> torch.Tensor:
        raw = i32_to_bytes(values).reshape(-1, 4)
        columns = []
        for byte in range(4):
            parts = []
            for start, end in self._chunks(raw.shape[0]):
                parts.append(self.machine.movi(raw[start:end, byte]))
            columns.append(torch.cat(parts))
        return torch.stack(columns, 1)

    def dot_products(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        """计算一批共享输入的点积，返回 little-endian int32 字节。"""
        # inputs: [positions, terms], weights: [outputs, terms]
        positions, terms = inputs.shape
        outputs = weights.shape[0]
        flat_inputs = inputs[None, :, :].expand(outputs, positions, terms).reshape(-1)
        flat_weights = weights[:, None, :].expand(outputs, positions, terms).reshape(-1)
        products = self.multiply_many(flat_inputs, flat_weights, label)
        values = products.reshape(outputs * positions, terms, 4)
        stage = 0
        while values.shape[1] > 1:
            stage += 1
            term_count = values.shape[1]
            even = term_count - (term_count & 1)
            paired = values[:, :even].reshape(values.shape[0], even // 2, 2, 4)
            reduced = self.add_many(
                paired[:, :, 0].reshape(-1, 4),
                paired[:, :, 1].reshape(-1, 4),
            ).reshape(values.shape[0], even // 2, 4)
            if term_count & 1:
                reduced = torch.cat((reduced, values[:, -1:, :]), 1)
            values = reduced
            print(f"  [{label}] 神经归约 stage={stage} | 剩余项={values.shape[1]}")
        total = values[:, 0]
        bias_values = bias[:, None].expand(outputs, positions).reshape(-1)
        total = self.add_many(total, self.movi_u32(bias_values))
        return total.reshape(outputs, positions, 4)

    def requantize_relu_4096(self, accumulators: torch.Tensor) -> torch.Tensor:
        flat = accumulators.reshape(-1, 4)
        outputs: list[torch.Tensor] = []
        for start, end in self._chunks(flat.shape[0]):
            value = flat[start:end]
            shift4 = torch.full((end - start,), 4, device=self.device, dtype=torch.uint8)
            shift7 = torch.full((end - start,), 7, device=self.device, dtype=torch.uint8)
            zero = torch.zeros(end - start, device=self.device, dtype=torch.uint8)
            sign, _ = self.machine._rrr("SHR", value[:, 3], shift7)
            b1_high, _ = self.machine._rrr("SHR", value[:, 1], shift4)
            b2_low, _ = self.machine._rrr("SHL", value[:, 2], shift4)
            q0, _ = self.machine._rrr("OR", b1_high, b2_low)
            b2_high, _ = self.machine._rrr("SHR", value[:, 2], shift4)
            b3_low, _ = self.machine._rrr("SHL", value[:, 3], shift4)
            q1, _ = self.machine._rrr("OR", b2_high, b3_low)
            low_nibble, _ = self.machine._rrr(
                "AND", value[:, 1], torch.full_like(value[:, 1], 15)
            )
            _, nibble_flags = self.machine._rrr(
                "CMP", low_nibble, torch.full_like(low_nibble, 8)
            )
            _, low_flags = self.machine._rrr("CMP", value[:, 0], zero)
            parity, _ = self.machine._rrr("AND", q0, torch.ones_like(q0))
            greater_half = nibble_flags[:, FLAG_GF].bool() | (
                nibble_flags[:, FLAG_ZF].bool() & low_flags[:, FLAG_GF].bool()
            )
            exact_half = nibble_flags[:, FLAG_ZF].bool() & low_flags[:, FLAG_ZF].bool()
            increment = greater_half | (exact_half & parity.bool())
            quantized = self.machine.add_u16_increment(
                torch.stack((q0, q1), 1), increment.to(torch.uint8)
            )
            # round(acc / 4096) 首次得到 65536 的整数阈值。比较与饱和
            # 选择都由 Neural CPU 执行。
            saturation_threshold = 65535 * 4096 + 2048
            threshold_bytes = i32_to_bytes(
                torch.full(
                    (end - start,),
                    saturation_threshold,
                    device=self.device,
                    dtype=torch.int64,
                )
            )
            above, at_threshold, _ = self.machine.compare_unsigned(
                value, threshold_bytes
            )
            saturated = above | at_threshold
            quantized = self.machine.fill_ff_when(saturated, quantized)
            quantized = self.machine.zero_when(sign.bool(), quantized)
            expected_acc = bytes_to_i32(value)
            expected = round_divide_ties_to_even(
                torch.clamp_min(expected_acc, 0), 4096
            ).clamp(0, 65535)
            self.machine._record_error(
                "REQUANT4096",
                bytes_to_u16(quantized)[:, None],
                expected[:, None],
                "result",
            )
            outputs.append(quantized)
        return torch.cat(outputs).reshape(*accumulators.shape[:-1], 2)

    def requantize_relu_4080(self, accumulators: torch.Tensor) -> torch.Tensor:
        """conv1 专用的精确 round-to-even(acc / 4080)。"""
        flat = accumulators.reshape(-1, 4)
        outputs: list[torch.Tensor] = []
        denominator = 4080
        half = denominator // 2
        for start, end in self._chunks(flat.shape[0]):
            value = flat[start:end]
            size = value.shape[0]
            sign, _ = self.machine._rrr(
                "SHR", value[:, 3], torch.full((size,), 7, device=self.device, dtype=torch.uint8)
            )
            remainder = self.machine.zero_when(sign.bool(), value)
            quotient = torch.zeros((size, 2), device=self.device, dtype=torch.uint8)
            divisor = i32_to_bytes(torch.full((size,), denominator, device=self.device))
            iterations = 0
            while True:
                greater, equal, _ = self.machine.compare_unsigned(remainder, divisor)
                active = greater | equal
                if not bool(active.any().item()):
                    break
                # 外部控制器只根据 CPU 比较标志发射活跃 lane；减法和商递增
                # 都由 Neural CPU 执行，写回与真实 SIMD predication 相同。
                indices = torch.nonzero(active, as_tuple=False).flatten()
                remainder[indices] = self.machine.sub_u32(
                    remainder[indices], divisor[indices]
                )
                quotient[indices] = self.machine.add_u16_increment(
                    quotient[indices],
                    torch.ones(indices.numel(), device=self.device, dtype=torch.uint8),
                )
                iterations += 1
                if iterations > 65535:
                    raise RuntimeError("conv1 定点除法异常：商超过 uint16。")
            half_bytes = i32_to_bytes(torch.full((size,), half, device=self.device))
            greater, equal, _ = self.machine.compare_unsigned(remainder, half_bytes)
            parity, _ = self.machine._rrr(
                "AND", quotient[:, 0], torch.ones(size, device=self.device, dtype=torch.uint8)
            )
            increment = greater | (equal & parity.bool())
            quotient = self.machine.add_u16_increment(quotient, increment.to(torch.uint8))
            expected_acc = bytes_to_i32(value)
            expected = round_divide_ties_to_even(
                torch.clamp_min(expected_acc, 0), denominator
            ).clamp(0, 65535)
            self.machine._record_error(
                "REQUANT4080",
                bytes_to_u16(quotient)[:, None],
                expected[:, None],
                "result",
            )
            outputs.append(quotient)
            print(f"  [conv1] 精确除法最大循环={iterations}")
        return torch.cat(outputs).reshape(*accumulators.shape[:-1], 2)

    def max_pool2d(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [channels, height, width, 2]
        channels, height, width, _ = inputs.shape
        candidates = torch.stack(
            (
                inputs[:, 0::2, 0::2],
                inputs[:, 0::2, 1::2],
                inputs[:, 1::2, 0::2],
                inputs[:, 1::2, 1::2],
            ),
            dim=-2,
        )
        current = candidates[..., 0, :].reshape(-1, 2)
        for index in range(1, 4):
            other = candidates[..., index, :].reshape(-1, 2)
            greater, equal, _ = self.compare_many(current, other)
            current = self.machine.select_bytes(greater | equal, current, other)
        return current.reshape(channels, height // 2, width // 2, 2)


# =============================================================================
# CNN 编译与执行
# =============================================================================


class NeuralCNNExecutor:
    def __init__(
        self,
        backend: NeuralFixedPointBackend,
        model: QuantizedCNN,
        exact_trace: dict[str, torch.Tensor],
    ):
        self.backend = backend
        self.model = model
        self.params = {k: v.to(backend.device) for k, v in quantized_parameters(model).items()}
        self.exact_trace = {k: v.to(backend.device) for k, v in exact_trace.items()}

    def _check_layer(self, name: str, actual: torch.Tensor) -> None:
        expected = self.exact_trace[name].squeeze(0)
        actual_values = bytes_to_u16(actual) if actual.shape[-1] == 2 else bytes_to_i32(actual)
        differences = int(torch.count_nonzero(actual_values != expected).item())
        maximum_difference = int((actual_values - expected).abs().amax().item())
        print(
            f"[{name}] shape={tuple(expected.shape)} | 与精确整数核差异={differences} | "
            f"max_abs_diff={maximum_difference}"
        )
        if differences and Config.STOP_ON_NEURAL_ERROR:
            raise NeuralExecutionError(f"{name} 层输出与整数真值不一致。")

    def _conv(
        self, inputs: torch.Tensor, name: str, divisor: int, pool: bool
    ) -> torch.Tensor:
        # inputs 是 [C,H,W] uint16 数值或其双字节表示。
        values = bytes_to_u16(inputs) if inputs.shape[-1] == 2 else inputs.to(torch.int64)
        patches = F.unfold(values[None].to(torch.float64), 3, padding=1)[0].T.to(torch.int64)
        weight = self.params[f"{name}.weight"].reshape(self.params[f"{name}.weight"].shape[0], -1)
        bias = self.params[f"{name}.bias"]
        print(
            f"\n=== {name}: positions={patches.shape[0]:,}, terms={patches.shape[1]:,}, "
            f"outputs={weight.shape[0]:,}, MAC={patches.shape[0] * patches.shape[1] * weight.shape[0]:,} ==="
        )
        accum = self.backend.dot_products(patches, weight, bias, name)
        quantized = (
            self.backend.requantize_relu_4080(accum)
            if divisor == 4080
            else self.backend.requantize_relu_4096(accum)
        )
        height = values.shape[1]
        width = values.shape[2]
        quantized = quantized.reshape(weight.shape[0], height, width, 2)
        if pool:
            quantized = self.backend.max_pool2d(quantized)
        trace_name = {"conv1b": "conv1b_pool", "conv2b": "conv2b_pool"}.get(name, name)
        self._check_layer(trace_name, quantized)
        return quantized

    def _linear(
        self, inputs: torch.Tensor, name: str, requantize: bool
    ) -> torch.Tensor:
        values = bytes_to_u16(inputs).reshape(1, -1)
        weight = self.params[f"{name}.weight"]
        bias = self.params[f"{name}.bias"]
        print(
            f"\n=== {name}: terms={values.shape[1]:,}, outputs={weight.shape[0]:,}, "
            f"MAC={values.shape[1] * weight.shape[0]:,} ==="
        )
        accum = self.backend.dot_products(values, weight, bias, name).reshape(weight.shape[0], 4)
        if requantize:
            result = self.backend.requantize_relu_4096(accum)
        else:
            result = accum
        trace_name = "fc2_logits" if name == "fc2" else name
        self._check_layer(trace_name, result)
        return result

    def run(self, image_u8: torch.Tensor) -> tuple[int, torch.Tensor]:
        if image_u8.ndim == 4:
            if image_u8.shape[0] != 1:
                raise ValueError("完整 Neural CPU 模式当前一次执行一张图像。")
            image_u8 = image_u8[0]
        if image_u8.ndim != 3 or image_u8.shape[0] != 1:
            raise ValueError(
                f"MNIST 输入必须为 [1,28,28] 或 [1,1,28,28]，实际为 {tuple(image_u8.shape)}。"
            )
        x: torch.Tensor = image_u8.to(self.backend.device, torch.int64)
        x = self._conv(x, "conv1", divisor=4080, pool=False)
        if Config.STOP_AFTER_LAYER == "conv1":
            return -1, torch.empty(0)
        x = self._conv(x, "conv1b", divisor=4096, pool=True)
        if Config.STOP_AFTER_LAYER == "conv1b":
            return -1, torch.empty(0)
        x = self._conv(x, "conv2", divisor=4096, pool=False)
        if Config.STOP_AFTER_LAYER == "conv2":
            return -1, torch.empty(0)
        x = self._conv(x, "conv2b", divisor=4096, pool=True)
        if Config.STOP_AFTER_LAYER == "conv2b":
            return -1, torch.empty(0)
        x = self._linear(x, "fc1", requantize=True)
        if Config.STOP_AFTER_LAYER == "fc1":
            return -1, torch.empty(0)
        logits_bytes = self._linear(x, "fc2", requantize=False)
        logits = bytes_to_i32(logits_bytes)
        prediction = self.backend.argmax_i32(logits_bytes)
        return prediction, logits


# =============================================================================
# 入口
# =============================================================================


@torch.inference_mode()
def evaluate_integer_test_set(
    model: QuantizedCNN, images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[float, int]:
    correct = 0
    qat_disagreements = 0
    for start in range(0, len(images), Config.TEST_BATCH_SIZE):
        end = min(start + Config.TEST_BATCH_SIZE, len(images))
        batch_u8 = images[start:end].to(device)
        batch_float = batch_u8.to(torch.float32) / 255.0
        qat = model(batch_float).argmax(1)
        integer_logits, _ = integer_forward_trace(model, batch_u8)
        integer = integer_logits.argmax(1)
        correct += int((integer.cpu() == labels[start:end]).sum().item())
        qat_disagreements += int((integer != qat).sum().item())
    return correct / len(images), qat_disagreements


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
    cnn_path, cpu_path, data_dir = resolve_paths()
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"CNN：{cnn_path}")
    print(f"Neural CPU：{cpu_path}")
    print(f"MNIST：{data_dir}")
    print(f"结果：{Config.RESULT_DIR}")

    images, labels = read_idx_gzip(data_dir)
    if not 0 <= Config.IMAGE_INDEX < len(images):
        raise IndexError(f"IMAGE_INDEX 必须位于 [0, {len(images) - 1}]。")
    cnn, cnn_checkpoint = load_cnn(cnn_path, device)
    print(
        f"CNN checkpoint：epoch={cnn_checkpoint.get('epoch')} | "
        f"QAT test accuracy={cnn_checkpoint.get('qat_test_accuracy')}"
    )
    if Config.RUN_INTEGER_TEST_SET:
        started = time.perf_counter()
        accuracy, disagreements = evaluate_integer_test_set(
            cnn, images, labels, device
        )
        print(
            f"整数测试集：accuracy={accuracy:.6%} | QAT disagreements={disagreements}/10000 | "
            f"耗时={time.perf_counter() - started:.1f}s"
        )

    image_u8 = images[Config.IMAGE_INDEX : Config.IMAGE_INDEX + 1].to(device)
    label = int(labels[Config.IMAGE_INDEX].item())
    float_prediction = int(cnn(image_u8.to(torch.float32) / 255.0).argmax(1).item())
    exact_logits, exact_trace = integer_forward_trace(cnn, image_u8)
    integer_prediction = int(exact_logits.argmax(1).item())
    estimated_lane_ops = estimate_neural_lane_operations(cnn, exact_trace)
    print(
        f"选中样本 index={Config.IMAGE_INDEX} | label={label} | "
        f"QAT={float_prediction} | integer={integer_prediction}"
    )
    print(f"预计完整指令级执行约 {estimated_lane_ops:,} Neural CPU lane-op。")
    print(
        "数值执行模式：CSD 常量乘法 + Neural CPU 累加/量化/池化/argmax；"
        "外部仅负责存储、寻址与 lane 调度。"
    )
    if float_prediction != integer_prediction:
        raise RuntimeError("QAT 与精确整数核在目标样本上不一致。")
    if not Config.RUN_NEURAL:
        return

    if device.type != "cuda":
        raise RuntimeError("完整 Neural CNN 指令级推理需要 CUDA GPU。")

    cpu, cpu_checkpoint = load_cpu(cpu_path, device)
    cpu_config = cpu_checkpoint.get("config", {})
    print(
        f"Neural CPU checkpoint：step={cpu_checkpoint.get('step')} | "
        f"51 -> {cpu_config.get('HIDDEN_SIZE', 1024)} x "
        f"{int(cpu_config.get('HIDDEN_LAYERS', 2)) + 1} -> 39"
    )
    machine = NeuralByteMachine(cpu, device)
    backend = NeuralFixedPointBackend(machine)
    executor = NeuralCNNExecutor(backend, cnn, exact_trace)
    started = time.perf_counter()
    status = "ok"
    neural_prediction = -1
    neural_logits = torch.empty(0)
    error_message = None
    try:
        neural_prediction, neural_logits = executor.run(image_u8)
    except NeuralExecutionError as error:
        status = "neural_error"
        error_message = str(error)
        print(f"\n首次 Neural CPU 错误：{error}")
    elapsed = time.perf_counter() - started
    summary = {
        "status": status,
        "image_index": Config.IMAGE_INDEX,
        "label": label,
        "qat_prediction": float_prediction,
        "integer_prediction": integer_prediction,
        "neural_prediction": neural_prediction,
        "neural_logits": neural_logits.detach().cpu().tolist(),
        "elapsed_seconds": elapsed,
        "estimated_lane_operations": estimated_lane_ops,
        "numeric_execution": "neural_cpu_csd_full",
        "constant_multiplication": "NAF/CSD shift-add; no multiplication LUT",
        "external_controller_scope": [
            "tensor_storage",
            "address_generation",
            "convolution_window_unfold",
            "lane_scheduling",
            "predicated_writeback",
        ],
        "lane_operations_per_second": machine.audit.neural_lane_operations / max(elapsed, 1e-9),
        "audit": asdict(machine.audit),
        "error": error_message,
        "cnn_model": str(cnn_path),
        "cpu_model": str(cpu_path),
    }
    summary_path = Config.RESULT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Neural CPU MNIST 推理汇总 ===")
    print(
        f"status={status} | label={label} | QAT={float_prediction} | "
        f"integer={integer_prediction} | neural={neural_prediction}"
    )
    print(
        f"neural_calls={machine.audit.neural_calls:,} | "
        f"lane_ops={machine.audit.neural_lane_operations:,} | "
        f"result_errors={machine.audit.result_bit_errors:,} | "
        f"flag_errors={machine.audit.flag_bit_errors:,} | "
        f"control_errors={machine.audit.control_bit_errors:,}"
    )
    print(
        f"耗时={elapsed:.1f}s | lane_ops/s={summary['lane_operations_per_second']:,.0f}"
    )
    print(f"汇总：{summary_path}")
    if status != "ok":
        raise NeuralExecutionError(error_message or "Neural CPU 推理失败。")
    if neural_prediction != integer_prediction:
        raise RuntimeError("Neural CPU 最终分类与精确整数 CNN 不一致。")


if __name__ == "__main__":
    main()

# %%
