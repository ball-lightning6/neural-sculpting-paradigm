"""后续实验的原子记录、身份检查与合作式停止；不导入旧运行目录。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch


STOP = False


def request_stop(signum=None, frame=None):
    global STOP
    STOP = True


def install_signals():
    global STOP
    STOP = False
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)


def expired(deadline):
    return STOP or time.monotonic() >= deadline


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def seed(*parts):
    return int(digest(parts)[:15], 16)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def array_hash(value):
    value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()


def atomic_replace(source, target):
    # Windows索引器/杀毒程序可短暂占用目标；仍坚持原子替换，不退回删除后写入。
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(.02 * (2 ** attempt))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    atomic_replace(tmp, path)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(to_cpu(value), tmp)
    atomic_replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k].copy() for k in arrays.files}


def guard_json(path, expected):
    path = Path(path)
    if path.exists():
        if read_json(path) != clean(expected):
            raise RuntimeError(f"身份/科学配置不一致，请另建输出目录：{path}")
    else:
        atomic_json(path, expected)


def code_hashes(root):
    root = Path(root)
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))
            if not p.name.startswith(("notebook_", "test_", "build_", "validate_"))}


def environment(device, allow_cpu=False):
    device = torch.device(device)
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("正式运行必须使用 CUDA；CPU 仅供显式短测试。")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不自动退回 CPU。")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cpu":
        torch.set_num_threads(1)
    return {
        "device_type": device.type, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "python": platform.python_version(), "dtype": "float32 network / float64 loss", "tf32": False,
    }


def record_runtime(root, env, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **env, "operational_config": clean(config)}
    with (root / "runtime.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(root, message):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z") + " | " + message
    print(line, flush=True)
    with (Path(root) / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


class RunLock:
    """阻止并行启动；异常 kill 后的陈旧锁需人工确认，再移除。"""

    def __init__(self, root):
        self.path = Path(root) / "ACTIVE_RUN.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "time": time.time(), "host": platform.node()}, stream)
        except FileExistsError as error:
            raise RuntimeError(f"已有运行锁；先确认没有旧进程，勿重复启动：{self.path}") from error
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def freeze(root, paths, extra=None):
    root = Path(root)
    paths = [Path(p) for p in paths]
    if not paths or any(not p.resolve().is_relative_to(root.resolve()) for p in paths):
        raise RuntimeError("冻结清单须非空，且文件须在条件目录内")
    record = {"files": {str(p.relative_to(root)): file_hash(p) for p in sorted(paths)},
              "extra": clean(extra or {}), "policy": "先冻结静态预测，随后才运行优化器与测试标签评分。"}
    guard_json(root / "forecast_frozen.json", record)
    guard_json(root / "forecast_frozen_identity.json", {
        "manifest_sha256": file_hash(root / "forecast_frozen.json"), "files": sorted(record["files"])})
    return record


def verify_freeze(root):
    root = Path(root)
    record = read_json(root / "forecast_frozen.json")
    if not isinstance(record, dict) or not isinstance(record.get("files"), dict) or not record["files"]:
        raise RuntimeError("冻结清单为空或结构非法")
    anchor = root / "forecast_frozen_identity.json"
    if not anchor.exists() or read_json(anchor) != {
        "manifest_sha256": file_hash(root / "forecast_frozen.json"), "files": sorted(record["files"])}:
        raise RuntimeError("冻结清单身份不匹配或尚未完整提交")
    for name, expected in record["files"].items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError("冻结文件条目非法")
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise RuntimeError("冻结记录已改变：" + name)
    return record


def settings(defaults, root):
    path = Path(root) / "launch_settings.json"
    changes = read_json(path) if path.exists() else {}
    unknown = set(changes) - set(defaults)
    if unknown:
        raise ValueError("未知顶部配置：" + ", ".join(sorted(unknown)))
    return {**defaults, **changes}
