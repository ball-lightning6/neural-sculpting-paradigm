"""追加训练集大小维度；复用原 CV 科学核心，结果必须另存。"""

import signal
from pathlib import Path

import profiles as p
import shared_support as s
import static_reference as original


def configuration(root, original_dir, device, hours):
    root, original_dir = Path(root).resolve(), Path(original_dir).resolve()
    output = root / "results_cv_extension"
    if (root.is_relative_to(original_dir) or original_dir.is_relative_to(root)
            or not output.resolve().is_relative_to(root)):
        raise RuntimeError("追加 CV 必须写新目录，不能覆盖或经链接写入旧结果。")
    if original.source_hash() != p.STATIC_SOURCE_SHA256:
        raise RuntimeError("CV 参考核心不是原字节版本")
    return {**original.CONFIG, "device": device, "total_hours": hours, "output_dir": str(output),
            "mnist_root": str(original_dir / "data"), "networks": p.CV_NETWORKS,
            "include_anchor": False, "partitions": 3, "datasets_per_partition": 2,
            "sample_sizes": [8, 64], "include_label_controls": False,
            "replicas": 4, "particles": 1024, "sweeps": 32, "refresh_sweeps": 96}


def run(root, original_dir, device, hours):
    cfg = configuration(root, original_dir, device, hours)
    cases = original.make_cases(cfg)
    if len(cases) * len(cfg["networks"]) != 24:
        raise RuntimeError("追加 CV 必须恰好为24条件")
    s.guard_json(Path(cfg["output_dir"]) / "CONTROLLER_IDENTITY.json", {
        "wrapper_sha256": s.file_hash(__file__), "core_sha256": original.source_hash(),
        "scientific_sha256": original.scientific_hash(cfg), "matrix": p.protocol("cv")["cv_extension"]})
    stop_requested = False
    old_loader = original.load_mnist
    handlers = {}

    def on_stop(*args):
        nonlocal stop_requested
        stop_requested = True
        original.request_stop(*args)

    def guarded_loader(config):
        # 原核心入口会重置 STOP；在其初始化后的必经数据入口重验独立的停止标志。
        if stop_requested:
            raise KeyboardInterrupt
        data = old_loader(config)
        if stop_requested:
            original.request_stop()
        return data

    try:
        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                handlers[number] = signal.signal(number, on_stop)
        original.load_mnist = guarded_loader
        print("CV 追加：n=8/64，3个固定随机数字分组，2份训练数据，CNN W4和MLP。", flush=True)
        print("静态先验、优化器和数据种子沿用原定义；新增组用32/96混合预算，不覆盖原n=32。", flush=True)
        original.run(cfg)
    finally:
        original.load_mnist = old_loader
        for number, handler in handlers.items():
            signal.signal(number, handler)
    return s.read_json(Path(cfg["output_dir"]) / "STATUS.json")
