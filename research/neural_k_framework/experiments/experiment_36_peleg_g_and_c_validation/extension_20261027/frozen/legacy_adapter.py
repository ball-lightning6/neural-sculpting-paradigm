"""旧模型的只读适配；源代码副本保持原 hash，不能作为运行入口。"""

from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import torch
import torch.nn as nn

import common as c


EXPECTED_SOURCE = "fa25ff63ff04ffa269f700db83d02e37d223b277485d45657f52c8fd5c8829af"
REFERENCE_SOURCE = Path(__file__).with_name("legacy_static_reference.py").resolve()
if c.file_hash(REFERENCE_SOURCE) != EXPECTED_SOURCE:
    raise RuntimeError("旧模型参考源码 hash 不匹配，拒绝导入")

import legacy_static_reference as old

if Path(old.__file__).resolve() != REFERENCE_SOURCE:
    raise RuntimeError("旧模型参考模块被其它搜索路径覆盖")

CASES = [
    ("cnn_w4__anchor_0_7_n16", {"kind": "cnn", "scale": 4}),
    ("cnn_w4__partition0_seed1_n32", {"kind": "cnn", "scale": 4}),
    ("cnn_w4__partition2_seed2_n32", {"kind": "cnn", "scale": 4}),
    ("mlp_256_128__partition0_seed0_n32_flip_quarter", {"kind": "mlp", "hidden": [256, 128]}),
]


class LegacyModel:
    def __init__(self, spec):
        if c.file_hash(Path(old.__file__)) != EXPECTED_SOURCE:
            raise RuntimeError("旧模型参考源码 hash 不匹配")
        self.old = old.build_model(spec)
        self.spec = {"kind": "legacy_uniform_box", "network": spec, "source": EXPECTED_SOURCE}
        self.size = self.old["parameters"]
        self.bounds = torch.full((self.size,), .2, dtype=torch.float32)
        self.bounds[self.old["biases"]] = 0
        self.blocks = [(f"weight{i}", layer[0]) for i, layer in enumerate(self.old["layers"])] + [("all", slice(0, self.size))]

    def logits(self, theta, x):
        z = old.forward(theta, x, self.old)
        # 与旧 energies 完全相同：先转换float64，再作两类logit差。
        return z[..., 1].double() - z[..., 0].double()

    def reference(self, row):
        layers = []
        kind = self.old["spec"]["kind"]
        if kind == "mlp":
            layers.append(nn.Flatten())
        for i, (ws, bs, co, ci, k) in enumerate(self.old["layers"]):
            if kind == "cnn" and i < 2:
                layer = nn.Conv2d(ci, co, k, dtype=row.dtype, device=row.device)
            else:
                if kind == "cnn" and i == 2:
                    layers.append(nn.Flatten())
                layer = nn.Linear(ci, co, dtype=row.dtype, device=row.device)
            with torch.no_grad():
                layer.weight.copy_(row[ws].reshape_as(layer.weight))
                layer.bias.copy_(row[bs])
            layers.append(layer)
            if i + 1 < len(self.old["layers"]):
                layers.append(nn.ReLU())
                if kind == "cnn" and i < 2:
                    layers.append(nn.MaxPool2d(2))
        class Gap(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                out = self.net(x)
                return out[:, 1].double() - out[:, 0].double()
        return Gap()


def _relative_path(root, relative):
    """兼容旧 Linux/Windows 清单，但不接受绝对路径、逃逸或链接越界。"""
    if not isinstance(relative, str):
        raise RuntimeError("冻结清单路径必须是字符串")
    posix = PurePosixPath(relative.replace("\\", "/"))
    if posix.is_absolute() or PureWindowsPath(relative).drive or ".." in posix.parts or not posix.parts:
        raise RuntimeError("冻结清单含越界路径：" + relative)
    path = (Path(root) / Path(*posix.parts)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise RuntimeError("冻结清单含越界链接：" + relative)
    return posix.as_posix(), path


def _audit_original(root, name):
    root = Path(root).resolve()
    specs = dict(CASES)
    if name not in specs:
        raise ValueError("未知原始条件：" + str(name))
    protocol = c.read_json(root / "protocol.json")
    if protocol.get("source_sha256") != EXPECTED_SOURCE:
        raise RuntimeError("原实验不是已复核的源码版本，不自动套用复核协议")
    cfg = protocol.get("config", {})
    if protocol.get("scientific_sha256") != old.scientific_hash(cfg):
        raise RuntimeError("原实验科学配置 hash 不匹配")
    if cfg.get("prior_scale") != .2 or cfg.get("fixed_zero_bias") is not True:
        raise RuntimeError("原实验不是 U[-0.2,0.2]、bias 固定零的测度")
    if cfg.get("nodes") != old.CONFIG["nodes"]:
        raise RuntimeError("原实验 BCE 节点不匹配")
    network_name, case_name = name.split("__", 1)
    network = next((s for s in cfg.get("networks", []) if s.get("name") == network_name), None)
    if network is None or {k: v for k, v in network.items() if k != "name"} != specs[name]:
        raise RuntimeError("原实验网络与条件名不一致")
    _, folder = _relative_path(root, name)
    frozen = c.read_json(folder / "forecast_frozen.json")
    if any(frozen.get(k) != protocol.get(k) for k in ("source_sha256", "scientific_sha256")):
        raise RuntimeError("原冻结记录与协议源码/科学配置 hash 不一致")
    entries = frozen.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("原冻结清单为空或非法")
    checked = {}
    for relative, expected in entries.items():
        canonical, path = _relative_path(folder, relative)
        if canonical in checked:
            raise RuntimeError("原冻结清单含重复路径：" + relative)
        if not path.is_file() or c.file_hash(path) != expected:
            raise RuntimeError("原冻结预测 hash 不匹配：" + relative)
        checked[canonical] = expected
    replicas = cfg.get("replicas")
    if type(replicas) is not int or replicas < 1:
        raise RuntimeError("原实验副本数非法")
    required = {"dataset.npz", "dataset_manifest.json", "gc.json"}
    if (folder / "gc_forecast.npz").exists():
        required.add("gc_forecast.npz")
    for replica in range(replicas):
        subdir = folder / f"replica_{replica}"
        required.add(f"replica_{replica}/final.json")
        required.update(p.relative_to(folder).as_posix() for p in subdir.glob("*.npz"))
        final_path = subdir / "final.json"
        if not final_path.is_file():
            raise RuntimeError("原静态池尚无终态记录")
        final = c.read_json(final_path)
        if final.get("status") not in {"target_reached", "unresolved_budget", "no_survivors"}:
            raise RuntimeError("原静态池尚未完成或明确停止")
        required.update(f"replica_{replica}/{node['label']}.npz" for node in final.get("nodes", []))
        if final["status"] == "target_reached":
            required.update(f"replica_{replica}/{node}.npz" for node in ["hard_fit", *map(old.node_label, cfg["nodes"])])
    if not required.issubset(checked):
        raise RuntimeError("原冻结清单缺少必要文件：" + ", ".join(sorted(required - checked.keys())))
    metadata = c.read_json(folder / "dataset_manifest.json")
    expected_case = next((case for case in old.make_cases(cfg) if case["name"] == case_name), None)
    if expected_case is None or metadata.get("case") != expected_case:
        raise RuntimeError("原数据条件与协议不一致")
    return folder, protocol, metadata


def original_case(original_root, name):
    folder, _, metadata = _audit_original(original_root, name)
    data = c.read_npz(folder / "dataset.npz")
    if metadata.get("arrays") != {k: c.array_hash(v) for k, v in data.items()}:
        raise RuntimeError("原始数据 hash 不匹配")
    if not {"x", "y", "train_ids", "probe_x", "probe_y", "probe_ids"}.issubset(data):
        raise RuntimeError("原始数据缺少必要数组")
    for prefix, ids_key in (("", "train_ids"), ("probe_", "probe_ids")):
        x, y, ids = data[prefix + "x"], data[prefix + "y"], data[ids_key]
        if (x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (1, 28, 28)
                or y.dtype != np.int64 or y.ndim != 1 or ids.ndim != 1
                or len(x) != len(y) or len(ids) != len(y) or not len(y)
                or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()):
            raise RuntimeError("原始数据形状、dtype 或标签非法")
    if len(data["y"]) != metadata["case"]["n"]:
        raise RuntimeError("原训练样本数与条件不一致")
    return folder, data, metadata


def _reference_target(destination, name):
    if Path(name).name != name or "/" in name or "\\" in name:
        raise RuntimeError("引用清单文件名非法")
    if name.startswith("sgd_bce_") and name.endswith(".npz"):
        relative = Path("optimizer") / name.removeprefix("sgd_")
    elif name == "gc_forecast.npz":
        relative = Path("gc") / "forecast.npz"
    elif name in {"sgd.json", "gc.json"}:
        relative = Path(name)
    else:
        raise RuntimeError("未知历史引用文件：" + name)
    target = Path(destination) / relative
    if not target.resolve().is_relative_to(Path(destination).resolve()):
        raise RuntimeError("引用目标含越界链接")
    return target


def verify_reference(destination, expected_hash=None):
    """分析可单独重跑，但不得绕过首次导入的 SGD/GC 文件身份。"""
    destination = Path(destination).resolve()
    path = destination / "reference_identity.json"
    if not path.resolve().is_relative_to(destination):
        raise RuntimeError("引用身份文件含越界链接")
    if not path.is_file():
        raise RuntimeError("历史引用身份文件缺失")
    if expected_hash is not None and c.file_hash(path) != expected_hash:
        raise RuntimeError("冻结的历史引用身份已改变")
    identity = c.read_json(path)
    expected = set()
    for name, digest in identity["source_files"].items():
        target = _reference_target(destination, name)
        expected.add(target)
        if not target.is_file() or c.file_hash(target) != digest:
            raise RuntimeError("历史引用 hash 不匹配：" + name)
    present = {p for p in (destination / "sgd.json", destination / "gc.json") if p.exists()}
    present.update((destination / "optimizer").glob("*.npz"))
    present.update((destination / "gc").glob("*.npz"))
    if present != expected:
        raise RuntimeError("历史引用含未登记文件或缺失文件")
    return identity


def import_reference(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source.parent) or source.parent.is_relative_to(destination):
        raise RuntimeError("引用目标不得覆盖、包含或位于原结果目录内")
    _audit_original(source.parent, source.name)
    files = [source / "sgd.json", source / "gc.json"]
    files += sorted(source.glob("sgd_bce_*.npz")) + sorted(source.glob("gc_forecast.npz"))
    # SGD 在原静态冻结后才产生，单独固定首次导入时的文件集合与逐文件 hash。
    hashes = {p.name: c.file_hash(_relative_path(source, p.name)[1]) for p in files if p.is_file()}
    identity = {"source_files": hashes, "source_audit": {
        "protocol": c.file_hash(source.parent / "protocol.json"),
        "forecast_frozen": c.file_hash(source / "forecast_frozen.json"),
        "dataset_manifest": c.file_hash(source / "dataset_manifest.json"),
    }}
    planned = {}
    for path in files:
        if path.name not in hashes:
            continue
        planned[path] = _reference_target(destination, path.name)
    identity_path = destination / "reference_identity.json"
    if not identity_path.resolve().is_relative_to(destination):
        raise RuntimeError("引用身份文件含越界链接")
    c.guard_json(identity_path, identity)
    for path, target in planned.items():
        if target.exists():
            if c.file_hash(target) != hashes[path.name]:
                raise RuntimeError("已复制的原始对照改变")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            if not temporary.resolve().is_relative_to(destination):
                raise RuntimeError("引用临时文件含越界链接")
            temporary.write_bytes(path.read_bytes())
            if c.file_hash(temporary) != hashes[path.name]:
                raise RuntimeError("复制期间原始对照改变")
            temporary.replace(target)
    verify_reference(destination)
