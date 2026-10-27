"""加载冻结科学函数；不执行旧机器路径、归档读取或队列入口。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_sources():
    records = read_json(ROOT / "SOURCE_MANIFEST.json")
    for item in records:
        path = (ROOT / item["file"]).resolve()
        if not path.is_relative_to(FROZEN):
            raise RuntimeError("冻结源码路径越界")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("冻结源码校验失败：" + item["file"])
    return {item["file"]: item["sha256"] for item in records}


def _full(name):
    path = FROZEN / (name + ".py")
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("同名模块来自公开包以外：" + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _selected(filename, names, bindings=None, constants=()):
    """保留 AST 原节点，不重写函数体；只跳过历史启动器和外部数据加载。"""
    path = FROZEN / filename
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            keys = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if keys & set(constants):
                selected.append(node)
                found.update(keys)
    if found != set(names) | set(constants):
        raise RuntimeError("冻结定义清单不匹配：" + filename)
    module = ModuleType("e36_" + path.stem)
    module.__dict__.update(
        __file__=str(path), np=np, torch=torch, F=F, math=math, time=time,
        hashlib=hashlib, json=json, os=os, sys=sys, Path=Path,
    )
    module.__dict__.update(bindings or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def load_backend(output=None):
    sources = verify_sources()
    core = _full("legacy_static_reference")
    common = _full("common")
    adapter = _full("legacy_adapter")
    engine = _full("engine")
    local_engine = _full("local_engine")
    output = Path(output or ROOT / "runs").resolve()
    if not output.is_relative_to(ROOT):
        raise RuntimeError("公开入口只写入 extension_20261027 内")
    shared = SimpleNamespace(
        ROOT=output, read=read_json, dump=common.atomic_json,
        replace=common.atomic_replace, object_hash=common.digest,
    )
    stp = _selected("wide_stp_control.py", {"stp_step"})
    broad = _selected(
        "broad_worker.py", {"save", "record", "run_group"},
        {"s": shared, "load_core": lambda: core, "stp_step": stp.stp_step},
    )
    bindings = dict(
        ROOT=output, c=common, e=local_engine, old=core,
        LegacyModel=adapter.LegacyModel,
        stop_requested=lambda: common.STOP or (output / "STOP").exists(),
        dump=common.atomic_json, now=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    functions = {"collect", "save_optimizer", "optimize", "sample"}
    wide = _selected("wide_worker.py", functions, bindings)
    cloud = _selected("cloud_worker.py", functions, {**bindings, "e": engine})
    adam = _selected(
        "adam_probe.py",
        {"node_name", "atomic_json", "observe", "energy", "interpolate", "serialize", "run_model"},
        {"ROOT": output, "c": common, "old": core}, constants={"CONFIG"},
    )
    recheck = _selected("recheck.py", {"_sampler"}, {"e": engine}, constants={"TIERS"})
    anchor_reference = _full("anchor_reference")
    anchor = _selected(
        "anchor_control.py",
        {"interrupted", "sha", "atomic_json", "atomic_torch", "atomic_npz", "node_name", "log",
         "select_three", "capture_nodes", "cpu_state", "restore", "new_state", "run_stp", "run_sgd",
         "export_predictions"},
        {"HERE": output / "anchor_steps", "REFERENCE": output / "anchor_reference"},
        constants={"CONFIG", "STOP"},
    )
    anchor_budget = _selected("anchor_budget.py", {"group_name"}, constants={"CONFIG"})
    return SimpleNamespace(
        core=core, common=common, adapter=adapter, engine=engine, local_engine=local_engine,
        broad=broad, stp=stp, wide=wide, cloud=cloud, adam=adam, recheck=recheck, sources=sources,
        anchor_reference=anchor_reference, anchor=anchor, anchor_budget=anchor_budget,
    )


def source_identity():
    records = verify_sources()
    for name in ("portable.py", "reproduce.py"):
        records[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return records
