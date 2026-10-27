"""py_compile、显式导入和纯 CPU 单元检查；不运行 SMC。"""

import importlib
import io
import json
import platform
import py_compile
from pathlib import Path
import sys
import unittest

from runtime import HERE, verify_bundle


def main():
    output = HERE / "validation"
    output.mkdir(exist_ok=True)
    scripts = sorted([*HERE.glob("*.py"), *HERE.glob("frozen/*.py"), *HERE.glob("historical_20260913/source/*.py")])
    compiled = []
    for index, path in enumerate(scripts):
        py_compile.compile(str(path), cfile=str(output / f"compile_{index}.pyc"), doraise=True)
        compiled.append(path.relative_to(HERE).as_posix())
    dependencies = {}
    for name in ("numpy", "torch", "matplotlib"):
        module = importlib.import_module(name)
        dependencies[name] = str(module.__version__)
    from run_natural_measures import load_historical_core
    historical = load_historical_core()
    import torch
    torch.set_num_threads(1)
    import test_lightweight
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_lightweight))
    report = {"date": "2026-10-27", "python": platform.python_version(),
              "dependencies": dependencies, "compiled_files": compiled,
              "source_files_verified": verify_bundle(),
              "historical_imports": [historical.__name__, historical.core.__name__, historical.core.legacy.__name__],
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "sampling_started": False, "training_started": False, "gpu_exercised": False,
              "test_output": stream.getvalue()}
    (output / "VALIDATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for index in range(len(scripts)):
        (output / f"compile_{index}.pyc").unlink()
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
