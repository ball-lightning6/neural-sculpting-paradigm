"""轻量 CPU 核验：不训练、不执行任何 SMC 更新、不完整枚举。"""

import importlib
import itertools
import json
from pathlib import Path
import platform
import py_compile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "alignment"))
import numpy as np
import torch
import config
from data import random_table, verify_data
from review import main as review
from minimax.test_analyze import TestEnumeration


class PublicChecks(unittest.TestCase):
    def test_random_tables(self):
        self.assertEqual(verify_data()["prefixes"], 32)
        from loss.experiment import make_data
        import frozen_memory as m
        for dataset in range(4):
            for actual in (make_data(dataset), m.make_data(dataset, m.CONFIG)):
                for a, b in zip(actual, random_table(dataset)):
                    np.testing.assert_array_equal(a, b)

    def test_forward_and_range(self):
        import frozen_memory as m
        from loss.experiment import numpy_forward
        from alignment.alignment_bridge import neural_scores
        for hidden in ((32,), (32,32), (32,32,32)):
            _, layers, size = m.parameter_blocks(hidden)
            theta = np.random.default_rng(5).normal(size=(3, size))
            x, labels, order = random_table(0)
            x, labels = x[order[:8]].astype(float), labels[order[:8]].astype(float)
            raw = theta.copy()
            for start, _, stop, fan_in, _ in layers:
                raw[:, start:stop] /= np.sqrt(fan_in)
            expected = numpy_forward(raw, x, layers)
            tt, xx, yy = [torch.tensor(v, dtype=torch.float64) for v in (theta, x, labels)]
            np.testing.assert_allclose(m.forward_logits(tt, xx, layers).numpy(), expected, rtol=1e-12, atol=1e-12)
            bce, spread = neural_scores(tt, xx, yy, layers)
            margins = expected * (2*labels-1)
            np.testing.assert_allclose(bce.numpy(), np.logaddexp(0, -margins).mean(1), atol=1e-12)
            np.testing.assert_allclose(spread.numpy(), np.ptp(margins, axis=1), atol=1e-12)

    def test_agreement(self):
        from agreement.analyze_agreement import metrics
        pred = np.array([[0,1,1],[0,1,0],[1,1,0]], dtype=bool)
        expected = np.mean([np.mean(a == b) for a,b in itertools.combinations(pred, 2)])
        self.assertAlmostEqual(metrics(pred, np.array([0,1,0]))["agreement"], expected)

    def test_structure(self):
        from structure.audit import minimum_tree_leaves, linear_fit
        labels = np.zeros(256, dtype=np.uint8)
        labels[[1,2]] = 1
        self.assertEqual(minimum_tree_leaves(labels, np.arange(4)), 4)
        self.assertFalse(linear_fit(labels, np.arange(4))["separable"])

    def test_loss_statistics(self):
        from loss.analyze import summarize_losses
        stats = summarize_losses(np.ones((3, 8)))
        self.assertEqual(stats["max_to_mean_median"], 1.)
        self.assertIsNone(stats["pairwise_sample_rank_spearman_mean"])

    def test_calibration_and_configs(self):
        from smc.run_calibration import rule_data, configure
        from smc.protocols import configuration
        self.assertEqual(configuration("base_two")["POOLS"], 4)
        self.assertEqual(len(configuration("base_two")["COUNTS"]), 8)
        for rule in ("parity1", "majority3", "mux3", "parity2", "parity4"):
            x, y, order = rule_data(rule)
            self.assertEqual(len(y), 256)
            self.assertEqual(configure(rule, ROOT/"_runs"/"unused", "cuda:0", 1)["PARTICLES"], 4096)
        x, y, _ = rule_data("parity4")
        np.testing.assert_array_equal(y, x[:, :4].sum(1).astype(int) % 2)

    def test_guards(self):
        import run
        for task in ("agreement", "loss", "fit", "base_one", "enhanced", "alignment", "calibration", "minimax"):
            with patch.object(config, "TASK", task):
                with self.assertRaises(SystemExit):
                    run.main()

    def test_external_dependency_is_explicit(self):
        from minimax.analyze import input_directory
        with self.assertRaises(RuntimeError):
            input_directory("missing")

    def test_editable_configuration_is_reported(self):
        from review import verify_sources
        original = Path.read_bytes
        def simulated_edit(path):
            value = original(path)
            return value + b"\n" if path == ROOT / "config.py" else value
        with patch.object(Path, "read_bytes", simulated_edit):
            self.assertEqual(len(verify_sources()["edited_configuration"]), 1)


def main():
    cache = ROOT / "_runs" / "compile"
    cache.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in ROOT.rglob("*.py") if "_runs" not in p.parts)
    for p in paths:
        py_compile.compile(str(p), cfile=str(cache / (p.relative_to(ROOT).as_posix().replace("/", "_") + "c")), doraise=True)
    modules = (
        "frozen_memory", "memory_geometry", "smc.memory_followup", "smc.protocols", "smc.run_calibration",
        "alignment.alignment_bridge", "alignment.run_alignment", "agreement.train_agreement",
        "agreement.analyze_agreement", "loss.experiment", "loss.analyze", "structure.audit_data",
        "structure.audit", "minimax.analyze", "run", "plot_evidence", "review", "data", "config",
    )
    for name in modules:
        importlib.import_module(name)
    review()
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(PublicChecks),
        unittest.defaultTestLoader.loadTestsFromTestCase(TestEnumeration),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    from check_report_pair import check
    bilingual = check()
    output = {
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "compiled_modules": len(paths), "imported_modules": len(modules), "unit_tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "bilingual": bilingual,
        "training_started": False, "smc_started": False, "full_minimax_started": False,
        "raw_snapshot_audit_repeated": False,
    }
    (ROOT / "TEST_RESULTS.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

