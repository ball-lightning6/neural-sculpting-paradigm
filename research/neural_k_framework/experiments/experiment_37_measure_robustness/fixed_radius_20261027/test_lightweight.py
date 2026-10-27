"""独立纯 CPU 单元检查；禁止进入任何 SMC 层、训练或 CUDA 调用。"""

import contextlib
import io
import math
import sys
import unittest
from unittest import mock

import numpy as np
import torch

import analyze_published as analysis
from runtime import HERE, checked_output, fixed_config, load_fixed_core, read_json, verify_bundle
from run_natural_measures import load_historical_core

m = load_fixed_core()


class LightweightTests(unittest.TestCase):
    def setUp(self):
        for name in ("run", "smc_level", "initial_state"):
            self.enterContext(mock.patch.object(m, name, side_effect=AssertionError("禁止运行 SMC")))
        for name in ("is_available", "_lazy_init", "set_device", "synchronize", "get_rng_state_all", "set_rng_state_all"):
            self.enterContext(mock.patch.object(torch.cuda, name, side_effect=AssertionError("禁止调用 CUDA")))

    def test_frozen_files_and_24_pool_science(self):
        self.assertGreaterEqual(verify_bundle(), 10)
        cfg = fixed_config()
        expected = read_json(HERE / "fixed_radius_config.json")
        for key, value in expected.items():
            self.assertEqual(cfg[key], value)
        jobs = m.make_jobs(cfg)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({job["seed"] for job in jobs}), 24)
        self.assertEqual({job["dimension"] for job in jobs}, {97})
        self.assertEqual(m.PROTOCOL, "fixed_radius_angular_smc_independent_pilot_v2_cpu_kthvalue")
        for key in set(expected) - {"targets", "radii"}:
            self.assertEqual(cfg[key], m.CONFIG[key])

    def test_kthvalue_all_orders_dtype_and_rng(self):
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        torch.use_deterministic_algorithms(True)
        before = torch.get_rng_state().clone()
        for dtype in (torch.float32, torch.float64):
            values = torch.tensor([0.7, 1e-30, 0.0, 9.0, 0.7, 1.0, 0.3, 2.0], dtype=dtype)
            values = values.repeat(3)[::2].requires_grad_(True)
            ordered = values.detach().sort().values
            for order in range(1, len(values) + 1):
                actual = m.cpu_pilot_kthvalue(values, order)
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(actual.dtype, dtype)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(torch.equal(actual, ordered[order - 1]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_detach_cpu_precedes_kthvalue(self):
        events = []
        values = torch.tensor([0.9, 0.2, 0.7], dtype=torch.float64)
        real_kthvalue = torch.kthvalue

        class Probe:
            def detach(self):
                events.append("detach")
                return self

            def cpu(self):
                events.append("cpu")
                return values

        def guarded(tensor, order):
            self.assertEqual(tensor.device.type, "cpu")
            events.append("kthvalue")
            return real_kthvalue(tensor, order)

        with mock.patch.object(torch, "kthvalue", side_effect=guarded):
            self.assertEqual(float(m.cpu_pilot_kthvalue(Probe(), 2)), 0.7)
        self.assertEqual(events, ["detach", "cpu", "kthvalue"])

    def test_labels_and_independent_numpy_forward(self):
        x = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(float)
        np.testing.assert_array_equal(m.bit_inputs().numpy(), x)
        labels = {"projection": x[:, 0], "majority3": (x[:, :3].sum(1) >= 2),
                  "parity2": ((x[:, 0] + x[:, 1]) % 2)}
        for target, expected in labels.items():
            np.testing.assert_array_equal(m.target_labels(target).numpy(), expected)
        theta = np.sin(np.arange(3 * 97).reshape(3, 97) / 17)
        raw = theta.copy()
        raw[:, :80] *= 0.5
        raw[:, 80:] *= 0.25
        hidden = np.tanh(np.einsum("xi,mhi->mxh", x, raw[:, :64].reshape(-1, 16, 4)) + raw[:, None, 64:80])
        logits = np.einsum("mxh,mh->mx", hidden, raw[:, 80:96]) + raw[:, 96, None]
        actual = m.forward_logits(torch.tensor(theta), torch.tensor(x), [16])
        np.testing.assert_allclose(actual.numpy(), logits, rtol=1e-12, atol=1e-12)

    def test_sphere_proposal_preserves_radius(self):
        rng = torch.Generator(device="cpu").manual_seed(37)
        unit = m.sample_sphere(16, 97, "cpu", torch.float64, rng)
        proposed = m.sphere_proposal(unit, 0.1 / math.sqrt(97), rng)
        for radius in (20.0, 40.0):
            self.assertLess(m.radius_diagnostics(proposed, radius)["max_abs_relative_radius_error"], 1e-14)

    def test_probability_pooling_not_mean_K(self):
        self.assertAlmostEqual(analysis.pooled_K([1.0, 3.0]), -math.log2(0.3125))
        self.assertNotEqual(analysis.pooled_K([1.0, 3.0]), 2.0)
        self.assertAlmostEqual(analysis.pooled_K([1200.0, 1202.0]), 1200-math.log2(0.625))
        self.assertIsNone(analysis.pooled_K([1.0, None]))
        self.assertIsNone(m.mean_log2_mass([-1.0, None]))

    def test_missing_nodes_not_zero_and_failure_categories(self):
        state = {"nodes": [{"status": "pending", "K_angular_bits": 10.0}],
                 "target_index": 0, "threshold": 0.01, "log2_mass": -20.0}
        missing = m.mark_unreached(state, "stalled_levels_budget")
        self.assertIsNone(missing["nodes"][0]["log2_angular_mass"])
        self.assertIsNone(missing["nodes"][0]["K_angular_bits"])
        summary = m.failure_summary({"job_summaries": [missing, {"phase": "failed", "reason": "internal_error"}]})
        self.assertEqual(summary["jobs_scientific_unreached"], 1)
        self.assertEqual(summary["jobs_errors"], 1)
        self.assertEqual(state["nodes"][0]["K_angular_bits"], 10.0)

    def test_two_precision_coordinates_are_distinct(self):
        margin_delta = -math.log(math.expm1(0.001)) + math.log(math.expm1(0.01))
        log2_delta = math.log2(0.01 / 0.001)
        self.assertGreater(abs(margin_delta-log2_delta), 1)
        result, _, _ = analysis.check_data()
        self.assertEqual((result["completed_pools"], result["stalled_pools"]), (16, 8))
        r40 = [r for r in result["contrasts"] if r["radius"] == 40]
        self.assertEqual([f'{r["delta_K"]:.2f}' for r in r40], ["14.66", "27.81", "69.04"])
        self.assertEqual([f'{r["c_log2_precision"]:.2f}' for r in r40], ["2.99", "5.67", "14.07"])

    def test_historical_core_and_portable_panel(self):
        core = load_historical_core()
        self.assertEqual(core.core.legacy.parameter_count(), 97)
        self.assertAlmostEqual(core.scale_for(72, 97), 72/math.sqrt(99))
        panel = read_json(HERE / "historical_20260913/panel.json")
        self.assertEqual(len(panel["function_ids"]), 20)
        self.assertEqual(panel["function_ids"][:4], [0, 255, 4080, 27030])
        self.assertEqual(panel["student_df"], 8)

    def test_default_entrypoints_do_not_sample_or_write(self):
        import run_fixed_radius
        import run_natural_measures
        historical = load_historical_core()
        before = set(HERE.rglob("*"))
        with mock.patch.object(historical, "main", side_effect=AssertionError("禁止运行旧面板")):
            for module in (run_fixed_radius, run_natural_measures):
                with mock.patch.object(sys, "argv", [module.__file__]), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
        self.assertEqual(set(HERE.rglob("*")), before)

    def test_output_cannot_overwrite_sources(self):
        for output in (".", "..", "frozen", "data", "runs"):
            with self.assertRaises(ValueError):
                checked_output(output)
        self.assertEqual(checked_output("runs/new").parent.name, "runs")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
