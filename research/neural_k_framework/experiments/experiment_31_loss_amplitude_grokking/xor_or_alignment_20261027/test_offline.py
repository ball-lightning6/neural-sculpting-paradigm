"""发布包轻量核验：只用固定数组，不训练、不抽样、不执行 SMC。"""

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

import run_experiment as exp


ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class OfflineTests(unittest.TestCase):
    def test_frozen_sampler_and_data_hashes(self):
        provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8-sig"))
        for name in ("run_experiment.py", "bridge_diagnostic.py", "verify_bridge.py", "test_experiment.py"):
            source = next(r for r in provenance["source_files"] if r["source"] == f"research/xor_or_alignment_20261024/{name}")
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), source["sha256"])
        for row in provenance["data_files"]:
            self.assertEqual(hashlib.sha256((ROOT / "data" / row["file"]).read_bytes()).hexdigest(), row["sha256"])

    def test_truth_tables_and_independent_forward(self):
        x = exp.inputs("cpu")
        a, b, c, _ = x.long().T
        powers = 1 << torch.arange(16)
        self.assertEqual(int(((a ^ b) * powers).sum()), 0x0FF0)
        self.assertEqual(int((((a ^ b) | c) * powers).sum()), 0xCFFC)
        atoms = [0, (1 << 16) - 1] + [int((column * powers).sum()) for column in x.long().T]
        one_gate = {u ^ ((1 << 16) - 1) for u in atoms}
        for u in atoms:
            for v in atoms:
                one_gate.update((u & v, u | v, u ^ v))
        self.assertNotIn(exp.RULES["f"], atoms)
        self.assertIn(exp.RULES["f"], one_gate)
        self.assertNotIn(exp.RULES["g"], set(atoms) | one_gate)
        self.assertEqual(exp.RULES["f"] | atoms[4], exp.RULES["g"])
        theta = torch.linspace(-2, 2, 3 * 97).reshape(3, 97)
        np.testing.assert_allclose(exp.forward(theta, x).numpy(), exp.numpy_forward(theta.numpy()), atol=2e-6, rtol=2e-6)
        for rule in exp.RULES:
            for kind in exp.KINDS:
                scorer, _, _ = exp.scorer_for(rule, kind, "cpu")
                _, independent = exp.numpy_measure(theta.numpy(), exp.RULES[rule], kind)
                np.testing.assert_allclose(scorer(theta).numpy(), independent["score"], atol=2e-6, rtol=2e-6)

    def test_events_and_nonnegative_loss_budget(self):
        margins = torch.tensor([[1., 9., 1., 9.], [3., 3., 3., 3.]])
        mean, maximum, variance = exp.features(margins, torch.ones(4))
        self.assertTrue(bool((maximum >= mean).all()))
        self.assertTrue(bool((mean + .1 * variance >= mean).all()))
        self.assertEqual(float(variance[1]), 0.)
        self.assertAlmostEqual(float(mean[1]), math.log1p(math.exp(-3)), places=7)
        g = next(r for r in rows("primary_aggregates.csv") if r["rule"] == "g" and r["kind"] == "mean_bce" and float(r["threshold"]) == .001)
        self.assertAlmostEqual((float(g["c0_bce_mean"]) + float(g["c1_bce_mean"])) / 2, float(g["bce_mean"]), places=12)
        self.assertLess(float(g["c0_bce_mean"]), .002)
        self.assertGreater(float(g["c0_bce_mean"]), .001)

    def test_recorded_order_and_observed_grid_node(self):
        expected = {("primary", "mean_bce"): -37.79, ("primary", "aligned"): 48.91,
                    ("confirm", "max_bce"): 29.47, ("confirm", "aligned"): 50.28}
        for (tier, kind), value in expected.items():
            data = rows(f"{tier}_differences.csv")
            endpoint = next(r for r in data if r["kind"] == kind and float(r["threshold"]) == .001)
            self.assertAlmostEqual(float(endpoint["delta_K_median"]), value, delta=.006)
        for tier in ("primary", "confirm"):
            data = rows(f"{tier}_differences.csv")
            for threshold, sign in ((.12, -1), (.08, 1)):
                node = next(r for r in data if r["kind"] == "aligned" and float(r["threshold"]) == threshold)
                self.assertGreater(sign * float(node["delta_K_median"]), 0)
        bridge = json.loads((ROOT / "data/bridge_verification.json").read_text(encoding="utf-8"))
        row = next(r for r in bridge["differences"] if r["threshold"] == .001 and r["kind"] == "aligned")
        self.assertAlmostEqual(row["K_g"] - row["K_f"], row["delta_K"], places=12)
        self.assertAlmostEqual(row["delta_K"], 49.77, delta=.006)


if __name__ == "__main__":
    unittest.main(verbosity=2)
