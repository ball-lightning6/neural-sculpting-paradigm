"""工程自检；不将小预算测试当作新科学结果。"""

import copy
import json
import math
import unittest

import numpy as np
import torch

import run_experiment as exp


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU 自检需要 CUDA")
        cls.device = torch.device(exp.DEVICE)
        torch.cuda.set_device(cls.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)

    def test_rule_ids_and_numpy_forward(self):
        x = exp.inputs(self.device)
        a, b, c, d = x.long().T
        f, g = a ^ b, (a ^ b) | c
        powers = 1 << torch.arange(16, device=self.device)
        self.assertEqual(int((f * powers).sum()), exp.RULES["f"])
        self.assertEqual(int((g * powers).sum()), exp.RULES["g"])
        streams = exp.make_generators(self.device, 800, 2, 20261024)
        theta = exp.normal_batch((2, 128, 97), streams["prior"], self.device)
        for scale in (1, 4, 12):
            z = exp.forward(theta * scale, x).cpu().numpy().reshape(-1, 16)
            expected = exp.numpy_forward((theta * scale).cpu().numpy())
            np.testing.assert_allclose(z, expected, atol=4e-5, rtol=4e-5)
        for rule in exp.RULES:
            for kind in exp.KINDS:
                scorer, _, _ = exp.scorer_for(rule, kind, self.device)
                actual = scorer(theta).cpu().numpy().ravel()
                _, expected = exp.numpy_measure(theta.cpu().numpy().reshape(-1, 97), exp.RULES[rule], kind)
                np.testing.assert_allclose(actual, expected["score"], atol=2e-6, rtol=2e-6)

    def test_signed_margin_and_budget(self):
        signs = torch.tensor([1., -1., 1., -1.], device=self.device)
        z = 3 * signs
        mean, maximum, variance = exp.features(z[None], signs)
        self.assertEqual(float(variance), 0.)
        self.assertAlmostEqual(float(mean), math.log1p(math.exp(-3)), places=7)
        self.assertEqual(float(mean), float(maximum))
        z2 = torch.tensor([[9., -1., 9., -1.]], device=self.device)
        mean2, maximum2, variance2 = exp.features(z2, signs)
        self.assertGreater(float(maximum2), float(mean2))
        self.assertGreater(float(variance2), 0.)

    def test_exact_rng_resume_and_no_pool_mixing(self):
        scorer, _, _ = exp.scorer_for("g", "aligned", self.device)
        streams = exp.make_generators(self.device, 801, 2, 20261024)
        state = exp.initialize(2, 128, self.device, streams, scorer)
        state["ancestry"][1] += 10000
        saved = copy.deepcopy(state)
        rng = {name: [g.get_state().clone() for g in group] for name, group in streams.items()}
        exp.step(state, scorer, streams, [.5, .3], 1, 2)
        for name, group in streams.items():
            for g, r in zip(group, rng[name]):
                g.set_state(r)
        exp.step(saved, scorer, streams, [.5, .3], 1, 2)
        for key in ("theta", "score", "K", "ancestry", "scales"):
            torch.testing.assert_close(state[key], saved[key], atol=0, rtol=0)
        self.assertTrue(bool((state["ancestry"][0] < 128).all()))
        self.assertTrue(bool((state["ancestry"][1] >= 10000).all()))

    def test_analytic_halfspace(self):
        streams = exp.make_generators(self.device, 802, 4, 20261024)
        scorer = lambda theta: -theta[..., 0]
        state = exp.initialize(4, 4096, self.device, streams, scorer)
        thresholds = [0., -1., -2.]
        observed = [[] for _ in thresholds]
        for _ in range(50):
            hit, _ = exp.step(state, scorer, streams, thresholds, 2, 10)
            for r, yes in enumerate(hit):
                if yes:
                    observed[state["node"][r]].append(float(state["K"][r]))
                    state["node"][r] += 1
            if all(v == len(thresholds) for v in state["node"]):
                break
        for t, values in zip(thresholds, observed):
            expected = -math.log2(.5 * math.erfc(-t / math.sqrt(2)))
            self.assertEqual(len(values), 4)
            self.assertLess(max(abs(v-expected) for v in values), .25)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    exp.atomic_json(exp.ROOT / "ENGINEERING_TESTS.json", dict(
        passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=[str(v) for v in result.failures], errors=[str(v) for v in result.errors],
        gpu=torch.cuda.get_device_name(torch.device(exp.DEVICE)),
        sampler_sha256=exp.digest(exp.ROOT / "run_experiment.py")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
