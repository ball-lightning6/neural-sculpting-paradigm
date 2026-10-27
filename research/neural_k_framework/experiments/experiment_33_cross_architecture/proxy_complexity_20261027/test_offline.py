"""无求解器的逻辑/聚合数据测试；不启动神经训练或采样。"""

import itertools
import io
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import audit_existing_proxies as audit
from exact_circuits import CNF, apply_gate, atom_tables, check_assignment, check_lrat, gate_clauses, replay_witness
from recompute_calibration import calculate
from run_synthesis import build_panel


ROOT = Path(__file__).resolve().parent


class OfflineTests(unittest.TestCase):
    def test_gate_semantics(self):
        for op in ("and", "or", "xor", "not"):
            for a, b, value in itertools.product((False, True), repeat=3):
                cnf = CNF()
                selected, output = cnf.var(), cnf.var()
                gate_clauses(cnf, selected, output, op, a, b)
                expected = bool(apply_gate(op, int(a), int(b), 1))
                self.assertEqual(check_assignment(cnf, [selected, output if value else -output]), value == expected)

    def test_one_hot_encoding(self):
        cnf = CNF()
        variables = [cnf.var() for _ in range(4)]
        cnf.exactly_one(variables)
        for bits in itertools.product((False, True), repeat=4):
            possible = any(check_assignment(cnf, [i + 1 if v else -(i + 1) for i, v in enumerate(bits + rest)])
                           for rest in itertools.product((False, True), repeat=cnf.variables - 4))
            self.assertEqual(possible, sum(bits) == 1)

    def test_lrat_acceptance_and_corruption(self):
        problem, proof = ROOT / "p.cnf", ROOT / "p.lrat"
        content = {problem: "p cnf 1 2\n1 0\n-1 0\n", proof: "3 0 1 2 0\n"}
        def memory_open(path, mode="r", *args, **kwargs):
            value = content[path]
            return io.BytesIO(value.encode("ascii")) if "b" in mode else io.StringIO(value)
        with patch.object(Path, "open", memory_open):
            self.assertEqual(check_lrat(problem, proof)["additions"], 1)
            for bad in ("3 0 1 0\n", "3 0 1 -2 0\n"):
                content[proof] = bad
                with self.assertRaises(ValueError):
                    check_lrat(problem, proof)

    def test_panel_and_explicit_witness(self):
        panel = build_panel()
        self.assertEqual(len(panel), 20)
        self.assertEqual(sum(r["n"] == 3 for r in panel), 10)
        atoms, names = atom_tables(3)
        target = (atoms[2] & atoms[3]) | atoms[4]
        witness = {"n": 3, "language": "aon_paid", "atoms": names,
                   "gates": [{"op": "and", "a": 2, "b": 3}, {"op": "or", "a": 5, "b": 4}],
                   "outputs": [{"signal": 6, "invert": False}], "targets": [target]}
        self.assertEqual(replay_witness(witness), [target])

    def test_recorded_calibration_and_reversals(self):
        report = calculate()
        self.assertEqual((report["recorded_certificate_count"], report["recorded_exact_count"]), (60, 59))
        self.assertEqual((report["shared_directions"], report["pair_count"]), (43, 45))
        self.assertEqual((report["negative_medians"], report["all_cross_pool_negative"]), (20, 19))
        self.assertEqual(report["affine_parity_ids"], [0x3C, 0x69])
        self.assertEqual((report["unresolved"][0]["lower_bound"], report["unresolved"][0]["upper_bound"]), ("8", "10"))
        saved = {}
        def capture(path, text, *args, **kwargs):
            saved[path.name] = text
            return len(text)
        with patch.object(Path, "write_text", capture), patch("builtins.print"):
            audit.main()
        result = json.loads(saved["LOCAL_EVIDENCE_AUDIT.json"])
        self.assertEqual(result["proxy_count_in_old_correlations"], 19)
        self.assertEqual(result["checks"], {"influence_recomputed_max_error": 0., "walsh_mean_degree_equals_influence_max_error": 0., "old_spearman_max_error": 0.})
        document = (ROOT.parent / "COMPLEXITY_CALIBRATION_20261027.md").read_text(encoding="utf-8")
        for proxy in audit.KEY_PROXIES:
            row = next(line for line in document.splitlines() if line.startswith(f"| {proxy} |"))
            observed = re.findall(r"-?\d+\.\d{3}", row)
            expected = []
            for measurement in ("K_at_0p001", "slope_0p01_to_0p001", "slope_0p0025_to_0p001"):
                values = [r["spearman"] for r in result["correlations"] if r["scope"] == "balanced_orbit_representatives" and r["proxy"] == proxy and r["measurement"] == measurement]
                expected.extend((f"{min(values):.3f}", f"{max(values):.3f}"))
            self.assertEqual(observed, expected, proxy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
