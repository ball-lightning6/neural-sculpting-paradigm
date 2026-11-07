"""Standard-library tests for a copied, standalone public package."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid


HERE = Path(__file__).resolve().parent
FILES = ("recompute.py", "test_public_only.py", "README_zh.md", "README_en.md", "audit_summary.json",
         "source_manifest.json", "cross_architecture_curves.csv", "radial_curves.csv",
         "radial_reweighting.py", "radial_reweighting.json")


class PublicOnlyTests(unittest.TestCase):
    def setUp(self):
        self.package = (HERE / (".public-check-" + uuid.uuid4().hex)).resolve()
        self.assertTrue(self.package.is_relative_to(HERE) and self.package != HERE)
        self.package.mkdir()
        self.addCleanup(self.clean_package)
        for name in FILES:
            shutil.copyfile(HERE / name, self.package / name)

    def clean_package(self):
        self.assertTrue(self.package.is_relative_to(HERE) and self.package != HERE)
        shutil.rmtree(self.package)

    def run_check(self, *args, script="recompute.py"):
        return subprocess.run([sys.executable, "-I", "-S", "-B", str(self.package / script), *args],
                              cwd=self.package, capture_output=True, text=True, timeout=30)

    def csv_rows(self):
        with (self.package / "cross_architecture_curves.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def write_csv(self, rows):
        with (self.package / "cross_architecture_curves.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_standalone_package(self):
        result = self.run_check("--public-only", "--documents")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["cross_curve_nodes"], 234)
        self.assertEqual(report["radial_condition_nodes"], 462)
        self.assertEqual(report["readout_summary_rows_checked"], 6)
        self.assertIn("not_smc_replication", report["mode"])
        self.assertTrue(report["documents"]["links_exist"])

    def test_missing_originals_block_full_audit(self):
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Original result directories are absent", result.stderr)

    def test_changed_endpoint_is_rejected(self):
        rows = self.csv_rows()
        row = next(row for row in rows if float(row["epsilon"]) == .0001)
        row["K_pool_0"] = str(float(row["K_pool_0"])+1)
        self.write_csv(rows)
        result = self.run_check("--public-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("public_cross_endpoint_K_mean", result.stderr)

    def test_duplicate_node_is_rejected(self):
        rows = self.csv_rows()
        self.write_csv(rows+[rows[0]])
        result = self.run_check("--public-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate node", result.stderr)

    def test_missing_csv_is_rejected(self):
        (self.package / "radial_curves.csv").unlink()
        self.assertNotEqual(self.run_check("--public-only").returncode, 0)

    def test_public_mode_cannot_write(self):
        before = {name: hashlib.sha256((self.package / name).read_bytes()).hexdigest() for name in FILES}
        result = self.run_check("--public-only", "--write")
        self.assertEqual(result.returncode, 2)
        after = {name: hashlib.sha256((self.package / name).read_bytes()).hexdigest() for name in FILES}
        self.assertEqual(before, after)

    def test_radial_standalone_and_write_guard(self):
        script = "radial_reweighting.py"
        record = self.package / "radial_reweighting.json"
        before = record.read_bytes()
        for flags in ((), ("--public-only",)):
            result = self.run_check(*flags, script=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["passed"])
            self.assertFalse(report["written"] or report["weight_ess_recomputed"])
            self.assertEqual(report["source_sha256_matched"], 0)
            self.assertEqual((report["curves"], report["curve_nodes"], report["readme_rows_checked_per_language"]),
                             (24, 144, 8))
        for flags in (("--write",), ("--public-only", "--write"), ("--public-only", "--from-sources")):
            self.assertEqual(self.run_check(*flags, script=script).returncode, 2)
        self.assertNotEqual(self.run_check("--from-sources", "--write", script=script).returncode, 0)
        self.assertEqual(before, record.read_bytes())

    def test_radial_public_data_checks(self):
        record = self.package / "radial_reweighting.json"
        original = record.read_bytes()
        for change in ("K_bits", "config", "duplicate", "ess"):
            with self.subTest(change=change):
                data = json.loads(original)
                row = next(r for r in data["curves"] if (r["rule"], r["family"], r["parameter"])
                           == ("parity4", "radial_power_p", 1))
                if change == "K_bits":
                    row["K_bits"][2] += 1
                elif change == "config":
                    data["config"]["gaussian_ratio"] = "sigma/sigma0"
                elif change == "duplicate":
                    data["curves"].append(row)
                else:
                    row["min_weight_ess_fraction"] = 0.8
                record.write_text(json.dumps(data), encoding="utf-8")
                self.assertNotEqual(self.run_check("--public-only", script="radial_reweighting.py").returncode, 0)

    def test_radial_public_checks_each_report(self):
        for lang in ("zh", "en"):
            with self.subTest(lang=lang):
                path = self.package / f"README_{lang}.md"
                original = path.read_bytes()
                self.assertIn(b"| 1.5 | 1.9138 |", original)
                path.write_bytes(original.replace(b"| 1.5 | 1.9138 |", b"| 1.5 | 1.0000 |", 1))
                self.assertNotEqual(self.run_check("--public-only", script="radial_reweighting.py").returncode, 0)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
