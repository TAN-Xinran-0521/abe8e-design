"""Tests for strict FoldX saved-output parsing."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from foldx_results import read_energy_table, read_mutations, parse_mutation


class FoldXParserTests(unittest.TestCase):
    def write(self, directory, text):
        path = Path(directory) / "sample.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_header_order_not_column_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(
                directory,
                "preamble\nPdb\ttotal energy\tSD\nmodel_1\t-0.4\t0.02\n"
            )
            table = read_energy_table(path, with_sd=True)
            self.assertEqual(table.loc["model_1", "energy"], -0.4)
            self.assertEqual(table.loc["model_1", "sd"], 0.02)

    def test_duplicate_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(
                directory,
                "Pdb\tSD\ttotal energy\nm\t0.1\t-1\nm\t0.1\t-1\n"
            )
            with self.assertRaises(ValueError):
                read_energy_table(path, with_sd=True)

    def test_nonfinite_energy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "Pdb\ttotal energy\nm\tnan\n")
            with self.assertRaises(ValueError):
                read_energy_table(path)

    def test_missing_sd_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "Pdb\ttotal energy\nm\t-1\n")
            with self.assertRaises(ValueError):
                read_energy_table(path, with_sd=True)

    def test_mutation_reference_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            parse_mutation("C1A", "AC")

    def test_single_mutation_chain_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "AA1C;\n")
            self.assertEqual(read_mutations(path, "AC", "A"), ["A1C"])
            with self.assertRaises(ValueError):
                read_mutations(path, "AC", "B")

    def test_duplicate_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "AA1C;\nAA1C;\n")
            with self.assertRaises(ValueError):
                read_mutations(path, "AC", "A")


if __name__ == "__main__":
    unittest.main()