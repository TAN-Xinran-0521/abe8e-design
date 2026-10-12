"""Coordinate regression tests for the tested no-bulge, 3-prime PAM format."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_true_offtarget import mark_on_target


def candidate(guide, chrom, start, end, strand):
    return pd.DataFrame([{
        "grna": guide,
        "ot_chrom": chrom,
        "ot_start": start,
        "ot_end": end,
        "strand": strand,
    }])


def hit(guide, chrom, position, strand, mismatches=1):
    return {
        "grna": guide,
        "chrom": chrom,
        "position": position,
        "strand": strand,
        "mismatches": mismatches,
    }


class CoordinateTests(unittest.TestCase):
    def test_hfe_plus_strand(self):
        guide = "ACGTACCAGGTGGAGCACCC"
        candidates = candidate(
            guide, "chr6", 26092909, 26092928, "+"
        )
        hits = pd.DataFrame([
            hit(guide, "chr6", 26092908, "+", mismatches=1)
        ])
        result = mark_on_target(hits, candidates, pam_length=3)
        row = result.iloc[0]
        self.assertEqual(row["hit_proto_start_1based"], 26092909)
        self.assertEqual(row["hit_proto_end_1based"], 26092928)
        self.assertTrue(bool(row["is_on_target"]))

    def test_lmna_minus_strand(self):
        guide = "GGTCCACCCACCTGGGCTCC"
        candidates = candidate(
            guide, "chr1", 156138599, 156138618, "-"
        )
        hits = pd.DataFrame([
            hit(guide, "chr1", 156138595, "-", mismatches=1)
        ])
        result = mark_on_target(hits, candidates, pam_length=3)
        row = result.iloc[0]
        self.assertEqual(row["hit_proto_start_1based"], 156138599)
        self.assertEqual(row["hit_proto_end_1based"], 156138618)
        self.assertTrue(bool(row["is_on_target"]))

    def test_nearby_and_other_exact_matches_are_retained(self):
        guide = "ACGTACCAGGTGGAGCACCC"
        candidates = candidate(
            guide, "chr6", 26092909, 26092928, "+"
        )
        hits = pd.DataFrame([
            # Nearby but shifted by one base.
            hit(guide, "chr6", 26092909, "+", mismatches=1),
            # Same normalized protospacer span, but opposite strand.
            hit(guide, "chr6", 26092905, "-", mismatches=0),
            # Exact sequence match elsewhere is not the intended target.
            hit(guide, "chr6", 100000, "+", mismatches=0),
        ])
        result = mark_on_target(hits, candidates, pam_length=3)
        self.assertFalse(bool(result["is_on_target"].any()))

    def test_missing_coordinates_are_rejected(self):
        guide = "ACGTACCAGGTGGAGCACCC"
        candidates = candidate(guide, "chr6", None, 26092928, "+")
        hits = pd.DataFrame([
            hit(guide, "chr6", 26092908, "+")
        ])
        with self.assertRaises(ValueError):
            mark_on_target(hits, candidates, pam_length=3)

    def test_broad_tolerance_is_rejected(self):
        guide = "ACGTACCAGGTGGAGCACCC"
        candidates = candidate(
            guide, "chr6", 26092909, 26092928, "+"
        )
        hits = pd.DataFrame([
            hit(guide, "chr6", 26092908, "+")
        ])
        with self.assertRaises(ValueError):
            mark_on_target(
                hits, candidates, tolerance=30, pam_length=3
            )


if __name__ == "__main__":
    unittest.main()