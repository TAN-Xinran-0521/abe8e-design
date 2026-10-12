"""Core unit and local-model regression tests.

These tests do not retrain models or run genome searches.
The model regression loads the trusted repository model only.
"""
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bprime_features import (
    sequence_to_onehot,
    phys_from_grna,
    build_X,
    validate_protospacer,
)
from bystander_prior import bystander_risk
from design_sgrna import pareto_front
from run_true_offtarget import rerank


class FeatureTests(unittest.TestCase):
    def test_onehot_order(self):
        encoded = np.asarray(sequence_to_onehot("ATGC" * 5)).reshape(20, 4)
        np.testing.assert_array_equal(encoded[:4], np.eye(4, dtype=int))
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(20))

    def test_feature_shape_dtype_and_duplicate_tm(self):
        frame = pd.DataFrame({"grna": ["ACGTACCAGGTGGAGCACCC"]})
        matrix = build_X(frame)
        self.assertEqual(matrix.shape, (1, 83))
        self.assertEqual(matrix.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix[0, 81], matrix[0, 82])
        self.assertEqual(len(phys_from_grna(frame.iloc[0]["grna"])), 3)

    def test_sequence_validation(self):
        self.assertEqual(
            validate_protospacer("atgc" * 5),
            "ATGC" * 5
        )
        for bad in ["A" * 19, "A" * 21, "A" * 19 + "N", ""]:
            with self.subTest(sequence=bad):
                with self.assertRaises(ValueError):
                    validate_protospacer(bad)


class BystanderTests(unittest.TestCase):
    def setUp(self):
        self.prior = {position: 1.0 for position in range(1, 21)}

    def test_target_is_not_a_bystander(self):
        result = bystander_risk("A" + "C" * 19, 1, self.prior)
        self.assertEqual(result["n_bystander_A"], 0)
        self.assertEqual(result["bystander_ratio"], 0.0)

    def test_two_equal_weight_as(self):
        result = bystander_risk("AA" + "C" * 18, 1, self.prior)
        self.assertEqual(result["n_bystander_A"], 1)
        self.assertAlmostEqual(result["bystander_ratio"], 0.5, places=8)

    def test_invalid_target_position(self):
        for position in [0, -1, 21]:
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    bystander_risk("A" * 20, position, self.prior)

    def test_incomplete_prior(self):
        incomplete = dict(self.prior)
        del incomplete[20]
        with self.assertRaises(ValueError):
            bystander_risk("A" * 20, 1, incomplete)


class RankingTests(unittest.TestCase):
    def test_pareto_uses_weighted_bystander(self):
        # A has a lower old window count but a worse weighted score.
        # With equal efficiency and off-target score, B must dominate A.
        frame = pd.DataFrame({
            "pred_efficiency": [0.5, 0.5],
            "bystander_A": [0, 1],
            "bystander_weighted": [0.6, 0.2],
            "offtarget_risk": [0.1, 0.1],
        })
        result = pareto_front(frame)
        self.assertEqual(
            result["pareto_optimal"].tolist(),
            [False, True]
        )

    def test_explicit_zero_weight_is_preserved(self):
        candidates = pd.DataFrame({
            "candidate_id": ["test_a", "test_b"],
            "grna": ["A" * 20, "C" * 20],
            "predicted_efficiency": [0.5, 0.4],
            "bystander_A_count": [1, 2],
            "bystander_weighted": [0.2, 0.3],
        })
        summary = pd.DataFrame({
            "grna": ["A" * 20, "C" * 20],
            "total_offtargets": [1, 2],
            "on_target_hits": [1, 1],
            "offtarget_risk_raw": [1.0, 2.0],
        })
        # Partially specified weights reproduce the previous zero-weight bug.
        result = rerank(
            candidates, summary,
            scenario="therapy", w=(0.0, None, None)
        )
        np.testing.assert_array_equal(
            result["weight_eff"].to_numpy(),
            np.zeros(len(result))
        )
        expected = (
            -0.3 * result["bystander_weighted"]
            -0.3 * result["offtarget_risk_norm"]
        )
        np.testing.assert_allclose(
            result["composite_score_true_offtarget"],
            expected
        )


class ModelRegressionTests(unittest.TestCase):
    def test_fixed_hfe_predictions(self):
        model_path = ROOT / "models/xgboost_design_v2.pkl"
        self.assertTrue(
            model_path.is_file(),
            "Required local B-prime model is missing."
        )
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

        frame = pd.DataFrame({
            "grna": [
                "TACGTACCAGGTGGAGCACC",
                "ACGTACCAGGTGGAGCACCC",
                "CGTACCAGGTGGAGCACCCA",
            ]
        })
        observed = np.clip(model.predict(build_X(frame)), 0, 1)
        expected = np.asarray([
            0.24527668952941895,
            0.33468392491340637,
            0.28835147619247437,
        ])
        np.testing.assert_allclose(
            observed, expected, rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()