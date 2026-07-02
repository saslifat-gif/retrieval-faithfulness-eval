"""Tests for validation/validate.py — confusion/metrics math."""
import unittest

import validate


class TestConfusion(unittest.TestCase):
    def test_counts(self):
        pairs = [(True, True), (True, False), (False, True), (False, False), (True, True)]
        self.assertEqual(validate.confusion(pairs), {"tp": 2, "fp": 1, "fn": 1, "tn": 1})


class TestMetrics(unittest.TestCase):
    def test_perfect_detector(self):
        m = validate.metrics({"tp": 5, "fp": 0, "fn": 0, "tn": 5})
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["n"], 10)

    def test_zero_division_yields_none(self):
        # no positive predictions -> precision undefined; no gold positives -> recall undefined
        m = validate.metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 4})
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["f1"])
        self.assertEqual(m["accuracy"], 1.0)

    def test_all_wrong_predictions_give_f1_zero(self):
        m = validate.metrics({"tp": 0, "fp": 3, "fn": 2, "tn": 0})
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)

    def test_rounding(self):
        m = validate.metrics({"tp": 1, "fp": 2, "fn": 0, "tn": 0})
        self.assertEqual(m["precision"], 0.333)


class TestEvaluatePredictor(unittest.TestCase):
    def test_predictor_applied_to_traces(self):
        rows = [
            {"trace": {"flag": True}, "gold": True},
            {"trace": {"flag": False}, "gold": True},
            {"trace": {"flag": False}, "gold": False},
        ]
        m = validate.evaluate_predictor(rows, lambda t: t["flag"])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 0, 1, 1))


class TestTraceHelpers(unittest.TestCase):
    def test_trace_key(self):
        self.assertEqual(validate.trace_key({"id": "q1", "run_index": 2}), "q1::2")

    def test_is_scored_trace(self):
        scored = {"gold_final_answer": "x", "leakage_flags": [], "has_unverified_substitution": False}
        self.assertTrue(validate.is_scored_trace(scored))
        self.assertFalse(validate.is_scored_trace({"gold_final_answer": "x"}))


if __name__ == "__main__":
    unittest.main()
