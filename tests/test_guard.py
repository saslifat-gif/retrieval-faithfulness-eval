"""Tests for grounding/guard.py — the control loop, lexical scorer only (offline)."""
import unittest

import guard

DOCS = [
    "The Eiffel Tower is located in Paris, France.",
    "The Eiffel Tower was completed in 1889.",
]
QUERY = "Where is the Eiffel Tower?"
GROUNDED = "The Eiffel Tower is located in Paris. It was completed in 1889."
HALLUCINATED = "The Eiffel Tower is located in Paris. It is 450 meters tall."


class TestCheck(unittest.TestCase):
    def test_grounded_response_passes(self):
        report = guard.check(QUERY, DOCS, GROUNDED)
        self.assertTrue(report.grounded)
        self.assertEqual(report.unsupported, [])
        self.assertEqual(report.feedback(), "")

    def test_hallucinated_sentence_is_flagged_with_feedback(self):
        report = guard.check(QUERY, DOCS, HALLUCINATED)
        self.assertFalse(report.grounded)
        self.assertEqual(len(report.unsupported), 1)
        self.assertIn("450 meters", report.unsupported[0].text)
        self.assertIn("450 meters", report.feedback())
        self.assertLess(report.weakest_score, report.threshold)

    def test_empty_query_or_response_raises(self):
        with self.assertRaises(ValueError):
            guard.check("", DOCS, GROUNDED)
        with self.assertRaises(ValueError):
            guard.check(QUERY, DOCS, "   ")


class TestGroundedAnswer(unittest.TestCase):
    @staticmethod
    def retrieve(query: str) -> list[str]:
        return DOCS

    def test_regenerates_on_feedback_then_answers(self):
        def generate(query, documents, feedback):
            return HALLUCINATED if feedback is None else GROUNDED

        result = guard.grounded_answer(QUERY, retrieve=self.retrieve, generate=generate)
        self.assertEqual(result.action, "answered")
        self.assertTrue(result.grounded)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.answer, GROUNDED)

    def test_abstains_after_max_attempts(self):
        def generate(query, documents, feedback):
            return HALLUCINATED

        result = guard.grounded_answer(
            QUERY, retrieve=self.retrieve, generate=generate, max_attempts=2
        )
        self.assertEqual(result.action, "abstained")
        self.assertFalse(result.grounded)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.answer, guard.ABSTAIN_MESSAGE)

    def test_on_fail_best_returns_best_effort(self):
        def generate(query, documents, feedback):
            return HALLUCINATED

        result = guard.grounded_answer(
            QUERY, retrieve=self.retrieve, generate=generate, max_attempts=1, on_fail="best"
        )
        self.assertEqual(result.action, "returned_best_effort")
        self.assertEqual(result.answer, HALLUCINATED)

    def test_invalid_on_fail_raises(self):
        with self.assertRaises(ValueError):
            guard.grounded_answer(
                QUERY, retrieve=self.retrieve, generate=lambda q, d, f: GROUNDED, on_fail="explode"
            )


if __name__ == "__main__":
    unittest.main()
