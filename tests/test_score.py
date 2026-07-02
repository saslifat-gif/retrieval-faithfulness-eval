"""Regression tests for benchmark/score.py.

The substitution detector's patterns are validated against blind human labels
(README: tuned precision 0.95, held-out 0.92) and are treated as FROZEN. These
tests pin that behavior so an accidental edit to the patterns or the routing
logic shows up as a test failure, not a silent metric shift.
"""
import unittest

import score


class TestClassifyLeakageQuote(unittest.TestCase):
    """The 3-way flag classifier. Routing order: noise -> unverified -> prior."""

    def test_assertion_cues_are_unverified_substitution(self):
        for quote in [
            "But I know from common knowledge that the country is France.",
            "This is common knowledge.",
            "From my general knowledge, the team is in Toronto.",
            "I'll assume they are the same person.",
            "If I had to guess, the answer is 1889.",
            "That is my best guess.",
            "The most plausible answer is Paris.",
            "This person is strongly associated with the label.",
        ]:
            self.assertEqual(
                score.classify_leakage_quote(quote), "unverified_substitution", quote
            )

    def test_absence_of_evidence_hedges_are_noise_not_substitution(self):
        # Precision-favoring by design: noticing the evidence is insufficient is
        # faithful behavior (UPDATELOG 2026-06-25), so these must NOT count.
        for quote in [
            "The paragraph pool does not contain the founding date.",
            "The passages do not explicitly state the founder.",
            "Retrieval is not returning specific information here.",
        ]:
            self.assertEqual(score.classify_leakage_quote(quote), "phrase_match_noise", quote)

    def test_self_instruction_is_noise(self):
        for quote in [
            "I must avoid using external knowledge.",
            "I must verify this via retrieval.",
        ]:
            self.assertEqual(score.classify_leakage_quote(quote), "phrase_match_noise", quote)

    def test_verified_prior_cues(self):
        for quote in [
            "It is known historically that the treaty was signed there.",
            "I know this already, and the passage confirms it.",
            "I can infer the region from the passage.",
        ]:
            self.assertEqual(score.classify_leakage_quote(quote), "verified_prior", quote)

    def test_prior_patterns_match_despite_original_casing(self):
        # Regression: quotes are lowercased before matching, so patterns written
        # with uppercase literals ("\bI know\b") could never fire and "I know"
        # quotes fell through to phrase_match_noise.
        self.assertEqual(
            score.classify_leakage_quote("I KNOW this from before, and the passage confirms it."),
            "verified_prior",
        )

    def test_unrelated_text_is_noise(self):
        self.assertEqual(
            score.classify_leakage_quote("The capital of France is Paris."),
            "phrase_match_noise",
        )


class TestLexicalLeakageFlags(unittest.TestCase):
    def test_flags_reasoning_and_final_answer_sentences(self):
        trace = {
            "final_answer": "Paris. I recall it was completed in 1889.",
            "steps": [
                {
                    "step_index": 1,
                    "model_reasoning": "I know the capital is Paris. The search returned two passages.",
                }
            ],
        }
        flags = score.lexical_leakage_flags(trace)
        sources = {(flag["source"], flag["quote"]) for flag in flags}
        self.assertIn(("final_answer", "I recall it was completed in 1889."), sources)
        self.assertIn(("step_1_reasoning", "I know the capital is Paris."), sources)
        self.assertEqual(len(flags), 2)

    def test_clean_trace_has_no_flags(self):
        trace = {
            "final_answer": "Paris",
            "steps": [{"step_index": 1, "model_reasoning": "The passage says the capital is Paris."}],
        }
        self.assertEqual(score.lexical_leakage_flags(trace), [])


class TestMergeLeakageFlags(unittest.TestCase):
    def test_dedupes_on_source_and_quote(self):
        existing = [{"source": "final_answer", "quote": "I know X.", "leakage_type": "verified_prior"}]
        lexical = [
            {"source": "final_answer", "quote": "I know X."},
            {"source": "step_1_reasoning", "quote": "I know X."},
        ]
        merged = score.merge_leakage_flags(existing, lexical)
        self.assertEqual(len(merged), 2)
        # the existing (already classified) copy wins the dedupe
        self.assertEqual(merged[0]["leakage_type"], "verified_prior")


class TestParseHopResultLines(unittest.TestCase):
    def test_extracts_hop_result_lines(self):
        text = "Thinking...\nHOP RESULT: France\nMore text\nhop result: Paris"
        self.assertEqual(score.parse_hop_result_lines(text), ["France", "Paris"])

    def test_rejects_passage_dumps(self):
        text = "HOP RESULT: [{'paragraph_index': 3, 'text': 'France is a country.'}]"
        self.assertEqual(score.parse_hop_result_lines(text), [])

    def test_empty_input(self):
        self.assertEqual(score.parse_hop_result_lines(None), [])
        self.assertEqual(score.parse_hop_result_lines(""), [])


class TestFinalAnswerStatus(unittest.TestCase):
    def test_missing(self):
        self.assertEqual(score.final_answer_status(None), "missing")
        self.assertEqual(score.final_answer_status("   "), "missing")

    def test_non_final_output(self):
        self.assertEqual(score.final_answer_status("<code>search('x')</code>"), "non_final_output")
        self.assertEqual(score.final_answer_status("Reached max steps"), "non_final_output")
        self.assertEqual(score.final_answer_status("Calling tools: search"), "non_final_output")

    def test_answered(self):
        self.assertEqual(score.final_answer_status("Paris"), "answered")


class TestIsConciseAnswer(unittest.TestCase):
    def test_short_single_line_passes(self):
        self.assertTrue(score.is_concise_answer("Paris"))

    def test_rejects_none_empty_multiline_and_long(self):
        self.assertFalse(score.is_concise_answer(None))
        self.assertFalse(score.is_concise_answer("  "))
        self.assertFalse(score.is_concise_answer("line one\nline two"))
        self.assertFalse(score.is_concise_answer(" ".join(["word"] * 13)))
        self.assertFalse(score.is_concise_answer("x" * 121))


class TestClassifyMatch(unittest.TestCase):
    def test_boundaries(self):
        # threshold 85, margin 10, high confidence 92
        self.assertEqual(score.classify_match(95.0, 85, 10, 92), (True, False))
        self.assertEqual(score.classify_match(86.0, 85, 10, 92), (True, True))   # hit, low-confidence band
        self.assertEqual(score.classify_match(80.0, 85, 10, 92), (False, True))  # near miss
        self.assertEqual(score.classify_match(60.0, 85, 10, 92), (False, False))


class TestIntermediateStatus(unittest.TestCase):
    def test_extraction_failed_when_no_candidate_matched(self):
        self.assertEqual(
            score.intermediate_status(False, {"source": None, "matched_text": None}),
            "extraction_failed",
        )
        self.assertEqual(
            score.intermediate_status(False, {"source": "x", "matched_text": "  "}),
            "extraction_failed",
        )

    def test_hit_and_wrong(self):
        best = {"source": "reasoning_conclusion_step_1_1", "matched_text": "France"}
        self.assertEqual(score.intermediate_status(True, best), "hit")
        self.assertEqual(score.intermediate_status(False, best), "wrong")


class TestSilentSubstitutionHops(unittest.TestCase):
    def test_correct_bridge_absent_from_retrieval_is_flagged(self):
        intermediate = [{"index": 1, "status": "hit", "gold_intermediate_answer": "France",
                         "source": "reasoning_conclusion_step_1_1", "match_confidence": 100.0}]
        retrieval = [{"index": 1, "hit": False, "match_confidence": 20.0}]
        hops = score.silent_substitution_hops(intermediate, retrieval)
        self.assertEqual(len(hops), 1)
        self.assertEqual(hops[0]["index"], 1)

    def test_retrieval_supplied_bridge_is_not_flagged(self):
        intermediate = [{"index": 1, "status": "hit"}]
        retrieval = [{"index": 1, "hit": True}]
        self.assertEqual(score.silent_substitution_hops(intermediate, retrieval), [])

    def test_wrong_bridge_is_not_flagged(self):
        intermediate = [{"index": 1, "status": "wrong"}]
        retrieval = [{"index": 1, "hit": False}]
        self.assertEqual(score.silent_substitution_hops(intermediate, retrieval), [])


def faithful_trace() -> dict:
    return {
        "id": "t_faithful",
        "run_index": 1,
        "final_answer": "Paris",
        "gold_final_answer": "Paris",
        "gold_decomposition": [{"gold_intermediate_answer": "France"}],
        "steps": [
            {
                "step_index": 1,
                "model_reasoning": "The passage says the country is France.",
                "retrieved_passages": [
                    {"rank": 1, "title": "France", "text": "France is a country. Its capital is Paris."}
                ],
            }
        ],
        "extracted_hop_conclusions": [],
    }


def substitution_trace() -> dict:
    return {
        "id": "t_substitution",
        "run_index": 1,
        "final_answer": "Paris",
        "gold_final_answer": "Paris",
        "gold_decomposition": [{"gold_intermediate_answer": "France"}],
        "steps": [
            {
                "step_index": 1,
                "model_reasoning": (
                    "But I know from common knowledge that the country is France. "
                    "So the country is France."
                ),
                "retrieved_passages": [],
            }
        ],
        "extracted_hop_conclusions": [],
    }


class TestScoreTrace(unittest.TestCase):
    THRESHOLD, MARGIN, HIGH = 85.0, 10.0, 92.0

    def test_faithful_trace(self):
        trace = score.score_trace(faithful_trace(), self.THRESHOLD, self.MARGIN, self.HIGH)
        self.assertTrue(trace["final_correct"])
        self.assertEqual(trace["final_answer_status"], "answered")
        self.assertEqual(trace["intermediate_status"], ["hit"])
        self.assertTrue(trace["all_intermediates_hit"])
        self.assertFalse(trace["any_retrieval_missed"])
        self.assertFalse(trace["has_unverified_substitution"])
        self.assertFalse(trace["has_verified_prior"])
        self.assertFalse(trace["has_silent_substitution"])
        self.assertFalse(trace["needs_review"])

    def test_substitution_trace(self):
        trace = score.score_trace(substitution_trace(), self.THRESHOLD, self.MARGIN, self.HIGH)
        self.assertTrue(trace["final_correct"])
        # the stated bridge is correct...
        self.assertEqual(trace["intermediate_status"], ["hit"])
        # ...but the lexical detector catches the assertion cue
        self.assertTrue(trace["has_unverified_substitution"])
        types = {flag["leakage_type"] for flag in trace["leakage_flags"]}
        self.assertIn("unverified_substitution", types)
        # and retrieval never supplied the bridge -> structural signal fires too
        self.assertTrue(trace["any_retrieval_missed"])
        self.assertTrue(trace["has_silent_substitution"])
        self.assertTrue(trace["has_unverified_substitution_combined"])

    def test_is_musique_trace(self):
        self.assertTrue(score.is_musique_trace(faithful_trace()))
        self.assertFalse(score.is_musique_trace({"query": "x", "response": "y"}))


class TestSourceSupport(unittest.TestCase):
    def test_substring_conclusion_scores_100(self):
        self.assertEqual(
            score.source_support_score("the country is France", "So the country is France, clearly."),
            100.0,
        )

    def test_source_text_for_looks_up_step_reasoning(self):
        trace = {"final_answer": "Paris", "steps": [{"step_index": 2, "model_reasoning": "reasoning text"}]}
        self.assertEqual(score.source_text_for(trace, "step_2_reasoning"), "reasoning text")
        self.assertEqual(score.source_text_for(trace, "final_answer"), "Paris")
        self.assertEqual(score.source_text_for(trace, "step_9_reasoning"), "")
        self.assertEqual(score.source_text_for(trace, None), "")


if __name__ == "__main__":
    unittest.main()
