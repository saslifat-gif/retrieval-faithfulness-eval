"""Tests for grounding/adapter.py — the pure/lexical paths only.

No model downloads: the embedding/NLI/HHEM scorers are exercised by their
callers at runtime; here we pin key normalization, sentence parsing, and the
lexical end-to-end `score_row` contract that guard.py and the web GUI rely on.
"""
import unittest

import adapter


class TestNormalizeKey(unittest.TestCase):
    def test_strips_trailing_punctuation(self):
        # gold unsupported keys sometimes carry trailing periods ("a.")
        self.assertEqual(adapter.normalize_key("a."), "a")

    def test_lowercases(self):
        self.assertEqual(adapter.normalize_key("0A"), "0a")

    def test_none_and_empty(self):
        self.assertEqual(adapter.normalize_key(None), "")
        self.assertEqual(adapter.normalize_key("  "), "")


class TestKeySet(unittest.TestCase):
    def test_json_string_list(self):
        self.assertEqual(adapter.key_set('["a.", "b"]'), {"a", "b"})

    def test_plain_list_and_dict(self):
        self.assertEqual(adapter.key_set(["a", "b."]), {"a", "b"})
        self.assertEqual(adapter.key_set({"a": 1, "b": 2}), {"a", "b"})

    def test_none_and_scalar(self):
        self.assertEqual(adapter.key_set(None), set())
        self.assertEqual(adapter.key_set("a."), {"a"})


class TestAsBool(unittest.TestCase):
    def test_truthy_strings(self):
        for value in ("1", "true", "Yes", " y "):
            self.assertTrue(adapter.as_bool(value), value)

    def test_falsy_strings_and_numbers(self):
        for value in ("0", "false", "no", ""):
            self.assertFalse(adapter.as_bool(value), value)
        self.assertFalse(adapter.as_bool(0))
        self.assertTrue(adapter.as_bool(1.0))

    def test_bool_passthrough(self):
        self.assertTrue(adapter.as_bool(True))
        self.assertFalse(adapter.as_bool(False))


class TestSentenceRecords(unittest.TestCase):
    def test_key_text_pairs(self):
        records = adapter.sentence_records([["0a", "First."], ["0b", "Second."]], "d")
        self.assertEqual(records, [{"key": "0a", "text": "First."}, {"key": "0b", "text": "Second."}])

    def test_dict_form(self):
        records = adapter.sentence_records({"a": "One.", "b": "Two."}, "r")
        self.assertEqual({r["key"] for r in records}, {"a", "b"})

    def test_json_string_form(self):
        records = adapter.sentence_records('[["a", "Hi there."]]', "r")
        self.assertEqual(records, [{"key": "a", "text": "Hi there."}])

    def test_plain_string_splits_into_sentences(self):
        records = adapter.sentence_records("One. Two.", "r")
        self.assertEqual(records, [{"key": "r0", "text": "One."}, {"key": "r1", "text": "Two."}])

    def test_fallback_texts_used_when_raw_empty(self):
        records = adapter.sentence_records(None, "d", ["Doc one. Doc two."])
        self.assertEqual(records, [{"key": "00", "text": "Doc one."}, {"key": "01", "text": "Doc two."}])


class TestBestDocumentSupport(unittest.TestCase):
    def test_picks_best_matching_sentence(self):
        documents = [
            {"key": "0a", "text": "The Eiffel Tower is in Paris."},
            {"key": "0b", "text": "It was completed in 1889."},
        ]
        best = adapter.best_document_support("The tower was completed in 1889.", documents)
        self.assertEqual(best["document_sentence_key"], "0b")
        self.assertGreater(best["score"], 75.0)


def ragbench_row() -> dict:
    return {
        "id": "ex1",
        "question": "Where is the Eiffel Tower?",
        "documents": ["The Eiffel Tower is in Paris. It was completed in 1889."],
        "documents_sentences": [["0a", "The Eiffel Tower is in Paris."], ["0b", "It was completed in 1889."]],
        "response": "The Eiffel Tower is in Paris. It is 450 meters tall.",
        "response_sentences": [["a", "The Eiffel Tower is in Paris."], ["b", "It is 450 meters tall."]],
        # trailing period: must still match response key "b" after normalization
        "unsupported_response_sentence_keys": ["b."],
        "adherence_score": False,
    }


class TestScoreRowLexical(unittest.TestCase):
    def test_end_to_end(self):
        record = adapter.score_row(ragbench_row(), 0, 75.0, "ragbench", "delucionqa", "test")
        self.assertIsNotNone(record)
        self.assertEqual(record["gold_unsupported_response_sentence_keys"], ["b"])
        self.assertTrue(record["gold_ungrounded_response"])
        self.assertEqual(record["predicted_unsupported_response_sentence_keys"], ["b"])
        self.assertTrue(record["predicted_ungrounded_response"])

        by_key = {s["response_sentence_key"]: s for s in record["sentence_scores"]}
        self.assertTrue(by_key["a"]["predicted_supported"])
        self.assertTrue(by_key["a"]["gold_supported"])
        self.assertFalse(by_key["b"]["predicted_supported"])
        self.assertFalse(by_key["b"]["gold_supported"])
        self.assertEqual(by_key["a"]["best_document_sentence_key"], "0a")

    def test_response_granularity_scores_whole_response(self):
        record = adapter.score_row(ragbench_row(), 0, 75.0, "d", "c", "s", granularity="response")
        self.assertEqual(len(record["sentence_scores"]), 1)
        self.assertEqual(record["sentence_scores"][0]["response_sentence_key"], "whole")

    def test_rejects_empty_response_or_question(self):
        row = ragbench_row()
        row["response"] = ""
        self.assertIsNone(adapter.score_row(row, 0, 75.0, "d", "c", "s"))
        row = ragbench_row()
        row["question"] = ""
        self.assertIsNone(adapter.score_row(row, 0, 75.0, "d", "c", "s"))


class TestChooseSplits(unittest.TestCase):
    def test_non_datasetdict_passthrough(self):
        rows = [{"x": 1}]
        self.assertEqual(adapter.choose_splits(rows, "test"), [("test", rows)])

    def test_datasetdict_auto_prefers_test(self):
        from datasets import Dataset, DatasetDict

        dd = DatasetDict(
            {
                "train": Dataset.from_list([{"x": 1}]),
                "test": Dataset.from_list([{"x": 2}]),
            }
        )
        splits = adapter.choose_splits(dd, "auto")
        self.assertEqual([name for name, _ in splits], ["test"])
        self.assertEqual(len(adapter.choose_splits(dd, "all")), 2)
        with self.assertRaises(SystemExit):
            adapter.choose_splits(dd, "nope")


if __name__ == "__main__":
    unittest.main()
