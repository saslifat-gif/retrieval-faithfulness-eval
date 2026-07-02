"""Tests for common/text.py — shared normalization and fuzzy matching."""
import unittest

from common.text import match_score, normalize, split_sentences


class TestNormalize(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize("Hello, World!"), "hello world")

    def test_drops_articles(self):
        self.assertEqual(normalize("The Eiffel Tower is a landmark"), "eiffel tower is landmark")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize("  too   many\tspaces \n"), "too many spaces")

    def test_none_and_empty(self):
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize(""), "")

    def test_non_string_input(self):
        self.assertEqual(normalize(1889), "1889")


class TestMatchScore(unittest.TestCase):
    def test_exact_match_is_100(self):
        self.assertEqual(match_score("Paris", "paris"), 100.0)

    def test_article_and_punctuation_insensitive(self):
        self.assertEqual(match_score("The Beatles!", "Beatles"), 100.0)

    def test_empty_side_is_0(self):
        self.assertEqual(match_score("", "Paris"), 0.0)
        self.assertEqual(match_score("Paris", None), 0.0)

    def test_disjoint_strings_score_low(self):
        self.assertLess(match_score("Paris", "quantum chromodynamics"), 50.0)

    def test_token_subset_scores_high(self):
        # token_set_ratio: candidate containing the gold tokens scores 100
        self.assertEqual(match_score("the answer is Paris", "Paris"), 100.0)


class TestSplitSentences(unittest.TestCase):
    def test_basic_split(self):
        self.assertEqual(
            split_sentences("First sentence. Second one! Third?"),
            ["First sentence.", "Second one!", "Third?"],
        )

    def test_newlines_folded_before_split(self):
        self.assertEqual(
            split_sentences("Line one.\nStill sentence two. Three."),
            ["Line one.", "Still sentence two.", "Three."],
        )

    def test_empty_input(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences(None), [])

    def test_no_terminator_is_one_sentence(self):
        self.assertEqual(split_sentences("no punctuation here"), ["no punctuation here"])


if __name__ == "__main__":
    unittest.main()
