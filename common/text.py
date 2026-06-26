"""Shared text-normalization and fuzzy-matching helpers.

Used by both the grounding instrument (`grounding/`) and the MuSiQue benchmark
(`benchmark/`). Kept dependency-light: rapidfuzz + stdlib only.
"""
from __future__ import annotations

import re
import string
from typing import Any

from rapidfuzz import fuzz


ARTICLES = {"a", "an", "the"}


def normalize(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in ARTICLES]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def match_score(candidate: Any, gold: Any) -> float:
    norm_candidate = normalize(candidate)
    norm_gold = normalize(gold)
    if not norm_candidate or not norm_gold:
        return 0.0
    if norm_candidate == norm_gold:
        return 100.0
    return float(fuzz.token_set_ratio(norm_candidate, norm_gold))


def split_sentences(text: Any) -> list[str]:
    if not text:
        return []
    clean = re.sub(r"\s+", " ", str(text)).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
