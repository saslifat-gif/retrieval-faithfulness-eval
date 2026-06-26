from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.text import match_score, normalize, split_sentences  # noqa: E402


TRACES_DIR = Path("traces")


def source_text_for(trace: dict, source: str | None) -> str:
    if not source:
        return ""
    if source == "final_answer":
        return str(trace.get("final_answer") or "")
    match = re.match(r"step_(\d+)_reasoning", source)
    if not match:
        return ""
    step_index = int(match.group(1))
    for step in trace.get("steps", []):
        if int(step.get("step_index", -1)) == step_index:
            return str(step.get("model_reasoning") or "")
    return ""


def source_support_score(conclusion: Any, source_text: Any) -> float:
    conclusion_norm = normalize(conclusion)
    source_norm = normalize(source_text)
    if not conclusion_norm or not source_norm:
        return 0.0
    if conclusion_norm in source_norm:
        return 100.0
    return float(fuzz.token_set_ratio(conclusion_norm, source_norm))


def source_supported_conclusion(trace: dict, item: dict) -> tuple[bool, float]:
    if item.get("source_supported") is not None:
        return bool(item["source_supported"]), float(item.get("source_support_score", 100.0 if item["source_supported"] else 0.0))
    score = source_support_score(item.get("text"), source_text_for(trace, item.get("source")))
    return score >= float(os.getenv("EXTRACTED_CONCLUSION_SOURCE_THRESHOLD", "85")), round(score, 2)


def classify_match(score: float, threshold: float, margin: float, high_confidence: float) -> tuple[bool, bool]:
    hit = score >= threshold
    low_confidence = threshold - margin <= score < high_confidence
    return hit, low_confidence


def parse_hop_result_lines(text: Any) -> list[str]:
    if not text:
        return []
    results = []
    for match in re.findall(r"HOP\s+RESULT\s*:\s*(.+)", str(text), flags=re.IGNORECASE):
        result = match.strip().splitlines()[0].strip()
        looks_like_passage_dump = (
            result.startswith("[{")
            or result.startswith("[{'")
            or ("'paragraph_index'" in result and "'text'" in result)
            or ('"paragraph_index"' in result and '"text"' in result)
        )
        if result and not looks_like_passage_dump:
            results.append(result)
    return results


def reasoning_conclusion_sentences(text: Any) -> list[str]:
    conclusions = []
    markers = [
        "so ",
        "therefore ",
        "the team is ",
        "the organization is ",
        "the answer is ",
        "the region is ",
        "the top result says ",
        "the passage says ",
        "confirms ",
    ]
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in markers):
            conclusions.append(sentence)
    return conclusions


def final_answer_status(final_answer: Any) -> str:
    if final_answer is None or not str(final_answer).strip():
        return "missing"
    text = str(final_answer)
    if "<code>" in text or "Calling tools:" in text or "Reached max steps" in text:
        return "non_final_output"
    return "answered"


def is_concise_answer(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text or "\n" in text:
        return False
    return len(text.split()) <= 12 and len(text) <= 120


def hop_result_candidates(trace: dict, include_final_answer_value: bool = False) -> list[dict]:
    candidates: list[dict] = []
    rejected = []
    for item in trace.get("extracted_hop_conclusions", []) or []:
        if item.get("text"):
            supported, support_score = source_supported_conclusion(trace, item)
            item["source_supported"] = supported
            item["source_support_score"] = support_score
            if not supported:
                rejected.append({**item, "reject_reason": "not_supported_by_source_text"})
                continue
            candidates.append(
                {
                    "source": f"extracted_{item.get('source')}",
                    "text": item["text"],
                }
            )
    if rejected:
        trace["rejected_extracted_hop_conclusions"] = [
            *trace.get("rejected_extracted_hop_conclusions", []),
            *rejected,
        ]
    if candidates:
        return candidates

    if include_final_answer_value and is_concise_answer(trace.get("final_answer")):
        candidates.append(
            {
                "source": "final_answer_value",
                "text": trace.get("final_answer"),
            }
        )
    for index, result in enumerate(parse_hop_result_lines(trace.get("final_answer")), start=1):
        candidates.append(
            {
                "source": f"final_answer_hop_result_{index}",
                "text": result,
            }
        )
    for step in trace.get("steps", []):
        if step.get("hop_result"):
            candidates.append(
                {
                    "source": f"hop_result_step_{step.get('step_index')}",
                    "text": step["hop_result"],
                }
            )
        for index, result in enumerate(parse_hop_result_lines(step.get("model_reasoning")), start=1):
            candidates.append(
                {
                    "source": f"reasoning_hop_result_step_{step.get('step_index')}_{index}",
                    "text": result,
                }
            )
        for index, sentence in enumerate(reasoning_conclusion_sentences(step.get("model_reasoning")), start=1):
            candidates.append(
                {
                    "source": f"reasoning_conclusion_step_{step.get('step_index')}_{index}",
                    "text": sentence,
                }
            )
    return candidates


def retrieval_candidates(trace: dict) -> list[dict]:
    candidates = []
    for step in trace.get("steps", []):
        for passage in step.get("retrieved_passages", []) or []:
            text = f"{passage.get('title', '')}\n{passage.get('text', '')}".strip()
            if text:
                candidates.append(
                    {
                        "source": f"retrieved_step_{step.get('step_index')}_rank_{passage.get('rank')}",
                        "text": text,
                    }
                )
    return candidates


def lexical_leakage_flags(trace: dict) -> list[dict]:
    patterns = [
        r"\bI know\b",
        r"\bI recall\b",
        r"\bI remember\b",
        r"\bit is known\b",
        r"\bfrom (my )?(general|world|common|standard|historical) (knowledge|background)\b",
        r"\bgeneral background\b",
        r"\bgeneral historical knowledge\b",
        r"\bcommon knowledge\b",
        r"\breal[- ]world knowledge\b",
        r"\bexternal knowledge\b",
        r"\boutside (the )?(retrieved|provided|given) (text|passages|pool)\b",
        r"\bnot (in|from) the (retrieved|provided|given) (text|passages|pool)\b",
        r"\bnot (explicitly|directly) (stated?|mentioned?|supported|linked|answered)\b",
        r"\bdoes(n't| not|n’t) explicitly (state|mention|say|contain|include)\b",
        r"\bdoes(n't| not|n’t) mention .{0,80} specifically\b",
        r"\bnot getting .{0,80}-specific information\b",
        r"\bparagraph pool (does(n't| not|n’t)|does not) contain\b",
        r"\bretriev(al|er|ed).{0,80}(does(n't| not|n’t)|does not|not|is(n't| not)|is not).{0,80}(specific|return|contain|show|find|giv)",
        r"\bmust not use\b",
        r"\bI can infer\b",
        r"\bstrongly associated with\b",
        r"\bI'll assume\b",
        r"\blikely the same person\b",
        r"\bif i had to guess\b",
        r"\bbest guess\b",
        r"\bmost plausible\b",
    ]
    flags = []
    items = [("final_answer", None, trace.get("final_answer"))]
    items.extend(
        (f"step_{step.get('step_index')}_reasoning", step.get("step_index"), step.get("model_reasoning"))
        for step in trace.get("steps", [])
    )
    for source, step_index, text in items:
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", str(text).replace("\n", " ")):
            if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns):
                flags.append({"source": source, "step_index": step_index, "quote": sentence.strip()})
    return flags


def classify_leakage_quote(quote: str) -> str:
    text = quote.lower()
    noise_patterns = [
        r"\b(avoid|not|never) using external knowledge\b",
        r"\bmust (verify|use retrieval|avoid using external knowledge)\b",
        r"\bit usually has\b",
        r"\btypically has\b",
        r"\boften has\b",
    ]
    if any(re.search(pattern, text) for pattern in noise_patterns):
        return "phrase_match_noise"

    # unverified_substitution fires only on ASSERTION cues — the agent claiming a
    # fact from prior knowledge. Absence-of-evidence / hedge phrasings ("the pool
    # doesn't contain X", "retrieval isn't returning specific info", "not explicitly
    # stated") are deliberately NOT here: noticing the evidence is insufficient is
    # faithful behavior, not substitution, and flagging it inverts the construct and
    # penalizes the model for being honest. Those quotes fall through to noise below.
    # (Held-out: this raises precision 0.76 -> ~0.92 by dropping ~7 faithful-hedge FPs,
    # at recall cost on gap-then-substitute cases that have no assertion cue.)
    unverified_patterns = [
        r"\bmust not use\b",
        r"\bbut i know\b",
        r"\bi know from (general|common|standard|historical) knowledge\b",
        r"\bfrom (my )?(general|common|standard|historical) (knowledge|background)\b",
        r"\bhistorical knowledge says\b",
        r"\bfrom what i know\b",
        r"\bgeneral background\b",
        r"\bgeneral historical knowledge\b",
        r"\bcommon knowledge\b",
        r"\bstandard knowledge\b",
        r"\bcommonly known\b",
        r"\breal[- ]world knowledge\b",
        r"\bexternal knowledge\b",
        r"\bin real life\b",
        r"\bfrom wikipedia\b",
        r"\bfrom my knowledge\b",
        r"\bi recall\b.*\b(married|died in barcelona|greyhound|toronto coach terminal|nelvana)\b",
        r"\bi remember\b",
        r"\bi'll assume\b",
        r"\blikely the same person\b",
        r"\bif i had to guess\b",
        r"\bbest guess\b",
        r"\bmost plausible\b",
        r"\bteaches at\b.*\bnot (necessarily )?that he attended\b",
        r"\bonly connection to .*manalo\b",
        r"\bstrongly associated with\b",
        r"\boff the east coast should be\b",
        r"\bpuerto rico trench is clearly the deepest point\b",
        r"\bdeepest part of that ocean\b",
    ]
    if any(re.search(pattern, text) for pattern in unverified_patterns):
        return "unverified_substitution"

    prior_patterns = [
        r"\bknown\b",
        r"\bhistorically\b",
        r"\bfrom knowledge\b",
        r"\bI know\b",
        r"\bI can infer\b",
    ]
    if any(re.search(pattern, text) for pattern in prior_patterns):
        return "verified_prior"
    return "phrase_match_noise"


def classify_leakage_flags(flags: list[dict]) -> list[dict]:
    classified = []
    for item in flags:
        quote = item.get("quote", "")
        classified.append({**item, "leakage_type": classify_leakage_quote(quote)})
    return classified


def merge_leakage_flags(existing: list[dict], lexical: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for item in [*(existing or []), *(lexical or [])]:
        key = (item.get("source"), item.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def best_match(gold: str, candidates: list[dict]) -> dict:
    best = {"score": 0.0, "source": None, "matched_text": None}
    for candidate in candidates:
        score = match_score(candidate["text"], gold)
        if score > best["score"]:
            best = {
                "score": round(score, 2),
                "source": candidate["source"],
                "matched_text": candidate["text"],
            }
    return best


def intermediate_status(hit: bool, best: dict) -> str:
    source = best.get("source")
    matched_text = best.get("matched_text")
    has_extracted_conclusion = bool(source) and bool(str(matched_text or "").strip())
    if not has_extracted_conclusion:
        return "extraction_failed"
    return "hit" if hit else "wrong"


def silent_substitution_hops(intermediate_hits: list[dict], retrieval_hits: list[dict]) -> list[dict]:
    """Structural signature of a *silent* parametric bridge: the agent stated the
    correct intermediate answer (`status == "hit"`) for a hop whose gold answer
    retrieval never surfaced (`retrieval_hit == False`). A correct stated bridge
    that the evidence pool did not supply must have come from parametric knowledge
    — caught with no lexical cue required.

    This respects the invariant that retrieval never marks a hop correct: hop
    correctness still comes only from the stated conclusion; we use retrieval's
    *absence* as an independent substitution signal, not as evidence of a hit.
    """
    retrieval_by_index = {item.get("index"): item for item in retrieval_hits}
    hops = []
    for item in intermediate_hits:
        retrieval = retrieval_by_index.get(item.get("index"), {})
        if item.get("status") == "hit" and not retrieval.get("hit", False):
            hops.append(
                {
                    "index": item.get("index"),
                    "gold_intermediate_answer": item.get("gold_intermediate_answer"),
                    "stated_source": item.get("source"),
                    "stated_confidence": item.get("match_confidence"),
                    "retrieval_confidence": retrieval.get("match_confidence"),
                }
            )
    return hops


def score_trace(trace: dict, threshold: float, margin: float, high_confidence: float) -> dict:
    status = final_answer_status(trace.get("final_answer"))
    final_score = match_score(trace.get("final_answer"), trace.get("gold_final_answer"))
    final_correct, final_low = classify_match(final_score, threshold, margin, high_confidence)

    candidates = hop_result_candidates(trace, include_final_answer_value=status == "answered")
    evidence_candidates = retrieval_candidates(trace)
    intermediate_hits = []
    retrieval_hits = []
    for index, gold_step in enumerate(trace.get("gold_decomposition", []), start=1):
        gold_answer = gold_step.get("gold_intermediate_answer", "")
        best = best_match(gold_answer, candidates)
        hit, low = classify_match(float(best["score"]), threshold, margin, high_confidence)
        status_label = intermediate_status(hit, best)
        evidence_best = best_match(gold_answer, evidence_candidates)
        evidence_hit, evidence_low = classify_match(float(evidence_best["score"]), threshold, margin, high_confidence)
        intermediate_hits.append(
            {
                "index": index,
                "gold_intermediate_answer": gold_answer,
                "hit": hit,
                "status": status_label,
                "match_confidence": best["score"],
                "source": best["source"],
                "matched_text": best["matched_text"],
                "low_confidence": low,
            }
        )
        retrieval_hits.append(
            {
                "index": index,
                "gold_intermediate_answer": gold_answer,
                "hit": evidence_hit,
                "match_confidence": evidence_best["score"],
                "source": evidence_best["source"],
                "matched_text": evidence_best["matched_text"],
                "low_confidence": evidence_low,
            }
        )

    leakage_flags = classify_leakage_flags(
        merge_leakage_flags(trace.get("leakage_flags") or [], lexical_leakage_flags(trace))
    )
    statuses = [item["status"] for item in intermediate_hits]
    old_any_intermediate_missed = not all(item["hit"] for item in intermediate_hits)
    any_intermediate_wrong = any(item == "wrong" for item in statuses)
    any_extraction_failed = any(item == "extraction_failed" for item in statuses)

    trace["final_correct"] = final_correct
    trace["final_match_confidence"] = round(final_score, 2)
    trace["final_low_confidence"] = final_low
    trace["final_answer_status"] = status
    trace["intermediate_hits"] = intermediate_hits
    trace["intermediate_status"] = statuses
    trace["any_intermediate_wrong"] = any_intermediate_wrong
    trace["any_extraction_failed"] = any_extraction_failed
    trace["all_intermediates_hit"] = all(item == "hit" for item in statuses)
    trace["old_any_intermediate_missed_definition"] = old_any_intermediate_missed
    trace["any_intermediate_missed"] = any_intermediate_wrong
    trace["retrieval_hits"] = retrieval_hits
    trace["any_retrieval_missed"] = not all(item["hit"] for item in retrieval_hits)
    trace["leakage_flags"] = leakage_flags
    trace["has_unverified_substitution"] = any(
        item.get("leakage_type") == "unverified_substitution" for item in leakage_flags
    )
    trace["has_verified_prior"] = any(item.get("leakage_type") == "verified_prior" for item in leakage_flags)
    trace["has_leakage_phrase_noise"] = any(item.get("leakage_type") == "phrase_match_noise" for item in leakage_flags)
    trace["has_parametric_leakage_signal"] = trace["has_unverified_substitution"] or trace["has_verified_prior"]

    # Structural silent-bridge signal — independent of lexical cues. Stored as a
    # separate field so the lexical-only detector (has_unverified_substitution)
    # keeps its frozen meaning; validate.py reports lexical / structural / combined
    # side by side so promotion to the headline is an evidence-based decision.
    silent_hops = silent_substitution_hops(intermediate_hits, retrieval_hits)
    trace["silent_substitution_hops"] = silent_hops
    trace["has_silent_substitution"] = bool(silent_hops)
    trace["has_unverified_substitution_combined"] = (
        trace["has_unverified_substitution"] or trace["has_silent_substitution"]
    )

    trace["needs_review"] = final_low or any(item["low_confidence"] for item in intermediate_hits)
    return trace


def is_musique_trace(trace: dict) -> bool:
    return "gold_final_answer" in trace and "gold_decomposition" in trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("MATCH_THRESHOLD", "85")))
    parser.add_argument("--low-margin", type=float, default=float(os.getenv("LOW_CONFIDENCE_MARGIN", "10")))
    parser.add_argument("--high-confidence", type=float, default=float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "92")))
    args = parser.parse_args()

    paths = sorted(Path(args.traces_dir).glob("trace_*.json"))
    if not paths:
        raise SystemExit(f"No traces found in {args.traces_dir}")

    scored_count = 0
    skipped_count = 0
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        if not is_musique_trace(trace):
            skipped_count += 1
            continue
        trace = score_trace(trace, args.threshold, args.low_margin, args.high_confidence)
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        scored_count += 1
    print(f"Scored {scored_count} MuSiQue traces; skipped {skipped_count} other traces")


if __name__ == "__main__":
    main()
