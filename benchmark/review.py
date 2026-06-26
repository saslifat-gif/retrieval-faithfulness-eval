from __future__ import annotations

import argparse
import json
from pathlib import Path


TRACES_DIR = Path("traces")


def short(text: str | None, limit: int = 500) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def ask_bool(prompt: str, current: bool) -> bool | None:
    default = "y" if current else "n"
    while True:
        answer = input(f"{prompt} [y/n/skip] current={default}: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        if answer in {"skip", "s", ""}:
            return None
        print("Enter y, n, or skip.")


def candidate_trace(trace: dict) -> bool:
    if "gold_final_answer" not in trace or "gold_decomposition" not in trace:
        return False
    if trace.get("needs_review"):
        return True
    return bool(trace.get("final_correct") and trace.get("any_intermediate_missed"))


def review_trace(path: Path) -> bool:
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not candidate_trace(trace):
        return False

    print("\n" + "=" * 80)
    print(path)
    print(f"Question: {trace.get('question')}")
    print(f"Gold final: {trace.get('gold_final_answer')}")
    print(f"Final answer: {trace.get('final_answer')}")
    print(f"final_correct={trace.get('final_correct')} any_intermediate_missed={trace.get('any_intermediate_missed')}")
    print(f"any_retrieval_missed={trace.get('any_retrieval_missed')} leakage={trace.get('has_parametric_leakage_signal')}")

    if trace.get("leakage_flags"):
        print("\nLeakage signals:")
        for item in trace["leakage_flags"][:5]:
            print(f"- {item.get('source')}: {short(item.get('quote'), 220)}")

    if trace.get("extracted_hop_conclusions"):
        print("\nExtracted conclusions:")
        for item in trace["extracted_hop_conclusions"][:10]:
            print(f"- {item.get('source')}: {short(item.get('text'), 180)}")

    changed = False
    for item in trace.get("intermediate_hits", []):
        print("-" * 80)
        print(f"Intermediate {item.get('index')}: {item.get('gold_intermediate_answer')}")
        print(f"auto_hit={item.get('hit')} confidence={item.get('match_confidence')} source={item.get('source')}")
        print(f"matched: {short(item.get('matched_text'))}")
        retrieval = next(
            (hit for hit in trace.get("retrieval_hits", []) if hit.get("index") == item.get("index")),
            None,
        )
        if retrieval:
            print(
                "retrieval: "
                f"hit={retrieval.get('hit')} confidence={retrieval.get('match_confidence')} "
                f"source={retrieval.get('source')}"
            )
        should_review = item.get("low_confidence") or (
            trace.get("final_correct") and trace.get("any_intermediate_missed")
        )
        if not should_review:
            continue
        corrected = ask_bool("Is this intermediate hit label correct as HIT?", bool(item.get("hit")))
        if corrected is None:
            continue
        item["hit"] = corrected
        item["reviewed"] = True
        item["low_confidence"] = False
        note = input("Optional note: ").strip()
        if note:
            item["review_note"] = note
        changed = True

    if changed:
        trace["human_reviewed"] = True
        trace["any_intermediate_missed"] = not all(item.get("hit") for item in trace.get("intermediate_hits", []))
        trace["needs_review"] = any(item.get("low_confidence") for item in trace.get("intermediate_hits", []))
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    args = parser.parse_args()

    paths = sorted(Path(args.traces_dir).glob("trace_*.json"))
    shown = 0
    for path in paths:
        if review_trace(path):
            shown += 1
    print(f"\nReview queue shown: {shown}")


if __name__ == "__main__":
    main()
