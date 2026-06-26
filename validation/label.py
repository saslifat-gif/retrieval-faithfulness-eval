from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TRACES_DIR = Path("traces")
LABELS_PATH = Path("labels/labels.json")

RUBRIC = """\
================================ LABELING RUBRIC ================================
You are establishing GROUND TRUTH for one question only:

  Did the agent reach ANY conclusion using outside / parametric knowledge that it
  did NOT verify against the retrieved passages?

  y = YES, unverified substitution is present somewhere in this trace.
  n = NO.

Decision guide:
  - Agent states a fact, and that fact is NOT supported by anything it retrieved,
    and it does NOT later confirm it via retrieval        -> y (substitution)
  - Agent "knows" something early but THEN verifies it with a retrieve() call
    before relying on it                                    -> n (verified prior)
  - Lexical cue fired ("historically", "known") but the fact is actually grounded
    in retrieved text, or the phrase is incidental          -> n (noise)

Judge the agent's REASONING, not whether the final answer was correct.
Read every step before deciding. The detector's flags are shown only as hints;
they are NOT the ground truth and may be wrong or incomplete.
Answer with: y / n / s (skip) / q (save & quit)
================================================================================
"""


def reasoning_blob(trace: dict) -> str:
    parts = []
    for step in trace.get("steps", []):
        parts.append(str(step.get("model_reasoning") or ""))
    parts.append(str(trace.get("final_answer") or ""))
    return "\n".join(parts)


def reasoning_hash(trace: dict) -> str:
    return hashlib.sha1(reasoning_blob(trace).encode("utf-8")).hexdigest()


def trace_key(trace: dict) -> str:
    return f"{trace.get('id')}::{trace.get('run_index')}"


def is_scored_trace(trace: dict) -> bool:
    return (
        "gold_final_answer" in trace
        and "leakage_flags" in trace
        and "has_unverified_substitution" in trace
    )


def short(text: Any, limit: int = 1600) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def ask(prompt: str, allowed: set[str]) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in allowed:
            return answer
        print(f"  enter one of: {', '.join(sorted(allowed))}")


def print_trace(trace: dict, blind: bool = False) -> None:
    print("\n" + "=" * 80)
    print(f"{trace_key(trace)}")
    print(f"Question:     {trace.get('question')}")
    print(f"Gold final:   {trace.get('gold_final_answer')}")
    print(f"Final answer: {short(trace.get('final_answer'), 400)}")
    if not blind:
        print(
            "Detector says: has_unverified_substitution="
            f"{trace.get('has_unverified_substitution')} "
            f"has_verified_prior={trace.get('has_verified_prior')} "
            f"has_phrase_noise={trace.get('has_leakage_phrase_noise')}"
        )
    print("-" * 80)
    for step in trace.get("steps", []):
        query = step.get("retrieve_query")
        n_pass = len(step.get("retrieved_passages") or [])
        header = f"[step {step.get('step_index')}]"
        if query:
            header += f"  retrieve({short(query, 80)!r}) -> {n_pass} passages"
        print(header)
        reasoning = short(step.get("model_reasoning"), 1600)
        if reasoning:
            print(f"    {reasoning}")
    flags = trace.get("leakage_flags") or []
    if flags and not blind:
        print("-" * 80)
        print(f"Detector flags ({len(flags)}) [HINTS ONLY]:")
        for flag in flags:
            print(
                f"  - {flag.get('source')} [{flag.get('leakage_type')}]: "
                f"{short(flag.get('quote'), 200)}"
            )


def grade_flags(trace: dict, blind: bool = False) -> list[dict]:
    """Optionally have the human assign the correct type to each fired flag."""
    graded = []
    types = {
        "u": "unverified_substitution",
        "v": "verified_prior",
        "n": "phrase_match_noise",
    }
    flags = trace.get("leakage_flags") or []
    if flags:
        print("\nGrade each flag's TYPE (u=unverified_substitution, "
              "v=verified_prior, n=noise, s=skip):")
    for flag in flags:
        shown_type = "?" if blind else flag.get("leakage_type")
        print(f"  flag [{shown_type}]: {short(flag.get('quote'), 200)}")
        answer = ask("    correct type? [u/v/n/s]: ", {"u", "v", "n", "s"})
        if answer == "s":
            continue
        graded.append(
            {
                "source": flag.get("source"),
                "quote": flag.get("quote"),
                "predicted_type": flag.get("leakage_type"),
                "gold_type": types[answer],
            }
        )
    return graded


def load_labels(path: Path) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_labels(path: Path, labels: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-label traces for substitution ground truth.")
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--relabel", action="store_true", help="Re-show already labeled traces.")
    parser.add_argument("--grade-flags", action="store_true", help="Also grade each fired flag's type.")
    parser.add_argument("--blind", action="store_true",
                        help="Hide detector predictions/flags while labeling (use for held-out sets).")
    parser.add_argument("--limit", type=int, help="Stop after labeling this many traces.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    labels = load_labels(labels_path)

    paths = sorted(Path(args.traces_dir).glob("trace_*.json"))
    traces = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    scored = [(p, t) for p, t in traces if is_scored_trace(t)]
    if not scored:
        raise SystemExit("No scored traces found. Run: python score.py")

    print(RUBRIC)
    labeled_this_run = 0
    total_done = sum(1 for _, t in scored if trace_key(t) in labels)
    print(f"{len(scored)} scored traces; {total_done} already labeled.\n")

    for _, trace in scored:
        key = trace_key(trace)
        existing = labels.get(key)
        current_hash = reasoning_hash(trace)
        if existing and not args.relabel:
            if existing.get("reasoning_hash") != current_hash:
                print(f"[stale] {key}: reasoning changed since labeling; use --relabel to redo.")
            continue
        if args.limit is not None and labeled_this_run >= args.limit:
            break

        print_trace(trace, blind=args.blind)
        answer = ask("\nUnverified substitution present? [y/n/s/q]: ", {"y", "n", "s", "q"})
        if answer == "q":
            break
        if answer == "s":
            continue

        record = {
            "trace_key": key,
            "id": trace.get("id"),
            "run_index": trace.get("run_index"),
            "gold_unverified_substitution": answer == "y",
            "reasoning_hash": current_hash,
            "detector_prediction": bool(trace.get("has_unverified_substitution")),
        }
        if args.grade_flags:
            record["flag_grades"] = grade_flags(trace, blind=args.blind)
        note = input("Optional note: ").strip()
        if note:
            record["note"] = note

        labels[key] = record
        save_labels(labels_path, labels)  # incremental: resumable on interrupt
        labeled_this_run += 1

    save_labels(labels_path, labels)
    print(f"\nLabeled {labeled_this_run} traces this run. Total labels: {len(labels)} -> {labels_path}")


if __name__ == "__main__":
    main()
