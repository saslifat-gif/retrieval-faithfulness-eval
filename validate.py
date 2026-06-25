from __future__ import annotations

import argparse
import json
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable


TRACES_DIR = Path("traces")
LABELS_PATH = Path("labels/labels.json")
REPORT_PATH = Path("validation_report.json")


def is_scored_trace(trace: dict) -> bool:
    return (
        "gold_final_answer" in trace
        and "leakage_flags" in trace
        and "has_unverified_substitution" in trace
    )


def trace_key(trace: dict) -> str:
    return f"{trace.get('id')}::{trace.get('run_index')}"


def confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """pairs of (prediction, gold)."""
    tp = sum(1 for pred, gold in pairs if pred and gold)
    fp = sum(1 for pred, gold in pairs if pred and not gold)
    fn = sum(1 for pred, gold in pairs if not pred and gold)
    tn = sum(1 for pred, gold in pairs if not pred and not gold)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(cm: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else (0.0 if (precision is not None and recall is not None) else None)
    )
    accuracy = (tp + tn) / n if n else None
    return {
        "n": n,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
        **cm,
    }


def evaluate_predictor(
    rows: list[dict], predict: Callable[[dict], bool]
) -> dict[str, Any]:
    pairs = [(predict(row["trace"]), row["gold"]) for row in rows]
    return metrics(confusion(pairs))


def fmt(value: Any) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def print_block(title: str, m: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(f"  precision={fmt(m['precision'])}  recall={fmt(m['recall'])}  "
          f"f1={fmt(m['f1'])}  accuracy={fmt(m['accuracy'])}")
    print(f"  tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}  (n={m['n']})")


def flag_classifier_report(labels: dict[str, dict], traces_by_key: dict[str, dict]) -> dict[str, Any] | None:
    """Per-flag type accuracy, when flags were graded during labeling."""
    rows = []
    for key, record in labels.items():
        current_flags = {
            (flag.get("source"), flag.get("quote")): flag.get("leakage_type")
            for flag in (traces_by_key.get(key, {}).get("leakage_flags") or [])
        }
        for grade in record.get("flag_grades", []) or []:
            predicted_type = current_flags.get(
                (grade.get("source"), grade.get("quote")),
                grade.get("predicted_type"),
            )
            rows.append((predicted_type, grade.get("gold_type")))
    if not rows:
        return None
    classes = ["unverified_substitution", "verified_prior", "phrase_match_noise"]
    correct = sum(1 for pred, gold in rows if pred == gold)
    per_class = {}
    for cls in classes:
        tp = sum(1 for pred, gold in rows if pred == cls and gold == cls)
        fp = sum(1 for pred, gold in rows if pred == cls and gold != cls)
        fn = sum(1 for pred, gold in rows if pred != cls and gold == cls)
        per_class[cls] = metrics({"tp": tp, "fp": fp, "fn": fn, "tn": len(rows) - tp - fp - fn})
    return {
        "graded_flag_count": len(rows),
        "type_accuracy": round(correct / len(rows), 3),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the substitution detector against hand labels."
    )
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--include-stale", action="store_true",
                        help="Include labels whose trace reasoning changed since labeling.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"No labels at {labels_path}. Run: python label.py")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    traces_by_key = {}
    for path in sorted(Path(args.traces_dir).glob("trace_*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        if is_scored_trace(trace):
            traces_by_key[trace_key(trace)] = trace

    rows = []
    missing = 0
    stale = 0
    for key, record in labels.items():
        trace = traces_by_key.get(key)
        if trace is None:
            missing += 1
            continue
        # gold labels describe the agent's reasoning; if reasoning changed, the
        # label may no longer apply. Predictions, by contrast, are expected to change.
        blob = "\n".join(
            [str(s.get("model_reasoning") or "") for s in trace.get("steps", [])]
            + [str(trace.get("final_answer") or "")]
        )
        if record.get("reasoning_hash") and record["reasoning_hash"] != sha1(blob.encode()).hexdigest():
            stale += 1
            if not args.include_stale:
                continue
        rows.append({"trace": trace, "gold": bool(record.get("gold_unverified_substitution"))})

    if not rows:
        raise SystemExit("No usable (label, scored-trace) pairs found.")

    n = len(rows)
    n_pos = sum(1 for r in rows if r["gold"])
    n_neg = n - n_pos
    majority_accuracy = max(n_pos, n_neg) / n

    detector = evaluate_predictor(rows, lambda t: bool(t.get("has_unverified_substitution")))
    baseline_any_flag = evaluate_predictor(rows, lambda t: bool(t.get("leakage_flags")))
    # Structural silent-bridge signal and the lexical-OR-structural combination.
    # Computed live from the two base fields so the comparison holds even if a
    # trace was scored before has_unverified_substitution_combined existed.
    structural = evaluate_predictor(rows, lambda t: bool(t.get("has_silent_substitution")))
    combined = evaluate_predictor(
        rows, lambda t: bool(t.get("has_unverified_substitution")) or bool(t.get("has_silent_substitution"))
    )
    classifier = flag_classifier_report(labels, traces_by_key)

    total_scored = len(traces_by_key)
    coverage = round(n / total_scored, 3) if total_scored else None

    report = {
        "label_count": len(labels),
        "usable_pairs": n,
        "scored_traces": total_scored,
        "label_coverage": coverage,
        "labels_missing_trace": missing,
        "stale_labels": stale,
        "stale_included": args.include_stale,
        "gold_positive": n_pos,
        "gold_negative": n_neg,
        "gold_positive_rate": round(n_pos / n, 3),
        "majority_class_accuracy": round(majority_accuracy, 3),
        "detector": detector,
        "detector_structural_only": structural,
        "detector_lexical_or_structural": combined,
        "baseline_any_lexical_flag": baseline_any_flag,
        "flag_type_classifier": classifier,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 80)
    print("SUBSTITUTION DETECTOR VALIDATION")
    print("=" * 80)
    print(f"usable label/trace pairs = {n}  (coverage {fmt(coverage)} of {total_scored} scored)")
    print(f"gold positives = {n_pos}  negatives = {n_neg}  positive_rate = {n_pos / n:.3f}")
    if missing:
        print(f"WARNING: {missing} labels reference traces not found / not scored.")
    if stale:
        word = "INCLUDED" if args.include_stale else "EXCLUDED"
        print(f"WARNING: {stale} stale labels (reasoning changed since labeling) [{word}].")

    print("\n--- bars to beat ---")
    print(f"  majority-class accuracy   = {majority_accuracy:.3f}")
    print_block("baseline: any lexical flag fired (ignores the 3-way classifier)", baseline_any_flag)

    print_block(">>> DETECTOR (lexical only): has_unverified_substitution", detector)
    print_block("    structural only: has_silent_substitution", structural)
    print_block(">>> COMBINED: lexical OR structural", combined)

    if classifier:
        print(f"\n--- flag-type classifier ({classifier['graded_flag_count']} graded flags) ---")
        print(f"  overall type accuracy = {classifier['type_accuracy']:.3f}")
        for cls, m in classifier["per_class"].items():
            print(f"  {cls:24s} precision={fmt(m['precision'])} recall={fmt(m['recall'])} f1={fmt(m['f1'])}")
    else:
        print("\n(flag-type classifier metrics: none — re-run label.py --grade-flags to populate)")

    # plain-language verdict
    print("\n--- read ---")
    if detector["n"] < 30:
        print(f"  CAUTION: only {detector['n']} labeled pairs; treat numbers as directional, not stable.")
    d_acc = detector["accuracy"]
    if d_acc is not None and d_acc <= majority_accuracy:
        print("  The detector does NOT beat always-guessing-the-majority-class. Signal is not yet usable.")
    b_f1 = baseline_any_flag["f1"]
    d_f1 = detector["f1"]
    if d_f1 is not None and b_f1 is not None and d_f1 <= b_f1:
        print("  The 3-way classifier does NOT beat raw lexical matching — it may add no value over regex.")
    # Does the structural signal earn its place? Report its marginal effect on the
    # lexical detector: recall it recovers vs. precision it costs.
    d_rec, c_rec = detector["recall"], combined["recall"]
    d_prec, c_prec = detector["precision"], combined["precision"]
    if None not in (d_rec, c_rec, d_prec, c_prec):
        rec_gain = c_rec - d_rec
        prec_cost = d_prec - c_prec
        if rec_gain > 0:
            print(f"  Structural signal recovers recall +{rec_gain:.3f} "
                  f"(catches silent bridges) at precision {prec_cost:+.3f}.")
        elif combined["fp"] > detector["fp"]:
            print(f"  Structural signal adds no recall here but costs "
                  f"{combined['fp'] - detector['fp']} false positives — not worth combining on this set.")
        else:
            print("  Structural signal changes nothing on this set (no silent bridges among labeled positives).")
    print(f"\nsaved {args.report}")


if __name__ == "__main__":
    main()
