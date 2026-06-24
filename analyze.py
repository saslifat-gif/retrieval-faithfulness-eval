from __future__ import annotations

import json
from pathlib import Path


TRACES_DIR = Path("traces")
SUMMARY_PATH = Path("stage1_summary.json")


def mean(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def main() -> None:
    paths = sorted(TRACES_DIR.glob("trace_*.json"))
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    scored = [
        trace
        for trace in traces
        if "gold_final_answer" in trace
        and "gold_decomposition" in trace
        and "final_correct" in trace
        and "any_intermediate_missed" in trace
    ]
    if not scored:
        raise SystemExit("No scored traces found. Run: python score.py")

    final_correct = [bool(trace["final_correct"]) for trace in scored]
    any_wrong = [bool(trace.get("any_intermediate_wrong", trace.get("any_intermediate_missed"))) for trace in scored]
    any_extraction_failed = [bool(trace.get("any_extraction_failed")) for trace in scored]
    divergence = [fc and wrong for fc, wrong in zip(final_correct, any_wrong)]

    crosstab = {
        "final_false_any_wrong_false": 0,
        "final_false_any_wrong_true": 0,
        "final_true_any_wrong_false": 0,
        "final_true_any_wrong_true": 0,
    }
    extraction_health = {"hit": 0, "wrong": 0, "extraction_failed": 0}
    low_confidence_count = 0
    final_answer_status_counts: dict[str, int] = {}
    retrieval_missed = []
    leakage_signals = []
    unverified_substitution = []
    verified_prior = []
    leakage_phrase_noise = []
    leakage_type_counts: dict[str, int] = {
        "unverified_substitution": 0,
        "verified_prior": 0,
        "phrase_match_noise": 0,
    }
    reclassified_extraction_gaps = 0
    for trace in scored:
        trace_any_wrong = bool(trace.get("any_intermediate_wrong", trace.get("any_intermediate_missed")))
        key = f"final_{str(bool(trace['final_correct'])).lower()}_any_wrong_{str(trace_any_wrong).lower()}"
        crosstab[key] += 1
        statuses = trace.get("intermediate_status") or [
            item.get("status", "hit" if item.get("hit") else "wrong")
            for item in trace.get("intermediate_hits", [])
        ]
        for status in statuses:
            extraction_health[status] = extraction_health.get(status, 0) + 1
        status = trace.get("final_answer_status", "unknown")
        final_answer_status_counts[status] = final_answer_status_counts.get(status, 0) + 1
        if trace.get("final_low_confidence"):
            low_confidence_count += 1
        low_confidence_count += sum(1 for item in trace.get("intermediate_hits", []) if item.get("low_confidence"))
        retrieval_missed.append(bool(trace.get("any_retrieval_missed")))
        leakage_signals.append(bool(trace.get("has_parametric_leakage_signal")))
        unverified_substitution.append(bool(trace.get("has_unverified_substitution")))
        verified_prior.append(bool(trace.get("has_verified_prior")))
        leakage_phrase_noise.append(bool(trace.get("has_leakage_phrase_noise")))
        for item in trace.get("leakage_flags", []):
            leakage_type = item.get("leakage_type", "phrase_match_noise")
            leakage_type_counts[leakage_type] = leakage_type_counts.get(leakage_type, 0) + 1
        old_any_missed = bool(trace.get("old_any_intermediate_missed_definition", trace.get("any_intermediate_missed")))
        if bool(trace["final_correct"]) and old_any_missed and not trace_any_wrong:
            reclassified_extraction_gaps += 1

    summary = {
        "trace_count": len(scored),
        "final_accuracy": mean(final_correct),
        "divergence_rate": mean(divergence),
        "frac_any_intermediate_wrong": mean(any_wrong),
        "frac_any_extraction_failed": mean(any_extraction_failed),
        "crosstab": crosstab,
        "extraction_health": extraction_health,
        "final_answer_status_counts": final_answer_status_counts,
        "frac_any_retrieval_missed": mean(retrieval_missed),
        "frac_parametric_leakage_signal": mean(leakage_signals),
        "frac_unverified_substitution": mean(unverified_substitution),
        "frac_verified_prior": mean(verified_prior),
        "frac_leakage_phrase_noise": mean(leakage_phrase_noise),
        "leakage_type_counts": leakage_type_counts,
        "divergence_with_retrieval_miss_rate": mean(
            [
                bool(trace["final_correct"])
                and bool(trace.get("any_intermediate_wrong", trace.get("any_intermediate_missed")))
                and bool(trace.get("any_retrieval_missed"))
                for trace in scored
            ]
        ),
        "divergence_with_leakage_signal_rate": mean(
            [
                bool(trace["final_correct"])
                and bool(trace.get("any_intermediate_wrong", trace.get("any_intermediate_missed")))
                and bool(trace.get("has_parametric_leakage_signal"))
                for trace in scored
            ]
        ),
        "reclassified_extraction_gaps": reclassified_extraction_gaps,
        "low_confidence_match_count": low_confidence_count,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"trace_count = {summary['trace_count']}")
    print(f"final_accuracy = {summary['final_accuracy']:.3f}")
    print(f"divergence_rate = {summary['divergence_rate']:.3f}")
    print(f"frac_any_intermediate_wrong = {summary['frac_any_intermediate_wrong']:.3f}")
    print(f"frac_any_extraction_failed = {summary['frac_any_extraction_failed']:.3f}")
    print("crosstab final_correct x any_intermediate_wrong:")
    for key, value in crosstab.items():
        print(f"  {key}: {value}")
    print("extraction_health:")
    for key in ["hit", "wrong", "extraction_failed"]:
        print(f"  {key}: {extraction_health.get(key, 0)}")
    print("final_answer_status_counts:")
    for key, value in sorted(final_answer_status_counts.items()):
        print(f"  {key}: {value}")
    print(f"frac_any_retrieval_missed = {summary['frac_any_retrieval_missed']:.3f}")
    print(f"frac_parametric_leakage_signal = {summary['frac_parametric_leakage_signal']:.3f}")
    print(f"frac_unverified_substitution = {summary['frac_unverified_substitution']:.3f}")
    print(f"frac_verified_prior = {summary['frac_verified_prior']:.3f}")
    print(f"frac_leakage_phrase_noise = {summary['frac_leakage_phrase_noise']:.3f}")
    print("leakage_type_counts:")
    for key in ["unverified_substitution", "verified_prior", "phrase_match_noise"]:
        print(f"  {key}: {leakage_type_counts.get(key, 0)}")
    print(f"divergence_with_retrieval_miss_rate = {summary['divergence_with_retrieval_miss_rate']:.3f}")
    print(f"divergence_with_leakage_signal_rate = {summary['divergence_with_leakage_signal_rate']:.3f}")
    print(f"reclassified_extraction_gaps = {reclassified_extraction_gaps}")
    print(f"low_confidence_match_count = {low_confidence_count}")
    print(f"saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
