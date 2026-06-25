from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".hf_cache/datasets").resolve()))

from datasets import DatasetDict, load_dataset

from score import match_score, split_sentences


OUT_DIR = Path("ragbench")
DEFAULT_RECORDS_PATH = OUT_DIR / "delucionqa_records.json"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_key(value: Any) -> str:
    """Canonical sentence key. RAGBench keys are alphanumeric (doc "0a", response
    "a"), but some gold `unsupported_response_sentence_keys` carry trailing
    punctuation ("a."). Strip to alphanumerics so gold and response keys share one
    namespace; without this, period-suffixed gold keys silently never match and
    their sentences are miscounted as supported."""
    return re.sub(r"[^0-9a-z]", "", as_text(value).lower())


def key_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {normalize_key(value)} - {""}
    if isinstance(value, dict):
        return {normalize_key(key) for key in value.keys()} - {""}
    if isinstance(value, (list, tuple, set)):
        return {normalize_key(item) for item in value} - {""}
    return {normalize_key(value)} - {""}


def sentence_records(raw: Any, fallback_prefix: str, fallback_texts: list[str] | None = None) -> list[dict]:
    records: list[dict] = []

    def add(key: Any, text: Any) -> None:
        key_text = normalize_key(key)
        text_text = as_text(text)
        if key_text and text_text:
            records.append({"key": key_text, "text": text_text})

    def walk(value: Any, prefix: str) -> None:
        if value is None:
            return
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                for index, sentence in enumerate(split_sentences(value)):
                    add(f"{prefix}{index}", sentence)
                return
            walk(parsed, prefix)
            return
        if isinstance(value, dict):
            if "text" in value:
                add(value.get("key") or value.get("id") or value.get("sentence_id"), value.get("text"))
                return
            for key, text in value.items():
                add(key, text)
            return
        if isinstance(value, (list, tuple)):
            if len(value) == 2 and not isinstance(value[0], (list, tuple, dict)):
                add(value[0], value[1])
                return
            for index, item in enumerate(value):
                walk(item, f"{prefix}{index}")

    walk(raw, fallback_prefix)
    if records or not fallback_texts:
        return records

    for doc_index, text in enumerate(fallback_texts):
        for sentence_index, sentence in enumerate(split_sentences(text)):
            add(f"{doc_index}{sentence_index}", sentence)
    return records


def best_document_support(response_sentence: str, document_sentences: list[dict]) -> dict:
    best = {"score": 0.0, "document_sentence_key": None, "document_sentence": None}
    for item in document_sentences:
        score = match_score(response_sentence, item["text"])
        if score > best["score"]:
            best = {
                "score": round(score, 2),
                "document_sentence_key": item["key"],
                "document_sentence": item["text"],
            }
    return best


def score_row(row: dict, row_index: int, threshold: float, dataset: str, config: str, split: str) -> dict | None:
    response = as_text(row.get("response"))
    query = as_text(row.get("question") or row.get("query"))
    if not response or not query:
        return None

    documents = [as_text(item) for item in (row.get("documents") or []) if as_text(item)]
    document_sentences = sentence_records(row.get("documents_sentences"), "d", documents)
    response_sentences = sentence_records(row.get("response_sentences"), "r", [response])
    if not document_sentences or not response_sentences:
        return None

    gold_unsupported = key_set(row.get("unsupported_response_sentence_keys"))
    gold_adherence = as_bool(row.get("adherence_score"))
    sentence_scores = []
    predicted_unsupported = set()
    for item in response_sentences:
        best = best_document_support(item["text"], document_sentences)
        supported = best["score"] >= threshold
        if not supported:
            predicted_unsupported.add(item["key"])
        sentence_scores.append(
            {
                "response_sentence_key": item["key"],
                "response_sentence": item["text"],
                "predicted_supported": supported,
                "gold_supported": item["key"] not in gold_unsupported,
                "support_score": best["score"],
                "best_document_sentence_key": best["document_sentence_key"],
                "best_document_sentence": best["document_sentence"],
            }
        )

    return {
        "id": as_text(row.get("id") or row.get("example_id") or row.get("question_id") or f"row_{row_index}"),
        "row_index": row_index,
        "dataset": dataset,
        "config": config,
        "split": split,
        "query": query,
        "response": response,
        "adherence_score": gold_adherence,
        "gold_ungrounded_response": not gold_adherence,
        "gold_unsupported_response_sentence_keys": sorted(gold_unsupported),
        "predicted_ungrounded_response": bool(predicted_unsupported),
        "predicted_unsupported_response_sentence_keys": sorted(predicted_unsupported),
        "threshold": threshold,
        "relevance_score": row.get("relevance_score"),
        "utilization_score": row.get("utilization_score"),
        "all_utilized_sentence_keys": sorted(key_set(row.get("all_utilized_sentence_keys"))),
        "sentence_scores": sentence_scores,
    }


def choose_splits(dataset: Any, preferred_split: str) -> list[tuple[str, Any]]:
    if not isinstance(dataset, DatasetDict):
        return [(preferred_split, dataset)]
    if preferred_split == "all":
        return [(name, dataset[name]) for name in dataset.keys()]
    if preferred_split != "auto":
        if preferred_split not in dataset:
            raise SystemExit(f"Split {preferred_split!r} not found. Available: {', '.join(dataset.keys())}")
        return [(preferred_split, dataset[preferred_split])]
    for name in ("test", "validation", "dev", "train"):
        if name in dataset:
            return [(name, dataset[name])]
    first = next(iter(dataset.keys()))
    return [(first, dataset[first])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sentence-level grounding records from RAGBench/DelucionQA."
    )
    parser.add_argument("--dataset", default="galileo-ai/ragbench")
    parser.add_argument("--config", default="delucionqa")
    parser.add_argument("--split", default="all",
                        help="Dataset split to use, 'auto' for test/validation/train preference, or 'all'.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional row limit for smoke tests. 0 means all rows.")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("RAGBENCH_SUPPORT_THRESHOLD", "75")))
    parser.add_argument("--out", default=str(DEFAULT_RECORDS_PATH))
    args = parser.parse_args()

    print(f"Loading {args.dataset}/{args.config} split={args.split} ...", flush=True)
    dataset = load_dataset(args.dataset, args.config)
    splits = choose_splits(dataset, args.split)

    records = []
    for split_name, rows in splits:
        for row_index, row in enumerate(rows):
            if args.limit and len(records) >= args.limit:
                break
            record = score_row(dict(row), row_index, args.threshold, args.dataset, args.config, split_name)
            if record is not None:
                records.append(record)
        if args.limit and len(records) >= args.limit:
            break

    if not records:
        raise SystemExit("No usable RAGBench records found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {out_path} with {len(records)} records from {args.dataset}/{args.config}/{args.split} "
        f"(threshold={args.threshold:g})"
    )


if __name__ == "__main__":
    main()
