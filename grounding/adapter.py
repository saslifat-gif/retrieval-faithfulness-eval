from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".hf_cache/datasets").resolve()))

from datasets import DatasetDict, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.text import match_score, split_sentences  # noqa: E402


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


_EMBEDDER: Any = None


def get_embedder(model_name: str) -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def best_supports_embedding(response_texts: list[str], document_sentences: list[dict], model: Any) -> list[dict]:
    """Max-pooled cosine support per response sentence. Cosine (normalized embeddings)
    is scaled to 0-100 so it shares the lexical scorer's range and the same threshold
    sweep applies. This is the entailment-aware upgrade: it sees supported paraphrase /
    synthesis that token overlap misses."""
    doc_texts = [item["text"] for item in document_sentences]
    resp_emb = model.encode(response_texts, normalize_embeddings=True, convert_to_numpy=True)
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = resp_emb @ doc_emb.T
    bests = []
    for index in range(len(response_texts)):
        best_doc = int(sims[index].argmax())
        bests.append(
            {
                "score": round(float(sims[index][best_doc]) * 100.0, 2),
                "document_sentence_key": document_sentences[best_doc]["key"],
                "document_sentence": document_sentences[best_doc]["text"],
            }
        )
    return bests


_NLI: Any = None


def get_nli_model(model_name: str) -> Any:
    global _NLI
    if _NLI is None:
        from sentence_transformers import CrossEncoder

        _NLI = CrossEncoder(model_name)
    return _NLI


def _entailment_index(nli_model: Any) -> int:
    id2label = getattr(getattr(nli_model, "model", None), "config", None)
    labels = getattr(id2label, "id2label", {}) if id2label is not None else {}
    for index, label in labels.items():
        if "entail" in str(label).lower():
            return int(index)
    return 1  # cross-encoder/nli-* convention: 0=contradiction, 1=entailment, 2=neutral


def best_supports_nli(response_texts: list[str], document_sentences: list[dict], embedder: Any,
                      nli_model: Any, top_k: int) -> list[dict]:
    """Entailment support, not similarity. Use embeddings to select the top-k closest
    document sentences as the premise, then score whether that premise *entails* the
    response sentence with an NLI cross-encoder. Entailment probability (0-100) is the
    support — this is the thing similarity scorers cannot measure: whether the context
    actually licenses the claim, vs merely being on-topic."""
    import numpy as np

    doc_texts = [item["text"] for item in document_sentences]
    resp_emb = embedder.encode(response_texts, normalize_embeddings=True, convert_to_numpy=True)
    doc_emb = embedder.encode(doc_texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = resp_emb @ doc_emb.T
    entail_idx = _entailment_index(nli_model)
    k = min(top_k, len(doc_texts))

    pairs = []
    top_keys = []
    for index in range(len(response_texts)):
        order = np.argsort(-sims[index])[:k]
        premise = " ".join(doc_texts[int(j)] for j in order)
        pairs.append((premise, response_texts[index]))
        top_keys.append(document_sentences[int(order[0])])

    probs = nli_model.predict(pairs, apply_softmax=True, show_progress_bar=False)
    bests = []
    for index in range(len(response_texts)):
        entail = float(probs[index][entail_idx])
        bests.append(
            {
                "score": round(entail * 100.0, 2),
                "document_sentence_key": top_keys[index]["key"],
                "document_sentence": top_keys[index]["text"],
            }
        )
    return bests


_HHEM: Any = None


def get_hhem_model(model_name: str) -> Any:
    global _HHEM
    if _HHEM is None:
        from transformers import AutoModelForSequenceClassification

        _HHEM = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        _HHEM.eval()
    return _HHEM


def best_supports_hhem(response_texts: list[str], document_sentences: list[dict], embedder: Any,
                       hhem_model: Any, top_k: int) -> list[dict]:
    """Purpose-built faithfulness scorer (Vectara HHEM). Same premise selection as NLI,
    but HHEM.predict returns a factual-consistency score (1 = grounded, 0 = hallucinated)
    trained for exactly 'is this generation supported by the source' — the task vanilla
    MNLI under-entails."""
    import numpy as np

    doc_texts = [item["text"] for item in document_sentences]
    resp_emb = embedder.encode(response_texts, normalize_embeddings=True, convert_to_numpy=True)
    doc_emb = embedder.encode(doc_texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = resp_emb @ doc_emb.T
    k = min(top_k, len(doc_texts))

    pairs = []
    top_keys = []
    for index in range(len(response_texts)):
        order = np.argsort(-sims[index])[:k]
        premise = " ".join(doc_texts[int(j)] for j in order)
        pairs.append((premise, response_texts[index]))
        top_keys.append(document_sentences[int(order[0])])

    scores = hhem_model.predict(pairs)
    scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
    bests = []
    for index in range(len(response_texts)):
        bests.append(
            {
                "score": round(float(scores[index]) * 100.0, 2),
                "document_sentence_key": top_keys[index]["key"],
                "document_sentence": top_keys[index]["text"],
            }
        )
    return bests


def score_row(row: dict, row_index: int, threshold: float, dataset: str, config: str, split: str,
              method: str = "lexical", model: Any = None, nli_model: Any = None, top_k: int = 5,
              hhem_model: Any = None, granularity: str = "sentence") -> dict | None:
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
    # response granularity = HHEM's native mode: score the whole response against the
    # whole document set (one call), instead of fragmenting into sentences. Only the
    # response-level metrics are meaningful here (gold sentence keys do not apply).
    response_units = [{"key": "whole", "text": response}] if granularity == "response" else response_sentences
    response_texts = [item["text"] for item in response_units]
    if method == "hhem" and granularity == "response":
        premise = "\n".join(documents) if documents else " ".join(item["text"] for item in document_sentences)
        raw = hhem_model.predict([(premise, text) for text in response_texts])
        raw = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        supports = [{"score": round(float(value) * 100.0, 2), "document_sentence_key": None,
                     "document_sentence": None} for value in raw]
    elif method == "hhem":
        supports = best_supports_hhem(response_texts, document_sentences, model, hhem_model, top_k)
    elif method == "nli":
        supports = best_supports_nli(response_texts, document_sentences, model, nli_model, top_k)
    elif method == "embedding":
        supports = best_supports_embedding(response_texts, document_sentences, model)
    else:
        supports = [best_document_support(text, document_sentences) for text in response_texts]
    sentence_scores = []
    predicted_unsupported = set()
    for item, best in zip(response_units, supports):
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
        "method": method,
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
    parser.add_argument("--method", choices=["lexical", "embedding", "nli", "hhem"], default="lexical",
                        help="Grounding scorer: lexical fuzzy match, embedding cosine, NLI entailment, "
                             "or HHEM faithfulness.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="NLI/HHEM sentence mode: embedding-closest doc sentences to concatenate as premise.")
    parser.add_argument("--granularity", choices=["sentence", "response"], default="sentence",
                        help="'response' scores the whole response vs the whole document set (HHEM native mode); "
                             "only response-level metrics are meaningful then.")
    parser.add_argument("--threshold", type=float,
                        default=float(os.getenv("RAGBENCH_SUPPORT_THRESHOLD", "75")),
                        help="Support cutoff on the 0-100 scale (sweep in validate.py finds the best).")
    parser.add_argument("--out", default=str(DEFAULT_RECORDS_PATH))
    args = parser.parse_args()

    model = None
    nli_model = None
    hhem_model = None
    if args.method in ("embedding", "nli", "hhem"):
        embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        print(f"Loading embedding model {embed_model} ...", flush=True)
        model = get_embedder(embed_model)
    if args.method == "nli":
        nli_name = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
        print(f"Loading NLI model {nli_name} ...", flush=True)
        nli_model = get_nli_model(nli_name)
    if args.method == "hhem":
        hhem_name = os.getenv("HHEM_MODEL", "vectara/hallucination_evaluation_model")
        print(f"Loading HHEM model {hhem_name} ...", flush=True)
        hhem_model = get_hhem_model(hhem_name)

    print(f"Loading {args.dataset}/{args.config} split={args.split} method={args.method} ...", flush=True)
    dataset = load_dataset(args.dataset, args.config)
    splits = choose_splits(dataset, args.split)

    records = []
    for split_name, rows in splits:
        for row_index, row in enumerate(rows):
            if args.limit and len(records) >= args.limit:
                break
            record = score_row(dict(row), row_index, args.threshold, args.dataset, args.config, split_name,
                               args.method, model, nli_model, args.top_k, hhem_model, args.granularity)
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
