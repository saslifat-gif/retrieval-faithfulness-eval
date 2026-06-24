from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".hf_cache/datasets").resolve()))

from datasets import load_dataset
from huggingface_hub import hf_hub_download


OUTPUT_PATH = Path("questions.json")
DEFAULT_CANDIDATES = [
    ("bdsaglam/musique", "answerable", "musique_ans_v1.0_dev.jsonl"),
    ("voidful/MuSiQue", None, "musique_ans_v1.0_dev.jsonl"),
    ("dgslibisey/MuSiQue", None, "musique_ans_v1.0_dev.jsonl"),
]
DEFAULT_SPLITS = ["validation", "dev", "answerable", "train"]


def first_value(row: dict, names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_decomposition(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    items: list[dict] = []
    if isinstance(raw, dict):
        raw = raw.get("steps") or raw.get("decomposition") or raw.get("questions") or []
    if not isinstance(raw, list):
        return []

    for part in raw:
        if isinstance(part, str):
            continue
        sub_question = first_value(part, ["sub_question", "question", "q", "decomposed_question"])
        answer = first_value(part, ["gold_intermediate_answer", "answer", "a", "intermediate_answer"])
        if sub_question and answer:
            items.append(
                {
                    "sub_question": as_text(sub_question),
                    "gold_intermediate_answer": as_text(answer),
                }
            )
    return items


def normalize_paragraphs(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [{"title": "", "text": raw, "is_supporting": None}]

    if isinstance(raw, dict):
        raw = raw.get("paragraphs") or raw.get("contexts") or raw.get("context") or []
    if not isinstance(raw, list):
        return []

    paragraphs: list[dict] = []
    for index, para in enumerate(raw):
        if isinstance(para, str):
            paragraphs.append({"title": "", "text": para, "is_supporting": None})
            continue
        title = first_value(para, ["title", "heading", "wiki_title"]) or ""
        text = first_value(para, ["paragraph_text", "text", "context", "passage", "paragraph"]) or ""
        supporting = first_value(para, ["is_supporting", "supporting", "is_support"])
        if text:
            paragraphs.append(
                {
                    "index": index,
                    "title": as_text(title),
                    "text": as_text(text),
                    "is_supporting": supporting,
                }
            )
    return paragraphs


def normalize_row(row: dict, row_index: int) -> dict | None:
    answerable = first_value(row, ["answerable", "is_answerable"])
    if answerable is False:
        return None

    question = first_value(row, ["question", "question_text", "query"])
    final_answer = first_value(row, ["gold_final_answer", "answer", "gold_answer", "final_answer"])
    decomposition = normalize_decomposition(
        first_value(row, ["question_decomposition", "decomposition", "decomposed_questions", "steps"])
    )
    paragraphs = normalize_paragraphs(first_value(row, ["paragraphs", "contexts", "context"]))

    if not question or not final_answer or len(decomposition) not in {2, 3} or not paragraphs:
        return None

    return {
        "id": as_text(first_value(row, ["id", "qid", "question_id"]) or f"row_{row_index}"),
        "question": as_text(question),
        "gold_final_answer": as_text(final_answer),
        "decomposition": decomposition,
        "paragraphs": paragraphs,
    }


def load_candidate(dataset_name: str, config: str | None, split: str):
    if config:
        return load_dataset(dataset_name, config, split=split)
    return load_dataset(dataset_name, split=split)


def normalized_rows_from_dataset(dataset) -> list[dict]:
    rows: list[dict] = []
    for idx, row in enumerate(dataset):
        normalized = normalize_row(dict(row), idx)
        if normalized is not None:
            rows.append(normalized)
    return rows


def load_rows(dataset_names: list[str], config: str | None, split_names: list[str]) -> tuple[str, str, list[dict]]:
    errors: list[str] = []
    for dataset_name in dataset_names:
        for split in split_names:
            print(f"Trying datasets.load_dataset({dataset_name!r}, config={config!r}, split={split!r})", flush=True)
            try:
                dataset = load_candidate(dataset_name, config, split)
            except Exception as exc:
                errors.append(f"{dataset_name}/{split}: {type(exc).__name__}: {str(exc)[:160]}")
                continue

            rows = normalized_rows_from_dataset(dataset)
            if rows:
                return dataset_name, split, rows
            errors.append(f"{dataset_name}/{split}: loaded but no normalized 2/3-hop answerable rows")

    raise SystemExit("Could not load MuSiQue-Ans from configured candidates:\n" + "\n".join(errors))


def load_rows_from_hf_files(dataset_names: list[str]) -> tuple[str, str, list[dict]]:
    errors: list[str] = []
    for repo_id, _, filename in DEFAULT_CANDIDATES:
        if dataset_names and repo_id not in dataset_names:
            continue
        print(f"Trying HF file {repo_id}/{filename}", flush=True)
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            dataset = load_dataset(
                "json",
                data_files=local_path,
                split="train",
                cache_dir=os.environ["HF_DATASETS_CACHE"],
            )
        except Exception as exc:
            errors.append(f"{repo_id}/{filename}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        rows = normalized_rows_from_dataset(dataset)
        if rows:
            return repo_id, filename, rows
        errors.append(f"{repo_id}/{filename}: loaded but no normalized 2/3-hop answerable rows")
    raise SystemExit("Could not load MuSiQue-Ans jsonl files:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("MUSIQUE_SAMPLE_SIZE", "30")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SAMPLE_SEED", "1729")))
    parser.add_argument("--dataset", default=os.getenv("MUSIQUE_DATASET", ""))
    parser.add_argument("--config", default=os.getenv("MUSIQUE_CONFIG") or None)
    parser.add_argument("--splits", default=os.getenv("MUSIQUE_SPLITS", ",".join(DEFAULT_SPLITS)))
    args = parser.parse_args()

    dataset_names = [args.dataset] if args.dataset else [item[0] for item in DEFAULT_CANDIDATES]
    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    if os.getenv("MUSIQUE_USE_BUILDER", "0") == "1":
        try:
            dataset_name, split_name, rows = load_rows(dataset_names, args.config, split_names)
        except SystemExit as dataset_error:
            print(dataset_error, flush=True)
            print("Falling back to direct HF jsonl files parsed with datasets.load_dataset('json').", flush=True)
            dataset_name, split_name, rows = load_rows_from_hf_files(dataset_names)
    else:
        dataset_name, split_name, rows = load_rows_from_hf_files(dataset_names)

    rng = random.Random(args.seed)
    if len(rows) < args.sample_size:
        raise SystemExit(f"Only found {len(rows)} usable rows; need {args.sample_size}.")
    sample = rng.sample(rows, args.sample_size)

    OUTPUT_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {OUTPUT_PATH} with {len(sample)} questions from {dataset_name}/{split_name} "
        f"(usable 2/3-hop rows: {len(rows)}, seed: {args.seed})"
    )


if __name__ == "__main__":
    main()
