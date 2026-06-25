from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import completion
from rapidfuzz import fuzz


TRACES_DIR = Path("traces")
LEAKAGE_PATTERNS = [
    r"\bI know\b",
    r"\bI recall\b",
    r"\bI remember\b",
    r"\bit is known\b",
    r"\bfrom (my )?(general|world|common|standard|historical) knowledge\b",
    r"\bgeneral historical knowledge\b",
    r"\bcommon knowledge\b",
    r"\breal[- ]world knowledge\b",
    r"\bexternal knowledge\b",
    r"\boutside (the )?(retrieved|provided|given) (text|passages|pool)\b",
    r"\bnot (in|from) the (retrieved|provided|given) (text|passages|pool)\b",
    r"\bnot (explicitly|directly) (stated?|mentioned?|supported|linked|answered)\b",
    r"\bdoes(n't| not|n’t) explicitly (state|mention|say|contain|include)\b",
    r"\bparagraph pool (does(n't| not|n’t)|does not) contain\b",
    r"\bmust not use\b",
    r"\bI can infer\b",
    r"\bI'll assume\b",
    r"\blikely the same person\b",
    r"\bif I had to guess\b",
    r"\bbest guess\b",
    r"\bmost plausible\b",
]


def make_model_config() -> dict:
    load_dotenv()
    api_key_env = os.getenv("MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY")
    return {
        "model": os.getenv("EXTRACTOR_MODEL_ID")
        or os.getenv("MODEL_ID")
        or f"deepseek/{os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}",
        "api_key": os.getenv(api_key_env) or os.getenv("DEEPSEEK_API_KEY"),
        "api_base": os.getenv("MODEL_API_BASE") or os.getenv("DEEPSEEK_BASE_URL") or None,
    }


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"conclusions": [], "leakage_quotes": [], "error": "no_json_object", "raw": text}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {"conclusions": [], "leakage_quotes": [], "error": str(exc), "raw": text}


def lexical_leakage_quotes(text: Any) -> list[str]:
    if not text:
        return []
    quotes = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text).replace("\n", " ")):
        if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in LEAKAGE_PATTERNS):
            quotes.append(sentence.strip())
    return quotes


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def source_support_score(conclusion: str, source_text: str) -> float:
    conclusion_norm = normalize_text(conclusion)
    source_norm = normalize_text(source_text)
    if not conclusion_norm or not source_norm:
        return 0.0
    if conclusion_norm in source_norm:
        return 100.0
    return float(fuzz.token_set_ratio(conclusion_norm, source_norm))


def split_supported_conclusions(conclusions: list[dict], source_texts: dict[str, str]) -> tuple[list[dict], list[dict]]:
    supported = []
    rejected = []
    threshold = float(os.getenv("EXTRACTED_CONCLUSION_SOURCE_THRESHOLD", "85"))
    for item in conclusions:
        source_text = source_texts.get(item["source"], "")
        score = round(source_support_score(item["text"], source_text), 2)
        item["source_support_score"] = score
        item["source_supported"] = score >= threshold
        if item["source_supported"]:
            supported.append(item)
        else:
            item["reject_reason"] = "not_supported_by_source_text"
            rejected.append(item)
    return supported, rejected


def call_extractor(text: str, question: str, config: dict) -> dict:
    prompt = (
        "Extract only conclusions the AGENT stated in this text. Do not infer new facts. "
        "A conclusion is a short answer-like claim the agent reached, such as an entity, place, date, organization, or final answer. "
        "Ignore code, retrieved passage JSON, debug prints, paragraph indices, and search queries. "
        "Also flag parametric leakage: quotes where the agent says or implies it used knowledge outside the provided retrieval. "
        "Return strict JSON with keys: conclusions (array of short strings), leakage_quotes (array of exact short quotes).\n\n"
        f"Question: {question}\n\n"
        f"Agent text:\n{text}"
    )
    response = completion(
        model=config["model"],
        api_key=config["api_key"],
        api_base=config["api_base"],
        messages=[
            {"role": "system", "content": "You are a strict JSON information extraction tool."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return extract_json(response.choices[0].message.content)


def normalize_extraction(data: dict, source: str, step_index: int | None) -> tuple[list[dict], list[dict]]:
    conclusions = []
    leakage = []
    for item in data.get("conclusions", []) or []:
        text = str(item).strip()
        if text:
            conclusions.append({"source": source, "step_index": step_index, "text": text})
    for item in data.get("leakage_quotes", []) or []:
        text = str(item).strip()
        if text:
            leakage.append({"source": source, "step_index": step_index, "quote": text})
    return conclusions, leakage


def extract_trace(path: Path, config: dict, reextract: bool) -> bool:
    trace = json.loads(path.read_text(encoding="utf-8"))
    if "gold_final_answer" not in trace or "gold_decomposition" not in trace:
        return False
    if trace.get("extracted_hop_conclusions") and not reextract:
        return True

    conclusions: list[dict] = []
    leakage: list[dict] = []
    extractor_errors: list[dict] = []
    source_texts: dict[str, str] = {}
    question = trace.get("question", "")

    for step in trace.get("steps", []):
        text = step.get("model_reasoning") or ""
        if not text:
            continue
        source = f"step_{step.get('step_index')}_reasoning"
        source_texts[source] = text
        data = call_extractor(text, question, config)
        if data.get("error"):
            extractor_errors.append({"source": source, "error": data["error"], "raw": data.get("raw")})
        step_conclusions, step_leakage = normalize_extraction(data, source, step.get("step_index"))
        conclusions.extend(step_conclusions)
        leakage.extend(step_leakage)
        leakage.extend(
            {"source": source, "step_index": step.get("step_index"), "quote": quote}
            for quote in lexical_leakage_quotes(text)
        )

    final_answer = trace.get("final_answer")
    if final_answer and not (isinstance(final_answer, str) and len(final_answer.split()) <= 12 and "\n" not in final_answer):
        source_texts["final_answer"] = str(final_answer)
        data = call_extractor(str(final_answer), question, config)
        if data.get("error"):
            extractor_errors.append({"source": "final_answer", "error": data["error"], "raw": data.get("raw")})
        final_conclusions, final_leakage = normalize_extraction(data, "final_answer", None)
        conclusions.extend(final_conclusions)
        leakage.extend(final_leakage)

    leakage.extend(
        {"source": "final_answer", "step_index": None, "quote": quote}
        for quote in lexical_leakage_quotes(final_answer)
    )

    seen_conclusions = set()
    deduped_conclusions = []
    for item in conclusions:
        key = (item["source"], item["text"].lower())
        if key not in seen_conclusions:
            seen_conclusions.add(key)
            deduped_conclusions.append(item)

    seen_leakage = set()
    deduped_leakage = []
    for item in leakage:
        key = (item["source"], item["quote"].lower())
        if key not in seen_leakage:
            seen_leakage.add(key)
            deduped_leakage.append(item)

    supported_conclusions, rejected_conclusions = split_supported_conclusions(deduped_conclusions, source_texts)

    trace["extracted_hop_conclusions"] = supported_conclusions
    trace["rejected_extracted_hop_conclusions"] = rejected_conclusions
    trace["leakage_flags"] = deduped_leakage
    trace["has_parametric_leakage_signal"] = bool(deduped_leakage)
    trace["extractor_errors"] = extractor_errors
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = make_model_config()
    paths = sorted(Path(args.traces_dir).glob("trace_*.json"))
    if args.limit:
        paths = paths[: args.limit]

    count = 0
    for path in paths:
        if extract_trace(path, config, args.reextract):
            count += 1
            print(f"extracted {path}")
    print(f"Extracted hop conclusions for {count} MuSiQue traces")


if __name__ == "__main__":
    main()
