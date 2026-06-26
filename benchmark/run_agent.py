from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from smolagents import CodeAgent, LiteLLMModel, tool


QUESTIONS_PATH = Path("questions.json")
TRACES_DIR = Path("traces")
CURRENT_PARAGRAPHS: list[dict] = []
BM25_INDEX: BM25Okapi | None = None
TOKENIZED_PARAGRAPHS: list[list[str]] = []
TOOL_EVENTS: list[dict] = []


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def set_retrieval_pool(paragraphs: list[dict]) -> None:
    global CURRENT_PARAGRAPHS, BM25_INDEX, TOKENIZED_PARAGRAPHS, TOOL_EVENTS
    CURRENT_PARAGRAPHS = paragraphs
    TOKENIZED_PARAGRAPHS = [
        tokenize(f"{paragraph.get('title', '')} {paragraph.get('text', '')}") for paragraph in paragraphs
    ]
    BM25_INDEX = BM25Okapi(TOKENIZED_PARAGRAPHS)
    TOOL_EVENTS = []


@tool
def retrieve(query: str) -> str:
    """Search this question's paragraph pool and return top passages as JSON.

    Args:
        query: Search query for the current MuSiQue question's local passages.
    """
    if BM25_INDEX is None:
        return json.dumps({"error": "retrieval pool is not initialized"})

    top_k = int(os.getenv("TOP_K", "5"))
    scores = BM25_INDEX.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
    results = []
    for rank, idx in enumerate(ranked, start=1):
        paragraph = CURRENT_PARAGRAPHS[idx]
        results.append(
            {
                "rank": rank,
                "paragraph_index": paragraph.get("index", idx),
                "title": paragraph.get("title", ""),
                "text": paragraph.get("text", ""),
                "score": round(float(scores[idx]), 4),
            }
        )
    TOOL_EVENTS.append({"query": query, "passages": results})
    return json.dumps(results, ensure_ascii=False)


def make_model() -> LiteLLMModel:
    load_dotenv()
    api_key_env = os.getenv("MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY")
    api_key = os.getenv(api_key_env) or os.getenv("DEEPSEEK_API_KEY")
    model_id = os.getenv("MODEL_ID") or f"deepseek/{os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}"
    api_base = os.getenv("MODEL_API_BASE") or os.getenv("DEEPSEEK_BASE_URL") or None
    extra: dict[str, Any] = {}
    # Per-call timeout so a single hung request fails fast instead of blocking
    # the worker for litellm's 600s default. Opt-in via env to keep behavior
    # identical when unset.
    timeout = os.getenv("MODEL_TIMEOUT")
    if timeout:
        extra["timeout"] = float(timeout)
    return LiteLLMModel(model_id=model_id, api_key=api_key, api_base=api_base, **extra)


def safe_trace_id(raw_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id)[:120]


def parse_retrieve_query(code: str | None) -> str | None:
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "retrieve":
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
            for keyword in node.keywords:
                if keyword.arg == "query" and isinstance(keyword.value, ast.Constant):
                    return str(keyword.value.value)
    return None


def parse_hop_result(text: Any) -> str | None:
    if not text:
        return None
    matches = re.findall(r"HOP\s+RESULT\s*:\s*(.+)", str(text), flags=re.IGNORECASE)
    if not matches:
        return None
    result = matches[-1].strip().splitlines()[0].strip()
    looks_like_passage_dump = (
        result.startswith("[{")
        or result.startswith("[{'")
        or ("'paragraph_index'" in result and "'text'" in result)
        or ('"paragraph_index"' in result and '"text"' in result)
    )
    if looks_like_passage_dump:
        return None
    return result


def extract_hop_result(observation: Any, model_reasoning: Any) -> str | None:
    return parse_hop_result(observation) or parse_hop_result(model_reasoning)


def step_to_dict(step: Any, event_cursor: int) -> tuple[dict | None, int]:
    if not hasattr(step, "step_number") or getattr(step, "is_final_answer", False):
        return None, event_cursor

    code_action = getattr(step, "code_action", None)
    model_reasoning = getattr(step, "model_output", None)
    observation = getattr(step, "observations", None)
    retrieve_query = parse_retrieve_query(code_action)
    retrieved_passages: list[dict] = []

    if (retrieve_query or (code_action and "retrieve(" in code_action)) and event_cursor < len(TOOL_EVENTS):
        event = TOOL_EVENTS[event_cursor]
        event_cursor += 1
        retrieve_query = retrieve_query or event["query"]
        retrieved_passages = event["passages"]

    item = {
        "step_index": int(getattr(step, "step_number")),
        "retrieve_query": retrieve_query,
        "retrieved_passages": retrieved_passages,
        "model_reasoning": model_reasoning,
        "hop_result": extract_hop_result(observation, model_reasoning),
        "observation": observation,
        "code_action": code_action,
        "error": str(getattr(step, "error", None)) if getattr(step, "error", None) else None,
    }
    if any(item.get(key) for key in ["retrieve_query", "hop_result", "model_reasoning", "observation", "error"]):
        return item, event_cursor
    return None, event_cursor


def extract_steps(agent: CodeAgent) -> list[dict]:
    memory = getattr(agent, "memory", None)
    raw_steps = getattr(memory, "steps", []) if memory is not None else []
    steps: list[dict] = []
    event_cursor = 0
    for raw_step in raw_steps:
        item, event_cursor = step_to_dict(raw_step, event_cursor)
        if item is not None:
            steps.append(item)
    return steps


def run_one(question: dict, run_index: int, max_steps: int) -> dict:
    set_retrieval_pool(question["paragraphs"])
    model = make_model()
    agent = CodeAgent(
        tools=[retrieve],
        model=model,
        additional_authorized_imports=["json"],
        verbosity_level=0,
    )
    prompt = (
        "You are answering a MuSiQue multi-hop question. Use only retrieve(query) for evidence. "
        "Retrieval is scoped to this question's provided paragraph pool; do not use web knowledge. "
        "Solve the question hop by hop and state your intermediate conclusions naturally in your reasoning. "
        "Use json.loads on retrieve outputs before reading passages. Finish with only the final answer value.\n\n"
        f"Question: {question['question']}"
    )

    final_answer = None
    error = None
    try:
        result = agent.run(prompt, max_steps=max_steps)
        final_answer = getattr(result, "output", result)
    except Exception as exc:
        error = str(exc)

    steps = extract_steps(agent)
    missing_hop_results = [
        step["step_index"] for step in steps if step.get("retrieve_query") and not step.get("hop_result")
    ]
    return {
        "id": question["id"],
        "run_index": run_index,
        "question": question["question"],
        "gold_final_answer": question["gold_final_answer"],
        "gold_decomposition": question["decomposition"],
        "steps": steps,
        "final_answer": final_answer,
        "missing_hop_result_steps": missing_hop_results,
        "error": error,
    }


def run_and_write(question: dict, run_index: int, max_steps: int, path_str: str) -> str:
    """Run one trace and persist it. Top-level (picklable) so a process pool can
    dispatch it. Each worker process has its own copy of the module-global
    retrieval state (CURRENT_PARAGRAPHS / BM25_INDEX / TOOL_EVENTS), so concurrent
    traces never clobber each other's pool or interleave TOOL_EVENTS."""
    trace = run_one(question, run_index, max_steps)
    Path(path_str).write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path_str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=int(os.getenv("RUNS_PER_QUESTION", "3")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "8")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("RUN_WORKERS", "1")),
                        help="Parallel trace processes. 1 = sequential (default). "
                             "Traces are independent and API-bound, so this scales "
                             "near-linearly until the model's rate limit.")
    parser.add_argument("--qid", help="Optional single question id for smoke tests.")
    parser.add_argument("--limit", type=int, help="Optional number of questions for smoke tests.")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH),
                        help="Questions file to run (default questions.json).")
    parser.add_argument("--traces-dir", default=str(TRACES_DIR),
                        help="Where to write traces (default traces/).")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    traces_dir = Path(args.traces_dir)
    if not questions_path.exists():
        raise SystemExit(f"{questions_path} missing. Run: python load_data.py --out {questions_path}")

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.qid:
        questions = [item for item in questions if item["id"] == args.qid]
        if not questions:
            raise SystemExit(f"No question found for id={args.qid}")
    if args.limit:
        questions = questions[: args.limit]

    traces_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (question, run_index, args.max_steps,
         str(traces_dir / f"trace_{safe_trace_id(question['id'])}_{run_index}.json"))
        for question in questions
        for run_index in range(1, args.runs + 1)
    ]
    total = len(tasks)

    if args.workers <= 1:
        for count, task in enumerate(tasks, start=1):
            print(f"[{count}/{total}] {task[3]}", flush=True)
            run_and_write(*task)
        return

    # I/O-bound fan-out across processes. Process (not thread) isolation keeps the
    # module-global retrieval state per-trace; a single hung API call only stalls
    # its own worker, not the whole batch.
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"Running {total} traces across {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_and_write, *task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                future.result()
                print(f"[{count}/{total}] done {task[3]}", flush=True)
            except Exception as exc:  # keep the batch alive; surface the failure
                print(f"[{count}/{total}] FAILED {task[3]}: {exc}", flush=True)


if __name__ == "__main__":
    main()
