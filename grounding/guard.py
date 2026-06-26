"""Grounding guardrail: turn the grounding scorer into a control signal for a
live RAG system.

This does NOT bring its own retriever or generator — you pass those in as
callables, so the guard sits on top of *any* pipeline (LlamaIndex, LangChain,
your own service, or benchmark/run_agent.py). It reuses the scorers in
grounding/adapter.py: there is no second copy of the grounding logic.

Two entry points:

  check(query, documents, response)      -> GroundingReport   (one-shot verdict)
  grounded_answer(query, retrieve=..., generate=...)          (the control loop)

The loop is the "improve a real system" part: generate -> score -> if the answer
is not grounded, either regenerate with feedback naming the unsupported
sentences, or abstain. Bounded by the detector's measured ceiling (AUROC ~0.68
on DelucionQA), so the honest deployment is a high-threshold abstain/triage gate,
not a hands-off filter. See README "RAG grounding instrument".

    python grounding/guard.py        # runs an offline demo (lexical, no API/models)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))          # grounding/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (for common.*)
from adapter import (  # noqa: E402
    get_embedder,
    get_hhem_model,
    get_nli_model,
    score_row,
)


DEFAULT_THRESHOLD = float(os.getenv("RAGBENCH_SUPPORT_THRESHOLD", "75"))
ABSTAIN_MESSAGE = (
    "I can't answer this confidently from the retrieved sources."
)

# A generator: (query, documents, feedback) -> response text. `feedback` is None
# on the first attempt, then a revision instruction naming unsupported sentences.
Generator = Callable[[str, list[str], "str | None"], str]
# A retriever: query -> list of document strings.
Retriever = Callable[[str], Sequence[str]]


@dataclass
class SentenceVerdict:
    text: str
    support_score: float
    supported: bool
    best_document_sentence: str | None


@dataclass
class GroundingReport:
    grounded: bool
    weakest_score: float
    threshold: float
    method: str
    sentences: list[SentenceVerdict] = field(default_factory=list)

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [s for s in self.sentences if not s.supported]

    def feedback(self) -> str:
        """A revision instruction the generator can act on."""
        if not self.unsupported:
            return ""
        bullets = "\n".join(f"- {s.text}" for s in self.unsupported)
        return (
            "Your previous answer included statements not supported by the "
            "retrieved documents. Rewrite the answer using ONLY facts present in "
            "the documents, and drop or correct these unsupported statements:\n"
            f"{bullets}"
        )


@dataclass
class GuardResult:
    answer: str
    grounded: bool
    action: str            # "answered" | "returned_best_effort" | "abstained"
    attempts: int
    report: GroundingReport


def load_models(method: str, top_k: int = 5):
    """Lazily load whatever the chosen scorer needs (mirrors adapter.main)."""
    model = nli_model = hhem_model = None
    if method in ("embedding", "nli", "hhem"):
        model = get_embedder(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    if method == "nli":
        nli_model = get_nli_model(os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small"))
    if method == "hhem":
        hhem_model = get_hhem_model(os.getenv("HHEM_MODEL", "vectara/hallucination_evaluation_model"))
    return model, nli_model, hhem_model


def check(
    query: str,
    documents: Sequence[str],
    response: str,
    *,
    method: str = "lexical",
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
    granularity: str = "sentence",
    models: tuple | None = None,
) -> GroundingReport:
    """Score one (query, documents, response) triple. Reuses adapter.score_row.

    granularity='sentence' gives per-sentence verdicts (so feedback can name the
    bad sentences) but over-flags long sentences (HHEM's 512-token premise limit).
    granularity='response' scores the whole response in one shot — more accurate
    for HHEM, at the cost of coarser feedback."""
    if not response.strip() or not query.strip():
        raise ValueError("check() needs a non-empty query and response")
    model, nli_model, hhem_model = models if models is not None else load_models(method, top_k)
    row = {"question": query, "documents": [str(d) for d in documents], "response": response}
    record = score_row(
        row, 0, threshold, "live", "live", "live",
        method, model, nli_model, top_k, hhem_model, granularity,
    )
    if record is None:
        raise ValueError("could not score this triple (empty documents or response sentences)")
    sentences = [
        SentenceVerdict(
            text=s["response_sentence"],
            support_score=s["support_score"],
            supported=s["predicted_supported"],
            best_document_sentence=s["best_document_sentence"],
        )
        for s in record["sentence_scores"]
    ]
    weakest = min((s.support_score for s in sentences), default=0.0)
    return GroundingReport(
        grounded=not record["predicted_ungrounded_response"],
        weakest_score=weakest,
        threshold=threshold,
        method=method,
        sentences=sentences,
    )


def grounded_answer(
    query: str,
    *,
    retrieve: Retriever,
    generate: Generator,
    method: str = "lexical",
    threshold: float = DEFAULT_THRESHOLD,
    max_attempts: int = 3,
    reretrieve: bool = False,
    on_fail: str = "abstain",       # "abstain" | "best"
    top_k: int = 5,
    granularity: str = "sentence",
) -> GuardResult:
    """Generate an answer, verify it is grounded, and on failure regenerate with
    feedback (and optionally re-retrieve). After `max_attempts`, either abstain or
    return the best-effort attempt. `retrieve` and `generate` are YOUR pipeline."""
    if on_fail not in ("abstain", "best"):
        raise ValueError("on_fail must be 'abstain' or 'best'")
    models = load_models(method, top_k)
    documents = list(retrieve(query))
    attempts: list[tuple[str, GroundingReport]] = []
    feedback: str | None = None

    for attempt in range(1, max_attempts + 1):
        response = generate(query, documents, feedback)
        report = check(query, documents, response, method=method, threshold=threshold,
                       top_k=top_k, granularity=granularity, models=models)
        attempts.append((response, report))
        if report.grounded:
            return GuardResult(response, True, "answered", attempt, report)
        feedback = report.feedback()
        if reretrieve:
            documents = list(retrieve(query))

    best_response, best_report = max(attempts, key=lambda ar: ar[1].weakest_score)
    if on_fail == "best":
        return GuardResult(best_response, False, "returned_best_effort", max_attempts, best_report)
    return GuardResult(ABSTAIN_MESSAGE, False, "abstained", max_attempts, best_report)


def _demo() -> None:
    """Offline demo (lexical scorer, no API keys or models). A fake generator
    first hallucinates a sentence, then fixes it once it gets feedback."""
    docs = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower was completed in 1889.",
    ]

    def retrieve(query: str) -> list[str]:
        return docs

    def generate(query: str, documents: list[str], feedback: str | None) -> str:
        if feedback is None:
            # first try: one grounded sentence + one fabricated one
            return "The Eiffel Tower is located in Paris. It is 450 meters tall."
        # after feedback: drop the unsupported claim
        return "The Eiffel Tower is located in Paris. It was completed in 1889."

    print("=== one-shot check (the hallucinated answer) ===")
    report = check("Where is the Eiffel Tower?", docs,
                   "The Eiffel Tower is located in Paris. It is 450 meters tall.")
    print(f"grounded={report.grounded}  weakest={report.weakest_score}  threshold={report.threshold}")
    for s in report.sentences:
        mark = "ok " if s.supported else "BAD"
        print(f"  [{mark}] {s.support_score:6.2f}  {s.text}")

    print("\n=== control loop (regenerate-on-ungrounded) ===")
    result = grounded_answer("Where is the Eiffel Tower?", retrieve=retrieve, generate=generate)
    print(f"action={result.action}  grounded={result.grounded}  attempts={result.attempts}")
    print(f"answer: {result.answer}")


if __name__ == "__main__":
    _demo()
