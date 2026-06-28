"""FastAPI backend for the Grounding Inspector GUI.

Wraps the existing scorer in grounding/guard.py — there is no second copy of the
grounding logic here. Serves the Stitch-designed HTML and exposes one scoring
endpoint that the Inspector page calls.

    python grounding/server.py            # http://127.0.0.1:8000
    uvicorn grounding.server:app --reload # alternative

The `lexical` method is instant and needs no model download or API key; the
`embedding` / `nli` / `hhem` methods download a model on first use (the first
call is slow). Models are cached per method so later calls are fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
sys.path.insert(0, str(ROOT))                 # for common.*
sys.path.insert(0, str(ROOT / "grounding"))   # for guard / adapter

import guard  # noqa: E402  (guard.check / guard.load_models)

app = FastAPI(title="Grounding Inspector")

# Cache loaded models per method so switching back does not reload. Lexical needs
# nothing, so it never appears here.
_MODELS: dict[str, tuple] = {}


def models_for(method: str) -> tuple:
    if method == "lexical":
        return (None, None, None)
    if method not in _MODELS:
        _MODELS[method] = guard.load_models(method)
    return _MODELS[method]


def split_documents(raw: str) -> list[str]:
    """One document per paragraph (blank-line separated). Falls back to per-line
    if the user didn't leave blank lines."""
    raw = (raw or "").strip()
    if not raw:
        return []
    chunks = [c.strip() for c in raw.replace("\r\n", "\n").split("\n\n")]
    chunks = [c for c in chunks if c]
    if len(chunks) <= 1:
        chunks = [line.strip() for line in raw.split("\n") if line.strip()]
    return chunks


class CheckRequest(BaseModel):
    question: str = ""
    documents: str = ""
    response: str = ""
    method: str = "lexical"
    threshold: float = 75.0
    granularity: str = Field(default="sentence")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "inspector.html")


@app.get("/benchmarks")
def benchmarks() -> FileResponse:
    return FileResponse(WEB / "benchmarks.html")


@app.post("/api/check")
def api_check(req: CheckRequest) -> JSONResponse:
    if req.method not in ("lexical", "embedding", "nli", "hhem"):
        raise HTTPException(400, f"unknown method {req.method!r}")
    documents = split_documents(req.documents)
    if not documents:
        raise HTTPException(400, "Add at least one retrieved document.")
    try:
        report = guard.check(
            req.question,
            documents,
            req.response,
            method=req.method,
            threshold=float(req.threshold),
            granularity=req.granularity,
            models=models_for(req.method),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return JSONResponse(
        {
            "grounded": report.grounded,
            "weakest_score": report.weakest_score,
            "threshold": report.threshold,
            "method": report.method,
            "granularity": req.granularity,
            "unsupported_count": len(report.unsupported),
            "sentence_count": len(report.sentences),
            "sentences": [
                {
                    "text": s.text,
                    "support_score": s.support_score,
                    "supported": s.supported,
                    "best_document_sentence": s.best_document_sentence,
                }
                for s in report.sentences
            ],
            "feedback": report.feedback(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
