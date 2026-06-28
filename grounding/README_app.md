# Grounding Inspector — web GUI

A small web front-end for the sentence-level grounding scorer. It wraps the
existing `grounding/guard.py` (`check()`) — there is no second copy of the
scoring logic — and serves two pages:

- **`/`** — the Inspector. Paste a question, the retrieved documents (one per
  paragraph), and a candidate response; pick a method, granularity, and support
  threshold; see each response sentence marked supported / unsupported against
  the documents, plus an overall grounded / ungrounded verdict and (when
  something is unsupported) the revision feedback the guard would send back.
- **`/benchmarks`** — the real DelucionQA method comparison and the MuSiQue
  substitution-detector validation. Every number on this page is the real
  result from the repo README, not a placeholder.

## Run

```bash
conda activate ml
pip install -r requirements.txt        # adds fastapi + uvicorn
python grounding/server.py             # http://127.0.0.1:8000
```

Or with autoreload during development:

```bash
uvicorn grounding.server:app --reload --port 8000
```

## Methods

| Method      | Cost                                              |
|-------------|---------------------------------------------------|
| `lexical`   | Instant, offline, no model download or API key.   |
| `embedding` | Downloads a sentence-embedding model on first use.|
| `nli`       | Downloads an NLI cross-encoder on first use.       |
| `hhem`      | Downloads the Vectara HHEM model on first use.     |

The non-lexical methods are slow on the **first** call (model download), then
cached per method for the life of the process. The honest deployment is a
high-threshold abstain/triage gate, not a hands-off filter — the best local
method tops out at AUROC ~0.68 (see `/benchmarks`).

## Deploy (optional)

Any host that runs ASGI works (`uvicorn grounding.server:app`). For a public
demo, default users to `lexical` so a free CPU host stays responsive.
