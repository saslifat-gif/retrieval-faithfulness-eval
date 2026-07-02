# Update Log

## 2026-07-02

### Added
- **`tests/` — offline regression suite (77 tests, stdlib `unittest`, no pytest /
  models / API).** Run `python -m unittest` from the repo root. Pins the behavior
  the project's headline numbers depend on:
  - the frozen leakage patterns and 3-way routing in `classify_leakage_quote`
    (assertion → `unverified_substitution`, hedge/self-instruction → noise,
    prior cues → `verified_prior`), plus `score_trace` end-to-end on synthetic
    faithful and substitution traces;
  - `common/text.py` normalization/matching, the RAGBench adapter's key
    normalization + lexical `score_row` contract (including the trailing-period
    gold-key case), `guard.py`'s check/regenerate/abstain loop, and
    `validate.py`'s confusion/metrics math.

### Fixed
- **`classify_leakage_quote` prior-cue patterns could never match.** The quote is
  lowercased before matching but `prior_patterns` contained uppercase literals
  (`\bI know\b`, `\bI can infer\b`) matched case-sensitively — so those cues were
  dead and their quotes fell through to `phrase_match_noise` instead of
  `verified_prior`. Patterns are now lowercase. The headline
  `unverified_substitution` bucket is checked *before* prior patterns, so the
  validated precision/recall numbers are unaffected; only the
  `verified_prior` / `phrase_match_noise` split (and `frac_verified_prior` /
  `frac_leakage_phrase_noise` in `analyze.py`) shifts on re-score.
- Stale `--top-k` help text in `grounding/adapter.py` still described the
  pre-max-pool "concatenate as premise" behavior.
- Untracked `.DS_Store` from git and added it to `.gitignore`.

## 2026-06-28

### Added
- **Grounding Inspector web GUI (`grounding/server.py`, `grounding/web/`).** A
  FastAPI front-end for the sentence-level grounding scorer, wrapping the existing
  `guard.check()` — no second copy of the scoring logic. Serves two pages and one
  `POST /api/check` endpoint, caching models per method:
  - **Inspector** (`/`): single JS-driven page handling empty → loading →
    grounded/ungrounded states. Paste a question, retrieved documents (one per
    paragraph), and a response; pick method / granularity / threshold; see
    per-sentence supported/unsupported verdicts, the overall banner, and the
    regenerate feedback. In `response` granularity the per-sentence "best match"
    line is hidden and the verdict reworded (a single match can't represent a
    multi-claim answer).
  - **Benchmarks** (`/benchmarks`): the real DelucionQA method-comparison and
    MuSiQue substitution-detector numbers from the README.
  - `grounding/README_app.md` documents how to run; `fastapi` + `uvicorn` added to
    `requirements.txt`.

### Fixed
- **NLI/HHEM sentence scorers over-flagged grounded claims.**
  `best_supports_nli` / `best_supports_hhem` concatenated the top-k closest document
  sentences into one premise and made a single cross-encoder call; an irrelevant
  neighbour could dilute a small NLI model into a wrong `neutral` verdict (a clearly
  supported sentence scored **6.69/100**). They now score each candidate document
  sentence separately and **max-pool**, matching the lexical/embedding scorers — same
  example **→ 99.47/100**, and `best_document_sentence` is now meaningful. Shared
  logic factored into `_topk_candidate_pairs` / `_maxpool_bests`. HHEM's native
  whole-response mode (`granularity='response'`) is unchanged.

### Known gap / next
- The `nli` and `hhem (sentence)` rows in the README / benchmarks tables predate the
  max-pool change and should be re-run to refresh (likely to improve). Other rows
  (lexical, embedding, `hhem (response)`) are unaffected.

## 2026-06-26

### Changed
- **Repository restructured to lead with the grounding instrument as the product.**
  The flat script layout is now split into focused areas, run from the repo root:
  - `grounding/` — the RAG grounding instrument (was `ragbench_adapter.py` →
    `grounding/adapter.py`).
  - `benchmark/` — the MuSiQue retrieval-faithfulness harness (`load_data.py`,
    `run_agent.py`, `extract_hops.py`, `score.py`, `analyze.py`, `review.py`,
    `held_out.py`) — research benchmark and entry point.
  - `validation/` — shared metric-validation track (`label.py`, `validate.py`).
  - `common/text.py` — shared `normalize` / `match_score` / `split_sentences`, so
    the grounding product no longer imports from the benchmark's `score.py`.
- Subpackage scripts add the repo root to `sys.path` so `from common.text import …`
  resolves regardless of CWD; `held_out.py` subprocess calls repointed to relocated
  siblings. `README.md` rewritten grounding-first.
- Removed `CLAUDE.md` from the repo; renamed `CHANGELOG.md` → `UPDATELOG.md`.

### Added
- **`grounding/guard.py` — grounding control loop.** Turns the scorer into a control
  signal for a live RAG system. Generator/retriever-agnostic (passed in as callables)
  and reuses `adapter.score_row` (no second copy of the grounding logic):
  - `check(query, documents, response) -> GroundingReport` — one-shot verdict.
  - `grounded_answer(query, retrieve=, generate=)` — generate → score → regenerate
    with feedback on ungrounded, else abstain after `max_attempts`.
  - `granularity=` selects per-sentence (feedback can name bad sentences) vs
    whole-response scoring.

### Validated
- **Selective-prediction (abstain gate) on full DelucionQA (n=1,826).** Against gold
  adherence (baseline 93.4% faithful), the abstain gate improves *delivered*
  faithfulness only marginally and at steep coverage cost (e.g. `nli` reaches 97.0%
  delivered but answers just 13% of questions). Confirms the gate's value is triage,
  not an automatic filter — consistent with the ~0.68 AUROC ceiling.
- **Regenerate mode, live (DeepSeek + HHEM response-gate, n=20 DelucionQA).** With a
  hallucination-prone generator, regenerate recovered both ungrounded answers, lifting
  coverage **90% → 100%** for ~2 extra API calls and delivering **zero** ungrounded
  answers — strictly beating pure-abstain.

### Investigated
- **Gate granularity dominates outcomes.** HHEM *sentence*-gate badly over-flags this
  generator's output (512-token premise truncation chops long grounded sentences;
  prompt-induced preambles like "Based on the documents…" tank scores) — only 40% of
  genuinely-faithful answers passed. The *response*-gate (+ a no-preamble prompt)
  fixed it: 100% of faithful answers pass while still catching real problems. Motivated
  the `granularity=` parameter on `guard.py`.

## 2026-06-25

### Added
- **Parallel trace execution** in `run_agent.py`: `--workers N` (also `RUN_WORKERS`
  env) fans out independent, API-bound traces across processes for near-linear
  speedup until the model's rate limit. Process isolation keeps the module-global
  retrieval state (`BM25_INDEX` / `TOOL_EVENTS`) per-trace, preserving the
  query-to-passage realignment. Default `1` keeps prior sequential behavior.
- **`MODEL_TIMEOUT`** env var: per-call timeout so one hung API request fails fast
  instead of stalling the whole batch (replaces LiteLLM's 600s default when set).
- **`held_out.py --workers`**: passthrough so held-out generation uses the parallel
  runner.
- **Held-out generalization validation**: ran a fresh-seed (`20260625`) held-out set
  and validated the substitution detector with frozen patterns.

### Changed
- **Substitution detector is now precision-favoring.** `classify_leakage_quote` routes
  a flag to `unverified_substitution` only on *assertion* cues (the agent claiming a
  fact from prior knowledge). *Absence-of-evidence* / hedge phrasing ("the pool doesn't
  contain X", "retrieval isn't returning specific info", "not explicitly stated") no
  longer counts as substitution — that is faithful behavior, and flagging it inverted
  the construct and penalized honesty.
  - Tuned (n=90): precision 0.875 → **0.952**, F1 0.913 → **0.930** (1 FP).
  - Held-out (n=90): precision 0.760 → **0.923** (recall 1.000 → 0.632, the chosen
    trade — every surviving flag is a real assertion).
- Extended lexical vocabulary for genuine assertion cues (`general background`,
  `strongly associated with`).
- `score.py` now re-merges current lexical regex hits with existing extractor flags on
  re-score, so new patterns take effect without re-running `extract_hops.py --reextract`.

### Investigated and rejected
- **Structural silent-bridge signal (retrieval-absence).** Hypothesis: a stated-correct
  bridge that retrieval never supplied indicates substitution. Refuted on data — every
  held-out false-negative hop had the gold bridge present in retrieval at confidence
  100. In MuSiQue the gold entity is almost always in the pool, so retrieval-absence is
  not a usable discriminator. Scaffolding (`has_silent_substitution`,
  `silent_substitution_hops`, `has_unverified_substitution_combined`, and the
  structural/combined columns in `validate.py`) remains as a documented negative
  control.

### Known gap / next
- ~7 held-out false negatives are *silent* substitutions (fabricated bridge, no lexical
  cue). Closing this needs an **ungrounded-bridge** detector: an asserted intermediate
  whose value appears in neither gold nor the retrieved passages — distinct from both
  lexical cues and retrieval-absence.
