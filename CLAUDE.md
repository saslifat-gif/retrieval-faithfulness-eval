# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Stage-1 evaluation harness that measures whether a `smolagents` multi-hop QA
agent reaches correct answers **through retrieved evidence** or by substituting
its own parametric knowledge. It is evaluation only — no corruption injection, RL,
or training. The headline signal is `frac_unverified_substitution`; step-faithfulness
`divergence_rate` is a secondary diagnostic. See `README.md` for metric definitions
and the current small-`n` validation numbers.

## Setup & commands

```bash
conda activate ml                 # the expected environment
pip install -r requirements.txt
```

Requires a local `.env` (gitignored) with `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`,
`DEEPSEEK_MODEL`. Optional overrides (`MODEL_ID`, `EXTRACTOR_MODEL_ID`, `TOP_K`,
`MAX_STEPS`, `RUNS_PER_QUESTION`, `MATCH_THRESHOLD`, `EXTRACTED_CONCLUSION_SOURCE_THRESHOLD`,
…) are read from env at runtime — there is no central config file; each script
calls `os.getenv` with its own defaults.

There is no test suite or linter. The smoke test is the substitute for tests:

```bash
python run_agent.py --limit 1 --runs 1 --max-steps 12   # single question, single run
python run_agent.py --qid <id> --runs 1                  # one specific question by id
```

## The pipeline is an ordered, stateful sequence

Each stage reads and **mutates the per-trace JSON files in `traces/` in place**, adding
new keys for the next stage. Run them in this exact order; a later stage silently
produces empty/degraded output if an earlier one did not run.

1. `python load_data.py` → writes `questions.json` (samples 30 answerable MuSiQue
   2-hop/3-hop questions from HF; 4-hop intentionally skipped; seeded, default seed 1729).
2. `python run_agent.py --runs 3 --max-steps 12` → runs the `CodeAgent` per question/run,
   writes `traces/trace_{id}_{run}.json` with steps, retrieval calls, reasoning, final answer.
3. `python extract_hops.py --reextract` → LLM extractor recovers stated hop conclusions
   into `extracted_hop_conclusions`; also fills `leakage_flags`. `--reextract` forces
   re-running even if conclusions already exist.
4. `python score.py` → scores final/intermediate hits, retrieval coverage, leakage types;
   writes the scoring keys back into each trace.
5. `python analyze.py` → aggregates all scored traces into `stage1_summary.json` and prints metrics.
6. `python review.py` then re-run `python analyze.py` → optional human review queue.
   **Run `review.py` from an activated shell, not `conda run`** — it needs interactive stdin.

`questions.json`, `traces/`, `shop.db`, `stage1_summary.json`, and `.hf_cache/` are
gitignored. `questions.example.json` is the schema-only safe-to-commit example.
Never publish `questions.json` (contains gold answers).

## Metric-validation track (`label.py` → `validate.py`)

Separate from the scoring pipeline. This measures whether the headline detector
(`has_unverified_substitution`) actually agrees with human judgment — i.e. whether
the product's core number is trustworthy. Stdlib only; no model calls.

```bash
python label.py                 # interactive: hand-label each scored trace y/n for substitution
python label.py --grade-flags   # additionally grade each fired flag's type (for classifier metrics)
python validate.py              # precision/recall/F1 of the detector vs labels, + baselines to beat
```

- Run `label.py` from an **interactive shell** (it reads stdin), like `review.py`.
  It is resumable: labels save after every answer, and already-labeled traces are
  skipped unless `--relabel`.
- Gold labels live in `labels/labels.json` — a **separate store, gitignored** — keyed
  by `"{id}::{run_index}"`. They are never written back into the trace files, so
  re-running the pipeline cannot clobber them.
- Each label records a `reasoning_hash` of the agent's reasoning text. Gold describes
  *the agent's reasoning*, so if you re-run `run_agent.py` and the reasoning changes,
  `validate.py` marks that label **stale** and excludes it (override with `--include-stale`).
  Re-running only the *detector* (`extract_hops`/`score`) does not invalidate labels —
  that is the whole point of validating predictions against fixed human truth.
- `validate.py` reports the detector against two bars it must beat: **majority-class
  accuracy** and a **raw-lexical-flag baseline** (any flag fired, ignoring the 3-way
  classifier). If the detector doesn't beat the lexical baseline, the classifier adds
  no value over regex. Writes `validation_report.json`.
- Recall is only meaningful because labeling reads the **full** reasoning, not just
  fired flags — so it captures substitution the regex missed (false negatives).

**Held-out generalization** (`held_out.py`): tuning the patterns against the labeled
set and validating on that same set yields an *in-sample* number. To check the metric
generalizes, build a fresh set with a different sampling seed and validate with the
patterns frozen:

```bash
python held_out.py --seed <new-seed> --runs 3 --max-steps 12   # generate+run+extract+score into heldout/
python label.py    --traces-dir heldout/traces --labels heldout/labels.json --blind
python validate.py --traces-dir heldout/traces --labels heldout/labels.json --report heldout/validation_report.json
```

`held_out.py` writes only under `heldout/` (gitignored) so it can never touch the tuned
`traces/` or `labels/`, and it refuses `--seed 1729` (the tuned default). Label held-out
sets with `--blind` so the detector's prediction can't anchor the human judgment.
A large recall drop here means the patterns overfit the tuned traces, not the construct.
The path args added for this (`load_data.py --out`, `run_agent.py --questions/--traces-dir`)
also make the main pipeline runnable against any alternate location.

## Key design invariants — preserve these when editing

- **Model behavior is kept distinct from instrument failure.** Intermediate hops have
  three statuses: `hit`, `wrong` (agent stated a conclusion that mismatched gold), and
  `extraction_failed` (no conclusion recovered — an instrument-health signal, *not* a
  reasoning failure). Do not collapse `extraction_failed` into `wrong`.
- **Retrieved passages never mark an intermediate hop correct.** Hop correctness is
  judged only against conclusions the agent *stated*; retrieval coverage
  (`retrieval_hits` / `any_retrieval_missed`) is a separate, parallel diagnostic.
- **Extracted conclusions are source-validated before use.** In both `extract_hops.py`
  and `score.py`, a candidate conclusion is rejected unless it actually appears in the
  agent's own reasoning/answer text (fuzzy score ≥ `EXTRACTED_CONCLUSION_SOURCE_THRESHOLD`,
  default 85). Rejected ones go to `rejected_extracted_hop_conclusions`. This guards
  against the extractor inventing facts.
- **Leakage detection is two-layered and heuristic.** An LLM extractor flags leakage
  quotes plus a lexical regex pass (`lexical_leakage_flags` / `LEAKAGE_PATTERNS`).
  `classify_leakage_quote` then sorts each quote into `unverified_substitution`,
  `verified_prior`, or `phrase_match_noise`. Flags are a review queue + aggregate
  signal, not ground-truth labels. The regex pattern lists in `extract_hops.py` and
  `score.py` must be kept consistent if you change one.
- **Matching uses normalized fuzzy comparison** (`rapidfuzz.token_set_ratio` after
  lowercasing, punctuation strip, article removal), thresholded at `MATCH_THRESHOLD`
  (default 85), with a low-confidence band feeding `needs_review`.

## Agent contract (`run_agent.py`)

The agent gets exactly one tool, `retrieve(query)`, doing BM25 over **only the current
question's paragraph pool** (no web/global index). The system prompt forbids web
knowledge and asks for natural hop-by-hop reasoning. Retrieval is captured out-of-band
via the module-global `TOOL_EVENTS` list, then realigned to steps in `step_to_dict` by
parsing `retrieve(...)` calls out of each step's code action with `ast`. If you change
the tool signature or how steps are recorded, keep that query-to-passage realignment intact.
