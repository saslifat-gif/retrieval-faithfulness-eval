# RAG Grounding & Retrieval-Faithfulness

Tools for measuring whether an LLM answer is actually **grounded in retrieved
evidence**, or whether the model substituted its own parametric knowledge while
still looking correct.

The repository has two parts:

- **`grounding/` — the product.** A sentence-level grounding instrument for RAG
  answers: given a question, the retrieved documents, and a generated response,
  it flags response sentences that are not supported by the documents. This is
  the practical, deployable piece and maps directly onto production RAG traffic.
- **`benchmark/` — the research benchmark and entry point.** The MuSiQue
  retrieval-faithfulness harness: it runs a `smolagents` multi-hop QA agent over
  answerable MuSiQue questions and detects *unverified substitution* — the agent
  reaching a correct answer through prior knowledge instead of retrieval. This is
  where the faithfulness construct is defined, exercised end-to-end, and validated
  against human labels.
- **`validation/` — shared metric-validation track.** `label.py` / `validate.py`
  measure whether either detector actually agrees with human judgment
  (precision / recall / F1 against gold labels), so the headline numbers are
  trustworthy rather than just plausible.

This is evaluation only. There is no corruption injection, RL, or training.

```text
.
├── grounding/        ★ RAG grounding instrument (the product)
│   └── adapter.py        RAGBench/DelucionQA adapter + lexical/embedding/nli/hhem scorers
├── benchmark/        MuSiQue retrieval-faithfulness harness (research / entry point)
│   ├── load_data.py      sample answerable MuSiQue questions
│   ├── run_agent.py      run the CodeAgent, capture traces
│   ├── extract_hops.py   recover stated hop conclusions post-hoc
│   ├── score.py          score final / intermediate / retrieval / leakage
│   ├── analyze.py        aggregate into stage1_summary.json
│   ├── review.py         interactive human review queue
│   └── held_out.py       build a frozen-pattern held-out set
├── validation/       shared metric-validation track
│   ├── label.py          hand-label traces for substitution ground truth
│   └── validate.py       precision/recall/F1 of a detector vs labels, + baselines
└── common/
    └── text.py           shared normalization + fuzzy matching (match_score, split_sentences)
```

Run everything **from the repository root** (data directories `traces/`,
`labels/`, `ragbench/`, `heldout/` are resolved relative to the working
directory).

## Setup

```bash
conda activate ml
pip install -r requirements.txt
```

Create `.env` locally. Do not commit it.

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Optional
MODEL_ID=deepseek/deepseek-chat
MODEL_API_KEY_ENV=DEEPSEEK_API_KEY
MODEL_API_BASE=
EXTRACTOR_MODEL_ID=deepseek/deepseek-chat
RUNS_PER_QUESTION=3
MAX_STEPS=12
TOP_K=5
MATCH_THRESHOLD=85
EXTRACTED_CONCLUSION_SOURCE_THRESHOLD=85
```

---

# `grounding/` — RAG grounding instrument (the product)

Given a RAG triple — `(question, retrieved documents, response)` — the grounding
instrument marks each response sentence supported / unsupported by checking its
best support against the documents, and rolls that up to a response-level
faithfulness signal. No gold labels are required at inference time; gold is used
only to *validate* the instrument (below).

Build scored records from RAGBench/DelucionQA and validate them:

```bash
python grounding/adapter.py
python validation/validate.py --ragbench-records ragbench/delucionqa_records.json \
                              --report ragbench/validation_report.json
```

The adapter writes only under gitignored `ragbench/`. Each record maps:

- `question` -> `query`
- `documents_sentences` -> addressable retrieved context
- `response_sentences` -> answer sentences to check
- `unsupported_response_sentence_keys` -> sentence-level gold unsupported labels
- `adherence_score` -> response-level gold label

Four scorers are pluggable via `--method` (plus `--granularity response` for
HHEM's native whole-response-vs-whole-document mode):

```bash
python grounding/adapter.py --method lexical                            # fuzzy token overlap
python grounding/adapter.py --method embedding --threshold 50           # sentence-embedding cosine
python grounding/adapter.py --method nli      --threshold 50           # NLI entailment (cross-encoder)
python grounding/adapter.py --method hhem     --threshold 50           # Vectara HHEM faithfulness
python grounding/adapter.py --method hhem --granularity response --threshold 50   # HHEM native
```

## Grounding result (full DelucionQA)

On full DelucionQA (`n=1,826`, `10,027` response sentences) against RAGBench's
LLM-judged adherence labels (gold ungrounded rate only 0.066). AUROC is the
product-relevant metric — F1 at a fixed threshold is punished by the 6.6%
positive rate even when ranking is fine:

```text
method (granularity)        best response F1   AUROC   why
lexical                          0.144          0.58    token overlap; over-flags paraphrase
embedding (cosine)               0.151          0.58    relatedness != support
nli (vanilla MNLI)               0.133          ~0.50   under-entails grounded paraphrase
hhem (sentence)                  0.158          0.65    right tool, hard granularity
hhem (response, native)          0.198          0.68    best of all — still not product-grade
```

**Finding: no affordable local method reaches product quality here.** The
progression is informative. Lexical and embedding are *similarity* measures, and
a plausible RAG hallucination is on-topic, so similarity cannot separate
"supported" from "on-topic-but-fabricated." Vanilla NLI scores grounded
paraphrase as *neutral* (median entailment 4/100 on genuinely-supported
sentences) and over-flags everything. HHEM, a purpose-built faithfulness model,
is best calibrated (grounded median 93–96) and gives the widest separation, but
still tops out at AUROC 0.68 — because DelucionQA marks a response ungrounded for
even one bad sentence, while a whole-response consistency score is dominated by
the mostly-grounded bulk (ungrounded median 89 vs grounded 96).

Both granularities cap at AUROC ~0.65–0.68. The practical implication: grounding
at this difficulty needs an **LLM judge** (which is what RAGBench's gold labels
are), with its cost/latency, or a narrower scope — a coarse "review this" triage
queue that tolerates ~0.68 ranking with a human in the loop. The cheap,
hands-off detector is not reachable on DelucionQA-grade sentence-level grounding.

> Note: this RAGBench instrument validates response-vs-document grounding. It does
> **not** validate the MuSiQue lexical substitution detector below, because
> RAGBench has no reasoning trace.

---

# `benchmark/` — MuSiQue retrieval-faithfulness harness

Standard agent evaluations usually check only final answers; this harness asks
whether a correct answer was actually reached through retrieved evidence, or by
substituting the model's parametric memory for tool use. It is the research
benchmark that defines and exercises the faithfulness construct end-to-end.

The core signal is:

```text
frac_unverified_substitution =
P(trace has unverified substitution) over all scored traces
```

The harness also reports step-faithfulness divergence:

```text
divergence_rate = P(final_correct AND any_intermediate_wrong)
```

## What it does

1. Loads answerable MuSiQue questions from Hugging Face.
2. Runs a `smolagents` `CodeAgent` with one local retrieval tool.
3. Captures full traces: retrieval calls, retrieved passages, reasoning text,
   and final answer.
4. Runs a post-hoc extractor over the trace reasoning to recover stated hop
   conclusions.
5. Scores final answers, intermediate-hop status, retrieval coverage, extraction
   health, and leakage/substitution signals.

## Run order

```bash
python benchmark/load_data.py                           # sample ~30 answerable 2/3-hop questions
python benchmark/run_agent.py --runs 3 --max-steps 12   # run agent, capture traces
python benchmark/extract_hops.py --reextract            # recover stated hop conclusions
python benchmark/score.py                               # score traces
python benchmark/analyze.py                             # aggregate metrics
```

Traces are independent and API-bound, so they parallelize across processes.
`--workers N` fans out N traces at once (near-linear speedup until the model's
rate limit); `MODEL_TIMEOUT` caps a single hung call so it can't stall the batch:

```bash
MODEL_TIMEOUT=90 python benchmark/run_agent.py --runs 3 --max-steps 12 --workers 6
```

Smoke test (single question, single run):

```bash
python benchmark/run_agent.py --limit 1 --runs 1 --max-steps 12
```

Review candidate traces manually, then re-aggregate:

```bash
python benchmark/review.py
python benchmark/analyze.py
```

Run `review.py` from an activated shell, not through `conda run`, because it
requires interactive keyboard input.

## Metrics

The analyzer reports:

- `final_accuracy`
- `divergence_rate`: correct final answer with at least one stated wrong hop
- `frac_any_intermediate_wrong`
- `frac_any_extraction_failed`
- `frac_unverified_substitution`
- `frac_verified_prior`
- `frac_leakage_phrase_noise`
- `extraction_health`: hit / wrong / extraction_failed counts
- `final_answer_status_counts`: answered / non_final_output
- `leakage_type_counts`: flag-level counts by leakage type

Important distinction:

- `wrong`: the agent stated a conclusion and it did not match gold.
- `extraction_failed`: no conclusion was recovered; this is instrument health,
  not an agent reasoning failure.
- `unverified_substitution`: the trace indicates the agent used knowledge not
  supported by retrieved passages.
- `verified_prior`: the agent appears to know something early but later verifies
  it through retrieval.
- `phrase_match_noise`: a leakage phrase matched but is not meaningful leakage.

Leakage detection is heuristic. It uses lexical cues, then classifies each fired
flag as `unverified_substitution`, `verified_prior`, or `phrase_match_noise`. The
classifier is **precision-favoring**: only *assertion* cues — the agent claiming a
fact from prior knowledge ("I know from general background…", "strongly associated
with…", "common knowledge") — count as substitution. *Absence-of-evidence* phrasing
("the passages don't contain X", "retrieval isn't returning specific info") is
deliberately treated as noise, because noticing the evidence is insufficient is
faithful behavior, not substitution — flagging it would invert the construct and
penalize honesty. The flags are a review queue and aggregate signal, validated
against human labels (see *Validating the detector*), not a ground-truth label on
their own.

## Current small validation result

After source-validated extraction on 15 traces, the current output is:

```text
trace_count = 15
final_accuracy = 0.933
divergence_rate = 0.067
frac_any_intermediate_wrong = 0.067
frac_any_extraction_failed = 0.200
frac_any_retrieval_missed = 0.000
frac_parametric_leakage_signal = 0.467
frac_unverified_substitution = 0.267
frac_verified_prior = 0.400
frac_leakage_phrase_noise = 0.600

leakage_type_counts:
  unverified_substitution: 11
  verified_prior: 12
  phrase_match_noise: 18
```

The fractions are per-trace. `leakage_type_counts` are flag-level counts. These
numbers are validation signals, not stable benchmark claims.

---

# `validation/` — validating the detectors

The scoring pipeline produces a substitution *number*; this track measures whether
that number is *trustworthy* — whether the detector agrees with human judgment.
Stdlib only, no model calls.

```bash
python validation/label.py        # interactive: hand-label each trace y/n for substitution
python validation/validate.py     # precision / recall / F1 of the detector vs labels, + baselines
```

- Gold labels live in `labels/labels.json`, a **separate gitignored store** keyed
  by `"{id}::{run_index}"`. They are never written into trace files, so re-running
  the pipeline cannot clobber them. Each label records a `reasoning_hash`; if the
  agent's reasoning changes, `validate.py` marks the label stale and excludes it.
  Re-running only the *detector* (`extract_hops`/`score`) does not invalidate
  labels — that is the point of validating predictions against fixed human truth.
- `validate.py` reports the detector against two bars: **majority-class accuracy**
  and a **raw-lexical-flag baseline**. If it doesn't beat the lexical baseline,
  the classifier adds nothing over regex.

## Held-out generalization

Tuning patterns on a set and validating on that same set is in-sample. To check
the metric *generalizes*, build a fresh set with a different seed and validate
with the patterns frozen:

```bash
python benchmark/held_out.py --seed <new-seed> --runs 3 --max-steps 12 --workers 6
python validation/label.py    --traces-dir heldout/traces --labels heldout/labels.json --blind
python validation/validate.py --traces-dir heldout/traces --labels heldout/labels.json \
                              --report heldout/validation_report.json
```

`held_out.py` writes only under `heldout/` and refuses the tuned default seed.
Label held-out sets `--blind` so the detector's prediction cannot anchor the
human judgment. A large recall drop here means the patterns overfit rather than
capture the construct.

## Detector validation result

The substitution detector is **precision-favoring**: when it flags a trace, that
is almost always a real prior-knowledge assertion. Measured against blind human
labels:

```text
                 precision  recall   f1    accuracy   tp/fp/fn/tn
tuned (n=90)       0.952    0.909  0.930    0.967      20/1/2/67
held-out (n=90)    0.923    0.632  0.750    0.911      12/1/7/70
```

Held-out precision holds (~0.92), confirming the construct generalizes. The
held-out recall drop is a known, bounded cost: the detector reliably catches
*verbalized* substitution but misses *silent* bridges — an agent fabricating an
intermediate fact with no lexical cue (e.g. asserting "Lake County, California"
when the evidence says Oregon). Closing that gap needs a structural
ungrounded-bridge signal, not more lexical cues; a retrieval-absence signal was
tried and refuted (MuSiQue's gold bridge is almost always in the retrieved pool,
so absence is not the discriminator).

---

# Artifacts

- `questions.json`: generated sampled MuSiQue questions with gold final and
  intermediate answers, ignored by git. Do not publish this file.
- `questions.example.json`: schema-only example file safe to commit.
- `traces/trace_{id}_{run}.json`: serialized traces, ignored by git.
- `extracted_hop_conclusions`: source-validated post-hoc conclusions inside
  each trace.
- `retrieval_hits`: retrieval coverage diagnostics.
- `leakage_flags`: typed substitution / prior / noise flags.
- `stage1_summary.json`: aggregate benchmark metrics, ignored by git.
- `ragbench/`: grounding records + validation report, ignored by git.

# Notes

This harness intentionally separates model behavior from instrument failure.
Extraction gaps are reported as extraction gaps, not as wrong reasoning. Retrieved
passages are never used to mark an intermediate hop correct; retrieval coverage
is a separate diagnostic.

Current constraints:

- Dataset: answerable MuSiQue 2-hop/3-hop questions only; 4-hop questions are
  intentionally skipped.
- Agent: tested with `smolagents` and DeepSeek chat through LiteLLM.
- Retrieval: local BM25 over each question's paragraph pool; embedding retrieval
  is not implemented here.
- Operating point matters: `MAX_STEPS`, `TOP_K`, and model choice change the
  balance between final accuracy, max-step failures, and substitution behavior.
- Leakage labels should be audited before treating a run as benchmark-quality.
</content>
</invoke>
