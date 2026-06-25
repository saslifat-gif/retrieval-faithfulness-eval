# MuSiQue Retrieval-Faithfulness Harness

Standard agent evaluations usually check only final answers; this harness asks
whether a correct answer was actually reached through retrieved evidence, or by
substituting the model's parametric memory for tool use.

Stage-1 evaluation harness for measuring whether a `smolagents` multi-hop QA run
uses retrieved evidence faithfully, or substitutes parametric/background
knowledge while still producing a correct final answer.

The core signal is:

```text
frac_unverified_substitution =
P(trace has unverified substitution) over all scored traces
```

The harness also reports step-faithfulness divergence:

```text
divergence_rate = P(final_correct AND any_intermediate_wrong)
```

In the current small validation run (`n=15` traces), divergence appears rare,
while unverified substitution remained visible after extractor cleanup. Treat
substitution as the main project signal and divergence as a secondary diagnostic
until larger runs stabilize the rates.

This is evaluation only. There is no corruption injection, RL, or training.

## What It Does

1. Loads answerable MuSiQue questions from Hugging Face.
2. Runs a `smolagents` `CodeAgent` with one local retrieval tool.
3. Captures full traces: retrieval calls, retrieved passages, reasoning text,
   and final answer.
4. Runs a post-hoc extractor over the trace reasoning to recover stated hop
   conclusions.
5. Scores final answers, intermediate-hop status, retrieval coverage, extraction
   health, and leakage/substitution signals.

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
against human labels (see *Validating the Metric*), not a ground-truth label on
their own.

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

## Run Order

Load and sample 30 answerable 2-hop/3-hop MuSiQue questions:

```bash
python load_data.py
```

Run the agent and capture traces:

```bash
python run_agent.py --runs 3 --max-steps 12
```

Traces are independent and API-bound, so they parallelize across processes.
`--workers N` fans out N traces at once (near-linear speedup until the model's
rate limit); `MODEL_TIMEOUT` caps a single hung call so it can't stall the batch:

```bash
MODEL_TIMEOUT=90 python run_agent.py --runs 3 --max-steps 12 --workers 6
```

Smoke test:

```bash
python run_agent.py --limit 1 --runs 1 --max-steps 12
```

Extract hop conclusions post-hoc:

```bash
python extract_hops.py --reextract
```

Score traces:

```bash
python score.py
```

Analyze:

```bash
python analyze.py
```

Review candidate traces manually:

```bash
python review.py
python analyze.py
```

Run `review.py` from an activated shell, not through `conda run`, because it
requires interactive keyboard input.

## Validating the Metric

The scoring pipeline produces a substitution *number*; this track measures whether
that number is *trustworthy* — whether the detector agrees with human judgment.
Stdlib only, no model calls.

```bash
python label.py                 # interactive: hand-label each trace y/n for substitution
python validate.py              # precision / recall / F1 of the detector vs labels, + baselines
```

- Gold labels live in `labels/labels.json`, a **separate gitignored store** keyed by
  `"{id}::{run_index}"`. They are never written into trace files, so re-running the
  pipeline cannot clobber them. Each label records a `reasoning_hash`; if the agent's
  reasoning changes, `validate.py` marks the label stale and excludes it. Re-running
  only the *detector* (`extract_hops`/`score`) does not invalidate labels — that is
  the point of validating predictions against fixed human truth.
- `validate.py` reports the detector against two bars: **majority-class accuracy** and
  a **raw-lexical-flag baseline**. If it doesn't beat the lexical baseline, the
  classifier adds nothing over regex.

### Held-out generalization

Tuning patterns on a set and validating on that same set is in-sample. To check the
metric *generalizes*, build a fresh set with a different seed and validate with the
patterns frozen:

```bash
python held_out.py --seed <new-seed> --runs 3 --max-steps 12 --workers 6
python label.py    --traces-dir heldout/traces --labels heldout/labels.json --blind
python validate.py --traces-dir heldout/traces --labels heldout/labels.json \
                   --report heldout/validation_report.json
```

`held_out.py` writes only under `heldout/` and refuses the tuned default seed. Label
held-out sets `--blind` so the detector's prediction cannot anchor the human judgment.
A large recall drop here means the patterns overfit rather than capture the construct.

## Detector Validation Result

The substitution detector is **precision-favoring**: when it flags a trace, that is
almost always a real prior-knowledge assertion. Measured against blind human labels:

```text
                 precision  recall   f1    accuracy   tp/fp/fn/tn
tuned (n=90)       0.952    0.909  0.930    0.967      20/1/2/67
held-out (n=90)    0.923    0.632  0.750    0.911      12/1/7/70
```

Held-out precision holds (~0.92), confirming the construct generalizes. The held-out
recall drop is a known, bounded cost: the detector reliably catches *verbalized*
substitution but misses *silent* bridges — an agent fabricating an intermediate fact
with no lexical cue (e.g. asserting "Lake County, California" when the evidence says
Oregon). Closing that gap needs a structural ungrounded-bridge signal, not more
lexical cues; a retrieval-absence signal was tried and refuted (MuSiQue's gold bridge
is almost always in the retrieved pool, so absence is not the discriminator).

## RAGBench Grounding Adapter

RAGBench/DelucionQA validates a different instrument: sentence-level grounding of a
final response against retrieved documents. It does **not** validate the MuSiQue
lexical substitution detector, because RAGBench has no reasoning trace.

Build scored records and validate them:

```bash
python ragbench_adapter.py
python validate.py --ragbench-records ragbench/delucionqa_records.json \
                   --report ragbench/validation_report.json
```

The adapter writes only under gitignored `ragbench/`. Each record maps:

- `question` -> `query`
- `documents_sentences` -> addressable retrieved context
- `response_sentences` -> answer sentences to check
- `unsupported_response_sentence_keys` -> sentence-level gold unsupported labels
- `adherence_score` -> response-level gold label

The cheap predictor marks a response sentence unsupported when its best fuzzy lexical
match against any document sentence falls below `RAGBENCH_SUPPORT_THRESHOLD`
(default `75`). On full DelucionQA (`n=1,826`, `10,027` response sentences), this
stdlib lexical check is not strong enough:

```text
response-level: P=0.077 R=0.742 F1=0.139 Acc=0.395
sentence-level: P=0.044 R=0.481 F1=0.080 Acc=0.743
best swept response F1=0.144; best swept sentence F1=0.085
```

The failure mode is grounded paraphrase and synthesis: RAGBench's gold labels are
LLM-judged entailment labels, while the cheap check is lexical overlap. That is a
useful negative result: DelucionQA justifies an embedding/NLI/judge upgrade for
production grounding, rather than relying on fuzzy sentence matching.

## Current Small Validation Result

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

## Artifacts

- `questions.json`: generated sampled MuSiQue questions with gold final and
  intermediate answers, ignored by git. Do not publish this file.
- `questions.example.json`: schema-only example file safe to commit.
- `traces/trace_{id}_{run}.json`: serialized traces, ignored by git.
- `extracted_hop_conclusions`: source-validated post-hoc conclusions inside
  each trace.
- `retrieval_hits`: retrieval coverage diagnostics.
- `leakage_flags`: typed substitution / prior / noise flags.
- `stage1_summary.json`: aggregate metrics, ignored by git.

## Notes

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
