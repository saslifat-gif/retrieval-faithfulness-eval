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

Leakage detection is heuristic. It uses lexical cues such as "I know",
"historically", "from my knowledge", or "not in the provided passages", then
separates obvious phrase-match noise from stronger substitution language. The
flags are meant to create a review queue and aggregate signal, not a final
ground-truth label without spot checks.

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
