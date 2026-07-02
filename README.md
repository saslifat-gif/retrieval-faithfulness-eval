# RAG Grounding & Retrieval-Faithfulness

**Your RAG agent gets the right answer — but did it use the retrieved evidence, or
its own memory?** Standard evals check the final answer and call it a pass. This
toolkit catches the answers that are *correct but unfaithful*: the model answered
from parametric knowledge while the retrieval was ignored, wrong, or decorative.

It measures whether an LLM answer is actually **grounded in retrieved evidence**,
at two levels — sentence-level grounding of RAG responses, and trace-level
substitution detection on a multi-hop QA agent. Evaluation only — no corruption
injection, RL, or training.

## Headline results

A blind-labeled substitution detector that holds precision out of sample, and a
grounding study showing no cheap local method clears product quality:

```text
substitution detector vs blind human labels   precision  recall    f1
  tuned   (n=90)                                 0.95     0.91     0.93
  held-out(n=90)                                 0.92     0.63     0.75

grounding on full DelucionQA (n=1,826)         best F1    AUROC
  best local method (HHEM, response-native)      0.198     0.68   ← not product-grade
```

See `stage1_summary.example.json` for a full real benchmark run (no setup needed
to read it). Detail and method tables are below.

- **`grounding/` — the product.** A sentence-level grounding instrument for RAG
  answers: given a question, retrieved documents, and a response, it flags
  response sentences not supported by the documents.
- **`benchmark/` — the research benchmark.** The MuSiQue retrieval-faithfulness
  harness: runs a `smolagents` multi-hop QA agent over answerable MuSiQue
  questions and detects *unverified substitution* — reaching a correct answer
  through prior knowledge instead of retrieval.
- **`validation/` — metric validation.** Measures whether either detector agrees
  with human judgment (precision / recall / F1 against gold labels).

```text
.
├── grounding/        ★ RAG grounding instrument (the product)
│   └── adapter.py        RAGBench/DelucionQA adapter + lexical/embedding/nli/hhem scorers
├── benchmark/        MuSiQue retrieval-faithfulness harness
│   ├── load_data.py      sample answerable MuSiQue questions
│   ├── run_agent.py      run the CodeAgent, capture traces
│   ├── extract_hops.py   recover stated hop conclusions post-hoc
│   ├── score.py          score final / intermediate / retrieval / leakage
│   ├── analyze.py        aggregate into stage1_summary.json
│   ├── review.py         interactive human review queue
│   └── held_out.py       build a frozen-pattern held-out set
├── validation/       metric-validation track
│   ├── label.py          hand-label traces for substitution ground truth
│   └── validate.py       precision/recall/F1 of a detector vs labels, + baselines
├── common/
│   └── text.py           shared normalization + fuzzy matching
└── tests/            regression tests pinning the frozen detector patterns
```

Run everything **from the repository root** (`traces/`, `labels/`, `ragbench/`,
`heldout/` are resolved relative to the working directory).

## Setup

```bash
conda activate ml
pip install -r requirements.txt
```

Create `.env` locally (do not commit):

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Optional: MODEL_ID, MODEL_API_KEY_ENV, MODEL_API_BASE, EXTRACTOR_MODEL_ID,
# RUNS_PER_QUESTION, MAX_STEPS, TOP_K, MATCH_THRESHOLD,
# EXTRACTED_CONCLUSION_SOURCE_THRESHOLD
```

Run the regression tests (offline — no models, API keys, or pytest needed).
The leakage patterns are validated against blind human labels and treated as
frozen; the suite pins their behavior so an accidental edit fails a test
instead of silently shifting the metric:

```bash
python -m unittest
```

---

# `grounding/` — RAG grounding instrument

Given a RAG triple — `(question, retrieved documents, response)` — it marks each
response sentence supported / unsupported against the documents and rolls that up
to a response-level faithfulness signal. No gold labels needed at inference time;
gold is used only to validate.

```bash
python grounding/adapter.py
python validation/validate.py --ragbench-records ragbench/delucionqa_records.json \
                              --report ragbench/validation_report.json
```

The adapter writes only under gitignored `ragbench/`. Four scorers are pluggable
via `--method` (plus `--granularity response` for HHEM's native mode):

```bash
python grounding/adapter.py --method lexical                          # fuzzy token overlap
python grounding/adapter.py --method embedding --threshold 50         # sentence-embedding cosine
python grounding/adapter.py --method nli       --threshold 50         # NLI entailment
python grounding/adapter.py --method hhem      --threshold 50         # Vectara HHEM faithfulness
python grounding/adapter.py --method hhem --granularity response --threshold 50
```

## Result (full DelucionQA)

`n=1,826`, `10,027` response sentences, gold ungrounded rate 0.066. AUROC is the
product-relevant metric — F1 at a fixed threshold is punished by the 6.6%
positive rate even when ranking is fine:

```text
method (granularity)        best response F1   AUROC
lexical                          0.144          0.58
embedding (cosine)               0.151          0.58
nli (vanilla MNLI)               0.133          ~0.50
hhem (sentence)                  0.158          0.65
hhem (response, native)          0.198          0.68
```

**Finding: no affordable local method reaches product quality here.** Similarity
measures (lexical, embedding) cannot separate "supported" from
"on-topic-but-fabricated"; vanilla NLI over-flags grounded paraphrase; HHEM is
best calibrated but caps at AUROC 0.68. Product-grade grounding at this difficulty
needs an **LLM judge** (with its cost/latency) or a narrower human-in-the-loop
triage scope.

> This RAGBench instrument validates response-vs-document grounding only. It does
> not validate the MuSiQue substitution detector below (RAGBench has no reasoning
> trace).

---

# `benchmark/` — MuSiQue retrieval-faithfulness harness

Asks whether a correct answer was reached through retrieved evidence, or by
substituting the model's parametric memory for tool use.

```text
frac_unverified_substitution = P(trace has unverified substitution) over scored traces
divergence_rate              = P(final_correct AND any_intermediate_wrong)
```

It loads answerable MuSiQue questions, runs a `smolagents` `CodeAgent` with one
local retrieval tool, captures full traces, runs a post-hoc extractor to recover
stated hop conclusions, then scores final answers, hop status, retrieval coverage,
and leakage/substitution signals.

## Run order

```bash
python benchmark/load_data.py                           # sample ~30 answerable 2/3-hop questions
python benchmark/run_agent.py --runs 3 --max-steps 12   # run agent, capture traces
python benchmark/extract_hops.py --reextract            # recover stated hop conclusions
python benchmark/score.py                               # score traces
python benchmark/analyze.py                             # aggregate metrics
```

Traces parallelize across processes. `--workers N` fans out N traces at once;
`MODEL_TIMEOUT` caps a single hung call:

```bash
MODEL_TIMEOUT=90 python benchmark/run_agent.py --runs 3 --max-steps 12 --workers 6
python benchmark/run_agent.py --limit 1 --runs 1 --max-steps 12          # smoke test
```

Review candidate traces manually, then re-aggregate (run `review.py` from an
activated shell, not `conda run` — it needs interactive input):

```bash
python benchmark/review.py
python benchmark/analyze.py
```

## Metrics

`analyze.py` reports `final_accuracy`, `divergence_rate`,
`frac_any_intermediate_wrong`, `frac_any_extraction_failed`,
`frac_unverified_substitution`, `frac_verified_prior`,
`frac_leakage_phrase_noise`, plus `extraction_health`,
`final_answer_status_counts`, and `leakage_type_counts`.

Key distinctions:

- `wrong`: agent stated a conclusion that did not match gold.
- `extraction_failed`: no conclusion recovered — instrument health, not a
  reasoning failure.
- `unverified_substitution`: agent used knowledge not supported by retrieved
  passages.
- `verified_prior`: agent knew something early but later verified it via retrieval.
- `phrase_match_noise`: a leakage phrase matched but is not meaningful leakage.

Leakage detection is heuristic and **precision-favoring**: only *assertion* cues
(claiming a fact from prior knowledge) count as substitution. *Absence-of-evidence*
phrasing ("the passages don't contain X") is treated as noise — noticing evidence
is insufficient is faithful behavior, not substitution. The flags are a review
queue and aggregate signal, validated against human labels, not ground truth on
their own.

---

# `validation/` — validating the detectors

Measures whether the substitution number is trustworthy — whether the detector
agrees with human judgment. Stdlib only, no model calls.

```bash
python validation/label.py        # interactive: hand-label each trace y/n
python validation/validate.py     # precision / recall / F1 vs labels, + baselines
```

Gold labels live in `labels/labels.json`, a separate gitignored store keyed by
`"{id}::{run_index}"`, never written into trace files. Each label records a
`reasoning_hash`; if the agent's reasoning changes, the label is marked stale and
excluded. `validate.py` reports against two bars: majority-class accuracy and a
raw-lexical-flag baseline (if it doesn't beat lexical, the classifier adds nothing
over regex).

## Held-out generalization

To check the metric generalizes, build a fresh set with a different seed and
validate with the patterns frozen:

```bash
python benchmark/held_out.py --seed <new-seed> --runs 3 --max-steps 12 --workers 6
python validation/label.py    --traces-dir heldout/traces --labels heldout/labels.json --blind
python validation/validate.py --traces-dir heldout/traces --labels heldout/labels.json \
                              --report heldout/validation_report.json
```

`held_out.py` writes only under `heldout/` and refuses the tuned default seed.

## Result

```text
                 precision  recall   f1    accuracy   tp/fp/fn/tn
tuned (n=90)       0.952    0.909  0.930    0.967      20/1/2/67
held-out (n=90)    0.923    0.632  0.750    0.911      12/1/7/70
```

Held-out precision holds (~0.92), confirming the construct generalizes. The
recall drop is a known, bounded cost: the detector catches *verbalized*
substitution but misses *silent* bridges (an agent fabricating an intermediate
fact with no lexical cue). Closing that gap needs a structural ungrounded-bridge
signal, not more lexical cues.

---

# Artifacts & notes

- `questions.json` (gitignored, do not publish) / `questions.example.json`
  (schema-only, safe to commit).
- `traces/trace_{id}_{run}.json`: serialized traces (gitignored).
- `stage1_summary.json`: aggregate benchmark metrics (gitignored);
  `stage1_summary.example.json` is a committed real n=90 run, safe to read.
- `ragbench/`: grounding records + validation report (gitignored).

Constraints: answerable MuSiQue 2-hop/3-hop only (4-hop skipped); tested with
`smolagents` + DeepSeek chat via LiteLLM; local BM25 retrieval (no embedding
retrieval); `MAX_STEPS`, `TOP_K`, and model choice shift the accuracy/substitution
balance. Audit leakage labels before treating a run as benchmark-quality.
</content>
</invoke>
