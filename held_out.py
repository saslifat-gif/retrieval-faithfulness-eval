"""Build a fresh held-out evaluation set for the substitution detector.

Generates a NEW seeded MuSiQue sample, runs the agent, then extracts and scores
with the CURRENT (frozen) detector patterns — all into an isolated directory so it
never touches the tuned-on traces/ or labels/labels.json. It deliberately STOPS
before labeling: the held-out labels must be made blind (without seeing detector
predictions) for the test to mean anything.

    python held_out.py --seed 20240607 --runs 3 --max-steps 12

Then, in an interactive shell:

    python label.py    --traces-dir heldout/traces --labels heldout/labels.json --blind
    python validate.py --traces-dir heldout/traces --labels heldout/labels.json \
                       --report heldout/validation_report.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"step failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True,
                        help="Sampling seed. MUST differ from the tuned set (default 1729) to be held-out.")
    parser.add_argument("--dir", default="heldout", help="Isolated output directory (default heldout/).")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    if args.seed == 1729:
        raise SystemExit("--seed 1729 is the tuned set's default; pick a different seed for a real held-out test.")

    out = Path(args.dir)
    questions = out / "questions.json"
    traces = out / "traces"
    labels = out / "labels.json"
    report = out / "validation_report.json"
    py = sys.executable

    print("=" * 80)
    print(f"HELD-OUT SET  dir={out}  seed={args.seed}  n={args.sample_size}  runs={args.runs}")
    print("Patterns are frozen: extract/score run the current code, no re-tuning.")
    print("=" * 80)

    run([py, "load_data.py", "--seed", str(args.seed), "--sample-size", str(args.sample_size),
         "--out", str(questions)])
    run([py, "run_agent.py", "--questions", str(questions), "--traces-dir", str(traces),
         "--runs", str(args.runs), "--max-steps", str(args.max_steps)])
    run([py, "extract_hops.py", "--traces-dir", str(traces), "--reextract"])
    run([py, "score.py", "--traces-dir", str(traces)])

    print("\n" + "=" * 80)
    print("Generated + scored. Now do the BLIND human step, then validate frozen:")
    print(f"\n  {py} label.py    --traces-dir {traces} --labels {labels} --blind")
    print(f"  {py} validate.py --traces-dir {traces} --labels {labels} --report {report}")
    print("\nCompare that validate.py output to the in-sample numbers. A large drop in")
    print("recall means the patterns overfit the tuned set rather than the construct.")
    print("=" * 80)


if __name__ == "__main__":
    main()
