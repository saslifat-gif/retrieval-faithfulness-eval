"""Regression tests for the grounding instrument and the MuSiQue benchmark.

Stdlib `unittest` only — no pytest dependency, no model downloads, no API calls.
Run from the repo root (either form; `-t .` is required with `discover -s` so
this package __init__ sets up sys.path):

    python -m unittest
    python -m unittest discover -s tests -t . -v

The scripts under `benchmark/`, `grounding/`, and `validation/` are flat scripts
(not packages), so make them importable by module name here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ("", "benchmark", "grounding", "validation"):
    path = str(ROOT / subdir) if subdir else str(ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
