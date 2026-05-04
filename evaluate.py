"""
Backward-compatible evaluation entry point.

Delegates all logic to src/evaluation/eval_runner.py.
Can still be run directly: poetry run python evaluate.py
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from src.evaluation.eval_runner import run_evaluation  # noqa: E402

if __name__ == "__main__":
    run_evaluation()
