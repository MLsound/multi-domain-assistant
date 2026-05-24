#!/usr/bin/env python3
"""
CLI wrapper for the Knowledge Assistant evaluation runner.

Usage:
    poetry run python scripts/run_evaluation.py
    poetry run python scripts/run_evaluation.py --questions data/eval/test_suite.json
    poetry run python scripts/run_evaluation.py --output reports/my_run.json

See `docs/CONTEXT.md` for details on the evaluation domains (Science, Software, User).

Requires GOOGLE_API_KEY in environment (or .env file).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Knowledge Assistant evaluation suite"
    )
    parser.add_argument(
        "--questions",
        default="data/eval/test_suite.json",
        help="Path to test suite JSON file (default: data/eval/test_suite.json)",
    )
    parser.add_argument(
        "--output",
        default="reports/evaluation_results.json",
        help="Output path for results JSON (default: reports/evaluation_results.json)",
    )
    args = parser.parse_args()

    from src.evaluation.eval_runner import run_evaluation

    results = run_evaluation(
        test_suite_path=args.questions,
        output_path=args.output,
    )

    if results:
        print(f"\nResults written to: {args.output}")
    else:
        print("Evaluation failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
