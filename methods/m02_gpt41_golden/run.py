#!/usr/bin/env python3
"""
m02_gpt41_golden - main entry point.

Usage:
    python run.py                          # all domains
    python run.py --domains regulatory insurance
    python run.py --no-resume              # ignore partial results
    python run.py --concurrency 3 --voting-rounds 5
"""
import argparse
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from config import DOMAINS, LOGS_DIR, METHOD_NAME, OUTPUT_DIR  # noqa: E402


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def main():
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} - Golden Label Generation")
    parser.add_argument("--domains", nargs="+", default=None, choices=DOMAINS,
                        help="domains to process (default: all)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"parallel API calls (default: {config.CONCURRENCY})")
    parser.add_argument("--voting-rounds", type=int, default=None,
                        help=f"number of voting rounds (default: {config.VOTING_ROUNDS})")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore partial results")
    args = parser.parse_args()

    if args.concurrency:
        config.CONCURRENCY = args.concurrency
    if args.voting_rounds:
        config.VOTING_ROUNDS = args.voting_rounds

    domains = args.domains or DOMAINS
    resume = not args.no_resume
    log_file = setup_logging()

    print("=" * 60)
    print(f"  {METHOD_NAME}")
    print("=" * 60)
    print(f"  Domains: {domains}")
    print(f"  Reasoning model: {config.REASONING_MODEL}")
    print(f"  Embedding model: {config.EMBEDDING_MODEL}")
    print(f"  Concurrency: {config.CONCURRENCY}")
    print(f"  Voting rounds: {config.VOTING_ROUNDS}")
    print(f"  Resume: {resume}")
    print(f"  Log: {log_file}")
    print("=" * 60)
    print()

    from pipeline import GoldenPipeline  # noqa: E402

    pipeline = GoldenPipeline()
    total_start = time.time()
    pipeline.run(domains=domains, resume=resume)
    pipeline.save_results()

    print()
    print("=" * 60)
    print(f"  DONE in {time.time() - total_start:.1f}s")
    print(f"  Questions: {len(pipeline.results)}")
    print(f"  Output: {OUTPUT_DIR / 'answer.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
