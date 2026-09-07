#!/usr/bin/env python3
"""
m03_fulltext_baseline - entry point.

Usage:
    python run.py                          # all domains
    python run.py --domains insurance regulatory
    python run.py --no-resume              # ignore partial results
    python run.py --concurrency 8
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure both the project root and this method dir are importable.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from config import DOMAINS, LOGS_DIR, METHOD_NAME, OUTPUT_DIR  # noqa: E402


def setup_logging() -> Path:
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
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} - run")
    parser.add_argument("--domains", nargs="+", default=None,
                        help=f"domains to process (default: all). Choices: {DOMAINS}")
    parser.add_argument("--no-resume", action="store_true",
                        help="start fresh, ignore partial results")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="parallel API calls (overrides config.CONCURRENCY)")
    args = parser.parse_args()

    if args.concurrency:
        config.CONCURRENCY = args.concurrency

    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    domains = args.domains or DOMAINS
    resume = not args.no_resume

    if not resume:
        partial = OUTPUT_DIR / "partial_results.json"
        if partial.exists():
            partial.unlink()
            logger.info("Cleared partial results for fresh run")

    print("=" * 60)
    print(f"  {METHOD_NAME} - Full-text Input Baseline")
    print("=" * 60)
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Domains: {domains}")
    print(f"  Concurrency: {config.CONCURRENCY}")
    print(f"  Resume: {resume}")
    print(f"  Log: {log_file}")
    print("=" * 60)
    print()

    from pipeline import FullTextPipeline  # noqa: E402

    start = time.time()
    pipeline = FullTextPipeline()
    pipeline.run(domains=domains, resume=resume)
    pipeline.save_results()
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"  DONE in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Total tokens: {pipeline.token_stats['total_tokens']:,}")
    print(f"  Questions: {len(pipeline.results)}")
    print(f"  Output: {OUTPUT_DIR / 'answer.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
