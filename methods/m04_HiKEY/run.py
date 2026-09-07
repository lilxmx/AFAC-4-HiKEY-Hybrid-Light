#!/usr/bin/env python3
"""
m04_HiKEY entry point - HiKEY-style hierarchical retrieval for financial QA.

Usage:
    # Build HiKEY index (run once before inference)
    python run.py --build-index --domains financial_reports

    # Run inference
    python run.py --domains financial_reports

    # Run all domains (HiKEY for financial_reports, fallback for others)
    python run.py

    # Run with custom concurrency
    python run.py --concurrency 8 --domains financial_reports
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

from config import DOMAINS, METHOD_NAME, METHOD_ROOT
import config


def main():
    parser = argparse.ArgumentParser(
        description=f"{METHOD_NAME}: HiKEY-style hierarchical retrieval for financial QA"
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Domains to process (default: all)"
    )
    parser.add_argument(
        "--build-index", action="store_true",
        help="Build HiKEY hierarchical index (run once before inference)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignore partial results"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Number of concurrent workers"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name (e.g. qwen-plus, qwen-max)"
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="Override retrieval top-k"
    )
    args = parser.parse_args()

    # Apply overrides
    if args.concurrency:
        config.CONCURRENCY = args.concurrency
    if args.model:
        config.MODEL_NAME = args.model
    if args.topk:
        config.RETRIEVAL_TOPK = args.topk

    # Setup logging (console only; file logs are managed by RunContext in runs/ dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {METHOD_NAME}")
    logger.info(f"Model: {config.MODEL_NAME}, Concurrency: {config.CONCURRENCY}")

    domains = args.domains or DOMAINS

    if args.build_index:
        # Index building mode
        logger.info(f"Building HiKEY index for domains: {domains}")
        from pipeline import build_hikey_index
        build_hikey_index(domains)
        logger.info("Index building complete.")
        return

    # Inference mode
    from pipeline import HiKEYPipeline
    pipeline = HiKEYPipeline()
    pipeline.run(domains=domains, resume=not args.no_resume)
    summary = pipeline.save_results()

    logger.info(f"Run complete. Summary: {summary}")


if __name__ == "__main__":
    main()
