"""
m01_baseline_qwen - quick run script (preprocess + inference + save).

Usage:
    python run.py [--domains insurance regulatory ...]
    python run.py --preprocess-only
    python run.py --inference-only
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Make project root importable so `from methods._shared.* import ...` works.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agent.pipeline import Pipeline  # noqa: E402
from agent.config import DOMAINS, CONCURRENCY, METHOD_NAME, LOGS_DIR  # noqa: E402
from agent import config as method_config  # noqa: E402

from methods._shared.config_base import (  # noqa: E402
    compute_token_score,
    estimate_final_multiplier,
)


def setup_logging() -> Path:
    log_file = LOGS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} - full pipeline")
    parser.add_argument("--domains", nargs="+", default=None, choices=DOMAINS,
                        help="domains to process (default: all)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="only build BM25 indices")
    parser.add_argument("--inference-only", action="store_true",
                        help="only run inference (indices must exist)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"parallel API calls (default: {CONCURRENCY})")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore partial results")
    args = parser.parse_args()

    if args.concurrency:
        method_config.CONCURRENCY = args.concurrency

    domains = args.domains or DOMAINS
    log_file = setup_logging()

    print("=" * 60)
    print(f"  {METHOD_NAME}")
    print("=" * 60)
    print(f"  Domains: {domains}")
    mode = "preprocess-only" if args.preprocess_only else (
        "inference-only" if args.inference_only else "full"
    )
    print(f"  Mode: {mode}")
    print(f"  Concurrency: {method_config.CONCURRENCY}")
    print(f"  Log: {log_file}")
    print("=" * 60)
    print()

    pipeline = Pipeline()
    total_start = time.time()

    if not args.inference_only:
        print("[Phase 1] Offline preprocessing (no token cost)")
        t0 = time.time()
        pipeline.build_indices(domains=domains)
        print(f"  Preprocessing done in {time.time() - t0:.1f}s\n")

    if args.preprocess_only:
        print("Preprocessing complete. Exiting.")
        return

    print("[Phase 2] Inference (token cost)")
    t0 = time.time()
    pipeline.run(domains=domains, resume=not args.no_resume)
    print(f"  Inference done in {time.time() - t0:.1f}s\n")

    print("[Phase 3] Saving results")
    pipeline.save_results()

    total_tokens = pipeline.token_stats["total_tokens"]
    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total time: {time.time() - total_start:.1f}s")
    print(f"  Questions answered: {len(pipeline.results)}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Token score: {compute_token_score(total_tokens):.4f}")
    print(f"  Estimated multiplier: {estimate_final_multiplier(total_tokens):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
