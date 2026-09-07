"""
Online inference script.
Loads pre-built indices, retrieves evidence, and calls Qwen API for reasoning.
This step DOES consume API tokens.

Usage:
    python -m script.run_inference [--domains insurance regulatory ...]
"""
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.pipeline import Pipeline
from agent.config import DOMAINS


def main():
    parser = argparse.ArgumentParser(description="Run inference on questions")
    parser.add_argument(
        '--domains', nargs='+', default=None,
        choices=DOMAINS,
        help='Domains to process (default: all)'
    )
    parser.add_argument(
        '--build-if-missing', action='store_true',
        help='Build indices if not found on disk'
    )
    args = parser.parse_args()
    
    domains = args.domains if args.domains else DOMAINS
    
    print("=" * 60)
    print("AFAC2026 Task4 Baseline - Online Inference")
    print("=" * 60)
    print(f"Domains: {domains}")
    print()
    
    pipeline = Pipeline()
    
    # Load or build indices
    print("[Step 1] Loading indices...")
    start_time = time.time()
    if args.build_if_missing:
        pipeline.load_indices(domains=domains)
    else:
        # Try to load, build if missing
        pipeline.load_indices(domains=domains)
    
    elapsed = time.time() - start_time
    print(f"  Indices loaded in {elapsed:.1f}s")
    print()
    
    # Run inference
    print("[Step 2] Running inference...")
    start_time = time.time()
    pipeline.run_inference(domains=domains)
    elapsed = time.time() - start_time
    print(f"  Inference complete in {elapsed:.1f}s")
    print()
    
    # Save results
    print("[Step 3] Saving results...")
    pipeline.save_results()
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(pipeline.results)}")
    print(f"Total prompt tokens: {pipeline.token_stats['total_prompt']:,}")
    print(f"Total completion tokens: {pipeline.token_stats['total_completion']:,}")
    print(f"Total tokens: {pipeline.token_stats['total_tokens']:,}")
    
    # Calculate token score
    total_tokens = pipeline.token_stats['total_tokens']
    token_score = max(0, min(1, (5_000_000 - total_tokens) / 5_000_000))
    print(f"TokenScore: {token_score:.4f}")
    print(f"Output saved to: {pipeline.results and 'output/answer.csv, output/evidence.json'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
