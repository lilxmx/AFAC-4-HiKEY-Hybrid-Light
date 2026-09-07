"""
Offline preprocessing script.
Parses all documents and builds BM25 indices for each domain.
This step does NOT consume any API tokens.

Usage:
    python -m script.preprocess [--domains insurance regulatory ...]
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.pipeline import Pipeline
from agent.config import DOMAINS


def main():
    parser = argparse.ArgumentParser(description="Preprocess documents and build indices")
    parser.add_argument(
        '--domains', nargs='+', default=None,
        choices=DOMAINS,
        help='Domains to process (default: all)'
    )
    args = parser.parse_args()
    
    domains = args.domains if args.domains else DOMAINS
    
    print("=" * 60)
    print("AFAC2026 Task4 Baseline - Offline Preprocessing")
    print("=" * 60)
    print(f"Domains: {domains}")
    print()
    
    pipeline = Pipeline()
    pipeline.build_indices(domains=domains)
    
    print()
    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
