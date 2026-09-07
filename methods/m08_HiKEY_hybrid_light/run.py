#!/usr/bin/env python3
"""Entry point for m08_HiKEY_hybrid_light."""
import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from config import DOMAINS, METHOD_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{METHOD_NAME}: HiKEY hybrid-light retrieval"
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to process")
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial results")
    parser.add_argument("--concurrency", type=int, default=None, help="Worker count")
    parser.add_argument("--model", type=str, default=None, help="Override Qwen model")
    parser.add_argument("--final-topk", type=int, default=None, help="Final deduped evidence count")
    parser.add_argument("--option-topk", type=int, default=None, help="Per-option search top-k")
    parser.add_argument("--global-topk", type=int, default=None, help="Question-only search top-k")
    parser.add_argument("--evidence-budget", type=int, default=None, help="Evidence token budget")
    args = parser.parse_args()

    if args.concurrency:
        config.CONCURRENCY = args.concurrency
    if args.model:
        config.MODEL_NAME = args.model
    if args.final_topk:
        config.FINAL_EVIDENCE_TOPK = args.final_topk
    if args.option_topk:
        config.OPTION_TOPK = args.option_topk
    if args.global_topk:
        config.GLOBAL_TOPK = args.global_topk
    if args.evidence_budget:
        config.EVIDENCE_TOKEN_BUDGET = args.evidence_budget

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting %s", METHOD_NAME)
    logger.info(
        "Model=%s coder=%s final_topk=%s option_topk=%s",
        config.MODEL_NAME,
        config.CODER,
        config.FINAL_EVIDENCE_TOPK,
        config.OPTION_TOPK,
    )

    from methods.m08_HiKEY_hybrid_light.pipeline import HiKEYHybridLightPipeline

    domains = args.domains or DOMAINS
    pipeline = HiKEYHybridLightPipeline()
    pipeline.run(domains=domains, resume=not args.no_resume)
    summary = pipeline.save_results(extra_summary={
        "retrieval": {
            "global_topk": config.GLOBAL_TOPK,
            "option_topk": config.OPTION_TOPK,
            "final_evidence_topk": config.FINAL_EVIDENCE_TOPK,
            "evidence_token_budget": config.EVIDENCE_TOKEN_BUDGET,
            "strategy": "hybrid_light",
        }
    })
    logger.info("Run complete. Summary: %s", summary)


if __name__ == "__main__":
    main()
