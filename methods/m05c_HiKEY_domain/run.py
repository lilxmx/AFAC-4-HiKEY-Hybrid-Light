#!/usr/bin/env python3
"""Entry point for m05c_HiKEY_domain."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods._shared.config_base import DOMAINS
from methods.m05c_HiKEY_domain import config
from methods.m05c_HiKEY_domain.evaluation import load_ground_truth, normalize_answer, write_comparison_files
from methods.m05c_HiKEY_domain.pipeline import HiKEYDomainPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{config.METHOD_NAME}: HiKEY + domain-specific retrieval")
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to process")
    parser.add_argument("--qid", nargs="+", default=None, help="Optional qid filter for smoke tests")
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial results")
    parser.add_argument("--concurrency", type=int, default=None, help="Worker count")
    parser.add_argument("--model", type=str, default=None, help="Override Qwen model")
    parser.add_argument("--final-topk", type=int, default=None, help="Final evidence blocks sent to model")
    parser.add_argument("--option-topk", type=int, default=None, help="Per-option HiKEY search top-k")
    parser.add_argument("--domain-card-topk", type=int, default=None, help="Per-query domain-card top-k")
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
    if args.domain_card_topk:
        config.DOMAIN_CARD_TOPK = args.domain_card_topk
    if args.evidence_budget:
        config.EVIDENCE_TOKEN_BUDGET = args.evidence_budget

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting %s", config.METHOD_NAME)
    logger.info(
        "model=%s final_topk=%s option_topk=%s domain_card_topk=%s evidence_budget=%s",
        config.MODEL_NAME,
        config.FINAL_EVIDENCE_TOPK,
        config.OPTION_TOPK,
        config.DOMAIN_CARD_TOPK,
        config.EVIDENCE_TOKEN_BUDGET,
    )

    pipeline = HiKEYDomainPipeline()
    domains = args.domains or DOMAINS
    pipeline.run(domains=domains, resume=not args.no_resume, qid_filter=args.qid)

    if config.GROUND_TRUTH_PATH.exists():
        gt = load_ground_truth(config.GROUND_TRUTH_PATH)
        for qid, result in pipeline.results.items():
            if qid in gt:
                result["topk_answer_hit"] = normalize_answer(result.get("answer")) == normalize_answer(gt[qid])

    eval_summary = {}
    if config.GROUND_TRUTH_PATH.exists():
        eval_summary = write_comparison_files(pipeline.output_dir, pipeline.results, config.GROUND_TRUTH_PATH)
        logger.info("Saved ground-truth comparison files to %s", pipeline.output_dir)
        logger.info("Comparison summary: %s", eval_summary)
    else:
        logger.warning("Ground truth file not found: %s", config.GROUND_TRUTH_PATH)

    summary = pipeline.save_results(extra_summary={
        "retrieval": {
            "question_topk": config.QUESTION_TOPK,
            "option_topk": config.OPTION_TOPK,
            "domain_card_topk": config.DOMAIN_CARD_TOPK,
            "strong_domain_card_domains": sorted(config.STRONG_DOMAIN_CARD_DOMAINS),
            "weak_domain_card_topk": config.WEAK_DOMAIN_CARD_TOPK,
            "weak_per_option_evidence": config.WEAK_PER_OPTION_EVIDENCE,
            "weak_domain_card_min_score": config.WEAK_DOMAIN_CARD_MIN_SCORE,
            "per_option_evidence": config.PER_OPTION_EVIDENCE,
            "final_evidence_topk": config.FINAL_EVIDENCE_TOPK,
            "evidence_token_budget": config.EVIDENCE_TOKEN_BUDGET,
        },
        "ground_truth_eval": eval_summary,
    })
    logger.info("Run complete. Summary: %s", summary)


if __name__ == "__main__":
    main()
