#!/usr/bin/env python3
"""Entry point for m06_claim_verify.

Usage:
    python methods/m06_claim_verify/run.py
    python methods/m06_claim_verify/run.py --domains financial_reports insurance
    python methods/m06_claim_verify/run.py --no-counter  # disable counter-search
    python methods/m06_claim_verify/run.py --max-rounds 1  # single-round only

Background:
    nohup python -u methods/m06_claim_verify/run.py > /dev/null 2>&1 &
"""
import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods._shared.config_base import DOMAINS  # noqa: E402
from methods.m06_claim_verify import config as _cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{_cfg.METHOD_NAME}: Claim-Centric Verification Agent"
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to process")
    parser.add_argument("--qids", nargs="+", default=None, help="Only process these question IDs")
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial results")
    parser.add_argument("--concurrency", type=int, default=None, help="Worker count")
    parser.add_argument("--model", type=str, default=None, help="Override Qwen model")
    parser.add_argument("--max-rounds", type=int, default=None, help="Max retrieval rounds (1 or 2)")
    parser.add_argument("--no-counter", action="store_true", help="Disable counter-search")
    parser.add_argument("--no-verifier", action="store_true", help="Disable LLM verifier (debug)")
    parser.add_argument(
        "--global-topk", type=int, default=None, help="Global search top-k"
    )
    parser.add_argument(
        "--option-topk", type=int, default=None, help="Per-option search top-k"
    )
    parser.add_argument(
        "--retriever", type=str, default=None, help="Retriever name (default: hikey)"
    )
    args = parser.parse_args()

    # Apply CLI overrides
    if args.concurrency:
        _cfg.CONCURRENCY = args.concurrency
    if args.model:
        _cfg.MODEL_NAME = args.model
        _cfg.AGENT_CONFIG.model_name = args.model
    if args.max_rounds is not None:
        _cfg.AGENT_CONFIG.max_rounds = args.max_rounds
    if args.no_counter:
        _cfg.AGENT_CONFIG.enable_counter_search = False
    if args.no_verifier:
        _cfg.AGENT_CONFIG.use_llm_verifier = False
    if args.global_topk:
        _cfg.AGENT_CONFIG.global_topk = args.global_topk
    if args.option_topk:
        _cfg.AGENT_CONFIG.option_topk = args.option_topk
    if args.retriever:
        _cfg.RETRIEVER_NAME = args.retriever

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting %s", _cfg.METHOD_NAME)
    logger.info(
        "Config: retriever=%s model=%s max_rounds=%d counter=%s verifier=%s",
        _cfg.RETRIEVER_NAME,
        _cfg.AGENT_CONFIG.model_name,
        _cfg.AGENT_CONFIG.max_rounds,
        _cfg.AGENT_CONFIG.enable_counter_search,
        _cfg.AGENT_CONFIG.use_llm_verifier,
    )

    from methods.m06_claim_verify.pipeline import M06ClaimVerifyPipeline

    domains = args.domains or DOMAINS
    pipeline = M06ClaimVerifyPipeline()
    pipeline.run(domains=domains, resume=not args.no_resume, qid_filter=args.qids)
    summary = pipeline.save_results(extra_summary={
        "agent_config": {
            "version": _cfg.AGENT_CONFIG.version,
            "max_rounds": _cfg.AGENT_CONFIG.max_rounds,
            "enable_counter_search": _cfg.AGENT_CONFIG.enable_counter_search,
            "enable_option_fallback": _cfg.AGENT_CONFIG.enable_option_fallback,
            "global_topk": _cfg.AGENT_CONFIG.global_topk,
            "option_topk": _cfg.AGENT_CONFIG.option_topk,
            "counter_topk": _cfg.AGENT_CONFIG.counter_topk,
            "use_llm_verifier": _cfg.AGENT_CONFIG.use_llm_verifier,
        },
        "retriever": _cfg.RETRIEVER_NAME,
    })
    logger.info("Run complete. Summary: %s", summary)


if __name__ == "__main__":
    main()
