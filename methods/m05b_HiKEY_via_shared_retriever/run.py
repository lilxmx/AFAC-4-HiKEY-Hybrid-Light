#!/usr/bin/env python3
"""Entry point for m05b_HiKEY_via_shared_retriever.

This is a thin port of m05's ``run.py``. Behaviour stays identical;
only the underlying retrieval implementation is swapped to go through
``methods._shared.retrieval``.
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

# m05b's own config (only overrides identity + retriever name).
# Import via the absolute package path so this works regardless of which
# method was imported first in the current Python process (m05 also has
# a top-level ``config.py``).
from methods.m05b_HiKEY_via_shared_retriever import (  # noqa: E402
    config as _m05b_config,
)

# m05 config exposes the *retrieval / LLM* knobs that m05b inherits
# unchanged. We mutate it via M05_* env vars or the CLI flags below so
# the parent pipeline's class-level constants pick up overrides.
from methods.m05_HiKEY_question_option import config as _m05_config  # noqa: E402

from methods.m05_HiKEY_question_option.config import DOMAINS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{_m05b_config.METHOD_NAME}: m05 ported onto the shared Retriever API"
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to process")
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial results")
    parser.add_argument("--concurrency", type=int, default=None, help="Worker count")
    parser.add_argument("--model", type=str, default=None, help="Override Qwen model")
    parser.add_argument(
        "--final-topk", type=int, default=None, help="Final deduped evidence count"
    )
    parser.add_argument(
        "--option-topk", type=int, default=None, help="Per-option search top-k"
    )
    parser.add_argument(
        "--global-topk", type=int, default=None, help="Question-only search top-k"
    )
    parser.add_argument(
        "--evidence-budget", type=int, default=None, help="Evidence token budget"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default=None,
        help="Retriever name to build via methods._shared.retrieval (default: hikey)",
    )
    args = parser.parse_args()

    # Apply CLI overrides to the *m05* config module — m05b inherits
    # those values at runtime, so this keeps a single source of truth
    # for retrieval/LLM hyper-parameters.
    if args.concurrency:
        _m05_config.CONCURRENCY = args.concurrency
    if args.model:
        _m05_config.MODEL_NAME = args.model
    if args.final_topk:
        _m05_config.FINAL_EVIDENCE_TOPK = args.final_topk
    if args.option_topk:
        _m05_config.OPTION_TOPK = args.option_topk
    if args.global_topk:
        _m05_config.GLOBAL_TOPK = args.global_topk
    if args.evidence_budget:
        _m05_config.EVIDENCE_TOKEN_BUDGET = args.evidence_budget
    if args.retriever:
        _m05b_config.RETRIEVER_NAME = args.retriever

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting %s", _m05b_config.METHOD_NAME)
    logger.info(
        "Retriever=%s model=%s coder=%s final_topk=%s option_topk=%s",
        _m05b_config.RETRIEVER_NAME,
        _m05_config.MODEL_NAME,
        _m05b_config.CODER,
        _m05_config.FINAL_EVIDENCE_TOPK,
        _m05_config.OPTION_TOPK,
    )

    from methods.m05b_HiKEY_via_shared_retriever.pipeline import (
        HiKEYQuestionOptionPipelineV2,
    )

    domains = args.domains or DOMAINS
    pipeline = HiKEYQuestionOptionPipelineV2()
    pipeline.run(domains=domains, resume=not args.no_resume)
    summary = pipeline.save_results(extra_summary={
        "retriever": _m05b_config.RETRIEVER_NAME,
        "retrieval": {
            "global_topk": _m05_config.GLOBAL_TOPK,
            "option_topk": _m05_config.OPTION_TOPK,
            "final_evidence_topk": _m05_config.FINAL_EVIDENCE_TOPK,
            "evidence_token_budget": _m05_config.EVIDENCE_TOKEN_BUDGET,
        },
    })
    logger.info("Run complete. Summary: %s", summary)


if __name__ == "__main__":
    main()
