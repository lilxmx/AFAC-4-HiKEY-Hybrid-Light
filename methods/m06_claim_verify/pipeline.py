"""m06 Pipeline: Claim-Centric Verification Agent.

This pipeline integrates the shared QA Agent framework with the existing
BaseRunner infrastructure. It:
1. Builds a retriever via the shared retrieval factory
2. Instantiates the BaseAgentWorkflow with m06's AgentConfig
3. Delegates each question to the workflow's `run()` method
4. Returns results in the format BaseRunner expects

The pipeline is a drop-in replacement for m05's pipeline — same runner
interface, same output format, but internally uses the multi-round
claim verification workflow.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods._shared.config_base import infer_domain_from_qid
from methods._shared.llm import make_dashscope_client
from methods._shared.pipeline import BaseRunner
from methods._shared.retrieval import build_retriever
from methods._shared.qa_agent import BaseAgentWorkflow

from methods.m06_claim_verify import config as _cfg

logger = logging.getLogger(__name__)


class M06ClaimVerifyPipeline(BaseRunner):
    """m06 pipeline: claim-centric verification with multi-round retrieval.

    Inherits BaseRunner for concurrency, resume, and output management.
    Internally delegates to BaseAgentWorkflow for the QA logic.
    """

    method_name = _cfg.METHOD_NAME
    model_name = _cfg.MODEL_NAME

    def __init__(self) -> None:
        super().__init__(
            output_dir=_cfg.METHOD_ROOT / "output",
            logs_dir=_cfg.METHOD_ROOT / "logs",
            concurrency=_cfg.CONCURRENCY,
            coder=_cfg.CODER,
            run_desc=_cfg.RUN_DESC,
        )

        # LLM client
        self.client = make_dashscope_client(api_key=_cfg.DASHSCOPE_API_KEY)

        # Retriever (same as m05b — uses shared retrieval factory)
        self._retriever = build_retriever(
            _cfg.RETRIEVER_NAME,
            index_dir=_cfg.HIKEY_INDEX_DIR,
            max_siblings=_cfg.MAX_SIBLINGS,
            skip_doc_routing=_cfg.SKIP_DOC_ROUTING,
            **(_cfg.RETRIEVER_EXTRA_KWARGS or {}),
        )

        # Agent workflow
        self.workflow = BaseAgentWorkflow(
            retriever=self._retriever,
            llm_client=self.client,
            config=_cfg.AGENT_CONFIG,
        )

        logger.info(
            "[%s] initialized: retriever=%s model=%s config_version=%s",
            _cfg.METHOD_NAME,
            _cfg.RETRIEVER_NAME,
            _cfg.MODEL_NAME,
            _cfg.AGENT_CONFIG.version,
        )

    def setup(self, domains: List[str]) -> None:
        """Load HiKEY indices for the specified domains."""
        warmup = getattr(self._retriever, "warmup", None)
        if callable(warmup):
            for domain in domains:
                warmup(domains=[domain])
                logger.info("[%s] warmed up domain: %s", _cfg.METHOD_NAME, domain)

    def process_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        """Process one question through the claim verification workflow."""
        qid = q["qid"]
        domain = q.get("domain") or infer_domain_from_qid(qid)
        q["domain"] = domain

        try:
            # Run the full agent workflow
            state = self.workflow.run(q)

            # Convert to BaseRunner result format
            result = state.to_result_dict()
            result["domain"] = domain

            logger.debug(
                "[%s] qid=%s answer=%s claims=%s round=%d tools=%d",
                _cfg.METHOD_NAME,
                qid,
                result["answer"],
                {k: v.status for k, v in state.claims.items()},
                state.round_id,
                len(state.tool_history),
            )

            return result

        except Exception as e:
            logger.exception("[%s] qid=%s failed: %s", _cfg.METHOD_NAME, qid, e)
            return {
                "qid": qid,
                "answer": "A",
                "raw_answer": f"ERROR: {e}",
                "domain": domain,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }


__all__ = ["M06ClaimVerifyPipeline"]
