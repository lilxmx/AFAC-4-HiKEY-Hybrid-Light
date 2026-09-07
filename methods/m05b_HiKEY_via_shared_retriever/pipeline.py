"""m05b pipeline: m05 with the index manager swapped for the shared
``Retriever`` API.

The class layout follows m05 verbatim. The only substantive change is
that ``self.index_manager`` now holds a thin adapter around a retriever
built via ``methods._shared.retrieval.build_retriever(...)``. The
adapter exposes the two methods m05's pipeline calls
(``ensure_domain_loaded`` / ``pack_query``) and forwards them to the
shared retriever's protocol methods.

Goal: prove that a QA pipeline can be ported onto the shared API with
zero behavioural drift, and give m06 a clean starting point.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse all the building blocks from m05. We deliberately import the
# functions (not re-define them) so behaviour stays in lock-step.
from methods.m05_HiKEY_question_option.pipeline import (  # noqa: E402
    HiKEYQuestionOptionPipeline as _M05Pipeline,
    _anchor_key,
    _anchor_score,
    _merge_evidence_packs,
    _target_doc_ids,
    format_evidence_for_prompt,
    format_options,
)

from methods._shared.retrieval import (  # noqa: E402
    EvidencePack,
    Retriever,
    RetrievalFilters,
    build_retriever,
)

# Import via the absolute package path to avoid colliding with m05's
# top-level ``import config`` (m05 puts its own dir on sys.path, so a
# bare ``import config`` would resolve to whichever method was imported
# first in the current process).
from methods.m05b_HiKEY_via_shared_retriever import (  # noqa: E402
    config as _m05b_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter: make a shared ``Retriever`` look like the legacy
# ``HiKEYQuestionOptionIndexManager`` so we can drop it into m05's class
# without touching its retrieval-call sites.
# ---------------------------------------------------------------------------

class SharedRetrieverIndexAdapter:
    """Legacy-shaped wrapper around a :class:`Retriever`.

    Implements the two methods m05's pipeline calls:
    - ``ensure_domain_loaded(domain)``
    - ``pack_query(query, topk, target_doc_ids)``
    """

    def __init__(self, retriever: Retriever):
        self._retriever = retriever

    @property
    def retriever(self) -> Retriever:
        """Expose the underlying retriever for downstream / debugging use."""
        return self._retriever

    # m05 surface -----------------------------------------------------

    def ensure_domain_loaded(self, domain: str) -> None:
        warmup = getattr(self._retriever, "warmup", None)
        if callable(warmup):
            warmup(domains=[domain])

    def pack_query(
        self,
        query: str,
        topk: int,
        target_doc_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        # Prefer the dedicated legacy shim when the retriever provides
        # one (``HiKEYRetriever`` does). It guarantees the exact m05
        # output shape including the ``_hikey`` bundle layout.
        legacy = getattr(self._retriever, "pack_query_legacy", None)
        if callable(legacy):
            return legacy(query, topk=topk, target_doc_ids=target_doc_ids)

        # Generic fallback: call the protocol and translate.
        filters: RetrievalFilters = {}
        if target_doc_ids:
            filters["doc_ids"] = list(target_doc_ids)
        pack: EvidencePack = self._retriever.retrieve(
            query, mode="global", topk=topk, filters=filters, query_label="global"
        )
        evidence_pack: List[Dict[str, Any]] = []
        for item in pack.get("items", []) or []:
            structured = item.get("structured", {}) or {}
            bundle = structured.get("_hikey") or {
                "anchor": structured,
                "ancestry": {},
                "siblings": structured.get("siblings", []) or [],
            }
            bundle = dict(bundle)
            bundle["source_doc_id"] = item.get("doc_id")
            evidence_pack.append(bundle)
        return {
            "query": query,
            "evidence_pack": evidence_pack,
            "routed_docs": pack.get("meta", {}).get("routed_docs", []),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HiKEYQuestionOptionPipelineV2(_M05Pipeline):
    """m05 pipeline rewired onto the shared retrieval API.

    Inherits all retrieval orchestration / prompting / reflection logic
    from m05 unchanged. Only ``__init__`` differs: ``self.index_manager``
    is now a :class:`SharedRetrieverIndexAdapter` wrapping a retriever
    built via the registry.
    """

    method_name = _m05b_config.METHOD_NAME
    model_name = _m05b_config.MODEL_NAME

    def __init__(self) -> None:
        # Skip _M05Pipeline.__init__ to avoid instantiating the legacy
        # IndexManager, but reuse BaseRunner.__init__ via super-super.
        from methods._shared.pipeline import BaseRunner
        from methods._shared.llm import make_dashscope_client

        BaseRunner.__init__(
            self,
            output_dir=_m05b_config.METHOD_ROOT / "output",
            logs_dir=_m05b_config.METHOD_ROOT / "logs",
            concurrency=_m05b_config.CONCURRENCY,
            coder=_m05b_config.CODER,
            run_desc=_m05b_config.RUN_DESC,
        )
        self.client = make_dashscope_client(api_key=_m05b_config.DASHSCOPE_API_KEY)

        retriever = build_retriever(
            _m05b_config.RETRIEVER_NAME,
            index_dir=_m05b_config.HIKEY_INDEX_DIR,
            max_siblings=_m05b_config.MAX_SIBLINGS,
            skip_doc_routing=_m05b_config.SKIP_DOC_ROUTING,
            **(_m05b_config.RETRIEVER_EXTRA_KWARGS or {}),
        )
        self.index_manager = SharedRetrieverIndexAdapter(retriever)

        logger.info(
            "[%s] using retriever=%s (index_dir=%s)",
            _m05b_config.METHOD_NAME,
            _m05b_config.RETRIEVER_NAME,
            _m05b_config.HIKEY_INDEX_DIR,
        )


__all__ = [
    "HiKEYQuestionOptionPipelineV2",
    "SharedRetrieverIndexAdapter",
    # Re-export so callers that only know m05b can still see the helpers.
    "format_options",
    "format_evidence_for_prompt",
    "_merge_evidence_packs",
    "_anchor_key",
    "_anchor_score",
    "_target_doc_ids",
]
