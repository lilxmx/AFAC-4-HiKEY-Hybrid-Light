"""Evidence Compressor: distribute and compress raw evidence into ClaimMemory.

Two responsibilities:
1. EvidenceDistributor: assign evidence items to relevant options/claims
2. EvidenceCompressor: compress raw evidence into per-option ClaimMemory

The compressor produces the "working memory" that the verifier reads,
keeping prompts bounded while preserving key facts.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .schemas import ClaimMemory, QAState
from .config import AgentConfig

logger = logging.getLogger(__name__)


class EvidenceDistributor:
    """Assign evidence items from the pool to relevant options/claims.

    Rules:
    1. Evidence from a group is assigned to all options that group covers
    2. Evidence whose content mentions option-specific keywords gets extra assignment
    3. Same evidence can serve multiple options (shared evidence)
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def distribute(self, state: QAState) -> QAState:
        """Assign evidence_ids to each OptionClaimState."""
        for opt_key, claim_state in state.claims.items():
            relevant_ids: List[str] = []

            for eid, item in state.raw_evidence_pool.items():
                # Check if this evidence was retrieved for this option
                query_labels = item.get("query_labels", []) or []
                covers = item.get("_covers_options", []) or []

                is_relevant = (
                    opt_key in covers
                    or f"option_{opt_key}" in query_labels
                    or f"选项{opt_key}" in query_labels
                    or "global" in query_labels
                    or "题干" in query_labels
                )

                if is_relevant:
                    relevant_ids.append(eid)

            # Sort by score descending, keep all relevant evidence (no cap)
            relevant_ids.sort(
                key=lambda eid: float(
                    state.raw_evidence_pool[eid].get("score", 0) or 0
                ),
                reverse=True,
            )
            claim_state.support_evidence_ids = relevant_ids

        return state


class EvidenceCompressor:
    """Compress raw evidence into per-option ClaimMemory.

    First version: simple extraction of key facts from evidence content.
    Future versions can use LLM-based summarization.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def compress(self, state: QAState) -> QAState:
        """Build ClaimMemory for each option from distributed evidence.

        No truncation or capping — preserve full evidence content to
        maximize recall and QA accuracy. Token budget is not a concern
        at this stage.
        """
        for opt_key, claim_state in state.claims.items():
            evidence_ids = claim_state.support_evidence_ids
            key_facts: List[str] = []

            for eid in evidence_ids:
                item = state.raw_evidence_pool.get(eid, {})
                content = item.get("content", "")
                if content:
                    key_facts.append(content)

            memory = ClaimMemory(
                option=opt_key,
                claim_summary=claim_state.original_text,
                key_facts=key_facts,
                support_evidence_ids=list(evidence_ids),
            )
            state.working_memory[opt_key] = memory

        return state


def run_compression(state: QAState, config: AgentConfig) -> QAState:
    """Execute distribution + compression in sequence."""
    distributor = EvidenceDistributor(config)
    compressor = EvidenceCompressor(config)

    state = distributor.distribute(state)
    state = compressor.compress(state)
    return state
