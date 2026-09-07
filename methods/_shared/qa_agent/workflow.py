"""Base Agent Workflow: the inheritable multi-round pipeline.

This is the core orchestration class that subclasses (m06, m07, m08...)
override to customize specific steps. The default implementation provides
a complete 2-round workflow:

    Round 0: Planning (planner + claim extractor + group builder)
    Round 1: Broad evidence acquisition
    Round 1.5: Evidence compression + claim verification
    Round 2: Gap-driven retrieval (if reflector triggers it)
    Round 2.5: Re-verification of affected options
    Final: Answer aggregation

Subclasses typically override:
- retrieve_round1(): to change retrieval strategy
- retrieve_round2(): to add new tools (VLM, calculator, etc.)
- Custom planner/verifier/reflector via config or constructor injection
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from .schemas import (
    EvidenceGroup,
    QAState,
    ReflectionAction,
    ToolCallRecord,
)
from .config import AgentConfig
from .planner import run_planning
from .compressor import run_compression
from .verifier import ClaimVerifier
from .reflector import ReflectionController
from .aggregator import AnswerAggregator

logger = logging.getLogger(__name__)


class BaseAgentWorkflow:
    """Orchestrates the full claim-centric QA pipeline for one question.

    This class is designed to be:
    1. Instantiated once per pipeline (not per question)
    2. Called via `run(question_item)` for each question
    3. Subclassed for different strategies (m06, m07, m08...)

    The retriever is injected via constructor — it must satisfy the
    `Retriever` protocol from `methods._shared.retrieval.api`.
    """

    def __init__(
        self,
        retriever,  # Retriever protocol from _shared.retrieval
        llm_client,  # OpenAI-compatible client
        config: AgentConfig,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config

        # Sub-components (can be overridden by subclasses)
        self.verifier = ClaimVerifier(config, llm_client=llm_client)
        self.reflector = ReflectionController(config)
        self.aggregator = AnswerAggregator(config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, question_item: Dict[str, Any]) -> QAState:
        """Process one question through the full agent workflow."""
        # Initialize state
        state = QAState.from_question(question_item)
        state.round_id = 0

        try:
            # Round 0: Planning
            state = self.plan(state)

            # Round 1: Broad evidence acquisition
            state.round_id = 1
            state = self.retrieve_round1(state)

            # Round 1.5: Compress + verify
            state = self.compress_evidence(state)
            state = self.verify_claims(state)

            # Reflection: decide if round 2 is needed
            actions = self.reflect(state)

            # Round 2: Gap-driven retrieval (conditional)
            if actions and state.round_id < self.config.max_rounds:
                state.round_id = 2
                state = self.retrieve_round2(state, actions)
                # Re-compress and re-verify affected options
                state = self.compress_evidence(state)
                affected_options = list(set(a.target_option for a in actions))
                state = self.reverify_claims(state, affected_options)

            # Final: Aggregate answer
            state = self.aggregate_answer(state)

        except Exception as e:
            logger.exception("[%s] Workflow failed: %s", state.qid, e)
            # Fallback: try to aggregate whatever we have
            if not state.final_answer:
                state.final_answer = "A"
            state.debug_trace["error"] = str(e)

        return state

    # ------------------------------------------------------------------
    # Pipeline stages (override in subclasses)
    # ------------------------------------------------------------------

    def plan(self, state: QAState) -> QAState:
        """Planning phase: parse question, extract claims, build groups."""
        return run_planning(state, self.config, llm_client=self.llm_client)

    def retrieve_round1(self, state: QAState) -> QAState:
        """Round 1: execute all evidence groups and populate evidence pool.

        Default implementation: iterate over evidence_groups, call retriever
        for each, merge results into raw_evidence_pool.
        """
        for group in state.evidence_groups:
            t0 = time.time()

            filters = {}
            if group.doc_scope:
                filters["doc_ids"] = group.doc_scope

            pack = self.retriever.retrieve(
                group.query,
                mode=group.mode,
                topk=group.topk,
                filters=filters if filters else None,
                query_label=group.group_id,
            )

            # Ingest items into evidence pool
            for item in pack.get("items", []) or []:
                eid = item.get("evidence_id", "")
                if eid and eid not in state.raw_evidence_pool:
                    # Tag which options this evidence covers
                    item["_covers_options"] = list(group.covers_options)
                    state.raw_evidence_pool[eid] = item
                elif eid in state.raw_evidence_pool:
                    # Merge coverage
                    existing = state.raw_evidence_pool[eid]
                    for opt in group.covers_options:
                        if opt not in existing.get("_covers_options", []):
                            existing.setdefault("_covers_options", []).append(opt)
                    # Merge query labels
                    for lbl in item.get("query_labels", []) or []:
                        if lbl not in existing.get("query_labels", []):
                            existing.setdefault("query_labels", []).append(lbl)

            # Record tool call
            state.tool_history.append(ToolCallRecord(
                tool_name=f"retrieve_{group.group_type}",
                round_id=state.round_id,
                query=group.query[:200],
                target_options=list(group.covers_options),
                result_count=len(pack.get("items", []) or []),
                latency_ms=(time.time() - t0) * 1000,
            ))

        return state

    def compress_evidence(self, state: QAState) -> QAState:
        """Compress raw evidence into per-option working memory."""
        return run_compression(state, self.config)

    def verify_claims(self, state: QAState) -> QAState:
        """Verify all options using the ClaimVerifier."""
        return self.verifier.verify_all(state)

    def reflect(self, state: QAState) -> List[ReflectionAction]:
        """Run reflection to detect gaps and plan round 2."""
        return self.reflector.reflect(state)

    def retrieve_round2(self, state: QAState, actions: List[ReflectionAction]) -> QAState:
        """Round 2: execute reflection-driven retrieval actions.

        Enhanced over Round 1:
        - option_search: uses increased topk (round2_option_topk) and diversified query
        - metric_search: uses key_metrics for precise field-level retrieval
        - counter_search: search for contradicting evidence
        - Effective dedup: skips already-retrieved evidence IDs, retrieves extra
          to compensate for expected overlap
        """
        # Collect existing evidence IDs for dedup awareness
        existing_eids = set(state.raw_evidence_pool.keys())

        for action in actions:
            t0 = time.time()

            filters = {}
            if state.doc_scope:
                filters["doc_ids"] = state.doc_scope

            # Determine mode and topk based on action type
            mode = "global"
            topk = self.config.counter_topk

            if action.tool == "counter_search":
                mode = "counter"
                topk = self.config.counter_topk
            elif action.tool == "option_search":
                mode = "option"
                # Strategy B: increased budget for round 2 retries
                if action.params.get("increased_budget"):
                    topk = self.config.round2_option_topk
                else:
                    topk = self.config.option_topk
            elif action.tool == "metric_search":
                # Strategy C: metric-focused retrieval
                mode = "field"
                topk = self.config.round2_metric_topk

            pack = self.retriever.retrieve(
                action.query,
                mode=mode,
                topk=topk,
                filters=filters if filters else None,
                query_label=f"r2_{action.tool}_{action.target_option}",
            )

            # Ingest into pool with dedup tracking
            new_count = 0
            for item in pack.get("items", []) or []:
                eid = item.get("evidence_id", "")
                if not eid:
                    continue
                if eid not in state.raw_evidence_pool:
                    # Genuinely new evidence
                    item["_covers_options"] = [action.target_option]
                    item["_round"] = 2  # Tag as round 2 evidence
                    state.raw_evidence_pool[eid] = item
                    new_count += 1
                else:
                    # Already exists — just extend coverage
                    existing = state.raw_evidence_pool[eid]
                    if action.target_option not in existing.get("_covers_options", []):
                        existing.setdefault("_covers_options", []).append(action.target_option)

            # If counter-search, mark evidence as counter
            if action.tool == "counter_search":
                claim_state = state.claims.get(action.target_option)
                if claim_state:
                    for item in pack.get("items", []) or []:
                        eid = item.get("evidence_id", "")
                        if eid and eid not in claim_state.counter_evidence_ids:
                            claim_state.counter_evidence_ids.append(eid)

            state.tool_history.append(ToolCallRecord(
                tool_name=action.tool,
                round_id=state.round_id,
                query=action.query[:200],
                target_options=[action.target_option],
                result_count=len(pack.get("items", []) or []),
                latency_ms=(time.time() - t0) * 1000,
                params={"new_evidence": new_count, "topk": topk},
            ))

            logger.info(
                "[%s] Round 2 %s for %s: retrieved %d items, %d new (topk=%d)",
                state.qid, action.tool, action.target_option,
                len(pack.get("items", []) or []), new_count, topk,
            )

        return state

    def reverify_claims(self, state: QAState, target_options: List[str]) -> QAState:
        """Re-verify only the affected options after round 2."""
        return self.verifier.reverify(state, target_options)

    def aggregate_answer(self, state: QAState) -> QAState:
        """Produce the final answer from claim states."""
        return self.aggregator.aggregate(state)
