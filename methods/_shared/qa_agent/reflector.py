"""Reflection Controller: gap detection and round-2 action planning.

The reflector examines the verification results and decides whether
additional retrieval is needed. It does NOT change answers directly —
it only produces ReflectionAction objects that tell round 2 what to do.

Trigger conditions:
1. Option status = mixed with low confidence (evidence gap)
2. Option contains universal keywords (均/全部/所有) but not all sub-claims verified
3. Option contains negation keywords (不/未/无) — needs counter-evidence
4. Option contains threshold/comparison keywords — may need calculation
5. Multi-select with no supported options (something is wrong)
6. Single-select with multiple supported options (ambiguity)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .schemas import QAState, ReflectionAction
from .config import AgentConfig

logger = logging.getLogger(__name__)


class ReflectionController:
    """Analyze verification results and plan round-2 actions."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def reflect(self, state: QAState) -> List[ReflectionAction]:
        """Examine state and return actions for round 2.

        Returns empty list if no further retrieval is needed.
        """
        actions: List[ReflectionAction] = []

        # Check global anomalies first
        actions.extend(self._check_global_anomalies(state))

        # Check per-option gaps
        for opt_key, claim_state in state.claims.items():
            actions.extend(self._check_option_gaps(state, opt_key))

        # Deduplicate and cap
        actions = self._deduplicate_actions(actions)
        if len(actions) > self.config.max_tool_calls - len(state.tool_history):
            actions = actions[: self.config.max_tool_calls - len(state.tool_history)]

        state.reflection_actions = actions
        return actions

    def _check_global_anomalies(self, state: QAState) -> List[ReflectionAction]:
        """Detect global issues that need intervention."""
        actions = []
        statuses = {k: v.status for k, v in state.claims.items()}
        supported = [k for k, s in statuses.items() if s == "supported"]
        mixed_low = [k for k, v in state.claims.items() if v.status == "mixed" and v.confidence < 0.4]
        contradicted = [k for k, s in statuses.items() if s == "contradicted"]
        non_contradicted = [k for k, s in statuses.items() if s != "contradicted" and s != "supported"]

        # Multi-select with no supported options → something is wrong
        if state.answer_format == "multi" and not supported:
            # Try option-specific search for all mixed-low options
            for opt_key in mixed_low[:3]:
                actions.append(ReflectionAction(
                    tool="option_search",
                    target_option=opt_key,
                    query=self._build_diversified_query(state, opt_key),
                    reason="Multi-select with no supported options; retry search",
                    params={"increased_budget": True},
                ))
                # Also try metric-focused retrieval
                metric_query = self._build_metric_query(state, opt_key)
                if metric_query:
                    actions.append(ReflectionAction(
                        tool="metric_search",
                        target_option=opt_key,
                        query=metric_query,
                        reason=f"Metric-focused retrieval for {opt_key} (no supported options)",
                        params={"increased_budget": True},
                    ))

        # Multi-select with only 1 supported option → likely missed correct options
        # This is a critical safeguard: multi-select should typically have 2+ answers
        if state.answer_format == "multi" and len(supported) == 1:
            # Re-search for non-contradicted options (they might be correct but
            # the verifier was too conservative)
            candidates = non_contradicted if non_contradicted else contradicted
            for opt_key in candidates[:3]:
                claim = state.claims[opt_key]
                # Tightened threshold (was 0.98, now 0.92): give Round 2 more
                # chances to overturn contradicted-but-not-rock-solid options
                if claim.status == "contradicted" and claim.confidence >= 0.92:
                    continue
                # Strategy B: diversified option_search with increased budget
                actions.append(ReflectionAction(
                    tool="option_search",
                    target_option=opt_key,
                    query=self._build_diversified_query(state, opt_key),
                    reason=f"Multi-select with only 1 supported; retry {opt_key} (was {claim.status})",
                    params={"increased_budget": True},
                ))
                # Strategy C: metric-focused retrieval if structured info available
                metric_query = self._build_metric_query(state, opt_key)
                if metric_query:
                    actions.append(ReflectionAction(
                        tool="metric_search",
                        target_option=opt_key,
                        query=metric_query,
                        reason=f"Metric-focused retrieval for {opt_key} using key_metrics",
                        params={"increased_budget": True},
                    ))
                # Strategy D: per-entity search for multi-entity options
                # (e.g. "两家公司均...", "四款产品中...")
                entity_queries = self._build_per_entity_queries(state, opt_key)
                for entity_label, entity_query in entity_queries:
                    actions.append(ReflectionAction(
                        tool="option_search",
                        target_option=opt_key,
                        query=entity_query,
                        reason=f"Per-entity search for {opt_key}: {entity_label}",
                        params={"increased_budget": True, "entity": entity_label},
                    ))

        # Single-select with multiple supported → need disambiguation
        if state.answer_format == "mcq" and len(supported) > 1:
            for opt_key in supported:
                if self.config.enable_counter_search:
                    actions.append(ReflectionAction(
                        tool="counter_search",
                        target_option=opt_key,
                        query=self._build_counter_query(state, opt_key),
                        reason="Multiple supported in single-select; seek counter-evidence",
                    ))

        return actions

    def _check_option_gaps(self, state: QAState, opt_key: str) -> List[ReflectionAction]:
        """Check if a specific option needs additional retrieval."""
        actions = []
        claim_state = state.claims[opt_key]
        opt_text = claim_state.original_text

        # Skip contradicted options entirely (no point seeking more evidence)
        # Tightened from 0.7 to 0.85: be more willing to retry borderline
        # contradicted options as the verifier may be over-strict
        if claim_state.status == "contradicted" and claim_state.confidence > 0.85:
            return actions

        # Skip well-supported options unless they have high-risk keywords
        if claim_state.status == "supported" and claim_state.confidence > 0.8:
            if not self._has_risk_needing_counter(opt_key, state):
                return actions

        # Insufficient evidence (mixed with low confidence) → retry with diversified search
        # Raised threshold from 0.4 to 0.5: also retry medium-confidence mixed options
        if claim_state.status == "mixed" and claim_state.confidence < 0.5:
            actions.append(ReflectionAction(
                tool="option_search",
                target_option=opt_key,
                query=self._build_diversified_query(state, opt_key),
                reason=f"Option {opt_key} has uncertain evidence (mixed, confidence={claim_state.confidence:.2f})",
                params={"increased_budget": True},
            ))
            # Also try metric-focused retrieval for low-confidence options
            metric_query = self._build_metric_query(state, opt_key)
            if metric_query:
                actions.append(ReflectionAction(
                    tool="metric_search",
                    target_option=opt_key,
                    query=metric_query,
                    reason=f"Metric-focused retrieval for uncertain {opt_key}",
                ))
            # Per-entity search for multi-entity uncertain options
            entity_queries = self._build_per_entity_queries(state, opt_key)
            for entity_label, entity_query in entity_queries:
                actions.append(ReflectionAction(
                    tool="option_search",
                    target_option=opt_key,
                    query=entity_query,
                    reason=f"Per-entity search for uncertain {opt_key}: {entity_label}",
                    params={"increased_budget": True, "entity": entity_label},
                ))

        # Negation/universal keywords → counter-search
        if self.config.enable_counter_search and self._has_risk_needing_counter(opt_key, state):
            # Only if we haven't already done counter-search for this option
            already_countered = any(
                t.tool_name == "counter_search" and opt_key in t.target_options
                for t in state.tool_history
            )
            if not already_countered:
                actions.append(ReflectionAction(
                    tool="counter_search",
                    target_option=opt_key,
                    query=self._build_counter_query(state, opt_key),
                    reason=f"Option {opt_key} contains risk keywords needing counter-evidence",
                ))

        return actions

    def _has_risk_needing_counter(self, opt_key: str, state: QAState) -> bool:
        """Check if option has risk flags that warrant counter-search.

        Reads risk_flags from the claim state (set by LLM or rule-based).
        Supports both formats:
        - LLM format: ["universal", "negation"]
        - Rule-based format: ["universal:均", "negation:未"]
        """
        claim_state = state.claims.get(opt_key)
        if not claim_state or not claim_state.risk_flags:
            return False

        # Normalize flags: extract the type part before ':'
        flag_types = {f.split(":")[0] for f in claim_state.risk_flags}
        return bool(flag_types & {"negation", "universal", "threshold"})

    def _build_diversified_query(self, state: QAState, opt_key: str) -> str:
        """Build a diversified query for Round 2 option_search.

        Unlike Round 1 which uses "question + option_text", Round 2 extracts
        the core claim from the option and reformulates the query to search
        from a different angle. This avoids retrieving the same results.
        """
        claim_state = state.claims[opt_key]
        opt_text = claim_state.original_text

        # Extract structured info from atomic claims
        atomic = claim_state.atomic_claims[0] if claim_state.atomic_claims else None
        parts = []

        if atomic and atomic.company:
            parts.append(atomic.company)
        if atomic and atomic.year:
            parts.append(str(atomic.year))

        # Use key metrics as the core search terms (more specific than full option text)
        if atomic and atomic.metrics:
            parts.extend(atomic.metrics[:3])

        # If we have structured info, build a focused query
        if parts:
            # Combine structured info with a condensed version of the option
            focused_query = " ".join(parts)
            return f"{focused_query} {opt_text}"

        # Fallback: use option text alone (without question stem to get different results)
        # This is different from Round 1 which uses "question + option"
        return f"{opt_text}"

    def _build_metric_query(self, state: QAState, opt_key: str) -> Optional[str]:
        """Build a metric-focused query using structured claim info.

        Strategy C: use company + year + key_metrics for precise field-level
        retrieval. Returns None if insufficient structured info is available.
        """
        claim_state = state.claims[opt_key]
        atomic = claim_state.atomic_claims[0] if claim_state.atomic_claims else None

        if not atomic:
            return None

        # Need at least key_metrics to build a meaningful metric query
        if not atomic.metrics:
            return None

        parts = []
        if atomic.company:
            parts.append(atomic.company)
        if atomic.year:
            parts.append(f"{atomic.year}年")

        # Use each metric as a focused search term
        for metric in atomic.metrics[:3]:
            parts.append(metric)

        if len(parts) < 2:
            return None

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Strategy D: Multi-entity per-entity search
    # ------------------------------------------------------------------

    # Keywords that indicate multi-entity questions (need per-entity search)
    # Note: these are checked against the QUESTION STEM, not option text
    MULTI_ENTITY_KEYWORDS = [
        "两家公司", "两份文档", "两款产品",
        "三家公司", "三份文档", "三款产品",
        "四家公司", "四份文档", "四款产品",
        "多家公司", "多份文档", "多款产品",
        "各家公司", "各产品", "各份文档",
        "均", "全部", "所有", "都",
    ]

    def _has_multi_entity(self, text: str) -> bool:
        """Detect if text (question or option) contains multi-entity language."""
        return any(kw in text for kw in self.MULTI_ENTITY_KEYWORDS)

    def _build_per_entity_queries(
        self, state: QAState, opt_key: str,
    ) -> List[tuple]:
        """Build separate queries per entity for multi-entity questions.

        Strategy D: two scenarios:
        1. Question stem contains multi-entity language (e.g. "四款产品") AND
           each option IS an entity (e.g. option A = "平安智盈金生") →
           search using the option text itself as the entity identifier
        2. Option text contains multi-entity language (e.g. "两家公司均...") →
           search using doc_scope entries as entity identifiers

        Returns a list of (entity_label, query) tuples. Empty if no per-entity
        search is needed.
        """
        claim_state = state.claims[opt_key]
        opt_text = claim_state.original_text

        # Scenario 1: question stem has multi-entity language, each option is an entity
        # This covers cases like ins_a_019 where options are product/company names
        if self._has_multi_entity(state.question):
            # Extract a focused keyword from the question (what we're looking for)
            # Use the option text (entity name) + a key phrase from the question
            q_keywords = self._extract_question_keywords(state.question)
            entity_query = f"{opt_text} {q_keywords}"
            return [(opt_text, entity_query)]

        # Scenario 2: option text itself has multi-entity language
        if self._has_multi_entity(opt_text):
            # Need at least 2 doc_ids in scope to do per-entity search
            if not state.doc_scope or len(state.doc_scope) < 2:
                return []
            queries = []
            for doc_id in state.doc_scope[:4]:  # cap at 4 entities
                entity_query = f"{opt_text}"
                queries.append((doc_id, entity_query))
            return queries

        return []

    def _extract_question_keywords(self, question: str) -> str:
        """Extract key predicate/condition from question stem for entity search.

        For example:
        - "假设四款产品的投保人均在犹豫期内解除合同，则以下哪些产品会退还全部已交保险费？"
          → "犹豫期 退还 已交保险费"
        - "以下哪些公司的净利润超过10亿？" → "净利润"
        """
        import re
        # Remove common question prefixes/suffixes
        q = re.sub(r'^假设[^，。]*[，。]', '', question)
        q = re.sub(r'以下哪[些个].*?[，。]?', '', q)
        q = re.sub(r'[（(][^）)]*[）)]', '', q)  # remove parentheses
        q = re.sub(r'^则[^，。]*?(?=会|是|有|包含|属于|超过|低于|达到|退还|支付|分配)', '', q.strip())  # remove leading 则...
        q = re.sub(r'^[则哪些个]*产品[会是]', '', q.strip())  # remove "则产品会"
        # Take the last meaningful clause (usually the actual question)
        parts = re.split(r'[，。？]', q)
        meaningful = [p.strip() for p in parts if len(p.strip()) > 4]
        if meaningful:
            return meaningful[-1][:30]  # cap at 30 chars
        return question[-30:]  # fallback: last 30 chars of question

    def _build_counter_query(self, state: QAState, opt_key: str) -> str:
        """Build a counter-evidence query for an option.

        Strategy: negate the claim or search for contradicting evidence.
        """
        opt_text = state.claims[opt_key].original_text

        # Simple negation: flip key words
        counter_text = opt_text
        negation_pairs = [
            ("未实施", "实施了"),
            ("不实施", "实施"),
            ("不送", "送"),
            ("不派", "派"),
            ("未", "已"),
            ("不", ""),
            ("无", "有"),
            ("均", "并非全部"),
            ("全部", "部分"),
            ("所有", "部分"),
            ("超过", "未超过"),
            ("低于", "高于"),
            ("不超过", "超过"),
        ]

        for pos, neg in negation_pairs:
            if pos in counter_text:
                counter_text = counter_text.replace(pos, neg, 1)
                break

        return f"{state.question}\n反证检索: {counter_text}"

    def _deduplicate_actions(self, actions: List[ReflectionAction]) -> List[ReflectionAction]:
        """Remove duplicate actions (same tool + same option + same entity)."""
        seen = set()
        unique = []
        for action in actions:
            entity = action.params.get("entity", "") if action.params else ""
            key = (action.tool, action.target_option, entity)
            if key not in seen:
                seen.add(key)
                unique.append(action)
        return unique
