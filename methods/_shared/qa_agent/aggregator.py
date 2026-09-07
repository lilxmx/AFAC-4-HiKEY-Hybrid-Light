"""Answer Aggregator: combine claim verification results into final answer.

The aggregator uses rule-based logic (not LLM) to produce the final
answer from per-option verification states. This keeps the final step
deterministic and fast.

Rules:
- Multi-select: choose all "supported" options + "mixed" with high confidence
- Single-select: choose the single "supported" option (or best candidate)
- True/False: supported → correct, contradicted → incorrect
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from .schemas import QAState
from .config import AgentConfig

logger = logging.getLogger(__name__)


class AnswerAggregator:
    """Aggregate per-option verification results into a final answer."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def aggregate(self, state: QAState) -> QAState:
        """Produce final_answer on the state based on claim statuses."""
        answer_format = state.answer_format

        if answer_format == "multi":
            answer = self._aggregate_multi(state)
        elif answer_format == "tf":
            answer = self._aggregate_tf(state)
        else:  # mcq
            answer = self._aggregate_mcq(state)

        state.final_answer = sanitize_answer(answer, answer_format)
        return state

    def _aggregate_multi(self, state: QAState) -> str:
        """Multi-select: choose all supported options.

        Strategy:
        1. All 'supported' options are selected
        2. 'mixed' options with high confidence are also included (they likely
           have partial support that the verifier couldn't fully confirm)
        3. Fallback: pick highest confidence non-contradicted options
        """
        supported = []
        mixed_high = []
        mixed_all = []
        contradicted = []

        for opt_key, claim_state in sorted(state.claims.items()):
            if claim_state.status == "supported":
                supported.append((opt_key, claim_state.confidence))
            elif claim_state.status == "mixed" and claim_state.confidence >= 0.6:
                mixed_high.append((opt_key, claim_state.confidence))
            elif claim_state.status == "mixed":
                mixed_all.append((opt_key, claim_state.confidence))
            elif claim_state.status == "contradicted":
                contradicted.append((opt_key, claim_state.confidence))

        # Primary: all supported options
        result = [k for k, _ in supported]

        # Also include mixed-high options (partial support is still support
        # in multi-select context)
        for k, c in mixed_high:
            if k not in result:
                result.append(k)

        if result:
            return "".join(sorted(result))

        # Fallback: if nothing is supported, pick highest confidence non-contradicted
        best_fallback = [(k, c) for k, c in mixed_all if c >= 0.4]
        best_fallback.extend(mixed_high)
        if best_fallback:
            best_fallback.sort(key=lambda x: x[1], reverse=True)
            picks = [k for k, c in best_fallback[:3]]
            if picks:
                return "".join(sorted(picks))

        # Last resort: return "A"
        return "A"

    def _aggregate_mcq(self, state: QAState) -> str:
        """Single-select: choose the best supported option."""
        supported = []
        candidates = []

        for opt_key, claim_state in sorted(state.claims.items()):
            if claim_state.status == "supported":
                supported.append((opt_key, claim_state.confidence))
            if claim_state.status != "contradicted":
                candidates.append((opt_key, claim_state.confidence))

        # Ideal: exactly one supported
        if len(supported) == 1:
            return supported[0][0]

        # Multiple supported: pick highest confidence
        if supported:
            supported.sort(key=lambda x: x[1], reverse=True)
            return supported[0][0]

        # No supported: pick highest confidence non-contradicted
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return "A"

    def _aggregate_tf(self, state: QAState) -> str:
        """True/False: check if the proposition is supported or contradicted.

        Convention: A = correct/true, B = incorrect/false
        (actual meaning depends on the question's option text)
        """
        # For TF questions, we typically have options A and B
        # A is usually "正确/对" and B is usually "错误/不对"
        # We verify the question stem as a claim

        a_state = state.claims.get("A")
        b_state = state.claims.get("B")

        if a_state and b_state:
            if a_state.status == "supported" and b_state.status != "supported":
                return "A"
            if b_state.status == "supported" and a_state.status != "supported":
                return "B"
            if a_state.status == "contradicted":
                return "B"
            if b_state.status == "contradicted":
                return "A"
            # Both mixed/uncertain: pick higher confidence
            if a_state.confidence >= b_state.confidence:
                return "A"
            return "B"

        return "A"


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

def sanitize_answer(answer: str, answer_format: str) -> str:
    """Clean and validate the final answer string.

    Ensures:
    - Only valid letters (A-D) are present
    - Multi-select answers are sorted and deduplicated
    - Empty answers get a safe fallback
    """
    if not answer:
        return "A"

    # Extract only valid option letters
    valid_letters = set("ABCD")
    letters = [c.upper() for c in answer if c.upper() in valid_letters]

    if not letters:
        return "A"

    # Deduplicate and sort
    letters = sorted(set(letters))

    if answer_format == "mcq":
        # Single select: only keep first
        return letters[0]
    elif answer_format == "tf":
        # True/false: only A or B
        tf_letters = [l for l in letters if l in ("A", "B")]
        return tf_letters[0] if tf_letters else "A"
    else:
        # Multi-select: return all
        return "".join(letters)
