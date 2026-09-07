"""Core data structures for the Claim-Centric QA Agent.

These schemas define the state that flows through the agent workflow.
They are intentionally decoupled from the retrieval layer — the agent
consumes EvidenceItem/EvidencePack from `methods._shared.retrieval.api`
and wraps them in its own claim-centric state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Atomic Claim: smallest verifiable unit within an option
# ---------------------------------------------------------------------------

@dataclass
class AtomicClaim:
    """One atomic, independently verifiable statement extracted from an option.

    Example:
        Option C: "两家公司在对应年份均未实施资本公积金转增股本"
        → AtomicClaim(claim_id="C1", text="宁德时代2024未实施资本公积金转增股本", ...)
        → AtomicClaim(claim_id="C2", text="美的集团2025未实施资本公积金转增股本", ...)
    """

    claim_id: str
    text: str
    option: str  # parent option letter: A/B/C/D

    # Structured metadata (populated by planner)
    company: Optional[str] = None
    year: Optional[int] = None
    doc_id: Optional[str] = None
    topic: Optional[str] = None
    metrics: List[str] = field(default_factory=list)

    # Claim operation type (helps reflector decide tools)
    operation: Optional[str] = None
    # "lookup" | "compare" | "calculate" | "negate" | "universal" | "threshold"

    # Verification state (updated by verifier)
    status: str = "unknown"
    # "unknown" | "supported" | "contradicted" | "mixed"
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Option Claim State: per-option aggregation of atomic claims
# ---------------------------------------------------------------------------

@dataclass
class OptionClaimState:
    """Verification state for one option (A/B/C/D).

    An option may contain one or more atomic claims. The option-level
    status is derived from its atomic claims:
    - All supported → supported
    - Any contradicted → contradicted
    - Mix of supported + uncertain → mixed
    - All uncertain → mixed (with low confidence)
    """

    option: str  # "A", "B", "C", "D"
    original_text: str
    atomic_claims: List[AtomicClaim] = field(default_factory=list)

    # Aggregated status (set by verifier/aggregator)
    status: str = "unknown"
    # "unknown" | "supported" | "contradicted" | "mixed"
    confidence: float = 0.0

    # Evidence tracking
    support_evidence_ids: List[str] = field(default_factory=list)
    counter_evidence_ids: List[str] = field(default_factory=list)

    # Gap detection (set by reflector)
    missing_fields: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    # Verifier reasoning (for debug trace)
    reason: str = ""


# ---------------------------------------------------------------------------
# Evidence Group: shared retrieval unit across multiple claims
# ---------------------------------------------------------------------------

@dataclass
class EvidenceGroup:
    """A retrieval task that covers one or more claims.

    Instead of retrieving per-option (m05 style), m06 groups claims by
    shared (doc_id, company, year, topic) and issues fewer, more focused
    queries. This reduces redundant retrieval while improving recall.
    """

    group_id: str
    group_type: str
    # "global" | "topic" | "field" | "counter" | "option_fallback" | "vlm"

    query: str
    doc_scope: List[str] = field(default_factory=list)
    covers_options: List[str] = field(default_factory=list)
    covers_claim_ids: List[str] = field(default_factory=list)

    # Structured hints
    company: Optional[str] = None
    year: Optional[int] = None
    topic: Optional[str] = None
    metrics: List[str] = field(default_factory=list)

    # Retrieval parameters
    topk: int = 10
    mode: str = "global"  # maps to RetrievalMode


# ---------------------------------------------------------------------------
# Claim Memory: compressed evidence state per option
# ---------------------------------------------------------------------------

@dataclass
class ClaimMemory:
    """Compressed working memory for one option after evidence distribution.

    This is what the verifier actually reads — not the full raw evidence,
    but a distilled summary of key facts, support/counter evidence, and
    identified gaps.
    """

    option: str
    claim_summary: str

    # Verification state
    status: str = "unknown"
    confidence: float = 0.0

    # Compressed facts
    key_facts: List[str] = field(default_factory=list)
    support_evidence_ids: List[str] = field(default_factory=list)
    counter_evidence_ids: List[str] = field(default_factory=list)

    # Gaps
    missing_fields: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool Call Record: trace of every tool invocation
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """Record of a single tool invocation for traceability."""

    tool_name: str
    round_id: int
    query: str = ""
    target_options: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result_count: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Reflection Action: what the reflector tells round 2 to do
# ---------------------------------------------------------------------------

@dataclass
class ReflectionAction:
    """One action the reflector requests for round 2."""

    tool: str
    # "counter_search" | "option_search" | "field_lookup" | "calculator" | "vlm_inspect"

    target_option: str
    query: str = ""
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QAState: the master state object for one question
# ---------------------------------------------------------------------------

@dataclass
class QAState:
    """Complete state for processing one question through the agent workflow.

    This is the single object that flows through all pipeline stages:
    plan → retrieve → compress → verify → reflect → retrieve2 → aggregate.
    """

    # --- Question identity ---
    qid: str
    question: str
    options: Dict[str, str]  # {"A": "...", "B": "...", ...}
    answer_format: str  # "mcq" | "multi" | "tf"
    domain: str = ""
    doc_ids: List[str] = field(default_factory=list)

    # --- Planning output ---
    question_type: Optional[str] = None
    doc_scope: List[str] = field(default_factory=list)
    global_query: str = ""
    required_fields: List[str] = field(default_factory=list)

    # --- Claim state ---
    claims: Dict[str, OptionClaimState] = field(default_factory=dict)
    # key = option letter ("A", "B", ...)

    evidence_groups: List[EvidenceGroup] = field(default_factory=list)

    # --- Evidence pool ---
    # raw evidence items keyed by evidence_id (from retrieval API)
    raw_evidence_pool: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # compressed per-option memory
    working_memory: Dict[str, ClaimMemory] = field(default_factory=dict)

    # --- Execution trace ---
    tool_history: List[ToolCallRecord] = field(default_factory=list)
    round_id: int = 0
    reflection_actions: List[ReflectionAction] = field(default_factory=list)

    # --- Output ---
    final_answer: Optional[str] = None
    aggregator_reasoning: str = ""

    # --- Token tracking ---
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # --- Debug ---
    debug_trace: Dict[str, Any] = field(default_factory=dict)

    # --- Factory ---
    @classmethod
    def from_question(cls, q: Dict[str, Any]) -> "QAState":
        """Create a QAState from a raw question dict (as loaded from JSON)."""
        from methods._shared.config_base import infer_domain_from_qid

        qid = q["qid"]
        domain = q.get("domain") or infer_domain_from_qid(qid)
        doc_ids = q.get("doc_ids", []) or []
        if not doc_ids and q.get("doc_id"):
            doc_ids = [q["doc_id"]]

        return cls(
            qid=qid,
            question=q.get("question", ""),
            options=q.get("options", {}),
            answer_format=q.get("answer_format", "mcq"),
            domain=domain,
            doc_ids=doc_ids,
            doc_scope=list(doc_ids),
            # Use the dataset-provided question type directly
            question_type=q.get("type") or None,
        )

    def add_tokens(self, prompt: int = 0, completion: int = 0, total: int = 0) -> None:
        """Accumulate token usage from an LLM call."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

    def to_result_dict(self) -> Dict[str, Any]:
        """Convert to the dict format BaseRunner.process_question expects."""
        return {
            "qid": self.qid,
            "answer": self.final_answer or "A",
            "raw_answer": self.aggregator_reasoning,
            "domain": self.domain,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "evidence": [
                self.raw_evidence_pool.get(eid, {}).get("content", "")
                for eid in list(self.raw_evidence_pool.keys())[:5]
            ],
            "retrieval_meta": {
                "tool_calls": len(self.tool_history),
                "rounds": self.round_id,
                "evidence_pool_size": len(self.raw_evidence_pool),
                "claims": {
                    opt: {"status": cs.status, "confidence": cs.confidence}
                    for opt, cs in self.claims.items()
                },
                "reflection_actions": len(self.reflection_actions),
            },
        }
