"""Shared QA Agent framework for claim-centric verification pipelines.

This package provides reusable building blocks for m06+ QA methods:
- schemas: QAState, OptionClaimState, AtomicClaim, ClaimMemory, EvidenceGroup
- config: AgentConfig dataclass
- planner: QuestionPlanner, ClaimExtractor, EvidenceGroupBuilder
- verifier: ClaimVerifier (LLM-based option-level verification)
- reflector: ReflectionController (gap detection + round-2 action planning)
- aggregator: AnswerAggregator + sanitizer
- compressor: EvidenceCompressor (raw evidence → ClaimMemory)
- workflow: BaseAgentWorkflow (inheritable multi-round pipeline)
"""
from .schemas import (
    QAState,
    OptionClaimState,
    AtomicClaim,
    ClaimMemory,
    EvidenceGroup,
    ToolCallRecord,
    ReflectionAction,
)
from .config import AgentConfig
from .workflow import BaseAgentWorkflow

__all__ = [
    "QAState",
    "OptionClaimState",
    "AtomicClaim",
    "ClaimMemory",
    "EvidenceGroup",
    "ToolCallRecord",
    "ReflectionAction",
    "AgentConfig",
    "BaseAgentWorkflow",
]
