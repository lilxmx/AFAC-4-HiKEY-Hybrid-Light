"""Shared pipeline infrastructure: BaseRunner + result writers."""
from .base_runner import BaseRunner, QuestionResult
from .output import (
    save_answer_csv,
    save_evidence_json,
    save_run_summary,
    compute_domain_stats,
)

__all__ = [
    "BaseRunner",
    "QuestionResult",
    "save_answer_csv",
    "save_evidence_json",
    "save_run_summary",
    "compute_domain_stats",
]
