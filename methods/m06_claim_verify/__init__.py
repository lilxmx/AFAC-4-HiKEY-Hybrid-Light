"""m06_claim_verify: Claim-Centric Verification Agent.

First implementation of the shared QA Agent framework. Uses:
- Option-level claim extraction (rule-based, v1)
- Grouped retrieval (global + per-option, m05-compatible)
- LLM-based per-option verification
- Reflection-driven counter-search for risky options
- Rule-based answer aggregation
"""
