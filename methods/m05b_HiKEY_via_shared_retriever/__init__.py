"""m05b_HiKEY_via_shared_retriever package.

Behaviour-equivalent rewrite of m05 that goes through the shared
``methods._shared.retrieval`` API instead of talking to the m04 HiKEY
parser directly. Useful as the entry point for m06+ work and as a safe
proof that the shared API does not regress m05 numbers.
"""
