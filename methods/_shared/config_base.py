"""
Shared base configuration for all methods.

Provides:
- Common path constants (PROJECT_ROOT, RAW_BASE, QUESTIONS_DIR, etc.)
- QUESTION_FILES / DOMAINS mapping
- Proxy & .env auto-loading helpers
- TOKEN_BUDGET and token-score utilities

Each method's config.py should:
    from methods._shared.config_base import *
and then add method-specific constants on top.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ============================================================
# Project root resolution
# ============================================================
# This file lives at: AFAC-4/methods/_shared/config_base.py
# So project root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # AFAC-4/

# ============================================================
# Proxy configuration (Tencent internal network needs proxy for public APIs)
# ============================================================
PROXY_URL = "http://star-proxy.oa.com:3128"


def setup_proxy(proxy_url: str = PROXY_URL) -> None:
    """Set http_proxy / https_proxy env vars if not already set."""
    if not os.getenv("http_proxy"):
        os.environ["http_proxy"] = proxy_url
    if not os.getenv("https_proxy"):
        os.environ["https_proxy"] = proxy_url


def load_project_dotenv() -> None:
    """Load .env from project root if present."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


# Auto-apply on import (matches the behavior all method configs already had)
load_project_dotenv()
if os.getenv("AFAC_USE_PROXY", "0") == "1":
    setup_proxy()
# setup_proxy()

# ============================================================
# Dataset paths
# ============================================================
DATASET_ROOT = PROJECT_ROOT / "public_dataset_upload"
RAW_BASE = DATASET_ROOT / "raw"
QUESTIONS_DIR = DATASET_ROOT / "questions" / "group_a"

# ============================================================
# Domain & question file mapping
# ============================================================
DOMAINS = [
    "insurance",
    "regulatory",
    "financial_contracts",
    "financial_reports",
    "research",
]

QUESTION_FILES = {
    "insurance": QUESTIONS_DIR / "insurance_questions.json",
    "regulatory": QUESTIONS_DIR / "regulatory_questions.json",
    "financial_contracts": QUESTIONS_DIR / "financial_contracts_questions.json",
    "financial_reports": QUESTIONS_DIR / "financial_reports_questions.json",
    "research": QUESTIONS_DIR / "research_questions.json",
}

# Mapping from qid prefix to domain (used in summary stats)
QID_PREFIX_TO_DOMAIN = {
    "ins_": "insurance",
    "reg_": "regulatory",
    "fc_": "financial_contracts",
    "fin_": "financial_reports",
    "res_": "research",
}


def infer_domain_from_qid(qid: str) -> str:
    """Infer the domain from a qid by its prefix."""
    for prefix, domain in QID_PREFIX_TO_DOMAIN.items():
        if qid.startswith(prefix):
            return domain
    return "unknown"


# ============================================================
# Shared caches
# ============================================================
# (a) Parsed text cache (lives under methods/_shared/_cache/, scoped to methods/)
# New layout: _cache/{domain}/{doc_id}/fulltext.txt + page_XXX.txt
SHARED_CACHE_ROOT = Path(__file__).resolve().parent / "_cache"
# Legacy constants kept for backward compatibility (point to cache root)
SHARED_FULLTEXT_CACHE = SHARED_CACHE_ROOT
SHARED_PARSE_BLOCKS_CACHE = SHARED_CACHE_ROOT

# (b) Project-level shared caches that already exist (BM25 index built once, reused
#     across m01/m02). Path matches the existing AFAC-4/cache/bm25_index/.
PROJECT_CACHE_ROOT = PROJECT_ROOT / "cache"
PROJECT_BM25_INDEX_DIR = PROJECT_CACHE_ROOT / "bm25_index"
PROJECT_EMBEDDING_CACHE_DIR = PROJECT_CACHE_ROOT / "embeddings"


# ============================================================
# Scoring utilities
# ============================================================
TOKEN_BUDGET = 5_000_000  # Official competition budget


def compute_token_score(total_tokens: int, budget: int = TOKEN_BUDGET) -> float:
    """TokenScore = max(0, min(1, (Budget - TotalTokens) / Budget))."""
    if total_tokens <= 0:
        return 0.0
    return max(0.0, min(1.0, (budget - total_tokens) / budget))


def estimate_final_multiplier(total_tokens: int, budget: int = TOKEN_BUDGET) -> float:
    """FinalScore multiplier = 0.7 + 0.3 * TokenScore."""
    return 0.7 + 0.3 * compute_token_score(total_tokens, budget)


# ============================================================
# Concurrency default (methods may override)
# ============================================================
DEFAULT_CONCURRENCY = 5


__all__ = [
    "PROJECT_ROOT",
    "PROXY_URL",
    "setup_proxy",
    "load_project_dotenv",
    "DATASET_ROOT",
    "RAW_BASE",
    "QUESTIONS_DIR",
    "DOMAINS",
    "QUESTION_FILES",
    "QID_PREFIX_TO_DOMAIN",
    "infer_domain_from_qid",
    "SHARED_CACHE_ROOT",
    "SHARED_CACHE_ROOT",
    "SHARED_FULLTEXT_CACHE",
    "SHARED_PARSE_BLOCKS_CACHE",
    "PROJECT_CACHE_ROOT",
    "PROJECT_BM25_INDEX_DIR",
    "PROJECT_EMBEDDING_CACHE_DIR",
    "TOKEN_BUDGET",
    "compute_token_score",
    "estimate_final_multiplier",
    "DEFAULT_CONCURRENCY",
]
