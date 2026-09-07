"""
m02_gpt41_golden - configuration.

Inherits common paths/QUESTION_FILES/proxy from _shared.config_base.
Adds m02-specific: GPT-4.1 (Azure) reasoning + text-embedding-3-large (OpenRouter)
hybrid retrieval + voting.
"""
import os
import sys
from pathlib import Path

# Make `methods.*` importable when running `python run.py` directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import (  # noqa: F401
    PROJECT_ROOT,
    PROJECT_BM25_INDEX_DIR,
    PROJECT_EMBEDDING_CACHE_DIR,
)

# ============================================================
# Method identity
# ============================================================
METHOD_NAME = "m02_gpt41_golden"

# ============================================================
# Azure OpenAI Configuration (GPT-4.1 for reasoning)
# ============================================================
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_BASE_URL = os.getenv("AZURE_BASE_URL")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
REASONING_MODEL = "gpt-4.1"

# ============================================================
# OpenRouter Configuration (embeddings)
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = "openai/text-embedding-3-large"
EMBEDDING_DIM = 3072

# ============================================================
# Output paths (per-method)
# ============================================================
METHOD_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = METHOD_ROOT / "output"
LOGS_DIR = METHOD_ROOT / "logs"

# Re-use the project-level shared BM25 indices built by m01.
BASELINE_INDICES_DIR = PROJECT_BM25_INDEX_DIR

# Embedding cache lives here (per-method or shared - keep per-method for now)
EMBEDDING_CACHE_DIR = METHOD_ROOT / "embedding_cache"

# ============================================================
# Retrieval parameters
# ============================================================
BM25_TOP_K = 10           # BM25 retrieval top-k (boosted vs m01)
EMBEDDING_TOP_K = 10      # embedding retrieval top-k
RRF_K = 60                # RRF constant (standard value)
FINAL_TOP_K = 8           # final merged top-k
MAX_EVIDENCE_CHARS = 6000 # generous for GPT-4.1's 1M context

# ============================================================
# Reasoning parameters
# ============================================================
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.3
VOTING_ROUNDS = 3
CONCURRENCY = 5

# ============================================================
# Financial Contracts pipeline (Skill v6.0)
# ============================================================
FC_MAX_EVIDENCE_CHARS_PER_DOC = 4000
FC_VOTING_ROUNDS = 3
FC_ENABLE_QUERY_PLANNER = True

# ============================================================
# Embedding parameters
# ============================================================
EMBEDDING_BATCH_SIZE = 20
