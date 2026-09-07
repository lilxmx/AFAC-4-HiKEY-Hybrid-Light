"""
m01_baseline_qwen - configuration.

Inherits common paths/QUESTION_FILES/proxy from _shared.config_base.
Only adds m01-specific knobs (chunking, retrieval params, output dirs).
"""
import os
import sys
from pathlib import Path

# Make `methods.*` importable when running `python run.py` from the method dir.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import (  # noqa: F401
    PROJECT_ROOT,
    PROJECT_BM25_INDEX_DIR,
)

# ============================================================
# Method identity
# ============================================================
METHOD_NAME = "m01_baseline_qwen"

# ============================================================
# LLM (Qwen-plus via Dashscope)
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.1

# ============================================================
# Output paths (per-method)
# ============================================================
METHOD_ROOT = Path(__file__).resolve().parent.parent  # methods/m01_baseline_qwen/
OUTPUT_DIR = METHOD_ROOT / "output"
LOGS_DIR = METHOD_ROOT / "logs"

# Use the project-level shared BM25 index dir so m01/m02 reuse the same indices.
INDICES_DIR = PROJECT_BM25_INDEX_DIR

# ============================================================
# Chunking parameters
# ============================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ============================================================
# Retrieval parameters
# ============================================================
BM25_TOP_K = 5
MAX_EVIDENCE_CHARS = 3000

# ============================================================
# Concurrency
# ============================================================
CONCURRENCY = 5
