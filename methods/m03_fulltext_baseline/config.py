"""
m03_fulltext_baseline - configuration.

Inherits all common paths/QUESTION_FILES/proxy/etc. from _shared.config_base.
Only adds m03-specific knobs.
"""
import sys
from pathlib import Path

# Make `methods.*` importable when running `python run.py` directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Pull in everything common (paths, DOMAINS, QUESTION_FILES, proxy, token_budget, ...)
from methods._shared.config_base import *  # noqa: F401,F403  (re-export)
from methods._shared.config_base import PROJECT_ROOT  # noqa: F401  (used by sys.path users)

import os

# ----------------------------------------------------------------
# Method-specific settings
# ----------------------------------------------------------------
METHOD_NAME = "m03_fulltext_baseline"

# LLM: Qwen-plus via Dashscope (128K context)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_NAME = "qwen-plus"

# Output paths (per-method)
METHOD_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = METHOD_ROOT / "output"
LOGS_DIR = METHOD_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Inference knobs
MAX_OUTPUT_TOKENS = 64       # answers are 1-4 letters, plenty of room
TEMPERATURE = 0.0            # deterministic
CONCURRENCY = 5              # parallel API calls
