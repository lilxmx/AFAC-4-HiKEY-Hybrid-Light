"""m06_claim_verify configuration.

Inherits all retrieval/LLM parameters from m05 and adds agent-specific
configuration via AgentConfig.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import DOMAINS, PROJECT_ROOT
from methods._shared.qa_agent.config import AgentConfig

# --- Identity / IO --------------------------------------------------------

METHOD_NAME = "m06_claim_verify"
METHOD_ID = METHOD_NAME
METHOD_ROOT = Path(__file__).resolve().parent

# --- LLM ------------------------------------------------------------------

MODEL_NAME = os.getenv("M06_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# --- Retrieval (reuse m05's HiKEY cache) ----------------------------------

HIKEY_INDEX_DIR = PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"
RETRIEVER_NAME = os.getenv("M06_RETRIEVER", "hikey")
RETRIEVER_EXTRA_KWARGS: dict = {}

# --- Execution -------------------------------------------------------------

CODER = os.getenv("CODER", "qhl")
RUN_DESC = os.getenv("M06_RUN_DESC", "claim-verify-v1")
CONCURRENCY = int(os.getenv("M06_CONCURRENCY", "10"))

# --- HiKEY retriever params -----------------------------------------------

MAX_SIBLINGS = int(os.getenv("M06_MAX_SIBLINGS", "12"))
SKIP_DOC_ROUTING = os.getenv("M06_SKIP_DOC_ROUTING", "1") != "0"

# --- Agent Config ----------------------------------------------------------
# All agent-specific parameters are in this dataclass.
# Override via M06_* env vars or by editing defaults here.

AGENT_CONFIG = AgentConfig(
    version="m06",

    # Execution control
    max_rounds=int(os.getenv("M06_MAX_ROUNDS", "2")),
    max_tool_calls=int(os.getenv("M06_MAX_TOOL_CALLS", "12")),

    # Retrieval
    global_topk=int(os.getenv("M06_GLOBAL_TOPK", "10")),
    option_topk=int(os.getenv("M06_OPTION_TOPK", "10")),
    option_fallback_topk=int(os.getenv("M06_OPTION_FALLBACK_TOPK", "8")),
    counter_topk=int(os.getenv("M06_COUNTER_TOPK", "8")),
    final_evidence_topk=int(os.getenv("M06_FINAL_EVIDENCE_TOPK", "28")),

    # Evidence budget
    max_evidence_per_claim=int(os.getenv("M06_MAX_EVIDENCE_PER_CLAIM", "8")),
    max_raw_evidence_for_verifier=int(os.getenv("M06_MAX_RAW_EVIDENCE_FOR_VERIFIER", "6")),
    evidence_token_budget=int(os.getenv("M06_EVIDENCE_TOKEN_BUDGET", "24000")),

    # Feature toggles
    enable_grouped_retrieval=os.getenv("M06_ENABLE_GROUPED_RETRIEVAL", "1") != "0",
    enable_option_fallback=os.getenv("M06_ENABLE_OPTION_FALLBACK", "1") != "0",
    enable_counter_search=os.getenv("M06_ENABLE_COUNTER_SEARCH", "1") != "0",
    enable_field_lookup=False,
    enable_calculator=False,
    enable_vlm_inspect=False,

    # LLM
    use_llm_planner=False,  # v1: rule-based planner
    use_llm_verifier=True,
    use_llm_aggregator=False,  # v1: rule-based aggregator

    # LLM params
    model_name=MODEL_NAME,
    temperature=float(os.getenv("M06_TEMPERATURE", "0.1")),
    enable_thinking=os.getenv("M06_ENABLE_THINKING", "1") != "0",
    thinking_budget=int(os.getenv("M06_THINKING_BUDGET", "10000")),
    max_output_tokens=int(os.getenv("M06_MAX_OUTPUT_TOKENS", "12000")),

    # Debug
    debug_trace=os.getenv("M06_DEBUG_TRACE", "1") != "0",
    save_intermediate=os.getenv("M06_SAVE_INTERMEDIATE", "1") != "0",
)
