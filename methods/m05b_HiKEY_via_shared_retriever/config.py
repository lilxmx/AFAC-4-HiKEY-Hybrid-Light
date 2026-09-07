"""m05b configuration.

Re-exports the m05 config verbatim and only changes:
- ``METHOD_NAME`` / ``METHOD_ROOT``  — so output goes to a separate dir
- ``RUN_DESC``                       — so summaries are easy to tell apart
- ``RETRIEVER_NAME``                 — selects the shared-retriever impl

All other knobs stay tied to m05's config module on purpose: the goal of
m05b is to demonstrate that swapping the retrieval layer leaves
behaviour intact, so we deliberately do *not* introduce M05B_* env
overrides for retrieval/LLM parameters. If you want different
hyper-parameters, set the corresponding M05_* vars; m05b will pick them
up automatically.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Pull every constant from m05 so we never silently drift.
from methods.m05_HiKEY_question_option.config import *  # noqa: F401,F403
from methods.m05_HiKEY_question_option import config as _m05_config  # noqa: F401

# --- Identity / IO --------------------------------------------------------

METHOD_NAME = "m05b_HiKEY_via_shared_retriever"
METHOD_ID = METHOD_NAME
METHOD_ROOT = Path(__file__).resolve().parent

RUN_DESC = os.getenv("M05B_RUN_DESC", "shared-retriever-port")

# --- Retriever selection (the only new knob vs m05) -----------------------

# Name registered in ``methods._shared.retrieval``. Default keeps m05
# behaviour. Set M05B_RETRIEVER=hybrid (after a teammate registers it)
# to swap implementations without touching pipeline code.
RETRIEVER_NAME = os.getenv("M05B_RETRIEVER", "hikey")

# Free-form kwargs forwarded to ``build_retriever``. Most users won't
# need this; it is here so future retrievers can be parametrised from
# env without code changes.
RETRIEVER_EXTRA_KWARGS: dict = {}
