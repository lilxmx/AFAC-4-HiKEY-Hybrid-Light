"""
m05_HiKEY_question_option configuration.

This method reuses the HiKEY cache built for m04, but retrieves evidence with
question + option queries before sending one deduplicated evidence pack to Qwen.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import DOMAINS, PROJECT_ROOT

METHOD_NAME = "m05_HiKEY_question_option"
METHOD_ID = METHOD_NAME
METHOD_ROOT = Path(__file__).resolve().parent

MODEL_NAME = os.getenv("M05_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Reuse the m04 HiKEY parser cache: doc_card / sections / units / field_cards.
HIKEY_INDEX_DIR = PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"

# Output owner. BaseRunner will write to runs/{CODER}/{METHOD_NAME}/{run_id}/.
CODER = os.getenv("CODER", "qhl")
RUN_DESC = os.getenv("M05_RUN_DESC", "question-option")

CONCURRENCY = int(os.getenv("M05_CONCURRENCY", "10"))
TEMPERATURE = float(os.getenv("M05_TEMPERATURE", "0.1"))

# Qwen thinking mode follows m04 defaults. Disable via M05_ENABLE_THINKING=0.
ENABLE_THINKING = os.getenv("M05_ENABLE_THINKING", "1") != "0"
THINKING_BUDGET = int(os.getenv("M05_THINKING_BUDGET", "10000"))
MAX_OUTPUT_TOKENS = int(os.getenv("M05_MAX_OUTPUT_TOKENS", "12000"))
MAX_OUTPUT_TOKENS_REFLECTION = int(os.getenv("M05_MAX_OUTPUT_TOKENS_REFLECTION", "16000"))

# Retrieval: search question and question+each-option, then deduplicate.
GLOBAL_TOPK = int(os.getenv("M05_GLOBAL_TOPK", "10"))
OPTION_TOPK = int(os.getenv("M05_OPTION_TOPK", "10"))
FINAL_EVIDENCE_TOPK = int(os.getenv("M05_FINAL_EVIDENCE_TOPK", "28"))
MAX_SIBLINGS = int(os.getenv("M05_MAX_SIBLINGS", "12"))

# Keep the final prompt bounded even though we run multiple local searches.
EVIDENCE_TOKEN_BUDGET = int(os.getenv("M05_EVIDENCE_TOKEN_BUDGET", "24000"))
SKIP_DOC_ROUTING = os.getenv("M05_SKIP_DOC_ROUTING", "1") != "0"

HIKEY_DOMAINS = ["financial_reports", "insurance", "regulatory", "financial_contracts", "research"]

SYSTEM_PROMPT = """你是一位金融文档分析专家。你必须严格根据给定证据回答选择题。

只输出答案字母，不要输出解释。
单选题输出一个字母，例如 A。
多选题输出所有正确选项字母，按字母顺序排列，例如 ACD。
判断题输出 A 或 B，以题目选项含义为准。"""

SYSTEM_PROMPT_REFLECTION = """你是一位金融文档分析专家。请严格根据证据逐项核验选项。
最后一行只输出最终答案字母。"""

FORMAT_HINTS = {
    "mcq": "本题是单选题，只能从 A/B/C/D 中选择唯一正确答案。",
    "multi": "本题是多选题，需要选择所有正确选项，答案按字母顺序排列，不要遗漏正确选项。",
    "tf": "本题是判断题，请按题目选项含义输出 A 或 B。",
}

DOMAIN_HINTS = {
    "insurance": "注意保险责任、免责条款、等待期、赔付条件、责任范围和例外条件。",
    "regulatory": "注意法规适用范围、义务强度、时限要求、例外条件，以及“应当/可以/不得”的区别。",
    "financial_contracts": "注意发行金额与注册金额、评级、期限、利率、回售赎回、违约条款和主体角色。",
    "financial_reports": "注意年份、单位、同比、每股/每10股、归母/扣非、研发投入/研发费用等口径差异。",
    "research": "注意研报中的趋势、图表数值、单位、公司/行业对比和结论适用范围。",
}

QA_PROMPT_TEMPLATE = """下面是从相关文档中检索到的证据。证据已经用“题干”和“A/B/C/D各选项”分别检索后去重合并。

{evidence}

---

{format_hint}
{domain_hint}

题目：
{question}

选项：
{options}

请根据证据判断哪个选项正确。只输出答案字母，不要输出其他内容。"""

MULTI_QA_PROMPT_TEMPLATE = """下面是从相关文档中检索到的证据。证据已经用“题干”和“A/B/C/D各选项”分别检索后去重合并。

{evidence}

---

{format_hint}
{domain_hint}

题目：
{question}

选项：
{options}

请逐项判断 A/B/C/D 是否被证据支持，最后只输出所有正确选项字母，例如 AC 或 BCD。不要输出解释。"""

TF_PROMPT_TEMPLATE = """下面是从相关文档中检索到的证据。证据已经用“题干”和“A/B/C/D各选项”分别检索后去重合并。

{evidence}

---

{format_hint}
{domain_hint}

题目：
{question}

选项：
{options}

请根据证据判断题目陈述是否正确。只输出 A 或 B，不要输出其他内容。"""

MULTI_REFLECTION_PROMPT = """你之前对一道多选题只选择了 {previous_answer}。这可能遗漏了其他正确选项。

证据：
{evidence}

题目：
{question}

选项：
{options}

{domain_hint}

请重新逐项判断 A/B/C/D。最后一行只输出最终答案字母，按字母顺序排列。"""
