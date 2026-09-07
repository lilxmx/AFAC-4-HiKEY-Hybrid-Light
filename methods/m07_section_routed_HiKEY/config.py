"""
m07_section_routed_HiKEY configuration.

This method keeps m05's question+option evidence retrieval, but first asks
Qwen to rerank candidate HiKEY sections. The final retrieval is restricted to
the routed section subtrees, with global fallback when local evidence looks
insufficient.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import PROJECT_ROOT

METHOD_NAME = "m07_section_routed_HiKEY"
METHOD_ID = METHOD_NAME
METHOD_ROOT = Path(__file__).resolve().parent

MODEL_NAME = os.getenv("M07_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

HIKEY_INDEX_DIR = PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"
REFERENCE_ANSWERS_PATH = Path(
    os.getenv("M07_REFERENCE_ANSWERS", str(PROJECT_ROOT / "GPT-Pro-Answer" / "gpt_pro_all_answers.json"))
)

CODER = os.getenv("CODER", "qhl")
RUN_DESC = os.getenv("M07_RUN_DESC", "section-routed")

CONCURRENCY = int(os.getenv("M07_CONCURRENCY", "8"))
TEMPERATURE = float(os.getenv("M07_TEMPERATURE", "0.1"))

ENABLE_THINKING = os.getenv("M07_ENABLE_THINKING", "1") != "0"
THINKING_BUDGET = int(os.getenv("M07_THINKING_BUDGET", "10000"))
MAX_OUTPUT_TOKENS = int(os.getenv("M07_MAX_OUTPUT_TOKENS", "12000"))
MAX_OUTPUT_TOKENS_REFLECTION = int(os.getenv("M07_MAX_OUTPUT_TOKENS_REFLECTION", "16000"))
MAX_OUTPUT_TOKENS_RERANK = int(os.getenv("M07_MAX_OUTPUT_TOKENS_RERANK", "1800"))

# Section routing.
ENABLE_SECTION_ROUTING = os.getenv("M07_ENABLE_SECTION_ROUTING", "1") != "0"
ENABLE_LLM_SECTION_RERANK = os.getenv("M07_ENABLE_LLM_SECTION_RERANK", "1") != "0"
SECTION_CANDIDATE_TOPK = int(os.getenv("M07_SECTION_CANDIDATE_TOPK", "40"))
ROUTED_SECTION_TOPM = int(os.getenv("M07_ROUTED_SECTION_TOPM", "5"))
MULTI_ROUTED_SECTION_TOPM = int(os.getenv("M07_MULTI_ROUTED_SECTION_TOPM", "10"))
SECTION_PREVIEW_CHARS = int(os.getenv("M07_SECTION_PREVIEW_CHARS", "180"))

# Evidence retrieval inside routed sections.
GLOBAL_TOPK = int(os.getenv("M07_GLOBAL_TOPK", "8"))
OPTION_TOPK = int(os.getenv("M07_OPTION_TOPK", "8"))
FINAL_EVIDENCE_TOPK = int(os.getenv("M07_FINAL_EVIDENCE_TOPK", "24"))
MAX_SIBLINGS = int(os.getenv("M07_MAX_SIBLINGS", "8"))
EVIDENCE_TOKEN_BUDGET = int(os.getenv("M07_EVIDENCE_TOKEN_BUDGET", "22000"))
SKIP_DOC_ROUTING = os.getenv("M07_SKIP_DOC_ROUTING", "1") != "0"

# Fallback and diagnostics.
ENABLE_GLOBAL_FALLBACK = os.getenv("M07_ENABLE_GLOBAL_FALLBACK", "1") != "0"
MIN_ROUTED_EVIDENCE = int(os.getenv("M07_MIN_ROUTED_EVIDENCE", "6"))
MIN_OPTION_PROXY_COVERAGE = float(os.getenv("M07_MIN_OPTION_PROXY_COVERAGE", "0.5"))

SYSTEM_PROMPT = """你是一位金融文档分析专家。你必须严格根据给定证据回答选择题。

只输出答案字母，不要输出解释。
单选题输出一个字母，例如 A。
多选题输出所有正确选项字母，按字母顺序排列，例如 ACD。
判断题输出 A 或 B，以题目选项含义为准。"""

SYSTEM_PROMPT_REFLECTION = """你是一位金融文档分析专家。请严格根据证据逐项核验选项。
最后一行只输出最终答案字母。"""

SECTION_RERANK_SYSTEM_PROMPT = """你是金融长文档检索路由专家。你的任务是根据题目和候选章节路径，判断哪些章节最可能包含作答证据。
只输出候选章节编号，按相关性从高到低排列，不要解释。"""

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

QA_PROMPT_TEMPLATE = """下面是从相关文档中检索到的证据。证据已经先经过章节路由，再用“题干”和“A/B/C/D各选项”分别检索后去重合并。
{evidence}

---

{format_hint}
{domain_hint}

题目：{question}

选项：
{options}

请根据证据判断哪一个选项正确。只输出答案字母，不要输出其他内容。"""

MULTI_QA_PROMPT_TEMPLATE = """下面是从相关文档中检索到的证据。证据已经先经过章节路由，再用“题干”和“A/B/C/D各选项”分别检索后去重合并。
{evidence}

---

{format_hint}
{domain_hint}

题目：{question}

选项：
{options}

请逐项判断 A/B/C/D 是否被证据支持，最后只输出所有正确选项字母，例如 AC 或 BCD。不要输出解释。"""

TF_PROMPT_TEMPLATE = """下面是从相关文档中检索到的证据。证据已经先经过章节路由，再用“题干”和“A/B/C/D各选项”分别检索后去重合并。
{evidence}

---

{format_hint}
{domain_hint}

题目：{question}

选项：
{options}

请根据证据判断题目陈述是否正确。只输出 A 或 B，不要输出其他内容。"""

MULTI_REFLECTION_PROMPT = """你之前对一道多选题只选择了 {previous_answer}。这可能遗漏了其他正确选项。
证据：
{evidence}

题目：{question}

选项：
{options}

{domain_hint}

请重新逐项判断 A/B/C/D。最后一行只输出最终答案字母，按字母顺序排列。"""

SECTION_RERANK_PROMPT = """请根据题目和候选章节，选出最可能包含作答证据的章节。

题目：{question}

选项：
{options}

题型：{answer_format}
领域：{domain}

候选章节：
{sections}

要求：
1. 只从候选编号中选择。
2. 优先选择能验证具体选项事实的章节，而不是泛泛相关章节。
3. 对法规题，优先选择包含义务、禁止、例外、期限、处罚、适用范围的条款章节。
4. 对财报题，优先选择包含指标、表格、单位、年份的章节。
5. 输出最多 {topm} 个编号，按相关性从高到低排列。

输出格式示例：
<<<3>>>, <<<1>>>, <<<7>>>"""
