"""Configuration for m05c_HiKEY_domain.

m05c keeps the HiKEY shared document hierarchy, then adds lightweight
domain-specific cards and an option-level verifier prompt.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import PROJECT_ROOT

METHOD_NAME = "m05c_HiKEY_domain"
METHOD_ID = METHOD_NAME
METHOD_ROOT = Path(__file__).resolve().parent

CODER = os.getenv("CODER", "qhl")
RUN_DESC = os.getenv("M05C_RUN_DESC", "domain")

MODEL_NAME = os.getenv("M05C_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

HIKEY_INDEX_DIR = PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"
GROUND_TRUTH_PATH = PROJECT_ROOT / "GPT-Pro-Answer" / "gpt_pro_all_answers.json"

CONCURRENCY = int(os.getenv("M05C_CONCURRENCY", "6"))
TEMPERATURE = float(os.getenv("M05C_TEMPERATURE", "0.0"))

ENABLE_THINKING = os.getenv("M05C_ENABLE_THINKING", "0") == "1"
THINKING_BUDGET = int(os.getenv("M05C_THINKING_BUDGET", "10000"))
MAX_OUTPUT_TOKENS = int(os.getenv("M05C_MAX_OUTPUT_TOKENS", "12000"))

# Retrieval is intentionally wider locally and narrower in the final prompt.
QUESTION_TOPK = int(os.getenv("M05C_QUESTION_TOPK", "10"))
OPTION_TOPK = int(os.getenv("M05C_OPTION_TOPK", "10"))
DOMAIN_CARD_TOPK = int(os.getenv("M05C_DOMAIN_CARD_TOPK", "10"))
PER_OPTION_EVIDENCE = int(os.getenv("M05C_PER_OPTION_EVIDENCE", "10"))
FINAL_EVIDENCE_TOPK = int(os.getenv("M05C_FINAL_EVIDENCE_TOPK", "28"))
MAX_SIBLINGS = int(os.getenv("M05C_MAX_SIBLINGS", "12"))
EVIDENCE_TOKEN_BUDGET = int(os.getenv("M05C_EVIDENCE_TOKEN_BUDGET", "24000"))
SKIP_DOC_ROUTING = os.getenv("M05C_SKIP_DOC_ROUTING", "1") != "0"

# Domain-card policy:
# - regulatory has proven useful with strong DomainCard reservation.
# - other domains keep DomainCard as weak rerank candidates so weak cards do
#   not displace stronger HiKEY evidence.
STRONG_DOMAIN_CARD_DOMAINS = {
    d.strip()
    for d in os.getenv("M05C_STRONG_DOMAIN_CARD_DOMAINS", "regulatory").split(",")
    if d.strip()
}
WEAK_DOMAIN_CARD_TOPK = int(os.getenv("M05C_WEAK_DOMAIN_CARD_TOPK", "4"))
WEAK_PER_OPTION_EVIDENCE = int(os.getenv("M05C_WEAK_PER_OPTION_EVIDENCE", "1"))
WEAK_DOMAIN_CARD_MIN_SCORE = float(os.getenv("M05C_WEAK_DOMAIN_CARD_MIN_SCORE", "4.0"))
REGULATORY_DOMAIN_SCORE_BOOST = float(os.getenv("M05C_REGULATORY_DOMAIN_SCORE_BOOST", "12.0"))
WEAK_DOMAIN_SCORE_BOOST = float(os.getenv("M05C_WEAK_DOMAIN_SCORE_BOOST", "-1.5"))

SYSTEM_PROMPT = """你是严谨的金融长文档选择题裁决器。你必须只依据给定证据逐项判断。
输出 JSON，字段必须包含 option_verdicts 和 answer。
option_verdicts 中每个选项的 verdict 只能是 supported、contradicted 或 insufficient。
answer 只输出最终答案字母：单选题一个字母，多选题按字母顺序输出多个字母，判断题输出 A 或 B。"""

VERIFIER_PROMPT_TEMPLATE = """下面是从文档中召回并压缩后的证据。证据按选项组织，可能包含 HiKEY 通用证据和领域专用 DomainCard。

题型：{answer_format}
领域：{domain}
领域判断提示：{domain_hint}

题目：
{question}

选项：
{options}

证据：
{evidence}

请逐项判断每个选项是否被证据支持。要求：
1. supported 表示选项的关键事实都能被证据直接支持。
2. contradicted 表示证据直接反驳选项关键事实。
3. insufficient 表示证据不足以支持该选项，不要因为相似就判正确。
4. 多选题选所有 supported 选项；单选题选最被支持的一个；判断题按选项含义输出 A 或 B。

只输出 JSON，不要输出解释性散文。示例：
{{"option_verdicts":{{"A":{{"verdict":"supported","evidence_ids":["A-1"],"reason":"..."}}}},"answer":"A"}}"""

DOMAIN_HINTS = {
    "financial_reports": "重点核对公司、年份、指标名、单位、同比方向、分红口径，特别区分归母/扣非、每股/每10股。",
    "financial_contracts": "重点核对发行人、发行规模、债项/主体评级、期限、利率、主承销商、受托管理人、担保和信息披露条款。",
    "insurance": "重点核对保险责任、免责、等待期、免赔额、赔付比例、账户价值、现金价值和退保费用计算。",
    "regulatory": "重点核对主体、义务强度、条件、期限、施行日期和例外条款，严格区分应当/可以/不得。",
    "research": "重点核对年份、市场规模、增速、CAGR、预测口径、公司/行业对比和图表数值。",
}

CONFIG_SNAPSHOT = {
    "question_topk": QUESTION_TOPK,
    "option_topk": OPTION_TOPK,
    "domain_card_topk": DOMAIN_CARD_TOPK,
    "per_option_evidence": PER_OPTION_EVIDENCE,
    "final_evidence_topk": FINAL_EVIDENCE_TOPK,
    "evidence_token_budget": EVIDENCE_TOKEN_BUDGET,
    "max_siblings": MAX_SIBLINGS,
    "strong_domain_card_domains": sorted(STRONG_DOMAIN_CARD_DOMAINS),
    "weak_domain_card_topk": WEAK_DOMAIN_CARD_TOPK,
    "weak_per_option_evidence": WEAK_PER_OPTION_EVIDENCE,
    "weak_domain_card_min_score": WEAK_DOMAIN_CARD_MIN_SCORE,
    "enable_thinking": ENABLE_THINKING,
    "thinking_budget": THINKING_BUDGET,
}
