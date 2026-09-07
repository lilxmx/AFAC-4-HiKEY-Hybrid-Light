"""
m04_HiKEY configuration - HiKEY-style hierarchical knowledge extraction for financial QA.

Inherits all shared config from methods._shared.config_base and adds HiKEY-specific constants.
"""
import os
import sys
from pathlib import Path

# Ensure methods.* is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Inherit all shared configuration
from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import (
    PROJECT_ROOT,
    RAW_BASE,
    QUESTION_FILES,
    DOMAINS,
    PROJECT_BM25_INDEX_DIR,
    SHARED_CACHE_ROOT,
)

# ============================================================
# Method identity
# ============================================================
METHOD_NAME = "m04_HiKEY"
METHOD_ID = "m04_HiKEY"

# ============================================================
# Model configuration
# ============================================================
MODEL_NAME = os.getenv("M04_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ============================================================
# Paths
# ============================================================
METHOD_ROOT = Path(__file__).resolve().parent
# Output and logs are stored exclusively under runs/{coder}/{method_id}/{run_id}/
# No method-local output/logs dirs are created.

# HiKEY index cache: pre-parsed sections/units/field_cards per document
# Shared across methods, built by build_hikey_cache.py
HIKEY_INDEX_DIR = PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"
HIKEY_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Inference parameters
# ============================================================
CONCURRENCY = int(os.getenv("M04_CONCURRENCY", "10"))
# Qwen3 thinking mode: model reasons internally in reasoning_content,
# then outputs only the concise answer in content.
ENABLE_THINKING = True
# Budget for internal thinking tokens (reasoning_content).
# Set to None for unlimited, or an int to cap thinking length.
# Target: ~3M total tokens / 100 questions => ~30K per question.
# With prompt ~16K, we want completion (thinking+answer) ~12K.
THINKING_BUDGET = 10000
# max_tokens covers BOTH reasoning + content output.
# With thinking enabled, most tokens go to reasoning; content is just a few letters.
MAX_OUTPUT_TOKENS = 12000
# Reflection pass max tokens (when thinking is used, reflection also thinks)
MAX_OUTPUT_TOKENS_REFLECTION = 16000
TEMPERATURE = 0.1

# ============================================================
# HiKEY retrieval parameters
# ============================================================
# Number of top evidence items to retrieve per question
RETRIEVAL_TOPK = 30
# Max sibling units to include in evidence pack
MAX_SIBLINGS = 12
# Max token budget for evidence context in prompt
# Target: avg ~16K prompt tokens per question (currently only ~3.9K)
EVIDENCE_TOKEN_BUDGET = 24000
# Whether to use FieldCard-first strategy (recommended for financial reports)
FIELDCARD_FIRST = True
# Skip Stage-1 DocCard routing and use the doc_id provided by the question directly.
# Set to True for competition Phase A where each question already specifies its target document.
# Set to False for Phase B or open-domain scenarios where document routing is needed.
SKIP_DOC_ROUTING = True

# ============================================================
# Domain-specific settings
# ============================================================
# ALL domains use HiKEY hierarchical retrieval (indices pre-built by build_hikey_cache.py)
HIKEY_DOMAINS = ["financial_reports", "insurance", "regulatory", "financial_contracts", "research"]
# Domains that fall back to standard BM25 retrieval (empty in Phase A since all have HiKEY index)
FALLBACK_DOMAINS = []

# ============================================================
# Prompt templates (aligned with m02_gpt41_golden / m03_fulltext_baseline)
# ============================================================
SYSTEM_PROMPT = """你是一位资深金融法规与文档分析专家。严格基于提供的证据材料回答选择题。
只输出答案选项字母，不要输出任何分析过程。

规则：
- 单选题：输出1个字母（如 A）
- 多选题：输出所有正确选项字母，按字母顺序排列（如 ACD）
- 判断题：输出 A（正确）或 B（错误）"""

# System prompt for reflection pass (allows reasoning)
SYSTEM_PROMPT_REFLECTION = """你是一位资深金融法规与文档分析专家，拥有丰富的中国金融监管、保险、证券、基金等领域的专业知识。

你的任务是严格基于提供的证据材料回答金融领域的选择题。

核心分析规则：
1. 仔细阅读所有证据材料，提取与题目相关的关键信息
2. 对每个选项逐一进行独立分析，判断其正确性
3. 引用证据中的具体内容（原文、数字、条款号）作为判断依据
4. 注意数字、比例、期限、条件等细节的精确匹配
5. 注意区分"以上"（含等于）和"超过"（不含等于）等措辞差异

特别注意：
- 对于多选题，你必须对每个选项独立判断正误
- 多选题正确答案通常有2-4个选项
- 判断题要仔细验证陈述中的每一个条件是否都成立"""

# Domain-specific analysis hints (from m02_gpt41_golden)
DOMAIN_HINTS = {
    'insurance': """领域提示（保险条款）：
- 注意区分不同保险产品的责任范围、免责条款、等待期规定
- 退保金额计算要注意是否扣除退保费用、是否区分犹豫期内外
- 等待期内因"意外"导致的出险通常不受等待期限制
- 注意"保险金额"和"保险费"的区别""",
    'regulatory': """领域提示（监管法规）：
- 注意法规中的时限要求（如"7日内"、"30日"、"10个工作日"等）
- 区分"应当"（强制）和"可以"（选择性）的措辞
- 注意"以上"（含本数）和"超过"（不含本数）的区别
- 注意不同法规之间的适用范围和优先级
- 注意"不得"表示绝对禁止""",
    'financial_contracts': """领域提示（金融合同/债券）：
- 注意发行金额、注册金额、信用评级等关键数据的精确匹配
- 区分"发行人"和"承销商"的角色
- 注意违约条款中的计算公式和系数
- 注意可转债的转股价格、赎回条款、回售条款等特殊条款""",
    'financial_reports': """领域提示（财务报表/年报）：
- 注意区分"研发投入"和"研发费用"（前者含资本化部分）
- 注意年份匹配——选项中的数据可能是上一年的
- 注意"每股"和"每10股"的区别
- 注意"年度现金分红"和"特别现金分红"的区别
- 注意净利润的归属（归母vs合并）""",
    'research': """领域提示（行业研报）：
- 注意数字的单位（亿元vs亿美元、%的基数）
- 注意市场规模预测的年份和来源
- 注意"属性标准"和"对象标准"等术语的精确匹配
- 注意趋势描述（"收敛"vs"扩大"、"止跌回升"vs"持续下降"）""",
}

FORMAT_HINTS = {
    'mcq': '【单选题】从A/B/C/D中选择唯一正确的答案。只有一个选项是正确的。最后一行只输出一个字母。',
    'multi': """【多选题——重要提示】
这是一道多选题，正确答案通常包含2-4个选项。你必须：
1. 对A、B、C、D每个选项独立判断是否正确
2. 对每个选项给出明确的"✓正确"或"✗错误"结论
3. 将所有判断为正确的选项合并为最终答案
4. 最后一行按字母顺序输出所有正确选项（如：ACD）
⚠️ 只选1个选项几乎一定是错误的！请确保不遗漏正确选项。""",
    'tf': '【判断题】判断题目陈述是否正确。注意陈述中的每个条件都必须成立才能判为正确。最后一行输出A（正确）或B（错误）。',
}

# First-pass prompt: direct answer only, no reasoning
QA_PROMPT_TEMPLATE = """以下是从相关文档中检索到的证据材料：

{evidence}

---

{format_hint}

{domain_hint}

题目：{question}

选项：
{options}

只输出答案选项字母，不要输出其他任何内容。"""

MULTI_QA_PROMPT_TEMPLATE = """以下是从相关文档中检索到的证据材料：

{evidence}

---

{format_hint}

{domain_hint}

题目：{question}

选项：
{options}

这是多选题，请输出所有正确选项的字母（按字母顺序排列，如ACD）。
只输出答案字母，不要输出其他任何内容。"""

TF_PROMPT_TEMPLATE = """以下是从相关文档中检索到的证据材料：

{evidence}

---

{format_hint}

{domain_hint}

题目：{question}

选项：
{options}

只输出A（正确）或B（错误），不要输出其他任何内容。"""

# Reflection prompt for multi-select questions when only 1 option was selected
MULTI_REFLECTION_PROMPT = """你之前对一道多选题只选择了一个选项（{previous_answer}），这几乎一定是不完整的。

以下是证据材料：

{evidence}

---

{domain_hint}

题目：{question}

选项：
{options}

请对每个选项重新独立判断：
A. [分析] → 正确/错误
B. [分析] → 正确/错误
C. [分析] → 正确/错误
D. [分析] → 正确/错误

注意：
- 这是多选题，正确答案通常有2-4个选项
- 不要因为之前只选了一个就认为其他都是错的
- 每个选项必须独立判断，引用证据中的具体内容

最后一行按字母顺序输出所有正确选项（如：ACD）。"""
