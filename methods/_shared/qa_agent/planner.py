"""Question Planner + Claim Extractor + Evidence Group Builder.

This module handles the "planning" phase of the agent workflow:
1. QuestionPlanner: parse question metadata, identify doc scope, question type
2. ClaimExtractor: decompose each option into atomic claims (LLM-based)
3. EvidenceGroupBuilder: merge related claims into shared retrieval groups

The primary strategy uses LLM for claim analysis (risk detection, operation
type, entity extraction). Rule-based detection is kept as fallback only.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    AtomicClaim,
    EvidenceGroup,
    OptionClaimState,
    QAState,
)
from .config import AgentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based fallback (only used when LLM is unavailable or fails)
# ---------------------------------------------------------------------------

# Strong logic keywords that indicate claims needing special handling
UNIVERSAL_KEYWORDS = ["均", "全部", "所有", "都"]
NEGATION_KEYWORDS = ["不", "未", "无", "没有", "不实施", "不送", "不派", "拟不", "暂不"]
THRESHOLD_KEYWORDS = ["超过", "低于", "不超过", "不少于", "高于", "至少", "不低于"]
COMPARISON_KEYWORDS = ["同比增长", "同比下降", "环比", "增长率", "下降"]
CALCULATION_KEYWORDS = ["合计", "总额", "占比", "比例", "之和"]

def detect_risk_flags_rule_based(text: str) -> List[str]:
    """Fallback: detect risk keywords using hardcoded rules.

    NOTE: This is only used when LLM-based extraction fails.
    Known limitations:
    - "每10股" triggers universal:每 incorrectly
    - Cannot detect semantic negation (e.g. "拟不进行")
    - Cannot distinguish "均" in different contexts
    """
    flags = []
    for kw in UNIVERSAL_KEYWORDS:
        if kw in text:
            flags.append(f"universal:{kw}")
            break
    for kw in NEGATION_KEYWORDS:
        if kw in text:
            flags.append(f"negation:{kw}")
            break
    for kw in THRESHOLD_KEYWORDS:
        if kw in text:
            flags.append(f"threshold:{kw}")
            break
    for kw in COMPARISON_KEYWORDS:
        if kw in text:
            flags.append(f"comparison:{kw}")
            break
    for kw in CALCULATION_KEYWORDS:
        if kw in text:
            flags.append(f"calculation:{kw}")
            break
    return flags

def detect_operation_rule_based(text: str, flags: List[str]) -> str:
    """Fallback: infer operation type from risk flags."""
    flag_types = [f.split(":")[0] for f in flags]
    if "calculation" in flag_types:
        return "calculate"
    if "threshold" in flag_types:
        return "threshold"
    if "comparison" in flag_types:
        return "compare"
    if "universal" in flag_types:
        return "universal"
    if "negation" in flag_types:
        return "negate"
    return "lookup"


# ---------------------------------------------------------------------------
# QuestionPlanner
# ---------------------------------------------------------------------------

class QuestionPlanner:
    """Parse question metadata and set up the QAState for retrieval.

    Uses the question's own `type` field (from the dataset) as the primary
    question_type. Only falls back to heuristic inference if the field is
    missing.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def plan(self, state: QAState) -> QAState:
        """Fill in planning fields on the QAState."""
        # Set doc_scope from question's doc_ids
        state.doc_scope = list(state.doc_ids) if state.doc_ids else []

        # Build global query (question stem, used for broad retrieval)
        state.global_query = state.question

        # Use the dataset-provided question type directly if available
        # (stored in state during from_question), otherwise infer
        if not state.question_type:
            state.question_type = self._infer_question_type(state)

        return state

    def _infer_question_type(self, state: QAState) -> str:
        """Fallback heuristic when question type is not provided."""
        q = state.question
        if "以下" in q and ("正确" in q or "错误" in q):
            return "option_verification"
        if "哪" in q or "属于" in q:
            return "selection"
        if any(kw in q for kw in ["计算", "多少", "金额", "比例"]):
            return "calculation"
        if any(kw in q for kw in ["比较", "对比", "区别"]):
            return "comparison"
        return "factual"


# ---------------------------------------------------------------------------
# LLM-based Claim Analysis Prompt
# ---------------------------------------------------------------------------

CLAIM_ANALYSIS_SYSTEM_PROMPT = """你是金融文档分析专家。你的任务是分析题目中每个选项的结构特征，为后续的证据检索和核验提供指导。

你需要对每个选项输出以下信息：
1. risk_flags: 该选项包含的特殊逻辑特征（列表）
2. operation: 核验该选项需要的操作类型
3. company: 选项涉及的公司名称（如有）
4. year: 选项涉及的年份（如有）
5. key_metrics: 选项涉及的关键指标/字段名（如有）

risk_flags 可选值：
- "universal": 全称命题（均、全部、所有、都、每一个、任何）
- "negation": 否定命题（不、未、无、没有、拟不、暂不、不实施）
- "threshold": 阈值比较（超过、低于、不超过、不少于、高于、至少）
- "comparison": 同比/环比比较（同比增长、同比下降、环比）
- "calculation": 需要计算（合计、总额、占比、比例、之和、总金额）
- "temporal": 时间限定（截至、期间、年末、年初）
- "conditional": 条件限定（如果、若、在...情况下）

operation 可选值：
- "lookup": 简单事实查找
- "compare": 数值或属性比较
- "calculate": 需要数学计算
- "negate": 验证否定命题
- "universal": 验证全称命题（需要验证所有实体都满足）
- "threshold": 验证是否超过/低于某个阈值

只输出 JSON，不要输出多余内容。"""

CLAIM_ANALYSIS_USER_TEMPLATE = """题目：{question}

选项：
{options_text}

涉及文档：{doc_ids}

请分析每个选项，输出 JSON 格式：
{{
  "A": {{
    "risk_flags": ["..."],
    "operation": "...",
    "company": "公司名或null",
    "year": 年份数字或null,
    "key_metrics": ["指标1", "指标2"]
  }},
  "B": {{ ... }},
  ...
}}"""

# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """Decompose each option into one or more atomic claims.

    Primary strategy: LLM-based analysis
    - Calls LLM to analyze all options at once
    - Extracts risk_flags, operation, company, year, key_metrics
    - More accurate than keyword matching (handles context, negation scope, etc.)

    Fallback strategy: rule-based keyword detection
    - Used when LLM is unavailable or call fails
    - Known to have false positives (e.g. "每10股" → universal)
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client

    def extract(self, state: QAState) -> QAState:
        """Populate state.claims with OptionClaimState for each option."""
        # Try LLM-based extraction first
        llm_analysis = None
        if self.llm_client and self.config.use_llm_planner:
            llm_analysis = self._llm_analyze_claims(state)

        # Build claim states for each option
        for opt_key, opt_text in state.options.items():
            if llm_analysis and opt_key in llm_analysis:
                # Use LLM analysis result
                analysis = llm_analysis[opt_key]
                risk_flags = analysis.get("risk_flags", [])
                operation = analysis.get("operation", "lookup")
                company = analysis.get("company")
                year = analysis.get("year")
                key_metrics = analysis.get("key_metrics", [])
            else:
                # Fallback to rule-based
                risk_flags = detect_risk_flags_rule_based(opt_text)
                operation = detect_operation_rule_based(opt_text, risk_flags)
                company = self._extract_company_rule_based(opt_text, state)
                year = self._extract_year_rule_based(opt_text)
                key_metrics = []

            claim = AtomicClaim(
                claim_id=f"{opt_key}1",
                text=opt_text,
                option=opt_key,
                operation=operation,
                company=company,
                year=year,
                metrics=key_metrics,
            )

            claim_state = OptionClaimState(
                option=opt_key,
                original_text=opt_text,
                atomic_claims=[claim],
                risk_flags=risk_flags,
            )
            state.claims[opt_key] = claim_state

        return state

    # ------------------------------------------------------------------
    # LLM-based analysis
    # ------------------------------------------------------------------

    def _llm_analyze_claims(self, state: QAState) -> Optional[Dict[str, Any]]:
        """Call LLM to analyze all options at once.

        Returns a dict like:
        {
            "A": {"risk_flags": [...], "operation": "...", ...},
            "B": {...},
            ...
        }
        Returns None if LLM call fails.
        """
        options_text = "\n".join(
            f"{k}. {v}" for k, v in state.options.items()
        )
        doc_ids_text = ", ".join(state.doc_ids) if state.doc_ids else "未指定"

        user_prompt = CLAIM_ANALYSIS_USER_TEMPLATE.format(
            question=state.question,
            options_text=options_text,
            doc_ids=doc_ids_text,
        )

        try:
            extra_body = {}
            if self.config.enable_thinking:
                extra_body["enable_thinking"] = True
                extra_body["thinking_budget"] = min(self.config.thinking_budget, 4000)

            response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": CLAIM_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=0.1,
                **({
                    "extra_body": extra_body
                } if extra_body else {}),
            )

            msg = response.choices[0].message
            raw_content = msg.content or ""

            # Track tokens
            if response.usage:
                state.add_tokens(
                    prompt=response.usage.prompt_tokens,
                    completion=response.usage.completion_tokens,
                    total=response.usage.total_tokens,
                )

            # Parse JSON from response
            result = self._parse_claim_analysis(raw_content)
            if result:
                logger.info("[%s] LLM claim analysis succeeded for %d options",
                           state.qid, len(result))
            return result

        except Exception as e:
            logger.warning("[%s] LLM claim analysis failed, using rule-based fallback: %s",
                          state.qid, e)
            return None

    def _parse_claim_analysis(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON output from LLM claim analysis."""
        raw = raw.strip()

        # Try direct JSON parse
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to find ```json ... ``` block
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if code_match:
            try:
                result = json.loads(code_match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Rule-based fallback helpers
    # ------------------------------------------------------------------

    def _extract_company_rule_based(self, text: str, state: QAState) -> Optional[str]:
        """Fallback: try to extract company name from option text."""
        # Common Chinese company name patterns
        company_pattern = re.search(
            r'([一-龥]{2,}(?:集团|公司|银行|保险|证券|基金|时代|电器))', text
        )
        if company_pattern:
            return company_pattern.group(1)
        return None

    def _extract_year_rule_based(self, text: str) -> Optional[int]:
        """Fallback: try to extract year from option text."""
        match = re.search(r'20[12]\d', text)
        if match:
            return int(match.group())
        return None


# ---------------------------------------------------------------------------
# EvidenceGroupBuilder
# ---------------------------------------------------------------------------

class EvidenceGroupBuilder:
    """Build evidence groups from atomic claims.

    Groups claims that share (doc_scope, company, year, topic) into
    shared retrieval tasks. This reduces redundant queries while
    maintaining coverage.

    First version strategy:
    1. Always create a "global" group with the question stem
    2. Create one "option" group per option (m05-compatible fallback)
    3. If grouped_retrieval is enabled, also try to merge related options
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def build(self, state: QAState) -> QAState:
        """Populate state.evidence_groups."""
        groups: List[EvidenceGroup] = []
        group_counter = 0

        # Group 0: Global query (question stem)
        group_counter += 1
        groups.append(EvidenceGroup(
            group_id=f"G{group_counter:02d}",
            group_type="global",
            query=state.global_query or state.question,
            doc_scope=state.doc_scope,
            covers_options=list(state.options.keys()),
            covers_claim_ids=[
                c.claim_id
                for cs in state.claims.values()
                for c in cs.atomic_claims
            ],
            topk=self.config.global_topk,
            mode="global",
        ))

        # Option-level groups (m05-compatible, always enabled)
        if self.config.enable_option_fallback:
            for opt_key, opt_text in state.options.items():
                group_counter += 1
                option_query = f"{state.question}\n{opt_key}. {opt_text}"
                claim_ids = [
                    c.claim_id for c in state.claims[opt_key].atomic_claims
                ]
                groups.append(EvidenceGroup(
                    group_id=f"G{group_counter:02d}",
                    group_type="option_fallback",
                    query=option_query,
                    doc_scope=state.doc_scope,
                    covers_options=[opt_key],
                    covers_claim_ids=claim_ids,
                    topk=self.config.option_topk,
                    mode="option",
                ))

        state.evidence_groups = groups
        return state


# ---------------------------------------------------------------------------
# Convenience: combined planning step
# ---------------------------------------------------------------------------

def run_planning(state: QAState, config: AgentConfig, llm_client=None) -> QAState:
    """Execute the full planning phase: plan → extract claims → build groups.

    Flow:
    1. QuestionPlanner: fill metadata (doc_scope, global_query, question_type)
    2. ClaimExtractor: analyze options via LLM (fallback: rule-based)
    3. EvidenceGroupBuilder: create retrieval task list
    """
    planner = QuestionPlanner(config)
    extractor = ClaimExtractor(config, llm_client=llm_client)
    group_builder = EvidenceGroupBuilder(config)

    state = planner.plan(state)
    state = extractor.extract(state)
    state = group_builder.build(state)

    return state
