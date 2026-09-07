"""Claim Verifier: LLM-based option-level verification.

The verifier takes each option's ClaimMemory + raw evidence and produces
a structured judgment: supported / contradicted / mixed.

Key design:
- Each option is verified independently (no cross-option contamination)
- Output is structured JSON for programmatic consumption
- Reasoning is preserved for debug trace
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .schemas import OptionClaimState, QAState
from .config import AgentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verifier Prompts
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = """你是一位金融文档事实核验专家。你的任务是严格根据给定证据判断一个选项是否正确。

核心原则：
1. 只依据证据中的明确事实进行判断，不要依赖常识或推测
2. **语义等价不算矛盾**：选项与证据用词不同但含义一致时（如"市场价格"与"前N个交易日均价"、"年度内"与"截至年末"），应判 supported，不要因表述差异判 contradicted
3. **严格区分 contradicted 与 mixed**：
   - 只有当证据中有明确的、可量化的、与选项直接对立的事实时，才判 contradicted（如选项说"超过10%"，证据明确说"5%"）
   - 仅仅是"证据没提到"或"信息不完整"，必须判 mixed，不要判 contradicted
   - 部分维度匹配、部分维度未提及，应判 mixed
4. **数值验证必须实际计算**：
   - 选项含具体数值（百分比、金额、比例）时，必须从证据中提取相应数值进行计算
   - 现金分红比例 = 现金分红总额 / 当年净利润，必须找到这两个数值后实际计算
   - 同比增长率 = (本期 - 上期) / 上期 × 100%，不要凭感觉判断
   - 如果计算所需的关键数值在证据中找不到，判 mixed
5. **多实体题目**：当选项涉及多个公司/产品/年份时（如"两家公司均..."、"四款产品中..."），必须每个实体都在证据中找到支持，才能判 supported；只找到一部分应判 mixed
6. **数据口径**：注意年份、单位、每股/每10股、归母/扣非、合并/母公司、税前/税后等口径差异
7. **置信度校准**：
   - contradicted 的 confidence 上限 0.9，除非有非常明确的数值矛盾才用 0.95+
   - supported 的 confidence 与证据明确度成正比
   - mixed 的 confidence 反映你对"这个选项是对还是错"的把握度，越不确定越接近 0

判断标准：
- supported：证据明确支持该选项的全部内容（数据吻合、事实一致或语义等价）
- contradicted：证据中存在与选项直接、可量化矛盾的事实（数据不符、事实相反）
- mixed：证据不确定——包括证据不足无法判断、组合命题中部分支持部分矛盾、或证据模糊不清晰等情况。confidence 越低表示越不确定

只输出 JSON，不要输出多余内容。"""

VERIFIER_USER_TEMPLATE_MCQ = """题目（单选题）：{question}

当前验证选项 {option}：{option_text}

{domain_hint}

相关证据：
{evidence_text}

请严格根据证据判断选项 {option} 是否正确。

判断指引：
1. 本题是单选题，只有一个正确答案，请仔细核对证据中的具体数据
2. **语义等价**：选项的表述与证据有差异但含义一致时，应判 supported
3. **数值类选项**：必须从证据中提取关键数值并实际计算
4. **谨慎使用 contradicted**：只有明确数值矛盾才用，否则判 mixed

输出格式：
{{"status": "supported|contradicted|mixed", "confidence": 0.0-1.0, "reason": "一句话解释，引用关键数据。若涉及计算请写出计算过程"}}"""

VERIFIER_USER_TEMPLATE_MULTI = """题目（多选题）：{question}

全部选项：
{all_options_text}

当前验证选项 {option}：{option_text}

{domain_hint}

相关证据：
{evidence_text}

请严格根据证据判断选项 {option} 是否正确。

判断指引：
1. 多选题至少有 2 个正确答案，请独立判断当前选项，不要因为其他选项可能正确就排除当前选项
2. **谨慎使用 contradicted**：只有证据中有明确的、可量化的反驳事实，才判 contradicted；如果只是"没找到对应数据"或"表述不完全相同"，请判 mixed
3. **语义等价**：选项的表述与证据有差异但含义一致（如"市场价格"vs"近20个交易日均价"），应判 supported
4. **数值类选项**：必须从证据中提取关键数值，实际计算后再判断（如分红比例、增长率、占比等）
5. **多实体选项**：选项中提到多个公司/产品/年份时，必须每个都被证据支持才能判 supported
6. 优先判 mixed 而非 contradicted——给后续反思留出修正机会

输出格式：
{{"status": "supported|contradicted|mixed", "confidence": 0.0-1.0, "reason": "一句话解释，引用关键数据。若涉及计算请写出计算过程"}}"""

VERIFIER_USER_TEMPLATE_TF = """题目（判断题）：{question}

选项：
{all_options_text}

当前验证选项 {option}：{option_text}

{domain_hint}

相关证据：
{evidence_text}

请严格根据证据判断选项 {option} 的陈述是否正确。

输出格式：
{{"status": "supported|contradicted|mixed", "confidence": 0.0-1.0, "reason": "一句话解释，引用关键数据"}}"""


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """Verify each option independently using LLM.

    The verifier reads the compressed ClaimMemory and a subset of raw
    evidence, then outputs a structured judgment per option.
    """

    def __init__(self, config: AgentConfig, llm_client=None, model_name: str = None):
        self.config = config
        self.llm_client = llm_client
        self.model_name = model_name or config.model_name

    def verify_all(self, state: QAState) -> QAState:
        """Verify all options in the state."""
        for opt_key in state.claims:
            self._verify_option(state, opt_key)
        return state

    def reverify(self, state: QAState, target_options: List[str]) -> QAState:
        """Re-verify only specific options (after round 2 retrieval)."""
        for opt_key in target_options:
            if opt_key in state.claims:
                self._verify_option(state, opt_key)
        return state

    def _verify_option(self, state: QAState, opt_key: str) -> None:
        """Verify a single option and update its claim state."""
        claim_state = state.claims[opt_key]
        memory = state.working_memory.get(opt_key)

        if not memory or not memory.key_facts:
            # No evidence available — mark as mixed with zero confidence
            claim_state.status = "mixed"
            claim_state.confidence = 0.0
            claim_state.reason = "No evidence found for this option"
            return

        # Build evidence text for the verifier
        evidence_text = self._format_evidence_for_verifier(state, opt_key)

        # Call LLM
        if self.llm_client is None:
            # Fallback: mark as mixed if no LLM
            claim_state.status = "mixed"
            claim_state.confidence = 0.3
            return

        # Select prompt template based on answer_format
        domain_hint = self.config.domain_hints.get(state.domain, "")
        all_options_text = "\n".join(
            f"{k}. {v}" for k, v in sorted(state.options.items())
        )

        if state.answer_format == "multi":
            user_prompt = VERIFIER_USER_TEMPLATE_MULTI.format(
                question=state.question,
                option=opt_key,
                option_text=claim_state.original_text,
                all_options_text=all_options_text,
                domain_hint=domain_hint,
                evidence_text=evidence_text,
            )
        elif state.answer_format == "tf":
            user_prompt = VERIFIER_USER_TEMPLATE_TF.format(
                question=state.question,
                option=opt_key,
                option_text=claim_state.original_text,
                all_options_text=all_options_text,
                domain_hint=domain_hint,
                evidence_text=evidence_text,
            )
        else:  # mcq
            user_prompt = VERIFIER_USER_TEMPLATE_MCQ.format(
                question=state.question,
                option=opt_key,
                option_text=claim_state.original_text,
                domain_hint=domain_hint,
                evidence_text=evidence_text,
            )

        try:
            response = self._call_llm(VERIFIER_SYSTEM_PROMPT, user_prompt)
            result = self._parse_verifier_output(response["raw_answer"])

            raw_status = result.get("status", "mixed")
            # Normalize: map legacy 'insufficient' to 'mixed'
            if raw_status == "insufficient":
                raw_status = "mixed"
            claim_state.status = raw_status
            claim_state.confidence = float(result.get("confidence", 0.5))
            claim_state.reason = result.get("reason", "")

            # Track tokens
            state.add_tokens(
                prompt=response.get("prompt_tokens", 0),
                completion=response.get("completion_tokens", 0),
                total=response.get("total_tokens", 0),
            )

        except Exception as e:
            logger.warning("[%s] Verifier failed for option %s: %s", state.qid, opt_key, e)
            claim_state.status = "mixed"
            claim_state.confidence = 0.3
            claim_state.reason = f"Verifier error: {e}"

    def _format_evidence_for_verifier(self, state: QAState, opt_key: str) -> str:
        """Format evidence for the verifier prompt."""
        memory = state.working_memory.get(opt_key)
        if not memory:
            return "(无证据)"

        lines = []
        for i, fact in enumerate(memory.key_facts, 1):
            lines.append(f"[证据{i}] {fact}")

        return "\n\n".join(lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call the LLM and return raw response."""
        extra_body = {}
        if self.config.enable_thinking:
            extra_body["enable_thinking"] = True
            extra_body["thinking_budget"] = self.config.thinking_budget

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,  # Verifier output is short
            temperature=self.config.temperature,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        msg = response.choices[0].message
        return {
            "raw_answer": msg.content or "",
            "reasoning_text": getattr(msg, "reasoning_content", None) or "",
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

    def _parse_verifier_output(self, raw: str) -> Dict[str, Any]:
        """Parse the JSON output from the verifier LLM."""
        # Try to extract JSON from the response
        raw = raw.strip()

        # Try direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract status from text
        status = "mixed"
        for s in ["supported", "contradicted", "mixed"]:
            if s in raw.lower():
                status = s
                break
        # Normalize legacy 'insufficient' to 'mixed'
        if "insufficient" in raw.lower() and status == "mixed":
            status = "mixed"

        return {"status": status, "confidence": 0.5, "reason": raw[:100]}
