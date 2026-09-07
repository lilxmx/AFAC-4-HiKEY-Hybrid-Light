"""
Financial Contracts Answer Agent.
Implements structured field verification with three-value judgment (supported/contradicted/insufficient).
Follows Skill v6.0: atomic claim evaluation + field verification table + domain-specific rules.

Migrated from Golden_label_baseline. Uses Qwen API instead of GPT-4.1.
NO voting mechanism - single inference. Reflection retained for multi-select under-selection.
"""
import re
import time
from typing import Dict, List, Optional
from openai import OpenAI

from .config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME, MAX_OUTPUT_TOKENS
)

# ============================================================
# Answer Agent System Prompt (Skill v6.0 compliant)
# ============================================================
FC_ANSWER_SYSTEM_PROMPT = """你是金融合同领域的 Answer Agent，专门处理债券募集说明书、可转债募集说明书、重大资产重组报告书等文档的选择题和判断题。

## 核心工作流

1. **固定 doc_ids 顺序**：第一份文档 = doc_ids[0]，第二份文档 = doc_ids[1]。严禁自行调换。

2. **构建字段核验表**：在分析前，先列出每个文档中与题目相关的关键字段值。

3. **三值判断**：对每个原子命题（atomic claim），只能标记为：
   - supported：有直接证据支持
   - contradicted：有直接证据反驳
   - insufficient：证据不足，无法判断
   
   **不要把 insufficient 当成正确。**

4. **标准化规则**：
   - 金额统一比较（亿元 vs 万元）
   - 区分"本期发行金额"和"注册金额/本次发行总额"
   - 区分"主体评级"和"债项评级"
   - "-" 表示无评级/未评级，不是负值
   - 区分"合并口径"和"母公司口径"
   - 百分比范围判断必须严格（如 66.38% 不在 63%-66% 之间）

5. **关键避坑规则**：
   - "均/都/全部" = AND 逻辑，任一文档不满足则为 false
   - "明确披露/明确标注" 要求直接证据，不能靠推断
   - 中介机构角色不可互推（主承销商 ≠ 受托管理人）
   - 公司类别不能只靠名称判断
   - 重组报告书中的"发行"多指发行股份，不是发行债券
   - 可转债的回售/赎回是保护性条款
   - "以上"含等于，"超过"不含等于

## 输出格式

最后一行必须单独输出答案字母。"""


# ============================================================
# Format-specific prompts
# ============================================================
FC_FORMAT_HINTS = {
    'mcq': """【单选题】从选项中选择唯一正确的答案。
- 必须对每个选项独立做三值判断
- 只有 supported 的选项才能选
- 如果多个选项看似 supported，做排他性检查（字段口径、doc_id、本期/注册金额是否混淆）
- 最后一行只输出一个字母""",

    'multi': """【多选题——重要提示】
- 必须对 A/B/C/D 每个选项独立做三值判断
- 选择所有 fully supported 的选项
- 不能因为找到一个正确选项就停止分析
- 多选题通常有 2-4 个正确答案
- 最后一行按字母顺序输出所有正确选项（如：ACD）
⚠️ 只选 1 个选项几乎一定是不完整的！""",

    'tf': """【判断题】
- 将题目陈述拆分为多个独立子命题
- 对每个子命题在对应文档中做三值判断
- 如果题干包含"均/都/全部"，使用 AND 逻辑
- 所有子命题都 supported → A（正确）
- 任一子命题 contradicted 或 insufficient → B（错误）
- 最后一行输出 A 或 B""",
}


class FCAnswerAgent:
    """Financial Contracts Answer Agent with structured field verification (single inference)."""

    def __init__(self):
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        self.model = MODEL_NAME

    def answer(self, question: str, options: Dict[str, str],
               evidence_text: str, answer_format: str,
               doc_ids: List[str],
               atomic_claims: List[Dict] = None) -> Dict:
        """
        Answer a financial contracts question using structured verification.
        Single inference (no voting). Reflection only for multi-select under-selection.

        Returns:
            Dict with: answer, raw_answer, reasoning, prompt_tokens, completion_tokens, total_tokens
        """
        prompt = self._build_prompt(
            question, options, evidence_text, answer_format, doc_ids, atomic_claims
        )

        # Single inference
        result = self._call_api(prompt, temperature=0.1)
        if not result:
            return {
                'answer': 'A',
                'raw_answer': 'ERROR',
                'reasoning': '',
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            }

        raw_answer = result['raw_answer']
        answer = self._normalize_answer(raw_answer, answer_format)
        prompt_tokens = result['usage']['prompt_tokens']
        completion_tokens = result['usage']['completion_tokens']
        total_tokens = result['usage']['total_tokens']

        # Reflection: only for multi-select questions when only 1 option was selected
        if answer_format == 'multi' and len(answer) == 1:
            reflected = self._multi_select_reflection(
                question, options, evidence_text, answer,
                doc_ids, atomic_claims
            )
            if reflected and len(reflected['answer']) > 1:
                answer = reflected['answer']
                raw_answer = raw_answer + "\n\n[Reflection]\n" + reflected['raw_answer']
                prompt_tokens += reflected['usage']['prompt_tokens']
                completion_tokens += reflected['usage']['completion_tokens']
                total_tokens += reflected['usage']['total_tokens']

        return {
            'answer': answer,
            'raw_answer': raw_answer,
            'reasoning': raw_answer,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }

    def _build_prompt(self, question: str, options: Dict[str, str],
                      evidence_text: str, answer_format: str,
                      doc_ids: List[str],
                      atomic_claims: List[Dict] = None) -> str:
        """Build the structured prompt for the Answer Agent."""
        format_hint = FC_FORMAT_HINTS.get(answer_format, FC_FORMAT_HINTS['mcq'])

        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"

        doc_mapping = "文档顺序（严格固定）：\n"
        for i, doc_id in enumerate(doc_ids):
            doc_mapping += f"  第{i+1}份文档 = {doc_id}\n"

        claims_text = ""
        if atomic_claims:
            claims_text = "\n## 原子命题拆解（由 Query Planner 生成）：\n"
            for claim_group in atomic_claims:
                opt = claim_group.get('option', '?')
                logic = claim_group.get('logic', 'AND')
                claims_text += f"\n选项 {opt}（逻辑: {logic}）：\n"
                for c in claim_group.get('claims', []):
                    doc_id_c = c.get('doc_id', '?')
                    field = c.get('field', '?')
                    operator = c.get('operator', 'verify')
                    target = c.get('target', '?')
                    explicit = c.get('requires_explicit_evidence', False)
                    claims_text += f"  - [{doc_id_c}] {field} {operator} \"{target}\""
                    if explicit:
                        claims_text += " （需要明确证据）"
                    claims_text += "\n"

        if answer_format == 'multi':
            analysis_instruction = """请严格按以下步骤分析：

步骤1：构建字段核验表
从证据中提取与题目相关的所有关键字段值，按文档组织。

步骤2：逐项三值判断
对每个选项 A/B/C/D，基于字段核验表和原子命题，判断：
- supported：证据直接支持
- contradicted：证据直接反驳
- insufficient：证据不足

步骤3：汇总
选择所有 supported 的选项。

最后一行单独输出答案字母（如：ACD）。"""

        elif answer_format == 'mcq':
            analysis_instruction = """请严格按以下步骤分析：

步骤1：构建字段核验表
从证据中提取与题目相关的所有关键字段值。

步骤2：逐项三值判断
对每个选项做 supported/contradicted/insufficient 判断。

步骤3：排他性检查
确认只有一个选项是 fully supported。如果多个看似成立，检查：
- 字段口径是否一致？
- doc_id 是否正确？
- 本期/注册金额是否混淆？
- 主体评级/债项评级是否混淆？

最后一行单独输出答案字母（如：B）。"""

        else:  # tf
            analysis_instruction = """请严格按以下步骤分析：

步骤1：命题拆解
将题目陈述拆分为独立子命题，明确每个子命题对应哪个文档。

步骤2：逐一验证
对每个子命题在对应文档的证据中做三值判断。

步骤3：逻辑聚合
- 如果题干包含"均/都/全部"：AND 逻辑
- 所有子命题 supported → A（正确）
- 任一子命题 contradicted 或 insufficient → B（错误）

最后一行单独输出 A 或 B。"""

        prompt = f"""## 证据材料（按文档组织）

{evidence_text}

---

## 题目信息

{doc_mapping}

{format_hint}

题目：{question}

选项：
{options_text.strip()}
{claims_text}

---

{analysis_instruction}"""

        return prompt

    def _call_api(self, prompt: str, temperature: float = 0.1) -> Optional[Dict]:
        """Call Qwen API with retry logic."""
        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': FC_ANSWER_SYSTEM_PROMPT},
                        {'role': 'user', 'content': prompt},
                    ],
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=temperature,
                )
                return {
                    'raw_answer': response.choices[0].message.content,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens,
                    }
                }
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  [RETRY] FC Answer Agent API failed: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [ERROR] FC Answer Agent failed after {retries} attempts: {e}")
                    return None

    def _multi_select_reflection(self, question: str, options: Dict[str, str],
                                  evidence_text: str, current_answer: str,
                                  doc_ids: List[str],
                                  atomic_claims: List[Dict] = None) -> Optional[Dict]:
        """Reflection for multi-select questions where only 1 option was selected."""
        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"

        doc_mapping = ""
        for i, doc_id in enumerate(doc_ids):
            doc_mapping += f"  第{i+1}份文档 = {doc_id}\n"

        unselected = [opt for opt in sorted(options.keys()) if opt not in current_answer]

        reflection_prompt = f"""你之前对一道多选题进行了分析，但只选择了一个选项（{current_answer}）。

⚠️ 重要提醒：这是一道【多选题】，正确答案通常包含 2-4 个选项。只选 1 个选项几乎一定是不完整的。

请你重新仔细审视每个未被选中的选项（{', '.join(unselected)}），使用三值判断法重新评估。

文档顺序：
{doc_mapping}

证据材料：
{evidence_text}

题目：{question}

选项：
{options_text.strip()}

你之前选择了：{current_answer}

请对每个选项重新独立做三值判断：
A. [重新分析] → supported/contradicted/insufficient → ✓/✗
B. [重新分析] → supported/contradicted/insufficient → ✓/✗
C. [重新分析] → supported/contradicted/insufficient → ✓/✗
D. [重新分析] → supported/contradicted/insufficient → ✓/✗

特别注意：
- 每个选项必须独立判断，引用证据中的具体内容
- 不要因为之前只选了一个就认为其他都是错的
- 检查是否有遗漏的 supported 选项

最后一行按字母顺序输出所有正确选项（如：ACD）。"""

        result = self._call_api(reflection_prompt, temperature=0.1)
        if result:
            reflected = self._normalize_answer(result['raw_answer'], 'multi')
            if len(reflected) >= len(current_answer):
                return {
                    'answer': reflected,
                    'raw_answer': result['raw_answer'],
                    'usage': result['usage'],
                }
        return None

    def _normalize_answer(self, raw_answer: str, answer_format: str) -> str:
        """Extract normalized answer from model output."""
        if not raw_answer or raw_answer.startswith("ERROR"):
            return 'A'

        lines = raw_answer.strip().split('\n')

        for line_idx in range(min(5, len(lines))):
            line = lines[-(line_idx + 1)].strip()
            if not line:
                continue

            letters = re.findall(r'[A-D]', line.upper())
            if letters and len(line) < 20:
                if answer_format == 'mcq':
                    return letters[0]
                elif answer_format == 'multi':
                    return ''.join(sorted(set(letters)))
                elif answer_format == 'tf':
                    valid = [l for l in letters if l in ['A', 'B']]
                    return valid[0] if valid else 'A'

        answer_patterns = [
            r'最终答案[是为：:]\s*([A-D]+)',
            r'答案[是为：:]\s*([A-D]+)',
            r'正确答案[是为：:]\s*([A-D]+)',
            r'准确选项[是为：:]\s*([A-D]+)',
            r'选择?\s*([A-D]+)\s*$',
            r'综上[，,]?\s*(?:选择?|答案[是为]?)\s*([A-D]+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, raw_answer, re.MULTILINE)
            if match:
                letters = list(match.group(1).upper())
                if answer_format == 'mcq':
                    return letters[0]
                elif answer_format == 'multi':
                    return ''.join(sorted(set(letters)))
                elif answer_format == 'tf':
                    valid = [l for l in letters if l in ['A', 'B']]
                    return valid[0] if valid else 'A'

        last_part = '\n'.join(lines[-5:]) if len(lines) >= 5 else raw_answer
        letters = re.findall(r'[A-D]', last_part.upper())

        if not letters:
            return 'A'

        if answer_format == 'mcq':
            return letters[-1]
        elif answer_format == 'multi':
            return ''.join(sorted(set(letters)))
        elif answer_format == 'tf':
            valid = [l for l in letters if l in ['A', 'B']]
            return valid[-1] if valid else 'A'

        return letters[0]
