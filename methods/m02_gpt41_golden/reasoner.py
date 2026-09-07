"""
GPT-4.1 Reasoner with Self-Consistency Voting.
Uses Azure OpenAI GPT-4.1 for high-quality reasoning.
Implements multi-round voting for answer reliability.
"""
import re
import time
from typing import Dict, List, Tuple
from collections import Counter
from openai import OpenAI

from config import (
    AZURE_API_KEY, AZURE_BASE_URL,
    REASONING_MODEL, MAX_OUTPUT_TOKENS, TEMPERATURE, VOTING_ROUNDS
)


SYSTEM_PROMPT = """你是一位资深金融法规与文档分析专家，拥有丰富的中国金融监管、保险、证券、基金等领域的专业知识。

你的任务是严格基于提供的证据材料回答金融领域的选择题。

核心分析规则：
1. 仔细阅读所有证据材料，提取与题目相关的关键信息
2. 对每个选项逐一进行独立分析，判断其正确性
3. 引用证据中的具体内容（原文、数字、条款号）作为判断依据
4. 注意数字、比例、期限、条件等细节的精确匹配——差一个字都可能导致选项错误
5. 注意区分"以上"（含等于）和"超过"（不含等于）等措辞差异
6. 在分析结束后，在最后一行单独输出最终答案的字母

特别注意：
- 对于多选题，你必须对每个选项独立判断正误，不要因为已经找到一个正确选项就停止分析
- 多选题通常有2-4个正确答案，只选1个几乎一定是错误的
- 判断题要仔细验证陈述中的每一个条件是否都成立"""


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

# Domain-specific analysis hints
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


class GoldenReasoner:
    """GPT-4.1 based reasoner with self-consistency voting."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_BASE_URL,
        )
        self.model = REASONING_MODEL
    
    def reason_with_voting(self, question: str, options: Dict[str, str],
                           evidence: str, answer_format: str,
                           domain: str = None,
                           voting_rounds: int = VOTING_ROUNDS) -> Dict:
        """
        Reason about a question using self-consistency voting.
        
        For multi-select questions, uses option-level voting:
        - Each round independently judges each option
        - Final answer includes options that are selected in majority of rounds
        
        Args:
            question: Question text
            options: Dict of option key -> option text
            evidence: Formatted evidence string
            answer_format: One of 'mcq', 'multi', 'tf'
            domain: Domain name for domain-specific hints
            voting_rounds: Number of voting rounds
        
        Returns:
            Dict with keys: answer, raw_answers, confidence, reasoning
        """
        prompt = self._build_prompt(question, options, evidence, answer_format, domain)
        
        # Collect answers from multiple rounds
        answers = []
        raw_answers = []
        temperatures = self._get_temperatures(voting_rounds)
        
        for i, temp in enumerate(temperatures):
            result = self._call_api(prompt, temp)
            if result:
                raw_answers.append(result['raw_answer'])
                answer = normalize_answer(result['raw_answer'], answer_format)
                answers.append(answer)
            else:
                raw_answers.append("ERROR")
                answers.append("A")  # Default fallback
        
        # Voting strategy depends on answer format
        if answer_format == 'multi':
            # Option-level voting for multi-select
            final_answer, confidence = self._option_level_vote(answers)
        else:
            # Standard majority voting for single-select and tf
            final_answer, confidence = self._majority_vote(answers, answer_format)
        
        # Multi-select reflection: if only 1 option selected, likely missed some
        if answer_format == 'multi' and len(final_answer) == 1:
            reflection_result = self._multi_select_reflection(
                question, options, evidence, final_answer,
                answers, raw_answers, domain
            )
            if reflection_result and len(reflection_result) > 1:
                # Reflection found more options - use it
                final_answer = reflection_result
                confidence = 0.85  # Slightly lower confidence for reflected answer
                answers.append(reflection_result)  # Record reflection vote
        
        # If low confidence, try general reflection
        elif confidence < 0.67 and len(answers) >= 2:
            reflection_result = self._reflect(
                question, options, evidence, answer_format,
                answers, raw_answers, domain
            )
            if reflection_result:
                final_answer = reflection_result
                confidence = 0.8  # Boosted confidence after reflection
        
        return {
            'answer': final_answer,
            'raw_answers': raw_answers,
            'all_votes': answers,
            'confidence': confidence,
            'reasoning': raw_answers[0] if raw_answers else "",
        }
    
    def _build_prompt(self, question: str, options: Dict[str, str],
                      evidence: str, answer_format: str,
                      domain: str = None) -> str:
        """Build detailed CoT prompt with domain-specific hints."""
        format_hint = FORMAT_HINTS.get(answer_format, FORMAT_HINTS['mcq'])
        domain_hint = DOMAIN_HINTS.get(domain, '') if domain else ''
        
        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"
        
        # Different analysis structure for multi-select vs single-select
        if answer_format == 'multi':
            analysis_instruction = """请按以下结构分析：
1. 概括证据材料中与题目相关的关键信息
2. 对每个选项独立判断（必须逐一分析，不能跳过）：
   A. [分析内容] → ✓正确 / ✗错误
   B. [分析内容] → ✓正确 / ✗错误
   C. [分析内容] → ✓正确 / ✗错误
   D. [分析内容] → ✓正确 / ✗错误
3. 汇总所有标记为✓的选项，组成最终答案

最后一行请单独输出答案字母（如：ACD）。"""
        elif answer_format == 'mcq':
            analysis_instruction = """请按以下步骤分析：
1. 概括证据材料中与题目相关的关键信息
2. 逐一分析每个选项，判断其正确性，引用证据中的具体内容
3. 特别注意数字、比例、期限、条件等细节是否精确匹配
4. 排除明显错误的选项，确定唯一正确答案

最后一行请单独输出答案字母（如：A）。"""
        else:  # tf
            analysis_instruction = """请按以下步骤分析：
1. 将题目陈述拆分为多个独立条件
2. 逐一验证每个条件是否有证据支持
3. 所有条件都成立才能判为正确（A），任一条件不成立则判为错误（B）

最后一行请单独输出A（正确）或B（错误）。"""
        
        prompt = f"""以下是从相关文档中检索到的证据材料：

{evidence}

---

{format_hint}

{domain_hint}

题目：{question}

选项：
{options_text.strip()}

{analysis_instruction}"""
        
        return prompt
    
    def _call_api(self, prompt: str, temperature: float) -> Dict:
        """Call GPT-4.1 API with retry logic."""
        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
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
                    print(f"  [RETRY] API call failed: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [ERROR] API call failed after {retries} attempts: {e}")
                    return None
    
    def _reflect(self, question: str, options: Dict[str, str],
                 evidence: str, answer_format: str,
                 previous_answers: List[str], previous_reasoning: List[str],
                 domain: str = None) -> str:
        """
        Reflection step: review previous answers and reasoning to resolve disagreements.
        """
        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"
        
        domain_hint = DOMAIN_HINTS.get(domain, '') if domain else ''
        format_hint = FORMAT_HINTS.get(answer_format, FORMAT_HINTS['mcq'])
        
        # Build reflection prompt
        prev_analysis = ""
        for i, (ans, reasoning) in enumerate(zip(previous_answers, previous_reasoning)):
            if reasoning != "ERROR":
                snippet = reasoning[-500:] if len(reasoning) > 500 else reasoning
                prev_analysis += f"\n--- 分析{i+1}（答案：{ans}）---\n{snippet}\n"
        
        reflection_prompt = f"""以下是对同一道题目的多次分析，但得出了不同的答案。请你仔细审视这些分析，找出其中的逻辑错误或遗漏，给出最终正确答案。

{format_hint}

{domain_hint}

证据材料：
{evidence}

题目：{question}

选项：
{options_text.strip()}

之前的多次分析：
{prev_analysis}

各次答案：{', '.join(previous_answers)}

请重新仔细分析，特别注意：
1. 之前分析中可能存在的遗漏（尤其是多选题中被忽略的正确选项）
2. 数字、期限等细节是否精确匹配
3. 措辞差异（如"以上"vs"超过"）

最后一行单独输出最终答案字母。"""
        
        result = self._call_api(reflection_prompt, temperature=0.1)
        if result:
            return normalize_answer(result['raw_answer'], answer_format)
        return None
    
    def _option_level_vote(self, answers: List[str]) -> Tuple[str, float]:
        """
        Option-level voting for multi-select questions.
        
        Instead of voting on the complete answer string,
        votes on each option independently:
        - If option X appears in >= 50% of answers, include it
        
        This prevents the "only select one" problem.
        
        Returns:
            (final_answer, confidence)
        """
        if not answers:
            return 'A', 0.0
        
        n = len(answers)
        option_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for ans in answers:
            for letter in ans:
                if letter in option_counts:
                    option_counts[letter] += 1
        
        # Include options that appear in majority (>= 50%) of votes
        threshold = n / 2.0
        selected = [opt for opt, count in sorted(option_counts.items()) if count >= threshold]
        
        if not selected:
            # Fallback: take the most common complete answer
            counter = Counter(answers)
            winner, _ = counter.most_common(1)[0]
            return winner, 0.5
        
        final_answer = ''.join(selected)
        
        # Confidence: average of selected options' vote ratios
        if selected:
            avg_ratio = sum(option_counts[opt] / n for opt in selected) / len(selected)
            confidence = avg_ratio
        else:
            confidence = 0.5
        
        return final_answer, confidence
    
    def _multi_select_reflection(self, question: str, options: Dict[str, str],
                                  evidence: str, current_answer: str,
                                  previous_answers: List[str],
                                  previous_reasoning: List[str],
                                  domain: str = None) -> str:
        """
        Reflection specifically for multi-select questions where only 1 option was selected.
        
        This addresses the common GPT-4.1 failure mode of selecting only one option
        in multi-select questions. The reflection explicitly asks the model to
        reconsider each unselected option.
        
        Returns:
            Reflected answer (may have more options), or None if reflection fails.
        """
        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"
        
        domain_hint = DOMAIN_HINTS.get(domain, '') if domain else ''
        
        # Build the reflection prompt
        unselected = [opt for opt in sorted(options.keys()) if opt not in current_answer]
        
        reflection_prompt = f"""你之前对一道多选题进行了分析，但只选择了一个选项（{current_answer}）。

⚠️ 重要提醒：这是一道【多选题】，正确答案通常包含 2-4 个选项。只选 1 个选项几乎一定是不完整的。

请你重新仔细审视每个未被选中的选项（{', '.join(unselected)}），判断它们是否也是正确的。

{domain_hint}

证据材料：
{evidence}

题目：{question}

选项：
{options_text.strip()}

你之前选择了：{current_answer}

请对每个选项重新独立判断：
A. [重新分析] → ✓正确 / ✗错误
B. [重新分析] → ✓正确 / ✗错误
C. [重新分析] → ✓正确 / ✗错误
D. [重新分析] → ✓正确 / ✗错误

特别注意：
- 不要因为之前只选了一个就认为其他都是错的
- 每个选项必须独立判断，引用证据中的具体内容
- 如果证据确实支持某个选项，就应该选上

最后一行按字母顺序输出所有正确选项（如：ACD）。"""
        
        result = self._call_api(reflection_prompt, temperature=0.1)
        if result:
            reflected_answer = normalize_answer(result['raw_answer'], 'multi')
            # Only accept if reflection found more options
            # (don't let reflection reduce options)
            if len(reflected_answer) >= len(current_answer):
                return reflected_answer
        return None
    
    def _get_temperatures(self, n: int) -> List[float]:
        """Generate temperature values for voting rounds."""
        if n == 1:
            return [0.1]
        elif n == 2:
            return [0.1, 0.4]
        elif n == 3:
            return [0.1, 0.3, 0.5]
        elif n == 5:
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            # Linear interpolation
            return [0.1 + (0.5 - 0.1) * i / (n - 1) for i in range(n)]
    
    def _majority_vote(self, answers: List[str], answer_format: str) -> Tuple[str, float]:
        """
        Perform majority voting on answers.
        
        Returns:
            (winning_answer, confidence)
            confidence = votes_for_winner / total_votes
        """
        if not answers:
            return 'A', 0.0
        
        counter = Counter(answers)
        winner, count = counter.most_common(1)[0]
        confidence = count / len(answers)
        
        return winner, confidence


def normalize_answer(raw_answer: str, answer_format: str) -> str:
    """
    Extract normalized answer from model output.
    Enhanced version with better pattern matching.
    """
    if not raw_answer or raw_answer.startswith("ERROR"):
        return 'A'
    
    # Try to extract from the last line first
    lines = raw_answer.strip().split('\n')
    
    # Check last few lines for answer
    for line_idx in range(min(3, len(lines))):
        line = lines[-(line_idx + 1)].strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Very short line with only letters = likely the answer
        letters = re.findall(r'[A-D]', line.upper())
        if letters and len(line) < 15:
            if answer_format == 'mcq':
                return letters[0]
            elif answer_format == 'multi':
                return ''.join(sorted(set(letters)))
            elif answer_format == 'tf':
                valid = [l for l in letters if l in ['A', 'B']]
                return valid[0] if valid else 'A'
    
    # Fallback: look for explicit answer patterns
    answer_patterns = [
        r'最终答案[是为：:]\s*([A-D]+)',
        r'答案[是为：:]\s*([A-D]+)',
        r'正确答案[是为：:]\s*([A-D]+)',
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
    
    # Last resort: extract from last 5 lines
    last_part = '\n'.join(lines[-5:]) if len(lines) >= 5 else raw_answer
    letters = re.findall(r'[A-D]', last_part.upper())
    
    if not letters:
        return 'A'
    
    if answer_format == 'mcq':
        return letters[-1]  # Take the last letter mentioned
    elif answer_format == 'multi':
        return ''.join(sorted(set(letters)))
    elif answer_format == 'tf':
        valid = [l for l in letters if l in ['A', 'B']]
        return valid[-1] if valid else 'A'
    
    return letters[0]
