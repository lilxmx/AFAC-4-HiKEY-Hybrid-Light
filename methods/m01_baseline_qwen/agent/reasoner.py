"""
Qwen LLM reasoner - handles API calls and prompt construction.
"""
from typing import Dict, Optional

from methods._shared.llm import make_dashscope_client, normalize_answer

from .config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME,
    MAX_OUTPUT_TOKENS, TEMPERATURE
)


SYSTEM_PROMPT = """你是金融文档分析专家。请严格基于以下提供的证据材料回答问题。
规则：
1. 只能使用证据中明确提到的信息，不得使用证据之外的知识
2. 逐项分析每个选项是否正确
3. 在回答的最后一行，单独输出最终答案的字母（如：A 或 ACD）"""


FORMAT_HINTS = {
    'mcq': '【单选题】请从选项中选择唯一正确的答案，最后一行只输出一个字母。',
    'multi': '【多选题】请选择所有正确的选项，最后一行按字母顺序输出（如：ACD）。',
    'tf': '【判断题】请判断题目陈述是否正确，最后一行输出A（正确）或B（错误）。',
}


class Reasoner:
    """Handles LLM reasoning via Qwen API."""
    
    def __init__(self):
        self.client = make_dashscope_client(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.model = MODEL_NAME
    
    def reason(self, question: str, options: Dict[str, str],
               evidence: str, answer_format: str) -> Dict:
        """
        Call Qwen API to reason about the question.
        
        Args:
            question: Question text
            options: Dict of option key -> option text
            evidence: Formatted evidence string
            answer_format: One of 'mcq', 'multi', 'tf'
        
        Returns:
            Dict with keys: raw_answer, answer, prompt_tokens, completion_tokens, total_tokens
        """
        prompt = self._build_prompt(question, options, evidence, answer_format)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            
            raw_answer = response.choices[0].message.content
            usage = response.usage
            
            # Extract normalized answer
            answer = normalize_answer(raw_answer, answer_format)
            
            return {
                'raw_answer': raw_answer,
                'answer': answer,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
            }
        
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            # Return default answer on failure
            default = 'A'
            return {
                'raw_answer': f"ERROR: {str(e)}",
                'answer': default,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            }
    
    def _build_prompt(self, question: str, options: Dict[str, str],
                      evidence: str, answer_format: str) -> str:
        """Build the user prompt for the LLM."""
        format_hint = FORMAT_HINTS.get(answer_format, FORMAT_HINTS['mcq'])
        
        # Build options text
        options_text = ""
        for key in sorted(options.keys()):
            options_text += f"{key}. {options[key]}\n"
        
        prompt = f"""证据材料：
{evidence}

---

{format_hint}

题目：{question}

选项：
{options_text.strip()}

请逐项分析每个选项，最后在最末行单独输出答案字母。"""
        
        return prompt
