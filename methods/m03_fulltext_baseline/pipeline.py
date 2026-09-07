"""
m03_fulltext_baseline - pipeline.

Strategy: load full text of every referenced doc, concat, send to Qwen-plus,
post-process answer letter. No retrieval, no chunking.

Subclasses BaseRunner so all the boilerplate (concurrency, resume, output
writing) lives in _shared.
"""
import logging
from typing import Dict

from methods._shared.llm import make_dashscope_client, normalize_answer
from methods._shared.parsers import parse_document_fulltext
from methods._shared.pipeline import BaseRunner

from config import (
    CONCURRENCY,
    DASHSCOPE_API_KEY,
    LOGS_DIR,
    MAX_OUTPUT_TOKENS,
    METHOD_NAME,
    MODEL_NAME,
    OUTPUT_DIR,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你将阅读若干金融长文档，并回答选择题。
请只依据文档内容判断，不要使用外部知识。
对每个选项分别判断是否被文档支持。
最后只输出答案字母。

单选题：只输出一个字母，例如 A
多选题：输出所有正确选项，按字母顺序排列，例如 AC
判断题：按题目选项要求输出 A 或 B
不要输出解释。"""


class FullTextPipeline(BaseRunner):
    """Full-text input baseline using Qwen-plus."""

    method_name = METHOD_NAME
    model_name = MODEL_NAME

    def __init__(self):
        super().__init__(
            output_dir=OUTPUT_DIR,
            logs_dir=LOGS_DIR,
            concurrency=CONCURRENCY,
            save_evidence=False,  # full-text method has no per-question evidence
        )
        self.client = make_dashscope_client(api_key=DASHSCOPE_API_KEY)

    # BaseRunner hook
    def process_question(self, question: Dict) -> Dict:
        qid = question["qid"]
        domain = question["domain"]
        doc_ids = question.get("doc_ids", [])
        answer_format = question.get("answer_format", "mcq")

        # Step 1: load full text of each referenced doc (cached on disk).
        doc_texts = []
        for doc_id in doc_ids:
            text = parse_document_fulltext(doc_id, domain)
            if text:
                doc_texts.append(f"=== 文档: {doc_id} ===\n{text}")
            else:
                logger.warning(f"[{qid}] failed to load doc: {doc_id}")
        full_context = "\n\n".join(doc_texts)

        # Step 2: build prompt
        options_text = "\n".join(f"{k}. {v}" for k, v in question["options"].items())
        if answer_format == "multi":
            format_hint = "（本题为多选题，可能有多个正确答案）"
        elif answer_format == "tf":
            format_hint = "（本题为判断题）"
        else:
            format_hint = "（本题为单选题，只有一个正确答案）"

        user_prompt = f"""以下是相关文档内容：

{full_context}

---

请回答以下问题{format_hint}：

{question['question']}

选项：
{options_text}

请只输出答案字母："""

        # Step 3: call LLM
        raw_answer = ""
        prompt_tokens = completion_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            raw_answer = (response.choices[0].message.content or "").strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        except Exception as e:
            logger.error(f"[{qid}] LLM call failed: {e}")
            raw_answer = ""

        # Step 4: normalize
        answer = normalize_answer(raw_answer, answer_format)

        return {
            "qid": qid,
            "domain": domain,
            "answer": answer,
            "raw_answer": raw_answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "doc_count": len(doc_ids),
            "context_chars": len(full_context),
        }
