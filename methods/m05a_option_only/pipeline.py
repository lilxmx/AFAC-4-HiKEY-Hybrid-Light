"""
m05a pipeline: HiKEY retrieval with option-only searches (ablation — no question stem).

Compared with m05, this method removes the question-stem-only retrieval pass.
It runs local retrieval for:
  - question + option A
  - question + option B
  - question + option C
  - question + option D
Then it deduplicates all evidence items and sends only the best items to Qwen.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods._shared.config_base import infer_domain_from_qid
from methods._shared.llm import make_dashscope_client, normalize_answer
from methods._shared.pipeline import BaseRunner

from methods.m04_HiKEY.hikey_financial_parser import (
    COMPANY_ALIASES,
    METRIC_ALIASES,
    QueryPlanner,
    SimpleRetriever,
    stable_hash,
)

from config import (
    CODER,
    CONCURRENCY,
    DASHSCOPE_API_KEY,
    DOMAIN_HINTS,
    ENABLE_THINKING,
    EVIDENCE_TOKEN_BUDGET,
    FINAL_EVIDENCE_TOPK,
    FORMAT_HINTS,
    HIKEY_INDEX_DIR,
    MAX_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS_REFLECTION,
    MAX_SIBLINGS,
    METHOD_NAME,
    METHOD_ROOT,
    MODEL_NAME,
    MULTI_QA_PROMPT_TEMPLATE,
    MULTI_REFLECTION_PROMPT,
    OPTION_TOPK,
    QA_PROMPT_TEMPLATE,
    RUN_DESC,
    SKIP_DOC_ROUTING,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_REFLECTION,
    TEMPERATURE,
    TF_PROMPT_TEMPLATE,
    THINKING_BUDGET,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index Manager (reused from m05, identical logic)
# ---------------------------------------------------------------------------

class HiKEYOptionOnlyIndexManager:
    """Loads m04 HiKEY indices and performs target-doc retrieval."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.retrievers: Dict[str, SimpleRetriever] = {}
        self.doc_cards: Dict[str, Dict[str, Any]] = {}
        self._loaded_domains: set[str] = set()

    def ensure_domain_loaded(self, domain: str) -> None:
        if domain in self._loaded_domains:
            return

        manifest_path = self.index_dir / domain / "manifest.json"
        if not manifest_path.exists():
            logger.warning("HiKEY manifest not found for domain=%s: %s", domain, manifest_path)
            self._loaded_domains.add(domain)
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for doc_entry in manifest.get("documents", []):
            doc_id = doc_entry["doc_id"]
            doc_dir = self.index_dir / domain / doc_id
            sections_path = doc_dir / "sections.jsonl"
            units_path = doc_dir / "units.jsonl"
            field_cards_path = doc_dir / "field_cards.jsonl"
            if not sections_path.exists() or not units_path.exists():
                logger.warning("Missing HiKEY files for %s/%s", domain, doc_id)
                continue

            try:
                retriever = SimpleRetriever(
                    sections_path,
                    units_path,
                    field_cards_path if field_cards_path.exists() else None,
                )
                self.retrievers[doc_id] = retriever

                doc_card_path = doc_dir / "doc_card.json"
                if doc_card_path.exists():
                    with open(doc_card_path, "r", encoding="utf-8") as f:
                        self.doc_cards[doc_id] = json.load(f)

                logger.info(
                    "Loaded HiKEY index %s/%s (sections=%s units=%s fields=%s)",
                    domain,
                    doc_id,
                    len(retriever.sections),
                    len(retriever.units),
                    len(retriever.field_cards),
                )
            except Exception as e:
                logger.warning("Failed to load HiKEY index %s/%s: %s", domain, doc_id, e)

        self._loaded_domains.add(domain)

    def route_document(self, query: str) -> List[str]:
        """Fallback routing for open-domain mode."""
        plan = QueryPlanner.plan(query)
        candidates: List[Tuple[str, float]] = []
        for doc_id, doc_card in self.doc_cards.items():
            score = 0.0
            doc_company = doc_card.get("company", "")
            for company in plan.companies:
                aliases = COMPANY_ALIASES.get(company, [company])
                if doc_company == company or any(a in doc_company for a in aliases):
                    score += 10.0
                    break

            doc_year = doc_card.get("year")
            if doc_year and doc_year in plan.years:
                score += 5.0

            section_text = " ".join(doc_card.get("top_sections", []))
            for metric in plan.metrics:
                aliases = METRIC_ALIASES.get(metric, [metric])
                if any(a in section_text for a in aliases):
                    score += 1.0

            if score > 0:
                candidates.append((doc_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in candidates[:3]]

    def pack_query(
        self,
        query: str,
        topk: int,
        target_doc_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        if target_doc_ids and SKIP_DOC_ROUTING:
            candidate_docs = target_doc_ids
        else:
            candidate_docs = self.route_document(query)

        if not candidate_docs:
            candidate_docs = list(self.retrievers.keys())

        all_items: List[Dict[str, Any]] = []
        routed_docs: List[str] = []
        per_doc_topk = max(3, topk // max(len(candidate_docs), 1))

        for doc_id in candidate_docs:
            retriever = self.retrievers.get(doc_id)
            if not retriever:
                continue
            pack = retriever.pack_evidence(query, topk=per_doc_topk, max_siblings=MAX_SIBLINGS)
            for item in pack.get("evidence_pack", []):
                item["source_doc_id"] = doc_id
                all_items.append(item)
            routed_docs.append(doc_id)

        all_items.sort(key=lambda x: x.get("anchor", {}).get("score", 0), reverse=True)
        return {
            "query": query,
            "evidence_pack": all_items[:topk],
            "routed_docs": routed_docs,
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_options(q: Dict[str, Any]) -> str:
    options = q.get("options", {}) or {}
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options.keys()))


def _target_doc_ids(q: Dict[str, Any]) -> List[str]:
    doc_ids = q.get("doc_ids", []) or []
    if not doc_ids and q.get("doc_id"):
        doc_ids = [q["doc_id"]]
    return doc_ids


def _anchor_key(item: Dict[str, Any]) -> str:
    anchor = item.get("anchor", {})
    return (
        anchor.get("unit_id")
        or anchor.get("field_id")
        or anchor.get("source_unit_id")
        or stable_hash(json.dumps(anchor, ensure_ascii=False, sort_keys=True))
    )


def _anchor_score(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("anchor", {}).get("score", 0) or 0)
    except Exception:
        return 0.0


def _merge_evidence_packs(packs: List[Tuple[str, str, Dict[str, Any]]], final_topk: int) -> Dict[str, Any]:
    """Deduplicate evidence by anchor id and boost items hit by multiple queries."""
    merged: Dict[str, Dict[str, Any]] = {}

    for label, query, pack in packs:
        for item in pack.get("evidence_pack", []):
            key = _anchor_key(item)
            if key not in merged:
                item = dict(item)
                item["_query_labels"] = [label]
                item["_queries"] = [query]
                item["_best_score"] = _anchor_score(item)
                merged[key] = item
            else:
                existing = merged[key]
                if label not in existing["_query_labels"]:
                    existing["_query_labels"].append(label)
                existing["_queries"].append(query)
                existing["_best_score"] = max(existing["_best_score"], _anchor_score(item))

    items = list(merged.values())

    def final_score(item: Dict[str, Any]) -> float:
        return float(item.get("_best_score", 0)) + 1.25 * (len(item.get("_query_labels", [])) - 1)

    items.sort(key=final_score, reverse=True)
    return {
        "query": "options-only",
        "evidence_pack": items[:final_topk],
        "total_unique_evidence": len(items),
    }


def format_evidence_for_prompt(evidence_pack: Dict[str, Any], token_budget: int) -> str:
    items = evidence_pack.get("evidence_pack", [])
    if not items:
        return "(未找到相关证据)"

    lines: List[str] = []
    char_budget = token_budget * 2
    used = 0

    for i, item in enumerate(items, 1):
        anchor = item.get("anchor", {})
        ancestry = item.get("ancestry", {})
        siblings = item.get("siblings", [])
        labels = ", ".join(item.get("_query_labels", []))

        block: List[str] = [f"[证据 {i}]"]
        if labels:
            block.append(f"命中来源: {labels}")

        doc_id = item.get("source_doc_id") or anchor.get("doc_id")
        if doc_id:
            block.append(f"文档: {doc_id}")

        section_path = ancestry.get("section_path") or anchor.get("section_path", "")
        if section_path:
            block.append(f"章节路径: {section_path}")

        page = anchor.get("source_page") or anchor.get("page") or ancestry.get("start_page")
        if page:
            block.append(f"页码: {page}")

        obj_type = anchor.get("object_type", "")
        if obj_type == "field_card":
            metric = anchor.get("metric", "")
            value_map = anchor.get("value_map", {})
            unit = anchor.get("unit", "")
            table_name = anchor.get("table_name", "")
            if metric:
                block.append(f"指标: {metric}")
            if unit:
                block.append(f"单位: {unit}")
            if table_name:
                block.append(f"表格: {table_name}")
            block.append(f"数值: {json.dumps(value_map, ensure_ascii=False)}")
        else:
            text = anchor.get("text", "")
            if text:
                block.append(f"内容: {text[:900]}")
            row_header = anchor.get("row_header", "")
            if row_header:
                block.append(f"行标题: {row_header}")
            unit = anchor.get("unit", "")
            if unit:
                block.append(f"单位: {unit}")
            table_name = anchor.get("table_name", "")
            if table_name:
                block.append(f"表格: {table_name}")

        sibling_texts = []
        for s in siblings[:4]:
            s_text = s.get("text", "")[:220]
            if s_text:
                sibling_texts.append(s_text)
        if sibling_texts:
            block.append("相关上下文: " + " | ".join(sibling_texts))

        block_text = "\n".join(block)
        if used + len(block_text) > char_budget:
            break
        lines.append(block_text)
        used += len(block_text)

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HiKEYOptionOnlyPipeline(BaseRunner):
    """Option-only retrieval pipeline (ablation of m05)."""

    method_name = METHOD_NAME
    model_name = MODEL_NAME

    def __init__(self):
        super().__init__(
            output_dir=METHOD_ROOT / "output",
            logs_dir=METHOD_ROOT / "logs",
            concurrency=CONCURRENCY,
            coder=CODER,
            run_desc=RUN_DESC,
        )
        self.client = make_dashscope_client(api_key=DASHSCOPE_API_KEY)
        self.index_manager = HiKEYOptionOnlyIndexManager(HIKEY_INDEX_DIR)

    def setup(self, domains: List[str]) -> None:
        for domain in domains:
            self.index_manager.ensure_domain_loaded(domain)

    def process_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        qid = q["qid"]
        domain = q.get("domain") or infer_domain_from_qid(qid)
        question_text = q.get("question", "")
        answer_format = q.get("answer_format", "mcq")

        try:
            evidence_text, evidence_meta = self._retrieve_option_only_evidence(
                question_text, domain, q
            )
            options_text = format_options(q)
            format_hint = FORMAT_HINTS.get(answer_format, FORMAT_HINTS["mcq"])
            domain_hint = DOMAIN_HINTS.get(domain, "")

            if answer_format == "multi":
                prompt = MULTI_QA_PROMPT_TEMPLATE.format(
                    question=question_text,
                    options=options_text,
                    evidence=evidence_text,
                    format_hint=format_hint,
                    domain_hint=domain_hint,
                )
            elif answer_format == "tf":
                prompt = TF_PROMPT_TEMPLATE.format(
                    question=question_text,
                    options=options_text,
                    evidence=evidence_text,
                    format_hint=format_hint,
                    domain_hint=domain_hint,
                )
            else:
                prompt = QA_PROMPT_TEMPLATE.format(
                    question=question_text,
                    options=options_text,
                    evidence=evidence_text,
                    format_hint=format_hint,
                    domain_hint=domain_hint,
                )

            response = self._call_qwen(SYSTEM_PROMPT, prompt, MAX_OUTPUT_TOKENS)
            raw_answer = response["raw_answer"]
            reasoning_text = response["reasoning_text"]
            answer = normalize_answer(raw_answer, answer_format=answer_format)
            prompt_tokens = response["prompt_tokens"]
            completion_tokens = response["completion_tokens"]
            total_tokens = response["total_tokens"]

            # Multi-select reflection
            if answer_format == "multi" and len(answer) <= 1:
                reflection_prompt = MULTI_REFLECTION_PROMPT.format(
                    previous_answer=answer,
                    evidence=evidence_text,
                    question=question_text,
                    options=options_text,
                    domain_hint=domain_hint,
                )
                try:
                    reflection = self._call_qwen(
                        SYSTEM_PROMPT_REFLECTION,
                        reflection_prompt,
                        MAX_OUTPUT_TOKENS_REFLECTION,
                    )
                    reflection_answer = normalize_answer(
                        reflection["raw_answer"],
                        answer_format="multi",
                    )
                    if len(reflection_answer) > len(answer):
                        answer = reflection_answer
                        raw_answer = raw_answer + "\n[REFLECTION]\n" + reflection["raw_answer"]
                        reasoning_text = (
                            reasoning_text
                            + "\n[REFLECTION_THINKING]\n"
                            + reflection["reasoning_text"]
                        )
                    prompt_tokens += reflection["prompt_tokens"]
                    completion_tokens += reflection["completion_tokens"]
                    total_tokens += reflection["total_tokens"]
                except Exception as e:
                    logger.warning("[%s] reflection failed: %s", qid, e)

            return {
                "qid": qid,
                "answer": answer,
                "raw_answer": raw_answer,
                "reasoning": reasoning_text[:3000] if reasoning_text else "",
                "domain": domain,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "evidence": [evidence_text[:3000]],
                "retrieval_meta": evidence_meta,
            }
        except Exception as e:
            logger.exception("[%s] failed", qid)
            return {
                "qid": qid,
                "answer": "A",
                "raw_answer": f"ERROR: {e}",
                "domain": domain,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def _call_qwen(self, system_prompt: str, user_prompt: str, max_tokens: int) -> Dict[str, Any]:
        extra_body = {}
        if ENABLE_THINKING:
            extra_body["enable_thinking"] = True
            if THINKING_BUDGET:
                extra_body["thinking_budget"] = THINKING_BUDGET

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
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

    def _retrieve_option_only_evidence(
        self,
        question_text: str,
        domain: str,
        q: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Retrieve evidence using ONLY option queries (no question stem query).

        This is the key ablation difference from m05:
        - m05: stem(10) + optionA(10) + optionB(10) + optionC(10) + optionD(10) = 50
        - m05a: optionA(13) + optionB(13) + optionC(13) + optionD(13) = 52
        """
        target_doc_ids = _target_doc_ids(q) if SKIP_DOC_ROUTING else None
        packs: List[Tuple[str, str, Dict[str, Any]]] = []

        # NO question stem retrieval — this is the ablation!

        # Only option-based retrieval
        options = q.get("options", {}) or {}
        for key in sorted(options.keys()):
            option_query = f"{question_text}\n{key}. {options[key]}"
            option_pack = self.index_manager.pack_query(
                option_query,
                topk=OPTION_TOPK,
                target_doc_ids=target_doc_ids,
            )
            packs.append((f"选项{key}", option_query, option_pack))

        merged = _merge_evidence_packs(packs, final_topk=FINAL_EVIDENCE_TOPK)
        evidence_text = format_evidence_for_prompt(merged, EVIDENCE_TOKEN_BUDGET)

        if not evidence_text or evidence_text == "(未找到相关证据)":
            # Last-resort: search all options together
            all_options = " ".join(str(v) for v in options.values())
            fallback_query = f"{question_text}\n{all_options}".strip()
            fallback_pack = self.index_manager.pack_query(
                fallback_query,
                topk=FINAL_EVIDENCE_TOPK,
                target_doc_ids=target_doc_ids,
            )
            fallback_merged = _merge_evidence_packs(
                [("全部选项", fallback_query, fallback_pack)],
                final_topk=FINAL_EVIDENCE_TOPK,
            )
            evidence_text = format_evidence_for_prompt(fallback_merged, EVIDENCE_TOKEN_BUDGET)
            merged = fallback_merged

        meta = {
            "target_doc_ids": target_doc_ids or [],
            "query_count": len(packs),
            "final_evidence_count": len(merged.get("evidence_pack", [])),
            "total_unique_evidence": merged.get("total_unique_evidence", 0),
            "ablation": "option-only (no stem query)",
        }
        return evidence_text, meta
