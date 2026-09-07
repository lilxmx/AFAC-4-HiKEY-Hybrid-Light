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
    ENABLE_GLOBAL_FALLBACK,
    ENABLE_LLM_SECTION_RERANK,
    ENABLE_SECTION_ROUTING,
    ENABLE_THINKING,
    EVIDENCE_TOKEN_BUDGET,
    FINAL_EVIDENCE_TOPK,
    FORMAT_HINTS,
    GLOBAL_TOPK,
    HIKEY_INDEX_DIR,
    MAX_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS_REFLECTION,
    MAX_OUTPUT_TOKENS_RERANK,
    MAX_SIBLINGS,
    METHOD_NAME,
    METHOD_ROOT,
    MIN_OPTION_PROXY_COVERAGE,
    MIN_ROUTED_EVIDENCE,
    MODEL_NAME,
    MULTI_QA_PROMPT_TEMPLATE,
    MULTI_REFLECTION_PROMPT,
    OPTION_TOPK,
    QA_PROMPT_TEMPLATE,
    REFERENCE_ANSWERS_PATH,
    ROUTED_SECTION_TOPM,
    RUN_DESC,
    SKIP_DOC_ROUTING,
    SECTION_CANDIDATE_TOPK,
    SECTION_PREVIEW_CHARS,
    SECTION_RERANK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_REFLECTION,
    TEMPERATURE,
    TF_PROMPT_TEMPLATE,
    THINKING_BUDGET,
)
from diagnostics import build_diagnostics, load_reference_answers, summarize_diagnostics
from section_router import SectionRouteResult, SectionRouter

logger = logging.getLogger(__name__)


class HiKEYSectionRoutedIndexManager:
    """Loads m04 HiKEY indices and supports section-limited retrieval."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.retrievers: Dict[str, SimpleRetriever] = {}
        self.doc_cards: Dict[str, Dict[str, Any]] = {}
        self.section_cards: Dict[str, Dict[str, Any]] = {}
        self.section_docs: Dict[str, List[str]] = {}
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

                for sec in retriever.sections:
                    self.section_cards[sec.get("section_id")] = sec
                    self.section_docs.setdefault(doc_id, []).append(sec.get("section_id"))

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

    def section_docs_for_doc(self, doc_id: str) -> List[str]:
        return list(self.section_docs.get(doc_id, []))

    def pack_query(
        self,
        query: str,
        topk: int,
        target_doc_ids: Optional[List[str]],
        allowed_section_ids: Optional[set[str]] = None,
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
            if allowed_section_ids is None:
                pack = retriever.pack_evidence(query, topk=per_doc_topk, max_siblings=MAX_SIBLINGS)
            else:
                pack = self._pack_evidence_restricted(
                    retriever,
                    query=query,
                    topk=per_doc_topk,
                    allowed_section_ids=allowed_section_ids,
                )
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

    def _pack_evidence_restricted(
        self,
        retriever: SimpleRetriever,
        query: str,
        topk: int,
        allowed_section_ids: set[str],
    ) -> Dict[str, Any]:
        """Pack evidence after restricting anchors to routed sections.

        The over-fetch is deliberate: if we take a tiny global top-k first and
        filter afterwards, routed sections can be starved by unrelated global
        hits. This keeps the section route meaningful without changing m04.
        """
        overfetch = max(80, topk * 30)
        anchors = []
        for anchor in retriever.search(query, topk=overfetch):
            section_id = self._anchor_section_id(retriever, anchor)
            if section_id in allowed_section_ids:
                anchors.append(anchor)
            if len(anchors) >= topk:
                break

        packed = []
        used_ids = set()
        for anchor in anchors:
            section_id = self._anchor_section_id(retriever, anchor)
            if anchor.get("object_type") == "field_card":
                source_unit = retriever.unit_by_id.get(anchor.get("source_unit_id"), {})
            else:
                source_unit = anchor if anchor.get("unit_id") else {}
            sec = retriever.section_by_id.get(section_id, {})

            siblings = []
            for u in retriever.units_by_section.get(section_id, []):
                if u.get("unit_id") == source_unit.get("unit_id"):
                    continue
                if (
                    u.get("unit_type") in {"heading", "financial_field", "table_row", "table"}
                    or any(k in u.get("text", "") for k in ["单位", "调整", "同比", "不实施", "现金分红"])
                ):
                    siblings.append(u)
                if len(siblings) >= MAX_SIBLINGS:
                    break

            item = {
                "anchor": anchor,
                "ancestry": {
                    "section_id": section_id,
                    "section_path": sec.get("section_path") or anchor.get("section_path"),
                    "start_page": sec.get("start_page"),
                    "end_page": sec.get("end_page"),
                    "title": sec.get("title"),
                },
                "siblings": siblings,
            }
            anchor_id = anchor.get("unit_id") or anchor.get("field_id") or stable_hash(json.dumps(anchor, ensure_ascii=False))
            if anchor_id not in used_ids:
                used_ids.add(anchor_id)
                packed.append(item)
        return {"query": query, "evidence_pack": packed}

    @staticmethod
    def _anchor_section_id(retriever: SimpleRetriever, anchor: Dict[str, Any]) -> str:
        if anchor.get("object_type") == "field_card":
            source_unit = retriever.unit_by_id.get(anchor.get("source_unit_id"), {})
            return source_unit.get("section_id", "") or anchor.get("section_id", "")
        return anchor.get("section_id", "")

    def route_document(self, query: str) -> List[str]:
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
        "query": "question+options",
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


def _section_ids_from_route(route_result: SectionRouteResult, index_manager: HiKEYSectionRoutedIndexManager) -> set[str]:
    section_ids: set[str] = set()
    for sec in route_result.selected_sections:
        section_id = sec.get("section_id")
        if not section_id:
            continue
        section_ids.add(section_id)
        for child_id in index_manager.section_docs_for_doc(sec.get("doc_id", "")):
            child = index_manager.section_cards.get(child_id, {})
            path = child.get("section_path", "")
            parent_path = sec.get("section_path", "")
            if path and parent_path and path.startswith(parent_path):
                section_ids.add(child_id)
    return section_ids


class HiKEYSectionRoutedPipeline(BaseRunner):
    """Section-routed question+option retrieval pipeline."""

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
        self.index_manager = HiKEYSectionRoutedIndexManager(HIKEY_INDEX_DIR)
        self.reference_answers = load_reference_answers(REFERENCE_ANSWERS_PATH)
        self.section_router = SectionRouter(
            llm_call=self._call_qwen,
            candidate_topk=SECTION_CANDIDATE_TOPK,
            topm=ROUTED_SECTION_TOPM,
            preview_chars=SECTION_PREVIEW_CHARS,
        )
        self._diagnostics: Dict[str, Dict[str, Any]] = {}

    def setup(self, domains: List[str]) -> None:
        for domain in domains:
            self.index_manager.ensure_domain_loaded(domain)

    def process_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        qid = q["qid"]
        domain = q.get("domain") or infer_domain_from_qid(qid)
        question_text = q.get("question", "")
        answer_format = q.get("answer_format", "mcq")

        try:
            target_doc_ids = _target_doc_ids(q) if SKIP_DOC_ROUTING else None
            route_result = None
            allowed_section_ids = None
            if ENABLE_SECTION_ROUTING:
                route_result = self.section_router.route(q, self.index_manager.retrievers, target_doc_ids)
                allowed_section_ids = _section_ids_from_route(route_result, self.index_manager)

            evidence_text, evidence_meta = self._retrieve_question_option_evidence(
                question_text, domain, q, allowed_section_ids, route_result
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
            route_prompt_tokens = route_result.prompt_tokens if route_result else 0
            route_completion_tokens = route_result.completion_tokens if route_result else 0
            route_total_tokens = route_result.total_tokens if route_result else 0
            prompt_tokens = response["prompt_tokens"] + route_prompt_tokens
            completion_tokens = response["completion_tokens"] + route_completion_tokens
            total_tokens = response["total_tokens"] + route_total_tokens

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

            retrieval_diag = build_diagnostics(
                q=q,
                pred_answer=answer,
                evidence_text=evidence_text,
                reference_answers=self.reference_answers,
                threshold=MIN_OPTION_PROXY_COVERAGE,
            )
            self._diagnostics[qid] = retrieval_diag

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
                "retrieval_diagnostics": retrieval_diag,
                "route_meta": route_result.to_meta() if route_result else {},
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
                "retrieval_diagnostics": {
                    "reference_available": False,
                    "gold_answer": "",
                    "pred_answer": "A",
                    "answer_exact_match": False,
                    "gold_any_option_proxy_hit": False,
                    "gold_all_options_proxy_hit": False,
                    "supported_gold_options": [],
                    "missing_gold_options": [],
                    "option_proxy": {},
                },
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

    def _retrieve_question_option_evidence(
        self,
        question_text: str,
        domain: str,
        q: Dict[str, Any],
        allowed_section_ids: Optional[set[str]],
        route_result: Optional[SectionRouteResult],
    ) -> Tuple[str, Dict[str, Any]]:
        target_doc_ids = _target_doc_ids(q) if SKIP_DOC_ROUTING else None
        packs: List[Tuple[str, str, Dict[str, Any]]] = []
        section_limit = allowed_section_ids

        global_pack = self.index_manager.pack_query(
            question_text,
            topk=GLOBAL_TOPK,
            target_doc_ids=target_doc_ids,
            allowed_section_ids=section_limit,
        )
        packs.append(("题干", question_text, global_pack))

        options = q.get("options", {}) or {}
        for key in sorted(options.keys()):
            option_query = f"{question_text}\n{key}. {options[key]}"
            option_pack = self.index_manager.pack_query(
                option_query,
                topk=OPTION_TOPK,
                target_doc_ids=target_doc_ids,
                allowed_section_ids=section_limit,
            )
            packs.append((f"选项{key}", option_query, option_pack))

        merged = _merge_evidence_packs(packs, final_topk=FINAL_EVIDENCE_TOPK)
        evidence_text = format_evidence_for_prompt(merged, EVIDENCE_TOKEN_BUDGET)

        routed_count = len(merged.get("evidence_pack", []))
        used_global_fallback = False
        if ENABLE_GLOBAL_FALLBACK and routed_count < MIN_ROUTED_EVIDENCE:
            all_options = " ".join(str(v) for v in options.values())
            fallback_query = f"{question_text}\n{all_options}".strip()
            fallback_pack = self.index_manager.pack_query(
                fallback_query,
                topk=FINAL_EVIDENCE_TOPK,
                target_doc_ids=target_doc_ids,
                allowed_section_ids=None,
            )
            fallback_merged = _merge_evidence_packs(
                [("题干+全部选项", fallback_query, fallback_pack)],
                final_topk=FINAL_EVIDENCE_TOPK,
            )
            fallback_text = format_evidence_for_prompt(fallback_merged, EVIDENCE_TOKEN_BUDGET)
            if len(fallback_text) > len(evidence_text):
                evidence_text = fallback_text
                merged = fallback_merged
                used_global_fallback = True

        meta = {
            "target_doc_ids": target_doc_ids or [],
            "query_count": len(packs),
            "final_evidence_count": len(merged.get("evidence_pack", [])),
            "total_unique_evidence": merged.get("total_unique_evidence", 0),
            "section_routing_enabled": ENABLE_SECTION_ROUTING,
            "section_rerank_enabled": ENABLE_LLM_SECTION_RERANK,
            "used_global_fallback": used_global_fallback,
            "route_selected_sections": route_result.to_meta().get("selected_sections", []) if route_result else [],
        }
        return evidence_text, meta

    def save_results(self, extra_summary: Optional[Dict] = None) -> Dict:
        summary_extra = dict(extra_summary or {})
        summary_extra["retrieval_diagnostics"] = summarize_diagnostics(self.results)
        summary_extra["routing"] = {
            "enable_section_routing": ENABLE_SECTION_ROUTING,
            "enable_llm_section_rerank": ENABLE_LLM_SECTION_RERANK,
            "section_candidate_topk": SECTION_CANDIDATE_TOPK,
            "routed_section_topm": ROUTED_SECTION_TOPM,
            "min_routed_evidence": MIN_ROUTED_EVIDENCE,
            "min_option_proxy_coverage": MIN_OPTION_PROXY_COVERAGE,
        }
        summary = super().save_results(extra_summary=summary_extra)
        diag_path = self.output_dir / "retrieval_diagnostics.json"
        question_diagnostics = {
            qid: r.get("retrieval_diagnostics", {})
            for qid, r in self.results.items()
            if r.get("retrieval_diagnostics")
        }
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary_extra["retrieval_diagnostics"],
                    "questions": question_diagnostics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Saved retrieval_diagnostics.json: %s", diag_path)
        return summary
