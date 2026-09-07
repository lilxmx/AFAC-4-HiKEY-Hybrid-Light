"""Qwen section reranking for m07."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from methods.m04_HiKEY.hikey_financial_parser import (
    COMPANY_ALIASES,
    METRIC_ALIASES,
    QueryPlanner,
    SimpleRetriever,
    metric_aliases_for_query,
    simple_tokens,
)

from config import (
    MAX_OUTPUT_TOKENS_RERANK,
    MULTI_ROUTED_SECTION_TOPM,
    ROUTED_SECTION_TOPM,
    SECTION_CANDIDATE_TOPK,
    SECTION_PREVIEW_CHARS,
    SECTION_RERANK_PROMPT,
    SECTION_RERANK_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class SectionRouteResult:
    candidate_sections: List[Dict[str, Any]] = field(default_factory=list)
    selected_sections: List[Dict[str, Any]] = field(default_factory=list)
    rerank_raw_answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    routing_mode: str = "llm_section_rerank"

    def to_meta(self) -> Dict[str, Any]:
        return {
            "routing_mode": self.routing_mode,
            "candidate_count": len(self.candidate_sections),
            "selected_count": len(self.selected_sections),
            "selected_sections": [
                {
                    "rank": i + 1,
                    "doc_id": s.get("doc_id"),
                    "section_id": s.get("section_id"),
                    "section_path": s.get("section_path"),
                    "score": s.get("score"),
                    "source_labels": s.get("source_labels", []),
                }
                for i, s in enumerate(self.selected_sections)
            ],
            "rerank_raw_answer": self.rerank_raw_answer[:1200],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class SectionRouter:
    """Build section candidates locally, then ask Qwen to rerank them."""

    def __init__(
        self,
        llm_call: Callable[[str, str, int], Dict[str, Any]],
        candidate_topk: int = SECTION_CANDIDATE_TOPK,
        topm: int = ROUTED_SECTION_TOPM,
        preview_chars: int = SECTION_PREVIEW_CHARS,
    ):
        self.llm_call = llm_call
        self.candidate_topk = candidate_topk
        self.topm = topm
        self.preview_chars = preview_chars

    def route(
        self,
        q: Dict[str, Any],
        retrievers: Dict[str, SimpleRetriever],
        target_doc_ids: Optional[List[str]],
    ) -> SectionRouteResult:
        topm = MULTI_ROUTED_SECTION_TOPM if q.get("answer_format") == "multi" else self.topm
        candidates = self._build_candidates(q, retrievers, target_doc_ids)
        if not candidates:
            return SectionRouteResult(routing_mode="no_section_candidates")

        prompt = SECTION_RERANK_PROMPT.format(
            question=q.get("question", ""),
            options=self._format_options(q),
            answer_format=q.get("answer_format", "mcq"),
            domain=q.get("domain", ""),
            sections=self._format_sections(candidates),
            topm=topm,
        )

        try:
            response = self.llm_call(SECTION_RERANK_SYSTEM_PROMPT, prompt, MAX_OUTPUT_TOKENS_RERANK)
            selected = self._parse_selected(response.get("raw_answer", ""), candidates)
            if not selected:
                selected = candidates[:topm]
            return SectionRouteResult(
                candidate_sections=candidates,
                selected_sections=selected[:topm],
                rerank_raw_answer=response.get("raw_answer", ""),
                prompt_tokens=response.get("prompt_tokens", 0),
                completion_tokens=response.get("completion_tokens", 0),
                total_tokens=response.get("total_tokens", 0),
                routing_mode="llm_section_rerank",
            )
        except Exception as e:
            logger.warning("[%s] section rerank failed: %s", q.get("qid"), e)
            return SectionRouteResult(
                candidate_sections=candidates,
                selected_sections=candidates[:topm],
                rerank_raw_answer=f"ERROR: {e}",
                routing_mode="local_candidate_fallback",
            )

    def _build_candidates(
        self,
        q: Dict[str, Any],
        retrievers: Dict[str, SimpleRetriever],
        target_doc_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        question = q.get("question", "")
        options = q.get("options", {}) or {}
        query_specs: List[Tuple[str, str]] = [("question", question)]
        for key in sorted(options.keys()):
            query_specs.append((f"option_{key}", f"{question}\n{key}. {options[key]}"))
        if options:
            query_specs.append(("all_options", question + "\n" + " ".join(str(v) for v in options.values())))

        candidate_docs = target_doc_ids or list(retrievers.keys())
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        per_query_topk = max(self.candidate_topk, self.topm * 3)

        for doc_id in candidate_docs:
            retriever = retrievers.get(doc_id)
            if not retriever:
                continue
            for label, query in query_specs:
                for sec in retriever.search(query, topk=per_query_topk, object_type="section"):
                    section_id = sec.get("section_id")
                    if not section_id:
                        continue
                    key = (doc_id, section_id)
                    score = float(sec.get("score", 0) or 0)
                    if key not in merged:
                        item = dict(sec)
                        item["doc_id"] = doc_id
                        item["source_labels"] = [label]
                        item["best_score"] = score
                        item["route_score"] = score + self._domain_section_boost(q, sec)
                        merged[key] = item
                    else:
                        item = merged[key]
                        if label not in item["source_labels"]:
                            item["source_labels"].append(label)
                        item["best_score"] = max(float(item.get("best_score", 0) or 0), score)
                        item["route_score"] = max(float(item.get("route_score", 0) or 0), score + self._domain_section_boost(q, sec))

        items = list(merged.values())
        for item in items:
            item["route_score"] = round(float(item.get("route_score", 0) or 0) + 1.1 * (len(item.get("source_labels", [])) - 1), 4)
            item["score"] = item["route_score"]
        items.sort(key=lambda x: x.get("route_score", 0), reverse=True)
        return items[:self.candidate_topk]

    def _domain_section_boost(self, q: Dict[str, Any], sec: Dict[str, Any]) -> float:
        text = json.dumps(sec, ensure_ascii=False)
        query = q.get("question", "") + " " + " ".join(str(v) for v in (q.get("options", {}) or {}).values())
        plan = QueryPlanner.plan(query)
        boost = 0.0
        for metric in plan.metrics:
            aliases = METRIC_ALIASES.get(metric, [metric])
            if any(a and a in text for a in aliases):
                boost += 3.0
        for year in plan.years:
            if str(year) in text:
                boost += 1.0
        for company in plan.companies:
            if company in text or any(a and a in text for a in COMPANY_ALIASES.get(company, [])):
                boost += 1.0

        domain = q.get("domain", "")
        priors = {
            "financial_reports": ["主要财务指标", "财务指标", "分红", "研发", "现金流", "利润", "收入"],
            "regulatory": ["总则", "监督管理", "信息披露", "法律责任", "处罚", "报告", "监管", "规则"],
            "insurance": ["保险责任", "责任免除", "等待期", "给付", "现金价值", "退保"],
            "financial_contracts": ["发行条款", "发行", "评级", "募集资金", "违约", "回售", "赎回"],
            "research": ["结论", "图", "表", "趋势", "行业", "公司", "预测"],
        }.get(domain, [])
        if any(p in text for p in priors):
            boost += 1.5

        path_depth = str(sec.get("section_path", "")).count(">") + str(sec.get("section_path", "")).count("/")
        boost += min(path_depth, 4) * 0.15

        for number in re.findall(r"20\d{2}|\d+(?:\.\d+)?%?", query):
            if number and number in text:
                boost += 0.4
        return boost

    def _format_sections(self, candidates: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, sec in enumerate(candidates, 1):
            preview = str(sec.get("text_preview", "") or "")[:self.preview_chars]
            blocks.append(
                f"[{i}] doc_id={sec.get('doc_id')} section_id={sec.get('section_id')}\n"
                f"路径: {sec.get('section_path') or sec.get('title')}\n"
                f"页码: {sec.get('start_page', '')}-{sec.get('end_page', '')}\n"
                f"本地分数: {sec.get('route_score', sec.get('score'))}\n"
                f"摘要: {preview}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _format_options(q: Dict[str, Any]) -> str:
        options = q.get("options", {}) or {}
        return "\n".join(f"{k}. {options[k]}" for k in sorted(options.keys()))

    def _parse_selected(self, raw: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nums = [int(x) for x in re.findall(r"<<<\s*(\d+)\s*>>>", raw or "")]
        if not nums:
            nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", raw or "")]
        selected = []
        seen = set()
        for n in nums:
            idx = n - 1
            if 0 <= idx < len(candidates) and idx not in seen:
                selected.append(candidates[idx])
                seen.add(idx)
        return selected
