from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for p in (_PROJECT_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods._shared.llm import make_dashscope_client  # noqa: E402
from methods._shared.pipeline import BaseRunner  # noqa: E402
from methods.m04_HiKEY.hikey_financial_parser import (  # noqa: E402
    COMPANY_ALIASES,
    METRIC_ALIASES,
    QueryPlanner,
    SimpleRetriever,
    stable_hash,
)

import config as _config
from diagnostics import build_diagnostics, load_reference_answers, summarize_diagnostics

sys.modules["config"] = _config

from methods.m05_HiKEY_question_option.pipeline import (  # noqa: E402
    HiKEYQuestionOptionIndexManager as _M05IndexManager,
    HiKEYQuestionOptionPipeline as _M05Pipeline,
    _merge_evidence_packs,
    _target_doc_ids,
    format_evidence_for_prompt,
)

logger = logging.getLogger(__name__)


DOMAIN_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "financial_reports": {
        "terms": [
            "营业收入",
            "归母净利润",
            "扣非",
            "净利润",
            "现金流",
            "研发投入",
            "研发费用",
            "分红",
            "每股",
            "同比",
            "环比",
            "资产负债率",
        ],
        "weight": 1.0,
    },
    "regulatory": {
        "terms": [
            "总则",
            "适用范围",
            "应当",
            "可以",
            "不得",
            "例外",
            "处罚",
            "报告",
            "核实",
            "差异",
            "存量",
            "施行",
            "期限",
            "监管",
            "管理",
            "义务",
        ],
        "weight": 1.2,
    },
    "insurance": {
        "terms": [
            "保险责任",
            "责任免除",
            "等待期",
            "犹豫期",
            "给付",
            "退保",
            "现金价值",
            "受益",
            "合同解除",
            "身故",
            "疾病",
            "赔付",
            "免责",
        ],
        "weight": 1.1,
    },
    "financial_contracts": {
        "terms": [
            "发行",
            "募集资金",
            "评级",
            "回售",
            "赎回",
            "违约",
            "担保",
            "期限",
            "票面",
            "到期",
            "本金",
            "利息",
            "条款",
        ],
        "weight": 1.0,
    },
    "research": {
        "terms": [
            "结论",
            "图",
            "表",
            "趋势",
            "预测",
            "行业",
            "公司",
            "同比",
            "环比",
            "销售",
            "收入",
            "利润",
            "毛利率",
            "增长",
        ],
        "weight": 0.8,
    },
}

GENERIC_PENALTY_TERMS = [
    "目录",
    "封面",
    "声明",
    "致辞",
    "前言",
    "附录",
    "索引",
]


class HiKEYHybridLightIndexManager(_M05IndexManager):
    """m05-style HiKEY loader with local structure-aware scoring."""

    def __init__(self, index_dir: Path):
        super().__init__(index_dir)
        self.doc_domains: Dict[str, str] = {}

    def ensure_domain_loaded(self, domain: str) -> None:
        if domain in self._loaded_domains:
            return

        super().ensure_domain_loaded(domain)

        manifest_path = self.index_dir / domain / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for doc_entry in manifest.get("documents", []):
                doc_id = doc_entry.get("doc_id")
                if doc_id and doc_id in self.retrievers:
                    self.doc_domains[doc_id] = domain
        except Exception as exc:
            logger.warning("Failed to map doc to domain for %s: %s", domain, exc)

    def pack_query(
        self,
        query: str,
        topk: int,
        target_doc_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        if target_doc_ids and _config.SKIP_DOC_ROUTING:
            candidate_docs = target_doc_ids
        else:
            candidate_docs = self.route_document(query)

        if not candidate_docs:
            candidate_docs = list(self.retrievers.keys())

        plan = QueryPlanner.plan(query)
        all_items: List[Dict[str, Any]] = []
        routed_docs: List[str] = []
        per_doc_topk = max(4, topk // max(len(candidate_docs), 1) + 1)

        for doc_id in candidate_docs:
            retriever = self.retrievers.get(doc_id)
            if not retriever:
                continue
            pack = retriever.pack_evidence(query, topk=per_doc_topk, max_siblings=_config.MAX_SIBLINGS)
            for item in pack.get("evidence_pack", []):
                item = dict(item)
                item["source_doc_id"] = doc_id
                hybrid_score = self._score_item(query, plan, doc_id, item)
                anchor = dict(item.get("anchor", {}) or {})
                anchor["score"] = float(anchor.get("score", 0) or 0) + hybrid_score
                item["anchor"] = anchor
                item["_hybrid_score"] = hybrid_score
                all_items.append(item)
            routed_docs.append(doc_id)

        all_items.sort(key=lambda x: x.get("anchor", {}).get("score", 0), reverse=True)
        return {
            "query": query,
            "evidence_pack": all_items[:topk],
            "routed_docs": routed_docs,
        }

    def _score_item(
        self,
        query: str,
        plan: Any,
        doc_id: str,
        item: Dict[str, Any],
    ) -> float:
        anchor = item.get("anchor", {}) or {}
        ancestry = item.get("ancestry", {}) or {}
        siblings = item.get("siblings", []) or []
        hay = self._build_haystack(anchor, ancestry, siblings)
        domain = self.doc_domains.get(doc_id, "")

        score = 0.0
        score += self._structure_bonus(anchor, ancestry)
        score += self._entity_bonus(plan, anchor, hay)
        score += self._domain_bonus(domain, anchor, hay)
        score -= self._generic_penalty(anchor, hay)

        numbers = re.findall(r"20\d{2}|\d+(?:\.\d+)?%?", query)
        if numbers and any(num in hay for num in numbers):
            hits = sum(1 for num in numbers if num in hay)
            score += min(hits, 3) * _config.HYBRID_NUMERIC_BONUS

        return round(score, 4)

    def _build_haystack(self, anchor: Dict[str, Any], ancestry: Dict[str, Any], siblings: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for key in ("section_path", "title", "text", "row_header", "metric", "raw_metric", "unit", "table_name"):
            value = anchor.get(key)
            if value:
                parts.append(str(value))
        for key in ("section_path", "title"):
            value = ancestry.get(key)
            if value:
                parts.append(str(value))
        for sibling in siblings[:3]:
            for key in ("text", "row_header", "metric", "table_name"):
                value = sibling.get(key)
                if value:
                    parts.append(str(value))
        return " ".join(parts)

    def _structure_bonus(self, anchor: Dict[str, Any], ancestry: Dict[str, Any]) -> float:
        obj_type = anchor.get("object_type", "")
        unit_type = anchor.get("unit_type", "")

        score = 0.0
        if obj_type == "field_card":
            score += _config.HYBRID_FIELD_CARD_BONUS
        elif unit_type == "financial_field":
            score += _config.HYBRID_FINANCIAL_FIELD_BONUS
        elif unit_type == "table_row":
            score += _config.HYBRID_TABLE_ROW_BONUS
        elif unit_type == "table":
            score += _config.HYBRID_TABLE_BONUS
        elif unit_type == "heading":
            score += _config.HYBRID_HEADING_BONUS

        section_path = str(ancestry.get("section_path") or anchor.get("section_path") or "")
        depth = section_path.count(">") + section_path.count("/")
        score += min(depth, 4) * _config.HYBRID_DEPTH_BONUS
        return score

    def _entity_bonus(self, plan: Any, anchor: Dict[str, Any], hay: str) -> float:
        score = 0.0
        metric = anchor.get("metric")
        row_header = anchor.get("row_header")

        for company in getattr(plan, "companies", []) or []:
            aliases = COMPANY_ALIASES.get(company, [company])
            if company in hay or any(alias and alias in hay for alias in aliases):
                score += _config.HYBRID_COMPANY_BONUS

        for year in getattr(plan, "years", []) or []:
            if str(year) in hay:
                score += _config.HYBRID_YEAR_BONUS

        for m in getattr(plan, "metrics", []) or []:
            aliases = METRIC_ALIASES.get(m, [m])
            if metric == m or row_header == m:
                score += _config.HYBRID_METRIC_EXACT_BONUS
            elif any(alias and alias in hay for alias in aliases):
                score += _config.HYBRID_METRIC_ALIAS_BONUS
        return score

    def _domain_bonus(self, domain: str, anchor: Dict[str, Any], hay: str) -> float:
        spec = DOMAIN_KEYWORDS.get(domain, {})
        terms = spec.get("terms", [])
        weight = float(spec.get("weight", 1.0) or 1.0)
        hits = sum(1 for term in terms if term and term in hay)
        score = min(hits, 3) * weight

        if domain == "financial_reports" and anchor.get("object_type") == "field_card":
            score += 1.0
        if domain == "regulatory" and any(term in hay for term in ("应当", "不得", "可以", "施行", "期限")):
            score += 0.8
        if domain == "insurance" and any(term in hay for term in ("等待期", "犹豫期", "退保", "现金价值")):
            score += 0.8
        if domain == "financial_contracts" and any(term in hay for term in ("发行", "回售", "赎回", "违约", "期限")):
            score += 0.8
        if domain == "research" and any(term in hay for term in ("图", "表", "趋势", "预测", "结论")):
            score += 0.6

        return score

    def _generic_penalty(self, anchor: Dict[str, Any], hay: str) -> float:
        penalty = 0.0
        if anchor.get("unit_type") == "page_image_stub":
            penalty += 1.0
        if any(term in hay for term in GENERIC_PENALTY_TERMS):
            penalty += _config.HYBRID_GENERIC_PENALTY
        return penalty


class HiKEYHybridLightPipeline(_M05Pipeline):
    """m05 pipeline with local structure-aware hybrid retrieval."""

    method_name = _config.METHOD_NAME
    model_name = _config.MODEL_NAME

    def __init__(self) -> None:
        BaseRunner.__init__(
            self,
            output_dir=_config.METHOD_ROOT / "output",
            logs_dir=_config.METHOD_ROOT / "logs",
            concurrency=_config.CONCURRENCY,
            coder=_config.CODER,
            run_desc=_config.RUN_DESC,
        )
        self.client = make_dashscope_client(api_key=_config.DASHSCOPE_API_KEY)
        self.index_manager = HiKEYHybridLightIndexManager(_config.HIKEY_INDEX_DIR)
        self.reference_answers = load_reference_answers(_config.REFERENCE_ANSWERS_PATH)
        logger.info("[%s] using local hybrid retrieval", _config.METHOD_NAME)

    def _retrieve_question_option_evidence(
        self,
        question_text: str,
        domain: str,
        q: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        target_doc_ids = _target_doc_ids(q) if _config.SKIP_DOC_ROUTING else None
        packs: List[Tuple[str, str, Dict[str, Any]]] = []

        global_pack = self.index_manager.pack_query(
            question_text,
            topk=_config.GLOBAL_TOPK,
            target_doc_ids=target_doc_ids,
        )
        packs.append(("question", question_text, global_pack))

        options = q.get("options", {}) or {}
        for key in sorted(options.keys()):
            option_query = f"{question_text}\n{key}. {options[key]}"
            option_pack = self.index_manager.pack_query(
                option_query,
                topk=_config.OPTION_TOPK,
                target_doc_ids=target_doc_ids,
            )
            packs.append((f"option_{key}", option_query, option_pack))

        merged = _merge_evidence_packs(packs, final_topk=_config.FINAL_EVIDENCE_TOPK)
        evidence_text = format_evidence_for_prompt(merged, _config.EVIDENCE_TOKEN_BUDGET)

        if not evidence_text or evidence_text == "(未找到相关证据)":
            all_options = " ".join(str(v) for v in options.values())
            fallback_query = f"{question_text}\n{all_options}".strip()
            fallback_pack = self.index_manager.pack_query(
                fallback_query,
                topk=_config.FINAL_EVIDENCE_TOPK,
                target_doc_ids=target_doc_ids,
            )
            fallback_merged = _merge_evidence_packs(
                [("question+all_options", fallback_query, fallback_pack)],
                final_topk=_config.FINAL_EVIDENCE_TOPK,
            )
            fallback_text = format_evidence_for_prompt(fallback_merged, _config.EVIDENCE_TOKEN_BUDGET)
            if len(fallback_text) > len(evidence_text):
                evidence_text = fallback_text
                merged = fallback_merged

        meta = {
            "target_doc_ids": target_doc_ids or [],
            "query_count": len(packs),
            "final_evidence_count": len(merged.get("evidence_pack", [])),
            "total_unique_evidence": merged.get("total_unique_evidence", 0),
            "retrieval_mode": "hybrid_light",
            "merged_evidence_pack": merged.get("evidence_pack", []),
        }
        return evidence_text, meta

    def process_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        result = super().process_question(q)
        try:
            result["retrieval_diagnostics"] = build_diagnostics(
                q=q,
                pred_answer=result.get("answer", ""),
                retrieval_meta=result.get("retrieval_meta", {}) or {},
                reference_answers=self.reference_answers,
                threshold=_config.MIN_OPTION_PROXY_COVERAGE,
            )
        except Exception as exc:
            logger.warning("[%s] diagnostics failed: %s", q.get("qid"), exc)
            result["retrieval_diagnostics"] = {
                "reference_available": False,
                "gold_answer": "",
                "pred_answer": result.get("answer", ""),
                "answer_exact_match": False,
                "gold_any_option_proxy_hit": False,
                "gold_all_options_proxy_hit": False,
                "supported_gold_options": [],
                "missing_gold_options": [],
                "topk_hit_count": 0,
                "topk_hit_indices": [],
                "topk_hit_items": [],
                "topk_items": [],
                "option_proxy": {},
            }
        return result

    def save_results(self, extra_summary: Optional[Dict] = None) -> Dict:
        summary_extra = dict(extra_summary or {})
        summary_extra["retrieval_diagnostics"] = summarize_diagnostics(self.results)
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


__all__ = [
    "HiKEYHybridLightIndexManager",
    "HiKEYHybridLightPipeline",
]
