"""m05c pipeline: HiKEY hierarchy + domain-specific retrieval cards."""
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

from methods._shared.config_base import infer_domain_from_qid
from methods._shared.llm import make_dashscope_client, normalize_answer
from methods._shared.pipeline import BaseRunner
from methods.m04_HiKEY.hikey_financial_parser import QueryPlanner, SimpleRetriever, stable_hash

from methods.m05c_HiKEY_domain.domain_cards import DomainCard, DomainCardIndex, build_domain_cards
from methods.m05c_HiKEY_domain import config

logger = logging.getLogger(__name__)


def format_options(q: Dict[str, Any]) -> str:
    options = q.get("options", {}) or {}
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options.keys()))


def _target_doc_ids(q: Dict[str, Any]) -> List[str]:
    return [str(x) for x in q.get("doc_ids", []) or q.get("documents", []) or []]


def _anchor_key(item: Dict[str, Any]) -> str:
    anchor = item.get("anchor", {})
    return (
        anchor.get("unit_id")
        or anchor.get("field_id")
        or item.get("card_id")
        or stable_hash(json.dumps(item, ensure_ascii=False, sort_keys=True))
    )


def _anchor_score(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("score") or item.get("anchor", {}).get("score") or 0.0)
    except Exception:
        return 0.0


class DomainHiKEYIndexManager:
    """Loads HiKEY retrievers and builds per-domain DomainCard indices."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.retrievers: Dict[str, SimpleRetriever] = {}
        self.doc_domain: Dict[str, str] = {}
        self.doc_cards: Dict[str, Dict[str, Any]] = {}
        self.domain_cards = DomainCardIndex()
        self._loaded_domains: set[str] = set()

    def ensure_domain_loaded(self, domain: str) -> None:
        if domain in self._loaded_domains:
            return
        manifest_path = self.index_dir / domain / "manifest.json"
        if not manifest_path.exists():
            logger.warning("HiKEY manifest not found for domain=%s: %s", domain, manifest_path)
            self._loaded_domains.add(domain)
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded_cards = 0
        for doc_entry in manifest.get("documents", []):
            doc_id = doc_entry["doc_id"]
            doc_dir = self.index_dir / domain / doc_id
            sections_path = doc_dir / "sections.jsonl"
            units_path = doc_dir / "units.jsonl"
            field_cards_path = doc_dir / "field_cards.jsonl"
            if not sections_path.exists() or not units_path.exists():
                continue
            retriever = SimpleRetriever(
                sections_path,
                units_path,
                field_cards_path if field_cards_path.exists() else None,
            )
            self.retrievers[doc_id] = retriever
            self.doc_domain[doc_id] = domain
            doc_card_path = doc_dir / "doc_card.json"
            if doc_card_path.exists():
                self.doc_cards[doc_id] = json.loads(doc_card_path.read_text(encoding="utf-8"))
            cards = build_domain_cards(domain, doc_id, retriever)
            self.domain_cards.add_cards(cards)
            loaded_cards += len(cards)
        self._loaded_domains.add(domain)
        logger.info("Loaded domain=%s retrievers=%s domain_cards=%s", domain, len(self.retrievers), loaded_cards)

    def pack_hikey_query(self, query: str, topk: int, target_doc_ids: Optional[List[str]], max_siblings: int) -> Dict[str, Any]:
        doc_ids = target_doc_ids or [doc_id for doc_id in self.retrievers.keys()]
        per_doc_topk = max(1, min(topk, 4))
        all_items: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            retriever = self.retrievers.get(doc_id)
            if not retriever:
                continue
            pack = retriever.pack_evidence(query, topk=per_doc_topk, max_siblings=max_siblings)
            for item in pack.get("evidence_pack", []):
                item = dict(item)
                item["source_doc_id"] = doc_id
                all_items.append(item)
        all_items.sort(key=_anchor_score, reverse=True)
        return {"query": query, "evidence_pack": all_items[:topk]}

    def search_domain_cards(
        self,
        query: str,
        domain: str,
        topk: int,
        target_doc_ids: Optional[List[str]],
    ) -> List[DomainCard]:
        return self.domain_cards.search(query, domain=domain, topk=topk, target_doc_ids=target_doc_ids)


def _merge_hikey_packs(packs: List[Tuple[str, str, Dict[str, Any]]], final_topk: int) -> List[Dict[str, Any]]:
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
                ex = merged[key]
                if label not in ex["_query_labels"]:
                    ex["_query_labels"].append(label)
                ex["_queries"].append(query)
                ex["_best_score"] = max(ex["_best_score"], _anchor_score(item))

    def final_score(item: Dict[str, Any]) -> float:
        return float(item.get("_best_score", 0)) + 1.1 * (len(item.get("_query_labels", [])) - 1)

    items = list(merged.values())
    items.sort(key=final_score, reverse=True)
    return items[:final_topk]


def _hikey_to_prompt_block(item: Dict[str, Any], evidence_id: str) -> str:
    anchor = item.get("anchor", {})
    ancestry = item.get("ancestry", {})
    labels = ", ".join(item.get("_query_labels", []))
    lines = [f"[{evidence_id}] HiKEY/{anchor.get('object_type', anchor.get('unit_type', 'evidence'))}"]
    if labels:
        lines.append(f"命中来源: {labels}")
    doc_id = item.get("source_doc_id") or anchor.get("doc_id")
    if doc_id:
        lines.append(f"文档: {doc_id}")
    section = ancestry.get("section_path") or anchor.get("section_path")
    if section:
        lines.append(f"章节: {section}")
    page = anchor.get("source_page") or anchor.get("page") or ancestry.get("start_page")
    if page:
        lines.append(f"页码: {page}")
    if anchor.get("object_type") == "field_card":
        structured = {
            "metric": anchor.get("metric"),
            "unit": anchor.get("unit"),
            "value_map": anchor.get("value_map"),
            "table_name": anchor.get("table_name"),
        }
        lines.append(f"结构化字段: {json.dumps(structured, ensure_ascii=False)[:900]}")
    text = anchor.get("text") or anchor.get("row_header") or json.dumps(anchor, ensure_ascii=False)
    lines.append(f"内容: {str(text)[:900]}")
    siblings = []
    for sibling in item.get("siblings", [])[:2]:
        s_text = sibling.get("text", "")
        if s_text:
            siblings.append(s_text[:180])
    if siblings:
        lines.append("相邻上下文: " + " | ".join(siblings))
    return "\n".join(lines)


def _dedup_prompt_blocks(
    blocks: List[Tuple[str, float, str]],
    limit: int,
    token_budget: int,
    reserve_domain_cards: bool,
) -> Tuple[str, List[str]]:
    seen = set()
    chosen: List[Tuple[str, float, str]] = []

    domain_blocks = [b for b in blocks if "-D" in b[0]]
    hikey_blocks = [b for b in blocks if "-D" not in b[0]]
    reserve_domain = min(max(4, limit // 2), len(domain_blocks), limit) if reserve_domain_cards else 0

    ordered: List[Tuple[str, float, str]] = []
    if reserve_domain:
        ordered.extend(sorted(domain_blocks, key=lambda x: x[1], reverse=True)[:reserve_domain])
        ordered.extend(sorted(hikey_blocks, key=lambda x: x[1], reverse=True))
        ordered.extend(sorted(domain_blocks, key=lambda x: x[1], reverse=True)[reserve_domain:])
    else:
        ordered.extend(sorted(blocks, key=lambda x: x[1], reverse=True))

    for eid, score, block in ordered:
        clean = re.sub(r"\s+", "", block)[:220]
        if clean in seen:
            continue
        seen.add(clean)
        chosen.append((eid, score, block))
        if len(chosen) >= limit:
            break
    char_budget = token_budget * 2
    used = 0
    parts = []
    ids = []
    for eid, _, block in chosen:
        if used + len(block) > char_budget:
            break
        parts.append(block)
        ids.append(eid)
        used += len(block)
    return "\n\n".join(parts) if parts else "(未找到相关证据)", ids


class HiKEYDomainPipeline(BaseRunner):
    method_name = config.METHOD_NAME
    model_name = config.MODEL_NAME

    def __init__(self) -> None:
        super().__init__(
            output_dir=config.METHOD_ROOT / "output",
            logs_dir=config.METHOD_ROOT / "logs",
            concurrency=config.CONCURRENCY,
            coder=config.CODER,
            run_desc=config.RUN_DESC,
        )
        self.client = make_dashscope_client(api_key=config.DASHSCOPE_API_KEY)
        self.index_manager = DomainHiKEYIndexManager(config.HIKEY_INDEX_DIR)

    def setup(self, domains: List[str]) -> None:
        for domain in domains:
            self.index_manager.ensure_domain_loaded(domain)

    def process_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        qid = q["qid"]
        domain = q.get("domain") or infer_domain_from_qid(qid)
        answer_format = q.get("answer_format", "mcq")
        question = q.get("question", "")
        try:
            evidence_text, retrieval_meta = self._retrieve_domain_evidence(q, domain)
            prompt = config.VERIFIER_PROMPT_TEMPLATE.format(
                answer_format=answer_format,
                domain=domain,
                domain_hint=config.DOMAIN_HINTS.get(domain, ""),
                question=question,
                options=format_options(q),
                evidence=evidence_text,
            )
            response = self._call_qwen(config.SYSTEM_PROMPT, prompt)
            raw_answer = response["raw_answer"]
            answer = self._extract_answer(raw_answer, answer_format)
            return {
                "qid": qid,
                "answer": answer,
                "raw_answer": raw_answer,
                "reasoning": response.get("reasoning_text", "")[:3000],
                "domain": domain,
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "total_tokens": response["total_tokens"],
                "evidence": [evidence_text[:5000]],
                "retrieval_meta": retrieval_meta,
                # With only answer-letter ground truth available, this is equal
                # to answer correctness and is finalized in evaluation.py.
                "topk_answer_hit": None,
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

    def _retrieve_domain_evidence(self, q: Dict[str, Any], domain: str) -> Tuple[str, Dict[str, Any]]:
        question = q.get("question", "")
        target_doc_ids = _target_doc_ids(q) if config.SKIP_DOC_ROUTING else None
        hikey_packs: List[Tuple[str, str, Dict[str, Any]]] = []
        domain_blocks: List[Tuple[str, float, str]] = []
        per_option_counts: Dict[str, int] = {}
        strong_domain_cards = domain in config.STRONG_DOMAIN_CARD_DOMAINS
        domain_card_topk = config.DOMAIN_CARD_TOPK if strong_domain_cards else config.WEAK_DOMAIN_CARD_TOPK
        per_option_evidence = config.PER_OPTION_EVIDENCE if strong_domain_cards else config.WEAK_PER_OPTION_EVIDENCE
        domain_score_boost = (
            config.REGULATORY_DOMAIN_SCORE_BOOST if strong_domain_cards else config.WEAK_DOMAIN_SCORE_BOOST
        )

        question_pack = self.index_manager.pack_hikey_query(
            question, config.QUESTION_TOPK, target_doc_ids, config.MAX_SIBLINGS
        )
        hikey_packs.append(("题干", question, question_pack))

        question_cards = self.index_manager.search_domain_cards(question, domain, domain_card_topk, target_doc_ids)
        for i, card in enumerate(self._filter_domain_cards(question_cards, strong_domain_cards), 1):
            eid = f"Q-D{i}"
            domain_blocks.append((eid, card.score + domain_score_boost, card.to_prompt_block(eid)))

        options = q.get("options", {}) or {}
        for key in sorted(options.keys()):
            option_query = f"{question}\n{key}. {options[key]}"
            option_pack = self.index_manager.pack_hikey_query(
                option_query, config.OPTION_TOPK, target_doc_ids, config.MAX_SIBLINGS
            )
            hikey_packs.append((f"选项{key}", option_query, option_pack))
            cards = self.index_manager.search_domain_cards(option_query, domain, domain_card_topk, target_doc_ids)
            cards = self._filter_domain_cards(cards, strong_domain_cards)
            per_option_counts[key] = len(cards)
            for i, card in enumerate(cards[:per_option_evidence], 1):
                eid = f"{key}-D{i}"
                domain_blocks.append((eid, card.score + domain_score_boost, card.to_prompt_block(eid)))

        hikey_items = _merge_hikey_packs(hikey_packs, final_topk=config.FINAL_EVIDENCE_TOPK)
        hikey_blocks: List[Tuple[str, float, str]] = []
        for i, item in enumerate(hikey_items, 1):
            eid = f"H{i}"
            hikey_blocks.append((eid, _anchor_score(item), _hikey_to_prompt_block(item, eid)))

        all_blocks = domain_blocks + hikey_blocks
        evidence_text, evidence_ids = _dedup_prompt_blocks(
            all_blocks,
            limit=config.FINAL_EVIDENCE_TOPK,
            token_budget=config.EVIDENCE_TOKEN_BUDGET,
            reserve_domain_cards=strong_domain_cards,
        )
        meta = {
            "target_doc_ids": target_doc_ids or [],
            "query_count": 1 + len(options),
            "final_evidence_count": len(evidence_ids),
            "domain_card_count": len(domain_blocks),
            "domain_card_policy": "strong_reserved" if strong_domain_cards else "weak_rerank_only",
            "hikey_candidate_count": len(hikey_items),
            "per_option_domain_card_counts": per_option_counts,
            "topk_evidence_ids": evidence_ids,
            "topk_definition": "final evidence blocks sent to the model after local domain-card and HiKEY reranking",
        }
        return evidence_text, meta

    @staticmethod
    def _filter_domain_cards(cards: List[DomainCard], strong_domain_cards: bool) -> List[DomainCard]:
        if strong_domain_cards:
            return cards
        return [card for card in cards if card.score >= config.WEAK_DOMAIN_CARD_MIN_SCORE]

    def _call_qwen(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        extra_body = {}
        if config.ENABLE_THINKING:
            extra_body["enable_thinking"] = True
            if config.THINKING_BUDGET:
                extra_body["thinking_budget"] = config.THINKING_BUDGET
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=config.MAX_OUTPUT_TOKENS,
            temperature=config.TEMPERATURE,
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

    @staticmethod
    def _extract_answer(raw_answer: str, answer_format: str) -> str:
        try:
            m = re.search(r"\{.*\}", raw_answer, re.S)
            if m:
                obj = json.loads(m.group(0))
                if obj.get("answer"):
                    return normalize_answer(str(obj["answer"]), answer_format)
        except Exception:
            pass
        return normalize_answer(raw_answer, answer_format)
