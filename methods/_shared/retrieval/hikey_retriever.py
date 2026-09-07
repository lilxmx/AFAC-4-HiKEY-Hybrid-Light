"""HiKEY retriever — the default ``RetrieverBase`` implementation.

This wraps the existing m04 HiKEY parser cache (``sections.jsonl`` /
``units.jsonl`` / ``field_cards.jsonl`` / ``doc_card.json`` /
``manifest.json``) so that any QA pipeline can consume it through the
shared ``Retriever`` protocol.

Behavioural goal
----------------
By default this class reproduces m05's retrieval behaviour bit for bit:
the same ``SimpleRetriever.pack_evidence`` calls, the same per-doc topk
slicing, the same fallback when no ``target_doc_ids`` are provided, and
the same merge logic (via ``RetrieverBase.merge``).

Extension points (in addition to the ones from ``RetrieverBase``)
----------------
- ``_load_index(domain, doc_id, doc_dir)`` — swap the underlying index
  type (e.g. plug in a Lucene/BM25 hybrid).
- ``_per_doc_topk(...)`` — change how a global ``topk`` is split across
  candidate docs.
- ``_score_route_candidate(...)`` — customise document routing weights.
- ``render_content`` (from base) — change prompt-text rendering.
- ``build_subqueries`` (from base) — inject synonyms / counter-queries.

Teammates who want to *replace* the underlying retriever entirely
should write a sibling class implementing the ``Retriever`` protocol
rather than monkey-patch this one.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .api import (
    EvidenceItem,
    EvidencePack,
    RetrievalFilters,
    RetrievalMode,
    RetrieverBase,
    SourceTag,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import of the m04 parser to avoid pulling heavy deps at import time.
# ---------------------------------------------------------------------------

def _import_m04_parser():
    """Return the m04 hikey_financial_parser module on demand.

    The parser is only needed once an index is actually loaded, so we
    keep the import here. We make sure the project root is on
    ``sys.path`` so that ``methods.m04_HiKEY`` resolves correctly when
    this module is imported from arbitrary entry points.
    """
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from methods.m04_HiKEY import hikey_financial_parser as _hkp  # noqa: WPS433
    return _hkp


# ---------------------------------------------------------------------------
# HiKEYRetriever
# ---------------------------------------------------------------------------

class HiKEYRetriever(RetrieverBase):
    """Default HiKEY-backed retriever, behaviour-compatible with m05."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        max_siblings: int = 12,
        skip_doc_routing: bool = True,
        default_route_topn: int = 3,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.max_siblings = max_siblings
        self.skip_doc_routing = skip_doc_routing
        self.default_route_topn = default_route_topn

        # Resolved lazily via ``warmup`` / ``_ensure_domain_loaded``.
        self._retrievers: Dict[str, Any] = {}     # doc_id -> SimpleRetriever
        self._doc_cards: Dict[str, Dict[str, Any]] = {}
        self._loaded_domains: set[str] = set()
        self._doc_to_domain: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self, *, domains: Optional[List[str]] = None) -> None:
        for domain in domains or []:
            self._ensure_domain_loaded(domain)

    def _ensure_domain_loaded(self, domain: str) -> None:
        if domain in self._loaded_domains:
            return

        manifest_path = self.index_dir / domain / "manifest.json"
        if not manifest_path.exists():
            logger.warning(
                "HiKEY manifest not found for domain=%s: %s",
                domain,
                manifest_path,
            )
            self._loaded_domains.add(domain)
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest.get("documents", []):
            doc_id = entry["doc_id"]
            doc_dir = self.index_dir / domain / doc_id
            try:
                self._load_index(domain, doc_id, doc_dir)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to load HiKEY index %s/%s: %s", domain, doc_id, exc
                )

        self._loaded_domains.add(domain)

    # Extension point ---------------------------------------------------
    def _load_index(self, domain: str, doc_id: str, doc_dir: Path) -> None:
        """Load a single document index. Override to swap the index type."""
        sections_path = doc_dir / "sections.jsonl"
        units_path = doc_dir / "units.jsonl"
        field_cards_path = doc_dir / "field_cards.jsonl"

        if not sections_path.exists() or not units_path.exists():
            logger.warning("Missing HiKEY files for %s/%s", domain, doc_id)
            return

        hkp = _import_m04_parser()
        retriever = hkp.SimpleRetriever(
            sections_path,
            units_path,
            field_cards_path if field_cards_path.exists() else None,
        )
        self._retrievers[doc_id] = retriever
        self._doc_to_domain[doc_id] = domain

        doc_card_path = doc_dir / "doc_card.json"
        if doc_card_path.exists():
            with open(doc_card_path, "r", encoding="utf-8") as f:
                self._doc_cards[doc_id] = json.load(f)

        logger.info(
            "Loaded HiKEY index %s/%s (sections=%s units=%s fields=%s)",
            domain,
            doc_id,
            len(retriever.sections),
            len(retriever.units),
            len(retriever.field_cards),
        )

    # ------------------------------------------------------------------
    # Step 1: query understanding
    # ------------------------------------------------------------------

    def plan_query(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        filters: Optional[RetrievalFilters],
    ) -> Dict[str, Any]:
        # We piggy-back on m04's QueryPlanner so downstream code can
        # see companies/years/metrics. Subclasses can override to add
        # negation flags, claim parsing, etc.
        try:
            hkp = _import_m04_parser()
            plan = hkp.QueryPlanner.plan(query)
            from dataclasses import asdict
            return asdict(plan)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Step 2: document routing
    # ------------------------------------------------------------------

    def resolve_targets(
        self,
        query: str,
        plan: Dict[str, Any],
        filters: Optional[RetrievalFilters],
    ) -> List[str]:
        # Hard constraint short-circuit (matches m05 SKIP_DOC_ROUTING=1).
        if filters and filters.get("doc_ids") and self.skip_doc_routing:
            return list(filters["doc_ids"])

        # Otherwise: score loaded doc_cards against the query plan.
        candidates: List[Tuple[str, float]] = []
        for doc_id, doc_card in self._doc_cards.items():
            score = self._score_route_candidate(query, plan, doc_id, doc_card)
            if score > 0:
                candidates.append((doc_id, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in candidates[: self.default_route_topn]]

    # Extension point ---------------------------------------------------
    def _score_route_candidate(
        self,
        query: str,
        plan: Dict[str, Any],
        doc_id: str,
        doc_card: Dict[str, Any],
    ) -> float:
        """Rank a single doc_card against a query plan.

        Mirrors m05 ``HiKEYQuestionOptionIndexManager.route_document``.
        """
        try:
            hkp = _import_m04_parser()
            COMPANY_ALIASES = hkp.COMPANY_ALIASES
            METRIC_ALIASES = hkp.METRIC_ALIASES
        except Exception:
            COMPANY_ALIASES = {}
            METRIC_ALIASES = {}

        score = 0.0
        doc_company = doc_card.get("company", "")
        for company in plan.get("companies", []) or []:
            aliases = COMPANY_ALIASES.get(company, [company])
            if doc_company == company or any(a in doc_company for a in aliases):
                score += 10.0
                break

        doc_year = doc_card.get("year")
        if doc_year and doc_year in (plan.get("years", []) or []):
            score += 5.0

        section_text = " ".join(doc_card.get("top_sections", []) or [])
        for metric in plan.get("metrics", []) or []:
            aliases = METRIC_ALIASES.get(metric, [metric])
            if any(a in section_text for a in aliases):
                score += 1.0

        return score

    # ------------------------------------------------------------------
    # Step 3: query rewriting (default: identity; subclass for counter / field)
    # ------------------------------------------------------------------
    # `RetrieverBase.build_subqueries` already returns ``[query]``; we
    # leave it as-is so m05's behaviour is preserved.

    # ------------------------------------------------------------------
    # Step 4: actual search
    # ------------------------------------------------------------------

    def run_search(
        self,
        subquery: str,
        *,
        mode: RetrievalMode,
        topk: int,
        target_doc_ids: List[str],
        plan: Dict[str, Any],
        filters: Optional[RetrievalFilters],
    ) -> List[Dict[str, Any]]:
        # If the caller did not give us doc_ids and routing returned
        # nothing useful, fall back to *all* loaded docs (matches m05).
        candidate_docs = list(target_doc_ids) or list(self._retrievers.keys())
        if not candidate_docs:
            return []

        per_doc_topk = self._per_doc_topk(topk, len(candidate_docs))

        all_items: List[Dict[str, Any]] = []
        for doc_id in candidate_docs:
            retriever = self._retrievers.get(doc_id)
            if retriever is None:
                continue
            try:
                pack = retriever.pack_evidence(
                    subquery,
                    topk=per_doc_topk,
                    max_siblings=self.max_siblings,
                )
            except Exception as exc:
                logger.warning("pack_evidence failed for %s: %s", doc_id, exc)
                continue

            for raw in pack.get("evidence_pack", []):
                # ``raw`` is the HiKEY {anchor, ancestry, siblings} bundle.
                flat = self._flatten_hikey_item(raw, doc_id)
                all_items.append(flat)

        # Sort by score (descending). The base ``postprocess_items``
        # will re-sort but doing it here keeps the final cap meaningful.
        all_items.sort(
            key=lambda h: float(h.get("score", 0.0) or 0.0), reverse=True
        )
        return all_items[:topk]

    # Extension point ---------------------------------------------------
    def _per_doc_topk(self, total_topk: int, num_docs: int) -> int:
        """How many hits to ask for from each candidate doc."""
        return max(3, total_topk // max(num_docs, 1))

    # ------------------------------------------------------------------
    # Bridging: HiKEY raw bundle -> flat dict consumable by render_items
    # ------------------------------------------------------------------

    def _flatten_hikey_item(
        self,
        bundle: Dict[str, Any],
        doc_id: str,
    ) -> Dict[str, Any]:
        """Flatten a HiKEY {anchor, ancestry, siblings} bundle.

        The flattened dict carries everything ``compute_evidence_id`` /
        ``render_content`` / ``render_items`` need. The original bundle
        is also kept under ``_hikey`` for downstream consumers that want
        the full structure.
        """
        anchor = bundle.get("anchor", {}) or {}
        ancestry = bundle.get("ancestry", {}) or {}
        siblings = bundle.get("siblings", []) or []

        flat: Dict[str, Any] = dict(anchor)
        flat["doc_id"] = doc_id
        flat["source_doc_id"] = doc_id
        flat["section_path"] = (
            ancestry.get("section_path")
            or anchor.get("section_path")
            or ""
        )
        flat["page"] = (
            anchor.get("source_page")
            or anchor.get("page")
            or ancestry.get("start_page")
        )
        flat["unit_type"] = (
            anchor.get("unit_type")
            or anchor.get("object_type")
            or "text_block"
        )
        flat["siblings"] = siblings
        flat["_hikey"] = bundle  # full original bundle
        flat["source"] = self.default_source
        # ``score`` is already on anchor; surface it at top level for sort.
        flat["score"] = float(anchor.get("score", 0.0) or 0.0)
        return flat

    # ------------------------------------------------------------------
    # Step 6: presentation
    # ------------------------------------------------------------------

    @property
    def default_source(self) -> SourceTag:
        return "hikey"

    def render_content(self, raw_hit: Dict[str, Any]) -> str:
        """Reproduce m05's per-evidence formatting.

        Kept identical to ``format_evidence_for_prompt`` per-block layout
        so that swapping m05 onto this retriever yields the same prompts.
        """
        bundle = raw_hit.get("_hikey") or {}
        anchor = bundle.get("anchor", raw_hit) or raw_hit
        ancestry = bundle.get("ancestry", {}) or {}
        siblings = bundle.get("siblings", raw_hit.get("siblings", [])) or []

        block: List[str] = []

        doc_id = raw_hit.get("doc_id") or anchor.get("doc_id")
        if doc_id:
            block.append(f"文档: {doc_id}")

        section_path = (
            ancestry.get("section_path")
            or anchor.get("section_path")
            or raw_hit.get("section_path", "")
        )
        if section_path:
            block.append(f"章节路径: {section_path}")

        page = (
            anchor.get("source_page")
            or anchor.get("page")
            or ancestry.get("start_page")
            or raw_hit.get("page")
        )
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
            block.append(
                f"数值: {json.dumps(value_map, ensure_ascii=False)}"
            )
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

        sibling_texts: List[str] = []
        for s in siblings[:4]:
            s_text = (s.get("text") or "")[:220]
            if s_text:
                sibling_texts.append(s_text)
        if sibling_texts:
            block.append("相关上下文: " + " | ".join(sibling_texts))

        return "\n".join(block)

    # ------------------------------------------------------------------
    # Convenience: a packer that mirrors the legacy ``pack_query`` API.
    # Useful while migrating m05 in-place; new code should call ``retrieve``.
    # ------------------------------------------------------------------

    def pack_query_legacy(
        self,
        query: str,
        topk: int,
        target_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Legacy-shaped output kept for transitional callers."""
        filters: RetrievalFilters = {}
        if target_doc_ids:
            filters["doc_ids"] = list(target_doc_ids)
        pack: EvidencePack = self.retrieve(
            query, mode="global", topk=topk, filters=filters
        )
        # Re-emit the m05-style ``evidence_pack`` list of HiKEY bundles.
        legacy_items: List[Dict[str, Any]] = []
        for it in pack.get("items", []):
            structured = it.get("structured", {}) or {}
            bundle = structured.get("_hikey") or {
                "anchor": structured,
                "ancestry": {},
                "siblings": structured.get("siblings", []),
            }
            bundle = dict(bundle)
            bundle["source_doc_id"] = it.get("doc_id")
            legacy_items.append(bundle)
        return {
            "query": query,
            "evidence_pack": legacy_items,
            "routed_docs": pack.get("meta", {}).get("routed_docs", []),
        }


__all__ = ["HiKEYRetriever"]
