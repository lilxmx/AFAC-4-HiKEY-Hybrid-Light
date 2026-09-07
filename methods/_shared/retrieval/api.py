"""Retrieval API contract shared across QA methods (m05/m06/...).

Goals
-----
1. Decouple QA pipelines from any specific retriever implementation
   (HiKEY / Hybrid / Dense / VLM ...).
2. Provide a stable interface the QA Agent can rely on:
       retrieve(query, *, mode, topk, filters, query_label) -> EvidencePack
       merge(packs, *, final_topk) -> EvidencePack
3. Keep the door wide open for teammates to extend retrieval logic
   without touching QA code, by exposing fine-grained override hooks
   on `RetrieverBase`.

Design principles
-----------------
- The contract is described with `TypedDict`s and a `Protocol`, so any
  duck-typed object with the right shape qualifies as a Retriever.
- `RetrieverBase` is an *optional* helper. Teammates can subclass it and
  override one or two hooks, or write their own class from scratch as
  long as it satisfies the `Retriever` protocol.
- The core `retrieve()` flow is split into small steps so that each
  step can be customised independently:
        plan_query      -> understand the query (companies/years/...)
        resolve_targets -> decide which doc_ids to search
        build_subqueries-> rewrite / expand / negate the query
        run_search      -> the actual scoring (BM25 / dense / hybrid)
        postprocess     -> filter, dedup, rerank
        render_items    -> turn raw hits into EvidenceItem with `content`
- All long-tail customisation (negation expansion, field filters, VLM
  fallback, reranker injection, score blending) maps to one of the
  steps above.

This file intentionally has NO heavy imports so that it stays cheap
to import from configs and small utilities.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Retrieval "mode" expresses the *intent* of a query, not the algorithm.
# Implementations are free to honour, ignore, or extend these modes.
#   - "global"  : search with the question stem (broad recall)
#   - "option"  : search with question + a single option text
#   - "field"   : target a specific structured field (e.g. dividend per share)
#   - "counter" : look for evidence that *contradicts* a claim (negation)
#   - "vlm"     : route to a visual / table-image fallback retriever
#   - "custom"  : reserved escape hatch for teammates' new modes
RetrievalMode = Literal["global", "option", "field", "counter", "vlm", "custom"]

# Where a result came from. Teammates may add new tags; QA code should
# treat unknown sources as opaque strings.
SourceTag = str  # e.g. "bm25" | "dense" | "field" | "vlm" | "hybrid" | ...


class RetrievalFilters(TypedDict, total=False):
    """Soft + hard filters a QA agent may pass down.

    Implementations SHOULD honour ``doc_ids`` as a hard constraint when
    set (no routing fallback). All other keys are advisory.
    """

    doc_ids: List[str]                  # hard: only search inside these docs
    domains: List[str]                  # which corpus partitions to load
    companies: List[str]                # soft hint, e.g. ["宁德时代"]
    years: List[int]                    # soft hint, e.g. [2024]
    section_priors: List[str]           # boost matches in these section_paths
    unit_types: List[str]               # whitelist, e.g. ["field_card","table_row"]
    exclude_unit_types: List[str]       # blacklist, e.g. ["page_image_stub"]
    extras: Dict[str, Any]              # escape hatch for impl-specific flags


class EvidenceItem(TypedDict, total=False):
    """One unit of retrieved evidence as seen by the QA agent.

    Required-by-convention fields: ``evidence_id``, ``doc_id``, ``content``,
    ``score``, ``source``, ``query_labels``. Other fields are best-effort
    and may be absent depending on the underlying retriever.

    The ``content`` field MUST be prompt-ready text. The QA agent is
    expected to be able to concatenate ``content`` blocks directly into
    a prompt without further restructuring.
    """

    evidence_id: str                    # stable & unique within a run
    doc_id: str
    page: Optional[int]
    section_path: str
    unit_type: str                      # text_block / field_card / table_row / vlm_table / ...
    content: str                        # prompt-ready rendering
    structured: Dict[str, Any]          # raw anchor / value_map / row_header / ... for verifiers
    siblings: List[Dict[str, Any]]      # neighbouring units (HiKEY-style)
    score: float
    source: SourceTag
    query_labels: List[str]             # which sub-queries hit this item (e.g. ["global","option_A"])
    queries: List[str]                  # the actual query strings that hit


class EvidencePack(TypedDict, total=False):
    """Result of a single ``retrieve`` call or a ``merge`` of several."""

    query: str
    mode: RetrievalMode
    items: List[EvidenceItem]
    meta: Dict[str, Any]                # routed_docs, total_unique, dedup_count, latency_ms, ...


# ---------------------------------------------------------------------------
# Protocol (duck-typed contract)
# ---------------------------------------------------------------------------

@runtime_checkable
class Retriever(Protocol):
    """Minimal contract the QA agent depends on.

    Anything implementing these two methods can be plugged in. Implementations
    may add extra methods (warmup, reload, stats, ...) and the QA layer is
    free to detect and use them via ``hasattr``.
    """

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "global",
        topk: int = 10,
        filters: Optional[RetrievalFilters] = None,
        query_label: Optional[str] = None,
    ) -> EvidencePack: ...

    def merge(
        self,
        packs: Sequence[EvidencePack],
        *,
        final_topk: int,
    ) -> EvidencePack: ...


# ---------------------------------------------------------------------------
# Base class with extension hooks
# ---------------------------------------------------------------------------

# Type alias for a custom score blending function. Returns the final ranking
# score given the per-item state collected during merge.
ScoreFn = Callable[[Dict[str, Any]], float]


class RetrieverBase(ABC):
    """Optional base class that splits ``retrieve`` into overridable steps.

    Subclasses are expected to override at least ``run_search``. Everything
    else (query planning, doc routing, post-processing, content rendering)
    has a sensible default and can be customised independently.

    A subclass that wants to keep the m05/HiKEY behaviour intact should not
    need to touch any of these hooks. A subclass that wants to add a new
    capability (dense recall, reranker, VLM fallback, counter-query
    rewriting) typically only needs to override one or two of them.
    """

    # --- lifecycle --------------------------------------------------------

    def warmup(self, *, domains: Optional[List[str]] = None) -> None:
        """Load indices for the given domains. Default: no-op.

        Implementations that need eager loading (e.g. HiKEY manifests)
        should override this. The QA pipeline will call ``warmup`` once
        before processing questions.
        """

    # --- step 1: query understanding -------------------------------------

    def plan_query(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        filters: Optional[RetrievalFilters],
    ) -> Dict[str, Any]:
        """Parse the query into a lightweight plan.

        Default returns an empty dict. Override to extract companies,
        years, metrics, negation flags, etc. The plan is forwarded to
        ``run_search`` and ``postprocess_items``.
        """
        return {}

    # --- step 2: document routing ----------------------------------------

    def resolve_targets(
        self,
        query: str,
        plan: Dict[str, Any],
        filters: Optional[RetrievalFilters],
    ) -> List[str]:
        """Return the list of ``doc_id``s to actually search in.

        Default behaviour:
        - if ``filters['doc_ids']`` is provided, treat it as a hard
          constraint and return it as-is;
        - otherwise return an empty list, signalling "search everything
          loaded" (the concrete subclass decides what that means).
        """
        if filters and filters.get("doc_ids"):
            return list(filters["doc_ids"])
        return []

    # --- step 3: query rewriting / expansion -----------------------------

    def build_subqueries(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        plan: Dict[str, Any],
    ) -> List[str]:
        """Return one or more concrete query strings to dispatch to search.

        Default returns ``[query]``. Override to:
        - inject metric synonyms for ``mode="field"``;
        - flip a claim into its negation for ``mode="counter"``;
        - split a long question into multiple shorter probes.
        """
        return [query]

    # --- step 4: actual scoring (the only required hook) -----------------

    @abstractmethod
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
        """Run the scorer and return raw hits as plain dicts.

        Each raw hit should at minimum carry enough information for
        ``render_items`` to build an ``EvidenceItem``. Subclasses may
        attach any extra fields they want.
        """

    # --- step 5: post-processing -----------------------------------------

    def postprocess_items(
        self,
        raw_hits: List[Dict[str, Any]],
        *,
        mode: RetrievalMode,
        plan: Dict[str, Any],
        filters: Optional[RetrievalFilters],
    ) -> List[Dict[str, Any]]:
        """Filter / rerank / cap the raw hits before rendering.

        Default applies ``filters['unit_types']`` /
        ``filters['exclude_unit_types']`` if present, then sorts by
        ``score`` descending. Override to plug in a reranker, dedup
        rules, score normalisation, etc.
        """
        if filters:
            allow = set(filters.get("unit_types", []) or [])
            deny = set(filters.get("exclude_unit_types", []) or [])
            if allow:
                raw_hits = [h for h in raw_hits if h.get("unit_type") in allow
                            or h.get("object_type") in allow]
            if deny:
                raw_hits = [h for h in raw_hits if h.get("unit_type") not in deny
                            and h.get("object_type") not in deny]
        raw_hits.sort(key=lambda h: float(h.get("score", 0.0) or 0.0), reverse=True)
        return raw_hits

    # --- step 6: presentation --------------------------------------------

    def compute_evidence_id(self, raw_hit: Dict[str, Any]) -> str:
        """Derive a stable id for deduplication and traceability.

        Default: ``{doc_id}::{unit_id|field_id|hash}``. Override if your
        retriever has a better natural key (e.g. ``chunk_id``).
        """
        from hashlib import md5

        doc_id = raw_hit.get("doc_id") or raw_hit.get("source_doc_id") or ""
        natural = (
            raw_hit.get("unit_id")
            or raw_hit.get("field_id")
            or raw_hit.get("chunk_id")
            or md5(repr(sorted(raw_hit.items())).encode("utf-8")).hexdigest()[:16]
        )
        return f"{doc_id}::{natural}" if doc_id else str(natural)

    def render_content(self, raw_hit: Dict[str, Any]) -> str:
        """Render a prompt-ready text for a raw hit. Default: best-effort.

        QA agents will splice this string straight into prompts. Override
        to customise the exact textual representation (e.g. emit
        Markdown, JSON, or a domain-specific format).
        """
        text = raw_hit.get("text") or raw_hit.get("content") or ""
        if text:
            return str(text)
        # Field-card style fallback.
        metric = raw_hit.get("metric") or raw_hit.get("row_header") or ""
        value_map = raw_hit.get("value_map")
        unit = raw_hit.get("unit") or ""
        parts = []
        if metric:
            parts.append(f"指标: {metric}")
        if unit:
            parts.append(f"单位: {unit}")
        if value_map:
            import json as _json
            parts.append(f"数值: {_json.dumps(value_map, ensure_ascii=False)}")
        return "\n".join(parts)

    def render_items(
        self,
        raw_hits: List[Dict[str, Any]],
        *,
        mode: RetrievalMode,
        query: str,
        query_label: str,
    ) -> List[EvidenceItem]:
        """Convert raw hits into ``EvidenceItem``s.

        Default builds a generic mapping. Override if you want richer
        structured fields, alternative text rendering, etc.
        """
        items: List[EvidenceItem] = []
        for h in raw_hits:
            item: EvidenceItem = {
                "evidence_id": self.compute_evidence_id(h),
                "doc_id": h.get("doc_id") or h.get("source_doc_id") or "",
                "page": h.get("page") or h.get("source_page"),
                "section_path": h.get("section_path") or "",
                "unit_type": h.get("unit_type") or h.get("object_type") or "text_block",
                "content": self.render_content(h),
                "structured": h,
                "siblings": h.get("siblings", []) or [],
                "score": float(h.get("score", 0.0) or 0.0),
                "source": h.get("source") or self.default_source,
                "query_labels": [query_label] if query_label else [],
                "queries": [query] if query else [],
            }
            items.append(item)
        return items

    # --- step 7: orchestration -------------------------------------------

    @property
    def default_source(self) -> SourceTag:
        """Tag used when a raw hit does not carry its own ``source``."""
        return "unknown"

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "global",
        topk: int = 10,
        filters: Optional[RetrievalFilters] = None,
        query_label: Optional[str] = None,
    ) -> EvidencePack:
        """Default orchestration: plan -> route -> rewrite -> search -> render.

        Subclasses normally don't need to override this method. If you
        do, make sure the returned object satisfies ``EvidencePack``.
        """
        label = query_label or mode
        plan = self.plan_query(query, mode=mode, filters=filters)
        target_doc_ids = self.resolve_targets(query, plan, filters)
        subqueries = self.build_subqueries(query, mode=mode, plan=plan) or [query]

        raw_all: List[Dict[str, Any]] = []
        for sq in subqueries:
            hits = self.run_search(
                sq,
                mode=mode,
                topk=topk,
                target_doc_ids=target_doc_ids,
                plan=plan,
                filters=filters,
            )
            raw_all.extend(hits)

        raw_all = self.postprocess_items(
            raw_all, mode=mode, plan=plan, filters=filters
        )[:topk]

        items = self.render_items(
            raw_all, mode=mode, query=query, query_label=label
        )

        return {
            "query": query,
            "mode": mode,
            "items": items,
            "meta": {
                "routed_docs": target_doc_ids,
                "subqueries": subqueries,
                "raw_count": len(raw_all),
                "plan": plan,
            },
        }

    # --- merge -----------------------------------------------------------

    def merge_score_fn(self, item_state: Dict[str, Any]) -> float:
        """Final ranking score during ``merge``. Override to customise.

        ``item_state`` is the accumulated state of one merged item:
            {
                "best_score": float,
                "labels":     List[str],
                ...the merged EvidenceItem fields...
            }

        Default: best raw score + small bonus for being hit by multiple
        sub-queries (matches m05's ``_merge_evidence_packs`` behaviour).
        """
        labels = item_state.get("query_labels", []) or []
        return float(item_state.get("score", 0.0)) + 1.25 * max(0, len(labels) - 1)

    def merge(
        self,
        packs: Sequence[EvidencePack],
        *,
        final_topk: int,
    ) -> EvidencePack:
        """Deduplicate items across packs, accumulate ``query_labels``,
        rerank with ``merge_score_fn`` and trim to ``final_topk``.

        This default mirrors m05's ``_merge_evidence_packs`` semantics
        but operates on the new ``EvidenceItem`` shape.
        """
        merged: Dict[str, EvidenceItem] = {}
        for pack in packs:
            for item in pack.get("items", []) or []:
                key = item.get("evidence_id") or self.compute_evidence_id(
                    item.get("structured", {}) or {}
                )
                if key not in merged:
                    new_item = dict(item)
                    new_item["query_labels"] = list(item.get("query_labels", []) or [])
                    new_item["queries"] = list(item.get("queries", []) or [])
                    new_item["score"] = float(item.get("score", 0.0) or 0.0)
                    merged[key] = new_item  # type: ignore[assignment]
                else:
                    cur = merged[key]
                    for lbl in item.get("query_labels", []) or []:
                        if lbl not in cur["query_labels"]:
                            cur["query_labels"].append(lbl)
                    for q in item.get("queries", []) or []:
                        cur["queries"].append(q)
                    cur["score"] = max(
                        float(cur.get("score", 0.0) or 0.0),
                        float(item.get("score", 0.0) or 0.0),
                    )

        items = list(merged.values())
        items.sort(key=lambda it: self.merge_score_fn(dict(it)), reverse=True)

        return {
            "query": "+".join(p.get("query", "") for p in packs)[:200],
            "mode": "custom",
            "items": items[:final_topk],
            "meta": {
                "input_packs": len(packs),
                "total_unique": len(items),
                "kept": min(final_topk, len(items)),
            },
        }


__all__ = [
    "RetrievalMode",
    "SourceTag",
    "RetrievalFilters",
    "EvidenceItem",
    "EvidencePack",
    "Retriever",
    "RetrieverBase",
    "ScoreFn",
]
