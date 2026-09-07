"""
m02_gpt41_golden - end-to-end pipeline.

Subclasses BaseRunner so loading questions, concurrent inference,
resume, and result writing all come from _shared.

m02-specific responsibilities kept here:
  - Load BM25 indices (built by m01) + build embedding indices.
  - Hybrid retrieval (BM25 + embedding via RRF fusion).
  - Self-consistency voting reasoning with GPT-4.1.
  - FC specialized pipeline (Skill v6.0).
"""
import logging
from typing import Dict, List, Optional

from methods._shared.indexer import BM25Index
from methods._shared.pipeline import BaseRunner

from config import (
    BASELINE_INDICES_DIR,
    CONCURRENCY,
    DOMAINS,
    LOGS_DIR,
    METHOD_NAME,
    OUTPUT_DIR,
    REASONING_MODEL,
    VOTING_ROUNDS,
)
from embedding_index import EmbeddingIndex
from fc_pipeline import FCPipeline
from hybrid_retriever import HybridRetriever, format_evidence
from reasoner import GoldenReasoner

logger = logging.getLogger(__name__)


class GoldenPipeline(BaseRunner):
    """Hybrid BM25+Embedding retrieval + GPT-4.1 voting reasoning."""

    method_name = METHOD_NAME
    model_name = REASONING_MODEL

    def __init__(self):
        super().__init__(
            output_dir=OUTPUT_DIR,
            logs_dir=LOGS_DIR,
            concurrency=CONCURRENCY,
            save_evidence=True,
        )
        self.bm25_indices: Dict[str, BM25Index] = {}
        self.embedding_indices: Dict[str, EmbeddingIndex] = {}
        self.retrievers: Dict[str, HybridRetriever] = {}
        self.reasoner: Optional[GoldenReasoner] = None
        self.fc_pipeline: Optional[FCPipeline] = None

    # ============================================================
    # BaseRunner.setup hook (load indices + initialize agents)
    # ============================================================
    def setup(self, domains: Optional[List[str]] = None) -> None:
        domains = domains or DOMAINS
        logger.info("=" * 60)
        logger.info("Setting up m02 Golden Label Pipeline")
        logger.info("=" * 60)

        for domain in domains:
            # 1) Load BM25 index (built by m01 / shared)
            bm25_path = BASELINE_INDICES_DIR / f"{domain}_index.pkl"
            if not bm25_path.exists():
                logger.error(
                    f"  BM25 index not found: {bm25_path}. "
                    f"Run m01 preprocessing first or build it manually."
                )
                continue
            bm25 = BM25Index()
            bm25.load(bm25_path)
            self.bm25_indices[domain] = bm25
            logger.info(f"[{domain}] BM25 loaded: {len(bm25.chunks)} chunks")

            # 2) Build / load embedding index (cached on disk)
            emb = EmbeddingIndex()
            emb.build(bm25.chunks, domain)
            self.embedding_indices[domain] = emb
            logger.info(f"[{domain}] Embedding ready: {len(emb.chunks)} chunks")

            # 3) Hybrid retriever
            self.retrievers[domain] = HybridRetriever(bm25, emb)

            # 4) FC specialized pipeline
            if domain == "financial_contracts":
                self.fc_pipeline = FCPipeline()
                self.fc_pipeline.setup(bm25, emb)
                logger.info("[FC] Specialized Skill v6.0 pipeline initialized")

        # Reasoner is shared across non-FC domains
        if self.reasoner is None:
            self.reasoner = GoldenReasoner()
        logger.info("Setup complete.")

    # ============================================================
    # Per-question processor
    # ============================================================
    def process_question(self, q: Dict) -> Dict:
        qid = q["qid"]
        domain = q["domain"]
        question = q["question"]
        options = q["options"]
        answer_format = q["answer_format"]
        doc_ids = q.get("doc_ids", [])

        # ---- FC specialized pipeline ----
        if domain == "financial_contracts" and self.fc_pipeline is not None:
            try:
                fc_result = self.fc_pipeline.process_question(q)
                # FC pipeline doesn't expose token usage; ensure required keys exist.
                fc_result.setdefault("prompt_tokens", 0)
                fc_result.setdefault("completion_tokens", 0)
                fc_result.setdefault("total_tokens", 0)
                fc_result.setdefault("raw_answer", fc_result.get("reasoning", ""))
                return fc_result
            except Exception as e:
                logger.error(f"[FC] failed for {qid}: {e}, falling back to generic")

        # ---- Generic hybrid pipeline ----
        retriever = self.retrievers.get(domain)
        if retriever is None:
            return {
                "qid": qid, "domain": domain,
                "answer": "A", "raw_answer": "no retriever",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "evidence": [], "confidence": 0,
            }

        evidence_chunks = retriever.retrieve(
            question=question,
            options=options,
            doc_ids=doc_ids if doc_ids else None,
        )
        evidence_text = format_evidence(evidence_chunks)

        result = self.reasoner.reason_with_voting(
            question=question,
            options=options,
            evidence=evidence_text,
            answer_format=answer_format,
            domain=domain,
            voting_rounds=VOTING_ROUNDS,
        )

        evidence_info = [{
            "doc_id": c.doc_id,
            "page_num": c.page_num,
            "section_title": c.section_title,
            "text_snippet": c.text[:300],
        } for c in evidence_chunks]

        return {
            "qid": qid,
            "domain": domain,
            "answer_format": answer_format,
            "answer": result["answer"],
            "confidence": result.get("confidence", 0),
            "all_votes": result.get("all_votes", []),
            "evidence": evidence_info,
            "reasoning": (result.get("reasoning") or "")[:2000],
            # GoldenReasoner doesn't yet aggregate token usage across votes.
            # Keep fields present so BaseRunner / output writers don't crash.
            "raw_answer": (result.get("reasoning") or "")[:1000],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
