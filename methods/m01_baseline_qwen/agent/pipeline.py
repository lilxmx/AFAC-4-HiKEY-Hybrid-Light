"""
m01_baseline_qwen - end-to-end pipeline.

Subclasses BaseRunner so that loading questions, concurrent inference,
resume support, and result writing all come from _shared.

m01-specific responsibilities kept here:
  - Build BM25 indices from raw documents (offline preprocessing).
  - Wire up the FC specialized pipeline for financial_contracts.
  - Per-question retrieve + reason logic for non-FC domains.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from methods._shared.indexer import BM25Index, Chunk, create_chunks
from methods._shared.parsers import parse_document_blocks
from methods._shared.pipeline import BaseRunner

from .config import (
    BM25_TOP_K,
    CONCURRENCY,
    DOMAINS,
    INDICES_DIR,
    LOGS_DIR,
    METHOD_NAME,
    MODEL_NAME,
    OUTPUT_DIR,
    QUESTION_FILES,
)
from .fc_pipeline import FCPipeline
from .reasoner import Reasoner
from .retriever import Retriever, format_evidence

logger = logging.getLogger(__name__)


class Pipeline(BaseRunner):
    """End-to-end pipeline for AFAC2026 Task4 baseline (m01)."""

    method_name = METHOD_NAME
    model_name = MODEL_NAME

    def __init__(self):
        super().__init__(
            output_dir=OUTPUT_DIR,
            logs_dir=LOGS_DIR,
            concurrency=CONCURRENCY,
            save_evidence=True,
        )
        self.indices: Dict[str, BM25Index] = {}
        self.reasoner: Optional[Reasoner] = None
        self.fc_pipeline: Optional[FCPipeline] = None

    # ============================================================
    # Index building (offline preprocessing - no token cost)
    # ============================================================
    def build_indices(self, domains: Optional[List[str]] = None) -> None:
        """Parse all documents in `domains` and build per-domain BM25 indices."""
        domains = domains or DOMAINS
        for domain in domains:
            logger.info(f"Building BM25 index for domain: {domain}")
            index = self._build_domain_index_parallel(domain)
            self.indices[domain] = index
            index_path = INDICES_DIR / f"{domain}_index.pkl"
            index.save(index_path)
            logger.info(f"  Saved index: {index_path} ({len(index.chunks)} chunks)")

    def load_indices(self, domains: Optional[List[str]] = None) -> None:
        """Load BM25 indices from disk; build the missing ones."""
        domains = domains or DOMAINS
        for domain in domains:
            index_path = INDICES_DIR / f"{domain}_index.pkl"
            if index_path.exists():
                logger.info(f"Loading BM25 index for {domain}")
                idx = BM25Index()
                idx.load(index_path)
                self.indices[domain] = idx
                logger.info(f"  Loaded {len(idx.chunks)} chunks")
            else:
                logger.warning(f"  Index not found: {index_path}, building...")
                idx = self._build_domain_index_parallel(domain)
                idx.save(index_path)
                self.indices[domain] = idx

    def _build_domain_index_parallel(self, domain: str) -> BM25Index:
        questions = self._load_questions(domain)
        needed_doc_ids = sorted({d for q in questions for d in q.get("doc_ids", [])})
        logger.info(f"  Domain {domain}: {len(needed_doc_ids)} unique doc_ids")

        all_chunks: List[Chunk] = []
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(needed_doc_ids)))) as pool:
            futures = {pool.submit(self._parse_one_doc, d, domain): d for d in needed_doc_ids}
            for fut in as_completed(futures):
                doc_id = futures[fut]
                try:
                    all_chunks.extend(fut.result())
                except Exception as e:
                    logger.error(f"  Failed to parse {doc_id}: {e}")

        logger.info(f"  Total chunks for {domain}: {len(all_chunks)}")
        idx = BM25Index()
        idx.build(all_chunks)
        return idx

    @staticmethod
    def _parse_one_doc(doc_id: str, domain: str) -> List[Chunk]:
        blocks = parse_document_blocks(doc_id, domain)
        if not blocks:
            return []
        return create_chunks(blocks, doc_id, domain)

    @staticmethod
    def _load_questions(domain: str) -> List[Dict]:
        import json
        qfile = QUESTION_FILES.get(domain)
        if qfile is None or not qfile.exists():
            return []
        with open(qfile, "r", encoding="utf-8") as f:
            return json.load(f)

    # ============================================================
    # BaseRunner hooks
    # ============================================================
    def setup(self, domains: List[str]) -> None:
        # Make sure indices are available (load from disk or build).
        self.load_indices(domains)

        # Initialize reasoner once.
        if self.reasoner is None:
            self.reasoner = Reasoner()

        # Specialized FC pipeline if needed.
        if "financial_contracts" in domains and "financial_contracts" in self.indices:
            self.fc_pipeline = FCPipeline()
            self.fc_pipeline.setup(self.indices["financial_contracts"])
            logger.info("Initialized specialized FC pipeline (Skill v6.0)")

    def process_question(self, q: Dict) -> Dict:
        qid = q["qid"]
        domain = q["domain"]
        question = q["question"]
        options = q["options"]
        answer_format = q["answer_format"]
        doc_ids = q.get("doc_ids", [])

        index = self.indices.get(domain)
        if index is None:
            return {
                "qid": qid, "domain": domain,
                "answer": "A", "raw_answer": "NO INDEX",
                "evidence": [], "evidence_text": "",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }

        # ---- Specialized FC pipeline ----
        if domain == "financial_contracts" and self.fc_pipeline is not None:
            try:
                fc_result = self.fc_pipeline.process_question(q)
                # FC pipeline doesn't expose token usage; ensure required keys exist
                fc_result.setdefault("prompt_tokens", 0)
                fc_result.setdefault("completion_tokens", 0)
                fc_result.setdefault("total_tokens", 0)
                fc_result.setdefault("raw_answer", fc_result.get("reasoning", ""))
                return fc_result
            except Exception as e:
                logger.error(f"FC pipeline failed for {qid}: {e}, falling back to generic pipeline")

        # ---- Generic pipeline ----
        retriever = Retriever(index)
        evidence_chunks = retriever.retrieve(
            question=question,
            options=options,
            doc_ids=doc_ids if doc_ids else None,
            domain=domain,
        )
        evidence_text = format_evidence(evidence_chunks)
        result = self.reasoner.reason(
            question=question,
            options=options,
            evidence=evidence_text,
            answer_format=answer_format,
        )
        evidence_info = [{
            "doc_id": c.doc_id,
            "page_num": c.page_num,
            "section_title": c.section_title,
            "text_snippet": c.text[:200],
        } for c in evidence_chunks]

        return {
            "qid": qid,
            "domain": domain,
            "answer": result["answer"],
            "raw_answer": result["raw_answer"],
            "evidence": evidence_info,
            "evidence_text": evidence_text,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        }
