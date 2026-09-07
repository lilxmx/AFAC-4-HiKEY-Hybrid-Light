"""
Hybrid retriever combining BM25 and Embedding search using Reciprocal Rank Fusion (RRF).
Provides significantly better evidence recall than BM25 alone.
"""
from typing import List, Dict, Tuple

from config import (
    BM25_TOP_K, EMBEDDING_TOP_K, RRF_K, FINAL_TOP_K, MAX_EVIDENCE_CHARS
)
from methods._shared.indexer import BM25Index, Chunk
from embedding_index import EmbeddingIndex


class HybridRetriever:
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF) to combine
    BM25 keyword search with embedding semantic search.
    
    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each retrieval system i
    """
    
    def __init__(self, bm25_index: BM25Index, embedding_index: EmbeddingIndex):
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        # Build lookup dict for fast chunk index resolution
        self._chunk_lookup: Dict[tuple, int] = {}
        for i, c in enumerate(embedding_index.chunks):
            self._chunk_lookup[(c.doc_id, c.chunk_id)] = i
    
    def retrieve(self, question: str, options: Dict[str, str],
                 doc_ids: List[str] = None) -> List[Chunk]:
        """
        Retrieve evidence using hybrid BM25 + Embedding with RRF fusion.
        
        Strategy:
        1. BM25 search with question + each option
        2. Embedding search with question + each option
        3. RRF fusion of all results
        4. Return top-k fused results
        
        Args:
            question: Question text
            options: Dict of option key -> option text
            doc_ids: Optional list of doc_ids to restrict search
        
        Returns:
            List of evidence Chunk objects, ranked by RRF score
        """
        # Build multiple queries: question stem + question+option combinations
        queries = [question]
        for opt_key, opt_text in options.items():
            queries.append(f"{question} {opt_text}")
        
        # Collect BM25 rankings
        bm25_rankings = []  # List of ranked lists (each is list of chunk indices)
        for query in queries:
            results = self.bm25_index.search(query, top_k=BM25_TOP_K, doc_ids=doc_ids)
            # Convert to chunk indices
            ranked_indices = []
            for chunk, score in results:
                # Find chunk index in the index's chunk list
                idx = self._find_chunk_index(chunk)
                if idx is not None:
                    ranked_indices.append(idx)
            bm25_rankings.append(ranked_indices)
        
        # Collect Embedding rankings
        embedding_rankings = []
        for query in queries:
            results = self.embedding_index.search(query, top_k=EMBEDDING_TOP_K, doc_ids=doc_ids)
            ranked_indices = [idx for idx, score in results]
            embedding_rankings.append(ranked_indices)
        
        # Apply RRF fusion
        all_rankings = bm25_rankings + embedding_rankings
        fused_scores = self._rrf_fusion(all_rankings)
        
        # Sort by fused score and get top-k
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in sorted_chunks[:FINAL_TOP_K]]
        
        # Convert indices to Chunk objects
        evidence_chunks = []
        for idx in top_indices:
            if 0 <= idx < len(self.embedding_index.chunks):
                evidence_chunks.append(self.embedding_index.chunks[idx])
        
        # Truncate to max evidence length
        evidence_chunks = self._truncate_evidence(evidence_chunks, MAX_EVIDENCE_CHARS)
        
        return evidence_chunks
    
    def _find_chunk_index(self, chunk: Chunk) -> int:
        """Find the index of a chunk in the embedding index's chunk list (O(1) lookup)."""
        return self._chunk_lookup.get((chunk.doc_id, chunk.chunk_id))
    
    def _rrf_fusion(self, rankings: List[List[int]]) -> Dict[int, float]:
        """
        Apply Reciprocal Rank Fusion to multiple ranked lists.
        
        Args:
            rankings: List of ranked lists (each containing chunk indices)
        
        Returns:
            Dict mapping chunk_index -> fused RRF score
        """
        scores = {}
        
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                if idx not in scores:
                    scores[idx] = 0.0
                scores[idx] += 1.0 / (RRF_K + rank + 1)  # rank is 0-indexed
        
        return scores
    
    def _truncate_evidence(self, chunks: List[Chunk], max_chars: int) -> List[Chunk]:
        """Truncate evidence list to fit within max character limit."""
        result = []
        total_chars = 0
        
        for chunk in chunks:
            if total_chars + len(chunk.text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    truncated = Chunk(
                        doc_id=chunk.doc_id,
                        domain=chunk.domain,
                        page_num=chunk.page_num,
                        section_title=chunk.section_title,
                        text=chunk.text[:remaining],
                        chunk_id=chunk.chunk_id,
                    )
                    result.append(truncated)
                break
            result.append(chunk)
            total_chars += len(chunk.text)
        
        return result


def format_evidence(chunks: List[Chunk]) -> str:
    """Format evidence chunks into a readable string for the LLM."""
    if not chunks:
        return "（未找到相关证据）"
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[证据{i}] 文档: {chunk.doc_id}"
        if chunk.section_title:
            header += f" | 章节: {chunk.section_title}"
        if chunk.page_num > 0:
            header += f" | 页码: {chunk.page_num}"
        parts.append(f"{header}\n{chunk.text}")
    
    return "\n\n---\n\n".join(parts)
