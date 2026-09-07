"""
Evidence retriever that combines BM25 search with number matching.
Implements option-level retrieval strategy.
"""
import re
from typing import List, Dict, Tuple

from methods._shared.indexer import BM25Index, Chunk
from .config import BM25_TOP_K, MAX_EVIDENCE_CHARS


class Retriever:
    """Retrieves relevant evidence chunks for a given question."""
    
    def __init__(self, index: BM25Index):
        self.index = index
    
    def retrieve(self, question: str, options: Dict[str, str],
                 doc_ids: List[str] = None, domain: str = None,
                 top_k: int = BM25_TOP_K) -> List[Chunk]:
        """
        Retrieve evidence using option-level retrieval strategy.
        
        Args:
            question: Question text
            options: Dict of option key -> option text
            doc_ids: List of doc_ids to restrict search
            domain: Domain name
            top_k: Top-k per query
        
        Returns:
            Deduplicated and ranked list of evidence chunks
        """
        all_results: List[Tuple[Chunk, float]] = []
        
        # 1. Search with question stem
        stem_results = self.index.search(question, top_k=top_k, doc_ids=doc_ids)
        all_results.extend(stem_results)
        
        # 2. Search with each option (option-level retrieval)
        for opt_key, opt_text in options.items():
            query = question + " " + opt_text
            opt_results = self.index.search(query, top_k=max(2, top_k // 2), doc_ids=doc_ids)
            all_results.extend(opt_results)
        
        # 3. Number-based retrieval supplement
        numbers = self._extract_query_numbers(question, options)
        for num in numbers[:5]:  # Limit to top 5 numbers
            num_chunks = self.index.search_by_number(num, doc_ids=doc_ids)
            for chunk in num_chunks[:2]:
                all_results.append((chunk, 1.0))  # Give a base score
        
        # 4. Deduplicate and rank
        evidence = self._deduplicate_and_rank(all_results)
        
        # 5. Truncate to max evidence length
        evidence = self._truncate_evidence(evidence, MAX_EVIDENCE_CHARS)
        
        return evidence
    
    def _extract_query_numbers(self, question: str, options: Dict[str, str]) -> List[str]:
        """Extract significant numbers from question and options."""
        combined = question + " " + " ".join(options.values())
        patterns = [
            r'\d+\.?\d*%',
            r'\d+\.?\d*[万亿]?元',
            r'\d+\.?\d*[万亿]',
            r'\d+个?[工作]*日',
            r'\d+个月',
        ]
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, combined)
            numbers.extend(matches)
        return list(set(numbers))
    
    def _deduplicate_and_rank(self, results: List[Tuple[Chunk, float]]) -> List[Chunk]:
        """Deduplicate chunks and rank by cumulative score."""
        # Use (doc_id, chunk_id) as unique key
        seen = {}
        for chunk, score in results:
            key = (chunk.doc_id, chunk.chunk_id)
            if key not in seen:
                seen[key] = (chunk, score)
            else:
                # Accumulate score for duplicates
                existing_chunk, existing_score = seen[key]
                seen[key] = (existing_chunk, existing_score + score)
        
        # Sort by cumulative score
        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in ranked]
    
    def _truncate_evidence(self, chunks: List[Chunk], max_chars: int) -> List[Chunk]:
        """Truncate evidence list to fit within max character limit."""
        result = []
        total_chars = 0
        
        for chunk in chunks:
            if total_chars + len(chunk.text) > max_chars:
                # Include partial chunk if we have room
                remaining = max_chars - total_chars
                if remaining > 200:  # Only include if meaningful
                    # Create a truncated copy
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
