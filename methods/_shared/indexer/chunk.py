"""
Chunk data structure: a fixed-size piece of text with metadata.
Used by BM25 index, embedding index, hybrid retriever, etc.
"""
import re
from typing import Dict, List


class Chunk:
    """A text chunk with provenance metadata."""

    __slots__ = (
        "doc_id",
        "domain",
        "page_num",
        "section_title",
        "text",
        "chunk_id",
        "numbers",
        "keywords",
    )

    def __init__(
        self,
        doc_id: str,
        domain: str,
        page_num: int,
        section_title: str,
        text: str,
        chunk_id: int = 0,
    ):
        self.doc_id = doc_id
        self.domain = domain
        self.page_num = page_num
        self.section_title = section_title
        self.text = text
        self.chunk_id = chunk_id
        self.numbers = self._extract_numbers()
        self.keywords: List[str] = []

    def _extract_numbers(self) -> List[str]:
        patterns = [
            r"\d+\.?\d*%",
            r"\d+\.?\d*[万亿]?元",
            r"\d+\.?\d*[万亿]",
            r"\d{4}年",
            r"\d+个?[工作]*日",
            r"\d+个月",
            r"\d+\.?\d*",
        ]
        numbers = set()
        for pat in patterns:
            numbers.update(re.findall(pat, self.text))
        return list(numbers)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "domain": self.domain,
            "page_num": self.page_num,
            "section_title": self.section_title,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "numbers": self.numbers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        c = cls(
            doc_id=d["doc_id"],
            domain=d["domain"],
            page_num=d["page_num"],
            section_title=d["section_title"],
            text=d["text"],
            chunk_id=d.get("chunk_id", 0),
        )
        c.numbers = d.get("numbers", [])
        return c


def create_chunks(
    blocks: List[Dict],
    doc_id: str,
    domain: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Slice document blocks into roughly fixed-size chunks with overlap.

    Args:
        blocks: List of {text, page_num, section_title}.
        doc_id: Document identifier.
        domain: Domain name.
        chunk_size: Target characters per chunk.
        overlap: Overlap (in characters) between consecutive chunks.

    Returns:
        List[Chunk] with consecutive chunk_ids starting from 0.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    for block in blocks:
        text = block["text"]
        page_num = block["page_num"]
        section_title = block.get("section_title", "")

        if len(text) <= chunk_size:
            chunks.append(Chunk(doc_id, domain, page_num, section_title, text, chunk_id))
            chunk_id += 1
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(Chunk(doc_id, domain, page_num, section_title, text[start:end], chunk_id))
                chunk_id += 1
                start = end - overlap
                if start <= 0:
                    break

    return chunks
