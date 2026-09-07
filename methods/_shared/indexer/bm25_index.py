"""
BM25 index over a list of Chunk objects.
Uses jieba for Chinese tokenization and rank_bm25.BM25Okapi for scoring.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jieba
from rank_bm25 import BM25Okapi

from .chunk import Chunk


class BM25Index:
    """BM25 index over a chunk corpus, with auxiliary doc_id / number lookup."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.doc_id_to_chunks: Dict[str, List[int]] = {}
        self.number_to_chunks: Dict[str, List[int]] = {}

    # ---------- build / load / save ----------
    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = []
        self.doc_id_to_chunks = {}
        self.number_to_chunks = {}

        for i, chunk in enumerate(chunks):
            self.tokenized_corpus.append(list(jieba.cut(chunk.text)))
            self.doc_id_to_chunks.setdefault(chunk.doc_id, []).append(i)
            for num in chunk.numbers:
                self.number_to_chunks.setdefault(num, []).append(i)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self, path: Path) -> None:
        data = {
            "chunks": [c.to_dict() for c in self.chunks],
            "doc_id_to_chunks": self.doc_id_to_chunks,
            "number_to_chunks": self.number_to_chunks,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = [Chunk.from_dict(d) for d in data["chunks"]]
        self.doc_id_to_chunks = data["doc_id_to_chunks"]
        self.number_to_chunks = data["number_to_chunks"]
        self.tokenized_corpus = [list(jieba.cut(c.text)) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ---------- search ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """BM25 search, optionally restricted to a list of doc_ids."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)

        if doc_ids:
            valid = set()
            for did in doc_ids:
                valid.update(self.doc_id_to_chunks.get(did, []))
            for i in range(len(scores)):
                if i not in valid:
                    scores[i] = 0.0

        scored = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], score) for i, score in scored[:top_k]]

    def search_by_number(
        self,
        number: str,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Chunk]:
        if number not in self.number_to_chunks:
            return []
        indices = self.number_to_chunks[number]
        if doc_ids:
            valid = set()
            for did in doc_ids:
                valid.update(self.doc_id_to_chunks.get(did, []))
            indices = [i for i in indices if i in valid]
        return [self.chunks[i] for i in indices]
