"""
Embedding index builder and searcher.
Uses OpenRouter's text-embedding-3-large for semantic retrieval.
Caches embeddings to disk to avoid redundant API calls.
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL,
    EMBEDDING_DIM, EMBEDDING_BATCH_SIZE, EMBEDDING_CACHE_DIR
)
from methods._shared.indexer import Chunk


class EmbeddingIndex:
    """Vector index for semantic search using text-embedding-3-large."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (n_chunks, dim)
        self.doc_id_to_indices: dict = {}  # doc_id -> list of chunk indices
    
    def build(self, chunks: List[Chunk], domain: str, force_rebuild: bool = False):
        """
        Build embedding index for a list of chunks.
        Uses cached embeddings if available.
        
        Args:
            chunks: List of Chunk objects
            domain: Domain name (for cache file naming)
            force_rebuild: If True, ignore cache and rebuild
        """
        self.chunks = chunks
        
        # Build doc_id index
        self.doc_id_to_indices = {}
        for i, chunk in enumerate(chunks):
            if chunk.doc_id not in self.doc_id_to_indices:
                self.doc_id_to_indices[chunk.doc_id] = []
            self.doc_id_to_indices[chunk.doc_id].append(i)
        
        # Check cache
        cache_path = EMBEDDING_CACHE_DIR / f"{domain}_embeddings.npy"
        meta_path = EMBEDDING_CACHE_DIR / f"{domain}_meta.json"
        
        if not force_rebuild and cache_path.exists() and meta_path.exists():
            # Load from cache
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('n_chunks') == len(chunks):
                print(f"  Loading cached embeddings for {domain} ({len(chunks)} chunks)")
                self.embeddings = np.load(str(cache_path))
                return
        
        # Build embeddings via API
        print(f"  Building embeddings for {domain} ({len(chunks)} chunks)...")
        texts = [chunk.text for chunk in chunks]
        self.embeddings = self._batch_embed(texts)
        
        # Save to cache
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), self.embeddings)
        with open(meta_path, 'w') as f:
            json.dump({'n_chunks': len(chunks), 'dim': EMBEDDING_DIM, 'model': EMBEDDING_MODEL}, f)
        print(f"  Cached embeddings to {cache_path}")
    
    def search(self, query: str, top_k: int = 10,
               doc_ids: List[str] = None) -> List[Tuple[int, float]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            doc_ids: Optional list of doc_ids to restrict search
        
        Returns:
            List of (chunk_index, similarity_score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Embed query
        query_emb = self._embed_single(query)
        if query_emb is None:
            return []
        
        # Compute cosine similarity
        # Normalize embeddings for cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        chunk_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normalized_embeddings = self.embeddings / chunk_norms
        
        similarities = normalized_embeddings @ query_norm  # (n_chunks,)
        
        # Filter by doc_ids if specified
        if doc_ids:
            valid_indices = set()
            for did in doc_ids:
                if did in self.doc_id_to_indices:
                    valid_indices.update(self.doc_id_to_indices[did])
            
            # Mask out invalid indices
            mask = np.zeros(len(similarities), dtype=bool)
            for idx in valid_indices:
                mask[idx] = True
            similarities = similarities * mask
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
        
        return results
    
    def _embed_single(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text string."""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"  [WARNING] Embedding failed: {e}")
            return None
    
    def _batch_embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts with rate limiting and retry.
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM)
        """
        all_embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        batch_size = EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate very long texts to avoid token limits
            batch = [t[:2000] for t in batch]
            
            retries = 3
            for attempt in range(retries):
                try:
                    response = self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch,
                        encoding_format="float"
                    )
                    for j, item in enumerate(response.data):
                        all_embeddings[i + j] = np.array(item.embedding, dtype=np.float32)
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        print(f"  [RETRY] Embedding batch {i//batch_size} failed: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  [ERROR] Embedding batch {i//batch_size} failed after {retries} attempts: {e}")
            
            # Progress indicator
            if (i // batch_size) % 10 == 0:
                print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return all_embeddings
