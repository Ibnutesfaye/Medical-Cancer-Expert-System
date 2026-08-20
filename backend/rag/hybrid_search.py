"""
Hybrid Search — BM25 + FAISS + Cross-Encoder Reranker
=======================================================
Replaces simple FAISS-only retrieval with a three-stage pipeline:

Stage 1:  BM25 keyword search  (catches exact medical terms)
Stage 2:  FAISS semantic search (catches meaning-based matches)
Stage 3:  Cross-Encoder reranking (re-scores top candidates by relevance)

This combination significantly reduces hallucination and improves
recall for medical terminology (drug names, cancer subtypes, etc.)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class HybridMedicalSearch:
    """
    Hybrid retrieval combining BM25 + FAISS with cross-encoder reranking.

    Usage:
        search = HybridMedicalSearch()
        search.build_index(documents)          # list of {"text": ..., "metadata": ...}
        results = search.retrieve(query, top_k=5)
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self._embedding_model  = None
        self._bm25             = None
        self._faiss_index      = None
        self._reranker         = None
        self._documents: list[dict] = []
        self._built = False

    def _load_models(self):
        """Lazy-load heavy models only when first needed."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(
                    'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    max_length=512
                )
                print("  [HybridSearch] Cross-encoder reranker loaded ✓")
            except Exception as e:
                print(f"  [HybridSearch] Reranker unavailable: {e} — using score fallback")
                self._reranker = None

    def build_index(self, documents: list[dict]):
        """
        Build BM25 + FAISS indexes from document chunks.

        Args:
            documents: list of {"text": str, "metadata": dict}
        """
        if not documents:
            return

        self._documents = documents
        self._load_models()

        # BM25 index
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc["text"].lower().split() for doc in documents]
            self._bm25 = BM25Okapi(tokenized)
            print(f"  [HybridSearch] BM25 index built: {len(documents)} docs")
        except ImportError:
            print("  [HybridSearch] rank-bm25 not installed — BM25 disabled")
            self._bm25 = None

        # FAISS index
        try:
            import faiss
            texts = [doc["text"] for doc in documents]
            embeddings = self._embedding_model.encode(
                texts, batch_size=32, show_progress_bar=False
            )
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)   # inner product = cosine after normalize
            self._faiss_index.add(embeddings)
            print(f"  [HybridSearch] FAISS index built: {len(documents)} vectors, dim={dim}")
        except ImportError:
            print("  [HybridSearch] faiss not installed — FAISS disabled")
            self._faiss_index = None

        self._built = True

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-k most relevant documents using hybrid search.

        Returns:
            list of {"text": str, "metadata": dict, "score": float}
        """
        if not self._built or not self._documents:
            return []

        self._load_models()
        candidates: dict[int, float] = {}   # doc_idx → score

        # Stage 1: BM25 keyword retrieval
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            bm25_top_k  = int(top_k * 3)
            bm25_indices = np.argsort(bm25_scores)[-bm25_top_k:][::-1]
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            for idx in bm25_indices:
                candidates[int(idx)] = float(bm25_scores[idx]) / max_bm25 * 0.4

        # Stage 2: FAISS semantic retrieval
        if self._faiss_index is not None:
            q_vec = self._embedding_model.encode([query]).astype(np.float32)
            import faiss as _faiss
            _faiss.normalize_L2(q_vec)
            scores, indices = self._faiss_index.search(q_vec, top_k * 3)
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    candidates[int(idx)] = candidates.get(int(idx), 0) + float(score) * 0.6

        if not candidates:
            return []

        # Stage 3: Cross-encoder reranking on top candidates
        candidate_list = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        candidate_list = candidate_list[:top_k * 2]  # take top N for reranking

        if self._reranker is not None and len(candidate_list) > 0:
            pairs = [(query, self._documents[idx]["text"]) for idx, _ in candidate_list]
            rerank_scores = self._reranker.predict(pairs)
            reranked = sorted(
                zip(rerank_scores, [idx for idx, _ in candidate_list]),
                key=lambda x: x[0], reverse=True
            )
            final_indices = [idx for _, idx in reranked[:top_k]]
        else:
            final_indices = [idx for idx, _ in candidate_list[:top_k]]

        return [
            {
                "text":     self._documents[idx]["text"],
                "metadata": self._documents[idx].get("metadata", {}),
                "score":    float(candidates.get(idx, 0.0)),
            }
            for idx in final_indices
            if idx < len(self._documents)
        ]

    @property
    def is_ready(self) -> bool:
        return self._built


# Singleton
_hybrid_search: Optional[HybridMedicalSearch] = None


def get_hybrid_search() -> HybridMedicalSearch:
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridMedicalSearch()
    return _hybrid_search
