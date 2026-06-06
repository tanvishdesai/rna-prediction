"""Hybrid RAG retrieval (dense + BM25 + RRF + cross-encoder) and Ollama generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_DIR = Path(__file__).parent / "data" / "index"
EMBED_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


class BioRAGPipeline:
    def __init__(self) -> None:
        chunks_path = INDEX_DIR / "chunks.json"
        index_path = INDEX_DIR / "bio_index.faiss"
        if not chunks_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "Index not found. Run: python scripts/download_data.py && python build_index.py"
            )
        with chunks_path.open(encoding="utf-8") as fh:
            self.chunks: List[Dict] = json.load(fh)
        self.texts = [c["text"] for c in self.chunks]
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(index_path))
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        self.reranker = CrossEncoder(RERANK_MODEL)

    def _dense_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _sparse_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query.split())
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    @staticmethod
    def _rrf(
        dense: List[Tuple[int, float]], sparse: List[Tuple[int, float]], k: int = RRF_K
    ) -> List[int]:
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        dense = self._dense_search(query, 20)
        sparse = self._sparse_search(query, 20)
        fused_ids = self._rrf(dense, sparse)[:20]
        candidates = [self.chunks[i]["text"] for i in fused_ids]
        pairs = [(query, p) for p in candidates]
        rerank_scores = self.reranker.predict(pairs)
        top_indices = np.argsort(rerank_scores)[::-1][:top_k]
        results = []
        for ri in top_indices:
            orig_idx = fused_ids[ri]
            c = self.chunks[orig_idx]
            results.append({
                "text": c["text"],
                "source": c["source"],
                "gene": c.get("gene", ""),
            })
        return results

    def generate(self, query: str, passages: List[Dict], use_ollama: bool = True) -> str:
        context = "\n\n".join(
            f"[{i+1}] ({p['source']}, {p.get('gene','')}) {p['text'][:600]}"
            for i, p in enumerate(passages)
        )
        prompt = f"""You are a bioinformatics assistant. Answer using ONLY the provided context.
Cite each claim with [source number].

Context:
{context}

Question: {query}
Answer:"""

        if use_ollama:
            try:
                result = subprocess.run(
                    ["ollama", "run", "mistral", prompt],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Fallback: extractive summary when Ollama unavailable
        return (
            f"Based on retrieved sources for '{query}':\n"
            + "\n".join(f"[{i+1}] {p['text'][:300]}…" for i, p in enumerate(passages))
        )

    def answer(self, query: str) -> Dict:
        passages = self.retrieve(query)
        answer = self.generate(query, passages)
        return {"answer": answer, "sources": passages}


_pipeline: BioRAGPipeline | None = None


def get_pipeline() -> BioRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BioRAGPipeline()
    return _pipeline
