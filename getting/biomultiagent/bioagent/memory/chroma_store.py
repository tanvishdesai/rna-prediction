"""ChromaDB session memory for multi-turn bioinformatics queries."""

from __future__ import annotations

import time
from typing import List

import chromadb


class SessionMemory:
    def __init__(self, persist_dir: str = "./chroma_data") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection("bio_session_memory")

    def store(self, session_id: str, query: str, result: str) -> None:
        doc_id = f"{session_id}_{int(time.time() * 1000)}"
        self._collection.add(
            documents=[f"Q: {query}\nA: {result}"],
            ids=[doc_id],
            metadatas=[{"session_id": session_id}],
        )

    def retrieve(self, session_id: str, query: str, n: int = 3) -> List[str]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._collection.count()),
            where={"session_id": session_id},
        )
        return results.get("documents", [[]])[0]
