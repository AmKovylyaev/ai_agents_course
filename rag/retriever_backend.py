from __future__ import annotations

import json
import pickle
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.utils import RAGConfig, RetrievedChunk, tokenize_code, rrf_rerank


class LocalSentenceTransformerEmbedder:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus: list[str] = []

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._chunk_ids = data["chunk_ids"]
        self._corpus = data["corpus"]

        if self._corpus:
            tokenized = [tokenize_code(text) for text in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []

        query_tokens = tokenize_code(query)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))
        return results


class FAISSVectorStore:
    def __init__(self):
        self._index = None
        self._id_map: list[str] = []

    def load(self, path: Path) -> None:
        self._index = faiss.read_index(str(path))
        id_map_path = path.with_suffix(".ids.json")
        with open(id_map_path, "r", encoding="utf-8") as f:
            self._id_map = json.load(f)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if self._index is None:
            return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        k = min(k, len(self._id_map))
        if k == 0:
            return []

        distances, indices = self._index.search(query, k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self._id_map):
                results.append((self._id_map[i], float(dist)))
        return results


class ChunkStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get(self, chunk_id: str) -> dict | None:
        cursor = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


class HybridRetriever:
    def __init__(
        self,
        config: RAGConfig,
        vector_store: FAISSVectorStore,
        bm25_index: BM25Index,
        chunk_store: ChunkStore,
        embedder: LocalSentenceTransformerEmbedder,
    ):
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunk_store = chunk_store
        self.embedder = embedder

    @classmethod
    def from_storage(cls, storage_path: str | Path, config: RAGConfig | None = None) -> "HybridRetriever":
        config = config or RAGConfig()
        storage_path = Path(storage_path)

        vector_store = FAISSVectorStore()
        faiss_path = storage_path / "faiss_index.bin"
        if faiss_path.exists():
            vector_store.load(faiss_path)

        bm25_index = BM25Index()
        bm25_path = storage_path / "bm25_index.pkl"
        if bm25_path.exists():
            bm25_index.load(bm25_path)

        chunk_store = ChunkStore(storage_path / "chunks.db")
        embedder = LocalSentenceTransformerEmbedder(config)

        return cls(config, vector_store, bm25_index, chunk_store, embedder)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        cell_type_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        k = k or self.config.top_k
        search_k = k * 3

        bm25_results = self.bm25_index.search(query, k=search_k)
        query_embedding = self.embedder.embed(query)
        semantic_results = self.vector_store.search(query_embedding, k=search_k)

        combined = rrf_rerank(bm25_results, semantic_results)
        results: list[RetrievedChunk] = []

        for chunk_id, rrf_score in combined:
            chunk = self.chunk_store.get(chunk_id)
            if chunk is None:
                continue

            if source_filter and source_filter not in (chunk.get("source_file") or ""):
                continue
            if cell_type_filter and chunk.get("cell_type") != cell_type_filter:
                continue
            if section_filter and chunk.get("section_type") != section_filter:
                continue

            bm25_score = next((s for cid, s in bm25_results if cid == chunk_id), 0.0)
            semantic_score = next((s for cid, s in semantic_results if cid == chunk_id), 0.0)

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.get("text", ""),
                    source_file=chunk.get("source_file", ""),
                    file_name=chunk.get("file_name", ""),
                    cell_index=chunk.get("cell_index"),
                    cell_type=chunk.get("cell_type", ""),
                    section_type=chunk.get("section_type", ""),
                    heading=chunk.get("heading"),
                    score=rrf_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                )
            )

            if len(results) >= k:
                break

        return results

    def semantic_search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.config.top_k
        query_embedding = self.embedder.embed(query)
        semantic_results = self.vector_store.search(query_embedding, k=k)

        results: list[RetrievedChunk] = []
        for chunk_id, score in semantic_results:
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.get("text", ""),
                    source_file=chunk.get("source_file", ""),
                    file_name=chunk.get("file_name", ""),
                    cell_index=chunk.get("cell_index"),
                    cell_type=chunk.get("cell_type", ""),
                    section_type=chunk.get("section_type", ""),
                    heading=chunk.get("heading"),
                    score=score,
                    semantic_score=score,
                )
            )
        return results

    def bm25_search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.config.top_k
        bm25_results = self.bm25_index.search(query, k=k)

        results: list[RetrievedChunk] = []
        for chunk_id, score in bm25_results:
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.get("text", ""),
                    source_file=chunk.get("source_file", ""),
                    file_name=chunk.get("file_name", ""),
                    cell_index=chunk.get("cell_index"),
                    cell_type=chunk.get("cell_type", ""),
                    section_type=chunk.get("section_type", ""),
                    heading=chunk.get("heading"),
                    score=score,
                    bm25_score=score,
                )
            )
        return results

    def format_for_prompt(
        self,
        results: list[RetrievedChunk],
        max_chunks: int = 5,
        max_code_len: int = 900,
    ) -> str:
        if not results:
            return "No relevant notebook chunks found."

        lines = ["=== RELEVANT NOTEBOOK EXAMPLES ===\n"]

        for i, r in enumerate(results[:max_chunks], 1):
            text = r.text
            if len(text) > max_code_len:
                text = text[:max_code_len] + "\n... (truncated)"

            header = f"[{r.file_name}"
            if r.heading:
                header += f" - {r.heading}"
            header += f"] (score={r.score:.4f}, bm25={r.bm25_score:.4f}, sem={r.semantic_score:.4f})"

            lines.append(f"### Example {i}: {header}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def close(self):
        self.chunk_store.close()