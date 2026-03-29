from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.notebook_loader import load_documents_from_path
from rag.notebook_chunker import chunk_documents
from rag.utils import RAGConfig, tokenize_code

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbedder:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [np.array(v, dtype=np.float32) for v in vectors]


class RAGIndexer:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: list[np.ndarray] = []

    def _get_embedder(self) -> LocalSentenceTransformerEmbedder:
        return LocalSentenceTransformerEmbedder(self.config)

    def generate_embeddings(self, chunks: list[dict[str, Any]]) -> list[np.ndarray]:
        embedder = self._get_embedder()
        texts = [chunk["text"] for chunk in chunks]
        logger.info("Generating embeddings for %s chunks...", len(texts))
        return embedder.embed_many(texts)

    def build_bm25_index(self, chunks: list[dict[str, Any]]) -> BM25Okapi:
        corpus = [tokenize_code(chunk["text"]) for chunk in chunks]
        return BM25Okapi(corpus)

    def build_faiss_index(self, embeddings: list[np.ndarray]):
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings_matrix)
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)
        return index

    def save(self, storage_path: str | Path) -> None:
        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        if self.embeddings:
            faiss_index = self.build_faiss_index(self.embeddings)
            faiss.write_index(faiss_index, str(storage / "faiss_index.bin"))

            id_map = [chunk["chunk_id"] for chunk in self.chunks]
            with open(storage / "faiss_index.ids.json", "w", encoding="utf-8") as f:
                json.dump(id_map, f, ensure_ascii=False, indent=2)

        if self.chunks:
            bm25_data = {
                "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
                "corpus": [chunk["text"] for chunk in self.chunks],
            }
            with open(storage / "bm25_index.pkl", "wb") as f:
                pickle.dump(bm25_data, f)

        db_path = storage / "chunks.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT,
                source_file TEXT,
                file_name TEXT,
                cell_index INTEGER,
                cell_type TEXT,
                heading TEXT,
                section_type TEXT
            )
            """
        )

        for chunk in self.chunks:
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks
                (id, text, source_file, file_name, cell_index, cell_type, heading, section_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["source_file"],
                    chunk["file_name"],
                    chunk["cell_index"],
                    chunk["cell_type"],
                    chunk.get("heading"),
                    chunk["section_type"],
                ),
            )

        conn.commit()
        conn.close()
        logger.info("Saved %s chunks to %s", len(self.chunks), storage)

    def index_notebooks(self, notebooks_root: str | Path, max_chars: int = 1600) -> int:
        docs = load_documents_from_path(notebooks_root)
        chunks = chunk_documents(docs, max_chars=max_chars)

        if not chunks:
            logger.info("No chunks to index")
            return 0

        embeddings = self.generate_embeddings(chunks)
        self.chunks = chunks
        self.embeddings = embeddings
        return len(chunks)