from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k: int = 5
    threshold: float = 0.0
    storage_path: str = "./artifacts/rag/notebook_index"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    file_name: str
    cell_index: int | None
    cell_type: str
    section_type: str
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "file_name": self.file_name,
            "cell_index": self.cell_index,
            "cell_type": self.cell_type,
            "section_type": self.section_type,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "semantic_score": self.semantic_score,
            "heading": self.heading,
        }


def tokenize_code(text: str) -> list[str]:
    tokens = re.split(r"[^a-zA-Z0-9]+", text)
    result: list[str] = []

    for token in tokens:
        if not token:
            continue

        camel_split = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", token)
        if camel_split:
            result.extend(camel_split)
        else:
            result.append(token)

    return [t.lower() for t in result if t]


def rrf_rerank(
    bm25_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)

    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] += 1 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(semantic_results):
        scores[chunk_id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)