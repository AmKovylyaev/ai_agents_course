from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.indexer import RAGIndexer
from rag.retriever_backend import HybridRetriever
from rag.utils import RAGConfig

import re

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"disregard .* instructions",
    r"system prompt",
    r"you are chatgpt",
    r"override .* rules",
]

DANGEROUS_CODE_PATTERNS = [
    r"os\.system\(",
    r"subprocess\.",
    r"eval\(",
    r"exec\(",
    r"socket\.",
    r"requests\.(post|get)\(",
]

def is_safe_retrieved_chunk(text: str) -> bool:
    lowered = text.lower()

    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return False

    for pat in DANGEROUS_CODE_PATTERNS:
        if re.search(pat, lowered):
            return False

    return True


def sanitize_retrieved_results(results: list[dict], min_score: float = 0.0) -> list[dict]:
    safe_results = []

    for r in results:
        text = r.get("text", "")
        score = float(r.get("score", 0.0))

        if score < min_score:
            continue
        if not is_safe_retrieved_chunk(text):
            continue

        safe_results.append(r)

    return safe_results

def build_notebook_rag_index(
    notebooks_root: str | Path,
    index_dir: str | Path,
    max_chars: int = 1600,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    config = RAGConfig(
        embedding_model=embedding_model,
        embedding_dim=384,
        storage_path=str(index_dir),
    )
    indexer = RAGIndexer(config)
    total_chunks = indexer.index_notebooks(notebooks_root=notebooks_root, max_chars=max_chars)

    if total_chunks > 0:
        indexer.save(index_dir)

    return {
        "notebooks_root": str(notebooks_root),
        "index_dir": str(index_dir),
        "n_chunks": total_chunks,
        "embedding_model": embedding_model,
    }


def search_notebooks_tool(
    query: str,
    index_dir: str | Path,
    top_k: int = 5,
    search_type: str = "hybrid",
    source_filter: str | None = None,
    cell_type_filter: str | None = None,
    section_filter: str | None = None,
) -> list[dict[str, Any]]:
    retriever = HybridRetriever.from_storage(index_dir)

    try:
        if search_type == "semantic":
            results = retriever.semantic_search(query=query, k=top_k)
        elif search_type == "bm25":
            results = retriever.bm25_search(query=query, k=top_k)
        else:
            results = retriever.retrieve(
                query=query,
                k=top_k,
                source_filter=source_filter,
                cell_type_filter=cell_type_filter,
                section_filter=section_filter,
            )
        return [r.to_dict() for r in results]
    finally:
        retriever.close()


def format_retrieved_chunks_for_prompt(
    index_dir: str | Path,
    results: list[dict[str, Any]],
    max_chunks: int = 5,
    max_code_len: int = 900,
) -> str:
    retriever = HybridRetriever.from_storage(index_dir)
    try:
        from rag.utils import RetrievedChunk

        typed_results = [RetrievedChunk(**r) for r in results]
        return retriever.format_for_prompt(
            results=typed_results,
            max_chunks=max_chunks,
            max_code_len=max_code_len,
        )
    finally:
        retriever.close()


def inject_rag_context_into_state(
    state: dict[str, Any],
    query: str,
    top_k: int = 5,
    search_type: str = "hybrid",
    source_filter: str | None = None,
    cell_type_filter: str | None = None,
    section_filter: str | None = None,
) -> dict[str, Any]:
    state = dict(state)
    index_dir = state.get("rag_index_dir", "")
    if not index_dir:
        state["rag_context"] = "RAG index is not configured."
        state["rag_results"] = []
        return state

    results = search_notebooks_tool(
        query=query,
        index_dir=index_dir,
        top_k=top_k,
        search_type=search_type,
        source_filter=source_filter,
        cell_type_filter=cell_type_filter,
        section_filter=section_filter,
    )

    prompt_context = format_retrieved_chunks_for_prompt(
        index_dir=index_dir,
        results=results,
        max_chunks=top_k,
        max_code_len=900,
    )

    state["rag_results"] = results
    state["rag_context"] = prompt_context
    state["rag_query"] = query
    state["rag_search_type"] = search_type
    return state