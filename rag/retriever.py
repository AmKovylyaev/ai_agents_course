from __future__ import annotations

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity


def retrieve_top_k(
    query: str,
    index: dict[str, Any],
    top_k: int = 5,
    section_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k chunks by cosine similarity.
    """
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks = index["chunks"]

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).ravel()

    results: list[dict[str, Any]] = []
    ranked_indices = sims.argsort()[::-1]

    for idx in ranked_indices:
        score = float(sims[idx])
        chunk = chunks[idx]

        if section_filter and chunk.get("section_type") != section_filter:
            continue

        enriched = dict(chunk)
        enriched["score"] = score
        results.append(enriched)

        if len(results) >= top_k:
            break

    return results