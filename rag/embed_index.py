from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def build_tfidf_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a lightweight retriever index using TF-IDF.
    """
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        raise ValueError("No chunks provided for indexing")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(texts)

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }


def save_index(index: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(index["vectorizer"], out_dir / "vectorizer.joblib")
    joblib.dump(index["matrix"], out_dir / "matrix.joblib")

    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(index["chunks"], f, ensure_ascii=False, indent=2)


def load_index(index_dir: str | Path) -> dict[str, Any]:
    index_dir = Path(index_dir)

    vectorizer = joblib.load(index_dir / "vectorizer.joblib")
    matrix = joblib.load(index_dir / "matrix.joblib")

    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }