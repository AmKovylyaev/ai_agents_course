from __future__ import annotations

import re
from typing import Any


SECTION_PATTERNS = [
    ("imports", r"\b(import |from .+ import )"),
    ("preprocessing", r"\b(imputer|fillna|standardscaler|minmaxscaler|ordinalencoder|onehotencoder|columntransformer|pipeline)\b"),
    ("feature_engineering", r"\b(feature|encoding|transform|interaction|polynomial|extract|derive)\b"),
    ("training", r"\b(fit\(|randomforest|catboost|lightgbm|xgboost|logisticregression|linearregression|classifier|regressor)\b"),
    ("evaluation", r"\b(accuracy|f1|precision|recall|roc_auc|rmse|mae|r2_score|classification_report|confusion_matrix)\b"),
    ("inference", r"\b(predict\(|predict_proba\(|infer|inference)\b"),
    ("submission", r"\b(submission|sample_submission|to_csv)\b"),
]


def detect_section_type(text: str, fallback_cell_type: str = "") -> str:
    lowered = text.lower()

    for section_name, pattern in SECTION_PATTERNS:
        if re.search(pattern, lowered):
            return section_name

    if fallback_cell_type == "markdown":
        return "markdown"
    if fallback_cell_type == "code":
        return "code"

    return "misc"


def split_long_text(text: str, max_chars: int = 1600, overlap: int = 200) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_documents(documents: list[dict[str, Any]], max_chars: int = 1600) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for doc_id, doc in enumerate(documents):
        text = doc.get("text", "").strip()
        if not text:
            continue

        section_type = detect_section_type(text, fallback_cell_type=doc.get("cell_type", ""))
        text_chunks = split_long_text(text, max_chars=max_chars, overlap=200)

        for local_idx, chunk_text in enumerate(text_chunks):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{local_idx}",
                    "source_file": doc.get("source_file"),
                    "file_name": doc.get("file_name"),
                    "file_type": doc.get("file_type"),
                    "cell_index": doc.get("cell_index"),
                    "cell_type": doc.get("cell_type"),
                    "heading": doc.get("heading"),
                    "section_type": section_type,
                    "text": chunk_text,
                }
            )

    return chunks