from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".ipynb", ".py", ".md", ".txt"}


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _load_ipynb(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(_safe_read_text(path))
    cells = raw.get("cells", [])

    docs: list[dict[str, Any]] = []
    current_heading: str | None = None

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = str(source)

        text = text.strip()
        if not text:
            continue

        if cell_type == "markdown":
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("# "):
                    current_heading = line[2:].strip()
                elif line.startswith("## "):
                    current_heading = line[3:].strip()

        docs.append(
            {
                "source_file": str(path),
                "file_name": path.name,
                "file_type": "ipynb",
                "cell_index": idx,
                "cell_type": cell_type,
                "heading": current_heading,
                "text": text,
            }
        )

    return docs


def _load_py(path: Path) -> list[dict[str, Any]]:
    text = _safe_read_text(path).strip()
    if not text:
        return []

    return [
        {
            "source_file": str(path),
            "file_name": path.name,
            "file_type": "py",
            "cell_index": None,
            "cell_type": "code",
            "heading": None,
            "text": text,
        }
    ]


def _load_textlike(path: Path) -> list[dict[str, Any]]:
    text = _safe_read_text(path).strip()
    if not text:
        return []

    return [
        {
            "source_file": str(path),
            "file_name": path.name,
            "file_type": path.suffix.lstrip("."),
            "cell_index": None,
            "cell_type": "markdown",
            "heading": None,
            "text": text,
        }
    ]


def load_documents_from_path(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    documents: list[dict[str, Any]] = []

    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".ipynb":
            documents.extend(_load_ipynb(path))
        elif path.suffix.lower() == ".py":
            documents.extend(_load_py(path))
        else:
            documents.extend(_load_textlike(path))

    return documents