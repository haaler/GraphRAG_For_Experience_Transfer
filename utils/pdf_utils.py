from __future__ import annotations

import os
from typing import List, Dict

try:
    from pypdf import PdfReader
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'pypdf'. Please install it (pip install pypdf)."
    ) from e

from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_text_by_page(path: str) -> List[str]:
    """
    Extract raw text from a PDF file, one string per page.

    Returns an empty list if the file does not exist or has no extractable text.
    """
    if not os.path.isfile(path):
        return []

    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return pages


def chunk_pages(
    pages: List[str], *, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Dict[str, object]]:
    """
    Split each page's text into overlapping chunks suitable for embedding.

    Returns a list of dicts: {"page": int, "idx": int, "text": str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    all_chunks: List[Dict[str, object]] = []
    for i, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        parts = splitter.split_text(page_text)
        for j, part in enumerate(parts):
            if not part.strip():
                continue
            all_chunks.append({"page": i, "idx": j, "text": part})
    return all_chunks

