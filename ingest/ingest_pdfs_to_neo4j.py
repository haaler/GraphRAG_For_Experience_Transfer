from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings

from utils.pdf_utils import extract_pdf_text_by_page, chunk_pages


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def ingest_from_json(
    json_path: str,
    url: str,
    username: str,
    password: str,
    *,
    index_name: str = "pdf_chunks",
    embedding_model: str = "qwen3-embedding:0.6b",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    limit: Optional[int] = None,
) -> None:
    """
    Ingest PDFs listed in a JSON file into Neo4j as Document and Chunk nodes, then build a vector index.

    The JSON is expected to be a list of objects containing at least a 'path' and 'title' (or 'Title').
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    graph = Neo4jGraph(url=url, username=username, password=password)

    processed = 0
    for item in items:
        if limit is not None and processed >= limit:
            break

        pdf_path = item.get("path") or item.get("Path")
        if not pdf_path or not os.path.isfile(pdf_path):
            # Skip entries without a valid file path
            continue

        title = item.get("title") or item.get("Title") or os.path.basename(pdf_path)
        file_name = item.get("FileName") or os.path.basename(pdf_path)
        mime_type = item.get("MIMEType") or item.get("mime_type") or "application/pdf"
        page_count = _as_int(item.get("PageCount") or item.get("pages") or 0)

        pages = extract_pdf_text_by_page(pdf_path)
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            # Still register the document even if no text extracted
            query = (
                "MERGE (d:Document {path: $path})\n"
                "SET d.title = $title, d.file_name = $file_name, d.page_count = $page_count, d.mime_type = $mime_type"
            )
            graph.query(query, params={
                "path": pdf_path,
                "title": title,
                "file_name": file_name,
                "page_count": page_count,
                "mime_type": mime_type,
            })
            processed += 1
            continue

        # Upsert document and its chunks in one parameterized query
        upsert = (
            "MERGE (d:Document {path: $path})\n"
            "SET d.title = $title, d.file_name = $file_name, d.page_count = $page_count, d.mime_type = $mime_type\n"
            "WITH d\n"
            "UNWIND $chunks AS chunk\n"
            "MERGE (c:Chunk {doc_path: $path, page: chunk.page, idx: chunk.idx})\n"
            "SET c.content = chunk.text\n"
            "MERGE (d)-[:HAS_CHUNK]->(c)"
        )

        graph.query(
            upsert,
            params={
                "path": pdf_path,
                "title": title,
                "file_name": file_name,
                "page_count": page_count if page_count else len(pages),
                "mime_type": mime_type,
                "chunks": chunks,
            },
        )

        processed += 1

    # Build or refresh a vector index over Chunk nodes using their content
    Neo4jVector.from_existing_graph(
        OllamaEmbeddings(model=embedding_model),
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        node_label="Chunk",
        text_node_properties=["content"],
        embedding_node_property="embedding",
    )

    print(f"Ingestion complete. Processed {processed} documents. Vector index: {index_name}")


__all__ = ["ingest_from_json"]

