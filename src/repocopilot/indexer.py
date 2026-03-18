from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .http_llm import embed

def iter_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in settings.exclude_dirs for part in p.parts):
            continue
        if p.suffix.lower() in settings.include_ext:
            rel = p.as_posix().replace("\\", "/")
            if any(rel.endswith(x.replace("\\", "/")) for x in getattr(settings, "exclude_files", [])):
                continue
            yield p

def chunk_text(text: str) -> List[str]:
    maxc = settings.max_chars_per_chunk
    ov = settings.overlap_chars
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + maxc)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        i = max(0, end - ov)
    return chunks

def get_client():
    Path(settings.index_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.index_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

def get_collection(client):
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

def reset_collection(client) -> None:
    # Elimina e ricrea la collection (utile se reindicizzi)
    try:
        client.delete_collection(name=settings.collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=settings.collection_name, metadata={"hnsw:space": "cosine"})

def index_repo(repo_path: str, reset: bool = False) -> Tuple[int, int]:
    root = Path(repo_path).resolve()
    client = get_client()
    if reset:
        reset_collection(client)
    col = get_collection(client)

    n_files = 0
    n_chunks = 0
    ids, docs, metas = [], [], []

    for f in iter_files(root):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        n_files += 1
        chunks = chunk_text(text)
        for j, c in enumerate(chunks):
            cid = f"{f.as_posix()}::chunk{j}"
            ids.append(cid)
            docs.append(f"FILE: {f.as_posix()}\n\n{c}")
            metas.append({"path": f.as_posix()})
            n_chunks += 1

            if len(docs) >= 64:
                embs = embed(docs)
                col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                ids, docs, metas = [], [], []

    if docs:
        embs = embed(docs)
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    return n_files, n_chunks