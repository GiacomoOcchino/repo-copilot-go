from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any
import json
import hashlib
import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .http_llm import embed
from .chunking import chunk_file  # <--- nuovo


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _state_path() -> Path:
    # dentro index_dir
    return Path(settings.index_dir) / "index_state.json"


def _load_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"files": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def _save_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in settings.exclude_dirs for part in p.parts):
            continue
        if p.suffix.lower() in settings.include_ext:
            yield p


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
    try:
        client.delete_collection(name=settings.collection_name)
    except Exception:
        pass
    client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )


def index_repo(repo_path: str, reset: bool = False) -> Tuple[int, int]:
    root = Path(repo_path).resolve()
    client = get_client()
    if reset:
        reset_collection(client)

    col = get_collection(client)
    state = {"files": {}} if reset else _load_state()
    files_state: Dict[str, Any] = state.get("files", {})

    # per gestire deletions
    seen_paths = set()

    n_files = 0
    n_chunks = 0

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []

    def flush():
        nonlocal batch_ids, batch_docs, batch_metas
        if not batch_docs:
            return
        embs = embed(batch_docs)
        col.add(
            ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs
        )
        batch_ids, batch_docs, batch_metas = [], [], []

    for f in iter_files(root):
        rel = f.relative_to(root).as_posix()  # portabile
        seen_paths.add(rel)

        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        file_hash = _sha256_text(text)
        prev = files_state.get(rel)

        # skip unchanged
        if prev and prev.get("hash") == file_hash:
            continue

        # se era già indicizzato, elimina vecchi chunk
        if prev and prev.get("chunk_ids"):
            try:
                col.delete(ids=prev["chunk_ids"])
            except Exception:
                # se delete non supporta ids in quella versione, ignora (ma di solito sì)
                pass

        n_files += 1

        chunks = chunk_file(
            f, text, maxc=settings.max_chars_per_chunk, overlap=settings.overlap_chars
        )

        chunk_ids = []
        for j, c in enumerate(chunks):
            cid = f"{rel}::chunk{j}"
            chunk_ids.append(cid)
            batch_ids.append(cid)
            batch_docs.append(f"FILE: {rel}\n\n{c}")
            batch_metas.append({"path": rel})
            n_chunks += 1

            if len(batch_docs) >= 64:
                flush()

        files_state[rel] = {"hash": file_hash, "chunk_ids": chunk_ids}

    flush()

    # deletions: file non più presente
    removed = [p for p in list(files_state.keys()) if p not in seen_paths]
    for rel in removed:
        prev = files_state.get(rel)
        if prev and prev.get("chunk_ids"):
            try:
                col.delete(ids=prev["chunk_ids"])
            except Exception:
                pass
        files_state.pop(rel, None)

    state["files"] = files_state
    _save_state(state)

    return n_files, n_chunks
