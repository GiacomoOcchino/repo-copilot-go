from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

from .config import settings
from .http_llm import embed, chat
from .indexer import get_client, get_collection, iter_files
from .schemas import AnswerWithCitations, Source, Citation

from .deterministic import (
    try_deterministic_entrypoint,
    try_deterministic_cli_commands,
    try_deterministic_default_models,
    try_deterministic_doctor,
    try_deterministic_embeddings_where,
)

# -----------------------
# JSON parsing (robusto)
# -----------------------
def _extract_json(text: str) -> str:
    # fenced ```json { ... } ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # first {...}
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return text

def _repair_json(js: str) -> str:
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    ob = js.count("{") - js.count("}")
    osq = js.count("[") - js.count("]")
    if osq > 0:
        js += "]" * osq
    if ob > 0:
        js += "}" * ob
    return js

def _parse_to_model(raw_text: str) -> AnswerWithCitations:
    # caso patologico: il modello restituisce più oggetti JSON separati (come nel tuo answer.md)
    if raw_text.strip().startswith("{\n  \"ref\"") or raw_text.strip().startswith("{\n \"ref\""):
        return AnswerWithCitations(
            answer_md=raw_text.strip(),
            citations=[],
            confidence="bassa",
            open_questions=["Il modello ha restituito oggetti JSON separati, non un singolo oggetto conforme allo schema."],
        )

    try:
        data = json.loads(raw_text)
    except Exception:
        js = _extract_json(raw_text)
        js = _repair_json(js)
        data = json.loads(js)

    # normalizza answer_md se lista
    if isinstance(data.get("answer_md"), list):
        data["answer_md"] = "\n".join(str(x) for x in data["answer_md"])

    if "citations" not in data or data["citations"] is None:
        data["citations"] = []

    return AnswerWithCitations.model_validate(data)

# -----------------------
# Evidence injection (riga di codice, non docstring)
# -----------------------
def _repo_root() -> Path:
    return Path(".").resolve()

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _extract_lines_around_regex(text: str, rx: re.Pattern, radius: int = 10, max_chars: int = 1200) -> str:
    lines = text.splitlines()
    hit: Optional[int] = None
    for i, ln in enumerate(lines):
        if rx.search(ln):
            hit = i
            break
    if hit is None:
        return ""
    a = max(0, hit - radius)
    b = min(len(lines), hit + radius + 1)
    snippet = "\n".join(lines[a:b])
    if not any(rx.search(ln) for ln in snippet.splitlines()):
        return ""
    return snippet[:max_chars]

def inject_retrieve_evidence(question: str) -> List[Source]:
    ql = question.lower()
    if not any(k in ql for k in ("retrieve", "col.query", "embed", "embedding", "embeddings", "query_embeddings")):
        return []

    root = _repo_root()
    evidences: List[Source] = []

    rx_embed_call = re.compile(r"^\s*q_emb\s*=\s*embed\(", re.IGNORECASE)
    rx_query_call = re.compile(r"^\s*res\s*=\s*col\.query\(", re.IGNORECASE)
    rx_embed_def = re.compile(r"^\s*def\s+embed\s*\(", re.IGNORECASE)

    # find file with both call-sites
    target = None
    target_text = ""
    for f in iter_files(root):
        if f.suffix.lower() != ".py":
            continue
        txt = _read_text(f)
        if rx_embed_call.search(txt) and rx_query_call.search(txt):
            target, target_text = f, txt
            break

    if target:
        embed_call = _extract_lines_around_regex(target_text, rx_embed_call, radius=8, max_chars=900)
        query_call = _extract_lines_around_regex(target_text, rx_query_call, radius=10, max_chars=1100)
        if embed_call:
            evidences.append(Source(ref="", chunk_id="evidence::embed-call", path=str(target), excerpt=embed_call))
        if query_call:
            evidences.append(Source(ref="", chunk_id="evidence::col.query", path=str(target), excerpt=query_call))

    # embed definition
    for f in iter_files(root):
        if f.suffix.lower() != ".py":
            continue
        txt = _read_text(f)
        if not rx_embed_def.search(txt):
            continue
        embed_def = _extract_lines_around_regex(txt, rx_embed_def, radius=18, max_chars=1600)
        if embed_def:
            evidences.append(Source(ref="", chunk_id="evidence::embed-def", path=str(f), excerpt=embed_def))
            break

    return evidences

# -----------------------
# Retrieval (Chroma)
# -----------------------
def retrieve(question: str) -> List[Source]:
    client = get_client()
    col = get_collection(client)

    evidences = inject_retrieve_evidence(question)

    q = question
    ql = question.lower()
    if any(k in ql for k in ("retrieve", "col.query", "embed", "embedding", "embeddings")):
        q = question + "\nKeywords: q_emb = embed( res = col.query( query_embeddings include metadatas distances"

    q_emb = embed([q])[0]
    n = max(settings.top_k, 10)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    sources: List[Source] = []
    for chunk_id, doc, meta in zip(ids, docs, metas):
        sources.append(Source(ref="", chunk_id=chunk_id, path=meta.get("path", "unknown"), excerpt=(doc[:2000] if doc else "")))

    # prepend evidences and dedup
    sources = evidences + sources
    seen = set()
    deduped: List[Source] = []
    for s in sources:
        key = (s.chunk_id, s.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    sources = deduped[:settings.top_k]

    # assign refs
    for i, s in enumerate(sources, start=1):
        s.ref = f"S{i}"
    return sources

# -----------------------
# Citation verification
# -----------------------
def _verify_and_filter_citations(ans: AnswerWithCitations, sources: List[Source]) -> AnswerWithCitations:
    source_map = {s.ref: s.excerpt for s in sources}
    kept: List[Citation] = []
    for c in ans.citations:
        ex = source_map.get(c.ref, "")
        if ex and c.quote and c.quote in ex:
            kept.append(c)
    ans.citations = kept
    if not kept and ans.confidence == "alta":
        ans.confidence = "media"
    return ans

# -----------------------
# Main ASK (Strategy A)
# -----------------------
def answer_with_citations(question: str) -> Tuple[AnswerWithCitations, List[Source], str]:
    # deterministic first
    for resolver in (
        try_deterministic_entrypoint,
        try_deterministic_cli_commands,
        try_deterministic_default_models,
        try_deterministic_doctor,
        try_deterministic_embeddings_where,
    ):
        out = resolver(question)
        if out is not None:
            ans, srcs = out
            return ans, srcs, ""

    sources = retrieve(question)

    allowed_refs = ", ".join(s.ref for s in sources)
    sources_block = "\n".join(
        f"[{s.ref}] path={s.path} (chunk_id={s.chunk_id})\nexcerpt:\n{s.excerpt}\n"
        for s in sources
    )

    system = (
        "Sei RepoCopilot.\n"
        "Regole:\n"
        "- Usa SOLO le fonti fornite.\n"
        "- Rispondi SOLO con JSON valido (niente markdown).\n"
        f"- citations[].ref deve essere uno tra: {allowed_refs}\n"
        "- citations[].quote deve essere copiato ESATTAMENTE dalle fonti.\n"
        "Schema:\n"
        "{\"answer_md\": string, \"citations\": [{\"ref\": string, \"quote\": string}], \"confidence\": \"alta|media|bassa\", \"open_questions\": [string]}"
    )

    user = (
        f"DOMANDA:\n{question}\n\n"
        f"FONTI:\n{sources_block}\n\n"
        "Richiesta:\n"
        "- Spiega in 3-6 bullet.\n"
        "- Aggiungi 1-3 citazioni (copiaincolla righe dagli excerpt).\n"
    )

    raw = chat(system=system, user=user, temperature=0.0, max_tokens=900)
    try:
        model = _parse_to_model(raw)
    except Exception:
        # retry JSON-only
        raw2 = chat(system=system + "\nATTENZIONE: SOLO JSON valido.", user=user, temperature=0.0, max_tokens=900)
        try:
            model = _parse_to_model(raw2)
        except Exception:
            model = AnswerWithCitations(
                answer_md=raw.strip(),
                citations=[],
                confidence="bassa",
                open_questions=["Output non JSON valido anche dopo retry."],
            )
            return model, sources , raw

    model = _verify_and_filter_citations(model, sources)
    return model, sources, raw